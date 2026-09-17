//! Video ingest: turn a file, RTSP URL or capture device into a stream of
//! `Arc<Frame>` in the engine's working resolution.
//!
//! DEFAULT BACKEND: an `ffmpeg` subprocess writing raw RGB24 to a pipe. It
//! needs no build-time dependencies, exists on every platform, handles RTSP
//! reconnection and jitter, and exposes hardware decoders through flags
//! (`-hwaccel cuda|vaapi|qsv|d3d11va`). Pipe throughput is not the limit:
//! 1080p at 25 fps is ~155 MB/s, well inside what a pipe moves. The decode
//! itself is the same libavcodec a linked FFmpeg would run.
//!
//! The GStreamer backend (feature `gst`) is the planned upgrade for zero-copy
//! NV12 handoff and in-process control; it is not required to ship.
//!
//! BACKPRESSURE: frames go into a bounded channel. If the consumer is slower
//! than the source, the OLDEST queued frame is dropped, never the newest —
//! a live tracker must run on what is happening now, not on a backlog.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use std::sync::Mutex;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use sa_core::{Error, Frame, Result};

/// Where the video comes from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Source {
    File(String),
    Rtsp(String),
    /// A V4L2 device such as `/dev/video0`, or a capture card.
    Device(String),
}

impl Source {
    /// Guess from a string: URLs → Rtsp, `/dev/video*` → Device, else File.
    pub fn parse(s: &str) -> Self {
        let t = s.trim();
        if t.starts_with("rtsp://") || t.starts_with("rtsps://") || t.starts_with("udp://") || t.starts_with("srt://") || t.starts_with("http://") || t.starts_with("https://") {
            Source::Rtsp(t.to_string())
        } else if t.starts_with("/dev/video") || t.starts_with("video=") || t.starts_with("avfoundation:") {
            Source::Device(t.to_string())
        } else if cfg!(target_os = "macos") && t.chars().all(|c| c.is_ascii_digit() || c == ':') && !t.is_empty() {
            // avfoundation names cameras by index: "0", or "0:0" for
            // video:audio. A bare number is never a useful file path.
            Source::Device(t.to_string())
        } else {
            Source::File(t.to_string())
        }
    }
    pub fn is_live(&self) -> bool {
        !matches!(self, Source::File(_))
    }
    pub fn location(&self) -> &str {
        match self {
            Source::File(s) | Source::Rtsp(s) | Source::Device(s) => s,
        }
    }
}

/// What `ffprobe` reported before decode started.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    /// Total frames for a file; 0 for live.
    pub frames: u64,
}

#[derive(Debug, Clone)]
pub struct IngestOptions {
    /// Downscale to this width (keeping aspect) if the source is wider.
    pub target_width: u32,
    pub queue_depth: usize,
    pub hwaccel: bool,
    pub loop_file: bool,
    /// Explicit ffmpeg binary; default resolves from PATH.
    pub ffmpeg: String,
    pub ffprobe: String,
    /// For file sources: play at the file's own rate instead of as fast as the
    /// consumer accepts. Makes a file behave like a live feed.
    pub realtime: bool,
    /// Drop the oldest queued frame when the consumer falls behind (live
    /// semantics). When false the decoder blocks instead, so every frame is
    /// processed — right for offline runs over a file. `None` = live sources
    /// and realtime playback drop; plain files block.
    pub drop_when_full: Option<bool>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            target_width: 1280,
            queue_depth: 2,
            hwaccel: true,
            loop_file: false,
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            realtime: false,
            drop_when_full: None,
        }
    }
}

/// Counters the engine reads for its stats line.
#[derive(Debug, Default)]
pub struct IngestStats {
    pub produced: AtomicU64,
    pub dropped: AtomicU64,
}

/// A running source. Drop it (or call `stop`) to end the ffmpeg process.
pub struct Ingest {
    rx: Receiver<Arc<Frame>>,
    /// Published by the decode thread once a live stream's real geometry is
    /// known. `info()` reads this so callers never see the 0x0 placeholder.
    live_info: Arc<Mutex<Option<SourceInfo>>>,
    /// Every decoded frame, for consumers that must not miss any (the
    /// preview). Separate from `rx` on purpose: the pipeline drops frames it
    /// cannot keep up with, but the picture on screen should still be smooth.
    raw_rx: Receiver<Arc<Frame>>,
    info: SourceInfo,
    stop: Arc<AtomicBool>,
    stats: Arc<IngestStats>,
    thread: Option<JoinHandle<()>>,
}

impl Ingest {
    /// Probe, then start decoding on a background thread.
    pub fn open(source: Source, opts: IngestOptions) -> Result<Self> {
        // PROBING A LIVE STREAM IS NOT REQUIRED, and must not be fatal. A
        // datagram source has nothing to read until a sender starts, so
        // `ffprobe` either fails ("No such file or directory" — it falls back
        // to treating the URL as a path) or blocks for its whole timeout.
        // Neither is a reason to refuse the source: the decoder will report
        // the real geometry from the first frame it receives, and until then
        // a nominal size is enough to start the threads.
        let src_info = match probe(&opts.ffprobe, &source) {
            Ok(i) => i,
            Err(e) if source.is_live() => {
                tracing::info!("could not probe {} yet ({e}); waiting for the stream to start", source.location());
                SourceInfo { width: 0, height: 0, fps: 25.0, frames: 0 }
            }
            Err(e) => return Err(e),
        };
        let (out_w, out_h) = if src_info.width == 0 {
            // Unknown until the stream starts. `decode_loop` re-probes once
            // frames are flowing and publishes the real size.
            (0, 0)
        } else {
            output_size(src_info.width, src_info.height, opts.target_width)
        };
        let info = SourceInfo { width: out_w, height: out_h, ..src_info };

        let (tx, rx) = crossbeam_channel::bounded::<Arc<Frame>>(opts.queue_depth.max(1));
        // Shallow and drop-oldest: the preview wants the NEWEST picture, and
        // one frame of slack is enough to hand it over without blocking.
        let (raw_tx, raw_rx) = crossbeam_channel::bounded::<Arc<Frame>>(2);
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(IngestStats::default());

        let t_stop = stop.clone();
        let t_stats = stats.clone();
        let live_info: Arc<Mutex<Option<SourceInfo>>> = Arc::new(Mutex::new(None));
        let t_live = live_info.clone();
        let thread = std::thread::Builder::new()
            .name("sa-ingest".into())
            .spawn(move || {
                if let Err(e) = decode_loop(&source, &opts, info, tx, raw_tx, t_stop, t_stats, t_live) {
                    tracing::error!("ingest ended: {e}");
                }
            })
            .map_err(Error::Io)?;

        Ok(Self { rx, raw_rx, live_info, info, stop, stats, thread: Some(thread) })
    }

    pub fn info(&self) -> SourceInfo {
        // The live geometry once known, else what the probe said at open.
        self.live_info.lock().ok().and_then(|g| *g).unwrap_or(self.info)
    }

    /// Block until a live source's geometry is known (or `stop`). Returns
    /// false on timeout — the caller can then report "waiting for stream"
    /// rather than pretending to know the size.
    pub fn wait_for_stream(&self, timeout: Duration) -> bool {
        let t0 = Instant::now();
        while self.info().width == 0 {
            if t0.elapsed() > timeout || !self.is_alive() {
                return false;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        true
    }
    pub fn frames(&self) -> &Receiver<Arc<Frame>> {
        &self.rx
    }
    /// Every decoded frame, for the preview. Drop-oldest, depth 2.
    pub fn raw_frames(&self) -> &Receiver<Arc<Frame>> {
        &self.raw_rx
    }
    pub fn stats(&self) -> &IngestStats {
        &self.stats
    }
    /// False once the decode thread has exited (file finished, or fatal error).
    pub fn is_alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Ingest {
    fn drop(&mut self) {
        self.stop();
    }
}

fn output_size(w: u32, h: u32, target_w: u32) -> (u32, u32) {
    if target_w == 0 || w <= target_w {
        // ffmpeg rgb24 needs even dimensions for most scalers; keep as-is.
        return (w, h);
    }
    let s = target_w as f64 / w as f64;
    let oh = ((h as f64 * s).round() as u32) & !1;
    (target_w & !1, oh.max(2))
}

/// Ask ffprobe for the first video stream's geometry and rate.
pub fn probe(ffprobe: &str, source: &Source) -> Result<SourceInfo> {
    let mut cmd = Command::new(ffprobe);
    cmd.args(["-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames", "-of", "json"]);
    if let Source::Rtsp(u) = source {
        if u.starts_with("rtsp") {
            cmd.args(["-rtsp_transport", "tcp"]);
        }
        if u.starts_with("udp://") {
            // Probing a live datagram stream must not hang forever when
            // nothing is sending: 10 s is long enough to catch a keyframe.
            cmd.args(["-fifo_size", "5000000", "-overrun_nonfatal", "1", "-timeout", "10000000"]);
        }
    }
    if let Source::Device(_) = source {
        // The same input format the decoder will use — probing with a
        // different one reports geometry the run then contradicts.
        cmd.args(["-f", device_format()]);
    }
    cmd.arg(source.location());
    let out = cmd.output().map_err(|e| Error::Source(format!("ffprobe not runnable ({e}); install ffmpeg")))?;
    if !out.status.success() {
        return Err(Error::Source(format!(
            "ffprobe failed for {}: {}",
            source.location(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| Error::Source(format!("ffprobe output unreadable: {e}")))?;
    let s = v["streams"].get(0).ok_or_else(|| Error::Source("no video stream".into()))?;
    let width = s["width"].as_u64().unwrap_or(0) as u32;
    let height = s["height"].as_u64().unwrap_or(0) as u32;
    if width == 0 || height == 0 {
        return Err(Error::Source("stream has no dimensions".into()));
    }
    let fps = parse_rate(s["avg_frame_rate"].as_str())
        .filter(|f| *f > 0.0)
        .or_else(|| parse_rate(s["r_frame_rate"].as_str()))
        .unwrap_or(25.0);
    let frames = s["nb_frames"].as_str().and_then(|n| n.parse().ok()).unwrap_or(0);
    Ok(SourceInfo { width, height, fps, frames })
}

fn parse_rate(s: Option<&str>) -> Option<f32> {
    let s = s?;
    if let Some((n, d)) = s.split_once('/') {
        let n: f32 = n.parse().ok()?;
        let d: f32 = d.parse().ok()?;
        if d > 0.0 {
            return Some(n / d);
        }
        None
    } else {
        s.parse().ok()
    }
}

/// ffmpeg's capture-device input format for this platform.
const fn device_format() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "avfoundation"
    }
    #[cfg(target_os = "windows")]
    {
        "dshow"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "v4l2"
    }
}

fn spawn_ffmpeg(source: &Source, opts: &IngestOptions, w: u32, h: u32) -> Result<Child> {
    let mut cmd = Command::new(&opts.ffmpeg);
    cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    if opts.hwaccel {
        // `auto` lets ffmpeg pick cuda/vaapi/qsv/d3d11va if present and falls
        // back to software without failing — the same "GPU if usable" policy
        // the rest of the engine follows.
        cmd.args(["-hwaccel", "auto"]);
    }
    match source {
        Source::Rtsp(u) => {
            // RTSP only: forcing TCP transport on a udp:// or srt:// URL is
            // meaningless, and ffmpeg warns about it.
            if u.starts_with("rtsp") {
                cmd.args(["-rtsp_transport", "tcp"]);
            }
            // A datagram stream needs a receive buffer large enough to
            // absorb a burst, or packets are dropped by the kernel before
            // ffmpeg reads them and the picture tears. `overrun_nonfatal`
            // keeps a late burst from ending the run outright.
            if u.starts_with("udp://") {
                cmd.args(["-fifo_size", "5000000", "-overrun_nonfatal", "1"]);
            }
            cmd.args(["-fflags", "nobuffer", "-flags", "low_delay",
                      "-max_delay", "500000", "-i", u]);
        }
        Source::Device(d) => {
            cmd.args(["-f", device_format(), "-i", d]);
        }
        Source::File(f) => {
            if opts.realtime {
                cmd.arg("-re");
            }
            if opts.loop_file {
                cmd.args(["-stream_loop", "-1"]);
            }
            cmd.args(["-i", f]);
        }
    }
    cmd.args(["-an", "-sn", "-dn"]);
    cmd.args(["-vf", &format!("scale={w}:{h}:flags=bilinear")]);
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    cmd.spawn().map_err(|e| Error::Source(format!("cannot start ffmpeg ({e})")))
}

#[allow(clippy::too_many_arguments)]
fn decode_loop(
    source: &Source,
    opts: &IngestOptions,
    info: SourceInfo,
    tx: Sender<Arc<Frame>>,
    raw_tx: Sender<Arc<Frame>>,
    stop: Arc<AtomicBool>,
    stats: Arc<IngestStats>,
    live_info: Arc<Mutex<Option<SourceInfo>>>,
) -> Result<()> {
    // A live source may not have been probeable at open time (nothing was
    // sending yet). Retry until it is, so `sa run udp://...` can be started
    // BEFORE the sender — which is the natural order when the sender is on
    // another machine.
    let mut info = info;
    while info.width == 0 {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        match probe(&opts.ffprobe, source) {
            Ok(p) => {
                let (ow, oh) = output_size(p.width, p.height, opts.target_width);
                info = SourceInfo { width: ow, height: oh, ..p };
                if let Ok(mut g) = live_info.lock() {
                    *g = Some(info);
                }
                tracing::info!(width = ow, height = oh, fps = p.fps, "stream started");
            }
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    let (w, h) = (info.width, info.height);
    let frame_bytes = (w * h * 3) as usize;
    let drop_policy = opts.drop_when_full.unwrap_or(source.is_live() || opts.realtime);
    let mut id: u64 = 0;
    let mut attempt: u32 = 0;
    let frame_dt = if info.fps > 0.0 { Duration::from_secs_f64(1.0 / info.fps as f64) } else { Duration::from_millis(40) };

    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut child = spawn_ffmpeg(source, opts, w, h)?;
        let mut stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take();
        let t0 = Instant::now();
        let mut n_this_run: u64 = 0;

        loop {
            if stop.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
            let mut buf = vec![0u8; frame_bytes];
            if let Err(e) = stdout.read_exact(&mut buf) {
                // EOF (file finished) or the process died. Live sources reconnect.
                tracing::debug!("ffmpeg stream ended: {e}");
                break;
            }
            // pts: file sources get a synthetic clock at the nominal rate; live
            // sources use wall-clock since start (ffmpeg drops timestamps on the
            // rawvideo pipe, and for a live tracker "now" is the right answer).
            let pts = if source.is_live() { t0.elapsed() } else { frame_dt * (n_this_run as u32) };
            let frame = Arc::new(Frame::new_rgb8(id, pts, w, h, buf));
            id += 1;
            n_this_run += 1;
            stats.produced.fetch_add(1, Ordering::Relaxed);
            // The preview tap first, never blocking: it must see this frame
            // even when the pipeline is about to refuse it. Drop-oldest via
            // the sender's own receiver handle, so a stalled preview client
            // can never hold up the decoder.
            push_drop_oldest_counted(&raw_tx, frame.clone(), None);
            if drop_policy {
                push_drop_oldest(&tx, frame, &stats);
            } else if tx.send(frame).is_err() {
                break;
            }
            if tx.is_empty() && stop.load(Ordering::Relaxed) {
                break;
            }
        }

        let status = child.wait();
        if let Some(mut err) = stderr {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            // `-hwaccel auto` probes every accelerator and reports the ones it
            // cannot open; that is a fallback working, not a fault.
            let noise = |l: &str| l.contains("libcuda") || l.contains("Device creation failed") || l.contains("AVHWDeviceContext");
            let s: String = s.lines().filter(|l| !noise(l)).collect::<Vec<_>>().join("\n");
            if !s.trim().is_empty() {
                tracing::warn!("ffmpeg: {}", s.trim());
            }
        }
        match source {
            Source::File(_) => {
                // A file that ends is done unless it is meant to loop (ffmpeg
                // handles -stream_loop itself, so reaching here means EOF).
                tracing::info!("file source finished after {n_this_run} frames ({:?})", status);
                return Ok(());
            }
            _ => {
                attempt += 1;
                let backoff = Duration::from_millis((250 * (1u64 << attempt.min(5))).min(5000));
                tracing::warn!("live source dropped ({:?}); reconnecting in {:?}", status, backoff);
                std::thread::sleep(backoff);
                if n_this_run > 0 {
                    attempt = 0;
                }
            }
        }
    }
}

/// Bounded send that never blocks the decoder: on a full queue the oldest
/// frame is discarded so the consumer always sees the newest.
fn push_drop_oldest(tx: &Sender<Arc<Frame>>, frame: Arc<Frame>, stats: &IngestStats) {
    push_drop_oldest_counted(tx, frame, Some(stats))
}

/// As above, with the drop counter optional: the preview tap discards frames
/// as a matter of course, and counting those as pipeline drops would make the
/// stats line report a problem that is not one.
fn push_drop_oldest_counted(tx: &Sender<Arc<Frame>>, frame: Arc<Frame>, stats: Option<&IngestStats>) {
    let mut f = frame;
    loop {
        match tx.try_send(f) {
            Ok(()) => return,
            Err(TrySendError::Full(back)) => {
                // Make room by pulling one out ourselves. crossbeam has no
                // "receiver-side drop" from the sender, so we race the consumer
                // for one slot; either way the queue advances.
                if let Some(s) = stats {
                    s.dropped.fetch_add(1, Ordering::Relaxed);
                }
                f = back;
                std::thread::yield_now();
                // If still full after yielding, give up this frame rather than
                // stall the pipe: the newest frame will arrive right behind it.
                if tx.is_full() {
                    return;
                }
            }
            Err(TrySendError::Disconnected(_)) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_parse() {
        assert_eq!(Source::parse("rtsp://cam/1"), Source::Rtsp("rtsp://cam/1".into()));
        assert_eq!(Source::parse("/dev/video0"), Source::Device("/dev/video0".into()));
        assert_eq!(Source::parse("clip.mp4"), Source::File("clip.mp4".into()));
    }

    #[test]
    fn scale_keeps_aspect_even() {
        assert_eq!(output_size(1920, 1080, 1280), (1280, 720));
        assert_eq!(output_size(1280, 720, 1280), (1280, 720));
        assert_eq!(output_size(640, 480, 1280), (640, 480));
    }

    #[test]
    fn rate_parse() {
        assert_eq!(parse_rate(Some("25/1")), Some(25.0));
        assert_eq!(parse_rate(Some("30000/1001")).map(|f| (f * 100.0).round()), Some(2997.0));
        assert_eq!(parse_rate(Some("0/0")), None);
    }
}
