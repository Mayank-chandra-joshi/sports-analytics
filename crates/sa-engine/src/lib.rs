//! The engine: one `Engine::start` runs the whole pipeline on background
//! threads and hands back a stream of `FrameState`. Nothing here knows
//! about Tauri or the CLI.
//!
//! Threads:
//!   ingest    — ffmpeg → Arc<Frame>                      (sa-ingest)
//!   pipeline  — detect → track → teams → pitch → analytics → FrameState
//!   keypoints — pitch keypoint model every K frames → Calibrator::solve
//!   render    — overlays + JPEG for the preview, recorder, session log
//!
//! Every channel between them is bounded and drop-oldest; the pipeline
//! always works on the newest frame the decoder produced.

pub mod mjpeg;
pub mod recorder;
pub mod scenecut;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use sa_analytics::{BallParams, BallTracker, LiveAnalytics};
use sa_core::profile::{ClassMap, FieldModel};
use sa_core::{BallState, Class, Config, Error, Frame, FrameState, LiveStats, Result, StageLatency, Team, Track};
use sa_identity::{kit_reading, on_playing_surface, surface_colour, LiveTeamClassifier, KIT_MIN_CONTRAST};
use sa_infer::{Detector, Embedder, MockDetector, OsnetEmbedder, PitchKeypoints, YoloDetector, YoloOptions, YoloPoseKeypoints};
use sa_ingest::{Ingest, IngestOptions, Source, SourceInfo};
use sa_pitch::{foot_point, Calibrator, CalibratorParams, Football, FootballDims};
use sa_track::{Input, Tracker};

pub use mjpeg::MjpegServer;
pub use recorder::{Recorder, SessionLog};
pub use sa_ingest::Source as VideoSource;

/// The furthest apart two detections may be before association stops being
/// meaningful. At 25 fps this is 160 ms — about one stride of a running
/// player, which the motion model bridges. Past it the tracker is
/// extrapolating, and a ring lands where nobody is.
const MAX_DETECT_STRIDE: u64 = 4;

/// A tracked person must be at least this fraction of the frame height.
/// Below it there is nothing to associate on and nothing to read a kit
/// colour from.
const MIN_PERSON_FRAC: f32 = 0.04;
/// ...and at most this much. Anything larger is a close-up or an
/// interview, where "tracking" means following one face around and the
/// pitch-space metrics are meaningless anyway.
const MAX_PERSON_FRAC: f32 = 0.75;

/// Detections fewer than this and the median says nothing about "a typical
/// player", so the relative test is skipped and only the absolute one above
/// applies.
const MIN_PEOPLE_FOR_SCALE: usize = 4;
/// How far a person's height may sit from the frame's median before they are
/// judged to be on a different plane from the players. Generous on both
/// sides: a goalkeeper far upfield is genuinely half the size of a near
/// player, and rejecting a real player is worse than admitting a spectator.
const PERSON_SCALE_LO: f32 = 0.45;
const PERSON_SCALE_HI: f32 = 2.2;

/// A snapshot of the tracker that the preview can advance on its own,
/// without touching the pipeline's live state.
type Predictor = Box<dyn Fn(u64) -> Vec<Track> + Send>;
type SharedPredictor = Arc<Mutex<Option<Predictor>>>;

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub config: Config,
    pub source: Source,
    /// Directory the model paths in `config` are relative to.
    pub root: PathBuf,
    /// Serve an MJPEG preview on loopback.
    pub preview: bool,
    /// Play a file at its own rate (behave like a live feed).
    pub realtime: bool,
    /// Draw overlays into the preview/recording (the desktop UI draws its own).
    pub burn_overlays: bool,
}

/// What `Engine::stop` returns.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Summary {
    pub frames_processed: u64,
    pub frames_dropped: u64,
    pub avg_fps: f32,
    pub analytics: serde_json::Value,
    pub recording: Option<PathBuf>,
    pub session_log: Option<PathBuf>,
}

pub struct Engine {
    stop: Arc<AtomicBool>,
    states: Receiver<Arc<FrameState>>,
    latest: Arc<Mutex<Option<Arc<FrameState>>>>,
    calibrator: Arc<Mutex<Calibrator>>,
    mjpeg: Option<MjpegServer>,
    info: SourceInfo,
    pipeline: Option<std::thread::JoinHandle<Result<Summary>>>,
    processed: Arc<AtomicU64>,
}

impl Engine {
    pub fn start(opts: EngineOptions) -> Result<Self> {
        sa_infer::init()?;
        let cfg = opts.config.clone();

        // -- ingest --------------------------------------------------------
        let ingest = Ingest::open(
            opts.source.clone(),
            IngestOptions {
                target_width: cfg.video.target_width,
                queue_depth: cfg.video.queue_depth,
                hwaccel: cfg.video.hwaccel,
                loop_file: cfg.video.loop_file,
                realtime: opts.realtime,
                ..Default::default()
            },
        )?;
        // A live source may have nothing sending yet — the decoder learns the
        // geometry from the first frames. Wait briefly so `info()` is real
        // before the preview and the UI size themselves from it; if nothing
        // arrives, carry on anyway rather than refusing the source, because
        // the sender may simply start later.
        if opts.source.is_live() && ingest.info().width == 0 {
            tracing::info!("waiting for the stream to start…");
            if !ingest.wait_for_stream(std::time::Duration::from_secs(30)) {
                tracing::warn!("no data yet on {} — the engine will keep waiting", opts.source.location());
            }
        }
        let info = ingest.info();
        tracing::info!(?info, source = ?opts.source, "source open");

        // -- models --------------------------------------------------------
        let classes = match cfg.detection.class_map.as_str() {
            "roboflow_football" => ClassMap::roboflow_football(),
            _ => ClassMap::coco(),
        };
        let det_path = opts.root.join(&cfg.detection.model);
        let detector: Box<dyn Detector> = if det_path.is_file() {
            Box::new(YoloDetector::load(
                &det_path,
                YoloOptions { input_size: cfg.detection.input_size, conf: cfg.detection.conf, iou_nms: cfg.detection.iou_nms, threads: cfg.detection.threads, classes },
            )?)
        } else {
            tracing::warn!(path = %det_path.display(), "detector model not found — running with a mock detector");
            Box::new(MockDetector)
        };
        let embedder: Option<Box<dyn Embedder>> = match (&cfg.reid.enabled, &cfg.reid.model) {
            (true, Some(p)) if opts.root.join(p).is_file() => Some(Box::new(OsnetEmbedder::load(&opts.root.join(p), cfg.detection.threads)?)),
            (true, _) => {
                tracing::warn!("reid enabled but no model file — disabled");
                None
            }
            _ => None,
        };
        let keypoints: Option<Box<dyn PitchKeypoints>> = match &cfg.pitch.keypoint_model {
            Some(p) if cfg.pitch.enabled && opts.root.join(p).is_file() => {
                Some(Box::new(YoloPoseKeypoints::load(&opts.root.join(p), cfg.pitch.keypoint_input, cfg.pitch.keypoint_conf, cfg.detection.threads)?))
            }
            Some(p) => {
                tracing::warn!(path = %opts.root.join(p).display(), "pitch keypoint model not found — manual calibration only");
                None
            }
            None => None,
        };

        // -- shared state --------------------------------------------------
        let field: Box<dyn FieldModel> = Box::new(Football::new(FootballDims::with_size(cfg.pitch.length_m, cfg.pitch.width_m)));
        let calibrator = Arc::new(Mutex::new(Calibrator::new(
            field,
            CalibratorParams { ransac_px: cfg.pitch.ransac_px, min_inliers: cfg.pitch.min_inliers, max_gap: cfg.pitch.max_gap, min_keypoints: 6 },
        )));
        let stop = Arc::new(AtomicBool::new(false));
        let (state_tx, state_rx) = crossbeam_channel::bounded::<Arc<FrameState>>(4);
        let state_rx_drain = state_rx.clone();
        let latest = Arc::new(Mutex::new(None));
        let processed = Arc::new(AtomicU64::new(0));
        let mjpeg = if opts.preview { Some(MjpegServer::start(0).map_err(Error::Io)?) } else { None };

        // -- keypoint side thread ------------------------------------------
        let (kp_tx, kp_rx) = crossbeam_channel::bounded::<Arc<Frame>>(1);
        if let Some(mut kp) = keypoints {
            let cal = calibrator.clone();
            let stop2 = stop.clone();
            std::thread::Builder::new().name("sa-keypoints".into()).spawn(move || {
                while !stop2.load(Ordering::Relaxed) {
                    let Ok(frame) = kp_rx.recv_timeout(std::time::Duration::from_millis(200)) else { continue };
                    let t0 = Instant::now();
                    match kp.keypoints(&frame) {
                        Ok(pts) => {
                            let rms = cal.lock().solve(frame.id, &pts);
                            tracing::debug!(frame = frame.id, n = pts.len(), ?rms, ms = t0.elapsed().as_millis(), "keypoints");
                        }
                        Err(e) => tracing::warn!("keypoints failed: {e}"),
                    }
                }
            })?;
        }

        // -- render thread -------------------------------------------------
        // THE PREVIEW IS NOT THE PIPELINE. Detection runs at whatever the
        // hardware allows; the picture should still move at the source rate,
        // or the app looks broken when it is merely slow. So the preview is
        // fed from ingest's lossless tap and carries the newest overlay
        // state, motion-predicted to the frame being shown.
        let (render_tx, render_rx) = crossbeam_channel::bounded::<(Arc<Frame>, Arc<FrameState>)>(2);
        // The newest overlay state, published by the pipeline and read by the
        // preview for frames the detector never processed.
        let overlay: Arc<Mutex<Option<Arc<FrameState>>>> = Arc::new(Mutex::new(None));
        // The tracker's own prediction function, shared read-only so the
        // preview can advance rings to the frame it is about to show.
        let predictor: SharedPredictor = Arc::new(Mutex::new(None));

        // -- preview thread: every decoded frame, overlays predicted forward.
        if mjpeg.is_some() {
            let stop_p = stop.clone();
            let mj = mjpeg.clone();
            let raw = ingest.raw_frames().clone();
            let oc = cfg.output.clone();
            let ov = overlay.clone();
            let pred = predictor.clone();
            let burn_preview = opts.burn_overlays;
            let (fl, fw) = (cfg.pitch.length_m, cfg.pitch.width_m);
            std::thread::Builder::new().name("sa-preview".into()).spawn(move || {
                let mut buf: Vec<u8> = Vec::new();
                while !stop_p.load(Ordering::Relaxed) {
                    let Ok(frame) = raw.recv_timeout(std::time::Duration::from_millis(200)) else { continue };
                    let Some(m) = &mj else { break };
                    if !m.wanted() {
                        continue;
                    }
                    buf.clear();
                    buf.extend_from_slice(&frame.data);
                    // Only when the caller wants them burned in. The desktop
                    // app draws its own overlays on a canvas over the video,
                    // so burning a second set here puts TWO rings and two
                    // labels on every player — and they disagree, because one
                    // is in frame pixels and the other in display pixels.
                    if let (true, Some(st)) = (burn_preview, ov.lock().clone()) {
                        // Advance the rings to THIS frame. Without it they sit
                        // on the last processed position and visibly trail the
                        // players whenever detection is slower than the source.
                        let mut shown = (*st).clone();
                        shown.frame_id = frame.id;
                        if let Some(f) = pred.lock().as_ref() {
                            let moved = f(frame.id);
                            if !moved.is_empty() {
                                // Keep the team/pitch fields the pipeline
                                // resolved; take only the predicted geometry.
                                let by_id: HashMap<u32, &Track> = st.tracks.iter().map(|t| (t.id, t)).collect();
                                shown.tracks = moved
                                    .into_iter()
                                    .map(|mut t| {
                                        if let Some(src) = by_id.get(&t.id) {
                                            t.team = src.team;
                                            t.pitch = src.pitch;
                                        }
                                        t
                                    })
                                    .collect();
                            }
                        }
                        let mut c = sa_render::Canvas::new(frame.width, frame.height, &mut buf);
                        sa_render::draw_overlays(&mut c, &shown);
                        let (pw, ph, pd) = sa_render::render_pad(&shown, fl, fw, (frame.width / 4).max(200));
                        sa_render::composite_pad(&mut c, (pw, ph, &pd), 0.85, 10);
                    }
                    // Publish WITH the frame id. The picture and the overlay
                    // travel by different routes — a socket into an <img> the
                    // browser decodes on its own schedule, and an IPC channel
                    // into a canvas — so nothing makes them agree unless the
                    // frame says which frame it is. The UI matches the state
                    // to this id and the rings sit on the players instead of
                    // running ahead of them.
                    m.publish_with_id(frame.id, sa_render::encode_jpeg(frame.width, frame.height, &buf, oc.preview_width, oc.preview_quality));
                }
            })?;
        }

        // -- render thread: the RECORDER only (one entry per processed frame,
        //    so the output video is the analysis, not the preview).
        {
            let stop3 = stop.clone();
            let oc = cfg.output.clone();
            let burn = opts.burn_overlays;
            let (fl, fw) = (cfg.pitch.length_m, cfg.pitch.width_m);
            let record_path = if oc.record { Some(oc.record_path.clone().unwrap_or_else(|| opts.root.join("outputs/session.mp4"))) } else { None };
            let fps = info.fps;
            let ffmpeg = "ffmpeg".to_string();
            let root = opts.root.clone();
            std::thread::Builder::new().name("sa-render".into()).spawn(move || {
                let mut recorder: Option<Recorder> = None;
                let mut buf: Vec<u8> = Vec::new();
                while !stop3.load(Ordering::Relaxed) {
                    let Ok((frame, st)) = render_rx.recv_timeout(std::time::Duration::from_millis(200)) else { continue };
                    let Some(p) = &record_path else { continue };
                    buf.clear();
                    buf.extend_from_slice(&frame.data);
                    if burn {
                        let mut c = sa_render::Canvas::new(frame.width, frame.height, &mut buf);
                        sa_render::draw_overlays(&mut c, &st);
                        let (pw, ph, pd) = sa_render::render_pad(&st, fl, fw, (frame.width / 4).max(200));
                        sa_render::composite_pad(&mut c, (pw, ph, &pd), 0.85, 10);
                    }
                    if recorder.is_none() {
                        let p = if p.is_absolute() { p.clone() } else { root.join(p) };
                        match Recorder::start(&p, frame.width, frame.height, fps, &ffmpeg) {
                            Ok(r) => recorder = Some(r),
                            Err(e) => tracing::error!("recorder: {e}"),
                        }
                    }
                    if let Some(r) = recorder.as_mut() {
                        if let Err(e) = r.write(&buf) {
                            tracing::error!("recorder write: {e}");
                        }
                    }
                }
                if let Some(r) = recorder.take() {
                    if let Ok((p, n)) = r.finish() {
                        tracing::info!(path = %p.display(), frames = n, "recording closed");
                    }
                }
            })?;
        }

        // -- pipeline thread -----------------------------------------------
        let pipeline = {
            let stop4 = stop.clone();
            let cal = calibrator.clone();
            let latest2 = latest.clone();
            let processed2 = processed.clone();
            let cfg2 = cfg.clone();
            let root = opts.root.clone();
            let overlay2 = overlay.clone();
            let predictor2 = predictor.clone();
            let live = opts.source.is_live() || opts.realtime;
            std::thread::Builder::new().name("sa-pipeline".into()).spawn(move || {
                run_pipeline(cfg2, root, ingest, live, detector, embedder, cal, kp_tx, render_tx, (state_tx, state_rx_drain), latest2, overlay2, predictor2, processed2, stop4)
            })?
        };

        Ok(Self { stop, states: state_rx, latest, calibrator, mjpeg, info, pipeline: Some(pipeline), processed })
    }

    pub fn info(&self) -> SourceInfo {
        self.info
    }
    pub fn states(&self) -> &Receiver<Arc<FrameState>> {
        &self.states
    }
    pub fn latest(&self) -> Option<Arc<FrameState>> {
        self.latest.lock().clone()
    }
    pub fn preview_url(&self) -> Option<String> {
        self.mjpeg.as_ref().map(|m| m.url())
    }
    pub fn processed(&self) -> u64 {
        self.processed.load(Ordering::Relaxed)
    }
    pub fn is_running(&self) -> bool {
        self.pipeline.as_ref().is_some_and(|p| !p.is_finished())
    }

    /// User-placed homography (image px → field m), or None to clear.
    pub fn set_manual_calibration(&self, h: Option<[[f64; 3]; 3]>) {
        self.calibrator.lock().set_manual(h.map(|a| sa_pitch::from_array(&a)));
    }

    pub fn stop(mut self) -> Result<Summary> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(m) = &self.mjpeg {
            m.stop();
        }
        match self.pipeline.take() {
            Some(h) => h.join().map_err(|_| Error::Other("pipeline thread panicked".into()))?,
            None => Err(Error::Other("already stopped".into())),
        }
    }

    /// Block until the source ends (file) or `stop` is called elsewhere.
    pub fn wait(mut self) -> Result<Summary> {
        match self.pipeline.take() {
            Some(h) => {
                let r = h.join().map_err(|_| Error::Other("pipeline thread panicked".into()))?;
                if let Some(m) = &self.mjpeg {
                    m.stop();
                }
                self.stop.store(true, Ordering::Relaxed);
                r
            }
            None => Err(Error::Other("already stopped".into())),
        }
    }
}

/// Bounded send that discards the OLDEST queued item when full. The
/// consumer (UI) may be slower than the pipeline; it must see the newest
/// state, not a backlog.
fn push_drop_oldest<T>(tx: &Sender<T>, rx: &Receiver<T>, v: T) {
    match tx.try_send(v) {
        Ok(()) => {}
        Err(TrySendError::Full(v)) => {
            let _ = rx.try_recv();
            let _ = tx.try_send(v);
        }
        Err(TrySendError::Disconnected(_)) => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pipeline(
    cfg: Config,
    root: PathBuf,
    mut ingest: Ingest,
    // True for a live source or a file played in real time — the cases where
    // ingest drops frames rather than blocking the decoder.
    source_is_live: bool,
    mut detector: Box<dyn Detector>,
    mut embedder: Option<Box<dyn Embedder>>,
    calibrator: Arc<Mutex<Calibrator>>,
    kp_tx: Sender<Arc<Frame>>,
    render_tx: Sender<(Arc<Frame>, Arc<FrameState>)>,
    (state_tx, state_rx): (Sender<Arc<FrameState>>, Receiver<Arc<FrameState>>),
    latest: Arc<Mutex<Option<Arc<FrameState>>>>,
    overlay: Arc<Mutex<Option<Arc<FrameState>>>>,
    predictor: SharedPredictor,
    processed: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
) -> Result<Summary> {
    let info = ingest.info();
    let mut tracker = Tracker::new(cfg.tracker.clone());
    let mut teams = LiveTeamClassifier::new(60, 2);
    let mut ball = BallTracker::new(BallParams::default());
    let mut scene = scenecut::SceneCut::new(cfg.video.scene_cut_corr);
    let dims = calibrator.lock().field().dims();
    let mut analytics = LiveAnalytics::new(cfg.analytics.clone(), dims, info.fps);
    let mut session = match &cfg.output.session_log {
        Some(p) => Some(SessionLog::create(&if p.is_absolute() { p.clone() } else { root.join(p) })?),
        None => None,
    };
    let margin = cfg.pitch.on_pitch_margin_m;
    // 0 = ADAPT: keep detection roughly in step with the source by measuring
    // how long it actually takes on this machine. A fixed number cannot be
    // right for both a laptop CPU and a workstation GPU, and the failure mode
    // of guessing too low is the one the user sees — frames dropped at
    // arbitrary moments, and rings that trail the players.
    let adaptive = cfg.video.detect_every == 0;
    let mut n_detects: u64 = 0;
    let mut skipped_stale: u64 = 0;
    let surface_tol = cfg.detection.surface_tolerance;
    // Live sources (and a file played in real time) drop frames by design;
    // an offline file is processed in full.
    let drop_stale = source_is_live;
    let mut detect_every = if adaptive { 1 } else { cfg.video.detect_every as u64 };
    let mut det_ms_ema = 0.0f32;
    let reembed_every = cfg.reid.reembed_interval.max(1) as u64;

    let t_start = Instant::now();
    let mut fps_ema = 0.0f32;
    let mut last_t = Instant::now();
    let mut n = 0u64;
    let mut embeddings: HashMap<u32, Arc<[f32]>> = HashMap::new();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        // Take the NEWEST frame, discarding anything queued behind it. The
        // pipeline runs slower than the source by construction (detection is
        // the bottleneck), so the queue is usually non-empty; draining it
        // here means the frame we detect on is the most recent one rather
        // than whichever the bounded channel happened to keep. Ingest's own
        // drop-oldest does the same thing, but only once the queue is FULL —
        // by which point the frame it hands over is already stale.
        // ...but ONLY when frames are being produced faster than we consume
        // them, which is exactly when ingest is in drop-oldest mode. For an
        // offline file the decoder BLOCKS on us instead, so every frame it
        // hands over is one it has already committed to — dropping those
        // silently loses work the caller asked for, and makes a run over a
        // file non-deterministic. `sa bench` and `--log` both depend on
        // seeing every frame.
        if drop_stale {
            while ingest.frames().len() > 1 {
                let _ = ingest.frames().try_recv();
                skipped_stale += 1;
            }
        }
        let frame = match ingest.frames().recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(f) => f,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // A file source that has finished leaves the channel empty and
                // its thread gone; a live source just had a hiccup.
                if ingest.stats().produced.load(Ordering::Relaxed) > 0 && ingest.frames().is_empty() && !ingest_alive(&ingest) {
                    break;
                }
                continue;
            }
            Err(_) => break,
        };
        let t0 = Instant::now();
        let mut lat = StageLatency::default();

        // SHOT CUT: the people in the last shot are not in this one. Drop the
        // tracks rather than let them coast onto whatever the new shot shows
        // — that is what scatters rings across a close-up.
        if cfg.video.scene_cut_corr > 0.0 && scene.update(&frame) {
            let n_before = tracker.track_count();
            tracker.clear_tracks();
            ball = BallTracker::new(BallParams::default());
            if n_before > 0 {
                tracing::info!(frame = frame.id, dropped_tracks = n_before, "shot cut");
            }
        }

        // -- detect ----------------------------------------------------------
        // On a skipped frame the tracker still advances (its filters are
        // stepped by the real elapsed frames in `update_at`), so boxes keep
        // moving; only the expensive part is omitted.
        let detect_now = frame.id % detect_every.max(1) == 0;
        let mut dets = if detect_now { detector.detect(&frame)? } else { Vec::new() };
        if detect_now {
            // SIZE SANITY. A person occupying most of the frame height is a
            // close-up, not a player on a pitch, and a person a handful of
            // pixels tall is texture. Neither can be tracked usefully and
            // both produce rings in the wrong place, so drop them here
            // rather than downstream — the tracker should never see them.
            let fh = frame.height as f32;
            dets.retain(|d| {
                if !d.class.is_person() {
                    return true;
                }
                let h = d.bbox.h();
                h >= fh * MIN_PERSON_FRAC && h <= fh * MAX_PERSON_FRAC
            });
            // SCALE AGAINST THIS FRAME'S OWN PLAYERS, not a fixed fraction.
            // In a wide shot the players are ~60 px tall and the spectators
            // behind them are 200-500 px; in a tight shot the same player is
            // 300 px and nothing else is a person at all. No constant
            // separates those two cases, but the frame's own MEDIAN does:
            // a football frame is mostly players, so the median detection is
            // a player by construction, and anything several times that is
            // not on the same plane — it is closer to the camera, which on a
            // pitch means it is not on the pitch.
            //
            // Measured on this repo's clip: real players sat at 54-77 px
            // (median 62) while the crowd reached 539. A fixed 4%-75% band
            // admitted everything from 29 to 540 and let those through.
            let people: Vec<f32> = dets.iter().filter(|d| d.class.is_person()).map(|d| d.bbox.h()).collect();
            if people.len() >= MIN_PEOPLE_FOR_SCALE {
                let mut hs = people.clone();
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = hs[hs.len() / 2];
                let (lo, hi) = (median * PERSON_SCALE_LO, median * PERSON_SCALE_HI);
                dets.retain(|d| {
                    if !d.class.is_person() {
                        return true;
                    }
                    let h = d.bbox.h();
                    h >= lo && h <= hi
                });
            }
            // ...and ON THE PITCH. Size cannot separate a player from a
            // spectator behind the goal, but the ground under their feet
            // can. Skipped when the filter would reject nearly everyone,
            // which means the assumption (a dominant playing surface) does
            // not hold for this footage — an indoor court, a tight close-up
            // — and a filter that rejects everybody is worse than none.
            if surface_tol > 0.0 {
                let surface = surface_colour(&frame);
                let before = dets.len();
                let kept: Vec<_> = dets
                    .iter()
                    .filter(|d| !d.class.is_person() || on_playing_surface(&frame, &d.bbox, surface, surface_tol))
                    .cloned()
                    .collect();
                let people_before = dets.iter().filter(|d| d.class.is_person()).count();
                let people_after = kept.iter().filter(|d| d.class.is_person()).count();
                if people_before == 0 || people_after * 3 >= people_before {
                    dets = kept;
                } else {
                    tracing::debug!(before, people_before, people_after, "surface filter would reject most detections — skipped");
                }
            }
        }
        lat.detect_ms = t0.elapsed().as_secs_f32() * 1000.0;
        if detect_now {
            det_ms_ema = if det_ms_ema == 0.0 { lat.detect_ms } else { 0.8 * det_ms_ema + 0.2 * lat.detect_ms };
            n_detects += 1;
            // RE-EVALUATED ON EVERY DETECTION, not on a frame-number schedule.
            // Gating this on `frame.id % 25 == 0` meant the check itself only
            // fired when the stride happened to divide 25 — so the stride
            // ratcheted up to its cap and could never come back down.
            if adaptive && n_detects.is_multiple_of(8) {
                let per_frame_ms = 1000.0 / info.fps.max(1.0);
                // ASSOCIATION, not throughput, is what bounds this. Skipping
                // frames to "keep up with the source" is the wrong goal: a
                // player crosses several box-widths in half a second, the IoU
                // gate then rejects every real match, and the result is rings
                // sitting where nobody is. A stride of 4 at 25 fps means
                // 160 ms between detections, which a Kalman filter bridges
                // well. Beyond that the tracker is guessing, so we would
                // rather run behind the source and be right.
                let want = (det_ms_ema / per_frame_ms).ceil().clamp(1.0, MAX_DETECT_STRIDE as f32) as u64;
                if want != detect_every {
                    tracing::info!(detect_ms = det_ms_ema, from = detect_every, to = want, "adapting detect_every");
                    detect_every = want;
                }
            }
        }

        // -- optional ReID embeddings every N frames -------------------------
        let t1 = Instant::now();
        let embed_now = embedder.is_some() && frame.id.is_multiple_of(reembed_every);
        let inputs: Vec<Input<'_>> = dets
            .iter()
            .map(|d| {
                let emb = if embed_now && d.class.is_person() {
                    frame.crop(&d.bbox).and_then(|(w, h, rgb)| embedder.as_mut().and_then(|e| e.embed(&rgb, w, h).ok())).map(Arc::from)
                } else {
                    None
                };
                Input { det: d, embedding: emb }
            })
            .collect();

        // -- track -----------------------------------------------------------
        // The REAL frame id, not "one more": at 3 fps on a 25 fps source the
        // gap is ~8 frames, and a one-step prediction lands an eighth of the
        // way to where the player actually is.
        let mut tracks: Vec<Track> = if detect_now {
            tracker.update_at(frame.id, &inputs, (0.0, 0.0))
        } else {
            tracker.predict_at(frame.id)
        };
        for t in &tracks {
            if let Some(e) = &t.embedding {
                embeddings.insert(t.id, e.clone());
            }
        }
        lat.track_ms = t1.elapsed().as_secs_f32() * 1000.0;

        // A track keeps whatever size it was created at until it dies, so a
        // detection admitted while the frame was ambiguous — the opening
        // close-up, a frame with two people in it — survives into the wide
        // shot that follows and draws a body-sized ring over the grass.
        // Re-apply the scale test to the TRACKS, every frame, so the answer
        // follows the shot rather than the moment the track was born.
        if detect_now {
            let people: Vec<f32> = tracks.iter().filter(|t| t.class.is_person()).map(|t| t.bbox.h()).collect();
            if people.len() >= MIN_PEOPLE_FOR_SCALE {
                let mut hs = people;
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = hs[hs.len() / 2];
                let (lo, hi) = (median * PERSON_SCALE_LO, median * PERSON_SCALE_HI);
                tracks.retain(|t| !t.class.is_person() || (t.bbox.h() >= lo && t.bbox.h() <= hi));
            }
        }

        // -- teams -----------------------------------------------------------
        let t2 = Instant::now();
        let mut kits: HashMap<u32, [u8; 3]> = HashMap::new();
        for t in &tracks {
            if t.class.is_person() && matches!(t.state, sa_core::TrackState::Confirmed) {
                let (rgb, contrast) = kit_reading(&frame, &t.bbox, 0.0);
                if contrast >= KIT_MIN_CONTRAST {
                    kits.insert(t.id, rgb);
                }
            }
        }
        let assignment = teams.observe_frame(&kits).clone();
        for t in tracks.iter_mut() {
            t.team = match kits.get(&t.id) {
                Some(rgb) if assignment.confidence > 0.0 => assignment.classify(*rgb),
                _ => assignment.team(t.id),
            };
            if t.class == Class::Referee {
                t.team = Team::Referee;
            }
        }
        lat.identity_ms = t2.elapsed().as_secs_f32() * 1000.0;

        // -- pitch -----------------------------------------------------------
        let t3 = Instant::now();
        if cfg.pitch.enabled && frame.id % cfg.pitch.keyframe_every.max(1) as u64 == 0 {
            let _ = kp_tx.try_send(frame.clone());
        }
        let calibration = { calibrator.lock().at(frame.id, frame.width, frame.height) };
        if let Some(cal) = &calibration {
            let field = calibrator.lock();
            let f = field.field();
            for t in tracks.iter_mut() {
                let p = foot_point(&t.bbox, 0.0);
                t.pitch = cal.project(p).filter(|q| f.on_field(*q, margin));
            }
        }
        lat.pitch_ms = t3.elapsed().as_secs_f32() * 1000.0;

        // -- ball ------------------------------------------------------------
        let ball_dets: Vec<sa_core::Detection> = dets.iter().filter(|d| d.class == Class::Ball).cloned().collect();
        let ball_state = ball.update(frame.id, &ball_dets).map(|b| BallState {
            image: b.xy,
            pitch: calibration.as_ref().and_then(|c| c.project(b.xy)),
            seen: b.seen,
            conf: b.conf,
        });

        // -- analytics -------------------------------------------------------
        let t4 = Instant::now();
        let mut stats = LiveStats::default();
        if cfg.analytics.enabled && calibration.is_some() {
            let bxy = ball_state.as_ref().and_then(|b| b.pitch.map(|p| (p, b.seen)));
            analytics.observe(frame.id, &tracks, bxy);
            analytics.fill_stats(&tracks, &mut stats);
        }
        lat.analytics_ms = t4.elapsed().as_secs_f32() * 1000.0;

        // -- state -----------------------------------------------------------
        n += 1;
        processed.store(n, Ordering::Relaxed);
        // DETECTION rate, not loop rate: a motion-predicted frame costs
        // microseconds, so counting it would report thousands of fps and say
        // nothing about whether the machine is keeping up.
        if detect_now {
            let dt = last_t.elapsed().as_secs_f32();
            last_t = Instant::now();
            let inst = if dt > 0.0 { 1.0 / dt } else { 0.0 };
            fps_ema = if fps_ema == 0.0 { inst } else { 0.9 * fps_ema + 0.1 * inst };
        }
        lat.total_ms = t0.elapsed().as_secs_f32() * 1000.0;
        stats.fps = fps_ema;
        stats.display_fps = info.fps;
        stats.detect_every = detect_every as u32;
        // Frames the pipeline never saw. Both causes are the same thing —
        // detection is slower than the source — and reporting them together
        // is what makes the number actionable.
        stats.dropped = ingest.stats().dropped.load(Ordering::Relaxed) + skipped_stale;
        stats.latency = lat;
        // Strip embeddings from the published state; they are engine-internal.
        for t in tracks.iter_mut() {
            t.embedding = None;
        }
        let st = Arc::new(FrameState {
            frame_id: frame.id,
            pts_ms: frame.pts.as_millis() as u64,
            width: frame.width,
            height: frame.height,
            tracks,
            ball: ball_state,
            calibration,
            teams: assignment.colours(),
            stats,
        });
        *latest.lock() = Some(st.clone());
        *overlay.lock() = Some(st.clone());
        // Hand the preview a snapshot it can advance on its own. A clone of
        // the tracker is cheap (a few dozen small filters) and keeps the
        // preview off the pipeline's own state entirely — no lock contention
        // on the hot path, and nothing it does can perturb tracking.
        {
            let snap = tracker.clone();
            *predictor.lock() = Some(Box::new(move |f: u64| snap.predict_at(f)));
        }
        // LIVE: never block the pipeline for a slow consumer — discard the
        // oldest queued state so the UI always sees the newest.
        // OFFLINE: the consumer must see EVERY state (`sa bench` measures
        // them, `--log` records them), so block instead. Stealing from
        // `state_rx` here would also race the real consumer for its own
        // messages, which is what deadlocked `sa bench`.
        if drop_stale {
            push_drop_oldest(&state_tx, &state_rx, st.clone());
        } else {
            // Blocking, so an offline consumer sees every state — but it
            // must still notice `stop`. A plain `send` here deadlocks the
            // moment the consumer has what it wants and calls `stop()`,
            // because `stop()` joins this thread while this thread waits on
            // that consumer. Time-boxed retries break the cycle.
            let mut pending = Some(st.clone());
            while let Some(v) = pending.take() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                match state_tx.send_timeout(v, std::time::Duration::from_millis(100)) {
                    Ok(()) => {}
                    Err(crossbeam_channel::SendTimeoutError::Timeout(v)) => pending = Some(v),
                    Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => break,
                }
            }
        }
        let _ = render_tx.try_send((frame.clone(), st.clone()));
        if let Some(s) = session.as_mut() {
            let _ = s.write(&st);
        }
        if n.is_multiple_of(100) {
            tracing::info!(frame = frame.id, fps = format!("{fps_ema:.1}"), tracks = st.tracks.len(), det_ms = format!("{:.1}", lat.detect_ms), calibrated = st.calibration.is_some(), "pipeline");
        }
        let alive: Vec<u32> = st.tracks.iter().map(|t| t.id).collect();
        teams.retain(&alive);
        embeddings.retain(|k, _| alive.contains(k));
    }

    ingest.stop();
    let elapsed = t_start.elapsed().as_secs_f32();
    let session_path = match session.take() {
        Some(s) => {
            let p = s.path.clone();
            let _ = s.finish();
            Some(p)
        }
        None => None,
    };
    let analytics_json = analytics.to_json();
    if let Some(p) = &cfg.output.session_log {
        let dir = if p.is_absolute() { p.clone() } else { root.join(p) };
        if let Some(d) = dir.parent() {
            let _ = std::fs::write(d.join("analytics.json"), serde_json::to_string_pretty(&analytics_json).unwrap_or_default());
        }
    }
    Ok(Summary {
        frames_processed: n,
        frames_dropped: ingest.stats().dropped.load(Ordering::Relaxed),
        avg_fps: if elapsed > 0.0 { n as f32 / elapsed } else { 0.0 },
        analytics: analytics_json,
        recording: if cfg.output.record { Some(cfg.output.record_path.clone().unwrap_or_else(|| root.join("outputs/session.mp4"))) } else { None },
        session_log: session_path,
    })
}

fn ingest_alive(ingest: &Ingest) -> bool {
    ingest.is_alive()
}

/// Homography (image px → field m) from ≥4 user-picked pairs, for the manual
/// calibration path. Exposed here so the desktop shell needs no direct
/// dependency on `sa-pitch`.
pub fn homography_from_points(image: &[sa_core::Point2], field: &[sa_core::Point2]) -> Option<[[f64; 3]; 3]> {
    let h = sa_pitch::homography::dlt(image, field)?;
    if !sa_pitch::homography::plausible(&h, 105.0, 68.0) {
        return None;
    }
    Some(sa_pitch::to_array(&h))
}
