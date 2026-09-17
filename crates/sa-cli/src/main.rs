//! `sa` — run the engine headless, benchmark it, or inspect models.
//!
//!   sa run <source> [--config sa.toml] [--record out.mp4] [--log session.jsonl] [--preview] [--realtime]
//!   sa bench <source> [--frames N]
//!   sa models

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use sa_core::Config;
use sa_engine::{Engine, EngineOptions, VideoSource};

#[derive(Parser)]
#[command(name = "sa", version, about = "sports-analytics engine")]
struct Cli {
    /// Project root (where `models/` and `runtime/` live). Defaults to the
    /// directory containing this binary's workspace, or cwd.
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// TOML config file; defaults are used for anything missing.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Process a source to completion (file) or until Ctrl-C (live).
    Run {
        source: String,
        /// Write an annotated MP4.
        #[arg(long)]
        record: Option<PathBuf>,
        /// Write a JSONL session log (one FrameState per line) and analytics.json.
        #[arg(long)]
        log: Option<PathBuf>,
        /// Serve an MJPEG preview on loopback and print its URL.
        #[arg(long)]
        preview: bool,
        /// Play a file at its native rate, like a live feed.
        #[arg(long)]
        realtime: bool,
        /// Detector model path (overrides config).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Pitch keypoint model path (overrides config).
        #[arg(long)]
        keypoints: Option<PathBuf>,
    },
    /// Measure throughput and per-stage latency.
    Bench {
        source: String,
        #[arg(long, default_value_t = 300)]
        frames: u64,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// List models in `models/` with their manifest status.
    Models,
}

fn find_root(explicit: Option<PathBuf>) -> PathBuf {
    if let Some(r) = explicit {
        return r;
    }
    // Walk up from the executable looking for a `models/` directory.
    let mut cur = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
    while let Some(d) = cur {
        if d.join("models").is_dir() {
            return d;
        }
        cur = d.parent().map(|p| p.to_path_buf());
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn load_config(root: &std::path::Path, path: Option<PathBuf>) -> anyhow::Result<Config> {
    let p = path.or_else(|| {
        let d = root.join("sa.toml");
        if d.is_file() { Some(d) } else { None }
    });
    Ok(match p {
        Some(p) => Config::from_toml(&std::fs::read_to_string(&p)?)?,
        None => Config::default(),
    })
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .compact()
        .init();
    let cli = Cli::parse();
    let root = find_root(cli.root.clone());
    let mut cfg = load_config(&root, cli.config.clone())?;

    match cli.cmd {
        Cmd::Run { source, record, log, preview, realtime, model, keypoints } => {
            if let Some(m) = model {
                cfg.detection.model = m;
            }
            if let Some(k) = keypoints {
                cfg.pitch.keypoint_model = Some(k);
            }
            if let Some(r) = record {
                cfg.output.record = true;
                cfg.output.record_path = Some(r);
            }
            if let Some(l) = log {
                cfg.output.session_log = Some(l);
            }
            let engine = Engine::start(EngineOptions {
                config: cfg,
                source: VideoSource::parse(&source),
                root: root.clone(),
                preview,
                realtime,
                burn_overlays: true,
            })?;
            if let Some(u) = engine.preview_url() {
                println!("preview: {u}");
            }
            let info = engine.info();
            println!("source: {}x{} @ {:.2} fps, {} frames", info.width, info.height, info.fps, info.frames);
            let states = engine.states().clone();
            let printer = std::thread::spawn(move || {
                let mut last = Instant::now();
                while let Ok(st) = states.recv() {
                    if last.elapsed().as_secs_f32() >= 1.0 {
                        last = Instant::now();
                        let s = &st.stats;
                        println!(
                            "frame {:>6} | {:>5.1} fps | det {:>5.1}ms trk {:>4.1}ms id {:>4.1}ms pitch {:>4.1}ms | {:>2} tracks | ball {} | calib {} | poss A {:.0}% B {:.0}%",
                            st.frame_id, s.fps, s.latency.detect_ms, s.latency.track_ms, s.latency.identity_ms, s.latency.pitch_ms,
                            st.tracks.len(),
                            st.ball.as_ref().map_or("-", |b| if b.seen { "seen" } else { "held" }),
                            st.calibration.as_ref().map_or("none".to_string(), |c| format!("{:?}", c.source)),
                            s.possession_a * 100.0, s.possession_b * 100.0
                        );
                    }
                }
            });
            let summary = engine.wait()?;
            let _ = printer.join();
            println!(
                "done: {} frames, {} dropped, {:.1} fps avg",
                summary.frames_processed, summary.frames_dropped, summary.avg_fps
            );
            if let Some(p) = summary.recording {
                println!("recording: {}", p.display());
            }
            if let Some(p) = summary.session_log {
                println!("session log: {}", p.display());
            }
        }
        Cmd::Bench { source, frames, model } => {
            if let Some(m) = model {
                cfg.detection.model = m;
            }
            cfg.analytics.enabled = true;
            // A BENCHMARK MEASURES THE DETECTOR, so it must run on every
            // frame — the adaptive stride would otherwise skip most of them
            // and report the loop's speed (fractions of a millisecond)
            // rather than detection's. Setting it here, not asking the user
            // to remember a config override, is what makes `sa bench`
            // comparable between runs.
            cfg.video.detect_every = 1;
            let engine = Engine::start(EngineOptions { config: cfg, source: VideoSource::parse(&source), root: root.clone(), preview: false, realtime: false, burn_overlays: false })?;
            let states = engine.states().clone();
            let t0 = Instant::now();
            let mut det = Vec::new();
            let mut trk = Vec::new();
            let mut idn = Vec::new();
            let mut tot = Vec::new();
            let mut n = 0u64;
            while n < frames {
                // Generous: one detection can take several hundred ms on a
                // loaded CPU, and a short timeout here ends the benchmark
                // early and silently on exactly the machines it matters for.
                let Ok(st) = states.recv_timeout(std::time::Duration::from_secs(30)) else { break };
                det.push(st.stats.latency.detect_ms);
                trk.push(st.stats.latency.track_ms);
                idn.push(st.stats.latency.identity_ms);
                tot.push(st.stats.latency.total_ms);
                n += 1;
            }
            let wall = t0.elapsed().as_secs_f32();
            let _ = engine.stop();
            let med = |v: &mut Vec<f32>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if v.is_empty() { 0.0 } else { v[v.len() / 2] }
            };
            let p95 = |v: &Vec<f32>| if v.is_empty() { 0.0 } else { v[(v.len() as f32 * 0.95) as usize] };
            let (md, mt, mi, mtot) = (med(&mut det), med(&mut trk), med(&mut idn), med(&mut tot));
            println!("frames: {n}  wall: {wall:.1}s  throughput: {:.1} fps", n as f32 / wall.max(1e-3));
            println!("detect   median {md:.1} ms  p95 {:.1} ms", p95(&det));
            println!("track    median {mt:.2} ms  p95 {:.2} ms", p95(&trk));
            println!("identity median {mi:.2} ms  p95 {:.2} ms", p95(&idn));
            println!("total    median {mtot:.1} ms  p95 {:.1} ms", p95(&tot));
        }
        Cmd::Models => {
            let dir = root.join("models");
            let manifest = sa_infer::manifest::Manifest::load(&dir).ok();
            for e in std::fs::read_dir(&dir)? {
                let e = e?;
                let name = e.file_name().to_string_lossy().to_string();
                if !name.ends_with(".onnx") {
                    continue;
                }
                let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                let listed = manifest.as_ref().and_then(|m| m.models.iter().find(|x| x.file == name));
                let status = match listed {
                    Some(m) if !m.sha256.is_empty() => match sa_infer::manifest::sha256_file(&e.path()) {
                        Ok(h) if h.eq_ignore_ascii_case(&m.sha256) => "verified".to_string(),
                        Ok(_) => "HASH MISMATCH".to_string(),
                        Err(e) => format!("unreadable: {e}"),
                    },
                    Some(_) => "listed (no hash)".into(),
                    None => "not in manifest".into(),
                };
                println!("{name:40} {:>8.1} MB  {status}", size as f64 / 1e6);
            }
        }
    }
    Ok(())
}
