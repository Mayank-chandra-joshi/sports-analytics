//! Tauri shell over `sa-engine`. The engine runs in-process; the UI gets a
//! stream of `FrameState` over an IPC channel and shows the picture from the
//! engine's loopback MJPEG server.

use std::path::PathBuf;

use parking_lot::Mutex;
use sa_core::{Config, FrameState};
use sa_engine::{Engine, EngineOptions, VideoSource};
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};

#[derive(Default)]
struct AppState {
    engine: Mutex<Option<Engine>>,
    config: Mutex<Config>,
    root: Mutex<PathBuf>,
}

#[derive(serde::Serialize)]
struct StartInfo {
    preview_url: Option<String>,
    width: u32,
    height: u32,
    fps: f32,
    frames: u64,
}

fn resolve_root(app: &AppHandle) -> PathBuf {
    // Dev: the workspace root holds models/ and runtime/. Bundled: the
    // resource dir. Walk up from the executable looking for `models/`.
    let mut cur = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
    while let Some(d) = cur {
        if d.join("models").is_dir() {
            return d;
        }
        cur = d.parent().map(|p| p.to_path_buf());
    }
    app.path().resource_dir().unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
}

#[tauri::command]
fn get_config(state: State<'_, AppState>) -> Config {
    state.config.lock().clone()
}

#[tauri::command]
fn set_config(state: State<'_, AppState>, config: Config) {
    *state.config.lock() = config;
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Vec<String> {
    let dir = state.root.lock().join("models");
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".onnx")).collect())
        .unwrap_or_default();
    v.sort();
    v
}

#[tauri::command]
fn start(state: State<'_, AppState>, source: String, realtime: bool, on_state: Channel<FrameState>) -> Result<StartInfo, String> {
    let mut guard = state.engine.lock();
    if guard.is_some() {
        return Err("engine already running".into());
    }
    let cfg = state.config.lock().clone();
    let root = state.root.lock().clone();
    let engine = Engine::start(EngineOptions {
        config: cfg,
        source: VideoSource::parse(&source),
        root,
        preview: true,
        realtime,
        burn_overlays: false,
    })
    .map_err(|e| e.to_string())?;
    let info = engine.info();
    let rx = engine.states().clone();
    // Forward every state to the webview. `Channel` is Tauri's streaming
    // primitive; the receiver is drop-oldest so a slow UI never backs up
    // the pipeline.
    std::thread::Builder::new()
        .name("sa-ui-feed".into())
        .spawn(move || {
            while let Ok(st) = rx.recv() {
                if on_state.send((*st).clone()).is_err() {
                    break;
                }
            }
        })
        .map_err(|e| e.to_string())?;
    let out = StartInfo { preview_url: engine.preview_url(), width: info.width, height: info.height, fps: info.fps, frames: info.frames };
    *guard = Some(engine);
    Ok(out)
}

#[tauri::command]
fn stop(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let engine = state.engine.lock().take().ok_or("engine not running")?;
    let summary = engine.stop().map_err(|e| e.to_string())?;
    serde_json::to_value(summary).map_err(|e| e.to_string())
}

#[tauri::command]
fn is_running(state: State<'_, AppState>) -> bool {
    state.engine.lock().as_ref().map_or(false, |e| e.is_running())
}

#[tauri::command]
fn latest_state(state: State<'_, AppState>) -> Option<FrameState> {
    state.engine.lock().as_ref().and_then(|e| e.latest()).map(|s| (*s).clone())
}

/// `h` is image px → field metres, row-major. `None` clears the override.
#[tauri::command]
fn set_manual_calibration(state: State<'_, AppState>, h: Option<[[f64; 3]; 3]>) -> Result<(), String> {
    let g = state.engine.lock();
    let e = g.as_ref().ok_or("engine not running")?;
    e.set_manual_calibration(h);
    Ok(())
}

/// Solve a homography from ≥4 (image px, field m) pairs picked in the UI
/// and apply it. Returns the matrix so the UI can store it as a preset.
#[tauri::command]
fn calibrate_from_points(state: State<'_, AppState>, image: Vec<[f32; 2]>, field: Vec<[f32; 2]>) -> Result<[[f64; 3]; 3], String> {
    use sa_core::Point2;
    if image.len() < 4 || image.len() != field.len() {
        return Err("need at least four point pairs".into());
    }
    let src: Vec<Point2> = image.iter().map(|p| Point2::new(p[0], p[1])).collect();
    let dst: Vec<Point2> = field.iter().map(|p| Point2::new(p[0], p[1])).collect();
    let h = sa_pitch_dlt(&src, &dst).ok_or("points are degenerate")?;
    if let Some(e) = state.engine.lock().as_ref() {
        e.set_manual_calibration(Some(h));
    }
    Ok(h)
}

fn sa_pitch_dlt(src: &[sa_core::Point2], dst: &[sa_core::Point2]) -> Option<[[f64; 3]; 3]> {
    // sa-engine re-exports nothing from sa-pitch; go through the engine's
    // public helper to keep the shell's dependency list short.
    sa_engine::homography_from_points(src, dst)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).try_init();
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let root = resolve_root(app.handle());
            let mut config = Config::default();
            if let Ok(s) = std::fs::read_to_string(root.join("sa.toml")) {
                if let Ok(c) = Config::from_toml(&s) {
                    config = c;
                }
            }
            tracing::info!(root = %root.display(), "app root");
            app.manage(AppState { engine: Mutex::new(None), config: Mutex::new(config), root: Mutex::new(root) });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            set_config,
            list_models,
            start,
            stop,
            is_running,
            latest_state,
            set_manual_calibration,
            calibrate_from_points
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
