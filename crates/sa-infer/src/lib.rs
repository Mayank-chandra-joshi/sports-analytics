//! Model inference behind traits, so the engine never names a framework.
//!
//! ONNX Runtime via `ort`. Execution providers are chosen at session build
//! from what this build was compiled with and what the machine has — the
//! order is TensorRT → CUDA → OpenVINO → DirectML → CoreML → CPU, and the one
//! actually used is logged. CPU always works.
//!
//! Every model is a local file. `manifest::verify` checks its hash before a
//! session is created; nothing here can download anything.

pub mod detector;
pub mod embedder;
pub mod keypoints;
pub mod manifest;
pub mod preprocess;
pub mod session;

pub use detector::{Detector, MockDetector, YoloDetector, YoloOptions};
pub use embedder::{Embedder, OsnetEmbedder};
pub use keypoints::{Keypoint, PitchKeypoints, YoloPoseKeypoints};
pub use session::{build_session, Provider};

/// Initialise ONNX Runtime once per process. Safe to call repeatedly.
///
/// The runtime is a shared library loaded at startup (`load-dynamic`), so the
/// same binary can ship with a CPU-only or a CUDA/TensorRT build of
/// `libonnxruntime` without recompiling. Resolution order:
///   1. `SA_ORT_DYLIB` env var (explicit path)
///   2. `runtime/libonnxruntime.{so,dylib,dll}` next to the executable, or in cwd
///   3. `ORT_DYLIB_PATH` / the system loader, as `ort` itself does
pub fn init() -> sa_core::Result<()> {
    use std::sync::OnceLock;
    static DONE: OnceLock<Result<(), String>> = OnceLock::new();
    DONE.get_or_init(|| -> Result<(), String> {
        let path = find_runtime();
        let env = match &path {
            Some(p) => ort::init_from(p).map_err(|e| format!("onnxruntime at {}: {e:?}", p.display()))?,
            None => ort::init(),
        };
        env.with_name("sports-analytics").commit();
        tracing::info!(runtime = ?path, "onnxruntime loaded");
        Ok(())
    })
    .clone()
    .map_err(sa_core::Error::Model)
}

fn find_runtime() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("SA_ORT_DYLIB") {
        return Some(p.into());
    }
    let lib = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("runtime").join(lib));
            candidates.push(dir.join(lib));
            // cargo run: target/debug/sa -> ../../runtime
            if let Some(root) = dir.parent().and_then(|p| p.parent()) {
                candidates.push(root.join("runtime").join(lib));
            }
        }
    }
    candidates.push(std::path::PathBuf::from("runtime").join(lib));
    candidates.into_iter().find(|p| p.is_file())
}
