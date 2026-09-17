//! Session construction with execution-provider selection.

use std::path::Path;

#[allow(unused_imports)]
use ort::ep::ExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use sa_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    TensorRt,
    Cuda,
    OpenVino,
    DirectMl,
    CoreMl,
    Cpu,
}

/// Build a session for `path`. Providers this binary was compiled with are
/// tried in order of speed; ONNX Runtime falls back to CPU for any that are
/// unavailable on this machine, so the result always runs. Which one won is
/// reported in `Provider`, best-effort from what was registered.
pub fn build_session(path: &Path, threads: usize) -> Result<(Session, Provider)> {
    let mut builder = Session::builder().map_err(|e| Error::Model(format!("session builder: {e}")))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Model(e.to_string()))?;
    // 0 = PHYSICAL cores, not logical. `available_parallelism` reports SMT
    // threads, and oversubscribing a convolution-heavy graph makes it slower,
    // not faster: measured on this repo's yolo11n-512, 2 threads 117 ms,
    // 4 threads 80 ms, 6 threads 88 ms, 8 threads 115 ms on a 4-core/8-thread
    // Ryzen. The two hyperthreads of a core contend for one FMA unit, and the
    // extra synchronisation costs more than the parallelism returns. Leaving
    // the other threads free also matters here — decode, JPEG and tracking
    // all want CPU at the same time.
    let threads = if threads > 0 { threads } else { physical_cores() };
    builder = builder.with_intra_threads(threads).map_err(|e| Error::Model(e.to_string()))?;

    // REPORT WHAT THE MACHINE ACCEPTED, not what was compiled in. A binary
    // built `--features coreml` and run on a machine without it would
    // otherwise claim acceleration while quietly running on the CPU — and the
    // only symptom is "it feels slow", which is exactly the question a user
    // cannot answer for themselves. `is_available()` asks the runtime.
    let mut eps: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    #[allow(unused_mut)]
    let mut chosen = Provider::Cpu;
    #[allow(unused_mut)]
    let mut compiled: Vec<&'static str> = Vec::new();

    #[cfg(feature = "tensorrt")]
    {
        compiled.push("tensorrt");
        let ep = ort::ep::TensorRT::default().with_fp16(true);
        if ep.is_available().unwrap_or(false) && chosen == Provider::Cpu {
            chosen = Provider::TensorRt;
        }
        eps.push(ep.build());
    }
    #[cfg(feature = "cuda")]
    {
        compiled.push("cuda");
        let ep = ort::ep::CUDA::default();
        if ep.is_available().unwrap_or(false) && chosen == Provider::Cpu {
            chosen = Provider::Cuda;
        }
        eps.push(ep.build());
    }
    #[cfg(feature = "openvino")]
    {
        compiled.push("openvino");
        let ep = ort::ep::OpenVINO::default();
        if ep.is_available().unwrap_or(false) && chosen == Provider::Cpu {
            chosen = Provider::OpenVino;
        }
        eps.push(ep.build());
    }
    #[cfg(feature = "directml")]
    {
        compiled.push("directml");
        let ep = ort::ep::DirectML::default();
        if ep.is_available().unwrap_or(false) && chosen == Provider::Cpu {
            chosen = Provider::DirectMl;
        }
        eps.push(ep.build());
    }
    #[cfg(feature = "coreml")]
    {
        compiled.push("coreml");
        let ep = ort::ep::CoreML::default();
        if ep.is_available().unwrap_or(false) && chosen == Provider::Cpu {
            chosen = Provider::CoreMl;
        }
        eps.push(ep.build());
    }
    if !compiled.is_empty() && chosen == Provider::Cpu {
        tracing::warn!(
            compiled = ?compiled,
            "built with hardware acceleration but none is available here — running on CPU"
        );
    }
    eps.push(ort::ep::CPU::default().build());
    builder = builder.with_execution_providers(eps).map_err(|e| Error::Model(format!("execution providers: {e}")))?;

    let session = builder
        .commit_from_file(path)
        .map_err(|e| Error::Model(format!("load {}: {e}", path.display())))?;
    tracing::info!(model = %path.display(), provider = ?chosen, "session ready");
    Ok((session, chosen))
}

/// Physical cores, falling back to half the logical count (the usual SMT
/// ratio) and finally to a safe 4.
fn physical_cores() -> usize {
    #[cfg(target_os = "linux")]
    {
        // Count distinct (physical id, core id) pairs in /proc/cpuinfo.
        if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut seen = std::collections::HashSet::new();
            let (mut phys, mut core) = (None, None);
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("physical id") {
                    phys = v.split(':').nth(1).and_then(|x| x.trim().parse::<u32>().ok());
                } else if let Some(v) = line.strip_prefix("core id") {
                    core = v.split(':').nth(1).and_then(|x| x.trim().parse::<u32>().ok());
                } else if line.trim().is_empty() {
                    if let (Some(p), Some(c)) = (phys, core) {
                        seen.insert((p, c));
                    }
                    phys = None;
                    core = None;
                }
            }
            if let (Some(p), Some(c)) = (phys, core) {
                seen.insert((p, c));
            }
            if !seen.is_empty() {
                return seen.len();
            }
        }
    }
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (logical / 2).max(1)
}

/// The static dims of the first input, if the model declares them.
pub fn input_dims(session: &Session) -> Option<Vec<i64>> {
    let outlet = session.inputs().first()?;
    outlet.dtype().tensor_shape().map(|s| s.to_vec())
}
