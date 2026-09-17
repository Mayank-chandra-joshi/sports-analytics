//! Annotated-video recorder and the session log. The recorder pipes raw RGB
//! into an `ffmpeg` subprocess (libx264, or a hardware encoder if the flags
//! say so) — the mirror image of ingest, same zero-build-dependency reason.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use sa_core::{Error, FrameState, Result};

pub struct Recorder {
    child: Child,
    path: PathBuf,
    frames: u64,
}

impl Recorder {
    pub fn start(path: &Path, w: u32, h: u32, fps: f32, ffmpeg: &str) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let child = Command::new(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{w}x{h}"), "-r", &format!("{:.3}", fps.max(1.0)), "-i", "-"])
            .args(["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Source(format!("cannot start ffmpeg encoder ({e})")))?;
        Ok(Self { child, path: path.to_path_buf(), frames: 0 })
    }

    pub fn write(&mut self, rgb: &[u8]) -> Result<()> {
        if let Some(stdin) = self.child.stdin.as_mut() {
            stdin.write_all(rgb)?;
            self.frames += 1;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(PathBuf, u64)> {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
        Ok((self.path, self.frames))
    }
}

/// One JSON line per frame. Cheap to append, trivial to replay.
pub struct SessionLog {
    file: std::io::BufWriter<std::fs::File>,
    pub path: PathBuf,
    lines: u64,
}

impl SessionLog {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let f = std::fs::File::create(path)?;
        Ok(Self { file: std::io::BufWriter::with_capacity(1 << 16, f), path: path.to_path_buf(), lines: 0 })
    }

    pub fn write(&mut self, st: &FrameState) -> Result<()> {
        serde_json::to_writer(&mut self.file, st).map_err(|e| Error::Other(e.to_string()))?;
        self.file.write_all(b"\n")?;
        self.lines += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        self.file.flush()?;
        Ok(self.lines)
    }
}
