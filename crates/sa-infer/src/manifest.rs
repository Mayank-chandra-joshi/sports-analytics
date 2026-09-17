//! The offline model registry: `models/manifest.toml` lists every model the
//! engine may load, with a hash. A model that is not listed, or whose file
//! does not match, is refused — nothing is ever fetched to fill the gap.

use std::path::{Path, PathBuf};

use sa_core::{Error, Result};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub task: String,
    pub file: String,
    #[serde(default)]
    pub input: Vec<u32>,
    #[serde(default)]
    pub classes: Vec<String>,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub licence: String,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    #[serde(default, rename = "model")]
    pub models: Vec<ModelEntry>,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("manifest.toml");
        let s = std::fs::read_to_string(&p).map_err(|e| Error::Config(format!("{}: {e}", p.display())))?;
        toml::from_str(&s).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
    }

    pub fn find(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Resolve and verify `name`; returns the path to load.
    pub fn resolve(&self, dir: &Path, name: &str) -> Result<PathBuf> {
        let e = self.find(name).ok_or_else(|| Error::Model(format!("model `{name}` not in manifest")))?;
        let p = dir.join(&e.file);
        if !e.sha256.is_empty() {
            let got = sha256_file(&p)?;
            if !got.eq_ignore_ascii_case(&e.sha256) {
                return Err(Error::Model(format!("{}: hash mismatch (manifest {}, file {})", p.display(), e.sha256, got)));
            }
        }
        Ok(p)
    }
}

pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p).map_err(|e| Error::Model(format!("{}: {e}", p.display())))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(hex::encode(h.finalize()))
}
