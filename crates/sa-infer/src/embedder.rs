//! Appearance embeddings for ReID. OSNet (torchreid export, 256×128 input,
//! ImageNet normalisation) is the reference backbone — the same weights the
//! POC's `sportsreid_core.py` uses, exported to ONNX.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use sa_core::{Error, Result};

use crate::preprocess::crop_to_chw;
use crate::session::build_session;

pub trait Embedder: Send {
    /// L2-normalised vector for one RGB crop.
    fn embed(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<f32>>;
    fn dim(&self) -> usize;
}

pub struct OsnetEmbedder {
    session: Session,
    input_name: String,
    in_w: u32,
    in_h: u32,
    dim: usize,
}

impl OsnetEmbedder {
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        let (session, _) = build_session(path, threads)?;
        let input_name = session.inputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "input".into());
        // Size from the model when declared; the OSNet convention otherwise.
        let (in_h, in_w) = match crate::session::input_dims(&session) {
            Some(d) if d.len() == 4 && d[2] > 0 && d[3] > 0 => (d[2] as u32, d[3] as u32),
            _ => (256, 128),
        };
        let mut me = Self { session, input_name, in_w, in_h, dim: 0 };
        // Probe the output width once with a blank crop.
        let probe = vec![0u8; (in_w * in_h * 3) as usize];
        me.dim = me.embed(&probe, in_w, in_h)?.len();
        Ok(me)
    }
}

impl Embedder for OsnetEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
        let chw = crop_to_chw(rgb, w, h, self.in_w, self.in_h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
        let input = Tensor::from_array(([1usize, 3, self.in_h as usize, self.in_w as usize], chw))
            .map_err(|e| Error::Model(format!("input tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| Error::Model(format!("run: {e}")))?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| Error::Model(e.to_string()))?;
        let mut v = data.to_vec();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-8 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        Ok(v)
    }
}

/// Cosine similarity of two L2-normalised vectors.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
