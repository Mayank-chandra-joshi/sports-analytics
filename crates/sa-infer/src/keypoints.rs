//! Pitch landmark detection. The reference model is a YOLOv8-pose export
//! trained on the Roboflow *football-field-detection* set: one "object" (the
//! pitch) with 32 keypoints, output `[1, 4+1+32*3, N]` — the same model the
//! POC's `pitchkp.py` drives. Any pose-style export with `K` keypoints works;
//! the profile's `FieldModel::landmarks()` says what each index means.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use sa_core::{Error, Frame, Point2, Result};

use crate::preprocess::Preprocessor;
use crate::session::{build_session, input_dims};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    pub index: u16,
    pub image: Point2,
    pub conf: f32,
}

pub trait PitchKeypoints: Send {
    /// Landmarks in frame pixels with per-point confidence. Empty when the
    /// pitch was not found.
    fn keypoints(&mut self, frame: &Frame) -> Result<Vec<Keypoint>>;
}

pub struct YoloPoseKeypoints {
    session: Session,
    pre: Preprocessor,
    input_name: String,
    min_conf: f32,
}

impl YoloPoseKeypoints {
    pub fn load(path: &Path, input_size: u32, min_conf: f32, threads: usize) -> Result<Self> {
        let (session, _) = build_session(path, threads)?;
        let (in_w, in_h) = match input_dims(&session) {
            Some(d) if d.len() == 4 && d[2] > 0 && d[3] > 0 => (d[3] as u32, d[2] as u32),
            _ => (input_size, input_size),
        };
        let input_name = session.inputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "images".into());
        Ok(Self { session, pre: Preprocessor::with_shape(in_w, in_h), input_name, min_conf })
    }
}

impl PitchKeypoints for YoloPoseKeypoints {
    fn keypoints(&mut self, frame: &Frame) -> Result<Vec<Keypoint>> {
        let lb = self.pre.letterbox(frame)?;
        let (iw, ih) = (self.pre.in_w() as usize, self.pre.in_h() as usize);
        let input = Tensor::from_array(([1usize, 3, ih, iw], self.pre.tensor.clone()))
            .map_err(|e| Error::Model(format!("input tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| Error::Model(format!("run: {e}")))?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| Error::Model(e.to_string()))?;
        let dims: Vec<i64> = shape.to_vec();
        if dims.len() != 3 {
            return Err(Error::Model(format!("unexpected pose output {dims:?}")));
        }
        let (c, n) = (dims[1] as usize, dims[2] as usize);
        if c < 5 + 3 {
            return Ok(Vec::new());
        }
        let k = (c - 5) / 3;
        // Best "pitch" candidate by objectness.
        let mut best = (0usize, 0.0f32);
        for i in 0..n {
            let obj = data[4 * n + i];
            if obj > best.1 {
                best = (i, obj);
            }
        }
        if best.1 < self.min_conf {
            return Ok(Vec::new());
        }
        let i = best.0;
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let x = data[(5 + j * 3) * n + i];
            let y = data[(5 + j * 3 + 1) * n + i];
            let conf = data[(5 + j * 3 + 2) * n + i];
            if conf < self.min_conf {
                continue;
            }
            let (fx, fy) = lb.unmap(x, y);
            if fx < 0.0 || fy < 0.0 || fx > frame.width as f32 || fy > frame.height as f32 {
                continue;
            }
            out.push(Keypoint { index: j as u16, image: Point2::new(fx, fy), conf });
        }
        Ok(out)
    }
}
