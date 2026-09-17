//! Object detection. `YoloDetector` runs any Ultralytics-exported YOLO
//! (v8/v11, output `[1, 4+nc, N]`) or a transposed `[1, N, 4+nc]` variant.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use sa_core::{BBox, Detection, Error, Frame, Result};
use sa_core::profile::ClassMap;

use crate::preprocess::{nms, Preprocessor};
use crate::session::{build_session, input_dims, Provider};

pub trait Detector: Send {
    /// Detections in frame pixel coordinates, already class-mapped and NMS'd.
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct YoloOptions {
    pub input_size: u32,
    pub conf: f32,
    pub iou_nms: f32,
    pub threads: usize,
    pub classes: ClassMap,
}

pub struct YoloDetector {
    session: Session,
    pre: Preprocessor,
    opts: YoloOptions,
    input_name: String,
    provider: Provider,
    name: String,
}

impl YoloDetector {
    pub fn load(path: &Path, mut opts: YoloOptions) -> Result<Self> {
        let (session, provider) = build_session(path, opts.threads)?;
        // Trust the model's declared input over config when it is static —
        // including a NON-SQUARE one, which is the point on 16:9 footage.
        let (mut in_w, mut in_h) = (opts.input_size, opts.input_size);
        if let Some(d) = input_dims(&session) {
            if d.len() == 4 && d[2] > 0 && d[3] > 0 {
                in_h = d[2] as u32;
                in_w = d[3] as u32;
                opts.input_size = in_w.max(in_h);
            }
        }
        let input_name = session.inputs().first().map(|o| o.name().to_string()).unwrap_or_else(|| "images".into());
        let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "yolo".into());
        Ok(Self { pre: Preprocessor::with_shape(in_w, in_h), session, opts, input_name, provider, name })
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }
}

impl Detector for YoloDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let lb = self.pre.letterbox(frame)?;
        let (iw, ih) = (self.pre.in_w() as usize, self.pre.in_h() as usize);
        let input = Tensor::from_array(([1usize, 3, ih, iw], self.pre.tensor.clone()))
            .map_err(|e| Error::Model(format!("input tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])
            .map_err(|e| Error::Model(format!("run: {e}")))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("output: {e}")))?;
        let dims: Vec<i64> = shape.to_vec();
        if dims.len() != 3 {
            return Err(Error::Model(format!("unexpected output shape {dims:?}")));
        }
        // Ultralytics: [1, 4+nc, N]. Some exports: [1, N, 4+nc]. Tell them
        // apart by which axis is the small one.
        let (c, n, transposed) = if dims[1] < dims[2] {
            (dims[1] as usize, dims[2] as usize, false)
        } else {
            (dims[2] as usize, dims[1] as usize, true)
        };
        let nc = c - 4;
        let at = |row: usize, col: usize| -> f32 {
            if transposed { data[row * c + col] } else { data[col * n + row] }
        };
        let mut raw: Vec<(BBox, f32, u16)> = Vec::new();
        for i in 0..n {
            let mut best = 0.0f32;
            let mut best_c = 0usize;
            for k in 0..nc {
                let v = at(i, 4 + k);
                if v > best {
                    best = v;
                    best_c = k;
                }
            }
            if best < self.opts.conf {
                continue;
            }
            // Only classes the profile knows about survive: everything else
            // is a chair or a bottle and costs NMS time for nothing.
            if self.opts.classes.map(best_c as u16).is_none() {
                continue;
            }
            let (cx, cy, w, h) = (at(i, 0), at(i, 1), at(i, 2), at(i, 3));
            let (x1, y1) = lb.unmap(cx - w / 2.0, cy - h / 2.0);
            let (x2, y2) = lb.unmap(cx + w / 2.0, cy + h / 2.0);
            raw.push((BBox::new(x1, y1, x2, y2).clamped(frame.width as f32, frame.height as f32), best, best_c as u16));
        }
        nms(&mut raw, self.opts.iou_nms);
        Ok(raw
            .into_iter()
            .filter_map(|(b, conf, cls)| self.opts.classes.map(cls).map(|class| Detection { class, bbox: b, conf }))
            .collect())
    }
}

/// Deterministic stand-in for tests and for running the pipeline with no
/// model present: emits nothing.
pub struct MockDetector;

impl Detector for MockDetector {
    fn detect(&mut self, _frame: &Frame) -> Result<Vec<Detection>> {
        Ok(Vec::new())
    }
    fn name(&self) -> &str {
        "mock"
    }
}
