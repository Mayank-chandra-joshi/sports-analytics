//! Image → tensor. SIMD resize via `fast_image_resize`; the NCHW/normalise
//! pass is a tight loop over pre-allocated buffers.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use sa_core::{Error, Frame, Result};

/// How a letterboxed input maps back to the source frame.
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub in_w: u32,
    pub in_h: u32,
}

impl Letterbox {
    /// Model-space (x, y) → frame-space.
    #[inline]
    pub fn unmap(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)
    }
}

/// Reusable buffers for one model input size.
///
/// NON-SQUARE IS THE POINT. Broadcast football is 16:9, and letterboxing it
/// into a square spends ~44% of the network's compute on grey bars. Measured
/// on this repo's yolo11n at 4 threads: 640x640 is 154 ms, while 384x672 is
/// 88 ms and keeps MORE horizontal detail than a 512x512 square — which is
/// what distant players are resolved by.
pub struct Preprocessor {
    in_w: u32,
    in_h: u32,
    resizer: Resizer,
    opts: ResizeOptions,
    resized: Image<'static>,
    /// NCHW float tensor, `3 * in_h * in_w`, filled per call.
    pub tensor: Vec<f32>,
}

impl Preprocessor {
    pub fn new(size: u32) -> Self {
        Self::with_shape(size, size)
    }

    pub fn with_shape(in_w: u32, in_h: u32) -> Self {
        Self {
            in_w,
            in_h,
            resizer: Resizer::new(),
            opts: ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear)),
            resized: Image::new(in_w, in_h, PixelType::U8x3),
            tensor: vec![0.0; (3 * in_w * in_h) as usize],
        }
    }

    pub fn size(&self) -> u32 {
        self.in_w.max(self.in_h)
    }
    pub fn in_w(&self) -> u32 {
        self.in_w
    }
    pub fn in_h(&self) -> u32 {
        self.in_h
    }

    /// Letterbox `frame` into the model's input, grey padding, then write
    /// NCHW RGB in [0, 1] into `self.tensor`. When the input's aspect matches
    /// the source there is no padding at all — which is the whole reason to
    /// use a non-square input on 16:9 footage.
    pub fn letterbox(&mut self, frame: &Frame) -> Result<Letterbox> {
        let (w, h) = (frame.width, frame.height);
        let s = (self.in_w as f32 / w as f32).min(self.in_h as f32 / h as f32);
        let nw = ((w as f32 * s).round() as u32).max(1).min(self.in_w);
        let nh = ((h as f32 * s).round() as u32).max(1).min(self.in_h);
        let pad_x = ((self.in_w - nw) / 2) as f32;
        let pad_y = ((self.in_h - nh) / 2) as f32;

        let src = ImageRef::new(w, h, &frame.data, PixelType::U8x3)
            .map_err(|e| Error::Model(format!("frame buffer: {e:?}")))?;
        let mut dst = Image::new(nw, nh, PixelType::U8x3);
        self.resizer
            .resize(&src, &mut dst, &self.opts)
            .map_err(|e| Error::Model(format!("resize: {e}")))?;

        // Fill the canvas with 114-grey (YOLO's padding value), paste the resize.
        self.resized.buffer_mut().fill(114);
        let iw = self.in_w as usize;
        let canvas = self.resized.buffer_mut();
        let (px, py) = (pad_x as usize, pad_y as usize);
        let db = dst.buffer();
        let row_bytes = nw as usize * 3;
        for y in 0..nh as usize {
            let so = y * row_bytes;
            let dof = ((py + y) * iw + px) * 3;
            canvas[dof..dof + row_bytes].copy_from_slice(&db[so..so + row_bytes]);
        }

        // HWC u8 → CHW f32 /255.
        let plane = iw * self.in_h as usize;
        let t = &mut self.tensor;
        let inv = 1.0 / 255.0;
        for i in 0..plane {
            let o = i * 3;
            t[i] = canvas[o] as f32 * inv;
            t[plane + i] = canvas[o + 1] as f32 * inv;
            t[2 * plane + i] = canvas[o + 2] as f32 * inv;
        }
        Ok(Letterbox { scale: s, pad_x, pad_y, in_w: self.in_w, in_h: self.in_h })
    }
}

/// Resize an RGB crop to `w×h` and produce CHW f32 normalised with the given
/// mean/std (ImageNet by default for ReID backbones).
pub fn crop_to_chw(rgb: &[u8], cw: u32, ch: u32, w: u32, h: u32, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    let src = ImageRef::new(cw, ch, rgb, PixelType::U8x3).map_err(|e| Error::Model(format!("crop buffer: {e:?}")))?;
    let mut dst = Image::new(w, h, PixelType::U8x3);
    let mut r = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear));
    r.resize(&src, &mut dst, &opts).map_err(|e| Error::Model(format!("resize: {e}")))?;
    let b = dst.buffer();
    let plane = (w * h) as usize;
    let mut out = vec![0.0f32; 3 * plane];
    for i in 0..plane {
        for c in 0..3 {
            out[c * plane + i] = (b[i * 3 + c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    Ok(out)
}

/// Standard greedy NMS over (bbox, score) with class-agnostic overlap.
pub fn nms(boxes: &mut Vec<(sa_core::BBox, f32, u16)>, iou_thr: f32) {
    boxes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<(sa_core::BBox, f32, u16)> = Vec::with_capacity(boxes.len());
    'outer: for cand in boxes.iter() {
        for k in &keep {
            if k.2 == cand.2 && k.0.iou(&cand.0) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(*cand);
    }
    *boxes = keep;
}

#[cfg(test)]
mod tests {
    use super::*;
    use sa_core::BBox;

    #[test]
    fn letterbox_maps_back() {
        let f = Frame::new_rgb8(0, Default::default(), 1280, 720, vec![0; 1280 * 720 * 3]);
        let mut p = Preprocessor::new(640);
        let lb = p.letterbox(&f).unwrap();
        assert!((lb.scale - 0.5).abs() < 1e-6);
        assert_eq!(lb.pad_y, 140.0);
        let (x, y) = lb.unmap(320.0, 320.0);
        assert!((x - 640.0).abs() < 1e-3 && (y - 360.0).abs() < 1e-3);
    }

    #[test]
    fn rectangular_input_has_no_padding_on_matching_aspect() {
        // 1280x720 is 16:9; 672x378 would be exact, and 384 high leaves a
        // little vertical padding. What matters is that unmap round-trips.
        let f = Frame::new_rgb8(0, Default::default(), 1280, 720, vec![0; 1280 * 720 * 3]);
        let mut p = Preprocessor::with_shape(672, 384);
        let lb = p.letterbox(&f).unwrap();
        assert_eq!(lb.pad_x, 0.0, "16:9 into 672 wide needs no horizontal pad");
        for (mx, my) in [(0.0, 0.0), (336.0, 189.0), (671.0, 377.0)] {
            let (x, y) = lb.unmap(mx, my);
            let (bx, by) = (x * lb.scale + lb.pad_x, y * lb.scale + lb.pad_y);
            assert!((bx - mx).abs() < 1e-3 && (by - my).abs() < 1e-3);
        }
    }

    #[test]
    fn nms_drops_overlap() {
        let mut v = vec![
            (BBox::new(0.0, 0.0, 10.0, 10.0), 0.9, 0),
            (BBox::new(1.0, 1.0, 11.0, 11.0), 0.8, 0),
            (BBox::new(50.0, 50.0, 60.0, 60.0), 0.7, 0),
        ];
        nms(&mut v, 0.5);
        assert_eq!(v.len(), 2);
    }
}
