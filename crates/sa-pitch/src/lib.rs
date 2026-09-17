//! Pitch calibration: field models, homography maths, keyframe tracks, and
//! the `Calibrator` that turns keypoint detections into a per-frame
//! `Calibration` (or honestly, none).

pub mod football;
pub mod homography;
pub mod keyframes;

use sa_core::profile::FieldModel;
use sa_core::{CalibSource, Calibration, Point2};
use sa_infer::Keypoint;

pub use football::{Football, FootballDims};
pub use homography::{apply, from_array, to_array, H};
pub use keyframes::KeyframeTrack;

/// Where a detection touches the ground. Side-on: bottom centre. From
/// overhead the feet are under the body, so blend toward the box centre by
/// `overhead` in [0, 1] (the POC's `foot_point`).
pub fn foot_point(b: &sa_core::BBox, overhead: f32) -> Point2 {
    let f = b.foot();
    let c = b.center();
    let t = overhead.clamp(0.0, 1.0);
    Point2::new(f.x, (1.0 - t) * f.y + t * c.y)
}

/// Fraction of a coarse grid over the field that projects inside the image.
pub fn coverage(h_img_to_field: &H, img_w: u32, img_h: u32, field: &dyn FieldModel) -> f32 {
    let Some(inv) = h_img_to_field.try_inverse() else { return 0.0 };
    let d = field.dims();
    let (nx, ny) = (21, 14);
    let mut inside = 0;
    for i in 0..nx {
        for j in 0..ny {
            let p = Point2::new((i as f32 + 0.5) / nx as f32 * d.length, (j as f32 + 0.5) / ny as f32 * d.width);
            if let Some(q) = apply(&inv, p) {
                if q.x >= 0.0 && q.y >= 0.0 && q.x < img_w as f32 && q.y < img_h as f32 {
                    inside += 1;
                }
            }
        }
    }
    inside as f32 / (nx * ny) as f32
}

#[derive(Debug, Clone)]
pub struct CalibratorParams {
    pub ransac_px: f32,
    pub min_inliers: usize,
    pub max_gap: u32,
    pub min_keypoints: usize,
}

impl Default for CalibratorParams {
    fn default() -> Self {
        Self { ransac_px: 6.0, min_inliers: 6, max_gap: 30, min_keypoints: 6 }
    }
}

/// Owns the keyframe track and a manual override. Thread-safe by being
/// plain data; the engine wraps it in a mutex shared between the keypoint
/// thread (which calls `solve`) and the frame loop (which calls `at`).
pub struct Calibrator {
    field: Box<dyn FieldModel>,
    landmarks: Vec<(u16, Point2)>,
    track: KeyframeTrack,
    manual: Option<H>,
    params: CalibratorParams,
}

impl Calibrator {
    pub fn new(field: Box<dyn FieldModel>, params: CalibratorParams) -> Self {
        let d = field.dims();
        let landmarks = field.landmarks().into_iter().map(|(id, _, p)| (id.0, p)).collect();
        Self { track: KeyframeTrack::new(d.length, d.width, params.max_gap), field, landmarks, manual: None, params }
    }

    pub fn field(&self) -> &dyn FieldModel {
        self.field.as_ref()
    }

    /// Solve a homography from this frame's keypoints and, if plausible,
    /// bank it as a keyframe. Returns the fit's RMS in pixels.
    pub fn solve(&mut self, frame: u64, kps: &[Keypoint]) -> Option<f32> {
        if kps.len() < self.params.min_keypoints {
            return None;
        }
        let mut img = Vec::with_capacity(kps.len());
        let mut fld = Vec::with_capacity(kps.len());
        for k in kps {
            if let Some((_, p)) = self.landmarks.iter().find(|(i, _)| *i == k.index) {
                img.push(k.image);
                fld.push(*p);
            }
        }
        if img.len() < self.params.min_keypoints {
            return None;
        }
        // RANSAC in IMAGE space (field → image) so the threshold is in pixels,
        // which is where the model's noise lives; then invert.
        let r = homography::ransac(&fld, &img, self.params.ransac_px as f64, 200, self.params.min_inliers)?;
        let h = r.h.try_inverse()?;
        let d = self.field.dims();
        if !homography::plausible(&h, d.length, d.width) {
            tracing::debug!(frame, "keypoint fit rejected as implausible");
            return None;
        }
        self.track.add(frame, h);
        tracing::debug!(frame, inliers = r.inliers.len(), rms_px = r.rms, "keyframe solved");
        Some(r.rms as f32)
    }

    /// Set (or clear) a user-placed homography. Manual wins over the track.
    pub fn set_manual(&mut self, h: Option<H>) {
        self.manual = h;
    }

    pub fn track(&self) -> &KeyframeTrack {
        &self.track
    }

    /// The calibration to use for `frame`, if any.
    pub fn at(&self, frame: u64, img_w: u32, img_h: u32) -> Option<Calibration> {
        let (h, source) = match self.manual {
            Some(h) => (h, CalibSource::Manual),
            None => self.track.h_at(frame)?,
        };
        let cov = coverage(&h, img_w, img_h, self.field.as_ref());
        let confidence = match source {
            CalibSource::Manual => 1.0,
            CalibSource::Keyframe => 0.9,
            CalibSource::Interpolated => 0.75,
            CalibSource::Tracked => 0.5,
        };
        Some(Calibration { h: to_array(&h), confidence, source, coverage: cov })
    }
}
