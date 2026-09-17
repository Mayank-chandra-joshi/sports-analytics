//! A calibration for EVERY frame, from a calibration on SOME frames — the
//! POC's `KeyframeTrack`, ported. Solve a keyframe every N frames on a side
//! thread, interpolate between neighbours by blending the field's corner
//! quad in image space and re-solving (homographies do not average
//! elementwise), refuse to bridge gaps wider than `max_gap`, and hold only
//! briefly past the ends. For a live stream the "future" neighbour does not
//! exist yet, so `h_at` for the newest frames holds the last keyframe — and
//! says so through `CalibSource::Tracked`.

use std::collections::BTreeMap;

use sa_core::{CalibSource, Point2};

use crate::homography::{apply, dlt, H};

#[derive(Debug, Clone)]
pub struct KeyframeTrack {
    keys: BTreeMap<u64, H>,
    field_len: f32,
    field_w: f32,
    max_gap: u64,
    /// Keep this many most-recent keys (live streams never end).
    keep: usize,
}

impl KeyframeTrack {
    pub fn new(field_len: f32, field_w: f32, max_gap: u32) -> Self {
        Self { keys: BTreeMap::new(), field_len, field_w, max_gap: max_gap as u64, keep: 256 }
    }

    pub fn add(&mut self, frame: u64, h: H) {
        self.keys.insert(frame, h);
        while self.keys.len() > self.keep {
            let first = *self.keys.keys().next().unwrap();
            self.keys.remove(&first);
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    pub fn latest(&self) -> Option<(u64, &H)> {
        self.keys.iter().next_back().map(|(k, v)| (*k, v))
    }

    fn quad(&self, h: &H) -> Option<[Point2; 4]> {
        let inv = h.try_inverse()?;
        let c = [
            Point2::new(0.0, 0.0),
            Point2::new(self.field_len, 0.0),
            Point2::new(self.field_len, self.field_w),
            Point2::new(0.0, self.field_w),
        ];
        let mut q = [Point2::default(); 4];
        for i in 0..4 {
            q[i] = apply(&inv, c[i])?;
        }
        Some(q)
    }

    /// Homography for `frame`, or None when nothing honest can be said.
    pub fn h_at(&self, frame: u64) -> Option<(H, CalibSource)> {
        if self.keys.is_empty() {
            return None;
        }
        if let Some(h) = self.keys.get(&frame) {
            return Some((*h, CalibSource::Keyframe));
        }
        let edge = (self.max_gap / 2).max(1);
        let prev = self.keys.range(..frame).next_back();
        let next = self.keys.range(frame..).next();
        match (prev, next) {
            (Some((&i0, h0)), Some((&i1, h1))) => {
                if i1 - i0 > self.max_gap {
                    return None;
                }
                let t = (frame - i0) as f32 / (i1 - i0).max(1) as f32;
                let (Some(q0), Some(q1)) = (self.quad(h0), self.quad(h1)) else {
                    return Some((if t < 0.5 { *h0 } else { *h1 }, CalibSource::Interpolated));
                };
                let q: Vec<Point2> = (0..4).map(|k| Point2::new((1.0 - t) * q0[k].x + t * q1[k].x, (1.0 - t) * q0[k].y + t * q1[k].y)).collect();
                let model = vec![
                    Point2::new(0.0, 0.0),
                    Point2::new(self.field_len, 0.0),
                    Point2::new(self.field_len, self.field_w),
                    Point2::new(0.0, self.field_w),
                ];
                match dlt(&q, &model) {
                    Some(h) => Some((h, CalibSource::Interpolated)),
                    None => Some((if t < 0.5 { *h0 } else { *h1 }, CalibSource::Interpolated)),
                }
            }
            (Some((&i0, h0)), None) => {
                // Newest frames on a live feed: hold the last solve briefly.
                if frame - i0 <= edge { Some((*h0, CalibSource::Tracked)) } else { None }
            }
            (None, Some((&i1, h1))) => {
                if i1 - frame <= edge { Some((*h1, CalibSource::Tracked)) } else { None }
            }
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn interpolates_between_pans() {
        let f2i_a = Matrix3::new(9.0, -2.0, 120.0, 0.5, 6.0, 80.0, 0.0002, 0.004, 1.0);
        let f2i_b = Matrix3::new(9.0, -2.0, 220.0, 0.5, 6.0, 80.0, 0.0002, 0.004, 1.0); // panned 100px
        let mut t = KeyframeTrack::new(105.0, 68.0, 30);
        t.add(0, f2i_a.try_inverse().unwrap());
        t.add(20, f2i_b.try_inverse().unwrap());
        let (h, src) = t.h_at(10).unwrap();
        assert_eq!(src, CalibSource::Interpolated);
        // The field origin should land halfway between the two pans.
        let inv = h.try_inverse().unwrap();
        let p = apply(&inv, Point2::new(0.0, 0.0)).unwrap();
        assert!((p.x - 170.0).abs() < 1.0, "x={}", p.x);
        assert!(t.h_at(200).is_none());
        assert_eq!(t.h_at(25).unwrap().1, CalibSource::Tracked);
        assert_eq!(t.h_at(0).unwrap().1, CalibSource::Keyframe);
    }
}
