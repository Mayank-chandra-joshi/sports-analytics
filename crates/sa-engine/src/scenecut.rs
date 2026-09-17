//! Shot-cut detection.
//!
//! Broadcast football cuts constantly — wide shot, close-up, replay, crowd.
//! Every track from the previous shot is meaningless in the next one, but
//! nothing in the tracker knows that: the boxes simply fail to associate and
//! then coast on motion, which paints rings across a close-up of somebody's
//! face. The POC hit the same failure and solved it the same way.
//!
//! Measured globally on a downscaled frame, so it costs microseconds: a
//! coarse RGB histogram, compared with the previous frame by correlation.
//! Deliberately conservative — it fires on abrupt whole-frame changes, so
//! continuous play (even a fast pan) is unaffected.

pub struct SceneCut {
    prev: Option<Vec<f32>>,
    /// Correlation below this between consecutive frames is a cut.
    min_corr: f32,
    bins: usize,
}

impl SceneCut {
    pub fn new(min_corr: f32) -> Self {
        Self { prev: None, min_corr, bins: 8 }
    }

    /// True when this frame begins a new shot.
    pub fn update(&mut self, frame: &sa_core::Frame) -> bool {
        let h = self.histogram(frame);
        let cut = match &self.prev {
            Some(p) => correlation(p, &h) < self.min_corr,
            None => false,
        };
        self.prev = Some(h);
        cut
    }

    /// Coarse 3D RGB histogram over a sparse pixel sample. Sampling every
    /// 16th pixel in each direction is ~4000 pixels of a 720p frame — far
    /// more than enough to tell one shot from another, and cheap enough to
    /// run on every frame.
    fn histogram(&self, frame: &sa_core::Frame) -> Vec<f32> {
        let b = self.bins;
        let mut h = vec![0.0f32; b * b * b];
        let step = 16u32;
        let mut n = 0.0f32;
        let mut y = 0;
        while y < frame.height {
            let row = frame.row(y);
            let mut x = 0;
            while x < frame.width {
                let o = x as usize * 3;
                let r = (row[o] as usize * b) >> 8;
                let g = (row[o + 1] as usize * b) >> 8;
                let bl = (row[o + 2] as usize * b) >> 8;
                h[(r * b + g) * b + bl] += 1.0;
                n += 1.0;
                x += step;
            }
            y += step;
        }
        if n > 0.0 {
            h.iter_mut().for_each(|v| *v /= n);
        }
        h
    }
}

/// Pearson correlation of two histograms, as OpenCV's HISTCMP_CORREL.
fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    let den = (da * db).sqrt();
    if den > 1e-12 { num / den } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sa_core::Frame;

    fn solid(id: u64, c: [u8; 3]) -> Frame {
        let (w, h) = (320u32, 180u32);
        let mut d = vec![0u8; (w * h * 3) as usize];
        for p in d.chunks_mut(3) {
            p.copy_from_slice(&c);
        }
        Frame::new_rgb8(id, Default::default(), w, h, d)
    }

    #[test]
    fn fires_on_a_cut_not_on_continuity() {
        let mut sc = SceneCut::new(0.6);
        assert!(!sc.update(&solid(0, [30, 120, 40])), "first frame is never a cut");
        assert!(!sc.update(&solid(1, [31, 121, 41])), "a near-identical frame is not a cut");
        assert!(sc.update(&solid(2, [200, 40, 180])), "a wholly different frame is a cut");
        assert!(!sc.update(&solid(3, [200, 40, 180])));
    }
}
