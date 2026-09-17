//! Ball track with the POC's policy: gate on size and on distance from the
//! last known position (a ball cannot teleport), interpolate SHORT gaps and
//! flag them, leave long gaps empty. `possession` refuses interpolated
//! positions, so a gap can never manufacture a pass.

use sa_core::{BBox, Detection, Point2};

#[derive(Debug, Clone)]
pub struct BallParams {
    pub max_jump_px: f32,
    pub min_size_px: f32,
    pub max_size_px: f32,
    pub max_gap_frames: u32,
    /// Frames without a detection before the track is considered lost and
    /// the jump gate is released (the ball may have gone anywhere).
    pub reacquire_after: u32,
}

impl Default for BallParams {
    fn default() -> Self {
        Self { max_jump_px: 160.0, min_size_px: 3.0, max_size_px: 60.0, max_gap_frames: 12, reacquire_after: 25 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BallPoint {
    pub frame: u64,
    pub xy: Point2,
    pub seen: bool,
    pub conf: f32,
}

pub struct BallTracker {
    p: BallParams,
    last: Option<BallPoint>,
    lost: u32,
    seen_frames: u64,
    total_frames: u64,
    /// Grid cells (40 px) where a "ball" has sat without moving, and for how
    /// many consecutive frames. A penalty spot, a bald head in the crowd or a
    /// ball on the touchline scores as `sports ball` every frame and never
    /// moves; a ball in play does not stay in one cell for a second.
    static_cells: std::collections::HashMap<(i32, i32), u32>,
}

const STATIC_CELL_PX: f32 = 40.0;
const STATIC_AFTER_FRAMES: u32 = 20;

impl BallTracker {
    pub fn new(p: BallParams) -> Self {
        Self { p, last: None, lost: 0, seen_frames: 0, total_frames: 0, static_cells: Default::default() }
    }

    fn cell(d: &Detection) -> (i32, i32) {
        let c = d.bbox.center();
        ((c.x / STATIC_CELL_PX) as i32, (c.y / STATIC_CELL_PX) as i32)
    }

    /// Bump the counters of every cell seen this frame; decay the rest.
    fn update_static(&mut self, dets: &[Detection]) {
        let seen: std::collections::HashSet<(i32, i32)> = dets.iter().map(Self::cell).collect();
        self.static_cells.retain(|_, n| { *n = n.saturating_sub(1); *n > 0 });
        for c in seen {
            *self.static_cells.entry(c).or_insert(0) += 2;
        }
    }

    fn is_static(&self, d: &Detection) -> bool {
        self.static_cells.get(&Self::cell(d)).is_some_and(|n| *n >= STATIC_AFTER_FRAMES)
    }

    /// Feed this frame's ball detections; get the current ball state (seen,
    /// interpolated/held, or None).
    pub fn update(&mut self, frame: u64, dets: &[Detection]) -> Option<BallPoint> {
        self.total_frames += 1;
        self.update_static(dets);
        let mut best: Option<&Detection> = None;
        let mut best_score = f32::NEG_INFINITY;
        for d in dets {
            let s = d.bbox.w().max(d.bbox.h());
            if s < self.p.min_size_px || s > self.p.max_size_px {
                continue;
            }
            if self.is_static(d) {
                continue;
            }
            let mut score = d.conf;
            if let Some(l) = &self.last {
                if self.lost < self.p.reacquire_after {
                    let dist = d.bbox.center().dist(&l.xy);
                    if dist > self.p.max_jump_px * (1.0 + self.lost as f32 * 0.25) {
                        continue;
                    }
                    score -= dist / self.p.max_jump_px * 0.2;
                }
            }
            if score > best_score {
                best_score = score;
                best = Some(d);
            }
        }
        match best {
            Some(d) => {
                let pt = BallPoint { frame, xy: d.bbox.center(), seen: true, conf: d.conf };
                self.last = Some(pt);
                self.lost = 0;
                self.seen_frames += 1;
                Some(pt)
            }
            None => {
                self.lost += 1;
                match self.last {
                    Some(l) if self.lost <= self.p.max_gap_frames => {
                        Some(BallPoint { frame, xy: l.xy, seen: false, conf: 0.0 })
                    }
                    _ => None,
                }
            }
        }
    }

    pub fn detection_rate(&self) -> f32 {
        if self.total_frames == 0 { 0.0 } else { self.seen_frames as f32 / self.total_frames as f32 }
    }

    pub fn last_bbox_hint(&self) -> Option<BBox> {
        self.last.map(|l| BBox::from_cxcywh(l.xy.x, l.xy.y, 40.0, 40.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sa_core::Class;

    fn ball(x: f32, conf: f32) -> Detection {
        Detection { class: Class::Ball, bbox: BBox::new(x, 100.0, x + 10.0, 110.0), conf }
    }

    #[test]
    fn rejects_stationary_decoy_and_holds_gap() {
        let mut t = BallTracker::new(BallParams::default());
        // Known ball first, then a higher-confidence decoy far away: the jump
        // gate keeps the lock on the moving ball.
        t.update(0, &[ball(100.0, 0.6)]).unwrap();
        for i in 1..5 {
            let real = ball(100.0 + 8.0 * i as f32, 0.6);
            let decoy = ball(600.0, 0.9);
            let s = t.update(i, &[real, decoy]).unwrap();
            assert!(s.seen && s.xy.x < 200.0);
        }
        let s = t.update(5, &[]).unwrap();
        assert!(!s.seen);
        for f in 6..40 {
            let _ = t.update(f, &[]);
        }
        assert!(t.update(40, &[]).is_none());
        assert!((t.detection_rate() - 5.0 / 41.0).abs() < 1e-3);
    }

    #[test]
    fn static_decoy_is_suppressed_from_cold() {
        let mut t = BallTracker::new(BallParams::default());
        // A penalty spot scores every frame from the start. After a second it
        // is recognised as static, and the real ball wins from then on even
        // though the decoy's confidence is higher.
        for f in 0..30 {
            let _ = t.update(f, &[ball(600.0, 0.9)]);
        }
        let s = t.update(31, &[ball(600.0, 0.9), ball(100.0, 0.5)]);
        // Jump gate from the (wrong) lock at 600 is released only after
        // `reacquire_after`; until then neither is accepted, which is honest.
        assert!(s.map_or(true, |p| !p.seen || p.xy.x < 200.0));
        for f in 32..70 {
            let _ = t.update(f, &[ball(600.0, 0.9)]);
        }
        let s = t.update(70, &[ball(600.0, 0.9), ball(100.0, 0.5)]).unwrap();
        assert!(s.seen && s.xy.x < 200.0);
    }
}
