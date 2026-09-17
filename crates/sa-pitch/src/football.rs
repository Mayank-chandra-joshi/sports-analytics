//! The football field model. Dimensions are parameters (a school pitch is
//! not 105×68), and the 32 landmarks follow the Roboflow football-field
//! keypoint model's index order — the same table the POC's `pitchkp.py`
//! derives, expressed here as fractions of THIS pitch's dimensions so a
//! non-standard pitch resolves to the right metres or to nothing.

use sa_core::profile::{FieldDims, FieldModel, LandmarkId, Segment};
use sa_core::Point2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FootballDims {
    pub length: f32,
    pub width: f32,
    pub penalty_depth: f32,
    pub penalty_width: f32,
    pub goal_area_depth: f32,
    pub goal_area_width: f32,
    pub penalty_mark: f32,
    pub circle_r: f32,
}

impl FootballDims {
    pub const STANDARD: FootballDims = FootballDims {
        length: 105.0,
        width: 68.0,
        penalty_depth: 16.5,
        penalty_width: 40.32,
        goal_area_depth: 5.5,
        goal_area_width: 18.32,
        penalty_mark: 11.0,
        circle_r: 9.15,
    };

    /// Standard markings scaled to a different outer size.
    pub fn with_size(length: f32, width: f32) -> Self {
        Self { length, width, ..Self::STANDARD }
    }
}

pub struct Football {
    pub dims: FootballDims,
}

impl Football {
    pub fn new(dims: FootballDims) -> Self {
        Self { dims }
    }

    /// The 32 landmarks in metres, in keypoint-model order.
    pub fn template(&self) -> [(& 'static str, Point2); 32] {
        let d = self.dims;
        let (l, w) = (d.length, d.width);
        let (pw, pd) = (d.penalty_width, d.penalty_depth);
        let (gw, gd) = (d.goal_area_width, d.goal_area_depth);
        let (r, ps) = (d.circle_r, d.penalty_mark);
        let p = |x: f32, y: f32| Point2::new(x, y);
        [
            ("goal_left x touch_far", p(0.0, 0.0)),
            ("goal_left x penalty_far", p(0.0, (w - pw) / 2.0)),
            ("goal_left x goal_area_far", p(0.0, (w - gw) / 2.0)),
            ("goal_left x goal_area_near", p(0.0, (w + gw) / 2.0)),
            ("goal_left x penalty_near", p(0.0, (w + pw) / 2.0)),
            ("goal_left x touch_near", p(0.0, w)),
            ("goal_area_left x goal_area_far", p(gd, (w - gw) / 2.0)),
            ("goal_area_left x goal_area_near", p(gd, (w + gw) / 2.0)),
            ("penalty_mark_left", p(ps, w / 2.0)),
            ("penalty_left x penalty_far", p(pd, (w - pw) / 2.0)),
            ("penalty_left x goal_area_far", p(pd, (w - gw) / 2.0)),
            ("penalty_left x goal_area_near", p(pd, (w + gw) / 2.0)),
            ("penalty_left x penalty_near", p(pd, (w + pw) / 2.0)),
            ("halfway x touch_far", p(l / 2.0, 0.0)),
            ("halfway x circle_far", p(l / 2.0, w / 2.0 - r)),
            ("halfway x circle_near", p(l / 2.0, w / 2.0 + r)),
            ("halfway x touch_near", p(l / 2.0, w)),
            ("penalty_right x penalty_far", p(l - pd, (w - pw) / 2.0)),
            ("penalty_right x goal_area_far", p(l - pd, (w - gw) / 2.0)),
            ("penalty_right x goal_area_near", p(l - pd, (w + gw) / 2.0)),
            ("penalty_right x penalty_near", p(l - pd, (w + pw) / 2.0)),
            ("penalty_mark_right", p(l - ps, w / 2.0)),
            ("goal_area_right x goal_area_far", p(l - gd, (w - gw) / 2.0)),
            ("goal_area_right x goal_area_near", p(l - gd, (w + gw) / 2.0)),
            ("goal_right x touch_far", p(l, 0.0)),
            ("goal_right x penalty_far", p(l, (w - pw) / 2.0)),
            ("goal_right x goal_area_far", p(l, (w - gw) / 2.0)),
            ("goal_right x goal_area_near", p(l, (w + gw) / 2.0)),
            ("goal_right x penalty_near", p(l, (w + pw) / 2.0)),
            ("goal_right x touch_near", p(l, w)),
            ("circle_left", p(l / 2.0 - r, w / 2.0)),
            ("circle_right", p(l / 2.0 + r, w / 2.0)),
        ]
    }
}

impl FieldModel for Football {
    fn dims(&self) -> FieldDims {
        FieldDims { length: self.dims.length, width: self.dims.width }
    }

    fn segments(&self) -> Vec<Segment> {
        let d = self.dims;
        let (l, w) = (d.length, d.width);
        let seg = |x1: f32, y1: f32, x2: f32, y2: f32| Segment { a: Point2::new(x1, y1), b: Point2::new(x2, y2) };
        let mut v = vec![
            seg(0.0, 0.0, l, 0.0),
            seg(l, 0.0, l, w),
            seg(l, w, 0.0, w),
            seg(0.0, w, 0.0, 0.0),
            seg(l / 2.0, 0.0, l / 2.0, w),
        ];
        // Penalty areas and goal areas at both ends.
        for (x0, dir) in [(0.0f32, 1.0f32), (l, -1.0)] {
            for (depth, width) in [(d.penalty_depth, d.penalty_width), (d.goal_area_depth, d.goal_area_width)] {
                if depth <= 0.0 || width <= 0.0 {
                    continue;
                }
                let y1 = (w - width) / 2.0;
                let y2 = (w + width) / 2.0;
                let xi = x0 + dir * depth;
                v.push(seg(x0, y1, xi, y1));
                v.push(seg(xi, y1, xi, y2));
                v.push(seg(xi, y2, x0, y2));
            }
        }
        // Centre circle and penalty arcs sampled as polylines.
        if d.circle_r > 0.0 {
            let (cx, cy, r) = (l / 2.0, w / 2.0, d.circle_r);
            let n = 48;
            for i in 0..n {
                let a0 = (i as f32) / n as f32 * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32) / n as f32 * std::f32::consts::TAU;
                v.push(seg(cx + r * a0.cos(), cy + r * a0.sin(), cx + r * a1.cos(), cy + r * a1.sin()));
            }
            // Arcs: the part of a circle of radius r around the penalty mark
            // that lies outside the penalty area.
            if d.penalty_mark > 0.0 && d.penalty_depth > d.penalty_mark {
                let dx = d.penalty_depth - d.penalty_mark;
                if dx < r {
                    let half = (dx / r).acos();
                    for (mx, dir) in [(d.penalty_mark, 1.0f32), (l - d.penalty_mark, -1.0)] {
                        let m = 16;
                        for i in 0..m {
                            let a0 = -half + (i as f32) / m as f32 * 2.0 * half;
                            let a1 = -half + ((i + 1) as f32) / m as f32 * 2.0 * half;
                            v.push(seg(mx + dir * r * a0.cos(), cy + r * a0.sin(), mx + dir * r * a1.cos(), cy + r * a1.sin()));
                        }
                    }
                }
            }
        }
        v
    }

    fn landmarks(&self) -> Vec<(LandmarkId, &'static str, Point2)> {
        self.template().iter().enumerate().map(|(i, (n, p))| (LandmarkId(i as u16), *n, *p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_poc_order() {
        let f = Football::new(FootballDims::STANDARD);
        let t = f.template();
        assert_eq!(t[0].1, Point2::new(0.0, 0.0));
        assert_eq!(t[8].1, Point2::new(11.0, 34.0));
        assert_eq!(t[13].1, Point2::new(52.5, 0.0));
        assert_eq!(t[29].1, Point2::new(105.0, 68.0));
        assert!((t[30].1.x - (52.5 - 9.15)).abs() < 1e-4);
        assert_eq!(f.landmarks().len(), 32);
    }

    #[test]
    fn segments_present() {
        let f = Football::new(FootballDims::STANDARD);
        assert!(f.segments().len() > 60);
        assert!(f.on_field(Point2::new(50.0, 30.0), 0.0));
        assert!(!f.on_field(Point2::new(-5.0, 30.0), 2.0));
    }
}
