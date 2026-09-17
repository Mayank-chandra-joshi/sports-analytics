//! Constant-velocity Kalman filter on `(cx, cy, w, h, vx, vy, vw, vh)` — the
//! BoT-SORT state, which tracks box size as well as position so a player
//! running toward the camera is predicted at the right scale.

use nalgebra::{SMatrix, SVector};

type State = SVector<f32, 8>;
type Cov = SMatrix<f32, 8, 8>;
type Meas = SVector<f32, 4>;

#[derive(Debug, Clone)]
pub struct BoxKalman {
    pub x: State,
    pub p: Cov,
    f: SMatrix<f32, 8, 8>,
    h: SMatrix<f32, 4, 8>,
    std_pos: f32,
    std_vel: f32,
}

impl BoxKalman {
    pub fn new(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let mut f = SMatrix::<f32, 8, 8>::identity();
        for i in 0..4 {
            f[(i, i + 4)] = 1.0;
        }
        let mut hm = SMatrix::<f32, 4, 8>::zeros();
        for i in 0..4 {
            hm[(i, i)] = 1.0;
        }
        let std_pos = 1.0 / 20.0;
        let std_vel = 1.0 / 160.0;
        let mut p = Cov::zeros();
        let s = [2.0 * std_pos * h, 2.0 * std_pos * h, 2.0 * std_pos * w, 2.0 * std_pos * h,
                 10.0 * std_vel * h, 10.0 * std_vel * h, 10.0 * std_vel * w, 10.0 * std_vel * h];
        for i in 0..8 {
            p[(i, i)] = s[i] * s[i];
        }
        Self { x: State::from([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0]), p, f, h: hm, std_pos, std_vel }
    }

    pub fn predict(&mut self) {
        let (w, h) = (self.x[2].max(1.0), self.x[3].max(1.0));
        let s = [self.std_pos * h, self.std_pos * h, self.std_pos * w, self.std_pos * h,
                 self.std_vel * h, self.std_vel * h, self.std_vel * w, self.std_vel * h];
        let mut q = Cov::zeros();
        for i in 0..8 {
            q[(i, i)] = s[i] * s[i];
        }
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + q;
    }

    pub fn update(&mut self, cx: f32, cy: f32, w: f32, h: f32) {
        let hh = self.x[3].max(1.0);
        let ww = self.x[2].max(1.0);
        let s = [self.std_pos * hh, self.std_pos * hh, self.std_pos * ww, self.std_pos * hh];
        let mut r = SMatrix::<f32, 4, 4>::zeros();
        for i in 0..4 {
            r[(i, i)] = s[i] * s[i];
        }
        let z = Meas::from([cx, cy, w, h]);
        let y = z - self.h * self.x;
        let sm = self.h * self.p * self.h.transpose() + r;
        let Some(si) = sm.try_inverse() else { return };
        let k = self.p * self.h.transpose() * si;
        self.x += k * y;
        let i8 = Cov::identity();
        self.p = (i8 - k * self.h) * self.p;
    }

    /// Shift the predicted position by a camera translation (pixels).
    pub fn shift(&mut self, dx: f32, dy: f32) {
        self.x[0] += dx;
        self.x[1] += dy;
    }

    pub fn bbox(&self) -> sa_core::BBox {
        sa_core::BBox::from_cxcywh(self.x[0], self.x[1], self.x[2].max(1.0), self.x[3].max(1.0))
    }

    pub fn velocity(&self) -> (f32, f32) {
        (self.x[4], self.x[5])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_constant_velocity() {
        let mut k = BoxKalman::new(100.0, 100.0, 20.0, 40.0);
        for i in 1..=20 {
            k.predict();
            k.update(100.0 + 5.0 * i as f32, 100.0, 20.0, 40.0);
        }
        k.predict();
        let b = k.bbox().center();
        assert!((b.x - 205.0).abs() < 3.0, "x={}", b.x);
        assert!((k.velocity().0 - 5.0).abs() < 1.0);
    }
}
