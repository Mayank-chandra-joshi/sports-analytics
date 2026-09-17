//! Homography estimation: normalised DLT, and RANSAC over point
//! correspondences. Pure `nalgebra`; no OpenCV.
//!
//! Convention throughout the engine: `H` maps IMAGE pixels → FIELD metres.

use nalgebra::{DMatrix, Matrix3, Vector3};
use sa_core::Point2;

pub type H = Matrix3<f64>;

/// Apply `h` to a point. `None` when the point maps to infinity.
pub fn apply(h: &H, p: Point2) -> Option<Point2> {
    let v = h * Vector3::new(p.x as f64, p.y as f64, 1.0);
    if v.z.abs() < 1e-12 || !v.z.is_finite() {
        return None;
    }
    let (x, y) = (v.x / v.z, v.y / v.z);
    if x.is_finite() && y.is_finite() {
        Some(Point2::new(x as f32, y as f32))
    } else {
        None
    }
}

pub fn to_array(h: &H) -> [[f64; 3]; 3] {
    let mut a = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            a[r][c] = h[(r, c)];
        }
    }
    a
}

pub fn from_array(a: &[[f64; 3]; 3]) -> H {
    Matrix3::from_fn(|r, c| a[r][c])
}

fn normalise(pts: &[Point2]) -> (Matrix3<f64>, Vec<(f64, f64)>) {
    let n = pts.len() as f64;
    let (mx, my) = pts.iter().fold((0.0, 0.0), |(sx, sy), p| (sx + p.x as f64, sy + p.y as f64));
    let (mx, my) = (mx / n, my / n);
    let mean_d = pts.iter().map(|p| ((p.x as f64 - mx).powi(2) + (p.y as f64 - my).powi(2)).sqrt()).sum::<f64>() / n;
    let s = if mean_d > 1e-12 { 2f64.sqrt() / mean_d } else { 1.0 };
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    let out = pts.iter().map(|p| (s * (p.x as f64 - mx), s * (p.y as f64 - my))).collect();
    (t, out)
}

/// Direct linear transform from ≥4 correspondences `src → dst`.
pub fn dlt(src: &[Point2], dst: &[Point2]) -> Option<H> {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return None;
    }
    let (ts, s) = normalise(src);
    let (td, d) = normalise(dst);
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let (x, y) = s[i];
        let (u, v) = d[i];
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }
    // Null vector of A = eigenvector of AᵀA with the smallest eigenvalue.
    // (A thin SVD of an 8×9 system does not return the 9th singular vector,
    // which is exactly the one we need for the minimal 4-point case.)
    let ata = a.transpose() * &a;
    let ata9: nalgebra::SMatrix<f64, 9, 9> = nalgebra::SMatrix::<f64, 9, 9>::from_iterator(ata.iter().copied());
    let eig = nalgebra::SymmetricEigen::new(ata9);
    let (mut best, mut best_val) = (0usize, f64::INFINITY);
    for (i, v) in eig.eigenvalues.iter().enumerate() {
        if *v < best_val {
            best_val = *v;
            best = i;
        }
    }
    let row = eig.eigenvectors.column(best);
    let hn = Matrix3::new(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]);
    let td_inv = td.try_inverse()?;
    let h = td_inv * hn * ts;
    if !h.iter().all(|v| v.is_finite()) {
        return None;
    }
    let s = h[(2, 2)];
    Some(if s.abs() > 1e-12 { h / s } else { h })
}

#[derive(Debug, Clone)]
pub struct RansacResult {
    pub h: H,
    pub inliers: Vec<usize>,
    /// Mean reprojection error of the inliers, in `dst` units.
    pub rms: f64,
}

/// A tiny deterministic PRNG so runs are reproducible without a dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n.max(1)
    }
}

/// RANSAC homography `src → dst`. `thr` is the reprojection threshold in
/// `dst` units (pass image→field and threshold in metres, or field→image and
/// threshold in pixels — the caller picks the space it can reason about).
pub fn ransac(src: &[Point2], dst: &[Point2], thr: f64, iters: usize, min_inliers: usize) -> Option<RansacResult> {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return None;
    }
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let mut best: Option<(Vec<usize>, H)> = None;
    let thr2 = thr * thr;
    let iters = if n == 4 { 1 } else { iters };
    for _ in 0..iters {
        // Four distinct indices.
        let mut idx = [0usize; 4];
        let mut k = 0;
        let mut guard = 0;
        while k < 4 && guard < 1000 {
            guard += 1;
            let c = rng.next(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        if k < 4 {
            break;
        }
        let s: Vec<Point2> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<Point2> = idx.iter().map(|&i| dst[i]).collect();
        let Some(h) = dlt(&s, &d) else { continue };
        let inl: Vec<usize> = (0..n)
            .filter(|&i| match apply(&h, src[i]) {
                Some(p) => {
                    let e = (p.x as f64 - dst[i].x as f64).powi(2) + (p.y as f64 - dst[i].y as f64).powi(2);
                    e <= thr2
                }
                None => false,
            })
            .collect();
        if best.as_ref().is_none_or(|(b, _)| inl.len() > b.len()) {
            best = Some((inl, h));
        }
    }
    let (inliers, _) = best?;
    if inliers.len() < min_inliers.max(4) {
        return None;
    }
    // Refit on all inliers.
    let s: Vec<Point2> = inliers.iter().map(|&i| src[i]).collect();
    let d: Vec<Point2> = inliers.iter().map(|&i| dst[i]).collect();
    let h = dlt(&s, &d)?;
    let mut se = 0.0;
    for &i in &inliers {
        if let Some(p) = apply(&h, src[i]) {
            se += (p.x as f64 - dst[i].x as f64).powi(2) + (p.y as f64 - dst[i].y as f64).powi(2);
        }
    }
    let rms = (se / inliers.len() as f64).sqrt();
    Some(RansacResult { h, inliers, rms })
}

/// Reject a homography that no real camera could produce: the field's
/// corners must map to finite image points, in a convex order, with the
/// pitch not folded over itself.
pub fn plausible(h_img_to_field: &H, field_len: f32, field_w: f32) -> bool {
    let Some(inv) = h_img_to_field.try_inverse() else { return false };
    let corners = [Point2::new(0.0, 0.0), Point2::new(field_len, 0.0), Point2::new(field_len, field_w), Point2::new(0.0, field_w)];
    let mut img = Vec::with_capacity(4);
    for c in corners {
        match apply(&inv, c) {
            Some(p) if p.x.abs() < 1e6 && p.y.abs() < 1e6 => img.push(p),
            _ => return false,
        }
    }
    // Consistent winding (all cross products same sign) ⇒ convex quad.
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = img[i];
        let b = img[(i + 1) % 4];
        let c = img[(i + 2) % 4];
        let cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if sign == 0.0 {
            sign = cr.signum();
        } else if cr.signum() != sign && cr != 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth() -> (H, Vec<Point2>, Vec<Point2>) {
        // A believable broadcast-like projection: field → image.
        let h_f2i = Matrix3::new(9.0, -2.0, 120.0, 0.5, 6.0, 80.0, 0.0002, 0.004, 1.0);
        let h = h_f2i.try_inverse().unwrap();
        let field: Vec<Point2> = (0..8)
            .flat_map(|i| (0..5).map(move |j| Point2::new(i as f32 * 15.0, j as f32 * 17.0)))
            .collect();
        let img: Vec<Point2> = field.iter().map(|p| apply(&h_f2i, *p).unwrap()).collect();
        (h, img, field)
    }

    #[test]
    fn dlt_recovers_exact() {
        let (h_true, img, field) = synth();
        let h = dlt(&img, &field).unwrap();
        for (i, f) in img.iter().zip(&field) {
            let p = apply(&h, *i).unwrap();
            assert!(p.dist(f) < 1e-2, "{:?} vs {:?}", p, f);
        }
        let _ = h_true;
    }

    #[test]
    fn ransac_survives_outliers() {
        let (_, mut img, field) = synth();
        // Corrupt 30% of the image points.
        for k in (0..img.len()).step_by(3) {
            img[k].x += 200.0;
        }
        let r = ransac(&img, &field, 0.5, 300, 8).unwrap();
        assert!(r.inliers.len() >= 26, "inliers {}", r.inliers.len());
        assert!(r.rms < 0.1);
        assert!(plausible(&r.h, 105.0, 68.0));
    }
}
