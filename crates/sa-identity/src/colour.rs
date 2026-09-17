//! Kit colour reading and the Lab colour space. Ported from the POC's
//! `kit_reading` / `_lab`: clustering in Lab because Euclidean distance there
//! tracks perceived difference, which RGB does not.

use sa_core::{BBox, Frame};

/// sRGB (0-255) → CIE L*a*b* (OpenCV scaling: L 0-255, a/b 0-255 offset 128),
/// so thresholds tuned on the POC's cv2 output carry over unchanged.
pub fn rgb_to_lab(rgb: [u8; 3]) -> [f32; 3] {
    fn lin(c: u8) -> f32 {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.950456;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.088754;
    fn f(t: f32) -> f32 {
        if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
    }
    let (fx, fy, fz) = (f(x), f(y), f(z));
    let l = if y > 0.008856 { 116.0 * y.cbrt() - 16.0 } else { 903.3 * y };
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);
    [l * 255.0 / 100.0, a + 128.0, bb + 128.0]
}

#[inline]
pub fn lab_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Read a detection's shirt colour and how strongly it stands out from its
/// surroundings. `(rgb, contrast)`; contrast is `inf` side-on (the torso band
/// is nearly all player) and a background-relative ratio from overhead.
pub fn kit_reading(frame: &Frame, b: &BBox, overhead: f32) -> ([u8; 3], f32) {
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    let (x1, y1, x2, y2) = (b.x1.max(0.0), b.y1.max(0.0), b.x2.min(fw), b.y2.min(fh));
    if x2 <= x1 + 1.0 || y2 <= y1 + 1.0 {
        return ([200, 200, 200], 0.0);
    }
    if overhead >= 0.5 {
        return overhead_reading(frame, x1, y1, x2, y2);
    }
    // Side-on: the torso band — 18-48% down, middle half across.
    let h = y2 - y1;
    let w = x2 - x1;
    let cy1 = (y1 + 0.18 * h) as u32;
    let cy2 = ((y1 + 0.48 * h) as u32).max(cy1 + 1).min(frame.height);
    let cx1 = (x1 + 0.25 * w) as u32;
    let cx2 = ((x2 - 0.25 * w) as u32).max(cx1 + 1).min(frame.width);
    let mut acc = [0u64; 3];
    let mut n = 0u64;
    for y in cy1..cy2 {
        let row = frame.row(y);
        for x in cx1..cx2 {
            let o = x as usize * 3;
            acc[0] += row[o] as u64;
            acc[1] += row[o + 1] as u64;
            acc[2] += row[o + 2] as u64;
            n += 1;
        }
    }
    if n == 0 {
        return ([200, 200, 200], 0.0);
    }
    ([(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8], f32::INFINITY)
}

/// From nadir a box is mostly grass: the shirt is the 30% of pixels most
/// UNLIKE a collar sampled just outside the box.
fn overhead_reading(frame: &Frame, x1: f32, y1: f32, x2: f32, y2: f32) -> ([u8; 3], f32) {
    let (x1, y1, x2, y2) = (x1 as u32, y1 as u32, x2 as u32, y2 as u32);
    let pad_x = ((x2 - x1) as f32 * 0.2).max(2.0) as u32;
    let pad_y = ((y2 - y1) as f32 * 0.2).max(2.0) as u32;
    let (ox1, oy1) = (x1.saturating_sub(pad_x), y1.saturating_sub(pad_y));
    let (ox2, oy2) = ((x2 + pad_x).min(frame.width), (y2 + pad_y).min(frame.height));
    let mut bg: Vec<[f32; 3]> = Vec::new();
    for y in oy1..oy2 {
        let row = frame.row(y);
        for x in ox1..ox2 {
            if x >= x1 && x < x2 && y >= y1 && y < y2 {
                continue;
            }
            let o = x as usize * 3;
            bg.push([row[o] as f32, row[o + 1] as f32, row[o + 2] as f32]);
        }
    }
    let mut inner: Vec<[f32; 3]> = Vec::new();
    for y in y1..y2 {
        let row = frame.row(y);
        for x in x1..x2 {
            let o = x as usize * 3;
            inner.push([row[o] as f32, row[o + 1] as f32, row[o + 2] as f32]);
        }
    }
    if bg.len() < 8 || inner.is_empty() {
        let m = median3(&inner);
        return ([m[0] as u8, m[1] as u8, m[2] as u8], 0.0);
    }
    let bgm = median3(&bg);
    let mut d: Vec<(f32, usize)> = inner.iter().enumerate().map(|(i, p)| (dist3(*p, bgm), i)).collect();
    d.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = ((d.len() as f32 * 0.3) as usize).max(4).min(d.len());
    let top: Vec<[f32; 3]> = d[..n].iter().map(|(_, i)| inner[*i]).collect();
    let m = median3(&top);
    let spread = {
        let mut v: Vec<f32> = bg.iter().map(|p| dist3(*p, bgm)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] + 1e-6
    };
    ([m[0] as u8, m[1] as u8, m[2] as u8], dist3(m, bgm) / spread)
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn median3(v: &[[f32; 3]]) -> [f32; 3] {
    if v.is_empty() {
        return [0.0; 3];
    }
    let mut out = [0.0; 3];
    for c in 0..3 {
        let mut col: Vec<f32> = v.iter().map(|p| p[c]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out[c] = col[col.len() / 2];
    }
    out
}

/// Contrast a reading must have over the surrounding pitch to count as a
/// shirt from overhead (the POC measured 5.0 for the weakest player, 2.7
/// for the strongest non-player).
pub const KIT_MIN_CONTRAST: f32 = 3.0;

/// Is this detection standing on the playing surface?
///
/// A COCO detector finds people in the crowd, in the dugout and behind the
/// goal net, and every one of them becomes a track with a ring. They cannot
/// be told apart from players by size or confidence — only by WHERE they
/// are. The pitch is the dominant green region of a football frame, so a
/// detection whose feet sit on green is on the pitch and one whose feet sit
/// on a wall of spectators is not.
///
/// This is the cheap, no-calibration version of the POC's `player_kits`
/// bounds test (which projects the foot point through a homography). It
/// measures the ground UNDER the box rather than assuming any colour: the
/// reference is the frame's own dominant hue, so it follows floodlight,
/// shadow and any grass shade — and an indoor or artificial surface simply
/// makes every detection agree with it, which fails open rather than
/// silently rejecting everybody.
pub fn on_playing_surface(frame: &Frame, b: &BBox, surface: [f32; 3], tol: f32) -> bool {
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    // A band just BELOW the feet: the ground the player stands on, not the
    // player. Clamped into the frame for a box at the bottom edge.
    let h = b.h().max(4.0);
    let y1 = (b.y2 - h * 0.06).clamp(0.0, fh - 2.0) as u32;
    let y2 = (b.y2 + h * 0.12).clamp(y1 as f32 + 1.0, fh) as u32;
    let x1 = (b.x1 + b.w() * 0.2).clamp(0.0, fw - 2.0) as u32;
    let x2 = (b.x2 - b.w() * 0.2).clamp(x1 as f32 + 1.0, fw) as u32;
    let mut px: Vec<[f32; 3]> = Vec::new();
    let mut y = y1;
    while y < y2 {
        let row = frame.row(y);
        let mut x = x1;
        while x < x2 {
            let o = x as usize * 3;
            px.push([row[o] as f32, row[o + 1] as f32, row[o + 2] as f32]);
            x += 2;
        }
        y += 1;
    }
    if px.len() < 8 {
        return true; // nothing to judge on: fail open
    }
    let m = median3(&px);
    lab_dist(rgb_to_lab([m[0] as u8, m[1] as u8, m[2] as u8]), surface) <= tol
}

/// The frame's dominant surface colour in Lab — the median over a coarse
/// sample of the lower two-thirds, where a football pitch is.
pub fn surface_colour(frame: &Frame) -> [f32; 3] {
    let mut px: Vec<[f32; 3]> = Vec::new();
    let y0 = frame.height / 3;
    let mut y = y0;
    while y < frame.height {
        let row = frame.row(y);
        let mut x = 0;
        while x < frame.width {
            let o = x as usize * 3;
            px.push([row[o] as f32, row[o + 1] as f32, row[o + 2] as f32]);
            x += 12;
        }
        y += 8;
    }
    let m = median3(&px);
    rgb_to_lab([m[0] as u8, m[1] as u8, m[2] as u8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip_sanity() {
        let white = rgb_to_lab([255, 255, 255]);
        assert!((white[0] - 255.0).abs() < 2.0);
        assert!((white[1] - 128.0).abs() < 2.0);
        let red = rgb_to_lab([255, 0, 0]);
        assert!(red[1] > 180.0, "red a*={}", red[1]);
        let d = lab_dist(rgb_to_lab([200, 0, 0]), rgb_to_lab([220, 10, 10]));
        let d2 = lab_dist(rgb_to_lab([200, 0, 0]), rgb_to_lab([0, 0, 200]));
        assert!(d < d2);
    }

    #[test]
    fn side_on_reading_is_torso_mean() {
        let (w, h) = (100u32, 200u32);
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 3) as usize;
                if (36..96).contains(&y) {
                    data[o] = 250;
                } else {
                    data[o + 1] = 250;
                }
            }
        }
        let f = Frame::new_rgb8(0, Default::default(), w, h, data);
        let (rgb, c) = kit_reading(&f, &BBox::new(0.0, 0.0, 100.0, 200.0), 0.0);
        assert_eq!(rgb, [250, 0, 0]);
        assert!(c.is_infinite());
    }
}
