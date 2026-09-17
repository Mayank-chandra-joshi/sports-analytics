//! Drawing without a dependency: an RGB8 canvas with lines, rects, ellipses
//! and a 5×7 bitmap font. Enough for boxes, ground rings, ids and the pad.
//! The desktop UI draws its own overlays on a canvas; this exists for the
//! recorder, the CLI and the MJPEG preview.

use sa_core::{FrameState, Point2, Team, TrackState};

pub struct Canvas<'a> {
    pub w: u32,
    pub h: u32,
    pub data: &'a mut [u8],
}

pub type Rgb = [u8; 3];
pub const GREEN: Rgb = [0, 220, 0];
pub const ORANGE: Rgb = [255, 165, 0];
pub const WHITE: Rgb = [235, 235, 235];
pub const GREY: Rgb = [150, 150, 150];
pub const YELLOW: Rgb = [255, 230, 0];
pub const BLUE: Rgb = [40, 120, 255];

impl<'a> Canvas<'a> {
    pub fn new(w: u32, h: u32, data: &'a mut [u8]) -> Self {
        debug_assert_eq!(data.len(), (w * h * 3) as usize);
        Self { w, h, data }
    }

    #[inline]
    pub fn put(&mut self, x: i32, y: i32, c: Rgb) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let o = (y as usize * self.w as usize + x as usize) * 3;
        self.data[o] = c[0];
        self.data[o + 1] = c[1];
        self.data[o + 2] = c[2];
    }

    #[inline]
    pub fn blend(&mut self, x: i32, y: i32, c: Rgb, a: f32) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let o = (y as usize * self.w as usize + x as usize) * 3;
        // `k` walks the colour channel, indexing BOTH the source colour and
        // the destination pixel — iterating one of them would still need the
        // index for the other.
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            let v = self.data[o + k] as f32;
            self.data[o + k] = (v + (c[k] as f32 - v) * a) as u8;
        }
    }

    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Rgb, thick: i32) {
        let (mut x0, mut y0, x1, y1) = (x0.round() as i32, y0.round() as i32, x1.round() as i32, y1.round() as i32);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let r = (thick / 2).max(0);
        loop {
            for oy in -r..=r {
                for ox in -r..=r {
                    self.put(x0 + ox, y0 + oy, c);
                }
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    pub fn rect(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, c: Rgb, thick: i32) {
        self.line(x1, y1, x2, y1, c, thick);
        self.line(x2, y1, x2, y2, c, thick);
        self.line(x2, y2, x1, y2, c, thick);
        self.line(x1, y2, x1, y1, c, thick);
    }

    pub fn fill_rect(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, c: Rgb, alpha: f32) {
        for y in y1.max(0)..y2.min(self.h as i32) {
            for x in x1.max(0)..x2.min(self.w as i32) {
                self.blend(x, y, c, alpha);
            }
        }
    }

    /// Ellipse outline (rx, ry radii), used for ground rings.
    pub fn ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, c: Rgb, thick: i32) {
        let n = ((rx + ry) * 1.5).clamp(16.0, 120.0) as usize;
        let mut prev = (cx + rx, cy);
        for i in 1..=n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            let p = (cx + rx * a.cos(), cy + ry * a.sin());
            self.line(prev.0, prev.1, p.0, p.1, c, thick);
            prev = p;
        }
    }

    pub fn fill_ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32, c: Rgb, alpha: f32) {
        let (x0, x1) = ((cx - rx).floor() as i32, (cx + rx).ceil() as i32);
        let (y0, y1) = ((cy - ry).floor() as i32, (cy + ry).ceil() as i32);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = (x as f32 - cx) / rx.max(0.5);
                let dy = (y as f32 - cy) / ry.max(0.5);
                if dx * dx + dy * dy <= 1.0 {
                    self.blend(x, y, c, alpha);
                }
            }
        }
    }

    pub fn text(&mut self, x: i32, y: i32, s: &str, c: Rgb, scale: i32) {
        let mut cx = x;
        for ch in s.chars() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.put(cx + col * scale + sx, y + row as i32 * scale + sy, c);
                            }
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
}

/// 5×7 glyphs for digits, upper-case letters and a few symbols.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '#' => [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
        _ => [0; 7],
    }
}

pub fn team_colour(team: Team, teams: Option<&sa_core::state::TeamColours>) -> Rgb {
    match (team, teams) {
        (Team::A, Some(t)) => t.a,
        (Team::B, Some(t)) => t.b,
        (Team::Referee, Some(t)) => t.referee.unwrap_or(YELLOW),
        (Team::Unknown, _) => GREY,
        _ => BLUE,
    }
}

/// Draw tracks (ground ring + label), the ball, and the stats line onto an
/// RGB frame in place.
pub fn draw_overlays(canvas: &mut Canvas<'_>, st: &FrameState) {
    for t in &st.tracks {
        let b = &t.bbox;
        // A lost track is a PREDICTION, not a sighting, and this pipeline
        // detects a few times a second — so one missed detection already
        // means ~150 ms of guessing, over which a running player clears his
        // own width. Draw nothing rather than a ring beside him: an absent
        // marker reads as "briefly lost", a misplaced one reads as "wrong".
        let (c, alpha) = match t.state {
            TrackState::Lost { .. } => continue,
            _ => (team_colour(t.team, st.teams.as_ref()), 0.25),
        };
        let rx = (b.w() * 0.55).clamp(6.0, 60.0);
        let ry = rx * 0.35;
        if t.class.is_person() {
            canvas.fill_ellipse(b.center().x, b.y2, rx, ry, c, alpha);
            canvas.ellipse(b.center().x, b.y2, rx, ry, c, 2);
            canvas.text((b.center().x - 5.0) as i32, (b.y2 + ry + 3.0) as i32, &t.label.to_string(), WHITE, 2);
        } else {
            canvas.rect(b.x1, b.y1, b.x2, b.y2, WHITE, 1);
        }
    }
    if let Some(ball) = &st.ball {
        let c = if ball.seen { WHITE } else { ORANGE };
        canvas.ellipse(ball.image.x, ball.image.y, 7.0, 7.0, c, 2);
    }
    let s = &st.stats;
    let line = format!(
        "FPS {:.0}  LAT {:.0}MS  DET {:.0}  A {:.0}% B {:.0}%",
        s.fps, s.latency.total_ms, s.latency.detect_ms, s.possession_a * 100.0, s.possession_b * 100.0
    );
    canvas.fill_rect(6, 6, 6 + line.len() as i32 * 12 + 6, 28, [0, 0, 0], 0.55);
    canvas.text(10, 10, &line, WHITE, 2);
}

/// Render the 2D tactical pad: a top-down pitch with player dots. `width`
/// pixels wide; height follows the field aspect.
pub fn render_pad(st: &FrameState, field_len: f32, field_w: f32, width: u32) -> (u32, u32, Vec<u8>) {
    let margin = 10.0f32;
    let scale = (width as f32 - 2.0 * margin) / field_len;
    let h = (field_w * scale + 2.0 * margin).round() as u32;
    let mut data = vec![0u8; (width * h * 3) as usize];
    let mut c = Canvas::new(width, h, &mut data);
    c.fill_rect(0, 0, width as i32, h as i32, [58, 122, 58], 1.0);
    let px = |x: f32, y: f32| (margin + x * scale, margin + y * scale);
    let (x0, y0) = px(0.0, 0.0);
    let (x1, y1) = px(field_len, field_w);
    c.rect(x0, y0, x1, y1, WHITE, 1);
    let (hx, _) = px(field_len / 2.0, 0.0);
    c.line(hx, y0, hx, y1, WHITE, 1);
    let (cx, cy) = px(field_len / 2.0, field_w / 2.0);
    c.ellipse(cx, cy, 9.15 * scale, 9.15 * scale, WHITE, 1);
    for (xe, dir) in [(0.0f32, 1.0f32), (field_len, -1.0)] {
        for (d, w) in [(16.5f32, 40.32f32), (5.5, 18.32)] {
            let (ax, ay) = px(xe, (field_w - w) / 2.0);
            let (bx, by) = px(xe + dir * d, (field_w + w) / 2.0);
            c.rect(ax, ay, bx, by, WHITE, 1);
        }
    }
    match &st.calibration {
        None => {
            c.text((width / 2 - 60) as i32, (h / 2 - 6) as i32, "NOT CALIBRATED", WHITE, 2);
        }
        Some(cal) => {
            for t in &st.tracks {
                let Some(p) = t.pitch else { continue };
                if !t.class.is_person() {
                    continue;
                }
                let (x, y) = px(p.x.clamp(-2.0, field_len + 2.0), p.y.clamp(-2.0, field_w + 2.0));
                let col = team_colour(t.team, st.teams.as_ref());
                c.fill_ellipse(x, y, 5.0, 5.0, col, 1.0);
                c.ellipse(x, y, 5.0, 5.0, [20, 20, 20], 1);
            }
            if let Some(b) = &st.ball {
                if let Some(p) = b.pitch {
                    let (x, y) = px(p.x, p.y);
                    c.fill_ellipse(x, y, 3.5, 3.5, WHITE, 1.0);
                }
            }
            if let Some(ox) = st.stats.offside_x {
                let (lx, _) = px(ox, 0.0);
                c.line(lx, y0, lx, y1, YELLOW, 1);
            }
            let note = format!("{:?} {:.0}%", cal.source, cal.coverage * 100.0);
            c.text(margin as i32, (h as f32 - margin - 8.0) as i32, &note, WHITE, 1);
        }
    }
    (width, h, data)
}

/// Blend `pad` into the bottom-right corner of `canvas`.
pub fn composite_pad(canvas: &mut Canvas<'_>, pad: (u32, u32, &[u8]), alpha: f32, margin: i32) {
    let (pw, ph, pd) = pad;
    let x0 = canvas.w as i32 - pw as i32 - margin;
    let y0 = canvas.h as i32 - ph as i32 - margin;
    for y in 0..ph as i32 {
        for x in 0..pw as i32 {
            let o = ((y as u32 * pw + x as u32) * 3) as usize;
            canvas.blend(x0 + x, y0 + y, [pd[o], pd[o + 1], pd[o + 2]], alpha);
        }
    }
    canvas.rect(x0 as f32, y0 as f32, (x0 + pw as i32) as f32, (y0 + ph as i32) as f32, WHITE, 1);
}

/// Downscale (if needed) and JPEG-encode an RGB frame for the preview stream.
pub fn encode_jpeg(w: u32, h: u32, rgb: &[u8], max_width: u32, quality: u8) -> Vec<u8> {
    use fast_image_resize::images::{Image, ImageRef};
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let (ow, oh, buf): (u32, u32, std::borrow::Cow<[u8]>) = if max_width > 0 && w > max_width {
        let s = max_width as f32 / w as f32;
        let nh = ((h as f32 * s).round() as u32).max(2) & !1;
        let src = ImageRef::new(w, h, rgb, PixelType::U8x3).expect("frame buffer");
        let mut dst = Image::new(max_width, nh, PixelType::U8x3);
        let mut r = Resizer::new();
        let _ = r.resize(&src, &mut dst, &ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear)));
        (max_width, nh, std::borrow::Cow::Owned(dst.into_vec()))
    } else {
        (w, h, std::borrow::Cow::Borrowed(rgb))
    };
    let mut out = Vec::with_capacity((ow * oh) as usize / 4);
    let enc = jpeg_encoder::Encoder::new(&mut out, quality);
    let _ = enc.encode(&buf, ow as u16, oh as u16, jpeg_encoder::ColorType::Rgb);
    out
}

/// Foot point in field metres for a track, exposed so the UI and the pad agree.
pub fn pad_point(p: Point2) -> Point2 {
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_without_panicking() {
        let (w, h) = (320u32, 180u32);
        let mut data = vec![0u8; (w * h * 3) as usize];
        let mut c = Canvas::new(w, h, &mut data);
        c.line(-10.0, -10.0, 400.0, 200.0, GREEN, 3);
        c.ellipse(160.0, 90.0, 50.0, 20.0, WHITE, 2);
        c.text(5, 5, "FPS 25 A 50%", WHITE, 2);
        let st = FrameState { frame_id: 0, pts_ms: 0, width: w, height: h, tracks: vec![], ball: None, calibration: None, teams: None, stats: Default::default() };
        draw_overlays(&mut c, &st);
        let (pw, ph, pd) = render_pad(&st, 105.0, 68.0, 200);
        composite_pad(&mut c, (pw, ph, &pd), 0.8, 8);
        let jpg = encode_jpeg(w, h, &data, 160, 70);
        assert!(jpg.len() > 100 && jpg[0] == 0xFF && jpg[1] == 0xD8);
    }
}
