//! Plain geometry used across stages. `f32` throughout: pixel and metre
//! precision never need more, and it halves the bandwidth of every state message.

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn dist(&self, o: &Point2) -> f32 {
        ((self.x - o.x).powi(2) + (self.y - o.y).powi(2)).sqrt()
    }
}

impl From<(f32, f32)> for Point2 {
    fn from((x, y): (f32, f32)) -> Self {
        Self { x, y }
    }
}

/// Axis-aligned box in pixels, `x1 <= x2`, `y1 <= y2`.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    pub fn from_cxcywh(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self { x1: cx - w / 2.0, y1: cy - h / 2.0, x2: cx + w / 2.0, y2: cy + h / 2.0 }
    }
    #[inline]
    pub fn w(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }
    #[inline]
    pub fn h(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }
    #[inline]
    pub fn area(&self) -> f32 {
        self.w() * self.h()
    }
    #[inline]
    pub fn center(&self) -> Point2 {
        Point2::new((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }
    /// Where the box meets the ground for a side-on camera: bottom centre.
    #[inline]
    pub fn foot(&self) -> Point2 {
        Point2::new((self.x1 + self.x2) * 0.5, self.y2)
    }
    pub fn iou(&self, o: &BBox) -> f32 {
        let ix1 = self.x1.max(o.x1);
        let iy1 = self.y1.max(o.y1);
        let ix2 = self.x2.min(o.x2);
        let iy2 = self.y2.min(o.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        if inter <= 0.0 {
            return 0.0;
        }
        let union = self.area() + o.area() - inter;
        if union > 0.0 {
            inter / union
        } else {
            0.0
        }
    }
    pub fn scaled(&self, s: f32) -> BBox {
        BBox::new(self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)
    }
    pub fn clamped(&self, w: f32, h: f32) -> BBox {
        BBox::new(self.x1.clamp(0.0, w), self.y1.clamp(0.0, h), self.x2.clamp(0.0, w), self.y2.clamp(0.0, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basic() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        assert!((a.iou(&b) - 25.0 / 175.0).abs() < 1e-6);
        assert_eq!(a.iou(&BBox::new(20.0, 20.0, 30.0, 30.0)), 0.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }
}
