//! A decoded video frame. Pixels live in an `Arc` so every stage shares one
//! buffer; nothing downstream of ingest copies image data.

use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PixelFormat {
    /// Packed 8-bit RGB, row-major, no padding. The engine's working format.
    Rgb8,
}

#[derive(Debug, Clone)]
pub struct Frame {
    /// Monotonic per source. Gaps mean frames were dropped under load.
    pub id: u64,
    /// Presentation timestamp from the source (file pts or wall clock for live).
    pub pts: Duration,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Arc<[u8]>,
}

impl Frame {
    pub fn new_rgb8(id: u64, pts: Duration, width: u32, height: u32, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), (width * height * 3) as usize);
        Self { id, pts, width, height, format: PixelFormat::Rgb8, data: Arc::from(data) }
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.width as usize * 3
    }

    /// Row-major view of one pixel row.
    #[inline]
    pub fn row(&self, y: u32) -> &[u8] {
        let s = self.stride();
        let o = y as usize * s;
        &self.data[o..o + s]
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let o = y as usize * self.stride() + x as usize * 3;
        [self.data[o], self.data[o + 1], self.data[o + 2]]
    }

    /// Copy of the region `bbox` (clamped to the frame). Used for identity
    /// crops; small and rare compared with the frame itself.
    pub fn crop(&self, bbox: &crate::BBox) -> Option<(u32, u32, Vec<u8>)> {
        let x1 = bbox.x1.max(0.0).floor() as u32;
        let y1 = bbox.y1.max(0.0).floor() as u32;
        let x2 = (bbox.x2.ceil() as u32).min(self.width);
        let y2 = (bbox.y2.ceil() as u32).min(self.height);
        if x2 <= x1 || y2 <= y1 {
            return None;
        }
        let (w, h) = (x2 - x1, y2 - y1);
        let mut out = Vec::with_capacity((w * h * 3) as usize);
        for y in y1..y2 {
            let r = self.row(y);
            out.extend_from_slice(&r[x1 as usize * 3..x2 as usize * 3]);
        }
        Some((w, h, out))
    }
}
