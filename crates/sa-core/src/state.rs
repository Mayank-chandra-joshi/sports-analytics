//! What one processed frame produces. `FrameState` is the single message
//! type that leaves the engine: the UI overlay, the session log and the
//! recorder all consume it. Small by construction — no pixels.

use std::sync::Arc;

use crate::geometry::{BBox, Point2};
use crate::profile::{Class, Team};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Detection {
    pub class: Class,
    pub bbox: BBox,
    pub conf: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost { frames: u32 },
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Track {
    pub id: u32,
    pub class: Class,
    pub bbox: BBox,
    pub conf: f32,
    pub state: TrackState,
    pub team: Team,
    /// Field position in metres, only when the frame is calibrated.
    pub pitch: Option<Point2>,
    /// Appearance vector, present when ReID ran for this track recently.
    #[serde(skip)]
    pub embedding: Option<Arc<[f32]>>,
    /// Display number for the UI — stable for the track's lifetime.
    pub label: u32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BallState {
    pub image: Point2,
    pub pitch: Option<Point2>,
    /// False when the position is interpolated across a gap.
    pub seen: bool,
    pub conf: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CalibSource {
    /// Solved on this exact frame by the keypoint model.
    Keyframe,
    /// Blended between two neighbouring keyframes.
    Interpolated,
    /// Carried by the camera tracker from an anchor frame.
    Tracked,
    /// Placed by the user.
    Manual,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// Image pixels (engine working resolution) -> field metres, row-major 3x3.
    pub h: [[f64; 3]; 3],
    pub confidence: f32,
    pub source: CalibSource,
    /// Fraction of the field model inside the frame.
    pub coverage: f32,
}

impl Calibration {
    pub fn project(&self, p: Point2) -> Option<Point2> {
        let h = &self.h;
        let x = p.x as f64;
        let y = p.y as f64;
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        if w.abs() < 1e-9 || !w.is_finite() {
            return None;
        }
        let px = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
        let py = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
        if px.is_finite() && py.is_finite() {
            Some(Point2::new(px as f32, py as f32))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TeamColours {
    pub a: [u8; 3],
    pub b: [u8; 3],
    pub referee: Option<[u8; 3]>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct StageLatency {
    pub decode_ms: f32,
    pub detect_ms: f32,
    pub track_ms: f32,
    pub identity_ms: f32,
    pub pitch_ms: f32,
    pub analytics_ms: f32,
    pub total_ms: f32,
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct LiveStats {
    /// Detections per second — the rate identity is actually resolved at.
    pub fps: f32,
    /// Source frames per second reaching the screen.
    pub display_fps: f32,
    /// One detection every N source frames; the rest are motion-predicted.
    pub detect_every: u32,
    pub dropped: u64,
    pub latency: StageLatency,
    pub possession_a: f32,
    pub possession_b: f32,
    pub distance_a_m: f32,
    pub distance_b_m: f32,
    pub passes_a: u32,
    pub passes_b: u32,
    pub ball_seen_rate: f32,
    /// Offside line x in metres for the attacking team, when computable.
    pub offside_x: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrameState {
    pub frame_id: u64,
    pub pts_ms: u64,
    pub width: u32,
    pub height: u32,
    pub tracks: Vec<Track>,
    pub ball: Option<BallState>,
    pub calibration: Option<Calibration>,
    pub teams: Option<TeamColours>,
    pub stats: LiveStats,
}
