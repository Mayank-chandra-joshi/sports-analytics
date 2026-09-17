//! The multi-sport seam. Football is the first `SportProfile`; a second sport
//! supplies a new `FieldModel`, class map and tracker parameters and the stage
//! graph is untouched.

use crate::geometry::Point2;

/// What the detector saw. `Other` carries the raw class id for anything the
/// profile does not map, so nothing is silently dropped at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Class {
    Player,
    Goalkeeper,
    Referee,
    Ball,
    Other(u16),
}

impl Class {
    pub fn is_person(&self) -> bool {
        matches!(self, Class::Player | Class::Goalkeeper | Class::Referee)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Team {
    A,
    B,
    Referee,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FieldDims {
    /// Along the x axis, metres (football: 105).
    pub length: f32,
    /// Along the y axis, metres (football: 68).
    pub width: f32,
}

/// A painted straight line on the field, metres.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    pub a: Point2,
    pub b: Point2,
}

/// Stable identifier for a landmark the keypoint model can predict. The
/// number is the model's output index; the name is for the UI and presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LandmarkId(pub u16);

/// Everything geometric a sport's playing surface provides. Metres, origin at
/// one corner, x along the length, y along the width.
pub trait FieldModel: Send + Sync {
    fn dims(&self) -> FieldDims;
    /// Straight painted lines. Curves are sampled into short segments.
    fn segments(&self) -> Vec<Segment>;
    /// Named landmarks and their field coordinates, in keypoint-model order.
    fn landmarks(&self) -> Vec<(LandmarkId, &'static str, Point2)>;
    /// Is a field-space point within `margin_m` of the playing surface?
    fn on_field(&self, p: Point2, margin_m: f32) -> bool {
        let d = self.dims();
        p.x >= -margin_m && p.x <= d.length + margin_m && p.y >= -margin_m && p.y <= d.width + margin_m
    }
}

/// Maps a detector's raw class ids to the engine's `Class`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassMap {
    pub entries: Vec<(u16, Class)>,
}

impl ClassMap {
    /// COCO: person=0, sports ball=32. What the untuned yolo11 weights emit.
    pub fn coco() -> Self {
        Self { entries: vec![(0, Class::Player), (32, Class::Ball)] }
    }
    /// Roboflow football-players-detection: ball=0, goalkeeper=1, player=2, referee=3.
    pub fn roboflow_football() -> Self {
        Self { entries: vec![(0, Class::Ball), (1, Class::Goalkeeper), (2, Class::Player), (3, Class::Referee)] }
    }
    pub fn map(&self, raw: u16) -> Option<Class> {
        self.entries.iter().find(|(r, _)| *r == raw).map(|(_, c)| *c)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrackerParams {
    /// Detections at or above this confidence start/extend tracks (first pass).
    pub high_thresh: f32,
    /// Detections between this and `high_thresh` may only extend existing tracks (ByteTrack second pass).
    pub low_thresh: f32,
    /// Minimum IoU for an association.
    pub match_iou: f32,
    /// Frames a lost track is kept before deletion.
    pub track_buffer: u32,
    /// Confirmations needed before a track is reported.
    pub min_hits: u32,
    /// Physical bound used by re-acquire gates (metres per second).
    pub max_speed_mps: f32,
    /// Weight of appearance (cosine) vs IoU when embeddings are available.
    pub appearance_weight: f32,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            high_thresh: 0.45,
            low_thresh: 0.15,
            // A player moves further between two PROCESSED frames than
            // between two source frames whenever detection is the bottleneck,
            // so the boxes that should associate overlap less. 0.25 was tuned
            // against a 25 fps pipeline; at 3-6 fps it rejects real matches
            // and the ids churn — which is what produces #322 for a
            // fifteen-player scene. The Kalman prediction covers most of the
            // gap (see `Tracker::update_at`); this covers the rest.
            match_iou: 0.15,
            track_buffer: 60,
            // One detection is enough to draw a ring. `min_hits: 2` costs a
            // player two DETECTIONS to appear, which at one detection in
            // three is six source frames — long enough that on a moving
            // broadcast shot half the team never shows. A false positive
            // that lasts one frame is a far smaller sin than a real player
            // with no marker.
            min_hits: 1,
            max_speed_mps: 10.0,
            appearance_weight: 0.3,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TeamRule {
    /// Cluster kit colours into `n_teams`, with the odd-one-out kit as officials.
    KitColour { n_teams: u8, officials: bool },
    None,
}

/// One sport, fully described. Built once at startup from config.
pub struct SportProfile {
    pub name: &'static str,
    pub field: Box<dyn FieldModel>,
    pub classes: ClassMap,
    pub tracker: TrackerParams,
    pub team_rule: TeamRule,
    /// Real-world height of a standing player, the metre ruler for pixel gates.
    pub person_height_m: f32,
}
