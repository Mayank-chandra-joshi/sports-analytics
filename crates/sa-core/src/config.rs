//! Engine configuration. Typed, serde-backed, and every field has a default
//! that runs. Thresholds and their names follow the reference POC's
//! `config.yaml` so tuning knowledge carries over.

use std::path::PathBuf;

use crate::profile::{TeamRule, TrackerParams};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Config {
    pub video: VideoConfig,
    pub detection: DetectionConfig,
    pub tracker: TrackerParams,
    pub reid: ReidConfig,
    pub pitch: PitchConfig,
    pub teams: TeamRule,
    pub analytics: AnalyticsConfig,
    pub output: OutputConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            video: VideoConfig::default(),
            detection: DetectionConfig::default(),
            tracker: TrackerParams::default(),
            reid: ReidConfig::default(),
            pitch: PitchConfig::default(),
            teams: TeamRule::KitColour { n_teams: 2, officials: true },
            analytics: AnalyticsConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    /// Engine working width. Frames wider than this are downscaled at ingest;
    /// all boxes, gates and calibrations live in this space.
    pub target_width: u32,
    /// Process one frame in every N for the HEAVY stages (detect + ReID),
    /// motion-predicting the rest. 1 = every frame; 0 = ADAPT to measured
    /// detector speed (the default). Raising it is what keeps a slow
    /// detector in step with the source: the alternative is ingest dropping
    /// frames at unpredictable moments, which costs the same and gives no
    /// control. Tracking degrades gracefully because the Kalman filter is
    /// stepped by the real elapsed frames either way.
    pub detect_every: u32,
    /// Bounded channel depth between stages. 2 = "newest and one behind".
    pub queue_depth: usize,
    /// Frame-to-frame histogram correlation below which a SHOT CUT is
    /// declared and every track dropped. Broadcast footage cuts constantly
    /// and tracks never survive a cut, so carrying them across only paints
    /// rings over the next shot. 0 disables.
    pub scene_cut_corr: f32,
    /// Prefer hardware decode when the source supports it.
    pub hwaccel: bool,
    /// Loop a file source when it ends (useful for demos of a live feed).
    pub loop_file: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self { target_width: 1280, detect_every: 0, queue_depth: 2, scene_cut_corr: 0.5, hwaccel: true, loop_file: false }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DetectionConfig {
    pub model: PathBuf,
    pub input_size: u32,
    pub conf: f32,
    pub iou_nms: f32,
    /// Which class map the model's outputs follow: "coco" or "roboflow_football".
    pub class_map: String,
    /// Inference threads for the CPU provider (0 = runtime default).
    pub threads: usize,
    /// How far (Lab distance) the ground under a detection may sit from the
    /// frame's dominant playing-surface colour before it is rejected as
    /// off-pitch — a spectator, a substitute, someone behind the goal. 0
    /// disables the test. Self-disabling when it would reject most people.
    pub surface_tolerance: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            // A 16:9 input, not a square one. Broadcast football is 16:9, so
            // a square input spends ~44% of the network on grey letterbox
            // bars. Measured on a 4-core Ryzen at 4 threads: 640x640 = 154 ms,
            // 512x512 = 90 ms, 384x672 = 88 ms — the same cost as the square
            // 512 but with 672 px of horizontal detail instead of 512, which
            // is what resolves distant players. `input_size` is now only a
            // fallback: the loader reads the real shape from the model.
            model: PathBuf::from("models/yolo11n-coco-384x672.onnx"),
            input_size: 672,
            // 0.30 lets a COCO detector label chunks of crowd and printed
            // figures on a jersey as "person". On a broadcast close-up that
            // is most of what it finds, and every one of them becomes a
            // ring. Players in play are detected well above 0.45; the
            // spectators mostly are not.
            conf: 0.45,
            iou_nms: 0.5,
            class_map: "coco".into(),
            threads: 0,
            // Generous: it only has to separate grass from a wall of
            // spectators, and being wrong in the strict direction deletes
            // real players.
            surface_tolerance: 60.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ReidConfig {
    pub enabled: bool,
    pub model: Option<PathBuf>,
    /// Re-embed a track every N frames; the POC measured this as the
    /// dominant per-frame cost when done per detection.
    pub reembed_interval: u32,
}

impl Default for ReidConfig {
    fn default() -> Self {
        Self { enabled: false, model: None, reembed_interval: 6 }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PitchConfig {
    pub enabled: bool,
    /// Keypoint model (YOLOv8-pose style, 32 landmarks). None = manual only.
    pub keypoint_model: Option<PathBuf>,
    pub keypoint_input: u32,
    pub keypoint_conf: f32,
    /// Solve a keyframe every N frames on the side thread.
    pub keyframe_every: u32,
    /// Interpolate across gaps up to this many frames; beyond, report none.
    pub max_gap: u32,
    pub ransac_px: f32,
    pub min_inliers: usize,
    pub length_m: f32,
    pub width_m: f32,
    /// Metres outside the lines a point may sit and still count as on the pitch.
    pub on_pitch_margin_m: f32,
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            keypoint_model: None,
            keypoint_input: 640,
            keypoint_conf: 0.5,
            keyframe_every: 15,
            max_gap: 30,
            ransac_px: 6.0,
            min_inliers: 6,
            length_m: 105.0,
            width_m: 68.0,
            on_pitch_margin_m: 2.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AnalyticsConfig {
    pub enabled: bool,
    /// Nearest player within this many metres of the ball may hold possession.
    pub possession_radius_m: f32,
    /// ...for at least this many consecutive frames.
    pub possession_min_frames: u32,
    /// Steps implying more than this speed are rejected as tracker jumps.
    pub speed_ceiling_mps: f32,
    pub smoothing_window: usize,
    pub heatmap_cells: (u32, u32),
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            possession_radius_m: 2.5,
            possession_min_frames: 3,
            speed_ceiling_mps: 12.0,
            smoothing_window: 5,
            heatmap_cells: (21, 14),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// MJPEG preview quality (1-100).
    pub preview_quality: u8,
    /// Cap the preview stream at this width to bound JPEG cost. 0 = the
    /// engine's working width, so the picture and the overlay share one
    /// coordinate space (recommended; see the note on the default).
    pub preview_width: u32,
    pub draw_overlays_in_preview: bool,
    pub record: bool,
    pub record_path: Option<PathBuf>,
    pub session_log: Option<PathBuf>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            // 0 = the engine's own working width, which is what overlay
            // coordinates are in. Downscaling the preview saves real encode
            // cost (measured: 1280w/q80 is 11.7 ms and 396 KB per frame,
            // 960w/q75 is 9.3 ms and 179 KB), but it also makes the picture
            // and the overlay two different pixel sizes — and any layout rule
            // that treats them differently then puts a constant offset
            // between the rings and the players. Quality is where to save
            // instead: it costs nothing in geometry.
            preview_quality: 70,
            preview_width: 0,
            draw_overlays_in_preview: true,
            record: false,
            record_path: None,
            session_log: None,
        }
    }
}

impl Config {
    pub fn from_toml(s: &str) -> crate::Result<Self> {
        toml::from_str(s).map_err(|e| crate::Error::Config(e.to_string()))
    }
    pub fn from_json(s: &str) -> crate::Result<Self> {
        serde_json::from_str(s).map_err(|e| crate::Error::Config(e.to_string()))
    }
}
