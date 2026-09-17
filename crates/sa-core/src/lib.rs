//! Shared vocabulary for every engine crate: frames, detections, tracks, pitch
//! geometry, the per-frame state handed to the UI, and the sport profile
//! abstraction. Deliberately free of I/O and heavy dependencies so every other
//! crate can depend on it without dragging anything in.

pub mod config;
pub mod frame;
pub mod geometry;
pub mod profile;
pub mod state;

pub use config::Config;
pub use frame::{Frame, PixelFormat};
pub use geometry::{BBox, Point2};
pub use profile::{Class, FieldDims, FieldModel, LandmarkId, Segment, SportProfile, Team};
pub use state::{BallState, Calibration, CalibSource, Detection, FrameState, LiveStats, StageLatency, Track, TrackState};

/// Errors that any stage can raise. Stages report; the engine decides.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("source error: {0}")]
    Source(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
