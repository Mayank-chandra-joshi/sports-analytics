//! Identity: who is on which team, and (later) which one is the target.
//!
//! Teams come from kit colour, clustered live and refreshed as more players
//! appear — no vision model, no palette. The single-target lock cascade from
//! the POC is scheduled for M4 and will live here as `lock`.

pub mod colour;
pub mod teams;

pub use colour::{kit_reading, on_playing_surface, rgb_to_lab, surface_colour, KIT_MIN_CONTRAST};
pub use teams::{assign_teams_from_kits, LiveTeamClassifier, TeamAssignment};
