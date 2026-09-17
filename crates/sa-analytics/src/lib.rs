//! Match analytics computed live, from PITCH-SPACE tracks.
//!
//! THE ONE RULE (from the POC): every function here takes metres, never
//! pixels. No calibration ⇒ no metrics. A speed from an uncalibrated frame
//! would be a confident, wrong number.
//!
//! Definitions carried over unchanged:
//!  * distance/speed — smooth position before differentiating; REJECT (not
//!    clamp) steps that imply superhuman speed or span > 1 s;
//!  * possession — nearest player to a SEEN ball within `radius`, held for
//!    `min_frames`; a handover is a pass, complete when same team;
//!  * offside — the line is on the SECOND-last defender;
//!  * occupation — share of the field nearer one team than the other;
//!  * heatmap — 2D histogram over the field.

pub mod ball;

use std::collections::{HashMap, VecDeque};

use sa_core::config::AnalyticsConfig;
use sa_core::profile::FieldDims;
use sa_core::{LiveStats, Point2, Team, Track};

pub use ball::{BallParams, BallPoint, BallTracker};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TrackMetrics {
    pub track_id: u32,
    pub team: Option<Team>,
    pub frames: u32,
    pub distance_m: f32,
    pub top_speed_ms: f32,
    pub seconds: f32,
    pub rejected_steps: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Pass {
    pub frame: u64,
    pub from_id: u32,
    pub to_id: u32,
    pub team: Team,
    pub distance_m: f32,
    pub complete: bool,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Spell {
    pub track_id: u32,
    pub team: Team,
    pub start: u64,
    pub end: u64,
}

struct TrackHist {
    /// Recent positions for smoothing: (frame, x, y).
    win: VecDeque<(u64, f32, f32)>,
    last_smoothed: Option<(u64, Point2)>,
    metrics: TrackMetrics,
    last_seen: u64,
}

pub struct Heatmap {
    pub cells: (u32, u32),
    pub counts: Vec<f32>,
    pub team_counts: HashMap<Team, Vec<f32>>,
}

impl Heatmap {
    fn new(cells: (u32, u32)) -> Self {
        Self { cells, counts: vec![0.0; (cells.0 * cells.1) as usize], team_counts: HashMap::new() }
    }
    fn add(&mut self, dims: FieldDims, p: Point2, team: Team) {
        let cx = ((p.x / dims.length) * self.cells.0 as f32).floor();
        let cy = ((p.y / dims.width) * self.cells.1 as f32).floor();
        if cx < 0.0 || cy < 0.0 || cx >= self.cells.0 as f32 || cy >= self.cells.1 as f32 {
            return;
        }
        let i = (cy as u32 * self.cells.0 + cx as u32) as usize;
        self.counts[i] += 1.0;
        let n = (self.cells.0 * self.cells.1) as usize;
        self.team_counts.entry(team).or_insert_with(|| vec![0.0; n])[i] += 1.0;
    }
}

/// Everything accumulated over a session, updated once per calibrated frame.
pub struct LiveAnalytics {
    cfg: AnalyticsConfig,
    dims: FieldDims,
    fps: f32,
    tracks: HashMap<u32, TrackHist>,
    // possession
    holder_run: Option<(u32, Team, u64, u64)>, // (track, team, start, last)
    spells: Vec<Spell>,
    passes: Vec<Pass>,
    team_seconds: HashMap<Team, f32>,
    last_pos: HashMap<u32, Point2>,
    pub heatmap: Heatmap,
    frames_seen: u64,
    ball_seen: u64,
    ball_total: u64,
    /// Running mean x per team, to infer attacking direction.
    team_x_sum: HashMap<Team, (f64, u64)>,
}

impl LiveAnalytics {
    pub fn new(cfg: AnalyticsConfig, dims: FieldDims, fps: f32) -> Self {
        let heat = Heatmap::new(cfg.heatmap_cells);
        Self {
            cfg,
            dims,
            fps: if fps > 0.0 { fps } else { 25.0 },
            tracks: HashMap::new(),
            holder_run: None,
            spells: Vec::new(),
            passes: Vec::new(),
            team_seconds: HashMap::new(),
            last_pos: HashMap::new(),
            heatmap: heat,
            frames_seen: 0,
            ball_seen: 0,
            ball_total: 0,
            team_x_sum: HashMap::new(),
        }
    }

    pub fn dims(&self) -> FieldDims {
        self.dims
    }

    /// One calibrated frame: tracks carry `pitch: Some(..)`; `ball` is the
    /// ball's field position and whether it was actually seen.
    pub fn observe(&mut self, frame: u64, tracks: &[Track], ball: Option<(Point2, bool)>) {
        self.frames_seen += 1;
        let win = self.cfg.smoothing_window.max(1);
        let ceiling = self.cfg.speed_ceiling_mps;
        let fps = self.fps;

        for t in tracks {
            let Some(p) = t.pitch else { continue };
            if !t.class.is_person() {
                continue;
            }
            self.last_pos.insert(t.id, p);
            self.heatmap.add(self.dims, p, t.team);
            if matches!(t.team, Team::A | Team::B) {
                let e = self.team_x_sum.entry(t.team).or_insert((0.0, 0));
                e.0 += p.x as f64;
                e.1 += 1;
            }
            let h = self.tracks.entry(t.id).or_insert_with(|| TrackHist {
                win: VecDeque::with_capacity(win + 1),
                last_smoothed: None,
                metrics: TrackMetrics { track_id: t.id, ..Default::default() },
                last_seen: frame,
            });
            h.metrics.team = Some(t.team);
            h.metrics.frames += 1;
            h.last_seen = frame;
            h.win.push_back((frame, p.x, p.y));
            while h.win.len() > win {
                h.win.pop_front();
            }
            // Smoothed position = mean of the window (edge-replicated by
            // construction: a short window at the start is just shorter).
            let n = h.win.len() as f32;
            let sx = h.win.iter().map(|w| w.1).sum::<f32>() / n;
            let sy = h.win.iter().map(|w| w.2).sum::<f32>() / n;
            let sp = Point2::new(sx, sy);
            if let Some((lf, lp)) = h.last_smoothed {
                let dt = (frame.saturating_sub(lf)) as f32 / fps;
                let step = sp.dist(&lp);
                if dt > 0.0 && dt <= 1.0 {
                    let v = step / dt;
                    if v <= ceiling {
                        h.metrics.distance_m += step;
                        h.metrics.seconds += dt;
                        if v > h.metrics.top_speed_ms {
                            h.metrics.top_speed_ms = v;
                        }
                    } else {
                        h.metrics.rejected_steps += 1;
                    }
                } else if dt > 1.0 {
                    h.metrics.rejected_steps += 1;
                }
            }
            h.last_smoothed = Some((frame, sp));
        }

        // Possession.
        self.ball_total += 1;
        if let Some((bxy, seen)) = ball {
            if seen {
                self.ball_seen += 1;
                let mut best: Option<(f32, &Track)> = None;
                for t in tracks {
                    if let Some(p) = t.pitch {
                        if !t.class.is_person() || t.team == Team::Referee {
                            continue;
                        }
                        let d = p.dist(&bxy);
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, t));
                        }
                    }
                }
                if let Some((d, t)) = best {
                    if d <= self.cfg.possession_radius_m {
                        self.note_holder(frame, t.id, t.team);
                    } else {
                        self.close_run(frame);
                    }
                }
            }
        }
        // Drop history for tracks gone > 5 s.
        if self.frames_seen.is_multiple_of(250) {
            let cutoff = frame.saturating_sub((5.0 * fps) as u64);
            self.tracks.retain(|_, h| h.last_seen >= cutoff);
        }
    }

    fn note_holder(&mut self, frame: u64, id: u32, team: Team) {
        match &mut self.holder_run {
            Some((rid, _, _, last)) if *rid == id && frame - *last <= 2 => {
                *last = frame;
            }
            _ => {
                self.close_run(frame);
                self.holder_run = Some((id, team, frame, frame));
            }
        }
    }

    fn close_run(&mut self, _frame: u64) {
        let Some((id, team, start, end)) = self.holder_run.take() else { return };
        let frames = end - start + 1;
        if frames < self.cfg.possession_min_frames as u64 {
            return;
        }
        let spell = Spell { track_id: id, team, start, end };
        if let Some(prev) = self.spells.last() {
            if prev.track_id != id {
                let dist = match (self.last_pos.get(&prev.track_id), self.last_pos.get(&id)) {
                    (Some(a), Some(b)) => a.dist(b),
                    _ => 0.0,
                };
                self.passes.push(Pass {
                    frame: start,
                    from_id: prev.track_id,
                    to_id: id,
                    team: prev.team,
                    distance_m: dist,
                    complete: prev.team == team && team != Team::Unknown,
                });
            }
        }
        *self.team_seconds.entry(team).or_insert(0.0) += frames as f32 / self.fps;
        self.spells.push(spell);
    }

    /// The goal x each team attacks, inferred from where it spends its time.
    pub fn attacking_goal_x(&self, team: Team) -> Option<f32> {
        let (sum, n) = self.team_x_sum.get(&team)?;
        if *n < 50 {
            return None;
        }
        let mean = (*sum / *n as f64) as f32;
        Some(if mean < self.dims.length / 2.0 { self.dims.length } else { 0.0 })
    }

    /// Offside line for `attacking`: x of the second-last defender.
    pub fn offside_line(&self, tracks: &[Track], attacking: Team) -> Option<f32> {
        let goal_x = self.attacking_goal_x(attacking)?;
        let mut defenders: Vec<f32> = tracks
            .iter()
            .filter(|t| t.class.is_person() && matches!(t.team, Team::A | Team::B) && t.team != attacking)
            .filter_map(|t| t.pitch.map(|p| p.x))
            .collect();
        if defenders.len() < 2 {
            return None;
        }
        if goal_x < self.dims.length / 2.0 {
            defenders.sort_by(|a, b| a.partial_cmp(b).unwrap());
        } else {
            defenders.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }
        Some(defenders[1])
    }

    /// Fraction of the field nearer team A than team B (grid-sampled Voronoi).
    pub fn occupation(&self, tracks: &[Track]) -> Option<(f32, f32)> {
        let a: Vec<Point2> = tracks.iter().filter(|t| t.team == Team::A).filter_map(|t| t.pitch).collect();
        let b: Vec<Point2> = tracks.iter().filter(|t| t.team == Team::B).filter_map(|t| t.pitch).collect();
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let (nx, ny) = (42, 28);
        let mut na = 0;
        for i in 0..nx {
            for j in 0..ny {
                let p = Point2::new((i as f32 + 0.5) / nx as f32 * self.dims.length, (j as f32 + 0.5) / ny as f32 * self.dims.width);
                let da = a.iter().map(|q| q.dist(&p)).fold(f32::INFINITY, f32::min);
                let db = b.iter().map(|q| q.dist(&p)).fold(f32::INFINITY, f32::min);
                if da <= db {
                    na += 1;
                }
            }
        }
        let fa = na as f32 / (nx * ny) as f32;
        Some((fa, 1.0 - fa))
    }

    pub fn fill_stats(&self, tracks: &[Track], stats: &mut LiveStats) {
        let tot: f32 = self.team_seconds.values().sum();
        if tot > 0.0 {
            stats.possession_a = self.team_seconds.get(&Team::A).copied().unwrap_or(0.0) / tot;
            stats.possession_b = self.team_seconds.get(&Team::B).copied().unwrap_or(0.0) / tot;
        }
        let (mut da, mut db) = (0.0, 0.0);
        for h in self.tracks.values() {
            match h.metrics.team {
                Some(Team::A) => da += h.metrics.distance_m,
                Some(Team::B) => db += h.metrics.distance_m,
                _ => {}
            }
        }
        stats.distance_a_m = da;
        stats.distance_b_m = db;
        stats.passes_a = self.passes.iter().filter(|p| p.team == Team::A && p.complete).count() as u32;
        stats.passes_b = self.passes.iter().filter(|p| p.team == Team::B && p.complete).count() as u32;
        stats.ball_seen_rate = if self.ball_total > 0 { self.ball_seen as f32 / self.ball_total as f32 } else { 0.0 };
        stats.offside_x = self.offside_line(tracks, Team::A);
    }

    pub fn track_metrics(&self) -> Vec<TrackMetrics> {
        let mut v: Vec<TrackMetrics> = self.tracks.values().map(|h| h.metrics.clone()).collect();
        v.sort_by_key(|m| m.track_id);
        v
    }
    pub fn passes(&self) -> &[Pass] {
        &self.passes
    }
    pub fn spells(&self) -> &[Spell] {
        &self.spells
    }

    pub fn to_json(&self) -> serde_json::Value {
        let tot: f32 = self.team_seconds.values().sum();
        serde_json::json!({
            "field": { "length": self.dims.length, "width": self.dims.width },
            "fps": self.fps,
            "frames": self.frames_seen,
            "tracks": self.track_metrics(),
            "passes": self.passes,
            "spells": self.spells,
            "possession_share": {
                "A": if tot > 0.0 { self.team_seconds.get(&Team::A).copied().unwrap_or(0.0) / tot } else { 0.0 },
                "B": if tot > 0.0 { self.team_seconds.get(&Team::B).copied().unwrap_or(0.0) / tot } else { 0.0 },
            },
            "ball_seen_rate": if self.ball_total > 0 { self.ball_seen as f32 / self.ball_total as f32 } else { 0.0 },
            "heatmap": { "cells": self.heatmap.cells, "counts": self.heatmap.counts },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sa_core::{BBox, Class, TrackState};

    fn track(id: u32, team: Team, x: f32, y: f32) -> Track {
        Track { id, class: Class::Player, bbox: BBox::default(), conf: 1.0, state: TrackState::Confirmed, team, pitch: Some(Point2::new(x, y)), embedding: None, label: id }
    }

    #[test]
    fn distance_and_speed_on_a_run() {
        let cfg = AnalyticsConfig { smoothing_window: 1, ..Default::default() };
        let mut a = LiveAnalytics::new(cfg, FieldDims { length: 105.0, width: 68.0 }, 25.0);
        // 5 m/s for 10 s = 50 m.
        for f in 0..=250u64 {
            let x = 10.0 + 5.0 * f as f32 / 25.0;
            a.observe(f, &[track(1, Team::A, x, 30.0)], None);
        }
        let m = &a.track_metrics()[0];
        assert!((m.distance_m - 50.0).abs() < 0.5, "{}", m.distance_m);
        assert!((m.top_speed_ms - 5.0).abs() < 0.2);
        // A teleport is rejected, not counted.
        a.observe(251, &[track(1, Team::A, 90.0, 30.0)], None);
        assert_eq!(a.track_metrics()[0].rejected_steps, 1);
    }

    #[test]
    fn one_handover_is_one_pass() {
        let cfg = AnalyticsConfig { possession_min_frames: 3, ..Default::default() };
        let mut a = LiveAnalytics::new(cfg, FieldDims { length: 105.0, width: 68.0 }, 25.0);
        let p1 = track(1, Team::A, 20.0, 30.0);
        let p2 = track(2, Team::A, 40.0, 30.0);
        for f in 0..10 {
            a.observe(f, &[p1.clone(), p2.clone()], Some((Point2::new(20.5, 30.0), true)));
        }
        for f in 10..15 {
            a.observe(f, &[p1.clone(), p2.clone()], Some((Point2::new(30.0, 30.0), true))); // in flight
        }
        for f in 15..25 {
            a.observe(f, &[p1.clone(), p2.clone()], Some((Point2::new(40.5, 30.0), true)));
        }
        a.observe(25, &[p1.clone(), p2.clone()], Some((Point2::new(60.0, 30.0), true)));
        assert_eq!(a.passes().len(), 1);
        assert!(a.passes()[0].complete);
        assert!((a.passes()[0].distance_m - 20.0).abs() < 1e-3);
    }

    #[test]
    fn offside_uses_second_last_defender() {
        let cfg = AnalyticsConfig::default();
        let mut a = LiveAnalytics::new(cfg, FieldDims { length: 105.0, width: 68.0 }, 25.0);
        // Team A lives in the left half => attacks the right goal (x=105).
        let mut ts = vec![track(1, Team::A, 30.0, 30.0), track(2, Team::A, 40.0, 30.0)];
        ts.push(track(10, Team::B, 100.0, 34.0)); // keeper
        ts.push(track(11, Team::B, 80.0, 20.0)); // last outfield defender
        ts.push(track(12, Team::B, 70.0, 40.0));
        for f in 0..60 {
            a.observe(f, &ts, None);
        }
        assert_eq!(a.attacking_goal_x(Team::A), Some(105.0));
        assert_eq!(a.offside_line(&ts, Team::A), Some(80.0));
    }
}
