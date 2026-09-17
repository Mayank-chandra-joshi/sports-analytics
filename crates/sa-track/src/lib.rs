//! Multi-object tracking in the BoT-SORT family:
//!
//!  * Kalman prediction on `(cx, cy, w, h)` + velocities (`kalman.rs`);
//!  * ByteTrack's two-pass association — high-confidence detections first,
//!    then the low-confidence leftovers may only EXTEND existing tracks, which
//!    is what keeps a briefly-occluded player alive without spawning junk;
//!  * optional appearance fusion: when tracks and detections carry ReID
//!    embeddings, the IoU cost is blended with cosine distance;
//!  * camera-motion compensation: the caller passes the frame's background
//!    shift and every prediction is moved by it BEFORE matching, so a pan
//!    does not read as every player sprinting sideways.
//!
//! Identity vetoes, crowd handling and re-acquire policy live in
//! `sa-identity`; this crate only says "these boxes are the same object".

pub mod assign;
pub mod kalman;

use sa_core::profile::TrackerParams;
use sa_core::{BBox, Class, Detection, Team, Track, TrackState};
use std::sync::Arc;

use assign::gated_assign;
use kalman::BoxKalman;

#[derive(Debug, Clone)]
struct Tracklet {
    id: u32,
    class: Class,
    kf: BoxKalman,
    last_box: BBox,
    conf: f32,
    hits: u32,
    lost: u32,
    confirmed: bool,
    embedding: Option<Arc<[f32]>>,
    label: u32,
}

#[derive(Clone)]
pub struct Tracker {
    params: TrackerParams,
    tracks: Vec<Tracklet>,
    next_id: u32,
    next_label: u32,
    frame: u64,
}

/// A detection plus an optional appearance vector for this frame.
pub struct Input<'a> {
    pub det: &'a Detection,
    pub embedding: Option<Arc<[f32]>>,
}

impl Tracker {
    pub fn new(params: TrackerParams) -> Self {
        Self { params, tracks: Vec::new(), next_id: 1, next_label: 0, frame: 0 }
    }

    pub fn params(&self) -> &TrackerParams {
        &self.params
    }

    /// Advance one frame. `camera_shift` is the background translation in
    /// pixels since the previous frame (0,0 when unknown).
    pub fn update(&mut self, inputs: &[Input<'_>], camera_shift: (f32, f32)) -> Vec<Track> {
        self.update_at(self.frame + 1, inputs, camera_shift)
    }

    /// Advance to source frame `frame_id`. When the detector cannot keep up
    /// with the source, consecutive processed frames are several source
    /// frames apart — a player has then moved several frames' worth of
    /// distance, and a single Kalman step under-predicts by exactly that
    /// factor, which is what breaks the IoU gate and churns ids. Stepping
    /// the filter by the REAL elapsed frames keeps the prediction where the
    /// player actually is.
    pub fn update_at(&mut self, frame_id: u64, inputs: &[Input<'_>], camera_shift: (f32, f32)) -> Vec<Track> {
        let steps = frame_id.saturating_sub(self.frame).clamp(1, 30) as u32;
        self.frame = frame_id;
        let p = self.params.clone();

        // Predict, compensating for camera motion.
        for t in &mut self.tracks {
            for _ in 0..steps {
                t.kf.predict();
            }
            if camera_shift.0 != 0.0 || camera_shift.1 != 0.0 {
                t.kf.shift(camera_shift.0, camera_shift.1);
            }
        }

        // Split detections by confidence (ByteTrack).
        let mut high: Vec<usize> = Vec::new();
        let mut low: Vec<usize> = Vec::new();
        for (i, inp) in inputs.iter().enumerate() {
            if inp.det.conf >= p.high_thresh {
                high.push(i);
            } else if inp.det.conf >= p.low_thresh {
                low.push(i);
            }
        }

        // Pass 1: all tracks (confirmed first; tentative join too) vs high.
        let track_idx: Vec<usize> = (0..self.tracks.len()).collect();
        let (m1, ut1, ud1) = self.associate(&track_idx, &high, inputs, true);
        let mut matched_tracks = vec![false; self.tracks.len()];
        for (ti, di) in m1 {
            let t = track_idx[ti];
            self.hit(t, inputs, high[di]);
            matched_tracks[t] = true;
        }

        // Pass 2: still-unmatched CONFIRMED tracks vs low-confidence dets.
        // Lower IoU bar, no appearance (low-conf crops are unreliable).
        let rem_tracks: Vec<usize> = ut1.iter().map(|&i| track_idx[i]).filter(|&t| self.tracks[t].confirmed).collect();
        let (m2, _ut2, _) = self.associate(&rem_tracks, &low, inputs, false);
        for (ti, di) in m2 {
            let t = rem_tracks[ti];
            self.hit(t, inputs, low[di]);
            matched_tracks[t] = true;
        }

        // Unmatched tracks age; unmatched high detections start new tracks.
        for (t, matched) in matched_tracks.iter().enumerate() {
            if !matched {
                self.tracks[t].lost += 1;
            }
        }
        for di in ud1 {
            let i = high[di];
            self.spawn(inputs, i);
        }
        // `track_buffer` is a DURATION ("keep a lost player this long"), but
        // it is counted in PROCESSED frames. With detection running one frame
        // in N those are N source frames apart, so a buffer of 60 keeps dead
        // tracks alive for 60*N frames — which is why the track count climbs
        // and the screen fills with rings nobody is under. Scale it down by
        // the same factor the frames were scaled up.
        let buffer = (p.track_buffer / steps.max(1)).max(3);
        self.tracks.retain(|t| t.lost <= buffer && (t.confirmed || t.lost == 0));

        self.tracks
            .iter()
            .filter(|t| t.confirmed || t.hits >= p.min_hits)
            .map(|t| Track {
                id: t.id,
                class: t.class,
                bbox: if t.lost == 0 {
                    t.last_box
                } else {
                    // As in `predict_at`: the filter's position, the last
                    // sighting's size.
                    let c = t.kf.bbox().center();
                    BBox::from_cxcywh(c.x, c.y, t.last_box.w(), t.last_box.h())
                },
                conf: t.conf,
                state: if t.lost > 0 {
                    TrackState::Lost { frames: t.lost }
                } else if t.confirmed {
                    TrackState::Confirmed
                } else {
                    TrackState::Tentative
                },
                team: Team::Unknown,
                pitch: None,
                embedding: t.embedding.clone(),
                label: t.label,
            })
            .collect()
    }

    fn associate(&self, tracks: &[usize], dets: &[usize], inputs: &[Input<'_>], use_appearance: bool)
        -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if tracks.is_empty() || dets.is_empty() {
            return (Vec::new(), (0..tracks.len()).collect(), (0..dets.len()).collect());
        }
        let p = &self.params;
        let (rows, cols) = (tracks.len(), dets.len());
        let mut cost = vec![1.0f32; rows * cols];
        for (r, &t) in tracks.iter().enumerate() {
            let tr = &self.tracks[t];
            let pred = tr.kf.bbox();
            for (c, &d) in dets.iter().enumerate() {
                let inp = &inputs[d];
                if inp.det.class.is_person() != tr.class.is_person() {
                    cost[r * cols + c] = f32::INFINITY;
                    continue;
                }
                let iou = pred.iou(&inp.det.bbox);
                let mut cst = 1.0 - iou;
                if use_appearance && p.appearance_weight > 0.0 {
                    if let (Some(a), Some(b)) = (&tr.embedding, &inp.embedding) {
                        let cos = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
                        let app = (1.0 - cos).clamp(0.0, 1.0);
                        cst = (1.0 - p.appearance_weight) * cst + p.appearance_weight * app;
                    }
                }
                cost[r * cols + c] = cst;
            }
        }
        let max_cost = 1.0 - p.match_iou;
        gated_assign(&cost, rows, cols, max_cost)
    }

    fn hit(&mut self, t: usize, inputs: &[Input<'_>], di: usize) {
        let inp = &inputs[di];
        let b = inp.det.bbox;
        let c = b.center();
        let tr = &mut self.tracks[t];
        tr.kf.update(c.x, c.y, b.w(), b.h());
        tr.last_box = b;
        tr.conf = inp.det.conf;
        tr.hits += 1;
        tr.lost = 0;
        if tr.hits >= self.params.min_hits {
            tr.confirmed = true;
        }
        if let Some(e) = &inp.embedding {
            // Exponential blend keeps the reference stable across a turn.
            tr.embedding = Some(match &tr.embedding {
                Some(old) => {
                    let mut v: Vec<f32> = old.iter().zip(e.iter()).map(|(o, n)| 0.9 * o + 0.1 * n).collect();
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    v.iter_mut().for_each(|x| *x /= n);
                    Arc::from(v)
                }
                None => e.clone(),
            });
        }
    }

    fn spawn(&mut self, inputs: &[Input<'_>], di: usize) {
        let inp = &inputs[di];
        let b = inp.det.bbox;
        let c = b.center();
        let id = self.next_id;
        self.next_id += 1;
        // Display numbers are for a human reading the screen, so they must
        // stay small and stable. Reuse the lowest number no live track is
        // using rather than counting up forever — otherwise a clip with
        // fifteen players on screen shows #322 after a few id churns, which
        // reads as "the tracker is lost" even when it is not.
        let used: std::collections::HashSet<u32> = self.tracks.iter().map(|t| t.label).collect();
        let label = (0..).find(|n| !used.contains(n)).unwrap_or(self.next_label);
        self.next_label = self.next_label.max(label + 1);
        self.tracks.push(Tracklet {
            id,
            class: inp.det.class,
            kf: BoxKalman::new(c.x, c.y, b.w(), b.h()),
            last_box: b,
            conf: inp.det.conf,
            hits: 1,
            lost: 0,
            confirmed: self.params.min_hits <= 1,
            embedding: inp.embedding.clone(),
            label,
        });
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.frame = 0;
    }

    /// Drop every track, keeping the frame clock. For a SHOT CUT: the people
    /// in the previous shot are not in this one, and carrying their boxes
    /// across leaves rings scattered over a close-up of somebody else. The
    /// clock is kept so `update_at` still measures elapsed frames correctly.
    pub fn clear_tracks(&mut self) {
        self.tracks.clear();
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Where every confirmed track is at `frame_id`, WITHOUT consuming a
    /// detection — the motion prediction only. The UI calls this for frames
    /// the detector never saw, so an overlay keeps up with the picture
    /// instead of sitting on the last processed position.
    ///
    /// Read-only by construction: it clones each filter rather than
    /// advancing it, so the real state is untouched and a later `update_at`
    /// behaves exactly as it would have.
    pub fn predict_at(&self, frame_id: u64) -> Vec<Track> {
        let steps = frame_id.saturating_sub(self.frame).clamp(0, 30) as u32;
        let min_hits = self.params.min_hits;
        self.tracks
            .iter()
            // The SAME set `update` reports. Filtering to confirmed-only here
            // made every tentative player disappear on the frames between
            // detections, which on a fast shot is most of them — the screen
            // then shows a handful of rings for a full team.
            .filter(|t| t.confirmed || t.hits >= min_hits)
            .map(|t| {
                let mut kf = t.kf.clone();
                for _ in 0..steps {
                    kf.predict();
                }
                // POSITION from the filter, SIZE from the last real sighting.
                // The filter estimates w/h as well, and with only a few
                // observations that estimate lags — the predicted box comes
                // out shorter than the player, its bottom edge rises, and the
                // ground ring drawn there floats above their feet. Width and
                // height barely change over a third of a second anyway, so
                // taking them from the last observation is both more accurate
                // and more stable.
                let c = kf.bbox().center();
                let (w, h) = (t.last_box.w(), t.last_box.h());
                Track {
                    id: t.id,
                    class: t.class,
                    bbox: BBox::from_cxcywh(c.x, c.y, w, h),
                    conf: t.conf,
                    state: if t.lost > 0 {
                        TrackState::Lost { frames: t.lost }
                    } else if t.confirmed {
                        TrackState::Confirmed
                    } else {
                        TrackState::Tentative
                    },
                    team: Team::Unknown,
                    pitch: None,
                    embedding: None,
                    label: t.label,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, conf: f32) -> Detection {
        Detection { class: Class::Player, bbox: BBox::new(x, y, x + 20.0, y + 50.0), conf }
    }

    #[test]
    fn keeps_id_through_motion_and_gap() {
        let mut tr = Tracker::new(TrackerParams { min_hits: 1, ..Default::default() });
        let mut id = None;
        for i in 0..10 {
            let d = det(100.0 + 4.0 * i as f32, 100.0, 0.9);
            let out = tr.update(&[Input { det: &d, embedding: None }], (0.0, 0.0));
            assert_eq!(out.len(), 1);
            match id {
                None => id = Some(out[0].id),
                Some(v) => assert_eq!(out[0].id, v),
            }
        }
        // Two frames with nothing, then it re-appears on trajectory.
        for _ in 0..2 {
            let out = tr.update(&[], (0.0, 0.0));
            assert!(matches!(out[0].state, TrackState::Lost { .. }));
        }
        let d = det(100.0 + 4.0 * 12.0, 100.0, 0.9);
        let out = tr.update(&[Input { det: &d, embedding: None }], (0.0, 0.0));
        assert_eq!(out[0].id, id.unwrap());
        assert_eq!(out[0].state, TrackState::Confirmed);
    }

    #[test]
    fn low_conf_extends_but_never_spawns() {
        let mut tr = Tracker::new(TrackerParams { min_hits: 1, ..Default::default() });
        let weak = det(300.0, 300.0, 0.2);
        let out = tr.update(&[Input { det: &weak, embedding: None }], (0.0, 0.0));
        assert!(out.is_empty());
        let strong = det(300.0, 300.0, 0.9);
        let out = tr.update(&[Input { det: &strong, embedding: None }], (0.0, 0.0));
        assert_eq!(out.len(), 1);
        let weak2 = det(303.0, 300.0, 0.2);
        let out = tr.update(&[Input { det: &weak2, embedding: None }], (0.0, 0.0));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].state, TrackState::Confirmed);
    }

    #[test]
    fn camera_shift_keeps_lock() {
        let mut tr = Tracker::new(TrackerParams { min_hits: 1, ..Default::default() });
        let d = det(100.0, 100.0, 0.9);
        let out = tr.update(&[Input { det: &d, embedding: None }], (0.0, 0.0));
        let id = out[0].id;
        // Camera pans 30px: the same player appears 30px right.
        let d2 = det(130.0, 100.0, 0.9);
        let out = tr.update(&[Input { det: &d2, embedding: None }], (30.0, 0.0));
        assert_eq!(out[0].id, id);
    }
}
