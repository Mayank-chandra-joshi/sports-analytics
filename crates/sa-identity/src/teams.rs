//! Team and referee assignment from kit colour — the POC's
//! `assign_teams_from_kits` and `LiveTeamClassifier`, ported.
//!
//! No colour is named anywhere in this file. The two kits are whatever this
//! clip's players wear; the referee is the minority kit that matches
//! neither; "same shirt" is measured against this clip's own scatter. Every
//! rule compares readings to each other, never to a palette.

use std::collections::HashMap;

use sa_core::state::TeamColours;
use sa_core::Team;

use crate::colour::{lab_dist, rgb_to_lab};

#[derive(Debug, Clone, Default)]
pub struct TeamAssignment {
    pub team_of: HashMap<u32, Team>,
    pub colour_of: HashMap<Team, [u8; 3]>,
    pub confidence: f32,
}

impl TeamAssignment {
    pub fn team(&self, track_id: u32) -> Team {
        self.team_of.get(&track_id).copied().unwrap_or(Team::Unknown)
    }

    /// Which team a kit colour belongs to, against the learned kits. The bar
    /// is half the distance between the two teams — a colour must be nearer
    /// one team than the teams are to each other. Referee must clear the same
    /// bar AND be nearest.
    pub fn classify(&self, rgb: [u8; 3]) -> Team {
        let (Some(a), Some(b)) = (self.colour_of.get(&Team::A), self.colour_of.get(&Team::B)) else {
            return Team::Unknown;
        };
        let (la, lb, lx) = (rgb_to_lab(*a), rgb_to_lab(*b), rgb_to_lab(rgb));
        let gap = lab_dist(la, lb);
        let (da, db) = (lab_dist(lx, la), lab_dist(lx, lb));
        if let Some(r) = self.colour_of.get(&Team::Referee) {
            let dr = lab_dist(lx, rgb_to_lab(*r));
            if dr < da.min(db) && dr <= 0.5 * gap {
                return Team::Referee;
            }
        }
        if da.min(db) > 0.5 * gap {
            return Team::Unknown;
        }
        if da <= db { Team::A } else { Team::B }
    }

    pub fn colours(&self) -> Option<TeamColours> {
        Some(TeamColours {
            a: *self.colour_of.get(&Team::A)?,
            b: *self.colour_of.get(&Team::B)?,
            referee: self.colour_of.get(&Team::Referee).copied(),
            confidence: self.confidence,
        })
    }
}

fn median_rgb(samples: &[[u8; 3]]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let mut v: Vec<u8> = samples.iter().map(|s| s[c]).collect();
        v.sort_unstable();
        out[c] = v[v.len() / 2];
    }
    out
}

/// Typical scatter of one track's readings about its own median, in Lab —
/// the scale on which two colours count as the same shirt.
fn sample_spread(tracks: &HashMap<u32, Vec<[u8; 3]>>, ids: &[u32]) -> f32 {
    let mut per = Vec::new();
    for t in ids {
        let s = &tracks[t];
        if s.len() < 2 {
            continue;
        }
        let labs: Vec<[f32; 3]> = s.iter().map(|c| rgb_to_lab(*c)).collect();
        let med = crate::colour::median3(&labs);
        let mut d: Vec<f32> = labs.iter().map(|l| lab_dist(*l, med)).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per.push(d[d.len() / 2]);
    }
    if per.is_empty() {
        return 0.0;
    }
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per[per.len() / 2]
}

/// k-means++ with restarts on 3-vectors; returns (labels, centres, compactness).
fn kmeans(feats: &[[f32; 3]], k: usize, restarts: usize) -> (Vec<usize>, Vec<[f32; 3]>, f32) {
    let n = feats.len();
    let k = k.min(n).max(1);
    let mut best: Option<(Vec<usize>, Vec<[f32; 3]>, f32)> = None;
    let mut seed = 0x2545F4914F6CDD1Du64 ^ n as u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..restarts.max(1) {
        // k-means++ seeding
        let mut centres: Vec<[f32; 3]> = vec![feats[(rnd() * n as f64) as usize % n]];
        while centres.len() < k {
            let d2: Vec<f64> = feats
                .iter()
                .map(|f| centres.iter().map(|c| lab_dist(*f, *c)).fold(f32::INFINITY, f32::min).powi(2) as f64)
                .collect();
            let sum: f64 = d2.iter().sum();
            let mut r = rnd() * sum;
            let mut pick = n - 1;
            for (i, d) in d2.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    pick = i;
                    break;
                }
            }
            centres.push(feats[pick]);
        }
        let mut labels = vec![0usize; n];
        for _ in 0..40 {
            let mut changed = false;
            for (i, f) in feats.iter().enumerate() {
                let mut bi = 0;
                let mut bd = f32::INFINITY;
                for (j, c) in centres.iter().enumerate() {
                    let d = lab_dist(*f, *c);
                    if d < bd {
                        bd = d;
                        bi = j;
                    }
                }
                if labels[i] != bi {
                    labels[i] = bi;
                    changed = true;
                }
            }
            let mut sums = vec![[0.0f32; 3]; k];
            let mut cnt = vec![0usize; k];
            for (i, f) in feats.iter().enumerate() {
                for c in 0..3 {
                    sums[labels[i]][c] += f[c];
                }
                cnt[labels[i]] += 1;
            }
            for j in 0..k {
                if cnt[j] > 0 {
                    for c in 0..3 {
                        centres[j][c] = sums[j][c] / cnt[j] as f32;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let compact: f32 = feats.iter().enumerate().map(|(i, f)| lab_dist(*f, centres[labels[i]]).powi(2)).sum();
        if best.as_ref().is_none_or(|b| compact < b.2) {
            best = Some((labels, centres, compact));
        }
    }
    best.unwrap()
}

/// The clustering itself, over `{track_id: [kit_rgb, ...]}`.
pub fn assign_teams_from_kits(tracks: &HashMap<u32, Vec<[u8; 3]>>, n_teams: usize) -> TeamAssignment {
    if tracks.len() < n_teams.max(2) {
        return TeamAssignment::default();
    }
    let mut ids: Vec<u32> = tracks.keys().copied().collect();
    ids.sort_unstable();
    // Median over the track's frames, THEN cluster.
    let feats: Vec<[f32; 3]> = ids.iter().map(|t| rgb_to_lab(median_rgb(&tracks[t]))).collect();

    // Over-split, then merge what is the same kit.
    let k = (n_teams + 4).max(5).min(ids.len());
    let (labels, centres, compact) = kmeans(&feats, k, 8);
    let within = (compact / ids.len().max(1) as f32).sqrt();
    let mut widest = 0.0f32;
    for i in 0..k {
        for j in i + 1..k {
            widest = widest.max(lab_dist(centres[i], centres[j]));
        }
    }
    let spread = sample_spread(tracks, &ids);
    let merge_bar = (3.0 * spread).max(2.3).max(0.15 * widest);

    let mut group: Vec<usize> = (0..k).collect();
    let gmean = |group: &Vec<usize>, g: usize| -> [f32; 3] {
        let mem: Vec<[f32; 3]> = (0..k).filter(|&c| group[c] == g).map(|c| centres[c]).collect();
        let mut m = [0.0f32; 3];
        for p in &mem {
            for c in 0..3 {
                m[c] += p[c] / mem.len() as f32;
            }
        }
        m
    };
    loop {
        let mut gs: Vec<usize> = group.clone();
        gs.sort_unstable();
        gs.dedup();
        if gs.len() <= 1 {
            break;
        }
        let mut best: Option<(f32, usize, usize)> = None;
        for x in 0..gs.len() {
            for y in x + 1..gs.len() {
                let d = lab_dist(gmean(&group, gs[x]), gmean(&group, gs[y]));
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, gs[x], gs[y]));
                }
            }
        }
        match best {
            Some((d, bi, bj)) if d < merge_bar => {
                for g in group.iter_mut() {
                    if *g == bj {
                        *g = bi;
                    }
                }
            }
            _ => break,
        }
    }
    let labels: Vec<usize> = labels.iter().map(|&c| group[c]).collect();
    let mut gids: Vec<usize> = group.clone();
    gids.sort_unstable();
    gids.dedup();
    if gids.len() < 2 {
        return TeamAssignment::default();
    }
    let gcentre: HashMap<usize, [f32; 3]> = gids.iter().map(|&g| (g, gmean(&group, g))).collect();
    let gsize: HashMap<usize, usize> = gids.iter().map(|&g| (g, labels.iter().filter(|&&l| l == g).count())).collect();

    // Two biggest groups are the teams; a referee is a MINORITY in a
    // DIFFERENT kit — at least half the team gap from both.
    let mut order = gids.clone();
    order.sort_by_key(|g| std::cmp::Reverse(gsize[g]));
    let players = [order[0], order[1]];
    let gap = lab_dist(gcentre[&players[0]], gcentre[&players[1]]);
    let odd: HashMap<usize, f32> = order[2..]
        .iter()
        .map(|&g| (g, players.iter().map(|p| lab_dist(gcentre[&g], gcentre[p])).fold(f32::INFINITY, f32::min)))
        .collect();
    let min_team = gsize[&players[0]].min(gsize[&players[1]]);
    let cand: Vec<usize> = order[2..].iter().copied().filter(|g| gsize[g] * 2 <= min_team && odd[g] >= 0.5 * gap).collect();
    let referee_cl = cand
        .iter()
        .copied()
        .min_by(|a, b| odd[b].partial_cmp(&odd[a]).unwrap().then(gsize[a].cmp(&gsize[b])));

    let mut names: HashMap<usize, Team> = HashMap::new();
    let mut folded: HashMap<usize, usize> = HashMap::new();
    if let Some(r) = referee_cl {
        names.insert(r, Team::Referee);
    }
    for &g in &order[2..] {
        if names.contains_key(&g) {
            continue;
        }
        if odd[&g] <= 0.5 * gap {
            let p = *players.iter().min_by(|a, b| lab_dist(gcentre[&g], gcentre[a]).partial_cmp(&lab_dist(gcentre[&g], gcentre[b])).unwrap()).unwrap();
            folded.insert(g, p);
        } else {
            names.insert(g, Team::Unknown);
        }
    }

    // Which is A: the darker kit, or if lightness cannot separate them, the
    // axis that separates these two most reliably.
    let (mut a_cl, mut b_cl) = (players[0], players[1]);
    let (ca, cb) = (gcentre[&a_cl], gcentre[&b_cl]);
    let mut resid = [0.0f32; 3];
    for (i, f) in feats.iter().enumerate() {
        let c = gcentre[&labels[i]];
        for k in 0..3 {
            resid[k] += (f[k] - c[k]).powi(2);
        }
    }
    let spread_ax: Vec<f32> = resid.iter().map(|r| (r / feats.len() as f32).sqrt() + 1e-6).collect();
    let axis = if (ca[0] - cb[0]).abs() <= spread_ax[0] {
        if (ca[1] - cb[1]).abs() / spread_ax[1] >= (ca[2] - cb[2]).abs() / spread_ax[2] { 1 } else { 2 }
    } else {
        0
    };
    if ca[axis] > cb[axis] {
        std::mem::swap(&mut a_cl, &mut b_cl);
    }
    names.insert(a_cl, Team::A);
    names.insert(b_cl, Team::B);
    for (g, p) in &folded {
        let t = names[p];
        names.insert(*g, t);
    }
    let team_of: HashMap<u32, Team> = ids.iter().enumerate().map(|(i, t)| (*t, names[&labels[i]])).collect();
    let mut colour_of = HashMap::new();
    for (cl, nm) in [(Some(a_cl), Team::A), (Some(b_cl), Team::B), (referee_cl, Team::Referee)] {
        let Some(cl) = cl else { continue };
        let members: Vec<[u8; 3]> = ids.iter().enumerate().filter(|(i, _)| labels[*i] == cl).map(|(_, t)| median_rgb(&tracks[t])).collect();
        if !members.is_empty() {
            colour_of.insert(nm, median_rgb(&members));
        }
    }
    let between = gap;
    let confidence = (between / (between + within + 1e-6)).clamp(0.0, 1.0);
    TeamAssignment { team_of, colour_of, confidence }
}

/// Accumulates kit readings per track and re-clusters periodically, so the
/// labelling converges as more of both teams come into shot.
pub struct LiveTeamClassifier {
    min_tracks: usize,
    min_samples: usize,
    refresh: u32,
    cap: usize,
    kits: HashMap<u32, Vec<[u8; 3]>>,
    assignment: TeamAssignment,
    since: u32,
    n_teams: usize,
}

impl LiveTeamClassifier {
    pub fn new(refresh: u32, n_teams: usize) -> Self {
        Self { min_tracks: 4, min_samples: 5, refresh, cap: 400, kits: HashMap::new(), assignment: TeamAssignment::default(), since: u32::MAX / 2, n_teams }
    }

    pub fn observe_frame(&mut self, kits: &HashMap<u32, [u8; 3]>) -> &TeamAssignment {
        for (t, c) in kits {
            let v = self.kits.entry(*t).or_default();
            if v.len() < self.cap {
                v.push(*c);
            }
        }
        self.since = self.since.saturating_add(1);
        if self.since >= self.refresh {
            let ready: HashMap<u32, Vec<[u8; 3]>> = self.kits.iter().filter(|(_, v)| v.len() >= self.min_samples).map(|(k, v)| (*k, v.clone())).collect();
            if ready.len() >= self.min_tracks {
                self.since = 0;
                self.assignment = assign_teams_from_kits(&ready, self.n_teams);
            }
        }
        &self.assignment
    }

    pub fn assignment(&self) -> &TeamAssignment {
        &self.assignment
    }

    /// Forget tracks not seen for a while (ids churn on a live stream).
    pub fn retain(&mut self, alive: &[u32]) {
        if self.kits.len() > 512 {
            self.kits.retain(|k, _| alive.contains(k));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_kits_and_a_referee() {
        let mut tracks = HashMap::new();
        let jitter = |base: [u8; 3], i: u8| [base[0].saturating_add(i % 5), base[1].saturating_add(i % 3), base[2].saturating_add(i % 4)];
        for t in 0..8u32 {
            tracks.insert(t, (0..6).map(|i| jitter([200, 20, 20], i + t as u8)).collect());
        }
        for t in 8..16u32 {
            tracks.insert(t, (0..6).map(|i| jitter([230, 230, 230], i + t as u8)).collect());
        }
        tracks.insert(99, (0..6).map(|i| jitter([250, 220, 20], i)).collect());
        let a = assign_teams_from_kits(&tracks, 2);
        assert!(a.confidence > 0.5, "conf {}", a.confidence);
        assert_eq!(a.team(0), Team::A, "red is darker => A");
        assert_eq!(a.team(9), Team::B);
        assert_eq!(a.team(99), Team::Referee);
        assert_eq!(a.classify([210, 30, 30]), Team::A);
        assert_eq!(a.classify([20, 20, 200]), Team::Unknown);
    }

    #[test]
    fn one_kit_reports_no_confidence() {
        let mut tracks = HashMap::new();
        for t in 0..10u32 {
            tracks.insert(t, vec![[100, 100, 100]; 6]);
        }
        let a = assign_teams_from_kits(&tracks, 2);
        assert_eq!(a.confidence, 0.0);
    }
}
