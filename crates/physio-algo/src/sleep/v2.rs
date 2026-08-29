//! V2 (cardiorespiratory) sleep stager — the 5.0/MG default. Per-night z-scored HR / HR-variability /
//! motion emissions, a deep gate on the 11-min HR-flatness percentile, a soft sleep-cycle prior, a
//! self-calibrating jerk wake gate, an R-R RSA respiration term, and Viterbi transition smoothing. The
//! per-epoch coefficients are fixed a-priori from sleep physiology + population base rates.
//!
//! Mirrors the shipped app's V2 recipe. Absent signal scores the neutral centre, so a sparse channel never
//! blocks a stage. Outputs are wellness estimates, never medical advice.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::common::{flatten_rr, median, ZScore};
use super::params::Params;
use super::input::{AccelSample, HrSample, SleepInput};
use super::{SleepStage, StageSegment};

/// Consecutive non-wake epochs that mark sleep onset for the onset-anchored priors.
const SUSTAINED_ONSET_EPOCHS: usize = 10;
/// One epoch in minutes, the unit the onset-anchored REM guard grades over.
const EPOCH_MIN: f64 = 0.5;

// Stage indices for the emission/transition arrays: order [deep, rem, light, awake].
const DEEP: usize = 0;
const REM: usize = 1;
const LIGHT: usize = 2;
const AWAKE: usize = 3;

/// Column order of every emission row and of both transition-matrix axes.
pub const STAGE_ORDER: [SleepStage; 4] =
    [SleepStage::Deep, SleepStage::Rem, SleepStage::Light, SleepStage::Wake];

/// Deep-eligibility HR-flatness percentile gate of the shipped recipe.
pub const DEEP_GATE_THRESH: f64 = Params::SHIPPED.deep_gate_thresh;

/// One 30 s epoch's recipe features. `None` means "no measurement" (scored neutral).
struct Epoch {
    start: i64,
    hr: Option<f64>,
    hr_var: Option<f64>,
    hr_flat11: Option<f64>,
    move_frac: Option<f64>,
    jerk_max: f64,
    resp_reg: Option<f64>,
    /// Inter-epoch rotation in degrees. Frame-invariant, so it survives the strap being re-donned.
    turn: Option<f64>,
    clock: f64,
    jerk_scale: f64,
}

/// Stage a detected in-bed span with the shipped V2 recipe; segments tile `[start, end]`.
pub fn stage(input: &SleepInput) -> Vec<StageSegment> {
    stage_with(input, &Params::SHIPPED)
}

/// One night's epoch features, already extracted. Only `jerk_move_mult` changes these, so a sweep over
/// the emission/prior/transition axes can extract once and re-label many times.
pub struct Prepared {
    feats: Vec<Epoch>,
    start: i64,
    end: i64,
}

/// Extract a night's epoch features under `p`. Pair with [`stage_prepared`].
pub fn prepare(input: &SleepInput, p: &Params) -> Prepared {
    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let mut hr = input.hr.clone();
    hr.sort_by_key(|h| h.ts);
    let mut rr = flatten_rr(&input.rr);
    rr.sort_by_key(|a| a.0);
    let feats = features(input.start, input.end, &grav, &hr, &rr, p);
    Prepared { feats, start: input.start, end: input.end }
}

/// Label already-extracted features under `p`; segments tile the prepared span.
pub fn stage_prepared(prep: &Prepared, p: &Params) -> Vec<StageSegment> {
    segments_of(prep, &stage_epochs(&prep.feats, p))
}

/// The per-epoch log-emissions a prepared night hands the decoder, in [`STAGE_ORDER`] columns.
/// [`viterbi`] over these reproduces [`stage_prepared`]'s labels exactly, so a caller can score the path
/// search and the emissions feeding it apart.
pub fn emissions_prepared(prep: &Prepared, p: &Params) -> Vec<[f64; 4]> {
    final_emissions(&prep.feats, p)
}

/// The anchor [`emissions_prepared`] resolves. Split out so `pipeline` can record it as its own step:
/// resolving it runs a probe decode, and a linear step order otherwise hides that.
pub(super) fn anchor_of(prep: &Prepared, p: &Params) -> Anchor {
    resolve_anchor(&prep.feats, p)
}

/// [`emissions_prepared`] with the anchor supplied. At [`anchor_of`] the two are identical.
pub(super) fn emissions_at(prep: &Prepared, p: &Params, anchor: Anchor) -> Vec<[f64; 4]> {
    emissions(&prep.feats, p, anchor)
}

/// The unix second each prepared epoch opens at, index-for-index with [`emissions_prepared`]. An epoch
/// with neither HR nor gravity is dropped, so the sequence can skip and a reference has to be aligned by
/// time rather than by position.
pub fn epoch_starts(prep: &Prepared) -> Vec<i64> {
    prep.feats.iter().map(|f| f.start).collect()
}

/// Tile the prepared span from one label per epoch — staging's last step, so a caller that decoded its own
/// path gets the segments [`stage_prepared`] would have returned. `labels` must be one per prepared epoch.
pub fn segments_of(prep: &Prepared, labels: &[SleepStage]) -> Vec<StageSegment> {
    if prep.feats.is_empty() {
        return vec![StageSegment { start: prep.start, end: prep.end, stage: SleepStage::Light }];
    }
    segments_from(&prep.feats, labels, prep.start, prep.end)
}

/// Stage with an explicit recipe. The tuning path; `stage` is what everything else calls.
pub fn stage_with(input: &SleepInput, p: &Params) -> Vec<StageSegment> {
    stage_prepared(&prepare(input, p), p)
}

/// Tile `[start, end]` from one label per epoch, merging equal-stage neighbours.
fn segments_from(feats: &[Epoch], labels: &[SleepStage], start: i64, end: i64) -> Vec<StageSegment> {
    let mut segments: Vec<StageSegment> = Vec::new();
    let n = feats.len();
    for (i, f) in feats.iter().enumerate() {
        let stage = labels[i];
        let seg_start = if i == 0 { start } else { f.start };
        let seg_end = if i == n - 1 { end } else { feats[i + 1].start };
        match segments.last_mut() {
            Some(last) if last.stage == stage => last.end = seg_end,
            _ => segments.push(StageSegment { start: seg_start, end: seg_end, stage }),
        }
    }
    segments
}

fn features(
    start: i64,
    end: i64,
    grav: &[AccelSample],
    hr: &[HrSample],
    rr: &[(i64, f64)],
    p: &Params,
) -> Vec<Epoch> {
    if end <= start {
        return Vec::new();
    }
    let span = (end - start).max(1) as f64;

    // Per-second HR mean.
    let mut hr_sum: HashMap<i64, f64> = HashMap::new();
    let mut hr_cnt: HashMap<i64, i64> = HashMap::new();
    for s in hr {
        *hr_sum.entry(s.ts).or_insert(0.0) += s.bpm as f64;
        *hr_cnt.entry(s.ts).or_insert(0) += 1;
    }
    let mut sec_hr: HashMap<i64, f64> = HashMap::with_capacity(hr_sum.len());
    for (k, v) in &hr_sum {
        sec_hr.insert(*k, v / hr_cnt[k] as f64);
    }

    // Per-second gravity mean (x, y, z).
    let mut gx: HashMap<i64, f64> = HashMap::new();
    let mut gy: HashMap<i64, f64> = HashMap::new();
    let mut gz: HashMap<i64, f64> = HashMap::new();
    let mut gc: HashMap<i64, i64> = HashMap::new();
    for g in grav {
        *gx.entry(g.ts).or_insert(0.0) += g.x;
        *gy.entry(g.ts).or_insert(0.0) += g.y;
        *gz.entry(g.ts).or_insert(0.0) += g.z;
        *gc.entry(g.ts).or_insert(0) += 1;
    }
    let mut sec_g: HashMap<i64, (f64, f64, f64)> = HashMap::with_capacity(gc.len());
    for (k, c) in &gc {
        let d = *c as f64;
        sec_g.insert(*k, (gx[k] / d, gy[k] / d, gz[k] / d));
    }

    // R-R bucketed by second (for the RSA window).
    let mut rr_by: HashMap<i64, Vec<f64>> = HashMap::new();
    for (ts, ms) in rr {
        rr_by.entry(*ts).or_default().push(*ms);
    }

    // Prefix sums over the integer-second HR axis → O(1) windowed population std.
    let (axis_lo, sum_px, sum_sq_px, cnt_px) = if sec_hr.is_empty() {
        (0i64, vec![0.0], vec![0.0], vec![0i64])
    } else {
        let lo = *sec_hr.keys().min().unwrap();
        let hi = *sec_hr.keys().max().unwrap();
        let len = (hi - lo + 1) as usize;
        let mut sp = vec![0.0; len + 1];
        let mut sqp = vec![0.0; len + 1];
        let mut cp = vec![0i64; len + 1];
        for i in 0..len {
            let v = sec_hr.get(&(lo + i as i64)).copied();
            sp[i + 1] = sp[i] + v.unwrap_or(0.0);
            sqp[i + 1] = sqp[i] + v.map(|x| x * x).unwrap_or(0.0);
            cp[i + 1] = cp[i] + if v.is_some() { 1 } else { 0 };
        }
        (lo, sp, sqp, cp)
    };
    let axis_hi_excl = axis_lo + (cnt_px.len() as i64 - 1);

    let std_of_seconds = |lo: i64, hi: i64| -> Option<f64> {
        if cnt_px.len() <= 1 {
            return None;
        }
        let q_lo = lo.clamp(axis_lo, axis_hi_excl);
        let q_hi = hi.clamp(axis_lo, axis_hi_excl);
        if q_hi <= q_lo {
            return None;
        }
        let a = (q_lo - axis_lo) as usize;
        let b = (q_hi - axis_lo) as usize;
        let n = cnt_px[b] - cnt_px[a];
        if n < 2 {
            return None;
        }
        let sum = sum_px[b] - sum_px[a];
        let sum_sq = sum_sq_px[b] - sum_sq_px[a];
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        Some(if variance < 0.0 { 0.0 } else { variance }.sqrt())
    };

    // PASS 1 — per-epoch quantities except moveFrac; pool every per-second jerk.
    struct Raw {
        start: i64,
        hr: Option<f64>,
        hr_var: Option<f64>,
        hr_flat11: Option<f64>,
        jerks: Vec<f64>,
        gap_sec: i64,
        jerk_max: f64,
        resp_reg: Option<f64>,
        clock: f64,
    }
    let mut raws: Vec<Raw> = Vec::new();
    let mut all_jerks: Vec<f64> = Vec::new();
    let first_e = ((start + 29) / 30) * 30;
    let mut e = first_e;
    while e < end {
        let mut hrs: Vec<f64> = Vec::new();
        let mut gseq: Vec<(f64, f64, f64)> = Vec::new();
        let mut s = e;
        while s < e + 30 {
            if let Some(v) = sec_hr.get(&s) {
                hrs.push(*v);
            }
            if let Some(v) = sec_g.get(&s) {
                gseq.push(*v);
            }
            s += 1;
        }
        if hrs.is_empty() && gseq.is_empty() {
            e += 30;
            continue;
        }

        let mut jerks: Vec<f64> = Vec::new();
        let mut i = 1usize;
        while i < gseq.len().max(1) {
            let a = gseq[i - 1];
            let b = gseq[i];
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            let dz = a.2 - b.2;
            jerks.push((dx * dx + dy * dy + dz * dz).sqrt());
            i += 1;
        }
        all_jerks.extend_from_slice(&jerks);
        let jerk_max = jerks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jerk_max = if jerks.is_empty() { 0.0 } else { jerk_max };

        let hr_mean = if hrs.is_empty() { None } else { Some(hrs.iter().sum::<f64>() / hrs.len() as f64) };
        let hr_var = std_of_seconds(e - 150, e + 30 + 150);
        let hr_flat11 = std_of_seconds(e - 330, e + 30 + 360);

        let mut beats: Vec<(f64, f64)> = Vec::new();
        let mut bs = e - 90;
        while bs < e + 120 {
            if let Some(vs) = rr_by.get(&bs) {
                for v in vs {
                    beats.push((bs as f64, v.clamp(300.0, 2000.0)));
                }
            }
            bs += 1;
        }
        beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        let resp_reg = resp_regularity(&beats);

        raws.push(Raw {
            start: e,
            hr: hr_mean,
            hr_var,
            hr_flat11,
            gap_sec: (gseq.len() as i64 - 1).max(1),
            jerk_max,
            resp_reg,
            clock: (e + 15 - start) as f64 / span,
            jerks,
        });
        e += 30;
    }

    // Rotation between consecutive epochs, from the same gravity the jerk features read. Binned on the
    // epoch grid and read by epoch start, because `raws` skips every epoch with neither HR nor gravity.
    let turns = {
        let last = raws.last().map_or(first_e, |r| r.start + 30);
        let post = super::posture::posture_series(grav, first_e, last, 30);
        let t = super::posture::turn_series(&post);
        raws.iter()
            .map(|r| t.get(((r.start - first_e) / 30) as usize).copied().flatten())
            .collect::<Vec<_>>()
    };

    let jerk_scale = if all_jerks.is_empty() { 1e-6 } else { median(&all_jerks) };
    let move_thr = jerk_scale * p.jerk_move_mult;

    // PASS 2 — move fraction against the night-relative threshold.
    raws.into_iter()
        .enumerate()
        .map(|(idx, r)| {
            // No gravity in the epoch = no motion evidence; absent, not "perfectly still".
            let observed = !r.jerks.is_empty();
            let moves = r.jerks.iter().filter(|&&j| j > move_thr).count();
            Epoch {
                start: r.start,
                hr: r.hr,
                hr_var: r.hr_var,
                hr_flat11: r.hr_flat11,
                move_frac: observed.then(|| moves as f64 / r.gap_sec as f64),
                jerk_max: r.jerk_max,
                resp_reg: r.resp_reg,
                turn: turns.get(idx).copied().flatten(),
                clock: r.clock,
                jerk_scale,
            }
        })
        .collect()
}

/// RSA respiration regularity: tachogram → 4 Hz resample → detrend → band-limited DFT peak/sum over the
/// 0.15–0.40 Hz band. Higher = more regular breathing. `None` when there are too few beats.
pub fn resp_regularity(beats: &[(f64, f64)]) -> Option<f64> {
    if beats.len() < 12 {
        return None;
    }
    let t0 = beats[0].0;
    let t_n = beats[beats.len() - 1].0;
    if t_n <= t0 {
        return None;
    }
    let n = ((t_n - t0) / 0.25 - 1e-9).ceil() as i64;
    if n < 16 {
        return None;
    }
    let n = n as usize;

    let mut y = vec![0.0f64; n];
    let mut seg = 0usize;
    for (i, yi) in y.iter_mut().enumerate() {
        let t = t0 + 0.25 * i as f64;
        while seg < beats.len() - 2 && beats[seg + 1].0 < t {
            seg += 1;
        }
        let ta = beats[seg].0;
        let tb = beats[seg + 1].0;
        let va = beats[seg].1;
        let vb = beats[seg + 1].1;
        *yi = if tb <= ta { va } else { va + ((t - ta) / (tb - ta)).clamp(0.0, 1.0) * (vb - va) };
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    for v in y.iter_mut() {
        *v -= mean;
    }

    let k_lo = (0.15 * 0.25 * n as f64).ceil() as i64;
    let k_hi = (0.40 * 0.25 * n as f64).floor() as i64;
    if k_hi < k_lo || k_lo < 0 {
        return None;
    }
    let mut max_p = 0.0;
    let mut sum_p = 0.0;
    for k in k_lo..=k_hi {
        let mut re = 0.0;
        let mut im = 0.0;
        let w = -2.0 * PI * k as f64 / n as f64;
        for (j, &yj) in y.iter().enumerate() {
            let a = w * j as f64;
            re += yj * a.cos();
            im += yj * a.sin();
        }
        let p = re * re + im * im;
        sum_p += p;
        if p > max_p {
            max_p = p;
        }
    }
    if sum_p == 0.0 {
        None
    } else {
        Some(max_p / sum_p)
    }
}

/// Where the time-of-night priors are measured from. `Probe` is the first pass of a two-pass staging: it
/// has no labels yet, so the early-REM guard is off and cannot bias the onset it is looking for.
#[derive(Clone, Copy)]
pub(super) enum Anchor {
    Probe,
    /// Read from the window, which is what a one-pass staging has.
    Window,
    /// Read from the epoch sleep began at.
    Onset(usize),
}

/// Soft sleep-cycle prior: deep concentrated early (decays), REM raised through the night less `guard`.
fn cycle_prior(c: f64, guard: f64, p: &Params) -> [f64; 4] {
    let mut pr = [0.0; 4];
    pr[DEEP] = p.cycle_deep_scale * (1.0 - c / p.cycle_deep_decay).max(0.0);
    pr[REM] = p.cycle_rem_scale * c.min(p.cycle_rem_ramp_cap) - guard;
    pr
}

/// Time-of-night for the cycle prior. `clock` is a fraction of the WINDOW, so rebasing the pair against
/// the onset epoch's own value re-expresses it as a fraction of the sleep period in the same units.
fn cycle_clock(c: f64, feats: &[Epoch], anchor: Anchor, p: &Params) -> f64 {
    let Anchor::Onset(o) = anchor else { return c };
    if !p.cycle_clock_from_onset {
        return c;
    }
    let c0 = feats[o].clock;
    if c0 >= 1.0 { c } else { ((c - c0) / (1.0 - c0)).clamp(0.0, 1.0) }
}

/// The early-REM suppression for one epoch. Zero `cycle_rem_onset_minutes` keeps the step at a fraction
/// of the session; a positive one grades the same magnitude to zero over that many minutes past onset,
/// clamped so a pre-onset epoch is never penalised harder than onset itself.
fn rem_guard(idx: usize, c: f64, anchor: Anchor, p: &Params) -> f64 {
    match anchor {
        Anchor::Probe => 0.0,
        Anchor::Onset(o) if p.cycle_rem_onset_minutes > 0.0 => {
            let minutes = (idx as f64 - o as f64) * EPOCH_MIN;
            p.cycle_rem_early_penalty * (1.0 - minutes / p.cycle_rem_onset_minutes).clamp(0.0, 1.0)
        }
        _ if c < p.cycle_rem_early_frac => p.cycle_rem_early_penalty,
        _ => 0.0,
    }
}

/// First epoch of the earliest run of at least `SUSTAINED_ONSET_EPOCHS` consecutive non-wake labels.
fn sustained_onset(labels: &[SleepStage]) -> Option<usize> {
    let mut run = 0usize;
    for (i, l) in labels.iter().enumerate() {
        run = if *l == SleepStage::Wake { 0 } else { run + 1 };
        if run == SUSTAINED_ONSET_EPOCHS {
            return Some(i + 1 - SUSTAINED_ONSET_EPOCHS);
        }
    }
    None
}

/// Motion-quiescent: movement was OBSERVED and was none, with peak jerk at/below the night floor × the
/// gate multiplier. An epoch with no gravity cannot be quiescent — absence is not stillness.
fn motion_quiescent(f: &Epoch, p: &Params) -> bool {
    f.move_frac.is_some_and(|m| m <= 0.0) && f.jerk_max <= f.jerk_scale * p.jerk_gate_mult
}

fn dz(z: f64, deadzone: f64) -> f64 {
    if deadzone <= 0.0 {
        z
    } else if z > deadzone {
        z - deadzone
    } else if z < -deadzone {
        z + deadzone
    } else {
        0.0
    }
}

/// Viterbi most-likely path over the per-epoch log-emissions with the sticky transition matrix and a
/// uniform start. Columns are [`STAGE_ORDER`]; ties resolve to the earlier stage. A zero transition is
/// floored at 1e-9 rather than forbidden, so no path is unreachable.
#[allow(clippy::needless_range_loop)]
pub fn viterbi(em_seq: &[[f64; 4]], transition: &[[f64; 4]; 4]) -> Vec<SleepStage> {
    if em_seq.is_empty() {
        return Vec::new();
    }
    let mut log_t = [[0.0f64; 4]; 4];
    for (fi, row) in transition.iter().enumerate() {
        for (ti, &v) in row.iter().enumerate() {
            log_t[fi][ti] = v.max(1e-9).ln();
        }
    }
    let mut v = em_seq[0];
    let mut back: Vec<[usize; 4]> = Vec::new();
    for em in &em_seq[1..] {
        let mut new_v = [0.0f64; 4];
        let mut bp = [0usize; 4];
        for s in 0..4 {
            let mut best_prev = 0usize;
            let mut best_val = v[0] + log_t[0][s];
            for p in 1..4 {
                let value = v[p] + log_t[p][s];
                if value > best_val {
                    best_val = value;
                    best_prev = p;
                }
            }
            new_v[s] = best_val + em[s];
            bp[s] = best_prev;
        }
        v = new_v;
        back.push(bp);
    }
    let mut last = 0usize;
    let mut last_v = v[0];
    for s in 1..4 {
        if v[s] > last_v {
            last_v = v[s];
            last = s;
        }
    }
    let mut path = vec![last];
    for bp in back.iter().rev() {
        last = bp[last];
        path.push(last);
    }
    path.reverse();
    path.into_iter().map(idx_to_stage).collect()
}

fn idx_to_stage(i: usize) -> SleepStage {
    match i {
        DEEP => SleepStage::Deep,
        REM => SleepStage::Rem,
        LIGHT => SleepStage::Light,
        _ => SleepStage::Wake,
    }
}

/// The log-emissions the decoder is handed, under whichever anchor `p` selects. An onset-anchored prior
/// needs a staging to find the onset, so it stages once with the guard off first.
fn final_emissions(feats: &[Epoch], p: &Params) -> Vec<[f64; 4]> {
    emissions(feats, p, resolve_anchor(feats, p))
}

/// Which time-of-night anchor `p` selects. An onset-anchored prior needs a staging to find the
/// onset, so it stages once with the guard off first.
fn resolve_anchor(feats: &[Epoch], p: &Params) -> Anchor {
    if p.cycle_rem_onset_minutes > 0.0 || p.cycle_clock_from_onset {
        let probe = viterbi(&emissions(feats, p, Anchor::Probe), &p.transition);
        return Anchor::Onset(sustained_onset(&probe).unwrap_or(0));
    }
    Anchor::Window
}

/// Run the full recipe over a night's epochs and return one stage label per epoch. All normalisation
/// (z-scores, the HR-flatness percentile) is within the night.
fn stage_epochs(feats: &[Epoch], p: &Params) -> Vec<SleepStage> {
    if feats.is_empty() {
        return Vec::new();
    }
    viterbi(&final_emissions(feats, p), &p.transition)
}


// Slot each weight owns, in [`WEIGHT_NAMES`], in [`weights_of`] and in the [`Terms`] design columns.
// Nothing but these constants ties the three together, so all three read them.
const W_DEEP_HRV: usize = 0;
const W_DEEP_HR: usize = 1;
const W_DEEP_MOTION: usize = 2;
const W_DEEP_GATE_SLOPE: usize = 3;
const W_REM_HRV: usize = 4;
const W_REM_MOTION: usize = 5;
const W_REM_HR: usize = 6;
const W_AWAKE_MOTION: usize = 7;
const W_AWAKE_HRV: usize = 8;
const W_AWAKE_HR: usize = 9;
const W_AWAKE_TURN: usize = 10;
const W_RESP: usize = 11;

/// The twelve emission weights, in the order [`emission_terms`] lays out its design.
pub const WEIGHT_NAMES: [&str; 12] = {
    let mut n = [""; 12];
    n[W_DEEP_HRV] = "deep_hrv";
    n[W_DEEP_HR] = "deep_hr";
    n[W_DEEP_MOTION] = "deep_motion";
    n[W_DEEP_GATE_SLOPE] = "deep_gate_slope";
    n[W_REM_HRV] = "rem_hrv";
    n[W_REM_MOTION] = "rem_motion";
    n[W_REM_HR] = "rem_hr";
    n[W_AWAKE_MOTION] = "awake_motion";
    n[W_AWAKE_HRV] = "awake_hrv";
    n[W_AWAKE_HR] = "awake_hr";
    n[W_AWAKE_TURN] = "awake_turn";
    n[W_RESP] = "resp_weight";
    n
};

/// The weights [`WEIGHT_NAMES`] refers to, read off `p` into the slot each name owns.
pub fn weights_of(p: &Params) -> [f64; 12] {
    let mut w = [0.0f64; 12];
    w[W_DEEP_HRV] = p.deep_hrv;
    w[W_DEEP_HR] = p.deep_hr;
    w[W_DEEP_MOTION] = p.deep_motion;
    w[W_DEEP_GATE_SLOPE] = p.deep_gate_slope;
    w[W_REM_HRV] = p.rem_hrv;
    w[W_REM_MOTION] = p.rem_motion;
    w[W_REM_HR] = p.rem_hr;
    w[W_AWAKE_MOTION] = p.awake_motion;
    w[W_AWAKE_HRV] = p.awake_hrv;
    w[W_AWAKE_HR] = p.awake_hr;
    w[W_AWAKE_TURN] = p.awake_turn;
    w[W_RESP] = p.resp_weight;
    w
}

/// The emission split into `design[e][c][j]`, what weight `j` contributes to class `c`, and
/// `fixed[e][c]`, the rest. `clamped[e]` marks the awake-cardiac `min(0.0)`, which is neither.
pub struct Terms {
    pub design: Vec<[[f64; 12]; 4]>,
    pub fixed: Vec<[f64; 4]>,
    pub clamped: Vec<bool>,
}

impl Terms {
    /// Rebuild one epoch's emission from a weight vector. At `weights_of(p)` it is exactly what
    /// [`emissions_prepared`] returns, because that path is this one.
    pub fn emission(&self, e: usize, w: &[f64; 12]) -> [f64; 4] {
        let (d, f) = (&self.design[e], &self.fixed[e]);
        let mut em = [0.0f64; 4];
        for c in 0..4 {
            let mut acc = f[c];
            for j in 0..12 {
                // The awake cardiac pair is summed first so the clamp can act on the pair.
                if c == AWAKE && (j == W_AWAKE_HRV || j == W_AWAKE_HR) {
                    continue;
                }
                acc += w[j] * d[c][j];
            }
            if c == AWAKE {
                let card = w[W_AWAKE_HRV] * d[c][W_AWAKE_HRV] + w[W_AWAKE_HR] * d[c][W_AWAKE_HR];
                acc += if self.clamped[e] { card.min(0.0) } else { card };
            }
            em[c] = acc;
        }
        em
    }
}

/// Decompose a prepared night's emissions into [`Terms`], under the same anchor
/// [`emissions_prepared`] resolves. An onset anchor is itself staged under `p`'s weights, so the
/// cycle prior baked into `fixed` holds for those weights alone.
pub fn emission_terms(prep: &Prepared, p: &Params) -> Terms {
    terms(&prep.feats, p, resolve_anchor(&prep.feats, p))
}


/// The recipe itself, with the weighted parts kept apart from the rest. [`emissions`] is this summed
/// at `p`'s own weights, so the emission is written here and nowhere else.
fn terms(feats: &[Epoch], p: &Params, anchor: Anchor) -> Terms {
    let blp = p.base_log_prior();
    let zhr = ZScore::build(&feats.iter().map(|f| f.hr).collect::<Vec<_>>());
    let zhv = ZScore::build(&feats.iter().map(|f| f.hr_var).collect::<Vec<_>>());
    let zmv = ZScore::build(&feats.iter().map(|f| f.move_frac).collect::<Vec<_>>());
    let zrg = ZScore::build(&feats.iter().map(|f| f.resp_reg).collect::<Vec<_>>());
    // turn spans three orders of magnitude within a night (p50 ~0.1 deg, max ~140), so a z-score hands
    // a small NEGATIVE to the great majority of epochs and a huge positive to a handful, which lowers
    // AWAKE across the whole night. Rank it instead, the way hr_flat11 is ranked.
    let mut tsorted: Vec<f64> = feats.iter().filter_map(|f| f.turn).collect();
    tsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut fsorted: Vec<f64> = feats.iter().filter_map(|f| f.hr_flat11).collect();
    fsorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |sorted: &[f64], value: Option<f64>| -> f64 {
        match value {
            Some(v) if !sorted.is_empty() => {
                sorted.partition_point(|s| *s <= v) as f64 / sorted.len() as f64
            }
            _ => 0.5,
        }
    };

    let mut out = Terms { design: Vec::new(), fixed: Vec::new(), clamped: Vec::new() };
    for (i, f) in feats.iter().enumerate() {
        let zhrv = zhr.apply(f.hr);
        let zhvv = zhv.apply(f.hr_var);
        let zmvv = zmv.apply(f.move_frac);
        let hinge = (pct(&fsorted, f.hr_flat11) - p.deep_gate_thresh).max(0.0);
        // Trust the awake cardiac term where R-R backs it; clamp it where only heart rate does.
        let rr_backed = p.clamp_only_without_rr && f.resp_reg.is_some();
        // Rotation is evidence of wake on its own terms, and it survives the stillness clamp. Centred
        // on the median so it is symmetric: a still epoch pushes AWAKE down as much as a rotating one
        // pushes it up, and a night with no rotation at all is left alone.
        let tp = (pct(&tsorted, f.turn) - 0.5) * 2.0;
        let rz = zrg.apply(f.resp_reg);

        let mut d = [[0.0f64; 12]; 4];
        d[DEEP][W_DEEP_HRV] = zhvv;
        d[DEEP][W_DEEP_HR] = zhrv;
        d[DEEP][W_DEEP_MOTION] = zmvv;
        d[DEEP][W_DEEP_GATE_SLOPE] = -hinge;
        d[REM][W_REM_HRV] = zhvv;
        d[REM][W_REM_MOTION] = zmvv;
        d[REM][W_REM_HR] = zhrv;
        d[AWAKE][W_AWAKE_MOTION] = zmvv;
        d[AWAKE][W_AWAKE_HRV] = dz(zhvv, p.awake_deadzone);
        d[AWAKE][W_AWAKE_HR] = dz(zhrv, p.awake_deadzone);
        d[AWAKE][W_AWAKE_TURN] = tp;
        d[DEEP][W_RESP] = rz;
        d[REM][W_RESP] = -rz;

        let mut fx = blp;
        let pr = cycle_prior(cycle_clock(f.clock, feats, anchor, p), rem_guard(i, f.clock, anchor, p), p);
        for (s, v) in pr.iter().enumerate() {
            fx[s] += v;
        }
        if f.jerk_max > f.jerk_scale * p.jerk_gate_mult {
            fx[AWAKE] += p.motion_gate_boost;
        }

        out.design.push(d);
        out.fixed.push(fx);
        // Stillness silences the cardiac term - unless the heart is running well above this night's own
        // mean, which is a wind-down and not sleep. INFINITY restores the unconditional clamp.
        out.clamped.push(motion_quiescent(f, p) && zhrv < p.quiescent_hr_z_max && !rr_backed);
    }
    out
}

/// Per-epoch log-emissions under `p`, with the time-of-night priors read from `anchor`. [`terms`] holds
/// the recipe; this is it summed at `p`'s own twelve weights.
fn emissions(feats: &[Epoch], p: &Params, anchor: Anchor) -> Vec<[f64; 4]> {
    let t = terms(feats, p, anchor);
    let w = weights_of(p);
    (0..t.design.len()).map(|e| t.emission(e, &w)).collect()
}

#[cfg(test)]
mod terms_tests {
    use super::*;
    use crate::sleep::{AccelSample, HrSample, RrRun, SleepInput};

    /// `Params::SHIPPED` with the one weight `name` refers to moved by 1.0. The match is the only
    /// statement of which field each name means, so a name that reaches no field is a failure.
    fn bumped(name: &str) -> Params {
        let mut p = Params::SHIPPED;
        match name {
            "deep_hrv" => p.deep_hrv += 1.0,
            "deep_hr" => p.deep_hr += 1.0,
            "deep_motion" => p.deep_motion += 1.0,
            "deep_gate_slope" => p.deep_gate_slope += 1.0,
            "rem_hrv" => p.rem_hrv += 1.0,
            "rem_motion" => p.rem_motion += 1.0,
            "rem_hr" => p.rem_hr += 1.0,
            "awake_motion" => p.awake_motion += 1.0,
            "awake_hrv" => p.awake_hrv += 1.0,
            "awake_hr" => p.awake_hr += 1.0,
            "awake_turn" => p.awake_turn += 1.0,
            "resp_weight" => p.resp_weight += 1.0,
            other => panic!("{other} is named but reaches no Params field"),
        }
        p
    }

    /// Ten minutes of flat HR with the wrist rolling onto a new face twice, so `turn` varies over the
    /// night instead of holding one value the rank transform would flatten.
    pub(super) fn rotating_night() -> SleepInput {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..600).map(|i| HrSample { ts: start + i, bpm: 60 }).collect();
        let accel: Vec<AccelSample> = (0..600)
            .map(|i| match i / 30 {
                5 | 12 => AccelSample { ts: start + i, x: 1.0, y: 0.0, z: 0.0 },
                e if e > 12 => AccelSample { ts: start + i, x: 0.0, y: 1.0, z: 0.0 },
                _ => AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 },
            })
            .collect();
        SleepInput { start, end: start + 600, hr, rr: Vec::new(), accel }
    }

    /// `secs` seconds of one beat each, 1000 ms swinging 40 ms at 0.25 Hz. The swing is what carries
    /// the respiration term: a flat tachogram has no power and yields none at all.
    pub(super) fn rsa_rr(start: i64, secs: i64) -> Vec<RrRun> {
        (0..secs)
            .map(|i| {
                let ms = 1000.0 + 40.0 * (2.0 * PI * 0.25 * i as f64).sin();
                RrRun { ts: start + i, intervals: vec![ms as u16] }
            })
            .collect()
    }

    /// `awake_turn` ships at 0.0, so no staged night can see its column - pin the column itself. It is
    /// the night-rank of `turn` centred on the median and spread to [-1, 1], and it reaches AWAKE alone.
    #[test]
    fn the_turn_column_is_the_centred_night_rank_of_the_rotation() {
        let prep = prepare(&rotating_night(), &Params::SHIPPED);
        let t = emission_terms(&prep, &Params::SHIPPED);
        let turns: Vec<f64> = prep.feats.iter().filter_map(|f| f.turn).collect();
        assert!(turns.len() >= 10, "the fixture must carry rotation on most epochs");
        assert!(turns.iter().cloned().fold(f64::MIN, f64::max) > 1.0, "and it must actually rotate");

        let mut nonzero = 0usize;
        for (e, f) in prep.feats.iter().enumerate() {
            let rank = match f.turn {
                Some(v) => turns.iter().filter(|s| **s <= v).count() as f64 / turns.len() as f64,
                None => 0.5,
            };
            assert_eq!((rank - 0.5) * 2.0, t.design[e][AWAKE][W_AWAKE_TURN], "epoch {e}");
            for c in [DEEP, REM, LIGHT] {
                assert_eq!(0.0, t.design[e][c][W_AWAKE_TURN], "epoch {e}: turn reaches AWAKE alone");
            }
            nonzero += usize::from(t.design[e][AWAKE][W_AWAKE_TURN] != 0.0);
        }
        assert!(nonzero >= 5, "a column of zeros would agree with anything");
    }

    /// Each named weight must land in its own slot and move the stages its name claims, no others. Two
    /// shipped weights share a value with another, so only the slot and the stage part a swapped pair.
    #[test]
    fn every_named_weight_moves_exactly_the_stages_its_name_claims() {
        let (start, end) = (0i64, 3600i64);
        let hr: Vec<HrSample> = (0..3600)
            .map(|t| HrSample { ts: t, bpm: (58.0 + 9.0 * (t as f64 / 300.0).sin()) as u16 })
            .collect();
        let accel: Vec<AccelSample> = (0..3600)
            .map(|t| {
                let a = if t % 300 < 40 { 0.5 * ((t % 7) as f64) / 7.0 } else { 0.0 };
                AccelSample { ts: t, x: a, y: 0.0, z: (1.0f64 - a * a).max(0.0).sqrt() }
            })
            .collect();
        let rr: Vec<RrRun> =
            (0..600).map(|k| RrRun { ts: k * 6, intervals: vec![950, 1040, 990] }).collect();
        let prep = prepare(&SleepInput { start, end, hr, rr, accel }, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        let base = weights_of(&Params::SHIPPED);
        for (j, &name) in WEIGHT_NAMES.iter().enumerate() {
            let w = weights_of(&bumped(name));
            assert_ne!(base[j], w[j], "{name} must be the weight in slot {j}");
            assert_eq!(1, (0..12).filter(|k| base[*k] != w[*k]).count(), "{name} moved another slot");
            let want: &[usize] = match name.split('_').next().expect("a named weight") {
                "deep" => &[DEEP],
                "rem" => &[REM],
                "awake" => &[AWAKE],
                _ => &[DEEP, REM],
            };
            for c in 0..4 {
                let moved = (0..terms.design.len())
                    .any(|e| (terms.emission(e, &base)[c] - terms.emission(e, &w)[c]).abs() > 1e-9);
                assert_eq!(want.contains(&c), moved, "{name} (slot {j}) against class {c}");
            }
        }
    }

    /// The shipped recipe always resolves an onset anchor over a fully populated night. This is the
    /// other side: window-anchored, with an HR gap, a gravity gap and an R-R gap.
    #[test]
    fn a_window_anchored_night_with_absent_channels_decomposes() {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..900)
            .filter(|i| !(300..600).contains(i))
            .map(|i| HrSample { ts: start + i, bpm: 58 + (i % 5) as u16 })
            .collect();
        let accel: Vec<AccelSample> =
            (0..600).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect();
        let input = SleepInput { start, end: start + 900, hr, rr: rsa_rr(start, 300), accel };

        let p =
            Params { cycle_rem_onset_minutes: 0.0, cycle_clock_from_onset: false, ..Params::SHIPPED };
        let prep = prepare(&input, &p);
        // A gap is a channel that comes and goes. Asserting absence alone would also hold for a
        // channel missing all night, so each one is asserted on both sides.
        fn gapped(feats: &[Epoch], present: impl Fn(&Epoch) -> bool) -> bool {
            let n = feats.iter().filter(|f| present(f)).count();
            n > 0 && n < feats.len()
        }
        assert!(gapped(&prep.feats, |f| f.hr.is_some()), "the fixture must carry an HR gap");
        assert!(gapped(&prep.feats, |f| f.move_frac.is_some()), "and a gravity gap");
        assert!(gapped(&prep.feats, |f| f.resp_reg.is_some()), "and an R-R gap");

        let t = emission_terms(&prep, &p);
        assert!(matches!(resolve_anchor(&prep.feats, &p), Anchor::Window),
            "no onset minutes and no onset clock is window-anchored");
        assert!(matches!(resolve_anchor(&prep.feats, &Params::SHIPPED), Anchor::Onset(_)),
            "the shipped prior is anchored on a staging, so its fixed part is weight-dependent");
        let clock_only = Params { cycle_clock_from_onset: true, ..p };
        assert!(matches!(resolve_anchor(&prep.feats, &clock_only), Anchor::Onset(_)),
            "an onset clock resolves an onset on its own, with no onset minutes to ask for one");
        assert_eq!(prep.feats.len(), t.design.len(), "one design row per epoch");
        let w = weights_of(&p);
        for (e, f) in prep.feats.iter().enumerate() {
            assert!(t.emission(e, &w).iter().all(|v| v.is_finite()), "epoch {e} is not finite");
            if f.move_frac.is_none() {
                assert!(!t.clamped[e], "epoch {e}: an absent accelerometer cannot assert stillness");
            }
        }
    }

    /// Twenty minutes of flat gravity, so every epoch is motion-quiescent and clamped, over a heart rate
    /// that is low and swinging for the first half and high and steady for the second. That parts the two
    /// cardiac z-scores in sign, which is what the pair clamp turns on.
    fn still_night_with_opposed_cardiac_terms() -> SleepInput {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..1200)
            .map(|i| {
                let swing = 55.0 + 18.0 * (2.0 * PI * i as f64 / 60.0).sin();
                HrSample { ts: start + i, bpm: if i < 600 { swing as u16 } else { 78 } }
            })
            .collect();
        let accel: Vec<AccelSample> =
            (0..1200).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect();
        SleepInput { start, end: start + 1200, hr, rr: Vec::new(), accel }
    }

    /// The other side of that night: quiet and R-R-backed for the first half, restless and hot for the
    /// second, so the night carries unclamped epochs and a respiration column that is not flat. Motion
    /// is a sixth of the seconds, which keeps the night's median jerk at zero.
    fn restless_night_with_respiration() -> SleepInput {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..1200)
            .map(|i| {
                let bpm = if i < 600 { 52 + (i % 3) as u16 } else { 76 + (i % 5) as u16 };
                HrSample { ts: start + i, bpm }
            })
            .collect();
        let accel: Vec<AccelSample> = (0..1200)
            .map(|i| {
                let a = if i >= 600 && i % 30 < 10 { 0.4 * ((i % 7) as f64) / 7.0 } else { 0.0 };
                AccelSample { ts: start + i, x: a, y: 0.0, z: (1.0f64 - a * a).max(0.0).sqrt() }
            })
            .collect();
        SleepInput { start, end: start + 1200, hr, rr: rsa_rr(start, 600), accel }
    }

    /// The clamp acts on the summed awake cardiac PAIR, the one thing in [`Terms`] a fitter cannot treat
    /// as linear. Where the deadzoned HRV-z and HR-z disagree in sign, `min(a + b, 0)` and the per-term
    /// `min(a, 0) + min(b, 0)` are different numbers, so the emission has to name which one it is.
    #[test]
    fn the_awake_cardiac_clamp_acts_on_the_summed_pair_not_on_each_term() {
        let prep = prepare(&still_night_with_opposed_cardiac_terms(), &Params::SHIPPED);
        let t = emission_terms(&prep, &Params::SHIPPED);
        let w = weights_of(&Params::SHIPPED);

        let mut opposed = 0usize;
        for e in 0..t.design.len() {
            let d = &t.design[e][AWAKE];
            let (a, b) = (w[W_AWAKE_HRV] * d[W_AWAKE_HRV], w[W_AWAKE_HR] * d[W_AWAKE_HR]);
            if !t.clamped[e] || a * b >= 0.0 {
                continue;
            }
            let rest = t.fixed[e][AWAKE]
                + w[W_AWAKE_MOTION] * d[W_AWAKE_MOTION]
                + w[W_AWAKE_TURN] * d[W_AWAKE_TURN];
            let got = t.emission(e, &w)[AWAKE];
            assert_eq!(rest + (a + b).min(0.0), got, "epoch {e}: the pair is clamped whole");
            assert!((rest + a.min(0.0) + b.min(0.0) - got).abs() > 1e-9,
                "epoch {e}: a per-term clamp must be a different emission, not the same one");
            opposed += 1;
        }
        assert!(opposed >= 5, "the fixture must carry clamped epochs whose cardiac terms disagree in sign");

        // The loop's first assert is inert wherever the pair sums negative, because the clamp does not
        // bite there. A constructed epoch whose opposed pair sums POSITIVE parts all three readings:
        // whole-pair, unclamped and per-term are three different numbers.
        let mut design = [[0.0f64; 12]; 4];
        design[AWAKE][W_AWAKE_HRV] = 4.0;
        design[AWAKE][W_AWAKE_HR] = -1.0;
        design[AWAKE][W_AWAKE_MOTION] = 0.25;
        let fixed = [0.0, 0.0, 0.0, 1.0];
        let a = w[W_AWAKE_HRV] * design[AWAKE][W_AWAKE_HRV];
        let b = w[W_AWAKE_HR] * design[AWAKE][W_AWAKE_HR];
        assert!(a * b < 0.0 && a + b > 0.0, "the constructed pair must oppose in sign and sum positive");
        let rest = fixed[AWAKE]
            + w[W_AWAKE_MOTION] * design[AWAKE][W_AWAKE_MOTION]
            + w[W_AWAKE_TURN] * design[AWAKE][W_AWAKE_TURN];

        let bites = Terms { design: vec![design], fixed: vec![fixed], clamped: vec![true] };
        let got = bites.emission(0, &w)[AWAKE];
        assert_eq!(rest, got, "a positive opposed pair is clamped away whole");
        assert!((got - (rest + a + b)).abs() > 1e-9, "and it is not left unclamped");
        assert!((got - (rest + a.min(0.0) + b.min(0.0))).abs() > 1e-9, "nor clamped per term");

        let free = Terms { design: vec![design], fixed: vec![fixed], clamped: vec![false] };
        assert!((free.emission(0, &w)[AWAKE] - (rest + a + b)).abs() < 1e-12,
            "an unclamped epoch keeps the whole pair, so the flag is what decides");
    }

    /// The decomposition IS the emission: `fixed + sum(w * design)`, awake cardiac pair clamped whole,
    /// must reproduce [`emissions_prepared`] on every epoch and class. Expanded here instead of through
    /// [`Terms::emission`], so the check is independent of it; run under both anchors.
    #[test]
    fn the_decomposition_reproduces_the_emission_under_both_anchors() {
        // The emission written out from the parts: every weight but the awake cardiac pair, then that
        // pair summed and clamped as one. Same accumulation order, so the comparison can be exact.
        let expand = |t: &Terms, e: usize, w: &[f64; 12]| -> [f64; 4] {
            let mut em = [0.0f64; 4];
            for (c, out) in em.iter_mut().enumerate() {
                let d = &t.design[e][c];
                let mut acc = t.fixed[e][c];
                for (j, dj) in d.iter().enumerate() {
                    if c == AWAKE && (j == W_AWAKE_HRV || j == W_AWAKE_HR) {
                        continue;
                    }
                    acc += w[j] * dj;
                }
                if c == AWAKE {
                    let card = w[W_AWAKE_HRV] * d[W_AWAKE_HRV] + w[W_AWAKE_HR] * d[W_AWAKE_HR];
                    acc += if t.clamped[e] { card.min(0.0) } else { card };
                }
                *out = acc;
            }
            em
        };

        let window =
            Params { cycle_rem_onset_minutes: 0.0, cycle_clock_from_onset: false, ..Params::SHIPPED };
        let nights = [
            ("still", still_night_with_opposed_cardiac_terms()),
            ("restless", restless_night_with_respiration()),
        ];
        let anchors = [("onset", true, Params::SHIPPED), ("window", false, window)];
        let (mut clamped_seen, mut free_pair_seen, mut resp_seen) = (false, false, false);
        for (night, input) in &nights {
            for (label, onset_anchored, p) in anchors {
                let prep = prepare(input, &p);
                assert!(prep.feats.len() >= 30, "{night}/{label}: a night long enough to grade");
                assert_eq!(onset_anchored, matches!(resolve_anchor(&prep.feats, &p), Anchor::Onset(_)),
                    "{night}/{label}: the two passes must cover both anchors");
                let t = emission_terms(&prep, &p);
                let em = emissions_prepared(&prep, &p);
                let w = weights_of(&p);
                assert_eq!(em.len(), t.design.len(), "{night}/{label}: one design row per emission row");
                for (e, row) in em.iter().enumerate() {
                    assert_eq!(*row, expand(&t, e, &w), "{night}/{label}: epoch {e}");
                    let d = &t.design[e][AWAKE];
                    let card = w[W_AWAKE_HRV] * d[W_AWAKE_HRV] + w[W_AWAKE_HR] * d[W_AWAKE_HR];
                    clamped_seen |= t.clamped[e];
                    free_pair_seen |= !t.clamped[e] && card > 0.0;
                    resp_seen |= t.design[e][DEEP][W_RESP] != 0.0;
                }
                // The window-anchored guard is a step at `cycle_rem_early_frac` of the WINDOW clock and
                // nothing else in the REM prior steps, so the epoch that crosses it must rise by exactly
                // the penalty over the ramp's own even stride. Pins the clock the guard is handed.
                if !onset_anchored {
                    let frac = p.cycle_rem_early_frac;
                    let cross = (1..t.fixed.len() - 1)
                        .find(|&e| prep.feats[e - 1].clock < frac && prep.feats[e].clock >= frac)
                        .unwrap_or_else(|| panic!("{night}: no epoch crosses the early-REM boundary"));
                    let rise = |e: usize| t.fixed[e][REM] - t.fixed[e - 1][REM];
                    assert!((rise(cross) - rise(cross + 1) - p.cycle_rem_early_penalty).abs() < 1e-9,
                        "{night}: the crossing epoch must lift the REM prior by the penalty");
                }
            }
        }
        assert!(clamped_seen, "no epoch is clamped, so the clamped branch is unexercised");
        assert!(free_pair_seen,
            "no unclamped epoch keeps a positive cardiac pair, so an unconditional clamp would pass");
        assert!(resp_seen, "no epoch carries a respiration term, so the resp column is unexercised");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::input::RrRun;

    /// One minute of HR with gravity over only the first half: the epochs that saw the accelerometer
    /// carry a motion reading, the epochs that did not carry `None`.
    fn half_blind_epochs() -> Vec<Epoch> {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..60).map(|i| HrSample { ts: start + i, bpm: 55 }).collect();
        let accel: Vec<AccelSample> =
            (0..30).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect();
        let input = SleepInput { start, end: start + 60, hr, rr: Vec::<RrRun>::new(), accel };
        let mut grav = input.accel.clone();
        grav.sort_by_key(|g| g.ts);
        features(input.start, input.end, &grav, &input.hr, &[], &Params::SHIPPED)
    }

    /// A still, hot night, optionally carrying the swinging R-R of `terms_tests::rsa_rr`. R-R feeds
    /// only `resp_reg`, so the two nights differ in nothing else.
    fn still_hot_night(with_rr: bool) -> SleepInput {
        let start = 1_749_513_600i64;
        let hr: Vec<HrSample> = (0..600)
            .map(|i| HrSample { ts: start + i, bpm: if i < 300 { 50 } else { 90 } })
            .collect();
        let accel: Vec<AccelSample> =
            (0..600).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect();
        let rr = if with_rr { super::terms_tests::rsa_rr(start, 600) } else { Vec::new() };
        SleepInput { start, end: start + 600, hr, rr, accel }
    }

    /// Pins `quiescent_hr_z_max`: INFINITY must reproduce the unconditional clamp exactly, and a finite
    /// ceiling must let a still epoch whose heart rate sits above the night mean keep its awake cardiac
    /// term.
    #[test]
    fn the_quiescent_hr_ceiling_decides_whether_a_still_epoch_keeps_its_cardiac_term() {
        assert_eq!(Params::SHIPPED.quiescent_hr_z_max, f64::INFINITY, "shipped is the old behaviour");

        // Gravity flat throughout and HR stepping up, so the late epochs are motion-quiescent AND well
        // above the night's own mean.
        let input = still_hot_night(false);

        let awake_of = |z: f64| {
            let p = Params { quiescent_hr_z_max: z, ..Params::SHIPPED };
            let prep = prepare(&input, &p);
            let em = emissions_prepared(&prep, &p);
            // The last epoch: still, and at the top of the night's HR range.
            em.last().map(|e| e[AWAKE]).expect("emissions for a 10-minute night")
        };
        let clamped = awake_of(f64::INFINITY);
        let free = awake_of(0.0);
        assert!(free > clamped,
            "a hot still epoch must score MORE awake once the ceiling lets its cardiac term through:              {free} vs {clamped}");
        assert_eq!(clamped, awake_of(f64::MAX), "any ceiling above the data clamps identically");
    }

    /// Pins BOTH operands of the R-R clamp exemption. Under SHIPPED the exemption is off, so a still
    /// epoch is clamped whether or not R-R backs it; under the candidate the R-R night keeps its
    /// cardiac term. Turning the `&&` into an `||` makes SHIPPED behave as the candidate.
    #[test]
    fn the_rr_exemption_is_off_under_shipped_and_reads_the_respiration_term_under_the_candidate() {
        const { assert!(!Params::SHIPPED.clamp_only_without_rr) };

        let awake_of = |input: &SleepInput, p: &Params| {
            emissions_prepared(&prepare(input, p), p)
                .last()
                .map(|e| e[AWAKE])
                .expect("emissions for a 10-minute night")
        };
        let (with_rr, without_rr) = (still_hot_night(true), still_hot_night(false));

        let prep = prepare(&with_rr, &Params::SHIPPED);
        assert!(prep.feats.last().expect("epochs").resp_reg.is_some(),
            "the R-R night must actually produce a respiration term or this proves nothing");
        assert!(prepare(&without_rr, &Params::SHIPPED).feats.last().expect("epochs").resp_reg.is_none());

        assert_eq!(
            awake_of(&with_rr, &Params::SHIPPED), awake_of(&without_rr, &Params::SHIPPED),
            "under SHIPPED the presence of R-R must not change whether a still epoch is clamped");

        let cand = Params { clamp_only_without_rr: true, ..Params::SHIPPED };
        assert!(awake_of(&with_rr, &cand) > awake_of(&with_rr, &Params::SHIPPED),
            "the candidate must let an R-R-backed still epoch keep its awake cardiac term");
        assert_eq!(awake_of(&without_rr, &cand), awake_of(&without_rr, &Params::SHIPPED),
            "with no R-R there is nothing to exempt, so the candidate is SHIPPED");
    }

    /// The port's safety property: `awake_turn` ships at 0.0, so carrying the feature must not move
    /// a single emission. Without this the port is a silent recipe change on every user's night.
    #[test]
    fn carrying_turn_at_the_shipped_weight_changes_no_emission() {
        let input = still_hot_night(true);
        let base = emissions_prepared(&prepare(&input, &Params::SHIPPED), &Params::SHIPPED);
        let zeroed = Params { awake_turn: 0.0, ..Params::SHIPPED };
        assert_eq!(Params::SHIPPED.awake_turn, 0.0, "the shipped weight is zero");
        assert_eq!(base, emissions_prepared(&prepare(&input, &zeroed), &zeroed));
    }

    /// And it must actually DO something at a non-zero weight, or the wiring is decorative.
    #[test]
    fn a_nonzero_turn_weight_moves_the_awake_emission_on_a_rotating_night() {
        // turn must VARY, not merely be large: it is z-scored per night, so a night that rotates
        // by the same amount every epoch has zero variance and correctly contributes nothing.
        let input = super::terms_tests::rotating_night();
        let base = emissions_prepared(&prepare(&input, &Params::SHIPPED), &Params::SHIPPED);
        let weighted = Params { awake_turn: 1.0, ..Params::SHIPPED };
        let moved = emissions_prepared(&prepare(&input, &weighted), &weighted);
        assert_eq!(base.len(), moved.len());
        assert!(base.iter().zip(&moved).any(|(a, b)| a[AWAKE] != b[AWAKE]),
            "a rotating night at weight 1.0 must move some AWAKE emission");
        // Only AWAKE may move: turn enters no other stage's row.
        for (a, b) in base.iter().zip(&moved) {
            for st in [DEEP, REM, LIGHT] {
                assert_eq!(a[st], b[st], "turn must not touch stage {st}");
            }
        }
    }

    /// `turn` must be filed against the epoch that rotated on a night whose window does not open on the
    /// 30 s grid AND whose middle epoch carries neither channel. Binning the posture from the raw window,
    /// or reading it by position among the kept epochs, mis-files the rotation on one axis each.
    #[test]
    fn a_rotation_is_filed_against_the_epoch_that_rotated_across_a_dropped_one() {
        let start = 1_749_513_615i64;
        let first_e = 1_749_513_630i64; // the first 30 s boundary at or after `start`
        assert_ne!(0, start % 30, "the window must not open on the epoch grid");
        let rotates = first_e + 4 * 30;

        let (mut hr, mut accel) = (Vec::new(), Vec::new());
        for e in (0..6i64).filter(|e| *e != 2) {
            for s in 0..30i64 {
                let ts = first_e + e * 30 + s;
                let up = ts < rotates;
                hr.push(HrSample { ts, bpm: 55 });
                let (x, z) = if up { (0.0, 1.0) } else { (1.0, 0.0) };
                accel.push(AccelSample { ts, x, y: 0.0, z });
            }
        }
        let input = SleepInput { start, end: first_e + 6 * 30, hr, rr: Vec::<RrRun>::new(), accel };
        let feats = prepare(&input, &Params::SHIPPED).feats;

        assert_eq!(5, feats.len(), "the epoch with neither channel is dropped");
        assert_eq!(first_e, feats[0].start, "epochs open on the grid, not on the window");
        let turned: Vec<i64> =
            feats.iter().filter(|f| f.turn.is_some_and(|t| t > 1.0)).map(|f| f.start).collect();
        assert_eq!(vec![rotates], turned, "one rotation, filed against the epoch that made it");
        let t = feats[3].turn.expect("the rotated epoch carries a turn");
        assert!((t - 90.0).abs() < 1e-6, "a quarter turn, got {t}");
        assert_eq!(None, feats[2].turn, "the epoch after a dropped one has no previous orientation");
    }

    #[test]
    fn an_epoch_without_gravity_reports_no_motion_reading() {
        let feats = half_blind_epochs();
        assert_eq!(feats.len(), 2, "two 30 s epochs");
        assert_eq!(feats[0].move_frac, Some(0.0), "gravity present and still -> a measured zero");
        assert_eq!(feats[1].move_frac, None, "no gravity -> no reading, not a measured zero");
    }

    #[test]
    fn an_epoch_without_gravity_is_not_motion_quiescent() {
        // The quiescent gate clamps the awake-cardiac term, so asserting it on absent motion would let a
        // dropout suppress wake. Only an epoch that actually SAW a still wrist may be quiescent.
        let feats = half_blind_epochs();
        assert!(motion_quiescent(&feats[0], &Params::SHIPPED), "an observed still wrist is quiescent");
        assert!(!motion_quiescent(&feats[1], &Params::SHIPPED), "an absent accelerometer cannot assert stillness");
    }

    #[test]
    fn sustained_onset_is_the_first_epoch_of_the_first_long_non_wake_run() {
        let w = SleepStage::Wake;
        let l = SleepStage::Light;
        // A 9-epoch run is too short; the 10-epoch run that follows it is the onset.
        let mut labels = vec![w; 3];
        labels.extend(vec![l; 9]);
        labels.push(w);
        labels.extend(vec![l; 10]);
        assert_eq!(Some(13), sustained_onset(&labels));
        assert_eq!(None, sustained_onset(&[l; SUSTAINED_ONSET_EPOCHS - 1]));
    }

    #[test]
    fn the_onset_anchored_guard_grades_from_onset_not_from_the_window() {
        let p = Params::SHIPPED;
        assert!(p.cycle_rem_onset_minutes > 0.0, "the shipped guard is onset-anchored");
        let onset = 40usize;
        let full = p.cycle_rem_early_penalty;
        // Full magnitude at onset, nothing at onset + the grading width, and never negative before onset.
        assert_eq!(full, rem_guard(onset, 0.9, Anchor::Onset(onset), &p));
        let past = onset + (p.cycle_rem_onset_minutes / EPOCH_MIN) as usize;
        assert_eq!(0.0, rem_guard(past, 0.9, Anchor::Onset(onset), &p));
        assert_eq!(full, rem_guard(0, 0.9, Anchor::Onset(onset), &p));
        // A window anchor keeps the old step, and the probe pass carries no guard at all.
        assert_eq!(full, rem_guard(0, 0.01, Anchor::Window, &p));
        assert_eq!(0.0, rem_guard(0, 0.9, Anchor::Window, &p));
        assert_eq!(0.0, rem_guard(0, 0.01, Anchor::Probe, &p));
    }

    #[test]
    fn the_onset_anchored_clock_rebases_only_when_asked() {
        let feats = half_blind_epochs();
        let shipped = Params::SHIPPED;
        let from_onset = Params { cycle_clock_from_onset: true, ..shipped };
        let c0 = feats[0].clock;
        assert_eq!(0.75, cycle_clock(0.75, &feats, Anchor::Onset(0), &shipped), "off by default");
        assert_eq!(0.75, cycle_clock(0.75, &feats, Anchor::Window, &from_onset), "no anchor, no rebase");
        assert_eq!(
            (0.75 - c0) / (1.0 - c0),
            cycle_clock(0.75, &feats, Anchor::Onset(0), &from_onset),
            "rebased onto the onset epoch's own fraction"
        );
        assert_eq!(0.0, cycle_clock(0.0, &feats, Anchor::Onset(1), &from_onset), "clamped below onset");

        // An onset at the very end of the window leaves nothing to rebase onto. The guard returns the
        // clock untouched; without it the divisor is zero and the epoch's prior collapses to 0.0.
        let mut degenerate = half_blind_epochs();
        degenerate[0].clock = 1.0;
        assert_eq!(0.75, cycle_clock(0.75, &degenerate, Anchor::Onset(0), &from_onset));
    }

    // ── the decoder, isolated from the emissions feeding it ───────────────────────────────────────

    /// Column index of a stage — the inverse of [`STAGE_ORDER`], so a test can talk in indices.
    fn idx_of(s: SleepStage) -> usize {
        STAGE_ORDER.iter().position(|&x| x == s).unwrap()
    }

    /// Total log-score of one path: its emissions plus every transition it takes. Written independently
    /// of the decoder, so a test can rank paths without using the thing under test.
    fn path_score(em: &[[f64; 4]], t: &[[f64; 4]; 4], path: &[usize]) -> f64 {
        let mut s = em[0][path[0]];
        for i in 1..path.len() {
            s += t[path[i - 1]][path[i]].max(1e-9).ln() + em[i][path[i]];
        }
        s
    }

    /// Best path and its score by exhaustive enumeration of all 4^n. The oracle the decoder is checked
    /// against; feasible only for a handful of epochs, which is why it lives in a test.
    fn brute_force(em: &[[f64; 4]], t: &[[f64; 4]; 4]) -> (Vec<usize>, f64) {
        let n = em.len();
        let mut best = (Vec::new(), f64::NEG_INFINITY);
        for code in 0..4usize.pow(n as u32) {
            let path: Vec<usize> = (0..n).map(|i| (code >> (2 * i)) & 3).collect();
            let s = path_score(em, t, &path);
            if s > best.1 {
                best = (path, s);
            }
        }
        best
    }

    /// Deterministic LCG in [-4, 4) — the same emission sequences on every run, at a spread comparable to
    /// the log-transitions so neither term trivially dominates.
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64) * 8.0 - 4.0
    }

    /// A second matrix beside the shipped one, with a hard zero row so the 1e-9 floor is exercised.
    const PROBE_T: [[f64; 4]; 4] = [
        [0.50, 0.20, 0.20, 0.10],
        [0.10, 0.40, 0.40, 0.10],
        [0.25, 0.25, 0.25, 0.25],
        [0.00, 0.00, 0.50, 0.50],
    ];

    #[test]
    fn the_decoder_returns_the_maximum_likelihood_path_on_every_crafted_sequence() {
        // 80 sequences of 7 epochs, alternating between the shipped matrix and one with a zero row, each
        // checked against exhaustive enumeration of all 16,384 paths. A decoder that matched once would
        // prove nothing.
        let mut state = 0x5eed_1234_u64;
        let mut checked = 0usize;
        for case in 0..80 {
            let t = if case % 2 == 0 { Params::SHIPPED.transition } else { PROBE_T };
            let em: Vec<[f64; 4]> =
                (0..7).map(|_| [lcg(&mut state), lcg(&mut state), lcg(&mut state), lcg(&mut state)]).collect();
            let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
            let (want, best) = brute_force(&em, &t);
            assert_eq!(path_score(&em, &t, &got), best, "case {case}: decoded path is not optimal");
            assert_eq!(got, want, "case {case}: decoded path differs from the enumerated best");
            checked += 1;
        }
        assert_eq!(80, checked);
    }

    #[test]
    fn every_injected_path_is_recovered_when_the_emissions_name_it() {
        // All 1,024 paths over 5 epochs, injected as an emission margin large enough to outweigh any
        // transition (the 1e-9 floor is worth 20.7 log units, and a deviation can gain at most two of
        // them). Recovery of one path is an anecdote; this is the whole space.
        let t = Params::SHIPPED.transition;
        let mut recovered = 0usize;
        for code in 0..1024usize {
            let want: Vec<usize> = (0..5).map(|i| (code >> (2 * i)) & 3).collect();
            let em: Vec<[f64; 4]> = want
                .iter()
                .map(|&s| {
                    let mut row = [0.0; 4];
                    row[s] = 200.0;
                    row
                })
                .collect();
            let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
            assert_eq!(got, want, "path {code} not recovered");
            recovered += 1;
        }
        assert_eq!(1024, recovered);
    }

    #[test]
    fn the_transitions_override_a_one_epoch_emission_blip() {
        // Hand-computable: REM-REM-REM scores 10 + 2*ln(0.92) = 9.83, while following the middle epoch's
        // own argmax into deep scores 11 + ln(0.00333) + ln(0.012) = 0.87. Without the transition term the
        // decoder would be a per-epoch argmax, so this is what separates the two.
        let t = Params::SHIPPED.transition;
        let mut em = [[0.0f64; 4]; 3];
        em[0][REM] = 5.0;
        em[1][DEEP] = 1.0;
        em[2][REM] = 5.0;
        let got: Vec<usize> = viterbi(&em, &t).into_iter().map(idx_of).collect();
        assert_eq!(vec![REM, REM, REM], got, "the sticky diagonal must smooth a single blip");
        assert_eq!(DEEP, (0..4).max_by(|a, b| em[1][*a].total_cmp(&em[1][*b])).unwrap(), "argmax disagrees");
        // And it is not unconditional: widen the blip's margin past the two transitions it must pay for
        // and the decoder follows the evidence instead.
        em[1][DEEP] = 12.0;
        assert_eq!(DEEP, idx_of(viterbi(&em, &t)[1]), "a large enough margin must win");
    }

    #[test]
    fn the_decoder_is_total_on_degenerate_input() {
        let t = Params::SHIPPED.transition;
        assert!(viterbi(&[], &t).is_empty(), "no epochs, no labels");
        // One epoch has no transition to pay, so it is the emission argmax alone.
        assert_eq!(SleepStage::Wake, viterbi(&[[0.0, 1.0, 2.0, 3.0]], &t)[0]);
        // Everything equal under a uniform matrix: every path ties and the earliest column wins.
        let uniform = [[0.25; 4]; 4];
        assert!(viterbi(&[[0.0; 4]; 6], &uniform).iter().all(|&s| s == SleepStage::Deep));
    }

    /// Twenty minutes of HR and gravity, still for the first half and jittering in the second, so the
    /// staging has a path to search rather than one flat answer.
    fn crafted_night() -> Prepared {
        let start = 1_749_513_600i64;
        let (mut hr, mut accel) = (Vec::new(), Vec::new());
        for i in 0..1200i64 {
            let bpm = if i < 600 { 52 + (i % 3) as u16 } else { 66 + (i % 5) as u16 };
            hr.push(HrSample { ts: start + i, bpm });
            let jitter = if i < 600 { 0.0 } else { 0.02 * (i % 7) as f64 };
            accel.push(AccelSample { ts: start + i, x: jitter, y: 0.0, z: 1.0 - jitter });
        }
        let input = SleepInput { start, end: start + 1200, hr, rr: Vec::<RrRun>::new(), accel };
        prepare(&input, &Params::SHIPPED)
    }

    #[test]
    fn the_exported_emissions_decode_to_the_labels_the_stager_ships() {
        // The exported pair has to BE the shipped staging, or a harness scoring them scores a
        // reconstruction of the pipeline instead of the pipeline.
        let prep = crafted_night();
        let p = Params::SHIPPED;
        let em = emissions_prepared(&prep, &p);
        assert_eq!(em.len(), prep.feats.len(), "one emission row per epoch");
        assert_eq!(em.len(), epoch_starts(&prep).len(), "one epoch start per emission row");
        assert!(em.len() >= 30, "a night long enough for the path search to matter");
        let via_decoder = segments_of(&prep, &viterbi(&em, &p.transition));
        assert_eq!(stage_prepared(&prep, &p), via_decoder);
    }

    #[test]
    fn every_transition_row_is_a_distribution_and_a_zero_is_only_a_floor() {
        // The rows are read as probabilities and logged, so a row that does not sum to 1 would weight one
        // stage's whole outgoing mass. A 0 cell is floored, not forbidden.
        for (i, row) in Params::SHIPPED.transition.iter().enumerate() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9, "row {i} is not a distribution");
        }
        let t = Params::SHIPPED.transition;
        assert_eq!(0.0, t[AWAKE][DEEP], "the awake row's zeros are what the floor has to cover");
        let em = [[0.0, 0.0, 0.0, 50.0], [50.0, 0.0, 0.0, 0.0]];
        assert_eq!(vec![SleepStage::Wake, SleepStage::Deep], viterbi(&em, &t), "a zero cell is traversable");
    }

    #[test]
    fn an_absent_motion_reading_z_scores_neutral() {
        // The z-scorer must place "no reading" at the centre, not at the bottom of the night's motion
        // distribution, so a dropout never reads as the stillest stretch of the night.
        let z = ZScore::build(&[Some(0.0), Some(4.0), Some(8.0), None]);
        assert_eq!(z.apply(None), 0.0);
        assert!(z.apply(Some(0.0)) < 0.0, "a measured zero sits below the night mean");
        assert!(z.apply(None) > z.apply(Some(0.0)), "absence must not out-still a measured zero");
    }
}
