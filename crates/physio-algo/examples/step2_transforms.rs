//! Step 2 of the sleep pipeline: how a feature is NORMALISED before it reaches the emission.
//!
//!   cargo run --release -p physio-algo --example step2_transforms
//!
//! Rebuilds v2's emission from its own `Terms`, swaps ONE transform at a time, re-decodes under the
//! shipped transition, and reports the paired per-night kappa difference on the same nights. A
//! variant is chosen on the two training cohorts and reported once on the held-out one.

mod common;

use std::collections::BTreeMap;

use common::lr::design_row;
use common::{
    cardiac_series, dirs_of, mean as amean, median, onset_of, read_accel, read_hr, read_meta,
    read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::posture::{posture_series, turn_series};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, epoch_starts_v2, flatten_rr, params::Params,
    prepare_v2, resp_regularity, weights_of, AccelSample, HrSample, SleepInput,
};
use physio_algo::stats;

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;

// STAGE_ORDER positions: the emission's own column order, [deep, rem, light, awake].
const DEEP: usize = 0;
const REM: usize = 1;
const AWAKE: usize = 3;

// Slots in `weights_of`, mirrored so a design column is read by name here too.
const W_DEEP_HRV: usize = 0;
const W_DEEP_HR: usize = 1;
const W_DEEP_MOTION: usize = 2;
const W_DEEP_GATE: usize = 3;
const W_REM_HRV: usize = 4;
const W_REM_MOTION: usize = 5;
const W_REM_HR: usize = 6;
const W_AWAKE_MOTION: usize = 7;
const W_AWAKE_HRV: usize = 8;
const W_AWAKE_HR: usize = 9;
const W_AWAKE_TURN: usize = 10;
const W_RESP: usize = 11;

/// Extra tanv1 design columns past `Features::N`, as `fit_residual` lays them out.
const N_NL: usize = 6;

// ---------------------------------------------------------------------------------------------
// transforms

/// How a per-night column is put on a common scale before the fixed weights multiply it.
#[derive(Clone, Copy, PartialEq)]
enum Tr {
    /// Mean and population sd, what `ZScore` does.
    Z,
    /// Median and 1.4826 x MAD: same units, robust location and scale.
    RobustZ,
    /// Midrank mapped through the normal quantile, so a rank lands on the z scale.
    Rankit,
}

/// Shape of the deep-gate threshold on the flatness rank.
#[derive(Clone, Copy, PartialEq)]
enum HingeK {
    Hard,
    Soft(f64),
    /// No rectification at all: the ablation that says whether the hinge earns its shape.
    Linear,
}

/// How the deep gate turns this night's HR flatness into the 0..1 quantity it thresholds.
#[derive(Clone, Copy, PartialEq)]
enum GateK {
    /// Empirical rank, `bisect_right / n`. What v2 does.
    Rank,
    /// The same rank with ties averaged and the half-step bias removed.
    Midrank,
    /// Normal CDF of the z-score: the same 0..1 scale, so `deep_gate_thresh` keeps its meaning.
    NormalCdf,
}

/// Shape of the awake cardiac deadzone.
#[derive(Clone, Copy, PartialEq)]
enum DeadK {
    Hard,
    Tanh,
}

/// Shape and arity of the stillness clamp on the awake cardiac pair.
#[derive(Clone, Copy, PartialEq)]
enum ClampK {
    Pair,
    PairSoft(f64),
    PerTerm,
}

/// One transform recipe. `SHIPPED_V` is v2 exactly, so every row is one axis moved off it.
#[derive(Clone, Copy, PartialEq)]
struct Variant {
    card: Tr,
    motion: Tr,
    resp: Tr,
    /// Estimate the location and scale over the decoded sleep period only, not the whole window.
    stats_on_sleep: bool,
    /// Score a missing flatness as no gate at all, instead of `pct`'s median stand-in.
    gate_abstain: bool,
    gate: GateK,
    hinge: HingeK,
    dead: DeadK,
    clamp: ClampK,
}

const SHIPPED_V: Variant = Variant {
    card: Tr::Z,
    motion: Tr::Z,
    resp: Tr::Z,
    stats_on_sleep: false,
    gate_abstain: false,
    gate: GateK::Rank,
    hinge: HingeK::Hard,
    dead: DeadK::Hard,
    clamp: ClampK::Pair,
};

/// Abramowitz and Stegun 7.1.26 for `erf`, wrapped as the standard normal CDF, |error| < 1.5e-7.
fn phi(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    0.5 * (1.0 + s * (1.0 - poly * (-a * a).exp()))
}

/// Acklam's rational approximation to the standard normal quantile, |error| < 1.15e-9.
fn phi_inv(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.38357751867269e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let lo = 0.02425;
    if p <= 0.0 {
        return -8.0;
    }
    if p >= 1.0 {
        return 8.0;
    }
    if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > 1.0 - lo {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// Normalise a per-night column, estimating the statistics over the `sample` epochs only. A missing
/// value scores the neutral centre 0.0, which is what `ZScore::apply` does.
fn normalise(v: &[Option<f64>], sample: &[bool], t: Tr) -> Vec<f64> {
    let present: Vec<f64> =
        v.iter().zip(sample).filter(|(_, s)| **s).filter_map(|(x, _)| *x).collect();
    if present.is_empty() {
        return vec![0.0; v.len()];
    }
    match t {
        Tr::Z => {
            let m = stats::mean(&present);
            let sd0 = stats::population_sd(&present);
            let sd = if sd0 == 0.0 { 1.0 } else { sd0 };
            v.iter().map(|x| x.map_or(0.0, |x| (x - m) / sd)).collect()
        }
        Tr::RobustZ => {
            let m = stats::median(&present);
            let dev: Vec<f64> = present.iter().map(|x| (x - m).abs()).collect();
            let mad = 1.4826 * stats::median(&dev);
            let sd = if mad > 0.0 {
                mad
            } else if stats::population_sd(&present) > 0.0 {
                stats::population_sd(&present)
            } else {
                1.0
            };
            v.iter().map(|x| x.map_or(0.0, |x| (x - m) / sd)).collect()
        }
        Tr::Rankit => {
            let mut s = present.clone();
            s.sort_by(f64::total_cmp);
            let n = s.len() as f64;
            v.iter()
                .map(|x| {
                    x.map_or(0.0, |x| {
                        let lo = s.partition_point(|q| *q < x) as f64;
                        let hi = s.partition_point(|q| *q <= x) as f64;
                        phi_inv((0.5 * (lo + hi) / n).clamp(0.5 / n, 1.0 - 0.5 / n))
                    })
                })
                .collect()
        }
    }
}

/// Within-night rank as `bisect_right / n`, the transform the deep gate and the rotation term read.
fn uniform_pct(v: &[Option<f64>]) -> Vec<f64> {
    let mut s: Vec<f64> = v.iter().flatten().copied().collect();
    s.sort_by(f64::total_cmp);
    v.iter()
        .map(|x| match x {
            Some(x) if !s.is_empty() => s.partition_point(|q| *q <= *x) as f64 / s.len() as f64,
            _ => 0.5,
        })
        .collect()
}

fn hinge_of(x: f64, k: HingeK) -> f64 {
    match k {
        HingeK::Hard => x.max(0.0),
        HingeK::Linear => x,
        HingeK::Soft(s) if x / s > 30.0 => x,
        HingeK::Soft(s) => s * (1.0 + (x / s).exp()).ln(),
    }
}

/// Within-night midrank in 0..1: ties share the average rank, and the half-step bias is removed.
fn midrank_pct(v: &[Option<f64>]) -> Vec<f64> {
    let mut s: Vec<f64> = v.iter().flatten().copied().collect();
    s.sort_by(f64::total_cmp);
    let n = s.len() as f64;
    v.iter()
        .map(|x| match x {
            Some(x) if !s.is_empty() => {
                let lo = s.partition_point(|q| *q < *x) as f64;
                let hi = s.partition_point(|q| *q <= *x) as f64;
                0.5 * (lo + hi) / n
            }
            _ => 0.5,
        })
        .collect()
}

fn dead_of(z: f64, d: f64, k: DeadK) -> f64 {
    if d <= 0.0 {
        return z;
    }
    match k {
        DeadK::Hard if z > d => z - d,
        DeadK::Hard if z < -d => z + d,
        DeadK::Hard => 0.0,
        DeadK::Tanh => z - d * (z / d).tanh(),
    }
}

/// Smooth `min(x, 0.0)`: the softplus of the negative part, same asymptotes.
fn soft_min0(x: f64, s: f64) -> f64 {
    if -x / s > 30.0 {
        x
    } else {
        -s * (1.0 + (-x / s).exp()).ln()
    }
}

// ---------------------------------------------------------------------------------------------
// loading

/// One night, held as the raw per-epoch quantities plus everything v2 computed around them.
struct Night {
    epochs: usize,
    hr: Vec<Option<f64>>,
    hr_var: Vec<Option<f64>>,
    mv: Vec<Option<f64>>,
    flat: Vec<Option<f64>>,
    turn: Vec<Option<f64>>,
    resp: Vec<Option<f64>>,
    /// v2's own emission and the parts of it this harness does not touch.
    em: Vec<[f64; CLASSES]>,
    fixed: Vec<[f64; CLASSES]>,
    clamped: Vec<bool>,
    design: Vec<[[f64; 12]; CLASSES]>,
    /// Epochs the decoded v2 path calls the sleep period, for the sleep-anchored statistics.
    sleep: Vec<bool>,
    truth: Vec<Option<usize>>,
    /// The tanv1 design row per epoch, for the imputation census.
    tan: Vec<Vec<f64>>,
}

fn sec_mean_hr(hr: &[HrSample]) -> BTreeMap<i64, f64> {
    let mut acc: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for s in hr {
        let e = acc.entry(s.ts).or_insert((0.0, 0.0));
        e.0 += s.bpm as f64;
        e.1 += 1.0;
    }
    acc.into_iter().map(|(t, (a, c))| (t, a / c)).collect()
}

fn sec_mean_g(g: &[AccelSample]) -> BTreeMap<i64, (f64, f64, f64)> {
    let mut acc: BTreeMap<i64, (f64, f64, f64, f64)> = BTreeMap::new();
    for s in g {
        let e = acc.entry(s.ts).or_insert((0.0, 0.0, 0.0, 0.0));
        e.0 += s.x;
        e.1 += s.y;
        e.2 += s.z;
        e.3 += 1.0;
    }
    acc.into_iter().map(|(t, v)| (t, (v.0 / v.3, v.1 / v.3, v.2 / v.3))).collect()
}

/// Population sd of per-second heart rate over `[lo, hi)`, the statistic v2's prefix sums compute.
fn sd_seconds(sec: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> Option<f64> {
    let v: Vec<f64> = sec.range(lo..hi).map(|(_, b)| *b).collect();
    if v.len() < 2 {
        return None;
    }
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    Some((v.iter().map(|x| x * x).sum::<f64>() / n - m * m).max(0.0).sqrt())
}

/// The six raw per-epoch quantities v2 normalises, on v2's own epoch grid. Recomputed here so the
/// harness holds the values BEFORE the transform, which `Terms` only exposes after it.
#[allow(clippy::type_complexity)]
fn raw_of(
    input: &SleepInput,
    starts: &[i64],
    p: &Params,
) -> (
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
    Vec<Option<f64>>,
) {
    let mut grav = input.accel.clone();
    grav.sort_by_key(|g| g.ts);
    let sec_hr = sec_mean_hr(&input.hr);
    let sec_g = sec_mean_g(&grav);
    let mut rr_by: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (ts, ms) in flatten_rr(&input.rr) {
        rr_by.entry(ts).or_default().push(ms);
    }

    let mut hr = Vec::with_capacity(starts.len());
    let mut hr_var = Vec::with_capacity(starts.len());
    let mut flat = Vec::with_capacity(starts.len());
    let mut resp = Vec::with_capacity(starts.len());
    let mut per_jerk: Vec<Vec<f64>> = Vec::with_capacity(starts.len());
    let mut gaps: Vec<i64> = Vec::with_capacity(starts.len());
    let mut all_jerks: Vec<f64> = Vec::new();

    for &e in starts {
        let mut hrs: Vec<f64> = Vec::new();
        let mut gseq: Vec<(f64, f64, f64)> = Vec::new();
        for s in e..e + EPOCH {
            if let Some(v) = sec_hr.get(&s) {
                hrs.push(*v);
            }
            if let Some(v) = sec_g.get(&s) {
                gseq.push(*v);
            }
        }
        let j: Vec<f64> = gseq
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let (dx, dy, dz) = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        all_jerks.extend_from_slice(&j);
        gaps.push((gseq.len() as i64 - 1).max(1));
        per_jerk.push(j);

        hr.push((!hrs.is_empty()).then(|| hrs.iter().sum::<f64>() / hrs.len() as f64));
        hr_var.push(sd_seconds(&sec_hr, e - 150, e + EPOCH + 150));
        flat.push(sd_seconds(&sec_hr, e - 330, e + EPOCH + 360));

        let mut beats: Vec<(f64, f64)> = rr_by
            .range(e - 90..e + 120)
            .flat_map(|(t, vs)| vs.iter().map(|v| (*t as f64, v.clamp(300.0, 2000.0))))
            .collect();
        beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
        resp.push(resp_regularity(&beats));
    }

    let scale = if all_jerks.is_empty() { 1e-6 } else { stats::median(&all_jerks) };
    let thr = scale * p.jerk_move_mult;
    let mv: Vec<Option<f64>> = per_jerk
        .iter()
        .zip(&gaps)
        .map(|(j, g)| {
            (!j.is_empty()).then(|| j.iter().filter(|&&x| x > thr).count() as f64 / *g as f64)
        })
        .collect();

    let first_e = ((input.start + EPOCH - 1) / EPOCH) * EPOCH;
    let last = starts.last().map_or(first_e, |s| s + EPOCH);
    let post = posture_series(&grav, first_e, last, EPOCH);
    let ts = turn_series(&post);
    let turn: Vec<Option<f64>> =
        starts.iter().map(|s| ts.get(((s - first_e) / EPOCH) as usize).copied().flatten()).collect();

    (hr, hr_var, mv, flat, turn, resp)
}

fn load(set: &str) -> Vec<Night> {
    let p = Params::SHIPPED;
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let rawt = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if rawt.is_empty() || accel.is_empty() {
            continue;
        }
        let n = n_meta.max(rawt.keys().max().copied().unwrap_or(0) + 1);
        let (hrs, rr) = (read_hr(dir), read_rr(dir));
        let feats = extract(&accel, w0, w1, &cardiac_series(w0, n, EPOCH, &hrs, &rr));
        let input = SleepInput { start: w0, end: w1, hr: hrs, rr, accel };
        let prep = prepare_v2(&input, &p);
        let em = emissions_v2(&prep, &p);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let terms = emission_terms(&prep, &p);
        let starts = epoch_starts_v2(&prep);
        assert_eq!(em.len(), n, "{}: {n} truth epochs against {} emissions", dir.display(), em.len());
        let (hr, hr_var, mv, flat, turn, resp) = raw_of(&input, &starts, &p);

        let path: Vec<usize> = decode_v2(&em, &p.transition).iter().map(|s| stage_idx(*s)).collect();
        let mut sleep = vec![false; em.len()];
        match onset_of(&path) {
            Some(o) => {
                let off = path.iter().rposition(|l| *l != 0).unwrap_or(em.len() - 1);
                for s in sleep.iter_mut().take(off + 1).skip(o) {
                    *s = true;
                }
            }
            None => sleep.iter_mut().for_each(|s| *s = true),
        }
        if sleep.iter().filter(|s| **s).count() < 20 {
            sleep.iter_mut().for_each(|s| *s = true);
        }

        let tan = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = feats[e].values().to_vec();
                v.extend_from_slice(&[
                    -d[DEEP][W_DEEP_GATE],
                    d[AWAKE][W_AWAKE_HRV],
                    d[AWAKE][W_AWAKE_HR],
                    d[AWAKE][W_AWAKE_TURN],
                    d[DEEP][W_RESP],
                    if terms.clamped[e] { 1.0 } else { 0.0 },
                ]);
                v
            })
            .collect();

        let truth = (0..em.len())
            .map(|k| {
                rawt.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();

        out.push(Night {
            epochs: em.len(),
            hr,
            hr_var,
            mv,
            flat,
            turn,
            resp,
            em,
            fixed: terms.fixed.clone(),
            clamped: terms.clamped.clone(),
            design: terms.design.clone(),
            sleep,
            truth,
            tan,
        });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// emission rebuild

/// Sum one epoch's design against the weights, applying whichever clamp shape `k` names. At
/// `ClampK::Pair` this is `Terms::emission` exactly.
fn sum_em(d: &[[f64; 12]; CLASSES], fx: &[f64; CLASSES], clamped: bool, w: &[f64; 12], k: ClampK)
    -> [f64; CLASSES] {
    let mut em = [0.0f64; CLASSES];
    for c in 0..CLASSES {
        let mut acc = fx[c];
        for (j, wj) in w.iter().enumerate() {
            if c == AWAKE && (j == W_AWAKE_HRV || j == W_AWAKE_HR) {
                continue;
            }
            acc += wj * d[c][j];
        }
        if c == AWAKE {
            let a = w[W_AWAKE_HRV] * d[c][W_AWAKE_HRV];
            let b = w[W_AWAKE_HR] * d[c][W_AWAKE_HR];
            acc += if !clamped {
                a + b
            } else {
                match k {
                    ClampK::Pair => (a + b).min(0.0),
                    ClampK::PairSoft(s) => soft_min0(a + b, s),
                    ClampK::PerTerm => a.min(0.0) + b.min(0.0),
                }
            };
        }
        em[c] = acc;
    }
    em
}

/// This night's design columns under `v`, then the emission they sum to.
fn emissions_under(nt: &Night, v: &Variant, p: &Params) -> Vec<[f64; CLASSES]> {
    let all = vec![true; nt.epochs];
    let sample: &[bool] = if v.stats_on_sleep { &nt.sleep } else { &all };
    let zhr = normalise(&nt.hr, sample, v.card);
    let zhv = normalise(&nt.hr_var, sample, v.card);
    let zmv = normalise(&nt.mv, sample, v.motion);
    let rz = normalise(&nt.resp, sample, v.resp);
    let fpct = match v.gate {
        GateK::Rank => uniform_pct(&nt.flat),
        GateK::Midrank => midrank_pct(&nt.flat),
        // Same 0..1 scale as the rank, so the gate threshold and slope keep their meaning.
        GateK::NormalCdf => normalise(&nt.flat, sample, Tr::Z)
            .iter()
            .zip(&nt.flat)
            .map(|(z, raw)| if raw.is_some() { phi(*z) } else { 0.5 })
            .collect(),
    };
    let tpct = uniform_pct(&nt.turn);
    let w = weights_of(p);

    (0..nt.epochs)
        .map(|e| {
            let mut d = [[0.0f64; 12]; CLASSES];
            d[DEEP][W_DEEP_HRV] = zhv[e];
            d[DEEP][W_DEEP_HR] = zhr[e];
            d[DEEP][W_DEEP_MOTION] = zmv[e];
            d[DEEP][W_DEEP_GATE] = if v.gate_abstain && nt.flat[e].is_none() {
                0.0
            } else {
                -hinge_of(fpct[e] - p.deep_gate_thresh, v.hinge)
            };
            d[REM][W_REM_HRV] = zhv[e];
            d[REM][W_REM_MOTION] = zmv[e];
            d[REM][W_REM_HR] = zhr[e];
            d[AWAKE][W_AWAKE_MOTION] = zmv[e];
            d[AWAKE][W_AWAKE_HRV] = dead_of(zhv[e], p.awake_deadzone, v.dead);
            d[AWAKE][W_AWAKE_HR] = dead_of(zhr[e], p.awake_deadzone, v.dead);
            d[AWAKE][W_AWAKE_TURN] = (tpct[e] - 0.5) * 2.0;
            d[DEEP][W_RESP] = rz[e];
            d[REM][W_RESP] = -rz[e];
            sum_em(&d, &nt.fixed[e], nt.clamped[e], &w, v.clamp)
        })
        .collect()
}

/// Per-night kappa of a decoded emission against truth; `None` where too few labelled epochs.
fn kappa_of(nt: &Night, em: &[[f64; CLASSES]], p: &Params) -> Option<f64> {
    let path: Vec<usize> = decode_v2(em, &p.transition).iter().map(|s| stage_idx(*s)).collect();
    let (mut pr, mut tr) = (Vec::new(), Vec::new());
    for (k, want) in nt.truth.iter().enumerate() {
        if let Some(t) = want {
            pr.push(path[k]);
            tr.push(*t);
        }
    }
    (tr.len() >= MIN_EPOCHS).then(|| kappa4(&confusion4(&pr, &tr)))
}

/// Per-night kappa difference against v2 on the same nights, and the share of epochs whose decoded
/// label moved. A near-zero delta means nothing only when the labels DID move.
fn arm(nights: &[Night], v: &Variant, p: &Params) -> (Vec<f64>, f64) {
    let (mut d, mut moved, mut tot) = (Vec::new(), 0usize, 0usize);
    for nt in nights {
        let em = emissions_under(nt, v, p);
        let a = decode_v2(&em, &p.transition);
        let b = decode_v2(&nt.em, &p.transition);
        moved += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        tot += a.len();
        let Some(bk) = kappa_of(nt, &nt.em, p) else { continue };
        d.push(kappa_of(nt, &em, p).expect("same nights") - bk);
    }
    (d, 100.0 * moved as f64 / tot.max(1) as f64)
}

/// One variant's paired mean and 95% bar against v2, one entry per cohort in load order.
struct Scored {
    name: String,
    per_cohort: Vec<(f64, f64)>,
}

fn verdict(d: &[f64]) -> (f64, f64, String) {
    let Some((m, bar)) = paired_bar(d) else { return (f64::NAN, f64::NAN, "-".into()) };
    let v = if m.abs() > bar {
        format!("{} ({:.2}x)", if m > 0.0 { "BEATS v2" } else { "worse" }, m.abs() / bar)
    } else {
        "matches".into()
    };
    (m, bar, v)
}

// ---------------------------------------------------------------------------------------------

fn positive_controls(loaded: &[(&str, Vec<Night>)], p: &Params) {
    let w = weights_of(p);
    let (mut worst_rebuild, mut worst_raw, mut path_diff, mut nights) = (0.0f64, 0.0f64, 0usize, 0);
    for (_, ns) in loaded {
        for nt in ns {
            nights += 1;
            let mine = emissions_under(nt, &SHIPPED_V, p);
            for (e, want) in nt.em.iter().enumerate() {
                let got = sum_em(&nt.design[e], &nt.fixed[e], nt.clamped[e], &w, ClampK::Pair);
                for c in 0..CLASSES {
                    worst_rebuild = worst_rebuild.max((got[c] - want[c]).abs());
                    worst_raw = worst_raw.max((mine[e][c] - want[c]).abs());
                }
            }
            let a = decode_v2(&mine, &p.transition);
            let b = decode_v2(&nt.em, &p.transition);
            path_diff += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        }
    }
    println!("CONTROL  {nights} nights");
    println!("  rebuild from Terms vs emissions_v2   max |diff| {worst_rebuild:.3e}");
    println!("  rebuild from RAW   vs emissions_v2   max |diff| {worst_raw:.3e}");
    println!("  epochs whose decoded label differs   {path_diff}");
    assert!(worst_rebuild < 1e-12, "the Terms rebuild is not v2's own emission");
    assert!(worst_raw < 1e-7, "the recomputed raw columns do not reproduce v2's design");
    assert_eq!(path_diff, 0, "the shipped variant must decode to v2's own path");
}

/// Missing rate of each v2 column, per cohort. v2 scores a missing value the neutral centre, which
/// is the same fabrication `design_row` performs with the train mean.
fn v2_missing_census(loaded: &[(&str, Vec<Night>)]) {
    println!("\nV2 COLUMN COVERAGE (share of epochs the column is ABSENT and scored neutral)");
    println!("  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "cohort", "hr", "hr_var", "move", "flat", "turn", "resp");
    for (name, ns) in loaded {
        let mut miss = [0usize; 6];
        let mut tot = 0usize;
        for nt in ns {
            tot += nt.epochs;
            let cols: [&Vec<Option<f64>>; 6] =
                [&nt.hr, &nt.hr_var, &nt.mv, &nt.flat, &nt.turn, &nt.resp];
            for (i, c) in cols.iter().enumerate() {
                miss[i] += c.iter().filter(|v| v.is_none()).count();
            }
        }
        let pct = |k: usize| 100.0 * miss[k] as f64 / tot.max(1) as f64;
        println!(
            "  {name:<14} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            pct(0), pct(1), pct(2), pct(3), pct(4), pct(5)
        );
    }

    // A column absent for a WHOLE night takes the empty branch of `ZScore::build` or of `pct`, and
    // those two branches do different things: the z abstains, the rank stands in at the median.
    println!("  nights where the column is absent END TO END (the empty-sample branches)");
    for (name, ns) in loaded {
        let mut whole = [0usize; 6];
        for nt in ns {
            let cols: [&Vec<Option<f64>>; 6] =
                [&nt.hr, &nt.hr_var, &nt.mv, &nt.flat, &nt.turn, &nt.resp];
            for (i, c) in cols.iter().enumerate() {
                whole[i] += usize::from(c.iter().all(Option::is_none));
            }
        }
        println!(
            "  {name:<14} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}   of {} nights",
            whole[0], whole[1], whole[2], whole[3], whole[4], whole[5], ns.len()
        );
    }
}

/// Dynamic range and upper tail of each column, within a night. `terms` justifies ranking `turn`
/// because it spans three orders of magnitude in one night; this is whether the columns v2
/// z-scores instead do the same thing.
fn shape_census(loaded: &[(&str, Vec<Night>)]) {
    let names = ["hr", "hr_var", "move_frac", "flat", "turn", "resp"];
    println!("\nCOLUMN SHAPE WITHIN A NIGHT (median across that cohort's nights)");
    println!(
        "  {:<13} {:<10} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "cohort", "column", "p50", "max", "max/p50", "skew", "z>3 share"
    );
    for (cname, ns) in loaded {
        for (i, n) in names.iter().enumerate() {
            let (mut p50s, mut maxs, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
            let (mut skews, mut tails, mut zero_p50) = (Vec::new(), Vec::new(), 0usize);
            for nt in ns {
                let col = match i {
                    0 => &nt.hr,
                    1 => &nt.hr_var,
                    2 => &nt.mv,
                    3 => &nt.flat,
                    4 => &nt.turn,
                    _ => &nt.resp,
                };
                let v: Vec<f64> = col.iter().flatten().copied().collect();
                if v.len() < 10 {
                    continue;
                }
                let (m, mx) = (stats::median(&v), stats::max(&v));
                p50s.push(m);
                maxs.push(mx);
                if m > 0.0 {
                    ratios.push(mx / m);
                } else {
                    zero_p50 += 1;
                }
                let (mu, sd) = (stats::mean(&v), stats::population_sd(&v));
                if sd > 0.0 {
                    skews.push(v.iter().map(|x| ((x - mu) / sd).powi(3)).sum::<f64>() / v.len() as f64);
                    tails.push(v.iter().filter(|x| (**x - mu) / sd > 3.0).count() as f64 / v.len() as f64);
                }
            }
            if p50s.is_empty() {
                continue;
            }
            let ratio = if ratios.is_empty() {
                format!("p50=0 x{zero_p50}")
            } else if zero_p50 > 0 {
                format!("{:.0} (+{zero_p50})", stats::median(&ratios))
            } else {
                format!("{:.1}", stats::median(&ratios))
            };
            println!(
                "  {cname:<13} {n:<10} {:>10.4} {:>10.4} {ratio:>10} {:>8.2} {:>9.2}%",
                stats::median(&p50s),
                stats::median(&maxs),
                stats::median(&skews),
                100.0 * stats::median(&tails)
            );
        }
    }
}

/// Night length and window composition against v2's own per-night kappa. A z-score over a 4 h window
/// and over a 12 h one are different instruments only if the difference shows up here.
fn length_census(loaded: &[(&str, Vec<Night>)], p: &Params) {
    println!("\nNIGHT LENGTH AND WINDOW COMPOSITION vs V2 KAPPA");
    println!(
        "  {:<14} {:>5} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9}",
        "cohort", "n", "min ep", "med ep", "max ep", "r(len,k)", "wake frac", "r(wake,k)"
    );
    for (name, ns) in loaded {
        let (mut len, mut kap, mut wake) = (Vec::new(), Vec::new(), Vec::new());
        for nt in ns {
            let Some(k) = kappa_of(nt, &nt.em, p) else { continue };
            let lab: Vec<usize> = nt.truth.iter().flatten().copied().collect();
            len.push(nt.epochs as f64);
            kap.push(k);
            wake.push(lab.iter().filter(|l| **l == 0).count() as f64 / lab.len().max(1) as f64);
        }
        let mut sorted = len.clone();
        sorted.sort_by(f64::total_cmp);
        println!(
            "  {name:<14} {:>5} {:>7.0} {:>7.0} {:>7.0} {:>9.3} {:>9.3} {:>9.3}",
            len.len(),
            sorted[0],
            median(&mut sorted.clone()),
            sorted[sorted.len() - 1],
            stats::pearson(&len, &kap).unwrap_or(f64::NAN),
            amean(&wake),
            stats::pearson(&wake, &kap).unwrap_or(f64::NAN)
        );
    }

    // The sleep-anchored variant should pay off most where the window is least like the sleep
    // period. If it does not correlate, the window is not the instrument being blamed.
    println!("\n  sleep-period-anchored statistics: does the gain track how much window is NOT sleep?");
    let v = Variant { stats_on_sleep: true, ..SHIPPED_V };
    for (name, ns) in loaded {
        let (mut frac, mut d) = (Vec::new(), Vec::new());
        for nt in ns {
            let Some(bk) = kappa_of(nt, &nt.em, p) else { continue };
            let vk = kappa_of(nt, &emissions_under(nt, &v, p), p).expect("same night");
            frac.push(1.0 - nt.sleep.iter().filter(|s| **s).count() as f64 / nt.epochs as f64);
            d.push(vk - bk);
        }
        println!(
            "    {name:<14} mean non-sleep share of window {:.3}   pearson r with the delta {:>7.3}",
            amean(&frac),
            stats::pearson(&frac, &d).unwrap_or(f64::NAN)
        );
    }
}

/// Which tanv1 design columns a cohort never carries, and which standardise to a constant. Both
/// paths in `col_stats` are silent, so the only way to know is to count.
fn tanv1_census(loaded: &[(&str, Vec<Night>)]) {
    let ncol = Features::N + N_NL;
    let mut names: Vec<String> = Features::NAMES.iter().map(|s| (*s).to_string()).collect();
    for n in ["nl_deep_hinge", "nl_dz_hr_var", "nl_dz_hr", "nl_turn_rank", "nl_resp_z", "nl_clamp"] {
        names.push(n.to_string());
    }

    println!("\nTANV1 DESIGN: columns materially imputed (NaN share per cohort, > 0 only)");
    let mut cov: Vec<Vec<f64>> = Vec::new();
    for (_, ns) in loaded {
        let mut miss = vec![0usize; ncol];
        let mut tot = 0usize;
        for nt in ns {
            for r in &nt.tan {
                tot += 1;
                for (c, v) in r.iter().enumerate() {
                    if !v.is_finite() {
                        miss[c] += 1;
                    }
                }
            }
        }
        cov.push(miss.iter().map(|m| 100.0 * *m as f64 / tot.max(1) as f64).collect());
    }
    println!("  {:<18} {:>10} {:>10} {:>10}", "column", loaded[0].0, loaded[1].0, loaded[2].0);
    let mut any = false;
    for c in 0..ncol {
        if cov.iter().any(|v| v[c] > 0.0) {
            any = true;
            println!("  {:<18} {:>9.2}% {:>9.2}% {:>9.2}%", names[c], cov[0][c], cov[1][c], cov[2][c]);
        }
    }
    if !any {
        println!("  (none: every column is finite on every epoch of all three cohorts)");
    }

    println!("\nTANV1 DESIGN: per-fold column statistics that hit a silent guard");
    for (held, _) in loaded {
        let x: Vec<Vec<f64>> = loaded
            .iter()
            .filter(|(c, _)| c != held)
            .flat_map(|(_, ns)| ns.iter().flat_map(|nt| nt.tan.iter().cloned()))
            .collect();
        let mut flat_cols = Vec::new();
        let mut empty_cols = Vec::new();
        for c in 0..ncol {
            let v: Vec<f64> = x.iter().map(|r| r[c]).filter(|v| v.is_finite()).collect();
            if v.is_empty() {
                empty_cols.push(names[c].clone());
            } else if stats::population_sd(&v) <= 1e-12 {
                flat_cols.push(names[c].clone());
            }
        }
        println!(
            "  train without {held:<14} constant columns {:?}  all-NaN columns {:?}",
            flat_cols, empty_cols
        );
    }

    // The held-out cohort's rows standardised against the TRAIN statistics: how far the imputed
    // zero sits from where that cohort's own rows actually land.
    println!("\nTANV1 DESIGN: held-out mean of each standardised column, worst 6 by |mean|");
    for (held, ns) in loaded {
        let x: Vec<Vec<f64>> = loaded
            .iter()
            .filter(|(c, _)| c != held)
            .flat_map(|(_, tn)| tn.iter().flat_map(|nt| nt.tan.iter().cloned()))
            .collect();
        let (m, sd) = common::lr::standardise_cols(&x);
        let mut acc = vec![0.0f64; ncol];
        let mut cnt = 0usize;
        for nt in ns {
            for r in &nt.tan {
                let d = design_row(r, &m, &sd, &[]);
                cnt += 1;
                for c in 0..ncol {
                    acc[c] += d[c];
                }
            }
        }
        let mut idx: Vec<usize> = (0..ncol).collect();
        idx.sort_by(|a, b| (acc[*b] / cnt as f64).abs().total_cmp(&(acc[*a] / cnt as f64).abs()));
        let top: Vec<String> = idx[..6]
            .iter()
            .map(|c| format!("{}={:+.2}", names[*c], acc[*c] / cnt as f64))
            .collect();
        println!("  held-out {held:<14} {}", top.join("  "));
    }
}

// ---------------------------------------------------------------------------------------------
// tanv1 side: what a missing column becomes, and where the standardising statistics come from

/// Where a tanv1 design row's location and scale come from, and what a NaN becomes.
#[derive(Clone, Copy, PartialEq)]
enum Policy {
    /// Pooled TRAIN mean and sd, NaN imputed to the train mean. What `design_row` does.
    TrainMean,
    /// The same, plus one indicator per column that is ever missing in train.
    Flagged,
    /// Each night's OWN mean and sd, so no cross-cohort statistic reaches the row at all.
    PerNight,
}

const FIT_ITERS: usize = 6_000;
const FIT_TOL: f64 = 1e-10;
const FIT_LR: f64 = 0.5;
const FIT_L2: f64 = 1.0;
const FIT_POWER: f64 = 0.5;

/// Column mean and population sd over the finite values of one night, with `col_stats`' own guards.
fn night_stats(rows: &[Vec<f64>], ncol: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut m, mut s) = (vec![0.0; ncol], vec![1.0; ncol]);
    for c in 0..ncol {
        let v: Vec<f64> = rows.iter().map(|r| r[c]).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        m[c] = stats::mean(&v);
        let sd = stats::population_sd(&v);
        s[c] = if sd > 1e-12 { sd } else { 1.0 };
    }
    (m, s)
}

/// One night's design under `pol`. Every arm ends in the bias column the optimiser exempts from L2.
fn build_rows(nt: &Night, pol: Policy, m: &[f64], sd: &[f64], flags: &[usize]) -> Vec<Vec<f64>> {
    let ncol = m.len();
    let (nm, nsd) = if pol == Policy::PerNight { night_stats(&nt.tan, ncol) } else { (vec![], vec![]) };
    nt.tan
        .iter()
        .map(|r| match pol {
            Policy::TrainMean => design_row(r, m, sd, &[]),
            Policy::Flagged => {
                let mut d = design_row(r, m, sd, &[]);
                let bias = d.pop().expect("design_row appends a bias");
                d.extend(flags.iter().map(|c| f64::from(!r[*c].is_finite())));
                d.push(bias);
                d
            }
            Policy::PerNight => design_row(r, &nm, &nsd, &[]),
        })
        .collect()
}

/// Our class index -> the emission's own column.
fn col_of(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(physio_algo::sleep::STAGE_ORDER[*c]) == class).expect("class")
}

fn class_weights(y: &[usize]) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for c in y {
        n[*c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(FIT_POWER)
        } else {
            0.0
        };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] as f64 * w[c]).sum::<f64>() / y.len() as f64;
    for v in w.iter_mut() {
        *v /= mass;
    }
    w
}

/// Multinomial logistic regression on top of a fixed per-class offset, as `fit_residual` runs it.
/// Zero weights leave the offset untouched, which IS the shipped recipe.
fn fit_residual(x: &[Vec<f64>], off: &[[f64; CLASSES]], y: &[usize]) -> (Vec<Vec<f64>>, bool) {
    let p = x[0].len();
    let cw = class_weights(y);
    let mut th = vec![vec![0.0f64; p]; CLASSES];
    let mut last = f64::MAX;
    let mut converged = false;
    for _ in 0..FIT_ITERS {
        let mut g = vec![vec![0.0f64; p]; CLASSES];
        let mut nll = 0.0f64;
        for ((row, o), &lab) in x.iter().zip(off).zip(y) {
            let mut z = [0.0f64; CLASSES];
            for c in 0..CLASSES {
                z[c] = o[col_of(c)] + th[c].iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
            }
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let ex: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
            let sum: f64 = ex.iter().sum();
            nll -= cw[lab] * (ex[lab] / sum).max(1e-300).ln();
            for c in 0..CLASSES {
                let err = cw[lab] * (ex[c] / sum - if c == lab { 1.0 } else { 0.0 });
                for (gi, xi) in g[c].iter_mut().zip(row) {
                    *gi += err * xi;
                }
            }
        }
        let nll = nll / x.len() as f64;
        let drop = last - nll;
        if (0.0..FIT_TOL).contains(&drop) {
            converged = true;
            break;
        }
        last = nll;
        let scale = FIT_LR / x.len() as f64;
        let decay = (1.0 - FIT_LR * FIT_L2).max(0.0);
        for c in 0..CLASSES {
            for j in 0..p {
                th[c][j] = decay * th[c][j] - scale * g[c][j];
            }
        }
    }
    (th, converged)
}

/// Paired per-night kappa of v2 and of the corrected emission, under one imputation policy.
fn score_policy(
    nights: &[Night],
    th: &[Vec<f64>],
    pol: Policy,
    m: &[f64],
    sd: &[f64],
    flags: &[usize],
    p: &Params,
) -> Vec<f64> {
    let mut d = Vec::new();
    for nt in nights {
        let rows = build_rows(nt, pol, m, sd, flags);
        let em: Vec<[f64; CLASSES]> = rows
            .iter()
            .zip(&nt.em)
            .map(|(r, o)| {
                let mut out = *o;
                for c in 0..CLASSES {
                    out[col_of(c)] += th[c].iter().zip(r).map(|(a, b)| a * b).sum::<f64>();
                }
                out
            })
            .collect();
        let (Some(bk), Some(vk)) = (kappa_of(nt, &nt.em, p), kappa_of(nt, &em, p)) else { continue };
        d.push(vk - bk);
    }
    d
}

/// The three policies over the same folds. The `TrainMean` row is the control and must land on
/// `fit_residual`'s own held-out numbers, or nothing below it is comparable.
fn imputation_arms(loaded: &[(&str, Vec<Night>)], p: &Params) {
    let ncol = Features::N + N_NL;
    println!("\nTANV1 IMPUTATION POLICY (L2 {FIT_L2}, held-out paired delta vs v2)");
    println!("  {:<14} {:<12} {:>10} {:>9} {:>5} {:>6}   verdict", "held out", "policy", "paired d", "bar +/-", "n", "cols");
    for (h, (held, hn)) in loaded.iter().enumerate() {
        let train: Vec<&Night> =
            loaded.iter().enumerate().filter(|(i, _)| *i != h).flat_map(|(_, (_, n))| n.iter()).collect();
        let raw: Vec<Vec<f64>> = train.iter().flat_map(|nt| nt.tan.iter().cloned()).collect();
        let (m, sd) = common::lr::standardise_cols(&raw);
        let flags: Vec<usize> =
            (0..ncol).filter(|c| raw.iter().any(|r| !r[*c].is_finite())).collect();

        for pol in [Policy::TrainMean, Policy::Flagged, Policy::PerNight] {
            let mut x: Vec<Vec<f64>> = Vec::new();
            let mut off: Vec<[f64; CLASSES]> = Vec::new();
            let mut y: Vec<usize> = Vec::new();
            for nt in &train {
                let rows = build_rows(nt, pol, &m, &sd, &flags);
                for (k, t) in nt.truth.iter().enumerate() {
                    if let Some(t) = t {
                        x.push(rows[k].clone());
                        off.push(nt.em[k]);
                        y.push(*t);
                    }
                }
            }
            let width = x[0].len();
            let (th, conv) = fit_residual(&x, &off, &y);
            let d = score_policy(hn, &th, pol, &m, &sd, &flags, p);
            let (mm, bar, v) = verdict(&d);
            let tag = match pol {
                Policy::TrainMean => "train-mean",
                Policy::Flagged => "flagged",
                Policy::PerNight => "per-night",
            };
            println!(
                "  {held:<14} {tag:<12} {mm:>+10.4} {bar:>9.4} {:>5} {width:>6}   {v}{}",
                d.len(),
                if conv { "" } else { "  [DID NOT CONVERGE]" }
            );
        }
    }
}

fn main() {
    let p = Params::SHIPPED;
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    assert_eq!(loaded.len(), 3, "all three cohorts must be under the fixture root");

    positive_controls(&loaded, &p);
    v2_missing_census(&loaded);
    shape_census(&loaded);
    length_census(&loaded, &p);
    tanv1_census(&loaded);

    // Every variant is SHIPPED_V with one axis moved, so a row's effect is that axis alone.
    let mut variants: Vec<(String, Variant)> = Vec::new();
    for (tag, t) in [("rankit", Tr::Rankit), ("robustz", Tr::RobustZ)] {
        variants.push((format!("cardiac={tag}"), Variant { card: t, ..SHIPPED_V }));
        variants.push((format!("motion={tag}"), Variant { motion: t, ..SHIPPED_V }));
        variants.push((format!("resp={tag}"), Variant { resp: t, ..SHIPPED_V }));
        variants.push((
            format!("all-three={tag}"),
            Variant { card: t, motion: t, resp: t, ..SHIPPED_V },
        ));
    }
    variants.push(("stats-on-sleep-period".into(), Variant { stats_on_sleep: true, ..SHIPPED_V }));
    variants.push(("gate=normal-cdf".into(), Variant { gate: GateK::NormalCdf, ..SHIPPED_V }));
    variants.push(("gate=midrank".into(), Variant { gate: GateK::Midrank, ..SHIPPED_V }));
    variants.push(("gate-missing=abstain".into(), Variant { gate_abstain: true, ..SHIPPED_V }));
    variants.push(("hinge=linear".into(), Variant { hinge: HingeK::Linear, ..SHIPPED_V }));
    for s in [0.02, 0.05, 0.1] {
        variants.push((format!("hinge=softplus s={s}"), Variant { hinge: HingeK::Soft(s), ..SHIPPED_V }));
    }
    variants.push(("deadzone=tanh".into(), Variant { dead: DeadK::Tanh, ..SHIPPED_V }));
    for s in [0.1, 0.25, 0.5] {
        variants.push((format!("clamp=softplus s={s}"), Variant { clamp: ClampK::PairSoft(s), ..SHIPPED_V }));
    }
    variants.push(("clamp=per-term".into(), Variant { clamp: ClampK::PerTerm, ..SHIPPED_V }));

    println!("\nPER-COHORT PAIRED DELTA vs v2 (positive = the variant beats v2 on the same nights)");
    println!("  churn = share of epochs whose decoded label moved, so a null is told from a no-op");
    println!(
        "  {:<24} {:>17} {:>7} {:>17} {:>7} {:>17} {:>7}",
        "variant", loaded[0].0, "churn", loaded[1].0, "churn", loaded[2].0, "churn"
    );
    let mut table: Vec<Scored> = Vec::new();
    for (name, v) in &variants {
        let mut row = Vec::new();
        let mut cells = Vec::new();
        for (_, ns) in &loaded {
            let (d, churn) = arm(ns, v, &p);
            let (m, bar, _) = verdict(&d);
            row.push((m, bar));
            cells.push(format!("{m:+.4}+/-{bar:.4}"));
            cells.push(format!("{churn:.2}%"));
        }
        println!(
            "  {:<24} {:>17} {:>7} {:>17} {:>7} {:>17} {:>7}",
            name, cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]
        );
        table.push(Scored { name: name.clone(), per_cohort: row });
    }

    println!("\nBASELINE v2 median kappa per cohort");
    for (name, ns) in &loaded {
        let mut k: Vec<f64> = ns.iter().filter_map(|nt| kappa_of(nt, &nt.em, &p)).collect();
        println!("  {name:<14} n={:<4} median {:.4}", k.len(), median(&mut k));
    }

    // Selection: the variant is chosen on the two TRAINING cohorts and reported once on the held-out
    // one. A non-fitted transform has nothing to fit, so the inner check is that both training
    // cohorts agree on the sign; that agreement is printed beside the choice.
    println!("\nHELD-OUT REPORT (variant chosen on the two training cohorts, reported once)");
    for (h, (held, _)) in loaded.iter().enumerate() {
        let tr: Vec<usize> = (0..3).filter(|i| *i != h).collect();
        let best = table
            .iter()
            .max_by(|a, b| {
                let sa = amean(&tr.iter().map(|i| a.per_cohort[*i].0).collect::<Vec<_>>());
                let sb = amean(&tr.iter().map(|i| b.per_cohort[*i].0).collect::<Vec<_>>());
                sa.total_cmp(&sb)
            })
            .expect("a non-empty variant table");
        let agree = tr.iter().all(|i| best.per_cohort[*i].0 > 0.0);
        let (m, bar) = best.per_cohort[h];
        let v = if m.abs() > bar {
            format!("{} ({:.2}x)", if m > 0.0 { "BEATS v2" } else { "worse" }, m.abs() / bar)
        } else {
            "matches".to_string()
        };
        println!(
            "  held-out {held:<14} chose {:<26} train mean {:+.4} (both positive: {agree})   \
             held-out {m:+.4} +/- {bar:.4}   {v}",
            best.name,
            amean(&tr.iter().map(|i| best.per_cohort[*i].0).collect::<Vec<_>>())
        );
    }

    imputation_arms(&loaded, &p);
}
