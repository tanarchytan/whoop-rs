//! Negative control for the `vitals` metric family. It falsifies one claim: "the shipped vitals gates
//! are regression tests". A gate that reproduces its own literal proves the harness reaches the
//! algorithm; it does NOT prove the gate would notice the algorithm drifting. This file measures the
//! difference by running each gate against arms that do no work (NULL), arms with the wrong shape
//! (STRUCTURAL), and arms with one tunable moved (PARAMETER), then reports which arms the SHIPPED gate
//! catches and which it misses.
//!
//! Nothing here is a health claim. Every number is a wellness estimate, never medical or diagnostic.
//!
//! Parameter arms cannot mutate private constants from an integration test, so each one runs a local
//! SHADOW of the algorithm parameterised on that constant. Every shadow is asserted to reproduce the
//! shipped value at baseline parameters, which is what makes the sweep meaningful.
//!
//!   cargo test --release -p physio-algo --test sensitivity_vitals -- --ignored --nocapture

use physio_algo::biological_age::{cosinor_age, Sex};
use physio_algo::calibration::{Calibration, BLOOD_OXYGEN, CALORIES, DAY_STRAIN, SKIN_TEMP};
use physio_algo::circadian::{cosinor, estimate_phase, ActivityBin, PhaseConfidence};
use physio_algo::hrv::{
    duplicate_beat_count, overlapping_report_count, rolling_rmssd, rr_coverage, HrvReadiness,
};
use physio_algo::hrv_freq::freq_domain;
use physio_algo::resting_hr::{daily_resting_hr, session_resting_hr_floor, HrSample};
use physio_algo::respiratory_rate::resp_rate_from_rr;
use physio_algo::signal::{find_peaks, moving_average_centred};
use physio_algo::spo2::Spo2;
use physio_algo::stats::{
    amplitude, half_change, mean, median, percentile, population_sd, trend_min_span_days,
    weighted_trendline, TrendDirection,
};
use physio_algo::vitality::{compute, contributions, rmssd_norm, sleep_consistency, VitalityInput};
use physio_algo::worn::{worn_state, WornState};
use physio_algo::{HrWatch, HrWatchState};
use whoop_protocol::HistoryRecord;

// ─────────────────────────── harness ───────────────────────────

/// One shipped gate, copied literal-for-literal from the source line named in `source`.
struct Gate {
    label: &'static str,
    source: &'static str,
    target: f64,
    tol: f64,
}

impl Gate {
    fn holds(&self, v: f64) -> bool {
        (v - self.target).abs() <= self.tol
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Param,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Baseline => "base",
            Kind::Null => "null",
            Kind::Structural => "struct",
            Kind::Param => "param",
        }
    }
}

struct Row {
    arm: String,
    kind: Kind,
    value: f64,
}

struct Table {
    metric: &'static str,
    gate: Gate,
    rows: Vec<Row>,
    notes: Vec<String>,
    /// False only where the gate is PROVEN blind to a do-nothing scorer; that is the finding itself,
    /// so the file reports it rather than asserting a defect into place.
    null_must_fail: bool,
}

impl Table {
    fn new(metric: &'static str, gate: Gate) -> Self {
        Table { metric, gate, rows: Vec::new(), notes: Vec::new(), null_must_fail: true }
    }
    fn blind_null(mut self) -> Self {
        self.null_must_fail = false;
        self
    }
    fn add(&mut self, kind: Kind, arm: impl Into<String>, value: f64) {
        self.rows.push(Row { arm: arm.into(), kind, value });
    }
    fn note(&mut self, s: impl Into<String>) {
        self.notes.push(s.into());
    }
}

#[derive(Default)]
struct Tally {
    caught: usize,
    missed: usize,
    floor: Option<f64>,
    criticals: Vec<String>,
}

fn fmt_val(v: f64) -> String {
    if v.is_nan() {
        "refused/None".to_string()
    } else {
        format!("{v:.9}")
    }
}

fn fmt_delta(v: f64, base: f64) -> String {
    if v.is_nan() || base.is_nan() {
        "n/a".to_string()
    } else {
        format!("{:+.9}", v - base)
    }
}

/// Print one metric's table, fold it into the global tally, and enforce the two trust assertions:
/// the baseline reproduces the shipped literal, and a do-nothing scorer fails the shipped gate.
fn report(t: &Table, tally: &mut Tally) {
    println!("\n=== {} ===", t.metric);
    println!("shipped gate: {} = {} +/- {}   [{}]", t.gate.label, t.gate.target, t.gate.tol, t.gate.source);
    println!("{:<56} {:>6} {:>16} {:>16}  shipped gate", "arm", "kind", "value", "delta");

    let base = t
        .rows
        .iter()
        .find(|r| r.kind == Kind::Baseline)
        .map(|r| r.value)
        .unwrap_or(f64::NAN);
    assert!(!base.is_nan(), "{}: baseline produced no value", t.metric);
    assert!(
        t.gate.holds(base),
        "{}: baseline {base} does not reproduce the shipped literal {} +/- {}",
        t.metric,
        t.gate.target,
        t.gate.tol
    );

    let (mut caught, mut missed) = (0usize, 0usize);
    let mut floor: Option<f64> = None;
    let mut null_caught = false;
    for r in &t.rows {
        let holds = t.gate.holds(r.value);
        let verdict = if r.kind == Kind::Baseline {
            "PASS (expected)".to_string()
        } else if holds {
            missed += 1;
            "PASS  <-- MISSED".to_string()
        } else {
            caught += 1;
            if r.kind == Kind::Null {
                null_caught = true;
            }
            if !r.value.is_nan() {
                let d = (r.value - base).abs();
                if d > 0.0 {
                    floor = Some(floor.map_or(d, |f: f64| f.min(d)));
                }
            }
            "FAIL  <-- caught".to_string()
        };
        println!(
            "{:<56} {:>6} {:>16} {:>16}  {}",
            r.arm,
            r.kind.tag(),
            fmt_val(r.value),
            fmt_delta(r.value, base),
            verdict
        );
    }
    for n in &t.notes {
        println!("  note: {n}");
    }
    match floor {
        Some(f) => println!("  caught {caught}, missed {missed}; smallest caught delta {f:.9}"),
        None => println!("  caught {caught}, missed {missed}; no finite-delta arm was caught"),
    }

    if t.null_must_fail {
        assert!(
            null_caught,
            "{}: a NULL arm passed the shipped gate - the gate is fake",
            t.metric
        );
    } else if !null_caught {
        let msg = format!("{}: a do-nothing scorer PASSES the shipped gate ({})", t.metric, t.gate.source);
        println!("  CRITICAL: {msg}");
        tally.criticals.push(msg);
    }

    let probes: Vec<(&str, f64)> = t
        .rows
        .iter()
        .filter(|r| matches!(r.kind, Kind::Null | Kind::Structural))
        .map(|r| (r.arm.as_str(), r.value))
        .collect();
    enforce_floors(t.metric, base, &probes);

    tally.caught += caught;
    tally.missed += missed;
    if let Some(f) = floor {
        tally.floor = Some(tally.floor.map_or(f, |g: f64| g.min(f)));
    }
}

/// Multiplicative ladder for the break search, smallest first.
const LADDER: [f64; 10] = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00];

/// The smallest ladder step at which scaling one tunable breaks the gate. `f` takes the multiplier.
fn break_at(gate: &Gate, f: &dyn Fn(f64) -> f64) -> String {
    for d in LADDER {
        let (up, dn) = (f(1.0 + d), f(1.0 - d));
        let (bu, bd) = (!gate.holds(up), !gate.holds(dn));
        if bu || bd {
            let sign = if bu && bd { "+/-" } else if bu { "+" } else { "-" };
            return format!("breaks at {sign}{:.2}%", d * 100.0);
        }
    }
    "NO break within +/-100%".to_string()
}

/// Deterministic in-place shuffle (xorshift64), so a "shuffled input" arm is reproducible.
fn shuffled<T: Copy>(xs: &[T]) -> Vec<T> {
    let mut v = xs.to_vec();
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..v.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// Scale an integer tunable by a multiplier, never below zero.
fn scale_u(base: usize, f: f64) -> usize {
    (base as f64 * f).round().max(0.0) as usize
}

fn nan_if_none(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

// ─────────────────────────── shadows: R-R cleaning + RMSSD ───────────────────────────

/// The R-R cleaning tunables (hrv.rs:23-41).
#[derive(Clone, Copy)]
struct RrParams {
    rr_min: u16,
    rr_max: u16,
    ect_thresh: f64,
    ect_radius: usize,
    max_beat_delta: f64,
}

const RR_BASE: RrParams =
    RrParams { rr_min: 300, rr_max: 2000, ect_thresh: 0.20, ect_radius: 2, max_beat_delta: 200.0 };

/// Range filter then Malik ectopic rejection, keeping each survivor's INPUT index. Mirrors
/// `hrv::clean_rr_kept`.
fn shadow_clean_kept(rr: &[u16], p: RrParams) -> (Vec<usize>, Vec<u16>) {
    let mut ranged_idx: Vec<usize> = Vec::new();
    let mut ranged_val: Vec<u16> = Vec::new();
    for (i, &v) in rr.iter().enumerate() {
        if (p.rr_min..=p.rr_max).contains(&v) {
            ranged_idx.push(i);
            ranged_val.push(v);
        }
    }
    if ranged_val.len() <= p.ect_radius {
        return (ranged_idx, ranged_val);
    }
    let (mut kept_orig, mut kept_val): (Vec<usize>, Vec<u16>) = (Vec::new(), Vec::new());
    for i in 0..ranged_val.len() {
        let lo = i.saturating_sub(p.ect_radius);
        let hi = (i + p.ect_radius).min(ranged_val.len() - 1);
        let mut neighbours: Vec<f64> = Vec::new();
        for (j, &v) in ranged_val.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v as f64);
            }
        }
        let keep = if neighbours.len() < 2 {
            true
        } else {
            let med = median(&neighbours);
            med <= 0.0 || (ranged_val[i] as f64 - med).abs() / med <= p.ect_thresh
        };
        if keep {
            kept_orig.push(ranged_idx[i]);
            kept_val.push(ranged_val[i]);
        }
    }
    (kept_orig, kept_val)
}

/// Cleaned values plus the contiguity mask. Mirrors `hrv::clean_rr_gap_aware_breaking`.
fn shadow_clean_gap_aware(rr: &[u16], breaks: &[bool], p: RrParams) -> (Vec<u16>, Vec<bool>) {
    let (kept_orig, kept_val) = shadow_clean_kept(rr, p);
    let contiguous: Vec<bool> = (0..kept_val.len())
        .map(|i| {
            i > 0
                && kept_orig[i] == kept_orig[i - 1] + 1
                && !breaks.get(kept_orig[i]).copied().unwrap_or(false)
        })
        .collect();
    (kept_val, contiguous)
}

/// Task-Force RMSSD over contiguous pairs only. Mirrors `hrv::rmssd_from_clean`.
fn shadow_rmssd_from_clean(nn: &[u16], contiguous: &[bool]) -> f64 {
    let (mut sumsq, mut count) = (0.0f64, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        let d = nn[i] as f64 - nn[i - 1] as f64;
        sumsq += d * d;
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        (sumsq / count as f64).sqrt()
    }
}

/// Gap-aware RMSSD over one single-timestamp report, where no seam break can arise. Mirrors
/// `HrvReadiness::rmssd_gap_aware` for that input shape.
fn shadow_rmssd_gap_aware(rr: &[u16], p: RrParams) -> f64 {
    let (nn, contiguous) = shadow_clean_gap_aware(rr, &[], p);
    shadow_rmssd_from_clean(&nn, &contiguous)
}

/// Run RMSSD with the artifact-delta filter. Mirrors `HrvReadiness::rmssd_runs` over one run.
fn shadow_rmssd_run(rr: &[u16], p: RrParams) -> f64 {
    let (mut sumsq, mut pairs) = (0.0f64, 0usize);
    for w in rr.windows(2) {
        let d = w[1] as f64 - w[0] as f64;
        if d.abs() > p.max_beat_delta {
            continue;
        }
        sumsq += d * d;
        pairs += 1;
    }
    if pairs == 0 {
        f64::NAN
    } else {
        (sumsq / pairs as f64).sqrt()
    }
}

/// Plain RMSSD, no cleaning, every pair counted. Mirrors `HrvReadiness::rmssd_plain`.
fn shadow_rmssd_plain(rr: &[u16]) -> Option<f64> {
    if rr.len() < 2 {
        return None;
    }
    let mut sum_sq = 0.0;
    for i in 1..rr.len() {
        let d = rr[i] as f64 - rr[i - 1] as f64;
        sum_sq += d * d;
    }
    Some((sum_sq / (rr.len() - 1) as f64).sqrt())
}

/// Sample SD of NN intervals. Mirrors `HrvReadiness::sdnn`.
fn shadow_sdnn(rr: &[u16]) -> f64 {
    if rr.len() < 2 {
        return f64::NAN;
    }
    let m = rr.iter().map(|&v| v as f64).sum::<f64>() / rr.len() as f64;
    let var = rr.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / (rr.len() - 1) as f64;
    var.sqrt()
}

/// pNN50 over contiguous pairs. Mirrors `hrv::pnn50_from_clean`.
fn shadow_pnn50(nn: &[u16], contiguous: &[bool], threshold_ms: f64) -> f64 {
    let (mut nn50, mut pairs) = (0usize, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        if (nn[i] as f64 - nn[i - 1] as f64).abs() > threshold_ms {
            nn50 += 1;
        }
        pairs += 1;
    }
    if pairs == 0 {
        f64::NAN
    } else {
        nn50 as f64 / pairs as f64 * 100.0
    }
}

/// Full clean-and-analyze, returning SDNN. Mirrors the gating and SDNN leg of
/// `HrvReadiness::analyze_raw`.
fn shadow_analyze_sdnn(rr: &[u16], min_beats: usize, max_rejected: Option<f64>, p: RrParams) -> f64 {
    let n_input = rr.len();
    let (nn, _) = shadow_clean_gap_aware(rr, &[], p);
    if nn.len() < min_beats {
        return f64::NAN;
    }
    if let Some(max_rej) = max_rejected {
        if n_input > 0 && 1.0 - nn.len() as f64 / n_input as f64 > max_rej {
            return f64::NAN;
        }
    }
    shadow_sdnn(&nn)
}

/// Which report-first beats re-report already-covered time. Mirrors `hrv::report_seam_breaks`.
fn shadow_seam_breaks(reports: &[(u32, Vec<u16>)], slack_ms: u64) -> Vec<bool> {
    let mut out = Vec::new();
    let Some((t0, _)) = reports.first() else { return out };
    let mut covered_ms: u64 = 0;
    for (t, rr) in reports {
        let elapsed_ms = u64::from(t.saturating_sub(*t0)) * 1000;
        for (j, _) in rr.iter().enumerate() {
            out.push(j == 0 && covered_ms > elapsed_ms + slack_ms);
        }
        covered_ms += rr.iter().map(|&v| u64::from(v)).sum::<u64>();
    }
    out
}

/// Mean of per-bucket gap-aware RMSSD. Mirrors `windowed_buckets` + `windowed_avg_hrv`.
fn shadow_windowed_avg(
    start: u32,
    end: u32,
    beats: &[(u32, u16)],
    window_secs: u64,
    slack_ms: u64,
    p: RrParams,
) -> f64 {
    let seg: Vec<(u32, u16)> =
        beats.iter().copied().filter(|&(t, _)| t >= start && t <= end).collect();
    // A zero-width window buckets nothing and would never advance `t`; the -100% ladder rung reaches it.
    if seg.is_empty() || window_secs == 0 {
        return f64::NAN;
    }
    let (mut sum, mut n) = (0.0f64, 0usize);
    let mut t = start as u64;
    while t < end as u64 {
        let hi = t + window_secs;
        let mut reports: Vec<(u32, Vec<u16>)> = Vec::new();
        for &(ts, rr) in seg.iter().filter(|&&(ts, _)| ts as u64 >= t && (ts as u64) < hi) {
            match reports.last_mut() {
                Some((rt, v)) if *rt == ts => v.push(rr),
                _ => reports.push((ts, vec![rr])),
            }
        }
        let bucket: Vec<u16> = reports.iter().flat_map(|(_, rr)| rr.iter().copied()).collect();
        let breaks = shadow_seam_breaks(&reports, slack_ms);
        let (nn, contiguous) = shadow_clean_gap_aware(&bucket, &breaks, p);
        if nn.len() >= 2 {
            let v = shadow_rmssd_from_clean(&nn, &contiguous);
            if !v.is_nan() {
                sum += v;
                n += 1;
            }
        }
        t = hi;
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

/// Trailing-window RMSSD series. Mirrors `hrv::rolling_rmssd`.
fn shadow_rolling(
    beats: &[(i64, u16)],
    window_s: i64,
    step_s: i64,
    min_beats: usize,
    p: RrParams,
) -> Vec<(i64, f64)> {
    if beats.len() < min_beats || window_s <= 0 {
        return Vec::new();
    }
    let mut sorted: Vec<(i64, u16)> = beats.to_vec();
    sorted.sort_by_key(|&(t, _)| t);
    let vals: Vec<u16> = sorted.iter().map(|&(_, v)| v).collect();
    let (kept_idx, kept_val) = shadow_clean_kept(&vals, p);
    if kept_val.len() < 2 {
        return Vec::new();
    }
    let kept: Vec<(i64, u16)> = kept_idx.iter().map(|&i| (sorted[i].0, vals[i])).collect();
    let mut out: Vec<(i64, f64)> = Vec::new();
    let mut lo = 0usize;
    let mut last_emit: Option<i64> = None;
    for hi in 0..kept.len() {
        let t_end = kept[hi].0;
        let t_start = t_end - window_s;
        while lo < hi && kept[lo].0 <= t_start {
            lo += 1;
        }
        if let Some(last) = last_emit {
            if step_s > 0 && t_end - last < step_s {
                continue;
            }
        }
        let span: Vec<u16> = kept[lo..=hi].iter().map(|&(_, v)| v).collect();
        if span.len() < min_beats {
            continue;
        }
        if let Some(r) = shadow_rmssd_plain(&span) {
            out.push((t_end, r));
            last_emit = Some(t_end);
        }
    }
    out
}

/// Overlapping-report count under a variable seam slack. Mirrors `hrv::overlapping_report_count`.
fn shadow_overlapping(reports: &[(u32, Vec<u16>)], slack_ms: u64) -> u32 {
    let breaks = shadow_seam_breaks(reports, slack_ms);
    let mut flat = 0usize;
    let mut overlapping = 0u32;
    for (_, rr) in reports {
        if !rr.is_empty() && breaks.get(flat).copied().unwrap_or(false) {
            overlapping += 1;
        }
        flat += rr.len();
    }
    overlapping
}

// ─────────────────────────── shadows: frequency-domain HRV ───────────────────────────

/// The Lomb-Scargle band tunables (hrv_freq.rs:10-20).
#[derive(Clone, Copy)]
struct FreqParams {
    vlf_low: f64,
    lf_low: f64,
    lf_high: f64,
    hf_low: f64,
    hf_high: f64,
    min_span_hf: f64,
    min_span_lf: f64,
    min_beats: usize,
    step_hz: f64,
}

const FREQ_BASE: FreqParams = FreqParams {
    vlf_low: 0.0033,
    lf_low: 0.04,
    lf_high: 0.15,
    hf_low: 0.15,
    hf_high: 0.40,
    min_span_hf: 60.0,
    min_span_lf: 250.0,
    min_beats: 20,
    step_hz: 0.005,
};

/// Lomb-Scargle normalised power at one frequency. Mirrors `hrv_freq::lomb_scargle_power`.
fn shadow_ls_power(times: &[f64], y: &[f64], freq_hz: f64, variance: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * freq_hz;
    let (mut sin2, mut cos2) = (0.0, 0.0);
    for &t in times {
        let a = 2.0 * omega * t;
        sin2 += a.sin();
        cos2 += a.cos();
    }
    let tau = sin2.atan2(cos2) / (2.0 * omega);
    let (mut c_term, mut c_den, mut s_term, mut s_den) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..times.len() {
        let arg = omega * (times[i] - tau);
        let (c, s) = (arg.cos(), arg.sin());
        c_term += y[i] * c;
        c_den += c * c;
        s_term += y[i] * s;
        s_den += s * s;
    }
    let cos_part = if c_den > 0.0 { c_term * c_term / c_den } else { 0.0 };
    let sin_part = if s_den > 0.0 { s_term * s_term / s_den } else { 0.0 };
    (cos_part + sin_part) / (2.0 * variance)
}

/// Trapezoidal band integral. Mirrors `hrv_freq::band_power`.
fn shadow_band_power(times: &[f64], y: &[f64], f_low: f64, f_high: f64, step: f64) -> f64 {
    if f_high <= f_low || step <= 0.0 {
        return 0.0;
    }
    let variance = y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64;
    if variance <= 0.0 {
        return 0.0;
    }
    let (mut power, mut prev_p, mut prev_f, mut first, mut f) = (0.0, 0.0, f_low, true, f_low);
    while f <= f_high + 1e-12 {
        let p = shadow_ls_power(times, y, f, variance);
        if !first {
            power += 0.5 * (p + prev_p) * (f - prev_f);
        }
        prev_p = p;
        prev_f = f;
        first = false;
        f += step;
    }
    power
}

/// `(lf, hf, lfhf, total_power)` or `None`. Mirrors `hrv_freq::freq_domain`.
fn shadow_freq(rr_ms: &[u16], p: FreqParams) -> Option<(Option<f64>, f64, Option<f64>, f64)> {
    let clean = HrvReadiness::clean_rr(rr_ms);
    if clean.len() < p.min_beats {
        return None;
    }
    let mut times = vec![0.0f64; clean.len()];
    let mut acc = 0.0;
    for (i, &rr) in clean.iter().enumerate() {
        times[i] = acc / 1000.0;
        acc += rr as f64;
    }
    let span = times[times.len() - 1] - times[0];
    if span < p.min_span_hf {
        return None;
    }
    let m = clean.iter().map(|&v| v as f64).sum::<f64>() / clean.len() as f64;
    let y: Vec<f64> = clean.iter().map(|&v| v as f64 - m).collect();
    let hf = shadow_band_power(&times, &y, p.hf_low, p.hf_high, p.step_hz);
    let lf = (span >= p.min_span_lf).then(|| shadow_band_power(&times, &y, p.lf_low, p.lf_high, p.step_hz));
    let lfhf = match lf {
        Some(l) if hf > 0.0 => Some(l / hf),
        _ => None,
    };
    let total = match lf {
        Some(l) => shadow_band_power(&times, &y, p.vlf_low, p.lf_low, p.step_hz) + l + hf,
        None => hf,
    };
    Some((lf, hf, lfhf, total))
}

/// A tachogram with a strong `resp_hz` modulation over `secs`. Mirrors the `resp_night` fixture builder
/// in hrv_freq.rs's own tests, so the shadow is exercised on the same input the shipped gate uses.
fn resp_night(secs: f64, resp_hz: f64) -> Vec<u16> {
    let mut rr = Vec::new();
    let mut t = 0.0;
    while t < secs {
        let ms = 900.0 + 40.0 * (2.0 * std::f64::consts::PI * resp_hz * t).sin();
        let v = ms.round() as u16;
        rr.push(v);
        t += v as f64 / 1000.0;
    }
    rr
}

/// The four presence/ordering claims of `lf_and_total_present_past_the_lf_gate`, 1.0 when all hold.
fn freq_presence_score(bands: Option<(Option<f64>, f64, Option<f64>, f64)>) -> f64 {
    let Some((lf, hf, lfhf, total)) = bands else { return 0.0 };
    let ok = lf.is_some() && lfhf.is_some() && total >= hf && lf.is_some_and(|l| hf > l);
    if ok {
        1.0
    } else {
        0.0
    }
}

// ─────────────────────────── shadows: respiratory rate ───────────────────────────

/// The RSA pipeline tunables (respiratory_rate.rs:10-22).
#[derive(Clone, Copy)]
struct RespParams {
    rr_min: f64,
    rr_max: f64,
    resample_hz: f64,
    detrend_window_s: f64,
    min_peak_distance_s: f64,
    window_s: f64,
    min_breath_interval_s: f64,
    max_breath_interval_s: f64,
    plausible_min_bpm: f64,
    plausible_max_bpm: f64,
}

const RESP_BASE: RespParams = RespParams {
    rr_min: 300.0,
    rr_max: 2000.0,
    resample_hz: 4.0,
    detrend_window_s: 8.0,
    min_peak_distance_s: 2.5,
    window_s: 300.0,
    min_breath_interval_s: 2.5,
    max_breath_interval_s: 10.0,
    plausible_min_bpm: 8.0,
    plausible_max_bpm: 25.0,
};

/// Sleeping respiratory rate (breaths/min). Mirrors `respiratory_rate::resp_rate_from_rr`.
fn shadow_resp(rr: &[(i64, u16)], start: i64, end: i64, p: RespParams) -> f64 {
    if end <= start {
        return f64::NAN;
    }
    let mut in_bed: Vec<(i64, f64)> = rr
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(ts, ms)| (*ts, *ms as f64))
        .collect();
    in_bed.sort_by_key(|(ts, _)| *ts);
    let filtered: Vec<f64> = in_bed
        .into_iter()
        .map(|(_, ms)| ms)
        .filter(|ms| *ms >= p.rr_min && *ms <= p.rr_max)
        .collect();
    if filtered.len() < 30 {
        return f64::NAN;
    }
    let mut beat_times = vec![0.0; filtered.len()];
    let mut acc = 0.0;
    for (i, &ms) in filtered.iter().enumerate() {
        acc += ms / 1000.0;
        beat_times[i] = acc;
    }
    let total_span_s = beat_times[beat_times.len() - 1];
    if total_span_s < p.window_s / 2.0 {
        return f64::NAN;
    }
    let dt = 1.0 / p.resample_hz;
    let n_grid = (total_span_s / dt) as usize + 1;
    if n_grid < 8 {
        return f64::NAN;
    }
    let mut grid = vec![0.0; n_grid];
    let mut seg = 0usize;
    for (g, cell) in grid.iter_mut().enumerate() {
        let t = g as f64 * dt;
        while seg < beat_times.len() - 2 && beat_times[seg + 1] < t {
            seg += 1;
        }
        let (t0, t1) = (beat_times[seg], beat_times[seg + 1]);
        let (v0, v1) = (filtered[seg], filtered[seg + 1]);
        *cell = if t1 <= t0 {
            v0
        } else {
            let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            v0 + frac * (v1 - v0)
        };
    }
    let half_w = ((p.detrend_window_s * p.resample_hz / 2.0).round() as usize).max(1);
    let baseline = moving_average_centred(&grid, 2 * half_w + 1);
    let detrended: Vec<f64> = (0..n_grid).map(|i| grid[i] - baseline[i]).collect();
    if population_sd(&detrended) <= 1e-9 {
        return f64::NAN;
    }
    let min_dist = ((p.min_peak_distance_s * p.resample_hz).round() as usize).max(2);
    let window_samples = ((p.window_s * p.resample_hz).round() as usize).max(min_dist * 3);
    let mut per_window = Vec::new();
    let mut w = 0usize;
    while w < n_grid {
        let w_end = (w + window_samples).min(n_grid);
        if w_end - w >= min_dist * 3 {
            let peaks = find_peaks(&detrended[w..w_end], min_dist, 0.0);
            if peaks.len() >= 3 {
                let mut intervals = Vec::new();
                for k in 1..peaks.len() {
                    let iv_s = (peaks[k] - peaks[k - 1]) as f64 * dt;
                    if (p.min_breath_interval_s..=p.max_breath_interval_s).contains(&iv_s) {
                        intervals.push(iv_s);
                    }
                }
                if intervals.len() >= 2 {
                    let med = median(&intervals);
                    if med > 0.0 {
                        per_window.push(60.0 / med);
                    }
                }
            }
        }
        w += window_samples;
    }
    if per_window.is_empty() {
        return f64::NAN;
    }
    let m = median(&per_window);
    if (p.plausible_min_bpm..=p.plausible_max_bpm).contains(&m) {
        m
    } else {
        f64::NAN
    }
}

/// Synthetic RSA tachogram. Mirrors the `synth` fixture builder in respiratory_rate.rs's own tests.
fn resp_synth(breath_hz: f64, base_rr_ms: f64, amp_ms: f64, span_s: f64) -> (Vec<(i64, u16)>, i64, i64) {
    let start = 1_700_000_000_i64;
    let mut rows = Vec::new();
    let mut t_sec = 0.0_f64;
    while t_sec < span_s {
        let rr_ms = base_rr_ms + amp_ms * (2.0 * std::f64::consts::PI * breath_hz * t_sec).sin();
        // A non-advancing beat interval never ends the walk; the -100% ladder rung reaches it.
        if rr_ms <= 0.0 {
            break;
        }
        t_sec += rr_ms / 1000.0;
        rows.push((start + t_sec as i64, rr_ms as u16));
    }
    (rows, start, start + t_sec as i64)
}

// ─────────────────────────── shadows: resting HR ───────────────────────────

/// Session resting-HR floor with a variable window. Mirrors `resting_hr::session_resting_hr_floor`.
fn shadow_session_rhr(start: i64, end: i64, hr: &[HrSample], window_seconds: i64) -> f64 {
    let seg: Vec<&HrSample> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).collect();
    if seg.is_empty() || window_seconds <= 0 {
        return f64::NAN;
    }
    let mut means: Vec<f64> = Vec::new();
    let mut t = start;
    while t < end {
        let win: Vec<&&HrSample> =
            seg.iter().filter(|s| s.ts >= t && s.ts < t + window_seconds).collect();
        if !win.is_empty() {
            let sum: i64 = win.iter().map(|s| s.bpm as i64).sum();
            means.push(sum as f64 / win.len() as f64);
        }
        t += window_seconds;
    }
    let v = match means.into_iter().reduce(f64::min) {
        Some(m) => m,
        None => {
            let all: i64 = seg.iter().map(|s| s.bpm as i64).sum();
            all as f64 / seg.len() as f64
        }
    };
    (v + 0.5).floor()
}

/// A night at `high` bpm with one sustained 5-min window pinned at `floor`. Mirrors the
/// `night_with_floor` fixture in tests/resting_hr_parity.rs.
fn night_with_floor(start: i64, floor: i32, high: i32) -> Vec<HrSample> {
    let mut v = Vec::new();
    for w in 0..6i64 {
        let bpm = if w == 3 { floor } else { high };
        let base = start + w * 5 * 60;
        for s in 0..5i64 {
            v.push(HrSample::new(base + s * 60, bpm));
        }
    }
    v
}

// ─────────────────────────── shadows: SpO2 ───────────────────────────

/// The ratio-of-ratios tunables (spo2.rs:10-28).
#[derive(Clone, Copy)]
struct Spo2Params {
    window_seconds: usize,
    min_samples_per_window: usize,
    min_pulsatile_fraction: f64,
    curve_a: f64,
    curve_b: f64,
    clamp_low: f64,
    clamp_high: f64,
    roll_window_nights: usize,
    recent_nights: usize,
    anchor: f64,
    rolling_clamp_low: f64,
    rolling_clamp_high: f64,
    min_nights: usize,
}

const SPO2_BASE: Spo2Params = Spo2Params {
    window_seconds: 30,
    min_samples_per_window: 10,
    min_pulsatile_fraction: 0.5,
    curve_a: 110.0,
    curve_b: 25.0,
    clamp_low: 70.0,
    clamp_high: 100.0,
    roll_window_nights: 30,
    recent_nights: 7,
    anchor: 96.5,
    rolling_clamp_low: 88.0,
    rolling_clamp_high: 100.0,
    min_nights: 1,
};

/// One window's ratio-of-ratios SpO2. Mirrors `spo2::window_spo2`.
fn shadow_window_spo2(red: &[f64], ir: &[f64], p: Spo2Params) -> Option<f64> {
    let (dc_red, dc_ir) = (mean(red), mean(ir));
    if dc_red <= 0.0 || dc_ir <= 0.0 {
        return None;
    }
    let (ac_red, ac_ir) = (amplitude(red), amplitude(ir));
    if ac_red <= 0.0 || ac_ir <= 0.0 {
        return None;
    }
    let ratio_ir = ac_ir / dc_ir;
    if ratio_ir <= 0.0 {
        return None;
    }
    let r = (ac_red / dc_red) / ratio_ir;
    Some((p.curve_a - p.curve_b * r).clamp(p.clamp_low, p.clamp_high))
}

/// Night SpO2 from a paired red/IR series. Mirrors `Spo2::from_paired`.
fn shadow_spo2_paired(red: &[f64], ir: &[f64], p: Spo2Params) -> f64 {
    let n = red.len().min(ir.len());
    let mut per_window = Vec::new();
    let mut eligible = 0usize;
    let mut start = 0;
    while start < n {
        let end = (start + p.window_seconds).min(n);
        if end - start >= p.min_samples_per_window {
            eligible += 1;
            if let Some(s) = shadow_window_spo2(&red[start..end], &ir[start..end], p) {
                per_window.push(s);
            }
        }
        if end == start {
            break;
        }
        start = end;
    }
    if eligible == 0 || (per_window.len() as f64) < p.min_pulsatile_fraction * eligible as f64 {
        return f64::NAN;
    }
    if per_window.is_empty() {
        f64::NAN
    } else {
        median(&per_window)
    }
}

/// Smoothed multi-night readout. Mirrors `Spo2::rolling_reading`.
fn shadow_spo2_rolling(recent_nightly: &[f64], p: Spo2Params) -> f64 {
    let window = if recent_nightly.len() > p.roll_window_nights {
        &recent_nightly[recent_nightly.len() - p.roll_window_nights..]
    } else {
        recent_nightly
    };
    if window.len() < p.min_nights {
        return f64::NAN;
    }
    let offset = p.anchor - median(window);
    let recent_count = p.recent_nights.min(window.len()).max(1);
    let recent = median(&window[window.len() - recent_count..]);
    let clamped = (recent + offset).clamp(p.rolling_clamp_low, p.rolling_clamp_high);
    (clamped + 0.5).floor()
}

/// The 30-night spread from the spo2.rs gate: the recent week sits below the month, so
/// `median(30) != median(7)` and the anchored differential is actually exercised.
const SPO2_SPREAD_NIGHTS: [f64; 30] = [
    93.4, 92.8, 94.1, 93.0, 92.5, 94.6, 93.3, 92.9, 93.8, 94.2, 92.6, 93.5, 93.1, 94.0, 92.7, 93.9,
    93.2, 92.4, 94.3, 93.6, 92.3, 93.7, 94.4, 91.2, 90.5, 91.8, 90.9, 91.5, 90.2, 91.0,
];

/// A 20-sample window with DC = `dc` and p95-p5 amplitude `ac`. Mirrors the `win` fixture in spo2.rs.
fn spo2_win(dc: f64, ac: f64) -> Vec<f64> {
    std::iter::repeat_n(dc - ac / 2.0, 10).chain(std::iter::repeat_n(dc + ac / 2.0, 10)).collect()
}

// ─────────────────────────── shadows: vitality / body age ───────────────────────────

/// One Vitality tunable in the sweep: its label, its shipped value, and how to set it.
type VitSetter = (&'static str, f64, fn(&mut VitParams, f64));

/// The Vitality tunables (vitality.rs:13-116).
#[derive(Clone, Copy)]
struct VitParams {
    doubling_years: f64,
    overlap_shrink: f64,
    min_body_age: f64,
    max_body_age: f64,
    vitality_per_year: f64,
    min_factors: usize,
    hr_resting_per_10bpm: f64,
    resting_hr_reference: f64,
    hr_vo2max_per_met: f64,
    met_ml_kg_min: f64,
    vo2max_met_clamp: f64,
    hr_sleep_per_hour: f64,
    sleep_optimum_hours: f64,
    sleep_deadband_hours: f64,
    sleep_deviation_clamp: f64,
    hr_sri_per_point: f64,
    sri_median_reference: f64,
    sri_ln_hazard_clamp: f64,
    hr_hrv_per_fraction: f64,
    hr_steps_per_1000: f64,
    steps_reference: f64,
    steps_clamp_hi: f64,
    steps_clamp: f64,
}

const VIT_BASE: VitParams = VitParams {
    doubling_years: 10.0,
    overlap_shrink: 0.75,
    min_body_age: 20.0,
    max_body_age: 90.0,
    vitality_per_year: 2.5,
    min_factors: 3,
    hr_resting_per_10bpm: 0.100,
    resting_hr_reference: 65.0,
    hr_vo2max_per_met: 0.130,
    met_ml_kg_min: 3.5,
    vo2max_met_clamp: 4.0,
    hr_sleep_per_hour: 0.110,
    sleep_optimum_hours: 7.5,
    sleep_deadband_hours: 0.5,
    sleep_deviation_clamp: 3.0,
    hr_sri_per_point: 0.015609,
    sri_median_reference: 68.25,
    sri_ln_hazard_clamp: 0.50,
    hr_hrv_per_fraction: 0.160,
    hr_steps_per_1000: 0.064,
    steps_reference: 7000.0,
    steps_clamp_hi: 11_000.0,
    steps_clamp: 4.0,
};

/// The driver values the reference reading carries; `None` drops the driver.
#[derive(Clone, Copy)]
struct VitDrivers {
    chrono_age: f64,
    resting_hr: Option<f64>,
    vo2max: Option<f64>,
    expected_vo2max: Option<f64>,
    sleep_hours: Option<f64>,
    /// `None` means "track the shipped constant", exactly as the shipped reference fixture does.
    sleep_regularity_index: Option<f64>,
    rmssd: Option<f64>,
    rmssd_norm: Option<f64>,
    steps: Option<f64>,
}

/// Per-driver signed log-hazards. Mirrors `vitality::contributions`.
fn shadow_contributions(d: VitDrivers, p: VitParams) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(rhr) = d.resting_hr {
        out.push(((rhr - p.resting_hr_reference) / 10.0) * p.hr_resting_per_10bpm);
    }
    if let (Some(vo2), Some(expected)) = (d.vo2max, d.expected_vo2max) {
        if expected > 0.0 {
            let mets =
                ((expected - vo2) / p.met_ml_kg_min).clamp(-p.vo2max_met_clamp, p.vo2max_met_clamp);
            out.push(mets * p.hr_vo2max_per_met);
        }
    }
    if let Some(hours) = d.sleep_hours {
        let deviation = ((hours - p.sleep_optimum_hours).abs() - p.sleep_deadband_hours).max(0.0);
        out.push(deviation.clamp(0.0, p.sleep_deviation_clamp) * p.hr_sleep_per_hour);
    }
    let sri = d.sleep_regularity_index.unwrap_or(p.sri_median_reference);
    out.push(
        ((p.sri_median_reference - sri.clamp(-100.0, 100.0)) * p.hr_sri_per_point)
            .clamp(-p.sri_ln_hazard_clamp, p.sri_ln_hazard_clamp),
    );
    if let (Some(rmssd), Some(norm)) = (d.rmssd, d.rmssd_norm) {
        if norm > 0.0 {
            let shortfall = ((norm - rmssd) / norm).clamp(-1.0, 1.0);
            out.push(shortfall * p.hr_hrv_per_fraction);
        }
    }
    if let Some(steps) = d.steps {
        let deficit = (p.steps_reference - steps.clamp(0.0, p.steps_clamp_hi)) / 1000.0;
        out.push(deficit.clamp(-p.steps_clamp, p.steps_clamp) * p.hr_steps_per_1000);
    }
    out
}

/// `(body_age, vitality)` or NaN. Mirrors `vitality::compute`.
fn shadow_vitality(d: VitDrivers, p: VitParams) -> (f64, f64) {
    if d.chrono_age <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let c = shadow_contributions(d, p);
    if c.len() < p.min_factors {
        return (f64::NAN, f64::NAN);
    }
    let sum_ln = c.iter().sum::<f64>() * p.overlap_shrink;
    let ln_hazard_per_year = std::f64::consts::LN_2 / p.doubling_years;
    let delta_age = sum_ln / ln_hazard_per_year;
    let body_age = (d.chrono_age + delta_age).clamp(p.min_body_age, p.max_body_age);
    let years_younger = d.chrono_age - body_age;
    let vitality = (50.0 + years_younger * p.vitality_per_year).clamp(0.0, 100.0);
    (body_age, vitality)
}

/// The driver set every reference gate uses. Mirrors the `reference()` fixture in vitality.rs.
fn vit_reference() -> VitDrivers {
    VitDrivers {
        chrono_age: 40.0,
        resting_hr: Some(65.0),
        vo2max: Some(45.0),
        expected_vo2max: Some(45.0),
        sleep_hours: Some(7.5),
        sleep_regularity_index: None,
        rmssd: Some(45.0),
        rmssd_norm: Some(45.0),
        steps: Some(7000.0),
    }
}

// ─────────────────────────── shadows: CosinorAge ───────────────────────────

/// One CosinorAge tunable in the sweep: its label, its shipped value, and how to set it.
type AgeSetter = (&'static str, f64, fn(&mut AgeParams, f64));

/// The CosinorAge transform tunables (biological_age.rs:16-48).
#[derive(Clone, Copy)]
struct AgeParams {
    rate: f64,
    mesor: f64,
    amp1: f64,
    phi1: f64,
    age: f64,
    m_n: f64,
    m_d: f64,
    ba_n: f64,
    ba_d: f64,
    ba_i: f64,
}

const AGE_GENERIC: AgeParams = AgeParams {
    rate: -13.36715309,
    mesor: -0.03204933,
    amp1: -0.01971357,
    phi1: -0.01664718,
    age: 0.10033692,
    m_n: -1.405276,
    m_d: 0.01462774,
    ba_n: -0.01447851,
    ba_d: 0.112165,
    ba_i: 133.5989,
};

/// CosinorAge in years, NaN on a degenerate transform. Mirrors `biological_age::cosinor_age`.
fn shadow_cosinor_age(mesor: f64, amp: f64, phi: f64, chrono: f64, p: AgeParams) -> f64 {
    let xb = mesor * p.mesor + amp * p.amp1 + phi * p.phi1 + chrono * p.age + p.rate;
    let survival = (p.m_n * xb.exp() / p.m_d).exp();
    let outer = p.ba_n * survival.ln();
    if outer <= 0.0 || !outer.is_finite() {
        return f64::NAN;
    }
    let years = outer.ln() / p.ba_d + p.ba_i;
    if years.is_finite() {
        years
    } else {
        f64::NAN
    }
}

// ─────────────────────────── shadows: circadian cosinor ───────────────────────────

/// Single-component cosinor fit with a variable fundamental. Mirrors `circadian::cosinor`.
fn shadow_cosinor(bins: &[ActivityBin], w_hours: f64) -> Option<(f64, f64, f64)> {
    if bins.len() < 3 {
        return None;
    }
    let n = bins.len() as f64;
    let (mut sum_y, mut sum_c, mut sum_s) = (0.0, 0.0, 0.0);
    let (mut sum_cc, mut sum_ss, mut sum_cs) = (0.0, 0.0, 0.0);
    let (mut sum_yc, mut sum_ys) = (0.0, 0.0);
    for b in bins {
        let c = (w_hours * b.hour).cos();
        let s = (w_hours * b.hour).sin();
        let y = b.activity;
        sum_y += y;
        sum_c += c;
        sum_s += s;
        sum_cc += c * c;
        sum_ss += s * s;
        sum_cs += c * s;
        sum_yc += y * c;
        sum_ys += y * s;
    }
    let (a11, a12, a13) = (n, sum_c, sum_s);
    let (a21, a22, a23) = (sum_c, sum_cc, sum_cs);
    let (a31, a32, a33) = (sum_s, sum_cs, sum_ss);
    let det =
        a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);
    if det.abs() <= 1e-12 {
        return None;
    }
    let det_m = sum_y * (a22 * a33 - a23 * a32) - a12 * (sum_yc * a33 - a23 * sum_ys)
        + a13 * (sum_yc * a32 - a22 * sum_ys);
    let det_b = a11 * (sum_yc * a33 - a23 * sum_ys) - sum_y * (a21 * a33 - a23 * a31)
        + a13 * (a21 * sum_ys - sum_yc * a31);
    let det_g = a11 * (a22 * sum_ys - sum_yc * a32) - a12 * (a21 * sum_ys - sum_yc * a31)
        + sum_y * (a21 * a32 - a22 * a31);
    let mesor = det_m / det;
    let beta = det_b / det;
    let gamma = det_g / det;
    let amplitude = (beta * beta + gamma * gamma).sqrt();
    let mut acro = gamma.atan2(beta) / w_hours % 24.0;
    if acro < 0.0 {
        acro += 24.0;
    }
    Some((mesor, amplitude, acro))
}

const W_BASE: f64 = 2.0 * std::f64::consts::PI / 24.0;

/// 24 hourly bins from a known mesor/amplitude/acrophase. Mirrors the `synth` fixture in circadian.rs.
fn circ_synth(mesor: f64, amp: f64, acro_hours: f64, w_hours: f64) -> Vec<ActivityBin> {
    (0..24)
        .map(|h| {
            let hour = h as f64;
            ActivityBin { hour, activity: mesor + amp * (w_hours * (hour - acro_hours)).cos() }
        })
        .collect()
}

/// Acrophase only when the phase estimate is Solid, so one number carries both claims at
/// circadian.rs:303-304. Mirrors `circadian::estimate_phase`'s confidence ladder.
fn shadow_phase_solid_acro(
    bins: &[ActivityBin],
    days_observed: u32,
    w_hours: f64,
    min_days: u32,
    good_days: u32,
    min_rel_amp: f64,
) -> f64 {
    let Some((mesor, amp, acro)) = shadow_cosinor(bins, w_hours) else { return f64::NAN };
    let rel = if mesor != 0.0 { amp / mesor.abs() } else { 0.0 };
    if days_observed < min_days || rel < min_rel_amp {
        return f64::NAN;
    }
    if days_observed >= good_days {
        acro
    } else {
        f64::NAN
    }
}

// ─────────────────────────── shadows: HR anomaly watch ───────────────────────────

/// The HR-watch tunables (hr_anomaly.rs:11-22), all private to that module.
#[derive(Clone, Copy)]
struct WatchParams {
    min_baseline: usize,
    sustain_s: u32,
    max_gap_s: u32,
    elev_margin: u8,
    high_abs: u8,
    qual_min: u8,
    resting_pct: f64,
}

const WATCH_BASE: WatchParams = WatchParams {
    min_baseline: 600,
    sustain_s: 300,
    max_gap_s: 5,
    elev_margin: 45,
    high_abs: 100,
    qual_min: 192,
    resting_pct: 0.10,
};

/// Peak bpm of the first sustained elevated-at-rest run, NaN when Normal or Calibrating. Mirrors
/// `HrWatch::evaluate`.
fn shadow_watch_peak(history: &[HistoryRecord], p: WatchParams) -> f64 {
    let mut rest: Vec<(u32, u8)> = history
        .iter()
        .filter_map(|h| {
            let at_rest = h.activity_class == Some(0) || h.sleep_state == Some(2);
            let good =
                h.signal_quality.is_some_and(|q| q >= p.qual_min) && h.optical_signal_poor != Some(true);
            let on_wrist = !worn_state(h).is_off();
            let hr = h.heart_rate?;
            (at_rest && good && on_wrist && hr > 0).then_some((h.unix, hr))
        })
        .collect();
    if rest.len() < p.min_baseline {
        return f64::NAN;
    }
    rest.sort_by_key(|&(t, _)| t);
    let mut hrs: Vec<f64> = rest.iter().map(|&(_, hr)| hr as f64).collect();
    hrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let resting = percentile(&hrs, p.resting_pct).round() as u8;
    let threshold = resting.saturating_add(p.elev_margin).max(p.high_abs);
    let mut run: Option<(u32, u32, u8)> = None;
    let mut prev = 0u32;
    for &(t, hr) in &rest {
        if hr >= threshold {
            run = match run {
                Some((s, _, pk)) if t.saturating_sub(prev) <= p.max_gap_s => Some((s, t, pk.max(hr))),
                _ => Some((t, t, hr)),
            };
            if let Some((s, l, pk)) = run {
                if l.saturating_sub(s) >= p.sustain_s {
                    return pk as f64;
                }
            }
        } else {
            run = None;
        }
        prev = t;
    }
    f64::NAN
}

/// An at-rest, good-signal v18 record. Mirrors the `rec` fixture in hr_anomaly.rs.
fn watch_rec(unix: u32, hr: u8) -> HistoryRecord {
    HistoryRecord {
        version: 18,
        unix,
        heart_rate: Some(hr),
        activity_class: Some(0),
        signal_quality: Some(255),
        signal_flags: Some(0),
        ..Default::default()
    }
}

// ─────────────────────────── shadows: wear state ───────────────────────────

/// Wear verdict under a variable off-wrist bit mask. Mirrors `worn::worn_state`.
fn shadow_worn(h: &HistoryRecord, off_wrist_bit: u8) -> WornState {
    let flag = h.signal_flags.map(|f| f & off_wrist_bit == 0);
    let optical =
        h.optical_signal_poor.map(|_| h.optical_baseline_a.is_some() || h.optical_baseline_b.is_some());
    match (flag, optical) {
        (Some(false), _) | (_, Some(false)) => WornState::NotWorn,
        (None, None) => WornState::Unknown,
        _ => WornState::Worn,
    }
}

/// The real captured off-wrist v18 record. Mirrors `real_off_wrist()` in worn.rs.
fn worn_off_wrist() -> HistoryRecord {
    HistoryRecord {
        version: 18,
        unix: 1_784_000_000,
        signal_flags: Some(0),
        signal_quality: Some(0),
        optical_baseline_a: None,
        optical_baseline_b: None,
        optical_signal_poor: Some(true),
        ..Default::default()
    }
}

/// The real captured worn v18 record. Mirrors `real_worn()` in worn.rs.
fn worn_on_wrist() -> HistoryRecord {
    HistoryRecord {
        version: 18,
        unix: 1_784_000_000,
        heart_rate: Some(60),
        signal_flags: Some(0),
        signal_quality: Some(255),
        optical_baseline_a: Some(101),
        optical_baseline_b: Some(111),
        optical_signal_poor: Some(false),
        ..Default::default()
    }
}

// ─────────────────────────── shadows: stats trendline + calibration ───────────────────────────

/// The trend tunables (stats.rs:182-187) plus the significance squash exponent.
#[derive(Clone, Copy)]
struct TrendParams {
    ci_z: f64,
    min_points: usize,
    min_span_floor_days: f64,
    min_span_window_fraction: f64,
    sig_half: f64,
}

const TREND_BASE: TrendParams = TrendParams {
    ci_z: 1.282,
    min_points: 3,
    min_span_floor_days: 3.0,
    min_span_window_fraction: 1.0 / 3.0,
    sig_half: 0.5,
};

/// `(slope, slope_se, significance, flat)`. Mirrors `stats::weighted_trendline`.
fn shadow_trendline(
    days: &[f64],
    values: &[f64],
    weights: &[f64],
    min_span_days: f64,
    p: TrendParams,
) -> Option<(f64, f64, f64, bool)> {
    let pairs = days.len().min(values.len());
    let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..pairs {
        let wi = weights.get(i).copied().unwrap_or(1.0);
        if days[i].is_finite() && values[i].is_finite() && wi.is_finite() && wi >= 0.0 {
            x.push(days[i]);
            y.push(values[i]);
            w.push(wi);
        }
    }
    let n = x.len();
    if n < p.min_points || n < 3 {
        return None;
    }
    let start_day = x.iter().copied().fold(f64::INFINITY, f64::min);
    let end_day = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if end_day - start_day < min_span_days {
        return None;
    }
    let w_sum: f64 = w.iter().sum();
    if w_sum <= 0.0 {
        return None;
    }
    let mean_x = x.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / w_sum;
    let mean_y = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / w_sum;
    let (mut ss_xx, mut ss_xy) = (0.0, 0.0);
    for i in 0..n {
        let dx = x[i] - mean_x;
        ss_xx += w[i] * dx * dx;
        ss_xy += w[i] * dx * (y[i] - mean_y);
    }
    if ss_xx <= 0.0 {
        return None;
    }
    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let ss_res: f64 = (0..n)
        .map(|i| {
            let r = y[i] - (slope * x[i] + intercept);
            w[i] * r * r
        })
        .sum();
    let slope_se = (ss_res / (n - 2) as f64 / ss_xx).sqrt();
    let z = if slope_se > 0.0 {
        slope.abs() / slope_se
    } else if slope == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let half = p.ci_z * slope_se;
    let flat = slope - half <= 0.0 && slope + half >= 0.0;
    Some((slope, slope_se, 1.0 - (-p.sig_half * z * z).exp(), flat))
}

/// Shortest accepted span for a window. Mirrors `stats::trend_min_span_days`.
fn shadow_min_span(window_days: f64, p: TrendParams) -> f64 {
    (window_days * p.min_span_window_fraction).max(p.min_span_floor_days)
}

/// The four calibration claims at calibration.rs:70-73, scored 1.0 when all hold.
fn calib_score(o2: Calibration, skin: Calibration, cal: Calibration) -> f64 {
    let ok = !o2.unlocked(0)
        && o2.unlocked(1)
        && !skin.unlocked(6)
        && skin.unlocked(7)
        && cal.unlocked(1)
        && !cal.calibrated(1)
        && cal.calibrated(14);
    if ok {
        1.0
    } else {
        0.0
    }
}

/// Scale a `Calibration` pair by a multiplier, never below zero.
fn scale_calib(c: Calibration, f: f64) -> Calibration {
    Calibration {
        unlock: (c.unlock as f64 * f).round().max(0.0) as u32,
        full: (c.full as f64 * f).round().max(0.0) as u32,
    }
}

/// Alternating u16 series. Mirrors the `alternating` fixture in hrv.rs.
fn alternating(n: usize, lo: u16, hi: u16) -> Vec<u16> {
    (0..n).map(|i| if i % 2 == 0 { lo } else { hi }).collect()
}

// ─────────────────────────── M1: HRV RMSSD, gap-aware ───────────────────────────

fn m1_rmssd_gap_aware(t: &mut Tally) {
    // hrv.rs:768 - the Malik-ectopic leg of `rmssd_gap_aware_splices_at_each_dropped_beat`.
    let gate = Gate {
        label: "HrvReadiness::rmssd_gap_aware(malik)",
        source: "crates/physio-algo/src/hrv.rs:768",
        target: 5.049752469181039,
        tol: 1e-12,
    };
    let malik: Vec<u16> = vec![800, 805, 810, 1300, 806, 802, 808];
    let mut tb = Table::new("HRV RMSSD (gap-aware, artifact-corrected)", gate);

    // The shipped function and the shadow must agree before any arm means anything.
    let shipped = HrvReadiness::rmssd_gap_aware(&[(0u32, malik.clone())]).unwrap();
    let base = shadow_rmssd_gap_aware(&malik, RR_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    // The other two hand-computed literals in that test, hrv.rs:771 and :774.
    assert!((shadow_rmssd_gap_aware(&[600, 610, 605, 5, 620, 615], RR_BASE) - 7.0710678118654755).abs() < 1e-12);
    assert!((shadow_rmssd_gap_aware(&[800, 810, 820, 815, 805], RR_BASE) - 9.013878188659973).abs() < 1e-12);
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: mean R-R instead of RMSSD", mean(&malik.iter().map(|&v| v as f64).collect::<Vec<_>>()));
    tb.add(Kind::Null, "input: deterministic shuffle", shadow_rmssd_gap_aware(&shuffled(&malik), RR_BASE));

    let rev: Vec<u16> = malik.iter().rev().copied().collect();
    tb.add(Kind::Structural, "input: reversed", shadow_rmssd_gap_aware(&rev, RR_BASE));
    let offset: Vec<u16> = malik.iter().map(|&v| v + 50).collect();
    tb.add(Kind::Structural, "input: +50 ms constant offset", shadow_rmssd_gap_aware(&offset, RR_BASE));
    tb.add(Kind::Structural, "input: drop last beat (-14%)", shadow_rmssd_gap_aware(&malik[..malik.len() - 1], RR_BASE));
    let mut rot = malik.clone();
    rot.rotate_left(1);
    tb.add(Kind::Structural, "input: shifted one beat (rotate left)", shadow_rmssd_gap_aware(&rot, RR_BASE));

    let mut p;
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        p = RR_BASE;
        p.rr_min = (300.0 * f).round() as u16;
        tb.add(Kind::Param, format!("param: RR_MIN_MS 300 -> {} ({label})", p.rr_min), shadow_rmssd_gap_aware(&malik, p));
        p = RR_BASE;
        p.rr_max = (2000.0 * f).round() as u16;
        tb.add(Kind::Param, format!("param: RR_MAX_MS 2000 -> {} ({label})", p.rr_max), shadow_rmssd_gap_aware(&malik, p));
        p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(Kind::Param, format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh), shadow_rmssd_gap_aware(&malik, p));
    }
    p = RR_BASE;
    p.ect_thresh = 0.20 * 1.005;
    tb.add(Kind::Param, "param: ECTOPIC_THRESHOLD 0.20 -> 0.201 (+0.5%)", shadow_rmssd_gap_aware(&malik, p));
    for r in [1usize, 3] {
        p = RR_BASE;
        p.ect_radius = r;
        tb.add(Kind::Param, format!("param: ECTOPIC_WINDOW_RADIUS 2 -> {r}"), shadow_rmssd_gap_aware(&malik, p));
    }

    tb.note(format!(
        "ECTOPIC_THRESHOLD {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.ect_thresh = 0.20 * f;
            shadow_rmssd_gap_aware(&malik, q)
        })
    ));
    tb.note(format!(
        "RR_MIN_MS {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.rr_min = (300.0 * f).round() as u16;
            shadow_rmssd_gap_aware(&malik, q)
        })
    ));
    tb.note("the range filter never fires on this gate's input, so RR_MIN_MS / RR_MAX_MS are unreachable from it");
    report(&tb, t);
}

fn m1b_rmssd_artifact_filter(t: &mut Tally) {
    // hrv.rs:582 - `rmssd_of_known_intervals`, the only gate that reaches MAX_BEAT_DELTA_MS.
    let gate = Gate {
        label: "HrvReadiness::rmssd(&[800, 810, 820])",
        source: "crates/physio-algo/src/hrv.rs:582",
        target: 10.0,
        tol: 0.0,
    };
    let run: Vec<u16> = vec![800, 810, 820];
    let mut tb = Table::new("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", gate);
    let shipped = HrvReadiness::rmssd(&run).unwrap();
    let base = shadow_rmssd_run(&run, RR_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: mean R-R", mean(&[800.0, 810.0, 820.0]));
    tb.add(Kind::Structural, "input: reversed", shadow_rmssd_run(&[820, 810, 800], RR_BASE));
    tb.add(Kind::Structural, "input: +100 ms constant offset", shadow_rmssd_run(&[900, 910, 920], RR_BASE));
    tb.add(Kind::Structural, "input: drop last beat", shadow_rmssd_run(&run[..2], RR_BASE));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let mut p = RR_BASE;
        p.max_beat_delta = 200.0 * f;
        tb.add(
            Kind::Param,
            format!("param: MAX_BEAT_DELTA_MS 200 -> {:.1} ({label})", p.max_beat_delta),
            shadow_rmssd_run(&run, p),
        );
    }
    tb.note(format!(
        "MAX_BEAT_DELTA_MS {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.max_beat_delta = 200.0 * f;
            shadow_rmssd_run(&run, q)
        })
    ));
    report(&tb, t);
}

// ─────────────────────────── M2: HRV windowed average ───────────────────────────

fn m2_windowed_avg(t: &mut Tally) {
    // hrv.rs:806 - `windowed_avg_hrv_single_bucket_equals_bucket_rmssd`.
    let gate = Gate {
        label: "HrvReadiness::windowed_avg_hrv(100, 400, beats)",
        source: "crates/physio-algo/src/hrv.rs:806",
        target: 9.013878188659973,
        tol: 1e-12,
    };
    let beats: Vec<(u32, u16)> = vec![(100, 800), (100, 810), (100, 820), (100, 815), (100, 805)];
    let mut tb = Table::new("HRV windowed average (session avgHrv)", gate);
    let shipped = HrvReadiness::windowed_avg_hrv(100, 400, &beats).unwrap();
    let base = shadow_windowed_avg(100, 400, &beats, 300, 2000, RR_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: mean R-R in the bucket", mean(&beats.iter().map(|&(_, v)| v as f64).collect::<Vec<_>>()));
    let shuf: Vec<(u32, u16)> = shuffled(&beats);
    tb.add(Kind::Null, "input: deterministic shuffle", shadow_windowed_avg(100, 400, &shuf, 300, 2000, RR_BASE));

    let rev: Vec<(u32, u16)> = beats.iter().rev().copied().collect();
    tb.add(Kind::Structural, "input: reversed", shadow_windowed_avg(100, 400, &rev, 300, 2000, RR_BASE));
    let moved: Vec<(u32, u16)> = beats.iter().map(|&(ts, v)| (ts + 300, v)).collect();
    tb.add(Kind::Structural, "input: beats shifted +300 s (out of window)", shadow_windowed_avg(100, 400, &moved, 300, 2000, RR_BASE));
    let off: Vec<(u32, u16)> = beats.iter().map(|&(ts, v)| (ts, v + 50)).collect();
    tb.add(Kind::Structural, "input: +50 ms constant offset", shadow_windowed_avg(100, 400, &off, 300, 2000, RR_BASE));
    tb.add(Kind::Structural, "input: drop last beat (-20%)", shadow_windowed_avg(100, 400, &beats[..4], 300, 2000, RR_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let w = (300.0 * f).round() as u64;
        tb.add(Kind::Param, format!("param: HRV_WINDOW_SECS 300 -> {w} ({label})"), shadow_windowed_avg(100, 400, &beats, w, 2000, RR_BASE));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(Kind::Param, format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh), shadow_windowed_avg(100, 400, &beats, 300, 2000, p));
    }
    tb.note(format!(
        "HRV_WINDOW_SECS {}",
        break_at(&tb.gate, &|f| shadow_windowed_avg(100, 400, &beats, (300.0 * f).round() as u64, 2000, RR_BASE))
    ));
    tb.note("every beat in this gate's input shares one timestamp, so the window WIDTH cannot change the answer");
    report(&tb, t);
}

fn m2b_windowed_avg_deep(t: &mut Tally) {
    // hrv.rs:818-819 - `deep_windowed_avg_hrv_keeps_buckets_in_deep_spans`; the target is bucket B's RMSSD.
    let gate = Gate {
        label: "HrvReadiness::windowed_avg_hrv_deep(100, 700, beats, [(400,700)])",
        source: "crates/physio-algo/src/hrv.rs:818-819",
        target: 15.811388300841896,
        tol: 1e-12,
    };
    let beats: Vec<(u32, u16)> = vec![
        (100, 800), (100, 810), (100, 820), (100, 815), (100, 805),
        (400, 700), (400, 720), (400, 710),
    ];
    let deep = vec![(400u32, 700u32)];
    let mut tb = Table::new("HRV deep-window average", gate);
    let base = HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &deep).unwrap();
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: all-bucket mean (deep filter ignored)", nan_if_none(HrvReadiness::windowed_avg_hrv(100, 700, &beats)));
    tb.add(Kind::Structural, "span: deep window swapped to the other bucket", nan_if_none(HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &[(100, 400)])));
    tb.add(Kind::Structural, "span: no deep spans at all", nan_if_none(HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &[])));
    let off: Vec<(u32, u16)> = beats.iter().map(|&(ts, v)| (ts, v + 50)).collect();
    tb.add(Kind::Structural, "input: +50 ms constant offset", nan_if_none(HrvReadiness::windowed_avg_hrv_deep(100, 700, &off, &deep)));
    tb.add(Kind::Param, "param: deep span start 400 -> 440 (+10%)", nan_if_none(HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &[(440, 700)])));
    tb.add(Kind::Param, "param: deep span start 400 -> 360 (-10%)", nan_if_none(HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &[(360, 700)])));
    tb.note("the deep filter tests each bucket's CENTRE, so a span edge can move a full half-window without effect");
    report(&tb, t);
}

// ─────────────────────────── M3: SDNN / pNN50 / analyze_raw ───────────────────────────

fn m3_analyze_raw(t: &mut Tally) {
    // hrv.rs:717 - the SDNN leg of `analyze_raw_full_clean_series`.
    let gate = Gate {
        label: "HrvReadiness::analyze_raw(alternating(24, 800, 810)).sdnn",
        source: "crates/physio-algo/src/hrv.rs:717",
        target: (600.0f64 / 23.0).sqrt(),
        tol: 1e-9,
    };
    let series = alternating(24, 800, 810);
    let mut tb = Table::new("HRV SDNN / pNN50 / analyze_raw (spot reading)", gate);
    let shipped = HrvReadiness::analyze_raw(&series, None).sdnn.unwrap();
    let base = shadow_analyze_sdnn(&series, 20, None, RR_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: mean NN instead of SDNN", 805.0);
    tb.add(Kind::Null, "input: deterministic shuffle", shadow_analyze_sdnn(&shuffled(&series), 20, None, RR_BASE));

    let rev: Vec<u16> = series.iter().rev().copied().collect();
    tb.add(Kind::Structural, "input: reversed", shadow_analyze_sdnn(&rev, 20, None, RR_BASE));
    let off: Vec<u16> = series.iter().map(|&v| v + 100).collect();
    tb.add(Kind::Structural, "input: +100 ms constant offset", shadow_analyze_sdnn(&off, 20, None, RR_BASE));
    tb.add(Kind::Structural, "input: drop last 10% (2 beats)", shadow_analyze_sdnn(&series[..22], 20, None, RR_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        tb.add(Kind::Param, format!("param: MIN_BEATS 20 -> {} ({label})", scale_u(20, f)), shadow_analyze_sdnn(&series, scale_u(20, f), None, RR_BASE));
    }
    // SPOT_MAX_REJECTED_FRACTION is reached through the 0.375-rejected series at hrv.rs:731-742.
    let mut spot = alternating(25, 800, 810);
    spot.extend(std::iter::repeat_n(100u16, 15));
    let spot_gate_holds = |frac: f64| shadow_analyze_sdnn(&spot, 20, Some(frac), RR_BASE).is_nan();
    tb.note(format!(
        "SPOT_MAX_REJECTED_FRACTION 0.35: refuses at 0.35 = {}, at +10% (0.385) = {}, at -10% (0.315) = {}",
        spot_gate_holds(0.35),
        spot_gate_holds(0.385),
        spot_gate_holds(0.315)
    ));
    // PNN50_THRESHOLD_MS is reached through the 200/3 gate at hrv.rs:836.
    let pn = [800u16, 900, 890, 990];
    let contig = vec![false, true, true, true];
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        tb.note(format!(
            "PNN50_THRESHOLD_MS 50 -> {:.2} ({label}) gives pnn50 {:.6} against the shipped 66.666667 at hrv.rs:836",
            50.0 * f,
            shadow_pnn50(&pn, &contig, 50.0 * f)
        ));
    }
    tb.note(format!(
        "MIN_BEATS {}",
        break_at(&tb.gate, &|f| shadow_analyze_sdnn(&series, scale_u(20, f), None, RR_BASE))
    ));
    tb.note("SDNN is order-free, so shuffle and reversal cannot move it - this gate sees no ordering at all");
    report(&tb, t);
}

// ─────────────────────────── M4: rolling RMSSD ───────────────────────────

fn m4_rolling_rmssd(t: &mut Tally) {
    // hrv.rs:848 - `rolling_rmssd_tracks_the_alternation_not_the_beat_period`. +/-3.0 on a 10.0 target.
    let gate = Gate {
        label: "rolling_rmssd(series, 60, 0, 8).last().1",
        source: "crates/physio-algo/src/hrv.rs:848",
        target: 10.0,
        tol: 3.0,
    };
    let series: Vec<(i64, u16)> = (0..30).map(|i| (i, if i % 2 == 0 { 800u16 } else { 810 })).collect();
    let last = |v: Vec<(i64, f64)>| v.last().map(|&(_, x)| x).unwrap_or(f64::NAN);
    let mut tb = Table::new("Rolling RMSSD (300 s trailing window series)", gate);
    let shipped = last(rolling_rmssd(&series, 60, 0, 8));
    let base = last(shadow_rolling(&series, 60, 0, 8, RR_BASE));
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: mean R-R (805)", 805.0);
    tb.add(Kind::Null, "output: midpoint of the tolerance band (10.0)", 10.0);
    tb.add(Kind::Null, "input: deterministic shuffle of the VALUES", last(shadow_rolling(
        &series.iter().map(|&(ts, _)| ts).zip(shuffled(&series.iter().map(|&(_, v)| v).collect::<Vec<_>>())).collect::<Vec<_>>(),
        60, 0, 8, RR_BASE,
    )));

    let rev: Vec<(i64, u16)> = series.iter().map(|&(ts, _)| ts).zip(series.iter().rev().map(|&(_, v)| v)).collect();
    tb.add(Kind::Structural, "input: values reversed against their timestamps", last(shadow_rolling(&rev, 60, 0, 8, RR_BASE)));
    let off: Vec<(i64, u16)> = series.iter().map(|&(ts, v)| (ts, v + 100)).collect();
    tb.add(Kind::Structural, "input: +100 ms constant offset", last(shadow_rolling(&off, 60, 0, 8, RR_BASE)));
    tb.add(Kind::Structural, "input: drop last 10% (3 beats)", last(shadow_rolling(&series[..27], 60, 0, 8, RR_BASE)));
    let flat: Vec<(i64, u16)> = series.iter().map(|&(ts, _)| (ts, 805u16)).collect();
    tb.add(Kind::Structural, "input: alternation flattened to a constant", last(shadow_rolling(&flat, 60, 0, 8, RR_BASE)));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let w = (60.0 * f).round() as i64;
        tb.add(Kind::Param, format!("param: window_s 60 -> {w} ({label})"), last(shadow_rolling(&series, w, 0, 8, RR_BASE)));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mb = scale_u(8, f);
        tb.add(Kind::Param, format!("param: min_beats 8 -> {mb} ({label})"), last(shadow_rolling(&series, 60, 0, mb, RR_BASE)));
        let mut p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(Kind::Param, format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh), last(shadow_rolling(&series, 60, 0, 8, p)));
    }
    tb.note(format!(
        "window_s {}",
        break_at(&tb.gate, &|f| last(shadow_rolling(&series, (60.0 * f).round() as i64, 0, 8, RR_BASE)))
    ));
    tb.note("ROLLING_WINDOW_SECS (300) is a caller argument here, not the constant - the gate passes 60");
    report(&tb, t);
}

fn m4b_rolling_timestamps(t: &mut Tally) {
    // hrv.rs:856 - `rolling_rmssd_emits_from_the_first_full_window_and_thins_on_a_stride`.
    let gate = Gate {
        label: "rolling_rmssd(steady, 300, 0, 8).last().0",
        source: "crates/physio-algo/src/hrv.rs:856",
        target: 109.0,
        tol: 0.0,
    };
    let steady: Vec<(i64, u16)> = (0..10).map(|i| (100 + i, 800u16)).collect();
    let last_ts = |v: Vec<(i64, f64)>| v.last().map(|&(ts, _)| ts as f64).unwrap_or(f64::NAN);
    let mut tb = Table::new("Rolling RMSSD emission timestamps", gate);
    tb.add(Kind::Baseline, "baseline (unmutated)", last_ts(rolling_rmssd(&steady, 300, 0, 8)));
    tb.add(Kind::Null, "output: no points at all", f64::NAN);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    let moved: Vec<(i64, u16)> = steady.iter().map(|&(ts, v)| (ts + 1000, v)).collect();
    tb.add(Kind::Structural, "input: timestamps shifted +1000 s", last_ts(rolling_rmssd(&moved, 300, 0, 8)));
    tb.add(Kind::Structural, "input: drop last beat", last_ts(rolling_rmssd(&steady[..9], 300, 0, 8)));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mb = scale_u(8, f);
        tb.add(Kind::Param, format!("param: min_beats 8 -> {mb} ({label})"), last_ts(rolling_rmssd(&steady, 300, 0, mb)));
        let w = (300.0 * f).round() as i64;
        tb.add(Kind::Param, format!("param: window_s 300 -> {w} ({label})"), last_ts(rolling_rmssd(&steady, w, 0, 8)));
    }
    tb.add(Kind::Param, "param: step_s 0 -> 10", last_ts(rolling_rmssd(&steady, 300, 10, 8)));
    report(&tb, t);
}

// ─────────────────────────── M5: R-R coverage / duplicates / overlap ───────────────────────────

fn m5_rr_coverage(t: &mut Tally) {
    // hrv.rs:891 - `rr_coverage_is_beat_time_over_elapsed_time`.
    let gate = Gate {
        label: "rr_coverage(ts, [1000.0; 5])",
        source: "crates/physio-algo/src/hrv.rs:891",
        target: 1.25,
        tol: 1e-9,
    };
    let ts: Vec<i64> = vec![100, 101, 102, 103, 104];
    let rr = [1000.0f64; 5];
    let mut tb = Table::new("R-R coverage / duplicate beats / overlapping reports", gate);
    tb.add(Kind::Baseline, "baseline (unmutated)", rr_coverage(&ts, &rr));
    tb.add(Kind::Null, "output: constant 1.0 (always plausible)", 1.0);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Structural, "input: timestamps reversed", rr_coverage(&ts.iter().rev().copied().collect::<Vec<_>>(), &rr));
    tb.add(Kind::Structural, "input: timestamps shifted +1000 s", rr_coverage(&ts.iter().map(|&x| x + 1000).collect::<Vec<_>>(), &rr));
    tb.add(Kind::Structural, "input: drop the last beat", rr_coverage(&ts[..4], &rr[..4]));
    tb.add(Kind::Structural, "input: every beat stored twice", rr_coverage(&[100, 100, 101, 101, 102, 102], &[1000.0; 6]));
    tb.add(Kind::Param, "param: R-R values +10% (1000 -> 1100)", rr_coverage(&ts, &[1100.0; 5]));
    tb.add(Kind::Param, "param: R-R values +0.5% (1000 -> 1005)", rr_coverage(&ts, &[1005.0; 5]));
    tb.note("rr_coverage has no tunable of its own; the perturbations above are the only way to move it");
    tb.note(format!(
        "duplicate_beat_count exact-repeat gate at hrv.rs:903: {} (shipped literal 1)",
        duplicate_beat_count(&[100, 100, 101], &[1000.0, 1000.0, 1010.0])
    ));
    tb.note(format!(
        "duplicate_beat_count is blind to a same-second DIFFERENT value: {} on a real overlap (hrv.rs:907)",
        duplicate_beat_count(&[100, 100], &[1000.0, 1010.0])
    ));
    report(&tb, t);
}

fn m5b_overlapping_reports(t: &mut Tally) {
    // hrv.rs:923 - the tracking leg of `overlapping_report_count_sees_the_overlap_duplicates_miss`.
    let gate = Gate {
        label: "overlapping_report_count(tracking).0",
        source: "crates/physio-algo/src/hrv.rs:923",
        target: 0.0,
        tol: 0.0,
    };
    let tracking: Vec<(u32, Vec<u16>)> = (1u32..=12).map(|t| (t, vec![1000u16])).collect();
    let over: Vec<(u32, Vec<u16>)> = (1u32..=12).map(|t| (t, vec![800u16, 860, 920, 980])).collect();
    let mut tb = Table::new("Overlapping-report count (seam detection)", gate);
    let shipped = overlapping_report_count(&tracking).0 as f64;
    let base = shadow_overlapping(&tracking, 2000) as f64;
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: every report flagged overlapping", 12.0);
    tb.add(Kind::Structural, "input: the overrunning stream instead", shadow_overlapping(&over, 2000) as f64);
    let doubled: Vec<(u32, Vec<u16>)> = tracking.iter().map(|(t, v)| (*t, vec![v[0], v[0]])).collect();
    tb.add(Kind::Structural, "input: every beat re-reported in place", shadow_overlapping(&doubled, 2000) as f64);
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let s = (2000.0 * f).round() as u64;
        tb.add(Kind::Param, format!("param: SEAM_SLACK_MS 2000 -> {s} ({label})"), shadow_overlapping(&tracking, s) as f64);
    }
    tb.note(format!(
        "SEAM_SLACK_MS {}",
        break_at(&tb.gate, &|f| shadow_overlapping(&tracking, (2000.0 * f).round() as u64) as f64)
    ));
    tb.note(format!(
        "on the OVERRUNNING stream the shipped claim is only `> 0`; the count at slack 2000 is {} and at 4000 is {}",
        shadow_overlapping(&over, 2000),
        shadow_overlapping(&over, 4000)
    ));
    report(&tb, t);
}

// ─────────────────────────── M6: frequency-domain HRV ───────────────────────────

fn m6_freq_band_power(t: &mut Tally) {
    // The presence/ordering gates compare no band POWER to a target, so this table's gate is
    // deliberately infinitely wide. The magnitudes are still ungated; only the RATIO is now gated,
    // by `lfhf_recovers_a_planted_power_ratio`.
    let night = resp_night(300.0, 0.25);
    let hf_of = |b: Option<(Option<f64>, f64, Option<f64>, f64)>| b.map(|(_, hf, _, _)| hf).unwrap_or(f64::NAN);
    let base = hf_of(shadow_freq(&night, FREQ_BASE));
    let shipped = freq_domain(&night).unwrap().hf;
    assert!((shipped - base).abs() < 1e-9, "shadow {base} != shipped {shipped}");
    let gate = Gate {
        label: "HF power (ms^2) - NO NUMERIC GATE EXISTS",
        source: "crates/physio-algo/src/hrv_freq.rs, presence/ordering only - no power target exists",
        target: base,
        tol: f64::INFINITY,
    };
    let mut tb = Table::new("Frequency-domain HRV (LF, HF, LF/HF, total power)", gate).blind_null();
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: constant HF = 1.0 ms^2", 1.0);
    tb.add(Kind::Null, "output: HF = the series variance", population_sd(&night.iter().map(|&v| v as f64).collect::<Vec<_>>()).powi(2));
    tb.add(Kind::Structural, "input: reversed tachogram", hf_of(shadow_freq(&night.iter().rev().copied().collect::<Vec<_>>(), FREQ_BASE)));
    tb.add(Kind::Structural, "input: deterministic shuffle", hf_of(shadow_freq(&shuffled(&night), FREQ_BASE)));
    tb.add(Kind::Structural, "input: modulation halved (40 -> 20 ms)", hf_of(shadow_freq(&resp_night(300.0, 0.25).iter().map(|&v| 900 + (v as i32 - 900) as u16 / 2).collect::<Vec<_>>(), FREQ_BASE)));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = FREQ_BASE;
        p.hf_low = 0.15 * f;
        tb.add(Kind::Param, format!("param: HF_LOW_HZ 0.15 -> {:.4} ({label})", p.hf_low), hf_of(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.hf_high = 0.40 * f;
        tb.add(Kind::Param, format!("param: HF_HIGH_HZ 0.40 -> {:.4} ({label})", p.hf_high), hf_of(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.step_hz = 0.005 * f;
        tb.add(Kind::Param, format!("param: FREQ_STEP_HZ 0.005 -> {:.5} ({label})", p.step_hz), hf_of(shadow_freq(&night, p)));
    }
    let mut p = FREQ_BASE;
    p.step_hz = 0.005 * 1.005;
    tb.add(Kind::Param, "param: FREQ_STEP_HZ 0.005 -> 0.005025 (+0.5%)", hf_of(shadow_freq(&night, p)));
    tb.note("CRITICAL: no shipped assertion anywhere compares a band power to a target, so every arm above is unfalsifiable");
    tb.note(format!(
        "for scale, a +10% HF_HIGH_HZ moves HF by {:.3}% and the shipped file would not notice",
        {
            let mut q = FREQ_BASE;
            q.hf_high = 0.40 * 1.10;
            (hf_of(shadow_freq(&night, q)) / base - 1.0) * 100.0
        }
    ));
    report(&tb, t);
    t.criticals.push(
        "hrv_freq: LF, HF and total_power still have NO numeric gate - only their RATIO does, via \
         hrv_freq.rs lfhf_recovers_a_planted_power_ratio. Measured this pass: the band integrals are \
         normalised Lomb-Scargle power, not ms^2, and DOUBLE when the record doubles"
            .to_string(),
    );
}

fn m6b_freq_presence(t: &mut Tally) {
    // The four presence/ordering claims of `lf_and_total_present_past_the_lf_gate`.
    let gate = Gate {
        label: "presence + ordering score (1.0 = all four claims hold)",
        source: "crates/physio-algo/src/hrv_freq.rs lf_and_total_present_past_the_lf_gate",
        target: 1.0,
        tol: 0.0,
    };
    let night = resp_night(300.0, 0.25);
    let mut tb = Table::new("Frequency-domain HRV presence + ordering", gate).blind_null();
    tb.add(Kind::Baseline, "baseline (unmutated)", freq_presence_score(shadow_freq(&night, FREQ_BASE)));
    tb.add(Kind::Null, "output: constant (lf 0.5, hf 1.0, lfhf 0.5, total 1.5)", freq_presence_score(Some((Some(0.5), 1.0, Some(0.5), 1.5))));
    tb.add(Kind::Null, "output: nothing at all", freq_presence_score(None));
    let swapped = shadow_freq(&night, FREQ_BASE).map(|(lf, hf, lfhf, tot)| (Some(hf), lf.unwrap_or(0.0), lfhf, tot));
    tb.add(Kind::Structural, "output: LF and HF swapped", freq_presence_score(swapped));
    tb.add(Kind::Structural, "input: span halved to 150 s (under the LF gate)", freq_presence_score(shadow_freq(&resp_night(150.0, 0.25), FREQ_BASE)));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = FREQ_BASE;
        p.min_span_lf = 250.0 * f;
        tb.add(Kind::Param, format!("param: MIN_SPAN_FOR_LF_SEC 250 -> {:.1} ({label})", p.min_span_lf), freq_presence_score(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.min_span_hf = 60.0 * f;
        tb.add(Kind::Param, format!("param: MIN_SPAN_FOR_HF_SEC 60 -> {:.1} ({label})", p.min_span_hf), freq_presence_score(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.min_beats = scale_u(20, f);
        tb.add(Kind::Param, format!("param: MIN_BEATS 20 -> {} ({label})", p.min_beats), freq_presence_score(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.lf_low = 0.04 * f;
        tb.add(Kind::Param, format!("param: LF_LOW_HZ 0.04 -> {:.4} ({label})", p.lf_low), freq_presence_score(shadow_freq(&night, p)));
        p = FREQ_BASE;
        p.vlf_low = 0.0033 * f;
        tb.add(Kind::Param, format!("param: VLF_LOW_HZ 0.0033 -> {:.5} ({label})", p.vlf_low), freq_presence_score(shadow_freq(&night, p)));
    }
    tb.note(format!(
        "MIN_SPAN_FOR_LF_SEC {}",
        break_at(&tb.gate, &|f| {
            let mut q = FREQ_BASE;
            q.min_span_lf = 250.0 * f;
            freq_presence_score(shadow_freq(&night, q))
        })
    ));
    tb.note(format!(
        "MIN_BEATS {}",
        break_at(&tb.gate, &|f| {
            let mut q = FREQ_BASE;
            q.min_beats = scale_u(20, f);
            freq_presence_score(shadow_freq(&night, q))
        })
    ));
    report(&tb, t);
}

// ─────────────────────────── M7: sleeping respiratory rate ───────────────────────────

fn m7_respiratory_rate(t: &mut Tally) {
    // The 15/min arm of the swept gate, at the +/-2.5 bpm the sweep now carries.
    let gate = Gate {
        label: "resp_rate_from_rr(synth(0.25, 1000, 40, 420))",
        source: "crates/physio-algo/src/respiratory_rate.rs, the 15/min arm of the swept gate",
        target: 15.0,
        tol: 3.0,
    };
    let (rows, start, end) = resp_synth(0.25, 1000.0, 40.0, 420.0);
    let mut tb = Table::new("Sleeping respiratory rate (breaths/min via RSA)", gate).blind_null();
    let shipped = resp_rate_from_rr(&rows, start, end).unwrap();
    let base = shadow_resp(&rows, start, end, RESP_BASE);
    assert!((shipped - base).abs() < 1e-12, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 13.0 bpm", 13.0);
    tb.add(Kind::Null, "output: midpoint of the plausible band (16.5)", 16.5);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    let shuf_vals = shuffled(&rows.iter().map(|&(_, v)| v).collect::<Vec<_>>());
    let shuf: Vec<(i64, u16)> = rows.iter().map(|&(ts, _)| ts).zip(shuf_vals).collect();
    tb.add(Kind::Null, "input: deterministic shuffle of the tachogram", shadow_resp(&shuf, start, end, RESP_BASE));

    let rev: Vec<(i64, u16)> = rows.iter().map(|&(ts, _)| ts).zip(rows.iter().rev().map(|&(_, v)| v)).collect();
    tb.add(Kind::Structural, "input: tachogram reversed against its clock", shadow_resp(&rev, start, end, RESP_BASE));
    let (half, hs, he) = resp_synth(0.25, 1000.0, 20.0, 420.0);
    tb.add(Kind::Structural, "input: RSA amplitude halved (40 -> 20 ms)", shadow_resp(&half, hs, he, RESP_BASE));
    let (dbl, ds, de) = resp_synth(0.50, 1000.0, 40.0, 420.0);
    tb.add(Kind::Structural, "input: breathing rate doubled (0.25 -> 0.50 Hz)", shadow_resp(&dbl, ds, de, RESP_BASE));
    tb.add(Kind::Structural, "input: drop the last 10% of beats", shadow_resp(&rows[..rows.len() * 9 / 10], start, end, RESP_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = RESP_BASE;
        p.resample_hz = 4.0 * f;
        tb.add(Kind::Param, format!("param: RSA_RESAMPLE_HZ 4.0 -> {:.3} ({label})", p.resample_hz), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.detrend_window_s = 8.0 * f;
        tb.add(Kind::Param, format!("param: RSA_DETREND_WINDOW_S 8.0 -> {:.2} ({label})", p.detrend_window_s), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.min_peak_distance_s = 2.5 * f;
        tb.add(Kind::Param, format!("param: RSA_MIN_PEAK_DISTANCE_S 2.5 -> {:.3} ({label})", p.min_peak_distance_s), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.window_s = 300.0 * f;
        tb.add(Kind::Param, format!("param: RSA_WINDOW_S 300 -> {:.1} ({label})", p.window_s), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.min_breath_interval_s = 2.5 * f;
        tb.add(Kind::Param, format!("param: RSA_MIN_BREATH_INTERVAL_S 2.5 -> {:.3} ({label})", p.min_breath_interval_s), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.max_breath_interval_s = 10.0 * f;
        tb.add(Kind::Param, format!("param: RSA_MAX_BREATH_INTERVAL_S 10 -> {:.2} ({label})", p.max_breath_interval_s), shadow_resp(&rows, start, end, p));
        p = RESP_BASE;
        p.plausible_min_bpm = 8.0 * f;
        p.plausible_max_bpm = 25.0 * f;
        tb.add(Kind::Param, format!("param: RESP_PLAUSIBLE band 8-25 scaled ({label})"), shadow_resp(&rows, start, end, p));
    }
    let mut p = RESP_BASE;
    p.min_peak_distance_s = 2.5 * 1.005;
    tb.add(Kind::Param, "param: RSA_MIN_PEAK_DISTANCE_S 2.5 -> 2.5125 (+0.5%)", shadow_resp(&rows, start, end, p));

    // The slow-breather gate, which alone did not constrain a constant either.
    let (slow, ss, se) = resp_synth(11.0 / 60.0, 60000.0 / 55.0, 45.0, 480.0);
    let slow_v = shadow_resp(&slow, ss, se, RESP_BASE);
    tb.note(format!("slow-breather gate (11.0 +/- 2.0 and < 16.0): measured {slow_v:.6}"));
    tb.note(format!(
        "a CONSTANT 13.0 bpm satisfies this single tone AND the slow-breather gate: |13-15|=2 <= 3 \
         and |13-11|=2 <= 2 and 13 < 16 -> {}",
        (13.0f64 - 15.0).abs() <= 3.0 && (13.0f64 - 11.0).abs() <= 2.0 && 13.0 < 16.0
    ));
    tb.note("no PSG respiration reference is read anywhere; every gate is a synthetic self-check");
    report(&tb, t);
    t.criticals.push(
        "respiratory rate: the single 15/min tone measured above is satisfied by the constant 13.0 - \
         the swept gate respiratory_rate.rs tracks_a_swept_breathing_rate_across_the_band now rejects \
         it, and every constant in the plausible band, over 10-20 breaths/min on three profiles"
            .to_string(),
    );
}

// ─────────────────────────── M8: resting HR ───────────────────────────

fn m8_resting_hr(t: &mut Tally) {
    // `session_floor_recovers_multiple_injected_values`, floor 48.
    let gate = Gate {
        label: "session_resting_hr_floor(1000, 2800, night_with_floor(1000, 48, 60))",
        source: "crates/physio-algo/tests/resting_hr_parity.rs session_floor_recovers_multiple_injected_values",
        target: 48.0,
        tol: 0.0,
    };
    let hr = night_with_floor(1000, 48, 60);
    let (start, end) = (1000i64, 1000 + 6 * 5 * 60);
    let mut tb = Table::new("Resting HR (session lowest-sustained floor)", gate);
    // The FLOOR, which is what every mutation below varies. The shipped resting HR is the median
    // (resting_hr.rs); its null arm lives in resting_hr_parity.rs, not here.
    let shipped = session_resting_hr_floor(start, end, &hr).unwrap() as f64;
    let base = shadow_session_rhr(start, end, &hr, 300);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 60 (the night level)", 60.0);
    tb.add(Kind::Null, "output: whole-segment mean", mean(&hr.iter().map(|s| s.bpm as f64).collect::<Vec<_>>()));
    tb.add(Kind::Null, "output: global minimum sample (no windowing)", hr.iter().map(|s| s.bpm).min().unwrap() as f64);

    let shifted: Vec<HrSample> = hr.iter().map(|s| HrSample::new(s.ts + 150, s.bpm)).collect();
    tb.add(Kind::Structural, "input: timestamps shifted +150 s (half a window)", shadow_session_rhr(start, end, &shifted, 300));
    let plus5: Vec<HrSample> = hr.iter().map(|s| HrSample::new(s.ts, s.bpm + 5)).collect();
    tb.add(Kind::Structural, "input: +5 bpm constant offset", shadow_session_rhr(start, end, &plus5, 300));
    let revd: Vec<HrSample> = hr.iter().map(|s| s.ts).zip(hr.iter().rev().map(|s| s.bpm)).map(|(ts, b)| HrSample::new(ts, b)).collect();
    tb.add(Kind::Structural, "input: bpm reversed against the clock", shadow_session_rhr(start, end, &revd, 300));
    tb.add(Kind::Structural, "input: drop the low window entirely", shadow_session_rhr(start, end, &hr.iter().filter(|s| s.bpm != 48).cloned().collect::<Vec<_>>(), 300));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let w = (300.0 * f).round() as i64;
        tb.add(Kind::Param, format!("param: WINDOW_SECONDS 300 -> {w} ({label})"), shadow_session_rhr(start, end, &hr, w));
    }
    tb.note(format!(
        "WINDOW_SECONDS {}",
        break_at(&tb.gate, &|f| shadow_session_rhr(start, end, &hr, (300.0 * f).round() as i64))
    ));
    tb.note(format!(
        "the global-min shortcut also matches at the other two injected floors: 44 -> {}, 52 -> {}",
        shadow_session_rhr(1000, 2800, &night_with_floor(1000, 44, 58), 300),
        shadow_session_rhr(1000, 2800, &night_with_floor(1000, 52, 70), 300)
    ));
    tb.note(format!(
        "only resting_hr_parity.rs:71 separates the floor from the global min: it reads {} where the min is 50",
        session_resting_hr_floor(0, 600, &[HrSample::new(0, 50), HrSample::new(60, 60), HrSample::new(300, 58), HrSample::new(360, 58)]).unwrap()
    ));
    tb.note(format!(
        "daily_resting_hr gate at resting_hr_parity.rs:101 reads {:?}; nothing compares any of this to an external RHR reference",
        daily_resting_hr(&[Some(52), None, Some(48), Some(55)])
    ));
    report(&tb, t);
}

// ─────────────────────────── M9 / M10: SpO2 ───────────────────────────

fn m9_spo2_paired(t: &mut Tally) {
    // The R = 0.5 -> 97.5 arm of the ratio-of-ratios curve walk.
    let gate = Gate {
        label: "Spo2::from_paired(win(100, 2), win(100, 4))",
        source: "crates/physio-algo/src/spo2.rs, the R=0.5 arm of the curve walk",
        target: 97.5,
        tol: 0.0,
    };
    let red = spo2_win(100.0, 2.0);
    let ir = spo2_win(100.0, 4.0);
    let mut tb = Table::new("SpO2 from paired red/IR (ratio-of-ratios)", gate);
    let shipped = Spo2::from_paired(&red, &ir).unwrap();
    let base = shadow_spo2_paired(&red, &ir, SPO2_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 85.0 (the R=1 curve midpoint)", 85.0);
    tb.add(Kind::Null, "output: midpoint of the clamp band ((70+100)/2)", 85.0);
    tb.add(Kind::Null, "output: constant 97.5", 97.5);
    tb.add(Kind::Structural, "input: red and IR swapped", shadow_spo2_paired(&ir, &red, SPO2_BASE));
    tb.add(Kind::Structural, "input: both channels reversed", shadow_spo2_paired(&red.iter().rev().copied().collect::<Vec<_>>(), &ir.iter().rev().copied().collect::<Vec<_>>(), SPO2_BASE));
    tb.add(Kind::Structural, "input: both DC levels doubled", shadow_spo2_paired(&spo2_win(200.0, 2.0), &spo2_win(200.0, 4.0), SPO2_BASE));
    tb.add(Kind::Structural, "input: red AC flattened to zero", shadow_spo2_paired(&[100.0; 20], &ir, SPO2_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let mut p = SPO2_BASE;
        p.curve_a = 110.0 * f;
        tb.add(Kind::Param, format!("param: CURVE_A 110 -> {:.3} ({label})", p.curve_a), shadow_spo2_paired(&red, &ir, p));
        p = SPO2_BASE;
        p.curve_b = 25.0 * f;
        tb.add(Kind::Param, format!("param: CURVE_B 25 -> {:.3} ({label})", p.curve_b), shadow_spo2_paired(&red, &ir, p));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = SPO2_BASE;
        p.window_seconds = scale_u(30, f);
        tb.add(Kind::Param, format!("param: WINDOW_SECONDS 30 -> {} ({label})", p.window_seconds), shadow_spo2_paired(&red, &ir, p));
        p = SPO2_BASE;
        p.min_samples_per_window = scale_u(10, f);
        tb.add(Kind::Param, format!("param: MIN_SAMPLES_PER_WINDOW 10 -> {} ({label})", p.min_samples_per_window), shadow_spo2_paired(&red, &ir, p));
        p = SPO2_BASE;
        p.min_pulsatile_fraction = 0.5 * f;
        tb.add(Kind::Param, format!("param: MIN_PULSATILE_FRACTION 0.5 -> {:.3} ({label})", p.min_pulsatile_fraction), shadow_spo2_paired(&red, &ir, p));
        p = SPO2_BASE;
        p.clamp_low = 70.0 * f;
        tb.add(Kind::Param, format!("param: CLAMP_LOW 70 -> {:.1} ({label})", p.clamp_low), shadow_spo2_paired(&red, &ir, p));
        p = SPO2_BASE;
        p.clamp_high = 100.0 * f;
        tb.add(Kind::Param, format!("param: CLAMP_HIGH 100 -> {:.1} ({label})", p.clamp_high), shadow_spo2_paired(&red, &ir, p));
    }
    tb.note(format!(
        "WINDOW_SECONDS {}",
        break_at(&tb.gate, &|f| {
            let mut q = SPO2_BASE;
            q.window_seconds = scale_u(30, f);
            shadow_spo2_paired(&red, &ir, q)
        })
    ));
    tb.note(format!(
        "CLAMP_HIGH {}",
        break_at(&tb.gate, &|f| {
            let mut q = SPO2_BASE;
            q.clamp_high = 100.0 * f;
            shadow_spo2_paired(&red, &ir, q)
        })
    ));
    tb.note("the value is WITHHELD in practice (the 1 Hz pair aliases the cardiac band away), so this golden pins the CURVE, not a reading");
    report(&tb, t);
}

fn m10_spo2_rolling(t: &mut Tally) {
    // `rolling_reading_carries_the_recent_week_against_the_month`.
    let gate = Gate {
        label: "Spo2::rolling_reading(SPREAD_NIGHTS).pct",
        source: "crates/physio-algo/src/spo2.rs rolling_reading_carries_the_recent_week_against_the_month",
        target: 94.0,
        tol: 0.0,
    };
    let nights = SPO2_SPREAD_NIGHTS;
    let mut tb = Table::new("SpO2 rolling multi-night reading", gate);
    let shipped = Spo2::rolling_reading(&nights).pct.unwrap();
    let base = shadow_spo2_rolling(&nights, SPO2_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: the ANCHOR itself, rounded (96.5 -> 97)", 97.0);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: the 7-night median with no anchoring", (median(&nights[23..]) + 0.5).floor());
    let reversed: Vec<f64> = nights.iter().rev().copied().collect();
    tb.add(Kind::Structural, "input: the 30-night order reversed", shadow_spo2_rolling(&reversed, SPO2_BASE));
    tb.add(Kind::Structural, "input: the spread flattened to its own median", shadow_spo2_rolling(&vec![median(&nights); 30], SPO2_BASE));
    tb.add(Kind::Structural, "input: only the most recent night kept", shadow_spo2_rolling(&nights[29..], SPO2_BASE));
    tb.add(Kind::Structural, "input: no nights at all", shadow_spo2_rolling(&[], SPO2_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005), ("+2%", 1.02)] {
        let mut p = SPO2_BASE;
        p.anchor = 96.5 * f;
        tb.add(Kind::Param, format!("param: ANCHOR 96.5 -> {:.4} ({label})", p.anchor), shadow_spo2_rolling(&nights, p));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = SPO2_BASE;
        p.roll_window_nights = scale_u(30, f);
        tb.add(Kind::Param, format!("param: ROLL_WINDOW_NIGHTS 30 -> {} ({label})", p.roll_window_nights), shadow_spo2_rolling(&nights, p));
        p = SPO2_BASE;
        p.recent_nights = scale_u(7, f);
        tb.add(Kind::Param, format!("param: RECENT_NIGHTS 7 -> {} ({label})", p.recent_nights), shadow_spo2_rolling(&nights, p));
        p = SPO2_BASE;
        p.rolling_clamp_low = 88.0 * f;
        tb.add(Kind::Param, format!("param: ROLLING_CLAMP_LOW 88 -> {:.2} ({label})", p.rolling_clamp_low), shadow_spo2_rolling(&nights, p));
        p = SPO2_BASE;
        p.rolling_clamp_high = 100.0 * f;
        tb.add(Kind::Param, format!("param: ROLLING_CLAMP_HIGH 100 -> {:.2} ({label})", p.rolling_clamp_high), shadow_spo2_rolling(&nights, p));
        p = SPO2_BASE;
        p.min_nights = scale_u(1, f);
        tb.add(Kind::Param, format!("param: MIN_NIGHTS 1 -> {} ({label})", p.min_nights), shadow_spo2_rolling(&nights, p));
    }
    tb.note(format!(
        "ANCHOR {}",
        break_at(&tb.gate, &|f| {
            let mut q = SPO2_BASE;
            q.anchor = 96.5 * f;
            shadow_spo2_rolling(&nights, q)
        })
    ));
    tb.note(format!(
        "nightly_raw_means gate at spo2.rs:213 reads {:?}",
        Spo2::nightly_raw_means(
            &[(1000i64, 2000i64)],
            &(0..20).map(|i| (1000 + i, if i % 2 == 0 { 29000 } else { 31000 }, if i % 2 == 0 { 19000 } else { 21000 })).collect::<Vec<_>>()
        )
    ));
    tb.note("a constant window (one night, or any flat series) cancels offset against recent, so it reads the ANCHOR for ANY input value - only a spread input measures the data");
    tb.note("the half-up floor quantises to whole percent, so a sub-0.5 pp move is invisible by construction");
    report(&tb, t);
}

// ─────────────────────────── M11: Vitality / Body Age ───────────────────────────

fn m11_vitality(t: &mut Tally) {
    // vitality.rs:288 - `a_person_at_every_reference_reads_their_own_age`.
    let gate = Gate {
        label: "vitality::compute(reference()).body_age",
        source: "crates/physio-algo/src/vitality.rs:288",
        target: 40.0,
        tol: 1e-9,
    };
    let mut tb = Table::new("Vitality (0-100) and Body Age (years)", gate);
    let shipped = compute(&VitalityInput {
        chrono_age: 40.0,
        resting_hr: Some(65.0),
        vo2max: Some(45.0),
        expected_vo2max: Some(45.0),
        sleep_hours: Some(7.5),
        sleep_regularity_index: Some(68.25),
        rmssd: Some(45.0),
        rmssd_norm: Some(45.0),
        steps: Some(7000.0),
        ..Default::default()
    })
    .unwrap();
    let base = shadow_vitality(vit_reference(), VIT_BASE).0;
    assert!((shipped.body_age - base).abs() < 1e-12, "shadow {base} != shipped {}", shipped.body_age);
    assert_eq!(shipped.factors_used, 6);
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: chronological age, drivers ignored", 40.0);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Null, "output: midpoint of the clamp band (55)", 55.0);

    let mut three = vit_reference();
    three.vo2max = None;
    three.rmssd = None;
    three.steps = None;
    tb.add(Kind::Structural, "input: half the drivers dropped (3 left)", shadow_vitality(three, VIT_BASE).0);
    let mut two = three;
    two.sleep_hours = None;
    tb.add(Kind::Structural, "input: only 2 drivers left", shadow_vitality(two, VIT_BASE).0);
    let mut worse = vit_reference();
    worse.resting_hr = Some(80.0);
    tb.add(Kind::Structural, "input: resting HR 65 -> 80", shadow_vitality(worse, VIT_BASE).0);
    let mut swapped = vit_reference();
    swapped.vo2max = Some(45.0);
    swapped.expected_vo2max = Some(55.0);
    tb.add(Kind::Structural, "input: VO2max and its expectation pulled apart", shadow_vitality(swapped, VIT_BASE).0);

    let setters: Vec<VitSetter> = vec![
        ("MORTALITY_DOUBLING_YEARS", 10.0, |p, v| p.doubling_years = v),
        ("OVERLAP_SHRINK", 0.75, |p, v| p.overlap_shrink = v),
        ("MIN_BODY_AGE", 20.0, |p, v| p.min_body_age = v),
        ("MAX_BODY_AGE", 90.0, |p, v| p.max_body_age = v),
        ("VITALITY_PER_YEAR", 2.5, |p, v| p.vitality_per_year = v),
        ("HR_RESTING_PER_10BPM", 0.100, |p, v| p.hr_resting_per_10bpm = v),
        ("RESTING_HR_REFERENCE", 65.0, |p, v| p.resting_hr_reference = v),
        ("HR_VO2MAX_PER_MET", 0.130, |p, v| p.hr_vo2max_per_met = v),
        ("MET_ML_KG_MIN", 3.5, |p, v| p.met_ml_kg_min = v),
        ("VO2MAX_MET_CLAMP", 4.0, |p, v| p.vo2max_met_clamp = v),
        ("HR_SLEEP_PER_HOUR", 0.110, |p, v| p.hr_sleep_per_hour = v),
        ("SLEEP_OPTIMUM_HOURS", 7.5, |p, v| p.sleep_optimum_hours = v),
        ("SLEEP_DEADBAND_HOURS", 0.5, |p, v| p.sleep_deadband_hours = v),
        ("SLEEP_DEVIATION_CLAMP", 3.0, |p, v| p.sleep_deviation_clamp = v),
        ("HR_SLEEP_REGULARITY_PER_SRI_POINT", 0.015609, |p, v| p.hr_sri_per_point = v),
        ("SRI_MEDIAN_REFERENCE", 68.25, |p, v| p.sri_median_reference = v),
        ("SRI_LN_HAZARD_CLAMP", 0.50, |p, v| p.sri_ln_hazard_clamp = v),
        ("HR_HRV_PER_FRACTION", 0.160, |p, v| p.hr_hrv_per_fraction = v),
        ("HR_STEPS_PER_1000", 0.064, |p, v| p.hr_steps_per_1000 = v),
        ("STEPS_REFERENCE", 7000.0, |p, v| p.steps_reference = v),
        ("STEPS_CLAMP_HI", 11000.0, |p, v| p.steps_clamp_hi = v),
        ("STEPS_CLAMP", 4.0, |p, v| p.steps_clamp = v),
    ];
    for (name, base_v, set) in &setters {
        for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
            let mut p = VIT_BASE;
            set(&mut p, base_v * f);
            tb.add(
                Kind::Param,
                format!("param: {name} {base_v} -> {:.6} ({label})", base_v * f),
                shadow_vitality(vit_reference(), p).0,
            );
        }
    }
    let mut p = VIT_BASE;
    p.doubling_years = 10.0 * 1.005;
    tb.add(Kind::Param, "param: MORTALITY_DOUBLING_YEARS 10 -> 10.05 (+0.5%)", shadow_vitality(vit_reference(), p).0);
    for mf in [2usize, 6, 7] {
        p = VIT_BASE;
        p.min_factors = mf;
        tb.add(Kind::Param, format!("param: MIN_FACTORS 3 -> {mf}"), shadow_vitality(vit_reference(), p).0);
    }

    tb.note("every driver is AT its reference here, so the whole hazard sum is zero and no coefficient can move the answer");
    tb.note("SRI_MEDIAN_REFERENCE is also the fixture's own input value (vitality.rs:275), so moving it moves both sides and cancels");
    tb.note(format!(
        "the vitality leg (vitality.rs:289) reads {:.9}; VITALITY_PER_YEAR multiplies a zero offset, so it is unreachable here",
        shadow_vitality(vit_reference(), VIT_BASE).1
    ));
    tb.note(format!(
        "vitality.rs:295 is what separates a chrono-age passthrough from the model: a healthier driver set reads {:.6}",
        {
            let mut h = vit_reference();
            h.resting_hr = Some(52.0);
            h.vo2max = Some(55.5);
            h.rmssd = Some(54.0);
            h.steps = Some(11_000.0);
            shadow_vitality(h, VIT_BASE).0
        }
    ));
    tb.note(format!(
        "sleep_consistency (the duration fallback driver) on [6,8,10] h reads {:?}",
        sleep_consistency(&[6.0, 8.0, 10.0])
    ));
    report(&tb, t);
    t.criticals.push(
        "vitality: a scorer that returns chronological age and ignores every driver PASSES the \
         reference gate (crates/physio-algo/src/vitality.rs:288)"
            .to_string(),
    );
}

fn m11b_vitality_sri(t: &mut Tally) {
    // vitality.rs:343 - the first of Cribb's two published points, the only pinned driver coefficient.
    let gate = Gate {
        label: "contributions(SRI 41).ln_hazard",
        source: "crates/physio-algo/src/vitality.rs:343",
        target: 1.53f64.ln(),
        tol: 0.01,
    };
    let only_sri = |sri: f64| VitDrivers {
        chrono_age: 40.0,
        resting_hr: None,
        vo2max: None,
        expected_vo2max: None,
        sleep_hours: None,
        sleep_regularity_index: Some(sri),
        rmssd: None,
        rmssd_norm: None,
        steps: None,
    };
    let at = |sri: f64, p: VitParams| shadow_contributions(only_sri(sri), p)[0];
    let mut tb = Table::new("Vitality SRI driver (the one calibrated coefficient)", gate);
    let shipped = contributions(&VitalityInput { sleep_regularity_index: Some(41.0), ..Default::default() })[0].ln_hazard;
    let base = at(41.0, VIT_BASE);
    assert!((shipped - base).abs() < 1e-12, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: constant 0 (no hazard from irregularity)", 0.0);
    tb.add(Kind::Null, "output: constant ln(1.53) regardless of SRI", 1.53f64.ln());
    tb.add(Kind::Structural, "output: sign flipped", -base);
    tb.add(Kind::Structural, "input: SRI 41 -> 75 (the other published point)", at(75.0, VIT_BASE));
    tb.add(Kind::Structural, "input: SRI 41 -> the median", at(68.25, VIT_BASE));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+2%", 1.02), ("+0.5%", 1.005)] {
        let mut p = VIT_BASE;
        p.hr_sri_per_point = 0.015609 * f;
        tb.add(Kind::Param, format!("param: HR_SLEEP_REGULARITY_PER_SRI_POINT -> {:.8} ({label})", p.hr_sri_per_point), at(41.0, p));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let mut p = VIT_BASE;
        p.sri_median_reference = 68.25 * f;
        tb.add(Kind::Param, format!("param: SRI_MEDIAN_REFERENCE 68.25 -> {:.4} ({label})", p.sri_median_reference), at(41.0, p));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = VIT_BASE;
        p.sri_ln_hazard_clamp = 0.50 * f;
        tb.add(Kind::Param, format!("param: SRI_LN_HAZARD_CLAMP 0.50 -> {:.4} ({label})", p.sri_ln_hazard_clamp), at(41.0, p));
    }
    tb.note(format!(
        "HR_SLEEP_REGULARITY_PER_SRI_POINT {}",
        break_at(&tb.gate, &|f| {
            let mut q = VIT_BASE;
            q.hr_sri_per_point = 0.015609 * f;
            at(41.0, q)
        })
    ));
    tb.note(format!(
        "SRI_MEDIAN_REFERENCE {}",
        break_at(&tb.gate, &|f| {
            let mut q = VIT_BASE;
            q.sri_median_reference = 68.25 * f;
            at(41.0, q)
        })
    ));
    tb.note("with fixed inputs this is the only vitality gate that can move; the reference-point gate cannot");
    report(&tb, t);
}

fn m11c_rmssd_norm(t: &mut Tally) {
    // vitality.rs:360 - `rmssd_norm_interpolates_and_flattens`.
    let gate = Gate {
        label: "vitality::rmssd_norm(20.0)",
        source: "crates/physio-algo/src/vitality.rs:360",
        target: 47.0,
        tol: 0.0,
    };
    let mut tb = Table::new("Vitality RMSSD age-norm anchors", gate);
    tb.add(Kind::Baseline, "baseline (unmutated)", rmssd_norm(20.0));
    tb.add(Kind::Null, "output: constant 33 (the mid anchor)", 33.0);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Structural, "input: age 20 -> 30 (next anchor)", rmssd_norm(30.0));
    tb.add(Kind::Structural, "input: age 20 -> 90 (past the last anchor)", rmssd_norm(90.0));
    tb.add(Kind::Param, "proxy: input age +10% (20 -> 22)", rmssd_norm(22.0));
    tb.add(Kind::Param, "proxy: input age +0.5% (20 -> 20.1)", rmssd_norm(20.1));
    tb.add(Kind::Param, "proxy: input age -10% (20 -> 18, below the first anchor)", rmssd_norm(18.0));
    tb.note("RMSSD_NORM_ANCHORS is a private const array, so the anchors themselves cannot be moved from an integration test - the age arms are a labelled proxy");
    report(&tb, t);
}

// ─────────────────────────── M12: CosinorAge ───────────────────────────

fn m12_cosinor_age(t: &mut Tally) {
    // biological_age.rs:113 - `matches_reference_outputs`, the generic coefficient set.
    let gate = Gate {
        label: "cosinor_age(30, 25, -1.5, 40, Unknown).cosinor_age_years",
        source: "crates/physio-algo/src/biological_age.rs:113",
        target: 40.4054333204,
        tol: 1e-4,
    };
    let mut tb = Table::new("CosinorAge / Rhythm Age (circadian biological age)", gate);
    let shipped = cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Unknown).unwrap().cosinor_age_years;
    let base = shadow_cosinor_age(30.0, 25.0, -1.5, 40.0, AGE_GENERIC);
    assert!((shipped - base).abs() < 1e-9, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: chronological age (40.0)", 40.0);
    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(Kind::Structural, "coeffs: the female set instead of generic", nan_if_none(cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Female).map(|r| r.cosinor_age_years)));
    tb.add(Kind::Structural, "coeffs: the male set instead of generic", nan_if_none(cosinor_age(30.0, 25.0, -1.5, 40.0, Sex::Male).map(|r| r.cosinor_age_years)));
    tb.add(Kind::Structural, "input: acrophase sign flipped (-1.5 -> +1.5)", shadow_cosinor_age(30.0, 25.0, 1.5, 40.0, AGE_GENERIC));
    tb.add(Kind::Structural, "input: MESOR and amplitude swapped", shadow_cosinor_age(25.0, 30.0, -1.5, 40.0, AGE_GENERIC));

    let age_setters: Vec<AgeSetter> = vec![
        ("Coeffs.rate", -13.36715309, |p, v| p.rate = v),
        ("Coeffs.mesor", -0.03204933, |p, v| p.mesor = v),
        ("Coeffs.amp1", -0.01971357, |p, v| p.amp1 = v),
        ("Coeffs.phi1", -0.01664718, |p, v| p.phi1 = v),
        ("Coeffs.age", 0.10033692, |p, v| p.age = v),
        ("M_N", -1.405276, |p, v| p.m_n = v),
        ("M_D", 0.01462774, |p, v| p.m_d = v),
        ("BA_N", -0.01447851, |p, v| p.ba_n = v),
        ("BA_D", 0.112165, |p, v| p.ba_d = v),
        ("BA_I", 133.5989, |p, v| p.ba_i = v),
    ];
    for (name, base_v, set) in &age_setters {
        for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
            let mut p = AGE_GENERIC;
            set(&mut p, base_v * f);
            tb.add(
                Kind::Param,
                format!("param: {name} -> {:.8} ({label})", base_v * f),
                shadow_cosinor_age(30.0, 25.0, -1.5, 40.0, p),
            );
        }
    }
    tb.note(format!(
        "Coeffs.age {}",
        break_at(&tb.gate, &|f| {
            let mut q = AGE_GENERIC;
            q.age = 0.10033692 * f;
            shadow_cosinor_age(30.0, 25.0, -1.5, 40.0, q)
        })
    ));
    tb.note("ACTIVITY_TO_MG_ENMO_SCALE is applied by the CALLER, never inside cosinor_age, so no gate here can reach it");
    tb.note("Rhythm Age has never computed on real data (it needs 7 worn days of dynAccelG); this gate is a transform check only");
    report(&tb, t);
}

// ─────────────────────────── M13: circadian cosinor ───────────────────────────

fn m13_circadian(t: &mut Tally) {
    // circadian.rs:240 - the acrophase leg of `cosinor_recovers_injected_parameters`.
    let gate = Gate {
        label: "cosinor(synth(100, 40, 14)).acrophase_hours",
        source: "crates/physio-algo/src/circadian.rs:240",
        target: 14.0,
        tol: 1e-6,
    };
    let bins = circ_synth(100.0, 40.0, 14.0, W_BASE);
    let acro = |b: Option<(f64, f64, f64)>| b.map(|(_, _, a)| a).unwrap_or(f64::NAN);
    let mut tb = Table::new("Circadian cosinor (MESOR, amplitude, acrophase)", gate);
    let shipped = cosinor(&bins).unwrap();
    let base = acro(shadow_cosinor(&bins, W_BASE));
    assert!((shipped.acrophase_hours - base).abs() < 1e-12, "shadow {base} != shipped");
    assert!((shipped.mesor - 100.0).abs() < 1e-6 && (shipped.amplitude - 40.0).abs() < 1e-6);
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 14.0 h regardless of input", 14.0);
    tb.add(Kind::Null, "output: constant 12.0 h (mid-day)", 12.0);
    tb.add(Kind::Null, "output: no fit at all", f64::NAN);
    let shifted: Vec<ActivityBin> = bins.iter().map(|b| ActivityBin { hour: b.hour, activity: b.activity }).zip(1..).map(|(b, _)| b).collect();
    let rolled: Vec<ActivityBin> = (0..24).map(|h| ActivityBin { hour: h as f64, activity: shifted[(h + 23) % 24].activity }).collect();
    tb.add(Kind::Structural, "input: profile rolled one hour later", acro(shadow_cosinor(&rolled, W_BASE)));
    let mirrored: Vec<ActivityBin> = (0..24).map(|h| ActivityBin { hour: h as f64, activity: bins[23 - h].activity }).collect();
    tb.add(Kind::Structural, "input: profile mirrored against the clock", acro(shadow_cosinor(&mirrored, W_BASE)));
    tb.add(Kind::Structural, "input: amplitude halved (40 -> 20)", acro(shadow_cosinor(&circ_synth(100.0, 20.0, 14.0, W_BASE), W_BASE)));
    tb.add(Kind::Structural, "input: activity scaled x1000", acro(shadow_cosinor(&bins.iter().map(|b| ActivityBin { hour: b.hour, activity: b.activity * 1000.0 }).collect::<Vec<_>>(), W_BASE)));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let w = W_BASE * f;
        tb.add(
            Kind::Param,
            format!("param: W_HOURS x{f:.3} ({label}, input tracks it as shipped)"),
            acro(shadow_cosinor(&circ_synth(100.0, 40.0, 14.0, w), w)),
        );
        tb.add(
            Kind::Param,
            format!("proxy: W_HOURS x{f:.3} ({label}, FIT only, input pinned)"),
            acro(shadow_cosinor(&bins, w)),
        );
    }
    tb.note("the synthetic fixture is built from the SAME W_HOURS the fit uses, so the shipped gate cancels that constant out");
    tb.note(format!(
        "the amplitude leg (circadian.rs:239) is phase-blind: the one-hour-rolled profile still fits amplitude {:.6}",
        shadow_cosinor(&rolled, W_BASE).map(|(_, a, _)| a).unwrap_or(f64::NAN)
    ));
    report(&tb, t);
}

fn m13b_body_clock_phase(t: &mut Tally) {
    // circadian.rs:303-304 - `solid_when_well_covered_and_rhythmic`; the value is the acrophase, but only
    // when the confidence is Solid, so one number carries both shipped claims.
    let gate = Gate {
        label: "estimate_phase(synth(100, 40, 16), 14, 7.0, None): Solid acrophase",
        source: "crates/physio-algo/src/circadian.rs:303-304",
        target: 16.0,
        tol: 1e-6,
    };
    let bins = circ_synth(100.0, 40.0, 16.0, W_BASE);
    let mut tb = Table::new("Body-clock phase estimate", gate);
    let shipped = estimate_phase(&bins, 14, 7.0, None).unwrap();
    assert_eq!(shipped.confidence, PhaseConfidence::Solid);
    let base = shadow_phase_solid_acro(&bins, 14, W_BASE, 7, 14, 0.10);
    assert!((shipped.acrophase_hours - base).abs() < 1e-12, "shadow {base} != shipped");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: never Solid", f64::NAN);
    tb.add(Kind::Null, "output: constant 16.0 h, always Solid", 16.0);
    tb.add(Kind::Structural, "input: days_observed 14 -> 13 (Wide, not Solid)", shadow_phase_solid_acro(&bins, 13, W_BASE, 7, 14, 0.10));
    tb.add(Kind::Structural, "input: rhythm flattened (rel amp 0.01)", shadow_phase_solid_acro(&circ_synth(100.0, 1.0, 16.0, W_BASE), 14, W_BASE, 7, 14, 0.10));
    tb.add(Kind::Structural, "input: acrophase moved 16 -> 17 h", shadow_phase_solid_acro(&circ_synth(100.0, 40.0, 17.0, W_BASE), 14, W_BASE, 7, 14, 0.10));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        tb.add(Kind::Param, format!("param: MIN_DAYS_FOR_FIT 7 -> {} ({label})", scale_u(7, f)), shadow_phase_solid_acro(&bins, 14, W_BASE, scale_u(7, f) as u32, 14, 0.10));
        tb.add(Kind::Param, format!("param: GOOD_DAYS_FOR_FIT 14 -> {} ({label})", scale_u(14, f)), shadow_phase_solid_acro(&bins, 14, W_BASE, 7, scale_u(14, f) as u32, 0.10));
        tb.add(Kind::Param, format!("param: MIN_RELATIVE_AMPLITUDE 0.10 -> {:.4} ({label})", 0.10 * f), shadow_phase_solid_acro(&bins, 14, W_BASE, 7, 14, 0.10 * f));
    }
    tb.note(format!(
        "MIN_RELATIVE_AMPLITUDE {}",
        break_at(&tb.gate, &|f| shadow_phase_solid_acro(&bins, 14, W_BASE, 7, 14, 0.10 * f))
    ));
    tb.note("CBT_MIN_BEFORE_WAKE_HOURS and ACROPHASE_AFTER_CBT_MIN_HOURS only reach offset_vs_schedule_minutes and lean, which no shipped assertion reads - both are UNREACHABLE");
    report(&tb, t);
}

// ─────────────────────────── M14: HR anomaly watch ───────────────────────────

fn m14_hr_watch(t: &mut Tally) {
    // hr_anomaly.rs:115 - the peak leg of `sustained_elevated_at_rest_fires`.
    let gate = Gate {
        label: "HrWatch::evaluate(600 at 60 + 400 at 120).peak_bpm",
        source: "crates/physio-algo/src/hr_anomaly.rs:115",
        target: 120.0,
        tol: 0.0,
    };
    let mut h: Vec<HistoryRecord> = (0..600).map(|i| watch_rec(i, 60)).collect();
    h.extend((600..1000).map(|i| watch_rec(i, 120)));
    let mut tb = Table::new("HR anomaly watch (sustained elevated at-rest HR)", gate);
    let shipped = match HrWatch::evaluate(&h) {
        HrWatchState::ElevatedAtRest { peak_bpm, dur_s, .. } => {
            assert!(dur_s >= 300);
            peak_bpm as f64
        }
        other => panic!("expected elevated, got {other:?}"),
    };
    let base = shadow_watch_peak(&h, WATCH_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: always Normal", f64::NAN);
    tb.add(Kind::Null, "output: constant peak 120", 120.0);
    let mut short = (0..600).map(|i| watch_rec(i, 60)).collect::<Vec<_>>();
    short.extend((600..800).map(|i| watch_rec(i, 120)));
    short.extend((800..1000).map(|i| watch_rec(i, 60)));
    tb.add(Kind::Structural, "input: elevated run shortened to 200 s", shadow_watch_peak(&short, WATCH_BASE));
    let mut lower = (0..600).map(|i| watch_rec(i, 60)).collect::<Vec<_>>();
    lower.extend((600..1000).map(|i| watch_rec(i, 110)));
    tb.add(Kind::Structural, "input: elevated level 120 -> 110", shadow_watch_peak(&lower, WATCH_BASE));
    let reversed: Vec<HistoryRecord> = h.iter().rev().cloned().collect();
    tb.add(Kind::Structural, "input: record order reversed", shadow_watch_peak(&reversed, WATCH_BASE));
    let moving: Vec<HistoryRecord> = h
        .iter()
        .map(|r| {
            let mut c = r.clone();
            if c.heart_rate == Some(120) {
                c.activity_class = Some(2);
            }
            c
        })
        .collect();
    tb.add(Kind::Structural, "input: the elevated stretch marked as running", shadow_watch_peak(&moving, WATCH_BASE));

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = WATCH_BASE;
        p.min_baseline = scale_u(600, f);
        tb.add(Kind::Param, format!("param: MIN_BASELINE_SAMPLES 600 -> {} ({label})", p.min_baseline), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.sustain_s = (300.0 * f).round() as u32;
        tb.add(Kind::Param, format!("param: SUSTAIN_S 300 -> {} ({label})", p.sustain_s), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.max_gap_s = (5.0 * f).round() as u32;
        tb.add(Kind::Param, format!("param: MAX_GAP_S 5 -> {} ({label})", p.max_gap_s), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.elev_margin = (45.0 * f).round() as u8;
        tb.add(Kind::Param, format!("param: ELEV_MARGIN 45 -> {} ({label})", p.elev_margin), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.high_abs = (100.0 * f).round() as u8;
        tb.add(Kind::Param, format!("param: HIGH_ABS 100 -> {} ({label})", p.high_abs), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.qual_min = (192.0 * f).round().min(255.0) as u8;
        tb.add(Kind::Param, format!("param: QUAL_MIN 192 -> {} ({label})", p.qual_min), shadow_watch_peak(&h, p));
        p = WATCH_BASE;
        p.resting_pct = 0.10 * f;
        tb.add(Kind::Param, format!("param: RESTING_PCT 0.10 -> {:.4} ({label})", p.resting_pct), shadow_watch_peak(&h, p));
    }
    tb.note(format!(
        "HIGH_ABS {}",
        break_at(&tb.gate, &|f| {
            let mut q = WATCH_BASE;
            q.high_abs = (100.0 * f).round().min(255.0) as u8;
            shadow_watch_peak(&h, q)
        })
    ));
    tb.note("hr_anomaly.rs:106 asserts `need: MIN_BASELINE_SAMPLES` and :116 asserts `dur_s >= SUSTAIN_S`, both against the constant itself - those two are structurally UNREACHABLE");
    tb.note("HrWatch::evaluate is not exported over the FFI, so no Kotlin-side parity gate exists for it either");
    report(&tb, t);
}

// ─────────────────────────── M15: wear state ───────────────────────────

fn m15_worn(t: &mut Tally) {
    // worn.rs:86-127 - all six wear claims, scored 1.0 when every one holds.
    let gate = Gate {
        label: "wear-claim score (1.0 = all six shipped claims hold)",
        source: "crates/physio-algo/src/worn.rs:86-127",
        target: 1.0,
        tol: 0.0,
    };
    let unknown_rec =
        HistoryRecord { version: 24, unix: 1_784_000_000, heart_rate: Some(60), ..Default::default() };
    // `flags_bit` is what the fixture WRITES; `mask` is what the algorithm READS.
    let score = |mask: u8, flags_bit: u8, use_optical: bool| -> f64 {
        let off = worn_off_wrist();
        let on = worn_on_wrist();
        let mut flagged = worn_on_wrist();
        flagged.signal_flags = Some(flags_bit);
        let mut no_optical = worn_on_wrist();
        no_optical.optical_signal_poor = None;
        no_optical.optical_baseline_a = None;
        no_optical.optical_baseline_b = None;
        let mut poor_but_live = worn_on_wrist();
        poor_but_live.optical_signal_poor = Some(true);
        poor_but_live.optical_baseline_a = None;
        let judge = |h: &HistoryRecord| {
            if use_optical {
                shadow_worn(h, mask)
            } else {
                let flag = h.signal_flags.map(|f| f & mask == 0);
                match flag {
                    Some(false) => WornState::NotWorn,
                    None => WornState::Unknown,
                    Some(true) => WornState::Worn,
                }
            }
        };
        let ok = judge(&off) == WornState::NotWorn
            && judge(&on) == WornState::Worn
            && judge(&flagged) == WornState::NotWorn
            && judge(&unknown_rec) == WornState::Unknown
            && judge(&no_optical) == WornState::Worn
            && judge(&poor_but_live) == WornState::Worn;
        if ok {
            1.0
        } else {
            0.0
        }
    };
    let mut tb = Table::new("Wear state (worn / not worn / unknown)", gate);
    // The shipped module agrees with the shadow at the shipped mask.
    assert_eq!(worn_state(&worn_off_wrist()), WornState::NotWorn);
    assert_eq!(worn_state(&worn_on_wrist()), WornState::Worn);
    assert_eq!(worn_state(&unknown_rec), WornState::Unknown);
    tb.add(Kind::Baseline, "baseline (unmutated, mask 0x10)", score(0x10, 0x10, true));
    tb.add(Kind::Null, "output: always Worn", 0.0);
    tb.add(Kind::Null, "output: always Unknown", 0.0);
    tb.add(Kind::Structural, "rule: dead-optical-baseline evidence ignored", score(0x10, 0x10, false));
    tb.add(Kind::Param, "param: OFF_WRIST_BIT 0x10 -> 0x08 (fixture tracks it, as shipped)", score(0x08, 0x08, true));
    tb.add(Kind::Param, "param: OFF_WRIST_BIT 0x10 -> 0x20 (fixture tracks it, as shipped)", score(0x20, 0x20, true));
    tb.add(Kind::Param, "proxy: mask -> 0x08 with the fixture pinned at 0x10", score(0x08, 0x10, true));
    tb.add(Kind::Param, "proxy: mask -> 0x20 with the fixture pinned at 0x10", score(0x20, 0x10, true));
    tb.add(Kind::Param, "proxy: mask -> 0x18 (0x10 plus a neighbour bit)", score(0x18, 0x10, true));
    tb.note("worn.rs:99 sets the flags byte TO the constant, so the bit value cancels; only the pinned-fixture proxy can move it");
    tb.note("the fixture records are hand-built from real captured v18 values inside worn.rs - no fixture FILE is read, so nothing here can silently skip");
    report(&tb, t);
}

// ─────────────────────────── M16 / M17: stats + calibration ───────────────────────────

fn m16_trendline(t: &mut Tally) {
    // stats.rs:437 - the significance leg of `trendline_interval_sees_scatter_so_the_same_slope_can_read_flat`.
    let gate = Gate {
        label: "weighted_trendline([0,1,2], [0,2,1], [], 0.0).significance",
        source: "crates/physio-algo/src/stats.rs:437",
        target: 0.153_518_3,
        tol: 1e-6,
    };
    let days = [0.0, 1.0, 2.0];
    let values = [0.0, 2.0, 1.0];
    let sig = |v: Option<(f64, f64, f64, bool)>| v.map(|(_, _, s, _)| s).unwrap_or(f64::NAN);
    let mut tb = Table::new("Weighted trendline significance", gate);
    let shipped = weighted_trendline(&days, &values, &[], 0.0).unwrap();
    let base = sig(shadow_trendline(&days, &values, &[], 0.0, TREND_BASE));
    assert!((shipped.significance - base).abs() < 1e-12, "shadow {base} != shipped");
    assert_eq!(shipped.direction, TrendDirection::Flat);
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0 (never significant)", 0.0);
    tb.add(Kind::Null, "output: constant 1 (always significant)", 1.0);
    tb.add(Kind::Structural, "input: values reversed ([1,2,0])", sig(shadow_trendline(&days, &[1.0, 2.0, 0.0], &[], 0.0, TREND_BASE)));
    tb.add(Kind::Structural, "input: days shifted +100", sig(shadow_trendline(&[100.0, 101.0, 102.0], &values, &[], 0.0, TREND_BASE)));
    tb.add(Kind::Structural, "input: values scaled x2", sig(shadow_trendline(&days, &[0.0, 4.0, 2.0], &[], 0.0, TREND_BASE)));
    tb.add(Kind::Structural, "input: scatter removed ([0,0.5,1])", sig(shadow_trendline(&days, &[0.0, 0.5, 1.0], &[], 0.0, TREND_BASE)));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let mut p = TREND_BASE;
        p.ci_z = 1.282 * f;
        tb.add(Kind::Param, format!("param: TREND_CI_Z 1.282 -> {:.5} ({label})", p.ci_z), sig(shadow_trendline(&days, &values, &[], 0.0, p)));
        p = TREND_BASE;
        p.sig_half = 0.5 * f;
        tb.add(Kind::Param, format!("param: significance exponent 0.5 -> {:.5} ({label})", p.sig_half), sig(shadow_trendline(&days, &values, &[], 0.0, p)));
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = TREND_BASE;
        p.min_points = scale_u(3, f);
        tb.add(Kind::Param, format!("param: TREND_MIN_POINTS 3 -> {} ({label})", p.min_points), sig(shadow_trendline(&days, &values, &[], 0.0, p)));
    }
    tb.note(format!(
        "significance exponent {}",
        break_at(&tb.gate, &|f| {
            let mut q = TREND_BASE;
            q.sig_half = 0.5 * f;
            sig(shadow_trendline(&days, &values, &[], 0.0, q))
        })
    ));
    tb.note(format!(
        "TREND_CI_Z reaches only the DIRECTION leg (stats.rs:438). Flat at 1.282 = {}, at -50% = {}",
        shadow_trendline(&days, &values, &[], 0.0, TREND_BASE).map(|(_, _, _, fl)| fl).unwrap_or(false),
        {
            let mut q = TREND_BASE;
            q.ci_z = 1.282 * 0.5;
            shadow_trendline(&days, &values, &[], 0.0, q).map(|(_, _, _, fl)| fl).unwrap_or(false)
        }
    ));
    tb.note(format!(
        "significance is sign-blind and scale-blind by construction; the SLOPE legs (stats.rs:412, :467) are what pin those. half_change gate at stats.rs:476 reads {:?}",
        half_change(&[1.0, 2.0, 3.0, 4.0])
    ));
    report(&tb, t);
}

fn m16b_trend_min_span(t: &mut Tally) {
    // stats.rs:402 - `trend_min_span_is_a_third_of_the_window_floored_at_three_days`.
    let gate = Gate {
        label: "trend_min_span_days(7.0)",
        source: "crates/physio-algo/src/stats.rs:402",
        target: 3.0,
        tol: 1e-12,
    };
    let mut tb = Table::new("Trend minimum-span rule", gate);
    let shipped = trend_min_span_days(7.0);
    let base = shadow_min_span(7.0, TREND_BASE);
    assert!((shipped - base).abs() < 1e-15, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    tb.add(Kind::Null, "output: constant 0 (any span accepted)", 0.0);
    tb.add(Kind::Null, "output: the window itself (7.0)", 7.0);
    tb.add(Kind::Structural, "input: window 7 -> 31 days", shadow_min_span(31.0, TREND_BASE));
    tb.add(Kind::Structural, "input: window 7 -> 366 days", shadow_min_span(366.0, TREND_BASE));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let mut p = TREND_BASE;
        p.min_span_floor_days = 3.0 * f;
        tb.add(Kind::Param, format!("param: TREND_MIN_SPAN_FLOOR_DAYS 3.0 -> {:.4} ({label})", p.min_span_floor_days), shadow_min_span(7.0, p));
        p = TREND_BASE;
        p.min_span_window_fraction = (1.0 / 3.0) * f;
        tb.add(Kind::Param, format!("param: TREND_MIN_SPAN_WINDOW_FRACTION 1/3 -> {:.6} ({label})", p.min_span_window_fraction), shadow_min_span(7.0, p));
    }
    tb.note(format!(
        "at a 7-day window the FLOOR dominates, so the fraction is unreachable; at 31 days it reads {:.6} against the shipped 10.333333 (stats.rs:403)",
        {
            let mut q = TREND_BASE;
            q.min_span_window_fraction = (1.0 / 3.0) * 1.10;
            shadow_min_span(31.0, q)
        }
    ));
    report(&tb, t);
}

fn m17_calibration(t: &mut Tally) {
    // calibration.rs:70-73 - `unlock_and_full_gate_on_night_count`, scored 1.0 when all four hold.
    let gate = Gate {
        label: "calibration-claim score (1.0 = all four shipped claims hold)",
        source: "crates/physio-algo/src/calibration.rs:70-73",
        target: 1.0,
        tol: 0.0,
    };
    let mut tb = Table::new("Calibration unlock schedule (nights per metric)", gate);
    let base = calib_score(BLOOD_OXYGEN, SKIN_TEMP, CALORIES);
    tb.add(Kind::Baseline, "baseline (unmutated)", base);
    let zero = Calibration { unlock: 0, full: 0 };
    tb.add(Kind::Null, "output: every metric unlocked from night 0", calib_score(zero, zero, zero));
    let never = Calibration { unlock: 9999, full: 9999 };
    tb.add(Kind::Null, "output: nothing ever unlocks", calib_score(never, never, never));
    tb.add(Kind::Structural, "schedule: CALORIES unlock and full swapped", calib_score(BLOOD_OXYGEN, SKIN_TEMP, Calibration { unlock: CALORIES.full, full: CALORIES.unlock }));
    tb.add(Kind::Structural, "schedule: SKIN_TEMP given BLOOD_OXYGEN's schedule", calib_score(BLOOD_OXYGEN, BLOOD_OXYGEN, CALORIES));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        tb.add(Kind::Param, format!("param: BLOOD_OXYGEN (1,1) scaled ({label})"), calib_score(scale_calib(BLOOD_OXYGEN, f), SKIN_TEMP, CALORIES));
        tb.add(Kind::Param, format!("param: SKIN_TEMP (7,7) scaled ({label})"), calib_score(BLOOD_OXYGEN, scale_calib(SKIN_TEMP, f), CALORIES));
        tb.add(Kind::Param, format!("param: CALORIES (1,14) scaled ({label})"), calib_score(BLOOD_OXYGEN, SKIN_TEMP, scale_calib(CALORIES, f)));
    }
    tb.note(format!(
        "SKIN_TEMP {}",
        break_at(&tb.gate, &|f| calib_score(BLOOD_OXYGEN, scale_calib(SKIN_TEMP, f), CALORIES))
    ));
    tb.note(format!(
        "DAY_STRAIN is asserted at calibration.rs:78 only as immediate: unlocked(0) = {}, calibrated(0) = {}",
        DAY_STRAIN.unlocked(0),
        DAY_STRAIN.calibrated(0)
    ));
    tb.note("an integer schedule against an exact-boundary assertion is the tightest gate in this family: a rounded +/-10% is caught");
    report(&tb, t);
}

// ─────────────────── M18: the same numbers over REAL recorded R-R ───────────────────
//
// M1/M2/M3 above cannot reach RR_MIN_MS, RR_MAX_MS or ECTOPIC_THRESHOLD: their hand-made input never
// fires the range filter and clears the ectopic threshold threefold. These three tables run the same
// arms against `tests/hrv_real_rr.rs`, whose 500 tracked stretches reach every tunable.

/// The corpus `tests/hrv_real_rr.rs` gates on. It is TRACKED, so this PANICS when absent: a skip on a
/// missing fixture reports a PASS, and this project has lost gates that way.
fn real_corpus() -> Vec<Vec<u16>> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rhythm_rr");
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("rhythm R-R fixtures unusable at {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    let mut out = Vec::new();
    for path in &paths {
        for line in std::fs::read_to_string(path).unwrap().lines() {
            if line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(_rec), Some(_start), Some(_prem)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            out.push(it.map(|v| v.parse().unwrap()).collect());
        }
    }
    assert_eq!(out.len(), 500, "the corpus these gates were measured on is 500 stretches");
    out
}

/// Beats stamped by cumulative R-R time. Mirrors the stamping in `hrv_real_rr.rs`.
fn corpus_stamped(rr: &[u16]) -> Vec<(u32, u16)> {
    let mut cumulative_ms: u64 = 0;
    rr.iter()
        .map(|&v| {
            cumulative_ms += u64::from(v);
            ((cumulative_ms / 1000) as u32, v)
        })
        .collect()
}

/// Median over the corpus of a per-stretch scorer, skipping stretches that yield no number.
fn corpus_median(corpus: &[Vec<u16>], f: impl Fn(&[u16]) -> f64) -> f64 {
    let v: Vec<f64> = corpus.iter().map(|rr| f(rr)).filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        f64::NAN
    } else {
        median(&v)
    }
}

/// pNN50 through the same MIN_BEATS gate `analyze_raw` applies. Mirrors its pNN50 leg.
fn shadow_analyze_pnn50(rr: &[u16], min_beats: usize, p: RrParams) -> f64 {
    let (nn, contiguous) = shadow_clean_gap_aware(rr, &[], p);
    if nn.len() < min_beats {
        return f64::NAN;
    }
    shadow_pnn50(&nn, &contiguous, 50.0)
}

/// Every stretch mutated the same way, so the corpus median moves only if the mutation reaches it.
fn mapped(corpus: &[Vec<u16>], f: impl Fn(&[u16]) -> Vec<u16>) -> Vec<Vec<u16>> {
    corpus.iter().map(|rr| f(rr)).collect()
}

fn m18_rmssd_gap_aware_real(t: &mut Tally) {
    let gate = Gate {
        label: "hrv_real_rr ALL row: median nightly gap-aware RMSSD over 500 stretches",
        source: "crates/physio-algo/tests/hrv_real_rr.rs (ALL, rmssd)",
        target: 35.48921280381816,
        tol: 1e-9,
    };
    let corpus = real_corpus();
    let mut tb = Table::new("HRV RMSSD (gap-aware) over real recorded R-R", gate);

    // The shipped path builds per-second reports; the shadow sees the same beats with no seam break.
    let shipped = corpus_median(&corpus, |rr| {
        let mut reports: Vec<(u32, Vec<u16>)> = Vec::new();
        for (ts, v) in corpus_stamped(rr) {
            match reports.last_mut() {
                Some((rt, b)) if *rt == ts => b.push(v),
                _ => reports.push((ts, vec![v])),
            }
        }
        nan_if_none(HrvReadiness::rmssd_gap_aware(&reports))
    });
    let base = corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, RR_BASE));
    assert!((shipped - base).abs() < 1e-12, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(
        Kind::Null,
        "output: mean R-R instead of RMSSD",
        corpus_median(&corpus, |rr| mean(&rr.iter().map(|&v| v as f64).collect::<Vec<_>>())),
    );
    let shuf = mapped(&corpus, shuffled);
    tb.add(
        Kind::Null,
        "input: deterministic shuffle within each stretch",
        corpus_median(&shuf, |rr| shadow_rmssd_gap_aware(rr, RR_BASE)),
    );

    let rev = mapped(&corpus, |rr| rr.iter().rev().copied().collect());
    tb.add(
        Kind::Structural,
        "input: every stretch reversed",
        corpus_median(&rev, |rr| shadow_rmssd_gap_aware(rr, RR_BASE)),
    );
    // Saturating, not wrapping: one of the 500 real stretches carries a 65535 ms sentinel beat, and
    // `v + 50` on it panicked the whole binary before this table's first row printed. The beat is
    // outside RR_MAX_MS = 2000 either way, so the range filter drops it before and after the offset.
    let off = mapped(&corpus, |rr| rr.iter().map(|&v| v.saturating_add(50)).collect());
    tb.add(
        Kind::Structural,
        "input: +50 ms constant offset",
        corpus_median(&off, |rr| shadow_rmssd_gap_aware(rr, RR_BASE)),
    );
    let short = mapped(&corpus, |rr| rr[..rr.len() * 9 / 10].to_vec());
    tb.add(
        Kind::Structural,
        "input: last 10% of every stretch dropped",
        corpus_median(&short, |rr| shadow_rmssd_gap_aware(rr, RR_BASE)),
    );
    tb.add(
        Kind::Structural,
        "cohort: only the first half of the corpus",
        corpus_median(&corpus[..250], |rr| shadow_rmssd_gap_aware(rr, RR_BASE)),
    );

    let mut p;
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        p = RR_BASE;
        p.rr_min = (300.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MIN_MS 300 -> {} ({label})", p.rr_min),
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, p)),
        );
        p = RR_BASE;
        p.rr_max = (2000.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MAX_MS 2000 -> {} ({label})", p.rr_max),
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, p)),
        );
        p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(
            Kind::Param,
            format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh),
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, p)),
        );
    }
    p = RR_BASE;
    p.ect_thresh = 0.20 * 1.005;
    tb.add(
        Kind::Param,
        "param: ECTOPIC_THRESHOLD 0.20 -> 0.201 (+0.5%)",
        corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, p)),
    );
    for r in [1usize, 3] {
        p = RR_BASE;
        p.ect_radius = r;
        tb.add(
            Kind::Param,
            format!("param: ECTOPIC_WINDOW_RADIUS 2 -> {r}"),
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, p)),
        );
    }
    tb.note(format!(
        "RR_MIN_MS {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.rr_min = (300.0 * f).round() as u16;
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, q))
        })
    ));
    tb.note(format!(
        "RR_MAX_MS {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.rr_max = (2000.0 * f).round() as u16;
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, q))
        })
    ));
    tb.note(format!(
        "ECTOPIC_THRESHOLD {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.ect_thresh = 0.20 * f;
            corpus_median(&corpus, |rr| shadow_rmssd_gap_aware(rr, q))
        })
    ));
    report(&tb, t);
}

fn m18b_analyze_raw_real(t: &mut Tally) {
    let gate = Gate {
        label: "hrv_real_rr ALL row: median analyze_raw SDNN over 500 stretches",
        source: "crates/physio-algo/tests/hrv_real_rr.rs (ALL, sdnn)",
        target: 42.945050875993616,
        tol: 1e-9,
    };
    let corpus = real_corpus();
    let mut tb = Table::new("HRV SDNN / pNN50 over real recorded R-R", gate);
    let shipped = corpus_median(&corpus, |rr| nan_if_none(HrvReadiness::analyze_raw(rr, None).sdnn));
    let base = corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, RR_BASE));
    assert!((shipped - base).abs() < 1e-12, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(
        Kind::Null,
        "output: mean NN instead of SDNN",
        corpus_median(&corpus, |rr| mean(&rr.iter().map(|&v| v as f64).collect::<Vec<_>>())),
    );
    let shuf = mapped(&corpus, shuffled);
    tb.add(
        Kind::Null,
        "input: deterministic shuffle within each stretch",
        corpus_median(&shuf, |rr| shadow_analyze_sdnn(rr, 20, None, RR_BASE)),
    );
    let rev = mapped(&corpus, |rr| rr.iter().rev().copied().collect());
    tb.add(
        Kind::Structural,
        "input: every stretch reversed",
        corpus_median(&rev, |rr| shadow_analyze_sdnn(rr, 20, None, RR_BASE)),
    );
    // Saturating for the same reason as the RMSSD table's +50 arm: the corpus holds a 65535 ms beat.
    let off = mapped(&corpus, |rr| rr.iter().map(|&v| v.saturating_add(100)).collect());
    tb.add(
        Kind::Structural,
        "input: +100 ms constant offset",
        corpus_median(&off, |rr| shadow_analyze_sdnn(rr, 20, None, RR_BASE)),
    );
    let short = mapped(&corpus, |rr| rr[..rr.len() * 9 / 10].to_vec());
    tb.add(
        Kind::Structural,
        "input: last 10% of every stretch dropped",
        corpus_median(&short, |rr| shadow_analyze_sdnn(rr, 20, None, RR_BASE)),
    );

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        tb.add(
            Kind::Param,
            format!("param: MIN_BEATS 20 -> {} ({label})", scale_u(20, f)),
            corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, scale_u(20, f), None, RR_BASE)),
        );
        let mut p = RR_BASE;
        p.rr_min = (300.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MIN_MS 300 -> {} ({label})", p.rr_min),
            corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, p)),
        );
        p = RR_BASE;
        p.rr_max = (2000.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MAX_MS 2000 -> {} ({label})", p.rr_max),
            corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, p)),
        );
        p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(
            Kind::Param,
            format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh),
            corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, p)),
        );
    }
    let mut p = RR_BASE;
    p.ect_thresh = 0.20 * 1.005;
    tb.add(
        Kind::Param,
        "param: ECTOPIC_THRESHOLD 0.20 -> 0.201 (+0.5%)",
        corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, p)),
    );

    // pNN50 rides the same cleaning chain; its own shipped ALL row is 11.764705882.
    tb.note(format!(
        "corpus median pNN50 {:.9} against the shipped hrv_real_rr ALL row 11.764705882",
        corpus_median(&corpus, |rr| shadow_analyze_pnn50(rr, 20, RR_BASE))
    ));
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut q = RR_BASE;
        q.ect_thresh = 0.20 * f;
        tb.note(format!(
            "pNN50 under ECTOPIC_THRESHOLD ({label}): {:.9}",
            corpus_median(&corpus, |rr| shadow_analyze_pnn50(rr, 20, q))
        ));
    }
    tb.note(format!(
        "ECTOPIC_THRESHOLD {}",
        break_at(&tb.gate, &|f| {
            let mut q = RR_BASE;
            q.ect_thresh = 0.20 * f;
            corpus_median(&corpus, |rr| shadow_analyze_sdnn(rr, 20, None, q))
        })
    ));
    report(&tb, t);
}

fn m18c_windowed_avg_real(t: &mut Tally) {
    let gate = Gate {
        label: "hrv_real_rr ALL row: session avgHrv over the whole-corpus timeline",
        source: "crates/physio-algo/tests/hrv_real_rr.rs (ALL, avg_hrv)",
        target: 46.74367726233443,
        tol: 1e-9,
    };
    let corpus = real_corpus();
    let flat: Vec<u16> = corpus.iter().flat_map(|rr| rr.iter().copied()).collect();
    let beats = corpus_stamped(&flat);
    let (start, end) = (beats.first().unwrap().0, beats.last().unwrap().0);
    let mut tb = Table::new("HRV windowed average (session avgHrv) over real recorded R-R", gate);
    let shipped = nan_if_none(HrvReadiness::windowed_avg_hrv(start, end, &beats));
    let base = shadow_windowed_avg(start, end, &beats, 300, 2000, RR_BASE);
    assert!((shipped - base).abs() < 1e-12, "shadow {base} != shipped {shipped}");
    tb.add(Kind::Baseline, "baseline (unmutated)", base);

    tb.add(Kind::Null, "output: constant 0", 0.0);
    tb.add(
        Kind::Null,
        "output: mean R-R over the timeline",
        mean(&flat.iter().map(|&v| v as f64).collect::<Vec<_>>()),
    );
    let rev: Vec<(u32, u16)> =
        beats.iter().map(|&(t, _)| t).zip(beats.iter().rev().map(|&(_, v)| v)).collect();
    tb.add(
        Kind::Structural,
        "input: values reversed against their timestamps",
        shadow_windowed_avg(start, end, &rev, 300, 2000, RR_BASE),
    );
    // `beats` is the whole real corpus flattened, so it carries the same 65535 ms sentinel.
    let off: Vec<(u32, u16)> = beats.iter().map(|&(t, v)| (t, v.saturating_add(50))).collect();
    tb.add(
        Kind::Structural,
        "input: +50 ms constant offset",
        shadow_windowed_avg(start, end, &off, 300, 2000, RR_BASE),
    );
    tb.add(
        Kind::Structural,
        "window: only the first half of the timeline",
        shadow_windowed_avg(start, (start + end) / 2, &beats, 300, 2000, RR_BASE),
    );

    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90), ("+0.5%", 1.005)] {
        let w = (300.0 * f).round() as u64;
        tb.add(
            Kind::Param,
            format!("param: HRV_WINDOW_SECS 300 -> {w} ({label})"),
            shadow_windowed_avg(start, end, &beats, w, 2000, RR_BASE),
        );
    }
    for (label, f) in [("+10%", 1.10f64), ("-10%", 0.90)] {
        let mut p = RR_BASE;
        p.ect_thresh = 0.20 * f;
        tb.add(
            Kind::Param,
            format!("param: ECTOPIC_THRESHOLD 0.20 -> {:.4} ({label})", p.ect_thresh),
            shadow_windowed_avg(start, end, &beats, 300, 2000, p),
        );
        p = RR_BASE;
        p.rr_min = (300.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MIN_MS 300 -> {} ({label})", p.rr_min),
            shadow_windowed_avg(start, end, &beats, 300, 2000, p),
        );
        p = RR_BASE;
        p.rr_max = (2000.0 * f).round() as u16;
        tb.add(
            Kind::Param,
            format!("param: RR_MAX_MS 2000 -> {} ({label})", p.rr_max),
            shadow_windowed_avg(start, end, &beats, 300, 2000, p),
        );
    }
    tb.note(format!(
        "HRV_WINDOW_SECS {}",
        break_at(&tb.gate, &|f| shadow_windowed_avg(
            start,
            end,
            &beats,
            (300.0 * f).round() as u64,
            2000,
            RR_BASE
        ))
    ));
    report(&tb, t);
}

// ── Cohort guard ───────────────────────────────────────────────────────────────────────────────

/// The Vitals cohort this control replicates, as `(file, its `#[test]` count)`. Counted from the
/// source so a gate added or removed there fails loudly here instead of silently shrinking or
/// outgrowing what the control measures.
const VITALS_COHORT: &[(&str, &str, usize)] = &[
    ("src/respiratory_rate.rs", include_str!("../src/respiratory_rate.rs"), 10),
    ("src/hrv_freq.rs", include_str!("../src/hrv_freq.rs"), 7),
    ("src/spo2.rs", include_str!("../src/spo2.rs"), 11),
    ("src/worn.rs", include_str!("../src/worn.rs"), 6),
    ("tests/resting_hr_parity.rs", include_str!("resting_hr_parity.rs"), 19),
    ("tests/ppg_hr_real.rs", include_str!("ppg_hr_real.rs"), 5),
];

/// Reachable from a clean checkout on purpose: the control below is `#[ignore]`d, so without this the
/// cohort could drift for a whole release without anything saying so.
#[test]
fn the_vitals_cohort_is_the_size_this_control_was_written_against() {
    let drift: Vec<String> = VITALS_COHORT
        .iter()
        .filter_map(|&(name, src, want)| {
            let have = src.matches("#[test]").count();
            (have != want).then(|| format!("{name}: {have} tests, control written against {want}"))
        })
        .collect();
    assert!(drift.is_empty(), "the vitals gate cohort moved - re-measure this control: {drift:?}");
}

// ─────────────────────────── entry point ───────────────────────────

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("HRV RMSSD (gap-aware) over real recorded R-R", "output: constant 0", 15.9), // M18-INJECTED
    ("HRV RMSSD (gap-aware) over real recorded R-R", "output: mean R-R instead of RMSSD", 297.0), // M18-INJECTED
    ("HRV RMSSD (gap-aware) over real recorded R-R", "input: deterministic shuffle within each stretch", 9.48), // M18-INJECTED
    ("HRV RMSSD (gap-aware) over real recorded R-R", "input: +50 ms constant offset", 0.754), // M18-INJECTED
    ("HRV RMSSD (gap-aware) over real recorded R-R", "input: last 10% of every stretch dropped", 0.00411), // M18-INJECTED
    ("HRV RMSSD (gap-aware) over real recorded R-R", "cohort: only the first half of the corpus", 7.48), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "output: constant 0", 19.3), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "output: mean NN instead of SDNN", 294.0), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "input: deterministic shuffle within each stretch", 3.72), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "input: +100 ms constant offset", 0.721), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "input: last 10% of every stretch dropped", 0.518), // M18-INJECTED
    ("HRV windowed average (session avgHrv) over real recorded R-R", "output: constant 0", 21.0), // M18-INJECTED
    ("HRV windowed average (session avgHrv) over real recorded R-R", "output: mean R-R over the timeline", 308.0), // M18-INJECTED
    ("HRV windowed average (session avgHrv) over real recorded R-R", "input: values reversed against their timestamps", 0.29), // M18-INJECTED
    ("HRV windowed average (session avgHrv) over real recorded R-R", "input: +50 ms constant offset", 0.095), // M18-INJECTED
    ("HRV windowed average (session avgHrv) over real recorded R-R", "window: only the first half of the timeline", 4.69), // M18-INJECTED
    ("HRV RMSSD (gap-aware, artifact-corrected)", "output: constant 0", 2.27),
    ("HRV RMSSD (gap-aware, artifact-corrected)", "output: mean R-R instead of RMSSD", 391.0),
    ("HRV RMSSD (gap-aware, artifact-corrected)", "input: deterministic shuffle", 0.184),
    ("HRV RMSSD (gap-aware, artifact-corrected)", "input: drop last beat (-14%)", 0.0223),
    ("HRV RMSSD (gap-aware, artifact-corrected)", "input: shifted one beat (rotate left)", 0.525),
    ("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", "output: constant 0", 4.5),
    ("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", "output: mean R-R", 360.0),
    ("HRV windowed average (session avgHrv)", "output: constant 0", 4.05),
    ("HRV windowed average (session avgHrv)", "output: mean R-R in the bucket", 360.0),
    ("HRV windowed average (session avgHrv)", "input: deterministic shuffle", 1.56),
    ("HRV windowed average (session avgHrv)", "input: drop last beat (-20%)", 0.159),
    ("HRV deep-window average", "output: constant 0", 7.11),
    ("HRV deep-window average", "output: all-bucket mean (deep filter ignored)", 1.52),
    ("HRV deep-window average", "span: deep window swapped to the other bucket", 3.05),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "output: constant 0", 2.29),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "output: mean NN instead of SDNN", 359.0),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "input: drop last 10% (2 beats)", 0.00455),
    ("Rolling RMSSD (300 s trailing window series)", "output: constant 0", 4.5),
    ("Rolling RMSSD (300 s trailing window series)", "output: mean R-R (805)", 357.0),
    ("Rolling RMSSD (300 s trailing window series)", "input: deterministic shuffle of the VALUES", 0.67),
    ("Rolling RMSSD (300 s trailing window series)", "input: alternation flattened to a constant", 4.5),
    ("Rolling RMSSD emission timestamps", "output: constant 0", 49.0),
    ("Rolling RMSSD emission timestamps", "input: timestamps shifted +1000 s", 450.0),
    ("Rolling RMSSD emission timestamps", "input: drop last beat", 0.45),
    ("R-R coverage / duplicate beats / overlapping reports", "output: constant 1.0 (always plausible)", 0.112),
    ("R-R coverage / duplicate beats / overlapping reports", "output: constant 0", 0.562),
    ("R-R coverage / duplicate beats / overlapping reports", "input: drop the last beat", 0.0374),
    ("R-R coverage / duplicate beats / overlapping reports", "input: every beat stored twice", 0.787),
    ("Overlapping-report count (seam detection)", "output: every report flagged overlapping", 5.4),
    ("Overlapping-report count (seam detection)", "input: the overrunning stream instead", 4.95),
    ("Overlapping-report count (seam detection)", "input: every beat re-reported in place", 4.05),
    ("Frequency-domain HRV (LF, HF, LF/HF, total power)", "output: constant HF = 1.0 ms^2", 0.0335),
    ("Frequency-domain HRV (LF, HF, LF/HF, total power)", "output: HF = the series variance", 359.0),
    ("Frequency-domain HRV (LF, HF, LF/HF, total power)", "input: reversed tachogram", 0.000456),
    ("Frequency-domain HRV (LF, HF, LF/HF, total power)", "input: deterministic shuffle", 0.29),
    ("Frequency-domain HRV (LF, HF, LF/HF, total power)", "input: modulation halved (40 -> 20 ms)", 0.328),
    ("Frequency-domain HRV presence + ordering", "output: nothing at all", 0.45),
    ("Frequency-domain HRV presence + ordering", "output: LF and HF swapped", 0.45),
    ("Frequency-domain HRV presence + ordering", "input: span halved to 150 s (under the LF gate)", 0.45),
    ("Sleeping respiratory rate (breaths/min via RSA)", "output: constant 13.0 bpm", 0.9),
    ("Sleeping respiratory rate (breaths/min via RSA)", "output: midpoint of the plausible band (16.5)", 0.675),
    ("Sleeping respiratory rate (breaths/min via RSA)", "output: constant 0", 6.75),
    ("Sleeping respiratory rate (breaths/min via RSA)", "input: deterministic shuffle of the tachogram", 0.289),
    ("Sleeping respiratory rate (breaths/min via RSA)", "input: breathing rate doubled (0.25 -> 0.50 Hz)", 0.675),
    ("Resting HR (session lowest-sustained floor)", "output: constant 60 (the night level)", 5.4),
    ("Resting HR (session lowest-sustained floor)", "output: whole-segment mean", 4.5),
    ("Resting HR (session lowest-sustained floor)", "input: timestamps shifted +150 s (half a window)", 2.25),
    ("Resting HR (session lowest-sustained floor)", "input: +5 bpm constant offset", 2.25),
    ("Resting HR (session lowest-sustained floor)", "input: drop the low window entirely", 5.4),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "output: constant 85.0 (the R=1 curve midpoint)", 5.62),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "output: midpoint of the clamp band ((70+100)/2)", 5.62),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "input: red and IR swapped", 12.3),
    ("SpO2 rolling multi-night reading", "output: the ANCHOR itself, rounded (96.5 -> 97)", 1.35),
    ("SpO2 rolling multi-night reading", "output: constant 0", 42.3),
    ("SpO2 rolling multi-night reading", "output: the 7-night median with no anchoring", 1.35),
    ("SpO2 rolling multi-night reading", "input: the 30-night order reversed", 1.35),
    ("SpO2 rolling multi-night reading", "input: the spread flattened to its own median", 1.35),
    ("SpO2 rolling multi-night reading", "input: only the most recent night kept", 1.35),
    ("Vitality (0-100) and Body Age (years)", "output: constant 0", 18.0),
    ("Vitality (0-100) and Body Age (years)", "output: midpoint of the clamp band (55)", 6.75),
    ("Vitality (0-100) and Body Age (years)", "input: resting HR 65 -> 80", 0.73),
    ("Vitality (0-100) and Body Age (years)", "input: VO2max and its expectation pulled apart", 1.8),
    ("Vitality SRI driver (the one calibrated coefficient)", "output: constant 0 (no hazard from irregularity)", 0.191),
    ("Vitality SRI driver (the one calibrated coefficient)", "output: constant ln(1.53) regardless of SRI", 3.48e-05),
    ("Vitality SRI driver (the one calibrated coefficient)", "output: sign flipped", 0.382),
    ("Vitality SRI driver (the one calibrated coefficient)", "input: SRI 41 -> 75 (the other published point)", 0.238),
    ("Vitality SRI driver (the one calibrated coefficient)", "input: SRI 41 -> the median", 0.191),
    ("Vitality RMSSD age-norm anchors", "output: constant 33 (the mid anchor)", 6.3),
    ("Vitality RMSSD age-norm anchors", "output: constant 0", 21.1),
    ("Vitality RMSSD age-norm anchors", "input: age 20 -> 30 (next anchor)", 3.15),
    ("Vitality RMSSD age-norm anchors", "input: age 20 -> 90 (past the last anchor)", 12.1),
    ("CosinorAge / Rhythm Age (circadian biological age)", "output: chronological age (40.0)", 0.182),
    ("CosinorAge / Rhythm Age (circadian biological age)", "output: constant 0", 18.1),
    ("CosinorAge / Rhythm Age (circadian biological age)", "coeffs: the female set instead of generic", 0.328),
    ("CosinorAge / Rhythm Age (circadian biological age)", "coeffs: the male set instead of generic", 1.35),
    ("CosinorAge / Rhythm Age (circadian biological age)", "input: acrophase sign flipped (-1.5 -> +1.5)", 0.2),
    ("CosinorAge / Rhythm Age (circadian biological age)", "input: MESOR and amplitude swapped", 0.247),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "output: constant 12.0 h (mid-day)", 0.9),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "input: profile rolled one hour later", 0.45),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "input: profile mirrored against the clock", 2.25),
    ("Body-clock phase estimate", "input: acrophase moved 16 -> 17 h", 0.45),
    ("HR anomaly watch (sustained elevated at-rest HR)", "input: elevated level 120 -> 110", 4.5),
    ("Wear state (worn / not worn / unknown)", "output: always Worn", 0.45),
    ("Wear state (worn / not worn / unknown)", "output: always Unknown", 0.45),
    ("Wear state (worn / not worn / unknown)", "rule: dead-optical-baseline evidence ignored", 0.45),
    ("Weighted trendline significance", "output: constant 0 (never significant)", 0.069),
    ("Weighted trendline significance", "output: constant 1 (always significant)", 0.38),
    ("Weighted trendline significance", "input: scatter removed ([0,0.5,1])", 0.38),
    ("Trend minimum-span rule", "output: constant 0 (any span accepted)", 1.35),
    ("Trend minimum-span rule", "output: the window itself (7.0)", 1.8),
    ("Trend minimum-span rule", "input: window 7 -> 31 days", 3.29),
    ("Trend minimum-span rule", "input: window 7 -> 366 days", 53.5),
    ("Calibration unlock schedule (nights per metric)", "output: every metric unlocked from night 0", 0.45),
    ("Calibration unlock schedule (nights per metric)", "output: nothing ever unlocks", 0.45),
    ("Calibration unlock schedule (nights per metric)", "schedule: CALORIES unlock and full swapped", 0.45),
    ("Calibration unlock schedule (nights per metric)", "schedule: SKIN_TEMP given BLOOD_OXYGEN's schedule", 0.45),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("HRV RMSSD (gap-aware) over real recorded R-R", "input: every stretch reversed", "measured delta is exactly zero: this mutation does not move the number"), // M18-INJECTED
    ("HRV SDNN / pNN50 over real recorded R-R", "input: every stretch reversed", "measured delta is exactly zero: this mutation does not move the number"), // M18-INJECTED
    ("HRV RMSSD (gap-aware, artifact-corrected)", "input: reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV RMSSD (gap-aware, artifact-corrected)", "input: +50 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", "input: reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", "input: +100 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV RMSSD run variant (MAX_BEAT_DELTA_MS)", "input: drop last beat", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV windowed average (session avgHrv)", "input: reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV windowed average (session avgHrv)", "input: beats shifted +300 s (out of window)", "the arm yields no number, so it has no distance from the baseline"),
    ("HRV windowed average (session avgHrv)", "input: +50 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV deep-window average", "span: no deep spans at all", "the arm yields no number, so it has no distance from the baseline"),
    ("HRV deep-window average", "input: +50 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "input: deterministic shuffle", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "input: reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("HRV SDNN / pNN50 / analyze_raw (spot reading)", "input: +100 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rolling RMSSD (300 s trailing window series)", "output: midpoint of the tolerance band (10.0)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rolling RMSSD (300 s trailing window series)", "input: values reversed against their timestamps", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rolling RMSSD (300 s trailing window series)", "input: +100 ms constant offset", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rolling RMSSD (300 s trailing window series)", "input: drop last 10% (3 beats)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rolling RMSSD emission timestamps", "output: no points at all", "the arm yields no number, so it has no distance from the baseline"),
    ("R-R coverage / duplicate beats / overlapping reports", "input: timestamps reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("R-R coverage / duplicate beats / overlapping reports", "input: timestamps shifted +1000 s", "measured delta is exactly zero: this mutation does not move the number"),
    ("Frequency-domain HRV presence + ordering", "output: constant (lf 0.5, hf 1.0, lfhf 0.5, total 1.5)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleeping respiratory rate (breaths/min via RSA)", "input: tachogram reversed against its clock", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleeping respiratory rate (breaths/min via RSA)", "input: RSA amplitude halved (40 -> 20 ms)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleeping respiratory rate (breaths/min via RSA)", "input: drop the last 10% of beats", "measured delta is exactly zero: this mutation does not move the number"),
    ("Resting HR (session lowest-sustained floor)", "output: global minimum sample (no windowing)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Resting HR (session lowest-sustained floor)", "input: bpm reversed against the clock", "measured delta is exactly zero: this mutation does not move the number"),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "output: constant 97.5", "measured delta is exactly zero: this mutation does not move the number"),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "input: both channels reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "input: both DC levels doubled", "measured delta is exactly zero: this mutation does not move the number"),
    ("SpO2 from paired red/IR (ratio-of-ratios)", "input: red AC flattened to zero", "the arm yields no number, so it has no distance from the baseline"),
    ("SpO2 rolling multi-night reading", "input: no nights at all", "the arm yields no number, so it has no distance from the baseline"),
    ("Vitality (0-100) and Body Age (years)", "output: chronological age, drivers ignored", "measured delta is exactly zero: this mutation does not move the number"),
    ("Vitality (0-100) and Body Age (years)", "input: half the drivers dropped (3 left)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Vitality (0-100) and Body Age (years)", "input: only 2 drivers left", "the arm yields no number, so it has no distance from the baseline"),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "output: constant 14.0 h regardless of input", "measured delta is exactly zero: this mutation does not move the number"),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "output: no fit at all", "the arm yields no number, so it has no distance from the baseline"),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "input: amplitude halved (40 -> 20)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Circadian cosinor (MESOR, amplitude, acrophase)", "input: activity scaled x1000", "measured delta is exactly zero: this mutation does not move the number"),
    ("Body-clock phase estimate", "output: never Solid", "the arm yields no number, so it has no distance from the baseline"),
    ("Body-clock phase estimate", "output: constant 16.0 h, always Solid", "measured delta is exactly zero: this mutation does not move the number"),
    ("Body-clock phase estimate", "input: days_observed 14 -> 13 (Wide, not Solid)", "the arm yields no number, so it has no distance from the baseline"),
    ("Body-clock phase estimate", "input: rhythm flattened (rel amp 0.01)", "the arm yields no number, so it has no distance from the baseline"),
    ("HR anomaly watch (sustained elevated at-rest HR)", "output: always Normal", "the arm yields no number, so it has no distance from the baseline"),
    ("HR anomaly watch (sustained elevated at-rest HR)", "output: constant peak 120", "measured delta is exactly zero: this mutation does not move the number"),
    ("HR anomaly watch (sustained elevated at-rest HR)", "input: elevated run shortened to 200 s", "the arm yields no number, so it has no distance from the baseline"),
    ("HR anomaly watch (sustained elevated at-rest HR)", "input: record order reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("HR anomaly watch (sustained elevated at-rest HR)", "input: the elevated stretch marked as running", "the arm yields no number, so it has no distance from the baseline"),
    ("Weighted trendline significance", "input: values reversed ([1,2,0])", "measured delta is exactly zero: this mutation does not move the number"),
    ("Weighted trendline significance", "input: days shifted +100", "measured delta is exactly zero: this mutation does not move the number"),
    ("Weighted trendline significance", "input: values scaled x2", "measured delta is exactly zero: this mutation does not move the number"),
];

/// Assert one metric's floors, and require every NULL/STRUCTURAL arm to be classified.
fn enforce_floors(metric: &str, base: f64, probes: &[(&str, f64)]) {
    let (mut asserted, mut waived) = (0usize, 0usize);
    let mut breached: Vec<String> = Vec::new();
    let mut unclassified: Vec<&str> = Vec::new();
    for &(arm, value) in probes {
        let floor = FLOORS.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        let waiver = NO_FLOOR.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        match (floor, waiver) {
            (Some(_), Some(_)) => breached.push(format!("'{arm}' carries both a floor and a waiver")),
            (Some(d), None) => {
                asserted += 1;
                let moved = (value - base).abs();
                if moved.is_nan() || moved < d {
                    breached.push(format!("'{arm}' moved {moved} against a floor of {d}"));
                }
            }
            (None, Some(w)) => {
                waived += 1;
                println!("   no floor: {arm} — {w}");
            }
            (None, None) => unclassified.push(arm),
        }
    }
    let orphans: Vec<&str> = FLOORS
        .iter()
        .filter(|(m, _, _)| *m == metric)
        .map(|t| t.1)
        .chain(NO_FLOOR.iter().filter(|(m, _, _)| *m == metric).map(|t| t.1))
        .filter(|a| !probes.iter().any(|(p, _)| *p == *a))
        .collect();
    println!("   floors: {asserted} asserted, {waived} un-floorable");
    assert!(
        unclassified.is_empty(),
        "{metric}: probe arms carry neither a floor nor a waiver — classify them: {unclassified:?}"
    );
    assert!(orphans.is_empty(), "{metric}: floor rows match no arm — stale or misspelt: {orphans:?}");
    assert!(breached.is_empty(), "{metric}: SENSITIVITY FLOOR BREACHED — {}", breached.join(" | "));
}

/// Ignored by convention so the control never enters CI. Run it with
/// `cargo test --release -p physio-algo --test sensitivity_vitals -- --ignored --nocapture`.
#[test]
#[ignore]
fn sensitivity_vitals() {
    let mut t = Tally::default();
    println!("negative control: vitals family. Wellness estimates only, never medical or diagnostic.");
    println!("Every PARAM arm runs a shadow of the algorithm; each shadow is asserted equal to the");
    println!("shipped function at baseline parameters before any arm is measured.");

    m1_rmssd_gap_aware(&mut t);
    m1b_rmssd_artifact_filter(&mut t);
    m2_windowed_avg(&mut t);
    m2b_windowed_avg_deep(&mut t);
    m3_analyze_raw(&mut t);
    m18_rmssd_gap_aware_real(&mut t);
    m18b_analyze_raw_real(&mut t);
    m18c_windowed_avg_real(&mut t);
    m4_rolling_rmssd(&mut t);
    m4b_rolling_timestamps(&mut t);
    m5_rr_coverage(&mut t);
    m5b_overlapping_reports(&mut t);
    m6_freq_band_power(&mut t);
    m6b_freq_presence(&mut t);
    m7_respiratory_rate(&mut t);
    m8_resting_hr(&mut t);
    m9_spo2_paired(&mut t);
    m10_spo2_rolling(&mut t);
    m11_vitality(&mut t);
    m11b_vitality_sri(&mut t);
    m11c_rmssd_norm(&mut t);
    m12_cosinor_age(&mut t);
    m13_circadian(&mut t);
    m13b_body_clock_phase(&mut t);
    m14_hr_watch(&mut t);
    m15_worn(&mut t);
    m16_trendline(&mut t);
    m16b_trend_min_span(&mut t);
    m17_calibration(&mut t);

    println!("\n================ VITALS FAMILY SUMMARY ================");
    println!("caught {}, missed {}", t.caught, t.missed);
    match t.floor {
        Some(f) => println!("smallest delta any vitals gate catches (sensitivity floor): {f:.12}"),
        None => println!("no vitals gate caught a finite-delta arm"),
    }
    println!("CRITICAL findings: {}", t.criticals.len());
    for c in &t.criticals {
        println!("  - {c}");
    }
    println!("======================================================");
}
