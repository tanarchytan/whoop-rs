//! In-bed span detection — the gravity-stillness spine that carves candidate sleep runs from a night's
//! streams, before staging. A rolling stillness fraction over per-sample gravity deltas classifies each
//! sample sleep/active, latched through [`DetectParams`] so opening a run takes more stillness than
//! holding one; runs are built, short runs merged, a short HR-vouched mid-sleep wake absorbed, and
//! (only when gravity is sparse) HR-vouched data gaps bridged. So one night reaches staging as one
//! window. Pure and deterministic. [`detect_sessions`] is the module's exported entry and the one
//! [`super::analyze`] runs; [`detect_sessions_with`] takes the thresholds, so a caller can score a
//! window against the spine that wrote it rather than the current one.

use std::collections::HashMap;

use super::common::median;
use super::input::{AccelSample, HrSample};
use super::{SleepStage, StageSegment};
use crate::resting_hr;

const GRAVITY_STILL_THRESHOLD_G: f64 = 0.01;
const STILL_WINDOW_MIN: i64 = 15;
pub const MAX_GAP_MIN: i64 = 20;
const MERGE_MIN: i64 = 15;
const DEFAULT_INTERVAL_S: f64 = 60.0;
const MIN_WINDOW_SAMPLES: i64 = 3;
const HR_SLEEP_BASELINE_MULT: f64 = 1.05;
pub const SPARSE_GRAVITY_SPAN_FRAC: f64 = 0.5;
const HR_SLEEP_BAND_MULT: f64 = HR_SLEEP_BASELINE_MULT;
const SPARSE_BRIDGE_GAP_MIN: i64 = 90;

const QUIESCENT_POSTURE_VAR_G2: f64 = 0.05;
const QUIESCENT_STABLE_FRAC: f64 = 0.90;
const QUIESCENT_MIN_STABLE_MINUTES: i64 = 20;
const QUIESCENT_HR_SLEEP_MULT: f64 = 1.30;
const HR_REFINE_MIN_SAMPLES: usize = 30;
pub const DAYTIME_BAND_START_HOUR: i64 = 11;
pub const DAYTIME_BAND_END_HOUR: i64 = 20;
const SECONDS_PER_DAY: i64 = 86_400;
const DAYTIME_MIN_SLEEP_MIN: i64 = 90;
const DAYTIME_RESTING_HR_MULT: f64 = 0.95;
const MORNING_STILLNESS_WINDOW_MIN: i64 = 180;
const MORNING_REONSET_RESTING_HR_MULT: f64 = 0.90;
const BAND_STATE_ASLEEP: i32 = 2;
const MORNING_REONSET_BAND_ASLEEP_FRAC: f64 = 0.6;
const OFF_WRIST_HR_GAP_MIN: i64 = 20;
const MAX_OFF_WRIST_SLEEP_FRACTION: f64 = 0.5;
const HR_DENSE_SPACING_S: i64 = 600;
const MAX_MAIN_SLEEP_SPAN_S: i64 = 16 * 60 * 60;
const NIGHT_CONTINUATION_GAP_MIN: i64 = 90;

/// The detector's swept thresholds. `still_enter`/`still_exit` are the two fractions
/// [`classify_still`] switches a sleep run on and off at, latched so a run needs `still_enter` to open
/// and holds until the rolling fraction drops below `still_exit`; `min_sleep_min` is the duration a
/// candidate run must EXCEED to be kept; `wake_absorb_max_min` is the longest mid-sleep wake run
/// [`absorb_short_wake`] folds back into the night when HR vouches for it (0 = never), so a stir
/// reaches staging as one window rather than two half-nights normalised apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectParams {
    pub still_enter: f64,
    pub still_exit: f64,
    pub min_sleep_min: i64,
    pub wake_absorb_max_min: i64,
}

impl DetectParams {
    /// The thresholds [`detect_sessions`] runs, and so the window every new session is cut on.
    pub const SHIPPED: DetectParams =
        DetectParams { still_enter: 0.80, still_exit: 0.65, min_sleep_min: 60, wake_absorb_max_min: 45 };

    /// The single-threshold spine that wrote every already-stored session. Kept so a stored-versus-fresh
    /// comparison stays attributable to one detector after the enter/exit split.
    pub const PRE_HYSTERESIS: DetectParams =
        DetectParams { still_enter: 0.70, still_exit: 0.70, min_sleep_min: 60, wake_absorb_max_min: 0 };
}

impl Default for DetectParams {
    fn default() -> Self {
        DetectParams::SHIPPED
    }
}

/// A contiguous run of one class over `[start, end]` wall-clock unix seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Period {
    pub is_sleep: bool,
    pub start: i64,
    pub end: i64,
}

/// Per-sample movement proxy = L2 magnitude of the gravity change vs the previous sample; first = 0.
pub(super) fn gravity_deltas(grav: &[AccelSample]) -> Vec<f64> {
    let mut out = Vec::with_capacity(grav.len());
    for (i, r) in grav.iter().enumerate() {
        if i == 0 {
            out.push(0.0);
        } else {
            let p = &grav[i - 1];
            let (dx, dy, dz) = (p.x - r.x, p.y - r.y, p.z - r.z);
            out.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    out
}

fn window_size(times: &[i64]) -> i64 {
    let interval = crate::stats::median_gap_s(times, DEFAULT_INTERVAL_S);
    ((STILL_WINDOW_MIN * 60) as f64 / interval).trunc() as i64
}

fn window_size_clamped(times: &[i64]) -> i64 {
    window_size(times).max(MIN_WINDOW_SAMPLES)
}

/// Largest spacing between consecutive timestamps (s), no upper cap; 0.0 for `<2` samples.
fn largest_gap_s(times: &[i64]) -> f64 {
    let mut mx = 0.0;
    for w in times.windows(2) {
        let g = (w[1] - w[0]) as f64;
        if g > mx {
            mx = g;
        }
    }
    mx
}

/// True when gravity is too sparse to trust the gravity-only spine across gaps: its timespan covers
/// `< SPARSE_GRAVITY_SPAN_FRAC` of the HR timespan, or the largest gravity gap exceeds `MAX_GAP_MIN`.
pub(super) fn is_gravity_sparse(grav: &[AccelSample], hr: &[HrSample]) -> bool {
    if grav.len() < 2 || hr.len() < 2 {
        return false;
    }
    let hr_span = (hr[hr.len() - 1].ts - hr[0].ts) as f64;
    if hr_span <= 0.0 {
        return false;
    }
    let grav_span = (grav[grav.len() - 1].ts - grav[0].ts) as f64;
    if grav_span < SPARSE_GRAVITY_SPAN_FRAC * hr_span {
        return true;
    }
    let times: Vec<i64> = grav.iter().map(|g| g.ts).collect();
    largest_gap_s(&times) > (MAX_GAP_MIN * 60) as f64
}

/// True when HR stays in the sleep band (`<= baseline x HR_SLEEP_BAND_MULT`) across `(a, b]`; false with
/// no baseline or no HR in the interval (cannot vouch → treat as a real break).
fn hr_sleep_band_across(a: i64, b: i64, hr: &[HrSample], baseline: Option<f64>) -> bool {
    let Some(baseline) = baseline else { return false };
    let seg: Vec<&HrSample> = hr.iter().filter(|h| h.ts > a && h.ts <= b).collect();
    if seg.is_empty() {
        return false;
    }
    let mean = seg.iter().map(|h| h.bpm as f64).sum::<f64>() / seg.len() as f64;
    mean <= baseline * HR_SLEEP_BAND_MULT
}

/// Per-sample sleep flags from a rolling fraction of "still" samples (prefix-summed to O(n)), latched
/// through [`DetectParams`]: opening needs `still_enter`, closing needs a drop below `still_exit`.
pub(super) fn classify_still(grav: &[AccelSample], deltas: &[f64], p: &DetectParams) -> Vec<bool> {
    let n = grav.len();
    if n < 2 {
        return vec![false; n];
    }
    let times: Vec<i64> = grav.iter().map(|g| g.ts).collect();
    let half = (window_size_clamped(&times) / 2) as usize;
    let mut still_prefix = vec![0i64; n + 1];
    for i in 0..n {
        still_prefix[i + 1] = still_prefix[i] + i64::from(deltas[i] < GRAVITY_STILL_THRESHOLD_G);
    }
    let mut flags = Vec::with_capacity(n);
    let mut in_run = false;
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let frac = (still_prefix[hi] - still_prefix[lo]) as f64 / (hi - lo) as f64;
        in_run = if in_run { frac >= p.still_exit } else { frac >= p.still_enter };
        flags.push(in_run);
    }
    flags
}

/// Collapse per-sample flags into contiguous runs, breaking on class change or a `> MAX_GAP_MIN` gap.
/// When `sparse`, a pure gravity gap does not close a sleep run while HR stays in the sleep band across it.
pub(super) fn build_runs(
    grav: &[AccelSample],
    flags: &[bool],
    sparse: bool,
    hr: &[HrSample],
    baseline: Option<f64>,
) -> Vec<Period> {
    let n = grav.len();
    if n == 0 {
        return Vec::new();
    }
    let times: Vec<i64> = grav.iter().map(|g| g.ts).collect();
    let max_gap_s = MAX_GAP_MIN * 60;
    let mut periods = Vec::new();
    let mut run_start = 0usize;
    for i in 1..=n {
        let close = if i == n {
            true
        } else {
            let class_changed = flags[i] != flags[run_start];
            let mut gap_exceeded = (times[i] - times[i - 1]) > max_gap_s;
            if sparse
                && gap_exceeded
                && !class_changed
                && flags[run_start]
                && hr_sleep_band_across(times[i - 1], times[i], hr, baseline)
            {
                gap_exceeded = false;
            }
            class_changed || gap_exceeded
        };
        if close {
            periods.push(Period {
                is_sleep: flags[run_start],
                start: times[run_start],
                end: times[i - 1],
            });
            run_start = i;
        }
    }
    periods
}

/// Absorb runs shorter than `MERGE_MIN` minutes into their neighbours.
pub(super) fn merge_periods(periods: &[Period]) -> Vec<Period> {
    if periods.is_empty() {
        return Vec::new();
    }
    let threshold = MERGE_MIN * 60;
    let mut pending: Vec<Period> = periods.to_vec();
    let mut merged: Vec<Period> = Vec::new();
    let mut i = 0usize;
    while i < pending.len() {
        let current = pending[i];
        if (current.end - current.start) > threshold {
            merged.push(current);
            i += 1;
            continue;
        }
        let has_prev = i > 0 && !merged.is_empty();
        let has_next = i + 1 < pending.len();
        let bridges_same = has_prev && has_next && pending[i - 1].is_sleep == pending[i + 1].is_sleep;
        if bridges_same {
            let prev = merged.pop().unwrap();
            merged.push(Period { is_sleep: prev.is_sleep, start: prev.start, end: pending[i + 1].end });
            i += 2;
        } else if has_next {
            pending[i + 1] = Period { is_sleep: pending[i + 1].is_sleep, start: current.start, end: pending[i + 1].end };
            i += 1;
        } else if has_prev {
            let prev = merged.pop().unwrap();
            merged.push(Period { is_sleep: prev.is_sleep, start: prev.start, end: current.end });
            i += 1;
        } else {
            i += 1;
        }
    }
    merged
}

/// Merge two sleep runs left ADJACENT in the list by a data gap (never by a wake run) when the
/// intervening HR stays in the sleep band. `max_gap_min <= 0` is a no-op.
pub(super) fn bridge_sleep_gap(periods: &[Period], max_gap_min: i64, hr: &[HrSample], baseline: Option<f64>) -> Vec<Period> {
    if max_gap_min <= 0 || periods.is_empty() {
        return periods.to_vec();
    }
    let bridge_gap_s = max_gap_min * 60;
    let mut out: Vec<Period> = Vec::new();
    for p in periods {
        if let Some(last) = out.last() {
            if last.is_sleep && p.is_sleep {
                let gap = p.start - last.end;
                if (0..=bridge_gap_s).contains(&gap) && hr_sleep_band_across(last.end, p.start, hr, baseline) {
                    let start = last.start;
                    out.pop();
                    out.push(Period { is_sleep: true, start, end: p.end });
                    continue;
                }
            }
        }
        out.push(*p);
    }
    out
}

/// Fold a short wake run wedged between two sleep runs back into the night when HR stays in the sleep
/// band across it, so a mid-night stir does not split one night into two separately staged spans.
/// `max_min <= 0` is a no-op.
pub(super) fn absorb_short_wake(
    periods: &[Period],
    max_min: i64,
    hr: &[HrSample],
    baseline: Option<f64>,
) -> Vec<Period> {
    if max_min <= 0 || periods.len() < 3 {
        return periods.to_vec();
    }
    let max_s = max_min * 60;
    let mut out: Vec<Period> = Vec::new();
    let mut i = 0usize;
    while i < periods.len() {
        let cur = periods[i];
        let absorb = !cur.is_sleep
            && i + 1 < periods.len()
            && periods[i + 1].is_sleep
            && out.last().is_some_and(|l| l.is_sleep)
            && (cur.end - cur.start) <= max_s
            && hr_sleep_band_across(cur.start, cur.end, hr, baseline);
        if absorb {
            let prev = out.pop().expect("out.last() was Some in the guard above");
            out.push(Period { is_sleep: true, start: prev.start, end: periods[i + 1].end });
            i += 2;
            continue;
        }
        out.push(cur);
        i += 1;
    }
    out
}

/// Day HR baseline = median bpm over all HR samples; `None` if none.
pub(super) fn hr_baseline(hr: &[HrSample]) -> Option<f64> {
    if hr.is_empty() {
        return None;
    }
    Some(median(&hr.iter().map(|h| h.bpm as f64).collect::<Vec<_>>()))
}

/// Population posture variance (g^2) of a minute's gravity vectors: summed per-axis mean-squared deviation
/// over n. `None` below 2 samples (a lone sample has zero variance = a false "stable").
pub(super) fn posture_variance_g2(samples: &[AccelSample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
    for s in samples {
        sx += s.x;
        sy += s.y;
        sz += s.z;
    }
    let (mx, my, mz) = (sx / n, sy / n, sz / n);
    let mut sum_sq = 0.0;
    for s in samples {
        let (dx, dy, dz) = (s.x - mx, s.y - my, s.z - mz);
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    Some(sum_sq / n)
}

/// True when the run is deeply motion-quiescent: `>= QUIESCENT_STABLE_FRAC` of the minutes with enough
/// gravity to judge are posture-stable, over `>= QUIESCENT_MIN_STABLE_MINUTES` judged minutes.
fn run_is_deeply_quiescent(p: Period, grav: &[AccelSample]) -> bool {
    if grav.is_empty() || p.end <= p.start {
        return false;
    }
    let mut by_minute: HashMap<i64, Vec<AccelSample>> = HashMap::new();
    for g in grav {
        if g.ts >= p.start && g.ts < p.end {
            by_minute.entry(g.ts / 60).or_default().push(*g);
        }
    }
    let (mut judged, mut stable) = (0i64, 0i64);
    for samples in by_minute.values() {
        let Some(v) = posture_variance_g2(samples) else { continue };
        judged += 1;
        if v < QUIESCENT_POSTURE_VAR_G2 {
            stable += 1;
        }
    }
    if judged < QUIESCENT_MIN_STABLE_MINUTES {
        return false;
    }
    stable as f64 / judged as f64 >= QUIESCENT_STABLE_FRAC
}

/// HR-confirm a run: its MEDIAN HR must sit in the sleep band (`<= baseline x mult`). The band is
/// `HR_SLEEP_BASELINE_MULT`, widened to `QUIESCENT_HR_SLEEP_MULT` on a deeply motion-quiescent run.
pub(super) fn confirm_sleep_with_hr(
    p: Period,
    hr: &[HrSample],
    baseline: Option<f64>,
    grav: &[AccelSample],
    sleep_hr_baseline: Option<f64>,
) -> bool {
    let Some(eff_baseline) = sleep_hr_baseline.or(baseline) else { return true };
    let seg: Vec<f64> = hr.iter().filter(|h| h.ts >= p.start && h.ts <= p.end).map(|h| h.bpm as f64).collect();
    if seg.len() < HR_REFINE_MIN_SAMPLES {
        return true;
    }
    let mult = if run_is_deeply_quiescent(p, grav) { QUIESCENT_HR_SLEEP_MULT } else { HR_SLEEP_BASELINE_MULT };
    median(&seg) <= eff_baseline * mult
}

/// True when the run's center, shifted to local time, lands in `[DAYTIME_BAND_START_HOUR, END_HOUR)`.
pub(super) fn is_daytime_center(p: Period, tz_offset_s: i64) -> bool {
    let center = p.start + (p.end - p.start) / 2;
    let hour = (center + tz_offset_s).rem_euclid(SECONDS_PER_DAY) / 3_600;
    (DAYTIME_BAND_START_HOUR..DAYTIME_BAND_END_HOUR).contains(&hour)
}

/// True when the run's local-time onset is OUTSIDE the daytime band (sleep began overnight).
pub(super) fn is_overnight_onset(start: i64, tz_offset_s: i64) -> bool {
    let hour = (start + tz_offset_s).rem_euclid(SECONDS_PER_DAY) / 3_600;
    !(DAYTIME_BAND_START_HOUR..DAYTIME_BAND_END_HOUR).contains(&hour)
}

/// Stricter bar for a daytime-centered run: long enough AND a genuine resting-HR dip. `true` = keep.
fn passes_daytime_guard(p: Period, resting_hr: Option<i64>, baseline: Option<f64>) -> bool {
    if (p.end - p.start) < DAYTIME_MIN_SLEEP_MIN * 60 {
        return false;
    }
    let (Some(baseline), Some(resting)) = (baseline, resting_hr) else { return false };
    resting as f64 <= baseline * DAYTIME_RESTING_HR_MULT
}

/// True when the strap's own banked band sleep_state over the run reads predominantly "asleep".
fn band_state_confirms_asleep(p: Period, band_sleep_state: &[(i64, i32)]) -> bool {
    let in_block: Vec<i32> = band_sleep_state.iter().filter(|(t, _)| *t >= p.start && *t <= p.end).map(|(_, s)| *s).collect();
    if in_block.is_empty() {
        return false;
    }
    let asleep = in_block.iter().filter(|&&s| s == BAND_STATE_ASLEEP).count();
    asleep as f64 / in_block.len() as f64 >= MORNING_REONSET_BAND_ASLEEP_FRAC
}

/// Morning-stillness nap suppression: a daytime run starting within `MORNING_STILLNESS_WINDOW_MIN` of an
/// overnight wake must clear the daytime guard AND either a band-state re-onset or the stronger HR dip.
pub(super) fn passes_morning_stillness_guard(
    p: Period,
    resting_hr: Option<i64>,
    baseline: Option<f64>,
    morning_wake_end: Option<i64>,
    band_sleep_state: &[(i64, i32)],
) -> bool {
    let suspected = match morning_wake_end {
        Some(end) => p.start >= end && (p.start - end) <= MORNING_STILLNESS_WINDOW_MIN * 60,
        None => false,
    };
    if !suspected {
        return passes_daytime_guard(p, resting_hr, baseline);
    }
    if !passes_daytime_guard(p, resting_hr, baseline) {
        return false;
    }
    if band_state_confirms_asleep(p, band_sleep_state) {
        return true;
    }
    let (Some(baseline), Some(resting)) = (baseline, resting_hr) else { return false };
    resting as f64 <= baseline * MORNING_REONSET_RESTING_HR_MULT
}

/// The `>= OFF_WRIST_HR_GAP_MIN`-minute HR-coverage gaps within the run as `[start, end)` spans — a
/// wrist-off proxy. Gated on a dense HR stream; `[]` when HR is too sparse to assert off-wrist.
fn off_wrist_hr_gap_spans(p: Period, hr: &[HrSample]) -> Vec<(i64, i64)> {
    if hr.is_empty() || p.end <= p.start {
        return Vec::new();
    }
    let mut sorted_all = hr.to_vec();
    sorted_all.sort_by_key(|h| h.ts);
    let stream_span = sorted_all[sorted_all.len() - 1].ts - sorted_all[0].ts;
    if stream_span >= HR_DENSE_SPACING_S && (hr.len() as i64) < stream_span / HR_DENSE_SPACING_S {
        return Vec::new();
    }
    let gap_s = OFF_WRIST_HR_GAP_MIN * 60;
    let mut seg: Vec<i64> = hr.iter().filter(|h| h.ts >= p.start && h.ts <= p.end).map(|h| h.ts).collect();
    seg.sort_unstable();
    if seg.is_empty() {
        return if (p.end - p.start) >= gap_s { vec![(p.start, p.end)] } else { Vec::new() };
    }
    let mut spans = Vec::new();
    if seg[0] - p.start >= gap_s {
        spans.push((p.start, seg[0]));
    }
    for w in seg.windows(2) {
        if w[1] - w[0] >= gap_s {
            spans.push((w[0], w[1]));
        }
    }
    if p.end - seg[seg.len() - 1] >= gap_s {
        spans.push((seg[seg.len() - 1], p.end));
    }
    spans
}

/// Fractional off-wrist coverage of a run in `[0, 1]`: the union of the HR-gap spans and the supplied
/// wrist-off intervals (clipped to the run), over the run duration.
pub(super) fn off_wrist_fraction(p: Period, hr: &[HrSample], wrist_off: &[(i64, i64)]) -> f64 {
    let dur = p.end - p.start;
    if dur <= 0 {
        return 0.0;
    }
    let mut spans = off_wrist_hr_gap_spans(p, hr);
    for &(ws, we) in wrist_off {
        let (s, e) = (ws.max(p.start), we.min(p.end));
        if e > s {
            spans.push((s, e));
        }
    }
    if spans.is_empty() {
        return 0.0;
    }
    spans.sort_by_key(|s| s.0);
    let (mut covered, mut cur_start, mut cur_end) = (0i64, spans[0].0, spans[0].1);
    for &(s, e) in &spans[1..] {
        if s <= cur_end {
            cur_end = cur_end.max(e);
        } else {
            covered += cur_end - cur_start;
            (cur_start, cur_end) = (s, e);
        }
    }
    covered += cur_end - cur_start;
    covered as f64 / dur as f64
}

/// Sleep efficiency in `[0, 1]`: asleep / in-bed, asleep = in-bed − wake. In-bed is the `[start, end]`
/// SPAN, so a stage gap counts as asleep; `stages` must lie inside the span.
pub fn efficiency(start: i64, end: i64, stages: &[StageSegment]) -> f64 {
    let in_bed = (end - start) as f64;
    if in_bed <= 0.0 {
        return 0.0;
    }
    let wake: f64 = stages.iter().filter(|s| s.stage == SleepStage::Wake).map(|s| (s.end - s.start) as f64).sum();
    ((in_bed - wake).max(0.0) / in_bed).min(1.0)
}

/// One accepted in-bed span with its session resting HR (the median; the guards use the floor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedSpan {
    pub start: i64,
    pub end: i64,
    pub resting_hr: Option<i32>,
}

/// The band's own activity ladder, one code per second. We read `ASLEEP` and nothing else, which
/// throws away the two codes that mark the BOUNDARIES of a night.
///
/// Measured over two wearers' whole stores, 2026-08-18:
///
/// | code | HR | walking | when |
/// |---|---|---|---|
/// | `AWAKE` 0 | highest | 11-22% | 80% of daylight, peaks 13:00 |
/// | `SETTLED` 1 | above sleep | 0.4-0.6% | peaks 23:00, ends AT sleep onset |
/// | `ASLEEP` 2 | lowest | 0.2% | the sleep PERIOD - half to three quarters of it is truly awake |
/// | `EMERGING` 3 | above sleep | 4-10% | never before sleep; 60-80% after the last asleep second |
///
/// `ASLEEP` is a period marker, NOT evidence of sleep: on the two nights with wearer truth it is 50%
/// and 75% truly awake. These codes may feed a window; they may never score one.
pub const BAND_STATE_AWAKE: i32 = 0;
pub const BAND_STATE_SETTLED: i32 = 1;
pub const BAND_STATE_EMERGING: i32 = 3;

/// Shortest run of a boundary code that counts, so one stray second cannot move a window.
const BAND_BOUNDARY_MIN_S: i64 = 5 * 60;
/// How far either side of a detected span to look for one.
const BAND_BOUNDARY_LOOKAROUND_S: i64 = 4 * 3600;

/// The in-bed window around a detected sleep span, from the band's own boundary codes.
///
/// Opens at the start of the `SETTLED` run that runs up to `start`, and closes at the end of the
/// `EMERGING` run that follows `end`. Returns the span unchanged where the band does not say - half
/// our stores bank no sleep state at all, so absence is ordinary and must cost nothing.
///
/// This is the in-bed window, not the sleep window: `start`/`end` stay where staging put them.
pub fn band_in_bed_window(start: i64, end: i64, band: &[(i64, i32)]) -> (i64, i64) {
    (
        run_back(start, band, BAND_STATE_SETTLED).unwrap_or(start),
        run_forward(end, band, BAND_STATE_EMERGING).unwrap_or(end),
    )
}

/// Earliest second of the contiguous `code` run ending at or just before `at`. `None` when there is
/// no such run, or it is shorter than [`BAND_BOUNDARY_MIN_S`].
fn run_back(at: i64, band: &[(i64, i32)], code: i32) -> Option<i64> {
    let lo = at - BAND_BOUNDARY_LOOKAROUND_S;
    let mut earliest = None;
    // Walk backwards from `at`, allowing the run to be interrupted by nothing at all.
    for (ts, st) in band.iter().rev().filter(|(t, _)| *t < at && *t >= lo) {
        if *st == code {
            earliest = Some(*ts);
        } else {
            break;
        }
    }
    earliest.filter(|e| at - e >= BAND_BOUNDARY_MIN_S)
}

/// Last second of the contiguous `code` run starting at or just after `at`.
fn run_forward(at: i64, band: &[(i64, i32)], code: i32) -> Option<i64> {
    let hi = at + BAND_BOUNDARY_LOOKAROUND_S;
    let mut latest = None;
    for (ts, st) in band.iter().filter(|(t, _)| *t > at && *t <= hi) {
        if *st == code {
            latest = Some(*ts);
        } else {
            break;
        }
    }
    latest.filter(|l| l - at >= BAND_BOUNDARY_MIN_S)
}

/// [`detect_sessions_with`] under [`DetectParams::SHIPPED`] — the path the app runs.
pub fn detect_sessions(
    hr: &[HrSample],
    accel: &[AccelSample],
    tz_offset_s: i64,
    wrist_off: &[(i64, i64)],
    band_sleep_state: &[(i64, i32)],
    sleep_hr_baseline: Option<f64>,
) -> Vec<DetectedSpan> {
    detect_sessions_with(
        hr,
        accel,
        tz_offset_s,
        wrist_off,
        band_sleep_state,
        sleep_hr_baseline,
        &DetectParams::SHIPPED,
    )
}

/// Build the stillness spine, then run the gate loop, returning accepted sleep spans in start order. The
/// gate order and the cross-run continuation chain are load-bearing; a dropped run never re-anchors the chain.
pub fn detect_sessions_with(
    hr: &[HrSample],
    accel: &[AccelSample],
    tz_offset_s: i64,
    wrist_off: &[(i64, i64)],
    band_sleep_state: &[(i64, i32)],
    sleep_hr_baseline: Option<f64>,
    params: &DetectParams,
) -> Vec<DetectedSpan> {
    let mut grav = accel.to_vec();
    grav.sort_by_key(|g| g.ts);
    if grav.len() < 2 {
        return Vec::new();
    }
    let mut hr_s = hr.to_vec();
    hr_s.sort_by_key(|h| h.ts);
    let rhr: Vec<resting_hr::HrSample> =
        hr_s.iter().map(|h| resting_hr::HrSample { ts: h.ts, bpm: h.bpm as i32 }).collect();

    let baseline = hr_baseline(&hr_s);
    let sparse = is_gravity_sparse(&grav, &hr_s);
    let deltas = gravity_deltas(&grav);
    let flags = classify_still(&grav, &deltas, params);
    let runs = build_runs(&grav, &flags, sparse, &hr_s, baseline);
    let runs = merge_periods(&runs);
    // A data gap only splits sleep runs when gravity is sparse; a wake run splits them on either path.
    let runs = bridge_sleep_gap(&runs, if sparse { SPARSE_BRIDGE_GAP_MIN } else { 0 }, &hr_s, baseline);
    let runs = absorb_short_wake(&runs, params.wake_absorb_max_min, &hr_s, baseline);

    let min_sleep_s = params.min_sleep_min * 60;
    let continuation_gap_s = NIGHT_CONTINUATION_GAP_MIN * 60;
    let mut sessions: Vec<DetectedSpan> = Vec::new();
    let mut chain_prev_end: Option<i64> = None;
    let mut chain_from_overnight = false;

    for p in runs.iter().filter(|p| p.is_sleep) {
        if (p.end - p.start) <= min_sleep_s || (p.end - p.start) > MAX_MAIN_SLEEP_SPAN_S {
            continue;
        }
        if !confirm_sleep_with_hr(*p, &hr_s, baseline, &grav, sleep_hr_baseline) {
            continue;
        }
        if off_wrist_fraction(*p, &hr_s, wrist_off) >= MAX_OFF_WRIST_SLEEP_FRACTION {
            continue;
        }
        // Two different questions, two different statistics. The GUARDS below compare against a
        // baseline multiple and were tuned against the lowest-sustained floor, so they keep it. The
        // value carried downstream is the reported resting HR, which is the median.
        let resting_floor = resting_hr::session_resting_hr_floor(p.start, p.end, &rhr);
        let resting = resting_hr::session_resting_hr(p.start, p.end, &rhr);
        let continues_chain = chain_prev_end.is_some_and(|e| p.start - e <= continuation_gap_s);
        let is_night_tail = continues_chain && chain_from_overnight;
        let morning_wake_end = if chain_from_overnight { chain_prev_end } else { None };
        let is_daytime = is_daytime_center(*p, tz_offset_s);
        let passes_morning = if is_daytime {
            passes_morning_stillness_guard(*p, resting_floor.map(|r| r as i64), baseline, morning_wake_end, band_sleep_state)
        } else {
            true
        };
        if is_daytime && !passes_morning && !is_night_tail {
            continue;
        }
        sessions.push(DetectedSpan { start: p.start, end: p.end, resting_hr: resting });
        if !continues_chain {
            chain_from_overnight = is_overnight_onset(p.start, tz_offset_s);
        }
        chain_prev_end = Some(p.end);
    }
    sessions.sort_by_key(|s| s.start);
    sessions
}

/// Per-epoch motion magnitudes over `[start, end]` on the 30 s epoch grid: each entry is the epoch's summed
/// `|Δgravity|`. `[]` when there is too little gravity to grid (the caller then persists nothing).
pub(super) fn session_epoch_motion(start: i64, end: i64, grav: &[AccelSample]) -> Vec<f64> {
    let seg: Vec<AccelSample> = grav.iter().copied().filter(|g| g.ts >= start && g.ts <= end).collect();
    if seg.len() < 2 || end <= start {
        return Vec::new();
    }
    let deltas = gravity_deltas(&seg);
    let (startf, endf) = (start as f64, end as f64);
    let n = (((endf - startf) / 30.0).ceil() as usize).max(1);
    let mut counts = vec![0.0f64; n];
    for (k, g) in seg.iter().enumerate() {
        let ts = g.ts as f64;
        let idx = if ts < startf || ts >= endf {
            (ts == endf).then_some(n - 1)
        } else {
            Some((((ts - startf) / 30.0) as usize).min(n - 1))
        };
        if let Some(i) = idx {
            counts[i] += deltas[k];
        }
    }
    counts
}

/// The strap's own band sleep-state gridded onto the same 30 s epochs: each epoch takes the last sample in
/// its window, carrying forward when empty; lead-in takes the first sample. `[]` when no band samples.
pub(super) fn session_epoch_sleep_state(start: i64, end: i64, sleep_state: &[(i64, i32)]) -> Vec<i32> {
    let mut seg: Vec<(i64, i32)> = sleep_state.iter().copied().filter(|(t, _)| *t >= start && *t <= end).collect();
    seg.sort_by_key(|s| s.0);
    if seg.is_empty() || end <= start {
        return Vec::new();
    }
    let n_epochs = (((end - start) as f64 / 30.0).ceil() as usize).max(1);
    let mut out = vec![seg[0].1; n_epochs];
    let mut last = seg[0].1;
    let mut si = 0usize;
    for (i, slot) in out.iter_mut().enumerate() {
        let epoch_end = start + (i as i64 + 1) * 30;
        while si < seg.len() && seg[si].0 < epoch_end {
            last = seg[si].1;
            si += 1;
        }
        *slot = last;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }
    fn h(ts: i64, bpm: u16) -> HrSample {
        HrSample { ts, bpm }
    }

    /// Pins GRAVITY_STILL_THRESHOLD_G, the constant the whole detector rests on: a sample counts as
    /// still when its delta is UNDER it. A mutation sweep raised it 50% with nothing noticing, and it
    /// is the exact constant any change to the detection boundary would touch.
    #[test]
    fn the_stillness_threshold_is_where_it_says_it_is() {
        assert_eq!(GRAVITY_STILL_THRESHOLD_G, 0.01);
        let p = DetectParams::default();
        // A run of samples each moving by `step` per sample, long enough to fill the rolling window.
        let run = |step: f64| {
            let g: Vec<AccelSample> =
                (0..600).map(|i| a(i, 0.0, 0.0, 1.0 + i as f64 * step)).collect();
            let d = gravity_deltas(&g);
            let flags = classify_still(&g, &d, &p);
            flags.iter().filter(|b| **b).count()
        };
        let under = GRAVITY_STILL_THRESHOLD_G * 0.5;
        let over = GRAVITY_STILL_THRESHOLD_G * 1.5;
        assert!(run(under) > 0, "movement under the threshold must read as still");
        assert_eq!(run(over), 0, "movement over it must not");
    }

    /// Band codes as a per-second stream over `[a, b)`.
    fn band(runs: &[(i64, i64, i32)]) -> Vec<(i64, i32)> {
        let mut v = Vec::new();
        for (a, b, code) in runs {
            for t in *a..*b {
                v.push((t, *code));
            }
        }
        v
    }

    #[test]
    fn the_in_bed_window_opens_on_settled_and_closes_on_emerging() {
        let b = band(&[(0, 600, BAND_STATE_SETTLED), (600, 3600, 2), (3600, 4200, BAND_STATE_EMERGING)]);
        assert_eq!(band_in_bed_window(600, 3600, &b), (0, 4199));
    }

    /// Absence must cost nothing: half our stores bank no sleep state at all, and a short run is not
    /// a boundary. Both return the span untouched rather than guessing.
    #[test]
    fn the_in_bed_window_is_unchanged_without_a_qualifying_run() {
        assert_eq!(band_in_bed_window(600, 3600, &[]), (600, 3600), "no band data");
        let short = band(&[(540, 600, BAND_STATE_SETTLED), (600, 3600, 2)]);
        assert_eq!(band_in_bed_window(600, 3600, &short), (600, 3600), "a 1-min run is not bed entry");
    }

    /// The run must REACH the boundary. A settled stretch with waking time between it and sleep is
    /// somebody sitting down earlier in the evening, not getting into bed.
    #[test]
    fn a_settled_run_interrupted_by_waking_does_not_open_the_window() {
        let b = band(&[(0, 900, BAND_STATE_SETTLED), (900, 1200, BAND_STATE_AWAKE), (1200, 3600, 2)]);
        assert_eq!(band_in_bed_window(1200, 3600, &b).0, 1200, "the awake stretch breaks the run");
    }

    #[test]
    fn gravity_deltas_first_zero_then_l2() {
        let g = vec![a(0, 0.0, 0.0, 1.0), a(1, 0.0, 0.0, 1.0), a(2, 3.0, 4.0, 1.0)];
        let d = gravity_deltas(&g);
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[2], 5.0);
    }

    #[test]
    fn window_size_falls_back_and_tracks_cadence() {
        // Too few samples to time -> DEFAULT_INTERVAL_S -> 15 min / 60 s = 15 windows.
        assert_eq!(window_size(&[100]), (STILL_WINDOW_MIN * 60 / DEFAULT_INTERVAL_S as i64));
        // 1 s cadence -> a 15-min window spans 900 samples.
        assert_eq!(window_size(&[0, 1, 2, 3]), STILL_WINDOW_MIN * 60);
        // A 400 s gap is excluded (>= 300); the remaining 2 s gap sets the cadence.
        assert_eq!(window_size(&[0, 2, 402]), STILL_WINDOW_MIN * 60 / 2);
    }

    #[test]
    fn sparse_by_span_and_by_gap() {
        // dense: gravity spans the HR window, small gaps
        let grav: Vec<_> = (0..600).map(|t| a(t, 0.0, 0.0, 1.0)).collect();
        let hr: Vec<_> = (0..600).map(|t| h(t, 60)).collect();
        assert!(!is_gravity_sparse(&grav, &hr));
        // clumped: one > MAX_GAP_MIN (20 min) gap trips it
        let grav2 = vec![a(0, 0.0, 0.0, 1.0), a(1, 0.0, 0.0, 1.0), a(2000, 0.0, 0.0, 1.0)];
        let hr2: Vec<_> = (0..2000).map(|t| h(t, 60)).collect();
        assert!(is_gravity_sparse(&grav2, &hr2));
    }

    #[test]
    fn build_runs_breaks_on_class_change_and_gap() {
        let grav: Vec<_> = (0..10).map(|t| a(t, 0.0, 0.0, 1.0)).collect();
        let flags = vec![true, true, true, false, false, false, true, true, true, true];
        let runs = build_runs(&grav, &flags, false, &[], None);
        assert_eq!(runs.len(), 3);
        assert!(runs[0].is_sleep && !runs[1].is_sleep && runs[2].is_sleep);
    }

    /// Gravity timestamps one minute apart, so the stillness window spans exactly 15 samples.
    fn minute_grid(n: usize) -> Vec<AccelSample> {
        (0..n as i64).map(|i| a(i * 60, 0.0, 0.0, 1.0)).collect()
    }

    #[test]
    fn hysteresis_opens_on_enter_and_holds_down_to_exit() {
        let n = 90usize;
        let grav = minute_grid(n);
        // Four moving samples per 15 -> every interior window reads a still fraction of 11/15 = 0.733,
        // which sits between the two thresholds and so behaves differently opening and holding.
        let mid = |i: usize| f64::from([0usize, 4, 8, 12].contains(&(i % 15)));
        let approach: Vec<f64> =
            (0..n).map(|i| if i < 30 { 1.0 } else if i < 60 { mid(i) } else { 0.0 }).collect();
        let leave: Vec<f64> =
            (0..n).map(|i| if i < 30 { 0.0 } else if i < 60 { mid(i) } else { 1.0 }).collect();
        let asleep_mid = |d: &[f64], p: &DetectParams| classify_still(&grav, d, p)[45];

        assert!(asleep_mid(&approach, &DetectParams::PRE_HYSTERESIS)); // one threshold: it opened a run
        assert!(!asleep_mid(&approach, &DetectParams::SHIPPED)); // now too weak to open one
        assert!(asleep_mid(&leave, &DetectParams::SHIPPED)); // but still strong enough to hold one open
    }

    #[test]
    fn hr_baseline_is_median_bpm() {
        assert_eq!(hr_baseline(&[]), None);
        assert_eq!(hr_baseline(&[h(0, 50), h(1, 60), h(2, 70)]), Some(60.0));
    }

    fn sleep(start: i64, end: i64) -> Period {
        Period { is_sleep: true, start, end }
    }

    #[test]
    fn daytime_center_and_overnight_onset_use_local_hour() {
        let tz = 2 * 3600; // UTC+2
        let noon_utc = 11 * 3600; // 13:00 local
        assert!(is_daytime_center(sleep(noon_utc, noon_utc + 600), tz));
        let night_utc = 21 * 3600; // 23:00 local
        assert!(is_overnight_onset(night_utc, tz));
        assert!(!is_daytime_center(sleep(night_utc, night_utc + 600), tz));
    }

    #[test]
    fn efficiency_is_asleep_over_in_bed() {
        let stages = vec![
            StageSegment { start: 0, end: 100, stage: SleepStage::Light },
            StageSegment { start: 100, end: 150, stage: SleepStage::Wake },
        ];
        assert!((efficiency(0, 150, &stages) - 100.0 / 150.0).abs() < 1e-12);
        assert_eq!(efficiency(0, 0, &stages), 0.0);
    }

    // A gap no segment covers is IN BED and asleep, because the denominator is the span. Summing the
    // segments instead would read 50/70; the span reads 80/100, and the two only agree when they tile.
    #[test]
    fn efficiency_counts_an_uncovered_gap_as_asleep() {
        let stages = vec![
            StageSegment { start: 0, end: 50, stage: SleepStage::Light },
            StageSegment { start: 80, end: 100, stage: SleepStage::Wake },
        ];
        assert!((efficiency(0, 100, &stages) - 0.80).abs() < 1e-12);
        assert!((efficiency(0, 100, &stages) - 50.0 / 70.0).abs() > 0.08);
    }

    // Wake past the corrected wake time is not clipped away, so an edit must reclip before asking.
    #[test]
    fn efficiency_reads_wake_outside_the_span_too() {
        let stages = vec![StageSegment { start: 100, end: 200, stage: SleepStage::Wake }];
        assert_eq!(efficiency(0, 100, &stages), 0.0);
    }

    #[test]
    fn confirm_hr_trusts_gravity_under_min_samples_then_bands_on_median() {
        let p = sleep(0, 4000);
        let few: Vec<_> = (0..10).map(|t| h(t, 200)).collect();
        assert!(confirm_sleep_with_hr(p, &few, Some(60.0), &[], None));
        let hot: Vec<_> = (0..40).map(|t| h(t, 100)).collect();
        assert!(!confirm_sleep_with_hr(p, &hot, Some(60.0), &[], None));
        let cool: Vec<_> = (0..40).map(|t| h(t, 60)).collect();
        assert!(confirm_sleep_with_hr(p, &cool, Some(60.0), &[], None));
    }

    #[test]
    fn off_wrist_density_gate_disables_gap_proxy_but_events_still_count() {
        let p = sleep(0, 3600);
        let hr = vec![h(0, 60), h(3600, 60)]; // 2 samples over 3600 s -> proxy off
        assert_eq!(off_wrist_fraction(p, &hr, &[]), 0.0);
        assert!((off_wrist_fraction(p, &hr, &[(600, 2400)]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn off_wrist_gap_proxy_active_on_dense_stream() {
        let p = sleep(0, 3600);
        let hr: Vec<_> = (0..1800).map(|t| h(t, 60)).collect(); // dense first half, then a hole
        let f = off_wrist_fraction(p, &hr, &[]);
        assert!((0.49..0.51).contains(&f), "frac={f}");
    }

    #[test]
    fn band_state_confirms_asleep_at_threshold() {
        let p = sleep(0, 100);
        assert!(band_state_confirms_asleep(p, &[(10, 2), (20, 2), (30, 2), (40, 0)])); // 0.75
        assert!(!band_state_confirms_asleep(p, &[(10, 2), (20, 0), (30, 0), (40, 3)])); // 0.25
    }

    // ── gate parity pins ───────────────────────────────────────────────────────────────────────────

    const REF_MID: i64 = 1_749_513_600; // a fixed midnight (ref % 86400 == 0)
    fn at_hour(h: i64) -> i64 {
        REF_MID + h * 3600
    }
    fn still_gravity(start: i64, dur: i64) -> Vec<AccelSample> {
        (0..dur).map(|i| a(start + i, 0.0, 0.0, 1.0)).collect()
    }
    fn active_gravity(start: i64, dur: i64) -> Vec<AccelSample> {
        (0..dur).map(|i| a(start + i, (i % 2) as f64 * 0.5, 0.0, 1.0)).collect()
    }
    fn hr_stream(start: i64, dur: i64, bpm: u16) -> Vec<HrSample> {
        (0..dur).map(|i| h(start + i, bpm)).collect()
    }

    #[test]
    fn off_wrist_spans_and_fraction_precise() {
        let p = sleep(0, 3600);
        let dense: Vec<_> = (0..=3600).map(|t| h(t, 50)).collect();
        assert!(off_wrist_hr_gap_spans(p, &dense).is_empty());
        assert_eq!(off_wrist_fraction(p, &dense, &[]), 0.0);
        let mut gappy: Vec<HrSample> = (0..=600).map(|t| h(t, 50)).collect();
        gappy.extend((1860..=3600).map(|t| h(t, 50))); // 21-min interior gap 600..1860
        assert_eq!(off_wrist_hr_gap_spans(p, &gappy), vec![(600, 1860)]);
        assert!((off_wrist_fraction(p, &gappy, &[]) - 1260.0 / 3600.0).abs() < 1e-9);
        assert!((off_wrist_fraction(p, &gappy, &[(800, 1500)]) - 1260.0 / 3600.0).abs() < 1e-9); // overlap, no double-count
        assert!((off_wrist_fraction(p, &gappy, &[(2400, 3000)]) - 1860.0 / 3600.0).abs() < 1e-9); // disjoint adds
        assert!(off_wrist_hr_gap_spans(p, &[]).is_empty());
        assert_eq!(off_wrist_fraction(p, &[], &[]), 0.0);
    }

    #[test]
    fn sparse_hr_disables_off_wrist_proxy() {
        let p = sleep(0, 5400);
        let sparse: Vec<_> = [0i64, 1500, 3000, 4500].iter().map(|&t| h(t, 52)).collect();
        assert!(off_wrist_hr_gap_spans(p, &sparse).is_empty());
        assert_eq!(off_wrist_fraction(p, &sparse, &[]), 0.0);
        assert!(off_wrist_fraction(p, &sparse, &[(0, 3000)]) >= 0.5); // explicit events stay authoritative
    }

    fn detect(hr: &[HrSample], grav: &[AccelSample], tz: i64, wrist_off: &[(i64, i64)]) -> Vec<DetectedSpan> {
        detect_sessions(hr, grav, tz, wrist_off, &[], None)
    }

    #[test]
    fn daytime_short_window_rejected() {
        let (ds, dd) = (at_hour(10), 3 * 3600);
        let (ns, nd) = (ds + dd, 70 * 60); // 70 min < 90 min daytime minimum, center in [11,20)
        let mut grav = active_gravity(ds, dd);
        grav.extend(still_gravity(ns, nd));
        let mut hr = hr_stream(ds, dd, 72);
        hr.extend(hr_stream(ns, nd, 50));
        assert!(detect(&hr, &grav, 0, &[]).is_empty());
    }

    #[test]
    fn daytime_quality_nap_registers() {
        let (ds, dd) = (at_hour(10), 3 * 3600);
        let (ns, nd) = (ds + dd, 120 * 60); // 120 min + a real HR dip
        let mut grav = active_gravity(ds, dd);
        grav.extend(still_gravity(ns, nd));
        let mut hr = hr_stream(ds, dd, 72);
        hr.extend(hr_stream(ns, nd, 50));
        let s = detect(&hr, &grav, 0, &[]);
        assert_eq!(s.len(), 1);
        assert!(s[0].start >= ns && s[0].start < ns + 10 * 60);
        assert_eq!(s[0].resting_hr, Some(50));
    }

    #[test]
    fn overnight_short_window_unchanged() {
        let (ds, dd) = (at_hour(0), 3 * 3600);
        let (ss, sd) = (ds + dd, 70 * 60); // 70 min overnight (center ~03:35) > 60 min base
        let mut grav = active_gravity(ds, dd);
        grav.extend(still_gravity(ss, sd));
        let mut hr = hr_stream(ds, dd, 72);
        hr.extend(hr_stream(ss, sd, 50));
        let s = detect(&hr, &grav, 0, &[]);
        assert_eq!(s.len(), 1);
        assert!(s[0].start >= ss && s[0].start < ss + 10 * 60);
    }

    #[test]
    fn tz_offset_shifts_window_into_daytime() {
        let (start, dur) = (at_hour(2), 70 * 60);
        let grav = still_gravity(start, dur);
        let hr = hr_stream(start, dur, 50);
        assert_eq!(detect(&hr, &grav, 0, &[]).len(), 1); // overnight at tz 0
        assert!(detect(&hr, &grav, 10 * 3600, &[]).is_empty()); // daytime at +10h -> rejected
    }

    #[test]
    fn empty_hr_daytime_no_crash() {
        let grav = still_gravity(at_hour(13), 120 * 60);
        assert!(detect(&[], &grav, 0, &[]).is_empty());
    }

    #[test]
    fn off_wrist_short_tail_kept_but_long_off_wrist_dropped() {
        let start = at_hour(2);
        let dur = 90 * 60;
        let grav = still_gravity(start, dur);
        let hr = hr_stream(start, dur, 50);
        assert_eq!(detect(&hr, &grav, 0, &[]).len(), 1);
        // 5-min blip (~5.5%) keeps it; a wrist-off covering >=50% drops it
        assert_eq!(detect(&hr, &grav, 0, &[(start + 30 * 60, start + 35 * 60)]).len(), 1);
        assert!(detect(&hr, &grav, 0, &[(start + 5 * 60, start + dur)]).is_empty());
    }

    #[test]
    fn real_night_with_short_off_wrist_tail_kept() {
        let start = at_hour(1);
        let (worn, tail) = (210 * 60, 30 * 60); // 3.5 h worn + 30 min off-wrist tail (~12.5%)
        let grav = still_gravity(start, worn + tail);
        let hr = hr_stream(start, worn, 50); // HR stops at the wake
        assert_eq!(detect(&hr, &grav, 0, &[]).len(), 1);
    }

    fn sparse_still_gravity(start: i64, dur: i64, every: i64) -> Vec<AccelSample> {
        let mut out = Vec::new();
        let mut t = 0;
        while t < dur {
            out.push(a(start + t, 0.0, 0.0, 1.0));
            t += every;
        }
        out
    }

    #[test]
    fn sparse_gravity_night_not_shredded() {
        let (start, dur) = (at_hour(1), 6 * 3600);
        let grav = sparse_still_gravity(start, dur, 25 * 60); // each 25-min gap > maxGapMin
        let hr = hr_stream(start, dur, 50);
        assert!(is_gravity_sparse(&grav, &hr));
        let s = detect(&hr, &grav, 0, &[]);
        assert_eq!(s.len(), 1);
        assert!(s[0].end - s[0].start > 5 * 3600);
        assert_eq!(s[0].resting_hr, Some(50));
    }

    #[test]
    fn dense_gravity_night_unchanged() {
        let (start, dur) = (at_hour(2), 6 * 3600);
        let grav = still_gravity(start, dur);
        let hr = hr_stream(start, dur, 50);
        assert!(!is_gravity_sparse(&grav, &hr));
        let s = detect(&hr, &grav, 0, &[]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].start, start);
        assert_eq!(s[0].end, start + dur - 1);
        assert_eq!(s[0].resting_hr, Some(50));
    }

    #[test]
    fn gravity_sparse_gate_conditions() {
        let start = 6_000_000i64;
        let hr = hr_stream(start, 6 * 3600, 50);
        assert!(is_gravity_sparse(&still_gravity(start, 30 * 60), &hr)); // (a) short span
        assert!(is_gravity_sparse(&sparse_still_gravity(start, 6 * 3600, 25 * 60), &hr)); // (b) big gaps
        assert!(!is_gravity_sparse(&still_gravity(start, 6 * 3600), &hr)); // (c) dense
        assert!(!is_gravity_sparse(&sparse_still_gravity(start, 6 * 3600, 25 * 60), &[])); // (d) no HR span
        let mut clumped = still_gravity(start, 160 * 60); // (e) spans night, small median gap, one big dropout
        clumped.extend(still_gravity(start + (160 + 40) * 60, 160 * 60));
        assert!(is_gravity_sparse(&clumped, &hr));
    }

    #[test]
    fn clumped_gravity_with_long_dropout_bridged() {
        let start = at_hour(2);
        let (block, gap) = (40 * 60, 30 * 60);
        let mut grav = still_gravity(start, block);
        grav.extend(still_gravity(start + block + gap, block));
        let hr = hr_stream(start, 2 * block + gap, 50);
        assert!(is_gravity_sparse(&grav, &hr));
        let s = detect(&hr, &grav, 0, &[]);
        assert_eq!(s.len(), 1);
        assert!(s[0].end - s[0].start > 2 * block);
    }

    #[test]
    fn build_runs_dense_splits_on_real_gap() {
        let start = 5_000_000i64;
        let mut grav = still_gravity(start, 40 * 60);
        grav.extend(still_gravity(start + 40 * 60 + 30 * 60, 40 * 60)); // 30-min gap
        let deltas = gravity_deltas(&grav);
        let flags = classify_still(&grav, &deltas, &DetectParams::SHIPPED);
        assert!(build_runs(&grav, &flags, false, &[], None).len() >= 2);
    }

    #[test]
    fn overnight_tail_past_noon_kept() {
        let (n_start, n_dur) = (at_hour(2), 8 * 3600); // 02:00 -> 10:00 night
        let (w_start, w_dur) = (n_start + n_dur, 40 * 60); // 40-min morning stir (active, HR 70)
        let (t_start, t_dur) = (w_start + w_dur, 2 * 3600); // 10:40 -> 12:40 tail (still, HR 50, daytime center)
        let mut grav = still_gravity(n_start, n_dur);
        grav.extend(active_gravity(w_start, w_dur));
        grav.extend(still_gravity(t_start, t_dur));
        let mut hr = hr_stream(n_start, n_dur, 50);
        hr.extend(hr_stream(w_start, w_dur, 70));
        hr.extend(hr_stream(t_start, t_dur, 50));
        let s = detect(&hr, &grav, 0, &[]);
        let latest = s.iter().map(|x| x.end).max().unwrap_or(0);
        assert!(latest >= t_start + t_dur - 10 * 60); // the overnight tail is kept, wake not truncated
    }

    #[test]
    fn hr_confirm_median_survives_spike_but_rejects_elevated() {
        let start = 1_000_000i64;
        let p = Period { is_sleep: true, start, end: start + 600 };
        let spiky: Vec<_> = (0..600).map(|i| h(start + i, if i < 30 { 190 } else { 48 })).collect();
        assert!(confirm_sleep_with_hr(p, &spiky, Some(50.0), &[], None)); // median 48 <= 52.5
        let hot: Vec<_> = (0..600).map(|i| h(start + i, 60)).collect();
        assert!(!confirm_sleep_with_hr(p, &hot, Some(50.0), &[], None)); // median 60 > 52.5
    }

    fn still_with_turnovers(start: i64, dur: i64, every_min: i64) -> Vec<AccelSample> {
        (0..dur)
            .map(|i| {
                let minute = i / 60;
                let in_burst = minute % every_min == 0 && minute > 0 && (i % 60) < 4;
                if in_burst {
                    a(start + i, 0.35, 0.30, 0.87)
                } else {
                    a(start + i, 0.0, 0.0, 1.0)
                }
            })
            .collect()
    }
    fn walking_gravity(start: i64, dur: i64) -> Vec<AccelSample> {
        (0..dur).map(|i| a(start + i, 0.5 * ((i % 4) as f64).sin(), 0.5 * ((i % 4) as f64).cos(), 0.7)).collect()
    }
    fn elevated_flat_hr(start: i64, dur: i64, base: u16) -> Vec<HrSample> {
        (0..dur).map(|i| h(start + i, base + ((i / 90) % 3) as u16)).collect()
    }

    #[test]
    fn span_cap_drops_overlong_but_keeps_15h() {
        let (s1, over) = (at_hour(22), 18 * 3600);
        assert!(detect(&hr_stream(s1, over, 50), &still_gravity(s1, over), 0, &[]).is_empty()); // 18h > 16h cap
        let (s2, ok) = (at_hour(21), 15 * 3600);
        assert_eq!(detect(&hr_stream(s2, ok, 50), &still_gravity(s2, ok), 0, &[]).len(), 1); // 15h <= cap
    }

    /// A night broken by one stir: `[still d1][active stir][still d2]`, the stir at `stir_bpm`.
    fn stirred_night(d1: i64, stir: i64, stir_bpm: u16, d2: i64) -> (Vec<AccelSample>, Vec<HrSample>) {
        let s1 = at_hour(23) - 86_400;
        let (s_stir, s2) = (s1 + d1, s1 + d1 + stir);
        let mut grav = still_gravity(s1, d1);
        grav.extend(active_gravity(s_stir, stir));
        grav.extend(still_gravity(s2, d2));
        let mut hr = hr_stream(s1, d1, 50);
        hr.extend(hr_stream(s_stir, stir, stir_bpm));
        hr.extend(hr_stream(s2, d2, 50));
        (grav, hr)
    }

    #[test]
    fn short_hr_vouched_stir_is_absorbed_and_the_night_stays_one_span() {
        let (four_h, three_h) = (4 * 3600, 3 * 3600);
        let (g, h) = stirred_night(four_h, 30 * 60, 50, three_h);
        let one = detect(&h, &g, 0, &[]);
        assert_eq!(one.len(), 1, "a 30-min sleep-band stir is inside wake_absorb_max_min");
        assert!(one[0].end - one[0].start > four_h + three_h, "the stir is inside the span");
        // Longer than the absorb window: the night legitimately splits in two.
        let (g, h) = stirred_night(four_h, 70 * 60, 50, three_h);
        assert_eq!(detect(&h, &g, 0, &[]).len(), 2);
        // Same width, but HR out of the sleep band: nothing vouches for it, so it splits.
        let (g, h) = stirred_night(four_h, 30 * 60, 90, three_h);
        assert_eq!(detect(&h, &g, 0, &[]).len(), 2);
    }

    #[test]
    fn morning_stillness_guard_cases() {
        let daytime = |sh: i64, dm: i64| Period { is_sleep: true, start: at_hour(sh), end: at_hour(sh) + dm * 60 };
        let wake = at_hour(8);
        let p = daytime(9, 120);
        assert!(!passes_morning_stillness_guard(p, Some(74), Some(80.0), Some(wake), &[])); // 74>72 re-onset bar
        assert!(passes_morning_stillness_guard(p, Some(70), Some(78.0), Some(wake), &[])); // clear dip 70<=70.2
        assert!(passes_morning_stillness_guard(daytime(14, 120), Some(70), Some(80.0), None, &[])); // no wake -> daytime bar
        let band: Vec<(i64, i32)> = (0..100).map(|i| (p.start + i * 60, if i < 80 { 2 } else { 1 })).collect();
        assert!(passes_morning_stillness_guard(p, Some(74), Some(80.0), Some(wake), &band)); // band rescues
    }

    #[test]
    fn run_is_deeply_quiescent_cases() {
        let (start, dur) = (1_000_000i64, 60 * 60);
        let p = Period { is_sleep: true, start, end: start + dur };
        assert!(run_is_deeply_quiescent(p, &still_gravity(start, dur)));
        assert!(run_is_deeply_quiescent(p, &still_with_turnovers(start, dur, 90))); // occasional turn-overs still >90% stable
        assert!(!run_is_deeply_quiescent(p, &walking_gravity(start, dur)));
        assert!(!run_is_deeply_quiescent(p, &[]));
    }

    #[test]
    fn motionless_elevated_confirmed_only_with_gravity() {
        let (start, dur) = (1_000_000i64, 90 * 60);
        let p = Period { is_sleep: true, start, end: start + dur };
        let hr = elevated_flat_hr(start, dur, 58); // median ~59
        assert!(confirm_sleep_with_hr(p, &hr, Some(48.0), &still_gravity(start, dur), None)); // 59 <= 48*1.30
        assert!(!confirm_sleep_with_hr(p, &hr, Some(48.0), &[], None)); // 59 > 48*1.05 strict
        let hot = elevated_flat_hr(start, dur, 72); // median ~73 > 48*1.30=62.4
        assert!(!confirm_sleep_with_hr(p, &hot, Some(48.0), &still_gravity(start, dur), None)); // floor holds
    }

    #[test]
    fn session_epoch_motion_grids() {
        let (start, dur) = (at_hour(2), 90 * 60);
        let m = session_epoch_motion(start, start + dur, &still_gravity(start, dur));
        assert_eq!(m.len(), 180);
        assert!(m.iter().all(|&v| v >= 0.0));
        assert!(m.iter().sum::<f64>().abs() < 1e-6); // a perfectly still stream has ~zero motion
        assert!(session_epoch_motion(0, 1800, &[]).is_empty());
    }

    #[test]
    fn session_epoch_sleep_state_grids() {
        let start = 1_000_000i64;
        let band: Vec<(i64, i32)> = (0..(90 * 60 / 30)).map(|i| (start + i * 30, 2)).collect();
        let s = session_epoch_sleep_state(start, start + 90 * 60, &band);
        assert_eq!(s.len(), 180);
        assert!(s.iter().all(|&x| x == 2));
        assert_eq!(session_epoch_sleep_state(0, 180, &[(0, 0), (75, 2), (160, 3)]), vec![0, 0, 2, 2, 2, 3]);
        assert!(session_epoch_sleep_state(0, 1800, &[]).is_empty());
        assert!(session_epoch_sleep_state(0, 1800, &[(9000, 2)]).is_empty()); // out-of-window ignored
    }
}
