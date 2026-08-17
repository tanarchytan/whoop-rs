//! Motion-aware wake refinement — a post-pass over a staged hypnogram that reclassifies hot-but-still
//! wake to light. For each long wake segment with no walking cadence and posture stable outside a
//! minority of burst minutes, non-burst minutes become light; burst minutes (+/- a pad) stay wake, so
//! the pass only ever shrinks wake. Gated on a per-minute density self-check over the observed streams.

use std::collections::{HashMap, HashSet};

use super::detect::posture_variance_g2;
use super::input::{AccelSample, StepSample};
use super::{SleepStage, StageSegment};

const SUSTAINED_WALK_TICKS_PER_MINUTE: i32 = 10;
const SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES: i32 = 2;
const SINGLE_MINUTE_WALK_TICKS: i32 = 40;
const MIN_GRAVITY_SAMPLES_PER_MINUTE_FOR_VARIANCE: i32 = 2;
const MIN_STEP_SAMPLES_PER_MINUTE_FOR_DENSITY: i32 = 1;
const MIN_DENSE_MINUTE_COVERAGE_FRACTION: f64 = 0.80;

/// Which wake segments the pass may convert, and how much of one it keeps. [`RefineParams::SHIPPED`] is
/// what [`refine`] runs; the other settings exist so a harness can sweep the eligibility rule and record
/// a verdict rather than argue one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefineParams {
    /// Shortest wake segment the pass will look at. Below it a segment is returned untouched.
    pub min_wake_segment_seconds: i64,
    /// Per-minute posture variance (g²) under which a minute counts as still.
    pub stable_posture_variance_g2: f64,
    /// Share of a segment's minutes that must be still before any of it converts.
    pub min_stable_minute_fraction: f64,
    /// Minutes either side of a burst minute that stay wake with it.
    pub burst_pad_minutes: i64,
    /// Leave the window's first and last segment alone. Segments alternate stage, so this is exactly the
    /// leading and trailing wake run: sleep-onset latency and post-wake in-bed time, which are awake.
    pub skip_window_edges: bool,
}

impl RefineParams {
    pub const SHIPPED: Self = Self {
        min_wake_segment_seconds: 5 * 60,
        stable_posture_variance_g2: 0.05,
        min_stable_minute_fraction: 0.80,
        burst_pad_minutes: 1,
        skip_window_edges: true,
    };
}

impl Default for RefineParams {
    fn default() -> Self {
        Self::SHIPPED
    }
}

/// Reclassify non-burst minutes of eligible wake segments to light. `segments` must tile one contiguous
/// window in order. Byte-identical passthrough when empty, degenerate, or the density gate declines.
pub fn refine(segments: &[StageSegment], grav: &[AccelSample], steps: &[StepSample]) -> Vec<StageSegment> {
    refine_with(segments, grav, steps, &RefineParams::SHIPPED)
}

/// [`refine`] under a chosen eligibility rule. The density gate is not part of `p` — it judges the streams,
/// not the recipe, so every setting is gated identically and a sweep compares like with like.
pub fn refine_with(
    segments: &[StageSegment], grav: &[AccelSample], steps: &[StepSample], p: &RefineParams,
) -> Vec<StageSegment> {
    let (Some(first), Some(last)) = (segments.first(), segments.last()) else { return segments.to_vec() };
    let (window_start, window_end) = (first.start, last.end);
    if window_end <= window_start || !is_motion_dense(window_start, window_end, grav, steps) {
        return segments.to_vec();
    }
    let grav_by_minute = bucket_by_minute(grav);
    let ticks_by_minute = walk_class_ticks_per_minute(steps);
    let mut out: Vec<StageSegment> = Vec::new();
    let n = segments.len();
    for (idx, seg) in segments.iter().enumerate() {
        if p.skip_window_edges && (idx == 0 || idx + 1 == n) {
            append_merging(*seg, &mut out);
            continue;
        }
        for piece in refine_segment(*seg, &grav_by_minute, &ticks_by_minute, p) {
            append_merging(piece, &mut out);
        }
    }
    out
}

/// True when both the gravity and step streams are dense enough over `[start, end)` to trust per-minute
/// locomotion/posture evidence. Judges the observed streams directly, never a strap model. Exported so a
/// caller can ask whether [`refine`] will act BEFORE calling it — the gate is otherwise silent.
pub fn is_motion_dense(start: i64, end: i64, grav: &[AccelSample], steps: &[StepSample]) -> bool {
    let (g, s) = motion_density(start, end, grav, steps);
    g >= MIN_DENSE_MINUTE_COVERAGE_FRACTION && s >= MIN_DENSE_MINUTE_COVERAGE_FRACTION
}

/// The gravity and step dense-minute fractions the gate compares against its threshold, and that
/// threshold. Exported so a caller can say WHICH stream declined instead of only that one did.
pub fn motion_density(start: i64, end: i64, grav: &[AccelSample], steps: &[StepSample]) -> (f64, f64) {
    (
        dense_minute_fraction(grav, start, end, MIN_GRAVITY_SAMPLES_PER_MINUTE_FOR_VARIANCE, |g| g.ts),
        dense_minute_fraction(steps, start, end, MIN_STEP_SAMPLES_PER_MINUTE_FOR_DENSITY, |s| s.ts),
    )
}

/// The dense-minute fraction both streams must reach for [`refine`] to act.
pub const MIN_DENSE_FRACTION: f64 = MIN_DENSE_MINUTE_COVERAGE_FRACTION;

/// Fraction of the wall-clock minutes tiling `[start, end)` that carry at least `min_per_minute` samples.
fn dense_minute_fraction<T>(samples: &[T], start: i64, end: i64, min_per_minute: i32, ts: impl Fn(&T) -> i64) -> f64 {
    if end <= start {
        return 0.0;
    }
    let (first, last) = (start / 60, (end - 1) / 60);
    if last < first {
        return 0.0;
    }
    let mut counts: HashMap<i64, i32> = HashMap::new();
    for s in samples {
        let m = ts(s) / 60;
        if (first..=last).contains(&m) {
            *counts.entry(m).or_insert(0) += 1;
        }
    }
    let total = last - first + 1;
    let dense = (first..=last).filter(|m| counts.get(m).copied().unwrap_or(0) >= min_per_minute).count();
    dense as f64 / total as f64
}

/// Refine one segment. Non-wake, short, or well-evidenced-active segments pass through unchanged; only a
/// segment that reads "hot-but-still" (no locomotion AND stable posture) has its non-burst minutes lit.
fn refine_segment(
    seg: StageSegment,
    grav_by_minute: &HashMap<i64, Vec<AccelSample>>,
    ticks_by_minute: &HashMap<i64, i32>,
    p: &RefineParams,
) -> Vec<StageSegment> {
    if seg.stage != SleepStage::Wake || seg.end - seg.start < p.min_wake_segment_seconds {
        return vec![seg];
    }
    let mins = minutes(seg.start, seg.end);
    if mins.is_empty() || has_locomotion(&mins, ticks_by_minute) {
        return vec![seg];
    }
    let Some(burst) = stable_burst_minutes(&mins, grav_by_minute, p) else { return vec![seg] };
    let (first_minute, last_minute) = (mins[0], mins[mins.len() - 1]);
    let mut keep_wake: HashSet<i64> = HashSet::new();
    for &m in &burst {
        let (lo, hi) = (first_minute.max(m - p.burst_pad_minutes), last_minute.min(m + p.burst_pad_minutes));
        for p in lo..=hi {
            keep_wake.insert(p);
        }
    }
    let mut result: Vec<StageSegment> = Vec::new();
    let n = mins.len();
    for (idx, &m) in mins.iter().enumerate() {
        let stage = if keep_wake.contains(&m) { SleepStage::Wake } else { SleepStage::Light };
        let start = if idx == 0 { seg.start } else { m * 60 };
        let end = if idx == n - 1 { seg.end } else { (m + 1) * 60 };
        append_merging(StageSegment { start, end, stage }, &mut result);
    }
    result
}

/// True when `mins` shows real walking: a single minute `>= SINGLE_MINUTE_WALK_TICKS`, or
/// `>= SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES` in a row each `>= SUSTAINED_WALK_TICKS_PER_MINUTE`.
fn has_locomotion(mins: &[i64], ticks_by_minute: &HashMap<i64, i32>) -> bool {
    let mut consecutive = 0;
    for m in mins {
        let ticks = ticks_by_minute.get(m).copied().unwrap_or(0);
        if ticks >= SINGLE_MINUTE_WALK_TICKS {
            return true;
        }
        if ticks >= SUSTAINED_WALK_TICKS_PER_MINUTE {
            consecutive += 1;
            if consecutive >= SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

/// The burst (not posture-stable) minutes when at least `p.min_stable_minute_fraction` of `mins` are
/// stable; `None` when too few are stable to trust. A minute with too little gravity to judge is a burst.
fn stable_burst_minutes(
    mins: &[i64], grav_by_minute: &HashMap<i64, Vec<AccelSample>>, p: &RefineParams,
) -> Option<HashSet<i64>> {
    let mut burst: HashSet<i64> = HashSet::new();
    let mut stable = 0i64;
    let empty: Vec<AccelSample> = Vec::new();
    for &m in mins {
        let samples = grav_by_minute.get(&m).unwrap_or(&empty);
        match posture_variance_g2(samples) {
            Some(v) if v < p.stable_posture_variance_g2 => stable += 1,
            _ => {
                burst.insert(m);
            }
        }
    }
    if (stable as f64) / (mins.len() as f64) < p.min_stable_minute_fraction {
        return None;
    }
    Some(burst)
}

/// Per-minute walk-class tick cadence: the shared wrap-aware counter delta between consecutive samples,
/// attributed to the later sample's minute, kept only when its class is walk (1) or run (2). A sync-gap
/// or reboot delta is dropped by the kernel, so it can never read as a minute of locomotion.
fn walk_class_ticks_per_minute(steps: &[StepSample]) -> HashMap<i64, i32> {
    let mut sorted = steps.to_vec();
    sorted.sort_by_key(|s| s.ts);
    let mut out: HashMap<i64, i32> = HashMap::new();
    for w in sorted.windows(2) {
        let cur = w[1];
        let Some(cls) = cur.activity_class else { continue };
        if cls != 1 && cls != 2 {
            continue;
        }
        let Some(delta) = crate::steps::tick_delta(&w[0], &cur) else { continue };
        *out.entry(cur.ts / 60).or_insert(0) += delta as i32;
    }
    out
}

/// The wall-clock minute indices (unix seconds / 60) tiling `[start, end)`; empty for a degenerate window.
fn minutes(start: i64, end: i64) -> Vec<i64> {
    if end <= start {
        return Vec::new();
    }
    let (first, last) = (start / 60, (end - 1) / 60);
    if last < first {
        return Vec::new();
    }
    (first..=last).collect()
}

fn bucket_by_minute(grav: &[AccelSample]) -> HashMap<i64, Vec<AccelSample>> {
    let mut out: HashMap<i64, Vec<AccelSample>> = HashMap::new();
    for &g in grav {
        out.entry(g.ts / 60).or_default().push(g);
    }
    out
}

/// Append `piece`, merging into the previous segment when the stage matches and the two are contiguous.
fn append_merging(piece: StageSegment, out: &mut Vec<StageSegment>) {
    if let Some(last) = out.last_mut() {
        if last.stage == piece.stage && last.end == piece.start {
            last.end = piece.end;
            return;
        }
    }
    out.push(piece);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }
    fn st(ts: i64, counter: u16, cls: Option<u8>) -> StepSample {
        StepSample { ts, counter, activity_class: cls }
    }

    /// Still streams dense enough for the gate over `minutes`: two gravity samples and one non-walking
    /// step sample a minute, against the 2/min and 1/min over 80% of minutes it wants.
    fn still_streams(minutes: i64) -> (Vec<AccelSample>, Vec<StepSample>) {
        let (mut grav, mut steps) = (Vec::new(), Vec::new());
        for m in 0..minutes {
            grav.push(a(m * 60, 0.0, 0.0, 1.0));
            grav.push(a(m * 60 + 30, 0.0, 0.0, 1.0));
            steps.push(st(m * 60, 100, Some(0)));
        }
        (grav, steps)
    }

    /// A window with sleep on both sides of one wake run, which is what the shipped rule converts.
    fn interior_wake(wake_min: i64) -> Vec<StageSegment> {
        let (w0, w1) = (600, 600 + wake_min * 60);
        vec![
            StageSegment { start: 0, end: w0, stage: SleepStage::Light },
            StageSegment { start: w0, end: w1, stage: SleepStage::Wake },
            StageSegment { start: w1, end: w1 + 600, stage: SleepStage::Light },
        ]
    }

    /// A window holding one short and one long wake run at its edges, plus one in its interior, so every
    /// eligibility knob has something to change.
    fn three_wake_runs() -> Vec<StageSegment> {
        vec![
            StageSegment { start: 0, end: 180, stage: SleepStage::Wake }, // 3 min, leading
            StageSegment { start: 180, end: 1200, stage: SleepStage::Light },
            StageSegment { start: 1200, end: 1800, stage: SleepStage::Wake }, // 10 min, interior
            StageSegment { start: 1800, end: 2400, stage: SleepStage::Light },
            StageSegment { start: 2400, end: 3000, stage: SleepStage::Wake }, // 10 min, trailing
        ]
    }

    #[test]
    fn hot_but_still_wake_reclassified_to_light() {
        let segs = interior_wake(10);
        let (grav, steps) = still_streams(30);
        let out = refine(&segs, &grav, &steps);
        assert_eq!(out.len(), 1, "the whole window merges to one light run");
        assert_eq!(out[0].stage, SleepStage::Light);
        assert_eq!((out[0].start, out[0].end), (0, 1800));
    }

    /// The H fix: sleep-onset latency and post-wake in-bed time are awake, and converting them removed
    /// true wake on every second of it. Reverting `skip_window_edges` fails this.
    #[test]
    fn leading_and_trailing_wake_is_never_converted() {
        let segs = three_wake_runs();
        let (grav, steps) = still_streams(50);
        let out = refine(&segs, &grav, &steps);
        assert_eq!(out.first().copied(), Some(segs[0]), "the leading wake run must survive");
        assert_eq!(out.last().copied(), Some(segs[4]), "the trailing wake run must survive");
        // The interior run is the one it is for, and it does convert.
        assert!(!out.iter().any(|s| s.stage == SleepStage::Wake && s.start == 1200));
    }

    #[test]
    fn sparse_stream_declines_and_passes_through() {
        let segs = interior_wake(10);
        let grav: Vec<_> = (0..30).map(|m| a(m * 60, 0.0, 0.0, 1.0)).collect(); // 1/min < the required 2
        assert_eq!(refine(&segs, &grav, &[]), segs);
    }

    /// The exported gate must agree with what `refine` then does, or a harness reporting "refined" off
    /// `is_motion_dense` would pool spans the refinement declined.
    #[test]
    fn the_exported_gate_predicts_whether_the_refinement_acts() {
        let segs = interior_wake(10);
        let (grav, steps) = still_streams(30);
        assert!(is_motion_dense(0, 1800, &grav, &steps));
        assert_ne!(refine(&segs, &grav, &steps), segs);
        // One step sample every third minute is 33% coverage, under the 80% the gate wants.
        let sparse: Vec<_> = steps.iter().copied().step_by(3).collect();
        assert!(!is_motion_dense(0, 1800, &grav, &sparse));
        assert_eq!(refine(&segs, &grav, &sparse), segs);
        assert!(!is_motion_dense(0, 1800, &grav, &[]));
        assert_eq!(refine(&segs, &grav, &[]), segs);
    }

    /// The pass rewrites wake to light and nothing else, so any statistic taken over deep or REM seconds
    /// is identical on both paths. Several harnesses stay on the unrefined staging because of this.
    #[test]
    fn deep_and_rem_seconds_are_untouched() {
        let segs = vec![
            StageSegment { start: 0, end: 600, stage: SleepStage::Light },
            StageSegment { start: 600, end: 1200, stage: SleepStage::Wake },
            StageSegment { start: 1200, end: 1800, stage: SleepStage::Deep },
            StageSegment { start: 1800, end: 2400, stage: SleepStage::Rem },
            StageSegment { start: 2400, end: 3000, stage: SleepStage::Light },
        ];
        let (grav, steps) = still_streams(50);
        let out = refine(&segs, &grav, &steps);
        assert_ne!(out, segs, "the gate must accept, or this proves nothing");
        let seconds = |v: &[StageSegment], want: SleepStage| -> Vec<i64> {
            v.iter().filter(|s| s.stage == want).flat_map(|s| s.start..s.end).collect()
        };
        assert_eq!(seconds(&out, SleepStage::Deep), seconds(&segs, SleepStage::Deep));
        assert_eq!(seconds(&out, SleepStage::Rem), seconds(&segs, SleepStage::Rem));
    }

    /// `refine` is `refine_with` under `SHIPPED`, so a sweep's baseline row is the shipped pass itself.
    #[test]
    fn shipped_params_reproduce_the_constant_pass() {
        let segs = interior_wake(10);
        let (grav, steps) = still_streams(30);
        let out = refine(&segs, &grav, &steps);
        assert_ne!(out, segs, "the gate must accept, or this proves nothing");
        assert_eq!(out, refine_with(&segs, &grav, &steps, &RefineParams::SHIPPED));
        assert_eq!(out, refine_with(&segs, &grav, &steps, &RefineParams::default()));
    }

    /// WHICH minutes the shipped pass leaves awake, not how many. The three runs are 3 + 10 + 10 min, so
    /// converting the interior run and converting the trailing one both total 780 s of wake: a gate on the
    /// total alone cannot tell the shipped pass from one that keeps the wrong run.
    #[test]
    fn the_shipped_pass_converts_the_interior_run_and_only_that_one() {
        let segs = three_wake_runs();
        let (grav, steps) = still_streams(50);
        let out = refine(&segs, &grav, &steps);
        assert_eq!(
            out,
            vec![
                StageSegment { start: 0, end: 180, stage: SleepStage::Wake },
                StageSegment { start: 180, end: 2400, stage: SleepStage::Light },
                StageSegment { start: 2400, end: 3000, stage: SleepStage::Wake },
            ]
        );
        let wake_s = |v: &[StageSegment]| -> i64 {
            v.iter().filter(|s| s.stage == SleepStage::Wake).map(|s| s.end - s.start).sum()
        };
        // The alias: same 780 s of wake, the wrong 600 of them.
        let kept_the_trailing_run = vec![
            StageSegment { start: 0, end: 180, stage: SleepStage::Wake },
            StageSegment { start: 180, end: 1200, stage: SleepStage::Light },
            StageSegment { start: 1200, end: 1800, stage: SleepStage::Wake },
            StageSegment { start: 1800, end: 3000, stage: SleepStage::Light },
        ];
        assert_eq!(780, wake_s(&out));
        assert_eq!(wake_s(&out), wake_s(&kept_the_trailing_run));
        assert_ne!(out, kept_the_trailing_run);
        // The two do-nothing passes: leave everything, and convert everything.
        assert_eq!(1380, wake_s(&segs));
        let all_light = vec![StageSegment { start: 0, end: 3000, stage: SleepStage::Light }];
        assert_ne!(out, segs);
        assert_ne!(out, all_light);
    }

    /// Each knob must move the outcome, or a sweep over it would report a ceiling that is really a no-op.
    /// The totals below are also the two null arms: 1380 s is the untouched input, 0 s is convert-everything.
    #[test]
    fn each_eligibility_knob_changes_the_outcome() {
        let segs = three_wake_runs();
        let (grav, steps) = still_streams(50);
        let wake_s = |p: &RefineParams| -> i64 {
            refine_with(&segs, &grav, &steps, p)
                .iter()
                .filter(|s| s.stage == SleepStage::Wake)
                .map(|s| s.end - s.start)
                .sum()
        };
        let shipped = RefineParams::SHIPPED;
        let edges = RefineParams { skip_window_edges: false, ..shipped };
        // Shipped keeps both edge runs (3 + 10 min) and converts the interior one.
        assert_eq!(wake_s(&shipped), 780);
        // With the edges eligible, only the 3-min run is left, under the 5-min floor.
        assert_eq!(
            refine_with(&segs, &grav, &steps, &edges),
            vec![
                StageSegment { start: 0, end: 180, stage: SleepStage::Wake },
                StageSegment { start: 180, end: 3000, stage: SleepStage::Light },
            ]
        );
        // A 1-min floor takes that too.
        assert_eq!(wake_s(&RefineParams { min_wake_segment_seconds: 60, ..edges }), 0);
        // Nothing converts once a still minute has to beat zero variance: the input, unchanged.
        assert_eq!(refine_with(&segs, &grav, &steps, &RefineParams { stable_posture_variance_g2: 0.0, ..edges }), segs);
        // A burst pad wide enough to reach every minute of a run keeps all of it awake.
        assert_eq!(
            wake_s(&RefineParams { stable_posture_variance_g2: 0.0, burst_pad_minutes: 60, ..edges }),
            1380
        );
    }

    #[test]
    fn walk_ticks_and_locomotion_gate() {
        let steps = vec![st(120, 100, Some(1)), st(150, 140, Some(1))]; // +40 in minute 2
        let ticks = walk_class_ticks_per_minute(&steps);
        assert_eq!(ticks.get(&2).copied().unwrap_or(0), 40);
        assert!(has_locomotion(&[2], &ticks));
        let still = vec![st(120, 100, Some(0)), st(150, 200, Some(0))];
        assert!(walk_class_ticks_per_minute(&still).is_empty());
    }

    /// Pins SUSTAINED_WALK_TICKS_PER_MINUTE and the consecutive-minute rule it feeds. The sweep raised
    /// it 50% unnoticed: the test above only exercised SINGLE_MINUTE_WALK_TICKS, which is four times
    /// larger, so any sustained threshold under 40 satisfied it.
    #[test]
    fn the_sustained_walk_cadence_is_where_it_says_it_is() {
        assert_eq!(SUSTAINED_WALK_TICKS_PER_MINUTE, 10);
        assert_eq!(SUSTAINED_WALK_MIN_CONSECUTIVE_MINUTES, 2);
        let ticks = |per_min: i32| -> HashMap<i64, i32> {
            (1..=3).map(|m| (m, per_min)).collect()
        };
        let one_under = SUSTAINED_WALK_TICKS_PER_MINUTE - 1;
        assert!(!has_locomotion(&[1, 2, 3], &ticks(one_under)),
            "a cadence under the sustained threshold is not locomotion however long it runs");
        assert!(has_locomotion(&[1, 2, 3], &ticks(SUSTAINED_WALK_TICKS_PER_MINUTE)),
            "at the threshold, two consecutive minutes are");
        assert!(!has_locomotion(&[1], &ticks(SUSTAINED_WALK_TICKS_PER_MINUTE)),
            "one minute at the threshold is not: the consecutive rule must hold too");
    }

    /// Pins MIN_DENSE_MINUTE_COVERAGE_FRACTION. The sweep raised it 10% unnoticed.
    #[test]
    fn the_density_gate_declines_just_below_its_coverage_fraction() {
        assert_eq!(MIN_DENSE_MINUTE_COVERAGE_FRACTION, 0.80);
        let minutes = 100i64;
        let (grav, steps) = still_streams(minutes);
        let end = minutes * 60;
        assert!(is_motion_dense(0, end, &grav, &steps), "full coverage is dense");
        // Thin the step stream to just under the fraction: the gate wants a step sample in 80% of
        // minutes, so 75% must decline.
        let thin: Vec<StepSample> =
            steps.iter().filter(|s| (s.ts / 60) % 4 != 0).cloned().collect();
        assert!(!is_motion_dense(0, end, &grav, &thin),
            "75% step coverage is under the fraction and must decline");
    }
}
