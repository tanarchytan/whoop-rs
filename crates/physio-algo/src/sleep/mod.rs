//! Sleep staging — per-30 s-epoch hypnogram over a detected in-bed span.
//!
//! The V2 (cardiorespiratory) recipe stages every strap: z-scored HR / HR-variability / motion emissions, a
//! deep gate on HR-flatness, a soft sleep-cycle prior, a self-calibrating jerk wake gate, an R-R RSA
//! respiration term, and Viterbi transition smoothing.
//!
//! Inputs are protocol-free (see [`input`]). [`analyze`] runs the whole pipeline (detect → stage → refine)
//! over a night's streams, and each stage is exported on its own so a caller can run just one:
//! [`detect_sessions`] (or [`detect_sessions_with`] under a chosen [`DetectParams`]) returns the in-bed
//! windows without staging them, [`stage_v2`] stages one already-detected `[start, end]` span,
//! [`refine_wake`] runs the wake refinement over a staging the caller already has, and
//! [`stage_refined`] is the two together. Staging splits once more: [`emissions_v2`] is the per-epoch
//! log-evidence and [`decode_v2`] the path search over it, so the two can be scored apart. Pure +
//! deterministic.
//!
//! [`refine_wake`]'s density gate declines silently on a sparse or absent step stream, so it is exported
//! as [`motion_dense`]: ask it first and a caller can never report an unrefined span as a refined one.

mod common;
mod detect;
mod input;
mod mainnight;
pub mod metrics;
pub mod params;
pub mod posture;
mod refine;
mod v2;

use crate::hrv::HrvReadiness;

pub use input::{AccelSample, HrSample, RrRun, SleepInput, StepSample};
pub use params::Params;
pub use detect::{detect_sessions, detect_sessions_with, efficiency, DetectParams, DetectedSpan,
    DAYTIME_BAND_END_HOUR, DAYTIME_BAND_START_HOUR, MAX_GAP_MIN, SPARSE_GRAVITY_SPAN_FRAC};
pub use v2::{emissions_prepared as emissions_v2, epoch_starts as epoch_starts_v2, prepare as prepare_v2,
    segments_of as segments_v2, stage as stage_v2, stage_prepared as stage_v2_prepared,
    stage_with as stage_v2_with, viterbi as decode_v2, Prepared, DEEP_GATE_THRESH, STAGE_ORDER};
pub use refine::{
    is_motion_dense as motion_dense, motion_density, refine as refine_wake, refine_with as refine_wake_with,
    RefineParams, MIN_DENSE_FRACTION,
};
pub use mainnight::{
    bridge_adjacent, bridged_night_groups, habitual_midsleep_sec, habitual_midsleep_series,
    main_night_group_indices, main_night_group_indices_scored, main_night_index, main_night_index_scored,
    main_night_selection, main_night_selection_scored, BridgedNightGroup, HistoryBlock, MainNightReason,
    MainNightSelection, NightBlock, ScoredNightBlock, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS,
    OVERNIGHT_END_HOUR, OVERNIGHT_START_HOUR,
};

/// The stream bundle for one detection window: raw per-sample signals plus the wrist-off intervals, band
/// sleep-state `(ts, state)`, and the local UTC offset the daytime guard needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SleepStreams {
    pub hr: Vec<HrSample>,
    pub rr: Vec<RrRun>,
    pub accel: Vec<AccelSample>,
    pub steps: Vec<StepSample>,
    pub tz_offset_s: i64,
    pub wrist_off: Vec<(i64, i64)>,
    pub band_sleep_state: Vec<(i64, i32)>,
}

/// One detected sleep session: its span, the motion-refined V2 hypnogram, the derived scalars, and the
/// per-30 s-epoch motion + band sleep-state grids the caller persists beside the hypnogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    pub start: i64,
    pub end: i64,
    pub efficiency: f64,
    pub resting_hr: Option<i32>,
    pub avg_hrv: Option<f64>,
    pub segments: Vec<StageSegment>,
    pub motion_grid: Vec<f64>,
    pub sleep_state_grid: Vec<i32>,
}

/// Detect in-bed spans from a window's streams and stage each with the V2 recipe + motion-aware wake
/// refinement, returning one [`Session`] per accepted span with its resting HR and windowed average HRV.
pub fn analyze(streams: &SleepStreams) -> Vec<Session> {
    let spans = detect::detect_sessions(
        &streams.hr,
        &streams.accel,
        streams.tz_offset_s,
        &streams.wrist_off,
        &streams.band_sleep_state,
        None,
    );
    let mut beats: Vec<(u32, u16)> = Vec::new();
    for run in &streams.rr {
        for &ms in &run.intervals {
            beats.push((run.ts as u32, ms));
        }
    }
    beats.sort_by_key(|b| b.0);

    let mut out = Vec::with_capacity(spans.len());
    for span in spans {
        let input = SleepInput {
            start: span.start,
            end: span.end,
            hr: streams.hr.clone(),
            rr: streams.rr.clone(),
            accel: streams.accel.clone(),
        };
        let segments = refine::refine(&v2::stage(&input), &streams.accel, &streams.steps);
        let efficiency = detect::efficiency(span.start, span.end, &segments);
        let avg_hrv = HrvReadiness::windowed_avg_hrv(span.start as u32, span.end as u32, &beats);
        let motion_grid = detect::session_epoch_motion(span.start, span.end, &streams.accel);
        let sleep_state_grid = detect::session_epoch_sleep_state(span.start, span.end, &streams.band_sleep_state);
        out.push(Session {
            start: span.start,
            end: span.end,
            efficiency,
            resting_hr: span.resting_hr,
            avg_hrv,
            segments,
            motion_grid,
            sleep_state_grid,
        });
    }
    out
}

/// Stage a single detected span with the V2 recipe + the motion-aware wake refinement — the single-span
/// re-stage a caller runs after editing a session's bounds.
pub fn stage_refined(input: &SleepInput, steps: &[StepSample]) -> Vec<StageSegment> {
    if !is_stageable(input) {
        return Vec::new();
    }
    refine::refine(&v2::stage(input), &input.accel, steps)
}

/// Beats per hour of span below which a night cannot be staged.
///
/// A resting adult produces roughly 3,000 beats an hour, so this floor is two orders of
/// magnitude below a real night. It is deliberately far from the boundary: it exists to catch
/// spans with NO cardiac signal, not to judge marginal ones.
pub const MIN_BEATS_PER_HOUR: f64 = 30.0;

/// Fraction of the span that must carry heart rate.
pub const MIN_HR_COVERAGE: f64 = 0.10;

/// Whether a span carries enough signal for a hypnogram to mean anything.
///
/// Deep and REM are separated ONLY by the R-R-derived `hr_var` term, so a span without R-R
/// cannot distinguish them: both emissions collapse to their base rate and the night is scored
/// light and awake by construction. Six of David's paired nights did exactly that, emitting deep
/// and REM at exactly 0.0 minutes against WHOOP's own 74 and 64. Eleven more were import-sink
/// rows with no samples at all.
///
/// Returning an empty hypnogram is the honest answer: the caller keeps whatever it already had,
/// which for an imported night is WHOOP's own scoring.
pub fn is_stageable(input: &SleepInput) -> bool {
    let span_s = (input.end - input.start).max(0) as f64;
    if span_s < 600.0 {
        return false;
    }
    let hours = span_s / 3600.0;
    let beats: usize = input.rr.iter().map(|r| r.intervals.len()).sum();
    if (beats as f64) < MIN_BEATS_PER_HOUR * hours {
        return false;
    }
    let hr_in_span = input.hr.iter().filter(|s| s.ts >= input.start && s.ts <= input.end).count();
    (hr_in_span as f64) >= MIN_HR_COVERAGE * span_s
}

/// A sleep stage. String forms are `"wake" | "light" | "deep" | "rem"` for cross-platform parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SleepStage {
    Wake,
    Light,
    Deep,
    Rem,
}

impl SleepStage {
    /// The JSON label, identical across the platform twins.
    pub fn as_str(self) -> &'static str {
        match self {
            SleepStage::Wake => "wake",
            SleepStage::Light => "light",
            SleepStage::Deep => "deep",
            SleepStage::Rem => "rem",
        }
    }
}

/// A contiguous run of one stage. Times are wall-clock unix seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSegment {
    pub start: i64,
    pub end: i64,
    pub stage: SleepStage,
}

#[cfg(test)]
mod golden_tests;


#[cfg(test)]
mod stageable_tests {
    use super::*;
    use crate::sleep::input::{HrSample, RrRun};

    /// A span with `hr_hz` heart-rate samples a second and `beats` R-R intervals.
    fn night(hours: f64, hr_per_s: f64, beats: usize) -> SleepInput {
        let span = (hours * 3600.0) as i64;
        let n_hr = (span as f64 * hr_per_s) as i64;
        SleepInput {
            start: 0,
            end: span,
            hr: (0..n_hr).map(|i| HrSample { ts: i, bpm: 60 }).collect(),
            rr: if beats == 0 {
                Vec::new()
            } else {
                vec![RrRun { ts: 0, intervals: vec![1000u16; beats] }]
            },
            accel: Vec::new(),
        }
    }

    #[test]
    fn a_real_night_is_stageable() {
        // 8 h, one HR sample a second, ~3,600 beats an hour.
        assert!(is_stageable(&night(8.0, 1.0, 28_800)));
    }

    #[test]
    fn the_import_sink_is_not_stageable() {
        // The `my-whoop` rows: a span, and nothing else. Eleven of David's nights looked like this
        // and were staged anyway.
        assert!(!is_stageable(&night(8.0, 0.0, 0)));
    }

    #[test]
    fn hr_without_rr_is_not_stageable() {
        // The six paired nights that emitted deep and REM at exactly 0.0 minutes: HR present, R-R
        // absent, so `hr_var` is absent for every epoch and deep/REM can never win.
        assert!(!is_stageable(&night(8.0, 1.0, 0)));
    }

    #[test]
    fn a_trickle_of_beats_is_not_enough() {
        // 100 beats across 8 hours is 12.5 an hour, well under the 30 floor.
        assert!(!is_stageable(&night(8.0, 1.0, 100)));
    }

    #[test]
    fn rr_without_hr_coverage_is_not_stageable() {
        // Beats present but the HR series barely covers the span.
        assert!(!is_stageable(&night(8.0, 0.01, 28_800)));
    }

    #[test]
    fn a_span_too_short_to_mean_anything_is_not_stageable() {
        assert!(!is_stageable(&night(0.1, 1.0, 400)));
    }

    #[test]
    fn an_unstageable_span_yields_no_segments_rather_than_a_hypnogram() {
        // The property that matters to the app: no stages at all, so an imported night keeps the
        // durations it arrived with instead of being overwritten by a fabricated one.
        assert!(stage_refined(&night(8.0, 0.0, 0), &[]).is_empty());
    }
}
