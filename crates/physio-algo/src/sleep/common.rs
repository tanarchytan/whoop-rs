//! Shared helpers for the V2 sleep stager: a per-night z-scorer and the R-R run flattener. The numeric
//! primitives come from `crate::stats`. Kept private to the `sleep` module.

use super::input::RrRun;

/// The analysis window the beat-window features are taken over, seconds: nine 30 s epochs. One
/// definition for the order statistics and the band powers alike, re-exported by `cardiac` and
/// `hrv_bands` so each keeps its own path.
pub const WINDOW_S: f64 = 270.0;
pub(super) use crate::stats::median;
use crate::stats::population_sd;

/// A per-night z-scorer over present values: population std, with a flat channel (0 std → 1) neutral,
/// and a missing value scoring the neutral centre 0.
pub(super) struct ZScore {
    mean: f64,
    sd: f64,
    empty: bool,
}

impl ZScore {
    pub(super) fn build(vals: &[Option<f64>]) -> Self {
        let present: Vec<f64> = vals.iter().filter_map(|v| *v).collect();
        if present.is_empty() {
            return ZScore { mean: 0.0, sd: 1.0, empty: true };
        }
        let mean = present.iter().sum::<f64>() / present.len() as f64;
        let sd0 = population_sd(&present);
        let sd = if sd0 == 0.0 { 1.0 } else { sd0 };
        ZScore { mean, sd, empty: false }
    }

    pub(super) fn apply(&self, value: Option<f64>) -> f64 {
        match value {
            _ if self.empty => 0.0,
            None => 0.0,
            Some(v) => (v - self.mean) / self.sd,
        }
    }
}

/// Whole-second beat stamps spread back out by their own intervals: `(t_seconds, rr_ms)`, ascending.
/// The reconstruction `v2` runs before any cardiac statistic sees a beat, and what `cardiac::extract`
/// and the screening harnesses consume. Re-exported as `sleep::cardiac::reconstruct_beats`.
pub fn reconstruct_beats(runs: &[RrRun]) -> Vec<(f64, f64)> {
    let mut beats: Vec<(f64, f64)> = Vec::new();
    for run in runs {
        let mut off = 0.0;
        for (k, ms) in run.intervals.iter().enumerate() {
            let ms = (*ms as f64).clamp(300.0, 2000.0);
            // An interval is the gap from the PREVIOUS beat, so the first of a run sits on the stamp.
            if k > 0 {
                off += ms / 1000.0;
            }
            beats.push((run.ts as f64 + off, ms));
        }
    }
    beats.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    beats
}

/// Flatten grouped R-R runs into `(ts, rr_ms)` pairs in emission order — the shape the V2 stager buckets by
/// second. A run reports several beats under one whole-second anchor.
pub fn flatten_rr(runs: &[RrRun]) -> Vec<(i64, f64)> {
    let mut out = Vec::new();
    for run in runs {
        for &ms in &run.intervals {
            out.push((run.ts, ms as f64));
        }
    }
    out
}
