//! Shared helpers for the V2 sleep stager: a per-night z-scorer and the R-R run flattener. The numeric
//! primitives come from `crate::stats`. Kept private to the `sleep` module.

use super::input::RrRun;
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
