//! tanv1: v2 at its null config, plus abstention holes. Labels are never changed by abstention.

mod params;

pub use params::{Tanv1Cfg, BASE};

use super::abstain;
use super::input::SleepInput;
use super::v2;
use super::StageSegment;

/// Hypnogram segments (tiling the span) plus the spans the engine declines to score.
#[derive(Debug, Clone, PartialEq)]
pub struct Staging {
    pub segments: Vec<StageSegment>,
    /// `(start, end)` unix seconds, ascending and non-overlapping; empty when abstention is off.
    pub unscored: Vec<(i64, i64)>,
    /// Start of each prepared epoch (unix seconds), so a caller that changes `segments` afterwards
    /// (the motion refinement) can recompute the holes with [`abstain::holes`].
    pub starts: Vec<i64>,
}

/// Stage `input` with v2 under `cfg.base` (via `v2::prepare` + `v2::stage_prepared`, which read every
/// `Params` field), then mark the epochs `cfg.abstain` refuses.
pub fn stage(input: &SleepInput, cfg: &Tanv1Cfg) -> Staging {
    let prep = v2::prepare(input, &cfg.base);
    let segments = v2::stage_prepared(&prep, &cfg.base);
    let starts = v2::epoch_starts(&prep);
    let unscored = cfg.abstain.map_or_else(Vec::new, |c| abstain::holes(&segments, &starts, c));
    Staging { segments, unscored, starts }
}

#[cfg(test)]
mod tests;
