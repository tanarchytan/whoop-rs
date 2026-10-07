//! tanv1: v2 at its null config, plus abstention holes. Labels are never changed by abstention.

mod params;

pub use params::{Tanv1Cfg, BASE};

use super::abstain;
use super::input::SleepInput;
use super::v2;
use super::{SleepStage, StageSegment};

/// Hypnogram segments (tiling the span) plus the spans the engine declines to score.
#[derive(Debug, Clone, PartialEq)]
pub struct Staging {
    pub segments: Vec<StageSegment>,
    /// `(start, end)` unix seconds, ascending and non-overlapping; empty when abstention is off.
    pub unscored: Vec<(i64, i64)>,
}

/// Stage `input` with v2 under `cfg.base` (via `v2::prepare` + `v2::stage_prepared`, which read every
/// `Params` field), then mark the epochs `cfg.abstain` refuses.
pub fn stage(input: &SleepInput, cfg: &Tanv1Cfg) -> Staging {
    let prep = v2::prepare(input, &cfg.base);
    let segments = v2::stage_prepared(&prep, &cfg.base);
    let Some(coverage) = cfg.abstain else { return Staging { segments, unscored: Vec::new() } };
    let starts = v2::epoch_starts(&prep);
    let labels = labels_at(&segments, &starts);
    let unscored = abstain::spans(&starts, &abstain::far_from_edge(&labels, coverage));
    Staging { segments, unscored }
}

/// The stage of the segment holding each of `starts` (ascending).
fn labels_at(segments: &[StageSegment], starts: &[i64]) -> Vec<SleepStage> {
    let mut k = 0;
    starts
        .iter()
        .map(|&t| {
            while k + 1 < segments.len() && segments[k].end <= t {
                k += 1;
            }
            segments.get(k).map_or(SleepStage::Light, |s| s.stage)
        })
        .collect()
}

#[cfg(test)]
mod tests;
