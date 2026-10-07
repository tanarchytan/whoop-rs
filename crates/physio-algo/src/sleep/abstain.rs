//! Abstention: which epochs of a hypnogram to decline to label.
//!
//! Every epoch gets a stage whether the evidence supports one or not. The rule here needs the decoded
//! path only, no emissions: epochs next to a decoded stage change are the ones most likely to be
//! placed a step early or late, so they are refused first and the rest keep their label. Measured at
//! [`DEFAULT_COVERAGE`] it is ahead of a class-matched random drop of the same size, with no lone
//! single-epoch holes (see [`DEFAULT_COVERAGE`]).
//!
//! This is a wellness display aid, not a diagnosis: a refused epoch means "not confident enough to
//! draw a stage here", nothing more.

use super::features::EPOCH_S;
use super::SleepStage;

/// Fewest epochs kept after abstention. A night shorter than this refuses nothing, and a longer one
/// never keeps fewer, however low the coverage asked for.
pub const MIN_EPOCHS: usize = 20;

/// Fraction of epochs KEPT by default (the rest are refused).
///
/// Measured over 20 random draws in `_r13/abstain-20draws.log` (gate A3): at 80% coverage the
/// class-matched random null LOSES to far-from-edge by 2.19x (dreamt), 2.44x (aauwss) and 3.10x
/// (sleep-accel) of the paired bar, and 0% of its refusal runs are a lone epoch. At 90% it does not
/// clear the null on aauwss, so this is 80%, not higher.
pub const DEFAULT_COVERAGE: f64 = 0.80;

/// Epochs from each epoch to the nearest decoded stage change, which lies BETWEEN two epochs, so the
/// pair either side of it both score 0. A night that never changes stage scores every epoch
/// [`f64::INFINITY`].
pub fn to_edge(labels: &[SleepStage]) -> Vec<f64> {
    let edges: Vec<i64> =
        (1..labels.len()).filter(|k| labels[*k] != labels[k - 1]).map(|k| k as i64).collect();
    (0..labels.len() as i64)
        .map(|k| {
            edges
                .iter()
                .map(|e| (e - k).abs().min((e - 1 - k).abs()) as f64)
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Of the epoch indices `idx`, the ones to refuse, ascending: everything past the top
/// `max(MIN_EPOCHS, round(len * coverage))` by `score` (higher keeps), capped at `idx.len()`. Ties
/// break by index, so of equally scored epochs the LATER ones are refused. `idx` need not be the
/// whole night: a caller scoring only the epochs a reference labels passes just those.
pub fn refuse_among(score: &[f64], idx: &[usize], coverage: f64) -> Vec<usize> {
    let take = ((idx.len() as f64 * coverage).round() as usize).max(MIN_EPOCHS).min(idx.len());
    let mut order = idx.to_vec();
    order.sort_by(|a, b| score[*b].total_cmp(&score[*a]).then(a.cmp(b)));
    let mut dropped = order[take..].to_vec();
    dropped.sort_unstable();
    dropped
}

/// `refused[e]` for each epoch of a night: `true` where the stage should not be drawn. Keeps the
/// `round(n * coverage)` epochs farthest from a decoded stage change (at least [`MIN_EPOCHS`]).
pub fn far_from_edge(labels: &[SleepStage], coverage: f64) -> Vec<bool> {
    let all: Vec<usize> = (0..labels.len()).collect();
    let mut refused = vec![false; labels.len()];
    for k in refuse_among(&to_edge(labels), &all, coverage) {
        refused[k] = true;
    }
    refused
}

/// Contiguous refused runs as `(start_ts, end_ts)`, in seconds. `starts[e]` is epoch `e`'s start; a
/// run ends where the epoch after it starts, or [`EPOCH_S`] after the last epoch's start when the
/// run reaches the end of the night.
pub fn spans(starts: &[i64], refused: &[bool]) -> Vec<(i64, i64)> {
    let n = starts.len().min(refused.len());
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if !refused[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < n && refused[j + 1] {
            j += 1;
        }
        out.push((starts[i], starts.get(j + 1).copied().unwrap_or(starts[j] + EPOCH_S)));
        i = j + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use SleepStage::{Deep, Light, Rem, Wake};

    fn blocks(stages: &[SleepStage], len: usize) -> Vec<SleepStage> {
        stages.iter().flat_map(|s| std::iter::repeat_n(*s, len)).collect()
    }

    fn runs(refused: &[bool]) -> Vec<(usize, usize)> {
        spans(&(0..refused.len() as i64).collect::<Vec<_>>(), refused)
            .into_iter()
            .map(|(a, b)| (a as usize, b as usize))
            .collect()
    }

    #[test]
    fn full_coverage_refuses_nothing() {
        let night = blocks(&[Wake, Light, Deep, Rem], 20);
        assert!(far_from_edge(&night, 1.0).iter().all(|r| !r));
    }

    #[test]
    fn refused_count_follows_the_floor_and_the_round() {
        for n in [0usize, 5, 19, 20, 25, 50, 100, 1000] {
            let night: Vec<_> = (0..n).map(|k| if k % 7 < 3 { Wake } else { Light }).collect();
            let kept = ((0.8 * n as f64).round() as usize).max(MIN_EPOCHS).min(n);
            let got = far_from_edge(&night, 0.8).iter().filter(|r| **r).count();
            assert_eq!(got, n - kept, "n = {n}");
        }
    }

    #[test]
    fn a_night_without_a_stage_change_refuses_its_latest_epochs() {
        let refused = far_from_edge(&[Light; 100], 0.8);
        let want: Vec<bool> = (0..100).map(|k| k >= 80).collect();
        assert_eq!(refused, want);
    }

    #[test]
    fn a_multi_stage_night_has_no_lone_epoch_hole() {
        let refused = far_from_edge(&blocks(&[Wake, Light, Deep, Rem], 20), 0.8);
        assert_eq!(runs(&refused), vec![(18, 22), (37, 43), (57, 63)]);
        assert!(runs(&refused).iter().all(|(a, b)| b - a > 1));
    }

    #[test]
    fn spans_close_at_the_next_epoch_and_at_the_night_end() {
        let starts: Vec<i64> = (0..8).map(|k| 1000 + 30 * k).collect();
        let refused = [false, true, true, false, true, false, true, true];
        assert_eq!(spans(&starts, &refused), vec![(1030, 1090), (1120, 1150), (1180, 1240)]);
        assert!(spans(&starts, &[false; 8]).is_empty());
    }
}
