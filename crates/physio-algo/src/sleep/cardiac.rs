//! Cardiac features over one window of beats: interval order statistics and the successive-difference
//! family.
//!
//! The single producer for both the stager and the screening harnesses, so what is measured is what
//! runs. Input is the `(t_seconds, rr_ms)` sequence `v2` reconstructs from whole-second runs and
//! `hrv_bands` resamples; nothing here touches the frequency domain.

use crate::stats::{mean, percentile, population_sd};

/// The beat reconstruction every window here is taken over, re-exported so `cardiac::extract` and its
/// input have one import. Defined once in `sleep::common`.
pub use super::common::reconstruct_beats;

/// Analysis window the order statistics are centred on, seconds: nine 30 s epochs. `cardiac_emit`
/// centres it on the epoch; the screening harnesses read it from here.
pub const WINDOW_S: f64 = 270.0;

/// Quantiles taken of both the absolute and the detrended interval series.
pub const PCTS: [f64; 7] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];

/// Fewest beats a window needs. Below this the tail quantiles are reading individual beats.
pub const MIN_BEATS: usize = 20;

/// Longest gap two beats may span and still count as successive. Past it a beat was dropped, so the
/// difference is not a beat-to-beat one and would report the dropout instead of the rhythm.
const MAX_ADJACENT_S: f64 = 2.5;

/// Difference threshold of pNN50, ms.
const NN_MS: f64 = 50.0;

/// Column names of [`Block::row`], in its order.
pub const NAMES: [&str; 18] = [
    "rr_p05",
    "rr_p10",
    "rr_p25",
    "rr_p50",
    "rr_p75",
    "rr_p90",
    "rr_p95",
    "rr_dt_p05",
    "rr_dt_p10",
    "rr_dt_p25",
    "rr_dt_p50",
    "rr_dt_p75",
    "rr_dt_p90",
    "rr_dt_p95",
    "mean_hr",
    "rmssd",
    "pnn50",
    "mean_abs_drr",
];

/// One window's cardiac block. The successive-difference three are `None` when the window holds no
/// adjacent pair, which a coverage gap can cause while the quantiles remain measurable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    /// [`PCTS`] of the interval series, ms.
    pub abs: [f64; 7],
    /// The same quantiles after a least-squares linear trend in time is removed from the window.
    pub detrended: [f64; 7],
    /// Beats per minute implied by the window's mean interval.
    pub mean_hr: f64,
    pub rmssd: Option<f64>,
    pub pnn50: Option<f64>,
    pub mean_abs_drr: Option<f64>,
}

impl Block {
    /// The block as one row in [`NAMES`] order, a missing difference statistic as NaN.
    pub fn row(&self) -> [f64; 18] {
        let mut out = [f64::NAN; 18];
        out[..7].copy_from_slice(&self.abs);
        out[7..14].copy_from_slice(&self.detrended);
        out[14] = self.mean_hr;
        out[15] = self.rmssd.unwrap_or(f64::NAN);
        out[16] = self.pnn50.unwrap_or(f64::NAN);
        out[17] = self.mean_abs_drr.unwrap_or(f64::NAN);
        out
    }
}

/// Least-squares linear trend in time removed. "Detrended" is not defined by the sources this follows,
/// so this reading of it is ours and is labelled as such wherever it is reported.
fn detrend(t: &[f64], v: &[f64]) -> Vec<f64> {
    let (mt, mv) = (mean(t), mean(v));
    let sxx: f64 = t.iter().map(|x| (x - mt).powi(2)).sum();
    let sxy: f64 = t.iter().zip(v).map(|(x, y)| (x - mt) * (y - mv)).sum();
    let slope = if sxx > f64::EPSILON { sxy / sxx } else { 0.0 };
    t.iter().zip(v).map(|(x, y)| y - (mv + slope * (x - mt))).collect()
}

/// Quantiles at [`PCTS`] of an unsorted series.
fn quantiles(v: &[f64]) -> [f64; 7] {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    PCTS.map(|p| percentile(&s, p))
}

/// The block over `[t0, t1]`, or `None` when the window carries fewer than [`MIN_BEATS`]. `beats` is
/// ascending in time.
pub fn extract(beats: &[(f64, f64)], t0: f64, t1: f64) -> Option<Block> {
    let win: Vec<(f64, f64)> = beats.iter().copied().filter(|(t, _)| *t >= t0 && *t <= t1).collect();
    if win.len() < MIN_BEATS {
        return None;
    }
    let t: Vec<f64> = win.iter().map(|b| b.0).collect();
    let rr: Vec<f64> = win.iter().map(|b| b.1).collect();

    // A difference only counts where the two beats really are neighbours; across a dropout the pair
    // spans several beats and its difference describes the gap.
    let d: Vec<f64> = win
        .windows(2)
        .filter(|w| w[1].0 - w[0].0 <= MAX_ADJACENT_S)
        .map(|w| w[1].1 - w[0].1)
        .collect();
    let n = d.len() as f64;
    let some = |x: f64| (!d.is_empty()).then_some(x);

    Some(Block {
        abs: quantiles(&rr),
        detrended: quantiles(&detrend(&t, &rr)),
        mean_hr: 60_000.0 / mean(&rr),
        rmssd: some((d.iter().map(|x| x * x).sum::<f64>() / n.max(1.0)).sqrt()),
        pnn50: some(d.iter().filter(|x| x.abs() > NN_MS).count() as f64 / n.max(1.0)),
        mean_abs_drr: some(d.iter().map(|x| x.abs()).sum::<f64>() / n.max(1.0)),
    })
}

/// Per-recording z-score of one column. Missing stays missing, and a column with fewer than two
/// present values or no spread returns all-missing rather than a manufactured zero.
pub fn zscore_column(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = v.iter().flatten().copied().collect();
    if present.len() < 2 {
        return vec![None; v.len()];
    }
    let (m, sd) = (mean(&present), population_sd(&present));
    if sd <= 0.0 {
        return vec![None; v.len()];
    }
    v.iter().map(|o| o.map(|x| (x - m) / sd)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` beats at a constant interval, starting at `t0`.
    fn steady(n: usize, t0: f64, rr_ms: f64) -> Vec<(f64, f64)> {
        (0..n).map(|k| (t0 + k as f64 * rr_ms / 1000.0, rr_ms)).collect()
    }

    #[test]
    fn a_short_window_has_no_block() {
        let b = steady(MIN_BEATS - 1, 0.0, 1000.0);
        assert!(extract(&b, 0.0, 300.0).is_none());
        let b = steady(MIN_BEATS, 0.0, 1000.0);
        assert!(extract(&b, 0.0, 300.0).is_some());
    }

    #[test]
    fn a_steady_rhythm_has_no_variability_and_the_stated_rate() {
        let blk = extract(&steady(60, 0.0, 800.0), 0.0, 300.0).unwrap();
        assert!((blk.mean_hr - 75.0).abs() < 1e-9);
        assert_eq!(blk.abs, [800.0; 7]);
        assert_eq!(blk.rmssd, Some(0.0));
        assert_eq!(blk.pnn50, Some(0.0));
        assert_eq!(blk.mean_abs_drr, Some(0.0));
    }

    #[test]
    fn detrending_removes_a_linear_ramp_and_the_absolute_quantiles_keep_it() {
        // Linear in TIME, which is the trend `detrend` removes: beats at 1 Hz, rr = 1000 + 4t.
        let b: Vec<(f64, f64)> =
            (0..60).map(|k| (k as f64, 1000.0 + 4.0 * k as f64)).collect();
        let blk = extract(&b, 0.0, 300.0).unwrap();
        assert!(blk.detrended.iter().all(|x| x.abs() < 1e-6), "{:?}", blk.detrended);
        assert!(blk.abs[6] - blk.abs[0] > 200.0, "the absolute spread survives detrending");
    }

    #[test]
    fn a_dropout_pair_is_excluded_from_the_differences() {
        // Two steady runs 30 s apart. The one pair spanning the gap would contribute a 400 ms jump.
        let mut b = steady(30, 0.0, 800.0);
        b.extend(steady(30, 60.0, 1200.0));
        let blk = extract(&b, 0.0, 300.0).unwrap();
        assert_eq!(blk.rmssd, Some(0.0), "the straddling pair is the only non-zero difference");
        assert_eq!(blk.pnn50, Some(0.0));

        // Same beats with the gap closed: the pair is now adjacent and must count.
        let mut c = steady(30, 0.0, 800.0);
        c.extend(steady(30, 24.0, 1200.0));
        let blk = extract(&c, 0.0, 300.0).unwrap();
        assert!(blk.rmssd.unwrap() > 0.0);
        assert!(blk.pnn50.unwrap() > 0.0);
    }

    #[test]
    fn the_row_is_in_names_order_and_carries_nan_for_a_missing_statistic() {
        let blk = extract(&steady(60, 0.0, 800.0), 0.0, 300.0).unwrap();
        let r = blk.row();
        assert_eq!(r.len(), NAMES.len());
        assert_eq!(r[14], blk.mean_hr);
        let missing = Block { rmssd: None, pnn50: None, mean_abs_drr: None, ..blk };
        assert!(missing.row()[15..].iter().all(|x| x.is_nan()));
    }

    #[test]
    fn a_flat_or_too_short_column_z_scores_to_missing_not_to_zero() {
        assert_eq!(zscore_column(&[Some(1.0), Some(1.0)]), vec![None, None]);
        assert_eq!(zscore_column(&[Some(1.0), None]), vec![None, None]);
        let z = zscore_column(&[Some(1.0), Some(3.0), None]);
        assert_eq!(z, vec![Some(-1.0), Some(1.0), None]);
    }
}
