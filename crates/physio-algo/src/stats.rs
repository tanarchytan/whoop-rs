//! Small pure statistics shared by the metrics: mean, the series extremes, sample/population SD, OLS
//! slope, median, median sample gap, percentile, and the robust pulsatile amplitude (p95 − p5).

/// Textbook two-sided 95% critical values, keyed by degrees of freedom.
const T95: [(usize, f64); 12] = [
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (9, 2.262), (12, 2.179),
    (19, 2.093), (30, 2.042), (39, 2.023), (59, 2.001), (119, 1.980),
];

/// Two-sided 95% critical value at `df`, floored at df 1 (the widest, most conservative row); 1.96
/// is ~11% too narrow at df 12. Rounds df DOWN to the previous row — the value falls as df rises, so
/// rounding up would return a bar narrower than the truth.
pub fn t95_df(df: usize) -> f64 {
    let df = df.max(1);
    T95.iter().rev().find(|(k, _)| *k <= df).expect("the table starts at df 1").1
}

/// Arithmetic mean; `0.0` for an empty slice.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Smallest value in a series; `0.0` when empty, NaN when any element is NaN. The low end of a chart's
/// own axis, sharing an owner with the [`mean`] drawn between the two extremes. Unlike the internal
/// `fold(f64::INFINITY, f64::min)`, it neither skips a NaN nor returns `±∞` for an empty series.
pub fn min(xs: &[f64]) -> f64 {
    extreme(xs, false)
}

/// Largest value in a series; `0.0` when empty, NaN when any element is NaN. See [`min`].
pub fn max(xs: &[f64]) -> f64 {
    extreme(xs, true)
}

/// Shared body of [`min`] and [`max`]: NaN propagates instead of being skipped, and a ±0.0 tie resolves
/// by sign — the two rules a platform `Math.max`/`Math.min` fold follows.
fn extreme(xs: &[f64], want_max: bool) -> f64 {
    let Some((&first, rest)) = xs.split_first() else {
        return 0.0;
    };
    rest.iter().fold(first, |a, &b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if a == b {
            // Only ±0.0 compare equal while differing: max prefers +0.0, min prefers -0.0.
            if a.is_sign_negative() == want_max { b } else { a }
        } else if (b > a) == want_max {
            b
        } else {
            a
        }
    })
}

/// Sample standard deviation (n − 1); `0.0` for fewer than two points.
pub fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Population standard deviation (divide by n); `0.0` for an empty slice. The per-night / per-window
/// spread the z-scorers use, as distinct from the n−1 [`sample_sd`] the baselines use.
pub fn population_sd(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    if var < 0.0 {
        0.0
    } else {
        var.sqrt()
    }
}

/// OLS slope of `ys` over x = 0, 1, 2, …; `0.0` for fewer than two points or a degenerate x-spread.
pub fn least_squares_slope(ys: &[f64]) -> f64 {
    if ys.len() < 2 {
        return 0.0;
    }
    let mean_x = (ys.len() - 1) as f64 / 2.0;
    let mean_y = mean(ys);
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &y) in ys.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// OLS line of `ys` over x = 0, 1, 2, … as `(slope, intercept)`. `(0.0, mean)` for fewer than two points.
/// The single source for both the slope (`least_squares_slope`) and a full linear detrend.
pub fn least_squares_line(ys: &[f64]) -> (f64, f64) {
    let slope = least_squares_slope(ys);
    let mean_x = ys.len().saturating_sub(1) as f64 / 2.0;
    (slope, mean(ys) - slope * mean_x)
}

/// Median: the middle on odd counts, the mean of the two middles on even counts; `0.0` when empty.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Median spacing (s) between consecutive timestamps, restricted to plausible `(0, 300)` gaps, floored at
/// 1.0; `fallback` when no plausible gap exists. Not a true median: takes the upper-middle after sort.
/// The one sample-cadence estimate every per-sample duration credit is derived from.
pub fn median_gap_s(times: &[i64], fallback: f64) -> f64 {
    if times.len() < 2 {
        return fallback;
    }
    let mut gaps: Vec<f64> = Vec::new();
    for w in times.windows(2) {
        let g = (w[1] - w[0]) as f64;
        if g > 0.0 && g < 300.0 {
            gaps.push(g);
        }
    }
    if gaps.is_empty() {
        return fallback;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gaps[gaps.len() / 2].max(1.0)
}

/// Linear-interpolated percentile over an ascending-sorted slice; `p` in 0..=1; `0.0` when empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = p * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Robust pulsatile amplitude of a window: p95 − p5, so a lone spike moves neither tail.
pub fn amplitude(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&s, 0.95) - percentile(&s, 0.05)
}

/// Pearson correlation of two equal-length series; `None` for < 2 pairs or a zero-variance series. The
/// per-strap "is this field signal against a known reference" number — computed, never assumed.
pub fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return None;
    }
    let (mx, my) = (mean(&xs[..n]), mean(&ys[..n]));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (xs[i] - mx, ys[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// Fewest overlapping pairs a correlation may be shown from; two points fit any line exactly.
pub const CORRELATION_MIN_PAIRS: usize = 3;

/// `|r|` floors for [`correlation_strength`], each inclusive: an `|r|` on a floor is the band above.
pub const CORRELATION_WEAK_FLOOR: f64 = 0.1;
pub const CORRELATION_MODERATE_FLOOR: f64 = 0.3;
pub const CORRELATION_STRONG_FLOOR: f64 = 0.5;
pub const CORRELATION_VERY_STRONG_FLOOR: f64 = 0.7;

/// How strong a correlation reads. Each surface words these its own way; none owns the cut points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrelationStrength {
    Negligible,
    Weak,
    Moderate,
    Strong,
    VeryStrong,
}

/// Strength band of a Pearson `r`, cut by the `CORRELATION_*_FLOOR` constants on `|r|`, so −0.62 and
/// +0.62 read equally strong. Direction is the caller's to word. Feed it a finite `r`: one that is
/// not a number falls past every floor into the top band.
pub fn correlation_strength(r: f64) -> CorrelationStrength {
    let m = r.abs();
    if m < CORRELATION_WEAK_FLOOR {
        CorrelationStrength::Negligible
    } else if m < CORRELATION_MODERATE_FLOOR {
        CorrelationStrength::Weak
    } else if m < CORRELATION_STRONG_FLOOR {
        CorrelationStrength::Moderate
    } else if m < CORRELATION_VERY_STRONG_FLOOR {
        CorrelationStrength::Strong
    } else {
        CorrelationStrength::VeryStrong
    }
}

/// A per-strap linear calibration `reference ≈ scale·field + offset`, with the `r` that says how far to
/// trust it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearFit {
    pub scale: f64,
    pub offset: f64,
    pub r: f64,
}

/// Least-squares fit of `reference` onto `field` (`ref ≈ scale·field + offset`) plus the Pearson `r`.
/// `None` for < 2 pairs or a field with no spread. How the client derives a device-specific coefficient
/// from one strap's own captures instead of hardcoding another strap's number.
pub fn linear_fit(field: &[f64], reference: &[f64]) -> Option<LinearFit> {
    let n = field.len().min(reference.len());
    if n < 2 {
        return None;
    }
    let (mf, mr) = (mean(&field[..n]), mean(&reference[..n]));
    let (mut sfr, mut sff) = (0.0, 0.0);
    for i in 0..n {
        let df = field[i] - mf;
        sfr += df * (reference[i] - mr);
        sff += df * df;
    }
    if sff <= 0.0 {
        return None;
    }
    let scale = sfr / sff;
    Some(LinearFit { scale, offset: mr - scale * mf, r: pearson(&field[..n], &reference[..n])? })
}

// ── Weighted trendline over day offsets ───────────────────────────────────

/// 80 % two-sided normal quantile: the interval half-width in standard errors.
const TREND_CI_Z: f64 = 1.282;
/// Fewest points a trend may be fitted from; also the `n − 2` the residual variance needs.
const TREND_MIN_POINTS: usize = 3;
/// Shortest span a trend may cover, and the fraction of the requested window it must otherwise reach.
const TREND_MIN_SPAN_FLOOR_DAYS: f64 = 3.0;
const TREND_MIN_SPAN_WINDOW_FRACTION: f64 = 1.0 / 3.0;

/// Shortest observed span a `window_days`-wide request accepts: a third of the window, never under three
/// days. Stops three near-adjacent days passing as a trend across a long window.
pub fn trend_min_span_days(window_days: f64) -> f64 {
    (window_days * TREND_MIN_SPAN_WINDOW_FRACTION).max(TREND_MIN_SPAN_FLOOR_DAYS)
}

/// Which way a series moved once its interval is accounted for. `Flat` means the interval straddles
/// zero, so the direction is not separable from noise — it is not "no movement".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrendDirection {
    Rising,
    Falling,
    Flat,
}

/// A weighted linear trend over day offsets, carrying its own uncertainty so no caller has to pick a
/// slope threshold. `slope` is per day; `total_change` spans `start_day`..`end_day`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trendline {
    pub slope: f64,
    pub intercept: f64,
    pub slope_se: f64,
    pub slope_ci_lo: f64,
    pub slope_ci_hi: f64,
    pub start_day: f64,
    pub end_day: f64,
    /// Fitted, not observed — the two points a trend line is drawn between.
    pub start_value: f64,
    pub end_value: f64,
    pub total_change: f64,
    pub total_change_ci_lo: f64,
    pub total_change_ci_hi: f64,
    /// |slope| in standard errors — unbounded, so it still ranks two already-certain trends.
    pub slope_z: f64,
    /// Monotone squash of `slope_z` into [0, 1). A display scalar, not a probability.
    pub significance: f64,
    pub direction: TrendDirection,
    pub n: usize,
}

/// Weighted least-squares trend of `values` over `days` (x in days, not sample index), with a residual
/// based 80 % interval on the slope. `weights` may be empty for unit weights.
/// `None` under three finite points, under `min_span_days` of observed span, or with no weighted x-spread.
pub fn weighted_trendline(
    days: &[f64],
    values: &[f64],
    weights: &[f64],
    min_span_days: f64,
) -> Option<Trendline> {
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
    if n < TREND_MIN_POINTS {
        return None;
    }
    let start_day = x.iter().copied().fold(f64::INFINITY, f64::min);
    let end_day = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = end_day - start_day;
    if span < min_span_days {
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
    // No weighted spread in x: the slope is undefined and the interval would be infinite.
    if ss_xx <= 0.0 {
        return None;
    }
    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    // Residual-based standard error, so the interval widens on a scattered series. A declared-variance
    // SE (1/ss_xx) cannot see scatter, and we have no validated per-metric measurement CV to declare.
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
    let half = TREND_CI_Z * slope_se;
    let (slope_ci_lo, slope_ci_hi) = (slope - half, slope + half);
    let direction = if slope_ci_lo > 0.0 {
        TrendDirection::Rising
    } else if slope_ci_hi < 0.0 {
        TrendDirection::Falling
    } else {
        TrendDirection::Flat
    };
    Some(Trendline {
        slope,
        intercept,
        slope_se,
        slope_ci_lo,
        slope_ci_hi,
        start_day,
        end_day,
        start_value: slope * start_day + intercept,
        end_value: slope * end_day + intercept,
        total_change: slope * span,
        total_change_ci_lo: slope_ci_lo * span,
        total_change_ci_hi: slope_ci_hi * span,
        slope_z: z,
        significance: 1.0 - (-0.5 * z * z).exp(),
        direction,
        n,
    })
}

/// Second-half mean minus first-half mean of a series, the odd point going to the second half.
/// `None` under four points — the window-change number a trend chip shows.
pub fn half_change(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    let mid = values.len() / 2;
    Some(mean(&values[mid..]) - mean(&values[..mid]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_sd_slope_basic() {
        assert_eq!(mean(&[2.0, 4.0, 6.0]), 4.0);
        assert!((sample_sd(&[2.0, 4.0, 6.0]) - 2.0).abs() < 1e-12);
        assert!((least_squares_slope(&[1.0, 2.0, 3.0, 4.0]) - 1.0).abs() < 1e-12);
        assert_eq!(least_squares_slope(&[5.0]), 0.0);
    }

    #[test]
    fn least_squares_line_matches_slope_and_recovers_intercept() {
        // y = 2x + 3 over x = 0..4 → slope 2, intercept 3.
        let ys = [3.0, 5.0, 7.0, 9.0, 11.0];
        let (slope, intercept) = least_squares_line(&ys);
        assert!((slope - 2.0).abs() < 1e-12 && (intercept - 3.0).abs() < 1e-12);
        assert!((slope - least_squares_slope(&ys)).abs() < 1e-12);
        assert_eq!(least_squares_line(&[42.0]), (0.0, 42.0)); // < 2 points → (0, mean)
    }

    /// The parity oracle for the ported chart axes: the labels the app's own `max()`/`min()` printed
    /// beside this module's `mean` before the extremes moved here. One series of 5-minute HR bucket
    /// means (a day's heart-rate rail) and one of daily trend points.
    #[test]
    fn min_max_reproduce_the_axis_labels_the_app_printed() {
        let bpm = [72.4, 58.2, 61.0, 143.8, 99.5, 58.2, 84.1];
        assert_eq!(min(&bpm), 58.2);
        assert_eq!(max(&bpm), 143.8);
        // The row is max / mean / min, and all three now come from this module.
        assert!((mean(&bpm) - 82.457_142_857_142_86).abs() < 1e-12);

        let recovery = [64.0, 71.0, 48.0, 88.0, 55.0];
        assert_eq!(min(&recovery), 48.0);
        assert_eq!(max(&recovery), 88.0);
        assert_eq!(mean(&recovery), 65.2);
    }

    #[test]
    fn min_max_are_empty_safe_and_nan_propagating() {
        // Empty is 0.0 like every other reduction here, never a panic and never ±∞ on an axis.
        assert_eq!(min(&[]), 0.0);
        assert_eq!(max(&[]), 0.0);
        assert_eq!(min(&[42.0]), 42.0);
        assert_eq!(max(&[42.0]), 42.0);
        // NaN propagates rather than being skipped, so an axis cannot read healthier than its mean:
        // `mean` already returns NaN for the same series.
        assert!(min(&[1.0, f64::NAN, 3.0]).is_nan());
        assert!(max(&[1.0, f64::NAN, 3.0]).is_nan());
        assert!(max(&[f64::NAN]).is_nan());
    }

    /// A ±0.0 tie resolves by sign, the one input where a plain comparison answers differently.
    #[test]
    fn min_max_resolve_a_zero_tie_the_way_the_app_does() {
        assert!(max(&[-0.0, 0.0]).is_sign_positive());
        assert!(max(&[0.0, -0.0]).is_sign_positive());
        assert!(min(&[-0.0, 0.0]).is_sign_negative());
        assert!(min(&[0.0, -0.0]).is_sign_negative());
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[]), 0.0); // empty is 0, never an index panic
    }

    #[test]
    fn population_sd_divides_by_n() {
        // [2,4,6]: mean 4, squared devs 4+0+4 = 8 -> /3 = 2.667, sqrt ~1.633 (vs 2.0 for sample_sd).
        assert!((population_sd(&[2.0, 4.0, 6.0]) - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(population_sd(&[]), 0.0);
        assert_eq!(population_sd(&[7.0]), 0.0);
    }

    #[test]
    fn median_gap_uses_upper_middle_and_drops_implausible() {
        assert_eq!(median_gap_s(&[100], 60.0), 60.0); // too few to time -> fallback
        assert_eq!(median_gap_s(&[0, 1, 2, 3], 60.0), 1.0);
        assert_eq!(median_gap_s(&[0, 2, 402], 60.0), 2.0); // the 400 s gap is excluded
        assert_eq!(median_gap_s(&[0, 900], 1.0), 1.0); // no plausible gap -> fallback
    }

    #[test]
    fn percentile_empty_is_zero_not_a_panic() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[42.0], 0.9), 42.0);
    }

    #[test]
    fn amplitude_is_p95_minus_p5() {
        let win: Vec<f64> = std::iter::repeat_n(98.0, 10).chain(std::iter::repeat_n(102.0, 10)).collect();
        assert!((amplitude(&win) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn linear_fit_recovers_a_known_line() {
        // reference = 2·field + 3, perfectly correlated.
        let field = [1.0, 2.0, 3.0, 4.0, 5.0];
        let reference: Vec<f64> = field.iter().map(|&x| 2.0 * x + 3.0).collect();
        assert!((pearson(&field, &reference).unwrap() - 1.0).abs() < 1e-12);
        let fit = linear_fit(&field, &reference).unwrap();
        assert!((fit.scale - 2.0).abs() < 1e-9 && (fit.offset - 3.0).abs() < 1e-9 && (fit.r - 1.0).abs() < 1e-9);
        // A flat field has no spread — nothing to calibrate.
        assert!(linear_fit(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn trend_min_span_is_a_third_of_the_window_floored_at_three_days() {
        assert!((trend_min_span_days(7.0) - 3.0).abs() < 1e-12); // 2.33 → floored
        assert!((trend_min_span_days(31.0) - 31.0 / 3.0).abs() < 1e-12);
        assert!((trend_min_span_days(366.0) - 122.0).abs() < 1e-12);
    }

    #[test]
    fn trendline_recovers_a_known_line_and_its_fitted_endpoints() {
        let days = [0.0, 1.0, 2.0, 3.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let t = weighted_trendline(&days, &values, &[], 3.0).unwrap();
        assert!((t.slope - 1.0).abs() < 1e-12 && (t.intercept - 1.0).abs() < 1e-12);
        assert!((t.start_value - 1.0).abs() < 1e-12 && (t.end_value - 4.0).abs() < 1e-12);
        assert!((t.total_change - 3.0).abs() < 1e-12);
        assert_eq!(t.n, 4);
        // No scatter → zero SE → the interval collapses onto the slope and the call is Rising.
        assert_eq!(t.slope_se, 0.0);
        assert!((t.significance - 1.0).abs() < 1e-12);
        assert_eq!(t.direction, TrendDirection::Rising);
    }

    #[test]
    fn trendline_x_is_days_not_sample_index() {
        // Three samples 0, 1 and 30 days in on a 1.0/day line. Index-x would read 15.0/sample.
        let t = weighted_trendline(&[0.0, 1.0, 30.0], &[0.0, 1.0, 30.0], &[], 3.0).unwrap();
        assert!((t.slope - 1.0).abs() < 1e-12);
        assert!((t.total_change - 30.0).abs() < 1e-12);
    }

    #[test]
    fn trendline_interval_sees_scatter_so_the_same_slope_can_read_flat() {
        // y = 0.5x + 0.5 through [0,2,1]: ss_xx 2, ss_res 1.5 → se sqrt(0.75).
        let t = weighted_trendline(&[0.0, 1.0, 2.0], &[0.0, 2.0, 1.0], &[], 0.0).unwrap();
        assert!((t.slope - 0.5).abs() < 1e-12);
        assert!((t.slope_se - 0.75_f64.sqrt()).abs() < 1e-12);
        assert!((t.slope_z - 0.5 / 0.75_f64.sqrt()).abs() < 1e-12);
        assert!((t.significance - 0.153_518_3).abs() < 1e-6);
        assert_eq!(t.direction, TrendDirection::Flat); // the interval straddles zero

        // The same +0.5/day with no scatter is called, which a fixed slope threshold cannot do.
        let clean = weighted_trendline(&[0.0, 1.0, 2.0], &[0.0, 0.5, 1.0], &[], 0.0).unwrap();
        assert!((clean.slope - 0.5).abs() < 1e-12);
        assert_eq!(clean.direction, TrendDirection::Rising);
        assert!(clean.slope_se < t.slope_se);
    }

    #[test]
    fn trendline_gates_points_span_and_a_degenerate_x_spread() {
        assert!(weighted_trendline(&[0.0, 5.0], &[1.0, 2.0], &[], 0.0).is_none()); // n < 3
        assert!(weighted_trendline(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[], 3.0).is_none()); // span 2 < 3
        assert!(weighted_trendline(&[0.0, 1.5, 3.0], &[1.0, 2.0, 3.0], &[], 3.0).is_some()); // span == min

        // All the x-spread carries zero weight: without this gate the interval is ±infinity.
        let flat_w = &[1.0, 1.0, 0.0];
        assert!(weighted_trendline(&[0.0, 0.0, 20.0], &[1.0, 2.0, 3.0], flat_w, 0.0).is_none());
        // Non-finite rows are dropped, and dropping enough of them fails the point gate.
        let holed = weighted_trendline(&[0.0, 1.0, 2.0, 3.0], &[1.0, f64::NAN, 3.0, 4.0], &[], 0.0);
        assert_eq!(holed.unwrap().n, 3);
        assert!(weighted_trendline(&[0.0, 1.0, 2.0], &[1.0, f64::NAN, 3.0], &[], 0.0).is_none());
    }

    #[test]
    fn trendline_weights_shift_the_fit_toward_the_trusted_points() {
        let days = [0.0, 1.0, 2.0, 3.0];
        let values = [0.0, 1.0, 2.0, 30.0];
        let plain = weighted_trendline(&days, &values, &[], 3.0).unwrap();
        assert!((plain.slope - 9.1).abs() < 1e-9); // the outlier drags the unweighted fit
        let low = &[1.0, 1.0, 1.0, 0.001];
        let downweighted = weighted_trendline(&days, &values, low, 3.0).unwrap();
        assert!((downweighted.slope - 1.0).abs() < 0.05); // back on the 1.0/day line
    }

    #[test]
    fn half_change_needs_four_points_and_gives_the_odd_one_to_the_recent_half() {
        assert_eq!(half_change(&[1.0, 2.0, 3.0]), None);
        assert!((half_change(&[1.0, 2.0, 3.0, 4.0]).unwrap() - 2.0).abs() < 1e-12);
        // 5 points: first half is [1,2], second [3,4,5] → 4.0 − 1.5.
        assert!((half_change(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap() - 2.5).abs() < 1e-12);
    }

    /// The exact band the Compare, Mind and Insights sentences read before the ladder moved here.
    /// Every floor is inclusive: an `|r|` sitting on it belongs to the band above.
    #[test]
    fn correlation_bands_are_cut_at_the_declared_floors() {
        assert_eq!(correlation_strength(0.0), CorrelationStrength::Negligible);
        assert_eq!(correlation_strength(0.099), CorrelationStrength::Negligible);
        assert_eq!(correlation_strength(0.1), CorrelationStrength::Weak);
        assert_eq!(correlation_strength(0.299), CorrelationStrength::Weak);
        assert_eq!(correlation_strength(0.3), CorrelationStrength::Moderate);
        assert_eq!(correlation_strength(0.499), CorrelationStrength::Moderate);
        assert_eq!(correlation_strength(0.5), CorrelationStrength::Strong);
        assert_eq!(correlation_strength(0.699), CorrelationStrength::Strong);
        assert_eq!(correlation_strength(0.7), CorrelationStrength::VeryStrong);
        assert_eq!(correlation_strength(1.0), CorrelationStrength::VeryStrong);
    }

    /// Direction is the caller's to word, so the band reads off the magnitude alone.
    #[test]
    fn a_negative_r_bands_as_its_mirror() {
        for r in [0.05, 0.2, 0.4, 0.6, 0.9] {
            assert_eq!(correlation_strength(-r), correlation_strength(r), "|r| = {r}");
        }
    }

    /// Two points fit any line exactly, so the display gate sits above them.
    #[test]
    fn the_display_gate_is_three_pairs() {
        assert_eq!(CORRELATION_MIN_PAIRS, 3);
        assert!(pearson(&[1.0, 2.0], &[2.0, 4.0]).is_some(), "pearson itself still answers for two");
    }
}
