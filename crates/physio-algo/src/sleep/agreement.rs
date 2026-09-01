//! Recording-wise agreement: the summary measures a night is judged on, and Bland-Altman over them.
//!
//! Epoch-wise kappa answers "how often do the two agree on this epoch". It cannot answer "does the
//! device report the right amount of deep sleep", which is what a user reads. A stager can hold kappa
//! and still be an hour out on total sleep time, so both belong in a report.
//!
//! Every function here takes labels on the Wake / Light / Deep / Rem index map, one per epoch, over
//! the SAME window for device and reference. The window is the caller's to declare: these numbers are
//! meaningless across two different spans.

/// One recording's summary measures, in MINUTES except `efficiency` in percent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NightSummary {
    pub tst: f64,
    /// Wake between the first and last sleep epoch. Wake outside that is latency or trailing wake.
    pub waso: f64,
    /// Epochs before the first sleep epoch. **Window-relative**: if the window already starts at
    /// sleep, this is near zero by construction and says nothing about the wearer.
    pub sol: f64,
    /// `tst` over the whole window, in percent.
    pub efficiency: f64,
    pub light: f64,
    pub deep: f64,
    pub rem: f64,
}

const WAKE: usize = 0;

/// Summarise one hypnogram. `epoch_min` is the epoch length in minutes.
pub fn summarise(labels: &[usize], epoch_min: f64) -> Option<NightSummary> {
    if labels.is_empty() || epoch_min <= 0.0 {
        return None;
    }
    let asleep = |i: usize| labels[i] != WAKE && labels[i] < 4;
    let first = (0..labels.len()).find(|i| asleep(*i));
    let last = (0..labels.len()).rev().find(|i| asleep(*i));
    let count = |class: usize| labels.iter().filter(|l| **l == class).count() as f64 * epoch_min;

    let (light, deep, rem) = (count(1), count(2), count(3));
    let tst = light + deep + rem;
    let window = labels.len() as f64 * epoch_min;
    let (sol, waso) = match (first, last) {
        (Some(a), Some(b)) => (
            a as f64 * epoch_min,
            labels[a..=b].iter().filter(|l| **l == WAKE).count() as f64 * epoch_min,
        ),
        // Never asleep: the whole window is latency and there is no interval to hold WASO.
        _ => (window, 0.0),
    };
    Some(NightSummary { tst, waso, sol, efficiency: 100.0 * tst / window, light, deep, rem })
}

/// Bland-Altman over paired per-recording values, plus the proportional-bias slope.
///
/// A constant bias and a significant slope cannot both stand: once the slope is real the bias is a
/// LINE, and [`Agreement::bias`] is then only its value at the mean. Branch on
/// [`Agreement::proportional`] before quoting anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    pub n: usize,
    /// Mean of `device - reference`. Positive means the device over-reports. **Only meaningful on
    /// its own when `!proportional`** — otherwise it is one point on a sloped line.
    pub bias: f64,
    /// Sample SD of the differences. Carries the slope's spread when one exists, so it is the WRONG
    /// scale for limits under proportional bias; use `resid_sd`.
    pub sd: f64,
    pub loa_lo: f64,
    pub loa_hi: f64,
    /// OLS slope of the difference on the REFERENCE, and the line's value at reference zero. Away
    /// from zero the error depends on the magnitude and a single bias figure describes no one.
    ///
    /// The textbook Bland-Altman slope regresses on the pair MEAN. That is deliberately NOT what
    /// this is: the mean contains half the difference, so it manufactures a negative slope from an
    /// unbiased device. Regressing on the reference is valid only because ours is a gold standard
    /// rather than a second device, and it is not carried beside this one — an invalid statistic
    /// kept next to a valid one is how the wrong one gets quoted.
    pub slope_ref: f64,
    pub intercept_ref: f64,
    /// SD of the residuals about the reference line, `n-2` degrees of freedom. The scale for limits
    /// once the slope is real; `sd` carries the slope's spread and is too wide.
    pub resid_sd: f64,
    /// `slope_ref / SE`, at `n-2` df. NaN when the reference has no spread to regress on.
    pub slope_t: f64,
    /// Whether `slope_ref` is significant at 95%. **Not** whether it is large: a tiny slope on many
    /// recordings is real, and a steep one on four is not.
    pub proportional: bool,
    /// Observed range of the REFERENCE, so a caller can quote the fitted bias at each end rather
    /// than a single number that describes neither.
    pub ref_lo: f64,
    pub ref_hi: f64,
}

impl Agreement {
    /// The fitted bias at a reference value. Equals [`Agreement::bias`] everywhere when flat.
    pub fn bias_at(&self, reference: f64) -> f64 {
        self.intercept_ref + self.slope_ref * reference
    }

    /// Limits about the fitted line. Under proportional bias these move with the measurement, which
    /// is the point; `loa_lo`/`loa_hi` are the flat answer and are only valid when `!proportional`.
    pub fn loa_at(&self, reference: f64) -> (f64, f64) {
        let half = LOA_Z * self.resid_sd;
        (self.bias_at(reference) - half, self.bias_at(reference) + half)
    }
}

/// The 95% limits-of-agreement multiplier. Normal-theory, as Bland and Altman define them.
const LOA_Z: f64 = 1.96;

/// `device` and `reference` are paired per recording and must be the same length.
pub fn bland_altman(device: &[f64], reference: &[f64]) -> Option<Agreement> {
    let n = device.len();
    if n < 2 || reference.len() != n {
        return None;
    }
    let diff: Vec<f64> = device.iter().zip(reference).map(|(d, r)| d - r).collect();
    let bias = diff.iter().sum::<f64>() / n as f64;
    let sd = (diff.iter().map(|d| (d - bias).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let syy: f64 = diff.iter().map(|d| (d - bias).powi(2)).sum();

    // Regress the difference on the REFERENCE. No spread there leaves nothing to regress on, so the
    // slope is unknown rather than zero. Residuals: SSE = Syy - Sxy^2/Sxx; t = slope/SE at n-2 df,
    // so it needs three pairs to say anything. A perfect fit divides by zero: with a real slope that
    // is maximally significant, with a flat one it is a constant offset and there is nothing to find.
    let rbar = reference.iter().sum::<f64>() / n as f64;
    let rxx: f64 = reference.iter().map(|x| (x - rbar).powi(2)).sum();
    let rxy: f64 = reference.iter().zip(&diff).map(|(x, d)| (x - rbar) * (d - bias)).sum();
    let slope_ref = if rxx > f64::EPSILON { rxy / rxx } else { f64::NAN };
    let (resid_sd, slope_t) = if rxx > f64::EPSILON && n > 2 {
        let sse = (syy - rxy * rxy / rxx).max(0.0);
        let rsd = (sse / (n - 2) as f64).sqrt();
        let se = rsd / rxx.sqrt();
        let t = match () {
            _ if se > f64::EPSILON => slope_ref / se,
            _ if slope_ref.abs() > f64::EPSILON => f64::INFINITY.copysign(slope_ref),
            _ => 0.0,
        };
        (rsd, t)
    } else {
        (sd, f64::NAN)
    };
    let proportional = !slope_t.is_nan() && slope_t.abs() > crate::stats::t95_df(n - 2);

    Some(Agreement {
        n,
        bias,
        sd,
        loa_lo: bias - LOA_Z * sd,
        loa_hi: bias + LOA_Z * sd,
        slope_ref,
        intercept_ref: bias - slope_ref * rbar,
        resid_sd,
        slope_t,
        proportional,
        ref_lo: reference.iter().copied().fold(f64::INFINITY, f64::min),
        ref_hi: reference.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20 epochs of 0.5 min: wake, 4 light, wake, 3 deep, 2 rem, then wake to the end.
    fn hypnogram() -> Vec<usize> {
        let mut v = vec![0usize];
        v.extend([1, 1, 1, 1]);
        v.push(0);
        v.extend([2, 2, 2]);
        v.extend([3, 3]);
        v.extend(std::iter::repeat_n(0usize, 20 - v.len()));
        v
    }

    #[test]
    fn the_summary_splits_latency_from_waso_from_trailing_wake() {
        let s = summarise(&hypnogram(), 0.5).expect("a labelled night");
        assert_eq!(2.0, s.light, "4 light epochs at 0.5 min");
        assert_eq!(1.5, s.deep);
        assert_eq!(1.0, s.rem);
        assert_eq!(4.5, s.tst, "tst is light + deep + rem, never the window");
        assert_eq!(0.5, s.sol, "one wake epoch before the first sleep epoch");
        assert_eq!(0.5, s.waso, "the ONE wake epoch between first and last sleep");
        // The nine trailing wake epochs are in neither: they are not latency and not WASO.
        assert_eq!(45.0, s.efficiency, "4.5 of a 10 minute window");
    }

    /// A night that never sleeps has no interval for WASO to live in, and the whole window is
    /// latency. Reporting WASO = window here would double-count the same wake.
    #[test]
    fn a_night_with_no_sleep_is_all_latency_and_no_waso() {
        let s = summarise(&[0usize; 10], 0.5).expect("labelled");
        assert_eq!(0.0, s.tst);
        assert_eq!(5.0, s.sol, "the whole window");
        assert_eq!(0.0, s.waso);
        assert_eq!(0.0, s.efficiency);
        assert_eq!(None, summarise(&[], 0.5), "no epochs is not a night");
        assert_eq!(None, summarise(&[1, 1], 0.0), "a zero-length epoch is not a scale");
    }

    /// Hand-computed: differences are +10 four times and +20 once, so bias 12 and sd 4.472.
    #[test]
    fn bland_altman_reproduces_a_hand_computed_bias_and_limits() {
        let reference = [100.0, 200.0, 300.0, 400.0, 500.0];
        let device = [110.0, 210.0, 310.0, 410.0, 520.0];
        let a = bland_altman(&device, &reference).expect("five pairs");
        assert_eq!(5, a.n);
        assert!((a.bias - 12.0).abs() < 1e-12, "bias {}", a.bias);
        assert!((a.sd - 20.0f64.sqrt()).abs() < 1e-12, "sd {}", a.sd);
        assert!((a.loa_lo - (12.0 - 1.96 * 20.0f64.sqrt())).abs() < 1e-12);
        assert!((a.loa_hi - (12.0 + 1.96 * 20.0f64.sqrt())).abs() < 1e-12);
    }

    /// The reason proportional bias is reported separately: these two have the SAME bias and the
    /// same limits, and one of them has an error that grows with the measurement.
    #[test]
    fn proportional_bias_separates_a_constant_offset_from_a_growing_one() {
        let reference = [100.0, 200.0, 300.0, 400.0, 500.0];
        let flat = [120.0, 220.0, 320.0, 420.0, 520.0];
        let grows = [100.0, 210.0, 320.0, 430.0, 540.0];

        let a = bland_altman(&flat, &reference).unwrap();
        let b = bland_altman(&grows, &reference).unwrap();
        assert!((a.bias - b.bias).abs() < 1e-9, "the two must share a bias to make the point");
        assert!(a.slope_ref.abs() < 1e-9, "a constant offset has no proportional bias");
        assert!(b.slope_ref > 0.09, "a growing error must show as a slope, got {}", b.slope_ref);
        assert!(b.proportional, "and it must resolve, t={}", b.slope_t);
    }

    /// Two degeneracies that must not be answered the same way. A constant DIFFERENCE has a real
    /// slope of zero; no spread in the REFERENCE has no slope at all. Regressing on the pair mean
    /// would tell these apart differently, which is one more reason not to.
    #[test]
    fn the_two_degenerate_cases_are_told_apart() {
        let offset = bland_altman(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(0.0, offset.slope_ref, "a constant offset is zero proportional bias, not unknown");
        assert_eq!(0.0, offset.slope_t, "a perfect flat fit is t=0, not t=inf");
        assert_eq!(1.0, offset.bias);

        // A constant reference: nothing to regress the difference on.
        let no_spread = bland_altman(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0]).unwrap();
        assert!(no_spread.slope_ref.is_nan(), "no spread in the reference cannot yield a slope");
        assert!(!no_spread.proportional, "and an unknown slope is not a resolved one");

        assert_eq!(None, bland_altman(&[1.0], &[1.0]), "one pair is not agreement");
        assert_eq!(None, bland_altman(&[1.0, 2.0], &[1.0]), "unpaired input is a caller bug");
    }

    /// The defect this branch exists for. A constant band and a real slope cannot both stand: once
    /// the slope is real the bias is a LINE, and the single number describes neither end of it.
    #[test]
    fn a_real_slope_makes_the_bias_a_line_and_the_flat_band_wrong() {
        let reference = [100.0, 200.0, 300.0, 400.0, 500.0];
        let grows = [100.0, 210.0, 320.0, 430.0, 540.0];
        let a = bland_altman(&grows, &reference).unwrap();

        assert!(a.proportional, "a perfectly proportional error must be flagged, t={}", a.slope_t);
        assert!((a.bias - 20.0).abs() < 1e-9, "the flat bias is 20 for everyone");
        // And it is right for nobody: the fitted line runs from ~0 to ~40 across the observed range.
        assert!(a.bias_at(a.ref_lo) < 2.0, "{}", a.bias_at(a.ref_lo));
        assert!(a.bias_at(a.ref_hi) > 38.0, "{}", a.bias_at(a.ref_hi));

        // The flat limits are ~+/-27 wide on a relationship with NO residual scatter at all.
        assert!(a.sd > 14.0, "the raw sd carries the slope's spread: {}", a.sd);
        assert!(a.resid_sd < 1e-6, "residual scatter about the line is nil: {}", a.resid_sd);
        let (lo, hi) = a.loa_at(a.ref_hi);
        assert!(hi - lo < 1e-5, "so the limits about the line collapse: {lo} .. {hi}");
    }

    /// It must branch on SIGNIFICANCE, not on the slope's size. A steep slope on few noisy pairs is
    /// not evidence; a shallow one on many is. Branching on `|slope| > k` fails both of these.
    #[test]
    fn the_branch_is_significance_and_not_the_slopes_magnitude() {
        // Steep (0.57) but only 3 pairs, and the middle one far off the line: t = 1.16 at df 1.
        let steep = bland_altman(&[100.0, 400.0, 400.0], &[100.0, 200.0, 300.0]).unwrap();
        // Exactly 0.5 against the reference: diff [0,200,100] on reference [100,200,300].
        assert!(steep.slope_ref >= 0.5, "must be steep to make the point: {}", steep.slope_ref);
        assert!(!steep.proportional, "3 scattered pairs cannot resolve it, t={}", steep.slope_t);

        // Shallow (0.02) but clean and over 40 pairs: resolvable.
        let refr: Vec<f64> = (0..40).map(|i| 100.0 + 10.0 * i as f64).collect();
        let dev: Vec<f64> = refr
            .iter()
            .enumerate()
            .map(|(i, r)| r + 0.02 * r + if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let shallow = bland_altman(&dev, &refr).unwrap();
        assert!(shallow.slope_ref < 0.03, "must be shallow: {}", shallow.slope_ref);
        assert!(shallow.proportional, "40 clean pairs resolve it, t={}", shallow.slope_t);
    }

    /// A flat relationship must NOT be flagged, or every measure on the card reads as sloped and the
    /// branch says nothing. Includes the perfect-fit flat case, which divides 0 by 0.
    #[test]
    fn a_flat_relationship_is_not_proportional() {
        // The noise pattern has period 4 (+3,-3,-3,+3), which is orthogonal to a linear trend over
        // each block. An alternating +/- pattern is NOT: it correlates with the pair mean and puts
        // a real slope into what is supposed to be the flat control.
        let refr: Vec<f64> = (0..32).map(|i| 100.0 + 10.0 * i as f64).collect();
        let noisy: Vec<f64> = refr
            .iter()
            .enumerate()
            .map(|(i, r)| r + 12.0 + if i % 4 == 0 || i % 4 == 3 { 3.0 } else { -3.0 })
            .collect();
        let a = bland_altman(&noisy, &refr).unwrap();
        assert!(!a.proportional, "a constant offset with noise is not proportional bias");
        // Not exactly zero, and it cannot be: the pair MEAN carries half the noise the DIFFERENCE
        // carries, so the two are coupled by construction. Judge it against the scatter it sits in
        // - the line moves 0.164 across the whole range inside noise of +/-3.
        const NOISE: f64 = 3.0;
        let span = (a.bias_at(a.ref_hi) - a.bias_at(a.ref_lo)).abs();
        assert!(span < 0.1 * NOISE, "the trend must vanish inside the scatter: moved {span}");

        // Perfect constant offset: se is 0/0. It must read flat, not infinitely significant.
        let exact = bland_altman(&[11.0, 21.0, 31.0, 41.0], &[1.0, 11.0, 21.0, 31.0]).unwrap();
        assert_eq!(0.0, exact.slope_t, "a perfect FLAT fit is t=0, not t=inf");
        assert!(!exact.proportional);
        assert_eq!(10.0, exact.bias);

        // Fewer than three pairs cannot fit a line at all, and must not claim to.
        let two = bland_altman(&[1.0, 5.0], &[0.0, 1.0]).unwrap();
        assert!(two.slope_t.is_nan() && !two.proportional, "n=2 has no residual df");
    }
}
