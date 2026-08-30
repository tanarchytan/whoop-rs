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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    pub n: usize,
    /// Mean of `device - reference`. Positive means the device over-reports.
    pub bias: f64,
    /// Sample SD of the differences.
    pub sd: f64,
    pub loa_lo: f64,
    pub loa_hi: f64,
    /// OLS slope of the difference on the pair mean. Away from zero, the error depends on the
    /// magnitude, and a single bias figure describes no one.
    pub slope: f64,
    /// Correlation between the difference and the pair mean; the slope's strength.
    pub r: f64,
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
    let mean: Vec<f64> = device.iter().zip(reference).map(|(d, r)| (d + r) / 2.0).collect();

    let bias = diff.iter().sum::<f64>() / n as f64;
    let sd = (diff.iter().map(|d| (d - bias).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

    let mbar = mean.iter().sum::<f64>() / n as f64;
    let sxx: f64 = mean.iter().map(|m| (m - mbar).powi(2)).sum();
    let sxy: f64 = mean.iter().zip(&diff).map(|(m, d)| (m - mbar) * (d - bias)).sum();
    let syy: f64 = diff.iter().map(|d| (d - bias).powi(2)).sum();
    // Two different degeneracies. No spread in the MEANS leaves nothing to regress on, so the slope
    // is unknown. Constant DIFFERENCES give a real slope of zero, and only the correlation is 0/0.
    let slope = if sxx > f64::EPSILON { sxy / sxx } else { f64::NAN };
    let r = if sxx > f64::EPSILON && syy > f64::EPSILON {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    };

    Some(Agreement { n, bias, sd, loa_lo: bias - LOA_Z * sd, loa_hi: bias + LOA_Z * sd, slope, r })
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
        assert!(a.slope.abs() < 1e-9, "a constant offset has no proportional bias");
        assert!(b.slope > 0.09, "a growing error must show as a slope, got {}", b.slope);
        assert!(b.r > 0.99, "and it must be a strong one, got {}", b.r);
    }

    /// Two degeneracies that must not be answered the same way. A constant DIFFERENCE has a real
    /// slope of zero and no correlation; no spread in the MEANS has no slope at all.
    #[test]
    fn the_two_degenerate_cases_are_told_apart() {
        let offset = bland_altman(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(0.0, offset.slope, "a constant offset is zero proportional bias, not unknown");
        assert!(offset.r.is_nan(), "but its correlation is 0/0");
        assert_eq!(1.0, offset.bias);

        // Every pair means 1.5, so there is nothing to regress the difference on.
        let no_spread = bland_altman(&[1.0, 2.0, 3.0], &[2.0, 1.0, 0.0]).unwrap();
        assert!(no_spread.slope.is_nan(), "no spread in the means cannot yield a slope");

        assert_eq!(None, bland_altman(&[1.0], &[1.0]), "one pair is not agreement");
        assert_eq!(None, bland_altman(&[1.0, 2.0], &[1.0]), "unpaired input is a caller bug");
    }
}
