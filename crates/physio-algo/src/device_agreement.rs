//! Device-to-device agreement: how far apart two devices are when they measure the same thing, and
//! what fixed offset would close the gap.
//!
//! Reported, never blended. A day's scores stay attributed to one source; nothing here merges two
//! devices into one number. Bland-Altman is the method because it answers the question honestly:
//! a mean difference (bias) plus the interval most differences fall in, rather than a single
//! "accuracy" figure that hides how wide the disagreement is.
//!
//! Two devices agreeing is two devices agreeing, not proof either is right. Only a reference
//! measurement can say that, and neither of these is one.

use crate::stats::{mean, sample_sd};

/// One moment measured by both devices. [reference] is the device being compared AGAINST, so a
/// positive bias means [other] reads high.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedSample {
    pub reference: f64,
    pub other: f64,
}

/// How two devices compare over a set of paired measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    /// Paired measurements the figures rest on. Everything else is meaningless without it.
    pub n: usize,
    /// Mean of `other - reference`. Positive means `other` reads high. This is the calibratable part.
    pub bias: f64,
    /// Mean absolute difference. Unlike [bias] this does not cancel, so a device that is wild in both
    /// directions cannot look well-behaved.
    pub mean_abs: f64,
    /// Sample SD of the differences: how consistent the disagreement is. A small SD with a large bias
    /// is correctable; a large SD is not.
    pub sd: f64,
    /// Bland-Altman limits of agreement, bias +/- 1.96 SD. About 95% of differences fall inside.
    pub loa_low: f64,
    pub loa_high: f64,
}

/// Standard-normal multiplier for the 95% limits of agreement.
const LOA_Z: f64 = 1.96;

/// Compare two devices over [pairs]. `None` when there is nothing to compare.
///
/// A single pair yields a bias with `sd` and both limits at 0 — true, and useless. Read [Agreement::n]
/// before reading anything else; that is why it is the first field.
pub fn agreement(pairs: &[PairedSample]) -> Option<Agreement> {
    if pairs.is_empty() {
        return None;
    }
    let diffs: Vec<f64> = pairs.iter().map(|p| p.other - p.reference).collect();
    let bias = mean(&diffs);
    let mean_abs = mean(&diffs.iter().map(|d| d.abs()).collect::<Vec<f64>>());
    let sd = if diffs.len() < 2 { 0.0 } else { sample_sd(&diffs) };
    Some(Agreement {
        n: diffs.len(),
        bias,
        mean_abs,
        sd,
        loa_low: bias - LOA_Z * sd,
        loa_high: bias + LOA_Z * sd,
    })
}

/// Fewest paired measurements before an offset may be derived. Below this the bias is an anecdote:
/// one night's difference between two devices says nothing about the next.
pub const MIN_PAIRS_FOR_OFFSET: usize = 7;

/// Widest difference-SD an offset may be derived from, in the metric's own units, as a multiple of
/// the bias. A device that disagrees inconsistently cannot be corrected by a constant, and pretending
/// otherwise would bake noise into a stored number.
pub const MAX_SD_OVER_BIAS: f64 = 2.0;

/// Why an offset was refused, so the refusal can be shown rather than silently producing nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffsetRefusal {
    /// Fewer than [MIN_PAIRS_FOR_OFFSET] paired measurements.
    TooFewPairs,
    /// The disagreement is too inconsistent for a constant to correct.
    TooScattered,
    /// The devices already agree; correcting would add a number for no reason.
    AlreadyAgrees,
}

/// The constant that would bring `other` onto `reference`: subtract it from `other`.
///
/// Refuses rather than guessing. An offset is only meaningful when there is enough of it, it is
/// consistent, and it is worth applying — [negligible] is the metric's own "close enough" in its own
/// units, which the caller owns because it is a unit question, not a statistics one.
/// Scatter is judged BEFORE the bias size, and that order matters. Two devices that disagree by +20
/// and -20 alternately have a mean difference near zero; checking the bias first would call that
/// "already agrees", which is the worst answer available — it reports agreement precisely when the
/// devices are furthest apart. Scatter is compared against [negligible] as well as against the bias,
/// so ordinary measurement noise around a near-zero bias is not mistaken for disagreement.
pub fn offset_for(a: &Agreement, negligible: f64) -> Result<f64, OffsetRefusal> {
    if a.n < MIN_PAIRS_FOR_OFFSET {
        return Err(OffsetRefusal::TooFewPairs);
    }
    if a.sd > MAX_SD_OVER_BIAS * a.bias.abs() && a.sd > negligible {
        return Err(OffsetRefusal::TooScattered);
    }
    if a.bias.abs() <= negligible {
        return Err(OffsetRefusal::AlreadyAgrees);
    }
    Ok(a.bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(reference: &[f64], other: &[f64]) -> Vec<PairedSample> {
        reference
            .iter()
            .zip(other.iter())
            .map(|(&r, &o)| PairedSample { reference: r, other: o })
            .collect()
    }

    #[test]
    fn no_pairs_yields_nothing() {
        assert_eq!(agreement(&[]), None);
    }

    #[test]
    fn a_constant_offset_is_recovered_exactly() {
        // Every sample reads 3.0 high, with no scatter at all.
        let r = [60.0, 62.0, 58.0, 61.0, 59.0, 63.0, 57.0];
        let o: Vec<f64> = r.iter().map(|x| x + 3.0).collect();
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert_eq!(a.n, 7);
        assert!((a.bias - 3.0).abs() < 1e-9);
        assert!((a.mean_abs - 3.0).abs() < 1e-9);
        assert!(a.sd.abs() < 1e-9, "a constant offset has no scatter");
        assert!((a.loa_low - 3.0).abs() < 1e-9);
        assert!((a.loa_high - 3.0).abs() < 1e-9);
    }

    #[test]
    fn bias_cancels_but_mean_absolute_does_not() {
        // Alternating +5 / -5: the devices disagree badly, and a mean difference alone would hide it.
        let r = [60.0, 60.0, 60.0, 60.0];
        let o = [65.0, 55.0, 65.0, 55.0];
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert!(a.bias.abs() < 1e-9, "the mean difference cancels to nothing");
        assert!((a.mean_abs - 5.0).abs() < 1e-9, "the typical difference is still 5");
        assert!(a.sd > 5.0, "and the scatter says so");
    }

    #[test]
    fn limits_of_agreement_bracket_the_bias() {
        let r = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0];
        let o = [62.0, 64.0, 63.0, 66.0, 65.0, 68.0, 67.0];
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert!(a.loa_low < a.bias && a.bias < a.loa_high);
        assert!((a.loa_high - a.loa_low - 2.0 * LOA_Z * a.sd).abs() < 1e-9);
    }

    #[test]
    fn one_pair_reports_a_bias_and_no_spread() {
        let a = agreement(&pairs(&[60.0], &[64.0])).unwrap();
        assert_eq!(a.n, 1);
        assert!((a.bias - 4.0).abs() < 1e-9);
        assert_eq!(a.sd, 0.0, "one pair cannot have a spread");
    }

    #[test]
    fn an_offset_needs_enough_pairs() {
        let r = [60.0, 61.0, 62.0];
        let o: Vec<f64> = r.iter().map(|x| x + 5.0).collect();
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert_eq!(offset_for(&a, 0.5), Err(OffsetRefusal::TooFewPairs));
    }

    #[test]
    fn a_consistent_offset_is_returned() {
        let r = [60.0, 62.0, 58.0, 61.0, 59.0, 63.0, 57.0];
        let o: Vec<f64> = r.iter().map(|x| x + 4.0).collect();
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert_eq!(offset_for(&a, 0.5), Ok(4.0));
    }

    #[test]
    fn a_scattered_disagreement_is_refused() {
        // A small mean difference sitting inside huge scatter: no constant fixes this.
        let r = [60.0; 8];
        let o = [80.0, 40.0, 78.0, 42.0, 81.0, 39.0, 79.0, 42.0];
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert_eq!(offset_for(&a, 0.5), Err(OffsetRefusal::TooScattered));
    }

    #[test]
    fn devices_that_already_agree_get_no_offset() {
        let r = [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0];
        let o: Vec<f64> = r.iter().map(|x| x + 0.2).collect();
        let a = agreement(&pairs(&r, &o)).unwrap();
        assert_eq!(offset_for(&a, 0.5), Err(OffsetRefusal::AlreadyAgrees));
    }

    #[test]
    fn the_offset_direction_brings_other_onto_reference() {
        let r = [60.0, 62.0, 58.0, 61.0, 59.0, 63.0, 57.0];
        let o: Vec<f64> = r.iter().map(|x| x + 4.0).collect();
        let a = agreement(&pairs(&r, &o)).unwrap();
        let offset = offset_for(&a, 0.5).unwrap();
        let corrected: Vec<f64> = o.iter().map(|x| x - offset).collect();
        let after = agreement(&pairs(&r, &corrected)).unwrap();
        assert!(after.bias.abs() < 1e-9, "subtracting the offset closes the gap");
    }
}
