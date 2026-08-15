//! Wellness blood pressure from single-site PPG, and only ever after a cuff has said what the
//! numbers mean.
//!
//! **A cuff calibration is mandatory. Without one this module returns no number at all.** That is
//! stricter than population-anchored alternatives, deliberately: a population model reads heart rate
//! as if it were pressure, so a person with a naturally high resting heart rate is told their
//! pressure is high when it is not. Measured on this project's own band: 135/87 estimated against a
//! true value near 120. A number that wrong is worse than no number.
//!
//! **Wellness only, never medical.** Single-site PPG cannot be a sphygmomanometer: it has no second
//! site, so no true pulse transit time, and at these sample rates the dicrotic notch that diastole
//! leans on is mostly unresolvable. Diastolic is the weaker of the pair and says so.

use crate::stats::{linear_fit, mean, sample_sd};

/// Pulse-shape features from one still capture window. The inputs a calibration maps onto pressure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpFeatures {
    /// Heart rate for the window, from the strap's own per-second figure rather than from the
    /// optical trace: the strap's is motion-robust, and optical autocorrelation locks onto motion.
    pub hr: f64,
    /// Systolic upstroke time (s), foot to peak. The most calibratable shape feature at these rates.
    pub upstroke_s: f64,
    /// Pulse width at half height (s).
    pub width_s: f64,
    /// Perfusion index, AC over DC.
    pub perfusion: f64,
    /// Beats the window actually contributed. Few beats means a thin estimate.
    pub n_beats: usize,
}

/// How far a window can be trusted, and why not when it cannot.
#[derive(Clone, Debug, PartialEq)]
pub enum BpResult {
    /// No number, and the reason to show the user.
    NotReady(String),
    Estimated(BpEstimate),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpEstimate {
    pub systolic: u16,
    pub diastolic: u16,
    /// SIGNAL confidence 0-100 for the window. **Not** the accuracy of the pressure: a perfectly
    /// clean trace still yields a wellness estimate. Never label this "accuracy" in any surface.
    pub signal_confidence: f64,
    /// Always true. There is no path through this module that produces a medical measurement.
    pub wellness_only: bool,
}

/// The per-user map from pulse shape to pressure, learned from cuff readings.
#[derive(Clone, Debug, PartialEq)]
pub struct BpCal {
    /// systolic = sbp_scale * inverse-upstroke + sbp_offset
    pub sbp_scale: f64,
    pub sbp_offset: f64,
    pub dbp_scale: f64,
    pub dbp_offset: f64,
    /// Cuff pairs the fit rests on.
    pub n: usize,
    /// Correlation of the systolic fit. Low means the shape did not track the cuff and the
    /// calibration should not be trusted.
    pub sbp_r: f64,
}

/// Fewest cuff readings a calibration may be built from. Two points define a line and prove nothing;
/// the spread across a range is what separates a real relationship from an accident.
pub const MIN_CUFF_READINGS: usize = 3;

/// Weakest systolic correlation a calibration may be kept at. Below this the pulse shape did not
/// track the cuff, and fitting a line to it would dress noise as a measurement.
pub const MIN_CAL_R: f64 = 0.5;

/// Narrowest cuff systolic spread (mmHg) worth fitting. Readings all taken at rest describe one
/// point; a line through them extrapolates on nothing.
pub const MIN_CUFF_SPREAD: f64 = 8.0;

/// Physiological clamps. A value outside these is not refused, it is impossible from this method.
pub const SBP_MIN: f64 = 80.0;
pub const SBP_MAX: f64 = 200.0;
pub const DBP_MIN: f64 = 40.0;
pub const DBP_MAX: f64 = 130.0;

/// Lowest signal confidence that may produce a number.
pub const MIN_SIGNAL_CONFIDENCE: f64 = 60.0;

/// Fewest beats in a window.
pub const MIN_BEATS: usize = 15;

/// Why a calibration was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalRefusal {
    TooFewReadings,
    TooNarrowRange,
    ShapeDoesNotTrack,
}

/// Build a per-user calibration from simultaneous (features, cuff systolic, cuff diastolic) triples.
///
/// The predictor is the INVERSE upstroke time: a stiffer, higher-pressure arterial tree gives a
/// faster upstroke, so pressure rises as upstroke time falls. Using the inverse keeps the fitted
/// relationship a straight line over the range a person actually spans.
pub fn fit_user(readings: &[(BpFeatures, u16, u16)]) -> Result<BpCal, CalRefusal> {
    if readings.len() < MIN_CUFF_READINGS {
        return Err(CalRefusal::TooFewReadings);
    }
    let sbps: Vec<f64> = readings.iter().map(|(_, s, _)| *s as f64).collect();
    let dbps: Vec<f64> = readings.iter().map(|(_, _, d)| *d as f64).collect();
    let spread = sbps.iter().cloned().fold(f64::MIN, f64::max) - sbps.iter().cloned().fold(f64::MAX, f64::min);
    if spread < MIN_CUFF_SPREAD {
        return Err(CalRefusal::TooNarrowRange);
    }
    let predictor: Vec<f64> = readings
        .iter()
        .map(|(f, _, _)| if f.upstroke_s > 0.0 { 1.0 / f.upstroke_s } else { 0.0 })
        .collect();

    let sfit = linear_fit(&predictor, &sbps).ok_or(CalRefusal::ShapeDoesNotTrack)?;
    // Signed, not absolute. A stiffer, higher-pressure arterial tree gives a FASTER upstroke, so
    // pressure must rise with the inverse upstroke. A strong negative correlation is not a
    // calibration pointing the other way, it is a sign the captures and the cuff readings do not
    // belong to each other — mismatched pairs, a moving wrist, a cuff read at the wrong moment.
    // Accepting it on magnitude alone would fit a confident line through exactly that mistake.
    if sfit.r < MIN_CAL_R {
        return Err(CalRefusal::ShapeDoesNotTrack);
    }
    let dfit = linear_fit(&predictor, &dbps).ok_or(CalRefusal::ShapeDoesNotTrack)?;

    Ok(BpCal {
        sbp_scale: sfit.scale,
        sbp_offset: sfit.offset,
        dbp_scale: dfit.scale,
        dbp_offset: dfit.offset,
        n: readings.len(),
        sbp_r: sfit.r,
    })
}

/// Signal confidence 0-100 for a window: enough beats, beating regularly, and a perfusion the optics
/// can actually see. Deliberately conservative — a low score costs a retry, a false high costs a
/// wrong number on a health surface.
pub fn signal_confidence(f: &BpFeatures, beat_interval_cv: f64) -> f64 {
    if f.n_beats < MIN_BEATS {
        return 0.0;
    }
    // Beat-to-beat regularity dominates: an irregular window means the foot/peak marks are landing
    // on noise, whatever the other terms say.
    let regularity = (1.0 - (beat_interval_cv / 0.25)).clamp(0.0, 1.0);
    let beats = ((f.n_beats as f64 - MIN_BEATS as f64) / 30.0).clamp(0.0, 1.0);
    let perfusion = (f.perfusion / 0.02).clamp(0.0, 1.0);
    100.0 * (0.6 * regularity + 0.2 * beats + 0.2 * perfusion)
}

/// Estimate pressure for a window. **Returns `NotReady` unless a cuff calibration exists.**
pub fn estimate(f: &BpFeatures, cal: Option<&BpCal>, beat_interval_cv: f64) -> BpResult {
    let Some(cal) = cal else {
        return BpResult::NotReady(
            "Calibrate against a blood-pressure cuff first. Without it this can only guess from heart \
             rate, which reads a fast pulse as high pressure."
                .to_string(),
        );
    };
    if f.upstroke_s <= 0.0 {
        return BpResult::NotReady("No usable pulse shape in this window.".to_string());
    }
    let confidence = signal_confidence(f, beat_interval_cv);
    if confidence < MIN_SIGNAL_CONFIDENCE {
        return BpResult::NotReady(
            "Signal too noisy. Sit still with the band snug and try again.".to_string(),
        );
    }

    let predictor = 1.0 / f.upstroke_s;
    let systolic = (cal.sbp_scale * predictor + cal.sbp_offset).clamp(SBP_MIN, SBP_MAX);
    let diastolic = (cal.dbp_scale * predictor + cal.dbp_offset).clamp(DBP_MIN, DBP_MAX);
    if diastolic >= systolic {
        return BpResult::NotReady("The estimate did not come out physiological.".to_string());
    }
    BpResult::Estimated(BpEstimate {
        systolic: systolic.round() as u16,
        diastolic: diastolic.round() as u16,
        signal_confidence: confidence,
        wellness_only: true,
    })
}

/// Coefficient of variation of the beat-to-beat intervals, the regularity term
/// [signal_confidence] takes. 0 when there is nothing to measure.
pub fn beat_interval_cv(intervals_s: &[f64]) -> f64 {
    if intervals_s.len() < 2 {
        return 0.0;
    }
    let m = mean(intervals_s);
    if m <= 0.0 {
        return 0.0;
    }
    sample_sd(intervals_s) / m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(upstroke_s: f64, hr: f64) -> BpFeatures {
        BpFeatures { hr, upstroke_s, width_s: 0.24, perfusion: 0.03, n_beats: 40 }
    }

    /// Three cuff readings spanning a real range, where a faster upstroke goes with higher pressure.
    fn cuff_set() -> Vec<(BpFeatures, u16, u16)> {
        vec![
            (feat(0.400, 62.0), 110, 70),
            (feat(0.320, 78.0), 125, 80),
            (feat(0.270, 95.0), 140, 90),
        ]
    }

    #[test]
    fn without_a_calibration_there_is_no_number() {
        let r = estimate(&feat(0.32, 70.0), None, 0.05);
        match r {
            BpResult::NotReady(reason) => assert!(reason.contains("cuff"), "reason names the cuff"),
            BpResult::Estimated(_) => panic!("estimated without a calibration"),
        }
    }

    #[test]
    fn a_calibration_needs_more_than_two_readings() {
        let one = vec![(feat(0.32, 70.0), 120, 80)];
        assert_eq!(fit_user(&one), Err(CalRefusal::TooFewReadings));
        let two = vec![(feat(0.32, 70.0), 120, 80), (feat(0.28, 80.0), 132, 85)];
        assert_eq!(fit_user(&two), Err(CalRefusal::TooFewReadings));
    }

    #[test]
    fn readings_all_at_one_pressure_are_refused() {
        // Three readings, but every cuff value the same: nothing to fit a slope to.
        let flat = vec![
            (feat(0.40, 62.0), 120, 80),
            (feat(0.32, 78.0), 121, 80),
            (feat(0.27, 95.0), 122, 81),
        ];
        assert_eq!(fit_user(&flat), Err(CalRefusal::TooNarrowRange));
    }

    #[test]
    fn a_shape_that_does_not_track_the_cuff_is_refused() {
        // Upstroke wanders with no relation to the cuff values.
        let noisy = vec![
            (feat(0.32, 70.0), 110, 70),
            (feat(0.33, 71.0), 145, 92),
            (feat(0.315, 70.5), 118, 74),
            (feat(0.325, 70.2), 138, 88),
        ];
        assert_eq!(fit_user(&noisy), Err(CalRefusal::ShapeDoesNotTrack));
    }

    #[test]
    fn a_calibration_reproduces_the_cuff_readings_it_was_built_from() {
        let cal = fit_user(&cuff_set()).unwrap();
        for (f, sbp, dbp) in cuff_set() {
            match estimate(&f, Some(&cal), 0.05) {
                BpResult::Estimated(e) => {
                    assert!(
                        (e.systolic as i32 - sbp as i32).abs() <= 3,
                        "systolic {} vs cuff {}", e.systolic, sbp,
                    );
                    assert!(
                        (e.diastolic as i32 - dbp as i32).abs() <= 3,
                        "diastolic {} vs cuff {}", e.diastolic, dbp,
                    );
                }
                BpResult::NotReady(r) => panic!("refused a calibrated window: {r}"),
            }
        }
    }

    /// The lesson from the withdrawn PPG-HR estimate: matching once proves nothing. It has to TRACK.
    #[test]
    fn the_estimate_tracks_a_varying_input_rather_than_matching_once() {
        let cal = fit_user(&cuff_set()).unwrap();
        let mut previous = 0u16;
        // Upstroke shortening across the range must move systolic monotonically upward.
        for upstroke in [0.42, 0.38, 0.34, 0.30, 0.26] {
            match estimate(&feat(upstroke, 70.0), Some(&cal), 0.05) {
                BpResult::Estimated(e) => {
                    assert!(
                        e.systolic > previous,
                        "systolic did not rise as upstroke shortened: {} then {}", previous, e.systolic,
                    );
                    previous = e.systolic;
                }
                BpResult::NotReady(r) => panic!("refused a clean window: {r}"),
            }
        }
    }

    #[test]
    fn a_noisy_window_is_refused_rather_than_guessed() {
        let cal = fit_user(&cuff_set()).unwrap();
        // Beat intervals scattered by a quarter of their own length: the marks are on noise.
        let r = estimate(&feat(0.32, 78.0), Some(&cal), 0.40);
        assert!(matches!(r, BpResult::NotReady(_)), "a noisy window must not produce a number");
    }

    #[test]
    fn too_few_beats_is_no_confidence_at_all() {
        let thin = BpFeatures { n_beats: 5, ..feat(0.32, 78.0) };
        assert_eq!(signal_confidence(&thin, 0.05), 0.0);
    }

    #[test]
    fn every_estimate_is_marked_wellness_only() {
        let cal = fit_user(&cuff_set()).unwrap();
        match estimate(&feat(0.32, 78.0), Some(&cal), 0.05) {
            BpResult::Estimated(e) => assert!(e.wellness_only),
            BpResult::NotReady(r) => panic!("refused: {r}"),
        }
    }

    #[test]
    fn beat_interval_cv_is_zero_without_enough_beats() {
        assert_eq!(beat_interval_cv(&[]), 0.0);
        assert_eq!(beat_interval_cv(&[0.8]), 0.0);
    }

    #[test]
    fn beat_interval_cv_rises_with_scatter() {
        let steady = beat_interval_cv(&[0.80, 0.81, 0.79, 0.80]);
        let scattered = beat_interval_cv(&[0.60, 1.00, 0.65, 1.05]);
        assert!(scattered > steady * 5.0, "steady {steady} vs scattered {scattered}");
    }
}
