//! Rest (sleep performance) composite, 0–100. Weighted sum of duration vs personal need, efficiency,
//! restorative (deep+REM) minutes against need, and sleep/wake consistency. Pure; absent consistency
//! defaults neutral.
//!
//! The restorative term measures MINUTES against the need, not the share of whatever was slept. As a
//! share it was duration-blind: a short night with ordinary architecture cleared the target and took
//! full marks, so three of the four terms could sit near maximum on a night that was simply too short.
//! Measured on 13 nights scored by both this composite and WHOOP's own: the share form ran +11.1 mean
//! (high on 13 of 13), the minutes form +7.4. Still high — the remainder is not in this term.

pub const W_DURATION: f64 = 0.50;
pub const W_EFFICIENCY: f64 = 0.20;
pub const W_RESTORATIVE: f64 = 0.20;
pub const W_CONSISTENCY: f64 = 0.10;
pub const DEFAULT_SLEEP_NEED_HOURS: f64 = 8.0;
pub const RESTORATIVE_TARGET_SHARE: f64 = 0.50;
pub const DEEP_SHARE_TARGET: f64 = 0.13;
pub const DEEP_FLOOR_FACTOR: f64 = 0.5;
pub const NEUTRAL_CONSISTENCY: f64 = 0.5;

/// Rest composite, or `None` when there is no asleep time.
pub fn rest(
    asleep_seconds: f64,
    efficiency: f64,
    deep_seconds: f64,
    rem_seconds: f64,
    sleep_need_hours: Option<f64>,
    consistency: Option<f64>,
) -> Option<f64> {
    if asleep_seconds <= 0.0 {
        return None;
    }
    let asleep_hours = asleep_seconds / 3600.0;
    let need_hours = sleep_need_hours.unwrap_or(DEFAULT_SLEEP_NEED_HOURS).max(1e-9);

    let duration_score = (asleep_hours / need_hours * 100.0).min(100.0);
    let efficiency_score = (efficiency * 100.0).clamp(0.0, 100.0);
    // Minutes against the need, so a short night cannot clear the target on architecture alone.
    let restorative_target_seconds = need_hours * 3600.0 * RESTORATIVE_TARGET_SHARE;
    let deep_adequacy = ((deep_seconds / asleep_seconds) / DEEP_SHARE_TARGET).clamp(0.0, 1.0);
    let deep_factor = DEEP_FLOOR_FACTOR + (1.0 - DEEP_FLOOR_FACTOR) * deep_adequacy;
    let restorative_score =
        ((deep_seconds + rem_seconds) / restorative_target_seconds * 100.0).min(100.0) * deep_factor;
    let consistency_score = ((consistency.unwrap_or(NEUTRAL_CONSISTENCY)) * 100.0).clamp(0.0, 100.0);

    let weighted = W_DURATION * duration_score
        + W_EFFICIENCY * efficiency_score
        + W_RESTORATIVE * restorative_score
        + W_CONSISTENCY * consistency_score;
    Some((weighted * 100.0).round() / 100.0)
}

/// Healthy floor for the personal sleep-need estimate (hours).
pub const MIN_SLEEP_NEED_HOURS: f64 = 7.5;

/// Personal sleep need (hours) = the mean of recent nightly asleep hours, floored at [MIN_SLEEP_NEED_HOURS].
/// Non-positive nights are ignored; an empty window returns the floor. Feeds `rest`'s `sleep_need_hours`.
pub fn personal_sleep_need_hours(recent_asleep_hours: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u32;
    for &h in recent_asleep_hours {
        if h > 0.0 {
            sum += h;
            n += 1;
        }
    }
    if n == 0 {
        return MIN_SLEEP_NEED_HOURS;
    }
    (sum / n as f64).max(MIN_SLEEP_NEED_HOURS)
}

/// How many equal tiers a Rest driver's 0-100 is read in; the word and swatch per tier are the caller's.
pub const DRIVER_TIERS: u32 = 3;

/// Which tier a driver's 0-100 `percent` falls in, counting from 0. Evenly split, top tier closed at
/// 100; anything above the scale, below it, or not a number clamps into range.
pub fn driver_tier(percent: f64) -> u32 {
    let raw = (percent / 100.0 * DRIVER_TIERS as f64) as i64;
    raw.clamp(0, DRIVER_TIERS as i64 - 1) as u32
}

/// The tier a driver LIGHTS: its own [`driver_tier`], mirrored end-for-end when `higher_is_better` is
/// false because that driver's 0 is the good end. Only which tier lights moves, never the value.
pub fn driver_tier_lit(percent: f64, higher_is_better: bool) -> u32 {
    let tier = driver_tier(percent);
    if higher_is_better {
        tier
    } else {
        DRIVER_TIERS - 1 - tier
    }
}

/// Where a tier sits on a 0..1 ramp — bottom at 0, top at 1 — so the tier swatches sample one scale.
pub fn driver_tier_position(tier: u32) -> f64 {
    tier.min(DRIVER_TIERS - 1) as f64 / (DRIVER_TIERS - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in sleep-need estimator, so a do-nothing one can be run against the same rows.
    type NullNeed = fn(&[f64]) -> f64;

    /// The need is the mean of the WORN nights only, floored. `[9, 0, 9]` separates that from a plain
    /// mean over the whole window: ignoring the blank gives 9.0, counting it gives 6.0 and floors to 7.5.
    #[test]
    fn personal_need_averages_worn_nights_only_and_floors_at_seven_and_a_half() {
        assert_eq!(personal_sleep_need_hours(&[]), MIN_SLEEP_NEED_HOURS);
        assert_eq!(personal_sleep_need_hours(&[6.0, 0.0, 6.5]), MIN_SLEEP_NEED_HOURS); // mean 6.25 < floor
        assert!((personal_sleep_need_hours(&[8.0, 9.0, 8.5]) - 8.5).abs() < 1e-9); // mean above the floor
        assert!((personal_sleep_need_hours(&[9.0, 0.0, 9.0]) - 9.0).abs() < 1e-9); // the blank is not a night

        // Every do-nothing need disagrees with at least one row above, so none of them can pass this test.
        let cases: [&[f64]; 4] = [&[], &[6.0, 0.0, 6.5], &[8.0, 9.0, 8.5], &[9.0, 0.0, 9.0]];
        let want = [MIN_SLEEP_NEED_HOURS, MIN_SLEEP_NEED_HOURS, 8.5, 9.0];
        let nulls: [(&str, NullNeed); 3] = [
            ("always the floor", |_| MIN_SLEEP_NEED_HOURS),
            ("always the default", |_| DEFAULT_SLEEP_NEED_HOURS),
            ("plain mean, blanks counted", |v| {
                if v.is_empty() {
                    MIN_SLEEP_NEED_HOURS
                } else {
                    (v.iter().sum::<f64>() / v.len() as f64).max(MIN_SLEEP_NEED_HOURS)
                }
            }),
        ];
        for (name, f) in nulls {
            assert!(
                cases.iter().zip(want).any(|(c, w)| (f(c) - w).abs() > 1e-9),
                "the '{name}' scorer reproduces every row, so this gate is blind to it"
            );
        }
    }

    /// The need reaches the composite through TWO terms now: duration, and the restorative target it
    /// scales. A change in need moves Rest by the sum of both, which is what makes a short night unable
    /// to earn restorative credit it did not sleep for.
    #[test]
    fn the_personal_need_enters_rest_through_duration_and_restorative() {
        let night = |need: f64| {
            rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(need), Some(1.0)).unwrap()
        };
        let at_need = personal_sleep_need_hours(&[9.0, 0.0, 9.0]); // 9.0 h
        let floored = personal_sleep_need_hours(&[6.0, 0.0, 6.5]); // 7.5 h, so duration saturates at 100
        assert!((at_need - 9.0).abs() < 1e-9 && (floored - MIN_SLEEP_NEED_HOURS).abs() < 1e-9);
        // 8 h against a 9 h need scores 88.888…; against 7.5 h it saturates at 100.
        let duration_delta = 100.0 - (8.0 / 9.0 * 100.0);
        // 4.4 h of deep+REM clears a 7.5 h need's 3.75 h target but not a 9 h need's 4.5 h target.
        let restorative_delta = 100.0 - (4.4 / 4.5 * 100.0);
        let moved = W_DURATION * duration_delta + W_RESTORATIVE * restorative_delta;
        assert!((night(floored) - night(at_need) - moved).abs() < 0.01);
        // Duration alone no longer accounts for it: that is the change.
        assert!((night(floored) - night(at_need) - W_DURATION * duration_delta).abs() > 0.1);
        assert!(night(floored) > night(at_need), "a lower need cannot score a night worse");
    }

    /// Rest is exactly the four weighted terms. The perfect night is 99.0 = 0.5*100 + 0.2*95 + 0.2*100 +
    /// 0.1*100, and moving one input moves the total by that weight times the term it changed.
    #[test]
    fn each_rest_weight_moves_the_composite_by_its_own_share() {
        let perfect = rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert_eq!(99.0, perfect);
        // Six hours against the same eight-hour need. Duration 100 -> 75, AND restorative 100 -> 82.5,
        // because 3.3 h of deep+REM no longer clears the need's 4 h target. A short night loses on both,
        // which is the whole point of scoring restorative in minutes rather than as a share.
        let short = rest(6.0 * 3600.0, 0.95, 0.30 * 6.0 * 3600.0, 0.25 * 6.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        let short_moved = W_DURATION * 25.0 + W_RESTORATIVE * (100.0 - 82.5);
        assert!((perfect - short - short_moved).abs() < 1e-9, "short night, got {short}");
        assert!(short < perfect - W_DURATION * 25.0, "a short night must lose more than duration alone");
        // Efficiency 95 -> 85.
        let leaky = rest(8.0 * 3600.0, 0.85, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert!((perfect - leaky - W_EFFICIENCY * 10.0).abs() < 1e-9, "efficiency weight, got {leaky}");
        // Restorative 100 -> 50: a 0.25 deep+REM share against the 0.50 target, deep still at its own.
        let thin = rest(8.0 * 3600.0, 0.95, 0.13 * 8.0 * 3600.0, 0.12 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        assert!((perfect - thin - W_RESTORATIVE * 50.0).abs() < 1e-9, "restorative weight, got {thin}");
        // Consistency 100 -> 0.
        let erratic = rest(8.0 * 3600.0, 0.95, 0.30 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(0.0)).unwrap();
        assert!((perfect - erratic - W_CONSISTENCY * 100.0).abs() < 1e-9, "consistency weight, got {erratic}");

        assert!((1.0 - (W_DURATION + W_EFFICIENCY + W_RESTORATIVE + W_CONSISTENCY)).abs() < 1e-12);
        // No constant scorer reproduces the five nights. The thin and erratic ones both land on 89.0, so
        // that pair alone cannot separate the two weights; each is pinned by its own delta above.
        let all = [perfect, short, leaky, thin, erratic];
        assert_eq!(4, {
            let mut d: Vec<f64> = all.to_vec();
            d.sort_by(|a, b| a.partial_cmp(b).expect("rest returns finite scores"));
            d.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            d.len()
        });
        for c in all {
            assert!(all.iter().any(|v| (v - c).abs() > 1e-9), "the constant {c} would satisfy every night");
        }
    }

    /// Pins DEFAULT_SLEEP_NEED_HOURS and DEEP_FLOOR_FACTOR. A mutation sweep moved both 10% with
    /// nothing noticing: every other Rest test passes `Some(8.0)` for the need, so the default is
    /// never exercised, and none of them puts deep at zero AND checks the exact halving.
    #[test]
    fn the_default_need_and_the_deep_floor_are_where_they_say_they_are() {
        assert_eq!(DEFAULT_SLEEP_NEED_HOURS, 8.0);
        assert_eq!(DEEP_FLOOR_FACTOR, 0.5);

        // Omitting the need must equal passing the default, and must NOT equal passing anything else.
        let night = |need: Option<f64>| {
            rest(7.0 * 3600.0, 0.90, 0.20 * 7.0 * 3600.0, 0.20 * 7.0 * 3600.0, need, Some(0.8)).unwrap()
        };
        assert_eq!(night(None), night(Some(DEFAULT_SLEEP_NEED_HOURS)), "None must take the default");
        assert!((night(None) - night(Some(DEFAULT_SLEEP_NEED_HOURS * 1.1))).abs() > 1e-9,
            "a different need must give a different score, or the default is unpinned");

        // Deep at zero takes the restorative term to exactly DEEP_FLOOR_FACTOR of its deep-adequate
        // value, so the gap between them pins the constant rather than merely its direction.
        let rem_only = rest(8.0 * 3600.0, 0.95, 0.0, 0.50 * 8.0 * 3600.0, Some(8.0), Some(1.0)).unwrap();
        let deep_ok = rest(
            8.0 * 3600.0, 0.95, DEEP_SHARE_TARGET * 8.0 * 3600.0,
            0.50 * 8.0 * 3600.0 - DEEP_SHARE_TARGET * 8.0 * 3600.0, Some(8.0), Some(1.0),
        )
        .unwrap();
        let restorative_full = 100.0;
        let expected_gap = W_RESTORATIVE * restorative_full * (1.0 - DEEP_FLOOR_FACTOR);
        assert!((deep_ok - rem_only - expected_gap).abs() < 1e-9,
            "zero deep must cost exactly (1 - DEEP_FLOOR_FACTOR) of the restorative weight:              {deep_ok} - {rem_only} != {expected_gap}");
    }

    #[test]
    fn short_night_scores_low() {
        // 4h asleep, 80% efficiency, same stage proportions
        let r = rest(4.0 * 3600.0, 0.80, 0.30 * 4.0 * 3600.0, 0.25 * 4.0 * 3600.0, Some(8.0), Some(0.5));
        let v = r.unwrap();
        assert!(v < 70.0 && v > 30.0, "got {v}");
    }

    #[test]
    fn zero_deep_halves_restorative() {
        // No deep sleep → deepFactor = 0.5 → restorative term halved
        let normal = rest(8.0 * 3600.0, 0.90, 0.25 * 8.0 * 3600.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0));
        let no_deep = rest(8.0 * 3600.0, 0.90, 0.0, 0.25 * 8.0 * 3600.0, Some(8.0), Some(1.0));
        assert!(normal.unwrap() > no_deep.unwrap());
    }

    #[test]
    fn no_asleep_returns_null() {
        assert!(rest(0.0, 0.90, 3600.0, 3600.0, None, None).is_none());
        assert!(rest(-1.0, 0.90, 3600.0, 3600.0, None, None).is_none());
    }

    #[test]
    fn consistency_defaults_neutral() {
        let with_neutral = rest(8.0 * 3600.0, 0.90, 7200.0, 7200.0, Some(8.0), Some(0.5));
        let with_default = rest(8.0 * 3600.0, 0.90, 7200.0, 7200.0, Some(8.0), None);
        assert!((with_neutral.unwrap() - with_default.unwrap()).abs() < 0.01);
    }

    /// The exact tier the Android strip drew before the rule moved here, so the port is provably
    /// behaviour-preserving rather than merely plausible.
    #[test]
    fn driver_tiers_split_the_range_in_thirds() {
        assert_eq!(driver_tier(0.0), 0);
        assert_eq!(driver_tier(33.0), 0);
        assert_eq!(driver_tier(34.0), 1);
        assert_eq!(driver_tier(66.0), 1);
        assert_eq!(driver_tier(67.0), 2);
        assert_eq!(driver_tier(100.0), 2);
        assert_eq!(driver_tier(140.0), 2, "above the scale still reads as the top tier");
    }

    /// A percent that is negative or not a number reads as the bottom tier, never as a panic or a
    /// tier the swatch list cannot index.
    #[test]
    fn driver_tier_floors_a_value_off_the_scale() {
        assert_eq!(driver_tier(-10.0), 0);
        assert_eq!(driver_tier(f64::NAN), 0);
        assert_eq!(driver_tier(f64::NEG_INFINITY), 0);
        assert_eq!(driver_tier(f64::INFINITY), DRIVER_TIERS - 1);
    }

    /// A driver whose 0 is the good end lights the mirrored tier; the tier index itself is unmoved.
    #[test]
    fn a_lower_is_better_driver_lights_the_mirrored_tier() {
        assert_eq!(driver_tier_lit(10.0, true), 0);
        assert_eq!(driver_tier_lit(10.0, false), 2);
        assert_eq!(driver_tier_lit(50.0, true), 1);
        assert_eq!(driver_tier_lit(50.0, false), 1);
        assert_eq!(driver_tier_lit(95.0, true), 2);
        assert_eq!(driver_tier_lit(95.0, false), 0);
    }

    /// The three tiers spread across the whole ramp, so the legend reads as one scale end to end.
    #[test]
    fn tier_positions_span_the_ramp() {
        assert_eq!(driver_tier_position(0), 0.0);
        assert_eq!(driver_tier_position(1), 0.5);
        assert_eq!(driver_tier_position(2), 1.0);
        assert_eq!(driver_tier_position(99), 1.0, "a tier past the top clamps rather than overshoots");
    }
}
