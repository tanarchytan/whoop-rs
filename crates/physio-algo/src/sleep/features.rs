//! The tanv1 feature vector: one named, fitted-model-ready record per epoch.
//!
//! v2 reads seven features and computes each over ONE window. The MESA benchmark that measures our
//! modality uses **370 actigraphy features over sliding windows from 30 s to 10 min**, and the
//! systematic review is explicit that a 30-second segment does not carry enough cardiorespiratory
//! information - a 4.5-minute centred window does. Our cardiac features already respect that; our
//! motion features do not, and that is the largest cheap gap we have.
//!
//! So this is not "more features" for its own sake. It is the same small set of physical quantities
//! read at the timescales the literature says they live at, plus the interaction terms v2 hard-codes
//! as branches.
//!
//! Everything here is descriptive: no thresholds, no scoring, no stage decision. [`Features::NAMES`]
//! is index-for-index with [`Features::values`] so a fitted weight vector can never silently
//! transpose against the wrong column - the failure mode that makes a fitted model unreviewable.

use super::input::AccelSample;
use super::posture::{posture_series, turn_series, Posture};

/// Epoch length. Everything windowed is centred on the epoch, never trailing, so a feature does not
/// lead or lag the label it is scored against.
pub const EPOCH_S: i64 = 30;

/// Absolute floor on the stillness scale, in g.
///
/// A per-night median collapses on a quiet night - a mostly-still night has a median inter-second
/// gravity delta of ZERO, and every ratio against it becomes 0 or infinite. That happened here on
/// first write, and it is the same failure the swing quantile hit on a real night the same day. A
/// physical floor is defensible because gravity is measured in g on an absolute scale, unlike a
/// per-wearer arbitrary unit: sleeping deltas sit far below this and any real movement above it.
pub const STILL_SCALE_FLOOR_G: f64 = 0.01;

/// Centred window half-widths in seconds: 30 s, 2, 5 and 10 minutes.
///
/// The short end is the epoch itself, the long end is where the MESA benchmark's actigraphy features
/// stop. Four is enough to see a trend without making the vector mostly redundant.
pub const WINDOWS_S: [i64; 4] = [30, 120, 300, 600];

/// One epoch, fully described. `None` is "not measurable here", which a fitted model must be handed
/// as an explicit missing-indicator rather than as a zero - a zero is a claim of no movement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Features {
    /// Epoch start, unix seconds.
    pub start: i64,
    /// Per-window motion energy: mean, max and the fraction of seconds above this night's median.
    /// Indexed by [`WINDOWS_S`].
    pub motion_mean: [Option<f64>; 4],
    pub motion_max: [Option<f64>; 4],
    pub motion_frac: [Option<f64>; 4],
    /// Per-window rotation: total turn and the largest single inter-epoch turn.
    pub turn_sum: [Option<f64>; 4],
    pub turn_max: [Option<f64>; 4],
    /// Within-epoch orientation spread.
    pub swing: Option<f64>,
    /// The interaction v2 hard-codes as `motion_quiescent(f) && ... .min(0.0)`. Supplied as a
    /// PRODUCT so a linear model can represent what v2 needs a branch for.
    pub still_x_cardiac: Option<f64>,
    /// Cardiac, carried through from the caller so this module stays motion-only in what it computes.
    pub hr_z: Option<f64>,
    pub hr_var_z: Option<f64>,
}

impl Features {
    /// Column names, index-for-index with [`Features::values`].
    pub const NAMES: [&'static str; 24] = [
        "motion_mean_30", "motion_mean_120", "motion_mean_300", "motion_mean_600",
        "motion_max_30", "motion_max_120", "motion_max_300", "motion_max_600",
        "motion_frac_30", "motion_frac_120", "motion_frac_300", "motion_frac_600",
        "turn_sum_30", "turn_sum_120", "turn_sum_300", "turn_sum_600",
        "turn_max_30", "turn_max_120", "turn_max_300", "turn_max_600",
        "swing", "still_x_cardiac", "hr_z", "hr_var_z",
    ];

    /// The vector, in [`Features::NAMES`] order. `None` becomes `f64::NAN` so a caller must decide
    /// what missing means rather than inheriting a silent zero.
    pub fn values(&self) -> [f64; 24] {
        let n = |o: Option<f64>| o.unwrap_or(f64::NAN);
        [
            n(self.motion_mean[0]), n(self.motion_mean[1]), n(self.motion_mean[2]), n(self.motion_mean[3]),
            n(self.motion_max[0]), n(self.motion_max[1]), n(self.motion_max[2]), n(self.motion_max[3]),
            n(self.motion_frac[0]), n(self.motion_frac[1]), n(self.motion_frac[2]), n(self.motion_frac[3]),
            n(self.turn_sum[0]), n(self.turn_sum[1]), n(self.turn_sum[2]), n(self.turn_sum[3]),
            n(self.turn_max[0]), n(self.turn_max[1]), n(self.turn_max[2]), n(self.turn_max[3]),
            n(self.swing), n(self.still_x_cardiac), n(self.hr_z), n(self.hr_var_z),
        ]
    }
}

/// Per-second gravity means over `[a, b)`, and the deltas between them.
///
/// `grav` must be sorted by `ts` - the same contract `posture_series` states. The range is found by
/// binary search rather than by filtering the whole slice: this is called once per epoch per window,
/// so a full rescan makes the cost quadratic in night length and multi-day bridged sessions much
/// worse than a linear extrapolation suggests.
fn deltas(grav: &[AccelSample], a: i64, b: i64) -> Vec<f64> {
    let lo = grav.partition_point(|g| g.ts < a);
    let hi = grav.partition_point(|g| g.ts < b);
    let mut per: std::collections::BTreeMap<i64, (f64, f64, f64, f64)> = Default::default();
    for g in grav[lo..hi].iter() {
        let e = per.entry(g.ts).or_insert((0.0, 0.0, 0.0, 0.0));
        e.0 += g.x;
        e.1 += g.y;
        e.2 += g.z;
        e.3 += 1.0;
    }
    let seq: Vec<(f64, f64, f64)> =
        per.values().map(|v| (v.0 / v.3, v.1 / v.3, v.2 / v.3)).collect();
    seq.windows(2)
        .map(|w| {
            ((w[0].0 - w[1].0).powi(2) + (w[0].1 - w[1].1).powi(2) + (w[0].2 - w[1].2).powi(2)).sqrt()
        })
        .collect()
}

/// NOT `crate::stats::median`. That one returns 0.0 on empty and interpolates the middle pair; this
/// returns NaN on empty deliberately, so an absent stream cannot masquerade as a real scale. Do not
/// "de-duplicate" these two without changing the callers.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Per-epoch features over `[start, end)`. `hr_z` and `hr_var_z` are the caller's per-night z-scores,
/// one per epoch; a short slice leaves the tail's cardiac columns missing rather than shifting them.
pub fn extract(
    grav: &[AccelSample],
    start: i64,
    end: i64,
    hr_z: &[Option<f64>],
    hr_var_z: &[Option<f64>],
) -> Vec<Features> {
    if end <= start {
        return Vec::new();
    }
    let n = ((end - start) / EPOCH_S) as usize;
    let post: Vec<Option<Posture>> = posture_series(grav, start, end, EPOCH_S);
    let turns = turn_series(&post);

    // The night's own scale, floored. p75 rather than the median because half a quiet night's
    // deltas are identically zero and a median of zero normalises nothing.
    let all = deltas(grav, start, end);
    let night_scale = {
        let mut v = all.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p75 = if v.is_empty() { 0.0 } else { v[(v.len() * 3 / 4).min(v.len() - 1)] };
        p75.max(STILL_SCALE_FLOOR_G)
    };

    (0..n)
        .map(|k| {
            let mid = start + k as i64 * EPOCH_S + EPOCH_S / 2;
            let mut f = Features {
                start: start + k as i64 * EPOCH_S,
                swing: post.get(k).and_then(|p| p.map(|p| p.swing)),
                hr_z: hr_z.get(k).copied().flatten(),
                hr_var_z: hr_var_z.get(k).copied().flatten(),
                ..Default::default()
            };
            for (w, half) in WINDOWS_S.iter().enumerate() {
                let (a, b) = (mid - half / 2, mid + half / 2);
                let d = deltas(grav, a, b);
                if !d.is_empty() {
                    f.motion_mean[w] = Some(d.iter().sum::<f64>() / d.len() as f64);
                    f.motion_max[w] = Some(d.iter().cloned().fold(f64::MIN, f64::max));
                    // Against the FLOORED scale, not the raw median. A quiet night's median delta
                    // is exactly 0.0, so the old `night_med > 0.0` guard left motion_frac missing on
                    // every epoch of precisely the nights this feature exists for.
                    let over = d.iter().filter(|x| **x > night_scale).count();
                    f.motion_frac[w] = Some(over as f64 / d.len() as f64);
                }
                // Rotation over the same window, in epochs rather than seconds.
                let lo = k.saturating_sub((half / 2 / EPOCH_S) as usize);
                let hi = (k + (half / 2 / EPOCH_S) as usize + 1).min(turns.len());
                let seg: Vec<f64> = turns[lo..hi.max(lo)].iter().flatten().copied().collect();
                if !seg.is_empty() {
                    f.turn_sum[w] = Some(seg.iter().sum());
                    f.turn_max[w] = Some(seg.iter().cloned().fold(f64::MIN, f64::max));
                }
            }
            // Stillness x cardiac, as a product rather than a branch. Stillness ramps linearly from
            // 1 at zero motion to 0 at `night_scale` (the floored p75), so the term is the cardiac
            // evidence that survives being still - what v2's clamp decides with an `if`.
            if let (Some(m), Some(h)) = (f.motion_mean[0], f.hr_z) {
                let still = (1.0 - m / night_scale).clamp(0.0, 1.0);
                f.still_x_cardiac = Some(still * h);
            }
            f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }

    /// A still night: every window measurable, motion at zero, nothing missing that is present.
    fn still(n: i64) -> Vec<AccelSample> {
        (0..n).map(|i| s(i, 0.0, 0.0, 1.0)).collect()
    }

    /// The failure this prevents: a fitted weight vector silently transposed against the wrong
    /// column, which no test of the model's accuracy would ever catch.
    ///
    /// The first version of this test asserted only `NAMES.len() == values().len()`, i.e. `24 == 24`.
    /// That would have passed with `swing` and `still_x_cardiac` swapped, with `turn_max[0]` emitted
    /// where `turn_max_600` is named, or with a field wholly omitted and a wrong one appended. It
    /// tested the one thing that could not go wrong. Every field now carries a UNIQUE marker value
    /// and each is asserted against its own name.
    #[test]
    fn every_value_lands_in_the_column_its_name_claims() {
        let mut f = Features::default();
        // Distinct per field AND per window, so any transposition changes a number.
        for w in 0..WINDOWS_S.len() {
            f.motion_mean[w] = Some(100.0 + w as f64);
            f.motion_max[w] = Some(200.0 + w as f64);
            f.motion_frac[w] = Some(300.0 + w as f64);
            f.turn_sum[w] = Some(400.0 + w as f64);
            f.turn_max[w] = Some(500.0 + w as f64);
        }
        f.swing = Some(600.0);
        f.still_x_cardiac = Some(700.0);
        f.hr_z = Some(800.0);
        f.hr_var_z = Some(900.0);

        let v = f.values();
        assert_eq!(Features::NAMES.len(), v.len());
        let expect: [(&str, f64); 24] = [
            ("motion_mean_30", 100.0), ("motion_mean_120", 101.0),
            ("motion_mean_300", 102.0), ("motion_mean_600", 103.0),
            ("motion_max_30", 200.0), ("motion_max_120", 201.0),
            ("motion_max_300", 202.0), ("motion_max_600", 203.0),
            ("motion_frac_30", 300.0), ("motion_frac_120", 301.0),
            ("motion_frac_300", 302.0), ("motion_frac_600", 303.0),
            ("turn_sum_30", 400.0), ("turn_sum_120", 401.0),
            ("turn_sum_300", 402.0), ("turn_sum_600", 403.0),
            ("turn_max_30", 500.0), ("turn_max_120", 501.0),
            ("turn_max_300", 502.0), ("turn_max_600", 503.0),
            ("swing", 600.0), ("still_x_cardiac", 700.0),
            ("hr_z", 800.0), ("hr_var_z", 900.0),
        ];
        for (i, (name, want)) in expect.iter().enumerate() {
            assert_eq!(Features::NAMES[i], *name, "column {i} is misnamed");
            assert_eq!(v[i], *want, "column {i} ({name}) carries the wrong field");
        }
    }

    /// The HIGH the review found: a genuinely quiet night has a median inter-second delta of exactly
    /// zero, and the old guard left `motion_frac` missing on every epoch of precisely the nights the
    /// feature exists for.
    #[test]
    fn motion_frac_is_measured_on_a_quiet_night_rather_than_missing() {
        let f = extract(&still(1200), 0, 1200, &[], &[]);
        for w in 0..WINDOWS_S.len() {
            assert_eq!(f[20].motion_frac[w], Some(0.0),
                "window {w}: a still night must report a measured zero fraction, not None");
        }
        assert!(!f[20].values()[8].is_nan(), "and it must not reach the vector as NaN");
    }

    #[test]
    fn a_still_night_reports_zero_motion_rather_than_missing_motion() {
        let g = still(1200);
        let f = extract(&g, 0, 1200, &[], &[]);
        assert_eq!(f.len(), 40);
        let mid = &f[20];
        for w in 0..WINDOWS_S.len() {
            assert_eq!(mid.motion_mean[w], Some(0.0), "window {w} must measure zero, not None");
        }
        assert!(mid.swing.is_some_and(|s| s < 1e-9));
    }

    #[test]
    fn an_absent_stream_reports_missing_rather_than_still() {
        let f = extract(&[], 0, 600, &[], &[]);
        assert_eq!(f.len(), 20);
        assert!(f[10].motion_mean.iter().all(|m| m.is_none()), "no gravity is not zero movement");
        assert!(f[10].values()[0].is_nan(), "and it must reach the vector as NaN, not 0.0");
    }

    /// The point of the multi-window family: a brief burst is loud in the 30 s window and diluted in
    /// the 10 min one, and the RATIO is what tells a twitch from a period of activity.
    #[test]
    fn a_brief_burst_shows_in_the_short_window_and_is_diluted_in_the_long_one() {
        let mut g = still(1200);
        // One second of real movement in the middle.
        for i in 600..602 {
            g[i as usize] = s(i, 0.5, 0.0, 0.87);
        }
        let f = extract(&g, 0, 1200, &[], &[]);
        let k = 20; // the epoch containing second 600
        let short = f[k].motion_mean[0].expect("30 s window");
        let long = f[k].motion_mean[3].expect("10 min window");
        assert!(short > long * 3.0,
            "a burst must be louder in the short window: {short} vs {long}");
        assert!(f[k].motion_max[0].unwrap() > 0.1, "and the peak must survive");
    }

    /// The interaction v2 needs a branch for. Same cardiac evidence, different motion, different
    /// value - which is exactly what a linear model cannot do without this column.
    #[test]
    fn stillness_times_cardiac_separates_two_epochs_a_linear_model_would_tie() {
        let mut g = still(1200);
        // Genuinely MOVING, not merely held somewhere new: a block at a new constant orientation has
        // zero internal delta, because only the transition into it moves. That is the whole reason
        // `swing` exists, and the first version of this test fell for it.
        for i in 600..630 {
            let a = if i % 2 == 0 { 0.5 } else { 0.0 };
            g[i as usize] = s(i, a, 0.0, (1.0f64 - a * a).sqrt());
        }
        let hr: Vec<Option<f64>> = (0..40).map(|_| Some(2.0)).collect();
        let f = extract(&g, 0, 1200, &hr, &[]);
        let moving = f[20].still_x_cardiac.expect("moving epoch");
        let quiet = f[5].still_x_cardiac.expect("still epoch");
        assert!(quiet > moving,
            "identical hr_z, but the still epoch must carry more surviving cardiac: {quiet} vs {moving}");
    }

    #[test]
    fn windows_are_centred_so_a_feature_neither_leads_nor_lags_its_label() {
        // A burst at the very start is seen by epoch 0's window and not by a later epoch's.
        let mut g = still(1200);
        for i in 0..2 {
            g[i as usize] = s(i, 0.5, 0.0, 0.87);
        }
        let f = extract(&g, 0, 1200, &[], &[]);
        assert!(f[0].motion_max[0].unwrap() > 0.1, "epoch 0 sees its own burst");
        assert!(f[30].motion_max[0].unwrap() < 1e-6, "an epoch 15 min later does not");
    }
}
