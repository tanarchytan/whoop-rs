//! The tanv1 feature vector: one named, fitted-model-ready record per epoch.
//!
//! The same small set of physical quantities the shipped recipe reads, but each over four centred
//! timescales rather than one, plus the products it hard-codes as branches and the elapsed-night
//! clock it derives a temporal prior from.
//!
//! Everything here is descriptive: no thresholds, no scoring, no stage decision. [`Features::NAMES`]
//! is index-for-index with [`Features::values`] so a fitted weight vector can never silently
//! transpose against the wrong column - the failure mode that makes a fitted model unreviewable.

use super::input::AccelSample;
use super::posture::{posture_series, turn_series, Posture};

/// Epoch length. Everything windowed is centred on the epoch, never trailing, so a feature does not
/// lead or lag the label it is scored against.
pub const EPOCH_S: i64 = 30;

/// Absolute floor on the stillness scale, in g. A quiet night's median inter-second gravity delta is
/// exactly zero, so every ratio against it is 0 or infinite; gravity is on an absolute scale, so a
/// physical floor is defensible where a per-wearer one is not.
pub const STILL_SCALE_FLOOR_G: f64 = 0.01;

/// Fewest consecutive-second deltas before the night's p75 is trusted as a scale. Below this the p75
/// is not a stable order statistic of the night's movement, so the physical floor is used instead.
/// Fragmented captures go down to 3% coverage.
pub const MIN_SCALE_DELTAS: usize = 120;

/// Centred window WIDTHS in seconds - `(mid - w/2, mid + w/2)` spans exactly `w`, not `2w`. The short
/// end is the epoch itself, the long end is where the benchmark's actigraphy features stop.
pub const WINDOWS_S: [i64; 4] = [30, 120, 300, 600];

/// The cardiac quantities per epoch, computed by the caller so this module stays motion-only.
/// `hr_flat_pct` is the transform the shipped recipe's deep gate reads; without it no column here can
/// express deep, and a fitted model's deep numbers describe the missing feature rather than itself.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cardiac {
    pub hr_z: Option<f64>,
    pub hr_var_z: Option<f64>,
    /// Rank in 0..1 of this epoch's ~12-minute HR standard deviation among the night's own epochs.
    pub hr_flat_pct: Option<f64>,
    /// Per-night z-score of RSA respiration regularity. The ONLY R-R-fed quantity: `hr_var_z` is a
    /// per-second heart-rate spread and is present with or without beats, so it cannot stand in.
    pub resp_z: Option<f64>,
}

/// One epoch, fully described. `None` is "not measurable here", which a fitted model must be handed
/// as an explicit missing-indicator rather than as a zero - a zero is a claim of no movement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Features {
    /// Epoch start, unix seconds.
    pub start: i64,
    /// Per-window motion energy: mean and max of the consecutive-second gravity deltas, and the
    /// fraction of them above this night's floored p75 scale. Indexed by [`WINDOWS_S`].
    pub motion_mean: [Option<f64>; 4],
    pub motion_max: [Option<f64>; 4],
    pub motion_frac: [Option<f64>; 4],
    /// Per-window rotation: total turn and the largest single inter-epoch turn. `turn_sum[0]` is
    /// NOT emitted - the 30 s window spans exactly one inter-epoch turn, so its sum and its max are
    /// the same number, and a fitted model would spend four parameters on a duplicate column.
    pub turn_sum: [Option<f64>; 4],
    pub turn_max: [Option<f64>; 4],
    /// Within-epoch orientation spread.
    pub swing: Option<f64>,
    /// The clamp v2 hard-codes as a branch, supplied as PRODUCTS a linear model can represent. Both,
    /// because that clamp reads the HR-level and the HR-variability term; one alone is half of it.
    pub still_x_cardiac: Option<f64>,
    pub still_x_hrvar: Option<f64>,
    /// Cardiac, carried through from the caller so this module stays motion-only in what it computes.
    pub hr_z: Option<f64>,
    pub hr_var_z: Option<f64>,
    /// Within-night rank of the long-window HR standard deviation. The deep-separating quantity.
    pub hr_flat_pct: Option<f64>,
    /// RSA respiration regularity, z-scored within the night. v2 weights this into DEEP and out of
    /// REM, and it is the only place beats reach either recipe.
    pub resp_z: Option<f64>,
    /// Elapsed fraction of the span, 0..1. v2 builds a temporal prior off exactly this quantity and
    /// the same span, so a design matrix without it cannot see what the shipped recipe sees.
    /// Anchored to the SPAN, not to sleep onset - see the note on [`extract`].
    pub clock: Option<f64>,
    /// Fraction of the LONGEST window that lies inside the span, 0..1. The first and last ~10 epochs
    /// cannot have a full centred window, and without this column a fitted model reads that
    /// systematic edge bias as signal.
    pub win_cov_600: Option<f64>,
}

impl Features {
    /// Column names, index-for-index with [`Features::values`].
    pub const NAMES: [&'static str; Self::N] = [
        "motion_mean_30", "motion_mean_120", "motion_mean_300", "motion_mean_600",
        "motion_max_30", "motion_max_120", "motion_max_300", "motion_max_600",
        "motion_frac_30", "motion_frac_120", "motion_frac_300", "motion_frac_600",
        "turn_sum_120", "turn_sum_300", "turn_sum_600",
        "turn_max_30", "turn_max_120", "turn_max_300", "turn_max_600",
        "swing", "still_x_cardiac", "still_x_hrvar", "hr_z", "hr_var_z", "hr_flat_pct", "resp_z",
        "clock", "win_cov_600",
    ];

    /// Column count. One constant so a consumer sizes its design matrix from here rather than
    /// repeating the number and drifting when a column is added.
    pub const N: usize = 28;

    /// The vector, in [`Features::NAMES`] order. `None` becomes `f64::NAN` so a caller must decide
    /// what missing means rather than inheriting a silent zero.
    pub fn values(&self) -> [f64; Self::N] {
        let n = |o: Option<f64>| o.unwrap_or(f64::NAN);
        [
            n(self.motion_mean[0]), n(self.motion_mean[1]), n(self.motion_mean[2]), n(self.motion_mean[3]),
            n(self.motion_max[0]), n(self.motion_max[1]), n(self.motion_max[2]), n(self.motion_max[3]),
            n(self.motion_frac[0]), n(self.motion_frac[1]), n(self.motion_frac[2]), n(self.motion_frac[3]),
            n(self.turn_sum[1]), n(self.turn_sum[2]), n(self.turn_sum[3]),
            n(self.turn_max[0]), n(self.turn_max[1]), n(self.turn_max[2]), n(self.turn_max[3]),
            n(self.swing), n(self.still_x_cardiac), n(self.still_x_hrvar), n(self.hr_z),
            n(self.hr_var_z), n(self.hr_flat_pct), n(self.resp_z),
            n(self.clock), n(self.win_cov_600),
        ]
    }
}

/// Per-second gravity means over `[a, b)`, and the deltas between them. `grav` must be sorted by
/// `ts`. Found by binary search, not by filtering the slice: this runs once per epoch per window, so
/// a full rescan is quadratic in night length.
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
    let seq: Vec<(i64, f64, f64, f64)> =
        per.iter().map(|(t, v)| (*t, v.0 / v.3, v.1 / v.3, v.2 / v.3)).collect();
    // Only CONSECUTIVE seconds. Two map entries either side of a dropout are adjacent in the map but
    // minutes apart in time, and pairing them attributes a whole gap's movement to one second - a
    // large delta indistinguishable from real motion, flowing into motion_max and the night scale.
    seq.windows(2)
        .filter(|w| w[1].0 - w[0].0 == 1)
        .map(|w| {
            ((w[0].1 - w[1].1).powi(2) + (w[0].2 - w[1].2).powi(2) + (w[0].3 - w[1].3).powi(2)).sqrt()
        })
        .collect()
}

/// Per-epoch features over `[start, end)`. `card` is one entry per epoch; a short slice leaves the
/// tail's cardiac columns missing rather than shifting them. `clock` is a fraction of THIS span, so
/// a span opening before sleep and one opening at sleep give the same epoch different values.
pub fn extract(grav: &[AccelSample], start: i64, end: i64, card: &[Cardiac]) -> Vec<Features> {
    if end <= start {
        return Vec::new();
    }
    // Sorted DEFENSIVELY rather than trusting the contract: `deltas` binary-searches and
    // `posture_series` walks a forward cursor, so unsorted input returns a wrong range silently -
    // no panic, no NaN, just wrong numbers.
    let owned: Vec<AccelSample>;
    let grav = if grav.windows(2).all(|w| w[0].ts <= w[1].ts) {
        grav
    } else {
        owned = { let mut v = grav.to_vec(); v.sort_by_key(|g| g.ts); v };
        &owned
    };
    let n = ((end - start) / EPOCH_S) as usize;
    let post: Vec<Option<Posture>> = posture_series(grav, start, end, EPOCH_S);
    let turns = turn_series(&post);

    // The night's own scale, floored. p75 rather than the median because half a quiet night's
    // deltas are identically zero and a median of zero normalises nothing.
    let night_scale = {
        let mut v = deltas(grav, start, end);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p75 = if v.len() < MIN_SCALE_DELTAS {
            0.0
        } else {
            v[v.len() * 3 / 4]
        };
        p75.max(STILL_SCALE_FLOOR_G)
    };

    (0..n)
        .map(|k| {
            let mid = start + k as i64 * EPOCH_S + EPOCH_S / 2;
            let c = card.get(k).copied().unwrap_or_default();
            let elapsed = (mid - start) as f64;
            let mut f = Features {
                start: start + k as i64 * EPOCH_S,
                swing: post.get(k).and_then(|p| p.map(|p| p.swing)),
                hr_z: c.hr_z,
                hr_var_z: c.hr_var_z,
                hr_flat_pct: c.hr_flat_pct,
                resp_z: c.resp_z,
                clock: Some(elapsed / (end - start) as f64),
                ..Default::default()
            };
            for (w, width) in WINDOWS_S.iter().enumerate() {
                // Clamped to the span. A window running off the end does not wrap or read nothing,
                // it is short - and how short is recorded on the longest window below.
                let (a, b) = ((mid - width / 2).max(start), (mid + width / 2).min(end));
                if w == WINDOWS_S.len() - 1 {
                    f.win_cov_600 = Some((b - a) as f64 / *width as f64);
                }
                let d = deltas(grav, a, b);
                if !d.is_empty() {
                    f.motion_mean[w] = Some(d.iter().sum::<f64>() / d.len() as f64);
                    f.motion_max[w] = Some(d.iter().cloned().fold(f64::MIN, f64::max));
                    // Against the FLOORED scale, not the raw median: a quiet night's median delta is
                    // exactly 0.0, so a median guard leaves motion_frac missing on precisely the
                    // nights this feature exists for.
                    let over = d.iter().filter(|x| **x > night_scale).count();
                    f.motion_frac[w] = Some(over as f64 / d.len() as f64);
                }
                // Rotation over the same window. A turn sits at the boundary BEFORE its own epoch, so
                // the slice runs a half-epoch later than the index to stay centred on the midpoint:
                // exactly `width / EPOCH_S` boundaries, the same half-open span the deltas use.
                let m = (width / 2 / EPOCH_S) as usize;
                let lo = (k + usize::from(m > 0)).saturating_sub(m);
                let hi = (k + m + 1).min(turns.len());
                let seg: Vec<f64> = turns[lo..hi].iter().flatten().copied().collect();
                if !seg.is_empty() {
                    f.turn_sum[w] = Some(seg.iter().sum());
                    f.turn_max[w] = Some(seg.iter().cloned().fold(f64::MIN, f64::max));
                }
            }
            // Stillness ramps linearly from 1 at zero motion to 0 at `night_scale`, so each product
            // is the cardiac evidence that survives being still - what v2's clamp decides with an
            // `if`. Both terms get one, because that clamp reads both.
            if let Some(m) = f.motion_mean[0] {
                let still = (1.0 - m / night_scale).clamp(0.0, 1.0);
                f.still_x_cardiac = f.hr_z.map(|h| still * h);
                f.still_x_hrvar = f.hr_var_z.map(|h| still * h);
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
        f.still_x_hrvar = Some(750.0);
        f.hr_z = Some(800.0);
        f.hr_var_z = Some(900.0);
        f.hr_flat_pct = Some(950.0);
        f.resp_z = Some(955.0);
        f.clock = Some(960.0);
        f.win_cov_600 = Some(1000.0);

        let v = f.values();
        assert_eq!(Features::NAMES.len(), v.len());
        assert_eq!(Features::N, v.len(), "N must be the real width, or a consumer sizes wrong");
        let expect: [(&str, f64); Features::N] = [
            ("motion_mean_30", 100.0), ("motion_mean_120", 101.0),
            ("motion_mean_300", 102.0), ("motion_mean_600", 103.0),
            ("motion_max_30", 200.0), ("motion_max_120", 201.0),
            ("motion_max_300", 202.0), ("motion_max_600", 203.0),
            ("motion_frac_30", 300.0), ("motion_frac_120", 301.0),
            ("motion_frac_300", 302.0), ("motion_frac_600", 303.0),
            ("turn_sum_120", 401.0), ("turn_sum_300", 402.0), ("turn_sum_600", 403.0),
            ("turn_max_30", 500.0), ("turn_max_120", 501.0),
            ("turn_max_300", 502.0), ("turn_max_600", 503.0),
            ("swing", 600.0), ("still_x_cardiac", 700.0), ("still_x_hrvar", 750.0),
            ("hr_z", 800.0), ("hr_var_z", 900.0), ("hr_flat_pct", 950.0), ("resp_z", 955.0),
            ("clock", 960.0), ("win_cov_600", 1000.0),
        ];
        // The 30 s rotation sum is deliberately absent: it equals turn_max_30 on every epoch.
        assert!(!Features::NAMES.contains(&"turn_sum_30"), "a duplicate column must not be emitted");
        for (i, (name, want)) in expect.iter().enumerate() {
            assert_eq!(Features::NAMES[i], *name, "column {i} is misnamed");
            assert_eq!(v[i], *want, "column {i} ({name}) carries the wrong field");
        }
    }

    /// The names hard-code the widths, so a change to [`WINDOWS_S`] alone turns six columns into
    /// lies: the placement test compares NAMES against `values()` and never against the widths.
    #[test]
    fn every_windowed_name_states_the_width_it_is_actually_computed_over() {
        assert_eq!(WINDOWS_S, [30, 120, 300, 600], "the column names below spell out these widths");
        for (i, name) in Features::NAMES.iter().take(19).enumerate() {
            // Three motion blocks of four, then turn_sum without its 30 s column, then turn_max.
            let w = match i {
                0..=11 => i % 4,
                12..=14 => i - 11,
                _ => i - 15,
            };
            assert!(name.ends_with(&format!("_{}", WINDOWS_S[w])),
                "column {i} ({name}) names a width it is not computed over");
        }
        let cov = Features::NAMES.iter().position(|n| n.starts_with("win_cov")).expect("coverage column");
        assert_eq!(Features::NAMES[cov], format!("win_cov_{}", WINDOWS_S[3]),
            "coverage names the longest window");

        // The name is true by construction; the REACH is not. One turn at the boundary into epoch 10
        // must be seen by exactly `width / EPOCH_S` epochs, or the rotation family lags its label.
        let mut g = still(1200);
        for i in 300..1200 {
            g[i as usize] = s(i, 1.0, 0.0, 0.0);
        }
        let f = extract(&g, 0, 1200, &[]);
        for (w, width) in WINDOWS_S.iter().enumerate() {
            let reached = f.iter().filter(|x| x.turn_max[w].is_some_and(|t| t > 1e-9)).count() as i64;
            assert_eq!(reached * EPOCH_S, *width,
                "turn_max_{width} reaches {reached} epochs, which is {} s", reached * EPOCH_S);
        }
    }

    /// A genuinely quiet night has a median inter-second delta of exactly zero, so a guard on that
    /// median leaves `motion_frac` missing on every epoch of precisely the nights it exists for.
    #[test]
    fn motion_frac_is_measured_on_a_quiet_night_rather_than_missing() {
        let f = extract(&still(1200), 0, 1200, &[]);
        for w in 0..WINDOWS_S.len() {
            assert_eq!(f[20].motion_frac[w], Some(0.0),
                "window {w}: a still night must report a measured zero fraction, not None");
        }
        assert!(!f[20].values()[8].is_nan(), "and it must not reach the vector as NaN");
    }

    #[test]
    fn a_still_night_reports_zero_motion_rather_than_missing_motion() {
        let g = still(1200);
        let f = extract(&g, 0, 1200, &[]);
        assert_eq!(f.len(), 40);
        let mid = &f[20];
        for w in 0..WINDOWS_S.len() {
            assert_eq!(mid.motion_mean[w], Some(0.0), "window {w} must measure zero, not None");
        }
        assert!(mid.swing.is_some_and(|s| s < 1e-9));
    }

    #[test]
    fn an_absent_stream_reports_missing_rather_than_still() {
        let f = extract(&[], 0, 600, &[]);
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
        let f = extract(&g, 0, 1200, &[]);
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
        // Genuinely MOVING, not merely held somewhere new: a block at a new constant orientation
        // has zero internal delta, because only the transition into it moves.
        for i in 600..630 {
            let a = if i % 2 == 0 { 0.5 } else { 0.0 };
            g[i as usize] = s(i, a, 0.0, (1.0f64 - a * a).sqrt());
        }
        let card: Vec<Cardiac> = (0..40)
            .map(|_| Cardiac { hr_z: Some(2.0), hr_var_z: Some(3.0), ..Default::default() })
            .collect();
        let f = extract(&g, 0, 1200, &card);
        let moving = f[20].still_x_cardiac.expect("moving epoch");
        let quiet = f[5].still_x_cardiac.expect("still epoch");
        assert!(quiet > moving,
            "identical hr_z, but the still epoch must carry more surviving cardiac: {quiet} vs {moving}");
        // BOTH cardiac terms get a product, because v2's clamp reads both.
        let moving_v = f[20].still_x_hrvar.expect("moving epoch");
        let quiet_v = f[5].still_x_hrvar.expect("still epoch");
        assert!(quiet_v > moving_v,
            "the hr_var half must behave the same way: {quiet_v} vs {moving_v}");
    }

    /// v2 builds a temporal prior from the elapsed clock, so a design matrix without one cannot see
    /// what the shipped recipe sees. It must span 0..1 across the span and rise monotonically.
    #[test]
    fn the_clock_spans_the_span_and_rises_monotonically() {
        let span: i64 = 4 * 5400;
        let f = extract(&still(span), 0, span, &[]);
        let first = f.first().expect("epochs").clock.expect("clock");
        let last = f.last().expect("epochs").clock.expect("clock");
        assert!(first < 0.01 && last > 0.99, "the clock must span the night: {first} to {last}");
        assert!(f.windows(2).all(|w| w[0].clock < w[1].clock), "and it must be monotonic");

        // Anchored to the SPAN, so a span that starts an hour early shifts every value.
        let shifted = extract(&still(span + 3600), -3600, span, &[]);
        let same_epoch = shifted[(3600 / EPOCH_S) as usize].clock.expect("clock");
        assert!(same_epoch > first + 0.1,
            "the same wall-clock epoch reads later in a span that starts earlier: {same_epoch}");
    }

    /// The deep-separating quantity is CARRIED, not invented here: a caller that supplies it must see
    /// it in the named column, and one that does not must see missing rather than a silent zero.
    #[test]
    fn the_flatness_rank_reaches_its_own_column_and_is_missing_when_unsupplied() {
        let g = still(600);
        let card: Vec<Cardiac> =
            (0..20).map(|k| Cardiac { hr_flat_pct: Some(k as f64 / 20.0), ..Default::default() }).collect();
        let f = extract(&g, 0, 600, &card);
        let col = Features::NAMES.iter().position(|n| *n == "hr_flat_pct").expect("named column");
        assert_eq!(f[7].values()[col], 7.0 / 20.0, "the rank must land in its own column");

        let bare = extract(&g, 0, 600, &[]);
        assert!(bare[7].values()[col].is_nan(), "unsupplied must be NaN, never a rank of zero");
    }

    /// The only beat-fed column, and no other column can stand in for it: if `extract` stops carrying
    /// it, every vector loses its whole R-R channel and nothing else moves.
    #[test]
    fn respiration_regularity_reaches_its_own_column_and_is_missing_when_unsupplied() {
        let g = still(600);
        let card: Vec<Cardiac> =
            (0..20).map(|k| Cardiac { resp_z: Some(k as f64 / 20.0), ..Default::default() }).collect();
        let f = extract(&g, 0, 600, &card);
        let col = Features::NAMES.iter().position(|n| *n == "resp_z").expect("named column");
        assert_eq!(f[7].values()[col], 7.0 / 20.0, "the R-R channel must land in its own column");

        let bare = extract(&g, 0, 600, &[]);
        assert!(bare[7].values()[col].is_nan(), "unsupplied must be NaN, never a regularity of zero");
    }

    /// Two entries either side of a dropout are adjacent in the map but minutes apart in time, and
    /// pairing them attributes a whole gap's movement to one second.
    #[test]
    fn a_dropout_does_not_manufacture_one_enormous_delta() {
        // Still at one orientation, a 95 s hole, then still at a completely different one.
        let mut g: Vec<AccelSample> = (0..5).map(|i| s(i, 0.0, 0.0, 1.0)).collect();
        g.extend((100..105).map(|i| s(i, 1.0, 0.0, 0.0)));
        let f = extract(&g, 0, 300, &[]);
        let peak = f.iter().filter_map(|x| x.motion_max[3]).fold(0.0f64, f64::max);
        assert!(peak < 1e-6,
            "the gap must not be read as movement: two still stretches, peak delta {peak}");
    }

    /// A real stream carries several samples a second, and `deltas` collapses each second to its MEAN
    /// before differencing. Summing instead rescales every motion column by the sample rate.
    #[test]
    fn a_multi_rate_stream_measures_the_same_motion_as_a_one_hz_one() {
        // Every literal is a multiple of 1/8, so each second's four samples average to the 1 Hz value
        // exactly and the two arms produce bit-identical deltas.
        let amp = |i: i64| if i % 2 == 1 { 0.5 } else { 0.0 };
        let one_hz: Vec<AccelSample> = (0..600).map(|i| s(i, amp(i), 0.0, 1.0)).collect();
        let multi: Vec<AccelSample> = (0..600)
            .flat_map(|i| [-0.375, -0.125, 0.125, 0.375].map(|d| s(i, amp(i) + d, 0.0, 1.0)))
            .collect();
        let a = extract(&one_hz, 0, 600, &[]);
        let b = extract(&multi, 0, 600, &[]);
        assert_eq!(a.len(), b.len());
        assert!(a[10].motion_mean[0].is_some_and(|m| m > 0.1), "the night must actually move");
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.motion_mean, y.motion_mean, "epoch {}: the rate must not scale the mean", x.start);
            assert_eq!(x.motion_max, y.motion_max, "epoch {}: nor the peak", x.start);
            assert_eq!(x.motion_frac, y.motion_frac, "epoch {}: nor the fraction over scale", x.start);
        }
    }

    /// A fragmented night leaves a handful of consecutive-second deltas, and p75 over a handful sits
    /// at the MAXIMUM - one movement becomes the scale and every other epoch reads as a MEASURED
    /// zero, which nothing downstream can tell from a still night.
    #[test]
    fn a_fragmented_night_does_not_let_one_burst_become_the_whole_scale() {
        // TWO widely separated 3-second bursts, so only FOUR consecutive-second deltas survive.
        // The count matters: at n=4 the p75 index lands on the MAXIMUM, and only being far under
        // MIN_SCALE_DELTAS keeps that burst out of the scale.
        let mut g: Vec<AccelSample> = Vec::new();
        for t in [0i64, 1200] {
            for i in 0..3 {
                let moved = t == 1200 && i == 2;
                g.push(if moved { s(t + i, 1.0, 0.0, 0.0) } else { s(t + i, 0.0, 0.0, 1.0) });
            }
        }
        let f = extract(&g, 0, 1800, &[]);
        // Under MIN_SCALE_DELTAS the floor is used, so the burst cannot inflate the scale and the
        // epoch that really moved must still stand out.
        let moved_epoch = f.iter().find(|x| x.start == 1200).expect("epoch at 1200");
        assert!(moved_epoch.motion_frac[0].is_some_and(|v| v > 0.0),
            "the epoch that genuinely moved must report a non-zero fraction, got {:?}",
            moved_epoch.motion_frac[0]);
    }

    /// The binary search in `deltas` needs sorted input. Unsorted input must give the SAME answer,
    /// not a silently wrong one.
    #[test]
    fn unsorted_input_gives_the_same_answer_rather_than_a_silently_wrong_one() {
        let mut g = still(600);
        for i in 300..330 {
            let a = if i % 2 == 0 { 0.5 } else { 0.0 };
            g[i as usize] = s(i, a, 0.0, (1.0f64 - a * a).sqrt());
        }
        let sorted = extract(&g, 0, 600, &[]);
        let mut shuffled = g.clone();
        shuffled.reverse();
        let out = extract(&shuffled, 0, 600, &[]);
        assert_eq!(sorted.len(), out.len());
        for (a, b) in sorted.iter().zip(&out) {
            assert_eq!(a.motion_max, b.motion_max, "epoch {} differs on unsorted input", a.start);
            assert_eq!(a.motion_mean, b.motion_mean);
        }
    }

    /// An edge epoch cannot have a full 10-minute centred window, and without this column a fitted
    /// model reads that systematic truncation as signal.
    #[test]
    fn edge_epochs_declare_their_truncated_window() {
        let f = extract(&still(2400), 0, 2400, &[]);
        assert!(f[0].win_cov_600.unwrap() < 0.6, "epoch 0 has at most half a centred 10-min window");
        assert!((f[40].win_cov_600.unwrap() - 1.0).abs() < 1e-9, "the middle has a full one");
        assert!(f.last().unwrap().win_cov_600.unwrap() < 0.6, "and so does the last epoch");
    }

    #[test]
    fn windows_are_centred_so_a_feature_neither_leads_nor_lags_its_label() {
        // Epoch 20's 30 s window is [600, 630) centred and would be [585, 615) trailing, so a burst
        // at second 620 is reached only by the centred one - and a trailing epoch 21 would lag onto it.
        let mut g = still(1200);
        for i in 620..622 {
            g[i as usize] = s(i, 0.5, 0.0, 0.87);
        }
        let f = extract(&g, 0, 1200, &[]);
        assert!(f[20].motion_max[0].unwrap() > 0.1, "epoch 20's own window covers second 620");
        assert!(f[19].motion_max[0].unwrap() < 1e-6, "the epoch before must not lead onto it");
        assert!(f[21].motion_max[0].unwrap() < 1e-6, "and the epoch after must not lag onto it");
    }

    /// Rotation is seven columns and the only thing that sees a roll-over. Each window spans a fixed
    /// number of epochs, and the 30 s one spans exactly one turn - which is why its sum is not emitted.
    #[test]
    fn a_roll_over_reaches_the_rotation_columns_over_the_window_each_one_names() {
        // Flat, one epoch on the side, flat again: two 90-degree turns, at epochs 10 and 11.
        let mut g = still(1200);
        for i in 300..330 {
            g[i as usize] = s(i, 1.0, 0.0, 0.0);
        }
        let f = extract(&g, 0, 1200, &[]);
        assert!((f[10].turn_max[0].expect("the rotating epoch") - 90.0).abs() < 1e-6);
        assert_eq!(f[5].turn_max[0], Some(0.0), "a still epoch turns by zero, not by nothing");
        assert_eq!(f[0].turn_max[0], None, "nothing precedes the first epoch");
        // 120 s spans 4 boundaries, so it carries both turns; 600 s spans 20, nine back and ten on.
        assert!((f[10].turn_sum[1].expect("120 s window") - 180.0).abs() < 1e-6);
        assert_eq!(f[25].turn_sum[3], Some(0.0), "epoch 25 is 14 epochs out, past the 20-boundary window");
        for (k, x) in f.iter().enumerate() {
            assert_eq!(x.turn_sum[0], x.turn_max[0],
                "epoch {k}: the 30 s window spans one turn, so its sum IS its max");
        }
    }

    /// `swing` is per-epoch, and a whole-column index error is invisible on a uniform night because
    /// every epoch's value is the same number.
    #[test]
    fn swing_reports_the_epoch_it_belongs_to_rather_than_the_night() {
        let mut g = still(1200);
        for i in 300..330 {
            let a = (i - 300) as f64 * std::f64::consts::TAU / 30.0;
            g[i as usize] = s(i, a.cos(), a.sin(), 0.0);
        }
        let f = extract(&g, 0, 1200, &[]);
        assert!(f[10].swing.expect("the sweeping epoch") > 0.9, "a full sweep cancels");
        assert!(f[9].swing.expect("before") < 1e-9, "its neighbours held one orientation");
        assert!(f[11].swing.expect("after") < 1e-9);
        assert!(f[0].swing.expect("epoch 0") < 1e-9, "and no epoch may report epoch 0's spread");
    }

    /// p75, not the median: they land on different scales whenever the near-zero half of a night runs
    /// past the midpoint, and the scale sets `motion_frac` on all four windows.
    #[test]
    fn the_night_scale_is_the_p75_delta_and_not_the_median() {
        // Four blocks of alternating x, so each block's inter-second delta is its own amplitude.
        // Sorted over the night, the median lands on 0.05 g and the p75 on 0.5 g.
        let amp = |i: i64| match i {
            0..=599 => 0.05,
            600..=1019 => 0.5,
            1020..=1109 => 0.1,
            _ => 0.9,
        };
        let g: Vec<AccelSample> =
            (0..1200).map(|i| s(i, if i % 2 == 1 { amp(i) } else { 0.0 }, 0.0, 1.0)).collect();
        let f = extract(&g, 0, 1200, &[]);
        assert_eq!(f[38].motion_frac[0], Some(1.0), "0.9 g deltas clear either scale");
        assert_eq!(f[35].motion_frac[0], Some(0.0),
            "0.1 g deltas sit under a p75 of 0.5 g - against the median's 0.05 g they would read 1.0");
    }

    /// The floor is the entire scale on a quiet or fragmented night, so its VALUE decides what counts
    /// as movement there. One delta either side of it pins the number, not just its sign.
    #[test]
    fn the_stillness_floor_is_the_scale_a_quiet_night_is_measured_against() {
        assert_eq!(STILL_SCALE_FLOOR_G, 0.01, "the literal steps below spell out this floor");
        // Far under MIN_SCALE_DELTAS, so the p75 is not trusted. The two steps straddle the floor by
        // 1% of it: 0.0101 g over, 0.0099 g under.
        let step = |i: i64| if i < 5 { 0.0 } else if i < 10 { 0.0101 } else { 0.02 };
        let g: Vec<AccelSample> = (0..15).map(|i| s(i, step(i), 0.0, 1.0)).collect();
        let f = extract(&g, 0, 30, &[]);
        let frac = f[0].motion_frac[0].expect("a measured fraction");
        assert!((frac * 14.0 - 1.0).abs() < 1e-9,
            "exactly one of the 14 deltas clears a 0.01 g floor, got a fraction of {frac}");
    }

    /// Either side of the threshold the night is measured against a different scale, so only the
    /// constant's own value picks the side. The counts are LITERAL: feeding the constant back in as
    /// its own input would pass for any value the span can hold.
    #[test]
    fn the_scale_switches_at_min_scale_deltas_and_not_at_some_other_count() {
        assert_eq!(MIN_SCALE_DELTAS, 120, "the literal counts below spell out this threshold");
        // Every delta is 0.2 g: over the 0.01 g floor, and not over a measured p75 of 0.2 g.
        let night = |secs: i64| -> Vec<AccelSample> {
            (0..secs).map(|i| s(i, if i % 2 == 1 { 0.2 } else { 0.0 }, 0.0, 1.0)).collect()
        };
        // 120 samples is 119 consecutive-second deltas and 121 is 120; the span holds up to 149.
        let below = extract(&night(120), 0, 150, &[]);
        let at = extract(&night(121), 0, 150, &[]);
        assert_eq!(below[0].motion_frac[0], Some(1.0), "119 deltas, one short: the floor is the scale");
        assert_eq!(at[0].motion_frac[0], Some(0.0), "at 120 the night's own p75 is");
    }

    /// The products are a linear ramp, not merely a decreasing one: an ordering assertion holds for
    /// any slope, and the slope is what decides how much cardiac evidence survives being still.
    #[test]
    fn stillness_ramps_to_a_known_value_rather_than_merely_downwards() {
        // Deltas of 0.5 g all night, so the p75 scale is 0.5; epoch 5 alone moves at half of it.
        let mut g: Vec<AccelSample> =
            (0..1200).map(|i| s(i, if i % 2 == 1 { 0.5 } else { 0.0 }, 0.0, 1.0)).collect();
        for i in 150..180 {
            g[i as usize] = s(i, if i % 2 == 1 { 0.25 } else { 0.0 }, 0.0, 1.0);
        }
        let card: Vec<Cardiac> = (0..40)
            .map(|_| Cardiac { hr_z: Some(2.0), hr_var_z: Some(3.0), ..Default::default() })
            .collect();
        let f = extract(&g, 0, 1200, &card);
        assert_eq!(f[5].still_x_cardiac, Some(1.0), "half the night's scale leaves half of hr_z");
        assert_eq!(f[5].still_x_hrvar, Some(1.5), "and half of hr_var_z");
        assert_eq!(f[10].still_x_cardiac, Some(0.0), "at the night's scale none of it survives");
    }

    /// The two ends the ramp never reaches: motion PAST the scale, where only the clamp keeps the
    /// cardiac evidence from changing sign, and an epoch with no gravity at all, where a product
    /// would be a claim of no movement.
    #[test]
    fn stillness_clamps_past_the_scale_and_stays_missing_without_gravity() {
        let card: Vec<Cardiac> = (0..40)
            .map(|_| Cardiac { hr_z: Some(2.0), hr_var_z: Some(3.0), ..Default::default() })
            .collect();

        // Deltas of 0.5 g all night, so the p75 scale is 0.5; epoch 5 alone moves at TWICE it.
        let mut g: Vec<AccelSample> =
            (0..1200).map(|i| s(i, if i % 2 == 1 { 0.5 } else { 0.0 }, 0.0, 1.0)).collect();
        for i in 150..180 {
            g[i as usize] = s(i, if i % 2 == 1 { 1.0 } else { 0.0 }, 0.0, 1.0);
        }
        let over = extract(&g, 0, 1200, &card);
        assert_eq!(over[5].motion_mean[0], Some(1.0), "epoch 5 must move at twice the 0.5 g scale");
        assert_eq!(over[5].still_x_cardiac, Some(0.0),
            "past the scale none of hr_z survives - it must not turn negative and grow");
        assert_eq!(over[5].still_x_hrvar, Some(0.0), "and the hr_var half must not either");

        // Gravity stops halfway through the span, so epoch 15's own 30 s window holds none of it.
        let bare = extract(&still(300), 0, 600, &card);
        assert_eq!(bare[15].hr_z, Some(2.0), "the cardiac evidence is present");
        assert_eq!(bare[15].motion_mean[0], None, "but this epoch's window has no gravity");
        assert_eq!(bare[15].still_x_cardiac, None, "so the product is missing, not full-strength");
        assert_eq!(bare[15].still_x_hrvar, None);
    }
}
