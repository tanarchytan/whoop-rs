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
/// index sits at or near the MAXIMUM, so one movement becomes the whole night's scale and every other
/// epoch reads as a measured zero. Fragmented captures go down to 3% coverage.
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
    /// Per-window motion energy: mean, max and the fraction of seconds above this night's median.
    /// Indexed by [`WINDOWS_S`].
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

/// Per-epoch features over `[start, end)`. `card` is the caller's per-night cardiac series, one entry
/// per epoch; a short slice leaves the tail's cardiac columns missing rather than shifting them.
///
/// `clock` is a fraction of THIS span, so what it means depends entirely on where the caller puts
/// the boundaries. A span that opens well before sleep and one that opens at sleep give the same
/// epoch different values, and a model fitted on one distribution extrapolates on the other.
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
    let all = deltas(grav, start, end);
    let night_scale = {
        let mut v = all.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p75 = if v.len() < MIN_SCALE_DELTAS {
            0.0
        } else {
            v[(v.len() * 3 / 4).min(v.len() - 1)]
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
                    // Against the FLOORED scale, not the raw median. A quiet night's median delta
                    // is exactly 0.0, so the old `night_med > 0.0` guard left motion_frac missing on
                    // every epoch of precisely the nights this feature exists for.
                    let over = d.iter().filter(|x| **x > night_scale).count();
                    f.motion_frac[w] = Some(over as f64 / d.len() as f64);
                }
                // Rotation over the same window, in epochs rather than seconds.
                let lo = k.saturating_sub((width / 2 / EPOCH_S) as usize);
                let hi = (k + (width / 2 / EPOCH_S) as usize + 1).min(turns.len());
                let seg: Vec<f64> = turns[lo..hi.max(lo)].iter().flatten().copied().collect();
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
    /// A width-only assertion passes with two fields swapped, so every field carries a UNIQUE
    /// marker and is asserted against its own name.
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

    /// A fragmented night leaves a handful of consecutive-second deltas, and p75 over a handful sits
    /// at the MAXIMUM - one movement becomes the scale and every other epoch reads as a MEASURED
    /// zero, which nothing downstream can tell from a still night.
    #[test]
    fn a_fragmented_night_does_not_let_one_burst_become_the_whole_scale() {
        // TWO widely separated 3-second bursts, so only FOUR consecutive-second deltas survive.
        // The count matters: p75's index is `(n*3/4).min(n-1)`, which lands on the MAXIMUM at n=4.
        // At n=6 the index is 4, never the maximum, and the guard is not reached at all.
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
        // A burst at the very start is seen by epoch 0's window and not by a later epoch's.
        let mut g = still(1200);
        for i in 0..2 {
            g[i as usize] = s(i, 0.5, 0.0, 0.87);
        }
        let f = extract(&g, 0, 1200, &[]);
        assert!(f[0].motion_max[0].unwrap() > 0.1, "epoch 0 sees its own burst");
        assert!(f[30].motion_max[0].unwrap() < 1e-6, "an epoch 15 min later does not");
    }
}
