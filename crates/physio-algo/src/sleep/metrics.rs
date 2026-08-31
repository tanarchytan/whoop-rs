//! Hypnogram scoring: one confusion matrix, the numbers read off it, and the paired bar that says
//! whether two runs differ at all.
//!
//! Kappa is a whole-night agreement figure dominated by the stages that hold the most epochs, so it
//! barely moves when a twenty-minute wake bout is missed. Per-class recall and [`bout_score`] are what
//! a user complaint is about. Every function here takes `[truth][pred]`, the order the harnesses fill.

/// `[truth][pred]`, the order every harness in this crate fills.
pub type Confusion4 = [[i64; 4]; 4];

/// Stage index in a confusion matrix, matching `SleepStage as usize`.
pub const WAKE: usize = 0;

/// Tally predictions against truth. Both slices are stage indices on the same epoch grid; the shorter
/// length wins, so a caller cannot silently score epochs it does not have.
pub fn confusion4(pred: &[usize], truth: &[usize]) -> Confusion4 {
    let mut cm = [[0i64; 4]; 4];
    for (p, t) in pred.iter().zip(truth) {
        if *p < 4 && *t < 4 {
            cm[*t][*p] += 1;
        }
    }
    cm
}

/// `[truth][pred]` over Wake / NREM / REM, the split most wearable papers report.
pub type Confusion3 = [[i64; 3]; 3];

/// Cohen's kappa over any square confusion matrix. Zero for an empty matrix and for any matrix where
/// chance already explains everything, so a scorer that emits one stage all night cannot reach a target
/// by doing nothing.
fn kappa_of<const N: usize>(cm: &[[i64; N]; N]) -> f64 {
    let tot: i64 = cm.iter().flatten().sum();
    if tot == 0 {
        return 0.0;
    }
    let tot = tot as f64;
    let po = (0..N).map(|i| cm[i][i]).sum::<i64>() as f64 / tot;
    let mut pe = 0.0;
    for (j, row_j) in cm.iter().enumerate() {
        let col: i64 = cm.iter().map(|r| r[j]).sum();
        let row: i64 = row_j.iter().sum();
        pe += col as f64 * row as f64;
    }
    pe /= tot * tot;
    if pe >= 1.0 { 0.0 } else { (po - pe) / (1.0 - pe) }
}

pub fn kappa4(cm: &Confusion4) -> f64 {
    kappa_of(cm)
}

pub fn kappa3(cm: &Confusion3) -> f64 {
    kappa_of(cm)
}

/// Fold Light and Deep into one NREM class. Merging an already-scored four-class result is what the
/// field does rather than training a dedicated three-class model, so the two numbers describe the
/// SAME staging and can be reported side by side.
pub fn merge3(cm: &Confusion4) -> Confusion3 {
    const TO3: [usize; 4] = [0, 1, 1, 2];
    let mut out = [[0i64; 3]; 3];
    for (t, row) in cm.iter().enumerate() {
        for (p, n) in row.iter().enumerate() {
            out[TO3[t]][TO3[p]] += n;
        }
    }
    out
}

/// Truth marginals of a confusion matrix: the share of epochs each class truly holds.
pub fn truth_marginals(cm: &Confusion4) -> [f64; 4] {
    let tot: i64 = cm.iter().flatten().sum();
    let mut out = [0.0; 4];
    if tot == 0 {
        return out;
    }
    for (i, row) in cm.iter().enumerate() {
        out[i] = row.iter().sum::<i64>() as f64 / tot as f64;
    }
    out
}

/// The bonus kappa's OWN optimal rule adds to each class's posterior.
///
/// `1 - kappa` is a ratio of two linear forms in the confusion matrix, so the rule that maximises it
/// is not `argmax` of the posterior: it maximises `eta_j + (1 - kappa)(1 - t_j)`. The second term is
/// this. It grows as a class gets RARER, so kappa pays for calling rare classes whether or not the
/// evidence improved — which is why an engine must not be selected on kappa alone.
pub fn kappa_class_bonus(cm: &Confusion4) -> [f64; 4] {
    let k = kappa_of(cm);
    let t = truth_marginals(cm);
    std::array::from_fn(|j| (1.0 - k) * (1.0 - t[j]))
}

/// Kappa after relabelling `mass` (a fraction of all epochs) from predicted class `from` to predicted
/// class `to`, leaving the number CORRECT untouched.
///
/// Same accuracy, different kappa. What it costs to move is the exchange rate between the metric and
/// nothing at all; a gain smaller than this is the metric moving, not the engine.
pub fn kappa_after_reassignment(cm: &Confusion4, from: usize, to: usize, mass: f64) -> Option<f64> {
    let tot: i64 = cm.iter().flatten().sum();
    if tot == 0 || from > 3 || to > 3 || !(0.0..=1.0).contains(&mass) {
        return None;
    }
    let tot = tot as f64;
    let po = (0..4).map(|i| cm[i][i]).sum::<i64>() as f64 / tot;
    let t = truth_marginals(cm);
    let mut q: [f64; 4] =
        std::array::from_fn(|j| cm.iter().map(|r| r[j]).sum::<i64>() as f64 / tot);
    // A class cannot give away more prediction mass than it holds.
    if mass > q[from] {
        return None;
    }
    q[from] -= mass;
    q[to] += mass;
    let pe: f64 = (0..4).map(|j| t[j] * q[j]).sum();
    Some(if pe >= 1.0 { 0.0 } else { (po - pe) / (1.0 - pe) })
}

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Percentile bootstrap interval for the POOLED kappa, resampling RECORDINGS with replacement.
/// Resampling epochs instead would treat one long night as many independent observations; the night
/// is the unit of measurement. `seed` fixes the resample so a published interval is reproducible.
pub fn bootstrap_kappa_ci(
    nights: &[Confusion4],
    iters: usize,
    alpha: f64,
    seed: u64,
) -> Option<(f64, f64)> {
    let n = nights.len();
    if n < 2 || iters == 0 || !(0.0..0.5).contains(&alpha) {
        return None;
    }
    let mut s = seed;
    let mut draws = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut pooled = [[0i64; 4]; 4];
        for _ in 0..n {
            s = splitmix(s);
            let pick = &nights[(s >> 11) as usize % n];
            for (out_row, add_row) in pooled.iter_mut().zip(pick) {
                for (o, a) in out_row.iter_mut().zip(add_row) {
                    *o += *a;
                }
            }
        }
        draws.push(kappa_of(&pooled));
    }
    draws.sort_by(f64::total_cmp);
    let lo = ((alpha / 2.0) * iters as f64).floor() as usize;
    let hi = (((1.0 - alpha / 2.0) * iters as f64).ceil() as usize).saturating_sub(1);
    Some((draws[lo.min(iters - 1)], draws[hi.min(iters - 1)]))
}

/// Share of this class's true epochs given that class. `None` when the class never occurs, which is a
/// different fact from scoring zero on it.
pub fn recall(cm: &Confusion4, class: usize) -> Option<f64> {
    let actual: i64 = cm[class].iter().sum();
    (actual > 0).then(|| cm[class][class] as f64 / actual as f64)
}

/// Share of the epochs given this class that truly are it. `None` when the class is never predicted.
pub fn precision(cm: &Confusion4, class: usize) -> Option<f64> {
    let called: i64 = cm.iter().map(|r| r[class]).sum();
    (called > 0).then(|| cm[class][class] as f64 / called as f64)
}

/// Share of everything that is NOT this class correctly not called it. Recall's counterweight: a scorer
/// that calls every epoch wake has wake recall 1.0 and wake specificity 0.0.
pub fn specificity(cm: &Confusion4, class: usize) -> Option<f64> {
    let mut neg = 0i64;
    let mut fp = 0i64;
    for (t, row) in cm.iter().enumerate() {
        if t == class {
            continue;
        }
        neg += row.iter().sum::<i64>();
        fp += row[class];
    }
    (neg > 0).then(|| (neg - fp) as f64 / neg as f64)
}

/// Maximal runs of one class, as `(start, len)` over epoch indices, keeping only runs of `min_len`.
pub fn bouts(seq: &[usize], class: usize, min_len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < seq.len() {
        if seq[i] != class {
            i += 1;
            continue;
        }
        let start = i;
        while i < seq.len() && seq[i] == class {
            i += 1;
        }
        if i - start >= min_len {
            out.push((start, i - start));
        }
    }
    out
}

/// Bout-level agreement for one class. Predicting the class everywhere detects every bout, so
/// `spurious` and [`BoutScore::precision`] are reported beside it. [`BoutScore::coverage`] is the
/// continuous read; `detected` is BINARY at `min_overlap`, so a long bout is all-or-nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoutScore {
    pub truth_bouts: usize,
    /// Truth bouts covered by at least `min_overlap` of their epochs. Binary - see the type note.
    pub detected: usize,
    pub pred_bouts: usize,
    /// Predicted bouts overlapping no truth bout of the class at all. A weak precision arm on
    /// purpose: one overlapping epoch clears it, so it catches only wholly invented bouts.
    pub spurious: usize,
    /// Epochs of truth bouts we called the class, and the total in them. The continuous read.
    pub covered_epochs: usize,
    pub truth_bout_epochs: usize,
}

impl BoutScore {
    /// Detected over truth bouts. `None` when the class never occurs in truth. BINARY at the overlap
    /// floor, so prefer [`BoutScore::coverage`] when comparing two candidates.
    pub fn recall(&self) -> Option<f64> {
        (self.truth_bouts > 0).then(|| self.detected as f64 / self.truth_bouts as f64)
    }

    /// Share of all truth-bout epochs we called the class. Continuous, so partial progress on a long
    /// bout is visible where [`BoutScore::recall`] rounds it to zero.
    pub fn coverage(&self) -> Option<f64> {
        (self.truth_bout_epochs > 0)
            .then(|| self.covered_epochs as f64 / self.truth_bout_epochs as f64)
    }

    /// Non-spurious over predicted bouts. `None` when the class is never predicted as a bout.
    pub fn precision(&self) -> Option<f64> {
        (self.pred_bouts > 0).then(|| (self.pred_bouts - self.spurious) as f64 / self.pred_bouts as f64)
    }
}

/// Score one class bout-wise. A truth bout counts as detected when `min_overlap` of its epochs carry the
/// class; a predicted bout is spurious when not one of its epochs is truly the class.
pub fn bout_score(
    pred: &[usize], truth: &[usize], class: usize, min_len: usize, min_overlap: f64,
) -> BoutScore {
    let n = pred.len().min(truth.len());
    let (pred, truth) = (&pred[..n], &truth[..n]);
    let (tb, pb) = (bouts(truth, class, min_len), bouts(pred, class, min_len));
    let hits: Vec<usize> =
        tb.iter().map(|(s, l)| pred[*s..*s + *l].iter().filter(|p| **p == class).count()).collect();
    let detected = tb
        .iter()
        .zip(&hits)
        .filter(|((_, l), hit)| **hit as f64 >= min_overlap * *l as f64)
        .count();
    let covered_epochs: usize = hits.iter().sum();
    let truth_bout_epochs: usize = tb.iter().map(|(_, l)| *l).sum();
    let spurious = pb
        .iter()
        .filter(|(s, l)| !truth[*s..*s + *l].contains(&class))
        .count();
    BoutScore {
        truth_bouts: tb.len(),
        detected,
        pred_bouts: pb.len(),
        spurious,
        covered_epochs,
        truth_bout_epochs,
    }
}

/// Textbook two-sided 95% critical values, keyed by degrees of freedom.
const T95: [(usize, f64); 12] = [
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (9, 2.262), (12, 2.179),
    (19, 2.093), (30, 2.042), (39, 2.023), (59, 2.001), (119, 1.980),
];

/// Two-sided 95% critical value at `n-1` degrees of freedom, floored at df 1 (the widest, most
/// conservative row); 1.96 is ~11% too narrow at n=13. Rounds df DOWN to the previous row - the
/// value falls as df rises, so rounding up would return a bar narrower than the truth.
fn t95(n: usize) -> f64 {
    let df = n.saturating_sub(1).max(1);
    T95.iter().rev().find(|(k, _)| *k <= df).expect("the table starts at df 1").1
}

/// Mean paired difference and the delta this sample size can resolve, `t * sd / sqrt(n)`. Two arms'
/// MEDIANS are separate order statistics whose difference moves when one subject changes rank; a
/// mean inside the bar is noise whatever those medians say.
pub fn paired_bar(deltas: &[f64]) -> Option<(f64, f64)> {
    let n = deltas.len();
    if n < 2 {
        return None;
    }
    let m = deltas.iter().sum::<f64>() / n as f64;
    let sd = (deltas.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    Some((m, t95(n) * sd / (n as f64).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bonus is what makes kappa unsafe to select on: it pays MORE for a rarer class, so a rule
    /// that maximises kappa is not the rule that maximises accuracy.
    #[test]
    fn the_kappa_bonus_grows_as_a_class_gets_rarer() {
        // Truth marginals 0.5 / 0.3 / 0.15 / 0.05, imperfectly staged so kappa is under 1.
        let cm: Confusion4 = [[40, 5, 3, 2], [6, 20, 3, 1], [3, 3, 8, 1], [1, 1, 1, 2]];
        let t = truth_marginals(&cm);
        assert!((t[0] - 0.50).abs() < 1e-12 && (t[3] - 0.05).abs() < 1e-12, "{t:?}");
        let b = kappa_class_bonus(&cm);
        assert!(b[3] > b[2] && b[2] > b[1] && b[1] > b[0], "rarer must pay more: {b:?}");

        // At kappa 1 there is nothing left to gain, so every bonus vanishes - the two claims need
        // different matrices, and asserting both on one is how this test first contradicted itself.
        let perfect: Confusion4 = [[50, 0, 0, 0], [0, 30, 0, 0], [0, 0, 15, 0], [0, 0, 0, 5]];
        let pb = kappa_class_bonus(&perfect);
        assert!(pb.iter().all(|x| x.abs() < 1e-12), "a perfect matrix has no bonus to give: {pb:?}");
    }

    /// Hand-computed. Kappa 0.5 and a class holding 20% of the truth gives (1-0.5)(1-0.2) = 0.4.
    #[test]
    fn the_bonus_is_one_minus_kappa_times_one_minus_the_marginal() {
        let cm: Confusion4 = [[40, 10, 5, 5], [10, 15, 3, 2], [5, 3, 8, 4], [5, 2, 4, 9]];
        let (k, t) = (kappa4(&cm), truth_marginals(&cm));
        let b = kappa_class_bonus(&cm);
        for j in 0..4 {
            assert!((b[j] - (1.0 - k) * (1.0 - t[j])).abs() < 1e-12, "class {j}");
        }
    }

    /// Accuracy is untouched and kappa still moves. That gap is the metric, not the engine, and it is
    /// the whole reason D11 says kappa is a report rather than an objective.
    #[test]
    fn relabelling_toward_a_rarer_class_moves_kappa_at_constant_accuracy() {
        // Light (class 1) is common in truth, deep (class 2) is rare.
        let cm: Confusion4 = [[30, 8, 2, 4], [7, 60, 6, 5], [3, 9, 10, 2], [5, 7, 3, 25]];
        let base = kappa4(&cm);
        let moved = kappa_after_reassignment(&cm, 1, 2, 0.016).expect("mass is available");
        assert!(moved > base, "light -> deep must raise kappa: {base:.6} -> {moved:.6}");

        assert_eq!(
            Some(base),
            kappa_after_reassignment(&cm, 1, 2, 0.0),
            "moving nothing must change nothing"
        );
        assert_eq!(None, kappa_after_reassignment(&cm, 1, 2, 0.99), "cannot give away mass it lacks");
        assert_eq!(None, kappa_after_reassignment(&cm, 4, 0, 0.01), "class 4 does not exist");
    }


    /// Hand-computed: totals 30, 22 on the diagonal, every row and column 10, so pe is exactly 1/3
    /// and kappa is exactly 0.6. A formula that drifts cannot land on a round number by accident.
    #[test]
    fn three_class_kappa_matches_a_hand_computed_matrix() {
        let cm: Confusion3 = [[8, 1, 1], [1, 7, 2], [1, 2, 7]];
        assert!((kappa3(&cm) - 0.6).abs() < 1e-12, "got {}", kappa3(&cm));
    }

    /// Merging must move epochs between cells, never create or destroy them, and it must fold LIGHT
    /// and DEEP together rather than any other pair.
    #[test]
    fn merge3_folds_light_and_deep_and_conserves_every_epoch() {
        let mut cm: Confusion4 = [[0; 4]; 4];
        let mut k = 1;
        for row in cm.iter_mut() {
            for cell in row.iter_mut() {
                *cell = k;
                k += 1;
            }
        }
        let m = merge3(&cm);
        assert_eq!(
            cm.iter().flatten().sum::<i64>(),
            m.iter().flatten().sum::<i64>(),
            "merging must conserve the epoch count"
        );
        // Light/Deep truth x Light/Deep pred is the 2x2 block cm[1..3][1..3].
        assert_eq!(cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2], m[1][1], "NREM is Light+Deep");
        assert_eq!(cm[0][0], m[0][0], "wake must not merge with anything");
        assert_eq!(cm[3][3], m[2][2], "rem must not merge with anything");
    }

    /// The whole point of the three-class report: light-vs-deep confusion stops being an error.
    #[test]
    fn merging_forgives_light_deep_confusion_and_nothing_else() {
        // Perfect except that every deep epoch is called light.
        let cm: Confusion4 = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 10, 0, 0], [0, 0, 0, 10]];
        assert!(kappa3(&merge3(&cm)) > kappa4(&cm), "merging must forgive it");
        assert!((kappa3(&merge3(&cm)) - 1.0).abs() < 1e-12, "and forgive it completely");

        // Same shape of error, but REM called wake: merging cannot help.
        let across: Confusion4 = [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [10, 0, 0, 0]];
        assert!(kappa3(&merge3(&across)) < 1.0, "a wake/REM error must survive merging");
    }

    #[test]
    fn the_bootstrap_brackets_the_point_estimate_and_repeats_for_a_seed() {
        let nights: Vec<Confusion4> = (0..20)
            .map(|i| {
                let d = i as i64 % 5;
                [[20 + d, 3, 0, 1], [4, 30, 5, 3], [0, 6, 15 - d, 1], [2, 4, 1, 12]]
            })
            .collect();
        let mut pooled = [[0i64; 4]; 4];
        for n in &nights {
            for (o, a) in pooled.iter_mut().zip(n) {
                for (x, y) in o.iter_mut().zip(a) {
                    *x += *y;
                }
            }
        }
        let point = kappa4(&pooled);
        let (lo, hi) = bootstrap_kappa_ci(&nights, 400, 0.05, 7).expect("20 nights is enough");
        assert!(lo < point && point < hi, "{lo} .. {hi} must bracket {point}");
        assert_eq!(
            Some((lo, hi)),
            bootstrap_kappa_ci(&nights, 400, 0.05, 7),
            "the same seed must give the same interval"
        );
        assert_ne!(
            Some((lo, hi)),
            bootstrap_kappa_ci(&nights, 400, 0.05, 8),
            "a different seed must resample differently"
        );
        assert_eq!(None, bootstrap_kappa_ci(&nights[..1], 400, 0.05, 7), "one night is not a corpus");
    }

    /// Fewer recordings must widen the interval. An interval that ignores n would report the same
    /// precision for 4 nights as for 20, which is the trap a per-epoch bootstrap falls into.
    #[test]
    fn the_interval_widens_when_there_are_fewer_nights() {
        let nights: Vec<Confusion4> = (0..24)
            .map(|i| {
                let d = i as i64 % 7;
                [[18 + d, 4, 1, 2], [5, 28 - d, 6, 4], [1, 7, 14, 2], [3, 5, 2, 11 + d]]
            })
            .collect();
        let wide = bootstrap_kappa_ci(&nights[..4], 600, 0.05, 3).unwrap();
        let tight = bootstrap_kappa_ci(&nights, 600, 0.05, 3).unwrap();
        assert!(
            wide.1 - wide.0 > tight.1 - tight.0,
            "4 nights {:.4} must be wider than 24 nights {:.4}",
            wide.1 - wide.0,
            tight.1 - tight.0
        );
    }

    /// The overlap floor is inclusive, and it is the boundary `recall` turns on. Nothing sat exactly
    /// on it before, so the comparison could be tightened without a failure.
    #[test]
    fn a_truth_bout_at_exactly_the_overlap_floor_counts_as_detected() {
        let truth = vec![1usize; 10];
        let at_floor: Vec<usize> = (0..10).map(|i| usize::from(i < 5)).collect();
        let below: Vec<usize> = (0..10).map(|i| usize::from(i < 4)).collect();

        assert_eq!(bout_score(&at_floor, &truth, 1, 2, 0.5).detected, 1, "5 of 10 is the floor");
        assert_eq!(bout_score(&below, &truth, 1, 2, 0.5).detected, 0, "4 of 10 is below it");
    }

    /// Ragged input is expected: a hypnogram and its reference can differ by an epoch. Both series are
    /// truncated to the shorter, and taking the longer would index past the end of one of them.
    #[test]
    fn bout_score_truncates_to_the_shorter_series() {
        let long = vec![1usize; 10];
        let short = vec![1usize; 4];

        let pred_short = bout_score(&short, &long, 1, 2, 0.5);
        assert_eq!((pred_short.truth_bouts, pred_short.detected), (1, 1));
        assert_eq!(pred_short.truth_bout_epochs, 4, "truth is cut to the prediction's length");

        let truth_short = bout_score(&long, &short, 1, 2, 0.5);
        assert_eq!((truth_short.truth_bouts, truth_short.detected), (1, 1));
        assert_eq!(truth_short.truth_bout_epochs, 4);
    }

    #[test]
    fn kappa_reproduces_a_hand_computed_matrix() {
        let cm = [[8, 2, 0, 0], [2, 8, 0, 0], [0, 0, 8, 2], [0, 0, 2, 8]];
        assert!((kappa4(&cm) - 0.55 / 0.75).abs() < 1e-12, "got {}", kappa4(&cm));
        assert_eq!(kappa4(&[[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]), 1.0);
    }

    #[test]
    fn kappa_reports_zero_where_agreement_is_meaningless() {
        assert_eq!(kappa4(&[[0; 4]; 4]), 0.0, "an empty matrix is not agreement");
        assert_eq!(kappa4(&[[40, 0, 0, 0], [0; 4], [0; 4], [0; 4]]), 0.0, "one class both sides");
    }

    #[test]
    fn confusion_indexes_truth_first_and_ignores_the_tail_it_cannot_pair() {
        let cm = confusion4(&[1, 1, 1], &[0, 0]);
        assert_eq!(cm[0][1], 2, "two truth-wake epochs called light");
        assert_eq!(cm.iter().flatten().sum::<i64>(), 2, "the unpaired third epoch is not scored");
    }

    /// The counterweight has to hold or recall alone would reward calling everything wake.
    #[test]
    fn calling_every_epoch_wake_has_full_recall_and_zero_specificity() {
        let cm = confusion4(&[WAKE; 6], &[WAKE, WAKE, 1, 1, 2, 3]);
        assert_eq!(recall(&cm, WAKE), Some(1.0));
        assert_eq!(specificity(&cm, WAKE), Some(0.0));
        assert_eq!(precision(&cm, WAKE), Some(2.0 / 6.0));
    }

    /// The `< 4` guard is load-bearing and was unguarded: `sleep_eval` uses index 4 as its UNLABELLED
    /// sentinel and feeds it straight in, so dropping the check indexes a [[i64;4];4] out of bounds and
    /// panics on real input.
    #[test]
    fn an_out_of_range_index_is_ignored_rather_than_indexed() {
        const UNLABELLED: usize = 4;
        let cm = confusion4(&[UNLABELLED, 1, UNLABELLED], &[0, 1, UNLABELLED]);
        assert_eq!(cm.iter().flatten().sum::<i64>(), 1, "only the one in-range pair is scored");
        assert_eq!(cm[1][1], 1);
    }

    /// Specificity must exclude the class's OWN row: everything in it is a positive, so counting it as
    /// a negative inflates the score. The class row here is MIXED, the only shape where including it
    /// changes the answer.
    #[test]
    fn specificity_excludes_the_class_row_even_when_that_row_is_mixed() {
        // Three truth-wake epochs, one called wake and two called light; plus one true light.
        let cm = confusion4(&[WAKE, 1, 1, 1], &[WAKE, WAKE, WAKE, 1]);
        assert_eq!((cm[0][0], cm[0][1], cm[1][1]), (1, 2, 1), "the class row must be MIXED here");
        // Only the single true-light epoch is a negative, and it was not called wake.
        assert_eq!(specificity(&cm, WAKE), Some(1.0));
    }

    #[test]
    fn a_class_that_never_occurs_reports_none_rather_than_zero() {
        let cm = confusion4(&[1, 1], &[1, 1]);
        assert_eq!(recall(&cm, WAKE), None, "absent in truth");
        assert_eq!(precision(&cm, WAKE), None, "never predicted");
        assert_eq!(specificity(&cm, WAKE), Some(1.0), "nothing was wrongly called wake");
    }

    #[test]
    fn bouts_are_maximal_runs_at_or_over_the_minimum() {
        let seq = [0, 0, 0, 1, 0, 1, 1, 0, 0];
        assert_eq!(bouts(&seq, WAKE, 2), vec![(0, 3), (7, 2)], "the lone epoch at 4 is under the floor");
        assert_eq!(bouts(&seq, WAKE, 1), vec![(0, 3), (4, 1), (7, 2)]);
        assert_eq!(bouts(&[], WAKE, 1), vec![]);
    }

    /// The defect this whole metric exists for: a long true wake bout scored almost entirely as sleep.
    #[test]
    fn a_wake_bout_covered_below_the_overlap_floor_counts_as_missed() {
        let truth: Vec<usize> = vec![WAKE; 10];
        let mut pred = vec![1usize; 10];
        pred[0] = WAKE;
        pred[1] = WAKE;
        let s = bout_score(&pred, &truth, WAKE, 4, 0.5);
        assert_eq!(s.truth_bouts, 1);
        assert_eq!(s.detected, 0, "2 of 10 epochs is under a half-overlap floor");
        assert_eq!(s.recall(), Some(0.0));
        assert_eq!(s.pred_bouts, 0, "the 2-epoch call is under the 4-epoch bout floor");
    }

    #[test]
    fn an_invented_bout_is_spurious_and_costs_precision() {
        let truth = [1usize, 1, 1, 1, 1, 1, 1, 1];
        let pred = [WAKE, WAKE, WAKE, WAKE, 1, 1, 1, 1];
        let s = bout_score(&pred, &truth, WAKE, 4, 0.5);
        assert_eq!(s.truth_bouts, 0);
        assert_eq!(s.recall(), None, "no true wake bout to find");
        assert_eq!((s.pred_bouts, s.spurious), (1, 1));
        assert_eq!(s.precision(), Some(0.0));
    }

    /// The flaw `coverage` exists for, as a test. Two predictions of a long bout, 20% and 45% found:
    /// `recall` calls both a total miss and cannot tell them apart, while coverage sees the gap.
    #[test]
    fn coverage_separates_partial_finds_that_binary_recall_rounds_to_zero() {
        let truth = vec![WAKE; 100];
        let part = |found: usize| {
            let mut p = vec![1usize; 100];
            for (j, slot) in p.iter_mut().enumerate() {
                if j % 2 == 0 && j / 2 < found {
                    *slot = WAKE;
                }
            }
            bout_score(&p, &truth, WAKE, 10, 0.5)
        };
        let (weak, better) = (part(20), part(45));
        assert_eq!(weak.recall(), Some(0.0), "20% of a long bout is a miss to binary recall");
        assert_eq!(better.recall(), Some(0.0), "and so is 45% - recall cannot separate them");
        assert!((weak.coverage().unwrap() - 0.20).abs() < 1e-9, "{:?}", weak.coverage());
        assert!((better.coverage().unwrap() - 0.45).abs() < 1e-9, "{:?}", better.coverage());
        assert!(better.coverage() > weak.coverage(), "coverage MUST see what recall cannot");
    }

    #[test]
    fn coverage_is_none_where_no_bout_exists_rather_than_zero() {
        let s = bout_score(&[WAKE; 8], &[1usize; 8], WAKE, 10, 0.5);
        assert_eq!(s.truth_bouts, 0);
        assert_eq!(s.coverage(), None, "no truth bout is not zero coverage");
    }

    #[test]
    fn a_fully_covered_bout_is_detected_and_not_spurious() {
        let truth = [1usize, WAKE, WAKE, WAKE, WAKE, 1];
        let pred = [1usize, WAKE, WAKE, WAKE, WAKE, 1];
        let s = bout_score(&pred, &truth, WAKE, 4, 0.5);
        assert_eq!((s.truth_bouts, s.detected, s.pred_bouts, s.spurious), (1, 1, 1, 0));
        assert_eq!((s.recall(), s.precision()), (Some(1.0), Some(1.0)));
    }

    /// The bar must BRACKET the true critical value: narrower turns noise into a finding, wider
    /// reports a real difference as noise. Every value here is the textbook two-sided 95% point at
    /// that df; each row is pinned by an `n` landing ON it and an `n` one df short of it.
    #[test]
    fn the_bar_brackets_the_true_critical_value() {
        // Slack is earned only by a df that falls BETWEEN rows; its worst case is df=8 reading the
        // df=5 row, 2.571 against 2.306. A df landing ON a row must return that row unchanged.
        const ROUNDING_SLACK: f64 = 1.12;
        for (n, truth) in [(2usize, 12.706), (3, 4.303), (4, 3.182), (5, 2.776), (6, 2.571),
                           (7, 2.447), (9, 2.306), (10, 2.262), (12, 2.201), (13, 2.179),
                           (14, 2.160), (19, 2.101), (20, 2.093), (22, 2.080), (30, 2.045),
                           (31, 2.042), (36, 2.030), (39, 2.024), (40, 2.023), (59, 2.002),
                           (60, 2.001), (119, 1.980), (120, 1.980)] {
            let on_a_row = T95.iter().any(|(k, _)| *k == n - 1);
            let ceiling = if on_a_row { truth } else { truth * ROUNDING_SLACK };
            assert!(t95(n) >= truth - 1e-9,
                    "n={n} (df={}) needs at least {truth}, got {}", n - 1, t95(n));
            assert!(t95(n) <= ceiling + 1e-9,
                    "n={n} (df={}) may not exceed {ceiling}, got {}", n - 1, t95(n));
        }
    }

    /// Against a hand-computed value, on a series with real spread. The degenerate cases below
    /// cannot see the formula: a constant series has bar 0 for ANY scale factor, and a mean-zero
    /// one satisfies `|m| < bar` for any positive bar.
    #[test]
    fn the_bar_matches_a_hand_computed_value() {
        // n=5, mean 0.03. Deviations -0.02,-0.01,0,0.01,0.02 -> sample sd = sqrt(0.001/4) = 0.015811.
        // df=4, t=2.776, so bar = 2.776 * 0.015811 / sqrt(5) = 0.019631.
        let (m, bar) = paired_bar(&[0.01, 0.02, 0.03, 0.04, 0.05]).expect("n=5");
        assert!((m - 0.03).abs() < 1e-12, "mean {m}");
        assert!((bar - 0.019631).abs() < 1e-5, "bar {bar}, want 0.019631");
    }

    #[test]
    fn a_constant_difference_resolves_and_a_symmetric_one_does_not() {
        let (m, bar) = paired_bar(&[0.05; 20]).expect("n=20");
        assert!(m > bar, "a constant offset has zero spread and must resolve");
        let alt: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 0.05 } else { -0.05 }).collect();
        let (m, bar) = paired_bar(&alt).expect("n=20");
        assert!(m.abs() < bar, "a mean-zero difference must not resolve: {m} vs {bar}");
    }

    #[test]
    fn fewer_than_two_pairs_cannot_answer() {
        assert!(paired_bar(&[]).is_none());
        assert!(paired_bar(&[0.1]).is_none());
    }
}
