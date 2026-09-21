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

/// Harmonic mean of this class's recall and precision, and 0.0 when it occurs or is predicted but
/// nothing about it is right. `None` ONLY when the class is absent from truth AND never predicted,
/// which is the one case it is not a class of this cohort's problem. Feeds [`macro_f1`].
pub fn f1(cm: &Confusion4, class: usize) -> Option<f64> {
    let (r, p) = (recall(cm, class), precision(cm, class));
    if r.is_none() && p.is_none() {
        return None;
    }
    let (r, p) = (r.unwrap_or(0.0), p.unwrap_or(0.0));
    Some(if r + p > 0.0 { 2.0 * r * p / (r + p) } else { 0.0 })
}

/// Mean of the per-class recalls over the classes that occur. Each term conditions on TRUTH, so a
/// change in class balance leaves every one of them alone - which is what makes this, and not a mean
/// of F1, safe to select an engine on.
pub fn balanced_accuracy(cm: &Confusion4) -> Option<f64> {
    let r: Vec<f64> = (0..4).filter_map(|c| recall(cm, c)).collect();
    (!r.is_empty()).then(|| r.iter().sum::<f64>() / r.len() as f64)
}

/// Mean of the per-class F1s over the classes that occur OR are predicted. Calling a class MORE
/// cannot buy it the way it buys [`balanced_accuracy`], and never calling it AT ALL scores it zero
/// rather than dropping its term, so abolishing a class lowers this. Compare arms WITHIN one cohort.
pub fn macro_f1(cm: &Confusion4) -> Option<f64> {
    let f: Vec<f64> = (0..4).filter_map(|c| f1(cm, c)).collect();
    (!f.is_empty()).then(|| f.iter().sum::<f64>() / f.len() as f64)
}

/// The worst per-class recall. The mean can hide a class scoring zero; this cannot, and it is the
/// scalarisation to prefer when one rare class is the thing being fixed.
pub fn min_recall(cm: &Confusion4) -> Option<f64> {
    (0..4).filter_map(|c| recall(cm, c)).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.min(v))))
}

/// Mean and spread of a per-confusion statistic across RECORDINGS, over those that can carry it.
/// A pooled matrix weights a long night more than a short one and reports no spread at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub mean: f64,
    /// Population standard deviation across recordings; 0.0 when only one contributes.
    pub sd: f64,
    /// Recordings on which the statistic was defined. Fewer than the cohort means a class is absent
    /// from some nights, and the mean is over a different set than the cohort.
    pub n: usize,
}

pub fn per_recording(cms: &[Confusion4], f: impl Fn(&Confusion4) -> Option<f64>) -> Option<Spread> {
    let v: Vec<f64> = cms.iter().filter_map(&f).collect();
    if v.is_empty() {
        return None;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    Some(Spread { mean, sd, n: v.len() })
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

/// Two-sided 95% critical value at `n-1` degrees of freedom. One table, in `crate::stats`, so the
/// paired bar here and the slope test in `agreement` cannot drift apart.
fn t95(n: usize) -> f64 {
    crate::stats::t95_df(n.saturating_sub(1))
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

/// Two per-recording series paired BY RECORDING, over the recordings both carry, in id order. What
/// [`paired_bar`] needs as input: position is not identity, so an arm that cannot score a night
/// leaves every later night charged against a different one.
pub fn pair_by_id(
    base: &std::collections::BTreeMap<usize, f64>,
    arm: &std::collections::BTreeMap<usize, f64>,
) -> (Vec<f64>, Vec<f64>) {
    base.iter().filter_map(|(id, b)| arm.get(id).map(|a| (*b, *a))).unzip()
}

/// What a paired headline says once the arm's worst class is looked at too. The mean of per-night
/// differences cannot see that an arm bought its gap by abolishing a class; [`paired_verdict`]
/// refuses to call that AHEAD, and this is what it returns instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// The mean sits inside the bar this many paired nights can resolve.
    Matches,
    /// Ahead by this many bars, with no class left worse off than the baseline's worst.
    Ahead(f64),
    /// Behind by this many bars.
    Behind(f64),
    /// Ahead by `bars` while the worst per-class recall FELL by `min_recall_delta`. Not a win.
    Degenerate { bars: f64, min_recall_delta: f64 },
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Matches => write!(f, "matches"),
            Verdict::Ahead(x) => write!(f, "AHEAD ({x:.2}x)"),
            Verdict::Behind(x) => write!(f, "behind ({x:.2}x)"),
            Verdict::Degenerate { bars, min_recall_delta } => {
                write!(f, "DEGENERATE ({bars:.2}x, min recall {min_recall_delta:+.4})")
            }
        }
    }
}

/// The verdict on a paired mean and its bar, refusing AHEAD when the arm's worst per-class recall
/// fell against the baseline's. Any drop is enough: no threshold separates a gap bought by giving a
/// class up from one earned. `min_recall_delta` is None when there is no baseline to difference.
pub fn paired_verdict(mean: f64, bar: f64, min_recall_delta: Option<f64>) -> Verdict {
    if mean.abs() <= bar {
        return Verdict::Matches;
    }
    let bars = mean.abs() / bar;
    if mean < 0.0 {
        return Verdict::Behind(bars);
    }
    match min_recall_delta {
        Some(d) if d < 0.0 => Verdict::Degenerate { bars, min_recall_delta: d },
        _ => Verdict::Ahead(bars),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bonus is what makes kappa unsafe to select on: it pays MORE for a rarer class, so a rule
    /// that maximises kappa is not the rule that maximises accuracy.
    /// THE reason `macro_f1` exists beside `balanced_accuracy`: calling a rare class more lifts
    /// balanced accuracy and cannot lift macro F1, because the second charges the precision the
    /// first ignores. Both are computed on the SAME pair of matrices.
    #[test]
    fn calling_a_rare_class_more_buys_balanced_accuracy_and_not_macro_f1() {
        // Truth: 80 of the common class, 20 of the rare one. Rows are truth, columns prediction.
        let tight: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 76, 4], [0, 0, 10, 10]];
        // The rare class called more than twice as often: its recall rises 0.50 -> 0.75 and the
        // common class gives up 0.95 -> 0.80, so the unweighted mean of recalls RISES.
        let loose: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 64, 16], [0, 0, 5, 15]];

        let (ba0, ba1) = (balanced_accuracy(&tight).unwrap(), balanced_accuracy(&loose).unwrap());
        let (f0, f1v) = (macro_f1(&tight).unwrap(), macro_f1(&loose).unwrap());
        assert!(ba1 > ba0, "calling deep more must lift balanced accuracy: {ba0} -> {ba1}");
        assert!(f1v < f0, "and it must NOT lift macro F1: {f0} -> {f1v}");

        // Macro F1 is the unweighted mean of the per-class F1s that exist, nothing else.
        let want: f64 = (0..4).filter_map(|c| f1(&loose, c)).sum::<f64>()
            / (0..4).filter_map(|c| f1(&loose, c)).count() as f64;
        assert!((f1v - want).abs() < 1e-12);
    }

    /// The other direction of the same asymmetry, which used to read as a WIN: a class never
    /// predicted has no precision, and dropping its term from the mean RAISED macro F1 while recall
    /// collapsed. It now scores zero and keeps its term, so abolishing a class costs what it should.
    #[test]
    fn abolishing_a_class_lowers_macro_f1_because_its_term_scores_zero() {
        // Same truth as above: 80 of the common class, 20 of the rare one. Rows truth, columns called.
        let tight: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 76, 4], [0, 0, 10, 10]];
        // The rare class never called once. Its recall is 0.0 and its precision does not exist.
        let silent: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 80, 0], [0, 0, 20, 0]];

        assert_eq!(2, (0..4).filter_map(|c| f1(&tight, c)).count(), "both classes must carry an F1");
        assert_eq!(Some(0.0), f1(&silent, 3), "the silent class scores zero rather than dropping out");
        assert_eq!(2, (0..4).filter_map(|c| f1(&silent, c)).count(), "so the mean is over the same two");
        assert_eq!(Some(0.0), recall(&silent, 3), "it is scored zero on recall, not absent");

        let (f0, f1v) = (macro_f1(&tight).unwrap(), macro_f1(&silent).unwrap());
        let (ba0, ba1) = (balanced_accuracy(&tight).unwrap(), balanced_accuracy(&silent).unwrap());
        assert!((f0 - 0.752).abs() < 5e-4, "hand-computed: {f0}");
        assert!((f1v - 0.444).abs() < 5e-4, "and 0.889 under the old drop-out rule: {f1v}");
        assert!(f1v < f0, "abolishing the rare class must LOWER macro F1: {f0} -> {f1v}");
        assert!(ba1 < ba0, "and balanced accuracy falls with it: {ba0} -> {ba1}");
        assert_eq!(Some(0.0), min_recall(&silent), "min recall is what does not hide it");
    }

    /// The two remaining cases, each pinned because each is a different fact. A class this cohort
    /// does not have is not part of its problem and must leave the mean; a class it does not have
    /// but the arm calls anyway is every call wrong, and inventing one may not be free.
    #[test]
    fn a_class_absent_from_truth_drops_out_only_while_nothing_predicts_it() {
        // Class 3 in neither truth nor prediction: class 2 alone carries the mean.
        let neither: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 80, 0], [0, 0, 0, 0]];
        assert_eq!(None, recall(&neither, 3), "absent from truth");
        assert_eq!(None, precision(&neither, 3), "and never called");
        assert_eq!(None, f1(&neither, 3), "so it is not a class of this cohort's problem");
        assert_eq!(Some(1.0), macro_f1(&neither), "only class 2 is in the mean");

        // Same truth, and the arm calls class 3 on four epochs that are truly class 2. Recall is
        // undefined, precision is 0.0, and the term is 0.0 rather than absent.
        let invented: Confusion4 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 76, 4], [0, 0, 0, 0]];
        assert_eq!(Some(0.0), precision(&invented, 3), "nothing it called class 3 truly is");
        assert_eq!(None, recall(&invented, 3), "and there is no class 3 to recall");
        assert_eq!(Some(0.0), f1(&invented, 3), "an invented class costs a zero, not nothing");
        // The drop-out answer would have been class 2's F1 alone. Keeping the zero halves it, which
        // is the isolation: class 2 is identical in both readings of the SAME matrix.
        let kept = macro_f1(&invented).unwrap();
        assert!((kept - f1(&invented, 2).unwrap() / 2.0).abs() < 1e-12, "{kept}");
        assert!(kept < f1(&invented, 2).unwrap(), "inventing an absent class may not be free");
    }

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
        use crate::stats::t95_df;
        for (n, truth) in [(2usize, 12.706), (3, 4.303), (4, 3.182), (5, 2.776), (6, 2.571),
                           (7, 2.447), (9, 2.306), (10, 2.262), (12, 2.201), (13, 2.179),
                           (14, 2.160), (19, 2.101), (20, 2.093), (22, 2.080), (30, 2.045),
                           (31, 2.042), (36, 2.030), (39, 2.024), (40, 2.023), (59, 2.002),
                           (60, 2.001), (119, 1.980), (120, 1.980)] {
            // Rounding down changes the value only AT a row, so df sits on one exactly when it
            // differs from df-1. df 1 is the first row and has nothing below it.
            let on_a_row = n == 2 || t95_df(n - 1) != t95_df(n - 2);
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

    /// A confusion matrix for a FIXED classifier `m[truth][pred]` under the given class prevalences,
    /// scaled to whole counts. Changing only `pri` changes the cohort, never the engine.
    fn under(pri: [f64; 4], m: [[f64; 4]; 4]) -> Confusion4 {
        let mut cm = [[0i64; 4]; 4];
        for t in 0..4 {
            for p in 0..4 {
                cm[t][p] = (pri[t] * m[t][p] * 1_000_000.0).round() as i64;
            }
        }
        cm
    }

    /// One fixed engine: decent wake and light, deep leaking to light as every published non-EEG
    /// stager reports, mediocre REM.
    const FIXED: [[f64; 4]; 4] = [
        [0.70, 0.20, 0.02, 0.08],
        [0.10, 0.78, 0.06, 0.06],
        [0.02, 0.60, 0.30, 0.08],
        [0.12, 0.40, 0.03, 0.45],
    ];

    #[test]
    fn recall_is_prevalence_invariant_and_f1_is_not() {
        let cohorts = [
            [0.250, 0.610, 0.034, 0.106],
            [0.100, 0.550, 0.200, 0.150],
            [0.450, 0.400, 0.050, 0.100],
        ];
        let cms: Vec<Confusion4> = cohorts.into_iter().map(|p| under(p, FIXED)).collect();
        for c in 0..4 {
            let rs: Vec<f64> = cms.iter().map(|cm| recall(cm, c).unwrap()).collect();
            let fs: Vec<f64> = cms.iter().map(|cm| f1(cm, c).unwrap()).collect();
            let span = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min);
            assert!(span(&rs) < 1e-4, "class {c} recall moved across cohorts: {rs:?}");
            assert!(span(&fs) > 0.05, "class {c} F1 should move with prevalence: {fs:?}");
        }
        // Deep is the class it distorts most, and in the direction that flatters a deep-rich cohort.
        assert!(f1(&cms[1], 2).unwrap() - f1(&cms[0], 2).unwrap() > 0.15);
        for cm in &cms {
            assert!((balanced_accuracy(cm).unwrap() - 0.5575).abs() < 1e-3);
        }
    }

    #[test]
    fn min_recall_reports_the_class_the_mean_hides() {
        let cm = under([0.25, 0.61, 0.034, 0.106], FIXED);
        assert!((min_recall(&cm).unwrap() - 0.30).abs() < 1e-6);
        // A class that never occurs is absent from both, rather than scoring zero.
        let mut none_deep = cm;
        none_deep[2] = [0; 4];
        assert!(min_recall(&none_deep).unwrap() > 0.30);
    }

    #[test]
    fn per_recording_averages_nights_not_epochs_and_counts_who_could_answer() {
        let big = under([0.25, 0.61, 0.034, 0.106], FIXED);
        let mut small = big;
        for row in small.iter_mut() {
            for v in row.iter_mut() {
                *v /= 100;
            }
        }
        let mut no_deep = big;
        no_deep[2] = [0; 4];

        // Pooling lets the long night dominate; averaging recordings gives each one vote.
        let s = per_recording(&[big, small, no_deep], |cm| recall(cm, 2)).unwrap();
        assert_eq!(s.n, 2, "the deep-free night cannot answer and must not be counted as zero");
        assert!((s.mean - 0.30).abs() < 1e-3);
        assert!(s.sd < 1e-3);
        assert!(per_recording(&[no_deep], |cm| recall(cm, 2)).is_none());
    }

    /// Position is not identity. An arm that cannot score a night drops it, and a positional zip
    /// then charges every later night against a different one - here 0.08 instead of 0.03.
    #[test]
    fn pairing_is_by_recording_and_not_by_position() {
        use std::collections::BTreeMap;
        let base = BTreeMap::from([(0, 0.30), (1, 0.40), (2, 0.50)]);
        // Night 1 missing from the arm, and inserted out of order so key order is the only order.
        let arm = BTreeMap::from([(2, 0.55), (0, 0.31)]);

        let (b, a) = pair_by_id(&base, &arm);
        assert_eq!(b, vec![0.30, 0.50], "the night the arm could not score must leave the pair");
        assert_eq!(a, vec![0.31, 0.55], "and the survivors must stay in id order, not insert order");
        let paired = paired_bar(&b.iter().zip(&a).map(|(x, y)| y - x).collect::<Vec<_>>())
            .expect("two pairs")
            .0;
        assert!((paired - 0.03).abs() < 1e-12, "paired by id: {paired}");

        // The answer a positional zip gives on the same two series, which is a different night pair.
        let wrong: Vec<f64> = base.values().zip(arm.values()).map(|(x, y)| y - x).collect();
        let by_position = paired_bar(&wrong).expect("two pairs").0;
        assert!((by_position - 0.08).abs() < 1e-12, "by position: {by_position}");
        assert!(
            (by_position - paired).abs() > 0.04,
            "a case the two orderings agree on cannot prove the pairing: {by_position} vs {paired}"
        );
    }

    /// The guard is on AHEAD alone. An arm that is behind or inside its bar is already not being
    /// quoted as a win, and flagging those catches honest noise: `sleep-accel / conditioned
    /// diagonal, beta 0.25` matches at +0.0037 with min recall down 0.0030 and needs no catching.
    #[test]
    fn only_an_ahead_bought_by_a_fallen_min_recall_is_degenerate() {
        assert_eq!(Verdict::Matches, paired_verdict(0.0037, 0.0050, Some(-0.0030)));
        assert!(matches!(paired_verdict(-0.0375, 0.0215, Some(-0.1003)), Verdict::Behind(_)));
        // Equal is not a fall, and the baseline arm itself has nothing to difference against.
        assert!(matches!(paired_verdict(0.0375, 0.0215, Some(0.0)), Verdict::Ahead(_)));
        assert!(matches!(paired_verdict(0.0375, 0.0215, None), Verdict::Ahead(_)));
        // A rise is the honest shape: aauwss / cardiac emission lambda 1.0 moved min recall
        // 0.4232 -> 0.5651. Its own headline sat inside its bar, so raise it past one here.
        assert!(matches!(paired_verdict(0.0500, 0.0468, Some(0.1419)), Verdict::Ahead(_)));
        // Exactly on the bar is not a gap in either direction, whatever the recalls did.
        assert_eq!(Verdict::Matches, paired_verdict(0.0215, 0.0215, Some(-0.5)));
        assert_eq!("DEGENERATE (1.74x, min recall -0.1003)",
                   format!("{}", paired_verdict(0.0375, 0.0215, Some(-0.1003))));
    }

    /// The measured incident, from `dev-notes/_r11/border_report_no_time_term.txt`: the arm every
    /// other reading called the worst printed the card's ONLY AHEAD in 50 paired arms, having given
    /// rem up. Its worst class recall had fallen 0.1723 -> 0.0720, and that is what now reads.
    #[test]
    fn the_dreamt_time_term_ablation_reads_degenerate_not_ahead() {
        // Row marginals are the measured dreamt truth shares over a nominal 100,000 epochs and the
        // diagonals the measured per-class recalls. `min_recall` reads rows, so the scale cancels.
        let rows = [25_100i64, 61_000, 3_400, 10_500];
        let build = |recalls: [f64; 4]| -> Confusion4 {
            let mut cm = [[0i64; 4]; 4];
            for c in 0..4 {
                let hit = (rows[c] as f64 * recalls[c]).round() as i64;
                cm[c][c] = hit;
                // Every miss into one other column; which one cannot move a row's recall.
                cm[c][if c == WAKE { 1 } else { WAKE }] = rows[c] - hit;
            }
            cm
        };
        let null = build([0.427, 0.780, 0.1723, 0.388]);
        let ablated = build([0.443, 0.876, 0.098, 0.0720]);

        let (n_min, a_min) = (min_recall(&null).unwrap(), min_recall(&ablated).unwrap());
        assert!((n_min - 0.1723).abs() < 1e-3, "measured null min recall: {n_min}");
        assert!((a_min - 0.0720).abs() < 1e-3, "measured ablated min recall: {a_min}");
        let delta = a_min - n_min;
        assert!((delta + 0.1003).abs() < 1e-3, "the measured fall: {delta}");

        // The measured paired headline over the same 100 nights: +0.0375 against a bar of 0.0215.
        let v = paired_verdict(0.0375, 0.0215, Some(delta));
        assert!(matches!(v, Verdict::Degenerate { .. }), "the card's only AHEAD in 50: {v}");
        assert!(format!("{v}").contains("min recall -0.100"), "it must name the drop: {v}");
        // And without the guard it is the reading that was printed, quoted, and believed.
        assert_eq!(Verdict::Ahead(0.0375 / 0.0215), paired_verdict(0.0375, 0.0215, None));
    }
}
