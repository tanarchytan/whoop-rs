//! Hypnogram scoring: one confusion matrix and the numbers read off it.
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

/// Cohen's kappa over four classes. Zero for an empty matrix and for any matrix where chance already
/// explains everything, so a scorer that emits one stage all night cannot reach a target by doing nothing.
pub fn kappa4(cm: &Confusion4) -> f64 {
    let tot: i64 = cm.iter().flatten().sum();
    if tot == 0 {
        return 0.0;
    }
    let tot = tot as f64;
    let po = (0..4).map(|i| cm[i][i]).sum::<i64>() as f64 / tot;
    let mut pe = 0.0;
    for (j, row_j) in cm.iter().enumerate() {
        let col: i64 = cm.iter().map(|r| r[j]).sum();
        let row: i64 = row_j.iter().sum();
        pe += col as f64 * row as f64;
    }
    pe /= tot * tot;
    if pe >= 1.0 { 0.0 } else { (po - pe) / (1.0 - pe) }
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

/// Bout-level agreement for one class, with the arm that stops it being gamed. Predicting the class
/// everywhere detects every bout, so `spurious` and [`BoutScore::precision`] are reported beside it.
///
/// [`BoutScore::coverage`] is the number to read first. `detected` is BINARY at `min_overlap`, which
/// makes a long bout all-or-nothing: a 409-minute wake bout found at 22% scores exactly the same as
/// one found at 0%, and a change that lifted it to 27% was reported as "unchanged" for a whole
/// session. Coverage is continuous and sees that.
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

#[cfg(test)]
mod tests {
    use super::*;

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
    /// panics on real input. Found by a mutation sweep, not by anything failing.
    #[test]
    fn an_out_of_range_index_is_ignored_rather_than_indexed() {
        const UNLABELLED: usize = 4;
        let cm = confusion4(&[UNLABELLED, 1, UNLABELLED], &[0, 1, UNLABELLED]);
        assert_eq!(cm.iter().flatten().sum::<i64>(), 1, "only the one in-range pair is scored");
        assert_eq!(cm[1][1], 1);
    }

    /// Specificity must exclude the class's OWN row: everything in it is a positive, so counting it as
    /// a negative inflates the score. A mutation removing the skip went unnoticed because the existing
    /// case had an all-diagonal class row, where including it changes nothing.
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
    /// `recall` calls both a total miss and cannot tell them apart, while coverage sees the gap. A
    /// real 409-minute reading bout sat at 22% and a change lifting it to 27% was reported as
    /// "unchanged" for a whole session because of exactly this.
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
}
