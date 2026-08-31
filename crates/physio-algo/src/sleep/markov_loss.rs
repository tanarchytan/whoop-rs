//! A staging loss that prices misplaced BOUNDARIES as well as misclassified epochs.
//!
//! Three costs: one per wrong epoch, one for calling a boundary that is not there, one for missing
//! a boundary that is. All three at 1.0 scores like plain accuracy; raising the invented-boundary
//! cost is what discourages fragmenting a continuous bout. Nothing here decodes - it scores a
//! finished hypnogram against truth, which is what selecting the costs needs first.

/// Per-class epoch cost plus the two boundary costs. `fc` is per class because a missed deep epoch
/// and a missed wake epoch are not obliged to cost the same, and we have never decided that they do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub fc: [f64; 4],
    /// Charged when the prediction puts a boundary where truth has none.
    pub ft: f64,
    /// Charged when truth has a boundary and the prediction runs straight through it.
    pub fh: f64,
}

impl Costs {
    /// All costs 1.0. Scores the same ordering as counting mistakes, so it is the arm every other
    /// setting is compared against.
    pub const UNIT: Costs = Costs { fc: [1.0; 4], ft: 1.0, fh: 1.0 };

    /// Epoch costs only. The boundary terms vanish and this reduces to a weighted error count.
    pub fn epochs_only(fc: [f64; 4]) -> Costs {
        Costs { fc, ft: 0.0, fh: 0.0 }
    }
}

/// When the invented-boundary cost applies to a boundary that IS at the right index but goes to the
/// wrong state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionRule {
    /// `ft` only where truth genuinely held. A boundary at the right index costs its epoch errors
    /// and nothing more, which keeps `ft` a pure "you fragmented a continuous bout" penalty.
    OnlyInvented,
    /// `ft` whenever the predicted pair is not the true pair. A boundary to the wrong state pays
    /// twice: once as a wrong epoch, once as a wrong boundary.
    AnyMismatch,
}

/// The boundary term for one adjacent pair.
fn boundary(pred: (usize, usize), truth: (usize, usize), c: &Costs, rule: TransitionRule) -> f64 {
    let (moved_p, moved_t) = (pred.0 != pred.1, truth.0 != truth.1);
    match (moved_p, moved_t) {
        (false, true) => c.fh,
        (true, false) => c.ft,
        (true, true) if rule == TransitionRule::AnyMismatch && pred != truth => c.ft,
        _ => 0.0,
    }
}

/// Total loss of a predicted hypnogram: one epoch cost per position plus one boundary cost per
/// adjacent pair. `None` if the two differ in length or are empty, which is a caller bug and not a
/// zero-cost staging.
pub fn sequence_loss(
    pred: &[usize],
    truth: &[usize],
    c: &Costs,
    rule: TransitionRule,
) -> Option<f64> {
    if pred.is_empty() || pred.len() != truth.len() {
        return None;
    }
    let mut total = 0.0;
    for (p, t) in pred.iter().zip(truth) {
        if p != t {
            total += c.fc[*t];
        }
    }
    for i in 0..pred.len() - 1 {
        total += boundary((pred[i], pred[i + 1]), (truth[i], truth[i + 1]), c, rule);
    }
    Some(total)
}

/// The cost of one adjacent pair with its FIRST epoch charged, which is the form that sums over
/// overlapping pairs without charging an epoch twice.
pub fn pair_loss(
    pred: (usize, usize),
    truth: (usize, usize),
    c: &Costs,
    rule: TransitionRule,
) -> f64 {
    let epoch = if pred.0 != truth.0 { c.fc[truth.0] } else { 0.0 };
    epoch + boundary(pred, truth, c, rule)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAIRS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

    /// The published two-state cost matrix, as symbol sets, rows = predicted pair, cols = true pair.
    /// Encoded so a change to [`boundary`] has to keep reproducing it.
    fn published(pred: (usize, usize), truth: (usize, usize)) -> &'static [&'static str] {
        match (pred, truth) {
            ((0, 0), (0, 0)) | ((0, 1), (0, 1)) | ((1, 0), (1, 0)) | ((1, 1), (1, 1)) => &[],
            ((0, 0), (0, 1)) | ((1, 1), (1, 0)) => &["FH"],
            ((0, 1), (0, 0)) | ((1, 0), (1, 1)) => &["FT"],
            ((0, 0), (1, 1)) | ((1, 1), (0, 0)) => &["FC"],
            ((0, 0), (1, 0)) | ((1, 1), (0, 1)) => &["FC", "FH"],
            _ => &["FC", "FT"],
        }
    }

    fn value(symbols: &[&str], c: &Costs, truth_first: usize) -> f64 {
        symbols
            .iter()
            .map(|s| match *s {
                "FC" => c.fc[truth_first],
                "FT" => c.ft,
                _ => c.fh,
            })
            .sum()
    }

    #[test]
    fn any_mismatch_reproduces_the_published_matrix_and_only_invented_differs_on_two_cells() {
        let c = Costs { fc: [3.0, 5.0, 0.0, 0.0], ft: 7.0, fh: 11.0 };
        let (mut matched, mut differed) = (0, 0);
        for pred in PAIRS {
            for truth in PAIRS {
                let want = value(published(pred, truth), &c, truth.0);
                assert_eq!(
                    pair_loss(pred, truth, &c, TransitionRule::AnyMismatch),
                    want,
                    "pred {pred:?} truth {truth:?}"
                );
                if pair_loss(pred, truth, &c, TransitionRule::OnlyInvented) == want {
                    matched += 1;
                } else {
                    differed += 1;
                }
            }
        }
        assert_eq!(matched, 14);
        assert_eq!(differed, 2, "the two cells are the opposite-direction boundaries");
    }

    #[test]
    fn the_two_cells_are_exactly_the_opposite_direction_boundaries() {
        let c = Costs { fc: [1.0; 4], ft: 7.0, fh: 11.0 };
        for (pred, truth) in [((0, 1), (1, 0)), ((1, 0), (0, 1))] {
            let invented = pair_loss(pred, truth, &c, TransitionRule::OnlyInvented);
            let mismatch = pair_loss(pred, truth, &c, TransitionRule::AnyMismatch);
            assert_eq!(mismatch - invented, 7.0, "{pred:?} vs {truth:?}");
        }
    }

    /// Four classes can hold a boundary at the right index that goes to the wrong state. Two classes
    /// cannot, so the published matrix does not decide this case and the rules split on it.
    #[test]
    fn a_right_index_wrong_destination_boundary_is_free_only_under_only_invented() {
        let c = Costs { fc: [1.0; 4], ft: 7.0, fh: 11.0 };
        let (pred, truth) = ((1, 2), (1, 3));
        assert_eq!(pair_loss(pred, truth, &c, TransitionRule::OnlyInvented), 0.0);
        assert_eq!(pair_loss(pred, truth, &c, TransitionRule::AnyMismatch), 7.0);
    }

    #[test]
    fn unit_costs_score_a_plain_mistake_count_when_boundaries_are_ignored() {
        let pred = [0, 1, 1, 2, 0];
        let truth = [0, 1, 2, 2, 1];
        let c = Costs::epochs_only([1.0; 4]);
        let got = sequence_loss(&pred, &truth, &c, TransitionRule::OnlyInvented).unwrap();
        assert_eq!(got, 2.0);
    }

    #[test]
    fn fragmenting_a_continuous_bout_costs_only_under_a_nonzero_invented_cost() {
        let truth = [2; 10];
        let mut split = truth;
        split[5] = 1;
        let free = Costs { fc: [1.0; 4], ft: 0.0, fh: 0.0 };
        let priced = Costs { fc: [1.0; 4], ft: 4.0, fh: 0.0 };
        let r = TransitionRule::OnlyInvented;
        assert_eq!(sequence_loss(&split, &truth, &free, r).unwrap(), 1.0);
        // One wrong epoch, but TWO invented boundaries - into the wrong state and back out.
        assert_eq!(sequence_loss(&split, &truth, &priced, r).unwrap(), 9.0);
    }

    #[test]
    fn a_missed_boundary_is_charged_to_the_class_that_was_there() {
        let truth = [1, 1, 1, 2, 2, 2];
        let flat = [1; 6];
        let c = Costs { fc: [0.0, 0.0, 6.0, 0.0], ft: 0.0, fh: 5.0 };
        // Three wrong deep epochs at 6, plus the one boundary run straight through.
        assert_eq!(sequence_loss(&flat, &truth, &c, TransitionRule::OnlyInvented).unwrap(), 23.0);
    }

    #[test]
    fn a_length_mismatch_is_not_a_free_staging() {
        let c = Costs::UNIT;
        let r = TransitionRule::OnlyInvented;
        assert!(sequence_loss(&[0, 1], &[0], &c, r).is_none());
        assert!(sequence_loss(&[], &[], &c, r).is_none());
    }
}
