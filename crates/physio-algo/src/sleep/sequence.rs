//! Hypnogram STRUCTURE: the properties a confusion matrix cannot see.
//!
//! Every number in [`metrics`](super::metrics) is a function of the confusion matrix, and a confusion
//! matrix is invariant to shuffling the hypnogram - permute both series the same way and it is
//! unchanged while every bout is destroyed. So a change aimed at run length cannot be read there at
//! all. This module counts adjacent-epoch transitions, bout lengths and their distance to truth.
//!
//! Everything is accumulated over time-CONTIGUOUS segments: a night whose labels have holes is added
//! as several segments, so no transition is ever counted across a gap or across a night boundary.
//! Both the prediction and the truth are cut at the same edges, which is what makes them comparable.

/// Per-class occupancy, adjacent-pair transitions and bout lengths over one or more segments.
#[derive(Clone, Debug)]
pub struct Structure {
    /// Epochs held by each class.
    pub occupancy: [i64; 4],
    /// `[from][to]` over adjacent epoch pairs, the diagonal (staying) included.
    pub trans: [[i64; 4]; 4],
    /// Every maximal run's length in epochs, per class, censored runs included.
    pub bouts: [Vec<usize>; 4],
    /// Runs touching a segment edge, whose true length is at least what was recorded.
    pub censored: [usize; 4],
    pub segments: usize,
}

impl Default for Structure {
    fn default() -> Self {
        Self {
            occupancy: [0; 4],
            trans: [[0; 4]; 4],
            bouts: std::array::from_fn(|_| Vec::new()),
            censored: [0; 4],
            segments: 0,
        }
    }
}

impl Structure {
    /// Add one time-contiguous segment. Out-of-range labels are dropped, which breaks the segment
    /// rather than inventing an adjacency across the hole.
    pub fn add(&mut self, seq: &[usize]) {
        if seq.iter().any(|c| *c >= 4) {
            for part in seq.split(|c| *c >= 4) {
                self.add(part);
            }
            return;
        }
        if seq.is_empty() {
            return;
        }
        self.segments += 1;
        for c in seq {
            self.occupancy[*c] += 1;
        }
        for w in seq.windows(2) {
            self.trans[w[0]][w[1]] += 1;
        }
        let (mut start, mut i) = (0usize, 0usize);
        while i < seq.len() {
            let c = seq[i];
            while i < seq.len() && seq[i] == c {
                i += 1;
            }
            self.bouts[c].push(i - start);
            if start == 0 || i == seq.len() {
                self.censored[c] += 1;
            }
            start = i;
        }
    }

    /// Build from one contiguous segment.
    pub fn one(seq: &[usize]) -> Self {
        let mut s = Self::default();
        s.add(seq);
        s
    }

    /// Build from segments in one call.
    pub fn of(segments: &[Vec<usize>]) -> Self {
        let mut s = Self::default();
        for seg in segments {
            s.add(seg);
        }
        s
    }

    fn pairs(&self) -> i64 {
        self.trans.iter().flatten().sum()
    }

    /// Share of adjacent pairs that change class. `None` when nothing is adjacent to anything.
    pub fn fi(&self) -> Option<f64> {
        let all = self.pairs();
        let same: i64 = (0..4).map(|i| self.trans[i][i]).sum();
        (all > 0).then(|| (all - same) as f64 / all as f64)
    }

    /// Share of pairs LEAVING this class. `1/fi_class` is the mean bout length only when no run of it
    /// touches the segment's END: such a run adds pairs but no exit, so it inflates the reciprocal.
    /// `None` when the class never occurs adjacent to anything.
    pub fn fi_class(&self, c: usize) -> Option<f64> {
        let from: i64 = self.trans[c].iter().sum();
        (from > 0).then(|| (from - self.trans[c][c]) as f64 / from as f64)
    }

    /// Mean bout length in epochs, censored runs counted at their observed length.
    pub fn mean_bout(&self, c: usize) -> Option<f64> {
        let b = &self.bouts[c];
        (!b.is_empty()).then(|| b.iter().sum::<usize>() as f64 / b.len() as f64)
    }

    /// Share of this class's epochs living in bouts of at least `len`. The upper-tail read: a class
    /// can hold the right number of epochs and put none of them in a long run.
    pub fn tail_mass(&self, c: usize, len: usize) -> Option<f64> {
        let total: usize = self.bouts[c].iter().sum();
        (total > 0).then(|| {
            self.bouts[c].iter().filter(|l| **l >= len).sum::<usize>() as f64 / total as f64
        })
    }

    /// Off-diagonal transitions holding less than `max_share` of all adjacent pairs. Derived from
    /// whichever structure is the REFERENCE, never imported as a constant.
    pub fn rare_set(&self, max_share: f64) -> [[bool; 4]; 4] {
        let all = self.pairs().max(1) as f64;
        std::array::from_fn(|i| {
            std::array::from_fn(|j| i != j && (self.trans[i][j] as f64 / all) < max_share)
        })
    }

    /// Rare transitions per adjacent pair. Report it for the truth in the same row: the reference's
    /// own rate is the only thing that says whether a prediction's rate is high.
    pub fn tvr(&self, rare: &[[bool; 4]; 4]) -> Option<f64> {
        let all = self.pairs();
        let hit: i64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .filter(|(i, j)| rare[*i][*j])
            .map(|(i, j)| self.trans[i][j])
            .sum();
        (all > 0).then(|| hit as f64 / all as f64)
    }
}

/// Wasserstein-1 between two bout-length samples, in epochs: the area between their empirical CDFs.
/// Unsigned, so read it beside the two mean bout lengths, which say which way.
pub fn bout_w1(a: &Structure, b: &Structure, class: usize) -> Option<f64> {
    let (x, y) = (&a.bouts[class], &b.bouts[class]);
    if x.is_empty() || y.is_empty() {
        return None;
    }
    let mut x: Vec<usize> = x.clone();
    let mut y: Vec<usize> = y.clone();
    x.sort_unstable();
    y.sort_unstable();
    let mut pts: Vec<usize> = x.iter().chain(&y).copied().collect();
    pts.sort_unstable();
    pts.dedup();
    let cdf = |v: &[usize], t: usize| v.partition_point(|q| *q <= t) as f64 / v.len() as f64;
    let mut area = 0.0;
    for w in pts.windows(2) {
        area += (cdf(&x, w[0]) - cdf(&y, w[0])).abs() * (w[1] - w[0]) as f64;
    }
    Some(area)
}

/// Absorb every run shorter than `min_len` into the class before it. The instrument's CONTROL: it
/// lengthens bouts without touching the epoch scores much, which is the failure a confusion matrix
/// cannot report. Not a staging step.
pub fn min_run_smooth(seq: &[usize], min_len: usize) -> Vec<usize> {
    let mut out = seq.to_vec();
    for _ in 0..out.len().min(16) {
        let mut changed = false;
        let (mut start, mut i) = (0usize, 0usize);
        while i < out.len() {
            let c = out[i];
            while i < out.len() && out[i] == c {
                i += 1;
            }
            if i - start < min_len && start > 0 {
                let prev = out[start - 1];
                out[start..i].fill(prev);
                changed = true;
            }
            start = i;
        }
        if !changed {
            return out;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::metrics::confusion4;

    /// The reason this module exists. Permuting both series identically leaves the confusion matrix
    /// bit-for-bit unchanged and destroys every bout, so no confusion-matrix number can read a change
    /// aimed at run length.
    #[test]
    fn a_confusion_matrix_cannot_see_what_a_shuffle_destroys() {
        let truth: Vec<usize> = (0..40).map(|i| usize::from(i >= 20)).collect();
        let pred: Vec<usize> = (0..40).map(|i| usize::from(i >= 22)).collect();
        // Deal the same indices out in a fixed interleave: same pairs, different order.
        let order: Vec<usize> = (0..40).map(|i| (i * 17) % 40).collect();
        let (st, sp): (Vec<usize>, Vec<usize>) =
            order.iter().map(|k| (truth[*k], pred[*k])).unzip();

        assert_eq!(confusion4(&pred, &truth), confusion4(&sp, &st), "the matrix must be identical");
        let before = Structure::one(&pred).fi().unwrap();
        let after = Structure::one(&sp).fi().unwrap();
        assert!(before < 0.06 && after > 0.4, "FI must move: {before:.4} -> {after:.4}");
    }

    /// Hand-computed. 10 epochs, 9 adjacent pairs, 2 of them changing class.
    #[test]
    fn fragmentation_is_changes_over_adjacent_pairs() {
        let s = Structure::one(&[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]);
        assert!((s.fi().unwrap() - 2.0 / 9.0).abs() < 1e-12, "{:?}", s.fi());
        // FIVE pairs start in class 0, not six: the final epoch is class 0 and has no successor.
        assert!((s.fi_class(0).unwrap() - 1.0 / 5.0).abs() < 1e-12, "{:?}", s.fi_class(0));
        assert!((s.fi_class(1).unwrap() - 1.0 / 4.0).abs() < 1e-12);
        assert_eq!(s.bouts[0], vec![3, 3]);
        assert_eq!(s.censored[0], 2, "both class-0 runs touch an edge");
        assert_eq!(s.censored[1], 0);
        assert!((s.mean_bout(1).unwrap() - 4.0).abs() < 1e-12);
    }

    /// `1/fi_class` is the mean bout length only where nothing is cut by the segment's end. A run
    /// that IS cut contributes pairs with no exit and inflates it - the reason the reciprocal is a
    /// convenience and [`Structure::mean_bout`] is the number to read.
    #[test]
    fn the_reciprocal_of_the_exit_rate_is_the_mean_bout_only_when_nothing_is_cut() {
        // No class-1 run touches the end: runs of 2 and 4, mean 3.
        let clean = Structure::one(&[1, 1, 0, 1, 1, 1, 1, 0]);
        assert_eq!(clean.bouts[1], vec![2, 4]);
        assert_eq!(clean.censored[1], 1, "the LEADING run is cut, and it still has an exit");
        assert!((1.0 / clean.fi_class(1).unwrap() - 3.0).abs() < 1e-12, "{:?}", clean.fi_class(1));

        // Same night with a trailing class-1 run: 7 pairs start in class 1, still only 2 exits.
        let cut = Structure::one(&[1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
        assert_eq!(cut.bouts[1], vec![2, 4, 2], "the trailing run is still recorded");
        assert!((1.0 / cut.fi_class(1).unwrap() - 3.5).abs() < 1e-12, "{:?}", cut.fi_class(1));
        assert!((cut.mean_bout(1).unwrap() - 8.0 / 3.0).abs() < 1e-12, "the mean is unaffected");
    }

    /// Segments never join. Two nights ending and starting in different classes must not produce a
    /// transition between them, which is how a per-cohort tally invents structure that never happened.
    #[test]
    fn segments_do_not_transition_into_each_other() {
        let split = Structure::of(&[vec![0, 0, 0], vec![1, 1, 1]]);
        assert_eq!(split.trans[0][1], 0, "no transition across a segment edge");
        assert_eq!(split.pairs(), 4, "3+3 epochs give 2+2 pairs, not 5");
        assert_eq!(split.segments, 2);

        let joined = Structure::one(&[0, 0, 0, 1, 1, 1]);
        assert_eq!(joined.trans[0][1], 1);
    }

    /// A hole in the labels breaks the segment rather than closing over it.
    #[test]
    fn an_out_of_range_label_splits_rather_than_bridges() {
        const GAP: usize = 4;
        let s = Structure::one(&[0, 0, GAP, 1, 1]);
        assert_eq!(s.trans[0][1], 0, "the gap must not become a transition");
        assert_eq!(s.occupancy, [2, 2, 0, 0], "the gap holds no class");
        assert_eq!(s.segments, 2);
    }

    /// THE FALSIFIER for the whole instrument: an over-smoothed variant must separate from the
    /// original on the bout distance, and it must do so where kappa is nearly silent.
    #[test]
    fn the_bout_distance_separates_an_over_smoothed_variant() {
        // Long class-0 bouts a smoother keeps, and single-epoch intrusions it erases. Only the
        // intrusions move, so epoch agreement stays high while class 1's bouts nearly double.
        let mut truth = Vec::new();
        for _ in 0..12 {
            truth.extend(std::iter::repeat_n(1usize, 7));
            truth.push(0);
            truth.extend(std::iter::repeat_n(1usize, 6));
            truth.extend(std::iter::repeat_n(0usize, 6));
        }
        let pred = truth.clone();
        let smoothed = min_run_smooth(&pred, 3);

        let t = Structure::one(&truth);
        let raw = Structure::one(&pred);
        let over = Structure::one(&smoothed);

        assert!(over.mean_bout(1).unwrap() > raw.mean_bout(1).unwrap() * 1.5, "it must over-smooth");
        assert!(over.fi().unwrap() < raw.fi().unwrap(), "and fragment less");

        let d_raw = bout_w1(&raw, &t, 1).unwrap();
        let d_over = bout_w1(&over, &t, 1).unwrap();
        assert!(d_raw < 1e-12, "an identical hypnogram is distance 0, got {d_raw}");
        assert!(d_over > 5.0, "the over-smoothed one must be far: {d_over}");

        // And the epoch-wise view barely notices: that gap IS the instrument's justification.
        let k = crate::sleep::metrics::kappa4(&confusion4(&smoothed, &truth));
        assert!(k > 0.6, "kappa stays high while every short bout is gone: {k:.4}");
    }

    /// The distance is unsigned, so a too-long and a too-short prediction can score alike. The mean
    /// bout lengths beside it are what give the direction.
    #[test]
    fn the_distance_is_unsigned_and_the_means_give_the_direction() {
        let t = Structure::one(&[1; 6].repeat(4)); // one 24-epoch bout
        let short = Structure::one(&[vec![1, 1, 0], vec![1, 1, 0]].concat().repeat(4));
        let long = Structure::one(&[1; 60]);
        assert!(bout_w1(&short, &t, 1).is_some() && bout_w1(&long, &t, 1).is_some());
        assert!(short.mean_bout(1).unwrap() < t.mean_bout(1).unwrap());
        assert!(long.mean_bout(1).unwrap() > t.mean_bout(1).unwrap());
    }

    /// The rare set comes from the reference, and the reference's OWN rate against it is what says
    /// whether a prediction's rate is high. A set derived elsewhere cannot be read at all.
    #[test]
    fn the_rare_set_is_derived_from_the_reference_and_scores_it_too() {
        // Truth moves 0<->1 freely and reaches 2 exactly once.
        let mut truth = [0usize, 0, 1, 1, 0, 0, 1, 1].repeat(8);
        truth[3] = 2;
        let t = Structure::of(&[truth]);
        let rare = t.rare_set(0.02);
        assert!(rare[1][2] || rare[2][1], "the single excursion must be rare");
        assert!(!rare[0][1] && !rare[1][0], "the common pair must not be");
        assert!(!rare[0][0], "staying is never a rare transition");

        let own = t.tvr(&rare).unwrap();
        assert!(own > 0.0 && own < 0.02 * 4.0, "the reference's own rate: {own:.4}");

        // A prediction that invents the rare move everywhere must read far above the reference.
        let noisy = Structure::one(&[0usize, 2, 1, 2].repeat(16));
        assert!(noisy.tvr(&rare).unwrap() > own * 5.0);
    }

    /// The upper tail is the deep defect's shape: same epoch count, none of it in a long run.
    #[test]
    fn tail_mass_separates_scattered_epochs_from_one_long_bout() {
        let scattered = Structure::one(&[2usize, 0, 0, 0].repeat(20));
        let massed = Structure::one(&[vec![2usize; 20], vec![0; 60]].concat());
        assert_eq!(scattered.occupancy[2], massed.occupancy[2], "same number of epochs");
        assert_eq!(scattered.tail_mass(2, 10), Some(0.0));
        assert_eq!(massed.tail_mass(2, 10), Some(1.0));
    }

    #[test]
    fn an_absent_class_answers_none_rather_than_zero() {
        let s = Structure::one(&[0usize, 0, 1, 1]);
        assert_eq!(s.mean_bout(2), None);
        assert_eq!(s.fi_class(2), None);
        assert_eq!(s.tail_mass(2, 3), None);
        assert_eq!(bout_w1(&s, &s, 2), None);
        assert_eq!(Structure::default().fi(), None);
        assert_eq!(Structure::one(&[0usize]).fi(), None, "one epoch has no adjacent pair");
    }

    /// A single pass leaves runs the absorption itself creates; the fixed point is what the control
    /// promises. Two adjacent short runs must end up in ONE class, not merely renamed.
    #[test]
    fn smoothing_runs_to_a_fixed_point() {
        let seq = vec![1usize, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1];
        let out = min_run_smooth(&seq, 3);
        let s = Structure::one(&out);
        assert!(s.bouts[0].is_empty() && s.bouts[1] == vec![13], "everything absorbed: {out:?}");
        assert_eq!(min_run_smooth(&out, 3), out, "already smooth is a fixed point");
        // A leading short run has nothing before it and must survive rather than be dropped.
        assert_eq!(min_run_smooth(&[0usize, 1, 1, 1], 3), vec![0, 1, 1, 1]);
    }
}
