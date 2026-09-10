//! The R-R order-statistic family as an EMISSION: v2's four class scores plus a scaled linear
//! discriminant over the fourteen percentile columns of [`cardiac::Block::row`].
//!
//! Reached through `pipeline::EmitCfg::V2PlusCardiac`. The columns are built here from the night's
//! own R-R on v2's epoch grid, so the rows a harness fits on and the rows the emission reads are
//! produced by one function. The map itself is fitted elsewhere, held out by recording.

use core::hash::{Hash, Hasher};

use super::cardiac::{self, reconstruct_beats, WINDOW_S};
use super::features::EPOCH_S;
use super::v2::{epoch_starts, Prepared, STAGE_ORDER};
use super::{SleepInput, SleepStage};
use crate::lda::{Lda, CLASSES};

/// Columns the emission reads: [`cardiac::Block::row`] 0..14, seven absolute quantiles then seven
/// detrended. `mean_hr` and the three difference summaries are v2's business, not this arm's.
pub const COLS: usize = 14;

/// Class order the discriminant is FITTED in — the truth encoding of the PSG corpora. The emission
/// is in [`STAGE_ORDER`], which is a different order; [`CardiacEmit::delta`] maps by name.
pub const FIT_ORDER: [SleepStage; CLASSES] =
    [SleepStage::Wake, SleepStage::Light, SleepStage::Deep, SleepStage::Rem];

/// Position of a stage in [`FIT_ORDER`]. Both orders are constants, so the mapping is fixed at
/// compile time rather than assumed from position.
fn fit_index(s: SleepStage) -> usize {
    match FIT_ORDER.iter().position(|x| *x == s) {
        Some(i) => i,
        None => panic!("every stage must appear in FIT_ORDER"),
    }
}

/// A fitted cardiac emission: the per-class affine map over [`COLS`], and the scale it is added to
/// v2's emission at. Standardisation is folded into the coefficients, so this is `Copy` and the
/// whole `SleepConfig` stays a value.
#[derive(Clone, Copy, Debug)]
pub struct CardiacEmit {
    /// Per class in [`FIT_ORDER`], one coefficient per column.
    weight: [[f64; COLS]; CLASSES],
    /// Per class in [`FIT_ORDER`], the discriminant at the all-zero row.
    bias: [f64; CLASSES],
    /// Thousandths. An integer so the config stays `Eq + Hash`; 0 is v2 unchanged.
    pub lambda_milli: u32,
}

impl CardiacEmit {
    /// Read the fitted map out of an [`Lda`] over exactly [`COLS`] columns by probing its
    /// discriminants at the origin and at each basis vector — the rule is affine, so that is the
    /// map. `None` when the fit is a different width.
    pub fn from_lda(lda: &Lda, lambda_milli: u32) -> Option<Self> {
        let bias = lda.scores(&[0.0; COLS])?;
        let mut weight = [[0.0; COLS]; CLASSES];
        for j in 0..COLS {
            let mut e = [0.0; COLS];
            e[j] = 1.0;
            let s = lda.scores(&e)?;
            for c in 0..CLASSES {
                weight[c][j] = s[c] - bias[c];
            }
        }
        Some(CardiacEmit { weight, bias, lambda_milli })
    }

    /// The scale itself.
    pub fn lambda(&self) -> f64 {
        f64::from(self.lambda_milli) / 1000.0
    }

    /// What this arm ADDS to v2's emission row, in [`STAGE_ORDER`] columns. A row carrying any
    /// missing column contributes nothing rather than a manufactured zero-valued feature.
    pub fn delta(&self, row: &[f64; COLS]) -> [f64; CLASSES] {
        if row.iter().any(|v| !v.is_finite()) {
            return [0.0; CLASSES];
        }
        let lambda = self.lambda();
        core::array::from_fn(|c| {
            let k = fit_index(STAGE_ORDER[c]);
            let d: f64 = (0..COLS).map(|j| row[j] * self.weight[k][j]).sum::<f64>() + self.bias[k];
            lambda * d
        })
    }

    /// Every f64 as its bit pattern, so equality and hashing are total. Two fits differing in one
    /// coefficient must not collide in anything keyed on `SleepConfig`.
    fn bits(&self) -> ([[u64; COLS]; CLASSES], [u64; CLASSES], u32) {
        (
            core::array::from_fn(|c| core::array::from_fn(|j| self.weight[c][j].to_bits())),
            core::array::from_fn(|c| self.bias[c].to_bits()),
            self.lambda_milli,
        )
    }
}

impl PartialEq for CardiacEmit {
    fn eq(&self, other: &Self) -> bool {
        self.bits() == other.bits()
    }
}

impl Eq for CardiacEmit {}

impl Hash for CardiacEmit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits().hash(state);
    }
}

/// The fourteen columns per prepared epoch, z-scored within the night, `None` where the window holds
/// fewer than [`cardiac::MIN_BEATS`] beats or a column has no spread across the night. Windows are
/// [`WINDOW_S`] centred on the epoch. One producer: a harness fits on exactly these rows.
pub fn columns(input: &SleepInput, prep: &Prepared) -> Vec<Option<[f64; COLS]>> {
    let beats = reconstruct_beats(&input.rr);
    let starts = epoch_starts(prep);
    let raw: Vec<Option<[f64; COLS]>> = starts
        .iter()
        .map(|s| {
            let centre = *s as f64 + EPOCH_S as f64 / 2.0;
            let (lo, hi) = (centre - WINDOW_S / 2.0, centre + WINDOW_S / 2.0);
            // Beats ascend, so the window is a slice; filtering the whole night per epoch is the
            // same answer and quadratic in the night.
            let a = beats.partition_point(|b| b.0 < lo);
            let b = beats.partition_point(|b| b.0 <= hi);
            cardiac::extract(&beats[a..b], lo, hi).map(|blk| {
                let r = blk.row();
                core::array::from_fn(|j| r[j])
            })
        })
        .collect();

    let mut z = vec![[f64::NAN; COLS]; starts.len()];
    for j in 0..COLS {
        let col: Vec<Option<f64>> =
            raw.iter().map(|r| r.map(|v| v[j]).filter(|x| x.is_finite())).collect();
        for (i, v) in cardiac::zscore_column(&col).into_iter().enumerate() {
            z[i][j] = v.unwrap_or(f64::NAN);
        }
    }
    z.into_iter().map(|r| r.iter().all(|x| x.is_finite()).then_some(r)).collect()
}

/// v2's emissions with the fitted discriminant added. `cols` is index-for-index with `base`; an
/// epoch with no columns keeps v2's row exactly.
pub fn add(
    base: &[[f64; CLASSES]],
    cols: &[Option<[f64; COLS]>],
    emit: &CardiacEmit,
) -> Vec<[f64; CLASSES]> {
    base.iter()
        .enumerate()
        .map(|(i, row)| match cols.get(i).copied().flatten() {
            Some(x) => {
                let d = emit.delta(&x);
                core::array::from_fn(|c| row[c] + d[c])
            }
            None => *row,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::input::{AccelSample, HrSample, RrRun};
    use crate::sleep::params::Params;
    use crate::sleep::pipeline::{run_to, DecodeCfg, EmitCfg, SleepConfig, StepId};
    use crate::sleep::v2::prepare;

    /// A night with real beats in every window: four 40-minute phases with different rates and
    /// different beat-to-beat spread, so the percentile columns have something to separate.
    fn night() -> SleepInput {
        let start = 1_749_517_200i64;
        let dur = 4 * 40 * 60;
        let (mut accel, mut hr, mut rr) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..dur {
            let ph = (i / (40 * 60)) as usize;
            let restless = ph == 3 && (i % 20) < 6;
            let j = if restless { 0.25 } else { 0.0 };
            accel.push(AccelSample { ts: start + i, x: j, y: 0.0, z: 1.0 - j });
            let bpm = [50i64, 58, 54, 70][ph] + (i / 30) % 3;
            hr.push(HrSample { ts: start + i, bpm: bpm as u16 });
            let wobble = [4i64, 40, 16, 24][ph] * [0, 1, 0, -1][(i % 4) as usize];
            let ms = 60_000 / bpm + wobble;
            // Two beats on some seconds, the way the wire delivers them.
            let mut run = vec![ms as u16];
            if i % 3 == 0 {
                run.push((ms - 6) as u16);
            }
            rr.push(RrRun { ts: start + i, intervals: run });
        }
        SleepInput { start, end: start + dur, hr, rr, accel }
    }

    /// A fit over the night's own columns against a label that is a function of the phase, which is
    /// enough for a non-degenerate map. What it predicts is not the subject; that it MOVES is.
    fn fitted(lambda_milli: u32) -> CardiacEmit {
        let input = night();
        let prep = prepare(&input, &Params::SHIPPED);
        let cols = columns(&input, &prep);
        let starts = epoch_starts(&prep);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (i, c) in cols.iter().enumerate() {
            if let Some(row) = c {
                x.push(row.to_vec());
                y.push((((starts[i] - input.start) / (40 * 60)) as usize) % CLASSES);
            }
        }
        let lda = Lda::fit(&x, &y, 1e-2).expect("four phases give four classes");
        CardiacEmit::from_lda(&lda, lambda_milli).expect("fourteen columns")
    }

    /// THE mapping test. `STAGE_ORDER` is `[Deep, Rem, Light, Wake]` and the fit is in
    /// `[Wake, Light, Deep, Rem]`; mapping by position instead of by name relabels every class.
    #[test]
    fn the_emission_column_of_a_stage_carries_that_stages_discriminant() {
        assert_eq!([2, 3, 1, 0].map(|i| FIT_ORDER[i]), STAGE_ORDER, "the two orders must differ");
        assert_eq!(2, fit_index(SleepStage::Deep));
        assert_eq!(3, fit_index(SleepStage::Rem));
        assert_eq!(1, fit_index(SleepStage::Light));
        assert_eq!(0, fit_index(SleepStage::Wake));

        // A map that scores exactly one fitted class, and only that class's emission column moves.
        for k in 0..CLASSES {
            let mut e = CardiacEmit {
                weight: [[0.0; COLS]; CLASSES],
                bias: [0.0; CLASSES],
                lambda_milli: 1000,
            };
            e.bias[k] = 5.0;
            let d = e.delta(&[0.0; COLS]);
            let want = STAGE_ORDER.iter().position(|s| *s == FIT_ORDER[k]).unwrap();
            for (c, v) in d.iter().enumerate() {
                assert_eq!(if c == want { 5.0 } else { 0.0 }, *v, "class {k} landed in column {c}");
            }
        }
    }

    /// The probe must reproduce the discriminant it was read out of, or the emission is a different
    /// model from the one that was fitted and scored.
    #[test]
    fn the_probed_map_reproduces_the_discriminant_it_came_from() {
        let input = night();
        let prep = prepare(&input, &Params::SHIPPED);
        let cols = columns(&input, &prep);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (i, c) in cols.iter().enumerate() {
            if let Some(row) = c {
                x.push(row.to_vec());
                y.push(i % CLASSES);
            }
        }
        let lda = Lda::fit(&x, &y, 1e-2).unwrap();
        let e = CardiacEmit::from_lda(&lda, 1000).unwrap();
        let mut checked = 0;
        for row in cols.iter().flatten() {
            let want = lda.scores(row).unwrap();
            let got = e.delta(row);
            for c in 0..CLASSES {
                let k = fit_index(STAGE_ORDER[c]);
                assert!((got[c] - want[k]).abs() < 1e-9, "column {c}: {} vs {}", got[c], want[k]);
            }
            checked += 1;
        }
        assert!(checked > 50, "only {checked} rows compared");
    }

    #[test]
    fn lambda_scales_the_delta_and_a_missing_row_contributes_nothing() {
        let one = fitted(1000);
        let half = CardiacEmit { lambda_milli: 500, ..one };
        let row = [0.4, -0.2, 0.1, 0.9, -1.1, 0.3, 0.7, 0.2, -0.6, 0.0, 0.5, -0.3, 0.8, -0.9];
        let (a, b) = (one.delta(&row), half.delta(&row));
        assert!(a.iter().any(|v| v.abs() > 1e-9), "a fitted map must move something");
        for c in 0..CLASSES {
            assert!((a[c] / 2.0 - b[c]).abs() < 1e-12, "lambda must scale linearly");
        }
        let mut missing = row;
        missing[3] = f64::NAN;
        assert_eq!([0.0; CLASSES], one.delta(&missing), "a missing column is not a zero feature");

        let base = vec![[1.0, 2.0, 3.0, 4.0]; 2];
        assert_eq!(base, add(&base, &[None, None], &one), "no columns means v2 unchanged");
    }

    /// The seam's contract: at lambda 0 the arm IS the shipped emission, digest and labels alike.
    /// `golden_tests::golden_input` is private to its own module, so the night is built here.
    #[test]
    fn lambda_zero_reproduces_the_shipped_hypnogram_and_a_nonzero_lambda_moves_it() {
        let input = night();
        let p = Params::SHIPPED;
        let arm = |lambda_milli: u32| SleepConfig {
            emit: EmitCfg::V2PlusCardiac(CardiacEmit { lambda_milli, ..fitted(1000) }),
            decode: DecodeCfg::Viterbi,
        };
        let null = run_to(&input, &SleepConfig::shipped(), &p, StepId::Decode);
        let zero = run_to(&input, &arm(0), &p, StepId::Decode);
        let moved = run_to(&input, &arm(2000), &p, StepId::Decode);

        assert_eq!(
            null.digest_of(StepId::Emit),
            zero.digest_of(StepId::Emit),
            "lambda 0 must emit v2"
        );
        assert_eq!(null.stages, zero.stages, "lambda 0 must stage label for label");
        assert!(null.stages.as_ref().is_some_and(|s| s.len() > 100), "the night must stage");
        assert_ne!(
            null.digest_of(StepId::Emit),
            moved.digest_of(StepId::Emit),
            "lambda 2 must emit differently"
        );
        assert_ne!(null.stages, moved.stages, "and it must reach the labels");
    }

    /// The window reads the night's own beats. A night with no R-R carries no columns at all, which
    /// is the case every non-R-R cohort takes.
    #[test]
    fn columns_are_present_where_beats_are_and_absent_where_they_are_not() {
        let input = night();
        let prep = prepare(&input, &Params::SHIPPED);
        let cols = columns(&input, &prep);
        let have = cols.iter().flatten().count();
        assert!(have > cols.len() / 2, "{have} of {} epochs carried columns", cols.len());

        let bare = SleepInput { rr: Vec::new(), ..night() };
        let bare_prep = prepare(&bare, &Params::SHIPPED);
        assert_eq!(0, columns(&bare, &bare_prep).iter().flatten().count(), "no beats, no columns");
    }

    /// Equality and hashing are on the coefficients, not on lambda alone: two folds' fits at the
    /// same lambda are different configs and must not collide.
    #[test]
    fn two_different_fits_are_different_configs() {
        use std::collections::HashSet;
        let a = fitted(1000);
        let mut b = a;
        b.weight[0][0] += 1e-9;
        assert_ne!(a, b);
        assert_eq!(a, fitted(1000), "the same fit twice is the same config");
        let set: HashSet<EmitCfg> =
            [EmitCfg::V2, EmitCfg::V2PlusCardiac(a), EmitCfg::V2PlusCardiac(b)]
                .into_iter()
                .collect();
        assert_eq!(3, set.len(), "hash must separate them too");
    }
}
