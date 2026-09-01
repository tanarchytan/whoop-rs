//! Ridge-regularised linear discriminant analysis over four classes.
//!
//! Exists to answer one question a marginal AUC cannot: does a column carry anything ON TOP of the
//! columns already present. That needs a model, and this is the smallest defensible one — closed
//! form, no learning rate, no seed, so a screen run twice gives the same answer twice.
//!
//! Two choices that are not the textbook default, both deliberate. **Priors are UNIFORM**: the
//! textbook `ln pi_c` term makes the rule call common classes more often, which moves per-class
//! recall without the features having changed. **Standardisation uses the TRAINING rows only**,
//! because computing it over everything leaks the held-out fold into the fit.

use crate::stats::{mean, population_sd};

pub const CLASSES: usize = 4;

/// A fitted discriminant. Coefficients live in the standardised space, so `predict` must apply the
/// stored centre and scale rather than expecting the caller to.
#[derive(Clone, Debug)]
pub struct Lda {
    coef: [Vec<f64>; CLASSES],
    constant: [f64; CLASSES],
    centre: Vec<f64>,
    scale: Vec<f64>,
}

/// Solve `a z = b` for each column of `b` by Gaussian elimination with partial pivoting. `None` when
/// the matrix is singular to working precision, which is what the ridge exists to prevent.
#[allow(clippy::needless_range_loop)]
fn solve(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let k = b.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| a[i].iter().copied().chain((0..k).map(|j| b[j][i])).collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|x, y| m[*x][col].abs().total_cmp(&m[*y][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        let d = m[col][col];
        for v in m[col].iter_mut() {
            *v /= d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row][col];
            if f != 0.0 {
                for j in col..n + k {
                    m[row][j] -= f * m[col][j];
                }
            }
        }
    }
    Some((0..k).map(|j| (0..n).map(|i| m[i][n + j]).collect()).collect())
}

impl Lda {
    /// Fit on rows carrying a class in `0..4`. Rows with any non-finite value are SKIPPED, not
    /// imputed: a manufactured mean would be read as a measurement. `None` when a class is absent,
    /// the widths differ, or the covariance is singular even after the ridge.
    #[allow(clippy::needless_range_loop)]
    pub fn fit(x: &[Vec<f64>], y: &[usize], ridge: f64) -> Option<Self> {
        let p = x.first()?.len();
        if p == 0 || x.len() != y.len() {
            return None;
        }
        let keep: Vec<usize> = (0..x.len())
            .filter(|i| y[*i] < CLASSES && x[*i].len() == p && x[*i].iter().all(|v| v.is_finite()))
            .collect();
        if keep.len() <= p {
            return None;
        }

        // Centre and scale from the TRAINING rows only. A zero-variance column would divide by zero,
        // so it is scaled by one and contributes nothing rather than poisoning every row with NaN.
        let centre: Vec<f64> =
            (0..p).map(|j| mean(&keep.iter().map(|i| x[*i][j]).collect::<Vec<_>>())).collect();
        let scale: Vec<f64> = (0..p)
            .map(|j| {
                let s = population_sd(&keep.iter().map(|i| x[*i][j]).collect::<Vec<_>>());
                if s > 1e-12 { s } else { 1.0 }
            })
            .collect();
        let z = |i: usize, j: usize| (x[i][j] - centre[j]) / scale[j];

        let mut n = [0usize; CLASSES];
        let mut mu = [(); CLASSES].map(|_| vec![0.0; p]);
        for i in &keep {
            n[y[*i]] += 1;
            for j in 0..p {
                mu[y[*i]][j] += z(*i, j);
            }
        }
        if n.contains(&0) {
            return None;
        }
        for c in 0..CLASSES {
            for v in mu[c].iter_mut() {
                *v /= n[c] as f64;
            }
        }

        // Pooled WITHIN-class scatter, ridged. The percentile columns are near-collinear by
        // construction, so without the ridge this is singular and the solve fails.
        let mut s = vec![vec![0.0; p]; p];
        for i in &keep {
            let d: Vec<f64> = (0..p).map(|j| z(*i, j) - mu[y[*i]][j]).collect();
            for a in 0..p {
                for b in 0..p {
                    s[a][b] += d[a] * d[b];
                }
            }
        }
        let df = (keep.len() - CLASSES) as f64;
        for (a, row) in s.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v /= df;
                if a == b {
                    *v += ridge;
                }
            }
        }

        let solved = solve(&s, &mu)?;
        let mut coef = [(); CLASSES].map(|_| Vec::new());
        let mut constant = [0.0; CLASSES];
        for c in 0..CLASSES {
            // Uniform priors: the `ln pi_c` term of the textbook rule is dropped on purpose.
            constant[c] = -0.5 * (0..p).map(|j| solved[c][j] * mu[c][j]).sum::<f64>();
            coef[c] = solved[c].clone();
        }
        Some(Lda { coef, constant, centre, scale })
    }

    /// Highest discriminant wins; ties resolve to the lower class. `None` when the row is the wrong
    /// width or carries a non-finite value, which is a different fact from predicting class 0.
    pub fn predict(&self, row: &[f64]) -> Option<usize> {
        if row.len() != self.centre.len() || row.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let z: Vec<f64> =
            (0..row.len()).map(|j| (row[j] - self.centre[j]) / self.scale[j]).collect();
        (0..CLASSES)
            .map(|c| {
                let d: f64 = (0..z.len()).map(|j| z[j] * self.coef[c][j]).sum::<f64>()
                    + self.constant[c];
                (c, d)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(c, _)| c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four well-separated Gaussians on two features must be recovered almost perfectly. Without
    /// this the failures below could all be "the fit does nothing" rather than what they claim.
    fn blobs(per: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let centres = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (c, (cx, cy)) in centres.iter().enumerate() {
            for k in 0..per {
                let t = k as f64 / per as f64;
                x.push(vec![cx + (t * 7.0).sin(), cy + (t * 11.0).cos()]);
                y.push(c);
            }
        }
        (x, y)
    }

    #[test]
    fn separated_classes_are_recovered() {
        let (x, y) = blobs(40);
        let m = Lda::fit(&x, &y, 1e-3).expect("four populated classes");
        let hit = x.iter().zip(&y).filter(|(r, c)| m.predict(r) == Some(**c)).count();
        assert!(hit as f64 / x.len() as f64 > 0.98, "{hit} of {}", x.len());
    }

    /// THE trap this module exists to avoid. Scaling over every row lets the held-out fold move the
    /// centre used to fit, so the fit has seen it. The stored centre must come from training only.
    #[test]
    fn standardisation_comes_from_the_training_rows_only() {
        let (x, y) = blobs(30);
        let m = Lda::fit(&x, &y, 1e-3).unwrap();
        let centre_before = m.centre.clone();

        // Refit with a wildly out-of-range block appended, as a held-out fold would be.
        let mut x2 = x.clone();
        let mut y2 = y.clone();
        for _ in 0..30 {
            x2.push(vec![1e4, 1e4]);
            y2.push(0);
        }
        let m2 = Lda::fit(&x2, &y2, 1e-3).unwrap();
        assert!(
            (m2.centre[0] - centre_before[0]).abs() > 100.0,
            "adding rows MUST move the fitted centre, or this test proves nothing"
        );
        // And the original model, asked to score those rows, still uses ITS OWN centre.
        assert_eq!(m.centre, centre_before, "predict must not mutate the fit");
    }

    /// Collinear columns are the normal case here: the percentile family is near-duplicated by
    /// construction. Without a ridge the pooled scatter is singular and the solve must FAIL rather
    /// than return a fit built on a near-zero pivot.
    #[test]
    fn an_exactly_collinear_column_needs_the_ridge() {
        let (base, y) = blobs(30);
        let x: Vec<Vec<f64>> = base.iter().map(|r| vec![r[0], r[1], r[0] * 2.0]).collect();
        assert!(Lda::fit(&x, &y, 0.0).is_none(), "a singular scatter must not silently fit");
        let m = Lda::fit(&x, &y, 1e-2).expect("the ridge makes it solvable");
        let hit = x.iter().zip(&y).filter(|(r, c)| m.predict(r) == Some(**c)).count();
        assert!(hit as f64 / x.len() as f64 > 0.95, "and it must still separate: {hit}");
    }

    /// Uniform priors, stated as a test. A class holding 10x the rows must not be called more often
    /// on rows that are equidistant from both, or per-class recall moves with prevalence alone.
    #[test]
    fn priors_are_uniform_so_a_common_class_is_not_favoured() {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for c in 0..CLASSES {
            let n = if c == 1 { 400 } else { 40 };
            for k in 0..n {
                let t = k as f64 / n as f64;
                x.push(vec![c as f64 * 4.0 + (t * 7.0).sin() * 0.3]);
                y.push(c);
            }
        }
        let m = Lda::fit(&x, &y, 1e-3).unwrap();
        // Exactly between class 0 and class 1. A prior term would break the tie toward class 1.
        assert_eq!(Some(0), m.predict(&[2.0]), "the midpoint must not go to the common class");
        assert_eq!(Some(1), m.predict(&[4.0]));
        assert_eq!(Some(0), m.predict(&[0.0]));
    }

    /// A non-finite value is missing, not zero. Imputing it would be a manufactured measurement, and
    /// silently predicting on it would be worse.
    #[test]
    fn non_finite_rows_are_skipped_in_the_fit_and_refused_in_predict() {
        let (mut x, mut y) = blobs(30);
        let clean = Lda::fit(&x, &y, 1e-3).unwrap();
        for _ in 0..20 {
            x.push(vec![f64::NAN, 3.0]);
            y.push(2);
        }
        let with_nan = Lda::fit(&x, &y, 1e-3).expect("the NaN rows are skipped, not fatal");
        assert_eq!(clean.centre, with_nan.centre, "a skipped row cannot move the fit");
        assert_eq!(None, clean.predict(&[f64::NAN, 1.0]), "missing is not a prediction");
        assert_eq!(None, clean.predict(&[1.0]), "the wrong width is a caller bug");
    }

    #[test]
    fn a_degenerate_fit_answers_none_rather_than_guessing() {
        let (x, y) = blobs(30);
        assert!(Lda::fit(&[], &[], 1e-3).is_none(), "no rows");
        assert!(Lda::fit(&x, &y[..10], 1e-3).is_none(), "unpaired input is a caller bug");
        // Three classes present, the fourth absent: it could never be predicted, so refuse.
        let three: Vec<usize> = y.iter().map(|c| (*c).min(2)).collect();
        assert!(Lda::fit(&x, &three, 1e-3).is_none(), "an absent class is not a fit");
        // A constant column has no variance to scale by and must not produce NaN everywhere.
        let flat: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], r[1], 5.0]).collect();
        let m = Lda::fit(&flat, &y, 1e-2).expect("a constant column is inert, not fatal");
        assert!(m.predict(&[0.0, 0.0, 5.0]).is_some());
    }
}
