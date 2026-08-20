//! Multinomial logistic regression, shared by the harnesses that fit one.
//!
//! Column count is [`Features::N`], read from the source of truth so adding a column can never
//! leave a design matrix silently narrow.

#![allow(dead_code)]

use physio_algo::sleep::features::Features;
use physio_algo::stats;

pub const CLASSES: usize = 4;
pub const NCOL: usize = Features::N;
/// Hard cap only - the loop exits on [`TOL`], and reaching it means the fit did NOT converge.
pub const ITERS: usize = 20_000;
/// Per-iteration loss change below which the fit is converged.
pub const TOL: f64 = 1e-9;
pub const LR: f64 = 0.5;
/// L2 penalty on every weight except the bias.
pub const L2: f64 = 1e-3;

/// Column means and sds over TRAIN only, ignoring NaN. Applied unchanged to held-out.
pub fn standardiser(x: &[[f64; NCOL]]) -> ([f64; NCOL], [f64; NCOL]) {
    let (m, s) = col_stats(x, NCOL);
    (std::array::from_fn(|c| m[c]), std::array::from_fn(|c| s[c]))
}

/// Fixed-width [`design_row`]: same standardise-impute-bias rule, for the array harnesses.
pub fn design(r: &[f64; NCOL], m: &[f64; NCOL], s: &[f64; NCOL], drop: &[usize]) -> Vec<f64> {
    design_row(r, m, s, drop)
}

/// Inverse-frequency weight per class, normalised so the TOTAL weighted mass equals the sample
/// count at every power. The divisor is the SAMPLE-weighted mean: an arithmetic mean across four
/// classes is dominated by the rare ones and rescales mass differently at each power.
pub fn class_weights(y: &[usize], power: f64) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for &c in y {
        n[c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(power)
        } else {
            0.0
        };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] as f64 * w[c]).sum::<f64>() / y.len() as f64;
    if mass > 0.0 {
        for v in w.iter_mut() {
            *v /= mass;
        }
    }
    w
}

/// Multinomial logistic regression by full-batch gradient descent. Deterministic: no shuffling, no
/// randomness, a deterministic stopping rule - two runs give identical weights.
/// The LAST column of every row must be the bias [`design_row`] appends: L2 exempts the intercept.
pub fn fit(x: &[Vec<f64>], y: &[usize], power: f64) -> Vec<Vec<f64>> {
    let p = x[0].len();
    debug_assert!(
        x.iter().all(|r| r.len() == p && r[p - 1] == 1.0),
        "fit exempts the last column from L2 as the intercept"
    );
    let cw = class_weights(y, power);
    let mut w = vec![vec![0.0f64; p]; CLASSES];
    let mut last_obj = f64::MAX;
    let mut converged = false;
    for it in 0..ITERS {
        let mut g = vec![vec![0.0f64; p]; CLASSES];
        let mut nll = 0.0f64;
        for (row, &lab) in x.iter().zip(y) {
            let mut z = [0.0f64; CLASSES];
            for c in 0..CLASSES {
                z[c] = w[c].iter().zip(row).map(|(a, b)| a * b).sum();
            }
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let ex: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
            let sum: f64 = ex.iter().sum();
            nll -= cw[lab] * (ex[lab] / sum).max(1e-300).ln();
            for c in 0..CLASSES {
                let err = cw[lab] * (ex[c] / sum - if c == lab { 1.0 } else { 0.0 });
                for (gi, xi) in g[c].iter_mut().zip(row) {
                    *gi += err * xi;
                }
            }
        }
        // The step below descends nll + (L2/2)||w||^2, so convergence is judged on that objective.
        let w_sq: f64 = w.iter().flat_map(|wc| wc[..p - 1].iter()).map(|v| v * v).sum();
        let obj = nll / x.len() as f64 + 0.5 * L2 * w_sq;
        if it % 2000 == 0 {
            println!("    iter {it:>5}  weighted objective {obj:.8}");
        }
        // Signed, not `abs()`. An increase is divergence, not convergence, and `abs()` would report
        // a step that went the WRONG way by less than the tolerance as a converged fit.
        let drop = last_obj - obj;
        if drop < 0.0 {
            println!("    iter {it:>5}  objective ROSE by {:.3e} - the step is too large", -drop);
        }
        if (0.0..TOL).contains(&drop) {
            println!("    converged at iter {it}, weighted objective {obj:.8}");
            converged = true;
            break;
        }
        last_obj = obj;
        let scale = LR / x.len() as f64;
        for c in 0..CLASSES {
            for j in 0..p {
                // No penalty on the bias: shrinking it would bias the base rates.
                let pen = if j + 1 == p { 0.0 } else { L2 * w[c][j] };
                w[c][j] -= scale * g[c][j] + LR * pen;
            }
        }
    }
    // An unconverged fit is not a result. Two arms stopped at a shared iteration cap sit at unequal
    // distances from their own optima, and the whole conclusion here is the small delta between them.
    assert!(converged, "the fit hit the {ITERS}-iteration cap without converging - raise ITERS or \
                        lower LR; the arm deltas are not comparable until both arms converge");
    w
}

/// Argmax of [`scores`]. A tie, and a row whose scores are all non-finite, take the lowest class.
pub fn predict(w: &[Vec<f64>], row: &[f64]) -> usize {
    let mut best = (0usize, f64::MIN);
    for (c, &z) in scores(w, row).iter().enumerate() {
        if z > best.1 {
            best = (c, z);
        }
    }
    best.0
}

/// Class scores for one row.
pub fn scores(w: &[Vec<f64>], row: &[f64]) -> [f64; CLASSES] {
    let mut z = [0.0f64; CLASSES];
    for (c, wc) in w.iter().enumerate() {
        z[c] = wc.iter().zip(row).map(|(a, b)| a * b).sum();
    }
    z
}

/// Column means and sds over TRAIN only, for a design of arbitrary width.
pub fn standardise_cols(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let ncol = x.first().map_or(0, |r| r.len());
    assert!(
        ncol > 0 && x.iter().all(|r| r.len() == ncol),
        "standardise_cols needs a non-empty rectangular design; got {} rows, first {ncol} wide",
        x.len()
    );
    col_stats(x, ncol)
}

/// Mean and population sd of each column's finite values. An sd at or below 1e-12 becomes 1.0, so
/// a constant column standardises to zero; an all-NaN column keeps mean 0 and sd 1.
fn col_stats(x: &[impl AsRef<[f64]>], ncol: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut m, mut s) = (vec![0.0; ncol], vec![1.0; ncol]);
    for c in 0..ncol {
        let v: Vec<f64> = x.iter().map(|r| r.as_ref()[c]).filter(|v| v.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        m[c] = stats::mean(&v);
        let sd = stats::population_sd(&v);
        s[c] = if sd > 1e-12 { sd } else { 1.0 };
    }
    (m, s)
}

/// Design row of arbitrary width: standardised, NaN imputed to the train mean, plus a bias.
/// `drop` zeroes columns so one optimiser can be run with features withheld.
pub fn design_row(r: &[f64], m: &[f64], s: &[f64], drop: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(r.len() + 1);
    for (c, v) in r.iter().enumerate() {
        out.push(if drop.contains(&c) || !v.is_finite() { 0.0 } else { (v - m[c]) / s[c] });
    }
    out.push(1.0);
    out
}
