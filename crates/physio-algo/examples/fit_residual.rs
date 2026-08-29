//! Start AT the shipped recipe and learn a correction, so matching it is guaranteed by construction.
//!
//!   cargo run --release -p physio-algo --example fit_residual
//!
//! Every fit so far started from all-zero weights and hoped the optimiser would rediscover the
//! shipped answer. It never does: handed that answer as four columns it still walked away, because
//! the likelihood optimum is not the shipped argmax. Hoping is the wrong mechanism.
//!
//! So the emission is the shipped one PLUS a correction:
//!
//!   em[c] = em_v2[c] + sum_j theta[c][j] * x[j]
//!
//! `theta` starts at zero, which IS the shipped recipe exactly - `fit_hybrid`'s positive control
//! pins that those emissions reproduce the shipped staging night for night. L2 pulls `theta` back to
//! zero, so a column only moves the answer where the data pays for the move. The fit cannot start
//! behind, and whatever it does is a measured correction rather than a rediscovery.
//!
//! Leave one cohort out, so a reported cohort never appears in its own fit.

mod common;

use common::lr::{design_row, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const N_BASE: usize = Features::N;
/// The shipped transforms a linear column can hold: the deep-gate hinge, the deadzoned cardiac
/// pair, the centred rotation rank, the respiration z, and the stillness-clamp indicator.
const N_NL: usize = 6;
/// Correction strengths to sweep. Weak leaves the correction free; strong drives it to zero, which
/// IS the shipped recipe, so the strong end of this sweep must land on `matches`.
const L2S: [f64; 2] = [0.3, 1.0];
/// A column a cohort never carries standardises to 0.0, which contributes zero to EVERY class score,
/// so it abstains rather than fabricating. Dropping it can only change the fit, never the scoring —
/// which is why removing `resp_z` on sleep-accel, at 0% beat coverage, moves nothing.
const RESP_COL: usize = N_BASE + 4;
const ITERS: usize = 6_000;
const TOL: f64 = 1e-10;
const LR: f64 = 0.5;
/// Inverse-frequency exponent on the class weight, matching the sibling harnesses.
const WEIGHT_POWER: f64 = 0.5;

struct Night {
    row: Vec<Vec<f64>>,
    /// The shipped emission per epoch, in [`STAGE_ORDER`] columns. The correction is added to this.
    offset: Vec<[f64; CLASSES]>,
    base: Vec<usize>,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let raw = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if raw.is_empty() || accel.is_empty() {
            continue;
        }
        let n = n_meta.max(raw.keys().max().copied().unwrap_or(0) + 1);
        let (hr, rr) = (read_hr(dir), read_rr(dir));
        let f = extract(&accel, w0, w1, &cardiac_series(w0, n, EPOCH, &hr, &rr));
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        assert!(f.len() >= em.len(), "{}: fewer feature rows than emissions", dir.display());
        let deep = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 2).expect("deep");
        let awake = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 0).expect("wake");
        let row = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = f[e].values().to_vec();
                v.extend_from_slice(&[
                    -d[deep][3],
                    d[awake][8],
                    d[awake][9],
                    d[awake][10],
                    d[deep][11],
                    if terms.clamped[e] { 1.0 } else { 0.0 },
                ]);
                v
            })
            .collect();
        let base =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { row, offset: em[..].to_vec(), base, truth });
    }
    out
}

/// Our class index -> the emission's column.
fn col_of(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

fn class_weights(y: &[usize]) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for c in y {
        n[*c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(WEIGHT_POWER)
        } else {
            0.0
        };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] as f64 * w[c]).sum::<f64>() / y.len() as f64;
    for v in w.iter_mut() {
        *v /= mass;
    }
    w
}

/// Multinomial logistic regression on top of a fixed per-class OFFSET.
///
/// `theta` starts at zero, which leaves the offset untouched, so iteration zero is the shipped
/// recipe. L2 pulls back to zero rather than to an arbitrary origin.
fn fit_residual(
    x: &[Vec<f64>],
    off: &[[f64; CLASSES]],
    y: &[usize],
    l2: f64,
) -> (Vec<Vec<f64>>, bool) {
    let p = x[0].len();
    let cw = class_weights(y);
    let mut th = vec![vec![0.0f64; p]; CLASSES];
    let mut last = f64::MAX;
    let mut converged = false;
    for _ in 0..ITERS {
        let mut g = vec![vec![0.0f64; p]; CLASSES];
        let mut nll = 0.0f64;
        for ((row, o), &lab) in x.iter().zip(off).zip(y) {
            let mut z = [0.0f64; CLASSES];
            for c in 0..CLASSES {
                // Our class c reads the offset's own column, then adds its correction.
                z[c] = o[col_of(c)] + th[c].iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
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
        let nll = nll / x.len() as f64;
        let drop = last - nll;
        if (0.0..TOL).contains(&drop) {
            converged = true;
            break;
        }
        last = nll;
        // Decay written as a FACTOR and floored at zero. Subtracting `LR * l2 * th` is the same
        // thing only while `LR * l2 < 1`; above that it overshoots past zero and oscillates, which
        // at l2 = 3.0 drove the correction to a kappa of -0.087 and tripped the convergence guard.
        let scale = LR / x.len() as f64;
        let decay = (1.0 - LR * l2).max(0.0);
        for c in 0..CLASSES {
            for j in 0..p {
                th[c][j] = decay * th[c][j] - scale * g[c][j];
            }
        }
    }
    (th, converged)
}

/// Kappa per night for the shipped recipe and for the corrected emission, on the same nights.
fn score(
    nights: &[Night],
    th: &[Vec<f64>],
    m: &[f64],
    sd: &[f64],
    drop: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let (mut bk, mut fk) = (Vec::new(), Vec::new());
    for nt in nights {
        let em: Vec<[f64; CLASSES]> = nt
            .row
            .iter()
            .zip(&nt.offset)
            .map(|(full, o)| {
                let d = design_row(full, m, sd, drop);
                let mut out = *o;
                for c in 0..CLASSES {
                    out[col_of(c)] += th[c].iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
                }
                out
            })
            .collect();
        let path: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let (mut bp, mut fp, mut t) = (Vec::new(), Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            bp.push(nt.base[k]);
            fp.push(path[k]);
            t.push(*want);
        }
        if t.len() >= MIN_EPOCHS {
            bk.push(kappa4(&confusion4(&bp, &t)));
            fk.push(kappa4(&confusion4(&fp, &t)));
        }
    }
    (bk, fk)
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 2 {
        println!("need at least two cohorts under the fixture root");
        return;
    }

    println!("The emission is the SHIPPED one plus a learned correction. theta starts at zero,");
    println!("which is the shipped recipe exactly, so the fit cannot begin behind it. L2 pulls the");
    println!("correction back to zero, so a column only moves the answer where the data pays.\n");

    // The guarantee, asserted rather than claimed: a zero correction must decode to the shipped
    // staging on every night, or the offset is not what this harness thinks it is.
    for (name, nights) in &loaded {
        let zero = vec![vec![0.0f64; N_BASE + N_NL + 1]; CLASSES];
        let (m, sd) = (vec![0.0; N_BASE + N_NL], vec![1.0; N_BASE + N_NL]);
        let (bk, fk) = score(nights, &zero, &m, &sd, &[]);
        assert_eq!(bk, fk, "{name}: a ZERO correction must be the shipped recipe exactly");
    }
    println!("  GUARANTEE holds: a zero correction decodes to the shipped staging on every night\n");

    // How much of each column a cohort actually carries. A column at 0% is not "missing data",
    // it is a column that cohort cannot answer, and imputing the train mean answers it anyway.
    println!("  beat coverage per cohort (resp_z present):");
    for (name, nights) in &loaded {
        let (mut have, mut all) = (0usize, 0usize);
        for nt in nights {
            for r in &nt.row {
                all += 1;
                if r[RESP_COL].is_finite() {
                    have += 1;
                }
            }
        }
        println!("    {name:<14} {:>6.1}%", 100.0 * have as f64 / all.max(1) as f64);
    }
    println!();

    println!("  {:<8} {:<7} {:<18} {:>7} {:>7}   {:>10} {:>9} {:>4}   verdict",
             "L2", "resp_z", "held-out cohort", "shipped", "corrected", "paired d", "bar +/-", "n");
    for l2 in L2S {
        for (drop_resp, tag) in [(false, "kept"), (true, "DROPPED")] {
        for (held, _) in &loaded {
            let train: Vec<&Night> =
                loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
            let rows: Vec<(Vec<f64>, [f64; CLASSES], usize)> = train
                .iter()
                .flat_map(|nt| {
                    nt.row.iter().zip(&nt.offset).zip(&nt.truth).filter_map(|((r, o), t)| {
                        t.map(|t| (r.clone(), *o, t))
                    })
                })
                .collect();
            let x: Vec<Vec<f64>> = rows.iter().map(|(r, _, _)| r.clone()).collect();
            let (m, sd) = standardise_cols(&x);
            let off: Vec<[f64; CLASSES]> = rows.iter().map(|(_, o, _)| *o).collect();
            let y: Vec<usize> = rows.iter().map(|(_, _, t)| *t).collect();
            // Dropping the column is the decisive test: if the imputed constant is what hurts a
            // cohort that never carries it, removing the column must move that cohort and not the
            // two that do carry it.
            let drop: Vec<usize> = if drop_resp { vec![RESP_COL] } else { Vec::new() };
            let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &drop)).collect();
            let (th, conv) = fit_residual(&dx, &off, &y, l2);

            let nights = &loaded.iter().find(|(c, _)| c == held).expect("held").1;
            let (bk, fk) = score(nights, &th, &m, &sd, &drop);
            let d: Vec<f64> = bk.iter().zip(&fk).map(|(a, b)| b - a).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let v = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("{} ({:.2}x)", if mean > 0.0 { "BEATS SHIPPED" } else { "worse" },
                        mean.abs() / bar)
            } else {
                "matches".to_string()
            };
            println!("  {:<8.3} {:<7} {:<18} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>4}   {v}{}",
                     l2, tag, format!("{held} n={}", nights.len()),
                     median(&mut bk.clone()), median(&mut fk.clone()), d.len(),
                     if conv { "" } else { "  [DID NOT CONVERGE]" });
        }
        }
    }
    println!("\nA large L2 drives the correction to zero, which is the shipped recipe, so the");
    println!("bottom of this sweep must converge on `matches`. Anything better than that is earned.");
}
