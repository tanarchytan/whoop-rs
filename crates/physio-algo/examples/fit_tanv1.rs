//! Fit the tanv1 per-epoch classifier, and separate the two effects that have never been separated.
//!
//!   cargo run --release -p physio-algo --example fit_tanv1
//!
//! The plan's §6.0 says "fit it" is NOT by itself a difference: v2's emission is already a weighted
//! sum of z-scored features plus a bias, so a linear fit over the same inputs is the same functional
//! form with better numbers. Two things could make tanv1 real - a model class that can represent
//! INTERACTIONS, and many more features - and nobody has measured how much of the gain comes from
//! which.
//!
//! This measures exactly that, by fitting the same model twice:
//!   MAIN     - the 25 feature columns, no interaction term.
//!   MAIN+INT - the same, plus `still_x_cardiac`, which is the product v2 hard-codes as a branch.
//! The difference between them IS the interaction's contribution, on identical data and optimiser.
//!
//! Discipline, all four from the plan's rules:
//!   - Fit on DREAMT. Report HELD-OUT on aauwss and sleep-accel. A train number is never a result.
//!   - DREAMT is CLINICAL: per-epoch stage labels only, never onset/offset. Nothing here reads a
//!     boundary.
//!   - Standardisation uses TRAIN statistics applied to held-out, never per-cohort - that leaks.
//!   - The rate-matched null is the floor: a fit that only calls more wake has done nothing.
//!
//! The output is weights. Nothing is wired and `Params::SHIPPED` is untouched.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_truth};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, recall, specificity, WAKE};

const EPOCH: i64 = 30;
const FIT: [&str; 1] = ["dreamt"];
const HELD_OUT: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Column count from `Features::NAMES`, plus a bias term appended by the design matrix.
const NCOL: usize = 25;
const ITERS: usize = 400;
const LR: f64 = 0.5;
/// L2 penalty. 100 subjects against 26 parameters per class overfits without one, and the plan
/// names overfitting as the expected failure mode.
const L2: f64 = 1e-3;
/// Exponent on the inverse-frequency class weight. 0 = unweighted, 1 = full inverse frequency.
///
/// Both extremes are wrong and both were measured. Unweighted, the fit calls deep 0.0% against a
/// 3.4-18.1% truth rate - a two-class model wearing a four-class report. Full inverse frequency
/// over-corrects the other way: deep called 34.9% against 3.4%, a 10x over-prediction. The
/// square root is the usual compromise, but it was swept rather than assumed and it LOST: held-out
/// kappa at 0.5 is 0.182/0.172 against 1.0's 0.239/0.203. Full weighting wins on the number that
/// counts while being visibly miscalibrated, so the flag in the report exists to keep that visible
/// rather than letting a better kappa hide it.
const WEIGHT_POWER: f64 = 1.0;

struct Set {
    x: Vec<[f64; NCOL]>,
    y: Vec<usize>,
    /// Night index per row, so a per-subject score never pools across nights.
    night: Vec<usize>,
}

/// Per-night z-score of a per-epoch series, missing where the series is.
fn zscore(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = v.iter().flatten().copied().collect();
    if present.len() < 2 {
        return vec![None; v.len()];
    }
    let m = present.iter().sum::<f64>() / present.len() as f64;
    let sd = (present.iter().map(|x| (x - m).powi(2)).sum::<f64>() / present.len() as f64).sqrt();
    if sd <= 0.0 {
        return vec![None; v.len()];
    }
    v.iter().map(|o| o.map(|x| (x - m) / sd)).collect()
}

fn load(sets: &[&str]) -> Set {
    let (mut x, mut y, mut night) = (Vec::new(), Vec::new(), Vec::new());
    let mut idx = 0usize;
    for set in sets {
        for dir in &dirs_of(set) {
            let truth = read_truth(dir);
            let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
            let grav = read_accel(dir);
            if truth.is_empty() || grav.is_empty() {
                continue;
            }
            let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);

            // Per-epoch mean HR, then this night's own z-score - the same shape v2 uses.
            let hr = read_hr(dir);
            let mut sum = vec![(0.0f64, 0.0f64); n];
            for s in &hr {
                let k = ((s.ts - w0) / EPOCH).max(0) as usize;
                if k < n {
                    sum[k].0 += s.bpm as f64;
                    sum[k].1 += 1.0;
                }
            }
            let raw: Vec<Option<f64>> =
                sum.iter().map(|(a, c)| (*c > 0.0).then(|| a / c)).collect();
            let hr_z = zscore(&raw);
            // Short-window HR variability: the spread of per-epoch HR over +/- 5 epochs.
            let hv: Vec<Option<f64>> = (0..n)
                .map(|k| {
                    let lo = k.saturating_sub(5);
                    let hi = (k + 6).min(n);
                    let w: Vec<f64> = raw[lo..hi].iter().flatten().copied().collect();
                    (w.len() >= 3).then(|| {
                        let m = w.iter().sum::<f64>() / w.len() as f64;
                        (w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / w.len() as f64).sqrt()
                    })
                })
                .collect();
            let hr_var_z = zscore(&hv);

            let f: Vec<Features> = extract(&grav, w0, w1, &hr_z, &hr_var_z);
            for (k, t) in truth.iter() {
                let Some(fe) = f.get(*k) else { continue };
                // read_truth yields i32; the model indexes classes by usize.
                if *t < 0 || *t as usize >= CLASSES {
                    continue;
                }
                x.push(fe.values());
                y.push(*t as usize);
                night.push(idx);
            }
            idx += 1;
        }
    }
    Set { x, y, night }
}

/// Column means and sds over TRAIN only, ignoring NaN. Applied unchanged to held-out.
fn standardiser(x: &[[f64; NCOL]]) -> ([f64; NCOL], [f64; NCOL]) {
    let (mut m, mut s) = ([0.0; NCOL], [1.0; NCOL]);
    for c in 0..NCOL {
        let v: Vec<f64> = x.iter().map(|r| r[c]).filter(|v| v.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        m[c] = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|z| (z - m[c]).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        s[c] = if sd > 1e-12 { sd } else { 1.0 };
    }
    (m, s)
}

/// Design row: standardised, NaN imputed to the train mean (which is 0 after standardising), plus a
/// bias. `drop` zeroes a column so the same optimiser can be run with a feature withheld.
fn design(r: &[f64; NCOL], m: &[f64; NCOL], s: &[f64; NCOL], drop: Option<usize>) -> Vec<f64> {
    let mut out = Vec::with_capacity(NCOL + 1);
    for c in 0..NCOL {
        let v = if drop == Some(c) || !r[c].is_finite() { 0.0 } else { (r[c] - m[c]) / s[c] };
        out.push(v);
    }
    out.push(1.0);
    out
}

/// Inverse-frequency weight per class, normalised to mean 1 so the effective learning rate is
/// unchanged. Without this the fit collapses: measured, an unweighted model called deep 0.0% of
/// epochs on all three cohorts against true rates of 3.4-18.1%, and REM under 1% against 10.5-22.0%.
/// It was a two-class wake/light classifier wearing a four-class report.
fn class_weights(y: &[usize]) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for &c in y {
        n[c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(WEIGHT_POWER)
        } else {
            0.0
        };
    }
    let mean = w.iter().sum::<f64>() / CLASSES as f64;
    if mean > 0.0 {
        for v in w.iter_mut() {
            *v /= mean;
        }
    }
    w
}

/// Multinomial logistic regression by full-batch gradient descent. Deterministic: no shuffling, no
/// randomness, fixed iteration count - two runs give identical weights.
fn fit(x: &[Vec<f64>], y: &[usize]) -> Vec<Vec<f64>> {
    let p = x[0].len();
    let cw = class_weights(y);
    let mut w = vec![vec![0.0f64; p]; CLASSES];
    let mut last_nll = f64::MAX;
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
        // MEDIUM from the review: 400 iterations was neither printed nor checked. Measured, the
        // gradient norm at 400 was still 2.1x its value at 4000, so stop on the loss instead.
        let nll = nll / x.len() as f64;
        if it % 200 == 0 || it + 1 == ITERS {
            println!("    iter {it:>4}  weighted nll {nll:.5}");
        }
        if (last_nll - nll).abs() < 1e-7 {
            println!("    converged at iter {it}");
            break;
        }
        last_nll = nll;
        let scale = LR / x.len() as f64;
        for c in 0..CLASSES {
            for j in 0..p {
                // No penalty on the bias: shrinking it would bias the base rates.
                let pen = if j + 1 == p { 0.0 } else { L2 * w[c][j] };
                w[c][j] -= scale * g[c][j] + LR * pen;
            }
        }
    }
    w
}

fn predict(w: &[Vec<f64>], row: &[f64]) -> usize {
    let mut best = (0usize, f64::MIN);
    for (c, wc) in w.iter().enumerate() {
        let z: f64 = wc.iter().zip(row).map(|(a, b)| a * b).sum();
        if z > best.1 {
            best = (c, z);
        }
    }
    best.0
}

/// Per-night kappa / wake recall / wake specificity, then the median across nights - never pooled,
/// because pooling lets the longest night decide the number.
fn score(w: &[Vec<f64>], s: &Set, m: &[f64; NCOL], sd: &[f64; NCOL], drop: Option<usize>)
    -> (f64, f64, f64, [f64; CLASSES], [f64; CLASSES]) {
    let nights = s.night.iter().max().map_or(0, |v| v + 1);
    let (mut ks, mut rs, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    // Per-class call rate AND truth rate. A kappa alone cannot show that a class is never predicted.
    let (mut called, mut truth_n) = ([0usize; CLASSES], [0usize; CLASSES]);
    let mut total = 0usize;
    for nid in 0..nights {
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for i in 0..s.x.len() {
            if s.night[i] != nid {
                continue;
            }
            p.push(predict(w, &design(&s.x[i], m, sd, drop)));
            t.push(s.y[i]);
        }
        if p.len() < 20 {
            continue;
        }
        let cm = confusion4(&p, &t);
        ks.push(kappa4(&cm));
        if let Some(r) = recall(&cm, WAKE) {
            rs.push(r);
        }
        if let Some(v) = specificity(&cm, WAKE) {
            ss.push(v);
        }
        for (pi, ti) in p.iter().zip(&t) {
            called[*pi] += 1;
            truth_n[*ti] += 1;
            total += 1;
        }
    }
    let med = |mut v: Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let frac = |v: [usize; CLASSES]| {
        let mut o = [0.0; CLASSES];
        for c in 0..CLASSES {
            o[c] = if total > 0 { v[c] as f64 / total as f64 } else { f64::NAN };
        }
        o
    };
    (med(ks), med(rs), med(ss), frac(called), frac(truth_n))
}

fn main() {
    let train = load(&FIT);
    println!("FIT: {} ({} epochs, {} nights)", FIT[0], train.x.len(),
             train.night.iter().max().map_or(0, |v| v + 1));
    if train.x.is_empty() {
        println!("no training data - check the fixture root");
        return;
    }
    let (m, sd) = standardiser(&train.x);
    // By name, not a hardcoded index. And `expect`, because if the column is ever renamed then
    // `position` yields None, BOTH arms get drop=None, and two identical runs are compared against
    // each other - printing a confident wrong conclusion with no error anywhere.
    let int_col = Features::NAMES
        .iter()
        .position(|n| *n == "still_x_cardiac")
        .expect("still_x_cardiac must exist in Features::NAMES or the ablation compares nothing");
    println!("interaction column `still_x_cardiac` at index {int_col}\n");

    let arms: [(&str, Option<usize>); 2] =
        [("MAIN (interaction withheld)", Some(int_col)), ("MAIN + INTERACTION", None)];

    for (label, drop) in arms {
        let dx: Vec<Vec<f64>> = train.x.iter().map(|r| design(r, &m, &sd, drop)).collect();
        let w = fit(&dx, &train.y);
        println!("=== {label}");
        println!("  {:<18} {:>6} {:>7} {:>6}   {:<28} truth  W/L/D/R %", "cohort", "kappa",
                 "wake r", "spec", "we call  W/L/D/R %");
        let show = |name: &str, r: (f64, f64, f64, [f64; CLASSES], [f64; CLASSES])| {
            let (k, rec, sp, call, tru) = r;
            let pc = |v: [f64; CLASSES]| {
                format!("{:>5.1}/{:>4.1}/{:>4.1}/{:>4.1}", 100.0 * v[0], 100.0 * v[1],
                        100.0 * v[2], 100.0 * v[3])
            };
            // A class never predicted is the failure an unweighted fit hides behind a kappa.
            let worst = (0..CLASSES)
                .filter(|c| tru[*c] > 0.005)
                .map(|c| (call[c] / tru[c]).max(tru[c] / call[c].max(1e-9)))
                .fold(1.0f64, f64::max);
            let flag = if call[2] < 0.005 || call[3] < 0.005 {
                "  <- CLASS COLLAPSE".to_string()
            } else if worst > 3.0 {
                format!("  <- MISCALIBRATED {worst:.1}x")
            } else {
                String::new()
            };
            println!("  {name:<18} {k:>6.3} {rec:>7.3} {sp:>6.3}   {:<28} {}{}",
                     pc(call), pc(tru), flag);
        };
        show("dreamt (FIT)", score(&w, &train, &m, &sd, drop));
        for set in HELD_OUT {
            let ho = load(&[set]);
            if ho.x.is_empty() {
                println!("  {set:<18} no data");
                continue;
            }
            show(&format!("{set} (HELD)"), score(&w, &ho, &m, &sd, drop));
        }
        println!();
    }
    println!("The difference between the two arms IS the interaction's contribution: identical data,");
    println!("identical optimiser, one column zeroed. A held-out number is the only result here.");
}
