//! The like-for-like comparison: v2 held to the SAME leave-one-cohort-out discipline as tanv1.
//!
//!   cargo run --release -p physio-algo --example fair_fight
//!
//! Every "tanv1 is behind" figure so far compared an OUT-OF-SAMPLE estimate against an IN-SAMPLE one.
//! `Params::SHIPPED` was hand-tuned while watching all three of these cohorts, so 0.290 / 0.425 /
//! 0.332 are training-set numbers. tanv1 is fitted on two cohorts and reported on the third, which it
//! has never seen. Those are not comparable and the gap between them has been read as a model
//! difference all day.
//!
//! Here every arm is fitted on the two training cohorts and reported once on the held-out one:
//!   V2 SHIPPED   the in-sample reference, printed but never used as the comparison
//!   V2 REFIT     v2's OWN twelve weights refitted honestly, keeping every non-linearity, the cycle
//!                prior, the gate and the transition exactly as shipped
//!   TANV1        its own features and its own fitted weights
//!
//! The fair fight is TANV1 against V2 REFIT. V2 SHIPPED is there to show what the privilege was worth.

mod common;

use common::lr::{design_row, fit as fit_lr, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, weights_of, SleepInput,
    Terms, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
/// v2's twelve fittable weights.
const NW: usize = 12;
/// Starting step. Halved whenever the objective rises, so the fit cannot oscillate past its own
/// optimum and report a cap-hit as a failure to converge.
const LR0: f64 = 1.0;
const ITERS: usize = 40_000;
const TOL: f64 = 1e-11;
/// Shortest run a stage may hold. Emission-blind, so every arm gets it.
const MIN_DWELL: [usize; CLASSES] = [1, 1, 6, 6];

struct Night {
    tan: Vec<Vec<f64>>,
    terms: Terms,
    shipped: Vec<[f64; CLASSES]>,
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
        out.push(Night {
            tan: (0..em.len()).map(|e| f[e].values().to_vec()).collect(),
            terms,
            shipped: em[..].to_vec(),
            truth: (0..em.len())
                .map(|k| {
                    raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
                })
                .collect(),
        });
    }
    out
}

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

/// Refit v2's own twelve weights by weighted multinomial descent through [`Terms::emission`], so
/// every non-linearity, the cycle prior, the gate and the clamp stay exactly as shipped.
fn refit_v2(nights: &[&Night]) -> [f64; NW] {
    let y: Vec<usize> = nights.iter().flat_map(|n| n.truth.iter().flatten().copied()).collect();
    let cw = class_weights(&y);
    let mut w = weights_of(&Params::SHIPPED);
    let mut last = f64::MAX;
    let mut lr = LR0;
    let mut converged = false;
    for _ in 0..ITERS {
        let mut g = [0.0f64; NW];
        let (mut nll, mut n) = (0.0f64, 0.0f64);
        for nt in nights {
            for (e, want) in nt.truth.iter().enumerate() {
                let Some(want) = want else { continue };
                let em = nt.terms.emission(e, &w);
                let mx = em.iter().cloned().fold(f64::MIN, f64::max);
                let ex: Vec<f64> = em.iter().map(|v| (v - mx).exp()).collect();
                let sum: f64 = ex.iter().sum();
                let col = col_of(*want);
                nll -= cw[*want] * (ex[col] / sum).max(1e-300).ln();
                n += 1.0;
                // d(loss)/d(w_j) = sum_c (p_c - 1{c=y}) * d(em_c)/d(w_j), and d(em_c)/d(w_j) is the
                // design cell, except where the awake clamp has zeroed the cardiac pair's gradient.
                for (j, gj) in g.iter_mut().enumerate() {
                    for (c, exc) in ex.iter().enumerate() {
                        let mut d = nt.terms.design[e][c][j];
                        if c == col_of(0) && (j == 8 || j == 9) && nt.terms.clamped[e] {
                            let card = w[8] * nt.terms.design[e][c][8] + w[9] * nt.terms.design[e][c][9];
                            if card > 0.0 {
                                d = 0.0;
                            }
                        }
                        *gj += cw[*want] * (exc / sum - if c == col { 1.0 } else { 0.0 }) * d;
                    }
                }
            }
        }
        let nll = nll / n;
        let drop = last - nll;
        if (0.0..TOL).contains(&drop) {
            converged = true;
            break;
        }
        // A rise means the step overshot; back it off rather than letting it ring.
        if drop < 0.0 {
            lr *= 0.5;
        }
        last = nll;
        for (j, gj) in g.iter().enumerate() {
            w[j] -= lr * gj / n;
        }
    }
    assert!(converged, "the v2 refit hit its cap; the arms are not comparable");
    w
}

fn argmax(em: &[[f64; CLASSES]]) -> Vec<usize> {
    em.iter()
        .map(|row| {
            let mut best = (0usize, f64::NEG_INFINITY);
            for (c, v) in row.iter().enumerate() {
                if *v > best.1 {
                    best = (c, *v);
                }
            }
            stage_idx(STAGE_ORDER[best.0])
        })
        .collect()
}

fn dwell_floor(path: &[usize]) -> Vec<usize> {
    let mut out = path.to_vec();
    let mut i = 0;
    while i < out.len() {
        let mut j = i;
        while j + 1 < out.len() && out[j + 1] == out[i] {
            j += 1;
        }
        if j + 1 - i < MIN_DWELL[out[i]] && i > 0 {
            let fill = out[i - 1];
            out[i..=j].fill(fill);
            i = i.saturating_sub(1);
            continue;
        }
        i = j + 1;
    }
    out
}

fn kappa_of(path: &[usize], truth: &[Option<usize>]) -> Option<f64> {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate() {
        if let Some(want) = want {
            p.push(path[k]);
            t.push(*want);
        }
    }
    (t.len() >= MIN_EPOCHS).then(|| kappa4(&confusion4(&p, &t)))
}

/// One arm's per-night kappa: at the emission's own argmax, and after the shipped decode plus floor.
fn walk(ems: &[Vec<[f64; CLASSES]>], nights: &[Night]) -> (Vec<f64>, Vec<f64>) {
    let (mut a, mut d) = (Vec::new(), Vec::new());
    for (em, nt) in ems.iter().zip(nights) {
        let path: Vec<usize> =
            decode_v2(em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        if let (Some(k1), Some(k2)) =
            (kappa_of(&argmax(em), &nt.truth), kappa_of(&dwell_floor(&path), &nt.truth))
        {
            a.push(k1);
            d.push(k2);
        }
    }
    (a, d)
}

fn verdict(base: &[f64], arm: &[f64]) -> String {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let Some((mean, bar)) = paired_bar(&d) else { return "-".into() };
    let tag = if mean.abs() <= bar {
        "matches".to_string()
    } else {
        format!("{} ({:.2}x)", if mean > 0.0 { "TANV1 AHEAD" } else { "tanv1 behind" }, mean.abs() / bar)
    };
    format!("{mean:>+8.4} {bar:>7.4}  {tag}")
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    println!("Like-for-like: v2 held to the SAME leave-one-cohort-out discipline as tanv1.");
    println!("V2 SHIPPED is IN-SAMPLE - its weights were tuned watching all three of these cohorts.");
    println!("V2 REFIT is the same recipe, same non-linearities, same prior and transition, with its");
    println!("twelve weights refitted on the two TRAIN cohorts only. That is the honest opponent.\n");

    for (held, hn) in &loaded {
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();

        // v2, refitted honestly, through its own decomposition.
        let wv = refit_v2(&tr);
        let em_refit: Vec<Vec<[f64; CLASSES]>> = hn
            .iter()
            .map(|nt| (0..nt.truth.len()).map(|e| nt.terms.emission(e, &wv)).collect())
            .collect();

        // tanv1, its own features and weights.
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for nt in &tr {
            for (e, t) in nt.truth.iter().enumerate() {
                if let Some(t) = t {
                    x.push(nt.tan[e].clone());
                    y.push(*t);
                }
            }
        }
        let (m, sd) = standardise_cols(&x);
        let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
        let wt = fit_lr(&dx, &y, WEIGHT_POWER);
        let em_tan: Vec<Vec<[f64; CLASSES]>> = hn
            .iter()
            .map(|nt| {
                (0..nt.truth.len())
                    .map(|e| {
                        let z = scores(&wt, &design_row(&nt.tan[e], &m, &sd, &[]));
                        std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
                    })
                    .collect()
            })
            .collect();

        let em_ship: Vec<Vec<[f64; CLASSES]>> = hn.iter().map(|nt| nt.shipped.clone()).collect();
        let (a_ship, d_ship) = walk(&em_ship, hn);
        let (a_refit, d_refit) = walk(&em_refit, hn);
        let (a_tan, d_tan) = walk(&em_tan, hn);

        println!("== {held} n={} ==", hn.len());
        println!("  {:<22} {:>8} {:>9}", "arm", "argmax", "decoded");
        println!("  {:<22} {:>8.3} {:>9.3}   IN-SAMPLE, not a valid opponent",
                 "V2 SHIPPED", median(&mut a_ship.clone()), median(&mut d_ship.clone()));
        println!("  {:<22} {:>8.3} {:>9.3}   honest, the real opponent",
                 "V2 REFIT (LOCO)", median(&mut a_refit.clone()), median(&mut d_refit.clone()));
        println!("  {:<22} {:>8.3} {:>9.3}   honest",
                 "TANV1 (LOCO)", median(&mut a_tan.clone()), median(&mut d_tan.clone()));
        println!("  what the privilege was worth: decoded {:+.4} shipped over its own honest refit",
                 median(&mut d_ship.clone()) - median(&mut d_refit.clone()));
        println!("  THE FAIR FIGHT, tanv1 vs v2-refit, paired per night:");
        println!("    argmax   {}", verdict(&a_refit, &a_tan));
        println!("    decoded  {}", verdict(&d_refit, &d_tan));
        println!("  for reference, tanv1 against the IN-SAMPLE shipped arm: {}",
                 verdict(&d_ship, &d_tan));
        println!();
    }
    println!("A model is only behind another if both were measured the same way. V2 SHIPPED has seen");
    println!("every night it is scored on; V2 REFIT and TANV1 have seen neither.");
}
