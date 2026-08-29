//! Is tanv1's emission simply too LOUD for the transition to argue with?
//!
//!   cargo run --release -p physio-algo --example emission_scale
//!
//! Measured, not assumed: v2's decode gains +0.06 to +0.12 kappa over its own argmax and tanv1's
//! gains about zero. Every explanation tried so far was about WHICH epochs each engine gets wrong.
//! There is a cruder one nobody has checked: the two emissions may not be on the same SCALE.
//!
//! Viterbi trades emission against `log(transition)`, whose stage-change charges are order 1-3 nats.
//! If tanv1's top-two margin is far wider than that, no transition can ever overturn an epoch, the
//! decode degenerates to the argmax, and the prior contributes nothing - which is precisely the
//! signature observed. v2's median margin is 0.56 nats, comfortably inside arguing distance.
//!
//! `tanv1_tune` swept gamma down to 0.25 and stopped. If the scale is the fault the optimum lies
//! below that floor, so this sweeps to 0.01 and reports, at every gamma, both the decoded kappa AND
//! the decoder gain over that same emission's argmax. A gain that climbs to v2's as gamma falls is
//! the whole answer.

mod common;

use common::lr::{design_row, fit as fit_lr, scores, standardise_cols};
use common::{
    cardiac_series, compare, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth,
    require_psg, stage_idx, Provenance,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, weights_of, SleepInput,
    Terms, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
const NW: usize = 12;
const LR0: f64 = 1.0;
const V2_ITERS: usize = 40_000;
const V2_TOL: f64 = 1e-11;
/// Down to a hundredth, because the previous sweep floored at 0.25 and chose it twice.
const GAMMAS: [f64; 10] = [0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0];

struct Night {
    tan: Vec<Vec<f64>>,
    terms: Terms,
    shipped: Vec<[f64; CLASSES]>,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    require_psg(set);
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

fn refit_v2(nights: &[&Night]) -> [f64; NW] {
    let y: Vec<usize> = nights.iter().flat_map(|n| n.truth.iter().flatten().copied()).collect();
    let cw = class_weights(&y);
    let mut w = weights_of(&Params::SHIPPED);
    let (mut last, mut lr, mut converged) = (f64::MAX, LR0, false);
    for _ in 0..V2_ITERS {
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
                for (j, gj) in g.iter_mut().enumerate() {
                    for (c, exc) in ex.iter().enumerate() {
                        let mut d = nt.terms.design[e][c][j];
                        if c == col_of(0) && (j == 8 || j == 9) && nt.terms.clamped[e] {
                            let card =
                                w[8] * nt.terms.design[e][c][8] + w[9] * nt.terms.design[e][c][9];
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
        if (0.0..V2_TOL).contains(&drop) {
            converged = true;
            break;
        }
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

fn tanv1_emissions(train: &[&Night], report: &[Night])
    -> (Vec<Vec<[f64; CLASSES]>>, [f64; CLASSES]) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for nt in train {
        for (e, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                x.push(nt.tan[e].clone());
                y.push(*t);
            }
        }
    }
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    let w = fit_lr(&dx, &y, WEIGHT_POWER);
    let cw = class_weights(&y);
    let ems = report
        .iter()
        .map(|nt| {
            (0..nt.truth.len())
                .map(|e| {
                    let z = scores(&w, &design_row(&nt.tan[e], &m, &sd, &[]));
                    std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
                })
                .collect()
        })
        .collect();
    (ems, cw)
}

/// Maximum-likelihood transition counted off TRAIN truth. Neither engine was tuned with it, which is
/// the point: the shipped one was hand-tuned on these very cohorts, jointly with v2's emission.
fn ml_transition(nights: &[&Night]) -> [[f64; CLASSES]; CLASSES] {
    let mut c = [[1.0f64; CLASSES]; CLASSES];
    for nt in nights {
        for k in 1..nt.truth.len() {
            if let (Some(a), Some(b)) = (nt.truth[k - 1], nt.truth[k]) {
                c[col_of(a)][col_of(b)] += 1.0;
            }
        }
    }
    for row in c.iter_mut() {
        let s: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    c
}

/// Decoded kappa under an arbitrary transition, so both engines can share one.
fn decoded_with(ems: &[Vec<[f64; CLASSES]>], nights: &[Night], t: &[[f64; CLASSES]; CLASSES])
    -> Vec<f64> {
    ems.iter()
        .zip(nights)
        .filter_map(|(em, nt)| {
            let path: Vec<usize> = decode_v2(em, t).iter().map(|s| stage_idx(*s)).collect();
            kappa_of(&path, &nt.truth)
        })
        .collect()
}

/// Per-class constants added to an emission before decoding, in [`STAGE_ORDER`] columns.
fn shifted(ems: &[Vec<[f64; CLASSES]>], add: [f64; CLASSES]) -> Vec<Vec<[f64; CLASSES]>> {
    ems.iter()
        .map(|em| em.iter().map(|r| std::array::from_fn(|c| r[c] + add[c])).collect())
        .collect()
}

/// Median top-two gap and median best-worst spread of an emission set, in nats.
fn scale_of(ems: &[Vec<[f64; CLASSES]>]) -> (f64, f64) {
    let (mut marg, mut span) = (Vec::new(), Vec::new());
    for em in ems {
        for r in em {
            let mut v = r.to_vec();
            v.sort_by(f64::total_cmp);
            marg.push(v[CLASSES - 1] - v[CLASSES - 2]);
            span.push(v[CLASSES - 1] - v[0]);
        }
    }
    (median(&mut marg), median(&mut span))
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

/// Per-night argmax kappa, decoded kappa and run count at one emission scale.
fn walk(ems: &[Vec<[f64; CLASSES]>], nights: &[Night], gamma: f64)
    -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut a, mut d, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for (em, nt) in ems.iter().zip(nights) {
        let scaled: Vec<[f64; CLASSES]> =
            em.iter().map(|row| std::array::from_fn(|c| gamma * row[c])).collect();
        let path: Vec<usize> =
            decode_v2(&scaled, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        if let (Some(ka), Some(kd)) =
            (kappa_of(&argmax(&scaled), &nt.truth), kappa_of(&path, &nt.truth))
        {
            a.push(ka);
            d.push(kd);
            let lab: Vec<usize> =
                (0..nt.truth.len()).filter(|k| nt.truth[*k].is_some()).map(|k| path[k]).collect();
            r.push((1 + (1..lab.len()).filter(|k| lab[*k] != lab[k - 1]).count()) as f64);
        }
    }
    (a, d, r)
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    println!("Is tanv1's emission too LOUD for the transition to argue with?\n");
    println!("The shipped transition's stage-change charges are order 1-3 nats. An emission whose");
    println!("top-two margin is far wider than that cannot be overturned, so the decode degenerates");
    println!("to the argmax and the prior contributes nothing.\n");

    for (held, hn) in &loaded {
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        let wv = refit_v2(&tr);
        let v2_em: Vec<Vec<[f64; CLASSES]>> = hn
            .iter()
            .map(|nt| (0..nt.truth.len()).map(|e| nt.terms.emission(e, &wv)).collect())
            .collect();
        let ship: Vec<Vec<[f64; CLASSES]>> = hn.iter().map(|n| n.shipped.clone()).collect();
        let (tan, cw) = tanv1_emissions(&tr, hn);

        let (vm, vs) = scale_of(&v2_em);
        let (sm, ss) = scale_of(&ship);
        let (tm, ts) = scale_of(&tan);
        println!("== {held} n={} ==", hn.len());
        println!("  emission scale, nats:   {:<16} margin {:>7.2}   best-worst span {:>7.2}",
                 "v2 SHIPPED", sm, ss);
        println!("                          {:<16} margin {:>7.2}   best-worst span {:>7.2}",
                 "v2 REFIT", vm, vs);
        println!("                          {:<16} margin {:>7.2}   best-worst span {:>7.2}   <-- {:.0}x v2",
                 "tanv1", tm, ts, tm / vm.max(1e-9));

        let (va, vd, _) = walk(&v2_em, hn, 1.0);
        println!("  v2 REFIT: argmax {:.3} -> decoded {:.3}, decoder gain {:+.4}",
                 median(&mut va.clone()), median(&mut vd.clone()),
                 compare(&va, &vd, Provenance::HeldOut).0);
        println!("  {:>6} {:>8} {:>8} {:>10} {:>7}   verdict vs V2 REFIT (decoded)",
                 "gamma", "argmax", "decoded", "dec gain", "runs");
        for g in GAMMAS {
            let (a, d, r) = walk(&tan, hn, g);
            let gain = compare(&a, &d, Provenance::HeldOut).0;
            println!("  {:>6} {:>8.3} {:>8.3} {:>+10.4} {:>7.0}   {}", g,
                     median(&mut a.clone()), median(&mut d.clone()), gain, median(&mut r.clone()),
                     compare(&vd, &d, Provenance::HeldOut).2);
        }

        // The fit is class-weighted, which deliberately REMOVES the base rate, so its output is not a
        // likelihood. Viterbi assumes one. These arms put the base rate back, two ways.
        let unweight: [f64; CLASSES] = std::array::from_fn(|c| -cw[stage_idx(STAGE_ORDER[c])].ln());
        let prior: [f64; CLASSES] =
            std::array::from_fn(|c| Params::SHIPPED.base_rate[c].ln());
        let both: [f64; CLASSES] = std::array::from_fn(|c| unweight[c] + prior[c]);
        println!("  putting the base rate back (gamma = 1, no refit):");
        println!("  {:<28} {:>8} {:>8} {:>10} {:>7}   verdict vs V2 REFIT", "arm", "argmax",
                 "decoded", "dec gain", "runs");
        for (label, add) in [("tanv1 raw", [0.0; CLASSES]), ("  - log(class weight)", unweight),
                             ("  + log(base_rate)", prior), ("  both", both)] {
            let (a, d, r) = walk(&shifted(&tan, add), hn, 1.0);
            println!("  {:<28} {:>8.3} {:>8.3} {:>+10.4} {:>7.0}   {}", label,
                     median(&mut a.clone()), median(&mut d.clone()),
                     compare(&a, &d, Provenance::HeldOut).0, median(&mut r.clone()),
                     compare(&vd, &d, Provenance::HeldOut).2);
        }

        // v2's transition was hand-tuned on these cohorts jointly with its emission. Give BOTH
        // engines one fitted on TRAIN truth only, so neither carries a memorised prior.
        let mlt = ml_transition(&tr);
        let (v2_ml, tan_ml) = (decoded_with(&v2_em, hn, &mlt), decoded_with(&tan, hn, &mlt));
        let (v2_sh, tan_sh) =
            (decoded_with(&v2_em, hn, &Params::SHIPPED.transition),
             decoded_with(&tan, hn, &Params::SHIPPED.transition));
        println!("  the transition nobody was tuned with (ML off TRAIN truth):");
        println!("    {:<26} v2 {:>6.3}   tanv1 {:>6.3}   tanv1 vs v2 {}", "SHIPPED transition",
                 median(&mut v2_sh.clone()), median(&mut tan_sh.clone()),
                 compare(&v2_sh, &tan_sh, Provenance::InSample("v2's transition")).2);
        println!("    {:<26} v2 {:>6.3}   tanv1 {:>6.3}   tanv1 vs v2 {}", "ML transition (honest)",
                 median(&mut v2_ml.clone()), median(&mut tan_ml.clone()),
                 compare(&v2_ml, &tan_ml, Provenance::HeldOut).2);
        println!("    what the shipped transition is worth: v2 {:+.4}, tanv1 {:+.4}",
                 compare(&v2_ml, &v2_sh, Provenance::HeldOut).0,
                 compare(&tan_ml, &tan_sh, Provenance::HeldOut).0);
        println!();
    }
    println!("`dec gain` is the decode minus its OWN argmax at that scale, so it isolates what the");
    println!("prior contributes. If it climbs toward v2's as gamma falls, the interface between the");
    println!("emission and the transition was the fault, not either component.");
}
