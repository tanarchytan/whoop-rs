//! A decoder built FOR tanv1: the transition optimised against decoded kappa, not against truth.
//!
//!   cargo run --release -p physio-algo --example tanv1_decoder
//!
//! v2's emission and its transition were tuned together. tanv1 has only ever borrowed half of that
//! pair, and the two repairs tried so far both aimed at the wrong target:
//!   - v2's transition, which was shaped around v2's emission scale and margins;
//!   - a maximum-likelihood transition counted off truth, which is the TRUE stage dynamics.
//!
//! Neither is what a decoder wants. The optimal decoder transition is whatever compensates for THIS
//! emission's error structure, and the only way to find it is to optimise the thing we score.
//!
//! So: coordinate ascent on all sixteen log-transition entries plus an emission scale, maximising
//! DECODED kappa on an inner leave-one-out over the training cohorts, reported once on the held-out
//! one. Three starts, best inner wins, so a bad basin cannot decide the answer.
//!
//! The control that can take the win away: the same search run on V2's OWN emission. If co-designing
//! a transition lifts v2 too, it is a decoder finding and not a tanv1 one.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
/// Multiplicative steps a single transition entry may take per coordinate visit.
const STEPS: [f64; 6] = [0.125, 0.25, 0.5, 2.0, 4.0, 8.0];
/// Emission scales the search may take. The fitted score has no natural scale against log(transition).
const GAMMAS: [f64; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
const PASSES: usize = 4;
/// Floor under any transition entry, so a coordinate cannot walk a path into being unreachable.
const FLOOR: f64 = 1e-6;

struct Night {
    tan: Vec<Vec<f64>>,
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
        let em = emissions_v2(&prepare_v2(&input, &Params::SHIPPED), &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        out.push(Night {
            tan: (0..em.len()).map(|e| f[e].values().to_vec()).collect(),
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

/// One cohort's emissions under an arm, cached so the search never refits or rescores features.
struct Arm {
    em: Vec<Vec<[f64; CLASSES]>>,
    truth: Vec<Vec<Option<usize>>>,
}

fn arm_shipped(nights: &[Night]) -> Arm {
    Arm {
        em: nights.iter().map(|n| n.shipped.clone()).collect(),
        truth: nights.iter().map(|n| n.truth.clone()).collect(),
    }
}

fn arm_tanv1(nights: &[Night], w: &[Vec<f64>], m: &[f64], sd: &[f64]) -> Arm {
    Arm {
        em: nights
            .iter()
            .map(|nt| {
                (0..nt.truth.len())
                    .map(|e| {
                        let z = scores(w, &design_row(&nt.tan[e], m, sd, &[]));
                        std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
                    })
                    .collect()
            })
            .collect(),
        truth: nights.iter().map(|n| n.truth.clone()).collect(),
    }
}

fn train_readout(nights: &[&Night]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for nt in nights {
        for (e, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                x.push(nt.tan[e].clone());
                y.push(*t);
            }
        }
    }
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    (fit(&dx, &y, WEIGHT_POWER), m, sd)
}

/// A decoder: sixteen transition entries and one emission scale.
#[derive(Clone)]
struct Dec {
    t: [[f64; CLASSES]; CLASSES],
    gamma: f64,
}

/// Per-night decoded kappa of one arm under one decoder.
fn score(a: &Arm, d: &Dec) -> Vec<f64> {
    let mut out = Vec::new();
    for (em, truth) in a.em.iter().zip(&a.truth) {
        let scaled: Vec<[f64; CLASSES]> =
            em.iter().map(|r| std::array::from_fn(|c| d.gamma * r[c])).collect();
        let path: Vec<usize> =
            decode_v2(&scaled, &d.t).iter().map(|s| stage_idx(*s)).collect();
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (k, want) in truth.iter().enumerate() {
            if let Some(want) = want {
                p.push(path[k]);
                t.push(*want);
            }
        }
        if t.len() >= MIN_EPOCHS {
            out.push(kappa4(&confusion4(&p, &t)));
        }
    }
    out
}

/// Mean per-cohort median kappa across the inner folds. The only quantity a search step may read.
fn inner_obj(arms: &[Arm], d: &Dec) -> f64 {
    arms.iter().map(|a| median(&mut score(a, d))).sum::<f64>() / arms.len() as f64
}

/// Coordinate ascent over the sixteen entries and the scale, on the inner folds only.
fn co_design(arms: &[Arm], start: Dec) -> Dec {
    let mut best = start;
    let mut obj = inner_obj(arms, &best);
    for _ in 0..PASSES {
        let before = obj;
        for g in GAMMAS {
            let mut cand = best.clone();
            cand.gamma = g;
            let o = inner_obj(arms, &cand);
            if o > obj {
                obj = o;
                best = cand;
            }
        }
        for i in 0..CLASSES {
            for j in 0..CLASSES {
                for s in STEPS {
                    let mut cand = best.clone();
                    cand.t[i][j] = (cand.t[i][j] * s).clamp(FLOOR, 1.0);
                    let o = inner_obj(arms, &cand);
                    if o > obj {
                        obj = o;
                        best = cand;
                    }
                }
            }
        }
        if obj <= before + 1e-6 {
            break;
        }
    }
    best
}

/// Maximum-likelihood transition off truth, one of the three starts.
fn ml_transition(nights: &[&Night]) -> [[f64; CLASSES]; CLASSES] {
    let ord = |c: usize| (0..CLASSES).find(|k| stage_idx(STAGE_ORDER[*k]) == c).expect("order");
    let mut c = [[1.0f64; CLASSES]; CLASSES];
    for nt in nights {
        for k in 1..nt.truth.len() {
            if let (Some(a), Some(b)) = (nt.truth[k - 1], nt.truth[k]) {
                c[ord(a)][ord(b)] += 1.0;
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

fn verdict(base: &[f64], arm: &[f64]) -> (f64, String) {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let Some((mean, bar)) = paired_bar(&d) else { return (f64::NAN, "-".into()) };
    let tag = if mean.abs() <= bar {
        "matches".to_string()
    } else {
        format!("{} ({:.2}x)", if mean > 0.0 { "AHEAD" } else { "behind" }, mean.abs() / bar)
    };
    (mean, format!("{mean:>+8.4} {bar:>7.4} {tag}"))
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    println!("A decoder built FOR tanv1: the transition optimised against DECODED KAPPA on an inner");
    println!("leave-one-out over the training cohorts, three starts, reported once on the held-out one.");
    println!("The control is the same search on V2's emission - a lift there is a decoder finding.\n");

    for (held, hn) in &loaded {
        let names: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        let (w, m, sd) = train_readout(&tr);

        // Inner arms: each training cohort scored under a readout fitted WITHOUT it.
        let (mut inner_tan, mut inner_v2) = (Vec::new(), Vec::new());
        for c in &names {
            let sub: Vec<&Night> = loaded
                .iter()
                .filter(|(n, _)| n != c && n != held)
                .flat_map(|(_, n)| n.iter())
                .collect();
            let iv = &loaded.iter().find(|(n, _)| n == c).expect("inner cohort").1;
            let (iw, im, isd) = train_readout(&sub);
            inner_tan.push(arm_tanv1(iv, &iw, &im, &isd));
            inner_v2.push(arm_shipped(iv));
        }

        let starts = [
            ("v2", Params::SHIPPED.transition),
            ("ml", ml_transition(&tr)),
            ("uniform", [[0.25; CLASSES]; CLASSES]),
        ];
        let outer_tan = arm_tanv1(hn, &w, &m, &sd);
        let outer_v2 = arm_shipped(hn);
        let base = score(&outer_v2, &Dec { t: Params::SHIPPED.transition, gamma: 1.0 });

        println!("== {held} n={} ==  shipped decoded {:.3}", hn.len(), median(&mut base.clone()));
        println!("  {:<24} {:>7} {:>8} {:>8}   {:>8}  verdict vs shipped",
                 "arm", "kappa", "inner", "transfer", "paired d");

        for (label, inner, outer) in
            [("tanv1", &inner_tan, &outer_tan), ("CONTROL v2 emission", &inner_v2, &outer_v2)]
        {
            let ref_dec = Dec { t: Params::SHIPPED.transition, gamma: 1.0 };
            let ref_inner = inner_obj(inner, &ref_dec);
            let mut best: Option<(&str, Dec, f64)> = None;
            for (sname, t) in &starts {
                let d = co_design(inner, Dec { t: *t, gamma: 1.0 });
                let o = inner_obj(inner, &d);
                if best.as_ref().is_none_or(|(_, _, b)| o > *b) {
                    best = Some((sname, d, o));
                }
            }
            let (sname, d, o) = best.expect("a start");
            let arm = score(outer, &d);
            let (mm, v) = verdict(&base, &arm);
            let gain = o - ref_inner;
            let plain = score(outer, &ref_dec);
            let (pm, _) = verdict(&base, &plain);
            let transfer = if gain.abs() > 1e-9 { (mm - pm) / gain } else { f64::NAN };
            println!("  {:<24} {:>7.3} {:>+8.4} {transfer:>8.2}   {v}   [from {sname}, g={}]",
                     label, median(&mut arm.clone()), gain, d.gamma);
            println!("  {:<24} {:>7.3}   before co-design, same emission, shipped transition",
                     "", median(&mut plain.clone()));
        }
        println!();
    }
    println!("`inner` is what the search bought on the folds it could see; `transfer` is how much of");
    println!("that survived on the cohort it could not. A transfer near zero means the sixteen free");
    println!("entries fitted the training cohorts and nothing else.");
}
