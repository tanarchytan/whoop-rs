//! Adopt the shipped emission's non-linearities into the fitted one, and measure each way of doing it.
//!
//!   cargo run --release -p physio-algo --example fit_hybrid
//!
//! The fitted emission has no non-linearity at all; the shipped one has four, and they are the part
//! a linear model cannot invent. `emission_terms` already computes every one of them, so they can be
//! handed to the fit as COLUMNS rather than reimplemented:
//!
//!   the deadzoned cardiac pair, the deep-gate HINGE on the flatness rank, the centred rotation
//!   RANK, the respiration z, the stillness-clamp indicator, and the fixed per-class priors.
//!
//! Four arms, all fitted identically, differing only in which columns exist:
//!   TANV1     - the 28 measured columns, no shipped structure.
//!   +TRANSFRM - plus the shipped transforms above. Can the fit USE the non-linearities?
//!   +EMISSION - plus the shipped emission's four values. Can it use the ANSWER?
//!   EMIT ONLY - the four values and nothing else. Standardisation is per-column and invertible,
//!               so this arm CAN express the shipped recipe exactly - and the fit does not choose
//!               it. That gap is the objective mismatch in its purest form, not a floor.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const FIT: &str = "dreamt";
const HELD: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
/// Widths of the three blocks a row can carry, in the order [`row_of`] lays them out.
const N_TANV1: usize = Features::N;
const N_TRANSFORM: usize = 9;

/// Which blocks an arm keeps.
#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    tanv1: bool,
    transform: bool,
    emission: bool,
}

const ARMS: [Arm; 4] = [
    Arm { name: "TANV1", tanv1: true, transform: false, emission: false },
    Arm { name: "+TRANSFORM", tanv1: true, transform: true, emission: false },
    Arm { name: "+EMISSION", tanv1: true, transform: false, emission: true },
    Arm { name: "EMISSION ONLY", tanv1: false, transform: false, emission: true },
];

struct Night {
    /// tanv1 columns, then the shipped transforms, then the shipped emission.
    row: Vec<Vec<f64>>,
    em: Vec<[f64; CLASSES]>,
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
        let terms = emission_terms(&prep, &Params::SHIPPED);
        let em = physio_algo::sleep::emissions_v2(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS || f.len() < em.len() {
            continue;
        }
        let (deep, rem, awake) = (
            stage_col(physio_algo::sleep::SleepStage::Deep),
            stage_col(physio_algo::sleep::SleepStage::Rem),
            stage_col(physio_algo::sleep::SleepStage::Wake),
        );
        let row = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = f[e].values().to_vec();
                // The shipped transforms, read straight off the decomposition so they cannot drift
                // from what the recipe actually computes.
                v.extend_from_slice(&[
                    d[deep][0],                                  // hr_var z
                    d[deep][1],                                  // hr z
                    d[deep][2],                                  // move_frac z
                    -d[deep][3],                                 // the deep-gate HINGE
                    d[awake][8],                                 // deadzoned hr_var z
                    d[awake][9],                                 // deadzoned hr z
                    d[awake][10],                                // centred rotation RANK
                    d[deep][11],                                 // respiration z
                    if terms.clamped[e] { 1.0 } else { 0.0 },    // the stillness clamp indicator
                ]);
                let _ = rem;
                v.extend_from_slice(&em[e]);
                v
            })
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { row, em, truth });
    }
    out
}

fn stage_col(s: physio_algo::sleep::SleepStage) -> usize {
    STAGE_ORDER.iter().position(|x| *x == s).expect("stage in STAGE_ORDER")
}

/// The columns an arm keeps, out of a full row.
fn row_of(full: &[f64], a: Arm) -> Vec<f64> {
    let mut v = Vec::new();
    if a.tanv1 {
        v.extend_from_slice(&full[..N_TANV1]);
    }
    if a.transform {
        v.extend_from_slice(&full[N_TANV1..N_TANV1 + N_TRANSFORM]);
    }
    if a.emission {
        v.extend_from_slice(&full[N_TANV1 + N_TRANSFORM..]);
    }
    v
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

/// Per-night kappa under one arm's fitted weights.
fn score(nights: &[Night], a: Arm, w: &[Vec<f64>], m: &[f64], sd: &[f64]) -> Vec<f64> {
    let to_order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    let mut ks = Vec::new();
    for nt in nights {
        let em: Vec<[f64; CLASSES]> = nt
            .row
            .iter()
            .map(|full| {
                let z = scores(w, &design_row(&row_of(full, a), m, sd, &[]));
                let mx = z.iter().cloned().fold(f64::MIN, f64::max);
                let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
                std::array::from_fn(|c| z[to_order[c]] - lse)
            })
            .collect();
        let path: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(path[k]);
            t.push(*want);
        }
        if p.len() >= MIN_EPOCHS {
            ks.push(kappa4(&confusion4(&p, &t)));
        }
    }
    ks
}

/// The shipped recipe on the same rows, decoded the same way.
fn shipped(nights: &[Night]) -> Vec<f64> {
    let mut ks = Vec::new();
    for nt in nights {
        let path: Vec<usize> = decode_v2(&nt.em, &Params::SHIPPED.transition)
            .iter()
            .map(|s| stage_idx(*s))
            .collect();
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(path[k]);
            t.push(*want);
        }
        if p.len() >= MIN_EPOCHS {
            ks.push(kappa4(&confusion4(&p, &t)));
        }
    }
    ks
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn main() {
    let train = load(FIT);
    if train.is_empty() {
        println!("no {FIT} nights - check the fixture root");
        return;
    }
    let cohorts: Vec<(String, Vec<Night>)> = std::iter::once((format!("{FIT} (FIT)"), load(FIT)))
        .chain(HELD.iter().map(|s| (format!("{s} (HELD)"), load(s))))
        .collect();

    println!("Every arm: same optimiser, same decoder, same shipped transition. Only the COLUMNS");
    println!("differ. EMISSION ONLY can reproduce the shipped recipe exactly, so it is the floor.\n");
    println!("  {:<16} {:>5} {:<22} {:>8} {:>8}   {:>10} {:>9} {:>5}   verdict",
             "arm", "cols", "cohort", "shipped", "arm", "paired d", "bar +/-", "n");

    for a in ARMS {
        let rows: Vec<(Vec<f64>, usize)> = train
            .iter()
            .flat_map(|nt| {
                nt.row.iter().zip(&nt.truth).filter_map(|(r, t)| t.map(|t| (row_of(r, a), t)))
            })
            .collect();
        let x: Vec<Vec<f64>> = rows.iter().map(|(r, _)| r.clone()).collect();
        let y: Vec<usize> = rows.iter().map(|(_, t)| *t).collect();
        let (m, sd) = standardise_cols(&x);
        let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
        let _ = class_weights(&y);
        let w = fit(&dx, &y, WEIGHT_POWER);
        let ncol = x[0].len();
        for (name, nights) in &cohorts {
            let base = shipped(nights);
            let arm = score(nights, a, &w, &m, &sd);
            let d: Vec<f64> = base.iter().zip(&arm).map(|(p, q)| q - p).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let v = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("{} ({:.2}x)", if mean > 0.0 { "BEATS SHIPPED" } else { "worse" },
                        mean.abs() / bar)
            } else {
                "inside the bar".to_string()
            };
            println!("  {:<16} {:>5} {:<22} {:>8.3} {:>8.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {v}",
                     if name.starts_with(FIT) { a.name } else { "" },
                     if name.starts_with(FIT) { ncol.to_string() } else { String::new() },
                     name, median(&base), median(&arm), d.len());
        }
        println!();
    }
    println!("If +TRANSFORM beats TANV1, the non-linearities were the missing part. EMISSION ONLY");
    println!("could express the shipped recipe and does not reach it: handed the answer as four");
    println!("columns, a likelihood fit still walks away from it.");
}
