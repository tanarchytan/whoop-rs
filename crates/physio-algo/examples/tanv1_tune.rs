//! Tune tanv1 against the HONEST opponent: v2 refitted under the same leave-one-cohort-out rule.
//!
//!   cargo run --release -p physio-algo --example tanv1_tune
//!
//! Everything before `fair_fight` tuned tanv1 against `Params::SHIPPED`, whose weights were selected
//! watching all three of these cohorts. That inflated every gap by up to 0.080 and led to the verdict
//! that the lever space was exhausted, reached by judging levers against a gap of 0.05 to 0.14. The
//! honest gap is 0.015 to 0.018, so levers dismissed as too small are worth re-testing.
//!
//! Four knobs, none of which needs a refit, so they sweep on a cached emission:
//!   gamma       emission scale. A fitted score has no natural scale against log(transition).
//!   beta        transition^beta. How hard the prior argues.
//!   dwell deep  minimum run length for DEEP. The one resolvable win of the six-step audit.
//!   dwell rem   the same for REM, which carried most of that win.
//!
//! Coordinate ascent, selected on an inner leave-one-out over the TRAIN cohorts, reported once on the
//! held-out one against V2 REFIT. Every comparison goes through `common::compare`, which will not
//! format a line without stating whether its baseline saw the nights it is scored on.

mod common;

use common::lr::{design_row, fit as fit_lr, scores, standardise_cols};
use common::{
    cardiac_series, compare, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth,
    stage_idx, Provenance,
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

const GAMMAS: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
const BETAS: [f64; 6] = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5];
const DWELLS: [usize; 6] = [1, 2, 4, 6, 9, 12];
const PASSES: usize = 4;

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

/// v2's twelve weights refitted through its own decomposition, so every non-linearity, the cycle
/// prior, the gate and the clamp stay exactly as shipped. This is the honest opponent.
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

/// The four knobs. `dwell` is indexed by our class order [wake, light, deep, rem].
#[derive(Clone, Copy, PartialEq)]
struct Knobs {
    gamma: f64,
    beta: f64,
    dwell: [usize; CLASSES],
}

impl Knobs {
    fn start() -> Self {
        Knobs { gamma: 1.0, beta: 1.0, dwell: [1, 1, 1, 1] }
    }
    fn line(&self) -> String {
        format!("g={:<4} b={:<4} deep>={} rem>={}", self.gamma, self.beta, self.dwell[2],
                self.dwell[3])
    }
}

fn tempered(beta: f64) -> [[f64; CLASSES]; CLASSES] {
    Params::SHIPPED.transition.map(|row| row.map(|v| v.powf(beta)))
}

/// Collapse runs shorter than that stage's floor into the stage the run came from.
fn dwell_floor(path: &[usize], min: &[usize; CLASSES]) -> Vec<usize> {
    let mut out = path.to_vec();
    let mut i = 0;
    while i < out.len() {
        let mut j = i;
        while j + 1 < out.len() && out[j + 1] == out[i] {
            j += 1;
        }
        if j + 1 - i < min[out[i]] && i > 0 {
            let fill = out[i - 1];
            out[i..=j].fill(fill);
            i = i.saturating_sub(1);
            continue;
        }
        i = j + 1;
    }
    out
}

/// Per-night kappa of one cached emission set under one knob setting.
fn score(ems: &[Vec<[f64; CLASSES]>], truth: &[Vec<Option<usize>>], k: Knobs) -> Vec<f64> {
    let t = tempered(k.beta);
    let mut out = Vec::new();
    for (em, tr) in ems.iter().zip(truth) {
        let scaled: Vec<[f64; CLASSES]> =
            em.iter().map(|r| std::array::from_fn(|c| k.gamma * r[c])).collect();
        let path: Vec<usize> = decode_v2(&scaled, &t).iter().map(|s| stage_idx(*s)).collect();
        let path = dwell_floor(&path, &k.dwell);
        let (mut p, mut g) = (Vec::new(), Vec::new());
        for (i, want) in tr.iter().enumerate() {
            if let Some(want) = want {
                p.push(path[i]);
                g.push(*want);
            }
        }
        if g.len() >= MIN_EPOCHS {
            out.push(kappa4(&confusion4(&p, &g)));
        }
    }
    out
}

/// One fold's cached material: tanv1's emissions, the honest v2's, and the truth.
struct Fold {
    tan: Vec<Vec<[f64; CLASSES]>>,
    v2: Vec<Vec<[f64; CLASSES]>>,
    truth: Vec<Vec<Option<usize>>>,
}

fn build_fold(train: &[&Night], report: &[Night]) -> Fold {
    let wv = refit_v2(train);
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
    let wt = fit_lr(&dx, &y, WEIGHT_POWER);
    Fold {
        tan: report
            .iter()
            .map(|nt| {
                (0..nt.truth.len())
                    .map(|e| {
                        let z = scores(&wt, &design_row(&nt.tan[e], &m, &sd, &[]));
                        std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
                    })
                    .collect()
            })
            .collect(),
        v2: report
            .iter()
            .map(|nt| (0..nt.truth.len()).map(|e| nt.terms.emission(e, &wv)).collect())
            .collect(),
        truth: report.iter().map(|n| n.truth.clone()).collect(),
    }
}

/// Mean paired gain of tanv1 over the HONEST v2 across the inner folds. The only thing a step reads.
fn inner_gain(inner: &[Fold], k: Knobs) -> f64 {
    inner
        .iter()
        .map(|f| {
            let base = score(&f.v2, &f.truth, Knobs::start());
            compare(&base, &score(&f.tan, &f.truth, k), Provenance::HeldOut).0
        })
        .sum::<f64>()
        / inner.len() as f64
}

fn ascend(inner: &[Fold]) -> Knobs {
    let mut best = Knobs::start();
    let mut obj = inner_gain(inner, best);
    for _ in 0..PASSES {
        let before = obj;
        let try_set = |f: &dyn Fn(&mut Knobs), best: &mut Knobs, obj: &mut f64| {
            let mut cand = *best;
            f(&mut cand);
            let o = inner_gain(inner, cand);
            if o > *obj {
                *obj = o;
                *best = cand;
            }
        };
        for g in GAMMAS {
            try_set(&|k: &mut Knobs| k.gamma = g, &mut best, &mut obj);
        }
        for b in BETAS {
            try_set(&|k: &mut Knobs| k.beta = b, &mut best, &mut obj);
        }
        for d in DWELLS {
            try_set(&|k: &mut Knobs| k.dwell[2] = d, &mut best, &mut obj);
            try_set(&|k: &mut Knobs| k.dwell[3] = d, &mut best, &mut obj);
        }
        if obj <= before + 1e-6 {
            break;
        }
    }
    best
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    println!("Tuning tanv1 against V2 REFIT - v2's own twelve weights refitted under the SAME");
    println!("leave-one-cohort-out rule. Knobs chosen on an inner leave-one-out over the TRAIN");
    println!("cohorts and applied once. Every line goes through the provenance guard.\n");

    for (held, hn) in &loaded {
        let names: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();

        let inner: Vec<Fold> = names
            .iter()
            .map(|c| {
                let sub: Vec<&Night> = loaded
                    .iter()
                    .filter(|(n, _)| n != c && n != held)
                    .flat_map(|(_, n)| n.iter())
                    .collect();
                let iv = &loaded.iter().find(|(n, _)| n == c).expect("inner cohort").1;
                build_fold(&sub, iv)
            })
            .collect();

        let k = ascend(&inner);
        let outer = build_fold(&tr, hn);
        let honest = score(&outer.v2, &outer.truth, Knobs::start());
        let plain = score(&outer.tan, &outer.truth, Knobs::start());
        let tuned = score(&outer.tan, &outer.truth, k);
        let shipped: Vec<Vec<[f64; CLASSES]>> = hn.iter().map(|n| n.shipped.clone()).collect();
        let in_sample = score(&shipped, &outer.truth, Knobs::start());

        println!("== {held} n={} ==   chosen inside: {}", hn.len(), k.line());
        println!("  {:<26} {:>7}   {:>8} {:>7}  verdict", "arm", "kappa", "paired d", "bar");
        println!("  {:<26} {:>7.3}   the honest baseline", "V2 REFIT (LOCO)",
                 median(&mut honest.clone()));
        println!("  {:<26} {:>7.3}   {}", "tanv1 untuned", median(&mut plain.clone()),
                 compare(&honest, &plain, Provenance::HeldOut).2);
        println!("  {:<26} {:>7.3}   {}", "tanv1 TUNED", median(&mut tuned.clone()),
                 compare(&honest, &tuned, Provenance::HeldOut).2);
        println!("  {:<26} {:>7.3}   {}", "  vs the in-sample v2", median(&mut in_sample.clone()),
                 compare(&in_sample, &tuned, Provenance::InSample("all 20+ params")).2);
        println!();
    }
    println!("The knobs need no refit, so they were swept on a cached emission. If this closes the");
    println!("gap, the earlier verdict that the lever space was exhausted was an artefact of judging");
    println!("these same levers against an inflated baseline.");
}
