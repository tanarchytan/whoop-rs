//! Refit the shipped emission's OWN twelve weights, keeping every non-linearity it has.
//!
//!   cargo run --release -p physio-algo --example fit_v2_weights
//!
//! Replacing the emission with a 116-parameter regression was measured and it loses held out. This
//! is the smaller move that was never tried: the shipped emission is LINEAR IN ITS OWN WEIGHTS given
//! its transforms, so the deadzone, the stillness clamp, the deep-gate hinge, the rank transforms,
//! the cycle prior and the pinned light class all stay exactly as they are, and only the twelve
//! numbers move. `emission_terms` is pinned by a test to reproduce the shipped emission bit for bit
//! at the shipped weights, so this optimises the real recipe rather than a lookalike.
//!
//! It starts AT the shipped weights, so it cannot begin worse than v2, and L2 pulls toward them
//! rather than toward zero - a weight only moves if the data pays for the move.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, params::Params, prepare_v2, weights_of, SleepInput, Terms,
    WEIGHT_NAMES,
};

const FIT: &str = "dreamt";
const HELD: [&str; 2] = ["aauwss", "sleep-accel"];
const NW: usize = 12;
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const ITERS: usize = 4_000;
const STEP: f64 = 0.02;
/// Pull toward the SHIPPED weights, not toward zero. The shipped values are evidence, so a weight
/// should only move where the data pays for the move.
const L2_TO_SHIPPED: f64 = 0.01;

/// What the search optimises. A fit maximises LIKELIHOOD; the product reports a DECODED KAPPA, and
/// they are not the same function of these weights.
#[derive(Clone, Copy, PartialEq)]
enum Objective {
    Likelihood,
    DecodedKappa,
}

struct Night {
    terms: Terms,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let raw = read_truth(dir);
        let Some((w0, w1, _)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if raw.is_empty() || accel.is_empty() {
            continue;
        }
        let input =
            SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if terms.design.len() < MIN_EPOCHS {
            continue;
        }
        let truth = (0..terms.design.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { terms, truth });
    }
    out
}

/// Our class index is [wake, light, deep, rem]; the emission's columns are STAGE_ORDER.
fn col_of(class: usize) -> usize {
    use physio_algo::sleep::STAGE_ORDER;
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

/// Weighted multinomial log-loss over the labelled epochs, and its gradient in the twelve weights.
///
/// The gradient is numeric. Twelve parameters against a closed form that has a clamp in it is not
/// worth hand-differentiating, and a wrong derivative is a silent wrong answer.
fn loss(nights: &[Night], w: &[f64; NW], cw: &[f64; CLASSES]) -> f64 {
    let mut total = 0.0;
    let mut n = 0.0;
    for nt in nights {
        for (e, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            let em = nt.terms.emission(e, w);
            let mx = em.iter().cloned().fold(f64::MIN, f64::max);
            let lse = mx + em.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            total += cw[*want] * (lse - em[col_of(*want)]);
            n += cw[*want];
        }
    }
    if n > 0.0 {
        total / n
    } else {
        f64::NAN
    }
}

fn class_weights(nights: &[Night]) -> [f64; CLASSES] {
    let mut cnt = [0.0f64; CLASSES];
    for nt in nights {
        for t in nt.truth.iter().flatten() {
            cnt[*t] += 1.0;
        }
    }
    let tot: f64 = cnt.iter().sum();
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if cnt[c] > 0.0 { (tot / (CLASSES as f64 * cnt[c])).sqrt() } else { 0.0 };
    }
    let mass: f64 = (0..CLASSES).map(|c| cnt[c] * w[c]).sum::<f64>() / tot;
    for v in w.iter_mut() {
        *v /= mass;
    }
    w
}

/// Per-night kappa under one weight vector.
fn score(nights: &[Night], w: &[f64; NW]) -> Vec<f64> {
    let mut ks = Vec::new();
    for nt in nights {
        let em: Vec<[f64; CLASSES]> =
            (0..nt.terms.design.len()).map(|e| nt.terms.emission(e, w)).collect();
        let path: Vec<usize> = decode_v2(&em, &Params::SHIPPED.transition)
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
    let shipped = weights_of(&Params::SHIPPED);
    let cw = class_weights(&train);
    println!("FIT on {FIT}: {} nights, class weights {:.2?}\n", train.len(), cw);

    // Coordinate descent. Twelve parameters, a clamp in the objective, and a start that is already
    // good - and one of the two objectives is a decoded kappa with no derivative at all.
    let search = |objective: Objective, step0: f64| -> [f64; NW] {
        let obj = |w: &[f64; NW]| -> f64 {
            let pen: f64 =
                (0..NW).map(|j| (w[j] - shipped[j]).powi(2)).sum::<f64>() * L2_TO_SHIPPED;
            let base = match objective {
                Objective::Likelihood => loss(&train, w, &cw),
                // Negated so both objectives are minimised.
                Objective::DecodedKappa => -median(&score(&train, w)),
            };
            base + pen
        };
        let mut w = shipped;
        let mut best = obj(&w);
        let mut step = step0;
        for _ in 0..ITERS {
            let mut moved = false;
            for j in 0..NW {
                for dir in [1.0, -1.0] {
                    let mut cand = w;
                    cand[j] += dir * step;
                    let v = obj(&cand);
                    if v < best - 1e-12 {
                        w = cand;
                        best = v;
                        moved = true;
                    }
                }
            }
            if !moved {
                step *= 0.5;
                if step < 1e-4 {
                    break;
                }
            }
        }
        w
    };

    let by_loss = search(Objective::Likelihood, STEP);
    println!("  fitted on LIKELIHOOD, step {STEP}");
    // Decoded kappa is PIECEWISE CONSTANT in the weights: it only moves when a label flips, so a
    // small step sees a flat objective and stops. A coarse start is the honest test of whether the
    // shipped point is a local optimum or the search simply could not reach past it.
    let by_kappa = search(Objective::DecodedKappa, 0.60);
    println!("  fitted on DECODED KAPPA, step 0.60");

    println!("
  {:<18} {:>9} {:>10} {:>10}", "weight", "shipped", "by loss", "by kappa");
    for j in 0..NW {
        println!("  {:<18} {:>9.3} {:>10.3} {:>10.3}", WEIGHT_NAMES[j], shipped[j], by_loss[j],
                 by_kappa[j]);
    }

    println!("\n  {:<22} {:>8} {:>8}   {:>10} {:>9} {:>5}   verdict",
             "cohort / objective", "shipped", "refit", "paired d", "bar +/-", "n");
    for set in [FIT, HELD[0], HELD[1]] {
        let nights = if set == FIT { load(FIT) } else { load(set) };
        if nights.is_empty() {
            println!("  {set:<22} no nights");
            continue;
        }
        let a = score(&nights, &shipped);
        let role = if set == FIT { "(FIT)" } else { "(HELD)" };
        for (label, cand) in [("by loss", &by_loss), ("by kappa", &by_kappa)] {
            let b = score(&nights, cand);
            let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| y - x).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let v = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("refit {} ({:.2}x)", if mean > 0.0 { "BETTER" } else { "WORSE" },
                        mean.abs() / bar)
            } else {
                "inside the bar".to_string()
            };
            println!("  {:<24} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {v}",
                     format!("{set} {role} {label}"), median(&a), median(&b), d.len());
        }
    }
    println!("\nEvery non-linearity, the pinned light class and the cycle prior are untouched. Only");
    println!("the twelve numbers moved, from a start that was already the shipped recipe.");
}
