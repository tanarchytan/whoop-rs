//! Refit the shipped emission's OWN twelve weights, keeping every non-linearity it has.
//!
//!   cargo run --release -p physio-algo --example fit_v2_weights
//!
//! The shipped emission is LINEAR IN ITS OWN WEIGHTS given its transforms, EXCEPT for the stillness
//! clamp: that one acts on the WEIGHTED awake-cardiac pair, so the objective is piecewise-linear and
//! non-convex in those two weights. The deadzone, that clamp, the deep-gate hinge, the rank
//! transforms, the cycle prior and the pinned light class all stay exactly as they are, and only the
//! twelve numbers move.
//!
//! The LIKELIHOOD arm searches over `Terms`, whose cycle prior is anchored on a staging under the
//! SHIPPED weights, so away from that point it is a surrogate. The DECODED KAPPA arm and every
//! printed kappa go through `emissions_v2`, which re-resolves that anchor under the weights being
//! scored, so the table is what those weights earn once they are set into `Params`.
//!
//! It starts at the shipped weights, and L2 pulls toward them rather than toward zero - a weight
//! only moves if the data pays for the move. With the clamp in it the objective is not convex, so
//! what each arm reports is a LOCAL optimum reached from that one start.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::{dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, weights_of, Prepared,
    SleepInput, Terms, WEIGHT_NAMES,
};

const FIT: &str = "dreamt";
const HELD: [&str; 2] = ["aauwss", "sleep-accel"];
const NW: usize = 12;
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const ITERS: usize = 4_000;
const STEP: f64 = 0.02;
/// Step below which the search stops. Reaching it is the ONLY converged exit; [`ITERS`] is a cap.
const STEP_FLOOR: f64 = 1e-4;
/// The decoded-kappa arm's step. Coarse because that objective is piecewise constant in the weights.
const KAPPA_STEP: f64 = 0.60;
/// Pull toward the SHIPPED weights, not toward zero. The shipped values are evidence, so a weight
/// should only move where the data pays for the move.
const L2_TO_SHIPPED: f64 = 0.01;

/// What the search optimises. A fit maximises LIKELIHOOD; the product reports a DECODED KAPPA, and
/// they are not the same function of these weights.
#[derive(Clone, Copy)]
enum Objective {
    Likelihood,
    DecodedKappa,
}

struct Night {
    /// The features the real staging path re-reads under each candidate's own weights.
    prep: Prepared,
    terms: Terms,
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
        let input =
            SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if terms.design.len() < MIN_EPOCHS {
            continue;
        }
        // `prepare_v2` DROPS an epoch with neither HR nor gravity. Indexing truth by design position
        // would then misalign every epoch after the hole rather than failing.
        assert_eq!(terms.design.len(), n, "{}: {n} epochs of truth against {} of design",
                   dir.display(), terms.design.len());
        let truth = (0..terms.design.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { prep, terms, truth });
    }
    out
}

/// The inverse of [`weights_of`]: the shipped recipe carrying these twelve weights and nothing else
/// moved. `main` round-trips a probe that is DISTINCT in every slot, so a slot that drifts out of
/// order fails rather than fits.
fn params_with(w: &[f64; NW]) -> Params {
    let [deep_hrv, deep_hr, deep_motion, deep_gate_slope, rem_hrv, rem_motion, rem_hr, awake_motion,
         awake_hrv, awake_hr, awake_turn, resp_weight] = *w;
    Params { deep_hrv, deep_hr, deep_motion, deep_gate_slope, rem_hrv, rem_motion, rem_hr,
             awake_motion, awake_hrv, awake_hr, awake_turn, resp_weight, ..Params::SHIPPED }
}

/// Our class index is [wake, light, deep, rem]; the emission's columns are STAGE_ORDER.
fn col_of(class: usize) -> usize {
    use physio_algo::sleep::STAGE_ORDER;
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

/// Weighted multinomial log-loss over the labelled epochs, under one weight vector.
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

/// The shared inverse-frequency weights at power 0.5, over every labelled epoch of the cohort.
fn class_weights(nights: &[Night]) -> [f64; CLASSES] {
    let labels: Vec<usize> =
        nights.iter().flat_map(|nt| nt.truth.iter().flatten().copied()).collect();
    common::lr::class_weights(&labels, 0.5)
}

/// Per-night kappa under one weight vector, through the staging path the product runs: the cycle
/// prior's onset anchor is re-resolved under these weights, not kept at the one SHIPPED resolved.
fn score(nights: &[Night], w: &[f64; NW]) -> Vec<f64> {
    let params = params_with(w);
    let mut ks = Vec::new();
    for nt in nights {
        let em = emissions_v2(&nt.prep, &params);
        let path: Vec<usize> =
            decode_v2(&em, &params.transition).iter().map(|s| stage_idx(*s)).collect();
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

/// How a search left its loop, from the step it stopped at. A run that ran out of iterations is still
/// moving, and its weights are wherever it happened to be, not an optimum.
fn exit_of(step: f64) -> String {
    if step < STEP_FLOOR {
        format!("converged at step {step:.1e}")
    } else {
        format!("UNCONVERGED - hit the {ITERS}-iteration cap still moving at step {step:.4}")
    }
}

fn main() {
    let train = load(FIT);
    if train.is_empty() {
        println!("no {FIT} nights - check the fixture root");
        return;
    }
    let shipped = weights_of(&Params::SHIPPED);
    // The shipped vector repeats 0.5 and 0.6, so a crossed pair round-trips through it unchanged.
    // Twelve distinct values are what makes the round trip discriminating.
    let probe: [f64; NW] = std::array::from_fn(|j| j as f64 + 1.0);
    assert_eq!(probe, weights_of(&params_with(&probe)), "params_with is not weights_of inverted");
    let cw = class_weights(&train);
    println!("FIT on {FIT}: {} nights, class weights {:.2?}\n", train.len(), cw);

    // Coordinate descent. Twelve parameters, a clamp in the objective, and a start that is already
    // good - and one of the two objectives is a decoded kappa with no derivative at all.
    let search = |objective: Objective, step0: f64| -> ([f64; NW], f64) {
        let obj = |w: &[f64; NW]| -> f64 {
            let pen: f64 =
                (0..NW).map(|j| (w[j] - shipped[j]).powi(2)).sum::<f64>() * L2_TO_SHIPPED;
            let base = match objective {
                Objective::Likelihood => loss(&train, w, &cw),
                // Negated so both objectives are minimised. This is the MEDIAN per-night kappa,
                // while the verdict below is a paired MEAN over the same nights, so the FIT row can
                // read WORSE on a refit the search selected.
                Objective::DecodedKappa => -median(&mut score(&train, w)),
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
                if step < STEP_FLOOR {
                    break;
                }
            }
        }
        (w, step)
    };

    let (by_loss, loss_step) = search(Objective::Likelihood, STEP);
    println!("  fitted on LIKELIHOOD, step {STEP} - {}", exit_of(loss_step));
    // Decoded kappa is PIECEWISE CONSTANT in the weights, and the median flattens it further: it
    // moves only when the CENTRAL night's labels flip, so a small step sees nothing and stops. A
    // coarse start is the honest test of whether the search could reach past the shipped point.
    let (by_kappa, kappa_step) = search(Objective::DecodedKappa, KAPPA_STEP);
    println!("  fitted on DECODED KAPPA, step {KAPPA_STEP} - {}", exit_of(kappa_step));

    println!("\n  {:<18} {:>9} {:>10} {:>10}", "weight", "shipped", "by loss", "by kappa");
    for j in 0..NW {
        println!("  {:<18} {:>9.3} {:>10.3} {:>10.3}", WEIGHT_NAMES[j], shipped[j], by_loss[j],
                 by_kappa[j]);
    }

    println!("\n  {:<28} {:>8} {:>8}   {:>10} {:>9} {:>5}   verdict",
             "cohort / objective", "shipped", "refit", "paired d", "bar +/-", "n");
    for set in [FIT, HELD[0], HELD[1]] {
        // The fit cohort is already in `train`; only the held-out sets need loading.
        let held = (set != FIT).then(|| load(set));
        let nights: &[Night] = held.as_deref().unwrap_or(&train);
        if nights.is_empty() {
            println!("  {set:<28} no nights");
            continue;
        }
        let a = score(nights, &shipped);
        let role = if set == FIT { "(FIT)" } else { "(HELD)" };
        for (label, cand) in [("by loss", &by_loss), ("by kappa", &by_kappa)] {
            let b = score(nights, cand);
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
            println!("  {:<28} {:>8.3} {:>8.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {v}",
                     format!("{set} {role} {label}"), median(&mut a.to_vec()),
                     median(&mut b.to_vec()), d.len());
        }
    }
    println!("\nEvery non-linearity, the pinned light class and the cycle prior keep their own recipe;");
    println!("the prior's onset anchor is re-resolved under whichever weights the row scores. Only the");
    println!("twelve numbers moved, from a start that was already the shipped recipe.");
}
