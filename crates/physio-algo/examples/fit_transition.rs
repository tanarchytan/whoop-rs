//! Fit the transition matrix, the one part of the recipe the emission fit never touched.
//!
//!   cargo run --release -p physio-algo --example fit_transition
//!
//! Staging is `viterbi(emissions(..), transition)`. Fitting the EMISSION was measured and it loses
//! held-out. The transition is a separate mechanism with 16 hand-set numbers, and unlike the
//! emission it has a closed-form maximum-likelihood estimate: count the transitions the reference
//! actually makes and normalise. No optimiser, no hyperparameter, nothing to overfit but the counts.
//!
//! The isolation is deliberate but PARTIAL: both arms decode the SAME shipped emissions over the
//! same epochs, so what moves is the path search alone. Adopting a counted matrix into `Params`
//! would move the emissions too - `cycle_rem_onset_minutes` is non-zero, so `emissions_v2` resolves
//! its cycle anchor with a probe decode over `p.transition`. That half is held at the shipped matrix
//! here and is NOT measured; a matrix that loses below could still win, or lose harder, in the
//! recipe.
//!
//! Every cohort takes a turn counting. Counting on one and losing on two others is equally what an
//! UNREPRESENTATIVE counting cohort looks like, and DREAMT is a clinical apnea set whose dynamics
//! may simply not transfer - a rotation separates that from a claim about the objective.
//!
//! Same caveat as the emission fit: the shipped matrix was chosen watching all three cohorts, so the
//! baseline is not blind and part of its held-out margin is exposure this estimate never had.

mod common;

use common::{dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

/// The three PSG cohorts; each takes a turn as the counting set.
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Added to every transition count. A stage pair the reference never happens to show is rare, not
/// impossible, and a zero row in a log-space decoder forbids a path outright.
const LAPLACE: f64 = 1.0;
/// Nights shorter than this are not scored, matching the emission harness.
const MIN_EPOCHS: usize = 20;

struct Night {
    /// Shipped emissions, one row per epoch, in [`STAGE_ORDER`] columns.
    em: Vec<[f64; CLASSES]>,
    /// Reference label per epoch in our class order, or `None` where unlabelled.
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
        let em = emissions_v2(&prep, &Params::SHIPPED);
        // `emissions_v2` DROPS an epoch carrying neither HR nor gravity, so positional indexing
        // into `raw` is only valid while the grid is complete. Checked BEFORE the length skip, or
        // the hardest-collapsed grid is the one that leaves silently instead of tripping it.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { em, truth });
    }
    out
}

/// Row-normalised transition counts over consecutive LABELLED epoch pairs, in [`STAGE_ORDER`].
///
/// Only pairs where both epochs carry a label count. A gap in the reference is not a transition,
/// and treating it as one teaches the matrix a jump the wearer never made.
fn count_transitions(nights: &[Night]) -> ([[f64; CLASSES]; CLASSES], usize, usize) {
    let order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    let mut cnt = [[LAPLACE; CLASSES]; CLASSES];
    let mut pairs = 0usize;
    for nt in nights {
        for w in nt.truth.windows(2) {
            let (Some(a), Some(b)) = (w[0], w[1]) else { continue };
            // Our class index -> the decoder's column.
            let (fi, ti) = (
                order.iter().position(|c| *c == a).expect("class in STAGE_ORDER"),
                order.iter().position(|c| *c == b).expect("class in STAGE_ORDER"),
            );
            cnt[fi][ti] += 1.0;
            pairs += 1;
        }
    }
    // Cells the reference barely visits are dominated by the prior rather than by data, and a
    // reader should know how many before reading the matrix as an estimate.
    let sparse = cnt.iter().flatten().filter(|c| **c - LAPLACE < 5.0).count();
    for row in cnt.iter_mut() {
        let s: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    (cnt, pairs, sparse)
}

/// Per-night kappa under one transition matrix.
fn score(nights: &[Night], t: &[[f64; CLASSES]; CLASSES]) -> Vec<f64> {
    let mut ks = Vec::new();
    for nt in nights {
        let path = decode_v2(&nt.em, t);
        let (mut p, mut tr) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(stage_idx(path[k]));
            tr.push(*want);
        }
        if p.len() < MIN_EPOCHS {
            continue;
        }
        ks.push(kappa4(&confusion4(&p, &tr)));
    }
    ks
}

fn main() {
    println!("Both arms decode the SAME shipped emissions; only the 4x4 differs, so what moves is");
    println!("the path search alone - the cycle anchor stays resolved under the shipped matrix.");
    println!("Each cohort takes a turn counting, because counting on ONE and losing on the others");
    println!("is also what an unrepresentative cohort looks like.\n");

    let mut loaded: Vec<(&str, Vec<Night>)> = Vec::new();
    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("  {set:<20} no nights - not in the rotation\n");
            continue;
        }
        loaded.push((set, nights));
    }
    if loaded.is_empty() {
        println!("no nights - check the fixture root");
        return;
    }

    // The shipped arm does not depend on the counting cohort, so it is decoded once per scored
    // cohort. `median` sorts, so it takes a copy and leaves the scores in night order for pairing.
    let shipped: Vec<(Vec<f64>, f64)> = loaded
        .iter()
        .map(|(_, nights)| {
            let ks = score(nights, &Params::SHIPPED.transition);
            let mut sorted = ks.clone();
            let m = median(&mut sorted);
            (ks, m)
        })
        .collect();

    for (from, train) in &loaded {
        let (fitted, pairs, sparse) = count_transitions(train);
        println!("=== COUNTED on {from}: {} nights, {pairs} pairs, {sparse}/16 cells under 5 obs",
                 train.len());
        let head: String = STAGE_ORDER
            .iter()
            .map(|s| format!(" {:>8}", format!("{s:?}").to_lowercase()))
            .collect();
        println!("  {:<7}{head}", "from");
        for (i, st) in STAGE_ORDER.iter().enumerate() {
            println!("  {:<7} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", format!("{st:?}"),
                     fitted[i][0], fitted[i][1], fitted[i][2], fitted[i][3]);
        }
        println!("  {:<24} {:>7} {:>7}   {:>10} {:>9} {:>5}   verdict",
                 "scored on", "shipped", "counted", "paired d", "bar +/-", "n");
        for (i, (name, nights)) in loaded.iter().enumerate() {
            let (bk, bmed) = &shipped[i];
            let mut fk = score(nights, &fitted);
            let d: Vec<f64> = bk.iter().zip(&fk).map(|(a, b)| b - a).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let role = if name == from { "(counted on)" } else { "(HELD)" };
            let verdict = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("{} ({:.2}x the bar)", if mean > 0.0 { "BETTER" } else { "WORSE" },
                        mean.abs() / bar)
            } else {
                "inside the bar - noise".to_string()
            };
            println!("  {:<24} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {verdict}",
                     format!("{name} {role}"), bmed, median(&mut fk), d.len());
        }
        println!();
    }

    println!("If the counted matrix loses on held-out cohorts NO MATTER which one it was counted");
    println!("from, the objective is wrong. If it only loses when counted on one of them, that");
    println!("cohort's dynamics are unusual and the objective is not what was measured.");
}
