//! Does the weighted decode pick the right epochs, or only change HOW MANY of each class it calls?
//!
//!   cargo run --release -p physio-algo --example flatten_null
//!
//! `inverse-prevalence fc` raises balanced accuracy on all three cohorts while kappa and macro F1
//! fall on all three. BA is a mean of per-class recalls, so calling a rare class more raises it
//! whatever the extra calls are worth. The permuted-cost control cannot separate the two: shuffling
//! which class holds which weight destroys the ALIGNMENT with prevalence, not the flattening.
//!
//! The null here matches the flattening and randomises the choice. Take v2's own labels and move
//! epochs at random until the per-class counts EQUAL the weighted decode's, night by night: the same
//! predicted distribution, carrying no information about WHICH epoch. An arm that does not beat it
//! bought its BA with arithmetic.
//!
//! The null is fair on these three metrics for the reason it would be unfair on a hypnogram: a
//! confusion matrix cannot see order, so scattering the moved epochs costs it nothing here.
//!
//! Costs are fitted leave-one-night-out from the training nights' truth counts, so no night prices
//! its own prevalence. Both arm and null are scored against the same v2 baseline, paired per night.
//! Every score is a MEAN OVER NIGHTS, not the pooled confusion the dataset gates print.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::cardiac_emit::FIT_ORDER;
use physio_algo::sleep::markov_loss::{self, Costs};
use physio_algo::sleep::metrics::{
    balanced_accuracy, confusion4, kappa4, macro_f1, paired_bar, Confusion4,
};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, posterior, prepare_v2, SleepInput, STAGE_ORDER,
};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
/// Powers of the inverse-prevalence cost vector. 0 is unit costs, which IS posterior-marginal, so
/// it separates what the DECODE RULE costs from what the weighting costs. No power of either arm
/// reaches a kappa gain: 0 of 42 cells.
const POWERS: [f64; 7] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];

struct Night {
    /// Every epoch's posterior over `STAGE_ORDER`, from v2's own emissions and transitions.
    post: Vec<[f64; 4]>,
    /// v2's own log-emissions in `STAGE_ORDER` columns, so the same costs can tilt a VITERBI arm.
    em: Vec<[f64; 4]>,
    /// v2's shipped labels, in `stage_idx` order.
    base: Vec<usize>,
    truth: Vec<Option<usize>>,
    /// Labelled epochs per truth class, for the leave-one-out prevalence.
    counts: [u64; CLASSES],
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
        let input = SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let base: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let truth: Vec<Option<usize>> = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        let mut counts = [0u64; CLASSES];
        for t in truth.iter().flatten() {
            counts[*t] += 1;
        }
        let post = posterior::forward_backward(&em, |_| Params::SHIPPED.transition);
        out.push(Night { post, em: em[..].to_vec(), base, truth, counts });
    }
    out
}

/// Deterministic pseudo-random draw, so two runs move the same epochs.
fn noise(seed: u64, k: usize) -> f64 {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(k as u64 + 1);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Inverse-prevalence costs from every night BUT `skip`, in the decoder's column order. `None` when
/// the remaining nights leave a class empty, which is what the prevalence would divide by.
fn costs_without(nights: &[Night], skip: usize, power: f64) -> Option<Costs> {
    let mut counts = [0u64; CLASSES];
    for (i, nt) in nights.iter().enumerate() {
        if i != skip {
            for (c, n) in counts.iter_mut().zip(&nt.counts) {
                *c += n;
            }
        }
    }
    let w = markov_loss::geometric_scale(markov_loss::inverse_prevalence(counts)?, power)?;
    let fc = markov_loss::reindex(w, FIT_ORDER, STAGE_ORDER)?;
    Some(Costs { fc, ..Costs::UNIT })
}

/// v2's labels with epochs moved at random until each class is called as often as `target` calls it.
/// Surplus epochs leave the over-called classes at random and fill the under-called ones in turn, so
/// the predicted distribution matches while the choice of epoch carries nothing.
///
/// Counted and moved over the LABELLED epochs alone, because those are the ones scored: matching on
/// the whole night instead leaves the two calling different shares of what the metrics see.
fn matched_shuffle(base: &[usize], target: &[usize], truth: &[Option<usize>], seed: u64) -> Vec<usize> {
    let scored: Vec<usize> = (0..base.len()).filter(|k| truth[*k].is_some()).collect();
    let (mut have, mut want) = ([0i64; CLASSES], [0i64; CLASSES]);
    for k in &scored {
        have[base[*k]] += 1;
        want[target[*k]] += 1;
    }
    let mut out = base.to_vec();
    // Epochs of each over-called class, in a fixed pseudo-random order: the first `surplus` move.
    let mut spare: Vec<usize> = Vec::new();
    for (c, (h, w)) in have.iter().zip(&want).enumerate() {
        let surplus = (h - w).max(0) as usize;
        if surplus == 0 {
            continue;
        }
        let mut of_class: Vec<usize> = scored.iter().copied().filter(|k| base[*k] == c).collect();
        of_class.sort_by(|a, b| noise(seed ^ c as u64, *a).total_cmp(&noise(seed ^ c as u64, *b)));
        spare.extend(of_class.into_iter().take(surplus));
    }
    spare.sort_by(|a, b| noise(seed, *a).total_cmp(&noise(seed, *b)));
    let mut next = spare.into_iter();
    for (c, (h, w)) in have.iter().zip(&want).enumerate() {
        for _ in 0..(w - h).max(0) {
            match next.next() {
                Some(k) => out[k] = c,
                // Deficits and surpluses sum to zero, so this cannot run dry.
                None => unreachable!("surplus is exactly the deficit"),
            }
        }
    }
    out
}

/// One night's confusion over its labelled epochs. `None` when the night carries no truth.
fn cm(pred: &[usize], truth: &[Option<usize>]) -> Option<Confusion4> {
    let (p, t): (Vec<usize>, Vec<usize>) = truth
        .iter()
        .enumerate()
        .filter_map(|(k, v)| v.map(|v| (pred[k], v)))
        .unzip();
    (p.len() >= MIN_EPOCHS).then(|| confusion4(&p, &t))
}

/// Each class's predicted share over its labelled share, pooled. 1.0 is neutral.
fn calling_ratio(pred: &[Vec<usize>], nights: &[Night]) -> [f64; CLASSES] {
    let (mut called, mut labelled) = ([0f64; CLASSES], [0f64; CLASSES]);
    for (p, nt) in pred.iter().zip(nights) {
        for (k, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                called[p[k]] += 1.0;
                labelled[*t] += 1.0;
            }
        }
    }
    let (c, l) = (called.iter().sum::<f64>(), labelled.iter().sum::<f64>());
    std::array::from_fn(|i| match labelled[i] {
        0.0 => f64::NAN,
        _ => (called[i] / c) / (labelled[i] / l),
    })
}

/// Paired mean and bar of `f` over the nights, arm minus reference. Prints `n/a` on an empty cohort.
fn delta(arm: &[Confusion4], reference: &[Confusion4], f: impl Fn(&Confusion4) -> Option<f64>) -> String {
    let d: Vec<f64> = arm
        .iter()
        .zip(reference)
        .filter_map(|(a, b)| Some(f(a)? - f(b)?))
        .collect();
    match paired_bar(&d) {
        None => "     n/a         ".to_string(),
        Some((m, bar)) => {
            let verdict = if m.abs() <= bar { "matches" } else if m > 0.0 { "AHEAD  " } else { "behind " };
            format!("{m:+.4} {bar:.4} {verdict}")
        }
    }
}

fn main() {
    println!("FLATTENING-MATCHED NULL for the weighted decode.\n");
    println!("`shuffled to match` holds v2's labels but calls each class as often as the weighted");
    println!("decode does, choosing which epochs AT RANDOM. The weighted decode earns its balanced");
    println!("accuracy only if it beats that row; matching it means the gain was the calling share.\n");

    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("== {set}: no nights ==\n");
            continue;
        }
        let base: Vec<Vec<usize>> = nights.iter().map(|nt| nt.base.clone()).collect();
        let base_cm: Vec<Confusion4> =
            nights.iter().zip(&base).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
        let r = calling_ratio(&base, &nights);
        println!("== {set}   n={} nights, {} scored ==", nights.len(), base_cm.len());
        println!("  {:<34} {:>8} {:>8} {:>8}   called/labelled w l d r",
                 "arm", "kappa4", "bal acc", "macroF1");
        // The MEAN over nights, matching the paired deltas below. Never the pooled-confusion kappa
        // the dataset gates print - the two differ and comparing across them is a false reading.
        let mean_night = |cms: &[Confusion4], f: fn(&Confusion4) -> Option<f64>| -> f64 {
            let v: Vec<f64> = cms.iter().filter_map(f).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        println!("  {:<34} {:>8.4} {:>8.4} {:>8.4}   {:.2} {:.2} {:.2} {:.2}",
                 "v2 shipped recipe (reference)",
                 mean_night(&base_cm, |c| Some(kappa4(c))), mean_night(&base_cm, balanced_accuracy),
                 mean_night(&base_cm, macro_f1), r[0], r[1], r[2], r[3]);

        for power in POWERS {
            let mut arm: Vec<Vec<usize>> = Vec::with_capacity(nights.len());
            for (i, nt) in nights.iter().enumerate() {
                let Some(costs) = costs_without(&nights, i, power) else { continue };
                arm.push(
                    posterior::decode_with_costs(&nt.post, &costs)
                        .iter()
                        .map(|s| stage_idx(*s))
                        .collect(),
                );
            }
            // The same costs as a PRIOR SHIFT on a viterbi decode. The posterior arm changes the
            // decode rule AND the weighting at once; this changes only the weighting.
            let mut tilt: Vec<Vec<usize>> = Vec::with_capacity(nights.len());
            for (i, nt) in nights.iter().enumerate() {
                let Some(costs) = costs_without(&nights, i, power) else { continue };
                let em: Vec<[f64; 4]> = nt
                    .em
                    .iter()
                    .map(|e| core::array::from_fn(|c| e[c] + costs.fc[c].ln()))
                    .collect();
                tilt.push(
                    decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect(),
                );
            }
            if arm.len() != nights.len() || tilt.len() != nights.len() {
                println!("  fc^{power}: a fold left a class empty, skipped");
                continue;
            }
            let null: Vec<Vec<usize>> = nights
                .iter()
                .zip(&arm)
                .enumerate()
                .map(|(i, (nt, a))| {
                    let s = matched_shuffle(&nt.base, a, &nt.truth, i as u64);
                    // The whole claim rests on the two calling the same counts, so check it here.
                    let count = |v: &[usize]| {
                        nt.truth.iter().enumerate().filter(|(_, t)| t.is_some()).fold(
                            [0usize; CLASSES],
                            |mut c, (k, _)| {
                                c[v[k]] += 1;
                                c
                            },
                        )
                    };
                    assert_eq!(count(&s), count(a), "night {i}: the shuffle did not match");
                    s
                })
                .collect();
            for (label, pred) in [("inverse-prevalence fc", &arm), ("shuffled to match", &null)] {
                let tag = format!("{label}, power {power}");
                let cms: Vec<Confusion4> =
                    nights.iter().zip(pred).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
                let q = calling_ratio(pred, &nights);
                println!("  {:<34} {:>8.4} {:>8.4} {:>8.4}   {:.2} {:.2} {:.2} {:.2}",
                         tag, mean_night(&cms, |c| Some(kappa4(c))),
                         mean_night(&cms, balanced_accuracy), mean_night(&cms, macro_f1),
                         q[0], q[1], q[2], q[3]);
            }
            let arm_cm: Vec<Confusion4> =
                nights.iter().zip(&arm).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
            let null_cm: Vec<Confusion4> =
                nights.iter().zip(&null).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
            println!("      fc^{power} vs the SHUFFLE at the same calling share, paired on {} night(s):",
                     arm_cm.len());
            println!("        kappa4  {}", delta(&arm_cm, &null_cm, |c| Some(kappa4(c))));
            println!("        bal acc {}", delta(&arm_cm, &null_cm, balanced_accuracy));
            println!("        macroF1 {}", delta(&arm_cm, &null_cm, macro_f1));
            println!("      the SHUFFLE vs v2, which is the arithmetic alone:");
            println!("        bal acc {}", delta(&null_cm, &base_cm, balanced_accuracy));
            // The selection pair, both arms against v2 on the same nights: a BA gain is only a
            // gain if kappa holds beside it.
            let v2_cm: Vec<Confusion4> = nights
                .iter()
                .zip(&arm)
                .filter_map(|(nt, p)| cm(p, &nt.truth).and(cm(&nt.base, &nt.truth)))
                .collect();
            let tilt_null: Vec<Vec<usize>> = nights
                .iter()
                .zip(&tilt)
                .enumerate()
                .map(|(i, (nt, a))| matched_shuffle(&nt.base, a, &nt.truth, i as u64))
                .collect();
            let tilt_cm: Vec<Confusion4> =
                nights.iter().zip(&tilt).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
            let tn_cm: Vec<Confusion4> =
                nights.iter().zip(&tilt_null).filter_map(|(nt, p)| cm(p, &nt.truth)).collect();
            let qt = calling_ratio(&tilt, &nights);
            println!("  {:<34} {:>8.4} {:>8.4} {:>8.4}   {:.2} {:.2} {:.2} {:.2}",
                     format!("TILTED VITERBI fc, power {power}"),
                     mean_night(&tilt_cm, |c| Some(kappa4(c))),
                     mean_night(&tilt_cm, balanced_accuracy), mean_night(&tilt_cm, macro_f1),
                     qt[0], qt[1], qt[2], qt[3]);
            println!("      tilted fc^{power} vs ITS OWN matched shuffle:");
            println!("        kappa4  {}", delta(&tilt_cm, &tn_cm, |c| Some(kappa4(c))));
            println!("        bal acc {}", delta(&tilt_cm, &tn_cm, balanced_accuracy));
            println!("      tilted fc^{power} vs V2 ITSELF, the selection pair:");
            println!("        kappa4  {}", delta(&tilt_cm, &v2_cm, |c| Some(kappa4(c))));
            println!("        bal acc {}", delta(&tilt_cm, &v2_cm, balanced_accuracy));
            println!("        macroF1 {}", delta(&tilt_cm, &v2_cm, macro_f1));
            println!("      fc^{power} vs V2 ITSELF, the selection pair:");
            println!("        kappa4  {}", delta(&arm_cm, &v2_cm, |c| Some(kappa4(c))));
            println!("        bal acc {}", delta(&arm_cm, &v2_cm, balanced_accuracy));
            println!("        macroF1 {}", delta(&arm_cm, &v2_cm, macro_f1));
        }
        println!();
    }
}
