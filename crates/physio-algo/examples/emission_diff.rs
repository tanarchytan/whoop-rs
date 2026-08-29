//! What the two emission models actually differ in, and which difference costs the kappa.
//!
//!   cargo run --release -p physio-algo --example emission_diff
//!
//! The shipped emission is not "a linear model with hand-set weights". It is a linear model with a
//! STRUCTURE, and the structure is the interesting part:
//!
//!   em[LIGHT] = bias only - no feature touches it. Light is the null hypothesis.
//!   em[DEEP]  = 3 z-scored features + a hinge on the flatness RANK + a respiration term
//!   em[REM]   = 3 z-scored features - the same respiration term
//!   em[AWAKE] = motion + a DEADZONED cardiac term that a stillness test can CLAMP + a centred
//!               rotation RANK that SHIPPED weights at 0.0 + a jerk gate
//!
//! Twelve weight slots, ELEVEN live under SHIPPED, four hand-placed non-linearities, and one class
//! pinned to its prior. The fit has 4 x 29 = 116 free parameters, no non-linearity, and nothing
//! pinned - it must LEARN that light is the default from data that is 49-61% light.
//!
//! This prints the largest fitted weights, per-class recall and precision for both engines, and an arm
//! with LIGHT's fitted weights zeroed AFTER the fit: a post-hoc ablation of the shipped structure's
//! single biggest prior, not a refit under that constraint.

mod common;

use common::lr::{design, fit, scores, standardiser, CLASSES, NCOL};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar, precision, recall};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const FIT: &str = "dreamt";
const HELD: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASS_NAME: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
/// How many of the NCOL columns the weight table prints, largest |weight| in any class first.
const TOP_COLUMNS: usize = 10;
/// Our class index for light, the one the shipped emission pins to its prior.
const LIGHT: usize = 1;

struct Night {
    x: Vec<[f64; NCOL]>,
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
        let em = emissions_v2(&prepare_v2(&input, &Params::SHIPPED), &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        // `emissions_v2` DROPS an epoch with neither HR nor gravity. Truncating features to the
        // emission length would then misalign everything after the hole rather than failing.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        assert!(f.len() >= em.len(), "{}: fewer feature rows than emissions", dir.display());
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { x: f[..em.len()].iter().map(|r| r.values()).collect(), em, truth });
    }
    out
}

/// Log-softmax emissions from the fitted weights, in STAGE_ORDER columns.
fn fitted_em(nt: &Night, w: &[Vec<f64>], m: &[f64; NCOL], sd: &[f64; NCOL]) -> Vec<[f64; CLASSES]> {
    let to_order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    nt.x
        .iter()
        .map(|row| {
            let z = scores(w, &design(row, m, sd, &[]));
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            std::array::from_fn(|c| z[to_order[c]] - lse)
        })
        .collect()
}

/// Pooled confusion over a cohort, plus per-night kappas for the paired test. A night under
/// MIN_EPOCHS labelled epochs enters neither, so both cover the same nights.
fn confuse(
    nights: &[Night],
    em_of: impl Fn(&Night) -> Vec<[f64; CLASSES]>,
) -> ([[i64; CLASSES]; CLASSES], Vec<f64>) {
    let mut cm = [[0i64; CLASSES]; CLASSES];
    let mut ks = Vec::new();
    for nt in nights {
        let pred: Vec<usize> = decode_v2(&em_of(nt), &Params::SHIPPED.transition)
            .iter()
            .map(|s| stage_idx(*s))
            .collect();
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(pred[k]);
            t.push(*want);
        }
        if p.len() >= MIN_EPOCHS {
            let night = confusion4(&p, &t);
            for (row, add) in cm.iter_mut().zip(&night) {
                for (cell, v) in row.iter_mut().zip(add) {
                    *cell += v;
                }
            }
            ks.push(kappa4(&night));
        }
    }
    (cm, ks)
}

fn per_class(cm: &[[i64; CLASSES]; CLASSES]) -> String {
    (0..CLASSES)
        .map(|c| {
            let r = recall(cm, c).unwrap_or(f64::NAN);
            let p = precision(cm, c).unwrap_or(f64::NAN);
            format!("{:>5.2}/{:<5.2}", r, p)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let train = load(FIT);
    if train.is_empty() {
        println!("no {FIT} nights - check the fixture root");
        return;
    }
    let rows: Vec<([f64; NCOL], usize)> = train
        .iter()
        .flat_map(|nt| nt.x.iter().zip(&nt.truth).filter_map(|(r, t)| t.map(|t| (*r, t))))
        .collect();
    let x: Vec<[f64; NCOL]> = rows.iter().map(|(r, _)| *r).collect();
    let y: Vec<usize> = rows.iter().map(|(_, t)| *t).collect();
    let (m, sd) = standardiser(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design(r, &m, &sd, &[])).collect();
    let w = fit(&dx, &y, WEIGHT_POWER);

    // LIGHT's feature weights zeroed AFTER the joint fit: a post-hoc ablation, not a refit under
    // the constraint. The other three rows keep values optimised WITH a free light row, so this is
    // what the shipped pin costs THIS fit, not what a light-pinned fit could reach.
    let mut ablated = w.clone();
    for v in ablated[LIGHT].iter_mut().take(NCOL) {
        *v = 0.0;
    }

    println!("\n=== THE FITTED WEIGHTS, standardised so columns are comparable");
    println!("Each row is a class; a large value means that column moves that class. The shipped");
    println!("emission gives LIGHT a weight of exactly zero on everything.");
    println!("Rows: the {TOP_COLUMNS} columns of {NCOL} with the largest |weight| in any class. The");
    println!("|weight| totals below cover all {NCOL}.\n");
    println!("  {:<16}{}", "column", CLASS_NAME.map(|c| format!("{c:>9}")).join(""));
    let mut ranked: Vec<(f64, usize)> = (0..NCOL)
        .map(|c| ((0..CLASSES).map(|k| w[k][c].abs()).fold(0.0, f64::max), c))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (_, c) in ranked.iter().take(TOP_COLUMNS) {
        println!("  {:<16} {:>8.3} {:>8.3} {:>8.3} {:>8.3}", Features::NAMES[*c],
                 w[0][*c], w[1][*c], w[2][*c], w[3][*c]);
    }
    let light_mass: f64 = (0..NCOL).map(|c| w[LIGHT][c].abs()).sum();
    let other_mass: f64 = (0..CLASSES)
        .filter(|k| *k != LIGHT)
        .map(|k| (0..NCOL).map(|c| w[k][c].abs()).sum::<f64>())
        .sum();
    println!("\n  total |weight| on LIGHT {light_mass:.1}, on the other three {other_mass:.1}");
    println!("  the shipped emission spends ZERO on light and lets the prior carry it.");

    println!("\n=== PER-CLASS recall/precision, and what ablating LIGHT does");
    println!("The kappa column is a per-cohort MEDIAN. The `ablation, paired` row is the MEAN of");
    println!("the per-night deltas over those same nights, not the difference of two medians.");
    println!("  {:<26} {:>14}   {}", "cohort / engine", "kappa (median)",
             CLASS_NAME.map(|c| format!("{c:>5}      ")).join(" "));
    for set in [FIT, HELD[0], HELD[1]] {
        // The fit cohort is already in `train`; only the held-out sets need loading.
        let held = (set != FIT).then(|| load(set));
        let nights: &[Night] = held.as_deref().unwrap_or(&train);
        if nights.is_empty() {
            println!("  {set:<26} no nights");
            continue;
        }
        let role = if set == FIT { "FITTED" } else { "HELD" };
        let (cb, kb) = confuse(nights, |nt| nt.em.clone());
        let (cf, kf) = confuse(nights, |nt| fitted_em(nt, w.as_slice(), &m, &sd));
        let (ca, ka) = confuse(nights, |nt| fitted_em(nt, ablated.as_slice(), &m, &sd));
        println!("  {:<26} {:>14.3}   {}", format!("{set} ({role}) shipped"),
                 median(&mut kb.clone()), per_class(&cb));
        println!("  {:<26} {:>14.3}   {}", "  tanv1", median(&mut kf.clone()), per_class(&cf));
        println!("  {:<26} {:>14.3}   {}", "  tanv1, LIGHT ablated", median(&mut ka.clone()),
                 per_class(&ca));
        let d: Vec<f64> = kf.iter().zip(&ka).map(|(a, b)| b - a).collect();
        let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
        let v = if !mean.is_finite() {
            "-".to_string()
        } else if mean.abs() > bar {
            format!("{} {:.2}x", if mean > 0.0 { "ABLATION HELPS" } else { "ablation hurts" },
                    mean.abs() / bar)
        } else {
            "inside the bar".into()
        };
        println!("  {:<26} {mean:>+14.4} +/-{bar:.4}   {} of {} nights   {v}",
                 "  ablation, paired (mean)", d.len(), nights.len());
    }
    println!("\nrecall/precision per class. The shipped engine's structure says light is what you");
    println!("get when nothing argues otherwise; the fit has to learn that from the data.");
}
