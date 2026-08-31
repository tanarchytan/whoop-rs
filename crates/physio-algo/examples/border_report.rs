//! THE BORDER: everything an engine must be measured on, for any engine.
//!
//!   cargo run --release -p physio-algo --example border_report
//!
//! Phase 0's deliverable. Not a report about the shipped recipe - a scoring card that takes a
//! [`SleepConfig`] and prints the same six blocks whatever produced the hypnogram. v2 appears once, as
//! a NULL READING - where the old engine happens to sit, never a target and never the subject.
//!
//! Six blocks, each answering something the others cannot:
//!   KAPPA4      epoch-wise agreement over wake/light/deep/REM
//!   KAPPA3      the same staging as wake/NREM/REM, which is what most wearable papers report
//!   CI          percentile bootstrap over RECORDINGS - how much the figure would move on new nights
//!   RECALL      per class, because kappa is dominated by whichever stage holds the most epochs
//!   AGREEMENT   Bland-Altman on the minutes a user reads; kappa cannot say if the AMOUNT is right
//!   SLOPE       proportional bias - away from zero the error depends on the magnitude and one bias
//!               figure describes nobody
//!
//! An arm is added by adding a `SleepConfig`, not by editing this file's scoring.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, require_psg};
use physio_algo::sleep::agreement::{bland_altman, summarise, NightSummary};
use physio_algo::sleep::metrics::{
    bootstrap_kappa_ci, confusion4, kappa3, kappa4, kappa_after_reassignment, kappa_class_bonus,
    merge3, recall, truth_marginals, Confusion4,
};
use physio_algo::sleep::pipeline::{run, SleepConfig};
use physio_algo::sleep::{epoch_starts_v2, params::Params, SleepInput};

const EPOCH: i64 = 30;
const EPOCH_MIN: f64 = 0.5;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASS_NAMES: [&str; 4] = ["wake", "light", "deep", "rem"];
/// Labels must cover this much of a night's own span before it can carry a summary; a sparse night
/// reports a short TST that is a labelling gap, not a wearer awake.
const MIN_LABEL_DENSITY: f64 = 0.95;
const BOOTSTRAP_DRAWS: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0x5EED;

struct Night {
    cm: Confusion4,
    /// `None` when the night is too sparsely labelled to summarise.
    pair: Option<(NightSummary, NightSummary)>,
}

/// Score one cohort under one config. Labels are aligned to truth BY TIME, not by position: an epoch
/// carrying neither HR nor gravity is dropped from the staging, so the sequences can differ in length.
fn score(ds: &str, cfg: &SleepConfig, p: &Params) -> Vec<Night> {
    require_psg(ds);
    let mut out = Vec::new();
    for dir in &dirs_of(ds) {
        let truth = read_truth(dir);
        let Some((w0, w1, _)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if truth.is_empty() || accel.is_empty() {
            continue;
        }
        let input = SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let st = run(&input, cfg, p);
        let (Some(stages), Some(prep)) = (st.stages.as_ref(), st.prepared.as_ref()) else { continue };

        let starts = epoch_starts_v2(prep);
        let mut at: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (s, lab) in starts.iter().zip(stages) {
            at.insert(*s, *lab as usize);
        }

        let (mut d, mut r) = (Vec::new(), Vec::new());
        for (k, t) in &truth {
            if !(0..4).contains(t) {
                continue;
            }
            if let Some(lab) = at.get(&(w0 + *k as i64 * EPOCH)) {
                d.push(*lab);
                r.push(*t as usize);
            }
        }
        if d.is_empty() {
            continue;
        }
        let span = *truth.keys().last().unwrap() - *truth.keys().next().unwrap() + 1;
        let dense = (d.len() as f64) >= MIN_LABEL_DENSITY * span as f64;
        out.push(Night {
            cm: confusion4(&d, &r),
            pair: dense.then(|| (summarise(&d, EPOCH_MIN), summarise(&r, EPOCH_MIN)))
                .and_then(|(a, b)| Some((a?, b?))),
        });
    }
    out
}

fn pooled(nights: &[Night]) -> Confusion4 {
    let mut cm = [[0i64; 4]; 4];
    for n in nights {
        for (o, a) in cm.iter_mut().zip(&n.cm) {
            for (x, y) in o.iter_mut().zip(a) {
                *x += *y;
            }
        }
    }
    cm
}

fn card(ds: &str, arm: &str, nights: &[Night]) {
    let cm = pooled(nights);
    let cms: Vec<Confusion4> = nights.iter().map(|n| n.cm).collect();
    let ci = bootstrap_kappa_ci(&cms, BOOTSTRAP_DRAWS, 0.05, BOOTSTRAP_SEED);
    let pairs: Vec<&(NightSummary, NightSummary)> =
        nights.iter().filter_map(|n| n.pair.as_ref()).collect();

    println!("\n== {ds}  arm: {arm}  n={} ==", nights.len());
    let ci_txt = match ci {
        Some((lo, hi)) => format!("{lo:.4} .. {hi:.4}  (+/-{:.4})", (hi - lo) / 2.0),
        None => "-".into(),
    };
    println!("  kappa4 {:.4}   kappa3 {:.4}   95% CI {ci_txt}", kappa4(&cm), kappa3(&merge3(&cm)));

    let per: Vec<String> = (0..4)
        .map(|c| match recall(&cm, c) {
            Some(v) => format!("{} {:.3}", CLASS_NAMES[c], v),
            None => format!("{} -", CLASS_NAMES[c]),
        })
        .collect();
    println!("  recall: {}", per.join("   "));

    println!("  {:<11} {:>8} {:>8} {:>19} {:>8} {:>7}", "measure", "bias", "sd", "95% LoA", "slope", "r");
    for (name, get) in [
        ("TST", (|s: &NightSummary| s.tst) as fn(&NightSummary) -> f64),
        ("WASO", |s| s.waso),
        ("efficiency", |s| s.efficiency),
        ("deep", |s| s.deep),
        ("rem", |s| s.rem),
    ] {
        let dev: Vec<f64> = pairs.iter().map(|(d, _)| get(d)).collect();
        let refr: Vec<f64> = pairs.iter().map(|(_, r)| get(r)).collect();
        match bland_altman(&dev, &refr) {
            Some(a) => println!(
                "  {name:<11} {:>8.1} {:>8.1} {:>8.1} .. {:>6.1} {:>8.3} {:>7.3}",
                a.bias, a.sd, a.loa_lo, a.loa_hi, a.slope, a.r
            ),
            None => println!("  {name:<11} too few pairs"),
        }
    }
    // D11: kappa is a ratio of linear forms, so its own optimal rule is not argmax of the posterior.
    let (bonus, t) = (kappa_class_bonus(&cm), truth_marginals(&cm));
    let bcells: Vec<String> =
        (0..4).map(|c| format!("{} +{:.3}", CLASS_NAMES[c], bonus[c])).collect();
    println!("  kappa bonus (D11): {}", bcells.join("   "));
    // The exchange rate: relabel 1% of epochs toward the rarest class, correct count untouched.
    let rarest = (0..4).min_by(|a, b| t[*a].total_cmp(&t[*b])).unwrap_or(0);
    let commonest = (0..4).max_by(|a, b| t[*a].total_cmp(&t[*b])).unwrap_or(0);
    if let Some(k2) = kappa_after_reassignment(&cm, commonest, rarest, 0.01) {
        println!(
            "  1% of predictions {} -> {} at the SAME accuracy: kappa {:.4} -> {:.4} ({:+.4})",
            CLASS_NAMES[commonest], CLASS_NAMES[rarest], kappa4(&cm), k2, k2 - kappa4(&cm)
        );
    }

    let sparse = nights.len() - pairs.len();
    if sparse > 0 {
        println!("  ({sparse} night(s) too sparsely labelled to summarise, scored epoch-wise only)");
    }
}

fn main() {
    // One entry per arm. A new engine is a new SleepConfig here, never a change to the scoring above.
    let arms: [(&str, SleepConfig); 1] = [("v2 shipped recipe (NULL READING)", SleepConfig::shipped())];
    let p = Params::SHIPPED;

    println!("THE BORDER — what any engine is measured on. Minutes, except efficiency in percent.");
    println!("Positive bias = the engine over-reports against PSG. Nothing here is a gate.");
    for ds in COHORTS {
        if !common::root(ds).is_dir() {
            println!("\n{ds}: missing");
            continue;
        }
        for (arm, cfg) in &arms {
            card(ds, arm, &score(ds, cfg, &p));
        }
    }
}
