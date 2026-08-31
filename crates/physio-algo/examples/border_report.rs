//! THE BORDER: everything an engine must be measured on, for any engine.
//!
//!   cargo run --release -p physio-algo --example border_report
//!
//! Phase 0's deliverable. Not a report about the shipped recipe - a scoring card that takes a
//! [`SleepConfig`] and prints the same six blocks whatever produced the hypnogram. v2 appears once, as
//! a NULL READING - where the old engine happens to sit, never a target and never the subject.
//!
//! Seven blocks, each answering something the others cannot:
//!   KAPPA4      epoch-wise agreement over wake/light/deep/REM
//!   KAPPA3      the same staging as wake/NREM/REM, which is what most wearable papers report
//!   CI          percentile bootstrap over RECORDINGS - how much the figure would move on new nights
//!   RECALL      per class, because kappa is dominated by whichever stage holds the most epochs
//!   AGREEMENT   Bland-Altman on the minutes a user reads; kappa cannot say if the AMOUNT is right
//!   SLOPE       proportional bias - away from zero the error depends on the magnitude and one bias
//!               figure describes nobody
//!   STRUCTURE   bout lengths and transition rates, which every block above is blind to - a confusion
//!               matrix is unchanged by shuffling the hypnogram. TRUTH is printed in the same row, so
//!               a rate can only be called high against this cohort's own
//!
//! An arm is added by adding a `SleepConfig`, not by editing this file's scoring.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, require_psg};
use physio_algo::sleep::agreement::{bland_altman, summarise, NightSummary};
use physio_algo::sleep::metrics::{
    balanced_accuracy, bootstrap_kappa_ci, confusion4, f1, kappa3, kappa4,
    kappa_after_reassignment, kappa_class_bonus, merge3, min_recall, per_recording, precision,
    recall, truth_marginals, Confusion4, Spread,
};
use physio_algo::sleep::pipeline::{run, SleepConfig};
use physio_algo::sleep::sequence::{bout_w1, min_run_smooth, Structure};
use physio_algo::sleep::{epoch_starts_v2, params::Params, SleepInput};

const EPOCH: i64 = 30;
const EPOCH_MIN: f64 = 0.5;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASS_NAMES: [&str; 4] = ["wake", "light", "deep", "rem"];
/// A transition holding less than this share of the cohort's own adjacent pairs is rare. Derived
/// from the truth every run, never carried as a published figure.
const RARE_SHARE: f64 = 0.005;
/// The upper tail the transition diagonal is aimed at, in epochs.
const TAIL_EPOCHS: usize = 20;
/// Runs shorter than this are absorbed in the CONTROL arm, which exists to prove the bout numbers
/// move when bout structure moves.
const SMOOTH_MIN: usize = 6;
/// Labels must cover this much of a night's own span before it can carry a summary; a sparse night
/// reports a short TST that is a labelling gap, not a wearer awake.
const MIN_LABEL_DENSITY: f64 = 0.95;
const BOOTSTRAP_DRAWS: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0x5EED;

struct Night {
    cm: Confusion4,
    /// `None` when the night is too sparsely labelled to summarise.
    pair: Option<(NightSummary, NightSummary)>,
    /// Prediction and truth cut into time-CONTIGUOUS runs of epochs, cut at the same places. A hole
    /// in the labels ends a segment, so no transition is ever counted across one.
    segs: Vec<(Vec<usize>, Vec<usize>)>,
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
        let mut segs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut prev: Option<usize> = None;
        for (k, t) in &truth {
            let staged = (0..4).contains(t).then(|| at.get(&(w0 + *k as i64 * EPOCH))).flatten();
            let Some(lab) = staged else {
                // An unlabelled or unstaged epoch is a hole, and a hole ends the segment.
                prev = None;
                continue;
            };
            d.push(*lab);
            r.push(*t as usize);
            if prev != Some(k.wrapping_sub(1)) {
                segs.push((Vec::new(), Vec::new()));
            }
            let seg = segs.last_mut().expect("a segment was just started");
            seg.0.push(*lab);
            seg.1.push(*t as usize);
            prev = Some(*k);
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
            segs,
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

    // THE SELECTION LINE. Per-class recall conditions on truth, so a cohort's class balance cannot
    // move it; F1 and precision carry the predicted share and do move, which is why they are
    // reported beside it and never selected on.
    let ba = |c: &Confusion4| balanced_accuracy(c);
    let show = |s: Option<Spread>| {
        s.map_or("  -  ".into(), |s| format!("{:.3} +/-{:.3} (n={})", s.mean, s.sd, s.n))
    };
    println!(
        "  balanced acc {:.4}  min recall {:.4}   per-night {}",
        balanced_accuracy(&cm).unwrap_or(f64::NAN),
        min_recall(&cm).unwrap_or(f64::NAN),
        show(per_recording(&cms, ba))
    );

    let tot: i64 = cm.iter().flatten().sum();
    let fmt = |v: Option<f64>| v.map_or("  -  ".into(), |x| format!("{x:.3}"));
    println!(
        "  {:<7} {:>8} {:>22} {:>10} {:>7} {:>9} {:>9} {:>8}",
        "class", "recall", "recall per-night", "precision", "F1", "pred %", "truth %", "miss %"
    );
    let t = truth_marginals(&cm);
    for c in 0..4 {
        let q = cm.iter().map(|r| r[c]).sum::<i64>() as f64 / tot.max(1) as f64;
        // Share of ALL epochs truly this class and called something else. Recall ranks the classes
        // equally; this ranks them by the epochs they actually cost, and the two disagree.
        let miss = t[c] * (1.0 - recall(&cm, c).unwrap_or(0.0));
        println!(
            "  {:<7} {:>8} {:>22} {:>10} {:>7} {:>8.1}% {:>8.1}% {:>7.1}%",
            CLASS_NAMES[c],
            fmt(recall(&cm, c)),
            show(per_recording(&cms, |x| recall(x, c))),
            fmt(precision(&cm, c)),
            fmt(f1(&cm, c)),
            q * 100.0,
            t[c] * 100.0,
            miss * 100.0
        );
    }

    // A constant bias and a significant slope cannot both stand. Where the slope resolves, the bias
    // is a LINE and the flat +/-band is the wrong scale, so the sloped row prints the fitted bias at
    // each end of the observed range and limits about the line instead.
    // `track` is the regression coefficient of device on reference, 1 + slope_ref: 1.0 means the
    // reported minutes follow truth one for one, 0.0 means they are the same number whatever the
    // truth was. A bias of zero with track near zero is a stopped clock, and reads as accurate.
    println!(
        "  {:<11} {:>8} {:>8} {:>19} {:>8} {:>7} {:>6}",
        "measure", "bias", "sd|resid", "95% LoA", "slope", "t", "track"
    );
    for (name, get) in [
        ("TST", (|s: &NightSummary| s.tst) as fn(&NightSummary) -> f64),
        ("WASO", |s| s.waso),
        ("efficiency", |s| s.efficiency),
        ("deep", |s| s.deep),
        ("rem", |s| s.rem),
    ] {
        let dev: Vec<f64> = pairs.iter().map(|(d, _)| get(d)).collect();
        let refr: Vec<f64> = pairs.iter().map(|(_, r)| get(r)).collect();
        let Some(a) = bland_altman(&dev, &refr) else {
            println!("  {name:<11} too few pairs");
            continue;
        };
        if a.proportional {
            let (lo, hi) = a.loa_at(a.ref_hi);
            println!(
                "  {name:<11} {:>8} {:>8.1} {:>8.1} .. {:>6.1} {:>8.3} {:>7.1} {:>6.2}  SLOPED: bias {:+.1} at {:.0} to {:+.1} at {:.0}",
                "-- line", a.resid_sd, lo, hi, a.slope_ref, a.slope_t, 1.0 + a.slope_ref,
                a.bias_at(a.ref_lo), a.ref_lo, a.bias_at(a.ref_hi), a.ref_hi
            );
        } else {
            println!(
                "  {name:<11} {:>8.1} {:>8.1} {:>8.1} .. {:>6.1} {:>8.3} {:>7.1} {:>6.2}",
                a.bias, a.sd, a.loa_lo, a.loa_hi, a.slope_ref, a.slope_t, 1.0 + a.slope_ref
            );
        }
    }
    // D11: kappa is a ratio of linear forms, so its own optimal rule is not argmax of the posterior.
    let bonus = kappa_class_bonus(&cm);
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

    structure(nights);

    let sparse = nights.len() - pairs.len();
    if sparse > 0 {
        println!("  ({sparse} night(s) too sparsely labelled to summarise, scored epoch-wise only)");
    }
}

/// Bout lengths and transition rates, with TRUTH beside every one of them. Nothing above this line
/// can see any of it: a confusion matrix is invariant to shuffling the hypnogram.
fn structure(nights: &[Night]) {
    let (mut pred, mut truth, mut ctrl) =
        (Structure::default(), Structure::default(), Structure::default());
    for n in nights {
        for (p, t) in &n.segs {
            pred.add(p);
            truth.add(t);
            ctrl.add(&min_run_smooth(p, SMOOTH_MIN));
        }
    }
    let Some(fi_p) = pred.fi() else { return };
    let fi_t = truth.fi().unwrap_or(f64::NAN);

    let m = |e: Option<f64>| e.map_or("  -  ".into(), |x| format!("{:.1}", x * EPOCH_MIN));
    let f = |x: Option<f64>| x.map_or("  -  ".into(), |v| format!("{v:.3}"));
    println!(
        "  STRUCTURE  {} segment(s)   fragmentation {fi_p:.4} vs truth {fi_t:.4}  ({:+.1}%)",
        pred.segments,
        (fi_p / fi_t - 1.0) * 100.0
    );
    println!(
        "  {:<7} {:>8} {:>8} {:>10} {:>10} {:>7} {:>9} {:>9} {:>5}",
        "class", "FI", "FI true", "bout min", "true min", "W1 min", ">=10min", "true >=", "cens"
    );
    for (c, name) in CLASS_NAMES.iter().enumerate() {
        println!(
            "  {:<7} {:>8} {:>8} {:>10} {:>10} {:>7} {:>9} {:>9} {:>5}",
            name,
            f(pred.fi_class(c)),
            f(truth.fi_class(c)),
            m(pred.mean_bout(c)),
            m(truth.mean_bout(c)),
            m(bout_w1(&pred, &truth, c)),
            f(pred.tail_mass(c, TAIL_EPOCHS)),
            f(truth.tail_mass(c, TAIL_EPOCHS)),
            pred.censored[c] + truth.censored[c],
        );
    }

    // The rare set is this cohort's own truth, and the truth's own rate against it is the only thing
    // that says whether the prediction's rate is high. The per-cell counts are printed because a
    // pooled rate hides a transition the arm prices at the 1e-9 floor and therefore never emits.
    let rare = truth.rare_set(RARE_SHARE);
    let cells: Vec<String> = (0..4)
        .flat_map(|i| (0..4).map(move |j| (i, j)))
        .filter(|(i, j)| rare[*i][*j])
        .map(|(i, j)| {
            format!("{}>{} {}/{}", CLASS_NAMES[i], CLASS_NAMES[j], pred.trans[i][j], truth.trans[i][j])
        })
        .collect();
    println!(
        "  TVR pred {} vs truth {} over {} rare transition(s) under {:.1}% of this cohort's pairs",
        f(pred.tvr(&rare)),
        f(truth.tvr(&rare)),
        cells.len(),
        RARE_SHARE * 100.0,
    );
    println!("    pred/truth counts: {}", cells.join("   "));

    // The control: absorb every short run and the numbers above must move. If they do not, this
    // block is not measuring bout structure and nothing read off it means anything.
    let w1 = |s: &Structure| {
        (0..4).filter_map(|c| bout_w1(s, &truth, c)).map(|x| x * EPOCH_MIN).sum::<f64>()
    };
    println!(
        "  CONTROL runs <{SMOOTH_MIN} epochs absorbed: FI {:.4} (from {fi_p:.4}), summed W1 {:.1} min (from {:.1})",
        ctrl.fi().unwrap_or(f64::NAN),
        w1(&ctrl),
        w1(&pred)
    );
}

/// Mix every transition row toward uniform, leaving the emissions untouched. At 1.0 the decoder
/// follows the emissions alone, which is the only arm that separates "the prior is doing the
/// smoothing" from "the emissions cannot discriminate". `Params::SHIPPED` is never mutated.
fn flattened(alpha: f64) -> Params {
    let mut p = Params::SHIPPED;
    for row in p.transition.iter_mut() {
        for v in row.iter_mut() {
            *v = (1.0 - alpha) * *v + alpha * 0.25;
        }
    }
    p
}

/// Open the two structural zeros in the wake row at the rate the PSG truth shows, taking the mass
/// from wake's self-loop and leaving every other row alone. Rows are `[deep, rem, light, awake]`,
/// so row 3 is wake. The rates are the pooled truth counts over wake epochs, not fitted per cohort.
fn wake_row_opened() -> Params {
    let mut p = Params::SHIPPED;
    let (to_deep, to_rem) = (0.0005, 0.007);
    p.transition[3][0] = to_deep;
    p.transition[3][1] = to_rem;
    p.transition[3][3] -= to_deep + to_rem;
    p
}

fn main() {
    // One entry per arm. A new engine is a new SleepConfig here, never a change to the scoring above.
    let arms: [(&str, SleepConfig, Params); 4] = [
        ("v2 shipped recipe (NULL READING)", SleepConfig::shipped(), Params::SHIPPED),
        ("transition 50% toward uniform", SleepConfig::shipped(), flattened(0.5)),
        ("transition UNIFORM - emissions alone", SleepConfig::shipped(), flattened(1.0)),
        ("wake>rem and wake>deep opened at truth's rate", SleepConfig::shipped(), wake_row_opened()),
    ];

    println!("THE BORDER — what any engine is measured on. Minutes, except efficiency in percent.");
    println!("Positive bias = the engine over-reports against PSG. Nothing here is a gate.");
    for ds in COHORTS {
        if !common::root(ds).is_dir() {
            println!("\n{ds}: missing");
            continue;
        }
        for (arm, cfg, p) in &arms {
            card(ds, arm, &score(ds, cfg, p));
        }
    }
}
