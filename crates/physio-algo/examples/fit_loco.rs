//! Leave one cohort out: fit on the union of the other two, report on the one held back.
//!
//!   cargo run --release -p physio-algo --example fit_loco
//!
//! Every negative so far shares one confound - fitted on DREAMT ALONE. That is not four independent
//! verdicts on the fitted emission, it is one configuration tested four ways, and the failure it
//! produced (leaning on `clock` and long-window rotation, the two columns that transfer worst) is
//! exactly what fitting one cohort rewards. A column can only be punished for failing to transfer if
//! the training set contains something to transfer TO.
//!
//! So: three folds, each fitting the union of two cohorts and reporting the third. 144 subjects
//! instead of 100, and transfer becomes the training signal rather than an afterthought.
//!
//! Two column sets, because the non-linearity arm already reached parity on one cohort:
//!   BASE  - the measured columns alone.
//!   +NL   - plus the shipped recipe's transforms, read off `emission_terms` so they cannot drift.
//!
//! And each is scored at three COVERAGES, because abstention is the other lever that worked and the
//! two have never been combined. Each engine refuses the epochs nearest ITS OWN decoded transitions,
//! so both give up the same number of epochs and the comparison stays matched.
//!
//! Every number is a PAIRED per-night difference against the shipped recipe on the same nights.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
/// Chosen inside a fit cohort by `fit_tanv1`'s own inner split, and fixed here so no fold selects
/// a hyperparameter against the cohort it is about to report.
const WEIGHT_POWER: f64 = 0.5;
const N_BASE: usize = Features::N;
/// The shipped transforms a linear model cannot invent: the deep-gate hinge, the deadzoned cardiac
/// pair, the centred rotation rank, the respiration z and the stillness-clamp indicator.
const N_NL: usize = 6;
/// Fractions of epochs KEPT. 1.00 is today's behaviour; below it each engine refuses its own
/// boundary epochs, which is the rule that beat the emission margin on both held-out cohorts.
const COVERAGE: [f64; 3] = [1.00, 0.90, 0.80];

struct Night {
    /// Base columns, then the shipped transforms.
    row: Vec<Vec<f64>>,
    /// The shipped recipe's own call per epoch, for the paired baseline.
    base: Vec<usize>,
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
        // Both grids are the whole window at 30 s, so equal lengths means equal epochs. Asserted
        // because `emissions_v2` DROPS an epoch carrying neither HR nor gravity, and a silent
        // offset would misalign every row after the hole.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        assert!(f.len() >= em.len(), "{}: fewer feature rows than emissions", dir.display());
        let deep = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 2).expect("deep");
        let awake = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 0).expect("wake");
        let row = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = f[e].values().to_vec();
                let nl: [f64; N_NL] = [
                    -d[deep][3],                              // the deep-gate hinge
                    d[awake][8],                              // deadzoned hr_var z
                    d[awake][9],                              // deadzoned hr z
                    d[awake][10],                             // centred rotation rank
                    d[deep][11],                              // respiration z
                    if terms.clamped[e] { 1.0 } else { 0.0 }, // the stillness clamp
                ];
                v.extend_from_slice(&nl);
                v
            })
            .collect();
        let base = decode_v2(&em, &Params::SHIPPED.transition)
            .iter()
            .map(|s| stage_idx(*s))
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { row, base, truth });
    }
    out
}

/// The columns one arm keeps.
fn cols(full: &[f64], with_nl: bool) -> Vec<f64> {
    if with_nl { full.to_vec() } else { full[..N_BASE].to_vec() }
}

/// Distance in epochs to the nearest change in `pred`. A path that never changes stage scores every
/// epoch infinite, and the index tie-break then keeps the earliest.
fn to_edge(pred: &[usize]) -> Vec<f64> {
    let edges: Vec<usize> = (1..pred.len()).filter(|k| pred[*k] != pred[k - 1]).collect();
    (0..pred.len())
        .map(|k| {
            edges.iter().map(|e| (*e as i64 - k as i64).abs() as f64).fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Kappa over the `keep` fraction of labelled epochs furthest from `pred`'s own transitions.
fn kappa_at(pred: &[usize], truth: &[Option<usize>], keep: f64) -> Option<f64> {
    let dist = to_edge(pred);
    let mut idx: Vec<usize> = (0..pred.len()).filter(|k| truth[*k].is_some()).collect();
    idx.sort_by(|a, b| dist[*b].partial_cmp(&dist[*a]).unwrap().then(a.cmp(b)));
    let take = ((idx.len() as f64 * keep).round() as usize).max(MIN_EPOCHS).min(idx.len());
    if take < MIN_EPOCHS {
        return None;
    }
    let (p, t): (Vec<usize>, Vec<usize>) =
        idx[..take].iter().map(|k| (pred[*k], truth[*k].unwrap())).unzip();
    Some(kappa4(&confusion4(&p, &t)))
}

/// Kappa per night for the shipped recipe, and for a fitted weight vector, on the same nights.
fn score(
    nights: &[Night],
    with_nl: bool,
    w: &[Vec<f64>],
    m: &[f64],
    sd: &[f64],
    keep: f64,
) -> (Vec<f64>, Vec<f64>) {
    let to_order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    let (mut bk, mut fk) = (Vec::new(), Vec::new());
    for nt in nights {
        let em: Vec<[f64; CLASSES]> = nt
            .row
            .iter()
            .map(|full| {
                let z = scores(w, &design_row(&cols(full, with_nl), m, sd, &[]));
                let mx = z.iter().cloned().fold(f64::MIN, f64::max);
                let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
                std::array::from_fn(|c| z[to_order[c]] - lse)
            })
            .collect();
        let path: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        // Each engine abstains on ITS OWN boundaries, which is what either would do in the product,
        // and both give up the same COUNT so the pairing stays matched.
        if let (Some(b), Some(f)) =
            (kappa_at(&nt.base, &nt.truth, keep), kappa_at(&path, &nt.truth, keep))
        {
            bk.push(b);
            fk.push(f);
        }
    }
    (bk, fk)
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 2 {
        println!("need at least two cohorts under the fixture root");
        return;
    }
    let total: usize = loaded.iter().map(|(_, n)| n.len()).sum();
    println!("Leave one cohort out. {total} nights across {} cohorts, so each fold fits the union",
             loaded.len());
    println!("of two and reports the third - transfer becomes the training signal rather than an");
    println!("afterthought. Every fit so far used ONE cohort, which is the confound this removes.\n");
    println!("  {:<10} {:>5} {:<18} {:>7} {:>7}   {:>10} {:>9} {:>4}   verdict",
             "arm", "keep", "held-out cohort", "shipped", "fitted", "paired d", "bar +/-", "n");

    for with_nl in [false, true] {
        let arm = if with_nl { "BASE +NL" } else { "BASE" };
        for (held, _) in &loaded {
            // Fit on every cohort EXCEPT the one being reported.
            let train: Vec<&Night> = loaded
                .iter()
                .filter(|(c, _)| c != held)
                .flat_map(|(_, n)| n.iter())
                .collect();
            let rows: Vec<(Vec<f64>, usize)> = train
                .iter()
                .flat_map(|nt| {
                    nt.row.iter().zip(&nt.truth).filter_map(|(r, t)| t.map(|t| (cols(r, with_nl), t)))
                })
                .collect();
            let x: Vec<Vec<f64>> = rows.iter().map(|(r, _)| r.clone()).collect();
            let y: Vec<usize> = rows.iter().map(|(_, t)| *t).collect();
            let (m, sd) = standardise_cols(&x);
            let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
            let w = fit(&dx, &y, WEIGHT_POWER);

            let nights = &loaded.iter().find(|(c, _)| c == held).expect("held cohort").1;
            for keep in COVERAGE {
                let (bk, fk) = score(nights, with_nl, &w, &m, &sd, keep);
                let d: Vec<f64> = bk.iter().zip(&fk).map(|(a, b)| b - a).collect();
                let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
                let v = if !mean.is_finite() {
                    "-".to_string()
                } else if mean.abs() > bar {
                    format!("{} ({:.2}x)", if mean > 0.0 { "BEATS SHIPPED" } else { "worse" },
                            mean.abs() / bar)
                } else {
                    "inside the bar".to_string()
                };
                println!("  {:<10} {:>4.0}% {:<18} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>4}   {v}",
                         arm, 100.0 * keep, format!("{held} n={}", nights.len()),
                         median(&mut bk.clone()), median(&mut fk.clone()), d.len());
            }
        }
        println!();
    }
    println!("A cohort held out of the FIT has never seen its own nights, so these are the first");
    println!("numbers here that are not about DREAMT.");
}
