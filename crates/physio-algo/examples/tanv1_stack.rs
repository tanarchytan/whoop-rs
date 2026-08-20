//! The whole stack against the whole shipped stack, at matched coverage.
//!
//!   cargo run --release -p physio-algo --example tanv1_stack
//!
//! Every lever so far was measured against the SHIPPED recipe on its own, which is the wrong
//! comparison: a lever is allowed to start behind if the stack ends ahead. And abstention helps the
//! shipped recipe too, so crediting it to tanv1 would be double counting.
//!
//! So both engines get the same treatment. Same Viterbi, same shipped transition, the same
//! refuse-nearest-your-own-transition rule at the same coverage, scored on the same epochs. The only
//! difference is which emissions go in: the shipped ones, or the fitted ones.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::lr::{design, fit, predict, scores, standardiser, NCOL};
use common::{
    cardiac_series, dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const FIT: &str = "dreamt";
const CLASSES: usize = 4;
const COVERAGE: [f64; 5] = [1.00, 0.95, 0.90, 0.80, 0.70];
const MIN_EPOCHS: usize = 20;
/// Chosen inside the fit cohort by `fit_tanv1`'s own inner split; fixed here so this harness does
/// not re-select it against the cohorts it reports.
const WEIGHT_POWER: f64 = 0.5;

struct Night {
    x: Vec<[f64; NCOL]>,
    /// Shipped emissions, [`STAGE_ORDER`] columns.
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
        let hr = read_hr(dir);
        let rr = read_rr(dir);
        let card = cardiac_series(w0, n, EPOCH, &hr, &rr);
        let f = extract(&accel, w0, w1, &card);
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS || f.len() < em.len() {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { x: f[..em.len()].iter().map(|r| r.values()).collect(), em, truth });
    }
    out
}

/// The fitted model's emissions for one night, log-softmax in [`STAGE_ORDER`] columns so the same
/// decoder and the same transition matrix apply unchanged.
fn fitted_emissions(
    nt: &Night,
    w: &[Vec<f64>],
    m: &[f64; NCOL],
    sd: &[f64; NCOL],
) -> Vec<[f64; CLASSES]> {
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

/// Distance in epochs to the nearest change in `pred`.
fn to_edge(pred: &[usize]) -> Vec<f64> {
    let edges: Vec<usize> = (1..pred.len()).filter(|k| pred[*k] != pred[k - 1]).collect();
    (0..pred.len())
        .map(|k| {
            edges.iter().map(|e| (*e as i64 - k as i64).abs() as f64).fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Kappa per night on the `keep` fraction furthest from the engine's OWN transitions.
///
/// Each engine abstains on its own boundaries, which is what either would do in the product. The
/// COUNT kept is identical, so the comparison is at matched coverage even though the epochs differ.
fn scored(nights: &[Night], em_of: impl Fn(&Night) -> Vec<[f64; CLASSES]>, keep: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for nt in nights {
        let em = em_of(nt);
        let pred: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let dist = to_edge(&pred);
        let mut idx: Vec<usize> = (0..pred.len()).filter(|k| nt.truth[*k].is_some()).collect();
        idx.sort_by(|a, b| dist[*b].partial_cmp(&dist[*a]).unwrap().then(a.cmp(b)));
        let take = ((idx.len() as f64 * keep).round() as usize).max(MIN_EPOCHS).min(idx.len());
        if take < MIN_EPOCHS {
            continue;
        }
        let (p, t): (Vec<usize>, Vec<usize>) =
            idx[..take].iter().map(|k| (pred[*k], nt.truth[*k].unwrap())).unzip();
        out.push(kappa4(&confusion4(&p, &t)));
    }
    out
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
    // Labelled rows only: an unlabelled epoch is decode context, never training data.
    let rows: Vec<([f64; NCOL], usize)> = train
        .iter()
        .flat_map(|nt| nt.x.iter().zip(&nt.truth).filter_map(|(r, t)| t.map(|t| (*r, t))))
        .collect();
    let x: Vec<[f64; NCOL]> = rows.iter().map(|(r, _)| *r).collect();
    let y: Vec<usize> = rows.iter().map(|(_, t)| *t).collect();
    let (m, sd) = standardiser(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design(r, &m, &sd, &[])).collect();
    println!("FIT on {FIT}: {} nights, {} labelled epochs, weight power {WEIGHT_POWER}\n",
             train.len(), x.len());
    let w = fit(&dx, &y, WEIGHT_POWER);
    let _ = predict(&w, &dx[0]);

    println!("Both engines: same decoder, same shipped transition, each abstaining on ITS OWN");
    println!("stage boundaries at the SAME coverage. Only the emissions differ.\n");
    println!("  {:<20} {:>5} {:>8} {:>8}   {:>10} {:>9} {:>5}   verdict",
             "cohort", "keep", "shipped", "tanv1", "paired d", "bar +/-", "n");

    for set in [FIT, "aauwss", "sleep-accel"] {
        let nights = if set == FIT { train.clone_ref() } else { load(set) };
        if nights.is_empty() {
            println!("  {set:<20} no nights");
            continue;
        }
        let role = if set == FIT { "(FITTED)" } else { "(HELD)" };
        for keep in COVERAGE {
            let base = scored(&nights, |nt| nt.em.clone(), keep);
            let tan = scored(&nights, |nt| fitted_emissions(nt, &w, &m, &sd), keep);
            let d: Vec<f64> = base.iter().zip(&tan).map(|(a, b)| b - a).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let verdict = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("tanv1 {} ({:.2}x)", if mean > 0.0 { "BETTER" } else { "WORSE" },
                        mean.abs() / bar)
            } else {
                "inside the bar".to_string()
            };
            println!("  {:<20} {:>4.0}% {:>8.3} {:>8.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {verdict}",
                     if keep == COVERAGE[0] { format!("{set} {role}") } else { String::new() },
                     100.0 * keep, median(&base), median(&tan), d.len());
        }
    }
    println!("\nA lever may start behind. The question is whether the STACK ends ahead, and the");
    println!("shipped stack gets the same abstention, so no gain is credited to tanv1 twice.");
}

/// Cheap re-borrow so the fit cohort is not loaded twice.
trait CloneRef {
    fn clone_ref(&self) -> Vec<Night>;
}
impl CloneRef for Vec<Night> {
    fn clone_ref(&self) -> Vec<Night> {
        self.iter()
            .map(|n| Night { x: n.x.clone(), em: n.em.clone(), truth: n.truth.clone() })
            .collect()
    }
}
