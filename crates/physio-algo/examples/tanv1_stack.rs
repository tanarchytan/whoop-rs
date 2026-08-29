//! The whole stack against the whole shipped stack, at matched coverage.
//!
//!   cargo run --release -p physio-algo --example tanv1_stack
//!
//! Every lever so far was measured against the SHIPPED recipe on its own, which is the wrong
//! comparison: a lever is allowed to start behind if the stack ends ahead. And abstention helps the
//! shipped recipe too, so crediting it to tanv1 would be double counting.
//!
//! So both engines get the same treatment. Same Viterbi, same shipped transition, the same
//! refuse-nearest-your-own-transition rule at the same coverage, scored on the same COUNT of
//! epochs. The only difference is which emissions go in: the shipped ones, or the fitted ones.
//!
//! That isolation is PARTIAL. The shipped transition was set against the SHIPPED emission scale, and
//! the fitted arm hands the same matrix log-softmax logits instead, so the prior weighs differently
//! in each arm and part of any held-out gap is an unretuned transition rather than the emissions.
//! `edges` and `cut` report how many stage changes each arm decoded and where its abstention cut
//! landed, so a reader can see whether the two abstentions selected comparably.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::lr::{design, fit, scores, standardiser, NCOL};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
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
/// Fewest epochs a night must carry to load, and the fewest retained after abstention - a night
/// short enough to hit that floor is scored above the printed `keep`.
const MIN_EPOCHS: usize = 20;
/// The DECODED-arm exponent `fit_tanv1` selected on the fit cohort's inner split; fixed here so this
/// harness does not re-select it against the cohorts it reports. A repin can move that selection and
/// this literal does not follow, so `TANV1_WEIGHT_POWER` overrides it as it does there.
const WEIGHT_POWER: f64 = 0.5;

/// [`WEIGHT_POWER`] unless `TANV1_WEIGHT_POWER` is set, matching `fit_tanv1`'s own override.
fn weight_power() -> f64 {
    std::env::var("TANV1_WEIGHT_POWER").ok().and_then(|v| v.parse().ok()).unwrap_or(WEIGHT_POWER)
}

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
        // `emissions_v2` DROPS an epoch carrying neither HR nor gravity, so positional indexing into
        // `raw` is only valid while the grid is complete. Checked BEFORE the length skip, or the
        // hardest-collapsed grid is the one that leaves silently instead of tripping it.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        if em.len() < MIN_EPOCHS {
            continue;
        }
        // `extract` derives its own count from `[w0, w1)`. A grid shorter than the emission grid is
        // the same misalignment the assert above catches, so it fails here rather than dropping the
        // night out of both arms unannounced.
        assert!(f.len() >= em.len(), "{}: {} features against {} emissions",
                dir.display(), f.len(), em.len());
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

/// The epochs `pred` changes stage on, which is also what the abstention below ranks against.
fn edges_of(pred: &[usize]) -> Vec<usize> {
    (1..pred.len()).filter(|k| pred[*k] != pred[k - 1]).collect()
}

/// Distance in epochs to the nearest change in `pred`, which lies BETWEEN two epochs, so the pair
/// either side of it both score 0. A night the decoder never changes stage on scores every epoch
/// [`f64::INFINITY`], so the index tie-break keeps the EARLIEST epochs.
fn to_edge(pred: &[usize]) -> Vec<f64> {
    let edges = edges_of(pred);
    (0..pred.len())
        .map(|k| {
            let k = k as i64;
            edges
                .iter()
                .map(|e| (*e as i64 - k).abs().min((*e as i64 - 1 - k).abs()) as f64)
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// One arm per night: kappa on the kept epochs, the stage changes the decoder made, and the
/// [`to_edge`] distance its cut landed on. The last two are how a reader sees whether two arms on
/// different emission scales abstained comparably against the one shared transition.
struct Arm {
    kappa: Vec<f64>,
    edges: Vec<f64>,
    cut: Vec<f64>,
}

/// Kappa per night on the `keep` fraction furthest from the engine's OWN transitions, beside the
/// edges it decoded and the distance its cut landed on. Each engine abstains on its own boundaries;
/// the COUNT kept is identical, so the comparison is at matched coverage even though epochs differ.
fn scored(nights: &[Night], em_of: impl Fn(&Night) -> Vec<[f64; CLASSES]>, keep: f64) -> Arm {
    let mut out = Arm { kappa: Vec::new(), edges: Vec::new(), cut: Vec::new() };
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
        out.kappa.push(kappa4(&confusion4(&p, &t)));
        out.edges.push(edges_of(&pred).len() as f64);
        out.cut.push(dist[idx[take - 1]]);
    }
    out
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
    let power = weight_power();
    println!("FIT on {FIT}: {} nights, {} labelled epochs, weight power {power}\n",
             train.len(), x.len());
    let w = fit(&dx, &y, power);

    println!("Both engines: same decoder, same shipped transition, each abstaining on ITS OWN");
    println!("stage boundaries at the SAME coverage. Only the emissions differ - and that transition");
    println!("was set against the SHIPPED emission scale, not against log-softmax logits, so part of");
    println!("any gap is an unretuned transition. `edg` (stage changes decoded per night) and `cut`");
    println!("(the distance the abstention cut landed on) say whether the two arms selected alike.\n");
    println!("  {:<20} {:>5} {:>8} {:>8} {:>10} {:>10}   {:>10} {:>9} {:>5}   verdict",
             "cohort", "keep", "shipped", "tanv1", "edg sh/tan", "cut sh/tan", "paired d",
             "bar +/-", "n");

    for set in [FIT, "aauwss", "sleep-accel"] {
        let loaded = (set != FIT).then(|| load(set));
        let nights: &[Night] = loaded.as_deref().unwrap_or(train.as_slice());
        if nights.is_empty() {
            println!("  {set:<20} no nights");
            continue;
        }
        let role = if set == FIT { "(FITTED)" } else { "(HELD)" };
        for keep in COVERAGE {
            let mut base = scored(nights, |nt| nt.em.clone(), keep);
            let mut tan = scored(nights, |nt| fitted_emissions(nt, &w, &m, &sd), keep);
            let d: Vec<f64> = base.kappa.iter().zip(&tan.kappa).map(|(a, b)| b - a).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let verdict = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("tanv1 {} ({:.2}x)", if mean > 0.0 { "BETTER" } else { "WORSE" },
                        mean.abs() / bar)
            } else {
                "inside the bar".to_string()
            };
            let edg = format!("{:.0}/{:.0}", median(&mut base.edges), median(&mut tan.edges));
            let cut = format!("{:.1}/{:.1}", median(&mut base.cut), median(&mut tan.cut));
            let arms = format!("{:>8.3} {:>8.3} {edg:>10} {cut:>10}",
                               median(&mut base.kappa), median(&mut tan.kappa));
            println!("  {:<20} {:>4.0}% {arms}   {mean:>+10.4} {bar:>9.4} {:>5}   {verdict}",
                     if keep == COVERAGE[0] { format!("{set} {role}") } else { String::new() },
                     100.0 * keep, d.len());
        }
    }
    println!("\nA lever may start behind. The question is whether the STACK ends ahead, and the");
    println!("shipped stack gets the same abstention, so no gain is credited to tanv1 twice.");
}
