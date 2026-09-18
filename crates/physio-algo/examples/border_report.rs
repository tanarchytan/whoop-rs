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

use std::collections::BTreeMap;

use common::screen::{PERM_SEED, RIDGE};
use common::{compare, dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, require_psg,
    Provenance};
use physio_algo::lda::Lda;
use physio_algo::sleep::agreement::{bland_altman, summarise, NightSummary};
use physio_algo::sleep::cardiac_emit::{self, CardiacEmit, COLS, FIT_ORDER};
use physio_algo::sleep::metrics::{
    balanced_accuracy, bootstrap_kappa_ci, confusion4, f1, kappa3, kappa4,
    kappa_after_reassignment, kappa_class_bonus, macro_f1, merge3, min_recall, pair_by_id,
    per_recording, precision,
    recall, truth_marginals, Confusion4, Spread,
};
use physio_algo::sleep::conditioned::ConditionedCfg;
use physio_algo::sleep::markov_loss::{self, Costs};
use physio_algo::sleep::pipeline::{run, CostsCfg, DecodeCfg, EmitCfg, SleepConfig};
use physio_algo::sleep::sequence::{bout_w1, min_run_smooth, Structure};
use physio_algo::sleep::{epoch_starts_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER};

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
/// At or below this many recordings a FITTED arm holds out one at a time; above it the fold-to-fold
/// spread is the smaller worry and the runtime is the larger, so it holds out a fifth.
const LORO_MAX_NIGHTS: usize = 40;
const FOLDS: usize = 5;

struct Night {
    /// The recording this scored, so two arms are paired by NIGHT and never by position: an arm
    /// whose config is `None` on a night drops it, and the arms then hold different rows.
    id: usize,
    cm: Confusion4,
    /// `None` when the night is too sparsely labelled to summarise.
    pair: Option<(NightSummary, NightSummary)>,
    /// Prediction and truth cut into time-CONTIGUOUS runs of epochs, cut at the same places. A hole
    /// in the labels ends a segment, so no transition is ever counted across one.
    segs: Vec<(Vec<usize>, Vec<usize>)>,
}

/// One recording, loaded once and scored under every arm.
struct Loaded {
    input: SleepInput,
    truth: BTreeMap<usize, i32>,
    train: TrainNight,
}

/// One recording's training rows: the fourteen z-scored cardiac columns from the LIBRARY producer,
/// each with the truth label of its own epoch. The emission reads the same rows from the same
/// function, so a fit is never against a quantity the engine does not see.
#[derive(Default)]
struct TrainNight {
    id: usize,
    rows: Vec<[f64; COLS]>,
    y: Vec<usize>,
    /// Prepared epochs whose window held too few beats to carry columns at all.
    missing: usize,
    epochs: usize,
    /// Labelled epochs per truth class, counted whether or not the epoch carries cardiac columns:
    /// the weighted decode needs a prevalence on a cohort that carries no R-R at all.
    truth_counts: [u64; 4],
}

/// How an arm gets its config: a function of the TRAINING recordings. A fixed arm ignores them; a
/// fitted one is refitted for every held-out recording and is never handed it.
type Fit = Box<dyn Fn(&[&TrainNight]) -> Option<SleepConfig>>;
struct Arm(Fit);

fn fixed(cfg: SleepConfig) -> Arm {
    Arm(Box::new(move |_| Some(cfg)))
}

/// Load a cohort once. Same order and same skips as the scoring loop, so a recording either carries
/// a training set and a card row or neither.
fn load(ds: &str) -> Vec<Loaded> {
    require_psg(ds);
    let mut out: Vec<Loaded> = Vec::new();
    for dir in &dirs_of(ds) {
        let truth = read_truth(dir);
        let Some((w0, w1, _)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if truth.is_empty() || accel.is_empty() {
            continue;
        }
        let input = SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let train = training_rows(out.len(), &input, &truth, w0);
        out.push(Loaded { input, truth, train });
    }
    out
}

/// The cardiac columns on v2's own epoch grid, kept where the epoch also carries a truth label.
fn training_rows(id: usize, input: &SleepInput, truth: &BTreeMap<usize, i32>, w0: i64) -> TrainNight {
    let prep = prepare_v2(input, &Params::SHIPPED);
    let cols = cardiac_emit::columns(input, &prep);
    let mut t = TrainNight { id, epochs: cols.len(), ..TrainNight::default() };
    for v in truth.values().filter(|v| (0..4).contains(*v)) {
        t.truth_counts[*v as usize] += 1;
    }
    for (s, c) in epoch_starts_v2(&prep).iter().zip(&cols) {
        let Some(row) = c else {
            t.missing += 1;
            continue;
        };
        let Ok(k) = usize::try_from((*s - w0) / EPOCH) else { continue };
        if let Some(v) = truth.get(&k).filter(|v| (0..4).contains(*v)) {
            t.rows.push(*row);
            t.y.push(*v as usize);
        }
    }
    t
}

/// One config per recording. A fitted arm's fold assignment lives here, and the training set for a
/// recording is asserted never to contain it.
fn configs(loaded: &[Loaded], arm: &Arm) -> Vec<Option<SleepConfig>> {
    let n = loaded.len();
    let folds = if n <= LORO_MAX_NIGHTS { n } else { FOLDS }.max(1);
    let mut cache: Vec<Option<Option<SleepConfig>>> = vec![None; folds];
    (0..n)
        .map(|i| {
            let g = i % folds;
            if cache[g].is_none() {
                let idx: Vec<usize> = (0..n).filter(|j| j % folds != g).collect();
                assert!(!idx.contains(&i), "recording {i} is in its own training set");
                assert!(idx.len() < n, "the training set must be a strict subset");
                let train: Vec<&TrainNight> = idx.iter().map(|j| &loaded[*j].train).collect();
                cache[g] = Some((arm.0)(&train));
            }
            cache[g].expect("the fold was just filled")
        })
        .collect()
}

/// Score one cohort under one arm's per-recording configs. Labels are aligned to truth BY TIME, not
/// by position: an epoch carrying neither HR nor gravity is dropped from the staging, so the
/// sequences can differ in length.
fn score(loaded: &[Loaded], cfgs: &[Option<SleepConfig>], p: &Params) -> Vec<Night> {
    let mut out = Vec::new();
    for (night, cfg) in loaded.iter().zip(cfgs) {
        let Some(cfg) = cfg else { continue };
        let (input, truth) = (&night.input, &night.truth);
        let w0 = input.start;
        let st = run(input, cfg, p);
        let (Some(stages), Some(prep)) = (st.stages.as_ref(), st.prepared.as_ref()) else { continue };

        let starts = epoch_starts_v2(prep);
        let mut at: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (s, lab) in starts.iter().zip(stages) {
            at.insert(*s, *lab as usize);
        }

        let (mut d, mut r) = (Vec::new(), Vec::new());
        let mut segs: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut prev: Option<usize> = None;
        for (k, t) in truth {
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
            id: night.train.id,
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

/// `Params::SHIPPED` was hand-tuned watching all three of these cohorts, so the null reading every
/// paired line below is measured against is an upper bound and not an opponent.
const V2_PROVENANCE: Provenance =
    Provenance::InSample("v2's 12 emission weights, transition, base rate and gates");

/// One arm's per-night macro F1, keyed by recording. The input [`pair_by_id`] needs; a night whose
/// confusion carries no class at all cannot answer and is absent rather than zero.
fn macro_f1_by_night(nights: &[Night]) -> BTreeMap<usize, f64> {
    nights.iter().filter_map(|n| Some((n.id, macro_f1(&n.cm)?))).collect()
}

/// Scores one arm, and returns its per-night macro F1 so the next arm can be paired against it.
/// `base` is the null reading's own series over the SAME cohort.
fn card(ds: &str, arm: &str, nights: &[Night], base: Option<&BTreeMap<usize, f64>>) -> BTreeMap<usize, f64> {
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
        "  balanced acc {:.4}  macro F1 {:.4}  min recall {:.4}   per-night {}",
        balanced_accuracy(&cm).unwrap_or(f64::NAN),
        macro_f1(&cm).unwrap_or(f64::NAN),
        min_recall(&cm).unwrap_or(f64::NAN),
        show(per_recording(&cms, ba))
    );

    // The only PAIRED line on the card. Everything above subtracts two pooled numbers produced by
    // two separate runs, which carries no bar; this differences the same nights under both arms.
    let mine = macro_f1_by_night(nights);
    let paired = base.map(|b| pair_by_id(b, &mine)).map_or_else(
        || "  (this arm IS the baseline)".to_string(),
        |(bv, av)| format!("  vs the null, paired on {} night(s): {}", bv.len(),
                           compare(&bv, &av, V2_PROVENANCE).2),
    );
    println!("  per-night macro F1 {}{paired}", show(per_recording(&cms, macro_f1)));

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

    structure(nights, &cm);

    let sparse = nights.len() - pairs.len();
    if sparse > 0 {
        println!("  ({sparse} night(s) too sparsely labelled to summarise, scored epoch-wise only)");
    }
    mine
}

/// Bout lengths and transition rates, with TRUTH beside every one of them. Nothing above this line
/// can see any of it: a confusion matrix is invariant to shuffling the hypnogram.
fn structure(nights: &[Night], cm: &Confusion4) {
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

    // The segments and the confusion matrix are built from the SAME epochs by two different paths.
    // A disagreement means the segment builder dropped or invented some, and then every number in
    // this block is describing a different night from every number above it.
    let cm_truth: [usize; 4] = std::array::from_fn(|c| cm[c].iter().sum::<i64>() as usize);
    let cm_pred: [usize; 4] = std::array::from_fn(|c| cm.iter().map(|r| r[c]).sum::<i64>() as usize);
    let seg_truth: [usize; 4] = std::array::from_fn(|c| truth.epochs(c));
    let seg_pred: [usize; 4] = std::array::from_fn(|c| pred.epochs(c));
    if seg_truth != cm_truth || seg_pred != cm_pred {
        println!("  !! SEGMENTS AND CONFUSION DISAGREE - every number below is unsafe");
        println!("     truth  seg {seg_truth:?}  vs cm {cm_truth:?}");
        println!("     pred   seg {seg_pred:?}  vs cm {cm_pred:?}");
    }

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

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Shuffle each column WITHIN one recording: distribution and per-night scaling survive, only the
/// alignment to stage is destroyed. Keyed on the recording's own id, so a fold cannot change the
/// draw a recording gets.
fn shuffled(t: &TrainNight) -> Vec<[f64; COLS]> {
    let mut rows = t.rows.clone();
    for k in 0..COLS {
        let mut r = PERM_SEED ^ splitmix(((t.id as u64) << 8) | k as u64);
        for i in (1..rows.len()).rev() {
            r = splitmix(r);
            let j = (r >> 11) as usize % (i + 1);
            let tmp = rows[i][k];
            rows[i][k] = rows[j][k];
            rows[j][k] = tmp;
        }
    }
    rows
}

/// Fit the cardiac emission on the TRAINING recordings only. `permute` is the falsifier: the same
/// rows with each column shuffled within its own recording, so the fit sees the family's
/// distribution and none of its alignment to stage.
fn cardiac_fit(train: &[&TrainNight], lambda_milli: u32, permute: bool) -> Option<SleepConfig> {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for t in train {
        let rows = if permute { shuffled(t) } else { t.rows.clone() };
        for (r, c) in rows.iter().zip(&t.y) {
            x.push(r.to_vec());
            y.push(*c);
        }
    }
    let lda = Lda::fit(&x, &y, RIDGE)?;
    let emit = EmitCfg::V2PlusCardiac(CardiacEmit::from_lda(&lda, lambda_milli)?);
    Some(SleepConfig { emit, decode: DecodeCfg::Viterbi })
}

fn cardiac_arm(lambda_milli: u32, permute: bool) -> Arm {
    Arm(Box::new(move |t| cardiac_fit(t, lambda_milli, permute)))
}

/// The decoder reads `fc` in `STAGE_ORDER`; truth labels are counted in `FIT_ORDER`. The re-index
/// lives in the library because nothing in an example is run by `cargo test`.
fn fc_in_stage_order(by_truth: [f64; 4]) -> [f64; 4] {
    markov_loss::reindex(by_truth, FIT_ORDER, STAGE_ORDER).expect("both orders hold every stage")
}

/// A seed that IS the fold: these ids are the fold's own training set, whichever recordings it
/// holds out, so the draw moves with the fold and with nothing else.
fn fold_seed(train: &[&TrainNight]) -> u64 {
    train.iter().fold(PERM_SEED, |h, t| splitmix(h ^ (t.id as u64).wrapping_add(1)))
}

/// The same four weights on different classes: the magnitudes survive, which class holds which does
/// not. A constant vector comes back unmoved - then the arm and its control are one arm, which is
/// the honest reading rather than a manufactured difference.
fn permuted_weights(fc: [f64; 4], seed: u64) -> [f64; 4] {
    if fc.iter().all(|v| *v == fc[0]) {
        return fc;
    }
    let mut r = seed;
    for _ in 0..16 {
        let mut idx = [0usize, 1, 2, 3];
        for i in (1..4).rev() {
            r = splitmix(r);
            idx.swap(i, (r >> 11) as usize % (i + 1));
        }
        let out: [f64; 4] = core::array::from_fn(|c| fc[idx[c]]);
        if out != fc {
            return out;
        }
    }
    // A non-constant vector is always moved by a rotation, so this cannot return the input.
    core::array::from_fn(|c| fc[(c + 1) % 4])
}

/// Inverse-prevalence per-class epoch costs from the TRAINING recordings' own class counts, raised
/// to `power` (0 is unit costs) and mapped into the decoder's columns. Nothing is fitted beyond
/// counting. `permute` is the falsifier: the same magnitudes on shuffled classes.
fn prevalence_costs(train: &[&TrainNight], power: f64, permute: bool) -> Option<Costs> {
    let mut counts = [0u64; 4];
    for t in train {
        for (c, n) in counts.iter_mut().zip(&t.truth_counts) {
            *c += n;
        }
    }
    let w = markov_loss::geometric_scale(markov_loss::inverse_prevalence(counts)?, power)?;
    let mut fc = fc_in_stage_order(w);
    if permute {
        fc = permuted_weights(fc, fold_seed(train));
    }
    Some(Costs { fc, ..Costs::UNIT })
}

/// The weighted decode on v2's own emissions - the class weighting moved into the DECODE rather
/// than the emission. `ft` and `fh` price a PAIR of epochs, so this per-epoch rule charges `fc`
/// alone and choosing the full three-cost `r` stays open.
fn loss_arm(power: f64, permute: bool) -> Arm {
    Arm(Box::new(move |t| {
        let costs = prevalence_costs(t, power, permute)?;
        Some(SleepConfig { emit: EmitCfg::V2, decode: DecodeCfg::MarkovLoss(CostsCfg(costs)) })
    }))
}

/// Both at once: the cardiac emission decoded under inverse-prevalence `fc`, each fitted on the
/// same held-out training set. The arm that asks whether the weighted decode repairs what the
/// emission's calling share costs.
fn cardiac_loss_arm(lambda_milli: u32, power: f64) -> Arm {
    Arm(Box::new(move |t| {
        let base = cardiac_fit(t, lambda_milli, false)?;
        let costs = prevalence_costs(t, power, false)?;
        Some(SleepConfig { emit: base.emit, decode: DecodeCfg::MarkovLoss(CostsCfg(costs)) })
    }))
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

/// Silence the time term. `cycle_prior` is the only reader of `Features::clock`, so zeroing its two
/// scales and the early-REM step leaves the emission with no time-of-night contribution at all. The
/// library pins that (`golden_tests::zeroing_the_cycle_scales_leaves_no_time_dependent_emission_term`).
fn no_time_term() -> Params {
    Params {
        cycle_deep_scale: 0.0,
        cycle_rem_scale: 0.0,
        cycle_rem_early_penalty: 0.0,
        ..Params::SHIPPED
    }
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
    // The conditioned arms are CONFIGS. Nothing below the arm table changes for them, which is
    // the seam's own test: an arm that needed the scoring edited would mean the seam does not work.
    let conditioned = |beta_milli: u32| SleepConfig {
        emit: EmitCfg::V2,
        decode: DecodeCfg::Conditioned(ConditionedCfg { beta_milli }),
    };
    let marginal = SleepConfig { emit: EmitCfg::V2, decode: DecodeCfg::PosteriorMarginal };
    let unit_loss =
        SleepConfig { emit: EmitCfg::V2, decode: DecodeCfg::MarkovLoss(CostsCfg(Costs::UNIT)) };
    // The cardiac arms are FITTED, held out by recording. Their permuted twins fit on the same
    // rows with the column-to-stage alignment destroyed: if the real arm's gain does not exceed
    // theirs, what moved was the fit's freedom and not the family.
    let arms: Vec<(&str, Arm, Params)> = vec![
        ("v2 shipped recipe (NULL READING)", fixed(SleepConfig::shipped()), Params::SHIPPED),
        ("transition 50% toward uniform", fixed(SleepConfig::shipped()), flattened(0.5)),
        ("transition UNIFORM - emissions alone", fixed(SleepConfig::shipped()), flattened(1.0)),
        ("wake>rem and wake>deep opened at truth's rate", fixed(SleepConfig::shipped()), wake_row_opened()),
        // The only time-term arm this corpus supports: every fixture night is rebased onto ONE
        // synthetic start, so a wall-clock or habitual-midpoint anchor carries no between-recording
        // variation and is not measurable here. Ablation only.
        ("no time term (cycle prior off)", fixed(SleepConfig::shipped()), no_time_term()),
        ("conditioned diagonal, beta 0.25", fixed(conditioned(250)), Params::SHIPPED),
        ("conditioned diagonal, beta 0.5", fixed(conditioned(500)), Params::SHIPPED),
        ("conditioned diagonal, beta 1.0", fixed(conditioned(1000)), Params::SHIPPED),
        ("cardiac emission, lambda 0.5", cardiac_arm(500, false), Params::SHIPPED),
        ("cardiac emission, lambda 1.0", cardiac_arm(1000, false), Params::SHIPPED),
        ("cardiac emission, lambda 2.0", cardiac_arm(2000, false), Params::SHIPPED),
        ("cardiac PERMUTED null, lambda 0.5", cardiac_arm(500, true), Params::SHIPPED),
        ("cardiac PERMUTED null, lambda 1.0", cardiac_arm(1000, true), Params::SHIPPED),
        ("cardiac PERMUTED null, lambda 2.0", cardiac_arm(2000, true), Params::SHIPPED),
        // The decode seam. `PosteriorMarginal` and `MarkovLoss` at unit costs are ONE rule and must
        // print the same card; the weighted arms move the class weighting into the decode, which is
        // the rule a class-balanced objective implies.
        ("posterior-marginal decode", fixed(marginal), Params::SHIPPED),
        ("loss-matched decode, UNIT costs", fixed(unit_loss), Params::SHIPPED),
        ("loss-matched decode, fc^0.5 inverse prevalence", loss_arm(0.5, false), Params::SHIPPED),
        ("loss-matched decode, inverse-prevalence fc", loss_arm(1.0, false), Params::SHIPPED),
        ("loss-matched decode, PERMUTED fc", loss_arm(1.0, true), Params::SHIPPED),
        ("cardiac lambda 1.0 + inverse-prevalence fc", cardiac_loss_arm(1000, 1.0), Params::SHIPPED),
    ];

    assert!(arms[0].0.contains("NULL READING"), "the first arm is the baseline every paired line differences against");

    println!("THE BORDER — what any engine is measured on. Minutes, except efficiency in percent.");
    println!("Positive bias = the engine over-reports against PSG. Nothing here is a gate.");
    for ds in COHORTS {
        if !common::root(ds).is_dir() {
            println!("\n{ds}: missing");
            continue;
        }
        let loaded = load(ds);
        let (miss, eps) = loaded
            .iter()
            .fold((0usize, 0usize), |a, l| (a.0 + l.train.missing, a.1 + l.train.epochs));
        let rows: usize = loaded.iter().map(|l| l.train.rows.len()).sum();
        println!(
            "
{ds}: {} recording(s); {miss} of {eps} prepared epochs ({:.1}%) carry no cardiac columns; {rows} labelled rows to fit on",
            loaded.len(),
            miss as f64 / eps.max(1) as f64 * 100.0
        );
        // For reading only. Every fitted arm refits this on its own fold's training recordings.
        let all: Vec<&TrainNight> = loaded.iter().map(|l| &l.train).collect();
        if let Some(c) = prevalence_costs(&all, 1.0, false) {
            println!(
                "  inverse-prevalence fc over ALL recordings, STAGE_ORDER [deep rem light wake], geometric mean 1: {:.3} {:.3} {:.3} {:.3}",
                c.fc[0], c.fc[1], c.fc[2], c.fc[3]
            );
        }
        // The first arm is the null reading, and it is what every later arm's paired line is
        // differenced against, on this cohort's own nights.
        let mut base: Option<BTreeMap<usize, f64>> = None;
        for (arm, a, p) in &arms {
            let f1 = card(ds, arm, &score(&loaded, &configs(&loaded, a), p), base.as_ref());
            base.get_or_insert(f1);
        }
    }
}
