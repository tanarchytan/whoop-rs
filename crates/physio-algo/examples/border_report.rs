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

use common::mesa::{self, MesaNight};
use common::screen::{EPOCH_S as MESA_EPOCH_S, PERM_SEED, RIDGE, WINDOW_S};
use common::refit::{self, FitNight, Fitted};
use common::{compare_guarded, dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, require_psg,
    stage_idx, Provenance};
use physio_algo::lda::Lda;
use physio_algo::sleep::agreement::{bland_altman, summarise, NightSummary};
use physio_algo::sleep::cardiac_emit::{self, CardiacEmit, COLS, FIT_ORDER};
use physio_algo::sleep::metrics::{
    balanced_accuracy, bootstrap_kappa_ci, confusion4, f1, kappa3, kappa4,
    kappa_after_reassignment, kappa_class_bonus, macro_f1, merge3, min_recall, pair_by_id,
    per_recording, precision,
    recall, truth_marginals, Confusion4, Spread,
};
use physio_algo::sleep::conditioned::{self, ConditionedCfg};
use physio_algo::sleep::markov_loss::{self, Costs};
use physio_algo::sleep::tanv1::BASE;
use physio_algo::sleep::pipeline::{run, CostsCfg, DecodeCfg, EmitCfg, SleepConfig};
use physio_algo::sleep::sequence::{bout_w1, min_run_smooth, Structure};
use physio_algo::sleep::{
    cardiac, decode_v2, emission_terms, epoch_starts_v2, params::Params, prepare_v2, weights_of, Prepared,
    SleepInput, SleepStage, Terms, STAGE_ORDER, WEIGHT_NAMES,
};

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
    /// Index into `COHORTS`. `id` is only unique WITHIN a cohort, so a union training set is
    /// checked on the pair, never on the id alone.
    cohort: usize,
    rows: Vec<[f64; COLS]>,
    y: Vec<usize>,
    /// Prepared epochs whose window held too few beats to carry columns at all.
    missing: usize,
    epochs: usize,
    /// Labelled epochs per truth class, counted whether or not the epoch carries cardiac columns:
    /// the weighted decode needs a prevalence on a cohort that carries no R-R at all.
    truth_counts: [u64; 4],
    /// v2's emission decomposition under SHIPPED, read only by the refit arms.
    refit: RefitData,
    /// The same decomposition under `tanv1::BASE` (clamp only without R-R), read by the clamp-on refit arm.
    refit_base: RefitData,
}

impl TrainNight {
    /// The decomposition a refit arm reads: SHIPPED's, or tanv1::BASE's when the arm is clamp-on.
    fn rd(&self, base: bool) -> &RefitData {
        if base { &self.refit_base } else { &self.refit }
    }
}

/// One recording on v2's prepared-epoch grid, index-for-index with `Terms`.
#[derive(Default)]
struct RefitData {
    terms: Option<Terms>,
    starts: Vec<i64>,
    /// Truth class per prepared epoch, `None` where the epoch carries no label.
    truth: Vec<Option<usize>>,
    /// Per-night z of hr_var and hr with no deadzone and no clamp.
    zhv: Vec<f64>,
    zhr: Vec<f64>,
    /// The awake-cardiac clamp flag: still and not R-R backed.
    still: Vec<bool>,
    /// v2's own `rr_backed` notion: the epoch's window gives a respiratory regularity.
    rr_backed: Vec<bool>,
}

/// How an arm gets its config: a function of the TRAINING recordings. A fixed arm ignores them; a
/// fitted one is refitted for every held-out recording and is never handed it.
type Fit = Box<dyn Fn(&[&TrainNight]) -> Option<SleepConfig>>;
struct Arm(Fit);

/// One row of the arm table: label, config, params, and the LABEL of the arm this one is built on.
/// `None` is measured against the null alone, which is what a standalone arm wants.
/// The sixth field is a second arm to pair against, printed after the card, when the question is
/// asked of two bases at once.
struct Row(&'static str, Engine, Params, Option<&'static str>, TrainSet, Option<&'static str>);

impl Row {
    fn also(mut self, other: &'static str) -> Self {
        self.5 = Some(other);
        self
    }
}

/// What produces an arm's labels: a library `SleepConfig`, or v2's emission with weights refitted here.
enum Engine {
    Lib(Arm),
    Refit(Refit),
    /// P1: an ORACLE. Each night decodes under a base rate tilted by its own PSG stage shares.
    PerNight(PerNight),
    /// P4: the hr_var rank gate on the AWAKE row, or its count-matched random twin.
    HrvGate(HrvGate),
    /// U7 / P5: a transition change on `tanv1::BASE`'s emission. A time-varying LIGHT->DEEP entry, a looser
    /// AWAKE self-loop, or both, optionally with the P4 gate on the emission.
    Trans(Trans),
    /// P6: the wake self-loop and a wake base-rate multiplier together, through the full engine's `Params`.
    WakeLoop(WakeLoop),
}

/// Which classes the oracle tilts (indexed in `STAGE_ORDER`: deep, rem, light, wake), the tilt power,
/// and whether the night's shares are swapped for another night's (the alignment-destroyed twin).
#[derive(Clone, Copy)]
struct PerNight {
    classes: [bool; 4],
    power: f64,
    permute: bool,
}

/// Columns added straight into the AWAKE row: no deadzone, no clamp, present on every epoch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extra {
    /// v2's twelve weights refitted, nothing added.
    None,
    /// Raw z(hr_var), z(hr).
    Raw,
    /// `Raw`, plus each times the still (clamp) flag.
    RawStill,
    /// `Raw`, zeroed on every epoch that is not R-R backed: the fitted analogue of `clamp_only_without_rr`.
    RawGated,
}

impl Extra {
    const NAMES: [&'static str; 4] = ["x_hr_var", "x_hr", "x_hr_var*still", "x_hr*still"];
    fn k(self) -> usize {
        match self {
            Extra::None => 0,
            Extra::Raw | Extra::RawGated => 2,
            Extra::RawStill => 4,
        }
    }
}

#[derive(Clone, Copy)]
struct Refit {
    extra: Extra,
    /// Train on the extra columns shuffled within each night; scoring still reads the real ones.
    permute: bool,
    /// Weight every training epoch so each cohort in the fold contributes equal total weight.
    balanced: bool,
    /// Design, fixed terms and clamp flags computed under `tanv1::BASE` (clamp only without R-R) instead of SHIPPED.
    base: bool,
}

/// Which recordings a fitted arm trains on, for a held-out recording of cohort C.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrainSet {
    /// The other recordings of C. Every arm on the card before the union arms.
    WithinCohort,
    /// The other recordings of C plus EVERY recording of the other cohorts. The single-cohort fit
    /// was the confound in the August tanv1 comparison: a column can only be punished for failing to
    /// transfer if the training set holds something to transfer to.
    Union,
}

/// A standalone arm.
fn arm(name: &'static str, a: Arm, p: Params) -> Row {
    Row(name, Engine::Lib(a), p, None, TrainSet::WithinCohort, None)
}

/// A standalone arm fitted on the union of all cohorts.
fn union_arm(name: &'static str, a: Arm, p: Params) -> Row {
    Row(name, Engine::Lib(a), p, None, TrainSet::Union, None)
}

/// An arm built on an earlier one, which it is paired against as well as against the null.
fn rung(name: &'static str, a: Arm, p: Params, on: &'static str) -> Row {
    Row(name, Engine::Lib(a), p, Some(on), TrainSet::WithinCohort, None)
}

/// An arm fitted on the union of all cohorts, paired against the arm it is built on.
fn union_rung(name: &'static str, a: Arm, p: Params, on: &'static str) -> Row {
    Row(name, Engine::Lib(a), p, Some(on), TrainSet::Union, None)
}

/// v2's emission refitted on the union of all cohorts, paired against `on` when given.
fn refit_arm(name: &'static str, extra: Extra, permute: bool, balanced: bool, on: Option<&'static str>) -> Row {
    Row(name, Engine::Refit(Refit { extra, permute, balanced, base: false }), Params::SHIPPED, on, TrainSet::Union, None)
}

/// As [`refit_arm`], but v2's decomposition is computed under `tanv1::BASE`.
fn refit_base_arm(name: &'static str, extra: Extra, balanced: bool, on: Option<&'static str>) -> Row {
    Row(name, Engine::Refit(Refit { extra, permute: false, balanced, base: true }), BASE, on, TrainSet::Union, None)
}

/// P1 oracle row, within-cohort reference, always paired against `on`. Every paired line it prints
/// carries the ORACLE in-sample marker: it reads the night's own truth and can never be a candidate.
fn per_night_arm(name: &'static str, spec: PerNight, on: &'static str) -> Row {
    Row(name, Engine::PerNight(spec), BASE, Some(on), TrainSet::WithinCohort, None)
}

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
    t.refit = refit_data(&prep, input, truth, w0, &Params::SHIPPED);
    t.refit_base = refit_data(&prep, input, truth, w0, &BASE);
    assert_eq!(t.refit.starts, t.refit_base.starts, "the clamp flag must not move the epoch grid");
    t
}

/// v2's decomposition plus the raw hr_var and hr z, which `Terms` holds only deadzoned in the AWAKE
/// row. The DEEP row carries them raw; the asserts pin that reading against the REM and AWAKE rows.
fn refit_data(prep: &Prepared, input: &SleepInput, truth: &BTreeMap<usize, i32>, w0: i64, p: &Params) -> RefitData {
    let terms = emission_terms(prep, p);
    let starts = epoch_starts_v2(prep);
    let pos = |s: SleepStage| STAGE_ORDER.iter().position(|x| *x == s).expect("stage in STAGE_ORDER");
    let (deep, rem, awake) = (pos(SleepStage::Deep), pos(SleepStage::Rem), pos(SleepStage::Wake));
    let (s_dhv, s_dhr) = (refit::slot("deep_hrv"), refit::slot("deep_hr"));
    let (s_rhv, s_ahv, s_ahr) = (refit::slot("rem_hrv"), refit::slot("awake_hrv"), refit::slot("awake_hr"));
    let dz = |z: f64| {
        let dz = p.awake_deadzone;
        if z > dz { z - dz } else if z < -dz { z + dz } else { 0.0 }
    };
    let (mut zhv, mut zhr) = (Vec::new(), Vec::new());
    for d in &terms.design {
        let (hv, hr) = (d[deep][s_dhv], d[deep][s_dhr]);
        assert_eq!(hv, d[rem][s_rhv], "the DEEP and REM rows disagree on the hr_var z");
        assert!((dz(hv) - d[awake][s_ahv]).abs() < 1e-12, "AWAKE hr_var is not the deadzoned DEEP-row z");
        assert!((dz(hr) - d[awake][s_ahr]).abs() < 1e-12, "AWAKE hr is not the deadzoned DEEP-row z");
        zhv.push(hv);
        zhr.push(hr);
    }
    let truth = starts
        .iter()
        .map(|s| {
            let k = usize::try_from((*s - w0) / EPOCH).ok()?;
            truth.get(&k).filter(|v| (0..4).contains(*v)).map(|v| *v as usize)
        })
        .collect();
    let rr_backed = refit::rr_backed(input, &starts);
    RefitData { still: terms.clamped.clone(), terms: Some(terms), starts, truth, zhv, zhr, rr_backed }
}

/// One config per recording of cohort `ci`. A fitted arm's fold assignment lives here, and the
/// training set for a recording is asserted never to contain it. `all` holds every cohort, loaded
/// before any fold is built, so a `Union` arm can train across them.
fn configs(all: &[Vec<Loaded>], ci: usize, arm: &Arm, set: TrainSet) -> Vec<Option<SleepConfig>> {
    per_fold(all, ci, &*arm.0, set)
}

/// Folds of a cohort of `n` recordings.
fn fold_count(n: usize) -> usize {
    if n <= LORO_MAX_NIGHTS { n } else { FOLDS }.max(1)
}

/// One fit per recording of cohort `ci`, for any fitted thing (a config, refitted weights).
fn per_fold<T: Clone>(
    all: &[Vec<Loaded>],
    ci: usize,
    fit: &dyn Fn(&[&TrainNight]) -> T,
    set: TrainSet,
) -> Vec<T> {
    let loaded = &all[ci];
    let n = loaded.len();
    let folds = fold_count(n);
    let mut cache: Vec<Option<T>> = vec![None; folds];
    (0..n)
        .map(|i| {
            let g = i % folds;
            if cache[g].is_none() {
                let idx: Vec<usize> = (0..n).filter(|j| j % folds != g).collect();
                assert!(!idx.contains(&i), "recording {i} is in its own training set");
                assert!(idx.len() < n, "the training set must be a strict subset");
                let mut train: Vec<&TrainNight> = idx.iter().map(|j| &loaded[*j].train).collect();
                if set == TrainSet::Union {
                    train.extend(
                        all.iter().enumerate().filter(|(c, _)| *c != ci).flat_map(|(_, l)| l.iter().map(|x| &x.train)),
                    );
                    // On (cohort, id), not position: ids repeat across cohorts. Every recording
                    // this fold holds out must be absent from every cohort's slice of the training set.
                    for h in (0..n).filter(|j| j % folds == g).map(|j| &loaded[j].train) {
                        assert!(
                            !train.iter().any(|t| (t.cohort, t.id) == (h.cohort, h.id)),
                            "held-out recording ({}, {}) is in the union training set", h.cohort, h.id
                        );
                    }
                }
                cache[g] = Some(fit(&train));
            }
            cache[g].clone().expect("the fold was just filled")
        })
        .collect()
}

/// Score one cohort under one arm's per-recording configs. Labels are aligned to truth BY TIME, not
/// by position: an epoch carrying neither HR nor gravity is dropped from the staging, so the
/// sequences can differ in length.
fn score(loaded: &[Loaded], cfgs: &[Option<SleepConfig>], p: &Params) -> Vec<Night> {
    score_with(loaded, |i, night| {
        let cfg = cfgs[i].as_ref()?;
        let st = run(&night.input, cfg, p);
        Some((epoch_starts_v2(st.prepared.as_ref()?), st.stages?))
    })
}

/// [`score`] over any source of labels: epoch starts and one stage per epoch, or `None` to drop the night.
fn score_with(
    loaded: &[Loaded],
    labels: impl Fn(usize, &Loaded) -> Option<(Vec<i64>, Vec<SleepStage>)>,
) -> Vec<Night> {
    let mut out = Vec::new();
    for (i, night) in loaded.iter().enumerate() {
        let Some((starts, stages)) = labels(i, night) else { continue };
        let (input, truth) = (&night.input, &night.truth);
        let w0 = input.start;
        let mut at: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (s, lab) in starts.iter().zip(&stages) {
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

/// One arm's per-night balanced accuracy, keyed by recording. Its own series because an arm may
/// move this and not macro F1, and then only a paired line on THIS statistic can resolve the gap.
fn ba_by_night(nights: &[Night]) -> BTreeMap<usize, f64> {
    nights.iter().filter_map(|n| Some((n.id, balanced_accuracy(&n.cm)?))).collect()
}

/// What one arm leaves behind for the next one to be judged against: its two per-night series and
/// its pooled worst-class recall. The last is what stops a headline bought by giving a class up
/// from printing AHEAD, so the card cannot report either gap without it.
struct Reading {
    f1: BTreeMap<usize, f64>,
    ba: BTreeMap<usize, f64>,
    min_recall: Option<f64>,
}

/// Scores one arm and returns its [`Reading`], so a later arm can be paired against it. `base` is
/// the null reading's, over the SAME cohort. `prev` is the rung this arm is BUILT ON, when that is
/// not the null: a combination arm's own question is what IT added, which the null line cannot
/// answer.
fn card(
    ds: &str,
    arm: &str,
    nights: &[Night],
    base: Option<&Reading>,
    prev: Option<(&str, &Reading)>,
    from: Provenance,
) -> Reading {
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
    let mine = Reading {
        f1: macro_f1_by_night(nights),
        ba: ba_by_night(nights),
        min_recall: min_recall(&cm),
    };
    // One differencer over either series and either base, so no two paired lines can drift apart.
    // `drop` is the worst-class guard and applies to every one of them. The provenance is v2's on
    // both sides: every arm descends from the same hand-tuned params, so pairing against a rung
    // rather than the null does not launder that.
    // `BORDER_NIGHTS` set prints the paired deltas themselves. Mean and bar cannot tell a few swung
    // nights from a uniform widening, and that difference decides whether an arm has a per-night
    // gate to find or nothing to find.
    let shapes = std::env::var_os("BORDER_NIGHTS").is_some();
    let against = |b: &Reading, pick: fn(&Reading) -> &BTreeMap<usize, f64>| {
        let (bv, av) = pair_by_id(pick(b), pick(&mine));
        let drop = b.min_recall.and_then(|x| Some(mine.min_recall? - x));
        let shape = if shapes {
            let d = pick(b)
                .iter()
                .filter_map(|(id, x)| pick(&mine).get(id).map(|y| format!("{id}:{:+.4}", y - x)))
                .collect::<Vec<_>>();
            format!("\n      per-night: {}", d.join(" "))
        } else {
            String::new()
        };
        format!(
            "paired on {} night(s): {}{shape}",
            bv.len(),
            compare_guarded(&bv, &av, from, drop).2
        )
    };
    let paired = |pick: fn(&Reading) -> &BTreeMap<usize, f64>| match base {
        None => "  (this arm IS the baseline)".to_string(),
        Some(b) => {
            let null = format!("  vs the null, {}", against(b, pick));
            match prev {
                Some((name, r)) => format!("{null}\n      vs {name}, {}", against(r, pick)),
                None => null,
            }
        }
    };
    println!("  per-night macro F1 {}{}", show(per_recording(&cms, macro_f1)), paired(|r| &r.f1));
    println!("  per-night balanced acc{}", paired(|r| &r.ba));

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

/// The columns a refit arm adds to the AWAKE row for one recording, epoch-major. `permute` shuffles
/// each raw z across the night's epochs (own draw per cohort, id, column): scale kept, alignment gone.
fn extras(t: &TrainNight, x: Extra, permute: bool) -> Vec<f64> {
    let r = &t.refit;
    let mut raw = [r.zhv.clone(), r.zhr.clone()];
    if x == Extra::RawGated {
        for col in raw.iter_mut() {
            for (v, backed) in col.iter_mut().zip(&r.rr_backed) {
                if !backed {
                    *v = 0.0;
                }
            }
        }
    }
    if permute {
        for (c, col) in raw.iter_mut().enumerate() {
            let mut z = PERM_SEED ^ splitmix(((t.cohort as u64) << 40) | ((t.id as u64) << 8) | (0x80 + c) as u64);
            for i in (1..col.len()).rev() {
                z = splitmix(z);
                col.swap(i, (z >> 11) as usize % (i + 1));
            }
        }
    }
    let mut out = Vec::with_capacity(x.k() * r.starts.len());
    for (e, (hv, hr)) in raw[0].iter().zip(&raw[1]).enumerate() {
        if x.k() >= 2 {
            out.extend([*hv, *hr]);
        }
        if x.k() == 4 {
            let s = f64::from(u8::from(r.still[e]));
            out.extend([hv * s, hr * s]);
        }
    }
    out
}

/// Refit v2's weights (plus the extras) on the TRAINING recordings only.
fn refit_fit(train: &[&TrainNight], spec: Refit) -> Option<Fitted> {
    let xs: Vec<Vec<f64>> = train.iter().map(|t| extras(t, spec.extra, spec.permute)).collect();
    let labelled = |t: &TrainNight| t.rd(spec.base).truth.iter().flatten().count() as f64;
    // Cohort balance: an epoch of cohort c weighs 1 / (cohorts in the fold x labelled epochs of c in
    // the fold), rescaled so the mean epoch weight is 1. A night's weight is that, uniformly.
    let mut per_cohort: BTreeMap<usize, f64> = BTreeMap::new();
    for t in train {
        *per_cohort.entry(t.cohort).or_default() += labelled(t);
    }
    per_cohort.retain(|_, e| *e > 0.0);
    let raw = |c: usize| 1.0 / (per_cohort.len() as f64 * per_cohort[&c]);
    let mass: f64 = per_cohort.iter().map(|(c, e)| e * raw(*c)).sum();
    let total: f64 = per_cohort.values().sum();
    let weight = |t: &TrainNight| {
        if spec.balanced && per_cohort.contains_key(&t.cohort) { raw(t.cohort) * total / mass } else { 1.0 }
    };
    let nights: Vec<FitNight> = train
        .iter()
        .zip(&xs)
        .filter_map(|(t, x)| {
            Some(FitNight { terms: t.rd(spec.base).terms.as_ref()?, extra: x, truth: &t.rd(spec.base).truth, weight: weight(t) })
        })
        .collect();
    (!nights.is_empty()).then(|| refit::fit(&nights, spec.extra.k()))
}

/// Labels from refitted weights: v2's emission plus the extras under v2's Viterbi and SHIPPED transition.
fn score_refit(loaded: &[Loaded], fits: &[Option<Fitted>], spec: Refit) -> Vec<Night> {
    let k = spec.extra.k();
    score_with(loaded, |i, night| {
        let f = fits[i].as_ref()?;
        let r = night.train.rd(spec.base);
        let terms = r.terms.as_ref()?;
        let x = extras(&night.train, spec.extra, false);
        let em: Vec<[f64; 4]> =
            (0..r.starts.len()).map(|e| refit::emission(terms, e, &x[e * k..(e + 1) * k], &f.w)).collect();
        Some((r.starts.clone(), decode_v2(&em, &BASE.transition)))
    })
}

/// Positive control: at v2's own weights and zero extras the refit path must give the pipeline's
/// labels and epoch grid on every night.
fn refit_control(all: &[Vec<Loaded>], clamp_on: bool) {
    let w = weights_of(&Params::SHIPPED);
    let w14: Vec<f64> = w.iter().copied().chain([0.0; 2]).collect();
    let mut n = 0;
    for l in all.iter().flatten() {
        let r = &l.train.refit;
        let Some(terms) = r.terms.as_ref() else { continue };
        let em: Vec<[f64; 4]> = (0..r.starts.len()).map(|e| terms.emission(e, &w)).collect();
        let x = extras(&l.train, Extra::Raw, false);
        for (e, want) in em.iter().enumerate() {
            assert_eq!(*want, refit::emission(terms, e, &x[e * 2..(e + 1) * 2], &w14), "zero-weight extras moved an emission");
        }
        let st = run(&l.input, &SleepConfig::shipped(), &Params::SHIPPED);
        assert_eq!(st.stages.as_deref(), Some(&decode_v2(&em, &Params::SHIPPED.transition)[..]),
            "the refit path at v2's weights does not reproduce the pipeline's labels");
        assert_eq!(st.prepared.as_ref().map(epoch_starts_v2).as_deref(), Some(&r.starts[..]));
        n += 1;
    }
    // The gate: extras are zero exactly where the epoch is not R-R backed, and a cohort with no R-R at all
    // (sleep-accel) carries an all-zero extra block, so A3's emission there is v2's twelve-weight emission.
    for (ci, ds) in COHORTS.iter().enumerate() {
        let (mut backed, mut epochs) = (0usize, 0usize);
        for l in &all[ci] {
            let x = extras(&l.train, Extra::RawGated, false);
            for (e, ok) in l.train.refit.rr_backed.iter().enumerate() {
                assert!(*ok || x[e * 2..(e + 1) * 2] == [0.0, 0.0], "gated extras nonzero on a non-R-R epoch");
                backed += usize::from(*ok);
            }
            epochs += l.train.refit.rr_backed.len();
        }
        println!("  R-R GATE {ds}: {backed} of {epochs} prepared epochs are R-R backed");
        if *ds == "sleep-accel" && !all[ci].is_empty() {
            assert_eq!(backed, 0, "sleep-accel is documented to carry no R-R; the gated arm would not equal A0's emission there");
        }
    }
    println!("REFIT CONTROL passes: at v2's own weights and zero extras the refit path reproduces the pipeline's");
    println!("labels and epoch grid on all {n} recordings; the DEEP/REM/AWAKE hr_var z columns agree.");
    if clamp_on {
        // The same control under tanv1::BASE: its decomposition at BASE's weights must give the pipeline's labels.
        let wb = weights_of(&BASE);
        let mut nb = 0;
        let mut flips = 0usize;
        for l in all.iter().flatten() {
            let r = &l.train.refit_base;
            let Some(terms) = r.terms.as_ref() else { continue };
            let em: Vec<[f64; 4]> = (0..r.starts.len()).map(|e| terms.emission(e, &wb)).collect();
            let st = run(&l.input, &SleepConfig::shipped(), &BASE);
            assert_eq!(st.stages.as_deref(), Some(&decode_v2(&em, &BASE.transition)[..]),
                "the BASE refit path at BASE's weights does not reproduce the pipeline's labels");
            flips += usize::from(l.train.refit.still != r.still);
            nb += 1;
        }
        println!("REFIT CONTROL (tanv1::BASE) passes on all {nb} recordings; {flips} recordings differ from SHIPPED in the clamp flag.");
    }
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt())
}

/// Fitted weights over the cohort's folds; the first `fold_count` recordings sit in distinct folds.
fn summarise_fits(name: &str, spec: Refit, fits: &[Option<Fitted>]) {
    let folds: Vec<&Fitted> = fits.iter().take(fold_count(fits.len())).filter_map(|f| f.as_ref()).collect();
    if folds.is_empty() {
        return;
    }
    let conv = folds.iter().filter(|f| f.converged).count();
    let it: Vec<f64> = folds.iter().map(|f| f.iters as f64).collect();
    println!("  refit weights, {name}: {} folds, {conv} converged, iters mean {:.0} max {:.0}",
        folds.len(), mean_sd(&it).0, it.iter().cloned().fold(0.0, f64::max));
    let names = WEIGHT_NAMES.iter().chain(&Extra::NAMES[..spec.extra.k()]);
    for (j, label) in names.enumerate() {
        let v: Vec<f64> = folds.iter().map(|f| f.w[j]).collect();
        let (m, sd) = mean_sd(&v);
        let pos = v.iter().filter(|x| **x > 0.0).count();
        let hand = if spec.base && j < refit::NW { format!("   hand {:+.4}", weights_of(&BASE)[j]) } else { String::new() };
        println!("    {label:<16} {m:>+8.4} +/- {sd:.4}   positive in {pos}/{}{}{hand}", v.len(),
            if j >= refit::NW { "   <- NEW" } else { "" });
    }
}

/// Marker for every paired line an oracle arm prints.
const ORACLE_PROVENANCE: Provenance = Provenance::InSample("the night's own PSG stage shares");

/// Laplace stage shares from truth counts, in truth-code order (wake, light, deep, rem).
fn laplace_shares(c: &[u64; 4]) -> [f64; 4] {
    let n: u64 = c.iter().sum();
    std::array::from_fn(|k| (c[k] + 1) as f64 / (n + 4) as f64)
}

/// A seeded derangement of `0..n`: one n-cycle over a shuffled order, so no index maps to itself.
fn derangement(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut r = seed;
    for i in (1..n).rev() {
        r = splitmix(r);
        order.swap(i, (r >> 11) as usize % (i + 1));
    }
    let mut d = vec![usize::MAX; n];
    for k in 0..n {
        d[order[k]] = order[(k + 1) % n];
    }
    assert!(d.iter().all(|j| *j < n), "the derangement is not a bijection");
    assert!(n < 2 || d.iter().enumerate().all(|(i, j)| i != *j), "the permutation has a fixed point");
    d
}

/// `tanv1::BASE` with the tilted classes' base rate scaled by (night share / reference share)^power.
/// Power 0 multiplies by exactly 1.0, so the params are BASE bit for bit.
fn oracle_params(spec: PerNight, reference: &[u64; 4], night: &[u64; 4]) -> Params {
    let (pi_c, pi_i) = (laplace_shares(reference), laplace_shares(night));
    let mut p = BASE;
    for (c, r) in p.base_rate.iter_mut().enumerate() {
        if spec.classes[c] {
            let k = stage_idx(STAGE_ORDER[c]);
            *r *= (pi_i[k] / pi_c[k]).powf(spec.power);
        }
    }
    p
}

fn oracle_labels(input: &SleepInput, p: &Params) -> Option<(Vec<i64>, Vec<SleepStage>)> {
    let st = run(input, &SleepConfig::shipped(), p);
    Some((epoch_starts_v2(st.prepared.as_ref()?), st.stages?))
}

/// Cohort `ci` scored under the oracle. The reference shares are the pooled truth counts of the
/// fold's TRAINING recordings (the held-out assert in `per_fold` still runs); the night's shares come
/// from itself, or from its derangement partner for the twin.
fn oracle_nights(all: &[Vec<Loaded>], ci: usize, spec: PerNight) -> Vec<Night> {
    let loaded = &all[ci];
    let pool = |t: &[&TrainNight]| t.iter().fold([0u64; 4], |a, n| std::array::from_fn(|k| a[k] + n.truth_counts[k]));
    let refs: Vec<[u64; 4]> = per_fold(all, ci, &pool, TrainSet::WithinCohort);
    let donor: Vec<usize> =
        if spec.permute { derangement(loaded.len(), PERM_SEED ^ ci as u64) } else { (0..loaded.len()).collect() };
    score_with(loaded, |i, night| {
        oracle_labels(&night.input, &oracle_params(spec, &refs[i], &loaded[donor[i]].train.truth_counts))
    })
}

/// Positive control: at power 0 the oracle's params are BASE and its labels are the clamp arm's on
/// every night. Also prints the derangement check. Printed once, before any card.
fn oracle_control(all: &[Vec<Loaded>]) {
    let clamp = Params { clamp_only_without_rr: true, ..Params::SHIPPED };
    assert!(BASE == clamp, "tanv1::BASE is no longer the clamp arm");
    let zero = PerNight { classes: [true; 4], power: 0.0, permute: false };
    println!("\nORACLE POSITIVE CONTROL (power 0 must equal params: clamp_only_without_rr on every night)");
    for (ci, ds) in COHORTS.iter().enumerate() {
        let loaded = &all[ci];
        let (mut same, mut n) = (0usize, 0usize);
        for l in loaded {
            let c = l.train.truth_counts;
            assert!(oracle_params(zero, &c, &c) == clamp, "power 0 moved the params");
            let (a, b) = (oracle_labels(&l.input, &oracle_params(zero, &c, &c)), oracle_labels(&l.input, &clamp));
            n += 1;
            same += (a.is_some() == b.is_some()
                && a.zip(b).is_none_or(|((sa, la), (sb, lb))| {
                    sa == sb && la.iter().map(|x| *x as usize).eq(lb.iter().map(|x| *x as usize))
                })) as usize;
        }
        let d = derangement(loaded.len(), PERM_SEED ^ ci as u64);
        println!(
            "  {ds:<12} labels identical on {same}/{n} nights; twin derangement of {} nights, {} fixed points",
            d.len(),
            d.iter().enumerate().filter(|(i, j)| i == *j).count()
        );
        assert_eq!(same, n, "the oracle at power 0 is not the clamp arm");
    }
}

/// P4: a per-night hr_var RANK gate on the AWAKE row, the cardiac analogue of v2's jerk gate. On an
/// R-R backed epoch whose within-night rank of hr_var (among the night's R-R backed epochs) is at least
/// `q`, `g` is added to the AWAKE emission. Two hand constants, no fit. The null adds the same `g` to
/// the same NUMBER of R-R backed epochs chosen at random, so only the rank is destroyed.
#[derive(Clone, Copy)]
struct HrvGate {
    g: f64,
    q: f64,
    null: bool,
}

/// P4 row, always paired against `on`.
fn hrv_gate_arm(name: &'static str, spec: HrvGate, on: &'static str) -> Row {
    Row(name, Engine::HrvGate(spec), BASE, Some(on), TrainSet::WithinCohort, None)
}

/// Which epochs the rank gate fires on. `zhv` is the raw per-night hr_var z, a monotone map of hr_var,
/// so its rank is hr_var's. Rank is the share of the night's R-R backed values at or below this one,
/// the way `v2::terms` ranks `turn`.
fn gate_mask(r: &RefitData, q: f64) -> Vec<bool> {
    let mut sorted: Vec<f64> = r.zhv.iter().zip(&r.rr_backed).filter(|(_, b)| **b).map(|(v, _)| *v).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("hr_var z is finite"));
    let n = sorted.len() as f64;
    r.zhv
        .iter()
        .zip(&r.rr_backed)
        .map(|(v, b)| *b && sorted.partition_point(|s| s <= v) as f64 / n >= q)
        .collect()
}

/// The count-matched null: `count` of the night's R-R backed epochs, uniformly at random (seeded per
/// recording and per `q`, so every `g` at one `q` shares its draw).
fn random_mask(r: &RefitData, count: usize, seed: u64) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..r.rr_backed.len()).filter(|e| r.rr_backed[*e]).collect();
    assert!(count <= idx.len(), "the null cannot box more epochs than the night has R-R backed");
    let mut rnd = seed;
    for i in 0..count {
        rnd = splitmix(rnd);
        let j = i + (rnd >> 11) as usize % (idx.len() - i);
        idx.swap(i, j);
    }
    let mut m = vec![false; r.rr_backed.len()];
    idx[..count].iter().for_each(|e| m[*e] = true);
    m
}

/// One night's gated labels under `tanv1::BASE`: its emission with `g` added to AWAKE where `mask`, decoded.
fn gate_labels(r: &RefitData, g: f64, mask: &[bool]) -> Option<(Vec<i64>, Vec<SleepStage>)> {
    let terms = r.terms.as_ref()?;
    let w = weights_of(&BASE);
    let aw = refit::awake_col();
    let em: Vec<[f64; 4]> = (0..r.starts.len())
        .map(|e| {
            let mut x = terms.emission(e, &w);
            if mask[e] {
                x[aw] += g;
            }
            x
        })
        .collect();
    Some((r.starts.clone(), decode_v2(&em, &BASE.transition)))
}

/// Cohort `ci` scored under the gate (or its count-matched null), with a firing summary printed.
fn gate_nights(loaded: &[Loaded], ci: usize, name: &str, spec: HrvGate) -> Vec<Night> {
    let fired = std::cell::RefCell::new(Vec::<(usize, usize, usize, usize)>::new());
    let out = score_with(loaded, |i, night| {
        let _ = i;
        let r = &night.train.refit_base;
        let real = gate_mask(r, spec.q);
        let count = real.iter().filter(|b| **b).count();
        let mask = if spec.null {
            let seed = PERM_SEED ^ splitmix(((ci as u64) << 40) | ((night.train.id as u64) << 8) | (spec.q * 1000.0) as u64);
            let m = random_mask(r, count, seed);
            assert_eq!(m.iter().filter(|b| **b).count(), count, "the null is not count-matched");
            m
        } else {
            real
        };
        let wake = Some(stage_idx(SleepStage::Wake));
        let hits = mask.iter().zip(&r.truth).filter(|(m, t)| **m && **t == wake).count();
        fired.borrow_mut().push((count, r.rr_backed.iter().filter(|b| **b).count(), r.starts.len(), hits));
        gate_labels(r, spec.g, &mask)
    });
    let f = fired.into_inner();
    let mut counts: Vec<usize> = f.iter().map(|x| x.0).collect();
    counts.sort_unstable();
    let (boost, backed, epochs, hits) = f.iter().fold((0, 0, 0, 0), |a, x| (a.0 + x.0, a.1 + x.1, a.2 + x.2, a.3 + x.3));
    println!(
        "  GATE FIRING ({name}, {}): boosted/night median {} min {} max {} (total {boost} of {backed} R-R backed, {epochs} prepared); truth-wake among boosted {:.1}%",
        COHORTS[ci],
        counts.get(counts.len() / 2).copied().unwrap_or(0),
        counts.first().copied().unwrap_or(0),
        counts.last().copied().unwrap_or(0),
        100.0 * hits as f64 / boost.max(1) as f64,
    );
    out
}

/// Positive control: at g = 0 the gate's labels are the clamp arm's on every night, and on a night the
/// gate never fires (every night of a cohort with no R-R) g = 4 is the clamp arm too. Printed once,
/// before any card.
fn gate_control(all: &[Vec<Loaded>]) {
    println!("\nP4 POSITIVE CONTROL (g = 0 must equal params: clamp_only_without_rr on every night)");
    let same_labels = |a: &Option<(Vec<i64>, Vec<SleepStage>)>, b: &Option<(Vec<i64>, Vec<SleepStage>)>| {
        a.is_some() == b.is_some()
            && a.iter().zip(b).all(|((sa, la), (sb, lb))| {
                sa == sb && la.iter().map(|x| *x as usize).eq(lb.iter().map(|x| *x as usize))
            })
    };
    for (ci, ds) in COHORTS.iter().enumerate() {
        let (mut same, mut n, mut silent) = (0usize, 0usize, 0usize);
        for l in &all[ci] {
            let r = &l.train.refit_base;
            let want = oracle_labels(&l.input, &BASE);
            let mask = gate_mask(r, 0.90);
            n += 1;
            same += usize::from(same_labels(&want, &gate_labels(r, 0.0, &mask)));
            if !mask.iter().any(|b| *b) {
                silent += 1;
                assert!(same_labels(&want, &gate_labels(r, 4.0, &mask)), "a night the gate never fires moved under g = 4");
            }
        }
        println!("  {ds:<12} g=0 labels identical on {same}/{n} nights; {silent} night(s) never fire at q 0.90 (g=4 identical there)");
        assert_eq!(same, n, "the gate at g = 0 is not the clamp arm");
    }
}

/// U7 + P5: the clamp arm's emission decoded under a changed transition. `tv` is `(a, shift_min)`: the
/// LIGHT row's light->deep logit is reduced by `a * u`, `u = clamp((minutes since sleep onset + shift_min) / 480,
/// 0, 1)` (Ernest Table 3: about 1.47 logits over 8 h; `TIME-VARYING-TRANSITIONS.md:17-52`). `stay` is the AWAKE
/// self-loop (shipped 0.90). `gate` is P4's `(g, q)` rank gate on the emission. `null` permutes each night's `u`
/// values across its epochs (same distribution, time structure destroyed).
#[derive(Clone, Copy)]
struct Trans {
    tv: Option<(f64, f64)>,
    stay: f64,
    gate: Option<(f64, f64)>,
    null: bool,
}

/// Minutes over which `u` runs 0 -> 1. Fixed, not per night: the covariate must not depend on the labels.
const U_SPAN_MIN: f64 = 480.0;
/// v2's `SUSTAINED_ONSET_EPOCHS`, private to `v2.rs` (frozen). Replicated so the onset notion is v2's.
const ONSET_RUN: usize = 10;

/// U7/P5 row, always paired against `on`.
fn trans_arm(name: &'static str, spec: Trans, on: &'static str) -> Row {
    Row(name, Engine::Trans(spec), BASE, Some(on), TrainSet::WithinCohort, None)
}

/// `BASE.transition` with the AWAKE self-loop at `stay`; the freed (or taken) mass moves through the other
/// allowed AWAKE exits in their existing proportions, so wake->deep and wake->rem stay 0.
fn stay_matrix(stay: f64) -> [[f64; 4]; 4] {
    let mut m = BASE.transition;
    let aw = 3;
    if stay == m[aw][aw] {
        return m;
    }
    let off: f64 = (0..4).filter(|j| *j != aw).map(|j| m[aw][j]).sum();
    for j in (0..4).filter(|j| *j != aw) {
        m[aw][j] = m[aw][j] * (1.0 - stay) / off;
    }
    m[aw][aw] = stay;
    m
}

/// The LIGHT row with the light->deep logit reduced by `a * u` and the row renormalised. Every other
/// logit is untouched and a zero entry stays zero. Row order is [deep, rem, light, awake].
fn tv_matrix(base: &[[f64; 4]; 4], a: f64, u: f64) -> [[f64; 4]; 4] {
    let mut m = *base;
    let (light, deep) = (2, 0);
    m[light][deep] = base[light][deep] * (-a * u).exp();
    let sum: f64 = m[light].iter().sum();
    m[light].iter_mut().for_each(|x| *x /= sum);
    m
}

/// First epoch of the first run of `ONSET_RUN` non-wake labels: v2's `sustained_onset`.
fn onset_of(labels: &[SleepStage]) -> Option<usize> {
    let mut run = 0usize;
    for (i, l) in labels.iter().enumerate() {
        run = if *l == SleepStage::Wake { 0 } else { run + 1 };
        if run == ONSET_RUN {
            return Some(i + 1 - ONSET_RUN);
        }
    }
    None
}

/// One night's `u` per prepared epoch, anchored at the onset of the clamp labels (no onset: all zero).
fn u_of(starts: &[i64], clamp: &[SleepStage], shift_min: f64) -> Vec<f64> {
    let Some(o) = onset_of(clamp) else { return vec![0.0; starts.len()] };
    starts
        .iter()
        .map(|s| (((*s - starts[o]) as f64 / 60.0 + shift_min) / U_SPAN_MIN).clamp(0.0, 1.0))
        .collect()
}

type Labels = Option<(Vec<i64>, Vec<SleepStage>)>;

/// One night's labels under `spec`. `seed` drives the `u` permutation of the null.
fn trans_labels(r: &RefitData, spec: Trans, seed: u64) -> Labels {
    let terms = r.terms.as_ref()?;
    let w = weights_of(&BASE);
    let aw = refit::awake_col();
    let plain: Vec<[f64; 4]> = (0..r.starts.len()).map(|e| terms.emission(e, &w)).collect();
    let mut em = plain.clone();
    if let Some((g, q)) = spec.gate {
        let mask = gate_mask(r, q);
        em.iter_mut().zip(&mask).filter(|(_, m)| **m).for_each(|(x, _)| x[aw] += g);
    }
    let base = stay_matrix(spec.stay);
    let Some((a, shift)) = spec.tv else { return Some((r.starts.clone(), decode_v2(&em, &base))) };
    let mut u = u_of(&r.starts, &decode_v2(&plain, &BASE.transition), shift);
    if spec.null {
        let mut rnd = seed;
        for i in (1..u.len()).rev() {
            rnd = splitmix(rnd);
            u.swap(i, (rnd >> 11) as usize % (i + 1));
        }
    }
    Some((r.starts.clone(), conditioned::viterbi_with(&em, |t| tv_matrix(&base, a, u[t]))))
}

/// Bout count and epochs per class over the SCORED epochs (truth-labelled, in time order), for the
/// prediction `at` (start -> class) or, when `at` is `None`, for the truth itself. Index = `stage_idx`.
fn bout_tally(
    truth: &BTreeMap<usize, i32>,
    w0: i64,
    at: Option<&std::collections::HashMap<i64, usize>>,
) -> ([usize; 4], [usize; 4]) {
    let (mut n, mut e) = ([0usize; 4], [0usize; 4]);
    let mut prev: Option<(usize, usize)> = None;
    for (k, t) in truth {
        let c = match at {
            Some(m) => (0..4).contains(t).then(|| m.get(&(w0 + *k as i64 * EPOCH))).flatten().copied(),
            None => (0..4).contains(t).then_some(*t as usize),
        };
        let Some(c) = c else {
            prev = None;
            continue;
        };
        e[c] += 1;
        if prev != Some((k.wrapping_sub(1), c)) {
            n[c] += 1;
        }
        prev = Some((*k, c));
    }
    (n, e)
}

/// Cohort `ci` scored under `spec`, with the deep and wake BOUT COUNT and mean length printed beside truth's.
fn trans_nights(loaded: &[Loaded], ci: usize, name: &str, spec: Trans) -> Vec<Night> {
    tally_nights(loaded, ci, name, |_, night| {
        let r = &night.train.refit_base;
        let seed = PERM_SEED ^ splitmix(((ci as u64) << 40) | ((night.train.id as u64) << 8) | 0x77);
        trans_labels(r, spec, seed)
    })
}

/// Cohort `ci` scored with `labels(i, night)`, and the deep and wake bout tally printed beside truth's.
fn tally_nights(loaded: &[Loaded], ci: usize, name: &str, labels: impl Fn(usize, &Loaded) -> Labels) -> Vec<Night> {
    let tally = std::cell::RefCell::new([[0usize; 4]; 4]);
    let out = score_with(loaded, |i, night| {
        let (starts, stages) = labels(i, night)?;
        let at: std::collections::HashMap<i64, usize> =
            starts.iter().copied().zip(stages.iter().map(|s| *s as usize)).collect();
        let (pn, pe) = bout_tally(&night.truth, night.input.start, Some(&at));
        let (tn, te) = bout_tally(&night.truth, night.input.start, None);
        let mut t = tally.borrow_mut();
        for c in 0..4 {
            t[0][c] += pn[c];
            t[1][c] += pe[c];
            t[2][c] += tn[c];
            t[3][c] += te[c];
        }
        Some((starts, stages))
    });
    let [p_n, p_e, t_n, t_e] = tally.into_inner();
    let mean = |n: usize, e: usize| e as f64 * EPOCH_MIN / n.max(1) as f64;
    let row = |c: usize| {
        format!("{} bouts mean {:.1} min (truth {} / {:.1})", p_n[c], mean(p_n[c], p_e[c]), t_n[c], mean(t_n[c], t_e[c]))
    };
    println!("  BOUTS ({name}, {}): deep {}; wake {}", COHORTS[ci], row(2), row(0));
    out
}

/// Positive controls, printed once before any card. Matrices: a = 0 and stay 0.90 are BASE's, rows sum to 1,
/// structural zeros survive, only the LIGHT row moves. Labels: a = 0 through the per-epoch decoder, and stay
/// 0.90, equal the clamp arm on every night; a = 1.5 must change some labels (the arm is not inert).
fn trans_control(all: &[Vec<Loaded>]) {
    println!("\nU7/P5 POSITIVE CONTROL (a = 0 and stay = 0.90 must equal params: clamp_only_without_rr on every night)");
    assert_eq!(stay_matrix(0.90), BASE.transition, "stay 0.90 is not BASE's matrix");
    assert_eq!(tv_matrix(&BASE.transition, 0.0, 1.0), BASE.transition, "a = 0 moved the matrix");
    for (a, u) in [(1.5, 1.0), (0.75, 0.4), (1.5, 0.0)] {
        let m = tv_matrix(&BASE.transition, a, u);
        assert!(m.iter().all(|r| (r.iter().sum::<f64>() - 1.0).abs() < 1e-12), "row sum moved at a {a} u {u}");
        for (i, row) in m.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                assert!(BASE.transition[i][j] != 0.0 || *x == 0.0, "a structural zero moved [{i}][{j}]");
            }
        }
    }
    let m = tv_matrix(&BASE.transition, 1.5, 1.0);
    assert!(m[2][0] < BASE.transition[2][0], "deep entry did not fall");
    assert!((0..4).filter(|r| *r != 2).all(|r| m[r] == BASE.transition[r]), "a row other than LIGHT moved");
    let loose = stay_matrix(0.80);
    assert!(loose[3][0] == 0.0 && loose[3][1] == 0.0 && (loose[3][2] - 0.20).abs() < 1e-12, "wake exits wrong: {:?}", loose[3]);
    let same = |a: &Labels, b: &Labels| {
        a.is_some() == b.is_some()
            && a.iter().zip(b).all(|((sa, la), (sb, lb))| sa == sb && la.iter().map(|x| *x as usize).eq(lb.iter().map(|x| *x as usize)))
    };
    for (ci, ds) in COHORTS.iter().enumerate() {
        let (mut n, mut tv0, mut st, mut moved) = (0usize, 0usize, 0usize, 0usize);
        for l in &all[ci] {
            let r = &l.train.refit_base;
            let want = oracle_labels(&l.input, &BASE);
            let flat = |tv, stay| Trans { tv, stay, gate: None, null: false };
            n += 1;
            tv0 += usize::from(same(&want, &trans_labels(r, flat(Some((0.0, 0.0)), 0.90), 1)));
            st += usize::from(same(&want, &trans_labels(r, flat(None, 0.90), 1)));
            moved += usize::from(!same(&want, &trans_labels(r, flat(Some((1.5, 0.0)), 0.90), 1)));
        }
        println!("  {ds:<12} a=0 labels identical on {tv0}/{n}; stay 0.90 identical on {st}/{n}; a=1.5 changes labels on {moved}/{n} nights");
        assert_eq!(tv0, n, "a = 0 through the per-epoch decoder is not the clamp arm");
        assert_eq!(st, n, "stay 0.90 is not the clamp arm");
    }
}

/// P6: the AWAKE self-loop `stay` together with a wake prior. Both are `Params` fields, so the whole engine
/// (probe anchor included) sees them as a shipped change would. `mult` scales `base_rate[AWAKE]` for every
/// night alike (label-free; rescaling all four classes would add one constant to every logit and move
/// nothing, so this is the tilt). `oracle` replaces it with P1's `W p0.5` per-night tilt: IN-SAMPLE.
#[derive(Clone, Copy)]
struct WakeLoop {
    stay: f64,
    mult: f64,
    oracle: bool,
}

const ORACLE_W_P05: PerNight = PerNight { classes: [false, false, false, true], power: 0.5, permute: false };

fn wl_params(spec: WakeLoop, reference: &[u64; 4], own: &[u64; 4]) -> Params {
    let mut p = if spec.oracle { oracle_params(ORACLE_W_P05, reference, own) } else { BASE };
    p.base_rate[3] *= spec.mult;
    Params { transition: stay_matrix(spec.stay), ..p }
}

fn wl_arm(name: &'static str, spec: WakeLoop, also: Option<&'static str>) -> Row {
    let r = Row(name, Engine::WakeLoop(spec), BASE, Some("params: clamp_only_without_rr"), TrainSet::WithinCohort, None);
    match also {
        Some(a) => r.also(a),
        None => r,
    }
}

/// P6 rows: 3 loops x 4 multipliers minus the clamp cell, then the in-sample oracle pair.
fn wake_loop_grid() -> Vec<Row> {
    let name = |stay: f64, mult: f64| -> &'static str { Box::leak(format!("wl loop {stay:.2} x{mult:.2}").into_boxed_str()) };
    let mut rows = Vec::new();
    for mult in [1.0, 1.25, 1.5, 2.0] {
        for stay in [0.90, 0.85, 0.80] {
            if stay == 0.90 && mult == 1.0 {
                continue;
            }
            let also = (stay < 0.90 && mult > 1.0).then(|| name(0.90, mult));
            rows.push(wl_arm(name(stay, mult), WakeLoop { stay, mult, oracle: false }, also));
        }
    }
    rows.push(wl_arm("wl oracle W p0.5 loop 0.90", WakeLoop { stay: 0.90, mult: 1.0, oracle: true }, None));
    rows.push(wl_arm("wl oracle W p0.5 loop 0.80", WakeLoop { stay: 0.80, mult: 1.0, oracle: true }, Some("wl oracle W p0.5 loop 0.90")));
    rows
}

/// Cohort `ci` scored under `spec`, with bout tallies.
fn wl_nights(all: &[Vec<Loaded>], ci: usize, name: &str, spec: WakeLoop) -> Vec<Night> {
    let loaded = &all[ci];
    let pool = |t: &[&TrainNight]| t.iter().fold([0u64; 4], |a, n| std::array::from_fn(|k| a[k] + n.truth_counts[k]));
    let refs: Vec<[u64; 4]> = per_fold(all, ci, &pool, TrainSet::WithinCohort);
    tally_nights(loaded, ci, name, |i, night| {
        oracle_labels(&night.input, &wl_params(spec, &refs[i], &loaded[i].train.truth_counts))
    })
}

/// Positive control: (stay 0.90, mult 1.0) is the clamp arm on every night; the two levers are not inert.
fn wl_control(all: &[Vec<Loaded>]) {
    println!("\nP6 POSITIVE CONTROL (stay 0.90, mult 1.0 must equal params: clamp_only_without_rr on every night)");
    let zero = WakeLoop { stay: 0.90, mult: 1.0, oracle: false };
    let c = [1u64; 4];
    assert!(wl_params(zero, &c, &c) == BASE, "stay 0.90 x1.0 is not BASE");
    let same = |a: &Labels, b: &Labels| {
        a.is_some() == b.is_some()
            && a.iter().zip(b).all(|((sa, la), (sb, lb))| sa == sb && la.iter().map(|x| *x as usize).eq(lb.iter().map(|x| *x as usize)))
    };
    for (ci, ds) in COHORTS.iter().enumerate() {
        let (mut n, mut ok, mut mv_m, mut mv_s) = (0usize, 0usize, 0usize, 0usize);
        for l in &all[ci] {
            let t = l.train.truth_counts;
            let want = oracle_labels(&l.input, &BASE);
            n += 1;
            ok += usize::from(same(&want, &oracle_labels(&l.input, &wl_params(zero, &t, &t))));
            mv_m += usize::from(!same(&want, &oracle_labels(&l.input, &wl_params(WakeLoop { mult: 2.0, ..zero }, &t, &t))));
            mv_s += usize::from(!same(&want, &oracle_labels(&l.input, &wl_params(WakeLoop { stay: 0.80, ..zero }, &t, &t))));
        }
        println!("  {ds:<12} labels identical on {ok}/{n}; x2.0 changes labels on {mv_m}/{n}; loop 0.80 changes labels on {mv_s}/{n}");
        assert_eq!(ok, n, "stay 0.90 x1.0 is not the clamp arm");
    }
}

/// One arm's per-night kappa4, keyed by recording. The card prints no paired kappa4 of its own.
fn kappa_by_night(nights: &[Night]) -> BTreeMap<usize, f64> {
    nights.iter().map(|n| (n.id, kappa4(&n.cm))).filter(|(_, k)| k.is_finite()).collect()
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

/// MESA nights the frozen map is fitted on. Fixed, name order, so the fit is one reproducible object.
const MESA_FIT_NIGHTS: usize = 400;

/// One MESA night as fit rows: the fourteen percentile columns through OUR wire format (TIMING),
/// windowed and z-scored within the night as `cardiac_multivariate` does, with the truth stage per
/// row. `permute` shuffles the stage labels within the night, keyed on `ni`.
fn mesa_night_rows(n: &MesaNight, ni: usize, permute: bool) -> (Vec<[f64; COLS]>, Vec<usize>) {
    let pairs: Vec<(f64, f64)> = mesa::degrade_timing(&n.beats).iter().map(|b| (b.t, b.rr)).collect();
    let (mut raw, mut ys) = (Vec::new(), Vec::new());
    for (e, s) in n.stage.iter().enumerate() {
        let Some(s) = s else { continue };
        let c = e as f64 * MESA_EPOCH_S + MESA_EPOCH_S / 2.0;
        if let Some(b) = cardiac::extract(&pairs, c - WINDOW_S / 2.0, c + WINDOW_S / 2.0) {
            let r = b.row();
            raw.push(core::array::from_fn(|j| r[j]));
            ys.push(*s);
        }
    }
    let mut z = vec![[f64::NAN; COLS]; raw.len()];
    for j in 0..COLS {
        let col: Vec<Option<f64>> = raw.iter().map(|r: &[f64; COLS]| r[j].is_finite().then_some(r[j])).collect();
        for (k, v) in cardiac::zscore_column(&col).into_iter().enumerate() {
            z[k][j] = v.unwrap_or(f64::NAN);
        }
    }
    if permute {
        let mut r = PERM_SEED ^ splitmix(ni as u64);
        for i in (1..ys.len()).rev() {
            r = splitmix(r);
            ys.swap(i, (r >> 11) as usize % (i + 1));
        }
    }
    (z, ys)
}

/// The discriminant fitted ONCE on MESA and frozen.
/// Cached per `permute`: every fold, cohort and lambda reads the same fit, and the card's training
/// nights are never consulted.
fn mesa_frozen(permute: bool) -> &'static Option<Lda> {
    static REAL: std::sync::OnceLock<Option<Lda>> = std::sync::OnceLock::new();
    static TWIN: std::sync::OnceLock<Option<Lda>> = std::sync::OnceLock::new();
    (if permute { &TWIN } else { &REAL }).get_or_init(|| {
        // MESA codes stages wake 0, light 1, deep 2, REM 3 (`mesa::stage_of`); the discriminant is
        // fitted in FIT_ORDER. Pin the two together, and the decoder's order via the same reindex.
        assert_eq!(FIT_ORDER.map(stage_idx), [0, 1, 2, 3], "MESA stage codes are not in FIT_ORDER");
        assert_eq!(fc_in_stage_order([0.0, 1.0, 2.0, 3.0]).map(|v| v as usize),
            STAGE_ORDER.map(stage_idx), "reindex disagrees with the stage codes");
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (ni, n) in mesa::nights(MESA_FIT_NIGHTS).iter().enumerate() {
            let (rows, ys) = mesa_night_rows(n, ni, permute);
            for (r, c) in rows.iter().zip(&ys) {
                if r.iter().all(|v| v.is_finite()) {
                    x.push(r.to_vec());
                    y.push(*c);
                }
            }
        }
        eprintln!("MESA-frozen fit{}: {} rows", if permute { " (PERMUTED labels)" } else { "" }, x.len());
        Lda::fit(&x, &y, RIDGE)
    })
}

/// The frozen MESA map at `lambda_milli`. Ignores the training recordings by construction.
fn mesa_arm(lambda_milli: u32, permute: bool) -> Arm {
    Arm(Box::new(move |_| {
        let emit = CardiacEmit::from_lda(mesa_frozen(permute).as_ref()?, lambda_milli)?;
        Some(SleepConfig { emit: EmitCfg::V2PlusCardiac(emit), decode: DecodeCfg::Viterbi })
    }))
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

/// Silence the time term. `Features::clock` reaches the emission only through `cycle_prior`, read by
/// `cycle_clock` and `rem_guard` and nowhere else, so zeroing the two scales and the early-REM penalty
/// leaves no time-of-night contribution. `golden_tests::zeroing_the_cycle_scales_...` pins that.
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
    let arms: Vec<Row> = vec![
        arm("v2 shipped recipe (NULL READING)", fixed(SleepConfig::shipped()), Params::SHIPPED),
        arm("transition 50% toward uniform", fixed(SleepConfig::shipped()), flattened(0.5)),
        arm("transition UNIFORM - emissions alone", fixed(SleepConfig::shipped()), flattened(1.0)),
        arm("wake>rem and wake>deep opened at truth's rate", fixed(SleepConfig::shipped()), wake_row_opened()),
        // The only time-term arm this corpus supports: every fixture night is rebased onto ONE
        // synthetic start, so a wall-clock or habitual-midpoint anchor carries no between-recording
        // variation and is not measurable here. Ablation only.
        arm("no time term (cycle prior off)", fixed(SleepConfig::shipped()), no_time_term()),
        // v2's two documented switches, paired against the null reading. `Params` only, so v2.rs is untouched.
        arm("params: clamp_only_without_rr", fixed(SleepConfig::shipped()),
            Params { clamp_only_without_rr: true, ..Params::SHIPPED }),
        arm("params: quiescent_hr_z_max 0.5", fixed(SleepConfig::shipped()),
            Params { quiescent_hr_z_max: 0.5, ..Params::SHIPPED }),
        arm("params: both", fixed(SleepConfig::shipped()),
            Params { clamp_only_without_rr: true, quiescent_hr_z_max: 0.5, ..Params::SHIPPED }),
        arm("conditioned diagonal, beta 0.25", fixed(conditioned(250)), Params::SHIPPED),
        arm("conditioned diagonal, beta 0.5", fixed(conditioned(500)), Params::SHIPPED),
        arm("conditioned diagonal, beta 1.0", fixed(conditioned(1000)), Params::SHIPPED),
        arm("cardiac emission, lambda 0.5", cardiac_arm(500, false), Params::SHIPPED),
        arm("cardiac emission, lambda 1.0", cardiac_arm(1000, false), Params::SHIPPED),
        arm("cardiac emission, lambda 2.0", cardiac_arm(2000, false), Params::SHIPPED),
        arm("cardiac PERMUTED null, lambda 0.5", cardiac_arm(500, true), Params::SHIPPED),
        arm("cardiac PERMUTED null, lambda 1.0", cardiac_arm(1000, true), Params::SHIPPED),
        arm("cardiac PERMUTED null, lambda 2.0", cardiac_arm(2000, true), Params::SHIPPED),
        // Fitted ONCE on 400 MESA nights and frozen: the same map on every card night, so any
        // per-night spread it has is not fold noise. Each real arm is paired against the
        // within-cohort arm at the same lambda as well as against the null.
        rung("cardiac MESA-frozen, lambda 0.5", mesa_arm(500, false), Params::SHIPPED,
             "cardiac emission, lambda 0.5"),
        rung("cardiac MESA-frozen, lambda 1.0", mesa_arm(1000, false), Params::SHIPPED,
             "cardiac emission, lambda 1.0"),
        arm("cardiac MESA-frozen PERMUTED null, lambda 0.5", mesa_arm(500, true), Params::SHIPPED),
        arm("cardiac MESA-frozen PERMUTED null, lambda 1.0", mesa_arm(1000, true), Params::SHIPPED),
        // The decode seam. `PosteriorMarginal` and `MarkovLoss` at unit costs are ONE rule and must
        // print the same card; the weighted arms move the class weighting into the decode, which is
        // the rule a class-balanced objective implies.
        arm("posterior-marginal decode", fixed(marginal), Params::SHIPPED),
        arm("loss-matched decode, UNIT costs", fixed(unit_loss), Params::SHIPPED),
        arm("loss-matched decode, fc^0.5 inverse prevalence", loss_arm(0.5, false), Params::SHIPPED),
        arm("loss-matched decode, inverse-prevalence fc", loss_arm(1.0, false), Params::SHIPPED),
        arm("loss-matched decode, PERMUTED fc", loss_arm(1.0, true), Params::SHIPPED),
        // The one arm that stacks two seams. Against the null it cannot say which of the two moved
        // it, so it also pairs against the emission it is built on.
        rung("cardiac lambda 1.0 + inverse-prevalence fc", cardiac_loss_arm(1000, 1.0),
             Params::SHIPPED, "cardiac emission, lambda 1.0"),
        // Fitted on the UNION: the other recordings of the held-out one's cohort plus every recording
        // of the other two. Paired against the within-cohort arm at the same lambda as well as the
        // null; the twin is the same union fit with column-to-stage alignment destroyed.
        union_rung("cardiac emission UNION, lambda 1.0", cardiac_arm(1000, false), Params::SHIPPED,
                   "cardiac emission, lambda 1.0"),
        union_arm("cardiac PERMUTED null UNION, lambda 1.0", cardiac_arm(1000, true), Params::SHIPPED),
        // v2's own emission with its twelve weights refitted on the union: A0 is the honest rung, A1/A2
        // add the awake cardiac z without deadzone or clamp, the twin trains A1 on shuffled columns.
        refit_arm("v2 REFIT (union)", Extra::None, false, false, None),
        refit_arm("refit + awake cardiac unclamped", Extra::Raw, false, false, Some("v2 REFIT (union)")),
        refit_arm("refit + awake cardiac unclamped PERMUTED null", Extra::Raw, true, false, Some("v2 REFIT (union)")),
        refit_arm("refit + awake cardiac unclamped x still", Extra::RawStill, false, false,
                  Some("refit + awake cardiac unclamped")),
        // U4b: the two diagnosed causes. A0b balances the union by cohort; A3 gates the unclamped pair
        // on R-R; A4 does both.
        refit_arm("v2 REFIT (union, cohort-balanced)", Extra::None, false, true, Some("v2 REFIT (union)")),
        refit_arm("refit + awake cardiac unclamped, R-R gated", Extra::RawGated, false, false, Some("v2 REFIT (union)")),
        refit_arm("refit + awake cardiac unclamped, R-R gated PERMUTED null", Extra::RawGated, true, false,
                  Some("v2 REFIT (union)")),
        refit_arm("refit + awake cardiac unclamped, R-R gated, cohort-balanced", Extra::RawGated, false, true,
                  Some("refit + awake cardiac unclamped, R-R gated"))
            .also("v2 REFIT (union, cohort-balanced)"),
        // M1: the winning recipe's twelve weights refitted honestly (cohort-balanced union), on tanv1::BASE's
        // clamp-on decomposition; and upstream ryanbr/noop #2613's halved RSA weight as a Params-only arm.
        refit_base_arm("refit on tanv1::BASE (clamp on), cohort-balanced", Extra::None, true,
                       Some("v2 REFIT (union, cohort-balanced)"))
            .also("params: clamp_only_without_rr"),
        rung("params: RSA weight halved (#2613), on clamp", fixed(SleepConfig::shipped()),
             Params { resp_weight: 0.3, ..BASE }, "params: clamp_only_without_rr"),
        // P1: ORACLE per-night prior ceiling. Each night's base rate is tilted by its own PSG stage shares,
        // so nothing here can ship; it asks whether a per-night prior can move kappa under Viterbi at all.
        // Each twin draws another night's shares (derangement within cohort) and precedes its real arm.
        per_night_arm("oracle W p0.5 PERMUTED null", PerNight { classes: [false, false, false, true], power: 0.5, permute: true },
                      "params: clamp_only_without_rr"),
        per_night_arm("oracle W p0.5", PerNight { classes: [false, false, false, true], power: 0.5, permute: false },
                      "params: clamp_only_without_rr").also("oracle W p0.5 PERMUTED null"),
        per_night_arm("oracle W p1.0 PERMUTED null", PerNight { classes: [false, false, false, true], power: 1.0, permute: true },
                      "params: clamp_only_without_rr"),
        per_night_arm("oracle W p1.0", PerNight { classes: [false, false, false, true], power: 1.0, permute: false },
                      "params: clamp_only_without_rr").also("oracle W p1.0 PERMUTED null"),
        per_night_arm("oracle ALL4 p0.5 PERMUTED null", PerNight { classes: [true; 4], power: 0.5, permute: true },
                      "params: clamp_only_without_rr"),
        per_night_arm("oracle ALL4 p0.5", PerNight { classes: [true; 4], power: 0.5, permute: false },
                      "params: clamp_only_without_rr").also("oracle ALL4 p0.5 PERMUTED null"),
        // P4: per-night hr_var rank gate on the AWAKE row, each against a count-matched random boost.
        hrv_gate_arm("hrv gate g2 q0.90 COUNT-MATCHED null", HrvGate { g: 2.0, q: 0.90, null: true }, "params: clamp_only_without_rr"),
        hrv_gate_arm("hrv gate g2 q0.90", HrvGate { g: 2.0, q: 0.90, null: false }, "params: clamp_only_without_rr").also("hrv gate g2 q0.90 COUNT-MATCHED null"),
        hrv_gate_arm("hrv gate g2 q0.95 COUNT-MATCHED null", HrvGate { g: 2.0, q: 0.95, null: true }, "params: clamp_only_without_rr"),
        hrv_gate_arm("hrv gate g2 q0.95", HrvGate { g: 2.0, q: 0.95, null: false }, "params: clamp_only_without_rr").also("hrv gate g2 q0.95 COUNT-MATCHED null"),
        hrv_gate_arm("hrv gate g4 q0.90 COUNT-MATCHED null", HrvGate { g: 4.0, q: 0.90, null: true }, "params: clamp_only_without_rr"),
        hrv_gate_arm("hrv gate g4 q0.90", HrvGate { g: 4.0, q: 0.90, null: false }, "params: clamp_only_without_rr").also("hrv gate g4 q0.90 COUNT-MATCHED null"),
        hrv_gate_arm("hrv gate g4 q0.95 COUNT-MATCHED null", HrvGate { g: 4.0, q: 0.95, null: true }, "params: clamp_only_without_rr"),
        hrv_gate_arm("hrv gate g4 q0.95", HrvGate { g: 4.0, q: 0.95, null: false }, "params: clamp_only_without_rr").also("hrv gate g4 q0.95 COUNT-MATCHED null"),
        // Controls: a = 0 and stay 0.90 ARE the clamp arm (asserted label for label); they exist to print its bout counts.
        trans_arm("tv light->deep a0.00 (== clamp)", Trans { tv: Some((0.0, 0.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr"),
        trans_arm("wake self-loop 0.90 (== clamp)", Trans { tv: None, stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr"),
        // U7: time-varying LIGHT->DEEP entry `a * u`. Each a has its permuted-u twin first, then the real arm,
        // then the anchor shifted 30 min later / earlier, paired against the unshifted real arm.
        trans_arm("tv light->deep a0.75 PERMUTED-u null", Trans { tv: Some((0.75, 0.0)), stay: 0.90, gate: None, null: true }, "params: clamp_only_without_rr"),
        trans_arm("tv light->deep a0.75", Trans { tv: Some((0.75, 0.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a0.75 PERMUTED-u null"),
        trans_arm("tv light->deep a0.75, anchor 30min later", Trans { tv: Some((0.75, -30.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a0.75"),
        trans_arm("tv light->deep a0.75, anchor 30min earlier", Trans { tv: Some((0.75, 30.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a0.75"),
        trans_arm("tv light->deep a1.50 PERMUTED-u null", Trans { tv: Some((1.50, 0.0)), stay: 0.90, gate: None, null: true }, "params: clamp_only_without_rr"),
        trans_arm("tv light->deep a1.50", Trans { tv: Some((1.50, 0.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a1.50 PERMUTED-u null"),
        trans_arm("tv light->deep a1.50, anchor 30min later", Trans { tv: Some((1.50, -30.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a1.50"),
        trans_arm("tv light->deep a1.50, anchor 30min earlier", Trans { tv: Some((1.50, 30.0)), stay: 0.90, gate: None, null: false }, "params: clamp_only_without_rr").also("tv light->deep a1.50"),
        // P5: AWAKE self-loop 0.90 -> 0.80, alone (i) and with P4's best gate g2 q0.95 (ii). (ii) is also paired against (i).
        trans_arm("wake self-loop 0.80", Trans { tv: None, stay: 0.80, gate: None, null: false }, "params: clamp_only_without_rr"),
        trans_arm("wake self-loop 0.80 + hrv gate g2 q0.95", Trans { tv: None, stay: 0.80, gate: Some((2.0, 0.95)), null: false }, "params: clamp_only_without_rr").also("wake self-loop 0.80"),
    ];
    let mut arms = arms;
    arms.extend(wake_loop_grid());

    // `BORDER_ARMS=a,b` keeps the null, every arm whose name contains one of the substrings, and
    // the arm each kept rung is built on. Unset is the full card.
    let arms: Vec<Row> = match std::env::var("BORDER_ARMS") {
        Ok(f) => {
            // ';' separates when present (arm names hold commas); a leading '=' matches the whole name.
            let pats: Vec<&str> = f.split(if f.contains(';') { ';' } else { ',' }).filter(|p| !p.is_empty()).collect();
            let hit = |name: &str, p: &str| p.strip_prefix('=').map_or_else(|| name.contains(p), |q| name == q);
            let mut keep: Vec<bool> = arms.iter().map(|r| r.0.contains("NULL READING") || pats.iter().any(|p| hit(r.0, p))).collect();
            for i in (0..arms.len()).rev() {
                if !keep[i] {
                    continue;
                }
                for base in [arms[i].3, arms[i].5].into_iter().flatten() {
                    keep[arms.iter().position(|x| x.0 == base).expect("rung base exists")] = true;
                }
            }
            arms.into_iter().zip(keep).filter_map(|(r, k)| k.then_some(r)).collect()
        }
        Err(_) => arms,
    };

    assert!(arms[0].0.contains("NULL READING"), "the first arm is the baseline every paired line differences against");
    // A rung can only be paired against something already scored, and a renamed arm must break the
    // table rather than silently pair against the null.
    for (i, r) in arms.iter().enumerate() {
        for on in [r.3, r.5].into_iter().flatten() {
            let at = arms.iter().position(|x| x.0 == on);
            assert!(at.is_some_and(|k| k < i), "{:?} is built on {on:?}, which is not an earlier arm", r.0);
        }
    }

    println!("THE BORDER — what any engine is measured on. Minutes, except efficiency in percent.");
    println!("Positive bias = the engine over-reports against PSG. Nothing here is a gate.");
    // Every cohort is loaded before any fold is built: a union arm trains across them.
    let all: Vec<Vec<Loaded>> = COHORTS
        .iter()
        .enumerate()
        .map(|(ci, ds)| {
            let mut l = if common::root(ds).is_dir() { load(ds) } else { Vec::new() };
            l.iter_mut().for_each(|x| x.train.cohort = ci);
            l
        })
        .collect();
    if arms.iter().any(|r| matches!(r.1, Engine::Refit(_))) {
        refit_control(&all, arms.iter().any(|r| matches!(&r.1, Engine::Refit(s) if s.base)));
    }
    if arms.iter().any(|r| matches!(r.1, Engine::PerNight(_))) {
        oracle_control(&all);
    }
    if arms.iter().any(|r| matches!(r.1, Engine::HrvGate(_))) {
        gate_control(&all);
    }
    if arms.iter().any(|r| matches!(r.1, Engine::Trans(_))) {
        trans_control(&all);
    }
    if arms.iter().any(|r| matches!(r.1, Engine::WakeLoop(_))) {
        wl_control(&all);
    }
    let mut summary: Vec<String> = Vec::new();
    for (ci, ds) in COHORTS.iter().enumerate() {
        if !common::root(ds).is_dir() {
            println!("\n{ds}: missing");
            continue;
        }
        let loaded = &all[ci];
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
        let own: Vec<&TrainNight> = loaded.iter().map(|l| &l.train).collect();
        if let Some(c) = prevalence_costs(&own, 1.0, false) {
            println!(
                "  inverse-prevalence fc over ALL recordings, STAGE_ORDER [deep rem light wake], geometric mean 1: {:.3} {:.3} {:.3} {:.3}",
                c.fc[0], c.fc[1], c.fc[2], c.fc[3]
            );
        }
        // The first arm is the null reading, and it is what every later arm's paired line is
        // differenced against, on this cohort's own nights.
        let mut readings: Vec<Reading> = Vec::new();
        let mut kappas: Vec<BTreeMap<usize, f64>> = Vec::new();
        for Row(name, a, p, on, set, also) in &arms {
            let prev = on.map(|n| {
                let k = arms.iter().position(|x| x.0 == n).expect("checked above");
                (n, &readings[k])
            });
            let nights = match a {
                Engine::Lib(a) => score(loaded, &configs(&all, ci, a, *set), p),
                Engine::PerNight(spec) => oracle_nights(&all, ci, *spec),
                Engine::HrvGate(spec) => gate_nights(loaded, ci, name, *spec),
                Engine::Trans(spec) => trans_nights(loaded, ci, name, *spec),
                Engine::WakeLoop(spec) => wl_nights(&all, ci, name, *spec),
                Engine::Refit(spec) => {
                    let fits = per_fold(&all, ci, &|t| refit_fit(t, *spec), *set);
                    println!("\n-- {ds}  arm: {name}");
                    summarise_fits(name, *spec, &fits);
                    score_refit(loaded, &fits, *spec)
                }
            };
            // An oracle reads the night's own truth: every paired line it prints says so.
            let (vs_null, vs_rung) = if matches!(a, Engine::PerNight(_)) || matches!(a, Engine::WakeLoop(w) if w.oracle) {
                (ORACLE_PROVENANCE, ORACLE_PROVENANCE)
            } else {
                (V2_PROVENANCE, Provenance::HeldOut)
            };
            let reading = card(ds, name, &nights, readings.first(), prev, vs_null);
            kappas.push(kappa_by_night(&nights));
            if let Engine::Refit(_) = a {
                // Paired per-night kappa4, which the card prints for no arm.
                let against = |b: &BTreeMap<usize, f64>, base_min: Option<f64>, from: Provenance| {
                    let (bv, av) = pair_by_id(b, kappas.last().expect("just pushed"));
                    let drop = base_min.and_then(|x| Some(reading.min_recall? - x));
                    format!("paired on {} night(s): {}", bv.len(), compare_guarded(&bv, &av, from, drop).2)
                };
                println!("  per-night kappa4 vs the null, {}", against(&kappas[0], readings[0].min_recall, V2_PROVENANCE));
                if let Some(n) = on {
                    let k = arms.iter().position(|x| x.0 == *n).expect("checked above");
                    println!("      vs {n}, {}", against(&kappas[k], readings[k].min_recall, Provenance::HeldOut));
                }
            }
            // Paired per-night kappa4 for EVERY arm, printed once at the end: the card prints none, and
            // adding it to a block would move blocks that must stay byte-identical.
            let kn = kappas.len() - 1;
            let vs_null = match readings.first() {
                Some(b) => paired_text(&kappas[0], &kappas[kn], b.min_recall, reading.min_recall, vs_null),
                None => "(this arm IS the baseline)".to_string(),
            };
            let mut line =
                format!("{ds:<12} {name}: kappa4 pooled {:.4}\n      vs the null, {vs_null}", kappa4(&pooled(&nights)));
            if let Some(n) = on {
                let k = arms.iter().position(|x| x.0 == *n).expect("checked above");
                line += &format!("\n      vs {n}, {}", paired_text(&kappas[k], &kappas[kn], readings[k].min_recall, reading.min_recall, vs_rung));
            }
            summary.push(line);
            if let Some(n) = also {
                let k = arms.iter().position(|x| x.0 == *n).expect("checked above");
                let (b, m) = (&readings[k], &reading);
                println!("  ALSO vs {n} ({ds}):");
                println!("      per-night macro F1 {}", paired_text(&b.f1, &m.f1, b.min_recall, m.min_recall, vs_rung));
                println!("      per-night balanced acc {}", paired_text(&b.ba, &m.ba, b.min_recall, m.min_recall, vs_rung));
                println!("      per-night kappa4 {}", paired_text(&kappas[k], &kappas[kn], b.min_recall, m.min_recall, vs_rung));
            }
            readings.push(reading);
        }
    }
    println!("\nPAIRED KAPPA4 SUMMARY (per-night, paired by recording id; min-recall guard as on the card)");
    for l in &summary {
        println!("{l}");
    }
}

/// Paired differencing of two per-night series `b` -> `a`, guarded by the worst-class recall fall.
fn paired_text(
    b: &BTreeMap<usize, f64>,
    a: &BTreeMap<usize, f64>,
    b_min: Option<f64>,
    a_min: Option<f64>,
    from: Provenance,
) -> String {
    let (bv, av) = pair_by_id(b, a);
    let drop = b_min.and_then(|x| Some(a_min? - x));
    format!("paired on {} night(s): {}", bv.len(), compare_guarded(&bv, &av, from, drop).2)
}
