//! What a candidate has to beat, and whether we could tell if it did.
//!
//!   cargo run --release -p physio-algo --example sleep_eval
//!
//! Every staging claim this project has made was a pooled kappa with no null arm, no held-out split and
//! no statement of what change is detectable. Three of them turned out to be noise. This harness reports
//! the four things that were missing: per-subject scores, null arms beside every number, the paired
//! per-subject sigma of a REALISTIC candidate, and the detectable delta that follows from it.
//!
//! DREAMT is the FIT cohort. AAUWSS and sleep-accel are HELD OUT — 13 and 31 subjects cannot both tune
//! and judge, and AAUWSS is the only cohort with gold ECG R-R, so spending it on fitting spends the one
//! clean read we have. sleep-accel carries no R-R at all and is the guard that a cardiac change does not
//! break the no-R-R path.
//!
//! Wake is the headline. Kappa is dominated by whichever stage holds the most epochs and moves by ~0.01
//! when a twenty-minute wake bout is missed, which is the whole complaint.

mod common;

use common::{
    dirs_of, mean, night_id, pre_retune, read_accel, read_hr, read_meta, read_rr, read_truth, stage_at,
    stage_idx,
};

use physio_algo::sleep::metrics::{bout_score, confusion4, kappa4, recall, specificity, BoutScore, WAKE};
use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, Prepared, SleepInput};

const EPOCH: i64 = 30;
/// Not a stage. Marks an epoch the reference did not label, so it breaks a bout and leaves the
/// confusion matrix rather than being scored as agreement.
const UNLABELLED: usize = 4;
const LIGHT: usize = 1;
/// A wake bout has to last this many epochs (5 min) before either side is asked about it.
const MIN_BOUT: usize = 10;
/// Share of a true bout's epochs that must carry the class before it counts as found.
const MIN_OVERLAP: f64 = 0.5;

const FIT: [&str; 1] = ["dreamt"];
const HELD_OUT: [&str; 2] = ["aauwss", "sleep-accel"];

struct Night {
    id: String,
    input: SleepInput,
    w0: i64,
    n: usize,
    /// Epoch index -> stage, labelled epochs only.
    truth: Vec<usize>,
}

fn load(dir: &std::path::Path) -> Option<Night> {
    let (w0, w1, n) = read_meta(dir)?;
    let raw = read_truth(dir);
    if raw.is_empty() {
        return None;
    }
    let mut truth = vec![UNLABELLED; n];
    for (k, t) in raw {
        if k < n && (0..4).contains(&t) {
            truth[k] = t as usize;
        }
    }
    Some(Night {
        id: night_id(dir).0,
        input: SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel: read_accel(dir) },
        w0,
        n,
        truth,
    })
}

/// One label per epoch, read at the midpoint so a segment boundary cannot fall between two reads.
fn stage_labels(night: &Night, prep: &Prepared, p: &Params) -> Vec<usize> {
    let segs = stage_v2_prepared(prep, p);
    (0..night.n)
        .map(|k| {
            let mid = night.w0 + k as i64 * EPOCH + EPOCH / 2;
            stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage))
        })
        .collect()
}

/// Deterministic per-night shuffle of our own labels. Keeps the marginal distribution and destroys the
/// timing, so it says what a score owes to calling the right AMOUNT of each stage.
fn shuffled(labels: &[usize], seed: u64) -> Vec<usize> {
    let mut out = labels.to_vec();
    let mut s = seed | 1;
    for i in (1..out.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        out.swap(i, (s >> 33) as usize % (i + 1));
    }
    out
}

/// The arms. `Recipe` mutates SHIPPED and re-stages; `Fixed` and `Shuffle` ignore the streams entirely
/// and exist so no number below can be read without its do-nothing floor beside it. `Rescue` is the
/// calibration ladder — see [`rescued`].
enum Arm {
    Recipe(Box<Params>),
    Fixed(usize),
    Shuffle,
    Rescue(f64),
}

/// A perfect feature that recovers exactly `frac` of the wake we currently miss. Takes the epochs where
/// truth is wake and we said sleep, and flips that fraction of them, spread evenly so a bout is thinned
/// rather than its head taken.
///
/// This is the only honest positive control: its size is CHOSEN, not borrowed from a paper measured on
/// another cohort with another model. It answers the one question a kill condition needs — how much of
/// the missed wake must a real feature find before this harness can prove it found anything.
fn rescued(pred: &[usize], truth: &[usize], frac: f64) -> Vec<usize> {
    let missed: Vec<usize> =
        (0..pred.len().min(truth.len())).filter(|i| truth[*i] == WAKE && pred[*i] != WAKE).collect();
    let take = (missed.len() as f64 * frac).round() as usize;
    let mut out = pred.to_vec();
    if take == 0 || missed.is_empty() {
        return out;
    }
    // Evenly spaced picks: taking a contiguous head would convert whole bouts and flatter the ladder.
    for j in 0..take {
        out[missed[j * missed.len() / take]] = WAKE;
    }
    out
}

/// Index into [`arms`] of the two paired references. `SMALL` is one threshold moved; `REAL` is the whole
/// previous shipped recipe, so the sigma between them brackets what a candidate is likely to produce.
const SHIPPED_ARM: usize = 0;
const SMALL_ARM: usize = 1;
const REAL_ARM: usize = 2;
const CAND_ARM: usize = 3;

/// The rescue ladder, as a fraction of currently-missed wake recovered.
const LADDER: [f64; 5] = [0.05, 0.10, 0.25, 0.50, 1.00];

fn arms() -> Vec<(&'static str, Arm)> {
    let nudge = Params { deep_gate_thresh: Params::SHIPPED.deep_gate_thresh + 0.05, ..Params::SHIPPED };
    let mut v: Vec<(&'static str, Arm)> = vec![
        ("shipped", Arm::Recipe(Box::new(Params::SHIPPED))),
        ("small: deep_gate +0.05", Arm::Recipe(Box::new(nudge))),
        ("real: the pre-retune recipe", Arm::Recipe(Box::new(pre_retune(&Params::SHIPPED)))),
        ("cand: clamp_only_without_rr",
            Arm::Recipe(Box::new(Params { clamp_only_without_rr: true, ..Params::SHIPPED }))),
        // The turn port, at three candidate weights. SHIPPED is 0.0, so the first row of this
        // family must reproduce shipped exactly; the others say what a fitted value could buy.
        ("cand: awake_turn 1.00",
            Arm::Recipe(Box::new(Params { awake_turn: 1.00, ..Params::SHIPPED }))),
        ("cand: awake_turn 1.50",
            Arm::Recipe(Box::new(Params { awake_turn: 1.50, ..Params::SHIPPED }))),
        ("cand: awake_turn 2.00",
            Arm::Recipe(Box::new(Params { awake_turn: 2.00, ..Params::SHIPPED }))),
        ("cand: turn 1.0 + clamp",
            Arm::Recipe(Box::new(Params { awake_turn: 1.0, clamp_only_without_rr: true, ..Params::SHIPPED }))),
        // THE NULL THAT MATTERS for any candidate that raises the wake rate: just call more wake,
        // via the AWAKE base rate, with no new information at all. A candidate only earns its place
        // if it beats this AT A MATCHED WAKE RATE. Coverage rises for free otherwise.
        ("null: more wake, base +0.05", Arm::Recipe(Box::new(bumped(0.05)))),
        ("null: more wake, base +0.10", Arm::Recipe(Box::new(bumped(0.10)))),
        ("null: more wake, base +0.15", Arm::Recipe(Box::new(bumped(0.15)))),
        ("null: more wake, base +0.20", Arm::Recipe(Box::new(bumped(0.20)))),
        ("null: always wake", Arm::Fixed(WAKE)),
        ("null: always light", Arm::Fixed(LIGHT)),
        ("null: shuffled ours", Arm::Shuffle),
    ];
    for (label, f) in ["rescue 5% of missed wake", "rescue 10%", "rescue 25%", "rescue 50%",
        "rescue 100% (oracle)"].iter().zip(LADDER) {
        v.push((label, Arm::Rescue(f)));
    }
    v
}

/// Every number for one subject under one arm. `None` where the subject cannot answer — a night with no
/// true wake bout has no wake-bout recall, and averaging a zero in its place invents a failure.
#[derive(Clone, Copy)]
struct Subject {
    kappa: f64,
    wake_recall: Option<f64>,
    wake_spec: Option<f64>,
    bout: BoutScore,
}

fn score_subject(pred_raw: &[usize], truth: &[usize]) -> Subject {
    // Mask our prediction wherever the reference is silent: a predicted bout over unlabelled epochs has
    // nothing to be right or wrong about, and counting it spurious would penalise a gap in the truth.
    let pred: Vec<usize> =
        pred_raw.iter().zip(truth).map(|(p, t)| if *t == UNLABELLED { UNLABELLED } else { *p }).collect();
    let cm = confusion4(&pred, truth);
    Subject {
        kappa: kappa4(&cm),
        wake_recall: recall(&cm, WAKE),
        wake_spec: specificity(&cm, WAKE),
        bout: bout_score(&pred, truth, WAKE, MIN_BOUT, MIN_OVERLAP),
    }
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn some(v: &[Option<f64>]) -> Vec<f64> {
    v.iter().filter_map(|x| *x).collect()
}

/// Mean +/- sd, or a dash where no subject could answer.
fn cell(v: &[f64]) -> String {
    if v.is_empty() { "     -      ".into() } else { format!("{:5.3} ±{:5.3}", mean(v), sd(v)) }
}

/// SHIPPED with the AWAKE base rate raised - more wake called, zero new information.
fn bumped(by: f64) -> Params {
    let mut p = Params::SHIPPED;
    p.base_rate[3] += by;
    p
}

fn main() {
    println!("epoch {EPOCH}s | wake bout >= {MIN_BOUT} epochs, found at >= {:.0}% overlap", MIN_OVERLAP * 100.0);
    println!("FIT: {}   HELD OUT: {}\n", FIT.join(", "), HELD_OUT.join(", "));

    let arms = arms();
    for cohort in FIT.iter().chain(&HELD_OUT) {
        let nights: Vec<Night> = dirs_of(cohort).iter().filter_map(|d| load(d)).collect();
        if nights.is_empty() {
            println!("=== {cohort}: no labelled nights under the fixture root\n");
            continue;
        }
        let preps: Vec<Prepared> =
            nights.iter().map(|n| prepare_v2(&n.input, &Params::SHIPPED)).collect();

        let role = if FIT.contains(cohort) { "FIT" } else { "HELD OUT" };
        println!("=== {cohort} ({role}), {} labelled subjects", nights.len());
        println!(
            "{:<28} {:>12} {:>12} {:>12} {:>12} {:>7}",
            "arm", "kappa4", "wake recall", "wake spec", "bout COVERAGE", "bouts"
        );

        // Per-subject scores for every arm, kept so a paired delta can be taken below.
        let mut per_arm: Vec<Vec<Subject>> = Vec::new();
        for (_, arm) in &arms {
            let subs: Vec<Subject> = nights
                .iter()
                .zip(&preps)
                .enumerate()
                .map(|(i, (night, prep))| {
                    let pred = match arm {
                        Arm::Recipe(p) => stage_labels(night, prep, p),
                        Arm::Fixed(s) => vec![*s; night.n],
                        Arm::Shuffle => {
                            shuffled(&stage_labels(night, prep, &Params::SHIPPED), i as u64 + 1)
                        }
                        Arm::Rescue(f) => rescued(
                            &stage_labels(night, prep, &Params::SHIPPED),
                            &night.truth,
                            *f,
                        ),
                    };
                    score_subject(&pred, &night.truth)
                })
                .collect();
            per_arm.push(subs);
        }

        for ((name, _), subs) in arms.iter().zip(&per_arm) {
            let k: Vec<f64> = subs.iter().map(|s| s.kappa).collect();
            let wr = some(&subs.iter().map(|s| s.wake_recall).collect::<Vec<_>>());
            let ws = some(&subs.iter().map(|s| s.wake_spec).collect::<Vec<_>>());
            let br = some(&subs.iter().map(|s| s.bout.coverage()).collect::<Vec<_>>());
            let bouts: usize = subs.iter().map(|s| s.bout.truth_bouts).sum();
            println!(
                "{:<28} {} {} {} {} {:>7}",
                name,
                cell(&k),
                cell(&wr),
                cell(&ws),
                cell(&br),
                bouts
            );
        }

        // What size of change this cohort can resolve. Subjects are their own controls, so this is the
        // sigma of the per-subject DELTA, not of the absolute kappa - the two differ by an order of
        // magnitude and quoting the second at a paired comparison is how a workable bar gets called noise.
        // A null arm decorrelates from shipped and would overstate it, so both references are recipes.
        let n = nights.len();
        println!("\n  paired against shipped, per subject (n={n}):");
        println!("{:<44} {:>9} {:>9} {:>13}", "  reference", "mean d", "sd d", "resolvable ±");
        for i in [SMALL_ARM, REAL_ARM, CAND_ARM] {
            let d: Vec<f64> =
                per_arm[SHIPPED_ARM].iter().zip(&per_arm[i]).map(|(a, b)| b.kappa - a.kappa).collect();
            println!(
                "{:<44} {:>+9.4} {:>9.4} {:>13.4}",
                format!("  {} (kappa)", arms[i].0),
                mean(&d),
                sd(&d),
                1.96 * sd(&d) / (n as f64).sqrt()
            );
        }
        // The headline moves on a different scale from kappa, so it needs its own bar.
        let bd: Vec<f64> = per_arm[SHIPPED_ARM]
            .iter()
            .zip(&per_arm[CAND_ARM])
            .filter_map(|(a, b)| Some(b.bout.coverage()? - a.bout.coverage()?))
            .collect();
        println!(
            "{:<44} {:>+9.4} {:>9.4} {:>13.4}",
            format!("  {} (bout coverage)", arms[CAND_ARM].0),
            mean(&bd),
            sd(&bd),
            1.96 * sd(&bd) / (bd.len() as f64).sqrt()
        );

        // The calibration ladder. Each rung recovers a KNOWN fraction of the wake we miss, so the first
        // rung that clears its own bar is the smallest real improvement this cohort can prove.
        println!("\n  the rescue ladder — how much missed wake must a feature find to be provable?");
        println!("{:<24} {:>9} {:>13} {:>7}   {:>9} {:>13} {:>7}",
            "  recovered", "d kappa", "resolvable ±", "seen?", "d bout", "resolvable ±", "seen?");
        let first_rescue = arms.len() - LADDER.len();
        for (r, frac) in (first_rescue..arms.len()).zip(LADDER) {
            let dk: Vec<f64> =
                per_arm[SHIPPED_ARM].iter().zip(&per_arm[r]).map(|(a, b)| b.kappa - a.kappa).collect();
            let db: Vec<f64> = per_arm[SHIPPED_ARM]
                .iter()
                .zip(&per_arm[r])
                .filter_map(|(a, b)| Some(b.bout.coverage()? - a.bout.coverage()?))
                .collect();
            let (rk, rb) =
                (1.96 * sd(&dk) / (n as f64).sqrt(), 1.96 * sd(&db) / (db.len().max(1) as f64).sqrt());
            println!("{:<24} {:>+9.4} {:>13.4} {:>7}   {:>+9.4} {:>13.4} {:>7}",
                format!("  {:.0}% of missed wake", frac * 100.0),
                mean(&dk), rk, if mean(&dk).abs() > rk { "yes" } else { "NO" },
                mean(&db), rb, if mean(&db).abs() > rb { "yes" } else { "NO" });
        }

        // Named, so a fix can be tried against the subjects it is supposed to help rather than the mean.
        let mut worst: Vec<(f64, &str, usize)> = nights
            .iter()
            .zip(&per_arm[0])
            .filter_map(|(n, s)| s.bout.recall().map(|r| (r, n.id.as_str(), s.bout.truth_bouts)))
            .collect();
        worst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.2.cmp(&a.2)));
        let worst: Vec<String> =
            worst.iter().take(3).map(|(r, id, b)| format!("{id} {:.2} ({b} bouts)", r)).collect();
        println!("  worst wake-bout recall under shipped: {}\n", worst.join(" · "));
    }
}
