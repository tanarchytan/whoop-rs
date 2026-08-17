//! Where sleep starts and where it stops, against PSG truth on 144 nights.
//!
//!   cargo run --release -p physio-algo --example onset_offset
//!
//! Every previous attempt at the boundary was judged against one wearer's recollection. The PSG
//! cohorts carry the answer already: the hypnogram's first and last sustained sleep ARE the onset and
//! the offset, per subject, on 144 nights.
//!
//! Truth is the first/last epoch beginning a run of at least [`SUSTAINED_EPOCHS`] scored asleep, so a
//! single mislabelled epoch cannot move a boundary by an hour.
//!
//! Two arms, because they fail differently and the fix differs:
//!   detector - `detect_sessions` on the raw streams, i.e. what decides the window at all
//!   staged   - the first/last non-wake epoch `stage_v2` emits inside the reference window
//!
//! COVERAGE is reported beside error, and it is the number that matters first: a detector that is
//! accurate on the third of nights it fires for is not accurate, it is selective.

mod common;

use common::{dirs_of, mean, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx, stage_at};

use physio_algo::sleep::{
    detect_sessions, params::Params, prepare_v2, stage_v2_prepared, AccelSample, SleepInput,
};

const EPOCH: i64 = 30;
/// A boundary must open a run this long to count, so one stray epoch cannot define the night.
const SUSTAINED_EPOCHS: usize = 10;
/// The band the wearer-facing claim is stated in.
const TOLERANCES_MIN: [i64; 4] = [5, 10, 15, 30];
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// PSG cohorts are lab recordings with no local clock, so no daytime guard can apply.
const TZ_OFFSET_S: i64 = 0;

/// First and last epoch index opening a sustained sleep run. `None` when the night never sleeps.
fn truth_bounds(truth: &[Option<usize>]) -> Option<(usize, usize)> {
    let asleep: Vec<bool> = truth.iter().map(|t| matches!(t, Some(s) if *s != 0)).collect();
    let run_from = |i: usize| asleep[i..].iter().take(SUSTAINED_EPOCHS).all(|b| *b);
    let run_to = |i: usize| asleep[..=i].iter().rev().take(SUSTAINED_EPOCHS).all(|b| *b);
    let first = (0..asleep.len().saturating_sub(SUSTAINED_EPOCHS)).find(|i| run_from(*i))?;
    let last = (SUSTAINED_EPOCHS..asleep.len()).rev().find(|i| run_to(*i))?;
    Some((first, last))
}

struct Night {
    w0: i64,
    n: usize,
    truth: Vec<Option<usize>>,
    input: SleepInput,
    accel: Vec<AccelSample>,
}

fn load(dir: &std::path::Path) -> Option<Night> {
    let (w0, w1, n) = read_meta(dir)?;
    let raw = read_truth(dir);
    if raw.is_empty() {
        return None;
    }
    let mut truth = vec![None; n];
    for (k, t) in raw {
        if k < n && (0..4).contains(&t) {
            truth[k] = Some(t as usize);
        }
    }
    let accel = read_accel(dir);
    Some(Night {
        input: SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel: accel.clone() },
        w0,
        n,
        truth,
        accel,
    })
}

/// Absolute error in minutes, and the signed one so a systematic direction shows.
fn err_min(ours: i64, truth: i64) -> f64 {
    (ours - truth) as f64 / 60.0
}

fn pct(v: &[f64], tol: i64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    100.0 * v.iter().filter(|e| e.abs() <= tol as f64).count() as f64 / v.len() as f64
}

fn report(name: &str, n_total: usize, on: &[f64], off: &[f64]) {
    println!("\n  {name}: fired on {} of {n_total} nights ({:.0}% coverage)", on.len(),
        100.0 * on.len() as f64 / n_total.max(1) as f64);
    for (label, v) in [("onset", on), ("offset", off)] {
        if v.is_empty() {
            println!("    {label:<7} no nights");
            continue;
        }
        let mut s: Vec<f64> = v.iter().map(|x| x.abs()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Coverage-weighted: a night the arm skipped counts as a miss, or a selective arm scores high.
        let within: Vec<String> = TOLERANCES_MIN
            .iter()
            .map(|t| format!("+-{t}m {:.0}%", pct(v, *t) * v.len() as f64 / n_total as f64))
            .collect();
        println!("    {label:<7} bias {:+6.1}m  |err| median {:5.1}m  of ALL nights: {}",
            mean(v), s[s.len() / 2], within.join("  "));
    }
}

fn main() {
    println!("PSG truth = first/last epoch opening a {SUSTAINED_EPOCHS}-epoch sustained sleep run");
    println!("percentages are of ALL nights in the cohort, so skipping a night counts as a miss");

    for cohort in COHORTS {
        let nights: Vec<Night> = dirs_of(cohort).iter().filter_map(|d| load(d)).collect();
        if nights.is_empty() {
            continue;
        }
        println!("\n=== {cohort}, {} labelled nights", nights.len());

        let (mut d_on, mut d_off, mut s_on, mut s_off) = (vec![], vec![], vec![], vec![]);
        let mut no_truth = 0;
        for night in &nights {
            let Some((a, b)) = truth_bounds(&night.truth) else {
                no_truth += 1;
                continue;
            };
            let (t_on, t_off) = (night.w0 + a as i64 * EPOCH, night.w0 + b as i64 * EPOCH);

            // Arm 1: the detector, on the raw streams, with nothing else to lean on.
            let spans = detect_sessions(&night.input.hr, &night.accel, TZ_OFFSET_S, &[], &[], None);
            if let Some(sp) = spans.iter().max_by_key(|s| s.end - s.start) {
                d_on.push(err_min(sp.start, t_on));
                d_off.push(err_min(sp.end, t_off));
            }

            // Arm 2: staging inside the reference window, which is the best case the emission layer
            // can reach - the window is handed to it, so only the labels can be wrong.
            let prep = prepare_v2(&night.input, &Params::SHIPPED);
            let segs = stage_v2_prepared(&prep, &Params::SHIPPED);
            let lab: Vec<usize> = (0..night.n)
                .map(|k| {
                    let mid = night.w0 + k as i64 * EPOCH + EPOCH / 2;
                    stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage))
                })
                .collect();
            let sleep: Vec<bool> = lab.iter().map(|l| *l != 0).collect();
            let f = (0..sleep.len().saturating_sub(SUSTAINED_EPOCHS))
                .find(|i| sleep[*i..].iter().take(SUSTAINED_EPOCHS).all(|b| *b));
            let l = (SUSTAINED_EPOCHS..sleep.len())
                .rev()
                .find(|i| sleep[..=*i].iter().rev().take(SUSTAINED_EPOCHS).all(|b| *b));
            if let (Some(f), Some(l)) = (f, l) {
                s_on.push(err_min(night.w0 + f as i64 * EPOCH, t_on));
                s_off.push(err_min(night.w0 + l as i64 * EPOCH, t_off));
            }
        }
        if no_truth > 0 {
            println!("  {no_truth} night(s) never reach a sustained sleep run in truth");
        }
        report("detector (detect_sessions on raw streams)", nights.len(), &d_on, &d_off);
        report("staged   (stage_v2 inside the reference window)", nights.len(), &s_on, &s_off);
    }
}
