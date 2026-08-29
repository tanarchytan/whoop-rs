//! Step 4 of the V2 sleep stager: the time-of-night cycle prior and the anchor it is measured from.
//!
//!   cargo run --release -p physio-algo --example step4_cycle
//!
//! Rebuilds the prior outside the stager so every variant shares one set of prior-free emissions, then
//! scores each arm as a paired per-night kappa difference against the shipped recipe on the same nights.
//! Nothing is written and no shipped behaviour is touched; this only measures.

mod common;

use common::{dirs_of, median, onset_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};

use physio_algo::sleep::metrics::{bouts, confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emissions_v2, epoch_starts_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const EPOCH_MIN: f64 = 0.5;
/// Column of each stage in the emission rows, asserted against [`STAGE_ORDER`] in `main`.
const COL_DEEP: usize = 0;
const COL_REM: usize = 1;
/// `stage_idx` classes, which are NOT the emission columns.
const DEEP_CLASS: usize = 2;
const REM_CLASS: usize = 3;
/// Consecutive REM epochs the prior-free decode must hold before it counts as a bout.
const REM_BOUT: usize = 6;
/// Consecutive non-wake epochs that mark onset, mirroring the stager's own rule.
const SUSTAINED: usize = 10;
/// Cap on the anchor fixed-point iteration before a night is called non-convergent.
const FIXED_ITERS: usize = 24;

// ---------------------------------------------------------------------------------------------
// The prior, rebuilt outside the stager.
// ---------------------------------------------------------------------------------------------

/// Which time-of-night the prior reads. `Window` is shipped: a fraction of the detection window.
#[derive(Clone, Copy, PartialEq)]
enum Clock {
    Window,
    FromOnset,
    /// Minutes since the anchor over a fixed horizon, so a short and a long night share a scale.
    Abs(f64),
}

/// The early-REM suppression's shape. `Ramp` is shipped; `Step` is the window-anchored fallback.
#[derive(Clone, Copy, PartialEq)]
enum Guard {
    Off,
    Step(f64, f64),
    Ramp(f64, f64),
    Exp(f64, f64),
}

/// How the REM term rises. `SelfPhase` reads the prior-free decode's own REM bouts instead.
#[derive(Clone, Copy, PartialEq)]
enum RemShape {
    Ramp,
    SelfPhase(f64),
}

/// Where the anchor comes from. `Probe` is shipped. `Truth` is an oracle and never selectable.
#[derive(Clone, Copy, PartialEq)]
enum AnchorPick {
    Probe,
    Free,
    FreeRem,
    Fixed,
    Truth,
    Shift(i64),
}

#[derive(Clone, Copy)]
struct Arm {
    deep_scale: f64,
    deep_decay: f64,
    rem_scale: f64,
    rem_cap: f64,
    clock: Clock,
    guard: Guard,
    rem: RemShape,
    anchor: AnchorPick,
}

fn shipped_arm() -> Arm {
    let p = Params::SHIPPED;
    Arm {
        deep_scale: p.cycle_deep_scale,
        deep_decay: p.cycle_deep_decay,
        rem_scale: p.cycle_rem_scale,
        rem_cap: p.cycle_rem_ramp_cap,
        clock: Clock::Window,
        guard: Guard::Ramp(p.cycle_rem_onset_minutes, p.cycle_rem_early_penalty),
        rem: RemShape::Ramp,
        anchor: AnchorPick::Probe,
    }
}

/// A bump in [0, 1] peaking on each prior-free REM epoch, the self-phase REM term's shape.
fn self_phase(e: usize, free_rem: &[usize], sigma_min: f64) -> f64 {
    let s = sigma_min / EPOCH_MIN;
    free_rem
        .iter()
        .map(|j| {
            let d = (e as f64 - *j as f64) / s;
            (-0.5 * d * d).exp()
        })
        .fold(0.0f64, f64::max)
}

/// The additive per-epoch prior one arm puts on the prior-free emissions, anchored at epoch `o`.
fn prior_at(clock: &[f64], free_rem: &[usize], arm: &Arm, o: usize) -> Vec<[f64; CLASSES]> {
    let c0 = clock.get(o).copied().unwrap_or(0.0);
    clock
        .iter()
        .enumerate()
        .map(|(e, &cw)| {
            let c = match arm.clock {
                Clock::Window => cw,
                Clock::FromOnset => {
                    if c0 >= 1.0 {
                        cw
                    } else {
                        ((cw - c0) / (1.0 - c0)).clamp(0.0, 1.0)
                    }
                }
                Clock::Abs(t) => ((e as f64 - o as f64) * EPOCH_MIN / t).clamp(0.0, 1.0),
            };
            let mins = (e as f64 - o as f64) * EPOCH_MIN;
            let g = match arm.guard {
                Guard::Off => 0.0,
                Guard::Step(frac, pen) => {
                    if cw < frac {
                        pen
                    } else {
                        0.0
                    }
                }
                Guard::Ramp(w, pen) => pen * (1.0 - mins / w).clamp(0.0, 1.0),
                Guard::Exp(w, pen) => pen * (-mins.max(0.0) / w).exp(),
            };
            let rise = match arm.rem {
                RemShape::Ramp => c.min(arm.rem_cap),
                RemShape::SelfPhase(sig) => self_phase(e, free_rem, sig),
            };
            let mut pr = [0.0; CLASSES];
            pr[COL_DEEP] = arm.deep_scale * (1.0 - c / arm.deep_decay).max(0.0);
            pr[COL_REM] = arm.rem_scale * rise - g;
            pr
        })
        .collect()
}

fn add_prior(em0: &[[f64; CLASSES]], pr: &[[f64; CLASSES]]) -> Vec<[f64; CLASSES]> {
    em0.iter()
        .zip(pr)
        .map(|(e, p)| {
            let mut out = *e;
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
            out
        })
        .collect()
}

fn decoded(em: &[[f64; CLASSES]]) -> Vec<usize> {
    decode_v2(em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect()
}

/// Per-epoch argmax of the emissions, the decoder switched off.
fn argmaxed(em: &[[f64; CLASSES]]) -> Vec<usize> {
    em.iter()
        .map(|r| {
            let mut best = 0usize;
            for c in 1..CLASSES {
                if r[c] > r[best] {
                    best = c;
                }
            }
            stage_idx(STAGE_ORDER[best])
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Nights.
// ---------------------------------------------------------------------------------------------

struct Night {
    /// The shipped emissions with the cycle prior removed, straight out of the stager.
    em0: Vec<[f64; CLASSES]>,
    clock: Vec<f64>,
    truth: Vec<Option<usize>>,
    /// The onset `resolve_anchor` computes: the probe decode's first sustained non-wake run.
    probe_onset: usize,
    probe_found: bool,
    /// The same rule over the prior-free decode, and over the shipped final staging.
    free_onset: Option<usize>,
    final_onset: Option<usize>,
    free_rem: Vec<usize>,
    first_free_rem: Option<usize>,
    truth_onset: Option<usize>,
    fixed_onset: usize,
    /// Whether the anchor iteration reached a fixed point, its cycle length if not, and its passes.
    fixed_converged: bool,
    fixed_cycle: usize,
    fixed_steps: usize,
    span_min: f64,
    first_truth: usize,
    n: usize,
}

fn probe_params() -> Params {
    Params {
        cycle_rem_early_penalty: 0.0,
        cycle_rem_onset_minutes: 0.0,
        cycle_clock_from_onset: false,
        ..Params::SHIPPED
    }
}

/// The shipped recipe with the whole cycle prior zeroed, which is the emission every arm builds on.
fn zero_params() -> Params {
    Params { cycle_deep_scale: 0.0, cycle_rem_scale: 0.0, ..probe_params() }
}

/// First epoch of the earliest run of `SUSTAINED` labelled non-wake epochs; a hole resets the run.
fn truth_onset_of(truth: &[Option<usize>]) -> Option<usize> {
    let mut run = 0usize;
    for (i, t) in truth.iter().enumerate() {
        run = match t {
            Some(c) if *c != 0 => run + 1,
            _ => 0,
        };
        if run == SUSTAINED {
            return Some(i + 1 - SUSTAINED);
        }
    }
    None
}

/// Iterate the anchor to a fixed point under the shipped arm: the landing epoch, whether it settled,
/// the cycle length if it did not, and how many passes it took.
fn iterate_anchor(clock: &[f64], em0: &[[f64; CLASSES]], o0: usize) -> (usize, bool, usize, usize) {
    let arm = shipped_arm();
    let mut seen: Vec<usize> = vec![o0];
    let mut o = o0;
    for k in 1..=FIXED_ITERS {
        let lab = decoded(&add_prior(em0, &prior_at(clock, &[], &arm, o)));
        let next = onset_of(&lab).unwrap_or(0);
        if next == o {
            return (o, true, 1, k);
        }
        if let Some(pos) = seen.iter().position(|s| *s == next) {
            return (next, false, seen.len() - pos, k);
        }
        seen.push(next);
        o = next;
    }
    (o, false, 0, FIXED_ITERS)
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
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em_shipped = emissions_v2(&prep, &Params::SHIPPED);
        if em_shipped.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em_shipped.len(), n, "{}: {n} truth epochs against {} emissions",
                   dir.display(), em_shipped.len());

        let starts = epoch_starts_v2(&prep);
        let span = (w1 - w0).max(1) as f64;
        let clock: Vec<f64> = starts.iter().map(|e| (e + 15 - w0) as f64 / span).collect();
        let em0 = emissions_v2(&prep, &zero_params());
        let probe = decoded(&emissions_v2(&prep, &probe_params()));
        let probe_found = onset_of(&probe).is_some();
        let probe_onset = onset_of(&probe).unwrap_or(0);

        let free = decoded(&em0);
        let free_onset = onset_of(&free);
        let rem_bouts = bouts(&free, REM_CLASS, REM_BOUT);
        let free_rem: Vec<usize> = rem_bouts.iter().flat_map(|(a, len)| *a..*a + *len).collect();
        let first_free_rem = rem_bouts.first().map(|(a, _)| *a);
        let final_onset = onset_of(&decoded(&em_shipped));

        let truth: Vec<Option<usize>> = (0..n)
            .map(|k| raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize))
            .collect();
        if truth.iter().filter(|t| t.is_some()).count() < MIN_EPOCHS {
            continue;
        }
        let first_truth = truth.iter().position(|t| t.is_some()).unwrap_or(0);
        let truth_onset = truth_onset_of(&truth);
        let (fixed_onset, fixed_converged, fixed_cycle, fixed_steps) =
            iterate_anchor(&clock, &em0, probe_onset);

        // The one control that makes every arm below meaningful: the prior rebuilt out here, added
        // back onto the stager's own prior-free emissions, must BE the shipped emissions.
        let rebuilt = add_prior(&em0, &prior_at(&clock, &free_rem, &shipped_arm(), probe_onset));
        for (a, b) in rebuilt.iter().zip(&em_shipped) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-9, "{}: rebuilt prior is not the shipped one", dir.display());
            }
        }
        // And the window-anchored step guard, which needs no probe at all, is the stager's own
        // `cycle_rem_onset_minutes = 0` path, so the arm below is a real parameter and not a fiction.
        let step_arm = Arm {
            guard: Guard::Step(Params::SHIPPED.cycle_rem_early_frac, Params::SHIPPED.cycle_rem_early_penalty),
            ..shipped_arm()
        };
        let mine = add_prior(&em0, &prior_at(&clock, &free_rem, &step_arm, probe_onset));
        let theirs = emissions_v2(&prep, &Params { cycle_rem_onset_minutes: 0.0, ..Params::SHIPPED });
        for (a, b) in mine.iter().zip(&theirs) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-9, "{}: the step-guard arm is not the stager's own", dir.display());
            }
        }
        // Mutate `cycle_rem_early_frac` to both extremes under SHIPPED. The onset-anchored guard is
        // the first match arm, so the window-clock step below it is unreachable and this must not move.
        for frac in [0.0, 1.0] {
            let mutated = emissions_v2(&prep, &Params { cycle_rem_early_frac: frac, ..Params::SHIPPED });
            assert_eq!(mutated, em_shipped, "{}: cycle_rem_early_frac {frac} moved the emission",
                       dir.display());
        }

        out.push(Night {
            em0,
            clock,
            truth,
            probe_onset,
            probe_found,
            free_onset,
            final_onset,
            free_rem,
            first_free_rem,
            truth_onset,
            fixed_onset,
            fixed_converged,
            fixed_cycle,
            fixed_steps,
            span_min: span / 60.0,
            first_truth,
            n,
        });
    }
    out
}

fn anchor_of(n: &Night, a: AnchorPick) -> usize {
    match a {
        AnchorPick::Probe => n.probe_onset,
        AnchorPick::Free => n.free_onset.unwrap_or(0),
        AnchorPick::FreeRem => n.first_free_rem.unwrap_or(n.probe_onset),
        AnchorPick::Fixed => n.fixed_onset,
        AnchorPick::Truth => n.truth_onset.unwrap_or(n.probe_onset),
        AnchorPick::Shift(d) => (n.probe_onset as i64 + d).clamp(0, n.n as i64 - 1) as usize,
    }
}

fn labels_of(n: &Night, arm: &Arm) -> Vec<usize> {
    let pr = prior_at(&n.clock, &n.free_rem, arm, anchor_of(n, arm.anchor));
    decoded(&add_prior(&n.em0, &pr))
}

fn argmax_of(n: &Night, arm: &Arm) -> Vec<usize> {
    let pr = prior_at(&n.clock, &n.free_rem, arm, anchor_of(n, arm.anchor));
    argmaxed(&add_prior(&n.em0, &pr))
}

fn kappa_of(n: &Night, lab: &[usize]) -> f64 {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in n.truth.iter().enumerate() {
        if let Some(w) = want {
            p.push(lab[k]);
            t.push(*w);
        }
    }
    kappa4(&confusion4(&p, &t))
}

// ---------------------------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------------------------

fn verdict(mean: f64, bar: f64) -> String {
    if !mean.is_finite() || !bar.is_finite() {
        "-".to_string()
    } else if mean.abs() > bar {
        format!("{} ({:.2}x)", if mean > 0.0 { "BEATS v2" } else { "worse" }, mean.abs() / bar)
    } else {
        "matches".to_string()
    }
}

fn row(label: &str, d: &[f64], k: &[f64]) -> String {
    let (mean, bar) = paired_bar(d).unwrap_or((f64::NAN, f64::NAN));
    format!(
        "  {label:<34} {:>7.3} {:>+9.4} {:>8.4} {:>4}   {}",
        median(&mut k.to_vec()),
        mean,
        bar,
        d.len(),
        verdict(mean, bar)
    )
}

/// Every candidate's per-night kappa on one cohort.
fn kappas(nights: &[Night], arm: &Arm) -> Vec<f64> {
    nights.iter().map(|n| kappa_of(n, &labels_of(n, arm))).collect()
}

fn deltas(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn pooled_mean(sets: &[&Vec<f64>]) -> f64 {
    let (s, n): (f64, usize) = sets.iter().fold((0.0, 0), |(s, n), v| (s + v.iter().sum::<f64>(), n + v.len()));
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

struct Cohort {
    name: &'static str,
    nights: Vec<Night>,
    base: Vec<f64>,
}

/// The selection score of one candidate over a set of cohorts: pooled by night, or by cohort mean.
/// dreamt carries 100 of the 144 nights, so the two rules are not the same question.
fn select_score(sets: &[&Vec<f64>], balanced: bool) -> f64 {
    if balanced {
        let m: Vec<f64> = sets.iter().map(|v| pooled_mean(&[v])).filter(|v| v.is_finite()).collect();
        pooled_mean(&[&m])
    } else {
        pooled_mean(sets)
    }
}

/// Nested selection: pick inside the training cohorts only, then report once on the held-out one.
/// The inner line is the same procedure run one cohort deeper, so the selection has its own estimate.
fn nested(family: &str, cands: &[(String, Arm)], cohorts: &[Cohort], table: &[Vec<Vec<f64>>]) {
    println!("\n=== {family} — nested selection, {} candidates ===", cands.len());
    println!("  {:<9} {:<8} {:<40} {:>10} {:>9} {:>4}   {:<16} inner: select on one train cohort, score the other",
             "held out", "rule", "selected on the training pair", "paired d", "bar +/-", "n", "verdict");
    for (h, held) in cohorts.iter().enumerate() {
        let train: Vec<usize> = (0..cohorts.len()).filter(|i| *i != h).collect();
        for (balanced, rule) in [(false, "pooled"), (true, "balanced")] {
            let pick = (0..cands.len())
                .max_by(|a, b| {
                    let ma = select_score(&train.iter().map(|t| &table[*a][*t]).collect::<Vec<_>>(), balanced);
                    let mb = select_score(&train.iter().map(|t| &table[*b][*t]).collect::<Vec<_>>(), balanced);
                    ma.total_cmp(&mb)
                })
                .expect("at least one candidate");
            let d = &table[pick][h];
            let (mean, bar) = paired_bar(d).unwrap_or((f64::NAN, f64::NAN));

            let mut inner = Vec::new();
            for (si, sel) in train.iter().enumerate() {
                let other = train[1 - si];
                let ip = (0..cands.len())
                    .max_by(|a, b| {
                        pooled_mean(&[&table[*a][*sel]]).total_cmp(&pooled_mean(&[&table[*b][*sel]]))
                    })
                    .expect("at least one candidate");
                let (m, b) = paired_bar(&table[ip][other]).unwrap_or((f64::NAN, f64::NAN));
                inner.push(format!("{}->{} {:+.4}+/-{:.4}", cohorts[*sel].name, cohorts[other].name, m, b));
            }
            println!(
                "  {:<9} {:<8} {:<40} {mean:>+10.4} {bar:>9.4} {:>4}   {:<16} {}",
                held.name,
                rule,
                cands[pick].0,
                d.len(),
                verdict(mean, bar),
                if balanced { String::new() } else { inner.join("   ") },
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------

fn main() {
    assert_eq!(stage_idx(STAGE_ORDER[COL_DEEP]), DEEP_CLASS, "emission column 0 is deep");
    assert_eq!(stage_idx(STAGE_ORDER[COL_REM]), REM_CLASS, "emission column 1 is rem");

    let mut cohorts: Vec<Cohort> = Vec::new();
    for name in COHORTS {
        let nights = load(name);
        if nights.is_empty() {
            println!("{name}: no nights under the fixture root");
            continue;
        }
        let base = kappas(&nights, &shipped_arm());
        cohorts.push(Cohort { name, nights, base });
    }
    if cohorts.len() < 3 {
        println!("need all three cohorts; stopping");
        return;
    }

    println!("The prior is rebuilt outside v2.rs and added back onto the stager's own prior-free");
    println!("emissions. Every night ASSERTS that this reproduces the shipped emissions to 1e-9, so an");
    println!("arm below differs from v2 in the prior alone.\n");
    for c in &cohorts {
        let mut k = c.base.clone();
        println!("  {:<14} n={:<4} v2 median kappa {:.3}", c.name, c.nights.len(), median(&mut k));
    }

    // ---- Q1 / Q4: the anchor ----------------------------------------------------------------
    println!("\n=== the anchor: what resolve_anchor's probe decode actually finds ===");
    println!("  {:<14} {:>8} {:>10} {:>11} {:>12} {:>11} {:>12} {:>10}",
             "cohort", "no run", "onset min", "truth min", "probe-truth", "final-probe", "not settled", "cycle>1");
    for c in &cohorts {
        let mut om: Vec<f64> = c.nights.iter().map(|n| n.probe_onset as f64 * EPOCH_MIN).collect();
        let mut tm: Vec<f64> =
            c.nights.iter().filter_map(|n| n.truth_onset).map(|o| o as f64 * EPOCH_MIN).collect();
        let mut pt: Vec<f64> = c
            .nights
            .iter()
            .filter_map(|n| n.truth_onset.map(|t| (n.probe_onset as f64 - t as f64) * EPOCH_MIN))
            .collect();
        let mut fp: Vec<f64> = c
            .nights
            .iter()
            .filter_map(|n| n.final_onset.map(|f| (f as f64 - n.probe_onset as f64) * EPOCH_MIN))
            .collect();
        println!(
            "  {:<14} {:>8} {:>10.1} {:>11.1} {:>12.1} {:>11.1} {:>12} {:>10}",
            c.name,
            c.nights.iter().filter(|n| !n.probe_found).count(),
            median(&mut om),
            median(&mut tm),
            median(&mut pt),
            median(&mut fp),
            c.nights.iter().filter(|n| !n.fixed_converged).count(),
            c.nights.iter().filter(|n| n.fixed_cycle > 1).count(),
        );
    }
    println!("  columns are medians in minutes: probe onset from window start, truth onset, their");
    println!("  difference, and how far the FINAL staging's own onset sits from the probe's.");

    let mut moved: Vec<f64> = Vec::new();
    let mut total = 0usize;
    let mut multi = 0usize;
    for c in &cohorts {
        for n in &c.nights {
            total += 1;
            multi += usize::from(n.fixed_steps > 1);
            if n.final_onset != Some(n.probe_onset) {
                moved.push((n.final_onset.unwrap_or(0) as f64 - n.probe_onset as f64) * EPOCH_MIN);
            }
        }
    }
    let worst = moved.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    println!("  the final staging disagrees with its own probe on {} of {total} nights, median {:.1} min,",
             moved.len(), median(&mut moved.clone()));
    println!("  worst {worst:.0} min; {multi} nights need more than one pass to reach a fixed point");

    // The anchor is only worth measuring if it moves labels at all: count the epochs it moves.
    println!("\n=== how many epochs the anchor and the guard actually move ===");
    println!("  {:<14} {:>16} {:>16} {:>16} {:>16}",
             "cohort", "oracle anchor", "anchor +20 min", "guard off", "guard as a step");
    let churn = |nights: &[Night], arm: &Arm| -> f64 {
        let (mut d, mut t) = (0usize, 0usize);
        for n in nights {
            let (a, b) = (labels_of(n, &shipped_arm()), labels_of(n, arm));
            t += a.len();
            d += a.iter().zip(&b).filter(|(x, y)| x != y).count();
        }
        100.0 * d as f64 / t.max(1) as f64
    };
    for c in &cohorts {
        println!(
            "  {:<14} {:>15.2}% {:>15.2}% {:>15.2}% {:>15.2}%",
            c.name,
            churn(&c.nights, &Arm { anchor: AnchorPick::Truth, ..shipped_arm() }),
            churn(&c.nights, &Arm { anchor: AnchorPick::Shift(40), ..shipped_arm() }),
            churn(&c.nights, &Arm { guard: Guard::Off, ..shipped_arm() }),
            churn(&c.nights, &Arm {
                guard: Guard::Step(Params::SHIPPED.cycle_rem_early_frac, Params::SHIPPED.cycle_rem_early_penalty),
                ..shipped_arm()
            }),
        );
    }

    // ---- Q2: the clock convention each cohort carries ----------------------------------------
    println!("\n=== the clock each cohort hands the prior ===");
    println!("  {:<14} {:>10} {:>14} {:>16} {:>16}", "cohort", "span min", "unlabelled %", "lead-in % of span", "probe onset % span");
    for c in &cohorts {
        let mut span: Vec<f64> = c.nights.iter().map(|n| n.span_min).collect();
        let mut unl: Vec<f64> = c
            .nights
            .iter()
            .map(|n| 100.0 * n.truth.iter().filter(|t| t.is_none()).count() as f64 / n.n as f64)
            .collect();
        let mut lead: Vec<f64> =
            c.nights.iter().map(|n| 100.0 * n.first_truth as f64 / n.n as f64).collect();
        let mut po: Vec<f64> =
            c.nights.iter().map(|n| 100.0 * n.probe_onset as f64 / n.n as f64).collect();
        println!("  {:<14} {:>10.1} {:>14.1} {:>16.1} {:>16.1}",
                 c.name, median(&mut span), median(&mut unl), median(&mut lead), median(&mut po));
    }

    // ---- the fixed arms ----------------------------------------------------------------------
    let ship = shipped_arm();
    let fixed_arms: Vec<(String, Arm)> = vec![
        ("shipped (control)".into(), ship),
        ("NULL: no cycle prior".into(), Arm { deep_scale: 0.0, rem_scale: 0.0, guard: Guard::Off, ..ship }),
        ("deep term only".into(), Arm { rem_scale: 0.0, guard: Guard::Off, ..ship }),
        ("rem term only".into(), Arm { deep_scale: 0.0, ..ship }),
        ("guard off".into(), Arm { guard: Guard::Off, ..ship }),
        ("rem ramp off, guard kept".into(), Arm { rem_scale: 0.0, ..ship }),
        ("guard as a hard step".into(), Arm {
            guard: Guard::Step(Params::SHIPPED.cycle_rem_early_frac, Params::SHIPPED.cycle_rem_early_penalty),
            ..ship
        }),
        ("clock rebased on onset".into(), Arm { clock: Clock::FromOnset, ..ship }),
        ("anchor: prior-free onset".into(), Arm { anchor: AnchorPick::Free, ..ship }),
        ("anchor: prior-free 1st REM".into(), Arm { anchor: AnchorPick::FreeRem, ..ship }),
        ("anchor: fixed point".into(), Arm { anchor: AnchorPick::Fixed, ..ship }),
        ("anchor: TRUTH (oracle)".into(), Arm { anchor: AnchorPick::Truth, ..ship }),
        ("anchor: probe -10 min".into(), Arm { anchor: AnchorPick::Shift(-20), ..ship }),
        ("anchor: probe +10 min".into(), Arm { anchor: AnchorPick::Shift(20), ..ship }),
    ];

    println!("\n=== fixed arms, paired against v2 per cohort ===");
    for c in &cohorts {
        println!("\n  --- {} n={} ---", c.name, c.nights.len());
        println!("  {:<34} {:>7} {:>9} {:>8} {:>4}   verdict", "arm", "median", "paired d", "bar +/-", "n");
        for (name, arm) in &fixed_arms {
            let k = kappas(&c.nights, arm);
            println!("{}", row(name, &deltas(&k, &c.base), &k));
        }
    }

    // ---- placement, which kappa can miss ------------------------------------------------------
    println!("\n=== first-REM latency from window start, median minutes ===");
    println!("  {:<14} {:>8} {:>12} {:>12} {:>12} {:>12}",
             "cohort", "TRUTH", "shipped", "guard off", "guard step", "NULL");
    let lat_arms: [(&str, Arm); 4] = [
        ("shipped", ship),
        ("guard off", Arm { guard: Guard::Off, ..ship }),
        ("guard step", Arm {
            guard: Guard::Step(Params::SHIPPED.cycle_rem_early_frac, Params::SHIPPED.cycle_rem_early_penalty),
            ..ship
        }),
        ("null", Arm { deep_scale: 0.0, rem_scale: 0.0, guard: Guard::Off, ..ship }),
    ];
    // Measured from the first LABELLED epoch, so DREAMT's unlabelled lead-in cannot inflate it, and a
    // prediction before the scoring starts is unverifiable and does not count.
    for from_labels in [false, true] {
        if from_labels {
            println!("  ... the same, measured from each night's FIRST LABELLED epoch:");
        }
        for c in &cohorts {
            let base = |n: &Night| if from_labels { n.first_truth } else { 0 };
            let mut truth_lat: Vec<f64> = c
                .nights
                .iter()
                .filter_map(|n| {
                    n.truth.iter().position(|t| *t == Some(REM_CLASS)).map(|k| (k - base(n)) as f64 * EPOCH_MIN)
                })
                .collect();
            let cols: Vec<String> = lat_arms
                .iter()
                .map(|(_, a)| {
                    let mut v: Vec<f64> = c
                        .nights
                        .iter()
                        .filter_map(|n| {
                            let b = base(n);
                            labels_of(n, a)[b..].iter().position(|s| *s == REM_CLASS).map(|k| k as f64 * EPOCH_MIN)
                        })
                        .collect();
                    format!("{:.1}", median(&mut v))
                })
                .collect();
            println!("  {:<14} {:>8.1} {:>12} {:>12} {:>12} {:>12}",
                     c.name, median(&mut truth_lat), cols[0], cols[1], cols[2], cols[3]);
        }
    }

    // ---- emission or decoder? -----------------------------------------------------------------
    println!("\n=== is the prior's value realised in the emission or in the decoder? ===");
    println!("  {:<14} {:>16} {:>16}   NULL minus v2, argmax against decoded", "cohort", "argmax d", "decoded d");
    let null = Arm { deep_scale: 0.0, rem_scale: 0.0, guard: Guard::Off, ..ship };
    for c in &cohorts {
        let base_am: Vec<f64> = c.nights.iter().map(|n| kappa_of(n, &argmax_of(n, &ship))).collect();
        let null_am: Vec<f64> = c.nights.iter().map(|n| kappa_of(n, &argmax_of(n, &null))).collect();
        let null_de = kappas(&c.nights, &null);
        let (ma, ba) = paired_bar(&deltas(&null_am, &base_am)).unwrap_or((f64::NAN, f64::NAN));
        let (md, bd) = paired_bar(&deltas(&null_de, &c.base)).unwrap_or((f64::NAN, f64::NAN));
        println!("  {:<14} {ma:>+10.4} {ba:>5.4} {md:>+10.4} {bd:>5.4}", c.name);
    }

    // ---- the selected families -----------------------------------------------------------------
    let mut families: Vec<(&str, Vec<(String, Arm)>)> = Vec::new();

    let mut scale = Vec::new();
    for m in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5] {
        scale.push((
            format!("prior x{m}"),
            Arm { deep_scale: ship.deep_scale * m, rem_scale: ship.rem_scale * m, ..ship },
        ));
    }
    families.push(("prior magnitude (x0 IS the null)", scale));

    let mut clocks = vec![("window (shipped)".to_string(), ship)];
    clocks.push(("rebased on onset".to_string(), Arm { clock: Clock::FromOnset, ..ship }));
    for t in [180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 600.0] {
        clocks.push((format!("absolute {t:.0} min horizon"), Arm { clock: Clock::Abs(t), ..ship }));
    }
    families.push(("the clock", clocks));

    let mut guards = vec![("off".to_string(), Arm { guard: Guard::Off, ..ship })];
    guards.push((
        "hard step at 12% of window".to_string(),
        Arm { guard: Guard::Step(0.12, 4.0), ..ship },
    ));
    for w in [30.0, 60.0, 90.0, 120.0, 180.0] {
        guards.push((format!("linear ramp {w:.0} min"), Arm { guard: Guard::Ramp(w, 4.0), ..ship }));
    }
    for w in [20.0, 40.0, 60.0] {
        guards.push((format!("exponential {w:.0} min"), Arm { guard: Guard::Exp(w, 4.0), ..ship }));
    }
    for pen in [2.0, 6.0, 8.0] {
        guards.push((format!("linear ramp 60 min, penalty {pen}"), Arm { guard: Guard::Ramp(60.0, pen), ..ship }));
    }
    families.push(("the early-REM guard", guards));

    let mut deep = Vec::new();
    for d in [0.25, 0.40, 0.55, 0.70, 1.00, 2.00] {
        for s in [0.6, 1.2, 1.8] {
            deep.push((format!("deep decay {d} scale {s}"), Arm { deep_decay: d, deep_scale: s, ..ship }));
        }
    }
    families.push(("the deep decay", deep));

    let mut remf = vec![("monotone ramp (shipped)".to_string(), ship)];
    for cap in [0.4, 0.6, 0.8] {
        remf.push((format!("ramp capped at {cap}"), Arm { rem_cap: cap, ..ship }));
    }
    for sig in [20.0, 30.0, 45.0, 60.0] {
        remf.push((
            format!("self-phase from prior-free REM, sigma {sig:.0} min"),
            Arm { rem: RemShape::SelfPhase(sig), ..ship },
        ));
    }
    for s in [0.5, 1.5, 2.0] {
        remf.push((format!("ramp scale {s}"), Arm { rem_scale: s, ..ship }));
    }
    families.push(("the REM term's shape", remf));

    let anchors: Vec<(String, Arm)> = vec![
        ("probe (shipped)".into(), ship),
        ("prior-free decode's onset".into(), Arm { anchor: AnchorPick::Free, ..ship }),
        ("prior-free decode's 1st REM".into(), Arm { anchor: AnchorPick::FreeRem, ..ship }),
        ("iterated to a fixed point".into(), Arm { anchor: AnchorPick::Fixed, ..ship }),
        ("probe -20 min".into(), Arm { anchor: AnchorPick::Shift(-40), ..ship }),
        ("probe -10 min".into(), Arm { anchor: AnchorPick::Shift(-20), ..ship }),
        ("probe +10 min".into(), Arm { anchor: AnchorPick::Shift(20), ..ship }),
        ("probe +20 min".into(), Arm { anchor: AnchorPick::Shift(40), ..ship }),
    ];
    families.push(("the anchor", anchors));

    for (name, cands) in &families {
        let table: Vec<Vec<Vec<f64>>> = cands
            .iter()
            .map(|(_, a)| cohorts.iter().map(|c| deltas(&kappas(&c.nights, a), &c.base)).collect())
            .collect();
        nested(name, cands, &cohorts, &table);
        println!("  per-cohort mean paired d for every candidate:");
        println!("  {:<44} {:>12} {:>12} {:>12}", "candidate", cohorts[0].name, cohorts[1].name, cohorts[2].name);
        for (i, (label, _)) in cands.iter().enumerate() {
            let m: Vec<String> = (0..cohorts.len())
                .map(|c| {
                    let (mean, bar) = paired_bar(&table[i][c]).unwrap_or((f64::NAN, f64::NAN));
                    format!("{mean:>+7.4}{}", if mean.abs() > bar { "*" } else { " " })
                })
                .collect();
            println!("  {label:<44} {:>12} {:>12} {:>12}", m[0], m[1], m[2]);
        }
    }

    println!("\n* marks a candidate whose paired mean clears its own 95% bar on that cohort. A star on a");
    println!("cohort a candidate was SELECTED on is not evidence; only the held-out line above is.");
}
