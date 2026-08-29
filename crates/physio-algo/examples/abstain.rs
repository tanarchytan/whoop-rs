//! Refusing to label an epoch: whether it helps, what it is really detecting, and what it looks
//! like in a hypnogram.
//!
//!   cargo run --release -p physio-algo --example abstain
//!
//! Every epoch gets a stage today whether the evidence supports one or not. Ranking epochs by
//! confidence and keeping the top fraction raises kappa on what is kept - but so does dropping
//! epochs at random, so the arms below are all scored at MATCHED COVERAGE.
//!
//! Two nulls. RANDOM proves a drop is not free. NEAR A DECODED TRANSITION needs no emissions at
//! all, but a sticky decoder only moves state where the emission margin is wide, so epochs beside
//! an edge carry WIDER margins than the rest: refusing them refuses the high-confidence epochs,
//! which is close to the INVERSE of the margin rule rather than a near-identical null. Clearing
//! `vs edge` is the cheaper of the two results; random is the bar that still has to be cleared.
//!
//! Coverage is not the only cost. Scattered single-epoch holes are unusable in a hypnogram, so every
//! rule reports the run-length of what it refuses.

mod common;

use common::{dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Fraction of epochs KEPT. 1.00 is today's behaviour.
const COVERAGE: [f64; 5] = [0.95, 0.90, 0.80, 0.70, 0.60];
/// Fewest epochs a night must carry to load, and the fewest retained after abstention - a night
/// short enough to hit that floor is scored above the printed `keep`.
const MIN_EPOCHS: usize = 20;
/// Half-width, in epochs, of the smoothing applied to the margin before ranking. Smoothing is what
/// turns scattered single-epoch refusals into stretches a hypnogram can draw.
const SMOOTH: usize = 4;

/// How an epoch's willingness-to-answer is scored. Higher keeps.
#[derive(Clone, Copy, PartialEq)]
enum Rule {
    Random,
    FarFromTransition,
    Margin,
    SmoothMargin,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::Random => "random",
            Rule::FarFromTransition => "far-from-edge",
            Rule::Margin => "margin",
            Rule::SmoothMargin => "margin smoothed",
        }
    }
}

struct Night {
    pred: Vec<usize>,
    truth: Vec<Option<usize>>,
    margin: Vec<f64>,
    /// Epochs to the nearest decoded stage change, which lies BETWEEN two epochs, so the pair either
    /// side of it both score 0. Needs the path only, no emissions. A night the decoder never changes
    /// stage on scores every epoch [`f64::INFINITY`], so the index tie-break keeps the EARLIEST ones.
    to_edge: Vec<f64>,
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
        let input = SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        // `truth` below indexes `raw` positionally, which holds only while the grid is complete.
        // Checked BEFORE the length skip, or the hardest-collapsed grid is the one that leaves
        // silently instead of tripping it.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let pred: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let margin: Vec<f64> = em
            .iter()
            .map(|row| {
                let mut v = *row;
                v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                v[0] - v[1]
            })
            .collect();
        let edges: Vec<usize> =
            (1..pred.len()).filter(|k| pred[*k] != pred[k - 1]).collect();
        let to_edge = (0..pred.len())
            .map(|k| {
                let k = k as i64;
                edges
                    .iter()
                    .map(|e| (*e as i64 - k).abs().min((*e as i64 - 1 - k).abs()) as f64)
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { pred, truth, margin, to_edge });
    }
    out
}

/// Deterministic pseudo-random score, so two runs refuse the same epochs.
fn noise(seed: u64, k: usize) -> f64 {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(k as u64 + 1);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Centred mean of `v` over +/- `SMOOTH` epochs.
fn smooth(v: &[f64]) -> Vec<f64> {
    (0..v.len())
        .map(|k| {
            let lo = k.saturating_sub(SMOOTH);
            let hi = (k + SMOOTH + 1).min(v.len());
            v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect()
}

/// Per-epoch keep-score under one rule. Higher keeps.
fn rank(nt: &Night, ni: usize, rule: Rule) -> Vec<f64> {
    match rule {
        Rule::Random => (0..nt.pred.len()).map(|k| noise(ni as u64, k)).collect(),
        Rule::FarFromTransition => nt.to_edge.clone(),
        Rule::Margin => nt.margin.clone(),
        Rule::SmoothMargin => smooth(&nt.margin),
    }
}

/// Kappa per night on the kept epochs, and the run-lengths of what was refused.
fn scored(nights: &[Night], keep: f64, rule: Rule) -> (Vec<f64>, Vec<usize>) {
    let (mut ks, mut runs) = (Vec::new(), Vec::new());
    for (ni, nt) in nights.iter().enumerate() {
        let mut idx: Vec<usize> = (0..nt.pred.len()).filter(|k| nt.truth[*k].is_some()).collect();
        let score = rank(nt, ni, rule);
        // Ties broken by index so two rules with equal scores still refuse the same COUNT.
        idx.sort_by(|a, b| score[*b].partial_cmp(&score[*a]).unwrap().then(a.cmp(b)));
        let take = ((idx.len() as f64 * keep).round() as usize).max(MIN_EPOCHS).min(idx.len());
        if take < MIN_EPOCHS {
            continue;
        }
        let (p, t): (Vec<usize>, Vec<usize>) =
            idx[..take].iter().map(|k| (nt.pred[*k], nt.truth[*k].unwrap())).unzip();
        ks.push(kappa4(&confusion4(&p, &t)));

        // Run-lengths of the refusals, in epoch order, over the LABELLED epochs only: an epoch the
        // reference is silent on is absent from `idx`, so it splits one refusal stretch into two.
        let mut dropped: Vec<usize> = idx[take..].to_vec();
        dropped.sort_unstable();
        let mut i = 0;
        while i < dropped.len() {
            let mut j = i;
            while j + 1 < dropped.len() && dropped[j + 1] == dropped[j] + 1 {
                j += 1;
            }
            runs.push(j - i + 1);
            i = j + 1;
        }
    }
    (ks, runs)
}

fn verdict(mean: f64, bar: f64) -> String {
    if !mean.is_finite() {
        "-".into()
    } else if mean.abs() > bar {
        format!("{} {:.2}x", if mean > 0.0 { "BEATS" } else { "LOSES" }, mean.abs() / bar)
    } else {
        "inside the bar".into()
    }
}

fn main() {
    println!("Kappa on the epochs KEPT, every rule at MATCHED coverage, paired per night.");
    println!("`vs random` proves a drop is not free. `vs edge` needs no emissions, but a sticky");
    println!("decoder only changes state where the margin is wide, so epochs beside an edge carry");
    println!("WIDER margins: refusing them refuses the high-confidence epochs, close to the INVERSE");
    println!("of the margin rule, so clearing `vs edge` is the cheaper of the two results.");
    println!("`1-run%` is the share of refusal RUNS that are a LONE 30 s epoch - a hypnogram");
    println!("cannot draw those. Runs are counted over the epochs the reference LABELS, so a gap");
    println!("in truth inside a refusal stretch reads as two runs rather than one.\n");

    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("{set}: no nights\n");
            continue;
        }
        let (mut full, _) = scored(&nights, 1.0, Rule::Margin);
        let n_scored = full.len();
        println!("=== {set} ({n_scored} of {} nights scored), full-coverage kappa {:.3}",
                 nights.len(), median(&mut full));
        // A verdict group is 8 + 1 + 14 = 23 wide, 14 being the longest string `verdict` returns.
        println!("  {:>5} {:<16} {:>7} {:<23} {:<23} {:>7} {:>7}",
                 "keep", "rule", "kappa", "vs random", "vs edge", "1-run%", "runs/n");
        for keep in COVERAGE {
            let (rnd, _) = scored(&nights, keep, Rule::Random);
            let (edge, edge_runs) = scored(&nights, keep, Rule::FarFromTransition);
            for rule in [Rule::FarFromTransition, Rule::Margin, Rule::SmoothMargin] {
                let (mut k, runs) = if rule == Rule::FarFromTransition {
                    (edge.clone(), edge_runs.clone())
                } else {
                    scored(&nights, keep, rule)
                };
                let dr: Vec<f64> = rnd.iter().zip(&k).map(|(a, b)| b - a).collect();
                let (mr, br) = paired_bar(&dr).unwrap_or((f64::NAN, f64::NAN));
                let ones = runs.iter().filter(|r| **r == 1).count();
                let pct = 100.0 * ones as f64 / runs.len().max(1) as f64;
                let vs_edge = if rule == Rule::FarFromTransition {
                    "  (is the null)".to_string()
                } else {
                    let de: Vec<f64> = edge.iter().zip(&k).map(|(a, b)| b - a).collect();
                    let (me, be) = paired_bar(&de).unwrap_or((f64::NAN, f64::NAN));
                    format!("{me:>+8.4} {}", verdict(me, be))
                };
                // `runs/n` is per SCORED night; `k` holds one kappa per night that passed the guard.
                let scored_nights = k.len();
                let kappa = median(&mut k);
                println!("  {:>4.0}% {:<16} {:>7.3} {:>+8.4} {:<14} {:<23} {:>6.0}% {:>7.1}",
                         100.0 * keep, rule.name(), kappa, mr, verdict(mr, br), vs_edge, pct,
                         runs.len() as f64 / scored_nights.max(1) as f64);
            }
        }
        println!();
    }
}
