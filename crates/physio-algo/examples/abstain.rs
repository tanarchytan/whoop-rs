//! Refusing to label an epoch: whether it helps, what it is really detecting, and what it looks
//! like in a hypnogram.
//!
//!   cargo run --release -p physio-algo --example abstain
//!
//! Every epoch gets a stage today whether the evidence supports one or not. Ranking epochs by
//! confidence and keeping the top fraction raises kappa on what is kept - but so does dropping
//! epochs at random, so the arms below are all scored at MATCHED COVERAGE.
//!
//! Two nulls, and the second is the one that matters. RANDOM proves a drop is not free. NEAR A
//! DECODED TRANSITION is a heuristic needing no emissions at all, and a Viterbi decoder only moves
//! state where the emission margin is wide, so the two are nearly the same event by construction.
//! A confidence signal has to beat THAT, not random.
//!
//! Coverage is not the only cost. Scattered single-epoch holes are unusable in a hypnogram, so every
//! rule reports the run-length of what it refuses.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Fraction of epochs KEPT. 1.00 is today's behaviour.
const COVERAGE: [f64; 5] = [0.95, 0.90, 0.80, 0.70, 0.60];
/// Fewest retained epochs before a night is scored at all.
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
    /// Epochs to the nearest decoded stage change. Needs the path only, no emissions.
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
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
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
                edges
                    .iter()
                    .map(|e| (*e as i64 - k as i64).abs() as f64)
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

        // Run-lengths of the refusals, in epoch order.
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

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
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
    println!("`vs random` proves a drop is not free. `vs edge` is the null that matters: refusing");
    println!("near a decoded transition needs no emissions, and a decoder only changes state where");
    println!("the margin is wide, so the two are nearly the same event.");
    println!("`1-run%` is the share of refusals that are a LONE 30 s epoch - a hypnogram cannot");
    println!("draw those.\n");

    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("{set}: no nights\n");
            continue;
        }
        let (full, _) = scored(&nights, 1.0, Rule::Margin);
        println!("=== {set} ({} nights), full-coverage kappa {:.3}", nights.len(), median(&full));
        println!("  {:>5} {:<16} {:>7} {:>16} {:>16} {:>7} {:>7}",
                 "keep", "rule", "kappa", "vs random", "vs edge", "1-run%", "runs/n");
        for keep in COVERAGE {
            let (rnd, _) = scored(&nights, keep, Rule::Random);
            let (edge, _) = scored(&nights, keep, Rule::FarFromTransition);
            for rule in [Rule::FarFromTransition, Rule::Margin, Rule::SmoothMargin] {
                let (k, runs) = scored(&nights, keep, rule);
                let dr: Vec<f64> = rnd.iter().zip(&k).map(|(a, b)| b - a).collect();
                let de: Vec<f64> = edge.iter().zip(&k).map(|(a, b)| b - a).collect();
                let (mr, br) = paired_bar(&dr).unwrap_or((f64::NAN, f64::NAN));
                let (me, be) = paired_bar(&de).unwrap_or((f64::NAN, f64::NAN));
                let ones = runs.iter().filter(|r| **r == 1).count();
                let pct = 100.0 * ones as f64 / runs.len().max(1) as f64;
                let vs_edge = if rule == Rule::FarFromTransition {
                    "  (is the null)".to_string()
                } else {
                    format!("{me:>+8.4} {}", verdict(me, be))
                };
                println!("  {:>4.0}% {:<16} {:>7.3} {:>+8.4} {:<7} {:>16} {:>6.0}% {:>7.1}",
                         100.0 * keep, rule.name(), median(&k), mr, verdict(mr, br), vs_edge, pct,
                         runs.len() as f64 / nights.len() as f64);
            }
        }
        println!();
    }
}
