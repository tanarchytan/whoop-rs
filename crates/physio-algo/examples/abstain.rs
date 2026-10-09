//! Refusing to label an epoch: whether it helps, what it is really detecting, and what it looks
//! like in a hypnogram.
//!
//!   cargo run --release -p physio-algo --example abstain
//!   ABSTAIN_DRAWS=20 cargo run ...   # average both random nulls over 20 draws (default 1)
//!   ABSTAIN_DECODE=base cargo run ... # decode under tanv1::BASE (what the Experimental engine ships)
//!                                     # instead of Params::SHIPPED (the default, the original R2 numbers)
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
use physio_algo::sleep::abstain::{self, refuse_among, MIN_EPOCHS};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::tanv1::BASE;
use physio_algo::sleep::{decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Fraction of epochs KEPT. 1.00 is today's behaviour.
const COVERAGE: [f64; 5] = [0.95, 0.90, 0.80, 0.70, 0.60];
/// Half-width, in epochs, of the smoothing applied to the margin before ranking. Smoothing is what
/// turns scattered single-epoch refusals into stretches a hypnogram can draw.
const SMOOTH: usize = 4;

/// How an epoch's willingness-to-answer is scored. Higher keeps.
#[derive(Clone, Copy, PartialEq)]
enum Rule {
    Random,
    ClassMatchedRandom,
    FarFromTransition,
    Margin,
    SmoothMargin,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Rule::Random => "random",
            Rule::ClassMatchedRandom => "random per class",
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
    /// [`abstain::to_edge`] of the decoded path. A night the decoder never changes stage on scores
    /// every epoch [`f64::INFINITY`], so the index tie-break keeps the EARLIEST ones.
    to_edge: Vec<f64>,
}

/// The decode the nights are staged under: `Params::SHIPPED`, or `tanv1::BASE` with `ABSTAIN_DECODE=base`.
fn decode_params() -> Params {
    if std::env::var("ABSTAIN_DECODE").is_ok_and(|v| v == "base") { BASE } else { Params::SHIPPED }
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
        let prm = decode_params();
        let prep = prepare_v2(&input, &prm);
        let em = emissions_v2(&prep, &prm);
        // `truth` below indexes `raw` positionally, which holds only while the grid is complete.
        // Checked BEFORE the length skip, or the hardest-collapsed grid is the one that leaves
        // silently instead of tripping it.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let path = decode_v2(&em, &prm.transition);
        let pred: Vec<usize> = path.iter().map(|s| stage_idx(*s)).collect();
        let margin: Vec<f64> = em
            .iter()
            .map(|row| {
                let mut v = *row;
                v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                v[0] - v[1]
            })
            .collect();
        let to_edge = abstain::to_edge(&path);
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { pred, truth, margin, to_edge });
    }
    out
}

/// Random draws averaged for the two random nulls: `ABSTAIN_DRAWS`, default 1.
fn draws() -> usize {
    std::env::var("ABSTAIN_DRAWS").ok().and_then(|v| v.parse().ok()).filter(|n| *n >= 1).unwrap_or(1)
}

/// Deterministic pseudo-random score, so two runs refuse the same epochs. `draw` 0 is the original
/// seed (the night index), so one draw reproduces the single-draw numbers exactly.
fn noise(seed: u64, draw: usize, k: usize) -> f64 {
    let seed = seed.wrapping_add((draw as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
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
fn rank(nt: &Night, ni: usize, draw: usize, rule: Rule) -> Vec<f64> {
    match rule {
        // `ClassMatchedRandom` takes its per-class quota from far-from-edge, so it cannot be a score.
        Rule::Random | Rule::ClassMatchedRandom => {
            (0..nt.pred.len()).map(|k| noise(ni as u64, draw, k)).collect()
        }
        Rule::FarFromTransition => unreachable!("far-from-edge is the library's refuse_among"),
        Rule::Margin => nt.margin.clone(),
        Rule::SmoothMargin => smooth(&nt.margin),
    }
}

/// What one rule leaves at one coverage: kappa per night on the kept epochs, the run-lengths of
/// what it refused, and the refused and labelled epoch counts per TRUTH class.
struct Kept {
    /// Per night; for a random rule the MEAN over the draws, so a paired delta against it is the
    /// mean over draws of the per-night deltas.
    kappas: Vec<f64>,
    /// Draws pooled into `runs`, `refused` and `labelled` (1 for the deterministic rules).
    draws: usize,
    runs: Vec<usize>,
    refused: [usize; CLASSES],
    labelled: [usize; CLASSES],
}

impl Kept {
    /// Each class's share of the refusals over its share of the labelled epochs. 1.0 is neutral; a
    /// rule that bought its kappa by giving one class up prints a large number on that class.
    fn class_bias(&self) -> [f64; CLASSES] {
        let (r, l) = (self.refused.iter().sum::<usize>(), self.labelled.iter().sum::<usize>());
        std::array::from_fn(|c| match (r, self.labelled[c]) {
            (0, _) | (_, 0) => f64::NAN,
            _ => (self.refused[c] as f64 / r as f64) / (self.labelled[c] as f64 / l as f64),
        })
    }
}

/// One night's labelled epochs as (kept, refused) at one coverage, refused in epoch order. `None`
/// when the night cannot reach `MIN_EPOCHS`.
fn split(nt: &Night, ni: usize, draw: usize, keep: f64, rule: Rule) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut idx: Vec<usize> = (0..nt.pred.len()).filter(|k| nt.truth[*k].is_some()).collect();
    if idx.len() < MIN_EPOCHS {
        return None;
    }
    if rule == Rule::ClassMatchedRandom {
        // Refuse at random WITHIN each truth class, the same count per class that far-from-edge
        // refuses. The plain random null drops proportionally, so it prices a drop but not a skew.
        let (_, edge) = split(nt, ni, draw, keep, Rule::FarFromTransition)?;
        let mut quota = [0usize; CLASSES];
        for k in &edge {
            quota[nt.truth[*k].unwrap()] += 1;
        }
        let mut dropped = Vec::with_capacity(edge.len());
        for (c, q) in quota.iter().enumerate() {
            let mut of_class: Vec<usize> =
                idx.iter().copied().filter(|k| nt.truth[*k] == Some(c)).collect();
            of_class.sort_by(|a, b| noise(ni as u64, draw, *a).total_cmp(&noise(ni as u64, draw, *b)));
            dropped.extend(of_class.into_iter().take(*q));
        }
        dropped.sort_unstable();
        let kept = idx.iter().copied().filter(|k| dropped.binary_search(k).is_err()).collect();
        return Some((kept, dropped));
    }
    if rule == Rule::FarFromTransition {
        let dropped = refuse_among(&nt.to_edge, &idx, keep);
        let kept = idx.iter().copied().filter(|k| dropped.binary_search(k).is_err()).collect();
        return Some((kept, dropped));
    }
    let take = ((idx.len() as f64 * keep).round() as usize).max(MIN_EPOCHS).min(idx.len());
    let score = rank(nt, ni, draw, rule);
    // Ties broken by index so two rules with equal scores still refuse the same COUNT.
    idx.sort_by(|a, b| score[*b].total_cmp(&score[*a]).then(a.cmp(b)));
    let mut dropped = idx[take..].to_vec();
    dropped.sort_unstable();
    Some((idx[..take].to_vec(), dropped))
}

/// Kappa per night on the kept epochs, and the run-lengths of what was refused. The two random
/// rules average kappa over `draws()` draws; the others are deterministic and use one.
fn scored(nights: &[Night], keep: f64, rule: Rule) -> Kept {
    let nd = if matches!(rule, Rule::Random | Rule::ClassMatchedRandom) { draws() } else { 1 };
    let (mut ks, mut runs) = (Vec::new(), Vec::new());
    let (mut refused, mut labelled) = ([0usize; CLASSES], [0usize; CLASSES]);
    for (ni, nt) in nights.iter().enumerate() {
        let mut sum = 0.0;
        let mut ok = true;
        for draw in 0..nd {
            let Some((kept, dropped)) = split(nt, ni, draw, keep, rule) else {
                ok = false;
                break;
            };
            let (p, t): (Vec<usize>, Vec<usize>) =
                kept.iter().map(|k| (nt.pred[*k], nt.truth[*k].unwrap())).unzip();
            sum += kappa4(&confusion4(&p, &t));

            // Run-lengths of the refusals, in epoch order, over the LABELLED epochs only: an epoch
            // the reference is silent on is absent from the grid, so it splits one stretch in two.
            for k in kept.iter().chain(&dropped) {
                labelled[nt.truth[*k].unwrap()] += 1;
            }
            for k in &dropped {
                refused[nt.truth[*k].unwrap()] += 1;
            }
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
        // Whether a night is short enough to skip does not depend on the draw.
        if ok {
            ks.push(sum / nd as f64);
        }
    }
    Kept { kappas: ks, draws: nd, runs, refused, labelled }
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
    println!("decode: {}\n", if decode_params() == BASE { "tanv1::BASE" } else { "Params::SHIPPED" });
    println!("Kappa on the epochs KEPT, every rule at MATCHED coverage, paired per night.");
    println!("`vs random` proves a drop is not free. `vs edge` needs no emissions, but a sticky");
    println!("decoder only changes state where the margin is wide, so epochs beside an edge carry");
    println!("WIDER margins: refusing them refuses the high-confidence epochs, close to the INVERSE");
    println!("of the margin rule, so clearing `vs edge` is the cheaper of the two results.");
    println!("`random per class` refuses the same COUNT PER TRUTH CLASS that far-from-edge does, so");
    println!("its `vs edge` prices the class skew alone: far-from-edge earns its gain only if that");
    println!("row LOSES. `refused w/l/d/r` is each class's refusal share over its labelled share.");
    println!("`1-run%` is the share of refusal RUNS that are a LONE 30 s epoch - a hypnogram");
    println!("cannot draw those. Runs are counted over the epochs the reference LABELS, so a gap");
    println!("in truth inside a refusal stretch reads as two runs rather than one.\n");

    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("{set}: no nights\n");
            continue;
        }
        let mut full = scored(&nights, 1.0, Rule::Margin).kappas;
        let n_scored = full.len();
        println!("=== {set} ({n_scored} of {} nights scored), full-coverage kappa {:.3}",
                 nights.len(), median(&mut full));
        // A verdict group is 8 + 1 + 14 = 23 wide, 14 being the longest string `verdict` returns.
        println!("  {:>5} {:<16} {:>7} {:<23} {:<23} {:>7} {:>7} {:>21}",
                 "keep", "rule", "kappa", "vs random", "vs edge", "1-run%", "runs/n",
                 "refused w/l/d/r");
        for keep in COVERAGE {
            let rnd = scored(&nights, keep, Rule::Random);
            let edge = scored(&nights, keep, Rule::FarFromTransition);
            for rule in [
                Rule::FarFromTransition,
                Rule::ClassMatchedRandom,
                Rule::Margin,
                Rule::SmoothMargin,
            ] {
                let got = if rule == Rule::FarFromTransition {
                    None
                } else {
                    Some(scored(&nights, keep, rule))
                };
                let this = got.as_ref().unwrap_or(&edge);
                let mut k = this.kappas.clone();
                let dr: Vec<f64> = rnd.kappas.iter().zip(&k).map(|(a, b)| b - a).collect();
                let (mr, br) = paired_bar(&dr).unwrap_or((f64::NAN, f64::NAN));
                let ones = this.runs.iter().filter(|r| **r == 1).count();
                let pct = 100.0 * ones as f64 / this.runs.len().max(1) as f64;
                let vs_edge = match &got {
                    None => "  (is the null)".to_string(),
                    Some(g) => {
                        let de: Vec<f64> =
                            edge.kappas.iter().zip(&g.kappas).map(|(a, b)| b - a).collect();
                        let (me, be) = paired_bar(&de).unwrap_or((f64::NAN, f64::NAN));
                        format!("{me:>+8.4} {}", verdict(me, be))
                    }
                };
                // Refusal share per truth class over that class's labelled share; 1.0 is neutral.
                // A rule that bought its kappa by giving a class up cannot hide it behind a median.
                let b = this.class_bias();
                // `runs/n` is per SCORED night; `k` holds one kappa per night that passed the guard.
                let scored_nights = k.len();
                let kappa = median(&mut k);
                println!("  {:>4.0}% {:<16} {:>7.3} {:>+8.4} {:<14} {:<23} {:>6.0}% {:>7.1} {:>21}",
                         100.0 * keep, rule.name(), kappa, mr, verdict(mr, br), vs_edge, pct,
                         this.runs.len() as f64 / (this.draws * scored_nights).max(1) as f64,
                         format!("{:.2}/{:.2}/{:.2}/{:.2}", b[0], b[1], b[2], b[3]));
            }
        }
        println!();
    }
}
