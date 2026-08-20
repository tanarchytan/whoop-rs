//! Refusing to label an epoch, and whether refusing HELPS or merely drops the hard ones.
//!
//!   cargo run --release -p physio-algo --example abstain
//!
//! Every epoch gets a stage today whether the evidence supports one or not. The controlled
//! signal-ladder study found confidence-based abstention worth more than any feature it tested, and
//! the recipe has none. This measures it on our own cohorts.
//!
//! **Abstention raises kappa for free**, because dropping epochs drops the hard ones. So the only
//! honest arm is against a RANDOM abstention at the SAME coverage: keeping 80% at random is the
//! floor any confidence signal has to clear. A gain over 100% coverage means nothing on its own.
//!
//! Nothing is fitted here - the confidence is read off the emissions the recipe already computes -
//! so all three cohorts are held out and the whole sweep is printed rather than a chosen point.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Fraction of epochs KEPT. 1.00 is today's behaviour.
const COVERAGE: [f64; 6] = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60];
/// Fewest retained epochs before a night is scored at all.
const MIN_EPOCHS: usize = 20;

struct Night {
    /// Decoded stage per epoch, our class index.
    pred: Vec<usize>,
    /// Reference label, or `None` where the epoch is unlabelled.
    truth: Vec<Option<usize>>,
    /// How separated the decoded class was from its nearest rival, per epoch.
    margin: Vec<f64>,
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
        let path = decode_v2(&em, &Params::SHIPPED.transition);
        // Top minus second, on the emission the decoder saw. A wide gap is an epoch the evidence
        // decides on its own; a narrow one is decided by the transition prior instead.
        let margin = em
            .iter()
            .map(|row| {
                let mut v = *row;
                v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                v[0] - v[1]
            })
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { pred: path.iter().map(|s| stage_idx(*s)).collect(), truth, margin });
    }
    out
}

/// Deterministic pseudo-random score per epoch, for the matched-coverage null. Seeded off the epoch
/// index so two runs abstain on the same epochs and the arms stay comparable.
fn noise(seed: u64, k: usize) -> f64 {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(k as u64 + 1);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Kappa over the `keep` fraction of LABELLED epochs with the highest `score`, per night.
fn scored(nights: &[Night], keep: f64, by_margin: bool) -> Vec<f64> {
    let mut out = Vec::new();
    for (ni, nt) in nights.iter().enumerate() {
        // Rank only labelled epochs: an unlabelled one cannot be right or wrong and must not
        // occupy a slot in the coverage budget.
        let mut idx: Vec<usize> = (0..nt.pred.len()).filter(|k| nt.truth[*k].is_some()).collect();
        let score = |k: usize| if by_margin { nt.margin[k] } else { noise(ni as u64, k) };
        idx.sort_by(|a, b| score(*b).partial_cmp(&score(*a)).unwrap());
        let take = ((idx.len() as f64 * keep).round() as usize).max(MIN_EPOCHS).min(idx.len());
        if take < MIN_EPOCHS {
            continue;
        }
        let (p, t): (Vec<usize>, Vec<usize>) =
            idx[..take].iter().map(|k| (nt.pred[*k], nt.truth[*k].unwrap())).unzip();
        out.push(kappa4(&confusion4(&p, &t)));
    }
    out
}

fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn main() {
    println!("Kappa on the epochs KEPT, ranked by emission margin, against keeping the same");
    println!("number AT RANDOM. The random column is the floor: dropping hard epochs raises kappa");
    println!("on its own, so only the margin-minus-random difference is a confidence signal.\n");

    for set in COHORTS {
        let nights = load(set);
        if nights.is_empty() {
            println!("{set}: no nights\n");
            continue;
        }
        println!("=== {set} ({} nights)", nights.len());
        println!("  {:>8} {:>8} {:>8}   {:>10} {:>9} {:>5}   verdict",
                 "coverage", "margin", "random", "paired d", "bar +/-", "n");
        let base = median(&scored(&nights, 1.0, true));
        for keep in COVERAGE {
            let m = scored(&nights, keep, true);
            let r = scored(&nights, keep, false);
            let d: Vec<f64> = r.iter().zip(&m).map(|(a, b)| b - a).collect();
            let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
            let verdict = if !mean.is_finite() {
                "-".to_string()
            } else if mean.abs() > bar {
                format!("{} ({:.2}x the bar)", if mean > 0.0 { "REAL" } else { "WORSE" },
                        mean.abs() / bar)
            } else {
                "inside the bar - noise".to_string()
            };
            println!("  {:>7.0}% {:>8.3} {:>8.3}   {mean:>+10.4} {bar:>9.4} {:>5}   {verdict}",
                     100.0 * keep, median(&m), median(&r), d.len());
        }
        println!("  full-coverage kappa {base:.3} - a margin column above it that does not beat");
        println!("  the random column has bought nothing.\n");
    }
}
