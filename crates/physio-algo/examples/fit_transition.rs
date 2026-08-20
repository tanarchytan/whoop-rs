//! Fit the transition matrix, the one part of the recipe the emission fit never touched.
//!
//!   cargo run --release -p physio-algo --example fit_transition
//!
//! Staging is `viterbi(emissions(..), transition)`. Fitting the EMISSION was measured and it loses
//! held-out. The transition is a separate mechanism with 16 hand-set numbers, and unlike the
//! emission it has a closed-form maximum-likelihood estimate: count the transitions the reference
//! actually makes and normalise. No optimiser, no hyperparameter, nothing to overfit but the counts.
//!
//! The isolation is exact. Both arms decode the SAME shipped emissions over the same epochs; only
//! the 4x4 matrix differs. Whatever moves is the transition and nothing else.
//!
//! Same discipline as the emission fit: counted on DREAMT, reported HELD-OUT, every claim a paired
//! per-night difference against its own bar. And the same caveat - the shipped matrix was chosen
//! watching all three cohorts, so the baseline is not blind and part of its held-out margin is
//! exposure this estimate never had.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar, recall, specificity, WAKE};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const FIT: &str = "dreamt";
const HELD_OUT: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Added to every transition count. A stage pair the reference never happens to show is rare, not
/// impossible, and a zero row in a log-space decoder forbids a path outright.
const LAPLACE: f64 = 1.0;
/// Nights shorter than this are not scored, matching the emission harness.
const MIN_EPOCHS: usize = 20;

struct Night {
    /// Shipped emissions, one row per epoch, in [`STAGE_ORDER`] columns.
    em: Vec<[f64; CLASSES]>,
    /// Reference label per epoch in our class order, or `None` where unlabelled.
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
        let input =
            SleepInput { start: w0, end: w1, hr: read_hr(dir), rr: read_rr(dir), accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        let _ = n;
        out.push(Night { em, truth });
    }
    out
}

/// Row-normalised transition counts over consecutive LABELLED epoch pairs, in [`STAGE_ORDER`].
///
/// Only pairs where both epochs carry a label count. A gap in the reference is not a transition,
/// and treating it as one teaches the matrix a jump the wearer never made.
fn count_transitions(nights: &[Night]) -> ([[f64; CLASSES]; CLASSES], usize) {
    let order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    let mut cnt = [[LAPLACE; CLASSES]; CLASSES];
    let mut pairs = 0usize;
    for nt in nights {
        for w in nt.truth.windows(2) {
            let (Some(a), Some(b)) = (w[0], w[1]) else { continue };
            // Our class index -> the decoder's column.
            let (fi, ti) = (
                order.iter().position(|c| *c == a).expect("class in STAGE_ORDER"),
                order.iter().position(|c| *c == b).expect("class in STAGE_ORDER"),
            );
            cnt[fi][ti] += 1.0;
            pairs += 1;
        }
    }
    for row in cnt.iter_mut() {
        let s: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    (cnt, pairs)
}

/// Per-night kappa, wake recall and wake specificity under one transition matrix.
fn score(nights: &[Night], t: &[[f64; CLASSES]; CLASSES]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut ks, mut rs, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    for nt in nights {
        let path = decode_v2(&nt.em, t);
        let (mut p, mut tr) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(stage_idx(path[k]));
            tr.push(*want);
        }
        if p.len() < MIN_EPOCHS {
            continue;
        }
        let cm = confusion4(&p, &tr);
        ks.push(kappa4(&cm));
        rs.push(recall(&cm, WAKE).unwrap_or(f64::NAN));
        ss.push(specificity(&cm, WAKE).unwrap_or(f64::NAN));
    }
    (ks, rs, ss)
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
    let train = load(FIT);
    if train.is_empty() {
        println!("no {FIT} nights - check the fixture root");
        return;
    }
    let (fitted, pairs) = count_transitions(&train);
    println!("COUNTED on {} ({} nights, {pairs} labelled epoch pairs)\n", FIT, train.len());

    println!("transition, rows = from, cols = to, in {:?}", STAGE_ORDER);
    println!("  {:<7} {:>9} {:>9} {:>9} {:>9}", "from", "deep", "rem", "light", "wake");
    for (i, st) in STAGE_ORDER.iter().enumerate() {
        let sh = Params::SHIPPED.transition[i];
        println!("  {:<7} {:>9.4} {:>9.4} {:>9.4} {:>9.4}   shipped", format!("{st:?}"),
                 sh[0], sh[1], sh[2], sh[3]);
        println!("  {:<7} {:>9.4} {:>9.4} {:>9.4} {:>9.4}   counted", "",
                 fitted[i][0], fitted[i][1], fitted[i][2], fitted[i][3]);
    }

    println!("\n{:<20} {:>7} {:>7} {:>7}   {:>10} {:>9} {:>6}   verdict",
             "cohort", "kappa", "wake r", "spec", "paired d", "bar +/-", "n");
    let cohorts: Vec<(String, Vec<Night>)> = std::iter::once((format!("{FIT} (COUNTED)"), train))
        .chain(HELD_OUT.iter().map(|s| (format!("{s} (HELD)"), load(s))))
        .collect();
    for (name, nights) in &cohorts {
        if nights.is_empty() {
            println!("{name:<20} no nights");
            continue;
        }
        let (bk, br, bs) = score(nights, &Params::SHIPPED.transition);
        let (fk, fr, fs) = score(nights, &fitted);
        let d: Vec<f64> = bk.iter().zip(&fk).map(|(a, b)| b - a).collect();
        let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
        let verdict = if mean.abs() > bar {
            format!("RESOLVED ({:.2}x the bar)", mean.abs() / bar)
        } else {
            "inside the bar - noise".to_string()
        };
        println!("{:<20} {:>7.3} {:>7.3} {:>7.3}   {:>10} {:>9} {:>6}   shipped",
                 name, median(&bk), median(&br), median(&bs), "", "", d.len());
        println!("{:<20} {:>7.3} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>6}   {verdict}",
                 "", median(&fk), median(&fr), median(&fs), d.len());
    }

    println!("\nBoth arms decode the SAME shipped emissions; only the matrix differs, so whatever");
    println!("moved is the transition. Held out is the only result - the counted matrix saw DREAMT.");
}
