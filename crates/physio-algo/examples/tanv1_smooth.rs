//! Is tanv1 double-smoothing? The decoder pays v2 +0.06 to +0.12 and tanv1 about nothing.
//!
//!   cargo run --release -p physio-algo --example tanv1_smooth
//!
//! Two readings fit that fact and they imply opposite designs:
//!   (a) the decoder is weak for tanv1        -> build a better decoder
//!   (b) tanv1's emission is ALREADY smoothed -> its long windows did the decoder's job first
//!
//! (b) predicts something (a) does not: tanv1's argmax path should already have far fewer runs than
//! v2's, before any decoding, because ten-minute windows correlate neighbouring epochs by
//! construction. And it predicts that a SHORT-WINDOW tanv1, with every 300 s and 600 s column and the
//! clock dropped, should fragment like v2 does and then collect the same decoder bonus.
//!
//! So this measures runs per night at every stage, and the decoder GAIN per arm, which is the
//! quantity that actually separates the two readings.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;

struct Night {
    tan: Vec<Vec<f64>>,
    shipped: Vec<[f64; CLASSES]>,
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
        let (hr, rr) = (read_hr(dir), read_rr(dir));
        let f = extract(&accel, w0, w1, &cardiac_series(w0, n, EPOCH, &hr, &rr));
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let em = emissions_v2(&prepare_v2(&input, &Params::SHIPPED), &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        out.push(Night {
            tan: (0..em.len()).map(|e| f[e].values().to_vec()).collect(),
            shipped: em[..].to_vec(),
            truth: (0..em.len())
                .map(|k| {
                    raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
                })
                .collect(),
        });
    }
    out
}

/// Columns whose window is at most `max_s` seconds, plus the cardiac and interaction columns that
/// carry no window. `clock` is the slowest column of all and goes with the long ones.
fn short_cols(max_s: usize) -> Vec<usize> {
    (0..Features::N)
        .filter(|j| {
            let n = Features::NAMES[*j];
            if n == "clock" {
                return false;
            }
            match n.rsplit('_').next().and_then(|t| t.parse::<usize>().ok()) {
                Some(w) => w <= max_s,
                None => true,
            }
        })
        .collect()
}

fn rows(nights: &[&Night], keep: &[usize]) -> (Vec<Vec<f64>>, Vec<usize>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for nt in nights {
        for (e, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                x.push(keep.iter().map(|j| nt.tan[e][*j]).collect());
                y.push(*t);
            }
        }
    }
    (x, y)
}

fn emission(nt: &Night, w: &[Vec<f64>], m: &[f64], sd: &[f64], keep: &[usize])
    -> Vec<[f64; CLASSES]> {
    (0..nt.truth.len())
        .map(|e| {
            let r: Vec<f64> = keep.iter().map(|j| nt.tan[e][*j]).collect();
            let z = scores(w, &design_row(&r, m, sd, &[]));
            std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
        })
        .collect()
}

fn argmax(em: &[[f64; CLASSES]]) -> Vec<usize> {
    em.iter()
        .map(|row| {
            let mut best = (0usize, f64::NEG_INFINITY);
            for (c, v) in row.iter().enumerate() {
                if *v > best.1 {
                    best = (c, *v);
                }
            }
            stage_idx(STAGE_ORDER[best.0])
        })
        .collect()
}

fn runs(p: &[usize]) -> f64 {
    (1 + (1..p.len()).filter(|k| p[*k] != p[k - 1]).count()) as f64
}

/// Kappa and run count over one night's labelled epochs.
fn score(path: &[usize], truth: &[Option<usize>]) -> Option<(f64, f64)> {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate() {
        if let Some(want) = want {
            p.push(path[k]);
            t.push(*want);
        }
    }
    (t.len() >= MIN_EPOCHS).then(|| (kappa4(&confusion4(&p, &t)), runs(&p)))
}

fn bar(base: &[f64], arm: &[f64]) -> String {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let Some((mean, bar)) = paired_bar(&d) else { return "-".into() };
    let tag = if mean.abs() <= bar {
        "matches".to_string()
    } else {
        format!("{} ({:.2}x)", if mean > 0.0 { "AHEAD" } else { "behind" }, mean.abs() / bar)
    };
    format!("{mean:>+8.4} {bar:>7.4} {tag}")
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    let full: Vec<usize> = (0..Features::N).collect();
    let short = short_cols(120);
    println!("Is tanv1 double-smoothing? The decoder pays v2 +0.06 to +0.12 and tanv1 about nothing.");
    println!("If tanv1's long windows already did the smoothing, its ARGMAX path is already short of");
    println!("runs before any decoding, and a short-window tanv1 should fragment and collect the bonus.\n");
    println!("  short-window set keeps {} of {} columns, dropping every 300 s and 600 s column and the",
             short.len(), Features::N);
    println!("  clock: {}\n",
             (0..Features::N).filter(|j| !short.contains(j))
                 .map(|j| Features::NAMES[j]).collect::<Vec<_>>().join(" "));

    for (held, hn) in &loaded {
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        println!("== {held} n={} ==", hn.len());
        println!("  {:<16} {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}   decoder gain",
                 "arm", "k argmx", "k decod", "truth r", "runs am", "runs dc", "truth r");

        let mut base_dec: Vec<f64> = Vec::new();
        for (label, keep) in
            [("shipped", None), ("tanv1 full", Some(&full)), ("tanv1 short", Some(&short))]
        {
            let fitted = keep.map(|k| {
                let (x, y) = rows(&tr, k);
                let (m, sd) = standardise_cols(&x);
                let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
                (fit(&dx, &y, WEIGHT_POWER), m, sd)
            });
            let (mut ka, mut kd, mut ra, mut rd, mut rt) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for nt in hn {
                let em = match (&fitted, keep) {
                    (Some((w, m, sd)), Some(k)) => emission(nt, w, m, sd, k),
                    _ => nt.shipped.clone(),
                };
                let a = argmax(&em);
                let d: Vec<usize> =
                    decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
                let t: Vec<usize> = nt.truth.iter().flatten().copied().collect();
                if let (Some((k1, r1)), Some((k2, r2))) =
                    (score(&a, &nt.truth), score(&d, &nt.truth))
                {
                    ka.push(k1);
                    kd.push(k2);
                    ra.push(r1);
                    rd.push(r2);
                    rt.push(runs(&t));
                }
            }
            let gain: Vec<f64> = ka.iter().zip(&kd).map(|(a, b)| b - a).collect();
            let (gm, gb) = paired_bar(&gain).unwrap_or((f64::NAN, f64::NAN));
            println!("  {:<16} {:>7.3} {:>7.3} {:>7.0}   {:>7.0} {:>7.0} {:>7.0}   {gm:>+7.4} +/- {gb:.4}",
                     label, median(&mut ka.clone()), median(&mut kd.clone()),
                     median(&mut rt.clone()), median(&mut ra.clone()), median(&mut rd.clone()),
                     median(&mut rt.clone()));
            if label == "shipped" {
                base_dec = kd.clone();
            } else {
                println!("  {:<16} decoded vs shipped decoded: {}", "", bar(&base_dec, &kd));
            }
        }
        println!();
    }
    println!("Reading: a decoder gain near zero beside an argmax run count already near the decoded one");
    println!("means the emission arrived smooth and the prior had nothing to add. If `tanv1 short`");
    println!("fragments at argmax and then collects a gain like the shipped arm's, the long windows are");
    println!("the cause and the decoder is not the defect.");
}
