//! Score a candidate feature on what the model gets WRONG, not on everything.
//!
//!   cargo run --release -p physio-algo --example residual_value
//!
//! Five features in a row were selected on single-feature AUC and delivered nothing in combination:
//! frequency HRV, off_posture, the axis pair, per-axis weights, and turn. turn is the clearest case
//! - the best AUC we have ever measured, 0.802, and worth zero marginal.
//!
//! The reason is that plain AUC asks "does this separate wake from sleep", when the question is
//! "does this separate the wake we are CURRENTLY MISSING from the sleep we currently get right". A
//! feature correlated with what the recipe already reads scores well on the first and nothing on the
//! second.
//!
//! Two numbers per candidate, both cheap:
//!   RESIDUAL AUC - computed only over epochs the shipped model gets wrong, plus the ones it gets
//!                  right, so it measures the separation still available.
//!   |r| vs jerk  - linear redundancy with the motion term already in the recipe.
//!
//! A candidate worth wiring has residual AUC well above 0.5 AND low redundancy. This harness exists
//! to be run BEFORE a port, not after.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth};

use physio_algo::sleep::hrv_bands::bands_series;
use physio_algo::sleep::movement::movement_series;
use physio_algo::sleep::posture::{posture_series, turn_series};
use common::stage_at;
use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput};

const EPOCH: i64 = 30;
const SETS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const MIN_PER_CLASS: usize = 10;

fn auc(pos: &[f64], neg: &[f64]) -> Option<f64> {
    if pos.len() < MIN_PER_CLASS || neg.len() < MIN_PER_CLASS {
        return None;
    }
    let mut all: Vec<(f64, u8)> =
        pos.iter().map(|v| (*v, 1u8)).chain(neg.iter().map(|v| (*v, 0u8))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut rank, mut i) = (0.0f64, 0usize);
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        rank += all[i..j].iter().filter(|x| x.1 == 1).count() as f64 * avg;
        i = j;
    }
    let (n1, n0) = (pos.len() as f64, neg.len() as f64);
    Some((rank - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return f64::NAN;
    }
    let (ma, mb) = (a[..n].iter().sum::<f64>() / n as f64, b[..n].iter().sum::<f64>() / n as f64);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return f64::NAN;
    }
    num / (da * db).sqrt()
}

fn jerk_peaks(grav: &[physio_algo::sleep::AccelSample], w0: i64, n: usize) -> Vec<Option<f64>> {
    let mut out = vec![None; n];
    let mut i = 0usize;
    for (k, slot) in out.iter_mut().enumerate() {
        let (a, b) = (w0 + k as i64 * EPOCH, w0 + (k as i64 + 1) * EPOCH);
        while i < grav.len() && grav[i].ts < a {
            i += 1;
        }
        let j = i + grav[i..].iter().take_while(|s| s.ts < b).count();
        let mut peak: Option<f64> = None;
        for (p, q) in grav[i..j].iter().zip(grav[i..j].iter().skip(1)) {
            let d = ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2)).sqrt();
            peak = Some(peak.map_or(d, |m: f64| m.max(d)));
        }
        *slot = peak;
    }
    out
}

fn main() {
    println!("Candidate value measured on the RESIDUAL: the epochs the shipped model gets wrong.");
    println!("plain AUC = over everything.  residual AUC = missed wake vs correctly-called sleep.");
    println!("|r| jerk  = redundancy with the motion term already in the recipe.\n");

    for set in SETS {
        let names = ["turn", "swing", "anisotropy", "lf_hf", "jerk (incumbent)"];
        let mut plain: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        let mut resid: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        let mut redun: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        let mut nights = 0usize;

        for dir in &dirs_of(set) {
            let truth = read_truth(dir);
            let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
            let (hr, grav, rr) = (read_hr(dir), read_accel(dir), read_rr(dir));
            if truth.is_empty() || grav.is_empty() {
                continue;
            }
            let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);
            let input = SleepInput { start: w0, end: w1, hr, rr, accel: grav.clone() };
            let segs = stage_v2_prepared(&prepare_v2(&input, &Params::SHIPPED), &Params::SHIPPED);

            let post = posture_series(&grav, w0, w0 + n as i64 * EPOCH, EPOCH);
            let turns = turn_series(&post);
            let mv = movement_series(&grav, &post, w0, w0 + n as i64 * EPOCH, EPOCH);
            let jerks = jerk_peaks(&grav, w0, n);
            let beats: Vec<(f64, f64)> = input
                .rr
                .iter()
                .flat_map(|r| r.intervals.iter().map(move |m| (r.ts as f64, *m as f64)))
                .collect();
            let bands = bands_series(&beats, w0 as f64, (w0 + n as i64 * EPOCH) as f64, EPOCH as f64);

            let val = |k: usize, f: usize| -> Option<f64> {
                match f {
                    0 => turns.get(k).copied().flatten(),
                    1 => post.get(k).and_then(|p| p.map(|p| p.swing)),
                    2 => mv.get(k).and_then(|m| m.anisotropy),
                    3 => bands.get(k).and_then(|b| b.as_ref()).and_then(|b| b.lf_hf),
                    _ => jerks.get(k).copied().flatten(),
                }
            };

            let mut any = false;
            for f in 0..names.len() {
                let (mut pp, mut pn) = (Vec::new(), Vec::new());
                let (mut rp, mut rn) = (Vec::new(), Vec::new());
                let (mut fv, mut jv) = (Vec::new(), Vec::new());
                for (k, t) in truth.iter() {
                    let Some(v) = val(*k, f) else { continue };
                    let is_wake = *t == 0;
                    let pred_wake = stage_at(&segs, w0 + *k as i64 * EPOCH)
                        .map(|s| s == physio_algo::sleep::SleepStage::Wake);
                    let Some(pred_wake) = pred_wake else { continue };
                    if is_wake { pp.push(v) } else { pn.push(v) }
                    // The residual: wake we MISSED against sleep we called correctly.
                    if is_wake && !pred_wake {
                        rp.push(v);
                    } else if !is_wake && !pred_wake {
                        rn.push(v);
                    }
                    if let Some(j) = val(*k, 4) {
                        fv.push(v);
                        jv.push(j);
                    }
                }
                if let Some(a) = auc(&pp, &pn) {
                    plain[f].push(a);
                    any = true;
                }
                if let Some(a) = auc(&rp, &rn) {
                    resid[f].push(a);
                }
                let r = pearson(&fv, &jv);
                if r.is_finite() {
                    redun[f].push(r.abs());
                }
            }
            if any {
                nights += 1;
            }
        }

        println!("=== {set}  ({nights} nights)");
        println!("  {:<18} {:>10} {:>13} {:>10}", "feature", "plain AUC", "residual AUC", "|r| jerk");
        for (f, name) in names.iter().enumerate() {
            if plain[f].is_empty() {
                continue;
            }
            println!("  {:<18} {:>10.3} {:>13.3} {:>10.2}", name, median(&plain[f]),
                     median(&resid[f]), median(&redun[f]));
        }
        println!();
    }
    println!("Read it as: a candidate is only worth wiring if its RESIDUAL AUC is well clear of 0.5.");
    println!("A high plain AUC with a residual near 0.5 means the recipe already has that information.");
}
