//! Three axes as THREE features, not one score.
//!
//!   cargo run --release -p physio-algo --example per_axis
//!
//! Every motion feature tried so far collapses (x, y, z) to one number: jerk to a norm, swing to a
//! spread, turn to an angle, anisotropy to a ratio. All of them answer "how much" and discard "in
//! which direction".
//!
//! A strap is worn one way round, so the device axes map to rough anatomical directions - along the
//! forearm, across the wrist, normal to the face. Motion along each means something different, and a
//! model with three weights can learn that where a model with one cannot. This scores each axis on
//! its own so the question is answered with numbers rather than plausibility.
//!
//! Caveat stated up front: the wearer can put the strap on rotated, so the axis-to-anatomy mapping is
//! a per-donning constant, not a universal one. A per-axis weight fitted across wearers is only
//! meaningful if the mapping is stable, and this harness measures whether it is - if the same axis
//! wins on every wearer, it is stable enough to weight.

mod common;

use common::{dirs_of, read_accel, read_meta, read_truth};
use physio_algo::sleep::AccelSample;

const EPOCH: i64 = 30;
const SETS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const MIN_PER_CLASS: usize = 10;
const AXES: [&str; 3] = ["x", "y", "z"];

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

/// Peak per-second absolute delta on each axis, plus the norm, per epoch.
/// Returns `[x, y, z, norm]` series so all four are read off identical epochs.
fn axis_series(grav: &[AccelSample], w0: i64, n: usize) -> [Vec<Option<f64>>; 4] {
    let mut out = [vec![None; n], vec![None; n], vec![None; n], vec![None; n]];
    let mut by_sec: std::collections::HashMap<i64, (f64, f64, f64, f64)> = Default::default();
    for g in grav {
        let e = by_sec.entry(g.ts).or_insert((0.0, 0.0, 0.0, 0.0));
        e.0 += g.x;
        e.1 += g.y;
        e.2 += g.z;
        e.3 += 1.0;
    }
    for k in 0..n {
        let (a, b) = (w0 + k as i64 * EPOCH, w0 + (k as i64 + 1) * EPOCH);
        let mut prev: Option<(f64, f64, f64)> = None;
        let mut peak = [0.0f64; 4];
        let mut seen = false;
        for s in a..b {
            let Some(v) = by_sec.get(&s) else { continue };
            let cur = (v.0 / v.3, v.1 / v.3, v.2 / v.3);
            if let Some(p) = prev {
                let d = [(p.0 - cur.0).abs(), (p.1 - cur.1).abs(), (p.2 - cur.2).abs()];
                let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                for i in 0..3 {
                    peak[i] = peak[i].max(d[i]);
                }
                peak[3] = peak[3].max(norm);
                seen = true;
            }
            prev = Some(cur);
        }
        if seen {
            for i in 0..4 {
                out[i][k] = Some(peak[i]);
            }
        }
    }
    out
}

fn main() {
    println!("Per-axis peak delta as three separate features, against the norm they collapse into.");
    println!("AUC of wake over sleep. If one axis consistently beats the others ACROSS wearers, the");
    println!("axis-to-anatomy mapping is stable enough to carry its own weight.\n");

    for set in SETS {
        let mut per: [Vec<f64>; 4] = Default::default();
        // Which axis won on each night, to test stability rather than only the average.
        let mut winner = [0usize; 3];
        let mut nights = 0usize;

        for dir in &dirs_of(set) {
            let truth = read_truth(dir);
            let Some((w0, _, n_meta)) = read_meta(dir) else { continue };
            let grav = read_accel(dir);
            if truth.is_empty() || grav.is_empty() {
                continue;
            }
            let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);
            let s = axis_series(&grav, w0, n);
            let mut got = [f64::NAN; 4];
            let mut ok = true;
            for i in 0..4 {
                let (mut p, mut q) = (Vec::new(), Vec::new());
                for (k, t) in truth.iter() {
                    let Some(v) = s[i].get(*k).copied().flatten() else { continue };
                    if *t == 0 { p.push(v) } else { q.push(v) }
                }
                match auc(&p, &q) {
                    Some(a) => got[i] = a,
                    None => ok = false,
                }
            }
            if !ok {
                continue;
            }
            nights += 1;
            for i in 0..4 {
                per[i].push(got[i]);
            }
            let mut best = 0usize;
            for i in 1..3 {
                if got[i] > got[best] {
                    best = i;
                }
            }
            winner[best] += 1;
        }

        println!("=== {set}  ({nights} nights)");
        for i in 0..3 {
            println!("  axis {}          AUC {:.3}   best on {} of {} nights ({:.0}%)",
                     AXES[i], median(&per[i]), winner[i], nights,
                     100.0 * winner[i] as f64 / nights.max(1) as f64);
        }
        println!("  norm (collapsed) AUC {:.3}   <- what we ship", median(&per[3]));
        let best_axis = (0..3).max_by(|a, b| {
            median(&per[*a]).partial_cmp(&median(&per[*b])).unwrap()
        }).unwrap();
        println!("  best single axis is {} at {:.3}, norm is {:.3}  -> {}",
                 AXES[best_axis], median(&per[best_axis]), median(&per[3]),
                 if median(&per[best_axis]) > median(&per[3]) {
                     "AN AXIS BEATS THE NORM"
                 } else {
                     "the norm beats every single axis"
                 });
        println!();
    }
    println!("A single axis beating the norm would mean the collapse throws information away.");
    println!("The norm winning means it does not, and three weights would only add parameters.");
}
