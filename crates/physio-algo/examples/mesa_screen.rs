//! Screen 1 of the roster: do the ORDER STATISTICS carry stage information, do they survive our
//! channel, and does per-recording normalisation change the answer?
//!
//!   cargo run --release -p physio-algo --example mesa_screen [nights]
//!
//! The features come from `physio_algo::sleep::cardiac`, the same producer a stager would call, so
//! this measures the code that would run and not a second implementation of it.
//!
//! Three arms on the SAME nights, because a feature that only works on ECG is not a candidate:
//!   EXACT     the corpus's own beat times at 1/256 s — is the information there at all
//!   TIMING    the same beats through our wire format and back — what whole-second stamps cost
//!   COVERAGE  beats dropped in runs to 60% — what PPG dropout costs
//!
//! Each arm is read twice. RAW pools every recording's values together, so a feature whose LEVEL
//! differs between people is judged on that difference. Z re-scores each column within its own
//! recording first, which is what the stager consumes, and asks whether the feature moves with stage
//! inside one night.
//!
//! Reported as one-vs-rest AUC per stage. 0.5 is no information; distance from 0.5 in EITHER
//! direction is signal, so the table prints the cells and the largest deviation separately.

mod common;

use common::mesa::{self, Beat};
use physio_algo::sleep::cardiac;

/// One night's rows and their stages, blocks only where the window could carry one.
fn night_rows(beats: &[Beat], stage: &[Option<usize>]) -> (Vec<[f64; 18]>, Vec<usize>) {
    let pairs: Vec<(f64, f64)> = beats.iter().map(|b| (b.t, b.rr)).collect();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (e, s) in stage.iter().enumerate() {
        let Some(s) = s else { continue };
        let centre = e as f64 * mesa::EPOCH_S + mesa::EPOCH_S / 2.0;
        if let Some(b) = cardiac::extract(&pairs, centre - mesa::WINDOW_S / 2.0, centre + mesa::WINDOW_S / 2.0) {
            x.push(b.row());
            y.push(*s);
        }
    }
    (x, y)
}

/// Every column z-scored within this night. NaN is the missing marker on the way in and out, so a
/// column the recording cannot carry stays missing instead of becoming a manufactured mean.
fn zscore_night(rows: &[[f64; 18]]) -> Vec<[f64; 18]> {
    let mut out = vec![[f64::NAN; 18]; rows.len()];
    for f in 0..18 {
        let col: Vec<Option<f64>> = rows.iter().map(|r| r[f].is_finite().then_some(r[f])).collect();
        for (k, z) in cardiac::zscore_column(&col).into_iter().enumerate() {
            out[k][f] = z.unwrap_or(f64::NAN);
        }
    }
    out
}

/// Mann-Whitney AUC of `f` separating class `c` from the rest. Ties take half credit.
fn auc(vals: &[f64], labels: &[usize], c: usize) -> Option<f64> {
    let mut idx: Vec<usize> = (0..vals.len()).filter(|i| vals[*i].is_finite()).collect();
    if idx.len() < 2 {
        return None;
    }
    idx.sort_by(|a, b| vals[*a].total_cmp(&vals[*b]));
    // Average ranks over ties so a flat feature scores exactly 0.5 rather than drifting.
    let mut rank = vec![0.0f64; vals.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && vals[idx[j + 1]] == vals[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            rank[idx[k]] = r;
        }
        i = j + 1;
    }
    let (mut n1, mut sum1) = (0.0f64, 0.0f64);
    for i in &idx {
        if labels[*i] == c {
            n1 += 1.0;
            sum1 += rank[*i];
        }
    }
    let n0 = idx.len() as f64 - n1;
    (n1 > 0.0 && n0 > 0.0).then(|| (sum1 - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))
}

/// Raw rows, per-night z-scored rows, and the labels, pooled over the nights.
fn collect(nights: &[mesa::MesaNight], arm: &str) -> (Vec<[f64; 18]>, Vec<[f64; 18]>, Vec<usize>) {
    let (mut raw, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for n in nights {
        let beats = match arm {
            "TIMING" => mesa::degrade_timing(&n.beats),
            "COVERAGE" => mesa::degrade_coverage(&n.beats, mesa::COVERAGE_KEEP, mesa::COVERAGE_SEED),
            _ => n.beats.clone(),
        };
        let (rows, labels) = night_rows(&beats, &n.stage);
        z.extend(zscore_night(&rows));
        raw.extend(rows);
        y.extend(labels);
    }
    (raw, z, y)
}

/// The four one-vs-rest cells of one column, and the largest deviation from 0.5 among them.
fn cells(x: &[[f64; 18]], y: &[usize], f: usize) -> (Vec<String>, f64) {
    let col: Vec<f64> = x.iter().map(|r| r[f]).collect();
    let a: Vec<Option<f64>> = (0..4).map(|c| auc(&col, y, c)).collect();
    let best = a.iter().flatten().map(|v| (v - 0.5).abs()).fold(0.0, f64::max);
    let text = a
        .iter()
        .map(|v| v.map_or("-".into(), |v| format!("{v:.3}")))
        .collect();
    (text, best)
}

fn main() {
    let limit: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(60);
    let nights = mesa::nights(limit);
    println!(
        "MESA order-statistics screen — {} nights, window {} s centred",
        nights.len(),
        mesa::WINDOW_S
    );
    println!("`rr_dt_*` are detrended by a least-squares linear fit; the sources do not define it.");
    println!("RAW pools recordings; Z re-scores each column within its recording first.\n");

    for arm in ["EXACT", "TIMING", "COVERAGE"] {
        let (raw, z, y) = collect(&nights, arm);
        let mut mix = [0usize; 4];
        for c in &y {
            mix[*c] += 1;
        }
        println!("== {arm} == {} epochs  (w/l/d/r {mix:?})", raw.len());
        println!(
            "  {:<14} {:>26}  {:>26}   {:>6} {:>6}",
            "feature", "-------- RAW --------", "--- PER-NIGHT Z ---", "raw", "z"
        );
        println!(
            "  {:<14} {:>6} {:>6} {:>6} {:>6}  {:>6} {:>6} {:>6} {:>6}",
            "", "wake", "light", "deep", "rem", "wake", "light", "deep", "rem"
        );
        let mut ranked: Vec<(f64, String)> = Vec::new();
        for (f, name) in cardiac::NAMES.iter().enumerate() {
            let (r, rb) = cells(&raw, &y, f);
            let (zc, zb) = cells(&z, &y, f);
            ranked.push((zb, (*name).to_string()));
            println!(
                "  {name:<14} {:>6} {:>6} {:>6} {:>6}  {:>6} {:>6} {:>6} {:>6}   {rb:.3}  {zb:.3}",
                r[0], r[1], r[2], r[3], zc[0], zc[1], zc[2], zc[3]
            );
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = ranked.iter().take(5).map(|(v, n)| format!("{n} {v:.3}")).collect();
        println!("  strongest under Z: {}\n", top.join(" | "));
    }
    println!("AUC is one-vs-rest; 0.5 is no information and distance from 0.5 either way is signal.");
    println!("A feature that only separates under EXACT is an ECG artefact, not a candidate.");
    println!("A feature that separates under RAW but not Z is reading who the sleeper is.");
}
