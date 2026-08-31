//! Screen 1 of the roster: do the ORDER STATISTICS carry stage information, and do they survive our
//! channel?
//!
//!   cargo run --release -p physio-algo --example mesa_screen [nights]
//!
//! `FEATURE-ROSTER.md` puts these first: seven percentiles of detrended and absolute HR/RR is the
//! widest gap between our feature set and radha2019's, it needs no spectral estimation, and it has no
//! sensitivity to sub-second timing. RMSSD / pNN50 / mean-|dRR| ride along because they are the other
//! constructs the corpus names that we do not compute.
//!
//! Three arms on the SAME nights, because a feature that only works on ECG is not a candidate:
//!   EXACT     MESA's own beat times at 1/256 s — is the information there at all
//!   TIMING    the same beats through our wire format and back — what whole-second stamps cost
//!   COVERAGE  beats dropped in runs to 60% — what PPG dropout costs
//!
//! Reported as one-vs-rest AUC per stage. 0.5 is no information; distance from 0.5 in EITHER
//! direction is signal, so the table prints |AUC-0.5| and the sign separately.

mod common;

use common::mesa::{self, Beat};

const EPOCH_S: f64 = 30.0;
/// radha2019 computes its features on a 4.5 min window centred on the epoch.
const WINDOW_S: f64 = 270.0;
const PCTS: [f64; 7] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];
const MIN_BEATS: usize = 20;
const COVERAGE_KEEP: f64 = 0.60;
const COVERAGE_SEED: u64 = 0xC0FFEE;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

/// Least-squares linear trend removed. "Detrended" is not defined in the source, so this is OUR
/// reading of it and is labelled as such wherever it is reported.
fn detrended(t: &[f64], v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let (mt, mv) = (t.iter().sum::<f64>() / n, v.iter().sum::<f64>() / n);
    let sxx: f64 = t.iter().map(|x| (x - mt).powi(2)).sum();
    let sxy: f64 = t.iter().zip(v).map(|(x, y)| (x - mt) * (y - mv)).sum();
    let slope = if sxx > f64::EPSILON { sxy / sxx } else { 0.0 };
    t.iter().zip(v).map(|(x, y)| y - (mv + slope * (x - mt))).collect()
}

fn names() -> Vec<String> {
    let mut n: Vec<String> = Vec::new();
    for p in PCTS {
        n.push(format!("rr_p{:02}", (p * 100.0) as u32));
    }
    for p in PCTS {
        n.push(format!("rr_dt_p{:02}", (p * 100.0) as u32));
    }
    n.extend(["mean_hr".into(), "rmssd".into(), "pnn50".into(), "mean_abs_drr".into()]);
    n
}

/// One epoch's feature row, or `None` when the window cannot carry one.
fn row(beats: &[Beat], centre: f64) -> Option<Vec<f64>> {
    let (lo, hi) = (centre - WINDOW_S / 2.0, centre + WINDOW_S / 2.0);
    let win: Vec<&Beat> = beats.iter().filter(|b| b.t >= lo && b.t <= hi).collect();
    if win.len() < MIN_BEATS {
        return None;
    }
    let t: Vec<f64> = win.iter().map(|b| b.t).collect();
    let rr: Vec<f64> = win.iter().map(|b| b.rr).collect();

    let mut abs = rr.clone();
    abs.sort_by(f64::total_cmp);
    let mut dt = detrended(&t, &rr);
    dt.sort_by(f64::total_cmp);

    let mut out: Vec<f64> = PCTS.iter().map(|p| percentile(&abs, *p)).collect();
    out.extend(PCTS.iter().map(|p| percentile(&dt, *p)));

    let mean_rr = rr.iter().sum::<f64>() / rr.len() as f64;
    out.push(60_000.0 / mean_rr);
    let d: Vec<f64> = rr.windows(2).map(|w| w[1] - w[0]).collect();
    out.push((d.iter().map(|x| x * x).sum::<f64>() / d.len().max(1) as f64).sqrt());
    out.push(d.iter().filter(|x| x.abs() > 50.0).count() as f64 / d.len().max(1) as f64);
    out.push(d.iter().map(|x| x.abs()).sum::<f64>() / d.len().max(1) as f64);
    Some(out)
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

fn collect(nights: &[mesa::MesaNight], arm: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for n in nights {
        let beats = match arm {
            "TIMING" => mesa::degrade_timing(&n.beats),
            "COVERAGE" => mesa::degrade_coverage(&n.beats, COVERAGE_KEEP, COVERAGE_SEED),
            _ => n.beats.clone(),
        };
        for (e, s) in n.stage.iter().enumerate() {
            let Some(s) = s else { continue };
            let centre = e as f64 * EPOCH_S + EPOCH_S / 2.0;
            if let Some(r) = row(&beats, centre) {
                x.push(r);
                y.push(*s);
            }
        }
    }
    (x, y)
}

fn main() {
    let limit: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(60);
    let nights = mesa::nights(limit);
    let names = names();
    println!("MESA order-statistics screen — {} nights, window {WINDOW_S} s centred", nights.len());
    println!("`rr_dt_*` are detrended by a least-squares linear fit; the source does not define it.\n");

    for arm in ["EXACT", "TIMING", "COVERAGE"] {
        let (x, y) = collect(&nights, arm);
        let mut mix = [0usize; 4];
        for c in &y {
            mix[*c] += 1;
        }
        println!("== {arm} == {} epochs  (w/l/d/r {mix:?})", x.len());
        println!("  {:<14} {:>8} {:>8} {:>8} {:>8}   best", "feature", "wake", "light", "deep", "rem");
        let mut ranked: Vec<(f64, String)> = Vec::new();
        for (f, name) in names.iter().enumerate() {
            let col: Vec<f64> = x.iter().map(|r| r[f]).collect();
            let a: Vec<Option<f64>> = (0..4).map(|c| auc(&col, &y, c)).collect();
            let cells: Vec<String> = a
                .iter()
                .map(|v| match v {
                    Some(v) => format!("{v:.3}"),
                    None => "-".into(),
                })
                .collect();
            let best = a.iter().flatten().map(|v| (v - 0.5).abs()).fold(0.0, f64::max);
            ranked.push((best, name.clone()));
            println!(
                "  {name:<14} {:>8} {:>8} {:>8} {:>8}   {best:.3}",
                cells[0], cells[1], cells[2], cells[3]
            );
        }
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = ranked.iter().take(5).map(|(v, n)| format!("{n} {v:.3}")).collect();
        println!("  strongest: {}\n", top.join(" | "));
    }
    println!("AUC is one-vs-rest; 0.5 is no information and distance from 0.5 either way is signal.");
    println!("A feature that only separates under EXACT is an ECG artefact, not a candidate.");
}
