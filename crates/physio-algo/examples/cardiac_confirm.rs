//! Screen 2, step 3: does the order-statistic family survive on a WRIST cohort?
//!
//!   cargo run --release -p physio-algo --example cardiac_confirm
//!
//! `cardiac_multivariate` measured +0.087 balanced accuracy on 400 MESA recordings. MESA is ECG.
//! It SELECTS a feature and it cannot confirm one, because the thing being asked is whether OUR
//! channel carries it — and MESA's channel is far better than ours even after both degradations.
//!
//! AAUWSS is the confirmation set: wrist optical, PSG-scored, and its R-R reaches us the way the
//! band delivers it — several beats sharing one whole-second stamp, reconstructed by
//! `common::reconstruct_beats` exactly as the stager does before any statistic sees them.
//!
//! DREAMT is excluded deliberately and not by oversight: 34.4% of its intervals repeat their
//! predecessor because the fixture builder synthesised them, so a percentile family read off it
//! would be scoring the builder.
//!
//! Thirteen recordings is few, so this is LEAVE-ONE-RECORDING-OUT rather than 5-fold, and the
//! permuted-column null is what decides anything. A confirmation that cannot fail is not one:
//! if the family lands on its null here, MESA selected an ECG artefact and the phase is over.

mod common;

use common::{dirs_of, read_meta, read_rr, read_truth, reconstruct_beats, require_psg};
use physio_algo::lda::Lda;
use physio_algo::sleep::cardiac;
use physio_algo::sleep::metrics::{balanced_accuracy, confusion4, recall, Confusion4};

const COHORT: &str = "aauwss";
const EPOCH_S: f64 = 30.0;
const WINDOW_S: f64 = 270.0;
const RIDGE: f64 = 1e-2;
const MEAN_HR: usize = 14;
const RR_SUMMARY: [usize; 3] = [15, 16, 17];
const PCTL: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
const PERM_SEED: u64 = 0x5EED_C0DE;
/// Two draws is thin, and it is what 13 recordings will carry in reasonable time. The null is
/// reported as its RANGE rather than a single figure so that thinness is visible.
const NULL_DRAWS: u64 = 3;

struct Row {
    night: usize,
    x: [f64; 18],
    y: usize,
}

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One row per labelled epoch that has enough beats, z-scored within its own night.
fn build() -> (Vec<Row>, usize) {
    require_psg(COHORT);
    let (mut out, mut nights) = (Vec::new(), 0usize);
    for dir in &dirs_of(COHORT) {
        let truth = read_truth(dir);
        let Some((w0, _, _)) = read_meta(dir) else { continue };
        let beats = reconstruct_beats(&read_rr(dir));
        if truth.is_empty() || beats.len() < 100 {
            continue;
        }
        let (mut raw, mut ys) = (Vec::new(), Vec::new());
        for (k, t) in &truth {
            if !(0..4).contains(t) {
                continue;
            }
            // Windows are centred on the epoch, in the fixture's own clock.
            let c = w0 as f64 + *k as f64 * EPOCH_S + EPOCH_S / 2.0;
            if let Some(b) = cardiac::extract(&beats, c - WINDOW_S / 2.0, c + WINDOW_S / 2.0) {
                raw.push(b.row());
                ys.push(*t as usize);
            }
        }
        if raw.len() < 50 {
            continue;
        }
        let mut z = vec![[f64::NAN; 18]; raw.len()];
        for f in 0..18 {
            let col: Vec<Option<f64>> =
                raw.iter().map(|r| r[f].is_finite().then_some(r[f])).collect();
            for (i, v) in cardiac::zscore_column(&col).into_iter().enumerate() {
                z[i][f] = v.unwrap_or(f64::NAN);
            }
        }
        for (x, y) in z.into_iter().zip(ys) {
            out.push(Row { night: nights, x, y });
        }
        nights += 1;
    }
    (out, nights)
}

/// Leave ONE recording out, every recording in turn. With 13 nights a 5-fold split would train on
/// 10 and test on 3, and the fold-to-fold spread would swamp the effect being measured.
fn loro(rows: &[Row], nights: usize, cols: &[usize]) -> Option<Confusion4> {
    let mut cm = [[0i64; 4]; 4];
    let mut used = 0;
    for held in 0..nights {
        let take = |train: bool| -> (Vec<Vec<f64>>, Vec<usize>) {
            let sel: Vec<&Row> = rows.iter().filter(|r| (r.night == held) != train).collect();
            (sel.iter().map(|r| cols.iter().map(|c| r.x[*c]).collect()).collect(),
             sel.iter().map(|r| r.y).collect())
        };
        let (xtr, ytr) = take(true);
        // A night whose held-out set lacks a class cannot be fitted against; skip it rather than
        // drop the class, and report how many contributed.
        let Some(m) = Lda::fit(&xtr, &ytr, RIDGE) else { continue };
        let (xte, yte) = take(false);
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (row, y) in xte.iter().zip(&yte) {
            if let Some(c) = m.predict(row) {
                p.push(c);
                t.push(*y);
            }
        }
        let f = confusion4(&p, &t);
        for (o, a) in cm.iter_mut().zip(&f) {
            for (x, y) in o.iter_mut().zip(a) {
                *x += *y;
            }
        }
        used += 1;
    }
    (used > 0).then_some(cm)
}

/// Shuffle the named columns WITHIN each night: distribution and per-night scaling survive, only the
/// alignment to stage is destroyed.
fn permuted(rows: &[Row], cols: &[usize], seed: u64) -> Vec<Row> {
    let mut out: Vec<Row> = rows.iter().map(|r| Row { night: r.night, x: r.x, y: r.y }).collect();
    let mut start = 0;
    while start < out.len() {
        let mut end = start;
        while end < out.len() && out[end].night == out[start].night {
            end += 1;
        }
        for (k, c) in cols.iter().enumerate() {
            let mut s = seed ^ splitmix((out[start].night as u64) << 8 | k as u64);
            for i in (start + 1..end).rev() {
                s = splitmix(s);
                let j = start + (s >> 11) as usize % (i - start + 1);
                let tmp = out[i].x[*c];
                out[i].x[*c] = out[j].x[*c];
                out[j].x[*c] = tmp;
            }
        }
        start = end;
    }
    out
}

fn ba(cm: &Confusion4) -> f64 {
    balanced_accuracy(cm).unwrap_or(f64::NAN)
}

fn calls(cm: &Confusion4) -> [f64; 4] {
    let tot: i64 = cm.iter().flatten().sum();
    std::array::from_fn(|c| cm.iter().map(|r| r[c]).sum::<i64>() as f64 / tot.max(1) as f64)
}

fn show(tag: &str, cm: &Confusion4) {
    let r: Vec<String> =
        (0..4).map(|c| recall(cm, c).map_or("  -  ".into(), |v| format!("{v:.3}"))).collect();
    let q = calls(cm);
    println!(
        "  {tag:<26} BA {:.4}   recall w/l/d/r  {}   call% {:.0}/{:.0}/{:.0}/{:.0}",
        ba(cm), r.join(" "), q[0] * 100.0, q[1] * 100.0, q[2] * 100.0, q[3] * 100.0
    );
}

fn main() {
    let (rows, nights) = build();
    let mut mix = [0usize; 4];
    for r in &rows {
        mix[r.y] += 1;
    }
    println!("CARDIAC CONFIRMATION on {COHORT} — {nights} recordings, {} epochs (w/l/d/r {mix:?})",
             rows.len());
    println!("Wrist optical, PSG-scored, R-R through our own whole-second stamps. Leave-one-");
    println!("recording-out. MESA selected this family; only a wrist cohort can confirm it.\n");

    let b0 = vec![MEAN_HR];
    let mut b1 = b0.clone();
    b1.extend(RR_SUMMARY);
    let mut all = b1.clone();
    all.extend(PCTL);

    let (Some(c0), Some(c1), Some(c14)) =
        (loro(&rows, nights, &b0), loro(&rows, nights, &b1), loro(&rows, nights, &all))
    else {
        println!("a baseline could not fit on any held-out night - nothing here would mean anything");
        return;
    };
    show("B0  mean HR", &c0);
    show("B1  + rmssd/pnn50/|dRR|", &c1);
    show("B1 + ALL 14 percentiles", &c14);

    let nulls: Vec<f64> = (0..NULL_DRAWS)
        .filter_map(|k| loro(&permuted(&rows, &PCTL, PERM_SEED ^ k), nights, &all).map(|c| ba(&c)))
        .collect();
    let lo = nulls.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = nulls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let d = ba(&c14) - ba(&c1);
    println!("\n  family d(BA) {d:+.4}");
    println!("  permuted-family null d(BA) over {NULL_DRAWS} draws: {:+.4} .. {:+.4}",
             lo - ba(&c1), hi - ba(&c1));

    let (q1, q14) = (calls(&c1)[2], calls(&c14)[2]);
    let (r1, r14) = (recall(&c1, 2).unwrap_or(f64::NAN), recall(&c14, 2).unwrap_or(f64::NAN));
    println!("\n  deep calling {:.1}% -> {:.1}% ({:.2}x) for recall {r1:.3} -> {r14:.3} ({:.2}x)",
             q1 * 100.0, q14 * 100.0, q14 / q1, r14 / r1);
    println!("  Sub-proportional would mean it calls deep more rather than better, which is the");
    println!("  check the transition arms failed. Super-proportional is the one that counts.");

    println!("\n  VERDICT: the family {} its permuted null on a wrist cohort.",
             if d > hi - ba(&c1) { "CLEARS" } else { "DOES NOT CLEAR" });
}
