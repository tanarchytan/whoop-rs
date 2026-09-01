//! Screen 2 of the roster: does the order-statistic family carry anything ON TOP of the R-R
//! summaries we could compute anyway?
//!
//!   cargo run --release -p physio-algo --example cardiac_multivariate [nights]
//!
//! `mesa_screen` reports a MARGINAL AUC per column, which says a column is not noise. It cannot say
//! the column adds anything, because the seven absolute percentiles are near-copies of each other
//! and of the median. Only a multivariate fit against a baseline answers that, and only held out.
//!
//! Nested baselines, every column from `sleep::cardiac` so this measures one producer:
//!   B0   mean HR alone - the LEVEL, which the shipped recipe already has as `hr_z`
//!   B1   B0 + RMSSD, pNN50, mean|dRR| - the standard R-R summaries, which it does NOT have
//!   +f   B1 plus one candidate percentile column
//!   +14  B1 plus the whole percentile family at once, which is the decision-relevant number
//!
//! Folds are by RECORDING. Splitting epochs would put the same sleeper on both sides and every
//! column would look informative.
//!
//! The gate is the PERMUTED-COLUMN NULL: the same column shuffled WITHIN its own night, which keeps
//! its distribution and its per-night scaling and destroys only its alignment to stage. A gain that
//! the null also produces is the fit's freedom, not the feature. Five earlier candidates died here.

mod common;

use common::mesa::{self, MesaNight};
use physio_algo::lda::Lda;
use physio_algo::sleep::cardiac;
use physio_algo::sleep::metrics::{balanced_accuracy, confusion4, recall, Confusion4};

const EPOCH_S: f64 = 30.0;
const WINDOW_S: f64 = 270.0;
const FOLDS: usize = 5;
/// Ridge on the pooled within-class scatter. The percentile columns are near-collinear by
/// construction, so without it the solve fails outright. Swept per arm, because a result that
/// depends on the regulariser is a result about the regulariser.
const RIDGE: f64 = 1e-2;
const MEAN_HR: usize = 14;
const RR_SUMMARY: [usize; 3] = [15, 16, 17];
/// The 14 candidates: seven absolute percentiles then seven detrended.
const PCTL: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
const PERM_SEED: u64 = 0x5EED_C0DE;

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

/// Per-night rows, each column z-scored within its own night — the form the stager consumes. A
/// column the night cannot carry stays NaN rather than becoming a manufactured mean.
///
/// `arm` degrades the beats first. EXACT asks whether the information is there at all; our band
/// stamps beats to the whole second and drops them, so TIMING and COVERAGE are what we could
/// actually read. A column that survives only under EXACT is an ECG artefact, not a candidate.
fn build(nights: &[MesaNight], arm: &str) -> Vec<Row> {
    let mut out = Vec::new();
    for (ni, n) in nights.iter().enumerate() {
        let beats = match arm {
            "TIMING" => mesa::degrade_timing(&n.beats),
            "COVERAGE" => mesa::degrade_coverage(&n.beats, mesa::COVERAGE_KEEP, mesa::COVERAGE_SEED),
            _ => n.beats.clone(),
        };
        let pairs: Vec<(f64, f64)> = beats.iter().map(|b| (b.t, b.rr)).collect();
        let (mut raw, mut ys) = (Vec::new(), Vec::new());
        for (e, s) in n.stage.iter().enumerate() {
            let Some(s) = s else { continue };
            let c = e as f64 * EPOCH_S + EPOCH_S / 2.0;
            if let Some(b) = cardiac::extract(&pairs, c - WINDOW_S / 2.0, c + WINDOW_S / 2.0) {
                raw.push(b.row());
                ys.push(*s);
            }
        }
        let mut z = vec![[f64::NAN; 18]; raw.len()];
        for f in 0..18 {
            let col: Vec<Option<f64>> =
                raw.iter().map(|r| r[f].is_finite().then_some(r[f])).collect();
            for (k, v) in cardiac::zscore_column(&col).into_iter().enumerate() {
                z[k][f] = v.unwrap_or(f64::NAN);
            }
        }
        for (x, y) in z.into_iter().zip(ys) {
            out.push(Row { night: ni, x, y });
        }
    }
    out
}

/// Held-out confusion over all folds, splitting by RECORDING. `None` if any fold cannot fit.
fn held_out_at(rows: &[Row], cols: &[usize], ridge: f64) -> Option<Confusion4> {
    let mut cm = [[0i64; 4]; 4];
    for fold in 0..FOLDS {
        let pick = |r: &Row, train: bool| (r.night % FOLDS == fold) != train;
        let take = |train: bool| -> (Vec<Vec<f64>>, Vec<usize>) {
            let sel: Vec<&Row> = rows.iter().filter(|r| pick(r, train)).collect();
            (sel.iter().map(|r| cols.iter().map(|c| r.x[*c]).collect()).collect(),
             sel.iter().map(|r| r.y).collect())
        };
        let (xtr, ytr) = take(true);
        let m = Lda::fit(&xtr, &ytr, ridge)?;
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
    }
    Some(cm)
}

fn held_out(rows: &[Row], cols: &[usize]) -> Option<Confusion4> {
    held_out_at(rows, cols, RIDGE)
}

/// Shuffle the named columns WITHIN each night. Distribution and per-night scaling survive; only the
/// alignment to stage is destroyed, which is exactly the thing being claimed.
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

/// `fonseca2015` eq 5: absolute standardised mean difference of one class against the rest, using
/// the pooled SD. A second ranking, so the ordering is not an artefact of using AUC.
fn asmd(rows: &[Row], f: usize, c: usize) -> Option<f64> {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for r in rows {
        if r.x[f].is_finite() {
            if r.y == c { &mut a } else { &mut b }.push(r.x[f]);
        }
    }
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var = |v: &[f64], mu: f64| v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    let (ma, mb) = (m(&a), m(&b));
    let pooled = (((a.len() - 1) as f64 * var(&a, ma) + (b.len() - 1) as f64 * var(&b, mb))
        / (a.len() + b.len() - 2) as f64)
        .sqrt();
    (pooled > 1e-12).then(|| ((ma - mb) / pooled).abs())
}

fn ba(cm: &Confusion4) -> f64 {
    balanced_accuracy(cm).unwrap_or(f64::NAN)
}

/// Predicted share of each class. Balanced accuracy is a mean of RECALLS, and a recall can be
/// bought by calling the class more often, so a gain is only a gain if the calling did not run
/// ahead of it. Precision is the same guard from the other side.
fn calls(cm: &Confusion4) -> [f64; 4] {
    let tot: i64 = cm.iter().flatten().sum();
    std::array::from_fn(|c| cm.iter().map(|r| r[c]).sum::<i64>() as f64 / tot.max(1) as f64)
}

fn main() {
    let limit: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(60);
    let nights = mesa::nights(limit);
    println!("CARDIAC MULTIVARIATE SCREEN — {} nights", nights.len());
    println!("Held out by RECORDING, {FOLDS} folds. Uniform priors in the fit, so a class is not");
    println!("called more just for being common. `call%` is printed because balanced accuracy is a");
    println!("mean of RECALLS, and a recall can be bought by calling the class more often.");
    for arm in ["EXACT", "TIMING", "COVERAGE"] {
        run(&nights, arm);
    }
    println!("\n  EXACT says the information exists in an ECG. TIMING and COVERAGE are what our own");
    println!("  channel could read, and a column that survives only under EXACT is not a candidate.");
    println!("  A column beating its own permuted null is one; five earlier candidates died there.");
}

fn run(nights: &[MesaNight], arm: &str) {
    let rows = build(nights, arm);
    let mut mix = [0usize; 4];
    for r in &rows {
        mix[r.y] += 1;
    }
    println!("\n\n===== {arm} =====  {} epochs (w/l/d/r {mix:?})", rows.len());

    let b0 = vec![MEAN_HR];
    let mut b1 = b0.clone();
    b1.extend(RR_SUMMARY);
    let (Some(c0), Some(c1)) = (held_out(&rows, &b0), held_out(&rows, &b1)) else {
        println!("a baseline fold could not fit — nothing below would mean anything");
        return;
    };
    let show = |tag: &str, cm: &Confusion4| {
        let r: Vec<String> =
            (0..4).map(|c| recall(cm, c).map_or("  -  ".into(), |v| format!("{v:.3}"))).collect();
        let q = calls(cm);
        println!(
            "  {tag:<26} BA {:.4}   recall w/l/d/r  {}   call% {:.0}/{:.0}/{:.0}/{:.0}",
            ba(cm), r.join(" "), q[0] * 100.0, q[1] * 100.0, q[2] * 100.0, q[3] * 100.0
        );
    };
    show("B0  mean HR", &c0);
    show("B1  + rmssd/pnn50/|dRR|", &c1);
    println!();

    // One candidate at a time, against B1, each with its own permuted-column null.
    println!(
        "  {:<12} {:>8} {:>9} {:>9} {:>8}   {:<23} call% w/l/d/r",
        "candidate", "BA", "d(BA)", "null d", "verdict", "recall w/l/d/r"
    );
    let mut ranked: Vec<(f64, usize, f64)> = Vec::new();
    for f in PCTL {
        let mut cols = b1.clone();
        cols.push(f);
        let Some(cm) = held_out(&rows, &cols) else { continue };
        let d = ba(&cm) - ba(&c1);
        let nulls: Vec<f64> = (0..2)
            .filter_map(|k| {
                let pr = permuted(&rows, &[f], PERM_SEED ^ k);
                held_out(&pr, &cols).map(|c| ba(&c) - ba(&c1))
            })
            .collect();
        let null = nulls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let verdict = if d > null { "over" } else { "NULL" };
        let r: Vec<String> =
            (0..4).map(|c| recall(&cm, c).map_or("  -  ".into(), |v| format!("{v:.3}"))).collect();
        let q = calls(&cm);
        println!(
            "  {:<12} {:>8.4} {:>+9.4} {:>+9.4} {:>8}   {:<23} {:.0}/{:.0}/{:.0}/{:.0}",
            cardiac::NAMES[f], ba(&cm), d, null, verdict, r.join(" "),
            q[0] * 100.0, q[1] * 100.0, q[2] * 100.0, q[3] * 100.0
        );
        ranked.push((d, f, null));
    }

    // The number that decides the phase: the whole family at once, against its own family null.
    let mut all = b1.clone();
    all.extend(PCTL);
    if let Some(cm) = held_out(&rows, &all) {
        let d = ba(&cm) - ba(&c1);
        let null = (0..2)
            .filter_map(|k| held_out(&permuted(&rows, &PCTL, PERM_SEED ^ k), &all).map(|c| ba(&c) - ba(&c1)))
            .fold(f64::NEG_INFINITY, f64::max);
        println!();
        show("B1 + ALL 14 percentiles", &cm);
        println!("  family d(BA) {d:+.4} against a permuted-family null of {null:+.4}");
        let sweep: Vec<String> = [1e-3, 1e-2, 1e-1, 1.0]
            .iter()
            .map(|r| {
                held_out_at(&rows, &all, *r)
                    .map_or(format!("{r:.0e}=fail"), |c| format!("{r:.0e}={:.4}", ba(&c)))
            })
            .collect();
        println!("  ridge sweep on the family BA: {}", sweep.join("  "));
    }

    println!("\n  ASMD (fonseca2015 eq 5), a second ranking so the order is not an AUC artefact:");
    println!("  {:<12} {:>7} {:>7} {:>7} {:>7}", "column", "wake", "light", "deep", "rem");
    let mut by_asmd: Vec<(f64, usize)> = Vec::new();
    for f in PCTL.iter().chain(&RR_SUMMARY).chain(std::iter::once(&MEAN_HR)) {
        let v: Vec<Option<f64>> = (0..4).map(|c| asmd(&rows, *f, c)).collect();
        let best = v.iter().flatten().cloned().fold(0.0, f64::max);
        by_asmd.push((best, *f));
        let cells: Vec<String> =
            v.iter().map(|x| x.map_or("  -  ".into(), |v| format!("{v:.3}"))).collect();
        println!("  {:<12} {:>7} {:>7} {:>7} {:>7}", cardiac::NAMES[*f], cells[0], cells[1], cells[2], cells[3]);
    }
    by_asmd.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top = |v: &[(f64, usize)]| -> String {
        v.iter().take(5).map(|(s, f)| format!("{} {s:.3}", cardiac::NAMES[*f])).collect::<Vec<_>>().join(" | ")
    };
    println!("\n  by held-out d(BA): {}", top(&ranked.iter().map(|(d, f, _)| (*d, *f)).collect::<Vec<_>>()));
    println!("  by ASMD:           {}", top(&by_asmd));
    println!("\n  A column beating its own permuted null is a candidate. One that does not is the fit's");
    println!("  freedom, and five earlier candidates died exactly there.");
}
