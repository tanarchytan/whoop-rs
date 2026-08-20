//! Does a DREAMT-fitted feature distribution describe a real WHOOP strap at all?
//!
//!   cargo run --release -p physio-algo --example tanv1_transfer
//!
//! `fit_tanv1` fits on DREAMT and reports held-out on two other PSG-labelled cohorts, all three
//! research hardware. The straps this ships on are in the user cohort, which has no stage truth -
//! but transfer is a question about DISTRIBUTIONS, and that needs no labels.
//!
//! EVERY row here is an all-epoch distribution, DREAMT included. Comparing DREAMT's LABELLED rows
//! against a real strap's whole night measures DREAMT's own labelling window, which opens ~24% into
//! its span: that comparison scores 0.346 against DREAMT itself, larger than any cohort distance
//! below. The labelled-only row is printed as that floor rather than used as the reference.

mod common;

use common::{cardiac_series, dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth,
    user_cohort};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::{AccelSample, HrSample, RrRun};
use std::collections::BTreeMap;

const EPOCH: i64 = 30;
const NCOL: usize = Features::N;
/// Shortest staged night worth a distribution. Below this the per-night z-scores are unstable.
const MIN_EPOCHS: usize = 120;
/// Nights per user store. Every store is a different size and one 338-night wearer would otherwise
/// decide the cohort's answer on its own.
const PER_STORE: usize = 40;
/// Beats in one second above which the second is a storage artefact rather than a rhythm.
const MAX_BEATS_PER_SEC: usize = 4;
/// One cohort's distance: mean over columns, its worst column, and the ranked list.
type Distance = (f64, f64, Vec<(f64, &'static str)>);

/// Columns a mean shift CANNOT measure: four are within-night transforms whose pooled mean is fixed
/// by construction (0 for a z-score, 0.5 for a rank) and `clock` averages to 0.5 for any whole
/// window. They read 0.000 for every whole-window cohort but NOT for a trimmed subset.
const DEGENERATE: [&str; 5] = ["hr_z", "hr_var_z", "hr_flat_pct", "resp_z", "clock"];

/// One night's feature rows, through the same cardiac pipeline the fitter uses.
fn night_rows(
    w0: i64,
    w1: i64,
    n: usize,
    hr: &[HrSample],
    rr: &[RrRun],
    grav: &[AccelSample],
) -> Vec<[f64; NCOL]> {
    let card = cardiac_series(w0, n, EPOCH, hr, rr);
    extract(grav, w0, w1, &card).iter().map(|f| f.values()).collect()
}

/// One PSG cohort's rows. `labelled_only` reproduces the subset the fitter standardises on; every
/// comparison below uses ALL epochs on both sides, because a real strap cannot offer the subset.
fn golden_rows(set: &str, labelled_only: bool) -> Vec<[f64; NCOL]> {
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let truth = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let grav = read_accel(dir);
        // The store arm's length rule, so both sides of every distance admit the same nights.
        let epochs = ((w1 - w0) / EPOCH).max(0) as usize;
        if truth.is_empty() || grav.is_empty() || epochs < MIN_EPOCHS {
            continue;
        }
        let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);
        let rows = night_rows(w0, w1, n, &read_hr(dir), &read_rr(dir), &grav);
        if labelled_only {
            out.extend(truth.keys().filter_map(|k| rows.get(*k).copied()));
        } else {
            out.extend(rows);
        }
    }
    out
}

/// Staged nights out of one real backup, EVENLY SPACED across the part of the store its raw streams
/// cover and capped so a large store cannot decide the cohort. Returns the rows and what was left
/// behind.
fn store_rows(path: &str) -> (Vec<[f64; NCOL]>, String) {
    let Ok(cx) = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) else {
        return (Vec::new(), "unreadable".into());
    };
    let Ok(mut q) = cx.prepare(
        "SELECT startTs, endTs FROM sleepSession WHERE stagesJSON IS NOT NULL AND stagesJSON != '' ORDER BY startTs",
    ) else {
        return (Vec::new(), "no sleepSession".into());
    };
    let all: Vec<(i64, i64)> = q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .collect();

    // A night outside hrSample/gravitySample can never contribute, so spending the cap on one lets
    // a store's stream tail, not its size, set how much it weighs.
    let stream_span = |sql: &str| -> Option<(i64, i64)> {
        cx.query_row(sql, [], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))).ok()
    };
    let (Some((h0, h1)), Some((g0, g1))) = (
        stream_span("SELECT MIN(ts), MAX(ts) FROM hrSample"),
        stream_span("SELECT MIN(ts), MAX(ts) FROM gravitySample"),
    ) else {
        return (Vec::new(), format!("{} sessions, no raw streams", all.len()));
    };
    let (t0, t1) = (h0.max(g0), h1.min(g1));

    // Long enough to z-score, then evenly spaced across the WHOLE of what the streams cover. Taking
    // the first N instead reads one wearer's earliest weeks as if they were the wearer.
    let long_enough: Vec<(i64, i64)> =
        all.iter().copied().filter(|(s, e)| ((e - s) / EPOCH).max(0) as usize >= MIN_EPOCHS).collect();
    let long: Vec<(i64, i64)> =
        long_enough.iter().copied().filter(|(s, e)| *e > t0 && *s < t1).collect();
    let picked: Vec<(i64, i64)> = if long.len() <= PER_STORE {
        long.clone()
    } else {
        (0..PER_STORE).map(|i| long[i * long.len() / PER_STORE]).collect()
    };

    let mut out = Vec::new();
    let mut contributed: Vec<(i64, i64)> = Vec::new();
    let mut short_stream = 0usize;
    let (mut dropped_secs, mut total_secs) = (0usize, 0usize);
    for (s, e) in &picked {
        let (s, e) = (*s, *e);
        let n = ((e - s) / EPOCH).max(0) as usize;
        let hr: Vec<HrSample> = cx
            .prepare_cached("SELECT ts, bpm FROM hrSample WHERE ts >= ?1 AND ts < ?2 ORDER BY ts")
            .and_then(|mut p| {
                p.query_map([s, e], |r| {
                    Ok(HrSample { ts: r.get(0)?, bpm: r.get::<_, i64>(1)? as u16 })
                })
                .map(|it| it.filter_map(Result::ok).collect())
            })
            .unwrap_or_default();
        let grav: Vec<AccelSample> = cx
            .prepare_cached(
                "SELECT ts, x, y, z FROM gravitySample WHERE ts >= ?1 AND ts < ?2 ORDER BY ts",
            )
            .and_then(|mut p| {
                p.query_map([s, e], |r| {
                    Ok(AccelSample { ts: r.get(0)?, x: r.get(1)?, y: r.get(2)?, z: r.get(3)? })
                })
                .map(|it| it.filter_map(Result::ok).collect())
            })
            .unwrap_or_default();
        let mut by: BTreeMap<i64, Vec<u16>> = BTreeMap::new();
        if let Ok(mut p) = cx.prepare_cached(
            "SELECT ts, rrMs FROM rrInterval WHERE ts >= ?1 AND ts < ?2 ORDER BY ts",
        ) {
            if let Ok(it) = p.query_map([s, e], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))) {
                for row in it.filter_map(Result::ok) {
                    by.entry(row.0).or_default().push(row.1 as u16);
                }
            }
        }
        // A second holding more beats than a heart can produce is a storage artefact, not a rhythm.
        // Dropped: this arm reads RAW stores while the fixture corpus is already repaired.
        let before = by.len();
        by.retain(|_, v| v.len() <= MAX_BEATS_PER_SEC);
        dropped_secs += before - by.len();
        total_secs += before;
        let rr: Vec<RrRun> =
            by.into_iter().map(|(ts, intervals)| RrRun { ts, intervals }).collect();
        if hr.is_empty() || grav.is_empty() {
            short_stream += 1;
            continue;
        }
        contributed.push((s, e));
        out.extend(night_rows(s, e, n, &hr, &rr, &grav));
    }
    let span_days = all.last().map_or(0, |l| (l.1 - all[0].0) / 86_400);
    // The span of the nights that actually reached `out`, not of the nights that were picked.
    let kept_days = contributed.last().map_or(0, |l| (l.1 - contributed[0].0) / 86_400);
    let beat_pct = if total_secs > 0 {
        100.0 * dropped_secs as f64 / total_secs as f64
    } else {
        0.0
    };
    // The four categories plus the contributed count sum to `all.len()`; drop one and a capped
    // store reads as a loader fault.
    let note = format!(
        "{} of {} sessions ({} short, {} off-stream, {} over the {PER_STORE}-night cap, {} no stream), {} of {} days, {beat_pct:.2}% impossible beat-seconds",
        contributed.len(), all.len(), all.len() - long_enough.len(),
        long_enough.len() - long.len(), long.len() - picked.len(), short_stream, kept_days,
        span_days
    );
    (out, note)
}

/// Mean and sd per column, ignoring NaN.
fn stats(x: &[[f64; NCOL]]) -> ([f64; NCOL], [f64; NCOL], [usize; NCOL]) {
    let (mut m, mut s, mut cnt) = ([f64::NAN; NCOL], [f64::NAN; NCOL], [0usize; NCOL]);
    for c in 0..NCOL {
        let v: Vec<f64> = x.iter().map(|r| r[c]).filter(|v| v.is_finite()).collect();
        cnt[c] = v.len();
        if v.len() < 2 {
            continue;
        }
        m[c] = v.iter().sum::<f64>() / v.len() as f64;
        s[c] = (v.iter().map(|z| (z - m[c]).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    }
    (m, s, cnt)
}

fn main() {
    println!("Standardised mean shift from DREAMT, in DREAMT sd units. ALL-EPOCH on both sides.");
    println!("{DEGENERATE:?} are EXCLUDED: a per-night z-score pools to 0, a rank and clock");
    println!("to 0.5 by construction, so they read 0.000 for every whole-window cohort but not");
    println!("for the labelled floor - leaving them in would flatter every row against it.\n");

    let train = golden_rows("dreamt", false);
    if train.is_empty() {
        println!("no DREAMT rows - check the fixture root");
        return;
    }
    let (tm, ts, _) = stats(&train);
    println!("DREAMT reference: {} rows (all epochs)\n", train.len());

    let mut cohorts: Vec<(String, Vec<[f64; NCOL]>)> = Vec::new();
    // The floor: DREAMT against ITSELF, labelled rows only. Any cohort distance below this one is
    // smaller than the artefact of which epochs carry a reference label.
    cohorts.push(("DREAMT labelled (FLOOR)".to_string(), golden_rows("dreamt", true)));
    for set in ["aauwss", "sleep-accel"] {
        cohorts.push((format!("{set} (PSG held-out)"), golden_rows(set, false)));
    }
    let mut all_user: Vec<[f64; NCOL]> = Vec::new();
    for (wearer, path) in user_cohort() {
        let (rows, note) = store_rows(&path);
        println!("  {wearer:<14} {note}");
        if rows.is_empty() {
            continue;
        }
        all_user.extend_from_slice(&rows);
        cohorts.push((format!("{wearer} (real strap)"), rows));
    }
    if all_user.is_empty() {
        println!("no user rows - is the cohort manifest reachable?");
    } else {
        cohorts.push(("ALL REAL STRAPS".to_string(), all_user));
    }

    // Mean AND max, each as a multiple of the floor's. They disagree, and reading only the mean
    // says real straps match the PSG cohorts while the max says they are worse on their worst column.
    let summarise = |rows: &[[f64; NCOL]]| -> Option<Distance> {
        if rows.len() < 100 {
            return None;
        }
        let (m, _, cnt) = stats(rows);
        let mut d: Vec<(f64, &str)> = (0..NCOL)
            .filter(|c| !DEGENERATE.contains(&Features::NAMES[*c]))
            .filter(|c| cnt[*c] > 100 && ts[*c].is_finite() && ts[*c] > 1e-12)
            .map(|c| (((m[c] - tm[c]) / ts[c]).abs(), Features::NAMES[c]))
            .filter(|(v, _)| v.is_finite())
            .collect();
        if d.is_empty() {
            return None;
        }
        let mean = d.iter().map(|(v, _)| v).sum::<f64>() / d.len() as f64;
        d.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        Some((mean, d[0].0, d))
    };

    let floor = cohorts.first().and_then(|(_, r)| summarise(r));
    println!("\n{:<26} {:>7} {:>7} {:>7} {:>7} {:>5}   worst column",
             "cohort", "mean", "xfloor", "max", "xfloor", "cols");
    for (name, rows) in &cohorts {
        let Some((mean, max, d)) = summarise(rows) else {
            println!("{name:<26} {:>7}   too few rows to compare", rows.len());
            continue;
        };
        let (fm, fx) = floor.as_ref().map_or((f64::NAN, f64::NAN), |(a, b, _)| (*a, *b));
        println!("{name:<26} {mean:>7.3} {:>7.2} {max:>7.3} {:>7.2} {:>5}   {} {:.2}",
                 mean / fm, max / fx, d.len(), d[0].1, d[0].0);
    }

    println!("\nRead every row against the FLOOR, not against zero, and read BOTH statistics. A");
    println!("cohort can match on the mean and be far out on its single worst column, which is what");
    println!("a fitted weight applied off its estimated range actually looks like.");
    println!("`cols` is how many columns survived: a cohort short of the floor's count is");
    println!("averaging a different set, so its xfloor is not a like-for-like ratio.");
}
