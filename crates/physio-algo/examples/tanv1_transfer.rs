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
        if truth.is_empty() || grav.is_empty() {
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

/// Staged nights out of one real backup, EVENLY SPACED across the store's whole span and capped so
/// a large store cannot decide the cohort. Returns the rows and what was left behind.
fn store_rows(path: &str) -> (Vec<[f64; NCOL]>, String) {
    let Ok(cx) = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) else {
        return (Vec::new(), "unreadable".into());
    };
    let Ok(mut q) = cx.prepare(
        "SELECT startTs, endTs FROM sleepSession WHERE stagesJSON IS NOT NULL AND stagesJSON != ''          ORDER BY startTs",
    ) else {
        return (Vec::new(), "no sleepSession".into());
    };
    let all: Vec<(i64, i64)> = q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .collect();

    // Long enough to z-score, then a stride across the WHOLE span. Taking the first N instead reads
    // one wearer's earliest weeks as if they were the wearer.
    let long: Vec<(i64, i64)> =
        all.iter().copied().filter(|(s, e)| ((e - s) / EPOCH).max(0) as usize >= MIN_EPOCHS).collect();
    let stride = long.len().div_ceil(PER_STORE).max(1);
    let picked: Vec<(i64, i64)> = long.iter().copied().step_by(stride).take(PER_STORE).collect();

    let mut out = Vec::new();
    let mut short_stream = 0usize;
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
        by.retain(|_, v| v.len() <= MAX_BEATS_PER_SEC);
        let rr: Vec<RrRun> =
            by.into_iter().map(|(ts, intervals)| RrRun { ts, intervals }).collect();
        if hr.is_empty() || grav.is_empty() {
            short_stream += 1;
            continue;
        }
        out.extend(night_rows(s, e, n, &hr, &rr, &grav));
    }
    let span_days = all.last().map_or(0, |l| (l.1 - all[0].0) / 86_400);
    let kept_days = picked.last().map_or(0, |l| (l.1 - picked[0].0) / 86_400);
    let note = format!(
        "{} of {} sessions ({} short, {} no stream), {} of {} days spanned, stride {}",
        picked.len(), all.len(), all.len() - long.len(), short_stream, kept_days, span_days, stride
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
    println!("Standardised mean shift from DREAMT, in DREAMT sd units. ALL-EPOCH on both sides.
");

    let train = golden_rows("dreamt", false);
    if train.is_empty() {
        println!("no DREAMT rows - check the fixture root");
        return;
    }
    let (tm, ts, _) = stats(&train);
    println!("DREAMT reference: {} rows (all epochs)
", train.len());

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

    println!("
{:<26} {:>8} {:>8} {:>8}   worst columns", "cohort", "rows", "mean|d|", "max|d|");
    for (name, rows) in &cohorts {
        if rows.len() < 100 {
            println!("{name:<26} {:>8}   too few rows to compare", rows.len());
            continue;
        }
        let (m, _, cnt) = stats(rows);
        let mut d: Vec<(f64, &str)> = (0..NCOL)
            .filter(|c| cnt[*c] > 100 && ts[*c].is_finite() && ts[*c] > 1e-12)
            .map(|c| (((m[c] - tm[c]) / ts[c]).abs(), Features::NAMES[c]))
            .filter(|(v, _)| v.is_finite())
            .collect();
        if d.is_empty() {
            println!("{name:<26} {:>8}   no comparable column", rows.len());
            continue;
        }
        let mean = d.iter().map(|(v, _)| v).sum::<f64>() / d.len() as f64;
        d.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let worst: Vec<String> = d.iter().take(3).map(|(v, n)| format!("{n} {v:.2}")).collect();
        println!("{name:<26} {:>8} {mean:>8.3} {:>8.3}   {}", rows.len(), d[0].0, worst.join(", "));
    }

    println!("
Read every row against the FLOOR, not against zero. A cohort below it is closer to");
    println!("DREAMT than DREAMT's own labelled subset is, and carries no transfer claim at all.");
}
