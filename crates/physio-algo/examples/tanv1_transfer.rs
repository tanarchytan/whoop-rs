//! Does a DREAMT-fitted feature distribution describe a real WHOOP strap at all?
//!
//!   cargo run --release -p physio-algo --example tanv1_transfer
//!
//! `fit_tanv1` fits on DREAMT and reports held-out on two other PSG-labelled cohorts. Both are
//! research hardware. The straps this ships on are in the user cohort, which has no stage truth -
//! but transfer is a question about DISTRIBUTIONS, and that needs no labels.
//!
//! For every column of the feature vector this prints the standardised mean shift from DREAMT's own
//! fitted rows. A column that sits far from DREAMT on real straps is one whose fitted weight is
//! being applied outside the range it was estimated on, whatever the held-out kappa says.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, user_cohort};
use physio_algo::sleep::features::{extract, Cardiac, Features};
use physio_algo::sleep::{flatten_rr, resp_regularity, AccelSample, HrSample, RrRun};
use std::collections::BTreeMap;

const EPOCH: i64 = 30;
const NCOL: usize = Features::N;
/// Shortest staged night worth a distribution. Below this the per-night z-scores are unstable.
const MIN_EPOCHS: usize = 120;
/// Nights per user store. Every store is a different size and one 338-night wearer would otherwise
/// decide the cohort's answer on its own.
const PER_STORE: usize = 40;

/// Per-night z-score of a per-epoch series, missing where the series is.
fn zscore(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = v.iter().flatten().copied().collect();
    if present.len() < 2 {
        return vec![None; v.len()];
    }
    let m = present.iter().sum::<f64>() / present.len() as f64;
    let sd = (present.iter().map(|x| (x - m).powi(2)).sum::<f64>() / present.len() as f64).sqrt();
    if sd <= 0.0 {
        return vec![None; v.len()];
    }
    v.iter().map(|o| o.map(|x| (x - m) / sd)).collect()
}

fn per_second_hr(hr: &[HrSample]) -> BTreeMap<i64, f64> {
    let mut acc: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for s in hr {
        let e = acc.entry(s.ts).or_insert((0.0, 0.0));
        e.0 += s.bpm as f64;
        e.1 += 1.0;
    }
    acc.into_iter().map(|(t, (a, c))| (t, a / c)).collect()
}

fn std_of_seconds(sec: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> Option<f64> {
    let v: Vec<f64> = sec.range(lo..hi).map(|(_, b)| *b).collect();
    if v.len() < 2 {
        return None;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    Some((v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt())
}

fn rank_pct(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut sorted: Vec<f64> = v.iter().flatten().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return vec![None; v.len()];
    }
    v.iter()
        .map(|o| o.map(|x| sorted.partition_point(|s| *s <= x) as f64 / sorted.len() as f64))
        .collect()
}

/// One night's feature rows, built exactly the way the fitter builds them.
fn night_rows(
    w0: i64,
    w1: i64,
    n: usize,
    hr: &[HrSample],
    rr: &[RrRun],
    grav: &[AccelSample],
) -> Vec<[f64; NCOL]> {
    let sec = per_second_hr(hr);
    let mut sum = vec![(0.0f64, 0.0f64); n];
    for s in hr {
        let k = ((s.ts - w0) / EPOCH).max(0) as usize;
        if k < n {
            sum[k].0 += s.bpm as f64;
            sum[k].1 += 1.0;
        }
    }
    let raw: Vec<Option<f64>> = sum.iter().map(|(a, c)| (*c > 0.0).then(|| a / c)).collect();
    let hr_z = zscore(&raw);
    let starts: Vec<i64> = (0..n).map(|k| w0 + k as i64 * EPOCH).collect();
    let hv: Vec<Option<f64>> =
        starts.iter().map(|e| std_of_seconds(&sec, e - 150, e + EPOCH + 150)).collect();
    let hr_var_z = zscore(&hv);
    let flat: Vec<Option<f64>> =
        starts.iter().map(|e| std_of_seconds(&sec, e - 330, e + EPOCH + 360)).collect();
    let flat_pct = rank_pct(&flat);

    let mut beats_by: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (ts, ms) in flatten_rr(rr) {
        beats_by.entry(ts).or_default().push(ms);
    }
    let resp: Vec<Option<f64>> = starts
        .iter()
        .map(|e| {
            let mut beats: Vec<(f64, f64)> = beats_by
                .range(e - 90..e + 120)
                .flat_map(|(t, vs)| vs.iter().map(|v| (*t as f64, v.clamp(300.0, 2000.0))))
                .collect();
            beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
            resp_regularity(&beats)
        })
        .collect();
    let resp_z = zscore(&resp);

    let card: Vec<Cardiac> = (0..n)
        .map(|k| Cardiac {
            hr_z: hr_z[k],
            hr_var_z: hr_var_z[k],
            hr_flat_pct: flat_pct[k],
            resp_z: resp_z[k],
        })
        .collect();
    extract(grav, w0, w1, &card).iter().map(|f| f.values()).collect()
}

/// The DREAMT rows the fitter actually standardises on: labelled epochs only.
fn dreamt_rows() -> Vec<[f64; NCOL]> {
    let mut out = Vec::new();
    for dir in &dirs_of("dreamt") {
        let truth = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let grav = read_accel(dir);
        if truth.is_empty() || grav.is_empty() {
            continue;
        }
        let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);
        let rows = night_rows(w0, w1, n, &read_hr(dir), &read_rr(dir), &grav);
        for k in truth.keys() {
            if let Some(r) = rows.get(*k) {
                out.push(*r);
            }
        }
    }
    out
}

fn golden_rows(set: &str) -> Vec<[f64; NCOL]> {
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
        for k in truth.keys() {
            if let Some(r) = rows.get(*k) {
                out.push(*r);
            }
        }
    }
    out
}

/// Staged nights out of one real backup, capped so a large store cannot dominate.
fn store_rows(path: &str) -> Vec<[f64; NCOL]> {
    let Ok(cx) = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) else {
        return Vec::new();
    };
    let Ok(mut q) = cx.prepare(
        "SELECT startTs, endTs FROM sleepSession WHERE stagesJSON IS NOT NULL AND stagesJSON != '' \
         ORDER BY startTs",
    ) else {
        return Vec::new();
    };
    let sessions: Vec<(i64, i64)> = q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .collect();

    let mut out = Vec::new();
    let mut used = 0usize;
    for (s, e) in sessions {
        if used >= PER_STORE {
            break;
        }
        let n = ((e - s) / EPOCH).max(0) as usize;
        if n < MIN_EPOCHS {
            continue;
        }
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
        // One run per second, matching how the fixtures carry beats.
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
        let rr: Vec<RrRun> =
            by.into_iter().map(|(ts, intervals)| RrRun { ts, intervals }).collect();
        if hr.is_empty() || grav.is_empty() {
            continue;
        }
        out.extend(night_rows(s, e, n, &hr, &rr, &grav));
        used += 1;
    }
    out
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
    println!("Standardised mean shift from DREAMT's fitted rows. |shift| is in DREAMT sd units:");
    println!("above 0.5 the fitted weight for that column is being applied off its estimated range.\n");

    let train = dreamt_rows();
    if train.is_empty() {
        println!("no DREAMT rows - check the fixture root");
        return;
    }
    let (tm, ts, _) = stats(&train);
    println!("DREAMT (fitted): {} rows\n", train.len());

    let mut cohorts: Vec<(String, Vec<[f64; NCOL]>)> = Vec::new();
    for set in ["aauwss", "sleep-accel"] {
        cohorts.push((format!("{set} (PSG held-out)"), golden_rows(set)));
    }
    // Every registered backup, then all of them pooled - one wearer is an anecdote.
    let mut all_user: Vec<[f64; NCOL]> = Vec::new();
    for (wearer, path) in user_cohort() {
        let rows = store_rows(&path);
        if rows.is_empty() {
            println!("  {wearer:<14} no scorable nights");
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

    println!("{:<26} {:>8} {:>8} {:>8}   worst columns", "cohort", "rows", "mean|d|", "max|d|");
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
        let worst: Vec<String> =
            d.iter().take(3).map(|(v, n)| format!("{n} {v:.2}")).collect();
        println!("{name:<26} {:>8} {mean:>8.3} {:>8.3}   {}", rows.len(), d[0].0, worst.join(", "));
    }

    println!("\nThe PSG rows are the distance the held-out kappa already measures. A real strap");
    println!("sitting further away means production transfer is worse than that kappa suggests.");
}
