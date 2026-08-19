//! Is the cardiac wake signal a property of the WEARER or of the night?
//!
//!   cargo run --release -p physio-algo --example wearer_variance          # the whole user cohort
//!   cargo run --release -p physio-algo --example wearer_variance -- <store.sqlite> [more...]
//!
//! The leading hypothesis is that a global `awake_hr` weight cannot serve two people, because hr_z
//! separates wake at AUC 0.827 for one wearer and 0.316 for another. Two wearers is an anecdote.
//!
//! No PSG cohort can settle it: they are one night per subject, so within-wearer variance is not
//! even defined there. Our stores are 16 to 338 nights of ONE person each, which is the only shape
//! that can answer, and it needs no per-epoch truth - the question is not "are we right" but "is
//! this parameter a stable property of a person".
//!
//! Decomposes per-night hr-wake AUC into within-wearer and between-wearer spread. If between
//! dominates, a per-wearer weight is justified and a global one is fitting to an average nobody has.
//! If within dominates, the sign flip was noise and B2b should be dropped.
//!
//! Labels are our own hypnogram. That biases the ABSOLUTE level, but it is the same engine for every
//! wearer, so it does not manufacture a between-wearer difference.

mod common;

use common::user_cohort;
use physio_algo::sleep::AccelSample;

const EPOCH: i64 = 30;
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

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn per_store(path: &str) -> Option<(String, Vec<f64>, Vec<f64>)> {
    let cx = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )
    .ok()?;
    let mut q = cx
        .prepare(
            "SELECT startTs, endTs, stagesJSON FROM sleepSession \
             WHERE stagesJSON IS NOT NULL AND stagesJSON != '' ORDER BY startTs",
        )
        .ok()?;
    let sessions: Vec<(i64, i64, String)> = q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .ok()?
        .filter_map(Result::ok)
        .collect();

    let (mut hr_aucs, mut mo_aucs) = (Vec::new(), Vec::new());
    for (s, e, sj) in &sessions {
        let n = ((e - s) / EPOCH).max(0) as usize;
        if n < 60 {
            continue;
        }
        let mut is_wake = vec![None; n];
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(sj) else { continue };
        for seg in parsed.as_array().into_iter().flatten() {
            let (Some(a), Some(b), Some(st)) =
                (seg["start"].as_i64(), seg["end"].as_i64(), seg["stage"].as_str())
            else {
                continue;
            };
            let mut t = a;
            while t < b {
                let k = ((t - s) / EPOCH) as usize;
                if t >= *s && k < n {
                    is_wake[k] = Some(st == "wake");
                }
                t += EPOCH;
            }
        }
        // Per-epoch mean heart rate, and peak gravity delta for the motion comparison.
        let mut hq = cx
            .prepare_cached("SELECT ts, bpm FROM hrSample WHERE ts >= ?1 AND ts < ?2")
            .ok()?;
        let mut hr_by: Vec<(f64, f64)> = vec![(0.0, 0.0); n];
        for row in hq.query_map([s, e], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))).ok()? {
            let Ok((ts, bpm)) = row else { continue };
            let k = ((ts - s) / EPOCH) as usize;
            if k < n {
                hr_by[k].0 += bpm as f64;
                hr_by[k].1 += 1.0;
            }
        }
        let mut gq = cx
            .prepare_cached("SELECT ts, x, y, z FROM gravitySample WHERE ts >= ?1 AND ts < ?2 ORDER BY ts")
            .ok()?;
        let grav: Vec<AccelSample> = gq
            .query_map([s, e], |r| {
                Ok(AccelSample { ts: r.get(0)?, x: r.get(1)?, y: r.get(2)?, z: r.get(3)? })
            })
            .ok()?
            .filter_map(Result::ok)
            .collect();
        let mut mo = vec![None; n];
        for w in grav.windows(2) {
            let k = ((w[0].ts - s) / EPOCH) as usize;
            if k < n {
                let d = ((w[0].x - w[1].x).powi(2) + (w[0].y - w[1].y).powi(2)
                    + (w[0].z - w[1].z).powi(2))
                .sqrt();
                mo[k] = Some(mo[k].map_or(d, |m: f64| m.max(d)));
            }
        }

        let (mut hp, mut hn, mut mp, mut mn) = (vec![], vec![], vec![], vec![]);
        for k in 0..n {
            let Some(w) = is_wake[k] else { continue };
            if hr_by[k].1 > 0.0 {
                let v = hr_by[k].0 / hr_by[k].1;
                if w { hp.push(v) } else { hn.push(v) }
            }
            if let Some(v) = mo[k] {
                if w { mp.push(v) } else { mn.push(v) }
            }
        }
        if let Some(a) = auc(&hp, &hn) {
            hr_aucs.push(a);
        }
        if let Some(a) = auc(&mp, &mn) {
            mo_aucs.push(a);
        }
    }
    let name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    (hr_aucs.len() >= 5).then_some((name, hr_aucs, mo_aucs))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Default to the registered cohort, so "run it on the user data" needs no path list.
    let paths: Vec<String> =
        if args.is_empty() { user_cohort().into_iter().map(|(_, p)| p).collect() } else { args };
    if paths.is_empty() {
        eprintln!("no stores: pass paths, or register some with dev-notes/cohort_add.py");
        return;
    }
    println!("Per-night AUC of heart rate for wake, one row per wearer.");
    println!("The question is whether the LEVEL is a property of the person.\n");
    println!("{:<26} {:>7} {:>10} {:>10} {:>12}", "wearer", "nights", "hr AUC", "within sd", "motion AUC");

    let (mut means, mut withins, mut all_nights) = (Vec::new(), Vec::new(), 0usize);
    for p in &paths {
        let Some((name, hr, mo)) = per_store(p) else {
            println!("{:<26} too few scorable nights", p);
            continue;
        };
        println!("{:<26} {:>7} {:>10.3} {:>10.3} {:>12.3}",
                 name, hr.len(), mean(&hr), sd(&hr), mean(&mo));
        means.push(mean(&hr));
        withins.push(sd(&hr));
        all_nights += hr.len();
    }
    if means.len() < 3 {
        println!("\nneed at least 3 wearers to decompose");
        return;
    }
    let between = sd(&means);
    let within = mean(&withins);
    println!("\n{} wearers, {} nights", means.len(), all_nights);
    println!("  BETWEEN-wearer sd of the per-wearer mean : {between:.3}");
    println!("  WITHIN-wearer sd, averaged               : {within:.3}");
    println!("  ratio between/within                     : {:.2}", between / within);
    println!("  range of wearer means                    : {:.3} .. {:.3}",
             means.iter().cloned().fold(f64::MAX, f64::min),
             means.iter().cloned().fold(f64::MIN, f64::max));
    println!();
    if between > within {
        println!("BETWEEN dominates: the cardiac wake signal is more a property of the PERSON than");
        println!("of the night, so one global weight is fitted to an average nobody has.");
    } else {
        println!("WITHIN dominates: night-to-night noise is larger than the person-to-person");
        println!("difference, so the sign flip was likely noise and a per-wearer weight is not");
        println!("supported by this data.");
    }
    println!("\nNo PSG cohort can produce this table: one night per subject leaves within-wearer");
    println!("variance undefined. Labels here are our own hypnogram, identical engine for everyone.");
}
