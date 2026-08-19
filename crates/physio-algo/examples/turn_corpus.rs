//! Does `turn` hold up beyond the three labelled nights, on a whole store?
//!
//!   cargo run --release -p physio-algo --example turn_corpus -- <store.sqlite>
//!
//! `turn` beat every other motion feature on 143 PSG nights. Two things that does not establish:
//! whether it holds on OUR sensor rather than a research accelerometer, and whether it holds across
//! one wearer's months rather than one night each from many.
//!
//! Uses the shipped [`turn`] and [`posture_series`], not a reimplementation - a harness that
//! recomputes the formula it is testing can agree with itself while both are wrong.
//!
//! WHAT THE REFERENCE IS. The corpus has no truth: its labels are OUR OWN hypnogram. That makes this
//! a consistency measure, with one thing going for it - `turn` is an input to nothing in the shipped
//! engine, so any agreement was not put there by construction. The labelled nights below it DO carry
//! wearer truth and are the part that can falsify anything.

use physio_algo::sleep::AccelSample;
use physio_algo::sleep::posture::posture_series;
use physio_algo::sleep::posture::turn as turn_of;

const EPOCH: i64 = 30;
/// A night needs this many epochs of each class before its AUC means anything.
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

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((q * s.len() as f64) as usize).min(s.len() - 1)]
}

/// Peak within-epoch delta per axis plus the norm, so all four read off identical epochs.
/// Index 0..2 are x/y/z, index 3 is the shipped norm.
fn jerk_axes(grav: &[AccelSample], w0: i64, n: usize) -> [Vec<Option<f64>>; 4] {
    let mut out = [vec![None; n], vec![None; n], vec![None; n], vec![None; n]];
    let mut i = 0usize;
    for k in 0..n {
        let (a, b) = (w0 + k as i64 * EPOCH, w0 + (k as i64 + 1) * EPOCH);
        while i < grav.len() && grav[i].ts < a {
            i += 1;
        }
        let j = i + grav[i..].iter().take_while(|s| s.ts < b).count();
        let seg = &grav[i..j];
        let mut peak = [0.0f64; 4];
        let mut seen = false;
        for (p, q) in seg.iter().zip(seg.iter().skip(1)) {
            let d = [(p.x - q.x).abs(), (p.y - q.y).abs(), (p.z - q.z).abs()];
            for t in 0..3 {
                peak[t] = peak[t].max(d[t]);
            }
            peak[3] = peak[3].max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            seen = true;
        }
        if seen {
            for t in 0..4 {
                out[t][k] = Some(peak[t]);
            }
        }
    }
    out
}

fn jerk_series(grav: &[AccelSample], w0: i64, n: usize) -> Vec<Option<f64>> {
    let [_, _, _, norm] = jerk_axes(grav, w0, n);
    norm
}

const LABELLED: &str = "C:/Users/DavidGillot/Projects/whoop/whoop-data/harnesses/labelled-nights";

/// The nights with WEARER truth. Small, and the only thing here that can falsify a claim.
fn labelled_nights() {
    println!("\n=== the labelled nights - WEARER TRUTH, the part that can falsify");
    let Ok(dirs) = std::fs::read_dir(LABELLED) else {
        println!("  fixture root unreadable: {LABELLED}");
        return;
    };
    let mut names: Vec<_> = dirs
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("truth.csv").exists() && p.join("gravity.csv").exists())
        .collect();
    names.sort();
    println!("  {:<18} {:>7} {:>8} {:>8}   {}", "night", "epochs", "turn", "jerk", "verdict");
    for dir in names {
        let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
        let Ok(gtext) = std::fs::read_to_string(dir.join("gravity.csv")) else { continue };
        let grav: Vec<AccelSample> = gtext
            .lines()
            .filter_map(|l| {
                let p: Vec<&str> = l.split(',').collect();
                (p.len() >= 4).then(|| {
                    Some(AccelSample {
                        ts: p[0].parse().ok()?,
                        x: p[1].parse().ok()?,
                        y: p[2].parse().ok()?,
                        z: p[3].parse().ok()?,
                    })
                })?
            })
            .collect();
        // truth.csv is startTs,endTs,isWake,note - the note contains commas, so take fields 0..3.
        let Ok(ttext) = std::fs::read_to_string(dir.join("truth.csv")) else { continue };
        let spans: Vec<(i64, i64, bool)> = ttext
            .lines()
            .filter_map(|l| {
                let p: Vec<&str> = l.split(',').collect();
                if p.len() < 3 {
                    return None;
                }
                Some((p[0].parse().ok()?, p[1].parse().ok()?, p[2].trim().parse::<i32>().ok()? == 1))
            })
            .collect();
        if grav.is_empty() || spans.is_empty() {
            println!("  {name:<18} no usable gravity or truth");
            continue;
        }
        let (w0, w1) = (grav[0].ts, grav[grav.len() - 1].ts + 1);
        let n = ((w1 - w0) / EPOCH).max(0) as usize;
        let post = posture_series(&grav, w0, w0 + n as i64 * EPOCH, EPOCH);
        let jerks = jerk_series(&grav, w0, n);
        let (mut tp, mut tn, mut jp, mut jn) = (vec![], vec![], vec![], vec![]);
        for k in 0..n {
            let mid = w0 + k as i64 * EPOCH + EPOCH / 2;
            let Some(w) = spans.iter().find(|(a, b, _)| mid >= *a && mid < *b).map(|s| s.2) else {
                continue;
            };
            if let (Some(prev), Some(cur)) = (k.checked_sub(1).and_then(|i| post[i]), post[k]) {
                if let Some(t) = turn_of(&prev, &cur) {
                    if w { tp.push(t) } else { tn.push(t) }
                }
            }
            if let Some(j) = jerks[k] {
                if w { jp.push(j) } else { jn.push(j) }
            }
        }
        match (auc(&tp, &tn), auc(&jp, &jn)) {
            (Some(t), Some(j)) => println!("  {name:<18} {:>7} {t:>8.3} {j:>8.3}   {}",
                                           tp.len() + tn.len(),
                                           if t > j { "turn wins" } else { "JERK WINS" }),
            _ => println!("  {name:<18} {:>7} too few epochs of one class", tp.len() + tn.len()),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: turn_corpus <store.sqlite>")?;
    let cx = rusqlite::Connection::open_with_flags(
        &path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;

    let mut q = cx.prepare(
        "SELECT startTs, endTs, stagesJSON FROM sleepSession \
         WHERE stagesJSON IS NOT NULL AND stagesJSON != '' ORDER BY startTs",
    )?;
    let sessions: Vec<(i64, i64, String)> = q
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(Result::ok)
        .collect();
    println!("{} staged sessions in {}\n", sessions.len(), path);

    let (mut turn_aucs, mut jerk_aucs) = (Vec::new(), Vec::new());
    let (mut wins, mut scored, mut skipped) = (0usize, 0usize, 0usize);
    // Per-axis, to ask on OUR sensor what the PSG cohorts said: is the device frame stable enough
    // for one axis to carry its own weight, or is the winner random?
    let mut ax: [Vec<f64>; 3] = Default::default();
    let mut ax_win = [0usize; 3];

    for (s, e, stages_json) in &sessions {
        let n = ((e - s) / EPOCH).max(0) as usize;
        if n < 60 {
            skipped += 1;
            continue;
        }
        // Our own hypnogram, per epoch. Not truth.
        let mut is_wake = vec![None; n];
        let parsed: serde_json::Value = match serde_json::from_str(stages_json) {
            Ok(v) => v,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        for seg in parsed.as_array().into_iter().flatten() {
            let (a, b, st) = (seg["start"].as_i64(), seg["end"].as_i64(), seg["stage"].as_str());
            let (Some(a), Some(b), Some(st)) = (a, b, st) else { continue };
            let mut t = a;
            while t < b {
                let k = ((t - s) / EPOCH) as usize;
                if t >= *s && k < n {
                    is_wake[k] = Some(st == "wake");
                }
                t += EPOCH;
            }
        }

        let mut g = cx.prepare_cached(
            "SELECT ts, x, y, z FROM gravitySample WHERE ts >= ?1 AND ts < ?2 ORDER BY ts",
        )?;
        let grav: Vec<AccelSample> = g
            .query_map([s, e], |r| {
                Ok(AccelSample { ts: r.get(0)?, x: r.get(1)?, y: r.get(2)?, z: r.get(3)? })
            })?
            .filter_map(Result::ok)
            .collect();
        if grav.len() < n * 10 {
            skipped += 1;
            continue;
        }

        let post = posture_series(&grav, *s, s + n as i64 * EPOCH, EPOCH);
        let axes = jerk_axes(&grav, *s, n);
        let jerks = axes[3].clone();
        {
            let mut got = [f64::NAN; 3];
            let mut ok = true;
            for t in 0..3 {
                let (mut p, mut q) = (Vec::new(), Vec::new());
                for k in 0..n {
                    let (Some(w), Some(v)) = (is_wake[k], axes[t][k]) else { continue };
                    if w { p.push(v) } else { q.push(v) }
                }
                match auc(&p, &q) {
                    Some(a) => got[t] = a,
                    None => ok = false,
                }
            }
            if ok {
                for t in 0..3 {
                    ax[t].push(got[t]);
                }
                let mut best = 0usize;
                for t in 1..3 {
                    if got[t] > got[best] {
                        best = t;
                    }
                }
                ax_win[best] += 1;
            }
        }
        let (mut tp, mut tn, mut jp, mut jn) = (vec![], vec![], vec![], vec![]);
        for k in 0..n {
            let Some(w) = is_wake[k] else { continue };
            if let (Some(prev), Some(cur)) = (k.checked_sub(1).and_then(|i| post[i]), post[k]) {
                if let Some(t) = turn_of(&prev, &cur) {
                    if w { tp.push(t) } else { tn.push(t) }
                }
            }
            if let Some(j) = jerks[k] {
                if w { jp.push(j) } else { jn.push(j) }
            }
        }
        match (auc(&tp, &tn), auc(&jp, &jn)) {
            (Some(t), Some(j)) => {
                turn_aucs.push(t);
                jerk_aucs.push(j);
                if t > j {
                    wins += 1;
                }
                scored += 1;
            }
            _ => skipped += 1,
        }
    }

    println!("scored {scored} nights, skipped {skipped} (too short, unparseable, or one-class)\n");
    if scored == 0 {
        println!("nothing to report");
        return Ok(());
    }
    println!("{:<10} {:>8} {:>8} {:>8} {:>8} {:>8}", "feature", "p10", "p25", "median", "p75", "p90");
    for (name, v) in [("turn", &turn_aucs), ("jerk", &jerk_aucs)] {
        println!("{:<10} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}", name,
                 pct(v, 0.10), pct(v, 0.25), pct(v, 0.50), pct(v, 0.75), pct(v, 0.90));
    }
    let deltas: Vec<f64> = turn_aucs.iter().zip(&jerk_aucs).map(|(t, j)| t - j).collect();
    let mean_d = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let sd = (deltas.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>()
        / (deltas.len() as f64 - 1.0).max(1.0))
        .sqrt();
    println!("\nturn beats jerk on {wins} of {scored} nights ({:.0}%)",
             100.0 * wins as f64 / scored as f64);
    println!("paired delta: mean {mean_d:+.4}, sd {sd:.4}, resolvable +/-{:.4}",
             1.96 * sd / (deltas.len() as f64).sqrt());
    let axn: usize = ax_win.iter().sum();
    if axn > 0 {
        println!("
per-axis on this store ({axn} nights) - is the device frame stable?");
        for (t, name) in ["x", "y", "z"].iter().enumerate() {
            println!("  axis {name}   AUC {:.3}   best on {:>3} of {axn} nights ({:.0}%)",
                     pct(&ax[t], 0.50), ax_win[t], 100.0 * ax_win[t] as f64 / axn as f64);
        }
        println!("  norm     AUC {:.3}   <- what we ship", pct(&jerk_aucs, 0.50));
    }
    labelled_nights();
    println!("\nThe corpus labels above are OUR hypnogram, not truth. turn feeds nothing in the shipped");
    println!("engine, so agreement was not built in - but only the labelled nights can falsify.");
    Ok(())
}
