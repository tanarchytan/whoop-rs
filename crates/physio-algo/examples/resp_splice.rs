//! Does the R-R splice actually move the respiratory rate on real nights?
//!
//!   python dev-notes/export_resp_beats.py <out.csv> <store.sqlite> ...
//!   cargo run --release -p physio-algo --example resp_splice -- <out.csv>
//!
//! `resp_rate_from_rr` used to discard the beat timestamps and rebuild the timeline by cumulative sum,
//! so a dropout was stitched shut and the join read as one enormous breath. The fix splits on gaps.
//!
//! A synthetic two-halves-and-a-gap test did NOT catch it, and that is the reason this harness exists:
//! the night's rate is a median over ~90 five-minute windows, and a median absorbs one corrupted window
//! without moving. The size of this defect is an empirical question about how many gaps a real night
//! carries, which no unit test can settle.
//!
//! Runs the SHIPPED function twice per night: once on the stored timeline, once on a re-stamped copy
//! that reproduces the pre-fix cumulative-sum clock.

use std::collections::BTreeMap;
use std::fs;

use physio_algo::respiratory_rate::resp_rate_from_rr;

/// One night as stored: its span and its beats.
type Night = (i64, i64, Vec<(i64, u16)>);

const GAP_S: i64 = 10;

/// The pre-fix timeline: the clock rebuilt from the intervals alone, so no gap ever exceeds the split
/// threshold and the whole night scores as one run.
fn spliced(beats: &[(i64, u16)]) -> Vec<(i64, u16)> {
    let mut out = Vec::with_capacity(beats.len());
    let (mut t, mut acc_ms) = (beats.first().map(|b| b.0).unwrap_or(0), 0i64);
    for (i, (_, ms)) in beats.iter().enumerate() {
        if i > 0 {
            acc_ms += *ms as i64;
            while acc_ms >= 1000 {
                acc_ms -= 1000;
                t += 1;
            }
        }
        out.push((t, *ms));
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: resp_splice <beats.csv>");
        return;
    };
    let text = fs::read_to_string(&path).expect("beats csv");

    // store -> night -> (start, end, beats)
    let mut by: BTreeMap<String, BTreeMap<i64, Night>> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            continue;
        }
        let (Ok(night), Ok(s), Ok(e), Ok(ts), Ok(ms)) = (
            f[1].parse::<i64>(),
            f[2].parse::<i64>(),
            f[3].parse::<i64>(),
            f[4].parse::<i64>(),
            f[5].parse::<u16>(),
        ) else {
            continue;
        };
        let entry = by.entry(f[0].to_string()).or_default().entry(night).or_insert((s, e, Vec::new()));
        entry.2.push((ts, ms));
    }

    println!("{:<22} {:>7} {:>9} {:>9} {:>9}  {:>7} {:>8}",
        "store", "nights", "moved", "med |d|", "max |d|", "gaps/n", "lost");
    let (mut all_d, mut all_n, mut all_moved) = (Vec::new(), 0usize, 0usize);
    for (store, nights) in &by {
        let (mut ds, mut moved, mut lost, mut gaps_tot, mut n) = (Vec::new(), 0usize, 0usize, 0usize, 0usize);
        for (s, e, beats) in nights.values() {
            if beats.len() < 30 {
                continue;
            }
            n += 1;
            gaps_tot += beats.windows(2).filter(|w| w[1].0 - w[0].0 > GAP_S).count();
            let after = resp_rate_from_rr(beats, *s, *e);
            let sp = spliced(beats);
            let before = resp_rate_from_rr(&sp, sp[0].0, sp[sp.len() - 1].0);
            match (before, after) {
                (Some(x), Some(y)) => {
                    let d = (y - x).abs();
                    ds.push(d);
                    if d > 0.05 {
                        moved += 1;
                    }
                }
                // The fix can only ever REMOVE windows, so this is a night the split left unscorable.
                (Some(_), None) => lost += 1,
                _ => {}
            }
        }
        if n == 0 {
            continue;
        }
        let (m, mx) = (median(&mut ds.clone()), ds.iter().cloned().fold(0.0f64, f64::max));
        println!("{store:<22} {n:>7} {:>9} {m:>9.3} {mx:>9.3}  {:>7.1} {lost:>8}",
            format!("{moved}/{n}"), gaps_tot as f64 / n as f64);
        all_d.extend(ds);
        all_n += n;
        all_moved += moved;
    }
    println!("\nacross all stores: {all_moved} of {all_n} nights moved by more than 0.05 bpm, median |delta| {:.3}",
        median(&mut all_d.clone()));
    println!("'lost' = nights the split leaves unscorable; the fix can only remove windows, never add.");
}
