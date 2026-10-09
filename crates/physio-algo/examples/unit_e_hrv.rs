//! Unit E2: which sleep window should the nightly HRV be measured over, against WHOOP's own HRV.
//!
//!   cargo run --release -p physio-algo --example unit_e_hrv
//!
//! For every night with raw R-R and a WHOOP export HRV: run the shipped detect -> stage -> refine
//! (`sleep::analyze_with(.., V2)`), match the detected session to WHOOP's main sleep, and score the SAME
//! beats several ways with the shipped `HrvReadiness` functions:
//!   whole   `windowed_avg_hrv` over the detected session (what 9.0.1 shipped, and `Session::avg_hrv`)
//!   deep    `nightly_hrv` over deep-stage buckets (what ships now)
//!   lastDeep   the last deep bout only (the strap-style "last slow-wave sleep" comparator)
//!   last90     deep buckets in the 90 minutes before the end of the last deep bout (last sleep cycle)
//!   whoopwin   `windowed_avg_hrv` over WHOOP's own sleep window (isolates window choice from detection)
//!   pooled     one RMSSD over the whole session (`rmssd_gap_aware`; the diagnostic line in the strap log)
//! plus the v103 filler share (`rr::is_rr_fill`) and the same measures with the filler dropped.
//!
//! WHOOP's HRV is a reference with no ground truth; every pair is two straps on one body, possibly on
//! two wrists. Reads `WHOOP_ANALYSIS_DIR`; writes per-night rows to `$WHOOP_ANALYSIS_DIR/out/`.

mod unit_e;

use physio_algo::hrv::{overlapping_report_count, rr_coverage, HrvReadiness, RrReport};
use physio_algo::rr::is_rr_fill;
use physio_algo::sleep::{analyze_with, Engine, SleepStage};
use unit_e::*;

const MIN_HR_COVERAGE: f64 = 0.7;
const MIN_BEATS_PER_HOUR: f64 = 600.0;
const PAD_S: i64 = 6 * 3600;
const LAST_CYCLE_S: u32 = 90 * 60;

const MEASURES: [&str; 8] = ["whole", "deep", "lastDeep", "last90", "whoopwin", "pooled", "whole-nofill", "deep-nofill"];

struct Night {
    wearer: &'static str,
    set: &'static str,
    day: i64,
    whoop: f64,
    v: [Option<f64>; 8],
    filler: f64,
    deep_min: f64,
    beats: usize,
    sess_min: f64,
    whoop_min: f64,
    /// Sum of R-R over the session's wall span. Above ~1.0 is physically impossible (double-counted beats).
    cov: f64,
    overlap: f64,
}

/// A night whose R-R coverage is above this was ingested with re-reported beats.
const DUP_COVERAGE: f64 = 1.3;

struct Wearer {
    name: &'static str,
    refname: &'static str,
    streams: &'static [&'static str],
}

fn spans_of(segs: &[physio_algo::sleep::StageSegment], stage: SleepStage) -> Vec<(u32, u32)> {
    segs.iter().filter(|s| s.stage == stage).map(|s| (s.start as u32, s.end as u32)).collect()
}

fn main() {
    let wearers = [
        Wearer { name: "rep(4.0)", refname: "rep", streams: &["rep-bak", "rep-rc2"] },
        Wearer { name: "dav(5.0/MG)", refname: "dav", streams: &["dav-409", "dav-910", "dav-360"] },
    ];
    let mut nights: Vec<Night> = Vec::new();
    for w in &wearers {
        let sets: Vec<(&'static str, Streams)> = w.streams.iter().map(|n| (*n, load(n))).collect();
        for c in cycles(w.refname) {
            let (Some(hrv), Some(on), Some(wake)) = (c.hrv, c.sleep_on, c.wake_on) else { continue };
            let best = sets
                .iter()
                .map(|(n, s)| {
                    let beats = slice_by(&s.rr, |x| x.0, on, wake).len();
                    (hr_coverage(&s.hr, on, wake), beats, *n, s)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            let hours = (wake - on) as f64 / 3600.0;
            if best.0 < MIN_HR_COVERAGE || (best.1 as f64) < MIN_BEATS_PER_HOUR * hours {
                continue;
            }
            let (set, s) = (best.2, best.3);
            let (a, b) = (on - PAD_S, wake + PAD_S);
            let ss = sleep_streams(s, a, b, c.tz);
            let sessions = analyze_with(&ss, Engine::V2);
            let Some(sess) = pick_session(&sessions, on, wake) else {
                println!("{} night {}: no detected session overlaps WHOOP's sleep, skipped", w.name, on);
                continue;
            };
            let (st, en) = (sess.start as u32, sess.end as u32);
            let beats_all: Vec<(u32, u16)> = slice_by(&s.rr, |x| x.0, a, b).iter().map(|&(t, v)| (t as u32, v)).collect();
            let hr_at = |t: u32| -> Option<u8> {
                let i = s.hr.binary_search_by_key(&(t as i64), |x| x.0).ok()?;
                Some(s.hr[i].1.clamp(0, 255) as u8)
            };
            let is_fill = |&(t, v): &(u32, u16)| is_rr_fill(v, hr_at(t));
            let beats_nofill: Vec<(u32, u16)> = beats_all.iter().copied().filter(|b| !is_fill(b)).collect();
            let in_sess = |&&(t, _): &&(u32, u16)| t >= st && t <= en;
            let n_sess = beats_all.iter().filter(in_sess).count();
            let n_fill = beats_all.iter().filter(in_sess).filter(|b| is_fill(b)).count();

            let deep = spans_of(&sess.segments, SleepStage::Deep);
            let reports = |bs: &[(u32, u16)]| -> Vec<RrReport> {
                let mut out: Vec<RrReport> = Vec::new();
                for &(t, v) in bs {
                    match out.last_mut() {
                        Some(r) if r.unix == t => r.rr.push(v),
                        _ => out.push(RrReport { unix: t, rr: vec![v], optical_signal_poor: None }),
                    }
                }
                out
            };
            let last_bout: Vec<(u32, u32)> = deep.last().copied().into_iter().collect();
            let last90: Vec<(u32, u32)> = match deep.last() {
                Some(&(_, e)) => deep.iter().copied().filter(|&(s0, _)| s0 + LAST_CYCLE_S >= e).collect(),
                None => Vec::new(),
            };
            let pooled = {
                let sess_beats: Vec<(u32, Vec<u16>)> = reports(&beats_all.iter().copied().filter(|&(t, _)| t >= st && t <= en).collect::<Vec<_>>())
                    .into_iter()
                    .map(|r| (r.unix, r.rr))
                    .collect();
                HrvReadiness::rmssd_gap_aware(&sess_beats)
            };
            let v = [
                HrvReadiness::windowed_avg_hrv(st, en, &beats_all),
                HrvReadiness::nightly_hrv(st, en, &reports(&beats_all), &deep),
                HrvReadiness::windowed_avg_hrv_deep(st, en, &beats_all, &last_bout),
                HrvReadiness::windowed_avg_hrv_deep(st, en, &beats_all, &last90),
                HrvReadiness::windowed_avg_hrv(on as u32, wake as u32, &beats_all),
                pooled,
                HrvReadiness::windowed_avg_hrv(st, en, &beats_nofill),
                HrvReadiness::nightly_hrv(st, en, &reports(&beats_nofill), &deep),
            ];
            let (_, _, dmin, _) = stage_minutes(sess);
            let in_s: Vec<&(u32, u16)> = beats_all.iter().filter(in_sess).collect();
            let cov = rr_coverage(&in_s.iter().map(|b| b.0 as i64).collect::<Vec<_>>(), &in_s.iter().map(|b| b.1 as f64).collect::<Vec<_>>());
            let runs: Vec<(u32, Vec<u16>)> = reports(&in_s.iter().map(|b| **b).collect::<Vec<_>>()).into_iter().map(|r| (r.unix, r.rr)).collect();
            let (ov, tot) = overlapping_report_count(&runs);
            nights.push(Night {
                wearer: w.name,
                set,
                day: on,
                whoop: hrv,
                v,
                filler: if n_sess > 0 { n_fill as f64 / n_sess as f64 } else { f64::NAN },
                deep_min: dmin,
                beats: n_sess,
                sess_min: (sess.end - sess.start) as f64 / 60.0,
                whoop_min: (wake - on) as f64 / 60.0,
                cov,
                overlap: if tot > 0 { ov as f64 / tot as f64 } else { f64::NAN },
            });
        }
    }

    // Per-night rows stay outside the repo.
    let mut csv = String::from("wearer,set,onset,whoop_hrv,filler,deep_min,beats,sess_min,whoop_min,rr_cov,overlap,");
    csv += &MEASURES.join(",");
    csv += "\n";
    for n in &nights {
        csv += &format!("{},{},{},{},{:.5},{:.1},{},{:.0},{:.0},{:.3},{:.3}", n.wearer, n.set, n.day, n.whoop, n.filler, n.deep_min, n.beats, n.sess_min, n.whoop_min, n.cov, n.overlap);
        for x in &n.v {
            csv += &format!(",{}", x.map_or(String::new(), |v| format!("{v:.2}")));
        }
        csv += "\n";
    }
    write_out("e2_nights.csv", &csv);

    println!("nights scored: {}", nights.len());
    for w in &wearers {
        let mine: Vec<&Night> = nights.iter().filter(|n| n.wearer == w.name).collect();
        println!(
            "{}: R-R coverage per night (sum R-R / wall span; 1.0 is physical): {}",
            w.name,
            mine.iter().map(|n| format!("{:.2}", n.cov)).collect::<Vec<_>>().join(" ")
        );
        let (dup, clean): (Vec<&Night>, Vec<&Night>) = mine.iter().partition(|n| n.cov > DUP_COVERAGE);
        report(&format!("{} CLEAN ingest (coverage <= {DUP_COVERAGE})", w.name), &clean);
        report(&format!("{} DUPLICATED ingest (coverage > {DUP_COVERAGE})", w.name), &dup);
    }
    let all_clean: Vec<&Night> = nights.iter().filter(|n| n.cov <= DUP_COVERAGE).collect();
    report("POOLED CLEAN nights, both wearers", &all_clean);
}

fn report(label: &str, ns: &[&Night]) {
    println!("\n=============== {label}: {} nights ===============", ns.len());
    if ns.is_empty() {
        return;
    }
    let whoop_all: Vec<f64> = ns.iter().map(|n| n.whoop).collect();
    let fillers: Vec<f64> = ns.iter().map(|n| n.filler).filter(|x| !x.is_nan()).collect();
    println!(
        "WHOOP HRV mean {:.1} median {:.1} sd {:.1} | R-R filler share per night: mean {:.3}% max {:.3}% | deep min per night: mean {:.0}",
        mean(&whoop_all), median(&whoop_all), sd(&whoop_all), 100.0 * mean(&fillers), 100.0 * fillers.iter().cloned().fold(0.0, f64::max),
        mean(&ns.iter().map(|n| n.deep_min).collect::<Vec<_>>())
    );
    println!(
        "  {:<13} {:>3} {:>7} {:>7} {:>8} {:>16} {:>6} {:>6} {:>7} {:>8}  over/under  null MAE (mean/p5)  const-median MAE",
        "measure", "n", "ours", "ratio", "delta", "[95% CI]", "MAE", "r", "sdlogR", "noVal"
    );
    let mut rng = Rng(0xBADC0DE);
    let const_med = median(&whoop_all);
    for (k, name) in MEASURES.iter().enumerate() {
        let pairs: Vec<(f64, f64)> = ns.iter().filter_map(|n| n.v[k].map(|v| (v, n.whoop))).collect();
        let none = ns.len() - pairs.len();
        if pairs.len() < 2 {
            println!("  {name:<13} n={} (too few)", pairs.len());
            continue;
        }
        let (ours, who): (Vec<f64>, Vec<f64>) = pairs.iter().cloned().unzip();
        let ratios: Vec<f64> = ours.iter().zip(&who).map(|(o, w)| o / w).collect();
        let logr: Vec<f64> = ratios.iter().map(|r| r.ln()).collect();
        let d: Vec<f64> = ours.iter().zip(&who).map(|(o, w)| o - w).collect();
        let (lo, hi) = boot_mean_ci(&d);
        let over = d.iter().filter(|x| **x > 0.0).count();
        let mut ms: Vec<f64> = (0..1000)
            .map(|_| {
                let mut t = who.clone();
                shuffle(&mut t, &mut rng);
                mae(&ours, &t)
            })
            .collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cm = mae(&vec![const_med; who.len()], &who);
        println!(
            "  {:<13} {:>3} {:>7} {:>7} {:>8} {:>16} {:>6} {:>6} {:>7} {:>8}  {:>4}/{:<4}  {:>8}/{:<8}  {:>6}",
            name, pairs.len(), fmt(mean(&ours)), fmt(exp_mean(&logr)), fmt(mean(&d)), format!("[{}, {}]", fmt(lo), fmt(hi)),
            fmt(mae(&ours, &who)), fmt(pearson(&ours, &who)), fmt(sd(&logr)), none, over, pairs.len() - over, fmt(mean(&ms)), fmt(ms[50]), fmt(cm)
        );
    }
    // Paired window comparison on the nights both measures exist.
    println!("  paired |error| difference on shared nights (negative = first is closer to WHOOP):");
    for (a, b) in [(1usize, 0usize), (2, 1), (3, 1), (4, 0), (1, 4), (7, 6), (1, 7)] {
        let dd: Vec<f64> = ns
            .iter()
            .filter_map(|n| Some((n.v[a]? - n.whoop).abs() - (n.v[b]? - n.whoop).abs()))
            .collect();
        if dd.len() < 3 {
            continue;
        }
        let (lo, hi) = boot_mean_ci(&dd);
        println!("    {:<13} minus {:<13} n={:>2}  mean {:>6}  [{}, {}]", MEASURES[a], MEASURES[b], dd.len(), fmt(mean(&dd)), fmt(lo), fmt(hi));
    }
    // Filler as a covariate: does the share track the deep-window error?
    let ef: Vec<(f64, f64)> = ns.iter().filter_map(|n| Some((n.filler, (n.v[1]? - n.whoop).abs()))).filter(|p| !p.0.is_nan()).collect();
    if ef.len() >= 4 {
        let (f, e): (Vec<f64>, Vec<f64>) = ef.into_iter().unzip();
        println!("  filler share vs |deep error|: n={} r={} (filler spans {:.3}%..{:.3}%)", f.len(), fmt(pearson(&f, &e)), 100.0 * f.iter().cloned().fold(f64::MAX, f64::min), 100.0 * f.iter().cloned().fold(0.0, f64::max));
    }
    let dnf: Vec<f64> = ns.iter().filter_map(|n| Some(n.v[1]? - n.v[7]?)).collect();
    if !dnf.is_empty() {
        println!("  dropping the filler moves deep HRV by mean {} ms (max abs {})", fmt(mean(&dnf)), fmt(dnf.iter().fold(0.0f64, |a, b| a.max(b.abs()))));
    }
}

fn exp_mean(logs: &[f64]) -> f64 {
    mean(logs).exp()
}
