//! Unit E3: 4.0 nights through both sleep engines, against WHOOP's own row for the same night.
//!
//!   cargo run --release -p physio-algo --example unit_e_night
//!
//! The reporter's rc2 debug capture (`records.jsonl`) is decoded here from its raw `frame` hex with
//! `whoop_protocol` (4.0 path) and checked field by field against what the app stored, then staged by
//! `sleep::analyze_with` under `Engine::V2` ("Original") and `Engine::Tanv1` ("Experimental"). Any other
//! 4.0 night that has raw R-R and a WHOOP export row goes through the same path.
//!
//! n is one to a handful. This is an anecdote, not a tuning target. WHOOP's stage minutes are a reference
//! with no ground truth (they are a residual: awake = in bed - asleep - no data), and each pair is a
//! different strap from the one that produced the raw. Reads `WHOOP_ANALYSIS_DIR`.

mod unit_e;

use physio_algo::hrv::rr_coverage;
use physio_algo::sleep::{analyze_with, Engine};
use serde_json::Value;
use unit_e::*;
use whoop_protocol::bytes::from_hex;
use whoop_protocol::family::Family;
use whoop_protocol::framing;
use whoop_protocol::records::{decode, Record};

const MIN_HR_COVERAGE: f64 = 0.7;
const PAD_S: i64 = 6 * 3600;

/// Decode `records.jsonl` from its wire bytes. Returns the stream set and a check tally:
/// (records, decoded, hr matches, rr matches, gravity matches, field mismatches).
fn decode_rc2() -> (Streams, [usize; 6]) {
    let text = read_lossy("raw/rc2/records.jsonl");
    let mut s = Streams::default();
    let mut tally = [0usize; 6];
    let mut seen_hr = std::collections::HashSet::new();
    for line in text.lines() {
        let Ok(j) = serde_json::from_str::<Value>(line) else { continue };
        tally[0] += 1;
        let Some(wire) = j["frame"].as_str().and_then(from_hex) else { continue };
        let Ok(fr) = framing::decode(Family::Gen4, &wire) else { continue };
        let Some(Record::History(h)) = decode(&fr) else { continue };
        tally[1] += 1;
        let t = h.unix as i64;
        if let Some(hr) = h.heart_rate {
            if j["heart_rate"].as_u64() == Some(hr as u64) { tally[2] += 1 } else { tally[5] += 1 }
            if seen_hr.insert(t) {
                s.hr.push((t, hr as i32));
            }
        }
        let want: Vec<u64> = j["rr_intervals"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
        if want == h.rr_intervals.iter().map(|&x| x as u64).collect::<Vec<_>>() { tally[3] += 1 } else { tally[5] += 1 }
        for &r in &h.rr_intervals {
            s.rr.push((t, r));
        }
        if let Some(g) = h.gravity {
            let near = |k: &str, v: f32| (j[k].as_f64().unwrap_or(f64::NAN) - v as f64).abs() < 1e-6;
            if near("gravity_x", g[0]) && near("gravity_y", g[1]) && near("gravity_z", g[2]) { tally[4] += 1 } else { tally[5] += 1 }
            s.grav.push((t, g[0] as f64, g[1] as f64, g[2] as f64));
        }
    }
    s.hr.sort_by_key(|x| x.0);
    s.rr.sort_by_key(|x| x.0);
    s.grav.sort_by_key(|x| x.0);
    s.grav.dedup_by_key(|x| x.0);
    (s, tally)
}

/// A named set of streams (one strap capture).
type NamedSets = Vec<(&'static str, Streams)>;

struct Run {
    span: f64,
    asleep: f64,
    wake: f64,
    light: f64,
    deep: f64,
    rem: f64,
    unscored: f64,
    on_d: f64,
    off_d: f64,
    /// Wake minutes that fall inside WHOOP's own sleep window (the rest is before onset / after wake).
    wake_in: f64,
}

fn run(ss: &physio_algo::sleep::SleepStreams, engine: Engine, on: i64, wake: i64) -> Option<Run> {
    let sessions = analyze_with(ss, engine);
    let s = pick_session(&sessions, on, wake)?;
    let (w, l, d, r) = stage_minutes(s);
    Some(Run {
        span: (s.end - s.start) as f64 / 60.0,
        asleep: l + d + r,
        wake: w,
        light: l,
        deep: d,
        rem: r,
        unscored: unscored_minutes(s),
        on_d: (s.start - on) as f64 / 60.0,
        off_d: (s.end - wake) as f64 / 60.0,
        wake_in: s
            .segments
            .iter()
            .filter(|g| g.stage == physio_algo::sleep::SleepStage::Wake)
            .map(|g| (g.end.min(wake) - g.start.max(on)).max(0) as f64 / 60.0)
            .sum(),
    })
}

fn main() {
    let (rc2, t) = decode_rc2();
    println!(
        "rc2 decode check (whoop_protocol, Gen4): {} records, {} decoded as history; HR matches {} , R-R lists match {}, gravity matches {}, field mismatches {}",
        t[0], t[1], t[2], t[3], t[4], t[5]
    );
    println!("  decoded streams: {} HR, {} beats, {} gravity", rc2.hr.len(), rc2.rr.len(), rc2.grav.len());

    let targets: Vec<(&str, &str, NamedSets)> = vec![
        ("rep(4.0)", "rep", vec![("rep-bak", load("rep-bak")), ("rep-rc2(decoded)", rc2)]),
        ("dav(4.0 FB:46)", "dav", vec![("dav-fb46", load("dav-fb46"))]),
    ];
    let mut all: Vec<(String, f64, Run, Run, [f64; 5])> = Vec::new();
    for (label, refname, sets) in &targets {
        for c in cycles(refname) {
            let (Some(on), Some(wake)) = (c.sleep_on, c.wake_on) else { continue };
            let best = sets
                .iter()
                .map(|(n, s)| (hr_coverage(&s.hr, on, wake), *n, s))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            if best.0 < MIN_HR_COVERAGE {
                continue;
            }
            let s = best.2;
            let ss = sleep_streams(s, on - PAD_S, wake + PAD_S, c.tz);
            let (Some(orig), Some(exp)) = (run(&ss, Engine::V2, on, wake), run(&ss, Engine::Tanv1, on, wake)) else {
                println!("{label} night {on}: no session overlaps WHOOP's sleep, skipped");
                continue;
            };
            let cov = {
                let b = slice_by(&s.rr, |x| x.0, on, wake);
                rr_coverage(&b.iter().map(|x| x.0).collect::<Vec<_>>(), &b.iter().map(|x| x.1 as f64).collect::<Vec<_>>())
            };
            let (wi, wa, ww, wl, wd, wr) = (c.inbed.unwrap_or(f64::NAN), c.asleep.unwrap_or(f64::NAN), c.awake.unwrap_or(f64::NAN), c.light.unwrap_or(f64::NAN), c.deep.unwrap_or(f64::NAN), c.rem.unwrap_or(f64::NAN));
            println!("\n--- {label} / {} ---  R-R coverage {cov:.2} {}", best.1, if cov > 1.3 { "(DUPLICATED ingest)" } else { "(clean)" });
            println!("  {:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9}", "", "in-bed", "asleep", "awake", "light", "deep", "rem", "onset d", "offset d");
            println!("  {:<22} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0}", "WHOOP export", wi, wa, ww, wl, wd, wr);
            for (n, r) in [("Original (V2)", &orig), ("Experimental (Tanv1)", &exp)] {
                println!("  {:<22} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>+8.0}m {:>+8.0}m   unscored {:.0} min, wake inside WHOOP window {:.0}", n, r.span, r.asleep, r.wake, r.light, r.deep, r.rem, r.on_d, r.off_d, r.unscored, r.wake_in);
            }
            if cov <= 1.3 || *label != "rep(4.0)" {
                all.push((format!("{label}/{}", best.1), cov, orig, exp, [wa, ww, wl, wd, wr]));
            }
            let _ = (wi, ww);
            let _ = (wl, wd, wr);
        }
    }

    // Aggregate over the nights with a clean R-R ingest: ours minus WHOOP, minutes per stage.
    println!("
=== aggregate over {} night(s) (rep nights with duplicated R-R excluded); ours minus WHOOP, mean minutes ===", all.len());
    println!("  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}", "", "asleep", "awake", "light", "deep", "rem", "unscored");
    for (name, pick) in [("Original (V2)", 0usize), ("Experimental (Tanv1)", 1)] {
        let col = |f: &dyn Fn(&Run, &[f64; 5]) -> f64| mean(&all.iter().map(|x| f(if pick == 0 { &x.2 } else { &x.3 }, &x.4)).collect::<Vec<_>>());
        println!(
            "  {:<22} {:>+8.0} {:>+8.0} {:>+8.0} {:>+8.0} {:>+8.0} {:>10.0}",
            name,
            col(&|r, w| r.asleep - w[0]),
            col(&|r, w| r.wake - w[1]),
            col(&|r, w| r.light - w[2]),
            col(&|r, w| r.deep - w[3]),
            col(&|r, w| r.rem - w[4]),
            col(&|r, _| r.unscored)
        );
    }
}
