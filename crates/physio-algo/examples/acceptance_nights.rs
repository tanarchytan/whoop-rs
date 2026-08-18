//! The candidate against the only WHOOP nights where the wearer told us the truth.
//!
//!   cargo run --release -p physio-algo --example acceptance_nights
//!
//! Two nights, real WHOOP hardware, per-epoch wake/sleep from the wearer. The project rule is that a
//! sleep change is checked here before any cohort kappa is quoted, and until the builder was extended
//! to emit the gravity VECTOR these nights could not be staged at all - `motion.csv` is a scalar and
//! `SleepInput` needs x/y/z, so the acceptance set was unusable by the very rule that names it.
//!
//! Truth is SELF-REPORTED. It is the wrong instrument and the right sensor, which is the opposite
//! trade from every PSG cohort, and that is exactly why both are needed.

use std::fs;
use std::path::Path;

use physio_algo::sleep::metrics::{bout_score, confusion4, recall, specificity, WAKE};
use physio_algo::sleep::{
    params::Params, prepare_v2, refine_wake, stage_v2_prepared, AccelSample, HrSample, RrRun,
    SleepInput, SleepStage, StepSample,
};

const ROOT: &str = "C:/Users/DavidGillot/Projects/whoop/whoop-data/harnesses/labelled-nights";
const EPOCH: i64 = 30;
const MIN_BOUT: usize = 10;

fn rows(p: &Path) -> Vec<Vec<f64>> {
    fs::read_to_string(p)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| l.split(',').map(|c| c.trim().parse::<f64>().ok()).collect())
                .collect()
        })
        .unwrap_or_default()
}

fn main() {
    let cand = Params { clamp_only_without_rr: true, ..Params::SHIPPED };
    println!("{:<18} {:<26} {:>9} {:>10} {:>11} {:>11}",
        "night", "recipe", "wake rec", "wake spec", "bout COV", "we call");

    for name in ["david-20260816", "reader-20260815"] {
        let d = Path::new(ROOT).join(name);
        let meta = rows(&d.join("meta.txt"));
        let m: Vec<f64> = fs::read_to_string(d.join("meta.txt"))
            .unwrap_or_default()
            .split_whitespace()
            .filter_map(|x| x.parse().ok())
            .collect();
        let _ = meta;
        if m.len() < 4 {
            println!("{name}: no meta.txt");
            continue;
        }
        let (w0, w1, n) = (m[1] as i64, m[2] as i64, m[3] as usize);

        let hr: Vec<HrSample> =
            rows(&d.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect();
        let rr: Vec<RrRun> = rows(&d.join("rr.csv"))
            .iter()
            .map(|r| RrRun { ts: r[0] as i64, intervals: vec![r[1] as u16] })
            .collect();
        let accel: Vec<AccelSample> = rows(&d.join("gravity.csv"))
            .iter()
            .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
            .collect();
        let steps: Vec<StepSample> = rows(&d.join("steps.csv"))
            .iter()
            .map(|r| StepSample {
                ts: r[0] as i64,
                counter: r[1] as u16,
                activity_class: (r[2] >= 0.0).then(|| r[2] as u8),
            })
            .collect();

        // Wearer truth as per-epoch wake/sleep; epochs outside every labelled span are UNKNOWN.
        let mut truth = vec![usize::MAX; n];
        // truth.csv carries a header and a free-text note, so `rows` (which needs every column to
        // parse) drops every line. Read the first three fields only.
        let truth_spans: Vec<(i64, i64, usize)> = fs::read_to_string(d.join("truth.csv"))
            .unwrap_or_default()
            .lines()
            .skip(1)
            .filter_map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                if f.len() < 3 {
                    return None;
                }
                Some((f[0].trim().parse().ok()?, f[1].trim().parse().ok()?, f[2].trim().parse().ok()?))
            })
            .collect();
        assert!(!truth_spans.is_empty(), "{name}: truth.csv parsed to nothing");
        for (a, b, is_wake) in truth_spans {
            for (k, slot) in truth.iter_mut().enumerate() {
                let mid = w0 + k as i64 * EPOCH + EPOCH / 2;
                if mid >= a && mid < b {
                    *slot = usize::from(is_wake != 1);
                }
            }
        }
        let labelled = truth.iter().filter(|t| **t != usize::MAX).count();

        let input = SleepInput { start: w0, end: w1, hr, rr, accel: accel.clone() };
        for (label, p) in [("shipped", Params::SHIPPED), ("cand: clamp_only_without_rr", cand)] {
            let prep = prepare_v2(&input, &p);
            let segs = refine_wake(&stage_v2_prepared(&prep, &p), &accel, &steps);
            let pred: Vec<usize> = (0..n)
                .map(|k| {
                    let mid = w0 + k as i64 * EPOCH + EPOCH / 2;
                    let st = segs.iter().find(|s| s.start <= mid && mid < s.end).map(|s| s.stage);
                    // Two classes only: the wearer reported wake or sleep, not stages.
                    usize::from(st.unwrap_or(SleepStage::Light) != SleepStage::Wake)
                })
                .collect();
            // Score only where the wearer spoke.
            let (p2, t2): (Vec<usize>, Vec<usize>) = truth
                .iter()
                .zip(&pred)
                .filter(|(t, _)| **t != usize::MAX)
                .map(|(t, p)| (*p, *t))
                .unzip();
            let cm = confusion4(&p2, &t2);
            let called = 100.0 * p2.iter().filter(|p| **p == WAKE).count() as f64 / p2.len() as f64;
            let b = bout_score(&p2, &t2, WAKE, MIN_BOUT, 0.5);
            println!("{:<18} {label:<26} {:>9} {:>10} {:>11} {:>10.1}%",
                format!("{name} ({labelled})"),
                recall(&cm, WAKE).map(|v| format!("{v:.3}")).unwrap_or("-".into()),
                specificity(&cm, WAKE).map(|v| format!("{v:.3}")).unwrap_or("-".into()),
                b.coverage().map(|v| format!("{v:.3}")).unwrap_or("-".into()),
                called);
        }
    }
    println!("\nwake rec = share of the wearer's reported wake we find; spec = share of their reported");
    println!("sleep we leave alone. Both matter: a change that only raises the first is adding wake.");
}
