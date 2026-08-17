//! The candidate on OUR hardware, not on someone else's PSG cohort.
//!
//!   cargo run --release -p physio-algo --example clamp_on_our_data
//!
//! DREAMT is an E4, sleep-accel an Apple Watch, AAUWSS a lab rig. None is a WHOOP strap, and every
//! number for `clamp_only_without_rr` so far came from one of them, staged WITHOUT `refine_wake`
//! because no PSG cohort ships the step stream its density gate needs.
//!
//! Three checks that only our own data can answer:
//!   1. does the gain survive `refine_wake`, which only ever shrinks wake and DOES run for users
//!   2. does wake stay plausible per wearer, or does one strap blow up
//!   3. the band's own `sleep_state` - independent of us, a SLEEP-PERIOD marker so recall only

mod common;

use common::{
    band_asleep_secs, dirs_of, mean, night_id, read_accel, read_band, read_hr, read_meta, read_rr,
    read_steps, stage_idx, RefineCensus,
};

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput, StageSegment};

fn wake_frac(segs: &[StageSegment], start: i64, end: i64) -> f64 {
    let span = (end - start).max(1) as f64;
    segs.iter().filter(|s| stage_idx(s.stage) == 0).map(|s| (s.end - s.start) as f64).sum::<f64>() / span
}

/// Share of the band's own asleep seconds we also call asleep. The band marks the sleep PERIOD, so it
/// cannot see mid-night wake and this is a recall-only read: it must not COLLAPSE, and a small drop is
/// the expected price of finding real wake.
fn band_agreement(segs: &[StageSegment], band: &[(i64, i32)], start: i64, end: i64) -> Option<f64> {
    let band_asleep = band_asleep_secs(band, start, end);
    if band_asleep <= 0 {
        return None;
    }
    let mut agree = 0i64;
    for s in segs.iter().filter(|s| stage_idx(s.stage) != 0) {
        agree += band_asleep_secs(band, s.start, s.end);
    }
    Some(agree as f64 / band_asleep as f64)
}

fn main() {
    let cand = Params { clamp_only_without_rr: true, ..Params::SHIPPED };
    let mut per_owner: std::collections::BTreeMap<String, Vec<[f64; 4]>> = Default::default();
    let (mut band_a, mut band_b) = (Vec::new(), Vec::new());
    let mut census = RefineCensus::default();
    let mut with_steps = 0usize;

    for set_name in ["ours", "whoop4", "strap", "killa5"] {
    for dir in dirs_of(set_name) {
        let Some((w0, w1, _)) = read_meta(&dir) else { continue };
        let steps = read_steps(&dir);
        let accel = read_accel(&dir);
        if accel.is_empty() {
            continue;
        }
        if !steps.is_empty() {
            with_steps += 1;
        }
        let band = read_band(&dir);
        let input =
            SleepInput { start: w0, end: w1, hr: read_hr(&dir), rr: read_rr(&dir), accel: accel.clone() };

        let mut row = [0.0f64; 4];
        for (i, p) in [Params::SHIPPED, cand].iter().enumerate() {
            let prep = prepare_v2(&input, p);
            let raw = stage_v2_prepared(&prep, p);
            let fine = census.refine(&raw, &accel, &steps);
            row[i * 2] = wake_frac(&raw, w0, w1);
            row[i * 2 + 1] = wake_frac(&fine, w0, w1);
            if let Some(b) = band_agreement(&fine, &band, w0, w1) {
                if i == 0 {
                    band_a.push(b)
                } else {
                    band_b.push(b)
                }
            }
        }
        let owner = night_id(&dir).0.split('_').next().unwrap_or("?").to_string();
        let _ = owner;
        per_owner.entry(set_name.to_string()).or_default().push(row);
    }
    }

    let all: Vec<[f64; 4]> = per_owner.values().flatten().copied().collect();
    if all.is_empty() {
        println!("no `ours` nights readable");
        return;
    }
    let col = |v: &[[f64; 4]], k: usize| v.iter().map(|r| r[k]).collect::<Vec<f64>>();

    println!("{} strap nights, {with_steps} carrying a step stream\n", all.len());
    println!("{:<16} {:>7} {:>11} {:>11} {:>11} {:>11}",
        "fixture set", "nights", "ship raw", "ship +ref", "cand raw", "cand +ref");
    for (owner, v) in &per_owner {
        println!("{:<16} {:>7} {:>10.2}% {:>10.2}% {:>10.2}% {:>10.2}%", owner, v.len(),
            100.0 * mean(&col(v, 0)), 100.0 * mean(&col(v, 1)),
            100.0 * mean(&col(v, 2)), 100.0 * mean(&col(v, 3)));
    }

    let added_raw = mean(&col(&all, 2)) - mean(&col(&all, 0));
    let added_ref = mean(&col(&all, 3)) - mean(&col(&all, 1));
    println!("\nwake the candidate adds, before refinement: {:+.2} pp", 100.0 * added_raw);
    println!("wake the candidate adds, after  refinement: {:+.2} pp", 100.0 * added_ref);
    if added_raw.abs() > 1e-9 {
        println!("the refinement keeps {:.0}% of it", 100.0 * added_ref / added_raw);
    }
    if !band_a.is_empty() {
        println!("\nband sleep_state recall (independent of us, sleep-PERIOD only, so a small drop is");
        println!("the price of finding real wake, and a collapse is a red flag):");
        println!("  shipped {:.3}   candidate {:.3}   delta {:+.3}",
            mean(&band_a), mean(&band_b), mean(&band_b) - mean(&band_a));
    }
    println!("\n{}", census.line("the strap nights"));
}
