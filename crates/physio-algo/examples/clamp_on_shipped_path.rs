//! Does the conditional clamp survive `refine_wake`, or does the refinement eat what it adds?
//!
//!   cargo run --release -p physio-algo --example clamp_on_shipped_path
//!
//! Every measurement of `quiescent_hr_z_max` so far ran on PSG cohorts, and `refine_wake` declines on
//! all 144 of those spans because its density gate needs step samples no PSG cohort ships. Real WHOOP
//! nights carry about 30,000 step samples each, so the refinement DOES run for users - and it only ever
//! rewrites wake to light.
//!
//! The candidate adds wake. If the refinement removes the same wake again, the PSG gain is invisible on
//! the path the app runs, which is exactly the mistake this project already made once and wrote down.
//!
//! No truth here, so this is a mechanism measurement, not an accuracy one: how much wake each recipe
//! calls, before and after refinement, on the strap nights that carry steps.

mod common;

use common::{dirs_of, mean, read_accel, read_hr, read_meta, read_rr, read_steps, stage_idx, RefineCensus};

use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput, StageSegment};

const CANDIDATE_Z: f64 = 0.5;

/// Wake seconds as a share of the span.
fn wake_frac(segs: &[StageSegment], start: i64, end: i64) -> f64 {
    let span = (end - start).max(1) as f64;
    segs.iter().filter(|s| stage_idx(s.stage) == 0).map(|s| (s.end - s.start) as f64).sum::<f64>() / span
}

fn main() {
    let cand = Params { quiescent_hr_z_max: CANDIDATE_Z, ..Params::SHIPPED };
    let mut rows: Vec<[f64; 4]> = Vec::new();
    let mut refined_nights = 0usize;
    let mut census = RefineCensus::default();

    for dir in dirs_of("ours") {
        let Some((w0, w1, _)) = read_meta(&dir) else { continue };
        let steps = read_steps(&dir);
        if steps.is_empty() {
            continue;
        }
        let accel = read_accel(&dir);
        let input =
            SleepInput { start: w0, end: w1, hr: read_hr(&dir), rr: read_rr(&dir), accel: accel.clone() };

        let mut out = [0.0f64; 4];
        for (i, p) in [Params::SHIPPED, cand].iter().enumerate() {
            let prep = prepare_v2(&input, p);
            let raw = stage_v2_prepared(&prep, p);
            let fine = census.refine(&raw, &accel, &steps);
            out[i * 2] = wake_frac(&raw, w0, w1);
            out[i * 2 + 1] = wake_frac(&fine, w0, w1);
        }
        if (out[0] - out[1]).abs() > 1e-9 || (out[2] - out[3]).abs() > 1e-9 {
            refined_nights += 1;
        }
        rows.push(out);
    }

    if rows.is_empty() {
        println!("no `ours` nights carry a step stream; nothing to measure");
        return;
    }
    let col = |k: usize| rows.iter().map(|r| r[k]).collect::<Vec<f64>>();
    println!("{} strap nights with a step stream, {} of them actually refined\n", rows.len(), refined_nights);
    println!("{:<34} {:>10} {:>10}", "", "wake %", "vs shipped");
    let base_raw = mean(&col(0));
    for (name, k) in
        [("shipped, stage only", 0usize), ("shipped, + refine_wake", 1), ("candidate, stage only", 2),
         ("candidate, + refine_wake", 3)]
    {
        let m = mean(&col(k));
        println!("{name:<34} {:>9.2}% {:>+10.2}", 100.0 * m, 100.0 * (m - base_raw));
    }

    // The question, stated as one number: how much of the wake the candidate adds is still there after
    // the refinement has had its turn.
    let added_raw = mean(&col(2)) - mean(&col(0));
    let added_ref = mean(&col(3)) - mean(&col(1));
    println!("\nwake the candidate ADDS before refinement: {:+.2} pp", 100.0 * added_raw);
    println!("wake the candidate ADDS after  refinement: {:+.2} pp", 100.0 * added_ref);
    if added_raw.abs() > 1e-9 {
        println!("the refinement keeps {:.0}% of it", 100.0 * added_ref / added_raw);
    }
    println!("\n{}", census.line("the strap nights"));
}
