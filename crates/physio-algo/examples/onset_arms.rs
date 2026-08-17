//! What actually fixes the 112-minute early onset? Arms, on PSG truth, one change each.
//!
//!   cargo run --release -p physio-algo --example onset_arms
//!
//! `onset_offset` established the defect: given the reference window, DREAMT staging calls sleep a
//! median 115 minutes before the subject falls asleep, while the detector alone is 31 minutes out. So
//! the emission layer is what puts the wind-down to sleep, and `motion_quiescent` is the suspect - it
//! clamps the awake cardiac term to at most zero whenever movement was observed and was none, leaving
//! AWAKE at ln(0.34) against LIGHT's ln(0.50). Lying still and awake, LIGHT wins by construction.
//!
//! Each arm is ONE change. Onset error is the headline; kappa rides beside it as the guard, because a
//! change that fixes onset by flooding the night with wake is not a fix.

mod common;

use common::{dirs_of, mean, read_accel, read_hr, read_meta, read_rr, read_truth, stage_at, stage_idx};

use physio_algo::sleep::metrics::{confusion4, kappa4};
use physio_algo::sleep::{params::Params, prepare_v2, stage_v2_prepared, SleepInput};

const EPOCH: i64 = 30;
const SUSTAINED: usize = 10;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];

/// First epoch opening a sustained sleep run.
fn onset_of(asleep: &[bool]) -> Option<usize> {
    (0..asleep.len().saturating_sub(SUSTAINED))
        .find(|i| asleep[*i..].iter().take(SUSTAINED).all(|b| *b))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// The arms. Every one is a single parameter move off SHIPPED, so a result is attributable.
fn arms() -> Vec<(&'static str, Params)> {
    let s = Params::SHIPPED;
    vec![
        ("shipped", s),
        // The clamp cannot be switched off by a parameter, so this is the nearest reachable probe:
        // widen the jerk gate so `motion_quiescent` almost never fires, which is what letting the
        // cardiac term speak during still wake would look like.
        // The CLEAN clamp-off arm. `jerk_gate_mult` was the earlier probe and it is confounded: the
        // same constant also gates `motion_gate_boost`, so moving it changes two mechanisms at once.
        ("quiescent_hr_z_max -inf (clamp NEVER)", Params { quiescent_hr_z_max: f64::NEG_INFINITY, ..s }),
        ("clamp_only_without_rr (THE CANDIDATE)", Params { clamp_only_without_rr: true, ..s }),
        ("awake_hr 0.4 -> 1.2", Params { awake_hr: 1.2, ..s }),
        ("awake_deadzone -> 0", Params { awake_deadzone: 0.0, ..s }),
        ("base_rate awake 0.34 -> 0.55", Params { base_rate: [0.55, 0.50, 0.15, 0.22], ..s }),
        ("cycle_rem_onset_minutes x2", Params { cycle_rem_onset_minutes: s.cycle_rem_onset_minutes * 2.0, ..s }),
        ("quiescent_hr_z_max 3.0", Params { quiescent_hr_z_max: 3.0, ..s }),
        ("quiescent_hr_z_max 2.0", Params { quiescent_hr_z_max: 2.0, ..s }),
        ("quiescent_hr_z_max 0.75", Params { quiescent_hr_z_max: 0.75, ..s }),
        ("quiescent_hr_z_max 0.25", Params { quiescent_hr_z_max: 0.25, ..s }),
        ("quiescent_hr_z_max -0.5", Params { quiescent_hr_z_max: -0.5, ..s }),
        ("quiescent_hr_z_max 1.5", Params { quiescent_hr_z_max: 1.5, ..s }),
        ("quiescent_hr_z_max 1.0", Params { quiescent_hr_z_max: 1.0, ..s }),
        ("quiescent_hr_z_max 0.5", Params { quiescent_hr_z_max: 0.5, ..s }),
        ("quiescent_hr_z_max 0.0", Params { quiescent_hr_z_max: 0.0, ..s }),
    ]
}

fn main() {
    println!("onset error in minutes (ours - truth); negative = we call sleep EARLY");
    println!("kappa is the guard: an arm that fixes onset by flooding wake is not a fix\n");

    for cohort in COHORTS {
        let mut nights = Vec::new();
        for dir in dirs_of(cohort) {
            let Some((w0, w1, n)) = read_meta(&dir) else { continue };
            let raw = read_truth(&dir);
            if raw.is_empty() {
                continue;
            }
            let mut truth = vec![usize::MAX; n];
            for (k, t) in raw {
                if k < n && (0..4).contains(&t) {
                    truth[k] = t as usize;
                }
            }
            let asleep: Vec<bool> = truth.iter().map(|t| *t != usize::MAX && *t != 0).collect();
            let Some(on) = onset_of(&asleep) else { continue };
            let input = SleepInput {
                start: w0,
                end: w1,
                hr: read_hr(&dir),
                rr: read_rr(&dir),
                accel: read_accel(&dir),
            };
            nights.push((input, w0, n, truth, on));
        }
        if nights.is_empty() {
            continue;
        }
        println!("=== {cohort}, {} nights", nights.len());
        println!("{:<42} {:>12} {:>12} {:>9}", "arm", "onset bias", "|err| med", "kappa");

        for (name, p) in arms() {
            let (mut errs, mut cm) = (Vec::new(), [[0i64; 4]; 4]);
            for (input, w0, n, truth, on) in &nights {
                let prep = prepare_v2(input, &p);
                let segs = stage_v2_prepared(&prep, &p);
                let lab: Vec<usize> = (0..*n)
                    .map(|k| {
                        let mid = w0 + k as i64 * EPOCH + EPOCH / 2;
                        stage_idx(stage_at(&segs, mid).unwrap_or_else(|| segs.last().unwrap().stage))
                    })
                    .collect();
                let sleep: Vec<bool> = lab.iter().map(|l| *l != 0).collect();
                if let Some(ours) = onset_of(&sleep) {
                    errs.push((ours as f64 - *on as f64) * EPOCH as f64 / 60.0);
                }
                for (k, t) in truth.iter().enumerate() {
                    if *t < 4 && k < lab.len() {
                        cm[*t][lab[k]] += 1;
                    }
                }
            }
            let mut abs: Vec<f64> = errs.iter().map(|e| e.abs()).collect();
            println!("{name:<42} {:>+11.1}m {:>11.1}m {:>9.4}", mean(&errs), median(&mut abs), kappa4(&cm));
        }
        println!();
    }
    println!("note: kappa here is POOLED over the cohort's epochs, not the per-subject mean sleep_eval");
    println!("prints. The two differ by ~0.015 and are not interchangeable.");
    let _ = confusion4(&[], &[]);
}
