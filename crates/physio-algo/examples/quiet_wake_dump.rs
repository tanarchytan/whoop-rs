//! Per-epoch v2 terms for the quiet-wake question (V3 design R4).
//!
//!   cargo run --release -p physio-algo --example quiet_wake_dump > quiet-wake.csv
//!
//! One row per LABELLED epoch of the three PSG cohorts: truth, v2's label at SHIPPED params and under
//! `clamp_only_without_rr`, the z-scored HR / HRV the emission reads, and the raw motion features.
//! `move_frac` and `jerk_max` are private to `v2.rs`, so they are recomputed here with the same recipe
//! and pinned against the emission's own z(move) column before any row is written.

mod common;

use std::collections::HashMap;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, epoch_starts_v2, params::Params, prepare_v2, resp_regularity,
    SleepInput,
};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// `design[DEEP][W_DEEP_HRV|W_DEEP_HR|W_DEEP_MOTION]` are z(hrv), z(hr), z(move); slots 0, 1, 2
/// (`v2.rs` W_DEEP_* consts, written at `d[DEEP][..] = zhvv / zhrv / zmvv`). DEEP is row 0 of STAGE_ORDER.
const DEEP_ROW: usize = 0;
const COL_HRV: usize = 0;
const COL_HR: usize = 1;
const COL_MOVE: usize = 2;

/// Same reconstruction as `v2::beats_in` (private): beats of the seconds `lo..hi`, spread by interval.
fn beats_in(rr_by: &HashMap<i64, Vec<f64>>, lo: i64, hi: i64) -> Vec<(f64, f64)> {
    let mut beats = Vec::new();
    for bs in lo..hi {
        let Some(vs) = rr_by.get(&bs) else { continue };
        let mut off = 0.0;
        for (k, v) in vs.iter().enumerate() {
            let ms = v.clamp(300.0, 2000.0);
            if k > 0 {
                off += ms / 1000.0;
            }
            beats.push((bs as f64 + off, ms));
        }
    }
    beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    beats
}

fn median_of(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { 0.0 } else if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    let clamp = Params { clamp_only_without_rr: true, ..Params::SHIPPED };
    println!("cohort,night,epoch,truth,pred_v2,pred_clamp,z_hr,z_hrv,move_frac,jerk_max,clamped,has_rr,to_onset_min");
    for set in COHORTS {
        for dir in dirs_of(set) {
            let truth = read_truth(&dir);
            let Some((w0, w1, _)) = read_meta(&dir) else { continue };
            let accel = read_accel(&dir);
            if truth.is_empty() || accel.is_empty() {
                continue;
            }
            let (hr, rr) = (read_hr(&dir), read_rr(&dir));
            let night = dir.file_name().unwrap().to_string_lossy().to_string();

            // Per-second gravity mean and R-R by second, as `features` builds them.
            let mut gsum: HashMap<i64, (f64, f64, f64, f64)> = HashMap::new();
            for g in &accel {
                let e = gsum.entry(g.ts).or_insert((0.0, 0.0, 0.0, 0.0));
                *e = (e.0 + g.x, e.1 + g.y, e.2 + g.z, e.3 + 1.0);
            }
            let mut rr_by: HashMap<i64, Vec<f64>> = HashMap::new();
            for run in &rr {
                for &ms in &run.intervals {
                    rr_by.entry(run.ts).or_default().push(ms as f64);
                }
            }

            let input = SleepInput { start: w0, end: w1, hr, rr, accel };
            let prep = prepare_v2(&input, &Params::SHIPPED);
            let starts = epoch_starts_v2(&prep);
            let terms = emission_terms(&prep, &Params::SHIPPED);
            let pred = |p: &Params| -> Vec<usize> {
                decode_v2(&emissions_v2(&prep, p), &p.transition).iter().map(|s| stage_idx(*s)).collect()
            };
            let (pv2, pcl) = (pred(&Params::SHIPPED), pred(&clamp));

            // Raw motion per epoch: jerks between consecutive seconds that carry gravity.
            let jerks_of: Vec<Vec<f64>> = starts
                .iter()
                .map(|&e| {
                    let g: Vec<(f64, f64, f64)> = (e..e + 30)
                        .filter_map(|s| gsum.get(&s).map(|v| (v.0 / v.3, v.1 / v.3, v.2 / v.3)))
                        .collect();
                    g.windows(2)
                        .map(|w| ((w[0].0 - w[1].0).powi(2) + (w[0].1 - w[1].1).powi(2) + (w[0].2 - w[1].2).powi(2)).sqrt())
                        .collect()
                })
                .collect();
            let mut all: Vec<f64> = jerks_of.iter().flatten().copied().collect();
            let scale = if all.is_empty() { 1e-6 } else { median_of(&mut all) };
            let thr = scale * Params::SHIPPED.jerk_move_mult;
            let gaps: Vec<i64> = starts
                .iter()
                .map(|&e| ((e..e + 30).filter(|s| gsum.contains_key(s)).count() as i64 - 1).max(1))
                .collect();
            let mv: Vec<Option<f64>> = jerks_of
                .iter()
                .zip(&gaps)
                .map(|(j, g)| (!j.is_empty()).then(|| j.iter().filter(|&&x| x > thr).count() as f64 / *g as f64))
                .collect();

            // Pin: recomputed move_frac must reproduce the emission's own z(move) column.
            let present: Vec<f64> = mv.iter().flatten().copied().collect();
            if !present.is_empty() {
                let m = present.iter().sum::<f64>() / present.len() as f64;
                let sd0 = (present.iter().map(|x| (x - m).powi(2)).sum::<f64>() / present.len() as f64).sqrt();
                let sd = if sd0 == 0.0 { 1.0 } else { sd0 };
                for (i, v) in mv.iter().enumerate() {
                    let z = v.map_or(0.0, |x| (x - m) / sd);
                    let d = terms.design[i][DEEP_ROW][COL_MOVE];
                    assert!((z - d).abs() < 1e-6, "{set}/{night} epoch {i}: move z {z} vs design {d}");
                }
            }

            let idx_of = |s: i64| ((s - w0) / 30) as usize;
            let onset = starts.iter().find(|&&s| matches!(truth.get(&idx_of(s)), Some(1..=3))).copied();
            let Some(onset) = onset else { continue };
            for (i, &s) in starts.iter().enumerate() {
                let Some(&t) = truth.get(&idx_of(s)) else { continue };
                if !(0..4).contains(&t) {
                    continue;
                }
                let has_rr = resp_regularity(&beats_in(&rr_by, s - 90, s + 120)).is_some();
                let jmax = jerks_of[i].iter().copied().fold(0.0f64, f64::max);
                println!(
                    "{set},{night},{},{t},{},{},{:.6},{:.6},{},{:.6},{},{},{:.1}",
                    idx_of(s), pv2[i], pcl[i], terms.design[i][DEEP_ROW][COL_HR], terms.design[i][DEEP_ROW][COL_HRV],
                    mv[i].map_or("nan".to_string(), |x| format!("{x:.6}")), jmax, terms.clamped[i] as u8,
                    has_rr as u8, (s - onset) as f64 / 60.0,
                );
            }
        }
    }
}
