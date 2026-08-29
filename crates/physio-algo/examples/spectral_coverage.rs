//! How many epochs can carry a frequency-domain HRV spectrum at all, per cohort.
//!
//!   cargo run --release -p physio-algo --example spectral_coverage
//!
//! `tanv1_next` ALREADY prints the per-cohort share; that part is duplicated here on purpose so this
//! runs in a second instead of ten minutes. What is new is the DISTRIBUTION - `bands_at` admits a
//! window on the interval coverage, and a cohort can clear the share while a quarter of its windows
//! sit far under the floor.

mod common;

use common::{dirs_of, median, pct, read_meta, read_rr, read_truth, require_psg};
use physio_algo::sleep::hrv_bands::{bands_series, MIN_COVERAGE, WINDOW_S};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];

/// Fraction of `[t0, t0 + WINDOW_S]` the intervals themselves account for - the quantity `bands_at`
/// admits on, recomputed here so the histogram and the gate cannot drift apart.
fn coverage_at(beats: &[(f64, f64)], t0: f64) -> f64 {
    let (a, b) = (t0, t0 + WINDOW_S);
    beats.iter().filter(|(t, _)| *t >= a && *t <= b).map(|(_, ms)| ms / 1000.0).sum::<f64>()
        / WINDOW_S
}

fn main() {
    println!("window {WINDOW_S} s, admitted at coverage >= {MIN_COVERAGE}\n");
    println!("{:<14} {:>7} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7}", "cohort", "nights", "epochs",
             "scored", "share", "p25", "median", "p75");

    for set in COHORTS {
        require_psg(set);
        let (mut nights, mut epochs, mut scored) = (0usize, 0usize, 0usize);
        let mut cov: Vec<f64> = Vec::new();

        for dir in &dirs_of(set) {
            let raw = read_truth(dir);
            let Some((w0, _w1, n_meta)) = read_meta(dir) else { continue };
            if raw.is_empty() {
                continue;
            }
            let n = n_meta.max(raw.keys().max().copied().unwrap_or(0) + 1);
            let rr = read_rr(dir);
            let beats: Vec<(f64, f64)> = rr
                .iter()
                .flat_map(|r| r.intervals.iter().map(move |ms| (r.ts as f64, f64::from(*ms))))
                .collect();
            let end = (w0 + n as i64 * EPOCH) as f64;
            let bands = bands_series(&beats, w0 as f64, end, EPOCH as f64);

            nights += 1;
            epochs += bands.len();
            scored += bands.iter().filter(|b| b.is_some()).count();
            for k in 0..bands.len() {
                let mid = w0 as f64 + (k as f64 + 0.5) * EPOCH as f64;
                cov.push(coverage_at(&beats, mid - WINDOW_S / 2.0));
            }
        }

        let share = if epochs == 0 { 0.0 } else { scored as f64 / epochs as f64 };
        let (p25, med, p75) = if cov.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (pct(&mut cov, 0.25), median(&mut cov), pct(&mut cov, 0.75))
        };
        println!("{set:<14} {nights:>7} {epochs:>8} {scored:>8} {:>8.1}% {p25:>7.3} {med:>7.3} {p75:>7.3}",
                 share * 100.0);
    }

    println!("\nA cohort whose median coverage sits under the floor cannot support the family at all,\n\
              and its SPECTRAL columns are dead columns wearing a name.");
}
