//! Adopt the shipped emission's non-linearities into the fitted one, and measure each way of doing it.
//!
//!   cargo run --release -p physio-algo --example fit_hybrid
//!
//! The fitted emission has no non-linearity at all; the shipped one has several, and they are the
//! part a linear model cannot invent. `emission_terms` already computes them, so the per-epoch ones
//! can be handed to the fit as COLUMNS rather than reimplemented:
//!
//!   the move-fraction z, the deep-gate HINGE on the flatness rank, the deadzoned cardiac pair,
//!   the centred rotation RANK, and the stillness-clamp indicator. The plain hr, hr_var and
//!   respiration z columns are absent because TANV1 already carries them, to 1e-14.
//!
//! `terms.fixed` is not handed as a TRANSFORM column: its cycle prior and its jerk gate are
//! epoch-varying per-class biases, so the TANV1 and +TRANSFORM arms cannot express them. The
//! emission columns carry them already - `Terms::emission` starts each class at `fixed`.
//!
//! Four arms, all fitted identically, differing only in which columns exist:
//!   TANV1         - the 28 measured columns, no shipped structure.
//!   +TRANSFORM    - plus the shipped transforms above. Can the fit USE the non-linearities?
//!   +EMISSION     - plus the shipped emission's four values. Can it use the ANSWER?
//!   EMISSION ONLY - the four values and nothing else. Standardisation is per-column and
//!                   invertible, so this arm CAN express the shipped recipe exactly. What it
//!                   falls short by is objective mismatch, not a capacity bound.
//!
//! Fitted on DREAMT, so DREAMT is not a result. The two held-out cohorts are.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const FIT: &str = "dreamt";
const HELD: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
/// Class-weight exponents to sweep. 0 = the plain likelihood; higher rebalances the loss toward the
/// rare classes, and away from the prior the data actually has.
const POWERS: [f64; 3] = [0.0, 0.25, 0.5];
/// Widths of the two leading blocks [`row_of`] lays out; the emission block is the tail.
const N_TANV1: usize = Features::N;
const N_TRANSFORM: usize = 6;

/// Which blocks an arm keeps.
#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    tanv1: bool,
    transform: bool,
    emission: bool,
}

const ARMS: [Arm; 4] = [
    Arm { name: "TANV1", tanv1: true, transform: false, emission: false },
    Arm { name: "+TRANSFORM", tanv1: true, transform: true, emission: false },
    Arm { name: "+EMISSION", tanv1: true, transform: false, emission: true },
    Arm { name: "EMISSION ONLY", tanv1: false, transform: false, emission: true },
];

struct Night {
    /// tanv1 columns, then the shipped transforms, then the shipped emission.
    row: Vec<Vec<f64>>,
    em: Vec<[f64; CLASSES]>,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let raw = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if raw.is_empty() || accel.is_empty() {
            continue;
        }
        let n = n_meta.max(raw.keys().max().copied().unwrap_or(0) + 1);
        let (hr, rr) = (read_hr(dir), read_rr(dir));
        let f = extract(&accel, w0, w1, &cardiac_series(w0, n, EPOCH, &hr, &rr));
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        let em = physio_algo::sleep::emissions_v2(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        // `emissions_v2` DROPS an epoch with neither HR nor gravity. Truncating features to the
        // emission length would then misalign everything after the hole rather than failing.
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        assert!(f.len() >= em.len(), "{}: fewer feature rows than emissions", dir.display());
        let (deep, awake) = (
            stage_col(physio_algo::sleep::SleepStage::Deep),
            stage_col(physio_algo::sleep::SleepStage::Wake),
        );
        let row = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = f[e].values().to_vec();
                // The shipped transforms, read straight off the decomposition so they cannot drift
                // from what the recipe actually computes. The width is `N_TRANSFORM` by type. The
                // plain hr, hr_var and respiration z are omitted - they duplicate TANV1 columns.
                let transform: [f64; N_TRANSFORM] = [
                    d[deep][2],                                  // move_frac z
                    -d[deep][3],                                 // the deep-gate HINGE
                    d[awake][8],                                 // deadzoned hr_var z
                    d[awake][9],                                 // deadzoned hr z
                    d[awake][10],                                // centred rotation RANK
                    if terms.clamped[e] { 1.0 } else { 0.0 },    // the stillness clamp indicator
                ];
                v.extend_from_slice(&transform);
                v.extend_from_slice(&em[e]);
                v
            })
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { row, em, truth });
    }
    out
}

fn stage_col(s: physio_algo::sleep::SleepStage) -> usize {
    STAGE_ORDER.iter().position(|x| *x == s).expect("stage in STAGE_ORDER")
}

/// The columns an arm keeps, out of a full row.
fn row_of(full: &[f64], a: Arm) -> Vec<f64> {
    let mut v = Vec::new();
    if a.tanv1 {
        v.extend_from_slice(&full[..N_TANV1]);
    }
    if a.transform {
        v.extend_from_slice(&full[N_TANV1..N_TANV1 + N_TRANSFORM]);
    }
    if a.emission {
        v.extend_from_slice(&full[N_TANV1 + N_TRANSFORM..]);
    }
    v
}

/// Hand-built weights over the four emission columns. Standardisation is per column and invertible,
/// so `sd[c] * (em[c]-m[c])/sd[c] + m[c]` is `em[c]` again and the log-softmax only subtracts a
/// per-epoch constant, which no Viterbi path can see.
fn control_weights(m: &[f64], sd: &[f64]) -> Vec<Vec<f64>> {
    let n = m.len();
    let to_order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    // Row `k` is our class k; column `j` is emission column j, which is STAGE_ORDER[j].
    (0..CLASSES)
        .map(|k| {
            let j = to_order.iter().position(|c| *c == k).expect("class in STAGE_ORDER");
            let mut row = vec![0.0; n + 1];
            row[j] = sd[j];
            row[n] = m[j];
            row
        })
        .collect()
}

/// Stop unless the control decodes the shipped path epoch for epoch, on every night. A negative
/// result is only worth reading once the positive control passes.
fn assert_control(cohorts: &[(String, Vec<Night>)], arm: Arm, m: &[f64], sd: &[f64]) {
    let w = control_weights(m, sd);
    for (name, nights) in cohorts {
        let (_, want) = score(nights, |nt| nt.em.clone());
        let (_, got) = score(nights, |nt| fitted_em(nt, arm, &w, m, sd));
        assert_eq!(want.len(), got.len(), "{name}: control scored a different night count");
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert_eq!(a.len(), b.len(), "{name} night {i}: different epoch count");
            if let Some(e) = a.iter().zip(b).position(|(x, y)| x != y) {
                panic!("{name} night {i} epoch {e}: the control decoded {} where the \
                        shipped recipe decodes {} - the emission plumbing is wrong, not the model",
                       b[e], a[e]);
            }
        }
    }
    println!("  CONTROL passes: hand-set weights over the emission columns reproduce the shipped");
    println!("  staging exactly on every night, so the arms measure the FIT, not the wiring.\n");
}

/// Log-softmax emissions from one arm's fitted weights, in STAGE_ORDER columns.
fn fitted_em(nt: &Night, a: Arm, w: &[Vec<f64>], m: &[f64], sd: &[f64]) -> Vec<[f64; CLASSES]> {
    let to_order: [usize; CLASSES] = std::array::from_fn(|c| stage_idx(STAGE_ORDER[c]));
    nt.row
        .iter()
        .map(|full| {
            let z = scores(w, &design_row(&row_of(full, a), m, sd, &[]));
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            std::array::from_fn(|c| z[to_order[c]] - lse)
        })
        .collect()
}

/// Per-night kappa under one emission source, and the decoded path each night was scored on. The
/// shipped recipe and every arm go through here, so both are filtered and paired identically.
fn score(
    nights: &[Night], em_of: impl Fn(&Night) -> Vec<[f64; CLASSES]>,
) -> (Vec<f64>, Vec<Vec<usize>>) {
    let (mut ks, mut paths) = (Vec::new(), Vec::new());
    for nt in nights {
        let path: Vec<usize> = decode_v2(&em_of(nt), &Params::SHIPPED.transition)
            .iter()
            .map(|s| stage_idx(*s))
            .collect();
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            p.push(path[k]);
            t.push(*want);
        }
        if p.len() >= MIN_EPOCHS {
            ks.push(kappa4(&confusion4(&p, &t)));
        }
        paths.push(path);
    }
    (ks, paths)
}

fn main() {
    let cohorts: Vec<(String, Vec<Night>)> = std::iter::once((format!("{FIT} (FIT)"), load(FIT)))
        .chain(HELD.iter().map(|s| (format!("{s} (HELD)"), load(s))))
        .collect();
    // `once` puts the FIT cohort first, and the fit trains on exactly those rows - loaded once.
    let train = &cohorts[0].1;
    if train.is_empty() {
        println!("no {FIT} nights - check the fixture root");
        return;
    }

    // The positive control, on the EMISSION ONLY geometry: those four columns can express the
    // shipped recipe, so the harness must be able to reach it before any arm's failure means
    // anything about fitting.
    {
        let a = ARMS[3];
        let x: Vec<Vec<f64>> = train
            .iter()
            .flat_map(|nt| nt.row.iter().map(|r| row_of(r, a)))
            .collect();
        let (m, sd) = standardise_cols(&x);
        assert_control(&cohorts, a, &m, &sd);
    }

    // The baseline reads only each night's emissions and truth, so it is the same under every arm
    // and every power. Decode it once per cohort.
    let baselines: Vec<Vec<f64>> =
        cohorts.iter().map(|(_, nights)| score(nights, |nt| nt.em.clone()).0).collect();

    println!("Every arm: same optimiser, same decoder, same shipped transition. Only the COLUMNS");
    println!("differ. EMISSION ONLY can reproduce the shipped recipe exactly, so what it falls");
    println!("short by is objective mismatch, not a capacity bound.\n");
    println!("  {:<14} {:>4} {:>5} {:<20} {:>7} {:>7}   {:>10} {:>9} {:>4}   verdict",
             "arm", "cols", "pow", "cohort", "shipped", "arm", "paired d", "bar +/-", "n");

    // EMISSION ONLY's most favourable HELD-OUT row - mean, bar, cohort, power - so the closing
    // verdict is read off the table instead of asserted. The FIT cohort is excluded: the fit trains
    // on it, so it is not a result.
    let mut best: Option<(f64, f64, String, f64)> = None;

    for a in ARMS {
        let rows: Vec<(Vec<f64>, usize)> = train
            .iter()
            .flat_map(|nt| {
                nt.row.iter().zip(&nt.truth).filter_map(|(r, t)| t.map(|t| (row_of(r, a), t)))
            })
            .collect();
        let x: Vec<Vec<f64>> = rows.iter().map(|(r, _)| r.clone()).collect();
        let y: Vec<usize> = rows.iter().map(|(_, t)| *t).collect();
        let (m, sd) = standardise_cols(&x);
        let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
        let ncol = x[0].len();
        for power in POWERS {
            let w = fit(&dx, &y, power);
            for ((name, nights), base) in cohorts.iter().zip(&baselines) {
                let (arm, _) = score(nights, |nt| fitted_em(nt, a, &w, &m, &sd));
                let d: Vec<f64> = base.iter().zip(&arm).map(|(p, q)| q - p).collect();
                let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
                let first = name.starts_with(FIT);
                if a.name == ARMS[3].name
                    && !first
                    && mean.is_finite()
                    && best.as_ref().is_none_or(|b| mean > b.0)
                {
                    best = Some((mean, bar, name.clone(), power));
                }
                let v = if !mean.is_finite() {
                    "-".to_string()
                } else if mean.abs() > bar {
                    format!("{} ({:.2}x)", if mean > 0.0 { "BEATS SHIPPED" } else { "worse" },
                            mean.abs() / bar)
                } else {
                    "inside the bar".to_string()
                };
                println!("  {:<14} {:>4} {:>5} {:<20} {:>7.3} {:>7.3}   {mean:>+10.4} {bar:>9.4} {:>4}   {v}",
                         if first { a.name } else { "" },
                         if first { ncol.to_string() } else { String::new() },
                         if first { format!("{power:.2}") } else { String::new() },
                         name, median(&mut base.to_vec()), median(&mut arm.to_vec()), d.len());
            }
        }
        println!();
    }
    println!("If +TRANSFORM beats TANV1, those transform columns were the missing part.");
    match best {
        None => {
            println!("EMISSION ONLY produced no comparable held-out row, so it says nothing here.")
        }
        Some((mean, bar, cohort, power)) => {
            let verdict = if mean > bar {
                "REACHES it: handed the answer as four columns, the fit finds its way back"
            } else if mean < -bar {
                "does not reach it: handed the answer as four columns, a likelihood fit still \
                 walks away from it"
            } else {
                "neither reaches nor loses to it: handed the answer as four columns, the fit \
                 lands inside the bar"
            };
            println!("EMISSION ONLY could express the shipped recipe and {verdict}.");
            println!("(the most favourable of {} held-out rows: {cohort} at power {power:.2}, \
                      {mean:+.4} +/- {bar:.4} - a nominal bar, not corrected for that choice)",
                     HELD.len() * POWERS.len());
        }
    }
}
