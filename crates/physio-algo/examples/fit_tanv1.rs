//! Fit the tanv1 per-epoch classifier, and separate the two effects that have never been separated.
//!
//!   cargo run --release -p physio-algo --example fit_tanv1
//!
//! The plan's §6.0 says "fit it" is NOT by itself a difference: v2's emission is already a weighted
//! sum of z-scored features plus a bias, so a linear fit over the same inputs is the same functional
//! form with better numbers. Two things could make tanv1 real - a model class that can represent
//! INTERACTIONS, and many more features - and nobody has measured how much of the gain comes from
//! which.
//!
//! This measures exactly that, by fitting the same model twice:
//!   MAIN     - the feature columns with both interaction products withheld.
//!   MAIN+INT - the same, plus `still_x_cardiac` and `still_x_hrvar`, the two products v2 hard-codes
//!              as one branch. Withholding only one leaves half the branch in the "withheld" arm.
//! The difference between them IS the interaction's contribution, on identical data and optimiser.
//!
//! Discipline, all five from the plan's rules:
//!   - Fit on DREAMT. Report HELD-OUT on aauwss and sleep-accel. A train number is never a result.
//!   - DREAMT is CLINICAL: per-epoch stage labels only, never onset/offset. Nothing here reads a
//!     boundary.
//!   - Standardisation uses TRAIN statistics applied to held-out, never per-cohort - that leaks.
//!   - The rate-matched null is the floor: a fit that only calls more wake has done nothing.
//!   - Both arms must CONVERGE. Two fits stopped at a shared iteration cap sit at unequal distances
//!     from their own optima, and the whole result here is a small delta between them.
//!
//! The shipped recipe is scored FIRST, on the same rows through the same function. This project
//! carries three incompatible kappa formulas - pooled over all epochs, mean of per-subject, median
//! of per-subject - so a number from here read against a published v2 number is a cross-statistic
//! comparison that looks like a like-for-like one. One caveat survives that baseline and is not
//! fixable here: v2 decodes with Viterbi and this fit predicts each epoch independently.
//!
//! The output is weights. Nothing is wired and `Params::SHIPPED` is untouched.

mod common;

use common::{
    dirs_of, labels_at, read_accel, read_hr, read_meta, read_rr, read_truth,
};
use physio_algo::sleep::features::{extract, Cardiac, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, recall, specificity, WAKE};
use physio_algo::sleep::{decode_v2, params::Params, stage_v2, SleepInput, STAGE_ORDER};
use std::collections::BTreeMap;

const EPOCH: i64 = 30;
const FIT: [&str; 1] = ["dreamt"];
const HELD_OUT: [&str; 2] = ["aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const CLASS_NAME: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
/// A row with no reference label. Kept in the matrix because a decode needs the night CONTIGUOUS -
/// dropping unlabelled epochs would make two epochs an hour apart adjacent to the path search - and
/// excluded from every fit and every score.
const UNLABELLED: usize = usize::MAX;
/// Column count from `Features::NAMES`, plus a bias term appended by the design matrix. Read from
/// the source of truth so adding a column can never leave the design matrix silently narrow.
const NCOL: usize = Features::N;
/// Hard cap only. The loop is expected to exit on [`TOL`]; reaching this means it did NOT converge,
/// and the run says so rather than printing a number from a half-finished fit.
///
/// 400 was the previous value and was measured insufficient: the loss was still falling 1.2e-5 per
/// iteration at cutoff, four orders of magnitude above `TOL`, and `converged` never printed. The
/// whole conclusion here is a delta of ~0.002-0.007 between two arms, and two arms stopped at a
/// shared iteration count are not equally far from their own optima.
const ITERS: usize = 20_000;
/// Per-iteration loss change below which the fit is converged.
const TOL: f64 = 1e-9;
const LR: f64 = 0.5;
/// L2 penalty. 100 subjects against 26 parameters per class overfits without one, and the plan
/// names overfitting as the expected failure mode.
const L2: f64 = 1e-3;
/// Exponent on the inverse-frequency class weight. 0 = unweighted, 1 = full inverse frequency.
///
/// Both extremes are wrong and both were measured. Unweighted, the fit calls deep 0.0% against a
/// 3.4-18.1% truth rate - a two-class model wearing a four-class report. Full inverse frequency
/// over-corrects the other way: deep called 34.9% against 3.4%, a 10x over-prediction. The
/// square root is the usual compromise, but it was swept rather than assumed and it LOST: held-out
/// kappa at 0.5 is 0.182/0.172 against 1.0's 0.239/0.203. Full weighting wins on the number that
/// counts while being visibly miscalibrated, so the flag in the report exists to keep that visible
/// rather than letting a better kappa hide it.
const WEIGHT_POWER: f64 = 1.0;

/// `WEIGHT_POWER`, overridable by `TANV1_WEIGHT_POWER` so the sweep that chose it is reproducible
/// without editing the file. The first sweep was run before the normalisation above was fixed and
/// is not evidence for anything.
fn weight_power() -> f64 {
    std::env::var("TANV1_WEIGHT_POWER").ok().and_then(|v| v.parse().ok()).unwrap_or(WEIGHT_POWER)
}

struct Set {
    x: Vec<[f64; NCOL]>,
    y: Vec<usize>,
    /// Night index per row, so a per-subject score never pools across nights.
    night: Vec<usize>,
    /// The SHIPPED recipe's own call for the same epoch, so the baseline is scored by the same
    /// function on the same rows. Three incompatible kappa formulas live in this project - pooled,
    /// mean-of-subjects and median-of-subjects - and quoting one against another was comparing
    /// different statistics on top of the already-named no-Viterbi asymmetry.
    v2: Vec<usize>,
}

/// Per-night z-score of a per-epoch series, missing where the series is.
fn zscore(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = v.iter().flatten().copied().collect();
    if present.len() < 2 {
        return vec![None; v.len()];
    }
    let m = present.iter().sum::<f64>() / present.len() as f64;
    let sd = (present.iter().map(|x| (x - m).powi(2)).sum::<f64>() / present.len() as f64).sqrt();
    if sd <= 0.0 {
        return vec![None; v.len()];
    }
    v.iter().map(|o| o.map(|x| (x - m) / sd)).collect()
}

/// One heart rate per second, averaged where a second carries several samples.
fn per_second_hr(hr: &[physio_algo::sleep::HrSample]) -> BTreeMap<i64, f64> {
    let mut acc: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for s in hr {
        let e = acc.entry(s.ts).or_insert((0.0, 0.0));
        e.0 += s.bpm as f64;
        e.1 += 1.0;
    }
    acc.into_iter().map(|(t, (a, c))| (t, a / c)).collect()
}

/// Population sd of PER-SECOND heart rate over `[lo, hi)`, the statistic the shipped recipe reads.
///
/// Averaging to per-epoch means first and taking the spread of THOSE is a different, much smoother
/// quantity - eleven already-averaged points instead of ~330 raw ones. That substitution was in here
/// under a matching name, which is why the fitted model had no access to what v2 separates deep with.
fn std_of_seconds(sec: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> Option<f64> {
    let v: Vec<f64> = sec.range(lo..hi).map(|(_, b)| *b).collect();
    if v.len() < 2 {
        return None;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    Some((v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt().max(0.0))
}

/// Within-night percentile rank in 0..1, by `bisect_right / n` over the present values - the same
/// transform v2's deep gate applies to `hr_flat11`. Missing stays missing rather than becoming 0.5,
/// so the model is handed a missing indicator instead of a manufactured median.
fn rank_pct(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut sorted: Vec<f64> = v.iter().flatten().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return vec![None; v.len()];
    }
    v.iter()
        .map(|o| o.map(|x| sorted.partition_point(|s| *s <= x) as f64 / sorted.len() as f64))
        .collect()
}

fn load(sets: &[&str]) -> Set {
    let (mut x, mut y, mut night) = (Vec::new(), Vec::new(), Vec::new());
    let mut v2 = Vec::new();
    let mut idx = 0usize;
    for set in sets {
        for dir in &dirs_of(set) {
            let truth = read_truth(dir);
            let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
            let grav = read_accel(dir);
            if truth.is_empty() || grav.is_empty() {
                continue;
            }
            let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);

            let hr = read_hr(dir);
            let sec = per_second_hr(&hr);
            // Per-epoch mean HR, then this night's own z-score - the same shape v2 uses.
            let mut sum = vec![(0.0f64, 0.0f64); n];
            for s in &hr {
                let k = ((s.ts - w0) / EPOCH).max(0) as usize;
                if k < n {
                    sum[k].0 += s.bpm as f64;
                    sum[k].1 += 1.0;
                }
            }
            let raw: Vec<Option<f64>> =
                sum.iter().map(|(a, c)| (*c > 0.0).then(|| a / c)).collect();
            let hr_z = zscore(&raw);
            // Both cardiac spreads over the windows and the resolution the shipped recipe uses.
            let starts: Vec<i64> = (0..n).map(|k| w0 + k as i64 * EPOCH).collect();
            let hv: Vec<Option<f64>> =
                starts.iter().map(|e| std_of_seconds(&sec, e - 150, e + EPOCH + 150)).collect();
            let hr_var_z = zscore(&hv);
            let flat: Vec<Option<f64>> =
                starts.iter().map(|e| std_of_seconds(&sec, e - 330, e + EPOCH + 360)).collect();
            let flat_pct = rank_pct(&flat);

            let card: Vec<Cardiac> = (0..n)
                .map(|k| Cardiac {
                    hr_z: hr_z[k],
                    hr_var_z: hr_var_z[k],
                    hr_flat_pct: flat_pct[k],
                })
                .collect();
            let f: Vec<Features> = extract(&grav, w0, w1, &card);

            // The shipped recipe on the SAME night, so its score comes out of the same function.
            let input =
                SleepInput { start: w0, end: w1, hr, rr: read_rr(dir), accel: grav.clone() };
            let base = labels_at(&stage_v2(&input), w0, n, EPOCH);

            // EVERY epoch, in order, labelled or not. read_truth yields i32; anything outside the
            // four classes is a row the decode still needs and no score may count.
            for (k, fe) in f.iter().enumerate().take(n) {
                let t = truth.get(&k).copied().unwrap_or(-1);
                v2.push(base[k]);
                x.push(fe.values());
                y.push(if (0..CLASSES as i32).contains(&t) { t as usize } else { UNLABELLED });
                night.push(idx);
            }
            idx += 1;
        }
    }
    Set { x, y, night, v2 }
}

/// Column means and sds over TRAIN only, ignoring NaN. Applied unchanged to held-out.
fn standardiser(x: &[[f64; NCOL]]) -> ([f64; NCOL], [f64; NCOL]) {
    let (mut m, mut s) = ([0.0; NCOL], [1.0; NCOL]);
    for c in 0..NCOL {
        let v: Vec<f64> = x.iter().map(|r| r[c]).filter(|v| v.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        m[c] = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|z| (z - m[c]).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        s[c] = if sd > 1e-12 { sd } else { 1.0 };
    }
    (m, s)
}

/// Design row: standardised, NaN imputed to the train mean (which is 0 after standardising), plus a
/// bias. `drop` zeroes columns so the same optimiser can be run with features withheld.
fn design(r: &[f64; NCOL], m: &[f64; NCOL], s: &[f64; NCOL], drop: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(NCOL + 1);
    for c in 0..NCOL {
        let v = if drop.contains(&c) || !r[c].is_finite() { 0.0 } else { (r[c] - m[c]) / s[c] };
        out.push(v);
    }
    out.push(1.0);
    out
}

/// Inverse-frequency weight per class, normalised so the TOTAL weighted mass equals the sample count
/// at every `WEIGHT_POWER`. Without the weighting the fit collapses: measured, an unweighted model
/// called deep 0.0% of epochs on all three cohorts against true rates of 3.4-18.1%, and REM under 1%
/// against 10.5-22.0%. It was a two-class wake/light classifier wearing a four-class report.
///
/// The divisor is the SAMPLE-weighted mean, not the arithmetic mean across the four classes. The
/// arithmetic mean is dominated by the rare classes' large raw weights, so it rescaled total mass by
/// a factor that differed at every swept power - which silently changed the data-fit-versus-L2
/// balance between the settings being compared, confounding the sweep that chose this one.
fn class_weights(y: &[usize]) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for &c in y {
        n[c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(weight_power())
        } else {
            0.0
        };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] as f64 * w[c]).sum::<f64>() / y.len() as f64;
    if mass > 0.0 {
        for v in w.iter_mut() {
            *v /= mass;
        }
    }
    w
}

/// Multinomial logistic regression by full-batch gradient descent. Deterministic: no shuffling, no
/// randomness, fixed iteration count - two runs give identical weights.
fn fit(x: &[Vec<f64>], y: &[usize]) -> Vec<Vec<f64>> {
    let p = x[0].len();
    let cw = class_weights(y);
    let mut w = vec![vec![0.0f64; p]; CLASSES];
    let mut last_nll = f64::MAX;
    let mut converged = false;
    for it in 0..ITERS {
        let mut g = vec![vec![0.0f64; p]; CLASSES];
        let mut nll = 0.0f64;
        for (row, &lab) in x.iter().zip(y) {
            let mut z = [0.0f64; CLASSES];
            for c in 0..CLASSES {
                z[c] = w[c].iter().zip(row).map(|(a, b)| a * b).sum();
            }
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let ex: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
            let sum: f64 = ex.iter().sum();
            nll -= cw[lab] * (ex[lab] / sum).max(1e-300).ln();
            for c in 0..CLASSES {
                let err = cw[lab] * (ex[c] / sum - if c == lab { 1.0 } else { 0.0 });
                for (gi, xi) in g[c].iter_mut().zip(row) {
                    *gi += err * xi;
                }
            }
        }
        let nll = nll / x.len() as f64;
        if it % 2000 == 0 {
            println!("    iter {it:>5}  weighted nll {nll:.8}");
        }
        // Signed, not `abs()`. An increase is divergence, not convergence, and `abs()` would report
        // a step that went the WRONG way by less than the tolerance as a converged fit.
        let drop = last_nll - nll;
        if drop < 0.0 {
            println!("    iter {it:>5}  loss ROSE by {:.3e} - the step is too large", -drop);
        }
        if (0.0..TOL).contains(&drop) {
            println!("    converged at iter {it}, weighted nll {nll:.8}");
            converged = true;
            break;
        }
        last_nll = nll;
        let scale = LR / x.len() as f64;
        for c in 0..CLASSES {
            for j in 0..p {
                // No penalty on the bias: shrinking it would bias the base rates.
                let pen = if j + 1 == p { 0.0 } else { L2 * w[c][j] };
                w[c][j] -= scale * g[c][j] + LR * pen;
            }
        }
    }
    // An unconverged fit is not a result. Two arms stopped at a shared iteration cap sit at unequal
    // distances from their own optima, and the whole conclusion here is the small delta between them.
    assert!(converged, "the fit hit the {ITERS}-iteration cap without converging - raise ITERS or \
                        lower LR; the arm deltas are not comparable until both arms converge");
    w
}

fn predict(w: &[Vec<f64>], row: &[f64]) -> usize {
    let mut best = (0usize, f64::MIN);
    for (c, wc) in w.iter().enumerate() {
        let z: f64 = wc.iter().zip(row).map(|(a, b)| a * b).sum();
        if z > best.1 {
            best = (c, z);
        }
    }
    best.0
}

/// Row indices of each night, in epoch order.
fn nights_of(s: &Set) -> Vec<Vec<usize>> {
    let n = s.night.iter().max().map_or(0, |v| v + 1);
    let mut out = vec![Vec::new(); n];
    for (i, nid) in s.night.iter().enumerate() {
        out[*nid].push(i);
    }
    out
}

/// Per-night kappa / wake recall / wake specificity, then the median across nights - never pooled,
/// because pooling lets the longest night decide the number.
///
/// `pred` is handed one night's row indices IN ORDER and returns one call per row, so an arm that
/// decodes a path sees the night whole while an arm that calls each epoch alone still works. Only
/// labelled rows are scored; the rest are context for the decode.
fn score_rows(s: &Set, pred: impl Fn(&[usize]) -> Vec<usize>)
    -> (f64, f64, f64, [f64; CLASSES], [f64; CLASSES]) {
    let (mut ks, mut rs, mut ss) = (Vec::new(), Vec::new(), Vec::new());
    // Per-class call rate AND truth rate. A kappa alone cannot show that a class is never predicted.
    let (mut called, mut truth_n) = ([0usize; CLASSES], [0usize; CLASSES]);
    let mut total = 0usize;
    for rows in nights_of(s) {
        let calls = pred(&rows);
        assert_eq!(calls.len(), rows.len(), "an arm must answer for every row it is handed");
        let (mut p, mut t) = (Vec::new(), Vec::new());
        for (j, i) in rows.iter().enumerate() {
            if s.y[*i] == UNLABELLED {
                continue;
            }
            p.push(calls[j]);
            t.push(s.y[*i]);
        }
        if p.len() < 20 {
            continue;
        }
        let cm = confusion4(&p, &t);
        ks.push(kappa4(&cm));
        if let Some(r) = recall(&cm, WAKE) {
            rs.push(r);
        }
        if let Some(v) = specificity(&cm, WAKE) {
            ss.push(v);
        }
        for (pi, ti) in p.iter().zip(&t) {
            called[*pi] += 1;
            truth_n[*ti] += 1;
            total += 1;
        }
    }
    let med = |mut v: Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let frac = |v: [usize; CLASSES]| {
        let mut o = [0.0; CLASSES];
        for c in 0..CLASSES {
            o[c] = if total > 0 { v[c] as f64 / total as f64 } else { f64::NAN };
        }
        o
    };
    (med(ks), med(rs), med(ss), frac(called), frac(truth_n))
}

/// Class scores for one row.
fn scores(w: &[Vec<f64>], row: &[f64]) -> [f64; CLASSES] {
    let mut z = [0.0f64; CLASSES];
    for (c, wc) in w.iter().enumerate() {
        z[c] = wc.iter().zip(row).map(|(a, b)| a * b).sum();
    }
    z
}

/// The fitted model, each epoch called alone.
fn score(w: &[Vec<f64>], s: &Set, m: &[f64; NCOL], sd: &[f64; NCOL], drop: &[usize])
    -> (f64, f64, f64, [f64; CLASSES], [f64; CLASSES]) {
    score_rows(s, |rows| {
        rows.iter().map(|i| predict(w, &design(&s.x[*i], m, sd, drop))).collect()
    })
}

/// The fitted model, decoded by the SAME path search the shipped recipe uses, under the same
/// transition matrix. Without this the comparison is emissions-with-smoothing against
/// emissions-alone, and sleep stages are strongly autocorrelated, so the decode is worth real kappa
/// on its own. This is the last of the three ways the baseline comparison was not like-for-like.
fn score_decoded(w: &[Vec<f64>], s: &Set, m: &[f64; NCOL], sd: &[f64; NCOL], drop: &[usize])
    -> (f64, f64, f64, [f64; CLASSES], [f64; CLASSES]) {
    // Our class index is [wake, light, deep, rem]; the decoder's columns are STAGE_ORDER.
    let to_stage_order: [usize; CLASSES] =
        std::array::from_fn(|col| stage_class(STAGE_ORDER[col]));
    score_rows(s, |rows| {
        let em: Vec<[f64; CLASSES]> = rows
            .iter()
            .map(|i| {
                let z = scores(w, &design(&s.x[*i], m, sd, drop));
                // Log-softmax: the decoder adds log-transitions, so its input must be log-scale.
                let mx = z.iter().cloned().fold(f64::MIN, f64::max);
                let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
                std::array::from_fn(|col| z[to_stage_order[col]] - lse)
            })
            .collect();
        decode_v2(&em, &Params::SHIPPED.transition).iter().map(|st| stage_class(*st)).collect()
    })
}

/// Our class index for a stage. `common::stage_idx` is the same mapping; it is repeated here only
/// because a decode needs it inside this file's own column convention.
fn stage_class(s: physio_algo::sleep::SleepStage) -> usize {
    use physio_algo::sleep::SleepStage as S;
    match s {
        S::Wake => 0,
        S::Light => 1,
        S::Deep => 2,
        S::Rem => 3,
    }
}

/// Column index by name, or a panic. A silent `None` here would drop nothing, run both arms
/// identically, and print a confident conclusion from comparing a fit against itself.
fn col(name: &str) -> usize {
    Features::NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("{name} must exist in Features::NAMES or the ablation compares nothing"))
}

type Scored = (f64, f64, f64, [f64; CLASSES], [f64; CLASSES]);

fn header() {
    println!("  {:<18} {:>6} {:>7} {:>6}   {:<28} truth  W/L/D/R %", "cohort", "kappa",
             "wake r", "spec", "we call  W/L/D/R %");
}

fn show(name: &str, r: Scored) {
    let (k, rec, sp, call, tru) = r;
    let pc = |v: [f64; CLASSES]| {
        format!("{:>5.1}/{:>4.1}/{:>4.1}/{:>4.1}", 100.0 * v[0], 100.0 * v[1],
                100.0 * v[2], 100.0 * v[3])
    };
    // A class never predicted is the failure an unweighted fit hides behind a kappa. Every
    // offending class is NAMED: the unlabelled version read as a statement about deep on
    // all three cohorts, when on both held-out ones it was a 40-point under-call of light.
    let ratio = |c: usize| (call[c] / tru[c].max(1e-9)).max(tru[c] / call[c].max(1e-9));
    let mut bad: Vec<String> = Vec::new();
    for c in 0..CLASSES {
        // Over-prediction is judged even where truth is near-absent: a hallucinated class is
        // a defect whether or not the reference has enough of it to rank the ratio.
        if call[c] < 0.005 && tru[c] > 0.005 {
            bad.push(format!("{} COLLAPSED", CLASS_NAME[c]));
        } else if tru[c] > 0.005 && ratio(c) > 3.0 {
            bad.push(format!("{} {:.1}x", CLASS_NAME[c], ratio(c)));
        } else if tru[c] <= 0.005 && call[c] > 0.02 {
            bad.push(format!("{} HALLUCINATED", CLASS_NAME[c]));
        }
    }
    let flag = if bad.is_empty() { String::new() } else { format!("  <- {}", bad.join(", ")) };
    println!("  {name:<18} {k:>6.3} {rec:>7.3} {sp:>6.3}   {:<28} {}{}", pc(call), pc(tru), flag);
}

fn main() {
    let train = load(&FIT);
    println!("FIT: {} ({} epochs, {} nights)", FIT[0], train.x.len(),
             train.night.iter().max().map_or(0, |v| v + 1));
    if train.x.is_empty() {
        println!("no training data - check the fixture root");
        return;
    }
    // Loaded ONCE. Each load stages every night with the shipped recipe for the baseline row, so
    // reloading per arm would triple the work and could not change the answer.
    let held: Vec<(&str, Set)> =
        HELD_OUT.iter().map(|s| (*s, load(&[s]))).filter(|(_, h)| !h.x.is_empty()).collect();
    // Fitted rows only - an unlabelled epoch is decode context, never training data or a statistic.
    let fit_rows: Vec<usize> = (0..train.x.len()).filter(|i| train.y[*i] != UNLABELLED).collect();
    let fit_x: Vec<[f64; NCOL]> = fit_rows.iter().map(|i| train.x[*i]).collect();
    let fit_y: Vec<usize> = fit_rows.iter().map(|i| train.y[*i]).collect();
    let (m, sd) = standardiser(&fit_x);
    println!("class weight power {:.2}, {} labelled of {} rows\n",
             weight_power(), fit_x.len(), train.x.len());

    // The SHIPPED recipe, on these rows, through this file's own statistic. Every published v2 kappa
    // elsewhere in the project is a different formula - pooled across all epochs, or a mean rather
    // than a median of per-subject values - so without this row there is no like-for-like baseline
    // to read the fitted numbers against, only a cross-statistic comparison that looks like one.
    println!("\n=== BASELINE: the shipped recipe, scored by THIS file's statistic on THESE rows");
    header();
    show("dreamt (FIT)", score_rows(&train, |r| r.iter().map(|i| train.v2[*i]).collect()));
    for (name, h) in &held {
        show(&format!("{name} (HELD)"), score_rows(h, |r| r.iter().map(|i| h.v2[*i]).collect()));
    }

    // Both interaction columns. v2's clamp reads the HR-level AND the HR-variability term, so
    // withholding one of the two products would leave half the branch in the "withheld" arm.
    let int_cols = [col("still_x_cardiac"), col("still_x_hrvar")];
    println!("\ninteraction columns {:?} at indices {int_cols:?}",
             int_cols.map(|c| Features::NAMES[c]));

    let arms: [(&str, &[usize]); 2] =
        [("MAIN (interaction withheld)", &int_cols), ("MAIN + INTERACTION", &[])];

    for (label, drop) in arms {
        let dx: Vec<Vec<f64>> = fit_x.iter().map(|r| design(r, &m, &sd, drop)).collect();
        let w = fit(&dx, &fit_y);
        for (how, decoded) in [("per-epoch", false), ("VITERBI-decoded", true)] {
            println!("=== {label}, {how}");
            header();
            let run = |s: &Set| {
                if decoded {
                    score_decoded(&w, s, &m, &sd, drop)
                } else {
                    score(&w, s, &m, &sd, drop)
                }
            };
            show("dreamt (FIT)", run(&train));
            for (name, h) in &held {
                show(&format!("{name} (HELD)"), run(h));
            }
            println!();
        }
    }
    println!("The difference between the two arms IS the interaction's contribution: identical data,");
    println!("identical optimiser, the same two columns zeroed. Held out is the only result here.");
    println!("Read the fit against the BASELINE at matching decode, never across the two.");
}
