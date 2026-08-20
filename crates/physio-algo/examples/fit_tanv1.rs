//! Fit the tanv1 per-epoch classifier, and separate the two effects that have never been separated.
//!
//!   cargo run --release -p physio-algo --example fit_tanv1
//!
//! The shipped emission is ALREADY a weighted sum of z-scored features plus a bias, so a linear fit
//! over the same inputs is the same functional form with better numbers. Two things could make this
//! different - a model class that can represent INTERACTIONS, and many more features - and nobody
//! has measured how much of any gain comes from which.
//!
//! Three arms, fitted identically, differing only in which columns are zeroed:
//!   MAIN       - both interaction products withheld.
//!   MAIN+INT   - everything. The difference from MAIN is the interaction's contribution.
//!   NO R-R     - `resp_z` withheld. The difference from MAIN+INT is the R-R channel's.
//!
//! Discipline:
//!   - Fit on DREAMT. Report HELD-OUT on aauwss and sleep-accel. A train number is never a result.
//!   - DREAMT is CLINICAL: per-epoch stage labels only, never onset/offset. Nothing reads a boundary.
//!   - Standardisation uses TRAIN statistics applied to held-out, never per-cohort - that leaks.
//!   - Both arms must CONVERGE. Two fits stopped at a shared cap sit at unequal distances from their
//!     own optima, and the results here are small deltas between them.
//!   - EVERY hyperparameter is chosen inside the fit cohort.
//!   - Every claim is a PAIRED per-night difference against its own bar. Two arms' medians are
//!     separate order statistics and subtracting them is a rank artefact.
//!
//! THE BASELINE IS NOT BLIND, and this is the largest caveat on every number here. The shipped
//! params were selected by a process that watched kappa on ALL THREE cohorts and rejected changes
//! that hurt the other two. This fit never sees either held-out cohort. So the baseline had
//! held-out exposure the fit did not, and part of its held-out margin is that exposure rather than
//! a better recipe. The direction of the bias is knowable; its size is not.
//!
//! Three more caveats. The class weighting rebalances the loss while `predict` takes a plain argmax,
//! so the printed call rates are not calibrated probabilities; undoing it post hoc bakes in DREAMT's
//! prevalence and makes held-out worse. `clock` is a fraction of the fixture window, whose labelled
//! part starts ~24% in on DREAMT and at zero on both held-out cohorts, so its coefficient is
//! extrapolated over every held-out sleep onset. And the class-weight exponent is selected ONCE, on
//! the full column set, then reused by the two ablation arms that are missing columns - holding the
//! weighting fixed across arms is the point, but it favours MAIN+INT, the comparand in both.
//!
//! The output is weights. Nothing is wired and `Params::SHIPPED` is untouched.

mod common;

use common::{
    cardiac_series, dirs_of, labels_at, read_accel, read_hr, read_meta, read_rr, read_truth,
    stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar, recall, specificity, WAKE};
use physio_algo::sleep::{decode_v2, params::Params, stage_v2, SleepInput, STAGE_ORDER};

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
/// Hard cap only - the loop exits on [`TOL`], and reaching this means it did NOT converge. Two arms
/// stopped at a shared iteration count are not equally far from their own optima, and the whole
/// result here is a small delta between them.
const ITERS: usize = 20_000;
/// Per-iteration loss change below which the fit is converged.
const TOL: f64 = 1e-9;
const LR: f64 = 0.5;
/// L2 penalty. 100 subjects against 30 parameters per class needs one. It does NOT explain the
/// held-out loss: swept to 1000x this, held-out degrades in step with the fit cohort, which is a
/// distribution gap rather than under-regularisation.
const L2: f64 = 1e-3;
/// Candidate exponents on the inverse-frequency class weight. 0 = unweighted, which collapses deep
/// and REM entirely; 1 = full inverse frequency, which over-calls them. Chosen by [`select_power`]
/// INSIDE the fit cohort, never against held-out.
const POWERS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Fixes the exponent instead of selecting it, for reproducing one setting by hand.
fn power_override() -> Option<f64> {
    std::env::var("TANV1_WEIGHT_POWER").ok().and_then(|v| v.parse().ok())
}

struct Set {
    x: Vec<[f64; NCOL]>,
    y: Vec<usize>,
    /// Night index per row, so a per-subject score never pools across nights.
    night: Vec<usize>,
    /// The SHIPPED recipe's own call for the same epoch, so the baseline is scored by the same
    /// function on the same rows. This project carries three incompatible kappa formulas, and
    /// quoting one against another compares different statistics.
    v2: Vec<usize>,
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
            let rr = read_rr(dir);
            let card = cardiac_series(w0, n, EPOCH, &hr, &rr);
            let f: Vec<Features> = extract(&grav, w0, w1, &card);

            // The shipped recipe on the SAME night, so its score comes out of the same function.
            let input = SleepInput { start: w0, end: w1, hr, rr, accel: grav.clone() };
            let base = labels_at(&stage_v2(&input), w0, n, EPOCH);

            // `n` comes from the meta window; `extract` derives its own count from `[w0, w1)`. If a
            // truth key ever ran past that window the loop below would silently stop short of it
            // rather than failing, which every other integrity violation in this corpus does not.
            assert!(f.len() >= n, "{}: {n} epochs of truth against {} of features",
                    dir.display(), f.len());

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

/// Inverse-frequency weight per class, normalised so the TOTAL weighted mass equals the sample
/// count at every power. The divisor is the SAMPLE-weighted mean: an arithmetic mean across four
/// classes is dominated by the rare ones and rescales mass differently at each power.
fn class_weights(y: &[usize], power: f64) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for &c in y {
        n[c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(power)
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
fn fit(x: &[Vec<f64>], y: &[usize], power: f64) -> Vec<Vec<f64>> {
    let p = x[0].len();
    let cw = class_weights(y, power);
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

/// The nights of `s` that `keep` accepts, renumbered from zero. Row order and alignment are
/// preserved, so a decode over the subset still sees each night contiguous.
fn subset(s: &Set, keep: impl Fn(usize) -> bool) -> Set {
    let (mut x, mut y, mut night, mut v2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut map = std::collections::BTreeMap::new();
    for i in 0..s.x.len() {
        if !keep(s.night[i]) {
            continue;
        }
        let next = map.len();
        let nid = *map.entry(s.night[i]).or_insert(next);
        x.push(s.x[i]);
        y.push(s.y[i]);
        v2.push(s.v2[i]);
        night.push(nid);
    }
    Set { x, y, night, v2 }
}

/// Choose the class-weight exponent INSIDE the fit cohort: fit on its even-numbered nights, score on
/// its odd ones. Choosing on held-out spends researcher freedom against the only clean estimate
/// there is, and here the two disagree - the fit cohort prefers a different exponent.
fn select_power(train: &Set, decoded: bool) -> f64 {
    if let Some(p) = power_override() {
        println!("class weight power {p:.2} (fixed by environment, not selected)");
        return p;
    }
    let inner_fit = subset(train, |n| n % 2 == 0);
    let inner_val = subset(train, |n| n % 2 == 1);
    let rows: Vec<usize> = (0..inner_fit.x.len()).filter(|i| inner_fit.y[*i] != UNLABELLED).collect();
    let ix: Vec<[f64; NCOL]> = rows.iter().map(|i| inner_fit.x[*i]).collect();
    let iy: Vec<usize> = rows.iter().map(|i| inner_fit.y[*i]).collect();
    let (m, sd) = standardiser(&ix);
    let dx: Vec<Vec<f64>> = ix.iter().map(|r| design(r, &m, &sd, &[])).collect();

    println!("SELECT class weight power for {} on the fit cohort alone: {} nights fit, {} scored",
             if decoded { "VITERBI-decoded" } else { "per-epoch" },
             inner_fit.night.iter().max().map_or(0, |v| v + 1),
             inner_val.night.iter().max().map_or(0, |v| v + 1));
    let mut best = (POWERS[0], f64::MIN);
    for p in POWERS {
        let w = fit(&dx, &iy, p);
        // Selected against the statistic it is REPORTED against. Both decodes currently pick the
        // same exponent, so this is a precaution rather than a live correction.
        let k = if decoded {
            score_decoded(&w, &inner_val, &m, &sd, &[]).kappa
        } else {
            score(&w, &inner_val, &m, &sd, &[]).kappa
        };
        println!("  power {p:.2}  inner-val kappa {k:.4}");
        if k > best.1 {
            best = (p, k);
        }
    }
    println!("  -> {:.2}\n", best.0);
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
fn score_rows(s: &Set, pred: impl Fn(&[usize]) -> Vec<usize>) -> Scored {
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
    Scored {
        kappa: med(ks.clone()),
        wake_recall: med(rs),
        wake_spec: med(ss),
        called: frac(called),
        truth: frac(truth_n),
        per_night: ks,
    }
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
fn score(w: &[Vec<f64>], s: &Set, m: &[f64; NCOL], sd: &[f64; NCOL], drop: &[usize]) -> Scored {
    score_rows(s, |rows| {
        rows.iter().map(|i| predict(w, &design(&s.x[*i], m, sd, drop))).collect()
    })
}

/// The fitted model, decoded by the SAME path search the shipped recipe uses under the same
/// transition matrix. Stages are strongly autocorrelated, so scoring a decoded baseline against
/// undecoded emissions measures the decode rather than the model.
fn score_decoded(w: &[Vec<f64>], s: &Set, m: &[f64; NCOL], sd: &[f64; NCOL], drop: &[usize])
    -> Scored {
    // Our class index is [wake, light, deep, rem]; the decoder's columns are STAGE_ORDER.
    let to_stage_order: [usize; CLASSES] =
        std::array::from_fn(|col| stage_idx(STAGE_ORDER[col]));
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
        decode_v2(&em, &Params::SHIPPED.transition).iter().map(|st| stage_idx(*st)).collect()
    })
}


/// Column index by name, or a panic. A silent `None` here would drop nothing, run both arms
/// identically, and print a confident conclusion from comparing a fit against itself.
fn col(name: &str) -> usize {
    Features::NAMES
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("{name} must exist in Features::NAMES or the ablation compares nothing"))
}

/// One arm's score on one cohort. `per_night` is kept because the medians of two arms are two
/// SEPARATE order statistics: subtracting them is a rank artefact, not a per-night effect, and it
/// reported a 0.035 kappa loss where the paired difference is 0.0008 and inside the noise.
struct Scored {
    kappa: f64,
    wake_recall: f64,
    wake_spec: f64,
    called: [f64; CLASSES],
    truth: [f64; CLASSES],
    per_night: Vec<f64>,
}

/// Mean paired difference and the resolvable bar for two arms scored on the same nights.
fn paired(a: &Scored, b: &Scored) -> (f64, f64, usize) {
    // Positional pairing is only meaningful if both arms scored the same nights in the same order.
    assert_eq!(a.per_night.len(), b.per_night.len(),
               "paired arms scored different night counts - the zip would misalign every pair");
    let d: Vec<f64> = a.per_night.iter().zip(&b.per_night).map(|(x, y)| y - x).collect();
    let n = d.len();
    paired_bar(&d).map_or((f64::NAN, f64::NAN, n), |(m, bar)| (m, bar, n))
}

/// One paired row: mean difference, the bar, and whether the difference clears it.
fn paired_row(name: &str, a: &Scored, b: &Scored) {
    let (mean, bar, n) = paired(a, b);
    let verdict = if mean.abs() > bar {
        format!("RESOLVED ({:.2}x the bar)", mean.abs() / bar)
    } else {
        "inside the bar - noise".to_string()
    };
    println!("  {name:<18} {mean:>+10.4} {bar:>10.4} {n:>6}   {verdict}");
}

fn header() {
    println!("  {:<18} {:>6} {:>7} {:>6}   {:<28} truth  W/L/D/R %", "cohort", "kappa",
             "wake r", "spec", "we call  W/L/D/R %");
}

fn show(name: &str, r: &Scored) {
    let (k, rec, sp, call, tru) = (r.kappa, r.wake_recall, r.wake_spec, r.called, r.truth);
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
    println!("{} labelled of {} rows\n", fit_x.len(), train.x.len());
    // One exponent per decode, each chosen against the statistic it is reported against.
    let powers = [select_power(&train, false), select_power(&train, true)];

    // The SHIPPED recipe, on these rows, through this file's own statistic. Every published v2 kappa
    // elsewhere in the project is a different formula - pooled across all epochs, or a mean rather
    // than a median of per-subject values - so without this row there is no like-for-like baseline
    // to read the fitted numbers against, only a cross-statistic comparison that looks like one.
    println!("\n=== BASELINE: the shipped recipe, scored by THIS file's statistic on THESE rows");
    header();
    let base: Vec<Scored> = std::iter::once(score_rows(&train, |r| {
        r.iter().map(|i| train.v2[*i]).collect()
    }))
    .chain(held.iter().map(|(_, h)| score_rows(h, |r| r.iter().map(|i| h.v2[*i]).collect())))
    .collect();
    show("dreamt (FIT)", &base[0]);
    for ((name, _), sc) in held.iter().zip(&base[1..]) {
        show(&format!("{name} (HELD)"), sc);
    }

    // Both interaction columns: the clamp reads the HR-level AND the HR-variability term, so
    // withholding one leaves half the branch in the "withheld" arm.
    let int_cols = [col("still_x_cardiac"), col("still_x_hrvar")];
    println!("\ninteraction columns {:?} at indices {int_cols:?}",
             int_cols.map(|c| Features::NAMES[c]));

    // A third arm isolates R-R. Its gain was first read off a delta between two commits that also
    // changed how the class weight is selected, which is an attribution rather than a measurement.
    let rr_col = [col("resp_z")];
    let arms: [(&str, &[usize]); 3] = [
        ("MAIN (interaction withheld)", &int_cols),
        ("MAIN + INTERACTION", &[]),
        ("NO R-R (resp_z withheld)", &rr_col),
    ];

    // Kept per arm and per decode so the interaction is judged by a PAIRED per-night difference.
    let mut by_arm: Vec<[Vec<Scored>; 2]> = Vec::new();
    let names: Vec<String> = std::iter::once("dreamt (FIT)".to_string())
        .chain(held.iter().map(|(n, _)| format!("{n} (HELD)")))
        .collect();

    for (label, drop) in arms {
        let dx: Vec<Vec<f64>> = fit_x.iter().map(|r| design(r, &m, &sd, drop)).collect();
        let mut both: [Vec<Scored>; 2] = [Vec::new(), Vec::new()];
        for (d, (how, decoded)) in [("per-epoch", false), ("VITERBI-decoded", true)].iter().enumerate() {
            let w = fit(&dx, &fit_y, powers[d]);
            println!("=== {label}, {how} (power {:.2})", powers[d]);
            header();
            let run = |s: &Set| {
                if *decoded {
                    score_decoded(&w, s, &m, &sd, drop)
                } else {
                    score(&w, s, &m, &sd, drop)
                }
            };
            both[d].push(run(&train));
            for (_, h) in &held {
                both[d].push(run(h));
            }
            for (name, sc) in names.iter().zip(&both[d]) {
                show(name, sc);
            }
            // The headline is a difference too, and a difference of medians is not a result.
            println!("  vs BASELINE, paired per night:");
            for (i, name) in names.iter().enumerate() {
                paired_row(name, &base[i], &both[d][i]);
            }
            println!();
        }
        by_arm.push(both);
    }

    // The interaction's contribution, per night rather than between two medians. Two arms' medians
    // are separate order statistics and their difference moves when ONE night changes rank.
    println!("=== THE INTERACTION, as a paired per-night difference (MAIN+INT minus MAIN)");
    println!("  {:<18} {:>10} {:>10} {:>6}   verdict", "cohort", "mean d", "bar +/-", "n");
    for (d, how) in ["per-epoch", "VITERBI-decoded"].iter().enumerate() {
        println!("  {how}");
        for (i, name) in names.iter().enumerate() {
            paired_row(name, &by_arm[0][d][i], &by_arm[1][d][i]);
        }
    }

    println!("
=== THE R-R CHANNEL, paired per night (MAIN+INT minus the same fit without resp_z)");
    println!("  A cohort with no beats is NOT a test: resp_z is missing on every epoch, `design`");
    println!("  imputes it to the same zero a withheld column gets, and both arms see the same");
    println!("  input. Its near-zero is arithmetic, not evidence.");
    println!("  {:<18} {:>10} {:>10} {:>6}   verdict", "cohort", "mean d", "bar +/-", "n");
    // Coverage, not presence. A cohort with a handful of beat-carrying nights would otherwise read
    // as a full test while most of its paired differences are identically zero.
    let coverage: Vec<f64> = std::iter::once(&train)
        .chain(held.iter().map(|(_, h)| h))
        .map(|s| {
            let c = col("resp_z");
            s.x.iter().filter(|r| r[c].is_finite()).count() as f64 / s.x.len().max(1) as f64
        })
        .collect();
    for (d, how) in ["per-epoch", "VITERBI-decoded"].iter().enumerate() {
        println!("  {how}");
        for (i, name) in names.iter().enumerate() {
            if coverage[i] < 0.5 {
                println!("  {name:<18} {:>10} {:>10} {:>6}   {:.0}% of epochs carry beats - NOT A TEST",
                         "-", "-", "-", 100.0 * coverage[i]);
                continue;
            }
            paired_row(&format!("{name} [{:.0}%]", 100.0 * coverage[i]),
                       &by_arm[2][d][i], &by_arm[1][d][i]);
        }
    }
    println!("\nThe paired row is the interaction's contribution. A difference of medians is not:");
    println!("it reported -0.035 where the paired mean is -0.0008 against a +/-0.0068 bar.");
    println!("Read the fit against the BASELINE at matching decode, never across the two.");
}
