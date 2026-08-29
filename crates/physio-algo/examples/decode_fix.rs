//! Repair the decode-side loss `emission_steps` located, and check the repair is not just a v2 tune.
//!
//!   cargo run --release -p physio-algo --example decode_fix
//!
//! `emission_steps` found the correction's argmax MATCHES v2 (+0.0043) while its decode loses 1.71x,
//! and named the mechanism: the correction flattens the emission, the sticky prior then wins more
//! often, runs per night fall 44 -> 32 against a truth of 51, and REM collapses. Two fixes follow
//! from that, and each has a control that can take the win away:
//!
//!   BETA        transition^beta, so beta < 1 charges less for a stage change. One scalar aimed
//!               straight at the measured over-smoothing. Its control is applying it to V2 ALONE:
//!               if the prior is simply too sticky, that is a v2 finding and not a tanv1 one.
//!   PERCEPTRON  train theta against the DECODE - the structured update, on epochs where the
//!               Viterbi path disagrees with truth, rather than per-epoch likelihood. This is the
//!               diagnosed mechanism addressed directly. Starting at theta = 0 it starts AT v2.
//!
//! Every hyperparameter is chosen by an inner leave-one-out over the TRAIN cohorts and applied once,
//! so no arm selects against the cohort it reports. Every number is paired per night against v2.

mod common;

use common::lr::{design_row, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::{extract, Features};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const CLASS_NAMES: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
const MIN_EPOCHS: usize = 20;
const NCOL: usize = Features::N + 6 + 1;
/// The likelihood fit `fit_residual` reports, reproduced so this harness compares against that arm.
const L2: f64 = 1.0;
const ITERS: usize = 6_000;
const TOL: f64 = 1e-10;
const LR: f64 = 0.5;
const WEIGHT_POWER: f64 = 0.5;
/// How much of the likelihood correction to apply; 0 is v2 exactly.
const ALPHAS: [f64; 5] = [0.0, 0.2, 0.5, 0.7, 1.0];
/// `transition^beta`. Below 1 the prior charges less for a stage change, so the path fragments more.
const BETAS: [f64; 7] = [0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5];
/// Floor under every transition entry. 0.0 keeps viterbi's own 1e-9, which makes `Wake -> Rem` and
/// `Wake -> Deep` cost 20.7 log units, so one wake epoch inside a REM bout cannot return to REM.
const FLOORS: [f64; 4] = [0.0, 1e-4, 1e-3, 1e-2];
/// Structured passes over the training nights. Pass 0 is theta = 0, which is v2.
const PASSES: usize = 24;
/// Structured step sizes to sweep, per LABELLED epoch. Untuned and unnormalised, one pass drove
/// |theta| to 3.4 and held-out kappa down 0.29, so the rate is selected inside like every knob.
const P_RATES: [f64; 5] = [0.02, 0.1, 0.5, 2.0, 8.0];

struct Night {
    row: Vec<Vec<f64>>,
    offset: Vec<[f64; CLASSES]>,
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
        let em = emissions_v2(&prep, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        let deep = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 2).expect("deep");
        let awake = STAGE_ORDER.iter().position(|s| stage_idx(*s) == 0).expect("wake");
        let row = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                let mut v = f[e].values().to_vec();
                v.extend_from_slice(&[
                    -d[deep][3],
                    d[awake][8],
                    d[awake][9],
                    d[awake][10],
                    d[deep][11],
                    if terms.clamped[e] { 1.0 } else { 0.0 },
                ]);
                v
            })
            .collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { row, offset: em[..].to_vec(), truth });
    }
    out
}

fn col_of(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

fn class_weights(y: &[usize]) -> [f64; CLASSES] {
    let mut n = [0usize; CLASSES];
    for c in y {
        n[*c] += 1;
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0 {
            (y.len() as f64 / (CLASSES as f64 * n[c] as f64)).powf(WEIGHT_POWER)
        } else {
            0.0
        };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] as f64 * w[c]).sum::<f64>() / y.len() as f64;
    for v in w.iter_mut() {
        *v /= mass;
    }
    w
}

/// `SHIPPED.transition` floored at `floor`, then raised to `beta`. Beta scales every charge at once;
/// the floor is a separate lever, because `Wake -> Deep` and `Wake -> Rem` are hard zeros that `powf`
/// leaves at zero and viterbi then charges 20.7 log units for.
fn tempered(beta: f64, floor: f64) -> [[f64; CLASSES]; CLASSES] {
    Params::SHIPPED.transition.map(|row| row.map(|v| v.max(floor).powf(beta)))
}

/// The likelihood fit: a correction on a fixed per-class offset, starting at zero.
fn fit_likelihood(x: &[Vec<f64>], off: &[[f64; CLASSES]], y: &[usize]) -> Vec<Vec<f64>> {
    let p = x[0].len();
    let cw = class_weights(y);
    let mut th = vec![vec![0.0f64; p]; CLASSES];
    let mut last = f64::MAX;
    let mut converged = false;
    for _ in 0..ITERS {
        let mut g = vec![vec![0.0f64; p]; CLASSES];
        let mut nll = 0.0f64;
        for ((row, o), &lab) in x.iter().zip(off).zip(y) {
            let mut z = [0.0f64; CLASSES];
            for c in 0..CLASSES {
                z[c] = o[col_of(c)] + th[c].iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
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
        let drop = last - nll;
        if (0.0..TOL).contains(&drop) {
            converged = true;
            break;
        }
        last = nll;
        let scale = LR / x.len() as f64;
        let decay = (1.0 - LR * L2).max(0.0);
        for c in 0..CLASSES {
            for j in 0..p {
                th[c][j] = decay * th[c][j] - scale * g[c][j];
            }
        }
    }
    assert!(converged, "the likelihood fit hit its cap; the arms are not comparable");
    th
}

/// Emissions with `alpha` of the correction applied, over a night's pre-built design rows.
fn emissions_at(design: &[Vec<f64>], off: &[[f64; CLASSES]], th: &[Vec<f64>], alpha: f64)
    -> Vec<[f64; CLASSES]> {
    design
        .iter()
        .zip(off)
        .map(|(d, o)| {
            let mut out = *o;
            for c in 0..CLASSES {
                out[col_of(c)] += alpha * th[c].iter().zip(d).map(|(a, b)| a * b).sum::<f64>();
            }
            out
        })
        .collect()
}

/// A night's standardised design, kept beside its offsets so nothing re-standardises per arm.
struct Prepped<'a> {
    design: Vec<Vec<f64>>,
    off: &'a [[f64; CLASSES]],
    truth: &'a [Option<usize>],
}

fn prep<'a>(nights: &'a [Night], m: &[f64], sd: &[f64]) -> Vec<Prepped<'a>> {
    nights
        .iter()
        .map(|nt| Prepped {
            design: nt.row.iter().map(|r| design_row(r, m, sd, &[])).collect(),
            off: &nt.offset,
            truth: &nt.truth,
        })
        .collect()
}

/// The averaged structured perceptron: decode with the current correction, and on every epoch whose
/// decoded label disagrees with truth push the correction toward truth and away from the decode.
/// Returns the averaged correction after each pass, index 0 being theta = 0, which is v2.
fn fit_structured(train: &[&Prepped], beta: f64, floor: f64, rate: f64) -> Vec<Vec<Vec<f64>>> {
    let y: Vec<usize> = train.iter().flat_map(|p| p.truth.iter().flatten().copied()).collect();
    let cw = class_weights(&y);
    // Per LABELLED EPOCH, not per disagreeing epoch. Unnormalised, a pass accumulates tens of
    // thousands of updates and |theta| reaches 3.4 after one, which is a runaway and not a fit.
    let rate = rate / y.len().max(1) as f64;
    let t = tempered(beta, floor);
    let mut th = vec![vec![0.0f64; NCOL]; CLASSES];
    let mut acc = vec![vec![0.0f64; NCOL]; CLASSES];
    let mut out = vec![th.clone()];
    let mut seen = 0.0f64;
    for _ in 0..PASSES {
        for p in train {
            let em = emissions_at(&p.design, p.off, &th, 1.0);
            let path: Vec<usize> = decode_v2(&em, &t).iter().map(|s| stage_idx(*s)).collect();
            for (k, want) in p.truth.iter().enumerate() {
                let Some(want) = want else { continue };
                if path[k] == *want {
                    continue;
                }
                let step = rate * cw[*want];
                for (j, x) in p.design[k].iter().enumerate() {
                    th[*want][j] += step * x;
                    th[path[k]][j] -= step * x;
                }
            }
        }
        // Average across passes: a perceptron's last iterate chases the most recent night.
        seen += 1.0;
        for c in 0..CLASSES {
            for j in 0..NCOL {
                acc[c][j] += th[c][j];
            }
        }
        out.push((0..CLASSES).map(|c| acc[c].iter().map(|v| v / seen).collect()).collect());
    }
    out
}

/// Decoded kappa per night under one (correction, alpha, beta).
fn score(nights: &[Prepped], th: &[Vec<f64>], alpha: f64, beta: f64, floor: f64) -> Vec<f64> {
    let t = tempered(beta, floor);
    let mut out = Vec::new();
    for p in nights {
        let em = emissions_at(&p.design, p.off, th, alpha);
        let path: Vec<usize> = decode_v2(&em, &t).iter().map(|s| stage_idx(*s)).collect();
        let (mut pr, mut tr) = (Vec::new(), Vec::new());
        for (k, want) in p.truth.iter().enumerate() {
            if let Some(want) = want {
                pr.push(path[k]);
                tr.push(*want);
            }
        }
        if tr.len() >= MIN_EPOCHS {
            out.push(kappa4(&confusion4(&pr, &tr)));
        }
    }
    out
}

/// Runs per night and pooled per-class recall, for the arms worth explaining.
fn shape(nights: &[Prepped], th: &[Vec<f64>], alpha: f64, beta: f64, floor: f64) -> (f64, [f64; CLASSES]) {
    let t = tempered(beta, floor);
    let (mut runs, mut cm) = (Vec::new(), [[0i64; CLASSES]; CLASSES]);
    for p in nights {
        let em = emissions_at(&p.design, p.off, th, alpha);
        let path: Vec<usize> = decode_v2(&em, &t).iter().map(|s| stage_idx(*s)).collect();
        let (mut pr, mut tr) = (Vec::new(), Vec::new());
        for (k, want) in p.truth.iter().enumerate() {
            if let Some(want) = want {
                pr.push(path[k]);
                tr.push(*want);
            }
        }
        if tr.len() < MIN_EPOCHS {
            continue;
        }
        runs.push((1 + (1..pr.len()).filter(|k| pr[*k] != pr[k - 1]).count()) as f64);
        let c = confusion4(&pr, &tr);
        for i in 0..CLASSES {
            for j in 0..CLASSES {
                cm[i][j] += c[i][j];
            }
        }
    }
    let recall = std::array::from_fn(|c| {
        100.0 * cm[c][c] as f64 / cm[c].iter().sum::<i64>().max(1) as f64
    });
    (median(&mut runs), recall)
}

fn verdict(base: &[f64], arm: &[f64]) -> (f64, f64, String) {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
    let v = if !mean.is_finite() {
        "-".to_string()
    } else if mean.abs() > bar {
        format!("{} ({:.2}x)", if mean > 0.0 { "BEATS V2" } else { "worse" }, mean.abs() / bar)
    } else {
        "matches".to_string()
    };
    (mean, bar, v)
}

/// Standardiser and likelihood correction over the union of `keep`.
fn fit_on(loaded: &[(&str, Vec<Night>)], keep: &[&str]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let rows: Vec<(Vec<f64>, [f64; CLASSES], usize)> = loaded
        .iter()
        .filter(|(c, _)| keep.contains(c))
        .flat_map(|(_, n)| n.iter())
        .flat_map(|nt| {
            nt.row.iter().zip(&nt.offset).zip(&nt.truth).filter_map(|((r, o), t)| {
                t.map(|t| (r.clone(), *o, t))
            })
        })
        .collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|(r, _, _)| r.clone()).collect();
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    let off: Vec<[f64; CLASSES]> = rows.iter().map(|(_, o, _)| *o).collect();
    let y: Vec<usize> = rows.iter().map(|(_, _, t)| *t).collect();
    (fit_likelihood(&dx, &off, &y), m, sd)
}

/// What one arm chose on the inner folds, and what it scored on the outer one.
struct Outcome {
    label: String,
    chosen: String,
    kappa: f64,
    mean: f64,
    bar: f64,
    verdict: String,
}

/// One knob setting: how much of the likelihood correction, and how the prior is shaped.
#[derive(Clone, Copy)]
struct Knobs {
    alpha: f64,
    beta: f64,
    floor: f64,
}

/// Mean paired gain across the inner folds, against each fold's own v2. `own` uses the correction
/// fitted for that fold rather than a shared one, which is what an honest alpha search needs.
fn gain(inner: &[(Vec<Prepped>, Vec<Vec<f64>>)], k: Knobs, own: bool) -> f64 {
    let zero = vec![vec![0.0f64; NCOL]; CLASSES];
    inner
        .iter()
        .map(|(p, th)| {
            let base = score(p, &zero, 0.0, 1.0, 0.0);
            let arm = score(p, if own { th } else { &zero }, k.alpha, k.beta, k.floor);
            verdict(&base, &arm).0
        })
        .sum::<f64>()
        / inner.len() as f64
}

/// The best (beta, floor) over the grid, with alpha fixed. `own` as in [`gain`].
fn best_prior(inner: &[(Vec<Prepped>, Vec<Vec<f64>>)], alpha: f64, own: bool) -> Knobs {
    let mut best = (Knobs { alpha, beta: 1.0, floor: 0.0 }, f64::MIN);
    for beta in BETAS {
        for floor in FLOORS {
            let k = Knobs { alpha, beta, floor };
            let g = gain(inner, k, own);
            if g > best.1 {
                best = (k, g);
            }
        }
    }
    best.0
}

fn row(label: &str, chosen: String, v2: &[f64], arm: Vec<f64>) -> Outcome {
    let (mean, bar, verdict) = verdict(v2, &arm);
    Outcome { label: label.into(), chosen, kappa: median(&mut arm.clone()), mean, bar, verdict }
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    let zero = vec![vec![0.0f64; NCOL]; CLASSES];

    println!("Repairing the decode-side loss. Every knob is chosen by an inner leave-one-out over the");
    println!("TRAIN cohorts and applied once, so no arm selects against what it reports. Paired per");
    println!("night against V2 on the held-out cohort.\n");

    for (held, hn) in &loaded {
        let train: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        let (th, m, sd) = fit_on(&loaded, &train);
        let outer = prep(hn, &m, &sd);
        let v2 = score(&outer, &zero, 0.0, 1.0, 0.0);

        let inner: Vec<(Vec<Prepped>, Vec<Vec<f64>>)> = train
            .iter()
            .map(|c| {
                let sub: Vec<&str> = train.iter().copied().filter(|x| x != c).collect();
                let nights = &loaded.iter().find(|(n, _)| n == c).expect("inner cohort").1;
                (prep(nights, &m, &sd), fit_on(&loaded, &sub).0)
            })
            .collect();

        let mut out: Vec<Outcome> = Vec::new();

        // The control first. A win here is the shipped prior being wrong, and belongs to v2.
        let k = best_prior(&inner, 0.0, false);
        out.push(row("V2 + prior", format!("b={:.2} f={:.0e}", k.beta, k.floor), &v2,
                     score(&outer, &zero, 0.0, k.beta, k.floor)));
        let v2_prior = score(&outer, &zero, 0.0, k.beta, k.floor);

        out.push(row("likelihood a=1", "-".into(), &v2, score(&outer, &th, 1.0, 1.0, 0.0)));

        let mut bk = (Knobs { alpha: 0.0, beta: 1.0, floor: 0.0 }, f64::MIN);
        for alpha in ALPHAS {
            let k = best_prior(&inner, alpha, true);
            let g = gain(&inner, k, true);
            if g > bk.1 {
                bk = (k, g);
            }
        }
        let lk = bk.0;
        out.push(row("likelihood + prior",
                     format!("a={:.2} b={:.2} f={:.0e}", lk.alpha, lk.beta, lk.floor), &v2,
                     score(&outer, &th, lk.alpha, lk.beta, lk.floor)));

        // The structured fit, with its pass count and prior chosen inside.
        // The prior is resolved by the control above, so the structured arm sweeps its own step
        // size and pass count under that prior rather than re-searching a 28-cell grid.
        let mut bs = (0usize, 0.0f64, f64::MIN);
        for rate in P_RATES {
            let seq: Vec<Vec<Vec<Vec<f64>>>> = (0..inner.len())
                .map(|i| {
                    let others: Vec<&Prepped> = inner
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .flat_map(|(_, (p, _))| p.iter())
                        .collect();
                    fit_structured(&others, k.beta, k.floor, rate)
                })
                .collect();
            for pass in 0..=PASSES {
                let g: f64 = inner
                    .iter()
                    .zip(&seq)
                    .map(|((p, _), s)| {
                        let base = score(p, &zero, 0.0, 1.0, 0.0);
                        verdict(&base, &score(p, &s[pass], 1.0, k.beta, k.floor)).0
                    })
                    .sum::<f64>()
                    / inner.len() as f64;
                if g > bs.2 {
                    bs = (pass, rate, g);
                }
            }
        }
        let train_prepped: Vec<Prepped> = train
            .iter()
            .flat_map(|c| {
                let nights = &loaded.iter().find(|(n, _)| n == c).expect("train cohort").1;
                prep(nights, &m, &sd)
            })
            .collect();
        let tp: Vec<&Prepped> = train_prepped.iter().collect();
        let seq = fit_structured(&tp, k.beta, k.floor, bs.1);
        out.push(row("STRUCTURED", format!("p={} rate={} b={:.2}", bs.0, bs.1, k.beta), &v2,
                     score(&outer, &seq[bs.0], 1.0, k.beta, k.floor)));

        println!("== {held} n={} held out, v2 {:.3} ==", hn.len(), median(&mut v2.clone()));
        println!("  {:<20} {:<22} {:>7}   {:>10} {:>9}   verdict",
                 "arm", "chosen inside", "kappa", "paired d", "bar +/-");
        for o in &out {
            println!("  {:<20} {:<22} {:>7.3}   {:>+10.4} {:>9.4}   {}",
                     o.label, o.chosen, o.kappa, o.mean, o.bar, o.verdict);
        }
        let (r0, k0) = shape(&outer, &zero, 0.0, 1.0, 0.0);
        let (r1, k1) = shape(&outer, &zero, 0.0, k.beta, k.floor);
        println!("  V2 + prior reshapes the path: runs/night {r0:.0} -> {r1:.0};  recall {}",
                 (0..CLASSES)
                     .map(|c| format!("{} {:.0}->{:.0}", CLASS_NAMES[c], k0[c], k1[c]))
                     .collect::<Vec<_>>()
                     .join("  "));

        // Against the repaired prior rather than plain v2: does the correction add anything ON TOP?
        let (mm, bb, vv) = verdict(&v2_prior, &score(&outer, &seq[bs.0], 1.0, k.beta, k.floor));
        println!("  STRUCTURED measured against V2 + prior instead of v2: {mm:+.4} +/- {bb:.4}  {vv}");

        // Does the structured update MOVE anything, and at what step size? A chosen pass of 0 is a
        // real negative only if the other rates were tried and are worse.
        for rate in P_RATES {
            let probe = fit_structured(&tp, k.beta, k.floor, rate);
            print!("  structured rate {rate:>4}, paired vs v2:");
            for pass in [1usize, 4, 12, PASSES] {
                let mag: f64 = probe[pass].iter().flatten().map(|v| v.abs()).sum::<f64>()
                    / (CLASSES * NCOL) as f64;
                let d = verdict(&v2, &score(&outer, &probe[pass], 1.0, k.beta, k.floor)).0;
                print!("  p{pass} {d:+.4} (|th| {mag:.3})");
            }
            println!();
        }
        println!();
    }
    println!("Read V2 + prior first on every cohort: if reshaping the prior lifts v2 on its own, that");
    println!("is a finding about the shipped transition matrix and it belongs to v2, not to tanv1.");
}
