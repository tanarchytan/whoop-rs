//! Walk the shipped emission to tanv1's one stage at a time, reporting at every stage.
//!
//!   cargo run --release -p physio-algo --example emission_steps
//!
//! `fit_residual` reports one number per cohort: sleep-accel loses 0.051 while the other two match.
//! One number cannot say WHERE it is lost. This runs the same fold and reports each stage between the
//! two arms' shared input and their decoded kappa, so the loss can be attributed to a stage:
//!
//!   0  the input is shared, and a zero correction IS the shipped recipe
//!   1  how large the correction is against the emission it moves
//!   2  the emission BEFORE the decoder - argmax kappa, which is the objective the fit optimises
//!   3  what the decoder then does to each arm, and whether the emission has outgrown the prior
//!   4  fragmentation against truth
//!   5  which class moved, pooled
//!   6  which nights moved
//!   7  dial the correction in from zero, on all three cohorts, so the curves can be compared
//!   8  which design column carries the damage
//!   9  which class's correction carries it
//!
//! Steps 2 and 3 are the pair that matters: a fit that improves the argmax and loses the decode is
//! the objective mismatch, and a fit that loses both is an emission that is simply worse here.

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
/// The cohort the step-by-step walk is about. The other two are fitted on, and appear only in step 7.
const HELD: &str = "sleep-accel";
const CLASSES: usize = 4;
const CLASS_NAMES: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
const MIN_EPOCHS: usize = 20;
/// The strength `fit_residual` reports its matching row at, so this walk is that row taken apart.
const L2: f64 = 1.0;
const ITERS: usize = 6_000;
const TOL: f64 = 1e-10;
const LR: f64 = 0.5;
const WEIGHT_POWER: f64 = 0.5;
/// How much of the correction to apply. 0 is the shipped recipe, 1 is what the fit chose.
const ALPHAS: [f64; 10] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6];
/// Columns to name in the step-8 table, by absolute effect.
const TOP_COLS: usize = 10;

struct Night {
    row: Vec<Vec<f64>>,
    /// The shipped emission per epoch, in [`STAGE_ORDER`] columns; the correction adds to this.
    offset: Vec<[f64; CLASSES]>,
    truth: Vec<Option<usize>>,
}

/// The measured features, then the shipped transforms, then the bias [`design_row`] appends.
fn col_names() -> Vec<String> {
    let mut v: Vec<String> = Features::NAMES.iter().map(|s| (*s).to_string()).collect();
    for n in ["nl_deep_hinge", "nl_dz_hr_var", "nl_dz_hr", "nl_turn_rank", "nl_resp_z", "nl_clamp"] {
        v.push(n.to_string());
    }
    v.push("bias".to_string());
    v
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
        assert!(f.len() >= em.len(), "{}: fewer feature rows than emissions", dir.display());
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

/// Our class index -> the emission's column.
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

/// `fit_residual`'s optimiser verbatim: a correction on top of a fixed per-class offset, starting at
/// zero. Reproduced rather than shared so this walk cannot report a different fit than that harness.
fn fit_residual(x: &[Vec<f64>], off: &[[f64; CLASSES]], y: &[usize], l2: f64) -> (Vec<Vec<f64>>, bool) {
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
        let decay = (1.0 - LR * l2).max(0.0);
        for c in 0..CLASSES {
            for j in 0..p {
                th[c][j] = decay * th[c][j] - scale * g[c][j];
            }
        }
    }
    (th, converged)
}

/// One night's emission with `alpha` of the correction applied.
fn emissions_at(nt: &Night, th: &[Vec<f64>], m: &[f64], sd: &[f64], alpha: f64) -> Vec<[f64; CLASSES]> {
    nt.row
        .iter()
        .zip(&nt.offset)
        .map(|(full, o)| {
            let d = design_row(full, m, sd, &[]);
            let mut out = *o;
            for c in 0..CLASSES {
                out[col_of(c)] += alpha * th[c].iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
            }
            out
        })
        .collect()
}

/// Per-epoch argmax of an emission, as our class index.
fn argmax_path(em: &[[f64; CLASSES]]) -> Vec<usize> {
    em.iter()
        .map(|row| {
            let mut best = (0usize, f64::NEG_INFINITY);
            for (col, v) in row.iter().enumerate() {
                if *v > best.1 {
                    best = (col, *v);
                }
            }
            stage_idx(STAGE_ORDER[best.0])
        })
        .collect()
}

/// Top-two gap of an emission row: how much evidence the decoder's prior has to overturn.
fn margin(row: &[f64; CLASSES]) -> f64 {
    let mut v = row.to_vec();
    v.sort_by(f64::total_cmp);
    v[CLASSES - 1] - v[CLASSES - 2]
}

/// Stage runs in a label sequence, which is one more than the number of changes.
fn runs(path: &[usize]) -> usize {
    1 + (1..path.len()).filter(|k| path[*k] != path[k - 1]).count()
}

#[derive(Default)]
struct Report {
    /// Decoded kappa per night, and the same before the decoder.
    decoded: Vec<f64>,
    argmax: Vec<f64>,
    cm_decoded: [[i64; CLASSES]; CLASSES],
    cm_argmax: [[i64; CLASSES]; CLASSES],
    /// Runs per night, decoded and in truth.
    runs_decoded: Vec<f64>,
    runs_truth: Vec<f64>,
    margins: Vec<f64>,
    /// Epochs the decoder moved away from the emission's own argmax, and epochs seen.
    prior_moved: usize,
    epochs: usize,
    /// Per-class magnitude of the correction, and the shipped emission's own spread.
    delta: [Vec<f64>; CLASSES],
    spread: Vec<f64>,
}

fn evaluate(nights: &[Night], th: &[Vec<f64>], m: &[f64], sd: &[f64], alpha: f64) -> Report {
    let mut r = Report::default();
    for nt in nights {
        let em = emissions_at(nt, th, m, sd, alpha);
        let path: Vec<usize> =
            decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
        let amax = argmax_path(&em);
        for (e, row) in em.iter().enumerate() {
            r.margins.push(margin(row));
            r.prior_moved += usize::from(path[e] != amax[e]);
            r.epochs += 1;
            for c in 0..CLASSES {
                r.delta[c].push((row[col_of(c)] - nt.offset[e][col_of(c)]).abs());
            }
            let mut v = nt.offset[e].to_vec();
            v.sort_by(f64::total_cmp);
            r.spread.push(v[CLASSES - 1] - v[0]);
        }
        let (mut dp, mut ap, mut t) = (Vec::new(), Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            dp.push(path[k]);
            ap.push(amax[k]);
            t.push(*want);
        }
        if t.len() < MIN_EPOCHS {
            continue;
        }
        let (cd, ca) = (confusion4(&dp, &t), confusion4(&ap, &t));
        r.decoded.push(kappa4(&cd));
        r.argmax.push(kappa4(&ca));
        for i in 0..CLASSES {
            for j in 0..CLASSES {
                r.cm_decoded[i][j] += cd[i][j];
                r.cm_argmax[i][j] += ca[i][j];
            }
        }
        r.runs_decoded.push(runs(&dp) as f64);
        r.runs_truth.push(runs(&t) as f64);
    }
    r
}

/// A copy of `th` with one design column, or one class's whole correction, zeroed.
fn masked(th: &[Vec<f64>], drop_col: Option<usize>, drop_class: Option<usize>) -> Vec<Vec<f64>> {
    let mut out = th.to_vec();
    for (c, row) in out.iter_mut().enumerate() {
        if drop_class == Some(c) {
            row.iter_mut().for_each(|v| *v = 0.0);
        }
        if let Some(j) = drop_col {
            row[j] = 0.0;
        }
    }
    out
}

/// Recall and precision per class off a pooled confusion, rows = truth.
fn per_class(cm: &[[i64; CLASSES]; CLASSES]) -> [(f64, f64); CLASSES] {
    std::array::from_fn(|c| {
        let hit = cm[c][c] as f64;
        let t: f64 = cm[c].iter().sum::<i64>() as f64;
        let p: f64 = (0..CLASSES).map(|r| cm[r][c]).sum::<i64>() as f64;
        (100.0 * hit / t.max(1.0), 100.0 * hit / p.max(1.0))
    })
}

/// The paired difference of two per-night series against its own 95% bar.
fn verdict(base: &[f64], arm: &[f64]) -> (f64, f64, String) {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let (mean, bar) = paired_bar(&d).unwrap_or((f64::NAN, f64::NAN));
    let v = if !mean.is_finite() {
        "-".to_string()
    } else if mean.abs() > bar {
        format!("{} ({:.2}x)", if mean > 0.0 { "BEATS" } else { "worse" }, mean.abs() / bar)
    } else {
        "matches".to_string()
    };
    (mean, bar, v)
}

/// Fit the union of every cohort except `held`, and return the correction with its standardiser.
fn fold(loaded: &[(&str, Vec<Night>)], held: &str) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let keep: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| *c != held).collect();
    fit_on(loaded, &keep, held)
}

/// Fit the union of the named cohorts. `what` names the fit in the convergence guard only.
fn fit_on(loaded: &[(&str, Vec<Night>)], keep: &[&str], what: &str) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
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
    let (th, conv) = fit_residual(&dx, &off, &y, L2);
    assert!(conv, "the {what} fold did not converge; its arms are not comparable");
    (th, m, sd)
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 2 {
        println!("need at least two cohorts under the fixture root");
        return;
    }
    let Some(nights) = loaded.iter().find(|(c, _)| *c == HELD).map(|(_, n)| n) else {
        println!("{HELD} is not under the fixture root");
        return;
    };
    let names = col_names();
    let (th, m, sd) = fold(&loaded, HELD);
    let base = evaluate(nights, &th, &m, &sd, 0.0);
    let full = evaluate(nights, &th, &m, &sd, 1.0);

    println!("Step by step from the shipped emission to tanv1's, on {HELD} (n={}), held out of a fit",
             nights.len());
    println!("over the other two cohorts at L2 {L2:.2}. Every stage is the SAME two arms.\n");

    // ---- 0 ------------------------------------------------------------------------------------
    println!("STEP 0  the arms share their input");
    let epochs: usize = nights.iter().map(|n| n.offset.len()).sum();
    let labelled: usize = nights.iter().flat_map(|n| &n.truth).filter(|t| t.is_some()).count();
    println!("  {epochs} epochs, {labelled} of them labelled, one prepared night per arm");
    println!("  a zero correction reproduces the shipped decode: {}",
             if base.decoded == evaluate(nights, &vec![vec![0.0; names.len()]; CLASSES], &m, &sd, 1.0).decoded {
                 "YES"
             } else {
                 "NO - the offset is not the shipped emission"
             });

    // ---- 1 ------------------------------------------------------------------------------------
    println!("\nSTEP 1  how large the correction is");
    println!("  the shipped emission spans {:.3} log units between its best and worst class (median)",
             median(&mut full.spread.clone()));
    for (c, name) in CLASS_NAMES.iter().enumerate() {
        let mut d = full.delta[c].clone();
        println!("    {name:<6} correction |d| median {:.3}, p90 {:.3}", median(&mut d), {
            d.sort_by(f64::total_cmp);
            d[(d.len() as f64 * 0.9) as usize]
        });
    }

    // ---- 2 ------------------------------------------------------------------------------------
    println!("\nSTEP 2  the emission BEFORE the decoder (argmax, which is what the fit optimises)");
    let (am, ab, av) = verdict(&base.argmax, &full.argmax);
    println!("  shipped {:.3}   corrected {:.3}   paired {am:+.4} +/- {ab:.4}   {av}",
             median(&mut base.argmax.clone()), median(&mut full.argmax.clone()));

    // ---- 3 ------------------------------------------------------------------------------------
    println!("\nSTEP 3  what the decoder then does");
    let (dm, db, dv) = verdict(&base.decoded, &full.decoded);
    println!("  shipped {:.3}   corrected {:.3}   paired {dm:+.4} +/- {db:.4}   {dv}",
             median(&mut base.decoded.clone()), median(&mut full.decoded.clone()));
    println!("  the decoder overrides the emission's own argmax on {:.1}% of epochs shipped, {:.1}% corrected",
             100.0 * base.prior_moved as f64 / base.epochs as f64,
             100.0 * full.prior_moved as f64 / full.epochs as f64);
    let switch: Vec<f64> = (0..CLASSES)
        .map(|f| {
            let t = &Params::SHIPPED.transition[f];
            let stay = t[f].max(1e-9).ln();
            let go = (0..CLASSES).filter(|x| *x != f).map(|x| t[x].max(1e-9).ln()).fold(f64::MIN, f64::max);
            stay - go
        })
        .collect();
    let cheapest = switch.iter().cloned().fold(f64::MAX, f64::min);
    println!("  the prior charges {:.2} log units for the cheapest stage change ({:.2} for the dearest)",
             cheapest, switch.iter().cloned().fold(f64::MIN, f64::max));
    for (tag, r) in [("shipped", &base), ("corrected", &full)] {
        let over = r.margins.iter().filter(|v| **v > cheapest).count();
        println!("    {tag:<10} emission margin median {:.2}, and {:.1}% of epochs already exceed that charge",
                 median(&mut r.margins.clone()), 100.0 * over as f64 / r.margins.len() as f64);
    }

    // ---- 4 ------------------------------------------------------------------------------------
    println!("\nSTEP 4  fragmentation");
    println!("  stage runs per night: truth {:.0}, shipped {:.0}, corrected {:.0}",
             median(&mut base.runs_truth.clone()), median(&mut base.runs_decoded.clone()),
             median(&mut full.runs_decoded.clone()));

    // ---- 5 ------------------------------------------------------------------------------------
    println!("\nSTEP 5  which class moved (pooled, rows = truth)");
    let (pb, pf) = (per_class(&base.cm_decoded), per_class(&full.cm_decoded));
    let (ab_, af) = (per_class(&base.cm_argmax), per_class(&full.cm_argmax));
    println!("  {:<6} {:>8} {:>10} {:>10}   {:>8} {:>10} {:>10}",
             "class", "recall", "corrected", "delta", "precision", "corrected", "delta");
    for c in 0..CLASSES {
        println!("  {:<6} {:>8.1} {:>10.1} {:>+10.1}   {:>8.1} {:>10.1} {:>+10.1}",
                 CLASS_NAMES[c], pb[c].0, pf[c].0, pf[c].0 - pb[c].0,
                 pb[c].1, pf[c].1, pf[c].1 - pb[c].1);
    }
    println!("  the same recall before the decoder, so a class the decode rescues is visible:");
    for c in 0..CLASSES {
        println!("    {:<6} argmax recall {:>6.1} -> {:>6.1} ({:+.1}), decoded {:+.1}",
                 CLASS_NAMES[c], ab_[c].0, af[c].0, af[c].0 - ab_[c].0, pf[c].0 - pb[c].0);
    }

    // ---- 6 ------------------------------------------------------------------------------------
    println!("\nSTEP 6  which nights moved");
    let mut d: Vec<f64> = base.decoded.iter().zip(&full.decoded).map(|(a, b)| b - a).collect();
    let (worse, better) = (d.iter().filter(|v| **v < -1e-9).count(), d.iter().filter(|v| **v > 1e-9).count());
    println!("  {worse} nights worse, {better} better, {} unchanged of {}",
             d.len() - worse - better, d.len());
    d.sort_by(f64::total_cmp);
    println!("  worst three {:+.3} {:+.3} {:+.3}, best three {:+.3} {:+.3} {:+.3}",
             d[0], d[1], d[2], d[d.len() - 1], d[d.len() - 2], d[d.len() - 3]);
    let trimmed: Vec<f64> = d[3..].to_vec();
    let (tm, tb) = paired_bar(&trimmed).unwrap_or((f64::NAN, f64::NAN));
    println!("  dropping the three worst nights leaves {tm:+.4} +/- {tb:.4}, so the loss is {} \
              a few nights",
             if tm.abs() > tb { "NOT carried by" } else { "carried by" });

    // ---- 7 ------------------------------------------------------------------------------------
    println!("\nSTEP 7  dial the correction in, each cohort on its own fold");
    println!("  {:<8}", "alpha");
    for (held, hn) in &loaded {
        let (t2, m2, s2) = if *held == HELD { (th.clone(), m.clone(), sd.clone()) } else { fold(&loaded, held) };
        let mut line = format!("  {:<14}", format!("{held} n={}", hn.len()));
        let zero = evaluate(hn, &t2, &m2, &s2, 0.0);
        for a in ALPHAS {
            let r = evaluate(hn, &t2, &m2, &s2, a);
            let (mm, bb, _) = verdict(&zero.decoded, &r.decoded);
            let flag = if a == 0.0 || mm.abs() <= bb { ' ' } else if mm > 0.0 { '+' } else { '-' };
            line.push_str(&format!(" {:>6.3}{flag}", median(&mut r.decoded.clone())));
        }
        println!("{line}");
    }
    print!("  {:<14}", "alpha");
    for a in ALPHAS {
        print!(" {a:>6.2} ");
    }
    println!("\n  a trailing - is resolvably worse than that cohort's own alpha=0, + resolvably better");

    // ---- 8 ------------------------------------------------------------------------------------
    println!("\nSTEP 8  which design column carries the damage on {HELD}");
    let mut effect: Vec<(f64, usize)> = (0..names.len())
        .map(|j| {
            let r = evaluate(nights, &masked(&th, Some(j), None), &m, &sd, 1.0);
            let (mm, _, _) = verdict(&base.decoded, &r.decoded);
            (mm - dm, j)
        })
        .collect();
    effect.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("  each row zeroes ONE column of the correction and re-decodes; a positive recovery means");
    println!("  that column was costing {HELD} kappa. Full correction sits at {dm:+.4}.");
    println!("  {:<18} {:>10} {:>12}", "column", "recovery", "paired d");
    for (rec, j) in effect.iter().take(TOP_COLS) {
        let r = evaluate(nights, &masked(&th, Some(*j), None), &m, &sd, 1.0);
        let (mm, _, v) = verdict(&base.decoded, &r.decoded);
        println!("  {:<18} {rec:>+10.4} {mm:>+12.4}   {v}", names[*j]);
    }

    // ---- 9 ------------------------------------------------------------------------------------
    println!("\nSTEP 9  which class's correction carries it");
    println!("  {:<18} {:>12}   verdict", "class zeroed", "paired d");
    for (c, name) in CLASS_NAMES.iter().enumerate() {
        let r = evaluate(nights, &masked(&th, None, Some(c)), &m, &sd, 1.0);
        let (mm, _, v) = verdict(&base.decoded, &r.decoded);
        println!("  {name:<18} {mm:>+12.4}   {v}");
    }
    // ---- 10 -----------------------------------------------------------------------------------
    println!("\nSTEP 10  choose alpha WITHOUT the reported cohort");
    println!("  step 7's peak was read off the cohort being reported, which is selection. Here alpha");
    println!("  is chosen by an inner leave-one-out over the two TRAIN cohorts and then applied once.");
    println!("  {:<14} {:>6} {:>9} {:>10}   {:>10} {:>9}   verdict",
             "held-out", "alpha", "shipped", "corrected", "paired d", "bar +/-");
    for (held, hn) in &loaded {
        let train: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        // Inner fold: fit one train cohort, score the other, and average the paired gain per alpha.
        let mut gain = [0.0f64; ALPHAS.len()];
        for inner in &train {
            let sub: Vec<&str> = train.iter().copied().filter(|c| c != inner).collect();
            let (it, im, isd) = fit_on(&loaded, &sub, inner);
            let iv = &loaded.iter().find(|(c, _)| c == inner).expect("inner cohort").1;
            let zero = evaluate(iv, &it, &im, &isd, 0.0);
            for (k, a) in ALPHAS.iter().enumerate() {
                let r = evaluate(iv, &it, &im, &isd, *a);
                gain[k] += verdict(&zero.decoded, &r.decoded).0 / train.len() as f64;
            }
        }
        let best = (0..ALPHAS.len()).max_by(|a, b| gain[*a].total_cmp(&gain[*b])).expect("alphas");
        let (t2, m2, s2) = fold(&loaded, held);
        let zero = evaluate(hn, &t2, &m2, &s2, 0.0);
        let r = evaluate(hn, &t2, &m2, &s2, ALPHAS[best]);
        let (mm, bb, v) = verdict(&zero.decoded, &r.decoded);
        println!("  {:<14} {:>6.2} {:>9.3} {:>10.3}   {mm:>+10.4} {bb:>9.4}   {v}",
                 format!("{held} n={}", hn.len()), ALPHAS[best],
                 median(&mut zero.decoded.clone()), median(&mut r.decoded.clone()));
    }
    println!("  the inner fits see ONE cohort each, which is the confound `fit_loco` removed, so this");
    println!("  selects a magnitude honestly and nothing else.");

    println!("\nRead steps 2 and 3 together: an argmax that improves while the decode loses is the");
    println!("objective mismatch, and both losing is an emission that is worse on this cohort.");
}
