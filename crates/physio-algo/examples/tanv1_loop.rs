//! Step-check v2 against tanv1, take the single best repair, re-check, repeat until nothing helps.
//!
//!   cargo run --release -p physio-algo --example tanv1_loop
//!
//! `emission_steps` diagnosed one configuration and `decode_fix` tried two repairs by hand. This runs
//! that cycle automatically: measure, propose every single-knob move, take the best, measure again.
//!
//! The whole value of a loop like this is in what it CANNOT see. Every accept/reject decision is made
//! on an inner leave-one-out over the TRAIN cohorts only. The held-out cohort is scored and printed
//! every round and never consulted, so the gap between the inner gain and the held-out gain is a
//! direct read of how much the loop is fooling itself. A loop that selects on what it reports would
//! climb forever and mean nothing.
//!
//! The ladder, optimised one rung at a time and then swept again because the rungs interact:
//!   alpha[c]  how much of the correction class c gets. Step 9 found wake's correction costs kappa
//!             while deep's helps, so the classes want different amounts.
//!   beta      transition^beta. The only lever that lifted v2 on its own.
//!   floor     a floor under the transition, reaching the Wake->Deep and Wake->Rem hard zeros.
//!   drop      zero one design column of the correction. Step 8 found `bias` the single largest cost.

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
const L2: f64 = 1.0;
const ITERS: usize = 6_000;
const TOL: f64 = 1e-10;
const LR: f64 = 0.5;
const WEIGHT_POWER: f64 = 0.5;
/// Levels a single class's share of the correction may take.
const ALPHA_LEVELS: [f64; 6] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
const BETAS: [f64; 8] = [0.5, 0.7, 0.85, 1.0, 1.1, 1.25, 1.5, 1.75];
const FLOORS: [f64; 4] = [0.0, 1e-4, 1e-3, 1e-2];
/// Full passes over the ladder before giving up on convergence.
const MAX_SWEEPS: usize = 6;
/// Inner gain a move must add to be taken. Below this the loop is chasing its own noise.
const EPSILON: f64 = 2e-4;

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

fn col_names() -> Vec<String> {
    let mut v: Vec<String> = Features::NAMES.iter().map(|s| (*s).to_string()).collect();
    for n in ["nl_deep_hinge", "nl_dz_hr_var", "nl_dz_hr", "nl_turn_rank", "nl_resp_z", "nl_clamp"] {
        v.push(n.to_string());
    }
    v.push("bias".to_string());
    v
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

fn tempered(beta: f64, floor: f64) -> [[f64; CLASSES]; CLASSES] {
    Params::SHIPPED.transition.map(|row| row.map(|v| v.max(floor).powf(beta)))
}

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

/// Where the loop currently stands. `alpha` all-zero with `beta` 1 and no floor IS v2 exactly.
#[derive(Clone, PartialEq)]
struct State {
    alpha: [f64; CLASSES],
    beta: f64,
    floor: f64,
    dropped: Vec<usize>,
}

impl State {
    /// v2: no correction, the shipped prior untouched.
    fn v2() -> Self {
        State { alpha: [0.0; CLASSES], beta: 1.0, floor: 0.0, dropped: Vec::new() }
    }
    /// The likelihood correction as `fit_residual` applies it.
    fn tanv1() -> Self {
        State { alpha: [1.0; CLASSES], beta: 1.0, floor: 0.0, dropped: Vec::new() }
    }
    fn describe(&self) -> String {
        let a: Vec<String> = (0..CLASSES)
            .map(|c| format!("{}{:.2}", &CLASS_NAMES[c][..1], self.alpha[c]))
            .collect();
        format!("a[{}] b={:.2} f={:.0e} drop={}", a.join(" "), self.beta, self.floor,
                self.dropped.len())
    }
}

/// One night's emissions under `st`: the shipped emission plus each class's share of the correction.
fn emissions_at(p: &Prepped, th: &[Vec<f64>], st: &State) -> Vec<[f64; CLASSES]> {
    p.design
        .iter()
        .zip(p.off)
        .map(|(d, o)| {
            let mut out = *o;
            for c in 0..CLASSES {
                if st.alpha[c] == 0.0 {
                    continue;
                }
                let dot: f64 = th[c]
                    .iter()
                    .zip(d)
                    .enumerate()
                    .filter(|(j, _)| !st.dropped.contains(j))
                    .map(|(_, (a, b))| a * b)
                    .sum();
                out[col_of(c)] += st.alpha[c] * dot;
            }
            out
        })
        .collect()
}

fn decoded(p: &Prepped, th: &[Vec<f64>], st: &State) -> Vec<usize> {
    let em = emissions_at(p, th, st);
    decode_v2(&em, &tempered(st.beta, st.floor)).iter().map(|s| stage_idx(*s)).collect()
}

fn score(nights: &[Prepped], th: &[Vec<f64>], st: &State) -> Vec<f64> {
    let mut out = Vec::new();
    for p in nights {
        let path = decoded(p, th, st);
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

/// The quantity the loop maximises: mean paired gain over v2 across the inner folds, each fold using
/// the correction fitted WITHOUT it. This is the only thing an accept decision may read.
fn inner_gain(inner: &[(Vec<Prepped>, Vec<Vec<f64>>)], st: &State) -> f64 {
    let v2 = State::v2();
    inner
        .iter()
        .map(|(p, th)| verdict(&score(p, th, &v2), &score(p, th, st)).0)
        .sum::<f64>()
        / inner.len() as f64
}

/// One rung of the ladder, optimised to convergence before the next begins.
#[derive(Clone, Copy, PartialEq)]
enum Step {
    /// How much of the correction each class carries: the emission's magnitude.
    Alpha,
    /// Which design columns the correction may use: the emission's content.
    Prune,
    /// `transition^beta`, what the decoder charges for a stage change.
    Prior,
    /// The floor under the transition, reaching the hard zeros `powf` cannot.
    Floor,
}

impl Step {
    fn name(self) -> &'static str {
        match self {
            Step::Alpha => "1 alpha  emission magnitude",
            Step::Prune => "2 prune  emission content",
            Step::Prior => "3 prior  decode stickiness",
            Step::Floor => "4 floor  the hard zeros",
        }
    }
}

const LADDER: [Step; 4] = [Step::Alpha, Step::Prune, Step::Prior, Step::Floor];
/// Columns the prune rung may remove. Uncapped it strips the correction to nothing one column at a
/// time, which is alpha = 0 taking 35 moves to arrive.
const MAX_PRUNE: usize = 8;

/// The moves belonging to one rung.
fn moves(step: Step, st: &State, names: &[String]) -> Vec<(String, State)> {
    let mut out = Vec::new();
    match step {
        Step::Alpha => {
            for (c, name) in CLASS_NAMES.iter().enumerate() {
                for a in ALPHA_LEVELS {
                    if (st.alpha[c] - a).abs() < 1e-12 {
                        continue;
                    }
                    let mut n = st.clone();
                    n.alpha[c] = a;
                    out.push((format!("alpha[{name}] {:.2}->{a:.2}", st.alpha[c]), n));
                }
            }
        }
        Step::Prune => {
            if st.alpha.iter().all(|a| *a == 0.0) || st.dropped.len() >= MAX_PRUNE {
                return out;
            }
            for (j, name) in names.iter().enumerate() {
                if st.dropped.contains(&j) {
                    continue;
                }
                let mut n = st.clone();
                n.dropped.push(j);
                out.push((format!("drop {name}"), n));
            }
        }
        Step::Prior => {
            for b in BETAS {
                if (st.beta - b).abs() < 1e-12 {
                    continue;
                }
                let mut n = st.clone();
                n.beta = b;
                out.push((format!("beta {:.2}->{b:.2}", st.beta), n));
            }
        }
        Step::Floor => {
            for f in FLOORS {
                if (st.floor - f).abs() < 1e-12 {
                    continue;
                }
                let mut n = st.clone();
                n.floor = f;
                out.push((format!("floor {:.0e}->{f:.0e}", st.floor), n));
            }
        }
    }
    out
}

/// Kappa of the emission's own argmax, before the decoder. Reported for the emission rungs so a local
/// improvement is visible; never selected on, because selecting on it is exactly what tanv1 did.
fn argmax_score(nights: &[Prepped], th: &[Vec<f64>], st: &State) -> Vec<f64> {
    let mut out = Vec::new();
    for p in nights {
        let em = emissions_at(p, th, st);
        let (mut pr, mut tr) = (Vec::new(), Vec::new());
        for (k, want) in p.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            let mut best = (0usize, f64::NEG_INFINITY);
            for (col, v) in em[k].iter().enumerate() {
                if *v > best.1 {
                    best = (col, *v);
                }
            }
            pr.push(stage_idx(STAGE_ORDER[best.0]));
            tr.push(*want);
        }
        if tr.len() >= MIN_EPOCHS {
            out.push(kappa4(&confusion4(&pr, &tr)));
        }
    }
    out
}

/// Optimise one rung to convergence on the inner folds, returning the moves it took.
fn optimise(step: Step, st: &mut State, inner: &[(Vec<Prepped>, Vec<Vec<f64>>)], names: &[String],
            best: &mut f64) -> Vec<String> {
    let mut taken = Vec::new();
    loop {
        let mut pick: Option<(String, State, f64)> = None;
        for (label, cand) in moves(step, st, names) {
            let g = inner_gain(inner, &cand);
            if pick.as_ref().is_none_or(|(_, _, bg)| g > *bg) {
                pick = Some((label, cand, g));
            }
        }
        let Some((label, cand, g)) = pick else { return taken };
        if g <= *best + EPSILON {
            return taken;
        }
        taken.push(format!("{label} [{:+.4}]", g - *best));
        *best = g;
        *st = cand;
    }
}

/// Walk the ladder repeatedly until one whole sweep takes no move.
fn run_ladder(start: State, inner: &[(Vec<Prepped>, Vec<Vec<f64>>)], names: &[String],
              outer: &[Prepped], th: &[Vec<f64>], base: &[f64], loud: bool) -> State {
    let mut st = start;
    let mut best = inner_gain(inner, &st);
    let base_arg = argmax_score(outer, th, &State::v2());
    for sweep in 1..=MAX_SWEEPS {
        let before = st.clone();
        for step in LADDER {
            let taken = optimise(step, &mut st, inner, names, &mut best);
            if !loud {
                continue;
            }
            let (mm, _, _) = verdict(base, &score(outer, th, &st));
            let (am, _, _) = verdict(&base_arg, &argmax_score(outer, th, &st));
            let transfer = if best.abs() > 1e-9 { mm / best } else { f64::NAN };
            println!("  s{sweep} {:<26} inner {best:>+8.4}  held {mm:>+8.4}  transfer {transfer:>6.2}  argmax {am:>+8.4}",
                     step.name());
            for t in &taken {
                println!("          take {t}");
            }
        }
        if st == before {
            if loud {
                println!("  converged: sweep {sweep} took no move");
            }
            return st;
        }
    }
    if loud {
        println!("  stopped at the {MAX_SWEEPS}-sweep cap, so this is NOT a converged answer");
    }
    st
}

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

/// Runs per night, pooled per-class recall, and truth's own run count.
fn shape(nights: &[Prepped], th: &[Vec<f64>], st: &State) -> (f64, [f64; CLASSES], f64) {
    let (mut runs, mut truth_runs, mut cm) = (Vec::new(), Vec::new(), [[0i64; CLASSES]; CLASSES]);
    for p in nights {
        let path = decoded(p, th, st);
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
        truth_runs.push((1 + (1..tr.len()).filter(|k| tr[*k] != tr[k - 1]).count()) as f64);
        let c = confusion4(&pr, &tr);
        for i in 0..CLASSES {
            for j in 0..CLASSES {
                cm[i][j] += c[i][j];
            }
        }
    }
    let recall =
        std::array::from_fn(|c| 100.0 * cm[c][c] as f64 / cm[c].iter().sum::<i64>().max(1) as f64);
    (median(&mut runs), recall, median(&mut truth_runs))
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    let names = col_names();

    println!("Optimise each rung to convergence, move to the next, then sweep the ladder again until a");
    println!("whole pass takes no move. Every accept is decided on an inner leave-one-out over the TRAIN");
    println!("cohorts. `held` is printed and NEVER consulted, so `transfer` (held / inner) reads how much");
    println!("of each rung's gain is real. `argmax` is the emission before the decoder.\n");

    for (held, hn) in &loaded {
        let train: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        let (th, m, sd) = fit_on(&loaded, &train);
        let outer = prep(hn, &m, &sd);
        let inner: Vec<(Vec<Prepped>, Vec<Vec<f64>>)> = train
            .iter()
            .map(|c| {
                let sub: Vec<&str> = train.iter().copied().filter(|x| x != c).collect();
                let nights = &loaded.iter().find(|(n, _)| n == c).expect("inner cohort").1;
                (prep(nights, &m, &sd), fit_on(&loaded, &sub).0)
            })
            .collect();

        let base = score(&outer, &th, &State::v2());
        let (r0, k0, truth_runs) = shape(&outer, &th, &State::v2());
        let tan = score(&outer, &th, &State::tanv1());
        println!("== {held} n={} ==  v2 {:.3}, tanv1 as fitted {:.3} ({:+.4}), runs {r0:.0} vs truth {truth_runs:.0}",
                 hn.len(), median(&mut base.clone()), median(&mut tan.clone()),
                 verdict(&base, &tan).0);

        let end = run_ladder(State::tanv1(), &inner, &names, &outer, &th, &base, true);
        let arm = score(&outer, &th, &end);
        let (mm, bb, vv) = verdict(&base, &arm);
        let (r1, k1, _) = shape(&outer, &th, &end);
        println!("  FINAL {}", end.describe());
        println!("        kappa {:.3}, paired {mm:+.4} +/- {bb:.4}   {vv}", median(&mut arm.clone()));
        println!("        runs {r0:.0} -> {r1:.0} (truth {truth_runs:.0});  recall {}",
                 (0..CLASSES)
                     .map(|c| format!("{} {:.0}->{:.0}", CLASS_NAMES[c], k0[c], k1[c]))
                     .collect::<Vec<_>>()
                     .join("  "));

        // The same ladder from V2. Landing in the same place means the correction did no work.
        let end2 = run_ladder(State::v2(), &inner, &names, &outer, &th, &base, false);
        let arm2 = score(&outer, &th, &end2);
        let (m2, b2, v2v) = verdict(&base, &arm2);
        println!("  CONTROL, same ladder started from V2: {}", end2.describe());
        println!("        kappa {:.3}, paired {m2:+.4} +/- {b2:.4}   {v2v}", median(&mut arm2.clone()));
        let (dm, db, dv) = verdict(&arm2, &arm);
        println!("  tanv1 finish measured against THAT control: {dm:+.4} +/- {db:.4}   {dv}\n");
    }
    println!("Where `transfer` is far below 1, that rung bought inner kappa that does not exist on a");
    println!("cohort it has never seen. Where the CONTROL matches the tanv1 finish, the correction did");
    println!("no work and the prior did all of it.");
}

