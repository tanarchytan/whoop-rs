//! Per-step parity: is tanv1's version of each step as good as v2's version, on that step's own terms?
//!
//!   cargo run --release -p physio-algo --example tanv1_steps
//!
//! Everything before this treated tanv1 as a CORRECTION on v2's emission (`em = em_v2 + theta.x`).
//! That construction guarantees parity at theta = 0, which is why it was built, but it structurally
//! makes tanv1 "v2 plus a patch" and it cannot become a different model. It also produced at least one
//! artefact: tanv1's motion column separates wake BETTER than v2's on a weight-free rank AUC and still
//! decoded worse, because it was forced through a coefficient tuned against a different degree of
//! zero-inflation. Given its own weight that is not a defect.
//!
//! So: no offset anywhere here. Each arm builds its OWN four-class emission from its OWN features with
//! its OWN fitted weights, and the ladder is walked one rung at a time:
//!
//!   A  feature information   per-epoch argmax kappa of a fitted readout, no decoder, no priors
//!   B  + the structured time-of-night prior, which v2 has and tanv1 currently does not
//!   C  + the decoder, which is the shipped Viterbi for every arm
//!   D  + the minimum-dwell floor, which is emission-blind so every arm gets it
//!
//! A rung is only worth crossing once tanv1 is at or above v2 on the rung below it. Weights and
//! finetuning come after, not before.

mod common;

use common::lr::{design_row, fit, scores, standardise_cols};
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
const WEIGHT_POWER: f64 = 0.5;
/// v2's own scalar features, before its weights: the three z-scores, the rectified flatness gate,
/// the centred rotation rank and the respiration z. Six numbers per epoch.
const N_V2: usize = 6;
const V2_NAMES: [&str; N_V2] = ["hr_z", "hr_var_z", "motion_z", "deep_hinge", "turn_rank", "resp_z"];
const N_TAN: usize = Features::N;
/// Shortest run a stage may hold after the dwell floor. Emission-blind, so every arm gets it.
const MIN_DWELL: [usize; CLASSES] = [1, 1, 6, 6];

struct Night {
    /// v2's six scalars per epoch.
    v2: Vec<[f64; N_V2]>,
    /// tanv1's measured columns per epoch.
    tan: Vec<Vec<f64>>,
    /// v2's structured time-of-night prior plus its motion gate, per class, in STAGE_ORDER columns.
    cycle: Vec<[f64; CLASSES]>,
    /// The shipped emission, for the reference arm only.
    shipped: Vec<[f64; CLASSES]>,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    let blp: [f64; CLASSES] = std::array::from_fn(|c| Params::SHIPPED.base_rate[c].ln());
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
        let v2 = (0..em.len())
            .map(|e| {
                let d = &terms.design[e];
                [d[deep][1], d[deep][0], d[deep][2], -d[deep][3], d[awake][10], d[deep][11]]
            })
            .collect();
        // `fixed` is the base prior plus the cycle prior plus the motion gate; subtracting the base
        // prior leaves the structured time-of-night term this ladder tests tanv1 against.
        let cycle = (0..em.len())
            .map(|e| std::array::from_fn(|c| terms.fixed[e][c] - blp[c]))
            .collect();
        let tan = (0..em.len()).map(|e| f[e].values().to_vec()).collect();
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { v2, tan, cycle, shipped: em[..].to_vec(), truth });
    }
    out
}

/// Which feature set an arm reads.
#[derive(Clone, Copy, PartialEq)]
enum Cols {
    V2,
    Tan,
}

impl Cols {
    fn row(self, nt: &Night, e: usize) -> Vec<f64> {
        match self {
            Cols::V2 => nt.v2[e].to_vec(),
            Cols::Tan => nt.tan[e].clone(),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Cols::V2 => "v2 six",
            Cols::Tan => "tanv1",
        }
    }
}

/// One inner fold: its readout, the transition fitted without it, and its nights.
type InnerFold<'a> = (Readout, [[f64; CLASSES]; CLASSES], &'a Vec<Night>);

/// A fitted four-class readout over one feature set, plus the standardiser it was fitted under.
struct Readout {
    w: Vec<Vec<f64>>,
    m: Vec<f64>,
    sd: Vec<f64>,
    cols: Cols,
}

fn train(nights: &[&Night], cols: Cols) -> Readout {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for nt in nights {
        for (e, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                x.push(cols.row(nt, e));
                y.push(*t);
            }
        }
    }
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    Readout { w: fit(&dx, &y, WEIGHT_POWER), m, sd, cols }
}

impl Readout {
    /// One night's emission in [`STAGE_ORDER`] columns, optionally with v2's structured prior added.
    fn emission(&self, nt: &Night, with_cycle: bool) -> Vec<[f64; CLASSES]> {
        (0..nt.truth.len())
            .map(|e| {
                let z = scores(&self.w, &design_row(&self.cols.row(nt, e), &self.m, &self.sd, &[]));
                std::array::from_fn(|c| {
                    z[stage_idx(STAGE_ORDER[c])] + if with_cycle { nt.cycle[e][c] } else { 0.0 }
                })
            })
            .collect()
    }
}

fn argmax(em: &[[f64; CLASSES]]) -> Vec<usize> {
    em.iter()
        .map(|row| {
            let mut best = (0usize, f64::NEG_INFINITY);
            for (c, v) in row.iter().enumerate() {
                if *v > best.1 {
                    best = (c, *v);
                }
            }
            stage_idx(STAGE_ORDER[best.0])
        })
        .collect()
}

/// Collapse runs shorter than that stage's floor into the neighbour the run came from.
fn dwell_floor(path: &[usize]) -> Vec<usize> {
    let mut out = path.to_vec();
    let mut i = 0;
    while i < out.len() {
        let mut j = i;
        while j + 1 < out.len() && out[j + 1] == out[i] {
            j += 1;
        }
        if j + 1 - i < MIN_DWELL[out[i]] && i > 0 {
            let fill = out[i - 1];
            out[i..=j].fill(fill);
            // Re-walk from the merged run so a cascade settles.
            i = i.saturating_sub(1);
            continue;
        }
        i = j + 1;
    }
    out
}

/// Per-night kappa of one label sequence against truth.
fn kappa_of(path: &[usize], truth: &[Option<usize>]) -> Option<f64> {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate() {
        if let Some(want) = want {
            p.push(path[k]);
            t.push(*want);
        }
    }
    (t.len() >= MIN_EPOCHS).then(|| kappa4(&confusion4(&p, &t)))
}

/// One rung of the ladder for one arm.
/// Maximum-likelihood transition counted off TRAIN truth, Laplace-smoothed, in [`STAGE_ORDER`]
/// columns. tanv1 has never had one: every arm so far decoded with v2's, which was tuned against
/// v2's emission scale and margin distribution.
fn fit_transition(nights: &[&Night]) -> [[f64; CLASSES]; CLASSES] {
    let mut c = [[1.0f64; CLASSES]; CLASSES];
    for nt in nights {
        let lab: Vec<Option<usize>> = nt.truth.clone();
        for k in 1..lab.len() {
            if let (Some(a), Some(b)) = (lab[k - 1], lab[k]) {
                c[col_order(a)][col_order(b)] += 1.0;
            }
        }
    }
    for row in c.iter_mut() {
        let s: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s;
        }
    }
    c
}

/// Our class index -> its [`STAGE_ORDER`] column.
fn col_order(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

/// How an arm decodes: whose transition, and how hard the emission pushes against it.
#[derive(Clone, Copy)]
struct Decode {
    /// `None` uses the shipped transition.
    own: Option<[[f64; CLASSES]; CLASSES]>,
    /// Multiplier on the emission. The fitted score has no natural scale against `log(transition)`.
    gamma: f64,
}

fn rung_decode(nights: &[Night], r: &Readout, with_cycle: bool, d: Decode, floor: bool) -> Vec<f64> {
    let t = d.own.unwrap_or(Params::SHIPPED.transition);
    nights
        .iter()
        .filter_map(|nt| {
            let em: Vec<[f64; CLASSES]> = r
                .emission(nt, with_cycle)
                .iter()
                .map(|row| std::array::from_fn(|c| d.gamma * row[c]))
                .collect();
            let mut path: Vec<usize> =
                decode_v2(&em, &t).iter().map(|s| stage_idx(*s)).collect();
            if floor {
                path = dwell_floor(&path);
            }
            kappa_of(&path, &nt.truth)
        })
        .collect()
}

fn rung(nights: &[Night], r: &Readout, with_cycle: bool, decode: bool, floor: bool) -> Vec<f64> {
    nights
        .iter()
        .filter_map(|nt| {
            let em = r.emission(nt, with_cycle);
            let mut path = if decode {
                decode_v2(&em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect()
            } else {
                argmax(&em)
            };
            if floor {
                path = dwell_floor(&path);
            }
            kappa_of(&path, &nt.truth)
        })
        .collect()
}

/// The shipped recipe walked up the same rungs, as the reference every arm is measured against.
fn rung_shipped(nights: &[Night], decode: bool, floor: bool) -> Vec<f64> {
    nights
        .iter()
        .filter_map(|nt| {
            let mut path = if decode {
                decode_v2(&nt.shipped, &Params::SHIPPED.transition)
                    .iter()
                    .map(|s| stage_idx(*s))
                    .collect()
            } else {
                argmax(&nt.shipped)
            };
            if floor {
                path = dwell_floor(&path);
            }
            kappa_of(&path, &nt.truth)
        })
        .collect()
}

fn verdict(base: &[f64], arm: &[f64]) -> String {
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let Some((mean, bar)) = paired_bar(&d) else { return "-".to_string() };
    let tag = if mean.abs() <= bar {
        "matches".to_string()
    } else {
        format!("{} ({:.2}x)", if mean > 0.0 { "AHEAD" } else { "behind" }, mean.abs() / bar)
    };
    format!("{mean:>+8.4} {bar:>7.4}  {tag}")
}

/// Weight-free one-vs-rest rank AUC of one column for one class, so a feature can be judged before
/// any weight touches it. Ties take half credit.
fn auc(vals: &[f64], lab: &[usize], class: usize) -> f64 {
    let mut idx: Vec<usize> = (0..vals.len()).filter(|k| vals[*k].is_finite()).collect();
    idx.sort_by(|a, b| vals[*a].total_cmp(&vals[*b]));
    let (mut pos, mut neg, mut rank_sum) = (0.0f64, 0.0f64, 0.0f64);
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && vals[idx[j + 1]] == vals[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if lab[idx[k]] == class {
                rank_sum += avg;
                pos += 1.0;
            } else {
                neg += 1.0;
            }
        }
        i = j + 1;
    }
    if pos == 0.0 || neg == 0.0 {
        return f64::NAN;
    }
    (rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }

    println!("Per-step parity, tanv1 against v2, each side with its OWN features and its OWN fitted");
    println!("weights. No offset: nothing here is v2 plus a correction. A rung is only worth crossing");
    println!("once tanv1 is at or above v2 on the rung below.\n");

    // Weight-free, before any weight: how well does each side's best column separate each class?
    println!("=== Feature information, no weights, pooled one-vs-rest rank AUC (0.5 is useless) ===");
    println!("  {:<14} {:<10} {:>7}  best column", "cohort", "class", "AUC");
    for (name, nights) in &loaded {
        let mut lab = Vec::new();
        let (mut v2c, mut tanc) = (vec![Vec::new(); N_V2], vec![Vec::new(); N_TAN]);
        for nt in nights {
            for (e, t) in nt.truth.iter().enumerate() {
                let Some(t) = t else { continue };
                lab.push(*t);
                for (j, s) in v2c.iter_mut().enumerate() {
                    s.push(nt.v2[e][j]);
                }
                for (j, s) in tanc.iter_mut().enumerate() {
                    s.push(nt.tan[e][j]);
                }
            }
        }
        for (c, cname) in CLASS_NAMES.iter().enumerate() {
            let pick = |cols: &[Vec<f64>], names: &dyn Fn(usize) -> String| {
                let mut best = (f64::NAN, String::new());
                for (j, s) in cols.iter().enumerate() {
                    let a = auc(s, &lab, c);
                    let a = if a.is_finite() { (a - 0.5).abs() + 0.5 } else { f64::NAN };
                    if a.is_finite() && !best.0.is_finite() || a > best.0 {
                        best = (a, names(j));
                    }
                }
                best
            };
            let bv = pick(&v2c, &|j| V2_NAMES[j].to_string());
            let bt = pick(&tanc, &|j| Features::NAMES[j].to_string());
            let flag = if bt.0 > bv.0 { "tanv1 ahead" } else { "v2 ahead" };
            println!("  {:<14} {:<10} v2 {:>5.3} {:<16}  tanv1 {:>5.3} {:<16}  {flag}",
                     name, cname, bv.0, bv.1, bt.0, bt.1);
        }
    }

    println!("\n=== The ladder, leave one cohort out, paired per night against the SHIPPED recipe ===");
    println!("  A emission only (argmax, no prior, no decoder)   B + v2's structured time-of-night prior");
    println!("  C + the shipped Viterbi                          D + the minimum-dwell floor\n");
    for (held, hn) in &loaded {
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        let (rv2, rtan) = (train(&tr, Cols::V2), train(&tr, Cols::Tan));
        println!("-- {held} n={} --", hn.len());
        println!("  {:<4} {:<8} {:>7}   {:>8} {:>7}  verdict vs shipped", "rung", "arm", "kappa",
                 "paired d", "bar");
        for (tag, cyc, dec, flr) in
            [("A", false, false, false), ("B", true, false, false), ("C", true, true, false),
             ("D", true, true, true)]
        {
            let base = rung_shipped(hn, dec, flr);
            println!("  {tag:<4} {:<8} {:>7.3}   {}", "shipped", median(&mut base.clone()),
                     if dec { "the reference" } else { "the reference (argmax of the shipped emission)" });
            for r in [&rv2, &rtan] {
                let a = rung(hn, r, cyc, dec, flr);
                println!("  {:<4} {:<8} {:>7.3}   {}", "", r.cols.name(), median(&mut a.clone()),
                         verdict(&base, &a));
            }
        }
        println!();
    }
    // ---- Rung C, built properly. tanv1 has never had a transition of its own.
    println!("=== Rung C rebuilt: tanv1 gets its OWN transition and its own emission scale ===");
    println!("Every arm below is tanv1's emission with NO cycle prior, since rung B showed v2's prior");
    println!("double-counts tanv1's learned clock. gamma and the transition are chosen on an inner");
    println!("leave-one-out over the TRAIN cohorts, never on the reported one.
");
    const GAMMAS: [f64; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    println!("  {:<14} {:<22} {:>7}   {:>8} {:>7}  verdict vs shipped", "held-out", "arm", "kappa",
             "paired d", "bar");
    for (held, hn) in &loaded {
        let names: Vec<&str> = loaded.iter().map(|(c, _)| *c).filter(|c| c != held).collect();
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        let rtan = train(&tr, Cols::Tan);
        let base = rung_shipped(hn, true, true);
        println!("  {:<14} {:<22} {:>7.3}   the reference", format!("{held} n={}", hn.len()),
                 "shipped + floor", median(&mut base.clone()));

        // Inner folds: fit on one train cohort, score the other, so nothing selects on `held`.
        let inner: Vec<InnerFold> = names
            .iter()
            .map(|c| {
                let sub: Vec<&Night> = loaded
                    .iter()
                    .filter(|(n, _)| n != c && n != held)
                    .flat_map(|(_, n)| n.iter())
                    .collect();
                let t = fit_transition(&sub);
                (train(&sub, Cols::Tan), t, &loaded.iter().find(|(n, _)| n == c).expect("inner").1)
            })
            .collect();

        let own = fit_transition(&tr);
        for (label, use_own) in [("v2 transition", false), ("OWN transition", true)] {
            // gamma selected inside, per arm
            let mut best = (1.0f64, f64::MIN);
            for g in GAMMAS {
                let mut acc = 0.0;
                for (ir, it, iv) in &inner {
                    let d = Decode { own: use_own.then_some(*it), gamma: g };
                    let b = rung_shipped(iv, true, true);
                    let a = rung_decode(iv, ir, false, d, true);
                    acc += paired_bar(&b.iter().zip(&a).map(|(x, y)| y - x).collect::<Vec<_>>())
                        .map_or(f64::MIN, |v| v.0);
                }
                if acc > best.1 {
                    best = (g, acc);
                }
            }
            let d = Decode { own: use_own.then_some(own), gamma: best.0 };
            let a = rung_decode(hn, &rtan, false, d, true);
            println!("  {:<14} {:<22} {:>7.3}   {}", "",
                     format!("tanv1, {label}, g={}", best.0), median(&mut a.clone()),
                     verdict(&base, &a));
        }
        println!();
    }
    println!("Rung A is the honest read of the feature sets: same fitter, same folds, own weights each,");
    println!("no prior and no decoder. Rung C above is the step tanv1 never had - it had been decoding");
    println!("with a transition tuned against a different emission's scale.");
}
