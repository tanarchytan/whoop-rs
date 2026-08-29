//! Step 3: whether the per-epoch emission is ASSEMBLED right, on v2's own emissions.
//!
//!   cargo run --release -p physio-algo --example step3_emission
//!
//! Four questions, each with the cheapest control that could take its answer away:
//!
//!   A  the base prior is added at EVERY epoch. Is that a double count, and what does moving it cost?
//!   B  which rescalings of the emission are decode NO-OPS, proved by identity and then measured
//!   C  emission scale against log(transition) - whether the ratio is identified at all
//!   D  the motion gate as a hard step against a smooth ramp
//!
//! Nothing here edits `src/`. Every arm is a transform of the shipped emission, decoded by the shipped
//! `decode_v2`, paired per night against the shipped arm on the same nights.

mod common;

use common::{
    dirs_of, median, median_avg, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, epoch_starts_v2, params::Params, prepare_v2, weights_of,
    AccelSample, SleepInput, WEIGHT_NAMES,
};
use std::collections::BTreeMap;

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Confusion-matrix class order, which is `stage_idx`, not the emission's column order.
const CLASS_NAMES: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
const MIN_EPOCHS: usize = 20;
/// Emission column of awake, the only class the motion gate reaches. Columns are STAGE_ORDER:
/// deep, rem, light, awake.
const AWAKE: usize = 3;
const LIGHT: usize = 2;
/// How much of the base prior stays on the emission at epochs past the first. 1.0 is shipped.
const SCALES: [f64; 7] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
/// Logistic sharpness of the motion-gate ramp in log jerk-ratio. 64 is the shipped hard step.
const SHARPS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 64.0];

type Emission = Vec<[f64; CLASSES]>;
type Transition = [[f64; CLASSES]; CLASSES];
/// One candidate: its name, the emission it decodes, and the transition it decodes under.
type Arm = (String, Box<dyn Fn(&Night) -> Emission>, Transition);

struct Night {
    em: Emission,
    truth: Vec<Option<usize>>,
    /// Epochs the shipped hard motion gate fired on, and the jerk ratio it thresholded.
    boosted: Vec<bool>,
    ratio: Vec<f64>,
}

/// How often each of the emission's switches is active, and how often it moves the number. A flag
/// that is set but changes nothing is not a non-linearity the decoder ever meets.
#[derive(Default, Clone, Copy)]
struct Census {
    epochs: usize,
    gate: usize,
    hinge: usize,
    deadzone: usize,
    clamp_set: usize,
    clamp_bites: usize,
}

impl Census {
    fn absorb(&mut self, o: &Census) {
        self.epochs += o.epochs;
        self.gate += o.gate;
        self.hinge += o.hinge;
        self.deadzone += o.deadzone;
        self.clamp_set += o.clamp_set;
        self.clamp_bites += o.clamp_bites;
    }
    fn pct(&self, n: usize) -> f64 {
        100.0 * n as f64 / self.epochs.max(1) as f64
    }
}

/// The design slot a named weight owns, so this harness cannot transpose against the wrong column.
fn slot(name: &str) -> usize {
    WEIGHT_NAMES.iter().position(|n| *n == name).expect("a named emission weight")
}

/// Peak inter-second gravity delta per epoch over the night's median, the ratio the shipped motion
/// gate thresholds. Rebuilt here because the stager keeps it inside its own feature pass.
fn jerk_ratios(accel: &[AccelSample], starts: &[i64]) -> Vec<f64> {
    let mut acc: BTreeMap<i64, (f64, f64, f64, f64)> = BTreeMap::new();
    for g in accel {
        let e = acc.entry(g.ts).or_insert((0.0, 0.0, 0.0, 0.0));
        e.0 += g.x;
        e.1 += g.y;
        e.2 += g.z;
        e.3 += 1.0;
    }
    let sec: BTreeMap<i64, (f64, f64, f64)> =
        acc.iter().map(|(k, v)| (*k, (v.0 / v.3, v.1 / v.3, v.2 / v.3))).collect();
    let mut all: Vec<f64> = Vec::new();
    let mut peak: Vec<f64> = Vec::with_capacity(starts.len());
    for s in starts {
        let seq: Vec<(f64, f64, f64)> = (*s..*s + 30).filter_map(|t| sec.get(&t).copied()).collect();
        let j: Vec<f64> = seq
            .windows(2)
            .map(|w| {
                let (dx, dy, dz) = (w[0].0 - w[1].0, w[0].1 - w[1].1, w[0].2 - w[1].2);
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        peak.push(j.iter().copied().fold(0.0f64, f64::max));
        all.extend_from_slice(&j);
    }
    let scale = median_avg(&mut all);
    peak.iter()
        .map(|p| if scale > 0.0 { p / scale } else if *p > 0.0 { f64::INFINITY } else { 0.0 })
        .collect()
}

/// One cohort's nights, plus how often the rebuilt jerk ratio disagrees with the gate the stager
/// actually fired. A nonzero count makes the ramp arm unreadable, so it is reported.
fn load(set: &str) -> (Vec<Night>, usize, Census) {
    let base_awake = Params::SHIPPED.base_rate[AWAKE].ln();
    let boost = Params::SHIPPED.motion_gate_boost;
    let w = weights_of(&Params::SHIPPED);
    let (s_hrv, s_hr, s_gate) = (slot("awake_hrv"), slot("awake_hr"), slot("deep_gate_slope"));
    let mut out = Vec::new();
    let mut mismatch = 0usize;
    let mut census = Census::default();
    for dir in &dirs_of(set) {
        let raw = read_truth(dir);
        let Some((w0, w1, n_meta)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if raw.is_empty() || accel.is_empty() {
            continue;
        }
        let n = n_meta.max(raw.keys().max().copied().unwrap_or(0) + 1);
        let (hr, rr) = (read_hr(dir), read_rr(dir));
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions", dir.display(),
                   em.len());
        let terms = emission_terms(&prep, &Params::SHIPPED);
        let boosted: Vec<bool> = (0..em.len())
            .map(|e| {
                let d = terms.fixed[e][AWAKE] - base_awake;
                assert!(d.abs() < 1e-9 || (d - boost).abs() < 1e-9,
                        "the awake fixed term is neither the base rate nor base rate plus the gate");
                d.abs() > 1e-9
            })
            .collect();
        let ratio = jerk_ratios(&input.accel, &epoch_starts_v2(&prep));
        mismatch += (0..em.len())
            .filter(|e| (ratio[*e] > Params::SHIPPED.jerk_gate_mult) != boosted[*e])
            .count();
        for (e, d) in terms.design.iter().enumerate().take(em.len()) {
            let card = w[s_hrv] * d[AWAKE][s_hrv] + w[s_hr] * d[AWAKE][s_hr];
            census.epochs += 1;
            census.gate += usize::from(boosted[e]);
            census.hinge += usize::from(d[0][s_gate] != 0.0);
            census.deadzone += usize::from(d[AWAKE][s_hrv] == 0.0 || d[AWAKE][s_hr] == 0.0);
            census.clamp_set += usize::from(terms.clamped[e]);
            census.clamp_bites += usize::from(terms.clamped[e] && card > 0.0);
        }
        let truth = (0..em.len())
            .map(|k| {
                raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night { em, truth, boosted, ratio });
    }
    (out, mismatch, census)
}

/// The shipped base prior in the log domain, in the emission's own column order.
fn blp() -> [f64; CLASSES] {
    std::array::from_fn(|c| Params::SHIPPED.base_rate[c].ln())
}

/// The shipped emission with `s` of the base prior left on epochs past the first. `s = 1` is shipped;
/// `s = 0` applies it once, at epoch 0, the way an HMM applies an initial distribution. `from_first`
/// takes epoch 0 too, so the night has no initial distribution at all.
fn prior_scaled(n: &Night, s: f64, from_first: bool) -> Emission {
    let b = blp();
    n.em
        .iter()
        .enumerate()
        .map(|(e, row)| {
            let keep = e == 0 && !from_first;
            std::array::from_fn(|c| row[c] - if keep { 0.0 } else { (1.0 - s) * b[c] })
        })
        .collect()
}

/// `transition[i][j] * base_rate[j]^pow`: the base prior moved off the emission onto the transition
/// columns. At `pow = 1 - s` this is the exact partner of [`prior_scaled`].
fn prior_transition(pow: f64) -> Transition {
    let r = Params::SHIPPED.base_rate;
    std::array::from_fn(|i| std::array::from_fn(|j| Params::SHIPPED.transition[i][j] * r[j].powf(pow)))
}

/// Every row shifted so it sums to one in probability: a per-epoch constant.
fn log_softmax(n: &Night) -> Emission {
    n.em
        .iter()
        .map(|row| {
            let mx = row.iter().copied().fold(f64::MIN, f64::max);
            let lse = mx + row.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            std::array::from_fn(|c| row[c] - lse)
        })
        .collect()
}

/// The shipped emission with the whole base prior divided out at every epoch, which renormalises
/// `base_rate` to sum 1. A per-epoch constant, so it must not move a path.
fn renormalised(n: &Night) -> Emission {
    let z: f64 = Params::SHIPPED.base_rate.iter().sum::<f64>().ln();
    n.em.iter().map(|row| std::array::from_fn(|c| row[c] - z)).collect()
}

/// The shipped hard motion gate replaced by a logistic ramp of sharpness `k` in log jerk-ratio. Large
/// `k` reproduces the step; the boost height and the threshold are the shipped ones either way.
fn ramped_gate(n: &Night, k: f64) -> Emission {
    gated(n, |r| {
        let boost = Params::SHIPPED.motion_gate_boost;
        match r {
            r if r <= 0.0 => 0.0,
            r if r.is_infinite() => boost,
            r => boost / (1.0 + (-k * (r.ln() - Params::SHIPPED.jerk_gate_mult.ln())).exp()),
        }
    })
}

/// The shipped emission with the motion gate removed entirely.
fn no_gate(n: &Night) -> Emission {
    gated(n, |_| 0.0)
}

/// The shipped emission with its awake motion-gate step swapped for `want(jerk ratio)`.
fn gated(n: &Night, want: impl Fn(f64) -> f64) -> Emission {
    let boost = Params::SHIPPED.motion_gate_boost;
    n.em
        .iter()
        .enumerate()
        .map(|(e, row)| {
            let had = if n.boosted[e] { boost } else { 0.0 };
            let d = want(n.ratio[e]) - had;
            std::array::from_fn(|c| row[c] + if c == AWAKE { d } else { 0.0 })
        })
        .collect()
}

fn path_of(em: &[[f64; CLASSES]], t: &Transition) -> Vec<usize> {
    decode_v2(em, t).iter().map(|s| stage_idx(*s)).collect()
}

/// Per-night kappa and the pooled confusion, over nights carrying enough labelled epochs.
fn score(
    nights: &[Night], em_of: &dyn Fn(&Night) -> Emission, t: &Transition,
) -> (Vec<f64>, [[i64; CLASSES]; CLASSES]) {
    let mut ks = Vec::new();
    let mut cm = [[0i64; CLASSES]; CLASSES];
    for n in nights {
        let path = path_of(&em_of(n), t);
        let (mut p, mut y) = (Vec::new(), Vec::new());
        for (k, want) in n.truth.iter().enumerate() {
            if let Some(w) = want {
                p.push(path[k]);
                y.push(*w);
            }
        }
        if y.len() < MIN_EPOCHS {
            continue;
        }
        let c = confusion4(&p, &y);
        ks.push(kappa4(&c));
        for (i, row) in c.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                cm[i][j] += v;
            }
        }
    }
    (ks, cm)
}

/// Every night's decoded label sequence under one arm, for identity checks that must be exact.
fn all_paths(nights: &[Night], em_of: &dyn Fn(&Night) -> Emission, t: &Transition) -> Vec<Vec<usize>> {
    nights.iter().map(|n| path_of(&em_of(n), t)).collect()
}

/// Epochs on which two arms decode a different label, and the total compared.
fn disagreement(a: &[Vec<usize>], b: &[Vec<usize>]) -> (usize, usize) {
    let (mut d, mut t) = (0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        for (p, q) in x.iter().zip(y) {
            t += 1;
            d += usize::from(p != q);
        }
    }
    (d, t)
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

/// Stay-minus-best-leave per row, in log units. Row normalisation cannot move it, so it reads the
/// same off a transition and off that transition with the base prior folded into its columns.
fn switch_charges(t: &Transition) -> [f64; CLASSES] {
    std::array::from_fn(|i| {
        let stay = t[i][i].max(1e-9).ln();
        let go = (0..CLASSES)
            .filter(|j| *j != i)
            .map(|j| t[i][j].max(1e-9).ln())
            .fold(f64::MIN, f64::max);
        stay - go
    })
}

/// Top-two gap of every epoch's emission: the evidence the transition has to overturn.
fn margins(nights: &[Night], em_of: &dyn Fn(&Night) -> Emission) -> Vec<f64> {
    let mut out = Vec::new();
    for n in nights {
        for row in em_of(n) {
            let mut v = row.to_vec();
            v.sort_by(f64::total_cmp);
            out.push(v[CLASSES - 1] - v[CLASSES - 2]);
        }
    }
    out
}

/// Recall per class off a pooled confusion, rows = truth.
fn recalls(cm: &[[i64; CLASSES]; CLASSES]) -> [f64; CLASSES] {
    std::array::from_fn(|c| {
        let t: i64 = cm[c].iter().sum();
        100.0 * cm[c][c] as f64 / t.max(1) as f64
    })
}

/// Pick a candidate by the mean paired gain over the TRAIN cohorts. The held-out cohort is never
/// scored here, so the arm it reports was chosen without it.
fn inner_pick(loaded: &[(&str, Vec<Night>)], held: &str, arms: &[Arm]) -> usize {
    let mut best = (f64::MIN, 0usize);
    for (k, (_, em_of, t)) in arms.iter().enumerate() {
        let (mut gain, mut seen) = (0.0f64, 0.0f64);
        for (_, nights) in loaded.iter().filter(|(c, _)| *c != held) {
            let (zero, _) = score(nights, &|n: &Night| n.em.clone(), &Params::SHIPPED.transition);
            let (arm, _) = score(nights, em_of.as_ref(), t);
            gain += verdict(&zero, &arm).0;
            seen += 1.0;
        }
        if seen > 0.0 && gain / seen > best.0 {
            best = (gain / seen, k);
        }
    }
    best.1
}

/// One inner-selected arm reported once on each held-out cohort.
fn report_selection(loaded: &[(&str, Vec<Night>)], arms: &[Arm], with_recall: bool) {
    let ship = |n: &Night| n.em.clone();
    println!("  {:<14} {:>8} {:>9} {:>9}   {:>10} {:>9}   verdict",
             "held-out", "picked", "shipped", "arm", "paired d", "bar +/-");
    for (held, nights) in loaded {
        let k = inner_pick(loaded, held, arms);
        let (base, cmb) = score(nights, &ship, &Params::SHIPPED.transition);
        let (arm, cma) = score(nights, arms[k].1.as_ref(), &arms[k].2);
        let (m, bar, v) = verdict(&base, &arm);
        println!("  {:<14} {:>8} {:>9.3} {:>9.3}   {m:>+10.4} {bar:>9.4}   {v}",
                 format!("{held} n={}", base.len()), arms[k].0,
                 median(&mut base.clone()), median(&mut arm.clone()));
        if with_recall {
            let (rb, ra) = (recalls(&cmb), recalls(&cma));
            let cells: Vec<String> =
                (0..CLASSES).map(|c| format!("{} {:+.1}", CLASS_NAMES[c], ra[c] - rb[c])).collect();
            println!("  {:<14} recall change: {}", "", cells.join("  "));
        }
    }
}

/// A paired-difference row over every cohort, one cell per candidate arm.
fn sweep_row(nights: &[Night], arms: &[Arm]) -> String {
    let (base, _) = score(nights, &|n: &Night| n.em.clone(), &Params::SHIPPED.transition);
    let mut line = String::new();
    for (_, em_of, t) in arms {
        let (arm, _) = score(nights, em_of.as_ref(), t);
        let (m, bar, _) = verdict(&base, &arm);
        let flag = if m.abs() <= bar { ' ' } else if m > 0.0 { '+' } else { '-' };
        line.push_str(&format!(" {m:>+7.3}{flag}"));
    }
    line
}

fn scale_arms() -> Vec<Arm> {
    SCALES
        .iter()
        .map(|s| {
            let s = *s;
            let f: Box<dyn Fn(&Night) -> Emission> = Box::new(move |n: &Night| prior_scaled(n, s, false));
            (format!("s={s:.2}"), f, Params::SHIPPED.transition)
        })
        .collect()
}

fn gate_arms() -> Vec<Arm> {
    SHARPS
        .iter()
        .map(|k| {
            let k = *k;
            let f: Box<dyn Fn(&Night) -> Emission> = Box::new(move |n: &Night| ramped_gate(n, k));
            (format!("k={k:.0}"), f, Params::SHIPPED.transition)
        })
        .collect()
}

/// Each row of `t` scaled to sum 1, so it reads as a stochastic transition matrix.
fn row_normalised(t: &Transition) -> Transition {
    std::array::from_fn(|i| {
        let z: f64 = t[i].iter().sum();
        std::array::from_fn(|j| t[i][j] / z)
    })
}

fn main() {
    let mut census = Census::default();
    let loaded: Vec<(&str, Vec<Night>)> = COHORTS
        .iter()
        .map(|c| {
            let (n, mismatch, cs) = load(c);
            if !n.is_empty() {
                let epochs: usize = n.iter().map(|x| x.em.len()).sum();
                println!("{c}: {} nights, {epochs} epochs, {mismatch} gate-flag mismatches", n.len());
            }
            census.absorb(&cs);
            (*c, n)
        })
        .filter(|(_, n)| !n.is_empty())
        .collect();
    if loaded.len() < 2 {
        println!("need at least two cohorts under the fixture root");
        return;
    }
    let shipped_t = Params::SHIPPED.transition;
    let ship = |n: &Night| n.em.clone();
    let b = blp();

    println!("\nbase_rate {:?} sums to {:.4}", Params::SHIPPED.base_rate,
             Params::SHIPPED.base_rate.iter().sum::<f64>());
    println!("log prior {:?}", b.map(|v| (v * 1000.0).round() / 1000.0));
    println!("light over deep {:.3} nats, every epoch\n", b[LIGHT] - b[0]);

    // ---- A1 --------------------------------------------------------------------------------
    println!("A1  what adding the prior at EVERY epoch actually is");
    println!("  claim: emission + log(pi) everywhere, under transition T, is the SAME decode as the");
    println!("  prior at epoch 0 only under T[i][j]*pi[j]. If it holds, the repetition is not a");
    println!("  double-applied initial distribution - it is a re-weighted transition on the wrong side.");
    let folded = prior_transition(1.0);
    for (c, nights) in &loaded {
        let a = all_paths(nights, &ship, &shipped_t);
        let z = all_paths(nights, &|n: &Night| prior_scaled(n, 0.0, false), &folded);
        let (d, t) = disagreement(&a, &z);
        println!("    {c:<12} {d} of {t} epochs differ");
    }

    // ---- A2 --------------------------------------------------------------------------------
    println!("\nA2  so the transition the decoder really runs is not the one written in Params");
    let names = ["deep", "rem", "light", "awake"];
    println!("  {:<7} {:>27}   {:>27}", "from", "written, row-normalised", "effective, row-normalised");
    for (i, name) in names.iter().enumerate() {
        let rw: f64 = shipped_t[i].iter().sum();
        let re: f64 = folded[i].iter().sum();
        let w: Vec<String> = shipped_t[i].iter().map(|v| format!("{:.3}", v / rw)).collect();
        let e: Vec<String> = folded[i].iter().map(|v| format!("{:.3}", v / re)).collect();
        println!("  {:<7} {:>27}   {:>27}", name, w.join(" "), e.join(" "));
    }
    let (cw, ce) = (switch_charges(&shipped_t), switch_charges(&folded));
    println!("  stay-minus-best-leave, nats: written {:?}", cw.map(|v| (v * 1000.0).round() / 1000.0));
    println!("                            effective {:?}", ce.map(|v| (v * 1000.0).round() / 1000.0));
    println!("  cheapest stage change: written {:.3}, effective {:.3}",
             cw.iter().copied().fold(f64::MAX, f64::min), ce.iter().copied().fold(f64::MAX, f64::min));

    // ---- B ---------------------------------------------------------------------------------
    println!("\nB  rescalings that CANNOT change a Viterbi path, argued then measured");
    println!("  a constant subtracted from all four classes at one epoch shifts EVERY path's score by");
    println!("  the same amount, so log-softmax and renormalising base_rate to sum 1 are both no-ops.");
    for (c, nights) in &loaded {
        let a = all_paths(nights, &ship, &shipped_t);
        let (d1, t) = disagreement(&a, &all_paths(nights, &log_softmax, &shipped_t));
        let (d2, _) = disagreement(&a, &all_paths(nights, &renormalised, &shipped_t));
        println!("    {c:<12} log-softmax {d1} of {t} epochs differ, base_rate/1.21 {d2}");
    }

    // ---- C ---------------------------------------------------------------------------------
    println!("\nC  emission scale against log(transition)");
    let gamma = 0.7;
    let floored: Transition = std::array::from_fn(|i| std::array::from_fn(|j| shipped_t[i][j].max(1e-9)));
    let tempered: Transition =
        std::array::from_fn(|i| std::array::from_fn(|j| floored[i][j].powf(1.0 / gamma)));
    for (c, nights) in &loaded {
        let hot = all_paths(
            nights,
            &|n: &Night| n.em.iter().map(|r| r.map(|v| v * gamma)).collect(),
            &floored,
        );
        let (d, t) = disagreement(&hot, &all_paths(nights, &ship, &tempered));
        let mut m = margins(nights, &ship);
        let mut mz = margins(nights, &|n: &Night| prior_scaled(n, 0.0, false));
        println!("    {c:<12} emission x{gamma} == transition^(1/{gamma}): {d} of {t} epochs differ");
        println!("    {:<12} median top-two margin {:.3} with pi on the emission, {:.3} with pi on \
                  the transition", "", median(&mut m), median(&mut mz));
    }
    println!("  emission temperature and transition temperature are ONE axis, already swept as beta.");
    println!("  Only the SUM is identified: the margin-to-charge ratio depends on which side pi sits.");
    let flat = loaded
        .iter()
        .flat_map(|(_, n)| n.iter())
        .map(|n| {
            let v: Vec<f64> = n.em.iter().map(|r| r[LIGHT]).collect();
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|x| (x - mean).abs()).fold(0.0f64, f64::max)
        })
        .fold(0.0f64, f64::max);
    println!("  light is the REFERENCE class: its emission moves at most {flat:.2e} inside a night, so");
    println!("  the base prior alone sets the level the other three are scored against.");

    // ---- A3 --------------------------------------------------------------------------------
    println!("\nA3  move the prior off the emission, on v2's own emissions (paired vs shipped)");
    println!("  s is how much of log(pi) stays at every epoch. s=1 is shipped, s=0 is epoch 0 only.");
    let arms = scale_arms();
    print!("  {:<14}", "cohort");
    for s in SCALES {
        print!(" {s:>7.2} ");
    }
    println!("  {:>9}", "none at all");
    for (c, nights) in &loaded {
        let (base, _) = score(nights, &ship, &shipped_t);
        let (arm, _) = score(nights, &|n: &Night| prior_scaled(n, 0.0, true), &shipped_t);
        println!("  {:<14}{}   {:>+9.3}", format!("{c} n={}", base.len()),
                 sweep_row(nights, &arms), verdict(&base, &arm).0);
    }
    println!("  cells are the PAIRED mean kappa change; + or - marks one outside its own 95% bar.");
    println!("  the last column drops the prior at epoch 0 too, so there is no initial distribution.");

    // ---- A4 --------------------------------------------------------------------------------
    println!("\nA4  choose s on the TRAIN cohorts, report it once on the held-out one");
    report_selection(&loaded, &arms, true);

    // ---- A5 --------------------------------------------------------------------------------
    println!("\nA5  the textbook restatement: prior at epoch 0, a STOCHASTIC transition carrying pi");
    println!("  A1's effective matrix does not sum to 1 per row. Normalising it is not free - it");
    println!("  subtracts log(row sum) from the epoch BEFORE each step, which is another per-epoch");
    println!("  class bias. This arm measures exactly that residual, and nothing else.");
    let normed = row_normalised(&folded);
    let z: Vec<f64> = (0..CLASSES).map(|i| folded[i].iter().sum::<f64>().ln()).collect();
    println!("  residual bias log(row sum) {:?}",
             z.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f64>>());
    for (c, nights) in &loaded {
        let (base, _) = score(nights, &ship, &shipped_t);
        let (arm, _) = score(nights, &|n: &Night| prior_scaled(n, 0.0, false), &normed);
        let (m, bar, v) = verdict(&base, &arm);
        println!("    {:<14} shipped {:.3}  arm {:.3}  paired {m:>+8.4} +/- {bar:.4}  {v}",
                 format!("{c} n={}", base.len()), median(&mut base.clone()), median(&mut arm.clone()));
    }

    // ---- E ---------------------------------------------------------------------------------
    println!("\nE  is the form linear-additive? It already carries four switches");
    println!("  {:<22} {:>9} {:>9}   shape", "switch", "epochs", "% of all");
    let rows = [
        ("deep gate hinge", census.hinge, "continuous ReLU past the flatness percentile"),
        ("awake cardiac deadzone", census.deadzone, "continuous, piecewise linear in z"),
        ("awake cardiac clamp set", census.clamp_set, "min(pair, 0), continuous"),
        ("  of which it bites", census.clamp_bites, "the pair was positive, so the number moved"),
        ("motion gate", census.gate, "DISCONTINUOUS step of 4.0 nats"),
    ];
    for (name, n, shape) in rows {
        println!("  {:<22} {:>9} {:>8.1}%   {shape}", name, n, census.pct(n));
    }
    println!("  exactly one of the four is a discontinuity, and D is the test of that one.");

    // ---- D ---------------------------------------------------------------------------------
    println!("\nD  the motion gate: hard step against a logistic ramp of sharpness k in log jerk-ratio");
    let fired: usize = loaded
        .iter()
        .flat_map(|(_, n)| n.iter())
        .map(|n| n.boosted.iter().filter(|x| **x).count())
        .sum();
    let all: usize = loaded.iter().flat_map(|(_, n)| n.iter()).map(|n| n.em.len()).sum();
    println!("  the gate fires on {fired} of {all} epochs ({:.1}%), which caps what any shape can move",
             100.0 * fired as f64 / all as f64);
    let gates = gate_arms();
    print!("  {:<14}", "cohort");
    for k in SHARPS {
        print!(" {k:>7.0} ");
    }
    println!("  {:>9}", "no gate");
    for (c, nights) in &loaded {
        let (base, _) = score(nights, &ship, &shipped_t);
        let (arm, _) = score(nights, &no_gate, &shipped_t);
        println!("  {:<14}{}   {:>+9.3}", format!("{c} n={}", base.len()),
                 sweep_row(nights, &gates), verdict(&base, &arm).0);
    }
    println!("  k=64 is the step reproduced to 1e-9, so that column is the harness agreeing with itself.");
    println!("  the last column removes the boost outright: the cheapest control on whether the gate's");
    println!("  SHAPE matters at all, or only its presence.");

    println!("\nD2  choose k on the TRAIN cohorts, report it once on the held-out one");
    report_selection(&loaded, &gates, false);

    println!("\nEvery arm is v2's own emission transformed and re-decoded. Nothing is fitted, so the only");
    println!("selection is the inner pick; A1, B and C are identities that either hold or do not.");
}
