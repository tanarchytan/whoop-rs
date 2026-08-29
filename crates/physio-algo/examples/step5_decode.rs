//! Step 5: is the DECODE computed the right way?
//!
//!   cargo run --release -p physio-algo --example step5_decode
//!
//! Viterbi maximises the joint probability of the whole PATH. Kappa scores each EPOCH on its own, and
//! the Bayes decoder for a per-epoch loss is the per-epoch posterior MARGINAL from forward-backward,
//! not the MAP path. Both run here over the same emissions and the same transition, on v2's own
//! emissions first and then on tanv1's, alongside the other contract questions: the start
//! distribution, the missing end term, an explicit dwell model, and the tie rule.
//!
//! Nothing in `src/` is touched. `viterbi_n` at k=1 with a uniform start is asserted equal to
//! `decode_v2` night for night, and the HSMM under the shipped geometric dwell is asserted equal to
//! the same path, so every arm is measured against a reproduction of the shipped decoder.

mod common;

use common::lr::{design_row, standardise_cols};
use common::{
    cardiac_series, dirs_of, median, read_accel, read_hr, read_meta, read_rr, read_truth, stage_idx,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar, recall, Confusion4};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, SleepInput, STAGE_ORDER,
};

const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const CLASS_NAMES: [&str; CLASSES] = ["wake", "light", "deep", "rem"];
const MIN_EPOCHS: usize = 20;
const EPOCH: i64 = 30;
/// The correction strength `fit_residual` and `emission_steps` report tanv1 at.
const L2: f64 = 1.0;
const ITERS: usize = 6_000;
const TOL: f64 = 1e-10;
const LR: f64 = 0.5;
const WEIGHT_POWER: f64 = 0.5;
/// The transition floor `viterbi` applies before taking a log.
const FLOOR: f64 = 1e-9;
/// Dwell phase counts the Erlang-expansion arms use. k=1 is the shipped chain.
const KS: [usize; 2] = [2, 3];
/// Longest dwell the explicit-duration tables hold, in epochs.
const DMAX: usize = 2400;
/// Weight of the geometric backstop mixed into an empirical dwell, swept by the inner selection.
const LAMBDAS: [f64; 2] = [0.1, 0.5];

// ---------------------------------------------------------------------------------------------
// Nights
// ---------------------------------------------------------------------------------------------

struct Night {
    /// Design columns for the tanv1 correction, one row per epoch.
    row: Vec<Vec<f64>>,
    /// The shipped log-emission per epoch, in [`STAGE_ORDER`] columns.
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
        let em = emissions_v2(&prep, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions", dir.display(),
                   em.len());
        assert!(em.len() <= DMAX, "{}: {} epochs is past the dwell table", dir.display(), em.len());
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
        out.push(Night { row, em: em[..].to_vec(), truth });
    }
    out
}

/// Our class index -> the emission's [`STAGE_ORDER`] column.
fn col_of(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

// ---------------------------------------------------------------------------------------------
// Decoder arms
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Dec {
    Viterbi,
    Posterior,
    /// Explicit-duration Viterbi under the shipped geometric dwell: the HSMM positive control.
    HsmmGeom,
    /// The same, but charging the pmf on the final segment instead of the survival.
    HsmmGeomPmfEnd,
    /// Explicit-duration Viterbi under a dwell estimated from TRAIN truth, mixed with a geometric.
    HsmmFit(f64),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Start {
    Uniform,
    Stationary,
    BaseRate,
}

/// One decoder configuration: which path rule, which epoch-0 distribution, how many Erlang phases.
#[derive(Clone, Copy, PartialEq)]
struct Arm {
    dec: Dec,
    start: Start,
    k: usize,
}

impl Arm {
    fn name(&self) -> String {
        let s = match self.start {
            Start::Uniform => "uniform",
            Start::Stationary => "stationary",
            Start::BaseRate => "base_rate",
        };
        match self.dec {
            Dec::Viterbi => format!("viterbi/{s}/k{}", self.k),
            Dec::Posterior => format!("posterior/{s}/k{}", self.k),
            Dec::HsmmGeom => "hsmm/geometric".to_string(),
            Dec::HsmmGeomPmfEnd => "hsmm/geom+pmf-end".to_string(),
            Dec::HsmmFit(l) => format!("hsmm/fit-dwell L{l}"),
        }
    }
    /// True when the arm reads TRAIN truth, so its selection has to refit per inner fold.
    fn fitted(&self) -> bool {
        matches!(self.dec, Dec::HsmmFit(_))
    }
}

const BASE: Arm = Arm { dec: Dec::Viterbi, start: Start::Uniform, k: 1 };
const POST: Arm = Arm { dec: Dec::Posterior, start: Start::Uniform, k: 1 };

/// Running counts a decode reports about itself, so numerical and tie behaviour is measured.
#[derive(Default, Clone, Copy)]
struct DecStats {
    /// Predecessor comparisons in the Viterbi recursion that came out exactly equal.
    pred_ties: usize,
    /// Per-epoch decisions where the top two scores were exactly equal.
    decide_ties: usize,
    /// Winning Viterbi edges whose transition probability was the 1e-9 floor.
    floor_edges: usize,
    non_finite: usize,
    /// Largest magnitude reached by a forward or backward accumulator.
    max_abs: f64,
    /// Posterior arms only: the max marginal probability of each per-epoch decision.
    margin_sum: f64,
    margin_n: usize,
    /// Longest single-class run the arm drew, in epochs.
    max_run: usize,
}

impl DecStats {
    fn absorb(&mut self, o: &DecStats) {
        self.pred_ties += o.pred_ties;
        self.decide_ties += o.decide_ties;
        self.floor_edges += o.floor_edges;
        self.non_finite += o.non_finite;
        self.max_abs = self.max_abs.max(o.max_abs);
        self.margin_sum += o.margin_sum;
        self.margin_n += o.margin_n;
        self.max_run = self.max_run.max(o.max_run);
    }
    fn margin(&self) -> f64 {
        self.margin_sum / self.margin_n.max(1) as f64
    }
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// Index of the largest entry, ties to the earliest, counting exact ties.
fn argmax(v: &[f64], st: &mut DecStats) -> usize {
    let mut best = (0usize, v[0]);
    let mut tie = false;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best.1 {
            best = (i, x);
            tie = false;
        } else if x == best.1 {
            tie = true;
        }
    }
    st.decide_ties += usize::from(tie);
    best.0
}

/// The shipped Viterbi over an arbitrary state count, plus a start term the shipped call leaves zero.
#[allow(clippy::needless_range_loop)]
fn viterbi_n(em: &[Vec<f64>], log_t: &[Vec<f64>], start: &[f64], st: &mut DecStats) -> Vec<usize> {
    let n = start.len();
    let mut v: Vec<f64> = (0..n).map(|s| start[s] + em[0][s]).collect();
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(em.len().saturating_sub(1));
    for e in &em[1..] {
        let mut nv = vec![0.0f64; n];
        let mut bp = vec![0usize; n];
        for s in 0..n {
            let mut best = (0usize, v[0] + log_t[0][s]);
            for p in 1..n {
                let val = v[p] + log_t[p][s];
                if val > best.1 {
                    best = (p, val);
                } else if val == best.1 {
                    st.pred_ties += 1;
                }
            }
            nv[s] = best.1 + e[s];
            bp[s] = best.0;
            st.max_abs = st.max_abs.max(nv[s].abs());
            st.non_finite += usize::from(!nv[s].is_finite());
        }
        v = nv;
        back.push(bp);
    }
    let mut last = argmax(&v, st);
    let mut path = vec![last];
    for bp in back.iter().rev() {
        last = bp[last];
        path.push(last);
    }
    path.reverse();
    path
}

/// Forward-backward in log space; returns the per-epoch posterior marginal, normalised to sum to 1.
/// Same potentials Viterbi maximises, so a per-epoch constant in the emission cancels here.
#[allow(clippy::needless_range_loop)]
fn forward_backward(
    em: &[Vec<f64>],
    log_t: &[Vec<f64>],
    start: &[f64],
    st: &mut DecStats,
) -> Vec<Vec<f64>> {
    let n = start.len();
    let t_len = em.len();
    let mut buf = vec![0.0f64; n];
    let mut alpha = vec![vec![f64::NEG_INFINITY; n]; t_len];
    for s in 0..n {
        alpha[0][s] = start[s] + em[0][s];
    }
    for t in 1..t_len {
        for s in 0..n {
            for p in 0..n {
                buf[p] = alpha[t - 1][p] + log_t[p][s];
            }
            alpha[t][s] = logsumexp(&buf) + em[t][s];
            st.max_abs = st.max_abs.max(alpha[t][s].abs());
            st.non_finite += usize::from(!alpha[t][s].is_finite());
        }
    }
    let mut beta = vec![vec![0.0f64; n]; t_len];
    for t in (0..t_len.saturating_sub(1)).rev() {
        for s in 0..n {
            for q in 0..n {
                buf[q] = log_t[s][q] + em[t + 1][q] + beta[t + 1][q];
            }
            beta[t][s] = logsumexp(&buf);
            st.max_abs = st.max_abs.max(beta[t][s].abs());
            st.non_finite += usize::from(!beta[t][s].is_finite());
        }
    }
    (0..t_len)
        .map(|t| {
            let g: Vec<f64> = (0..n).map(|s| alpha[t][s] + beta[t][s]).collect();
            let z = logsumexp(&g);
            g.iter().map(|v| (v - z).exp()).collect()
        })
        .collect()
}

/// Per-epoch argmax of the posterior marginal, recording how confident each decision was.
fn posterior_n(em: &[Vec<f64>], log_t: &[Vec<f64>], start: &[f64], st: &mut DecStats) -> Vec<usize> {
    forward_backward(em, log_t, start, st)
        .iter()
        .map(|g| {
            let i = argmax(g, st);
            st.margin_sum += g[i];
            st.margin_n += 1;
            i
        })
        .collect()
}

/// The 4k-state log transition of a k-phase Erlang expansion, floored exactly as `viterbi` floors.
///
/// State `c*k + j` is class `c` in phase `j`. Self-loop `p_c = 1 - k(1 - t[c][c])` keeps the MEAN
/// dwell equal to the shipped geometric mean; k=1 is the shipped chain unchanged.
fn log_chain(t: &[[f64; CLASSES]; CLASSES], k: usize) -> Vec<Vec<f64>> {
    let n = CLASSES * k;
    let mut m = vec![vec![0.0f64; n]; n];
    for c in 0..CLASSES {
        let stay = 1.0 - (k as f64) * (1.0 - t[c][c]);
        assert!(stay >= 0.0, "k={k} is longer than class {c}'s mean dwell; the mean would move");
        let leave = 1.0 - stay;
        let out_mass = 1.0 - t[c][c];
        for j in 0..k {
            m[c * k + j][c * k + j] = stay;
            if j + 1 < k {
                m[c * k + j][c * k + j + 1] = leave;
            } else {
                for d in 0..CLASSES {
                    if d != c {
                        m[c * k + j][d * k] = leave * t[c][d] / out_mass;
                    }
                }
            }
        }
    }
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v = v.max(FLOOR).ln();
        }
    }
    m
}

/// One emission row per expanded state: every phase of a class carries that class's emission.
fn expand_em(em: &[[f64; CLASSES]], k: usize) -> Vec<Vec<f64>> {
    em.iter().map(|e| (0..CLASSES * k).map(|s| e[s / k]).collect()).collect()
}

/// The epoch-0 log distribution an arm asks for, over the expanded state space.
fn start_vec(s: Start, p: &Params, log_t: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = CLASSES * k;
    match s {
        Start::Uniform => vec![0.0; n],
        // The chain's own stationary law: the textbook initial distribution for a stationary HMM.
        Start::Stationary => stationary(log_t).iter().map(|v| v.max(1e-12).ln()).collect(),
        // The shipped base rate. Already inside every emission, so this applies it a second time.
        Start::BaseRate => (0..n).map(|i| p.base_rate[i / k].ln()).collect(),
    }
}

/// Stationary distribution by power iteration on the exponentiated log transition.
#[allow(clippy::needless_range_loop)]
fn stationary(log_t: &[Vec<f64>]) -> Vec<f64> {
    let n = log_t.len();
    let t: Vec<Vec<f64>> = log_t.iter().map(|r| r.iter().map(|v| v.exp()).collect()).collect();
    let mut pi = vec![1.0 / n as f64; n];
    for _ in 0..10_000 {
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                next[j] += pi[i] * t[i][j];
            }
        }
        let sum: f64 = next.iter().sum();
        for v in next.iter_mut() {
            *v /= sum;
        }
        let moved: f64 = pi.iter().zip(&next).map(|(a, b)| (a - b).abs()).sum();
        pi = next;
        if moved < 1e-14 {
            break;
        }
    }
    pi
}

// ---------------------------------------------------------------------------------------------
// Explicit duration (HSMM)
// ---------------------------------------------------------------------------------------------

/// A dwell law per class: the pmf a completed segment pays, and the survival a censored one pays.
struct Dwell {
    log_pmf: Vec<Vec<f64>>,
    log_surv: Vec<Vec<f64>>,
}

/// The geometric dwell the shipped self-transition already implies, written analytically so the
/// truncation at [`DMAX`] costs nothing. With the renormalised exit row this IS the shipped HMM.
fn dwell_geometric(t: &[[f64; CLASSES]; CLASSES]) -> Dwell {
    let mut log_pmf = vec![vec![0.0f64; DMAX]; CLASSES];
    let mut log_surv = vec![vec![0.0f64; DMAX]; CLASSES];
    for c in 0..CLASSES {
        let a = t[c][c].max(FLOOR);
        for d in 1..=DMAX {
            log_surv[c][d - 1] = (d - 1) as f64 * a.ln();
            log_pmf[c][d - 1] = log_surv[c][d - 1] + (1.0 - a).max(FLOOR).ln();
        }
    }
    Dwell { log_pmf, log_surv }
}

/// A dwell estimated from TRAIN truth run lengths, mixed with the class's own moment-matched
/// geometric so every duration keeps mass and the tail past the observed maximum stays finite.
fn dwell_fitted(runs: &[Vec<usize>], t: &[[f64; CLASSES]; CLASSES], lambda: f64) -> Dwell {
    let mut log_pmf = vec![vec![0.0f64; DMAX]; CLASSES];
    let mut log_surv = vec![vec![0.0f64; DMAX]; CLASSES];
    for c in 0..CLASSES {
        let n = runs[c].len();
        let mean = if n == 0 {
            1.0 / (1.0 - t[c][c]).max(FLOOR)
        } else {
            runs[c].iter().sum::<usize>() as f64 / n as f64
        };
        let q = (1.0 / mean.max(1.0)).clamp(1e-6, 1.0 - 1e-9);
        let mut p = vec![0.0f64; DMAX];
        for &d in &runs[c] {
            if (1..=DMAX).contains(&d) {
                p[d - 1] += 1.0;
            }
        }
        let mass = n.max(1) as f64;
        for d in 1..=DMAX {
            p[d - 1] += lambda * mass * (1.0 - q).powi(d as i32 - 1) * q;
        }
        let total: f64 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= total;
        }
        let mut tail = 0.0f64;
        for d in (1..=DMAX).rev() {
            tail += p[d - 1];
            log_surv[c][d - 1] = tail.max(1e-300).ln();
            log_pmf[c][d - 1] = p[d - 1].max(1e-300).ln();
        }
    }
    Dwell { log_pmf, log_surv }
}

/// Truth run lengths per [`STAGE_ORDER`] column, over contiguous labelled blocks, dropping each
/// block's first and last run because both are censored by the block edge.
fn truth_runs(nights: &[&Night]) -> Vec<Vec<usize>> {
    fn flush(b: &[usize], out: &mut [Vec<usize>]) {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for &c in b {
            match runs.last_mut() {
                Some(l) if l.0 == c => l.1 += 1,
                _ => runs.push((c, 1)),
            }
        }
        if runs.len() >= 3 {
            for r in &runs[1..runs.len() - 1] {
                out[r.0].push(r.1);
            }
        }
    }
    let mut out = vec![Vec::new(); CLASSES];
    for nt in nights {
        let mut b: Vec<usize> = Vec::new();
        for t in &nt.truth {
            match t {
                Some(c) => b.push(col_of(*c)),
                None => {
                    flush(&b, &mut out);
                    b.clear();
                }
            }
        }
        flush(&b, &mut out);
    }
    out
}

/// Between-class log transition of the HSMM: the shipped row with its self-mass removed and the
/// rest renormalised, so the dwell law is the only thing saying how long a stage lasts.
fn hsmm_exit(t: &[[f64; CLASSES]; CLASSES]) -> [[f64; CLASSES]; CLASSES] {
    let mut a = [[f64::NEG_INFINITY; CLASSES]; CLASSES];
    for (q, row) in t.iter().enumerate() {
        let out_mass = 1.0 - row[q];
        for (c, v) in row.iter().enumerate() {
            if c != q {
                a[q][c] = v.max(FLOOR).ln() - out_mass.max(FLOOR).ln();
            }
        }
    }
    a
}

/// Segmental Viterbi over an explicit dwell law, uniform over the starting class.
///
/// `pmf_end` charges the completed-segment pmf on the last segment instead of the survival, which is
/// the naive end treatment; the survival is what makes this identical to the shipped HMM.
#[allow(clippy::needless_range_loop)]
fn hsmm_viterbi(
    em: &[[f64; CLASSES]],
    a: &[[f64; CLASSES]; CLASSES],
    dw: &Dwell,
    pmf_end: bool,
    st: &mut DecStats,
) -> Vec<usize> {
    let t_len = em.len();
    let neg = f64::NEG_INFINITY;
    let mut prefix = vec![vec![0.0f64; t_len + 1]; CLASSES];
    for c in 0..CLASSES {
        for t in 0..t_len {
            prefix[c][t + 1] = prefix[c][t] + em[t][c];
        }
    }
    // `into[u][c]` is the best score of a segmentation of [0, u) whose NEXT segment opens as c.
    let mut into = vec![[neg; CLASSES]; t_len + 1];
    let mut into_bp = vec![[0usize; CLASSES]; t_len + 1];
    into[0] = [0.0; CLASSES];
    let mut end = vec![[neg; CLASSES]; t_len + 1];
    let mut end_bp = vec![[(0usize, 0usize); CLASSES]; t_len + 1];
    for t in 1..=t_len {
        for c in 0..CLASSES {
            let (mut best, mut best_d) = (neg, 1usize);
            for d in 1..=t {
                let prev = into[t - d][c];
                if prev == neg {
                    continue;
                }
                let dur = if t == t_len && !pmf_end {
                    dw.log_surv[c][d - 1]
                } else {
                    dw.log_pmf[c][d - 1]
                };
                let s = prev + prefix[c][t] - prefix[c][t - d] + dur;
                if s > best {
                    best = s;
                    best_d = d;
                }
            }
            end[t][c] = best;
            end_bp[t][c] = (best_d, into_bp[t - best_d][c]);
            st.max_abs = st.max_abs.max(best.abs());
        }
        for c in 0..CLASSES {
            let (mut best, mut bq) = (neg, 0usize);
            for q in 0..CLASSES {
                let s = end[t][q] + a[q][c];
                if s > best {
                    best = s;
                    bq = q;
                }
            }
            into[t][c] = best;
            into_bp[t][c] = bq;
        }
    }
    let scores: Vec<f64> = (0..CLASSES).map(|c| end[t_len][c]).collect();
    let mut c = argmax(&scores, st);
    let mut t = t_len;
    let mut labels = vec![0usize; t_len];
    while t > 0 {
        let (d, prev) = end_bp[t][c];
        st.max_run = st.max_run.max(d);
        for u in (t - d)..t {
            labels[u] = c;
        }
        t -= d;
        c = prev;
    }
    labels
}

/// Decode one night under `arm`, returning class columns in [`STAGE_ORDER`].
fn decode(arm: Arm, em: &[[f64; CLASSES]], p: &Params, dw: &Dwell, st: &mut DecStats) -> Vec<usize> {
    match arm.dec {
        Dec::HsmmGeom | Dec::HsmmFit(_) => return hsmm_viterbi(em, &hsmm_exit(&p.transition), dw, false, st),
        Dec::HsmmGeomPmfEnd => return hsmm_viterbi(em, &hsmm_exit(&p.transition), dw, true, st),
        _ => {}
    }
    let log_t = log_chain(&p.transition, arm.k);
    let start = start_vec(arm.start, p, &log_t, arm.k);
    let xem = expand_em(em, arm.k);
    let path = match arm.dec {
        Dec::Posterior => posterior_n(&xem, &log_t, &start, st),
        _ => viterbi_n(&xem, &log_t, &start, st),
    };
    if arm.dec == Dec::Viterbi {
        for w in path.windows(2) {
            st.floor_edges += usize::from(log_t[w[0]][w[1]] <= FLOOR.ln());
        }
    }
    path.into_iter().map(|s| s / arm.k).collect()
}

// ---------------------------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------------------------

/// Everything one arm produces on one cohort: per-night kappa and the pooled shape descriptors.
struct ArmScore {
    kappa: Vec<f64>,
    runs: Vec<f64>,
    truth_runs: Vec<f64>,
    cm: Confusion4,
    /// Adjacent pairs whose class change is exactly zero in the shipped matrix.
    illegal: usize,
    pairs: usize,
    stats: DecStats,
    /// Epochs where this arm's label differs from the same-emission shipped baseline.
    differ: usize,
    epochs: usize,
}

fn runs_of(seq: &[usize]) -> f64 {
    1.0 + seq.windows(2).filter(|w| w[0] != w[1]).count() as f64
}

/// One arm's shape on one cohort, kept so the mechanism table can be printed after the kappa table.
struct Shape {
    runs: f64,
    truth_runs: f64,
    rem_recall: f64,
    illegal: usize,
    pairs: usize,
    stats: DecStats,
    differ: usize,
    epochs: usize,
}

fn score(
    nights: &[Night],
    ems: &[Vec<[f64; CLASSES]>],
    arm: Arm,
    p: &Params,
    dw: &Dwell,
    baseline: Option<&[Vec<usize>]>,
) -> (ArmScore, Vec<Vec<usize>>) {
    let mut out = ArmScore {
        kappa: Vec::new(),
        runs: Vec::new(),
        truth_runs: Vec::new(),
        cm: [[0i64; 4]; 4],
        illegal: 0,
        pairs: 0,
        stats: DecStats::default(),
        differ: 0,
        epochs: 0,
    };
    let mut paths = Vec::with_capacity(nights.len());
    for (i, nt) in nights.iter().enumerate() {
        let mut st = DecStats::default();
        let cols = decode(arm, &ems[i], p, dw, &mut st);
        // Legality is a statement about the SHIPPED 4x4, so it is read after any phase collapse.
        for w in cols.windows(2) {
            out.pairs += 1;
            out.illegal += usize::from(p.transition[w[0]][w[1]] == 0.0 && w[0] != w[1]);
        }
        let mut run = 1usize;
        for w in cols.windows(2) {
            run = if w[0] == w[1] { run + 1 } else { 1 };
            st.max_run = st.max_run.max(run);
        }
        out.stats.absorb(&st);
        let path: Vec<usize> = cols.iter().map(|c| stage_idx(STAGE_ORDER[*c])).collect();
        if let Some(b) = baseline {
            out.differ += path.iter().zip(&b[i]).filter(|(a, c)| a != c).count();
            out.epochs += path.len();
        }
        let (mut pp, mut tt) = (Vec::new(), Vec::new());
        for (k, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            pp.push(path[k]);
            tt.push(*want);
        }
        if tt.len() >= MIN_EPOCHS {
            let cm = confusion4(&pp, &tt);
            for (r, row) in cm.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    out.cm[r][c] += v;
                }
            }
            out.kappa.push(kappa4(&cm));
            out.runs.push(runs_of(&pp));
            out.truth_runs.push(runs_of(&tt));
        }
        paths.push(path);
    }
    (out, paths)
}

fn deltas(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn verdict(d: &[f64]) -> String {
    match paired_bar(d) {
        None => "  (n<2)".to_string(),
        Some((m, bar)) => {
            let tag = if m.abs() > bar {
                if m > 0.0 {
                    format!("BEATS v2 ({:.2}x)", m / bar)
                } else {
                    format!("worse ({:.2}x)", -m / bar)
                }
            } else {
                "matches".to_string()
            };
            let up = d.iter().filter(|x| **x > 0.0).count();
            let dn = d.iter().filter(|x| **x < 0.0).count();
            format!("{m:>+9.4} {bar:>8.4}  {up:>3}+/{dn:<3}-  {tag}")
        }
    }
}

fn mean_delta(d: &[f64]) -> f64 {
    if d.is_empty() {
        f64::NAN
    } else {
        d.iter().sum::<f64>() / d.len() as f64
    }
}

// ---------------------------------------------------------------------------------------------
// The tanv1 correction, reproduced from `fit_residual` so the second arm is the measured one
// ---------------------------------------------------------------------------------------------

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

/// Multinomial logistic regression on top of the shipped emission as a fixed offset; theta starts at
/// zero, which IS the shipped recipe.
fn fit_residual(x: &[Vec<f64>], off: &[[f64; CLASSES]], y: &[usize]) -> (Vec<Vec<f64>>, bool) {
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
    (th, converged)
}

/// Fit the correction on `train` and return the corrected emissions of `held`.
fn tanv1_emissions(train: &[&Night], held: &[Night]) -> (Vec<Vec<[f64; CLASSES]>>, bool) {
    let rows: Vec<(Vec<f64>, [f64; CLASSES], usize)> = train
        .iter()
        .flat_map(|nt| {
            nt.row
                .iter()
                .zip(&nt.em)
                .zip(&nt.truth)
                .filter_map(|((r, o), t)| t.map(|t| (r.clone(), *o, t)))
        })
        .collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|(r, _, _)| r.clone()).collect();
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    let off: Vec<[f64; CLASSES]> = rows.iter().map(|(_, o, _)| *o).collect();
    let y: Vec<usize> = rows.iter().map(|(_, _, t)| *t).collect();
    let (th, conv) = fit_residual(&dx, &off, &y);
    let out = held
        .iter()
        .map(|nt| {
            nt.row
                .iter()
                .zip(&nt.em)
                .map(|(full, o)| {
                    let d = design_row(full, &m, &sd, &[]);
                    let mut e = *o;
                    for c in 0..CLASSES {
                        e[col_of(c)] += th[c].iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
                    }
                    e
                })
                .collect()
        })
        .collect();
    (out, conv)
}

// ---------------------------------------------------------------------------------------------

fn arms() -> Vec<Arm> {
    let mut v = Vec::new();
    for dec in [Dec::Viterbi, Dec::Posterior] {
        for start in [Start::Uniform, Start::Stationary, Start::BaseRate] {
            v.push(Arm { dec, start, k: 1 });
        }
        for k in KS {
            v.push(Arm { dec, start: Start::Uniform, k });
        }
    }
    v.push(Arm { dec: Dec::HsmmGeom, start: Start::Uniform, k: 1 });
    v.push(Arm { dec: Dec::HsmmGeomPmfEnd, start: Start::Uniform, k: 1 });
    for l in LAMBDAS {
        v.push(Arm { dec: Dec::HsmmFit(l), start: Start::Uniform, k: 1 });
    }
    v
}

fn shipped_ems(nights: &[Night]) -> Vec<Vec<[f64; CLASSES]>> {
    nights.iter().map(|n| n.em.clone()).collect()
}

/// The dwell law an arm needs, estimated on `train` when the arm is a fitted one.
fn dwell_for(arm: Arm, p: &Params, train: &[&Night]) -> Dwell {
    match arm.dec {
        Dec::HsmmFit(l) => dwell_fitted(&truth_runs(train), &p.transition, l),
        _ => dwell_geometric(&p.transition),
    }
}

fn main() {
    let p = Params::SHIPPED;
    println!("STEP 5 - the DECODER. Viterbi maximises the PATH; kappa scores the EPOCH.\n");

    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    assert!(loaded.len() == 3, "all three cohorts must load or the folds are not the reported ones");
    let geom = dwell_geometric(&p.transition);

    // ---- Positive controls: both reproductions must BE the shipped decoder ----
    let (mut checked, mut hsmm_same, mut hsmm_tot) = (0usize, 0usize, 0usize);
    for (name, nights) in &loaded {
        for nt in nights {
            let mut st = DecStats::default();
            let mine = decode(BASE, &nt.em, &p, &geom, &mut st);
            let shipped: Vec<usize> = decode_v2(&nt.em, &p.transition)
                .iter()
                .map(|s| STAGE_ORDER.iter().position(|x| x == s).expect("stage in STAGE_ORDER"))
                .collect();
            assert_eq!(mine, shipped, "{name}: the reproduction is not decode_v2");
            let h = decode(Arm { dec: Dec::HsmmGeom, start: Start::Uniform, k: 1 }, &nt.em, &p, &geom, &mut st);
            hsmm_same += h.iter().zip(&shipped).filter(|(a, b)| a == b).count();
            hsmm_tot += shipped.len();
            checked += 1;
        }
    }
    let hsmm_agree = 100.0 * hsmm_same as f64 / hsmm_tot as f64;
    println!("  CONTROL 1: viterbi/uniform/k1 == decode_v2 on all {checked} nights");
    println!("  CONTROL 2: hsmm/geometric agrees with decode_v2 on {hsmm_agree:.4}% of {hsmm_tot} epochs");
    assert!(hsmm_agree > 99.9, "the HSMM under the shipped geometric dwell is not the shipped HMM");
    println!();

    // ---- The transition contract ----
    let zeros: Vec<String> = (0..CLASSES)
        .flat_map(|i| (0..CLASSES).map(move |j| (i, j)))
        .filter(|(i, j)| p.transition[*i][*j] == 0.0)
        .map(|(i, j)| {
            format!("{}->{}", CLASS_NAMES[stage_idx(STAGE_ORDER[i])], CLASS_NAMES[stage_idx(STAGE_ORDER[j])])
        })
        .collect();
    let st4 = stationary(&log_chain(&p.transition, 1));
    println!("  the transition contract");
    println!("    hard zeros (floored to 1e-9, ln = {:.2}): {}", FLOOR.ln(), zeros.join(", "));
    println!(
        "    cheapest stage change: {:.3} nats",
        (0..CLASSES)
            .map(|i| (0..CLASSES)
                .filter(|j| *j != i)
                .map(|j| p.transition[i][i].ln() - p.transition[i][j].max(FLOOR).ln())
                .fold(f64::INFINITY, f64::min))
            .fold(f64::INFINITY, f64::min)
    );
    print!("    stationary pi   ");
    for c in 0..CLASSES {
        print!("{}={:.3} ", CLASS_NAMES[stage_idx(STAGE_ORDER[c])], st4[c]);
    }
    println!();
    print!("    base_rate       ");
    for c in 0..CLASSES {
        print!("{}={:.3} ", CLASS_NAMES[stage_idx(STAGE_ORDER[c])], p.base_rate[c]);
    }
    println!("(sums to {:.3}; already inside EVERY emission)", p.base_rate.iter().sum::<f64>());
    let all_nights: Vec<&Night> = loaded.iter().flat_map(|(_, n)| n.iter()).collect();
    let tr = truth_runs(&all_nights);
    print!("    truth mean dwell");
    for c in 0..CLASSES {
        let m = tr[c].iter().sum::<usize>() as f64 / tr[c].len().max(1) as f64;
        print!(" {}={:.1}ep(n={})", CLASS_NAMES[stage_idx(STAGE_ORDER[c])], m, tr[c].len());
    }
    println!();
    print!("    shipped geometric mean dwell");
    for c in 0..CLASSES {
        print!(" {}={:.1}ep", CLASS_NAMES[stage_idx(STAGE_ORDER[c])], 1.0 / (1.0 - p.transition[c][c]));
    }
    println!("\n");

    // ---- Table A: every arm on v2's own emissions, per cohort ----
    println!("A. v2's OWN emissions, every decoder arm, paired per night against viterbi/uniform/k1");
    println!(
        "  {:<22} {:<17} {:>6}  {:>9} {:>8} {:>10}  verdict",
        "arm", "cohort", "median", "paired d", "bar +/-", "nights"
    );
    let mut base_paths: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut base_kappa: Vec<Vec<f64>> = Vec::new();
    for (_, nights) in &loaded {
        let (s, paths) = score(nights, &shipped_ems(nights), BASE, &p, &geom, None);
        base_kappa.push(s.kappa);
        base_paths.push(paths);
    }
    let all = arms();
    // `grid[arm][cohort]` under the FULL-train dwell, which is what the outer report uses.
    let mut grid: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut shape: Vec<Vec<Shape>> = Vec::new();
    for arm in &all {
        let mut row = Vec::new();
        let mut srow = Vec::new();
        for (ci, (name, nights)) in loaded.iter().enumerate() {
            let train: Vec<&Night> =
                loaded.iter().enumerate().filter(|(i, _)| *i != ci).flat_map(|(_, (_, n))| n.iter()).collect();
            let dw = dwell_for(*arm, &p, &train);
            let (s, _) = score(nights, &shipped_ems(nights), *arm, &p, &dw, Some(&base_paths[ci]));
            let d = deltas(&s.kappa, &base_kappa[ci]);
            println!(
                "  {:<22} {:<17} {:>6.3}  {}",
                arm.name(),
                format!("{name} n={}", s.kappa.len()),
                median(&mut s.kappa.clone()),
                verdict(&d)
            );
            srow.push(Shape {
                runs: median(&mut s.runs.clone()),
                truth_runs: median(&mut s.truth_runs.clone()),
                rem_recall: recall(&s.cm, 3).unwrap_or(f64::NAN),
                illegal: s.illegal,
                pairs: s.pairs,
                stats: s.stats,
                differ: s.differ,
                epochs: s.epochs,
            });
            row.push(d);
        }
        shape.push(srow);
        grid.push(row);
    }
    println!();

    // ---- Table B: the mechanism ----
    println!("B. shape of what each arm draws (pooled over the three cohorts)");
    println!(
        "  {:<22} {:>8} {:>7} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "arm", "runs/nt", "truth", "REM rec", "illegal", "differ%", "margin", "maxrun", "ties"
    );
    for (ai, arm) in all.iter().enumerate() {
        let s = &shape[ai];
        let runs = s.iter().map(|x| x.runs).sum::<f64>() / s.len() as f64;
        let trn = s.iter().map(|x| x.truth_runs).sum::<f64>() / s.len() as f64;
        let rem = s.iter().map(|x| x.rem_recall).sum::<f64>() / s.len() as f64;
        let ill = s.iter().map(|x| x.illegal).sum::<usize>();
        let dif = 100.0 * s.iter().map(|x| x.differ).sum::<usize>() as f64
            / s.iter().map(|x| x.epochs).sum::<usize>().max(1) as f64;
        let mut st = DecStats::default();
        for x in s {
            st.absorb(&x.stats);
        }
        let conf = if st.margin_n > 0 { format!("{:.3}", st.margin()) } else { "-".to_string() };
        println!(
            "  {:<22} {runs:>8.1} {trn:>7.1} {rem:>8.3} {ill:>9} {dif:>8.2} {conf:>8} {:>8} {:>8}",
            arm.name(),
            st.max_run,
            st.pred_ties + st.decide_ties
        );
    }
    println!();

    // ---- Why the Erlang expansion fails: it does not hold the stage-change charge fixed ----
    println!("B2. what the k-phase expansion does to the charge for LEAVING a stage, in nats");
    println!("  {:<6} {:<8} {:>8} {:>10} {:>14}", "k", "class", "self", "advance", "cheapest exit");
    for k in [1usize, KS[0], KS[1]] {
        for c in 0..CLASSES {
            let stay = 1.0 - (k as f64) * (1.0 - p.transition[c][c]);
            let leave = 1.0 - stay;
            let out_mass = 1.0 - p.transition[c][c];
            let cheapest = (0..CLASSES)
                .filter(|d| *d != c)
                .map(|d| leave * p.transition[c][d] / out_mass)
                .fold(0.0f64, f64::max);
            let adv = if k > 1 { format!("{:>10.3}", stay.ln() - leave.ln()) } else { "-".to_string() };
            println!(
                "  {k:<6} {:<8} {:>8.3} {adv:>10} {:>14.3}",
                CLASS_NAMES[stage_idx(STAGE_ORDER[c])],
                stay,
                stay.ln() - cheapest.ln()
            );
        }
    }
    println!("  A negative advance cost means racing to the last phase is CHEAPER than staying, so");
    println!("  the last phase's exit becomes the effective charge and the expansion un-sticks.\n");

    // ---- Numerics ----
    let mut tot = DecStats::default();
    for srow in &shape {
        for x in srow {
            tot.absorb(&x.stats);
        }
    }
    println!("C. numerics, ties and the floor");
    println!("    largest |accumulator| reached        {:.1}", tot.max_abs);
    println!("    f64 spacing there                    {:.2e}", f64::EPSILON * tot.max_abs);
    println!("    non-finite alpha / beta / score      {}", tot.non_finite);
    println!("    exact ties in a per-epoch decision   {}", tot.decide_ties);
    println!("    exact ties between predecessors      {}", tot.pred_ties);
    let floor_arm: usize = shape[0].iter().map(|x| x.stats.floor_edges).sum();
    let floor_pairs: usize = shape[0].iter().map(|x| x.pairs).sum();
    println!("    shipped path edges ON the 1e-9 floor {floor_arm} of {floor_pairs}");
    println!();

    // ---- Pooling all 144 nights: the cheapest control that could take the negative away ----
    println!("C2. all 144 nights POOLED into one paired test (a secondary: it mixes three populations)");
    for (ai, arm) in all.iter().enumerate() {
        if *arm == BASE {
            continue;
        }
        let pooled: Vec<f64> = grid[ai].iter().flatten().copied().collect();
        println!("  {:<22} {}", arm.name(), verdict(&pooled));
    }
    println!();

    // ---- Per-class recall, and where the posterior's disagreements sit ----
    println!("D0. per-class recall against truth, pooled over the three cohorts");
    print!("  {:<22}", "arm");
    for n in CLASS_NAMES {
        print!(" {n:>8}");
    }
    println!();
    for arm in all.iter() {
        let mut cm = [[0i64; 4]; 4];
        for (ci, (_, nights)) in loaded.iter().enumerate() {
            let train: Vec<&Night> =
                loaded.iter().enumerate().filter(|(i, _)| *i != ci).flat_map(|(_, (_, n))| n.iter()).collect();
            let dw = dwell_for(*arm, &p, &train);
            let (s, _) = score(nights, &shipped_ems(nights), *arm, &p, &dw, None);
            for (r, row) in s.cm.iter().enumerate() {
                for (c, v) in row.iter().enumerate() {
                    cm[r][c] += v;
                }
            }
        }
        print!("  {:<22}", arm.name());
        for c in 0..CLASSES {
            print!(" {:>8.3}", recall(&cm, c).unwrap_or(f64::NAN));
        }
        println!();
    }
    println!();

    println!("D1. the posterior's confidence where it agrees with the shipped path and where it does not");
    println!("  {:<14} {:>10} {:>12} {:>12}", "cohort", "disagree%", "conf agree", "conf differ");
    let log_t1 = log_chain(&p.transition, 1);
    let start1 = start_vec(Start::Uniform, &p, &log_t1, 1);
    for (ci, (name, nights)) in loaded.iter().enumerate() {
        let (mut ag, mut ag_n, mut df, mut df_n) = (0.0f64, 0usize, 0.0f64, 0usize);
        for (i, nt) in nights.iter().enumerate() {
            let mut st = DecStats::default();
            let g = forward_backward(&expand_em(&nt.em, 1), &log_t1, &start1, &mut st);
            for (t, row) in g.iter().enumerate() {
                let lab = stage_idx(STAGE_ORDER[argmax(row, &mut st)]);
                let conf = row.iter().copied().fold(0.0f64, f64::max);
                if lab == base_paths[ci][i][t] {
                    ag += conf;
                    ag_n += 1;
                } else {
                    df += conf;
                    df_n += 1;
                }
            }
        }
        println!(
            "  {:<14} {:>10.2} {:>12.3} {:>12.3}",
            name,
            100.0 * df_n as f64 / (ag_n + df_n).max(1) as f64,
            ag / ag_n.max(1) as f64,
            df / df_n.max(1) as f64
        );
    }
    println!();

    // ---- Table D: selection. Inner LOO over the two TRAIN cohorts, reported once on held-out ----
    println!("D. SELECTED on an inner LOO over the two TRAIN cohorts, then reported once on held-out");
    println!("  {:<14} {:<22} {:>9}  {:>9} {:>8} {:>10}  verdict", "held-out", "selected arm", "inner d", "paired d", "bar +/-", "nights");
    for (hi, (held, _)) in loaded.iter().enumerate() {
        let inner: Vec<usize> = (0..loaded.len()).filter(|i| *i != hi).collect();
        let mut best = (0usize, f64::NEG_INFINITY);
        for (ai, arm) in all.iter().enumerate() {
            // An inner fold validates on one train cohort with the dwell fitted on the OTHER.
            let s: f64 = inner
                .iter()
                .map(|vi| {
                    if !arm.fitted() {
                        return mean_delta(&grid[ai][*vi]);
                    }
                    let it: Vec<&Night> = inner
                        .iter()
                        .filter(|i| *i != vi)
                        .flat_map(|i| loaded[*i].1.iter())
                        .collect();
                    let dw = dwell_for(*arm, &p, &it);
                    let (sc, _) = score(&loaded[*vi].1, &shipped_ems(&loaded[*vi].1), *arm, &p, &dw, None);
                    mean_delta(&deltas(&sc.kappa, &base_kappa[*vi]))
                })
                .sum::<f64>()
                / inner.len() as f64;
            if s > best.1 {
                best = (ai, s);
            }
        }
        let d = &grid[best.0][hi];
        println!("  {:<14} {:<22} {:>+9.4}  {}", held, all[best.0].name(), best.1, verdict(d));
    }
    println!();

    // ---- Table E: the same decoders on tanv1's emissions ----
    println!("E. tanv1's emissions (L2 {L2}, correction fit on the two TRAIN cohorts of each fold)");
    println!("  {:<22} {:<17} {:>6}  {:>9} {:>8} {:>10}  verdict", "arm", "held-out", "median", "paired d", "bar +/-", "nights");
    for (hi, (held, nights)) in loaded.iter().enumerate() {
        let train: Vec<&Night> =
            loaded.iter().enumerate().filter(|(i, _)| *i != hi).flat_map(|(_, (_, n))| n.iter()).collect();
        let (ems, conv) = tanv1_emissions(&train, nights);
        assert!(conv, "{held}: the correction did not converge, so the arms are not comparable");
        let mut vit = Vec::new();
        for arm in [BASE, POST] {
            let (s, _) = score(nights, &ems, arm, &p, &geom, None);
            let d = deltas(&s.kappa, &base_kappa[hi]);
            println!(
                "  tanv1 {:<16} {:<17} {:>6.3}  {}",
                arm.name(),
                format!("{held} n={}", s.kappa.len()),
                median(&mut s.kappa.clone()),
                verdict(&d)
            );
            vit.push(s.kappa);
        }
        // The decoder swap on its own, both arms on tanv1's emission: does posterior rescue it?
        let d = deltas(&vit[1], &vit[0]);
        println!("  {:<22} {:<17} {:>6}  {}", "  posterior - viterbi", held, "", verdict(&d));
    }
    println!("\n  Every delta in A, D and E is against the SAME baseline: v2 emissions,");
    println!("  viterbi/uniform/k1, night for night. The last row of each E block is the two");
    println!("  tanv1 arms against EACH OTHER, which is the decoder swap with the emission held.");
}
