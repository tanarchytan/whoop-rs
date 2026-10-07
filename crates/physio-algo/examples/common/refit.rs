//! Refit v2's twelve emission weights (and optionally extra AWAKE-row columns) by weighted
//! multinomial descent through [`Terms::emission`], so every non-linearity, the cycle prior, the gate
//! and the clamp stay exactly as shipped.
//!
//! This is `fair_fight`'s fitter, generalised: the objective, the class weighting, the clamp-aware
//! gradient, the step rule and the stopping rule are the same. What is new is `k` extra columns added
//! straight into the AWAKE row (no deadzone, no clamp), and the gradient being summed over a FIXED
//! number of night chunks on scoped threads. The chunking is a function of the night count and never
//! of the machine, so the fit is the same object on any core count.
//!
//! With `k = 0` the weights are `fair_fight`'s, up to float summation order.

use std::thread;

use std::collections::HashMap;

use physio_algo::sleep::{params::Params, resp_regularity, weights_of, SleepInput, Terms, STAGE_ORDER, WEIGHT_NAMES};
use physio_algo::sleep::SleepStage;

use super::stage_idx;

pub const NW: usize = 12;
pub const CLASSES: usize = 4;
pub const WEIGHT_POWER: f64 = 0.5;
/// Starting step. Halved whenever the objective rises, so the fit cannot oscillate past its own
/// optimum and report a cap-hit as a failure to converge.
const LR0: f64 = 1.0;
pub const ITERS: usize = 40_000;
const TOL: f64 = 1e-11;
/// Gradient chunks per iteration. Fixed, so the float summation order does not depend on the host.
const CHUNKS: usize = 16;

/// Slot in the weight vector a name owns. Looked up, never assumed.
pub fn slot(name: &str) -> usize {
    WEIGHT_NAMES.iter().position(|w| *w == name).unwrap_or_else(|| panic!("{name} is not in WEIGHT_NAMES"))
}

/// The emission column a truth class (`stage_idx` code) sits in.
pub fn col_of(class: usize) -> usize {
    (0..CLASSES).find(|c| stage_idx(STAGE_ORDER[*c]) == class).expect("class in STAGE_ORDER")
}

/// The emission column of the AWAKE row.
pub fn awake_col() -> usize {
    STAGE_ORDER.iter().position(|s| *s == SleepStage::Wake).expect("wake in STAGE_ORDER")
}

/// One recording as the fitter sees it. `extra` is `k` values per epoch, epoch-major.
pub struct FitNight<'a> {
    pub terms: &'a Terms,
    pub extra: &'a [f64],
    pub truth: &'a [Option<usize>],
    /// Weight of every labelled epoch of this recording. 1.0 is the unweighted fit, bit for bit.
    pub weight: f64,
}

#[derive(Clone)]
pub struct Fitted {
    /// `NW` v2 weights, then the `k` extra AWAKE-row weights.
    pub w: Vec<f64>,
    pub iters: usize,
    pub converged: bool,
}

/// One epoch's emission: v2's, plus each extra column times its weight added to AWAKE.
pub fn emission(terms: &Terms, e: usize, extra_row: &[f64], w: &[f64]) -> [f64; CLASSES] {
    let mut em = terms.emission(e, w[..NW].try_into().expect("NW weights"));
    for (x, wi) in extra_row.iter().zip(&w[NW..]) {
        em[awake_col()] += wi * x;
    }
    em
}

/// Inverse-prevalence class weights over the WEIGHTED epoch counts (all weights 1.0 gives the plain counts).
fn class_weights(nights: &[FitNight]) -> [f64; CLASSES] {
    let mut n = [0.0f64; CLASSES];
    let mut total = 0.0f64;
    for nt in nights {
        for c in nt.truth.iter().flatten() {
            n[*c] += nt.weight;
            total += nt.weight;
        }
    }
    let mut w = [1.0f64; CLASSES];
    for c in 0..CLASSES {
        w[c] = if n[c] > 0.0 { (total / (CLASSES as f64 * n[c])).powf(WEIGHT_POWER) } else { 0.0 };
    }
    let mass: f64 = (0..CLASSES).map(|c| n[c] * w[c]).sum::<f64>() / total;
    for v in w.iter_mut() {
        *v /= mass;
    }
    w
}

/// Weighted NLL and its gradient over one chunk of nights.
fn chunk_grad(nights: &[FitNight], k: usize, w: &[f64], cw: &[f64; CLASSES]) -> (Vec<f64>, f64, f64) {
    let mut g = vec![0.0f64; NW + k];
    let (mut nll, mut n) = (0.0f64, 0.0f64);
    let (hrv, hr, aw) = (slot("awake_hrv"), slot("awake_hr"), awake_col());
    for nt in nights {
        for (e, want) in nt.truth.iter().enumerate() {
            let Some(want) = want else { continue };
            let xr = &nt.extra[e * k..(e + 1) * k];
            let em = emission(nt.terms, e, xr, w);
            let mx = em.iter().cloned().fold(f64::MIN, f64::max);
            let ex: [f64; CLASSES] = std::array::from_fn(|c| (em[c] - mx).exp());
            let sum: f64 = ex.iter().sum();
            let col = col_of(*want);
            nll -= cw[*want] * nt.weight * (ex[col] / sum).max(1e-300).ln();
            n += nt.weight;
            // d(loss)/d(w_j) = sum_c (p_c - 1{c=y}) * d(em_c)/d(w_j), and d(em_c)/d(w_j) is the design
            // cell, except where the awake clamp has zeroed the cardiac pair's gradient.
            let d = &nt.terms.design[e];
            for c in 0..CLASSES {
                let r = cw[*want] * nt.weight * (ex[c] / sum - if c == col { 1.0 } else { 0.0 });
                for j in 0..NW {
                    let mut cell = d[c][j];
                    if cell == 0.0 {
                        continue;
                    }
                    if c == aw && (j == hrv || j == hr) && nt.terms.clamped[e] {
                        let card = w[hrv] * d[c][hrv] + w[hr] * d[c][hr];
                        if card > 0.0 {
                            cell = 0.0;
                        }
                    }
                    g[j] += r * cell;
                }
                if c == aw {
                    for (i, x) in xr.iter().enumerate() {
                        g[NW + i] += r * x;
                    }
                }
            }
        }
    }
    (g, nll, n)
}

/// Fit on `nights`, warm-started from `Params::SHIPPED` with the extras at zero. Does NOT panic on a
/// cap-hit: the caller reads `converged`, so a batch of folds reports rather than dies.
pub fn fit(nights: &[FitNight], k: usize) -> Fitted {
    let cw = class_weights(nights);
    let mut w = weights_of(&Params::SHIPPED).to_vec();
    w.extend(std::iter::repeat_n(0.0, k));
    let size = nights.len().div_ceil(CHUNKS).max(1);
    let (mut last, mut lr, mut converged, mut iters) = (f64::MAX, LR0, false, 0);
    for it in 0..ITERS {
        iters = it + 1;
        let parts: Vec<(Vec<f64>, f64, f64)> = thread::scope(|s| {
            let hs: Vec<_> = nights
                .chunks(size)
                .map(|c| {
                    let (w, cw) = (&w, &cw);
                    s.spawn(move || chunk_grad(c, k, w, cw))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("gradient thread")).collect()
        });
        let mut g = vec![0.0f64; NW + k];
        let (mut nll, mut n) = (0.0f64, 0.0f64);
        for (pg, pn, pc) in &parts {
            for (a, b) in g.iter_mut().zip(pg) {
                *a += b;
            }
            nll += pn;
            n += pc;
        }
        let nll = nll / n;
        let drop = last - nll;
        if (0.0..TOL).contains(&drop) {
            converged = true;
            break;
        }
        // A rise means the step overshot; back it off rather than letting it ring.
        if drop < 0.0 {
            lr *= 0.5;
        }
        last = nll;
        for (wj, gj) in w.iter_mut().zip(&g) {
            *wj -= lr * gj / n;
        }
    }
    Fitted { w, iters, converged }
}

/// Beats of the seconds `lo..hi`, spread by interval: the reconstruction of `v2::beats_in` (private),
/// as `quiet_wake_dump` pins it.
fn beats_in(rr_by: &HashMap<i64, Vec<f64>>, lo: i64, hi: i64) -> Vec<(f64, f64)> {
    let mut beats = Vec::new();
    for bs in lo..hi {
        let Some(vs) = rr_by.get(&bs) else { continue };
        let mut off = 0.0;
        for (k, v) in vs.iter().enumerate() {
            let ms = v.clamp(300.0, 2000.0);
            if k > 0 {
                off += ms / 1000.0;
            }
            beats.push((bs as f64 + off, ms));
        }
    }
    beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    beats
}

/// Per epoch start: does the epoch's R-R window give v2 a respiratory regularity (`resp_reg.is_some()`,
/// v2's own `rr_backed` under `clamp_only_without_rr`)? Same computation as `quiet_wake_dump`'s `has_rr`.
pub fn rr_backed(input: &SleepInput, starts: &[i64]) -> Vec<bool> {
    let mut rr_by: HashMap<i64, Vec<f64>> = HashMap::new();
    for run in &input.rr {
        for &ms in &run.intervals {
            rr_by.entry(run.ts).or_default().push(ms as f64);
        }
    }
    starts.iter().map(|&s| resp_regularity(&beats_in(&rr_by, s - 90, s + 120)).is_some()).collect()
}
