//! The two levers whose effect could exceed this corpus's noise floor, both against the HONEST v2.
//!
//!   cargo run --release -p physio-algo --example tanv1_next
//!
//! `tanv1_tune` left tanv1 level with v2 refitted under the same leave-one-cohort-out rule, and the
//! post-hoc knobs bought nothing: the gap (0.005-0.033) sits inside the bars (0.022-0.048), so this
//! corpus cannot resolve superior from equal. Only a lever with a LARGER effect can settle it.
//!
//!   SPECTRAL   frequency-domain HRV, already implemented in `sleep::hrv_bands` and wired into
//!              nothing. Its own header names it "the feature family every published stager above
//!              kappa 0.5 has and this one has none of". sleep-accel carries ZERO R-R (all 31
//!              `rr.csv` are empty), so this arm can only move dreamt and aauwss and that is
//!              reported per cohort rather than pooled.
//!   PERMUTED   the null SPECTRAL needs. The same four columns with their 4-vectors shuffled among
//!              the epochs that have one, so marginals and missingness are untouched and only the
//!              alignment with truth is gone. Without it, "these features carry nothing" and "four
//!              more columns cost about 0.03 whatever is in them" predict the same table.
//!   PER-NIGHT  the same four columns with the three LOG POWERS centred on the night's median and
//!              divided by its 5-95 spread. Absolute band power varies by an order of magnitude
//!              between people, and a pooled standardiser cannot remove a per-subject offset.
//!              `hrv_bands` says so in its own header; this tests whether that is the defect.
//!   ABSTAIN    refuse the least-certain epochs. Measured at +0.055 elsewhere, roughly three times
//!              the current gap. Scored at MATCHED COVERAGE against two nulls, because dropping
//!              epochs at random also raises kappa on what is kept.
//!
//! Both arms are reported against V2 REFIT, v2's own twelve weights refitted on the two TRAIN
//! cohorts. Every line goes through `common::compare`, which will not format without stating the
//! baseline's provenance, and every cohort passes `require_psg` first.

mod common;

use common::lr::{design_row, fit as fit_lr, scores, standardise_cols};
use common::{
    cardiac_series, compare, dirs_of, median, pct, read_accel, read_hr, read_meta, read_rr,
    read_truth, require_psg, stage_idx, Provenance,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::hrv_bands::bands_series;
use physio_algo::sleep::metrics::{confusion4, kappa4};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, params::Params, prepare_v2, weights_of, SleepInput,
    Terms, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
const MIN_EPOCHS: usize = 20;
const WEIGHT_POWER: f64 = 0.5;
const NW: usize = 12;
const LR0: f64 = 1.0;
const V2_ITERS: usize = 40_000;
const V2_TOL: f64 = 1e-11;
/// Spectral columns appended to tanv1: log VLF, log LF, log HF and LF/(LF+HF).
const N_SPEC: usize = 4;
/// Fractions of labelled epochs KEPT by the abstention arms.
const COVERAGE: [f64; 3] = [1.00, 0.90, 0.80];

struct Night {
    tan: Vec<Vec<f64>>,
    spec: Vec<[f64; N_SPEC]>,
    /// `spec` shuffled within the night. The arm that prices four columns carrying nothing.
    perm: Vec<[f64; N_SPEC]>,
    /// `spec` with the three log powers scaled within the night.
    norm: Vec<[f64; N_SPEC]>,
    terms: Terms,
    truth: Vec<Option<usize>>,
}

/// Which spectral columns an arm carries.
#[derive(Clone, Copy, PartialEq)]
enum Spec {
    Off,
    Real,
    /// Same columns, same marginals, same missingness - only the alignment with truth is gone.
    Permuted,
    /// Same columns with the three log powers centred and scaled within the night.
    Normalised,
}

fn load(set: &str) -> Vec<Night> {
    require_psg(set);
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
        // (time, rr_ms) as `bands_series` wants it, flattened from the strap's per-second runs.
        let beats: Vec<(f64, f64)> = rr
            .iter()
            .flat_map(|r| r.intervals.iter().map(move |ms| (r.ts as f64, f64::from(*ms))))
            .collect();
        let bands = bands_series(&beats, w0 as f64, (w0 + n as i64 * EPOCH) as f64, EPOCH as f64);
        let input = SleepInput { start: w0, end: w1, hr, rr, accel };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let em = emissions_v2(&prep, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        if em.len() < MIN_EPOCHS {
            continue;
        }
        assert_eq!(em.len(), n, "{}: {n} epochs of truth against {} of emissions",
                   dir.display(), em.len());
        // A window that cannot carry a spectrum is NaN, not zero: `design_row` then abstains for it.
        let spec: Vec<[f64; N_SPEC]> = (0..em.len())
            .map(|e| match bands.get(e).and_then(|b| *b) {
                Some(b) => [
                    (b.vlf.max(1e-12)).ln(),
                    (b.lf.max(1e-12)).ln(),
                    (b.hf.max(1e-12)).ln(),
                    b.lf_nu.unwrap_or(f64::NAN),
                ],
                None => [f64::NAN; N_SPEC],
            })
            .collect();
        // Salted by cohort and by position, so the shuffle is fixed for a given fixture tree and two
        // nights never share one.
        let salt = set.bytes().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize))
            .wrapping_add(out.len());
        let perm = permute_spec(&spec, salt);
        let norm = normalise_spec(&spec);
        out.push(Night {
            tan: (0..em.len()).map(|e| f[e].values().to_vec()).collect(),
            spec,
            perm,
            norm,
            terms,
            truth: (0..em.len())
                .map(|k| {
                    raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize)
                })
                .collect(),
        });
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

/// v2's twelve weights refitted through its own decomposition: the honest opponent.
fn refit_v2(nights: &[&Night]) -> [f64; NW] {
    let y: Vec<usize> = nights.iter().flat_map(|n| n.truth.iter().flatten().copied()).collect();
    let cw = class_weights(&y);
    let mut w = weights_of(&Params::SHIPPED);
    let (mut last, mut lr, mut converged) = (f64::MAX, LR0, false);
    for _ in 0..V2_ITERS {
        let mut g = [0.0f64; NW];
        let (mut nll, mut n) = (0.0f64, 0.0f64);
        for nt in nights {
            for (e, want) in nt.truth.iter().enumerate() {
                let Some(want) = want else { continue };
                let em = nt.terms.emission(e, &w);
                let mx = em.iter().cloned().fold(f64::MIN, f64::max);
                let ex: Vec<f64> = em.iter().map(|v| (v - mx).exp()).collect();
                let sum: f64 = ex.iter().sum();
                let col = col_of(*want);
                nll -= cw[*want] * (ex[col] / sum).max(1e-300).ln();
                n += 1.0;
                for (j, gj) in g.iter_mut().enumerate() {
                    for (c, exc) in ex.iter().enumerate() {
                        let mut d = nt.terms.design[e][c][j];
                        if c == col_of(0) && (j == 8 || j == 9) && nt.terms.clamped[e] {
                            let card =
                                w[8] * nt.terms.design[e][c][8] + w[9] * nt.terms.design[e][c][9];
                            if card > 0.0 {
                                d = 0.0;
                            }
                        }
                        *gj += cw[*want] * (exc / sum - if c == col { 1.0 } else { 0.0 }) * d;
                    }
                }
            }
        }
        let nll = nll / n;
        let drop = last - nll;
        if (0.0..V2_TOL).contains(&drop) {
            converged = true;
            break;
        }
        if drop < 0.0 {
            lr *= 0.5;
        }
        last = nll;
        for (j, gj) in g.iter().enumerate() {
            w[j] -= lr * gj / n;
        }
    }
    assert!(converged, "the v2 refit hit its cap; the arms are not comparable");
    w
}

fn row(nt: &Night, e: usize, spec: Spec) -> Vec<f64> {
    let mut v = nt.tan[e].clone();
    match spec {
        Spec::Off => {}
        Spec::Real => v.extend_from_slice(&nt.spec[e]),
        Spec::Permuted => v.extend_from_slice(&nt.perm[e]),
        Spec::Normalised => v.extend_from_slice(&nt.norm[e]),
    }
    v
}

/// Shuffle the spectral 4-vectors among the epochs that HAVE one, moving each as a unit so the four
/// columns keep their joint correlation and lose only their alignment with truth. Epochs without a
/// spectrum keep their NaNs, so the missingness pattern the design row reads is untouched.
fn permute_spec(spec: &[[f64; N_SPEC]], salt: usize) -> Vec<[f64; N_SPEC]> {
    let mut out = spec.to_vec();
    let src: Vec<usize> = (0..spec.len()).filter(|i| spec[*i][0].is_finite()).collect();
    let key = pseudo_random(spec.len(), salt);
    let mut dst = src.clone();
    dst.sort_by(|a, b| key[*a].total_cmp(&key[*b]));
    assert!(src.len() < 2 || src.iter().zip(&dst).any(|(a, b)| a != b),
            "an identity permutation is not a null");
    for (to, from) in src.iter().zip(&dst) {
        out[*to] = spec[*from];
    }
    assert_eq!(src.len(), out.iter().filter(|r| r[0].is_finite()).count(),
               "a permutation must not change how many epochs carry a spectrum");
    out
}

/// Per-night robust scale of the three LOG POWERS: centre on the night's median, divide by its 5-95
/// spread. `lf_nu` is already a bounded ratio and is left alone. Absolute band power varies by an
/// order of magnitude between people, which a pooled standardiser cannot remove and this can.
fn normalise_spec(spec: &[[f64; N_SPEC]]) -> Vec<[f64; N_SPEC]> {
    let mut out = spec.to_vec();
    for c in 0..N_SPEC - 1 {
        let v: Vec<f64> = spec.iter().map(|r| r[c]).filter(|x| x.is_finite()).collect();
        if v.len() < 2 {
            continue;
        }
        let mid = median(&mut v.clone());
        let lo = pct(&mut v.clone(), 0.05);
        let hi = pct(&mut v.clone(), 0.95);
        let scale = if (hi - lo).abs() > 1e-12 { hi - lo } else { 1.0 };
        for r in out.iter_mut() {
            if r[c].is_finite() {
                r[c] = (r[c] - mid) / scale;
            }
        }
    }
    out
}

/// tanv1's emission per night, under one of the four spectral arms.
fn tanv1_emissions(train: &[&Night], report: &[Night], spec: Spec) -> Vec<Vec<[f64; CLASSES]>> {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for nt in train {
        for (e, t) in nt.truth.iter().enumerate() {
            if let Some(t) = t {
                x.push(row(nt, e, spec));
                y.push(*t);
            }
        }
    }
    let (m, sd) = standardise_cols(&x);
    let dx: Vec<Vec<f64>> = x.iter().map(|r| design_row(r, &m, &sd, &[])).collect();
    let w = fit_lr(&dx, &y, WEIGHT_POWER);
    report
        .iter()
        .map(|nt| {
            (0..nt.truth.len())
                .map(|e| {
                    let z = scores(&w, &design_row(&row(nt, e, spec), &m, &sd, &[]));
                    std::array::from_fn(|c| z[stage_idx(STAGE_ORDER[c])])
                })
                .collect()
        })
        .collect()
}

fn decode(em: &[[f64; CLASSES]]) -> Vec<usize> {
    decode_v2(em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect()
}

/// Which epochs an abstention rule keeps. Higher score = keep.
fn keep_mask(rank: &[f64], truth: &[Option<usize>], keep: f64) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..rank.len()).filter(|k| truth[*k].is_some()).collect();
    idx.sort_by(|a, b| rank[*b].total_cmp(&rank[*a]).then(a.cmp(b)));
    let take = ((idx.len() as f64 * keep).round() as usize).min(idx.len());
    let mut m = vec![false; rank.len()];
    for k in &idx[..take] {
        m[*k] = true;
    }
    m
}

/// Kappa over the kept, labelled epochs.
fn kappa_kept(path: &[usize], truth: &[Option<usize>], mask: &[bool]) -> Option<f64> {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate() {
        if let (Some(want), true) = (want, mask[k]) {
            p.push(path[k]);
            t.push(*want);
        }
    }
    (t.len() >= MIN_EPOCHS).then(|| kappa4(&confusion4(&p, &t)))
}

/// Distance to the nearest change in a decoded path: the cheapest abstention rule, and a null that
/// needs no emission at all.
fn to_edge(pred: &[usize]) -> Vec<f64> {
    let edges: Vec<usize> = (1..pred.len()).filter(|k| pred[*k] != pred[k - 1]).collect();
    (0..pred.len())
        .map(|k| edges.iter().map(|e| (*e as i64 - k as i64).abs() as f64).fold(f64::INFINITY, f64::min))
        .collect()
}

/// Top-two gap of an emission row: the confidence rule.
fn margin(em: &[[f64; CLASSES]]) -> Vec<f64> {
    em.iter()
        .map(|r| {
            let mut v = r.to_vec();
            v.sort_by(f64::total_cmp);
            v[CLASSES - 1] - v[CLASSES - 2]
        })
        .collect()
}

/// Deterministic pseudo-random rank, so RANDOM is reproducible across runs without a clock.
fn pseudo_random(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let mut h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64);
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            (h >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

fn main() {
    let loaded: Vec<(&str, Vec<Night>)> =
        COHORTS.iter().map(|c| (*c, load(c))).filter(|(_, n)| !n.is_empty()).collect();
    if loaded.len() < 3 {
        println!("need all three cohorts under the fixture root");
        return;
    }
    println!("Two levers against V2 REFIT, the honest opponent. Cohorts pass `require_psg`, every");
    println!("line passes the provenance guard, all selection is leave-one-cohort-out.\n");

    println!("=== spectral coverage: epochs whose window can carry a spectrum ===");
    for (name, nights) in &loaded {
        let (mut have, mut all) = (0usize, 0usize);
        for nt in nights {
            for s in &nt.spec {
                all += 1;
                have += usize::from(s[0].is_finite());
            }
        }
        println!("  {name:<14} {:>6.1}%  ({have} of {all} epochs)",
                 100.0 * have as f64 / all.max(1) as f64);
    }
    println!("  A cohort near 0% cannot be moved by this arm; its columns abstain to the train mean.\n");

    for (held, hn) in &loaded {
        let tr: Vec<&Night> =
            loaded.iter().filter(|(c, _)| c != held).flat_map(|(_, n)| n.iter()).collect();
        let wv = refit_v2(&tr);
        let v2_em: Vec<Vec<[f64; CLASSES]>> = hn
            .iter()
            .map(|nt| (0..nt.truth.len()).map(|e| nt.terms.emission(e, &wv)).collect())
            .collect();
        let plain = tanv1_emissions(&tr, hn, Spec::Off);
        let withspec = tanv1_emissions(&tr, hn, Spec::Real);
        let permuted = tanv1_emissions(&tr, hn, Spec::Permuted);
        let normalised = tanv1_emissions(&tr, hn, Spec::Normalised);

        println!("== {held} n={} ==", hn.len());
        println!("  {:<30} {:>7}   {:>8} {:>7}  verdict vs V2 REFIT", "arm", "kappa", "paired d",
                 "bar");

        // Full coverage first: does the spectral family move the emission at all?
        let mut base = Vec::new();
        // Captured by EXACT label. `contains("SPECTRAL")` also matches "SPECTRAL per-night", which
        // silently made two arms the same vector; `expect_once` refuses a second write.
        let (mut null_k, mut real_k, mut norm_k) = (None, None, None);
        let expect_once = |slot: &mut Option<Vec<f64>>, v: &[f64], what: &str| {
            assert!(slot.is_none(), "{what} captured twice");
            *slot = Some(v.to_vec());
        };
        for (label, ems) in [("V2 REFIT", &v2_em), ("tanv1", &plain), ("tanv1 + SPECTRAL", &withspec),
                             ("tanv1 + PERMUTED (null)", &permuted),
                             ("tanv1 + SPECTRAL per-night", &normalised)]
        {
            let k: Vec<f64> = hn
                .iter()
                .zip(ems)
                .filter_map(|(nt, em)| {
                    let mask = vec![true; nt.truth.len()];
                    kappa_kept(&decode(em), &nt.truth, &mask)
                })
                .collect();
            if label == "V2 REFIT" {
                base = k.clone();
                println!("  {:<30} {:>7.3}   the honest baseline", label, median(&mut k.clone()));
            } else {
                match label {
                    "tanv1 + SPECTRAL" => expect_once(&mut real_k, &k, label),
                    "tanv1 + PERMUTED (null)" => expect_once(&mut null_k, &k, label),
                    "tanv1 + SPECTRAL per-night" => expect_once(&mut norm_k, &k, label),
                    _ => {}
                }
                println!("  {:<30} {:>7.3}   {}", label, median(&mut k.clone()),
                         compare(&base, &k, Provenance::HeldOut).2);
            }
        }
        // The comparison the arm exists for: real columns against the same columns carrying nothing.
        if let (Some(null_k), Some(real_k), Some(norm_k)) = (&null_k, &real_k, &norm_k) {
            for (what, base, arm) in [
                ("  SPECTRAL vs PERMUTED", null_k, real_k),
                ("  per-night vs PERMUTED", null_k, norm_k),
                ("  per-night vs SPECTRAL", real_k, norm_k),
            ] {
                println!("  {what:<30} {:>7}   {}", "",
                         compare(base, arm, Provenance::HeldOut).2);
            }
        }

        // Abstention at matched coverage, both engines refusing by the SAME rule, plus two nulls.
        for keep in COVERAGE.iter().skip(1) {
            for (rule, use_margin, use_edge) in
                [("margin", true, false), ("nearest edge (null)", false, true),
                 ("random (null)", false, false)]
            {
                let (mut bk, mut ak) = (Vec::new(), Vec::new());
                for (i, (nt, em)) in hn.iter().zip(&plain).enumerate() {
                    let bp = decode(&v2_em[i]);
                    let ap = decode(em);
                    let (br, ar) = if use_margin {
                        (margin(&v2_em[i]), margin(em))
                    } else if use_edge {
                        (to_edge(&bp), to_edge(&ap))
                    } else {
                        let r = pseudo_random(nt.truth.len(), i);
                        (r.clone(), r)
                    };
                    let (bm, am) = (keep_mask(&br, &nt.truth, *keep),
                                    keep_mask(&ar, &nt.truth, *keep));
                    if let (Some(b), Some(a)) =
                        (kappa_kept(&bp, &nt.truth, &bm), kappa_kept(&ap, &nt.truth, &am))
                    {
                        bk.push(b);
                        ak.push(a);
                    }
                }
                println!("  {:<30} {:>7.3}   {}",
                         format!("  keep {:.0}%, {rule}", 100.0 * keep),
                         median(&mut ak.clone()), compare(&bk, &ak, Provenance::HeldOut).2);
            }
        }
        println!();
    }
    println!("The abstention rows compare tanv1 against V2 REFIT at MATCHED coverage, each engine");
    println!("refusing by its own version of the same rule, so the comparison stays paired. `random`");
    println!("is the bar a real rule has to clear; `nearest edge` needs no emission at all.");
    println!("Spectral columns are appended to tanv1 only, so their row is tanv1's gain from them.");
    println!("PERMUTED carries the same four columns shuffled within each night: read SPECTRAL");
    println!("against it, not against tanv1. If the two agree, the cost is column count and this");
    println!("harness cannot resolve a four-column addition of any content.");
}
