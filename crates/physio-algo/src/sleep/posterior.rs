//! Forward-backward posterior marginals over the four-class HMM, as a shared primitive.
//!
//! Viterbi returns one best path and no per-epoch probability. The posterior-marginal rule and the
//! loss-matched rule both price each epoch on its own, so both need `P(stage_t | whole night)` and
//! neither can be built on the decoders the tree has. This is that quantity, computed in the LOG
//! domain with log-sum-exp: a night is ~1000 epochs and a linear-domain product underflows to zero
//! long before the end of one.
//!
//! The transition arrives the way `conditioned::viterbi_with` takes it — `trans(t)` is the matrix
//! for the step INTO epoch `t` — so one implementation serves the fixed shipped matrix and the
//! per-epoch conditioned one alike, and the closure is called exactly once per step as the decoder
//! calls it. The same zero floor and a uniform start are applied, so these marginals assume exactly
//! what `v2::viterbi` assumes; the start is normalised where the decoder's is not, which is a
//! constant offset on every path and cannot move a marginal or an argmax.
//!
//! Consumed by the decode rules, not by a pipeline step of its own; nothing shipped decodes with it.
//! `markov_loss`'s `ft` and `fh` price where a boundary SITS, which is a property of a PAIR of
//! epochs, so no per-epoch rule below can charge them and a rule that prices them must decode over
//! adjacent pairs.

use super::markov_loss::Costs;
use super::{SleepStage, STAGE_ORDER};

/// The floor a zero transition is clamped to before logging. Mirrors the decoders' own, so a
/// structurally-forbidden step is improbable here in exactly the degree it is improbable there.
const TRANS_FLOOR: f64 = 1e-9;

/// Uniform start over the four stages. A constant offset on every path, so it moves no marginal and
/// no argmax; it is what makes [`log_likelihood`] a probability rather than an evidence weight.
const UNIFORM_START: f64 = 0.25;

fn log_sum_exp4(v: [f64; 4]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// Index of the largest value in `v`, ties to the earlier index — the rule both decoders here use.
fn argmax4(v: &[f64; 4]) -> usize {
    (1..4).fold(0usize, |b, s| if v[s] > v[b] { s } else { b })
}

/// `trans(t)` logged and floored, entry `t - 1` holding the step into epoch `t`. One call per step.
fn log_transitions(len: usize, trans: impl Fn(usize) -> [[f64; 4]; 4]) -> Vec<[[f64; 4]; 4]> {
    (1..len)
        .map(|t| {
            let a = trans(t);
            let mut l = [[0.0f64; 4]; 4];
            for (i, row) in a.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    l[i][j] = v.max(TRANS_FLOOR).ln();
                }
            }
            l
        })
        .collect()
}

/// Log forward messages `ln P(obs_0..t, stage_t)`, one row per epoch.
fn forward(em: &[[f64; 4]], log_t: &[[[f64; 4]; 4]]) -> Vec<[f64; 4]> {
    let s0 = UNIFORM_START.ln();
    let mut alpha: Vec<[f64; 4]> = Vec::with_capacity(em.len());
    alpha.push(std::array::from_fn(|s| em[0][s] + s0));
    for (t, e) in em.iter().enumerate().skip(1) {
        let a = &log_t[t - 1];
        let prev = alpha[t - 1];
        alpha.push(std::array::from_fn(|s| {
            log_sum_exp4([prev[0] + a[0][s], prev[1] + a[1][s], prev[2] + a[2][s], prev[3] + a[3][s]])
                + e[s]
        }));
    }
    alpha
}

/// Log backward messages `ln P(obs_t+1..T | stage_t)`, one row per epoch, the last row zero.
fn backward(em: &[[f64; 4]], log_t: &[[[f64; 4]; 4]]) -> Vec<[f64; 4]> {
    let mut beta = vec![[0.0f64; 4]; em.len()];
    for t in (0..em.len().saturating_sub(1)).rev() {
        let a = &log_t[t];
        let (e, nxt) = (em[t + 1], beta[t + 1]);
        beta[t] = std::array::from_fn(|s| {
            log_sum_exp4([
                a[s][0] + e[0] + nxt[0],
                a[s][1] + e[1] + nxt[1],
                a[s][2] + e[2] + nxt[2],
                a[s][3] + e[3] + nxt[3],
            ])
        });
    }
    beta
}

/// Per-epoch `P(stage | whole night)` in probability space, rows summing to 1; empty in, empty out.
/// `trans(t)` is the matrix for the step INTO epoch `t`; zeros are floored and the start uniform, as
/// in `v2::viterbi`. A row whose evidence has collapsed past `f64` falls back to uniform.
pub fn forward_backward(em: &[[f64; 4]], trans: impl Fn(usize) -> [[f64; 4]; 4]) -> Vec<[f64; 4]> {
    if em.is_empty() {
        return Vec::new();
    }
    let log_t = log_transitions(em.len(), trans);
    let alpha = forward(em, &log_t);
    let beta = backward(em, &log_t);
    alpha
        .iter()
        .zip(&beta)
        .map(|(a, b)| {
            let x: [f64; 4] = std::array::from_fn(|s| a[s] + b[s]);
            let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !m.is_finite() {
                return [0.25f64; 4];
            }
            let w: [f64; 4] = std::array::from_fn(|s| (x[s] - m).exp());
            let sum: f64 = w.iter().sum();
            if sum <= 0.0 || !sum.is_finite() {
                return [0.25f64; 4];
            }
            std::array::from_fn(|s| w[s] / sum)
        })
        .collect()
}

/// Log-likelihood of the whole emission sequence — `ln` of the summed probability of all `4^T`
/// paths, read off the forward pass [`forward_backward`] runs anyway. Empty input is 0.0.
pub fn log_likelihood(em: &[[f64; 4]], trans: impl Fn(usize) -> [[f64; 4]; 4]) -> f64 {
    if em.is_empty() {
        return 0.0;
    }
    let log_t = log_transitions(em.len(), trans);
    let alpha = forward(em, &log_t);
    log_sum_exp4(alpha[em.len() - 1])
}

/// Per-epoch argmax of the marginals through [`STAGE_ORDER`], ties to the earlier index. It
/// maximises expected correct epochs rather than path probability, so what it returns need not be a
/// legal path under the transition — that is the rule, not a defect.
pub fn posterior_marginal_decode(post: &[[f64; 4]]) -> Vec<SleepStage> {
    post.iter().map(|p| STAGE_ORDER[argmax4(p)]).collect()
}

/// Bayes-risk-minimising label per epoch under the per-class epoch costs `fc` ONLY, indexed in
/// [`STAGE_ORDER`] columns like `post`. `fc[c]` is charged on the TRUE class, so the rule is
/// `argmax p[a]*fc[a]`: `Costs::UNIT` is the plain argmax, and a DEARER class is called MORE.
pub fn decode_with_costs(post: &[[f64; 4]], costs: &Costs) -> Vec<SleepStage> {
    post.iter()
        .map(|p| {
            let w: [f64; 4] = std::array::from_fn(|s| p[s] * costs.fc[s]);
            STAGE_ORDER[argmax4(&w)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::conditioned::transition_at;
    use crate::sleep::{decode_v2, params::Params};
    use std::cell::Cell;

    fn col(s: SleepStage) -> usize {
        STAGE_ORDER.iter().position(|x| *x == s).expect("a stage is in STAGE_ORDER")
    }

    fn wiggly(n: usize) -> Vec<[f64; 4]> {
        (0..n)
            .map(|t| {
                let x = t as f64;
                [(x * 0.37).sin() * 2.0, (x * 0.11).cos() * 1.5, (x * 0.05).sin() - 0.5, (x * 0.23).cos()]
            })
            .collect()
    }

    fn softmax(v: &[f64; 4]) -> [f64; 4] {
        let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: [f64; 4] = std::array::from_fn(|s| (v[s] - m).exp());
        let sum: f64 = w.iter().sum();
        std::array::from_fn(|s| w[s] / sum)
    }

    /// Exact marginals and exact log-likelihood by enumerating all `4^T` paths. Shares no code with
    /// what it checks: it multiplies path weights in the linear domain, which is why `T <= 6`.
    fn brute_force(em: &[[f64; 4]], trans: &dyn Fn(usize) -> [[f64; 4]; 4]) -> (Vec<[f64; 4]>, f64) {
        let n = em.len();
        let mut acc = vec![[0.0f64; 4]; n];
        let mut total = 0.0f64;
        for code in 0..4usize.pow(n as u32) {
            let mut path = vec![0usize; n];
            let mut c = code;
            for p in path.iter_mut() {
                *p = c % 4;
                c /= 4;
            }
            let mut w = UNIFORM_START * em[0][path[0]].exp();
            for i in 1..n {
                w *= trans(i)[path[i - 1]][path[i]].max(TRANS_FLOOR) * em[i][path[i]].exp();
            }
            total += w;
            for (i, p) in path.iter().enumerate() {
                acc[i][*p] += w;
            }
        }
        (acc.iter().map(|r| std::array::from_fn(|s| r[s] / total)).collect(), total.ln())
    }

    /// The reference: exhaustive enumeration, on a transition that actually varies with `t`.
    #[test]
    fn marginals_match_exhaustive_enumeration_under_a_per_epoch_transition() {
        let base = Params::SHIPPED.transition;
        let vary = |t: usize| transition_at(&base, (t as f64) * 0.7, 1.3);
        let em: Vec<[f64; 4]> = (0..6)
            .map(|t| {
                let x = t as f64;
                [(x * 0.9).sin() * 1.7, (x * 1.3).cos() - 0.4, 0.6 - x * 0.21, (x * 0.5).sin() * 2.2]
            })
            .collect();
        assert_ne!(vary(1), vary(5), "the closure must actually vary with t");

        let (want, want_ll) = brute_force(&em, &vary);
        let got = forward_backward(&em, vary);
        assert_eq!(got.len(), want.len());
        for (t, (g, w)) in got.iter().zip(&want).enumerate() {
            for s in 0..4 {
                assert!((g[s] - w[s]).abs() < 1e-9, "epoch {t} stage {s}: {} vs {}", g[s], w[s]);
            }
        }
        assert!((log_likelihood(&em, vary) - want_ll).abs() < 1e-9, "log-likelihood must match too");
    }

    /// The underflow guard. Fifty-nat emissions over 1200 epochs are past `f64`'s linear range by
    /// hundreds of orders of magnitude. Rows summing to 1 is half the claim: a collapsed pass falls
    /// back to uniform, which also sums to 1, so the rows must also still be INFORMATIVE.
    #[test]
    fn rows_are_distributions_on_a_full_night_of_wild_emission_scales() {
        let base = Params::SHIPPED.transition;
        let em: Vec<[f64; 4]> = (0..1200)
            .map(|t| {
                let x = t as f64;
                std::array::from_fn(|s| (x * 0.017 + s as f64 * 1.9).sin() * 50.0)
            })
            .collect();
        let post = forward_backward(&em, |_| base);
        assert_eq!(post.len(), 1200);
        for (t, p) in post.iter().enumerate() {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "epoch {t} sums to {sum}");
            assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0), "epoch {t}: {p:?}");
            assert!(p.iter().any(|x| (x - 0.25).abs() > 1e-9), "epoch {t} collapsed to uniform");
        }
        let peaked = post.iter().filter(|p| p.iter().copied().fold(0.0, f64::max) > 0.99).count();
        assert!(peaked > 1000, "fifty-nat evidence must decide almost every epoch, not {peaked}");
    }

    /// With a uniform transition the epochs are independent, so each marginal is that epoch's own
    /// emission softmaxed and nothing else.
    #[test]
    fn uniform_transitions_reduce_the_marginals_to_the_emission_softmax() {
        let em = wiggly(50);
        let post = forward_backward(&em, |_| [[0.25f64; 4]; 4]);
        for (t, (p, e)) in post.iter().zip(&em).enumerate() {
            let want = softmax(e);
            for s in 0..4 {
                assert!((p[s] - want[s]).abs() < 1e-12, "epoch {t} stage {s}: {} vs {}", p[s], want[s]);
            }
        }
    }

    /// Coherence with the path search: every epoch of the Viterbi path carries posterior mass, and
    /// where the emissions leave no doubt the two rules return the same hypnogram.
    #[test]
    fn the_viterbi_path_is_supported_and_decisive_emissions_make_the_rules_agree() {
        let base = Params::SHIPPED.transition;
        let em = wiggly(400);
        let post = forward_backward(&em, |_| base);
        for (t, s) in decode_v2(&em, &base).iter().enumerate() {
            assert!(post[t][col(*s)] > 0.0, "epoch {t} of the viterbi path has zero posterior");
        }

        // Deep -> light -> REM -> light -> wake, so no leg needs a structurally-zero step.
        let plan =
            [SleepStage::Deep, SleepStage::Light, SleepStage::Rem, SleepStage::Light, SleepStage::Wake];
        let mut decisive = Vec::new();
        for stage in plan {
            for _ in 0..40 {
                let mut e = [-20.0f64; 4];
                e[col(stage)] = 0.0;
                decisive.push(e);
            }
        }
        let want = decode_v2(&decisive, &base);
        let got = posterior_marginal_decode(&forward_backward(&decisive, |_| base));
        assert_eq!(got, want, "decisive emissions must leave the two rules nothing to disagree on");
        assert!(want.iter().any(|s| *s != want[0]), "the sequence must actually change stage");
    }

    /// And where they legitimately differ. Column 0 has one certain successor with a good emission;
    /// column 1 spreads over four slightly better ones, so it wins the SUM while losing the MAX.
    #[test]
    fn the_two_rules_differ_where_mass_beats_the_best_single_path() {
        let a = [[1.0, 0.0, 0.0, 0.0], [0.25; 4], [0.25; 4], [0.25; 4]];
        let em = [[0.0, 0.0, -10.0, -10.0], [1.4, 1.5, 1.5, 1.5]];
        let path = decode_v2(&em, &a);
        let marg = posterior_marginal_decode(&forward_backward(&em, |_| a));
        assert_eq!(path[0], STAGE_ORDER[0], "the best path takes the certain successor");
        assert_eq!(marg[0], STAGE_ORDER[1], "the marginal takes the spread one");
        assert_ne!(marg, path, "the two rules are not the same rule");
        assert_eq!(marg[1], path[1], "and they differ at the first epoch only");
    }

    /// `Costs::UNIT` is the null: it must reproduce the plain argmax exactly. Then the direction of
    /// `fc` — it is charged on the TRUE class, so raising it buys that class more calls, not fewer.
    #[test]
    fn unit_costs_are_the_plain_argmax_and_fc_moves_a_class_in_the_charged_direction() {
        let base = Params::SHIPPED.transition;
        let post = forward_backward(&wiggly(400), |_| base);
        let plain = posterior_marginal_decode(&post);
        assert_eq!(decode_with_costs(&post, &Costs::UNIT), plain, "UNIT must be the argmax");

        let deep = col(SleepStage::Deep);
        let calls = |v: &[SleepStage]| v.iter().filter(|s| **s == SleepStage::Deep).count();
        let mut dear = Costs::UNIT;
        dear.fc[deep] = 10.0;
        let mut cheap = Costs::UNIT;
        cheap.fc[deep] = 0.1;
        assert!(
            calls(&decode_with_costs(&post, &dear)) > calls(&plain),
            "a dearer miss must be avoided by calling deep more"
        );
        assert!(
            calls(&decode_with_costs(&post, &cheap)) < calls(&plain),
            "a cheaper miss must be accepted by calling deep less"
        );
    }

    /// The closure is called once per STEP and never for epoch 0 — the shape `conditioned` uses. A
    /// second call per step would silently double the cost of a fitted transition.
    #[test]
    fn the_transition_closure_is_called_once_per_step() {
        let base = Params::SHIPPED.transition;
        let seen: Cell<usize> = Cell::new(0);
        let first: Cell<usize> = Cell::new(usize::MAX);
        let em = wiggly(20);
        let _ = forward_backward(&em, |t| {
            seen.set(seen.get() + 1);
            first.set(first.get().min(t));
            base
        });
        assert_eq!(seen.get(), 19, "one call per step into an epoch");
        assert_eq!(first.get(), 1, "epoch 0 has no step into it");
    }

    #[test]
    fn empty_and_length_one_inputs_do_not_panic() {
        let base = Params::SHIPPED.transition;
        assert!(forward_backward(&[], |_| base).is_empty());
        assert_eq!(log_likelihood(&[], |_| base), 0.0, "the empty product");
        assert!(posterior_marginal_decode(&[]).is_empty());
        assert!(decode_with_costs(&[], &Costs::UNIT).is_empty());

        let one = [[0.4f64, -1.0, 2.0, 0.1]];
        let post = forward_backward(&one, |_| base);
        assert_eq!(post.len(), 1);
        let want = softmax(&one[0]);
        for s in 0..4 {
            assert!((post[0][s] - want[s]).abs() < 1e-12, "one epoch is its own softmax");
        }
        let ll = log_likelihood(&one, |_| base);
        assert!((ll - (UNIFORM_START.ln() + log_sum_exp4(one[0]))).abs() < 1e-12, "ll {ll}");
        assert_eq!(posterior_marginal_decode(&post), vec![STAGE_ORDER[2]]);
    }
}
