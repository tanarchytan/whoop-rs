//! Observation-conditioned transitions: the one structural change priced above zero.
//!
//! An HMM's transition matrix cannot depend on the signal. Re-pricing it as a FIXED matrix was
//! measured dead in both directions; what survives is a transition that reads the epoch it enters.
//! This is the smallest form of that: the diagonal is scaled by the epoch's own motion, so a still
//! epoch keeps the shipped stickiness and a moving one both makes leaving cheaper AND makes staying
//! dearer, since the mass is conserved. Freed mass is returned in the shipped proportions, so a
//! structural zero stays zero.
//!
//! At `beta = 0` every matrix is the shipped one and the decode is `v2::viterbi` label for label:
//! the built-in null every arm is read against. The decoder is a replica of the frozen
//! `v2::viterbi` with a per-epoch matrix in place of one; a test pins the two together.

use super::{SleepStage, Terms, STAGE_ORDER, WEIGHT_NAMES};

/// How hard motion loosens the diagonal, in thousandths so the config stays `Eq + Hash`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConditionedCfg {
    pub beta_milli: u32,
}

impl ConditionedCfg {
    pub fn beta(self) -> f64 {
        self.beta_milli as f64 / 1000.0
    }
}

/// The floor `v2::viterbi` applies to a zero transition, so no path is unreachable.
const ZERO_FLOOR: f64 = 1e-9;

/// The motion quantity the emission reads, one value per epoch: the awake-row motion slot of the
/// design matrix. Missing is `NaN`, which [`transition_at`] treats as "no motion seen".
pub fn motion_of(terms: &Terms) -> Vec<f64> {
    let row = STAGE_ORDER.iter().position(|s| *s == SleepStage::Wake).expect("wake is a stage");
    let col = WEIGHT_NAMES.iter().position(|n| *n == "awake_motion").expect("named weight");
    terms.design.iter().map(|d| d[row][col]).collect()
}

/// The shipped matrix with every diagonal scaled by `exp(-beta * max(motion, 0))` and the freed
/// mass returned to that row's off-diagonals in their existing proportions. One-sided on purpose:
/// stillness never makes staying cheaper than shipped, only motion makes leaving cheaper.
pub fn transition_at(base: &[[f64; 4]; 4], motion: f64, beta: f64) -> [[f64; 4]; 4] {
    let m = if motion.is_finite() { motion.max(0.0) } else { 0.0 };
    let keep = (-beta * m).exp();
    let mut out = *base;
    for i in 0..4 {
        let stay = base[i][i];
        let off: f64 = (0..4).filter(|j| *j != i).map(|j| base[i][j]).sum();
        if off <= 0.0 {
            continue;
        }
        let freed = stay * (1.0 - keep);
        out[i][i] = stay - freed;
        for j in (0..4).filter(|j| *j != i) {
            out[i][j] = base[i][j] + freed * base[i][j] / off;
        }
    }
    out
}

/// Most-likely path under a matrix that may differ at every epoch. `trans(t)` is the matrix for the
/// step INTO epoch `t`; uniform start; ties resolve to the earlier stage index.
fn viterbi_with(em: &[[f64; 4]], trans: impl Fn(usize) -> [[f64; 4]; 4]) -> Vec<SleepStage> {
    if em.is_empty() {
        return Vec::new();
    }
    let mut v = em[0];
    let mut back: Vec<[usize; 4]> = Vec::with_capacity(em.len());
    for (t, e) in em.iter().enumerate().skip(1) {
        let a = trans(t);
        let mut new_v = [0.0f64; 4];
        let mut bp = [0usize; 4];
        for s in 0..4 {
            let (mut best_prev, mut best_val) = (0usize, v[0] + a[0][s].max(ZERO_FLOOR).ln());
            for p in 1..4 {
                let value = v[p] + a[p][s].max(ZERO_FLOOR).ln();
                if value > best_val {
                    best_val = value;
                    best_prev = p;
                }
            }
            new_v[s] = best_val + e[s];
            bp[s] = best_prev;
        }
        v = new_v;
        back.push(bp);
    }
    let mut last = (0..4).fold(0usize, |b, s| if v[s] > v[b] { s } else { b });
    let mut path = vec![last];
    for bp in back.iter().rev() {
        last = bp[last];
        path.push(last);
    }
    path.reverse();
    path.into_iter().map(|i| STAGE_ORDER[i]).collect()
}

/// Decode with the diagonal loosened by each epoch's own motion. `motion` is index-for-index with
/// `em`; a shorter slice reads as no motion past its end.
pub fn decode(em: &[[f64; 4]], base: &[[f64; 4]; 4], motion: &[f64], beta: f64) -> Vec<SleepStage> {
    viterbi_with(em, |t| transition_at(base, motion.get(t).copied().unwrap_or(0.0), beta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sleep::{decode_v2, params::Params};

    fn rows_sum_to_one(a: &[[f64; 4]; 4]) -> bool {
        a.iter().all(|r| (r.iter().sum::<f64>() - 1.0).abs() < 1e-12)
    }

    /// The built-in null: at beta 0 nothing moves, whatever the motion.
    #[test]
    fn beta_zero_returns_the_base_matrix_exactly() {
        let base = Params::SHIPPED.transition;
        assert_eq!(transition_at(&base, 5.0, 0.0), base);
        assert_eq!(transition_at(&base, 0.0, 2.0), base, "no motion, no change");
        assert_eq!(transition_at(&base, -3.0, 2.0), base, "one-sided: stillness does nothing");
        assert_eq!(transition_at(&base, f64::NAN, 2.0), base, "missing motion is no motion");
    }

    /// Motion lowers every diagonal, rows stay normalised, and a structural zero stays zero.
    #[test]
    fn motion_loosens_the_diagonal_and_keeps_zeros() {
        let base = Params::SHIPPED.transition;
        let a = transition_at(&base, 1.0, 1.0);
        assert!(rows_sum_to_one(&a), "{a:?}");
        for i in 0..4 {
            assert!(a[i][i] < base[i][i], "row {i} must loosen");
            for j in 0..4 {
                if base[i][j] == 0.0 {
                    assert_eq!(a[i][j], 0.0, "a structural zero must survive: [{i}][{j}]");
                }
            }
        }
        let harder = transition_at(&base, 3.0, 1.0);
        assert!(harder[2][2] < a[2][2], "more motion, looser still");
    }

    /// The decoder is a replica of the frozen one. On a constant matrix they must agree label for
    /// label, or every arm here is measured against a different null than it claims.
    #[test]
    fn a_constant_matrix_reproduces_v2_viterbi() {
        let base = Params::SHIPPED.transition;
        let em: Vec<[f64; 4]> = (0..400)
            .map(|t| {
                let x = t as f64;
                [(x * 0.37).sin() * 2.0, (x * 0.11).cos() * 1.5, (x * 0.05).sin() - 0.5, (x * 0.23).cos()]
            })
            .collect();
        let ours = decode(&em, &base, &[], 0.0);
        let theirs = decode_v2(&em, &base);
        assert_eq!(ours, theirs, "beta 0 must be v2::viterbi exactly");
        assert!(ours.iter().any(|s| *s != ours[0]), "the test sequence must actually change stage");
    }

    /// The mechanism exists. Viterbi is global, so a SUSTAINED preference beats any one-off cost;
    /// the case that separates the decoders is a short burst too weak to pay the fixed round trip
    /// (2.30 + 2.30 nats) but strong enough once motion at its edges loosens both. And the
    /// negative control: motion where there is nothing worth switching to must not invent a move.
    #[test]
    fn motion_at_a_burst_lets_a_weak_preference_switch_there() {
        let base = Params::SHIPPED.transition;
        let light = STAGE_ORDER.iter().position(|s| *s == SleepStage::Light).unwrap();
        let wake = STAGE_ORDER.iter().position(|s| *s == SleepStage::Wake).unwrap();
        let mut em = vec![[-8.0f64; 4]; 80];
        for (t, e) in em.iter_mut().enumerate() {
            e[light] = 0.0;
            if (40..43).contains(&t) {
                e[wake] = 1.0;
            }
        }
        let fixed = decode(&em, &base, &[], 0.0);
        assert!(fixed.iter().all(|s| *s == SleepStage::Light), "fixed must never switch: {fixed:?}");

        // Motion where the burst enters AND where it leaves, so both edges loosen.
        let mut motion = vec![0.0; 80];
        motion[40] = 4.0;
        motion[43] = 4.0;
        let cond = decode(&em, &base, &motion, 2.0);
        assert_eq!(&cond[38..40], &[SleepStage::Light; 2]);
        assert_eq!(&cond[40..43], &[SleepStage::Wake; 3], "the burst must be taken: {:?}", &cond[38..45]);
        assert_eq!(&cond[43..45], &[SleepStage::Light; 2], "and left again");

        // Motion with nothing worth switching to: staying costs ~8 nats at the moving epoch, but every
        // alternative costs its transition plus an 8-nat emission plus the return. It must stay put.
        let mut idle = vec![0.0; 80];
        idle[20] = 4.0;
        let quiet = decode(&em, &base, &idle, 2.0);
        assert!(quiet[..40].iter().all(|s| *s == SleepStage::Light), "motion alone must not invent a switch: {:?}", &quiet[18..23]);
    }

    #[test]
    fn empty_and_short_inputs_do_not_panic() {
        let base = Params::SHIPPED.transition;
        assert!(decode(&[], &base, &[], 1.0).is_empty());
        assert_eq!(decode(&[[0.0; 4]], &base, &[], 1.0).len(), 1);
        assert_eq!(decode(&[[0.0; 4]; 5], &base, &[1.0], 1.0).len(), 5, "short motion reads as none");
    }
}
