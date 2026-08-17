//! Wrist orientation, and the swing that separates lying still from being up.
//!
//! Every other motion feature in this crate reduces `(x, y, z)` to a scalar magnitude, so it knows how
//! much the wrist moved and nothing about how. Gravity's magnitude is ~1 g whatever a body does; its
//! DIRECTION is posture, and the way that direction behaves within an epoch separates a wrist resting
//! on a mattress from a wrist swinging beside a walking body.
//!
//! [`Posture::swing`] is the discriminator: a resting wrist holds one orientation, so its samples
//! concentrate; a moving one sweeps, so they cancel. [`turn`] is the complementary one, a rotation
//! between epochs with no swing inside either - rolling over.

use super::input::AccelSample;

/// One epoch of orientation. `dir` is the mean unit gravity vector; the two scalars are what a stager
/// reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Posture {
    /// Mean unit gravity, normalised. Which way is down, from the wrist's point of view.
    pub dir: [f64; 3],
    /// 0 = the wrist held one orientation for the whole epoch, 1 = it swept through many.
    /// One minus the resultant length of the unit vectors, so it needs no threshold to be meaningful.
    pub swing: f64,
    /// Samples the epoch was built from. A `swing` over one or two samples means nothing.
    pub n: usize,
}

/// Smallest sample count that can carry a direction at all.
pub const MIN_SAMPLES: usize = 3;

/// Mean orientation and swing over one epoch. `None` when the epoch is empty or every sample is
/// degenerate, which is a different fact from "the wrist was still".
pub fn posture_of(samples: &[AccelSample]) -> Option<Posture> {
    let (mut sx, mut sy, mut sz, mut n) = (0.0, 0.0, 0.0, 0usize);
    for s in samples {
        let m = (s.x * s.x + s.y * s.y + s.z * s.z).sqrt();
        if m < 1e-9 {
            continue;
        }
        sx += s.x / m;
        sy += s.y / m;
        sz += s.z / m;
        n += 1;
    }
    if n < MIN_SAMPLES {
        return None;
    }
    let nf = n as f64;
    let r = (sx * sx + sy * sy + sz * sz).sqrt() / nf;
    if r < 1e-9 {
        // Perfectly opposed samples: a real sweep, and there is no mean direction to report.
        return Some(Posture { dir: [0.0, 0.0, 0.0], swing: 1.0, n });
    }
    Some(Posture { dir: [sx / (r * nf), sy / (r * nf), sz / (r * nf)], swing: (1.0 - r).clamp(0.0, 1.0), n })
}

/// Angle in degrees between two epochs' orientations. `None` when either has no direction.
pub fn turn(a: &Posture, b: &Posture) -> Option<f64> {
    let dot: f64 = a.dir.iter().zip(&b.dir).map(|(p, q)| p * q).sum();
    if !dot.is_finite() || (a.swing >= 1.0 && a.n > 0 && a.dir == [0.0, 0.0, 0.0]) {
        return None;
    }
    if b.dir == [0.0, 0.0, 0.0] {
        return None;
    }
    Some(dot.clamp(-1.0, 1.0).acos().to_degrees())
}

/// Per-epoch postures over `[start, end)`, one entry per epoch, `None` where an epoch has too few
/// samples. Samples must be sorted by `ts`; out-of-order input lands in the wrong epoch.
pub fn posture_series(grav: &[AccelSample], start: i64, end: i64, epoch_s: i64) -> Vec<Option<Posture>> {
    if epoch_s <= 0 || end <= start {
        return Vec::new();
    }
    let n = ((end - start) / epoch_s) as usize;
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    for k in 0..n {
        let (a, b) = (start + k as i64 * epoch_s, start + (k as i64 + 1) * epoch_s);
        while i < grav.len() && grav[i].ts < a {
            i += 1;
        }
        let j = i + grav[i..].iter().take_while(|s| s.ts < b).count();
        out.push(posture_of(&grav[i..j]));
    }
    out
}

/// Turn angle per epoch against the previous one. `None` at the first epoch and wherever either
/// neighbour has no direction; a gap does NOT silently become "no rotation".
pub fn turn_series(p: &[Option<Posture>]) -> Vec<Option<f64>> {
    p.iter()
        .enumerate()
        .map(|(i, cur)| match (i.checked_sub(1).and_then(|j| p[j].as_ref()), cur.as_ref()) {
            (Some(prev), Some(c)) => turn(prev, c),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }

    #[test]
    fn a_wrist_held_in_one_orientation_has_no_swing() {
        let g: Vec<AccelSample> = (0..20).map(|i| s(i, 0.0, 0.0, 1.0)).collect();
        let p = posture_of(&g).unwrap();
        assert!(p.swing < 1e-9, "swing {}", p.swing);
        assert_eq!(p.dir, [0.0, 0.0, 1.0]);
    }

    /// The whole point of the module: a swept wrist and a still one have the SAME magnitude, and only
    /// swing tells them apart. Magnitude is 1 g in both, so any scalar feature sees no difference.
    #[test]
    fn a_swept_wrist_has_high_swing_at_identical_magnitude() {
        let still: Vec<AccelSample> = (0..24).map(|i| s(i, 0.0, 0.0, 1.0)).collect();
        let swept: Vec<AccelSample> = (0..24)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 24.0;
                s(i, a.cos(), a.sin(), 0.0)
            })
            .collect();
        for g in [&still, &swept] {
            for x in g.iter() {
                assert!(((x.x * x.x + x.y * x.y + x.z * x.z).sqrt() - 1.0).abs() < 1e-9);
            }
        }
        assert!(posture_of(&still).unwrap().swing < 1e-9);
        assert!(posture_of(&swept).unwrap().swing > 0.99, "a full sweep should cancel");
    }

    #[test]
    fn too_few_samples_report_nothing_rather_than_perfect_stillness() {
        assert_eq!(posture_of(&[]), None);
        assert_eq!(posture_of(&[s(0, 0.0, 0.0, 1.0), s(1, 0.0, 0.0, 1.0)]), None, "2 < MIN_SAMPLES");
        assert!(posture_of(&(0..3).map(|i| s(i, 0.0, 0.0, 1.0)).collect::<Vec<_>>()).is_some());
    }

    #[test]
    fn zero_vectors_are_skipped_not_counted_as_a_direction() {
        let g = vec![s(0, 0.0, 0.0, 0.0), s(1, 0.0, 0.0, 0.0), s(2, 0.0, 0.0, 1.0)];
        assert_eq!(posture_of(&g), None, "one usable sample of three is under the floor");
    }

    #[test]
    fn turn_measures_the_rotation_between_two_still_epochs() {
        let flat = posture_of(&(0..8).map(|i| s(i, 0.0, 0.0, 1.0)).collect::<Vec<_>>()).unwrap();
        let side = posture_of(&(0..8).map(|i| s(i, 1.0, 0.0, 0.0)).collect::<Vec<_>>()).unwrap();
        let t = turn(&flat, &side).unwrap();
        assert!((t - 90.0).abs() < 1e-6, "got {t}");
        assert!(turn(&flat, &flat).unwrap() < 1e-6);
    }

    #[test]
    fn a_series_reports_none_for_the_epochs_it_cannot_fill() {
        // Epoch 0 dense, epoch 1 empty, epoch 2 dense but rotated 90 degrees.
        let mut g: Vec<AccelSample> = (0..8).map(|i| s(i, 0.0, 0.0, 1.0)).collect();
        g.extend((60..68).map(|i| s(i, 1.0, 0.0, 0.0)));
        let p = posture_series(&g, 0, 90, 30);
        assert_eq!(p.len(), 3);
        assert!(p[0].is_some() && p[1].is_none() && p[2].is_some());

        let t = turn_series(&p);
        assert_eq!(t[0], None, "nothing precedes the first epoch");
        assert_eq!(t[1], None, "an empty epoch is not a rotation of zero");
        assert_eq!(t[2], None, "and neither is the epoch after it");
    }

    #[test]
    fn a_rotation_across_adjacent_epochs_is_reported() {
        let mut g: Vec<AccelSample> = (0..30).map(|i| s(i, 0.0, 0.0, 1.0)).collect();
        g.extend((30..60).map(|i| s(i, 1.0, 0.0, 0.0)));
        let t = turn_series(&posture_series(&g, 0, 60, 30));
        assert!((t[1].unwrap() - 90.0).abs() < 1e-6, "got {:?}", t[1]);
    }
}
