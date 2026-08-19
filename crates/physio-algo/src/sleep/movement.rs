//! Movement that keeps all three axes, including the ones `turn` and `swing` still throw away.
//!
//! [`super::posture`] recovered direction from magnitude, but its two scalars each collapse a
//! 3-vector again: `swing` to a spread and `turn` to one angle. The angle says HOW FAR the wrist
//! rotated and nothing about ABOUT WHAT, and the within-epoch deltas say how much moved and nothing
//! about in which plane.
//!
//! Two things survive that collapse and are recovered here. [`Movement::axis`] is the unit axis of
//! the inter-epoch rotation - rolling over turns about the body's long axis, reaching does not.
//! [`Movement::anisotropy`] is how concentrated the within-epoch motion is on one direction - a
//! swinging arm is close to planar, a restless one is not.
//!
//! Every measure is reference-free and device-frame-relative, so none of it needs a mounting
//! calibration and none of it changes when the strap is re-donned rotated.

use super::input::AccelSample;
use super::posture::Posture;

/// Fewest per-second samples in an epoch before its plane can be estimated.
pub const MIN_DELTAS: usize = 4;
/// Below this rotation the axis is numerically meaningless: two nearly parallel unit vectors have a
/// cross product dominated by their own noise.
pub const MIN_AXIS_DEG: f64 = 1.0;

/// One epoch's axis-preserving movement description.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Movement {
    /// Unit axis of the rotation from the previous epoch, right-handed. `None` when either epoch has
    /// no direction or the rotation is under [`MIN_AXIS_DEG`].
    pub axis: Option<[f64; 3]>,
    /// How concentrated the within-epoch motion is on a single direction. 1/3 = isotropic, 1 = all of
    /// it on one axis. `None` when the epoch carries too few deltas to have a plane.
    pub anisotropy: Option<f64>,
    /// Index of the device axis carrying the most within-epoch motion.
    pub dominant: Option<usize>,
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Unit rotation axis taking `a` to `b`. `None` when either is degenerate or they are too close to
/// parallel for the cross product to mean anything.
pub fn rotation_axis(a: &Posture, b: &Posture) -> Option<[f64; 3]> {
    let (p, q) = (a.dir, b.dir);
    if norm(p) < 1e-9 || norm(q) < 1e-9 {
        return None;
    }
    let dot: f64 = p.iter().zip(&q).map(|(x, y)| x * y).sum();
    if dot.clamp(-1.0, 1.0).acos().to_degrees() < MIN_AXIS_DEG {
        return None;
    }
    let c = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
    let n = norm(c);
    (n > 1e-12).then(|| [c[0] / n, c[1] / n, c[2] / n])
}

/// Angle in degrees between two rotation axes. Says whether successive rotations turn about the SAME
/// axis - a repetitive arm swing does, a restless sleeper does not.
pub fn axis_agreement(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Per-axis concentration of one epoch's motion, from the per-second gravity deltas.
///
/// Returns `(anisotropy, dominant axis)`. Anisotropy is the largest axis' share of total squared
/// delta, so it is bounded 1/3..1 and needs no scale: it is unchanged if the whole epoch is louder.
pub fn anisotropy_of(samples: &[AccelSample]) -> Option<(f64, usize)> {
    let mut sums = [0.0f64; 3];
    let mut n = 0usize;
    for (a, b) in samples.iter().zip(samples.iter().skip(1)) {
        let d = [b.x - a.x, b.y - a.y, b.z - a.z];
        for i in 0..3 {
            sums[i] += d[i] * d[i];
        }
        n += 1;
    }
    if n < MIN_DELTAS {
        return None;
    }
    let total: f64 = sums.iter().sum();
    if total < 1e-18 {
        // Perfectly still: there is no plane of motion, which is not the same as an isotropic one.
        return None;
    }
    let mut best = 0usize;
    for i in 1..3 {
        if sums[i] > sums[best] {
            best = i;
        }
    }
    Some((sums[best] / total, best))
}

/// Per-epoch [`Movement`] over `[start, end)`, index-for-index with [`super::posture::posture_series`].
/// `postures` must be that series for the same grid; a mismatched length is an empty result rather
/// than a silently misaligned one.
pub fn movement_series(
    grav: &[AccelSample],
    postures: &[Option<Posture>],
    start: i64,
    end: i64,
    epoch_s: i64,
) -> Vec<Movement> {
    if epoch_s <= 0 || end <= start {
        return Vec::new();
    }
    let n = ((end - start) / epoch_s) as usize;
    if postures.len() != n {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    for k in 0..n {
        let (a, b) = (start + k as i64 * epoch_s, start + (k as i64 + 1) * epoch_s);
        while i < grav.len() && grav[i].ts < a {
            i += 1;
        }
        let j = i + grav[i..].iter().take_while(|s| s.ts < b).count();
        let aniso = anisotropy_of(&grav[i..j]);
        let axis = match (k.checked_sub(1).and_then(|p| postures[p].as_ref()), postures[k].as_ref()) {
            (Some(prev), Some(cur)) => rotation_axis(prev, cur),
            _ => None,
        };
        out.push(Movement {
            axis,
            anisotropy: aniso.map(|x| x.0),
            dominant: aniso.map(|x| x.1),
        });
    }
    out
}

/// Angle between each epoch's rotation axis and the previous one. `None` wherever either is absent,
/// so a gap never reads as "the wrist kept turning the same way".
pub fn axis_agreement_series(m: &[Movement]) -> Vec<Option<f64>> {
    m.iter()
        .enumerate()
        .map(|(i, cur)| match (i.checked_sub(1).and_then(|p| m[p].axis.as_ref()), cur.axis.as_ref()) {
            (Some(a), Some(b)) => Some(axis_agreement(a, b)),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::posture::posture_of;

    fn s(ts: i64, x: f64, y: f64, z: f64) -> AccelSample {
        AccelSample { ts, x, y, z }
    }

    fn still(n: i64, v: (f64, f64, f64)) -> Vec<AccelSample> {
        (0..n).map(|i| s(i, v.0, v.1, v.2)).collect()
    }

    /// The whole point: two rotations of the SAME angle about DIFFERENT axes. `turn` cannot tell
    /// them apart and the axis can.
    #[test]
    fn equal_rotations_about_different_axes_are_distinguished_only_by_the_axis() {
        let flat = posture_of(&still(8, (0.0, 0.0, 1.0))).unwrap();
        let to_x = posture_of(&still(8, (1.0, 0.0, 0.0))).unwrap();
        let to_y = posture_of(&still(8, (0.0, 1.0, 0.0))).unwrap();
        // Both are exactly 90 degrees from flat, so the angle is identical.
        let t1 = super::super::posture::turn(&flat, &to_x).unwrap();
        let t2 = super::super::posture::turn(&flat, &to_y).unwrap();
        assert!((t1 - t2).abs() < 1e-9, "same angle: {t1} vs {t2}");
        let a1 = rotation_axis(&flat, &to_x).expect("a 90 degree turn has an axis");
        let a2 = rotation_axis(&flat, &to_y).expect("a 90 degree turn has an axis");
        assert!(axis_agreement(&a1, &a2) > 89.0, "the two axes must be far apart, got {}",
                axis_agreement(&a1, &a2));
    }

    #[test]
    fn a_rotation_too_small_to_localise_reports_no_axis() {
        let a = posture_of(&still(8, (0.0, 0.0, 1.0))).unwrap();
        let b = posture_of(&still(8, (0.001, 0.0, 1.0))).unwrap();
        assert_eq!(rotation_axis(&a, &b), None, "under MIN_AXIS_DEG the cross product is noise");
        assert_eq!(rotation_axis(&a, &a), None, "no rotation, no axis");
    }

    #[test]
    fn the_axis_floor_is_where_it_says_it_is() {
        assert_eq!(MIN_AXIS_DEG, 1.0);
        let flat = posture_of(&still(8, (0.0, 0.0, 1.0))).unwrap();
        let just_under = (0.9f64).to_radians();
        let just_over = (1.1f64).to_radians();
        let p = |a: f64| posture_of(&still(8, (a.sin(), 0.0, a.cos()))).unwrap();
        assert_eq!(rotation_axis(&flat, &p(just_under)), None);
        assert!(rotation_axis(&flat, &p(just_over)).is_some());
    }

    /// Single-axis motion is what a swinging arm looks like; equal motion on all three is what a
    /// tumbling one does. The scalar magnitude is IDENTICAL in both cases here by construction.
    #[test]
    fn planar_motion_is_anisotropic_and_tumbling_motion_is_not() {
        let planar: Vec<AccelSample> =
            (0..12).map(|i| s(i, if i % 2 == 0 { 0.0 } else { 0.3 }, 0.0, 1.0)).collect();
        let tumbling: Vec<AccelSample> = (0..12)
            .map(|i| {
                let d = if i % 2 == 0 { 0.0 } else { 0.3 / 3.0f64.sqrt() };
                s(i, d, d, 1.0 + d)
            })
            .collect();
        let (ap, dp) = anisotropy_of(&planar).unwrap();
        let (at, _) = anisotropy_of(&tumbling).unwrap();
        assert!(ap > 0.99, "planar motion is one axis: {ap}");
        assert_eq!(dp, 0, "and that axis is x");
        assert!((at - 1.0 / 3.0).abs() < 0.05, "equal on three axes is 1/3: {at}");
        assert!(ap > at + 0.5);
    }

    #[test]
    fn a_still_epoch_has_no_plane_rather_than_an_isotropic_one() {
        assert_eq!(anisotropy_of(&still(12, (0.0, 0.0, 1.0))), None, "no motion, no plane");
        assert_eq!(anisotropy_of(&still(2, (0.0, 0.0, 1.0))), None, "under MIN_DELTAS");
    }

    #[test]
    fn a_series_misaligned_with_its_postures_is_empty_not_shifted() {
        let g = still(60, (0.0, 0.0, 1.0));
        let p = super::super::posture::posture_series(&g, 0, 60, 30);
        assert_eq!(p.len(), 2);
        assert_eq!(movement_series(&g, &p, 0, 60, 30).len(), 2);
        // One posture short: a silently shifted alignment would be worse than no answer.
        assert!(movement_series(&g, &p[..1], 0, 60, 30).is_empty());
    }

    #[test]
    fn axis_agreement_is_absent_where_either_axis_is() {
        let m = vec![
            Movement { axis: None, anisotropy: None, dominant: None },
            Movement { axis: Some([0.0, 0.0, 1.0]), anisotropy: None, dominant: None },
            Movement { axis: Some([0.0, 1.0, 0.0]), anisotropy: None, dominant: None },
        ];
        let a = axis_agreement_series(&m);
        assert_eq!(a[0], None, "nothing precedes the first");
        assert_eq!(a[1], None, "the previous epoch has no axis");
        assert!((a[2].unwrap() - 90.0).abs() < 1e-6, "orthogonal axes are 90 degrees apart");
    }
}
