//! Frequency-domain HRV: the feature family every published stager above kappa 0.5 has and this one
//! has none of.
//!
//! We extract heart rate, RMSSD and a flatness run-length, all time-domain. The literature's stated REM
//! discriminator is "short-term variability in the LF and HF bands with little movement", which is
//! exactly the confusion a time-domain feature cannot resolve.
//!
//! Same front end as the respiration term: tachogram, resample, detrend, band-limited DFT. Only the
//! bands and the window differ. VLF's lower edge needs a long window - 0.015 Hz is one cycle per 67 s -
//! so [`WINDOW_S`] is 9 epochs, not the 3.5 the respiration term uses.

use std::f64::consts::PI;

/// Resample rate of the tachogram, Hz. Five times the top band edge.
const FS: f64 = 4.0;
/// Analysis window. Nine 30-s epochs, so VLF gets four cycles at its lower edge.
pub const WINDOW_S: f64 = 270.0;
/// Fewest beats that can carry a spectrum. Below this the interpolation invents the signal.
pub const MIN_BEATS: usize = 30;
/// Fraction of the window the R-R intervals must themselves account for. Under it the grid is mostly
/// pad and the spectrum describes the padding.
pub const MIN_COVERAGE: f64 = 0.5;

/// Band edges in Hz, standard short-term HRV.
pub const VLF: (f64, f64) = (0.015, 0.04);
pub const LF: (f64, f64) = (0.04, 0.15);
pub const HF: (f64, f64) = (0.15, 0.40);

/// Band powers for one window. Absolute powers are in ms^2 and vary by an order of magnitude between
/// people, so the normalised pair is what a cross-subject model should read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bands {
    pub vlf: f64,
    pub lf: f64,
    pub hf: f64,
    /// LF / HF. `None` when HF is zero, which is not a ratio of infinity.
    pub lf_hf: Option<f64>,
    /// LF / (LF + HF), bounded 0..1 and free of the scale every absolute power carries.
    pub lf_nu: Option<f64>,
}

/// Linear-interpolated, mean-removed tachogram over `[t0, t1]` at [`FS`]. The grid is sized by the
/// WINDOW, never by the beats, so a short burst cannot shrink the DFT until a band loses its bins.
/// Samples outside the beats are zero-padded, which adds no power of its own. `beats` is sorted.
fn tachogram(beats: &[(f64, f64)], t0: f64, t1: f64) -> Option<Vec<f64>> {
    let win: Vec<(f64, f64)> = beats.iter().copied().filter(|(t, _)| *t >= t0 && *t <= t1).collect();
    let span = t1 - t0;
    if win.len() < MIN_BEATS || span <= 0.0 {
        return None;
    }
    // The intervals must account for the window themselves; a beat COUNT cannot see a burst.
    if win.iter().map(|(_, ms)| ms / 1000.0).sum::<f64>() < MIN_COVERAGE * span {
        return None;
    }
    let n = (span * FS).floor() as usize;
    if n < 32 {
        return None;
    }
    let (first, last) = (win[0].0, win[win.len() - 1].0);
    let mut y = vec![0.0f64; n];
    let mut held = vec![false; n];
    let mut seg = 0usize;
    for (i, (yi, hi)) in y.iter_mut().zip(held.iter_mut()).enumerate() {
        let t = t0 + i as f64 / FS;
        if t < first || t > last {
            continue;
        }
        while seg + 2 < win.len() && win[seg + 1].0 < t {
            seg += 1;
        }
        let (ta, va) = win[seg];
        let (tb, vb) = win[seg + 1];
        *yi = if tb <= ta { va } else { va + ((t - ta) / (tb - ta)).clamp(0.0, 1.0) * (vb - va) };
        *hi = true;
    }
    let k = held.iter().filter(|h| **h).count();
    if k == 0 {
        return None;
    }
    let mean = y.iter().zip(&held).filter(|(_, h)| **h).map(|(v, _)| *v).sum::<f64>() / k as f64;
    for (v, h) in y.iter_mut().zip(&held) {
        *v = if *h { *v - mean } else { 0.0 };
    }
    Some(y)
}

/// Summed periodogram power over `[lo, hi)` Hz. Bin `k` sits at `k * FS / n`.
fn band_power(y: &[f64], lo: f64, hi: f64) -> f64 {
    let n = y.len();
    let k_lo = (lo * n as f64 / FS).ceil() as usize;
    let k_hi = (hi * n as f64 / FS).floor() as usize;
    if k_hi < k_lo || k_lo >= n / 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in k_lo..=k_hi.min(n / 2 - 1) {
        let w = -2.0 * PI * k as f64 / n as f64;
        let (mut re, mut im) = (0.0, 0.0);
        for (j, &yj) in y.iter().enumerate() {
            let a = w * j as f64;
            re += yj * a.cos();
            im += yj * a.sin();
        }
        total += (re * re + im * im) / (n * n) as f64;
    }
    total
}

/// Band powers over `[t0, t0 + WINDOW_S]`. `None` when the window cannot carry a spectrum, which is a
/// different fact from "this wearer has no HRV" and must not be mapped to zero by a caller.
pub fn bands_at(beats: &[(f64, f64)], t0: f64) -> Option<Bands> {
    let y = tachogram(beats, t0, t0 + WINDOW_S)?;
    let (vlf, lf, hf) =
        (band_power(&y, VLF.0, VLF.1), band_power(&y, LF.0, LF.1), band_power(&y, HF.0, HF.1));
    Some(Bands {
        vlf,
        lf,
        hf,
        lf_hf: (hf > 0.0).then(|| lf / hf),
        lf_nu: (lf + hf > 0.0).then(|| lf / (lf + hf)),
    })
}

/// One [`Bands`] per epoch, centred on the epoch so a boundary is not read from one side only.
pub fn bands_series(beats: &[(f64, f64)], start: f64, end: f64, epoch_s: f64) -> Vec<Option<Bands>> {
    if epoch_s <= 0.0 || end <= start {
        return Vec::new();
    }
    let n = ((end - start) / epoch_s) as usize;
    (0..n)
        .map(|k| {
            let mid = start + (k as f64 + 0.5) * epoch_s;
            bands_at(beats, mid - WINDOW_S / 2.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tachogram oscillating at `f` Hz around 1000 ms, sampled at every beat.
    fn synthetic(f: f64, secs: f64, amp: f64) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let mut t = 0.0;
        while t < secs {
            let rr = 1000.0 + amp * (2.0 * PI * f * t).sin();
            out.push((t, rr));
            t += rr / 1000.0;
        }
        out
    }

    #[test]
    fn power_lands_in_the_band_the_signal_was_put_in() {
        for (f, name) in [(0.025, "vlf"), (0.10, "lf"), (0.25, "hf")] {
            let b = bands_at(&synthetic(f, WINDOW_S + 5.0, 40.0), 0.0).expect(name);
            let (v, l, h) = (b.vlf, b.lf, b.hf);
            let top = match name {
                "vlf" => v,
                "lf" => l,
                _ => h,
            };
            assert!(top > v + l + h - top, "{name}: {v:.3} {l:.3} {h:.3} - power leaked out of band");
        }
    }

    /// The ratio is the point of the feature: breathing fast moves power HF-ward and LF/HF down.
    #[test]
    fn lf_hf_ratio_follows_where_the_power_is() {
        let slow = bands_at(&synthetic(0.10, WINDOW_S + 5.0, 40.0), 0.0).unwrap();
        let fast = bands_at(&synthetic(0.25, WINDOW_S + 5.0, 40.0), 0.0).unwrap();
        assert!(slow.lf_hf.unwrap() > 1.0, "0.10 Hz should be LF-dominant: {:?}", slow.lf_hf);
        assert!(fast.lf_hf.unwrap() < 1.0, "0.25 Hz should be HF-dominant: {:?}", fast.lf_hf);
        assert!(slow.lf_nu.unwrap() > fast.lf_nu.unwrap());
    }

    #[test]
    fn a_flat_tachogram_has_power_but_no_ratio_rather_than_a_fake_one() {
        let flat: Vec<(f64, f64)> = (0..300).map(|i| (i as f64, 1000.0)).collect();
        let b = bands_at(&flat, 0.0).unwrap();
        assert!(b.vlf < 1e-12 && b.lf < 1e-12 && b.hf < 1e-12, "{b:?}");
        assert_eq!(b.lf_hf, None, "0/0 is not a ratio");
        assert_eq!(b.lf_nu, None);
    }

    #[test]
    fn too_few_beats_report_nothing_rather_than_a_spectrum_of_the_interpolation() {
        let sparse: Vec<(f64, f64)> = (0..10).map(|i| (i as f64 * 20.0, 1000.0)).collect();
        assert_eq!(bands_at(&sparse, 0.0), None, "10 beats is under MIN_BEATS");
        assert_eq!(bands_at(&[], 0.0), None);
    }

    #[test]
    fn a_series_leaves_the_epochs_it_cannot_fill_empty() {
        // Beats for the first half of the span only.
        let beats = synthetic(0.25, 400.0, 30.0);
        let s = bands_series(&beats, 0.0, 900.0, 30.0);
        assert_eq!(s.len(), 30);
        assert!(s.iter().any(|b| b.is_some()), "the covered half should produce spectra");
        assert!(s.last().unwrap().is_none(), "the uncovered tail must not invent one");
    }

    #[test]
    fn the_window_is_long_enough_for_the_lowest_band_edge() {
        assert!(WINDOW_S * VLF.0 >= 4.0, "VLF needs at least four cycles in the window");
    }

    /// Pins MIN_BEATS to its value, not merely to "some floor exists". A mutation sweep found the
    /// constant could be raised 50% with nothing noticing: the sparse-input test used 10 beats, so any
    /// floor above 10 satisfied it.
    #[test]
    fn the_beat_floor_is_where_it_says_it_is() {
        assert_eq!(MIN_BEATS, 30, "changing this changes which windows produce a spectrum at all");
        // Long intervals, so coverage is satisfied either side of the floor and only the count can
        // decide. At a normal rate coverage binds first and this floor is unreachable.
        let at = |n: usize| {
            let mut t = 0.0;
            let b: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let rr = 5000.0 + 200.0 * (i as f64).sin();
                    t += rr / 1000.0;
                    (t, rr)
                })
                .collect();
            bands_at(&b, 0.0).is_some()
        };
        assert!(!at(MIN_BEATS - 1), "one beat under the floor must not produce a spectrum");
        assert!(at(MIN_BEATS + 1), "one beat over it must");
    }

    /// Pins MIN_COVERAGE. The rule reads the interval SUM against the window, so a dense burst fails
    /// it while a slower series spanning the window passes.
    #[test]
    fn the_coverage_floor_is_where_it_says_it_is() {
        assert_eq!(MIN_COVERAGE, 0.5, "half the window must be accounted for by real intervals");
        let covering = |secs: f64| {
            let b: Vec<(f64, f64)> = (0..secs as usize).map(|i| (i as f64, 1000.0)).collect();
            bands_at(&b, 0.0).is_some()
        };
        assert!(!covering(WINDOW_S * MIN_COVERAGE - 10.0), "under the floor must not score");
        assert!(covering(WINDOW_S * MIN_COVERAGE + 10.0), "over it must");
    }

    /// Forty beats inside eleven seconds used to size the DFT to its own span, where the VLF band has
    /// no bin at all and reported exactly 0.0 rather than nothing.
    #[test]
    fn a_burst_of_beats_is_rejected_rather_than_scored_on_its_own_span() {
        let burst: Vec<(f64, f64)> = (0..40).map(|i| (i as f64 * 0.27, 270.0)).collect();
        assert!(burst.len() > MIN_BEATS, "the count must not be what rejects it");
        assert_eq!(bands_at(&burst, 0.0), None, "11 s of beats cannot describe a 270 s window");
    }

    /// The grid is the window's, not the beats'. Sizing it by the observed span is what let a burst
    /// fall below the lowest band's resolution.
    #[test]
    fn the_grid_is_sized_by_the_window_not_by_the_beats() {
        let want = (WINDOW_S * FS) as usize;
        let dense = synthetic(0.05, WINDOW_S + 5.0, 40.0);
        assert_eq!(want, tachogram(&dense, 0.0, WINDOW_S).expect("dense").len());
        let partial: Vec<(f64, f64)> =
            dense.iter().copied().filter(|(t, _)| *t >= 50.0 && *t <= 215.0).collect();
        assert_eq!(want, tachogram(&partial, 0.0, WINDOW_S).expect("partial").len());
    }

    /// The pad sits at the mean of the covered samples, so it is exactly zero once the mean is
    /// removed. Holding the nearest beat's value instead leaves an offset no beat produced.
    #[test]
    fn the_uncovered_samples_are_padded_rather_than_held() {
        // A ramp, so the last beat's value is far from the mean and the two fills differ.
        let beats: Vec<(f64, f64)> = (0..160).map(|i| (i as f64, 800.0 + 2.5 * i as f64)).collect();
        let y = tachogram(&beats, 0.0, WINDOW_S).expect("160 s of 270 is over the coverage floor");
        let tail = &y[(160.0 * FS) as usize..];
        assert!(!tail.is_empty(), "the window must extend past the beats");
        assert!(tail.iter().all(|v| v.abs() < 1e-9), "the pad must be zero, got {}", tail[0]);
    }

    /// A gap must not swamp the band the beats actually carry.
    #[test]
    fn a_gapped_window_still_scores_the_band_the_signal_is_in() {
        let mut t = 0.0;
        let mut beats = Vec::new();
        while t < 160.0 {
            let rr = 1000.0 + 40.0 * (2.0 * PI * 0.25 * t).cos();
            beats.push((t, rr));
            t += rr / 1000.0;
        }
        let b = bands_at(&beats, 0.0).expect("160 s of 270 is over the coverage floor");
        assert!(b.hf > b.vlf + b.lf, "the 0.25 Hz signal must outweigh the gap: {b:?}");
    }

    /// Pins WINDOW_S. Raising it silently would change which epochs can be scored at all, and the
    /// sweep showed a 50% increase passing unnoticed.
    #[test]
    fn the_window_is_the_length_the_band_edges_were_chosen_for() {
        assert_eq!(WINDOW_S, 270.0, "nine 30-second epochs");
        // WINDOW_S sets how far the window REACHES, not a minimum length, so pin the reach: a signal
        // beyond the edge must not reach the answer, and one just inside must.
        let mut split = synthetic(0.25, WINDOW_S - 10.0, 40.0);
        let tail_from = split.last().unwrap().0;
        for (t, v) in synthetic(0.06, 400.0, 40.0) {
            split.push((tail_from + 20.0 + t, v));
        }
        let inside = bands_at(&split, 0.0).unwrap();
        assert!(inside.lf_hf.unwrap() < 1.0, "0.25 Hz inside the window must dominate: {inside:?}");
        // Start the same window past the edge and the far signal, which is LF, takes over.
        let beyond = bands_at(&split, tail_from + 20.0).unwrap();
        assert!(beyond.lf_hf.unwrap() > 1.0, "0.06 Hz beyond it is LF-dominant: {beyond:?}");
    }
}
