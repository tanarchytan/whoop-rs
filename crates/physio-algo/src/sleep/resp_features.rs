//! Respiration from the beat series: respiratory sinus arrhythmia over one window.
//!
//! One producer for the stager and the screening harnesses, as `cardiac` is for the interval order
//! statistics. The band, the beat floor and the summed HF power come from `hrv_bands`; the breath
//! cycles are this module's own band-passed surrogate of the same tachogram.

use std::f64::consts::PI;

use super::hrv_bands::{self, HF, MIN_BEATS, MIN_COVERAGE};
use super::resp_regularity;
use crate::stats::{mean, median, population_sd};

/// Samples on the analysis grid. A power of two, so ONE transform serves the band peak, the band
/// power and the band-pass; `hrv_bands` fixes the RATE instead and its grid is private.
const N: usize = 1024;

/// Fewest complete cycles a breath statistic needs. Under it the spread is reading two breaths.
pub const MIN_BREATHS: usize = 6;

/// Points each cycle is resampled to before consecutive cycles are correlated.
const CYCLE_POINTS: usize = 16;

/// Column names of [`RespBlock::row`], in its order.
pub const NAMES: [&str; 9] = [
    "resp_freq",
    "resp_freq_sd",
    "rsa_peak_power",
    "rsa_hf_power",
    "rsa_amp_median",
    "breath_len_sd",
    "breath_corr_mean",
    "breath_corr_sd",
    "resp_conc",
];

/// One window's respiratory block. Every breath-level statistic is `None` when the window carries
/// fewer than [`MIN_BREATHS`] cycles, which the spectral pair can survive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RespBlock {
    /// Frequency of the largest bin inside the RSA band, Hz.
    pub freq: f64,
    /// Spread of the cycle-wise rate `1 / length`, Hz.
    pub freq_sd: Option<f64>,
    /// `ln(1 + p)` of that largest bin's power, ms^2.
    pub peak_power: f64,
    /// `ln(1 + p)` of the summed HF band power, taken from `hrv_bands` rather than recomputed.
    /// `None` unless the window is exactly `hrv_bands::WINDOW_S` long, which is what that reads.
    pub hf_power: Option<f64>,
    /// Median peak-to-trough swing of the band-passed cycles, ms.
    pub amp_median: Option<f64>,
    /// Spread of the cycle lengths, s.
    pub breath_len_sd: Option<f64>,
    /// Mean of the correlation between each cycle's shape and the next.
    pub breath_corr_mean: Option<f64>,
    /// Spread of the same correlations.
    pub breath_corr_sd: Option<f64>,
    /// Peak-over-sum concentration inside the band, the one respiratory term the shipped recipe
    /// already reads.
    pub conc: Option<f64>,
}

impl RespBlock {
    /// The block as one row in [`NAMES`] order, a missing statistic as NaN.
    pub fn row(&self) -> [f64; 9] {
        [
            self.freq,
            self.freq_sd.unwrap_or(f64::NAN),
            self.peak_power,
            self.hf_power.unwrap_or(f64::NAN),
            self.amp_median.unwrap_or(f64::NAN),
            self.breath_len_sd.unwrap_or(f64::NAN),
            self.breath_corr_mean.unwrap_or(f64::NAN),
            self.breath_corr_sd.unwrap_or(f64::NAN),
            self.conc.unwrap_or(f64::NAN),
        ]
    }
}

/// In-place radix-2 decimation-in-time transform over [`N`] points.
fn fft(re: &mut [f64; N], im: &mut [f64; N]) {
    let mut j = 0usize;
    for i in 1..N {
        let mut bit = N >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= N {
        let ang = -2.0 * PI / len as f64;
        for start in (0..N).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (ur, ui) = (re[a], im[a]);
                let (vr, vi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
            }
        }
        len <<= 1;
    }
}

/// Linear-interpolated, mean-removed tachogram over `[t0, t1]` on [`N`] samples. Samples outside the
/// beats are zero, which is the mean and so adds no power. `None` when the window cannot carry one.
fn tachogram(beats: &[(f64, f64)], t0: f64, t1: f64) -> Option<[f64; N]> {
    let win: Vec<(f64, f64)> = beats.iter().copied().filter(|(t, _)| *t >= t0 && *t <= t1).collect();
    let span = t1 - t0;
    if win.len() < MIN_BEATS || span <= 0.0 {
        return None;
    }
    if win.iter().map(|(_, ms)| ms / 1000.0).sum::<f64>() < MIN_COVERAGE * span {
        return None;
    }
    let dt = span / N as f64;
    let (first, last) = (win[0].0, win[win.len() - 1].0);
    let mut y = [0.0f64; N];
    let mut held = [false; N];
    let mut seg = 0usize;
    for (i, (yi, hi)) in y.iter_mut().zip(held.iter_mut()).enumerate() {
        let t = t0 + i as f64 * dt;
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
    let m = y.iter().zip(&held).filter(|(_, h)| **h).map(|(v, _)| *v).sum::<f64>() / k as f64;
    for (v, h) in y.iter_mut().zip(&held) {
        *v = if *h { *v - m } else { 0.0 };
    }
    Some(y)
}

/// Bin range of the RSA band on a `span`-second window. Bin `k` sits at `k / span` Hz.
fn band_bins(span: f64) -> Option<(usize, usize)> {
    let lo = (HF.0 * span).ceil() as usize;
    let hi = (HF.1 * span).floor() as usize;
    (lo >= 1 && hi >= lo && hi < N / 2).then_some((lo, hi))
}

/// One breath of the surrogate: its length, its swing, and its shape on a common grid.
struct Cycle {
    len: f64,
    amp: f64,
    shape: [f64; CYCLE_POINTS],
}

/// Cycles between successive upward zero crossings. A span outside the band's own period range is
/// not a breath and is dropped rather than widening the spread.
fn cycles(y: &[f64; N], dt: f64) -> Vec<Cycle> {
    let mut marks: Vec<(usize, f64)> = Vec::new();
    for i in 1..N {
        if y[i - 1] < 0.0 && y[i] >= 0.0 {
            let frac = -y[i - 1] / (y[i] - y[i - 1]);
            marks.push((i, (i - 1) as f64 + frac));
        }
    }
    let (min_len, max_len) = (1.0 / HF.1, 1.0 / HF.0);
    let mut out = Vec::new();
    for w in marks.windows(2) {
        let len = (w[1].1 - w[0].1) * dt;
        if !(min_len..=max_len).contains(&len) {
            continue;
        }
        let seg = &y[w[0].0..w[1].0];
        if seg.len() < 2 {
            continue;
        }
        let hi = seg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = seg.iter().copied().fold(f64::INFINITY, f64::min);
        let mut shape = [0.0f64; CYCLE_POINTS];
        for (p, s) in shape.iter_mut().enumerate() {
            let x = p as f64 * (seg.len() - 1) as f64 / (CYCLE_POINTS - 1) as f64;
            let a = x.floor() as usize;
            let b = (a + 1).min(seg.len() - 1);
            *s = seg[a] + (x - a as f64) * (seg[b] - seg[a]);
        }
        out.push(Cycle { len, amp: hi - lo, shape });
    }
    out
}

/// Pearson correlation, `None` when either side has no spread.
fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let (ma, mb) = (mean(a), mean(b));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        num += (x - ma) * (y - mb);
        da += (x - ma).powi(2);
        db += (y - mb).powi(2);
    }
    (da > 1e-18 && db > 1e-18).then(|| num / (da * db).sqrt())
}

/// The block over `[t0, t1]`, or `None` when the window cannot carry a tachogram. `beats` is
/// ascending in time.
pub fn extract(beats: &[(f64, f64)], t0: f64, t1: f64) -> Option<RespBlock> {
    let span = t1 - t0;
    let y = tachogram(beats, t0, t1)?;
    let (k_lo, k_hi) = band_bins(span)?;

    let mut re = y;
    let mut im = [0.0f64; N];
    fft(&mut re, &mut im);

    let scale = (N * N) as f64;
    let (mut k_max, mut p_max) = (k_lo, f64::NEG_INFINITY);
    for k in k_lo..=k_hi {
        let p = (re[k] * re[k] + im[k] * im[k]) / scale;
        if p > p_max {
            p_max = p;
            k_max = k;
        }
    }

    // Keep the band and its conjugate half, then transform back: the same transform gives the
    // surrogate the breath cycles are counted on.
    let (mut br, mut bi) = ([0.0f64; N], [0.0f64; N]);
    for k in k_lo..=k_hi {
        br[k] = re[k];
        bi[k] = im[k];
        br[N - k] = re[N - k];
        bi[N - k] = im[N - k];
    }
    for v in bi.iter_mut() {
        *v = -*v;
    }
    fft(&mut br, &mut bi);
    for v in br.iter_mut() {
        *v /= N as f64;
    }

    let cy = cycles(&br, span / N as f64);
    let enough = cy.len() >= MIN_BREATHS;
    let lens: Vec<f64> = cy.iter().map(|c| c.len).collect();
    let corr: Vec<f64> = cy.windows(2).filter_map(|w| pearson(&w[0].shape, &w[1].shape)).collect();
    let paired = corr.len() + 1 >= MIN_BREATHS;

    let win: Vec<(f64, f64)> = beats.iter().copied().filter(|(t, _)| *t >= t0 && *t <= t1).collect();

    Some(RespBlock {
        freq: k_max as f64 / span,
        freq_sd: enough.then(|| population_sd(&lens.iter().map(|l| 1.0 / l).collect::<Vec<_>>())),
        peak_power: p_max.max(0.0).ln_1p(),
        hf_power: ((span - hrv_bands::WINDOW_S).abs() < 1e-6)
            .then(|| hrv_bands::bands_at(beats, t0))
            .flatten()
            .map(|b| b.hf.ln_1p()),
        amp_median: enough.then(|| median(&cy.iter().map(|c| c.amp).collect::<Vec<_>>())),
        breath_len_sd: enough.then(|| population_sd(&lens)),
        breath_corr_mean: paired.then(|| mean(&corr)),
        breath_corr_sd: paired.then(|| population_sd(&corr)),
        conc: resp_regularity(&win),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN: f64 = hrv_bands::WINDOW_S;

    /// Beats whose interval oscillates at `f` Hz around 1000 ms, sampled at every beat.
    fn breathing(f: f64, secs: f64, amp: f64) -> Vec<(f64, f64)> {
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
    fn the_peak_frequency_is_the_rate_the_beats_were_modulated_at() {
        for f in [0.16, 0.20, 0.25, 0.33] {
            let b = extract(&breathing(f, WIN + 5.0, 40.0), 0.0, WIN).expect("270 s of beats");
            assert!((b.freq - f).abs() < 0.01, "asked {f}, read {}", b.freq);
        }
    }

    #[test]
    fn the_breath_count_matches_the_rate_and_the_lengths_are_steady() {
        let b = extract(&breathing(0.25, WIN + 5.0, 40.0), 0.0, WIN).unwrap();
        // 0.25 Hz is a 4 s breath: the length spread of a pure tone is a rounding artefact.
        assert!(b.breath_len_sd.unwrap() < 0.05, "{:?}", b.breath_len_sd);
        assert!((b.freq_sd.unwrap()) < 0.005, "{:?}", b.freq_sd);
        // A pure tone repeats exactly, so consecutive cycles correlate at ~1 with no spread.
        assert!(b.breath_corr_mean.unwrap() > 0.99, "{:?}", b.breath_corr_mean);
        assert!(b.breath_corr_sd.unwrap() < 0.01, "{:?}", b.breath_corr_sd);
    }

    #[test]
    fn the_amplitude_tracks_the_modulation_depth() {
        let small = extract(&breathing(0.25, WIN + 5.0, 10.0), 0.0, WIN).unwrap();
        let large = extract(&breathing(0.25, WIN + 5.0, 40.0), 0.0, WIN).unwrap();
        assert!(large.amp_median.unwrap() > 3.0 * small.amp_median.unwrap(),
                "{:?} vs {:?}", small.amp_median, large.amp_median);
        assert!(large.peak_power > small.peak_power);
        assert!(large.hf_power.unwrap() > small.hf_power.unwrap());
    }

    /// An irregular breather must be separable from a metronome on the regularity columns, which is
    /// the whole reason they are here.
    #[test]
    fn an_irregular_rhythm_spreads_the_lengths_a_steady_one_does_not() {
        let steady = extract(&breathing(0.25, WIN + 5.0, 40.0), 0.0, WIN).unwrap();
        // Frequency wandering between 0.17 and 0.33 Hz over the window.
        let mut out = Vec::new();
        let (mut t, mut phase) = (0.0, 0.0);
        while t < WIN + 5.0 {
            let f = 0.25 + 0.08 * (2.0 * PI * 0.01 * t).sin();
            phase += 2.0 * PI * f * 0.001;
            let rr = 1000.0 + 40.0 * phase.sin();
            out.push((t, rr));
            t += rr / 1000.0;
        }
        let wobbly = extract(&out, 0.0, WIN).unwrap();
        assert!(wobbly.breath_len_sd.unwrap() > 5.0 * steady.breath_len_sd.unwrap(),
                "steady {:?} wobbly {:?}", steady.breath_len_sd, wobbly.breath_len_sd);
    }

    /// Two in-band tones beat against each other and add crossings the band-pass cannot remove, so
    /// the period gate is what keeps a half-cycle out of the breath statistics.
    #[test]
    fn a_span_outside_the_bands_period_range_is_not_counted_as_a_breath() {
        let dt = WIN / N as f64;
        let wave = |f: f64| -> [f64; N] {
            std::array::from_fn(|i| (2.0 * PI * f * i as f64 * dt).sin())
        };
        assert!(cycles(&wave(1.0), dt).is_empty(), "a 1 s cycle is not a breath");
        assert!(cycles(&wave(0.25), dt).len() >= MIN_BREATHS, "a 4 s cycle is");
    }

    #[test]
    fn too_few_beats_report_nothing_rather_than_a_spectrum_of_the_interpolation() {
        let sparse: Vec<(f64, f64)> = (0..10).map(|i| (i as f64 * 20.0, 1000.0)).collect();
        assert_eq!(extract(&sparse, 0.0, WIN), None);
        assert_eq!(extract(&[], 0.0, WIN), None);
    }

    /// Pins the beat floor SEPARATELY from the coverage floor. At a normal rate the coverage rule
    /// binds first and hides this one, which is exactly what the sparse case above cannot tell apart.
    #[test]
    fn the_beat_floor_is_where_it_says_it_is() {
        assert_eq!(MIN_BEATS, 30);
        // Long intervals, so coverage passes either side of the floor and only the count decides.
        let at = |n: usize| {
            let mut t = 0.0;
            let b: Vec<(f64, f64)> = (0..n)
                .map(|i| {
                    let rr = 5000.0 + 200.0 * (i as f64).sin();
                    t += rr / 1000.0;
                    (t, rr)
                })
                .collect();
            extract(&b, 0.0, WIN).is_some()
        };
        assert!(!at(MIN_BEATS - 1), "one beat under the floor must not produce a block");
        assert!(at(MIN_BEATS + 1), "one beat over it must");
    }

    /// A burst of beats cannot describe the window it sits in, the same rule `hrv_bands` applies.
    #[test]
    fn a_burst_is_rejected_rather_than_scored_on_its_own_span() {
        let burst: Vec<(f64, f64)> = (0..40).map(|i| (i as f64 * 0.27, 270.0)).collect();
        assert!(burst.len() > MIN_BEATS);
        assert_eq!(extract(&burst, 0.0, WIN), None);
    }

    /// A flat tachogram has no breathing in it, so the cycle statistics must be absent rather than
    /// a manufactured zero. The spectral pair still reports, at zero power.
    #[test]
    fn a_flat_rhythm_has_no_cycles_rather_than_perfect_ones() {
        let flat: Vec<(f64, f64)> = (0..300).map(|i| (i as f64, 1000.0)).collect();
        let b = extract(&flat, 0.0, WIN).unwrap();
        assert_eq!(b.breath_len_sd, None);
        assert_eq!(b.amp_median, None);
        assert_eq!(b.breath_corr_mean, None);
        assert!(b.peak_power < 1e-9, "{}", b.peak_power);
    }

    #[test]
    fn the_row_is_in_names_order_and_carries_nan_for_a_missing_statistic() {
        let b = extract(&breathing(0.25, WIN + 5.0, 40.0), 0.0, WIN).unwrap();
        let r = b.row();
        assert_eq!(r.len(), NAMES.len());
        assert_eq!(r[0], b.freq);
        assert_eq!(r[3], b.hf_power.unwrap());
        let missing = RespBlock { freq_sd: None, hf_power: None, ..b };
        assert!(missing.row()[1].is_nan() && missing.row()[3].is_nan());
    }

    /// The HF column is `hrv_bands`' own number, not a second one. If they ever diverge, one of the
    /// two has been re-implemented.
    #[test]
    fn the_hf_column_is_the_hrv_bands_number_and_not_a_second_copy() {
        let beats = breathing(0.25, WIN + 5.0, 40.0);
        let b = extract(&beats, 0.0, WIN).unwrap();
        let want = hrv_bands::bands_at(&beats, 0.0).unwrap().hf.ln_1p();
        assert_eq!(b.hf_power, Some(want));
    }

    /// A window of another length cannot borrow that number, because `bands_at` would be reading a
    /// different span than the one asked for.
    #[test]
    fn a_window_that_is_not_the_band_window_reports_no_hf_rather_than_the_wrong_one() {
        let beats = breathing(0.25, 400.0, 40.0);
        let b = extract(&beats, 0.0, 200.0).expect("200 s still carries a tachogram");
        assert_eq!(b.hf_power, None);
        assert!(b.peak_power.is_finite(), "the window's own spectrum is still measurable");
    }

    /// Pins the band to `hrv_bands`. Widening it silently would change what counts as a breath at
    /// both ends: the period gate and the bin range are the same two numbers.
    #[test]
    fn the_band_is_the_one_hrv_bands_defines() {
        assert_eq!(HF, (0.15, 0.40));
        let (lo, hi) = band_bins(hrv_bands::WINDOW_S).unwrap();
        assert_eq!((lo, hi), (41, 108), "0.15..0.40 Hz on a 270 s window");
        // A tone just outside the band must not be read as the respiratory rate.
        let b = extract(&breathing(0.10, WIN + 5.0, 40.0), 0.0, WIN).unwrap();
        assert!(b.freq >= HF.0, "an LF tone must not pull the peak below the band: {}", b.freq);
    }

    /// Pins MIN_BREATHS. A mutation sweep is what found `hrv_bands`' floors could move unnoticed.
    #[test]
    fn the_breath_floor_is_where_it_says_it_is() {
        assert_eq!(MIN_BREATHS, 6);
        // 0.16 Hz over 30 s of the window's beats is under the floor; over the whole window is not.
        let short = breathing(0.16, WIN + 5.0, 40.0);
        let b = extract(&short, 0.0, 30.0);
        assert!(b.is_none() || b.unwrap().breath_len_sd.is_none(), "5 breaths is under the floor");
        assert!(extract(&short, 0.0, WIN).unwrap().breath_len_sd.is_some());
    }

    #[test]
    fn the_concentration_column_is_the_shipped_term_over_the_same_window() {
        let beats = breathing(0.25, WIN + 5.0, 40.0);
        let b = extract(&beats, 0.0, WIN).unwrap();
        let win: Vec<(f64, f64)> = beats.iter().copied().filter(|(t, _)| *t <= WIN).collect();
        assert_eq!(b.conc, resp_regularity(&win));
    }
}
