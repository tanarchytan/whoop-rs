//! Approximate sleeping respiratory rate (breaths/min) from the R-R interval stream via respiratory
//! sinus arrhythmia: reconstruct a beat-interval tachogram, resample to 4 Hz, detrend, then per 5-min
//! window peak-pick the breathing modulation and take the median rate. Pure; `None` = no usable estimate.
//! The only respiratory source for both generations: the 4.0 v24 register decoded as `resp_raw` carries
//! a status byte, not a breathing signal, so nothing here reads it (measured; see `docs/algorithms.md`).

use crate::signal::{find_peaks, moving_average_centred};
use crate::stats::{median, population_sd};

const RR_MIN_MS: f64 = 300.0;
const RR_MAX_MS: f64 = 2000.0;

const RSA_RESAMPLE_HZ: f64 = 4.0;
const RSA_DETREND_WINDOW_S: f64 = 8.0;
const RSA_MIN_PEAK_DISTANCE_S: f64 = 2.5;
const RSA_WINDOW_S: f64 = 300.0;
const RSA_MIN_BREATH_INTERVAL_S: f64 = 2.5;
const RSA_MAX_BREATH_INTERVAL_S: f64 = 10.0;

/// Plausible sleeping-respiratory-rate band (bpm); an estimate outside it is `None`.
pub const RESP_PLAUSIBLE_MIN_BPM: f64 = 8.0;
pub const RESP_PLAUSIBLE_MAX_BPM: f64 = 25.0;

/// Sleeping respiratory rate over the in-bed `[start, end]` window (unix seconds), from `(ts, rr_ms)`
/// beats. An on-device wellness estimate, never a clinical measurement; `None` on too-little data.
pub fn resp_rate_from_rr(rr: &[(i64, u16)], start: i64, end: i64) -> Option<f64> {
    if end <= start {
        return None;
    }

    let mut in_bed: Vec<(i64, f64)> = rr
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(ts, ms)| (*ts, *ms as f64))
        .collect();
    in_bed.sort_by_key(|(ts, _)| *ts);
    in_bed.retain(|(_, ms)| *ms >= RR_MIN_MS && *ms <= RR_MAX_MS);
    if in_bed.len() < 30 {
        return None;
    }

    // Split where the clock advanced further than the beats account for. Cumulative-summing across a
    // dropout stitches the gap shut and the peak-picker reads the join as a breath; every night of
    // real wrist data carries such gaps. Within a run the cumulative sum is still what builds the
    // tachogram, because several beats share one whole-second stamp and the stamps alone are too
    // coarse to interpolate on.
    let mut rates: Vec<f64> = Vec::new();
    let mut run_start = 0usize;
    for i in 1..=in_bed.len() {
        let split = i == in_bed.len() || (in_bed[i].0 - in_bed[i - 1].0) as f64 > RR_GAP_S;
        if !split {
            continue;
        }
        window_rates(&in_bed[run_start..i], &mut rates);
        run_start = i;
    }
    resp_rate_of(rates)
}

/// Clock advance between consecutive kept beats above which the stream is treated as broken rather than
/// slow. A 2 s beat is the slowest [`RR_MAX_MS`] allows, so this tolerates several dropped or
/// out-of-band beats before declaring a gap.
const RR_GAP_S: f64 = 10.0;

/// Per-window breathing rates for ONE contiguous run of beats, appended to `out`. A run too short to
/// hold a window contributes nothing, which is why a gappy night degrades rather than disappears.
fn window_rates(run: &[(i64, f64)], out: &mut Vec<f64>) {
    if run.len() < 8 {
        return;
    }
    let filtered: Vec<f64> = run.iter().map(|(_, ms)| *ms).collect();
    let mut beat_times = vec![0.0; filtered.len()];
    let mut acc = 0.0;
    for (i, &ms) in filtered.iter().enumerate() {
        acc += ms / 1000.0;
        beat_times[i] = acc;
    }
    let total_span_s = beat_times[beat_times.len() - 1];
    if total_span_s < RSA_WINDOW_S / 2.0 {
        return;
    }

    let dt = 1.0 / RSA_RESAMPLE_HZ;
    let n_grid = (total_span_s / dt) as usize + 1;
    if n_grid < 8 {
        return;
    }
    let mut grid = vec![0.0; n_grid];
    let mut seg = 0usize;
    for (g, cell) in grid.iter_mut().enumerate() {
        let t = g as f64 * dt;
        while seg < beat_times.len() - 2 && beat_times[seg + 1] < t {
            seg += 1;
        }
        let (t0, t1) = (beat_times[seg], beat_times[seg + 1]);
        let (v0, v1) = (filtered[seg], filtered[seg + 1]);
        *cell = if t1 <= t0 {
            v0
        } else {
            let frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            v0 + frac * (v1 - v0)
        };
    }

    let half_w = ((RSA_DETREND_WINDOW_S * RSA_RESAMPLE_HZ / 2.0).round() as usize).max(1);
    let baseline = moving_average_centred(&grid, 2 * half_w + 1);
    let detrended: Vec<f64> = (0..n_grid).map(|i| grid[i] - baseline[i]).collect();
    if population_sd(&detrended) <= 1e-9 {
        return;
    }

    let min_dist = ((RSA_MIN_PEAK_DISTANCE_S * RSA_RESAMPLE_HZ).round() as usize).max(2);
    let window_samples = ((RSA_WINDOW_S * RSA_RESAMPLE_HZ).round() as usize).max(min_dist * 3);
    let mut w = 0usize;
    while w < n_grid {
        let w_end = (w + window_samples).min(n_grid);
        if w_end - w >= min_dist * 3 {
            let peaks = find_peaks(&detrended[w..w_end], min_dist, 0.0);
            if peaks.len() >= 3 {
                let mut intervals = Vec::with_capacity(peaks.len() - 1);
                for k in 1..peaks.len() {
                    let iv_s = (peaks[k] - peaks[k - 1]) as f64 * dt;
                    if (RSA_MIN_BREATH_INTERVAL_S..=RSA_MAX_BREATH_INTERVAL_S).contains(&iv_s) {
                        intervals.push(iv_s);
                    }
                }
                if intervals.len() >= 2 {
                    let med = median(&intervals);
                    if med > 0.0 {
                        out.push(60.0 / med);
                    }
                }
            }
        }
        w += window_samples;
    }
}

/// The night's rate from every window every run contributed: the median, gated to the plausible band.
fn resp_rate_of(rates: Vec<f64>) -> Option<f64> {
    if rates.is_empty() {
        return None;
    }
    let m = median(&rates);
    (RESP_PLAUSIBLE_MIN_BPM..=RESP_PLAUSIBLE_MAX_BPM).contains(&m).then_some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split, proved by the only construction that can prove it: fragments each too short to hold a
    /// window, carrying real modulation so the flat-tachogram early return cannot be what rejects them.
    ///
    /// Split, no run reaches the minimum span and the night is honestly unscorable. Spliced, the same
    /// beats concatenate into one long run and the function returns a number built entirely out of
    /// joins. Removing the gap test makes this test fail, which is why it is written this way and not
    /// as an assertion about a whole night's rate - a night's rate is a median over ~90 windows, and a
    /// median absorbs a handful of corrupted ones without moving. Measured on 151 real nights across
    /// five stores: only 30 move by more than 0.05 bpm, and by at most 0.9.
    #[test]
    fn fragments_too_short_for_a_window_are_not_concatenated_into_one() {
        let start = 1_700_000_000_i64;
        let mut rows: Vec<(i64, u16)> = Vec::new();
        for f in 0..40_i64 {
            // 100 s of 15 bpm breathing: real modulation, but under RSA_WINDOW_S / 2.
            let base = start + f * 3600;
            let mut t = 0.0_f64;
            while t < 100.0 {
                let rr = 1000.0 + 40.0 * (2.0 * std::f64::consts::PI * 0.25 * t).sin();
                t += rr / 1000.0;
                rows.push((base + t as i64, rr as u16));
            }
        }
        let end = rows.last().unwrap().0;
        assert!(rows.len() > 30, "the beat-count floor must not be what rejects this");
        assert!(
            rows.windows(2).any(|w| (w[1].1 as i64 - w[0].1 as i64).abs() > 10),
            "the tachogram must vary, or the flat-signal early return is what answers"
        );
        assert_eq!(
            resp_rate_from_rr(&rows, start, end),
            None,
            "40 fragments an hour apart are not one 4000-second night"
        );
    }

    /// Pins RR_GAP_S. The fragment test above proves a split HAPPENS; it does not pin WHERE, and a
    /// mutation sweep raised the threshold 100x with nothing noticing because its fragments sat an
    /// hour apart. A gap just over the constant must split; one just under must not.
    #[test]
    fn the_gap_threshold_is_where_it_says_it_is() {
        assert_eq!(RR_GAP_S, 10.0);
        let two_runs = |gap: i64| {
            let (a, start, mid) = synth(0.25, 1000.0, 40.0, 400.0);
            let (b, _, _) = synth(0.25, 1000.0, 40.0, 400.0);
            let mut rows = a;
            rows.extend(b.iter().map(|(t, ms)| (t - start + mid + gap, *ms)));
            let end = rows.last().unwrap().0;
            resp_rate_from_rr(&rows, start, end)
        };
        // Under the threshold the halves are one run, so the joined span clears the window minimum.
        assert!(two_runs(RR_GAP_S as i64 - 2).is_some(), "a sub-threshold gap must not split");
        // Over it they are two runs, each 400 s, each still long enough to score - so this asserts the
        // split happened by its effect on a run too short to survive one.
        let (short, s0, _) = synth(0.25, 1000.0, 40.0, 100.0);
        let mut frag = short.clone();
        frag.extend(short.iter().map(|(t, ms)| (t - s0 + s0 + 600, *ms)));
        let fe = frag.last().unwrap().0;
        assert_eq!(resp_rate_from_rr(&frag, s0, fe), None,
            "two 100 s fragments 600 s apart are two runs, neither long enough to score");
    }

    /// Pins the plausible-band ceiling. The sweep raised it 30% with nothing noticing.
    #[test]
    fn the_plausible_band_rejects_a_rate_above_its_ceiling() {
        assert_eq!(RESP_PLAUSIBLE_MAX_BPM, 25.0);
        assert_eq!(resp_rate_of(vec![RESP_PLAUSIBLE_MAX_BPM + 1.0; 5]), None, "above the ceiling");
        assert_eq!(resp_rate_of(vec![RESP_PLAUSIBLE_MIN_BPM - 1.0; 5]), None, "below the floor");
        assert!(resp_rate_of(vec![15.0; 5]).is_some(), "inside the band");
    }

    /// Synthetic tachogram: mean HR with a known-Hz RSA modulation, so the recovered rate can be
    /// cross-checked against the planted breathing frequency.
    fn synth(breath_hz: f64, base_rr_ms: f64, amp_ms: f64, span_s: f64) -> (Vec<(i64, u16)>, i64, i64) {
        let start = 1_700_000_000_i64;
        let mut rows = Vec::new();
        let mut t_sec = 0.0_f64;
        while t_sec < span_s {
            let rr_ms = base_rr_ms + amp_ms * (2.0 * std::f64::consts::PI * breath_hz * t_sec).sin();
            t_sec += rr_ms / 1000.0;
            rows.push((start + t_sec as i64, rr_ms as u16));
        }
        let end = start + t_sec as i64;
        (rows, start, end)
    }

    /// Planted breathing rates the sweep walks; the 10 bpm span is what no constant can cover.
    const SWEPT_BPM: [f64; 5] = [10.0, 12.0, 15.0, 18.0, 20.0];
    /// `(mean HR, RSA amplitude ms, span s)`: three tachogram densities per planted rate.
    const SWEPT_PROFILES: [(f64, f64, f64); 3] = [(60.0, 40.0, 600.0), (55.0, 45.0, 600.0), (70.0, 30.0, 900.0)];
    /// Worst measured error over the sweep is 2.0 bpm, at 18/min on the 60 bpm profile.
    const SWEPT_TOL_BPM: f64 = 2.5;

    /// Anything that turns an in-bed R-R window into a breathing rate: the shipped function, or a null.
    type RespScorer<'a> = &'a dyn Fn(&[(i64, u16)], i64, i64) -> Option<f64>;

    /// Arms of the sweep a scorer misses, as `(planted, profile HR, returned)`. Empty = it tracks.
    fn sweep_misses(scorer: RespScorer) -> Vec<(f64, f64, Option<f64>)> {
        let mut bad = Vec::new();
        for bpm in SWEPT_BPM {
            for (hr, amp, span) in SWEPT_PROFILES {
                let (rows, start, end) = synth(bpm / 60.0, 60_000.0 / hr, amp, span);
                let got = scorer(&rows, start, end);
                if !got.is_some_and(|v| (v - bpm).abs() <= SWEPT_TOL_BPM) {
                    bad.push((bpm, hr, got));
                }
            }
        }
        bad
    }

    /// The estimate tracks a planted rate across 10-20 breaths/min on three tachogram densities, so
    /// what is measured is the recovery of a VARYING rate, not one lucky tone.
    #[test]
    fn tracks_a_swept_breathing_rate_across_the_band() {
        assert!(sweep_misses(&resp_rate_from_rr).is_empty(), "{:?}", sweep_misses(&resp_rate_from_rr));
    }

    /// The null arm: no constant, and no refusal, survives the sweep. A single 15/min tone alone was
    /// satisfied by the constant 13.0, which also satisfied the slow-breather gate below.
    #[test]
    fn no_constant_breathing_rate_survives_the_sweep() {
        let mut c = RESP_PLAUSIBLE_MIN_BPM;
        while c <= RESP_PLAUSIBLE_MAX_BPM {
            assert!(!sweep_misses(&|_, _, _| Some(c)).is_empty(), "constant {c} passed the sweep");
            c += 0.5;
        }
        assert!(!sweep_misses(&|_, _, _| None).is_empty(), "a refusing scorer passed the sweep");
        // The old single-tone pair was blind to exactly this value.
        assert!(!sweep_misses(&|_, _, _| Some(13.0)).is_empty());
    }

    /// Sensitivity, not just accuracy: a 10 bpm move in the truth must move the estimate at least
    /// 6 bpm on every profile. Smallest measured spread is 8.24 bpm.
    #[test]
    fn a_ten_bpm_move_in_the_truth_moves_the_estimate() {
        for (hr, amp, span) in SWEPT_PROFILES {
            let at = |bpm: f64| {
                let (rows, s, e) = synth(bpm / 60.0, 60_000.0 / hr, amp, span);
                resp_rate_from_rr(&rows, s, e).expect("finite estimate")
            };
            let spread = at(20.0) - at(10.0);
            assert!(spread >= 6.0, "HR {hr}: 10 -> 20 bpm moved the estimate only {spread}");
        }
    }

    /// Recorded limit, not a desired one: the 2.5 s peak spacing caps the estimator at 24 breaths/min,
    /// so above ~20 on a slow tachogram adjacent breaths merge and the rate HALVES into a normal-looking
    /// value. `RESP_PLAUSIBLE_MAX_BPM` sits above that cap and does not catch it.
    #[test]
    fn above_the_peak_spacing_cap_the_rate_halves_inside_the_plausible_band() {
        let at = |bpm: f64, hr: f64, span: f64| {
            let (rows, s, e) = synth(bpm / 60.0, 60_000.0 / hr, 45.0, span);
            resp_rate_from_rr(&rows, s, e).expect("finite estimate")
        };
        let halved = at(24.0, 60.0, 600.0);
        assert!((halved - 12.0).abs() < 0.5, "24/min at HR 60 read {halved}");
        assert!((RESP_PLAUSIBLE_MIN_BPM..=RESP_PLAUSIBLE_MAX_BPM).contains(&halved), "and it is in band");
        // The band ceiling sits above the peak-spacing cap, which is why the halved value stays in band.
        const { assert!(60.0 / RSA_MIN_PEAK_DISTANCE_S < RESP_PLAUSIBLE_MAX_BPM) };
        // A faster tachogram carries the same rate, so it is the beat density, not the rate.
        assert!((at(24.0, 70.0, 900.0) - 24.0).abs() <= 1.0);
    }

    #[test]
    fn slow_breather_is_not_doubled() {
        // 11 breaths/min must read ~11, not the doubled ~20-21 harmonic.
        let (rows, start, end) = synth(11.0 / 60.0, 60000.0 / 55.0, 45.0, 480.0);
        let est = resp_rate_from_rr(&rows, start, end).expect("finite estimate");
        assert!((est - 11.0).abs() <= 2.0, "expected ~11, got {est}");
        assert!(est < 16.0, "must not double toward ~22, got {est}");
    }

    #[test]
    fn too_few_beats_is_none() {
        let start = 1_700_000_000_i64;
        let rows = vec![(start + 1, 1000u16), (start + 2, 1000), (start + 3, 1000)];
        assert!(resp_rate_from_rr(&rows, start, start + 10).is_none());
        assert!(resp_rate_from_rr(&[], start, start + 10).is_none());
    }

    #[test]
    fn empty_or_inverted_window_is_none() {
        let (rows, start, end) = synth(0.25, 1000.0, 40.0, 420.0);
        assert!(resp_rate_from_rr(&rows, end, start).is_none());
    }
}
