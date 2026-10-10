//! HRV-readiness: a log-domain rolling-baseline vs personal-normal-band reading over a nightly RMSSD
//! series. Input = the nightly RMSSD (ms) derived from decoded R-R; output = a tier + the band in ms.
//! Returns `None` (calibrating) below `MIN_NIGHTS` valid nights. Wellness only, never medical.

use crate::stats::{least_squares_slope, mean, median, sample_sd};
use whoop_protocol::HistoryRecord;

/// Seconds per calendar day, the bucket a nightly metric is keyed on.
pub const SECS_PER_DAY: u32 = 86_400;

const HRV_MIN_MS: f64 = 5.0;
const HRV_MAX_MS: f64 = 250.0;
const ROLL_WINDOW: usize = 7;
const LONG_WINDOW: usize = 60;
const LONG_WINDOW_FALLBACK: usize = 30;
const SWC_K: f64 = 0.5;
// The readiness tier mirrors WHOOP's recovery score, which unlocks after 3 recoveries; the 7-night
// baseline and long-window band keep refining past that.
const MIN_NIGHTS: usize = crate::calibration::RECOVERY_SCORE.unlock as usize;
const CV_TREND_WINDOW: usize = 28;
const LONG_SD_FLOOR: f64 = 1e-9;
/// A physiologically plausible R-R interval (ms); values outside are dropped before cleaning.
pub const RR_MIN_MS: u16 = 300;
pub const RR_MAX_MS: u16 = 2000;
/// Width (s) of the tumbling window the app pools per-bucket RMSSD over for its stored session avgHrv.
const HRV_WINDOW_SECS: u64 = 300;
/// A beat-to-beat R-R change beyond this (ms) is an artifact (ectopic/missed beat), not real variability,
/// so its squared difference is dropped from RMSSD — the standard HRV artifact-correction step.
const MAX_BEAT_DELTA_MS: f64 = 200.0;
/// Malik ectopic rejection: a beat deviating over this fraction from its local median is dropped.
pub const ECTOPIC_THRESHOLD: f64 = 0.20;
/// Half-width (beats) of the centred median window; a 5-beat window at radius 2.
pub const ECTOPIC_WINDOW_RADIUS: usize = 2;
/// Minimum clean NN intervals before a full [`HrvReadiness::analyze_raw`] result is trustworthy.
pub const MIN_BEATS: usize = 20;
/// Rejected-fraction ceiling for a SPOT reading: over this share dropped, [`HrvReadiness::analyze_raw`]
/// refuses even when [`MIN_BEATS`] survive. The nightly path passes `None` and skips the gate.
pub const SPOT_MAX_REJECTED_FRACTION: f64 = 0.35;
/// Trailing-window width (s) for [`rolling_rmssd`]: the shortest short-term recording the Task Force
/// recognises, so a window holds enough beats without smoothing away within-night swings.
pub const ROLLING_WINDOW_SECS: i64 = 300;
/// A successive |ΔNN| above this (ms) counts toward pNN50.
const PNN50_THRESHOLD_MS: f64 = 50.0;

/// Where the short baseline sits vs the personal normal band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadinessTier {
    Primed,
    Normal,
    Suppressed,
}

/// A readiness reading. The `*_ms` fields are back in milliseconds (exp of the log-domain the engine
/// works in). `overreaching_watch` is informational and never changes the tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvReadinessResult {
    pub tier: ReadinessTier,
    pub baseline7_ms: f64,
    pub normal_low_ms: f64,
    pub normal_high_ms: f64,
    pub overreaching_watch: bool,
}

/// Full HRV analysis over one raw R-R capture: cleaned RMSSD/SDNN/pNN50/meanNN plus the input and clean
/// beat counts. Fields are `None` (and `n_clean` 0) on an insufficient/refused reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvAnalysis {
    pub rmssd: Option<f64>,
    pub sdnn: Option<f64>,
    pub mean_nn: Option<f64>,
    pub pnn50: Option<f64>,
    pub n_input: u32,
    pub n_clean: u32,
}

/// One tumbling HRV bucket: its start (unix s), the clean beats inside it, and its gap-aware RMSSD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvBucket {
    pub start: u32,
    pub clean_beats: u32,
    pub rmssd: Option<f64>,
}

/// Counts from one cleaning pass: the input beats, the survivors of the range filter, and the survivors
/// of range + Malik ectopic. Ungated, so a trace can report them where `analyze_raw` reports zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrvCleanCounts {
    pub n_input: u32,
    pub n_ranged: u32,
    pub n_clean: u32,
}

/// True when the strap's own optical front-end reported a readable pulse, so that record's beats may be
/// trusted. The argument is the record's `optical_signal_poor`, the amplitude sentinel the front-end
/// raises when it has no amplitude to report; `None` is trusted, since only the 5.0/MG layout carries it.
pub fn rr_trusted(optical_signal_poor: Option<bool>) -> bool {
    optical_signal_poor != Some(true)
}

/// One record's R-R beats carrying the optical-quality flag that decides whether they count. The input
/// [`HrvReadiness::nightly_hrv`] takes, so the [`rr_trusted`] filter cannot be skipped by handing it
/// bare beats — which is how one caller of the nightly quantity came to apply no filter at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RrReport {
    pub unix: u32,
    pub rr: Vec<u16>,
    pub optical_signal_poor: Option<bool>,
}

/// [`covered_rr_coverage`] above which a night's stored R-R is taken as recorded more than once.
/// Measured: healthy nights 0.93-1.10, doubled nights 1.49-2.40.
pub const MAX_RR_COVERAGE: f64 = 1.3;
/// Longest gap (s) between stamps still counted as covered time in [`covered_rr_coverage`].
const COVER_GAP_SECS: i64 = 5;

/// Whether a night's R-R could be trusted for a nightly HRV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HrvQuality {
    Ok,
    DuplicatedBeats { coverage: f64 },
}

/// [`HrvReadiness::nightly_hrv_checked`]: `value` is `None` when refused or when no deep bucket yields one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NightlyHrv {
    pub value: Option<f64>,
    pub quality: HrvQuality,
}

/// Stage-by-stage survivor counts for one R-R series: input, after the range filter, after Malik ectopic.
pub fn clean_counts(rr_ms: &[u16]) -> HrvCleanCounts {
    let ranged = HrvReadiness::range_filter(rr_ms);
    HrvCleanCounts {
        n_input: rr_ms.len() as u32,
        n_ranged: ranged.len() as u32,
        n_clean: HrvReadiness::clean_rr(rr_ms).len() as u32,
    }
}

pub struct HrvReadiness;

impl HrvReadiness {
    /// Full clean-and-analyze over a raw R-R series (ms): range filter -> Malik ectopic -> gap-aware RMSSD +
    /// pNN50, with SDNN + meanNN over the clean beats. Empty (all `None`, `n_clean` 0) when fewer than
    /// [`MIN_BEATS`] clean survive, or -- when `max_rejected_fraction` is set (the spot honesty gate) -- when
    /// the dropped fraction exceeds it. The nightly path passes `None` (no gate), so it is unchanged.
    pub fn analyze_raw(rr_ms: &[u16], max_rejected_fraction: Option<f64>) -> HrvAnalysis {
        let n_input = rr_ms.len();
        let empty = HrvAnalysis {
            rmssd: None,
            sdnn: None,
            mean_nn: None,
            pnn50: None,
            n_input: n_input as u32,
            n_clean: 0,
        };
        let (nn, contiguous) = clean_rr_gap_aware(rr_ms);
        if nn.len() < MIN_BEATS {
            return empty;
        }
        if let Some(max_rej) = max_rejected_fraction {
            if n_input > 0 && 1.0 - nn.len() as f64 / n_input as f64 > max_rej {
                return empty;
            }
        }
        // meanNN over the clean beats; SDNN reuses [`Self::sdnn`] over the same beats (n >= MIN_BEATS here).
        let mean_nn = nn.iter().map(|&v| v as f64).sum::<f64>() / nn.len() as f64;
        HrvAnalysis {
            rmssd: rmssd_from_clean(&nn, &contiguous),
            sdnn: Self::sdnn(&nn),
            mean_nn: Some(mean_nn),
            pnn50: pnn50_from_clean(&nn, &contiguous),
            n_input: n_input as u32,
            n_clean: nn.len() as u32,
        }
    }

    /// Gap-aware, artifact-corrected RMSSD (ms): pools squared successive differences **within** each run
    /// of consecutive beats, never across the break between runs, and drops any single beat-to-beat change
    /// over `MAX_BEAT_DELTA_MS` (an ectopic/missed beat), so neither an offload gap nor an artifact can
    /// inflate it. `None` if no run has two beats. The caller splits R-R into runs on time gaps.
    pub fn rmssd_runs<'a>(runs: impl IntoIterator<Item = &'a [u16]>) -> Option<f64> {
        let (mut sumsq, mut pairs) = (0.0f64, 0usize);
        for run in runs {
            for w in run.windows(2) {
                let d = w[1] as f64 - w[0] as f64;
                if d.abs() > MAX_BEAT_DELTA_MS {
                    continue; // ectopic / missed beat, not physiological variability
                }
                sumsq += d * d;
                pairs += 1;
            }
        }
        (pairs > 0).then(|| (sumsq / pairs as f64).sqrt())
    }

    /// RMSSD (ms) of one run of consecutive R-R beats. `None` for < 2 beats. Uses the run-based
    /// path which drops individual beat deltas > `MAX_BEAT_DELTA_MS`.
    pub fn rmssd(rr_ms: &[u16]) -> Option<f64> {
        Self::rmssd_runs(std::iter::once(rr_ms))
    }

    /// Plain pNN50 (% of successive |dNN| > 50 ms) over an already-clean NN series, every pair counted.
    /// The formula alone, with no cleaning and no contiguity mask. `None` for < 2 values.
    pub fn pnn50_plain(nn: &[u16]) -> Option<f64> {
        pnn50_from_clean(nn, &vec![true; nn.len()])
    }

    /// Plain RMSSD without artifact filtering. Accepts u16 values.
    pub fn rmssd_plain(rr_ms: &[u16]) -> Option<f64> {
        if rr_ms.len() < 2 { return None; }
        let mut sum_sq = 0.0;
        for i in 1..rr_ms.len() {
            let d = rr_ms[i] as f64 - rr_ms[i - 1] as f64;
            sum_sq += d * d;
        }
        Some((sum_sq / (rr_ms.len() - 1) as f64).sqrt())
    }

    /// Range-filter R-R intervals (keep 300–2000 ms).
    pub fn range_filter(rr_ms: &[u16]) -> Vec<u16> {
        rr_ms.iter().copied().filter(|&v| (RR_MIN_MS..=RR_MAX_MS).contains(&v)).collect()
    }

    /// Sample standard deviation (ddof=1) of NN intervals (ms). `None` for < 2 values. Sums in f64 so a
    /// long clean series can't overflow a u16 accumulator.
    pub fn sdnn(rr_ms: &[u16]) -> Option<f64> {
        if rr_ms.len() < 2 {
            return None;
        }
        let mean = rr_ms.iter().map(|&v| v as f64).sum::<f64>() / rr_ms.len() as f64;
        let var = rr_ms.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (rr_ms.len() - 1) as f64;
        Some(var.sqrt())
    }

    /// Range + Malik-ectopic cleaned NN series (ms), in input order. The clean beats only (no contiguity
    /// mask); a successive-difference metric wants [`clean_rr_gap_aware`] instead.
    pub fn clean_rr(rr_ms: &[u16]) -> Vec<u16> {
        clean_rr_gap_aware(rr_ms).0
    }

    /// Gap-aware nightly RMSSD (ms) from one night's per-record `(unix, R-R)` beats. Flattens the beats in time order, range-filters and Malik-ectopic-cleans them, then
    /// pools only successive differences whose two beats were adjacent in the source: a dropped range or
    /// ectopic beat splices its neighbours apart and that difference is skipped. Divides by the
    /// contiguous-pair count. Input need not be sorted. `None` if no contiguous pair survives.
    pub fn rmssd_gap_aware(beats: &[(u32, Vec<u16>)]) -> Option<f64> {
        let mut order: Vec<&(u32, Vec<u16>)> = beats.iter().collect();
        order.sort_by_key(|(t, _)| *t);
        let reports: Vec<(u32, &[u16])> = order.iter().map(|(t, rr)| (*t, rr.as_slice())).collect();
        let flat: Vec<u16> = order.iter().flat_map(|(_, rr)| rr.iter().copied()).collect();
        let breaks = report_seam_breaks(&reports);
        let (nn, contiguous) = clean_rr_gap_aware_breaking(&flat, &breaks);
        rmssd_from_clean(&nn, &contiguous)
    }

    /// Windowed session RMSSD (ms): the mean of per-`HRV_WINDOW_SECS`-bucket gap-aware RMSSD over the
    /// `[start, end]` span — the stored session avgHrv. `beats` are `(unix, rr)` in the
    /// caller's chronological order; buckets tumble from `start`, each range+ectopic-cleaned then
    /// gap-aware-RMSSD'd (kept only with >= 2 clean beats and a surviving contiguous pair). No whole-night
    /// beat-count gate. `None` when no bucket yields a value.
    pub fn windowed_avg_hrv(start: u32, end: u32, beats: &[(u32, u16)]) -> Option<f64> {
        Self::windowed_avg_hrv_inner(start, end, beats, |_| true)
    }

    /// The windowing primitive under [`Self::nightly_hrv`]: per-bucket RMSSD like [windowed_avg_hrv],
    /// keeping only buckets whose center falls inside a `deep_spans` span. Bare beats carry no quality
    /// flag, so this applies NO trust filter — a displayed nightly value comes from `nightly_hrv`.
    pub fn windowed_avg_hrv_deep(start: u32, end: u32, beats: &[(u32, u16)], deep_spans: &[(u32, u32)]) -> Option<f64> {
        // Boxing so the closure outlives this call while the inner function borrows the spans.
        let spans: Vec<(u64, u64)> = deep_spans.iter().map(|&(s, e)| (s as u64, e as u64)).collect();
        Self::windowed_avg_hrv_inner(start, end, beats, move |t| {
            let center = t + HRV_WINDOW_SECS / 2;
            spans.iter().any(|&(ds, de)| center >= ds && center < de)
        })
    }

    /// Every `HRV_WINDOW_SECS` bucket tumbling from `start` to `end`, each with its clean-beat count and
    /// gap-aware RMSSD (`None` under two clean beats or with no surviving contiguous pair). The per-bucket
    /// breakdown behind [windowed_avg_hrv]; a caller that tags buckets by sleep stage reads this.
    pub fn windowed_buckets(start: u32, end: u32, beats: &[(u32, u16)]) -> Vec<HrvBucket> {
        let seg: Vec<(u32, u16)> = beats.iter().copied().filter(|&(t, _)| t >= start && t <= end).collect();
        if seg.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut t = start as u64;
        while t < end as u64 {
            let hi = t + HRV_WINDOW_SECS;
            // Beats sharing a second came from one report. Group them so a report that re-reports
            // time the previous one already covered breaks contiguity at its first beat.
            let mut reports: Vec<(u32, Vec<u16>)> = Vec::new();
            for &(ts, rr) in seg.iter().filter(|&&(ts, _)| ts as u64 >= t && (ts as u64) < hi) {
                match reports.last_mut() {
                    Some((rt, v)) if *rt == ts => v.push(rr),
                    _ => reports.push((ts, vec![rr])),
                }
            }
            let bucket: Vec<u16> = reports.iter().flat_map(|(_, rr)| rr.iter().copied()).collect();
            let as_slices: Vec<(u32, &[u16])> = reports.iter().map(|(t, rr)| (*t, rr.as_slice())).collect();
            let breaks = report_seam_breaks(&as_slices);
            let (nn, contiguous) = clean_rr_gap_aware_breaking(&bucket, &breaks);
            out.push(HrvBucket {
                start: t as u32,
                clean_beats: nn.len() as u32,
                rmssd: if nn.len() >= 2 { rmssd_from_clean(&nn, &contiguous) } else { None },
            });
            t = hi;
        }
        out
    }

    /// Common bucket-loop body shared by [windowed_avg_hrv] and [windowed_avg_hrv_deep].
    /// The `bucket_filter` predicate gates each bucket's start; it is called once per bucket.
    fn windowed_avg_hrv_inner<F: Fn(u64) -> bool>(
        start: u32, end: u32, beats: &[(u32, u16)], bucket_filter: F,
    ) -> Option<f64> {
        let (mut sum, mut n) = (0.0f64, 0usize);
        for b in Self::windowed_buckets(start, end, beats) {
            if let (true, Some(v)) = (bucket_filter(b.start as u64), b.rmssd) {
                sum += v;
                n += 1;
            }
        }
        (n > 0).then(|| sum / n as f64)
    }

    /// THE nightly HRV value (ms) behind every readiness word: the mean of per-bucket gap-aware RMSSD
    /// over deep-sleep buckets only, from records [`rr_trusted`] accepts. The window and the filter both
    /// live here, so no caller can produce a second version of this one number.
    pub fn nightly_hrv(
        start: u32,
        end: u32,
        reports: &[RrReport],
        deep_spans: &[(u32, u32)],
    ) -> Option<f64> {
        let mut trusted: Vec<&RrReport> =
            reports.iter().filter(|r| rr_trusted(r.optical_signal_poor)).collect();
        trusted.sort_by_key(|r| r.unix);
        let beats: Vec<(u32, u16)> =
            trusted.iter().flat_map(|r| r.rr.iter().map(|&v| (r.unix, v))).collect();
        Self::windowed_avg_hrv_deep(start, end, &beats, deep_spans)
    }

    /// [`Self::nightly_hrv`] behind the coverage gate: trusted beats in `[start, end]` whose [`rr_coverage`]
    /// exceeds [`MAX_RR_COVERAGE`] yield no value and `DuplicatedBeats`.
    pub fn nightly_hrv_checked(
        start: u32,
        end: u32,
        reports: &[RrReport],
        deep_spans: &[(u32, u32)],
    ) -> NightlyHrv {
        let (mut ts, mut ms) = (Vec::new(), Vec::new());
        for r in reports.iter().filter(|r| rr_trusted(r.optical_signal_poor) && r.unix >= start && r.unix <= end) {
            for &v in &r.rr {
                ts.push(r.unix as i64);
                ms.push(v as f64);
            }
        }
        let coverage = covered_rr_coverage(&ts, &ms);
        if coverage > MAX_RR_COVERAGE {
            return NightlyHrv { value: None, quality: HrvQuality::DuplicatedBeats { coverage } };
        }
        NightlyHrv { value: Self::nightly_hrv(start, end, reports, deep_spans), quality: HrvQuality::Ok }
    }

    /// Per-UTC-day series of [`Self::nightly_hrv`] from history records, oldest → newest, for
    /// `evaluate`. A sleep that straddles UTC midnight is split across the two days.
    ///
    /// Emits one entry per day that carries R-R; a day whose beats are all untrusted or all outside
    /// `deep_spans` yields `None`, and `evaluate` drops it along with the real gaps.
    pub fn nightly_rmssd(history: &[HistoryRecord], deep_spans: &[(u32, u32)]) -> Vec<Option<f64>> {
        let mut by_day: std::collections::BTreeMap<u32, Vec<RrReport>> = std::collections::BTreeMap::new();
        for h in history {
            if !h.rr_intervals.is_empty() {
                by_day.entry(h.unix / SECS_PER_DAY).or_default().push(RrReport {
                    unix: h.unix,
                    rr: h.rr_intervals.clone(),
                    optical_signal_poor: h.optical_signal_poor,
                });
            }
        }
        by_day
            .into_iter()
            .map(|(day, reports)| {
                let start = day * SECS_PER_DAY;
                Self::nightly_hrv(start, start + SECS_PER_DAY, &reports, deep_spans)
            })
            .collect()
    }

    /// Readiness over a nightly RMSSD series (ms), oldest → newest. `None` result = calibrating.
    ///
    /// A `None` slot is a missing night, and it is DROPPED, not held: the windows below count
    /// readings, not calendar days, so `baseline7` is the last 7 nights that produced a value and
    /// after a fortnight off-wrist it spans a month. Implausible nights (outside 5..250 ms) leave the
    /// same way. `cv_slope` then fits over index, which treats those unevenly spaced nights as evenly
    /// spaced. Pinned by `evaluate_compresses_gaps_it_does_not_honour_them`.
    ///
    /// Callers cannot correct for this by padding: the padding is what gets removed. Changing it moves
    /// a displayed readiness tier, so it lands as instrumentation beside the incumbent, not as a fix.
    pub fn evaluate(nightly_rmssd_ms: &[Option<f64>]) -> Option<HrvReadinessResult> {
        let valid: Vec<f64> = nightly_rmssd_ms
            .iter()
            .filter_map(|&v| v)
            .filter(|&v| (HRV_MIN_MS..=HRV_MAX_MS).contains(&v))
            .collect();
        if valid.len() < MIN_NIGHTS {
            return None;
        }

        let ell: Vec<f64> = valid.iter().map(|&v| v.max(1.0).ln()).collect();
        let baseline7 = mean(tail(&ell, ROLL_WINDOW));

        let long_win = if valid.len() >= LONG_WINDOW { LONG_WINDOW } else { LONG_WINDOW_FALLBACK };
        let long_ell = tail(&ell, long_win);
        let long_mean = mean(long_ell);
        let long_sd_raw = if long_ell.len() >= 2 { sample_sd(long_ell) } else { sample_sd(tail(&ell, ROLL_WINDOW)) };
        let long_sd = long_sd_raw.max(LONG_SD_FLOOR);

        let swc_half = SWC_K * long_sd;
        let normal_low = long_mean - swc_half;
        let normal_high = long_mean + swc_half;

        let tier = if baseline7 > normal_high {
            ReadinessTier::Primed
        } else if baseline7 >= normal_low {
            ReadinessTier::Normal
        } else {
            ReadinessTier::Suppressed
        };

        let overreaching_watch = cv_slope(&ell) < 0.0 && baseline7 < long_mean;

        Some(HrvReadinessResult {
            tier,
            baseline7_ms: baseline7.exp(),
            normal_low_ms: normal_low.exp(),
            normal_high_ms: normal_high.exp(),
            overreaching_watch,
        })
    }
}

/// Task-Force RMSSD (ms) over a cleaned series, pooling only successive differences whose two beats were
/// adjacent in the source (`contiguous[i]`). `None` when no contiguous pair survives.
fn rmssd_from_clean(nn: &[u16], contiguous: &[bool]) -> Option<f64> {
    let (mut sumsq, mut count) = (0.0f64, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        let d = nn[i] as f64 - nn[i - 1] as f64;
        sumsq += d * d;
        count += 1;
    }
    (count > 0).then(|| (sumsq / count as f64).sqrt())
}

/// Gap-aware pNN50 (% of contiguous successive |ΔNN| > 50 ms). `None` when no contiguous pair survives.
fn pnn50_from_clean(nn: &[u16], contiguous: &[bool]) -> Option<f64> {
    let (mut nn50, mut pairs) = (0usize, 0usize);
    for i in 1..nn.len() {
        if !contiguous[i] {
            continue;
        }
        if (nn[i] as f64 - nn[i - 1] as f64).abs() > PNN50_THRESHOLD_MS {
            nn50 += 1;
        }
        pairs += 1;
    }
    (pairs > 0).then(|| nn50 as f64 / pairs as f64 * 100.0)
}

/// The last `n` elements, or all if fewer.
fn tail(xs: &[f64], n: usize) -> &[f64] {
    &xs[xs.len().saturating_sub(n)..]
}

/// Range-filter then Malik ectopic-reject an ordered R-R series (ms), returning the clean beats and a
/// contiguity mask: `contiguous[i]` is true only when beats `i` and `i-1` were adjacent in the input
/// (no beat dropped between them). A splice from a dropped beat is where a successive-difference metric
/// must skip. Index 0 is always false.
fn clean_rr_gap_aware(rr: &[u16]) -> (Vec<u16>, Vec<bool>) {
    clean_rr_gap_aware_breaking(rr, &[])
}

/// As [`clean_rr_gap_aware`], with `breaks[i]` marking a beat that does not follow beat `i-1` in time.
/// The strap re-reports part of its previous window each second, so a report's first beat is not the
/// successor of the last beat of the report before it. An empty mask means no breaks.
fn clean_rr_gap_aware_breaking(rr: &[u16], breaks: &[bool]) -> (Vec<u16>, Vec<bool>) {
    let (kept_orig, kept_val) = clean_rr_kept(rr);
    let contiguous: Vec<bool> = (0..kept_val.len())
        .map(|i| {
            i > 0
                && kept_orig[i] == kept_orig[i - 1] + 1
                && !breaks.get(kept_orig[i]).copied().unwrap_or(false)
        })
        .collect();
    (kept_val, contiguous)
}

/// Range filter then Malik ectopic rejection, returning each survivor's index in the INPUT alongside its
/// value. The one cleaning pass; [`clean_rr_gap_aware_breaking`] turns the indices into a contiguity mask
/// and [`rolling_rmssd`] uses them to keep each survivor paired with its timestamp.
fn clean_rr_kept(rr: &[u16]) -> (Vec<usize>, Vec<u16>) {
    let mut ranged_idx: Vec<usize> = Vec::new();
    let mut ranged_val: Vec<u16> = Vec::new();
    for (i, &v) in rr.iter().enumerate() {
        if (RR_MIN_MS..=RR_MAX_MS).contains(&v) {
            ranged_idx.push(i);
            ranged_val.push(v);
        }
    }
    if ranged_val.len() <= ECTOPIC_WINDOW_RADIUS {
        return (ranged_idx, ranged_val);
    }
    let (mut kept_orig, mut kept_val): (Vec<usize>, Vec<u16>) = (Vec::new(), Vec::new());
    for i in 0..ranged_val.len() {
        let lo = i.saturating_sub(ECTOPIC_WINDOW_RADIUS);
        let hi = (i + ECTOPIC_WINDOW_RADIUS).min(ranged_val.len() - 1);
        let mut neighbours: Vec<f64> = Vec::with_capacity(hi - lo);
        for (j, &v) in ranged_val.iter().enumerate().take(hi + 1).skip(lo) {
            if j != i {
                neighbours.push(v as f64);
            }
        }
        let keep = if neighbours.len() < 2 {
            true
        } else {
            let med = median(&neighbours);
            med <= 0.0 || (ranged_val[i] as f64 - med).abs() / med <= ECTOPIC_THRESHOLD
        };
        if keep {
            kept_orig.push(ranged_idx[i]);
            kept_val.push(ranged_val[i]);
        }
    }
    (kept_orig, kept_val)
}

/// Rolling trailing-window RMSSD (ms) over a timestamped R-R series, the trace a within-night chart
/// plots. Cleans the WHOLE series once so an artifact never enters a window, then for each surviving beat
/// emits `(ts, rmssd_plain)` over the beats in `(ts - window_s, ts]` that hold >= `min_beats` survivors.
/// `step_s > 0` thins to one point per that many seconds of advance, measured against the last EMITTED
/// point. Empty below `min_beats` inputs or a non-positive window. Input need not be sorted.
pub fn rolling_rmssd(beats: &[(i64, u16)], window_s: i64, step_s: i64, min_beats: usize) -> Vec<(i64, f64)> {
    if beats.len() < min_beats || window_s <= 0 {
        return Vec::new();
    }
    let mut sorted: Vec<(i64, u16)> = beats.to_vec();
    sorted.sort_by_key(|&(t, _)| t);
    let vals: Vec<u16> = sorted.iter().map(|&(_, v)| v).collect();
    let (kept_idx, kept_val) = clean_rr_kept(&vals);
    if kept_val.len() < 2 {
        return Vec::new();
    }
    let kept: Vec<(i64, u16)> = kept_idx.iter().map(|&i| (sorted[i].0, vals[i])).collect();
    let mut out: Vec<(i64, f64)> = Vec::with_capacity(kept.len());
    let mut lo = 0usize;
    let mut last_emit: Option<i64> = None;
    for hi in 0..kept.len() {
        let t_end = kept[hi].0;
        let t_start = t_end - window_s;
        while lo < hi && kept[lo].0 <= t_start {
            lo += 1;
        }
        if let Some(last) = last_emit {
            if step_s > 0 && t_end - last < step_s {
                continue;
            }
        }
        let span: Vec<u16> = kept[lo..=hi].iter().map(|&(_, v)| v).collect();
        if span.len() < min_beats {
            continue;
        }
        if let Some(r) = HrvReadiness::rmssd_plain(&span) {
            out.push((t_end, r));
            last_emit = Some(t_end);
        }
    }
    out
}

/// Total beat-time (sum of R-R, ms) over the wall-clock span of the same beats. Above ~1.0 is physically
/// impossible: more beat-time than elapsed time means beats are double-counted or reports overlap.
/// `0.0` for fewer than two timestamps, no intervals, or a non-positive span.
pub fn rr_coverage(ts_sec: &[i64], rr_ms: &[f64]) -> f64 {
    if ts_sec.len() < 2 || rr_ms.is_empty() {
        return 0.0;
    }
    let (lo, hi) = (ts_sec.iter().min(), ts_sec.iter().max());
    let (Some(&lo), Some(&hi)) = (lo, hi) else { return 0.0 };
    let span_ms = (hi - lo) as f64 * 1000.0;
    if span_ms <= 0.0 {
        return 0.0;
    }
    rr_ms.iter().sum::<f64>() / span_ms
}

/// Like [`rr_coverage`] but over covered time: each gap between successive distinct stamps counts at most
/// [`COVER_GAP_SECS`], so wear gaps inside a night do not dilute a doubled stream. `0.0` for no beats.
pub fn covered_rr_coverage(ts_sec: &[i64], rr_ms: &[f64]) -> f64 {
    let mut t: Vec<i64> = ts_sec.to_vec();
    t.sort_unstable();
    t.dedup();
    if t.len() < 2 || rr_ms.is_empty() {
        return 0.0;
    }
    let covered: i64 = 1 + t.windows(2).map(|w| (w[1] - w[0]).min(COVER_GAP_SECS)).sum::<i64>();
    rr_ms.iter().sum::<f64>() / (covered as f64 * 1000.0)
}

/// Rows that repeat an earlier `(ts, rr_ms)` exactly: `total - distinct(ts, rr_ms)`. Measures byte-identical
/// re-inserts only. It cannot see the overlap that [`overlapping_report_count`] counts, where two reports
/// cover the same seconds with DIFFERENT values, so a low count here does not mean [`rr_coverage`] is sound.
pub fn duplicate_beat_count(ts_sec: &[i64], rr_ms: &[f64]) -> u32 {
    let n = ts_sec.len().min(rr_ms.len());
    let mut seen: std::collections::HashSet<(i64, u64)> = std::collections::HashSet::new();
    let mut dups = 0u32;
    for i in 0..n {
        if !seen.insert((ts_sec[i], rr_ms[i].to_bits())) {
            dups += 1;
        }
    }
    dups
}

/// `(overlapping, total)` per-second reports, where an overlapping report's first beat re-reports time
/// earlier reports already covered. This is the mechanism behind an [`rr_coverage`] above 1.0; the same
/// test drives the contiguity break in [`HrvReadiness::rmssd_gap_aware`].
pub fn overlapping_report_count(reports: &[(u32, Vec<u16>)]) -> (u32, u32) {
    let as_slices: Vec<(u32, &[u16])> = reports.iter().map(|(t, rr)| (*t, rr.as_slice())).collect();
    let breaks = report_seam_breaks(&as_slices);
    let mut flat = 0usize;
    let mut overlapping = 0u32;
    for (_, rr) in reports {
        if !rr.is_empty() && breaks.get(flat).copied().unwrap_or(false) {
            overlapping += 1;
        }
        flat += rr.len();
    }
    (overlapping, reports.len() as u32)
}

/// Marks each report's first beat that re-reports time already accounted for. The strap emits the last
/// few beats every second, so beat-time runs ahead of the clock; a report starting past the wall time is
/// repeating beats and its first difference is not a real successive pair. Whole-second timestamps make
/// a per-report test too noisy, so the comparison is cumulative.
fn report_seam_breaks(reports: &[(u32, &[u16])]) -> Vec<bool> {
    // Report times are whole seconds, so beat-time sits up to a second ahead of the clock without any
    // re-reporting. Only a lead beyond that is the strap repeating itself.
    const SEAM_SLACK_MS: u64 = 2_000;
    let mut out = Vec::new();
    let Some(&(t0, _)) = reports.first() else { return out };
    let mut covered_ms: u64 = 0;
    for &(t, rr) in reports {
        let elapsed_ms = u64::from(t.saturating_sub(t0)) * 1000;
        for (j, _) in rr.iter().enumerate() {
            out.push(j == 0 && covered_ms > elapsed_ms + SEAM_SLACK_MS);
        }
        covered_ms += rr.iter().map(|&v| u64::from(v)).sum::<u64>();
    }
    out
}

/// OLS slope of the rolling 7-night coefficient-of-variation series over the trailing `CV_TREND_WINDOW`.
fn cv_slope(ell: &[f64]) -> f64 {
    let start = (ROLL_WINDOW - 1).max(ell.len().saturating_sub(CV_TREND_WINDOW));
    let mut cv = Vec::new();
    for i in start..ell.len() {
        let w = &ell[i + 1 - ROLL_WINDOW..=i];
        let m = mean(w);
        cv.push(if m != 0.0 { 100.0 * sample_sd(w) / m } else { 0.0 });
    }
    least_squares_slope(&cv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmssd_of_known_intervals() {
        // successive diffs 10, 10 → mean square 100 → sqrt 10.
        assert_eq!(HrvReadiness::rmssd(&[800, 810, 820]), Some(10.0));
        assert_eq!(HrvReadiness::rmssd(&[800]), None);
    }

    #[test]
    fn rmssd_runs_pools_within_runs_only() {
        // Two runs; the break between them (810 → 100) must never be differenced.
        let a: &[u16] = &[800, 810];
        let b: &[u16] = &[100, 110];
        assert!((HrvReadiness::rmssd_runs([a, b]).unwrap() - 10.0).abs() < 1e-9);
        // No run with two beats → None.
        assert_eq!(HrvReadiness::rmssd_runs([[500u16].as_slice(), [600].as_slice()]), None);
        assert_eq!(HrvReadiness::rmssd_runs(std::iter::empty::<&[u16]>()), None);
    }

    #[test]
    fn rmssd_runs_excludes_artifact_beat_jumps() {
        // A 900 ms jump in the middle (an ectopic/missed beat) must not inflate RMSSD; only the clean
        // 10 ms diffs survive → ~10 ms, not the ~600 ms a raw pooling would give.
        let run: &[u16] = &[600, 610, 1500, 610, 600];
        let v = HrvReadiness::rmssd_runs([run]).unwrap();
        assert!(v < 20.0, "artifact beat-to-beat jumps should be excluded, got {v}");
    }

    /// The gap contract, pinned so it cannot be assumed away: `evaluate` counts READINGS, not calendar
    /// days. A caller that pads its missing nights with `None` gets exactly the same answer as one that
    /// omits them, because the padding is stripped before any window is taken. Both differ from the
    /// unbroken run only in which nights the last 7 actually are.
    #[test]
    fn evaluate_compresses_gaps_it_does_not_honour_them() {
        let nights: Vec<Option<f64>> = (0..14).map(|i| Some(50.0 + i as f64)).collect();

        // Same 14 readings, but the middle four nights were never worn.
        let mut padded = nights.clone();
        for slot in padded.iter_mut().skip(5).take(4) {
            *slot = None;
        }
        let omitted: Vec<Option<f64>> = padded.iter().copied().flatten().map(Some).collect();
        assert_eq!(padded.len(), 14);
        assert_eq!(omitted.len(), 10);

        let a = HrvReadiness::evaluate(&padded).expect("10 valid nights clears MIN_NIGHTS");
        let b = HrvReadiness::evaluate(&omitted).expect("same 10 readings");
        assert_eq!(a.baseline7_ms, b.baseline7_ms, "a None slot carries no information into evaluate");
        assert_eq!(a.normal_low_ms, b.normal_low_ms);
        assert_eq!(a.normal_high_ms, b.normal_high_ms);
        assert_eq!(a.overreaching_watch, b.overreaching_watch);

        // And the 7-night baseline really is the last 7 SURVIVING readings: dropping four mid-series
        // nights pulls older values into the window, so it does not match the unbroken fortnight.
        let unbroken = HrvReadiness::evaluate(&nights).expect("14 valid nights");
        assert_ne!(
            unbroken.baseline7_ms, a.baseline7_ms,
            "if these matched, the window would be counting days rather than readings"
        );
    }

    /// A deep span over every fixture second, so a filter fixture is not silently also a window fixture.
    fn all_deep() -> Vec<(u32, u32)> {
        vec![(0, 3 * SECS_PER_DAY)]
    }

    #[test]
    fn nightly_rmssd_produces_one_gap_aware_value_per_day() {
        let rec = |unix: u32, rr: Vec<u16>| HistoryRecord { version: 18, unix, rr_intervals: rr, ..Default::default() };
        // day 0: two contiguous records; day 1: one record with two beats.
        let hist = vec![rec(0, vec![600, 610]), rec(1, vec![605, 615]), rec(SECS_PER_DAY, vec![700, 720])];
        let series = HrvReadiness::nightly_rmssd(&hist, &all_deep());
        assert_eq!(series.len(), 2);
        assert!(series.iter().all(|v| v.is_some()));
    }

    /// The window half of the one nightly quantity: beats outside every deep span do not reach it. A
    /// producer that pooled the whole day would return the same number with the span moved away.
    #[test]
    fn nightly_hrv_reads_only_the_deep_window() {
        let rep = |unix: u32, rr: Vec<u16>| RrReport { unix, rr, optical_signal_poor: None };
        // Two 300 s buckets, each one report: 800/810 in the first, 700/900 in the second.
        let reports = vec![rep(10, vec![800, 810]), rep(310, vec![700, 900])];
        let first = HrvReadiness::nightly_hrv(0, 600, &reports, &[(0, 300)]).expect("bucket 1 is deep");
        let second = HrvReadiness::nightly_hrv(0, 600, &reports, &[(300, 600)]).expect("bucket 2 is deep");
        let both = HrvReadiness::nightly_hrv(0, 600, &reports, &[(0, 600)]).expect("both buckets are deep");
        assert!((first - 10.0).abs() < 1e-9, "first {first}");
        assert!((second - 200.0).abs() < 1e-9, "second {second}");
        assert!((both - 105.0).abs() < 1e-9, "the mean of the two buckets, {both}");
        assert_eq!(HrvReadiness::nightly_hrv(0, 600, &reports, &[]), None, "no deep window, no value");
    }

    /// The filter half, on the same fixture: a report the strap flagged contributes nothing, whichever
    /// window it sits in. Both halves must hold or the two producers of this number can disagree again.
    #[test]
    fn nightly_hrv_drops_a_flagged_report_inside_the_deep_window() {
        let rep = |unix: u32, rr: Vec<u16>, poor: bool| RrReport {
            unix, rr, optical_signal_poor: Some(poor),
        };
        let clean = vec![rep(10, vec![800, 810], false), rep(310, vec![700, 900], false)];
        let flagged = vec![rep(10, vec![800, 810], false), rep(310, vec![700, 900], true)];
        let deep = [(0u32, 600u32)];
        assert!((HrvReadiness::nightly_hrv(0, 600, &clean, &deep).unwrap() - 105.0).abs() < 1e-9);
        assert!((HrvReadiness::nightly_hrv(0, 600, &flagged, &deep).unwrap() - 10.0).abs() < 1e-9);
    }

    fn doubled_night(copies: usize) -> Vec<RrReport> {
        // One report per second, 1000 ms beats alternating 950/1050 so RMSSD is nonzero; `copies` stores each.
        (0..600u32)
            .flat_map(|t| (0..copies).map(move |_| RrReport {
                unix: t, rr: vec![if t % 2 == 0 { 950 } else { 1050 }], optical_signal_poor: None,
            }))
            .collect()
    }

    #[test]
    fn nightly_hrv_checked_passes_a_healthy_night() {
        let r = HrvReadiness::nightly_hrv_checked(0, 600, &doubled_night(1), &[(0, 600)]);
        assert_eq!(r.quality, HrvQuality::Ok);
        assert!(r.value.is_some());
    }

    #[test]
    fn nightly_hrv_checked_refuses_every_beat_stored_twice() {
        let r = HrvReadiness::nightly_hrv_checked(0, 600, &doubled_night(2), &[(0, 600)]);
        assert_eq!(r.value, None);
        assert!(matches!(r.quality, HrvQuality::DuplicatedBeats { coverage } if coverage > 1.9));
    }

    #[test]
    fn nightly_hrv_checked_boundary_is_strictly_above_the_limit() {
        // Two beats 5 s apart cover 6 s: coverage = (a + b) / 6000.
        let q = |a: u16, b: u16| {
            let rep = |unix: u32, v: u16| RrReport { unix, rr: vec![v], optical_signal_poor: None };
            HrvReadiness::nightly_hrv_checked(0, 600, &[rep(0, a), rep(5, b)], &[(0, 600)]).quality
        };
        assert_eq!(q(3900, 3900), HrvQuality::Ok, "exactly 1.3 passes");
        assert!(matches!(q(3901, 3900), HrvQuality::DuplicatedBeats { .. }));
    }

    #[test]
    fn covered_rr_coverage_ignores_a_long_gap() {
        // Two stored copies of a 1 s beat at 0..9 s and again at 3600..3609 s: a span-based ratio
        // dilutes to ~0.01, covered time keeps it above 1.
        let ts: Vec<i64> = (0..10).chain(3600..3610).flat_map(|t| [t, t]).collect();
        let ms = vec![1000.0; ts.len()];
        assert!(rr_coverage(&ts, &ms) < 0.1);
        let c = covered_rr_coverage(&ts, &ms);
        assert!((c - 40.0 / 24.0).abs() < 1e-9, "{c}"); // 40 s of beats over 1 + 9 + 5 + 9 covered s
        assert_eq!(covered_rr_coverage(&[5], &[1000.0]), 0.0);
    }

    #[test]
    fn rr_trusted_reads_only_the_raised_flag() {
        assert!(!rr_trusted(Some(true)));
        assert!(rr_trusted(Some(false)));
        assert!(rr_trusted(None)); // a layout without the flag stays trusted
    }

    #[test]
    fn nightly_rmssd_drops_the_beats_the_strap_could_not_read() {
        let rec = |unix: u32, rr: Vec<u16>, poor: bool| HistoryRecord {
            version: 18,
            unix,
            rr_intervals: rr,
            optical_signal_poor: Some(poor),
            ..Default::default()
        };
        // A steady day plus one record whose beats swing far wider, in range and past Malik ectopic.
        let steady: Vec<HistoryRecord> = (0u32..20).map(|t| rec(t, vec![800, 810], false)).collect();
        let mut noisy = steady.clone();
        noisy.push(rec(20, vec![850, 760], true));
        let deep = all_deep();
        assert_eq!(
            HrvReadiness::nightly_rmssd(&noisy, &deep),
            HrvReadiness::nightly_rmssd(&steady, &deep)
        );
        // The same record unflagged does reach the day, so the gate is what excluded it.
        let mut unflagged = steady.clone();
        unflagged.push(rec(20, vec![850, 760], false));
        assert_ne!(
            HrvReadiness::nightly_rmssd(&unflagged, &deep),
            HrvReadiness::nightly_rmssd(&steady, &deep)
        );
        // A day of nothing but flagged records yields no value rather than one built from them.
        let all_poor: Vec<HistoryRecord> = (0u32..20).map(|t| rec(t, vec![800, 810], true)).collect();
        assert_eq!(HrvReadiness::nightly_rmssd(&all_poor, &deep), vec![None]);
    }

    /// A stream shaped like the strap's: one report a second, each carrying more beat-time than a second,
    /// so beat-time runs ahead of the clock and the later reports are re-reporting.
    fn overlapping_stream() -> Vec<(u32, Vec<u16>)> {
        (1u32..=12).map(|t| (t, vec![830u16, 830])).collect()
    }

    #[test]
    fn rmssd_gap_aware_skips_the_seam_once_beat_time_runs_ahead() {
        // 3.56 s of beats a second: the lead clears the slack immediately, so every seam is dropped and
        // only the 60 ms in-report steps count. Flattened, the -180 ms seams would dominate.
        let reports: Vec<(u32, Vec<u16>)> =
            (1u32..=12).map(|t| (t, vec![800u16, 860, 920, 980])).collect();
        let got = HrvReadiness::rmssd_gap_aware(&reports).unwrap();
        assert!((got - 60.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn rmssd_gap_aware_keeps_seams_when_beat_time_tracks_the_clock() {
        // One beat a second at ~1000 ms: nothing is re-reported, so no seam may be dropped and every
        // successive pair counts.
        let reports: Vec<(u32, Vec<u16>)> =
            (1u32..=12).map(|t| (t, vec![if t % 2 == 0 { 1020u16 } else { 980 }])).collect();
        let got = HrvReadiness::rmssd_gap_aware(&reports).unwrap();
        assert!((got - 40.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn report_seam_breaks_only_after_the_lead_exceeds_the_slack() {
        let stream = overlapping_stream();
        let refs: Vec<(u32, &[u16])> = stream.iter().map(|(t, v)| (*t, v.as_slice())).collect();
        let breaks = report_seam_breaks(&refs);
        // 1660 ms a second against 1000: the lead passes the 2 s slack partway in, never before.
        assert!(!breaks[0], "the first report can never be a re-report");
        assert!(breaks.iter().any(|&b| b), "a sustained overrun must eventually break");
        let first = breaks.iter().position(|&b| b).unwrap();
        assert!(first >= 6, "broke too early at index {first}");
    }

    #[test]
    fn windowed_avg_hrv_skips_the_seam_within_a_bucket() {
        // Same overrunning shape, inside one 5-min bucket.
        let mut beats: Vec<(u32, u16)> = Vec::new();
        for t in 10u32..=40 {
            for v in [800u16, 860, 920, 980] {
                beats.push((t, v));
            }
        }
        let got = HrvReadiness::windowed_avg_hrv(10, 41, &beats).unwrap();
        assert!((got - 60.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn clean_rr_gap_aware_drops_ectopic_and_splices() {
        // A Malik-ectopic spike (1300) is dropped; its removal splices 810→806 apart so that difference is
        // not contiguous. Values hand-verified.
        let (nn, contig) = clean_rr_gap_aware(&[800, 805, 810, 1300, 806, 802, 808]);
        assert_eq!(nn, vec![800, 805, 810, 806, 802, 808]);
        assert_eq!(contig, vec![false, true, true, false, true, true]);
    }

    fn alternating(n: usize, lo: u16, hi: u16) -> Vec<u16> {
        (0..n).map(|i| if i % 2 == 0 { lo } else { hi }).collect()
    }

    #[test]
    fn analyze_raw_full_clean_series() {
        let a = HrvReadiness::analyze_raw(&alternating(24, 800, 810), None);
        assert_eq!((a.n_input, a.n_clean), (24, 24));
        assert!((a.rmssd.unwrap() - 10.0).abs() < 1e-9); // every successive Δ = 10
        assert_eq!(a.mean_nn, Some(805.0));
        assert_eq!(a.pnn50, Some(0.0)); // all |Δ| = 10 < 50
        assert!((a.sdnn.unwrap() - (600.0f64 / 23.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn analyze_raw_too_few_clean_is_empty() {
        let a = HrvReadiness::analyze_raw(&[800, 810, 820], None);
        assert_eq!(
            a,
            HrvAnalysis { rmssd: None, sdnn: None, mean_nn: None, pnn50: None, n_input: 3, n_clean: 0 }
        );
    }

    #[test]
    fn analyze_raw_spot_rejected_fraction_gate() {
        // 25 clean beats + 15 out-of-range (100 ms) = 40 input; cleaning keeps 25 (0.375 rejected).
        let mut s = alternating(25, 800, 810);
        s.extend(std::iter::repeat_n(100u16, 15));
        let open = HrvReadiness::analyze_raw(&s, None);
        assert_eq!((open.n_input, open.n_clean), (40, 25));
        assert!(open.rmssd.is_some());
        // Spot gate at 0.35: 0.375 > 0.35 → refused (empty, n_clean 0).
        let gated = HrvReadiness::analyze_raw(&s, Some(0.35));
        assert_eq!(gated.n_clean, 0);
        assert!(gated.rmssd.is_none());
        // A looser gate passes.
        assert!(HrvReadiness::analyze_raw(&s, Some(0.40)).rmssd.is_some());
    }

    #[test]
    fn analyze_raw_pnn50_counts_big_jumps() {
        // 20 beats alternating 700/800 → every contiguous |Δ| = 100 > 50 → pNN50 = 100 %.
        let a = HrvReadiness::analyze_raw(&alternating(20, 700, 800), None);
        assert_eq!(a.pnn50, Some(100.0));
    }

    #[test]
    fn sdnn_does_not_overflow_on_a_long_series() {
        // 300 × ~800 ms sums to ~240k, far past u16::MAX; a u16 accumulator would have wrapped/panicked.
        // Alternating 800/810 → sample SD = sqrt(7500 / 299) ≈ 5.0084.
        let s = HrvReadiness::analyze_raw(&alternating(300, 800, 810), None).sdnn.unwrap();
        assert!((s - (7500.0f64 / 299.0).sqrt()).abs() < 1e-9, "got {s}");
        assert_eq!(HrvReadiness::sdnn(&vec![900u16; 400]), Some(0.0)); // flat long series → zero spread
    }

    /// Three hand-computed splices, NOT a cross-language comparison: nothing outside this file holds
    /// these literals. The input never reaches RR_MIN_MS/RR_MAX_MS and clears ECTOPIC_THRESHOLD by 3x,
    /// so those three constants are invisible here — `tests/hrv_real_rr.rs` is what pins them.
    #[test]
    fn rmssd_gap_aware_splices_at_each_dropped_beat() {
        // Malik-ectopic splice: the dropped 1300 removes the 810→806 difference. sqrt(102/4).
        let malik = vec![(0u32, vec![800u16, 805, 810, 1300, 806, 802, 808])];
        assert!((HrvReadiness::rmssd_gap_aware(&malik).unwrap() - 5.049752469181039).abs() < 1e-12);
        // Out-of-range 5 ms beat is dropped and splices 605→620 apart. sqrt(100/2).
        let range = vec![(0u32, vec![600u16, 610, 605, 5, 620, 615])];
        assert!((HrvReadiness::rmssd_gap_aware(&range).unwrap() - 7.0710678118654755).abs() < 1e-12);
        // A clean series has no splices, so it equals the plain Task-Force RMSSD (÷ n-1). sqrt(325/4).
        let clean = vec![(0u32, vec![800u16, 810, 820, 815, 805])];
        assert!((HrvReadiness::rmssd_gap_aware(&clean).unwrap() - 9.013878188659973).abs() < 1e-12);
    }

    #[test]
    fn windowed_avg_hrv_means_per_bucket_rmssd() {
        // Bucket A [100,400): clean [800,810,820,815,805] → sqrt(325/4). Bucket B [400,700): [700,720,710]
        // → sqrt(500/2). avgHrv = mean of the two bucket RMSSDs.
        let a = 9.013878188659973f64;
        let b = 15.811388300841896f64;
        let beats: Vec<(u32, u16)> = vec![
            (100, 800), (100, 810), (100, 820), (100, 815), (100, 805),
            (400, 700), (400, 720), (400, 710),
        ];
        let got = HrvReadiness::windowed_avg_hrv(100, 700, &beats).unwrap();
        assert!((got - (a + b) / 2.0).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn windowed_avg_hrv_drops_buckets_without_a_contiguous_pair() {
        // A single in-range beat per bucket yields no successive difference → no bucket contributes → None.
        let beats: Vec<(u32, u16)> = vec![(100, 800), (400, 810)];
        assert!(HrvReadiness::windowed_avg_hrv(100, 700, &beats).is_none());
        // Beats outside [start, end] are filtered out before bucketing.
        let outside: Vec<(u32, u16)> = vec![(50, 800), (50, 810)];
        assert!(HrvReadiness::windowed_avg_hrv(100, 700, &outside).is_none());
    }

    #[test]
    fn windowed_avg_hrv_single_bucket_equals_bucket_rmssd() {
        // One bucket, one clean series → avgHrv equals that bucket's gap-aware RMSSD.
        let beats: Vec<(u32, u16)> = vec![(100, 800), (100, 810), (100, 820), (100, 815), (100, 805)];
        let got = HrvReadiness::windowed_avg_hrv(100, 400, &beats).unwrap();
        assert!((got - 9.013878188659973).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn deep_windowed_avg_hrv_keeps_buckets_in_deep_spans() {
        // Two buckets: [100,400) and [400,700). Only [400,700) is deep → result = bucket B RMSSD.
        let beats: Vec<(u32, u16)> = vec![
            (100, 800), (100, 810), (100, 820), (100, 815), (100, 805),
            (400, 700), (400, 720), (400, 710),
        ];
        let deep = vec![(400u32, 700u32)];
        let got = HrvReadiness::windowed_avg_hrv_deep(100, 700, &beats, &deep).unwrap();
        let b = 15.811388300841896f64; // sqrt(500/2) from [700,720,710]
        assert!((got - b).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn deep_windowed_avg_hrv_returns_none_without_deep_spans() {
        let beats: Vec<(u32, u16)> = vec![(100, 800), (100, 810)];
        let got = HrvReadiness::windowed_avg_hrv_deep(100, 400, &beats, &[]);
        assert!(got.is_none());
    }

    #[test]
    fn pnn50_plain_counts_every_pair_over_the_threshold() {
        // Alternating 700/800: every |d| = 100 > 50, so 100 %. A 10 ms alternation clears none.
        assert_eq!(HrvReadiness::pnn50_plain(&alternating(20, 700, 800)), Some(100.0));
        assert_eq!(HrvReadiness::pnn50_plain(&alternating(20, 800, 810)), Some(0.0));
        // Half the pairs over the threshold.
        let two_of_three = HrvReadiness::pnn50_plain(&[800, 900, 890, 990]).unwrap();
        assert!((two_of_three - 200.0 / 3.0).abs() < 1e-9, "got {two_of_three}");
        assert_eq!(HrvReadiness::pnn50_plain(&[800]), None);
    }

    #[test]
    fn rolling_rmssd_tracks_the_alternation_not_the_beat_period() {
        // ±10 ms alternation on a ~800 ms period: the curve settles near 10, nowhere near 800.
        let series: Vec<(i64, u16)> =
            (0..30).map(|i| (i, if i % 2 == 0 { 800u16 } else { 810 })).collect();
        let out = rolling_rmssd(&series, 60, 0, 8);
        assert!(!out.is_empty());
        assert!(out.iter().all(|&(_, v)| v < 100.0));
        assert!((out.last().unwrap().1 - 10.0).abs() < 3.0, "got {:?}", out.last());
    }

    #[test]
    fn rolling_rmssd_emits_from_the_first_full_window_and_thins_on_a_stride() {
        // One point per beat once the trailing window holds min_beats survivors: ts 100..109 → 107..109.
        let steady: Vec<(i64, u16)> = (0..10).map(|i| (100 + i, 800u16)).collect();
        let out = rolling_rmssd(&steady, 300, 0, 8);
        assert_eq!((out.first().unwrap().0, out.last().unwrap().0), (107, 109));
        // A stride emits fewer points, each at least step_s after the previous EMITTED one.
        let long: Vec<(i64, u16)> =
            (0..60).map(|i| (1000 + i, if i % 2 == 0 { 800u16 } else { 810 })).collect();
        let dense = rolling_rmssd(&long, 30, 0, 8);
        let thinned = rolling_rmssd(&long, 30, 10, 8);
        assert!(thinned.len() < dense.len());
        assert!(thinned.windows(2).all(|w| w[1].0 - w[0].0 >= 10));
    }

    #[test]
    fn rolling_rmssd_cleans_artifacts_and_refuses_a_degenerate_call() {
        // A 50 ms beat is out of range: it must never reach a window, so a steady series stays flat.
        let mut series: Vec<(i64, u16)> = (0..20).map(|i| (i, 800u16)).collect();
        series.push((100, 50));
        assert!(rolling_rmssd(&series, 300, 0, 8).iter().all(|&(_, v)| v < 5.0));
        // Two clusters 1000 s apart: a 60 s window never spans both, so no cross-cluster jump appears.
        let mut split: Vec<(i64, u16)> = (0..25).map(|i| (i, 800u16)).collect();
        split.extend((0..25).map(|i| (1000 + i, 820u16)));
        let out = rolling_rmssd(&split, 60, 0, 8);
        assert!(!out.is_empty() && out.iter().all(|&(_, v)| v < 30.0));
        // Too few beats, and a non-positive window, both yield nothing rather than a fabricated point.
        assert!(rolling_rmssd(&[], 300, 0, 8).is_empty());
        assert!(rolling_rmssd(&[(0, 800)], 300, 0, 8).is_empty());
        assert!(rolling_rmssd(&steady_ten(), 0, 0, 8).is_empty());
    }

    fn steady_ten() -> Vec<(i64, u16)> {
        (0..10).map(|i| (i, 800u16)).collect()
    }

    #[test]
    fn rr_coverage_is_beat_time_over_elapsed_time() {
        // 5 × 1000 ms of beat-time across a 4 s span → 1.25.
        let ts: Vec<i64> = vec![100, 101, 102, 103, 104];
        assert!((rr_coverage(&ts, &[1000.0; 5]) - 1.25).abs() < 1e-9);
        // Every beat stored twice on the same second: 6000 ms over 2 s → 3.0, the impossible-value signal.
        let dup: Vec<i64> = vec![100, 100, 101, 101, 102, 102];
        assert!((rr_coverage(&dup, &[1000.0; 6]) - 3.0).abs() < 1e-9);
        assert_eq!(rr_coverage(&[], &[]), 0.0);
        assert_eq!(rr_coverage(&[100], &[1000.0]), 0.0);
        assert_eq!(rr_coverage(&[100, 100], &[1000.0, 1000.0]), 0.0); // zero span
    }

    #[test]
    fn duplicate_beat_count_sees_exact_repeats_only() {
        assert_eq!(duplicate_beat_count(&[100, 101, 102], &[1000.0, 1010.0, 1020.0]), 0);
        assert_eq!(duplicate_beat_count(&[100, 100, 101], &[1000.0, 1000.0, 1010.0]), 1);
        assert_eq!(duplicate_beat_count(&[100, 100, 100], &[1000.0; 3]), 2);
        // Same second, different value: distinct beats, so the exact-repeat counter sees nothing --
        // which is exactly the overlap it cannot detect.
        assert_eq!(duplicate_beat_count(&[100, 100], &[1000.0, 1010.0]), 0);
    }

    #[test]
    fn overlapping_report_count_sees_the_overlap_duplicates_miss() {
        // 3.56 s of beat-time a second: reports re-report, and every value is distinct.
        let over: Vec<(u32, Vec<u16>)> = (1u32..=12).map(|t| (t, vec![800u16, 860, 920, 980])).collect();
        let (overlapping, total) = overlapping_report_count(&over);
        assert_eq!(total, 12);
        assert!(overlapping > 0, "a sustained overrun must be counted");
        let ts: Vec<i64> = over.iter().flat_map(|(t, rr)| rr.iter().map(|_| *t as i64)).collect();
        let ms: Vec<f64> = over.iter().flat_map(|(_, rr)| rr.iter().map(|&v| v as f64)).collect();
        assert_eq!(duplicate_beat_count(&ts, &ms), 0, "no exact repeats, yet coverage is over 1");
        assert!(rr_coverage(&ts, &ms) > 1.0);
        // One beat a second at ~1000 ms: nothing is re-reported, so nothing is flagged.
        let tracking: Vec<(u32, Vec<u16>)> = (1u32..=12).map(|t| (t, vec![1000u16])).collect();
        assert_eq!(overlapping_report_count(&tracking), (0, 12));
    }

    /// A named do-nothing scorer over input `I` returning `O`: the null arm every gate below rejects.
    type Scorer<'a, I, O> = (&'a str, &'a dyn Fn(I) -> O);

    /// The unlock is the recovery schedule's, counted in VALID nights: `None` slots and values outside
    /// the plausibility band are not nights. The previous gate asserted only the closed side of the edge.
    #[test]
    fn readiness_unlocks_on_the_recovery_schedule_and_counts_only_valid_nights() {
        let sched = crate::calibration::RECOVERY_SCORE;
        assert_eq!(MIN_NIGHTS, sched.unlock as usize, "the unlock must stay the schedule's, not a copy");
        assert!(HrvReadiness::evaluate(&[Some(50.0); MIN_NIGHTS - 1]).is_none());
        assert!(HrvReadiness::evaluate(&[Some(50.0); MIN_NIGHTS]).is_some(), "the edge must open");
        assert!(!sched.unlocked(MIN_NIGHTS - 1) && sched.unlocked(MIN_NIGHTS));

        // Thirty-one slots holding MIN_NIGHTS - 1 real nights stays shut, and one real night opens it.
        let mut padded: Vec<Option<f64>> = vec![Some(50.0); MIN_NIGHTS - 1];
        padded.extend(vec![None; 10]);
        padded.extend(vec![Some(HRV_MAX_MS + 0.001); 10]);
        padded.extend(vec![Some(HRV_MIN_MS - 0.001); 10]);
        assert!(HrvReadiness::evaluate(&padded).is_none(), "{} slots unlocked it", padded.len());
        // A scorer counting slots rather than nights answers the other way on exactly this input.
        assert_ne!(HrvReadiness::evaluate(&padded).is_some(), padded.len() >= MIN_NIGHTS);
        padded.push(Some(50.0));
        assert!(HrvReadiness::evaluate(&padded).is_some());

        // The band is inclusive at both ends, so a night AT the limit is a night.
        assert!(HrvReadiness::evaluate(&[Some(HRV_MIN_MS); MIN_NIGHTS]).is_some());
        assert!(HrvReadiness::evaluate(&[Some(HRV_MAX_MS); MIN_NIGHTS]).is_some());
        assert!(HrvReadiness::evaluate(&vec![Some(HRV_MIN_MS - 0.001); 40]).is_none());
        assert!(HrvReadiness::evaluate(&vec![Some(HRV_MAX_MS + 0.001); 40]).is_none());
    }

    #[test]
    fn flat_history_reads_normal() {
        let nights = vec![Some(50.0); 20];
        assert_eq!(HrvReadiness::evaluate(&nights).unwrap().tier, ReadinessTier::Normal);
    }

    /// RECORDED, not desired: at or below `ROLL_WINDOW` valid nights the 7-night tail and the long tail
    /// are the same slice, so `baseline7` equals `long_mean` bit for bit and no series of any shape can
    /// read anything but Normal, or raise the watch. The unlock opens at MIN_NIGHTS, four nights before
    /// the reading can carry a tier.
    #[test]
    fn no_history_at_or_under_seven_nights_can_read_anything_but_normal() {
        let shapes: [&dyn Fn(usize) -> f64; 4] = [
            &|i| if i % 2 == 0 { 20.0 } else { 120.0 },
            &|i| 20.0 + 15.0 * i as f64,
            &|i| 120.0 - 15.0 * i as f64,
            &|i| 5.0 + 245.0 * ((i * 7) % 5) as f64 / 4.0,
        ];
        for n in MIN_NIGHTS..=ROLL_WINDOW {
            for (k, shape) in shapes.iter().enumerate() {
                let nights: Vec<Option<f64>> = (0..n).map(|i| Some(shape(i))).collect();
                let r = HrvReadiness::evaluate(&nights).unwrap();
                assert_eq!(r.tier, ReadinessTier::Normal, "shape {k} over {n} nights left Normal");
                assert!(!r.overreaching_watch, "shape {k} over {n} nights raised the watch");
            }
        }
        // The eighth night is where the two windows can differ, so it is the first that can move.
        let ramp: Vec<Option<f64>> = (0..10).map(|i| Some(20.0 + 15.0 * i as f64)).collect();
        assert_eq!(HrvReadiness::evaluate(&ramp).unwrap().tier, ReadinessTier::Primed);
    }

    /// Twenty-three noisy or quiet nights then a week at one level. Two rise and two fall, and within
    /// each pair the tiers differ, so only the band decides them.
    fn band_cases() -> Vec<(&'static str, Vec<Option<f64>>, ReadinessTier)> {
        let history = |lo: f64, hi: f64, last: f64| -> Vec<Option<f64>> {
            let mut v: Vec<Option<f64>> =
                (0..23).map(|i| Some(if i % 2 == 0 { lo } else { hi })).collect();
            v.extend(vec![Some(last); 7]);
            v
        };
        vec![
            ("quiet history, week 4% up", history(49.0, 51.0, 53.0), ReadinessTier::Primed),
            ("noisy history, week 12% up", history(30.0, 70.0, 52.0), ReadinessTier::Normal),
            ("quiet history, week 4% down", history(49.0, 51.0, 47.0), ReadinessTier::Suppressed),
            ("noisy history, week 2% down", history(30.0, 70.0, 44.0), ReadinessTier::Normal),
        ]
    }

    #[test]
    fn readiness_tier_follows_the_normal_band_not_the_direction_of_the_last_week() {
        for (what, nights, want) in band_cases() {
            let r = HrvReadiness::evaluate(&nights).unwrap();
            assert_eq!(r.tier, want, "{what}: b7 {} in [{}, {}]",
                r.baseline7_ms, r.normal_low_ms, r.normal_high_ms);
            // The tier and the band are shown side by side, so they must agree.
            let from_band = if r.baseline7_ms > r.normal_high_ms { ReadinessTier::Primed }
                else if r.baseline7_ms >= r.normal_low_ms { ReadinessTier::Normal }
                else { ReadinessTier::Suppressed };
            assert_eq!(r.tier, from_band, "{what}: the tier contradicts the band it ships with");
        }
        // The step series the gate used to run on, with its band derived by hand: 13 nights at 40 and
        // 7 at 80 over one 20-night window, so the log-domain spread is 1820/400/19 of ln(2) squared.
        let mut step: Vec<Option<f64>> = vec![Some(40.0); 13];
        step.extend(vec![Some(80.0); 7]);
        let r = HrvReadiness::evaluate(&step).unwrap();
        let d = 2f64.ln();
        let long_mean = 40f64.ln() + 0.35 * d;
        let swc_half = SWC_K * (1820.0f64 / 400.0 / 19.0).sqrt() * d;
        assert!((r.baseline7_ms - 80.0).abs() < 1e-9, "got {}", r.baseline7_ms);
        assert!((r.normal_low_ms - (long_mean - swc_half).exp()).abs() < 1e-9, "got {}", r.normal_low_ms);
        assert!((r.normal_high_ms - (long_mean + swc_half).exp()).abs() < 1e-9, "got {}", r.normal_high_ms);
        // It clears the band by a third of the band's own value, which is why it never probed anything.
        assert!(r.baseline7_ms > r.normal_high_ms * 1.3);
    }

    #[test]
    fn a_do_nothing_readiness_tier_fails_the_band_cases() {
        let valid = |v: &[Option<f64>]| -> Vec<f64> { v.iter().flatten().copied().collect() };
        let always_normal = |_: &[Option<f64>]| ReadinessTier::Normal;
        let by_week_vs_history = |v: &[Option<f64>]| {
            let xs = valid(v);
            let week = mean(&xs[xs.len() - 7..]);
            let all = mean(&xs);
            if week > all { ReadinessTier::Primed } else if week < all { ReadinessTier::Suppressed }
            else { ReadinessTier::Normal }
        };
        let by_first_and_last = |v: &[Option<f64>]| {
            let xs = valid(v);
            if xs[xs.len() - 1] > xs[0] { ReadinessTier::Primed }
            else if xs[xs.len() - 1] < xs[0] { ReadinessTier::Suppressed }
            else { ReadinessTier::Normal }
        };
        let scorers: [Scorer<&[Option<f64>], ReadinessTier>; 3] = [
            ("always Normal", &always_normal),
            ("the last week against the whole history", &by_week_vs_history),
            ("the last night against the first", &by_first_and_last),
        ];
        let cases = band_cases();
        for (name, f) in scorers {
            let misses = cases.iter().any(|(_, nights, want)| f(nights) != *want);
            assert!(misses, "the do-nothing tier `{name}` reproduces every band case");
        }
    }

    /// The long window is 60 nights once 60 valid nights exist and 30 before that, so the band widens
    /// on the sixtieth night. Built with the older half far noisier than the recent half.
    #[test]
    fn the_normal_band_widens_when_the_sixty_night_window_opens() {
        let noisy: Vec<Option<f64>> = (0..30).map(|i| Some(if i % 2 == 0 { 30.0 } else { 80.0 })).collect();
        let quiet: Vec<Option<f64>> = (0..30).map(|i| Some(if i % 2 == 0 { 49.0 } else { 51.0 })).collect();
        let mut sixty = noisy.clone();
        sixty.extend(quiet.clone());
        let wide = HrvReadiness::evaluate(&sixty).unwrap();
        let narrow = HrvReadiness::evaluate(&sixty[1..]).unwrap();
        let quiet_only = HrvReadiness::evaluate(&quiet).unwrap();
        assert_eq!(LONG_WINDOW_FALLBACK, quiet.len(), "the fallback window is what the 59-night case uses");
        // At 59 valid nights the band is the last 30 nights' band, which here is the quiet half alone.
        assert!((narrow.normal_low_ms - quiet_only.normal_low_ms).abs() < 1e-9);
        assert!((narrow.normal_high_ms - quiet_only.normal_high_ms).abs() < 1e-9);
        // The sixtieth night pulls the noisy half in and the band opens by more than tenfold.
        let (w, n) = (wide.normal_high_ms - wide.normal_low_ms, narrow.normal_high_ms - narrow.normal_low_ms);
        assert!(w > 10.0 * n, "band {w} at 60 nights vs {n} at 59");
    }

    /// Thirty-five nights of an alternation whose amplitude and level each trend. `amp_step` sets the
    /// CV trend and `level_step` sets where the last week sits against the long mean.
    fn trending_nights(amp0: f64, amp_step: f64, level0: f64, level_step: f64) -> Vec<Option<f64>> {
        (0..35)
            .map(|i| {
                let amp = amp0 + amp_step * i as f64;
                Some(level0 + level_step * i as f64 + if i % 2 == 0 { amp } else { -amp })
            })
            .collect()
    }

    /// One overreaching arm: what it is, the nights, and whether the watch must be raised.
    type WatchCase = (&'static str, Vec<Option<f64>>, bool);

    /// The overreaching watch is a conjunction: the 7-night CV trend must be FALLING and the 7-night
    /// baseline must sit UNDER the long mean. All four arms read Normal, so the flag is not the tier.
    #[test]
    fn the_overreaching_watch_needs_a_falling_cv_and_a_baseline_under_the_long_mean() {
        let cases: [WatchCase; 4] = [
            ("cv falling, week under", trending_nights(12.0, -0.3, 55.0, -0.2), true),
            ("cv falling, week over", trending_nights(12.0, -0.3, 45.0, 0.2), false),
            ("cv rising, week under", trending_nights(1.0, 0.3, 55.0, -0.2), false),
            ("cv rising, week over", trending_nights(1.0, 0.3, 45.0, 0.2), false),
        ];
        for (what, nights, want) in &cases {
            let r = HrvReadiness::evaluate(nights).unwrap();
            assert_eq!(r.overreaching_watch, *want, "{what}");
            assert_eq!(r.tier, ReadinessTier::Normal, "{what}: the watch must not be read off the tier");
        }
        // And a plainly Suppressed week does not raise it, so `watch == suppressed` is not the rule.
        let mut falling: Vec<Option<f64>> = vec![Some(80.0); 13];
        falling.extend(vec![Some(40.0); 7]);
        let s = HrvReadiness::evaluate(&falling).unwrap();
        assert_eq!((s.tier, s.overreaching_watch), (ReadinessTier::Suppressed, false));

        let always_on = |_: &WatchCase| true;
        let always_off = |_: &WatchCase| false;
        let from_tier =
            |c: &WatchCase| HrvReadiness::evaluate(&c.1).unwrap().tier == ReadinessTier::Suppressed;
        let scorers: [Scorer<&WatchCase, bool>; 3] =
            [("always on", &always_on), ("always off", &always_off), ("the Suppressed tier", &from_tier)];
        for (name, f) in scorers {
            assert!(cases.iter().any(|c| f(c) != c.2), "the do-nothing watch `{name}` reproduces all four arms");
        }
    }

    #[test]
    fn implausible_and_missing_nights_are_dropped() {
        // 2 valid + a null + an out-of-range 400 → still only 2 valid → below MIN_NIGHTS → calibrating.
        let mut nights = vec![Some(50.0); 2];
        nights.push(None);
        nights.push(Some(400.0));
        assert!(HrvReadiness::evaluate(&nights).is_none());
    }
}
