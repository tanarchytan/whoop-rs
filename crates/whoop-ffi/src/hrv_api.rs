//! Beat-interval analysis: RMSSD and its variants, frequency domain, readiness, and the
//! respiration rate derived from the same intervals.

use crate::*;

/// HR from a v26 optical PPG buffer (24 Hz autocorrelation).
#[uniffi::export]
pub fn ppg_hr(samples: Vec<PpgSample>) -> Vec<PpgEstimate> {
    let s: Vec<ppg_hr::Sample> = samples.into_iter().map(|p| ppg_hr::Sample { ts: p.ts, value: p.value }).collect();
    ppg_hr::estimate(&s)
        .into_iter()
        .map(|e| PpgEstimate { ts: e.ts, bpm: e.bpm, conf: e.conf })
        .collect()
}

/// Gap-aware, artifact-corrected nightly RMSSD (ms) from per-record R-R runs.
#[uniffi::export]
pub fn hrv_rmssd_gap_aware(runs: Vec<RrRun>) -> Option<f64> {
    let beats: Vec<(u32, Vec<u16>)> = runs.into_iter().map(|r| (r.unix, r.rr)).collect();
    HrvReadiness::rmssd_gap_aware(&beats)
}

/// Windowed session avgHrv (ms): the mean of per-5-min-bucket gap-aware RMSSD over `[start, end]`, the
/// app's stored `SleepSession.avgHrv`. `runs` are the session's per-record `(unix, rr)` in chronological
/// order; buckets tumble from `start`. `None` when no bucket yields a value.
#[uniffi::export]
pub fn hrv_windowed_avg(start: u32, end: u32, runs: Vec<RrRun>) -> Option<f64> {
    let beats: Vec<(u32, u16)> = runs
        .into_iter()
        .flat_map(|r| {
            let unix = r.unix;
            r.rr.into_iter().map(move |v| (unix, v))
        })
        .collect();
    HrvReadiness::windowed_avg_hrv(start, end, &beats)
}

/// HRV-readiness over a nightly RMSSD series (oldest → newest; `None` slots = missing nights).
#[uniffi::export]
pub fn hrv_readiness(nightly_rmssd: Vec<Option<f64>>) -> Option<HrvReadinessInfo> {
    HrvReadiness::evaluate(&nightly_rmssd).map(|r| HrvReadinessInfo {
        tier: r.tier.into(),
        baseline7_ms: r.baseline7_ms,
        normal_low_ms: r.normal_low_ms,
        normal_high_ms: r.normal_high_ms,
        overreaching_watch: r.overreaching_watch,
    })
}

/// RMSSD (ms) of raw R-R values. `None` for <2 beats (filtered, drops deltas >200ms).
#[uniffi::export]
pub fn hrv_rmssd(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::rmssd(&rr_ms)
}

/// Plain unfiltered RMSSD (no artifact rejection); the raw counterpart to `hrv_rmssd`.
#[uniffi::export]
pub fn hrv_rmssd_plain(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::rmssd_plain(&rr_ms)
}

/// Plain pNN50 (% of successive |dNN| > 50 ms) over an already-clean NN series: the formula alone, no
/// cleaning and no contiguity mask. `None` for under two values.
#[uniffi::export]
pub fn hrv_pnn50_plain(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::pnn50_plain(&rr_ms)
}

/// Range-filter R-R values, keeping only 300–2000 ms.
#[uniffi::export]
pub fn hrv_range_filter(rr_ms: Vec<u16>) -> Vec<u16> {
    HrvReadiness::range_filter(&rr_ms)
}

/// Standard deviation of NN intervals (ms), sample std (ddof=1). `None` for <2 values.
#[uniffi::export]
pub fn hrv_sdnn(rr_ms: Vec<u16>) -> Option<f64> {
    HrvReadiness::sdnn(&rr_ms)
}

/// Range + Malik-ectopic cleaned NN series (ms), in input order. Ungated, so a cleaning trace can count
/// survivors where `hrv_analyze_raw` reports zero once its beat-count or rejected-fraction gate refuses.
#[uniffi::export]
pub fn hrv_clean_rr(rr_ms: Vec<u16>) -> Vec<u16> {
    HrvReadiness::clean_rr(&rr_ms)
}

/// Survivor counts from one cleaning pass: input, after the range filter, after Malik ectopic.
#[derive(uniffi::Record)]
pub struct HrvCleanCountsInfo {
    pub n_input: u32,
    pub n_ranged: u32,
    pub n_clean: u32,
}

/// Stage-by-stage cleaning counts, so a trace can name WHY beats were dropped. Ungated.
#[uniffi::export]
pub fn hrv_clean_counts(rr_ms: Vec<u16>) -> HrvCleanCountsInfo {
    let c = physio_algo::hrv::clean_counts(&rr_ms);
    HrvCleanCountsInfo { n_input: c.n_input, n_ranged: c.n_ranged, n_clean: c.n_clean }
}

/// The cleaning-pipeline tuning the app displays: R-R bounds, the clean-beat floor, the Malik threshold
/// and window, the spot rejected-fraction ceiling, and the rolling-trace window.
#[derive(uniffi::Record)]
pub struct HrvCleanCfgInfo {
    pub rr_min_ms: u16,
    pub rr_max_ms: u16,
    pub min_beats: u32,
    pub ectopic_threshold: f64,
    pub ectopic_window_radius: u32,
    pub spot_max_rejected_fraction: f64,
    pub rolling_window_secs: i64,
}

/// One source of truth for the cleaning constants, so a caller cannot hold a stale copy.
#[uniffi::export]
pub fn hrv_clean_cfg() -> HrvCleanCfgInfo {
    HrvCleanCfgInfo {
        rr_min_ms: physio_algo::hrv::RR_MIN_MS,
        rr_max_ms: physio_algo::hrv::RR_MAX_MS,
        min_beats: physio_algo::hrv::MIN_BEATS as u32,
        ectopic_threshold: physio_algo::hrv::ECTOPIC_THRESHOLD,
        ectopic_window_radius: physio_algo::hrv::ECTOPIC_WINDOW_RADIUS as u32,
        spot_max_rejected_fraction: physio_algo::hrv::SPOT_MAX_REJECTED_FRACTION,
        rolling_window_secs: physio_algo::hrv::ROLLING_WINDOW_SECS,
    }
}

/// One 5-min bucket of a session: its start (unix s), the clean beats in it, and its gap-aware RMSSD.
#[derive(uniffi::Record)]
pub struct HrvBucketInfo {
    pub start: u32,
    pub clean_beats: u32,
    pub rmssd: Option<f64>,
}

/// The per-bucket breakdown behind `hrv_windowed_avg`, for a caller that tags each bucket by sleep stage.
#[uniffi::export]
pub fn hrv_windowed_buckets(start: u32, end: u32, runs: Vec<RrRun>) -> Vec<HrvBucketInfo> {
    let beats: Vec<(u32, u16)> = runs.iter().flat_map(|r| r.rr.iter().map(|b| (r.unix, *b))).collect();
    HrvReadiness::windowed_buckets(start, end, &beats)
        .into_iter()
        .map(|b| HrvBucketInfo { start: b.start, clean_beats: b.clean_beats, rmssd: b.rmssd })
        .collect()
}

/// One point of the rolling RMSSD trace: the trailing window's end (unix s) and its RMSSD (ms).
#[derive(uniffi::Record)]
pub struct RollingRmssdPoint {
    pub ts: i64,
    pub rmssd: f64,
}

/// Rolling trailing-window RMSSD over a timestamped R-R series, for a within-day chart. `step_s > 0`
/// thins emission to one point per that many seconds of advance.
#[uniffi::export]
pub fn hrv_rolling_rmssd(
    beats: Vec<RrBeat>,
    window_s: i64,
    step_s: i64,
    min_beats: u32,
) -> Vec<RollingRmssdPoint> {
    let b: Vec<(i64, u16)> = beats.into_iter().map(|x| (x.ts, x.rr_ms)).collect();
    physio_algo::hrv::rolling_rmssd(&b, window_s, step_s, min_beats as usize)
        .into_iter()
        .map(|(ts, rmssd)| RollingRmssdPoint { ts, rmssd })
        .collect()
}

/// Total beat-time over elapsed wall-clock time for the same beats. Over ~1.0 means beats are
/// double-counted or reports overlap; `0.0` for under two timestamps or a non-positive span.
#[uniffi::export]
pub fn hrv_rr_coverage(ts_sec: Vec<i64>, rr_ms: Vec<f64>) -> f64 {
    physio_algo::hrv::rr_coverage(&ts_sec, &rr_ms)
}

/// Rows repeating an earlier `(ts, rr_ms)` exactly. Byte-identical re-inserts only; it cannot see the
/// value-differing report overlap that `hrv_overlapping_reports` counts.
#[uniffi::export]
pub fn hrv_duplicate_beat_count(ts_sec: Vec<i64>, rr_ms: Vec<f64>) -> u32 {
    physio_algo::hrv::duplicate_beat_count(&ts_sec, &rr_ms)
}

/// Per-second reports whose first beat re-reports time earlier reports already covered, and the total.
#[derive(uniffi::Record)]
pub struct RrOverlapInfo {
    pub overlapping: u32,
    pub total: u32,
}

/// The overlap behind an `hrv_rr_coverage` above 1.0: the same test that breaks contiguity in the
/// gap-aware RMSSD, reported as a count.
#[uniffi::export]
pub fn hrv_overlapping_reports(runs: Vec<RrRun>) -> RrOverlapInfo {
    let reports: Vec<(u32, Vec<u16>)> = runs.into_iter().map(|r| (r.unix, r.rr)).collect();
    let (overlapping, total) = physio_algo::hrv::overlapping_report_count(&reports);
    RrOverlapInfo { overlapping, total }
}

/// Full HRV analysis over a raw R-R capture (ms): cleaned RMSSD/SDNN/pNN50/meanNN + input/clean counts.
/// Fields are `None` (and `n_clean` 0) below the 20-clean-beat floor, or when `max_rejected_fraction` (the
/// spot honesty gate) is set and exceeded. The nightly path passes `None` (no gate).
#[derive(uniffi::Record)]
pub struct HrvAnalysisInfo {
    pub rmssd: Option<f64>,
    pub sdnn: Option<f64>,
    pub mean_nn: Option<f64>,
    pub pnn50: Option<f64>,
    pub n_input: u32,
    pub n_clean: u32,
}

/// Clean-and-analyze a raw R-R series in one call (the app's full spot/nightly HRV analysis path).
#[uniffi::export]
pub fn hrv_analyze_raw(rr_ms: Vec<u16>, max_rejected_fraction: Option<f64>) -> HrvAnalysisInfo {
    let a = HrvReadiness::analyze_raw(&rr_ms, max_rejected_fraction);
    HrvAnalysisInfo {
        rmssd: a.rmssd,
        sdnn: a.sdnn,
        mean_nn: a.mean_nn,
        pnn50: a.pnn50,
        n_input: a.n_input,
        n_clean: a.n_clean,
    }
}

/// Frequency-domain HRV bands (ms²): LF / HF / LF-HF / total power. `lf`/`lfhf` are `None` under the 250 s
/// LF span gate; the whole record is absent (`None`) under the 60 s HF gate or the 20-beat floor.
#[derive(uniffi::Record)]
pub struct HrvBandsInfo {
    pub lf: Option<f64>,
    pub hf: f64,
    pub lfhf: Option<f64>,
    pub total_power: f64,
}

/// Frequency-domain HRV over a time-ordered R-R series (ms) via the Lomb-Scargle periodogram.
#[uniffi::export]
pub fn hrv_freq_domain(rr_ms: Vec<u16>) -> Option<HrvBandsInfo> {
    physio_algo::hrv_freq::freq_domain(&rr_ms).map(|b| HrvBandsInfo {
        lf: b.lf,
        hf: b.hf,
        lfhf: b.lfhf,
        total_power: b.total_power,
    })
}

/// One R-R beat for respiratory-rate estimation (unix seconds + interval ms).
#[derive(uniffi::Record)]
pub struct RrBeat {
    pub ts: i64,
    pub rr_ms: u16,
}

/// Respiratory rate (breaths/min) from R-R via RSA. `None` when the signal is too thin.
#[uniffi::export]
pub fn resp_rate_from_rr(beats: Vec<RrBeat>, start: i64, end: i64) -> Option<f64> {
    let b: Vec<(i64, u16)> = beats.into_iter().map(|x| (x.ts, x.rr_ms)).collect();
    respiratory_rate::resp_rate_from_rr(&b, start, end)
}

/// THE nightly HRV value (ms) a readiness word is computed from: per-5min-bucket gap-aware RMSSD over
/// deep-sleep (SWS/N3) buckets only, from reports the strap's optical front end did not flag. Takes the
/// full segment list and filters for `SleepStage::Deep` internally. `None` when no deep bucket yields one.
#[uniffi::export]
pub fn hrv_nightly(
    start: u32,
    end: u32,
    reports: Vec<RrReport>,
    segments: Vec<SleepSegment>,
) -> Option<f64> {
    let r: Vec<physio_algo::hrv::RrReport> = reports
        .into_iter()
        .map(|x| physio_algo::hrv::RrReport {
            unix: x.unix,
            rr: x.rr,
            optical_signal_poor: x.optical_signal_poor,
        })
        .collect();
    let deep_spans: Vec<(u32, u32)> = segments
        .iter()
        .filter(|s| matches!(s.stage, SleepStage::Deep))
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    HrvReadiness::nightly_hrv(start, end, &r, &deep_spans)
}

/// [`hrv_nightly`] behind the R-R coverage gate. `duplicated_beats` means the night's stored beats
/// outnumber its covered time, so `value` is withheld; `coverage` is the measured beat-time ratio.
#[derive(uniffi::Record)]
pub struct HrvNightlyChecked {
    pub value: Option<f64>,
    pub duplicated_beats: bool,
    pub coverage: f64,
}

#[uniffi::export]
pub fn hrv_nightly_checked(
    start: u32,
    end: u32,
    reports: Vec<RrReport>,
    segments: Vec<SleepSegment>,
) -> HrvNightlyChecked {
    let r: Vec<physio_algo::hrv::RrReport> = reports
        .into_iter()
        .map(|x| physio_algo::hrv::RrReport { unix: x.unix, rr: x.rr, optical_signal_poor: x.optical_signal_poor })
        .collect();
    let deep_spans: Vec<(u32, u32)> = segments
        .iter()
        .filter(|s| matches!(s.stage, SleepStage::Deep))
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    let n = HrvReadiness::nightly_hrv_checked(start, end, &r, &deep_spans);
    match n.quality {
        physio_algo::hrv::HrvQuality::Ok => HrvNightlyChecked { value: n.value, duplicated_beats: false, coverage: 0.0 },
        physio_algo::hrv::HrvQuality::DuplicatedBeats { coverage } => {
            HrvNightlyChecked { value: None, duplicated_beats: true, coverage }
        }
    }
}

// ── Rest (sleep performance composite) ─────────────────────────────────────

/// Nocturnal RMSSD age norm (ms) — the reference the HRV driver is scored against.
#[uniffi::export]
pub fn vitality_rmssd_norm(for_age: f64) -> f64 {
    vitality::rmssd_norm(for_age)
}
