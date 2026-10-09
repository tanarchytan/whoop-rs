//! Read-only tuning tables, one record per algorithm, in the shape of `hrv_clean_cfg`. A frontend
//! reads a threshold through here instead of declaring its own copy, so the value has one owner.

use crate::*;

/// Charge (recovery) weights, logistic shape, band cuts and the two window gates.
#[derive(uniffi::Record)]
pub struct RecoveryCfgInfo {
    pub w_hrv: f64,
    pub w_rhr: f64,
    pub w_resp: f64,
    pub w_sleep: f64,
    pub w_skin_temp: f64,
    pub w_recovery_index: f64,
    pub w_activity_balance: f64,
    pub skin_temp_dev_scale: f64,
    pub recovery_index_scale_bpm_per_hr: f64,
    pub logistic_k: f64,
    pub logistic_z0: f64,
    pub band_red_max: f64,
    pub band_yellow_max: f64,
    pub sleep_perf_center: f64,
    pub sleep_perf_scale: f64,
    pub resting_hr_window_s: i64,
    pub recovery_index_min_bins: u32,
}

/// The Charge tuning. `skin_temp_dev_scale` must stay 1.0: halving it doubles that term's effective weight.
#[uniffi::export]
pub fn recovery_cfg() -> RecoveryCfgInfo {
    RecoveryCfgInfo {
        w_hrv: recovery::W_HRV,
        w_rhr: recovery::W_RHR,
        w_resp: recovery::W_RESP,
        w_sleep: recovery::W_SLEEP,
        w_skin_temp: recovery::W_SKIN_TEMP,
        w_recovery_index: recovery::W_RECOVERY_INDEX,
        w_activity_balance: recovery::W_ACTIVITY_BALANCE,
        skin_temp_dev_scale: recovery::SKIN_TEMP_DEV_SCALE,
        recovery_index_scale_bpm_per_hr: recovery::RECOVERY_INDEX_SCALE_BPM_PER_HR,
        logistic_k: recovery::LOGISTIC_K,
        logistic_z0: recovery::LOGISTIC_Z0,
        band_red_max: recovery::BAND_RED_MAX,
        band_yellow_max: recovery::BAND_YELLOW_MAX,
        sleep_perf_center: recovery::SLEEP_PERF_CENTER,
        sleep_perf_scale: recovery::SLEEP_PERF_SCALE,
        resting_hr_window_s: recovery::RESTING_HR_WINDOW_S,
        recovery_index_min_bins: recovery::RECOVERY_INDEX_MIN_BINS as u32,
    }
}

/// Rest (sleep performance) weights and the duration / restorative shape.
#[derive(uniffi::Record)]
pub struct RestCfgInfo {
    pub w_duration: f64,
    pub w_efficiency: f64,
    pub w_restorative: f64,
    pub w_consistency: f64,
    pub default_sleep_need_hours: f64,
    pub restorative_target_share: f64,
    pub deep_share_target: f64,
    pub deep_floor_factor: f64,
    pub neutral_consistency: f64,
}

/// The Rest tuning behind `rest_score`.
#[uniffi::export]
pub fn rest_cfg() -> RestCfgInfo {
    RestCfgInfo {
        w_duration: rest::W_DURATION,
        w_efficiency: rest::W_EFFICIENCY,
        w_restorative: rest::W_RESTORATIVE,
        w_consistency: rest::W_CONSISTENCY,
        default_sleep_need_hours: rest::DEFAULT_SLEEP_NEED_HOURS,
        restorative_target_share: rest::RESTORATIVE_TARGET_SHARE,
        deep_share_target: rest::DEEP_SHARE_TARGET,
        deep_floor_factor: rest::DEEP_FLOOR_FACTOR,
        neutral_consistency: rest::NEUTRAL_CONSISTENCY,
    }
}

/// Effort (strain) scale, log-map denominator, the two coverage gates, and the Day Strain axis a
/// display or an import boundary converts to.
#[derive(uniffi::Record)]
pub struct StrainCfgInfo {
    pub min_readings: u32,
    pub min_span_seconds: i64,
    pub max_strain: f64,
    pub denominator: f64,
    pub whoop_day_strain_max: f64,
    pub whoop_day_strain_to_effort: f64,
    pub effort_to_whoop_day_strain: f64,
}

/// The Effort tuning behind `strain_score`.
#[uniffi::export]
pub fn strain_cfg() -> StrainCfgInfo {
    StrainCfgInfo {
        min_readings: strain::MIN_READINGS as u32,
        min_span_seconds: strain::MIN_SPAN_SECONDS,
        max_strain: strain::MAX_STRAIN,
        denominator: strain::STRAIN_DENOMINATOR,
        whoop_day_strain_max: strain::WHOOP_DAY_STRAIN_MAX,
        whoop_day_strain_to_effort: strain::WHOOP_DAY_STRAIN_TO_EFFORT,
        effort_to_whoop_day_strain: strain::EFFORT_TO_WHOOP_DAY_STRAIN,
    }
}

/// Sex baselines, the Effort bump ceiling and the rounding grid behind `hydration_daily_goal_ml`.
#[derive(uniffi::Record)]
pub struct HydrationCfgInfo {
    pub baseline_male_ml: i32,
    pub baseline_female_ml: i32,
    pub baseline_other_ml: i32,
    pub max_effort_bump_ml: i32,
    pub round_to_ml: i32,
}

/// The daily fluid-goal tuning.
#[uniffi::export]
pub fn hydration_cfg() -> HydrationCfgInfo {
    HydrationCfgInfo {
        baseline_male_ml: hydration::BASELINE_MALE_ML,
        baseline_female_ml: hydration::BASELINE_FEMALE_ML,
        baseline_other_ml: hydration::BASELINE_OTHER_ML,
        max_effort_bump_ml: hydration::MAX_EFFORT_BUMP_ML,
        round_to_ml: hydration::ROUND_TO_ML,
    }
}

/// Nightly-baseline lifecycle gates: seed, full trust, and the fast-adapt window.
#[derive(uniffi::Record)]
pub struct BaselinesCfgInfo {
    pub min_nights_seed: i32,
    pub min_nights_trust: i32,
    pub early_adapt_nights: i32,
}

/// The cold-start gates every metric baseline shares, beside the per-metric `baseline_metric_cfg`.
#[uniffi::export]
pub fn baselines_cfg() -> BaselinesCfgInfo {
    BaselinesCfgInfo {
        min_nights_seed: baselines::MIN_NIGHTS_SEED,
        min_nights_trust: baselines::MIN_NIGHTS_TRUST,
        early_adapt_nights: baselines::EARLY_ADAPT_NIGHTS,
    }
}

/// Body-Age clamp and the reading's ± band.
#[derive(uniffi::Record)]
pub struct VitalityCfgInfo {
    pub min_body_age: f64,
    pub max_body_age: f64,
    pub band_years: f64,
}

/// The Vitality / Body Age tuning behind `vitality_compute`.
#[uniffi::export]
pub fn vitality_cfg() -> VitalityCfgInfo {
    VitalityCfgInfo {
        min_body_age: vitality::MIN_BODY_AGE,
        max_body_age: vitality::MAX_BODY_AGE,
        band_years: vitality::BAND_YEARS,
    }
}

/// Sleep-debt ledger window and the on-target band width (minutes).
#[derive(uniffi::Record)]
pub struct SleepDebtCfgInfo {
    pub default_window_nights: u32,
    pub on_target_band_min: f64,
}

/// The ledger tuning behind `sleep_debt_ledger`.
#[uniffi::export]
pub fn sleep_debt_cfg() -> SleepDebtCfgInfo {
    SleepDebtCfgInfo {
        default_window_nights: sleep_debt::DEFAULT_WINDOW_NIGHTS as u32,
        on_target_band_min: sleep_debt::ON_TARGET_BAND_MIN,
    }
}

/// Nap detector defaults: length bounds, stillness, HR settle margin and the smoothing window.
#[derive(uniffi::Record)]
pub struct NapDefaultsInfo {
    pub min_nap_min: i32,
    pub max_nap_min: i32,
    pub still_threshold_g: f64,
    pub hr_settle_margin_bpm: i32,
    pub smooth_window_s: f64,
}

/// The defaults a `NapConfigInfo` starts from before the user tunes it.
#[uniffi::export]
pub fn nap_defaults() -> NapDefaultsInfo {
    NapDefaultsInfo {
        min_nap_min: nap::DEFAULT_MIN_NAP_MIN,
        max_nap_min: nap::DEFAULT_MAX_NAP_MIN,
        still_threshold_g: nap::DEFAULT_STILL_THRESHOLD_G,
        hr_settle_margin_bpm: nap::DEFAULT_HR_SETTLE_MARGIN_BPM,
        smooth_window_s: nap::DEFAULT_SMOOTH_WINDOW_S,
    }
}

/// Sleep-detection window edges: the daytime-nap band, the session gap ceiling and the sparse-gravity
/// span fraction; plus the overnight band and the habitual-midsleep day floor and trailing window span.
#[derive(uniffi::Record)]
pub struct SleepWindowCfgInfo {
    pub daytime_band_start_hour: i64,
    pub daytime_band_end_hour: i64,
    pub max_gap_min: i64,
    pub sparse_gravity_span_frac: f64,
    pub overnight_start_hour: i64,
    pub overnight_end_hour: i64,
    pub habitual_min_days: u32,
    pub habitual_window_days: u32,
}

/// The window edges the sleep pipeline detects and picks a main night by.
#[uniffi::export]
pub fn sleep_window_cfg() -> SleepWindowCfgInfo {
    SleepWindowCfgInfo {
        daytime_band_start_hour: sleep::DAYTIME_BAND_START_HOUR,
        daytime_band_end_hour: sleep::DAYTIME_BAND_END_HOUR,
        max_gap_min: sleep::MAX_GAP_MIN,
        sparse_gravity_span_frac: sleep::SPARSE_GRAVITY_SPAN_FRAC,
        overnight_start_hour: sleep::OVERNIGHT_START_HOUR,
        overnight_end_hour: sleep::OVERNIGHT_END_HOUR,
        habitual_min_days: sleep::HABITUAL_MIN_DAYS as u32,
        habitual_window_days: sleep::HABITUAL_WINDOW_DAYS as u32,
    }
}

/// The R-R filler rule: an exact `fill_rr_ms` beat with a same-second HR strictly below `fill_max_hr_bpm`.
#[derive(uniffi::Record)]
pub struct RrFillCfgInfo {
    pub fill_rr_ms: u16,
    pub fill_max_hr_bpm: u32,
}

/// The filler constants, so a SQL backfill binds them instead of declaring its own copy.
#[uniffi::export]
pub fn rr_fill_cfg() -> RrFillCfgInfo {
    RrFillCfgInfo {
        fill_rr_ms: physio_algo::rr::FILL_RR_MS,
        fill_max_hr_bpm: physio_algo::rr::FILL_MAX_HR_BPM as u32,
    }
}

/// True when this beat is the 500 ms filler at a resting HR (`hr_bpm` from the same second, `None` when
/// the strap stored none: not flagged). Marked suspect by the caller, never deleted.
#[uniffi::export]
pub fn is_rr_fill(rr_ms: u16, hr_bpm: Option<u32>) -> bool {
    physio_algo::rr::is_rr_fill(rr_ms, hr_bpm.map(|h| h.min(u8::MAX as u32) as u8))
}

/// A standard BLE heart-rate (0x2A37) R-R word in ms: 1/1024 s per the spec, or plain ms when
/// `plain_ms` (a strap family that sends milliseconds on that characteristic).
#[uniffi::export]
pub fn standard_rr_ms(raw: u16, plain_ms: bool) -> u16 {
    physio_algo::rr::standard_rr_word_ms(raw, plain_ms)
}

#[cfg(test)]
mod rr_fill_tests {
    use super::*;

    #[test]
    fn ffi_matches_the_core_rule() {
        let c = rr_fill_cfg();
        assert_eq!((c.fill_rr_ms, c.fill_max_hr_bpm), (500, 100));
        assert!(is_rr_fill(500, Some(99)));
        assert!(!is_rr_fill(500, Some(100)));
        assert!(!is_rr_fill(500, Some(100_000)));
        assert!(!is_rr_fill(500, None));
        assert!(!is_rr_fill(512, Some(60)));
        assert_eq!((standard_rr_ms(1024, false), standard_rr_ms(1024, true)), (1000, 1024));
    }
}
