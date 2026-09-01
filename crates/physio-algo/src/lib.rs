//! Brand-neutral physiological algorithms — the analytics layer that turns decoded physiological
//! signals into wellness estimates. The core entry points take plain values (R-R runs, PPG samples,
//! accel/motion, per-epoch fields); a few history adapters accept an already-decoded `HistoryRecord`
//! slice — never a wire frame, never BLE. Pure and deterministic: no async, no IO. Absent signal
//! returns `None`, never a fabricated number. Outputs are wellness estimates, never medical advice.
//!
//! Modules are organised by physiological domain: `sleep`, `ppg`, `hrv`, `hrv_freq`, `spo2`, `calibration`,
//! `hr_anomaly`, `recovery`, `strain`, `stress`, `resting_hr`, `hr_zones`, `vo2max`,
//! `respiratory_rate`, `imu_features` and the shared `stats`.

pub mod device_agreement;
pub mod apportion;
pub mod baselines;
pub mod biological_age;
pub mod bp;
pub mod calories;
pub mod calibration;
pub mod circadian;
pub mod ecg;
pub mod hr_anomaly;
pub mod hr_sample;
pub mod hr_recovery;
pub mod hr_zones;
pub mod hrv;
pub mod hrv_freq;
pub mod hydration;
pub mod imu_features;
pub mod lda;
pub mod nap;
pub mod ppg;
pub mod ramps;
pub mod recovery;
pub mod recovery_drivers;
pub mod respiratory_rate;
pub mod rest;
pub mod resting_hr;
pub mod rr_irregularity;
pub mod sleep;
pub mod sleep_debt;
pub mod signal;
pub mod sleep_regularity;
pub mod spo2;
pub mod stats;
pub mod steps;
pub mod strain;
pub mod stress;
pub mod stress_onset;
pub mod vital_bands;
pub mod vitality;
pub mod vo2max;
pub mod workout;
pub mod illness;
pub mod hr_gap;
pub mod worn;

pub use calibration::Calibration;
pub use hr_anomaly::{HrWatch, HrWatchState};
pub use hr_sample::HrSample;
pub use hrv::{HrvAnalysis, HrvReadiness, HrvReadinessResult, ReadinessTier, SECS_PER_DAY};
pub use hrv_freq::HrvBands;
pub use nap::{NapCandidate, NapConfig, NapDecision, NapVerdict};
pub use spo2::{RollingReading, Spo2};
pub use stats::{linear_fit, pearson, LinearFit};
pub use worn::{worn_state, WornState};
