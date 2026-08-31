//! Negative controls for the RECOVERY family: Charge score, band/state, driver rows, Recovery Index,
//! banked nights, HRV readiness, personal baselines, illness-watch baseline z.
//!
//! The claim under test is NOT "recovery is correct". It is: **would the shipped gates in
//! `recovery.rs`, `recovery_drivers.rs`, `hrv.rs`, `baselines.rs` and `illness.rs` notice if the
//! algorithm broke?** Each metric gets a BASELINE arm that must reproduce the shipped figure, NULL arms
//! whose scorer does no work (the gate MUST fail; if it passes the gate is fake), STRUCTURAL arms that
//! keep the magnitude and break the shape, and a PARAMETER arm per tunable at x1.10 / x0.90 plus a
//! +0.5% floor probe where the tunable is continuous.
//!
//! Every target and tolerance below is copied verbatim from the shipped assertion with the `file:line`
//! it came from, so the control tests the real claim rather than a paraphrase. A replica must carry the
//! WHOLE cohort — replicating some of a module's tests reports blindness the omitted ones would catch,
//! so `base_gate_holds` covers all thirteen `baselines.rs` tests and asserts that count from the source.
//!
//! Constants are `const`, so an arm cannot mutate shipped code. Where a config struct is an argument
//! (`MetricCfg`, `NightGate`, `banked_nights` bounds, `night_skin_temp`'s `min_conf`) the arm drives the
//! SHIPPED function. Everywhere else it drives a REPLICA asserted equal to the shipped function at
//! shipped parameters before any arm runs. Private literals (`hrv.rs` windows, `baselines.rs` winsor and
//! outlier constants) are copied into the replica and named in a comment.
//!
//! Where a shipped assertion seeds its own probe from the constant it is meant to pin
//! (`state(STATE_LOW_FLOOR)`, `Some(HRV_MIN_MS)`, `sleep_perf: Some(SLEEP_PERF_CENTER)`), the arm moves
//! the probe with the constant, because that is what the real gate does.
//!
//! `#[ignore]`d: a measurement harness, not CI. Run with
//! `cargo test --release -p physio-algo --test sensitivity_recovery -- --ignored --nocapture`.
//!
//! Recovery, readiness and illness-watch outputs are wellness estimates, never medical, never diagnostic.

use physio_algo::baselines::{
    fold_history, fold_history_nights, night_skin_temp, night_verdict, night_verdicts, update,
    BaselineState, BaselineStatus, MetricCfg, NightChannels, NightGate, NightMetric, NightVerdict,
    EARLY_ADAPT_NIGHTS, MIN_NIGHTS_SEED, MIN_NIGHTS_TRUST, MIN_SLEEP_SECS, WORN_SKIN_TEMP_C,
};
use physio_algo::calibration::RECOVERY_SCORE;
use physio_algo::hrv::{HrvReadiness, ReadinessTier};
use physio_algo::illness::{
    baseline_window, baseline_z_at, baseline_z_series, BASELINE_GAP_NIGHTS, BASELINE_WINDOW_NIGHTS,
    MIN_BASELINE_NIGHTS, SD_FLOOR,
};
use physio_algo::recovery::{
    band, banked_nights, recovery, recovery_index_slope, score_of, state, DriverBaseline, HrSample,
    RecoveryInput, RecoveryState, BAND_RED_MAX, BAND_YELLOW_MAX, HRV_MAX_MS, HRV_MIN_MS, LOGISTIC_K,
    LOGISTIC_Z0, RECOVERY_INDEX_MIN_BINS, RECOVERY_INDEX_SCALE_BPM_PER_HR, RESTING_HR_WINDOW_S,
    SKIN_TEMP_DEV_SCALE, SLEEP_PERF_CENTER, SLEEP_PERF_SCALE, STATE_LOW_FLOOR, STATE_MODERATE_FLOOR,
    STATE_PEAK_FLOOR, STATE_PRIMED_FLOOR, W_ACTIVITY_BALANCE, W_HRV, W_RECOVERY_INDEX, W_RESP,
    W_RHR, W_SKIN_TEMP, W_SLEEP,
};
use physio_algo::recovery_drivers::{
    driver_rows, DriverKind, DriverVerdict, SKIN_TEMP_TYPICAL_BAND_C,
};
use physio_algo::stats;

// ─── shipped gates, copied verbatim ────────────────────────────────────────────────────────────

/// recovery.rs — the composite gate: nine pinned three-driver nights spanning 0 to 100, the z = 0
/// population anchor, strict monotonicity and a span over 99 points.
const GATE_ANCHOR_TARGET: f64 = 57.93;
const GATE_ANCHOR_TOL: f64 = 0.05;
const GATE_ANCHOR_EXACT: f64 = 57.932425214875;
const GATE_SPREAD_NIGHTS: [(f64, f64, f64, f64); 9] = [
    (26.0, 70.0, 0.40, 0.171112969842),
    (32.0, 66.0, 0.55, 1.011365438792),
    (38.0, 62.0, 0.65, 5.167958386150),
    (44.0, 58.0, 0.75, 22.521055297547),
    (50.0, 55.0, 0.85, 57.932425214875),
    (56.0, 52.0, 0.90, 85.376542210808),
    (62.0, 49.0, 0.95, 96.116741498880),
    (70.0, 46.0, 1.00, 99.316784313320),
    (80.0, 42.0, 1.05, 99.924954045721),
];
const GATE_SPREAD_TOL: f64 = 1e-9;
const GATE_SPREAD_MIN_SPAN: f64 = 99.0;
/// The two all-seven-driver nights, and the claim that seven neutral terms renormalise to the same
/// anchor three do. These are the only rows the four optional weights and both scales can move.
const GATE_SEVEN_SUPPORTING: f64 = 91.257225183808;
const GATE_SEVEN_LIMITING: f64 = 5.719331490657;
/// The `band` cases, verbatim.
const GATE_BAND_CASES: [(f64, &str); 6] = [
    (20.0, "red"),
    (33.9, "red"),
    (34.0, "yellow"),
    (66.9, "yellow"),
    (67.0, "green"),
    (95.0, "green"),
];
/// The six `state` probes written as literals. The other four rows of that test
/// (`state(STATE_LOW_FLOOR)` and friends) seed the probe from the constant, so they are handled
/// self-referentially in `state_cases_held`.
const GATE_STATE_ABS_CASES: [(f64, RecoveryState); 6] = [
    (0.0, RecoveryState::Depleted),
    (24.9, RecoveryState::Depleted),
    (49.9, RecoveryState::Low),
    (69.9, RecoveryState::Moderate),
    (87.9, RecoveryState::Primed),
    (100.0, RecoveryState::Peak),
];
/// The five contiguous runs the floors must cut over the 0.1 grid, as LITERALS, so raising a floor is
/// as visible as lowering one.
const GATE_STATE_RUN_STARTS: [(RecoveryState, f64); 5] = [
    (RecoveryState::Depleted, 0.0),
    (RecoveryState::Low, 25.0),
    (RecoveryState::Moderate, 50.0),
    (RecoveryState::Primed, 70.0),
    (RecoveryState::Peak, 88.0),
];
/// Five real three-driver nights that must land in the five states, in order.
const GATE_STATE_REACHED: [(f64, f64, f64); 5] =
    [(26.0, 70.0, 0.40), (46.0, 58.0, 0.75), (50.0, 55.0, 0.85), (56.0, 52.0, 0.90), (62.0, 49.0, 0.95)];
/// Ten inclusivity probes, the partition and the reachability walk.
const GATE_STATE_CHECKS: usize = 12;
/// recovery.rs — nine injected rates, the tightened tolerance, and the per-hour convergence walk.
const GATE_SLOPE_TARGETS: [f64; 9] = [0.0, -0.5, -1.0, -2.0, -4.0, -8.0, 1.0, 2.0, 5.0];
const GATE_SLOPE_TOL: f64 = 0.05;
const GATE_SLOPE_HOURS: [i64; 4] = [2, 3, 6, 9];
const GATE_SLOPE_HOURS_TOL: f64 = 0.10;
/// recovery.rs — the banked-nights probe table, the shipped count and the widened-bounds count.
const GATE_BANKED_NIGHTS: usize = 3;
const GATE_BANKED_WIDE: usize = 5;
const GATE_BANKED_PROBES: [(f64, usize); 10] = [
    (4.9999, 0),
    (5.0, 1),
    (55.0, 1),
    (250.0, 1),
    (250.0001, 0),
    (f64::NAN, 0),
    (f64::INFINITY, 0),
    (-1.0, 0),
    (0.0, 0),
    (80.0, 1),
];
/// The ordered full-night row vector. RE-DERIVED 2026-08-04 when the per-driver swing became an exact
/// Shapley share instead of a leave-one-out marginal; the old vector was
/// `[(Hrv, 23), (RestingHr, 4), (Sleep, 1), (RecoveryIndex, 1), (ActivityBalance, -1), (Respiratory, 0),
/// (SkinTemp, 0)]`. Every value below was computed by an independent implementation, not read back
/// out of the code it gates.
const GATE_FULL_NIGHT_ROWS: [(DriverKind, f64); 7] = [
    (DriverKind::Hrv, 26.0),
    (DriverKind::RestingHr, 6.0),
    (DriverKind::ActivityBalance, -3.0),
    (DriverKind::RecoveryIndex, 2.0),
    (DriverKind::Sleep, 1.0),
    (DriverKind::Respiratory, 1.0),
    (DriverKind::SkinTemp, 0.0),
];
/// recovery_drivers.rs:347 — `assert_eq!(hrv.delta_points, 32.0);`
const GATE_HRV_DELTA_NO_RHR: f64 = 32.0;
/// recovery_drivers.rs:433-434 — `assert_eq!(hrv_delta(5000.0), 42.0);` / `(-5000.0), -58.0`.
const GATE_HRV_DELTA_SAT_HIGH: f64 = 42.0;
const GATE_HRV_DELTA_SAT_LOW: f64 = -58.0;
/// recovery_drivers.rs:443 — `assert_eq!(zero_spread[0].delta_points, 42.0);`
const GATE_ZERO_SPREAD_TOP_DELTA: f64 = 42.0;
/// baselines.rs:293-294 — `assert_eq!(s.baseline, 30.0);` / `n_valid == 1`.
const GATE_COLD_START_BASELINE: f64 = 30.0;
const GATE_COLD_START_N_VALID: i32 = 1;
/// baselines.rs:301-303 — `baseline == 30.0`, `n_valid == 5`, `nights_since_update == 1`.
const GATE_NULL_HOLD_BASELINE: f64 = 30.0;
const GATE_NULL_HOLD_N_VALID: i32 = 5;
const GATE_NULL_HOLD_SINCE: i32 = 1;
/// baselines.rs:311 — `assert_eq!(s.baseline, 50.0); // unchanged`.
const GATE_OUT_OF_RANGE_BASELINE: f64 = 50.0;
/// baselines.rs:378-379 — `assert_eq!(s.n_valid, 5);` / `nights_since_update == 1`.
const GATE_REJECTED_NIGHT_N_VALID: i32 = 5;
const GATE_REJECTED_NIGHT_SINCE: i32 = 1;
/// baselines.rs:319-320 — `assert!(s.baseline < 50.0, ...);` / `assert!(s.n_valid >= 14, ...)`.
/// baselines.rs:365-366 — `assert_eq!(old.n_valid, 12); assert_eq!(new.n_valid, 9);`
const GATE_PER_METRIC_N_VALID: i32 = 12;
const GATE_CONJUNCTION_N_VALID: i32 = 9;
/// baselines.rs:369-370 — three nights must read `SkinTempImplausible`.
const GATE_NIGHTSTAND_REJECTIONS: usize = 3;
/// baselines.rs:422-425 — `median_c == 33.4`, `max_c == 34.0`, `(n_kept, n_total) == (3, 4)`.
/// Shipped baseline constants the replica pins against. These are DELIBERATELY not read from
/// `BaseParams`: an expectation that follows the arm moves with the mutation and sees nothing.
const GATE_EARLY_ADAPT_NIGHTS: i32 = 8;
const GATE_EARLY_HALF_LIFE_B: f64 = 3.0;
const GATE_MIN_NIGHTS_TRUST: i32 = 14;
const GATE_STALE_DAYS: i32 = 14;
const GATE_WINSOR_K: f64 = 3.0;
/// Deviation 26 > 5 x 5, so this night is seen and never folded. Deviation 25 is not PAST it.
const GATE_OUTLIER_HELD: f64 = 71.0;
const GATE_OUTLIER_FOLDED: f64 = 70.0;
/// One 200 ms night on a young HRV centre of 45, from `a_young_baseline_has_no_outlier_guard`.
const GATE_YOUNG_OUTLIER_BASELINE: f64 = 52.736230276;

const GATE_SKIN_MEDIAN_C: f64 = 33.4;
const GATE_SKIN_MAX_C: f64 = 34.0;
const GATE_SKIN_KEPT_TOTAL: (usize, usize) = (3, 4);
/// illness.rs:53 — `assert_eq!(r, 13..43);`
const GATE_WINDOW_START: usize = 13;
const GATE_WINDOW_END: usize = 43;
/// illness.rs:76 — `assert!(z > 2.0, "gapped z was {z}");`
const GATE_ILLNESS_FIRE_Z: f64 = 2.0;
/// illness.rs:95 — `assert!((full - holey).abs() < 1.0);`
const GATE_HOLEY_TOL: f64 = 1.0;

// ─── table harness ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Param,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Baseline => "baseline",
            Kind::Null => "null",
            Kind::Structural => "structural",
            Kind::Param => "param",
        }
    }
}

struct Arm {
    kind: Kind,
    name: String,
    value: f64,
    extra: Option<f64>,
    pct: Option<f64>,
    pass: bool,
}

impl Arm {
    fn new(kind: Kind, name: String, value: f64, pass: bool) -> Self {
        Self { kind, name, value, extra: None, pct: None, pass }
    }
    fn x(mut self, v: f64) -> Self {
        self.extra = Some(v);
        self
    }
    fn p(mut self, v: f64) -> Self {
        self.pct = Some(v);
        self
    }
}

/// Per-metric tally: gate failures on a mutated arm (caught), gate passes (missed), the smallest caught
/// |delta| and the smallest caught parameter move.
struct Score {
    caught: usize,
    missed: usize,
    delta_floor: Option<f64>,
    pct_floor: Option<f64>,
}

struct Table {
    metric: &'static str,
    gate: &'static str,
    value_label: &'static str,
    extra_label: Option<&'static str>,
    arms: Vec<Arm>,
}

impl Table {
    fn new(metric: &'static str, gate: &'static str, value_label: &'static str) -> Self {
        Self { metric, gate, value_label, extra_label: None, arms: Vec::new() }
    }
    fn extra(mut self, label: &'static str) -> Self {
        self.extra_label = Some(label);
        self
    }
    fn add(&mut self, a: Arm) {
        self.arms.push(a);
    }

    /// Print the table, assert only what must hold for the harness to be trustworthy (the baseline
    /// reproduces, every NULL arm fails), and return the tally.
    fn finish(self) -> Score {
        println!("\n=== {} ===", self.metric);
        println!("gate: {}", self.gate);
        match self.extra_label {
            Some(l) => println!("{:<58}{:>11}{:>11}{:>13}   gate", "arm", self.value_label, "delta", l),
            None => println!("{:<58}{:>11}{:>11}   gate", "arm", self.value_label, "delta"),
        }
        let base = self.arms.first().map(|a| a.value).unwrap_or(f64::NAN);
        let (mut caught, mut missed) = (0usize, 0usize);
        let (mut delta_floor, mut pct_floor): (Option<f64>, Option<f64>) = (None, None);
        for (i, a) in self.arms.iter().enumerate() {
            let delta = a.value - base;
            let verdict = if i == 0 {
                "PASS (expected)".to_string()
            } else if a.pass {
                "PASS  <-- MISSED".to_string()
            } else {
                "FAIL  <-- caught".to_string()
            };
            let name = format!("{}: {}", a.kind.tag(), a.name);
            match (self.extra_label, a.extra) {
                (Some(_), Some(x)) => {
                    println!("{name:<58}{:>11.4}{:>+11.4}{x:>13.4}   {verdict}", a.value, delta)
                }
                (Some(_), None) => {
                    println!("{name:<58}{:>11.4}{:>+11.4}{:>13}   {verdict}", a.value, delta, "-")
                }
                _ => println!("{name:<58}{:>11.4}{:>+11.4}   {verdict}", a.value, delta),
            }
            if i == 0 {
                continue;
            }
            if a.pass {
                missed += 1;
            } else {
                caught += 1;
                if delta.is_finite() && delta != 0.0 {
                    delta_floor = Some(delta_floor.map_or(delta.abs(), |f: f64| f.min(delta.abs())));
                }
                if let Some(p) = a.pct {
                    pct_floor = Some(pct_floor.map_or(p, |f: f64| f.min(p)));
                }
            }
        }
        println!(
            "caught {caught}, missed {missed}   smallest caught |delta| = {}   smallest caught param move = {}",
            delta_floor.map_or("n/a".into(), |v| format!("{v:.4}")),
            pct_floor.map_or("n/a".into(), |v| format!("{:.1}%", v * 100.0)),
        );
        assert!(
            self.arms.first().map(|a| a.pass).unwrap_or(false),
            "{}: the baseline arm must reproduce the shipped figure",
            self.metric
        );
        for a in self.arms.iter().filter(|a| a.kind == Kind::Null) {
            assert!(
                !a.pass,
                "CRITICAL — {}: NULL arm '{}' PASSES the shipped gate, so the gate is fake",
                self.metric, a.name
            );
        }
        let probes: Vec<(&str, f64)> = self
            .arms
            .iter()
            .filter(|a| matches!(a.kind, Kind::Null | Kind::Structural))
            .map(|a| (a.name.as_str(), a.value))
            .collect();
        enforce_floors(self.metric, base, &probes);
        Score { caught, missed, delta_floor, pct_floor }
    }
}

const PM10: [(&str, f64); 2] = [("x1.10", 1.10), ("x0.90", 0.90)];
const TINY: [(&str, f64); 1] = [("+0.5%", 1.005)];

// ─── replica of the Charge composite (recovery.rs:105-231) ─────────────────────────────────────

/// The tunables `recovery::recovery` reads, plus two control knobs (`hrv_sign`, `drop_hrv`) that are
/// NOT shipped constants and exist only to build structural arms.
#[derive(Clone, Copy)]
struct ScoreParams {
    w_hrv: f64,
    w_rhr: f64,
    w_resp: f64,
    w_sleep: f64,
    w_skin_temp: f64,
    w_recovery_index: f64,
    w_activity_balance: f64,
    skin_temp_dev_scale: f64,
    recovery_index_scale: f64,
    logistic_k: f64,
    logistic_z0: f64,
    sleep_perf_center: f64,
    sleep_perf_scale: f64,
    hrv_sign: f64,
    drop_hrv: bool,
}

impl ScoreParams {
    fn shipped() -> Self {
        Self {
            w_hrv: W_HRV,
            w_rhr: W_RHR,
            w_resp: W_RESP,
            w_sleep: W_SLEEP,
            w_skin_temp: W_SKIN_TEMP,
            w_recovery_index: W_RECOVERY_INDEX,
            w_activity_balance: W_ACTIVITY_BALANCE,
            skin_temp_dev_scale: SKIN_TEMP_DEV_SCALE,
            recovery_index_scale: RECOVERY_INDEX_SCALE_BPM_PER_HR,
            logistic_k: LOGISTIC_K,
            logistic_z0: LOGISTIC_Z0,
            sleep_perf_center: SLEEP_PERF_CENTER,
            sleep_perf_scale: SLEEP_PERF_SCALE,
            hrv_sign: 1.0,
            drop_hrv: false,
        }
    }
}

fn z_score_rep(value: f64, mean: f64, spread: f64) -> f64 {
    (value - mean) / (1.253 * spread).max(1e-9)
}

fn score_of_with(p: &ScoreParams, z: f64) -> f64 {
    (100.0 / (1.0 + (-p.logistic_k * (z - p.logistic_z0)).exp())).clamp(0.0, 100.0)
}

fn terms_with(p: &ScoreParams, i: &RecoveryInput) -> Vec<(DriverKind, f64, f64)> {
    let mut t: Vec<(DriverKind, f64, f64)> = Vec::new();
    if let (false, Some(b)) = (p.drop_hrv, i.hrv_baseline) {
        t.push((DriverKind::Hrv, p.hrv_sign * z_score_rep(i.hrv, b.mean, b.spread), p.w_hrv));
    }
    if let Some(b) = i.rhr_baseline {
        t.push((DriverKind::RestingHr, z_score_rep(b.mean, i.rhr, b.spread), p.w_rhr));
    }
    if let (Some(r), Some(b)) = (i.resp, i.resp_baseline) {
        t.push((DriverKind::Respiratory, z_score_rep(b.mean, r, b.spread), p.w_resp));
    }
    if let Some(sp) = i.sleep_perf {
        t.push((DriverKind::Sleep, (sp - p.sleep_perf_center) / p.sleep_perf_scale, p.w_sleep));
    }
    if let Some(d) = i.skin_temp_dev {
        t.push((DriverKind::SkinTemp, -d.abs() / p.skin_temp_dev_scale, p.w_skin_temp));
    }
    if let Some(s) = i.recovery_index_slope {
        t.push((DriverKind::RecoveryIndex, -s / p.recovery_index_scale, p.w_recovery_index));
    }
    if let (Some(e), Some(b)) = (i.prior_day_effort, i.effort_baseline) {
        t.push((DriverKind::ActivityBalance, z_score_rep(b.mean, e, b.spread), p.w_activity_balance));
    }
    t
}

fn score_with(p: &ScoreParams, i: &RecoveryInput) -> Option<f64> {
    if !i.hrv_baseline_usable {
        return None;
    }
    let terms = terms_with(p, i);
    if terms.is_empty() {
        return None;
    }
    let total: f64 = terms.iter().map(|t| t.2).sum();
    if total <= 0.0 {
        return None;
    }
    Some(score_of_with(p, terms.iter().map(|t| t.1 * t.2).sum::<f64>() / total))
}

// ─── replica of the driver breakdown (recovery_drivers.rs:78-209) ──────────────────────────────

const ROW_ORDER: [DriverKind; 7] = [
    DriverKind::Hrv,
    DriverKind::RestingHr,
    DriverKind::Sleep,
    DriverKind::Respiratory,
    DriverKind::SkinTemp,
    DriverKind::RecoveryIndex,
    DriverKind::ActivityBalance,
];

/// `SKIN_TEMP_TYPICAL_BAND_C` plus four control knobs that are not shipped constants.
#[derive(Clone, Copy)]
struct DriverKnobs {
    band_c: f64,
    reverse_sort: bool,
    zero_deltas: bool,
    uniform_delta: bool,
    trunc_round: bool,
}

impl DriverKnobs {
    fn shipped() -> Self {
        Self {
            band_c: SKIN_TEMP_TYPICAL_BAND_C,
            reverse_sort: false,
            zero_deltas: false,
            uniform_delta: false,
            trunc_round: false,
        }
    }
}

fn round_half_up(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

fn driver_rows_with(
    p: &ScoreParams, k: &DriverKnobs, i: &RecoveryInput,
) -> Vec<(DriverKind, f64, DriverVerdict)> {
    if !i.hrv_baseline_usable {
        return Vec::new();
    }
    let terms = terms_with(p, i);
    let total: f64 = terms.iter().map(|t| t.2).sum();
    if terms.is_empty() || total <= 0.0 {
        return Vec::new();
    }
    let round = |x: f64| if k.trunc_round { x.trunc() } else { round_half_up(x) };
    let actual = score_of_with(p, terms.iter().map(|t| t.1 * t.2).sum::<f64>() / total);
    // Exact Shapley shares, the replica of recovery_drivers::shapley_points.
    let n = terms.len();
    let value: Vec<f64> = (0..(1usize << n))
        .map(|mask| {
            let z: f64 = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| mask >> i & 1 == 1)
                .map(|(_, t)| t.1 * t.2)
                .sum();
            score_of_with(p, z / total)
        })
        .collect();
    let mut factorial = [1.0f64; 8];
    for i in 1..=n {
        factorial[i] = factorial[i - 1] * i as f64;
    }
    let delta = |idx: usize| -> f64 {
        if k.zero_deltas {
            return 0.0;
        }
        if k.uniform_delta {
            return round(actual - score_of_with(p, 0.0));
        }
        let bit = 1usize << idx;
        let share: f64 = (0..(1usize << n))
            .filter(|mask| mask & bit == 0)
            .map(|mask| {
                let size = mask.count_ones() as usize;
                factorial[size] * factorial[n - size - 1] / factorial[n]
                    * (value[mask | bit] - value[mask])
            })
            .sum();
        round(share)
    };
    let mut rows: Vec<(DriverKind, f64, DriverVerdict)> = Vec::new();
    for kind in ROW_ORDER {
        let Some(idx) = terms.iter().position(|t| t.0 == kind) else { continue };
        let verdict = match kind {
            DriverKind::SkinTemp => {
                let d = i.skin_temp_dev.unwrap_or(0.0);
                if d.abs() <= k.band_c {
                    DriverVerdict::Neutral
                } else if d > 0.0 {
                    DriverVerdict::LimitingHigh
                } else {
                    DriverVerdict::LimitingLow
                }
            }
            _ => {
                let z = terms[idx].1;
                if z > 0.0 {
                    DriverVerdict::Supporting
                } else if z < 0.0 {
                    DriverVerdict::Limiting
                } else {
                    DriverVerdict::Neutral
                }
            }
        };
        rows.push((kind, delta(idx), verdict));
    }
    rows.sort_by(|a, b| {
        let o = b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal);
        if k.reverse_sort {
            o.reverse()
        } else {
            o
        }
    });
    rows
}

// ─── shared fixtures, copied from the shipped tests ────────────────────────────────────────────

fn dbase(mean: f64, sigma: f64) -> DriverBaseline {
    DriverBaseline { mean, spread: sigma / 1.253 }
}

/// The neutral night behind the z = 0 anchor. `sleep_perf` is seeded from the
/// constant in the shipped test, so the arm moves it too.
fn neutral_input(p: &ScoreParams) -> RecoveryInput {
    RecoveryInput {
        hrv: 50.0,
        rhr: 55.0,
        hrv_baseline: Some(dbase(50.0, 6.0)),
        rhr_baseline: Some(dbase(55.0, 3.0)),
        sleep_perf: Some(p.sleep_perf_center),
        ..Default::default()
    }
}

/// recovery_drivers.rs:221-236 — every term present.
fn full_night() -> RecoveryInput {
    RecoveryInput {
        hrv: 62.0,
        rhr: 51.0,
        resp: Some(15.0),
        hrv_baseline: Some(dbase(50.0, 6.0)),
        rhr_baseline: Some(dbase(55.0, 3.0)),
        resp_baseline: Some(dbase(16.0, 2.0)),
        sleep_perf: Some(0.9),
        skin_temp_dev: Some(0.4),
        hrv_baseline_usable: true,
        recovery_index_slope: Some(-3.0),
        effort_baseline: Some(dbase(40.0, 15.0)),
        prior_day_effort: Some(75.0),
    }
}

/// recovery_drivers.rs:238-248 — the two-term input the saturation and verdict probes mutate.
fn two_term(mutate: impl FnOnce(&mut RecoveryInput)) -> RecoveryInput {
    let mut i = RecoveryInput {
        hrv: 50.0,
        rhr: 55.0,
        hrv_baseline: Some(dbase(50.0, 6.0)),
        rhr_baseline: Some(dbase(55.0, 3.0)),
        ..Default::default()
    };
    mutate(&mut i);
    i
}

// ─── metric 1: Charge score ────────────────────────────────────────────────────────────────────

/// One row of the pinned spread fixture: HRV, resting HR and sleep performance on fixed baselines.
fn spread_input(hrv: f64, rhr: f64, sleep_perf: f64) -> RecoveryInput {
    RecoveryInput {
        hrv,
        rhr,
        hrv_baseline: Some(dbase(50.0, 6.0)),
        rhr_baseline: Some(dbase(55.0, 3.0)),
        sleep_perf: Some(sleep_perf),
        ..Default::default()
    }
}

/// The all-seven-driver night on the limiting side. Its supporting twin is `full_night`.
fn limiting_night() -> RecoveryInput {
    RecoveryInput {
        hrv: 38.0,
        rhr: 62.0,
        resp: Some(18.0),
        hrv_baseline: Some(dbase(50.0, 6.0)),
        rhr_baseline: Some(dbase(55.0, 3.0)),
        resp_baseline: Some(dbase(16.0, 2.0)),
        sleep_perf: Some(0.62),
        skin_temp_dev: Some(0.9),
        hrv_baseline_usable: true,
        recovery_index_slope: Some(1.5),
        effort_baseline: Some(dbase(40.0, 15.0)),
        prior_day_effort: Some(88.0),
    }
}

/// Every one of the seven drivers exactly at its own baseline. `sleep_perf` is seeded from the
/// constant in the shipped test, so the arm moves it too.
fn neutral_seven(p: &ScoreParams) -> RecoveryInput {
    RecoveryInput {
        resp: Some(16.0),
        sleep_perf: Some(p.sleep_perf_center),
        skin_temp_dev: Some(0.0),
        recovery_index_slope: Some(0.0),
        effort_baseline: Some(dbase(40.0, 15.0)),
        prior_day_effort: Some(40.0),
        resp_baseline: Some(dbase(16.0, 2.0)),
        ..neutral_input(p)
    }
}

/// Within tolerance, and never within it when either side is NaN — a missing score must fail a gate,
/// not slip past a negated comparison.
fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// The shipped gate: the z = 0 anchor, every pinned row, monotonicity, the span, and the two
/// all-seven-driver nights with their shared-anchor claim.
fn score_gate_holds(p: &ScoreParams) -> bool {
    let anchor = score_with(p, &neutral_input(p)).unwrap_or(f64::NAN);
    if !near(anchor, GATE_ANCHOR_TARGET, GATE_ANCHOR_TOL) {
        return false;
    }
    let got: Vec<f64> = GATE_SPREAD_NIGHTS
        .iter()
        .map(|&(h, r, s, _)| score_with(p, &spread_input(h, r, s)).unwrap_or(f64::NAN))
        .collect();
    if !got.iter().zip(GATE_SPREAD_NIGHTS).all(|(g, (.., w))| near(*g, w, GATE_SPREAD_TOL)) {
        return false;
    }
    let span = got[got.len() - 1] - got[0];
    if !got.windows(2).all(|w| w[1] > w[0]) || span <= GATE_SPREAD_MIN_SPAN || span.is_nan() {
        return false;
    }
    let seven = [
        (score_with(p, &full_night()), GATE_SEVEN_SUPPORTING),
        (score_with(p, &limiting_night()), GATE_SEVEN_LIMITING),
        (score_with(p, &neutral_seven(p)), anchor),
    ];
    seven.iter().all(|&(g, w)| near(g.unwrap_or(f64::NAN), w, GATE_SPREAD_TOL))
}

/// The same gate as a constant scorer meets it: one value cannot reproduce eleven distinct nights.
fn const_scorer_gate_holds(v: f64) -> bool {
    near(v, GATE_ANCHOR_TARGET, GATE_ANCHOR_TOL)
        && GATE_SPREAD_NIGHTS.iter().all(|&(.., w)| near(v, w, GATE_SPREAD_TOL))
        && near(v, GATE_SEVEN_SUPPORTING, GATE_SPREAD_TOL)
        && near(v, GATE_SEVEN_LIMITING, GATE_SPREAD_TOL)
}

fn score_arm(kind: Kind, name: String, p: &ScoreParams) -> Arm {
    let anchor = score_with(p, &neutral_input(p)).unwrap_or(f64::NAN);
    let real = score_with(p, &full_night()).unwrap_or(f64::NAN);
    Arm::new(kind, name, anchor, score_gate_holds(p)).x(real)
}

fn score_param(
    t: &mut Table, label: &str, base: f64, muls: &[(&str, f64)], set: impl Fn(&mut ScoreParams, f64),
) {
    for &(tag, m) in muls {
        let mut p = ScoreParams::shipped();
        let v = base * m;
        set(&mut p, v);
        t.add(score_arm(Kind::Param, format!("{label} {base} -> {v:.4} ({tag})"), &p).p((m - 1.0).abs()));
    }
}

fn metric_charge_score() -> Score {
    for i in [neutral_input(&ScoreParams::shipped()), full_night(), two_term(|x| x.hrv = 71.0)] {
        let (a, b) = (score_with(&ScoreParams::shipped(), &i).unwrap(), recovery(&i).unwrap());
        assert!((a - b).abs() < 1e-12, "score replica {a} != shipped {b}");
    }
    assert!((score_of_with(&ScoreParams::shipped(), 0.37) - score_of(0.37)).abs() < 1e-12);

    let mut t = Table::new(
        "Recovery / Charge score (0-100 logistic composite)",
        "recovery.rs  the anchor + nine pinned nights + monotone + span > 99",
        "anchor",
    )
    .extra("full-night");
    t.add(score_arm(Kind::Baseline, "unmutated".into(), &ScoreParams::shipped()));

    t.add(
        Arm::new(
            Kind::Null,
            "constant scorer, every night 50.0".into(),
            50.0,
            const_scorer_gate_holds(50.0),
        )
        .x(50.0),
    );
    // The scorer the pre-repair gate could not see: right on the anchor, flat everywhere else.
    t.add(
        Arm::new(
            Kind::Null,
            "constant scorer, every night the z = 0 anchor".into(),
            GATE_ANCHOR_EXACT,
            const_scorer_gate_holds(GATE_ANCHOR_EXACT),
        )
        .x(GATE_ANCHOR_EXACT),
    );
    let mut p = ScoreParams::shipped();
    p.logistic_k = 0.0;
    t.add(score_arm(Kind::Null, "logistic slope k = 0, every z maps to 50".into(), &p));

    let mut p = ScoreParams::shipped();
    p.w_hrv = W_SKIN_TEMP;
    p.w_skin_temp = W_HRV;
    t.add(score_arm(Kind::Structural, "W_HRV and W_SKIN_TEMP swapped (0.55 <-> 0.05)".into(), &p));
    let mut p = ScoreParams::shipped();
    p.hrv_sign = -1.0;
    t.add(score_arm(Kind::Structural, "HRV z sign flipped, higher HRV reads worse".into(), &p));
    let mut p = ScoreParams::shipped();
    p.drop_hrv = true;
    t.add(score_arm(Kind::Structural, "HRV term dropped entirely".into(), &p));
    let mut p = ScoreParams::shipped();
    p.w_hrv = 1.0;
    p.w_rhr = 1.0;
    p.w_resp = 1.0;
    p.w_sleep = 1.0;
    p.w_skin_temp = 1.0;
    p.w_recovery_index = 1.0;
    p.w_activity_balance = 1.0;
    t.add(score_arm(Kind::Structural, "every driver weighted equally".into(), &p));

    score_param(&mut t, "W_HRV", W_HRV, &PM10, |p, v| p.w_hrv = v);
    score_param(&mut t, "W_RHR", W_RHR, &PM10, |p, v| p.w_rhr = v);
    score_param(&mut t, "W_RESP", W_RESP, &PM10, |p, v| p.w_resp = v);
    score_param(&mut t, "W_SLEEP", W_SLEEP, &PM10, |p, v| p.w_sleep = v);
    score_param(&mut t, "W_SKIN_TEMP", W_SKIN_TEMP, &PM10, |p, v| p.w_skin_temp = v);
    score_param(&mut t, "W_RECOVERY_INDEX", W_RECOVERY_INDEX, &PM10, |p, v| p.w_recovery_index = v);
    score_param(&mut t, "W_ACTIVITY_BALANCE", W_ACTIVITY_BALANCE, &PM10, |p, v| {
        p.w_activity_balance = v
    });
    score_param(&mut t, "SKIN_TEMP_DEV_SCALE", SKIN_TEMP_DEV_SCALE, &PM10, |p, v| {
        p.skin_temp_dev_scale = v
    });
    score_param(&mut t, "RECOVERY_INDEX_SCALE_BPM_PER_HR", RECOVERY_INDEX_SCALE_BPM_PER_HR, &PM10, |p, v| {
        p.recovery_index_scale = v
    });
    score_param(&mut t, "SLEEP_PERF_CENTER", SLEEP_PERF_CENTER, &PM10, |p, v| p.sleep_perf_center = v);
    score_param(&mut t, "SLEEP_PERF_SCALE", SLEEP_PERF_SCALE, &PM10, |p, v| p.sleep_perf_scale = v);
    score_param(&mut t, "LOGISTIC_K", LOGISTIC_K, &PM10, |p, v| p.logistic_k = v);
    score_param(&mut t, "LOGISTIC_Z0", LOGISTIC_Z0, &PM10, |p, v| p.logistic_z0 = v);
    score_param(&mut t, "LOGISTIC_K", LOGISTIC_K, &TINY, |p, v| p.logistic_k = v);

    let s = t.finish();
    let mut p = ScoreParams::shipped();
    p.logistic_k = LOGISTIC_K * 1.10;
    let out = score_with(&p, &neutral_input(&p)).unwrap();
    let expected = 100.0 / (1.0 + (-p.logistic_k * (0.0 - p.logistic_z0)).exp());
    println!(
        "note: the gate also recomputes `expected` from LOGISTIC_K/LOGISTIC_Z0, so at k x1.10 THAT row \
         still holds (|out - expected| = {:.2e} < 1e-9). On its own it pins nothing; the pinned nights do.",
        (out - expected).abs()
    );
    println!(
        "note: at z = 0 every driver sits on its own baseline, so the anchor alone is a function of \
         LOGISTIC_K and LOGISTIC_Z0 and no weight or scale can move it. Each weight and scale is caught \
         through the pinned nights instead: the nine three-driver rows carry HRV, resting HR and sleep, \
         and the two all-seven rows carry the other four."
    );
    s
}

// ─── metric 2: recovery band and Charge state ──────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Bands {
    red_max: f64,
    yellow_max: f64,
    swap: bool,
    force: Option<&'static str>,
}

impl Bands {
    fn shipped() -> Self {
        Self { red_max: BAND_RED_MAX, yellow_max: BAND_YELLOW_MAX, swap: false, force: None }
    }
}

fn band_of(b: &Bands, s: f64) -> &'static str {
    if let Some(x) = b.force {
        return x;
    }
    let raw = if s < b.red_max {
        "red"
    } else if s < b.yellow_max {
        "yellow"
    } else {
        "green"
    };
    match (b.swap, raw) {
        (true, "red") => "green",
        (true, "green") => "red",
        _ => raw,
    }
}

fn band_cases_held(b: &Bands) -> usize {
    GATE_BAND_CASES.iter().filter(|(s, want)| band_of(b, *s) == *want).count()
}

fn band_arm(kind: Kind, name: String, b: &Bands) -> Arm {
    let held = band_cases_held(b);
    Arm::new(kind, name, held as f64, held == GATE_BAND_CASES.len())
}

#[derive(Clone, Copy)]
struct Floors {
    lo: f64,
    mo: f64,
    pr: f64,
    pk: f64,
    swap: bool,
    force: Option<RecoveryState>,
}

impl Floors {
    fn shipped() -> Self {
        Self {
            lo: STATE_LOW_FLOOR,
            mo: STATE_MODERATE_FLOOR,
            pr: STATE_PRIMED_FLOOR,
            pk: STATE_PEAK_FLOOR,
            swap: false,
            force: None,
        }
    }
}

fn state_of(f: &Floors, s: f64) -> RecoveryState {
    if let Some(x) = f.force {
        return x;
    }
    let raw = if s < f.lo {
        RecoveryState::Depleted
    } else if s < f.mo {
        RecoveryState::Low
    } else if s < f.pr {
        RecoveryState::Moderate
    } else if s < f.pk {
        RecoveryState::Primed
    } else {
        RecoveryState::Peak
    };
    match (f.swap, raw) {
        (true, RecoveryState::Low) => RecoveryState::Primed,
        (true, RecoveryState::Primed) => RecoveryState::Low,
        _ => raw,
    }
}

/// The twelve checks the state gate makes. Ten are inclusivity probes, four of which seed their probe
/// from the constant they pin — that self-reference is the gate as shipped. The eleventh is the
/// literal partition and the twelfth walks five real Charge scores into the five states.
fn state_cases_held(f: &Floors) -> usize {
    let abs = GATE_STATE_ABS_CASES.iter().filter(|(s, want)| state_of(f, *s) == *want).count();
    let selfref = [
        (f.lo, RecoveryState::Low),
        (f.mo, RecoveryState::Moderate),
        (f.pr, RecoveryState::Primed),
        (f.pk, RecoveryState::Peak),
    ]
    .iter()
    .filter(|(s, want)| state_of(f, *s) == *want)
    .count();

    let mut runs: Vec<(RecoveryState, f64)> = Vec::new();
    for i in 0..=1000 {
        let s = i as f64 / 10.0;
        if runs.last().map(|r| r.0) != Some(state_of(f, s)) {
            runs.push((state_of(f, s), s));
        }
    }
    let partition = usize::from(runs == GATE_STATE_RUN_STARTS);

    let reached: Vec<RecoveryState> = GATE_STATE_REACHED
        .iter()
        .map(|&(h, r, s)| state_of(f, recovery(&spread_input(h, r, s)).unwrap()))
        .collect();
    let wanted: Vec<RecoveryState> = GATE_STATE_RUN_STARTS.iter().map(|r| r.0).collect();
    abs + selfref + partition + usize::from(reached == wanted)
}

fn state_arm(kind: Kind, name: String, f: &Floors) -> Arm {
    let held = state_cases_held(f);
    Arm::new(kind, name, held as f64, held == GATE_STATE_CHECKS)
}

fn metric_band_and_state() -> (Score, Score) {
    for s in [0.0, 12.5, 33.9, 34.0, 50.0, 66.9, 67.0, 88.0, 100.0] {
        assert_eq!(band_of(&Bands::shipped(), s), band(s), "band replica differs at {s}");
        assert_eq!(state_of(&Floors::shipped(), s), state(s), "state replica differs at {s}");
    }

    let mut t = Table::new(
        "Recovery band (red / yellow / green)",
        "recovery.rs:437-442  all six band cases hold",
        "cases/6",
    );
    t.add(band_arm(Kind::Baseline, "unmutated".into(), &Bands::shipped()));
    t.add(band_arm(
        Kind::Null,
        "constant band, always yellow".into(),
        &Bands { force: Some("yellow"), ..Bands::shipped() },
    ));
    t.add(band_arm(
        Kind::Null,
        "constant band, always red".into(),
        &Bands { force: Some("red"), ..Bands::shipped() },
    ));
    t.add(band_arm(
        Kind::Structural,
        "red and green labels swapped".into(),
        &Bands { swap: true, ..Bands::shipped() },
    ));
    t.add(band_arm(
        Kind::Structural,
        "both thresholds shifted +1.0 point".into(),
        &Bands { red_max: BAND_RED_MAX + 1.0, yellow_max: BAND_YELLOW_MAX + 1.0, ..Bands::shipped() },
    ));
    for &(tag, m) in PM10.iter().chain(TINY.iter()) {
        let v = BAND_RED_MAX * m;
        t.add(
            band_arm(
                Kind::Param,
                format!("BAND_RED_MAX {BAND_RED_MAX} -> {v:.4} ({tag})"),
                &Bands { red_max: v, ..Bands::shipped() },
            )
            .p((m - 1.0).abs()),
        );
        let v = BAND_YELLOW_MAX * m;
        t.add(
            band_arm(
                Kind::Param,
                format!("BAND_YELLOW_MAX {BAND_YELLOW_MAX} -> {v:.4} ({tag})"),
                &Bands { yellow_max: v, ..Bands::shipped() },
            )
            .p((m - 1.0).abs()),
        );
    }
    let band_score = t.finish();

    let mut t = Table::new(
        "Charge state (5-way floors)",
        "recovery.rs  ten inclusivity rows + the literal partition + five states reached",
        "cases/12",
    );
    t.add(state_arm(Kind::Baseline, "unmutated".into(), &Floors::shipped()));
    t.add(state_arm(
        Kind::Null,
        "constant state, always Moderate".into(),
        &Floors { force: Some(RecoveryState::Moderate), ..Floors::shipped() },
    ));
    t.add(state_arm(
        Kind::Structural,
        "Low and Primed labels swapped".into(),
        &Floors { swap: true, ..Floors::shipped() },
    ));
    t.add(state_arm(
        Kind::Structural,
        "every floor shifted +1.0 point".into(),
        &Floors {
            lo: STATE_LOW_FLOOR + 1.0,
            mo: STATE_MODERATE_FLOOR + 1.0,
            pr: STATE_PRIMED_FLOOR + 1.0,
            pk: STATE_PEAK_FLOOR + 1.0,
            ..Floors::shipped()
        },
    ));
    for &(tag, m) in PM10.iter() {
        for (label, base, set) in [
            ("STATE_LOW_FLOOR", STATE_LOW_FLOOR, 0usize),
            ("STATE_MODERATE_FLOOR", STATE_MODERATE_FLOOR, 1),
            ("STATE_PRIMED_FLOOR", STATE_PRIMED_FLOOR, 2),
            ("STATE_PEAK_FLOOR", STATE_PEAK_FLOOR, 3),
        ] {
            let v = base * m;
            let mut f = Floors::shipped();
            match set {
                0 => f.lo = v,
                1 => f.mo = v,
                2 => f.pr = v,
                _ => f.pk = v,
            }
            t.add(
                state_arm(Kind::Param, format!("{label} {base} -> {v:.4} ({tag})"), &f)
                    .p((m - 1.0).abs()),
            );
        }
    }
    let v = STATE_LOW_FLOOR * 1.005;
    t.add(
        state_arm(
            Kind::Param,
            format!("STATE_LOW_FLOOR {STATE_LOW_FLOOR} -> {v:.4} (+0.5%)"),
            &Floors { lo: v, ..Floors::shipped() },
        )
        .p(0.005),
    );
    let state_score = t.finish();
    println!(
        "note: the state gate is TWO-SIDED since the partition walk landed. Its five run starts are \
         literals (0, 25, 50, 70, 88), so raising a floor now fails as loudly as lowering one, and the \
         reachability walk requires five real Charge scores to land in five different states."
    );
    (band_score, state_score)
}

// ─── metric 3: recovery driver rows ────────────────────────────────────────────────────────────

fn hrv_delta(p: &ScoreParams, k: &DriverKnobs, hrv: f64) -> Option<f64> {
    driver_rows_with(p, k, &two_term(|i| i.hrv = hrv))
        .iter()
        .find(|r| r.0 == DriverKind::Hrv)
        .map(|r| r.1)
}

fn skin_verdict(p: &ScoreParams, k: &DriverKnobs, dev: f64) -> Option<DriverVerdict> {
    driver_rows_with(p, k, &two_term(|i| i.skin_temp_dev = Some(dev)))
        .iter()
        .find(|r| r.0 == DriverKind::SkinTemp)
        .map(|r| r.2)
}

fn driver_gate_holds(p: &ScoreParams, k: &DriverKnobs) -> bool {
    let rows = driver_rows_with(p, k, &full_night());
    let got: Vec<(DriverKind, f64)> = rows.iter().map(|r| (r.0, r.1)).collect();
    if got.as_slice() != GATE_FULL_NIGHT_ROWS.as_slice() {
        return false;
    }
    let mut no_rhr = full_night();
    no_rhr.rhr_baseline = None;
    let rows = driver_rows_with(p, k, &no_rhr);
    if rows.iter().any(|r| r.0 == DriverKind::RestingHr) {
        return false;
    }
    if rows.iter().find(|r| r.0 == DriverKind::Hrv).map(|r| r.1) != Some(GATE_HRV_DELTA_NO_RHR) {
        return false;
    }
    if hrv_delta(p, k, 5000.0) != Some(GATE_HRV_DELTA_SAT_HIGH) {
        return false;
    }
    if hrv_delta(p, k, -5000.0) != Some(GATE_HRV_DELTA_SAT_LOW) {
        return false;
    }
    let zero_spread = driver_rows_with(
        p,
        k,
        &RecoveryInput {
            hrv: 60.0,
            rhr: 55.0,
            hrv_baseline: Some(DriverBaseline { mean: 50.0, spread: 0.0 }),
            rhr_baseline: Some(dbase(55.0, 3.0)),
            ..Default::default()
        },
    );
    if zero_spread.first().map(|r| r.1) != Some(GATE_ZERO_SPREAD_TOP_DELTA) {
        return false;
    }
    // The thirteen band cases, with the 0.3 literals beside the two constant-seeded edges, and the
    // claim that no single-sided driver ever carries a side.
    let cases: [(f64, DriverVerdict); 13] = [
        (-1.0, DriverVerdict::LimitingLow),
        (-0.31, DriverVerdict::LimitingLow),
        (-0.30000001, DriverVerdict::LimitingLow),
        (-0.3, DriverVerdict::Neutral),
        (-k.band_c, DriverVerdict::Neutral),
        (-0.1, DriverVerdict::Neutral),
        (0.0, DriverVerdict::Neutral),
        (0.1, DriverVerdict::Neutral),
        (k.band_c, DriverVerdict::Neutral),
        (0.3, DriverVerdict::Neutral),
        (0.30000001, DriverVerdict::LimitingHigh),
        (0.31, DriverVerdict::LimitingHigh),
        (1.0, DriverVerdict::LimitingHigh),
    ];
    if cases.iter().any(|&(dev, want)| skin_verdict(p, k, dev) != Some(want)) {
        return false;
    }
    driver_rows_with(p, k, &full_night()).iter().filter(|r| r.0 != DriverKind::SkinTemp).all(|r| {
        matches!(
            r.2,
            DriverVerdict::Supporting | DriverVerdict::Neutral | DriverVerdict::Limiting
        )
    })
}

fn driver_arm(kind: Kind, name: String, p: &ScoreParams, k: &DriverKnobs) -> Arm {
    let value = driver_rows_with(p, k, &full_night())
        .iter()
        .find(|r| r.0 == DriverKind::Hrv)
        .map(|r| r.1)
        .unwrap_or(f64::NAN);
    Arm::new(kind, name, value, driver_gate_holds(p, k))
}

fn driver_param(t: &mut Table, label: &str, base: f64, set: impl Fn(&mut ScoreParams, f64)) {
    for &(tag, m) in PM10.iter() {
        let mut p = ScoreParams::shipped();
        let v = base * m;
        set(&mut p, v);
        t.add(
            driver_arm(Kind::Param, format!("{label} {base} -> {v:.4} ({tag})"), &p, &DriverKnobs::shipped())
                .p((m - 1.0).abs()),
        );
    }
}

fn metric_driver_rows() -> Score {
    let want = driver_rows(&full_night());
    let got = driver_rows_with(&ScoreParams::shipped(), &DriverKnobs::shipped(), &full_night());
    assert_eq!(want.len(), got.len(), "driver replica row count differs");
    for (a, b) in want.iter().zip(&got) {
        assert_eq!(a.kind, b.0, "driver replica kind differs");
        assert_eq!(a.delta_points, b.1, "driver replica delta differs");
        assert_eq!(a.verdict, b.2, "driver replica verdict differs");
    }

    let mut t = Table::new(
        "Recovery driver rows (per-driver Shapley share of the swing)",
        "recovery_drivers.rs:251-266 full-night vector + :347 = 32.0 + :433-434 = 42.0/-58.0 + :443 = 42.0 \
         + skin-temp verdicts",
        "hrv pts",
    );
    let sp = ScoreParams::shipped();
    t.add(driver_arm(Kind::Baseline, "unmutated".into(), &sp, &DriverKnobs::shipped()));
    t.add(driver_arm(
        Kind::Null,
        "every row reports 0 points".into(),
        &sp,
        &DriverKnobs { zero_deltas: true, ..DriverKnobs::shipped() },
    ));
    t.add(driver_arm(
        Kind::Null,
        "every row reports the whole composite swing".into(),
        &sp,
        &DriverKnobs { uniform_delta: true, ..DriverKnobs::shipped() },
    ));
    t.add(driver_arm(
        Kind::Structural,
        "sort reversed, smallest mover first".into(),
        &sp,
        &DriverKnobs { reverse_sort: true, ..DriverKnobs::shipped() },
    ));
    t.add(driver_arm(
        Kind::Structural,
        "round-half-up replaced by truncation".into(),
        &sp,
        &DriverKnobs { trunc_round: true, ..DriverKnobs::shipped() },
    ));
    let mut swapped = ScoreParams::shipped();
    swapped.w_hrv = W_RHR;
    swapped.w_rhr = W_HRV;
    t.add(driver_arm(
        Kind::Structural,
        "W_HRV and W_RHR swapped (0.55 <-> 0.20)".into(),
        &swapped,
        &DriverKnobs::shipped(),
    ));
    for &(tag, m) in PM10.iter().chain(TINY.iter()) {
        let v = SKIN_TEMP_TYPICAL_BAND_C * m;
        t.add(
            driver_arm(
                Kind::Param,
                format!("SKIN_TEMP_TYPICAL_BAND_C {SKIN_TEMP_TYPICAL_BAND_C} -> {v:.4} ({tag})"),
                &sp,
                &DriverKnobs { band_c: v, ..DriverKnobs::shipped() },
            )
            .p((m - 1.0).abs()),
        );
    }
    driver_param(&mut t, "W_HRV", W_HRV, |p, v| p.w_hrv = v);
    driver_param(&mut t, "W_RHR", W_RHR, |p, v| p.w_rhr = v);
    driver_param(&mut t, "W_RESP", W_RESP, |p, v| p.w_resp = v);
    driver_param(&mut t, "W_SLEEP", W_SLEEP, |p, v| p.w_sleep = v);
    driver_param(&mut t, "W_SKIN_TEMP", W_SKIN_TEMP, |p, v| p.w_skin_temp = v);
    driver_param(&mut t, "W_RECOVERY_INDEX", W_RECOVERY_INDEX, |p, v| p.w_recovery_index = v);
    driver_param(&mut t, "W_ACTIVITY_BALANCE", W_ACTIVITY_BALANCE, |p, v| p.w_activity_balance = v);
    driver_param(&mut t, "SKIN_TEMP_DEV_SCALE", SKIN_TEMP_DEV_SCALE, |p, v| p.skin_temp_dev_scale = v);
    driver_param(&mut t, "RECOVERY_INDEX_SCALE_BPM_PER_HR", RECOVERY_INDEX_SCALE_BPM_PER_HR, |p, v| {
        p.recovery_index_scale = v
    });
    driver_param(&mut t, "SLEEP_PERF_CENTER", SLEEP_PERF_CENTER, |p, v| p.sleep_perf_center = v);
    driver_param(&mut t, "SLEEP_PERF_SCALE", SLEEP_PERF_SCALE, |p, v| p.sleep_perf_scale = v);
    driver_param(&mut t, "LOGISTIC_K", LOGISTIC_K, |p, v| p.logistic_k = v);
    driver_param(&mut t, "LOGISTIC_Z0", LOGISTIC_Z0, |p, v| p.logistic_z0 = v);
    t.finish()
}

// ─── metric 4: Recovery Index (overnight HR-decline slope) ─────────────────────────────────────

/// Replica of recovery.rs:119-152 with the bin width and bin floor as arguments.
fn slope_replica(window_s: i64, min_bins: usize, hr: &[HrSample], start: i64, end: i64) -> Option<f64> {
    let seg: Vec<&HrSample> = hr.iter().filter(|s| s.ts >= start && s.ts <= end).collect();
    if seg.is_empty() {
        return None;
    }
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut t = start;
    while t < end {
        let bin_end = t + window_s;
        let win: Vec<f64> =
            seg.iter().filter(|s| s.ts >= t && s.ts < bin_end).map(|s| s.bpm as f64).collect();
        if !win.is_empty() {
            let mean = win.iter().sum::<f64>() / win.len() as f64;
            let midpoint_s = (t - start) as f64 + window_s as f64 / 2.0;
            points.push((midpoint_s / 3600.0, mean));
        }
        t += window_s;
    }
    if points.len() < min_bins {
        return None;
    }
    let n = points.len() as f64;
    let t_bar = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y_bar = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (t_hours, mean_bpm) in &points {
        let dt = t_hours - t_bar;
        num += dt * (mean_bpm - y_bar);
        den += dt * dt;
    }
    if den <= 1e-9 {
        return Some(0.0);
    }
    Some(num / den)
}

#[derive(Clone, Copy)]
struct SlopeCfg {
    window_s: i64,
    min_bins: usize,
    zero: bool,
    negate: bool,
    flat_input: bool,
    reverse_input: bool,
    drop_tail: bool,
}

impl SlopeCfg {
    fn shipped() -> Self {
        Self {
            window_s: RESTING_HR_WINDOW_S,
            min_bins: RECOVERY_INDEX_MIN_BINS,
            zero: false,
            negate: false,
            flat_input: false,
            reverse_input: false,
            drop_tail: false,
        }
    }
}

fn slope_eval(c: &SlopeCfg, hr: &[HrSample], start: i64, end: i64) -> Option<f64> {
    if c.zero {
        return Some(0.0);
    }
    let mut v = hr.to_vec();
    if c.flat_input && !v.is_empty() {
        let m = (v.iter().map(|s| s.bpm as f64).sum::<f64>() / v.len() as f64).round() as i32;
        for s in v.iter_mut() {
            s.bpm = m;
        }
    }
    if c.reverse_input {
        let bpms: Vec<i32> = v.iter().rev().map(|s| s.bpm).collect();
        for (s, b) in v.iter_mut().zip(bpms) {
            s.bpm = b;
        }
    }
    if c.drop_tail {
        let keep = v.len() * 9 / 10;
        v.truncate(keep);
    }
    let out = slope_replica(c.window_s, c.min_bins, &v, start, end)?;
    Some(if c.negate { -out } else { out })
}

/// A night at `start_bpm` with a known injected slope, sampled every 30 s.
fn slope_series(start_bpm: f64, slope_per_hour: f64, hours: i64) -> (Vec<HrSample>, i64, i64) {
    let origin: i64 = 100_000;
    let total = hours * 3600;
    let mut samples = Vec::new();
    let mut t = 0;
    while t < total {
        let bpm = start_bpm + slope_per_hour * (t as f64 / 3600.0);
        samples.push(HrSample::new(origin + t, bpm.round() as i32));
        t += 30;
    }
    (samples, origin, origin + total)
}

const SLOPE_FIXTURES: [(f64, f64); 4] = [(62.0, 0.0), (62.0, -1.0), (68.0, -4.0), (55.0, 2.0)];

/// recovery.rs — one sample dead-centre of each of `bins` consecutive windows.
fn one_sample_per_bin(bins: usize) -> (Vec<HrSample>, i64, i64) {
    let origin: i64 = 100_000;
    let samples = (0..bins)
        .map(|b| HrSample::new(origin + b as i64 * RESTING_HR_WINDOW_S + 150, 60 - b as i32))
        .collect();
    (samples, origin, origin + bins as i64 * RESTING_HR_WINDOW_S)
}

fn slope_gate_holds(c: &SlopeCfg) -> bool {
    if slope_eval(c, &[], 0, 1000).is_some() {
        return false;
    }
    // The coverage rule counts POPULATED BINS: refuse below the minimum, answer at and above it, and
    // refuse a dense five-bin night whatever its sample count.
    for bins in 1..=8usize {
        let (v, a, b) = one_sample_per_bin(bins);
        if slope_eval(c, &v, a, b).is_some() != (bins >= RECOVERY_INDEX_MIN_BINS) {
            return false;
        }
    }
    let origin: i64 = 100_000;
    let span = 5 * RESTING_HR_WINDOW_S;
    let dense: Vec<HrSample> = (0..span).map(|i| HrSample::new(origin + i, 60)).collect();
    if slope_eval(c, &dense, origin, origin + span).is_some() {
        return false;
    }
    // Empty bins inside the window do not count against it: six populated of thirteen passes.
    let gapped: Vec<HrSample> = [0i64, 1, 2, 10, 11, 12]
        .iter()
        .map(|&b| HrSample::new(origin + b * RESTING_HR_WINDOW_S + 10, 70 - b as i32))
        .collect();
    if slope_eval(c, &gapped, origin, origin + 13 * RESTING_HR_WINDOW_S).is_none() {
        return false;
    }

    let mut got = Vec::new();
    for target in GATE_SLOPE_TARGETS {
        let (v, a, b) = slope_series(65.0, target, 6);
        let Some(x) = slope_eval(c, &v, a, b) else { return false };
        if !near(x, target, GATE_SLOPE_TOL) {
            return false;
        }
        got.push((target, x));
    }
    got.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if !got.windows(2).all(|w| w[0].1 < w[1].1) {
        return false;
    }

    // The unit is per HOUR: the same rate over 2 to 9 hours reads the same and converges on it.
    let mut errs = Vec::new();
    for hours in GATE_SLOPE_HOURS {
        let (v, a, b) = slope_series(65.0, -2.0, hours);
        let Some(x) = slope_eval(c, &v, a, b) else { return false };
        errs.push((x + 2.0).abs());
    }
    errs.iter().all(|e| *e < GATE_SLOPE_HOURS_TOL) && errs.windows(2).all(|w| w[1] < w[0])
}

fn slope_arm(kind: Kind, name: String, c: &SlopeCfg) -> Arm {
    let (v, a, b) = slope_series(68.0, -4.0, 6);
    let value = slope_eval(c, &v, a, b).unwrap_or(f64::NAN);
    Arm::new(kind, name, value, slope_gate_holds(c))
}

fn metric_recovery_index() -> Score {
    for (bpm, slope) in SLOPE_FIXTURES {
        let (v, a, b) = slope_series(bpm, slope, 6);
        let got = slope_eval(&SlopeCfg::shipped(), &v, a, b);
        assert_eq!(got, recovery_index_slope(&v, a, b), "slope replica differs at {slope} bpm/h");
    }

    let mut t = Table::new(
        "Recovery Index (overnight HR-decline slope, bpm/hour)",
        "recovery.rs  populated-bin coverage + nine rates within 0.05 + monotone + per-hour convergence",
        "steep",
    );
    t.add(slope_arm(Kind::Baseline, "unmutated".into(), &SlopeCfg::shipped()));
    t.add(slope_arm(
        Kind::Null,
        "constant scorer, every night 0.0 bpm/h".into(),
        &SlopeCfg { zero: true, ..SlopeCfg::shipped() },
    ));
    t.add(slope_arm(
        Kind::Null,
        "input flattened to its own mean bpm".into(),
        &SlopeCfg { flat_input: true, ..SlopeCfg::shipped() },
    ));
    t.add(slope_arm(
        Kind::Structural,
        "slope sign flipped".into(),
        &SlopeCfg { negate: true, ..SlopeCfg::shipped() },
    ));
    t.add(slope_arm(
        Kind::Structural,
        "input reversed in time".into(),
        &SlopeCfg { reverse_input: true, ..SlopeCfg::shipped() },
    ));
    t.add(slope_arm(
        Kind::Structural,
        "last 10% of the night dropped".into(),
        &SlopeCfg { drop_tail: true, ..SlopeCfg::shipped() },
    ));
    for &(tag, m) in PM10.iter() {
        let v = (RESTING_HR_WINDOW_S as f64 * m).round() as i64;
        t.add(
            slope_arm(
                Kind::Param,
                format!("RESTING_HR_WINDOW_S {RESTING_HR_WINDOW_S} -> {v} ({tag})"),
                &SlopeCfg { window_s: v, ..SlopeCfg::shipped() },
            )
            .p((m - 1.0).abs()),
        );
    }
    for d in [1i64, -1] {
        let v = RECOVERY_INDEX_MIN_BINS as i64 + d;
        t.add(
            slope_arm(
                Kind::Param,
                format!("RECOVERY_INDEX_MIN_BINS {RECOVERY_INDEX_MIN_BINS} -> {v} ({d:+})"),
                &SlopeCfg { min_bins: v as usize, ..SlopeCfg::shipped() },
            )
            .p(1.0 / RECOVERY_INDEX_MIN_BINS as f64),
        );
    }
    let v = (RESTING_HR_WINDOW_S as f64 * 1.005).round() as i64;
    t.add(
        slope_arm(
            Kind::Param,
            format!("RESTING_HR_WINDOW_S {RESTING_HR_WINDOW_S} -> {v} (+0.5%)"),
            &SlopeCfg { window_s: v, ..SlopeCfg::shipped() },
        )
        .p(0.005),
    );
    t.finish()
}

// ─── metric 5: banked nights ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum BankMode {
    Shipped,
    Zero,
    CountAll,
    IgnoreBounds,
    SwapBounds,
    LiteralProbe,
}

/// The shipped fixture seeds its boundary night from `HRV_MIN_MS` itself. Every mode
/// but `LiteralProbe` reproduces that self-reference; `LiteralProbe` pins the probe at the literal 5.0
/// to show what the gate would have caught without it.
fn count_with(mode: BankMode, nights: &[Option<f64>], min: f64, max: f64) -> usize {
    match mode {
        BankMode::Zero => 0,
        BankMode::CountAll => nights.len(),
        BankMode::IgnoreBounds => nights.iter().filter(|v| v.is_some()).count(),
        BankMode::SwapBounds => banked_nights(nights, max, min),
        _ => banked_nights(nights, min, max),
    }
}

fn bank_nights(min: f64, mode: BankMode) -> [Option<f64>; 6] {
    let probe = if mode == BankMode::LiteralProbe { 5.0 } else { min };
    [Some(55.0), None, Some(3.0), Some(300.0), Some(80.0), Some(probe)]
}

fn banked_count(min: f64, max: f64, mode: BankMode) -> usize {
    count_with(mode, &bank_nights(min, mode), min, max)
}

/// The shipped gate: the single-night probe table (literals beside the two constant-seeded edges),
/// the six-night count, and the same series with the bounds widened to everything.
fn bank_gate_holds(min: f64, max: f64, mode: BankMode) -> bool {
    if count_with(mode, &[None], min, max) != 0 {
        return false;
    }
    for (v, want) in GATE_BANKED_PROBES {
        if count_with(mode, &[Some(v)], min, max) != want {
            return false;
        }
    }
    if count_with(mode, &[Some(min)], min, max) != 1 || count_with(mode, &[Some(max)], min, max) != 1 {
        return false;
    }
    banked_count(min, max, mode) == GATE_BANKED_NIGHTS
        && count_with(mode, &bank_nights(min, mode), 0.0, f64::INFINITY) == GATE_BANKED_WIDE
}

fn bank_arm(kind: Kind, name: String, min: f64, max: f64, mode: BankMode) -> Arm {
    let n = banked_count(min, max, mode);
    Arm::new(kind, name, n as f64, bank_gate_holds(min, max, mode))
}

fn metric_banked_nights() -> Score {
    let mut t = Table::new(
        "Banked nights (calibration progress count)",
        "recovery.rs  twelve single-night probes + the count of 3 + the widened-bounds count of 5",
        "count",
    );
    t.add(bank_arm(Kind::Baseline, "unmutated".into(), HRV_MIN_MS, HRV_MAX_MS, BankMode::Shipped));
    t.add(bank_arm(Kind::Null, "always 0 banked".into(), HRV_MIN_MS, HRV_MAX_MS, BankMode::Zero));
    t.add(bank_arm(
        Kind::Null,
        "count every slot, missing nights included".into(),
        HRV_MIN_MS,
        HRV_MAX_MS,
        BankMode::CountAll,
    ));
    t.add(bank_arm(
        Kind::Structural,
        "bounds ignored, any present value banks".into(),
        HRV_MIN_MS,
        HRV_MAX_MS,
        BankMode::IgnoreBounds,
    ));
    t.add(bank_arm(
        Kind::Structural,
        "min and max exchanged".into(),
        HRV_MIN_MS,
        HRV_MAX_MS,
        BankMode::SwapBounds,
    ));
    t.add(bank_arm(
        Kind::Structural,
        "HRV_MIN_MS x1.10 with the probe pinned at the literal 5.0".into(),
        HRV_MIN_MS * 1.10,
        HRV_MAX_MS,
        BankMode::LiteralProbe,
    ));
    for &(tag, m) in PM10.iter().chain(TINY.iter()) {
        let v = HRV_MIN_MS * m;
        t.add(
            bank_arm(
                Kind::Param,
                format!("HRV_MIN_MS {HRV_MIN_MS} -> {v:.4} ({tag})"),
                v,
                HRV_MAX_MS,
                BankMode::Shipped,
            )
            .p((m - 1.0).abs()),
        );
        let v = HRV_MAX_MS * m;
        t.add(
            bank_arm(
                Kind::Param,
                format!("HRV_MAX_MS {HRV_MAX_MS} -> {v:.4} ({tag})"),
                HRV_MIN_MS,
                v,
                BankMode::Shipped,
            )
            .p((m - 1.0).abs()),
        );
    }
    let s = t.finish();
    println!(
        "note: the fixture writes `Some(HRV_MIN_MS)` for its boundary night, so moving HRV_MIN_MS moves \
         the probe with it. The pinned range is (3.0, 55.0] for the min and [80.0, 300.0) for the max."
    );
    s
}

// ─── metric 6: HRV readiness tier ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum SeriesMut {
    Keep,
    Flatten,
    Reverse,
    DropNewest,
}

/// hrv.rs:11-20 — these are PRIVATE consts, so the literals are copied here. `min_nights` is the one
/// reachable member (`calibration::RECOVERY_SCORE.unlock`).
#[derive(Clone, Copy)]
struct HrvParams {
    hrv_min: f64,
    hrv_max: f64,
    roll_window: usize,
    long_window: usize,
    long_window_fallback: usize,
    swc_k: f64,
    min_nights: usize,
    cv_trend_window: usize,
    long_sd_floor: f64,
    series_mut: SeriesMut,
}

impl HrvParams {
    fn shipped() -> Self {
        Self {
            hrv_min: 5.0,
            hrv_max: 250.0,
            roll_window: 7,
            long_window: 60,
            long_window_fallback: 30,
            swc_k: 0.5,
            min_nights: RECOVERY_SCORE.unlock as usize,
            cv_trend_window: 28,
            long_sd_floor: 1e-9,
            series_mut: SeriesMut::Keep,
        }
    }
}

struct HrvOut {
    tier: ReadinessTier,
    baseline7_ms: f64,
    normal_high_ms: f64,
    watch: bool,
}

fn tail(xs: &[f64], n: usize) -> &[f64] {
    &xs[xs.len().saturating_sub(n)..]
}

fn apply_series_mut(m: SeriesMut, xs: &[Option<f64>]) -> Vec<Option<f64>> {
    match m {
        SeriesMut::Keep => xs.to_vec(),
        SeriesMut::Flatten => {
            let vals: Vec<f64> = xs.iter().flatten().copied().collect();
            let mean = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
            xs.iter().map(|v| v.map(|_| mean)).collect()
        }
        SeriesMut::Reverse => {
            let mut v = xs.to_vec();
            v.reverse();
            v
        }
        SeriesMut::DropNewest => {
            let mut v = xs.to_vec();
            v.pop();
            v
        }
    }
}

fn cv_slope_with(p: &HrvParams, ell: &[f64]) -> f64 {
    let start = (p.roll_window - 1).max(ell.len().saturating_sub(p.cv_trend_window));
    let mut cv = Vec::new();
    for i in start..ell.len() {
        let w = &ell[i + 1 - p.roll_window..=i];
        let m = stats::mean(w);
        cv.push(if m != 0.0 { 100.0 * stats::sample_sd(w) / m } else { 0.0 });
    }
    stats::least_squares_slope(&cv)
}

/// Replica of hrv.rs:307-348 (`HrvReadiness::evaluate`).
fn evaluate_with(p: &HrvParams, nightly: &[Option<f64>]) -> Option<HrvOut> {
    let nightly = apply_series_mut(p.series_mut, nightly);
    let valid: Vec<f64> = nightly
        .iter()
        .filter_map(|&v| v)
        .filter(|&v| (p.hrv_min..=p.hrv_max).contains(&v))
        .collect();
    if valid.len() < p.min_nights {
        return None;
    }
    let ell: Vec<f64> = valid.iter().map(|&v| v.max(1.0).ln()).collect();
    let baseline7 = stats::mean(tail(&ell, p.roll_window));
    let long_win = if valid.len() >= p.long_window { p.long_window } else { p.long_window_fallback };
    let long_ell = tail(&ell, long_win);
    let long_mean = stats::mean(long_ell);
    let long_sd_raw = if long_ell.len() >= 2 {
        stats::sample_sd(long_ell)
    } else {
        stats::sample_sd(tail(&ell, p.roll_window))
    };
    let long_sd = long_sd_raw.max(p.long_sd_floor);
    let swc_half = p.swc_k * long_sd;
    let (normal_low, normal_high) = (long_mean - swc_half, long_mean + swc_half);
    let tier = if baseline7 > normal_high {
        ReadinessTier::Primed
    } else if baseline7 >= normal_low {
        ReadinessTier::Normal
    } else {
        ReadinessTier::Suppressed
    };
    Some(HrvOut {
        tier,
        baseline7_ms: baseline7.exp(),
        normal_high_ms: normal_high.exp(),
        watch: cv_slope_with(p, &ell) < 0.0 && baseline7 < long_mean,
    })
}

fn hrv_flat() -> Vec<Option<f64>> {
    vec![Some(50.0); 20]
}
fn hrv_rising() -> Vec<Option<f64>> {
    let mut v = vec![Some(40.0); 13];
    v.extend(vec![Some(80.0); 7]);
    v
}
fn hrv_falling() -> Vec<Option<f64>> {
    let mut v = vec![Some(80.0); 13];
    v.extend(vec![Some(40.0); 7]);
    v
}
fn hrv_mixed() -> Vec<Option<f64>> {
    vec![Some(50.0), Some(50.0), None, Some(400.0)]
}

fn hrv_gate_holds(p: &HrvParams) -> bool {
    if evaluate_with(p, &vec![Some(50.0); p.min_nights.saturating_sub(1)]).is_some() {
        return false;
    }
    if evaluate_with(p, &hrv_flat()).map(|o| o.tier) != Some(ReadinessTier::Normal) {
        return false;
    }
    if evaluate_with(p, &hrv_rising()).map(|o| o.tier) != Some(ReadinessTier::Primed) {
        return false;
    }
    if evaluate_with(p, &hrv_falling()).map(|o| o.tier) != Some(ReadinessTier::Suppressed) {
        return false;
    }
    evaluate_with(p, &hrv_mixed()).is_none()
}

/// The margin the Primed decision turns on: `baseline7_ms - normal_high_ms` on the rising fixture.
fn hrv_arm(kind: Kind, name: String, p: &HrvParams) -> Arm {
    let (margin, watch) = match evaluate_with(p, &hrv_rising()) {
        Some(o) => (o.baseline7_ms - o.normal_high_ms, if o.watch { 1.0 } else { 0.0 }),
        None => (f64::NAN, f64::NAN),
    };
    Arm::new(kind, name, margin, hrv_gate_holds(p)).x(watch)
}

fn hrv_param(t: &mut Table, label: &str, base: f64, muls: &[(&str, f64)], set: impl Fn(&mut HrvParams, f64)) {
    for &(tag, m) in muls {
        let mut p = HrvParams::shipped();
        let v = base * m;
        set(&mut p, v);
        t.add(hrv_arm(Kind::Param, format!("{label} {base} -> {v:.4} ({tag})"), &p).p((m - 1.0).abs()));
    }
}

fn hrv_param_int(t: &mut Table, label: &str, base: usize, set: impl Fn(&mut HrvParams, usize)) {
    for d in [1i64, -1] {
        let v = (base as i64 + d).max(1) as usize;
        let mut p = HrvParams::shipped();
        set(&mut p, v);
        t.add(
            hrv_arm(Kind::Param, format!("{label} {base} -> {v} ({d:+})"), &p)
                .p(1.0 / base as f64),
        );
    }
}

fn metric_hrv_readiness() -> Score {
    for f in [hrv_flat(), hrv_rising(), hrv_falling(), hrv_mixed()] {
        let want = HrvReadiness::evaluate(&f);
        let got = evaluate_with(&HrvParams::shipped(), &f);
        assert_eq!(want.is_some(), got.is_some(), "hrv replica presence differs");
        if let (Some(w), Some(g)) = (want, got) {
            assert_eq!(w.tier, g.tier, "hrv replica tier differs");
            assert!((w.baseline7_ms - g.baseline7_ms).abs() < 1e-9, "hrv replica baseline differs");
            assert!((w.normal_high_ms - g.normal_high_ms).abs() < 1e-9, "hrv replica band differs");
            assert_eq!(w.overreaching_watch, g.watch, "hrv replica watch differs");
        }
    }

    let mut t = Table::new(
        "HRV readiness tier (Primed / Normal / Suppressed)",
        "hrv.rs:926 None + :932 Normal + :939 Primed + :943 Suppressed + :952 None",
        "primed mgn",
    )
    .extra("watch");
    t.add(hrv_arm(Kind::Baseline, "unmutated".into(), &HrvParams::shipped()));
    t.add(hrv_arm(
        Kind::Null,
        "every night replaced by the series mean".into(),
        &HrvParams { series_mut: SeriesMut::Flatten, ..HrvParams::shipped() },
    ));
    t.add(hrv_arm(
        Kind::Structural,
        "series reversed, oldest becomes newest".into(),
        &HrvParams { series_mut: SeriesMut::Reverse, ..HrvParams::shipped() },
    ));
    t.add(hrv_arm(
        Kind::Structural,
        "newest night dropped (series shifted one night)".into(),
        &HrvParams { series_mut: SeriesMut::DropNewest, ..HrvParams::shipped() },
    ));
    hrv_param_int(&mut t, "ROLL_WINDOW", 7, |p, v| p.roll_window = v);
    hrv_param_int(&mut t, "LONG_WINDOW", 60, |p, v| p.long_window = v);
    hrv_param_int(&mut t, "LONG_WINDOW_FALLBACK", 30, |p, v| p.long_window_fallback = v);
    hrv_param_int(&mut t, "MIN_NIGHTS", RECOVERY_SCORE.unlock as usize, |p, v| p.min_nights = v);
    hrv_param_int(&mut t, "CV_TREND_WINDOW", 28, |p, v| p.cv_trend_window = v);
    hrv_param(&mut t, "SWC_K", 0.5, &PM10, |p, v| p.swc_k = v);
    hrv_param(&mut t, "SWC_K", 0.5, &TINY, |p, v| p.swc_k = v);
    hrv_param(&mut t, "HRV_MIN_MS (hrv.rs private copy)", 5.0, &PM10, |p, v| p.hrv_min = v);
    hrv_param(&mut t, "HRV_MAX_MS (hrv.rs private copy)", 250.0, &PM10, |p, v| p.hrv_max = v);
    let s = t.finish();
    println!(
        "note: `overreaching_watch` (the only thing CV_TREND_WINDOW feeds) is never asserted, so that \
         tunable cannot be caught at any magnitude. hrv.rs:11-12 also re-declares HRV_MIN_MS / \
         HRV_MAX_MS privately while recovery.rs:56-57 declares them publicly and \
         baselines::MetricCfg::hrv() carries the same 5.0 / 250.0 again: three copies of one bound."
    );
    s
}

// ─── metric 7: personal baselines ──────────────────────────────────────────────────────────────

/// baselines.rs:5-12. `STALE_DAYS`, `EARLY_HALF_LIFE_B`, `HARD_OUTLIER_K`, `WINSOR_K` and
/// `EARLY_SPREAD_INFLATE` are PRIVATE, so their literals are copied here. `frozen` / `midpoint` /
/// `reverse_fold` are control knobs, not shipped constants.
#[derive(Clone, Copy)]
struct BaseParams {
    min_nights_seed: i32,
    min_nights_trust: i32,
    stale_days: i32,
    early_adapt_nights: i32,
    early_half_life_b: f64,
    hard_outlier_k: f64,
    winsor_k: f64,
    early_spread_inflate: f64,
    frozen: bool,
    midpoint: bool,
    reverse_fold: bool,
}

impl BaseParams {
    fn shipped() -> Self {
        Self {
            min_nights_seed: MIN_NIGHTS_SEED,
            min_nights_trust: MIN_NIGHTS_TRUST,
            stale_days: 14,
            early_adapt_nights: EARLY_ADAPT_NIGHTS,
            early_half_life_b: 3.0,
            hard_outlier_k: 5.0,
            winsor_k: 3.0,
            early_spread_inflate: 2.5,
            frozen: false,
            midpoint: false,
            reverse_fold: false,
        }
    }
}

fn lambda(half_life: f64) -> f64 {
    1.0 - 0.5f64.powf(1.0 / half_life)
}

fn status_with(p: &BaseParams, n_valid: i32, since: i32) -> BaselineStatus {
    if since > p.stale_days && n_valid >= p.min_nights_seed {
        BaselineStatus::Stale
    } else if n_valid < p.min_nights_seed {
        BaselineStatus::Calibrating
    } else if n_valid < p.min_nights_trust {
        BaselineStatus::Provisional
    } else {
        BaselineStatus::Trusted
    }
}

/// Replica of baselines.rs:68-125.
fn update_core(
    p: &BaseParams, state: Option<BaselineState>, value: Option<f64>, cfg: &MetricCfg,
) -> BaselineState {
    let lb = lambda(cfg.half_life_b);
    let ls = lambda(cfg.half_life_s);
    let Some(s) = state else {
        if let Some(v) = value {
            if v >= cfg.min_val && v <= cfg.max_val {
                return BaselineState {
                    baseline: v,
                    spread: cfg.floor_spread,
                    n_valid: 1,
                    nights_since_update: 0,
                    status: BaselineStatus::Calibrating,
                };
            }
        }
        return BaselineState {
            baseline: (cfg.min_val + cfg.max_val) / 2.0,
            spread: cfg.floor_spread,
            n_valid: 0,
            nights_since_update: 1,
            status: BaselineStatus::Calibrating,
        };
    };
    let hold = |m: i32| BaselineState {
        baseline: s.baseline,
        spread: s.spread,
        n_valid: s.n_valid,
        nights_since_update: m,
        status: status_with(p, s.n_valid, m),
    };
    let Some(v) = value else { return hold(s.nights_since_update + 1) };
    if !(cfg.min_val <= v && v <= cfg.max_val) {
        return hold(s.nights_since_update + 1);
    }
    let is_young = s.n_valid < p.early_adapt_nights;
    if s.n_valid >= p.min_nights_seed && !is_young && (v - s.baseline).abs() > p.hard_outlier_k * s.spread {
        return hold(0);
    }
    if s.n_valid == 0 {
        return BaselineState {
            baseline: v,
            spread: cfg.floor_spread,
            n_valid: 1,
            nights_since_update: 0,
            status: BaselineStatus::Calibrating,
        };
    }
    let eff_spread = if is_young { s.spread * p.early_spread_inflate } else { s.spread };
    let eff_lb = if is_young { lambda(p.early_half_life_b) } else { lb };
    let lo = s.baseline - p.winsor_k * eff_spread;
    let hi = s.baseline + p.winsor_k * eff_spread;
    let new_center = eff_lb * v.clamp(lo, hi) + (1.0 - eff_lb) * s.baseline;
    let new_spread = (ls * (v - new_center).abs() + (1.0 - ls) * s.spread).max(cfg.floor_spread);
    let n = s.n_valid + 1;
    BaselineState {
        baseline: new_center,
        spread: new_spread,
        n_valid: n,
        nights_since_update: 0,
        status: status_with(p, n, 0),
    }
}

fn update_with(
    p: &BaseParams, state: Option<BaselineState>, value: Option<f64>, cfg: &MetricCfg,
) -> BaselineState {
    let mut out = update_core(p, state, value, cfg);
    if p.frozen {
        if let Some(s) = state {
            out.baseline = s.baseline;
        }
    }
    if p.midpoint {
        out.baseline = (cfg.min_val + cfg.max_val) / 2.0;
    }
    out
}

fn fold_with(p: &BaseParams, values: &[Option<f64>], cfg: &MetricCfg) -> BaselineState {
    let mut vs = values.to_vec();
    if p.reverse_fold {
        vs.reverse();
    }
    let mut state: Option<BaselineState> = None;
    for v in &vs {
        state = Some(update_with(p, state, *v, cfg));
    }
    state.unwrap_or(BaselineState {
        baseline: (cfg.min_val + cfg.max_val) / 2.0,
        spread: cfg.floor_spread,
        n_valid: 0,
        nights_since_update: 0,
        status: BaselineStatus::Calibrating,
    })
}

fn out_of_band(v: Option<f64>, lo: f64, hi: f64) -> bool {
    v.is_some_and(|x| !(x.is_finite() && lo <= x && x <= hi))
}

fn under_floor(v: Option<f64>, floor: f64) -> bool {
    v.is_some_and(|x| !x.is_finite() || x < floor)
}

/// Replica of baselines.rs:223-239. The HRV band comes from the arm's `MetricCfg`; the resting-HR and
/// respiration bands stay shipped because no arm moves them.
fn night_verdict_with(gate: &NightGate, n: &NightChannels, hrv_cfg: &MetricCfg) -> NightVerdict {
    if under_floor(n.total_sleep_secs, gate.min_sleep_secs) {
        return NightVerdict::SleepTooShort;
    }
    let (lo, hi) = gate.worn_skin_temp_c;
    if out_of_band(n.skin_temp_c, lo, hi) || out_of_band(n.skin_temp_max_c, lo, hi) {
        return NightVerdict::SkinTempImplausible;
    }
    let rhr = MetricCfg::resting_hr();
    if out_of_band(n.resting_hr_bpm, rhr.min_val, rhr.max_val) {
        return NightVerdict::RestingHrImplausible;
    }
    if out_of_band(n.hrv_ms, hrv_cfg.min_val, hrv_cfg.max_val) {
        return NightVerdict::HrvImplausible;
    }
    let resp = MetricCfg::resp();
    if out_of_band(n.resp_bpm, resp.min_val, resp.max_val) {
        return NightVerdict::RespImplausible;
    }
    if gate.min_quality.is_some_and(|f| under_floor(n.quality, f)) {
        return NightVerdict::LowQuality;
    }
    NightVerdict::Valid
}

fn good_night() -> NightChannels {
    NightChannels {
        hrv_ms: Some(45.0),
        resting_hr_bpm: Some(55.0),
        resp_bpm: Some(14.0),
        skin_temp_c: Some(33.0),
        skin_temp_max_c: Some(34.2),
        total_sleep_secs: Some(7.0 * 3600.0),
        quality: Some(0.9),
    }
}

/// baselines.rs:355-361 — twelve nights, three of them off-wrist with a plausible-looking low HRV.
fn conjunction_nights() -> Vec<NightChannels> {
    let mut nights = vec![good_night(); 12];
    for i in [4usize, 5, 6] {
        nights[i].hrv_ms = Some(20.0);
        nights[i].skin_temp_c = Some(22.0);
        nights[i].skin_temp_max_c = Some(22.9);
    }
    nights
}

#[derive(Clone, Copy)]
struct BaseArm {
    p: BaseParams,
    cfg: MetricCfg,
    gate: NightGate,
    min_conf: f64,
}

impl BaseArm {
    fn shipped() -> Self {
        Self {
            p: BaseParams::shipped(),
            cfg: MetricCfg::hrv(),
            gate: NightGate::default(),
            min_conf: 0.3,
        }
    }
}

/// baselines.rs:316-321 — seed 50, then twenty nights at 45.
fn converged(a: &BaseArm) -> BaselineState {
    let mut s = update_with(&a.p, None, Some(50.0), &a.cfg);
    for _ in 0..20 {
        s = update_with(&a.p, Some(s), Some(45.0), &a.cfg);
    }
    s
}

/// The cohort this control replicates: EVERY `#[test]` in `baselines.rs`, counted from the source so a
/// test added or removed there fails loudly here instead of silently shrinking what the control measures.
const BASELINES_SHIPPED_TESTS: usize = 20;

/// One `each_channel_can_reject_alone` row: spoil a single channel, expect a single verdict.
type ChannelProbe = (fn(&mut NightChannels), NightVerdict);

fn baselines_shipped_test_count() -> usize {
    include_str!("../src/baselines.rs").matches("#[test]").count()
}

/// Replica of all twenty `#[test]`s in `baselines.rs`, in source order. A partial replica understates
/// the gate: `each_channel_can_reject_alone` (:383) is the only test that pins `MIN_SLEEP_SECS` downward
/// and `WORN_SKIN_TEMP_C.1` upward, and `rejected_night_skip_and_holds` (:374) the only one that pins
/// fold order.
///
/// The negated comparisons below are deliberate and must not be "simplified". `!(x < GATE)` is TRUE
/// for NaN, so a mutation that produces NaN fails this gate, which is the behaviour a negative
/// control needs. `x >= GATE` is FALSE for NaN, so the same mutation would slip through and the arm
/// would silently stop discriminating.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn base_gate_holds(a: &BaseArm) -> bool {
    // :291 cold_start_seeds_first_valid_value
    let s = update_with(&a.p, None, Some(30.0), &a.cfg);
    if s.baseline != GATE_COLD_START_BASELINE || s.n_valid != GATE_COLD_START_N_VALID {
        return false;
    }
    // :298 null_skip_and_hold
    let seeded = |baseline: f64| BaselineState {
        baseline,
        spread: 5.0,
        n_valid: 5,
        nights_since_update: 0,
        status: BaselineStatus::Provisional,
    };
    let held = update_with(&a.p, Some(seeded(30.0)), None, &a.cfg);
    if held.baseline != GATE_NULL_HOLD_BASELINE
        || held.n_valid != GATE_NULL_HOLD_N_VALID
        || held.nights_since_update != GATE_NULL_HOLD_SINCE
    {
        return false;
    }
    // :307 out_of_range_skip — 300 sits above the shipped HRV max of 250.
    if update_with(&a.p, Some(seeded(50.0)), Some(300.0), &a.cfg).baseline != GATE_OUT_OF_RANGE_BASELINE
    {
        return false;
    }
    // :327 fold_reproduces_the_ewma_centre_a_z_score_is_measured_against — the closed form, NOT the
    // old `baseline < 50` gate, which "return the last value" and "return the mean" both satisfy.
    let mut s = update_with(&a.p, None, Some(50.0), &a.cfg);
    // FIXED probe count and SHIPPED half-life in the expectation: if both followed the arm they
    // would move together and the mutation would be invisible, which is how the old gate failed.
    let young_folds = GATE_EARLY_ADAPT_NIGHTS - 1;
    for _ in 0..young_folds {
        s = update_with(&a.p, Some(s), Some(45.0), &a.cfg);
    }
    let young = 45.0 + 5.0 * 0.5f64.powf(young_folds as f64 / GATE_EARLY_HALF_LIFE_B);
    if (s.baseline - young).abs() > REPLICA_TOL || s.n_valid != GATE_EARLY_ADAPT_NIGHTS {
        return false;
    }
    for _ in 0..13 {
        s = update_with(&a.p, Some(s), Some(45.0), &a.cfg);
    }
    let steady = 45.0 + (young - 45.0) * 0.5f64.powf(13.0 / a.cfg.half_life_b);
    if (s.baseline - steady).abs() > REPLICA_TOL || s.spread != a.cfg.floor_spread {
        return false;
    }
    // :349 a_do_nothing_centre_fails_the_fold — a centre this gate cannot tell apart from the fold is
    // a centre the gate does not test. Each must MISS the fold on at least one of these series.
    let converging: Vec<Option<f64>> =
        std::iter::once(Some(50.0)).chain(std::iter::repeat_n(Some(45.0), 20)).collect();
    let with_outlier: Vec<Option<f64>> =
        std::iter::repeat_n(Some(45.0), 12).chain(std::iter::once(Some(120.0))).collect();
    for series in [&converging, &with_outlier] {
        let folded = fold_with(&a.p, series, &a.cfg).baseline;
        let last = series.iter().flatten().next_back().copied().unwrap();
        let xs: Vec<f64> = series.iter().flatten().copied().collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        if (last - folded).abs() <= 0.05 && (mean - folded).abs() <= 0.05 {
            return false;
        }
    }
    // :381 a_hard_outlier_night_is_seen_but_never_folded — past `hard_outlier_k` spreads the night is
    // SEEN but never folded; just inside it folds, winsorised to `winsor_k` spreads.
    // n_valid must be PAST `early_adapt_nights`, or the young branch runs and there is no
    // hard-outlier limb to test at all.
    let base = BaselineState { n_valid: a.p.early_adapt_nights + 2, ..seeded(45.0) };
    let held_out = update_with(&a.p, Some(base), Some(GATE_OUTLIER_HELD), &a.cfg);
    if held_out.baseline != 45.0
        || held_out.n_valid != base.n_valid
        || held_out.nights_since_update != 0
    {
        return false;
    }
    let folded = update_with(&a.p, Some(base), Some(GATE_OUTLIER_FOLDED), &a.cfg);
    let lb = 1.0 - 0.5f64.powf(1.0 / a.cfg.half_life_b);
    if folded.n_valid != base.n_valid + 1
        || (folded.baseline - (45.0 + GATE_WINSOR_K * base.spread * lb)).abs() > REPLICA_TOL
    {
        return false;
    }
    // :398 winsorised_fold_lifts_the_spread_off_its_floor — the limb a steady series cannot reach.
    if !(folded.spread > a.cfg.floor_spread)
        || fold_with(&a.p, &vec![Some(45.0); 21], &a.cfg).spread != a.cfg.floor_spread
    {
        return false;
    }
    // :414 a_young_baseline_has_no_outlier_guard — RECORDED, not desired. Under `early_adapt_nights`
    // there is no hard-outlier limb and the winsor window is `early_spread_inflate` times wider, so a
    // wild night still moves the centre; one fold later the same night is refused outright.
    let young_state = BaselineState { n_valid: GATE_EARLY_ADAPT_NIGHTS - 1, ..seeded(45.0) };
    let moved = update_with(&a.p, Some(young_state), Some(200.0), &a.cfg).baseline;
    if (moved - GATE_YOUNG_OUTLIER_BASELINE).abs() > 1e-9 {
        return false;
    }
    let grown = BaselineState { n_valid: GATE_EARLY_ADAPT_NIGHTS, ..seeded(45.0) };
    if update_with(&a.p, Some(grown), Some(200.0), &a.cfg).baseline != 45.0 {
        return false;
    }
    // :429 the_status_lifecycle_reaches_stale_and_recovers — `stale_days` missed nights are still
    // Trusted; the next one tips it, and a real night restores it.
    let mut lc = update_with(&a.p, None, Some(45.0), &a.cfg);
    if lc.status != BaselineStatus::Calibrating {
        return false;
    }
    for _ in 1..GATE_MIN_NIGHTS_TRUST {
        lc = update_with(&a.p, Some(lc), Some(45.0), &a.cfg);
    }
    if lc.status != BaselineStatus::Trusted {
        return false;
    }
    for _ in 0..GATE_STALE_DAYS {
        lc = update_with(&a.p, Some(lc), None, &a.cfg);
    }
    if lc.status != BaselineStatus::Trusted {
        return false;
    }
    lc = update_with(&a.p, Some(lc), None, &a.cfg);
    if lc.status != BaselineStatus::Stale || lc.n_valid != GATE_MIN_NIGHTS_TRUST {
        return false;
    }
    if update_with(&a.p, Some(lc), Some(45.0), &a.cfg).status != BaselineStatus::Trusted {
        return false;
    }
    // :331 a_plausible_night_is_valid
    if night_verdict_with(&a.gate, &good_night(), &a.cfg) != NightVerdict::Valid {
        return false;
    }
    // :336 missing_channels_never_reject
    let sparse = NightChannels { hrv_ms: Some(45.0), ..Default::default() };
    if night_verdict_with(&a.gate, &sparse, &a.cfg) != NightVerdict::Valid {
        return false;
    }
    // :342 nightstand_night_rejects_every_metric — the temperature rejects, the per-metric bands do not.
    let mut off_wrist = good_night();
    off_wrist.skin_temp_c = Some(22.4);
    off_wrist.skin_temp_max_c = Some(23.1);
    if night_verdict_with(&a.gate, &off_wrist, &a.cfg) != NightVerdict::SkinTempImplausible {
        return false;
    }
    if !(a.cfg.min_val <= 45.0 && 45.0 <= a.cfg.max_val) || MetricCfg::skin_temp().min_val > 22.4 {
        return false;
    }
    // :354 conjunction_changes_the_hrv_baseline_a_per_metric_gate_would_fold
    let nights = conjunction_nights();
    let per_metric: Vec<Option<f64>> = nights.iter().map(|n| n.hrv_ms).collect();
    let old = fold_with(&a.p, &per_metric, &a.cfg);
    let gated: Vec<Option<f64>> = nights
        .iter()
        .map(|n| if night_verdict_with(&a.gate, n, &a.cfg) == NightVerdict::Valid { n.hrv_ms } else { None })
        .collect();
    let new = fold_with(&a.p, &gated, &a.cfg);
    if old.n_valid != GATE_PER_METRIC_N_VALID || new.n_valid != GATE_CONJUNCTION_N_VALID {
        return false;
    }
    if !(new.baseline > old.baseline + 1.0) {
        return false;
    }
    if nights
        .iter()
        .filter(|n| night_verdict_with(&a.gate, n, &a.cfg) == NightVerdict::SkinTempImplausible)
        .count()
        != GATE_NIGHTSTAND_REJECTIONS
    {
        return false;
    }
    // :374 rejected_night_skip_and_holds_rather_than_disappearing — a resting-HR fold, so it runs the
    // shipped resting-HR cfg (no arm moves it); what the arm drives is the fold order and the gate.
    let mut six = vec![good_night(); 6];
    six[5].total_sleep_secs = Some(2.0 * 3600.0);
    let kept: Vec<Option<f64>> = six
        .iter()
        .map(|n| {
            if night_verdict_with(&a.gate, n, &a.cfg) == NightVerdict::Valid {
                n.resting_hr_bpm
            } else {
                None
            }
        })
        .collect();
    let s = fold_with(&a.p, &kept, &MetricCfg::resting_hr());
    if s.n_valid != GATE_REJECTED_NIGHT_N_VALID || s.nights_since_update != GATE_REJECTED_NIGHT_SINCE {
        return false;
    }
    // :383 each_channel_can_reject_alone — one channel spoiled at a time, the rest of the night good.
    let channel_probes: [ChannelProbe; 6] = [
        (|n| n.total_sleep_secs = Some(3.9 * 3600.0), NightVerdict::SleepTooShort),
        (|n| n.skin_temp_max_c = Some(41.0), NightVerdict::SkinTempImplausible),
        (|n| n.resting_hr_bpm = Some(180.0), NightVerdict::RestingHrImplausible),
        (|n| n.hrv_ms = Some(400.0), NightVerdict::HrvImplausible),
        (|n| n.resp_bpm = Some(60.0), NightVerdict::RespImplausible),
        (|n| n.skin_temp_c = Some(f64::NAN), NightVerdict::SkinTempImplausible),
    ];
    for (spoil, want) in channel_probes {
        let mut n = good_night();
        spoil(&mut n);
        if night_verdict_with(&a.gate, &n, &a.cfg) != want {
            return false;
        }
    }
    // :399 sleep_gate_is_inclusive_at_four_hours — probe and threshold are the same constant, so the
    // arm moves both together, exactly as the shipped test does.
    let mut n = good_night();
    n.total_sleep_secs = Some(a.gate.min_sleep_secs);
    if night_verdict_with(&a.gate, &n, &a.cfg) != NightVerdict::Valid {
        return false;
    }
    n.total_sleep_secs = Some(a.gate.min_sleep_secs - 1.0);
    if night_verdict_with(&a.gate, &n, &a.cfg) != NightVerdict::SleepTooShort {
        return false;
    }
    // :408 quality_floor_is_off_until_a_caller_supplies_one
    let mut q = good_night();
    q.quality = Some(0.05);
    if night_verdict_with(&a.gate, &q, &a.cfg) != NightVerdict::Valid {
        return false;
    }
    let floored = NightGate { min_quality: Some(0.5), ..a.gate };
    if night_verdict_with(&floored, &q, &a.cfg) != NightVerdict::LowQuality {
        return false;
    }
    q.quality = None;
    if night_verdict_with(&floored, &q, &a.cfg) != NightVerdict::Valid {
        return false;
    }
    // :581 night_skin_temp_medians_the_confident_samples_of_every_night
    let Some(s) = night_skin_temp(&[(33.0, 0.9), (33.4, 0.8), (5.0, 0.05), (34.0, 0.7)], a.min_conf)
    else {
        return false;
    };
    if s.median_c != GATE_SKIN_MEDIAN_C || s.max_c != GATE_SKIN_MAX_C {
        return false;
    }
    if (s.n_kept, s.n_total) != GATE_SKIN_KEPT_TOTAL || (s.coverage() - 0.75).abs() >= 1e-12 {
        return false;
    }
    if night_skin_temp(&[(33.0, 0.0)], a.min_conf).is_some()
        || night_skin_temp(&[], a.min_conf).is_some()
    {
        return false;
    }
    // :598 a_do_nothing_night_reducer_fails_the_skin_temp_nights — a reducer this gate cannot tell
    // apart from the median is a reducer the gate does not test.
    let temp_samples: [(f64, f64); 4] = [(33.0, 0.9), (33.4, 0.8), (5.0, 0.05), (34.0, 0.7)];
    let Some(t) = night_skin_temp(&temp_samples, a.min_conf) else { return false };
    let confident: Vec<f64> =
        temp_samples.iter().filter(|(_, c)| *c >= a.min_conf).map(|(v, _)| *v).collect();
    let mean = confident.iter().sum::<f64>() / confident.len() as f64;
    let all: Vec<f64> = temp_samples.iter().map(|(v, _)| *v).collect();
    for (dull, _name) in [(mean, "mean"), (t.max_c, "max"), (physio_algo::stats::median(&all), "unfiltered median")]
    {
        if (dull - t.median_c).abs() <= 1e-9 {
            return false;
        }
    }
    // :633 nightly_skin_temp_medians_drive_the_skin_temp_baseline — the chain the Vitals card reads:
    // per-night median into NightChannels into the baseline a deviation is measured against.
    let planted: Vec<f64> = (0..10).map(|i| 33.0 + i as f64 * 0.125).collect();
    let nights: Vec<NightChannels> = planted
        .iter()
        .map(|&c| {
            let s = night_skin_temp(
                &[(c - 0.5, 0.9), (c, 0.9), (c + 0.75, 0.9), (10.0, 0.01)],
                a.min_conf,
            )
            .expect("the planted night must reduce");
            NightChannels {
                skin_temp_c: Some(s.median_c),
                skin_temp_max_c: Some(s.max_c),
                total_sleep_secs: Some(7.0 * 3600.0),
                quality: Some(s.coverage()),
                ..Default::default()
            }
        })
        .collect();
    if !night_verdicts(&nights, &a.gate).iter().all(|v| v.valid()) {
        return false;
    }
    let chained = fold_history_nights(&nights, NightMetric::SkinTemp, &a.gate);
    let flat: Vec<NightChannels> = nights
        .iter()
        .map(|n| NightChannels { skin_temp_c: Some(33.4), ..*n })
        .collect();
    let constant = fold_history_nights(&flat, NightMetric::SkinTemp, &a.gate);
    (chained.baseline - constant.baseline).abs() > 1e-6
}

fn base_arm(kind: Kind, name: String, a: &BaseArm) -> Arm {
    Arm::new(kind, name, converged(a).baseline, base_gate_holds(a)).x(converged(a).spread)
}

fn base_param(t: &mut Table, label: &str, base: f64, muls: &[(&str, f64)], set: impl Fn(&mut BaseArm, f64)) {
    for &(tag, m) in muls {
        let mut a = BaseArm::shipped();
        let v = base * m;
        set(&mut a, v);
        t.add(base_arm(Kind::Param, format!("{label} {base} -> {v:.4} ({tag})"), &a).p((m - 1.0).abs()));
    }
}

fn base_param_int(t: &mut Table, label: &str, base: i32, set: impl Fn(&mut BaseArm, i32)) {
    for d in [1i32, -1] {
        let v = base + d;
        let mut a = BaseArm::shipped();
        set(&mut a, v);
        t.add(base_arm(Kind::Param, format!("{label} {base} -> {v} ({d:+})"), &a).p(1.0 / base as f64));
    }
}

/// Replica fidelity to the shipped fold. The two run the same arithmetic in a different association
/// order, so they agree to a ULP and not to the bit; anything larger means the replica has drifted.
const REPLICA_TOL: f64 = 1e-9;

fn assert_replica(twin: BaselineState, shipped: BaselineState, what: &str) {
    assert!(
        (twin.baseline - shipped.baseline).abs() < REPLICA_TOL
            && (twin.spread - shipped.spread).abs() < REPLICA_TOL
            && twin.n_valid == shipped.n_valid
            && twin.nights_since_update == shipped.nights_since_update
            && twin.status == shipped.status,
        "baseline replica differs {what}: twin {twin:?} vs shipped {shipped:?}"
    );
}

fn metric_personal_baselines() -> Score {
    assert_eq!(
        baselines_shipped_test_count(),
        BASELINES_SHIPPED_TESTS,
        "baselines.rs changed its test count: base_gate_holds replicates a fixed cohort, so it now \
         measures the wrong gate — re-derive it before quoting a caught/missed tally"
    );
    let ship = BaseArm::shipped();
    assert_replica(
        update_with(&ship.p, None, Some(30.0), &ship.cfg),
        update(None, Some(30.0), &ship.cfg),
        "on the cold start",
    );
    let series: Vec<Option<f64>> = vec![Some(50.0), None, Some(46.0), Some(300.0), Some(44.0), Some(45.5)];
    assert_replica(
        fold_with(&ship.p, &series, &ship.cfg),
        fold_history(&series, &ship.cfg),
        "on a folded series",
    );
    let nights = conjunction_nights();
    let g = NightGate::default();
    assert_eq!(
        night_verdicts(&nights, &g),
        nights.iter().map(|n| night_verdict_with(&g, n, &ship.cfg)).collect::<Vec<_>>(),
        "night-verdict replica differs"
    );
    assert_eq!(night_verdict(&good_night(), &g), NightVerdict::Valid);
    let gated: Vec<Option<f64>> = nights
        .iter()
        .map(|n| if night_verdict_with(&g, n, &ship.cfg) == NightVerdict::Valid { n.hrv_ms } else { None })
        .collect();
    assert_eq!(
        fold_with(&ship.p, &gated, &ship.cfg),
        fold_history_nights(&nights, NightMetric::Hrv, &g),
        "conjunction fold replica differs"
    );

    let mut t = Table::new(
        "Personal baselines (winsorized EWMA centre + abs-dev spread)",
        "baselines.rs — all 13 tests: :291 cold start + :298 null hold + :307 out-of-range + :315 \
         converged + :331/:336/:342 night verdicts + :354 conjunction + :374 skip-and-hold + :383 \
         each channel alone + :399 sleep edge + :408 quality floor + :420 skin temp",
        "converged",
    )
    .extra("spread");
    t.add(base_arm(Kind::Baseline, "unmutated".into(), &ship));
    t.add(base_arm(
        Kind::Null,
        "centre frozen at the first valid value".into(),
        &BaseArm { p: BaseParams { frozen: true, ..BaseParams::shipped() }, ..BaseArm::shipped() },
    ));
    t.add(base_arm(
        Kind::Null,
        "centre always the metric midpoint".into(),
        &BaseArm { p: BaseParams { midpoint: true, ..BaseParams::shipped() }, ..BaseArm::shipped() },
    ));
    t.add(base_arm(
        Kind::Structural,
        "history folded newest first".into(),
        &BaseArm { p: BaseParams { reverse_fold: true, ..BaseParams::shipped() }, ..BaseArm::shipped() },
    ));
    t.add(base_arm(
        Kind::Structural,
        "winsorisation off (WINSOR_K = 1e9)".into(),
        &BaseArm { p: BaseParams { winsor_k: 1e9, ..BaseParams::shipped() }, ..BaseArm::shipped() },
    ));
    base_param_int(&mut t, "MIN_NIGHTS_SEED", MIN_NIGHTS_SEED, |a, v| a.p.min_nights_seed = v);
    base_param_int(&mut t, "MIN_NIGHTS_TRUST", MIN_NIGHTS_TRUST, |a, v| a.p.min_nights_trust = v);
    base_param_int(&mut t, "STALE_DAYS", 14, |a, v| a.p.stale_days = v);
    base_param_int(&mut t, "EARLY_ADAPT_NIGHTS", EARLY_ADAPT_NIGHTS, |a, v| a.p.early_adapt_nights = v);
    base_param(&mut t, "EARLY_HALF_LIFE_B", 3.0, &PM10, |a, v| a.p.early_half_life_b = v);
    base_param(&mut t, "HARD_OUTLIER_K", 5.0, &PM10, |a, v| a.p.hard_outlier_k = v);
    base_param(&mut t, "WINSOR_K", 3.0, &PM10, |a, v| a.p.winsor_k = v);
    base_param(&mut t, "WINSOR_K", 3.0, &TINY, |a, v| a.p.winsor_k = v);
    base_param(&mut t, "EARLY_SPREAD_INFLATE", 2.5, &PM10, |a, v| a.p.early_spread_inflate = v);
    base_param(&mut t, "MetricCfg::hrv min_val", 5.0, &PM10, |a, v| a.cfg.min_val = v);
    base_param(&mut t, "MetricCfg::hrv max_val", 250.0, &PM10, |a, v| a.cfg.max_val = v);
    base_param(&mut t, "MetricCfg::hrv floor_spread", 5.0, &PM10, |a, v| a.cfg.floor_spread = v);
    base_param(&mut t, "MetricCfg::hrv half_life_b", 14.0, &PM10, |a, v| a.cfg.half_life_b = v);
    base_param(&mut t, "MetricCfg::hrv half_life_s", 21.0, &PM10, |a, v| a.cfg.half_life_s = v);
    base_param(&mut t, "NightGate MIN_SLEEP_SECS", MIN_SLEEP_SECS, &PM10, |a, v| {
        a.gate.min_sleep_secs = v
    });
    base_param(&mut t, "NightGate WORN_SKIN_TEMP_C lo", WORN_SKIN_TEMP_C.0, &PM10, |a, v| {
        a.gate.worn_skin_temp_c = (v, WORN_SKIN_TEMP_C.1)
    });
    base_param(&mut t, "NightGate WORN_SKIN_TEMP_C hi", WORN_SKIN_TEMP_C.1, &PM10, |a, v| {
        a.gate.worn_skin_temp_c = (WORN_SKIN_TEMP_C.0, v)
    });
    base_param(&mut t, "night_skin_temp min_conf", 0.3, &PM10, |a, v| a.min_conf = v);
    let s = t.finish();
    println!(
        "note: the spread is never pinned to a number by any shipped assertion — the `spread` column \
         moves freely under arms the gate calls PASS. The sleep-gate edge at :401 writes MIN_SLEEP_SECS \
         as both probe and threshold, so it is self-referential; :383 pins that constant downward \
         anyway with a literal 3.9 h night, and pins WORN_SKIN_TEMP_C's upper edge with a literal 41 C. \
         Neither constant is pinned in the other direction, and WORN_SKIN_TEMP_C's lower edge not at all."
    );
    s
}

// ─── metric 8: illness-watch baseline z-series ─────────────────────────────────────────────────

/// illness.rs:9-18. `zero` / `future` / `pop_sd` are control knobs, not shipped constants.
#[derive(Clone, Copy)]
struct IllParams {
    gap: usize,
    window: usize,
    min_nights: usize,
    sd_floor: f64,
    zero: bool,
    future: bool,
    pop_sd: bool,
}

impl IllParams {
    fn shipped() -> Self {
        Self {
            gap: BASELINE_GAP_NIGHTS,
            window: BASELINE_WINDOW_NIGHTS,
            min_nights: MIN_BASELINE_NIGHTS,
            sd_floor: SD_FLOOR,
            zero: false,
            future: false,
            pop_sd: false,
        }
    }
}

fn ill_window(p: &IllParams, i: usize) -> std::ops::Range<usize> {
    if p.future {
        let s = i + p.gap;
        return s..s + p.window;
    }
    let end = i.saturating_sub(p.gap);
    end.saturating_sub(p.window)..end
}

fn ill_z_at(p: &IllParams, values: &[Option<f64>], i: usize) -> Option<f64> {
    let value = *values.get(i)?.as_ref()?;
    if p.zero {
        return Some(0.0);
    }
    let r = ill_window(p, i);
    let lo = r.start.min(values.len());
    let hi = r.end.min(values.len()).max(lo);
    let xs: Vec<f64> = values[lo..hi].iter().flatten().copied().collect();
    if xs.len() < p.min_nights {
        return None;
    }
    let sd = if p.pop_sd { stats::population_sd(&xs) } else { stats::sample_sd(&xs) };
    Some((value - stats::mean(&xs)) / sd.max(p.sd_floor))
}

/// illness.rs:72-73 — 52 calm nights then an eight-night rise.
fn ill_fire_series() -> Vec<Option<f64>> {
    let mut vs: Vec<Option<f64>> =
        (0..52).map(|i| Some(if i % 2 == 0 { 54.0 } else { 56.0 })).collect();
    vs.extend((0..8).map(|_| Some(62.0)));
    vs
}

/// illness.rs:87-89 — 48 calm nights with a raised last night.
fn ill_holey_series() -> Vec<Option<f64>> {
    let mut vs: Vec<Option<f64>> =
        (0..48).map(|i| Some(if i % 2 == 0 { 54.0 } else { 56.0 })).collect();
    let i = vs.len() - 1;
    vs[i] = Some(60.0);
    vs
}

fn ill_gate_holds(p: &IllParams) -> bool {
    let r = ill_window(p, 46);
    if r.start != GATE_WINDOW_START || r.end != GATE_WINDOW_END || r.len() != p.window {
        return false;
    }
    if r.contains(&43) || r.contains(&45) || r.contains(&46) {
        return false;
    }
    let vs: Vec<Option<f64>> = (0..16).map(|i| Some(50.0 + i as f64)).collect();
    if ill_window(p, 2) != (0..0) || ill_z_at(p, &vs, 15).is_some() || ill_z_at(p, &vs, 0).is_some() {
        return false;
    }
    let fire = ill_fire_series();
    let i = fire.len() - 1;
    let Some(z) = ill_z_at(p, &fire, i) else { return false };
    if z <= GATE_ILLNESS_FIRE_Z {
        return false;
    }
    let gapless: Vec<f64> = fire[i - p.window..i].iter().flatten().copied().collect();
    let z0 = (62.0 - stats::mean(&gapless)) / stats::sample_sd(&gapless).max(p.sd_floor);
    if z0 >= GATE_ILLNESS_FIRE_Z {
        return false;
    }
    let mut vs = ill_holey_series();
    let i = vs.len() - 1;
    let Some(full) = ill_z_at(p, &vs, i) else { return false };
    for k in ill_window(p, i).take(10) {
        if k < vs.len() {
            vs[k] = None;
        }
    }
    let Some(holey) = ill_z_at(p, &vs, i) else { return false };
    if (full - holey).abs() >= GATE_HOLEY_TOL {
        return false;
    }
    for k in ill_window(p, i).take(20) {
        if k < vs.len() {
            vs[k] = None;
        }
    }
    if ill_z_at(p, &vs, i).is_some() {
        return false;
    }
    let flat: Vec<Option<f64>> = (0..48).map(|_| Some(55.0)).collect();
    ill_z_at(p, &flat, flat.len() - 1) == Some(0.0)
}

fn ill_arm(kind: Kind, name: String, p: &IllParams) -> Arm {
    let fire = ill_fire_series();
    let value = ill_z_at(p, &fire, fire.len() - 1).unwrap_or(f64::NAN);
    Arm::new(kind, name, value, ill_gate_holds(p))
}

fn ill_param_int(t: &mut Table, label: &str, base: usize, set: impl Fn(&mut IllParams, usize)) {
    for d in [1i64, -1] {
        let v = (base as i64 + d).max(0) as usize;
        let mut p = IllParams::shipped();
        set(&mut p, v);
        t.add(ill_arm(Kind::Param, format!("{label} {base} -> {v} ({d:+})"), &p).p(1.0 / base as f64));
    }
}

fn metric_illness_baseline() -> Score {
    let ship = IllParams::shipped();
    assert_eq!(ill_window(&ship, 46), baseline_window(46), "illness window replica differs");
    let fire = ill_fire_series();
    for i in [0usize, 15, 40, fire.len() - 1] {
        assert_eq!(ill_z_at(&ship, &fire, i), baseline_z_at(&fire, i), "illness z replica differs at {i}");
    }
    assert_eq!(baseline_z_series(&fire).len(), fire.len());

    let mut t = Table::new(
        "Illness-watch baseline z-series",
        "illness.rs:53 = 13..43 + :63-64 None + :76 z > 2.0 + :81 gapless < 2.0 + :95 |full-holey| < 1.0 \
         + :100 None + :108 = 0.0",
        "gapped z",
    );
    t.add(ill_arm(Kind::Baseline, "unmutated".into(), &ship));
    t.add(ill_arm(
        Kind::Null,
        "every day scores z = 0".into(),
        &IllParams { zero: true, ..IllParams::shipped() },
    ));
    t.add(ill_arm(
        Kind::Structural,
        "window taken from the FUTURE of the scored day".into(),
        &IllParams { future: true, ..IllParams::shipped() },
    ));
    t.add(ill_arm(
        Kind::Structural,
        "population SD (n) instead of sample SD (n-1)".into(),
        &IllParams { pop_sd: true, ..IllParams::shipped() },
    ));
    t.add(ill_arm(
        Kind::Structural,
        "gap removed, window abuts the scored day".into(),
        &IllParams { gap: 0, ..IllParams::shipped() },
    ));
    ill_param_int(&mut t, "BASELINE_GAP_NIGHTS", BASELINE_GAP_NIGHTS, |p, v| p.gap = v);
    ill_param_int(&mut t, "BASELINE_WINDOW_NIGHTS", BASELINE_WINDOW_NIGHTS, |p, v| p.window = v);
    ill_param_int(&mut t, "MIN_BASELINE_NIGHTS", MIN_BASELINE_NIGHTS, |p, v| p.min_nights = v);
    for &(tag, m) in PM10.iter().chain(TINY.iter()) {
        let v = SD_FLOOR * m;
        t.add(
            ill_arm(
                Kind::Param,
                format!("SD_FLOOR {SD_FLOOR:e} -> {v:e} ({tag})"),
                &IllParams { sd_floor: v, ..IllParams::shipped() },
            )
            .p((m - 1.0).abs()),
        );
    }
    let s = t.finish();
    println!(
        "note: integer tunables have no sub-unit probe, so their floor is +-1 by construction. \
         MIN_BASELINE_NIGHTS is pinned only to the open range (12, 20] by the two holey-window probes."
    );
    s
}

// ─── entry point ───────────────────────────────────────────────────────────────────────────────

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("Recovery / Charge score (0-100 logistic composite)", "constant scorer, every night 50.0", 3.56),
    ("Recovery / Charge score (0-100 logistic composite)", "logistic slope k = 0, every z maps to 50", 3.56),
    ("Recovery band (red / yellow / green)", "constant band, always yellow", 1.8),
    ("Recovery band (red / yellow / green)", "constant band, always red", 1.8),
    ("Recovery band (red / yellow / green)", "red and green labels swapped", 1.8),
    ("Recovery band (red / yellow / green)", "both thresholds shifted +1.0 point", 0.9),
    ("Charge state (5-way floors)", "constant state, always Moderate", 3.6),
    ("Charge state (5-way floors)", "Low and Primed labels swapped", 1.8),
    ("Recovery driver rows (per-driver Shapley share of the swing)", "every row reports 0 points", 10.3),
    ("Recovery driver rows (per-driver Shapley share of the swing)", "every row reports the whole composite swing", 4.5),
    ("Recovery driver rows (per-driver Shapley share of the swing)", "W_HRV and W_RHR swapped (0.55 <-> 0.20)", 6.75),
    ("Recovery Index (overnight HR-decline slope, bpm/hour)", "constant scorer, every night 0.0 bpm/h", 1.8),
    ("Recovery Index (overnight HR-decline slope, bpm/hour)", "input flattened to its own mean bpm", 1.8),
    ("Recovery Index (overnight HR-decline slope, bpm/hour)", "slope sign flipped", 3.6),
    ("Recovery Index (overnight HR-decline slope, bpm/hour)", "input reversed in time", 3.6),
    ("Recovery Index (overnight HR-decline slope, bpm/hour)", "last 10% of the night dropped", 0.00234),
    ("Banked nights (calibration progress count)", "always 0 banked", 1.35),
    ("Banked nights (calibration progress count)", "count every slot, missing nights included", 1.35),
    ("Banked nights (calibration progress count)", "bounds ignored, any present value banks", 0.9),
    ("Banked nights (calibration progress count)", "min and max exchanged", 1.35),
    ("Banked nights (calibration progress count)", "HRV_MIN_MS x1.10 with the probe pinned at the literal 5.0", 0.45),
    ("HRV readiness tier (Primed / Normal / Suppressed)", "every night replaced by the series mean", 8.81),
    ("HRV readiness tier (Primed / Normal / Suppressed)", "series reversed, oldest becomes newest", 18.0),
    ("HRV readiness tier (Primed / Normal / Suppressed)", "newest night dropped (series shifted one night)", 2.64),
    ("Personal baselines (winsorized EWMA centre + abs-dev spread)", "centre frozen at the first valid value", 2.01),
    ("Personal baselines (winsorized EWMA centre + abs-dev spread)", "centre always the metric midpoint", 36.8),
    ("Illness-watch baseline z-series", "every day scores z = 0", 1.05),
    ("Illness-watch baseline z-series", "population SD (n) instead of sample SD (n-1)", 0.0179),
    ("Illness-watch baseline z-series", "gap removed, window abuts the scored day", 0.281),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("Recovery / Charge score (0-100 logistic composite)", "constant scorer, every night the z = 0 anchor", "the arm IS the anchor, so the anchor column cannot move; the nine pinned nights are where it fails"),
    ("Recovery / Charge score (0-100 logistic composite)", "W_HRV and W_SKIN_TEMP swapped (0.55 <-> 0.05)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Recovery / Charge score (0-100 logistic composite)", "HRV z sign flipped, higher HRV reads worse", "measured delta is exactly zero: this mutation does not move the number"),
    ("Recovery / Charge score (0-100 logistic composite)", "HRV term dropped entirely", "measured delta is exactly zero: this mutation does not move the number"),
    ("Recovery / Charge score (0-100 logistic composite)", "every driver weighted equally", "measured delta is exactly zero: this mutation does not move the number"),
    ("Charge state (5-way floors)", "every floor shifted +1.0 point", "measured delta is exactly zero: this mutation does not move the number"),
    ("Recovery driver rows (per-driver Shapley share of the swing)", "sort reversed, smallest mover first", "measured delta is exactly zero: this mutation does not move the number"),
    ("Recovery driver rows (per-driver Shapley share of the swing)", "round-half-up replaced by truncation", "measured delta is exactly zero: this mutation does not move the number"),
    ("Personal baselines (winsorized EWMA centre + abs-dev spread)", "history folded newest first", "measured delta is exactly zero: this mutation does not move the number"),
    ("Personal baselines (winsorized EWMA centre + abs-dev spread)", "winsorisation off (WINSOR_K = 1e9)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Illness-watch baseline z-series", "window taken from the FUTURE of the scored day", "the arm yields no number, so it has no distance from the baseline"),
];

/// Assert one metric's floors, and require every NULL/STRUCTURAL arm to be classified.
fn enforce_floors(metric: &str, base: f64, probes: &[(&str, f64)]) {
    let (mut asserted, mut waived) = (0usize, 0usize);
    let mut breached: Vec<String> = Vec::new();
    let mut unclassified: Vec<&str> = Vec::new();
    for &(arm, value) in probes {
        let floor = FLOORS.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        let waiver = NO_FLOOR.iter().find(|(m, a, _)| *m == metric && *a == arm).map(|t| t.2);
        match (floor, waiver) {
            (Some(_), Some(_)) => breached.push(format!("'{arm}' carries both a floor and a waiver")),
            (Some(d), None) => {
                asserted += 1;
                let moved = (value - base).abs();
                if moved.is_nan() || moved < d {
                    breached.push(format!("'{arm}' moved {moved} against a floor of {d}"));
                }
            }
            (None, Some(w)) => {
                waived += 1;
                println!("   no floor: {arm} — {w}");
            }
            (None, None) => unclassified.push(arm),
        }
    }
    let orphans: Vec<&str> = FLOORS
        .iter()
        .filter(|(m, _, _)| *m == metric)
        .map(|t| t.1)
        .chain(NO_FLOOR.iter().filter(|(m, _, _)| *m == metric).map(|t| t.1))
        .filter(|a| !probes.iter().any(|(p, _)| *p == *a))
        .collect();
    println!("   floors: {asserted} asserted, {waived} un-floorable");
    assert!(
        unclassified.is_empty(),
        "{metric}: probe arms carry neither a floor nor a waiver — classify them: {unclassified:?}"
    );
    assert!(orphans.is_empty(), "{metric}: floor rows match no arm — stale or misspelt: {orphans:?}");
    assert!(breached.is_empty(), "{metric}: SENSITIVITY FLOOR BREACHED — {}", breached.join(" | "));
}

#[test]
#[ignore = "negative-control measurement harness, not CI"]
fn recovery_family_negative_controls() {
    let mut all = vec![metric_charge_score()];
    let (band_s, state_s) = metric_band_and_state();
    all.push(band_s);
    all.push(state_s);
    all.push(metric_driver_rows());
    all.push(metric_recovery_index());
    all.push(metric_banked_nights());
    all.push(metric_hrv_readiness());
    // Personal baselines runs last because its cohort assert aborts the binary when `baselines.rs`
    // gains or loses a test, and that must not cost the other tables their measurement.
    all.push(metric_illness_baseline());
    all.push(metric_personal_baselines());

    let caught: usize = all.iter().map(|s| s.caught).sum();
    let missed: usize = all.iter().map(|s| s.missed).sum();
    let dfloor = all.iter().filter_map(|s| s.delta_floor).fold(f64::INFINITY, f64::min);
    let pfloor = all.iter().filter_map(|s| s.pct_floor).fold(f64::INFINITY, f64::min);
    println!("\n=== RECOVERY FAMILY TOTAL ===");
    println!("caught {caught}, missed {missed}");
    if dfloor.is_finite() {
        println!("smallest caught |delta| across the family: {dfloor:.4}");
    }
    if pfloor.is_finite() {
        println!("smallest caught parameter move across the family: {:.1}%", pfloor * 100.0);
    }
}
