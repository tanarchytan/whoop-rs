//! Negative controls for the SLEEP metric family. The claim under test is NOT "the algorithms are
//! right" — it is "the shipped sleep gates would NOTICE if an algorithm changed". Each metric here is
//! driven three ways: a NULL arm (a scorer that does no work), STRUCTURAL arms (a wrong output SHAPE),
//! and PARAMETER arms (every tunable the algorithm reads, moved +/-10%). Every arm is scored against the
//! shipped gate's own target and tolerance, copied in as a `const` beside the `file:line` it came from,
//! so the control tests the real claim and not a paraphrase.
//!
//! What each table falsifies: "this gate is a regression check". A gate that only a NULL arm can break
//! is a REPRODUCTION check — it proves the harness reaches the algorithm and nothing more. The printed
//! `caught N, missed M` line and the sensitivity floor beneath it are the deliverable.
//!
//! Two rules this file follows deliberately. It asserts ONLY that the baseline reproduces the shipped
//! figure and that at least one NULL arm is rejected: a parameter arm that PASSES is the finding, never
//! a failure. And a NULL arm that passes is printed as `!! BLIND NULL` rather than asserted, because a
//! metric can be definitionally satisfied by a degenerate input (an SRI of +100 for a wearer who is
//! never asleep is arithmetically correct) — that is a statement about the gate's reach, for a later
//! phase to judge, not a bug to fail on here.
//!
//! Every kappa here is FOUR-class (wake/light/deep/REM); no three-class figure belongs beside them.
//!
//! Every control is `#[ignore]`d: they are controls, not CI gates, and the cohort ones need a corpus
//! that lives outside the repo. Run one family at a time with
//!   cargo test --release -p physio-algo --test sensitivity_sleep -- --ignored --nocapture
//! `shipped_gate_constants_match_their_sources` is the exception and runs from a clean checkout: it
//! re-reads every target copied in below out of the file that owns it, so a retune cannot leave a stale
//! number here scoring arms against a gate nobody has.
//!
//! Numbers printed here are wellness estimates, never medical or diagnostic.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::hrv::HrvReadiness;
use physio_algo::nap::{evaluate as nap_evaluate, NapConfig, NapVerdict};
use physio_algo::recovery::{recovery, DriverBaseline, RecoveryInput};
use physio_algo::rest::{personal_sleep_need_hours, rest};
use physio_algo::sleep::{
    analyze, detect_sessions_with, main_night_group_indices, main_night_index, prepare_v2,
    refine_wake_with, stage_v2, stage_v2_prepared, stage_v2_with, AccelSample, DetectParams,
    HrSample as SleepHr, NightBlock, Params, RefineParams, RrRun, SleepInput, SleepStage, SleepStreams,
    StageSegment, StepSample,
};
use physio_algo::sleep_debt::ledger;
use physio_algo::sleep_regularity::{coverage_spans, epoch_grid, sleep_regularity_index, EpochState, EPOCHS_PER_DAY};
use physio_algo::workout::GravitySample;
use physio_algo::HrSample;

// ---------------------------------------------------------------------------------------------
// Shipped gates, copied verbatim from the files that own them.
// ---------------------------------------------------------------------------------------------

/// The three cohort targets and their sizes, owned by `dataset_parity.rs`'s `assert_cohort` calls, and
/// re-read from that file by `shipped_gate_constants_match_their_sources`. Named by symbol, not by
/// line: a line reference in a control is stale the first time the file it points at is edited.
const DREAMT_KAPPA: f64 = 0.311;
const AAUWSS_KAPPA: f64 = 0.412;
/// `sleep-accel` carries no R-R on any of its 31 nights, so this is V2 WITHOUT its respiratory
/// channel — cardiac and motion only. `sensitivity_stage_v2_sleep_accel_no_respiratory_channel`
/// asserts that from the algorithm's side, by requiring both `resp_weight` arms to move it by zero.
const SLEEP_ACCEL_KAPPA: f64 = 0.379;
/// The one tolerance all three cohort gates use, `dataset_parity.rs`'s `KAPPA_TOL`.
const KAPPA_TOL: f64 = 0.008;
/// The cohort sizes the targets were measured over. A partly present corpus is a different cohort, not
/// a passing gate.
const DREAMT_N: usize = 100;
const AAUWSS_N: usize = 13;
const SLEEP_ACCEL_N: usize = 31;
/// Nights of each cohort that carry R-R at all — `dataset_parity.rs`'s `PSG_REQUIRED` third column.
const DREAMT_RR_NIGHTS: usize = 100;
const AAUWSS_RR_NIGHTS: usize = 13;
const SLEEP_ACCEL_RR_NIGHTS: usize = 0;

/// crates/physio-algo/src/sleep/golden_tests.rs:52-59 — the frozen 6-segment hypnogram, offsets from
/// the crafted night's start.
const GOLDEN_HYPNOGRAM: [(i64, i64, SleepStage); 6] = [
    (0, 5070, SleepStage::Deep),
    (5070, 5280, SleepStage::Light),
    (5280, 5550, SleepStage::Rem),
    (5550, 10740, SleepStage::Light),
    (10740, 16290, SleepStage::Rem),
    (16290, 21600, SleepStage::Wake),
];

/// crates/physio-algo/src/sleep/golden_tests.rs:150 — the golden night's trailing wake run, in seconds.
const GOLDEN_TRAILING_WAKE_S: i64 = 5_309;

/// crates/physio-algo/src/sleep/refine.rs:393 — wake seconds the shipped refinement leaves on the
/// three-run window.
const REFINE_SHIPPED_WAKE_S: i64 = 780;

/// crates/physio-algo/src/sleep/golden_tests.rs:178 — the detector's simple-still-night contract.
const DETECT_MIN_SPAN_S: i64 = 3600;

/// crates/physio-algo/src/sleep/mainnight.rs:511-517 — a 70-min overnight gap bridges, a 95-min one
/// does not, so the cutoff must land inside this half-open band (minutes).
const BRIDGE_CUTOFF_MIN_LO: i64 = 70;
const BRIDGE_CUTOFF_MIN_HI: i64 = 95;

/// crates/physio-algo/src/rest.rs:80
const REST_PERFECT_NIGHT_FLOOR: f64 = 85.0;
/// crates/physio-algo/src/rest.rs:73
const PERSONAL_NEED_TARGET: f64 = 8.5;
const PERSONAL_NEED_TOL: f64 = 1e-9;
/// crates/physio-algo/src/sleep_debt.rs:80
const DEBT_BALANCE_TOL: f64 = 0.1;
/// crates/physio-algo/src/sleep_debt.rs:98
const DEBT_WINDOW_NIGHTS: usize = 14;
/// crates/physio-algo/src/sleep_regularity.rs:137
const SRI_PERFECT: f64 = 100.0;
const SRI_TOL: f64 = 1e-9;
/// crates/physio-algo/src/nap.rs:227-272 — all six shipped scenarios must return their verdict.
const NAP_VERDICT_SHARE: f64 = 1.0;
/// crates/physio-algo/tests/sleep_error_propagation.rs:137-138
const CHAIN_GAIN_CEIL: f64 = 1.0;
const CHAIN_GAIN_FLOOR: f64 = 0.0;

const REF_MIDNIGHT: i64 = 1_749_513_600;
const SECONDS_PER_DAY: i64 = 86_400;

// ---------------------------------------------------------------------------------------------
// The arm table: one row per arm, PASS/FAIL of the SHIPPED gate, then caught/missed.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Baseline,
    Null,
    Structural,
    Parameter,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Baseline => "base ",
            Kind::Null => "null ",
            Kind::Structural => "struct",
            Kind::Parameter => "param",
        }
    }
}

struct Row {
    arm: String,
    kind: Kind,
    value: f64,
    pass: bool,
}

struct Table {
    metric: &'static str,
    gate: String,
    rows: Vec<Row>,
}

impl Table {
    fn new(metric: &'static str, gate: impl Into<String>) -> Self {
        Table { metric, gate: gate.into(), rows: Vec::new() }
    }

    fn push(&mut self, arm: impl Into<String>, kind: Kind, value: f64, pass: bool) {
        self.rows.push(Row { arm: arm.into(), kind, value, pass });
    }

    /// `(arm, delta from the baseline)` for every arm whose label contains `needle`. Lets a caller
    /// assert that one named knob does or does not reach the metric, which the caught/missed tally
    /// cannot say on its own.
    fn deltas_matching(&self, needle: &str) -> Vec<(&str, f64)> {
        let base = self.rows[0].value;
        self.rows
            .iter()
            .filter(|r| r.kind != Kind::Baseline && r.arm.contains(needle))
            .map(|r| (r.arm.as_str(), r.value - base))
            .collect()
    }

    /// Print the sheet, then assert the two things that must hold for the harness to be trustworthy.
    fn finish(self) {
        assert!(!self.rows.is_empty(), "{}: no arms were run", self.metric);
        assert!(self.rows[0].kind == Kind::Baseline, "{}: row 0 must be the baseline", self.metric);
        let base = self.rows[0].value;

        println!();
        println!("=== {} ===", self.metric);
        println!("shipped gate: {}", self.gate);
        println!("{:<62} {:>11} {:>11}  {:<5}  verdict", "arm", "value", "delta", "gate");
        for r in &self.rows {
            let delta = r.value - base;
            let verdict = match (r.kind, r.pass) {
                (Kind::Baseline, true) => "(expected)",
                (Kind::Baseline, false) => "!! BASELINE DOES NOT REPRODUCE",
                (Kind::Null, true) => "!! BLIND NULL — the gate does not see a no-work scorer",
                (_, false) => "<-- caught",
                (_, true) => "MISSED",
            };
            println!(
                "{:<62} {:>11.4} {:>+11.4}  {:<5}  {}",
                format!("[{}] {}", r.kind.tag(), r.arm),
                r.value,
                delta,
                if r.pass { "PASS" } else { "FAIL" },
                verdict
            );
        }

        let probes: Vec<&Row> = self.rows.iter().filter(|r| r.kind != Kind::Baseline).collect();
        let caught: Vec<&&Row> = probes.iter().filter(|r| !r.pass).collect();
        let missed: Vec<&&Row> = probes.iter().filter(|r| r.pass).collect();
        println!("caught {}, missed {}", caught.len(), missed.len());

        let floor = caught
            .iter()
            .map(|r| (r.value - base).abs())
            .filter(|d| d.is_finite() && *d > 0.0)
            .fold(f64::INFINITY, f64::min);
        if floor.is_finite() {
            println!("sensitivity floor: the smallest delta this gate catches is {floor:.4}");
        } else {
            println!("sensitivity floor: NONE — no arm with a non-zero delta was caught");
        }
        let ceiling = missed
            .iter()
            .map(|r| (r.value - base).abs())
            .filter(|d| d.is_finite())
            .fold(0.0f64, f64::max);
        println!("blind ceiling:     the largest delta this gate lets through is {ceiling:.4}");

        let blind: Vec<&str> =
            self.rows.iter().filter(|r| r.kind == Kind::Null && r.pass).map(|r| r.arm.as_str()).collect();
        if !blind.is_empty() {
            println!("!! BLIND NULLS ({}): {:?}", blind.len(), blind);
        }

        assert!(
            self.rows[0].pass,
            "{}: the BASELINE must reproduce the shipped figure — got {} = {}",
            self.metric, self.rows[0].arm, base
        );
        let nulls: Vec<&Row> = self.rows.iter().filter(|r| r.kind == Kind::Null).collect();
        assert!(!nulls.is_empty(), "{}: a negative control needs at least one NULL arm", self.metric);
        assert!(
            nulls.iter().any(|r| !r.pass),
            "{}: CRITICAL — every NULL arm PASSES the shipped gate, so the gate does not reach the \
             algorithm and proves nothing",
            self.metric
        );

        let probes: Vec<(&str, f64)> = self
            .rows
            .iter()
            .filter(|r| matches!(r.kind, Kind::Null | Kind::Structural))
            .map(|r| (r.arm.as_str(), r.value))
            .collect();
        enforce_floors(self.metric, base, &probes);
    }
}

/// The 1-based line `needle` sits on, so a gate string can name a real location instead of a number
/// that was true once. Fatal when the needle is gone: that means the gate this control mirrors has
/// been renamed or deleted, and the control is now mirroring nothing.
fn line_of(src: &str, needle: &str) -> usize {
    src.lines()
        .position(|l| l.contains(needle))
        .map(|i| i + 1)
        .unwrap_or_else(|| panic!("the shipped gate this control mirrors is gone: {needle:?}"))
}

/// Every target copied in at the head of this file, re-read from the file that owns it. Runs from a
/// clean checkout — it reads source text, never fixtures — so a retune on either side cannot leave a
/// control scoring arms against a gate nobody has.
#[test]
fn shipped_gate_constants_match_their_sources() {
    let parity = include_str!("dataset_parity.rs");
    for want in [
        format!("const KAPPA_TOL: f64 = {KAPPA_TOL};"),
        format!("assert_cohort(\"dreamt\", &s, {DREAMT_KAPPA}, {DREAMT_N});"),
        format!("assert_cohort(\"aauwss\", &s, {AAUWSS_KAPPA}, {AAUWSS_N});"),
        format!("assert_cohort(\"sleep-accel\", &s, {SLEEP_ACCEL_KAPPA}, {SLEEP_ACCEL_N});"),
        format!(
            "[(\"dreamt\", {DREAMT_N}, {DREAMT_RR_NIGHTS}), (\"aauwss\", {AAUWSS_N}, \
             {AAUWSS_RR_NIGHTS}), (\"sleep-accel\", {SLEEP_ACCEL_N}, {SLEEP_ACCEL_RR_NIGHTS})]"
        ),
    ] {
        assert!(parity.contains(&want), "dataset_parity.rs no longer contains {want:?}");
    }

    let mainnight = include_str!("../src/sleep/mainnight.rs");
    assert!(
        mainnight.contains("fn realistic_nap_never_beats_real_night_cold_start"),
        "the main-night control mirrors a shipped test that no longer exists"
    );
    assert_eq!(
        nap_vs_night_cases().len(),
        2_160,
        "the main-night control's pairing count is the number its gate string quotes"
    );
}

fn within(v: f64, target: f64, tol: f64) -> bool {
    (v - target).abs() < tol
}

/// Deterministic in-place shuffle, so a "shuffled input" arm is reproducible run to run.
fn lcg_shuffle<T>(v: &mut [T], mut seed: u64) {
    for i in (1..v.len()).rev() {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let j = ((seed >> 33) as usize) % (i + 1);
        v.swap(i, j);
    }
}

// ---------------------------------------------------------------------------------------------
// The V2 recipe's tunable surface, one knob at a time.
// ---------------------------------------------------------------------------------------------

type Get = fn(&Params) -> f64;
type Set = fn(&mut Params, f64);

/// Scale one transition row's diagonal and renormalise the row, so it stays a distribution (the golden
/// pins each row summing to 1.0 at golden_tests.rs:72-75).
fn bump_transition(p: &mut Params, row: usize, k: f64) {
    p.transition[row][row] *= k;
    let s: f64 = p.transition[row].iter().sum();
    for v in p.transition[row].iter_mut() {
        *v /= s;
    }
}

/// Every scalar coefficient `Params::SHIPPED` carries, with a reader and a multiplying writer.
fn knobs() -> Vec<(&'static str, Get, Set)> {
    vec![
        ("deep_hrv", |p| p.deep_hrv, |p, k| p.deep_hrv *= k),
        ("deep_hr", |p| p.deep_hr, |p, k| p.deep_hr *= k),
        ("deep_motion", |p| p.deep_motion, |p, k| p.deep_motion *= k),
        ("rem_hrv", |p| p.rem_hrv, |p, k| p.rem_hrv *= k),
        ("rem_motion", |p| p.rem_motion, |p, k| p.rem_motion *= k),
        ("rem_hr", |p| p.rem_hr, |p, k| p.rem_hr *= k),
        ("awake_motion", |p| p.awake_motion, |p, k| p.awake_motion *= k),
        ("awake_hrv", |p| p.awake_hrv, |p, k| p.awake_hrv *= k),
        ("awake_hr", |p| p.awake_hr, |p, k| p.awake_hr *= k),
        ("awake_deadzone", |p| p.awake_deadzone, |p, k| p.awake_deadzone *= k),
        ("deep_gate_thresh", |p| p.deep_gate_thresh, |p, k| p.deep_gate_thresh *= k),
        ("deep_gate_slope", |p| p.deep_gate_slope, |p, k| p.deep_gate_slope *= k),
        ("jerk_move_mult", |p| p.jerk_move_mult, |p, k| p.jerk_move_mult *= k),
        ("jerk_gate_mult", |p| p.jerk_gate_mult, |p, k| p.jerk_gate_mult *= k),
        ("motion_gate_boost", |p| p.motion_gate_boost, |p, k| p.motion_gate_boost *= k),
        ("resp_weight", |p| p.resp_weight, |p, k| p.resp_weight *= k),
        ("base_rate[deep]", |p| p.base_rate[0], |p, k| p.base_rate[0] *= k),
        ("base_rate[rem]", |p| p.base_rate[1], |p, k| p.base_rate[1] *= k),
        ("base_rate[light]", |p| p.base_rate[2], |p, k| p.base_rate[2] *= k),
        ("base_rate[awake]", |p| p.base_rate[3], |p, k| p.base_rate[3] *= k),
        ("cycle_deep_scale", |p| p.cycle_deep_scale, |p, k| p.cycle_deep_scale *= k),
        ("cycle_deep_decay", |p| p.cycle_deep_decay, |p, k| p.cycle_deep_decay *= k),
        ("cycle_rem_scale", |p| p.cycle_rem_scale, |p, k| p.cycle_rem_scale *= k),
        ("cycle_rem_early_frac", |p| p.cycle_rem_early_frac, |p, k| p.cycle_rem_early_frac *= k),
        ("cycle_rem_early_penalty", |p| p.cycle_rem_early_penalty, |p, k| p.cycle_rem_early_penalty *= k),
        ("cycle_rem_onset_minutes", |p| p.cycle_rem_onset_minutes, |p, k| p.cycle_rem_onset_minutes *= k),
        ("cycle_rem_ramp_cap", |p| p.cycle_rem_ramp_cap, |p, k| p.cycle_rem_ramp_cap *= k),
        ("transition[deep][deep]", |p| p.transition[0][0], |p, k| bump_transition(p, 0, k)),
        ("transition[rem][rem]", |p| p.transition[1][1], |p, k| bump_transition(p, 1, k)),
        ("transition[light][light]", |p| p.transition[2][2], |p, k| bump_transition(p, 2, k)),
        ("transition[awake][awake]", |p| p.transition[3][3], |p, k| bump_transition(p, 3, k)),
    ]
}

/// One recipe arm: its label, whether the change moves the extracted FEATURES (only `jerk_move_mult`
/// does, per `sleep::v2::Prepared`), and the recipe itself.
struct RecipeArm {
    name: String,
    kind: Kind,
    params: Params,
    reprepare: bool,
}

/// Every parameter arm: each knob at +10% and -10%, plus two +0.5% floor probes and the one boolean.
fn recipe_arms() -> Vec<RecipeArm> {
    let mut out = Vec::new();
    for (name, get, set) in knobs() {
        for k in [1.1f64, 0.9] {
            let mut p = Params::SHIPPED;
            set(&mut p, k);
            out.push(RecipeArm {
                name: format!("{name} {:.4} -> {:.4} (x{k:.2})", get(&Params::SHIPPED), get(&p)),
                kind: Kind::Parameter,
                params: p,
                reprepare: name == "jerk_move_mult",
            });
        }
    }
    // Floor probes: half a percent on the two knobs the worked example found the gate blindest to.
    for (name, get, set) in knobs() {
        if name != "deep_gate_thresh" && name != "base_rate[light]" {
            continue;
        }
        let mut p = Params::SHIPPED;
        set(&mut p, 1.005);
        out.push(RecipeArm {
            name: format!("{name} {:.4} -> {:.4} (x1.005, floor probe)", get(&Params::SHIPPED), get(&p)),
            kind: Kind::Parameter,
            params: p,
            reprepare: false,
        });
    }
    let mut clock = Params::SHIPPED;
    clock.cycle_clock_from_onset = true;
    out.push(RecipeArm {
        name: "cycle_clock_from_onset false -> true (boolean flip)".to_string(),
        kind: Kind::Parameter,
        params: clock,
        reprepare: false,
    });
    out
}

// ---------------------------------------------------------------------------------------------
// Output transforms — the NULL and STRUCTURAL arms, applied to a stage sequence.
// Stage integers are the harness convention: 0 wake, 1 light, 2 deep, 3 rem.
// ---------------------------------------------------------------------------------------------

fn out_all_light(p: &mut [i32]) {
    p.iter_mut().for_each(|v| *v = 1);
}
fn out_shuffle(p: &mut [i32]) {
    lcg_shuffle(p, 0x5EED_1234);
}
fn out_swap_deep_rem(p: &mut [i32]) {
    p.iter_mut().for_each(|v| {
        *v = match *v {
            2 => 3,
            3 => 2,
            x => x,
        }
    });
}
fn out_swap_wake_light(p: &mut [i32]) {
    p.iter_mut().for_each(|v| {
        *v = match *v {
            0 => 1,
            1 => 0,
            x => x,
        }
    });
}
fn out_reverse(p: &mut [i32]) {
    p.reverse();
}

/// Shift the sequence forward by `n` samples, holding the first value.
fn shift_by(p: &mut [i32], n: usize) {
    if p.is_empty() {
        return;
    }
    let head = p[0];
    // Equivalent to n rounds of "insert the original head at 0, drop the last": the series slides
    // right by n and the vacated head positions all carry the ORIGINAL first label. Rotating keeps
    // the length fixed, which is what lets every arm take a slice.
    let k = n.min(p.len());
    p.rotate_right(k);
    p[..k].iter_mut().for_each(|v| *v = head);
}

/// Blank the last tenth to Light.
fn drop_tail_tenth(p: &mut [i32]) {
    let n = p.len();
    let k = n - n / 10;
    for v in p[k..].iter_mut() {
        *v = 1;
    }
}

fn out_shift_one_epoch(p: &mut [i32]) {
    shift_by(p, 1);
}
fn out_drop_tail(p: &mut [i32]) {
    drop_tail_tenth(p);
}

/// One output arm: its label, its kind, and the mutation it applies to the stage series.
type OutputArm = (&'static str, Kind, fn(&mut [i32]));

/// The same arm shape when the mutation has to be boxed (per-second goldens build theirs at runtime).
type BoxedArm<'a> = (&'a str, Kind, Box<dyn Fn(&mut [i32])>);

/// One PARAMETER knob on a params struct: its label, a scaler, and a reader for the current value.
type Knob<P> = (&'static str, fn(&mut P, f64), fn(&P) -> f64);

/// A knob reached only through the INPUT, when the real constant is private: label plus scaler.
type Proxy = (&'static str, fn(&mut RestInput, f64));

/// The epoch-indexed NULL and STRUCTURAL arms every hypnogram gate is driven with.
fn output_arms() -> Vec<OutputArm> {
    vec![
        ("output: every epoch = Light", Kind::Null, out_all_light as fn(&mut [i32])),
        ("output: labels shuffled in place", Kind::Null, out_shuffle),
        ("output: Deep <-> REM swapped", Kind::Structural, out_swap_deep_rem),
        ("output: Wake <-> Light swapped", Kind::Structural, out_swap_wake_light),
        ("output: shifted +1 epoch", Kind::Structural, out_shift_one_epoch),
        ("output: reversed", Kind::Structural, out_reverse),
        ("output: last 10% blanked to Light", Kind::Structural, out_drop_tail),
    ]
}

// ---------------------------------------------------------------------------------------------
// Metric 1 — sleep stage hypnogram against the PSG cohorts (dataset kappa).
// ---------------------------------------------------------------------------------------------

const DEFAULT_ROOT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../../sleep-benchmark/fixtures_multi_clean3");

fn fixtures_root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_ROOT))
}

/// An unreadable stream is FATAL and names itself. A control that silently scored a degraded cohort
/// would report a sensitivity floor for a corpus nobody has.
fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("FIXTURE ABSENT OR UNREADABLE: {} ({e})", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|c| c.trim().parse::<f64>().expect("fixture cell must be a number")).collect())
        .collect()
}

struct Night {
    input: SleepInput,
    w0: i64,
    n_epochs: usize,
    truth: BTreeMap<usize, i32>,
}

fn load_night(dir: &Path) -> Night {
    let meta = fs::read_to_string(dir.join("meta.txt"))
        .unwrap_or_else(|e| panic!("FIXTURE ABSENT: meta.txt under {} ({e})", dir.display()));
    let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().expect("meta cell")).collect();
    let (w0, w1, n_epochs) = (m[1], m[2], m[3] as usize);

    let accel = read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    let hr = read_csv(&dir.join("hr.csv")).iter().map(|r| SleepHr { ts: r[0] as i64, bpm: r[1] as u16 }).collect();

    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(last) if last.ts == ts => last.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }

    let mut truth = BTreeMap::new();
    for row in read_csv(&dir.join("truth.csv")) {
        truth.insert(row[0] as usize, row[1] as i32);
    }
    Night { input: SleepInput { start: w0, end: w1, hr, rr, accel }, w0, n_epochs, truth }
}

fn stage_to_int(s: SleepStage) -> i32 {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// Probe each labelled epoch's midpoint against the tiled segments — the harness rule dataset_parity
/// uses, copied so both read the same number off the same staging.
fn predict_epochs(segs: &[StageSegment], w0: i64, n_epochs: usize) -> Vec<i32> {
    let fallback = segs.last().map(|s| s.stage).unwrap_or(SleepStage::Light);
    (0..n_epochs)
        .map(|k| {
            let mid = w0 + k as i64 * 30 + 15;
            let stage = segs.iter().find(|s| s.start <= mid && mid < s.end).map(|s| s.stage).unwrap_or(fallback);
            stage_to_int(stage)
        })
        .collect()
}

use physio_algo::sleep::metrics::kappa4 as cohen_kappa;

/// Drive every arm over one cohort. Each night is read once and prepared once; only `jerk_move_mult`
/// changes the extracted features, so every other recipe arm re-labels the same `Prepared`. The R-R
/// night count is asserted here because it decides which HALF of V2 the whole sheet below measures.
fn stage_sweep(ds: &str, target: f64, expect_n: usize, expect_rr: usize) -> Table {
    let root = fixtures_root().join(ds);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("FIXTURE ABSENT: dataset dir {} ({e})", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.join("meta.txt").exists())
        .collect();
    dirs.sort();

    let outs = output_arms();
    let recipes = recipe_arms();
    let n_arms = 1 + outs.len() + recipes.len();
    let mut cms = vec![[[0i64; 4]; 4]; n_arms];
    let (mut subjects, mut unlabelled, mut rr_nights) = (0usize, 0usize, 0usize);

    for dir in &dirs {
        let night = load_night(dir);
        // Counted, not dropped in silence: on a PSG cohort a labelless night is a broken fixture.
        if night.truth.is_empty() {
            unlabelled += 1;
            continue;
        }
        subjects += 1;
        if !night.input.rr.is_empty() {
            rr_nights += 1;
        }
        let prep = prepare_v2(&night.input, &Params::SHIPPED);
        let base_segs = stage_v2_prepared(&prep, &Params::SHIPPED);
        // The sweep re-labels one `Prepared` per night, which is only valid because `jerk_move_mult` is
        // the sole knob that moves the extracted features. Prove that on the first night rather than
        // trust the doc comment.
        if subjects == 1 {
            assert_eq!(
                stage_v2(&night.input),
                base_segs,
                "{ds}: re-labelling a Prepared under SHIPPED must equal a full stage_v2 pass"
            );
        }
        let base = predict_epochs(&base_segs, night.w0, night.n_epochs);

        let mut preds: Vec<Vec<i32>> = Vec::with_capacity(n_arms);
        preds.push(base.clone());
        for (_, _, f) in &outs {
            let mut p = base.clone();
            f(&mut p);
            preds.push(p);
        }
        for arm in &recipes {
            let segs = if arm.reprepare {
                stage_v2_with(&night.input, &arm.params)
            } else {
                stage_v2_prepared(&prep, &arm.params)
            };
            preds.push(predict_epochs(&segs, night.w0, night.n_epochs));
        }

        for (a, pred) in preds.iter().enumerate() {
            for (k, &t) in &night.truth {
                if *k < pred.len() && (0..4).contains(&t) {
                    cms[a][t as usize][pred[*k] as usize] += 1;
                }
            }
        }
    }

    assert_eq!(
        unlabelled, 0,
        "{ds}: {unlabelled} fixture nights carry an empty truth.csv and were dropped — this sheet \
         would report a sensitivity floor for a corpus nobody has"
    );
    assert_eq!(
        subjects, expect_n,
        "{ds}: scored {subjects} subjects, not the {expect_n} the target was measured over — the \
         cohort on disk is not the cohort the gate was written for"
    );
    assert_eq!(
        rr_nights, expect_rr,
        "{ds}: {rr_nights} of {subjects} nights carry R-R, not the {expect_rr} this sheet is labelled \
         for — `resp_weight` is the only V2 term that reads R-R, so this changes which half of V2 \
         every arm below measures"
    );

    let gate = format!(
        "dataset_parity.rs — |4-class kappa - {target}| < {KAPPA_TOL} over exactly {expect_n} \
         subjects (n={subjects} measured, {rr_nights} carrying R-R)"
    );
    let mut t = Table::new(cohort_name(ds), gate);
    let k0 = cohen_kappa(&cms[0]);
    t.push("baseline (unmutated stage_v2)", Kind::Baseline, k0, within(k0, target, KAPPA_TOL));
    for (i, (name, kind, _)) in outs.iter().enumerate() {
        let k = cohen_kappa(&cms[1 + i]);
        t.push(*name, *kind, k, within(k, target, KAPPA_TOL));
    }
    for (i, arm) in recipes.iter().enumerate() {
        let k = cohen_kappa(&cms[1 + outs.len() + i]);
        t.push(format!("param: {}", arm.name), arm.kind, k, within(k, target, KAPPA_TOL));
    }
    t
}

/// The metric names, shared by the sheets and by the floor tables that key off them. Named rather than
/// spelt out twice: a renamed cohort silently orphaned every one of its floors when they were literals.
const DREAMT_METRIC: &str = "sleep stage hypnogram — DREAMT PSG 4-class kappa";
const AAUWSS_METRIC: &str = "sleep stage hypnogram — AAUWSS PSG 4-class kappa";
const SLEEP_ACCEL_METRIC: &str =
    "sleep stage hypnogram — sleep-accel PSG 4-class kappa, NO respiratory channel";

fn cohort_name(ds: &str) -> &'static str {
    match ds {
        "dreamt" => DREAMT_METRIC,
        "aauwss" => AAUWSS_METRIC,
        _ => SLEEP_ACCEL_METRIC,
    }
}

/// Both `resp_weight` arms of a finished sweep, as deltas from the baseline. `resp_weight` is the only
/// V2 term that reads R-R, so a cohort with no R-R must show exactly zero here and one with R-R must
/// not — which is how each cohort test proves its own name from the algorithm's side.
fn resp_weight_deltas(t: &Table) -> Vec<(&str, f64)> {
    let arms = t.deltas_matching("resp_weight");
    assert_eq!(arms.len(), 2, "expected the two resp_weight parameter arms, got {arms:?}");
    arms
}

// ── Sensitivity floors ─────────────────────────────────────────────────────────────────────────

/// `(metric, arm, minimum |delta| from the baseline)`. A floor asserts the arm still MOVES the number,
/// which is what catches an algorithm that stopped being reached; each is 0.45x the delta measured
/// 2026-08-02, so it sits well below the observed move and well above zero.
const FLOORS: &[(&str, &str, f64)] = &[
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: every second = Light", 0.337),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: labels shuffled in place", 0.335),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: Deep <-> REM swapped", 0.226),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: Wake <-> Light swapped", 0.223),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: shifted +1 epoch (30 s)", 0.0031),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: shifted +1 second", 9.0e-05),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: reversed", 0.437),
    ("sleep frozen-golden hypnogram (whole V2 recipe)", "output: last 10% blanked to Light", 0.045),
    ("in-bed span detection (sleep session boundaries)", "input: gravity replaced by constant motion", 3230.0),
    ("in-bed span detection (sleep session boundaries)", "input: gravity replaced by shuffled motion vectors", 3230.0),
    ("in-bed span detection (sleep session boundaries)", "input: same night shifted into the 12:00-14:00 daytime band", 3230.0),
    ("in-bed span detection (sleep session boundaries)", "input: night truncated to 45 min (under min_sleep_min)", 3230.0),
    ("main-night bridging (overnight night-tail cutoff)", "output: never bridge", 40.0),
    ("main-night bridging (overnight night-tail cutoff)", "output: always bridge", 27.4),
    ("main-night bridging (overnight night-tail cutoff)", "input: 13:00 onset — the short daytime bridge, not the night tail", 13.5),
    ("main-night selection", "output: always pick block 0", 0.45),
    ("main-night selection", "output: always pick the later onset", 0.45),
    ("main-night selection", "input: nap onsets shifted +6 h (proxy for the private ALIGNMENT_* window)", 0.000405),
    ("main-night selection", "input: nap onsets shifted -6 h (same proxy, other direction)", 0.00063),
    ("nap detection (tri-state verdict)", "output: every verdict = Inconclusive", 0.225),
    ("nap detection (tri-state verdict)", "output: every verdict = Nap", 0.374),
    ("nap detection (tri-state verdict)", "output: every verdict = None", 0.3),
    ("nap detection (tri-state verdict)", "output: None <-> Inconclusive swapped", 0.374),
    ("nap detection (tri-state verdict)", "output: Nap <-> None swapped", 0.225),
    ("nap detection (tri-state verdict)", "input: gravity thinned to 1 in 2 (density proxy)", 0.075),
    ("nap detection (tri-state verdict)", "input: gravity thinned to 1 in 4 (density proxy)", 0.225),
    ("personal sleep need (hours)", "output: constant 7.5 (the floor)", 0.45),
    ("personal sleep need (hours)", "output: constant 8.0 (the default need)", 0.225),
    ("personal sleep need (hours)", "input: empty history", 0.45),
    ("personal sleep need (hours)", "input: a short-sleep window (must land on the floor)", 0.45),
    ("Rest (sleep-performance composite 0-100)", "output: constant 50", 22.0),
    ("Rest (sleep-performance composite 0-100)", "output: constant 90", 4.05),
    ("Rest (sleep-performance composite 0-100)", "input: efficiency zeroed", 8.55),
    ("Rest (sleep-performance composite 0-100)", "input: deep zeroed (the DEEP_FLOOR_FACTOR half)", 6.75),
    ("Rest (sleep-performance composite 0-100)", "input: night halved to 4 h", 11.2),
    ("sleep-debt ledger (rolling balance minutes)", "output: constant 999 balance", 449.0),
    ("sleep-debt ledger (rolling balance minutes)", "input: the surplus night dropped", 54.0),
    ("sleep-debt ledger (trailing window edge)", "output: constant 0 nights", 6.3),
    ("sleep-debt ledger (trailing window edge)", "input: every third night has no sleep data", 0.45),
    ("sleep-error propagation into Rest and Charge (chain gain)", "output: constant 0 gain (the chain swallows the error)", 0.11),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: a 0 pp error (nothing was hurt)", 0.11),
    ("sleep-error propagation into Rest and Charge (chain gain)", "output: constant 5.0 gain (the chain amplifies)", 2.13),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 10 pp Light -> Deep", 0.195),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 10 pp Light -> REM", 0.0323),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 6 pp Wake -> Light (all the Wake this night has is 6.25 pp)", 0.0482),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 20 pp Light -> Wake (gain still divided by 10)", 0.109),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 5 pp Light -> Wake (gain still divided by 10)", 0.0551),
    ("sleep-error propagation into Rest and Charge (chain gain)", "input: 1 pp Light -> Wake (gain still divided by 10)", 0.0993),
    ("Sleep Regularity Index", "input: epoch states shuffled within each day", 40.5),
    ("Sleep Regularity Index", "input: a one-minute-per-day drift", 0.125),
    ("Sleep Regularity Index", "input: bedtime alternating by 4 h", 45.0),
    ("Sleep Regularity Index", "input: a perfectly anti-phase schedule", 90.0),
    (AAUWSS_METRIC, "output: every epoch = Light", 0.185),
    (AAUWSS_METRIC, "output: labels shuffled in place", 0.185),
    (AAUWSS_METRIC, "output: Deep <-> REM swapped", 0.146),
    (AAUWSS_METRIC, "output: Wake <-> Light swapped", 0.119),
    (AAUWSS_METRIC, "output: shifted +1 epoch", 0.00107),
    (AAUWSS_METRIC, "output: reversed", 0.205),
    (AAUWSS_METRIC, "output: last 10% blanked to Light", 0.0085),
    (DREAMT_METRIC, "output: every epoch = Light", 0.14),
    (DREAMT_METRIC, "output: labels shuffled in place", 0.13),
    (DREAMT_METRIC, "output: Deep <-> REM swapped", 0.0323),
    (DREAMT_METRIC, "output: Wake <-> Light swapped", 0.191),
    (DREAMT_METRIC, "output: shifted +1 epoch", 0.00297),
    (DREAMT_METRIC, "output: reversed", 0.142),
    (DREAMT_METRIC, "output: last 10% blanked to Light", 0.0116),
    (SLEEP_ACCEL_METRIC, "output: every epoch = Light", 0.17),
    (SLEEP_ACCEL_METRIC, "output: labels shuffled in place", 0.165),
    (SLEEP_ACCEL_METRIC, "output: Deep <-> REM swapped", 0.118),
    (SLEEP_ACCEL_METRIC, "output: Wake <-> Light swapped", 0.126),
    (SLEEP_ACCEL_METRIC, "output: shifted +1 epoch", 0.00112),
    (SLEEP_ACCEL_METRIC, "output: reversed", 0.186),
    (SLEEP_ACCEL_METRIC, "output: last 10% blanked to Light", 0.0197),
    ("wake refinement (motion-aware wake -> light)", "output: refinement is a no-op (passthrough)", 270.0),
    ("wake refinement (motion-aware wake -> light)", "output: every wake second converted to Light", 351.0),
    ("wake refinement (motion-aware wake -> light)", "input: interior wake run cut to 4 min (under the shipped floor)", 108.0),
    ("wake refinement (motion-aware wake -> light)", "input: Wake <-> Light swapped before the pass", 351.0),
    ("wake refinement (motion-aware wake -> light)", "input: step stream thinned to 33% (proxy for MIN_DENSE_FRACTION)", 270.0),
];

/// `(metric, arm, why)`. Probe arms that cannot carry a floor, because the mutation does not move the
/// number at all. Their blindness is the finding, not a defect to assert away.
const NO_FLOOR: &[(&str, &str, &str)] = &[
    ("in-bed span detection (sleep session boundaries)", "input: HR stream removed entirely", "measured delta is exactly zero: this mutation does not move the number"),
    ("in-bed span detection (sleep session boundaries)", "input: 25-min gravity hole (proxy for MAX_GAP_MIN, a bare const)", "measured delta is exactly zero: this mutation does not move the number"),
    ("main-night bridging (overnight night-tail cutoff)", "input: 00:00 onset", "measured delta is exactly zero: this mutation does not move the number"),
    ("main-night selection", "output: pick the longest by clock span (no alignment work at all)", "measured delta is exactly zero: this mutation does not move the number"),
    ("main-night selection", "input: block order swapped", "measured delta is exactly zero: this mutation does not move the number"),
    ("main-night selection", "input: nap onsets shifted +30 min", "measured delta is exactly zero: this mutation does not move the number"),
    ("personal sleep need (hours)", "input: nights reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("personal sleep need (hours)", "input: nights shuffled", "measured delta is exactly zero: this mutation does not move the number"),
    ("personal sleep need (hours)", "input: two non-positive nights appended", "measured delta is exactly zero: this mutation does not move the number"),
    ("Rest (sleep-performance composite 0-100)", "input: deep and REM seconds swapped", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (rolling balance minutes)", "output: constant 0 balance", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (rolling balance minutes)", "input: empty series", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (rolling balance minutes)", "input: series reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (rolling balance minutes)", "input: both nights exactly on need", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (trailing window edge)", "output: constant 14 nights but the wrong oldest day", "measured delta is exactly zero: this mutation does not move the number"),
    ("sleep-debt ledger (trailing window edge)", "input: series reversed", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleep Regularity Index", "input: never asleep (a detector that does no work)", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleep Regularity Index", "input: always asleep", "measured delta is exactly zero: this mutation does not move the number"),
    ("Sleep Regularity Index", "input: never worn (every epoch unknown)", "the arm yields no number, so it has no distance from the baseline"),
    ("Sleep Regularity Index", "input: every day rotated by one epoch (a uniform 30 s shift)", "measured delta is exactly zero: this mutation does not move the number"),
    ("wake refinement (motion-aware wake -> light)", "input: the window mirrored in time (the 3-min run becomes trailing)", "measured delta is exactly zero: this mutation does not move the number"),
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
#[ignore = "negative control; the PSG corpus is a derived tree outside the repo (DREAMT is \
            credentialed, not redistributable) — set WHOOP_SLEEP_FIXTURES and run with --ignored"]
fn sensitivity_stage_v2_dreamt() {
    let t = stage_sweep("dreamt", DREAMT_KAPPA, DREAMT_N, DREAMT_RR_NIGHTS);
    assert!(
        resp_weight_deltas(&t).iter().any(|(_, d)| *d != 0.0),
        "DREAMT carries R-R on every night, so resp_weight must reach the kappa — it did not"
    );
    t.finish();
}

#[test]
#[ignore = "negative control; the PSG corpus is a derived tree outside the repo — set \
            WHOOP_SLEEP_FIXTURES and run with --ignored"]
fn sensitivity_stage_v2_aauwss() {
    let t = stage_sweep("aauwss", AAUWSS_KAPPA, AAUWSS_N, AAUWSS_RR_NIGHTS);
    assert!(
        resp_weight_deltas(&t).iter().any(|(_, d)| *d != 0.0),
        "AAUWSS carries R-R on every night, so resp_weight must reach the kappa — it did not"
    );
    t.finish();
}

/// The cohort with NO R-R. Its sheet scores V2's cardiac-and-motion path, and the two `resp_weight`
/// arms below prove it: they are the only arms that cannot move a figure here, because the term they
/// scale never fires. Every `caught`/`missed` count this test prints inherits that.
#[test]
#[ignore = "negative control; the PSG corpus is a derived tree outside the repo — set \
            WHOOP_SLEEP_FIXTURES and run with --ignored"]
fn sensitivity_stage_v2_sleep_accel_no_respiratory_channel() {
    let t = stage_sweep("sleep-accel", SLEEP_ACCEL_KAPPA, SLEEP_ACCEL_N, SLEEP_ACCEL_RR_NIGHTS);
    for (arm, d) in resp_weight_deltas(&t) {
        assert_eq!(
            d, 0.0,
            "sleep-accel carries no R-R, so `{arm}` must move the kappa by exactly zero — it moved \
             it by {d}. Either the cohort gained R-R or resp_weight now reads something else, and \
             either way this sheet is no longer a no-respiratory-channel measurement"
        );
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 2 — the frozen-golden hypnogram (the whole V2 recipe, stage for stage).
// ---------------------------------------------------------------------------------------------

fn rsa_wave(ph: usize, i: i64) -> i64 {
    let amp = [12i64, 60, 30, 20][ph];
    [0, amp, 0, -amp][(i % 4) as usize]
}

/// The crafted 4-phase night the frozen golden pins, rebuilt integer-for-integer.
fn golden_input() -> SleepInput {
    let start = REF_MIDNIGHT + 3_600;
    let phase: i64 = 90 * 60;
    let dur = phase * 4;
    let (mut accel, mut hr, mut rr) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..dur {
        let ts = start + i;
        let ph = (i / phase) as usize;
        let restless = ph == 3 && (i % 20) < 6;
        if restless {
            accel.push(AccelSample { ts, x: 0.2, y: 0.15, z: 0.96 });
        } else {
            accel.push(AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
        }
        let bpm: i64 = match ph {
            0 => 50,
            1 => 54 + [0, 1, 2, 3, 2, 1][((i / 20) % 6) as usize],
            2 => 56 + (i / 60) % 4,
            _ => 66 + (i / 30) % 6,
        };
        hr.push(SleepHr { ts, bpm: bpm as u16 });
        let rr_ms = 60_000 / bpm + rsa_wave(ph, i);
        rr.push(RrRun { ts, intervals: vec![rr_ms as u16] });
    }
    SleepInput { start, end: start + dur, hr, rr, accel }
}

fn golden_seconds() -> Vec<i32> {
    let mut out = vec![1i32; 21_600];
    for (a, b, stage) in GOLDEN_HYPNOGRAM {
        for v in out[a as usize..b as usize].iter_mut() {
            *v = stage_to_int(stage);
        }
    }
    out
}

fn segments_to_seconds(segs: &[StageSegment], start: i64, len: usize) -> Vec<i32> {
    let mut out = vec![1i32; len];
    for s in segs {
        let a = (s.start - start).max(0) as usize;
        let b = ((s.end - start).max(0) as usize).min(len);
        for v in out[a.min(len)..b].iter_mut() {
            *v = stage_to_int(s.stage);
        }
    }
    out
}

/// The 6-row segment table the golden asserts, derived back out of a per-second stage array.
fn runs_of(secs: &[i32]) -> Vec<(i64, i64, i32)> {
    let mut out: Vec<(i64, i64, i32)> = Vec::new();
    for (i, &v) in secs.iter().enumerate() {
        match out.last_mut() {
            Some(last) if last.2 == v => last.1 = i as i64 + 1,
            _ => out.push((i as i64, i as i64 + 1, v)),
        }
    }
    out
}

fn golden_runs() -> Vec<(i64, i64, i32)> {
    GOLDEN_HYPNOGRAM.iter().map(|(a, b, s)| (*a, *b, stage_to_int(*s))).collect()
}

fn matching_share(a: &[i32], b: &[i32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / n as f64
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_frozen_golden_hypnogram() {
    let input = golden_input();
    let len = (input.end - input.start) as usize;
    let golden = golden_seconds();
    let want = golden_runs();

    let gate = "golden_tests.rs:60-65 — the staged segments must equal the frozen 6-row table exactly \
                (value column = share of seconds matching the golden)";
    let mut t = Table::new("sleep frozen-golden hypnogram (whole V2 recipe)", gate);

    let base = segments_to_seconds(&stage_v2(&input), input.start, len);
    t.push(
        "baseline (unmutated stage_v2)",
        Kind::Baseline,
        matching_share(&base, &golden),
        runs_of(&base) == want,
    );

    // The golden is pinned per SECOND here, so an epoch shift is 30 samples, not 1.
    let mut second_arms: Vec<BoxedArm> = vec![("output: every second = Light", Kind::Null, Box::new(out_all_light))];
    second_arms.push(("output: labels shuffled in place", Kind::Null, Box::new(out_shuffle)));
    second_arms.push(("output: Deep <-> REM swapped", Kind::Structural, Box::new(out_swap_deep_rem)));
    second_arms.push(("output: Wake <-> Light swapped", Kind::Structural, Box::new(out_swap_wake_light)));
    second_arms.push(("output: shifted +1 epoch (30 s)", Kind::Structural, Box::new(|p| shift_by(p, 30))));
    second_arms.push(("output: shifted +1 second", Kind::Structural, Box::new(|p| shift_by(p, 1))));
    second_arms.push(("output: reversed", Kind::Structural, Box::new(out_reverse)));
    second_arms.push(("output: last 10% blanked to Light", Kind::Structural, Box::new(out_drop_tail)));
    for (name, kind, f) in &second_arms {
        let mut v = base.clone();
        f(&mut v);
        t.push(*name, *kind, matching_share(&v, &golden), runs_of(&v) == want);
    }

    for arm in recipe_arms() {
        let v = segments_to_seconds(&stage_v2_with(&input, &arm.params), input.start, len);
        t.push(format!("param: {}", arm.name), arm.kind, matching_share(&v, &golden), runs_of(&v) == want);
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 3 — in-bed span detection.
// ---------------------------------------------------------------------------------------------

/// The simple still night golden_tests.rs:163-182 drives the detector with: two hours, 50 bpm, gravity
/// pinned at (0, 0, 1).
fn still_night(start: i64, dur: i64) -> (Vec<SleepHr>, Vec<AccelSample>) {
    let mut hr = Vec::new();
    let mut accel = Vec::new();
    for i in 0..dur {
        hr.push(SleepHr { ts: start + i, bpm: 50 });
        accel.push(AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 });
    }
    (hr, accel)
}

/// Gravity that never settles: a walk of `amp` g on every sample, so no window can reach `still_enter`.
fn restless_gravity(start: i64, dur: i64, amp: f64) -> Vec<AccelSample> {
    (0..dur)
        .map(|i| {
            let s = if i % 2 == 0 { amp } else { -amp };
            AccelSample { ts: start + i, x: s, y: -s, z: 1.0 }
        })
        .collect()
}

struct DetectOutcome {
    spans: usize,
    seconds: i64,
    start_ok: bool,
}

fn detect_outcome(hr: &[SleepHr], accel: &[AccelSample], want_start: i64, p: &DetectParams) -> DetectOutcome {
    let spans = detect_sessions_with(hr, accel, 0, &[], &[], None, p);
    DetectOutcome {
        spans: spans.len(),
        seconds: spans.iter().map(|s| s.end - s.start).sum(),
        start_ok: spans.first().map(|s| s.start == want_start).unwrap_or(false),
    }
}

fn detect_gate(o: &DetectOutcome) -> bool {
    o.spans == 1 && o.start_ok && o.seconds >= DETECT_MIN_SPAN_S
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_in_bed_detection() {
    let dur = 2 * 3600;
    let (hr, accel) = still_night(REF_MIDNIGHT, dur);
    let gate = format!(
        "golden_tests.rs:176-178 — exactly one span, starting at the window start, at least \
         {DETECT_MIN_SPAN_S} s long (value column = detected in-bed seconds)"
    );
    let mut t = Table::new("in-bed span detection (sleep session boundaries)", gate);

    let base = detect_outcome(&hr, &accel, REF_MIDNIGHT, &DetectParams::SHIPPED);
    t.push("baseline (DetectParams::SHIPPED)", Kind::Baseline, base.seconds as f64, detect_gate(&base));

    // NULL arms: inputs that carry no stillness evidence at all.
    let noisy = restless_gravity(REF_MIDNIGHT, dur, 0.5);
    let o = detect_outcome(&hr, &noisy, REF_MIDNIGHT, &DetectParams::SHIPPED);
    t.push("input: gravity replaced by constant motion", Kind::Null, o.seconds as f64, detect_gate(&o));

    let mut shuffled = accel.clone();
    let vecs: Vec<(f64, f64, f64)> = restless_gravity(REF_MIDNIGHT, dur, 0.5).iter().map(|g| (g.x, g.y, g.z)).collect();
    let mut idx: Vec<usize> = (0..vecs.len()).collect();
    lcg_shuffle(&mut idx, 0xA11CE);
    for (i, s) in shuffled.iter_mut().enumerate() {
        let (x, y, z) = vecs[idx[i]];
        s.x = x;
        s.y = y;
        s.z = z;
    }
    let o = detect_outcome(&hr, &shuffled, REF_MIDNIGHT, &DetectParams::SHIPPED);
    // The still baseline is a constant vector, so shuffling THAT is a no-op; this shuffles motion.
    t.push("input: gravity replaced by shuffled motion vectors", Kind::Null, o.seconds as f64, detect_gate(&o));

    // STRUCTURAL arms.
    let (dhr, daccel) = still_night(REF_MIDNIGHT + 12 * 3600, dur);
    let o = detect_outcome(&dhr, &daccel, REF_MIDNIGHT + 12 * 3600, &DetectParams::SHIPPED);
    t.push("input: same night shifted into the 12:00-14:00 daytime band", Kind::Structural, o.seconds as f64, detect_gate(&o));

    let (shr, saccel) = still_night(REF_MIDNIGHT, 45 * 60);
    let o = detect_outcome(&shr, &saccel, REF_MIDNIGHT, &DetectParams::SHIPPED);
    t.push("input: night truncated to 45 min (under min_sleep_min)", Kind::Structural, o.seconds as f64, detect_gate(&o));

    let o = detect_outcome(&[], &accel, REF_MIDNIGHT, &DetectParams::SHIPPED);
    t.push("input: HR stream removed entirely", Kind::Structural, o.seconds as f64, detect_gate(&o));

    let holed: Vec<AccelSample> =
        accel.iter().copied().filter(|g| !(3000..3000 + 25 * 60).contains(&(g.ts - REF_MIDNIGHT))).collect();
    let o = detect_outcome(&hr, &holed, REF_MIDNIGHT, &DetectParams::SHIPPED);
    t.push("input: 25-min gravity hole (proxy for MAX_GAP_MIN, a bare const)", Kind::Structural, o.seconds as f64, detect_gate(&o));

    // PARAMETER arms — DetectParams is public with public fields, so these are the real knobs.
    let dknobs: Vec<Knob<DetectParams>> = vec![
        ("still_enter", |p, k| p.still_enter *= k, |p| p.still_enter),
        ("still_exit", |p, k| p.still_exit *= k, |p| p.still_exit),
        ("min_sleep_min", |p, k| p.min_sleep_min = (p.min_sleep_min as f64 * k).round() as i64, |p| p.min_sleep_min as f64),
        (
            "wake_absorb_max_min",
            |p, k| p.wake_absorb_max_min = (p.wake_absorb_max_min as f64 * k).round() as i64,
            |p| p.wake_absorb_max_min as f64,
        ),
    ];
    for (name, set, get) in dknobs {
        for k in [1.1f64, 0.9] {
            let mut p = DetectParams::SHIPPED;
            set(&mut p, k);
            let o = detect_outcome(&hr, &accel, REF_MIDNIGHT, &p);
            t.push(
                format!("param: {name} {:.3} -> {:.3} (x{k:.2})", get(&DetectParams::SHIPPED), get(&p)),
                Kind::Parameter,
                o.seconds as f64,
                detect_gate(&o),
            );
        }
    }
    let o = detect_outcome(&hr, &accel, REF_MIDNIGHT, &DetectParams::PRE_HYSTERESIS);
    t.push("param: the whole PRE_HYSTERESIS spine (0.70/0.70, absorb 0)", Kind::Parameter, o.seconds as f64, detect_gate(&o));
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 4 — motion-aware wake refinement.
// ---------------------------------------------------------------------------------------------

fn still_streams(minutes: i64) -> (Vec<AccelSample>, Vec<StepSample>) {
    let (mut grav, mut steps) = (Vec::new(), Vec::new());
    for m in 0..minutes {
        grav.push(AccelSample { ts: m * 60, x: 0.0, y: 0.0, z: 1.0 });
        grav.push(AccelSample { ts: m * 60 + 30, x: 0.0, y: 0.0, z: 1.0 });
        steps.push(StepSample { ts: m * 60, counter: 100, activity_class: Some(0) });
    }
    (grav, steps)
}

/// One short and one long wake run at the window edges plus one in its interior — the window
/// refine.rs:391-399 pins its wake ledger on.
fn three_wake_runs() -> Vec<StageSegment> {
    let seg = |a: i64, b: i64, stage: SleepStage| StageSegment { start: a, end: b, stage };
    vec![
        seg(0, 180, SleepStage::Wake),
        seg(180, 1200, SleepStage::Light),
        seg(1200, 1800, SleepStage::Wake),
        seg(1800, 2400, SleepStage::Light),
        seg(2400, 3000, SleepStage::Wake),
    ]
}

fn wake_seconds(segs: &[StageSegment]) -> i64 {
    segs.iter().filter(|s| s.stage == SleepStage::Wake).map(|s| s.end - s.start).sum()
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_wake_refinement() {
    let segs = three_wake_runs();
    let (grav, steps) = still_streams(50);
    let gate = format!("refine.rs:393 — wake seconds after the shipped pass == {REFINE_SHIPPED_WAKE_S}");
    let mut t = Table::new("wake refinement (motion-aware wake -> light)", gate);
    let pass = |v: i64| v == REFINE_SHIPPED_WAKE_S;

    let base = wake_seconds(&refine_wake_with(&segs, &grav, &steps, &RefineParams::SHIPPED));
    t.push("baseline (RefineParams::SHIPPED)", Kind::Baseline, base as f64, pass(base));

    // NULL arms at the output: a pass that does nothing, and one that converts everything.
    let v = wake_seconds(&segs);
    t.push("output: refinement is a no-op (passthrough)", Kind::Null, v as f64, pass(v));
    let all_light: Vec<StageSegment> =
        segs.iter().map(|s| StageSegment { stage: SleepStage::Light, ..*s }).collect();
    let v = wake_seconds(&all_light);
    t.push("output: every wake second converted to Light", Kind::Null, v as f64, pass(v));

    // STRUCTURAL arms.
    let end = segs.last().expect("three_wake_runs is non-empty").end;
    let mirrored: Vec<StageSegment> = segs
        .iter()
        .rev()
        .map(|s| StageSegment { start: end - s.end, end: end - s.start, stage: s.stage })
        .collect();
    let v = wake_seconds(&refine_wake_with(&mirrored, &grav, &steps, &RefineParams::SHIPPED));
    t.push("input: the window mirrored in time (the 3-min run becomes trailing)", Kind::Structural, v as f64, pass(v));

    let short_interior: Vec<StageSegment> = {
        let seg = |a: i64, b: i64, stage: SleepStage| StageSegment { start: a, end: b, stage };
        vec![
            seg(0, 180, SleepStage::Wake),
            seg(180, 1200, SleepStage::Light),
            seg(1200, 1440, SleepStage::Wake), // 4 min, under the 5-min floor
            seg(1440, 2400, SleepStage::Light),
            seg(2400, 3000, SleepStage::Wake),
        ]
    };
    let v = wake_seconds(&refine_wake_with(&short_interior, &grav, &steps, &RefineParams::SHIPPED));
    t.push("input: interior wake run cut to 4 min (under the shipped floor)", Kind::Structural, v as f64, pass(v));

    let swapped: Vec<StageSegment> = segs
        .iter()
        .map(|s| StageSegment {
            stage: match s.stage {
                SleepStage::Wake => SleepStage::Light,
                SleepStage::Light => SleepStage::Wake,
                x => x,
            },
            ..*s
        })
        .collect();
    let v = wake_seconds(&refine_wake_with(&swapped, &grav, &steps, &RefineParams::SHIPPED));
    t.push("input: Wake <-> Light swapped before the pass", Kind::Structural, v as f64, pass(v));

    // Proxy for MIN_DENSE_FRACTION, which is a bare const with no injection point: thin the step
    // stream until the density gate declines.
    let thin: Vec<StepSample> = steps.iter().copied().step_by(3).collect();
    let v = wake_seconds(&refine_wake_with(&segs, &grav, &thin, &RefineParams::SHIPPED));
    t.push("input: step stream thinned to 33% (proxy for MIN_DENSE_FRACTION)", Kind::Structural, v as f64, pass(v));

    // PARAMETER arms — RefineParams is public with public fields.
    let rknobs: Vec<Knob<RefineParams>> = vec![
        (
            "min_wake_segment_seconds",
            |p, k| p.min_wake_segment_seconds = (p.min_wake_segment_seconds as f64 * k).round() as i64,
            |p| p.min_wake_segment_seconds as f64,
        ),
        ("stable_posture_variance_g2", |p, k| p.stable_posture_variance_g2 *= k, |p| p.stable_posture_variance_g2),
        ("min_stable_minute_fraction", |p, k| p.min_stable_minute_fraction *= k, |p| p.min_stable_minute_fraction),
    ];
    for (name, set, get) in rknobs {
        for k in [1.1f64, 0.9] {
            let mut p = RefineParams::SHIPPED;
            set(&mut p, k);
            let v = wake_seconds(&refine_wake_with(&segs, &grav, &steps, &p));
            t.push(
                format!("param: {name} {:.3} -> {:.3} (x{k:.2})", get(&RefineParams::SHIPPED), get(&p)),
                Kind::Parameter,
                v as f64,
                pass(v),
            );
        }
    }
    for pad in [0i64, 2] {
        let p = RefineParams { burst_pad_minutes: pad, ..RefineParams::SHIPPED };
        let v = wake_seconds(&refine_wake_with(&segs, &grav, &steps, &p));
        t.push(format!("param: burst_pad_minutes 1 -> {pad}"), Kind::Parameter, v as f64, pass(v));
    }
    let p = RefineParams { skip_window_edges: false, ..RefineParams::SHIPPED };
    let v = wake_seconds(&refine_wake_with(&segs, &grav, &steps, &p));
    t.push("param: skip_window_edges true -> false (the pre-H rule)", Kind::Parameter, v as f64, pass(v));
    t.finish();

    // The second shipped refinement gate, on the analyze path: the golden night's trailing wake run.
    let input = golden_input();
    let dense: Vec<StepSample> = (input.start..input.end)
        .step_by(30)
        .map(|ts| StepSample { ts, counter: 100, activity_class: Some(0) })
        .collect();
    let streams = SleepStreams {
        hr: input.hr.clone(),
        rr: input.rr.clone(),
        accel: input.accel.clone(),
        steps: dense,
        tz_offset_s: 0,
        ..Default::default()
    };
    let sessions = analyze(&streams);
    let tail = *sessions[0].segments.last().expect("the golden night stages to at least one segment");
    println!();
    println!("=== wake refinement — analyze trailing-wake baseline ===");
    println!("shipped gate: golden_tests.rs:150 — the trailing wake run is {GOLDEN_TRAILING_WAKE_S} s");
    println!("baseline: stage={:?} seconds={}", tail.stage, tail.end - tail.start);
    assert_eq!(
        GOLDEN_TRAILING_WAKE_S,
        tail.end - tail.start,
        "the analyze-path trailing-wake baseline must reproduce the shipped figure"
    );
}

// ---------------------------------------------------------------------------------------------
// Metric 5 — main-night selection, and the bridge cutoff behind it.
// ---------------------------------------------------------------------------------------------

fn nb(start: i64, end: i64) -> NightBlock {
    NightBlock { start, end }
}

/// Every nap-versus-night pairing `realistic_nap_never_beats_real_night_cold_start` asserts in
/// `mainnight.rs`: the night must win all 2,160 of them. The count is re-derived by
/// `shipped_gate_constants_match_their_sources`; the line number is looked up, never written down.
fn nap_vs_night_cases() -> Vec<(NightBlock, NightBlock)> {
    let at = |h: i64| REF_MIDNIGHT + h * 3600;
    let night_starts = [at(20) - SECONDS_PER_DAY, at(22) - SECONDS_PER_DAY, at(23) - SECONDS_PER_DAY, at(0), at(1)];
    let nap_starts = [at(6), at(8), at(10), at(12), at(13), at(15), at(17), at(19), at(21)];
    let mut out = Vec::new();
    for &ns in &night_starts {
        for nh in [4i64, 5, 6, 7, 8, 9] {
            for &ps in &nap_starts {
                for pm in [20i64, 30, 45, 60, 90, 120, 150, 180] {
                    out.push((nb(ps, ps + pm * 60), nb(ns, ns + nh * 3600)));
                }
            }
        }
    }
    out
}

/// Share of cases where the selector picked the NIGHT block, whichever slot it sits in.
fn night_win_share(pick: impl Fn(&[NightBlock]) -> Option<usize>, swap: bool, nap_shift_s: i64) -> f64 {
    let cases = nap_vs_night_cases();
    let mut wins = 0usize;
    for (nap, night) in &cases {
        let nap = nb(nap.start + nap_shift_s, nap.end + nap_shift_s);
        let (blocks, night_idx) =
            if swap { ([*night, nap], 0usize) } else { ([nap, *night], 1usize) };
        if pick(&blocks) == Some(night_idx) {
            wins += 1;
        }
    }
    wins as f64 / cases.len() as f64
}

fn longest_by_clock(blocks: &[NightBlock]) -> Option<usize> {
    (0..blocks.len()).max_by_key(|&i| blocks[i].end - blocks[i].start)
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_main_night_selection() {
    let cases = nap_vs_night_cases().len();
    let line = line_of(
        include_str!("../src/sleep/mainnight.rs"),
        "fn realistic_nap_never_beats_real_night_cold_start",
    );
    let gate = format!(
        "mainnight.rs:{line} realistic_nap_never_beats_real_night_cold_start — every one of the \
         {cases} nap-vs-night pairings must select the NIGHT block (value column = share of cases \
         the night wins)"
    );
    let mut t = Table::new("main-night selection", gate);
    let pass = |v: f64| within(v, 1.0, 1e-12);

    let v = night_win_share(|b| main_night_index(b, 0, None), false, 0);
    t.push("baseline (main_night_index, cold start)", Kind::Baseline, v, pass(v));

    // NULL arms.
    let v = night_win_share(|b| (!b.is_empty()).then_some(0), false, 0);
    t.push("output: always pick block 0", Kind::Null, v, pass(v));
    let v = night_win_share(|b| (0..b.len()).max_by_key(|&i| b[i].start), false, 0);
    t.push("output: always pick the later onset", Kind::Null, v, pass(v));
    let v = night_win_share(longest_by_clock, false, 0);
    t.push("output: pick the longest by clock span (no alignment work at all)", Kind::Null, v, pass(v));

    // STRUCTURAL arms.
    let v = night_win_share(|b| main_night_index(b, 0, None), true, 0);
    t.push("input: block order swapped", Kind::Structural, v, pass(v));
    let v = night_win_share(|b| main_night_index(b, 0, None), false, 6 * 3600);
    t.push("input: nap onsets shifted +6 h (proxy for the private ALIGNMENT_* window)", Kind::Structural, v, pass(v));
    let v = night_win_share(|b| main_night_index(b, 0, None), false, -6 * 3600);
    t.push("input: nap onsets shifted -6 h (same proxy, other direction)", Kind::Structural, v, pass(v));
    let v = night_win_share(|b| main_night_index(b, 0, None), false, 30 * 60);
    t.push("input: nap onsets shifted +30 min", Kind::Structural, v, pass(v));

    // PARAMETER arms — the caller-supplied habitual anchor and the timezone offset are the only two
    // knobs a caller can reach; ALIGNMENT_BONUS_MIN / ALIGNMENT_ZERO_SEC / ALIGNMENT_FULL_WINDOW_SEC
    // are private consts with no injection point.
    let anchor = 3 * 3600 + 1800; // the cold-start anchor, 03:30, measured at mainnight.rs:380
    for k in [1.0f64, 1.1, 0.9] {
        let h = (anchor as f64 * k).round() as i64;
        let v = night_win_share(move |b| main_night_index(b, 0, Some(h)), false, 0);
        t.push(format!("param: habitual_midsleep_sec {anchor} -> {h} (x{k:.2})"), Kind::Parameter, v, pass(v));
    }
    for off in [3600i64, -3600, 6 * 3600] {
        let v = night_win_share(move |b| main_night_index(b, off, None), false, 0);
        t.push(format!("param: offset_s 0 -> {off} s"), Kind::Parameter, v, pass(v));
    }
    t.finish();
}

/// The largest gap (minutes) still bridged into one night group, measured rather than read — the two
/// bridge constants are private.
fn bridge_cutoff_min(onset_hour: i64, offset_s: i64) -> i64 {
    let a = REF_MIDNIGHT + onset_hour * 3600;
    let mut best = 0;
    for gap in 0..=150i64 {
        let b = a + 3 * 3600 + gap * 60;
        let group = main_night_group_indices(&[nb(a, a + 3 * 3600), nb(b, b + 4 * 3600)], offset_s, None);
        if group == Some(vec![0, 1]) {
            best = gap;
        }
    }
    best
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_main_night_bridge() {
    let gate = format!(
        "mainnight.rs:511-517 — a 70-min overnight gap bridges and a 95-min one does not, so the \
         measured cutoff must land in [{BRIDGE_CUTOFF_MIN_LO}, {BRIDGE_CUTOFF_MIN_HI}) minutes"
    );
    let mut t = Table::new("main-night bridging (overnight night-tail cutoff)", gate);
    let pass = |v: f64| (v as i64) >= BRIDGE_CUTOFF_MIN_LO && (v as i64) < BRIDGE_CUTOFF_MIN_HI;

    let v = bridge_cutoff_min(23, 0);
    t.push("baseline (23:00 onset, offset 0)", Kind::Baseline, v as f64, pass(v as f64));
    t.push("output: never bridge", Kind::Null, 0.0, pass(0.0));
    t.push("output: always bridge", Kind::Null, 150.0, pass(150.0));

    let v = bridge_cutoff_min(13, 0);
    t.push("input: 13:00 onset — the short daytime bridge, not the night tail", Kind::Structural, v as f64, pass(v as f64));
    let v = bridge_cutoff_min(0, 0);
    t.push("input: 00:00 onset", Kind::Structural, v as f64, pass(v as f64));

    for off in [3600i64, -3600, 12 * 3600] {
        let v = bridge_cutoff_min(23, off);
        t.push(format!("param: offset_s 0 -> {off} s"), Kind::Parameter, v as f64, pass(v as f64));
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 6 — Rest (sleep-performance composite).
// ---------------------------------------------------------------------------------------------

struct RestInput {
    asleep_s: f64,
    efficiency: f64,
    deep_s: f64,
    rem_s: f64,
    need_h: Option<f64>,
    consistency: Option<f64>,
}

impl RestInput {
    /// The perfect night rest.rs:79 scores: 8 h asleep, 95% efficient, 30% deep, 25% REM.
    fn perfect() -> Self {
        RestInput {
            asleep_s: 8.0 * 3600.0,
            efficiency: 0.95,
            deep_s: 0.30 * 8.0 * 3600.0,
            rem_s: 0.25 * 8.0 * 3600.0,
            need_h: Some(8.0),
            consistency: Some(1.0),
        }
    }
    fn score(&self) -> f64 {
        rest(self.asleep_s, self.efficiency, self.deep_s, self.rem_s, self.need_h, self.consistency)
            .expect("the perfect night has positive asleep time")
    }
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_rest_composite() {
    let gate = format!("rest.rs:80 — the perfect night must score above {REST_PERFECT_NIGHT_FLOOR}");
    let mut t = Table::new("Rest (sleep-performance composite 0-100)", gate);
    let pass = |v: f64| v > REST_PERFECT_NIGHT_FLOOR;

    let base = RestInput::perfect().score();
    t.push("baseline (rest.rs:79 perfect night)", Kind::Baseline, base, pass(base));

    // NULL arms: a scorer that ignores the night entirely.
    t.push("output: constant 50", Kind::Null, 50.0, pass(50.0));
    t.push("output: constant 90", Kind::Null, 90.0, pass(90.0));

    // STRUCTURAL arms.
    let mut s = RestInput::perfect();
    std::mem::swap(&mut s.deep_s, &mut s.rem_s);
    t.push("input: deep and REM seconds swapped", Kind::Structural, s.score(), pass(s.score()));
    let s = RestInput { efficiency: 0.0, ..RestInput::perfect() };
    t.push("input: efficiency zeroed", Kind::Structural, s.score(), pass(s.score()));
    let s = RestInput { deep_s: 0.0, ..RestInput::perfect() };
    t.push("input: deep zeroed (the DEEP_FLOOR_FACTOR half)", Kind::Structural, s.score(), pass(s.score()));
    let s = RestInput { asleep_s: 4.0 * 3600.0, ..RestInput::perfect() };
    t.push("input: night halved to 4 h", Kind::Structural, s.score(), pass(s.score()));

    // PARAMETER arms. Only sleep_need_hours and consistency are arguments; W_DURATION, W_EFFICIENCY,
    // W_RESTORATIVE, W_CONSISTENCY, DEFAULT_SLEEP_NEED_HOURS, RESTORATIVE_TARGET_SHARE,
    // DEEP_SHARE_TARGET, DEEP_FLOOR_FACTOR and NEUTRAL_CONSISTENCY are `pub const` with no injection
    // point, so the rest are input perturbations in the equivalent direction, labelled as proxies.
    for k in [1.1f64, 0.9, 1.005] {
        let s = RestInput { need_h: Some(8.0 * k), ..RestInput::perfect() };
        t.push(format!("param: sleep_need_hours 8.000 -> {:.3} (x{k:.3})", 8.0 * k), Kind::Parameter, s.score(), pass(s.score()));
    }
    for k in [1.1f64, 0.9] {
        let c = (1.0f64 * k).min(1.0);
        let s = RestInput { consistency: Some(c), ..RestInput::perfect() };
        t.push(format!("param: consistency 1.000 -> {c:.3} (x{k:.2})"), Kind::Parameter, s.score(), pass(s.score()));
    }
    let s = RestInput { consistency: None, ..RestInput::perfect() };
    t.push("param: consistency absent (proxy for NEUTRAL_CONSISTENCY)", Kind::Parameter, s.score(), pass(s.score()));
    let proxies: Vec<Proxy> = vec![
        ("asleep_seconds (proxy W_DURATION)", |s, k| s.asleep_s *= k),
        ("efficiency (proxy W_EFFICIENCY)", |s, k| s.efficiency = (s.efficiency * k).min(1.0)),
        ("deep+REM seconds (proxy W_RESTORATIVE)", |s, k| {
            s.deep_s *= k;
            s.rem_s *= k;
        }),
        ("deep seconds alone (proxy DEEP_SHARE_TARGET)", |s, k| s.deep_s *= k),
    ];
    for (name, set) in proxies {
        for k in [1.1f64, 0.9] {
            let mut s = RestInput::perfect();
            set(&mut s, k);
            t.push(format!("param: {name} x{k:.2}"), Kind::Parameter, s.score(), pass(s.score()));
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 7 — personal sleep need.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_personal_sleep_need() {
    let gate =
        format!("rest.rs:73 — personal_sleep_need_hours(&[8.0, 9.0, 8.5]) is {PERSONAL_NEED_TARGET} within {PERSONAL_NEED_TOL:e}");
    let mut t = Table::new("personal sleep need (hours)", gate);
    let pass = |v: f64| within(v, PERSONAL_NEED_TARGET, PERSONAL_NEED_TOL);
    let nights = [8.0f64, 9.0, 8.5];

    let base = personal_sleep_need_hours(&nights);
    t.push("baseline (mean of 8.0 / 9.0 / 8.5)", Kind::Baseline, base, pass(base));

    t.push("output: constant 7.5 (the floor)", Kind::Null, 7.5, pass(7.5));
    t.push("output: constant 8.0 (the default need)", Kind::Null, 8.0, pass(8.0));
    let v = personal_sleep_need_hours(&[]);
    t.push("input: empty history", Kind::Null, v, pass(v));

    let mut rev = nights;
    rev.reverse();
    let v = personal_sleep_need_hours(&rev);
    t.push("input: nights reversed", Kind::Structural, v, pass(v));
    let mut sh = nights;
    lcg_shuffle(&mut sh, 0xBEEF);
    let v = personal_sleep_need_hours(&sh);
    t.push("input: nights shuffled", Kind::Structural, v, pass(v));
    let v = personal_sleep_need_hours(&[8.0, 9.0, 8.5, 0.0, -1.0]);
    t.push("input: two non-positive nights appended", Kind::Structural, v, pass(v));
    let v = personal_sleep_need_hours(&[6.0, 0.0, 6.5]);
    t.push("input: a short-sleep window (must land on the floor)", Kind::Structural, v, pass(v));

    // No constant is reachable: MIN_SLEEP_NEED_HOURS and DEFAULT_SLEEP_NEED_HOURS are `pub const` with
    // no injection point, so every parameter arm here is an input perturbation.
    for k in [1.1f64, 0.9, 1.005] {
        let scaled: Vec<f64> = nights.iter().map(|h| h * k).collect();
        let v = personal_sleep_need_hours(&scaled);
        t.push(format!("param: every night x{k:.3} (proxy for the need constants)"), Kind::Parameter, v, pass(v));
    }
    for (i, label) in ["first", "second", "third"].iter().enumerate() {
        for k in [1.1f64, 0.9] {
            let mut n = nights;
            n[i] *= k;
            let v = personal_sleep_need_hours(&n);
            t.push(format!("param: {label} night x{k:.2}"), Kind::Parameter, v, pass(v));
        }
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 8 — the rolling sleep-debt ledger.
// ---------------------------------------------------------------------------------------------

fn debt_series(mins: &[f64]) -> Vec<(String, Option<f64>)> {
    mins.iter().enumerate().map(|(i, m)| (format!("d{}", i + 1), Some(*m))).collect()
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_sleep_debt_balance() {
    let gate = format!("sleep_debt.rs:80 — a 360/600 min pair against an 8 h need balances to |x| < {DEBT_BALANCE_TOL}");
    let mut t = Table::new("sleep-debt ledger (rolling balance minutes)", gate);
    let pass = |v: f64| v.abs() < DEBT_BALANCE_TOL;
    let series = debt_series(&[360.0, 600.0]);

    let base = ledger(&series, Some(8.0), Some(14)).balance_min;
    t.push("baseline (360/600 min, need 8 h, window 14)", Kind::Baseline, base, pass(base));

    t.push("output: constant 0 balance", Kind::Null, 0.0, pass(0.0));
    t.push("output: constant 999 balance", Kind::Null, 999.0, pass(999.0));
    let v = ledger(&[], Some(8.0), Some(14)).balance_min;
    t.push("input: empty series", Kind::Null, v, pass(v));

    let mut rev = series.clone();
    rev.reverse();
    let v = ledger(&rev, Some(8.0), Some(14)).balance_min;
    t.push("input: series reversed", Kind::Structural, v, pass(v));
    let v = ledger(&debt_series(&[480.0, 480.0]), Some(8.0), Some(14)).balance_min;
    t.push("input: both nights exactly on need", Kind::Structural, v, pass(v));
    let v = ledger(&debt_series(&[360.0]), Some(8.0), Some(14)).balance_min;
    t.push("input: the surplus night dropped", Kind::Structural, v, pass(v));

    // need_hours and window are genuine arguments. DEFAULT_NEED_HOURS is a PRIVATE const,
    // DEFAULT_WINDOW_NIGHTS and ON_TARGET_BAND_MIN are `pub const` with no injection point.
    for k in [1.1f64, 0.9, 1.005] {
        let v = ledger(&series, Some(8.0 * k), Some(14)).balance_min;
        t.push(format!("param: need_hours 8.000 -> {:.3} (x{k:.3})", 8.0 * k), Kind::Parameter, v, pass(v));
    }
    for w in [15usize, 13, 1] {
        let v = ledger(&series, Some(8.0), Some(w)).balance_min;
        t.push(format!("param: window 14 -> {w} nights"), Kind::Parameter, v, pass(v));
    }
    let v = ledger(&series, None, None).balance_min;
    t.push("param: both defaults taken (proxy for DEFAULT_NEED_HOURS, a private const)", Kind::Parameter, v, pass(v));
    for k in [1.1f64, 0.9] {
        let scaled = debt_series(&[360.0 * k, 600.0 * k]);
        let v = ledger(&scaled, Some(8.0), Some(14)).balance_min;
        t.push(format!("param: both nights x{k:.2}"), Kind::Parameter, v, pass(v));
    }
    t.finish();
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_sleep_debt_window() {
    let gate = format!(
        "sleep_debt.rs:98-100 — a 20-night series keeps exactly {DEBT_WINDOW_NIGHTS} nights, oldest \
         \"d6\" (value column = counted nights, scored only when the oldest day is right)"
    );
    let mut t = Table::new("sleep-debt ledger (trailing window edge)", gate);
    let series: Vec<(String, Option<f64>)> = (0..20).map(|i| (format!("d{i}"), Some(500.0))).collect();
    let measure = |need: Option<f64>, w: Option<usize>| -> (f64, bool) {
        let l = ledger(&series, need, w);
        let ok = l.night_count() == DEBT_WINDOW_NIGHTS && l.nights.first().map(|n| n.day.as_str()) == Some("d6");
        (l.night_count() as f64, ok)
    };

    let (v, ok) = measure(Some(8.0), Some(14));
    t.push("baseline (window 14 over 20 nights)", Kind::Baseline, v, ok);
    t.push("output: constant 0 nights", Kind::Null, 0.0, false);
    t.push("output: constant 14 nights but the wrong oldest day", Kind::Null, 14.0, false);

    let mut rev = series.clone();
    rev.reverse();
    let l = ledger(&rev, Some(8.0), Some(14));
    let ok = l.night_count() == DEBT_WINDOW_NIGHTS && l.nights.first().map(|n| n.day.as_str()) == Some("d6");
    t.push("input: series reversed", Kind::Structural, l.night_count() as f64, ok);
    let holed: Vec<(String, Option<f64>)> =
        series.iter().enumerate().map(|(i, (d, s))| (d.clone(), if i % 3 == 0 { None } else { *s })).collect();
    let l = ledger(&holed, Some(8.0), Some(14));
    let ok = l.night_count() == DEBT_WINDOW_NIGHTS && l.nights.first().map(|n| n.day.as_str()) == Some("d6");
    t.push("input: every third night has no sleep data", Kind::Structural, l.night_count() as f64, ok);

    for w in [15usize, 13, 20, 1] {
        let (v, ok) = measure(Some(8.0), Some(w));
        t.push(format!("param: window 14 -> {w}"), Kind::Parameter, v, ok);
    }
    for k in [1.1f64, 0.9] {
        let (v, ok) = measure(Some(8.0 * k), Some(14));
        t.push(format!("param: need_hours x{k:.2} (must not move the count)"), Kind::Parameter, v, ok);
    }
    let (v, ok) = measure(None, None);
    t.push("param: both defaults taken", Kind::Parameter, v, ok);
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 9 — Sleep Regularity Index.
// ---------------------------------------------------------------------------------------------

fn night_span(d: i64, start_hour: f64, hours: f64) -> (i64, i64) {
    let start = d * SECONDS_PER_DAY + (start_hour * 3600.0) as i64;
    (start, start + (hours * 3600.0) as i64)
}

/// Nights for days -1..days, so day 0 carries the morning block of the night before the grid.
fn sri_nights(days: i64, start_hour: f64, hours: f64) -> Vec<(i64, i64)> {
    (-1..days).map(|d| night_span(d, start_hour, hours)).collect()
}

fn full_cover(days: i64) -> Vec<(i64, i64)> {
    vec![(0, days * SECONDS_PER_DAY)]
}

fn constant_grid(days: usize, state: EpochState) -> Vec<Vec<EpochState>> {
    vec![vec![state; EPOCHS_PER_DAY]; days]
}

fn sri_or_nan(grid: &[Vec<EpochState>]) -> f64 {
    sleep_regularity_index(grid).unwrap_or(f64::NAN)
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_sleep_regularity_index() {
    let gate = format!("sleep_regularity.rs:137 — an identical 23:00-07:00 schedule is SRI {SRI_PERFECT} within {SRI_TOL:e}");
    let mut t = Table::new("Sleep Regularity Index", gate);
    let pass = |v: f64| within(v, SRI_PERFECT, SRI_TOL);

    let base_grid = epoch_grid(0, 8, &sri_nights(8, 23.0, 8.0), &full_cover(8));
    let base = sri_or_nan(&base_grid);
    t.push("baseline (identical schedule, 8 days)", Kind::Baseline, base, pass(base));

    // NULL arms.
    let mut shuffled = base_grid.clone();
    for (d, day) in shuffled.iter_mut().enumerate() {
        lcg_shuffle(day, 0xD1CE + d as u64);
    }
    let v = sri_or_nan(&shuffled);
    t.push("input: epoch states shuffled within each day", Kind::Null, v, pass(v));
    let v = sri_or_nan(&constant_grid(8, Some(false)));
    t.push("input: never asleep (a detector that does no work)", Kind::Null, v, pass(v));
    let v = sri_or_nan(&constant_grid(8, Some(true)));
    t.push("input: always asleep", Kind::Null, v, pass(v));
    let v = sri_or_nan(&constant_grid(8, None));
    t.push("input: never worn (every epoch unknown)", Kind::Null, v, pass(v));

    // STRUCTURAL arms.
    let mut rotated = base_grid.clone();
    for day in rotated.iter_mut() {
        day.rotate_right(1);
    }
    let v = sri_or_nan(&rotated);
    t.push("input: every day rotated by one epoch (a uniform 30 s shift)", Kind::Structural, v, pass(v));
    let mut drift = base_grid.clone();
    for (d, day) in drift.iter_mut().enumerate() {
        day.rotate_right(d * 2);
    }
    let v = sri_or_nan(&drift);
    t.push("input: a one-minute-per-day drift", Kind::Structural, v, pass(v));
    let alternating: Vec<(i64, i64)> =
        (-1..8).map(|d| night_span(d, if d.rem_euclid(2) == 0 { 22.0 } else { 2.0 }, 8.0)).collect();
    let v = sri_or_nan(&epoch_grid(0, 8, &alternating, &full_cover(8)));
    t.push("input: bedtime alternating by 4 h", Kind::Structural, v, pass(v));
    let inverted: Vec<(i64, i64)> = (0..8)
        .filter(|d| d % 2 == 0)
        .map(|d| (d * SECONDS_PER_DAY, d * SECONDS_PER_DAY + SECONDS_PER_DAY))
        .collect();
    let v = sri_or_nan(&epoch_grid(0, 8, &inverted, &full_cover(8)));
    t.push("input: a perfectly anti-phase schedule", Kind::Structural, v, pass(v));

    // PARAMETER arms. `days` and `first_local_midnight` are arguments to epoch_grid and `max_gap_s` to
    // coverage_spans; EPOCH_SECONDS, MIN_PAIRED_DAYS, MIN_PAIRED_COVERAGE and MAX_WEAR_GAP_SECONDS are
    // `pub const` read directly by the index with no injection point.
    for d in [9usize, 7, 6, 5] {
        let v = sri_or_nan(&epoch_grid(0, d, &sri_nights(d as i64, 23.0, 8.0), &full_cover(d as i64)));
        t.push(format!("param: days 8 -> {d}"), Kind::Parameter, v, pass(v));
    }
    for k in [1.1f64, 0.9] {
        let gap = (300.0 * k).round() as i64;
        // One 285 s reporting hole a day: the shipped 300 s bridges it, a tightened gap does not.
        let mut ts: Vec<i64> = Vec::new();
        for d in 0..8 {
            let day = d * SECONDS_PER_DAY;
            let mut s = 0;
            while s < SECONDS_PER_DAY {
                if !(43_200..43_200 + 285).contains(&s) {
                    ts.push(day + s);
                }
                s += 60;
            }
        }
        let covered = coverage_spans(&ts, gap);
        let v = sri_or_nan(&epoch_grid(0, 8, &sri_nights(8, 23.0, 8.0), &covered));
        t.push(format!("param: coverage_spans max_gap_s 300 -> {gap} (x{k:.2})"), Kind::Parameter, v, pass(v));
    }
    for shift in [8_640i64, -8_640] {
        let v = sri_or_nan(&epoch_grid(shift, 8, &sri_nights(8, 23.0, 8.0), &full_cover(8)));
        t.push(format!("param: first_local_midnight 0 -> {shift} s (10% of a day)"), Kind::Parameter, v, pass(v));
    }
    t.finish();
}

// ---------------------------------------------------------------------------------------------
// Metric 10 — nap detection.
// ---------------------------------------------------------------------------------------------

/// Gravity rows every `step` s over `[t0, t0+dur)` with per-record motion exactly `move_g`.
fn nap_gravity(t0: i64, dur: i64, step: i64, move_g: f64) -> Vec<GravitySample> {
    let mut out = Vec::new();
    let (mut t, mut x, mut flip) = (t0, 0.0f64, 1.0f64);
    while t < t0 + dur {
        x += flip * move_g;
        flip = -flip;
        out.push(GravitySample { ts: t, x, y: 0.0, z: 0.0 });
        t += step;
    }
    out
}

fn nap_hr(t0: i64, dur: i64, bpm: i32) -> Vec<HrSample> {
    (0..dur).step_by(30).map(|d| HrSample { ts: t0 + d, bpm }).collect()
}

struct NapCase {
    name: &'static str,
    gravity: Vec<GravitySample>,
    hr: Vec<HrSample>,
    resting: Option<i32>,
    enabled: bool,
    want: NapVerdict,
}

/// The six scenarios nap.rs:224-273 pins, each with the verdict it must return.
fn nap_cases() -> Vec<NapCase> {
    vec![
        NapCase {
            name: "disabled",
            gravity: nap_gravity(0, 3600, 30, 0.0),
            hr: Vec::new(),
            resting: None,
            enabled: false,
            want: NapVerdict::Inconclusive,
        },
        NapCase {
            name: "sparse window",
            gravity: nap_gravity(0, 40 * 300, 300, 0.0),
            hr: Vec::new(),
            resting: Some(55),
            enabled: true,
            want: NapVerdict::Inconclusive,
        },
        NapCase {
            name: "still + settled",
            gravity: nap_gravity(1000, 40 * 60, 30, 0.0),
            hr: nap_hr(1000, 40 * 60, 57),
            resting: Some(55),
            enabled: true,
            want: NapVerdict::Nap,
        },
        NapCase {
            name: "still + elevated HR",
            gravity: nap_gravity(1000, 40 * 60, 30, 0.0),
            hr: nap_hr(1000, 40 * 60, 80),
            resting: Some(55),
            enabled: true,
            want: NapVerdict::None,
        },
        NapCase {
            name: "moving",
            gravity: nap_gravity(1000, 40 * 60, 30, 0.3),
            hr: Vec::new(),
            resting: Some(55),
            enabled: true,
            want: NapVerdict::None,
        },
        NapCase {
            name: "two hours still",
            gravity: nap_gravity(1000, 120 * 60, 30, 0.0),
            hr: Vec::new(),
            resting: Some(55),
            enabled: true,
            want: NapVerdict::Inconclusive,
        },
    ]
}

fn nap_share(cfg: &NapConfig, map: impl Fn(NapVerdict) -> NapVerdict, thin: usize) -> f64 {
    let cases = nap_cases();
    let mut hits = 0usize;
    for c in &cases {
        let cfg = NapConfig { enabled: c.enabled && cfg.enabled, ..*cfg };
        let g: Vec<GravitySample> = c.gravity.iter().copied().step_by(thin.max(1)).collect();
        if map(nap_evaluate(&g, &c.hr, c.resting, &cfg).verdict) == c.want {
            hits += 1;
        }
    }
    hits as f64 / cases.len() as f64
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_nap_verdicts() {
    let gate = format!("nap.rs:227-272 — all six shipped scenarios return their verdict (share == {NAP_VERDICT_SHARE})");
    let mut t = Table::new("nap detection (tri-state verdict)", gate);
    let pass = |v: f64| within(v, NAP_VERDICT_SHARE, 1e-12);
    let shipped = NapConfig { enabled: true, ..Default::default() };

    println!();
    println!("nap scenarios the shipped gate pins:");
    for c in &nap_cases() {
        let cfg = NapConfig { enabled: c.enabled, ..shipped };
        let got = nap_evaluate(&c.gravity, &c.hr, c.resting, &cfg).verdict;
        println!("  {:<22} want {:?}, got {:?}", c.name, c.want, got);
    }

    let v = nap_share(&shipped, |x| x, 1);
    t.push("baseline (NapConfig::default thresholds)", Kind::Baseline, v, pass(v));

    for (name, want) in [
        ("output: every verdict = Inconclusive", NapVerdict::Inconclusive),
        ("output: every verdict = Nap", NapVerdict::Nap),
        ("output: every verdict = None", NapVerdict::None),
    ] {
        let v = nap_share(&shipped, move |_| want, 1);
        t.push(name, Kind::Null, v, pass(v));
    }

    let v = nap_share(&shipped, |x| match x {
        NapVerdict::None => NapVerdict::Inconclusive,
        NapVerdict::Inconclusive => NapVerdict::None,
        x => x,
    }, 1);
    t.push("output: None <-> Inconclusive swapped", Kind::Structural, v, pass(v));
    let v = nap_share(&shipped, |x| match x {
        NapVerdict::Nap => NapVerdict::None,
        NapVerdict::None => NapVerdict::Nap,
        x => x,
    }, 1);
    t.push("output: Nap <-> None swapped", Kind::Structural, v, pass(v));
    // Proxy for MAX_GAP_S / DEFAULT_MIN_GRAVITY_SAMPLES / DEFAULT_MAX_MEDIAN_GAP_S, which `evaluate`
    // reads as module consts rather than from the config.
    for thin in [2usize, 4] {
        let v = nap_share(&shipped, |x| x, thin);
        t.push(format!("input: gravity thinned to 1 in {thin} (density proxy)"), Kind::Structural, v, pass(v));
    }

    // PARAMETER arms — NapConfig is the algorithm's own params struct.
    let cknobs: Vec<Knob<NapConfig>> = vec![
        ("min_nap_minutes", |c, k| c.min_nap_minutes = (c.min_nap_minutes as f64 * k).round() as i32, |c| c.min_nap_minutes as f64),
        ("max_nap_minutes", |c, k| c.max_nap_minutes = (c.max_nap_minutes as f64 * k).round() as i32, |c| c.max_nap_minutes as f64),
        ("still_threshold_g", |c, k| c.still_threshold_g *= k, |c| c.still_threshold_g),
        (
            "hr_settle_margin_bpm",
            |c, k| c.hr_settle_margin_bpm = (c.hr_settle_margin_bpm as f64 * k).round() as i32,
            |c| c.hr_settle_margin_bpm as f64,
        ),
        ("smooth_window_seconds", |c, k| c.smooth_window_seconds *= k, |c| c.smooth_window_seconds),
    ];
    for (name, set, get) in cknobs {
        for k in [1.1f64, 0.9] {
            let mut c = shipped;
            set(&mut c, k);
            let v = nap_share(&c, |x| x, 1);
            t.push(
                format!("param: {name} {:.3} -> {:.3} (x{k:.2})", get(&shipped), get(&c)),
                Kind::Parameter,
                v,
                pass(v),
            );
        }
    }
    t.finish();

    // The confidence NUMBER has no shipped gate at all — nap.rs:247 only pins the 0..1 range — so it is
    // reported rather than scored.
    let d = nap_evaluate(&nap_cases()[2].gravity, &nap_cases()[2].hr, Some(55), &shipped);
    let c = d.candidate.expect("the still + settled scenario must offer a candidate");
    println!();
    println!("=== nap confidence — UNGATED, reported only ===");
    println!("shipped gate: nap.rs:247-248 pins only 0 < confidence <= 1 and mean_hr == Some(57)");
    println!("baseline confidence {:.4}, mean_hr {:?}", c.confidence, c.mean_hr);
    for k in [1.1f64, 0.9] {
        let cfg = NapConfig { max_nap_minutes: (90.0 * k).round() as i32, ..shipped };
        let d = nap_evaluate(&nap_cases()[2].gravity, &nap_cases()[2].hr, Some(55), &cfg);
        let conf = d.candidate.map(|x| x.confidence).unwrap_or(f64::NAN);
        println!("max_nap_minutes x{k:.2}: confidence {conf:.4} (delta {:+.4})", conf - c.confidence);
    }
    assert!(c.confidence > 0.0 && c.confidence <= 1.0, "the shipped confidence range must hold");
    assert_eq!(Some(57), c.mean_hr, "the shipped mean-HR baseline must reproduce");
}

// ---------------------------------------------------------------------------------------------
// Metric 11 — a sleep error propagating into Rest and Charge.
// ---------------------------------------------------------------------------------------------

const CHAIN_START: i64 = 1_749_513_600;
const CHAIN_HOURS: i64 = 8;

fn chain_beats() -> Vec<(u32, u16)> {
    let total = CHAIN_HOURS * 3600;
    (0..total)
        .map(|i| {
            let amp = 60 - 52 * i / total;
            let wobble = [0, amp, 0, -amp][(i % 4) as usize];
            ((CHAIN_START + i) as u32, (1_090 + wobble) as u16)
        })
        .collect()
}

fn chain_hypnogram() -> Vec<StageSegment> {
    let h = 3600;
    let seg = |a: i64, b: i64, stage: SleepStage| StageSegment { start: CHAIN_START + a, end: CHAIN_START + b, stage };
    vec![
        seg(0, h, SleepStage::Light),
        seg(h, 2 * h, SleepStage::Deep),
        seg(2 * h, 3 * h, SleepStage::Light),
        seg(3 * h, 3 * h + 1800, SleepStage::Wake),
        seg(3 * h + 1800, 4 * h, SleepStage::Light),
        seg(4 * h, 5 * h, SleepStage::Rem),
        seg(5 * h, CHAIN_HOURS * h, SleepStage::Light),
    ]
}

fn relabel(segs: &[StageSegment], pp: f64, from: SleepStage, to: SleepStage) -> Vec<StageSegment> {
    let span: i64 = segs.iter().map(|s| s.end - s.start).sum();
    let mut budget = (pp / 100.0 * span as f64).round() as i64;
    let mut out = Vec::new();
    for s in segs {
        if s.stage != from || budget <= 0 {
            out.push(*s);
            continue;
        }
        let take = budget.min(s.end - s.start);
        budget -= take;
        out.push(StageSegment { start: s.start, end: s.start + take, stage: to });
        if s.start + take < s.end {
            out.push(StageSegment { start: s.start + take, end: s.end, stage: from });
        }
    }
    assert_eq!(0, budget, "not enough {from:?} to move {pp} pp");
    out
}

struct ChainNight {
    asleep_s: f64,
    efficiency: f64,
    deep_s: f64,
    rem_s: f64,
    hrv: f64,
}

fn measure_chain(segs: &[StageSegment]) -> ChainNight {
    let secs = |want: SleepStage| -> f64 {
        segs.iter().filter(|s| s.stage == want).map(|s| (s.end - s.start) as f64).sum()
    };
    let (wake, light, deep, rem) =
        (secs(SleepStage::Wake), secs(SleepStage::Light), secs(SleepStage::Deep), secs(SleepStage::Rem));
    let in_bed = wake + light + deep + rem;
    let deep_spans: Vec<(u32, u32)> = segs
        .iter()
        .filter(|s| s.stage == SleepStage::Deep)
        .map(|s| (s.start as u32, s.end as u32))
        .collect();
    let hrv = HrvReadiness::windowed_avg_hrv_deep(
        CHAIN_START as u32,
        (CHAIN_START + CHAIN_HOURS * 3600) as u32,
        &chain_beats(),
        &deep_spans,
    )
    .expect("the crafted night must yield a deep-window HRV");
    ChainNight { asleep_s: light + deep + rem, efficiency: (light + deep + rem) / in_bed, deep_s: deep, rem_s: rem, hrv }
}

/// Every scalar the chain reads that a caller can actually supply.
#[derive(Clone, Copy)]
struct ChainKnobs {
    need_h: f64,
    rhr: f64,
    hrv_mean: f64,
    hrv_spread: f64,
    rhr_mean: f64,
    rhr_spread: f64,
}

impl ChainKnobs {
    const SHIPPED: ChainKnobs =
        ChainKnobs { need_h: 8.0, rhr: 52.0, hrv_mean: 45.0, hrv_spread: 8.0, rhr_mean: 55.0, rhr_spread: 3.0 };
}

fn chain_charge(hrv_from: &ChainNight, rest_from: &ChainNight, k: &ChainKnobs) -> f64 {
    let sleep_perf = rest(rest_from.asleep_s, rest_from.efficiency, rest_from.deep_s, rest_from.rem_s, Some(k.need_h), None)
        .expect("asleep time is positive")
        / 100.0;
    recovery(&RecoveryInput {
        hrv: hrv_from.hrv,
        rhr: k.rhr,
        hrv_baseline: Some(DriverBaseline { mean: k.hrv_mean, spread: k.hrv_spread }),
        rhr_baseline: Some(DriverBaseline { mean: k.rhr_mean, spread: k.rhr_spread }),
        sleep_perf: Some(sleep_perf),
        ..Default::default()
    })
    .expect("every driver is present")
}

/// The chain gain the shipped gate bounds: how much Charge moves per point of stage error.
fn chain_gain(pp: f64, from: SleepStage, to: SleepStage, k: &ChainKnobs) -> f64 {
    let base = measure_chain(&chain_hypnogram());
    let hurt = measure_chain(&relabel(&chain_hypnogram(), pp, from, to));
    (chain_charge(&base, &base, k) - chain_charge(&hurt, &hurt, k)).abs() / 10.0
}

#[test]
#[ignore = "negative control; not a CI gate — run with --ignored"]
fn sensitivity_sleep_error_propagation() {
    let gate = format!(
        "sleep_error_propagation.rs:137-138 — a 10 pp wake error must give a chain gain strictly \
         between {CHAIN_GAIN_FLOOR} and {CHAIN_GAIN_CEIL}"
    );
    let mut t = Table::new("sleep-error propagation into Rest and Charge (chain gain)", gate);
    let pass = |v: f64| v > CHAIN_GAIN_FLOOR && v < CHAIN_GAIN_CEIL;
    let k0 = ChainKnobs::SHIPPED;

    let base = chain_gain(10.0, SleepStage::Light, SleepStage::Wake, &k0);
    t.push("baseline (10 pp Light -> Wake)", Kind::Baseline, base, pass(base));

    // NULL arms: a chain that transmits nothing.
    t.push("output: constant 0 gain (the chain swallows the error)", Kind::Null, 0.0, pass(0.0));
    let v = chain_gain(0.0, SleepStage::Light, SleepStage::Wake, &k0);
    t.push("input: a 0 pp error (nothing was hurt)", Kind::Null, v, pass(v));
    t.push("output: constant 5.0 gain (the chain amplifies)", Kind::Null, 5.0, pass(5.0));

    // STRUCTURAL arms.
    for (name, from, to) in [
        ("input: 10 pp Light -> Deep", SleepStage::Light, SleepStage::Deep),
        ("input: 10 pp Light -> REM", SleepStage::Light, SleepStage::Rem),
    ] {
        let v = chain_gain(10.0, from, to, &k0);
        t.push(name, Kind::Structural, v, pass(v));
    }
    // The crafted night holds 1800 s of Wake in a 28800 s span, so 10 pp of Wake does not exist to undo.
    let v = chain_gain(6.0, SleepStage::Wake, SleepStage::Light, &k0);
    t.push("input: 6 pp Wake -> Light (all the Wake this night has is 6.25 pp)", Kind::Structural, v, pass(v));
    for pp in [20.0f64, 5.0, 1.0] {
        let v = chain_gain(pp, SleepStage::Light, SleepStage::Wake, &k0);
        t.push(format!("input: {pp} pp Light -> Wake (gain still divided by 10)"), Kind::Structural, v, pass(v));
    }

    // PARAMETER arms. The recovery weights (W_HRV, W_SLEEP, ...), LOGISTIC_K/Z0, SLEEP_PERF_CENTER and
    // the Rest weights are all `pub const` with no injection point; these are the caller-supplied ones.
    let ck: Vec<Knob<ChainKnobs>> = vec![
        ("sleep_need_hours", |c, k| c.need_h *= k, |c| c.need_h),
        ("rhr", |c, k| c.rhr *= k, |c| c.rhr),
        ("hrv_baseline.mean", |c, k| c.hrv_mean *= k, |c| c.hrv_mean),
        ("hrv_baseline.spread", |c, k| c.hrv_spread *= k, |c| c.hrv_spread),
        ("rhr_baseline.mean", |c, k| c.rhr_mean *= k, |c| c.rhr_mean),
        ("rhr_baseline.spread", |c, k| c.rhr_spread *= k, |c| c.rhr_spread),
    ];
    for (name, set, get) in ck {
        for k in [1.1f64, 0.9] {
            let mut c = k0;
            set(&mut c, k);
            let v = chain_gain(10.0, SleepStage::Light, SleepStage::Wake, &c);
            t.push(
                format!("param: {name} {:.3} -> {:.3} (x{k:.2})", get(&k0), get(&c)),
                Kind::Parameter,
                v,
                pass(v),
            );
        }
    }
    let mut c = k0;
    c.need_h *= 1.005;
    let v = chain_gain(10.0, SleepStage::Light, SleepStage::Wake, &c);
    t.push("param: sleep_need_hours x1.005 (floor probe)", Kind::Parameter, v, pass(v));
    t.finish();
}
