//! Shared fixture loading for the analysis harnesses, and the refinement guard.
//!
//! Every stream a harness reads out of the fixture corpus has its one reader here, `steps.csv` included —
//! the app runs detect -> stage -> refine_wake, and a harness with no step stream stages only the first
//! two. `verify_backup` is the one exception and reads a user-supplied backup export, not the corpus.
//! `tools/docs-vs-code.py` re-checks that, because this paragraph has now been false twice.
//!
//! [`RefineCensus`] is how a harness refines under SHIPPED params: it counts which side of the density
//! gate each span fell on, so an unrefined span is never pooled into a figure LABELLED refined. Two
//! harnesses reach past it on purpose — `wake_refine` sweeps `refine_wake_with` under other params, which
//! is its subject, and `sleep_to_charge` runs its own `motion_density` gate at load time.

#![allow(dead_code, unused_imports)]

pub mod cardiac;
pub mod mesa;
pub mod screen;
pub mod lr;
pub use cardiac::{cardiac_series, per_second_hr, rank_pct, std_of_seconds, zscore};

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    motion_density, params::Params, refine_wake, AccelSample, HrSample, RrRun, SleepStage, StageSegment,
    StepSample, MIN_DENSE_FRACTION,
};

/// The de-duplicated, fidelity-repaired corpus, and the default for every harness. The raw
/// `fixtures_multi` root holds each beat twice on 16 of its 92 `ours` nights and 6 of its 64 `continuous`
/// blocks, and `fixtures_multi_clean` still holds the second wearer-side duplication on 30 and 36 of them,
/// so defaulting to either would hand back a doubled R-R stream with no error. `_clean3` further repairs
/// `aauwss/subject_12`'s placeholder gravity and all 31 gap-filled `sleep-accel` HR streams, and adds the
/// DREAMT temp / BVP / apnea-event / demographics channels. `WHOOP_SLEEP_FIXTURES` overrides.
const DEFAULT_ROOT: &str = "C:/Users/DavidGillot/Projects/whoop/sleep-benchmark/fixtures_multi_clean3";

pub fn fixtures_root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_ROOT))
}

pub fn root(set: &str) -> PathBuf {
    fixtures_root().join(set)
}

/// The fixture directories of one set, sorted, so two harnesses iterate in the same order.
pub fn dirs_of(set: &str) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(root(set))
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    d.sort();
    d
}

/// One REQUIRED stream as rows of columns. An unreadable file is FATAL and names itself: reading empty
/// instead stages the night with that stream absent, which every harness reports as a real figure.
pub fn read_csv(path: &Path) -> Vec<Vec<f64>> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("fixture stream unreadable: {} ({e})", path.display()));
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| l.split(',').map(|c| parse_cell(c, path, i + 1)).collect())
        .collect()
}

/// One cell, naming the file and its 1-based line on a bad number.
fn parse_cell(cell: &str, path: &Path, line: usize) -> f64 {
    cell.trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("fixture stream {} line {line}: {cell:?} ({e})", path.display()))
}

/// A stream a set may legitimately not carry at all — `truth`, `band`, `steps`, `dynaccel`. ABSENT reads
/// empty, which those readers already treat as "this set cannot answer"; anything else still panics.
pub fn read_csv_optional(path: &Path) -> Vec<Vec<f64>> {
    if path.exists() { read_csv(path) } else { Vec::new() }
}

pub fn read_hr(dir: &Path) -> Vec<HrSample> {
    read_csv(&dir.join("hr.csv")).iter().map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 }).collect()
}

pub fn read_accel(dir: &Path) -> Vec<AccelSample> {
    read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect()
}

/// Consecutive same-timestamp rows of `rr.csv` grouped into one run, which is how the strap reported them.
pub fn read_rr(dir: &Path) -> Vec<RrRun> {
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let (ts, ms) = (row[0] as i64, row[1] as u16);
        match rr.last_mut() {
            Some(l) if l.ts == ts => l.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }
    rr
}

/// Whole-second beat stamps spread back out by their own intervals, the reconstruction the stager
/// runs before any cardiac statistic sees them. THE LIBRARY's, so a harness and the emission read
/// the same beats; `mesa::degrade_timing` models the same wire but rounds and does not clamp.
pub use physio_algo::sleep::cardiac::reconstruct_beats;

/// `truth.csv` as epoch index -> label. Empty when the set carries no labels, which is how a harness
/// tells "unlabelled" apart from "all wake" — several sets have no truth at all.
pub fn read_truth(dir: &Path) -> BTreeMap<usize, i32> {
    read_csv_optional(&dir.join("truth.csv")).iter().map(|r| (r[0] as usize, r[1] as i32)).collect()
}

/// `meta.txt` as (window start, window end, epoch count). A fixture night is rebased onto a synthetic
/// clock, so this is the FIXTURE window, not the real one. `None` means "no epoch grid here" and has
/// exactly two causes, both legitimate; every other shape is FATAL, because a `None` deletes the night
/// from all 20 harnesses that read this and the sheet still prints a total.
pub fn read_meta(dir: &Path) -> Option<(i64, i64, usize)> {
    let path = dir.join("meta.txt");
    // Not a fixture directory at all.
    let text = fs::read_to_string(&path).ok()?;
    let m: Vec<i64> = text.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    match m[..] {
        [_, w0, w1, epochs, ..] => Some((w0, w1, epochs as usize)),
        // `continuous` writes owner and device first and carries no epochs, so its two numbers ARE
        // the window; it is scored on instruments that need no epoch grid.
        [_, _] => None,
        _ => panic!("fixture meta is neither 4 numbers nor the 2-number continuous shape: {}", path.display()),
    }
}

/// Owner and real unix onset out of an `owner_device_day_onset` fixture directory name. The real onset is
/// the only link a rebased night keeps to the wearer's own clock.
pub fn night_id(dir: &Path) -> (String, i64) {
    let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
    let owner = name.split('_').next().unwrap_or("?").to_string();
    let onset = name.rsplit('_').next().and_then(|s| s.parse::<i64>().ok()).unwrap_or(0);
    (owner, onset)
}

/// Nearest-first 1:1 pairing of two (owner, onset) lists, inside `slack` seconds. Returns index pairs.
pub fn pair_nearest(a: &[(String, i64)], b: &[(String, i64)], slack: i64) -> Vec<(usize, usize)> {
    let mut cand: Vec<(i64, usize, usize)> = Vec::new();
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let d = (x.1 - y.1).abs();
            if x.0 == y.0 && d <= slack {
                cand.push((d, i, j));
            }
        }
    }
    cand.sort();
    let (mut used_a, mut used_b, mut out) = (Vec::new(), Vec::new(), Vec::new());
    for (_, i, j) in cand {
        if !used_a.contains(&i) && !used_b.contains(&j) {
            used_a.push(i);
            used_b.push(j);
            out.push((i, j));
        }
    }
    out.sort();
    out
}

/// The recipe the stager used before its current tune, over any base. Five harnesses score against it, so
/// it lives here rather than as a literal in each.
pub fn pre_retune(base: &Params) -> Params {
    Params {
        deep_hrv: -1.1, deep_hr: 0.0, deep_motion: -0.5,
        rem_hrv: 0.6, rem_motion: -0.6, rem_hr: 0.4,
        awake_motion: 1.0, awake_hrv: 0.8, awake_hr: 0.4, awake_deadzone: 0.0,
        deep_gate_thresh: 0.25, jerk_move_mult: 38.0, jerk_gate_mult: 55.0, motion_gate_boost: 2.0,
        base_rate: [0.18, 0.22, 0.50, 0.10],
        transition: [
            [0.86, 0.007, 0.126, 0.007],
            [0.005, 0.88, 0.10, 0.015],
            [0.06, 0.06, 0.85, 0.03],
            [0.01, 0.02, 0.27, 0.70],
        ],
        ..*base
    }
}

/// Two-class scoreboard against a wake/asleep reference. Never accuracy: these references call ~90% of
/// their seconds asleep. Against a period marker read [`TwoClass::recall`] only — `kappa` and
/// `precision` both reward under-calling wake there. Presentation stays with each harness.
#[derive(Default, Clone, Copy)]
pub struct TwoClass {
    pub n: i64,
    pub pred_wake: i64,
    pub true_wake: i64,
    pub hit: i64,
}

impl TwoClass {
    pub fn add(&mut self, pred_wake: bool, true_wake: bool) {
        self.n += 1;
        self.pred_wake += i64::from(pred_wake);
        self.true_wake += i64::from(true_wake);
        self.hit += i64::from(pred_wake && true_wake);
    }
    /// The share of seconds WE call wake.
    pub fn pred_pct(&self) -> f64 {
        100.0 * self.pred_wake as f64 / self.n.max(1) as f64
    }
    /// The share the REFERENCE calls wake. Print it beside `pred_pct`, or a ratio has no denominator.
    pub fn true_pct(&self) -> f64 {
        100.0 * self.true_wake as f64 / self.n.max(1) as f64
    }
    pub fn recall(&self) -> f64 {
        100.0 * self.hit as f64 / self.true_wake.max(1) as f64
    }
    /// Valid only against a true hypnogram. Against [`BAND_ASLEEP`]'s period marker the denominator
    /// counts unadjudicated epochs as false positives, so it RISES as a stager calls less wake —
    /// shipped 19.9, `pre-retune` at a third the wake 25.3. Never select on it.
    pub fn precision(&self) -> f64 {
        100.0 * self.hit as f64 / self.pred_wake.max(1) as f64
    }
    pub fn f1(&self) -> f64 {
        let (r, p) = (self.recall(), self.precision());
        if r + p <= 0.0 { 0.0 } else { 2.0 * r * p / (r + p) }
    }
    /// Cohen's kappa over the 2x2 table, which is accuracy with the base rate divided out. Carries
    /// [`Self::precision`]'s defect against a period marker — reporting only, never selection.
    pub fn kappa(&self) -> f64 {
        let n = self.n.max(1) as f64;
        let (pw, tw) = (self.pred_wake as f64, self.true_wake as f64);
        let agree = (self.hit as f64 + (n - pw - tw + self.hit as f64)) / n;
        let expect = (pw * tw + (n - pw) * (n - tw)) / (n * n);
        if expect >= 1.0 { 0.0 } else { (agree - expect) / (1.0 - expect) }
    }
}

/// The stage covering `ts` in a tiled hypnogram, or `None` past its end.
pub fn stage_at(segs: &[StageSegment], ts: i64) -> Option<SleepStage> {
    segs.iter().find(|g| g.start <= ts && ts < g.end).map(|g| g.stage)
}

/// The strap's `sleep_state` asleep code, and the `code != BAND_ASLEEP => awake` fold every band-scored
/// figure uses. Code 2 marks the sleep PERIOD, not per-epoch sleep: 42 runs of median 15,729 s over 29
/// nights, holding through 82.5% of our wake bouts. Awake is trustworthy, asleep is silent.
pub const BAND_ASLEEP: i32 = 2;

/// The label index of wake in a `Vec<usize>` epoch sequence — the same 0 [`stage_idx`] returns.
pub const WAKE: usize = 0;

/// The local calendar day `ts` falls in, as a day number. A main-night pick partitions on these.
pub fn local_day(ts: i64, offset_s: i64) -> i64 {
    (ts + offset_s).div_euclid(86_400)
}

/// Seconds the strap itself called asleep inside `[a, b]`, its band being 1 Hz.
pub fn band_asleep_secs(band: &[(i64, i32)], a: i64, b: i64) -> i64 {
    band.iter().filter(|(t, s)| *t >= a && *t <= b && *s == BAND_ASLEEP).count() as i64
}

/// First epoch of the first 10-epoch non-wake run: the onset rule the stager itself applies.
pub fn onset_of(seq: &[usize]) -> Option<usize> {
    let mut run = 0;
    for (i, &l) in seq.iter().enumerate() {
        run = if l == WAKE { 0 } else { run + 1 };
        if run == 10 {
            return Some(i - 9);
        }
    }
    None
}

/// The arithmetic mean, 0.0 rather than NaN on an empty slice.
pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// The strap's own asleep runs of at least `min_min` minutes, joining across gaps up to `tol_s`. The
/// reference every detection and wake figure is scored against.
pub fn asleep_runs(band: &[(i64, i32)], min_min: i64, tol_s: i64) -> Vec<(i64, i64)> {
    let (mut out, mut start, mut last) = (Vec::new(), None::<i64>, 0i64);
    for &(ts, st) in band {
        if st != BAND_ASLEEP {
            continue;
        }
        match start {
            None => start = Some(ts),
            Some(s) if ts - last > tol_s => {
                if last - s >= min_min * 60 {
                    out.push((s, last));
                }
                start = Some(ts);
            }
            _ => {}
        }
        last = ts;
    }
    if let Some(s) = start {
        if last - s >= min_min * 60 {
            out.push((s, last));
        }
    }
    out
}

/// The `p`-quantile by nearest rank, sorting in place. `pct(v, 0.5)` is the LOWER middle on an even
/// count, so it is a third thing again from [`median`] and [`median_avg`].
pub fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p) as usize]
}

/// The upper-middle order statistic, sorting in place. NOT the same as [`median_avg`] on an even count,
/// and published figures were taken under each, so the two are not interchangeable.
pub fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The median that averages the two middles on an even count. See [`median`] for why both exist.
pub fn median_avg(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// The `n` epoch labels of a hypnogram tiled from `w0`, each read at its epoch midpoint. Past the last
/// segment the final stage is held, so a caller always gets `n` labels.
pub fn labels_at(segs: &[StageSegment], w0: i64, n: usize, epoch_sec: i64) -> Vec<usize> {
    (0..n)
        .map(|k| {
            let mid = w0 + k as i64 * epoch_sec + epoch_sec / 2;
            stage_idx(stage_at(segs, mid).unwrap_or_else(|| segs.last().expect("staging is never empty here").stage))
        })
        .collect()
}

/// `STAGE_ORDER`'s index for a stage: the column order every confusion matrix and fraction table uses.
pub fn stage_idx(s: SleepStage) -> usize {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// Cohen's kappa over a 4-class confusion matrix (rows = truth, cols = predicted). Print it beside any
/// accuracy: these references call most epochs asleep, and raw accuracy rewards a config that says so.
pub use physio_algo::sleep::metrics::kappa4;

/// The strap's own `sleep_state`, 1 Hz, as `(ts, code)`.
pub fn read_band(dir: &Path) -> Vec<(i64, i32)> {
    read_csv_optional(&dir.join("band.csv")).iter().map(|r| (r[0] as i64, r[1] as i32)).collect()
}

/// The strap's own step stream. `-1` in column 3 means the class was not decoded, which is not "still".
/// A missing or empty file returns empty, which is exactly what makes the refinement decline in silence.
pub fn read_steps(dir: &Path) -> Vec<StepSample> {
    read_csv_optional(&dir.join("steps.csv"))
        .iter()
        .map(|r| StepSample {
            ts: r[0] as i64,
            counter: r[1] as u16,
            activity_class: (r[2] >= 0.0).then(|| r[2] as u8),
        })
        .collect()
}

/// The strap's own on-chip ENMO, `(ts, g)`, one row a second where the column is populated. Grafted onto
/// the pinned corpus by `sleep-benchmark/graft_dynaccel.py`; absent on every set but three `continuous`
/// blocks, so a reader must treat empty as "this block cannot answer" rather than "no motion".
pub fn read_dyn_accel(dir: &Path) -> Vec<(i64, f64)> {
    read_csv_optional(&dir.join("dynaccel.csv")).iter().map(|r| (r[0] as i64, r[1])).collect()
}

/// One night as WHOOP's own app reports it: its in-bed window and its stage shares, in `STAGE_ORDER`.
pub struct Export {
    pub owner: String,
    pub onset: i64,
    pub wake: i64,
    pub frac: [f64; 4],
}

/// Every export night under `whoop-labels.json`, onset-sorted. A row with a bad window or a missing share
/// is dropped rather than defaulted, so a night that reaches a table carries all four fractions.
pub fn read_export() -> Vec<Export> {
    let Ok(text) = fs::read_to_string(fixtures_root().join("whoop-labels.json")) else { return Vec::new() };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    let mut out = Vec::new();
    for (owner, nights) in root.as_object().into_iter().flatten() {
        for n in nights.as_array().into_iter().flatten() {
            let (Some(onset), Some(wake)) = (n["onset_unix"].as_i64(), n["wake_unix"].as_i64()) else {
                continue;
            };
            let frac = ["awake", "light", "deep", "rem"].map(|k| n[k].as_f64().unwrap_or(f64::NAN));
            if wake > onset && frac.iter().all(|v| v.is_finite()) {
                out.push(Export { owner: owner.clone(), onset, wake, frac });
            }
        }
    }
    out.sort_by_key(|e| e.onset);
    out
}

/// WHOOP's own published nightly HRV per wearer as `(owner, night onset unix, ms)`. The sibling of
/// [`read_export`]: same file shape, the other published quantity.
pub fn read_published_hrv() -> Vec<(String, i64, f64)> {
    let Ok(text) = fs::read_to_string(fixtures_root().join("whoop-hrv.json")) else { return Vec::new() };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    let mut out = Vec::new();
    for (owner, nights) in root.as_object().into_iter().flatten() {
        for n in nights.as_array().into_iter().flatten() {
            if let (Some(t), Some(h)) = (n["onset_unix"].as_i64(), n["hrv_ms"].as_f64()) {
                out.push((owner.clone(), t, h));
            }
        }
    }
    out
}

/// Which side of the refinement's density gate each span fell on, for one reported figure.
///
/// The gate wants a gravity sample twice a minute and a step sample once a minute over 80% of the
/// minutes, and it declines with no error. A figure mixing declined spans with refined ones is neither
/// path, so every harness that refines counts through this and prints [`RefineCensus::line`].
#[derive(Default, Clone, Copy)]
pub struct RefineCensus {
    /// Spans the gate accepted, so the refinement really ran.
    pub refined: usize,
    /// Spans the gate declined: staged by `stage_v2` alone, which is NOT the path the app runs.
    pub declined: usize,
    /// Refined spans whose labels actually moved.
    pub changed: usize,
    /// Of the declines, how many were the GRAVITY stream and how many the STEP stream. A gravity decline
    /// would happen on a device too; a step decline can be the fixture's gap rather than the strap's.
    pub declined_gravity: usize,
    pub declined_steps: usize,
}

impl RefineCensus {
    /// Refine one span's staging, counting the gate's answer. Returns the segments the app would hold.
    pub fn refine(&mut self, segs: &[StageSegment], grav: &[AccelSample], steps: &[StepSample]) -> Vec<StageSegment> {
        let (Some(f), Some(l)) = (segs.first(), segs.last()) else {
            self.declined += 1;
            return segs.to_vec();
        };
        if l.end <= f.start {
            self.declined += 1;
            return segs.to_vec();
        }
        let (g, s) = motion_density(f.start, l.end, grav, steps);
        if g < MIN_DENSE_FRACTION || s < MIN_DENSE_FRACTION {
            self.declined += 1;
            self.declined_gravity += usize::from(g < MIN_DENSE_FRACTION);
            self.declined_steps += usize::from(s < MIN_DENSE_FRACTION);
            return segs.to_vec();
        }
        self.refined += 1;
        let out = refine_wake(segs, grav, steps);
        self.changed += usize::from(out != segs);
        out
    }

    /// Fold another census in, so a per-span probe can be accumulated into a per-figure total.
    pub fn absorb(&mut self, other: &RefineCensus) {
        self.refined += other.refined;
        self.declined += other.declined;
        self.changed += other.changed;
        self.declined_gravity += other.declined_gravity;
        self.declined_steps += other.declined_steps;
    }

    pub fn total(&self) -> usize {
        self.refined + self.declined
    }

    /// True when every span counted went through the refinement, so the figure beside it is the app's path.
    pub fn all_refined(&self) -> bool {
        self.declined == 0 && self.refined > 0
    }

    /// One line naming the split. Print it beside any figure this census fed.
    pub fn line(&self, what: &str) -> String {
        let tail = if self.declined == 0 {
            " — refined throughout".to_string()
        } else {
            format!(
                " ({} on gravity, {} on steps) — MIXED, so this figure is neither path alone",
                self.declined_gravity, self.declined_steps
            )
        };
        format!(
            "   {what}: {} of {} spans REFINED ({} moved), {} DECLINED by the density gate{tail}",
            self.refined,
            self.total(),
            self.changed,
            self.declined,
        )
    }

    /// Stop a mixed figure being reported as refined. Use where the harness has already restricted itself
    /// to spans carrying a dense stream, so a decline means the restriction is wrong.
    pub fn require_all_refined(&self, what: &str) {
        assert!(
            self.all_refined(),
            "{what}: {} of {} spans were declined by the density gate, so this figure is a mixture",
            self.declined,
            self.total()
        );
    }
}

/// The registered user cohort: every backup a user has sent, as `(wearer, store path)`.
///
/// The counterweight to the PSG sets. PSG has per-epoch truth and two disqualifying flaws for some
/// questions - different hardware, and ONE NIGHT PER SUBJECT, which leaves within-wearer variance
/// undefined at any sample size. Ask this when the question is about a person rather than a night.
///
/// Registered by `dev-notes/cohort_add.py`, never hardcoded here: three harnesses carried their own
/// path lists before this existed and adding a store meant editing all three.
pub fn user_cohort() -> Vec<(String, String)> {
    const MANIFEST: &str =
        "C:/Users/DavidGillot/Projects/whoop/whoop-data/harnesses/user-cohort/manifest.json";
    let Ok(text) = fs::read_to_string(MANIFEST) else { return Vec::new() };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { return Vec::new() };
    v["stores"]
        .as_array()
        .into_iter()
        .flatten()
        // A store that cannot be scored is skipped here rather than contributing zero nights to a
        // total that then reads as coverage.
        .filter(|s| s["scorable"].as_bool() == Some(true))
        .filter_map(|s| {
            Some((s["wearer"].as_str()?.to_string(), s["path"].as_str()?.to_string()))
        })
        .collect()
}

/// Where a baseline's parameters came from, relative to the nights it is being scored on.
///
/// A baseline tuned on its own evaluation set is an UPPER BOUND, not an opponent, and a challenger
/// measured against it is understated by whatever that tuning was worth. [`compare`] cannot be called
/// without stating this, because three separate conclusions have been drawn against `Params::SHIPPED`
/// as if its cohort kappas were out-of-sample. They are not.
#[derive(Clone, Copy, PartialEq)]
pub enum Provenance {
    /// Fitted or selected without seeing any night in the cohort being scored.
    HeldOut,
    /// Selected while watching these nights. `what` names which parameters, so the reader can judge
    /// how much of any gap is the privilege rather than the model.
    InSample(&'static str),
}

/// A paired per-night comparison that refuses to be formatted without naming its baseline's
/// provenance. Returns `(mean, bar, line)`; the line carries a loud marker when the baseline is
/// in-sample, so an invalid comparison cannot be quoted as a clean one.
pub fn compare(base: &[f64], arm: &[f64], base_from: Provenance) -> (f64, f64, String) {
    compare_guarded(base, arm, base_from, None)
}

/// [`compare`] told what the arm's worst per-class recall did against the baseline's, so a headline
/// bought by abolishing a class reads DEGENERATE rather than AHEAD. The verdict lives in
/// `metrics::paired_verdict` because `cargo test` does not build a test inside an example.
pub fn compare_guarded(
    base: &[f64],
    arm: &[f64],
    base_from: Provenance,
    min_recall_delta: Option<f64>,
) -> (f64, f64, String) {
    assert_eq!(base.len(), arm.len(), "a paired comparison needs the same nights on both sides");
    let d: Vec<f64> = base.iter().zip(arm).map(|(a, b)| b - a).collect();
    let Some((mean, bar)) = physio_algo::sleep::metrics::paired_bar(&d) else {
        return (f64::NAN, f64::NAN, "too few paired nights".to_string());
    };
    let tag = physio_algo::sleep::metrics::paired_verdict(mean, bar, min_recall_delta).to_string();
    let line = match base_from {
        Provenance::HeldOut => format!("{mean:>+8.4} {bar:>7.4}  {tag}"),
        Provenance::InSample(what) => format!(
            "{mean:>+8.4} {bar:>7.4}  {tag}   <<< BASELINE IS IN-SAMPLE ({what}) - NOT A VALID GAP"
        ),
    };
    (mean, bar, line)
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    /// The marker is the whole point of the type, so a silent in-sample line is a failure.
    #[test]
    fn an_in_sample_baseline_is_labelled_and_a_held_out_one_is_not() {
        let (base, arm) = ([0.30, 0.32, 0.28, 0.31], [0.34, 0.35, 0.33, 0.36]);
        let (m1, _, held) = compare(&base, &arm, Provenance::HeldOut);
        let (m2, _, sample) = compare(&base, &arm, Provenance::InSample("12 emission weights"));
        assert_eq!(m1, m2, "provenance must not change the arithmetic, only how it is reported");
        assert!(!held.contains("IN-SAMPLE"), "a held-out baseline needs no warning: {held}");
        assert!(sample.contains("IN-SAMPLE") && sample.contains("12 emission weights"),
                "an in-sample baseline must name itself: {sample}");
    }
}

/// What a cohort's `truth.csv` actually IS. The filename says "truth" on every one of them.
///
/// `strap`, `whoop4` and `killa5` take theirs from `sleepSession.stagesJSON`, which is our OWN
/// on-board staging, so scoring a staging change against them measures how close the change is to v2
/// and the incumbent scores ~1.0 by construction. Nothing on disk distinguishes those 83 nights from
/// the 144 PSG ones, which is why this mapping exists in code rather than in a comment.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Reference {
    /// An independent per-epoch hypnogram. The only thing that can settle a staging verdict.
    Psg,
    /// Our own on-board output, kept as a port-versus-onboard consistency check. CIRCULAR for scoring.
    OurOwnOutput,
    /// The strap's `sleep_state`: two classes, and a sleep-PERIOD marker rather than per-epoch sleep.
    BandPeriod,
    /// No labels.
    Unlabelled,
}

/// The reference a fixture set carries. An unknown set is fatal rather than assumed labelled.
pub fn reference_of(set: &str) -> Reference {
    match set {
        "dreamt" | "aauwss" | "sleep-accel" => Reference::Psg,
        "strap" | "whoop4" | "killa5" => Reference::OurOwnOutput,
        "ours" => Reference::BandPeriod,
        "continuous" | "e9night" => Reference::Unlabelled,
        other => panic!("{other}: no declared reference. Read its builder before scoring against it"),
    }
}

/// Refuse to score a staging verdict against anything but an independent hypnogram.
pub fn require_psg(set: &str) {
    let r = reference_of(set);
    assert_eq!(r, Reference::Psg,
               "{set} carries {r:?}, not PSG. A 4-class verdict against it is not a verdict: \
                OurOwnOutput is circular and BandPeriod is a two-class period marker");
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    /// The whole point is refusing the circular sets, so a silent pass on `strap` is a failure.
    #[test]
    fn the_our_own_output_cohorts_are_refused_and_psg_is_allowed() {
        for set in ["dreamt", "aauwss", "sleep-accel"] {
            require_psg(set);
            assert_eq!(reference_of(set), Reference::Psg);
        }
        for set in ["strap", "whoop4", "killa5"] {
            assert_eq!(reference_of(set), Reference::OurOwnOutput, "{set} is our own staging");
            assert!(std::panic::catch_unwind(|| require_psg(set)).is_err(),
                    "{set} must be refused: scoring against our own output is circular");
        }
        assert_eq!(reference_of("ours"), Reference::BandPeriod);
        assert!(std::panic::catch_unwind(|| reference_of("made-up")).is_err(),
                "an unknown set must be fatal, not assumed labelled");
    }
}
