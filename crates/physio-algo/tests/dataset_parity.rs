//! Parity gate: run the ported stagers on the on-disk PSG fixtures and reproduce the shipped
//! Cohen's-kappa within a tight tolerance, over the SAME subject count it was measured on (a partly
//! present corpus is a different cohort, not a passing gate). Every kappa in the GATE table is
//! FOUR-class (wake/light/deep/REM) and no three-class figure belongs in it. `three_class_report`
//! prints its own table, separately, because most wearable papers report Wake/NREM/REM and we could
//! not say where we stand without it — it merges the same staging, targets nothing, and gates
//! nothing. Health figures are wellness estimates, never medical or diagnostic.
//!
//! Each cohort gate carries two do-nothing arms it must reject: a CONSTANT hypnogram, and `stage_v2`
//! re-run over the same windows with every signal flattened, which is what the priors alone reach.
//! `stream_census_and_required_streams` pins which streams each cohort actually carries, because the
//! subject-count assert catches a missing DIRECTORY and never a missing or hollow STREAM inside one.
//!
//! `ours` is the only IN-COHORT reference the corpus holds and it marks the sleep PERIOD, not per-epoch
//! sleep, so it cannot adjudicate wake within a night and no arm may be selected on it (`Truth::TwoClass`).
//! DREAMT and AAUWSS are therefore the only four-class guard there is, out-of-cohort or not.
//!
//! `resp_weight` is the only V2 term that reads R-R, so a cohort whose `rr.csv` is empty scores V2
//! WITHOUT its respiratory channel — cardiac and motion only. `sleep-accel` is exactly that cohort and
//! its gate says so; the census asserts the split so a later graft cannot leave the name wrong.
//!
//! The stagers run `stage_v2` alone. No PSG cohort carries a step stream at one sample a minute, so
//! `refine_wake` would decline on every one of them; this is the unrefined path and it is the only path
//! these cohorts can score.
//!
//! `kappa_arithmetic_*` and `epoch_probe_*` run from a clean checkout on in-file matrices. The cohort
//! gates are `#[ignore]`d because the corpus is a multi-gigabyte derived tree outside the repo (DREAMT
//! is credentialed and not redistributable); point `WHOOP_SLEEP_FIXTURES` at a fixture root, or rely on
//! the default below, then run:
//!   cargo test --release -p physio-algo --test dataset_parity -- --ignored --nocapture

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use physio_algo::sleep::{
    stage_v2, AccelSample, HrSample, RrRun, SleepInput, SleepStage, StageSegment,
};

/// The de-duplicated, fidelity-repaired corpus, resolved off this crate's manifest dir so it works from
/// any checkout. The raw `fixtures_multi` root holds each beat twice on some own-strap nights and
/// `fixtures_multi_clean` still holds a second wearer-side duplication, so defaulting to either would
/// score a doubled R-R stream. `_clean3` differs from `_clean2` in exactly three ways, all measured
/// 2026-08-17: `aauwss/subject_12` no longer carries 36.3% placeholder (0,0,1) gravity, all 31
/// `sleep-accel` HR streams are no longer gap-filled into 50-95% flat runs, and DREAMT gained
/// `temp.csv` / `bvp_8hz.csv` / `events.csv` / `participant_info.csv`. Every DREAMT stream this test
/// reads is byte-identical between the two roots, and all three cohort gates pass unchanged.
const DEFAULT_ROOT: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../../sleep-benchmark/fixtures_multi_clean3");

fn fixtures_root() -> PathBuf {
    std::env::var("WHOOP_SLEEP_FIXTURES").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_ROOT))
}

/// The one tolerance all three cohort gates use.
const KAPPA_TOL: f64 = 0.008;

/// How far a cohort's shipped kappa must sit above the same cohort's prior-only staging. Just over
/// half the smallest margin measured 2026-08-02 (DREAMT 0.2822, sleep-accel 0.3257, AAUWSS 0.3745), so
/// it sits well clear of every cohort and still fails the moment a stager stops reading its inputs.
const NULL_MARGIN: f64 = 0.15;

/// One REQUIRED stream as rows of columns. An unreadable file is FATAL and names itself: returning empty
/// instead scores that night with the stream absent, and `assert_eq!(n, 100)` still counts it as present,
/// so a degraded cohort reaches a headline kappa with nothing saying so.
fn read_csv(path: &Path) -> Vec<Vec<f64>> {
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

struct Fixture {
    input: SleepInput,
    w0: i64,
    n_epochs: usize,
    truth: BTreeMap<usize, i32>,
}

fn load_fixture(dir: &Path) -> Fixture {
    let meta = fs::read_to_string(dir.join("meta.txt"))
        .unwrap_or_else(|e| panic!("fixture meta unreadable: {} ({e})", dir.display()));
    let m: Vec<i64> = meta.split_whitespace().map(|x| x.parse().unwrap()).collect();
    let (w0, w1, n_epochs) = (m[1], m[2], m[3] as usize);

    let accel = read_csv(&dir.join("gravity.csv"))
        .iter()
        .map(|r| AccelSample { ts: r[0] as i64, x: r[1], y: r[2], z: r[3] })
        .collect();
    let hr = read_csv(&dir.join("hr.csv"))
        .iter()
        .map(|r| HrSample { ts: r[0] as i64, bpm: r[1] as u16 })
        .collect();

    // Group consecutive same-timestamp R-R rows into one run (col 0 = ts, col 1 = rr_ms).
    let mut rr: Vec<RrRun> = Vec::new();
    for row in read_csv(&dir.join("rr.csv")) {
        let ts = row[0] as i64;
        let ms = row[1] as u16;
        match rr.last_mut() {
            Some(last) if last.ts == ts => last.intervals.push(ms),
            _ => rr.push(RrRun { ts, intervals: vec![ms] }),
        }
    }

    let mut truth = BTreeMap::new();
    for row in read_csv(&dir.join("truth.csv")) {
        truth.insert(row[0] as usize, row[1] as i32);
    }

    Fixture {
        input: SleepInput { start: w0, end: w1, hr, rr, accel },
        w0,
        n_epochs,
        truth,
    }
}

fn stage_to_int(s: SleepStage) -> i32 {
    match s {
        SleepStage::Wake => 0,
        SleepStage::Light => 1,
        SleepStage::Deep => 2,
        SleepStage::Rem => 3,
    }
}

/// Probe each labelled epoch's midpoint against the tiled segments (the harness rule).
fn predict_epochs(segs: &[StageSegment], w0: i64, n_epochs: usize) -> Vec<i32> {
    (0..n_epochs)
        .map(|k| {
            let mid = w0 + k as i64 * 30 + 15;
            let stage = segs
                .iter()
                .find(|s| s.start <= mid && mid < s.end)
                .map(|s| s.stage)
                .unwrap_or(segs.last().unwrap().stage);
            stage_to_int(stage)
        })
        .collect()
}

use physio_algo::sleep::metrics::{bootstrap_kappa_ci, kappa3, kappa4 as cohen_kappa, merge3};

/// One cohort scored, with the two do-nothing arms beside it and the R-R reach the "cardiorespiratory"
/// label depends on.
struct Scored {
    kappa: f64,
    subjects: usize,
    /// Nights dropped for carrying no labels. Counted, not silent: the three PSG gates require zero.
    unlabelled: usize,
    /// Nights whose `rr.csv` carried at least one beat. Zero means V2 ran without `resp_weight`.
    rr_nights: usize,
    /// `stage_v2` over the same windows with every signal flattened. What the priors alone reach.
    flat_kappa: f64,
    /// The best a single-stage hypnogram reaches on this cohort's labels.
    const_kappa: f64,
    /// The SAME staging read as Wake / NREM / REM, which is what most wearable papers report.
    kappa3: f64,
    /// Percentile bootstrap over RECORDINGS of the pooled four-class kappa.
    ci: Option<(f64, f64)>,
}

/// The same night with its signal removed: one HR value throughout, gravity pinned upright, no R-R.
/// The timestamps and window are untouched, so the stager still sees a full night and answers from its
/// priors alone. That answer is the floor a real staging must clear.
fn flatten(input: &SleepInput) -> SleepInput {
    let bpm = input.hr.first().map(|h| h.bpm).unwrap_or(60);
    SleepInput {
        start: input.start,
        end: input.end,
        hr: input.hr.iter().map(|h| HrSample { ts: h.ts, bpm }).collect(),
        rr: Vec::new(),
        accel: input.accel.iter().map(|a| AccelSample { ts: a.ts, x: 0.0, y: 0.0, z: 1.0 }).collect(),
    }
}

/// The kappa of the best single-stage hypnogram against these labels. Analytically zero for every
/// cohort (a constant scorer's expected agreement equals its observed one), so it is the cheapest proof
/// that a gate rejects a scorer doing no work at all.
fn constant_scorer_kappa(cm_truth: &[i64; 4]) -> f64 {
    (0..4)
        .map(|c| {
            let mut cm = [[0i64; 4]; 4];
            for (t, n) in cm_truth.iter().enumerate() {
                cm[t][c] = *n;
            }
            cohen_kappa(&cm)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Score one cohort. `nulls` buys the flattened-input arm, which doubles the staging work, so the
/// nine-set report leaves it off and only the gates pay for it.
fn score_dataset(ds: &str, nulls: bool) -> Scored {
    let root = fixtures_root().join(ds);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|_| panic!("dataset dir missing: {}", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    let (mut cm, mut cm_flat, mut truth_marginal) = ([[0i64; 4]; 4], [[0i64; 4]; 4], [0i64; 4]);
    let (mut subjects, mut unlabelled, mut rr_nights) = (0, 0, 0);
    // Per night, so the interval resamples RECORDINGS. Resampling epochs would call one long night
    // many independent observations.
    let mut per_night: Vec<[[i64; 4]; 4]> = Vec::new();
    for dir in &dirs {
        if !dir.join("meta.txt").exists() {
            continue;
        }
        let fx = load_fixture(dir);
        // A fixture with no labels cannot be scored. Counted rather than dropped in silence: on the
        // PSG cohorts a labelless night is a broken fixture, not a smaller corpus.
        if fx.truth.is_empty() {
            unlabelled += 1;
            continue;
        }
        if !fx.input.rr.is_empty() {
            rr_nights += 1;
        }
        let mut night = [[0i64; 4]; 4];
        let pred = predict_epochs(&stage_v2(&fx.input), fx.w0, fx.n_epochs);
        let flat = nulls.then(|| predict_epochs(&stage_v2(&flatten(&fx.input)), fx.w0, fx.n_epochs));
        for (k, &t) in &fx.truth {
            if *k < pred.len() && (0..4).contains(&t) {
                cm[t as usize][pred[*k] as usize] += 1;
                night[t as usize][pred[*k] as usize] += 1;
                truth_marginal[t as usize] += 1;
                if let Some(f) = &flat {
                    cm_flat[t as usize][f[*k] as usize] += 1;
                }
            }
        }
        subjects += 1;
        per_night.push(night);
    }
    Scored {
        kappa: cohen_kappa(&cm),
        kappa3: kappa3(&merge3(&cm)),
        // 2000 draws and a fixed seed, so a printed interval is reproducible run to run.
        ci: bootstrap_kappa_ci(&per_night, 2000, 0.05, 0x5EED),
        subjects,
        unlabelled,
        rr_nights,
        flat_kappa: if nulls { cohen_kappa(&cm_flat) } else { f64::NAN },
        const_kappa: constant_scorer_kappa(&truth_marginal),
    }
}

/// Everything a cohort gate asserts except its own target: the cohort is whole, every night is
/// labelled, and both do-nothing arms fall outside the band the shipped figure sits in.
fn assert_cohort(ds: &str, s: &Scored, target: f64, expect_n: usize) {
    println!(
        "V2 {ds}: 4-class kappa={:.4} over {} subjects (target {target}), {} nights carry R-R; \
         nulls: constant {:.4}, flat-input {:.4}",
        s.kappa, s.subjects, s.rr_nights, s.const_kappa, s.flat_kappa
    );
    assert_eq!(
        s.unlabelled, 0,
        "{ds}: {} fixture nights carry an empty truth.csv and were dropped — the cohort on disk is not \
         the cohort the target was measured over",
        s.unlabelled
    );
    assert_eq!(
        s.subjects, expect_n,
        "{ds}: scored {} subjects, not the {expect_n} the target was measured over",
        s.subjects
    );
    assert!(
        (s.kappa - target).abs() < KAPPA_TOL,
        "{ds}: 4-class kappa {:.4} off target {target} by {:.4}",
        s.kappa,
        (s.kappa - target).abs()
    );
    assert!(
        (s.const_kappa - target).abs() >= KAPPA_TOL,
        "{ds}: a CONSTANT hypnogram scores {:.4} and passes this gate — the gate proves nothing",
        s.const_kappa
    );
    assert!(
        s.kappa - s.flat_kappa >= NULL_MARGIN,
        "{ds}: staging signal-free input scores {:.4} against the shipped {:.4} — a margin of {:.4}, \
         under the {NULL_MARGIN} floor. The cohort's kappa is coming from the priors, not the streams",
        s.flat_kappa,
        s.kappa,
        s.kappa - s.flat_kappa
    );
}

#[test]
#[ignore = "the PSG corpus is a derived tree outside the repo (DREAMT is credentialed, not \
            redistributable); set WHOOP_SLEEP_FIXTURES and run with --ignored"]
fn v2_dreamt_4class_kappa_matches_shipped() {
    let s = score_dataset("dreamt", true);
    assert_eq!(s.rr_nights, 100, "DREAMT: {} of 100 nights carry R-R, not all 100", s.rr_nights);
    assert_cohort("dreamt", &s, 0.311, 100);
}

#[test]
#[ignore = "the PSG corpus is a derived tree outside the repo; set WHOOP_SLEEP_FIXTURES and run \
            with --ignored"]
fn v2_aauwss_4class_kappa_matches_shipped() {
    let s = score_dataset("aauwss", true);
    assert_eq!(s.rr_nights, 13, "AAUWSS: {} of 13 nights carry R-R, not all 13", s.rr_nights);
    assert_cohort("aauwss", &s, 0.412, 13);
}

/// The third PSG cohort, and the one with NO R-R at all: every `rr.csv` under it is zero bytes, so
/// `resp_weight` never fires and this figure is V2's cardiac-and-motion path, not its cardiorespiratory
/// one. The `rr_nights == 0` assert is what keeps that name true — graft R-R in and this fails first.
#[test]
#[ignore = "the PSG corpus is a derived tree outside the repo; set WHOOP_SLEEP_FIXTURES and run \
            with --ignored"]
fn v2_sleep_accel_4class_kappa_matches_shipped_without_the_respiratory_channel() {
    let s = score_dataset("sleep-accel", true);
    assert_eq!(
        s.rr_nights, 0,
        "sleep-accel: {} nights now carry R-R. This gate and its name say the cohort has none, so the \
         0.379 target is no longer a no-respiratory-channel figure — re-measure before touching it",
        s.rr_nights
    );
    assert_cohort("sleep-accel", &s, 0.379, 31);
}

/// What a set's truth column IS, and therefore whether a four-class kappa against it means anything.
enum Truth {
    /// wake / light / deep / REM. `score_dataset`'s 4x4 matrix is the right instrument.
    FourClass,
    /// wake / asleep only. A four-class kappa is a CATEGORY ERROR here: the `1` that means ASLEEP would
    /// be scored against `light`. Such a set is named and left unscored rather than given a wrong number.
    ///
    /// `ours` is the only member and is weaker than "two-class" suggests: its `1` means "inside the
    /// sleep period", not "not awake" — 42 runs of median 15,729 s over 29 nights, holding through
    /// 82.5% of our wake bouts. Recall is valid, precision is not, and nothing may be selected on it.
    TwoClass,
    /// No labels at all.
    Unlabelled,
}

/// Every set the corpus holds, with what its labels are worth, so none sits unnamed — `whoop4` was doing
/// exactly that, 20 nights no harness read, and `ours` was doing it again with 92. The three gates above
/// are what fail on a kappa drift; what THIS one asserts is that the list below still covers the disk,
/// because a set nobody has judged is how the last two got their wrong verdicts.
#[test]
#[ignore = "the PSG corpus is a derived tree outside the repo; set WHOOP_SLEEP_FIXTURES and run with \
            --ignored for the full sheet"]
fn v2_all_datasets_report() {
    const SETS: [(&str, Truth, &str); 9] = [
        ("dreamt", Truth::FourClass, "PSG hypnogram — accuracy"),
        ("aauwss", Truth::FourClass, "PSG hypnogram — accuracy"),
        ("sleep-accel", Truth::FourClass, "PSG hypnogram — accuracy"),
        ("killa5", Truth::FourClass, "our own stagesJSON — CIRCULAR, consistency only"),
        ("strap", Truth::FourClass, "our own stagesJSON — CIRCULAR, consistency only"),
        ("whoop4", Truth::FourClass, "our own on-board stagesJSON — CIRCULAR, consistency only"),
        ("ours", Truth::TwoClass, "the strap's own sleep_state — INDEPENDENT of us, but a SLEEP-PERIOD marker: recall only, precision is not computable"),
        ("continuous", Truth::Unlabelled, "unlabelled — its band reference rides in band.csv, not truth.csv"),
        ("e9night", Truth::Unlabelled, "unlabelled — nothing"),
    ];
    println!("fixture root: {}", fixtures_root().display());
    println!(
        "{:<14} {:>9} {:>9} {:>11} {:>9}   what its truth is",
        "dataset", "kappa4", "scored", "unlabelled", "with R-R"
    );
    for (ds, truth, what) in SETS {
        let root = fixtures_root().join(ds);
        if !root.is_dir() {
            println!("{ds:<14} {:>9} {:>9} {:>11} {:>9}   {what}", "-", "missing", "-", "-");
            continue;
        }
        match truth {
            // `unlabelled` is printed, not hidden: `strap` drops nights here and the subject column
            // alone made that look like a smaller cohort rather than a partly broken one.
            Truth::FourClass => match score_dataset(ds, false) {
                s if s.subjects == 0 => {
                    println!("{ds:<14} {:>9} {:>9} {:>11} {:>9}   {what}", "-", 0, s.unlabelled, s.rr_nights)
                }
                s => println!(
                    "{ds:<14} {:>9.3} {:>9} {:>11} {:>9}   {what}",
                    s.kappa, s.subjects, s.unlabelled, s.rr_nights
                ),
            },
            // Scored elsewhere, on the instrument its labels fit; a number here would be the wrong one.
            _ => println!("{ds:<14} {:>9} {:>9} {:>11} {:>9}   {what}", "-", "n/a", "-", "-"),
        }
    }

    let Ok(entries) = fs::read_dir(fixtures_root()) else { return };
    let mut unnamed: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| !SETS.iter().any(|(s, _, _)| s == n))
        .collect();
    unnamed.sort();
    assert!(unnamed.is_empty(), "fixture sets on disk that this matrix does not name: {unnamed:?}");
}

/// Every stream a set could carry, and what its nights are missing.
struct StreamStat {
    missing: usize,
    empty: usize,
    /// Window coverage of each night that carries the file non-empty.
    cov: Vec<f64>,
}

/// One night's scored window as `(start, seconds)`. The labelled sets write four numbers and the window
/// is their epoch grid; `continuous` writes owner and device first and carries no epochs, so its two
/// numbers ARE the window. A shape that is neither is skipped rather than measured against a guess.
fn census_window(dir: &Path) -> Option<(i64, i64)> {
    let meta = fs::read_to_string(dir.join("meta.txt")).ok()?;
    let m: Vec<i64> = meta.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    match m[..] {
        [_, w0, _, epochs, ..] => Some((w0, epochs * 30)),
        [w0, w1] if w1 > w0 => Some((w0, w1 - w0)),
        _ => None,
    }
}

/// The share of `[w0, w0+win)` a stream's samples cover. `truth.csv` counts in 30 s epoch INDICES, not
/// unix seconds, so its rows are mapped onto the same clock first or every set would read 0.
fn window_coverage(path: &Path, w0: i64, win: i64, epoch_indexed: bool) -> Option<(usize, f64)> {
    let text = fs::read_to_string(path).ok()?;
    let step = if epoch_indexed { 30 } else { 1 };
    let mut secs: Vec<i64> = text
        .lines()
        .filter_map(|l| l.split(',').next()?.trim().parse::<f64>().ok())
        .map(|v| if epoch_indexed { w0 + v as i64 * 30 } else { v as i64 })
        .filter(|t| (w0..w0 + win).contains(t))
        .collect();
    let rows = secs.len();
    secs.sort_unstable();
    secs.dedup();
    Some((rows, (secs.len() as i64 * step) as f64 / win as f64))
}

/// Per PSG cohort: `(set, nights, nights carrying R-R)`. `gravity`, `hr` and `truth` must be present
/// and non-empty on EVERY night of all three; R-R splits them, and that split is what decides whether a
/// cohort's kappa is a cardiorespiratory figure or a cardiac-and-motion one.
const PSG_REQUIRED: [(&str, usize, usize); 3] =
    [("dreamt", 100, 100), ("aauwss", 13, 13), ("sleep-accel", 31, 0)];

/// What each set's nights actually carry, per stream: how many are MISSING the file, how many carry it
/// EMPTY, and how much of the scored window the rest cover. The reason it exists: the subject-count
/// assert catches a missing DIRECTORY and never a missing or hollow STREAM inside a present one. The
/// three PSG cohorts are then ASSERTED against `PSG_REQUIRED`, so a hollowed stream fails here rather
/// than quietly re-defining what a cohort gate measures.
#[test]
#[ignore = "the PSG corpus is a derived tree outside the repo; set WHOOP_SLEEP_FIXTURES and run with \
            --ignored for the stream census"]
fn stream_census_and_required_streams() {
    const STREAMS: [&str; 6] =
        ["gravity.csv", "hr.csv", "rr.csv", "truth.csv", "band.csv", "steps.csv"];
    let root = fixtures_root();
    println!("stream census over {}", root.display());
    println!("cov = share of the scored window carrying a sample; the three counts are cumulative");
    println!(
        "{:<13} {:<13} {:>7} {:>8} {:>6} {:>8} {:>8} {:>8} {:>9}",
        "set", "stream", "nights", "missing", "empty", "cov<10%", "cov<50%", "cov<90%", "med cov"
    );

    let mut sets: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("fixture root unreadable: {} ({e})", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    sets.sort();
    // `(set, stream) -> (nights, missing, empty)`, so the printed sheet and the asserts below read the
    // same counts rather than two walks that could disagree.
    let mut census: BTreeMap<(String, &str), (usize, usize, usize)> = BTreeMap::new();

    for set in &sets {
        let name = set.file_name().unwrap_or_default().to_string_lossy().to_string();
        let mut dirs: Vec<PathBuf> = fs::read_dir(set)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() && p.join("meta.txt").exists())
            .collect();
        dirs.sort();
        if dirs.is_empty() {
            continue;
        }
        for stream in STREAMS {
            let mut st = StreamStat { missing: 0, empty: 0, cov: Vec::new() };
            for dir in &dirs {
                let Some((w0, win)) = census_window(dir) else { continue };
                match window_coverage(&dir.join(stream), w0, win, stream == "truth.csv") {
                    None => st.missing += 1,
                    Some((0, _)) => st.empty += 1,
                    Some((_, c)) => st.cov.push(c),
                }
            }
            print_census_row(&name, stream, dirs.len(), &mut st);
            census.insert((name.clone(), stream), (dirs.len(), st.missing, st.empty));
        }
    }

    for (set, nights, with_rr) in PSG_REQUIRED {
        for stream in ["gravity.csv", "hr.csv", "truth.csv"] {
            let got = census.get(&(set.to_string(), stream));
            assert_eq!(
                got,
                Some(&(nights, 0, 0)),
                "{set}/{stream}: expected {nights} nights, none missing, none empty — got {got:?}. A \
                 cohort gate scoring this set is measuring a different corpus"
            );
        }
        let rr = census.get(&(set.to_string(), "rr.csv"));
        assert_eq!(
            rr,
            Some(&(nights, 0, nights - with_rr)),
            "{set}/rr.csv: expected {with_rr} of {nights} nights carrying beats — got {rr:?}. R-R \
             reach decides whether this cohort's kappa is a cardiorespiratory figure"
        );
    }
}

// ── The scoring machinery, on matrices this file carries ───────────────────────────────────────
// These run from a clean checkout with no corpus. Everything above reads the fixture tree; if these
// were `#[ignore]`d too, a broken kappa or a broken epoch probe would reach a headline unchallenged.

/// Hand-checked four-class kappa: 32 of 40 on the diagonal against balanced marginals is
/// (0.80 - 0.25) / (1 - 0.25).
#[test]
fn kappa_arithmetic_reproduces_a_hand_computed_matrix() {
    let cm = [[8, 2, 0, 0], [2, 8, 0, 0], [0, 0, 8, 2], [0, 0, 2, 8]];
    assert!((cohen_kappa(&cm) - 0.55 / 0.75).abs() < 1e-12, "got {}", cohen_kappa(&cm));
    assert_eq!(cohen_kappa(&[[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]]), 1.0);
}

/// The two degenerate inputs that must NOT report agreement: nothing scored, and one class on both
/// axes (perfect on the diagonal, but chance already explains all of it).
#[test]
fn kappa_arithmetic_reports_zero_where_agreement_is_meaningless() {
    assert_eq!(cohen_kappa(&[[0; 4]; 4]), 0.0, "an empty matrix is not agreement");
    assert_eq!(cohen_kappa(&[[40, 0, 0, 0], [0; 4], [0; 4], [0; 4]]), 0.0, "one class both sides");
}

/// The failure arm the cohort gates rest on: a scorer that emits one stage for the whole night reaches
/// exactly 0.0 whatever the labels are, so none of the three targets can be met by doing no work.
#[test]
fn constant_scorer_scores_zero_and_fails_every_cohort_target() {
    for marginal in [[100, 300, 50, 150], [1, 1, 1, 1], [7, 0, 0, 3]] {
        let k = constant_scorer_kappa(&marginal);
        assert_eq!(k, 0.0, "constant scorer on {marginal:?}");
        for target in [0.311, 0.412, 0.379] {
            assert!((k - target).abs() >= KAPPA_TOL, "a constant scorer would pass the {target} gate");
        }
    }
}

/// Epoch k is scored at its midpoint `w0 + 30k + 15`, and an epoch past the last segment holds that
/// segment's stage rather than dropping out of the matrix.
#[test]
fn epoch_probe_reads_midpoints_and_holds_the_last_stage() {
    let segs = [
        StageSegment { start: 1_000, end: 1_060, stage: SleepStage::Deep },
        StageSegment { start: 1_060, end: 1_090, stage: SleepStage::Rem },
    ];
    assert_eq!(predict_epochs(&segs, 1_000, 5), vec![2, 2, 3, 3, 3]);
}

/// One census line. Sorts the coverages in place for the median, so it takes them by `&mut`.
fn print_census_row(set: &str, stream: &str, nights: usize, st: &mut StreamStat) {
    if st.missing == nights {
        let dash = "-";
        println!(
            "{set:<13} {stream:<13} {nights:>7} {nights:>8} {dash:>6} {dash:>8} {dash:>8} {dash:>8} {dash:>9}"
        );
        return;
    }
    st.cov.sort_by(f64::total_cmp);
    let under = |t: f64| st.cov.iter().filter(|c| **c < t).count() + st.empty;
    let (u10, u50, u90) = (under(0.1), under(0.5), under(0.9));
    let med = st.cov.get(st.cov.len() / 2).copied().unwrap_or(0.0);
    println!(
        "{set:<13} {stream:<13} {nights:>7} {:>8} {:>6} {u10:>8} {u50:>8} {u90:>8} {med:>9.3}",
        st.missing, st.empty
    );
}

/// The three PSG cohorts read as Wake / NREM / REM, with a bootstrap interval on the four-class
/// figure. Neither number gates anything; both describe the SAME staging the gate table scores.
///
/// The interval is the point of this: the gates are quoted to four decimals on corpora of 100, 13 and
/// 31 nights, and until now nothing said which of those digits the corpus can actually resolve.
#[test]
#[ignore = "needs the multi-GB fixture corpus"]
fn three_class_report() {
    println!("{:<14} {:>9} {:>9} {:>22} {:>7}", "dataset", "kappa4", "kappa3", "95% CI on kappa4", "n");
    for ds in ["dreamt", "aauwss", "sleep-accel"] {
        if !fixtures_root().join(ds).is_dir() {
            println!("{ds:<14} {:>9}", "missing");
            continue;
        }
        let s = score_dataset(ds, false);
        let ci = match s.ci {
            Some((lo, hi)) => format!("{lo:.4} .. {hi:.4}  (+/-{:.4})", (hi - lo) / 2.0),
            None => "-".to_string(),
        };
        println!("{ds:<14} {:>9.4} {:>9.4} {ci:>22} {:>7}", s.kappa, s.kappa3, s.subjects);
    }
    println!("\nkappa3 merges Light and Deep; it is the same staging, not a second one. The interval");
    println!("resamples RECORDINGS, so it answers 'how much would this move on another 100 nights'.");
}
