//! Step 6 — post-decode refinement and segmentation, measured as a computation rather than a recipe.
//!
//!   cargo run --release -p physio-algo --example step6_refine
//!
//! 1  which kappa scale is which, and whether either of them has ever been through the refinement
//! 2  the density gate's verdict over every set on disk, split by the stream that declined
//! 3  the refined subset: what the pass costs where it actually runs, paired
//! 4  is the gate the right computation — the two fractions' distribution, then a sweep over both
//! 5  segmentation: the tiling, the epochs it drops, and the grid the refinement rewrites on
//! 6  a short-bout filter as its own step, selected on the training cohorts and reported once
//!
//! Sections 1, 5 and 6 read the three PSG cohorts (four-class truth, no step stream). Sections 2, 3 and
//! 4 read the real-strap sets, which are the only ones the refinement can act on. The two are never
//! pooled: the refined subset has no four-class truth anywhere in the corpus.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use common::{
    dirs_of, fixtures_root, median, read_accel, read_band, read_hr, read_meta, read_rr, read_steps,
    read_truth, stage_at, stage_idx, RefineCensus, TwoClass, BAND_ASLEEP,
};
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    detect_sessions, epoch_starts_v2, motion_density, params::Params, prepare_v2, refine_wake,
    segments_v2, stage_v2_prepared, AccelSample, Prepared, SleepInput, SleepStage, StageSegment,
    StepSample, MIN_DENSE_FRACTION, STAGE_ORDER,
};

const EPOCH: i64 = 30;
const PSG: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
const CLASSES: usize = 4;
/// Fewest labelled epochs a night must carry before its own kappa is reported.
const MIN_SCORED: usize = 20;
/// The shipped per-minute sample floors the density gate applies, reproduced here so a sweep can move
/// them. `assert_gate_reproduced` pins this pair against the exported `motion_density`.
const SHIPPED_GRAV_PER_MIN: usize = 2;
const SHIPPED_STEP_PER_MIN: usize = 1;

// ── loading ───────────────────────────────────────────────────────────────────────────────────────

/// One staged night: the streams the refinement reads, the v2 hypnogram, and the epoch-grid labels a
/// four-class reference is scored against.
struct Night {
    name: String,
    w0: i64,
    w1: i64,
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    prep: Prepared,
    segs: Vec<StageSegment>,
    pred: Vec<usize>,
    truth: Vec<Option<usize>>,
}

/// Epoch `k` read at its midpoint, holding the last stage past the end — the probe every cohort gate uses.
fn predict_epochs(segs: &[StageSegment], w0: i64, n: usize) -> Vec<usize> {
    let last = segs.last().map(|s| s.stage).unwrap_or(SleepStage::Light);
    (0..n)
        .map(|k| {
            let mid = w0 + k as i64 * EPOCH + EPOCH / 2;
            stage_idx(stage_at(segs, mid).unwrap_or(last))
        })
        .collect()
}

fn load_set(set: &str) -> Vec<Night> {
    let mut out = Vec::new();
    for dir in dirs_of(set) {
        let Some((w0, w1, n_meta)) = read_meta(&dir) else { continue };
        let (accel, hr) = (read_accel(&dir), read_hr(&dir));
        if accel.len() < 120 || hr.len() < 120 || w1 <= w0 {
            continue;
        }
        let raw = read_truth(&dir);
        let n = n_meta.max(raw.keys().max().copied().unwrap_or(0) + 1);
        let input = SleepInput { start: w0, end: w1, hr, rr: read_rr(&dir), accel: accel.clone() };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let segs = stage_v2_prepared(&prep, &Params::SHIPPED);
        let pred = predict_epochs(&segs, w0, n);
        let truth = (0..n)
            .map(|k| raw.get(&k).copied().filter(|t| (0..CLASSES as i32).contains(t)).map(|t| t as usize))
            .collect();
        out.push(Night {
            name: dir.file_name().unwrap_or_default().to_string_lossy().to_string(),
            w0,
            w1,
            accel,
            steps: read_steps(&dir),
            prep,
            segs,
            pred,
            truth,
        });
    }
    out
}

/// One night's kappa over the epochs it labels, `None` when too few to mean anything.
fn night_kappa(pred: &[usize], truth: &[Option<usize>]) -> Option<f64> {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate().take(pred.len()) {
        if let Some(w) = want {
            p.push(pred[k]);
            t.push(*w);
        }
    }
    (t.len() >= MIN_SCORED).then(|| kappa4(&confusion4(&p, &t)))
}

/// The confusion matrix one night contributes to a POOLED cohort figure.
fn night_confusion(pred: &[usize], truth: &[Option<usize>]) -> [[i64; 4]; 4] {
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (k, want) in truth.iter().enumerate().take(pred.len()) {
        if let Some(w) = want {
            p.push(pred[k]);
            t.push(*w);
        }
    }
    confusion4(&p, &t)
}

fn add_cm(a: &mut [[i64; 4]; 4], b: &[[i64; 4]; 4]) {
    for (ra, rb) in a.iter_mut().zip(b) {
        for (x, y) in ra.iter_mut().zip(rb) {
            *x += y;
        }
    }
}

// ── 1  the two scales ─────────────────────────────────────────────────────────────────────────────

fn section_scales(cohorts: &[(&str, Vec<Night>)]) {
    println!("1  the two kappa scales, and which of them is the refined path");
    println!("   A cohort figure is a POOLED confusion matrix over every night; a harness figure is the");
    println!("   MEDIAN of the per-night kappas. They are different statistics on the same staging, and");
    println!("   neither is the refined path — the census beside each row is the gate's own answer.\n");
    println!(
        "   {:<14} {:>7} {:>10} {:>14} {:>12}   refinement census",
        "cohort", "n", "pooled k4", "per-night med", "per-night n"
    );
    for (name, nights) in cohorts {
        let mut cm = [[0i64; 4]; 4];
        let mut per: Vec<f64> = Vec::new();
        let mut census = RefineCensus::default();
        for nt in nights {
            add_cm(&mut cm, &night_confusion(&nt.pred, &nt.truth));
            if let Some(k) = night_kappa(&nt.pred, &nt.truth) {
                per.push(k);
            }
            census.refine(&nt.segs, &nt.accel, &nt.steps);
        }
        println!(
            "   {name:<14} {:>7} {:>10.4} {:>14.4} {:>12}   {} refined / {} declined",
            nights.len(),
            kappa4(&cm),
            median(&mut per.clone()),
            per.len(),
            census.refined,
            census.declined
        );
        println!("{}", census.line(name));
    }
}

// ── 2  the gate over every set ────────────────────────────────────────────────────────────────────

/// A night's scored window out of `meta.txt`, in both shapes the corpus writes.
fn window_of(dir: &Path) -> Option<(i64, i64)> {
    let text = fs::read_to_string(dir.join("meta.txt")).ok()?;
    let m: Vec<i64> = text.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    match m[..] {
        [_, w0, w1, _, ..] if w1 > w0 => Some((w0, w1)),
        [w0, w1] if w1 > w0 => Some((w0, w1)),
        _ => None,
    }
}

/// Per-set gate verdict plus WHY a stream declined: absent from the fixture, or present and thin.
fn section_gate_census() {
    println!("\n2  the density gate over every set on disk, split by the stream that declined");
    println!("   `no step file` is the fixture carrying no `steps.csv` at all; `thin` is one present and");
    println!("   under the fraction. The first is a corpus or a device fact, the second is a gate fact.\n");
    println!(
        "   {:<14} {:>7} {:>9} {:>10} {:>11} {:>10} {:>9} {:>10}",
        "set", "nights", "REFINED", "decl grav", "no step file", "step thin", "med grav", "med step"
    );
    let mut sets: Vec<String> = fs::read_dir(fixtures_root())
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    sets.sort();
    for set in &sets {
        let (mut n, mut refined, mut dg, mut no_file, mut thin) = (0, 0, 0, 0, 0);
        let (mut gs, mut ss) = (Vec::new(), Vec::new());
        for dir in dirs_of(set) {
            let Some((w0, w1)) = window_of(&dir) else { continue };
            let accel = read_accel(&dir);
            if accel.len() < 120 {
                continue;
            }
            n += 1;
            let steps = read_steps(&dir);
            let (g, s) = motion_density(w0, w1, &accel, &steps);
            gs.push(g);
            ss.push(s);
            if g >= MIN_DENSE_FRACTION && s >= MIN_DENSE_FRACTION {
                refined += 1;
                continue;
            }
            dg += usize::from(g < MIN_DENSE_FRACTION);
            if s < MIN_DENSE_FRACTION {
                if steps.is_empty() {
                    no_file += 1;
                } else {
                    thin += 1;
                }
            }
        }
        if n == 0 {
            continue;
        }
        println!(
            "   {set:<14} {n:>7} {refined:>9} {dg:>10} {no_file:>11} {thin:>10} {:>10.3} {:>10.3}",
            median(&mut gs),
            median(&mut ss)
        );
    }
}

/// Median samples in the minutes that carry ANY. A stream at 1 Hz inside a fifth of the night's minutes
/// and absent in the rest is duty-cycled, not thin, and no coverage fraction separates those two.
fn samples_per_covered_minute(ts: &[i64]) -> f64 {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for t in ts {
        *counts.entry(t / 60).or_insert(0) += 1;
    }
    let mut v: Vec<f64> = counts.values().map(|c| *c as f64).collect();
    median(&mut v)
}

/// Which straps carry a step stream at all. A device that never emits one declines on every night it
/// ever records, which is a hardware fact and not a threshold the gate could be argued down to.
fn section_step_by_device() {
    println!("\n   the same decline by DEVICE, over the two real-strap sets:");
    println!(
        "   {:<16} {:>8} {:>13} {:>12} {:>22}",
        "device", "nights", "carry steps", "med frac", "med samples/cov minute"
    );
    let mut by: BTreeMap<String, (usize, usize, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for set in ["ours", "continuous"] {
        for dir in dirs_of(set) {
            let Some((w0, w1)) = window_of(&dir) else { continue };
            let accel = read_accel(&dir);
            if accel.len() < 120 {
                continue;
            }
            let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
            let dev = name.split('_').nth(1).unwrap_or("?").to_string();
            let steps = read_steps(&dir);
            let (_, s) = motion_density(w0, w1, &accel, &steps);
            let e = by.entry(dev).or_insert((0, 0, Vec::new(), Vec::new()));
            e.0 += 1;
            e.1 += usize::from(!steps.is_empty());
            e.2.push(s);
            if !steps.is_empty() {
                e.3.push(samples_per_covered_minute(&steps.iter().map(|t| t.ts).collect::<Vec<_>>()));
            }
        }
    }
    for (dev, (n, carry, mut fracs, mut dens)) in by {
        let d = if dens.is_empty() { f64::NAN } else { median(&mut dens) };
        println!("   {dev:<16} {n:>8} {carry:>13} {:>12.3} {d:>22.0}", median(&mut fracs));
    }
    println!("   A device at ~1 sample a second inside a FIFTH of the night's minutes is duty-cycling the");
    println!("   stream. The gate reads that as sparse, which is a capture schedule and not a signal fact.");
}

// ── 3  the refined subset ─────────────────────────────────────────────────────────────────────────

/// One detected span of a `continuous` block, with the band at 1 Hz over it.
struct Span {
    accel: Vec<AccelSample>,
    steps: Vec<StepSample>,
    band: Vec<(i64, i32)>,
    segs: Vec<StageSegment>,
}

fn continuous_spans() -> Vec<Span> {
    let mut out = Vec::new();
    for d in dirs_of("continuous") {
        let band = read_band(&d);
        let accel = read_accel(&d);
        if band.is_empty() || accel.len() < 120 {
            continue;
        }
        let (hr, rr, steps) = (read_hr(&d), read_rr(&d), read_steps(&d));
        for s in detect_sessions(&hr, &accel, 0, &[], &band, None) {
            let input = SleepInput {
                start: s.start,
                end: s.end,
                hr: hr.iter().filter(|h| h.ts >= s.start && h.ts < s.end).cloned().collect(),
                rr: rr.iter().filter(|r| r.ts >= s.start && r.ts < s.end).cloned().collect(),
                accel: accel.iter().filter(|g| g.ts >= s.start && g.ts < s.end).cloned().collect(),
            };
            if input.hr.len() < 120 || input.accel.len() < 120 {
                continue;
            }
            let segs = stage_v2_prepared(&prepare_v2(&input, &Params::SHIPPED), &Params::SHIPPED);
            out.push(Span {
                steps: steps.iter().filter(|t| t.ts >= s.start && t.ts < s.end).cloned().collect(),
                band: band.iter().filter(|(t, _)| *t >= s.start && *t < s.end).cloned().collect(),
                accel: input.accel,
                segs,
            });
        }
    }
    out
}

/// Two-class kappa and wake recall of one span against the band, paired before and after the pass.
fn section_refined_subset(spans: &[Span], ours: &[Night]) {
    println!("\n3  the refined subset: what the pass costs where it actually runs");
    println!("   The pass only ever SHRINKS wake, so every second it moves is either a false wake removed");
    println!("   or a true one lost, and only a per-second reference can tell which. Paired per span.\n");
    let mut census = RefineCensus::default();
    let (mut dk, mut dr) = (Vec::new(), Vec::new());
    // Our wake seconds split by what the pass did with them, counting the band's TRUSTWORTHY call.
    // Asleep is silent on this marker, so only the awake column adjudicates.
    let (mut flip, mut kept) = ((0i64, 0i64), (0i64, 0i64));
    for s in spans {
        let before = census.refined;
        let out = census.refine(&s.segs, &s.accel, &s.steps);
        if census.refined == before {
            continue;
        }
        let (mut a, mut b) = (TwoClass::default(), TwoClass::default());
        for &(ts, code) in &s.band {
            let (Some(x), Some(y)) = (stage_at(&s.segs, ts), stage_at(&out, ts)) else { continue };
            let awake = code != BAND_ASLEEP;
            a.add(x == SleepStage::Wake, awake);
            b.add(y == SleepStage::Wake, awake);
            if x == SleepStage::Wake {
                let cell = if y == SleepStage::Wake { &mut kept } else { &mut flip };
                cell.0 += i64::from(awake);
                cell.1 += 1;
            }
        }
        if a.n >= 600 && a.true_wake > 0 {
            dk.push(b.kappa() - a.kappa());
            dr.push(b.recall() - a.recall());
        }
    }
    println!("{}", census.line("continuous spans, band-scored"));
    let bar = |v: &[f64]| match paired_bar(v) {
        Some((m, b)) if m.abs() > b => format!("{m:+.4} +/- {b:.4}  RESOLVES"),
        Some((m, b)) => format!("{m:+.4} +/- {b:.4}  noise"),
        None => "too few pairs".to_string(),
    };
    println!("   paired per-span delta over the {} refined spans that carry band wake:", dk.len());
    println!("     band kappa2   {}", bar(&dk));
    println!("     wake recall   {} (percentage points)", bar(&dr));
    let pct = |c: (i64, i64)| 100.0 * c.0 as f64 / c.1.max(1) as f64;
    println!("\n   the control the flip count alone cannot give: our wake seconds, split by what the pass");
    println!("   did with them, against the band's AWAKE call — the only direction this marker can judge.");
    println!("   {:<28} {:>12} {:>18}", "our wake seconds", "seconds", "band says AWAKE");
    println!("   {:<28} {:>12} {:>17.1}%", "FLIPPED to light", flip.1, pct(flip));
    println!("   {:<28} {:>12} {:>17.1}%", "KEPT as wake", kept.1, pct(kept));
    println!(
        "   {:<28} {:>12} {:>17.1}%",
        "both (unrefined wake)",
        flip.1 + kept.1,
        pct((flip.0 + kept.0, flip.1 + kept.1))
    );

    // The other real-strap reference: the strap's own sleep-PERIOD marker. It cannot adjudicate wake
    // inside the period, so only recall is read here and nothing may be selected on it.
    let mut c2 = RefineCensus::default();
    let (mut n_scored, mut lost) = (0usize, 0i64);
    let mut drec: Vec<f64> = Vec::new();
    for nt in ours {
        let before = c2.refined;
        let out = c2.refine(&nt.segs, &nt.accel, &nt.steps);
        if c2.refined == before || nt.truth.iter().all(|t| t.is_none()) {
            continue;
        }
        let after = predict_epochs(&out, nt.w0, nt.pred.len());
        let (mut a, mut b) = (TwoClass::default(), TwoClass::default());
        for (k, want) in nt.truth.iter().enumerate().take(nt.pred.len()) {
            let Some(w) = want else { continue };
            // `ours` truth is a period marker: 1 means inside the sleep period, 0 outside it.
            let awake = *w == 0;
            a.add(nt.pred[k] == 0, awake);
            b.add(after[k] == 0, awake);
            lost += i64::from(nt.pred[k] == 0 && after[k] != 0);
        }
        if a.true_wake > 0 {
            drec.push(b.recall() - a.recall());
            n_scored += 1;
        }
    }
    println!("\n{}", c2.line("`ours` nights, period-marker scored"));
    println!(
        "   {n_scored} refined nights carry marked out-of-period epochs; paired wake-recall delta {} \
         ({} epochs de-waked)",
        bar(&drec),
        lost
    );
    println!("   Recall is the only valid read against a period marker, and the pass can only lower it,");
    println!("   so this arm can show a cost and can never show a benefit. It is a bound, not a verdict.");
}

// ── 4  is the gate the right computation ──────────────────────────────────────────────────────────

/// Fraction of the wall-clock minutes tiling `[start, end)` carrying at least `min_per_minute` samples.
/// A harness copy of the shipped counter so the sweep can move its two floors; pinned against
/// `motion_density` at the shipped pair before any row is printed.
fn dense_fraction(ts: &[i64], start: i64, end: i64, min_per_minute: usize) -> f64 {
    if end <= start || min_per_minute == 0 {
        return if end <= start { 0.0 } else { 1.0 };
    }
    let (first, last) = (start / 60, (end - 1) / 60);
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for t in ts {
        let m = t / 60;
        if (first..=last).contains(&m) {
            *counts.entry(m).or_insert(0) += 1;
        }
    }
    let total = last - first + 1;
    let dense = (first..=last).filter(|m| counts.get(m).copied().unwrap_or(0) >= min_per_minute).count();
    dense as f64 / total as f64
}

/// The reimplementation above must equal the shipped gate at the shipped floors, or every sweep row
/// below is measuring a different computation from the one that ships.
fn assert_gate_reproduced(w0: i64, w1: i64, accel: &[AccelSample], steps: &[StepSample]) {
    let (g, s) = motion_density(w0, w1, accel, steps);
    let gt: Vec<i64> = accel.iter().map(|a| a.ts).collect();
    let st: Vec<i64> = steps.iter().map(|a| a.ts).collect();
    let mine = (
        dense_fraction(&gt, w0, w1, SHIPPED_GRAV_PER_MIN),
        dense_fraction(&st, w0, w1, SHIPPED_STEP_PER_MIN),
    );
    assert!(
        (g - mine.0).abs() < 1e-12 && (s - mine.1).abs() < 1e-12,
        "the harness gate copy does not reproduce motion_density: {g:?}/{s:?} against {mine:?}"
    );
}

struct Frac {
    grav: Vec<f64>,
    step: Vec<f64>,
}

/// Both fractions over every real-strap night, at one pair of per-minute floors.
fn real_fractions(gmin: usize, smin: usize) -> Frac {
    let (mut grav, mut step) = (Vec::new(), Vec::new());
    for set in ["ours", "continuous"] {
        for dir in dirs_of(set) {
            let Some((w0, w1)) = window_of(&dir) else { continue };
            let accel = read_accel(&dir);
            if accel.len() < 120 {
                continue;
            }
            let steps = read_steps(&dir);
            if gmin == SHIPPED_GRAV_PER_MIN && smin == SHIPPED_STEP_PER_MIN {
                assert_gate_reproduced(w0, w1, &accel, &steps);
            }
            let gt: Vec<i64> = accel.iter().map(|a| a.ts).collect();
            let st: Vec<i64> = steps.iter().map(|a| a.ts).collect();
            grav.push(dense_fraction(&gt, w0, w1, gmin));
            step.push(dense_fraction(&st, w0, w1, smin));
        }
    }
    Frac { grav, step }
}

fn deciles(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (0..=10)
        .map(|i| {
            let k = ((s.len() - 1) * i) / 10;
            format!("{:.2}", s[k])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn section_gate_sweep() {
    println!("\n4  is the density gate the right computation");
    let base = real_fractions(SHIPPED_GRAV_PER_MIN, SHIPPED_STEP_PER_MIN);
    println!("   the harness copy of the gate reproduces `motion_density` on every night (asserted)\n");
    println!("   the two fractions over {} real-strap nights, as deciles p0..p100:", base.grav.len());
    println!("     gravity  {}", deciles(&base.grav));
    println!("     steps    {}", deciles(&base.step));
    let near = |v: &[f64], lo: f64, hi: f64| v.iter().filter(|x| **x > lo && **x < hi).count();
    println!(
        "   nights whose step fraction is strictly between 0.05 and 0.95: {} of {} — the quantity the",
        near(&base.step, 0.05, 0.95),
        base.step.len()
    );
    println!("   threshold sorts is almost two-valued, so WHERE the threshold sits barely sorts anything.");

    println!("\n   pass counts under both floors and a range of coverage fractions:");
    print!("   {:<24}", "floors (grav/min, step/min)");
    for t in [0.0f64, 0.20, 0.50, 0.70, 0.80, 0.90, 1.00] {
        print!(" {t:>7.2}");
    }
    println!();
    for (gmin, smin, tag) in [
        (SHIPPED_GRAV_PER_MIN, SHIPPED_STEP_PER_MIN, "2 / 1  (SHIPPED)"),
        (1, 1, "1 / 1"),
        (4, 1, "4 / 1"),
        (2, 2, "2 / 2"),
        (2, 0, "2 / none  (steps dropped)"),
    ] {
        let f = real_fractions(gmin, smin);
        print!("   {tag:<24}");
        for t in [0.0f64, 0.20, 0.50, 0.70, 0.80, 0.90, 1.00] {
            let pass = f
                .grav
                .iter()
                .zip(&f.step)
                .filter(|(g, s)| **g >= t && **s >= t)
                .count();
            print!(" {pass:>7}");
        }
        println!();
    }
    println!("   The step column is what moves the count: dropping the step requirement admits every");
    println!("   night whose gravity is dense, and no coverage fraction between 0.2 and 1.0 changes that.");
}

/// Whether the corpus can adjudicate a relaxed gate at all, and how much a relaxed gate would move.
fn section_gate_relaxed(ours: &[Night]) {
    println!("\n   who could judge a relaxed gate: real-strap nights by gate verdict and reference");
    println!("   {:<26} {:>10} {:>18}", "gate verdict", "nights", "carry a band ref");
    let mut cells = [(0usize, 0usize); 2];
    for set in ["ours", "continuous"] {
        for dir in dirs_of(set) {
            let Some((w0, w1)) = window_of(&dir) else { continue };
            let accel = read_accel(&dir);
            if accel.len() < 120 {
                continue;
            }
            let steps = read_steps(&dir);
            let (g, s) = motion_density(w0, w1, &accel, &steps);
            let i = usize::from(g >= MIN_DENSE_FRACTION && s >= MIN_DENSE_FRACTION);
            cells[i].0 += 1;
            cells[i].1 += usize::from(!read_band(&dir).is_empty());
        }
    }
    println!("   {:<26} {:>10} {:>18}", "REFINED", cells[1].0, cells[1].1);
    println!("   {:<26} {:>10} {:>18}", "declined", cells[0].0, cells[0].1);
    println!("   Every night carrying a per-second reference already passes. The nights the gate turns");
    println!("   away are exactly the ones with nothing to check the decision against.");

    println!("\n   how much a gravity-only gate would move, on the `ours` nights declining on STEPS alone.");
    println!("   The stand-in is a synthetic 1/min step stream of class 0 — present and not walking — so");
    println!("   this is the pass's UPPER bound there: a real stream could only hold more of it back.");
    let (mut n, mut changed) = (0usize, 0usize);
    let mut shares: Vec<f64> = Vec::new();
    for nt in ours {
        let (g, s) = motion_density(nt.w0, nt.w1, &nt.accel, &nt.steps);
        if g < MIN_DENSE_FRACTION || s >= MIN_DENSE_FRACTION {
            continue;
        }
        n += 1;
        assert_eq!(
            refine_wake(&nt.segs, &nt.accel, &nt.steps),
            nt.segs,
            "the shipped gate must decline this night, or it is not in this arm"
        );
        let synth: Vec<StepSample> = (nt.w0 / 60..=(nt.w1 - 1) / 60)
            .map(|m| StepSample { ts: m * 60, counter: 0, activity_class: Some(0) })
            .collect();
        let out = refine_wake(&nt.segs, &nt.accel, &synth);
        changed += usize::from(out != nt.segs);
        let wake = |v: &[StageSegment]| -> i64 {
            v.iter().filter(|g| g.stage == SleepStage::Wake).map(|g| g.end - g.start).sum()
        };
        shares.push(100.0 * (wake(&nt.segs) - wake(&out)) as f64 / (nt.w1 - nt.w0).max(1) as f64);
    }
    println!(
        "   {n} nights decline on steps alone; the pass would change {changed} of them, removing a median \
         {:.2}% of the window's seconds from wake",
        median(&mut shares)
    );

    // The gate is scoped to the WHOLE window; both things it guards are read per SEGMENT. So ask the
    // same question at the consumer's scope: are the minutes of the candidate wake runs dense?
    println!("\n   the gate's scope against its consumers': the locomotion veto and the posture test both");
    println!("   read only the minutes of the wake run being converted, but the gate judges the whole");
    println!("   window. The same two fractions taken over the candidate runs alone:");
    let (mut nights, mut with_eligible, mut seg_dense_nights, mut segs_seen, mut segs_dense) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for nt in ours {
        let (g, s) = motion_density(nt.w0, nt.w1, &nt.accel, &nt.steps);
        if g >= MIN_DENSE_FRACTION && s >= MIN_DENSE_FRACTION {
            continue;
        }
        nights += 1;
        let last = nt.segs.len().saturating_sub(1);
        let mut any_eligible = false;
        let mut any_dense = false;
        for (i, seg) in nt.segs.iter().enumerate() {
            if i == 0 || i == last || seg.stage != SleepStage::Wake || seg.end - seg.start < 300 {
                continue;
            }
            any_eligible = true;
            segs_seen += 1;
            let (sg, ss) = motion_density(seg.start, seg.end, &nt.accel, &nt.steps);
            if sg >= MIN_DENSE_FRACTION && ss >= MIN_DENSE_FRACTION {
                segs_dense += 1;
                any_dense = true;
            }
        }
        with_eligible += usize::from(any_eligible);
        seg_dense_nights += usize::from(any_dense);
    }
    println!(
        "   {nights} declined `ours` nights hold {segs_seen} eligible wake runs on {with_eligible} of them; \
         {segs_dense} of those runs pass the same gate at their own scope, on {seg_dense_nights} nights"
    );
}

// ── 5  segmentation ───────────────────────────────────────────────────────────────────────────────

/// The stage covering most of `[a, b)` and the share it covers.
fn majority_stage(segs: &[StageSegment], a: i64, b: i64) -> Option<(SleepStage, f64)> {
    let mut acc = [0i64; CLASSES];
    for g in segs {
        let (lo, hi) = (g.start.max(a), g.end.min(b));
        if hi > lo {
            acc[stage_idx(g.stage)] += hi - lo;
        }
    }
    let total: i64 = acc.iter().sum();
    if total == 0 {
        return None;
    }
    let best = (0..CLASSES).max_by_key(|c| acc[*c]).expect("four classes");
    let stage = STAGE_ORDER.iter().copied().find(|s| stage_idx(*s) == best).expect("stage");
    Some((stage, acc[best] as f64 / (b - a) as f64))
}

fn section_segmentation(cohorts: &[(&str, Vec<Night>)], ours: &[Night]) {
    println!("\n5  segmentation: the tiling, the epochs it drops, and the grid the pass rewrites on");

    // (a) The round trip. `segments_of` tiles labels and the midpoint probe reads them back; if that is
    // not the identity, every cohort figure is measuring the tiling and not the decoder.
    let mut broken = 0usize;
    for (_, nights) in cohorts {
        for nt in nights {
            let starts = epoch_starts_v2(&nt.prep);
            let labels: Vec<SleepStage> = starts
                .iter()
                .map(|t| stage_at(&nt.segs, t + EPOCH / 2).unwrap_or(SleepStage::Light))
                .collect();
            let re = segments_v2(&nt.prep, &labels);
            broken += usize::from(re != nt.segs);
        }
    }
    println!("   round trip labels -> segments_of -> midpoint probe: {broken} of {} nights differ",
             cohorts.iter().map(|(_, n)| n.len()).sum::<usize>());

    // (b) Epochs the feature extractor drops. Their wall-clock time is absorbed into the PREVIOUS
    // epoch's segment, so a dropped epoch silently inherits the label before it.
    let (mut nights_with_gaps, mut dropped, mut absorbed) = (0usize, 0i64, 0i64);
    let mut total_epochs = 0i64;
    for (_, nights) in cohorts {
        for nt in nights {
            let starts = epoch_starts_v2(&nt.prep);
            total_epochs += starts.len() as i64;
            let mut here = 0i64;
            for w in starts.windows(2) {
                let gap = w[1] - w[0];
                if gap > EPOCH {
                    here += gap / EPOCH - 1;
                    absorbed += gap - EPOCH;
                }
            }
            dropped += here;
            nights_with_gaps += usize::from(here > 0);
        }
    }
    println!(
        "   epochs dropped before staging: {dropped} of {total_epochs} on {nights_with_gaps} nights, \
         {absorbed} s absorbed into the preceding epoch's segment"
    );

    // (c) The refinement rewrites on a WALL-CLOCK MINUTE grid; the hypnogram is read on a 30 s epoch
    // grid anchored at the window start. Where the two are out of phase an epoch is split.
    let mut census = RefineCensus::default();
    let (mut off_grid, mut boundaries, mut split_epochs, mut flipped_by_minority) = (0usize, 0usize, 0i64, 0i64);
    let mut phase = [0usize; 2];
    for nt in ours {
        let before = census.refined;
        let out = census.refine(&nt.segs, &nt.accel, &nt.steps);
        if census.refined == before {
            continue;
        }
        phase[usize::from(nt.w0.rem_euclid(60) == 0)] += 1;
        for g in out.iter().skip(1) {
            boundaries += 1;
            off_grid += usize::from((g.start - nt.w0).rem_euclid(EPOCH) != 0);
        }
        let n = ((nt.w1 - nt.w0) / EPOCH).max(0);
        for k in 0..n {
            let (a, b) = (nt.w0 + k * EPOCH, nt.w0 + (k + 1) * EPOCH);
            let (Some((maj, share)), Some(mid)) = (majority_stage(&out, a, b), stage_at(&out, a + EPOCH / 2))
            else {
                continue;
            };
            if share < 1.0 {
                split_epochs += 1;
            }
            flipped_by_minority += i64::from(mid != maj);
        }
    }
    println!(
        "\n   on the {} `ours` nights the pass runs on: {} of {} window starts sit on a minute boundary",
        census.refined, phase[1], census.refined
    );
    println!(
        "   segment boundaries the pass leaves OFF the 30 s epoch grid: {off_grid} of {boundaries}"
    );
    println!(
        "   epochs split across two stages after the pass: {split_epochs}; epochs the midpoint probe \
         reads as the MINORITY stage: {flipped_by_minority}"
    );

    // (d) The edge guard only protects a window whose first or last segment IS wake. Detection starts
    // the session at sleep, so the run it was written for may not be there to protect.
    println!("\n   `skip_window_edges` guards the first and last SEGMENT, which is a wake RUN only when");
    println!("   the window opens or closes awake:");
    println!("   {:<26} {:>8} {:>14} {:>14}", "windows", "n", "opens on wake", "closes on wake");
    let count = |v: &[Night]| -> (usize, usize, usize) {
        let (mut lead, mut trail, mut n) = (0, 0, 0);
        for nt in v {
            let (Some(f), Some(l)) = (nt.segs.first(), nt.segs.last()) else { continue };
            n += 1;
            lead += usize::from(f.stage == SleepStage::Wake);
            trail += usize::from(l.stage == SleepStage::Wake);
        }
        (n, lead, trail)
    };
    for (name, nights) in cohorts {
        let (n, lead, trail) = count(nights);
        println!("   {:<26} {n:>8} {lead:>14} {trail:>14}", format!("{name} (fixture window)"));
    }
    let refined: Vec<&Night> = ours
        .iter()
        .filter(|nt| {
            let (g, s) = motion_density(nt.w0, nt.w1, &nt.accel, &nt.steps);
            g >= MIN_DENSE_FRACTION && s >= MIN_DENSE_FRACTION
        })
        .collect();
    let (mut rn, mut rlead, mut rtrail) = (0usize, 0usize, 0usize);
    for nt in &refined {
        let (Some(f), Some(l)) = (nt.segs.first(), nt.segs.last()) else { continue };
        rn += 1;
        rlead += usize::from(f.stage == SleepStage::Wake);
        rtrail += usize::from(l.stage == SleepStage::Wake);
    }
    println!("   {:<26} {rn:>8} {rlead:>14} {rtrail:>14}", "ours, gate-accepted");
}

// ── 6  a short-bout filter ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Scope {
    All,
    Wake,
    Deep,
    Rem,
    DeepRem,
}

#[derive(Clone, Copy, PartialEq)]
enum Absorb {
    Longer,
    Prev,
    Light,
}

/// Absorb every run shorter than `min_len` into a neighbour, shortest first. `Absorb::Light` is the
/// null arm: it ignores the neighbours, so a win there is a base-rate shift and not smoothing.
#[derive(Clone, Copy)]
struct Dwell {
    min_len: usize,
    scope: Scope,
    absorb: Absorb,
}

impl Dwell {
    fn eligible(&self, label: usize) -> bool {
        match self.scope {
            Scope::All => true,
            Scope::Wake => label == 0,
            Scope::Deep => label == 2,
            Scope::Rem => label == 3,
            Scope::DeepRem => label == 2 || label == 3,
        }
    }
    fn name(&self) -> String {
        let scope = match self.scope {
            Scope::All => "all",
            Scope::Wake => "wake",
            Scope::Deep => "deep",
            Scope::Rem => "rem",
            Scope::DeepRem => "deep+rem",
        };
        let absorb = match self.absorb {
            Absorb::Longer => "longer",
            Absorb::Prev => "prev",
            Absorb::Light => "->light",
        };
        format!("L>={} {scope}/{absorb}", self.min_len)
    }
}

/// Maximal runs as `(start, len)`.
fn runs(seq: &[usize]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < seq.len() {
        let start = i;
        while i < seq.len() && seq[i] == seq[start] {
            i += 1;
        }
        out.push((start, i - start));
    }
    out
}

fn apply_dwell(seq: &[usize], d: Dwell) -> Vec<usize> {
    let mut out = seq.to_vec();
    if d.min_len < 2 {
        return out;
    }
    // Each absorption merges a run into a neighbour, so the run count strictly falls and this bound
    // can never be reached without the break below firing first.
    for _ in 0..seq.len() {
        let r = runs(&out);
        if r.len() < 2 {
            break;
        }
        let mut cand: Vec<usize> = (0..r.len())
            .filter(|&i| r[i].1 < d.min_len && d.eligible(out[r[i].0]))
            .collect();
        cand.sort_by_key(|&i| r[i].1);
        let mut done = false;
        for i in cand {
            let old = out[r[i].0];
            let new = match d.absorb {
                Absorb::Light => 1,
                Absorb::Prev => {
                    if i > 0 {
                        out[r[i - 1].0]
                    } else {
                        out[r[i + 1].0]
                    }
                }
                Absorb::Longer => {
                    let prev = (i > 0).then(|| (r[i - 1].1, out[r[i - 1].0]));
                    let next = (i + 1 < r.len()).then(|| (r[i + 1].1, out[r[i + 1].0]));
                    match (prev, next) {
                        (Some(a), Some(b)) => {
                            if b.0 > a.0 {
                                b.1
                            } else {
                                a.1
                            }
                        }
                        (Some(a), None) => a.1,
                        (None, Some(b)) => b.1,
                        (None, None) => old,
                    }
                }
            };
            if new == old {
                continue;
            }
            for slot in out.iter_mut().skip(r[i].0).take(r[i].1) {
                *slot = new;
            }
            done = true;
            break;
        }
        if !done {
            break;
        }
    }
    out
}

/// The grid. `Absorb::Light` is the neighbour-blind null and rides on every scope, so a scope whose win
/// survives it is a base-rate shift rather than a smoothing one.
/// The filter on hand-made sequences, so a table of deltas is not the first thing that reads it. A
/// pass-through arm is included: with nothing under the floor the sequence must come back untouched.
fn assert_dwell() {
    let longer = |min_len, scope| Dwell { min_len, scope, absorb: Absorb::Longer };
    // A lone deep epoch between two light runs becomes light; the two runs then merge.
    assert_eq!(apply_dwell(&[1, 1, 1, 2, 1, 1, 1], longer(2, Scope::All)), vec![1; 7]);
    // Scope keeps it: the same run is left alone when only wake is eligible.
    assert_eq!(
        apply_dwell(&[1, 1, 1, 2, 1, 1, 1], longer(2, Scope::Wake)),
        vec![1, 1, 1, 2, 1, 1, 1]
    );
    // The longer neighbour wins, not the preceding one — the two rules disagree here and must.
    let split = [0usize, 0, 0, 3, 1, 1, 1, 1, 1];
    assert_eq!(apply_dwell(&split, longer(2, Scope::All)), vec![0, 0, 0, 1, 1, 1, 1, 1, 1]);
    assert_eq!(
        apply_dwell(&split, Dwell { min_len: 2, scope: Scope::All, absorb: Absorb::Prev }),
        vec![0, 0, 0, 0, 1, 1, 1, 1, 1]
    );
    // Nothing under the floor: byte-identical passthrough.
    let long = [1usize, 1, 1, 1, 2, 2, 2, 2];
    assert_eq!(apply_dwell(&long, longer(4, Scope::All)), long.to_vec());
    // `->light` ignores the neighbours: a short wake run between two REM runs goes to light.
    assert_eq!(
        apply_dwell(&[3, 3, 0, 3, 3], Dwell { min_len: 2, scope: Scope::All, absorb: Absorb::Light }),
        vec![3, 3, 1, 3, 3]
    );
}

fn candidates() -> Vec<Dwell> {
    let mut out = Vec::new();
    for min_len in [2usize, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30] {
        for scope in [Scope::All, Scope::Wake, Scope::Deep, Scope::Rem, Scope::DeepRem] {
            for absorb in [Absorb::Longer, Absorb::Prev, Absorb::Light] {
                out.push(Dwell { min_len, scope, absorb });
            }
        }
    }
    out
}

/// One night's paired delta under one candidate, `None` when the night carries too little truth.
fn dwell_delta(nt: &Night, d: Dwell, base: f64) -> Option<f64> {
    night_kappa(&apply_dwell(&nt.pred, d), &nt.truth).map(|k| k - base)
}

fn section_short_bout(cohorts: &[(&str, Vec<Night>)]) {
    println!("\n6  a short-bout filter as its own post-decode step");
    println!("   The decoder already under-runs truth, and the winning transition temperature made the");
    println!("   path smoother still, so the candidate is: absorb runs shorter than L into a neighbour.");
    println!("   Selected on the training pair only, then reported once on the held-out cohort.\n");
    assert_dwell();

    // Run counts first: the fact the candidate is derived from, per cohort rather than on one set.
    println!("   {:<14} {:>8} {:>12} {:>12} {:>14}", "cohort", "nights", "truth runs", "v2 runs", "v2 runs < 3");
    for (name, nights) in cohorts {
        let (mut tr, mut pr, mut short) = (Vec::new(), Vec::new(), Vec::new());
        for nt in nights {
            let (mut p, mut t) = (Vec::new(), Vec::new());
            for (k, want) in nt.truth.iter().enumerate().take(nt.pred.len()) {
                if let Some(w) = want {
                    p.push(nt.pred[k]);
                    t.push(*w);
                }
            }
            if t.len() < MIN_SCORED {
                continue;
            }
            tr.push(runs(&t).len() as f64);
            pr.push(runs(&p).len() as f64);
            short.push(runs(&p).iter().filter(|(_, l)| *l < 3).count() as f64);
        }
        println!(
            "   {name:<14} {:>8} {:>12.0} {:>12.0} {:>14.0}",
            tr.len(),
            median(&mut tr),
            median(&mut pr),
            median(&mut short)
        );
    }
    println!("   (medians of the per-night run counts, over the labelled epochs only)\n");

    let cands = candidates();
    // `delta[cohort][cand][night]`, and the base kappa each delta is measured from.
    let mut per: Vec<Vec<Vec<f64>>> = Vec::new();
    for (_, nights) in cohorts {
        let mut rows = vec![Vec::new(); cands.len()];
        for nt in nights {
            let Some(base) = night_kappa(&nt.pred, &nt.truth) else { continue };
            for (ci, c) in cands.iter().enumerate() {
                if let Some(d) = dwell_delta(nt, *c, base) {
                    rows[ci].push(d);
                }
            }
        }
        per.push(rows);
    }

    // Two selection rules and two grids, all four specifiable before seeing a held-out number. POOLED
    // maximises the mean over the training nights, so DREAMT's 100 dominate; MIN-COHORT maximises the
    // WORSE of the two training cohorts, which refuses a candidate that only one of them likes.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let full: Vec<usize> = (0..cands.len()).collect();
    let capped: Vec<usize> = (0..cands.len()).filter(|i| cands[*i].min_len <= 8).collect();
    println!(
        "   {:<14} {:<10} {:<8} {:<24} {:>11} {:>10} {:>6}   verdict",
        "held out", "rule", "grid", "selected on the training pair", "held d-k4", "bar +/-", "n"
    );
    for (hi, (hname, _)) in cohorts.iter().enumerate() {
        let tr: Vec<usize> = (0..cohorts.len()).filter(|i| *i != hi).collect();
        for (gname, grid) in [("L<=8", &capped), ("full", &full)] {
            for rule in ["POOLED", "MIN-COH"] {
                let scoreof = |ci: usize| -> f64 {
                    let arms: Vec<f64> = tr.iter().map(|t| mean(&per[*t][ci])).collect();
                    match rule {
                        "MIN-COH" => arms.iter().cloned().fold(f64::INFINITY, f64::min),
                        _ => {
                            let all: Vec<f64> =
                                tr.iter().flat_map(|t| per[*t][ci].iter().copied()).collect();
                            mean(&all)
                        }
                    }
                };
                let pick =
                    *grid.iter().max_by(|a, b| scoreof(**a).total_cmp(&scoreof(**b))).expect("a candidate");
                let held = &per[hi][pick];
                let (m, bar) = paired_bar(held).unwrap_or((f64::NAN, f64::NAN));
                let verdict = if !m.is_finite() {
                    "-"
                } else if m.abs() <= bar {
                    "matches"
                } else if m > 0.0 {
                    "BEATS v2"
                } else {
                    "worse"
                };
                println!(
                    "   {hname:<14} {rule:<10} {gname:<8} {:<24} {m:>+11.4} {bar:>10.4} {:>6}   {verdict}",
                    cands[pick].name(),
                    held.len()
                );
            }
        }
        // The inner LOO estimate of the POOLED rule on the full grid, and the oracle, for scale. The
        // gap between them and the held-out row above is what the selection itself costs.
        let mut train: Vec<Vec<f64>> = vec![Vec::new(); cands.len()];
        for (ci, row) in train.iter_mut().enumerate() {
            for t in &tr {
                row.extend_from_slice(&per[*t][ci]);
            }
        }
        let n_train = train[0].len();
        let sums: Vec<f64> = train.iter().map(|v| v.iter().sum()).collect();
        let inner: Vec<f64> = (0..n_train)
            .map(|i| {
                let best = (0..cands.len())
                    .max_by(|a, b| (sums[*a] - train[*a][i]).total_cmp(&(sums[*b] - train[*b][i])))
                    .expect("a candidate");
                train[best][i]
            })
            .collect();
        let (im, ib) = paired_bar(&inner).unwrap_or((f64::NAN, f64::NAN));
        let oracle = full
            .iter()
            .max_by(|a, b| mean(&per[hi][**a]).total_cmp(&mean(&per[hi][**b])))
            .copied()
            .expect("a candidate");
        println!(
            "   {:<14} inner LOO over {n_train} training nights {im:+.4} +/- {ib:.4} · oracle on the held \
             cohort {} {:+.4}",
            "",
            cands[oracle].name(),
            mean(&per[hi][oracle])
        );
    }

    // The cheapest controls: fixed settings nobody selected, the neighbour-blind null on the same
    // scope, and the two single-class scopes that say which class carries the effect.
    println!("\n   controls, every cohort, no selection at all:");
    println!("   {:<26} {:>12} {:>12} {:>12}", "fixed candidate", PSG[0], PSG[1], PSG[2]);
    let fixed = [
        Dwell { min_len: 2, scope: Scope::All, absorb: Absorb::Longer },
        Dwell { min_len: 4, scope: Scope::All, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::All, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::Wake, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::Deep, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::Rem, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::DeepRem, absorb: Absorb::Longer },
        Dwell { min_len: 8, scope: Scope::DeepRem, absorb: Absorb::Light },
        Dwell { min_len: 30, scope: Scope::DeepRem, absorb: Absorb::Longer },
    ];
    for c in fixed {
        let ci = cand_index(&cands, c);
        print!("   {:<26}", c.name());
        for rows in per.iter() {
            let v = &rows[ci];
            match paired_bar(v) {
                Some((m, b)) if m.abs() > b => print!(" {m:>+11.4}*"),
                Some((m, _)) => print!(" {m:>+11.4} "),
                None => print!(" {:>12}", "-"),
            }
        }
        println!();
    }
    println!("   * resolves against its own paired bar. `->light` ignores the neighbours entirely, so a");
    println!("   scope whose win survives it is a base-rate shift and not smoothing.");

    // What the adopted candidate DOES to the four classes, so a kappa gain by deleting a class we
    // over-predict is named rather than hidden.
    let adopted = Dwell { min_len: 8, scope: Scope::DeepRem, absorb: Absorb::Longer };
    println!("\n   what {} does per class, pooled over each cohort:", adopted.name());
    println!(
        "   {:<14} {:<8} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "cohort", "class", "pred %", "pred % new", "recall", "recall new", "runs"
    );
    for (name, nights) in cohorts {
        let (mut before, mut after) = ([[0i64; 4]; 4], [[0i64; 4]; 4]);
        let (mut rb, mut ra) = (Vec::new(), Vec::new());
        for nt in nights {
            let filtered = apply_dwell(&nt.pred, adopted);
            add_cm(&mut before, &night_confusion(&nt.pred, &nt.truth));
            add_cm(&mut after, &night_confusion(&filtered, &nt.truth));
            rb.push(runs(&nt.pred).len() as f64);
            ra.push(runs(&filtered).len() as f64);
        }
        let share = |cm: &[[i64; 4]; 4], c: usize| {
            let tot: i64 = cm.iter().flatten().sum();
            100.0 * cm.iter().map(|r| r[c]).sum::<i64>() as f64 / tot.max(1) as f64
        };
        let rec = |cm: &[[i64; 4]; 4], c: usize| {
            let a: i64 = cm[c].iter().sum();
            100.0 * cm[c][c] as f64 / a.max(1) as f64
        };
        for (c, label) in ["wake", "light", "deep", "rem"].iter().enumerate() {
            println!(
                "   {:<14} {label:<8} {:>10.1} {:>10.1} {:>12.1} {:>12.1} {:>10}",
                if c == 0 { *name } else { "" },
                share(&before, c),
                share(&after, c),
                rec(&before, c),
                rec(&after, c),
                if c == 0 {
                    format!("{:.0} -> {:.0}", median(&mut rb.clone()), median(&mut ra.clone()))
                } else {
                    String::new()
                }
            );
        }
    }

    // Not one night carrying the whole effect: the extremes under the adopted candidate.
    println!("\n   the extremes under {}, so no cohort rests on one night:", adopted.name());
    for (name, nights) in cohorts {
        let mut rows: Vec<(f64, &str)> = Vec::new();
        for nt in nights {
            let Some(base) = night_kappa(&nt.pred, &nt.truth) else { continue };
            if let Some(d) = dwell_delta(nt, adopted, base) {
                rows.push((d, nt.name.as_str()));
            }
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let helped = rows.iter().filter(|(d, _)| *d > 0.0).count();
        let hurt = rows.iter().filter(|(d, _)| *d < 0.0).count();
        if let (Some(lo), Some(hi)) = (rows.first(), rows.last()) {
            println!(
                "   {name:<14} {helped} helped / {hurt} hurt / {} unchanged · worst {:+.4} ({}) · best {:+.4} ({})",
                rows.len() - helped - hurt,
                lo.0,
                lo.1,
                hi.0,
                hi.1
            );
        }
    }

    println!("\n   This filter runs AFTER the decoder and reads no emission, so it applies to v2's own");
    println!("   path unchanged. Any win here is a v2 finding, not a tanv1 one.");
}

fn cand_index(cands: &[Dwell], want: Dwell) -> usize {
    cands
        .iter()
        .position(|x| x.min_len == want.min_len && x.scope == want.scope && x.absorb == want.absorb)
        .expect("candidate in the grid")
}

fn main() {
    let cohorts: Vec<(&str, Vec<Night>)> =
        PSG.iter().map(|c| (*c, load_set(c))).filter(|(_, n)| !n.is_empty()).collect();
    if cohorts.len() < 2 {
        println!("need at least two PSG cohorts under {}", fixtures_root().display());
        return;
    }
    section_scales(&cohorts);
    section_gate_census();
    section_step_by_device();

    let ours = load_set("ours");
    let spans = continuous_spans();
    if spans.is_empty() {
        println!("\nno `continuous` spans: sections 3 and 4 cannot run");
        return;
    }
    section_refined_subset(&spans, &ours);
    section_gate_sweep();
    section_gate_relaxed(&ours);
    section_segmentation(&cohorts, &ours);
    section_short_bout(&cohorts);
}
