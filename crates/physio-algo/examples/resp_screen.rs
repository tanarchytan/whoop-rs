//! Screen 3 of the roster: does respiration derived from the beat series carry anything ON TOP of
//! the confirmed cardiac family?
//!
//!   cargo run --release -p physio-algo --example resp_screen [nights] [skip]
//!
//! The two independent non-EEG reference stagers are respiration-first, and the only respiration our
//! channel can carry is the respiratory sinus arrhythmia inside the R-R series. `sleep::resp_features`
//! extracts it; this asks whether it is worth anything the cardiac family does not already have.
//!
//! The BASELINE is therefore the whole confirmed family - mean HR, the three R-R summaries and the
//! fourteen order statistics - not a bare mean HR. A respiration column that only re-states the
//! cardiac level would clear a weak baseline and mean nothing.
//!
//! Arms are EXACT-NN and TIMING-NN. NN keeps normal-to-normal intervals only, so an ectopic beat
//! cannot masquerade as a breath; TIMING pushes the beats through our own whole-second wire first.
//! There is no EXACT arm: a respiratory column that needs sub-second beat timing is not a candidate
//! for this band, so the question is only whether NN survives the wire.
//!
//! The gate is the PERMUTED-COLUMN NULL - the same column shuffled WITHIN its own night, which keeps
//! its distribution and its per-night scaling and destroys only its alignment to stage.
//!
//! MESA selects and AAUWSS confirms, exactly as the cardiac pair does. DREAMT never appears: 34.4%
//! of its intervals repeat their predecessor, so anything R-R read off it scores the fixture builder.

mod common;

use common::mesa::{self, MesaNight};
use common::screen::{self, MEAN_HR, PCTL, PERM_SEED, RIDGE, RR_SUMMARY};
use common::{dirs_of, read_meta, read_rr, read_truth, reconstruct_beats, require_psg};
use physio_algo::lda::Lda;
use physio_algo::sleep::metrics::{balanced_accuracy, confusion4, recall, Confusion4};
use physio_algo::sleep::{cardiac, resp_features};

const FOLDS: usize = 5;
/// Cardiac columns first, then respiratory, in each producer's own `NAMES` order.
const NC: usize = cardiac::NAMES.len();
const NR: usize = resp_features::NAMES.len();
const NF: usize = NC + NR;
/// Draws of the permuted null. Two on MESA is what 400 recordings carry in reasonable time; the
/// confirmation reports its RANGE over three, so the thinness is visible rather than averaged away.
const MESA_DRAWS: u64 = 2;
const CONFIRM_DRAWS: u64 = 3;
const COHORT: &str = "aauwss";

struct Row {
    night: usize,
    x: [f64; NF],
    y: usize,
}

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn col_name(f: usize) -> &'static str {
    if f < NC { cardiac::NAMES[f] } else { resp_features::NAMES[f - NC] }
}

/// The whole confirmed cardiac family, built from the shared column indices rather than `0..NC`, so
/// this reads the same set the two cardiac screens do.
fn baseline() -> Vec<usize> {
    let mut b = vec![MEAN_HR];
    b.extend(RR_SUMMARY);
    b.extend(PCTL);
    b
}

fn resp_cols() -> Vec<usize> {
    (NC..NF).collect()
}

/// One night's rows: both producers over the same centred window, each column then z-scored within
/// that night. A column the night cannot carry stays NaN rather than becoming a manufactured mean.
fn rows_of(beats: &[(f64, f64)], epochs: &[(f64, usize)], night: usize) -> Vec<Row> {
    let (mut raw, mut ys) = (Vec::new(), Vec::new());
    for (centre, y) in epochs {
        let (a, b) = (centre - screen::WINDOW_S / 2.0, centre + screen::WINDOW_S / 2.0);
        let (Some(c), Some(r)) = (cardiac::extract(beats, a, b), resp_features::extract(beats, a, b))
        else {
            continue;
        };
        let (cr, rr) = (c.row(), r.row());
        let mut x = [f64::NAN; NF];
        x[..NC].copy_from_slice(&cr);
        x[NC..].copy_from_slice(&rr);
        raw.push(x);
        ys.push(*y);
    }
    let mut z = vec![[f64::NAN; NF]; raw.len()];
    for f in 0..NF {
        let col: Vec<Option<f64>> = raw.iter().map(|r| r[f].is_finite().then_some(r[f])).collect();
        for (k, v) in cardiac::zscore_column(&col).into_iter().enumerate() {
            z[k][f] = v.unwrap_or(f64::NAN);
        }
    }
    z.into_iter().zip(ys).map(|(x, y)| Row { night, x, y }).collect()
}

/// Per-night row building, spread over the machine. Each night is independent and lands back in its
/// own slot, so the output is the same whatever the thread count.
fn build_mesa(nights: &[MesaNight], arm: &str) -> Vec<Row> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = nights.len().div_ceil(threads.max(1)).max(1);
    let mut out: Vec<Vec<Row>> = (0..nights.len()).map(|_| Vec::new()).collect();
    std::thread::scope(|s| {
        for (c, (inp, outp)) in nights.chunks(chunk).zip(out.chunks_mut(chunk)).enumerate() {
            s.spawn(move || {
                for (k, n) in inp.iter().enumerate() {
                    let beats = match arm {
                        "TIMING-NN" => mesa::degrade_timing(&mesa::nn_only(&n.beats)),
                        _ => mesa::nn_only(&n.beats),
                    };
                    let pairs: Vec<(f64, f64)> = beats.iter().map(|b| (b.t, b.rr)).collect();
                    let epochs: Vec<(f64, usize)> = n
                        .stage
                        .iter()
                        .enumerate()
                        .filter_map(|(e, s)| {
                            s.map(|s| (e as f64 * screen::EPOCH_S + screen::EPOCH_S / 2.0, s))
                        })
                        .collect();
                    outp[k] = rows_of(&pairs, &epochs, c * chunk + k);
                }
            });
        }
    });
    out.into_iter().flatten().collect()
}

/// The confirmation cohort: wrist optical, PSG-scored, R-R through our own whole-second stamps.
fn build_confirm() -> (Vec<Row>, usize) {
    require_psg(COHORT);
    let (mut out, mut nights) = (Vec::new(), 0usize);
    for dir in &dirs_of(COHORT) {
        let truth = read_truth(dir);
        let Some((w0, _, _)) = read_meta(dir) else { continue };
        let beats = reconstruct_beats(&read_rr(dir));
        if truth.is_empty() || beats.len() < 100 {
            continue;
        }
        let epochs: Vec<(f64, usize)> = truth
            .iter()
            .filter(|(_, t)| (0..4).contains(*t))
            .map(|(k, t)| {
                (w0 as f64 + *k as f64 * screen::EPOCH_S + screen::EPOCH_S / 2.0, *t as usize)
            })
            .collect();
        let rows = rows_of(&beats, &epochs, nights);
        if rows.len() < 50 {
            continue;
        }
        out.extend(rows);
        nights += 1;
    }
    (out, nights)
}

/// Held-out confusion, splitting by RECORDING. `None` if a fold cannot fit.
fn held_out(rows: &[Row], cols: &[usize]) -> Option<Confusion4> {
    let mut cm = [[0i64; 4]; 4];
    for fold in 0..FOLDS {
        accumulate(rows, cols, &mut cm, |r| r.night % FOLDS == fold)?;
    }
    Some(cm)
}

/// Leave ONE recording out, every recording in turn. With 13 nights a 5-fold split would train on 10
/// and test on 3, and the fold-to-fold spread would swamp the effect being measured.
fn loro(rows: &[Row], nights: usize, cols: &[usize]) -> Option<Confusion4> {
    let mut cm = [[0i64; 4]; 4];
    let mut used = 0;
    for held in 0..nights {
        if accumulate(rows, cols, &mut cm, |r| r.night == held).is_some() {
            used += 1;
        }
    }
    (used > 0).then_some(cm)
}

/// Fit on the rows `is_test` rejects, score the rest into `cm`. A fold whose training side lacks a
/// class cannot be fitted against; the caller decides whether that is fatal.
fn accumulate(
    rows: &[Row],
    cols: &[usize],
    cm: &mut Confusion4,
    is_test: impl Fn(&Row) -> bool,
) -> Option<()> {
    let take = |train: bool| -> (Vec<Vec<f64>>, Vec<usize>) {
        let sel: Vec<&Row> = rows.iter().filter(|r| is_test(r) != train).collect();
        (
            sel.iter().map(|r| cols.iter().map(|c| r.x[*c]).collect()).collect(),
            sel.iter().map(|r| r.y).collect(),
        )
    };
    let (xtr, ytr) = take(true);
    let m = Lda::fit(&xtr, &ytr, RIDGE)?;
    let (xte, yte) = take(false);
    let (mut p, mut t) = (Vec::new(), Vec::new());
    for (row, y) in xte.iter().zip(&yte) {
        if let Some(c) = m.predict(row) {
            p.push(c);
            t.push(*y);
        }
    }
    for (o, a) in cm.iter_mut().zip(&confusion4(&p, &t)) {
        for (x, y) in o.iter_mut().zip(a) {
            *x += *y;
        }
    }
    Some(())
}

/// Shuffle the named columns WITHIN each night. Distribution and per-night scaling survive; only the
/// alignment to stage is destroyed, which is exactly the thing being claimed.
fn permuted(rows: &[Row], cols: &[usize], seed: u64) -> Vec<Row> {
    let mut out: Vec<Row> = rows.iter().map(|r| Row { night: r.night, x: r.x, y: r.y }).collect();
    let mut start = 0;
    while start < out.len() {
        let mut end = start;
        while end < out.len() && out[end].night == out[start].night {
            end += 1;
        }
        for (k, c) in cols.iter().enumerate() {
            let mut s = seed ^ splitmix((out[start].night as u64) << 8 | k as u64);
            for i in (start + 1..end).rev() {
                s = splitmix(s);
                let j = start + (s >> 11) as usize % (i - start + 1);
                let tmp = out[i].x[*c];
                out[i].x[*c] = out[j].x[*c];
                out[j].x[*c] = tmp;
            }
        }
        start = end;
    }
    out
}

/// `fonseca2015` eq 5: absolute standardised mean difference of one class against the rest, over the
/// pooled SD. A second ranking, so the ordering is not an artefact of using the held-out fit.
fn asmd(rows: &[Row], f: usize, c: usize) -> Option<f64> {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for r in rows {
        if r.x[f].is_finite() {
            if r.y == c { &mut a } else { &mut b }.push(r.x[f]);
        }
    }
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let var =
        |v: &[f64], mu: f64| v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    let (ma, mb) = (m(&a), m(&b));
    let pooled = (((a.len() - 1) as f64 * var(&a, ma) + (b.len() - 1) as f64 * var(&b, mb))
        / (a.len() + b.len() - 2) as f64)
        .sqrt();
    (pooled > 1e-12).then(|| ((ma - mb) / pooled).abs())
}

fn ba(cm: &Confusion4) -> f64 {
    balanced_accuracy(cm).unwrap_or(f64::NAN)
}

/// Predicted share of each class. Balanced accuracy is a mean of RECALLS, and a recall can be bought
/// by calling the class more often, so a gain is only a gain if the calling did not run ahead of it.
fn calls(cm: &Confusion4) -> [f64; 4] {
    let tot: i64 = cm.iter().flatten().sum();
    std::array::from_fn(|c| cm.iter().map(|r| r[c]).sum::<i64>() as f64 / tot.max(1) as f64)
}

fn recalls(cm: &Confusion4) -> [f64; 4] {
    std::array::from_fn(|c| recall(cm, c).unwrap_or(f64::NAN))
}

fn show(tag: &str, cm: &Confusion4) {
    let r: Vec<String> = recalls(cm).iter().map(|v| format!("{v:.3}")).collect();
    let q = calls(cm);
    println!(
        "  {tag:<28} BA {:.4}   recall w/l/d/r  {}   call% {:.0}/{:.0}/{:.0}/{:.0}",
        ba(cm),
        r.join(" "),
        q[0] * 100.0,
        q[1] * 100.0,
        q[2] * 100.0,
        q[3] * 100.0
    );
}

/// The guard the transition arms failed: a recall that rose no faster than the calling share is the
/// engine calling the class more, not reading it better.
fn calling_guard(base: &Confusion4, with: &Confusion4) {
    println!("\n  CALLING SHARE beside every recall - sub-proportional is a loss wearing a gain:");
    println!("  {:<7} {:>16} {:>16} {:>9}", "class", "call% base->all", "recall base->all", "verdict");
    let (qb, qw, rb, rw) = (calls(base), calls(with), recalls(base), recalls(with));
    for (c, name) in ["wake", "light", "deep", "rem"].iter().enumerate() {
        let (dq, dr) = (qw[c] / qb[c], rw[c] / rb[c]);
        let verdict = if dr > dq { "over" } else { "SUB" };
        println!(
            "  {name:<7} {:>6.1} ->{:>6.1} {:>8.3} ->{:>7.3} {:>9}   x{dq:.2} calling for x{dr:.2} recall",
            qb[c] * 100.0,
            qw[c] * 100.0,
            rb[c],
            rw[c],
            verdict
        );
    }
}

fn asmd_table(rows: &[Row]) {
    println!("\n  ASMD (fonseca2015 eq 5), a second ranking on the respiratory columns:");
    println!("  {:<18} {:>7} {:>7} {:>7} {:>7}", "column", "wake", "light", "deep", "rem");
    for f in resp_cols() {
        let cells: Vec<String> = (0..4)
            .map(|c| asmd(rows, f, c).map_or("  -  ".into(), |v| format!("{v:.3}")))
            .collect();
        println!(
            "  {:<18} {:>7} {:>7} {:>7} {:>7}",
            col_name(f),
            cells[0],
            cells[1],
            cells[2],
            cells[3]
        );
    }
}

/// Present, per column: how many rows carry it at all. A column absent on most epochs cannot be
/// worth anything and would otherwise look merely weak.
fn coverage(rows: &[Row]) {
    println!("\n  column coverage (rows carrying the column at all):");
    for f in resp_cols() {
        let n = rows.iter().filter(|r| r.x[f].is_finite()).count();
        println!("  {:<18} {:>7.1}%", col_name(f), 100.0 * n as f64 / rows.len().max(1) as f64);
    }
}

fn mesa_arm(nights: &[MesaNight], arm: &str) {
    let rows = build_mesa(nights, arm);
    let mut mix = [0usize; 4];
    for r in &rows {
        mix[r.y] += 1;
    }
    println!("\n\n===== MESA {arm} =====  {} epochs (w/l/d/r {mix:?})", rows.len());

    let base = baseline();
    let Some(cb) = held_out(&rows, &base) else {
        println!("the baseline could not fit - nothing below would mean anything");
        return;
    };
    show("B_cardiac  (18 columns)", &cb);
    coverage(&rows);

    println!(
        "\n  {:<18} {:>8} {:>9} {:>9} {:>8}   {:<23} call% w/l/d/r",
        "+ candidate", "BA", "d(BA)", "null d", "verdict", "recall w/l/d/r"
    );
    let mut ranked: Vec<(f64, usize)> = Vec::new();
    for f in resp_cols() {
        let mut cols = base.clone();
        cols.push(f);
        let Some(cm) = held_out(&rows, &cols) else { continue };
        let d = ba(&cm) - ba(&cb);
        let null = (0..MESA_DRAWS)
            .filter_map(|k| {
                held_out(&permuted(&rows, &[f], PERM_SEED ^ k), &cols).map(|c| ba(&c) - ba(&cb))
            })
            .fold(f64::NEG_INFINITY, f64::max);
        let r: Vec<String> = recalls(&cm).iter().map(|v| format!("{v:.3}")).collect();
        let q = calls(&cm);
        println!(
            "  {:<18} {:>8.4} {:>+9.4} {:>+9.4} {:>8}   {:<23} {:.0}/{:.0}/{:.0}/{:.0}",
            col_name(f),
            ba(&cm),
            d,
            null,
            if d > null { "over" } else { "NULL" },
            r.join(" "),
            q[0] * 100.0,
            q[1] * 100.0,
            q[2] * 100.0,
            q[3] * 100.0
        );
        ranked.push((d, f));
    }

    let mut all = base.clone();
    all.extend(resp_cols());
    if let Some(cm) = held_out(&rows, &all) {
        let null = (0..MESA_DRAWS)
            .filter_map(|k| {
                held_out(&permuted(&rows, &resp_cols(), PERM_SEED ^ k), &all).map(|c| ba(&c) - ba(&cb))
            })
            .fold(f64::NEG_INFINITY, f64::max);
        println!();
        show("B_cardiac + ALL respiration", &cm);
        println!(
            "  family d(BA) {:+.4} against a permuted-family null of {null:+.4}",
            ba(&cm) - ba(&cb)
        );
        calling_guard(&cb, &cm);
    }

    asmd_table(&rows);
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top: Vec<String> =
        ranked.iter().take(4).map(|(d, f)| format!("{} {d:+.4}", col_name(*f))).collect();
    println!("\n  by held-out d(BA): {}", top.join(" | "));
}

fn confirm() {
    let (rows, nights) = build_confirm();
    let mut mix = [0usize; 4];
    for r in &rows {
        mix[r.y] += 1;
    }
    println!(
        "\n\n===== CONFIRMATION on {COHORT} =====  {nights} recordings, {} epochs (w/l/d/r {mix:?})",
        rows.len()
    );
    println!("  Wrist optical, PSG-scored, R-R through our own whole-second stamps. Leave-one-");
    println!("  recording-out. MESA selects on ECG; only a wrist cohort can confirm.\n");

    let base = baseline();
    let mut all = base.clone();
    all.extend(resp_cols());
    let (Some(cb), Some(ca)) = (loro(&rows, nights, &base), loro(&rows, nights, &all)) else {
        println!("a baseline could not fit on any held-out night - nothing here would mean anything");
        return;
    };
    show("B_cardiac  (18 columns)", &cb);
    show("B_cardiac + ALL respiration", &ca);
    coverage(&rows);

    let nulls: Vec<f64> = (0..CONFIRM_DRAWS)
        .filter_map(|k| {
            loro(&permuted(&rows, &resp_cols(), PERM_SEED ^ k), nights, &all).map(|c| ba(&c))
        })
        .collect();
    let lo = nulls.iter().cloned().fold(f64::INFINITY, f64::min) - ba(&cb);
    let hi = nulls.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - ba(&cb);
    let d = ba(&ca) - ba(&cb);
    println!("\n  family d(BA) {d:+.4}");
    println!("  permuted-family null d(BA) over {CONFIRM_DRAWS} draws: {lo:+.4} .. {hi:+.4}");

    println!(
        "\n  {:<18} {:>8} {:>9}   {:<23} call% w/l/d/r",
        "+ candidate", "BA", "d(BA)", "recall w/l/d/r"
    );
    for f in resp_cols() {
        let mut cols = base.clone();
        cols.push(f);
        let Some(cm) = loro(&rows, nights, &cols) else { continue };
        let r: Vec<String> = recalls(&cm).iter().map(|v| format!("{v:.3}")).collect();
        let q = calls(&cm);
        println!(
            "  {:<18} {:>8.4} {:>+9.4}   {:<23} {:.0}/{:.0}/{:.0}/{:.0}",
            col_name(f),
            ba(&cm),
            ba(&cm) - ba(&cb),
            r.join(" "),
            q[0] * 100.0,
            q[1] * 100.0,
            q[2] * 100.0,
            q[3] * 100.0
        );
    }

    calling_guard(&cb, &ca);
    asmd_table(&rows);
    println!(
        "\n  VERDICT: respiration {} its permuted null on a wrist cohort, on top of the cardiac family.",
        if d > hi { "CLEARS" } else { "DOES NOT CLEAR" }
    );
}

fn main() {
    let limit: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(400);
    let skip: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(0);
    println!("RESPIRATION SCREEN - {NR} columns on top of the {NC}-column cardiac family");
    println!("Held out by RECORDING. Uniform priors in the fit, so a class is not called more just");
    println!("for being common. Every gain is read against its own within-night permuted null.");

    if limit > 0 {
        let nights = mesa::nights_from(skip, limit);
        println!("\nMESA: {} recordings, skipping the first {skip}", nights.len());
        for arm in ["EXACT-NN", "TIMING-NN"] {
            mesa_arm(&nights, arm);
        }
    }
    confirm();
}
