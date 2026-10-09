//! Unit E1: NOOP Effort (strain.rs) against WHOOP's own day strain, and four candidate variants.
//!
//!   cargo run --release -p physio-algo --example unit_e_effort
//!
//! Pairs a WHOOP export strain (per cycle, and per logged workout) with the strap HR the same window
//! holds. The shipped Effort is `strain::strain` itself; the candidates are built from the same
//! gap policy (`hr_gap::creditable_seconds`) so a variant differs from the shipped path in its zone
//! weights or its denominator and in nothing else. Fitted variants are scored leave-one-wearer-out
//! (the denominator comes from the OTHER wearer) and leave-one-day-out (from every other day of the same
//! wearer), never in-sample. WHOOP's number is a reference with no ground truth, and every pair compares
//! two different straps on one wearer's body.
//!
//! Reads `WHOOP_ANALYSIS_DIR` (see `unit_e`). Writes per-pair rows to `$WHOOP_ANALYSIS_DIR/out/`, which
//! is outside any repo; only the printed summary is meant for a document.

mod unit_e;

use physio_algo::hr_gap::{creditable_seconds, GapPosition};
use physio_algo::hr_sample::HrSample;
use physio_algo::resting_hr::session_resting_hr;
use physio_algo::strain::{
    fit_strain_denominator, strain, tanaka_hrmax, trimp_to_strain, Method, EFFORT_TO_WHOOP_DAY_STRAIN,
    STRAIN_DENOMINATOR, edwards_trimp_interval,
};
use unit_e::*;

const MIN_COVERAGE: f64 = 0.85;
/// WHOOP day strain below this is a "quiet" day, at or above it an "active" one (0-21 axis).
const QUIET_BELOW: f64 = 8.0;
const BANISTER_B_MEN: f64 = 1.92;
const BANISTER_SCALE: f64 = 0.64;

fn half(g: f64, pos: GapPosition) -> f64 {
    if g <= 0.0 { 0.0 } else { creditable_seconds(g, pos) / 2.0 }
}

/// Interval-billed TRIMP with an arbitrary per-minute weight on %HRR. With the Edwards weights this is
/// `strain::edwards_trimp_interval` line for line (asserted below), so only `w` differs between variants.
fn trimp_with(hr: &[HrSample], rest: f64, reserve: f64, w: impl Fn(f64) -> f64) -> f64 {
    let n = hr.len();
    if n == 0 {
        return 0.0;
    }
    let gap = |a: usize, b: usize| (hr[a].ts - hr[b].ts).unsigned_abs() as f64;
    let mut total = 0.0;
    for (i, s) in hr.iter().enumerate() {
        let pct = (s.bpm as f64 - rest) / reserve * 100.0;
        let secs = if n == 1 {
            60.0
        } else if i == 0 {
            half(gap(1, 0), GapPosition::Interior) + half(gap(1, 0), GapPosition::Leading)
        } else if i == n - 1 {
            half(gap(i, i - 1), GapPosition::Interior) + half(gap(i, i - 1), GapPosition::Trailing)
        } else {
            half(gap(i + 1, i), GapPosition::Interior) + half(gap(i, i - 1), GapPosition::Interior)
        };
        total += w(pct) * secs / 60.0;
    }
    total
}

fn w_edwards(p: f64) -> f64 {
    [(90.0, 5.0), (80.0, 4.0), (70.0, 3.0), (60.0, 2.0), (50.0, 1.0)]
        .iter()
        .find(|(t, _)| p >= *t)
        .map_or(0.0, |(_, w)| *w)
}
fn w_floor40(p: f64) -> f64 {
    if p >= 50.0 { w_edwards(p) } else if p >= 40.0 { 0.5 } else { 0.0 }
}
fn w_floor30(p: f64) -> f64 {
    if p >= 40.0 { w_floor40(p) } else if p >= 30.0 { 0.25 } else { 0.0 }
}
fn w_banister(p: f64) -> f64 {
    let x = (p / 100.0).clamp(0.0, 1.0);
    if x > 0.0 { x * BANISTER_SCALE * (BANISTER_B_MEN * x).exp() } else { 0.0 }
}

struct Row {
    wearer: &'static str,
    kind: &'static str,
    id: i64,
    whoop: f64,
    trimp: [f64; 4], // edwards, floor40, floor30, banister
    shipped_lib: f64,
    cov: f64,
    rest: f64,
}

const VARIANTS: [&str; 4] = ["S edwards", "F40 +40-50%", "F30 +30-50%", "B banister"];

struct Wearer {
    name: &'static str,
    refname: &'static str,
    streams: &'static [&'static str],
    age: f64,
}

fn main() {
    let wearers = [
        Wearer { name: "rep(4.0)", refname: "rep", streams: &["rep-bak", "rep-rc2"], age: 66.0 },
        Wearer { name: "dav(5.0/MG)", refname: "dav", streams: &["dav-409", "dav-360", "dav-910"], age: 35.0 },
    ];
    let mut rows: Vec<Row> = Vec::new();
    let mut cal_day: Vec<(&str, f64, f64)> = Vec::new(); // wearer, calendar-day shipped, whoop
    for w in &wearers {
        let sets: Vec<(&str, Streams)> = w.streams.iter().map(|n| (*n, load(n))).collect();
        let cyc = cycles(w.refname);
        // E1_HRMAX_<REFNAME>=bpm overrides Tanaka for one wearer (a sensitivity, not a shipped choice).
        let hrmax = std::env::var(format!("E1_HRMAX_{}", w.refname.to_uppercase())).ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| tanaka_hrmax(w.age));
        let mut skipped = (0, 0, 0);
        for c in &cyc {
            let (Some(end), Some(ws), Some(on), Some(wake)) = (c.end, c.strain, c.sleep_on, c.wake_on) else {
                skipped.0 += 1;
                continue;
            };
            // Best-covered stream for the whole cycle.
            let best = sets.iter().map(|(n, s)| (hr_coverage(&s.hr, c.start, end), *n, s)).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            if best.0 < MIN_COVERAGE {
                skipped.1 += 1;
                continue;
            }
            let s = best.2;
            let hr: Vec<HrSample> = slice_by(&s.hr, |x| x.0, c.start, end).iter().map(|&(t, b)| HrSample { ts: t, bpm: b }).collect();
            let night: Vec<HrSample> = slice_by(&s.hr, |x| x.0, on, wake).iter().map(|&(t, b)| HrSample { ts: t, bpm: b }).collect();
            let Some(rest) = session_resting_hr(on, wake, &night).map(|v| v as f64) else {
                skipped.2 += 1;
                continue;
            };
            let rest = if std::env::var("E1_REST_WHOOP").is_ok() { c.rhr.unwrap_or(rest) } else { rest };
            let reserve = hrmax - rest;
            let t: [f64; 4] = [
                trimp_with(&hr, rest, reserve, w_edwards),
                trimp_with(&hr, rest, reserve, w_floor40),
                trimp_with(&hr, rest, reserve, w_floor30),
                trimp_with(&hr, rest, reserve, w_banister),
            ];
            let lib_t = edwards_trimp_interval(&hr, rest, reserve);
            assert!((lib_t - t[0]).abs() < 1e-9, "generic Edwards drifted from strain.rs: {lib_t} vs {}", t[0]);
            let lib = strain(&hr, Some(hrmax), rest, Method::Edwards, "male", STRAIN_DENOMINATOR).unwrap_or(0.0);
            rows.push(Row { wearer: w.name, kind: "cycle", id: c.start, whoop: ws, trimp: t, shipped_lib: lib, cov: best.0, rest });

            // Sensitivity: the SHIPPED calendar-day window (local midnight to midnight) for the cycle's start date.
            let day0 = ((c.start + c.tz).div_euclid(86400)) * 86400 - c.tz;
            let dcov = hr_coverage(&s.hr, day0, day0 + 86400);
            if dcov >= MIN_COVERAGE {
                let dh: Vec<HrSample> = slice_by(&s.hr, |x| x.0, day0, day0 + 86400).iter().map(|&(t, b)| HrSample { ts: t, bpm: b }).collect();
                cal_day.push((w.name, strain(&dh, Some(hrmax), rest, Method::Edwards, "male", STRAIN_DENOMINATOR).unwrap_or(0.0) * EFFORT_TO_WHOOP_DAY_STRAIN, ws));
            }
        }
        println!("{}: {} cycles in export, skipped open/no-strain {}, coverage<{:.0}% {}, no resting HR {}", w.name, cyc.len(), skipped.0, MIN_COVERAGE * 100.0, skipped.1, skipped.2);

        // Workouts: WHOOP activity strain over the logged window, resting HR from that cycle's sleep.
        let wl = workouts(w.refname);
        let mut wskip = [0usize; 5];
        for wo in wl {
            let Some(ws) = wo.strain else { wskip[0] += 1; continue };
            let Some(c) = cyc.iter().find(|c| c.start == wo.cstart) else { wskip[1] += 1; continue };
            let (Some(on), Some(wake)) = (c.sleep_on, c.wake_on) else { wskip[1] += 1; continue };
            let best = sets.iter().map(|(_, s)| (hr_coverage(&s.hr, wo.start, wo.end), s)).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            if best.0 < 0.9 || wo.end - wo.start < 600 {
                wskip[2] += 1;
                continue;
            }
            let s = best.1;
            let hr: Vec<HrSample> = slice_by(&s.hr, |x| x.0, wo.start, wo.end).iter().map(|&(t, b)| HrSample { ts: t, bpm: b }).collect();
            let night: Vec<HrSample> = slice_by(&s.hr, |x| x.0, on, wake).iter().map(|&(t, b)| HrSample { ts: t, bpm: b }).collect();
            let Some(rest) = session_resting_hr(on, wake, &night).map(|v| v as f64) else { wskip[3] += 1; continue };
            let rest = if std::env::var("E1_REST_WHOOP").is_ok() { c.rhr.unwrap_or(rest) } else { rest };
            let reserve = hrmax - rest;
            let t: [f64; 4] = [
                trimp_with(&hr, rest, reserve, w_edwards),
                trimp_with(&hr, rest, reserve, w_floor40),
                trimp_with(&hr, rest, reserve, w_floor30),
                trimp_with(&hr, rest, reserve, w_banister),
            ];
            let lib = strain(&hr, Some(hrmax), rest, Method::Edwards, "male", STRAIN_DENOMINATOR).unwrap_or(0.0);
            rows.push(Row { wearer: w.name, kind: "workout", id: wo.start, whoop: ws, trimp: t, shipped_lib: lib, cov: best.0, rest });
        }
        println!("{}: workouts skipped: no strain {}, no cycle/sleep {}, coverage<90% or <10 min {}, no resting HR {}", w.name, wskip[0], wskip[1], wskip[2], wskip[3]);
    }

    // Per-pair rows, outside the repo.
    let mut csv = String::from("wearer,kind,id,whoop,cov,rest,trimp_s,trimp_f40,trimp_f30,trimp_b,shipped_lib_axis100\n");
    for r in &rows {
        csv += &format!("{},{},{},{},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.2}\n", r.wearer, r.kind, r.id, r.whoop, r.cov, r.rest, r.trimp[0], r.trimp[1], r.trimp[2], r.trimp[3], r.shipped_lib);
    }
    write_out("e1_pairs.csv", &csv);

    for kind in ["cycle", "workout"] {
        println!("\n================ {kind} pairs ================");
        report(&rows, kind, &wearers.iter().map(|w| w.name).collect::<Vec<_>>());
    }

    println!("\n--- sensitivity: shipped Edwards on the CALENDAR day (local midnight) vs the cycle's WHOOP strain ---");
    for w in &wearers {
        let (p, t): (Vec<f64>, Vec<f64>) = cal_day.iter().filter(|c| c.0 == w.name).map(|c| (c.1, c.2)).unzip();
        println!("{:<12} n={:>3}  bias {:>6}  MAE {:>5}  r {:>5}", w.name, p.len(), fmt(bias(&p, &t)), fmt(mae(&p, &t)), fmt(pearson(&p, &t)));
    }
}

/// Predictions on the WHOOP 0-21 axis for every variant, for the rows of one kind.
fn predict(rows: &[&Row], all_kind: &[&Row]) -> Vec<(String, Vec<f64>)> {
    let ax = EFFORT_TO_WHOOP_DAY_STRAIN;
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    for (k, name) in VARIANTS.iter().enumerate() {
        out.push((name.to_string(), rows.iter().map(|r| trimp_to_strain(r.trimp[k], STRAIN_DENOMINATOR) * ax).collect()));
    }
    // Fitted denominators. LOWO: from the other wearer's pairs. LODO: from every other pair of this wearer.
    for (k, base) in [(0usize, "S"), (1, "F40"), (3, "B")] {
        let lowo: Vec<f64> = rows
            .iter()
            .map(|r| {
                let train: Vec<(f64, f64)> = all_kind.iter().filter(|o| o.wearer != r.wearer).map(|o| (o.trimp[k], o.whoop / ax)).collect();
                fit_strain_denominator(&train).map_or(f64::NAN, |d| trimp_to_strain(r.trimp[k], d) * ax)
            })
            .collect();
        out.push((format!("{base}+fitD LOWO"), lowo));
        let lodo: Vec<f64> = rows
            .iter()
            .map(|r| {
                let train: Vec<(f64, f64)> = all_kind.iter().filter(|o| o.wearer == r.wearer && o.id != r.id).map(|o| (o.trimp[k], o.whoop / ax)).collect();
                fit_strain_denominator(&train).map_or(f64::NAN, |d| trimp_to_strain(r.trimp[k], d) * ax)
            })
            .collect();
        out.push((format!("{base}+fitD LODO"), lodo));
        // Pooled leave-one-day-out: every OTHER pair of this kind, from both wearers.
        let loo: Vec<f64> = rows
            .iter()
            .map(|r| {
                let train: Vec<(f64, f64)> = all_kind.iter().filter(|o| !(o.wearer == r.wearer && o.id == r.id)).map(|o| (o.trimp[k], o.whoop / ax)).collect();
                fit_strain_denominator(&train).map_or(f64::NAN, |d| trimp_to_strain(r.trimp[k], d) * ax)
            })
            .collect();
        out.push((format!("{base}+fitD LOO-pooled"), loo));
    }
    out
}

fn report(rows: &[Row], kind: &str, names: &[&str]) {
    let all_kind: Vec<&Row> = rows.iter().filter(|r| r.kind == kind).collect();
    // Fitted denominators, for the record.
    for k in [0usize, 1, 3] {
        for w in names {
            let tr: Vec<(f64, f64)> = all_kind.iter().filter(|o| o.wearer != *w).map(|o| (o.trimp[k], o.whoop / EFFORT_TO_WHOOP_DAY_STRAIN)).collect();
            if let Ok(d) = fit_strain_denominator(&tr) {
                println!("  fitted D for {:<11} variant {:<12} trained on the other wearer: D = {:>10.1}  (ln D {:.2}, shipped ln 7201 = {:.2})", w, VARIANTS[k], d, d.ln(), STRAIN_DENOMINATOR.ln());
            }
        }
    }
    for w in names {
        let mine: Vec<&Row> = all_kind.iter().copied().filter(|r| r.wearer == *w).collect();
        for (cls, pick) in [("ALL", 0), ("QUIET (WHOOP<8)", 1), ("ACTIVE (WHOOP>=8)", 2)] {
            let sel: Vec<&Row> = mine.iter().copied().filter(|r| match pick { 0 => true, 1 => r.whoop < QUIET_BELOW, _ => r.whoop >= QUIET_BELOW }).collect();
            if sel.len() < 3 {
                println!("\n{w} {cls}: n={} (too few)", sel.len());
                continue;
            }
            let truth: Vec<f64> = sel.iter().map(|r| r.whoop).collect();
            let preds = predict(&sel, &all_kind);
            let base = preds[0].1.clone();
            println!("\n{w} {cls}: n={}  WHOOP mean {:.2}  (cov mean {:.2})", sel.len(), mean(&truth), mean(&sel.iter().map(|r| r.cov).collect::<Vec<_>>()));
            println!("  {:<16} {:>7} {:>6} {:>6} {:>6}  {:>8}  dMAE vs S [95% boot CI]", "variant", "bias", "MAE", "r", "rho", "meanPred");
            for (name, p) in &preds {
                if p.iter().any(|x| x.is_nan()) {
                    println!("  {name:<16} (no fit)");
                    continue;
                }
                let d: Vec<f64> = p.iter().zip(&truth).zip(&base).map(|((p, t), b)| (p - t).abs() - (b - t).abs()).collect();
                let (lo, hi) = boot_mean_ci(&d);
                println!("  {:<16} {:>7} {:>6} {:>6} {:>6}  {:>8}  {:>6} [{}, {}]", name, fmt(bias(p, &truth)), fmt(mae(p, &truth)), fmt(pearson(p, &truth)), fmt(spearman(p, &truth)), fmt(mean(p)), fmt(mean(&d)), fmt(lo), fmt(hi));
            }
            // Controls. Constant predictor = the OTHER wearer's mean WHOOP strain (what a day-blind scorer
            // fitted elsewhere would say) and this wearer's own mean (the best a constant can do).
            let other: Vec<f64> = all_kind.iter().filter(|o| o.wearer != *w).map(|o| o.whoop).collect();
            let c_other = vec![mean(&other); truth.len()];
            let c_own = vec![mean(&truth); truth.len()];
            println!("  control constant (other wearer mean {:.2}) MAE {}   constant (own mean) MAE {}", mean(&other), fmt(mae(&c_other, &truth)), fmt(mae(&c_own, &truth)));
            // Shuffled-label null for the fixed-D variants: permute WHOOP strain across this wearer's days.
            let mut rng = Rng(0xC0FFEE);
            let mut line = String::from("  control shuffled-label MAE (mean / 5th pct):");
            for (name, p) in preds.iter().take(4) {
                let mut ms: Vec<f64> = (0..1000)
                    .map(|_| {
                        let mut t = truth.clone();
                        for i in (1..t.len()).rev() {
                            t.swap(i, rng.below(i + 1));
                        }
                        mae(p, &t)
                    })
                    .collect();
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                line += &format!("  {} {:.2}/{:.2}", name.split(' ').next().unwrap(), mean(&ms), ms[50]);
            }
            println!("{line}");
        }
    }
}
