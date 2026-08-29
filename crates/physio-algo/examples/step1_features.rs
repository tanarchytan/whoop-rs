//! Step 1 — how each per-epoch feature is COMPUTED from the raw streams, and whether a better
//! computation exists.
//!
//!   cargo run --release -p physio-algo --example step1_features
//!
//! v2 extracts its own epoch features in `sleep/v2.rs`; tanv1 extracts a 28-column vector in
//! `sleep/features.rs` plus `examples/common/cardiac.rs`. Several columns claim the same physical
//! quantity. This measures whether they are the same computation, what the windows do at the edges
//! and across a gap, where an absent sample becomes a number, what `hr_var` is a statistic of, how
//! our motion proxy compares with the strap's own on-chip ENMO, and then decodes a repaired
//! computation against the shipped one, paired per night.
//!
//! Nothing here changes `src`. Every v2 quantity is RECOMPUTED here and pinned against the shipped
//! emission's own design columns first, so an arm differs in exactly one computation.

mod common;

use std::collections::BTreeMap;

use common::{
    cardiac_series, dirs_of, per_second_hr, read_accel, read_dyn_accel, read_hr, read_meta, read_rr,
    read_truth, stage_idx,
};
use physio_algo::sleep::features::extract;
use physio_algo::sleep::metrics::{confusion4, kappa4, paired_bar};
use physio_algo::sleep::{
    decode_v2, emission_terms, emissions_v2, epoch_starts_v2, flatten_rr, params::Params, prepare_v2,
    weights_of, AccelSample, HrSample, RrRun, SleepInput, Terms, WEIGHT_NAMES,
};
use physio_algo::stats::{least_squares_line, mean, median, pearson, population_sd};

const EPOCH: i64 = 30;
/// Train on the first two, report once on the third. sleep-accel is tanv1's own held-out cohort.
const TRAIN: [&str; 2] = ["dreamt", "aauwss"];
const HELD_OUT: &str = "sleep-accel";
/// The emission rows, in `STAGE_ORDER`.
const DEEP: usize = 0;
const REM: usize = 1;
const AWAKE: usize = 3;
/// Fewest beats in a window before a beat-derived HRV is quoted for that epoch.
const MIN_BEATS: usize = 20;
/// Fewest scored epochs before a night contributes a kappa.
const MIN_EPOCHS: usize = 20;
/// Floors to try on the night motion scale, in g. Selected on the training cohorts only.
const SCALE_FLOORS: [f64; 4] = [1e-5, 1e-4, 5e-4, 1e-3];
/// tanv1's own night-scale rule, mirrored here so an arm can read it: p75 of the consecutive-second
/// deltas, floored, and not trusted below this many deltas.
const TAN_FLOOR_G: f64 = 0.01;
const TAN_MIN_DELTAS: usize = 120;

fn slot(name: &str) -> usize {
    WEIGHT_NAMES.iter().position(|n| *n == name).expect("a named emission weight")
}

/// v2's per-night z-scorer, replicated: population sd, a flat channel neutral, a missing value 0.
struct Z {
    m: f64,
    sd: f64,
    empty: bool,
}

impl Z {
    fn build(v: &[Option<f64>]) -> Z {
        let p: Vec<f64> = v.iter().flatten().copied().collect();
        if p.is_empty() {
            return Z { m: 0.0, sd: 1.0, empty: true };
        }
        let sd = population_sd(&p);
        Z { m: mean(&p), sd: if sd == 0.0 { 1.0 } else { sd }, empty: false }
    }
    fn apply(&self, v: Option<f64>) -> f64 {
        match v {
            Some(x) if !self.empty => (x - self.m) / self.sd,
            _ => 0.0,
        }
    }
}

/// Population sd of the per-second heart rates in `[lo, hi)`, the statistic v2's HR windows read.
fn sd_seconds(sec: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> Option<f64> {
    let v: Vec<f64> = sec.range(lo..hi).map(|(_, b)| *b).collect();
    (v.len() >= 2).then(|| population_sd(&v))
}

/// Within-night percentile rank, `bisect_right / n` over the present values — v2's deep-gate transform.
fn rank_of(sorted: &[f64], v: Option<f64>) -> f64 {
    match v {
        Some(x) if !sorted.is_empty() => sorted.partition_point(|s| *s <= x) as f64 / sorted.len() as f64,
        _ => 0.5,
    }
}

fn sec_grav_of(a: &[AccelSample]) -> BTreeMap<i64, [f64; 3]> {
    let mut acc: BTreeMap<i64, ([f64; 3], f64)> = BTreeMap::new();
    for g in a {
        let e = acc.entry(g.ts).or_insert(([0.0; 3], 0.0));
        e.0[0] += g.x;
        e.0[1] += g.y;
        e.0[2] += g.z;
        e.1 += 1.0;
    }
    acc.into_iter().map(|(t, (s, c))| (t, [s[0] / c, s[1] / c, s[2] / c])).collect()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

fn pct(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

/// Rank AUC of `score` against a boolean label, ties averaged. Weight-free, so it compares two
/// computations of one quantity without either arm's fitted coefficients in the way.
fn auc(score: &[f64], positive: &[bool]) -> f64 {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|a, b| score[*a].total_cmp(&score[*b]));
    let mut rank = vec![0.0f64; score.len()];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in &idx[i..=j] {
            rank[*k] = r;
        }
        i = j + 1;
    }
    let np = positive.iter().filter(|p| **p).count() as f64;
    let nn = positive.len() as f64 - np;
    if np <= 0.0 || nn <= 0.0 {
        return f64::NAN;
    }
    let s: f64 = rank.iter().zip(positive).filter(|(_, p)| **p).map(|(r, _)| *r).sum();
    (s - np * (np + 1.0) / 2.0) / (np * nn)
}

/// One v2 epoch, recomputed. `gap_pairs` counts the deltas v2 forms across a dropout, which
/// `features::deltas` refuses to form; `mean_consec` is the same epoch's parameter-free motion.
struct V2Feat {
    start: i64,
    hr: Option<f64>,
    hr_var: Option<f64>,
    flat: Option<f64>,
    move_frac: Option<f64>,
    mean_consec: Option<f64>,
    jerks: Vec<f64>,
    jerk_max: f64,
    n_hr_epoch: usize,
    n_hr_var: usize,
    n_hr_flat: usize,
    n_grav: usize,
    gap_pairs: usize,
}

/// A night's v2 features and the night-level scale they share, recomputed from the raw streams.
struct V2Night {
    f: Vec<V2Feat>,
    jerk_scale: f64,
}

/// v2's own extraction, replicated: epochs on the absolute 30 s grid, per-second means, deltas
/// between successive PRESENT seconds whether or not they are adjacent, one night-median scale.
fn v2_feats(start: i64, end: i64, hr: &[HrSample], accel: &[AccelSample], p: &Params) -> V2Night {
    let sec_hr = per_second_hr(hr);
    let sec_g = sec_grav_of(accel);

    let mut raws: Vec<V2Feat> = Vec::new();
    let mut all = Vec::new();
    let mut e = start.div_euclid(EPOCH) * EPOCH;
    while e < start {
        e += EPOCH;
    }
    while e < end {
        let secs: Vec<i64> = (e..e + EPOCH).filter(|s| sec_g.contains_key(s)).collect();
        let hrs: Vec<f64> = (e..e + EPOCH).filter_map(|s| sec_hr.get(&s).copied()).collect();
        if hrs.is_empty() && secs.is_empty() {
            e += EPOCH;
            continue;
        }
        let (mut jerks, mut consec, mut gaps) = (Vec::new(), Vec::new(), 0usize);
        for w in secs.windows(2) {
            let d = dist(sec_g[&w[0]], sec_g[&w[1]]);
            jerks.push(d);
            if w[1] - w[0] == 1 {
                consec.push(d);
            } else {
                gaps += 1;
            }
        }
        all.extend_from_slice(&jerks);
        raws.push(V2Feat {
            start: e,
            hr: (!hrs.is_empty()).then(|| mean(&hrs)),
            hr_var: sd_seconds(&sec_hr, e - 150, e + EPOCH + 150),
            flat: sd_seconds(&sec_hr, e - 330, e + EPOCH + 360),
            move_frac: None,
            mean_consec: (!consec.is_empty()).then(|| mean(&consec)),
            jerk_max: jerks.iter().copied().fold(0.0, f64::max),
            jerks,
            n_hr_epoch: hrs.len(),
            n_hr_var: sec_hr.range(e - 150..e + EPOCH + 150).count(),
            n_hr_flat: sec_hr.range(e - 330..e + EPOCH + 360).count(),
            n_grav: secs.len(),
            gap_pairs: gaps,
        });
        e += EPOCH;
    }

    let jerk_scale = if all.is_empty() { 1e-6 } else { median(&all) };
    let thr = jerk_scale * p.jerk_move_mult;
    for r in raws.iter_mut() {
        let denom = (r.n_grav as i64 - 1).max(1) as f64;
        r.move_frac =
            (!r.jerks.is_empty()).then(|| r.jerks.iter().filter(|j| **j > thr).count() as f64 / denom);
    }
    V2Night { f: raws, jerk_scale }
}

/// tanv1's night motion scale over `[start, end)`: p75 of the consecutive-second deltas, floored,
/// and not trusted below [`TAN_MIN_DELTAS`]. Mirrors `features::extract`.
fn tan_scale(accel: &[AccelSample], start: i64, end: i64) -> f64 {
    let sec_g = sec_grav_of(accel);
    let keys: Vec<i64> = sec_g.range(start..end).map(|(t, _)| *t).collect();
    let mut v: Vec<f64> = keys
        .windows(2)
        .filter(|w| w[1] - w[0] == 1)
        .map(|w| dist(sec_g[&w[0]], sec_g[&w[1]]))
        .collect();
    v.sort_by(f64::total_cmp);
    let p75 = if v.len() < TAN_MIN_DELTAS { 0.0 } else { v[v.len() * 3 / 4] };
    p75.max(TAN_FLOOR_G)
}

/// A loaded night with both arms' features on the same epoch grid, plus the shipped decomposition.
struct Night {
    w0: i64,
    w1: i64,
    n: usize,
    hr: Vec<HrSample>,
    rr: Vec<RrRun>,
    accel: Vec<AccelSample>,
    v2: V2Night,
    terms: Terms,
    em_v2: Vec<[f64; 4]>,
    card: Vec<physio_algo::sleep::features::Cardiac>,
    motion_frac_30: Vec<Option<f64>>,
    motion_max_30: Vec<Option<f64>>,
    tan_scale: f64,
    truth: Vec<Option<usize>>,
}

fn load(set: &str) -> Vec<Night> {
    let mut out = Vec::new();
    for dir in &dirs_of(set) {
        let Some((w0, w1, n)) = read_meta(dir) else { continue };
        let accel = read_accel(dir);
        if accel.is_empty() || n < MIN_EPOCHS {
            continue;
        }
        let (hr, rr) = (read_hr(dir), read_rr(dir));
        let raw = read_truth(dir);
        let card = cardiac_series(w0, n, EPOCH, &hr, &rr);
        let feats = extract(&accel, w0, w1, &card);
        let input =
            SleepInput { start: w0, end: w1, hr: hr.clone(), rr: rr.clone(), accel: accel.clone() };
        let prep = prepare_v2(&input, &Params::SHIPPED);
        let terms = emission_terms(&prep, &Params::SHIPPED);
        let em_v2 = emissions_v2(&prep, &Params::SHIPPED);
        let v2 = v2_feats(w0, w1, &hr, &accel, &Params::SHIPPED);
        assert_eq!(
            epoch_starts_v2(&prep),
            v2.f.iter().map(|f| f.start).collect::<Vec<_>>(),
            "{}: the replicated epoch grid must be v2's own",
            dir.display()
        );
        let truth = v2
            .f
            .iter()
            .map(|f| {
                let k = ((f.start - w0) / EPOCH) as usize;
                raw.get(&k).copied().filter(|t| (0..4).contains(t)).map(|t| t as usize)
            })
            .collect();
        out.push(Night {
            w0,
            w1,
            n,
            tan_scale: tan_scale(&accel, w0, w1),
            hr,
            rr,
            accel,
            v2,
            terms,
            em_v2,
            card,
            motion_frac_30: feats.iter().map(|f| f.motion_frac[0]).collect(),
            motion_max_30: feats.iter().map(|f| f.motion_max[0]).collect(),
            truth,
        });
    }
    out
}

/// Largest absolute difference and how many pairs exceed a tolerance, for a pinning check.
fn worst(a: &[f64], b: &[f64], tol: f64) -> (f64, usize) {
    let (mut w, mut over) = (0.0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        w = w.max(d);
        over += usize::from(d > tol);
    }
    (w, over)
}

/// tanv1's motion columns read at the epochs v2 actually emitted, so both arms index one grid.
fn tan_at_v2_epochs(nt: &Night, col: &[Option<f64>]) -> Vec<Option<f64>> {
    nt.v2
        .f
        .iter()
        .map(|f| {
            let k = ((f.start - nt.w0) / EPOCH) as usize;
            if k < nt.n { col.get(k).copied().flatten() } else { None }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// sections
// ---------------------------------------------------------------------------------------------

/// The replication is the basis of everything below, so it is pinned against the shipped
/// emission's own design columns before a single number is read off it.
fn section0(sets: &[(&str, Vec<Night>)]) {
    println!("0  the replication, pinned against the shipped emission's design columns");
    println!(
        "   {:<14} {:>8} {:>11} {:>11} {:>11} {:>11} {:>10}",
        "cohort", "epochs", "hr z", "hr_var z", "motion z", "gate hinge", "hinge!="
    );
    for (name, nights) in sets {
        let (mut e, mut wh, mut wv, mut wm, mut wg, mut nh) = (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        for nt in nights {
            let zhr = Z::build(&nt.v2.f.iter().map(|f| f.hr).collect::<Vec<_>>());
            let zhv = Z::build(&nt.v2.f.iter().map(|f| f.hr_var).collect::<Vec<_>>());
            let zmv = Z::build(&nt.v2.f.iter().map(|f| f.move_frac).collect::<Vec<_>>());
            let mut fs: Vec<f64> = nt.v2.f.iter().filter_map(|f| f.flat).collect();
            fs.sort_by(f64::total_cmp);
            let mut cols: [(Vec<f64>, Vec<f64>); 4] = Default::default();
            for (k, f) in nt.v2.f.iter().enumerate() {
                let row = &nt.terms.design[k][DEEP];
                cols[0].0.push(zhr.apply(f.hr));
                cols[0].1.push(row[slot("deep_hr")]);
                cols[1].0.push(zhv.apply(f.hr_var));
                cols[1].1.push(row[slot("deep_hrv")]);
                cols[2].0.push(zmv.apply(f.move_frac));
                cols[2].1.push(row[slot("deep_motion")]);
                cols[3].0.push((rank_of(&fs, f.flat) - Params::SHIPPED.deep_gate_thresh).max(0.0));
                cols[3].1.push(-row[slot("deep_gate_slope")]);
            }
            e += nt.v2.f.len();
            wh = wh.max(worst(&cols[0].0, &cols[0].1, 0.0).0);
            wv = wv.max(worst(&cols[1].0, &cols[1].1, 0.0).0);
            wm = wm.max(worst(&cols[2].0, &cols[2].1, 0.0).0);
            let (g, over) = worst(&cols[3].0, &cols[3].1, 1e-12);
            wg = wg.max(g);
            nh += over;
        }
        println!("   {name:<14} {e:>8} {wh:>11.1e} {wv:>11.1e} {wm:>11.1e} {wg:>11.1e} {:>9.2}%", pct(nh, e));
    }
    println!("   Worst absolute difference per column, and the share of epochs whose deep-gate hinge");
    println!("   moves at all. hr and motion are bit-identical; hr_var differs in the last bits");
    println!("   because v2 sums squares over a prefix axis and this is a two-pass sd. That last bit");
    println!("   is enough to reorder a TIE BLOCK in the gate's `<=` rank, which is why the hinge");
    println!("   column is larger than the sd column that feeds it.\n");
}

/// The grid, the coverage each window actually gets, and where an absent sample becomes a number.
fn section1(sets: &[(&str, Vec<Night>)]) {
    println!("1  grid, coverage, and where absence becomes a value");
    println!(
        "   {:<14} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "cohort", "nights", "w0%30", "grav%", "hr%", "rr%", "no hr", "hrvar<2", "no grav"
    );
    for (name, nights) in sets {
        let (mut off, mut gs, mut hs, mut rs, mut span) = (0usize, 0i64, 0i64, 0i64, 0i64);
        let (mut ep, mut nohr, mut varshort, mut nograv) = (0usize, 0usize, 0usize, 0usize);
        for nt in nights {
            off += usize::from(nt.w0.rem_euclid(EPOCH) != 0);
            span += nt.w1 - nt.w0;
            gs += sec_grav_of(&nt.accel).range(nt.w0..nt.w1).count() as i64;
            hs += per_second_hr(&nt.hr).range(nt.w0..nt.w1).count() as i64;
            rs += nt.rr.iter().filter(|r| r.ts >= nt.w0 && r.ts < nt.w1).count() as i64;
            for f in &nt.v2.f {
                ep += 1;
                nohr += usize::from(f.n_hr_epoch == 0);
                varshort += usize::from(f.n_hr_var < 2);
                nograv += usize::from(f.move_frac.is_none());
            }
        }
        println!(
            "   {name:<14} {:>6} {off:>7} {:>6.1}% {:>6.1}% {:>6.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            nights.len(),
            100.0 * gs as f64 / span as f64,
            100.0 * hs as f64 / span as f64,
            100.0 * rs as f64 / span as f64,
            pct(nohr, ep),
            pct(varshort, ep),
            pct(nograv, ep)
        );
    }
    println!("   `no hr` / `no grav` / `hrvar<2` epochs score the NEUTRAL 0 after z-scoring, which is");
    println!("   the population mean of that channel - not a missing value the decoder can see.\n");

    println!("   HR-window occupancy: how many of the window's seconds actually carry a sample");
    println!(
        "   {:<14} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "cohort", "330s med", "330s p05", "720s med", "720s p05", "720s<50%"
    );
    for (name, nights) in sets {
        let (mut v, mut fl) = (Vec::new(), Vec::new());
        let (mut edge, mut ep) = (0usize, 0usize);
        for nt in nights {
            for f in &nt.v2.f {
                v.push(f.n_hr_var as f64);
                fl.push(f.n_hr_flat as f64);
                ep += 1;
                edge += usize::from(f.n_hr_flat * 2 < 720);
            }
        }
        v.sort_by(f64::total_cmp);
        fl.sort_by(f64::total_cmp);
        println!(
            "   {name:<14} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>10.1}%",
            quantile(&v, 0.5),
            quantile(&v, 0.05),
            quantile(&fl, 0.5),
            quantile(&fl, 0.05),
            pct(edge, ep)
        );
    }
    println!("   Nothing downstream records how full a window was. A 63-sample sd and a 720-sample");
    println!("   one are the same input to the z-score and to the deep-gate rank, and the cohorts");
    println!("   differ by a factor of five in cadence.\n");

    println!("   the night motion scale, and whether it degenerates");
    println!(
        "   {:<14} {:>12} {:>12} {:>12} {:>10} {:>11}",
        "cohort", "median jerk", "p05", "p95", "scale=0", "gap deltas"
    );
    for (name, nights) in sets {
        let mut s: Vec<f64> = nights.iter().map(|n| n.v2.jerk_scale).collect();
        s.sort_by(f64::total_cmp);
        let zero = s.iter().filter(|x| **x <= 0.0).count();
        let gaps: usize = nights.iter().flat_map(|n| n.v2.f.iter()).map(|f| f.gap_pairs).sum();
        println!(
            "   {name:<14} {:>12.5} {:>12.5} {:>12.5} {zero:>10} {gaps:>11}",
            quantile(&s, 0.5),
            quantile(&s, 0.05),
            quantile(&s, 0.95)
        );
    }
    // The same count on the real device stream, where dropouts are not repaired away.
    let (mut cg, mut ct, mut cb) = (0usize, 0usize, 0usize);
    for d in dirs_of("continuous") {
        let a = read_accel(&d);
        if a.len() < 120 {
            continue;
        }
        cb += 1;
        let sec_g = sec_grav_of(&a);
        let keys: Vec<i64> = sec_g.keys().copied().collect();
        let (lo, hi) = (keys[0], keys[keys.len() - 1] + 1);
        let mut e = lo.div_euclid(EPOCH) * EPOCH;
        while e < hi {
            let secs: Vec<i64> = sec_g.range(e..e + EPOCH).map(|(t, _)| *t).collect();
            for w in secs.windows(2) {
                ct += 1;
                cg += usize::from(w[1] - w[0] != 1);
            }
            e += EPOCH;
        }
    }
    println!(
        "   continuous ({cb} real wear blocks): {cg} of {ct} within-epoch deltas ({:.2}%) span a",
        pct(cg, ct)
    );
    println!("   dropout. v2 forms them anyway and calls the whole gap one second of movement;");
    println!("   `features::deltas` refuses to form them. The PSG fixtures carry a gravity sample");
    println!("   every second, so that difference cannot be scored on any labelled cohort.\n");
}

/// The same physical quantity, computed twice.
fn section2(sets: &[(&str, Vec<Night>)]) {
    println!("2  the same quantity, two computations");
    println!(
        "   {:<14} {:<14} {:>8} {:>10} {:>12} {:>9}",
        "cohort", "quantity", "pairs", "pearson r", "worst |diff|", "differ"
    );
    for (name, nights) in sets {
        let labels = ["hr_z", "hr_var_z", "hr_flat_pct", "motion (z)"];
        let mut rows: Vec<(Vec<f64>, Vec<f64>)> = vec![Default::default(); 4];
        for nt in nights {
            let zhr = Z::build(&nt.v2.f.iter().map(|f| f.hr).collect::<Vec<_>>());
            let zhv = Z::build(&nt.v2.f.iter().map(|f| f.hr_var).collect::<Vec<_>>());
            let zmv = Z::build(&nt.v2.f.iter().map(|f| f.move_frac).collect::<Vec<_>>());
            let tan = tan_at_v2_epochs(nt, &nt.motion_frac_30);
            let ztan = Z::build(&tan);
            let mut fs: Vec<f64> = nt.v2.f.iter().filter_map(|f| f.flat).collect();
            fs.sort_by(f64::total_cmp);
            for (i, f) in nt.v2.f.iter().enumerate() {
                let k = ((f.start - nt.w0) / EPOCH) as usize;
                if k >= nt.n {
                    continue;
                }
                let c = nt.card[k];
                if let Some(v) = c.hr_z {
                    rows[0].0.push(zhr.apply(f.hr));
                    rows[0].1.push(v);
                }
                if let Some(v) = c.hr_var_z {
                    rows[1].0.push(zhv.apply(f.hr_var));
                    rows[1].1.push(v);
                }
                if let Some(v) = c.hr_flat_pct {
                    rows[2].0.push(rank_of(&fs, f.flat));
                    rows[2].1.push(v);
                }
                if tan[i].is_some() {
                    rows[3].0.push(zmv.apply(f.move_frac));
                    rows[3].1.push(ztan.apply(tan[i]));
                }
            }
        }
        for (label, (a, b)) in labels.iter().zip(&rows) {
            let (w, over) = worst(a, b, 1e-6);
            println!(
                "   {name:<14} {label:<14} {:>8} {:>10.4} {:>12.1e} {:>8.1}%",
                a.len(),
                pearson(a, b).unwrap_or(f64::NAN),
                w,
                pct(over, a.len())
            );
        }
    }
    println!("   `differ` is the share of paired epochs more than 1e-6 apart. Three of the four");
    println!("   columns are the same computation reached twice. Motion is not.\n");

    println!("   the motion pair: v2 counts deltas over the night MEDIAN x 75, tanv1 over the night");
    println!("   p75 floored at 0.01 g. Whether the epoch moved at all:");
    println!(
        "   {:<14} {:>10} {:>11} {:>11} {:>11} {:>11}",
        "cohort", "epochs", "v2 moved%", "tanv1 mv%", "both", "disagree"
    );
    for (name, nights) in sets {
        let (mut ep, mut a, mut b, mut both, mut dis) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for nt in nights {
            let tan = tan_at_v2_epochs(nt, &nt.motion_frac_30);
            for (i, f) in nt.v2.f.iter().enumerate() {
                let (Some(t), Some(v)) = (tan[i], f.move_frac) else { continue };
                ep += 1;
                let (x, y) = (v > 0.0, t > 0.0);
                a += usize::from(x);
                b += usize::from(y);
                both += usize::from(x && y);
                dis += usize::from(x != y);
            }
        }
        println!(
            "   {name:<14} {ep:>10} {:>10.1}% {:>10.1}% {:>10.1}% {:>10.1}%",
            pct(a, ep),
            pct(b, ep),
            pct(both, ep),
            pct(dis, ep)
        );
    }

    println!("\n   which of the two is right, with no weights in the way: rank AUC for wake against");
    println!("   sleep, over the labelled epochs of each cohort");
    println!(
        "   {:<14} {:>10} {:>12} {:>12} {:>12}",
        "cohort", "epochs", "v2 frac", "tanv1 frac", "mean |dg|"
    );
    for (name, nights) in sets {
        let (mut a, mut b, mut c, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for nt in nights {
            let tan = tan_at_v2_epochs(nt, &nt.motion_frac_30);
            for (i, f) in nt.v2.f.iter().enumerate() {
                let Some(t) = nt.truth[i] else { continue };
                a.push(f.move_frac.unwrap_or(0.0));
                b.push(tan[i].unwrap_or(0.0));
                c.push(f.mean_consec.unwrap_or(0.0));
                y.push(t == 0);
            }
        }
        println!(
            "   {name:<14} {:>10} {:>12.4} {:>12.4} {:>12.4}",
            a.len(),
            auc(&a, &y),
            auc(&b, &y),
            auc(&c, &y)
        );
    }
    println!();
}

/// What `hr_var` is a statistic of: the spread of PER-SECOND heart rate over 330 s. The question is
/// whether it tracks beat-to-beat variability or the slow trend across the window.
fn section3(sets: &[(&str, Vec<Night>)]) {
    println!("3  is `hr_var` an HRV estimator?");
    println!(
        "   {:<14} {:>8} {:>11} {:>11} {:>11} {:>11} {:>9}",
        "cohort", "epochs", "r ln RMSSD", "r ln SDNN", "r trend sd", "trend share", "no beats"
    );
    for (name, nights) in sets {
        let (mut hv, mut rm, mut sd, mut tr, mut share) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut ep, mut nobeat) = (0usize, 0usize);
        for nt in nights {
            let sec = per_second_hr(&nt.hr);
            let mut by: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
            for (ts, ms) in flatten_rr(&nt.rr) {
                by.entry(ts).or_default().push(ms.clamp(300.0, 2000.0));
            }
            for f in &nt.v2.f {
                let Some(v) = f.hr_var else { continue };
                ep += 1;
                // The window's linear trend, and what is left once it is removed.
                let secs: Vec<f64> =
                    sec.range(f.start - 150..f.start + EPOCH + 150).map(|(_, b)| *b).collect();
                let (m, c) = least_squares_line(&secs);
                let fit: Vec<f64> = (0..secs.len()).map(|i| c + m * i as f64).collect();
                if v > 0.0 {
                    share.push(population_sd(&fit) / v);
                }
                let beats: Vec<f64> = by
                    .range(f.start - 150..f.start + EPOCH + 150)
                    .flat_map(|(_, x)| x.iter().copied())
                    .collect();
                if beats.len() < MIN_BEATS {
                    nobeat += 1;
                    continue;
                }
                let diffs: Vec<f64> = beats.windows(2).map(|w| (w[1] - w[0]).powi(2)).collect();
                hv.push(v);
                rm.push(mean(&diffs).sqrt().max(1e-6).ln());
                sd.push(population_sd(&beats).max(1e-6).ln());
                tr.push(population_sd(&fit));
            }
        }
        let r = |a: &[f64]| pearson(&hv, a).unwrap_or(f64::NAN);
        println!(
            "   {name:<14} {:>8} {:>11.4} {:>11.4} {:>11.4} {:>11.3} {:>8.1}%",
            hv.len(),
            r(&rm),
            r(&sd),
            r(&tr),
            if share.is_empty() { f64::NAN } else { median(&share) },
            pct(nobeat, ep)
        );
    }
    println!("   `trend share` is the median of sd(linear fit) / hr_var over the same 330 s window:");
    println!("   how much of the spread the channel reports is the window's slow drift rather than");
    println!("   anything beat-to-beat. `no beats` is the share of epochs carrying an hr_var with");
    println!("   too few beats behind them to have an HRV at all.\n");
}

/// Our motion proxy against the strap's own on-chip ENMO, on the blocks that carry both.
fn section4() {
    println!("4  our |delta gravity| proxy against the strap's own ENMO");
    println!(
        "   {:<46} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "block", "sec", "r (1 Hz)", "r (epoch)", "med ours", "med ENMO"
    );
    let (mut po, mut pe, mut quiet) = (Vec::new(), Vec::new(), Vec::new());
    let (mut top_hit, mut top_n) = (0usize, 0usize);
    let mut any = false;
    for d in dirs_of("continuous") {
        let dyn_g = read_dyn_accel(&d);
        if dyn_g.is_empty() {
            continue;
        }
        any = true;
        let sec_g = sec_grav_of(&read_accel(&d));
        let dmap: BTreeMap<i64, f64> = dyn_g.iter().copied().collect();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let mut bucket: BTreeMap<i64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for (t, v) in &dmap {
            let (Some(prev), Some(cur)) = (sec_g.get(&(t - 1)), sec_g.get(t)) else { continue };
            let j = dist(*prev, *cur);
            a.push(j);
            b.push(*v);
            let e = bucket.entry(t.div_euclid(EPOCH)).or_default();
            e.0.push(j);
            e.1.push(*v);
        }
        let (mut ea, mut eb) = (Vec::new(), Vec::new());
        for (x, y) in bucket.values() {
            ea.push(mean(x));
            eb.push(mean(y));
        }
        // The strap's own quietest tenth of epochs: what our channel reads where it says still.
        let mut sorted = eb.clone();
        sorted.sort_by(f64::total_cmp);
        let floor = quantile(&sorted, 0.10);
        for (x, y) in ea.iter().zip(&eb) {
            if *y <= floor {
                quiet.push(*x / mean(&ea).max(1e-12));
            }
        }
        // Do the two series pick out the SAME loudest tenth of epochs?
        let mut oa = ea.clone();
        oa.sort_by(f64::total_cmp);
        let (ta, tb) = (quantile(&oa, 0.90), quantile(&sorted, 0.90));
        for (x, y) in ea.iter().zip(&eb) {
            if *y > tb {
                top_n += 1;
                top_hit += usize::from(*x > ta);
            }
        }
        println!(
            "   {:<46} {:>8} {:>10.4} {:>10.4} {:>10.5} {:>10.5}",
            d.file_name().unwrap_or_default().to_string_lossy(),
            a.len(),
            pearson(&a, &b).unwrap_or(f64::NAN),
            pearson(&ea, &eb).unwrap_or(f64::NAN),
            median(&a),
            median(&b)
        );
        po.extend(a);
        pe.extend(b);
    }
    if !any {
        println!("   no `continuous` block carries dynaccel.csv under this fixture root\n");
        return;
    }
    println!(
        "\n   pooled 1 Hz: n={} r={:.4}. Median ours {:.5} g against the strap's {:.5} g.",
        po.len(),
        pearson(&po, &pe).unwrap_or(f64::NAN),
        median(&po),
        median(&pe)
    );
    println!(
        "   On the strap's own quietest tenth of epochs our channel still reads {:.0}% of its own",
        100.0 * median(&quiet)
    );
    println!("   block mean, so the two series disagree most where the strap says nothing happened.");
    println!(
        "   Of the {top_n} epochs in the strap's loudest tenth, {:.0}% are in ours as well.",
        pct(top_hit, top_n)
    );
    println!("   They are not the same physical quantity: a gravity difference is a ROTATION rate,");
    println!("   ENMO is the magnitude of linear acceleration. `motion_channel` swaps the channel");
    println!("   and scores the staging; this row says how far apart the inputs are.\n");
}

// ---------------------------------------------------------------------------------------------
// the arms
// ---------------------------------------------------------------------------------------------

/// `src` values carried onto `dst`'s marginal distribution by rank, so a substitution changes the
/// ORDERING of the epochs and nothing about the shape the shipped weight was tuned against.
fn quantile_map(src: &[Option<f64>], dst: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut s: Vec<f64> = src.iter().flatten().copied().collect();
    let mut d: Vec<f64> = dst.iter().flatten().copied().collect();
    s.sort_by(f64::total_cmp);
    d.sort_by(f64::total_cmp);
    src.iter()
        .map(|o| {
            o.and_then(|x| {
                if s.is_empty() || d.is_empty() {
                    return None;
                }
                let lo = s.partition_point(|v| *v < x) as f64;
                let hi = s.partition_point(|v| *v <= x) as f64;
                let q = (lo + hi) / 2.0 / s.len() as f64;
                Some(d[((q * (d.len() - 1) as f64).round() as usize).min(d.len() - 1)])
            })
        })
        .collect()
}

/// One epoch's Step-1 override. `None` leaves v2's own computation in place, so an all-`None`
/// override must reproduce the shipped emission exactly.
#[derive(Default, Clone, Copy)]
struct Over {
    motion_z: Option<f64>,
    hrv_z: Option<f64>,
    quiescent: Option<bool>,
    boost: Option<bool>,
    /// The deep-gate hinge, already thresholded.
    hinge: Option<f64>,
}

/// Which Step-1 computation an arm replaces.
#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// The replication itself, as a positive control.
    Replica,
    /// v2's motion feature replaced by tanv1's `motion_frac_30`, gates untouched.
    TanFrac,
    /// The same, with the stillness clamp and the wake gate moved onto tanv1's night scale.
    TanFracGated,
    /// tanv1's ORDERING of the epochs carried onto v2's own marginal distribution, so the shipped
    /// motion weight meets the shape it was tuned against and only the ranking changes.
    TanFracMatched,
    /// The thresholded fraction replaced by the epoch's mean consecutive-second delta.
    MeanJerk,
    /// v2's own recipe with the night median jerk floored, so a degenerate scale cannot collapse.
    ScaleFloor(f64),
    /// `hr_var` replaced by the log RMSSD of the beats in the same 330 s window.
    BeatHrv,
    /// The same with the sign flipped, the cheapest control on the shipped weights' polarity.
    BeatHrvNeg,
    /// The RMSSD ORDERING carried onto `hr_var`'s own marginal, the shape control.
    BeatHrvMatched,
    /// The deep gate's 720 s HR window re-centred on the epoch midpoint instead of 15 s past it.
    FlatCentred,
}

fn arm_name(a: Arm) -> String {
    match a {
        Arm::Replica => "replica (control)".into(),
        Arm::TanFrac => "motion <- tanv1 frac".into(),
        Arm::TanFracGated => "motion <- tanv1 + gates".into(),
        Arm::TanFracMatched => "motion <- tanv1 rank only".into(),
        Arm::MeanJerk => "motion <- mean |dg|".into(),
        Arm::ScaleFloor(f) => format!("scale floored at {f:.0e}"),
        Arm::BeatHrv => "hr_var <- ln RMSSD".into(),
        Arm::BeatHrvNeg => "hr_var <- -ln RMSSD".into(),
        Arm::BeatHrvMatched => "hr_var <- RMSSD rank only".into(),
        Arm::FlatCentred => "deep gate window centred".into(),
    }
}

/// The per-epoch overrides one arm asks for, on one night.
fn overrides(nt: &Night, arm: Arm) -> Vec<Over> {
    let n = nt.v2.f.len();
    let mut out = vec![Over::default(); n];
    let p = Params::SHIPPED;
    match arm {
        Arm::Replica => {}
        Arm::TanFrac | Arm::TanFracGated => {
            let frac = tan_at_v2_epochs(nt, &nt.motion_frac_30);
            let peak = tan_at_v2_epochs(nt, &nt.motion_max_30);
            let z = Z::build(&frac);
            for (i, o) in out.iter_mut().enumerate() {
                o.motion_z = Some(z.apply(frac[i]));
                if arm == Arm::TanFracGated {
                    let gate = nt.tan_scale * p.jerk_gate_mult;
                    o.quiescent =
                        Some(frac[i].is_some_and(|m| m <= 0.0) && peak[i].unwrap_or(0.0) <= gate);
                    o.boost = Some(peak[i].unwrap_or(0.0) > gate);
                }
            }
        }
        Arm::TanFracMatched => {
            let frac = tan_at_v2_epochs(nt, &nt.motion_frac_30);
            let own: Vec<Option<f64>> = nt.v2.f.iter().map(|f| f.move_frac).collect();
            let mapped = quantile_map(&frac, &own);
            let z = Z::build(&mapped);
            for (i, o) in out.iter_mut().enumerate() {
                o.motion_z = Some(z.apply(mapped[i]));
            }
        }
        Arm::MeanJerk => {
            let v: Vec<Option<f64>> = nt.v2.f.iter().map(|f| f.mean_consec).collect();
            let z = Z::build(&v);
            for (i, o) in out.iter_mut().enumerate() {
                o.motion_z = Some(z.apply(v[i]));
            }
        }
        Arm::ScaleFloor(floor) => {
            let scale = nt.v2.jerk_scale.max(floor);
            let (thr, gate) = (scale * p.jerk_move_mult, scale * p.jerk_gate_mult);
            let v: Vec<Option<f64>> = nt
                .v2
                .f
                .iter()
                .map(|f| {
                    let denom = (f.n_grav as i64 - 1).max(1) as f64;
                    (!f.jerks.is_empty())
                        .then(|| f.jerks.iter().filter(|j| **j > thr).count() as f64 / denom)
                })
                .collect();
            let z = Z::build(&v);
            for (i, o) in out.iter_mut().enumerate() {
                let f = &nt.v2.f[i];
                o.motion_z = Some(z.apply(v[i]));
                o.quiescent = Some(v[i].is_some_and(|m| m <= 0.0) && f.jerk_max <= gate);
                o.boost = Some(f.jerk_max > gate);
            }
        }
        Arm::FlatCentred => {
            let sec = per_second_hr(&nt.hr);
            let v: Vec<Option<f64>> = nt
                .v2
                .f
                .iter()
                .map(|f| sd_seconds(&sec, f.start - 345, f.start + EPOCH + 345))
                .collect();
            let mut s: Vec<f64> = v.iter().flatten().copied().collect();
            s.sort_by(f64::total_cmp);
            for (i, o) in out.iter_mut().enumerate() {
                o.hinge = Some((rank_of(&s, v[i]) - p.deep_gate_thresh).max(0.0));
            }
        }
        Arm::BeatHrv | Arm::BeatHrvNeg | Arm::BeatHrvMatched => {
            let sign = if arm == Arm::BeatHrvNeg { -1.0 } else { 1.0 };
            let mut by: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
            for (ts, ms) in flatten_rr(&nt.rr) {
                by.entry(ts).or_default().push(ms.clamp(300.0, 2000.0));
            }
            let v: Vec<Option<f64>> = nt
                .v2
                .f
                .iter()
                .map(|f| {
                    let b: Vec<f64> = by
                        .range(f.start - 150..f.start + EPOCH + 150)
                        .flat_map(|(_, x)| x.iter().copied())
                        .collect();
                    (b.len() >= MIN_BEATS).then(|| {
                        let d: Vec<f64> = b.windows(2).map(|w| (w[1] - w[0]).powi(2)).collect();
                        mean(&d).sqrt().max(1e-6).ln()
                    })
                })
                .collect();
            let v = if arm == Arm::BeatHrvMatched {
                quantile_map(&v, &nt.v2.f.iter().map(|f| f.hr_var).collect::<Vec<_>>())
            } else {
                v
            };
            let z = Z::build(&v);
            for (i, o) in out.iter_mut().enumerate() {
                o.hrv_z = Some(sign * z.apply(v[i]));
            }
        }
    }
    out
}

/// Rebuild a night's emissions with the overrides applied to the shipped decomposition. Everything
/// not overridden - weights, priors, transition, anchor - is the shipped recipe untouched.
fn emit(nt: &Night, over: &[Over]) -> Vec<[f64; 4]> {
    let p = Params::SHIPPED;
    let w = weights_of(&p);
    let (dm, rm, am) = (slot("deep_motion"), slot("rem_motion"), slot("awake_motion"));
    let (dh, rh, ah) = (slot("deep_hrv"), slot("rem_hrv"), slot("awake_hrv"));
    let mut t = Terms {
        design: nt.terms.design.clone(),
        fixed: nt.terms.fixed.clone(),
        clamped: nt.terms.clamped.clone(),
    };
    for (e, o) in over.iter().enumerate() {
        if let Some(z) = o.motion_z {
            t.design[e][DEEP][dm] = z;
            t.design[e][REM][rm] = z;
            t.design[e][AWAKE][am] = z;
        }
        if let Some(z) = o.hrv_z {
            let d = p.awake_deadzone;
            t.design[e][DEEP][dh] = z;
            t.design[e][REM][rh] = z;
            t.design[e][AWAKE][ah] = if z > d {
                z - d
            } else if z < -d {
                z + d
            } else {
                0.0
            };
        }
        if let Some(h) = o.hinge {
            t.design[e][DEEP][slot("deep_gate_slope")] = -h;
        }
        if let Some(q) = o.quiescent {
            t.clamped[e] = q;
        }
        if let Some(b) = o.boost {
            let was = nt.v2.f[e].jerk_max > nt.v2.jerk_scale * p.jerk_gate_mult;
            if b != was {
                t.fixed[e][AWAKE] += if b { p.motion_gate_boost } else { -p.motion_gate_boost };
            }
        }
    }
    (0..t.design.len()).map(|e| t.emission(e, &w)).collect()
}

/// One night's four-class kappa under an emission, or `None` when too little of it is labelled.
fn kappa_of(nt: &Night, em: &[[f64; 4]]) -> Option<f64> {
    let path: Vec<usize> = decode_v2(em, &Params::SHIPPED.transition).iter().map(|s| stage_idx(*s)).collect();
    let (mut pred, mut want) = (Vec::new(), Vec::new());
    for (k, t) in nt.truth.iter().enumerate() {
        if let Some(t) = t {
            pred.push(path[k]);
            want.push(*t);
        }
    }
    (want.len() >= MIN_EPOCHS).then(|| kappa4(&confusion4(&pred, &want)))
}

/// Paired per-night kappa deltas of one arm against the shipped recipe, on one cohort.
fn deltas_of(nights: &[Night], arm: Arm) -> Vec<f64> {
    let mut out = Vec::new();
    for nt in nights {
        let (Some(base), Some(got)) = (kappa_of(nt, &nt.em_v2), kappa_of(nt, &emit(nt, &overrides(nt, arm))))
        else {
            continue;
        };
        out.push(got - base);
    }
    out
}

fn verdict(d: &[f64]) -> String {
    match paired_bar(d) {
        None => "  (n<2)".to_string(),
        Some((m, bar)) => {
            let tag = if m.abs() > bar {
                if m > 0.0 { "BEATS SHIPPED" } else { "worse" }
            } else {
                "matches"
            };
            format!("{m:>+9.4} {bar:>8.4} {:>4}   {tag}", d.len())
        }
    }
}

/// Every arm on every cohort, then the one selection rule applied once.
fn section5(sets: &[(&str, Vec<Night>)]) {
    println!("5  the arms, decoded. Same weights, same priors, same transition, same anchor;");
    println!("   exactly one Step-1 computation replaced, scored per night against truth.\n");

    // The control has to be exact before any arm means anything.
    for (name, nights) in sets {
        for nt in nights {
            let got = emit(nt, &overrides(nt, Arm::Replica));
            assert_eq!(got, nt.em_v2, "{name}: an empty override must be the shipped emission");
            for (e, f) in nt.v2.f.iter().enumerate() {
                let q = f.move_frac.is_some_and(|m| m <= 0.0)
                    && f.jerk_max <= nt.v2.jerk_scale * Params::SHIPPED.jerk_gate_mult;
                assert_eq!(q, nt.terms.clamped[e], "{name}: the replicated stillness clamp differs");
            }
        }
    }
    println!("   CONTROL holds: an empty override reproduces the shipped emission bit for bit, and");
    println!("   the replicated stillness clamp is the shipped one on every epoch.\n");

    let mut arms: Vec<Arm> = vec![
        Arm::TanFrac,
        Arm::TanFracGated,
        Arm::TanFracMatched,
        Arm::MeanJerk,
        Arm::BeatHrv,
        Arm::BeatHrvNeg,
        Arm::BeatHrvMatched,
        Arm::FlatCentred,
    ];
    arms.extend(SCALE_FLOORS.map(Arm::ScaleFloor));

    println!("   {:<26} {:<14} {:>9} {:>8} {:>4}   verdict", "arm", "cohort", "paired d", "bar +/-", "n");
    let mut table: Vec<(Arm, Vec<(String, f64)>)> = Vec::new();
    for arm in &arms {
        let mut per = Vec::new();
        for (name, nights) in sets {
            let d = deltas_of(nights, *arm);
            println!("   {:<26} {:<14} {}", arm_name(*arm), name, verdict(&d));
            if let Some((m, _)) = paired_bar(&d) {
                per.push(((*name).to_string(), m));
            }
        }
        table.push((*arm, per));
    }

    // Selection: an arm must win on BOTH training cohorts, and a swept floor picks its best there.
    println!("\n   SELECTION - inner leave-one-out over the training cohorts {TRAIN:?} only.");
    let train_mean = |per: &[(String, f64)]| -> Option<f64> {
        let v: Vec<f64> = TRAIN.iter().filter_map(|c| per.iter().find(|(n, _)| n == c).map(|(_, m)| *m)).collect();
        (v.len() == TRAIN.len()).then(|| mean(&v))
    };
    let both_positive = |per: &[(String, f64)]| -> bool {
        TRAIN.iter().all(|c| per.iter().any(|(n, m)| n == c && *m > 0.0))
    };
    let mut best: Option<(Arm, f64)> = None;
    for (arm, per) in &table {
        let Some(m) = train_mean(per) else {
            println!("   {:<26} not defined on both training cohorts, so it cannot be selected", arm_name(*arm));
            continue;
        };
        let inert = TRAIN.iter().all(|c| per.iter().any(|(n, v)| n == c && *v == 0.0));
        println!(
            "   {:<26} train mean {m:>+8.4}   {}",
            arm_name(*arm),
            if both_positive(per) {
                "positive on both"
            } else if inert {
                "INERT on both, so nothing here can select it"
            } else {
                "not positive on both"
            }
        );
        if both_positive(per) && best.is_none_or(|(_, b)| m > b) {
            best = Some((*arm, m));
        }
    }
    match best {
        None => {
            println!("\n   No arm is positive on both training cohorts, so nothing is selected and the");
            println!("   held-out cohort answers nothing. The rows above are diagnostics, not results.");
        }
        Some((arm, m)) => {
            println!("\n   SELECTED on the training cohorts: {} (train mean {m:+.4})", arm_name(arm));
            let Some((_, nights)) = sets.iter().find(|(n, _)| *n == HELD_OUT) else { return };
            let d = deltas_of(nights, arm);
            println!("   reported once on {HELD_OUT}: {}", verdict(&d));
        }
    }
    println!("\n   Every arm here is v2 with one feature recomputed, so a win is a V2 finding and not");
    println!("   a tanv1 one: the repaired feature would feed both arms identically.\n");
}

fn main() {
    let sets: Vec<(&str, Vec<Night>)> = TRAIN
        .iter()
        .chain(std::iter::once(&HELD_OUT))
        .map(|c| (*c, load(c)))
        .filter(|(_, n)| !n.is_empty())
        .collect();
    if sets.is_empty() {
        println!("no cohort under the fixture root");
        return;
    }
    println!("Step 1 - the per-epoch feature COMPUTATION, audited against itself, against the");
    println!("strap's own figures, and then decoded. Nothing below changes src.\n");
    section0(&sets);
    section1(&sets);
    section2(&sets);
    section3(&sets);
    section4();
    section5(&sets);
}
