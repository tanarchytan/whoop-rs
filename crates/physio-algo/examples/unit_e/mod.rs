//! Shared loading and statistics for the Unit E analyses (effort, hrv window, 4.0 night).
//!
//! Personal health data stays OUTSIDE the repo: `WHOOP_ANALYSIS_DIR` (default `C:/work/whoop/_analysis`)
//! holds `data/<stream-set>/{hr,rr,gravity,band}.csv` and `ref/<wearer>_{cycles,sleeps,workouts}.csv`,
//! both written by the stdlib Python extractors beside them. Nothing here is committed with data.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub fn root() -> PathBuf {
    std::env::var("WHOOP_ANALYSIS_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("C:/work/whoop/_analysis"))
}

/// One stream set (one strap over one capture), all sorted by time.
#[derive(Default)]
pub struct Streams {
    pub hr: Vec<(i64, i32)>,
    /// Beats in emission order (file order within a second), `(second, rr_ms)`.
    pub rr: Vec<(i64, u16)>,
    pub grav: Vec<(i64, f64, f64, f64)>,
    pub band: Vec<(i64, i32)>,
}

fn lines(path: &PathBuf) -> Vec<String> {
    fs::read_to_string(path).map(|t| t.lines().map(str::to_string).collect()).unwrap_or_default()
}

/// A file under the analysis root as text, bad UTF-8 replaced; empty when absent.
pub fn read_lossy(rel: &str) -> String {
    fs::read(root().join(rel)).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

/// Write a per-row result file under `$WHOOP_ANALYSIS_DIR/out/`, which is outside every repo.
pub fn write_out(name: &str, text: &str) {
    let dir = root().join("out");
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(dir.join(name), text);
}

pub fn load(name: &str) -> Streams {
    let d = root().join("data").join(name);
    let mut s = Streams::default();
    for l in lines(&d.join("hr.csv")) {
        let mut it = l.split(',');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(t), Ok(v)) = (a.parse(), b.parse()) {
                s.hr.push((t, v));
            }
        }
    }
    s.hr.sort_by_key(|x| x.0);
    s.hr.dedup_by_key(|x| x.0);
    for l in lines(&d.join("rr.csv")) {
        let mut it = l.split(',');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(t), Ok(v)) = (a.parse(), b.parse::<f64>()) {
                s.rr.push((t, v as u16));
            }
        }
    }
    // Stable by second: beats inside one second keep their emission order.
    s.rr.sort_by_key(|x| x.0);
    for l in lines(&d.join("gravity.csv")) {
        let p: Vec<&str> = l.split(',').collect();
        if p.len() == 4 {
            if let (Ok(t), Ok(x), Ok(y), Ok(z)) = (p[0].parse(), p[1].parse(), p[2].parse(), p[3].parse()) {
                s.grav.push((t, x, y, z));
            }
        }
    }
    s.grav.sort_by_key(|x| x.0);
    s.grav.dedup_by_key(|x| x.0);
    for l in lines(&d.join("band.csv")) {
        let mut it = l.split(',');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if let (Ok(t), Ok(v)) = (a.parse(), b.parse::<f64>()) {
                s.band.push((t, v as i32));
            }
        }
    }
    s.band.sort_by_key(|x| x.0);
    s
}

pub fn slice_by<T>(v: &[T], key: impl Fn(&T) -> i64, a: i64, b: i64) -> &[T] {
    let i = v.partition_point(|x| key(x) < a);
    let j = v.partition_point(|x| key(x) < b);
    &v[i..j]
}

/// Fraction of the `[a, b)` seconds that carry an HR sample.
pub fn hr_coverage(hr: &[(i64, i32)], a: i64, b: i64) -> f64 {
    if b <= a {
        return 0.0;
    }
    slice_by(hr, |x| x.0, a, b).len() as f64 / (b - a) as f64
}

// ---- WHOOP export reference ----------------------------------------------------------------------------

fn table(name: &str) -> Vec<HashMap<String, String>> {
    let path = root().join("ref").join(name);
    let ls = lines(&path);
    let Some(h) = ls.first() else { return Vec::new() };
    let head: Vec<&str> = h.split(',').collect();
    ls[1..]
        .iter()
        .map(|l| head.iter().zip(l.split(',')).map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .collect()
}

fn f(m: &HashMap<String, String>, k: &str) -> Option<f64> {
    m.get(k).and_then(|v| v.parse().ok())
}

#[derive(Clone, Debug)]
pub struct Cycle {
    pub start: i64,
    pub end: Option<i64>,
    pub tz: i64,
    pub recovery: Option<f64>,
    pub rhr: Option<f64>,
    pub hrv: Option<f64>,
    pub strain: Option<f64>,
    pub sleep_on: Option<i64>,
    pub wake_on: Option<i64>,
    pub asleep: Option<f64>,
    pub inbed: Option<f64>,
    pub light: Option<f64>,
    pub deep: Option<f64>,
    pub rem: Option<f64>,
    pub awake: Option<f64>,
}

pub fn cycles(wearer: &str) -> Vec<Cycle> {
    table(&format!("{wearer}_cycles.csv"))
        .iter()
        .map(|m| Cycle {
            start: f(m, "cstart").unwrap() as i64,
            end: f(m, "cend").map(|x| x as i64),
            tz: f(m, "tzoff").unwrap_or(0.0) as i64,
            recovery: f(m, "recovery"),
            rhr: f(m, "rhr"),
            hrv: f(m, "hrv"),
            strain: f(m, "strain"),
            sleep_on: f(m, "sleep_on").map(|x| x as i64),
            wake_on: f(m, "wake_on").map(|x| x as i64),
            asleep: f(m, "asleep"),
            inbed: f(m, "inbed"),
            light: f(m, "light"),
            deep: f(m, "deep"),
            rem: f(m, "rem"),
            awake: f(m, "awake"),
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Sleep {
    pub cstart: i64,
    pub onset: i64,
    pub wake: i64,
    pub asleep: f64,
    pub inbed: f64,
    pub light: f64,
    pub deep: f64,
    pub rem: f64,
    pub awake: f64,
    pub nap: bool,
}

pub fn sleeps(wearer: &str) -> Vec<Sleep> {
    table(&format!("{wearer}_sleeps.csv"))
        .iter()
        .filter_map(|m| {
            Some(Sleep {
                cstart: f(m, "cstart")? as i64,
                onset: f(m, "onset")? as i64,
                wake: f(m, "wake")? as i64,
                asleep: f(m, "asleep")?,
                inbed: f(m, "inbed")?,
                light: f(m, "light")?,
                deep: f(m, "deep")?,
                rem: f(m, "rem")?,
                awake: f(m, "awake")?,
                nap: m.get("nap").map(|v| v == "True").unwrap_or(false),
            })
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Workout {
    pub cstart: i64,
    pub start: i64,
    pub end: i64,
    pub name: String,
    pub strain: Option<f64>,
    pub maxhr: Option<f64>,
    pub avghr: Option<f64>,
}

pub fn workouts(wearer: &str) -> Vec<Workout> {
    table(&format!("{wearer}_workouts.csv"))
        .iter()
        .filter_map(|m| {
            Some(Workout {
                cstart: f(m, "cstart")? as i64,
                start: f(m, "start")? as i64,
                end: f(m, "end")? as i64,
                name: m.get("name").cloned().unwrap_or_default(),
                strain: f(m, "strain"),
                maxhr: f(m, "maxhr"),
                avghr: f(m, "avghr"),
            })
        })
        .collect()
}

// ---- statistics ----------------------------------------------------------------------------------------

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

pub fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

pub fn mae(pred: &[f64], truth: &[f64]) -> f64 {
    mean(&pred.iter().zip(truth).map(|(p, t)| (p - t).abs()).collect::<Vec<_>>())
}

pub fn bias(pred: &[f64], truth: &[f64]) -> f64 {
    mean(&pred.iter().zip(truth).map(|(p, t)| p - t).collect::<Vec<_>>())
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 3 {
        return f64::NAN;
    }
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        f64::NAN
    } else {
        sab / (saa * sbb).sqrt()
    }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&ranks(a), &ranks(b))
}

/// Deterministic xorshift64*, so every bootstrap and permutation is reproducible.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// 95% percentile-bootstrap interval for the MEAN of `d` (paired differences), 4000 resamples.
pub fn boot_mean_ci(d: &[f64]) -> (f64, f64) {
    if d.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut means: Vec<f64> =
        (0..4000).map(|_| mean(&(0..d.len()).map(|_| d[rng.below(d.len())]).collect::<Vec<_>>())).collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (means[100], means[3899])
}

/// Shuffle `v` in place (Fisher-Yates).
pub fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}

pub fn fmt(x: f64) -> String {
    if x.is_nan() {
        "-".into()
    } else {
        format!("{x:.2}")
    }
}

// ---- sleep-engine plumbing -----------------------------------------------------------------------------

use physio_algo::sleep::{AccelSample, HrSample as SleepHr, RrRun, Session, SleepStage, SleepStreams};

/// Beats of one second share one run, in emission order: the grouping the app's `groupRuns` builds.
pub fn rr_runs(rr: &[(i64, u16)]) -> Vec<RrRun> {
    let mut out: Vec<RrRun> = Vec::new();
    for &(t, v) in rr {
        match out.last_mut() {
            Some(l) if l.ts == t => l.intervals.push(v),
            _ => out.push(RrRun { ts: t, intervals: vec![v] }),
        }
    }
    out
}

/// The sleep-engine input for `[a, b]` of one stream set.
pub fn sleep_streams(s: &Streams, a: i64, b: i64, tz_offset_s: i64) -> SleepStreams {
    SleepStreams {
        hr: slice_by(&s.hr, |x| x.0, a, b).iter().map(|&(t, v)| SleepHr { ts: t, bpm: v.clamp(0, 255) as u16 }).collect(),
        rr: rr_runs(slice_by(&s.rr, |x| x.0, a, b)),
        accel: slice_by(&s.grav, |x| x.0, a, b).iter().map(|&(t, x, y, z)| AccelSample { ts: t, x, y, z }).collect(),
        steps: Vec::new(),
        tz_offset_s,
        wrist_off: Vec::new(),
        band_sleep_state: slice_by(&s.band, |x| x.0, a, b).to_vec(),
    }
}

/// The detected session with the largest overlap with `[on, wake]`, if it covers at least half of it.
pub fn pick_session(sessions: &[Session], on: i64, wake: i64) -> Option<&Session> {
    let ov = |s: &Session| (s.end.min(wake) - s.start.max(on)).max(0);
    let best = sessions.iter().max_by_key(|s| ov(s))?;
    (ov(best) * 2 >= wake - on).then_some(best)
}

/// Minutes per stage of a session, `(wake, light, deep, rem)`.
pub fn stage_minutes(s: &Session) -> (f64, f64, f64, f64) {
    let mut m = [0.0f64; 4];
    for g in &s.segments {
        let i = match g.stage {
            SleepStage::Wake => 0,
            SleepStage::Light => 1,
            SleepStage::Deep => 2,
            SleepStage::Rem => 3,
        };
        m[i] += (g.end - g.start) as f64 / 60.0;
    }
    (m[0], m[1], m[2], m[3])
}

/// Minutes inside the session that the engine declines to score.
pub fn unscored_minutes(s: &Session) -> f64 {
    s.unscored.iter().map(|&(a, b)| (b - a) as f64 / 60.0).sum()
}
