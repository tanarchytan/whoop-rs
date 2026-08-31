//! MESA R-peak nights, plus the two transforms that turn its channel into ours.
//!
//! MESA carries beat times at 1/256 s with the PSG stage already joined per beat, over 1,971 nights.
//! It has NO wrist motion, so it cannot score an engine — it screens CARDIAC features, and it does so
//! on a channel far better than ours. [`degrade_timing`] and [`degrade_coverage`] are how that gap is
//! measured instead of assumed: a feature is worth building only if it survives them.
//!
//! Stage codes are AASM as NSRR writes them and were counted, not assumed: 0, 1, 2, 3, 5 appear and 4
//! does not, so N4 is already folded into N3.

use std::path::{Path, PathBuf};

/// One beat: its time in seconds from recording start, the interval that ended at it, and the epoch
/// it falls in.
#[derive(Clone, Copy, Debug)]
pub struct Beat {
    pub t: f64,
    pub rr: f64,
    pub epoch: usize,
}

pub struct MesaNight {
    pub id: String,
    pub beats: Vec<Beat>,
    /// Stage per epoch on our own index map (wake 0, light 1, deep 2, REM 3); `None` where unscored.
    pub stage: Vec<Option<usize>>,
}

/// NSRR stage code -> our index. 1 and 2 both fold to light; 4 never occurs in this corpus.
fn stage_of(code: &str) -> Option<usize> {
    match code.trim() {
        "0" => Some(0),
        "1" | "2" => Some(1),
        "3" | "4" => Some(2),
        "5" => Some(3),
        _ => None,
    }
}

/// Intervals outside this are not physiology; the same window the decoder applies.
const RR_MIN_MS: f64 = 250.0;
const RR_MAX_MS: f64 = 2500.0;

pub fn root() -> PathBuf {
    std::env::var("MESA_RPOINTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../whoop-data/datasets/mesa/annotations-rpoints"
            ))
        })
}

/// Header index of each column we read, so a reordered export cannot silently shift them.
fn columns(header: &str) -> Option<(usize, usize, usize)> {
    let cols: Vec<&str> = header.split(',').map(|c| c.trim().trim_matches('"')).collect();
    let find = |name: &str| cols.iter().position(|c| *c == name);
    Some((find("seconds")?, find("stage")?, find("epoch")?))
}

pub fn read_night(path: &Path) -> Option<MesaNight> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let (c_sec, c_stage, c_epoch) = columns(lines.next()?)?;

    let mut raw: Vec<(f64, usize, Option<usize>)> = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() <= c_sec.max(c_stage).max(c_epoch) {
            continue;
        }
        let (Ok(t), Ok(ep)) = (f[c_sec].trim().parse::<f64>(), f[c_epoch].trim().parse::<usize>())
        else {
            continue;
        };
        raw.push((t, ep, stage_of(f[c_stage])));
    }
    if raw.len() < 1000 {
        return None;
    }
    raw.sort_by(|a, b| a.0.total_cmp(&b.0));

    // `epoch` is 1-based in the export; index it from zero so it lines up with our own grids.
    let n = raw.iter().map(|r| r.1).max()?;
    let mut stage = vec![None; n];
    for (_, ep, s) in &raw {
        if *ep >= 1 && *ep <= n {
            stage[*ep - 1] = *s;
        }
    }

    let mut beats = Vec::with_capacity(raw.len());
    for w in raw.windows(2) {
        let rr = (w[1].0 - w[0].0) * 1000.0;
        if (RR_MIN_MS..=RR_MAX_MS).contains(&rr) {
            beats.push(Beat { t: w[1].0, rr, epoch: w[1].1.saturating_sub(1) });
        }
    }
    let id = path.file_stem()?.to_string_lossy().replace("-rpoint", "");
    Some(MesaNight { id, beats, stage })
}

/// `limit` nights in name order, so a screen is reproducible and a subset is a prefix.
pub fn nights(limit: usize) -> Vec<MesaNight> {
    let dir = root();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("MESA rpoints unreadable at {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "csv"))
        .collect();
    files.sort();
    files.iter().take(limit).filter_map(|p| read_night(p)).collect()
}

/// Push the beats through OUR wire format and back: a whole-second stamp per beat, several beats
/// sharing one, then the reconstruction `v2::beats_in` performs. What our timestamp resolution costs.
pub fn degrade_timing(beats: &[Beat]) -> Vec<Beat> {
    let mut out: Vec<Beat> = Vec::with_capacity(beats.len());
    let mut i = 0;
    while i < beats.len() {
        let sec = beats[i].t.floor();
        let mut j = i;
        let mut off = 0.0;
        while j < beats.len() && beats[j].t.floor() == sec {
            // The first beat of a stamp sits ON it; each later one is its own interval further along.
            if j > i {
                off += beats[j].rr / 1000.0;
            }
            out.push(Beat { t: sec + off, rr: beats[j].rr.round(), epoch: beats[j].epoch });
            j += 1;
        }
        i = j;
    }
    out
}

/// Drop beats to leave roughly `keep` of them, in RUNS rather than at random: PPG dropout is a gap
/// during motion, not independent noise, and a feature tolerant of scattered losses can still fail on
/// a gap. `seed` fixes the pattern.
pub fn degrade_coverage(beats: &[Beat], keep: f64, seed: u64) -> Vec<Beat> {
    if !(0.0..1.0).contains(&keep) {
        return beats.to_vec();
    }
    const RUN_BEATS: usize = 60;
    let mut out = Vec::with_capacity(beats.len());
    let mut s = seed | 1;
    let mut i = 0;
    while i < beats.len() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 11) as f64 / (1u64 << 53) as f64;
        let take = u < keep;
        for b in beats.iter().skip(i).take(RUN_BEATS) {
            if take {
                out.push(*b);
            }
        }
        i += RUN_BEATS;
    }
    out
}
