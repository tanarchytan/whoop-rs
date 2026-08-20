//! The cardiac columns of the tanv1 feature vector, in one place.
//!
//! Two harnesses carried a copy each and they had already drifted by a clamp; a drift that mattered
//! would have surfaced as a finding about the data rather than about the code.

#![allow(dead_code)]

use std::collections::BTreeMap;

use physio_algo::sleep::features::Cardiac;
use physio_algo::sleep::{flatten_rr, resp_regularity, HrSample, RrRun};

/// Per-night z-score of a per-epoch series, missing where the series is.
pub fn zscore(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let present: Vec<f64> = v.iter().flatten().copied().collect();
    if present.len() < 2 {
        return vec![None; v.len()];
    }
    let m = present.iter().sum::<f64>() / present.len() as f64;
    let sd = (present.iter().map(|x| (x - m).powi(2)).sum::<f64>() / present.len() as f64).sqrt();
    if sd <= 0.0 {
        return vec![None; v.len()];
    }
    v.iter().map(|o| o.map(|x| (x - m) / sd)).collect()
}

/// One heart rate per second, averaged where a second carries several samples.
pub fn per_second_hr(hr: &[HrSample]) -> BTreeMap<i64, f64> {
    let mut acc: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for s in hr {
        let e = acc.entry(s.ts).or_insert((0.0, 0.0));
        e.0 += s.bpm as f64;
        e.1 += 1.0;
    }
    acc.into_iter().map(|(t, (a, c))| (t, a / c)).collect()
}

/// Population sd of PER-SECOND heart rate over `[lo, hi)`, the statistic the shipped recipe reads.
/// Averaging to per-epoch means first and taking the spread of THOSE is a much smoother quantity.
pub fn std_of_seconds(sec: &BTreeMap<i64, f64>, lo: i64, hi: i64) -> Option<f64> {
    let v: Vec<f64> = sec.range(lo..hi).map(|(_, b)| *b).collect();
    if v.len() < 2 {
        return None;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    Some((v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt())
}

/// Within-night percentile rank in 0..1, `bisect_right / n` over the present values - the transform
/// the deep gate applies. Missing stays missing rather than becoming a manufactured median.
pub fn rank_pct(v: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut sorted: Vec<f64> = v.iter().flatten().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return vec![None; v.len()];
    }
    v.iter()
        .map(|o| o.map(|x| sorted.partition_point(|s| *s <= x) as f64 / sorted.len() as f64))
        .collect()
}

/// The four cardiac columns of the tanv1 feature vector, over `n` epochs from `w0`.
pub fn cardiac_series(
    w0: i64,
    n: usize,
    epoch: i64,
    hr: &[HrSample],
    rr: &[RrRun],
) -> Vec<Cardiac> {
    let sec = per_second_hr(hr);
    let mut sum = vec![(0.0f64, 0.0f64); n];
    for s in hr {
        let k = ((s.ts - w0) / epoch).max(0) as usize;
        if k < n {
            sum[k].0 += s.bpm as f64;
            sum[k].1 += 1.0;
        }
    }
    let raw: Vec<Option<f64>> = sum.iter().map(|(a, c)| (*c > 0.0).then(|| a / c)).collect();
    let hr_z = zscore(&raw);
    let starts: Vec<i64> = (0..n).map(|k| w0 + k as i64 * epoch).collect();
    let hv: Vec<Option<f64>> =
        starts.iter().map(|e| std_of_seconds(&sec, e - 150, e + epoch + 150)).collect();
    let hr_var_z = zscore(&hv);
    let flat: Vec<Option<f64>> =
        starts.iter().map(|e| std_of_seconds(&sec, e - 330, e + epoch + 360)).collect();
    let flat_pct = rank_pct(&flat);

    let mut beats_by: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (ts, ms) in flatten_rr(rr) {
        beats_by.entry(ts).or_default().push(ms);
    }
    let resp: Vec<Option<f64>> = starts
        .iter()
        .map(|e| {
            let mut beats: Vec<(f64, f64)> = beats_by
                .range(e - 90..e + 120)
                .flat_map(|(t, vs)| vs.iter().map(|v| (*t as f64, v.clamp(300.0, 2000.0))))
                .collect();
            beats.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
            resp_regularity(&beats)
        })
        .collect();
    let resp_z = zscore(&resp);

    (0..n)
        .map(|k| Cardiac {
            hr_z: hr_z[k],
            hr_var_z: hr_var_z[k],
            hr_flat_pct: flat_pct[k],
            resp_z: resp_z[k],
        })
        .collect()
}
