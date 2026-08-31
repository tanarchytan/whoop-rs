//! The four cardiac columns of the tanv1 feature vector, on the [`EPOCH_S`] grid.
//!
//! The single producer for every harness that fits one; they call [`cardiac_series`].

use std::collections::BTreeMap;

use physio_algo::sleep::features::{Cardiac, EPOCH_S};
use physio_algo::sleep::{flatten_rr, resp_regularity, HrSample, RrRun};
use physio_algo::stats::population_sd;

/// Per-night z-score of a per-epoch series, from the library so the harnesses and the stager share one.
pub use physio_algo::sleep::cardiac::zscore_column as zscore;

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
    Some(population_sd(&v))
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

/// The four cardiac columns of the tanv1 feature vector, over `n` epochs from `w0`. `features::extract`
/// buckets its other 24 columns on [`EPOCH_S`] and indexes this by the same k, so the grids must agree.
pub fn cardiac_series(
    w0: i64,
    n: usize,
    epoch: i64,
    hr: &[HrSample],
    rr: &[RrRun],
) -> Vec<Cardiac> {
    assert_eq!(epoch, EPOCH_S, "the cardiac grid must match the grid features::extract buckets on");
    let sec = per_second_hr(hr);
    // The epoch mean averages the PER-SECOND means, so an unevenly sampled second keeps one vote.
    let mut sum = vec![(0.0f64, 0.0f64); n];
    for (&t, &b) in sec.range(w0..w0 + n as i64 * epoch) {
        let k = ((t - w0) / epoch) as usize;
        sum[k].0 += b;
        sum[k].1 += 1.0;
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
