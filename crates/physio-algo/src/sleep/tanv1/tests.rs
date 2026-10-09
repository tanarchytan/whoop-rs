use super::*;
use crate::sleep::input::{AccelSample, HrSample};
use crate::sleep::{analyze, analyze_with, Engine, Params, SleepStreams};

/// Copy of `pipeline::tests::restless_start_night` (private there).
fn restless_start_night() -> SleepInput {
    let start = 1_749_517_200i64;
    let dur = 4 * 3_600i64;
    let spike = |i: i64| i < 2_400 && (i / 30) % 4 == 0;
    SleepInput {
        start,
        end: start + dur,
        accel: (0..dur)
            .map(|i| {
                let j = if spike(i) { 0.30 } else { 0.0 };
                AccelSample { ts: start + i, x: j, y: 0.0, z: 1.0 - j }
            })
            .collect(),
        hr: (0..dur).map(|i| HrSample { ts: start + i, bpm: if spike(i) { 72 } else { 52 } }).collect(),
        rr: Vec::new(),
    }
}

fn still_night() -> SleepInput {
    let start = 1_749_517_200i64;
    let dur = 2 * 3_600i64;
    SleepInput {
        start,
        end: start + dur,
        hr: (0..dur).map(|i| HrSample { ts: start + i, bpm: 50 }).collect(),
        accel: (0..dur).map(|i| AccelSample { ts: start + i, x: 0.0, y: 0.0, z: 1.0 }).collect(),
        rr: Vec::new(),
    }
}

fn empty_night() -> SleepInput {
    SleepInput { start: 1_749_517_200, end: 1_749_517_200 + 600, hr: Vec::new(), rr: Vec::new(), accel: Vec::new() }
}

fn fixtures() -> Vec<SleepInput> {
    vec![super::super::golden_tests::golden_input(), restless_start_night(), still_night(), empty_night()]
}

#[test]
fn null_config_is_v2_label_for_label() {
    for input in fixtures() {
        let got = stage(&input, &Tanv1Cfg::NULL);
        assert_eq!(v2::stage(&input), got.segments);
        assert!(got.unscored.is_empty());
    }
}

#[test]
fn analyze_with_v2_is_analyze() {
    let input = super::super::golden_tests::golden_input();
    let streams = SleepStreams { hr: input.hr, rr: input.rr, accel: input.accel, tz_offset_s: 0, ..Default::default() };
    let sessions = analyze_with(&streams, Engine::V2);
    assert_eq!(analyze(&streams), sessions);
    assert!(sessions.iter().all(|s| s.unscored.is_empty()));
}

#[test]
fn default_config_holes_never_relabel() {
    let input = super::super::golden_tests::golden_input();
    let null = stage(&input, &Tanv1Cfg::NULL);
    let got = stage(&input, &Tanv1Cfg::DEFAULT);
    assert_eq!(null.segments, got.segments);
    assert!(!got.unscored.is_empty());
    assert!(got.unscored.windows(2).all(|w| w[0].0 < w[0].1 && w[0].1 < w[1].0), "ordered, disjoint, gapped");
    let n = crate::sleep::epoch_starts_v2(&crate::sleep::prepare_v2(&input, &BASE)).len();
    let want = n - ((n as f64 * abstain::DEFAULT_COVERAGE).round() as usize).max(abstain::MIN_EPOCHS).min(n);
    let refused: i64 = got.unscored.iter().map(|(a, b)| b - a).sum();
    assert_eq!(want as i64 * 30, refused);
}

#[test]
fn tanv1_engine_reports_holes_through_analyze_with() {
    let input = super::super::golden_tests::golden_input();
    let streams = SleepStreams { hr: input.hr, rr: input.rr, accel: input.accel, tz_offset_s: 0, ..Default::default() };
    let v2s = analyze_with(&streams, Engine::V2);
    let t = analyze_with(&streams, Engine::Tanv1);
    assert_eq!(v2s.len(), t.len());
    assert!(t.iter().all(|s| !s.unscored.is_empty()));
}

#[test]
fn default_differs_from_null_only_in_clamp_and_abstain() {
    let d = Tanv1Cfg::DEFAULT;
    let mut base = d.base;
    base.clamp_only_without_rr = Params::SHIPPED.clamp_only_without_rr;
    assert!(d.base.clamp_only_without_rr);
    assert_eq!(base, Tanv1Cfg::NULL.base);
    assert_eq!(d.abstain, Some(abstain::DEFAULT_COVERAGE));
    assert_eq!(Tanv1Cfg::NULL.abstain, None);
}

/// The shipped abstention values, as literals: every other test reads them through the constants.
#[test]
fn shipped_abstention_is_80_percent_with_a_20_epoch_floor() {
    assert_eq!(Tanv1Cfg::DEFAULT.abstain, Some(0.80));
    assert_eq!(abstain::MIN_EPOCHS, 20);
}
