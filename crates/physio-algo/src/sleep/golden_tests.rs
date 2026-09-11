//! Frozen-golden + contract tests for the sleep stagers. The golden hypnogram pins the shipped V2 OUTPUT
//! — six segments on a crafted integer-only night — not the recipe that produced it; which coefficients
//! the table can actually see is measured beside it, and six of the twenty-six are blind to it. Any drift
//! in the twenty it does see fails immediately. Integer-literal input keeps the two languages bit-identical.

use super::input::{AccelSample, HrSample, RrRun, SleepInput, StepSample};
use super::params::Params;
use super::refine::{refine_with, RefineParams};
use super::v2::stage_with as stage_v2_with;
use super::{analyze, motion_dense, stage_v2, SleepStage, SleepStreams, StageSegment, DEEP_GATE_THRESH};

const REF_MIDNIGHT: i64 = 1_749_513_600;

fn rsa_wave(ph: usize, i: i64) -> i64 {
    let amp = [12i64, 60, 30, 20][ph];
    [0, amp, 0, -amp][(i % 4) as usize]
}

/// The crafted 4-phase night (deep-favorable → high-RSA → mild → restless) used by the frozen golden.
fn golden_input() -> SleepInput {
    let start = REF_MIDNIGHT + 3_600;
    let phase: i64 = 90 * 60;
    let dur = phase * 4;
    let mut accel = Vec::new();
    let mut hr = Vec::new();
    let mut rr = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        let ph = (i / phase) as usize;
        let restless = ph == 3 && (i % 20) < 6;
        if restless {
            accel.push(AccelSample { ts, x: 0.2, y: 0.15, z: 0.96 });
        } else {
            accel.push(AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
        }
        let bpm: i64 = match ph {
            0 => 50,
            1 => 54 + [0, 1, 2, 3, 2, 1][((i / 20) % 6) as usize],
            2 => 56 + (i / 60) % 4,
            _ => 66 + (i / 30) % 6,
        };
        hr.push(HrSample { ts, bpm: bpm as u16 });
        let rr_ms = 60_000 / bpm + rsa_wave(ph, i);
        rr.push(RrRun { ts, intervals: vec![rr_ms as u16] });
    }
    SleepInput { start, end: start + dur, hr, rr, accel }
}

/// The shipped V2 output on the crafted night, segment for segment. The three constant stagers below are
/// checked against the same table, so the golden is known to reject one that always answers a single stage.
#[test]
fn frozen_golden_hypnogram_v2() {
    let input = golden_input();
    let start = input.start;
    let segs = stage_v2(&input);
    let golden = [
        (0i64, 5070i64, SleepStage::Deep),
        (5070, 5280, SleepStage::Light),
        (5280, 5550, SleepStage::Rem),
        (5550, 10740, SleepStage::Light),
        (10740, 16290, SleepStage::Rem),
        (16290, 21600, SleepStage::Wake),
    ];
    assert_eq!(golden.len(), segs.len(), "segment count");
    for (k, g) in golden.iter().enumerate() {
        assert_eq!(start + g.0, segs[k].start, "seg {k} start");
        assert_eq!(start + g.1, segs[k].end, "seg {k} end");
        assert_eq!(g.2, segs[k].stage, "seg {k} stage");
    }
    let expected: Vec<StageSegment> =
        golden.iter().map(|g| StageSegment { start: start + g.0, end: start + g.1, stage: g.2 }).collect();
    assert_eq!(expected, segs);
    for stage in [SleepStage::Light, SleepStage::Deep, SleepStage::Wake] {
        let constant = vec![StageSegment { start, end: input.end, stage }];
        assert_ne!(expected, constant, "an always-{} stager must fail this table", stage.as_str());
    }
}

/// Which V2 coefficients the frozen table can see. Twenty of the twenty-six change it; the six below do
/// not, even at extreme values, so editing one of those is invisible to the golden and needs the fixture
/// sheet instead. A blind row that starts moving the table is a change to report, not to silence.
#[test]
fn the_frozen_golden_defends_twenty_of_the_twenty_six_v2_coefficients() {
    let input = golden_input();
    let base = stage_v2(&input);
    let p = Params::SHIPPED;
    let seen: Vec<(&str, Params)> = vec![
        ("deep_hrv", Params { deep_hrv: 0.0, ..p }),
        ("deep_hr", Params { deep_hr: 5.0, ..p }),
        ("deep_motion", Params { deep_motion: 5.0, ..p }),
        ("rem_hrv", Params { rem_hrv: 0.0, ..p }),
        ("rem_motion", Params { rem_motion: 0.0, ..p }),
        ("rem_hr", Params { rem_hr: 0.0, ..p }),
        ("awake_motion", Params { awake_motion: 10.0, ..p }),
        ("awake_hrv", Params { awake_hrv: 10.0, ..p }),
        ("awake_hr", Params { awake_hr: 10.0, ..p }),
        ("deep_gate_thresh", Params { deep_gate_thresh: 0.0, ..p }),
        ("deep_gate_slope", Params { deep_gate_slope: 0.0, ..p }),
        ("motion_gate_boost", Params { motion_gate_boost: 0.0, ..p }),
        ("resp_weight", Params { resp_weight: 0.0, ..p }),
        ("base_rate", Params { base_rate: [0.25, 0.25, 0.25, 0.25], ..p }),
        ("cycle_deep_scale", Params { cycle_deep_scale: 0.0, ..p }),
        ("cycle_deep_decay", Params { cycle_deep_decay: 0.1, ..p }),
        ("cycle_rem_scale", Params { cycle_rem_scale: 0.0, ..p }),
        ("cycle_rem_onset_minutes", Params { cycle_rem_onset_minutes: 600.0, ..p }),
        ("cycle_rem_ramp_cap", Params { cycle_rem_ramp_cap: 0.3, ..p }),
        ("transition", Params { transition: [[0.25; 4]; 4], ..p }),
    ];
    let blind: Vec<(&str, Params)> = vec![
        ("awake_deadzone", Params { awake_deadzone: 5.0, ..p }),
        ("jerk_move_mult", Params { jerk_move_mult: 1000.0, ..p }),
        ("jerk_gate_mult", Params { jerk_gate_mult: 1000.0, ..p }),
        ("cycle_rem_early_frac", Params { cycle_rem_early_frac: 1.0, ..p }),
        ("cycle_rem_early_penalty", Params { cycle_rem_early_penalty: 50.0, ..p }),
        ("cycle_clock_from_onset", Params { cycle_clock_from_onset: true, ..p }),
    ];
    assert_eq!(base, stage_v2_with(&input, &p), "the baseline row must be the shipped recipe itself");
    for (name, q) in &seen {
        assert_ne!(base, stage_v2_with(&input, q), "{name} must change the golden table");
    }
    for (name, q) in &blind {
        assert_eq!(base, stage_v2_with(&input, q), "{name} now moves the golden — move it to the seen list");
    }
    assert_eq!((20, 6), (seen.len(), blind.len()));
}

#[test]
fn tuned_deep_boundary_constants_are_pinned() {
    assert_eq!(0.40, DEEP_GATE_THRESH);
    assert_eq!([0.76, 0.012, 0.216, 0.012], Params::SHIPPED.transition[0]);
    for row in Params::SHIPPED.transition.iter() {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "transition row must sum to 1.0");
    }
}

#[test]
fn v2_segments_tile_the_span_contiguously() {
    let input = golden_input();
    let segs = stage_v2(&input);
    assert!(!segs.is_empty());
    assert_eq!(input.start, segs.first().unwrap().start);
    assert_eq!(input.end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start, "no gap/overlap");
        assert!(w[1].end > w[1].start, "non-empty");
    }
}

#[test]
fn v2_degenerate_input_falls_back_to_single_light_block() {
    let start = REF_MIDNIGHT;
    let end = start + 3_600;
    let input = SleepInput {
        start,
        end,
        hr: Vec::new(),
        rr: Vec::new(),
        accel: vec![AccelSample { ts: start, x: 0.0, y: 0.0, z: 1.0 }],
    };
    let segs = stage_v2(&input);
    assert_eq!(1, segs.len());
    assert_eq!(SleepStage::Light, segs[0].stage);
    assert_eq!(start, segs[0].start);
    assert_eq!(end, segs[0].end);
}

#[test]
fn analyze_still_night_stages_and_tiles_the_span() {
    let input = golden_input();
    let streams =
        SleepStreams { hr: input.hr.clone(), rr: input.rr.clone(), accel: input.accel.clone(), tz_offset_s: 0, ..Default::default() };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let segs = &sessions[0].segments;
    assert_eq!(sessions[0].start, segs.first().unwrap().start);
    assert_eq!(sessions[0].end, segs.last().unwrap().end);
    for w in segs.windows(2) {
        assert_eq!(w[0].end, w[1].start);
    }
}

/// The golden above carries no step stream, so `analyze`'s last stage declines on it. This drives the
/// same night WITH one. Its only wake run is its last, 5,309 s of post-wake in-bed time: the shipped
/// rule leaves it and the rule H opened with took all of it, so the fix is pinned end to end here.
#[test]
fn analyze_keeps_the_golden_night_trailing_wake_when_the_step_stream_is_dense() {
    let input = golden_input();
    let steps: Vec<StepSample> = (input.start..input.end)
        .step_by(30)
        .map(|ts| StepSample { ts, counter: 100, activity_class: Some(0) })
        .collect();
    let bare = SleepStreams {
        hr: input.hr.clone(),
        rr: input.rr.clone(),
        accel: input.accel.clone(),
        tz_offset_s: 0,
        ..Default::default()
    };
    let dense = SleepStreams { steps: steps.clone(), ..bare.clone() };
    let (a, b) = (analyze(&bare), analyze(&dense));
    assert_eq!((1, 1), (a.len(), b.len()));
    let span = (b[0].start, b[0].end);
    assert!(!motion_dense(span.0, span.1, &input.accel, &[]), "the bare golden must decline");
    assert!(motion_dense(span.0, span.1, &input.accel, &steps), "the dense one must be accepted");
    assert_eq!(a[0].segments, b[0].segments, "the shipped rule leaves a trailing wake run alone");
    let tail = *b[0].segments.last().unwrap();
    assert_eq!(SleepStage::Wake, tail.stage);
    assert_eq!(5_309, tail.end - tail.start);
    assert_eq!(b[0].end, tail.end);

    // The rule H opened with, over the same staging and the same streams: it takes the whole run.
    let span_steps: Vec<StepSample> = steps.iter().copied().filter(|s| s.ts >= span.0 && s.ts < span.1).collect();
    let span_accel: Vec<AccelSample> =
        input.accel.iter().copied().filter(|g| g.ts >= span.0 && g.ts < span.1).collect();
    let pre_h = RefineParams { skip_window_edges: false, ..RefineParams::SHIPPED };
    let old = refine_with(&b[0].segments, &span_accel, &span_steps, &pre_h);
    assert_ne!(SleepStage::Wake, old.last().unwrap().stage, "pre-H converted the trailing run");
}

/// The detected span on a window that is nothing but the night: both edges pinned, not a floor. Efficiency
/// is asserted at its value, not inside `0..=1` — the function clamps to that range, so the old range
/// assert held for any implementation at all.
#[test]
fn analyze_pins_the_span_of_a_window_that_is_all_still_night() {
    let start = REF_MIDNIGHT;
    let dur = 2 * 3600;
    let mut hr = Vec::new();
    let mut accel = Vec::new();
    for i in 0..dur {
        let ts = start + i;
        hr.push(HrSample { ts, bpm: 50 });
        accel.push(AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
    }
    let streams = SleepStreams { hr, accel, tz_offset_s: 0, ..Default::default() };
    let sessions = analyze(&streams);
    assert_eq!(1, sessions.len());
    let s = &sessions[0];
    assert_eq!((start, start + 7_199), (s.start, s.end));
    assert_eq!(1.0, s.efficiency, "no wake was staged, so every in-bed second is asleep");
    assert_eq!(Some(50), s.resting_hr);
    assert_eq!(s.start, s.segments.first().unwrap().start);
    assert_eq!(s.end, s.segments.last().unwrap().end);
}

/// The same two claims on a window that is mostly NOT sleep: four hours awake and moving at 78 bpm, seven
/// still at 50, three awake again, plus a ten-minute awake dip to 40. A whole-window detector and an
/// anchor read over the whole window both fail here; neither could fail the all-night fixture above.
#[test]
fn analyze_finds_the_night_inside_a_waking_day_and_anchors_resting_hr_to_it() {
    let (day_start, night_start) = (REF_MIDNIGHT - 4 * 3600, REF_MIDNIGHT);
    let (night_end, day_end) = (night_start + 7 * 3600, night_start + 10 * 3600);
    let (mut hr, mut accel) = (Vec::new(), Vec::new());
    for ts in day_start..day_end {
        let asleep = (night_start..night_end).contains(&ts);
        let dip = (day_start + 1_800..day_start + 2_400).contains(&ts);
        hr.push(HrSample { ts, bpm: if asleep { 50 } else if dip { 40 } else { 78 } });
        let f = ((ts % 4) as f64) * 0.25;
        accel.push(if asleep {
            AccelSample { ts, x: 0.0, y: 0.0, z: 1.0 }
        } else {
            AccelSample { ts, x: f, y: 0.3 - f, z: 0.9 }
        });
    }
    let sessions = analyze(&SleepStreams { hr, accel, tz_offset_s: 0, ..Default::default() });
    assert_eq!(1, sessions.len());
    let s = &sessions[0];
    assert_eq!((night_start + 271, night_start + 25_064), (s.start, s.end));
    // Both edges inside the true night, and 25,607 s short of the 14 h a whole-window detector returns.
    assert!(s.start >= night_start && s.end <= night_end);
    assert_eq!(25_607, (day_end - day_start) - (s.end - s.start));
    // The anchor is the SESSION floor: 50, not the 40 the awake dip would give over the whole window.
    assert_eq!(Some(50), s.resting_hr);
    assert_eq!(s.start, s.segments.first().unwrap().start);
    assert_eq!(s.end, s.segments.last().unwrap().end);
}

/// The conditioned decoder at beta 0 is the shipped one label for label. This is the null every
/// conditioned arm is read against, and it must hold on the REAL pipeline, not only in the unit test.
#[test]
fn conditioned_at_beta_zero_is_the_shipped_hypnogram() {
    use super::conditioned::ConditionedCfg;
    use super::pipeline::{run, DecodeCfg, EmitCfg, SleepConfig};

    let input = golden_input();
    let shipped = run(&input, &SleepConfig::shipped(), &Params::SHIPPED);
    let null = SleepConfig { emit: EmitCfg::V2, decode: DecodeCfg::Conditioned(ConditionedCfg { beta_milli: 0 }) };
    let cond = run(&input, &null, &Params::SHIPPED);
    assert_eq!(shipped.stages, cond.stages, "beta 0 must reproduce the shipped labels");
    assert!(!shipped.stages.as_ref().unwrap().is_empty());

    // And a real beta must be ABLE to move it, or the seam is wired to nothing.
    let loose = SleepConfig { emit: EmitCfg::V2, decode: DecodeCfg::Conditioned(ConditionedCfg { beta_milli: 3000 }) };
    let moved = run(&input, &loose, &Params::SHIPPED);
    assert_ne!(shipped.stages, moved.stages, "beta 3.0 on the golden night must change something");
}

/// The two posterior decode rules on the REAL pipeline: `MarkovLoss` at unit costs IS the marginal
/// rule, a dearer class is called more, and the marginal rule is not just Viterbi renamed. Without
/// the last assertion the seam could be wired to the shipped decoder and every arm would agree.
#[test]
fn the_posterior_rules_reach_the_pipeline_and_unit_costs_are_the_marginal_rule() {
    use super::markov_loss::Costs;
    use super::pipeline::{run, CostsCfg, DecodeCfg, EmitCfg, SleepConfig};
    use super::SleepStage;

    let input = golden_input();
    let arm = |d: DecodeCfg| run(&input, &SleepConfig { emit: EmitCfg::V2, decode: d }, &Params::SHIPPED);

    let shipped = arm(DecodeCfg::Viterbi);
    let marginal = arm(DecodeCfg::PosteriorMarginal);
    let unit = arm(DecodeCfg::MarkovLoss(CostsCfg(Costs::UNIT)));
    assert!(marginal.stages.as_ref().is_some_and(|s| s.len() > 100), "the night must stage");
    assert_eq!(marginal.stages, unit.stages, "unit costs must be the plain marginal argmax");
    assert_ne!(shipped.stages, marginal.stages, "the marginal rule must not be viterbi renamed");

    // `fc` is charged on the TRUE class, so a dearer class is called MORE. Indexed in STAGE_ORDER.
    let deep = |s: &Option<Vec<SleepStage>>| {
        s.as_ref().map_or(0, |v| v.iter().filter(|x| **x == SleepStage::Deep).count())
    };
    let dear = arm(DecodeCfg::MarkovLoss(CostsCfg(Costs { fc: [8.0, 1.0, 1.0, 1.0], ..Costs::UNIT })));
    assert!(
        deep(&dear.stages) > deep(&unit.stages),
        "a dearer deep must be called more: {} vs {}",
        deep(&dear.stages),
        deep(&unit.stages)
    );
}

/// The pipeline at `shipped()` must equal `stage_v2_with` segment for segment. Everything the staged
/// refactor measures runs through `run_to`, so a drift here invalidates all of it.
#[test]
fn pipeline_shipped_reproduces_stage_v2() {
    use super::pipeline::{run, SleepConfig, StepId};

    let input = golden_input();
    let p = Params::SHIPPED;
    let st = run(&input, &SleepConfig::shipped(), &p);

    let want = stage_v2_with(&input, &p);
    let got = st.stages.as_ref().expect("decode populates stages");
    assert!(!got.is_empty(), "the golden night must produce epochs");

    let mut flat = Vec::new();
    for seg in &want {
        let n = ((seg.end - seg.start) / 30).max(0) as usize;
        flat.extend(std::iter::repeat_n(seg.stage, n));
    }
    assert_eq!(flat.len(), got.len(), "epoch count must match stage_v2_with");
    assert_eq!(&flat, got, "pipeline labels must match stage_v2_with exactly");

    assert_eq!(st.reached, Some(StepId::Report));
    assert_eq!(st.trace.len(), 19, "every step traces, implemented or not");
}

/// The golden night's onset anchor sits at its first epoch, so it coincides with the window anchor and
/// forcing either one leaves the labels alone. That is why `pipeline_shipped_reproduces_stage_v2`
/// cannot defend the anchor and `emissions_depend_on_the_probe_anchor` has to.
#[test]
fn the_golden_night_anchors_at_its_first_epoch() {
    use super::pipeline::{run_to, SleepConfig, StepId};
    use super::v2::Anchor;

    let st = run_to(&golden_input(), &SleepConfig::shipped(), &Params::SHIPPED, StepId::Anchor);
    assert!(matches!(st.anchor, Some(Anchor::Onset(0))), "expected an onset anchor at epoch 0");
}

/// Stopping early must not change what the earlier steps produced.
#[test]
fn stopping_early_leaves_the_prefix_identical() {
    use super::pipeline::{run, run_to, SleepConfig, StepId};

    let input = golden_input();
    let p = Params::SHIPPED;
    let full = run(&input, &SleepConfig::shipped(), &p);
    let part = run_to(&input, &SleepConfig::shipped(), &p, StepId::Decode);

    assert!(part.stages.is_some(), "decode is step 16, so labels exist");
    assert_eq!(part.stages, full.stages, "stopping at 16 must not change the labels");
    assert_eq!(
        part.prefix_digest(StepId::Decode),
        full.prefix_digest(StepId::Decode),
        "the 1..=16 prefix digest must not depend on how far the run went"
    );

    // Without these, deleting the `upto` filter in `run_to` passes every other assertion here.
    assert_eq!(Some(StepId::Decode), part.reached, "stopping at 16 must reach 16, not 19");
    assert_eq!(16, part.trace.len(), "stopping at 16 must run 16 steps");

    let d = part.digest_of(StepId::Decode).expect("decode traces");
    assert_ne!(0, d, "a real decode must not digest to zero");

    let early = run_to(&input, &SleepConfig::shipped(), &p, StepId::Emit);
    assert!(early.stages.is_none(), "labels cannot exist before the decode step");
    assert!(early.emissions.is_some(), "but emissions must, since step 14 ran");
    assert_ne!(
        0,
        early.digest_of(StepId::Validate).expect("validate traces"),
        "step 1 must digest the input, not a constant"
    );
}
