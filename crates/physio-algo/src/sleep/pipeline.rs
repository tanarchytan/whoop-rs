//! Staged sleep pipeline: nineteen numbered steps, runnable to any point.
//!
//! [`run_to`] executes steps `1..=upto` and records a [`StepTrace`] per step. [`PipelineState`]
//! keeps each step's artifact plus a per-prefix digest, so a golden pins `1..=k` and the earliest
//! moved prefix names the step that changed.
//!
//! The order is a DAG, not a line: [`StepId::deps`] says what each step reads and
//! [`StepId::feedback`] says where it reaches forward. [`StepId::Anchor`] probe-decodes to find sleep
//! onset, so it runs [`StepId::Transition`] and [`StepId::Decode`] early; a prefix that stops at it
//! contains a decode.
//!
//! [`SleepConfig::shipped`] reproduces `v2::stage_with` — the UNREFINED epoch path, not `analyze`'s
//! refined output. Steps with no artifact of their own are pass-throughs that still trace; steps
//! 3..=9 currently fold into [`StepId::Assemble`], which digests the epoch grid and not the feature
//! values.

use super::v2::{anchor_of, emissions_at, epoch_starts, prepare, viterbi, Anchor, Prepared};
use super::{is_stageable, params::Params, SleepInput, SleepStage};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(FNV_PRIME)
}

/// Pipeline steps in execution order; `as u8` is the step number. Read [`StepId::deps`] before
/// scheduling any of them: 4..=9 are a DAG, not a parallel group.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum StepId {
    Validate = 1,
    Window = 2,
    Condition = 3,
    FeatMotion = 4,
    FeatTurn = 5,
    FeatCardiac = 6,
    FeatThermal = 7,
    FeatTime = 8,
    FeatInteract = 9,
    Assemble = 10,
    Impute = 11,
    Normalise = 12,
    Anchor = 13,
    Emit = 14,
    Transition = 15,
    Decode = 16,
    Refine = 17,
    Quality = 18,
    Report = 19,
}

impl StepId {
    pub const ALL: [StepId; 19] = [
        StepId::Validate,
        StepId::Window,
        StepId::Condition,
        StepId::FeatMotion,
        StepId::FeatTurn,
        StepId::FeatCardiac,
        StepId::FeatThermal,
        StepId::FeatTime,
        StepId::FeatInteract,
        StepId::Assemble,
        StepId::Impute,
        StepId::Normalise,
        StepId::Anchor,
        StepId::Emit,
        StepId::Transition,
        StepId::Decode,
        StepId::Refine,
        StepId::Quality,
        StepId::Report,
    ];

    /// `None` outside 1..=19; a clamped range would run a different pipeline than the one asked for.
    pub fn from_num(n: u8) -> Option<StepId> {
        StepId::ALL.into_iter().find(|s| *s as u8 == n)
    }

    /// What this step reads. `FeatInteract` is where the cross-family columns belong: it needs both
    /// `FeatMotion` and `FeatCardiac`, and neither of those needs the other.
    pub fn deps(self) -> &'static [StepId] {
        use StepId::*;
        match self {
            Validate | Transition => &[],
            Window => &[Validate],
            Condition => &[Window],
            FeatMotion | FeatTurn | FeatCardiac | FeatThermal => &[Condition],
            FeatTime => &[Window],
            FeatInteract => &[FeatMotion, FeatCardiac],
            Assemble => &[FeatMotion, FeatTurn, FeatCardiac, FeatThermal, FeatTime, FeatInteract],
            Impute => &[Assemble],
            Normalise => &[Impute],
            Anchor => &[Normalise],
            Emit => &[Normalise, Anchor],
            Decode => &[Emit, Transition],
            Refine => &[Decode],
            Quality => &[Decode],
            Report => &[Refine, Quality],
        }
    }

    /// Steps this one runs out of order. `Anchor` stages once to find onset, so it reaches forward;
    /// every other step reads only its [`deps`](StepId::deps).
    pub fn feedback(self) -> &'static [StepId] {
        match self {
            StepId::Anchor => &[StepId::Transition, StepId::Decode],
            _ => &[],
        }
    }

    /// Whether this step records an artifact of its own, or only traces.
    pub fn implemented(self) -> bool {
        matches!(
            self,
            StepId::Validate | StepId::Assemble | StepId::Anchor | StepId::Emit | StepId::Decode
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            StepId::Validate => "validate",
            StepId::Window => "window",
            StepId::Condition => "condition",
            StepId::FeatMotion => "feat_motion",
            StepId::FeatTurn => "feat_turn",
            StepId::FeatCardiac => "feat_cardiac",
            StepId::FeatThermal => "feat_thermal",
            StepId::FeatTime => "feat_time",
            StepId::FeatInteract => "feat_interact",
            StepId::Assemble => "assemble",
            StepId::Impute => "impute",
            StepId::Normalise => "normalise",
            StepId::Anchor => "anchor",
            StepId::Emit => "emit",
            StepId::Transition => "transition",
            StepId::Decode => "decode",
            StepId::Refine => "refine",
            StepId::Quality => "quality",
            StepId::Report => "report",
        }
    }
}

/// One step's outcome. `digest` hashes that step's artifact; 0 means the step produced none.
#[derive(Clone, Copy, Debug)]
pub struct StepTrace {
    pub step: StepId,
    pub digest: u64,
    pub note: &'static str,
}

/// Accumulating run state. `reached` is the last step executed.
pub struct PipelineState {
    pub reached: Option<StepId>,
    /// Epoch features from step 10; `None` before it runs.
    pub prepared: Option<Prepared>,
    /// Time-of-night anchor from step 13.
    pub(super) anchor: Option<Anchor>,
    /// Per-epoch log-emissions from step 14.
    pub emissions: Option<Vec<[f64; 4]>>,
    /// Labels from step 16.
    pub stages: Option<Vec<SleepStage>>,
    pub trace: Vec<StepTrace>,
}

impl PipelineState {
    fn new() -> Self {
        PipelineState {
            reached: None,
            prepared: None,
            anchor: None,
            emissions: None,
            stages: None,
            trace: Vec::new(),
        }
    }

    /// Replaces any existing entry for `step`, so re-running one step cannot double-count.
    fn record(&mut self, step: StepId, digest: u64, note: &'static str) {
        self.reached = Some(step);
        let t = StepTrace { step, digest, note };
        match self.trace.iter_mut().find(|e| e.step == step) {
            Some(e) => *e = t,
            None => self.trace.push(t),
        }
    }

    /// FNV-1a over `(step id, digest)` of steps `1..=upto`. What a prefix golden stores.
    /// Mixing the id means a prefix names which steps it covers, not just how many.
    pub fn prefix_digest(&self, upto: StepId) -> u64 {
        let mut h = FNV_OFFSET;
        for t in self.trace.iter().filter(|t| t.step <= upto) {
            h = fnv(h, t.step as u64);
            h = fnv(h, t.digest);
        }
        h
    }

    /// That step's own artifact digest, not the prefix.
    pub fn digest_of(&self, step: StepId) -> Option<u64> {
        self.trace.iter().find(|t| t.step == step).map(|t| t.digest)
    }
}

/// FNV-1a over the label sequence. Fixed indices, not `STAGE_ORDER`: a reordering there must not
/// silently change what a stored digest means. Pinned by `stage_index_map_is_frozen`.
fn digest_stages(v: &[SleepStage]) -> u64 {
    v.iter().fold(FNV_OFFSET, |h, s| {
        let k = match s {
            SleepStage::Wake => 0u64,
            SleepStage::Light => 1,
            SleepStage::Deep => 2,
            SleepStage::Rem => 3,
        };
        fnv(h, k)
    })
}

/// Quantised so the digest survives a last-bit float change but not a real one.
fn digest_emissions(em: &[[f64; 4]]) -> u64 {
    em.iter().flatten().fold(FNV_OFFSET, |h, v| fnv(h, (v * 1e6).round() as i64 as u64))
}

/// The epoch grid only. Feature VALUES are first pinned at step 14, because `prepare` returns them
/// behind an opaque `Prepared`.
fn digest_grid(prep: &Prepared) -> u64 {
    let starts = epoch_starts(prep);
    starts.iter().fold(fnv(FNV_OFFSET, starts.len() as u64), |h, s| fnv(h, *s as u64))
}

fn digest_anchor(a: Anchor) -> u64 {
    let (tag, v) = match a {
        Anchor::Probe => (0u64, 0u64),
        Anchor::Window => (1, 0),
        Anchor::Onset(i) => (2, i as u64),
    };
    fnv(fnv(FNV_OFFSET, tag), v)
}

/// Coarse shape of the input, enough that a different night gives a different step-1 digest.
fn digest_input(input: &SleepInput, ok: bool) -> u64 {
    let mut h = fnv(FNV_OFFSET, u64::from(ok));
    h = fnv(h, input.start as u64);
    h = fnv(h, input.end as u64);
    h = fnv(h, input.accel.len() as u64);
    h = fnv(h, input.hr.len() as u64);
    fnv(h, input.rr.len() as u64)
}

/// Which implementation each swappable component uses. A variant is a config, not a new harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SleepConfig {
    pub emit: EmitCfg,
}

/// Emission model. `V2` is the shipped recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EmitCfg {
    #[default]
    V2,
}

impl Default for SleepConfig {
    fn default() -> Self {
        SleepConfig::shipped()
    }
}

impl SleepConfig {
    /// Reproduces `v2::stage_with` label-for-label. Variants are named, not inherited from
    /// `#[default]`, so moving a default cannot silently redefine the control.
    pub fn shipped() -> Self {
        SleepConfig { emit: EmitCfg::V2 }
    }
}

/// Run steps `1..=upto`. Steps without an artifact trace with digest 0.
pub fn run_to(input: &SleepInput, cfg: &SleepConfig, p: &Params, upto: StepId) -> PipelineState {
    let mut st = PipelineState::new();

    for step in StepId::ALL.into_iter().filter(|s| *s <= upto) {
        match step {
            StepId::Validate => {
                let ok = is_stageable(input);
                st.record(step, digest_input(input, ok), "contract check");
            }
            // `prepare` conditions the channels and extracts every family, so 3..=9 land here too.
            StepId::Assemble => {
                let prep = prepare(input, p);
                let d = digest_grid(&prep);
                st.prepared = Some(prep);
                st.record(step, d, "epoch grid; 3..9 fold in");
            }
            // Reaches forward: an onset anchor is chosen by a probe decode under `p.transition`.
            StepId::Anchor => {
                let a = st.prepared.as_ref().map(|prep| anchor_of(prep, p));
                let d = a.map_or(0, digest_anchor);
                st.anchor = a;
                st.record(step, d, "probe decode");
            }
            StepId::Emit => {
                let em = match (st.prepared.as_ref(), st.anchor) {
                    (Some(prep), Some(a)) => match cfg.emit {
                        EmitCfg::V2 => emissions_at(prep, p, a),
                    },
                    _ => Vec::new(),
                };
                let d = digest_emissions(&em);
                st.emissions = Some(em);
                st.record(step, d, "v2 emissions");
            }
            StepId::Decode => {
                let labels = match st.emissions.as_ref() {
                    Some(em) => viterbi(em, &p.transition),
                    None => Vec::new(),
                };
                let d = digest_stages(&labels);
                st.stages = Some(labels);
                st.record(step, d, "viterbi");
            }
            // 11..=12 run inside `emissions_at`; 15 is a constant; 17..=19 do not exist yet.
            _ => st.record(step, 0, "pass-through"),
        }
    }
    st
}

/// All nineteen steps.
pub fn run(input: &SleepInput, cfg: &SleepConfig, p: &Params) -> PipelineState {
    run_to(input, cfg, p, StepId::Report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::input::{AccelSample, HrSample};

    /// Quiet night with scattered restless epochs early. A sticky transition smooths them into sleep
    /// and finds an early onset; a uniform one leaves them wake and finds a later one.
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
            hr: (0..dur)
                .map(|i| HrSample { ts: start + i, bpm: if spike(i) { 72 } else { 52 } })
                .collect(),
            rr: Vec::new(),
        }
    }

    /// `cycle_rem_onset_minutes = 0` takes the window branch and skips the probe entirely.
    fn windowed_params() -> Params {
        let mut w = Params::SHIPPED;
        w.cycle_rem_onset_minutes = 0.0;
        w.cycle_clock_from_onset = false;
        w
    }

    #[test]
    fn step_numbers_are_one_to_nineteen_and_ordered() {
        for (i, s) in StepId::ALL.iter().enumerate() {
            assert_eq!(*s as u8, i as u8 + 1, "{s:?} is out of position");
        }
        assert!(StepId::ALL.windows(2).all(|w| w[0] < w[1]), "StepId must sort by step number");
        assert_eq!(StepId::ALL.len(), StepId::Report as usize, "every step must be in ALL");
    }

    #[test]
    fn from_num_rejects_out_of_range_rather_than_clamping() {
        assert_eq!(StepId::from_num(1), Some(StepId::Validate));
        assert_eq!(StepId::from_num(19), Some(StepId::Report));
        assert_eq!(StepId::from_num(0), None);
        assert_eq!(StepId::from_num(20), None);
    }

    /// The numbering is only meaningful if every read is of an earlier step. One step reaches
    /// forward and it must be the declared one.
    #[test]
    fn deps_point_backwards_and_the_only_forward_reach_is_declared() {
        for s in StepId::ALL {
            for d in s.deps() {
                assert!(*d < s, "{s:?} reads {d:?}, which runs later");
            }
            for f in s.feedback() {
                assert!(*f > s, "{s:?} declares {f:?} as feedback but it runs earlier");
            }
        }
        let reaching: Vec<StepId> =
            StepId::ALL.into_iter().filter(|s| !s.feedback().is_empty()).collect();
        assert_eq!(vec![StepId::Anchor], reaching, "exactly one step may reach forward");
    }

    /// The cross-family columns need both families, and putting them in either parent forces that
    /// parent to read the other. Only a step of their own leaves 4 and 6 independent.
    #[test]
    fn the_interaction_step_needs_both_families_and_neither_family_needs_the_other() {
        assert!(StepId::FeatInteract.deps().contains(&StepId::FeatMotion));
        assert!(StepId::FeatInteract.deps().contains(&StepId::FeatCardiac));
        assert!(!StepId::FeatMotion.deps().contains(&StepId::FeatCardiac));
        assert!(!StepId::FeatCardiac.deps().contains(&StepId::FeatMotion));
    }

    #[test]
    fn prefix_digest_depends_on_the_digest_values_not_just_the_step_count() {
        let mut a = PipelineState::new();
        a.record(StepId::Validate, 11, "");
        a.record(StepId::Window, 22, "");
        let mut b = PipelineState::new();
        b.record(StepId::Validate, 11, "");
        b.record(StepId::Window, 99, "");
        assert_ne!(
            a.prefix_digest(StepId::Window),
            b.prefix_digest(StepId::Window),
            "same steps with different artifacts must not share a prefix digest"
        );
    }

    #[test]
    fn prefix_digest_covers_only_the_prefix() {
        let mut st = PipelineState::new();
        st.record(StepId::Validate, 11, "");
        st.record(StepId::Window, 22, "");
        st.record(StepId::Condition, 33, "");
        let a = st.prefix_digest(StepId::Window);
        assert_ne!(a, st.prefix_digest(StepId::Condition), "a further step must move the digest");

        let mut short = PipelineState::new();
        short.record(StepId::Validate, 11, "");
        short.record(StepId::Window, 22, "");
        assert_eq!(a, short.prefix_digest(StepId::Window), "same prefix, same digest");
    }

    #[test]
    fn re_recording_a_step_replaces_it_rather_than_appending() {
        let mut st = PipelineState::new();
        st.record(StepId::Validate, 11, "");
        let first = st.prefix_digest(StepId::Validate);
        st.record(StepId::Validate, 11, "");
        assert_eq!(1, st.trace.len(), "a re-run must not push a second entry");
        assert_eq!(first, st.prefix_digest(StepId::Validate), "and must not move the digest");
    }

    #[test]
    fn stage_index_map_is_frozen() {
        use SleepStage::*;
        // Changing the Wake/Light/Deep/Rem index map rebases every stored digest.
        assert_eq!(0x4475_327f_98e0_5411, digest_stages(&[Wake, Light, Deep, Rem]));
    }

    /// Step 13 is the feedback edge, so it must carry an artifact that moves when the branch does.
    /// A shared digest would mean the step recorded nothing and the cycle stayed hidden.
    #[test]
    fn the_anchor_step_records_which_branch_ran() {
        let input = restless_start_night();
        let a = run_to(&input, &SleepConfig::shipped(), &Params::SHIPPED, StepId::Anchor);
        let b = run_to(&input, &SleepConfig::shipped(), &windowed_params(), StepId::Anchor);
        assert!(a.prepared.is_some(), "the anchor needs epochs, so step 10 must have run");
        assert_ne!(
            a.digest_of(StepId::Anchor),
            b.digest_of(StepId::Anchor),
            "the probe branch and the window branch must not share an anchor digest"
        );
        assert!(a.emissions.is_none(), "stopping at 13 must not emit");
    }

    /// The emission is a function of the anchor, so it is not a pure function of the features and a
    /// cached prefix through 14 is only valid for the params that chose the anchor.
    #[test]
    fn emissions_depend_on_the_probe_anchor() {
        let input = restless_start_night();
        let windowed = windowed_params();
        let a = run_to(&input, &SleepConfig::shipped(), &Params::SHIPPED, StepId::Emit);
        let b = run_to(&input, &SleepConfig::shipped(), &windowed, StepId::Emit);

        assert_ne!(
            Params::SHIPPED.cycle_rem_onset_minutes, windowed.cycle_rem_onset_minutes,
            "the two arms must differ in the anchor branch they take"
        );
        let n = a.emissions.as_ref().map(Vec::len).unwrap_or(0);
        assert!(n > 100, "the night must produce epochs to compare; got {n}");
        assert_eq!(
            a.digest_of(StepId::Assemble),
            b.digest_of(StepId::Assemble),
            "the two arms must share their features, or the anchor is not what moved the emission"
        );
        assert_ne!(
            a.digest_of(StepId::Emit),
            b.digest_of(StepId::Emit),
            "the probe anchor must change what step 14 emits"
        );
    }

    #[test]
    fn digest_distinguishes_order_and_length() {
        use SleepStage::*;
        assert_ne!(digest_stages(&[Wake, Light]), digest_stages(&[Light, Wake]));
        assert_ne!(digest_stages(&[Wake]), digest_stages(&[Wake, Wake]));
    }
}
