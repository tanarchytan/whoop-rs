//! The V2 recipe's tunable surface, gathered in one struct so the emission weights, gates and priors are
//! read (and swept) in one place instead of scattered as literals. [`Params::SHIPPED`] is the tuned recipe
//! the stager uses by default; `stage` with anything else is the tuning path only.

/// Every coefficient the V2 stager reads. Field order mirrors the emission it feeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Deep emission: HR-variability, HR and motion weights.
    pub deep_hrv: f64,
    pub deep_hr: f64,
    pub deep_motion: f64,
    /// REM emission: HR-variability, motion and HR weights.
    pub rem_hrv: f64,
    pub rem_motion: f64,
    pub rem_hr: f64,
    /// Awake emission: motion weight, plus the dead-zoned cardiac pair.
    pub awake_motion: f64,
    pub awake_hrv: f64,
    pub awake_hr: f64,
    /// Below this |z| the cardiac terms read as noise and contribute nothing to the awake emission.
    pub awake_deadzone: f64,
    /// Deep-eligibility HR-flatness percentile gate and the slope of the penalty past it.
    pub deep_gate_thresh: f64,
    pub deep_gate_slope: f64,
    /// Multiples of the night's median jerk: the per-sample movement threshold, and the ceiling under
    /// which an epoch counts as quiescent.
    pub jerk_move_mult: f64,
    pub jerk_gate_mult: f64,

    /// Heart-rate z above which the motion-quiescent clamp does NOT apply.
    ///
    /// The clamp silences the awake cardiac term whenever the wrist is still, which is right
    /// mid-night (it stops cardiac noise inventing wake) and wrong before sleep onset (lying still
    /// and awake is exactly the case it hides). Measured on PSG: with the clamp always on, DREAMT
    /// onset lands 112 min early; with it almost never on, 22 min - and kappa falls 0.11 because
    /// phantom wake returns.
    ///
    /// This is the middle: keep the clamp, but let a body whose heart rate sits well above ITS OWN
    /// night mean speak anyway. `f64::INFINITY` reproduces the always-clamp behaviour exactly, and is
    /// what SHIPPED still carries - 0.5 is measured better on two of three cohorts but moves all three
    /// parity constants, so adopting it is a re-baseline and not a parameter edit.
    pub quiescent_hr_z_max: f64,

    /// Apply the motion-quiescent clamp ONLY to epochs with no R-R behind them.
    ///
    /// The clamp guards against a noisy cardiac term. R-R presence is read off `resp_reg`, which is
    /// the ONLY R-R-fed feature - `hr_var` is the per-second heart-rate standard deviation and is
    /// present with or without beats, so it cannot answer this. Measured: disabling the clamp outright gains
    /// kappa on both R-R cohorts (DREAMT +0.014, AAUWSS +0.019) and loses on the one with none
    /// (sleep-accel -0.016), which is the split this switch follows. `false` is the old behaviour.
    pub clamp_only_without_rr: bool,
    /// Added to the awake emission when peak jerk clears the gate multiple.
    pub motion_gate_boost: f64,
    /// Weight of the RSA respiration-regularity term (added to deep, subtracted from REM).
    pub resp_weight: f64,
    /// Stage base rates, in probability (logged before use), in [deep, rem, light, awake] order.
    pub base_rate: [f64; 4],
    /// Sleep-cycle prior: deep scale + the fraction of the night it decays over; REM scale, the early
    /// fraction it is suppressed in (read only when the guard is window-anchored), and the size of that
    /// suppression.
    pub cycle_deep_scale: f64,
    pub cycle_deep_decay: f64,
    pub cycle_rem_scale: f64,
    pub cycle_rem_early_frac: f64,
    pub cycle_rem_early_penalty: f64,
    /// Zero keeps the early-REM suppression a step below `cycle_rem_early_frac` of the session; a
    /// positive value grades the same magnitude to zero over that many minutes past detected onset.
    pub cycle_rem_onset_minutes: f64,
    /// Ceiling on the REM ramp's time-of-night input, so the prior stops growing past this fraction.
    pub cycle_rem_ramp_cap: f64,
    /// Measure time-of-night from detected sleep onset instead of from the window start, so bedtime
    /// latency stops shifting every epoch's position in the night. Costs a second staging pass.
    pub cycle_clock_from_onset: bool,
    /// Sticky transition matrix (rows = from, cols = to) in [deep, rem, light, awake] order.
    pub transition: [[f64; 4]; 4],
}

impl Params {
    /// The tuned recipe the stager ships. Every value here is measured, not chosen: changing one moves
    /// the benchmark, so treat it as data and re-run the fixture sheet after any edit.
    pub const SHIPPED: Params = Params {
        deep_hrv: -0.8,
        deep_hr: 0.5,
        deep_motion: -0.1,
        rem_hrv: 0.8,
        rem_motion: -0.4,
        rem_hr: 0.4,
        awake_motion: 1.0,
        awake_hrv: 0.5,
        awake_hr: 0.6,
        awake_deadzone: 0.30,
        deep_gate_thresh: 0.40,
        deep_gate_slope: 5.0,
        jerk_move_mult: 75.0,
        jerk_gate_mult: 35.0,
        quiescent_hr_z_max: f64::INFINITY,
        clamp_only_without_rr: false,
        motion_gate_boost: 4.0,
        resp_weight: 0.6,
        base_rate: [0.15, 0.22, 0.50, 0.34],
        cycle_deep_scale: 1.2,
        cycle_deep_decay: 0.55,
        cycle_rem_scale: 1.0,
        cycle_rem_early_frac: 0.12,
        cycle_rem_early_penalty: 4.0,
        cycle_rem_onset_minutes: 60.0,
        cycle_rem_ramp_cap: 1.0,
        cycle_clock_from_onset: false,
        transition: [
            [0.76, 0.012, 0.216, 0.012],
            [0.00333, 0.92, 0.06667, 0.01],
            [0.08, 0.08, 0.80, 0.04],
            [0.0, 0.0, 0.10, 0.90],
        ],
    };

    /// Stage base rates in the log domain the emissions add into.
    pub(super) fn base_log_prior(&self) -> [f64; 4] {
        [
            self.base_rate[0].ln(),
            self.base_rate[1].ln(),
            self.base_rate[2].ln(),
            self.base_rate[3].ln(),
        ]
    }
}

impl Default for Params {
    fn default() -> Self {
        Params::SHIPPED
    }
}
