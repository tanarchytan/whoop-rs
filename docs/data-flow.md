# Data & algorithm flow — whoop-rs (backend) + noop-tan (frontend)

Two projects, one contract. **whoop-rs owns every algorithm, decode and score. noop-tan owns
presentation, persistence, BLE transport and user policy.** The only seam is the uniffi FFI.
`RustScores.kt` is the main adapter and holds most of the crossings, but it is **not** the only file
that crosses: **36 hand-written Kotlin files under `main/` reach `uniffi.whoop_ffi` directly**,
including two screens. Counted by `tools/docs-vs-code.py`, so the day a screen stops calling Rust —
or a new one starts — the number moves and this page has to say so.

Re-counted 2026-10-09: **128 exported functions, all 128 called from hand-written app Kotlin**
(`is_rr_fill` gained its app caller when live R-R filler started being flagged at parse time). The `WhoopCodec`
object carries a further **32 methods**, and `protocol/RustCodec.kt` is the only hand-written Kotlin file
that names the type at all — it constructs both codecs `private`ly and exposes nothing that returns one, so
a codec cannot reach any other file. Which of the 32 it calls is **named, not counted**: a free function is
qualified by `uniffi.whoop_ffi.<name>` and a namesake cannot reach it, an object method has no such
namespace, and four matchers in a row answered this one with a wrong integer. A wrong name is legible where
a total moving from 16 to 18 is not.

**Called from Kotlin** — `new`, `decode_history`, `decode_live`, `decode_response`, `decode_metadata`,
`decode_ppg_frame`, `command_frame`, `get_battery_frame`, `buzz_frame_with`, `set_clock_frame`,
`set_clock_legacy_frame`, `alarm_set_frame`, `alarm_set_frame_gen4`, `alarm_disable_frame`,
`advertising_name_frame`, `set_config_frame`.

**Not called** — `client_hello`, `r22_frames`, `feed`, `reset`, `offload_start`, `offload_abort`,
`decode_imu_frame`, `get_hello_frame`, `get_data_range_frame`, `stop_raw_flood_frame`,
`toggle_realtime_hr_frame`, `reboot_frame`, `broadcast_hr_frame`, `run_haptics_frame`,
`advertising_name_frame_gen5`, `buzz_frame` (the one-pulse form). Kept as codec-parity API for the CLI and a later call site.

Those are three different things, and only the last is awaiting a call site:

- **A Kotlin twin holds the same wire bytes.** `client_hello` and `r22_frames`: `DeviceFamily.WHOOP5.clientHello`
  is `aa0108000001e67123019101363e5c8d`, the same 16 bytes as `GEN5_CLIENT_HELLO`, and
  `Whoop5Config.enableR22Sequence` is the same 16 flags in the same order as `config::R22_SEQUENCE`.
- **A Kotlin twin holds the same state machine.** `feed`, `reset`, `offload_start` and `offload_abort` are the
  sans-IO `Offload` door. The app reassembles in `protocol/Framing.kt` and drives the drain in
  `ble/Backfiller.kt`, calling `decode_history` / `decode_metadata` per frame instead, so the Rust machine
  runs for `whoopctl` only.
- The rest are frame builders and `decode_imu_frame`, with no Kotlin equivalent yet.

The Kotlin copies are the ones with callers, so each table and each machine exists twice and only one side
ships. Whether Kotlin should call across for them is a border decision, not drift.

The tables below carry **51** of them; **62 of the 128 appear in no shipped document**, which is a gap in the
map rather than a claim about the code. `tools/docs-vs-code.py` re-derives every count here and fails on
drift, and for the two lists above it prints the symmetric difference — which name appeared, which vanished.
It also refuses to guess: a call to a codec method name whose receiver it cannot resolve is reported as
unresolved rather than counted, which is what `reassembler.feed(bytes)` was when it published 18.

## The universal rule

> A number the user sees is computed in whoop-rs. Kotlin decides *when* to ask, *what to store*, and
> *how to word it* — never *what the number is*.

Concretely, a thing belongs in **whoop-rs** if it is any of: a decode of strap bytes, a statistic, a
score, a gate/threshold on physiology, or a tuning constant that feeds one. It belongs in **noop-tan**
if it is any of: a BLE connection, a Room row, a screen, a string the user reads, a preference, or a
policy the user sets (recalibration date, profile fallbacks, dismissal state).

## Logic flow — strap byte to rendered number

```
  STRAP (BLE notify)
        |                                     noop-tan owns the radio
        v
  WhoopBleClient ................... reassembles notifications
        |
        v
  WhoopCodec (FFI) ................. whoop-rs: deframe, CRC, decode records
        |                            -> HistoryRecord / PpgRecord / ImuRecord / OpticalRecord
        v
  WhoopRepository (Room) ........... noop-tan: persist raw per-second streams
        |
        v
  AnalyticsEngine / IntelligenceEngine
        |                            noop-tan: assemble a day's streams, decide WHICH window
        |                            and WHICH device owns the day
        v
  RustScores.kt .................... the main adapter: one thin wrapper per engine, no arithmetic
                                     (16 other main-source files also cross, two of them screens)
        |
        v
  whoop-rs physio-algo ............. every score, gate and statistic
        |                            -> DailyMetric fields, SleepSession, ExerciseSession
        v
  DailyMetric (Room) .............. noop-tan: one row per day, per device
        |
        v
  Screens ......................... noop-tan: wording, colour, layout, empty states
```

Two rails run alongside it:

- **Baselines.** Each nightly value folds into a per-metric EWMA baseline (`baseline_update` /
  `baseline_fold_history`). The tuning table (`baseline_metric_cfg`) lives in Rust so the app cannot
  hold a copy that drifts. Kotlin owns only the user's *recalibration epoch* — a policy, not maths.
- **Provenance.** A day can be sourced from the strap, an Apple Health import or Health Connect;
  `FusionResolver` picks a winner per field. That is arbitration, not computation, so it stays Kotlin.

## Device families

`DailyMetric.spo2Pct` is **family-agnostic by design but not equal-provenance**:

| family | how the percent is produced |
|---|---|
| 5.0 / MG | the strap's own computed SpO2 rides v18; noop-tan takes the in-bed median |
| 4.0 | no strap value exists. The paired red/IR ADC goes through ratio-of-ratios in whoop-rs, but a pulsatility gate **withholds** the result — see below |

In practice only the 5.0/MG figure reaches the field. On 4.0 the pair arrives at 1 Hz, so the 0.5 Hz
Nyquist limit sits below the 0.83-3.0 Hz cardiac band and the pulsatile component is aliased away before
decode. `spo2.rs` requires `MIN_PULSATILE_FRACTION` (0.5) of the eligible 30 s windows to be pulsatile and
returns `None` otherwise; measured on two straps and 2.1M samples, 89.3% and 98.2% of windows carried zero
red amplitude, so the gate declines. What the app stores on 4.0 is the raw red/IR ADC means, never a
percent. Withheld is not the same as uncalibrated: there is no weak number to treat with caution.

Everything else is family-independent: the record decoders fork by version byte inside whoop-rs, and
every algorithm above takes plain values.

## The whoop-rs surface

### Decode / wire

| export | what it computes |
|---|---|
| `data_range_newest` | Newest plausible unix banked, scanning EVERY byte offset of a GET_DATA_RANGE frame and preferring the newest non-future word (falls back to newest-any). This is the sync gate — it REPLACES the fixed-offset `Response::DataRange` newest read. |
| `data_range_oldest` | Oldest plausible unix banked (backlog depth), scanning only the aligned-from-7 grid (asymmetric with the newest scan by design, to dodge a WHOOP-4 straddle word). |
| `ppg_hr` | HR from a v26 optical PPG buffer (24 Hz autocorrelation). |
| `standard_rr_ms` | One standard BLE heart-rate (0x2A37) R-R word in ms: 1/1024 s per the spec (4.0, rounded), or plain ms passed through (5/MG). |
| `rr_fill_cfg` | The R-R filler constants (500 ms, same-second HR < 100) the app binds into its v103 `tsSuspect` SQL, so it holds no copy. |
| `is_rr_fill` | The filler predicate itself; the app's SQL is pinned to it by a test grid. Flagged rows are skipped by scoring reads, never deleted. |

### HRV

| export | what it computes |
|---|---|
| `hrv_rmssd_gap_aware` | Gap-aware, artifact-corrected nightly RMSSD (ms) from per-record R-R runs. |
| `hrv_windowed_avg` | Windowed session avgHrv (ms): the mean of per-5-min-bucket gap-aware RMSSD over `[start, end]`, the app's stored `SleepSession.avgHrv`. `runs` are the session's per-record `(unix, rr)` in chronological order; buckets tumble from `start`. `None` when no bucket yields a value. |
| `hrv_readiness` | HRV-readiness over a nightly RMSSD series (oldest → newest; `None` slots = missing nights). |
| `hrv_rmssd` | RMSSD (ms) of raw R-R values. `None` for <2 beats (filtered, drops deltas >200ms). |
| `hrv_rmssd_plain` | Plain unfiltered RMSSD (no artifact rejection); the raw counterpart to `hrv_rmssd`. |
| `hrv_range_filter` | Range-filter R-R values, keeping only 300–2000 ms. |
| `hrv_sdnn` | Standard deviation of NN intervals (ms), sample std (ddof=1). `None` for <2 values. |
| `hrv_analyze_raw` | Clean-and-analyze a raw R-R series in one call (the app's full spot/nightly HRV analysis path). |
| `hrv_freq_domain` | Frequency-domain HRV over a time-ordered R-R series (ms) via the Lomb-Scargle periodogram. |
| `hrv_nightly` | THE nightly HRV value (ms) a readiness word is computed from: per-5min-bucket gap-aware RMSSD over deep-sleep (SWS/N3) buckets only, from reports the strap's optical front end did not flag. Takes the full segment list and filters for `SleepStage::Deep` internally. `None` when no deep bucket yields one. |

### Sleep

| export | what it computes |
|---|---|
| `analyze_sleep_with` | Detect + stage a night's streams: one call carves the in-bed spans and returns one session each, staged by `engine` (`Original` = v2; `Experimental` = v2 + `clamp_only_without_rr` + far-from-edge abstention, whose refused stretches come back as `unscored` spans, drawn only). |
| `stage_sleep_refined_with` | Stage one already-detected in-bed span with `engine` + motion-aware wake refinement (the single-span edit self-heal path): per-30 s-epoch segments over `[start, end]`, plus the `unscored` spans. |
| `rest_score` | Rest (sleep performance) composite [0, 100] from a night's aggregates. `None` when there is no asleep time. Absent `sleep_need_hours` defaults to 8 h; absent `consistency` defaults to a neutral 0.5. |
| `sleep_debt_ledger` | Rolling sleep-debt ledger: Σ(slept − need) over the last `window` (default 14) nights with data. `need_hours` defaults to 8 h. Nights with no sleep are skipped, never zero-filled. |
| `personal_sleep_need_hours` | Personal sleep need (hours) = mean of recent nightly asleep hours, floored at 7.5. For the Rest score's sleep-need input. |
| `nap_evaluate` | Classify one candidate window for a short nap (tri-state, conservative — only PROPOSES a review card). |

### Recovery / effort

| export | what it computes |
|---|---|
| `recovery_score` | Recovery "Charge" score in [0, 100]. `None` at cold-start or when no driver is available. |
| `recovery_band` | Recovery colour band ("red" / "yellow" / "green") for a score. |
| `recovery_index_slope` | Overnight HR-decline slope (bpm/hour) — the recovery-index driver. |
| `recovery_banked_nights` | Count of nights carrying a usable nightly HRV — the calibration-progress count. |
| `strain_score` | Cardiovascular Effort (0–100) from an HR series. `None` without enough data or when HRR ≤ 0. |
| `strain_default_denominator` | The default strain denominator (log-map scale onto 0–100). |

### Heart rate

| export | what it computes |
|---|---|
| `session_resting_hr` | Lowest 5-min tumbling-window mean bpm floor over `[start, end]`. `None` with no samples. |
| `daily_resting_hr` | Daily resting HR = min of the per-session floors. |
| `hr_zones_for_age` | Age-derived (Tanaka) HR zones, or a manual max-HR override. |
| `hr_time_in_zone` | Seconds spent in each HR zone over an HR series, using age-derived (or override) zones. |

### Respiration / oxygen

| export | what it computes |
|---|---|
| `resp_rate_from_rr` | Respiratory rate (breaths/min) from R-R via RSA. `None` when the signal is too thin. |
| `spo2_from_paired` | SpO2 (%) from a 4.0 paired red/IR window (ratio-of-ratios). `None` if not pulsatile. |
| `nightly_spo2_raw_means` | Nightly integer-truncated means of the 4.0 raw red/IR PPG ADC over the detected in-bed `spans`, the app's stored `DailyMetric.spo2Red`/`spo2Ir`. A sample counts when its `ts` lies inside any span. `None` when either input is empty or no sample landed in-span. Raw ADC only, never a calibrated percent. |

### Stress

| export | what it computes |
|---|---|
| `stress_index` | Baevsky Stress Index from a raw R-R series (ms). `None` on too-few beats or a degenerate range. |
| `stress_components` | Full SI components from a raw R-R series (ms). |
| `daily_stress` | Daily autonomic stress (0–3) from today's RHR + HRV against the prior-days baseline. `None` on too few baseline days or no signal today. |
| `daytime_stress` | Score waking hours for autonomic activation against the day's own calm-hour quartiles (Q25 HR, Q75 RMSSD). Each hour needs its own HR gate applied by the caller (a `None` mean_hr hour is skipped). |
| `sleep_stress` | Score one sleep window's buckets on the same formula and the same 0–3 bands, with no hour-of-day filter. The caller passes ONLY the buckets inside the span, because a night crosses midnight. |

### Motion / energy

| export | what it computes |
|---|---|
| `imu_features` | Accel/gyro energy, jerk and gait-band cadence over IMU samples at `sample_rate_hz`. |
| `steps_counter` | Raw wrap-aware motion-tick total from step counter samples. None with fewer than 2 samples or no forward movement. The caller applies its stepTicksPerStep calibration. |
| `calories_estimate_day` | Whole-day energy (kcal) from HR samples, split into its resting and active halves. Seconds inside a `confirmed_bouts` span are billed exactly as `calories_estimate_bout` bills them; the rest face the higher unconfirmed-HR gate. `hrmax` absent -> resolved from the day's own peak, then age. |
| `calories_estimate_bout` | Bout energy (kcal, kJ) from a motion-confirmed span's HR samples, each weighted by elapsed time to the next. `hrmax` absent -> resolved from the bout's own peak, then age. |
| `activity_series` | Per-record motion-intensity series from a gravity stream — the shared motion spine the workout, nap and sedentary readings all measure against. |
| `smoothed_intensity` | Trailing rolling mean of the motion intensities over `window_s` — the smoothed spine the stillness and sedentary gates threshold against. |

### Ages / rhythm

| export | what it computes |
|---|---|
| `vo2max_estimate` | Non-exercise VO2max estimate (ml/kg/min) from the waist-circumference model. Wellness only. |
| `fitness_age_compute` | Full Fitness Age. `None` only if RHR or age is missing. |
| `rhythm_age_from_samples` | Circadian Rhythm Age from raw (unix, activity) samples + tz offset + chronological age + sex: bins the samples per LOCAL hour, fits a single-component cosinor, then the Gompertz biological-age transform. `None` when the fit is degenerate (< 3 populated hours). v1 is a RELATIVE index; the activity scale is not calibrated to the model's mg-ENMO training units. |
| `circadian_phase_from_samples` | Body-clock phase from raw (unix, activity) samples + tz offset, days observed, habitual wake hour, and an optional observed skin-temp minimum hour. `None` when the cosinor is degenerate. |

### Baselines

| export | what it computes |
|---|---|
| `baseline_metric_cfg` | One metric's baseline configuration by name ("hrv" / "resting_hr" / "resp" / "skin_temp" "strain"). The tuning table lives here so the app cannot hold a second copy that drifts from it; an unknown name yields `None`. |

## What deliberately stays in Kotlin

| area | why |
|---|---|
| `AnalyticsEngine` / `IntelligenceEngine` | orchestration: which window, which device owns the day, what to persist |
| `FusionResolver`, `DayOwnerResolver`, `MetricArbitrationPolicy` | source arbitration across bands/imports |
| `CalibrationMilestones`, `ScoreConfidence` | presentation of how far calibration has got |
| `CircadianEngine` (planner + wording), `FitnessAgeEngine` (readiness checklist) | user-facing copy and input gating over a Rust result |
| `ReadinessEngine` | multi-signal 5-level read used by the Coupled screen; the hero pill now uses the Rust HRV tier |

| `WeeklyDigest` | week-in-review shaping; its statistics now come from Rust |

## The Gompertz constant

Two displayed ages rest on a Gompertz slope, and both deserve scrutiny:

- **Body Age** (`vitality.rs`) uses `MORTALITY_DOUBLING_YEARS = 10.0`, sourced to Richmond & Roehner
  ([arXiv:1509.07271](https://arxiv.org/abs/1509.07271)), who fit ~10 years for humans above 35 across
  national and historical series. It previously used an unsourced 8, whose citation was deleted with the
  Swift tree. **The constant is a sensitivity parameter, not a correctness one**: a person at the
  population reference on every driver sums to zero hazard and reads their own age whatever it is set to.
  It only scales the size of a deviation — moving 8 to 10 widened every Body Age offset by exactly 1.25.
  The literature spans ~7-10 and the Strehler-Mildvan correlation says the Gompertz intercept and slope
  are not independent, so no single value is universally right.
- **Rhythm Age** (`biological_age.rs`) carries `ACTIVITY_TO_MG_ENMO_SCALE` with an explicitly
  **UNVALIDATED** 1.0 transfer factor between the strap's on-chip activity units and the model's mg-ENMO
  training units. That one is still open.

## Known deltas

1. ~~4.0 SpO2 is uncalibrated~~ **Superseded: it is WITHHELD, not weak.** The generic curve constants were
   never the binding problem — the 1 Hz sample rate is. The pulsatility gate above returns `None`, so no
   4.0 percent reaches the field and no provenance flag is needed for one. `spo2.rs` and
   `algorithms.md` state it the same way.
2. ~~Vitality coefficients unverified~~ **CLOSED 2026-07-26.** All six verified against Europe PMC and
   cited in `vitality.rs` with PMIDs and quoted figures. Two things that pass found and the code now
   states: only VO2max is published as a per-unit slope (steps, HRV and sleep regularity are quartile or
   percentile contrasts that we linearise; sleep duration's per-hour figure is ours), and our
   `sleepConsistency` is NOT the Sleep Regularity Index the cohort uses, so that source calibrates
   direction and reference point but not slope. Every coefficient sits at or below the published effect
   except resting HR, which falls between two disagreeing meta-analyses — the model under-states rather
   than over-states, which is the right direction for a wellness readout.
3. **Rhythm Age's activity scale** (above) remains unvalidated against real strap data.
4. **Sleep regularity is measuring the wrong thing, and half the fix has landed.** The Vitality driver
   takes `1 - CV of nightly sleep hours`, which is DURATION regularity: someone sleeping exactly 8 h a
   night from 22:00 one day and 02:00 the next scores perfectly regular, and that is precisely the group
   Cribb's 1.53 hazard sits in. `physio-algo/src/sleep_regularity.rs` now computes the real Sleep
   Regularity Index (Phillips 2017) from the hypnogram, with non-wear treated as UNKNOWN rather than
   awake so not wearing the strap cannot read as irregularity. Still to do: feed it the app's multi-day
   epoch grid, then recalibrate the coefficient against Cribb's published percentiles instead of the
   current invented scaling.


## Open: resting HR reads systematically low

Measured against a WHOOP export over one user's nights (2026-07-26). `session_resting_hr` takes the
MINIMUM 5-minute window mean in the sleep span, and `daily_resting_hr` then takes a minimum again across
sessions. A minimum is an extremum: it is set by the single quietest five minutes and moves with any
artifact.

| estimator | MAE vs reference | correlation |
|---|---:|---:|
| min of 5-min windows (shipped) | 10.31 | 0.130 |
| 20th percentile | 3.75 | 0.596 |
| 30th percentile | 2.12 | 0.756 |
| night mean | 1.20 | 0.893 |

**The BIAS is stable and real**: -10.57 on the early half of the nights, -10.06 on the late half,
independently. It reads about 10 bpm low, consistently.

**The CORRELATION ranking is NOT established**: split in half (n=4 each) the shipped estimator's
correlation swings from -0.421 to +0.775, so this data cannot say which replacement is best. NOT CHANGED
for that reason.

CONFOUND CONTROLLED: the two readings come from two straps on two wrists, so a sensor or placement
difference was the obvious rival explanation. It is not the cause. Comparing the SAME statistic from each
band (whole-cycle mean HR) over 8 cycles gives +1.8 bpm mean, +2.3 median — the noop band reads slightly
HIGHER, the opposite direction to the -10 bpm gap. Two independent statistics off the noop stream (cycle
mean, night mean) both land within ~2 bpm of their WHOOP equivalents; only the minimum-of-windows is an
outlier. A uniform sensor offset would move all of them together.

This matters beyond the display: resting HR feeds recovery and the Vitality driver at 0.100 per 10 bpm,
so a 10 bpm offset is ~0.1 log-hazard, roughly 1.4 years of Body Age.

To settle it: more nights on a build whose scores come from the current code, then re-run the comparison.
NOTE the general rule it illustrates — verification data is only valid for the code that produced it.
Raw streams (HR, R-R, gravity, sleep spans) are portable across app versions; computed columns are not.


## Workout detection vs WHOOP: a definitional difference, not a defect

Running the CURRENT detector over the backup's raw streams finds 0 workouts on 2026-07-24/25/26, where
the WHOOP export logs 7. Checked before concluding: on 07-25, 0 of 85 753 seconds reach 60 % HRR and only
34 reach 50 % (rest 78, max 187, peak 138 bpm, mean 90).

So the runs never clear `MIN_INTENSITY_Z2PLUS` — correctly. noop's workout means a sustained cardio
effort; WHOOP logged these as Walking at strain 4.1-4.7 on its own 0-21 scale, i.e. any recorded activity.
Two different questions, both answered right. Do NOT retune the intensity gate to match WHOOP's count
without deciding first which definition the app wants.
