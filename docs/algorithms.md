# Algorithms

Source of truth for every wellness algorithm NOOP ships. All decode, scoring, and derived-metric math
lives here in `crates/physio-algo` (pure, sans-IO, deterministic); `crates/whoop-ffi` exposes the subset
the apps consume over uniffi. The Kotlin/Swift frontends compute nothing themselves that appears below,
they call these functions. See `architecture.md` for the crate map and `sleep.md` for the sleep deep-dive.

Every entry point takes plain values (R-R runs, PPG samples, accel, per-epoch fields) or an already-decoded
`HistoryRecord` slice, never a wire frame and never BLE. Absent signal returns `None`, never a fabricated
number. Outputs are wellness estimates, never medical.

`physio-algo` carries **724 unit tests** (golden vectors, parity fixtures, synthetic sweeps); the
workspace runs **1041** (measured 2026-08-17 at HEAD; re-derive, never carry forward).
**Five** sleep-dataset tests read external fixtures and are `#[ignore]`d by default - three assert a
cohort kappa, the fourth prints the whole-corpus sheet and the fifth the stream census. Run
`cargo test -p physio-algo --test dataset_parity -- --ignored` to check the published kappas.
`tools/docs-vs-code.py` re-derives every count on this page from the source and fails on drift.

Every algorithm below is also tagged with the STRENGTH of its evidence, which is not the same as whether
it is wired:

- **hardware** — exercised against a real strap's own streams.
- **gold standard** — scored against an external labelled dataset (PSG hypnograms).
- **parity** — pinned against the previous implementation's figures, which still pass.
- **literature** — coefficients cited to named studies; see each module header for PMIDs.
- **unvalidated** — implemented and unit-tested, but nothing external confirms the output yet.

## Wiring legend

Each algorithm below is tagged with how it reaches the app:

- **FFI** — exported by `whoop-ffi` and called by the app today (the app owns no copy of the math).
- **Rust-only** — implemented here, not on the FFI surface. No app caller.
- **internal** — a shared helper other algorithms depend on, not a metric.

There is no **unwired** tag because there is nothing unwired: all 125 exported functions have a Kotlin
caller (measured 2026-08-06; re-derive, never carry forward). What the app still computes itself is tracked in the noop-tan `ALGORITHMS.md`.

---

## Module map

| # | Domain | Module | Wiring | Tests |
|---|---|---|---|---|
| 1 | Sleep detection + staging | `sleep/{detect,v2,refine,mainnight,common,input}.rs` | FFI | golden + 60+ |
| 2 | HR from PPG | `ppg.rs` | FFI | 6 |
| 3 | HRV / RMSSD / readiness | `hrv.rs` | FFI | 31 |
| 4 | Resting HR | `resting_hr.rs` | FFI | ✓ |
| 5 | Respiratory rate (RSA) | `respiratory_rate.rs` | FFI | ✓ |
| 6 | Effort / Strain | `strain.rs` | FFI | 19 |
| 7 | Charge / Recovery | `recovery.rs` | FFI | 20 |
| 8 | Personal baselines (EWMA) | `baselines.rs` | FFI | 17 |
| 9 | HR zones | `hr_zones.rs` | FFI | 15 |
| 10 | Fitness Age / VO2max | `vo2max.rs` | FFI | ✓ |
| 11 | Baevsky Stress Index | `stress/index.rs` | FFI | 4 |
| 12 | Stress onset (live) | `stress_onset.rs` | FFI | 5 |
| 13 | SpO2 | `spo2.rs` | FFI | ✓ |
| 14 | Calories | `calories.rs` | FFI | 18 |
| 15 | Workout detection | `workout.rs` | FFI | ✓ |
| 16 | Steps (5/MG counter) | `steps.rs` | FFI | ✓ |
| 17 | IMU activity features | `imu_features.rs` | FFI | 7 |
| 18 | Rest (sleep performance) | `rest.rs` | FFI | 5 |
| 19 | Sleep debt | `sleep_debt.rs` | FFI | 5 |
| 20 | Daily stress | `stress/daily.rs` | FFI | 5 |
| 21 | Windowed stress (day + night) | `stress/window.rs` | FFI (day only) | 13 |
| 22 | HR anomaly watch | `hr_anomaly.rs` | Rust-only | 7 |
| 23 | HRV frequency domain (Lomb-Scargle) | `hrv_freq.rs` | FFI | 3 |
| 24 | Short-nap detection | `nap.rs` | FFI | 6 |
| — | Calibration schedule | `calibration.rs` | internal | 4 |
| 25 | Vitality / Body Age | `vitality.rs` | FFI | 9 |
| 26 | Sleep Regularity Index | `sleep_regularity.rs` | FFI | 7 |
| 27 | Circadian phase + Rhythm Age | `circadian.rs`, `biological_age.rs` | FFI | ✓ |
| 28 | Daily hydration goal | `hydration.rs` | FFI | 8 |
| 29 | Vital banding (personal vs typical) | `vital_bands.rs` | FFI | 13 |
| — | Shared stats | `stats.rs` | internal | ✓ |

There is no `crates/whoop-metrics`. It was renamed to `physio-algo`, the shim that briefly stood in its
place is gone, and nothing in the workspace names it any more.

---

## 1. Sleep detection + staging  ·  FFI `analyze_sleep`, `stage_sleep_refined`, `main_night_*`

One border call, `analyze_sleep(streams)`, carves in-bed spans from raw signals, stages each, and returns
one `SleepSession` per detected night. `sleep.md` is the full record. In brief:

**Detection** (`sleep/detect.rs`)
```
gravity_delta = L2(Δx, Δy, Δz) per sample
still_sample  = rolling 15-min window, fraction < 0.01 g >= 0.70
runs          = collapse still samples, break on class change or > 20 min gap
sparse gravity: HR-vouched gap bridge (median HR <= baseline x 1.05 across gap)
merge runs < 15 min into neighbours
gate loop: reject <= 60 min / > 16 h; median HR > baseline x 1.05 (x1.30 if deeply motion-quiescent);
           off-wrist fraction >= 50%; daytime [11:00,20:00) unless >= 90 min + RHR <= baseline x 0.95;
           night-continuation chain anchors the overnight-onset run, tails <= 90 min kept
```

**Staging V2, cardiorespiratory** (`sleep/v2.rs`), the 5.0/MG default, per 30 s epoch z-scored within-night:
```
deep  = -0.8 z(hr_var) + 0.5 z(HR) - 0.1 z(move) - deep_gate + 0.6 z(resp_reg) + ln(0.15)
rem   = +0.8 z(hr_var) - 0.4 z(move) + 0.4 z(HR) - 0.6 z(resp_reg) + ln(0.22)
light = ln(0.50)
wake  = 1.0 z(move) + 0.5 dz(z(hr_var)) + 0.6 dz(z(HR)) + motion_gate_boost + ln(0.34)
deep_gate = 5 x max(0, hr_flat11_pct - 0.40);  motion-quiescent clamps the cardiac wake term to <= 0
resp_reg  = R-R tachogram -> 4 Hz resample -> detrend -> DFT peak/sum over 0.15-0.40 Hz
cycle prior: deep decays to 0 after 55% of night; rem suppressed in first 12%
Viterbi 4-state, sticky self-transitions (deep 0.76, rem 0.92, light 0.80, wake 0.90)
```
Validated on DREAMT n=100 (**4-class** kappa **0.312**), AAUWSS (**0.410**, n=13) and sleep-accel / Walch
(**0.378**, n=31), all re-measured 2026-08-17 on `fixtures_multi_clean3`. **These are near the commercial
floor, not the wrist-optical ceiling** — an earlier version of this line said the latter and that is
falsified: published wrist-PPG staging reaches 4-class kappa 0.638-0.65 on the same physical channels
(Radha 2021 n=60, Fonseca 2023 n=394), and independently-measured WHOOP 4.0 scores 0.37. See
`whoop/docs/SLEEP-RESEARCH-LANDSCAPE.md`.
Every kappa in this doc is 4-class over Wake/Light/Deep/Rem; the 3-class DREAMT figure is a different
measurement (0.3657) and is not comparable to any of them. `tests/dataset_parity.rs` asserts these three
against constants 0.311 / 0.412 / 0.379 at ±0.008, so DREAMT's measured 0.312 sits 0.001 from its
constant and the constant was left unchanged. The status table below quotes the same three; a frozen golden
hypnogram test pins the tuned constants.

**Motion-aware wake refinement** (`sleep/refine.rs`): a wake segment >= 5 min at the night motion floor with
stable posture and no locomotion demotes its non-burst minutes to light.

**Main-night selection** (`sleep/mainnight.rs`): `score = asleep_minutes + alignment_bonus`, alignment
targeting the habitual midsleep (circular mean over >= 14 days, else 03:30 local); adjacent blocks bridge.

## 2. HR from PPG  ·  FFI `ppg_hr`

`ppg.rs`. Autocorrelation pitch-detect over the v26 24 Hz single-wavelength optical buffer: detrend, ACF
across the 30-220 bpm lag band, pick the fundamental, report bpm + confidence. Fills only seconds the strap
banked no HR for; never overrides a stored HR.

## 3. HRV / RMSSD / readiness  ·  FFI `hrv_rmssd*`, `hrv_sdnn`, `hrv_range_filter`, `hrv_clean_*`, `hrv_analyze_raw`, `hrv_windowed_*`, `hrv_rolling_rmssd`, `hrv_rr_coverage`, `hrv_nightly`, `hrv_readiness`

`hrv.rs`, the `HrvReadiness` type. Every HRV statistic on the strap path is here. One second copy survives
app-side: the Breathe screen's live readout calls Kotlin's own `Hrv.rmssd`, the unfiltered form
`hrv_rmssd_plain` already exports.
```
range_filter: keep 300-2000 ms;  clean_rr = range then Malik ectopic (|beat - local median| > 20%,
              centred 5-beat window);  clean_counts reports input / ranged / clean, ungated
rmssd            = sqrt(mean((rr[i+1] - rr[i])^2))          (Task Force 1996)
rmssd_plain      = the same with no artifact rejection;  pnn50_plain = % |dNN| > 50 ms, every pair
sdnn             = sample SD of NN, ddof = 1
gap-aware        = clean, remembering each survivor's index. A successive difference counts only when the
                   two beats were ADJACENT in the source, so a dropped beat never splices its neighbours.
                   Divides by the contiguous-pair count, not n-1
report seam      = a guard for a stream holding more beat-time than the clock had. Contiguity breaks at
                   a report whose CUMULATIVE beat-time leads the wall clock by more than SEAM_SLACK_MS
                   (2 s). Beats are grouped one run per second.
                   NOT a strap behaviour: a raw capture decoded through the real codec has the 5.0
                   sending 24,461 distinct seconds from 24,462 records at 0.89x beat-time, so it never
                   re-reports its previous window. Two of five wearers' stored streams still run over
                   1.0x (1.38x, 1.59x) for reasons not established, and this guard is what contains them
rmssd_gap_aware  = the above over one night's per-report (unix, rr) runs
analyze_raw      = clean -> {rmssd, sdnn, mean_nn, pnn50, n_input, n_clean}; 20-beat floor + optional spot
                   rejected-fraction gate (0.35). Flat: no report grouping, so no seam break
windowed_buckets = per-5-min tumbling bucket: clean-beat count + gap-aware RMSSD
windowed_avg_hrv = mean of those bucket RMSSDs over the session (the stored avgHrv)
nightly_hrv      = same, buckets whose centre lands in a deep (N3) span only, AND over reports the strap
                   did not flag optical_signal_poor on. The DISPLAYED nightly HRV, and the only producer
                   of it: window and trust filter both live in one function so no caller can build a
                   second version. `nightly_rmssd` is this per UTC day, from decoded history
rolling_rmssd    = trailing-window rmssd_plain per surviving beat, optional emit stride (the day chart)
rr_coverage      = sum(rr) / elapsed ms. Over ~1.0 is impossible: beats double-counted or reports overlap
duplicate_beat_count  = rows repeating an earlier (ts, rr) EXACTLY. Byte-identical re-inserts only
overlapping_report_count = reports re-covering time already covered. THIS is the mechanism behind a
                   coverage over 1.0; the exact-duplicate count is not
readiness        = 7-night mean of ln(RMSSD) vs a smallest-worthwhile-change band (long mean +/- 0.5 SD)
                   -> primed / normal / suppressed + overreaching watch
```
Measured on 85 nights from 5 wearers, on a corpus with the R-R duplication removed: the seam rule leaves
the per-night median unmoved (+0.0%) and fires on 4 of 18 nights for the one wearer whose duplication was
removed, 21 of 29 and 13 of 19 for the two whose over-1.0x streams have no known cause, and 0 of 17 for a
wearer whose stream is already under 1.0x. On the same nights BEFORE de-duplication it read a median
-14.1% and moved 16 of that wearer's 18 nights, so most of its apparent value was our own duplicate rows.
The deep window reads a further median -13.9% below the whole night and yields nothing on 2 of 85 nights.
`SEAM_SLACK_MS` is bit-identical from 1 s to 300 s. Reproduce with the `hrv_seam` and `hrv_window`
examples and `tools/seam-slack-sweep.py`.

The DISPLAYED nightly HRV is the deep window, and that is what agrees with WHOOP: over 15 nights paired
to WHOOP's own published value, deep-only sits +0.6 ms from it (MAE 2.1, 12 of 15 inside +/-3 ms) and the
whole night +6.7 ms (MAE 6.0, over on 14 of 15). One wearer, cross-wrist (WHOOP left, noop right).

## 4. Resting HR  ·  FFI `session_resting_hr`, `daily_resting_hr`

`resting_hr.rs`. Session resting HR = lowest 5-min tumbling-window mean bpm over `[start, end]`. Daily
resting HR = min of the per-session floors.

## 5. Respiratory rate (RSA)  ·  FFI `resp_rate_from_rr`

`respiratory_rate.rs`. R-R tachogram -> 4 Hz resample -> 8 s detrend -> per 5-min window peak-pick the
breathing modulation -> median rate. Plausible band 8-25 bpm, else `None`.

Measured on the one real 4.0 offload (5.91 days, 3 nights,
`whoop-data/own-data/strap-data/mine/noop-debug-db-20260731`): 14.1 / 14.1 / 13.3 bpm per night, hourly
sub-windows 12-15 bpm, and 14.1 / 13.3 / 13.7 when repeated beats are dropped first, so beat duplication
moves it by under 1 bpm. First evidence the RSA path works on 4.0 hardware, not only on 5.0/MG.

**The 4.0 v24 register decoded as `resp_raw` (`gen4.rs`, inner u16 @76) is NOT respiration, and nothing
reads it.** Over 195,744 consecutive 1 Hz samples from that offload it takes 3 values: 3073 (99.0 %),
2817, 1793. The low byte is constant `0x01` in every sample and in the pinned real frame, so the 16-bit
magnitude is an artefact of packing a tag byte with a status byte; the only variation is the high byte
stepping 12 -> 11 (or -> 7 before 2026-07-27 00:50) for 27-29 s once every 1155 s exactly, and only on
nights the strap was worn. A 1 Hz breathing signal cannot be flat for 19 minutes at a time. Structurally
v24 sits +3 bytes from v18 for HR (17 vs 14), R-R (18/19 vs 15/16) and skin temp (68 vs 65), so @76/@77
align with v18 @73/@74 = `sleep_state_raw` and the tri-mode sleep-only SpO2/status byte, and both real
frames carry the same literal `01 0c 02 0c` block there. Do not re-investigate it as a respiration
channel; the name is the only thing respiratory about it.

## 6. Effort / Strain  ·  FFI `strain_score`, `strain_default_denominator`, `effort_on_axis`

`strain.rs`.
```
HRR = HRmax - RHR;  %HRR = clamp((HR - RHR) / HRR x 100, 0, 100)          (Karvonen 1957)
TRIMP per sample: zone_weight x its own inter-sample gap (dropout capped 20 min)
   Edwards (default) 5-zone weights at 50/60/70/80/90 %HRR, or Banister exponential
Effort = 100 x ln(TRIMP + 1) / ln(7201)                                   (Edwards 1993 / Banister 1991)
```
Denominator 7201 maps a 24 h top-zone day (5 x 1440 = 7200) to exactly 100.

A reader may ask for the other 0-21 Day Strain axis instead: `effort_on_axis` multiplies by 21/100, and an
import multiplies the other way by 100/21. Multiplying by one ratio is not the same operation as dividing
by the other, and only the division inverts an import exactly, so an export boundary divides. The stored
value never moves; only the displayed one converts.

## 7. Charge / Recovery  ·  FFI `recovery_score`, `recovery_band`, `recovery_index_slope`, `recovery_banked_nights`

`recovery.rs`. Weighted robust-z composite through a logistic squash.
```
z(x) = (x - mu) / max(1.253 x spread, 1e-9)                               (Plews 2013 / Buchheit 2014)
HRV 0.55 (higher better) · RHR 0.20 (lower) · Rest 0.15 · Respiration 0.05 (lower) · Skin temp 0.05 (|dev|)
present terms only, weights renormalise
Charge = clamp(100 / (1 + exp(-1.6 (composite_z + 0.20))), 0, 100)        (z = 0 -> ~58%)
```
Cold-start (HRV baseline unusable) returns `None`. Bands red < 34, yellow 34-67, green >= 67.

## 8. Personal baselines (EWMA)  ·  FFI `baseline_update`, `baseline_fold_history`, `baseline_metric_cfg_*`

`baselines.rs`. Per metric (HRV, RHR, respiration, skin temp, Effort):
```
centre alpha = 1 - 0.5^(1/14 nights);  spread alpha = 1 - 0.5^(1/21 nights)
winsor fold within +/- 3 x spread (spread tracks the unclamped deviation); hard-outlier reject > 5 x
status: calibrating < 4 nights · provisional 4-13 · trusted >= 14 · stale if > 14 nights since update
```

## 9. HR zones  ·  FFI `hr_zones_for_age`, `hr_time_in_zone`

`hr_zones.rs`. `HRmax = override ?? tanaka(208 - 0.7 age) ?? 220 - age`; five 10%-HRR bands from 50% up;
time-in-zone holds each sample until the next.

## 10. Fitness Age / VO2max  ·  FFI `vo2max_estimate`, `fitness_age_compute`

`vo2max.rs`. Nes 2011 HUNT non-exercise VO2max from age, sex, waist, RHR, PA-index; Fitness Age inverts the
same equation against a normative peer (RHR 65, PAI 5) so the body term cancels. Display band +/- 5 years.

## 11. Baevsky Stress Index  ·  FFI `stress_index`, `stress_components`

`stress/index.rs`. `SI = AMo / (2 x Mo x MxDMn)` over a cleaned R-R histogram (Mo modal R-R s, AMo
modal-bin share %, MxDMn range s). Tall-narrow-low-range reads high. Unbounded, so it is the one
stress reading that carries no 0-3 band.

## 12. Stress onset, live  ·  FFI `stress_onset_evaluate`

`stress_onset.rs`. Stateful, edge-triggered: fast RMSSD (last 60 beats) below 0.6 x a slow EWMA baseline
fires a JITAI nudge. Gated by resting HR band (55-100), recent motion, min 20 beats, and a 15-min refractory.

## 13. SpO2  ·  FFI `spo2_from_paired`, `nightly_spo2_raw_means`

`spo2.rs`. Ratio-of-ratios over the 4.0 v24 paired red/IR window (30 s, curve `110 - 25 R`, clamp 70-100),
plus a 30-night soft-anchored rolling readout. 5.0/MG v26 is a single wavelength with no red/IR pair, so a
percent is produced on 4.0 only; on 5/MG the app stores nightly raw red/IR ADC means, never a fabricated %.

## 14. Calories  ·  FFI `calories_estimate_day`, `calories_estimate_bout`

`calories.rs`. Per-second Keytel 2005 active energy above a HRR gate, revised Harris-Benedict BMR below it,
sex-specific coefficients. Approximate, not calorimetry.

The two gates are two different questions, not one constant forked in two:

| gate | value | question | applies to |
|---|---|---|---|
| `ACTIVITY_HRR_FRACTION` | 0.30 | is this second work? | seconds inside a bout `workout_detect` confirmed motion over |
| `TRUST_HRR_FRACTION` | 0.50 | how high must an HR run before an unconfirmed second is believed to be work? | every other second |

`calories_estimate_day` takes the confirmed bouts, so a detected workout's seconds are billed by the same
Keytel the bout tile shows — they used to be billed BMR, up to 8.1x lower across 94-119 bpm, on the same
screen as the bout number. What an unconfirmed elevated second should cost is NOT settled: `0.30 HRR` is the
light/moderate boundary but presupposes exercise, and an ungated day feed also carries caffeine, stress,
posture and PPG artifact. **`TRUST_HRR_FRACTION` 0.50 has no derivation** — it marks no boundary and was
chosen to move a total; arbitrating it needs motion-labelled whole-day ground truth the tree does not hold.

`calories_estimate_day` returns `total_kcal` / `resting_kcal` / `active_kcal`. Only `active_kcal` is
comparable to a phone's "active energy"; `total_kcal` is whole-day TDEE and includes BMR.

HRmax for both paths resolves through `strain::resolve_hrmax` (caller → observed peak → Tanaka → the single
`FALLBACK_HRMAX` 220), so no call site can hold its own fallback and gate the same person two ways.

## 15. Workout detection  ·  FFI `workout_detect`

`workout.rs`. A sustained window (>= 5 min) of elevated HR (RHR + 15 bpm) and sustained motion (> 0.20,
10 s smoothed), merged across short gaps, qualified by >= 50% of the bout in Edwards zone 2+. Per bout:
avg/peak HR, zone-time %, mean %HRR, strain, and calories.

## 16. Steps, 5/MG counter  ·  FFI `steps_counter`

`steps.rs`. Wrap-aware deltas of the strap's cumulative u16 step counter, dropping any delta >= 512
(sync-gap or reboot). Returns raw motion ticks; the caller applies its ticks-per-step calibration.

## 17. IMU activity features  ·  FFI `imu_features`

`imu_features.rs`. Over a window of decoded 100 Hz 6-axis IMU samples: accel-AC RMS energy (g), gyro energy
(deg/s), jerk RMS, and a gait-band (1.2-3.5 Hz) autocorrelation cadence with its own strength. A feature
for coarse activity classification, never a physiological gate.

## 18. Rest (sleep performance)  ·  FFI `rest_score`

`rest.rs`. `0.50 duration-vs-need + 0.20 efficiency + 0.20 restorative(deep+REM) + 0.10 consistency`,
0-100, deep-adequacy factor on the restorative term, 8 h default need.

## 19. Sleep debt  ·  FFI `sleep_debt_ledger`

`sleep_debt.rs`. Rolling `sum(slept - need)` over a 14-night window of nights with data (never zero-fills),
8 h need, on-target band +/- 30 min.

## 20. Daily stress  ·  FFI `daily_stress`

`stress/daily.rs`. `3 / (1 + exp(-(zRHR + zHRV)))` against up to 30 prior (RHR, HRV) nights, 14-day baseline
gate. Bands low [0,1), medium [1,2), high [2,3].

## 21. Windowed stress, day + night  ·  FFI `daytime_stress` + `sleep_stress`

`stress/window.rs`. **One formula, two derivations.** `windowed_stress(points, cfg)` scores a set of
buckets against the SAME set's own calm quartiles: `zHR` vs the Q25 bucket HR, `zHRV` vs the Q75 bucket
RMSSD, `3 / (1 + exp(-(zHR + zHRV)))`, plus the minutes spent in each band. No exercise gate. Peak hour on a
tie is the last (adopted app-side). A bucket with no mean HR is dropped, never invented; the caller applies
its own >= 300 HR-row gate before the border.

| derivation | cfg | note |
|---|---|---|
| `daytime_stress` | `hours: Some((6, 22))`, bucket 3600 s | shipped; pinned bit-for-bit to one real worn day |
| `sleep_stress` | `hours: None`, bucket 3600 s | the caller passes only in-span buckets |

Both cross the border as one `WindowedStressInfo`, which carries the band minutes and `high_share_pct` —
the high band's share of the scored minutes, `None` when nothing scored, so the app bands and tallies
nothing itself.

The night passes `None` rather than a range **because a sleep window crosses midnight** and one hour-of-day
range cannot express "22:00 to 06:00" — trying would silently drop half the night. The caller selects the
buckets; the core scores what it is given. `cfg.bucket_seconds` only converts scored buckets to band minutes
today: `HourPoint` is keyed by hour-of-day, so a sub-hour bucket needs a bucket index before that knob turns.

**Unvalidated, and a placeholder in the owner's own words.** Nothing external confirms either the day or the
night split, and WHOOP's numbers are not a ground truth we can check without their algorithm. The reference
is the window set's **own** calm quartiles, so a score is relative to that day or that night: **a uniformly
stressful night can read low because nothing in it was calm by comparison** (asserted, not just described,
by `uniformly_stressful_night_still_reads_neutral`). This caveat already applied to the shipped daytime
metric and was accepted there. Both derivations share one core on purpose, so replacing it replaces both.

The night ships at **hourly** first — the existing tested path, ~6 buckets for a 6 h night. If that reads
useless on real nights the knob is `bucket_seconds` and the next step is the 5-minute buckets
`hrv_windowed_buckets` already computes; per-minute copies WHOOP's resolution without their smoothing.

## 22. HR anomaly watch  ·  Rust-only `HrWatch`

`hr_anomaly.rs`. A sustained (>= 300 s) elevated-at-rest run over offloaded history: personal resting HR =
10th percentile of good-signal, on-wrist, at-rest samples; elevated = RHR + 45 or the 100 bpm floor. Flags
elevated only (low HR is never flagged), needs 600 baseline samples. Wellness nudge, never a diagnosis,
never real-time. No app caller and no Kotlin twin yet.

## 23. HRV frequency domain  ·  FFI `hrv_freq_domain`

`hrv_freq.rs`, the `HrvBands` type. LF / HF / LF-HF / total power over the R-R tachogram via the
Lomb-Scargle periodogram (estimated directly from the uneven samples, no resampling). Task Force (1996)
bands (LF 0.04-0.15 Hz, HF 0.15-0.40 Hz); span gates HF >= 60 s, LF (and LF/HF, total) >= 250 s; 20-beat
floor. Range + Malik-ectopic cleaned first. Approximate, non-clinical.

## 24. Short-nap detection  ·  FFI `nap_evaluate`

`nap.rs`, the `NapDecision` type. Tri-state (Nap / None / Inconclusive) over a candidate window: dense-gravity
eligibility gate (>= 20 rows, median inter-sample gap <= 90 s), the longest sustained-still run (reusing
`workout::activity_series` + `smoothed_intensity`), a `[min,max]`-minute length gate, and an HR-settled gate
(mean HR <= resting + margin when resting is known). Only PROPOSES a review card, never auto-writes sleep.

## 28. Daily hydration goal  ·  FFI `hydration_daily_goal_ml`, `hydration_cfg`

`hydration.rs`. `round50(sexBaseline + clamp(round(effort / 100 * 700), 0, 700))`, sex baselines 3700 /
2700 / 3200 ml, both roundings half-up. Derived from the body profile and the day's Effort only, never from
what was logged. Which quantities a quick-log tap adds is the frontend's, not this.

## 29. Vital banding  ·  FFI `vital_band`, `vital_typical_range`

`vital_bands.rs`. In range means `|z| <= 2` against the wearer's own baseline once its status is Trusted,
and inside the typical-adult window (resp 12-20, SpO2 95-100, RHR 40-60, HRV 40-120, skin abs 33-36, skin
dev +/-0.6) before that and again once it goes stale. A `MetricCfg`'s physiological bounds are an outer
guard only, never the in-range band. A skin-temp reading >= 20 degC is absolute, below it a deviation, and
a history is filtered to one kind before folding. Wellness bands, never clinical cut points.

## Internal helpers

- `calibration.rs` — WHOOP's per-metric unlock/full-calibration night schedule (blood O2 1, recovery 3,
  sleep consistency 5, skin temp 7, VO2max 14), so a readout appears on the same schedule the app uses.
- `stats.rs` — `mean`, `median`, `percentile`, `amplitude`, `pearson`, `linear_fit`.

---

## Status, sorted by strength of evidence

Every algorithm this crate exports is called by the app: there is no unwired backlog. **1 public
function in the crate is reached by no other crate** — `resting_hr::floor_mean_log_line`, which is a
byte-for-byte twin of a Kotlin log line that ships, so the Rust half is a parity control with no
caller rather than an algorithm going unused. `HrWatch` is off the FFI too but is not orphaned:
`whoopctl --hr-watch` prints it. That is the border in ONE
direction. In the other, **20 Kotlin files still carry maths of their own**, listed in
`noop-wt-tan/docs/ALGORITHMS.md` and re-derived against the code by
`dev-notes/noop-tan/audit_kotlin_algorithms.py`. What differs below is how strongly each is verified.

### ✅ Hardware-verified — run against a real 5.0/MG

| Algorithm | Evidence |
|---|---|
| HR from PPG, HRV (all variants), resting HR, respiratory rate | decoded from real streams over months of wear |
| Skin temperature, SpO2 % (5.0/MG) | cross-checked against the band's own readings |
| Steps, IMU activity features | wrap-aware counter and 100 Hz buffer verified on-device |
| Sleep Regularity Index | computes at 88 % coverage over a real week; correctly refuses below its gate |

### ✅ Gold-standard scored — against external PSG datasets

| Algorithm | Evidence |
|---|---|
| Sleep detection + staging (V2) | Cohen's **4-class** κ against PSG, re-measured 2026-08-17 on `fixtures_multi_clean3`: **0.312** DREAMT (100 subjects), **0.410** AAUWSS (13), **0.378** sleep-accel (31). Those three are asserted by `tests/dataset_parity.rs` against constants 0.311 / 0.412 / 0.379 at ±0.008. The 3-class DREAMT figure is a different measurement (0.3657) and is not comparable. Against our own stored hypnograms, which is a consistency read and not accuracy — the truth column is our own past output — the `--ignored` sheet prints 0.533 killa5 (13), 0.483 strap (46), 0.599 whoop4 (20); those three are printed, not asserted, and were reproduced unchanged on `fixtures_multi_clean3` on 2026-08-17. They are CIRCULAR by construction, so they are not evidence of accuracy. A re-tune was attempted and **rejected**: it gained 0.044 on the fitting set and lost up to 0.372 on held-out sets |

### ✅ Parity-tested — pinned to the previous implementation

Workout detection · bout + day calories · strain/Effort · recovery/Charge · HR zones · personal
baselines · Fitness Age / VO2max · Rest · sleep debt · daily + daytime stress (the daytime half re-pinned to a
real worn day, bit-for-bit, across the windowed-core refactor) · Baevsky SI · stress
onset · short-nap detection · HRV frequency domain. Each still reproduces the figures its Kotlin
predecessor produced.

### ⚠️ Literature-sourced — cited, but not validated on our data

| Algorithm | Caveat |
|---|---|
| Vitality / Body Age | Every coefficient cites a named meta-analysis (PMIDs in the module header). Only VO2max is published as a per-unit slope; steps, HRV and sleep regularity are quartile contrasts we linearise, and the sleep-duration figure is ours. The Gompertz doubling time is a SENSITIVITY parameter, not a constant of nature |

### ⚠️ Known-weak — documented rather than hidden

| Algorithm | Issue |
|---|---|
| Resting HR | Reads ~10 bpm below a reference band. The bias is stable across independent halves and is NOT explained by wrist or sensor differences (two bands agree to within 2 bpm on the same statistic). Unchanged because the data says something is wrong, not what to change it to |
| SpO2 from paired red/IR (4.0) | **Withheld: the sample rate forbids it.** The pair arrives at 1 Hz, so the Nyquist limit is 0.5 Hz while a cardiac waveform runs 0.83-3.0 Hz — the pulsatile component ratio-of-ratios reads is aliased away before decode. Confirmed on two straps (2.1M samples): 89.3% and 98.2% of 30 s windows had zero red amplitude, and the survivors produced ~80% for two healthy wearers. Restricting to in-bed windows makes it worse (95.7% flat), so it is not a sleep-only channel. A pulsatility gate returns None. The 5.0/MG path reads the strap's own value and is unaffected |
| Rhythm Age (CosinorAge) | Has **never computed** on real data — needs 7 worn days of on-chip motion. Its activity scale carries an unvalidated conversion factor that only a concurrent reference accelerometer can settle |
| 4.0 record decode | v24 and v25 are pinned to real captured 4.0 frames (`whoop-protocol/tests/fixtures/real_frames.json`), and the app's 4.0 offload runs through this decoder. What is still unexercised is `whoop-client`'s own 4.0 connect/bond path — no 4.0 has been bonded over the desktop radio, so `v5`/`v7`/`v9`/`v12` have only synthetic coverage |

### ⚠️ Unvalidated — implemented and unit-tested, nothing external confirms the output

| Algorithm | Caveat |
|---|---|
| Windowed stress, both derivations | Explicitly a **placeholder**. The reference is the window set's own calm quartiles, so a score is relative to that day or night and a uniformly stressful night reads low. The day half is separately parity-pinned to the figures its predecessor produced, which fixes it in place — it does not make it right |

### Rust-only, not on the FFI

| Algorithm | Note |
|---|---|
| HR anomaly watch (`HrWatch`) | Implemented here; no app surface yet |
