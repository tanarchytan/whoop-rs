# Sleep pipeline (`physio-algo::sleep`)

The whole WHOOP sleep pipeline lives here: detect the in-bed spans of a night, stage each into a
per-30 s-epoch hypnogram, refine wake, and derive the day's main night. Pure and deterministic — no BLE,
no IO, no async. The app (`noop-tan`) is a thin frontend: `analyzeSleep` is the whole-night door, and the
app calls **15 further `sleep_api` exports** around it (single-span restage, the main-night family,
debt/regularity/need, naps); see `noop-tan/android/SLEEP-BORDER.md` for what stays app-side.

## One entry: `analyze`

```rust
pub fn analyze(streams: &SleepStreams) -> Vec<Session>
```

`SleepStreams` is a night's raw signals (`hr`, `rr` runs, `accel`/gravity, `resp`, `steps`) plus
`tz_offset_s`, `wrist_off` intervals, and the strap's `band_sleep_state`. `analyze` runs, per accepted span:

```
detect_sessions  → v2::stage → refine::refine → efficiency + session_resting_hr + windowed avg_hrv + grids
```

and returns one `Session { start, end, efficiency, resting_hr, avg_hrv, segments, motion_grid, sleep_state_grid }`
per in-bed span. Also public: `stage_refined(input, steps)` (stage + refine one already-detected span, for
the app's edit self-heal) and the main-night functions (`main_night_index/_group_indices/_selection`,
`bridged_night_groups`, `habitual_midsleep_sec`).

## Modules

| file | role |
|---|---|
| `detect.rs` | the gravity-stillness detection spine (`is_gravity_sparse`→`gravity_deltas`→`classify_still`→`build_runs`→`merge_periods`→`bridge_sleep_gap`) + the `detect_sessions` gate loop, and the per-epoch `session_epoch_motion`/`session_epoch_sleep_state` grids |
| `v2.rs` | the V2 (cardiorespiratory) staging recipe — the DREAMT-tuned emissions + Viterbi; stages **every** strap. **FROZEN: do not edit it.** It is both the shipped recipe and the control every new arm is scored beside, and a control that moves is not a control. Reading `prepare` / `emission_terms` / `viterbi` / `emissions_at` is intended; a primitive both engines need belongs in `common.rs` or its own module |
| `pipeline.rs` | the staging path as **19 numbered steps** with `run_to(input, cfg, p, upto)`, a per-step artifact digest and a per-prefix digest, plus `StepId::deps()` (the DAG) and `StepId::feedback()` (the one forward reach, `Anchor` probe-decoding). 5 steps carry an artifact today; 3..=9 fold into `Assemble`. `SleepConfig` is the swap seam — `emit` and `decode`; the anchor probe is always the shipped decoder |
| `features.rs` | the per-epoch feature vector (`Features`, `Cardiac`, `EPOCH_S`) every emission reads |
| `cardiac.rs` | interval order statistics over one beat window — seven quantiles absolute, seven detrended, mean HR, RMSSD/pNN50/mean\|ΔRR\|. One producer for the stager and the screening harnesses, so what is measured is what runs. **Not yet on any staging path** |
| `hrv_bands.rs` | frequency-domain HRV (VLF/LF/HF) off a resampled tachogram. Rejected by a permuted null on our channel; kept because the rejection is a measurement |
| `resp_features.rs` | respiration off the same beat series: the RSA band peak and its power, the summed HF power taken FROM `hrv_bands` rather than recomputed, and breath-level length, rate and shape statistics off a band-passed surrogate of the same tachogram. Screened against the confirmed cardiac family and it clears its permuted null on both cohorts, at +0.020 MESA and +0.025 AAUWSS against that family's own +0.087 / +0.075 - and over half of that is `v2::resp_regularity`, which the shipped recipe already reads. Rate variability carries on wrist; every amplitude column loses value moving off ECG. **Not yet on any staging path** |
| `markov_loss.rs` | a staging loss that prices misplaced BOUNDARIES as well as misclassified epochs — per-class epoch cost, invented-boundary cost, missed-boundary cost. Scores a finished hypnogram; **nothing decodes with it yet** |
| `conditioned.rs` | an observation-conditioned transition: the shipped matrix with each epoch's diagonal loosened by that epoch's own motion, off-diagonal mass returned in shipped proportions so a structural zero stays zero. Its decoder replicates the frozen `v2::viterbi` with a per-epoch matrix and is pinned to it by test. `beta_milli: 0` is the shipped decode label for label - the built-in null. Reached through `DecodeCfg::Conditioned`; the anchor is always chosen under the shipped decoder |
| `posterior.rs` | forward-backward posterior marginals over the four-class HMM, in the log domain with log-sum-exp because a night underflows a linear-domain product. Takes the per-epoch transition closure `conditioned` takes and applies the same zero floor and uniform start, so it serves the fixed matrix and the conditioned one alike and its marginals assume what `v2::viterbi` assumes. The two decode rules that need marginals rest on it — `posterior_marginal_decode` (per-epoch argmax) and `decode_with_costs` (Bayes risk under `markov_loss::Costs`'s per-class `fc`; `ft`/`fh` are sequence-level, so no per-epoch rule can charge them and it says so). A shared primitive consumed by the decode seam, **not a 20th step**, and nothing on the shipped path decodes with it yet. Sized on real nights by `examples/posterior_check.rs`: the marginal rule differs from Viterbi on 5.45% of DREAMT epochs (7.80% AAUWSS, 8.09% sleep-accel, 8.52% `ours`), and the Viterbi path carries mean posterior 0.846 (0.813 / 0.812 / 0.800) |
| `sequence.rs` | hypnogram STRUCTURE, which no confusion-matrix number can see: per-class fragmentation, bout-length distributions and their Wasserstein-1 distance to truth, upper-tail mass, and a rare-transition rate whose rare set is derived from the reference itself. Accumulates over time-CONTIGUOUS segments, so a labelling hole never becomes a transition. `min_run_smooth` is the instrument's control, not a staging step |
| `metrics.rs` | the scoring primitives: `kappa4`/`kappa3`/`merge3`, `recall`/`precision`/`specificity`/`f1`, `balanced_accuracy`, `min_recall`, `per_recording`, `bout_score`, `bootstrap_kappa_ci` (resamples RECORDINGS), and the kappa-bonus pair. **Per-class RECALL is the selection object; F1 is a diagnostic** — F1's denominator carries the class priors, so it moves with prevalence exactly as kappa does |
| `agreement.rs` | night-summary agreement: `summarise` (TST/WASO/latency/efficiency/stage minutes) and `bland_altman` (bias, 95% LoA, proportional-bias slope) |
| `movement.rs` / `posture.rs` | the motion families the epoch grid buckets |
| `refine.rs` | motion-aware wake post-pass (hot-but-still WAKE → light; density self-gated on the observed streams). `RefineParams::SHIPPED.skip_window_edges` exempts the first and last epoch of a span, which is where sleep-onset latency and the final wake legitimately sit |
| `mainnight.rs` | main-night selection by a learned-timing score, the two-tier gap bridge, and the circular-mean habitual midsleep |
| `params.rs` | every V2 emission weight, gate and transition in one `Params` struct. `Params::SHIPPED` is the tuned recipe; `stage` with anything else is the tuning path only |
| `common.rs` | the per-night `ZScore` and the R-R run flattener `flatten_rr`. The numeric primitives (`median`, `population_sd`) live in `crate::stats` |
| `input.rs` | the protocol-free sample types (`HrSample`/`RrRun`/`AccelSample`/`StepSample`) and the `SleepInput` bundle they arrive in |

## The detection gate loop (order is load-bearing)

`detect_sessions` builds the stillness spine, then for each candidate sleep run applies, in order:
`minSleep(60 min)` → `maxSpan(16 h)` → `confirm_sleep_with_hr` (median HR in the sleep band, widened on a
deeply-motion-quiescent run) → `off_wrist_fraction` (< 0.5) → the daytime false-sleep / morning-stillness
guards. A **cross-night continuation chain** lets an overnight night's post-11:00 tail skip the daytime
guard; a dropped run never re-anchors the chain. Sparse gravity (a 5.0 backfill) enables an HR-vouched
gap bridge so a clumped night is not shredded — a dense 4.0 night is byte-identical to the ungated path.

## Recipe

V2 is universal — no per-strap gate. It was tuned on DREAMT PSG gold (n=100 wrist-optical + AASM); on real
4.0 (unconstrained) it produces the same operating point it does on gold, so no separate 4.0 profile is
needed. V1 (Cole-Kripke) is retired. Detection is a **gravity-stillness** spine, not Cole-Kripke.

## Tests

`detect.rs` / `refine.rs` / `mainnight.rs` carry unit tests plus ~30 cases ported byte-identical from the
app's Kotlin gate/main-night suites (off-wrist, daytime guard, sparse-gravity, night-continuation,
HR-confirm median, span-cap, morning-stillness, motion-corroborated wake, the realistic-nap sweep,
selection reasons, habitual learning). `golden_tests.rs` pins the V2 hypnogram frozen-golden.
`tests/dataset_parity.rs` (`--ignored`) asserts the DREAMT, AAUWSS and sleep-accel kappas and prints a
sheet naming every fixture set with what its truth column IS, so no set sits unscored and unnamed.
**1208 workspace tests, 0 clippy** (measured 2026-09-01; re-derive, never carry forward)**.**

**The `#[ignore]`d suite is not optional and nothing else runs it** — the cohort gates, every negative
control, and the source-text cross-checks that stop two files drifting apart all live there. Two had
gone stale unnoticed because it was never run: a cohort constant left at its pre-retune value, and a
baseline replica still running the weak gate its own shipped test was written to condemn. Neither is
reachable from `cargo test`, and **`cargo test -p physio-algo --lib` does not build `tests/` at all.**

## App-side border: complete

The Kotlin sleep algorithm is fully retired. `remFunnelDiagnostic` + its Test-Centre caller were deleted,
taking the whole Stage 1–3 epoch classifier (`buildEpochGrid`/`coleKripke`/`classifyOne`/…) with them, so
whoop-rs stages everything on the live path *and* in diagnostics. The main-night selection twin is gone too:
`SleepStageTotals`'s span-path selectors (`mainNightIndex` / `mainNightGroupIndices` / `mainNightSelection` /
`bridgedNightGroups`) and the `habitualMidsleepSec` learner now delegate to the `mainNight*` /
`bridgedNightGroups` / `habitualMidsleepSec` FFI (their Kotlin scoring/bridging/circular-mean bodies
deleted). The app's `MainNightConsistencyTest` suite runs through the FFI and passes byte-identical — the
cross-language parity net. What stays Kotlin is storage-coupled: the `stagesJSON` decode + the
`dailyAggregateHonoringEdits` edit-seam (its `...ByStages` selector scores decoded JSON minutes, which has
no Rust twin).
