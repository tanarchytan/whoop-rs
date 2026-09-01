# whoop-rs — project instructions

From-scratch **Rust WHOOP BLE client** (4.0/Harvard + 5.0/MG/Maverick). One pure wire codec + a
generic BLE core + a WHOOP client, reusable as a desktop CLI and (next) Android/iOS from one core.
Own-device / right-to-repair, non-commercial. **Firmware is read-only, never flashed.** Health metrics
are wellness estimates, never medical.

**Read `docs/architecture.md` first** — it's the authoritative map (crates, wire protocol, mobile plan,
build). This file is the working contract; `docs/` is shipped documentation; `dev-docs/` (git-ignored)
is working notes.

## Layout (8 crates, deps point strictly inward)

Two independent leaves: `whoop-protocol` (pure sans-IO codec, thiserror only) and `ble-core` (transport
trait + mock), which knows nothing about WHOOP. On the codec sit `physio-algo` (every decode-to-metric
algorithm: sleep, HRV, recovery, strain, SpO2 …, no BLE/IO), `whoop-store` (per-(person, strap)
calibration), `whoop-ffi` (uniffi → Kotlin/Swift) and `whoop-client`; on `ble-core` sits `ble-btleplug`,
the only crate that links a radio. `whoop-client` is `WhoopClient<T: BleTransport>` — generic over the
transport, so it does NOT depend on `ble-btleplug`; `whoopctl` is the one crate that joins the two sides.
There is no `whoop-metrics` crate — it became `physio-algo`. Full graph: `docs/architecture.md`.

## Build / test / toolchain

```bash
cd whoop-rs
cargo build
cargo test                 # 1202 passed, 0 failed, 52 #[ignore]d  (2026-09-01; re-derive, never carry forward)
cargo clippy --workspace --all-targets
cargo run -p whoopctl -- scan

# `cargo test -p physio-algo --lib` DOES NOT BUILD tests/ AT ALL. Reading its 792 as green while
# the workspace was red is how a stale cohort constant survived days of daily runs. Use `cargo test`.

# THE IGNORED SUITE IS NOT OPTIONAL, and nothing else runs it. 52 tests: the cohort gates, every
# negative control, and the source-text cross-checks that exist to stop two files drifting apart.
# Two of those had gone stale unnoticed because this was never run.
# ABSOLUTE paths. Cargo runs each test binary with ITS OWN PACKAGE as the working directory, so a
# relative ../whoop-firmware resolves inside crates/whoop-protocol and the firmware gate dies with
# NotFound - measured 2026-09-01. It reads as a broken GATE when it is a broken COMMAND.
WHOOP_ZBIN_DIR="$PWD/../whoop-firmware/.zbin-extract" \
WHOOP_CAPTURE="$PWD/../whoop-data/own-data/noop-raw-capture-260729-0928.jsonl" \
  cargo test --release --workspace -- --ignored > /tmp/ignored.log 2>&1
grep -E "^test result" /tmp/ignored.log      # expect: 52 passed, 0 failed
#   BOTH env vars are REQUIRED. Without them two data-gated tests PANIC rather than skipping, which
#   is the behaviour we want - a silent skip is how a gate stops proving anything. The 30 firmware
#   images live in a DOT-directory, so `ls whoop-firmware/strap/*.zbin` finds nothing and reads as
#   "the archive is gone"; captures are git-ignored because they hold personal biometric frames.
#   NEVER pipe this through `tail`: the pipe buffers to completion, so you see nothing for fifteen
#   minutes and then a truncated tail.

# The sleep corpus. `cargo test` does NOT reach it — every cohort gate is #[ignore]d behind a
# multi-GB tree outside the repo, so a green suite says nothing about staging. Run it whenever
# anything under sleep/ changes:
cargo test --release -p physio-algo --test dataset_parity -- --ignored --nocapture
#   expect: 6 passed, 0 failed; dreamt 0.3082 (n=100), aauwss 0.4120 (n=13), sleep-accel 0.3780 (n=31)
#   corpus: sleep-benchmark/fixtures_multi_clean3 (the pinned root; WHOOP_SLEEP_FIXTURES overrides)
# Those moved on 2026-08-31 when beats sharing a whole-second stamp stopped being stamped at the
# same instant: aauwss +0.0017 (real beats), dreamt -0.0041 (34% of its intervals are synthesised
# identical by the fixture builder), sleep-accel unchanged (carries no R-R at all).
# Those three are the POOLED-confusion kappa. The examples/ harnesses print the PER-NIGHT MEDIAN of the
# SAME staging: 0.290 / 0.425 / 0.332. Both are `stage_v2` UNREFINED — no PSG cohort carries a step
# stream, so refine_wake declines on all 144 (dataset_parity.rs:20). Never call either one "the app
# path", and never compare a pooled number to a median one.
```

Pinned to **MSVC** via an in-dir `rustup override` (btleplug's WinRT deps need the MSVC linker; the
windows-GNU `dlltool` is incomplete). Fresh clone: `rustup override set stable-x86_64-pc-windows-msvc`.

The workspace `[patch.crates-io]`es `btleplug` to a sibling `../btleplug` fork (the WinRT `add_peripheral`
by-address fix). A fresh clone needs that sibling checkout present until the upstream PR lands; drop the
patch once it does.

## Guardrails (always on)

- **`sleep/v2.rs` IS FROZEN. Do not edit it.** David, 2026-08-31: *"make it a hard no to edit v2.rs
  because thats the 'old' and tanv1 is the 'new'."* v2 is the shipped recipe and the CONTROL every
  tanv1 arm is scored against; a control that moves is not a control. New staging work goes in new
  modules and may READ v2 (`prepare`, `emission_terms`, `viterbi`) all it likes. A primitive both
  engines need belongs in `common.rs` or its own module, never grafted into v2. This binds every
  agent and subagent. If v2 turns out to carry a real defect, say so and stop — do not fix it in
  passing.

- **No monolithic creations.** Many small, cohesive files/modules; deps point one way (leaf → app).
  Keep `whoop-protocol` sans-IO (no BLE/async leaks in). Never fold storage/algos/UI into the codec.
  200–400 lines/file typical. A new concern is usually a new small module or crate, not a bigger one.
- **After every code run, ALWAYS de-cruft before declaring done:** run `cargo clippy --all-targets`,
  then check for **dead code** and **duplicate code / duplicate logic**, and **refactor when possible**
  (shared helpers, one source of truth). Refactors must be **behaviour-preserving** — no wire / byte
  offset / CRC change (those are hardware-verified). Unused *public codec-parity API* is intentional
  (the FFI/CLI wires it later) — keep it; delete only genuinely-orphaned/redundant code.
- **ALWAYS verify dependencies are on the latest stable.** `cargo update` for semver-compatible; check
  `cargo update --dry-run --verbose` for anything "behind latest" and bump the manifest for 0.x-major
  jumps, then rebuild + test. (Baseline 2026-07-15: rustc 1.96.1, btleplug 0.12, tokio 1, uniffi 0.32,
  thiserror 2, clap 4, futures 0.3, uuid 1 — all latest stable.)
- **Documentation flow:** keep **in-run notes in `dev-docs/`** (git-ignored) while working; **after the
  run, fold them into usable overall docs in `docs/`** (update `docs/architecture.md`, don't leave a pile
  of handoff files). One authoritative `architecture.md`, not N floating dev docs. Provenance +
  per-crate clean-state confirm live in `dev-docs/{external-sources,crates}.md`.
- **Verify by READING the `cargo test` / `cargo build` output** (all passed, 0 warnings, 0 clippy), never
  a piped exit code. That invariant must hold after every change.
- **Gated writes only.** `command::FORBIDDEN`/`DESTRUCTIVE` refuse firmware-load/trim/DFU/config-write
  on the blind path; legitimate ones (reboot, R22) have dedicated intentional methods a UI opt-in gates.
- **Nothing pushed / no PR / no `git init`-and-push without explicit approval.**

## Verification round — THREE times per step, not once at the end

Running it only at the end is how a step finishes and then has to be unpicked. Run it:

```bash
# 1. BEFORE the step. If it is red now, the red is not yours - and you know that before you start.
python tools/verify_round.py --fast

# 2. DURING, after each meaningful edit. Four seconds, no cargo.
python tools/verify_round.py --fast

# 3. AFTER, in full, scoped to the step. This one runs the IGNORED suite.
python tools/verify_round.py --since=<the commit the step started from>

python tools/verify_round.py --self-test    # the checks must fire on a planted defect
```

**The round has its own tests**, `tools/test_verify_round.py`, and they are not optional either:

```bash
python tools/test_verify_round.py    # 26 checks; also FAILS if a new check_* arrives untested
```

`every_check_is_covered` enumerates the `check_*` functions by introspection and requires each to
appear in `COVERED`. That is the guard against the state this file was in on 2026-09-01, when four
of seven checks had no test and the round still reported green. Both directions are asserted for
every check: a planted defect must fire it AND a clean input must not, because a check that always
fires is as useless as one that never does.

The gates live in `tools/` so they are versioned with what they gate. They were outside every
repository until 2026-09-01 — no history, no diff, no recovery.

**The baseline run is the one that stops the backpedalling.** On 2026-09-01 the ignored suite failed
at the END of a step on a firmware gate that had nothing to do with that step: the documented
command used a relative `WHOOP_ZBIN_DIR`, and cargo runs each test binary with its own PACKAGE as
the working directory. A baseline run would have shown it red before a line was written.

The script does the MECHANICAL half: suite green, recorded counts re-derived rather than carried,
clippy 0, constants not defined twice, public items added by this step that nothing calls, dataset
columns the loader never opens, mixed line endings. Scoped to `--since` on purpose — over the whole
tree the orphan check reports 380 items, nearly all the deliberately-unwired FFI surface, and a
report nobody reads catches nothing.

**A green script does not mean the round passed.** These are the ones it cannot do, each of which
caught something real on 2026-08-31/09-01:

1. **Break every new gate and watch it fire.** Then check the BREAK APPLIED — a search-and-replace
   against CRLF with `\n` anchors patches nothing, and reads as "the gate is broken" when the gate
   was never tested. Assert the anchor matched.
2. **Re-run the headline on a disjoint slice.** A prefix is reproducible and is still one slice: 60
   MESA nights gave +0.088, a disjoint 60 gave +0.042, 400 settled it at +0.087.
3. **Any recall gain: print the predicted share beside it.** Recall is invariant to the cohort's
   class balance, NOT to the engine calling a class more. Sub-proportional is a loss wearing a gain.
4. **After a refactor, reproduce the numbers bit-for-bit.** Not "looks similar" — identical.
5. **Read every comment you wrote and ask whether the code does it.** One claimed a ridge sweep that
   did not exist until the claim was checked.
6. **Ask what the instrument CANNOT see**, and write that down beside the result. A confusion matrix
   cannot see a shuffle; the segment/confusion cross-check compares membership, not order.
7. **Trace a number to the code path that produced it**, not to the constant that names it. The
   written transition diagonal said long deep bouts were impossible; the decoder emits them anyway.

## Style

KISS / DRY / YAGNI. Match the surrounding style. Typed `thiserror` in libraries, `anyhow` only in the
binary. Records fork by version byte; per-generation wire diffs are data on `HeaderSpec`, matched in one
place (`framing`).

### Comments (strict)

- **NEVER extensive comments.** A comment is at most **3 lines**, and only when it earns its place. The
  3-line cap is for in-body `///`/`//`; a crate/module-overview `//!` header may run a little longer to orient.
- **Only** say **what it does** and **where it connects** (in-tree). No narration, no history, no rationale essays.
- **NEVER refer to external things** in a comment — no PR/issue numbers, URLs, spec/doc section refs, other
  repos, source filenames it was ported from, or firmware/hardware version strings. Those belong in
  `dev-docs/`, never in code. Byte offsets, scales, and invariants are fine; provenance is not.
