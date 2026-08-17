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
cargo test                 # 1041 passed, 0 failed, 51 #[ignore]d  (2026-08-17; re-derive, never carry forward)
cargo clippy --all-targets
cargo run -p whoopctl -- scan

# The sleep corpus. `cargo test` does NOT reach it — every cohort gate is #[ignore]d behind a
# multi-GB tree outside the repo, so a green suite says nothing about staging. Run it whenever
# anything under sleep/ changes:
cargo test --release -p physio-algo --test dataset_parity -- --ignored --nocapture
#   expect: 5 passed, 0 failed; dreamt 0.3123 (n=100), aauwss 0.4103 (n=13), sleep-accel 0.3780 (n=31)
#   corpus: sleep-benchmark/fixtures_multi_clean3 (the pinned root; WHOOP_SLEEP_FIXTURES overrides)
```

Pinned to **MSVC** via an in-dir `rustup override` (btleplug's WinRT deps need the MSVC linker; the
windows-GNU `dlltool` is incomplete). Fresh clone: `rustup override set stable-x86_64-pc-windows-msvc`.

The workspace `[patch.crates-io]`es `btleplug` to a sibling `../btleplug` fork (the WinRT `add_peripheral`
by-address fix). A fresh clone needs that sibling checkout present until the upstream PR lands; drop the
patch once it does.

## Guardrails (always on)

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
