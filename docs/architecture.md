# whoop-rs — architecture

A from-scratch **Rust WHOOP BLE client** for the WHOOP 4.0 ("Harvard", GEN4) and 5.0 / MG
("Maverick" / "puffin", GEN5) bands. One pure wire codec + a generic BLE core + a WHOOP client on
top, reusable as a desktop CLI, an Android app, and an iOS app from a single core.

Scope: own-device / right-to-repair, non-commercial. Firmware is **read-only, never flashed**. Any
derived health metric is a wellness estimate, never medical. The byte-exact field offsets and CRC
parameters are hardware-verified and live in the code doc-comments; this document is the map.

---

## 1. Workspace layout

Eight crates. Dependencies point strictly inward — the pure codec knows nothing about BLE or async,
the BLE core knows nothing about WHOOP, only one crate ever links a radio backend, the derived-metric
layer reads decoded records without touching BLE or IO, and a thin store persists per-(person, strap)
calibration to SQLite.

```
whoop-protocol   pure sans-IO wire codec  (deps: thiserror only)         ── no BLE, no async
      ▲   ▲
      │   └──── physio-algo     every derived metric (sleep, HRV, …)      ── no BLE, no IO
ble-core         generic BLE transport trait + MockTransport             ── no WHOOP
      ▲                        ▲
whoop-client ────────────┐     │
  WhoopClient<T>         │  ble-btleplug   btleplug 0.12 backend (the only crate that links a radio)
      ▲                  │     ▲
      │                  └─────┘
whoopctl (CLI)     whoop-ffi (uniffi → Kotlin + Swift)
```

| Crate | Role | Size | Tests |
|---|---|---|---|
| `whoop-protocol` | Pure wire codec: framing, CRC, hex, records (opt-in `serde` Serialize), offload state machine, config/haptic/alarm builders, live-r22 decoder | ~2570 LOC | 67 |
| `ble-core` | `BleTransport` async trait + neutral `Notification`/`BleError` + `MockTransport` | ~165 LOC | 1 |
| `ble-btleplug` | btleplug 0.12 backend behind `BleTransport` (scan/connect/subscribe/notify/write) | ~440 LOC | 1 |
| `whoop-client` | `WhoopClient<T: BleTransport>` — bond, history sync (keep/wipe + lossless frame tap), monitor, gated writes, capture encode+decode, backfill policy | ~900 LOC | 17 |
| `physio-algo` | Every derived metric, pure and sans-IO: the whole sleep pipeline (detect / stage / refine / main-night), HRV, recovery, strain, resting HR, respiration, SpO2, calories, workout, steps, IMU features, the ages, vital banding, the hydration goal and the calibration timeline. `algorithms.md` is its per-formula record | ~10680 LOC | 308 + 4 `#[ignore]`d |
| `whoop-store` | SQLite (rusqlite, bundled) per-(person, strap) nightly persistence + milestone-gated baselines and per-strap fits | ~340 LOC | 4 |
| `whoopctl` | clap CLI, 16 subcommands (`scan`/`identify`/`info`/`pack`/`sync`/`monitor`/`hr`/`send`/`metrics`/`raw`/`wrist`/`r22on`/`buzz`/`reboot`/`ingest`/`decode-capture`) + `--person`/`--db`/`--hr-watch`, split into `cli` / `report` / `main` | ~760 LOC | 4 |
| `whoop-ffi` | uniffi surface (depends on whoop-protocol **and** physio-algo): `WhoopCodec` (31 methods — decode history/live/response + offload + command frames) plus 79 derived-metric free fns to Kotlin + iOS from one Rust source. `data-flow.md` tables the surface | ~2710 LOC | 23 |

**425 tests, 4 `#[ignore]`d, 0 warnings, 0 clippy lints.** The per-crate column sums to 425 under a
workspace run: `whoopctl` turns on `whoop-protocol`'s `serde` feature, and one test module is gated on
it, so that crate alone reads 66. `ble-btleplug` unit-tests only its pure `matches_whoop`
predicate; the radio path is hardware-integration-verified (`whoopctl scan`), not mockable in-process.

Golden parity for the Android shadow-decode: `whoop-protocol` pins a 40-frame consecutive v26 PPG
burst and battery/wrist-on EVENT frames (`real_frames.rs`), and `physio-algo` pins the derived
`ppg_hr` estimates (`ppg_hr_real.rs`), all from a real 5.0 on-device capture. The exact f32→f64
gravity widen is asserted byte-for-byte on the worn v18 frame.

---

## 2. Design decisions (locked)

- **btleplug 0.12, behind a `BleTransport` trait.** Pure Rust, no C++/CMake FFI, native async
  `Stream`, BSD-3. Chosen over simpleble (C++/cmake build burden, BUSL-1.1, preview bindings).
  It only appears in `ble-btleplug`; swapping to another backend touches nothing else.
- **One WHOOP module + a closed `Family { Gen4, Gen5 }` enum** — not split 4.0/5.0 modules, not
  trait-objects, not per-gen crates. Every per-generation FRAME-HEADER difference is *data* on a
  `HeaderSpec`, matched in exactly one place (`framing`). What is not header shape stays a branch at
  its own site, and there are **8** of those outside `framing`/`family`: three in `whoop-client` (the
  GEN5 client hello, the per-generation clock opcode, the GEN5-only R22 enable), the GEN5 event
  residual in `live`, three GEN5-only record versions in `records`, and one label in `whoopctl`.
  Adding a third generation makes every `match` a compiler-enforced porting checklist; those 8 are
  `if family ==`, so they would compile and be wrong.
- **Decode everything inner-relative.** GEN5 frame-absolute offsets = GEN4 + 4, and the inner
  record starts at byte 8 (GEN5) vs 4 (GEN4), so the +4 cancels — one decoder serves both
  generations for the shared types (realtime/event/metadata). Records fork by the **version byte**,
  which is many-to-one onto generation, so shared versions aren't duplicated.
- **Sans-IO `Offload` state machine.** `on_frame(&Frame) -> Vec<OffloadStep>` (Record / Ack /
  Complete) — no async, no BLE. The client performs the returned steps; tests replay the same
  vectors with no radio; the FFI drives the identical machine on mobile.
- **The transport trait returns an owned `'static` notification stream** so a caller can hold it
  while still issuing writes on `&self` (the borrow that otherwise fights the checker).
- **Adjudicated decode = one source of truth.** Where the Kotlin app and this codec once diverged, each
  field was adjudicated against real captures + external RE and the better implementation lives here, not
  a blind clone of either side. Locked choices: **ppgHr** does a sub-lag parabolic ACF refine
  (`ppg_hr` in `physio-algo`) — a real-data win on the 838 night (sub-lag MAE 2.12 vs integer-lag 2.50
  bpm on the clean windows), never worse across a synthetic 45-150 bpm noise sweep, with an
  integer-fallback on a bad concave fit; **battery SOC** divides the integer deci-percent in f64
  (`Response.Battery.percent` / `BatteryPack.socPct` are f64, 999 → exact 99.9); **gravity** is gated to
  |g| ∈ [0.5,1.5] (drops non-finite/garbage the ungated path would store — never fired on 1861 real v18
  frames). Raw scalars (skin-temp, activity, spo2, resp, sleep-state, steps) decode-agree and are stored
  raw. The 5.0 sleep spo2 percent is decoded here even though the app does not yet store it. Adjudication
  evidence is in `whoop-research/decoder-adjudication/`.

These converge with both mature Rust prior-art implementations (openwhoop, goose); see
`dev-docs/external-sources.md`.

---

## 3. The wire protocol

Every multi-byte integer is little-endian. A frame is `header + inner + CRC32(inner)`, where the
inner record is `[type][seq][cmd][payload]`. The per-generation shape is the `HeaderSpec` in
`whoop-protocol/src/family.rs`; framing and deframing consult it and never branch on generation
themselves.

| | GEN4 (Harvard) | GEN5 (Maverick / puffin) |
|---|---|---|
| SOF | `0xAA` | `0xAA 0x01` |
| Inner starts at byte | 4 | 8 |
| Declared-length u16 at | 1 | 2 |
| Header CRC | crc8 (poly `0x07`) over `frame[1..3]` | crc16-modbus (poly `0xA001`, init `0xFFFF`) over `frame[0..6]` |
| Payload CRC | crc32 (zlib, `0xEDB88320` reflected) over inner | same |
| Inner pad | none | zero-pad to a 4-byte multiple before length/CRC |

Frame integrity is reported via `Frame::crc_ok`, not an error — a structurally broken frame errors,
a checksum mismatch decodes with `crc_ok = false` (the offload machine drops those, so a forged
HISTORY_END can't advance the strap's trim cursor over unstored data).

The GEN5 CLIENT_HELLO is a byte-exact `GET_HELLO` frame (`whoop-protocol/src/hello.rs`), verified
against a real 5.0 frame and covered by a byte-identical encode test.

**Packet types** (`packet.rs`): 35 COMMAND, 36 COMMAND_RESPONSE, 40 REALTIME_DATA, 43
REALTIME_RAW_DATA, 47 HISTORICAL_DATA, 48 EVENT, 49 METADATA, 50 CONSOLE_LOGS, plus the puffin
aliases (37/38/53/54/56) which `canonical()` folds onto their base so routing is generation-agnostic.

**Historical record versions** (fork by version byte, `records/`): GEN4 = v5/7/9/12/24/25 (v24 = full
DSP block: HR, R-R, gravity, SpO₂ red/IR, skin-temp, respiration); GEN5/MG = v18 (per-second HR, R-R,
gravity, skin-temp, steps, activity, sleep-state, **and a sleep-only computed SpO₂ percent at inner 74** —
a tri-mode byte, %-range only), v20 (the R22 optical block — decoded, but never seen on a worn capture),
v26 (raw 24 Hz optical buffer), and the
100 Hz raw 6-axis IMU deep buffer (accel + gyro; the live IMU stream is firmware-refused, so this
historical path is the only way to reach it). The IMU buffer is identified by its own length + in-packet
sample counts, **not** a version byte (its place in the version scheme is unconfirmed — noop #423/#455),
so `decode` tries it first for GEN5; a normal v18/v26 frame is far too short to pass that gate. An
unmapped GEN4 version falls back to the v24 layout only if it passes a strict plausibility gate
(HR 25..230 and |g| ≈ 1); an unmapped GEN5 version is skipped. **These deep buffers only flow after the
R22 opt-in** — run `enable_r22()` before `sync_history()` (the official app never requests them). The
2140 B optical deep buffer (PPG/SpO₂/BP) is still undecoded upstream — captured raw only.

**Historical offload** (`offload.rs`): SEND_HISTORICAL_DATA(22) → type-47 chunks interleaved with
METADATA(HISTORY_START/END/COMPLETE); each HISTORY_END is answered with a HISTORICAL_DATA_RESULT(23)
ACK that echoes the 8-byte end_data — mandatory, or the strap re-serves the same chunk forever.

**Config / R22** (`config.rs`): SET_CONFIG(0x78) writes one feature flag (40-byte body);
SET_DEVICE_CONFIG(0x77) writes one device value (33-byte body). The 16-flag R22 sequence unlocks the
deep biometric (type-0x2F) streams. Reversible — it only changes what the strap banks. Deep records
arrive via history offload, not a live stream. `r22_frames(start_seq)` is the single source of the
16-frame sequence, shared by the client and the FFI.

**Bond order** (proven on hardware): connect → negotiate MTU → discover → subscribe standard
HR(2A37)/battery(2A19) → write CLIENT_HELLO **confirmed** (the just-works bond; the confirmed write
returns only once ATT-acked) → subscribe the encrypted vendor group (`fd4b0003/4/5/7`). Subscribing
the encrypted chars before the bond returns Insufficient-Authentication and wedges the link.

---

## 4. Safety model

The write surface is gated in `whoop-protocol/src/command.rs` + `whoop-client`:

- `FORBIDDEN` — firmware-load / trim / DFU / reboot / config-write / set-clock / adv-name /
  wrist-select opcodes. `send_raw` refuses them; the legitimate ones (reboot, R22, wrist-select) have
  dedicated, intentional methods that a UI opt-in gates above the client.
- `DESTRUCTIVE` — the subset that is never built from a payload a caller supplies (force-trim, DFU): the
  FFI's generic `command_frame` returns `None` for these, so no Kotlin caller can assemble one. The danger
  on force-trim is in the words, not the opcode, so the list stays as it is and the one door is a dedicated
  method that constructs its own bytes and takes no argument (`undo_trim`, below).
- Every gated write (`enable_r22`, `set_broadcast_hr`, `buzz`, `reboot`, `select_wrist`,
  `optical_collection`) is reversible /
  non-destructive and numbered from the client's running seq counter.
- **`undo_trim` is the one gated write that touches a destructive opcode.** It asks the strap to reset its
  history trim and read pointers back to oldest — FORCE_TRIM with both payload words set to the firmware's
  reset sentinel. Every other word pair on that opcode is a real trim, so the eight bytes live in
  `whoop-protocol/src/trim.rs`, the two builders there are the only source of a payload, and neither the
  method nor the `whoopctl undo-trim` subcommand accepts one. `undo_trim_dry_run` is its positive control:
  the same opcode and handler carrying the inert sentinel the strap declines, which the CLI sends unless
  `--i-agree-to-possible-loss-of-data` is passed. This moves pointers only — it cannot un-erase flash, so
  whether any already-offloaded record is still resident is not something the command can promise. Not on
  the FFI surface: desktop only.
- History offload drops CRC-bad frames so a forged HISTORY_END can't advance the trim cursor.
- **History sync never deletes the strap's data.** The ACK (`0x17`) only advances the strap's read
  pointer to release the next chunk; it does not free the buffer — only FORCE_TRIM (`0x19`, in FORBIDDEN,
  and sent by nothing but `undo_trim`'s pointer-reset payload) deletes. Keep-mode (`sync`, no `--wipe`) reads the first chunk without acking and leaves the
  read pointer put; `--wipe` acks forward through the whole backlog (needed to reach nights banked behind an
  already-consumed edge) but still deletes nothing — the records persist on the strap until it overwrites
  them. Each record reaches the sink **before** its chunk's ACK, so a sink error aborts before advancing,
  and a CRC-bad HISTORY_END can't move the pointer over unstored data.
- `whoopctl` refuses `--wipe` on any serial matching the local `WHOOPCTL_PROTECT` allowlist
  (comma-separated suffixes, checked against the serial read off the band, not the typed `--sn`), and
  refuses when the serial can't be read to verify it — a fail-safe, environment-local guard.
- `disconnect()` drops the link cleanly so an exclusive-bond band isn't left held into the next run.

---

## 5. Desktop client + CLI

`WhoopClient<T: BleTransport>` (`whoop-client/src/client.rs`) maps logical `Channel`s to per-family
vendor UUIDs (`uuids.rs`), runs `connect_and_bond`, and drives the pure `Offload` off the
notification stream. The read paths (`info`, `monitor`) share one `collect_frames` pump; `sync_history_with`
keeps its own loop because it needs an idle-reset timeout, async ACK writes, and an early return on
HISTORY_COMPLETE. It opens the notification stream **before** any triggering write (the
write-before-listen race fix) and aborts a stalled drain after an 8 s inactivity window. It streams
each decoded record to a caller sink before that chunk's ACK; the `ack` flag decides keep (first chunk,
no trim) vs wipe (full drain). `sync_history_capturing` adds a second tap that sees **every** reassembled
frame — including ones the offload machine drops as undecodable — which is how unmapped record versions
surface. `whoopctl sync` serializes each record to JSON Lines via the codec's opt-in `serde` feature;
`--raw <file>` additionally writes the lossless per-frame capture JSONL (the `capture.rs` schema, fed to
`tools/capture-layout.py`) and prints a frame histogram that flags any history version that didn't decode.

`whoop-client` also ports two pure noop policies: `BackfillPolicy` (`policy.rs` — 900 s periodic /
90 s event floors, empty-streak 2ⁿ backoff, clock-untrusted skip, reconnect 3/6/12/24/48/60) and the
lossless capture-tap JSONL (`capture.rs` — snake_case, byte-identical to noop `BackfillCaptureJsonl`,
so its output feeds the existing `tools/capture-layout.py` unchanged).

`whoopctl` is the clap CLI over the real btleplug transport: `scan` (list bands in range, each with its
identity read from GATT — scans every generation via an empty filter + name/service match, so a renamed or
4.0 band still appears), `identify` (connect unbonded, read name/serial/fw/hardware), `info`
(identity/battery/extended-fuel-gauge/data-range; a 4.0 serial+firmware come from the GET_HELLO_HARVARD reply
since the 4.0 omits the DeviceInfo GATT service), `pack` (5.0 battery-pack fuel gauge — serial/SOC/mV/pack-id),
`sync` (decode history to JSON Lines; keeps the strap by default, `--wipe` to drain), `monitor` (stream frames),
`send` (one opcode, FORBIDDEN-refused, prints the raw response), `undo-trim` (ask the strap to rewind its
history read/trim pointers to oldest; sends the inert probe unless `--i-agree-to-possible-loss-of-data` is
passed, and takes no payload either way), `r22on`, `buzz`, `reboot`, `wrist` (read the
body-location block, `--set left|right` to write it first), `ingest` (backfill
the calibration store from a saved capture). A band is targeted with `--address` (surest), `--sn`
(full or suffix — connects to each candidate and reads its serial, since the serial isn't advertised),
or `--name`. On-band behaviour is validated here, not in unit tests — a connect proves nothing about a
real strap.

The WHOOP identity (name/serial/model/fw/hardware) lives in standard GATT chars readable **without** a
bond, not in the advertisement (which is just the service UUID + RSSI) — so `scan`/`identify` work
unbonded, while the command channel (`info`/`sync`/`buzz`/…) needs the LE bond. Names are
user-customizable, so the serial (`0x2A25`) is the reliable ID. GATT strings are NUL-padded, cleaned via
`ble_core::gatt_string`.

Build/run: `cargo run -p whoopctl -- scan`.

**Derived metrics (`physio-algo`).** A pure analytics layer over decoded records — no BLE, no IO, no
algorithms in the codec. It is far wider than the few named here — the whole sleep pipeline, recovery,
strain, the ages — and `algorithms.md` is the per-formula record; what follows is only the part this
section's CLI story needs. `HrvReadiness` reads a log-domain baseline against a personal-normal ±0.5 SD band
from a nightly RMSSD series and returns a tier, `None` while calibrating. SpO2 has two paths: the 4.0 v24
paired red/IR via ratio-of-ratios (which a pulsatility gate then WITHHOLDS on real 4.0 data — see
`algorithms.md`), and the **5.0/MG computed scalar** the strap writes at v18 inner 74
during sleep — a tri-mode byte (a %-range value is a real reading; bit-7 saturation sentinels and sub-70
diagnostic codes gate to `None`), verified on a real overnight drain (363 sleep readings, 95-100%). The raw
red/IR pair still does not cross 5.0 BLE; the strap computes the % on-device. `calibration` encodes WHOOP's
per-feature unlock/full schedule (blood-oxygen 1 night, recovery 3, skin-temp 7, …) so readouts gate on the
same periods the app uses. `linear_fit(field, reference)` is the per-strap coefficient primitive — a
universal client computes a device-relative field's `{scale, offset, r}` from that strap's own captures
instead of hardcoding another strap's number. Metrics return `None` rather than a fabricated value.

**Calibration store (`whoop-store`).** Each `sync`/`ingest` segments a drain into nightly summaries (SpO2
median, RMSSD, HR range) and persists them to SQLite keyed on **(person, strap)** — a new person, or the
same person on a new strap, is a fresh key, so a shared or handed-off strap never calibrates against another
wearer's data. A metric's baseline finalizes once its (person, strap) reaches the calibration milestone.
`whoopctl --person <id>` picks the wearer; `ingest <capture>` backfills from a saved raw JSONL, no band.
Nightly RMSSD is **gap-aware and artifact-corrected**: the beats flatten in time order, then range-filter
([300,2000] ms) and Malik ectopic-clean (radius 2, reject over 20 % of the local median) tracking a
contiguity mask, and only successive differences whose two beats were adjacent in the source pool, divided
by the contiguous-pair count — on clean data this equals the plain Task-Force RMSSD (validated on the 838
drain: 29-37 ms, vs 150-184 ms without it). This cleaning is the single shared path behind `rmssd_gap_aware`
and the windowed `hrv_windowed_avg` (bit-for-bit with the Kotlin front-end's Malik path). `calibrate_fit` wires `linear_fit` into the store:
a device-relative field is fitted against a reference over the accumulated nights and the `{scale, offset,
r}` persisted to a `fit_baseline` table once the milestone is met. (Known limit: nights bucket by UTC
calendar day, so a sleep straddling UTC midnight splits across two rows — a physiological-day cutover is a
TODO.)

---

## 6. Mobile — one core, native radios, universal easy-connect

The sans-IO design pays off across platforms: **no BLE and no async cross the FFI.** `whoop-ffi`
(uniffi) exposes a `WhoopCodec` object that mirrors the whole client, so a native app can drop its own
decoder:
- **decode** — `feed(chan, bytes) -> [Step]` (reassemble + drive offload), `decode_history` (the full
  `HistorySummary` — every decoded field), `decode_live` (realtime HR/R-R, on-wrist r22, event/battery, console), `decode_response`
  (identity/battery/clock/data-range/firmware).
- **command frames to write** (the FFI never writes) — `client_hello` / `offload_start` / `offload_abort` /
  `r22_frames` / `get_hello`/`get_battery`/`get_data_range` / `stop_raw_flood` / `toggle_realtime_hr` /
  `reboot` / `buzz` / `broadcast_hr` / `set_config` / `alarm_set`/`alarm_disable`.
- **derived metrics** — 79 free fns, every one reached from hand-written Kotlin, one of them
  (`spo2_rolling_reading`) only through a Kotlin wrapper the app never calls. A few of the shapes:
  `ppg_hr`, `hrv_rmssd_gap_aware`, `hrv_windowed_avg` (the app's stored session `avgHrv` — the mean of
  per-5-min-bucket gap-aware RMSSD over a span), `hrv_readiness`, `analyze_sleep_with`, `recovery_score`,
  `nightly_spo2_raw_means` (integer-truncated 4.0 raw red/IR ADC means over the in-bed spans — raw ADC,
  never a calibrated percent). `data-flow.md` tables most of them with the count it covers; this is not
  the list.

Each platform does its own BLE natively and feeds notification bytes in. The loop this surface is built for
is: map notify UUID → `Chan`, call `feed`, and for each `Step` persist a record / write an `Ack` frame
confirmed / stop on `Complete` — `whoopctl` runs it; **noop-tan does not**, reassembling and driving the
drain in Kotlin and calling the per-frame decoders instead (`data-flow.md`). What stays native: the radio,
write orchestration (seq/gates), JSONL capture, standard-GATT profiles, and timezone→epoch. Bindings
generate via `uniffi-bindgen generate --language kotlin|swift`.

**Universal easy-connect** — "attach to a band already connected (held by the WHOOP app) without
scanning" — is a per-OS primitive, wired natively behind the same `attachToConnectedWhoop()` contract:

| OS | Easy-connect primitive |
|---|---|
| Android | `BluetoothManager.getConnectedDevices(GATT)` → attach our own `BluetoothGatt` (one ACL is multiplexed across apps — proven on David's band) |
| iOS / macOS | `CBCentralManager.retrieveConnectedPeripherals(withServices: [fd4b0001])` → `connect` |
| Desktop | no shared-ACL concept → btleplug scan by service (Linux: query BlueZ for already-connected) |

Packaging is toolchain-gated: Android via `cargo-ndk` → `.so` per ABI + Kotlin binding (needs the NDK);
iOS via an `.xcframework` for `aarch64-apple-ios` + SwiftUI (needs a Mac + Xcode).

---

## 7. Build, test, toolchain

```bash
cd whoop-rs
cargo build           # whole workspace
cargo test            # 425 tests (4 #[ignore]d dataset gates)
cargo clippy --all-targets
cargo run -p whoopctl -- scan
```

**Toolchain: pinned to MSVC** via an in-directory `rustup override` — btleplug's WinRT deps use
raw-dylib and need the MSVC linker (the windows-GNU `dlltool` is incomplete). On a fresh clone:
`rustup override set stable-x86_64-pc-windows-msvc` (requires VS Build Tools). This is a
Windows-toolchain detail only; the code is portable to Linux/macOS.

---

## 8. Status & next

- Phases **A** (codec complete), **B** (btleplug backend), **C** (robustness), **D** (`whoopctl` CLI) —
  **done**. The mobile foundation (`whoop-ffi`) compiles on host.
- **On-hardware (verified across real 5.0 bands):** `scan`, `identify`, and `--address`/`--sn`
  targeting work over an unbonded connection. The command channel is bond-gated (`0x80650005`); the
  WinRT `ConfirmOnly` pairing in `ensure_paired` establishes a bond WHOOP accepts (band in pairing mode
  the first time, off-body triple-double-tap), so `info`/`buzz`/`sync` all run bonded.
- **Get-and-decode + SpO₂ proven end-to-end.** A full overnight `--wipe` drain of a real 5.0 band
  (5AG0507838, 35 310 records, 11.6 h) decoded v18 (HR, gravity, skin-temp, steps, activity, sleep-state)
  **and located the computed SpO₂ at inner 74** (363 sleep readings, 95-100%), independently confirmed by
  external RE (`whoop-research/`). The per-(person, strap) calibration store ingested it into a finalized
  SpO₂ baseline (98%).
- **v26 raw-PPG + v21 IMU decoders validated.** The 838 dump's 1160 v26 frames validate `gen5::v26` +
  `ppg_hr` end-to-end (982 HR estimates, MAE 4.31 bpm vs the strap's own v18 HR, 89.5% within 5 bpm). v21's
  offsets and scales are **verified-by-crosscheck** — byte-for-byte identical to noop's hardware-verified
  gravity-shell capture; only the 100 Hz sample rate is inferred, and a native worn R22 capture is still the
  nice-to-have ground-truth stamp.
- **Next:** a physiological-day bucket for nightly metrics, a real worn R22 capture (v20 optical layout,
  live IMU streams), the Android app (cargo-ndk + Kotlin BLE + easy-connect + `WhoopCodec`), then iOS
  (Mac-gated). An opt-in wellness HR-at-rest watch is designed and awaiting sign-off — there is **no
  cardiac-arrest detector on the 5.0** (AFib/ECG is HeartKey/electrode, MG-only and firmware-blocked; the
  v18 "cardiac" bytes are PPG signal-quality/status, not events).

Working notes, source provenance, and the per-crate clean-state confirm live in the git-ignored
`dev-docs/` folder.
