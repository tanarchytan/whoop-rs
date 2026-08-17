# whoop-rs

A from-scratch **Rust WHOOP BLE client** — one pure wire codec, a generic BLE core, and a WHOOP
client on top. Reusable as a desktop CLI today, and as the shared core of the Android app
([noop-tan](https://github.com/tanarchytan/noop)).

Own-device, right-to-repair, **non-commercial**. Firmware is **read-only, never flashed**. Every
derived health number is a wellness estimate, **never medical**.

## What it does

### CLI (`whoopctl`)

```
whoopctl scan                         # list every WHOOP in range (4.0 + 5.0/MG, unbonded)
whoopctl identify --address <MAC>     # read serial/firmware/model without bonding
whoopctl info --sn <serial>           # bonded read: serial, firmware, hardware, battery
whoopctl pack --sn <serial>           # battery-pack fuel gauge: serial, SOC%, millivolts
whoopctl sync --sn <serial>           # offload stored history to JSON Lines (keep-mode)
whoopctl sync --wipe --raw out.jsonl  # drain the full backlog incl. raw frames
whoopctl monitor --sn <serial>        # stream reassembled frames to stdout (optionally --type)
whoopctl hr --sn <serial>             # live bpm from the standard 2a37 char (--broadcast to enable it)
whoopctl metrics --sn <serial>        # sync, then compute HRV/SpO2/PPG-HR on-device
whoopctl r22on --sn <serial>          # unlock deep R22 biometric streams
whoopctl raw --sn <serial>            # switch optical collection on, capture v20/v21, switch it off
whoopctl wrist --set right --sn ...   # read the strap's wear block, optionally write the wrist
whoopctl buzz --sn <serial>           # haptic buzz
whoopctl reboot --sn <serial>         # soft reboot
whoopctl send <op-hex> [payload-hex]  # raw command (gated — see Safety)
whoopctl ingest capture.jsonl         # decode a raw-capture file to history records
whoopctl decode-capture deep.jsonl    # decode a deep-buffer capture and report the v20/v21 breakdown
whoopctl --hr-watch monitor --sn ...  # wellness HR-at-rest watch (display only, no buzz)
```

Target a band by `--address` (MAC), `--sn` (full serial or suffix), or `--name`. `scan` and
`identify` work over an unbonded BLE connection; the command channel (`info` / `sync` / `buzz` /
…) needs the LE bond (pair once through your OS Bluetooth settings).

### Derived metrics (pure, no BLE, no IO)

Gap-aware artifact-corrected HRV/RMSSD, SpO2 (4.0 paired red/IR and 5.0/MG sleep-computed %),
HR-from-PPG with sub-lag parabolic refinement, resting HR, respiration, HR zones, VO2-max,
recovery, strain, sleep staging, IMU features, the WHOOP calibration timeline, and per-strap
linear-fit coefficients. All exposed through the uniffi FFI for Kotlin and Swift.

### Protocol codec (pure, sans-IO)

The wire codec (`whoop-protocol`) has zero runtime dependencies besides `thiserror`. It handles
framing, CRC (CRC8/CRC16-Modbus/CRC32), 40 command opcodes, the historical offload state machine,
and record decode for the 4.0 v5/v7/v9/v12/v24/v25 and 5.0/MG v18/v20/v26 layouts plus the 100 Hz
6-axis IMU deep buffer. Everything is inner-relative — one decoder serves both generations for
shared record versions.

## Quick start

```bash
git clone https://github.com/tanarchytan/whoop-rs.git
cd whoop-rs
cargo build
cargo run -p whoopctl -- scan
```

```bash
# identify a band (unbonded), then bond and read its info
whoopctl identify --address <MAC>
whoopctl info --sn <serial>

# sync last night's history to JSON Lines (keeps the strap's copy)
whoopctl sync --sn <serial> > night.jsonl

# unlock + drain deep biometric streams (R22 opt-in)
whoopctl r22on --sn <serial>
whoopctl sync --wipe --raw capture.jsonl --sn <serial>
```

## Workspace

| Crate | Role |
|---|---|
| `whoop-protocol` | Pure sans-IO wire codec: framing, CRC, records, offload state machine, config/alarm/haptic builders |
| `physio-algo` | Pure derived metrics + scoring: HRV, resting HR, SpO2, PPG-HR, recovery, strain, stress, sleep staging, IMU features, calibration timeline |
| `ble-core` | `BleTransport` async trait + `Notification`/`BleError` + `MockTransport` |
| `ble-btleplug` | btleplug 0.12 backend behind `BleTransport` (the only crate that links a radio) |
| `whoop-client` | `WhoopClient<T>` — bond, history sync, monitor, gated writes, capture, backfill policy |
| `whoop-store` | SQLite per-(person, strap) nightly persistence + milestone-gated baselines |
| `whoopctl` | The clap CLI over the real radio |
| `whoop-ffi` | uniffi surface → Kotlin + Swift from one Rust source |

Full design + byte-level wire-protocol map: **[docs/architecture.md](docs/architecture.md)**.

## Safety

The write surface is gated in `whoop-protocol` + `whoop-client`:

- **`FORBIDDEN`** opcodes (firmware-load, trim, DFU, config-write, set-clock, adv-name,
  wrist-select) are refused on the blind `send` path. Legitimate writes (reboot, R22) have
  dedicated methods that a UI opt-in gates above the client.
- **`DESTRUCTIVE`** (force-trim, DFU) is never built from a payload a caller supplies — the FFI's
  generic command builder refuses it outright. On force-trim the danger is in the payload words, not
  the opcode, so its one door is a dedicated method that constructs its own bytes and takes no
  argument: `undo_trim`, which asks the strap to reset its history trim and read pointers back to
  oldest. `whoopctl undo-trim` sends the inert probe the strap declines unless
  `--i-agree-to-possible-loss-of-data` is passed. It moves pointers; it cannot un-erase flash.
- **History sync never deletes strap data.** The ACK only advances the read pointer; only
  FORCE-TRIM deletes, it lives in `FORBIDDEN`, and nothing sends it but `undo_trim`'s reset payload.
- `whoopctl` refuses `--wipe` on any serial matching the local `WHOOPCTL_PROTECT` allowlist.

## Status

The codec, btleplug backend, robustness layer, and `whoopctl` CLI are done. The uniffi surface is
the shipped decode + scoring core of the Android app. Validated on real 5.0 bands end-to-end —
scan, identify, bond, info, buzz, and a full overnight drain (35,310 records, 11.6 h) that decoded
v18 and located the sleep SpO2 (363 readings, 95-100%). v26 raw-PPG and v21 IMU decoders verified
against real captures; the 4.0 v24/v25 decoders are pinned to real captured 4.0 frames.

`cargo test --workspace` (1041 passed, 51 `#[ignore]`d) + `cargo clippy --workspace
--all-targets` are green. Zero warnings.

## Build notes

Pinned to the **MSVC** toolchain on Windows (btleplug's WinRT deps need the MSVC linker). On a
fresh clone: `rustup override set stable-x86_64-pc-windows-msvc` (requires VS Build Tools). The
pure crates (`whoop-protocol`, `physio-algo`, `ble-core`, `whoop-ffi`) build on
any toolchain. macOS and Linux build as-is (btleplug uses CoreBluetooth / BlueZ respectively).

The workspace currently patches btleplug to a sibling `../btleplug` fork for a WinRT
`add_peripheral` by-address fix. A fresh clone needs that sibling present until the fix lands
upstream. Drop the `[patch.crates-io]` section in the root `Cargo.toml` once merged.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).
Own-device interoperability with hardware you own. Not for commercial use. Firmware stays
read-only; health outputs are wellness estimates and never medical.
