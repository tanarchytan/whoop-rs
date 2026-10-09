//! The stateful per-band codec object and the command frames the app writes back. The only
//! interior-mutable surface in this crate.

use crate::*;

pub(crate) fn result_to_u8(r: whoop_protocol::event::ResultCode) -> u8 {
    use whoop_protocol::event::ResultCode::*;
    match r {
        Failure => 0,
        Success => 1,
        Pending => 2,
        Unsupported => 3,
        Unknown(x) => x,
    }
}

/// The stateful codec the app drives: one per connected band. Interior-mutable so it presents the
/// `&self` methods uniffi objects require while owning the reassembler + offload state.
#[derive(uniffi::Object)]
pub struct WhoopCodec {
    family: Family,
    inner: Mutex<Inner>,
}

struct Inner {
    deframers: DeframerMap,
    offload: Offload,
}

#[uniffi::export]
impl WhoopCodec {
    #[uniffi::constructor]
    pub fn new(gen: Gen) -> Arc<Self> {
        let family: Family = gen.into();
        Arc::new(WhoopCodec {
            family,
            inner: Mutex::new(Inner { deframers: DeframerMap::new(family), offload: Offload::new(family) }),
        })
    }

    /// The GEN5 bond-open frame — write it (confirmed) after connecting to trigger the just-works bond.
    pub fn client_hello(&self) -> Vec<u8> {
        GEN5_CLIENT_HELLO.to_vec()
    }

    /// The SEND_HISTORICAL_DATA frame that starts a drain.
    pub fn offload_start(&self) -> Vec<u8> {
        self.inner.lock().expect("Mutex poisoned — a prior codec panic left the inner state corrupted").offload.start_frame()
    }

    /// The 16 R22 deep-data enable frames (write each confirmed, ~80 ms apart). Reversible.
    pub fn r22_frames(&self) -> Vec<Vec<u8>> {
        config::r22_frames(0)
    }

    /// Feed one native notification (its channel + bytes): reassembles frames and drives the offload,
    /// returning the steps to perform (persist records, write ACKs, stop on Complete).
    pub fn feed(&self, chan: Chan, bytes: Vec<u8>) -> Vec<Step> {
        let mut inner = self.inner.lock().expect("Mutex poisoned — a prior codec panic left the inner state corrupted");
        let frames = inner.deframers.push(chan.into(), &bytes);
        let mut out = Vec::new();
        for f in frames {
            for step in inner.offload.on_frame(&f) {
                out.push(to_step(step));
            }
        }
        out
    }

    /// Decode a single complete history frame (for offline replay of a captured file). A bad-CRC frame is
    /// rejected here (None) — a forged/corrupt record must not be stored past the trim.
    pub fn decode_history(&self, raw: Vec<u8>) -> Option<HistorySummary> {
        let frame = framing::decode(self.family, &raw).ok()?;
        if !frame.crc_ok {
            return None;
        }
        match records::decode(&frame)? {
            Record::History(h) => Some(h.into()),
            _ => None,
        }
    }

    /// Decode a single METADATA frame's offload-state fields (meta_type + unix + trim_cursor + crc_ok), so
    /// the app's offload state machine can recognise HISTORY_END/COMPLETE. `None` off a non-metadata frame.
    pub fn decode_metadata(&self, raw: Vec<u8>) -> Option<MetadataInfo> {
        let f = framing::decode(self.family, &raw).ok()?;
        if !matches!(f.packet().canonical(), PacketType::Metadata) {
            return None;
        }
        let m = live::metadata(&f)?;
        Some(MetadataInfo { meta_type: m.meta_type, unix: m.unix, trim_cursor: m.trim_cursor, crc_ok: f.crc_ok })
    }

    /// Decode a single v26 PPG waveform frame (the 24 optical samples) for offline replay / the deep buffers.
    /// Bad-CRC frames are rejected; `decode_ppg` version-gates, so a v18/v24 frame yields None (not 24 samples).
    pub fn decode_ppg_frame(&self, raw: Vec<u8>) -> Option<PpgFrame> {
        let f = framing::decode(self.family, &raw).ok()?;
        if !f.crc_ok {
            return None;
        }
        let p = records::decode_ppg(&f)?;
        Some(PpgFrame { unix: p.unix, record_id: p.record_id, samples: p.samples })
    }

    /// Decode a single v21 6-axis IMU buffer frame (accel/gyro columns) for the deep-buffer diagnostics.
    pub fn decode_imu_frame(&self, raw: Vec<u8>) -> Option<ImuFrame> {
        let f = framing::decode(self.family, &raw).ok()?;
        let m = records::decode_imu(&f)?;
        Some(ImuFrame {
            unix: m.unix,
            sample_rate_hz: m.sample_rate_hz,
            accel: m.accel.into_iter().flatten().collect(),
            gyro: m.gyro.into_iter().flatten().collect(),
        })
    }

    /// Drop any buffered partial frames — call on (re)connect.
    pub fn reset(&self) {
        self.inner.lock().expect("Mutex poisoned — a prior codec panic left the inner state corrupted").deframers.reset();
    }

    /// Decode a single live-notify frame (realtime HR/R-R, on-wrist r22 biometric, event/battery, or
    /// console text). Stateless; the app routes live-channel frames here, offload stays in `feed`.
    pub fn decode_live(&self, raw: Vec<u8>) -> Option<Live> {
        let f = framing::decode(self.family, &raw).ok()?;
        match f.packet().canonical() {
            PacketType::RealtimeData => live::realtime(&f).map(|r| Live::Realtime {
                unix: r.timestamp,
                heart_rate: r.heart_rate,
                rr_intervals: r.rr_intervals,
            }),
            PacketType::Event => live::event(&f).map(|e| {
                let batt = live::battery_event(&f);
                Live::Event {
                    number: e.number,
                    unix: e.timestamp,
                    battery_soc_deci: batt.map(|b| b.soc_deci),
                    battery_millivolts: batt.map(|b| b.millivolts),
                    battery_charging: batt.map(|b| b.charging),
                    payload_hex: live::event_payload_hex(&f),
                }
            }),
            PacketType::ConsoleLogs => console::text(&f).map(|text| Live::Console { text }),
            PacketType::HistoricalData if matches!(f.cmd(), 0x80 | 0x82) => live::r22_live(&f).map(|r| Live::R22 {
                unix: r.timestamp,
                hr_ch1: r.hr_ch1,
                hr_ch2: r.hr_ch2,
                accel: r.accel.to_vec(),
            }),
            _ => None,
        }
    }

    /// Decode a single command-response frame (identity/battery/clock/data-range/firmware).
    pub fn decode_response(&self, raw: Vec<u8>) -> Option<Response> {
        use response::CommandResponse as C;
        let f = framing::decode(self.family, &raw).ok()?;
        let (resp_cmd, result) = response::resp_status(&f);
        let result = result.map(result_to_u8);
        Some(match response::decode(&f)? {
            C::Battery { percent } => Response::Battery { resp_cmd, result, percent },
            C::Clock { unix } => Response::Clock { resp_cmd, result, unix },
            C::Hello { device_name, fw_version } => {
                Response::Hello { resp_cmd, result, device_name, fw_version: fw_version.map(|v| v.to_vec()) }
            }
            C::DataRange { oldest, newest } => Response::DataRange { resp_cmd, result, oldest, newest },
            C::VersionInfo { fw } => Response::Version { resp_cmd, result, fw: fw.to_vec() },
            C::ExtendedBattery { millivolts, remaining_mah, current_ma } => {
                Response::ExtendedBattery { resp_cmd, result, millivolts, remaining_mah, current_ma }
            }
            C::BatteryPack { serial, soc_pct, bt_addr } => {
                Response::BatteryPack { resp_cmd, result, serial, soc_pct, bt_addr }
            }
            C::NoBatteryPack => Response::NoBatteryPack { resp_cmd, result },
            C::Other { .. } => Response::Other { resp_cmd, result },
        })
    }

    // --- Command frames: return the bytes for the app to write (confirmed); the FFI never writes. ---

    /// Cleanly abort an in-flight history drain.
    pub fn offload_abort(&self) -> Vec<u8> {
        self.inner.lock().expect("Mutex poisoned — a prior codec panic left the inner state corrupted").offload.abort_frame()
    }

    /// GET_HELLO (identity + firmware), family-forked: Gen5 takes the `[0x01]` b3 selector, Gen4 the
    /// GET_HELLO_HARVARD opcode. The Gen4 body here is one byte, matching what the app sends;
    /// `whoop-client`'s `info` sends nine. Unsettled — no 4.0 has run against either on this radio.
    pub fn get_hello_frame(&self, seq: u8) -> Vec<u8> {
        match self.family {
            Family::Gen5 => framing::command(self.family, seq, command::GET_HELLO, &[0x01]),
            Family::Gen4 => framing::command(self.family, seq, command::GET_HELLO_HARVARD, &[0x00]),
        }
    }

    /// GET_BATTERY_LEVEL (also the bond-establishing write on 5/MG).
    pub fn get_battery_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::GET_BATTERY_LEVEL, &[0x00])
    }

    /// GET_DATA_RANGE (the oldest/newest banked seconds).
    pub fn get_data_range_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::GET_DATA_RANGE, &[0x00])
    }

    /// Stop the live type-43 raw flood (sent during the handshake).
    pub fn stop_raw_flood_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::SEND_R10_R11_REALTIME, &[0])
    }

    /// Toggle the realtime HR/R-R stream on the vendor channel.
    pub fn toggle_realtime_hr_frame(&self, seq: u8, on: bool) -> Vec<u8> {
        framing::command(self.family, seq, command::TOGGLE_REALTIME_HR, &[u8::from(on)])
    }

    /// Warm-reboot the strap (data kept). Confirmation-gated in the app.
    pub fn reboot_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::REBOOT_STRAP, &[])
    }

    /// One-shot 5/MG buzz.
    pub fn buzz_frame(&self, seq: u8) -> Vec<u8> {
        self.buzz_frame_with(seq, 1)
    }

    /// 5/MG buzz of `loops` pulses (0 and 1 are one pulse; more than 8 is clamped to 8).
    pub fn buzz_frame_with(&self, seq: u8, loops: u8) -> Vec<u8> {
        haptic::maverick_buzz_frame(seq, loops)
    }

    /// SET_DEVICE_CONFIG to advertise standard 0x180D HR (Garmin/Edge). Opt-in gated in the app.
    pub fn broadcast_hr_frame(&self, seq: u8, on: bool) -> Vec<u8> {
        config::device_frame(seq, "whoop_live_hr_in_adv_ind_pkt", if on { b'1' } else { b'0' })
    }

    /// SET_CONFIG for one runtime-named feature flag. Opt-in / deep-data gated in the app.
    pub fn set_config_frame(&self, seq: u8, name: String, value: u8) -> Vec<u8> {
        config::feature_frame_named(seq, &name, value)
    }

    /// SET_ALARM_TIME (5/MG, 20-byte body). `wake_epoch_ms` is resolved app-side. Experimental, UI-gated.
    pub fn alarm_set_frame(&self, seq: u8, wake_epoch_ms: u64, alarm_id: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ALARM_TIME, &alarm::build(wake_epoch_ms, alarm_id))
    }

    /// DISABLE_ALARM (5/MG form).
    pub fn alarm_disable_frame(&self, seq: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::DISABLE_ALARM, &alarm::disable_rev2())
    }

    /// Generic outbound COMMAND builder for opcodes without a dedicated method (get-clock/version,
    /// alarm-readback, run-alarm, stop-haptics, historical-data-result, plus the gated set-clock/adv-name
    /// the app allow-lists). Refuses the genuinely-destructive set (trim/DFU) so it can't be built here.
    pub fn command_frame(&self, seq: u8, cmd: u8, payload: Vec<u8>) -> Option<Vec<u8>> {
        if command::is_destructive(cmd) {
            return None;
        }
        Some(framing::command(self.family, seq, cmd, &payload))
    }

    /// SET_CLOCK — the 8-byte form newer firmware latches. Gated in the app.
    pub fn set_clock_frame(&self, seq: u8, now_unix: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_CLOCK, &clock::set_clock_payload(now_unix))
    }

    /// SET_CLOCK — the legacy 9-byte form older 4.0 firmware needs (a no-op on newer). Sent alongside the
    /// 8-byte form so either firmware latches.
    pub fn set_clock_legacy_frame(&self, seq: u8, now_unix: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_CLOCK, &clock::set_clock_payload_legacy(now_unix))
    }

    /// SET_ALARM_TIME — the WHOOP 4.0 9-byte body (minute-precision). `wake_epoch_secs` is resolved
    /// app-side. Experimental, UI-gated.
    pub fn alarm_set_frame_gen4(&self, seq: u8, wake_epoch_secs: u32) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ALARM_TIME, &alarm::whoop4_build(wake_epoch_secs))
    }

    /// RUN_HAPTICS_PATTERN — the 4.0 preset buzz `[pattern_id][loops][0][0][0]` (pattern 2 = graduated
    /// alarm buzz). On 5/MG the app remaps to the maverick buzz instead.
    pub fn run_haptics_frame(&self, seq: u8, pattern_id: u8, loops: u8) -> Vec<u8> {
        framing::command(self.family, seq, command::RUN_HAPTICS_PATTERN, &haptic::run_haptics_pattern(pattern_id, loops))
    }

    /// SET_ADVERTISING_NAME — rename the 4.0 strap's BLE advertising name (clamped to 24 UTF-8 bytes). The
    /// strap reboots to apply. Gated in the app.
    pub fn advertising_name_frame(&self, seq: u8, name: String) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ADVERTISING_NAME, &advertising::advertising_name_payload(&name))
    }

    /// SET_ADVERTISING_NAME (5/MG) — rename the strap's puffin BLE advertising name (clamped to 24 UTF-8
    /// bytes). The strap reboots to apply. Hardware-unverified (opcode/b3 inferred). Gated in the app.
    pub fn advertising_name_frame_gen5(&self, seq: u8, name: String) -> Vec<u8> {
        framing::command(self.family, seq, command::SET_ADVERTISING_NAME, &advertising::advertising_name_payload_gen5(&name))
    }
}

/// Newest plausible unix banked, scanning EVERY byte offset of a GET_DATA_RANGE frame and preferring the
/// newest non-future word (falls back to newest-any). This is the sync gate — it REPLACES the fixed-offset
/// `Response::DataRange` newest read.
#[uniffi::export]
pub fn data_range_newest(frame: Vec<u8>, wall_now_unix: u64, future_skew_seconds: u64) -> Option<u32> {
    response::data_range_scan_newest(&frame, wall_now_unix, future_skew_seconds)
}

/// Oldest plausible unix banked (backlog depth), scanning only the aligned-from-7 grid (asymmetric with
/// the newest scan by design, to dodge a WHOOP-4 straddle word).
#[uniffi::export]
pub fn data_range_oldest(frame: Vec<u8>) -> Option<u32> {
    response::data_range_scan_oldest(&frame)
}

/// Pages the strap has banked but not yet sent, from a GET_DATA_RANGE frame. Diagnostic: it reports how
/// far a sync has to go and never gates one.
#[uniffi::export]
pub fn data_range_pages_behind(frame: Vec<u8>) -> Option<u32> {
    response::data_range_pages_behind(&frame)
}
