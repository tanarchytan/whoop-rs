//! uniffi surface over the pure WHOOP codec. The mobile apps do BLE natively (Android BluetoothGatt,
//! iOS CoreBluetooth) — including the universal easy-connect (attach to an already-connected band) — and
//! feed notification bytes here; this returns decoded records and the exact frames to write back. No
//! async, no radio crosses the FFI: the sans-IO `Offload` is what makes one Rust core serve both apps.

uniffi::setup_scaffolding!();

pub(crate) use std::sync::{Arc, Mutex};

pub(crate) use physio_algo::{
    baselines, biological_age, calories, circadian, hr_recovery, hr_zones, hydration, imu_features, nap,
    ppg as ppg_hr,
    recovery, recovery_drivers, respiratory_rate, rest,
    resting_hr,
    sleep, sleep_debt, sleep_regularity, steps, strain, stress, stress_onset, vital_bands, vitality, vo2max,
    workout,
    HrvReadiness, Spo2,
};
pub(crate) use whoop_protocol::deframe::DeframerMap;
pub(crate) use whoop_protocol::hello::GEN5_CLIENT_HELLO;
pub(crate) use whoop_protocol::{
    advertising, alarm, clock, command, config, console, framing, haptic, live, records, response, Channel,
    Family, Offload, OffloadStep, PacketType, Record, Variant,
};

mod types;
pub use types::*;
mod codec;
pub use codec::*;
mod sleep_api;
pub use sleep_api::*;
mod hrv_api;
pub use hrv_api::*;
mod spo2_api;
pub use spo2_api::*;
mod scores_api;
pub use scores_api::*;
mod activity_api;
pub use activity_api::*;
mod baselines_api;
pub use baselines_api::*;
mod wellness_api;
pub use wellness_api::*;
mod cfg_api;
pub use cfg_api::*;
mod illness_api;
pub use illness_api::*;
mod bands_api;
pub use bands_api::*;
mod device_api;
pub use device_api::*;
mod pack_api;
pub use pack_api::*;

#[cfg(test)]
mod tests {
    use super::{Chan, Gen, Response, Step, WhoopCodec};
    use whoop_protocol::hello::GEN5_CLIENT_HELLO;
    use whoop_protocol::{command, framing, Family};

    fn v18_frame() -> Vec<u8> {
        let mut p = vec![0u8; 40];
        p[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        p[11] = 96; // hr @ inner 14
        framing::encode(Family::Gen5, 47, 18, 0, &p)
    }

    fn v21_imu_frame() -> Vec<u8> {
        let mut p = vec![0u8; 1229];
        p[13..15].copy_from_slice(&100u16.to_le_bytes()); // count_a @ inner 16
        p[17..19].copy_from_slice(&4096i16.to_le_bytes()); // ax[0]
        p[619..621].copy_from_slice(&100u16.to_le_bytes()); // count_b @ inner 622
        framing::encode(Family::Gen5, 47, 21, 0, &p)
    }

    #[test]
    fn feed_v18_yields_a_record_step() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let steps = codec.feed(Chan::Data, v18_frame());
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Record { summary } => {
                assert_eq!(summary.heart_rate, Some(96));
                assert_eq!(summary.unix, 1_784_000_000);
            }
            _ => panic!("expected Record"),
        }
    }

    #[test]
    fn feed_v21_yields_an_imu_step() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let steps = codec.feed(Chan::Data, v21_imu_frame());
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            Step::Imu { sample_rate_hz, accel, gyro, .. } => {
                assert_eq!(*sample_rate_hz, 100);
                assert_eq!(accel.len(), 300);
                assert_eq!(gyro.len(), 300);
                assert_eq!(accel[0], 4096);
            }
            _ => panic!("expected Imu"),
        }
    }

    #[test]
    fn decode_history_roundtrips_v18() {
        let codec = WhoopCodec::new(Gen::Gen5);
        assert_eq!(codec.decode_history(v18_frame()).unwrap().heart_rate, Some(96));
    }

    #[test]
    fn r22_frames_are_sixteen_set_configs() {
        let frames = WhoopCodec::new(Gen::Gen5).r22_frames();
        assert_eq!(frames.len(), 16);
        for f in &frames {
            assert_eq!(framing::decode(Family::Gen5, f).unwrap().cmd(), command::SET_CONFIG);
        }
    }

    #[test]
    fn client_hello_is_the_bond_frame() {
        assert_eq!(WhoopCodec::new(Gen::Gen5).client_hello(), GEN5_CLIENT_HELLO.to_vec());
    }

    #[test]
    fn reset_drops_a_buffered_partial_frame() {
        let codec = WhoopCodec::new(Gen::Gen5);
        let full = v18_frame();
        let _ = codec.feed(Chan::Data, full[..5].to_vec()); // partial
        codec.reset();
        assert_eq!(codec.feed(Chan::Data, full).len(), 1); // clean frame still decodes
    }

    #[test]
    fn decode_history_surfaces_the_wide_fields() {
        let mut p = vec![0u8; 90];
        p[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7
        p[11] = 96; // hr @ inner 14
        p[46..48].copy_from_slice(&123u16.to_le_bytes()); // steps @ inner 49
        p[52] = 2; // activity_class @ inner 55
        p[62..64].copy_from_slice(&3345u16.to_le_bytes()); // skin temp @ inner 65 (33.45 °C)
        p[71] = 97; // spo2_pct @ inner 74
        let s = WhoopCodec::new(Gen::Gen5)
            .decode_history(framing::encode(Family::Gen5, 47, 18, 0, &p))
            .unwrap();
        assert_eq!(s.heart_rate, Some(96));
        assert_eq!(s.steps, Some(123));
        assert_eq!(s.activity_class, Some(2));
        assert_eq!(s.skin_temp_raw, Some(3345));
        assert_eq!(s.spo2_pct, Some(97));
    }

    #[test]
    fn decode_live_folds_battery_into_event() {
        use super::Live;
        // Battery EVENT: payload = inner[3..]; inner 4/13/17/22 = payload 1/10/14/19.
        let mut p = vec![0u8; 24];
        p[1..5].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        p[10..12].copy_from_slice(&812u16.to_le_bytes()); // 81.2%
        p[14..16].copy_from_slice(&4100u16.to_le_bytes());
        p[19] = 1;
        let wire = framing::encode(Family::Gen5, 48, 0, 3, &p); // cmd 3 = BATTERY_LEVEL
        match WhoopCodec::new(Gen::Gen5).decode_live(wire).unwrap() {
            Live::Event { number, unix, battery_soc_deci, battery_millivolts, battery_charging, payload_hex } => {
                assert_eq!(number, 3);
                assert_eq!(unix, 1_784_000_000);
                assert_eq!(battery_soc_deci, Some(812));
                assert_eq!(battery_millivolts, Some(4100));
                assert_eq!(battery_charging, Some(true));
                assert!(payload_hex.is_some());
            }
            _ => panic!("expected Live::Event"),
        }
    }

    #[test]
    fn command_builders_carry_the_right_opcodes() {
        let c = WhoopCodec::new(Gen::Gen5);
        let cases: [(Vec<u8>, u8); 4] = [
            (c.get_battery_frame(0), command::GET_BATTERY_LEVEL),
            (c.get_data_range_frame(0), command::GET_DATA_RANGE),
            (c.reboot_frame(0), command::REBOOT_STRAP),
            (c.set_config_frame(0, "enable_r22_packets".into(), b'2'), command::SET_CONFIG),
        ];
        for (frame, op) in cases {
            assert_eq!(framing::decode(Family::Gen5, &frame).unwrap().cmd(), op);
        }
    }

    #[test]
    fn metrics_free_fns_compute() {
        let runs = vec![super::RrRun { unix: 0, rr: vec![600, 610, 605] }];
        assert!(super::hrv_rmssd_gap_aware(runs).unwrap() < 20.0);
        assert!(super::hrv_readiness(vec![Some(50.0); 5]).is_some());
        assert!(super::ppg_hr(vec![]).is_empty());
        let win = vec![super::RrRun { unix: 100, rr: vec![800, 810, 820, 815, 805] }];
        assert!((super::hrv_windowed_avg(100, 400, win).unwrap() - 9.013878188659973).abs() < 1e-12);
    }

    #[test]
    fn nightly_spo2_raw_means_truncates_in_span() {
        let spans = vec![super::Spo2Span { start: 1000, end: 2000 }];
        let samples = (0..20)
            .map(|i| super::Spo2RawSample {
                ts: 1000 + i,
                red: if i % 2 == 0 { 29000 } else { 31000 },
                ir: if i % 2 == 0 { 19000 } else { 21000 },
            })
            .collect();
        let m = super::nightly_spo2_raw_means(spans, samples).unwrap();
        assert_eq!((m.red, m.ir), (30000, 20000));
    }

    #[test]
    fn decode_metadata_reads_a_real_history_end() {
        let raw = whoop_protocol::bytes::from_hex(
            "aa011c00010023d1319102b949596a705d3b000000fdba010010000000000000f269faec",
        )
        .unwrap();
        let m = WhoopCodec::new(Gen::Gen5).decode_metadata(raw).unwrap();
        assert_eq!(m.meta_type, 2);
        assert_eq!(m.unix, 1_784_236_473);
        assert_eq!(m.trim_cursor, 113_405);
        assert!(m.crc_ok);
    }

    #[test]
    fn decode_metadata_rejects_a_non_metadata_frame() {
        assert!(WhoopCodec::new(Gen::Gen5).decode_metadata(v18_frame()).is_none());
    }

    #[test]
    fn decode_ppg_frame_reads_a_real_v26() {
        let raw = whoop_protocol::bytes::from_hex(
            "aa015000010035412f1a804b047b019452596aae0701004b8503006bfdcffd36fe50fe12ff73ff6dff42ffa7ffc9fff9ffe5ff5c005a007a00f20089003000dbfd2efd0bfe3ffeaefe3affc06c213c50070001001ddc65fe",
        )
        .unwrap();
        let p = WhoopCodec::new(Gen::Gen5).decode_ppg_frame(raw).unwrap();
        assert_eq!(p.record_id, Some(1099));
        assert_eq!(p.unix, 1_784_238_740);
        assert_eq!(p.samples.len(), 24);
        assert_eq!(p.samples[0], -661);
    }

    #[test]
    fn decode_history_rejects_a_bad_crc_frame() {
        // A real worn v18 frame from the protocol fixture; flipping one CRC byte must drop it (None), so a
        // corrupt/forged record can never be stored past the trim.
        let mut raw = whoop_protocol::bytes::from_hex(
            "aa01740001003fb12f1280733d8401b69f266a66460066025a0265020000000000007b0a8d656463ff0012163cf6a439bf2924fd3ed763fe3e3200aa000000000000000000f7000901f10b0007010c020c00000000000000000000000000000000000000000000000100656f1e1e0000009d61a7c00000003e862817",
        )
        .unwrap();
        let codec = WhoopCodec::new(Gen::Gen5);
        assert!(codec.decode_history(raw.clone()).is_some()); // good CRC decodes
        *raw.last_mut().unwrap() ^= 0xFF; // trash the CRC32 tail
        assert!(codec.decode_history(raw).is_none());
    }

    #[test]
    fn decode_ppg_frame_rejects_a_non_v26_frame() {
        // A v18 history frame carries no PPG waveform: the version gate returns None, not 24 samples.
        assert!(WhoopCodec::new(Gen::Gen5).decode_ppg_frame(v18_frame()).is_none());
    }

    #[test]
    fn decode_imu_frame_reads_a_synthetic_v21() {
        let f = WhoopCodec::new(Gen::Gen5).decode_imu_frame(v21_imu_frame()).unwrap();
        assert_eq!(f.sample_rate_hz, 100);
        assert_eq!(f.accel.len(), 300);
        assert_eq!(f.gyro.len(), 300);
        assert_eq!(f.accel[0], 4096);
    }

    #[test]
    fn data_range_scan_free_fns_pin_a_real_frame() {
        let frame = whoop_protocol::bytes::from_hex(
            "aa014c00010032d124f22204010140bb0100f9ba010001bb0100f9ba010010000000000002006a00000088ff1d001432b869cc4c00004549596ab83e00004549596ab83e0000ae49596aeb1100000000d0da9256",
        )
        .unwrap();
        assert_eq!(super::data_range_newest(frame.clone(), 1_784_236_480, 3600), Some(1_784_236_462));
        assert_eq!(super::data_range_oldest(frame), Some(1_778_385_408));
    }

    #[test]
    fn generic_command_frame_builds_and_refuses_destructive() {
        let c = WhoopCodec::new(Gen::Gen5);
        let f = c.command_frame(3, command::GET_ALARM_TIME, vec![0x00]).unwrap();
        assert_eq!(framing::decode(Family::Gen5, &f).unwrap().cmd(), command::GET_ALARM_TIME);
        // The genuinely-destructive set can't be built even through the generic door.
        assert!(c.command_frame(0, command::FORCE_TRIM, vec![]).is_none());
        assert!(c.command_frame(0, command::ENTER_BLE_DFU, vec![]).is_none());
    }

    #[test]
    fn gen4_encoder_frames_carry_the_right_opcodes_and_bodies() {
        let c = WhoopCodec::new(Gen::Gen4);
        let checks: [(Vec<u8>, u8); 5] = [
            (c.set_clock_frame(0, 1_784_000_000), command::SET_CLOCK),
            (c.set_clock_legacy_frame(0, 1_784_000_000), command::SET_CLOCK),
            (c.alarm_set_frame_gen4(0, 1_784_000_000), command::SET_ALARM_TIME),
            (c.run_haptics_frame(0, 2, 3), command::RUN_HAPTICS_PATTERN),
            (c.advertising_name_frame(0, "noop".into()), command::SET_ADVERTISING_NAME),
        ];
        for (frame, op) in checks {
            assert_eq!(framing::decode(Family::Gen4, &frame).unwrap().cmd(), op);
        }
        // The 8-byte vs legacy 9-byte set-clock bodies differ in length only.
        assert_eq!(framing::decode(Family::Gen4, &c.set_clock_frame(0, 1)).unwrap().payload().len(), 8);
        assert_eq!(framing::decode(Family::Gen4, &c.set_clock_legacy_frame(0, 1)).unwrap().payload().len(), 9);
    }

    #[test]
    fn decode_response_surfaces_cmd_and_result_on_every_variant() {
        let codec = WhoopCodec::new(Gen::Gen5);
        // A GET_DATA_RANGE reply: the success gate needs both the command byte and the status.
        let mut p = vec![0u8; 12];
        p[1] = 1; // result = SUCCESS
        p[3..7].copy_from_slice(&1_784_000_000u32.to_le_bytes());
        p[7..11].copy_from_slice(&1_784_200_000u32.to_le_bytes());
        let wire = framing::encode(Family::Gen5, 36, 0, command::GET_DATA_RANGE, &p);
        match codec.decode_response(wire).unwrap() {
            Response::DataRange { resp_cmd, result, oldest, newest } => {
                assert_eq!(resp_cmd, command::GET_DATA_RANGE);
                assert_eq!(result, Some(1));
                assert_eq!((oldest, newest), (1_784_000_000, 1_784_200_000));
            }
            _ => panic!("expected DataRange"),
        }
        // An unmodelled command still surfaces its cmd + status through Other.
        let mut p = vec![0u8; 8];
        p[1] = 2; // result = PENDING
        let wire = framing::encode(Family::Gen5, 36, 0, command::REBOOT_STRAP, &p);
        match codec.decode_response(wire).unwrap() {
            Response::Other { resp_cmd, result } => {
                assert_eq!(resp_cmd, command::REBOOT_STRAP);
                assert_eq!(result, Some(2));
            }
            _ => panic!("expected Other"),
        }
    }

    #[test]
    fn stage_sleep_refined_tiles_a_synthetic_night() {
        use super::{stage_sleep_refined_with, SleepAccelSample, SleepEngine, SleepHrSample, SleepInput, SleepRrRun, SleepStage};
        let start = 1_749_517_200i64;
        let dur = 90 * 60 * 4;
        let (mut hr, mut rr, mut accel) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..dur {
            let ts = start + i;
            let ph = (i / (90 * 60)) as usize;
            accel.push(SleepAccelSample { ts, x: 0.0, y: 0.0, z: 1.0 });
            let bpm: i64 = [50, 55, 58, 68][ph];
            hr.push(SleepHrSample { ts, bpm: bpm as u16 });
            rr.push(SleepRrRun { ts, intervals: vec![(60_000 / bpm) as u16] });
        }
        let input = SleepInput { start, end: start + dur, hr, rr, accel };
        let segs = stage_sleep_refined_with(input, Vec::new(), SleepEngine::Original).segments;
        assert!(!segs.is_empty());
        assert_eq!(segs.first().unwrap().start, start);
        assert_eq!(segs.last().unwrap().end, start + dur);
        // Contiguous tiling, each a valid stage.
        for w in segs.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(matches!(segs[0].stage, SleepStage::Wake | SleepStage::Light | SleepStage::Deep | SleepStage::Rem));
    }

    #[test]
    fn gen5_advertising_name_frame_has_the_puffin_body() {
        let c = WhoopCodec::new(Gen::Gen5);
        let frame = c.advertising_name_frame_gen5(0, "band".into());
        let f = framing::decode(Family::Gen5, &frame).unwrap();
        assert_eq!(f.cmd(), command::SET_ADVERTISING_NAME);
        // b3 selector 0x01, reserved 0x00, name, NUL terminator (GEN5 pads the inner to a 4-byte boundary).
        assert_eq!(&f.payload()[..7], b"\x01\x00band\x00");
    }
}
