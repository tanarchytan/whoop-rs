//! The uniffi record and enum types the apps see, plus the conversions to and from the codec
//! and algorithm crates. No logic: every item here is a shape or a mapping between two shapes.

use crate::*;

#[derive(uniffi::Enum, Clone, Copy)]
pub enum Gen {
    Gen4,
    Gen5,
}

impl From<Gen> for Family {
    fn from(g: Gen) -> Self {
        match g {
            Gen::Gen4 => Family::Gen4,
            Gen::Gen5 => Family::Gen5,
        }
    }
}

/// The logical notify channel a native notification arrived on (mapped from its characteristic UUID).
#[derive(uniffi::Enum, Clone, Copy)]
pub enum Chan {
    CmdNotify,
    Events,
    Data,
    Console,
}

impl From<Chan> for Channel {
    fn from(c: Chan) -> Self {
        match c {
            Chan::CmdNotify => Channel::CmdNotify,
            Chan::Events => Channel::Events,
            Chan::Data => Channel::Data,
            Chan::Console => Channel::Console,
        }
    }
}

/// A per-second history summary handed to the app (v18 on 5.0/MG, v5/v24 on 4.0).
#[derive(uniffi::Record, Clone)]
pub struct HistorySummary {
    pub version: u8,
    pub unix: u32,
    pub heart_rate: Option<u8>,
    pub rr_intervals: Vec<u16>,
    pub gravity: Option<Vec<f32>>, // [x, y, z]
    pub skin_temp_c: Option<f32>,
    pub skin_temp_raw: Option<u16>, // raw register; the consumer applies its family/device-specific °C scale
    pub spo2_red: Option<u16>, // 4.0 v24 raw red ADC
    pub spo2_ir: Option<u16>,  // 4.0 v24 raw IR ADC
    pub spo2_pct: Option<u8>,  // 5.0 v18 computed %, sleep-only
    pub resp_raw: Option<u16>, // 4.0 v24 raw respiration ADC
    pub steps: Option<u16>,
    pub activity_class: Option<u8>,
    pub sleep_state: Option<u8>,
    pub signal_flags: Option<u8>,   // 5.0 v18 PPG SIGPROC bitfield (bit 4 = off-wrist)
    pub signal_quality: Option<u8>, // 5.0 v18 PPG confidence, 255 = clean
    pub dynamic_acceleration_g: Option<f32>, // 5.0 v18 on-chip gravity-removed motion magnitude (g); the ENMO signal
    // 5.0 v18 optical front-end telemetry. Amplitudes are withheld when the band flags the second's
    // beat detection as untrustworthy, which `optical_signal_poor` reports instead.
    pub optical_baseline_a: Option<u8>, // 0 marks off-wrist
    pub optical_baseline_b: Option<u8>,
    pub optical_amp_a: Option<u8>,
    pub optical_amp_b: Option<u8>,
    pub optical_signal_poor: Option<bool>,
    pub record_index: Option<u32>, // 5.0 v18 dense per-second emission counter, independent of the clock
    pub temp_aux_1_raw: Option<u16>, // 5.0 v18 auxiliary thermal registers, deci-degrees C
    pub temp_aux_2_raw: Option<u16>,
    pub sleep_state_raw: Option<u8>, // 5.0 v18 whole packed byte; `sleep_state` is bits 4-5 of it
    // 5.0 v18 fields carried every second whose meaning is unpinned, named for their inner offset.
    pub raw_u8_28: Option<u8>,
    pub raw_u8_29: Option<u8>,
    pub raw_u16_30: Option<u16>,
    pub raw_f32_105: Option<f32>,
    pub raw_u16_26: Option<u16>,
    /// Every remaining per-second byte that carries information, packed. Opaque: store it whole.
    pub unpinned: Option<Vec<u8>>,
}

impl From<records::HistoryRecord> for HistorySummary {
    fn from(h: records::HistoryRecord) -> Self {
        HistorySummary {
            version: h.version,
            unix: h.unix,
            heart_rate: h.heart_rate,
            rr_intervals: h.rr_intervals,
            gravity: h.gravity.map(|g| g.to_vec()),
            skin_temp_c: h.skin_temp_c,
            skin_temp_raw: h.skin_temp_raw,
            spo2_red: h.spo2.map(|(r, _)| r),
            spo2_ir: h.spo2.map(|(_, i)| i),
            spo2_pct: h.spo2_pct,
            resp_raw: h.resp_raw,
            steps: h.steps,
            activity_class: h.activity_class,
            sleep_state: h.sleep_state,
            signal_flags: h.signal_flags,
            signal_quality: h.signal_quality,
            dynamic_acceleration_g: h.dynamic_acceleration_g,
            optical_baseline_a: h.optical_baseline_a,
            optical_baseline_b: h.optical_baseline_b,
            optical_amp_a: h.optical_amp_a,
            optical_amp_b: h.optical_amp_b,
            optical_signal_poor: h.optical_signal_poor,
            record_index: h.record_index,
            temp_aux_1_raw: h.temp_aux_1_raw,
            temp_aux_2_raw: h.temp_aux_2_raw,
            sleep_state_raw: h.sleep_state_raw,
            raw_u8_28: h.raw_u8_28,
            raw_u8_29: h.raw_u8_29,
            raw_u16_30: h.raw_u16_30,
            raw_f32_105: h.raw_f32_105,
            raw_u16_26: h.raw_u16_26,
            unpinned: h.unpinned,
        }
    }
}

/// One action the app must take while draining history.
#[derive(uniffi::Enum)]
pub enum Step {
    /// A decoded record — persist it.
    Record { summary: HistorySummary },
    /// A raw 24 Hz optical buffer (v26): a single AC-coupled waveform (no red/IR pair).
    Ppg { unix: u32, record_id: Option<u16>, samples: Vec<i16> },
    /// A raw 6-axis IMU buffer (v21): 100 Hz accel + gyro, interleaved x,y,z per sample (raw i16 LSB —
    /// scale accel by 1/4096 for g, gyro by 2000/32768 for deg/s).
    Imu { unix: u32, sample_rate_hz: u16, accel: Vec<i16>, gyro: Vec<i16> },
    /// A raw multi-wavelength optical buffer (v20): ~25 Hz × 6 channels, raw 20-bit signed ADC counts,
    /// flattened channel-major (25 samples of ch0, then ch1, …). Channels [0,1] green, [2,3] ambient,
    /// [4,5] red/IR (inferred). SpO2 % is not here — it rides the v18 summary.
    Optical { unix: u32, sample_rate_hz: u16, channels: u8, samples: Vec<i32> },
    /// Write these bytes (confirmed) to the command characteristic — the mandatory HISTORY_END ACK.
    Ack { frame: Vec<u8> },
    /// The drain finished.
    Complete,
}

pub(crate) fn to_step(s: OffloadStep) -> Step {
    match s {
        OffloadStep::Record(Record::History(h)) => Step::Record { summary: h.into() },
        OffloadStep::Record(Record::Ppg(p)) => Step::Ppg { unix: p.unix, record_id: p.record_id, samples: p.samples },
        OffloadStep::Record(Record::Imu(m)) => Step::Imu {
            unix: m.unix,
            sample_rate_hz: m.sample_rate_hz,
            accel: m.accel.into_iter().flatten().collect(),
            gyro: m.gyro.into_iter().flatten().collect(),
        },
        OffloadStep::Record(Record::Optical(o)) => Step::Optical {
            unix: o.unix,
            sample_rate_hz: o.sample_rate_hz,
            channels: o.channels.len() as u8,
            samples: o.channels.into_iter().flatten().collect(),
        },
        OffloadStep::Ack(frame) => Step::Ack { frame },
        OffloadStep::Complete => Step::Complete,
    }
}

/// A live-notify frame decoded off the realtime / event / console channels (separate from the offload path).
#[derive(uniffi::Enum)]
pub enum Live {
    Realtime { unix: u32, heart_rate: u8, rr_intervals: Vec<u16> },
    R22 { unix: u32, hr_ch1: u8, hr_ch2: Option<u8>, accel: Vec<f32> },
    /// An EVENT frame. `battery_*` are Some only for BATTERY_LEVEL (raw deci-% + mV + charging — the
    /// consumer divides the deci-% in f64). `payload_hex` is the Gen5 opaque residual (lowercase hex).
    Event {
        number: u8,
        unix: u32,
        battery_soc_deci: Option<u16>,
        battery_millivolts: Option<u16>,
        battery_charging: Option<bool>,
        payload_hex: Option<String>,
    },
    Console { text: String },
}

/// A decoded command response (identity, battery, clock, data range, firmware). Every variant carries
/// `resp_cmd` (the command byte replied to) and `result` (the status code, `None` on 4.0) so the live
/// COMMAND_RESPONSE handshake can gate on them.
#[derive(uniffi::Enum)]
pub enum Response {
    Battery { resp_cmd: u8, result: Option<u8>, percent: f64 },
    Clock { resp_cmd: u8, result: Option<u8>, unix: u32 },
    Hello { resp_cmd: u8, result: Option<u8>, device_name: String, fw_version: Option<Vec<u8>> },
    DataRange { resp_cmd: u8, result: Option<u8>, oldest: u32, newest: u32 },
    Version { resp_cmd: u8, result: Option<u8>, fw: Vec<u32> },
    ExtendedBattery { resp_cmd: u8, result: Option<u8>, millivolts: u16, remaining_mah: u16, current_ma: i16 },
    BatteryPack { resp_cmd: u8, result: Option<u8>, serial: String, soc_pct: f64, bt_addr: String },
    /// The strap answered GET_BATTERY_PACK_INFO with no pack attached — the app clears its pack row on
    /// this, where a missing reply must leave it alone.
    NoBatteryPack { resp_cmd: u8, result: Option<u8> },
    Other { resp_cmd: u8, result: Option<u8> },
}

/// METADATA offload-state fields: `meta_type` (1 start / 2 end / 3 complete) drives the drain, and for a
/// HISTORY_END `unix` + `trim_cursor` are the ack/advance cursor. `crc_ok` lets the app gate on a
/// checksum-valid frame (a forged HISTORY_END must not advance the trim over unstored data).
#[derive(uniffi::Record)]
pub struct MetadataInfo {
    pub meta_type: u8,
    pub unix: u32,
    pub trim_cursor: u32,
    pub crc_ok: bool,
}

/// A single-frame v26 24 Hz PPG waveform (24 i16 samples, single wavelength).
#[derive(uniffi::Record)]
pub struct PpgFrame {
    pub unix: u32,
    pub record_id: Option<u16>,
    pub samples: Vec<i16>,
}

/// A single-frame v21 100 Hz 6-axis IMU buffer: accel + gyro interleaved x,y,z per sample (raw i16 LSB —
/// scale accel by 1/4096 for g, gyro by 2000/32768 for deg/s).
#[derive(uniffi::Record)]
pub struct ImuFrame {
    pub unix: u32,
    pub sample_rate_hz: u16,
    pub accel: Vec<i16>,
    pub gyro: Vec<i16>,
}

/// One raw PPG sample for `ppg_hr` (wall-clock second + raw ADC value).
#[derive(uniffi::Record)]
pub struct PpgSample {
    pub ts: i64,
    pub value: i32,
}

/// A derived HR estimate from `ppg_hr`.
#[derive(uniffi::Record)]
pub struct PpgEstimate {
    pub ts: i64,
    pub bpm: i32,
    pub conf: f64,
}

/// One record's R-R for gap-aware RMSSD (the app builds these from a `HistorySummary`'s unix + rr_intervals).
#[derive(uniffi::Record)]
pub struct RrRun {
    pub unix: u32,
    pub rr: Vec<u16>,
}

/// One record's R-R plus the strap's `opticalSignalPoor` for that record. What [`crate::hrv_nightly`]
/// takes, so the trust filter travels with the beats instead of being a step a caller can omit.
#[derive(uniffi::Record)]
pub struct RrReport {
    pub unix: u32,
    pub rr: Vec<u16>,
    pub optical_signal_poor: Option<bool>,
}

#[derive(uniffi::Enum)]
pub enum ReadinessTier {
    Primed,
    Normal,
    Suppressed,
}

impl From<physio_algo::ReadinessTier> for ReadinessTier {
    fn from(t: physio_algo::ReadinessTier) -> Self {
        match t {
            physio_algo::ReadinessTier::Primed => ReadinessTier::Primed,
            physio_algo::ReadinessTier::Normal => ReadinessTier::Normal,
            physio_algo::ReadinessTier::Suppressed => ReadinessTier::Suppressed,
        }
    }
}

/// An HRV-readiness reading (log-domain baseline vs the personal normal band).
#[derive(uniffi::Record)]
pub struct HrvReadinessInfo {
    pub tier: ReadinessTier,
    pub baseline7_ms: f64,
    pub normal_low_ms: f64,
    pub normal_high_ms: f64,
    pub overreaching_watch: bool,
}

/// A staged sleep stage. String forms are `"wake" | "light" | "deep" | "rem"` for cross-platform parity.
#[derive(uniffi::Enum, Clone, Copy)]
pub enum SleepStage {
    Wake,
    Light,
    Deep,
    Rem,
}

impl From<sleep::SleepStage> for SleepStage {
    fn from(s: sleep::SleepStage) -> Self {
        match s {
            sleep::SleepStage::Wake => SleepStage::Wake,
            sleep::SleepStage::Light => SleepStage::Light,
            sleep::SleepStage::Deep => SleepStage::Deep,
            sleep::SleepStage::Rem => SleepStage::Rem,
        }
    }
}

/// A contiguous run of one stage over `[start, end]` wall-clock unix seconds.
#[derive(uniffi::Record, Clone)]
pub struct SleepSegment {
    pub start: i64,
    pub end: i64,
    pub stage: SleepStage,
}

/// One HR sample for the stager (`bpm` at unix second `ts`).
#[derive(uniffi::Record, Clone)]
pub struct SleepHrSample {
    pub ts: i64,
    pub bpm: u16,
}

/// A run of consecutive R-R intervals (ms) sharing one unix-second anchor `ts`.
#[derive(uniffi::Record, Clone)]
pub struct SleepRrRun {
    pub ts: i64,
    pub intervals: Vec<u16>,
}

/// One 3-axis accel / gravity sample (g) at unix second `ts`.
#[derive(uniffi::Record, Clone)]
pub struct SleepAccelSample {
    pub ts: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// The protocol-free input bundle for one detected in-bed span (`[start, end]` unix seconds).
#[derive(uniffi::Record, Clone)]
pub struct SleepInput {
    pub start: i64,
    pub end: i64,
    pub hr: Vec<SleepHrSample>,
    pub rr: Vec<SleepRrRun>,
    pub accel: Vec<SleepAccelSample>,
}

impl From<SleepInput> for sleep::SleepInput {
    fn from(i: SleepInput) -> Self {
        sleep::SleepInput {
            start: i.start,
            end: i.end,
            hr: i.hr.into_iter().map(|h| sleep::HrSample { ts: h.ts, bpm: h.bpm }).collect(),
            rr: i.rr.into_iter().map(|r| sleep::RrRun { ts: r.ts, intervals: r.intervals }).collect(),
            accel: i.accel.into_iter().map(|a| sleep::AccelSample { ts: a.ts, x: a.x, y: a.y, z: a.z }).collect(),
        }
    }
}

pub(crate) fn to_sleep_segments(segs: Vec<sleep::StageSegment>) -> Vec<SleepSegment> {
    segs.into_iter()
        .map(|s| SleepSegment { start: s.start, end: s.end, stage: s.stage.into() })
        .collect()
}

impl From<SleepStage> for sleep::SleepStage {
    fn from(s: SleepStage) -> Self {
        match s {
            SleepStage::Wake => sleep::SleepStage::Wake,
            SleepStage::Light => sleep::SleepStage::Light,
            SleepStage::Deep => sleep::SleepStage::Deep,
            SleepStage::Rem => sleep::SleepStage::Rem,
        }
    }
}

pub(crate) fn to_stage_segments(segs: Vec<SleepSegment>) -> Vec<sleep::StageSegment> {
    segs.into_iter()
        .map(|s| sleep::StageSegment { start: s.start, end: s.end, stage: s.stage.into() })
        .collect()
}

/// One step-counter sample: wrap-aware u16 `counter` + optional activity class (1=walk, 2=run) at `ts`.
#[derive(uniffi::Record, Clone)]
pub struct SleepStepSample {
    pub ts: i64,
    pub counter: u16,
    pub activity_class: Option<u8>,
}

/// An off-wrist `[start, end)` interval (unix seconds).
#[derive(uniffi::Record, Clone)]
pub struct WristOffInterval {
    pub start: i64,
    pub end: i64,
}

/// One band sleep-state sample: the strap's own `state` (0 wake/1 still/2 asleep/3 up) at `ts`.
#[derive(uniffi::Record, Clone)]
pub struct BandStateSample {
    pub ts: i64,
    pub state: i32,
}

/// The full night-window stream bundle for detection + staging (raw signals + off-wrist / band-state /
/// tz-offset). `analyze_sleep_with` carves the in-bed spans and stages each behind this one border call.
#[derive(uniffi::Record, Clone)]
pub struct SleepStreams {
    pub hr: Vec<SleepHrSample>,
    pub rr: Vec<SleepRrRun>,
    pub accel: Vec<SleepAccelSample>,
    pub steps: Vec<SleepStepSample>,
    pub tz_offset_s: i64,
    pub wrist_off: Vec<WristOffInterval>,
    pub band_sleep_state: Vec<BandStateSample>,
}

impl From<SleepStreams> for sleep::SleepStreams {
    fn from(s: SleepStreams) -> Self {
        sleep::SleepStreams {
            hr: s.hr.into_iter().map(|h| sleep::HrSample { ts: h.ts, bpm: h.bpm }).collect(),
            rr: s.rr.into_iter().map(|r| sleep::RrRun { ts: r.ts, intervals: r.intervals }).collect(),
            accel: s.accel.into_iter().map(|a| sleep::AccelSample { ts: a.ts, x: a.x, y: a.y, z: a.z }).collect(),
            steps: s
                .steps
                .into_iter()
                .map(|st| sleep::StepSample { ts: st.ts, counter: st.counter, activity_class: st.activity_class })
                .collect(),
            tz_offset_s: s.tz_offset_s,
            wrist_off: s.wrist_off.into_iter().map(|w| (w.start, w.end)).collect(),
            band_sleep_state: s.band_sleep_state.into_iter().map(|b| (b.ts, b.state)).collect(),
        }
    }
}

/// One detected sleep session: span, motion-refined hypnogram, derived scalars, and the per-30 s-epoch
/// motion + band sleep-state grids the caller persists.
#[derive(uniffi::Record, Clone)]
pub struct SleepSession {
    pub start: i64,
    pub end: i64,
    pub efficiency: f64,
    pub resting_hr: Option<i32>,
    pub avg_hrv: Option<f64>,
    pub segments: Vec<SleepSegment>,
    pub motion_grid: Vec<f64>,
    pub sleep_state_grid: Vec<i32>,
    /// Spans the engine declines to score, ascending; empty for the original engine.
    pub unscored: Vec<SleepSpan>,
}

/// A `[start, end)` span in unix seconds.
#[derive(uniffi::Record, Clone, Debug, PartialEq)]
pub struct SleepSpan {
    pub start: i64,
    pub end: i64,
}

pub(crate) fn to_sleep_spans(spans: Vec<(i64, i64)>) -> Vec<SleepSpan> {
    spans.into_iter().map(|(start, end)| SleepSpan { start, end }).collect()
}

/// Which staging engine to run: the shipped recipe, or the experimental one (abstention holes).
#[derive(uniffi::Enum, Clone, Copy, Debug, PartialEq)]
pub enum SleepEngine {
    Original,
    Experimental,
}

impl From<SleepEngine> for sleep::Engine {
    fn from(e: SleepEngine) -> Self {
        match e {
            SleepEngine::Original => sleep::Engine::V2,
            SleepEngine::Experimental => sleep::Engine::Tanv1,
        }
    }
}

/// A re-staged span: motion-refined hypnogram plus the spans the engine declines to score.
#[derive(uniffi::Record, Clone)]
pub struct RefinedStaging {
    pub segments: Vec<SleepSegment>,
    pub unscored: Vec<SleepSpan>,
}

impl From<sleep::Session> for SleepSession {
    fn from(s: sleep::Session) -> Self {
        SleepSession {
            start: s.start,
            end: s.end,
            efficiency: s.efficiency,
            resting_hr: s.resting_hr,
            avg_hrv: s.avg_hrv,
            segments: to_sleep_segments(s.segments),
            motion_grid: s.motion_grid,
            sleep_state_grid: s.sleep_state_grid,
            unscored: to_sleep_spans(s.unscored),
        }
    }
}

/// One candidate block for main-night selection: `[start, end]` unix seconds.
#[derive(uniffi::Record, Clone)]
pub struct MainNightBlock {
    pub start: i64,
    pub end: i64,
}

/// One candidate scored on what its stages DECODED: effective onset plus asleep and in-bed seconds.
#[derive(uniffi::Record, Clone)]
pub struct MainNightScoredBlock {
    pub onset: i64,
    pub asleep_s: f64,
    pub in_bed_s: f64,
}

/// Why the main-night selector chose the block it chose (for UI explainability).
#[derive(uniffi::Enum, Clone, Copy)]
pub enum MainNightReason {
    OnlyBlock,
    Longest,
    LongestNearUsual,
    AlignedToUsual,
}

impl From<sleep::MainNightReason> for MainNightReason {
    fn from(r: sleep::MainNightReason) -> Self {
        match r {
            sleep::MainNightReason::OnlyBlock => MainNightReason::OnlyBlock,
            sleep::MainNightReason::Longest => MainNightReason::Longest,
            sleep::MainNightReason::LongestNearUsual => MainNightReason::LongestNearUsual,
            sleep::MainNightReason::AlignedToUsual => MainNightReason::AlignedToUsual,
        }
    }
}

/// The resolved main-night pick plus why it won and the chosen block's asleep span (seconds).
#[derive(uniffi::Record, Clone)]
pub struct MainNightSel {
    pub index: u32,
    pub reason: MainNightReason,
    pub asleep_sec: i64,
}

/// One trailing-history block for learning habitual timing (`day_key` groups by local calendar day).
#[derive(uniffi::Record, Clone)]
pub struct SleepHistoryBlock {
    pub start: i64,
    pub end: i64,
    pub day_key: String,
}

/// One inter-fragment wake seam `[start, end)` inside a bridged night group.
#[derive(uniffi::Record, Clone)]
pub struct SleepGap {
    pub start: i64,
    pub end: i64,
}

/// One bridged night group: the composing original indices (ascending) + its inter-fragment wake seams.
#[derive(uniffi::Record, Clone)]
pub struct SleepBridgedGroup {
    pub indices: Vec<u32>,
    pub gaps: Vec<SleepGap>,
}
