//! The firmware transfer's public surface: how far a run is armed, what it may send before the load,
//! how far it got, what it reports and how it refuses. No IO and no state machine — `super` drives it.

use std::time::Duration;

use whoop_protocol::event::ResultCode;
use whoop_protocol::firmware::Tail;
use whoop_protocol::firmware_image::{ImageFault, ImageHeader};
use whoop_protocol::variant::Variant;

/// The lowest battery percent a transfer may run at. A caller may raise this floor, never lower it.
pub const BATTERY_FLOOR_PCT: f64 = 80.0;

/// Defaults for the per-command reply window and the tolerated timeouts across a whole transfer.
const DEFAULT_STEP_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_RETRY_BUDGET: u32 = 6;

/// How far the operator has armed the run. Ordered: each level implies the ones below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlashArm {
    /// Evaluate every gate, send no firmware command.
    #[default]
    Plan,
    /// START + LOAD×N + VERIFY. Nothing is committed; the running image is untouched.
    Stage,
    /// Also send PROCESS. The commit point.
    Commit,
}

/// What is sent before START to quiet the strap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quiesce {
    None,
    /// The abort alone.
    #[default]
    AbortOnly,
    /// Realtime-HR off, IMU-stream off, then the abort. Each is awaited so one command is outstanding
    /// at a time, but only the abort's result is checked.
    AppIdentical,
}

/// Furthest point a run reached. Ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlashStep {
    Planned,
    Quiesced,
    Started,
    Loaded,
    Verified,
    Committed,
}

#[derive(Clone, Debug)]
pub struct FlashOptions {
    /// Serial the connected strap must report, matched as a case-insensitive suffix. Anything too
    /// short to name one band is refused, so there is no wildcard.
    pub expect_serial: String,
    /// The board the strap must be, on top of the image-versus-strap family check that always runs.
    /// `None` accepts any board the image is built for; the CLI pins it to the carve-out's board.
    pub expect_variant: Option<Variant>,
    pub arm: FlashArm,
    /// Floor in percent, clamped up to [`BATTERY_FLOOR_PCT`].
    pub min_battery_pct: f64,
    pub tail: Tail,
    pub quiesce: Quiesce,
    /// Timeouts tolerated across the whole transfer before it aborts.
    pub retry_budget: u32,
    /// Per-command reply window.
    pub step_timeout: Duration,
}

impl Default for FlashOptions {
    fn default() -> Self {
        FlashOptions {
            expect_serial: String::new(),
            expect_variant: None,
            arm: FlashArm::Plan,
            min_battery_pct: BATTERY_FLOOR_PCT,
            tail: Tail::Pad,
            quiesce: Quiesce::AbortOnly,
            retry_budget: DEFAULT_RETRY_BUDGET,
            step_timeout: DEFAULT_STEP_TIMEOUT,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FlashProgress {
    Planned { chunks: usize, bytes: usize },
    Chunk { index: usize, chunks: usize, offset: u32, retries: u32 },
    Step { step: FlashStep, result: Option<ResultCode>, status: Option<u8> },
    /// The last event of any armed run, abort included, so the strap's state is stated even when the
    /// call returns an error and the report with it.
    Ended { reached: FlashStep, chunks_sent: usize, chunks_planned: usize, retries: u32 },
}

#[derive(Clone, Debug)]
pub struct FlashReport {
    pub serial: String,
    pub header: ImageHeader,
    pub battery_pct: Option<f64>,
    /// Negotiated ATT MTU, when the link reports one.
    pub mtu: Option<usize>,
    pub chunks_planned: usize,
    pub chunks_sent: usize,
    pub retries: u32,
    /// Frames seen on the stream that answered no outstanding command.
    pub stray_frames: usize,
    pub reached: FlashStep,
    pub start: Option<ResultCode>,
    pub verify: Option<ResultCode>,
    pub process: Option<ResultCode>,
    pub last_status: Option<u8>,
    pub elapsed: Duration,
}

#[derive(thiserror::Error, Debug)]
pub enum FlashFault {
    #[error("image fails its own header check: {0:?}")]
    Image(ImageFault),
    #[error("expected serial {0:?} is too short to name one band — refusing to flash")]
    NoExpectedSerial(String),
    #[error("link carries {have}-byte writes, the transfer needs {need} — refusing to flash")]
    LinkTooNarrow { have: usize, need: usize },
    #[error("strap serial unreadable — refusing to flash")]
    SerialUnreadable,
    #[error("strap {found} is not the flashable band {expected}")]
    WrongStrap { found: String, expected: String },
    #[error("firmware transfer is 5.0/MG only")]
    WrongFamily,
    #[error("strap reports firmware {0}, which is not a 5.0/MG line")]
    WrongStrapFirmware(String),
    #[error("strap hardware {0:?} is not a board this image can be sent to — refusing to flash")]
    ImageNotForStrap(String),
    #[error("strap hardware {found:?} is {have:?}, the flashable band is {want:?} — refusing to flash")]
    WrongBoard { found: String, have: Variant, want: Variant },
    #[error("battery unreadable — refusing to flash")]
    BatteryUnreadable,
    #[error("battery {have:.0}% is below the {floor:.0}% floor")]
    LowBattery { have: f64, floor: f64 },
    #[error("{step:?} got no reply")]
    NoReply { step: FlashStep },
    #[error("{step:?} refused: {result:?} status {status:?}")]
    Refused { step: FlashStep, result: Option<ResultCode>, status: Option<u8> },
    #[error("chunk at offset {offset} failed: {result:?} status {status:?}")]
    ChunkRefused { offset: u32, result: Option<ResultCode>, status: Option<u8> },
    #[error("retry budget spent at offset {offset}")]
    RetriesExhausted { offset: u32 },
}
