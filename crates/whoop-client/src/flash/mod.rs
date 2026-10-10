//! The firmware transfer driver: every gate first, then START → LOAD×N → VERIFY → PROCESS over the
//! generic transport. Frames come from `whoop_protocol::firmware`, so the opcodes stay refused on the
//! blind path; the window is one command and nothing is committed before VERIFY answers success.

use std::time::Duration;

use futures::stream::BoxStream;
use tokio::time::Instant;

use ble_core::{BleTransport, Notification};
use whoop_protocol::event::ResultCode;
use whoop_protocol::firmware::{self, CHUNK_AMBIQ};
use whoop_protocol::firmware_image::{self, ImageFault};
use whoop_protocol::response::{self, resp_status, CommandResponse};
use whoop_protocol::variant::Variant;
use whoop_protocol::{command, framing, Family, Frame};

use crate::client::WhoopClient;
use crate::error::Error;

mod types;
pub use types::{
    FlashArm, FlashFault, FlashOptions, FlashProgress, FlashReport, FlashStep, Quiesce, BATTERY_FLOOR_PCT,
};

/// Ceiling on the retry budget a caller may ask for.
const MAX_RETRY_BUDGET: u32 = 16;

/// Firmware line this transfer targets, read off the strap's identity char.
const GEN5_FW_PREFIX: &str = "50.";

/// ATT bytes a confirmed write spends on top of its payload.
const ATT_WRITE_OVERHEAD: usize = 3;

/// Shortest expected serial that can still name one band. A suffix match on fewer characters would
/// hold for straps this verb must never touch.
const MIN_EXPECTED_SERIAL: usize = 8;

/// Off bodies for the two quiesce toggles. They differ: the HR stream takes a bare state byte, the IMU
/// stream takes the revision marker first.
const HR_STREAM_OFF: [u8; 1] = [0x00];
const IMU_STREAM_OFF: [u8; 2] = [0x01, 0x00];


/// What a single command answered: its result code and, where the handler carries one, its status byte.
type Answer = (Option<ResultCode>, Option<u8>);

/// Gate, then transfer. Every gate runs before the first write, in every arm level; `FlashArm::Plan`
/// returns straight after them.
pub(crate) async fn run<T: BleTransport>(
    client: &mut WhoopClient<T>,
    image: &[u8],
    opts: &FlashOptions,
    mut on_progress: impl FnMut(FlashProgress),
) -> Result<FlashReport, Error> {
    let started = Instant::now();
    // Free checks first, so a wrong file is named before the strap is touched at all.
    let header = firmware_image::inspect(image).map_err(FlashFault::Image)?;
    if client.family() != Family::Gen5 {
        return Err(FlashFault::WrongFamily.into());
    }
    let mtu = check_link(client)?;
    let serial = check_strap(client, opts, header.product).await?;
    let battery = check_battery(client, opts).await?;

    // The same length the chunker reads, so the seq block reserved below always matches the frames built.
    let total = firmware_image::transfer_len(image).ok_or(FlashFault::Image(ImageFault::LengthMismatch))?;
    let mut report = FlashReport {
        serial,
        header,
        battery_pct: Some(battery),
        mtu,
        chunks_planned: total.div_ceil(CHUNK_AMBIQ),
        chunks_sent: 0,
        retries: 0,
        stray_frames: 0,
        reached: FlashStep::Planned,
        start: None,
        verify: None,
        process: None,
        last_status: None,
        elapsed: Duration::ZERO,
    };
    on_progress(FlashProgress::Planned { chunks: report.chunks_planned, bytes: total });
    if opts.arm == FlashArm::Plan {
        report.elapsed = started.elapsed();
        return Ok(report);
    }

    let outcome = transfer(client, image, opts, &mut report, &mut on_progress).await;
    report.elapsed = started.elapsed();
    on_progress(FlashProgress::Ended {
        reached: report.reached,
        chunks_sent: report.chunks_sent,
        chunks_planned: report.chunks_planned,
        retries: report.retries,
    });
    outcome.map(|()| report)
}

/// The write half, once every gate has passed. An abort here leaves the running image untouched and
/// PROCESS unsent; what it may leave behind is a partly written staging area, which the next run
/// overwrites from offset zero.
async fn transfer<T: BleTransport>(
    client: &mut WhoopClient<T>,
    image: &[u8],
    opts: &FlashOptions,
    report: &mut FlashReport,
    on_progress: &mut impl FnMut(FlashProgress),
) -> Result<(), Error> {
    let mut notes = client.open_notifications().await?;
    quiesce(client, &mut notes, opts, report, on_progress).await?;

    let seq = client.reserve_seq(1);
    let reply = ask(client, &mut notes, &firmware::start_frame(seq), command::START_FIRMWARE_LOAD_NEW, seq, opts, report)
        .await?
        .ok_or(FlashFault::NoReply { step: FlashStep::Started })?;
    let (result, status) = answer(report, &reply);
    report.start = result;
    on_progress(FlashProgress::Step { step: FlashStep::Started, result, status });
    if result != Some(ResultCode::Success) {
        return Err(FlashFault::Refused { step: FlashStep::Started, result, status }.into());
    }
    report.reached = FlashStep::Started;

    load(client, &mut notes, image, opts, report, on_progress).await?;

    let seq = client.reserve_seq(1);
    let reply = ask(client, &mut notes, &firmware::verify_frame(seq), command::VERIFY_FIRMWARE_IMAGE, seq, opts, report)
        .await?
        .ok_or(FlashFault::NoReply { step: FlashStep::Verified })?;
    let (result, status) = answer(report, &reply);
    report.verify = result;
    on_progress(FlashProgress::Step { step: FlashStep::Verified, result, status });
    // A refusal here is a recorded outcome, never retried and never worked around.
    if result != Some(ResultCode::Success) {
        return Ok(());
    }
    report.reached = FlashStep::Verified;

    if opts.arm < FlashArm::Commit {
        return Ok(());
    }
    commit(client, &mut notes, opts, report, on_progress).await
}

/// Quiet the strap ahead of the transfer. Each toggle is awaited so only one command is ever
/// outstanding, but its outcome is discarded; only the abort is checked.
async fn quiesce<T: BleTransport>(
    client: &mut WhoopClient<T>,
    notes: &mut BoxStream<'static, Notification>,
    opts: &FlashOptions,
    report: &mut FlashReport,
    on_progress: &mut impl FnMut(FlashProgress),
) -> Result<(), Error> {
    if opts.quiesce == Quiesce::None {
        return Ok(());
    }
    if opts.quiesce == Quiesce::AppIdentical {
        for (op, body) in [
            (command::TOGGLE_REALTIME_HR, &HR_STREAM_OFF[..]),
            (command::SET_IMU_DATA_STREAM, &IMU_STREAM_OFF[..]),
        ] {
            let seq = client.reserve_seq(1);
            let frame = framing::command(Family::Gen5, seq, op, body);
            ask(client, notes, &frame, op, seq, opts, report).await?;
        }
    }
    let seq = client.reserve_seq(1);
    let abort = framing::command(Family::Gen5, seq, command::ABORT_HISTORICAL_TRANSMITS, &[]);
    let reply = ask(client, notes, &abort, command::ABORT_HISTORICAL_TRANSMITS, seq, opts, report)
        .await?
        .ok_or(FlashFault::NoReply { step: FlashStep::Quiesced })?;
    let (result, status) = answer(report, &reply);
    on_progress(FlashProgress::Step { step: FlashStep::Quiesced, result, status });
    if result != Some(ResultCode::Success) {
        return Err(FlashFault::Refused { step: FlashStep::Quiesced, result, status }.into());
    }
    report.reached = FlashStep::Quiesced;
    Ok(())
}

/// Send every chunk in order, one outstanding at a time. A lost reply re-sends the SAME offset; an
/// explicit refusal aborts, since the strap's duplicate-address latch would reproduce it.
async fn load<T: BleTransport>(
    client: &mut WhoopClient<T>,
    notes: &mut BoxStream<'static, Notification>,
    image: &[u8],
    opts: &FlashOptions,
    report: &mut FlashReport,
    on_progress: &mut impl FnMut(FlashProgress),
) -> Result<(), Error> {
    let start_seq = client.reserve_seq(report.chunks_planned);
    let chunks = firmware::data_frames_with(image, start_seq, opts.tail)
        .ok_or(FlashFault::Image(ImageFault::LengthMismatch))?;
    let budget = opts.retry_budget.min(MAX_RETRY_BUDGET);

    for (index, chunk) in chunks.iter().enumerate() {
        let seq = start_seq.wrapping_add(index as u8);
        loop {
            match ask(client, notes, &chunk.frame, command::LOAD_FIRMWARE_DATA_NEW, seq, opts, report).await? {
                Some(reply) => match answer(report, &reply) {
                    (Some(ResultCode::Success), _) => break,
                    (result, status) => {
                        return Err(FlashFault::ChunkRefused { offset: chunk.offset, result, status }.into())
                    }
                },
                None => {
                    report.retries += 1;
                    if report.retries > budget {
                        return Err(FlashFault::RetriesExhausted { offset: chunk.offset }.into());
                    }
                }
            }
        }
        report.chunks_sent += 1;
        let progress =
            FlashProgress::Chunk { index, chunks: chunks.len(), offset: chunk.offset, retries: report.retries };
        on_progress(progress);
    }
    report.reached = FlashStep::Loaded;
    Ok(())
}

/// The one irreversible write. Serial and battery are re-read immediately before it, both over the
/// stream and link already open, so the strap sees no long gap between the check and the commit.
async fn commit<T: BleTransport>(
    client: &mut WhoopClient<T>,
    notes: &mut BoxStream<'static, Notification>,
    opts: &FlashOptions,
    report: &mut FlashReport,
    on_progress: &mut impl FnMut(FlashProgress),
) -> Result<(), Error> {
    check_serial(client, opts).await?;
    report.battery_pct = Some(recheck_battery(client, notes, opts, report).await?);

    let seq = client.reserve_seq(1);
    let frame = firmware::process_frame(seq);
    let reply = ask(client, notes, &frame, command::PROCESS_FIRMWARE_IMAGE_NEW, seq, opts, report)
        .await?
        .ok_or(FlashFault::NoReply { step: FlashStep::Committed })?;
    let (result, status) = answer(report, &reply);
    report.process = result;
    on_progress(FlashProgress::Step { step: FlashStep::Committed, result, status });
    if result == Some(ResultCode::Success) {
        report.reached = FlashStep::Committed;
    }
    Ok(())
}

/// Write one command and wait for the reply echoing its seq. `Ok(None)` means nothing final answered
/// inside the window; a PENDING does not end the wait, so it can only ever end in a timeout.
async fn ask<T: BleTransport>(
    client: &mut WhoopClient<T>,
    notes: &mut BoxStream<'static, Notification>,
    frame: &[u8],
    cmd: u8,
    seq: u8,
    opts: &FlashOptions,
    report: &mut FlashReport,
) -> Result<Option<Frame>, Error> {
    client.write_cmd(frame).await?;
    let deadline = Instant::now() + opts.step_timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(None);
        }
        let mut stray = report.stray_frames;
        let reply = client.await_response(notes, cmd, seq, left, &mut stray).await;
        report.stray_frames = stray;
        let Some(f) = reply else { return Ok(None) };
        if resp_status(&f).1 != Some(ResultCode::Pending) {
            return Ok(Some(f));
        }
    }
}

/// Read a reply's result code and its handler's status byte, recording the latter on the report.
fn answer(report: &mut FlashReport, reply: &Frame) -> Answer {
    let status = response::firmware_status(reply);
    if status.is_some() {
        report.last_status = status;
    }
    (resp_status(reply).1, status)
}

/// The link must carry a full-width chunk frame in one confirmed write, which is by far the widest
/// thing this stack sends. An unknown MTU passes: only a reported, too-small one refuses.
fn check_link<T: BleTransport>(client: &WhoopClient<T>) -> Result<Option<usize>, FlashFault> {
    let widest = firmware::data_frame(0, 0, &[0u8; CHUNK_AMBIQ]).map_or(0, |f| f.len());
    let need = widest + ATT_WRITE_OVERHEAD;
    match client.mtu() {
        Some(have) if have < need => Err(FlashFault::LinkTooNarrow { have, need }),
        other => Ok(other),
    }
}

/// Serial allowlist plus the firmware-line check. Comparison is case-insensitive suffix.
async fn check_strap<T: BleTransport>(
    client: &WhoopClient<T>,
    opts: &FlashOptions,
    product: u32,
) -> Result<String, FlashFault> {
    let found = check_serial(client, opts).await?;
    check_board(client, opts, product).await?;
    let fw = client.firmware_revision().await.unwrap_or_else(|| "unreadable".to_string());
    if !fw.starts_with(GEN5_FW_PREFIX) {
        return Err(FlashFault::WrongStrapFirmware(fw));
    }
    Ok(found)
}

/// Image family against the strap's own board. The strap's hardware revision is the only source of
/// 5.0-versus-MG (the image header has none), so an unreadable or unrecognised revision refuses, as does
/// a board the image is not built for, or one other than the board the caller pinned.
async fn check_board<T: BleTransport>(
    client: &WhoopClient<T>,
    opts: &FlashOptions,
    product: u32,
) -> Result<(), FlashFault> {
    let hw = client.hardware_revision().await.unwrap_or_default();
    let have = Variant::classify(&hw, client.family());
    if !firmware_image::runs_on(product).contains(&have) {
        return Err(FlashFault::ImageNotForStrap(hw));
    }
    match opts.expect_variant {
        Some(want) if want != have => Err(FlashFault::WrongBoard { found: hw, have, want }),
        _ => Ok(()),
    }
}

/// The serial half alone, so the pre-commit re-check costs one read rather than two.
async fn check_serial<T: BleTransport>(client: &WhoopClient<T>, opts: &FlashOptions) -> Result<String, FlashFault> {
    let expected = opts.expect_serial.trim();
    if expected.chars().count() < MIN_EXPECTED_SERIAL {
        return Err(FlashFault::NoExpectedSerial(expected.to_string()));
    }
    let found = client.serial().await.ok_or(FlashFault::SerialUnreadable)?;
    if !found.to_ascii_uppercase().ends_with(&expected.to_ascii_uppercase()) {
        return Err(FlashFault::WrongStrap { found, expected: expected.to_string() });
    }
    Ok(found)
}

/// Battery precondition on its own stream, for the gates that run before the transfer opens one.
/// Unreadable refuses — never "proceed because we could not check".
async fn check_battery<T: BleTransport>(client: &mut WhoopClient<T>, opts: &FlashOptions) -> Result<f64, Error> {
    let have = client.battery_level().await?.ok_or(FlashFault::BatteryUnreadable)?;
    against_floor(have, opts)
}

/// The same precondition over the transfer's open stream: one round trip instead of a fixed collect
/// window, so the gap it opens before the commit stays short.
async fn recheck_battery<T: BleTransport>(
    client: &mut WhoopClient<T>,
    notes: &mut BoxStream<'static, Notification>,
    opts: &FlashOptions,
    report: &mut FlashReport,
) -> Result<f64, Error> {
    let seq = client.reserve_seq(1);
    let frame = framing::command(Family::Gen5, seq, command::GET_BATTERY_LEVEL, &[0x01]);
    let reply = ask(client, notes, &frame, command::GET_BATTERY_LEVEL, seq, opts, report)
        .await?
        .ok_or(FlashFault::BatteryUnreadable)?;
    let Some(CommandResponse::Battery { percent }) = response::decode(&reply) else {
        return Err(FlashFault::BatteryUnreadable.into());
    };
    against_floor(percent, opts)
}

fn against_floor(have: f64, opts: &FlashOptions) -> Result<f64, Error> {
    let floor = opts.min_battery_pct.max(BATTERY_FLOOR_PCT);
    if have < floor {
        return Err(FlashFault::LowBattery { have, floor }.into());
    }
    Ok(have)
}
