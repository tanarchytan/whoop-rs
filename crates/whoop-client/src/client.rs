use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;

use ble_core::{BleTransport, Notification};
use whoop_protocol::deframe::DeframerMap;
use whoop_protocol::hello::GEN5_CLIENT_HELLO;
use whoop_protocol::event::ResultCode;
use whoop_protocol::response::{self, CommandResponse};
use whoop_protocol::{
    command, config, framing, haptic, trim, Channel, Family, Frame, Offload, OffloadStep, PacketType, Record,
};

use crate::error::Error;
use crate::uuids;

/// Inactivity window that ends a stalled history drain (strap off-wrist / dropped frame / CRC-bad
/// HISTORY_COMPLETE) — without it the drain would hang forever, since the real notify stream never ends.
const HIST_IDLE: Duration = Duration::from_secs(8);

/// How long `undo_trim` / `undo_trim_dry_run` listen for the strap's answer after their write.
const TRIM_REPLY_SECS: u64 = 2;

/// How long `battery_level` listens for the GET_BATTERY_LEVEL reply.
const R22_REPLY_WAIT: Duration = Duration::from_secs(3);
const BATTERY_REPLY_SECS: u64 = 2;

/// A connected WHOOP band over transport `T`. One `DeframerMap` keeps the notify channels frame-aligned;
/// the seq counter numbers outbound commands.
pub struct WhoopClient<T: BleTransport> {
    transport: T,
    family: Family,
    deframers: DeframerMap,
    seq: AtomicU8,
}

impl<T: BleTransport> WhoopClient<T> {
    pub fn new(transport: T, family: Family) -> Self {
        WhoopClient { transport, family, deframers: DeframerMap::new(family), seq: AtomicU8::new(0) }
    }

    pub fn family(&self) -> Family {
        self.family
    }

    /// Connect, ensure the OS bond, then the bond sequence: subscribe standard HR/battery (work unbonded)
    /// → write CLIENT_HELLO confirmed (the just-works bond) → subscribe the encrypted vendor group.
    /// Subscribing the encrypted chars before the bond wedges the link.
    pub async fn connect_and_bond(&mut self) -> Result<(), Error> {
        self.transport.connect().await?;
        self.transport.ensure_paired().await?;

        // Best-effort: not every stack exposes these, and they aren't bond-critical.
        let _ = self.transport.subscribe(uuids::HEART_RATE_MEASUREMENT).await;
        let _ = self.transport.subscribe(uuids::BATTERY_LEVEL).await;

        if self.family == Family::Gen5 {
            self.write_cmd(&GEN5_CLIENT_HELLO).await?;
        }
        for channel in [Channel::CmdNotify, Channel::Events, Channel::Data, Channel::Console] {
            self.transport.subscribe(uuids::characteristic(self.family, channel)).await?;
        }
        self.deframers.reset();
        Ok(())
    }

    /// Full drain of banked type-47 history (acks every chunk → advances the strap's trim cursor).
    pub async fn sync_history(&mut self) -> Result<Vec<Record>, Error> {
        self.sync_history_capturing(true, |_| Ok(()), |_| Ok(())).await
    }

    /// Drain to a record `sink` with no frame tap (see `sync_history_capturing`).
    pub async fn sync_history_with(
        &mut self,
        ack: bool,
        sink: impl FnMut(&Record) -> std::io::Result<()>,
    ) -> Result<Vec<Record>, Error> {
        self.sync_history_capturing(ack, |_| Ok(()), sink).await
    }

    /// Drain banked type-47 history. `ack = true` trims each chunk (a full wipe), `ack = false` keeps the
    /// strap (first chunk only). `on_frame` sees every reassembled frame (lossless tap, undecodable ones
    /// too); `sink` gets each decoded record. Both fire before that chunk's ACK.
    pub async fn sync_history_capturing(
        &mut self,
        ack: bool,
        mut on_frame: impl FnMut(&Frame) -> std::io::Result<()>,
        mut sink: impl FnMut(&Record) -> std::io::Result<()>,
    ) -> Result<Vec<Record>, Error> {
        let cmd_write = uuids::characteristic(self.family, Channel::CmdWrite);
        // WHOOP's command channel replies on notify, but the official app writes confirmed and the strap
        // only acts on acked writes — the transport runs these off the async executor so they can't hang it.
        let confirmed = true;
        let mut offload = Offload::new(self.family);
        let mut notifications = self.open_notifications().await?;
        self.transport.write(cmd_write, &offload.start_frame(), confirmed).await?;

        let mut records = Vec::new();
        loop {
            let note = tokio::select! {
                _ = tokio::time::sleep(HIST_IDLE) => {
                    let _ = self.transport.write(cmd_write, &offload.abort_frame(), confirmed).await;
                    break;
                }
                note = notifications.next() => note,
            };
            let Some(note) = note else { break };
            let Some(channel) = uuids::channel_of(self.family, note.uuid) else { continue };
            for frame in self.deframers.push(channel, &note.value) {
                on_frame(&frame)?;
                for step in offload.on_frame(&frame) {
                    match step {
                        OffloadStep::Record(record) => {
                            sink(&record)?;
                            records.push(record);
                        }
                        OffloadStep::Ack(wire) if ack => self.transport.write(cmd_write, &wire, confirmed).await?,
                        OffloadStep::Ack(_) => {}
                        OffloadStep::Complete => return Ok(records),
                    }
                }
            }
        }
        Ok(records)
    }

    /// Read one standard identity char and clean it (NUL/whitespace); `None` if unreadable or empty.
    async fn read_identity_string(&self, uuid: uuid::Uuid) -> Option<String> {
        let bytes = self.transport.read(uuid).await.ok()?;
        let s = ble_core::gatt_string(&bytes);
        (!s.is_empty()).then_some(s)
    }

    /// Read the strap serial from the standard identity char on the live connection (no bond needed).
    pub async fn serial(&self) -> Option<String> {
        self.read_identity_string(uuids::SERIAL_NUMBER).await
    }

    /// Read the strap's firmware revision from the standard identity char (no bond needed).
    pub async fn firmware_revision(&self) -> Option<String> {
        self.read_identity_string(uuids::FIRMWARE_REVISION).await
    }

    /// Negotiated ATT MTU, when the transport reports one. A confirmed write of `n` bytes needs `n + 3`.
    pub fn mtu(&self) -> Option<usize> {
        self.transport.mtu()
    }

    /// The strap's battery percent from GET_BATTERY_LEVEL. `None` when nothing answered.
    pub async fn battery_level(&mut self) -> Result<Option<f64>, Error> {
        let notifications = self.open_notifications().await?;
        self.send_raw(command::GET_BATTERY_LEVEL, &[0x01]).await?;
        let replies = self.collect_responses(notifications, BATTERY_REPLY_SECS).await?;
        Ok(replies.into_iter().find_map(|r| match r {
            CommandResponse::Battery { percent } => Some(percent),
            _ => None,
        }))
    }

    /// Stage or commit a firmware image on the connected strap. Gated: `opts.arm` defaults to
    /// `FlashArm::Plan`, which evaluates every precondition and sends no firmware command.
    pub async fn flash_firmware(
        &mut self,
        image: &[u8],
        opts: &crate::flash::FlashOptions,
        on_progress: impl FnMut(crate::flash::FlashProgress),
    ) -> Result<crate::flash::FlashReport, Error> {
        crate::flash::run(self, image, opts, on_progress).await
    }

    /// Collect live bpm from the standard HR characteristic (2a37) for `secs`. The strap only streams it
    /// when HR broadcast is on (see `set_broadcast_hr`); empty otherwise.
    pub async fn heart_rate(&mut self, secs: u64) -> Result<Vec<u8>, Error> {
        let _ = self.transport.subscribe(uuids::HEART_RATE_MEASUREMENT).await;
        let mut notifications = self.open_notifications().await?;
        let mut out = Vec::new();
        let deadline = tokio::time::sleep(Duration::from_secs(secs));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => break,
                note = notifications.next() => {
                    let Some(note) = note else { break };
                    if note.uuid == uuids::HEART_RATE_MEASUREMENT {
                        if let Some(bpm) = std_hr(&note.value) {
                            out.push(bpm);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Read-only identity/state sweep: open the stream, send the getters, then collect + decode their
    /// COMMAND_RESPONSE replies for a few seconds (stream opened before the writes → no missed responses).
    pub async fn info(&mut self) -> Result<Vec<CommandResponse>, Error> {
        let notifications = self.open_notifications().await?;
        let clock = if self.family == Family::Gen5 { command::GET_CLOCK_GEN5 } else { command::GET_CLOCK_GEN4 };
        // Hello opcode + body are per generation: GEN5 uses GET_HELLO (0x91) with b3 0x01; the 4.0's is
        // GET_HELLO_HARVARD (0x23) with a 9-byte client-time arg whose content is ignored.
        // UNSETTLED: the codec's `get_hello_frame` sends a 1-byte GEN4 body and the app that body, so the
        // two disagree. Neither has run against a 4.0 on this radio; a desktop 4.0 settles it.
        let (hello, hello_b3): (u8, &[u8]) = match self.family {
            Family::Gen5 => (command::GET_HELLO, &[0x01]),
            Family::Gen4 => (command::GET_HELLO_HARVARD, &[0u8; 9]),
        };
        // b3 (4th inner byte) is per-command; the wrong value is silently ignored. Hello/adv-name want
        // 0x01, data-range 0x00; the rest we leave bare.
        for (op, b3) in [
            (hello, hello_b3),
            (command::GET_ADVERTISING_NAME, &[0x01][..]),
            (command::REPORT_VERSION_INFO, &[][..]),
            (clock, &[][..]),
            (command::GET_BATTERY_LEVEL, &[0x01][..]), // the 4.0 needs b3 0x01 to reply; the 5.0 ignores it
            (command::GET_EXTENDED_BATTERY_INFO, &[0x01][..]), // 4.0 strap fuel gauge (5.0 answers unsupported)
            (command::GET_DATA_RANGE, &[0x00][..]),
        ] {
            self.send_raw(op, b3).await?;
        }
        self.collect_responses(notifications, 3).await
    }

    /// Read the 5.0 battery-pack fuel gauge (GET_BATTERY_PACK_INFO) — serial, SOC, mV, pack-id. Empty on a
    /// 4.0 (no pack command) or when no pack is attached.
    pub async fn battery_pack(&mut self) -> Result<Vec<CommandResponse>, Error> {
        let notifications = self.open_notifications().await?;
        self.send_raw(command::GET_BATTERY_PACK_INFO, &[0x01]).await?;
        self.collect_responses(notifications, 2).await
    }

    /// Send one command and collect every frame that arrives for `secs` — the probe behind `whoopctl send`,
    /// so a response can be inspected raw. Refuses the forbidden set, same as `send_raw`.
    pub async fn probe(&mut self, op: u8, payload: &[u8], secs: u64) -> Result<Vec<Frame>, Error> {
        let frame = self.checked_frame(op, payload)?;
        self.probe_frame(&frame, secs).await
    }

    /// Write one already-built command frame and collect every frame that arrives for `secs` (stream
    /// opened before the write → no missed reply). Shared by `probe` and the gated `force_trim`.
    async fn probe_frame(&mut self, frame: &[u8], secs: u64) -> Result<Vec<Frame>, Error> {
        let notifications = self.open_notifications().await?;
        self.write_cmd(frame).await?;
        let mut out = Vec::new();
        self.collect_frames(notifications, secs, |f| out.push(f)).await?;
        Ok(out)
    }

    /// Passively stream frames for `secs`, optionally filtered to one packet type (the raw-frame monitor).
    pub async fn monitor(&mut self, secs: u64, only_type: Option<u8>) -> Result<Vec<Frame>, Error> {
        let notifications = self.open_notifications().await?;
        let mut out = Vec::new();
        self.collect_frames(notifications, secs, |frame| {
            if only_type.is_none_or(|t| frame.packet_type() == t) {
                out.push(frame);
            }
        })
        .await?;
        Ok(out)
    }

    /// Switch the strap's optical collection on, capture every frame for `secs`, then switch it back off
    /// and kick a history backfill so the buffers it banked come across. No ACK, so nothing is trimmed.
    /// Opens the notify stream before the switch (race fix).
    ///
    /// The deep buffers ride this one command: the strap answers by enabling its on-demand R20 request
    /// and an optical collection source, then emits v20 (25 Hz, 6 optical channels) and v21 (100 Hz IMU)
    /// 1:1 — mostly banked as HISTORICAL_DATA, some live as REALTIME_RAW_DATA. Nothing persists, so the
    /// off write in `optical_collection` fully reverses it.
    pub async fn raw_stream<F: FnMut(Frame)>(&mut self, secs: u64, on_frame: F) -> Result<(), Error> {
        let notifications = self.open_notifications().await?;
        self.optical_collection(true).await?;
        let cmd_write = uuids::characteristic(self.family, Channel::CmdWrite);
        let mut offload = Offload::new(self.family);
        let _ = self.transport.write(cmd_write, &offload.start_frame(), true).await;
        let result = self.collect_frames(notifications, secs, on_frame).await;
        // Leave the strap as we found it — collection costs battery and fills the strap's backlog.
        let _ = self.optical_collection(false).await;
        result
    }

    /// Collect + decode the COMMAND_RESPONSE frames arriving on an already-opened stream for `secs`.
    async fn collect_responses(
        &mut self,
        notifications: BoxStream<'static, Notification>,
        secs: u64,
    ) -> Result<Vec<CommandResponse>, Error> {
        let mut out = Vec::new();
        self.collect_frames(notifications, secs, |frame| {
            if frame.packet().canonical() == PacketType::CommandResponse {
                if let Some(resp) = response::decode(&frame) {
                    out.push(resp);
                }
            }
        })
        .await?;
        Ok(out)
    }

    /// Wait on an already-open stream for the COMMAND_RESPONSE to `cmd` echoing `origin_seq`. Every
    /// notification is deframed so the stream stays aligned; frames that answer nothing count through
    /// `stray`. One push can carry an interim reply and its final one, so the LAST match in a batch
    /// wins. `None` on timeout or on a stream that ended.
    pub(crate) async fn await_response(
        &mut self,
        notifications: &mut BoxStream<'static, Notification>,
        cmd: u8,
        origin_seq: u8,
        timeout: Duration,
        stray: &mut usize,
    ) -> Option<Frame> {
        let deadline = tokio::time::sleep(timeout);
        tokio::pin!(deadline);
        loop {
            let note = tokio::select! {
                _ = &mut deadline => return None,
                note = notifications.next() => note?,
            };
            let Some(channel) = uuids::channel_of(self.family, note.uuid) else { continue };
            let mut hit = None;
            for frame in self.deframers.push(channel, &note.value) {
                let answers = frame.packet().canonical() == PacketType::CommandResponse
                    && frame.cmd() == cmd
                    && response::resp_origin_seq(&frame) == Some(origin_seq);
                if answers {
                    hit = Some(frame);
                } else {
                    *stray += 1;
                }
            }
            if hit.is_some() {
                return hit;
            }
        }
    }

    /// Pump every reassembled frame from an already-opened stream through `on_frame` until `secs` elapse.
    /// The caller opens the stream before its triggering writes (race fix). Shared by `info`/`monitor`;
    /// `sync_history` keeps its own loop (idle timeout + async ACK writes + early return).
    async fn collect_frames<F: FnMut(Frame)>(
        &mut self,
        mut notifications: BoxStream<'static, Notification>,
        secs: u64,
        mut on_frame: F,
    ) -> Result<(), Error> {
        let deadline = tokio::time::sleep(Duration::from_secs(secs));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => break,
                note = notifications.next() => {
                    let Some(note) = note else { break };
                    let Some(channel) = uuids::channel_of(self.family, note.uuid) else { continue };
                    for frame in self.deframers.push(channel, &note.value) {
                        on_frame(frame);
                    }
                }
            }
        }
        Ok(())
    }

    // ── Gated writes (reversible / non-destructive; the UI opt-in / confirmation lives above this) ──

    /// Send the 16-flag R22 deep-data enable sequence (GEN5 only; reversible). Returns the flags sent.
    pub async fn enable_r22(&self) -> Result<usize, Error> {
        if self.family != Family::Gen5 {
            return Ok(0);
        }
        let count = config::R22_SEQUENCE.len();
        // Reserve a contiguous seq block so the 16 frames carry the same running seqs as before.
        let start = self.reserve_seq(count);
        for frame in config::r22_frames(start) {
            self.write_cmd(&frame).await?;
        }
        Ok(count)
    }

    /// The R22 sequence, optionally with `enable_r22_v9_packets` at `v9`, reading each flag's reply.
    /// Returns one `(name, result)` per flag in send order; `None` means the strap never answered.
    pub async fn enable_r22_reporting(
        &mut self,
        v9: Option<u8>,
    ) -> Result<Vec<(String, Option<ResultCode>)>, Error> {
        if self.family != Family::Gen5 {
            return Ok(Vec::new());
        }
        let flags = config::r22_flags_ext(v9);
        let mut notes = self.open_notifications().await?;
        let mut out = Vec::with_capacity(flags.len());
        let mut stray = 0usize;
        for flag in &flags {
            let seq = self.next_seq();
            self.write_cmd(&config::feature_frame_named(seq, flag.name, flag.value)).await?;
            let reply = self
                .await_response(&mut notes, command::SET_CONFIG, seq, R22_REPLY_WAIT, &mut stray)
                .await;
            out.push((flag.name.to_string(), reply.as_ref().and_then(|f| response::resp_status(f).1)));
        }
        Ok(out)
    }

    /// Toggle the strap's standard-HR broadcast (Garmin/ANT). GEN5 only.
    pub async fn set_broadcast_hr(&self, on: bool) -> Result<(), Error> {
        let frame = config::device_frame(self.next_seq(), "whoop_live_hr_in_adv_ind_pkt", if on { b'1' } else { b'0' });
        self.write_cmd(&frame).await
    }

    /// Fire the one-shot maverick buzz (GEN5).
    pub async fn buzz(&self) -> Result<(), Error> {
        self.write_cmd(&haptic::maverick_buzz_frame(self.next_seq(), 1)).await
    }

    /// Warm-reboot the strap (opcode 29; stored data kept). Caller confirms; never automatic.
    pub async fn reboot(&self) -> Result<(), Error> {
        let frame = framing::command(self.family, self.next_seq(), command::REBOOT_STRAP, &[]);
        self.write_cmd(&frame).await
    }

    /// Switch the strap's raw optical collection on or off (opcode 107, payload `[revision, state]`).
    /// Session-scoped: it writes no persistent config, and the off form reverses it in full. While on,
    /// the strap produces the v20/v21 deep buffers; leaving it on costs battery and grows the backlog,
    /// so every caller pairs it with the off write.
    pub async fn optical_collection(&self, on: bool) -> Result<(), Error> {
        self.send_raw(command::SEND_OPTICAL_DATA, &[0x01, u8::from(on)]).await
    }

    /// Tell the strap which wrist it is worn on (opcode 123, payload `[revision, 0 left / 1 right]`).
    /// A persistent device-config write, so it stays out of the blind path and lives here; the caller
    /// confirms. The payload shape is inferred from the firmware's revision + wrist-value checks, so a
    /// strap that rejects it answers with a result code rather than changing anything.
    pub async fn select_wrist(&self, right: bool) -> Result<(), Error> {
        let payload = [0x01, u8::from(right)];
        let frame = framing::command(self.family, self.next_seq(), command::SELECT_WRIST, &payload);
        self.write_cmd(&frame).await
    }

    /// Ask the strap to reset its history trim and read pointers back to oldest, re-exposing records it
    /// has already handed over. Takes no payload: every other word pair on this opcode is a real trim, so
    /// the eight bytes are built here. Pointers only — it cannot un-erase flash. Returns the strap's reply.
    pub async fn undo_trim(&mut self) -> Result<Vec<Frame>, Error> {
        self.force_trim(trim::reset_to_oldest()).await
    }

    /// The positive control for `undo_trim` — same opcode, same handler, the inert payload the strap
    /// declines, so it proves the command lands without moving a pointer. An empty reply means nothing
    /// answered.
    pub async fn undo_trim_dry_run(&mut self) -> Result<Vec<Frame>, Error> {
        self.force_trim(trim::inert_probe()).await
    }

    /// The only writer of FORCE_TRIM, private so the payload can only come from `trim`'s two builders.
    async fn force_trim(&mut self, payload: [u8; 8]) -> Result<Vec<Frame>, Error> {
        let frame = framing::command(self.family, self.next_seq(), command::FORCE_TRIM, &payload);
        self.probe_frame(&frame, TRIM_REPLY_SECS).await
    }

    /// Send an arbitrary command opcode + payload, refusing the destructive / config-write set.
    pub async fn send_raw(&self, opcode: u8, payload: &[u8]) -> Result<(), Error> {
        let frame = self.checked_frame(opcode, payload)?;
        self.write_cmd(&frame).await
    }

    /// Build an outbound COMMAND frame, refusing the forbidden set — the one gate on the blind path.
    fn checked_frame(&self, opcode: u8, payload: &[u8]) -> Result<Vec<u8>, Error> {
        if command::is_forbidden(opcode) {
            return Err(Error::Forbidden(opcode));
        }
        Ok(framing::command(self.family, self.next_seq(), opcode, payload))
    }

    /// Connect (no bond) and read the standard identity chars (name/model/serial/fw/hardware). Used to
    /// confirm which band answered and to target by serial; none of these reads need the encrypted bond.
    pub async fn identify(&mut self) -> Result<Vec<(&'static str, String)>, Error> {
        self.transport.connect().await?;
        let mut out = Vec::new();
        for (label, uuid) in [
            ("name", uuids::DEVICE_NAME),
            ("model", uuids::MODEL_NUMBER),
            ("serial", uuids::SERIAL_NUMBER),
            ("firmware", uuids::FIRMWARE_REVISION),
            ("hardware", uuids::HARDWARE_REVISION),
        ] {
            if let Some(s) = self.read_identity_string(uuid).await {
                out.push((label, s));
            }
        }
        Ok(out)
    }

    /// Drop the link so the exclusive-bond band isn't left held (which would wedge the next session).
    pub async fn disconnect(&self) -> Result<(), Error> {
        self.transport.disconnect().await?;
        Ok(())
    }

    fn next_seq(&self) -> u8 {
        self.reserve_seq(1)
    }

    /// Reserve a contiguous block of `count` seq numbers and return the first. Truncating at 256 is
    /// exactly right for a mod-256 counter.
    pub(crate) fn reserve_seq(&self, count: usize) -> u8 {
        self.seq.fetch_add(count as u8, Ordering::Relaxed)
    }

    /// Open the merged notify stream. Callers open it BEFORE their triggering write (race fix).
    pub(crate) async fn open_notifications(&self) -> Result<BoxStream<'static, Notification>, Error> {
        Ok(self.transport.notifications().await?)
    }

    pub(crate) async fn write_cmd(&self, frame: &[u8]) -> Result<(), Error> {
        let cmd_write = uuids::characteristic(self.family, Channel::CmdWrite);
        self.transport.write(cmd_write, frame, true).await?;
        Ok(())
    }
}

/// Decode a BLE Heart Rate Measurement (0x2A37): byte 1 is bpm (uint8, or the low byte of a uint16 rate,
/// which equals bpm for any real heart rate). `None` if absent or zero.
fn std_hr(value: &[u8]) -> Option<u8> {
    value.get(1).copied().filter(|&b| b > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ble_core::{MockTransport, Notification};
    use whoop_protocol::framing;

    fn data_note(inner_frame: Vec<u8>) -> Notification {
        Notification { uuid: uuids::characteristic(Family::Gen5, Channel::Data), value: inner_frame }
    }

    /// One v18 record, a HISTORY_END (echo 1..8), then HISTORY_COMPLETE — a minimal single-chunk drain.
    fn history_script() -> Vec<Notification> {
        let mut v18 = vec![0u8; 80];
        v18[4..8].copy_from_slice(&1_784_000_000u32.to_le_bytes()); // unix @ inner 7 = payload 4
        let rec = framing::encode(Family::Gen5, 47, 18, 0, &v18);

        let mut end_body = vec![0u8; 18];
        end_body[10..18].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let end = framing::encode(Family::Gen5, PacketType::Metadata.to_u8(), 0, 2, &end_body);
        let complete = framing::encode(Family::Gen5, PacketType::Metadata.to_u8(), 0, 3, &[]);
        vec![data_note(rec), data_note(end), data_note(complete)]
    }

    #[tokio::test]
    async fn sync_history_emits_records_and_acks_until_complete() {
        let mock = MockTransport::new(history_script());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let records = client.sync_history().await.unwrap();
        assert_eq!(records.len(), 1);

        let writes = handle.writes();
        assert_eq!(writes.len(), 2); // SEND_HISTORICAL_DATA start, then the HISTORY_END ACK
        assert_eq!(framing::decode(Family::Gen5, &writes[0].1).unwrap().cmd(), command::SEND_HISTORICAL_DATA);
        let ack = framing::decode(Family::Gen5, &writes[1].1).unwrap();
        assert_eq!(ack.cmd(), command::HISTORICAL_DATA_RESULT);
        assert_eq!(&ack.payload()[..9], &[1, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[tokio::test]
    async fn sync_history_keep_mode_reads_without_acking() {
        let mock = MockTransport::new(history_script());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let mut seen = 0;
        let records = client
            .sync_history_with(false, |_| {
                seen += 1;
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(seen, 1); // the record was still decoded + sunk

        let writes = handle.writes();
        assert_eq!(writes.len(), 1); // only the start write — no HISTORY_END ACK, so the strap isn't trimmed
        assert_eq!(framing::decode(Family::Gen5, &writes[0].1).unwrap().cmd(), command::SEND_HISTORICAL_DATA);
    }

    #[tokio::test]
    async fn sync_history_sink_error_aborts_before_ack() {
        let mock = MockTransport::new(history_script());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let err = client
            .sync_history_with(true, |_| Err(std::io::Error::other("disk full")))
            .await;
        assert!(matches!(err, Err(Error::Sink(_))));
        // The record's sink failed before its chunk's ACK, so only the start write happened — nothing trimmed.
        assert_eq!(handle.writes().len(), 1);
    }

    #[tokio::test]
    async fn sync_history_capturing_taps_every_frame() {
        let mock = MockTransport::new(history_script());
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let (mut frames, mut recs) = (0, 0);
        client
            .sync_history_capturing(
                false,
                |_| {
                    frames += 1;
                    Ok(())
                },
                |_| {
                    recs += 1;
                    Ok(())
                },
            )
            .await
            .unwrap();
        assert_eq!(frames, 3); // record + HISTORY_END + HISTORY_COMPLETE — the last two decode to no record
        assert_eq!(recs, 1);
    }

    #[tokio::test]
    async fn send_raw_blocks_forbidden_opcodes() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let client = WhoopClient::new(mock, Family::Gen5);

        assert!(matches!(client.send_raw(command::FORCE_TRIM, &[]).await, Err(Error::Forbidden(_))));
        assert!(matches!(client.send_raw(command::SET_ADVERTISING_NAME, &[]).await, Err(Error::Forbidden(_))));
        client.send_raw(command::GET_BATTERY_LEVEL, &[]).await.unwrap();
        assert_eq!(handle.writes().len(), 1);
    }

    #[tokio::test]
    async fn enable_r22_writes_all_16_flags() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let client = WhoopClient::new(mock, Family::Gen5);

        assert_eq!(client.enable_r22().await.unwrap(), 16);
        let writes = handle.writes();
        assert_eq!(writes.len(), 16);
        for (_, wire) in &writes {
            let f = framing::decode(Family::Gen5, wire).unwrap();
            assert_eq!(f.cmd(), command::SET_CONFIG);
            assert_eq!(f.payload()[0], 0x01);
        }
    }

    fn note(channel: Channel, value: Vec<u8>) -> Notification {
        Notification { uuid: uuids::characteristic(Family::Gen5, channel), value }
    }

    #[tokio::test]
    async fn info_decodes_scripted_command_responses() {
        let mut body = vec![0u8; 20];
        body[2] = 93; // GEN5 battery = direct percent @ payload 2
        let resp = framing::encode(Family::Gen5, 36, 0, command::GET_BATTERY_LEVEL, &body);
        let mock = MockTransport::new(vec![note(Channel::CmdNotify, resp)]);
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let out = client.info().await.unwrap();
        assert!(out.iter().any(|r| matches!(r, CommandResponse::Battery { percent } if *percent == 93.0)));
    }

    #[tokio::test]
    async fn monitor_filters_to_one_packet_type() {
        let mut v18 = vec![0u8; 40];
        v18[4..8].copy_from_slice(&1u32.to_le_bytes());
        let hist = framing::encode(Family::Gen5, 47, 18, 0, &v18);
        let event = framing::encode(Family::Gen5, 48, 0, 3, &[0u8; 20]);
        let mock = MockTransport::new(vec![note(Channel::Data, hist), note(Channel::Data, event)]);
        let mut client = WhoopClient::new(mock, Family::Gen5);

        let frames = client.monitor(1, Some(47)).await.unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].packet_type(), 47);
    }

    #[tokio::test]
    async fn buzz_and_reboot_write_their_opcodes() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let client = WhoopClient::new(mock, Family::Gen5);

        client.buzz().await.unwrap();
        client.reboot().await.unwrap();
        let writes = handle.writes();
        assert_eq!(framing::decode(Family::Gen5, &writes[0].1).unwrap().cmd(), command::RUN_HAPTIC_PATTERN_MAVERICK);
        assert_eq!(framing::decode(Family::Gen5, &writes[1].1).unwrap().cmd(), command::REBOOT_STRAP);
    }

    /// Wrist-select is FORBIDDEN on the blind path (persistent config) yet reachable through its own
    /// method, and the side rides the second payload byte.
    #[tokio::test]
    async fn select_wrist_is_gated_to_its_own_method() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let client = WhoopClient::new(mock, Family::Gen5);

        assert!(client.send_raw(command::SELECT_WRIST, &[0x01, 1]).await.is_err());
        client.select_wrist(true).await.unwrap();
        client.select_wrist(false).await.unwrap();

        let writes = handle.writes();
        assert_eq!(writes.len(), 2);
        let sent = |i: usize| {
            let f = framing::decode(Family::Gen5, &writes[i].1).unwrap();
            (f.cmd(), f.payload()[1])
        };
        assert_eq!(sent(0), (command::SELECT_WRIST, 1));
        assert_eq!(sent(1), (command::SELECT_WRIST, 0));
    }

    /// The eight bytes the ERASE magic would lay down — what the two gated writes must never be.
    fn erase_magic() -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&trim::TRIM_ALL.to_le_bytes());
        out[4..].copy_from_slice(&trim::TRIM_ALL.to_le_bytes());
        out
    }

    /// The two words of the one write the mock saw, plus everything after them. A GEN5 command inner is
    /// padded to a 4-byte boundary, so the trailing pad is checked rather than compared into the words.
    fn sole_trim_write(handle: &MockTransport) -> ([u8; 8], Vec<u8>) {
        let writes = handle.writes();
        assert_eq!(writes.len(), 1);
        let f = framing::decode(Family::Gen5, &writes[0].1).unwrap();
        assert_eq!(f.cmd(), command::FORCE_TRIM);
        (f.payload()[..8].try_into().unwrap(), f.payload()[8..].to_vec())
    }

    /// The undo is opcode 25 carrying exactly eight `0xFD` bytes — never the eight `0xFE` of `TRIM_ALL`,
    /// which is one bit per byte away and erases instead.
    #[tokio::test]
    async fn undo_trim_writes_the_reset_magic_and_nothing_near_it() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        client.undo_trim().await.unwrap();

        let (words, pad) = sole_trim_write(&handle);
        assert_eq!(words, [0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD]);
        assert_ne!(words, erase_magic());
        assert!(pad.iter().all(|&b| b == 0), "only the GEN5 inner pad may follow the two words");
    }

    /// The dry run reaches the same handler with the sentinel the strap declines, so it moves no pointer.
    #[tokio::test]
    async fn undo_trim_dry_run_writes_the_inert_sentinel() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        client.undo_trim_dry_run().await.unwrap();

        let (words, pad) = sole_trim_write(&handle);
        assert_eq!(&words[..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        assert_ne!(words, [0xFD; 8]);
        assert_ne!(words, erase_magic());
        assert!(pad.iter().all(|&b| b == 0), "only the GEN5 inner pad may follow the two words");
    }

    /// The gated door does not relax the blind path: FORCE_TRIM stays refused by opcode there, whatever
    /// payload is offered, and stays in both lists.
    #[tokio::test]
    async fn force_trim_stays_forbidden_on_the_blind_path() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);

        assert!(matches!(client.send_raw(command::FORCE_TRIM, &[]).await, Err(Error::Forbidden(_))));
        assert!(matches!(client.send_raw(command::FORCE_TRIM, &[0xFD; 8]).await, Err(Error::Forbidden(_))));
        assert!(matches!(client.probe(command::FORCE_TRIM, &[0xFD; 8], 0).await, Err(Error::Forbidden(_))));
        assert!(handle.writes().is_empty());
        assert!(command::is_forbidden(command::FORCE_TRIM) && command::is_destructive(command::FORCE_TRIM));
    }

    /// The capture switches optical collection on, then off, and writes no persistent config on the way.
    #[tokio::test]
    async fn raw_stream_toggles_optical_collection_without_persistent_writes() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let mut client = WhoopClient::new(mock, Family::Gen5);
        client.raw_stream(0, |_| {}).await.unwrap();

        let sent: Vec<(u8, Vec<u8>)> = handle
            .writes()
            .iter()
            .filter_map(|(_, bytes)| framing::decode(Family::Gen5, bytes).ok())
            .map(|f| (f.cmd(), f.payload().to_vec()))
            .collect();

        let optical: Vec<u8> =
            sent.iter().filter(|(c, _)| *c == command::SEND_OPTICAL_DATA).map(|(_, p)| p[1]).collect();
        assert_eq!(optical, vec![0x01, 0x00], "on at the start, off at the end");
        assert!(
            !sent.iter().any(|(c, _)| matches!(*c, command::SET_CONFIG | command::SET_DEVICE_CONFIG)),
            "a capture must not write flash config"
        );
    }

    #[tokio::test]
    async fn gen4_enable_r22_is_a_noop() {
        let mock = MockTransport::new(Vec::new());
        let handle = mock.clone();
        let client = WhoopClient::new(mock, Family::Gen4);
        assert_eq!(client.enable_r22().await.unwrap(), 0);
        assert!(handle.writes().is_empty());
    }
}
