//! Rig for the firmware-transfer tests: a synthetic image the codec accepts, a scripted strap that
//! answers writes so the one-outstanding-command driver runs with no radio, and a transport that can
//! fail a write.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ble_core::{BleError, BleTransport, MockTransport, Notification, FIRMWARE_REVISION, HARDWARE_REVISION, SERIAL_NUMBER};
use futures::stream::BoxStream;
use uuid::Uuid;
use whoop_protocol::crc::crc32_zlib;
use whoop_protocol::firmware::CHUNK_AMBIQ;
use whoop_protocol::{command, framing, Channel, Family};

/// Identity the rig's strap reports. The firmware string only has to name the 5.0/MG line.
pub const SERIAL: &str = "5A00000001";
pub const FIRMWARE: &str = "50.0.0.0";
/// The rig strap is an MG board, like the one the carve-out names.
pub const HARDWARE: &str = "WS50_r00";

/// Header offsets, mirrored to build a fixture. `firmware_image::inspect` is the oracle here: a wrong
/// offset makes every test that uses the image fail rather than pass quietly.
const HEADER_LEN: usize = 512;
const LEN_AT: usize = 0x04;
const CONTAINER_AT: usize = 0x0C;
const PRODUCT_AT: usize = 0x10;
const HEADER_CRC_AT: usize = 0x1F8;
const CRC_COPY_AT: usize = 0x1FC;

/// A self-consistent image carrying `payload` body bytes.
pub fn image(payload: usize) -> Vec<u8> {
    let mut v = vec![0u8; HEADER_LEN + payload];
    for (i, b) in v[HEADER_LEN..].iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    v[LEN_AT..LEN_AT + 4].copy_from_slice(&(payload as u32).to_le_bytes());
    v[CONTAINER_AT..CONTAINER_AT + 4].copy_from_slice(&5u32.to_le_bytes());
    v[PRODUCT_AT..PRODUCT_AT + 4].copy_from_slice(&13u32.to_le_bytes());
    let payload_crc = crc32_zlib(&v[HEADER_LEN..]);
    v[0..4].copy_from_slice(&payload_crc.to_le_bytes());
    v[CRC_COPY_AT..CRC_COPY_AT + 4].copy_from_slice(&payload_crc.to_le_bytes());
    let header_crc = crc32_zlib(&v[8..HEADER_CRC_AT]);
    v[HEADER_CRC_AT..HEADER_CRC_AT + 4].copy_from_slice(&header_crc.to_le_bytes());
    v
}

/// A self-consistent image of another product line (6 = the 4.0 line): header CRC re-sealed, so only the
/// family disagrees.
pub fn image_of_product(product: u32) -> Vec<u8> {
    let mut v = three_chunks();
    v[PRODUCT_AT..PRODUCT_AT + 4].copy_from_slice(&product.to_le_bytes());
    let header_crc = crc32_zlib(&v[8..HEADER_CRC_AT]);
    v[HEADER_CRC_AT..HEADER_CRC_AT + 4].copy_from_slice(&header_crc.to_le_bytes());
    v
}

/// An image that splits into exactly three full chunks.
pub fn three_chunks() -> Vec<u8> {
    image(CHUNK_AMBIQ * 3 - HEADER_LEN)
}

/// An image long enough that the one-byte sequence counter wraps more than twice during the transfer.
pub fn wrapping_chunks() -> Vec<u8> {
    image(CHUNK_AMBIQ * 600 - HEADER_LEN)
}

/// Overwrite a u32 header field WITHOUT re-sealing, so the image fails its own check.
pub fn image_with_broken_field(offset: usize, value: u32) -> Vec<u8> {
    let mut v = three_chunks();
    v[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    v
}

/// The container marker offset, for the "you handed me the decompressed image" case.
pub const CONTAINER_OFFSET: usize = CONTAINER_AT;

/// How the strap answers one command occurrence.
#[derive(Clone, Copy, Debug)]
pub enum Quirk {
    /// No reply at all.
    Silence,
    /// A final result code other than success.
    Code(u8),
    /// A result code and a driver status byte.
    Status(u8, u8),
    /// A frame answering a different command, then the real reply.
    StrayFirst,
    /// The right opcode echoing the wrong origin seq, and nothing else.
    WrongSeq,
    /// A PENDING and nothing after it.
    PendingOnly,
    /// A PENDING and its final reply in ONE notification, so both land in a single deframer push.
    PendingThenFinal,
}

/// A scripted strap. Counts each opcode so a quirk can target the nth occurrence.
#[derive(Default)]
pub struct Strap {
    battery: Mutex<Vec<Option<u8>>>,
    counts: Mutex<HashMap<u8, usize>>,
    quirks: Mutex<HashMap<(u8, usize), Quirk>>,
    log: Mutex<Vec<(u8, u8)>>,
    loads: Mutex<Vec<u32>>,
    load_bodies: Mutex<Vec<Vec<u8>>>,
}

impl Strap {
    /// A healthy strap: battery well above the floor, every command answered with success.
    pub fn healthy() -> Arc<Self> {
        let s = Strap::default();
        *s.battery.lock().unwrap() = vec![Some(95)];
        Arc::new(s)
    }

    /// Battery percent per GET_BATTERY_LEVEL, in order; the last entry repeats. `None` answers nothing.
    pub fn battery(self: &Arc<Self>, values: &[Option<u8>]) -> Arc<Self> {
        *self.battery.lock().unwrap() = values.to_vec();
        self.clone()
    }

    /// Make the `nth` (0-based) occurrence of `cmd` answer differently.
    pub fn quirk(self: &Arc<Self>, cmd: u8, nth: usize, quirk: Quirk) -> Arc<Self> {
        self.quirks.lock().unwrap().insert((cmd, nth), quirk);
        self.clone()
    }

    /// Every `(opcode, seq)` written to the strap, in order.
    pub fn log(&self) -> Vec<(u8, u8)> {
        self.log.lock().unwrap().clone()
    }

    /// Opcodes written in order, with the battery poll dropped — it is a gate read, not the flow.
    pub fn flow(&self) -> Vec<u8> {
        self.log().into_iter().map(|(c, _)| c).filter(|&c| c != command::GET_BATTERY_LEVEL).collect()
    }

    /// Offsets carried by the LOAD frames written, in order (a re-send repeats its offset).
    pub fn load_offsets(&self) -> Vec<u32> {
        self.loads.lock().unwrap().clone()
    }

    /// The declared chunk length and body of the last LOAD frame written.
    pub fn last_load_body(&self) -> Option<Vec<u8>> {
        self.load_bodies.lock().unwrap().last().cloned()
    }

    fn answer(&self, wire: &[u8]) -> Vec<Notification> {
        let Ok(f) = framing::decode(Family::Gen5, wire) else { return Vec::new() };
        let (cmd, seq) = (f.cmd(), f.seq());
        self.log.lock().unwrap().push((cmd, seq));
        if cmd == command::LOAD_FIRMWARE_DATA_NEW {
            let p = f.payload();
            self.loads.lock().unwrap().push(u32::from_le_bytes(p[1..5].try_into().unwrap()));
            self.load_bodies.lock().unwrap().push(p.to_vec());
        }
        let nth = {
            let mut counts = self.counts.lock().unwrap();
            let slot = counts.entry(cmd).or_default();
            let nth = *slot;
            *slot += 1;
            nth
        };

        if cmd == command::GET_BATTERY_LEVEL {
            let queue = self.battery.lock().unwrap();
            let pick = queue.get(nth.min(queue.len().saturating_sub(1))).copied().flatten();
            return pick.map_or_else(Vec::new, |pct| vec![note(cmd, &[seq, 1, pct, 0])]);
        }

        match self.quirks.lock().unwrap().get(&(cmd, nth)).copied() {
            Some(Quirk::Silence) => Vec::new(),
            Some(Quirk::Code(c)) => vec![note(cmd, &body(cmd, seq, c, 0))],
            Some(Quirk::Status(c, s)) => vec![note(cmd, &body(cmd, seq, c, s))],
            Some(Quirk::PendingOnly) => vec![note(cmd, &body(cmd, seq, 2, 0))],
            Some(Quirk::PendingThenFinal) => {
                vec![batch(cmd, &[body(cmd, seq, 2, 0), body(cmd, seq, 1, 0)])]
            }
            Some(Quirk::WrongSeq) => vec![note(cmd, &body(cmd, seq.wrapping_add(1), 1, 0))],
            Some(Quirk::StrayFirst) => vec![
                note(command::GET_CLOCK_GEN5, &[seq, 1]),
                note(cmd, &body(cmd, seq, 1, 0)),
            ],
            None => vec![note(cmd, &body(cmd, seq, 1, 0))],
        }
    }
}

/// A COMMAND_RESPONSE body: origin-seq echo, result, then the handler's own bytes.
fn body(cmd: u8, seq: u8, result: u8, status: u8) -> Vec<u8> {
    match cmd {
        command::VERIFY_FIRMWARE_IMAGE => vec![seq, result, 0x01],
        command::ABORT_HISTORICAL_TRANSMITS => vec![seq, result],
        _ => vec![seq, result, 0x01, status],
    }
}

fn note(cmd: u8, payload: &[u8]) -> Notification {
    batch(cmd, std::slice::from_ref(&payload.to_vec()))
}

/// Several replies in one notification, so the deframer yields them all from a single push.
fn batch(cmd: u8, payloads: &[Vec<u8>]) -> Notification {
    let mut value = Vec::new();
    for p in payloads {
        value.extend_from_slice(&framing::encode(Family::Gen5, 36, 0, cmd, p));
    }
    Notification { uuid: whoop_client::characteristic(Family::Gen5, Channel::CmdNotify), value }
}

/// A transport backed by `strap`, serving the default identity and reporting no MTU.
pub fn transport(strap: &Arc<Strap>) -> MockTransport {
    identity(strap, Some(SERIAL), Some(FIRMWARE))
}

/// The same, on a link that reports a negotiated MTU.
pub fn transport_with_mtu(strap: &Arc<Strap>, mtu: usize) -> MockTransport {
    transport(strap).with_mtu(mtu)
}

/// A transport backed by `strap` serving a chosen identity; `None` reads as unreadable.
pub fn identity(strap: &Arc<Strap>, serial: Option<&str>, firmware: Option<&str>) -> MockTransport {
    let s = strap.clone();
    let mut t = MockTransport::with_responder(Arc::new(move |w: &[u8]| s.answer(w)));
    if let Some(serial) = serial {
        t = t.with_read(SERIAL_NUMBER, serial.as_bytes().to_vec());
    }
    if let Some(firmware) = firmware {
        t = t.with_read(FIRMWARE_REVISION, firmware.as_bytes().to_vec());
    }
    t.with_read(HARDWARE_REVISION, HARDWARE.as_bytes().to_vec())
}

/// The default rig, but reporting a chosen hardware revision; `None` reads as unreadable.
pub fn board(strap: &Arc<Strap>, hardware: Option<&str>) -> MockTransport {
    let s = strap.clone();
    let mut t = MockTransport::with_responder(Arc::new(move |w: &[u8]| s.answer(w)))
        .with_read(SERIAL_NUMBER, SERIAL.as_bytes().to_vec())
        .with_read(FIRMWARE_REVISION, FIRMWARE.as_bytes().to_vec());
    if let Some(hw) = hardware {
        t = t.with_read(HARDWARE_REVISION, hw.as_bytes().to_vec());
    }
    t
}

/// Wraps a transport and fails every write past `ok_writes`, to prove a mid-transfer link loss aborts.
#[derive(Clone)]
pub struct Flaky {
    inner: MockTransport,
    ok_writes: usize,
    seen: Arc<AtomicUsize>,
}

impl Flaky {
    pub fn new(inner: MockTransport, ok_writes: usize) -> Self {
        Flaky { inner, ok_writes, seen: Arc::new(AtomicUsize::new(0)) }
    }
}

#[async_trait]
impl BleTransport for Flaky {
    async fn connect(&mut self) -> Result<(), BleError> {
        self.inner.connect().await
    }

    async fn subscribe(&self, characteristic: Uuid) -> Result<(), BleError> {
        self.inner.subscribe(characteristic).await
    }

    async fn write(&self, characteristic: Uuid, data: &[u8], with_response: bool) -> Result<(), BleError> {
        if self.seen.fetch_add(1, Ordering::Relaxed) >= self.ok_writes {
            return Err(BleError::Backend("write timed out".into()));
        }
        self.inner.write(characteristic, data, with_response).await
    }

    async fn read(&self, characteristic: Uuid) -> Result<Vec<u8>, BleError> {
        self.inner.read(characteristic).await
    }

    async fn notifications(&self) -> Result<BoxStream<'static, Notification>, BleError> {
        self.inner.notifications().await
    }
}
