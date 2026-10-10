//! The clap command-line surface: global options and the subcommand set.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "whoopctl", about = "WHOOP BLE tool")]
pub(crate) struct Cli {
    /// Generation: 5 (5.0/MG, default) or 4 (4.0).
    #[arg(long, default_value_t = 5)]
    pub(crate) gen: u8,
    /// Target band by advertised name (else the first WHOOP found).
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Target band by BLE address (surest when several WHOOPs are in range).
    #[arg(long)]
    pub(crate) address: Option<String>,
    /// Target band by serial, full or suffix (connects to each candidate and reads its serial).
    #[arg(long)]
    pub(crate) sn: Option<String>,
    /// Who is wearing the strap. Calibration is per (person, strap): a new person, or the same person on
    /// a new strap, starts a fresh calibration period — you never calibrate against someone else's data.
    #[arg(long, default_value = "default")]
    pub(crate) person: String,
    /// Calibration store (SQLite) path.
    #[arg(long, default_value = "whoop-cal.db")]
    pub(crate) db: PathBuf,
    /// Opt in to the wellness HR watch: a display-only nudge on a sustained elevated at-rest heart rate.
    /// Retrospective and never medical; no device write, no buzz.
    #[arg(long, default_value_t = false)]
    pub(crate) hr_watch: bool,
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// List WHOOP bands in range.
    Scan {
        #[arg(long, default_value_t = 6)]
        secs: u64,
    },
    /// Connect (no bond) and read the standard identity chars (name/serial/fw). Read-only.
    Identify,
    /// Connect, bond, read identity/battery/data-range.
    Info,
    /// Connect, bond, read the 5.0 battery-pack fuel gauge (serial, SOC, mV, pack-id). 5.0 only.
    Pack,
    /// Connect, bond, read banked history, decode it to JSON Lines. Keeps the strap intact by default.
    Sync {
        /// Write decoded records here as JSON Lines.
        #[arg(long, default_value = "whoop-sync.jsonl")]
        out: PathBuf,
        /// Also write EVERY reassembled frame (lossless, including undecodable ones) here — the capture
        /// tap for offline field-layout analysis of unmapped records.
        #[arg(long)]
        raw: Option<PathBuf>,
        /// Drain the FULL history and advance the strap's trim cursor (the records leave the strap; the
        /// WHOOP app can no longer offload them). Default reads the first chunk and keeps everything.
        #[arg(long)]
        wipe: bool,
    },
    /// Connect, bond, stream frames (optionally one packet type).
    Monitor {
        #[arg(long)]
        r#type: Option<u8>,
        #[arg(long, default_value_t = 20)]
        secs: u64,
    },
    /// Connect, bond, stream live heart rate from the standard HR characteristic (2a37).
    Hr {
        #[arg(long, default_value_t = 15)]
        secs: u64,
        /// First enable the strap's HR broadcast (reversible) so 2a37 streams.
        #[arg(long)]
        broadcast: bool,
    },
    /// Send one command opcode (+ hex payload). Destructive/config-write opcodes are refused.
    Send { op: String, payload: Option<String> },
    /// Connect, bond, read history (keep-only, never trims), and compute derived metrics: SpO2 (4.0
    /// paired red/IR, or the 5.0/MG v26 channel ranking) and HRV-readiness from the R-R.
    Metrics,
    /// Switch optical collection on and capture the v20/v21 deep buffers it produces, then switch it off.
    Raw {
        #[arg(long, default_value_t = 20)]
        secs: u64,
        #[arg(long, default_value = "raw-stream.jsonl")]
        out: PathBuf,
    },
    /// Read the strap's body-location/status block, and with `--set left|right` write the wrist first.
    Wrist {
        /// Which wrist the strap is worn on. Omit to read only.
        #[arg(long)]
        set: Option<String>,
    },
    /// Ask the strap to rewind its history read/trim pointers to oldest, re-exposing records it has
    /// already handed over. Sends only the inert probe unless the agreement flag is given.
    UndoTrim {
        /// Send the real reset instead of the inert probe. It rewinds the strap's READ POINTER only: it
        /// cannot un-erase flash, so records the strap has already overwritten are gone either way. This
        /// takes no payload argument because a malformed payload on the same opcode trims rather than
        /// restores — the eight bytes are built in the client and cannot be supplied.
        #[arg(long = "i-agree-to-possible-loss-of-data")]
        agree: bool,
    },
    /// Enable the R22 deep-data streams (16 SET_CONFIG flags; reversible).
    R22on {
        /// Also send `enable_r22_v9_packets` at this value (1 or 2). Untried: the flag exists only on
        /// 50.41.1 and later, and no capture pins its value.
        #[arg(long = "with-v9", value_parser = ["1", "2"])]
        with_v9: Option<String>,
        /// Report each flag's reply instead of only the count sent.
        #[arg(long)]
        report: bool,
    },
    /// Fire the one-shot buzz.
    Buzz,
    /// Warm-reboot the strap (data kept).
    Reboot,
    /// Ingest a previously captured raw JSON Lines file into the calibration store (no band needed).
    Ingest {
        /// A raw-capture file (from `sync --raw`).
        capture: PathBuf,
    },
    /// Decode a deep-buffer capture (whoop-debug `deepcap` JSONL) and report the v20/v21 breakdown +
    /// sanity checks. No band needed.
    DecodeCapture {
        /// The deep-capture JSON Lines file.
        file: PathBuf,
    },
    /// Draw an ECG as stacked rhythm strips at a true, stated scale. Offline only for now: `--demo`
    /// streams a synthetic waveform, because the strap's packet layout and counts-per-mV are unread.
    Ecg(EcgArgs),
    /// Send a firmware image to the band. With no opt-in flag this only plans the transfer offline —
    /// it opens no radio and writes nothing.
    Flash(FlashArgs),
}

/// The firmware transfer's surface. Every level of arming is spelled out in full; there is no single
/// flag that goes from nothing to committed, and nothing here prompts.
#[derive(clap::Args)]
pub(crate) struct FlashArgs {
    /// The image to send: an already-extracted container file.
    #[arg(long)]
    pub(crate) image: PathBuf,
    /// Connect and evaluate every gate read-only. Sends no firmware command, so it cannot be combined
    /// with a flag that arms one.
    #[arg(long, conflicts_with_all = ["stage", "commit"])]
    pub(crate) check: bool,
    /// Open the load, send every chunk, then ask the strap to check them. Stops before the commit, so
    /// the running image is untouched.
    /// The value must be the last three digits of the one flashable band's serial (checked against the
    /// compiled-in allowlist here and against the connected strap at run time).
    #[arg(long = "i-agree-to-stage-firmware-on-band", value_name = "LAST3")]
    pub(crate) stage: Option<String>,
    /// Also send the commit. This is the irreversible step, and it implies the staging flag. Same
    /// value rule.
    #[arg(long = "i-agree-to-commit-firmware-on-band", value_name = "LAST3")]
    pub(crate) commit: Option<String>,
    /// Battery floor in percent. Values below 80 are refused.
    #[arg(long = "min-battery", default_value_t = 80.0)]
    pub(crate) min_battery: f64,
    /// Final chunk: `pad` to the full width, or the `exact` remaining bytes.
    #[arg(long, default_value = "pad")]
    pub(crate) tail: String,
    /// Timeouts tolerated across the whole transfer, capped at 16.
    #[arg(long, default_value_t = 6)]
    pub(crate) retries: u32,
    /// What to send before the load: `none`, `abort`, or the app's full `app` trio.
    #[arg(long, default_value = "abort")]
    pub(crate) quiesce: String,
}

/// The ECG renderer's surface. Every scale here is honoured or refused, never quietly changed.
#[derive(clap::Args)]
pub(crate) struct EcgArgs {
    /// Stream a synthetic waveform instead of a band. Required until the R16 layout is known.
    #[arg(long)]
    pub(crate) demo: bool,
    /// Which synthetic waveform: `pulse` (1 mV / 1 Hz square) or `ecg` (PQRST at 62 bpm).
    #[arg(long = "demo-signal", default_value = "pulse")]
    pub(crate) demo_signal: String,
    /// Inject lead-off spans, to see them marked rather than interpolated.
    #[arg(long = "demo-lead-off")]
    pub(crate) demo_lead_off: bool,
    /// Wall-clock multiplier: 1 streams in real time, 0 renders as fast as it can.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) speed: f64,
    #[arg(long, default_value_t = 30.0)]
    pub(crate) duration: f64,
    #[arg(long = "strip-seconds", default_value_t = 5.0)]
    pub(crate) strip_seconds: f64,
    /// Sweep speed. Honoured or refused — never silently changed to fit a narrow terminal.
    #[arg(long = "mm-per-s", default_value_t = 25.0)]
    pub(crate) mm_per_s: f64,
    /// Amplitude scale. Used only once a counts-per-mV is supplied.
    #[arg(long = "mm-per-mv", default_value_t = 10.0)]
    pub(crate) mm_per_mv: f64,
    /// ADC counts per millivolt. THERE IS NO DEFAULT: without it the y axis stays raw counts and every
    /// label says so. The strap's own conversion has not been read.
    #[arg(long = "counts-per-mv")]
    pub(crate) counts_per_mv: Option<f64>,
    /// Height of one strip's window, in screen millimetres.
    #[arg(long = "strip-mm", default_value_t = 24.0)]
    pub(crate) strip_mm: f64,
    /// Terminal cell height / width. A braille dot is square at 2.0; nothing can measure it, so it is
    /// assumed and printed as assumed.
    #[arg(long = "cell-aspect", default_value_t = 2.0)]
    pub(crate) cell_aspect: f64,
    /// Force the horizontal dot resolution instead of fitting it to the width.
    #[arg(long = "dots-per-mm")]
    pub(crate) dots_per_mm: Option<f64>,
    /// Sample rate. ASSUMED until the strap's rate code is read.
    #[arg(long = "sample-rate", default_value_t = 500.0)]
    pub(crate) sample_rate: f64,
    #[arg(long)]
    pub(crate) width: Option<usize>,
    #[arg(long)]
    pub(crate) height: Option<usize>,
    /// Drop the millimetre grid.
    #[arg(long = "no-grid")]
    pub(crate) no_grid: bool,
    /// Never emit colour, even to a terminal.
    #[arg(long = "no-colour")]
    pub(crate) no_colour: bool,
}
