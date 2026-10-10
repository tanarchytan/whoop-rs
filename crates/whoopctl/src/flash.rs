//! The `flash` verb: the serial allowlist, argument validation, progress printing and the read-only
//! re-identify after a commit. The transfer itself and every safety gate live in whoop-client.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

use whoop_client::{
    FlashArm, FlashFault, FlashOptions, FlashProgress, FlashReport, FlashStep, Quiesce, BATTERY_FLOOR_PCT,
};
use whoop_protocol::bytes::to_hex;
use whoop_protocol::firmware::{data_frames_with, Tail, CHUNK_AMBIQ};
use whoop_protocol::firmware_image::inspect;
use whoop_protocol::variant::Variant;

use crate::cli::{Cli, FlashArgs};

/// The only serials this verb will ever transfer to. Compiled in: no environment variable can widen it,
/// though `WHOOPCTL_PROTECT` can still narrow it.
const FLASHABLE_SERIALS: &[&str] = &["5AM0306316"];

/// The board the allowlisted band must be (316 is an MG, `WS50_`). The client checks this against the
/// strap's own hardware revision, so a serial that moved boards would still refuse.
const FLASHABLE_VARIANT: Variant = Variant::WhoopMg;

/// How many trailing serial digits a consent flag must repeat.
const CONSENT_DIGITS: usize = 3;

/// Ceiling on the retry budget, and how long the strap is given to reboot after a commit.
const MAX_RETRIES: u32 = 16;
const REBOOT_WAIT: Duration = Duration::from_secs(20);

/// How often the chunk line is reprinted.
const PROGRESS_EVERY: usize = 250;

pub(crate) async fn run(cli: &Cli, args: &FlashArgs) -> Result<()> {
    let image = std::fs::read(&args.image)?;
    let opts = options(cli, args)?;
    describe(&args.image, &image, opts.tail)?;

    if opts.arm == FlashArm::Plan && !args.check {
        println!("\nplan only — no radio was opened and nothing was written.");
        println!("add --check to evaluate the gates on the band.");
        return Ok(());
    }

    // Connect to the allowlisted band by serial, so a run with no selector cannot bond another strap.
    let mut client = crate::connect_serial(cli, &opts.expect_serial).await?;
    if let Some(serial) = client.serial().await {
        if crate::protect_list().iter().any(|p| serial.to_ascii_uppercase().ends_with(&p.to_ascii_uppercase())) {
            client.disconnect().await.ok();
            bail!("refusing to flash: strap {serial} is protected by WHOOPCTL_PROTECT");
        }
    }

    let started = Instant::now();
    let outcome = client.flash_firmware(&image, &opts, |p| print_progress(p, started)).await;
    client.disconnect().await.ok();
    let report = outcome.inspect_err(warn_if_commit_is_unknown)?;
    print_report(&report);

    if report.reached == FlashStep::Committed {
        reidentify(cli).await?;
    }
    // A refusal is a valid outcome to record, but a run that was armed and did not get there is not a
    // success to a script reading the exit code.
    let wanted = match opts.arm {
        FlashArm::Commit => FlashStep::Committed,
        FlashArm::Stage => FlashStep::Verified,
        FlashArm::Plan => return Ok(()),
    };
    if report.reached < wanted {
        bail!("armed for {wanted:?} but the run stopped at {:?}", report.reached);
    }
    Ok(())
}

/// Turn the flags into options, refusing anything the transfer cannot honour.
fn options(cli: &Cli, args: &FlashArgs) -> Result<FlashOptions> {
    let serial = expected_serial(cli)?;
    for given in [&args.stage, &args.commit].into_iter().flatten() {
        consent_names_band(given, &serial)?;
    }
    let arm = if args.commit.is_some() {
        FlashArm::Commit
    } else if args.stage.is_some() {
        FlashArm::Stage
    } else {
        FlashArm::Plan
    };
    let tail = match args.tail.as_str() {
        "pad" => Tail::Pad,
        "exact" => Tail::Exact,
        other => bail!("--tail must be pad or exact, got {other}"),
    };
    let quiesce = match args.quiesce.as_str() {
        "none" => Quiesce::None,
        "abort" => Quiesce::AbortOnly,
        "app" => Quiesce::AppIdentical,
        other => bail!("--quiesce must be none, abort or app, got {other}"),
    };
    // NaN fails this too, since it compares against nothing.
    if args.min_battery.partial_cmp(&BATTERY_FLOOR_PCT).is_none_or(|o| o.is_lt()) {
        bail!("--min-battery {} is below the {BATTERY_FLOOR_PCT:.0}% floor", args.min_battery);
    }
    if args.retries > MAX_RETRIES {
        println!("--retries {} is above the cap; using {MAX_RETRIES}", args.retries);
    }
    Ok(FlashOptions {
        expect_serial: serial,
        expect_variant: Some(FLASHABLE_VARIANT),
        arm,
        min_battery_pct: args.min_battery,
        tail,
        quiesce,
        retry_budget: args.retries.min(MAX_RETRIES),
        step_timeout: FlashOptions::default().step_timeout,
    })
}

/// A consent flag only counts when it repeats the last digits of the band it will write to, so a
/// command copied from a note about another strap is refused rather than obeyed.
fn consent_names_band(given: &str, serial: &str) -> Result<()> {
    let want = &serial[serial.len() - CONSENT_DIGITS..];
    if given.trim() != want {
        bail!("the consent flag names band {given:?}, but this verb writes only band {want} ({serial})");
    }
    Ok(())
}

/// The serial the strap must report. `--sn` may only pick from the allowlist, never add to it.
fn expected_serial(cli: &Cli) -> Result<String> {
    let Some(sn) = &cli.sn else { return Ok(FLASHABLE_SERIALS[0].to_string()) };
    FLASHABLE_SERIALS
        .iter()
        .find(|s| s.to_ascii_uppercase().ends_with(&sn.to_ascii_uppercase()))
        .map(|s| (*s).to_string())
        .ok_or_else(|| anyhow::anyhow!("--sn {sn} is not a flashable band; this verb writes only {FLASHABLE_SERIALS:?}"))
}

/// The offline plan: what the image is, how it splits, and the first and last frame it would send.
fn describe(path: &Path, image: &[u8], tail: Tail) -> Result<()> {
    let header = inspect(image).map_err(|f| anyhow::anyhow!("image fails its own header check: {f:?}"))?;
    let chunks = data_frames_with(image, 0, tail)
        .ok_or_else(|| anyhow::anyhow!("image header does not describe a transfer"))?;

    println!("image      {}", path.display());
    println!("  version    {}", header.version);
    println!("  product    {}   container {}", header.product, header.container);
    println!("  built      {}", header.built_unix);
    println!("  payload    {} bytes, crc {:#010x}", header.payload_len, header.payload_crc);
    println!("  transfer   {} bytes -> {} chunks of {CHUNK_AMBIQ} ({tail:?} tail)", image.len(), chunks.len());
    println!("  first      {}", to_hex(&chunks[0].frame));
    println!("  last       {}", to_hex(&chunks[chunks.len() - 1].frame));
    Ok(())
}

fn print_progress(progress: FlashProgress, started: Instant) {
    match progress {
        FlashProgress::Planned { chunks, bytes } => println!("planned {chunks} chunks, {bytes} bytes"),
        FlashProgress::Step { step, result, status } => println!("{step:?}: {result:?} status {status:?}"),
        // Printed on the abort paths too, where the report never comes back.
        FlashProgress::Ended { reached, chunks_sent, chunks_planned, retries } => {
            println!("reached {reached:?} — {chunks_sent} of {chunks_planned} chunks staged, {retries} retries");
            if reached < FlashStep::Committed {
                println!("nothing was committed; the running image is untouched.");
            }
        }
        FlashProgress::Chunk { index, chunks, offset, retries } => {
            let done = index + 1;
            if done % PROGRESS_EVERY != 0 && done != chunks {
                return;
            }
            let elapsed = started.elapsed().as_secs_f64();
            let eta = if done > 0 { elapsed / done as f64 * (chunks - done) as f64 } else { 0.0 };
            println!("  chunk {done}/{chunks} at offset {offset}, {retries} retries, ~{eta:.0}s left");
        }
    }
}

/// The one stop whose outcome is genuinely unknown from here: the commit was written and no reply came
/// back, so only re-reading the strap says whether it took.
fn warn_if_commit_is_unknown(err: &whoop_client::Error) {
    if let whoop_client::Error::Flash(FlashFault::NoReply { step: FlashStep::Committed }) = err {
        eprintln!("the commit was written and nothing answered — re-read the band with `identify` to see");
        eprintln!("which image it is running. Do not re-send it.");
    }
}

fn print_report(report: &FlashReport) {
    println!("\nstrap      {}", report.serial);
    println!("battery    {:?}", report.battery_pct);
    println!("link       mtu {:?}", report.mtu);
    println!("chunks     {} of {} sent, {} retries", report.chunks_sent, report.chunks_planned, report.retries);
    println!("stray      {} frames answered nothing", report.stray_frames);
    println!("results    start {:?}  verify {:?}  process {:?}", report.start, report.verify, report.process);
    println!("status     {:?}", report.last_status);
    println!("reached    {:?} in {:.1}s", report.reached, report.elapsed.as_secs_f64());
    if report.reached != FlashStep::Committed {
        println!("nothing was committed; the running image is untouched.");
    }
}

/// Read back what the strap now reports, once it has rebooted. Read-only.
async fn reidentify(cli: &Cli) -> Result<()> {
    println!("\ncommitted — waiting {}s for the strap to reboot", REBOOT_WAIT.as_secs());
    tokio::time::sleep(REBOOT_WAIT).await;
    let mut client = crate::make_client(cli);
    match client.identify().await {
        Ok(fields) => {
            for (label, value) in fields {
                println!("{label}: {value}");
            }
        }
        Err(e) => println!("could not re-read the strap: {e}"),
    }
    client.disconnect().await.ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Parse a `flash` invocation and resolve its options.
    fn parsed(extra: &[&str]) -> Result<FlashOptions> {
        let cli = parse(extra).expect("argv rejected");
        let crate::cli::Cmd::Flash(args) = &cli.cmd else { panic!("not the flash verb") };
        options(&cli, args)
    }

    fn parse(extra: &[&str]) -> Option<Cli> {
        let mut argv = vec!["whoopctl", "flash", "--image", "x.bin"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(argv).ok()
    }

    #[test]
    fn the_default_arm_is_plan_and_each_opt_in_arms_one_more_step() {
        assert_eq!(parsed(&[]).unwrap().arm, FlashArm::Plan);
        assert_eq!(parsed(&["--check"]).unwrap().arm, FlashArm::Plan);
        assert_eq!(parsed(&["--i-agree-to-stage-firmware-on-band", "316"]).unwrap().arm, FlashArm::Stage);
        assert_eq!(parsed(&["--i-agree-to-commit-firmware-on-band", "316"]).unwrap().arm, FlashArm::Commit);
    }

    /// `--check` promises to send no firmware command, so it cannot ride along with a flag that arms
    /// one — otherwise the safety flag is on the line and means nothing.
    #[test]
    fn check_cannot_be_combined_with_an_arming_flag() {
        assert!(parse(&["--check", "--i-agree-to-stage-firmware-on-band", "316"]).is_none());
        assert!(parse(&["--check", "--i-agree-to-commit-firmware-on-band", "316"]).is_none());
    }

    /// The consent flag must repeat the band's last three digits: the old 910 wording, another strap's
    /// digits and an empty value all refuse, and refusal happens before anything connects.
    #[test]
    fn the_consent_flag_must_name_the_allowlisted_band() {
        for flag in ["--i-agree-to-stage-firmware-on-band", "--i-agree-to-commit-firmware-on-band"] {
            for wrong in ["910", "409", "360", "", "3160"] {
                assert!(parsed(&[flag, wrong]).is_err(), "{flag} {wrong:?}");
            }
            assert!(parsed(&[flag, "316"]).is_ok());
        }
        // The retired 910-named flags no longer exist.
        assert!(parse(&["--i-agree-to-stage-firmware-on-band-910"]).is_none());
        assert!(parse(&["--i-agree-to-commit-firmware-on-band-910"]).is_none());
        assert_eq!(parsed(&[]).unwrap().expect_variant, Some(Variant::WhoopMg));
    }

    #[test]
    fn the_battery_floor_is_refused_below_the_built_in_one_and_the_retry_cap_holds() {
        assert!(parsed(&["--min-battery", "10"]).is_err());
        assert!(parsed(&["--min-battery", "nan"]).is_err());
        assert_eq!(parsed(&[]).unwrap().min_battery_pct, BATTERY_FLOOR_PCT);
        assert_eq!(parsed(&["--min-battery", "95"]).unwrap().min_battery_pct, 95.0);
        assert_eq!(parsed(&["--retries", "99"]).unwrap().retry_budget, MAX_RETRIES);
        assert_eq!(parsed(&["--retries", "2"]).unwrap().retry_budget, 2);
    }

    #[test]
    fn tail_and_quiesce_are_honoured_or_refused() {
        assert_eq!(parsed(&["--tail", "exact"]).unwrap().tail, Tail::Exact);
        assert_eq!(parsed(&[]).unwrap().tail, Tail::Pad);
        assert_eq!(parsed(&["--quiesce", "none"]).unwrap().quiesce, Quiesce::None);
        assert_eq!(parsed(&["--quiesce", "app"]).unwrap().quiesce, Quiesce::AppIdentical);
        assert!(parsed(&["--tail", "half"]).is_err());
        assert!(parsed(&["--quiesce", "shout"]).is_err());
    }

    /// `--sn` may only pick from the allowlist, never add to it, and there is no wildcard.
    #[test]
    fn the_serial_allowlist_cannot_be_widened() {
        let cli = |argv: Vec<&str>| Cli::try_parse_from(argv).unwrap();
        let default = cli(vec!["whoopctl", "flash", "--image", "x.bin"]);
        assert_eq!(expected_serial(&default).unwrap(), FLASHABLE_SERIALS[0]);

        let listed = cli(vec!["whoopctl", "--sn", "306316", "flash", "--image", "x.bin"]);
        assert_eq!(expected_serial(&listed).unwrap(), FLASHABLE_SERIALS[0]);

        // 316 only: the prod 5.0 (910), the test rig (409), the old MG (360) and a near-miss all refuse.
        for sn in ["5A00960910", "910", "5AG0546409", "409", "5AM0020360", "360", "5AM0306317", "317"] {
            let other = cli(vec!["whoopctl", "--sn", sn, "flash", "--image", "x.bin"]);
            assert!(expected_serial(&other).is_err(), "{sn} must not be flashable");
        }
        assert!(!FLASHABLE_SERIALS.is_empty() && !FLASHABLE_SERIALS.contains(&""));
    }
}
