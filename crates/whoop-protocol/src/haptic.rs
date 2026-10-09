//! Haptics. The 5/MG one-shot buzz is the maverick opcode with the "notify" preset body; the 4.0 preset
//! buzz rides RUN_HAPTICS_PATTERN.

use crate::command;
use crate::family::Family;
use crate::framing;

/// Most pulses one maverick buzz can ask for: `overallLoop` is the repeats after the first, capped at 7.
pub const MAVERICK_MAX_LOOPS: u8 = 8;

/// A ready-to-write 5/MG buzz (RUN_HAPTIC_PATTERN_MAVERICK 0x13, the notification preset) of `loops`
/// pulses. The body is `[revision][effects x8][loopControl u16 LE][overallLoop]`; `overallLoop` counts
/// repeats after the first pulse, so it is `loops - 1`, clamped to 0..=7. `loops` 0 and 1 are both one
/// pulse, the original one-shot frame.
pub fn maverick_buzz_frame(seq: u8, loops: u8) -> Vec<u8> {
    let overall_loop = loops.saturating_sub(1).min(MAVERICK_MAX_LOOPS - 1);
    let body = [0x01u8, 47, 152, 0, 0, 0, 0, 0, 0, 0, 0, overall_loop];
    framing::command(Family::Gen5, seq, command::RUN_HAPTIC_PATTERN_MAVERICK, &body)
}

/// RUN_HAPTICS_PATTERN body (5 bytes): `[pattern_id][loops][0][0][0]`. The 4.0 preset buzz (pattern 2 =
/// the graduated alarm buzz). A 5/MG strap only honours [`maverick_buzz_frame`], so nothing in
/// `whoop-client` sends this opcode; the caller picks the form for the family it is talking to.
pub fn run_haptics_pattern(pattern_id: u8, loops: u8) -> [u8; 5] {
    [pattern_id, loops, 0, 0, 0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `overallLoop` byte of the frame's body, read back from the bytes on the wire.
    fn overall_loop(loops: u8) -> u8 {
        let frame = maverick_buzz_frame(7, loops);
        // The body's last byte, then one pad byte and the 4-byte CRC.
        frame[frame.len() - 6]
    }

    #[test]
    fn one_pulse_is_the_original_constant_frame() {
        let original = framing::command(
            Family::Gen5, 7, command::RUN_HAPTIC_PATTERN_MAVERICK, &[0x01, 47, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert_eq!(maverick_buzz_frame(7, 1), original);
        assert_eq!(maverick_buzz_frame(7, 0), original);
    }

    #[test]
    fn overall_loop_counts_the_repeats_after_the_first_pulse() {
        for (loops, want) in [(1, 0), (2, 1), (3, 2), (5, 4), (8, 7), (255, 7), (0, 0)] {
            assert_eq!(overall_loop(loops), want, "loops = {loops}");
        }
    }

    #[test]
    fn run_haptics_pattern_body() {
        assert_eq!(run_haptics_pattern(2, 3), [2, 3, 0, 0, 0]);
    }
}
