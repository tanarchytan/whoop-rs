//! R-R interval helpers: the standard-profile unit conversion and filler detection.
//!
//! The strap emits an exact 500 ms R-R at rest as filler, not a measured beat. A real 500 ms beat is
//! 120 bpm, which a same-second HR under 100 contradicts, so that pair is flagged (never deleted).

/// A standard-profile (0x2A37) R-R word in ms. The spec unit is 1/1024 s; a strap that already sends
/// milliseconds (`plain_ms`) passes through unchanged. Rounded to the nearest ms.
pub fn standard_rr_word_ms(raw: u16, plain_ms: bool) -> u16 {
    if plain_ms { raw } else { (f64::from(raw) * 1000.0 / 1024.0).round() as u16 }
}

/// The filler value, in milliseconds.
pub const FILL_RR_MS: u16 = 500;

/// A filler candidate only counts as suspect when the same-second HR is strictly below this.
pub const FILL_MAX_HR_BPM: u8 = 100;

/// True when `rr_ms` is the filler value and the strap's own HR for that second is below
/// [`FILL_MAX_HR_BPM`]. With no HR the beat is not flagged (conservative: it cannot be told from a real one).
pub fn is_rr_fill(rr_ms: u16, hr_bpm: Option<u8>) -> bool {
    rr_ms == FILL_RR_MS && hr_bpm.is_some_and(|h| h < FILL_MAX_HR_BPM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_word_is_1024ths_unless_the_strap_sends_ms() {
        assert_eq!(standard_rr_word_ms(1024, false), 1000);
        assert_eq!(standard_rr_word_ms(512, false), 500);
        assert_eq!(standard_rr_word_ms(819, false), 800);
        assert_eq!(standard_rr_word_ms(u16::MAX, false), 63_999);
        assert_eq!(standard_rr_word_ms(819, true), 819);
    }

    #[test]
    fn flags_filler_at_resting_hr() {
        assert!(is_rr_fill(500, Some(62)));
        assert!(is_rr_fill(500, Some(99)));
        assert!(is_rr_fill(500, Some(0)));
    }

    #[test]
    fn hr_boundary_is_strict() {
        assert!(!is_rr_fill(500, Some(100)));
        assert!(!is_rr_fill(500, Some(140)));
    }

    #[test]
    fn missing_hr_is_not_flagged() {
        assert!(!is_rr_fill(500, None));
    }

    #[test]
    fn neighbours_and_512_are_never_flagged() {
        for rr in [488, 499, 501, 512, 600, 0] {
            assert!(!is_rr_fill(rr, Some(60)), "{rr}");
        }
    }

    /// A measured histogram shape (5,119 at 500, ~120 per neighbour): only the spike value matches.
    #[test]
    fn spike_shape_flags_only_the_spike() {
        let hist = [(498u16, 101u32), (499, 103), (500, 5119), (501, 138), (502, 147), (512, 165)];
        let flagged: u32 = hist.iter().filter(|(rr, _)| is_rr_fill(*rr, Some(70))).map(|(_, n)| n).sum();
        assert_eq!(flagged, 5119);
    }
}
