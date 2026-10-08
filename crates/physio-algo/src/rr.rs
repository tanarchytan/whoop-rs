//! R-R interval filler detection.
//!
//! A WHOOP 5/MG (and a 4.0) emits an exact 500 ms R-R at rest as filler, not a measured beat. On David's
//! straps 500 occurs 21x to 391x as often as its neighbours at HR < 80, 4x to 11x at 100+, with no spike at
//! 512 (so it is raw 500 ms, not a 1/1024 unit artefact). A real 500 ms beat is 120 bpm, which a same-second
//! HR under 100 contradicts. Measured: `dev-notes/_r14/v103-filler-mechanism.md`.

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

    /// Shape of the real E4:0B histogram (5,119 at 500, ~120 per neighbour): only the spike value matches.
    #[test]
    fn spike_shape_flags_only_the_spike() {
        let hist = [(498u16, 101u32), (499, 103), (500, 5119), (501, 138), (502, 147), (512, 165)];
        let flagged: u32 = hist.iter().filter(|(rr, _)| is_rr_fill(*rr, Some(70))).map(|(_, n)| n).sum();
        assert_eq!(flagged, 5119);
    }
}
