//! Configuration shared by the cardiac order-statistic screens, defined ONCE.
//!
//! `cardiac_multivariate` (MESA, selects) and `cardiac_confirm` (AAUWSS, confirms) must ask the
//! same question with the same knobs, or "the family confirmed on wrist" compares two different
//! families. Each carried its own copy of every one of these until the round flagged it.

/// The epoch grid, and the analysis window the order statistics are centred on.
pub const EPOCH_S: f64 = 30.0;
/// The analysis window, from the library so a screen and the emission read the same span.
pub use physio_algo::sleep::cardiac::WINDOW_S;
/// Ridge on the pooled within-class scatter. The percentile columns are near-collinear by
/// construction, so without it the solve fails outright; swept and inert over four decades.
pub const RIDGE: f64 = 1e-2;
/// Column indices into `cardiac::Block::row`: the level, the three standard R-R summaries, and
/// the fourteen percentile candidates (seven absolute, then seven detrended).
pub const MEAN_HR: usize = 14;
pub const RR_SUMMARY: [usize; 3] = [15, 16, 17];
pub const PCTL: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
/// Seed for the within-night permutation that builds the null.
pub const PERM_SEED: u64 = 0x5EED_C0DE;
