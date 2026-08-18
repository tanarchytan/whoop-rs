//! Main-night selection — pick the day's main sleep among detected blocks by a learned-timing score
//! (asleep minutes + an alignment bonus toward the habitual midsleep), bridging briefly-interrupted or
//! biphasic sleep into one night. Pure and deterministic; operates on numeric spans, never stored JSON.

use super::detect::is_overnight_onset;

const SECONDS_PER_DAY: i64 = 86_400;
pub const OVERNIGHT_START_HOUR: i64 = 20;
pub const OVERNIGHT_END_HOUR: i64 = 11;
const ALIGNMENT_BONUS_MIN: f64 = 90.0;
const ALIGNMENT_FULL_WINDOW_SEC: i64 = 2 * 3_600;
const ALIGNMENT_ZERO_SEC: i64 = 5 * 3_600;
const MEANINGFUL_BONUS_EPSILON: f64 = 1e-9;
const GAP_BRIDGE_MAX_MIN: i64 = 60;
const NIGHT_TAIL_BRIDGE_MAX_MIN: i64 = 90;
pub const HABITUAL_MIN_DAYS: usize = 14;
/// Trailing calendar span the per-day habitual series averages over. Three whole weeks, so a weekend
/// lie-in enters and leaves the window together, and the slack over `HABITUAL_MIN_DAYS` keeps a few
/// unworn nights from blanking the series.
pub const HABITUAL_WINDOW_DAYS: usize = 21;
const CIRCULAR_MEAN_MIN_RESULTANT: f64 = 1e-9;

/// One candidate block for selection: `[start, end]` unix seconds. A user wake/bed edit moves `end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NightBlock {
    pub start: i64,
    pub end: i64,
}

impl NightBlock {
    fn duration_s(&self) -> i64 {
        self.end - self.start
    }
}

/// One candidate scored on what its STAGES decoded: the effective onset plus the asleep and in-bed
/// seconds the hypnogram holds. A [`NightBlock`] is this with both set to the clock span, so the two
/// readings share one scorer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredNightBlock {
    pub onset: i64,
    pub asleep_s: f64,
    pub in_bed_s: f64,
}

impl ScoredNightBlock {
    /// A block read off the clock alone: asleep and in-bed are both its span.
    fn from_span(b: &NightBlock) -> Self {
        let d = b.duration_s() as f64;
        ScoredNightBlock { onset: b.start, asleep_s: d, in_bed_s: d }
    }
    fn midpoint_sec(&self) -> i64 {
        self.onset + (self.in_bed_s / 2.0) as i64
    }
}

/// One bridged group: the composing blocks (original indices, ascending) and the inter-fragment wake seams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgedNightGroup {
    pub indices: Vec<usize>,
    pub gaps: Vec<(i64, i64)>,
}

/// Why the selector chose the block it chose, for UI explainability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainNightReason {
    OnlyBlock,
    Longest,
    LongestNearUsual,
    AlignedToUsual,
}

/// The resolved pick plus the truth needed to explain it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MainNightSelection {
    pub index: usize,
    pub reason: MainNightReason,
    pub asleep_sec: i64,
}

/// One trailing-history block for learning habitual timing; `day_key` groups by local calendar day.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryBlock {
    pub start: i64,
    pub end: i64,
    pub day_key: String,
}

/// Local time-of-day in `[0, SECONDS_PER_DAY)` of a unix timestamp shifted east by `offset_s`.
fn local_sec_of_day(ts: i64, offset_s: i64) -> i64 {
    (ts + offset_s).rem_euclid(SECONDS_PER_DAY)
}

/// Local calendar-day number of a unix timestamp shifted east by `offset_s`.
fn local_day_index(ts: i64, offset_s: i64) -> i64 {
    (ts + offset_s).div_euclid(SECONDS_PER_DAY)
}

/// Smallest circular distance (seconds, `0..=43200`) between two times-of-day.
fn circular_distance_sec(a: i64, b: i64) -> i64 {
    let raw = (a - b).abs() % SECONDS_PER_DAY;
    raw.min(SECONDS_PER_DAY - raw)
}

/// The cold-start anchor: the center of the overnight band, as a time-of-day in seconds (03:30).
fn cold_start_anchor_sec() -> i64 {
    let start_sec = OVERNIGHT_START_HOUR * 3_600;
    let span = ((OVERNIGHT_END_HOUR - OVERNIGHT_START_HOUR) * 3_600 + SECONDS_PER_DAY) % SECONDS_PER_DAY;
    (start_sec + span / 2) % SECONDS_PER_DAY
}

/// Alignment bonus (minutes): full within `ALIGNMENT_FULL_WINDOW_SEC`, decaying to 0 by `ALIGNMENT_ZERO_SEC`.
fn alignment_bonus_minutes(block_mid_sec: i64, target_mid_sec: i64) -> f64 {
    let d = circular_distance_sec(block_mid_sec, target_mid_sec);
    if d <= ALIGNMENT_FULL_WINDOW_SEC {
        return ALIGNMENT_BONUS_MIN;
    }
    if d >= ALIGNMENT_ZERO_SEC {
        return 0.0;
    }
    let frac = (ALIGNMENT_ZERO_SEC - d) as f64 / (ALIGNMENT_ZERO_SEC - ALIGNMENT_FULL_WINDOW_SEC) as f64;
    ALIGNMENT_BONUS_MIN * frac
}

fn target_midsleep_sec(habitual_midsleep_sec: Option<i64>) -> i64 {
    habitual_midsleep_sec.unwrap_or_else(cold_start_anchor_sec)
}

/// Merge adjacent blocks separated by a wake gap shorter than `GAP_BRIDGE_MAX_MIN` into single blocks.
pub fn bridge_adjacent(blocks: &[NightBlock]) -> Vec<NightBlock> {
    if blocks.len() <= 1 {
        return blocks.to_vec();
    }
    let mut sorted = blocks.to_vec();
    sorted.sort_by_key(|b| b.start);
    let bridge_s = GAP_BRIDGE_MAX_MIN * 60;
    let mut out = vec![sorted[0]];
    for b in &sorted[1..] {
        let last = *out.last().expect("out is seeded from sorted[0]; never empty");
        let gap = b.start - last.end;
        if (0..bridge_s).contains(&gap) {
            *out.last_mut().expect("out non-empty: seeded + only grows") = NightBlock { start: last.start, end: last.end.max(b.end) };
        } else {
            out.push(*b);
        }
    }
    out
}

/// Every bridged group over `blocks`: the two-tier bridge (short-wake, plus the overnight night-tail
/// widening) without the winner pick. Groups ordered by start.
pub fn bridged_night_groups(blocks: &[NightBlock], offset_s: i64) -> Vec<BridgedNightGroup> {
    if blocks.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by_key(|&i| blocks[i].start);
    let bridge_s = GAP_BRIDGE_MAX_MIN * 60;
    let night_tail_bridge_s = NIGHT_TAIL_BRIDGE_MAX_MIN * 60;
    let mut bridged: Vec<NightBlock> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut gaps: Vec<Vec<(i64, i64)>> = Vec::new();
    for idx in order {
        let b = blocks[idx];
        if let Some(last) = bridged.last().copied() {
            let gap = b.start - last.end;
            let bridges = gap >= 0
                && (gap < bridge_s || (gap < night_tail_bridge_s && is_overnight_onset(b.start, offset_s)));
            if bridges {
                if gap > 0 {
                    gaps.last_mut().expect("gaps non-empty: built from |indices|-1 merge steps").push((last.end, b.start));
                }
                *bridged.last_mut().expect("bridged non-empty: seeded before merge loop") = NightBlock { start: last.start, end: last.end.max(b.end) };
                groups.last_mut().expect("groups non-empty: seeded before merge loop").push(idx);
                continue;
            }
        }
        bridged.push(b);
        groups.push(vec![idx]);
        gaps.push(Vec::new());
    }
    groups
        .into_iter()
        .zip(gaps)
        .map(|(mut g, gp)| {
            g.sort_unstable();
            BridgedNightGroup { indices: g, gaps: gp }
        })
        .collect()
}

/// Index of the day's main night by the learned-timing score (asleep minutes + alignment bonus). Highest
/// wins; exact ties break toward the earlier onset. `None` only for an empty list. Reads a block off the
/// clock alone — [`main_night_index_scored`] is the same score over decoded stage time.
pub fn main_night_index(blocks: &[NightBlock], offset_s: i64, habitual_midsleep_sec: Option<i64>) -> Option<usize> {
    let scored: Vec<ScoredNightBlock> = blocks.iter().map(ScoredNightBlock::from_span).collect();
    main_night_index_scored(&scored, offset_s, habitual_midsleep_sec)
}

/// The one main-night score: decoded asleep minutes plus the alignment bonus on the block's midpoint.
/// Highest wins; exact ties break toward the earlier onset. `None` only for an empty list.
pub fn main_night_index_scored(
    blocks: &[ScoredNightBlock],
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<usize> {
    if blocks.is_empty() {
        return None;
    }
    let target = target_midsleep_sec(habitual_midsleep_sec);
    let score = |b: &ScoredNightBlock| -> f64 {
        b.asleep_s / 60.0 + alignment_bonus_minutes(local_sec_of_day(b.midpoint_sec(), offset_s), target)
    };
    let mut best = 0usize;
    for i in 1..blocks.len() {
        let (cs, bs) = (score(&blocks[i]), score(&blocks[best]));
        let wins = if cs != bs { cs > bs } else { blocks[i].onset < blocks[best].onset };
        if wins {
            best = i;
        }
    }
    Some(best)
}

/// The original indices (ascending) of the main-night group: the main night plus any fragments bridged
/// into it. Scores the bridged spans, so a two-fragment night out-scores a lone nap.
pub fn main_night_group_indices(
    blocks: &[NightBlock],
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<Vec<usize>> {
    if blocks.is_empty() {
        return None;
    }
    let all = bridged_night_groups(blocks, offset_s);
    let bridged_spans: Vec<NightBlock> = all
        .iter()
        .map(|g| NightBlock {
            start: g.indices.iter().map(|&i| blocks[i].start).min().expect("group indices non-empty: built from block groups"),
            end: g.indices.iter().map(|&i| blocks[i].end).max().expect("group indices non-empty: built from block groups"),
        })
        .collect();
    let winner = main_night_index(&bridged_spans, offset_s, habitual_midsleep_sec)?;
    Some(all[winner].indices.clone())
}

/// The main-night group over blocks read off their DECODED stages: each candidate's span for bridging is
/// `[onset, onset + in_bed_s]`, and a group scores as the SUM of its fragments' asleep and in-bed time, so
/// an inter-fragment gap adds nothing. Indices ascending into the original list.
pub fn main_night_group_indices_scored(
    blocks: &[ScoredNightBlock],
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<Vec<usize>> {
    if blocks.is_empty() {
        return None;
    }
    let spans: Vec<NightBlock> = blocks
        .iter()
        .map(|b| NightBlock { start: b.onset, end: b.onset + b.in_bed_s as i64 })
        .collect();
    let all = bridged_night_groups(&spans, offset_s);
    let summed: Vec<ScoredNightBlock> = all
        .iter()
        .map(|g| ScoredNightBlock {
            onset: g.indices.iter().map(|&i| blocks[i].onset).min().expect("group indices non-empty: built from block groups"),
            asleep_s: g.indices.iter().map(|&i| blocks[i].asleep_s).sum(),
            in_bed_s: g.indices.iter().map(|&i| blocks[i].in_bed_s).sum(),
        })
        .collect();
    let winner = main_night_index_scored(&summed, offset_s, habitual_midsleep_sec)?;
    Some(all[winner].indices.clone())
}

/// The main-night pick plus why it won, for UI explainability. Reads a block off the clock alone;
/// [`main_night_selection_scored`] is the same pick over decoded stage time.
pub fn main_night_selection(
    blocks: &[NightBlock],
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<MainNightSelection> {
    let scored: Vec<ScoredNightBlock> = blocks.iter().map(ScoredNightBlock::from_span).collect();
    main_night_selection_scored(&scored, offset_s, habitual_midsleep_sec)
}

/// The scored main-night pick plus why it won: `asleep_sec` is the winner's decoded asleep time, and
/// "longest" ranks by that rather than by clock span.
pub fn main_night_selection_scored(
    blocks: &[ScoredNightBlock],
    offset_s: i64,
    habitual_midsleep_sec: Option<i64>,
) -> Option<MainNightSelection> {
    let idx = main_night_index_scored(blocks, offset_s, habitual_midsleep_sec)?;
    let chosen = blocks[idx];
    let asleep_sec = chosen.asleep_s as i64;
    let reason = if blocks.len() == 1 {
        MainNightReason::OnlyBlock
    } else {
        let mut dur_idx = 0usize;
        for i in 1..blocks.len() {
            let (c, b) = (blocks[i], blocks[dur_idx]);
            let wins = if c.asleep_s != b.asleep_s { c.asleep_s > b.asleep_s } else { c.onset < b.onset };
            if wins {
                dur_idx = i;
            }
        }
        let target = target_midsleep_sec(habitual_midsleep_sec);
        let chosen_bonus = alignment_bonus_minutes(local_sec_of_day(chosen.midpoint_sec(), offset_s), target);
        if idx != dur_idx {
            MainNightReason::AlignedToUsual
        } else if habitual_midsleep_sec.is_some() && chosen_bonus > MEANINGFUL_BONUS_EPSILON {
            MainNightReason::LongestNearUsual
        } else {
            MainNightReason::Longest
        }
    };
    Some(MainNightSelection { index: idx, reason, asleep_sec })
}

/// Midpoint unix second of a history block.
fn history_midpoint(b: &HistoryBlock) -> i64 {
    b.start + (b.end - b.start) / 2
}

/// The longest block of each local day, keyed by `day_key`. Selection-independent, so a nap drops out
/// without asking the main-night pick which block won.
fn longest_by_day(history: &[HistoryBlock]) -> std::collections::HashMap<&str, &HistoryBlock> {
    let mut out: std::collections::HashMap<&str, &HistoryBlock> = std::collections::HashMap::new();
    for b in history {
        let better = match out.get(b.day_key.as_str()) {
            None => true,
            Some(cur) => {
                let (bd, cd) = (b.end - b.start, cur.end - cur.start);
                bd > cd || (bd == cd && b.start < cur.start)
            }
        };
        if better {
            out.insert(b.day_key.as_str(), b);
        }
    }
    out
}

/// The user's habitual midsleep (local time-of-day seconds), or `None` on too little history. The circular
/// mean of the midpoint time-of-day of the longest block per local day (selection-independent).
pub fn habitual_midsleep_sec(history: &[HistoryBlock], offset_s: i64, min_days: usize) -> Option<i64> {
    let longest = longest_by_day(history);
    if longest.len() < min_days {
        return None;
    }
    let mid_secs: Vec<i64> =
        longest.values().map(|b| local_sec_of_day(history_midpoint(b), offset_s)).collect();
    circular_mean_sec(&mid_secs)
}

/// The habitual midsleep as it MOVES: one entry per local day in `history`, ascending, each the circular
/// mean over the trailing `window_days` calendar days ending on that day. `None` where the window holds
/// fewer than `min_days` days, so the consistency band leaves a gap rather than inventing a value.
pub fn habitual_midsleep_series(
    history: &[HistoryBlock],
    offset_s: i64,
    min_days: usize,
    window_days: usize,
) -> Vec<(String, Option<i64>)> {
    let longest = longest_by_day(history);
    let mut days: Vec<(i64, &str, i64)> = longest
        .iter()
        .map(|(k, b)| {
            let mid = history_midpoint(b);
            (local_day_index(mid, offset_s), *k, local_sec_of_day(mid, offset_s))
        })
        .collect();
    days.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    let span = window_days.max(1) as i64;
    let mut out = Vec::with_capacity(days.len());
    let mut lo = 0usize;
    for (i, day) in days.iter().enumerate() {
        while days[lo].0 <= day.0 - span {
            lo += 1;
        }
        let mids: Vec<i64> = days[lo..=i].iter().map(|d| d.2).collect();
        let mean = if mids.len() < min_days { None } else { circular_mean_sec(&mids) };
        out.push((day.1.to_string(), mean));
    }
    out
}

/// Circular mean of times-of-day (seconds) via the mean unit vector. `None` when the resultant is degenerate.
fn circular_mean_sec(secs: &[i64]) -> Option<i64> {
    if secs.is_empty() {
        return None;
    }
    let k = 2.0 * std::f64::consts::PI / SECONDS_PER_DAY as f64;
    let (mut sum_sin, mut sum_cos) = (0.0f64, 0.0f64);
    for &s in secs {
        let a = s as f64 * k;
        sum_sin += a.sin();
        sum_cos += a.cos();
    }
    let resultant = (sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / secs.len() as f64;
    if resultant < CIRCULAR_MEAN_MIN_RESULTANT {
        return None;
    }
    let mut ang = sum_sin.atan2(sum_cos);
    if ang < 0.0 {
        ang += 2.0 * std::f64::consts::PI;
    }
    let sec = (ang / k).round() as i64 % SECONDS_PER_DAY;
    Some(sec.rem_euclid(SECONDS_PER_DAY))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nb(start: i64, end: i64) -> NightBlock {
        NightBlock { start, end }
    }

    #[test]
    fn circular_distance_wraps_midnight() {
        // 23:30 vs 00:30 = 3600 s, not 82800
        assert_eq!(circular_distance_sec(23 * 3600 + 1800, 1800), 3600);
    }

    #[test]
    fn cold_start_anchor_is_0330() {
        assert_eq!(cold_start_anchor_sec(), 3 * 3600 + 1800);
    }

    #[test]
    fn alignment_bonus_full_then_decays_to_zero() {
        let t = 12_600; // 03:30
        assert_eq!(alignment_bonus_minutes(t, t), ALIGNMENT_BONUS_MIN);
        assert_eq!(alignment_bonus_minutes(t + ALIGNMENT_FULL_WINDOW_SEC, t), ALIGNMENT_BONUS_MIN);
        assert_eq!(alignment_bonus_minutes(t + ALIGNMENT_ZERO_SEC, t), 0.0);
        let mid = alignment_bonus_minutes(t + (ALIGNMENT_FULL_WINDOW_SEC + ALIGNMENT_ZERO_SEC) / 2, t);
        assert!(mid > 0.0 && mid < ALIGNMENT_BONUS_MIN);
    }

    #[test]
    fn bridge_adjacent_merges_short_gap_only() {
        let base = 1_700_000_000i64 - 1_700_000_000 % SECONDS_PER_DAY;
        let a = nb(base, base + 3600);
        let short = nb(base + 3600 + 30 * 60, base + 3 * 3600); // 30-min gap
        let far = nb(base + 6 * 3600, base + 7 * 3600); // 3h gap
        assert_eq!(bridge_adjacent(&[a, short]).len(), 1);
        assert_eq!(bridge_adjacent(&[a, far]).len(), 2);
    }

    #[test]
    fn longest_wins_cold_start_but_alignment_can_flip() {
        let base = 1_700_000_000i64 - 1_700_000_000 % SECONDS_PER_DAY; // local midnight (offset 0)
        // A: 6h, midpoint 14:00 (far from the 03:30 cold-start anchor) -> bonus 0, score 360
        let a = nb(base + 11 * 3600, base + 17 * 3600);
        // B: 5h, midpoint 03:30 (on anchor) -> bonus 90, score 390
        let b = nb(base + 3600, base + 6 * 3600);
        assert_eq!(main_night_index(&[a, b], 0, None), Some(1)); // alignment flips to the shorter B
        let sel = main_night_selection(&[a, b], 0, None).unwrap();
        assert_eq!(sel.reason, MainNightReason::AlignedToUsual);
    }

    #[test]
    fn group_indices_bridge_a_biphasic_night() {
        let base = 1_700_000_000i64 - 1_700_000_000 % SECONDS_PER_DAY;
        // two overnight fragments split by a 30-min wake, plus a far daytime nap
        let f1 = nb(base + 3600, base + 3 * 3600);
        let f2 = nb(base + 3 * 3600 + 30 * 60, base + 6 * 3600);
        let nap = nb(base + 14 * 3600, base + 15 * 3600);
        let group = main_night_group_indices(&[f1, f2, nap], 0, None).unwrap();
        assert_eq!(group, vec![0, 1]); // the two fragments, not the nap
    }

    /// A bridged group is scored as `min(start)..max(end)` over its fragments. The biphasic test above
    /// cannot see either collapse go wrong: its fragments are similar enough that the group still wins.
    /// Here one fragment carries nearly all the span, so a wrong collapse hands the night to the nap.
    #[test]
    fn a_bridged_group_is_scored_over_its_whole_span() {
        let base = 1_700_000_000i64 - 1_700_000_000 % SECONDS_PER_DAY;
        let nap = nb(base + 13 * 3600, base + 17 * 3600);

        // Long first fragment: taking the LATER start would shrink the group to 30 minutes.
        let head = nb(base, base + 7 * 3600);
        let tail = nb(base + 7 * 3600 + 30 * 60, base + 8 * 3600);
        assert_eq!(main_night_group_indices(&[head, tail, nap], 0, None), Some(vec![0, 1]));

        // Long second fragment: taking the EARLIER end would shrink it the same way.
        let stub = nb(base, base + 30 * 60);
        let bulk = nb(base + 3600, base + 8 * 3600);
        assert_eq!(main_night_group_indices(&[stub, bulk, nap], 0, None), Some(vec![0, 1]));
    }

    /// The scored twin sums asleep and in-bed time but takes the EARLIEST onset, which places the
    /// midpoint the alignment bonus is read from. A later onset moves it out of the aligned window.
    #[test]
    fn a_scored_group_takes_its_earliest_onset() {
        let base = 1_700_000_000i64 - 1_700_000_000 % SECONDS_PER_DAY;
        let sb = |onset: i64, asleep_min: f64, in_bed_h: f64| ScoredNightBlock {
            onset,
            asleep_s: asleep_min * 60.0,
            in_bed_s: in_bed_h * 3600.0,
        };
        let blocks = [
            sb(base, 420.0, 7.0),
            sb(base + 7 * 3600 + 30 * 60, 0.0, 0.5),
            sb(base + 13 * 3600, 450.0, 8.0),
        ];
        assert_eq!(main_night_group_indices_scored(&blocks, 0, None), Some(vec![0, 1]));
    }

    #[test]
    fn circular_mean_of_symmetric_times_and_degenerate() {
        // 03:00 and 04:00 -> mean 03:30
        let m = circular_mean_sec(&[3 * 3600, 4 * 3600]).unwrap();
        assert!((m - (3 * 3600 + 1800)).abs() <= 1);
        // antipodal (00:00 and 12:00) -> degenerate resultant -> None
        assert_eq!(circular_mean_sec(&[0, 12 * 3600]), None);
    }

    #[test]
    fn habitual_needs_min_days() {
        let hist: Vec<HistoryBlock> = (0..HABITUAL_MIN_DAYS)
            .map(|d| HistoryBlock {
                start: d as i64 * SECONDS_PER_DAY + 3600,
                end: d as i64 * SECONDS_PER_DAY + 6 * 3600,
                day_key: format!("d{d}"),
            })
            .collect();
        assert!(habitual_midsleep_sec(&hist, 0, HABITUAL_MIN_DAYS).is_some());
        assert!(habitual_midsleep_sec(&hist[..3], 0, HABITUAL_MIN_DAYS).is_none());
    }

    const REF: i64 = 1_749_513_600;
    fn at_hour(h: i64) -> i64 {
        REF + h * 3600
    }
    fn at_min(h: i64, m: i64) -> i64 {
        REF + h * 3600 + m * 60
    }
    fn sod(h: i64, m: i64) -> i64 {
        h * 3600 + m * 60
    }

    #[test]
    fn prefers_overnight_over_longer_daytime_nap() {
        let (night, nap) = (at_hour(23) - 86_400, at_hour(13));
        let blocks = [nb(nap, nap + 5 * 3600), nb(night, night + 4 * 3600)];
        assert_eq!(main_night_index(&blocks, 0, None), Some(1)); // overnight wins despite shorter
    }

    #[test]
    fn empty_and_score_tie_break_to_earlier_onset() {
        assert_eq!(main_night_index(&[], 0, None), None);
        let (early, late) = (at_min(0, 30), at_min(2, 30)); // mirrored either side of 03:30 anchor
        let blocks = [nb(late, late + 4 * 3600), nb(early, early + 4 * 3600)];
        assert_eq!(main_night_index(&blocks, 0, None), Some(1)); // tie -> earlier onset
    }

    /// Every realistic (nap, night) pairing, scored BOTH ways round. Asserting only the nap-first order
    /// would pass for a selector that always answers 1, so each pairing is also run night-first and must
    /// answer 0. The pairing count is asserted so the figure quoted for this gate stays checkable.
    #[test]
    fn realistic_nap_never_beats_real_night_cold_start_whichever_order_they_arrive_in() {
        let night_starts = [at_hour(20) - 86_400, at_hour(22) - 86_400, at_hour(23) - 86_400, at_hour(0), at_hour(1)];
        let nap_starts = [
            at_hour(6),
            at_hour(8),
            at_hour(10),
            at_hour(12),
            at_hour(13),
            at_hour(15),
            at_hour(17),
            at_hour(19),
            at_hour(21),
        ];
        let night_hours = [4i64, 5, 6, 7, 8, 9];
        let nap_minutes = [20i64, 30, 45, 60, 90, 120, 150, 180];
        let mut pairings = 0usize;
        for &ns in &night_starts {
            for nh in night_hours {
                for &ps in &nap_starts {
                    for pm in nap_minutes {
                        let (nap, night) = (nb(ps, ps + pm * 60), nb(ns, ns + nh * 3600));
                        assert_eq!(main_night_index(&[nap, night], 0, None), Some(1), "nap {pm}min vs night {nh}h");
                        assert_eq!(main_night_index(&[night, nap], 0, None), Some(0), "night {nh}h vs nap {pm}min");
                        pairings += 1;
                    }
                }
            }
        }
        assert_eq!(2_160, pairings);
        assert_eq!(night_starts.len() * night_hours.len() * nap_starts.len() * nap_minutes.len(), pairings);
    }

    #[test]
    fn ambiguous_long_daytime_beats_short_night_by_design() {
        let (night, day_long) = (at_hour(23) - 86_400, at_hour(12)); // 4h(+75 bonus=315) vs 6h(+0=360)
        let blocks = [nb(night, night + 4 * 3600), nb(day_long, day_long + 6 * 3600)];
        assert_eq!(main_night_index(&blocks, 0, None), Some(1));
    }

    #[test]
    fn habitual_shifts_pick_for_daytime_sleeper() {
        let habitual = sod(14, 0);
        let (day, night) = (at_hour(11), at_hour(23) - 86_400); // day mid 14:00 on habitual; night mid 02:00
        let blocks = [nb(night, night + 6 * 3600), nb(day, day + 6 * 3600)];
        assert_eq!(main_night_index(&blocks, 0, Some(habitual)), Some(1)); // habitual -> daytime
        assert_eq!(main_night_index(&blocks, 0, None), Some(0)); // cold-start -> overnight
    }

    #[test]
    fn group_indices_long_gap_night_tail_and_daytime_nap() {
        let a = at_hour(23) - 86_400;
        let b = a + 5 * 3600 + 5 * 3600; // 5h gap
        assert_eq!(main_night_group_indices(&[nb(a, a + 5 * 3600), nb(b, b + 2 * 3600)], 0, None), Some(vec![0]));
        let c = at_min(23, 30) - 86_400;
        let d = c + 3 * 3600 + 70 * 60; // 70-min overnight mid-night wake -> #861 wider bridge
        assert_eq!(main_night_group_indices(&[nb(c, c + 3 * 3600), nb(d, d + 4 * 3600)], 0, None), Some(vec![0, 1]));
        assert_eq!(bridge_adjacent(&[nb(c, c + 3 * 3600), nb(d, d + 4 * 3600)]).len(), 2); // <60 contract unchanged
        let (night, nap) = (at_hour(0), at_hour(13)); // daytime nap onset -> never a night-tail
        assert_eq!(main_night_group_indices(&[nb(night, night + 6 * 3600), nb(nap, nap + 90 * 60)], 0, None), Some(vec![0]));
        let e = at_hour(23) - 86_400;
        let f = e + 3 * 3600 + 95 * 60; // 95-min gap >= 90 -> no bridge even overnight
        assert_eq!(main_night_group_indices(&[nb(e, e + 3 * 3600), nb(f, f + 4 * 3600)], 0, None), Some(vec![1]));
    }

    #[test]
    fn selection_reasons() {
        let n = at_hour(23) - 86_400;
        let only = main_night_selection(&[nb(n, n + 7 * 3600 + 12 * 60)], 0, None).unwrap();
        assert_eq!((only.reason, only.asleep_sec), (MainNightReason::OnlyBlock, 7 * 3600 + 12 * 60));
        let (a, b) = (at_hour(22) - 86_400, at_hour(23) - 86_400 + 1800);
        let cold = main_night_selection(&[nb(a, a + 3 * 3600), nb(b, b + 6 * 3600)], 0, None).unwrap();
        assert_eq!((cold.index, cold.reason), (1, MainNightReason::Longest)); // cold-start -> longest
        let hab = sod(3, 0);
        let (night, nap) = (at_hour(23) - 86_400, at_hour(15)); // longest AND bonus-aligned
        let lnu = main_night_selection(&[nb(nap, nap + 3600), nb(night, night + 6 * 3600)], 0, Some(hab)).unwrap();
        assert_eq!((lnu.index, lnu.reason), (1, MainNightReason::LongestNearUsual));
        let (afternoon, n2) = (at_hour(13), at_hour(1)); // 5h(300) vs 4h+90 bonus(330) -> timing flips
        let atu = main_night_selection(&[nb(afternoon, afternoon + 5 * 3600), nb(n2, n2 + 4 * 3600)], 0, Some(hab)).unwrap();
        assert_eq!((atu.index, atu.reason, atu.asleep_sec), (1, MainNightReason::AlignedToUsual, 4 * 3600));
    }

    fn snb(onset: i64, asleep_s: f64, in_bed_s: f64) -> ScoredNightBlock {
        ScoredNightBlock { onset, asleep_s, in_bed_s }
    }

    /// The clock reading is the scored one with asleep = in-bed = the span, so a block staged as pure
    /// wake still scores its whole duration; read off its stages it scores nothing.
    #[test]
    fn a_block_with_no_sleep_wins_on_the_clock_and_loses_on_its_stages() {
        let (empty, real) = (at_hour(20) - 86_400, at_hour(2));
        let clock = [nb(empty, empty + 8 * 3600), nb(real, real + 5 * 3600)];
        assert_eq!(main_night_index(&clock, 0, None), Some(0)); // 480 min of clock beats 300
        let staged = [
            snb(empty, 0.0, 8.0 * 3600.0),          // eight hours, all of it wake
            snb(real, 4.5 * 3600.0, 5.0 * 3600.0),  // five hours holding 4.5 h of sleep
        ];
        assert_eq!(main_night_index_scored(&staged, 0, None), Some(1));
    }

    /// The scored reading bridges on `[onset, onset + in_bed_s]` and sums a group's fragments, so a night
    /// split by a short wake out-scores a lone nap without the gap counting as either term.
    #[test]
    fn a_scored_group_sums_its_fragments_and_excludes_the_gap() {
        let f1 = at_hour(23) - 86_400;
        let f2 = f1 + 3 * 3600 + 20 * 60; // 20 min out of bed after a 3 h fragment
        let nap = at_hour(14);
        let blocks = [
            snb(f1, 2.9 * 3600.0, 3.0 * 3600.0),
            snb(f2, 3.4 * 3600.0, 3.5 * 3600.0),
            snb(nap, 1.9 * 3600.0, 2.0 * 3600.0),
        ];
        assert_eq!(main_night_group_indices_scored(&blocks, 0, None), Some(vec![0, 1]));
        // The bare pick reports the winning FRAGMENT's own sleep; grouping is the caller's, above.
        let sel = main_night_selection_scored(&blocks, 0, None).unwrap();
        assert_eq!((sel.index, sel.asleep_sec), (1, (3.4 * 3600.0) as i64));
        assert_eq!(sel.reason, MainNightReason::Longest);
    }

    #[test]
    fn habitual_learns_regular_and_rejects_antipodal() {
        let mut hist = Vec::new();
        for d in 0..20i64 {
            let onset = at_hour(23) - 86_400 + d * 86_400; // 7h night, mid 02:30
            hist.push(HistoryBlock { start: onset, end: onset + 7 * 3600, day_key: format!("n{d}") });
            let nap = onset - 8 * 3600; // same-day nap, excluded by longest-per-day
            hist.push(HistoryBlock { start: nap, end: nap + 3600, day_key: format!("n{d}") });
        }
        assert!((habitual_midsleep_sec(&hist, 0, HABITUAL_MIN_DAYS).unwrap() - sod(2, 30)).abs() <= 1);
        let mut anti = Vec::new();
        for d in 0..16i64 {
            let onset = if d % 2 == 0 { at_min(20, 30) - 86_400 } else { at_min(8, 30) } + d * 86_400; // mids 00:00 / 12:00
            anti.push(HistoryBlock { start: onset, end: onset + 7 * 3600, day_key: format!("a{d}") });
        }
        assert_eq!(habitual_midsleep_sec(&anti, 0, HABITUAL_MIN_DAYS), None); // antipodal -> cold-start fallback
    }

    /// One 7 h night per day `d`, its onset shifted by `skew_sec` off 23:00 the evening before.
    fn nightly(days: std::ops::Range<i64>, skew_sec: impl Fn(i64) -> i64) -> Vec<HistoryBlock> {
        days.map(|d| {
            let onset = at_hour(23) - 86_400 + d * 86_400 + skew_sec(d);
            HistoryBlock { start: onset, end: onset + 7 * 3600, day_key: format!("n{d:02}") }
        })
        .collect()
    }

    fn defined(series: &[(String, Option<i64>)]) -> Vec<i64> {
        series.iter().filter_map(|(_, v)| *v).collect()
    }

    #[test]
    fn habitual_series_holds_still_for_a_stable_sleeper() {
        let hist = nightly(0..40, |_| 0);
        let series = habitual_midsleep_series(&hist, 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS);
        assert_eq!(series.len(), 40);
        assert!(series[..HABITUAL_MIN_DAYS - 1].iter().all(|(_, v)| v.is_none())); // cold start
        let vals = defined(&series);
        assert_eq!(vals.len(), 40 - (HABITUAL_MIN_DAYS - 1));
        assert!(vals.iter().all(|&v| (v - sod(2, 30)).abs() <= 1), "an unchanging habit must not drift");
    }

    #[test]
    fn habitual_series_is_none_under_the_day_floor() {
        let series = habitual_midsleep_series(&nightly(0..10, |_| 0), 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS);
        assert_eq!(series.len(), 10);
        assert!(series.iter().all(|(_, v)| v.is_none()));
        assert!(habitual_midsleep_series(&[], 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS).is_empty());
    }

    /// The slack between the window span and the day floor is what keeps a few unworn nights from
    /// blanking the band: the same history under a window equal to the floor loses days it holds here.
    #[test]
    fn habitual_series_survives_a_few_unworn_nights() {
        let hist: Vec<HistoryBlock> =
            nightly(0..40, |_| 0).into_iter().enumerate().filter(|(d, _)| !(20..24).contains(d)).map(|(_, b)| b).collect();
        assert_eq!(hist.len(), 36);
        let series = habitual_midsleep_series(&hist, 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS);
        assert!(series.last().expect("36 days in").1.is_some());
        let tight = habitual_midsleep_series(&hist, 0, HABITUAL_MIN_DAYS, HABITUAL_MIN_DAYS);
        assert!(defined(&tight).len() < defined(&series).len());
    }

    #[test]
    fn habitual_series_drifts_with_a_walking_bedtime() {
        const DAYS: i64 = 30;
        const STEP_SEC: i64 = 2 * 3600 / DAYS; // 2 h of walk spread across the month
        let series =
            habitual_midsleep_series(&nightly(0..DAYS, |d| d * STEP_SEC), 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS);
        let vals = defined(&series);
        assert!(vals.windows(2).all(|w| w[1] >= w[0]), "a one-way walk must never reverse");
        assert!(vals[vals.len() - 1] - vals[0] > 30 * 60, "the trailing mean must carry the walk");
        // Slightly: no one night moves the window mean by more than a few minutes.
        assert!(vals.windows(2).all(|w| w[1] - w[0] <= 5 * 60));
    }

    /// A habit that walks THROUGH midnight: the series must stay circular, so consecutive days differ by
    /// minutes even where the raw seconds wrap from 23:5x to 00:0x.
    #[test]
    fn habitual_series_crossing_midnight_never_jumps_a_day() {
        const DAYS: i64 = 30;
        let hist: Vec<HistoryBlock> = (0..DAYS)
            .map(|d| {
                let mid = at_min(23, 30) - 86_400 + d * 86_400 + d * 240;
                HistoryBlock {
                    start: mid - 3 * 3600 - 1800,
                    end: mid + 3 * 3600 + 1800,
                    day_key: format!("n{d:02}"),
                }
            })
            .collect();
        let vals = defined(&habitual_midsleep_series(&hist, 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS));
        assert!(vals.windows(2).all(|w| circular_distance_sec(w[1], w[0]) <= 5 * 60), "no 24 h jump");
        assert!(vals.windows(2).any(|w| (w[1] - w[0]).abs() > 12 * 3600), "the window must actually wrap");
    }

    #[test]
    fn habitual_series_last_day_matches_the_scalar_over_the_same_window() {
        let hist = nightly(0..40, |d| (d % 5) * 900);
        let series = habitual_midsleep_series(&hist, 0, HABITUAL_MIN_DAYS, HABITUAL_WINDOW_DAYS);
        let last = hist.iter().map(|b| local_day_index(history_midpoint(b), 0)).max().expect("40 days in");
        let tail: Vec<HistoryBlock> = hist
            .iter()
            .filter(|b| local_day_index(history_midpoint(b), 0) > last - HABITUAL_WINDOW_DAYS as i64)
            .cloned()
            .collect();
        let scalar = habitual_midsleep_sec(&tail, 0, HABITUAL_MIN_DAYS);
        assert!(scalar.is_some());
        assert_eq!(series.last().expect("40 days in").1, scalar);
    }
}
