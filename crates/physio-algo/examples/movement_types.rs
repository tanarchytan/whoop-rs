//! Does naming the KIND of movement beat measuring how much of it there was?
//!
//!   cargo run --release -p physio-algo --example movement_types
//!
//! Everything the stager knows about motion is one scalar per second: the norm of the inter-second
//! gravity delta. That answers "how much did the wrist move" and nothing about how. This scores the
//! alternatives side by side on identical epochs, then asks the question that actually decides the
//! design: does a categorical decomposition of movement carry information the scalar does not?
//!
//! Every number is AUC of wake over sleep against PSG truth, so it is threshold-free and the
//! features are comparable without tuning any of them. Reported per cohort, never pooled.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_truth};

use physio_algo::sleep::movement::{axis_agreement_series, movement_series};
use physio_algo::sleep::posture::{posture_series, turn_series, Posture};
use physio_algo::sleep::{AccelSample, HrSample};

const EPOCH: i64 = 30;
const SETS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];
/// Fewest labelled epochs of each class before a night can be scored at all.
const MIN_PER_CLASS: usize = 10;

/// One epoch's movement description. `None` where the epoch cannot support the measure, which is a
/// different fact from zero movement.
#[derive(Default, Clone, Copy)]
struct Move {
    /// Peak inter-second gravity delta - what the stager already reads.
    jerk: Option<f64>,
    /// Fraction of seconds whose delta cleared the night's own median. The other shipped scalar.
    move_frac: Option<f64>,
    /// Within-epoch orientation spread. Separates a swept wrist from a still one at equal magnitude.
    swing: Option<f64>,
    /// Rotation against the previous epoch. A roll-over with no sweep inside either epoch.
    turn: Option<f64>,
    /// Angle from this night's own median sleeping orientation. Self-calibrating, so it needs no
    /// reference capture and survives the band being re-mounted.
    off_posture: Option<f64>,
    /// Share of within-epoch motion on its single loudest axis. 1/3 isotropic, 1 planar.
    anisotropy: Option<f64>,
    /// Angle between this epoch's rotation axis and the previous one. Low = turning the same way
    /// twice, which a repetitive arm swing does and a restless sleeper does not.
    axis_agree: Option<f64>,
}

fn auc(pos: &[f64], neg: &[f64]) -> Option<f64> {
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut all: Vec<(f64, u8)> =
        pos.iter().map(|v| (*v, 1u8)).chain(neg.iter().map(|v| (*v, 0u8))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut rank_sum, mut i) = (0.0f64, 0usize);
    while i < all.len() {
        let mut j = i;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        rank_sum += all[i..j].iter().filter(|x| x.1 == 1).count() as f64 * avg;
        i = j;
    }
    let (n1, n0) = (pos.len() as f64, neg.len() as f64);
    Some((rank_sum - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn angle(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>().clamp(-1.0, 1.0).acos().to_degrees()
}

/// Per-epoch jerk peak and move fraction, rebuilt here so every feature reads the same epochs.
type Series = (Vec<Option<f64>>, Vec<Option<f64>>);

fn jerk_series(grav: &[AccelSample], w0: i64, n: usize) -> Series {
    let mut by_sec: std::collections::HashMap<i64, (f64, f64, f64, f64)> = Default::default();
    for g in grav {
        let e = by_sec.entry(g.ts).or_insert((0.0, 0.0, 0.0, 0.0));
        e.0 += g.x;
        e.1 += g.y;
        e.2 += g.z;
        e.3 += 1.0;
    }
    let (mut peaks, mut fracs, mut all) = (vec![None; n], vec![None; n], Vec::new());
    let mut per_epoch: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (k, slot) in per_epoch.iter_mut().enumerate() {
        let (a, b) = (w0 + k as i64 * EPOCH, w0 + (k as i64 + 1) * EPOCH);
        let mut prev: Option<(f64, f64, f64)> = None;
        for s in a..b {
            let Some(v) = by_sec.get(&s) else { continue };
            let cur = (v.0 / v.3, v.1 / v.3, v.2 / v.3);
            if let Some(p) = prev {
                let d = ((p.0 - cur.0).powi(2) + (p.1 - cur.1).powi(2) + (p.2 - cur.2).powi(2)).sqrt();
                slot.push(d);
                all.push(d);
            }
            prev = Some(cur);
        }
    }
    let scale = if all.is_empty() { 1e-6 } else { median(&all) };
    for k in 0..n {
        if per_epoch[k].is_empty() {
            continue;
        }
        peaks[k] = Some(per_epoch[k].iter().cloned().fold(f64::MIN, f64::max));
        let over = per_epoch[k].iter().filter(|d| **d > scale).count();
        fracs[k] = Some(over as f64 / per_epoch[k].len() as f64);
    }
    (peaks, fracs)
}

/// Movement TYPES. Cuts are this night's own quantiles, never absolute, so a type means the same
/// thing on a quiet sleeper and a restless one.
#[derive(PartialEq, Clone, Copy, Debug)]
enum Kind {
    Still,
    /// Motion with no orientation change - the wrist twitched where it lay.
    Fidget,
    /// Orientation changed between epochs with little sweep inside them - a roll-over.
    Shift,
    /// Sustained multi-orientation motion within the epoch - the arm was being used.
    Sweep,
}

const KINDS: [Kind; 4] = [Kind::Still, Kind::Fidget, Kind::Shift, Kind::Sweep];

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((q * s.len() as f64) as usize).min(s.len() - 1)]
}

fn classify(m: &[Move]) -> Vec<Option<Kind>> {
    let have = |f: fn(&Move) -> Option<f64>| -> Vec<f64> { m.iter().filter_map(f).collect() };
    let (js, ss, ts) = (have(|x| x.jerk), have(|x| x.swing), have(|x| x.turn));
    let (j_hi, s_hi, t_hi) = (quantile(&js, 0.75), quantile(&ss, 0.75), quantile(&ts, 0.75));
    m.iter()
        .map(|x| {
            let (j, s, t) = (x.jerk?, x.swing?, x.turn.unwrap_or(0.0));
            Some(if s >= s_hi && j >= j_hi {
                Kind::Sweep
            } else if t >= t_hi {
                Kind::Shift
            } else if j >= j_hi {
                Kind::Fidget
            } else {
                Kind::Still
            })
        })
        .collect()
}

fn features(hr: &[HrSample], grav: &[AccelSample], w0: i64, n: usize) -> Vec<Move> {
    let _ = hr;
    let post: Vec<Option<Posture>> = posture_series(grav, w0, w0 + n as i64 * EPOCH, EPOCH);
    let turns = turn_series(&post);
    let (jerks, fracs) = jerk_series(grav, w0, n);
    let mv = movement_series(grav, &post, w0, w0 + n as i64 * EPOCH, EPOCH);
    let agree = axis_agreement_series(&mv);
    let mut out: Vec<Move> = (0..n)
        .map(|k| Move {
            jerk: jerks.get(k).copied().flatten(),
            move_frac: fracs.get(k).copied().flatten(),
            swing: post.get(k).and_then(|p| p.map(|p| p.swing)),
            turn: turns.get(k).copied().flatten(),
            off_posture: None,
            anisotropy: mv.get(k).and_then(|m| m.anisotropy),
            axis_agree: agree.get(k).copied().flatten(),
        })
        .collect();
    // The reference is this night's own median sleeping orientation, so it is available at runtime
    // with no capture and no per-wearer constant. Approximated by the stillest quartile of epochs.
    let mut still: Vec<(f64, [f64; 3])> = post
        .iter()
        .enumerate()
        .filter_map(|(k, p)| p.map(|p| (out[k].jerk.unwrap_or(f64::MAX), p.dir)))
        .collect();
    still.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let take = (still.len() / 4).max(1);
    let mut acc = [0.0f64; 3];
    for (_, d) in still.iter().take(take) {
        for i in 0..3 {
            acc[i] += d[i];
        }
    }
    let norm = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
    if norm > 1e-9 {
        let refd = [acc[0] / norm, acc[1] / norm, acc[2] / norm];
        for (k, p) in post.iter().enumerate() {
            if let Some(p) = p {
                out[k].off_posture = Some(angle(&p.dir, &refd));
            }
        }
    }
    out
}

fn main() {
    println!("AUC of wake over sleep, per epoch. 0.5 is a coin. Per cohort, never pooled.\n");
    for set in SETS {
        let dirs = dirs_of(set);
        let named: [(&str, fn(&Move) -> Option<f64>); 7] = [
            ("jerk (SHIPPED)", |m| m.jerk),
            ("move_frac (SHIPPED)", |m| m.move_frac),
            ("swing", |m| m.swing),
            ("turn", |m| m.turn),
            ("off_posture", |m| m.off_posture),
            ("anisotropy (3-axis)", |m| m.anisotropy),
            ("axis_agree (3-axis)", |m| m.axis_agree),
        ];
        let mut per_feature: Vec<Vec<f64>> = vec![Vec::new(); named.len()];
        // type -> (wake epochs, total epochs), pooled across nights for the rate table.
        let mut kind_tot = [(0usize, 0usize); 4];
        // type -> counts per truth class. Truth is WAKE=0, LIGHT=1, DEEP=2, REM=3 (metrics::WAKE and
        // common::stage_idx), which is NOT the [deep, rem, light, wake] emission column order.
        let mut kind_stage = [[0usize; 4]; 4];
        let (mut nights, mut epochs) = (0usize, 0usize);

        for dir in &dirs {
            let truth = read_truth(dir);
            if truth.is_empty() {
                continue;
            }
            let (hr, grav) = (read_hr(dir), read_accel(dir));
            if grav.is_empty() {
                continue;
            }
            // The fixture window, NOT the directory name's real onset - a fixture night is rebased
            // onto a synthetic clock, so the two do not share an origin.
            let Some((w0, _, n_meta)) = read_meta(dir) else { continue };
            let n = n_meta.max(truth.keys().max().copied().unwrap_or(0) + 1);
            let m = features(&hr, &grav, w0, n);
            let is_wake = |k: &usize| truth.get(k).copied() == Some(0);
            let wake_n = truth.keys().filter(|k| is_wake(k)).count();
            if wake_n < MIN_PER_CLASS || truth.len() - wake_n < MIN_PER_CLASS {
                continue;
            }
            nights += 1;
            epochs += truth.len();

            for (fi, (_, get)) in named.iter().enumerate() {
                let (mut pos, mut neg) = (Vec::new(), Vec::new());
                for (k, _) in truth.iter() {
                    let Some(v) = m.get(*k).and_then(get) else { continue };
                    if is_wake(k) {
                        pos.push(v)
                    } else {
                        neg.push(v)
                    }
                }
                if let Some(a) = auc(&pos, &neg) {
                    per_feature[fi].push(a);
                }
            }
            for (ki, kind) in classify(&m).iter().enumerate() {
                let Some(kind) = kind else { continue };
                if !truth.contains_key(&ki) {
                    continue;
                }
                let slot = KINDS.iter().position(|k| k == kind).unwrap();
                kind_tot[slot].1 += 1;
                if is_wake(&ki) {
                    kind_tot[slot].0 += 1;
                }
                if let Some(t) = truth.get(&ki).copied() {
                    if (0..4).contains(&t) {
                        kind_stage[slot][t as usize] += 1;
                    }
                }
            }
        }

        println!("=== {set}  ({nights} nights, {epochs} labelled epochs)");
        for (fi, (name, _)) in named.iter().enumerate() {
            let v = &per_feature[fi];
            if v.is_empty() {
                println!("  {name:<22} no night could be scored");
                continue;
            }
            println!("  {:<22} AUC {:.3}  (median over {} nights)", name, median(v), v.len());
        }
        let total: usize = kind_tot.iter().map(|k| k.1).sum();
        let wake_all: usize = kind_tot.iter().map(|k| k.0).sum();
        let base = if total > 0 { wake_all as f64 / total as f64 } else { f64::NAN };
        println!("  movement TYPE, wake rate against a {:.1}% base rate:", 100.0 * base);
        for (i, kind) in KINDS.iter().enumerate() {
            let (w, t) = kind_tot[i];
            if t == 0 {
                println!("    {:<8} none", format!("{kind:?}"));
                continue;
            }
            let r = w as f64 / t as f64;
            println!("    {:<8} {:6} epochs, {:5.1}% wake  ({:+.1} pp vs base)",
                     format!("{kind:?}"), t, 100.0 * r, 100.0 * (r - base));
        }

        // Which stage does each movement type actually belong to? A type that only separates wake is
        // a wake detector; one that shifts REM against deep is a staging feature.
        let stage_base: Vec<f64> = (0..4)
            .map(|s| {
                let n: usize = kind_stage.iter().map(|k| k[s]).sum();
                if total > 0 { n as f64 / total as f64 } else { f64::NAN }
            })
            .collect();
        println!("  same types by STAGE (share of the type's epochs; night base in brackets):");
        println!("    {:<8} {:>14} {:>14} {:>14} {:>14}", "", "wake", "light", "deep", "rem");
        println!("    {:<8} {:>13.1}% {:>13.1}% {:>13.1}% {:>13.1}%   <- base",
                 "", 100.0 * stage_base[0], 100.0 * stage_base[1], 100.0 * stage_base[2],
                 100.0 * stage_base[3]);
        for (i, kind) in KINDS.iter().enumerate() {
            let n: usize = kind_stage[i].iter().sum();
            if n == 0 {
                continue;
            }
            print!("    {:<8}", format!("{kind:?}"));
            for s in 0..4 {
                let share = kind_stage[i][s] as f64 / n as f64;
                print!(" {:>7.1}% ({:+5.1})", 100.0 * share, 100.0 * (share - stage_base[s]));
            }
            println!();
        }
        println!();
    }
    println!("A type earns its place only if its wake rate is far from the base rate AND the");
    println!("separation is not already available from the scalar AUCs above it.");
}
