//! Does swing actually separate wake from sleep, and does it add anything a scalar has not got?
//!
//!   cargo run --release -p physio-algo --example posture_probe
//!
//! Before wiring a feature into an emission layer, measure whether it carries the information the
//! layer needs. Two questions, both on PSG truth:
//!
//!   1. how well does each feature alone rank wake above sleep (AUC over epochs)
//!   2. does swing carry anything the existing scalar jerk does not (AUC on their disagreements)
//!
//! Split by PRE-ONSET wake and MID-NIGHT wake, because those are different problems: pre-onset wake is
//! the wind-down, mid-night is the awakening. A feature can be good at one and useless at the other,
//! and pooling them would hide it.
//!
//! AUC 0.5 is a coin. Anything under ~0.6 will not carry an emission term on its own.

mod common;

use common::{dirs_of, read_accel, read_meta, read_truth};

use physio_algo::sleep::posture::{posture_series, turn_series};
use physio_algo::sleep::AccelSample;

const EPOCH: i64 = 30;
const SUSTAINED: usize = 10;
const COHORTS: [&str; 3] = ["dreamt", "aauwss", "sleep-accel"];

/// Scalar jerk per epoch, the feature the crate already has: sum of per-sample gravity deltas.
/// Rebuilt here so the comparison is like for like on the same epoch grid.
fn jerk_series(grav: &[AccelSample], start: i64, end: i64) -> Vec<Option<f64>> {
    let n = ((end - start) / EPOCH).max(0) as usize;
    let mut out = vec![None; n];
    let mut i = 0usize;
    for (k, slot) in out.iter_mut().enumerate() {
        let (a, b) = (start + k as i64 * EPOCH, start + (k as i64 + 1) * EPOCH);
        while i < grav.len() && grav[i].ts < a {
            i += 1;
        }
        let j = i + grav[i..].iter().take_while(|s| s.ts < b).count();
        let w = &grav[i..j];
        if w.len() < 2 {
            continue;
        }
        let mut m: f64 = 0.0;
        for p in w.windows(2) {
            let (dx, dy, dz) = (p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z);
            m = m.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        *slot = Some(m);
    }
    out
}

/// Probability a randomly drawn positive outranks a randomly drawn negative. Ties count a half.
fn auc(pos: &[f64], neg: &[f64]) -> Option<f64> {
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut all: Vec<(f64, bool)> =
        pos.iter().map(|v| (*v, true)).chain(neg.iter().map(|v| (*v, false))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Mid-ranks so a feature that is constant across both classes scores exactly 0.5.
    let mut rank_sum = 0.0;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += r * all[i..=j].iter().filter(|(_, p)| *p).count() as f64;
        i = j + 1;
    }
    let (np, nn) = (pos.len() as f64, neg.len() as f64);
    Some((rank_sum - np * (np + 1.0) / 2.0) / (np * nn))
}

#[derive(Default)]
struct Bucket {
    swing_w: Vec<f64>,
    swing_s: Vec<f64>,
    turn_w: Vec<f64>,
    turn_s: Vec<f64>,
    jerk_w: Vec<f64>,
    jerk_s: Vec<f64>,
}

fn line(name: &str, b: &Bucket) {
    let f = |a: Option<f64>| a.map(|v| format!("{v:.3}")).unwrap_or_else(|| "  -  ".into());
    println!(
        "  {name:<14} swing {:>6}   turn {:>6}   jerk(scalar) {:>6}    n wake {:>6} sleep {:>6}",
        f(auc(&b.swing_w, &b.swing_s)),
        f(auc(&b.turn_w, &b.turn_s)),
        f(auc(&b.jerk_w, &b.jerk_s)),
        b.swing_w.len(),
        b.swing_s.len()
    );
}

fn main() {
    println!("AUC of wake over sleep, per epoch. 0.5 = a coin, under ~0.6 will not carry a term.\n");
    for cohort in COHORTS {
        let (mut pre, mut mid) = (Bucket::default(), Bucket::default());
        let mut nights = 0;
        for dir in dirs_of(cohort) {
            let Some((w0, w1, n)) = read_meta(&dir) else { continue };
            let raw = read_truth(&dir);
            if raw.is_empty() {
                continue;
            }
            let mut truth = vec![None; n];
            for (k, t) in raw {
                if k < n && (0..4).contains(&t) {
                    truth[k] = Some(t as usize);
                }
            }
            let asleep: Vec<bool> = truth.iter().map(|t| matches!(t, Some(s) if *s != 0)).collect();
            let Some(onset) = (0..asleep.len().saturating_sub(SUSTAINED))
                .find(|i| asleep[*i..].iter().take(SUSTAINED).all(|b| *b))
            else {
                continue;
            };
            nights += 1;

            let mut grav = read_accel(&dir);
            grav.sort_by_key(|g| g.ts);
            let post = posture_series(&grav, w0, w1, EPOCH);
            let turns = turn_series(&post);
            let jerks = jerk_series(&grav, w0, w1);

            for k in 0..post.len().min(truth.len()) {
                let Some(t) = truth[k] else { continue };
                let b = if k < onset { &mut pre } else { &mut mid };
                let wake = t == 0;
                if let Some(p) = post[k] {
                    if wake { b.swing_w.push(p.swing) } else { b.swing_s.push(p.swing) }
                }
                if let Some(v) = turns[k] {
                    if wake { b.turn_w.push(v) } else { b.turn_s.push(v) }
                }
                if let Some(v) = jerks[k] {
                    if wake { b.jerk_w.push(v) } else { b.jerk_s.push(v) }
                }
            }
        }
        println!("=== {cohort}, {nights} nights");
        line("PRE-ONSET", &pre);
        line("MID-NIGHT", &mid);
        println!();
    }
}
