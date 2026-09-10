//! How far the posterior-marginal rule sits from Viterbi on real nights, and how much posterior
//! mass the Viterbi path actually carries.
//!
//!   cargo run --release -p physio-algo --example posterior_check
//!
//! Shipped emissions and `Params::SHIPPED.transition` throughout — the only thing that varies is the
//! decode RULE, so a difference here is the rule's and nothing else's. This is a size check, not a
//! card: no truth is read and no arm is selected. The three PSG cohorts plus `ours`.
//!
//! Also the two coherence checks worth having on real data rather than in a unit test: the mean
//! posterior of the path Viterbi returns, and that `decode_with_costs` at `Costs::UNIT` is the plain
//! marginal argmax on every night.

mod common;

use common::{dirs_of, read_accel, read_hr, read_meta, read_rr};

use physio_algo::sleep::markov_loss::Costs;
use physio_algo::sleep::posterior::{
    decode_with_costs, forward_backward, log_likelihood, posterior_marginal_decode,
};
use physio_algo::sleep::{decode_v2, emissions_v2, params::Params, prepare_v2, SleepInput, SleepStage};

const SETS: [&str; 4] = ["dreamt", "aauwss", "sleep-accel", "ours"];

struct Night {
    epochs: usize,
    differ: usize,
    path_post: f64,
    ll_per_epoch: f64,
    unit_matches: bool,
}

fn measure(input: &SleepInput, p: &Params) -> Option<Night> {
    let prep = prepare_v2(input, p);
    let em = emissions_v2(&prep, p);
    if em.is_empty() {
        return None;
    }
    let post = forward_backward(&em, |_| p.transition);
    let path = decode_v2(&em, &p.transition);
    let marg = posterior_marginal_decode(&post);
    let order = [SleepStage::Deep, SleepStage::Rem, SleepStage::Light, SleepStage::Wake];
    let col = |s: SleepStage| order.iter().position(|x| *x == s).expect("a stage");
    let mass: f64 = path.iter().enumerate().map(|(t, s)| post[t][col(*s)]).sum();
    Some(Night {
        epochs: em.len(),
        differ: path.iter().zip(&marg).filter(|(a, b)| a != b).count(),
        path_post: mass,
        ll_per_epoch: log_likelihood(&em, |_| p.transition),
        unit_matches: decode_with_costs(&post, &Costs::UNIT) == marg,
    })
}

fn main() {
    let p = &Params::SHIPPED;
    println!("posterior-marginal vs viterbi, shipped emissions and shipped transition\n");
    println!("{:<14} {:>6} {:>9} {:>8} {:>9} {:>11} {:>7}", "set", "nights", "epochs", "differ", "differ %", "mean path P", "UNIT ok");

    for set in SETS {
        let (mut nights, mut epochs, mut differ, mut mass, mut ll, mut unit_ok) = (0, 0usize, 0usize, 0.0f64, 0.0f64, true);
        let mut worst = (0.0f64, String::new());
        for d in dirs_of(set) {
            let Some((w0, w1, _)) = read_meta(&d) else { continue };
            let input = SleepInput {
                start: w0,
                end: w1,
                hr: read_hr(&d),
                rr: read_rr(&d),
                accel: read_accel(&d),
            };
            let Some(n) = measure(&input, p) else { continue };
            nights += 1;
            epochs += n.epochs;
            differ += n.differ;
            mass += n.path_post;
            ll += n.ll_per_epoch;
            unit_ok &= n.unit_matches;
            let frac = n.differ as f64 / n.epochs as f64;
            if frac > worst.0 {
                worst = (frac, d.file_name().unwrap_or_default().to_string_lossy().to_string());
            }
        }
        if nights == 0 {
            println!("{set:<14} {:>6}", 0);
            continue;
        }
        println!(
            "{set:<14} {nights:>6} {epochs:>9} {differ:>8} {:>8.3}% {:>11.4} {:>7}",
            100.0 * differ as f64 / epochs as f64,
            mass / epochs as f64,
            if unit_ok { "yes" } else { "NO" }
        );
        println!(
            "               mean log-likelihood per night {:.1}; worst night {:.2}% differ ({})",
            ll / nights as f64,
            100.0 * worst.0,
            worst.1
        );
    }
}
