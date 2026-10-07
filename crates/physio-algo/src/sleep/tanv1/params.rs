//! tanv1 configuration. Minimal on purpose: U4+ add emission and transition fields.

use super::super::abstain;
use super::super::params::Params;

/// The base v2 recipe tanv1 runs on: SHIPPED plus `clamp_only_without_rr`, adopted 2026-09-30.
///
/// Evidence: `_r13/border-params-switches.txt` (dreamt paired kappa4 +.0133 AHEAD 2.58x, macro F1 AHEAD
/// 2.74x; aauwss BA AHEAD 2.07x; sleep-accel identical) and `_r13/u4b-card.txt` (no fitted arm beats it).
pub const BASE: Params = Params { clamp_only_without_rr: true, ..Params::SHIPPED };

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tanv1Cfg {
    pub base: Params,
    /// Coverage to keep for far-from-edge abstention; `None` refuses nothing.
    pub abstain: Option<f64>,
}

impl Tanv1Cfg {
    /// The control: v2 with SHIPPED params, no holes. Must equal `v2::stage` label for label.
    pub const NULL: Tanv1Cfg = Tanv1Cfg { base: Params::SHIPPED, abstain: None };
    /// What tanv1 v1.0 ships: [`BASE`] plus abstention at [`abstain::DEFAULT_COVERAGE`].
    pub const DEFAULT: Tanv1Cfg = Tanv1Cfg { base: BASE, abstain: Some(abstain::DEFAULT_COVERAGE) };
}
