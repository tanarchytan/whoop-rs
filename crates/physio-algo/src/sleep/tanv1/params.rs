//! tanv1 configuration: the v2 base recipe and the abstention coverage.

use super::super::abstain;
use super::super::params::Params;

/// The base v2 recipe tanv1 runs on: SHIPPED plus `clamp_only_without_rr`, which lets the awake cardiac
/// terms speak while still whenever the epoch carries R-R.
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
