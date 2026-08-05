//! `POI1` — a single node.

use super::ElementKind;
use crate::atoms::ElementType;

/// 1 node. A `POI1` submesh is effectively a list of nodes: it carries no
/// reference frame, so it has no domain to be inside of and no facet.
pub struct Poi1;

impl ElementKind for Poi1 {
    fn element_type(&self) -> ElementType {
        ElementType::POI1
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[&[]]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0]
    }

    fn corner_count(&self) -> usize {
        1
    }

    fn ref_centroid(&self) -> &'static [f64] {
        &[]
    }

    /// A point has no extent. Reported as `0` rather than left undefined so
    /// generic code can compare measures without special-casing.
    fn ref_measure(&self) -> f64 {
        0.0
    }

    /// No reference frame, so no `ξ` is ever inside.
    fn contains_ref(&self, _xi: &[f64], _tol: f64) -> bool {
        false
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
}
