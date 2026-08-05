//! `QUA4` — the 4-node quadrangle.

use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 4-node quadrangle. Reference: `ξ, η ∈ [-1, +1]`. Local order (CCW):
/// `(-1, -1)`, `(1, -1)`, `(1, 1)`, `(-1, 1)`.
pub struct Qua4;

pub(super) const EDGES: &[[usize; 2]] = &[[0, 1], [1, 2], [2, 3], [3, 0]];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::SEG2,
        nodes: &[0, 1],
    },
    Facet {
        element_type: ElementType::SEG2,
        nodes: &[1, 2],
    },
    Facet {
        element_type: ElementType::SEG2,
        nodes: &[2, 3],
    },
    Facet {
        element_type: ElementType::SEG2,
        nodes: &[3, 0],
    },
];

/// Membership in the reference cube `[-1, 1]^d`, shared by `QUA*` and `HEX*`.
pub(super) fn contains_cube(xi: &[f64], tol: f64) -> bool {
    xi.iter().all(|&v| v >= -1.0 - tol && v <= 1.0 + tol)
}

/// Clamp into the reference cube `[-1, 1]^d`, shared by `QUA*` and `HEX*`.
pub(super) fn clamp_cube(xi: &mut [f64]) {
    for v in xi.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
}

impl ElementKind for Qua4 {
    fn element_type(&self) -> ElementType {
        ElementType::QUA4
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[&[-1.0, -1.0], &[1.0, -1.0], &[1.0, 1.0], &[-1.0, 1.0]]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1]
    }

    fn corner_count(&self) -> usize {
        4
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        &[0.0, 0.0]
    }

    fn ref_measure(&self) -> f64 {
        4.0
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_cube(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        clamp_cube(xi);
    }
}
