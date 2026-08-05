//! `QUA8` — the 8-node serendipity quadrangle.

use super::qua4::{clamp_cube, contains_cube, Qua4, EDGES};
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 8-node serendipity quadrangle (Lagrange-2 `QUA4`). Corners 0..3 as `QUA4`,
/// then mid-edge nodes 4..7 on edges `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)`.
/// Serendipity: edge nodes only, no centre.
pub struct Qua8;

/// Shared with `QUA9`, whose extra centre node changes nothing on the
/// boundary.
pub(super) const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[0, 1, 4],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[1, 2, 5],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[2, 3, 6],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[3, 0, 7],
    },
];

/// The eight boundary nodes, shared with `QUA9`.
pub(super) const REF_NODES: &[&[f64]] = &[
    &[-1.0, -1.0],
    &[1.0, -1.0],
    &[1.0, 1.0],
    &[-1.0, 1.0],
    &[0.0, -1.0],
    &[1.0, 0.0],
    &[0.0, 1.0],
    &[-1.0, 0.0],
];

impl ElementKind for Qua8 {
    fn element_type(&self) -> ElementType {
        ElementType::QUA8
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        REF_NODES
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 7, 6, 5, 4]
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
        Qua4.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Qua4.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_cube(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        clamp_cube(xi);
    }
}
