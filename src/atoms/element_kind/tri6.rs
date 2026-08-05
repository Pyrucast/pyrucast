//! `TRI6` — the 6-node quadratic triangle.

use super::tri3::{clamp_simplex, contains_simplex, Tri3, EDGES};
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 6-node quadratic triangle (Lagrange-2 `TRI3`). Corners 0..2 as `TRI3`,
/// then mid-edge nodes 3, 4, 5 on edges `(0,1)`, `(1,2)`, `(2,0)`.
pub struct Tri6;

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[0, 1, 3],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[1, 2, 4],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[2, 0, 5],
    },
];

impl ElementKind for Tri6 {
    fn element_type(&self) -> ElementType {
        ElementType::TRI6
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[0.5, 0.0],
            &[0.5, 0.5],
            &[0.0, 0.5],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 5, 4, 3]
    }

    fn corner_count(&self) -> usize {
        3
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        Tri3.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Tri3.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_simplex(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        clamp_simplex(xi);
    }
}
