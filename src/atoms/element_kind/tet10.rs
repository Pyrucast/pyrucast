//! `TET10` — the 10-node quadratic tetrahedron.

use super::tet4::{contains_simplex, Tet4, EDGES};
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 10-node quadratic tetrahedron (Lagrange-2 `TET4`). Corners 0..3 as `TET4`,
/// then mid-edge nodes 4..9 on edges `(0,1)`, `(1,2)`, `(2,0)`, `(0,3)`,
/// `(1,3)`, `(2,3)`.
pub struct Tet10;

/// The four `TRI6` faces: the `TET4` corner faces, each completed with the
/// mid node of its three edges — `mid(0,1) = 4`, `mid(1,2) = 5`,
/// `mid(2,0) = 6`, `mid(0,3) = 7`, `mid(1,3) = 8`, `mid(2,3) = 9`.
const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[1, 2, 3, 5, 9, 8],
    },
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[0, 3, 2, 7, 9, 6],
    },
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[0, 1, 3, 4, 8, 7],
    },
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[0, 2, 1, 6, 5, 4],
    },
];

impl ElementKind for Tet10 {
    fn element_type(&self) -> ElementType {
        ElementType::TET10
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[0.5, 0.0, 0.0],
            &[0.5, 0.5, 0.0],
            &[0.0, 0.5, 0.0],
            &[0.0, 0.0, 0.5],
            &[0.5, 0.0, 0.5],
            &[0.0, 0.5, 0.5],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 3, 6, 5, 4, 7, 9, 8]
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
        Tet4.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Tet4.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_simplex(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
}
