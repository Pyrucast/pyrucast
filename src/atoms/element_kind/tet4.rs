//! `TET4` — the 4-node tetrahedron.

use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 4-node tetrahedron. Reference: `ξ, η, ζ ∈ [0, 1]`, `ξ + η + ζ ≤ 1`. Local
/// order `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)` — face 0-1-2 CCW seen from
/// node 3.
pub struct Tet4;

pub(super) const EDGES: &[[usize; 2]] = &[[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::TRI3,
        nodes: &[1, 2, 3],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: &[0, 3, 2],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: &[0, 1, 3],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: &[0, 2, 1],
    },
];

/// Membership in the unit 3-simplex, shared with `TET10`.
pub(super) fn contains_simplex(xi: &[f64], tol: f64) -> bool {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    a >= -tol && b >= -tol && c >= -tol && a + b + c <= 1.0 + tol
}

impl ElementKind for Tet4 {
    fn element_type(&self) -> ElementType {
        ElementType::TET4
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 3]
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
        &[0.25, 0.25, 0.25]
    }

    fn ref_measure(&self) -> f64 {
        1.0 / 6.0
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_simplex(xi, tol)
    }

    /// A volume cell is never projected onto, so there is nothing to clamp.
    fn clamp_ref(&self, _xi: &mut [f64]) {}
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        out[0] = 1.0 - a - b - c;
        out[1] = a;
        out[2] = b;
        out[3] = c;
    }

    fn dshape_into(&self, _xi: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[
            -1.0, -1.0, -1.0, // dN0
            1.0, 0.0, 0.0, // dN1
            0.0, 1.0, 0.0, // dN2
            0.0, 0.0, 1.0, // dN3
        ]);
    }
}
