//! `TRI3` — the 3-node triangle.

use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 3-node triangle. Reference: the unit simplex `ξ, η ∈ [0, 1]`, `ξ + η ≤ 1`.
/// Local order (CCW): `(0, 0)`, `(1, 0)`, `(0, 1)`.
pub struct Tri3;

pub(super) const EDGES: &[[usize; 2]] = &[[0, 1], [1, 2], [2, 0]];

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
        nodes: &[2, 0],
    },
];

/// Clamp onto the unit simplex, shared with `TRI6`: make both coordinates
/// non-negative, then pull back onto the hypotenuse if `a + b > 1` — the
/// closest point on the line `a + b = 1`.
pub(super) fn clamp_simplex(xi: &mut [f64]) {
    xi[0] = xi[0].max(0.0);
    xi[1] = xi[1].max(0.0);
    let excess = xi[0] + xi[1] - 1.0;
    if excess > 0.0 {
        xi[0] -= excess / 2.0;
        xi[1] -= excess / 2.0;
        xi[0] = xi[0].max(0.0);
        xi[1] = xi[1].max(0.0);
        let s = xi[0] + xi[1];
        if s > 1.0 {
            xi[0] /= s;
            xi[1] /= s;
        }
    }
}

/// Membership in the unit triangle, shared with `TRI6`.
pub(super) fn contains_simplex(xi: &[f64], tol: f64) -> bool {
    let (a, b) = (xi[0], xi[1]);
    a >= -tol && b >= -tol && a + b <= 1.0 + tol
}

impl ElementKind for Tri3 {
    fn element_type(&self) -> ElementType {
        ElementType::TRI3
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1]
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
        &[1.0 / 3.0, 1.0 / 3.0]
    }

    fn ref_measure(&self) -> f64 {
        0.5
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_simplex(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        clamp_simplex(xi);
    }
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        out[0] = 1.0 - a - b;
        out[1] = a;
        out[2] = b;
    }

    fn dshape_into(&self, _xi: &[f64], out: &mut [f64]) {
        out.copy_from_slice(&[
            -1.0, -1.0, // dN0
            1.0, 0.0, // dN1
            0.0, 1.0, // dN2
        ]);
    }
}
