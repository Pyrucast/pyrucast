//! `QUA4` — the 4-node quadrangle.

use super::{ElementKind, Facet, Interpolation};
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
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        out[0] = 0.25 * (1.0 - a) * (1.0 - b);
        out[1] = 0.25 * (1.0 + a) * (1.0 - b);
        out[2] = 0.25 * (1.0 + a) * (1.0 + b);
        out[3] = 0.25 * (1.0 - a) * (1.0 + b);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        out.copy_from_slice(&[
            -0.25 * (1.0 - b),
            -0.25 * (1.0 - a), // dN0
            0.25 * (1.0 - b),
            -0.25 * (1.0 + a), // dN1
            0.25 * (1.0 + b),
            0.25 * (1.0 + a), // dN2
            -0.25 * (1.0 + b),
            0.25 * (1.0 - a), // dN3
        ]);
    }
    /// 2×2 Gauss-Legendre tensor product, corners in CCW order.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let a = 1.0 / 3.0_f64.sqrt();
        (
            vec![
                -a, -a, //
                a, -a, //
                a, a, //
                -a, a,
            ],
            vec![1.0; 4],
        )
    }
}
