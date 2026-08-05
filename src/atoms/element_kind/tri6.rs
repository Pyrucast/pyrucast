//! `TRI6` — the 6-node quadratic triangle.

use super::tri3::{clamp_simplex, contains_simplex, Tri3, EDGES};
use super::{ElementKind, Facet, Interpolation};
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
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    /// Complete quadratic on the unit triangle. `L1 = 1-ξ-η`, `L2 = ξ`,
    /// `L3 = η`; corners `L_i(2L_i-1)`, mid-edges `4 L_a L_b`.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        out.copy_from_slice(&[
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        // dL1 = (-1,-1), dL2 = (1,0), dL3 = (0,1).
        out.copy_from_slice(&[
            -(4.0 * l1 - 1.0),
            -(4.0 * l1 - 1.0), // dN0
            4.0 * l2 - 1.0,
            0.0, // dN1
            0.0,
            4.0 * l3 - 1.0, // dN2
            4.0 * (l1 - l2),
            -4.0 * l2, // dN3 = 4(L2 dL1 + L1 dL2)
            4.0 * l3,
            4.0 * l2, // dN4 = 4(L3 dL2 + L2 dL3)
            -4.0 * l3,
            4.0 * (l1 - l3), // dN5 = 4(L1 dL3 + L3 dL1)
        ]);
    }
}
