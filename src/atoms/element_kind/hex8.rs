//! `HEX8` — the 8-node hexahedron.

use super::qua4::contains_cube;
use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 8-node hexahedron. Reference: `ξ, η, ζ ∈ [-1, +1]`. Local order: bottom
/// face CCW (nodes 0..3), then top face CCW (nodes 4..7) — the convention
/// [`crate::ops::mesh::extrude`] produces.
pub struct Hex8;

pub(super) const EDGES: &[[usize; 2]] = &[
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Corner faces, outward-oriented: bottom, top, then the four sides.
pub(super) const CORNER_FACES: [&[usize]; 6] = [
    &[0, 3, 2, 1],
    &[4, 5, 6, 7],
    &[0, 1, 5, 4],
    &[1, 2, 6, 5],
    &[2, 3, 7, 6],
    &[0, 4, 7, 3],
];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[0],
    },
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[1],
    },
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[2],
    },
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[3],
    },
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[4],
    },
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[5],
    },
];

impl ElementKind for Hex8 {
    fn element_type(&self) -> ElementType {
        ElementType::HEX8
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[-1.0, -1.0, -1.0],
            &[1.0, -1.0, -1.0],
            &[1.0, 1.0, -1.0],
            &[-1.0, 1.0, -1.0],
            &[-1.0, -1.0, 1.0],
            &[1.0, -1.0, 1.0],
            &[1.0, 1.0, 1.0],
            &[-1.0, 1.0, 1.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 4, 7, 6, 5]
    }

    fn corner_count(&self) -> usize {
        8
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        &[0.0, 0.0, 0.0]
    }

    fn ref_measure(&self) -> f64 {
        8.0
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_cube(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        for (i, n) in self.ref_nodes().iter().enumerate() {
            out[i] = 0.125 * (1.0 + n[0] * a) * (1.0 + n[1] * b) * (1.0 + n[2] * c);
        }
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        for (i, n) in self.ref_nodes().iter().enumerate() {
            let (p, q, r) = (n[0], n[1], n[2]);
            let (u, v, w) = (1.0 + p * a, 1.0 + q * b, 1.0 + r * c);
            out[3 * i] = 0.125 * p * v * w;
            out[3 * i + 1] = 0.125 * q * u * w;
            out[3 * i + 2] = 0.125 * r * u * v;
        }
    }
}
