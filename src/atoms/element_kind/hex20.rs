//! `HEX20` — the 20-node serendipity hexahedron.

use super::hex8::{Hex8, EDGES};
use super::qua4::contains_cube;
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 20-node serendipity hexahedron (Lagrange-2 `HEX8`). Corners 0..7 as
/// `HEX8`, then mid-edge nodes 8..19: bottom `(0,1)`, `(1,2)`, `(2,3)`,
/// `(3,0)`, top `(4,5)`, `(5,6)`, `(6,7)`, `(7,4)`, vertical `(0,4)`,
/// `(1,5)`, `(2,6)`, `(3,7)`.
pub struct Hex20;

/// The six `QUA8` faces. Mid nodes follow the edge order: `mid(0,1) = 8` …
/// `mid(3,7) = 19`. The corner sequences are `HEX8`'s, so the two degrees
/// agree on adjacency.
pub(super) const FACE_NODES: [[usize; 8]; 6] = [
    [0, 3, 2, 1, 11, 10, 9, 8],
    [4, 5, 6, 7, 12, 13, 14, 15],
    [0, 1, 5, 4, 8, 17, 12, 16],
    [1, 2, 6, 5, 9, 18, 13, 17],
    [2, 3, 7, 6, 10, 19, 14, 18],
    [0, 4, 7, 3, 16, 15, 19, 11],
];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[0],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[1],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[2],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[3],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[4],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &FACE_NODES[5],
    },
];

/// The 20 boundary nodes, shared with `HEX27`.
pub(super) const REF_NODES: &[&[f64]] = &[
    &[-1.0, -1.0, -1.0],
    &[1.0, -1.0, -1.0],
    &[1.0, 1.0, -1.0],
    &[-1.0, 1.0, -1.0],
    &[-1.0, -1.0, 1.0],
    &[1.0, -1.0, 1.0],
    &[1.0, 1.0, 1.0],
    &[-1.0, 1.0, 1.0],
    &[0.0, -1.0, -1.0],
    &[1.0, 0.0, -1.0],
    &[0.0, 1.0, -1.0],
    &[-1.0, 0.0, -1.0],
    &[0.0, -1.0, 1.0],
    &[1.0, 0.0, 1.0],
    &[0.0, 1.0, 1.0],
    &[-1.0, 0.0, 1.0],
    &[-1.0, -1.0, 0.0],
    &[1.0, -1.0, 0.0],
    &[1.0, 1.0, 0.0],
    &[-1.0, 1.0, 0.0],
];

impl ElementKind for Hex20 {
    fn element_type(&self) -> ElementType {
        ElementType::HEX20
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        REF_NODES
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[
            0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17,
        ]
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
        Hex8.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Hex8.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_cube(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
}
