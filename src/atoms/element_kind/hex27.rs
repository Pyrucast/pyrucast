//! `HEX27` — the 27-node tri-quadratic hexahedron.

use super::hex8::{Hex8, EDGES};
use super::qua4::contains_cube;
use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 27-node tri-quadratic hexahedron (full Lagrange-2 `HEX8`). Corners 0..7 and
/// mid-edge nodes 8..19 as `HEX20`, then 6 face-centre nodes 20..25 (faces
/// `x-`, `x+`, `y-`, `y+`, `z-`, `z+`), then a body-centre node 26. The
/// complete `Q2` tensor product on the hex.
pub struct Hex27;

/// The six `QUA9` faces: `HEX20`'s corner-and-edge sequences, each closed by
/// its own centre node — `x- = 20`, `x+ = 21`, `y- = 22`, `y+ = 23`,
/// `z- = 24`, `z+ = 25`.
const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[0, 3, 2, 1, 11, 10, 9, 8, 24],
    },
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[4, 5, 6, 7, 12, 13, 14, 15, 25],
    },
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[0, 1, 5, 4, 8, 17, 12, 16, 22],
    },
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[1, 2, 6, 5, 9, 18, 13, 17, 21],
    },
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[2, 3, 7, 6, 10, 19, 14, 18, 23],
    },
    Facet {
        element_type: ElementType::QUA9,
        nodes: &[0, 4, 7, 3, 16, 15, 19, 11, 20],
    },
];

impl ElementKind for Hex27 {
    fn element_type(&self) -> ElementType {
        ElementType::HEX27
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
            &[-1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, -1.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, -1.0],
            &[0.0, 0.0, 1.0],
            &[0.0, 0.0, 0.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[
            0, 3, 2, 1, 4, 7, 6, 5, 11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17, 22, 23, 20, 21,
            24, 25, 26,
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
