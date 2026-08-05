//! `PYRA5` — the 5-node pyramid.

use super::{ElementKind, Facet};
use crate::atoms::ElementType;

/// 5-node pyramid: a square base and an apex. Reference: `ζ ∈ [0, 1]` with
/// `ξ, η ∈ [-(1-ζ), +(1-ζ)]`, so the square shrinks to a point at the apex.
/// Local order: base CCW seen from the apex (nodes 0..3 at `ζ = 0`), then the
/// apex.
///
/// This is the element that makes a hexahedron and a tetrahedron meet: its
/// square face matches a `HEX8` face and its four triangles match `TET4`
/// faces, so a hexahedral layer closes onto a tetrahedral core without a
/// hanging node.
pub struct Pyra5;

const EDGES: &[[usize; 2]] = &[
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
];

/// Corner faces, outward-oriented: the square base first, wound so its normal
/// points away from the apex, then the four side triangles.
pub(super) const CORNER_FACES: [&[usize]; 5] = [
    &[0, 3, 2, 1],
    &[0, 1, 4],
    &[1, 2, 4],
    &[2, 3, 4],
    &[3, 0, 4],
];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[0],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[1],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[2],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[3],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[4],
    },
];

impl ElementKind for Pyra5 {
    fn element_type(&self) -> ElementType {
        ElementType::PYRA5
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[-1.0, -1.0, 0.0],
            &[1.0, -1.0, 0.0],
            &[1.0, 1.0, 0.0],
            &[-1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 4]
    }

    fn corner_count(&self) -> usize {
        5
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    /// A quarter of the way up — the centroid of a pyramid, not the mean of
    /// its nodes.
    fn ref_centroid(&self) -> &'static [f64] {
        &[0.0, 0.0, 0.25]
    }

    /// Square base of side 2 tapering to a point: `∫₀¹ (2(1-ζ))² dζ = 4/3`.
    fn ref_measure(&self) -> f64 {
        4.0 / 3.0
    }

    /// The square cross-section shrinks as `ζ` climbs, so the bound on `ξ` and
    /// `η` is `1 - ζ` rather than a constant.
    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let half = 1.0 - c + tol;
        c >= -tol && c <= 1.0 + tol && a.abs() <= half && b.abs() <= half
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
}
