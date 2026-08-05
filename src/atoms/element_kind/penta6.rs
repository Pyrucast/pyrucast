//! `PENTA6` — the 6-node prism (pentahedron).

use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 6-node prism, the extrusion of a `TRI3` along `ζ`. Reference:
/// `ξ, η ∈ [0, 1]` with `ξ + η ≤ 1`, `ζ ∈ [0, 1]`. Local order: bottom
/// triangle CCW (nodes 0..2 at `ζ = 0`), then top triangle CCW (nodes 3..5).
pub struct Penta6;

pub(super) const EDGES: &[[usize; 2]] = &[
    [0, 1],
    [1, 2],
    [2, 0],
    [3, 4],
    [4, 5],
    [5, 3],
    [0, 3],
    [1, 4],
    [2, 5],
];

/// Corner faces, outward-oriented: the two triangular caps then the three
/// quadrilateral sides.
pub(super) const CORNER_FACES: [&[usize]; 5] = [
    &[0, 2, 1],
    &[3, 4, 5],
    &[0, 1, 4, 3],
    &[1, 2, 5, 4],
    &[2, 0, 3, 5],
];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[0],
    },
    Facet {
        element_type: ElementType::TRI3,
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
];

/// Membership in the reference prism, shared with `PENTA15`.
pub(super) fn contains_prism(xi: &[f64], tol: f64) -> bool {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    a >= -tol && b >= -tol && a + b <= 1.0 + tol && c >= -tol && c <= 1.0 + tol
}

impl ElementKind for Penta6 {
    fn element_type(&self) -> ElementType {
        ElementType::PENTA6
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 3, 5, 4]
    }

    fn corner_count(&self) -> usize {
        6
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        &[1.0 / 3.0, 1.0 / 3.0, 0.5]
    }

    fn ref_measure(&self) -> f64 {
        0.5
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_prism(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    /// Prism = `TRI3` (barycentric ξ, η) ⊗ `SEG2` (linear ζ ∈ [0, 1]).
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        out.copy_from_slice(&[
            l1 * (1.0 - c),
            l2 * (1.0 - c),
            l3 * (1.0 - c),
            l1 * c,
            l2 * c,
            l3 * c,
        ]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        out.copy_from_slice(&[
            -(1.0 - c),
            -(1.0 - c),
            -l1, // dN0
            1.0 - c,
            0.0,
            -l2, // dN1
            0.0,
            1.0 - c,
            -l3, // dN2
            -c,
            -c,
            l1, // dN3
            c,
            0.0,
            l2, // dN4
            0.0,
            c,
            l3, // dN5
        ]);
    }
    /// Tensor product of the 3-point `TRI3` rule (weights 1/6) with the
    /// 2-point Gauss rule mapped to `ζ ∈ [0, 1]` (weights 1/2).
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let g = 0.5 / 3.0_f64.sqrt();
        let tri = [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)];
        let zeta = [0.5 - g, 0.5 + g];
        let mut xi = Vec::with_capacity(6 * 3);
        let mut w = Vec::with_capacity(6);
        for &(a, b) in &tri {
            for &z in &zeta {
                xi.push(a);
                xi.push(b);
                xi.push(z);
                w.push((1.0 / 6.0) * 0.5);
            }
        }
        (xi, w)
    }
}
