//! `TET10` — the 10-node quadratic tetrahedron.

use super::tet4::{contains_simplex, Tet4, EDGES};
use super::{ElementKind, Facet, Interpolation};
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
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    /// Complete quadratic on the unit tetrahedron. `L0 = 1-ξ-η-ζ`, `L1 = ξ`,
    /// `L2 = η`, `L3 = ζ`.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let (l0, l1, l2, l3) = (1.0 - a - b - c, a, b, c);
        out.copy_from_slice(&[
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l0 * l1, // 4: (0,1)
            4.0 * l1 * l2, // 5: (1,2)
            4.0 * l2 * l0, // 6: (2,0)
            4.0 * l0 * l3, // 7: (0,3)
            4.0 * l1 * l3, // 8: (1,3)
            4.0 * l2 * l3, // 9: (2,3)
        ]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let (l0, l1, l2, l3) = (1.0 - a - b - c, a, b, c);
        // Gradients of the barycentric coordinates (each a length-3 row).
        let d = [
            [-1.0, -1.0, -1.0], // dL0
            [1.0, 0.0, 0.0],    // dL1
            [0.0, 1.0, 0.0],    // dL2
            [0.0, 0.0, 1.0],    // dL3
        ];
        let corner = |li: f64, dl: [f64; 3]| dl.map(|g| (4.0 * li - 1.0) * g);
        // 4 * (L_b dL_a + L_a dL_b)
        let edge = |la: f64, lb: f64, da: [f64; 3], db: [f64; 3]| {
            [0, 1, 2].map(|k| 4.0 * (lb * da[k] + la * db[k]))
        };
        let rows = [
            corner(l0, d[0]),
            corner(l1, d[1]),
            corner(l2, d[2]),
            corner(l3, d[3]),
            edge(l0, l1, d[0], d[1]),
            edge(l1, l2, d[1], d[2]),
            edge(l2, l0, d[2], d[0]),
            edge(l0, l3, d[0], d[3]),
            edge(l1, l3, d[1], d[3]),
            edge(l2, l3, d[2], d[3]),
        ];
        for (i, row) in rows.iter().enumerate() {
            out[3 * i..3 * i + 3].copy_from_slice(row);
        }
    }
}
