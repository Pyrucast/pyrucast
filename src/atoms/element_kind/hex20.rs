//! `HEX20` — the 20-node serendipity hexahedron.

use super::hex8::{Hex8, EDGES};
use super::qua4::contains_cube;
use super::quadrature::gauss3_1d;
use super::{ElementKind, Facet, Interpolation};
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
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    /// Serendipity on `[-1, 1]³`: a node's formula is read off its own
    /// reference coordinates — no zero component means a corner, one zero
    /// component names the axis the mid-edge node sits on.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        for (i, n) in REF_NODES.iter().enumerate() {
            let (p, q, r) = (n[0], n[1], n[2]);
            out[i] = if p != 0.0 && q != 0.0 && r != 0.0 {
                0.125
                    * (1.0 + p * a)
                    * (1.0 + q * b)
                    * (1.0 + r * c)
                    * (p * a + q * b + r * c - 2.0)
            } else if p == 0.0 {
                0.25 * (1.0 - a * a) * (1.0 + q * b) * (1.0 + r * c)
            } else if q == 0.0 {
                0.25 * (1.0 + p * a) * (1.0 - b * b) * (1.0 + r * c)
            } else {
                0.25 * (1.0 + p * a) * (1.0 + q * b) * (1.0 - c * c)
            };
        }
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        for (i, n) in REF_NODES.iter().enumerate() {
            let (p, q, r) = (n[0], n[1], n[2]);
            let row = &mut out[3 * i..3 * i + 3];
            if p != 0.0 && q != 0.0 && r != 0.0 {
                let (u, v, w) = (1.0 + p * a, 1.0 + q * b, 1.0 + r * c);
                row[0] = 0.125 * p * v * w * (2.0 * p * a + q * b + r * c - 1.0);
                row[1] = 0.125 * q * u * w * (p * a + 2.0 * q * b + r * c - 1.0);
                row[2] = 0.125 * r * u * v * (p * a + q * b + 2.0 * r * c - 1.0);
            } else if p == 0.0 {
                let (vy, wz) = (1.0 + q * b, 1.0 + r * c);
                row[0] = -0.5 * a * vy * wz;
                row[1] = 0.25 * (1.0 - a * a) * q * wz;
                row[2] = 0.25 * (1.0 - a * a) * vy * r;
            } else if q == 0.0 {
                let (ux, wz) = (1.0 + p * a, 1.0 + r * c);
                row[0] = 0.25 * p * (1.0 - b * b) * wz;
                row[1] = -0.5 * b * ux * wz;
                row[2] = 0.25 * ux * (1.0 - b * b) * r;
            } else {
                let (ux, vy) = (1.0 + p * a, 1.0 + q * b);
                row[0] = 0.25 * p * vy * (1.0 - c * c);
                row[1] = 0.25 * ux * q * (1.0 - c * c);
                row[2] = -0.5 * c * ux * vy;
            }
        }
    }
    /// 3×3×3 tensor product of the 3-point rule. Shared with `HEX27`.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let (p, w) = gauss3_1d();
        let mut xi = Vec::with_capacity(27 * 3);
        let mut wt = Vec::with_capacity(27);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    xi.push(p[i]);
                    xi.push(p[j]);
                    xi.push(p[k]);
                    wt.push(w[i] * w[j] * w[k]);
                }
            }
        }
        (xi, wt)
    }

    fn vtk_code(&self) -> u8 {
        25 // VTK_QUADRATIC_HEXAHEDRON
    }

    fn gmsh_code(&self) -> u32 {
        17
    }

    fn gmsh_permutation(&self) -> Option<&'static [usize]> {
        Some(&[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
        ])
    }

    fn linear_parent(&self) -> Option<ElementType> {
        Some(ElementType::HEX8)
    }
}
