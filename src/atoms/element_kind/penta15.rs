//! `PENTA15` — the 15-node serendipity prism.

use super::penta6::{contains_prism, Penta6, EDGES};
use super::quadrature::{gauss3_1d, tri6_gauss};
use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 15-node serendipity prism (Lagrange-2 `PENTA6`). Corners 0..5 as `PENTA6`,
/// then mid-edge nodes 6..14: bottom triangle `(0,1)`, `(1,2)`, `(2,0)`, top
/// triangle `(3,4)`, `(4,5)`, `(5,3)`, vertical `(0,3)`, `(1,4)`, `(2,5)`.
pub struct Penta15;

/// Two `TRI6` caps then three `QUA8` sides. Mid nodes follow the edge order:
/// `mid(0,1) = 6`, `mid(1,2) = 7`, `mid(2,0) = 8`, `mid(3,4) = 9`,
/// `mid(4,5) = 10`, `mid(5,3) = 11`, `mid(0,3) = 12`, `mid(1,4) = 13`,
/// `mid(2,5) = 14`.
const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[0, 2, 1, 8, 7, 6],
    },
    Facet {
        element_type: ElementType::TRI6,
        nodes: &[3, 4, 5, 9, 10, 11],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[0, 1, 4, 3, 6, 13, 9, 12],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[1, 2, 5, 4, 7, 14, 10, 13],
    },
    Facet {
        element_type: ElementType::QUA8,
        nodes: &[2, 0, 3, 5, 8, 12, 11, 14],
    },
];

impl ElementKind for Penta15 {
    fn element_type(&self) -> ElementType {
        ElementType::PENTA15
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[0.5, 0.0, 0.0],
            &[0.5, 0.5, 0.0],
            &[0.0, 0.5, 0.0],
            &[0.5, 0.0, 1.0],
            &[0.5, 0.5, 1.0],
            &[0.0, 0.5, 1.0],
            &[0.0, 0.0, 0.5],
            &[1.0, 0.0, 0.5],
            &[0.0, 1.0, 0.5],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 2, 1, 3, 5, 4, 8, 7, 6, 11, 10, 9, 12, 14, 13]
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
        Penta6.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Penta6.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_prism(xi, tol)
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    /// Serendipity prism: triangle `(L1, L2, L3)` × `ζ ∈ [0, 1]`.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, t) = (xi[0], xi[1], xi[2]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        let bot = |li: f64| li * (2.0 * li - 1.0) * (1.0 - t) - 2.0 * li * t * (1.0 - t);
        let top = |li: f64| li * (2.0 * li - 1.0) * t - 2.0 * li * t * (1.0 - t);
        out.copy_from_slice(&[
            bot(l1),
            bot(l2),
            bot(l3),
            top(l1),
            top(l2),
            top(l3),
            4.0 * l1 * l2 * (1.0 - t), // 6: bottom (0,1)
            4.0 * l2 * l3 * (1.0 - t), // 7: bottom (1,2)
            4.0 * l3 * l1 * (1.0 - t), // 8: bottom (2,0)
            4.0 * l1 * l2 * t,         // 9: top (3,4)
            4.0 * l2 * l3 * t,         // 10: top (4,5)
            4.0 * l3 * l1 * t,         // 11: top (5,3)
            4.0 * l1 * t * (1.0 - t),  // 12: vertical (0,3)
            4.0 * l2 * t * (1.0 - t),  // 13: vertical (1,4)
            4.0 * l3 * t * (1.0 - t),  // 14: vertical (2,5)
        ]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, t) = (xi[0], xi[1], xi[2]);
        let (l1, l2, l3) = (1.0 - a - b, a, b);
        // (a, b) gradients of the triangle coordinates.
        let dl = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]; // dL1, dL2, dL3
        let ls = [l1, l2, l3];

        // Corner node on triangle vertex `i`, `top` selects the ζ = 1 face.
        let corner = |i: usize, top: bool| {
            let li = ls[i];
            let f = li * (2.0 * li - 1.0); // f(Li)
            let coef = if top {
                (4.0 * li - 1.0) * t - 2.0 * (t - t * t)
            } else {
                (4.0 * li - 1.0) * (1.0 - t) - 2.0 * (t - t * t)
            };
            let dc = if top {
                f - 2.0 * li * (1.0 - 2.0 * t)
            } else {
                -f - 2.0 * li * (1.0 - 2.0 * t)
            };
            [coef * dl[i][0], coef * dl[i][1], dc]
        };
        // Mid-edge triangle node between vertices `i`,`j`, `top` selects ζ = 1.
        let tri_edge = |i: usize, j: usize, top: bool| {
            let (la, lb) = (ls[i], ls[j]);
            let zfac = if top { t } else { 1.0 - t };
            let da = 4.0 * zfac * (la * dl[j][0] + lb * dl[i][0]);
            let db = 4.0 * zfac * (la * dl[j][1] + lb * dl[i][1]);
            let dc = if top { 4.0 * la * lb } else { -4.0 * la * lb };
            [da, db, dc]
        };
        // Vertical mid-edge node on triangle vertex `i` (ζ = 1/2).
        let vertical = |i: usize| {
            let li = ls[i];
            [
                4.0 * (t - t * t) * dl[i][0],
                4.0 * (t - t * t) * dl[i][1],
                4.0 * li * (1.0 - 2.0 * t),
            ]
        };
        let rows = [
            corner(0, false),
            corner(1, false),
            corner(2, false),
            corner(0, true),
            corner(1, true),
            corner(2, true),
            tri_edge(0, 1, false),
            tri_edge(1, 2, false),
            tri_edge(2, 0, false),
            tri_edge(0, 1, true),
            tri_edge(1, 2, true),
            tri_edge(2, 0, true),
            vertical(0),
            vertical(1),
            vertical(2),
        ];
        for (i, row) in rows.iter().enumerate() {
            out[3 * i..3 * i + 3].copy_from_slice(row);
        }
    }
    /// Tensor product of the 6-point `TRI6` rule with the 3-point Gauss rule
    /// mapped to `ζ ∈ [0, 1]` (weights halved).
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let (tri_xi, tri_w) = tri6_gauss();
        let (gp, gw) = gauss3_1d();
        let mut xi = Vec::with_capacity(18 * 3);
        let mut w = Vec::with_capacity(18);
        for tg in 0..tri_w.len() {
            for k in 0..3 {
                xi.push(tri_xi[2 * tg]);
                xi.push(tri_xi[2 * tg + 1]);
                xi.push(0.5 + 0.5 * gp[k]);
                w.push(tri_w[tg] * 0.5 * gw[k]);
            }
        }
        (xi, w)
    }

    fn vtk_code(&self) -> u8 {
        26 // VTK_QUADRATIC_WEDGE
    }

    fn gmsh_code(&self) -> u32 {
        18
    }

    fn gmsh_permutation(&self) -> Option<&'static [usize]> {
        Some(&[0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11])
    }

    fn linear_parent(&self) -> Option<ElementType> {
        Some(ElementType::PENTA6)
    }
}
