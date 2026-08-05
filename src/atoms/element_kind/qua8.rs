//! `QUA8` — the 8-node serendipity quadrangle.

use super::qua4::{clamp_cube, contains_cube, Qua4, EDGES};
use super::quadrature::gauss3_1d;
use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 8-node serendipity quadrangle (Lagrange-2 `QUA4`). Corners 0..3 as `QUA4`,
/// then mid-edge nodes 4..7 on edges `(0,1)`, `(1,2)`, `(2,3)`, `(3,0)`.
/// Serendipity: edge nodes only, no centre.
pub struct Qua8;

/// Signs `(ξ_i, η_i)` of the four corners, counter-clockwise.
pub(super) const CORNER_SIGNS: [(f64, f64); 4] =
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// Shared with `QUA9`, whose extra centre node changes nothing on the
/// boundary.
pub(super) const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[0, 1, 4],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[1, 2, 5],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[2, 3, 6],
    },
    Facet {
        element_type: ElementType::SEG3,
        nodes: &[3, 0, 7],
    },
];

/// The eight boundary nodes, shared with `QUA9`.
pub(super) const REF_NODES: &[&[f64]] = &[
    &[-1.0, -1.0],
    &[1.0, -1.0],
    &[1.0, 1.0],
    &[-1.0, 1.0],
    &[0.0, -1.0],
    &[1.0, 0.0],
    &[0.0, 1.0],
    &[-1.0, 0.0],
];

impl ElementKind for Qua8 {
    fn element_type(&self) -> ElementType {
        ElementType::QUA8
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        REF_NODES
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 7, 6, 5, 4]
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
        Qua4.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Qua4.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        contains_cube(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        clamp_cube(xi);
    }
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        let corner = |xi_i: f64, eta_i: f64| {
            0.25 * (1.0 + xi_i * a) * (1.0 + eta_i * b) * (xi_i * a + eta_i * b - 1.0)
        };
        out.copy_from_slice(&[
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
            0.5 * (1.0 - a * a) * (1.0 - b), // 4: (0,-1)
            0.5 * (1.0 + a) * (1.0 - b * b), // 5: (1,0)
            0.5 * (1.0 - a * a) * (1.0 + b), // 6: (0,1)
            0.5 * (1.0 - a) * (1.0 - b * b), // 7: (-1,0)
        ]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b) = (xi[0], xi[1]);
        for (i, &(xi_i, eta_i)) in CORNER_SIGNS.iter().enumerate() {
            let v = 1.0 + eta_i * b;
            let u = 1.0 + xi_i * a;
            out[2 * i] = 0.25 * xi_i * v * (2.0 * xi_i * a + eta_i * b);
            out[2 * i + 1] = 0.25 * eta_i * u * (2.0 * eta_i * b + xi_i * a);
        }
        // Mid-edge nodes 4..7.
        out[8..].copy_from_slice(&[
            -a * (1.0 - b),
            -0.5 * (1.0 - a * a), // 4
            0.5 * (1.0 - b * b),
            -b * (1.0 + a), // 5
            -a * (1.0 + b),
            0.5 * (1.0 - a * a), // 6
            -0.5 * (1.0 - b * b),
            -b * (1.0 - a), // 7
        ]);
    }
    /// 3×3 tensor product of the 3-point rule (exact to degree 5 per
    /// direction). Shared with `QUA9`.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let (p, w) = gauss3_1d();
        let mut xi = Vec::with_capacity(9 * 2);
        let mut wt = Vec::with_capacity(9);
        for j in 0..3 {
            for i in 0..3 {
                xi.push(p[i]);
                xi.push(p[j]);
                wt.push(w[i] * w[j]);
            }
        }
        (xi, wt)
    }
}
