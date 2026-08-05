//! `QUA9` — the 9-node biquadratic quadrangle.

use super::qua4::{clamp_cube, contains_cube, Qua4, EDGES};
use super::qua8::Qua8;
use super::qua8::FACETS;
use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 9-node biquadratic quadrangle (full Lagrange-2 `QUA4`). Corners 0..3,
/// mid-edge nodes 4..7, then a **centre** node 8 at `(0, 0)`. Unlike the
/// serendipity `QUA8` it carries the central node — the complete `Q2` tensor
/// product. Its boundary, and therefore its facets, are the same as `QUA8`'s.
pub struct Qua9;

impl ElementKind for Qua9 {
    fn element_type(&self) -> ElementType {
        ElementType::QUA9
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[-1.0, -1.0],
            &[1.0, -1.0],
            &[1.0, 1.0],
            &[-1.0, 1.0],
            &[0.0, -1.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[-1.0, 0.0],
            &[0.0, 0.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 7, 6, 5, 4, 8]
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

    /// The full `Q2` tensor product: each shape function is a product of two
    /// 1-D quadratic Lagrange factors.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (lx, ly) = (lag1d(xi[0]), lag1d(xi[1]));
        for (n, &(i, j)) in NODES.iter().enumerate() {
            out[n] = lx[i] * ly[j];
        }
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (lx, ly) = (lag1d(xi[0]), lag1d(xi[1]));
        let (dlx, dly) = (dlag1d(xi[0]), dlag1d(xi[1]));
        for (n, &(i, j)) in NODES.iter().enumerate() {
            out[2 * n] = dlx[i] * ly[j];
            out[2 * n + 1] = lx[i] * dly[j];
        }
    }

    /// The same 3×3 rule as `QUA8` — the extra centre node raises the
    /// interpolation's completeness, not the degree to integrate.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        Qua8.gauss()
    }

    fn vtk_code(&self) -> u8 {
        28 // VTK_BIQUADRATIC_QUAD
    }

    fn gmsh_code(&self) -> u32 {
        10
    }

    fn linear_parent(&self) -> Option<ElementType> {
        Some(ElementType::QUA4)
    }
}

/// 1-D quadratic Lagrange basis on `[-1, 1]` at `t`, nodes `-1, 0, +1`.
/// Shared with `HEX27`, the other full `Q2` element.
pub(super) fn lag1d(t: f64) -> [f64; 3] {
    [0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)]
}

/// Its derivative.
pub(super) fn dlag1d(t: f64) -> [f64; 3] {
    [t - 0.5, -2.0 * t, t + 0.5]
}

/// (ξ, η) position of each node as an index into the 1-D basis
/// `[node@-1, node@0, node@+1]`. Corners 0..3, mid-edges 4..7, centre 8.
const NODES: [(usize, usize); 9] = [
    (0, 0),
    (2, 0),
    (2, 2),
    (0, 2), // corners
    (1, 0),
    (2, 1),
    (1, 2),
    (0, 1), // mid-edges
    (1, 1), // centre
];
