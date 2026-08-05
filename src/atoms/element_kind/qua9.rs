//! `QUA9` — the 9-node biquadratic quadrangle.

use super::qua4::{clamp_cube, contains_cube, Qua4, EDGES};
use super::qua8::FACETS;
use super::{ElementKind, Facet};
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
}
