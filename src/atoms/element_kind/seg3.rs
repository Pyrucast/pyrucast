//! `SEG3` — the 3-node quadratic segment.

use super::seg2::{Seg2, EDGES};
use super::ElementKind;
use crate::atoms::ElementType;

/// 3-node quadratic segment (Lagrange-2 `SEG2`). Corners 0, 1 at `ξ = ∓1`,
/// mid node 2 at `ξ = 0`.
pub struct Seg3;

impl ElementKind for Seg3 {
    fn element_type(&self) -> ElementType {
        ElementType::SEG3
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[&[-1.0], &[1.0], &[0.0]]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[1, 0, 2]
    }

    fn corner_count(&self) -> usize {
        2
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        Seg2.ref_centroid()
    }

    fn ref_measure(&self) -> f64 {
        Seg2.ref_measure()
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        Seg2.contains_ref(xi, tol)
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        Seg2.clamp_ref(xi);
    }
}
