//! `SEG2` — the 2-node segment.

use super::{ElementKind, Interpolation};
use crate::atoms::ElementType;

/// 2-node segment. Reference: `ξ ∈ [-1, +1]`, node 0 at `ξ = -1`, node 1 at
/// `ξ = +1`.
pub struct Seg2;

/// Shared by `SEG2` and `SEG3`: the segment's ends carry no orientation of
/// their own, so `facets()` stays empty and the chaining logic in
/// [`crate::ops::mesh::orient`] handles them directly.
pub(super) const EDGES: &[[usize; 2]] = &[[0, 1]];

impl ElementKind for Seg2 {
    fn element_type(&self) -> ElementType {
        ElementType::SEG2
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[&[-1.0], &[1.0]]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[1, 0]
    }

    fn corner_count(&self) -> usize {
        2
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    fn ref_centroid(&self) -> &'static [f64] {
        &[0.0]
    }

    fn ref_measure(&self) -> f64 {
        2.0
    }

    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        xi[0] >= -1.0 - tol && xi[0] <= 1.0 + tol
    }

    fn clamp_ref(&self, xi: &mut [f64]) {
        xi[0] = xi[0].clamp(-1.0, 1.0);
    }
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let x = xi[0];
        out[0] = 0.5 * (1.0 - x);
        out[1] = 0.5 * (1.0 + x);
    }

    fn dshape_into(&self, _xi: &[f64], out: &mut [f64]) {
        out[0] = -0.5;
        out[1] = 0.5;
    }
    /// 2-point Gauss-Legendre on `[-1, 1]`.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let a = 1.0 / 3.0_f64.sqrt();
        (vec![-a, a], vec![1.0, 1.0])
    }

    fn vtk_code(&self) -> u8 {
        3 // VTK_LINE
    }

    fn gmsh_code(&self) -> u32 {
        1
    }

    fn quadratic(&self) -> Option<ElementType> {
        Some(ElementType::SEG3)
    }
}
