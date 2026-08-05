//! `SEG3` — the 3-node quadratic segment.

use super::quadrature::gauss3_1d;
use super::seg2::{Seg2, EDGES};
use super::{ElementKind, Interpolation};
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
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange2)
    }

    /// Corners at `∓1`, mid node at `0`.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let x = xi[0];
        out.copy_from_slice(&[0.5 * x * (x - 1.0), 0.5 * x * (x + 1.0), 1.0 - x * x]);
    }

    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let x = xi[0];
        out.copy_from_slice(&[x - 0.5, x + 0.5, -2.0 * x]);
    }
    /// 3-point Gauss-Legendre on `[-1, 1]` (exact to degree 5).
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let (p, w) = gauss3_1d();
        (p.to_vec(), w.to_vec())
    }

    fn vtk_code(&self) -> u8 {
        21 // VTK_QUADRATIC_EDGE
    }

    fn gmsh_code(&self) -> u32 {
        8
    }

    fn linear_parent(&self) -> Option<ElementType> {
        Some(ElementType::SEG2)
    }
}
