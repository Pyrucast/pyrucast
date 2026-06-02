//! Finite-element interpolations on the reference element.
//!
//! An [`Interpolation`] is the **mathematical recipe** that builds the
//! shape functions `N_i(ξ)` and their reference derivatives
//! `∂N_i/∂ξ_j` for a given [`ElementType`]. It is independent of any
//! particular cell or coordinate set: every evaluation lives in the
//! reference frame of the element type (see [`crate::containers::mesh::element_type`]).
//!
//! Adding a new interpolation means:
//! - adding a variant to [`Interpolation`];
//! - extending [`Interpolation::is_compatible_with`], [`Interpolation::shape`]
//!   and [`Interpolation::dshape_dxi`] for every supported `ElementType`.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::Interpolation;
//!
//! // Lagrange-1 shape functions of a TRI3 at the centroid (1/3, 1/3).
//! let n = Interpolation::Lagrange1
//!     .shape(ElementType::TRI3, &[1.0 / 3.0, 1.0 / 3.0])
//!     .unwrap();
//! assert_eq!(n.len(), 3);
//! let s: f64 = n.iter().sum();
//! assert!((s - 1.0).abs() < 1e-12);  // partition of unity
//! ```

use crate::containers::mesh::ElementType;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported finite-element interpolations.
///
/// Names mix cast3m and standard FE terminology. See the module-level
/// documentation for the conventions on reference frames and node order.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    /// Linear Lagrange (P1 / Q1). One shape function per geometric node,
    /// equal to 1 at "its" node and 0 at the others.
    Lagrange1,
}

impl Interpolation {
    /// Whether this interpolation is defined for `element_type`.
    ///
    /// `POI1` is **always rejected**: a point element has no reference
    /// frame, hence no shape functions in the usual sense.
    pub fn is_compatible_with(self, element_type: ElementType) -> bool {
        match (self, element_type) {
            (_, ElementType::POI1) => false,
            (Self::Lagrange1, _) => true,
        }
    }

    /// Short name (cast3m-style).
    pub fn name(self) -> &'static str {
        match self {
            Self::Lagrange1 => "LAGRANGE1",
        }
    }

    /// Parse from a short name (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "LAGRANGE1" | "LAG1" => Some(Self::Lagrange1),
            _ => None,
        }
    }

    /// Evaluate the shape functions `N_i(ξ)` at the reference point `xi`.
    ///
    /// Returns a flat `Vec<f64>` of length `element_type.nodes_per_cell()`
    /// ordered like the cell's nodes. `xi` must have length
    /// `element_type.topological_dim()`.
    ///
    /// # Errors
    ///
    /// - `xi` has the wrong length;
    /// - the `(self, element_type)` pair is not supported (`POI1`, …).
    pub fn shape(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check_compat(element_type)?;
        check_xi_len(element_type, xi)?;
        match (self, element_type) {
            (Self::Lagrange1, ElementType::SEG2) => {
                let x = xi[0];
                Ok(vec![0.5 * (1.0 - x), 0.5 * (1.0 + x)])
            }
            (Self::Lagrange1, ElementType::TRI3) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![1.0 - a - b, a, b])
            }
            (Self::Lagrange1, ElementType::QUA4) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![
                    0.25 * (1.0 - a) * (1.0 - b),
                    0.25 * (1.0 + a) * (1.0 - b),
                    0.25 * (1.0 + a) * (1.0 + b),
                    0.25 * (1.0 - a) * (1.0 + b),
                ])
            }
            (Self::Lagrange1, ElementType::TET4) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                Ok(vec![1.0 - a - b - c, a, b, c])
            }
            (Self::Lagrange1, ElementType::HEX8) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let mut n = Vec::with_capacity(8);
                for &(xi_i, eta_i, zeta_i) in HEX8_REF_NODES.iter() {
                    n.push(0.125 * (1.0 + xi_i * a) * (1.0 + eta_i * b) * (1.0 + zeta_i * c));
                }
                Ok(n)
            }
            (Self::Lagrange1, ElementType::POI1) => unreachable!(), // ruled out above
        }
    }

    /// Evaluate the reference derivatives `∂N_i/∂ξ_j` at the reference
    /// point `xi`.
    ///
    /// Returns a flat row-major buffer of length
    /// `nodes_per_cell × topological_dim`, where entry
    /// `[i * topological_dim + j]` is `∂N_i/∂ξ_j`.
    ///
    /// # Errors
    ///
    /// - `xi` has the wrong length;
    /// - the `(self, element_type)` pair is not supported.
    pub fn dshape_dxi(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check_compat(element_type)?;
        check_xi_len(element_type, xi)?;
        match (self, element_type) {
            (Self::Lagrange1, ElementType::SEG2) => Ok(vec![-0.5, 0.5]),
            (Self::Lagrange1, ElementType::TRI3) => Ok(vec![
                -1.0, -1.0, // dN1
                1.0, 0.0, // dN2
                0.0, 1.0, // dN3
            ]),
            (Self::Lagrange1, ElementType::QUA4) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![
                    -0.25 * (1.0 - b), -0.25 * (1.0 - a), // dN1
                    0.25 * (1.0 - b), -0.25 * (1.0 + a), // dN2
                    0.25 * (1.0 + b), 0.25 * (1.0 + a), // dN3
                    -0.25 * (1.0 + b), 0.25 * (1.0 - a), // dN4
                ])
            }
            (Self::Lagrange1, ElementType::TET4) => Ok(vec![
                -1.0, -1.0, -1.0, // dN1
                1.0, 0.0, 0.0, // dN2
                0.0, 1.0, 0.0, // dN3
                0.0, 0.0, 1.0, // dN4
            ]),
            (Self::Lagrange1, ElementType::HEX8) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let mut out = Vec::with_capacity(8 * 3);
                for &(xi_i, eta_i, zeta_i) in HEX8_REF_NODES.iter() {
                    let one_a = 1.0 + xi_i * a;
                    let one_b = 1.0 + eta_i * b;
                    let one_c = 1.0 + zeta_i * c;
                    out.push(0.125 * xi_i * one_b * one_c);
                    out.push(0.125 * eta_i * one_a * one_c);
                    out.push(0.125 * zeta_i * one_a * one_b);
                }
                Ok(out)
            }
            (Self::Lagrange1, ElementType::POI1) => unreachable!(),
        }
    }

    fn check_compat(self, element_type: ElementType) -> Result<()> {
        if !self.is_compatible_with(element_type) {
            return Err(PyrucastError::Message(format!(
                "interpolation {} is not defined for {}",
                self, element_type
            )));
        }
        Ok(())
    }
}

fn check_xi_len(element_type: ElementType, xi: &[f64]) -> Result<()> {
    let expected = element_type.topological_dim();
    if xi.len() != expected {
        return Err(PyrucastError::Message(format!(
            "xi has length {}, expected {} for {}",
            xi.len(),
            expected,
            element_type
        )));
    }
    Ok(())
}

/// Reference coordinates of the 8 HEX8 nodes, in local order
/// (bottom face CCW then top face CCW).
const HEX8_REF_NODES: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
];

impl fmt::Display for Interpolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl crate::dump::Dump for Interpolation {
    fn dump_with(&self, _opts: &crate::dump::DumpOptions) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum of Lagrange shape functions equals 1 everywhere (partition of
    /// unity).
    fn check_partition_of_unity(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let n = interp.shape(et, xi).unwrap();
        let s: f64 = n.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "{} on {} at xi={:?}: sum N_i = {} ≠ 1",
            interp,
            et,
            xi,
            s
        );
    }

    /// Sum of reference derivatives along each direction is 0 (partition
    /// of unity, differentiated).
    fn check_derivatives_sum_to_zero(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let dn = interp.dshape_dxi(et, xi).unwrap();
        let n_nodes = et.nodes_per_cell();
        let ref_dim = et.topological_dim();
        for j in 0..ref_dim {
            let mut s = 0.0;
            for i in 0..n_nodes {
                s += dn[i * ref_dim + j];
            }
            assert!(
                s.abs() < 1e-12,
                "{} on {}: Σ_i dN_i/dξ_{} = {} ≠ 0",
                interp,
                et,
                j,
                s
            );
        }
    }

    /// At node `i`, `N_i = 1` and `N_j = 0` for `j ≠ i` (Kronecker
    /// property of Lagrange interpolations).
    fn check_kronecker(interp: Interpolation, et: ElementType, ref_nodes: &[Vec<f64>]) {
        for (i, xi) in ref_nodes.iter().enumerate() {
            let n = interp.shape(et, xi).unwrap();
            for (j, &v) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-12,
                    "{} on {} at node {}: N_{} = {} ≠ {}",
                    interp,
                    et,
                    i,
                    j,
                    v,
                    expected
                );
            }
        }
    }

    #[test]
    fn lagrange1_seg2() {
        // Reference nodes: ξ = ±1
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::SEG2,
            &[vec![-1.0], vec![1.0]],
        );
        for xi in [-1.0, -0.3, 0.0, 0.7, 1.0] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
        }
    }

    #[test]
    fn lagrange1_tri3() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TRI3,
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        for (a, b) in [
            (0.25, 0.25),
            (1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.0),
            (0.0, 0.5),
        ] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_qua4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::QUA4,
            &[
                vec![-1.0, -1.0],
                vec![1.0, -1.0],
                vec![1.0, 1.0],
                vec![-1.0, 1.0],
            ],
        );
        for (a, b) in [(0.0, 0.0), (0.3, -0.7), (-0.5, 0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_tet4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TET4,
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        for (a, b, c) in [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
        }
    }

    #[test]
    fn lagrange1_hex8() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::HEX8,
            &HEX8_REF_NODES
                .iter()
                .map(|&(a, b, c)| vec![a, b, c])
                .collect::<Vec<_>>(),
        );
        for (a, b, c) in [(0.0, 0.0, 0.0), (0.3, -0.7, 0.5), (-0.5, 0.5, -0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
        }
    }

    #[test]
    fn rejects_poi1() {
        assert!(!Interpolation::Lagrange1.is_compatible_with(ElementType::POI1));
        assert!(Interpolation::Lagrange1
            .shape(ElementType::POI1, &[])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::POI1, &[])
            .is_err());
    }

    #[test]
    fn rejects_bad_xi_length() {
        assert!(Interpolation::Lagrange1
            .shape(ElementType::SEG2, &[0.0, 0.0])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::TRI3, &[0.0])
            .is_err());
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", Interpolation::Lagrange1), "LAGRANGE1");
        assert_eq!(
            Interpolation::from_name("lagrange1"),
            Some(Interpolation::Lagrange1)
        );
        assert_eq!(Interpolation::from_name("LAG1"), Some(Interpolation::Lagrange1));
        assert_eq!(Interpolation::from_name("unknown"), None);
    }
}
