//! Quadrature rules on the reference element.
//!
//! A [`QuadratureRule`] returns a list of `(ξ_g, w_g)` pairs — reference
//! coordinates and weights — over the reference element of a given
//! [`ElementType`]. It is the **numerical** counterpart of an
//! [`crate::interpolation::Interpolation`]: the points at which shape
//! functions are sampled when integrating.
//!
//! The reference frames are those documented on each `ElementType`
//! variant (see [`crate::element_type`]). The weights are calibrated so
//! that the sum equals the measure of the reference domain (i.e., 2 for
//! SEG2, 1/2 for TRI3, 4 for QUA4, 1/6 for TET4, 8 for HEX8).
//!
//! # Example
//!
//! ```
//! use pyrucast::element_type::ElementType;
//! use pyrucast::quadrature::QuadratureRule;
//!
//! let (xi, w) = QuadratureRule::Gauss.points(ElementType::QUA4).unwrap();
//! assert_eq!(w.len(), 4);          // 2×2 Gauss-Legendre on [-1,1]²
//! let s: f64 = w.iter().sum();
//! assert!((s - 4.0).abs() < 1e-12);  // area of the reference square
//! assert_eq!(xi.len(), 4 * 2);       // n_g × ref_dim, flat
//! ```

use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported quadrature rules.
///
/// The current variant (`Gauss`) selects, for each `ElementType`, the
/// **standard rule that integrates the Lagrange-1 mass matrix exactly**
/// on a straight reference element:
///
/// | ElementType | Rule | n_g |
/// |---|---|---|
/// | SEG2 | Gauss-Legendre on `[-1, +1]` | 2 |
/// | TRI3 | Hammer mid-edge on the unit simplex | 3 |
/// | QUA4 | 2×2 Gauss-Legendre tensor product | 4 |
/// | TET4 | Hammer 4-point on the unit simplex (exact for degree 2) | 4 |
/// | HEX8 | 2×2×2 Gauss-Legendre tensor product | 8 |
///
/// `POI1` has no reference frame and is rejected.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum QuadratureRule {
    /// Standard Gauss rule, picked per element type as documented above.
    Gauss,
}

impl QuadratureRule {
    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gauss => "GAUSS",
        }
    }

    /// Parse from a short name (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "GAUSS" => Some(Self::Gauss),
            _ => None,
        }
    }

    /// Whether this rule is defined for `element_type`.
    pub fn is_compatible_with(self, element_type: ElementType) -> bool {
        !matches!(element_type, ElementType::POI1)
    }

    /// Number of integration points for this rule on `element_type`.
    pub fn point_count(self, element_type: ElementType) -> Result<usize> {
        self.check_compat(element_type)?;
        Ok(match (self, element_type) {
            (Self::Gauss, ElementType::SEG2) => 2,
            (Self::Gauss, ElementType::TRI3) => 3,
            (Self::Gauss, ElementType::QUA4) => 4,
            (Self::Gauss, ElementType::TET4) => 4,
            (Self::Gauss, ElementType::HEX8) => 8,
            (_, ElementType::POI1) => unreachable!(),
        })
    }

    /// Reference points and weights of the rule.
    ///
    /// Returns `(xi, w)`:
    /// - `xi` is a flat row-major buffer of length `n_g × ref_dim`,
    ///   with `xi[g * ref_dim .. (g+1) * ref_dim]` the coordinates of
    ///   the `g`-th integration point;
    /// - `w` has length `n_g`.
    pub fn points(self, element_type: ElementType) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check_compat(element_type)?;
        Ok(match (self, element_type) {
            (Self::Gauss, ElementType::SEG2) => {
                let a = 1.0 / 3.0_f64.sqrt();
                (vec![-a, a], vec![1.0, 1.0])
            }
            (Self::Gauss, ElementType::TRI3) => (
                vec![
                    0.5, 0.0, //
                    0.5, 0.5, //
                    0.0, 0.5,
                ],
                vec![1.0 / 6.0; 3],
            ),
            (Self::Gauss, ElementType::QUA4) => {
                let a = 1.0 / 3.0_f64.sqrt();
                (
                    vec![
                        -a, -a, //
                        a, -a, //
                        a, a, //
                        -a, a,
                    ],
                    vec![1.0; 4],
                )
            }
            (Self::Gauss, ElementType::TET4) => {
                let alpha = (5.0 - 5.0_f64.sqrt()) / 20.0;
                let beta = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
                (
                    vec![
                        alpha, alpha, alpha, //
                        beta, alpha, alpha, //
                        alpha, beta, alpha, //
                        alpha, alpha, beta,
                    ],
                    vec![1.0 / 24.0; 4],
                )
            }
            (Self::Gauss, ElementType::HEX8) => {
                let a = 1.0 / 3.0_f64.sqrt();
                let mut xi = Vec::with_capacity(8 * 3);
                for &z in &[-a, a] {
                    for &y in &[-a, a] {
                        for &x in &[-a, a] {
                            xi.push(x);
                            xi.push(y);
                            xi.push(z);
                        }
                    }
                }
                // The bottom face is traversed in row-major (x then y) so the
                // resulting order is not strictly CCW within a face. That is
                // fine: the rule is symmetric and the order is opaque to
                // callers, who only care about (xi_g, w_g) as a set.
                (xi, vec![1.0; 8])
            }
            (_, ElementType::POI1) => unreachable!(),
        })
    }

    fn check_compat(self, element_type: ElementType) -> Result<()> {
        if !self.is_compatible_with(element_type) {
            return Err(PyrucastError::Message(format!(
                "quadrature {} is not defined for {}",
                self, element_type
            )));
        }
        Ok(())
    }
}

impl fmt::Display for QuadratureRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_volume(et: ElementType) -> f64 {
        match et {
            ElementType::SEG2 => 2.0,
            ElementType::TRI3 => 0.5,
            ElementType::QUA4 => 4.0,
            ElementType::TET4 => 1.0 / 6.0,
            ElementType::HEX8 => 8.0,
            ElementType::POI1 => unreachable!(),
        }
    }

    #[test]
    fn weights_sum_to_reference_volume() {
        for et in [
            ElementType::SEG2,
            ElementType::TRI3,
            ElementType::QUA4,
            ElementType::TET4,
            ElementType::HEX8,
        ] {
            let (_xi, w) = QuadratureRule::Gauss.points(et).unwrap();
            let s: f64 = w.iter().sum();
            let expected = ref_volume(et);
            assert!(
                (s - expected).abs() < 1e-12,
                "{}: weights sum = {} ≠ {}",
                et,
                s,
                expected
            );
        }
    }

    #[test]
    fn buffer_layouts() {
        for et in [
            ElementType::SEG2,
            ElementType::TRI3,
            ElementType::QUA4,
            ElementType::TET4,
            ElementType::HEX8,
        ] {
            let n_g = QuadratureRule::Gauss.point_count(et).unwrap();
            let (xi, w) = QuadratureRule::Gauss.points(et).unwrap();
            assert_eq!(w.len(), n_g);
            assert_eq!(xi.len(), n_g * et.topological_dim());
        }
    }

    /// On SEG2, the 2-point Gauss rule must integrate `∫_{-1}^{1} ξ^p dξ`
    /// exactly for `p ≤ 3`.
    #[test]
    fn gauss_seg2_exact_up_to_degree_3() {
        let (xi, w) = QuadratureRule::Gauss.points(ElementType::SEG2).unwrap();
        for p in 0..=3 {
            let mut sum = 0.0;
            for g in 0..w.len() {
                sum += w[g] * xi[g].powi(p);
            }
            // ∫_{-1}^{1} ξ^p dξ = 2/(p+1) if p even, else 0.
            let expected = if p % 2 == 0 {
                2.0 / (p + 1) as f64
            } else {
                0.0
            };
            assert!(
                (sum - expected).abs() < 1e-12,
                "SEG2 Gauss not exact for ξ^{}: got {}, expected {}",
                p,
                sum,
                expected
            );
        }
    }

    /// On TRI3, the Hammer mid-edge rule must integrate any polynomial
    /// of degree ≤ 2 exactly. Test on `f(ξ, η) = ξ²` whose exact integral
    /// over the unit simplex is 1/12.
    #[test]
    fn gauss_tri3_exact_for_degree_2() {
        let (xi, w) = QuadratureRule::Gauss.points(ElementType::TRI3).unwrap();
        let mut sum = 0.0;
        for g in 0..w.len() {
            let a = xi[g * 2];
            sum += w[g] * a * a;
        }
        let expected = 1.0 / 12.0;
        assert!(
            (sum - expected).abs() < 1e-12,
            "TRI3 Gauss: ∫ξ² = {} ≠ {}",
            sum,
            expected
        );
    }

    #[test]
    fn rejects_poi1() {
        assert!(!QuadratureRule::Gauss.is_compatible_with(ElementType::POI1));
        assert!(QuadratureRule::Gauss.points(ElementType::POI1).is_err());
        assert!(QuadratureRule::Gauss.point_count(ElementType::POI1).is_err());
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", QuadratureRule::Gauss), "GAUSS");
        assert_eq!(
            QuadratureRule::from_name("gauss"),
            Some(QuadratureRule::Gauss)
        );
        assert_eq!(QuadratureRule::from_name("unknown"), None);
    }
}
