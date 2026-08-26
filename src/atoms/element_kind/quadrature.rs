//! [`QuadratureRule`] — how an element is integrated.
//!
//! Like [`Interpolation`](super::Interpolation), the enum is a **tag**: the
//! points and weights live in each element's own file, behind
//! [`ElementKind::gauss`](super::ElementKind::gauss). The reduced rule is not
//! written anywhere at all — it is the trait's default, one point at the
//! element's [`ref_centroid`](super::ElementKind::ref_centroid) carrying its
//! [`ref_measure`](super::ElementKind::ref_measure).
//!
//! The weights of any rule sum to the reference measure; that is the invariant
//! the tests check for every type at once.

use crate::atoms::ElementType;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported quadrature rules.
///
/// `Gauss` selects, for each element type, the **standard rule that integrates
/// the Lagrange-1 mass matrix exactly** on a straight reference element — see
/// each element's [`gauss`](super::ElementKind::gauss). `POI1` has no reference
/// frame and is rejected.
///
/// ```
/// # use pyrucast::atoms::{ElementType, QuadratureRule};
/// // La règle d'intégration, choisie par sous-espace EF. Ses poids somment
/// // à la mesure de l'élément de référence — 1/2 pour un triangle.
/// let (_xi, w) = QuadratureRule::Gauss.points(ElementType::TRI3)?;
/// assert!((w.iter().sum::<f64>() - 0.5).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum QuadratureRule {
    /// Standard Gauss rule, picked per element type.
    Gauss,
    /// **Reduced** integration: a single point at the element centroid (weight
    /// = the reference measure). Exact for constants only; used for
    /// selective/reduced integration — e.g. the shear term of a Timoshenko
    /// beam, to avoid shear locking.
    Reduced,
}

impl QuadratureRule {
    /// Short name.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, QuadratureRule};
    /// assert_eq!(QuadratureRule::Gauss.name(), "GAUSS");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Gauss => "GAUSS",
            Self::Reduced => "REDUCED",
        }
    }

    /// Whether this rule is defined for `element_type` — i.e. anything but
    /// `POI1`, which has no reference frame to integrate over.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, QuadratureRule};
    /// assert!(QuadratureRule::Gauss.is_compatible_with(ElementType::TRI3));
    /// // Un point n'a pas d'élément de référence sur quoi intégrer.
    /// assert!(!QuadratureRule::Gauss.is_compatible_with(ElementType::POI1));
    /// ```
    pub fn is_compatible_with(self, element_type: ElementType) -> bool {
        element_type.topological_dim() > 0
    }

    /// Number of integration points for this rule on `element_type`.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, QuadratureRule};
    /// assert_eq!(QuadratureRule::Gauss.point_count(ElementType::TRI3)?, 3);
    /// assert!(QuadratureRule::Gauss.point_count(ElementType::POI1).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn point_count(self, element_type: ElementType) -> Result<usize> {
        Ok(self.points(element_type)?.1.len())
    }

    /// Reference points and weights of the rule.
    ///
    /// Returns `(xi, w)`:
    /// - `xi` is a flat row-major buffer of length `n_g × ref_dim`, with
    ///   `xi[g * ref_dim .. (g+1) * ref_dim]` the coordinates of the `g`-th
    ///   integration point;
    /// - `w` has length `n_g`, and sums to the element's reference measure.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, QuadratureRule};
    /// // `xi` est à plat, ligne-major : n_g × ref_dim ; `w` a n_g entrées.
    /// let (xi, w) = QuadratureRule::Gauss.points(ElementType::QUA4)?;
    /// assert_eq!((xi.len(), w.len()), (4 * 2, 4));
    /// // La somme des poids est la mesure du carré de référence : 4.
    /// assert!((w.iter().sum::<f64>() - 4.0).abs() < 1e-12);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn points(self, element_type: ElementType) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check_compat(element_type)?;
        let k = element_type.as_kind();
        Ok(match self {
            Self::Gauss => k.gauss(),
            Self::Reduced => k.reduced(),
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

impl crate::named::Named for QuadratureRule {
    const LABEL: &'static str = "quadrature rule";
    const VALUES: &'static [Self] = &[Self::Gauss, Self::Reduced];

    fn name(self) -> &'static str {
        QuadratureRule::name(self)
    }
}

impl fmt::Display for QuadratureRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl crate::dump::Dump for QuadratureRule {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        self.to_string()
    }
}

// ─── Building blocks shared by several elements ─────────────────────────────

/// 3-point Gauss-Legendre rule on `[-1, 1]` (exact to degree 5): points
/// `(∓√(3/5), 0, +√(3/5))`, weights `(5/9, 8/9, 5/9)`.
///
/// The 1-D factor of every quadratic tensor-product rule (`SEG3`, `QUA8`/`QUA9`,
/// `HEX20`/`HEX27`, and the `ζ` direction of `PENTA15`).
pub(super) fn gauss3_1d() -> ([f64; 3], [f64; 3]) {
    let a = (3.0_f64 / 5.0).sqrt();
    ([-a, 0.0, a], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
}

/// 6-point degree-4 symmetric rule on the unit triangle (Dunavant). Returns
/// `(xi, w)` with `xi` row-major over 6 points × 2 coordinates; the weights
/// sum to the reference area `1/2`. Shared by `TRI6` and the triangular factor
/// of `PENTA15`.
pub(super) fn tri6_gauss() -> (Vec<f64>, Vec<f64>) {
    // Two 3-point orbits (b, b), (1-2b, b), (b, 1-2b).
    let b1 = 0.445_948_490_915_965;
    let b2 = 0.091_576_213_509_771;
    let w1 = 0.223_381_589_678_011 / 2.0;
    let w2 = 0.109_951_743_655_322 / 2.0;
    let orbit = |b: f64| {
        let a = 1.0 - 2.0 * b;
        [(b, b), (a, b), (b, a)]
    };
    let mut xi = Vec::with_capacity(6 * 2);
    let mut w = Vec::with_capacity(6);
    for (b, weight) in [(b1, w1), (b2, w2)] {
        for (x, y) in orbit(b) {
            xi.push(x);
            xi.push(y);
            w.push(weight);
        }
    }
    (xi, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::named::Named;

    /// Every element type carrying a reference frame (i.e. all but `POI1`),
    /// for the parametric quadrature tests. Derived from
    /// [`ElementType::ALL`] so a new type is covered the day it is declared.
    fn all_fe_types() -> impl Iterator<Item = ElementType> {
        ElementType::ALL
            .iter()
            .copied()
            .filter(|et| et.topological_dim() > 0)
    }

    #[test]
    fn weights_sum_to_reference_volume() {
        for et in all_fe_types() {
            let (_xi, w) = QuadratureRule::Gauss.points(et).unwrap();
            let s: f64 = w.iter().sum();
            let expected = et.as_kind().ref_measure();
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
        for et in all_fe_types() {
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

    /// Factorial helper for the exact simplex-monomial integrals.
    fn fact(n: u32) -> f64 {
        (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
    }

    /// The custom TRI6 (degree-4) rule integrates every monomial `ξ^p η^q`
    /// with `p + q ≤ 4` exactly; on the unit triangle
    /// `∫ ξ^p η^q = p! q! / (p+q+2)!`.
    #[test]
    fn gauss_tri6_exact_up_to_degree_4() {
        let (xi, w) = QuadratureRule::Gauss.points(ElementType::TRI6).unwrap();
        for p in 0..=4u32 {
            for q in 0..=(4 - p) {
                let mut sum = 0.0;
                for g in 0..w.len() {
                    sum += w[g] * xi[2 * g].powi(p as i32) * xi[2 * g + 1].powi(q as i32);
                }
                let expected = fact(p) * fact(q) / fact(p + q + 2);
                assert!(
                    (sum - expected).abs() < 1e-12,
                    "TRI6: ∫ξ^{p}η^{q} = {sum} ≠ {expected}"
                );
            }
        }
    }

    /// The custom TET10 (Keast degree-4) rule integrates every monomial
    /// `ξ^p η^q ζ^r` with `p + q + r ≤ 4` exactly; on the unit tetrahedron
    /// `∫ ξ^p η^q ζ^r = p! q! r! / (p+q+r+3)!`.
    #[test]
    fn gauss_tet10_exact_up_to_degree_4() {
        let (xi, w) = QuadratureRule::Gauss.points(ElementType::TET10).unwrap();
        for p in 0..=4u32 {
            for q in 0..=(4 - p) {
                for r in 0..=(4 - p - q) {
                    let mut sum = 0.0;
                    for g in 0..w.len() {
                        sum += w[g]
                            * xi[3 * g].powi(p as i32)
                            * xi[3 * g + 1].powi(q as i32)
                            * xi[3 * g + 2].powi(r as i32);
                    }
                    let expected = fact(p) * fact(q) * fact(r) / fact(p + q + r + 3);
                    assert!(
                        (sum - expected).abs() < 1e-12,
                        "TET10: ∫ξ^{p}η^{q}ζ^{r} = {sum} ≠ {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_poi1() {
        assert!(!QuadratureRule::Gauss.is_compatible_with(ElementType::POI1));
        assert!(QuadratureRule::Gauss.points(ElementType::POI1).is_err());
        assert!(QuadratureRule::Gauss
            .point_count(ElementType::POI1)
            .is_err());
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
