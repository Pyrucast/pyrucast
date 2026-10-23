//! The point algebra of large transformations — how a stress measure is
//! carried from one configuration to the other, given the deformation gradient
//! `F` of the point.
//!
//! Every conversion goes through the second Piola–Kirchhoff stress `S`, the
//! Lagrangian one a law reading Green–Lagrange writes:
//!
//! | measure | from `S` | to `S` |
//! |---|---|---|
//! | Cauchy `σ` | `J⁻¹ F S Fᵀ` | `J F⁻¹ σ F⁻ᵀ` |
//! | Kirchhoff `τ = J σ` | `F S Fᵀ` | `F⁻¹ τ F⁻ᵀ` |
//! | first Piola–Kirchhoff `P` | `F S` | `F⁻¹ P` |
//!
//! Pure, on the stack, branch-free: they run at every Gauss point. A degenerate
//! `F` (`det F = 0`) yields inf/NaN — the mathematical value, not a checked
//! case, as for the measures of
//! [`deformation`](fn@crate::ops::element_field::deformation).

use nalgebra::Matrix3;

/// `F⁻¹`, or NaN everywhere when `F` is singular.
fn inverse(f: &Matrix3<f64>) -> Matrix3<f64> {
    f.try_inverse()
        .unwrap_or_else(|| Matrix3::from_element(f64::NAN))
}

/// `S` from the Cauchy stress: `S = J F⁻¹ σ F⁻ᵀ`.
pub(crate) fn pk2_from_cauchy(sigma: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    let fi = inverse(f);
    f.determinant() * fi * sigma * fi.transpose()
}

/// `S` from the Kirchhoff stress: `S = F⁻¹ τ F⁻ᵀ`.
pub(crate) fn pk2_from_kirchhoff(tau: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    let fi = inverse(f);
    fi * tau * fi.transpose()
}

/// `S` from the first Piola–Kirchhoff stress: `S = F⁻¹ P`.
pub(crate) fn pk2_from_pk1(p: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    inverse(f) * p
}

/// The Cauchy stress from `S`: `σ = J⁻¹ F S Fᵀ`.
pub(crate) fn cauchy_from_pk2(s: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    f * s * f.transpose() / f.determinant()
}

/// The Kirchhoff stress from `S`: `τ = F S Fᵀ`.
pub(crate) fn kirchhoff_from_pk2(s: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    f * s * f.transpose()
}

/// The first Piola–Kirchhoff stress from `S`: `P = F S`.
pub(crate) fn pk1_from_pk2(s: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    f * s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deformation gradient with stretch, shear and rotation, `det F > 0`.
    fn some_f() -> Matrix3<f64> {
        Matrix3::new(1.2, 0.3, -0.1, 0.05, 0.9, 0.2, 0.1, -0.15, 1.1)
    }

    fn some_s() -> Matrix3<f64> {
        Matrix3::new(10.0, 2.0, -1.0, 2.0, -4.0, 0.5, -1.0, 0.5, 3.0)
    }

    #[test]
    fn every_measure_comes_back_to_the_same_pk2() {
        let (f, s) = (some_f(), some_s());
        let back = [
            pk2_from_cauchy(&cauchy_from_pk2(&s, &f), &f),
            pk2_from_kirchhoff(&kirchhoff_from_pk2(&s, &f), &f),
            pk2_from_pk1(&pk1_from_pk2(&s, &f), &f),
        ];
        for b in back {
            assert!((b - s).norm() < 1e-12 * s.norm());
        }
    }

    #[test]
    fn the_cauchy_stress_of_a_rotation_is_the_rotated_stress() {
        // F = R: σ = R S Rᵀ, J = 1 — objectivity, in one line.
        let (c, sn) = (0.6_f64, 0.8_f64);
        let r = Matrix3::new(c, -sn, 0.0, sn, c, 0.0, 0.0, 0.0, 1.0);
        let s = some_s();
        let sigma = cauchy_from_pk2(&s, &r);
        assert!((sigma - r * s * r.transpose()).norm() < 1e-12 * s.norm());
    }
}
