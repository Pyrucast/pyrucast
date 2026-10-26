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
//! | dual `T` of Hencky `H = ½ ln C` | `S̃_ab / θ_ab` | `S̃_ab = θ_ab T̃_ab` |
//!
//! The last row works in the eigenbasis of `C = FᵀF = Σ c_a N_a⊗N_a` (the
//! tilde): `S = T : ∂H/∂E`, and `∂H/∂E` acts there entry by entry, with
//! `θ_aa = 1/c_a` and `θ_ab = (ln c_a − ln c_b)/(c_a − c_b)` — the divided
//! difference of `ln`, whose limit for equal eigenvalues is `1/c`.
//!
//! The rotation `R` of the polar decomposition `F = R U` turns the material
//! axes of a law computed on the current configuration
//! ([`rotate_frame`](fn@crate::ops::element_field::rotate_frame)).
//!
//! Pure, on the stack, with nothing checked: they run at every Gauss point. A
//! degenerate `F` (`det F = 0`) yields inf/NaN — the mathematical value, not a checked
//! case, as for the measures of
//! [`deformation`](fn@crate::ops::element_field::deformation).

use nalgebra::{Matrix3, Vector3};

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

/// `(ln a − ln b)/(a − b)`, and its limit `1/b` when `a → b` — a series there,
/// where the quotient would lose every digit.
fn log_divided_difference(a: f64, b: f64) -> f64 {
    let x = (a - b) / b;
    if x.abs() < 1e-6 {
        (1.0 - x / 2.0 + x * x / 3.0) / b
    } else {
        (a / b).ln() / (a - b)
    }
}

/// The eigenbasis of `C = FᵀF` and the weights `θ_ab` of `∂H/∂E` in it.
fn hencky_weights(f: &Matrix3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let eig = (f.transpose() * f).symmetric_eigen();
    let c: Vector3<f64> = eig.eigenvalues;
    let theta = Matrix3::from_fn(|a, b| {
        if a == b {
            1.0 / c[a]
        } else {
            log_divided_difference(c[a], c[b])
        }
    });
    (eig.eigenvectors, theta)
}

/// `S` from the dual `T` of the Hencky strain: `S = T : ∂H/∂E`, entry by entry
/// in the eigenbasis of `C`.
pub(crate) fn pk2_from_hencky(t: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    let (q, theta) = hencky_weights(f);
    let in_basis = q.transpose() * t * q;
    q * in_basis.component_mul(&theta) * q.transpose()
}

/// The dual `T` of the Hencky strain from `S`: the inverse of
/// [`pk2_from_hencky`].
pub(crate) fn hencky_from_pk2(s: &Matrix3<f64>, f: &Matrix3<f64>) -> Matrix3<f64> {
    let (q, theta) = hencky_weights(f);
    let in_basis = q.transpose() * s * q;
    q * in_basis.component_div(&theta) * q.transpose()
}

/// The rotation `R` of the polar decomposition `F = R U`: `R = F U⁻¹`, with
/// `U⁻¹ = Σ c_a^(−½) N_a⊗N_a` in the eigenbasis of `C = FᵀF`.
pub(crate) fn polar_rotation(f: &Matrix3<f64>) -> Matrix3<f64> {
    let eig = (f.transpose() * f).symmetric_eigen();
    let q = eig.eigenvectors;
    let u_inverse = Matrix3::from_diagonal(&eig.eigenvalues.map(|c| 1.0 / c.sqrt()));
    f * q * u_inverse * q.transpose()
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
            pk2_from_hencky(&hencky_from_pk2(&s, &f), &f),
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

    #[test]
    fn at_rest_and_in_rotation_the_hencky_dual_is_s() {
        // C = I: every weight is 1, T and S coincide.
        let (c, sn) = (0.6_f64, 0.8_f64);
        let r = Matrix3::new(c, -sn, 0.0, sn, c, 0.0, 0.0, 0.0, 1.0);
        let t = some_s();
        for f in [Matrix3::identity(), r] {
            assert!((pk2_from_hencky(&t, &f) - t).norm() < 1e-12 * t.norm());
        }
    }

    #[test]
    fn under_a_stretch_the_hencky_dual_follows_the_divided_differences() {
        // F = diag(2, 1, 1): c = (4, 1, 1). The axial weight is 1/4, the
        // shear between the stretched axis and another (ln 4 − ln 1)/(4 − 1),
        // the shear between the two others their limit, 1.
        let f = Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let t = Matrix3::new(8.0, 3.0, 0.0, 3.0, 1.0, 5.0, 0.0, 5.0, 2.0);
        let s = pk2_from_hencky(&t, &f);
        assert!((s[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((s[(0, 1)] - 3.0 * 4.0_f64.ln() / 3.0).abs() < 1e-12);
        assert!((s[(1, 2)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn the_polar_rotation_is_the_rotation_a_stretch_is_composed_with() {
        // F = R₀ U₀, with U₀ symmetric positive (stretch and shear) and R₀ a
        // turn about an oblique axis: the decomposition gives R₀ back.
        let axis = nalgebra::Unit::new_normalize(Vector3::new(1.0, 2.0, 0.5));
        let r0 = nalgebra::Rotation3::from_axis_angle(&axis, 0.7).into_inner();
        let u0 = Matrix3::new(1.3, 0.2, -0.1, 0.2, 0.8, 0.15, -0.1, 0.15, 1.1);
        let r = polar_rotation(&(r0 * u0));
        assert!((r - r0).norm() < 1e-12, "{r}");
        // And a pure stretch turns nothing.
        assert!((polar_rotation(&u0) - Matrix3::identity()).norm() < 1e-12);
    }
}
