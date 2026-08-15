//! The cubic Hermite basis on the reference segment `ξ ∈ [-1, +1]`.
//!
//! This is the one basis that does **not** live in an element type's file, and
//! the reason is structural: every Lagrange basis has one function per
//! geometric node, so it is a property of the element. Hermite has **two per
//! node** — a value and a slope — so it is a property of the *interpolation
//! family*, and a `SEG2` carries it or not depending on what the space declares.
//!
//! ## The four functions
//!
//! Ordered by the degree of freedom they belong to, node by node:
//! `[w_A, w'_A, w_B, w'_B]`.
//!
//! ```text
//! H₁ = ¼(2 − 3ξ + ξ³)        H₂ = ¼(1 − ξ − ξ² + ξ³)
//! H₃ = ¼(2 + 3ξ − ξ³)        H₄ = ¼(−1 − ξ + ξ² + ξ³)
//! ```
//!
//! They are the **Kronecker basis of the C¹ interpolation**: at each end, one
//! function has value 1 and slope 0, the other value 0 and slope 1, and the two
//! from the far end vanish in both. That is what makes the deflection *and* its
//! slope continuous from one element to the next — the C¹ continuity a
//! fourth-order equation demands, and which no Lagrange basis provides.
//!
//! ## The slope is with respect to `ξ`, not `x`
//!
//! `H₂` and `H₄` carry a *reference* slope `∂w/∂ξ`, while the physical degree of
//! freedom of a beam is `θ = ∂w/∂x`. The two differ by the Jacobian,
//! `∂w/∂ξ = J ∂w/∂x` with `J = L/2` on a straight segment.
//!
//! Keeping the `J` **out** of this file is deliberate: it is exactly the
//! reference→physical mapping that every other basis goes through, so it
//! belongs where all the other Jacobians are applied and not baked into a
//! constant. Written the other way — with an `L` inside the shape function —
//! the basis would stop being a reference-element quantity, and could no longer
//! be tabulated once per element type.

/// Number of shape functions of the cubic Hermite basis on a segment.
pub const HERMITE3_SHAPE_COUNT: usize = 4;

/// The four values `Hᵢ(ξ)`.
pub fn shape(xi: f64) -> [f64; HERMITE3_SHAPE_COUNT] {
    let x2 = xi * xi;
    let x3 = x2 * xi;
    [
        0.25 * (2.0 - 3.0 * xi + x3),
        0.25 * (1.0 - xi - x2 + x3),
        0.25 * (2.0 + 3.0 * xi - x3),
        0.25 * (-1.0 - xi + x2 + x3),
    ]
}

/// The four first derivatives `∂Hᵢ/∂ξ`.
pub fn dshape(xi: f64) -> [f64; HERMITE3_SHAPE_COUNT] {
    let x2 = xi * xi;
    [
        0.25 * (-3.0 + 3.0 * x2),
        0.25 * (-1.0 - 2.0 * xi + 3.0 * x2),
        0.25 * (3.0 - 3.0 * x2),
        0.25 * (-1.0 + 2.0 * xi + 3.0 * x2),
    ]
}

/// The four second derivatives `∂²Hᵢ/∂ξ²` — **linear** in `ξ`.
///
/// This is what a Lagrange basis cannot give: a `SEG2` interpolates linearly,
/// so its second derivative is identically zero and a curvature built from it
/// is empty. Here the curvature varies across the element, which is the actual
/// solution of `(EIw'')'' = 0` on a free span.
pub fn d2shape(xi: f64) -> [f64; HERMITE3_SHAPE_COUNT] {
    [
        0.25 * (6.0 * xi),
        0.25 * (-2.0 + 6.0 * xi),
        0.25 * (-6.0 * xi),
        0.25 * (2.0 + 6.0 * xi),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defining property: at each end, the value pair reads the deflection
    /// and the slope pair reads the slope — a Kronecker delta on *both*.
    ///
    /// A basis that got any one of these sixteen numbers wrong would still look
    /// like a plausible cubic, and would still assemble a symmetric positive
    /// stiffness. It would simply solve a different problem.
    #[test]
    fn hermite_is_a_kronecker_basis_in_value_and_slope() {
        for (end, &xi) in [-1.0_f64, 1.0].iter().enumerate() {
            let n = shape(xi);
            let dn = dshape(xi);
            for i in 0..HERMITE3_SHAPE_COUNT {
                // Function 2·end is "the value at this end", 2·end+1 "the slope".
                let want_value = if i == 2 * end { 1.0 } else { 0.0 };
                let want_slope = if i == 2 * end + 1 { 1.0 } else { 0.0 };
                assert!(
                    (n[i] - want_value).abs() < 1e-14,
                    "H{}({xi}) = {}, expected {want_value}",
                    i + 1,
                    n[i]
                );
                assert!(
                    (dn[i] - want_slope).abs() < 1e-14,
                    "H'{}({xi}) = {}, expected {want_slope}",
                    i + 1,
                    dn[i]
                );
            }
        }
    }

    /// The two **rigid-body modes** are reproduced exactly, and with zero
    /// curvature — the property a stiffness matrix needs if it is not to resist
    /// a motion that costs no energy.
    ///
    /// Translation `w = c` has DOFs `[c, 0, c, 0]`, so it is the partition of
    /// unity of the *value* pair alone; rotation `w = aξ` has `[−a, a, a, a]`.
    /// (The slope functions do **not** cancel on their own — `H₂ + H₄ =
    /// ½ξ(ξ²−1)` — which is why the statement has to be made on the modes and
    /// not on the basis.)
    #[test]
    fn both_rigid_body_modes_are_exact_and_curvature_free() {
        /// The four degrees of freedom of a rigid mode, and the deflection they
        /// are meant to reproduce.
        type RigidMode = ([f64; 4], fn(f64) -> f64);
        let modes: [RigidMode; 2] = [
            ([1.0, 0.0, 1.0, 0.0], |_| 1.0), // translation w = 1
            ([-1.0, 1.0, 1.0, 1.0], |x| x),  // rotation    w = ξ
        ];
        for (dof, exact) in modes {
            for k in -10..=10 {
                let xi = k as f64 / 10.0;
                let w: f64 = (0..4).map(|i| shape(xi)[i] * dof[i]).sum();
                let kappa: f64 = (0..4).map(|i| d2shape(xi)[i] * dof[i]).sum();
                assert!((w - exact(xi)).abs() < 1e-14, "mode at {xi}: {w}");
                assert!(kappa.abs() < 1e-14, "rigid mode bends at {xi}: {kappa}");
            }
        }
    }

    /// The basis reproduces **any** cubic exactly — which is the whole reason a
    /// beam element built on it is nodally exact on an unloaded span.
    #[test]
    fn reproduces_an_arbitrary_cubic() {
        // p(ξ) = 2 − 3ξ + 0.5ξ² + 4ξ³, and its slope.
        let p = |x: f64| 2.0 - 3.0 * x + 0.5 * x * x + 4.0 * x * x * x;
        let dp = |x: f64| -3.0 + x + 12.0 * x * x;
        let d2p = |x: f64| 1.0 + 24.0 * x;
        let dof = [p(-1.0), dp(-1.0), p(1.0), dp(1.0)];

        for k in -10..=10 {
            let xi = k as f64 / 10.0;
            let interp = |b: [f64; 4]| (0..4).map(|i| b[i] * dof[i]).sum::<f64>();
            assert!((interp(shape(xi)) - p(xi)).abs() < 1e-12, "value at {xi}");
            assert!((interp(dshape(xi)) - dp(xi)).abs() < 1e-12, "slope at {xi}");
            assert!(
                (interp(d2shape(xi)) - d2p(xi)).abs() < 1e-12,
                "curvature at {xi}"
            );
        }
    }

    /// The second derivatives are the derivatives of the first — checked
    /// numerically, so an algebra slip in either cannot hide behind the other.
    #[test]
    fn second_derivatives_match_a_finite_difference_of_the_first() {
        let h = 1e-6;
        for k in -8..=8 {
            let xi = k as f64 / 10.0;
            let (a, b) = (dshape(xi + h), dshape(xi - h));
            let exact = d2shape(xi);
            for i in 0..HERMITE3_SHAPE_COUNT {
                let fd = (a[i] - b[i]) / (2.0 * h);
                assert!(
                    (fd - exact[i]).abs() < 1e-6,
                    "H''{}({xi}): exact {} vs FD {fd}",
                    i + 1,
                    exact[i]
                );
            }
        }
    }
}
