//! von Mises (J2) plasticity — perfect, or with linear isotropic hardening.
//!
//! The archetypal metal-plasticity law: yielding depends on the **deviator**
//! alone, so it is insensitive to hydrostatic pressure and plastic flow is
//! isochoric. The yield surface is a cylinder about the hydrostatic axis,
//!
//! ```text
//! f(σ, p) = q − σ_y(p)          q = √(3 J₂)
//! ```
//!
//! with `σ_y(p) = σ_y + H·p`. `H = 0` is the perfect law — one code path serves
//! both, which is why they share this file rather than duplicating a return map
//! that differs by a single term.
//!
//! ## The closed-form return
//!
//! Associated flow on a cylinder means the return is **radial**: the deviator
//! is scaled, its direction untouched, and the hydrostatic part is left alone.
//! Consistency gives the multiplier in one step, with no iteration:
//!
//! ```text
//! Δp = (q_trial − σ_y(p_A)) / (3μ + H)
//! ```
//!
//! That closed form is why von Mises does not go through the cutting plane: an
//! exact answer beats a converged one.

use crate::error::Result;
use crate::models::plastic::{deviator, elastic_tangent, von_mises_stress, MatParams, PrevState};

/// Radial return onto `q = σ_y + H·p`.
///
/// `hardening` is `H`; pass `0.0` for the perfect law. Returns the updated
/// `(σ, ε_p, p)`, all full 3-D.
pub fn return_map(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    hardening: f64,
) -> Result<([f64; 6], [f64; 6], f64)> {
    let sigma_y0 = mat.get("sigma_y")?;
    let q = von_mises_stress(trial);
    let yield_now = sigma_y0 + hardening * prev.p;
    let f = q - yield_now;
    if f <= 0.0 || q == 0.0 {
        return Ok((*trial, prev.eps_p, prev.p)); // elastic
    }
    // Consistency: q − 3μΔp = σ_y + H(p + Δp).
    let dp = f / (3.0 * mat.mu + hardening);
    let s_trial = deviator(trial);
    let mean = (trial[0] + trial[1] + trial[2]) / 3.0;
    let scale = (yield_now + hardening * dp) / q;
    // Flow direction n = (3/2)·s/q, so Δε_p = Δp·n = (3Δp/2q)·s — a tensor, so
    // the off-diagonals take the same factor with no engineering doubling.
    let factor = 1.5 * dp / q;

    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    for i in 0..6 {
        let s_new = s_trial[i] * scale;
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        eps_p[i] += factor * s_trial[i];
    }
    Ok((sigma, eps_p, prev.p + dp))
}

/// The consistent tangent `D_alg = ∂σ(B)/∂ε(B)` of the radial return, evaluated
/// at the trial stress — the classical algorithmic modulus,
///
/// ```text
/// D = K·1⊗1 + 2μθ·I_dev − 2μθ̄·n̂⊗n̂
/// θ = σ_y(p+Δp)/q_trial          θ̄ = 3μ/(3μ + H) − (1 − θ)
/// ```
///
/// with `n̂ = s_trial/‖s_trial‖` the **unit** deviatoric direction. For `H = 0`
/// the two coefficients collapse (`θ̄ = θ`) and this is exactly the perfect-J2
/// tangent — which is why hardening costs one extra term rather than a second
/// derivation.
///
/// Note the difference from the **continuum** elastoplastic modulus: `θ` here
/// accounts for the *finite* step, and dropping it would cost Newton its
/// quadratic convergence.
pub fn tangent(trial: &[f64; 6], mat: &MatParams, hardening: f64, p_prev: f64) -> [[f64; 6]; 6] {
    let (lambda, mu) = (mat.lambda, mat.mu);
    let sigma_y0 = mat.get("sigma_y").unwrap_or(0.0);
    let q = von_mises_stress(trial);
    let yield_now = sigma_y0 + hardening * p_prev;
    if q <= yield_now || q == 0.0 {
        return elastic_tangent(lambda, mu);
    }
    let dp = (q - yield_now) / (3.0 * mu + hardening);
    let theta = (yield_now + hardening * dp) / q;
    let theta_bar = 3.0 * mu / (3.0 * mu + hardening) - (1.0 - theta);

    let k = lambda + 2.0 * mu / 3.0;
    let coef = 2.0 * mu * theta;
    let mut d = [[0.0_f64; 6]; 6];
    // K·1⊗1 on the normal (top-left 3×3) block.
    for row in d.iter_mut().take(3) {
        for e in row.iter_mut().take(3) {
            *e += k;
        }
    }
    // 2μθ · I_dev (engineering: normal block ⅔/−⅓, shear diagonal ½).
    for (i, row) in d.iter_mut().enumerate().take(3) {
        for (j, e) in row.iter_mut().enumerate().take(3) {
            *e += coef * if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 };
        }
    }
    for i in 3..6 {
        d[i][i] += coef * 0.5;
    }
    // − 2μθ̄ · n̂⊗n̂, with `n̂` unit in the Frobenius sense (off-diagonals of the
    // deviator counted twice, as `s:s` does).
    let s = deviator(trial);
    let s_norm =
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
            .sqrt();
    if s_norm > 0.0 {
        let nv: [f64; 6] = std::array::from_fn(|i| s[i] / s_norm);
        let c = 2.0 * mu * theta_bar;
        for i in 0..6 {
            for j in 0..6 {
                d[i][j] -= c * nv[i] * nv[j];
            }
        }
    }
    d
}
