//! Drucker-Prager plasticity — pressure-sensitive, with non-associated flow.
//!
//! Soils, rocks, concrete and powders are **stronger in compression than in
//! tension**: their yielding depends on the hydrostatic pressure, which von
//! Mises ignores entirely. Drucker-Prager is the simplest surface that captures
//! it — a cone about the hydrostatic axis:
//!
//! ```text
//! f(σ) = q + α·I₁ − k          q = √(3 J₂),  I₁ = tr σ
//! ```
//!
//! `α` is the friction coefficient (the cone's slope, `α = 0` giving back von
//! Mises) and `k` the cohesion.
//!
//! ## Why the flow is non-associated
//!
//! Associated flow on this cone (`g = f`) would make the material dilate under
//! shear by exactly the amount its friction implies — which for real granular
//! media is far too much. So the plastic potential carries its **own** slope,
//! the dilatancy `ψ`:
//!
//! ```text
//! g(σ) = q + ψ·I₁          ψ ≤ α
//! ```
//!
//! `ψ = α` recovers associated flow; `ψ = 0` gives isochoric plastic flow with
//! frictional strength. The price is a **non-symmetric** consistent tangent —
//! which the assembler already supports, `MatrixLayout` carrying a `symmetric`
//! flag.
//!
//! ## The apex
//!
//! A cone has a tip, at `I₁ = k/α`, and a trial stress beyond it returns to that
//! point rather than to the cone's flank — the smooth return would otherwise
//! overshoot into a stress state with a negative equivalent stress, which has no
//! meaning. Detecting it is the one branch this law needs, and it is exactly the
//! case a naive implementation gets wrong under strong tension.

use crate::error::Result;
use crate::models::plastic::{
    deviator, elastic_tangent, i1, require_positive, von_mises_stress, MatParams, PlasticLaw,
    PrevState,
};

/// Read `(α, k, ψ)` and reject a non-physical set.
fn params(mat: &MatParams) -> Result<(f64, f64, f64)> {
    let alpha = mat.get("alpha")?;
    let k = require_positive(PlasticLaw::DruckerPrager, "k", mat.get("k")?)?;
    let psi = mat.get("psi")?;
    Ok((alpha, k, psi))
}

/// Return onto the Drucker-Prager cone, with the apex case handled separately.
pub fn return_map(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
) -> Result<([f64; 6], [f64; 6], f64)> {
    let (alpha, k, psi) = params(mat)?;
    let (mu, bulk) = (mat.mu, mat.bulk());
    let q_tr = von_mises_stress(trial);
    let i1_tr = i1(trial);
    let f = q_tr + alpha * i1_tr - k;
    if f <= 0.0 {
        return Ok((*trial, prev.eps_p, prev.p)); // elastic
    }

    // Smooth (flank) return. The flow direction ∂g/∂σ = (3/2)s/q + ψI, and the
    // elastic operator maps it to 3μ·s/q on the deviator and 9Kψ on the trace,
    // so consistency is linear in the multiplier — no iteration.
    let denom = 3.0 * mu + 9.0 * bulk * alpha * psi;
    let dlambda = f / denom;
    let q_new = q_tr - 3.0 * mu * dlambda;

    if q_new >= 0.0 && q_tr > 0.0 {
        let s_tr = deviator(trial);
        let mean_new = (i1_tr - 9.0 * bulk * psi * dlambda) / 3.0;
        let scale = q_new / q_tr;
        let mut sigma = [0.0; 6];
        let mut eps_p = prev.eps_p;
        for i in 0..6 {
            let s_new = s_tr[i] * scale;
            sigma[i] = if i < 3 { s_new + mean_new } else { s_new };
            // Δε_p = Δλ·(∂g/∂σ): deviatoric part (3Δλ/2q)·s, plus the dilatant
            // volumetric part ψΔλ on the diagonal.
            eps_p[i] += 1.5 * dlambda / q_tr * s_tr[i];
            if i < 3 {
                eps_p[i] += psi * dlambda;
            }
        }
        return Ok((sigma, eps_p, prev.p + dlambda));
    }

    // Apex return: the flank solution would push the equivalent stress negative,
    // which is meaningless. The whole deviator is shed and the stress collapses
    // onto the tip `I₁ = k/α`.
    apex_return(trial, prev, mat, alpha, k)
}

/// Collapse onto the cone's tip: `s = 0`, `I₁ = k/α`.
///
/// With `α = 0` the cone is a cylinder and has no apex, so this cannot be
/// reached — the flank return always succeeds there (it *is* von Mises).
fn apex_return(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    alpha: f64,
    k: f64,
) -> Result<([f64; 6], [f64; 6], f64)> {
    let mean_new = if alpha.abs() > f64::EPSILON {
        k / alpha / 3.0
    } else {
        i1(trial) / 3.0
    };
    let s_tr = deviator(trial);
    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    // Everything the elastic predictor built beyond the apex becomes plastic.
    let vol_drop = (i1(trial) - 3.0 * mean_new) / (9.0 * mat.bulk());
    for i in 0..6 {
        sigma[i] = if i < 3 { mean_new } else { 0.0 };
        eps_p[i] += s_tr[i] / (2.0 * mat.mu);
        if i < 3 {
            eps_p[i] += vol_drop;
        }
    }
    // The equivalent plastic strain shed in this step, in the same measure the
    // flank return uses (the multiplier of the deviatoric flow).
    let dp = von_mises_stress(trial) / (3.0 * mat.mu);
    Ok((sigma, eps_p, prev.p + dp))
}

// ─── On the tangent ─────────────────────────────────────────────────────────
//
// Drucker-Prager has a **closed-form return** (above) but takes its consistent
// tangent by finite differences, from
// [`crate::models::plastic::consistent_tangent`].
//
// That is a deliberate reversal of an earlier hand derivation. Non-associated
// flow makes `∂σ/∂ε` pick up an `m⊗n` term with `m ≠ n`, and getting its
// engineering-Voigt weighting right is fiddly: the derivation that *looked*
// correct was 24 % off, and only the finite-difference oracle in
// `tests/plastic_laws.rs` said so. A numerical tangent cannot be mis-derived,
// costs twelve evaluations of a closed-form update, and leaves Newton's
// quadratic convergence intact.
//
// The **apex** is the one case the numerical route still needs told about: the
// return pins the stress there, so `∂σ/∂ε` genuinely vanishes and the finite
// difference finds zero of its own accord. A body entirely at its apex therefore
// assembles a singular tangent — not an artefact, but the honest report that
// such a material carries no further load.
