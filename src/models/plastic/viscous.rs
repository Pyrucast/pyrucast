//! Rate-**dependent** laws — creep and viscoplasticity.
//!
//! A rate-independent law yields *instantly* when the stress reaches its
//! surface. A viscous one does not: the stress may sit **outside** the surface,
//! and the plastic flow that brings it back takes time. The overstress drives
//! the rate:
//!
//! ```text
//! ṗ = g(σ, p, …)
//! ```
//!
//! and the step is integrated implicitly, so the answer depends on `dt`. That is
//! why [`PlasticLaw::is_viscous`] exists and why these laws **error** without a
//! time increment: silently integrating a creep law as if it were instantaneous
//! would produce a plausible number and a wrong one.
//!
//! ## One solver for all of them
//!
//! Every law here reduces to a **scalar** equation in the plastic multiplier,
//!
//! ```text
//! R(Δp) = Δp − dt · g(q(Δp), p_A + Δp, …) = 0
//! ```
//!
//! because the flow stays radial (in the *shifted* deviatoric space when there
//! is a back stress). [`solve_multiplier`] does the Newton on it, with a
//! numerically differentiated residual and a bisection safety net — the rate
//! functions here are steep (`q^n` with `n` up to 20), and a bare Newton on them
//! diverges as readily as it converges.
//!
//! ## The five laws
//!
//! | law | rate | what it describes |
//! |---|---|---|
//! | Norton | `ṗ = (q/K)^n` | secondary (steady) creep |
//! | Lemaitre | `ṗ = (q/K)^N · p^(−M)` | primary creep, by strain hardening |
//! | Blackburn | `ṗ = ṗ_prim + B·sinh(βq)` | primary **and** secondary, with a saturating primary |
//! | Chaboche | `ṗ = ⟨(J(σ−X) − R − k)/K⟩^n` | viscoplasticity with kinematic + isotropic hardening |
//! | Lemaitre-Chaboche | the same, on `σ/(1−D)` | the above, plus ductile damage |
//!
//! The first three carry no back stress: creep laws describe a **monotonic**
//! régime where the Bauschinger effect a back stress models does not arise. The
//! last two do, which is exactly what lets them handle cyclic loading — and what
//! costs them seven extra internal variables.

use crate::error::{PyrucastError, Result};
use crate::models::plastic::{
    deviator, i1, require_positive, von_mises_stress, MatParams, PlasticLaw, PlasticStep, PrevState,
};

/// Solve `R(Δp) = Δp − dt·rate(Δp) = 0` for the plastic multiplier.
///
/// Newton with a numerical derivative, bracketed and backed by bisection. The
/// bracketing matters: a Norton exponent of 10 makes `rate` vary over decades
/// within one step, and an unguarded Newton either overshoots into a negative
/// multiplier or shoots off to infinity. Bisection cannot do either.
///
/// `rate(dp)` must return the flow rate that a multiplier `dp` would imply; it
/// is decreasing in `dp` (more flow relaxes the stress that drives it), which is
/// what makes the residual monotone and the bracket sound.
pub fn solve_multiplier(
    law: PlasticLaw,
    dt: f64,
    upper: f64,
    rate: impl Fn(f64) -> Result<f64>,
) -> Result<f64> {
    let residual = |dp: f64| -> Result<f64> { Ok(dp - dt * rate(dp)?) };
    // R(0) = −dt·rate(0) ≤ 0 when the law is flowing at all; R(upper) > 0 because
    // `upper` relaxes the whole overstress. So the root is bracketed.
    let (mut lo, mut hi) = (0.0_f64, upper);
    let r_lo = residual(lo)?;
    if r_lo >= 0.0 {
        return Ok(0.0); // not flowing
    }
    let mut r_hi = residual(hi)?;
    // Grow the bracket if the guess was too tight (a very stiff rate law).
    let mut grow = 0;
    while r_hi < 0.0 && grow < 60 {
        hi *= 2.0;
        r_hi = residual(hi)?;
        grow += 1;
    }
    if r_hi < 0.0 {
        return Err(PyrucastError::Message(format!(
            "plasticity ({law}): could not bracket the viscoplastic multiplier — the step may be \
             far too large for the rate law"
        )));
    }

    let mut dp = 0.5 * (lo + hi);
    for _ in 0..100 {
        let r = residual(dp)?;
        if r.abs() <= 1e-14 * (dp.abs() + 1.0) {
            return Ok(dp);
        }
        if r < 0.0 {
            lo = dp;
        } else {
            hi = dp;
        }
        // Newton, with a numerical slope; fall back to the bisection midpoint
        // whenever it would leave the bracket.
        let h = 1e-8 * (dp.abs() + 1e-12);
        let slope = (residual(dp + h)? - r) / h;
        let candidate = if slope.abs() > f64::MIN_POSITIVE {
            dp - r / slope
        } else {
            f64::NAN
        };
        dp = if candidate.is_finite() && candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    Ok(dp)
}

/// A **radial** viscous return: the flow direction is that of the trial
/// deviator, and only its magnitude is solved for.
///
/// Shared by the three creep laws, which have no back stress and therefore flow
/// along the trial deviator exactly.
fn radial_creep(
    law: PlasticLaw,
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: f64,
    rate: impl Fn(f64, f64) -> Result<f64>,
) -> Result<PlasticStep> {
    let q_tr = von_mises_stress(trial);
    if q_tr <= 0.0 {
        return Ok(PlasticStep::elastic(trial, prev));
    }
    // The whole deviator relaxed away is the largest multiplier that can make
    // sense; the solver grows the bracket if a stiffer law needs more.
    let upper = q_tr / (3.0 * mat.mu);
    let dp = solve_multiplier(law, dt, upper.max(1e-12), |dp| {
        rate((q_tr - 3.0 * mat.mu * dp).max(0.0), prev.p + dp)
    })?;
    Ok(scale_deviator(trial, prev, mat, dp, q_tr, Vec::new()))
}

/// Build the returned state from a solved multiplier, scaling the trial
/// deviator and leaving the hydrostatic part untouched (creep is isochoric).
fn scale_deviator(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dp: f64,
    q_tr: f64,
    vars: Vec<f64>,
) -> PlasticStep {
    let s_tr = deviator(trial);
    let mean = i1(trial) / 3.0;
    let scale = ((q_tr - 3.0 * mat.mu * dp) / q_tr).max(0.0);
    let factor = 1.5 * dp / q_tr;
    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    for i in 0..6 {
        let s_new = s_tr[i] * scale;
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        eps_p[i] += factor * s_tr[i];
    }
    PlasticStep {
        sigma,
        eps_p,
        p: prev.p + dp,
        vars,
    }
}

// ─── Norton ─────────────────────────────────────────────────────────────────

/// Norton-Odqvist secondary creep: `ṗ = (q/K)^n`.
///
/// The workhorse of steady-state creep. There is **no yield threshold**: any
/// stress creeps, however slowly, which is what distinguishes creep from
/// plasticity.
pub fn norton(trial: &[f64; 6], prev: &PrevState, mat: &MatParams, dt: f64) -> Result<PlasticStep> {
    let k = require_positive(PlasticLaw::CreepNorton, "K", mat.get("K")?)?;
    let n = mat.get("n")?;
    radial_creep(PlasticLaw::CreepNorton, trial, prev, mat, dt, |q, _p| {
        Ok((q / k).powf(n))
    })
}

// ─── Lemaitre ───────────────────────────────────────────────────────────────

/// Lemaitre primary creep, by **strain** hardening: `ṗ = (q/K)^N · p^(−M)`.
///
/// The accumulated strain itself slows the flow, which is what produces a
/// primary (decelerating) creep stage without any explicit time dependence — and
/// what makes the law usable under a varying load, where a time-hardening form
/// would be wrong.
///
/// `p` is floored at a tiny value so the very first step, where `p = 0` and the
/// rate would be infinite, starts from a finite one instead of a division by
/// zero. The floor is far below any meaningful strain.
pub fn lemaitre(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: f64,
) -> Result<PlasticStep> {
    let k = require_positive(PlasticLaw::CreepLemaitre, "K", mat.get("K")?)?;
    let n = mat.get("N")?;
    let m = mat.get("M")?;
    const FLOOR: f64 = 1e-12;
    radial_creep(PlasticLaw::CreepLemaitre, trial, prev, mat, dt, |q, p| {
        Ok((q / k).powf(n) * p.max(FLOOR).powf(-m))
    })
}

// ─── Blackburn ──────────────────────────────────────────────────────────────

/// Blackburn creep: a **saturating primary** stage plus a steady secondary one.
///
/// ```text
/// ṗ_prim = r · (ε_∞(q) − p_prim)        ε_∞(q) = A·sinh(α q)
/// ṗ      = ṗ_prim + B·sinh(β q)
/// ```
///
/// The primary strain approaches its asymptote `ε_∞` exponentially, so the
/// primary rate dies out on its own; the secondary term is what remains. The
/// `sinh` stress dependence is Blackburn's, and it is what lets one parameter
/// set span several decades of stress — a power law cannot.
///
/// The primary strain is tracked as its **own** internal variable rather than
/// inferred from the total: only then does the law integrate correctly under a
/// varying load, which is the whole reason to prefer a strain-based form to a
/// time-based one.
pub fn blackburn(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: f64,
) -> Result<PlasticStep> {
    let (a, alpha) = (mat.get("A_1")?, mat.get("alpha_1")?);
    let r = mat.get("r_1")?;
    let (b, beta) = (mat.get("B_s")?, mat.get("beta_s")?);
    let p_prim_a = prev.var(0);

    let q_tr = von_mises_stress(trial);
    if q_tr <= 0.0 {
        return Ok(PlasticStep::elastic(trial, prev));
    }
    // The primary state at the end of the step, implicit in `q`:
    //   p_prim = (p_prim(A) + dt·r·ε_∞(q)) / (1 + dt·r)
    let primary_end = |q: f64| (p_prim_a + dt * r * a * (alpha * q).sinh()) / (1.0 + dt * r);
    let upper = q_tr / (3.0 * mat.mu);
    let dp = solve_multiplier(PlasticLaw::CreepBlackburn, dt, upper.max(1e-12), |dp| {
        let q = (q_tr - 3.0 * mat.mu * dp).max(0.0);
        let prim_rate = r * (a * (alpha * q).sinh() - primary_end(q));
        Ok(prim_rate.max(0.0) + b * (beta * q).sinh())
    })?;
    let q_end = (q_tr - 3.0 * mat.mu * dp).max(0.0);
    Ok(scale_deviator(
        trial,
        prev,
        mat,
        dp,
        q_tr,
        vec![primary_end(q_end)],
    ))
}

// ─── Chaboche, and its damageable variant ───────────────────────────────────

/// Chaboche viscoplasticity — a Norton flow on the **shifted** overstress, with
/// Armstrong-Frederick kinematic hardening and saturating isotropic hardening.
///
/// ```text
/// f = J(σ − X) − R − k          ṗ = ⟨f/K⟩^n
/// Ẋ = (2/3)C ε̇_vp − γ X ṗ       Ṙ = b(Q − R) ṗ
/// ```
///
/// The back stress `X` is what makes this law usable under **cyclic** loading:
/// it translates the yield surface, so reverse yielding happens early — the
/// Bauschinger effect, which no isotropic law can produce. `γ` is what makes the
/// translation saturate rather than grow without bound.
///
/// `damage` turns it into the Lemaitre-Chaboche law: the stress driving the flow
/// becomes the **effective** one, `σ/(1−D)`, and the damage grows with the
/// plastic strain, `Ḋ = (Y/S)^s ṗ`. A material that damages flows faster, which
/// flows more damage — the coupling that produces tertiary creep and, eventually,
/// rupture at `D_c`.
///
/// ## The integration
///
/// The flow direction is **frozen at the trial** shifted deviator, which makes
/// the step radial in that space and reduces it to the same scalar equation as
/// the creep laws. Both hardening variables are then implicit in `Δp`:
///
/// ```text
/// X = (X_A + (2/3)C Δp n̂) / (1 + γΔp)          R = (R_A + b Q Δp) / (1 + b Δp)
/// ```
///
/// A fully implicit treatment would re-evaluate the direction, at the cost of a
/// tensor Newton; freezing it is the standard semi-implicit scheme, and its
/// error is second order in the step.
pub fn chaboche(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: f64,
    damage: bool,
) -> Result<PlasticStep> {
    let law = if damage {
        PlasticLaw::ViscoplasticLemaitreChaboche
    } else {
        PlasticLaw::ViscoplasticChaboche
    };
    let k0 = mat.get("k")?;
    let k_visc = require_positive(law, "K", mat.get("K")?)?;
    let n = mat.get("n")?;
    let (c, gamma) = (mat.get("C_1")?, mat.get("gamma_1")?);
    let (b, q_sat) = (mat.get("b")?, mat.get("Q")?);

    // State at A: the back stress (6), the isotropic drag (1), the damage (1).
    let x_a: [f64; 6] = std::array::from_fn(|i| prev.var(i));
    let r_a = prev.var(6);
    let d_a = if damage { prev.var(7) } else { 0.0 };

    // Everything is driven by the **effective** stress; with no damage the
    // division is by one and this is plain Chaboche.
    let one_minus_d = (1.0 - d_a).max(1e-6);
    let s_tr = deviator(trial);
    let shifted: [f64; 6] = std::array::from_fn(|i| s_tr[i] / one_minus_d - x_a[i]);
    let j_tr = von_mises_stress(&shifted);
    if j_tr <= 0.0 {
        return Ok(PlasticStep::elastic(trial, prev));
    }
    // The frozen flow direction, `n̂ = (3/2)(s − X)/J`.
    let dir: [f64; 6] = std::array::from_fn(|i| 1.5 * shifted[i] / j_tr);

    let r_end = |dp: f64| (r_a + b * q_sat * dp) / (1.0 + b * dp);
    // The back stress and the deviator at the end of the step, both explicit
    // once the direction is frozen.
    let x_end = |dp: f64| -> [f64; 6] {
        let scale = 1.0 / (1.0 + gamma * dp);
        std::array::from_fn(|i| (x_a[i] + (2.0 / 3.0) * c * dp * dir[i]) * scale)
    };
    // `J(σ̃ − X)` at the end of the step, computed on the **tensors** rather than
    // reduced to a scalar formula. The reduction is doable but delicate — the
    // back stress at A need not be parallel to the flow direction — and getting
    // it subtly wrong would be invisible. Building the tensor cannot be.
    let j_end = |dp: f64| -> f64 {
        let x = x_end(dp);
        let shifted_end: [f64; 6] =
            std::array::from_fn(|i| (s_tr[i] - 2.0 * mat.mu * dp * dir[i]) / one_minus_d - x[i]);
        von_mises_stress(&shifted_end)
    };
    let upper = j_tr * one_minus_d / (3.0 * mat.mu);
    let dp = solve_multiplier(law, dt, upper.max(1e-12), |dp| {
        let f = j_end(dp) - r_end(dp) - k0;
        Ok(if f > 0.0 { (f / k_visc).powf(n) } else { 0.0 })
    })?;

    // Update the state along the frozen direction.
    let x_new = x_end(dp);
    let mean = i1(trial) / 3.0;
    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    for i in 0..6 {
        let s_new = s_tr[i] - 2.0 * mat.mu * dp * dir[i];
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        eps_p[i] += dp * dir[i];
    }

    let mut vars: Vec<f64> = x_new.to_vec();
    vars.push(r_end(dp));
    if damage {
        // Lemaitre's damage law: the elastic energy release rate drives it.
        let s_par = require_positive(law, "S", mat.get("S")?)?;
        let s_exp = mat.get("s")?;
        let d_c = mat.get("D_c")?;
        // Y = σ̃_eq²·R_v/(2E); the triaxiality function R_v is taken at its
        // deviatoric value, the usual simplification for a proportional path.
        let sigma_eq = von_mises_stress(&sigma) / one_minus_d;
        let young = mat.mu * (3.0 * mat.lambda + 2.0 * mat.mu) / (mat.lambda + mat.mu);
        let y = sigma_eq * sigma_eq / (2.0 * young);
        let d_new = (d_a + (y / s_par).powf(s_exp) * dp).min(d_c.max(0.0));
        vars.push(d_new);
    }
    Ok(PlasticStep {
        sigma,
        eps_p,
        p: prev.p + dp,
        vars,
    })
}
