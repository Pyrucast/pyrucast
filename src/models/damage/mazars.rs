//! Mazars isotropic damage — the classical concrete law.
//!
//! Damage is driven by an **equivalent strain** built from the positive
//! principal strains, `ε̃ = √(Σ⟨ε_I⟩₊²)`, so it responds to extension and is
//! blind to hydrostatic compression. The stress is the effective one, degraded:
//! `σ = (1 − D)·C:ε`.
//!
//! Two branches, tension and compression, are blended by the weights `α_t`,
//! `α_c` derived from the split of the effective stress — which is what lets one
//! law describe a material an order of magnitude stronger in compression.
//!
//! The single history variable is `κ = max_t ε̃`: damage never heals.

use crate::error::Result;
use crate::models::damage::{elastic_stress, lame, pos, DamageUpdate, MatRead};
use nalgebra::Matrix3;

/// Material parameters of the Mazars model at one Gauss point.
pub struct MazarsParams {
    e: f64,
    nu: f64,
    eps_d0: f64,
    a_t: f64,
    b_t: f64,
    a_c: f64,
    b_c: f64,
}

/// One damage branch `D = 1 − eps_d0(1−A)/κ − A / exp(B (κ − eps_d0))`,
/// clamped to `[0, 1)`.
fn damage_branch(kappa: f64, eps_d0: f64, a: f64, b: f64) -> f64 {
    let d = 1.0 - eps_d0 * (1.0 - a) / kappa - a / (b * (kappa - eps_d0)).exp();
    d.clamp(0.0, 1.0 - 1e-12)
}

/// Mazars point update. Returns `(stress, damage, kappa)` (stress full 3-D Voigt).
fn mazars_update(eps: &[f64; 6], kappa_old: f64, p: &MazarsParams) -> ([f64; 6], f64, f64) {
    let (lambda, mu) = lame(p.e, p.nu);
    let sigma_eff = elastic_stress(eps, lambda, mu);

    // Principal strains (coaxial with the effective stress, isotropic elasticity).
    let tensor = Matrix3::new(
        eps[0], eps[5], eps[4], // [εxx, εxy, εxz]
        eps[5], eps[1], eps[3], // [εxy, εyy, εyz]
        eps[4], eps[3], eps[2], // [εxz, εyz, εzz]
    );
    let e_pr = tensor.symmetric_eigenvalues();

    // Equivalent strain ε̃ = √(Σ ⟨ε_I⟩₊²).
    let eps_eq = (e_pr.iter().map(|&x| pos(x).powi(2)).sum::<f64>()).sqrt();

    // History variable: never below the threshold, never decreasing.
    let kappa = kappa_old.max(p.eps_d0).max(eps_eq);
    if kappa <= p.eps_d0 {
        return (sigma_eff, 0.0, kappa); // undamaged
    }

    // Tension/compression split of the effective principal stresses
    // σ̃_I = λ·tr + 2μ·ε_I, then strains induced by each part via the
    // isotropic compliance (all coaxial ⇒ work in principal space).
    let tr = e_pr[0] + e_pr[1] + e_pr[2];
    let st: [f64; 3] = std::array::from_fn(|i| lambda * tr + 2.0 * mu * e_pr[i]);
    let stp: [f64; 3] = std::array::from_fn(|i| pos(st[i]));
    let stn: [f64; 3] = std::array::from_fn(|i| st[i].min(0.0));
    let sum_p: f64 = stp.iter().sum();
    let sum_n: f64 = stn.iter().sum();
    // ε^t_I = [(1+ν)σ̃⁺_I − ν Σσ̃⁺] / E ; ε^c_I likewise from σ̃⁻.
    let eps_t: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stp[i] - p.nu * sum_p) / p.e);
    let eps_c: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stn[i] - p.nu * sum_n) / p.e);

    let denom = eps_eq * eps_eq;
    let mut alpha_t = 0.0;
    let mut alpha_c = 0.0;
    if denom > 0.0 {
        for i in 0..3 {
            let w = pos(e_pr[i]);
            alpha_t += pos(eps_t[i]) * w;
            alpha_c += pos(eps_c[i]) * w;
        }
        alpha_t /= denom;
        alpha_c /= denom;
    }
    let alpha_t = alpha_t.clamp(0.0, 1.0);
    let alpha_c = alpha_c.clamp(0.0, 1.0);

    let d_t = damage_branch(kappa, p.eps_d0, p.a_t, p.b_t);
    let d_c = damage_branch(kappa, p.eps_d0, p.a_c, p.b_c);
    // β fixed to 1 (no shear correction).
    let damage = (alpha_t * d_t + alpha_c * d_c).clamp(0.0, 1.0 - 1e-12);

    let sigma: [f64; 6] = std::array::from_fn(|i| (1.0 - damage) * sigma_eff[i]);
    (sigma, damage, kappa)
}

/// The law's material contract and its history variable.
pub const MATERIAL: &[&str] = &["E", "nu", "eps_d0", "A_t", "B_t", "A_c", "B_c"];

/// One Mazars step: `(σ, D, κ)` from the strain and the previous history.
pub fn update(eps: &[f64; 6], prev: &[f64], mat: &MatRead) -> Result<DamageUpdate> {
    let p = MazarsParams {
        e: mat.get("E")?,
        nu: mat.get("nu")?,
        eps_d0: mat.get("eps_d0")?,
        a_t: mat.get("A_t")?,
        b_t: mat.get("B_t")?,
        a_c: mat.get("A_c")?,
        b_c: mat.get("B_c")?,
    };
    let kappa_old = prev.first().copied().unwrap_or(0.0);
    let (sigma, damage, kappa) = mazars_update(eps, kappa_old, &p);
    Ok(DamageUpdate {
        sigma,
        damage,
        vars: vec![kappa],
    })
}
