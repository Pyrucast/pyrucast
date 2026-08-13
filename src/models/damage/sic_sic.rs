//! SiC/SiC — **orthotropic** damage of a woven ceramic-matrix composite.
//!
//! A SiC/SiC composite is a silicon-carbide matrix reinforced by SiC fibre tows,
//! usually woven. It fails nothing like a metal or like concrete: the matrix
//! cracks first, in planes normal to the tow directions, while the fibres keep
//! carrying load across those cracks. The stiffness therefore falls **by
//! direction**, and by very different amounts in each.
//!
//! No isotropic damage variable can express that. This law carries **one damage
//! per material direction**:
//!
//! ```text
//! d_i = d_max,i · (1 − exp(−(⟨ε_i⟩₊ − ε_0,i)/ε_c,i))      for ⟨ε_i⟩₊ > ε_0,i
//! ```
//!
//! and degrades the orthotropic stiffness direction by direction. The positive
//! part is what matters: a matrix crack opens in **extension** and closes again
//! in compression, so a direction under compression is not degraded at all.
//!
//! ## The material frame is the weave
//!
//! The damage directions are the **material axes**, supplied exactly as for
//! [orthotropic elasticity](crate::models::symmetry) — by the vectors `V1`, `V2`
//! carried in the material field. That is not a coincidence: for a woven
//! composite they *are* the tow directions, and reusing the same frame means a
//! curved part (a wound tube, a shaped panel) gets its damage directions right
//! for free, cell by cell.
//!
//! ## Saturation, not failure
//!
//! Each `d_i` saturates at `d_max,i` rather than reaching one. That is the
//! physical statement that matrix cracking **does not** take the whole
//! stiffness: the fibres remain, and a saturated composite still carries load
//! along its tows. A law that let the damage reach one would predict a collapse
//! that does not happen.

use crate::error::Result;
use crate::models::damage::{lame, pos, DamageUpdate, MatRead};
use crate::models::symmetry;
use nalgebra::Matrix3;

/// The law's material contract: the elastic constants, then a threshold, a
/// characteristic strain and a saturation per direction — plus the material
/// frame, which the assembler resolves like any other component.
pub const MATERIAL_2D: &[&str] = &[
    "E", "nu", "eps_0_1", "eps_c_1", "d_max_1", "eps_0_2", "eps_c_2", "d_max_2", "eps_0_3",
    "eps_c_3", "d_max_3", "V1X", "V1Y",
];
/// The same, with the two axes a 3-D frame needs.
pub const MATERIAL_3D: &[&str] = &[
    "E", "nu", "eps_0_1", "eps_c_1", "d_max_1", "eps_0_2", "eps_c_2", "d_max_2", "eps_0_3",
    "eps_c_3", "d_max_3", "V1X", "V1Y", "V1Z", "V2X", "V2Y", "V2Z",
];

/// One SiC/SiC step.
///
/// `prev` carries the three history variables `κ_i = max_t ⟨ε_i⟩₊`, one per
/// material direction.
pub fn update(
    eps: &[f64; 6],
    prev: &[f64],
    mat: &MatRead,
    space_dim: usize,
) -> Result<DamageUpdate> {
    let e = mat.get("E")?;
    let nu = mat.get("nu")?;
    let (lambda, mu) = lame(e, nu);

    // The weave directions — the same frame an orthotropic elasticity would use.
    let r = symmetry::frame_rotation(mat.field, mat.cell, space_dim)?;

    // The strain in the material axes: `ε_mat = Rᵀ ε R`.
    let eps_global = Matrix3::new(
        eps[0], eps[5], eps[4], eps[5], eps[1], eps[3], eps[4], eps[3], eps[2],
    );
    let eps_mat = r.transpose() * eps_global * r;

    // One damage per direction, driven by the **positive** normal strain there:
    // a matrix crack opens in extension and closes in compression.
    let mut damages = [0.0_f64; 3];
    let mut kappas = [0.0_f64; 3];
    for i in 0..3 {
        let driver = pos(eps_mat[(i, i)]);
        let kappa = prev.get(i).copied().unwrap_or(0.0).max(driver);
        kappas[i] = kappa;
        let eps_0 = mat.get(&format!("eps_0_{}", i + 1))?;
        let eps_c = mat.get(&format!("eps_c_{}", i + 1))?.max(1e-30);
        let d_max = mat.get(&format!("d_max_{}", i + 1))?;
        damages[i] = if kappa > eps_0 {
            (d_max * (1.0 - (-(kappa - eps_0) / eps_c).exp())).clamp(0.0, 1.0 - 1e-12)
        } else {
            0.0
        };
    }

    // Degrade the stiffness **in the material axes**, direction by direction.
    // The normal block is scaled by `(1−d_i)(1−d_j)`, which keeps the operator
    // symmetric and degrades a coupling term as much as the weaker of the two
    // directions it couples; each shear takes the pair it shears.
    let tr = eps_mat[(0, 0)] + eps_mat[(1, 1)] + eps_mat[(2, 2)];
    let mut sigma_mat = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            let intact = if i == j {
                lambda * tr + 2.0 * mu * eps_mat[(i, i)]
            } else {
                2.0 * mu * eps_mat[(i, j)]
            };
            sigma_mat[(i, j)] = (1.0 - damages[i]).sqrt() * (1.0 - damages[j]).sqrt() * intact;
        }
    }

    // …then back to the global axes.
    let sigma_global = r * sigma_mat * r.transpose();
    let sigma = [
        sigma_global[(0, 0)],
        sigma_global[(1, 1)],
        sigma_global[(2, 2)],
        sigma_global[(1, 2)],
        sigma_global[(0, 2)],
        sigma_global[(0, 1)],
    ];

    Ok(DamageUpdate {
        sigma,
        // A scalar summary for visualisation; the state is the three below.
        damage: damages.iter().cloned().fold(0.0_f64, f64::max),
        vars: vec![
            kappas[0], kappas[1], kappas[2], damages[0], damages[1], damages[2],
        ],
    })
}
