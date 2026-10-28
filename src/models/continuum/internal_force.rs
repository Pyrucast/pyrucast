//! Internal forces of the continuum, `f = ∫ Bᵀ σ dΩ` — the `Bᵀ` side of the same
//! `B` the stiffness integrates.
//!
//! In large transformations the same loop runs on the reference configuration
//! with the **first Piola–Kirchhoff** stress `P = F S` in place of `σ`:
//! `f_ia = ∫ P_ab ∂N_i/∂X_b dV₀`, the transpose of `δE = sym(Fᵀ ∇₀δu)`. `P` is
//! not symmetric, so it is read whole — the only difference.
//!
//! It sits in the continuum module because it is a property of the
//! **modelling**, not of any law: elasticity, plasticity and damage all produce
//! a Voigt-named Cauchy stress, and the same kernel turns any of them into nodal
//! forces. It is also, read without any mechanics, the divergence of a
//! symmetric tensor — one weak divergence per row — which is why
//! [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)
//! drives it too, with no sub-model at all. Hence free functions.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::error::Result;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::CellGeom;

/// Axis suffixes used by the continuum-mechanics internal-force kernel to read
/// Voigt-named stress components (`sigma_xx`, `sigma_xy`, …).
const VOIGT_AXES: [&str; 3] = ["x", "y", "z"];

/// Continuum-mechanics internal-force element kernel `f_{i,a} = Σ_g Σ_b
/// (∂N_i/∂x_b) σ_ab |J| w` — one [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)
/// per row of the symmetric stress tensor `σ` (read in Voigt naming). Backs both
/// the [`crate::models::SubModelKind::internal_force_element`] default
/// (elasticity, Mazars, plasticity) and the model-free
/// [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)
/// tensor case. Fills
/// `fe` node-major / axis-minor (`fe[i * space_dim + a]`).
///
/// On an **axisymmetric** geometry the radial row gains the hoop term
/// `f_{i,r} += (N_i / r) σ_θθ` — the transpose of the `N_i / r` row the
/// strain-displacement matrix `B` carries there, so `∫ Bᵀσ` keeps matching `K·u`
/// for a linear law.
pub(crate) fn continuum_internal_force_element(
    geoms: &[CellGeom],
    stress: &SubElementField,
    lay: &[u32],
    fe: &mut [f64],
) -> Result<()> {
    tensor_internal_force(geoms, stress, lay, fe, voigt_stress_matrix)
}

/// The same kernel on a **non-symmetric** tensor read whole — the first
/// Piola–Kirchhoff `P = F S` of a law in large transformations, whose
/// `f_ia = Σ_g Σ_b (∂N_i/∂X_b) P_ab |J₀| w` is the transpose of
/// `δE = sym(Fᵀ ∇₀δu)` on the reference configuration. `lay` follows
/// [`full_matrix_reads`], closed by the hoop `P_θθ` on a body of revolution.
pub(crate) fn continuum_internal_force_element_full(
    geoms: &[CellGeom],
    stress: &SubElementField,
    lay: &[u32],
    fe: &mut [f64],
) -> Result<()> {
    tensor_internal_force(geoms, stress, lay, fe, full_stress_matrix)
}

/// The loop both kernels share; `read` fills the `d × d` tensor of a point from
/// its row. Generic, so each kernel is compiled with its own reader and the
/// Gauss loop tests nothing.
fn tensor_internal_force(
    geoms: &[CellGeom],
    stress: &SubElementField,
    lay: &[u32],
    fe: &mut [f64],
    read: impl Fn(&[f64], &[u32], usize, &mut [f64; 9]),
) -> Result<()> {
    let geom = &geoms[0];
    let d = geom.space_dim;
    let n_nodes = geom.n_nodes;
    let stride = stress.component_count();
    let values = stress.values();
    let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
    for g in 0..geom.n_gauss {
        let dn = &mut dn_buf[..n_nodes * d]; // [i * d + b]
        geom.dn_dx(g, dn)?;
        let w = geom.det_j_w(g);
        let start = (geom.cell * geom.n_gauss + g) * stride;
        let row = &values[start..start + stride];
        let mut sig = [0.0_f64; 9]; // [a * d + b]
        read(row, lay, d, &mut sig);
        // `zz` is the hoop component and only exists on a body of revolution.
        let hoop = if geom.axisymmetric {
            Some((
                geom.n_at_g(g),
                // The hoop closes the continuum's read list.
                row[lay[lay.len() - 1] as usize] / geom.radius(g),
            ))
        } else {
            None
        };
        for i in 0..n_nodes {
            for a in 0..d {
                let mut s = 0.0;
                for b in 0..d {
                    s += dn[i * d + b] * sig[a * d + b];
                }
                fe[i * d + a] += s * w;
            }
            if let Some((n, s_hoop)) = hoop {
                fe[i * d] += n[i] * s_hoop * w;
            }
        }
    }
    Ok(())
}

/// The Voigt component names of a symmetric `d×d` tensor under a given prefix —
/// `sigma_xx`, `sigma_yy`, `sigma_xy` for a stress, `a_xx`, … for anything else.
/// Backs the continuum-mechanics
/// [`crate::models::SubModelKind::internal_force_element`] default and the
/// tensor case of [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence);
/// read by component *name*, so a state field carrying extra `VAR1` components
/// (Mazars) is handled transparently.
pub(crate) fn voigt_matrix_reads(prefix: &str, space_dim: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(space_dim * (space_dim + 1) / 2);
    for i in 0..space_dim {
        for j in i..space_dim {
            names.push(format!("{prefix}_{}{}", VOIGT_AXES[i], VOIGT_AXES[j]));
        }
    }
    names
}

/// The symmetric stress tensor of one Gauss point, from its row.
///
/// `idx` gives the position of each [`voigt_matrix_reads`] name, resolved once
/// per zone; the tensor lands in a caller-owned `d × d` buffer. Neither the
/// names nor the buffer are built here: this runs once per Gauss point of every
/// cell, and it used to spend a `format!` per component doing it.
pub(crate) fn voigt_stress_matrix(row: &[f64], idx: &[u32], d: usize, sig: &mut [f64; 9]) {
    let mut k = 0;
    for i in 0..d {
        for j in i..d {
            let v = row[idx[k] as usize];
            k += 1;
            sig[i * d + j] = v;
            sig[j * d + i] = v; // symmetric
        }
    }
}

/// The component names of a **full** `d×d` tensor under `prefix`, row-major —
/// `P_xx`, `P_xy`, `P_yx`, `P_yy` in 2-D — the layout
/// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
/// gives the deformation gradient `F`, and the one a non-symmetric stress takes.
pub(crate) fn full_matrix_reads(prefix: &str, space_dim: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(space_dim * space_dim);
    for i in 0..space_dim {
        for j in 0..space_dim {
            names.push(format!("{prefix}_{}{}", VOIGT_AXES[i], VOIGT_AXES[j]));
        }
    }
    names
}

/// The full tensor of one Gauss point, from its row — the non-symmetric
/// counterpart of [`voigt_stress_matrix`]; `idx` follows [`full_matrix_reads`].
pub(crate) fn full_stress_matrix(row: &[f64], idx: &[u32], d: usize, sig: &mut [f64; 9]) {
    for k in 0..d * d {
        sig[k] = row[idx[k] as usize];
    }
}
