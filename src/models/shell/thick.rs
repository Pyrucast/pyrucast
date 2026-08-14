//! Reissner-Mindlin — the thick-shell formulation.
//!
//! The normal fibre stays straight but **not** normal: its rotation is an
//! independent field, so the transverse shear `γ = ∇w + θ` is a strain of its
//! own rather than something forced to zero. That is what makes the element work
//! for a thick shell, and what makes it need care for a thin one.
//!
//! ## The three strains
//!
//! In the local frame, with the transverse deflection `w` and the fibre
//! rotations `θ_x`, `θ_y`:
//!
//! ```text
//! membrane   ε = [∂u/∂x, ∂v/∂y, ∂u/∂y + ∂v/∂x]
//! bending    κ = [∂θ_y/∂x, −∂θ_x/∂y, ∂θ_y/∂y − ∂θ_x/∂x]
//! shear      γ = [∂w/∂x + θ_y, ∂w/∂y − θ_x]
//! ```
//!
//! and the laws that carry them, for a homogeneous section:
//!
//! ```text
//! D_m = Eh/(1−ν²)·[[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]]
//! D_b = D_m · h²/12
//! D_s = k_s·G·h                        k_s = 5/6
//! ```
//!
//! The bending law is the membrane one scaled by `h²/12`, which is the whole
//! content of « plane sections »: the same plane-stress material, integrated
//! across the thickness with a `z²` weight.
//!
//! ## Why the shear is integrated reduced
//!
//! As the shell thins, `D_s` (linear in `h`) overwhelms `D_b` (cubic in `h`) by
//! `1/h²`. Integrated at full quadrature, the shear term then imposes `γ = 0`
//! **pointwise**, which a linear element can only satisfy by refusing to bend at
//! all: the deflection collapses towards zero and no mesh refinement recovers it.
//! That is **shear locking**.
//!
//! Integrating the shear at a single point relaxes the constraint to a mean, the
//! element bends, and the answer converges. It is the same cure, and the same
//! mechanism, as the [Timoshenko beam](crate::models::timoshenko) — which is why
//! the two share the multi-quadrature layout rather than each inventing one.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::error::Result;
use crate::models::shell::{
    accumulate, local_derivatives, local_frame, membrane_and_drilling, to_global,
};
use crate::models::CellGeom;

/// The membrane law `D_m` of a homogeneous section (plane stress × thickness).
pub fn membrane_law(e: f64, nu: f64, h: f64) -> [[f64; 3]; 3] {
    let c = e * h / (1.0 - nu * nu);
    [
        [c, c * nu, 0.0],
        [c * nu, c, 0.0],
        [0.0, 0.0, c * (1.0 - nu) / 2.0],
    ]
}

/// The bending law `D_b = D_m · h²/12` — the same material, weighted by `z²`
/// across the thickness.
pub fn bending_law(e: f64, nu: f64, h: f64) -> [[f64; 3]; 3] {
    let m = membrane_law(e, nu, h);
    let s = h * h / 12.0;
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] * s))
}

/// The transverse-shear modulus `k_s·G·h`.
pub fn shear_law(e: f64, nu: f64, h: f64, k_s: f64) -> f64 {
    k_s * e / (2.0 * (1.0 + nu)) * h
}

/// The shear-correction factor: the material's own `k_s` if it carries one,
/// `5/6` otherwise — the value for a homogeneous rectangular section.
pub fn shear_factor(material: &SubElementField, cell: usize) -> f64 {
    match material.component_index("k_s") {
        Some(_) => material.value(cell, 0, "k_s").unwrap_or(5.0 / 6.0),
        None => 5.0 / 6.0,
    }
}

/// The local element stiffness of one facet, carried to the global axes.
///
/// `full` carries the membrane and bending terms, `reduced` the transverse
/// shear — two [`CellGeom`] over the same cell, differing only by quadrature.
pub fn element_stiffness(
    full: &CellGeom,
    reduced: &CellGeom,
    material: &SubElementField,
    ke: &mut [f64],
) -> Result<()> {
    let n = full.n_nodes;
    let side = 6 * n;
    let cell = full.cell;
    let (e, nu, h) = (
        material.value(cell, 0, "E")?,
        material.value(cell, 0, "nu")?,
        material.value(cell, 0, "h")?,
    );
    let db = bending_law(e, nu, h);
    let ds = shear_law(e, nu, h, shear_factor(material, cell));

    let frame = local_frame(full)?;
    let mut local = vec![vec![0.0_f64; side]; side];

    // ── Membrane and drilling: the part shared with every formulation ──────
    membrane_and_drilling(full, &frame, e, nu, h, &mut local)?;

    // ── Bending: full quadrature, on the independent fibre rotation ────────
    for g in 0..full.n_gauss {
        let dn = local_derivatives(full, &frame, g)?;
        let w = full.det_j_w(g)?;

        // Bending `κ` on (θ_x, θ_y) — local DOFs 6i+3, 6i+4.
        let mut bb = vec![vec![0.0; side]; 3];
        for i in 0..n {
            let (dx, dy) = (dn[i][0], dn[i][1]);
            let (tx, ty) = (6 * i + 3, 6 * i + 4);
            bb[0][ty] = dx;
            bb[1][tx] = -dy;
            bb[2][ty] = dy;
            bb[2][tx] = -dx;
        }
        accumulate(&mut local, &bb, &db, w, side);
    }

    // ── Transverse shear: reduced quadrature, against locking ──────────────
    for g in 0..reduced.n_gauss {
        let dn = local_derivatives(reduced, &frame, g)?;
        let shape = reduced.n_at_g(g)?;
        let w = reduced.det_j_w(g)?;
        // `γ` on (w, θ_x, θ_y) — local DOFs 6i+2, 6i+3, 6i+4.
        let mut bs = vec![vec![0.0; side]; 2];
        for i in 0..n {
            let (dx, dy) = (dn[i][0], dn[i][1]);
            let (wz, tx, ty) = (6 * i + 2, 6 * i + 3, 6 * i + 4);
            bs[0][wz] = dx;
            bs[0][ty] = shape[i];
            bs[1][wz] = dy;
            bs[1][tx] = -shape[i];
        }
        for a in 0..side {
            for row in bs.iter() {
                if row[a] == 0.0 {
                    continue;
                }
                for b in 0..side {
                    local[a][b] += ds * row[a] * row[b] * w;
                }
            }
        }
    }

    to_global(&local, &frame, n, ke);
    Ok(())
}
