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
use crate::models::shell::{local_derivatives, local_frame, to_global};
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

/// Weight of the drilling constraint, relative to `G·h`.
///
/// Small enough not to stiffen the shell, large enough to remove the
/// singularity. The constraint it weights is physical (`θ_z` should follow the
/// membrane rotation), so the answer is insensitive to the exact value over
/// several decades — which is what one wants from a regularisation.
const DRILLING_WEIGHT: f64 = 1e-3;

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
    let dm = membrane_law(e, nu, h);
    let db = bending_law(e, nu, h);
    let ds = shear_law(e, nu, h, shear_factor(material, cell));
    let g_mod = e / (2.0 * (1.0 + nu));

    let frame = local_frame(full)?;
    let mut local = vec![vec![0.0_f64; side]; side];

    // ── Membrane, bending and drilling: full quadrature ────────────────────
    for g in 0..full.n_gauss {
        let dn = local_derivatives(full, &frame, g)?;
        let shape = full.n_at_g(g)?;
        let w = full.det_j_w(g)?;

        // Membrane `ε` on (u, v) — local DOFs 6i+0, 6i+1.
        let mut bm = vec![vec![0.0; side]; 3];
        // Bending `κ` on (θ_x, θ_y) — local DOFs 6i+3, 6i+4.
        let mut bb = vec![vec![0.0; side]; 3];
        // Drilling residual `θ_z − ω_z` on (u, v, θ_z).
        let mut bd = vec![0.0; side];
        for i in 0..n {
            let (dx, dy) = (dn[i][0], dn[i][1]);
            let (u, v, tx, ty, tz) = (6 * i, 6 * i + 1, 6 * i + 3, 6 * i + 4, 6 * i + 5);
            bm[0][u] = dx;
            bm[1][v] = dy;
            bm[2][u] = dy;
            bm[2][v] = dx;

            bb[0][ty] = dx;
            bb[1][tx] = -dy;
            bb[2][ty] = dy;
            bb[2][tx] = -dx;

            // ω_z = ½(∂v/∂x − ∂u/∂y), so the residual picks up its negative.
            bd[u] = 0.5 * dy;
            bd[v] = -0.5 * dx;
            bd[tz] = shape[i];
        }
        accumulate(&mut local, &bm, &dm, w, side);
        accumulate(&mut local, &bb, &db, w, side);
        // The drilling constraint is a scalar: its « law » is one coefficient.
        let kd = DRILLING_WEIGHT * g_mod * h * w;
        for a in 0..side {
            if bd[a] == 0.0 {
                continue;
            }
            for b in 0..side {
                local[a][b] += kd * bd[a] * bd[b];
            }
        }
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

/// `local += Bᵀ D B · w` for a 3-component strain.
fn accumulate(local: &mut [Vec<f64>], b: &[Vec<f64>], d: &[[f64; 3]; 3], w: f64, side: usize) {
    // `D B` first: three rows, so the inner loop stays short and the intermediate
    // is what a reader can check against the law above.
    let mut db = vec![vec![0.0; side]; 3];
    for (r, row) in db.iter_mut().enumerate() {
        for (c, e) in row.iter_mut().enumerate() {
            *e = (0..3).map(|k| d[r][k] * b[k][c]).sum();
        }
    }
    for a in 0..side {
        let contributes = (0..3).any(|k| b[k][a] != 0.0);
        if !contributes {
            continue;
        }
        for bcol in 0..side {
            let acc: f64 = (0..3).map(|k| b[k][a] * db[k][bcol]).sum();
            local[a][bcol] += acc * w;
        }
    }
}
