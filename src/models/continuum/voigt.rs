//! Voigt nomenclature of the continuum: which strain and tangent components a
//! law reads, in which order, and how to lift them out of a Gauss-point row.
//!
//! Engineering Voigt throughout — shear strains are doubled (`γ = 2ε`), which is
//! what a `D` matrix in this convention expects:
//!
//! | space | Voigt vector |
//! |---|---|
//! | 2-D (plane stress, plane strain, axisymmetric) | `[εxx, εyy, εzz, γxy]` |
//! | 3-D | `[εxx, εyy, εzz, γyz, γxz, γxy]` |
//!
//! In 2-D, `εzz` is the out-of-plane strain — zero from the displacement in
//! plane stress and plane strain — or the hoop `εθθ = u_r / r` on a body of
//! revolution (`[εrr, εzz, εθθ, γrz]`, named after the plane axes).
//!
//! Nothing here allocates at a Gauss point: the readers take the field's own row
//! and a table of positions resolved once for the zone.

use crate::models::kernel::MAX_CELL_DOFS;

/// Six Voigt rows across a cell's width — the shape of the `B` and `D·B`
/// buffers. On the **stack**: a Gauss point allocates nothing.
pub(crate) type VoigtRows = [[f64; MAX_CELL_DOFS]; 6];

/// The normal components lead every Voigt vector, three of them in 2-D as in
/// 3-D; the shears follow.
pub(crate) const NORMAL_COUNT: usize = 3;

/// Voigt component count: 4 in 2-D, 6 in 3-D.
pub(crate) fn voigt_size(space_dim: usize) -> usize {
    crate::models::tensor::stress_count(space_dim)
}

/// The strain components a continuum law reads, **in Voigt order** — the
/// convention its indices assume, declared for
/// [`crate::models::Behavior::deformation_reads`] — under `prefix`, that of the
/// strain measure the law reads (`eps`, `E` or `H`).
///
/// The 2-D `zz` is produced by
/// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation)
/// like the in-plane ones: zero in the plane, the measured hoop on a body of
/// revolution.
pub(crate) fn strain_reads(prefix: &str, space_dim: usize) -> Vec<String> {
    crate::models::tensor::voigt_names(prefix, space_dim)
}

/// Read the engineering-Voigt strain of one Gauss point out of its row.
///
/// No name, no allocation: the row is the field's own buffer and `idx` says
/// where each component sits, resolved once for the zone. The shears, after
/// the three normal components, are doubled.
pub(crate) fn read_voigt_strain(deformation: &[f64], idx: &[u32]) -> [f64; 6] {
    let mut eps = [0.0_f64; 6];
    for (r, &i) in idx.iter().enumerate() {
        let v = deformation[i as usize];
        eps[r] = if r < NORMAL_COUNT { v } else { 2.0 * v };
    }
    eps
}

/// Strain-displacement matrix `B` (Voigt) from `∂N_i/∂x_a` (`dn_dx`, layout
/// `[i*space_dim + a]`). Shape `voigt_size × (space_dim·nodes)`, node-major
/// columns (matching [`crate::containers::matrix::DofOrdering::NodesThenVars`]).
///
/// In 2-D the rows are `[xx, yy, zz, xy]`. `hoop` fills the `zz` row on a body
/// of revolution: `Some((N, r))` — the shape values and the radius at the
/// Gauss point — gives `ε_θθ = Σ_i N_i u_{r,i} / r`; `None` leaves it zero, the
/// displacement of a plane model having no out-of-plane part.
pub(crate) fn b_matrix_into(
    dn_dx: &[f64],
    n_nodes: usize,
    space_dim: usize,
    hoop: Option<(&[f64], f64)>,
    b: &mut VoigtRows,
) -> usize {
    let v = voigt_size(space_dim);
    let dofs = space_dim * n_nodes;
    for row in b[..v].iter_mut() {
        row[..dofs].fill(0.0);
    }
    let dn = |i: usize, a: usize| dn_dx[i * space_dim + a];
    for i in 0..n_nodes {
        if space_dim == 2 {
            let (cx, cy) = (2 * i, 2 * i + 1);
            b[0][cx] = dn(i, 0); // εxx (εrr)
            b[1][cy] = dn(i, 1); // εyy (εzz of the meridian plane)
            if let Some((n, r)) = hoop {
                b[2][cx] = n[i] / r; // εθθ = u_r / r
            }
            b[3][cx] = dn(i, 1); // γxy (γrz)
            b[3][cy] = dn(i, 0);
        } else {
            let (cx, cy, cz) = (3 * i, 3 * i + 1, 3 * i + 2);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cz] = dn(i, 2); // εzz
            b[3][cy] = dn(i, 2); // γyz
            b[3][cz] = dn(i, 1);
            b[4][cx] = dn(i, 2); // γxz
            b[4][cz] = dn(i, 0);
            b[5][cx] = dn(i, 1); // γxy
            b[5][cy] = dn(i, 0);
        }
    }
    v
}
