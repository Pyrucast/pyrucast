//! Voigt nomenclature of the continuum: which strain and tangent components a
//! law reads, in which order, and how to lift them out of a Gauss-point row.
//!
//! Engineering Voigt throughout — shear strains are doubled (`γ = 2ε`), which is
//! what a `D` matrix in this convention expects:
//!
//! | kinematics | Voigt vector |
//! |---|---|
//! | plane stress / plane strain | `[εxx, εyy, γxy]` |
//! | axisymmetric | `[εrr, εzz, εθθ, γrz]`, named `[εxx, εyy, εzz, γxy]` |
//! | solid | `[εxx, εyy, εzz, γyz, γxz, γxy]` |
//!
//! Nothing here allocates at a Gauss point: the readers take the field's own row
//! and a table of positions resolved once for the zone.

use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::owned_components;
use crate::models::tensor::Kinematics;

/// Six lignes de Voigt sur la largeur d'une maille — la forme des tampons `B` et
/// `D·B`. Sur la **pile** : un point de Gauss n'alloue rien, et six `Vec` par
/// point, c'est ce que coûtait la version précédente.
pub(crate) type VoigtRows = [[f64; MAX_CELL_DOFS]; 6];

/// Voigt component count: 3 in 2-D plane, **4** axisymmetric (the hoop joins
/// them), 6 in 3-D.
pub(crate) fn voigt_size(space_dim: usize, kinematics: Kinematics) -> usize {
    match (space_dim, kinematics) {
        (2, Kinematics::Axisymmetric) => 4,
        (2, _) => 3,
        _ => 6,
    }
}

/// The strain components a continuum law reads, **in Voigt order** — the
/// convention its indices assume, declared for
/// [`crate::models::Behavior::deformation_reads`].
///
/// Axisymmetry is the odd one: its fourth slot is the *measured* hoop `eps_zz`,
/// produced by
/// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation),
/// not an assumption.
pub(crate) fn strain_reads(space_dim: usize, kinematics: Kinematics) -> Vec<String> {
    let names: &[&str] = if space_dim == 2 && kinematics.is_axisymmetric() {
        &["eps_xx", "eps_yy", "eps_zz", "eps_xy"]
    } else if space_dim == 2 {
        &["eps_xx", "eps_yy", "eps_xy"]
    } else {
        &["eps_xx", "eps_yy", "eps_zz", "eps_yz", "eps_xz", "eps_xy"]
    };
    owned_components(names)
}

/// How many **normal** components open the Voigt order: the ones the
/// engineering convention leaves alone. The shears that follow are doubled
/// (`γ = 2ε`), which is what a `D` matrix in engineering Voigt expects.
pub(crate) fn normal_count(space_dim: usize, kinematics: Kinematics) -> usize {
    if space_dim == 2 && kinematics.is_axisymmetric() {
        3
    } else if space_dim == 2 {
        2
    } else {
        3
    }
}

/// Read the engineering-Voigt strain of one Gauss point out of its row.
///
/// No name, no allocation: the row is the field's own buffer and `idx` says
/// where each component sits, resolved once for the zone.
pub(crate) fn read_voigt_strain(
    deformation: &[f64],
    idx: &[u32],
    space_dim: usize,
    kinematics: Kinematics,
) -> [f64; 6] {
    let mut eps = [0.0_f64; 6];
    let n = normal_count(space_dim, kinematics);
    for (r, &i) in idx.iter().enumerate() {
        let v = deformation[i as usize];
        eps[r] = if r < n { v } else { 2.0 * v };
    }
    eps
}

/// Strain-displacement matrix `B` (Voigt) from `∂N_i/∂x_a` (`dn_dx`, layout
/// `[i*space_dim + a]`). Shape `voigt_size × (space_dim·nodes)`, node-major
/// columns (matching [`crate::containers::matrix::DofOrdering::NodesThenVars`]).
///
/// `hoop` carries the axisymmetric extra: `Some((N, r))` — the shape values and
/// the radius at the Gauss point — adds the fourth row `ε_θθ = Σ_i N_i u_{r,i} / r`
/// and orders the rows `[rr, zz, θθ, rz]`. `None` gives the plane / solid `B`.
pub(crate) fn b_matrix_into(
    dn_dx: &[f64],
    n_nodes: usize,
    space_dim: usize,
    hoop: Option<(&[f64], f64)>,
    b: &mut VoigtRows,
) -> usize {
    let v = match hoop {
        Some(_) => 4,
        None => voigt_size(space_dim, Kinematics::PlaneStrain),
    };
    let dofs = space_dim * n_nodes;
    for row in b[..v].iter_mut() {
        row[..dofs].fill(0.0);
    }
    let dn = |i: usize, a: usize| dn_dx[i * space_dim + a];
    for i in 0..n_nodes {
        if let Some((n, r)) = hoop {
            let (cr, cz) = (2 * i, 2 * i + 1);
            b[0][cr] = dn(i, 0); // εrr
            b[1][cz] = dn(i, 1); // εzz
            b[2][cr] = n[i] / r; // εθθ = u_r / r
            b[3][cr] = dn(i, 1); // γrz
            b[3][cz] = dn(i, 0);
        } else if space_dim == 2 {
            let (cx, cy) = (2 * i, 2 * i + 1);
            b[0][cx] = dn(i, 0); // εxx
            b[1][cy] = dn(i, 1); // εyy
            b[2][cx] = dn(i, 1); // γxy
            b[2][cy] = dn(i, 0);
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
