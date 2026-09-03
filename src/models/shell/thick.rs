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
//! element bends, and the answer converges. The [Timoshenko
//! beam](crate::models::timoshenko) once met the same locking and was answered
//! the same way; it has since been replaced by an element that is *exact*, and
//! owns its interpolation rather than integrating one. So this is now the only
//! multi-quadrature element in the crate — the pattern is general, its user is
//! not.
//!
//! The alternative to relaxing the constraint is not to have one:
//! [discrete Kirchhoff](super::kirchhoff) drops the shear strain outright, and
//! has nothing left to lock.

use crate::containers::element_field::SubElementField;
use crate::containers::field::ABSENT_COMPONENT;
use crate::error::Result;
use crate::models::shell::{
    accumulate, accumulate_scalar, b_into, drilling_law, local_frame, shear_b_into, to_global,
    BendingSetup, ShellB, ShellMatrix, MAX_SHELL_DOFS, SHEAR_ROW, SHELL_STRAINS,
};
use crate::models::{CellGeom, ElementLayout};

/// The membrane law `D_m` of a homogeneous section (plane stress × thickness).
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// # use pyrucast::models::shell::thick;
/// // Contraintes planes × épaisseur : D_m est proportionnelle à h.
/// let a = thick::membrane_law(210_000.0, 0.3, 0.01);
/// let b = thick::membrane_law(210_000.0, 0.3, 0.02);
/// assert!((b[0][0] - 2.0 * a[0][0]).abs() < 1e-6);
/// ```
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
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// # use pyrucast::models::shell::thick;
/// // Le même matériau pondéré par z² sur l'épaisseur : D_b = D_m · h²/12.
/// let h = 0.01;
/// let m = thick::membrane_law(210_000.0, 0.3, h);
/// let b = thick::bending_law(210_000.0, 0.3, h);
/// assert!((b[0][0] - m[0][0] * h * h / 12.0).abs() < 1e-12);
/// ```
pub fn bending_law(e: f64, nu: f64, h: f64) -> [[f64; 3]; 3] {
    let m = membrane_law(e, nu, h);
    let s = h * h / 12.0;
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] * s))
}

/// The transverse-shear modulus `k_s·G·h`.
///
/// ```
/// # use pyrucast::models::shell::{self, ShellModel};
/// # use pyrucast::models::shell::thick;
/// // k_s·G·h, avec G = E / 2(1+ν).
/// let g = 210_000.0 / 2.6;
/// assert!((thick::shear_law(210_000.0, 0.3, 0.01, 5.0 / 6.0)
///          - 5.0 / 6.0 * g * 0.01).abs() < 1e-9);
/// ```
pub fn shear_law(e: f64, nu: f64, h: f64, k_s: f64) -> f64 {
    k_s * e / (2.0 * (1.0 + nu)) * h
}

/// The shear-correction factor: the material's own `k_s` if it carries one,
/// `5/6` otherwise — the value for a homogeneous rectangular section.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::{assemble_block, reduce_cells};
/// # use pyrucast::models::shell;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into(), "h".into()],
/// #     &[210_000.0, 0.3, 0.01]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(),
/// #                    "m_x".to_string(), "m_y".to_string(), "m_z".to_string()],
/// #               vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(),
/// #                    "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::shell::thick;
/// # use pyrucast::models::ElementLayout;
/// # use pyrucast::containers::field::ABSENT_COMPONENT;
/// // Sans `k_s` au matériau, la valeur d'une section rectangulaire
/// // homogène : 5/6. Le contrat facultatif d'une coque épaisse est
/// // `["rho", "k_s"]`, et ici les deux manquent.
/// let sans = ElementLayout {
///     material: vec![0, 1, 2],
///     optional_material: vec![ABSENT_COMPONENT, ABSENT_COMPONENT],
///     state: vec![],
/// };
/// assert!((thick::shear_factor(mat.read().point_values(0, 0)?, &sans) - 5.0 / 6.0).abs() < 1e-12);
/// // Avec, c'est celle du matériau qui l'emporte.
/// # let propre = SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into(), "h".into(), "k_s".into()],
/// #     &[210_000.0, 0.3, 0.01, 0.85])?;
/// let avec = ElementLayout {
///     material: vec![0, 1, 2],
///     optional_material: vec![ABSENT_COMPONENT, 3],
///     state: vec![],
/// };
/// assert!((thick::shear_factor(propre.point_values(0, 0)?, &avec) - 0.85).abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn shear_factor(row: &[f64], lay: &ElementLayout) -> f64 {
    match lay.optional_material[K_S_SLOT] {
        ABSENT_COMPONENT => 5.0 / 6.0,
        i => row[i as usize],
    }
}

/// Position of `k_s` in a thick shell's optional material (`["rho", "k_s"]`).
const K_S_SLOT: usize = 1;

/// The local element stiffness of one facet, carried to the global axes.
///
/// `full` carries the membrane and bending terms, `reduced` the transverse
/// shear — two [`CellGeom`] over the same cell, differing only by quadrature.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::kernel::{assemble_block, reduce_cells};
/// # use pyrucast::models::shell;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let mat = Handle::new(SubElementField::from_uniform_per_component(
/// #     zone.clone(), vec!["E".into(), "nu".into(), "h".into()],
/// #     &[210_000.0, 0.3, 0.01]).unwrap());
/// # let ddl = || (vec!["f_x".to_string(), "f_y".to_string(), "f_z".to_string(),
/// #                    "m_x".to_string(), "m_y".to_string(), "m_z".to_string()],
/// #               vec!["u_x".to_string(), "u_y".to_string(), "u_z".to_string(),
/// #                    "r_x".to_string(), "r_y".to_string(), "r_z".to_string()]);
/// # use pyrucast::models::shell::thick;
/// // Deux `CellGeom` : la quadrature **complète** pour la membrane et la
/// // flexion, la **réduite** pour le cisaillement transverse — c'est ce
/// // qui empêche le blocage en mince.
/// # use pyrucast::models::ElementLayout;
/// # use pyrucast::containers::field::ABSENT_COMPONENT;
/// // `E`, `nu`, `h` dans l'ordre du contrat ; ni `rho` ni `k_s` ici.
/// let lay = ElementLayout {
///     material: vec![0, 1, 2],
///     optional_material: vec![ABSENT_COMPONENT, ABSENT_COMPONENT],
///     state: vec![],
/// };
/// let (duals, primals) = ddl();
/// let bloc = assemble_block(
///     &[zone.clone(), zone.clone()], &support, &support, duals, primals,
///     DofOrdering::NodesThenVars, true, &mat, None,
///     |geoms, m, _s, ke| thick::element_stiffness(&geoms[0], &geoms[1], m, &lay, ke),
/// )?;
/// // Le bloc porte les six DDL de chaque nœud : 18 × 18 sur un TRI3.
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (18, 18));
/// // Et il est symétrique, comme toute raideur.
/// let d = bloc.dense();
/// assert!((0..18).all(|i| (0..18).all(|j| (d[i * 18 + j] - d[j * 18 + i]).abs() < 1e-6)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn element_stiffness(
    full: &CellGeom,
    reduced: &CellGeom,
    material: &SubElementField,
    lay: &ElementLayout,
    ke: &mut [f64],
) -> Result<()> {
    let n = full.n_nodes;
    let side = 6 * n;
    // `E`, `nu`, `h`, in the order `MATERIAL_COMPONENTS` declares.
    let row = material.row(full.cell, 0);
    let (e, nu, h) = (
        row[lay.material[0] as usize],
        row[lay.material[1] as usize],
        row[lay.material[2] as usize],
    );
    let dm = membrane_law(e, nu, h);
    let db = bending_law(e, nu, h);
    let kd = drilling_law(e, nu, h);
    let ds = shear_law(e, nu, h, shear_factor(row, lay));

    let frame = local_frame(full)?;
    // La rotation de fibre est un champ à part entière : sa courbure est son
    // gradient, et il n'y a rien à éliminer d'abord.
    let setup = BendingSetup::Direct;
    // Toute la matrice tient sur la pile : `Shell::new` a déjà refusé tout
    // élément plus large qu'un QUA4.
    let mut local: ShellMatrix = [0.0; MAX_SHELL_DOFS * MAX_SHELL_DOFS];
    let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];

    // ── Membrane, bending and drilling: full quadrature ───────────────────
    for g in 0..full.n_gauss {
        b_into(full, &frame, &setup, g, &mut b)?;
        let w = full.det_j_w(g);
        accumulate(&mut local, &b[0..3], &dm, w, side);
        accumulate(&mut local, &b[3..6], &db, w, side);
        accumulate_scalar(&mut local, &b[6..SHEAR_ROW], kd, w, side);
    }

    // ── Transverse shear: reduced quadrature, against locking ─────────────
    for g in 0..reduced.n_gauss {
        shear_b_into(reduced, &frame, g, &mut b)?;
        accumulate_scalar(&mut local, &b[SHEAR_ROW..], ds, reduced.det_j_w(g), side);
    }

    to_global(&local, &frame, n, ke);
    Ok(())
}
