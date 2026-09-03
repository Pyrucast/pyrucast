//! Generalised **strains of a shell**, at the Gauss points — the geometric
//! producer of the behaviour input for [`Shell`](crate::models::shell::Shell).
//!
//! Per element it applies the facet's own strain-displacement matrix
//! ([`crate::models::shell::b_into`]) to the nodal degrees of freedom, rotated
//! into the element's triad: the strains *are* `B · u`, and this operator is
//! that product and nothing else. Which is why it takes the
//! [`ShellModel`](crate::models::shell::ShellModel): the membrane, drilling and
//! shear rows are shared, but the **bending** ones are the whole difference
//! between the two formulations — a plain gradient of the fibre rotation for
//! Reissner-Mindlin, the discrete-Kirchhoff elimination for the other.
//!
//! | formulation | components produced |
//! |---|---|
//! | `thick` | `eps_xx, eps_yy, eps_xy`, `kappa_xx, kappa_yy, kappa_xy`, `drill`, `gamma_xz, gamma_yz` |
//! | `kirchhoff` | the same, without the two shear strains |
//!
//! All of them are read in the element's **local** frame, which is what the
//! shell physics expects and what makes them comparable between facets of
//! different orientation.
//!
//! ## Two quadratures, because the stiffness has two
//!
//! Membrane, bending and drilling are evaluated at each Gauss point. The
//! transverse shear is evaluated **once**, at the reduced point, and written to
//! every Gauss point — element-constant.
//!
//! That is not a simplification: it is the same reduced integration that keeps a
//! thin shell from locking, read from the other side. The stiffness integrates
//! `B_sᵀ D_s B_s` at that single point, so a recovery that sampled the shear
//! anywhere else would report a strain the element does not actually carry — and
//! `∫ Bᵀσ` would stop matching `K·u`.
//!
//! Feed the result to [`crate::ops::element_field::behavior::integrate`]; the
//! shell physics turns it into the generalised forces.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::Field;
use crate::containers::finite_element_space::{
    FiniteElementSpace, Interpolation, QuadratureRule, SubFiniteElementSpace,
};
use crate::containers::node_field::{NodeField, NodeFieldView};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel::{self, MAX_CELL_DOFS};
use crate::models::owned_components;
use crate::models::shell::{
    b_into, bending_setup, local_dofs, local_frame, shear_b_into, ShellB, ShellModel,
    MAX_SHELL_DOFS, SHEAR_ROW, SHELL_STRAINS,
};

/// Displacement + rotation DOFs read from the nodal field — a shell's six, the
/// same as a space frame's.
const DOFS: [&str; 6] = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"];

/// Generalised strains of a shell displacement/rotation `field` at the Gauss
/// points of every subspace of `fespace`.
///
/// The field must carry the six shell DOFs; each subspace must be a `TRI3` or
/// `QUA4` surface in a 3-D configuration, as the shell physics requires.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::{NodeField, SubNodeField};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::shell::ShellModel;
/// # use pyrucast::ops::element_field::shell_deformation;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let support = Handle::new(SubMesh::poi1_from_nodes(&n)?);
/// # let noms = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
/// #     .iter().map(|s| s.to_string()).collect();
/// # let mut u = SubNodeField::from_poi1(&support, noms)?;
/// // Un étirement uniforme le long de `x` : la facette est dans le plan
/// // z = 0 et sa première arête suit `x`, donc le repère local est le
/// // repère global.
/// u.set_value(n[1].id(), "u_x", 0.1)?;
/// let d = shell_deformation(&NodeField::from_sub(u), &fes, ShellModel::Thick)?;
/// let s = d.get(0)?.read();
/// // ε_xx = 0,1/2 = 0,05, et rien d'autre : ni flexion, ni vrillage.
/// assert!((s.value(0, 0, "eps_xx")? - 0.05).abs() < 1e-12);
/// assert!(s.value(0, 0, "kappa_xx")?.abs() < 1e-12);
/// assert!(s.value(0, 0, "drill")?.abs() < 1e-12);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn shell_deformation(
    field: &NodeField,
    fespace: &FiniteElementSpace,
    model: ShellModel,
) -> Result<ElementField> {
    let components = Field::components(field);
    let view = field.view()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        out.add_sub(Handle::new(subspace_shell_deformation(
            sub,
            &view,
            &components,
            model,
        )?))?;
    }
    Ok(out)
}

/// Generalised strains on one surface subspace.
fn subspace_shell_deformation(
    fespace: &Handle<SubFiniteElementSpace>,
    view: &NodeFieldView,
    components: &[String],
    model: ShellModel,
) -> Result<SubElementField> {
    kernel::require_field_basis(fespace, "shape values")?;
    // The field must carry every shell DOF up front (a clearer error than a
    // missing-component failure deep in the per-cell loop).
    for required in DOFS {
        if !components.iter().any(|c| c == required) {
            return Err(PyrucastError::Message(format!(
                "shell_deformation: the field must carry a '{required}' component (has: [{}])",
                components.join(", ")
            )));
        }
    }
    let reads = owned_components(&DOFS);
    let names = owned_components(model.strains());

    // ── Membrane, bending and drilling, at every Gauss point ──────────────
    //
    // `local_frame` and the bending setup are facts of the **cell**, rebuilt at
    // each point here rather than hoisted: a recovery runs once per solve, not
    // once per Newton iteration of an assembly, and the shared driver is what
    // keeps this operator off a parallel loop of its own.
    let full = kernel::nodal_pointwise(
        fespace,
        view,
        &reads,
        names[..SHEAR_ROW].to_vec(),
        |geom, g, dofs, out| {
            let frame = local_frame(geom)?;
            let setup = bending_setup(model, geom, &frame)?;
            let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];
            b_into(geom, &frame, &setup, g, &mut b)?;
            let mut d = [0.0_f64; MAX_CELL_DOFS];
            local_dofs(geom.n_nodes, &frame, dofs, &mut d);
            let side = 6 * geom.n_nodes;
            for (o, brow) in out.iter_mut().zip(b.iter()) {
                *o = (0..side).map(|i| brow[i] * d[i]).sum();
            }
            Ok(())
        },
    )?;
    if !model.has_transverse_shear() {
        return Ok(full);
    }

    // ── The transverse shear, at the reduced point alone ──────────────────
    let reduced = Handle::new(SubFiniteElementSpace::new(
        fespace.read().submesh(),
        Interpolation::Lagrange1,
        QuadratureRule::Reduced,
    )?);
    let shear = kernel::nodal_pointwise(
        &reduced,
        view,
        &reads,
        names[SHEAR_ROW..].to_vec(),
        |geom, g, dofs, out| {
            let frame = local_frame(geom)?;
            let mut b: ShellB = [[0.0; MAX_SHELL_DOFS]; SHELL_STRAINS];
            shear_b_into(geom, &frame, g, &mut b)?;
            let mut d = [0.0_f64; MAX_CELL_DOFS];
            local_dofs(geom.n_nodes, &frame, dofs, &mut d);
            let side = 6 * geom.n_nodes;
            for (o, brow) in out.iter_mut().zip(b[SHEAR_ROW..].iter()) {
                *o = (0..side).map(|i| brow[i] * d[i]).sum();
            }
            Ok(())
        },
    )?;

    // ── One field: the shear broadcast onto the full Gauss points ─────────
    let mut merged = SubElementField::new(fespace.clone(), names)?;
    let n_gauss = merged.gauss_count();
    for cell in 0..merged.cell_count() {
        let constant = shear.row(cell, 0);
        for g in 0..n_gauss {
            for (c, &v) in full.row(cell, g).iter().enumerate() {
                merged.set(cell, g, c, v)?;
            }
            for (c, &v) in constant.iter().enumerate() {
                merged.set(cell, g, SHEAR_ROW + c, v)?;
            }
        }
    }
    Ok(merged)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;

    /// One flat `TRI3` in the `z = 0` plane, and a six-DOF field over it.
    fn facet() -> (FiniteElementSpace, Vec<Node>, SubNodeField) {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<Node> = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&n).unwrap());
        let u = SubNodeField::from_poi1(&support, owned_components(&DOFS)).unwrap();
        (fes, n, u)
    }

    /// A **rigid** motion of the facet — a translation and a rotation about its
    /// own normal — strains nothing at all, drilling included. The drilling row
    /// is the one that would fail here if it were a diagonal penalty on `θ_z`
    /// rather than the residual against the membrane's own rotation.
    #[test]
    fn a_rigid_motion_strains_nothing() {
        let (fes, n, mut u) = facet();
        for (i, node) in n.iter().enumerate() {
            let p = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]][i];
            // u = t + ω × x, with ω about z: (−ω y, ω x).
            let w = 0.01;
            u.set_value(node.id(), "u_x", 0.3 - w * p[1]).unwrap();
            u.set_value(node.id(), "u_y", 0.7 + w * p[0]).unwrap();
            u.set_value(node.id(), "u_z", -0.2).unwrap();
            u.set_value(node.id(), "r_z", w).unwrap();
        }
        let d = shell_deformation(&NodeField::from_sub(u), &fes, ShellModel::Thick).unwrap();
        let s = d.get(0).unwrap().read();
        for c in ShellModel::Thick.strains() {
            let v = s.value(0, 0, c).unwrap();
            assert!(v.abs() < 1e-12, "{c} = {v} under a rigid motion");
        }
    }

    /// The transverse shear is **element-constant**: sampled at the reduced
    /// point, written at every Gauss point. That is what makes `∫ Bᵀσ` match
    /// the reduced-integrated stiffness.
    #[test]
    fn the_shear_is_element_constant() {
        let (fes, n, mut u) = facet();
        u.set_value(n[1].id(), "u_z", 0.05).unwrap();
        u.set_value(n[2].id(), "r_x", 0.02).unwrap();
        let d = shell_deformation(&NodeField::from_sub(u), &fes, ShellModel::Thick).unwrap();
        let s = d.get(0).unwrap().read();
        assert!(s.gauss_count() > 1, "a TRI3 carries several Gauss points");
        for c in ["gamma_xz", "gamma_yz"] {
            let first = s.value(0, 0, c).unwrap();
            for g in 1..s.gauss_count() {
                assert!((s.value(0, g, c).unwrap() - first).abs() < 1e-15);
            }
        }
        // …and it is not simply zero.
        assert!(s.value(0, 0, "gamma_xz").unwrap().abs() > 1e-6);
    }

    /// Discrete Kirchhoff reports no shear strain, its `γ` being imposed zero
    /// rather than integrated — the list stops where the reduced quadrature
    /// would have begun.
    #[test]
    fn discrete_kirchhoff_reports_no_shear() {
        let (fes, n, mut u) = facet();
        u.set_value(n[1].id(), "u_z", 0.05).unwrap();
        let d = shell_deformation(&NodeField::from_sub(u), &fes, ShellModel::Kirchhoff).unwrap();
        assert_eq!(d.get(0).unwrap().read().components().len(), SHEAR_ROW);
    }

    /// The field must carry every shell DOF, said up front rather than as a
    /// missing-component failure deep in the loop.
    #[test]
    fn rejects_missing_dof() {
        let (fes, n, _) = facet();
        let support = Handle::new(SubMesh::poi1_from_nodes(&n).unwrap());
        let u = SubNodeField::from_poi1(&support, owned_components(&DOFS[..5])).unwrap();
        let err = shell_deformation(&NodeField::from_sub(u), &fes, ShellModel::Thick).unwrap_err();
        assert!(format!("{err}").contains("must carry a 'r_z'"));
    }
}
