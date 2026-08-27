//! Internal nodal forces `f = ∫ Bᵀ σ dΩ` of a `Model` (Cast3m `BSIG`) — the
//! transpose of the deformation operator `B`, applied to the stress.
//!
//! This is the mechanical generalisation of
//! [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence): where
//! `divergence` scatters `Bᵀ q` of a scalar transport flux to the nodes,
//! `internal_forces` does the same for every behaviour-bearing sub-model, one
//! output component per dual DOF. It is the exact transpose of the geometric
//! deformation producer each physics uses
//! ([`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation),
//! [`crate::ops::element_field::beam_deformation`](fn@crate::ops::element_field::beam_deformation)),
//! so continuum solids, bars and beams are all handled — each physics supplies
//! its own `Bᵀ` kernel ([`crate::models::SubModelKind::internal_force_element`]), and
//! this layer only orchestrates the per-zone pairing and aggregation.
//!
//! `stresses` is the material-state aggregate produced by
//! [`crate::ops::element_field::behavior::integrate`] (`COMP`). For a **linear** law the result
//! equals `K·u` (the assembled stiffness applied to the solution); a non-linear
//! law departs from that tangent — which is the point of forming the residual
//! from the exact stresses (`r = f_ext − f_int`).
//!
//! Constraint sub-models (`Dirichlet`, …) carry no behaviour and are skipped.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::Model;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::{continuum_internal_force_element, kernel};

/// Axis suffixes for the displacement/force components of the model-free
/// continuum operator (`f_x`, `f_y`, `f_z`).
const AXES: [&str; 3] = ["x", "y", "z"];

/// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of `model` (Cast3m `BSIG`).
///
/// `stresses` is the material-state aggregate from
/// [`crate::ops::element_field::behavior::integrate`]. For each behaviour-bearing sub-model the
/// matching stress sub-field is paired by FE subspace (its behaviour FE
/// subspace), its `Bᵀ` kernel is integrated cell-by-cell and scattered to the
/// nodes. Returns a [`NodeField`] with one zone per behaviour-bearing sub-model
/// in model order, its components the sub-model's dual variables.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // La forme qui passe par le **modèle** : chaque sous-modèle y apporte
/// // son propre opérateur, une barre n'ayant pas le Bᵀ d'un continuum.
/// # let modele = model::elasticity(&fes, Kinematics::PlaneStress)?;
/// # let mut s = ElementField::new(&fes,
/// #     vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()])?;
/// # s.get(0)?.write().set_uniform("sigma_xx", 100.0)?;
/// let f = node_field::internal_forces(&s, &modele)?;
/// assert_eq!(f.get(0)?.read().components(),
///            &["f_x".to_string(), "f_y".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn internal_forces(stresses: &ElementField, model: &Model) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for h in model {
        let beh_fespace = h.read().behavior_fespace();
        let Some(beh_fespace) = beh_fespace else {
            continue; // constraint sub-model — no behaviour, no internal force
        };
        let stress = stresses.sub_for_fespace(&beh_fespace)?;
        let sub = h.read().build_internal_forces(&stress)?;
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

/// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of a **continuum-mechanics** stress
/// field, **without a model** (Cast3m `BSIG` for a plain solid).
///
/// A convenience for the volumetric case (elasticity, Mazars, plasticity), where
/// `B` is the universal symmetric gradient and the DOFs are always a
/// displacement: it applies the same continuum kernel as
/// [`crate::models::SubModelKind::internal_force_element`]'s default, so it needs only
/// the geometry (`fespace`) and the Voigt stress (`sigma_xx`, `sigma_xy`, …).
/// Each subspace of `fespace` is paired with its stress sub-field; the result
/// carries `space_dim` components `f_x, f_y, f_z` per node.
///
/// Bars and beams are **not** covered — their `B` is not the symmetric gradient
/// and their DOFs are not a displacement vector, so use
/// [`internal_forces`] (which dispatches per physics) for those.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::{element_field, geom, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// // ∫ Bᵀσ dΩ : le résidu mécanique. Un état de contrainte uniforme donne
/// // des forces nodales de **somme nulle** — l'équilibre global.
/// # let mut s = ElementField::new(&fes,
/// #     vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()])?;
/// # s.get(0)?.write().set_uniform("sigma_xx", 100.0)?;
/// let f = node_field::internal_forces_continuum(&s, &fes)?;
/// let total: f64 = (0..3)
///     .map(|i| f.get(0).unwrap().read().value(n[i].id(), "f_x").unwrap())
///     .sum();
/// assert!(total.abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn internal_forces_continuum(
    stresses: &ElementField,
    fespace: &FiniteElementSpace,
) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for sub in fespace {
        let (submesh, space_dim) = {
            let s = sub.read();
            (s.submesh(), s.space_dim())
        };
        let support = submesh.read().to_poi1()?;
        let dual_vars: Vec<String> = (0..space_dim).map(|a| format!("f_{}", AXES[a])).collect();
        let stress = stresses.sub_for_fespace(sub)?;
        let stress_guard = stress.read();
        let sub_nf = kernel::scatter_to_nodes(
            std::slice::from_ref(sub),
            &support,
            dual_vars,
            |geoms, fe| continuum_internal_force_element(geoms, &stress_guard, fe),
        )?;
        out.add_sub(Handle::new(sub_nf))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::model::SubModel;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::coords::Coords;
    use crate::models::tensor::Kinematics;
    use crate::ops::element_field::behavior::integrate;
    use crate::ops::element_field::deformation;
    use crate::ops::element_field::material_field;

    /// Elasticity on a single TRI3: for a linear law the internal forces equal
    /// `K·u` applied to the (linear) displacement solution.
    #[test]
    fn elasticity_internal_forces_match_k_times_u() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(
                SubModel::elasticity(fes.get(0).unwrap(), Kinematics::PlaneStress).unwrap(),
            ))
            .unwrap();
        let materials = material_field(&model, &[("E", 210.0), ("nu", 0.3)]).unwrap();

        // Arbitrary linear displacement field.
        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 0.02 * x - 0.01 * y).unwrap();
            u.set_value(n.id(), "u_y", 0.005 * x + 0.03 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        // f_int = ∫ Bᵀ σ  vs  K·u.
        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();
        let f_int = internal_forces(&stress, &model).unwrap();

        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        let ku = (&k * &u).unwrap();

        let fv = f_int.view().unwrap();
        let kv = ku.view().unwrap();
        let tol = 1e-9;
        for nid in [a.id(), b.id(), c.id()] {
            for comp in ["f_x", "f_y"] {
                let got = fv.value(nid, comp).unwrap();
                let want = kv.value(nid, comp).unwrap();
                assert!((got - want).abs() < tol, "{comp}@{nid:?}: {got} ≠ {want}");
            }
        }
    }

    /// The model-free continuum variant reproduces the model-based operator on a
    /// solid (elasticity), node for node.
    #[test]
    fn continuum_variant_matches_model_based() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(
                SubModel::elasticity(fes.get(0).unwrap(), Kinematics::PlaneStrain).unwrap(),
            ))
            .unwrap();
        let materials = material_field(&model, &[("E", 70.0), ("nu", 0.25)]).unwrap();

        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 0.03 * x + 0.01 * y).unwrap();
            u.set_value(n.id(), "u_y", -0.02 * x + 0.04 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();

        let via_model = internal_forces(&stress, &model).unwrap();
        let via_fespace = internal_forces_continuum(&stress, &fes).unwrap();

        let m = via_model.view().unwrap();
        let f = via_fespace.view().unwrap();
        let tol = 1e-12;
        for nid in [a.id(), b.id(), c.id()] {
            for comp in ["f_x", "f_y"] {
                assert!((m.value(nid, comp).unwrap() - f.value(nid, comp).unwrap()).abs() < tol);
            }
        }
    }

    /// Bar (truss): a pure axial stretch gives the equilibrating end forces
    /// `f_A = −N c`, `f_B = +N c` with `N = E·A·ε` along the bar axis.
    #[test]
    fn truss_internal_forces_are_equilibrating_end_forces() {
        let (e, area, dx, dy) = (100.0, 3.0, 3.0, 4.0);
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[dx, dy]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(SubModel::truss(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        let materials = material_field(&model, &[("E", e), ("A", area)]).unwrap();

        let len = (dx * dx + dy * dy).sqrt();
        let c = [dx / len, dy / len];
        // u = ε·L along the axis at B, zero at A ⇒ axial strain ε.
        let eps = 0.01;
        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        u.set_value(a.id(), "u_x", 0.0).unwrap();
        u.set_value(a.id(), "u_y", 0.0).unwrap();
        u.set_value(b.id(), "u_x", eps * len * c[0]).unwrap();
        u.set_value(b.id(), "u_y", eps * len * c[1]).unwrap();
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();
        let f_int = internal_forces(&stress, &model).unwrap();

        let n = e * area * eps; // axial force
        let fv = f_int.view().unwrap();
        let tol = 1e-9;
        assert!((fv.value(a.id(), "f_x").unwrap() + n * c[0]).abs() < tol);
        assert!((fv.value(a.id(), "f_y").unwrap() + n * c[1]).abs() < tol);
        assert!((fv.value(b.id(), "f_x").unwrap() - n * c[0]).abs() < tol);
        assert!((fv.value(b.id(), "f_y").unwrap() - n * c[1]).abs() < tol);
    }
}
