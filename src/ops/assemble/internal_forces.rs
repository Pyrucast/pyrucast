//! Internal nodal forces `f = ∫ Bᵀ σ dΩ` of a `Model` (Cast3m `BSIG`) — the
//! transpose of the deformation operator `B`, applied to the stress.
//!
//! This is the mechanical generalisation of
//! [`crate::ops::field::divergence`](fn@crate::ops::field::divergence): where
//! `divergence` scatters `Bᵀ q` of a scalar transport flux to the nodes,
//! `internal_forces` does the same for every behaviour-bearing sub-model, one
//! output component per dual DOF. It is the exact transpose of the geometric
//! deformation producer each physics uses
//! ([`crate::ops::field::deformation`](fn@crate::ops::field::deformation),
//! [`crate::ops::field::beam_deformation`](fn@crate::ops::field::beam_deformation)),
//! so continuum solids, bars and beams are all handled — each physics supplies
//! its own `Bᵀ` kernel ([`crate::models::SubModelKind::internal_force_element`]), and
//! this layer only orchestrates the per-zone pairing and aggregation.
//!
//! `stresses` is the material-state aggregate produced by
//! [`crate::ops::behavior::integrate`] (`COMP`). For a **linear** law the result
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
use crate::models::{continuum_internal_force_element, kernel};
use crate::store::{insert, read};

/// Axis suffixes for the displacement/force components of the model-free
/// continuum operator (`f_x`, `f_y`, `f_z`).
const AXES: [&str; 3] = ["x", "y", "z"];

/// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of `model` (Cast3m `BSIG`).
///
/// `stresses` is the material-state aggregate from
/// [`crate::ops::behavior::integrate`]. For each behaviour-bearing sub-model the
/// matching stress sub-field is paired by FE subspace (its behaviour FE
/// subspace), its `Bᵀ` kernel is integrated cell-by-cell and scattered to the
/// nodes. Returns a [`NodeField`] with one zone per behaviour-bearing sub-model
/// in model order, its components the sub-model's dual variables.
pub fn internal_forces(model: &Model, stresses: &ElementField) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for h in model {
        let beh_fespace = read(h)?.behavior_fespace();
        let Some(beh_fespace) = beh_fespace else {
            continue; // constraint sub-model — no behaviour, no internal force
        };
        let stress = stresses.sub_for_fespace(&beh_fespace)?;
        let sub = read(h)?.build_internal_forces(&stress)?;
        out.add_sub(insert(sub))?;
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
pub fn internal_forces_continuum(
    stresses: &ElementField,
    fespace: &FiniteElementSpace,
) -> Result<NodeField> {
    let mut out = NodeField::empty();
    for sub in fespace {
        let (submesh, space_dim) = {
            let s = read(sub)?;
            (s.submesh(), s.space_dim())
        };
        let support = read(&submesh)?.to_poi1()?;
        let dual_vars: Vec<String> = (0..space_dim).map(|a| format!("f_{}", AXES[a])).collect();
        let stress = stresses.sub_for_fespace(sub)?;
        let stress_guard = read(&stress)?;
        let sub_nf = kernel::scatter_to_nodes(
            std::slice::from_ref(sub),
            &support,
            dual_vars,
            |geoms, fe| continuum_internal_force_element(geoms, &stress_guard, fe),
        )?;
        out.add_sub(insert(sub_nf))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::model::SubModel;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::models::elasticity::ElasticityModel;
    use crate::ops::behavior::integrate;
    use crate::ops::build::material_field;
    use crate::ops::field::{beam_deformation, deformation};

    /// Elasticity on a single TRI3: for a linear law the internal forces equal
    /// `K·u` applied to the (linear) displacement solution.
    #[test]
    fn elasticity_internal_forces_match_k_times_u() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::elasticity(fes.get(0).unwrap(), ElasticityModel::PlaneStress).unwrap(),
            ))
            .unwrap();
        let materials = material_field(&model, &[("E", 210.0), ("nu", 0.3)]).unwrap();

        // Arbitrary linear displacement field.
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 0.02 * x - 0.01 * y).unwrap();
            u.set_value(n.id(), "u_y", 0.005 * x + 0.03 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        // f_int = ∫ Bᵀ σ  vs  K·u.
        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();
        let f_int = internal_forces(&model, &stress).unwrap();

        let k = crate::ops::assemble::stiffness(&model, &materials).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::elasticity(fes.get(0).unwrap(), ElasticityModel::PlaneStrain).unwrap(),
            ))
            .unwrap();
        let materials = material_field(&model, &[("E", 70.0), ("nu", 0.25)]).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        for (n, x, y) in [(&a, 0.0, 0.0), (&b, 1.0, 0.0), (&c, 0.0, 1.0)] {
            u.set_value(n.id(), "u_x", 0.03 * x + 0.01 * y).unwrap();
            u.set_value(n.id(), "u_y", -0.02 * x + 0.04 * y).unwrap();
        }
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();

        let via_model = internal_forces(&model, &stress).unwrap();
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[dx, dy]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::truss(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        let materials = material_field(&model, &[("E", e), ("A", area)]).unwrap();

        let len = (dx * dx + dy * dy).sqrt();
        let c = [dx / len, dy / len];
        // u = ε·L along the axis at B, zero at A ⇒ axial strain ε.
        let eps = 0.01;
        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
        u.set_value(a.id(), "u_x", 0.0).unwrap();
        u.set_value(a.id(), "u_y", 0.0).unwrap();
        u.set_value(b.id(), "u_x", eps * len * c[0]).unwrap();
        u.set_value(b.id(), "u_y", eps * len * c[1]).unwrap();
        let u = NodeField::from_sub(u);

        let strain = deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &strain, None, &materials, None).unwrap();
        let f_int = internal_forces(&model, &stress).unwrap();

        let n = e * area * eps; // axial force
        let fv = f_int.view().unwrap();
        let tol = 1e-9;
        assert!((fv.value(a.id(), "f_x").unwrap() + n * c[0]).abs() < tol);
        assert!((fv.value(a.id(), "f_y").unwrap() + n * c[1]).abs() < tol);
        assert!((fv.value(b.id(), "f_x").unwrap() - n * c[0]).abs() < tol);
        assert!((fv.value(b.id(), "f_y").unwrap() - n * c[1]).abs() < tol);
    }

    /// Beam (Timoshenko): the internal forces equal `K·u` for the linear law.
    #[test]
    fn beam_internal_forces_match_k_times_u() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::timoshenko(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        let materials =
            material_field(&model, &[("E", 3.0), ("I", 2.0), ("G", 5.0), ("A_s", 2.0)]).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut u = SubNodeField::from_poi1(&support, vec!["w".into(), "theta".into()]).unwrap();
        u.set_value(a.id(), "w", 0.0).unwrap();
        u.set_value(a.id(), "theta", 0.0).unwrap();
        u.set_value(b.id(), "w", 0.5).unwrap();
        u.set_value(b.id(), "theta", 0.2).unwrap();
        let u = NodeField::from_sub(u);

        let sect = beam_deformation(&u, &fes).unwrap();
        let stress = integrate(&model, &sect, None, &materials, None).unwrap();
        let f_int = internal_forces(&model, &stress).unwrap();

        let k = crate::ops::assemble::stiffness(&model, &materials).unwrap();
        let ku = (&k * &u).unwrap();

        let fv = f_int.view().unwrap();
        let kv = ku.view().unwrap();
        let tol = 1e-9;
        for nid in [a.id(), b.id()] {
            for comp in ["f_w", "m_theta"] {
                let got = fv.value(nid, comp).unwrap();
                let want = kv.value(nid, comp).unwrap();
                assert!((got - want).abs() < tol, "{comp}@{nid:?}: {got} ≠ {want}");
            }
        }
    }
}
