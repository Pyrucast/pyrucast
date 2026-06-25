//! Behaviour operator — integrate the constitutive law of a `Model`
//! (Cast3m `COMP`, « intégrer le comportement »).
//!
//! Where [`crate::ops::assemble::stiffness`] produces the *linearization* of
//! the model (a [`crate::containers::matrix::Matrix`]), [`integrate`]
//! produces the *exact* point-wise response as an
//! [`crate::containers::element_field::ElementField`].
//!
//! The deformation input is built **separately and geometrically** by
//! [`crate::ops::field::gradient`](fn@crate::ops::field::gradient) (`∇T`, …) or
//! [`crate::ops::field::deformation`](fn@crate::ops::field::deformation) (`ε`, …) — those depend only on the FE
//! space, not on the model. [`integrate`] then feeds that deformation (plus
//! the input internal-state variables `VAR0`) and the per-zone material to
//! each physics's law, returning the **material state**: the dual flux/stress
//! plus the updated state variables `VAR1`.
//!
//! It skips constraint sub-models (`Dirichlet`, …): a sub-model takes part
//! iff it declares a [`crate::containers::model::SubModel::behavior_fespace`].
//! The per-physics integrands live in [`crate::models`]; this layer only
//! orchestrates the loop, the per-zone field matching, and the aggregation.
//!
//! For a **linear** law the result is consistent with `stiffness`
//! (`∫ Bᵀ·flux = K·u`); a non-linear law departs from that tangent — that is
//! the whole point of integrating the behaviour exactly.

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::model::Model;
use crate::error::Result;
use crate::store::{insert, read};

/// Integrate the constitutive law of `model` (Cast3m `COMP`).
///
/// `deformation` is the behaviour-input aggregate (from
/// [`crate::ops::field::gradient`](fn@crate::ops::field::gradient) / [`crate::ops::field::deformation`](fn@crate::ops::field::deformation),
/// optionally carrying the input state `VAR0`); `materials` supplies the
/// per-zone material data. For each behaviour-bearing sub-model, the matching
/// deformation and material sub-fields are paired by FE subspace, and the
/// physics's law is integrated point-by-point. Returns the material-state
/// aggregate (dual flux/stress + updated state `VAR1`), one sub-field per
/// behaviour-bearing sub-model in model order.
pub fn integrate(
    model: &Model,
    deformation: &ElementField,
    materials: &ElementField,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for h in model {
        // Cheap read under the sub-model lock: which FE subspaces this
        // sub-model wants for its deformation and its material.
        let (beh_fespace, mat_fespace) =
            {
                let sub = read(h)?;
                (sub.behavior_fespace(), sub.material_fespace())
            };
        let Some(beh_fespace) = beh_fespace else {
            continue; // constraint sub-model — no behaviour
        };

        // Pair the per-zone fields by FE subspace (locks SubElementField,
        // outside the sub-model lock).
        let input = deformation.sub_for_fespace(&beh_fespace)?;
        let material = match mat_fespace {
            Some(fe) => Some(materials.sub_for_fespace(&fe)?),
            None => None,
        };

        let state = read(h)?.integrate_behavior(&input, material.as_ref())?;
        out.add_sub(insert(state))?;
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::containers::model::SubModel;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::ops::build::{material_field, material_field_per_sub_model};
    use crate::ops::field::gradient;

    /// SEG2 of length `L`, HeatConduction model (+ optional Dirichlet on the
    /// left node), and the linear nodal solution `T(a)=0, T(b)=dt`. Returns
    /// `(model, fes, solution)`.
    fn seg2(
        length: f64,
        dt: f64,
        dirichlet: bool,
    ) -> (Model, FiniteElementSpace, NodeField) {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        if dirichlet {
            let imposed =
                Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
            let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
            model
                .add_sub(insert(
                    SubModel::dirichlet(
                        "T".into(),
                        "q".into(),
                        &imposed,
                        &multiplier,
                        None,
                        None,
                    )
                    .unwrap(),
                ))
                .unwrap();
        }

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut sol = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        sol.set_value(a.id(), "T", 0.0).unwrap();
        sol.set_value(b.id(), "T", dt).unwrap();
        (model, fes, NodeField::from_sub(sol))
    }

    /// Full chain `gradient → integrate`: COMP returns the weak-form flux
    /// `k·∇T` and the Dirichlet sub-model is skipped.
    #[test]
    fn integrate_returns_weak_form_flux_and_skips_dirichlet() {
        let (model, fes, sol) = seg2(2.0, 3.0, true);
        let def = gradient(&sol, &fes).unwrap();
        let materials = material_field(&model, &[("k", 1.5)]).unwrap();

        let state = integrate(&model, &def, &materials).unwrap();
        assert_eq!(state.len(), 1, "only the HC sub-model carries a behaviour");
        {
            let s = read(&state.get(0).unwrap()).unwrap();
            assert_eq!(s.components(), &["flux_x".to_string()]);
            // weak-form flux = k·∇T = 1.5 · (3/2) = 2.25.
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "flux_x").unwrap() - 2.25).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn integrate_errors_when_material_missing() {
        let (model, fes, sol) = seg2(1.0, 1.0, false);
        let def = gradient(&sol, &fes).unwrap();
        let empty = ElementField::empty();
        let err = integrate(&model, &def, &empty).unwrap_err();
        assert!(format!("{err}").contains("no SubElementField"));
    }

    /// Two SEG2 zones with different conductivities: COMP picks the right
    /// material per zone, exactly like `assemble::stiffness`.
    #[test]
    fn integrate_picks_per_zone_material() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::empty();
        for pair in [[&n0, &n1], [&n1, &n2]] {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[pair[0].id(), pair[1].id()]).unwrap();
            mesh.add_sub(insert(sm)).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        model
            .add_sub(insert(SubModel::heat_conduction(fes.get(1).unwrap()).unwrap()))
            .unwrap();

        // Linear ramp T = x ⇒ ∇T = 1 everywhere.
        let support =
            insert(SubMesh::poi1_from_nodes(&[n0.clone(), n1.clone(), n2.clone()]).unwrap());
        let mut sol = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        sol.set_value(n0.id(), "T", 0.0).unwrap();
        sol.set_value(n1.id(), "T", 1.0).unwrap();
        sol.set_value(n2.id(), "T", 2.0).unwrap();
        let sol = NodeField::from_sub(sol);

        let def = gradient(&sol, &fes).unwrap();
        let materials =
            material_field_per_sub_model(&model, &[&[("k", 1.0)], &[("k", 4.0)]]).unwrap();
        let state = integrate(&model, &def, &materials).unwrap();
        assert_eq!(state.len(), 2);
        // Zone A: k = 1 ⇒ flux = 1; zone B: k = 4 ⇒ flux = 4.
        {
            let s = read(&state.get(0).unwrap()).unwrap();
            assert!((s.value(0, 0, "flux_x").unwrap() - 1.0).abs() < 1e-12);
        }
        {
            let s = read(&state.get(1).unwrap()).unwrap();
            assert!((s.value(0, 0, "flux_x").unwrap() - 4.0).abs() < 1e-12);
        }
    }
}
