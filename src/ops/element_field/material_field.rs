use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::model::{Model, SubModel};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};

/// Build the material [`SubElementField`] of a single material-hungry
/// sub-model from uniform `(component, value)` pairs.
///
/// Every **required** component (`material_components`) must be present, in
/// declaration order. Every **optional** component
/// (`optional_material_components`, e.g. `alpha` for thermal expansion) is kept
/// only if supplied, appended after the required ones. Any other pair is
/// dropped. Errors if `sub` needs no material (e.g. `Dirichlet`) or if a
/// required component is missing from `components_and_values`.
pub fn sub_material_field(
    sub: &SubModel,
    components_and_values: &[(&str, f64)],
) -> Result<SubElementField> {
    let fespace = sub.material_fespace().ok_or_else(|| {
        PyrucastError::Message("material_field: this sub-model does not need material data".into())
    })?;
    let required = sub
        .material_components()
        .expect("material_fespace().is_some() ⇒ material_components() is Some");
    let optional = sub.optional_material_components();
    let mut components: Vec<String> = Vec::with_capacity(required.len() + optional.len());
    let mut values: Vec<f64> = Vec::with_capacity(required.len() + optional.len());
    for req in &required {
        let v = components_and_values
            .iter()
            .find(|(c, _)| *c == req)
            .map(|(_, v)| *v)
            .ok_or_else(|| {
                PyrucastError::Message(format!(
                    "material_field: missing required component '{}' \
                     (this physics expects: {:?})",
                    req, required
                ))
            })?;
        components.push(req.clone());
        values.push(v);
    }
    // Optional components: kept only when the caller supplies them.
    for opt in optional {
        if let Some((_, v)) = components_and_values.iter().find(|(c, _)| c == opt) {
            components.push((*opt).to_string());
            values.push(*v);
        }
    }
    SubElementField::from_uniform_per_component(fespace, components, &values)
}

/// Build a material [`ElementField`] applying the same uniform
/// `(component, value)` pairs to **every** material-hungry sub-model of
/// `model`. Sub-models that need no material (`Dirichlet`, …) are skipped.
pub fn material_field(
    model: &Model,
    components_and_values: &[(&str, f64)],
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for h in model {
        let opt_sub = {
            let sub = read(h)?;
            if sub.material_fespace().is_none() {
                None
            } else {
                Some(sub_material_field(&sub, components_and_values)?)
            }
        };
        if let Some(sub) = opt_sub {
            out.add_sub(insert(sub))?;
        }
    }
    Ok(out)
}

/// Build a material [`ElementField`] with a **per-sub-model** uniform
/// `(component, value)` list.
///
/// `per_sub_model.len()` must equal `model.len()`. An empty
/// slot (`&[]`) skips that sub-model — typical for `Dirichlet`.
pub fn material_field_per_sub_model(
    model: &Model,
    per_sub_model: &[&[(&str, f64)]],
) -> Result<ElementField> {
    let n = model.len();
    if per_sub_model.len() != n {
        return Err(PyrucastError::Message(format!(
            "material_field_per_sub_model: {} list(s) supplied for {} sub-model(s)",
            per_sub_model.len(),
            n
        )));
    }
    let mut out = ElementField::empty();
    for (i, h) in model.iter().enumerate() {
        let spec = per_sub_model[i];
        if spec.is_empty() {
            continue;
        }
        let sub = sub_material_field(&*read(h)?, spec)?;
        out.add_sub(insert(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::atoms::NodeId;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::matrix::Matrix;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::store::Handle;

    /// HC on a single SEG2 (+ optional Dirichlet on the left node).
    /// Returns (coords, a_id, b_id, model).
    fn seg2_heat_model(
        length: f64,
        dirichlet_at_left: bool,
    ) -> (Handle<Coords>, NodeId, NodeId, Model) {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        if dirichlet_at_left {
            let imposed =
                Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
            let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
            model
                .add_sub(insert(
                    SubModel::dirichlet(
                        "T".into(),
                        "q".into(),
                        &imposed,
                        &multiplier,
                        None,
                        None,
                        Default::default(),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        (coords, a.id(), b.id(), model)
    }

    fn single_hc_sub() -> (Handle<Coords>, SubModel) {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let hc = SubModel::heat_conduction(fes.get(0).unwrap()).unwrap();
        (coords, hc)
    }

    // ── sub_material_field ──────────────────────────────────────────────

    #[test]
    fn sub_uniform_per_component() {
        let (_cfg, hc) = single_hc_sub();
        let sub = hc.material_fespace().unwrap();
        let mat = sub_material_field(&hc, &[("k", 2.5)]).unwrap();
        let n_g = read(&sub).unwrap().gauss_count();
        for g in 0..n_g {
            assert!((mat.value(0, g, "k").unwrap() - 2.5).abs() < 1e-12);
        }
    }

    #[test]
    fn sub_errors_on_dirichlet() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        let dir = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        assert!(sub_material_field(&dir, &[("k", 1.0)]).is_err());
    }

    #[test]
    fn sub_errors_on_empty_list() {
        let (_cfg, hc) = single_hc_sub();
        assert!(sub_material_field(&hc, &[]).is_err());
    }

    #[test]
    fn sub_errors_on_missing_required_component() {
        let (_cfg, hc) = single_hc_sub();
        // "rho" is not what HeatConduction needs ⇒ missing "k".
        let err = sub_material_field(&hc, &[("rho", 1.0)]).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("'k'"), "unexpected error: {}", msg);
    }

    #[test]
    fn sub_filters_extra_components() {
        let (_cfg, hc) = single_hc_sub();
        // `k` required; `rho`/`cp` optional (heat-capacity) kept because
        // supplied; a genuinely unknown component is dropped.
        let mat = sub_material_field(
            &hc,
            &[("k", 2.0), ("rho", 7.0), ("cp", 9.0), ("bogus", 1.0)],
        )
        .unwrap();
        assert_eq!(
            mat.components(),
            &["k".to_string(), "rho".to_string(), "cp".to_string()]
        );
        assert!((mat.value(0, 0, "k").unwrap() - 2.0).abs() < 1e-12);
        assert!((mat.value(0, 0, "rho").unwrap() - 7.0).abs() < 1e-12);
        assert!(mat.value(0, 0, "bogus").is_err());
    }

    // ── optional material components (e.g. elasticity `alpha`) ──────────────

    /// Elasticity on a single TRI3 (plane stress).
    fn single_elasticity_sub() -> (Handle<Coords>, SubModel) {
        use crate::models::elasticity::ElasticityModel;
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let el = SubModel::elasticity(fes.get(0).unwrap(), ElasticityModel::PlaneStress).unwrap();
        (coords, el)
    }

    #[test]
    fn optional_alpha_kept_when_supplied() {
        let (_cfg, el) = single_elasticity_sub();
        let mat = sub_material_field(&el, &[("E", 210e9), ("nu", 0.3), ("alpha", 1.2e-5)]).unwrap();
        assert_eq!(
            mat.components(),
            &["E".to_string(), "nu".to_string(), "alpha".to_string()]
        );
        assert!((mat.value(0, 0, "alpha").unwrap() - 1.2e-5).abs() < 1e-18);
    }

    #[test]
    fn optional_alpha_absent_is_valid() {
        let (_cfg, el) = single_elasticity_sub();
        // No alpha ⇒ plain elastic material, still valid (alpha never required).
        let mat = sub_material_field(&el, &[("E", 210e9), ("nu", 0.3)]).unwrap();
        assert_eq!(mat.components(), &["E".to_string(), "nu".to_string()]);
        assert!(mat.value(0, 0, "alpha").is_err());
    }

    #[test]
    fn unknown_component_still_dropped_with_optionals() {
        let (_cfg, el) = single_elasticity_sub();
        // "bogus" is neither required nor optional ⇒ dropped; alpha kept.
        let mat = sub_material_field(
            &el,
            &[("E", 1.0), ("nu", 0.2), ("alpha", 3.0), ("bogus", 7.0)],
        )
        .unwrap();
        assert_eq!(
            mat.components(),
            &["E".to_string(), "nu".to_string(), "alpha".to_string()]
        );
        assert!(mat.value(0, 0, "bogus").is_err());
    }

    // ── material_field (uniform over the whole model) ───────────────────

    #[test]
    fn uniform_skips_dirichlet_and_assembles() {
        let (_cfg, a_id, b_id, model) = seg2_heat_model(2.0, true);
        let materials = material_field(&model, &[("k", 1.5)]).unwrap();
        assert_eq!(materials.len(), 1, "only the HC slot is present");

        let k: Matrix = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        let tol = 1e-12;
        let expected = 1.5 / 2.0;
        assert!((k.get(a_id, "q", a_id, "T").unwrap() - expected).abs() < tol);
        assert!((k.get(b_id, "q", b_id, "T").unwrap() - expected).abs() < tol);
    }

    // ── material_field_per_sub_model ────────────────────────────────────

    #[test]
    fn per_sub_model_two_zones() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&n0)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        model
            .add_sub(insert(
                SubModel::dirichlet(
                    "T".into(),
                    "q".into(),
                    &imposed,
                    &multiplier,
                    None,
                    None,
                    Default::default(),
                )
                .unwrap(),
            ))
            .unwrap();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(1).unwrap()).unwrap(),
            ))
            .unwrap();

        // Slot lengths must match sub_model_count (3): HC, Dirichlet (skip), HC.
        let materials = material_field_per_sub_model(
            &model,
            &[
                &[("k", 1.0)], // zone A
                &[],           // Dirichlet — skip
                &[("k", 4.0)], // zone B
            ],
        )
        .unwrap();
        assert_eq!(materials.len(), 2, "only the two HC slots are present");

        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        let tol = 1e-12;
        let v = |i: NodeId, j: NodeId| k.get(i, "q", j, "T").unwrap();
        // n1 is shared between the two zones, so diagonal = 1.0 + 4.0 = 5.0.
        assert!((v(n0.id(), n0.id()) - 1.0).abs() < tol);
        assert!((v(n1.id(), n1.id()) - 5.0).abs() < tol);
        assert!((v(n2.id(), n2.id()) - 4.0).abs() < tol);
    }

    #[test]
    fn per_sub_model_length_mismatch_errors() {
        let (_cfg, _, _, model) = seg2_heat_model(1.0, true);
        // Model has 2 sub-models ; only 1 spec ⇒ error.
        let res = material_field_per_sub_model(&model, &[&[("k", 1.0)]]);
        assert!(res.is_err());
        let msg = format!("{}", res.unwrap_err());
        assert!(msg.contains("1") && msg.contains("2"));
    }
}
