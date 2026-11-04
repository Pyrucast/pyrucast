//! Behaviour operator — integrate the constitutive law of a `Model`
//! (Cast3m `COMP`, « intégrer le comportement »).
//!
//! Where [`crate::ops::matrix::stiffness`] produces the *linearization* of
//! the model (a [`crate::containers::matrix::Matrix`]), [`integrate`]
//! produces the *exact* point-wise response as an
//! [`crate::containers::element_field::ElementField`].
//!
//! The deformation input is built **separately and geometrically** by
//! [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient) (`∇T`, …) or
//! [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation) (`ε`, …) — those depend only on the FE
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
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::model::{Model, SubModel};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::Behavior;

/// Integrate the constitutive law of `model` (Cast3m `COMP`), stepping A → B.
///
/// `deformation` is the **end-of-step** behaviour input ε(B) (from
/// [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient) / [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation));
/// `prev` is the **state at A** — the converged output of the previous step
/// (σ(A), `VAR(A)`, ε(A)), or the rest state [`initial_state`] on the first
/// step; `materials` supplies the per-zone material data; `dt` is the time increment (`None` if
/// rate-independent). For each behaviour-bearing sub-model, the matching
/// deformation, previous-state and material sub-fields are paired by FE
/// subspace, and the physics's law is integrated point-by-point. Returns the
/// material-state aggregate at B (dual flux/stress + updated state `VAR1`), one
/// sub-field per behaviour-bearing sub-model in model order — the aggregate to
/// feed back as `prev` at the next step.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::element_field;
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let modele = model::elasticity(&fes, Kinematics::PlaneStress, MaterialSymmetry::Isotropic, StrainMeasure::Linear).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("E", 210e3), ("nu", 0.3)]).unwrap();
/// // The behaviour law, integrated Gauss point by Gauss point: a strain goes
/// // in, a stress comes out. In linear elasticity,
/// // σ_xx = E/(1−ν²)·ε_xx.
/// let mut eps = ElementField::new(&fes,
///     vec!["eps_xx".into(), "eps_yy".into(), "eps_xy".into()])?;
/// eps.get(0)?.write().set_uniform("eps_xx", 1e-3)?;
/// // Le premier pas part de l'état de repos.
/// let repos = element_field::behavior::initial_state(&modele, &materiaux)?;
/// let sig = element_field::behavior::integrate(&modele, &eps, &repos, &materiaux, None)?;
/// let attendu = 210e3 / (1.0 - 0.09) * 1e-3;
/// assert!((sig.get(0)?.read().value(0, 0, "sigma_xx")? - attendu).abs() < 1e-6);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn integrate(
    model: &Model,
    deformation: &ElementField,
    prev: &ElementField,
    materials: &ElementField,
    dt: Option<f64>,
) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for h in model {
        // Cheap read under the sub-model lock: which FE subspace this
        // sub-model wants for its deformation, and what its law reads.
        let (beh_fespace, reads) = {
            let sub = h.read();
            (
                sub.behavior_fespace(),
                sub.as_kind()
                    .as_behavior()
                    .map(|law| (law.deformation_reads(), previous_state_names(law))),
            )
        };
        let (Some(beh_fespace), Some((input_reads, prev_reads))) = (beh_fespace, reads) else {
            continue; // constraint sub-model — no behaviour
        };

        // Pair the per-zone fields by FE subspace (locks SubElementField,
        // outside the sub-model lock), and — when two laws share the subspace —
        // by what this law reads. The state at A is paired on the behaviour
        // subspace, exactly like the deformation.
        let input = deformation.sub_for_fespace_reading(&beh_fespace, &input_reads)?;
        let prev_zone = prev.sub_for_fespace_reading(&beh_fespace, &prev_reads)?;
        let mat = law_material(h, materials)?;

        // A rate-dependent law with no increment is refused here, before a
        // single point is integrated — where the message can name the physics,
        // and where the question is asked once instead of per Gauss point.
        let sub = h.read();
        let dt = match (dt, sub.requires_dt()) {
            (Some(dt), _) => dt,
            (None, false) => 0.0,
            (None, true) => {
                return Err(PyrucastError::Message(format!(
                    "{}: this law is rate-dependent and needs a time increment — \
                     pass `dt` to integrate_behavior",
                    sub.as_kind().label()
                )))
            }
        };

        let state = sub.integrate_behavior(&input, &prev_zone, &mat, dt)?;
        drop(sub);
        out.add_sub(Handle::new(state))?;
    }
    Ok(out)
}

/// The **rest state** of `model`: what each behaviour-bearing sub-model's law
/// holds before anything has happened — the state at A of a first step, to
/// hand to [`integrate`] and to [`crate::ops::matrix::tangent`].
///
/// One zone per behaviour-bearing sub-model, in model order, on its behaviour
/// subspace and with the components its law writes. Nearly every law rests at
/// zero (σ = 0, ε = 0, ε_p = 0, p = 0); a law whose internal variable starts
/// from a material constant seeds it from `materials` — Gurson's porosity from
/// `f_0` — which is why the material is an argument. Constraint sub-models,
/// which have no law, are skipped.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::ops::element_field::StrainMeasure;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::ops::element_field;
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let modele = model::plasticity_perfect(&fes, Kinematics::PlaneStrain, StrainMeasure::Linear).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("E", 210e3), ("nu", 0.3), ("sigma_y", 250.0)]).unwrap();
/// // L'état de repos porte tout ce que la loi écrit, à zéro.
/// let repos = element_field::behavior::initial_state(&modele, &materiaux)?;
/// let zone = repos.get(0)?;
/// assert!(zone.read().components().iter().any(|c| c == "p"));
/// assert_eq!(zone.read().value(0, 0, "sigma_xx")?, 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn initial_state(model: &Model, materials: &ElementField) -> Result<ElementField> {
    let mut out = ElementField::empty();
    for h in model {
        if h.read().as_kind().as_behavior().is_none() {
            continue; // constraint sub-model — no behaviour
        }
        let mat = law_material(h, materials)?;
        let state = h.read().initial_state(&mat)?;
        out.add_sub(Handle::new(state))?;
    }
    Ok(out)
}

/// The material zone the law of `h` reads, resolved by the components the
/// physics requires — so that a shared subspace carrying several
/// component-disjoint material zones gives each physics its own zone without
/// an explicit consolidate. Shared by [`integrate`] and [`initial_state`],
/// which must read the same zone.
fn law_material(h: &Handle<SubModel>, materials: &ElementField) -> Result<Handle<SubElementField>> {
    let (mat_fespace, mat_components) = {
        let sub = h.read();
        (sub.material_fespace(), sub.material_components())
    };
    // A physics with a behaviour declares a material FE subspace: the question
    // is settled here, once, and nothing below asks it again.
    let fe = mat_fespace.ok_or_else(|| {
        PyrucastError::Message(format!(
            "{}: a behaviour without material data cannot be integrated",
            h.read().as_kind().label()
        ))
    })?;
    match mat_components {
        Some(required) => materials.sub_for_fespace_with(&fe, &required),
        None => materials.sub_for_fespace(&fe),
    }
}

/// The names that single out a law's **previous state** among the zones of its
/// subspace: what it reads back, or — a law with no history — what it writes,
/// which its own previous output carries.
pub(crate) fn previous_state_names(law: &dyn Behavior) -> Vec<String> {
    let reads = law.state_reads();
    if reads.is_empty() {
        law.behavior_output_components()
    } else {
        reads
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::model::SubModel;
    use crate::containers::node_field::{NodeField, SubNodeField};
    use crate::coords::Coords;
    use crate::models::symmetry::MaterialSymmetry;
    use crate::ops::element_field::{beam_deformation, gradient};
    use crate::ops::element_field::{material_field, material_field_per_sub_model};

    /// SEG2 of length `L`, HeatConduction model (+ optional Dirichlet on the
    /// left node), and the linear nodal solution `T(a)=0, T(b)=dt`. Returns
    /// `(model, fes, solution)`.
    fn seg2(length: f64, dt: f64, dirichlet: bool) -> (Model, FiniteElementSpace, NodeField) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(
                SubModel::heat_conduction(fes.get(0).unwrap(), MaterialSymmetry::Isotropic)
                    .unwrap(),
            ))
            .unwrap();
        if dirichlet {
            let imposed =
                Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
            model
                .add_sub(Handle::new(
                    SubModel::dirichlet(&imposed, &model, "T", Default::default()).unwrap(),
                ))
                .unwrap();
        }

        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
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

        let state = integrate(
            &model,
            &def,
            &initial_state(&model, &materials).unwrap(),
            &materials,
            None,
        )
        .unwrap();
        assert_eq!(state.len(), 1, "only the HC sub-model carries a behaviour");
        {
            let s = state.get(0).unwrap().read();
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
        let materials = material_field(&model, &[("k", 1.5)]).unwrap();
        let rest = initial_state(&model, &materials).unwrap();
        let empty = ElementField::empty();
        let err = integrate(&model, &def, &rest, &empty, None).unwrap_err();
        assert!(format!("{err}").contains("no SubElementField"));
        // The rest state reads the same material zone, and is refused alike.
        let err = initial_state(&model, &empty).unwrap_err();
        assert!(format!("{err}").contains("no SubElementField"));
    }

    /// Two SEG2 zones with different conductivities: COMP picks the right
    /// material per zone, exactly like `crate::ops::matrix::stiffness`.
    #[test]
    fn integrate_picks_per_zone_material() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::empty();
        for pair in [[&n0, &n1], [&n1, &n2]] {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[pair[0].id(), pair[1].id()]).unwrap();
            mesh.add_sub(Handle::new(sm)).unwrap();
        }
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(Handle::new(
                SubModel::heat_conduction(fes.get(0).unwrap(), MaterialSymmetry::Isotropic)
                    .unwrap(),
            ))
            .unwrap();
        model
            .add_sub(Handle::new(
                SubModel::heat_conduction(fes.get(1).unwrap(), MaterialSymmetry::Isotropic)
                    .unwrap(),
            ))
            .unwrap();

        // Linear ramp T = x ⇒ ∇T = 1 everywhere.
        let support =
            Handle::new(SubMesh::poi1_from_nodes(&[n0.clone(), n1.clone(), n2.clone()]).unwrap());
        let mut sol = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        sol.set_value(n0.id(), "T", 0.0).unwrap();
        sol.set_value(n1.id(), "T", 1.0).unwrap();
        sol.set_value(n2.id(), "T", 2.0).unwrap();
        let sol = NodeField::from_sub(sol);

        let def = gradient(&sol, &fes).unwrap();
        let materials =
            material_field_per_sub_model(&model, &[&[("k", 1.0)], &[("k", 4.0)]]).unwrap();
        let state = integrate(
            &model,
            &def,
            &initial_state(&model, &materials).unwrap(),
            &materials,
            None,
        )
        .unwrap();
        assert_eq!(state.len(), 2);
        // Zone A: k = 1 ⇒ flux = 1; zone B: k = 4 ⇒ flux = 4.
        {
            let s = state.get(0).unwrap().read();
            assert!((s.value(0, 0, "flux_x").unwrap() - 1.0).abs() < 1e-12);
        }
        {
            let s = state.get(1).unwrap().read();
            assert!((s.value(0, 0, "flux_x").unwrap() - 4.0).abs() < 1e-12);
        }
    }

    /// Full chain `beam_deformation → integrate` for a plane frame: a known
    /// displacement/rotation state on a horizontal element (local = global)
    /// gives the hand-computed section forces `N = E·A·ε`, `M = E·I·κ`,
    /// `V = G·A_s·γ`.
    #[test]
    fn integrate_returns_frame_section_forces() {
        let l = 2.0;
        let (e, area, i, g, a_s) = (3.0, 4.0, 2.0, 5.0, 2.0);
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[l, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes =
            FiniteElementSpace::new(&mesh, crate::atoms::Interpolation::ModelEmbedded).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(Handle::new(
                SubModel::timoshenko(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();

        // Kinematics: u_x = 0.5·x ⇒ ε = 0.5; rz = 0.25·x ⇒ κ = 0.25;
        // u_y = 0, so γ = w'/L − θ_centre = 0 − (0 + 0.5)/2 = −0.25.
        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut sol =
            SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into(), "r_z".into()])
                .unwrap();
        sol.set_value(a.id(), "u_x", 0.0).unwrap();
        sol.set_value(b.id(), "u_x", 0.5 * l).unwrap();
        sol.set_value(a.id(), "r_z", 0.0).unwrap();
        sol.set_value(b.id(), "r_z", 0.25 * l).unwrap();
        let sol = NodeField::from_sub(sol);

        let materials = material_field(
            &model,
            &[("E", e), ("A", area), ("I", i), ("G", g), ("A_s", a_s)],
        )
        .unwrap();
        let def = beam_deformation(&sol, &fes, &materials).unwrap();

        let state = integrate(
            &model,
            &def,
            &initial_state(&model, &materials).unwrap(),
            &materials,
            None,
        )
        .unwrap();
        assert_eq!(state.len(), 1);
        let s = state.get(0).unwrap().read();
        // The state field carries the section forces **and** `Φ`: the exact beam gets
        // its interpolation from the material, and its residual reads that ratio again
        // at every iteration.
        assert_eq!(
            s.components(),
            &[
                "N".to_string(),
                "M".to_string(),
                "V".to_string(),
                "phi".to_string()
            ]
        );
        // Φ = 12EI/(G·A_s·L²) = 12·3·2/(5·2·4).
        assert!((s.value(0, 0, "phi").unwrap() - 1.8).abs() < 1e-12);
        let sub = fes.get(0).unwrap().read();
        // The axial force is constant — a bar's field really is linear — and so
        // is the shear, an unloaded span carrying a constant `V`.
        let v0 = s.value(0, 0, "V").unwrap();
        for gp in 0..s.gauss_count() {
            assert!((s.value(0, gp, "N").unwrap() - e * area * 0.5).abs() < 1e-12);
            assert!((s.value(0, gp, "V").unwrap() - v0).abs() < 1e-12);
        }
        // The **moment** varies: `M' = V`. What pins it is its integral, which
        // is `E·I·Δθ` whatever the shear compliance.
        let integral: f64 = (0..sub.gauss_count())
            .map(|gp| s.value(0, gp, "M").unwrap() * sub.gauss_weight(gp).unwrap())
            .sum::<f64>()
            * l
            / 2.0;
        assert!(
            (integral - e * i * 0.25 * l).abs() < 1e-10,
            "∫M = {integral}, expected {}",
            e * i * 0.25 * l
        );
    }
}
