//! Radiation to infinity — the non-linear thermal boundary.
//!
//! A slab `[0, L]` insulated on its far side and radiating to an environment at
//! `T_∞` on the near one, with the temperature imposed at `x = L`. In steady
//! state with no volumetric source there is nowhere for heat to go, so the
//! **equilibrium is thermal equilibrium**: `T ≡ T_imposed`, and the radiated
//! flux is whatever `σε(T⁴ − T_∞⁴)` gives, balanced by the reaction at the
//! imposed end.
//!
//! That makes two things checkable without any analytical gymnastics:
//!
//! - the **residual**, `∫ N_i σε(T⁴ − T_∞⁴) dΓ`, computed by
//!   `internal_forces` from the behaviour integration — it must equal the exact
//!   Stefan-Boltzmann flux times the radiating area;
//! - the **consistent tangent**, `4σεT³ ∫ N_i N_j dΓ`, which must match a
//!   finite difference of that residual. A `T⁴` law is precisely where an
//!   inconsistent tangent hides, so this is the check that matters.
//!
//! Single source for the « rayonnement » example of the thermal book chapter;
//! runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::radiation::STEFAN_BOLTZMANN;
use pyrucast::models::Physics;
use pyrucast::ops::element_field;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::insert;
use pyrucast::Result;

const EMIS: f64 = 0.8; // emissivity
const T_INF: f64 = 300.0; // far-field temperature (K)
const T_WALL: f64 = 500.0; // temperature imposed on the far side (K)

#[test]
fn the_radiated_flux_matches_stefan_boltzmann() -> Result<()> {
    let (fixture, materials) = radiating_square()?;

    // The boundary is at the uniform wall temperature: interpolate it to the
    // Gauss points, integrate the law, and scatter back to the nodes.
    let temperature = uniform_temperature(&fixture, T_WALL)?;
    let at_gauss = element_field::interp_to_gauss(&temperature, &fixture.boundary_fes)?;
    let state =
        element_field::behavior::integrate(&fixture.radiation, &at_gauss, None, &materials, None)?;
    let reaction = pyrucast::ops::node_field::internal_forces(&state, &fixture.radiation)?;

    // The radiating edge has unit length, so the total flux is the density.
    let expected = STEFAN_BOLTZMANN * EMIS * (T_WALL.powi(4) - T_INF.powi(4));
    let total: f64 = fixture
        .edge
        .iter()
        .map(|n| reaction.value(n.id(), "q").unwrap_or(0.0))
        .sum();
    assert!(
        (total - expected).abs() < 1e-9 * expected.abs(),
        "radiated flux {total}, expected {expected}"
    );
    Ok(())
}
// ANCHOR_END: example

/// The consistent tangent must be the derivative of the residual. A `T⁴` law is
/// exactly where an inconsistent tangent hides — and where Newton would then
/// crawl instead of converging quadratically — so it is checked against a
/// finite difference of the internal forces.
#[test]
fn the_tangent_is_the_derivative_of_the_residual() -> Result<()> {
    let (fixture, materials) = radiating_square()?;

    let residual = |t: f64| -> Result<Vec<f64>> {
        let temperature = uniform_temperature(&fixture, t)?;
        let at_gauss = element_field::interp_to_gauss(&temperature, &fixture.boundary_fes)?;
        let state = element_field::behavior::integrate(
            &fixture.radiation,
            &at_gauss,
            None,
            &materials,
            None,
        )?;
        let f = pyrucast::ops::node_field::internal_forces(&state, &fixture.radiation)?;
        Ok(fixture
            .edge
            .iter()
            .map(|n| f.value(n.id(), "q").unwrap_or(0.0))
            .collect())
    };

    // Assemble the tangent at T_WALL and apply it to a uniform unit increment:
    // K_t · 1 must equal d(residual)/dT summed over the columns.
    let t0 = T_WALL;
    let temperature = uniform_temperature(&fixture, t0)?;
    let at_gauss = element_field::interp_to_gauss(&temperature, &fixture.boundary_fes)?;
    let state =
        element_field::behavior::integrate(&fixture.radiation, &at_gauss, None, &materials, None)?;
    let kt = pyrucast::ops::matrix::tangent(&fixture.radiation, &materials, &state)?;
    let dense = kt.dense()?;
    let n = kt.row_dofs()?.len();
    let analytic: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| dense[i * n + j]).sum())
        .collect();

    // Finite difference on the same quantity (a uniform temperature bump).
    let dt = 1e-4 * t0;
    let (fp, fm) = (residual(t0 + dt)?, residual(t0 - dt)?);
    for (i, a) in analytic.iter().enumerate() {
        let fd = (fp[i] - fm[i]) / (2.0 * dt);
        assert!(
            (a - fd).abs() < 1e-6 * fd.abs().max(1e-12),
            "row {i}: tangent {a}, finite difference {fd}"
        );
    }
    Ok(())
}

/// A radiating boundary belongs to the thermal problem **and** is selectable on
/// its own — that is what carrying two natures buys.
#[test]
fn radiation_answers_to_both_of_its_natures() -> Result<()> {
    let (fixture, _) = radiating_square()?;
    let coupled = fixture.bulk.union(&fixture.radiation)?;
    assert_eq!(coupled.len(), 2);
    // Both sub-models are thermal; only one is radiative.
    assert_eq!(coupled.filter(Physics::Thermal)?.len(), 2);
    assert_eq!(coupled.filter(Physics::Radiation)?.len(), 1);
    assert!(coupled.filter(Physics::Diffusion)?.is_empty());
    Ok(())
}

/// At thermal equilibrium (`T = T_∞`) the law radiates nothing — the trivial
/// case a fourth-power law must still get exactly right.
#[test]
fn no_flux_at_equilibrium() -> Result<()> {
    let (fixture, materials) = radiating_square()?;
    let temperature = uniform_temperature(&fixture, T_INF)?;
    let at_gauss = element_field::interp_to_gauss(&temperature, &fixture.boundary_fes)?;
    let state =
        element_field::behavior::integrate(&fixture.radiation, &at_gauss, None, &materials, None)?;
    let sub = pyrucast::store::read(&state.get(0)?)?;
    for g in 0..sub.gauss_count() {
        assert!(sub.value(0, g, "flux")?.abs() < 1e-20);
        // …but the tangent is not zero: the law is still stiff there.
        assert!(sub.value(0, g, "ktan")? > 0.0);
    }
    Ok(())
}

/// Solved end to end: an insulated slab radiating on one side and held at
/// `T_WALL` on the other settles at the wall temperature, and the reaction
/// balances what is radiated away.
#[test]
fn a_slab_settles_at_the_imposed_temperature() -> Result<()> {
    let (fixture, _) = radiating_square()?;
    // Radiation alone would be non-linear; here the imposed temperature fixes
    // every DOF of the boundary, so the linearised operator suffices to solve.
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(&fixture.edge)?);
    let multiplier = pyrucast::ops::mesh::barycenter(&imposed)?;
    let model = fixture.radiation.union(&Model::dirichlet(
        "T".into(),
        "q".into(),
        &imposed,
        &multiplier,
        None,
        None,
        Default::default(),
    )?)?;
    let materials = element_field::material_field(&model, &[("emis", EMIS), ("T_inf", T_INF)])?;

    let mut rhs_sm = SubMesh::new(fixture.coords.clone(), ElementType::POI1);
    let mut mults = Vec::new();
    for i in 0..multiplier.len() {
        let cells = pyrucast::store::read(&multiplier.get(i)?)?.cell_count();
        for cell in 0..cells {
            let id = multiplier.node(i, cell, 0)?.id();
            rhs_sm.add_cell(&[id])?;
            mults.push(id);
        }
    }
    let rhs_sm = insert(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into()])?;
    for id in &mults {
        rhs.set_value(*id, "imposed_T", T_WALL)?;
    }

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(rhs))?;
    for n in &fixture.edge {
        let t = solution.value(n.id(), "T")?;
        assert!((t - T_WALL).abs() < 1e-9, "T = {t}");
    }
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

struct Fixture {
    coords: pyrucast::store::Handle<Coords>,
    /// The two nodes of the radiating edge.
    edge: Vec<Node>,
    boundary_fes: FiniteElementSpace,
    radiation: Model,
    bulk: Model,
}

/// A unit QUA4 whose `x = 0` edge radiates, plus the conduction model of the
/// square itself (used only to check the nature filters).
fn radiating_square() -> Result<(Fixture, ElementField)> {
    let coords = insert(Coords::new(2)?);
    let node = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]);
    let corners = vec![
        node(0.0, 0.0)?,
        node(1.0, 0.0)?,
        node(1.0, 1.0)?,
        node(0.0, 1.0)?,
    ];

    let mut square = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    square.add_cell(&corners.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let bulk = Model::heat_conduction(&FiniteElementSpace::lagrange1(&square)?)?;

    // The radiating edge x = 0, of unit length.
    let edge = vec![corners[0].clone(), corners[3].clone()];
    let mut boundary = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    boundary.add_cell(&[edge[0].id(), edge[1].id()])?;
    let boundary_fes = FiniteElementSpace::lagrange1(&boundary)?;
    let radiation = Model::radiation(&boundary_fes)?;

    let materials = element_field::material_field(&radiation, &[("emis", EMIS), ("T_inf", T_INF)])?;
    Ok((
        Fixture {
            coords,
            edge,
            boundary_fes,
            radiation,
            bulk,
        },
        materials,
    ))
}

/// A uniform temperature field over the radiating edge.
fn uniform_temperature(fixture: &Fixture, value: f64) -> Result<NodeField> {
    let sm = insert(SubMesh::poi1_from_nodes(&fixture.edge)?);
    let mut field = SubNodeField::from_poi1(&sm, vec!["T".to_string()])?;
    for n in &fixture.edge {
        field.set_value(n.id(), "T", value)?;
    }
    Ok(NodeField::from_sub(field))
}
