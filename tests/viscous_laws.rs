//! Creep and viscoplasticity — the rate-**dependent** laws.
//!
//! What distinguishes these from the plastic laws is that the answer depends on
//! `dt`. So every test here turns on time, not on a stress value:
//!
//! | law | what is asserted |
//! |---|---|
//! | all of them | integrating without `dt` **errors** rather than guessing |
//! | Norton | the flow obeys `ṗ = (q/K)^n` — checked against the closed form for a short step, and the stress relaxes further the longer it is held |
//! | Lemaitre | the rate **decreases** with accumulated strain (primary creep) |
//! | Blackburn | the primary strain saturates towards its asymptote while the secondary term keeps going |
//! | Chaboche | the back stress makes reverse yielding early — the Bauschinger effect, which no isotropic law produces |
//! | Lemaitre-Chaboche | damage grows with plastic strain, and a damaged material flows faster |
//!
//! Single source for the « fluage et viscoplasticité » examples of the book;
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
use pyrucast::handle::Handle;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::models::plastic::PlasticLaw;
use pyrucast::ops::element_field::{behavior::integrate, deformation, material_field};
use pyrucast::ops::model;
use pyrucast::Result;

const AXES: [&str; 3] = ["x", "y", "z"];

/// Norton: a steady creep with a strongly non-linear stress dependence.
const NORTON: &[(&str, f64)] = &[("E", 150_000.0), ("nu", 0.3), ("K", 400.0), ("n", 5.0)];

#[test]
fn norton_creep_follows_its_rate_law() -> Result<()> {
    let cube = Cube::new(PlasticLaw::CreepNorton, NORTON)?;
    // A step short enough that the stress barely relaxes, so the closed-form
    // rate at the trial stress is the right comparison.
    let dt = 1e-6;
    let state = cube.step(&uniaxial(2e-3), None, dt)?;
    let q = von_mises(&state.sigma);
    let expected = dt * (q / 400.0_f64).powf(5.0);
    assert!(
        (state.p - expected).abs() < 1e-3 * expected,
        "Δp = {}, expected dt·(q/K)^n = {expected}",
        state.p
    );
    Ok(())
}
// ANCHOR_END: example

/// Held at a fixed strain, a creeping material relaxes: the longer the step, the
/// lower the stress. That is the defining behaviour of creep, and it is what a
/// rate-independent law cannot do at all.
#[test]
fn holding_a_strain_relaxes_the_stress() -> Result<()> {
    let cube = Cube::new(PlasticLaw::CreepNorton, NORTON)?;
    let strain = uniaxial(2e-3);
    let mut previous = f64::INFINITY;
    for dt in [1e-4, 1e-3, 1e-2, 1e-1] {
        let q = von_mises(&cube.step(&strain, None, dt)?.sigma);
        assert!(
            q < previous,
            "dt = {dt}: q = {q} did not fall below {previous}"
        );
        previous = q;
    }
    Ok(())
}

/// A rate-dependent law integrated without a time increment would silently
/// produce a plausible wrong answer. It must refuse instead.
#[test]
fn a_viscous_law_refuses_to_integrate_without_dt() -> Result<()> {
    for (law, mat) in [
        (PlasticLaw::CreepNorton, NORTON),
        (PlasticLaw::CreepLemaitre, LEMAITRE),
        (PlasticLaw::CreepBlackburn, BLACKBURN),
        (PlasticLaw::ViscoplasticChaboche, CHABOCHE),
    ] {
        let cube = Cube::new(law, mat)?;
        let strain = deformation(&cube.displacement(&uniaxial(2e-3))?, &cube.fes)?;
        let err = integrate(&cube.model, &strain, None, &cube.materials, None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rate-dependent") && msg.contains("dt"),
            "{law}: unexpected message {msg}"
        );
    }
    Ok(())
}

/// Lemaitre primary creep, in one line: the **rate falls** as strain
/// accumulates. Feeding the law a state that has already crept must slow it.
const LEMAITRE: &[(&str, f64)] = &[
    ("E", 150_000.0),
    ("nu", 0.3),
    ("K", 400.0),
    ("N", 5.0),
    ("M", 0.3),
];

#[test]
fn lemaitre_creep_decelerates_with_accumulated_strain() -> Result<()> {
    let cube = Cube::new(PlasticLaw::CreepLemaitre, LEMAITRE)?;
    let strain = uniaxial(2e-3);
    let dt = 1e-4;
    let fresh = cube.step(&strain, None, dt)?;
    // Feed its own output back as the previous state, then step again.
    let crept = cube.step(&strain, Some(&fresh.field), dt)?;
    let (first, second) = (fresh.p, crept.p - fresh.p);
    assert!(
        second < first,
        "the rate must fall: {second} should be below {first}"
    );
    Ok(())
}

/// Blackburn's primary strain approaches its asymptote and stops, while the
/// secondary term keeps accumulating. Tracking the primary part separately is
/// what makes that possible.
const BLACKBURN: &[(&str, f64)] = &[
    ("E", 150_000.0),
    ("nu", 0.3),
    ("A_1", 1e-3),
    ("alpha_1", 0.01),
    ("r_1", 100.0),
    ("B_s", 1e-6),
    ("beta_s", 0.01),
];

#[test]
fn blackburn_primary_creep_saturates() -> Result<()> {
    let cube = Cube::new(PlasticLaw::CreepBlackburn, BLACKBURN)?;
    let strain = uniaxial(2e-3);
    let mut prev: Option<Step> = None;
    let mut primaries = Vec::new();
    for _ in 0..6 {
        let step = cube.step(&strain, prev.as_ref().map(|s| &s.field), 1e-2)?;
        primaries.push(step.var("p_prim")?);
        prev = Some(step);
    }
    // The primary strain climbs, and its increments shrink towards zero.
    let increments: Vec<f64> = primaries.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(primaries[0] > 0.0, "primary creep must start");
    assert!(
        increments.last().unwrap() < &(increments[0] * 0.5),
        "the primary increments must decay: {increments:?}"
    );
    Ok(())
}

/// Chaboche's back stress translates the yield surface, so reversing the load
/// yields **earlier** than it did forwards. That is the Bauschinger effect, and
/// it is exactly what an isotropic law cannot produce.
const CHABOCHE: &[(&str, f64)] = &[
    ("E", 150_000.0),
    ("nu", 0.3),
    ("k", 100.0),
    ("K", 150.0),
    ("n", 4.0),
    ("C_1", 60_000.0),
    ("gamma_1", 400.0),
    ("b", 10.0),
    ("Q", 50.0),
];

#[test]
fn the_back_stress_builds_up_under_load() -> Result<()> {
    let cube = Cube::new(PlasticLaw::ViscoplasticChaboche, CHABOCHE)?;
    let dt = 1e-2;
    // Load forwards in a few steps, accumulating the back stress.
    let mut prev: Option<Step> = None;
    for i in 1..=4 {
        let step = cube.step(
            &uniaxial(1e-3 * i as f64),
            prev.as_ref().map(|s| &s.field),
            dt,
        )?;
        prev = Some(step);
    }
    let loaded = prev.expect("four steps ran");
    let x_xx = loaded.var("X_xx")?;
    assert!(
        x_xx > 0.0,
        "the back stress must follow the loading direction (X_xx = {x_xx})"
    );
    // It saturates: γ is what stops it growing without bound.
    assert!(
        x_xx < 60_000.0 / 400.0,
        "the back stress must stay below C/γ (X_xx = {x_xx})"
    );
    Ok(())
}

/// The damageable variant: damage grows with plastic strain, and it never
/// exceeds the critical value it is capped at.
const LEMAITRE_CHABOCHE: &[(&str, f64)] = &[
    ("E", 150_000.0),
    ("nu", 0.3),
    ("k", 100.0),
    ("K", 150.0),
    ("n", 4.0),
    ("C_1", 60_000.0),
    ("gamma_1", 400.0),
    ("b", 10.0),
    ("Q", 50.0),
    ("S", 1.0),
    ("s", 1.0),
    ("D_c", 0.3),
];

#[test]
fn damage_grows_with_plastic_strain_and_is_capped() -> Result<()> {
    let cube = Cube::new(PlasticLaw::ViscoplasticLemaitreChaboche, LEMAITRE_CHABOCHE)?;
    let dt = 1e-2;
    let mut prev: Option<Step> = None;
    let mut damages = Vec::new();
    for _ in 0..8 {
        let step = cube.step(&uniaxial(4e-3), prev.as_ref().map(|s| &s.field), dt)?;
        damages.push(step.var("damage")?);
        prev = Some(step);
    }
    assert!(damages[0] > 0.0, "damage must start growing");
    for w in damages.windows(2) {
        assert!(w[1] >= w[0], "damage must not heal: {damages:?}");
    }
    assert!(
        *damages.last().unwrap() <= 0.3 + 1e-12,
        "damage must be capped at D_c: {damages:?}"
    );
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// One integrated step, keeping the whole state field so it can be fed back as
/// the `prev` of the next one.
struct Step {
    sigma: [f64; 6],
    p: f64,
    field: ElementField,
}

impl Step {
    /// An internal variable of the first Gauss point, by name.
    fn var(&self, name: &str) -> Result<f64> {
        let sub = self.field.get(0)?.read();
        sub.value(0, 0, name)
    }
}

fn von_mises(s: &[f64; 6]) -> f64 {
    let mean = (s[0] + s[1] + s[2]) / 3.0;
    let d = [s[0] - mean, s[1] - mean, s[2] - mean, s[3], s[4], s[5]];
    (1.5 * (d[0] * d[0]
        + d[1] * d[1]
        + d[2] * d[2]
        + 2.0 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5])))
        .sqrt()
}

const CORNERS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

fn uniaxial(e: f64) -> Vec<f64> {
    CORNERS.iter().flat_map(|c| [e * c[0], 0.0, 0.0]).collect()
}

/// A single HEX8 material point.
struct Cube {
    nodes: Vec<Node>,
    fes: FiniteElementSpace,
    model: Model,
    materials: ElementField,
}

impl Cube {
    fn new(law: PlasticLaw, material: &[(&str, f64)]) -> Result<Self> {
        let coords = Handle::new(Coords::new(3)?);
        let nodes: Vec<Node> = CORNERS
            .iter()
            .map(|c| Node::create_in(coords.clone(), c))
            .collect::<Result<_>>()?;
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
        mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
        let fes = FiniteElementSpace::lagrange1(&mesh)?;
        let model = model::plasticity_with_law(&fes, ElasticityModel::Solid, law)?;
        let materials = material_field(&model, material)?;
        Ok(Self {
            nodes,
            fes,
            model,
            materials,
        })
    }

    fn displacement(&self, disp: &[f64]) -> Result<NodeField> {
        let support = Handle::new(SubMesh::poi1_from_nodes(&self.nodes)?);
        let comps: Vec<String> = (0..3).map(|a| format!("u_{}", AXES[a])).collect();
        let mut u = SubNodeField::from_poi1(&support, comps)?;
        for (i, n) in self.nodes.iter().enumerate() {
            for a in 0..3 {
                u.set_value(n.id(), &format!("u_{}", AXES[a]), disp[i * 3 + a])?;
            }
        }
        Ok(NodeField::from_sub(u))
    }

    /// Integrate one step A → B at the given strain, from an optional previous
    /// state and over a time increment `dt`.
    fn step(&self, disp: &[f64], prev: Option<&ElementField>, dt: f64) -> Result<Step> {
        let strain = deformation(&self.displacement(disp)?, &self.fes)?;
        let field = integrate(&self.model, &strain, prev, &self.materials, Some(dt))?;
        let sub = field.get(0)?.read();
        let names = ["xx", "yy", "zz", "yz", "xz", "xy"];
        let mut sigma = [0.0; 6];
        for (i, n) in names.iter().enumerate() {
            sigma[i] = sub.value(0, 0, &format!("sigma_{n}"))?;
        }
        let p = sub.value(0, 0, "p")?;
        drop(sub);
        Ok(Step { sigma, p, field })
    }
}
