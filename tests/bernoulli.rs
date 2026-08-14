//! Euler-Bernoulli beams, against their closed-form solutions.
//!
//! A beam element earns its keep by being **nodally exact**: with Hermite cubics
//! the finite-element answer at the nodes is the analytical one, for any load
//! that leaves the span free of distributed forces. So the tests compare to the
//! textbook formulas at full precision, not within a discretisation tolerance —
//! and one element is enough.
//!
//! | case | closed form |
//! |---|---|
//! | cantilever, tip load `P` | `w = PL³/3EI`, `θ = PL²/2EI` |
//! | cantilever, tip moment `M` | `w = ML²/2EI`, `θ = ML/EI` |
//! | plane frame, axial pull | `u = NL/EA`, uncoupled from bending |
//! | space frame, torque | `φ = TL/GJ` |
//!
//! The last test is the one that separates this physics from Timoshenko: a
//! **stocky** beam deflects visibly more when shear is accounted for, and
//! Bernoulli is exactly the theory that says it does not.
//!
//! Single source for the « poutre d'Euler-Bernoulli » example of the book; runs
//! under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::ops::mesh;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::{insert, read};
use pyrucast::Result;

const E: f64 = 210_000.0;
const I: f64 = 1.0e-4;
const L: f64 = 2.0;

#[test]
fn a_cantilever_under_a_tip_load_matches_its_closed_form() -> Result<()> {
    const P: f64 = 50.0;
    let coords = insert(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[L])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;

    // Clamped at A: both the deflection and the rotation are held.
    let mut model = Model::bernoulli(&fes)?;
    for (var, dual) in [("w", "f_w"), ("theta", "m_theta")] {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a))?);
        let multiplier = mesh::barycenter(&imposed)?;
        model = model.union(&Model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("I", I)])?;

    // A point load at the free end.
    let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_w".into()])?;
    rhs.set_value(b.id(), "f_w", P)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(rhs))?;

    // Nodally exact: w = PL³/3EI, θ = PL²/2EI, to machine precision.
    let w = solution.value(b.id(), "w")?;
    let theta = solution.value(b.id(), "theta")?;
    let w_exact = P * L.powi(3) / (3.0 * E * I);
    let theta_exact = P * L * L / (2.0 * E * I);
    assert!(
        (w - w_exact).abs() < 1e-12 * w_exact,
        "w = {w}, exact {w_exact}"
    );
    assert!(
        (theta - theta_exact).abs() < 1e-12 * theta_exact,
        "θ = {theta}, exact {theta_exact}"
    );
    Ok(())
}
// ANCHOR_END: example

/// A tip **moment** bends the beam into a circular arc — a different closed form,
/// and one a wrong sign in the Hermite matrix would get wrong while still
/// passing the tip-load case.
#[test]
fn a_tip_moment_matches_its_closed_form() -> Result<()> {
    const M: f64 = 30.0;
    let (a, b, fes, coords) = beam_1d()?;
    let _ = coords;
    let model = clamped_1d(&a, &fes)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("I", I)])?;

    let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["m_theta".into()])?;
    rhs.set_value(b.id(), "m_theta", M)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(rhs))?;

    let w_exact = M * L * L / (2.0 * E * I);
    let theta_exact = M * L / (E * I);
    assert!((solution.value(b.id(), "w")? - w_exact).abs() < 1e-12 * w_exact);
    assert!((solution.value(b.id(), "theta")? - theta_exact).abs() < 1e-12 * theta_exact);
    Ok(())
}

/// The plane frame adds an axial term that must stay **uncoupled** from bending
/// for a straight member: pulling it lengthens it and bends it not at all.
#[test]
fn a_plane_frame_carries_axial_and_bending_independently() -> Result<()> {
    const A: f64 = 1e-2;
    const N: f64 = 1_000.0;
    let coords = insert(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[L, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;

    let mut model = Model::bernoulli(&fes)?;
    for (var, dual) in [("u_x", "f_x"), ("u_y", "f_y"), ("r_z", "m_z")] {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a))?);
        let multiplier = mesh::barycenter(&imposed)?;
        model = model.union(&Model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", E), ("A", A), ("I", I)])?;

    let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_x".into()])?;
    rhs.set_value(b.id(), "f_x", N)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(rhs))?;

    let u_exact = N * L / (E * A);
    assert!((solution.value(b.id(), "u_x")? - u_exact).abs() < 1e-12 * u_exact);
    assert!(
        solution.value(b.id(), "u_y")?.abs() < 1e-14,
        "an axial pull must not bend a straight member"
    );
    Ok(())
}

/// The space frame adds torsion, whose closed form `φ = TL/GJ` is independent of
/// everything else.
#[test]
fn a_space_frame_twists_by_its_closed_form() -> Result<()> {
    const G: f64 = 80_000.0;
    const J: f64 = 2e-5;
    const T: f64 = 40.0;
    let coords = insert(Coords::new(3)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[L, 0.0, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;

    let mut model = Model::bernoulli(&fes)?;
    for (var, dual) in [
        ("u_x", "f_x"),
        ("u_y", "f_y"),
        ("u_z", "f_z"),
        ("r_x", "m_x"),
        ("r_y", "m_y"),
        ("r_z", "m_z"),
    ] {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a))?);
        let multiplier = mesh::barycenter(&imposed)?;
        model = model.union(&Model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("E", E),
            ("A", 1e-2),
            ("I_y", I),
            ("I_z", I),
            ("J", J),
            ("G", G),
        ],
    )?;

    let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["m_x".into()])?;
    rhs.set_value(b.id(), "m_x", T)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(rhs))?;

    let phi_exact = T * L / (G * J);
    let phi = solution.value(b.id(), "r_x")?;
    assert!((phi - phi_exact).abs() < 1e-12 * phi_exact, "φ = {phi}");
    Ok(())
}

/// The physics that separates Bernoulli from Timoshenko: a **stocky** beam
/// deflects more when the shear compliance is kept. Bernoulli is exactly the
/// theory that says it does not, so the two must disagree here — and agree for a
/// slender one.
#[test]
fn bernoulli_and_timoshenko_differ_only_for_a_stocky_beam() -> Result<()> {
    const P: f64 = 50.0;
    const G: f64 = 80_000.0;
    // Both elements are now nodally exact, so one per member would do. The mesh
    // is kept anyway: it costs nothing and it proves the comparison is about the
    // two **theories**, not about either one's discretisation.
    const N_ELEMS: usize = 40;
    let tip = |shear_area: f64, length: f64| -> Result<(f64, f64)> {
        let coords = insert(Coords::new(1)?);
        let h = length / N_ELEMS as f64;
        let nodes: Vec<Node> = (0..=N_ELEMS)
            .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
            .collect::<Result<_>>()?;
        let (a, b) = (nodes[0].clone(), nodes[N_ELEMS].clone());
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        for i in 0..N_ELEMS {
            mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
        }
        // The two theories no longer share a space: Bernoulli interpolates its
        // deflection with cubic Hermite functions, Timoshenko with linear
        // Lagrange ones. Two spaces over the **same** mesh is exactly what the
        // comparison means.
        let fes_bern = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;
        let fes = FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded)?;

        let clamp = |model: Model| -> Result<Model> {
            let mut m = model;
            for (var, dual) in [("w", "f_w"), ("theta", "m_theta")] {
                let imposed =
                    Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a))?);
                let multiplier = mesh::barycenter(&imposed)?;
                m = m.union(&Model::dirichlet(
                    var.into(),
                    dual.into(),
                    &imposed,
                    &multiplier,
                    None,
                    None,
                    Default::default(),
                )?)?;
            }
            Ok(m)
        };
        let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
        let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_w".into()])?;
        rhs.set_value(b.id(), "f_w", P)?;
        let rhs = NodeField::from_sub(rhs);

        let bern = clamp(Model::bernoulli(&fes_bern)?)?;
        let bern_mat = pyrucast::ops::element_field::material_field(&bern, &[("E", E), ("I", I)])?;
        let w_bern = solve(&pyrucast::ops::matrix::stiffness(&bern, &bern_mat)?, &rhs)?
            .value(b.id(), "w")?;

        let timo = clamp(Model::timoshenko(&fes)?)?;
        let timo_mat = pyrucast::ops::element_field::material_field(
            &timo,
            &[("E", E), ("I", I), ("G", G), ("A_s", shear_area)],
        )?;
        let w_timo = solve(&pyrucast::ops::matrix::stiffness(&timo, &timo_mat)?, &rhs)?
            .value(b.id(), "w")?;
        Ok((w_bern, w_timo))
    };

    // Slender: the shear contribution is negligible, the two agree closely.
    let (slender_b, slender_t) = tip(1e-2, 20.0)?;
    assert!(
        (slender_t - slender_b).abs() < 0.01 * slender_b,
        "slender: {slender_b} vs {slender_t}"
    );
    // Stocky: shear matters, and Timoshenko is the softer of the two.
    let (stocky_b, stocky_t) = tip(1e-2, 0.5)?;
    assert!(
        stocky_t > 1.10 * stocky_b,
        "stocky: Timoshenko {stocky_t} should be well above Bernoulli {stocky_b}"
    );
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
fn beam_1d() -> Result<(
    Node,
    Node,
    FiniteElementSpace,
    pyrucast::store::Handle<Coords>,
)> {
    let coords = insert(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[L])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;
    Ok((a, b, fes, coords))
}

fn clamped_1d(a: &Node, fes: &FiniteElementSpace) -> Result<Model> {
    let mut model = Model::bernoulli(fes)?;
    for (var, dual) in [("w", "f_w"), ("theta", "m_theta")] {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(a))?);
        let multiplier = mesh::barycenter(&imposed)?;
        model = model.union(&Model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    Ok(model)
}

/// The section forces of a Bernoulli beam — the chain
/// `beam_deformation → integrate_behavior`, which no test exercised until the
/// day it turned out not to work at all.
///
/// A cantilever under a tip moment `M` carries that **same moment everywhere**:
/// `M' = V = 0`. It is the one loading whose moment a beam of any theory must
/// reproduce exactly, and it pins both ends of the chain at once.
///
/// The recovery asks for a material, since the curvature distribution depends
/// on `Φ = 12EI/(G·A_s·L²)`. A Bernoulli material carries no `G` and no `A_s` —
/// its theory has no shear — and that **absence** is what says `Φ = 0`. The
/// operator therefore needs no model to tell the two theories apart. Before
/// this test it demanded `G` outright and a Bernoulli beam could not recover
/// its own forces.
#[test]
fn a_bernoulli_beam_recovers_its_section_forces() -> Result<()> {
    const M: f64 = 30.0;
    let (a, b, fes, _coords) = beam_1d()?;
    let model = clamped_1d(&a, &fes)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("I", I)])?;

    let load_sm = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b))?);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["m_theta".into()])?;
    rhs.set_value(b.id(), "m_theta", M)?;
    let rhs = NodeField::from_sub(rhs);
    let u = solve(&pyrucast::ops::matrix::stiffness(&model, &materials)?, &rhs)?;

    let strains = pyrucast::ops::element_field::beam_deformation(&u, &fes, &materials)?;
    let forces = pyrucast::ops::element_field::behavior::integrate(
        &model, &strains, None, &materials, None,
    )?;
    let f = read(&forces.get(0)?)?;
    assert_eq!(f.components(), &["M".to_string()]);
    for g in 0..f.gauss_count() {
        let m = f.value(0, g, "M")?;
        assert!(
            (m - M).abs() < 1e-9 * M,
            "a tip moment is carried unchanged along the span: M({g}) = {m}"
        );
    }
    Ok(())
}
