//! Worked planar-frame (portique) example, exercised end-to-end through the
//! public API.
//!
//! A cantilever inclined at **45°**, clamped at the base (`u_x = u_y = rz = 0`),
//! with a tip load **perpendicular** to the beam axis. The load being purely
//! transverse, the tip deflects along that perpendicular by the Timoshenko
//! amount `δ = P·L³/(3·E·I) + P·L/(G·A_s)`; the global tip displacement is
//! `δ` projected on the perpendicular direction. This exercises the
//! local→global rotation of the frame element.
//!
//! Single source for the « portique » example of the mechanics book chapter;
//! runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

#[test]
fn frame_inclined_cantilever_perpendicular_load() -> Result<()> {
    const E: f64 = 1.0;
    const A: f64 = 1.0;
    const I: f64 = 1.0;
    const G: f64 = 30.0;
    const A_S: f64 = 1.0;
    const L: f64 = 1.0;
    const P: f64 = 1.0;
    const N: usize = 40;

    // Beam direction (45°) and perpendicular.
    let (c, s) = (
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    let (px, py) = (-s, c); // unit perpendicular
    let h = L / N as f64;

    // ── Maillage : N éléments SEG2 le long de la direction à 45° ───────────
    let coords = Handle::new(Coords::new(2)?);
    let nodes: Vec<Node> = (0..=N)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h * c, i as f64 * h * s]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::new(&mesh, Interpolation::ModelEmbedded)?;

    // ── Modèle : portique + encastrement complet à la base ─────────────────
    let clamp = |target: &Model, node: &Node, var: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesh::barycenter(&imposed)?;
        model::dirichlet(target, var, &imposed, &multiplier, Default::default())
    };
    let mut model = model::timoshenko(&fes)?;
    model = model.union(&clamp(&model, &nodes[0], "u_x")?)?;
    model = model.union(&clamp(&model, &nodes[0], "u_y")?)?;
    model = model.union(&clamp(&model, &nodes[0], "r_z")?)?;

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("A", A), ("I", I), ("G", G), ("A_s", A_S)],
    )?;

    // ── Chargement : force P perpendiculaire à la poutre, au bout libre ────
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[nodes[N].id()])?;
    let load_sm = Handle::new(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_x".into(), "f_y".into()])?;
    rhs.set_value(nodes[N].id(), "f_x", P * px)?;
    rhs.set_value(nodes[N].id(), "f_y", P * py)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let solution = solve(&pyrucast::ops::matrix::stiffness(&model, &materials)?, &rhs)?;

    // ── Comparaison : déplacement du bout = δ·(perpendiculaire) ────────────
    let delta = P * L.powi(3) / (3.0 * E * I) + P * L / (G * A_S);
    let ux = solution.value(nodes[N].id(), "u_x")?;
    let uy = solution.value(nodes[N].id(), "u_y")?;
    // Projection sur la perpendiculaire (= δ) et sur l'axe (≈ 0).
    let transverse = ux * px + uy * py;
    let axial = ux * c + uy * s;
    assert!(
        (transverse - delta).abs() < 1e-2 * delta,
        "transverse {transverse} ≠ {delta}"
    );
    assert!(axial.abs() < 1e-6, "déplacement axial {axial} ≈ 0");
    Ok(())
}
// ANCHOR_END: example
