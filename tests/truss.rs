//! Worked truss example, exercised end-to-end through the public API.
//!
//! A single horizontal bar of length `L`, section `A`, Young's modulus `E`,
//! clamped at the left end (`u_x = u_y = 0`) and supported transversally at the
//! right end (`u_y = 0`). An axial force `F` is applied at the right end. The
//! bar carries axial force only, so the analytical elongation is
//! `u_x = F·L / (E·A)`, and the support reaction equals `−F`.
//!
//! Single source for the « barre » example of the mechanics book chapter; runs
//! under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{assemble, build, mesher};
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn truss_bar_recovers_axial_elongation() -> Result<()> {
    const E: f64 = 210.0e9; // Young's modulus (Pa)
    const A: f64 = 1.0e-4; // section area (m²)
    const L: f64 = 2.0; // length (m)
    const F: f64 = 1000.0; // axial force at the right end (N)

    // ── Maillage : une barre SEG2 horizontale ──────────────────────────────
    let coords = insert(Coords::new(2)?);
    let n0 = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let n1 = Node::create_in(coords.clone(), &[L, 0.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    mesh.add_cell(&[n0.id(), n1.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : barre + appuis (Dirichlet homogènes) ──────────────────────
    // Homogeneous (u = 0) BCs: the imposed value defaults to 0, so we only need
    // to introduce the constraint. A bar has no transverse stiffness, hence
    // `u_y` is clamped at both nodes to make the system well-posed.
    let clamp = |node: &Node, var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesher::barycenter(&imposed)?;
        Model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
    };
    let mut model = Model::truss(&fes)?;
    model = model.union(&clamp(&n0, "u_x", "f_x")?)?;
    model = model.union(&clamp(&n0, "u_y", "f_y")?)?;
    model = model.union(&clamp(&n1, "u_y", "f_y")?)?;

    // ── Matériau E, A (Dirichlet ignoré automatiquement) ───────────────────
    let materials = build::material_field(&model, &[("E", E), ("A", A)])?;

    // ── Chargement : force axiale F au nœud droit ──────────────────────────
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[n1.id()])?;
    let load_sm = insert(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_x".into()])?;
    rhs.set_value(n1.id(), "f_x", F)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let stiffness = assemble::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison à l'analytique : u_x = F·L / (E·A) ─────────────────────
    let expected = F * L / (E * A);
    let ux = solution.value(n1.id(), "u_x")?;
    assert!(
        (ux - expected).abs() < 1e-10 * expected,
        "u_x = {ux}, attendu {expected}"
    );
    // Le nœud gauche est encastré, le bout droit ne bouge pas transversalement.
    assert!(solution.value(n0.id(), "u_x")?.abs() < 1e-18);
    assert!(solution.value(n1.id(), "u_y")?.abs() < 1e-18);

    Ok(())
}
// ANCHOR_END: example
