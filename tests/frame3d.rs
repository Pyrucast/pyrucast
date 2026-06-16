//! Worked 3-D frame (space frame) example, exercised end-to-end through the
//! public API.
//!
//! A cantilever along the global X axis, clamped at the base (all 6 DOFs), with
//! tip loads in `Y`, `Z` and a torsion moment about `X`. The responses are
//! decoupled and — the closed-form Timoshenko element being nodally exact for
//! end loads — match the analytical values:
//!
//! ```text
//! u_y = P_y·L³/(3·E·I_z) + P_y·L/(G·A_sy)
//! u_z = P_z·L³/(3·E·I_y) + P_z·L/(G·A_sz)
//! r_x = M_x·L/(G·J)
//! ```
//!
//! Single source for the « cadre 3-D » example of the mechanics book chapter;
//! runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Configuration, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{assemble, build, mesher};
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn frame3d_cantilever_bending_and_torsion() -> Result<()> {
    const E: f64 = 1.0;
    const A: f64 = 1.0;
    const IY: f64 = 1.0;
    const IZ: f64 = 2.0;
    const J: f64 = 1.0;
    const G: f64 = 0.5;
    const ASY: f64 = 10.0;
    const ASZ: f64 = 10.0;
    const L: f64 = 1.0;
    const PY: f64 = 1.0;
    const PZ: f64 = 1.0;
    const MX: f64 = 1.0;
    const N: usize = 2;

    // ── Maillage : N éléments SEG2 le long de l'axe X (config 3-D) ─────────
    let cfg = insert(Configuration::new(3)?);
    let h = L / N as f64;
    let nodes: Vec<Node> = (0..=N)
        .map(|i| Node::create_in(cfg.clone(), &[i as f64 * h, 0.0, 0.0]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
    for i in 0..N {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : cadre 3-D + encastrement complet (6 DOFs) à la base ───────
    let clamp = |node: &Node, var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesher::barycenter(&imposed)?;
        Model::dirichlet(var.into(), dual.into(), &imposed, &multiplier, None, None)
    };
    let mut model = Model::frame3d(&fes)?;
    for (var, dual) in [
        ("u_x", "f_x"),
        ("u_y", "f_y"),
        ("u_z", "f_z"),
        ("r_x", "m_x"),
        ("r_y", "m_y"),
        ("r_z", "m_z"),
    ] {
        model = model.union(&clamp(&nodes[0], var, dual)?)?;
    }

    let materials = build::material_field(
        &model,
        &[
            ("E", E), ("A", A), ("I_y", IY), ("I_z", IZ), ("J", J), ("G", G),
            ("A_sy", ASY), ("A_sz", ASZ),
        ],
    )?;

    // ── Chargement : f_y, f_z et m_x au bout libre ─────────────────────────
    let mut load_sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    load_sm.add_cell(&[nodes[N].id()])?;
    let load_sm = insert(load_sm);
    let mut rhs =
        SubNodeField::from_poi1(&load_sm, vec!["f_y".into(), "f_z".into(), "m_x".into()])?;
    rhs.set_value(nodes[N].id(), "f_y", PY)?;
    rhs.set_value(nodes[N].id(), "f_z", PZ)?;
    rhs.set_value(nodes[N].id(), "m_x", MX)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let solution = solve(&assemble::stiffness(&model, &materials)?, &rhs)?;

    // ── Comparaison à l'analytique (élément exact ⇒ nodalement exact) ──────
    let tip = nodes[N].id();
    let uy = PY * L.powi(3) / (3.0 * E * IZ) + PY * L / (G * ASY);
    let uz = PZ * L.powi(3) / (3.0 * E * IY) + PZ * L / (G * ASZ);
    let rx = MX * L / (G * J);
    let tol = 1e-9;
    assert!((solution.value(tip, "u_y")? - uy).abs() < tol, "u_y");
    assert!((solution.value(tip, "u_z")? - uz).abs() < tol, "u_z");
    assert!((solution.value(tip, "r_x")? - rx).abs() < tol, "r_x");
    // Axial DOF stays put (no axial load).
    assert!(solution.value(tip, "u_x")?.abs() < tol, "u_x ≈ 0");
    Ok(())
}
// ANCHOR_END: example
