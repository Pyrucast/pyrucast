//! Worked thermal example, exercised end-to-end through the **public** API.
//!
//! Steady 1-D conduction on `[0, 1]` with a heat source (Neumann flux `Q`) at
//! `x = 0` and an imposed temperature `T = 20` (Dirichlet) at `x = 1`. With no
//! volumetric generation the field is linear; the analytical solution is
//! `u(x) = 20 + (Q/k)·(1 − x)`, and the Lagrange multiplier (the reaction at
//! the imposed end) equals the injected source `Q` — a discrete energy balance.
//!
//! This file is the single source for the example shown in the book chapter
//! *« Conduction thermique »* (`book/src/thermique.md`), included verbatim via
//! the `example` anchor. Because it is a real integration test, it runs with
//! the rest of the suite under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::mesh;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

#[test]
fn thermal_line_recovers_analytical_solution() -> Result<()> {
    // ── Données du problème ────────────────────────────────────────────────
    const K: f64 = 1.0; // conductivité
    const Q: f64 = 10.0; // source de chaleur (flux de Neumann) en x = 0
    const T_IMPOSED: f64 = 20.0; // température imposée en x = 1
    const N_ELEMS: usize = 4;
    let h = 1.0 / N_ELEMS as f64;

    // ── Maillage : une ligne de SEG2 sur [0, 1] ────────────────────────────
    let coords = Handle::new(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N_ELEMS)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N_ELEMS {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : conduction + Dirichlet T = 20 en x = 1 ────────────────────
    // Le support des multiplicateurs est fabriqué depuis le nœud imposé par le
    // mesher `barycenter` (un nœud neuf colocalisé). Le modèle ne crée rien.
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(
        nodes.last().unwrap(),
    ))?);
    let multiplier = mesh::barycenter(&imposed)?;
    let mult = multiplier.node(0, 0, 0)?.id();

    let conduction = Model::heat_conduction(&fes)?;
    let dirichlet = Model::dirichlet(
        "T".into(),
        "q".into(),
        &imposed,
        &multiplier,
        None,
        None,
        Default::default(),
    )?;
    let model = conduction.union(&dirichlet)?;

    // ── Matériau : k uniforme (Dirichlet est ignoré automatiquement) ───────
    let materials = pyrucast::ops::element_field::material_field(&model, &[("k", K)])?;

    // ── Chargement : source Q en x = 0 (composante duale "q"), valeur imposée
    //    T = 20 au nœud-multiplicateur (slot "imposed_T") ───────────────────
    let node0 = nodes[0].id();
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[node0])?;
    load_sm.add_cell(&[mult])?;
    let load_sm = Handle::new(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["imposed_T".into(), "q".into()])?;
    rhs.set_value(node0, "q", Q)?;
    rhs.set_value(mult, "imposed_T", T_IMPOSED)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison à la solution analytique u(x) = 20 + (Q/k)(1 − x) ──────
    let tol = 1e-10;
    for (i, node) in nodes.iter().enumerate() {
        let x = i as f64 * h;
        let expected = T_IMPOSED + (Q / K) * (1.0 - x);
        let got = solution.value(node.id(), "T")?;
        assert!(
            (got - expected).abs() < tol,
            "T(x={x}) : obtenu {got}, attendu {expected}"
        );
    }
    // La réaction (multiplicateur de Lagrange) équilibre le flux injecté : λ = Q.
    let reaction = solution.value(mult, "lambda_T")?;
    assert!(
        (reaction - Q).abs() < tol,
        "réaction λ : obtenue {reaction}, attendue {Q}"
    );

    Ok(())
}
// ANCHOR_END: example
