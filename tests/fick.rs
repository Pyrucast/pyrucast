//! Fickian diffusion, exercised end-to-end through the **public** API.
//!
//! Steady 1-D diffusion on `[0, 1]` with an injected species flux `J` at
//! `x = 0` and an imposed concentration `c = 1` at `x = 1`. With no volumetric
//! source the profile is linear,
//!
//! ```text
//! c(x) = 1 + (J/D)·(1 − x)
//! ```
//!
//! and the Lagrange multiplier at the imposed end equals the injected flux `J`
//! — the discrete mass balance, the exact counterpart of the energy balance of
//! the thermal chapter.
//!
//! The operator is the one of heat conduction; what differs is the **physics**:
//! primal `c` and dual `j` instead of `T` and `q`, and the nature
//! `Physics::Diffusion`, so `model.filter("diffusion")` picks it out of a
//! coupled model without dragging the thermal part along. That separation is
//! asserted at the end of the second test.
//!
//! Single source for the « diffusion » example of the book chapter; runs under
//! `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::Physics;
use pyrucast::ops::mesh;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn fick_line_recovers_the_linear_profile() -> Result<()> {
    const D: f64 = 2.0; // diffusivity
    const J: f64 = 10.0; // injected species flux at x = 0
    const C_IMPOSED: f64 = 1.0; // concentration imposed at x = 1
    const N_ELEMS: usize = 4;
    let h = 1.0 / N_ELEMS as f64;

    // ── Maillage : une ligne de SEG2 sur [0, 1] ────────────────────────────
    let coords = insert(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N_ELEMS)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N_ELEMS {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : diffusion + Dirichlet c = 1 en x = 1 ──────────────────────
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(
        nodes.last().unwrap(),
    ))?);
    let multiplier = mesh::barycenter(&imposed)?;
    let mult = multiplier.node(0, 0, 0)?.id();

    let diffusion = Model::fick(&fes)?;
    let dirichlet = Model::dirichlet(
        "c".into(),
        "j".into(),
        &imposed,
        &multiplier,
        None,
        None,
        Default::default(),
    )?;
    let model = diffusion.union(&dirichlet)?;

    // ── Matériau : diffusivité uniforme ────────────────────────────────────
    let materials = pyrucast::ops::element_field::material_field(&model, &[("D", D)])?;

    // ── Chargement : flux J en x = 0, concentration imposée au multiplicateur
    let node0 = nodes[0].id();
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[node0])?;
    load_sm.add_cell(&[mult])?;
    let load_sm = insert(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["imposed_c".into(), "j".into()])?;
    rhs.set_value(node0, "j", J)?;
    rhs.set_value(mult, "imposed_c", C_IMPOSED)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison au profil analytique c(x) = 1 + (J/D)(1 − x) ───────────
    let tol = 1e-10;
    for (i, node) in nodes.iter().enumerate() {
        let x = i as f64 * h;
        let expected = C_IMPOSED + (J / D) * (1.0 - x);
        let got = solution.value(node.id(), "c")?;
        assert!(
            (got - expected).abs() < tol,
            "c(x={x}) : {got} ≠ {expected}"
        );
    }
    // Bilan de matière : la réaction au bord imposé équilibre le flux injecté.
    let reaction = solution.value(mult, "lambda_c")?;
    assert!((reaction - J).abs() < tol, "réaction λ : {reaction} ≠ {J}");
    Ok(())
}
// ANCHOR_END: example

/// Diffusion and conduction on the same mesh: two physics, two variable pairs,
/// two natures. The point is that they neither collide nor need consolidating —
/// the assembler resolves each material zone by the **components** its physics
/// requires (`D` here, `k` there), and `filter` separates the assembled model
/// again afterwards.
#[test]
fn diffusion_and_conduction_coexist_and_filter_apart() -> Result<()> {
    const D: f64 = 2.0;
    const K: f64 = 5.0;
    const N_ELEMS: usize = 3;
    let h = 1.0 / N_ELEMS as f64;

    let coords = insert(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N_ELEMS)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N_ELEMS {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    let model = Model::fick(&fes)?.union(&Model::heat_conduction(&fes)?)?;
    assert_eq!(model.len(), 2);

    // One material field carrying both zones; each physics picks its own.
    let materials = pyrucast::ops::element_field::material_field(&model, &[("D", D), ("k", K)])?;
    let full = pyrucast::ops::matrix::stiffness(&model, &materials)?;

    // The nature selectors split the model — and the assembled matrix — apart.
    let only_diffusion = model.filter(Physics::Diffusion)?;
    let only_thermal = model.filter(Physics::Thermal)?;
    assert_eq!(only_diffusion.len(), 1);
    assert_eq!(only_thermal.len(), 1);
    assert!(model.filter(Physics::Mechanical)?.is_empty());

    // The diffusion block alone is the diffusion part of the coupled matrix:
    // `D` scales the same Laplacian, so its (0,0) entry is `D/h` against `K/h`.
    let d_only = pyrucast::ops::matrix::stiffness(&only_diffusion, &materials)?;
    let k_only = pyrucast::ops::matrix::stiffness(&only_thermal, &materials)?;
    let d00 = d_only.dense()?[0];
    let k00 = k_only.dense()?[0];
    assert!((d00 / k00 - D / K).abs() < 1e-12, "{d00} / {k00}");
    // The coupled matrix carries both, so it is strictly larger than either.
    assert!(full.dense()?.len() > d_only.dense()?.len());
    Ok(())
}
