//! Worked Timoshenko-beam example, exercised end-to-end through the public API.
//!
//! A **slender** cantilever of length `L`, clamped at the left end
//! (`w = θ = 0`), with a transverse tip load `P`. The analytical Timoshenko
//! tip deflection is `w = P·L³/(3·E·I) + P·L/(G·A_s)` (bending + shear). With
//! reduced integration of the shear term the linear element **does not lock**:
//! refining the mesh converges to that value — a locking element would instead
//! return a deflection orders of magnitude too small.
//!
//! Single source for the « Timoshenko » example of the mechanics book chapter;
//! runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{assemble, build, mesher};
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn timoshenko_cantilever_converges_without_locking() -> Result<()> {
    const E: f64 = 1.0;
    const I: f64 = 1.0; // E·I = 1
    const G: f64 = 30.0;
    const A_S: f64 = 1.0; // G·A_s = 30 (slender ⇒ shear locking would be severe)
    const L: f64 = 1.0;
    const P: f64 = 1.0; // transverse tip load
    const N: usize = 40; // beam elements

    // ── Maillage : N éléments SEG2 alignés sur [0, L] (config 1-D) ─────────
    let coords = insert(Coords::new(1)?);
    let h = L / N as f64;
    let nodes: Vec<Node> = (0..=N)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : poutre + encastrement à gauche (w = θ = 0) ────────────────
    let clamp = |node: &Node, var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesher::barycenter(&imposed)?;
        Model::dirichlet(var.into(), dual.into(), &imposed, &multiplier, None, None)
    };
    let mut model = Model::timoshenko(&fes)?;
    model = model.union(&clamp(&nodes[0], "w", "f_w")?)?;
    model = model.union(&clamp(&nodes[0], "theta", "m_theta")?)?;

    // ── Matériau E, I, G, A_s ──────────────────────────────────────────────
    let materials = build::material_field(&model, &[("E", E), ("I", I), ("G", G), ("A_s", A_S)])?;

    // ── Chargement : force transverse P au bout libre (composante f_w) ─────
    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[nodes[N].id()])?;
    let load_sm = insert(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_w".into()])?;
    rhs.set_value(nodes[N].id(), "f_w", P)?;
    let rhs = NodeField::from_sub(rhs);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let solution = solve(&assemble::stiffness(&model, &materials)?, &rhs)?;

    // ── Comparaison : w_tip = P·L³/(3·E·I) + P·L/(G·A_s) ───────────────────
    let w_tip = solution.value(nodes[N].id(), "w")?;
    let analytical = P * L.powi(3) / (3.0 * E * I) + P * L / (G * A_S);
    assert!(
        (w_tip - analytical).abs() < 1e-2 * analytical,
        "w_tip = {w_tip}, analytique {analytical}"
    );
    Ok(())
}
// ANCHOR_END: example

// ANCHOR: comp
/// Section forces (COMP): the same cantilever, post-processed into bending
/// moment `M = E·I·θ'` and shear `V = G·A_s·(w'−θ)`. The shear is constant
/// (≈ −P) and the moment is linear (`|M(0)| ≈ P·L`, `|M(L)| ≈ 0`).
#[test]
fn timoshenko_section_forces_cantilever() -> Result<()> {
    use pyrucast::aggregate::Aggregate;
    use pyrucast::store::read;

    const E: f64 = 1.0;
    const I: f64 = 1.0;
    const G: f64 = 30.0;
    const A_S: f64 = 1.0;
    const L: f64 = 1.0;
    const P: f64 = 1.0;
    const N: usize = 40;
    let h = L / N as f64;

    let coords = insert(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * h]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    let clamp = |node: &Node, var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(node))?);
        let multiplier = mesher::barycenter(&imposed)?;
        Model::dirichlet(var.into(), dual.into(), &imposed, &multiplier, None, None)
    };
    let mut model = Model::timoshenko(&fes)?;
    model = model.union(&clamp(&nodes[0], "w", "f_w")?)?;
    model = model.union(&clamp(&nodes[0], "theta", "m_theta")?)?;
    let materials = build::material_field(&model, &[("E", E), ("I", I), ("G", G), ("A_s", A_S)])?;

    let mut load_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    load_sm.add_cell(&[nodes[N].id()])?;
    let load_sm = insert(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["f_w".into()])?;
    rhs.set_value(nodes[N].id(), "f_w", P)?;
    let rhs = NodeField::from_sub(rhs);
    let solution = solve(&assemble::stiffness(&model, &materials)?, &rhs)?;

    // (κ, γ) puis efforts de section M = EI·κ, V = GA_s·γ.
    let deformation = pyrucast::ops::field::beam_deformation(&solution, &fes)?;
    let forces = pyrucast::ops::behavior::integrate(&model, &deformation, &materials)?;
    let f = read(&forces.get(0)?)?;

    for cell in 0..f.cell_count() {
        assert!(
            (f.value(cell, 0, "V")?.abs() - P).abs() < 2e-2 * P,
            "V non constant ≈ P"
        );
    }
    assert!(
        (f.value(0, 0, "M")?.abs() - P * L).abs() < 5e-2 * P * L,
        "|M(0)| ≈ P·L"
    );
    assert!(
        f.value(f.cell_count() - 1, 0, "M")?.abs() < 5e-2 * P * L,
        "|M(L)| ≈ 0"
    );
    Ok(())
}
// ANCHOR_END: comp
