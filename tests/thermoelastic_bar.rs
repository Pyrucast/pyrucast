//! Worked uncoupled-thermomechanics example, exercised end-to-end through the
//! public API.
//!
//! A plane-stress bar `[0,L]×[0,H]` (QUA4 grid) is heated by a uniform `ΔT`.
//! Both x-ends are clamped (`u_x = 0`) and the bottom edge carries a `u_y = 0`
//! roller (rigid-body removal in y). With the free dilation blocked in x, the
//! exact response is uniform `σ_xx = −E·α·ΔT`, `σ_yy = 0`.
//!
//! The load is composed from the point bricks, no all-in-one operator:
//! `interp_to_gauss` (nodal T → Gauss), `thermal_strain` (EPTH), `integrate`
//! (σ = D:ε), `internal_forces` (BSIG) ⇒ `f_th`; then `σ = D:(ε(u) − ε_th)`.
//! `alpha` rides the material field as an optional component.
//!
//! Single source for the « thermomécanique » example of the mechanics book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::field::Field;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::element_field::{deformation, interp_to_gauss, thermal_strain};
use pyrucast::ops::mesh;
use pyrucast::ops::node_field::internal_forces;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::{insert, read};
use pyrucast::Result;

#[test]
fn thermoelastic_constrained_bar_stress() -> Result<()> {
    const E: f64 = 210_000.0;
    const NU: f64 = 0.3;
    const ALPHA: f64 = 1e-5;
    const T_REF: f64 = 20.0;
    const DT: f64 = 100.0;
    const NX: usize = 4;
    const NY: usize = 2;
    const L: f64 = 4.0;
    const H: f64 = 1.0;
    let (hx, hy) = (L / NX as f64, H / NY as f64);

    // ── Maillage QUA4 sur [0,L]×[0,H] ──────────────────────────────────────
    let coords = insert(Coords::new(2)?);
    let idx = |i: usize, j: usize| j * (NX + 1) + i;
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=NY {
        for i in 0..=NX {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * hx, j as f64 * hy],
            )?);
        }
    }
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..NY {
        for i in 0..NX {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // ── Modèle : élasticité + deux bords en x encastrés + appui u_y en bas ──
    let clamp = |nodes: &[Node], var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
        let multiplier = mesh::barycenter(&imposed)?;
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
    let left: Vec<Node> = (0..=NY).map(|j| grid[idx(0, j)].clone()).collect();
    let right: Vec<Node> = (0..=NY).map(|j| grid[idx(NX, j)].clone()).collect();
    let bottom: Vec<Node> = (0..=NX).map(|i| grid[idx(i, 0)].clone()).collect();
    let mut model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    model = model.union(&clamp(&left, "u_x", "f_x")?)?;
    model = model.union(&clamp(&right, "u_x", "f_x")?)?;
    model = model.union(&clamp(&bottom, "u_y", "f_y")?)?;

    // `alpha` supplied through the material field — an optional elastic component.
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", NU), ("alpha", ALPHA)],
    )?;

    // ── Température imposée T = T_ref + ΔT partout, portée aux points de Gauss
    let support = insert(SubMesh::poi1_from_nodes(&grid)?);
    let mut t_nodal = SubNodeField::from_poi1(&support, vec!["T".into()])?;
    for n in &grid {
        t_nodal.set_value(n.id(), "T", T_REF + DT)?;
    }
    let t_elem = interp_to_gauss(&NodeField::from_sub(t_nodal), &fes)?;

    // ── Charge thermique f_th = ∫ Bᵀ D ε_th (BSIG de σ_th = D:ε_th) ─────────
    let eps_th = thermal_strain(&t_elem, &materials, &fes, T_REF)?;
    let sig_th =
        pyrucast::ops::element_field::behavior::integrate(&model, &eps_th, None, &materials, None)?;
    let f_th = internal_forces(&model, &sig_th)?;

    // ── Assemblage + résolution ────────────────────────────────────────────
    let solution = solve(
        &pyrucast::ops::matrix::stiffness(&model, &materials)?,
        &f_th,
    )?;

    // ── Déplacement propre (u_x, u_y) puis σ = D:(ε(u) − ε_th) ─────────────
    let disp_support = insert(SubMesh::poi1_from_nodes(&grid)?);
    let mut disp = SubNodeField::from_poi1(&disp_support, vec!["u_x".into(), "u_y".into()])?;
    for n in &grid {
        disp.set_value(n.id(), "u_x", solution.value(n.id(), "u_x")?)?;
        disp.set_value(n.id(), "u_y", solution.value(n.id(), "u_y")?)?;
    }
    let eps = deformation(&NodeField::from_sub(disp), &fes)?;
    let eps_mech = eps.merge_field(&eps_th, |a, b| a - b)?;
    let sigma = pyrucast::ops::element_field::behavior::integrate(
        &model, &eps_mech, None, &materials, None,
    )?;

    // ── Vérification : σ_xx = −E·α·ΔT, σ_yy = 0 ────────────────────────────
    let expected = -E * ALPHA * DT;
    let tol = 1e-6 * expected.abs();
    let sub = read(&sigma.get(0)?)?;
    for cell in 0..sub.cell_count() {
        for g in 0..sub.gauss_count() {
            assert!((sub.value(cell, g, "sigma_xx")? - expected).abs() < tol);
            assert!(sub.value(cell, g, "sigma_yy")?.abs() < tol);
        }
    }
    Ok(())
}
// ANCHOR_END: example
