//! Consistent mass (`MASS`) and heat-capacity (`CAPA`) matrices, exercised
//! end-to-end through the public API.
//!
//! The consistent element mass of a **unit QUA4** is the classic
//! `(area/36)·[[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]` (with `∫N_iN_j`), which
//! gives sharp per-entry oracles; the whole-matrix sum equals `n·ρ·V` (mechanics,
//! one block per component) and `ρ·cp·V` (thermal, single DOF).
//!
//! Matrix rows are labelled by the **dual** variable (`f_x`/`f_y` for mechanics,
//! `q` for thermal), columns by the **primal** one (`u_x`/`u_y`, `T`).

use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::store::Handle;
use pyrucast::Result;

/// A single unit-square QUA4 `[0,1]²` and its four corner nodes (CCW).
fn unit_quad() -> Result<(FiniteElementSpace, [Node; 4])> {
    let coords = Handle::new(Coords::new(2)?);
    let n0 = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let n1 = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let n2 = Node::create_in(coords.clone(), &[1.0, 1.0])?;
    let n3 = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    mesh.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((fes, [n0, n1, n2, n3]))
}

#[test]
fn consistent_mass_of_unit_quad_matches_closed_form() -> Result<()> {
    const RHO: f64 = 2.0;
    let (fes, n) = unit_quad()?;
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", 1.0), ("nu", 0.3), ("rho", RHO)],
    )?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    let tol = 1e-12;

    // Per-component block = ρ·(1/36)·[[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]].
    let diag = RHO * 4.0 / 36.0; // node with itself
    let adj = RHO * 2.0 / 36.0; // edge-adjacent nodes
    let opp = RHO * 1.0 / 36.0; // diagonally-opposite node
    assert!((m.get(n[0].id(), "f_x", n[0].id(), "u_x")? - diag).abs() < tol);
    assert!((m.get(n[0].id(), "f_x", n[1].id(), "u_x")? - adj).abs() < tol);
    assert!((m.get(n[0].id(), "f_x", n[2].id(), "u_x")? - opp).abs() < tol);
    // Same for the u_y ↔ f_y block.
    assert!((m.get(n[0].id(), "f_y", n[0].id(), "u_y")? - diag).abs() < tol);
    // The mass is block-diagonal in the components: no u_y ↔ f_x coupling.
    assert!(m.get(n[0].id(), "f_x", n[0].id(), "u_y")?.abs() < tol);

    // Whole-matrix sum = space_dim · ρ · area (ΣN_i = 1 ⇒ Σ_ij ∫N_iN_j = area).
    let total: f64 = m.to_dmatrix()?.sum();
    assert!(
        (total - 2.0 * RHO * 1.0).abs() < 1e-10,
        "total mass = {total}"
    );

    // Consistent (not lumped): off-diagonal terms are present.
    assert!(m.get(n[0].id(), "f_x", n[1].id(), "u_x")?.abs() > 1e-6);
    Ok(())
}

#[test]
fn heat_capacity_of_unit_quad_matches_closed_form() -> Result<()> {
    const RHO: f64 = 3.0;
    const CP: f64 = 5.0;
    let (fes, n) = unit_quad()?;
    let model = Model::heat_conduction(&fes)?;
    // `k` is required by the physics; `rho`/`cp` are optional and kept because
    // supplied — the capacity matrix needs them, the conductivity does not.
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("k", 1.0), ("rho", RHO), ("cp", CP)],
    )?;

    let c = pyrucast::ops::matrix::mass(&model, &materials)?;
    let tol = 1e-12;
    let rc = RHO * CP;
    assert!((c.get(n[0].id(), "q", n[0].id(), "T")? - rc * 4.0 / 36.0).abs() < tol);
    assert!((c.get(n[0].id(), "q", n[1].id(), "T")? - rc * 2.0 / 36.0).abs() < tol);
    assert!((c.get(n[0].id(), "q", n[2].id(), "T")? - rc * 1.0 / 36.0).abs() < tol);

    let total: f64 = c.to_dmatrix()?.sum();
    assert!((total - rc * 1.0).abs() < 1e-10, "total capacity = {total}");
    Ok(())
}

#[test]
fn lumped_mass_is_diagonal_and_conserves_total() -> Result<()> {
    const RHO: f64 = 2.0;
    let (fes, n) = unit_quad()?;
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", 1.0), ("nu", 0.3), ("rho", RHO)],
    )?;

    let m = pyrucast::ops::matrix::mass(&model, &materials)?;
    let lumped = pyrucast::ops::matrix::lump(&m)?;
    let tol = 1e-12;

    // Each diagonal = its consistent-mass row sum = ρ·(4+2+1+2)/36 = ρ/4.
    assert!((lumped.get(n[0].id(), "f_x", n[0].id(), "u_x")? - RHO / 4.0).abs() < tol);
    // Off-diagonals vanish.
    assert!(lumped.get(n[0].id(), "f_x", n[1].id(), "u_x")?.abs() < tol);

    // Total mass is conserved by row-sum lumping.
    let (mc, ml): (f64, f64) = (m.to_dmatrix()?.sum(), lumped.to_dmatrix()?.sum());
    assert!((mc - ml).abs() < 1e-10, "consistent {mc} vs lumped {ml}");
    assert!((ml - 2.0 * RHO).abs() < 1e-10);
    Ok(())
}

#[test]
fn mass_requires_density() -> Result<()> {
    let (fes, _) = unit_quad()?;
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    // No `rho` supplied ⇒ the mass kernel must error clearly.
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", 1.0), ("nu", 0.3)])?;
    assert!(pyrucast::ops::matrix::mass(&model, &materials).is_err());
    Ok(())
}
