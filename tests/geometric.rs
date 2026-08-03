//! Geometric (initial-stress) stiffness `K_g = ∫ Gᵀ σ̂ G` (Cast3M `KSIG`),
//! exercised end-to-end through the public API.
//!
//! Under a **uniform uniaxial** stress `σ_xx = σ` on the unit QUA4, the block is
//! `K_g[(i,a),(j,a)] = σ · ∫ ∂N_i/∂x ∂N_j/∂x` (the same scalar on every
//! displacement component, `δ_ab`). For the unit Q1 element
//! `∫ ∂N_0/∂x ∂N_0/∂x = 1/3` and `∫ ∂N_0/∂x ∂N_1/∂x = −1/3`, and each row of the
//! scalar sub-matrix sums to 0 (a rigid translation stores no geometric energy).

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::{assemble, build};
use pyrucast::store::insert;
use pyrucast::Result;

fn unit_quad() -> Result<(FiniteElementSpace, [Node; 4])> {
    let coords = insert(Coords::new(2)?);
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
fn geometric_stiffness_uniaxial_stress_unit_quad() -> Result<()> {
    const SIG: f64 = 3.0;
    let (fes, n) = unit_quad()?;
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    let materials = build::material_field(&model, &[("E", 1.0), ("nu", 0.3)])?;

    // Uniform uniaxial stress σ_xx = SIG on the Gauss points.
    let stress_sub = SubElementField::from_uniform_per_component(
        fes.get(0)?,
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
        &[SIG, 0.0, 0.0],
    )?;
    let mut stress = ElementField::empty();
    stress.add_sub(insert(stress_sub))?;

    let kg = assemble::geometric(&model, &materials, &stress)?;
    let tol = 1e-12;

    // σ · ∫(∂N_0/∂x)² = σ/3, on both the u_x and (δ_ab) the u_y diagonal block.
    assert!((kg.get(n[0].id(), "f_x", n[0].id(), "u_x")? - SIG / 3.0).abs() < tol);
    assert!((kg.get(n[0].id(), "f_y", n[0].id(), "u_y")? - SIG / 3.0).abs() < tol);
    // σ · ∫ ∂N_0/∂x ∂N_1/∂x = −σ/3.
    assert!((kg.get(n[0].id(), "f_x", n[1].id(), "u_x")? + SIG / 3.0).abs() < tol);
    // No cross-component coupling.
    assert!(kg.get(n[0].id(), "f_x", n[0].id(), "u_y")?.abs() < tol);

    // Each x-x row sums to 0 (rigid translation carries no geometric stiffness).
    let row_sum: f64 = (0..4)
        .map(|j| kg.get(n[0].id(), "f_x", n[j].id(), "u_x").unwrap())
        .sum();
    assert!(row_sum.abs() < 1e-10, "row sum = {row_sum}");
    Ok(())
}

#[test]
fn geometric_stiffness_is_symmetric() -> Result<()> {
    const SIG: f64 = 2.0;
    let (fes, n) = unit_quad()?;
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    let materials = build::material_field(&model, &[("E", 1.0), ("nu", 0.3)])?;
    let stress_sub = SubElementField::from_uniform_per_component(
        fes.get(0)?,
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
        &[SIG, 0.5 * SIG, 0.25 * SIG],
    )?;
    let mut stress = ElementField::empty();
    stress.add_sub(insert(stress_sub))?;

    let kg = assemble::geometric(&model, &materials, &stress)?;
    let tol = 1e-12;
    for i in 0..4 {
        for j in 0..4 {
            let a = kg.get(n[i].id(), "f_x", n[j].id(), "u_x")?;
            let b = kg.get(n[j].id(), "f_x", n[i].id(), "u_x")?;
            assert!((a - b).abs() < tol, "asymmetry at ({i},{j})");
        }
    }
    Ok(())
}
