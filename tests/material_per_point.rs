//! A material that varies **inside** a cell — moduli depending on the
//! temperature at each Gauss point, a heterogeneous `E(x)` — enters the
//! stiffness and the mass point by point.
//!
//! The oracle is the energy, computed without either matrix: `uᵀKu` must equal
//! `∫ ε : D(x) : ε`, from `strain`, `elastic_stress` and `integral_element`,
//! and `uᵀMu` must equal `∫ ρ(x) |u|²`, from `interp_to_gauss`. A kernel that
//! read the material at one point only would integrate the wrong modulus.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::{elastic_stress, interp_to_gauss, material_field, strain};
use pyrucast::ops::measure::integral_element;
use pyrucast::ops::{matrix, model};
use pyrucast::Result;

/// One distorted QUA4 — a cell where nothing is uniform — its mesh, FE space
/// and corner nodes.
fn quad() -> Result<(Mesh, FiniteElementSpace, Vec<Node>)> {
    let coords = Handle::new(Coords::new(2)?);
    let nodes: Vec<Node> = [[0.0, 0.0], [2.0, 0.2], [1.8, 1.5], [0.1, 1.0]]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    mesh.add_cell(&[nodes[0].id(), nodes[1].id(), nodes[2].id(), nodes[3].id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((mesh, fes, nodes))
}

/// Plane-stress elasticity on `fes`, with `E` and `rho` different at every
/// Gauss point of the cell.
fn varying(fes: &FiniteElementSpace) -> Result<(Model, ElementField)> {
    let solid = model::elasticity(fes, Kinematics::PlaneStress)?;
    let materials = material_field(&solid, &[("E", 1.0), ("nu", 0.25), ("rho", 1.0)])?;
    {
        let mut zone = materials.get(0)?.write();
        for g in 0..zone.gauss_count() {
            zone.set_value(0, g, "E", 100.0 * (1.0 + g as f64))?;
            zone.set_value(0, g, "rho", 2.0 + 3.0 * g as f64)?;
        }
    }
    Ok((solid, materials))
}

/// A displacement with every component moving differently at every node.
fn displacement(mesh: &Mesh, nodes: &[Node]) -> Result<NodeField> {
    let u = NodeField::new(mesh, vec!["u_x".into(), "u_y".into()])?;
    {
        let mut zone = u.get(0)?.write();
        for (k, n) in nodes.iter().enumerate() {
            zone.set_value(n.id(), "u_x", 0.01 * (1.0 + k as f64))?;
            zone.set_value(n.id(), "u_y", -0.02 * (k as f64 * k as f64) + 0.005)?;
        }
    }
    Ok(u)
}

/// `Σ_nodes f·u` over the dual/primal pairs `(f_x, u_x)`, `(f_y, u_y)`.
fn work(f: &NodeField, u: &NodeField, nodes: &[Node]) -> Result<f64> {
    let mut total = 0.0;
    for n in nodes {
        total += f.value(n.id(), "f_x")? * u.value(n.id(), "u_x")?;
        total += f.value(n.id(), "f_y")? * u.value(n.id(), "u_y")?;
    }
    Ok(total)
}

/// A one-component field `name` on `fes`, valued by `at(gauss point)`.
fn per_point(
    fes: &FiniteElementSpace,
    name: &str,
    at: impl Fn(usize) -> Result<f64>,
) -> Result<ElementField> {
    let field = ElementField::new(fes, vec![name.into()])?;
    {
        let mut zone = field.get(0)?.write();
        for g in 0..zone.gauss_count() {
            zone.set_value(0, g, name, at(g)?)?;
        }
    }
    Ok(field)
}

#[test]
fn the_stiffness_integrates_the_modulus_of_every_gauss_point() -> Result<()> {
    let (mesh, fes, nodes) = quad()?;
    let (solid, materials) = varying(&fes)?;
    let u = displacement(&mesh, &nodes)?;

    let k = matrix::stiffness(&solid, &materials)?;
    let assembled = work(&(&k * &u)?, &u, &nodes)?;

    // ε : σ in tensor convention: the shear counts twice.
    let eps = strain(&solid, &u, &materials)?;
    let sig = elastic_stress(&solid, &eps, &materials)?;
    let (e, s) = (eps.get(0)?.read().clone(), sig.get(0)?.read().clone());
    let density = per_point(&fes, "w", |g| {
        Ok(e.value(0, g, "eps_xx")? * s.value(0, g, "sigma_xx")?
            + e.value(0, g, "eps_yy")? * s.value(0, g, "sigma_yy")?
            + 2.0 * e.value(0, g, "eps_xy")? * s.value(0, g, "sigma_xy")?)
    })?;
    let energy = integral_element(&density, "w")?;

    assert!(
        (assembled - energy).abs() <= 1e-12 * energy.abs(),
        "uᵀKu = {assembled}, ∫ ε:D(x):ε = {energy}"
    );
    Ok(())
}

#[test]
fn the_mass_integrates_the_density_of_every_gauss_point() -> Result<()> {
    let (mesh, fes, nodes) = quad()?;
    let (solid, materials) = varying(&fes)?;
    let u = displacement(&mesh, &nodes)?;

    let m = matrix::mass(&solid, &materials)?;
    let assembled = work(&(&m * &u)?, &u, &nodes)?;

    let at = interp_to_gauss(&u, &fes)?;
    let (v, mat) = (at.get(0)?.read().clone(), materials.get(0)?.read().clone());
    let density = per_point(&fes, "w", |g| {
        let (ux, uy) = (v.value(0, g, "u_x")?, v.value(0, g, "u_y")?);
        Ok(mat.value(0, g, "rho")? * (ux * ux + uy * uy))
    })?;
    let energy = integral_element(&density, "w")?;

    assert!(
        (assembled - energy).abs() <= 1e-12 * energy.abs(),
        "uᵀMu = {assembled}, ∫ ρ(x) |u|² = {energy}"
    );
    Ok(())
}
