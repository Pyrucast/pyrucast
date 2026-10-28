//! A material that varies **inside** a cell — moduli depending on the
//! temperature at each Gauss point, a heterogeneous `E(x)`, material axes that
//! turn — enters every matrix point by point: the stiffness and the mass of
//! the continuum, the conductivity and the capacity of heat conduction, the
//! diffusivity and the storage of Fick's law.
//!
//! The oracle is the energy, computed without the matrix: `uᵀKu` must equal
//! `∫ ε : D(x) : ε`, from `strain`, `elastic_stress` and `integral_element`;
//! `uᵀMu` must equal `∫ ρ(x) |u|²`, from `interp_to_gauss`; `TᵀKT` must equal
//! `∫ ∇T · K(x) ∇T`, from `gradient` and a tensor written out here. A kernel
//! that read the material at one point only would integrate the wrong one.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::StrainMeasure;
use pyrucast::ops::element_field::{
    elastic_stress, gradient, interp_to_gauss, material_field, strain,
};
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
    let solid = model::elasticity(
        fes,
        Kinematics::PlaneStress,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )?;
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

/// A scalar field `name` with a different value at every node.
fn scalar(mesh: &Mesh, nodes: &[Node], name: &str) -> Result<NodeField> {
    let f = NodeField::new(mesh, vec![name.into()])?;
    {
        let mut zone = f.get(0)?.write();
        for (k, n) in nodes.iter().enumerate() {
            zone.set_value(n.id(), name, 1.0 + 0.7 * k as f64 - 0.2 * (k * k) as f64)?;
        }
    }
    Ok(f)
}

/// An orthotropic transport law (`k_1`, `k_2`, axes `V1`) whose constants and
/// axes all change from one Gauss point to the next, checked against
/// `∫ ∇T · K(x) ∇T` with `K = R diag(k₁, k₂) Rᵀ` written out here; then its
/// storage term `prefix_storage` against `∫ c(x) T²`.
fn check_transport(
    law: &Model,
    primal: &str,
    dual: &str,
    names: (&str, &str, &str),
    constants: &[(&str, f64)],
) -> Result<()> {
    let (mesh, fes, nodes) = quad()?;
    let (k1_name, k2_name, storage) = names;
    let at = |g: usize| (2.0 + g as f64, 0.5 + 0.25 * g as f64, 0.4 * g as f64);
    let mut defaults: Vec<(String, f64)> = Vec::new();
    for zone in law {
        for name in zone.read().material_components().unwrap_or_default() {
            defaults.push((name, 1.0));
        }
    }
    defaults.push((storage.to_string(), 1.0));
    defaults.extend(constants.iter().map(|(n, v)| (n.to_string(), *v)));
    let pairs: Vec<(&str, f64)> = defaults.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let materials = material_field(law, &pairs)?;
    {
        let mut zone = materials.get(0)?.write();
        for g in 0..zone.gauss_count() {
            let (k1, k2, angle) = at(g);
            zone.set_value(0, g, k1_name, k1)?;
            zone.set_value(0, g, k2_name, k2)?;
            zone.set_value(0, g, "V1X", angle.cos())?;
            zone.set_value(0, g, "V1Y", angle.sin())?;
            zone.set_value(0, g, storage, 3.0 + g as f64)?;
        }
    }
    let t = scalar(&mesh, &nodes, primal)?;
    let work = |m: &pyrucast::containers::matrix::Matrix| -> Result<f64> {
        let mt = (m * &t)?;
        let mut total = 0.0;
        for n in &nodes {
            total += mt.value(n.id(), dual)? * t.value(n.id(), primal)?;
        }
        Ok(total)
    };

    let grad = gradient(&t, &fes)?.get(0)?.read().clone();
    let (gx, gy) = (format!("grad_{primal}_x"), format!("grad_{primal}_y"));
    let density = per_point(&fes, "w", |g| {
        let (k1, k2, angle) = at(g);
        let (c, s) = (angle.cos(), angle.sin());
        let (dx, dy) = (grad.value(0, g, &gx)?, grad.value(0, g, &gy)?);
        // Components along the material axes, weighted by their constants.
        let (along, across) = (c * dx + s * dy, -s * dx + c * dy);
        Ok(k1 * along * along + k2 * across * across)
    })?;
    let expected = integral_element(&density, "w")?;
    let assembled = work(&matrix::stiffness(law, &materials)?)?;
    assert!(
        (assembled - expected).abs() <= 1e-12 * expected.abs(),
        "Tᵀ K T = {assembled}, ∫ ∇T·K(x)∇T = {expected}"
    );

    let values = interp_to_gauss(&t, &fes)?.get(0)?.read().clone();
    let stored = per_point(&fes, "w", |g| {
        let v = values.value(0, g, primal)?;
        Ok((3.0 + g as f64) * v * v)
    })?;
    let expected = integral_element(&stored, "w")?;
    let assembled = work(&matrix::mass(law, &materials)?)?;
    assert!(
        (assembled - expected).abs() <= 1e-12 * expected.abs(),
        "Tᵀ C T = {assembled}, ∫ c(x) T² = {expected}"
    );
    Ok(())
}

#[test]
fn heat_conduction_integrates_its_conductivity_and_axes_at_every_gauss_point() -> Result<()> {
    let (_, fes, _) = quad()?;
    let law = model::heat_conduction(&fes, MaterialSymmetry::Orthotropic)?;
    // The capacity ρ·cp, with ρ = 1: cp carries the variation.
    check_transport(&law, "T", "q", ("k_1", "k_2", "cp"), &[("rho", 1.0)])
}

#[test]
fn fick_integrates_its_diffusivity_axes_and_storage_at_every_gauss_point() -> Result<()> {
    let (_, fes, _) = quad()?;
    let law = model::fick(&fes, "H2", MaterialSymmetry::Orthotropic)?;
    check_transport(&law, "c_H2", "j_H2", ("D_1_H2", "D_2_H2", "poro"), &[])
}
