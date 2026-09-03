//! Axisymmetric (body-of-revolution) computations, exercised end-to-end through
//! the public API.
//!
//! The geometry is the meridian plane `(r, z)` of
//! [`Coords::axisymmetric`](pyrucast::coords::Coords::axisymmetric):
//! `x = r`, `y = z`. Two things then differ from a plane computation, and each
//! gets its own checks here:
//!
//! - the **integration measure** `dΩ = 2πr |J| dξ`, carried by the geometry, so
//!   volumes, masses, distributed loads and the thermal conductivity follow
//!   without any mechanics involved;
//! - the **hoop strain** `ε_θθ = u_r / r`, carried by
//!   [`Kinematics::Axisymmetric`], which the meridian gradient cannot
//!   express.
//!
//! Reference solution: the Lamé thick-walled cylinder under internal pressure.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, Interpolation};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::model;

use pyrucast::handle::Handle;
use pyrucast::ops::node_field::FluxDensity;
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{element_field, measure, mesh, node_field};
use pyrucast::Result;

use std::f64::consts::PI;

/// A QUA4 grid over the meridian rectangle `r ∈ [r0, r1]`, `z ∈ [0, h]`,
/// `nr × nz` cells. Returns the nodes (row-major, `idx(i, j) = j*(nr+1) + i`),
/// the mesh and its Q1 space.
fn annulus(
    r0: f64,
    r1: f64,
    h: f64,
    nr: usize,
    nz: usize,
) -> Result<(Vec<Node>, Mesh, FiniteElementSpace)> {
    let coords = Handle::new(Coords::axisymmetric()?);
    let idx = |i: usize, j: usize| j * (nr + 1) + i;
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=nz {
        for i in 0..=nr {
            let r = r0 + (r1 - r0) * i as f64 / nr as f64;
            let z = h * j as f64 / nz as f64;
            grid.push(Node::create_in(coords.clone(), &[r, z])?);
        }
    }
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    for j in 0..nz {
        for i in 0..nr {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((grid, mesh, fes))
}

/// Homogeneous Dirichlet `var = 0` on every node of `nodes`.
fn clamp(nodes: &[Node], var: &str, dual: &str) -> Result<Model> {
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
    let multiplier = mesh::barycenter(&imposed)?;
    model::dirichlet(
        var.into(),
        dual.into(),
        &imposed,
        &multiplier,
        None,
        None,
        Default::default(),
    )
}

// ─── The measure: 2πr, with no mechanics involved ────────────────────────────

/// `∫ 1 dΩ` over the meridian rectangle is the **revolved** volume
/// `π(r1² − r0²)·h`, not the plane area — the geometry alone decides this.
#[test]
fn integral_measures_the_revolved_volume() -> Result<()> {
    let (grid, _mesh, fes) = annulus(1.0, 3.0, 2.0, 4, 2)?;

    let support = Handle::new(SubMesh::poi1_from_nodes(&grid)?);
    let mut ones = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &support,
        vec!["one".to_string()],
    )?;
    for n in &grid {
        ones.set_value(n.id(), "one", 1.0)?;
    }
    let ones = NodeField::from_sub(ones);

    let volume = measure::integral(&ones, &fes, "one")?;
    let expected = PI * (3.0_f64.powi(2) - 1.0) * 2.0;
    assert!(
        (volume - expected).abs() < 1e-9,
        "revolved volume {volume} ≠ {expected}"
    );
    Ok(())
}

/// The consistent mass matrix of a ring weighs the **whole ring**: summing every
/// entry gives `ρ · 2π r̄ · A` (Guldin), here `ρ · π(r1² − r0²) · h`.
#[test]
fn mass_matrix_weighs_the_whole_ring() -> Result<()> {
    const RHO: f64 = 7800.0;
    let (_grid, _mesh, fes) = annulus(1.0, 3.0, 2.0, 4, 2)?;
    let model = model::elasticity(&fes, Kinematics::Axisymmetric)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", 1.0), ("nu", 0.3), ("rho", RHO)],
    )?;

    let mass = pyrucast::ops::matrix::mass(&model, &materials)?;
    // 1ᵀ M 1 on the radial component: ∫ ρ (Σ_i N_i)(Σ_j N_j) dΩ = ρ·volume, the
    // shape functions being a partition of unity. So a unit radial "displacement"
    // gives nodal forces summing to the ring's mass.
    let support = Handle::new(SubMesh::poi1_from_nodes(&_grid)?);
    let mut ones = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &support,
        vec!["u_x".to_string(), "u_y".to_string()],
    )?;
    for n in &_grid {
        ones.set_value(n.id(), "u_x", 1.0)?;
        ones.set_value(n.id(), "u_y", 0.0)?;
    }
    let ones = NodeField::from_sub(ones);
    let m_ones = mass.mul_field(&ones)?;
    let total: f64 = _grid
        .iter()
        .map(|n| m_ones.value(n.id(), "f_x").unwrap())
        .sum();
    let expected = RHO * PI * (3.0_f64.powi(2) - 1.0) * 2.0;
    assert!(
        (total - expected).abs() / expected < 1e-12,
        "ring mass {total} ≠ {expected}"
    );
    Ok(())
}

// ─── The kinematics: the hoop strain ─────────────────────────────────────────

/// A uniform axial translation is a **rigid body motion**: no strain at all.
/// A uniform *radial* translation is not — it stretches the circumference, so
/// `ε_θθ = u_r / r ≠ 0`. That asymmetry is the whole point of the formulation.
#[test]
fn axial_translation_is_rigid_but_radial_translation_is_not() -> Result<()> {
    let (grid, _mesh, fes) = annulus(1.0, 2.0, 1.0, 2, 1)?;
    let support = Handle::new(SubMesh::poi1_from_nodes(&grid)?);

    let uniform = |ur: f64, uz: f64| -> Result<NodeField> {
        let mut u = pyrucast::containers::node_field::SubNodeField::from_poi1(
            &support,
            vec!["u_x".to_string(), "u_y".to_string()],
        )?;
        for n in &grid {
            u.set_value(n.id(), "u_x", ur)?;
            u.set_value(n.id(), "u_y", uz)?;
        }
        Ok(NodeField::from_sub(u))
    };

    // Axial translation: every component vanishes, hoop included.
    let strain = element_field::deformation(&uniform(0.0, 0.5)?, &fes)?;
    let s = strain.get(0)?.read();
    for g in 0..s.gauss_count() {
        for c in ["eps_xx", "eps_yy", "eps_xy", "eps_zz"] {
            assert!(
                s.value(0, g, c)?.abs() < 1e-14,
                "{c} = {} under axial translation",
                s.value(0, g, c)?
            );
        }
    }
    drop(s);

    // Radial translation: meridian strains vanish, the hoop does not.
    let strain = element_field::deformation(&uniform(0.5, 0.0)?, &fes)?;
    let s = strain.get(0)?.read();
    for g in 0..s.gauss_count() {
        for c in ["eps_xx", "eps_yy", "eps_xy"] {
            assert!(s.value(0, g, c)?.abs() < 1e-14);
        }
        assert!(
            s.value(0, g, "eps_zz")? > 0.1,
            "hoop strain should be ≈ 0.5/r, got {}",
            s.value(0, g, "eps_zz")?
        );
    }
    Ok(())
}

/// Uniform dilation `u_r = c·r`, `u_z = e·z` is a **constant** strain state
/// (`ε_rr = ε_θθ = c`, `ε_zz = e`), so Q1 reproduces it exactly — the patch test
/// of the hoop row of `B`.
#[test]
fn uniform_dilation_is_an_exact_constant_strain_state() -> Result<()> {
    const C: f64 = 1e-3;
    const EZ: f64 = -4e-4;
    let (grid, _mesh, fes) = annulus(1.0, 3.0, 2.0, 3, 2)?;
    let support = Handle::new(SubMesh::poi1_from_nodes(&grid)?);

    let mut u = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &support,
        vec!["u_x".to_string(), "u_y".to_string()],
    )?;
    for n in &grid {
        let (r, z) = {
            let c = n.position()?;
            (c[0], c[1])
        };
        u.set_value(n.id(), "u_x", C * r)?;
        u.set_value(n.id(), "u_y", EZ * z)?;
    }
    let u = NodeField::from_sub(u);

    let strain = element_field::deformation(&u, &fes)?;
    let s = strain.get(0)?.read();
    for cell in 0..s.cell_count() {
        for g in 0..s.gauss_count() {
            assert!((s.value(cell, g, "eps_xx")? - C).abs() < 1e-14);
            assert!((s.value(cell, g, "eps_zz")? - C).abs() < 1e-14, "hoop");
            assert!((s.value(cell, g, "eps_yy")? - EZ).abs() < 1e-14);
            assert!(s.value(cell, g, "eps_xy")?.abs() < 1e-14);
        }
    }
    Ok(())
}

// ─── The reference solution: Lamé ────────────────────────────────────────────

/// Thick-walled cylinder `a ≤ r ≤ b` under internal pressure `p`, in plane
/// strain (`u_z = 0` on both ends). The Lamé solution is
///
/// ```text
/// σ_rr = A − B/r²,  σ_θθ = A + B/r²,  u_r = (1+ν)/E · [(1−2ν)·A·r + B/r]
/// A = p a²/(b²−a²),  B = p a² b²/(b²−a²)
/// ```
///
/// Solves it on an `nr × 1` mesh, linear (`QUA4`) or quadratic (`QUA8`), and
/// returns `(max relative error on u_r, max stress error / p)`.
///
/// The internal pressure is applied as a distributed load on the inner edge:
/// because the geometry is axisymmetric, `flux` integrates `∫ 2πr N p` and
/// yields the correct total ring force with no manual factor. The solid and the
/// loaded edge live in **one** mesh so that `to_quadratic` gives them a *shared*
/// mid-edge node rather than two coincident ones.
fn lame_case(nr: usize, quadratic: bool) -> Result<(f64, f64)> {
    const E: f64 = 210_000.0;
    const NU: f64 = 0.3;
    const P: f64 = 100.0;
    const A: f64 = 1.0;
    const B: f64 = 2.0;
    const H: f64 = 0.5;

    let coords = Handle::new(Coords::axisymmetric()?);
    let idx = |i: usize, j: usize| j * (nr + 1) + i;
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=1 {
        for i in 0..=nr {
            grid.push(Node::create_in(
                coords.clone(),
                &[A + (B - A) * i as f64 / nr as f64, H * j as f64],
            )?);
        }
    }
    let mut mesh = Mesh::empty();
    let mut solid = SubMesh::new(coords.clone(), ElementType::QUA4);
    for i in 0..nr {
        solid.add_cell(&[
            grid[idx(i, 0)].id(),
            grid[idx(i + 1, 0)].id(),
            grid[idx(i + 1, 1)].id(),
            grid[idx(i, 1)].id(),
        ])?;
    }
    let mut edge = SubMesh::new(coords.clone(), ElementType::SEG2);
    edge.add_cell(&[grid[idx(0, 0)].id(), grid[idx(0, 1)].id()])?;
    mesh.add_sub(Handle::new(solid))?;
    mesh.add_sub(Handle::new(edge))?;

    let (mesh, interp) = if quadratic {
        (mesh::to_quadratic(&mesh)?, Interpolation::Lagrange2)
    } else {
        (mesh, Interpolation::Lagrange1)
    };
    let mut solid_mesh = Mesh::empty();
    solid_mesh.add_sub(mesh.get(0)?)?;
    let mut edge_mesh = Mesh::empty();
    edge_mesh.add_sub(mesh.get(1)?)?;
    let fes = FiniteElementSpace::new(&solid_mesh, interp)?;
    let edge_fes = FiniteElementSpace::new(&edge_mesh, interp)?;

    // Plane strain: u_z = 0 on both z faces (every node on them, mid-edge
    // nodes included — hence the coordinate filter rather than an index list).
    let poi1 = mesh::to_poi1(&solid_mesh)?;
    let nodes: Vec<Node> = (0..poi1.get(0)?.read().cell_count())
        .map(|c| poi1.node(0, c, 0))
        .collect::<Result<_>>()?;
    let ends: Vec<Node> = nodes
        .iter()
        .filter(|n| {
            let z = n.position().map(|c| c[1]).unwrap_or(f64::NAN);
            z.abs() < 1e-12 || (z - H).abs() < 1e-12
        })
        .cloned()
        .collect();
    let mut model = model::elasticity(&fes, Kinematics::Axisymmetric)?;
    model = model.union(&clamp(&ends, "u_y", "f_y")?)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU)])?;

    let load = pyrucast::ops::node_field::flux(&edge_fes, FluxDensity::Uniform(P), "f_x")?;
    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &load)?;

    let (a2, b2) = (A * A, B * B);
    let ca = P * a2 / (b2 - a2);
    let cb = P * a2 * b2 / (b2 - a2);

    // ── Displacement against Lamé ──────────────────────────────────────────
    let mut worst_u = 0.0_f64;
    for n in &nodes {
        let r = n.position()?[0];
        let exact = (1.0 + NU) / E * ((1.0 - 2.0 * NU) * ca * r + cb / r);
        worst_u = worst_u.max((solution.value(n.id(), "u_x")? - exact).abs() / exact.abs());
        assert!(solution.value(n.id(), "u_y")?.abs() < 1e-12, "plane strain");
    }

    // ── Stresses against Lamé, at the Gauss points ─────────────────────────
    // `solve` returns the Lagrange multipliers alongside the displacement, so
    // restrict onto a displacement-shaped field before differentiating.
    let sm = Handle::new(SubMesh::poi1_from_nodes(&nodes)?);
    let zero = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &sm,
        vec!["u_x".to_string(), "u_y".to_string()],
    )?;
    let displacement = node_field::restrict_like(&solution, &NodeField::from_sub(zero))?;
    let strain = element_field::deformation(&displacement, &fes)?;
    let stress =
        pyrucast::ops::element_field::behavior::integrate(&model, &strain, None, &materials, None)?;
    // The radius at each Gauss point, obtained the same way any field is: the
    // nodal coordinates interpolated onto the quadrature.
    let gauss_r = element_field::interp_to_gauss(&node_field::positions(&solid_mesh, None)?, &fes)?;
    let (s, rg) = (stress.get(0)?.read(), gauss_r.get(0)?.read());
    let mut worst_s = 0.0_f64;
    for cell in 0..s.cell_count() {
        for g in 0..s.gauss_count() {
            let r = rg.value(cell, g, "X")?;
            worst_s = worst_s.max((s.value(cell, g, "sigma_xx")? - (ca - cb / (r * r))).abs());
            worst_s = worst_s.max((s.value(cell, g, "sigma_zz")? - (ca + cb / (r * r))).abs());
        }
    }
    Ok((worst_u, worst_s / P))
}

/// Linear `QUA4`: displacement converges in O(h²), stress in O(h).
#[test]
fn lame_thick_cylinder_under_internal_pressure() -> Result<()> {
    let (u, s) = lame_case(40, false)?;
    assert!(u < 5e-3, "relative displacement error {u}");
    assert!(s < 0.04, "stress error {s} of p");
    Ok(())
}

/// Quadratic `QUA8`. **Not exact** — and it cannot be: the Lamé displacement
/// `c₁·r + c₂/r` has a *rational* term, which no Lagrange basis of any degree
/// spans. What quadratic elements buy is an order, not exactness.
///
/// The solution depends on `r` alone, so the discrete problem is a 1-D ODE and
/// the nodal values are superconvergent at `O(h^{2p})`: `O(h²)` for `p = 1`,
/// `O(h⁴)` for `p = 2`. Measured relative displacement error:
///
/// | nr | Q1 | ordre | Q2 | ordre |
/// |---:|---:|---:|---:|---:|
/// | 5  | 6.5e-3 | — | 2.6e-5 | — |
/// | 10 | 1.6e-3 | 1.97 | 1.7e-6 | 3.94 |
/// | 20 | 4.1e-4 | 1.99 | 1.1e-7 | 3.99 |
/// | 40 | 1.0e-4 | 2.00 | 6.8e-9 | 4.00 |
///
/// The stress (one derivative down) goes from `O(h)` to `O(h²)`.
#[test]
fn lame_quadratic_elements_gain_an_order_but_are_not_exact() -> Result<()> {
    let (u5, s5) = lame_case(5, true)?;
    let (u10, s10) = lame_case(10, true)?;

    // Far more accurate than Q1 — 1.7e-6 at nr = 10, where Q1 needs nr = 40 to
    // reach 1e-4.
    assert!(u10 < 1e-5, "quadratic displacement error {u10}");
    assert!(s10 < 0.01, "quadratic stress error {s10} of p");

    // …but genuinely converging, not exact: halving h divides the nodal error by
    // ~16 (order 4) and the stress error by ~4 (order 2).
    let order_u = (u5 / u10).log2();
    let order_s = (s5 / s10).log2();
    assert!(
        (3.5..4.5).contains(&order_u),
        "nodal displacement order {order_u}, expected ≈ 4"
    );
    assert!(
        (1.5..2.5).contains(&order_s),
        "stress order {order_s}, expected ≈ 2"
    );
    // The point of the test: a non-zero error at every refinement. `1/r` is not
    // a polynomial, so no Lagrange element reproduces Lamé exactly.
    assert!(u10 > 0.0 && s10 > 0.0);
    Ok(())
}

/// For a linear law the internal forces `∫ Bᵀσ dΩ` must equal `K·u`. On a body
/// of revolution this only holds if the hoop row of `B` and its transpose in the
/// internal-force kernel agree — so it is the check that pins the `N_i/r` term
/// on both sides.
#[test]
fn internal_forces_match_stiffness_times_displacement() -> Result<()> {
    const E: f64 = 210_000.0;
    const NU: f64 = 0.3;
    let (grid, _mesh, fes) = annulus(1.0, 2.5, 1.0, 4, 3)?;
    let support = Handle::new(SubMesh::poi1_from_nodes(&grid)?);

    let model = model::elasticity(&fes, Kinematics::Axisymmetric)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU)])?;

    // An arbitrary, non-rigid displacement field.
    let mut u = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &support,
        vec!["u_x".to_string(), "u_y".to_string()],
    )?;
    for n in &grid {
        let c = n.position()?;
        let (r, z) = (c[0], c[1]);
        u.set_value(n.id(), "u_x", 1e-3 * (r + 0.3 * z * z))?;
        u.set_value(n.id(), "u_y", 1e-3 * (0.2 * r * z - 0.1 * z))?;
    }
    let u = NodeField::from_sub(u);

    let strain = element_field::deformation(&u, &fes)?;
    let stress =
        pyrucast::ops::element_field::behavior::integrate(&model, &strain, None, &materials, None)?;
    let bsig = pyrucast::ops::node_field::internal_forces(&model, &stress)?;

    let stiffness = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let ku = stiffness.mul_field(&u)?;

    for n in &grid {
        for (primal, dual) in [("u_x", "f_x"), ("u_y", "f_y")] {
            let _ = primal;
            let a = bsig.value(n.id(), dual)?;
            let b = ku.value(n.id(), dual)?;
            assert!(
                (a - b).abs() < 1e-7 * (1.0 + b.abs()),
                "∫Bᵀσ [{dual}] = {a} ≠ K·u = {b}"
            );
        }
    }
    Ok(())
}

// ─── Thermal: the measure is enough ──────────────────────────────────────────

/// Steady conduction through a hollow cylinder with imposed temperatures on both
/// faces: `T(r) = T_a + (T_b − T_a)·ln(r/a)/ln(b/a)`. Nothing in the thermal
/// physics knows about revolution — the logarithmic profile comes out purely
/// from the `2πr` in the measure.
#[test]
fn heat_conduction_through_a_hollow_cylinder_is_logarithmic() -> Result<()> {
    const A: f64 = 1.0;
    const B: f64 = 4.0;
    const TA: f64 = 100.0;
    const TB: f64 = 20.0;
    const NR: usize = 40;

    let (grid, _mesh, fes) = annulus(A, B, 1.0, NR, 1)?;
    let idx = |i: usize, j: usize| j * (NR + 1) + i;

    // Imposed temperature on a face: the Dirichlet sub-model plus the value
    // written at its multiplier nodes' `imposed_T` slot.
    let mut model = model::heat_conduction(&fes)?;
    let mut mult_nodes: Vec<(Node, f64)> = Vec::new();
    for (nodes, value) in [
        (
            (0..=1).map(|j| grid[idx(0, j)].clone()).collect::<Vec<_>>(),
            TA,
        ),
        (
            (0..=1)
                .map(|j| grid[idx(NR, j)].clone())
                .collect::<Vec<_>>(),
            TB,
        ),
    ] {
        let m = Mesh::from_submesh(SubMesh::poi1_from_nodes(&nodes)?);
        let multiplier = mesh::barycenter(&m)?;
        for k in 0..nodes.len() {
            mult_nodes.push((multiplier.node(0, k, 0)?, value));
        }
        model = model.union(&model::dirichlet(
            "T".into(),
            "q".into(),
            &m,
            &multiplier,
            None,
            None,
            Default::default(),
        )?)?;
    }
    let materials = pyrucast::ops::element_field::material_field(&model, &[("k", 1.0)])?;

    let mult_sm = Handle::new(SubMesh::poi1_from_nodes(
        &mult_nodes
            .iter()
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>(),
    )?);
    let mut rhs = pyrucast::containers::node_field::SubNodeField::from_poi1(
        &mult_sm,
        vec!["imposed_T".to_string()],
    )?;
    for (n, v) in &mult_nodes {
        rhs.set_value(n.id(), "imposed_T", *v)?;
    }
    let rhs = NodeField::from_sub(rhs);

    let conductivity = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&conductivity, &rhs)?;

    let exact = |r: f64| TA + (TB - TA) * (r / A).ln() / (B / A).ln();
    for i in 0..=NR {
        let n = &grid[idx(i, 0)];
        let r = n.position()?[0];
        let t = solution.value(n.id(), "T")?;
        assert!(
            (t - exact(r)).abs() < 0.05,
            "T({r}) = {t}, analytic = {}",
            exact(r)
        );
    }
    Ok(())
}

// ─── Guard rails ─────────────────────────────────────────────────────────────

/// The model and the geometry must agree, both ways — a plane model on a body of
/// revolution would silently mix a plane law with the 2πr measure.
#[test]
fn model_and_geometry_must_agree() -> Result<()> {
    let (_grid, _mesh, axi) = annulus(1.0, 2.0, 1.0, 1, 1)?;
    let err = model::elasticity(&axi, Kinematics::PlaneStrain).unwrap_err();
    assert!(format!("{err}").contains("axisymmetric geometry"));

    // And the axisymmetric model on a plain Cartesian geometry.
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    let mut plane = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
    plane.add_cell(&[a.id(), b.id(), c.id()])?;
    let plane_fes = FiniteElementSpace::lagrange1(&plane)?;
    let err = model::elasticity(&plane_fes, Kinematics::Axisymmetric).unwrap_err();
    assert!(format!("{err}").contains("requires an axisymmetric geometry"));

    // The same two-way rule holds for the non-linear laws.
    for err in [
        model::plasticity_perfect(&axi, Kinematics::PlaneStrain).unwrap_err(),
        model::mazars(&axi, Kinematics::PlaneStrain).unwrap_err(),
    ] {
        assert!(format!("{err}").contains("axisymmetric geometry"));
    }
    assert!(model::plasticity_perfect(&axi, Kinematics::Axisymmetric).is_ok());
    assert!(model::mazars(&axi, Kinematics::Axisymmetric).is_ok());
    Ok(())
}

/// A continuum physics on a **manifold** (a boundary mesh: `SEG2` in 2-D) has no
/// meaning — `B` would be built from the tangent gradient and `Bᵀ D B` would be
/// rank-deficient in the normal direction. All three refuse it, on a plain
/// Cartesian geometry as much as on a revolved one.
#[test]
fn a_boundary_mesh_is_not_a_solid() -> Result<()> {
    for axisymmetric in [true, false] {
        let coords = Handle::new(if axisymmetric {
            Coords::axisymmetric()?
        } else {
            Coords::new(2)?
        });
        let a = Node::create_in(coords.clone(), &[1.0, 0.0])?;
        let b = Node::create_in(coords.clone(), &[2.0, 0.0])?;
        let mut line = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        line.add_cell(&[a.id(), b.id()])?;
        let fes = FiniteElementSpace::lagrange1(&line)?;
        let model = if axisymmetric {
            Kinematics::Axisymmetric
        } else {
            Kinematics::PlaneStrain
        };
        for err in [
            model::elasticity(&fes, model).unwrap_err(),
            model::plasticity_perfect(&fes, model).unwrap_err(),
            model::mazars(&fes, model).unwrap_err(),
        ] {
            assert!(
                format!("{err}").contains("manifold, not a"),
                "expected a manifold rejection, got: {err}"
            );
        }
    }
    Ok(())
}

// ─── Lois non linéaires : équivalence avec la loi 3-D ────────────────────────

/// A single-cell model of `kind` ("plasticity" / "mazars") on the given
/// geometry, plus its material field. `axisymmetric` picks the meridian-plane
/// QUA4 (2-D, hoop = `zz`) or the Cartesian HEX8 (full 3-D).
fn nonlinear_cell(
    kind: &str,
    axisymmetric: bool,
) -> Result<(Model, ElementField, FiniteElementSpace)> {
    let (mesh, model_kind) = if axisymmetric {
        let coords = Handle::new(Coords::axisymmetric()?);
        let n: Vec<Node> = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(r, z)| Node::create_in(coords.clone(), &[r, z]))
            .collect::<Result<_>>()?;
        let mut m = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        m.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())?;
        (m, Kinematics::Axisymmetric)
    } else {
        let coords = Handle::new(Coords::new(3)?);
        let n: Vec<Node> = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        .iter()
        .map(|&(x, y, z)| Node::create_in(coords.clone(), &[x, y, z]))
        .collect::<Result<_>>()?;
        let mut m = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
        m.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())?;
        (m, Kinematics::Full3D)
    };
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let model = match kind {
        "plasticity" => model::plasticity_perfect(&fes, model_kind)?,
        _ => model::mazars(&fes, model_kind)?,
    };
    let props: Vec<(&str, f64)> = if kind == "plasticity" {
        vec![("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)]
    } else {
        vec![
            ("E", 30_000.0),
            ("nu", 0.2),
            ("eps_d0", 1e-4),
            ("A_t", 0.8),
            ("B_t", 20_000.0),
            ("A_c", 1.4),
            ("B_c", 1900.0),
        ]
    };
    let materials = pyrucast::ops::element_field::material_field(&model, &props)?;
    Ok((model, materials, fes))
}

/// Impose a uniform strain state on a one-cell model and return its stress.
/// `(rr, zz, hoop, rz)` are the four axisymmetric components; the 3-D twin gets
/// the very same tensor with `ε_yz = ε_xz = 0`.
fn stress_of(
    model: &Model,
    materials: &ElementField,
    fes: &FiniteElementSpace,
    axisymmetric: bool,
    eps: (f64, f64, f64, f64),
    prev: Option<&ElementField>,
) -> Result<ElementField> {
    let (rr, zz, hoop, rz) = eps;
    let mut names = vec!["eps_xx", "eps_yy", "eps_zz", "eps_xy"];
    if !axisymmetric {
        names.extend(["eps_yz", "eps_xz"]);
    }
    let mut sub = pyrucast::containers::element_field::SubElementField::new(
        fes.get(0)?,
        names.iter().map(|s| s.to_string()).collect(),
    )?;
    sub.set_uniform("eps_xx", rr)?;
    sub.set_uniform("eps_yy", zz)?;
    sub.set_uniform("eps_zz", hoop)?;
    sub.set_uniform("eps_xy", rz)?;
    let mut strain = ElementField::empty();
    strain.add_sub(Handle::new(sub))?;
    pyrucast::ops::element_field::behavior::integrate(model, &strain, prev, materials, None)
}

/// The axisymmetric law must be the **same law** as the 3-D one restricted to
/// `[rr, zz, θθ, rz]`: fed the same strain tensor, it must return the same
/// stress. Run over several increments so the history path (`prev`) is
/// exercised too — this is what pins the state plumbing (`eps_p_*`, the echoed
/// `ε(A)`, and the `σ_zz` that axisymmetric carries in its dual rather than as
/// an echo).
#[test]
fn nonlinear_laws_agree_with_their_3d_twin() -> Result<()> {
    for kind in ["plasticity", "mazars"] {
        let (axi_m, axi_mat, axi_fes) = nonlinear_cell(kind, true)?;
        let (sol_m, sol_mat, sol_fes) = nonlinear_cell(kind, false)?;
        let (mut axi_prev, mut sol_prev) = (None, None);

        // A load path past yield / the damage threshold, with a non-proportional
        // shear step so the return map really works. Scaled per law: plasticity
        // yields around ε ≈ 3e-3, Mazars damages from eps_d0 = 1e-4 — and Mazars
        // is kept at *moderate* damage, since a fully damaged point carries
        // near-zero stress, which would make the comparison below vacuous again.
        let scale = if kind == "plasticity" { 1.0 } else { 0.02 };
        let path = [
            (2.0e-3, -2.0e-3, 5.0e-4, 0.0),
            (1.0e-2, -9.0e-3, 2.0e-3, 1.0e-3),
            (2.5e-2, -2.0e-2, 5.0e-3, 6.0e-3),
            (1.8e-2, -1.4e-2, 3.0e-3, 3.0e-3), // unloading
        ]
        .map(|(a, b, c, d): (f64, f64, f64, f64)| (a * scale, b * scale, c * scale, d * scale));
        for (step, &eps) in path.iter().enumerate() {
            let a = stress_of(&axi_m, &axi_mat, &axi_fes, true, eps, axi_prev.as_ref())?;
            let s = stress_of(&sol_m, &sol_mat, &sol_fes, false, eps, sol_prev.as_ref())?;
            let (ga, gs) = (a.get(0)?.read(), s.get(0)?.read());
            let mut peak = 0.0_f64;
            for comp in ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy"] {
                let (va, vs) = (ga.value(0, 0, comp)?, gs.value(0, 0, comp)?);
                peak = peak.max(vs.abs());
                assert!(
                    (va - vs).abs() < 1e-9 * (1.0 + vs.abs()),
                    "{kind} step {step}: axisymmetric {comp} = {va}, 3-D twin = {vs}"
                );
            }
            // …and the stresses compared must be of a real magnitude, else the
            // relative tolerance above would be met by any two near-zero values.
            assert!(
                peak > 1.0,
                "{kind} step {step}: stresses too small ({peak})"
            );
            // The 3-D twin must see no out-of-plane shear (axial symmetry).
            if kind == "plasticity" {
                for comp in ["sigma_yz", "sigma_xz"] {
                    assert!(gs.value(0, 0, comp)?.abs() < 1e-12);
                }
            }
            // Guard against a vacuous test: from the second step on, the law
            // must actually be non-linear (yielding / damaging), otherwise this
            // would only be comparing two elastic evaluations.
            if step >= 1 {
                let nonlinear = if kind == "plasticity" {
                    ga.value(0, 0, "p")?
                } else {
                    ga.value(0, 0, "damage")?
                };
                assert!(
                    nonlinear > 1e-6,
                    "{kind} step {step}: the load path stayed linear ({nonlinear}) — \
                     the comparison would prove nothing"
                );
            }
            drop(ga);
            drop(gs);
            axi_prev = Some(a);
            sol_prev = Some(s);
        }
    }
    Ok(())
}
