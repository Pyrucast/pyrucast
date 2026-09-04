//! Oriented heat conduction — orthotropic and anisotropic conductivity.
//!
//! The check is a **patch test**, which is what an oriented conductivity calls
//! for: a linear temperature field is harmonic for *any* constant conductivity
//! tensor (`∇·(K∇T) = 0` because `∇T` is constant), so imposing `T = x` on the
//! whole boundary of the unit square must reproduce `T = x` inside, whatever `K`
//! is. That isolates the tensor from the solution: if the assembly were using a
//! wrong `K`, the field would still be linear, so the test does not stop there —
//! it also reads the resulting **flux** back through the behaviour integration
//! and compares it to `K·∇T` computed by hand.
//!
//! With `∇T = (1, 0)` the weak-form flux is simply the first column of `K`, and
//! for an orthotropic material whose first axis makes an angle `θ` with `x`,
//!
//! ```text
//! K_xx = k₁cos²θ + k₂sin²θ        K_yx = (k₁ − k₂)·cosθ·sinθ
//! ```
//!
//! so the off-diagonal term is non-zero exactly when the material is both
//! anisotropic **and** misaligned — the case that a wrong rotation gets wrong.
//!
//! Single source for the « conduction orthotrope » example of the thermal book
//! chapter; runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

/// `N×N` QUA4 grid on the unit square.
const N: usize = 3;

#[test]
fn orthotropic_conduction_passes_the_linear_patch_test() -> Result<()> {
    const K1: f64 = 12.0; // along the first material axis
    const K2: f64 = 3.0; // transverse
    let theta = 30.0_f64.to_radians();
    let (c, s) = (theta.cos(), theta.sin());

    let (grid, fes, _coords) = unit_square()?;
    let (model, multipliers) = patch_model(&grid, &fes, MaterialSymmetry::Orthotropic)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("k_1", K1),
            ("k_2", K2),
            ("k_3", K1),
            ("V1X", c),
            ("V1Y", s),
        ],
    )?;

    let solution = solve_patch(&model, &materials, &grid, &multipliers)?;

    // The linear field must be reproduced exactly, tensor or no tensor.
    let h = 1.0 / N as f64;
    let tol = 1e-9;
    for j in 0..=N {
        for i in 0..=N {
            let x = i as f64 * h;
            let got = solution.value(grid[j * (N + 1) + i].id(), "T")?;
            assert!((got - x).abs() < tol, "T({x}) = {got}");
        }
    }

    // …and the flux must be the first column of the **rotated** tensor.
    let expect_xx = K1 * c * c + K2 * s * s;
    let expect_yx = (K1 - K2) * c * s;
    let (fx, fy) = uniform_flux(&model, &solution, &fes, &materials, &grid)?;
    assert!(
        (fx - expect_xx).abs() < 1e-9,
        "flux_x = {fx}, expected {expect_xx}"
    );
    assert!(
        (fy - expect_yx).abs() < 1e-9,
        "flux_y = {fy}, expected {expect_yx}"
    );
    Ok(())
}
// ANCHOR_END: example

/// The anisotropic law reads the symmetric tensor directly — same patch test,
/// and the flux must be its first column, unrotated (`V1` along `x`).
#[test]
fn anisotropic_conduction_reads_its_tensor() -> Result<()> {
    const K11: f64 = 9.0;
    const K12: f64 = 2.0;
    const K22: f64 = 4.0;

    let (grid, fes, _coords) = unit_square()?;
    let (model, multipliers) = patch_model(&grid, &fes, MaterialSymmetry::Anisotropic)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[
            ("k_11", K11),
            ("k_12", K12),
            ("k_13", 0.0),
            ("k_22", K22),
            ("k_23", 0.0),
            ("k_33", K11),
            ("V1X", 1.0),
            ("V1Y", 0.0),
        ],
    )?;

    let solution = solve_patch(&model, &materials, &grid, &multipliers)?;
    let (fx, fy) = uniform_flux(&model, &solution, &fes, &materials, &grid)?;
    assert!((fx - K11).abs() < 1e-9, "flux_x = {fx}");
    // The cross term is what makes the tensor anisotropic — `∇T = (1,0)` still
    // drives a flux across `y`.
    assert!((fy - K12).abs() < 1e-9, "flux_y = {fy}");
    Ok(())
}

/// Orthotropy with equal conductivities is isotropy, whatever the frame.
#[test]
fn orthotropy_with_equal_conductivities_ignores_its_frame() -> Result<()> {
    const K: f64 = 7.0;
    for angle_deg in [0.0_f64, 47.0, 90.0] {
        let a = angle_deg.to_radians();
        let (grid, fes, _coords) = unit_square()?;
        let (model, multipliers) = patch_model(&grid, &fes, MaterialSymmetry::Orthotropic)?;
        let materials = pyrucast::ops::element_field::material_field(
            &model,
            &[
                ("k_1", K),
                ("k_2", K),
                ("k_3", K),
                ("V1X", a.cos()),
                ("V1Y", a.sin()),
            ],
        )?;
        let solution = solve_patch(&model, &materials, &grid, &multipliers)?;
        let (fx, fy) = uniform_flux(&model, &solution, &fes, &materials, &grid)?;
        assert!((fx - K).abs() < 1e-9, "angle {angle_deg}°: flux_x = {fx}");
        assert!(fy.abs() < 1e-9, "angle {angle_deg}°: flux_y = {fy}");
    }
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

fn unit_square() -> Result<(
    Vec<Node>,
    FiniteElementSpace,
    pyrucast::handle::Handle<Coords>,
)> {
    let h = 1.0 / N as f64;
    let coords = Handle::new(Coords::new(2)?);
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * h, j as f64 * h],
            )?);
        }
    }
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..N {
        for i in 0..N {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((grid, fes, coords))
}

/// Orthotropic conduction plus one Dirichlet per **boundary** node, imposing the
/// linear field `T = x`. Returns the model and the multiplier node of each
/// constrained node, paired with the value to impose.
#[allow(clippy::type_complexity)]
fn patch_model(
    grid: &[Node],
    fes: &FiniteElementSpace,
    symmetry: MaterialSymmetry,
) -> Result<(Model, Vec<(pyrucast::atoms::NodeId, f64)>)> {
    let h = 1.0 / N as f64;
    let mut model = model::heat_conduction_with_symmetry(fes, symmetry)?;
    let mut multipliers = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            if i != 0 && i != N && j != 0 && j != N {
                continue; // interior node — left free, it is what we check
            }
            let node = grid[j * (N + 1) + i].clone();
            let imposed =
                Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&node))?);
            let multiplier = mesh::barycenter(&imposed)?;
            multipliers.push((multiplier.node(0, 0, 0)?.id(), i as f64 * h));
            model = model.union(&model::dirichlet(
                &model,
                "T",
                &imposed,
                &multiplier,
                Default::default(),
            )?)?;
        }
    }
    Ok((model, multipliers))
}

/// Build the right-hand side (the imposed values on the multiplier nodes) and
/// solve.
fn solve_patch(
    model: &Model,
    materials: &ElementField,
    grid: &[Node],
    multipliers: &[(pyrucast::atoms::NodeId, f64)],
) -> Result<NodeField> {
    let mut load_sm = SubMesh::new(grid[0].coords(), ElementType::POI1);
    for (mult, _) in multipliers {
        load_sm.add_cell(&[*mult])?;
    }
    let load_sm = Handle::new(load_sm);
    let mut rhs = SubNodeField::from_poi1(&load_sm, vec!["imposed_T".into()])?;
    for (mult, value) in multipliers {
        rhs.set_value(*mult, "imposed_T", *value)?;
    }
    let rhs = NodeField::from_sub(rhs);
    let stiffness = pyrucast::ops::matrix::stiffness(model, materials)?;
    solve(&stiffness, &rhs)
}

/// The (uniform) weak-form flux `K·∇T`, read back through the behaviour
/// integration at the first Gauss point of the first cell.
fn uniform_flux(
    model: &Model,
    solution: &NodeField,
    fes: &FiniteElementSpace,
    materials: &ElementField,
    grid: &[Node],
) -> Result<(f64, f64)> {
    // `solve` returns the Lagrange multipliers alongside the temperature;
    // restrict onto a temperature-shaped field before differentiating.
    let sm = Handle::new(SubMesh::poi1_from_nodes(grid)?);
    let zero = SubNodeField::from_poi1(&sm, vec!["T".to_string()])?;
    let temperature =
        pyrucast::ops::node_field::restrict_like(solution, &NodeField::from_sub(zero))?;
    let gradient = pyrucast::ops::element_field::gradient(&temperature, fes)?;
    let flux =
        pyrucast::ops::element_field::behavior::integrate(model, &gradient, None, materials, None)?;
    let sub = flux.get(0)?;
    let sub = sub.read();
    Ok((sub.value(0, 0, "flux_x")?, sub.value(0, 0, "flux_y")?))
}
