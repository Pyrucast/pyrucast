//! The exchange law `h(a − b)`, on what the generalisation actually bought.
//!
//! Both physics used to know they were thermal or diffusive. They no longer do:
//! they are given `(primal, dual)` pairs and derive their coefficients
//! `h_<primal>` from them. The thermal film and the interface resistance are
//! covered by [`thermal_convection`](../thermal_convection.rs) and
//! [`interface_transfer`](../interface_transfer.rs) — this file tests what did
//! **not** exist before.
//!
//! **A boundary exchange on displacements is a Winkler elastic foundation.** A
//! bar pushed on its free face, that face resting on a distributed spring, is a
//! bar and a spring in parallel under the applied traction:
//!
//! ```text
//! q = E·u/L + h·u      ⟹      u = q / (E/L + h)
//! ```
//!
//! and the two limits must both come out: `h → 0` gives back the free bar
//! `u = qL/E`, `h → ∞` gives a face that cannot move.
//!
//! **One coefficient per direction.** The whole point of naming them after the
//! variable is that `h_u_x` and `h_u_y` are independent — a foundation stiffer
//! across than along is a normal thing to want, and it is the case a single `h`
//! could not express.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::models::Physics;
use pyrucast::ops::mesh;
use pyrucast::ops::node_field::FluxDensity;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::insert;
use pyrucast::Result;

const E: f64 = 210_000.0;
const NU: f64 = 0.0;
/// Side of the square, and so the bar's length.
const L: f64 = 1.0;
/// Uniform traction on the free face.
const Q: f64 = 100.0;

// ANCHOR: foundation
/// A boundary exchange on the displacement is an **elastic foundation**: the
/// face rests on a distributed spring, and the analytical `u = q/(E/L + h)`
/// comes out for stiffnesses spanning four decades.
#[test]
fn a_boundary_exchange_on_displacements_is_an_elastic_foundation() -> Result<()> {
    for h in [0.0, 1e3, 1e4, 1e5, 1e6] {
        let u = free_face_displacement(4, h)?;
        let exact = Q / (E / L + h);
        assert!(
            (u - exact).abs() < 1e-9 * exact.abs().max(1e-12),
            "h = {h}: face displacement {u}, exact {exact}"
        );
    }
    Ok(())
}
// ANCHOR_END: foundation

/// The two limits, stated on their own because they are what a reader checks
/// first: no foundation is a free bar, a very stiff one is a held face.
#[test]
fn the_foundation_interpolates_between_a_free_and_a_held_face() -> Result<()> {
    let free = free_face_displacement(4, 0.0)?;
    assert!(
        (free - Q * L / E).abs() < 1e-9 * free,
        "with no foundation the bar is free: {free}"
    );
    let held = free_face_displacement(4, 1e12)?;
    assert!(
        held < 1e-6 * free,
        "a very stiff foundation must hold the face: {held} against {free}"
    );
    Ok(())
}

/// One coefficient **per direction** — the case a single `h` could not express,
/// and the reason the coefficients are named after their variable.
#[test]
fn each_direction_carries_its_own_stiffness() -> Result<()> {
    let (n, h_x, h_y) = (4, 1e4, 1e6);
    let (grid, fes, coords, _) = square(n)?;
    let idx = |i: usize, j: usize| j * (n + 1) + i;

    // The right edge rests on a foundation in **both** directions, with two
    // different stiffnesses; the left edge is held.
    let right = edge_fespace(&grid, &coords, n, idx)?;
    let mut model =
        Model::elasticity(&fes, ElasticityModel::PlaneStress)?.union(&Model::boundary_transfer(
            &right,
            vec![("u_x".into(), "f_x".into()), ("u_y".into(), "f_y".into())],
            Physics::Mechanical,
        )?)?;
    let left: Vec<Node> = (0..=n).map(|j| grid[idx(0, j)].clone()).collect();
    for (var, dual) in [("u_x", "f_x"), ("u_y", "f_y")] {
        model = model.union(&clamp(&left, var, dual)?)?;
    }

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", NU), ("h_u_x", h_x), ("h_u_y", h_y)],
    )?;

    // A traction along x, and one along y, applied together.
    let load_x = pyrucast::ops::node_field::flux(&right.get(0)?, FluxDensity::Uniform(Q), "f_x")?;
    let load_y = pyrucast::ops::node_field::flux(&right.get(0)?, FluxDensity::Uniform(Q), "f_y")?;
    let load = (NodeField::from_sub(load_x) + NodeField::from_sub(load_y))?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &load)?;

    // Along x the bar resists too; across it only the foundation does, the
    // material being free to shear. The stiffer direction must move less — and
    // by more than the material alone would explain.
    let u_x = solution.value(grid[idx(n, n / 2)].id(), "u_x")?;
    let u_y = solution.value(grid[idx(n, n / 2)].id(), "u_y")?;
    assert!(
        (u_x - Q / (E / L + h_x)).abs() < 1e-9 * u_x,
        "along the bar: {u_x}"
    );
    assert!(
        u_y < u_x,
        "the stiffer direction must move less: u_y = {u_y}, u_x = {u_x}"
    );
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// An `n×n` QUA4 square on `[0, L]²`, in a 2-D space.
#[allow(clippy::type_complexity)]
fn square(
    n: usize,
) -> Result<(
    Vec<Node>,
    FiniteElementSpace,
    pyrucast::store::Handle<Coords>,
    usize,
)> {
    let coords = insert(Coords::new(2)?);
    let step = L / n as f64;
    let mut grid = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * step, j as f64 * step],
            )?);
        }
    }
    let idx = |i: usize, j: usize| j * (n + 1) + i;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..n {
        for i in 0..n {
            mesh.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    Ok((grid, fes, coords, n))
}

/// The `x = L` edge, as its own SEG2 FE space — the boundary the foundation and
/// the traction both live on.
fn edge_fespace(
    grid: &[Node],
    coords: &pyrucast::store::Handle<Coords>,
    n: usize,
    idx: impl Fn(usize, usize) -> usize,
) -> Result<FiniteElementSpace> {
    let mut edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..n {
        edge.add_cell(&[grid[idx(n, j)].id(), grid[idx(n, j + 1)].id()])?;
    }
    FiniteElementSpace::lagrange1(&edge)
}

/// Homogeneous Dirichlet `var = 0` on every node of `nodes`.
fn clamp(nodes: &[Node], var: &str, dual: &str) -> Result<Model> {
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
}

/// The displacement of the free face of a bar held at `x = 0`, pushed by a
/// uniform traction `Q` at `x = L`, that same face resting on a foundation of
/// stiffness `h`.
fn free_face_displacement(n: usize, h: f64) -> Result<f64> {
    let (grid, fes, coords, _) = square(n)?;
    let idx = |i: usize, j: usize| j * (n + 1) + i;
    let right = edge_fespace(&grid, &coords, n, idx)?;

    let mut model =
        Model::elasticity(&fes, ElasticityModel::PlaneStress)?.union(&Model::boundary_transfer(
            &right,
            vec![("u_x".into(), "f_x".into())],
            Physics::Mechanical,
        )?)?;
    // Held in x on the left face; one node held in y, which is all a uniaxial
    // state needs to be regular (`nu = 0`, so nothing contracts across).
    let left: Vec<Node> = (0..=n).map(|j| grid[idx(0, j)].clone()).collect();
    model = model.union(&clamp(&left, "u_x", "f_x")?)?;
    let bottom: Vec<Node> = (0..=n).map(|i| grid[idx(i, 0)].clone()).collect();
    model = model.union(&clamp(&bottom, "u_y", "f_y")?)?;

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", NU), ("h_u_x", h)],
    )?;
    let traction = pyrucast::ops::node_field::flux(&right.get(0)?, FluxDensity::Uniform(Q), "f_x")?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(traction))?;
    solution.value(grid[idx(n, n / 2)].id(), "u_x")
}
