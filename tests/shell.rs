//! Reissner-Mindlin shells, on the two things a shell element must get right.
//!
//! **Bending.** A clamped square plate under a uniform load has a textbook
//! deflection `w = α·qa⁴/D` with `α = 0.00126` and `D = Eh³/12(1−ν²)`. A mesh of
//! bilinear facets converges to it from below; the test asks for the right value
//! within the discretisation error, and for **convergence** as the mesh refines.
//!
//! **Not locking.** As the plate thins, the transverse-shear stiffness overwhelms
//! the bending one by `1/h²`. Integrated fully, the element then refuses to bend
//! at all — shear locking — and the deflection collapses towards zero however
//! fine the mesh. Reduced integration is the cure, and the test that proves it
//! works is the only one that matters here: the **normalised** deflection
//! `w·D/qa⁴` must stay put as the thickness falls by two decades. A locking
//! element fails it by orders of magnitude.
//!
//! Membrane behaviour is checked separately, and so is the drilling degree of
//! freedom — which must remove the singularity **without** resisting a rigid
//! rotation of the facet about its own normal.
//!
//! Single source for the « coque épaisse » example of the book; runs under
//! `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::shell::ShellModel;
use pyrucast::ops::mesh;
use pyrucast::ops::node_field::FluxDensity;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::insert;
use pyrucast::Result;

const E: f64 = 210_000.0;
const NU: f64 = 0.3;
/// Side of the square plate.
const A: f64 = 1.0;
/// Uniform transverse pressure.
const Q: f64 = 1.0;

#[test]
fn a_clamped_plate_matches_its_textbook_deflection() -> Result<()> {
    let h = 0.01; // thin enough to compare with plate theory, thick enough to be safe
    let w = central_deflection(12, h)?;
    // Timoshenko & Woinowsky-Krieger: w_max = 0.00126 · qa⁴/D.
    let d = E * h * h * h / (12.0 * (1.0 - NU * NU));
    let exact = 0.00126 * Q * A.powi(4) / d;
    assert!(
        (w - exact).abs() < 0.06 * exact,
        "central deflection {w}, textbook {exact}"
    );
    Ok(())
}
// ANCHOR_END: example

/// The deflection must **converge** as the mesh refines, and from below — a
/// displacement formulation is too stiff, and gets less so.
#[test]
fn the_deflection_converges_with_the_mesh() -> Result<()> {
    let h = 0.01;
    let coarse = central_deflection(4, h)?;
    let medium = central_deflection(8, h)?;
    let fine = central_deflection(16, h)?;
    assert!(coarse < medium && medium < fine, "{coarse} {medium} {fine}");
    // …and the increments shrink: the sequence is converging, not drifting.
    assert!(
        fine - medium < 0.5 * (medium - coarse),
        "the increments must shrink: {coarse} {medium} {fine}"
    );
    Ok(())
}

/// **The** test for a Mindlin element. Normalised by the plate stiffness, the
/// deflection of a thin plate is a constant of the theory — independent of the
/// thickness. An element that locks loses it by orders of magnitude as `h`
/// falls; reduced integration of the shear is what keeps it.
#[test]
fn the_element_does_not_lock_as_the_plate_thins() -> Result<()> {
    let mut normalised = Vec::new();
    for h in [0.05, 0.01, 0.002, 0.0005] {
        let w = central_deflection(8, h)?;
        let d = E * h * h * h / (12.0 * (1.0 - NU * NU));
        normalised.push(w * d / (Q * A.powi(4)));
    }
    let (first, last) = (normalised[0], *normalised.last().unwrap());
    // Thin-plate theory is the limit, so the sequence settles rather than
    // collapsing. A locking element would send `last` towards zero.
    assert!(
        last > 0.5 * first,
        "the normalised deflection collapsed — the element locks: {normalised:?}"
    );
    // And the two thinnest agree closely: the limit has been reached.
    let (a, b) = (normalised[2], normalised[3]);
    assert!(
        (a - b).abs() < 0.05 * a,
        "the thin limit must settle: {normalised:?}"
    );
    Ok(())
}

/// A shell carries membrane forces too: stretched in its own plane, it must
/// behave as a plane-stress sheet, `u = NL/(Eh)`.
#[test]
fn a_stretched_shell_behaves_as_a_membrane() -> Result<()> {
    let h = 0.02;
    let (grid, fes, coords, n) = plate(4)?;
    let idx = |i: usize, j: usize| j * (n + 1) + i;

    let mut model = Model::shell(&fes, ShellModel::Thick)?;
    // Roller on the x = 0 edge (u_x), one point pinned in y and z, and every
    // rotation held: a pure membrane state.
    let left: Vec<Node> = (0..=n).map(|j| grid[idx(0, j)].clone()).collect();
    model = model.union(&clamp(&left, "u_x", "f_x")?)?;
    let bottom: Vec<Node> = (0..=n).map(|i| grid[idx(i, 0)].clone()).collect();
    model = model.union(&clamp(&bottom, "u_y", "f_y")?)?;
    let all: Vec<Node> = grid.clone();
    for (var, dual) in [
        ("u_z", "f_z"),
        ("r_x", "m_x"),
        ("r_y", "m_y"),
        ("r_z", "m_z"),
    ] {
        model = model.union(&clamp(&all, var, dual)?)?;
    }
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU), ("h", h)])?;

    // A uniform traction on the x = 1 edge.
    let mut edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..n {
        edge.add_cell(&[grid[idx(n, j)].id(), grid[idx(n, j + 1)].id()])?;
    }
    let edge_fes = FiniteElementSpace::lagrange1(&edge)?;
    let traction =
        pyrucast::ops::node_field::flux(&edge_fes.get(0)?, FluxDensity::Uniform(Q * h), "f_x")?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(traction))?;

    // Uniaxial stress σ = q (the traction per unit thickness), so u(1) = σ·L/E.
    let u = solution.value(grid[idx(n, n)].id(), "u_x")?;
    let exact = Q * A / E;
    assert!(
        (u - exact).abs() < 1e-9 * exact,
        "membrane stretch {u}, exact {exact}"
    );
    Ok(())
}

/// The drilling degree of freedom must remove the singularity **without**
/// resisting a rigid rotation of the facet about its own normal — which costs no
/// energy. Tying it to the membrane rotation is what achieves both; a diagonal
/// penalty, the tempting shortcut, would fail this.
#[test]
fn a_rigid_drilling_rotation_costs_no_energy() -> Result<()> {
    let h = 0.02;
    let (grid, fes, _coords, n) = plate(2)?;
    let model = Model::shell(&fes, ShellModel::Thick)?;
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU), ("h", h)])?;
    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;

    // A rigid rotation of the whole plate about z: u = (−y, x, 0), θ_z = 1.
    let sm = insert(SubMesh::poi1_from_nodes(&grid)?);
    let mut field = SubNodeField::from_poi1(&sm, comps_of())?;
    for node in &grid {
        let p = node.position()?;
        field.set_value(node.id(), "u_x", -p[1])?;
        field.set_value(node.id(), "u_y", p[0])?;
        field.set_value(node.id(), "r_z", 1.0)?;
    }
    let u = NodeField::from_sub(field);

    // The strain energy uᵀKu must vanish: a rigid motion is in the kernel.
    let energy = energy_of(&k, &u, &grid)?;
    // The comparison that means something: a **spurious** drilling field — the
    // same θ_z with no membrane displacement under it — must cost real energy,
    // which is what removes the singularity. The rigid motion costs none.
    let mut spurious = SubNodeField::from_poi1(&sm, comps_of())?;
    for node in &grid {
        spurious.set_value(node.id(), "r_z", 1.0)?;
    }
    let spurious = NodeField::from_sub(spurious);
    let stiff = energy_of(&k, &spurious, &grid)?;
    let _ = n;
    assert!(
        stiff > 0.0,
        "the drilling DOF must be restrained (got {stiff})"
    );
    assert!(
        energy.abs() < 1e-9 * stiff,
        "a rigid drilling rotation must cost no energy: {energy} against {stiff}"
    );
    Ok(())
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

/// The six shell DOF names.
fn comps_of() -> Vec<String> {
    ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The strain energy `uᵀKu` of a nodal field.
fn energy_of(
    k: &pyrucast::containers::matrix::Matrix,
    u: &NodeField,
    grid: &[Node],
) -> Result<f64> {
    let ku = k.mul_field(u)?;
    let mut energy = 0.0;
    for node in grid {
        for (var, dual) in [
            ("u_x", "f_x"),
            ("u_y", "f_y"),
            ("u_z", "f_z"),
            ("r_x", "m_x"),
            ("r_y", "m_y"),
            ("r_z", "m_z"),
        ] {
            energy += u.value(node.id(), var)? * ku.value(node.id(), dual)?;
        }
    }
    Ok(energy)
}

/// An `n×n` QUA4 plate on `[0, A]²`, flat in the `z = 0` plane of a 3-D space.
#[allow(clippy::type_complexity)]
fn plate(
    n: usize,
) -> Result<(
    Vec<Node>,
    FiniteElementSpace,
    pyrucast::store::Handle<Coords>,
    usize,
)> {
    let coords = insert(Coords::new(3)?);
    let step = A / n as f64;
    let mut grid = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * step, j as f64 * step, 0.0],
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

/// The central deflection of a clamped square plate under a uniform pressure,
/// on an `n×n` mesh of thickness `h`.
fn central_deflection(n: usize, h: f64) -> Result<f64> {
    let (grid, fes, coords, _) = plate(n)?;
    let idx = |i: usize, j: usize| j * (n + 1) + i;

    let mut model = Model::shell(&fes, ShellModel::Thick)?;
    // Clamped all round: every DOF held on the boundary.
    let mut boundary = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            if i == 0 || i == n || j == 0 || j == n {
                boundary.push(grid[idx(i, j)].clone());
            }
        }
    }
    for (var, dual) in [
        ("u_x", "f_x"),
        ("u_y", "f_y"),
        ("u_z", "f_z"),
        ("r_x", "m_x"),
        ("r_y", "m_y"),
        ("r_z", "m_z"),
    ] {
        model = model.union(&clamp(&boundary, var, dual)?)?;
    }
    // The in-plane DOFs and the drilling one play no part in pure bending, and a
    // flat plate leaves them unrestrained; hold them on the **interior** nodes so
    // the system is regular. Only the interior — the boundary already holds them,
    // and imposing a DOF twice gives two multipliers for one condition, which is
    // singular by construction.
    let interior: Vec<Node> = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| (i, j)))
        .filter(|&(i, j)| i != 0 && i != n && j != 0 && j != n)
        .map(|(i, j)| grid[idx(i, j)].clone())
        .collect();
    if !interior.is_empty() {
        for (var, dual) in [("u_x", "f_x"), ("u_y", "f_y"), ("r_z", "m_z")] {
            model = model.union(&clamp(&interior, var, dual)?)?;
        }
    }
    let materials =
        pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU), ("h", h)])?;

    // The uniform pressure, as consistent nodal loads on the surface itself.
    let surface = read_fespace(&fes)?;
    let load = pyrucast::ops::node_field::flux(&surface, FluxDensity::Uniform(Q), "f_z")?;
    let _ = coords;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solve(&k, &NodeField::from_sub(load))?;
    // The centre of an even mesh is a node.
    solution.value(grid[idx(n / 2, n / 2)].id(), "u_z")
}

fn read_fespace(
    fes: &FiniteElementSpace,
) -> Result<
    pyrucast::store::Handle<pyrucast::containers::finite_element_space::SubFiniteElementSpace>,
> {
    fes.get(0)
}
