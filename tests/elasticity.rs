//! Worked linear-elasticity example, exercised end-to-end through the public
//! API.
//!
//! Uniaxial tension of the unit square `[0,1]²` (QUA4 grid), **plane stress**:
//! rollers `u_x = 0` on the left edge and `u_y = 0` on the bottom edge, and a
//! uniform traction `S` applied on the right edge — turned into consistent
//! nodal forces by the [`flux`](pyrucast::ops::assemble::flux) operator. The
//! exact field is uniform stress `σ_xx = S` with the linear displacement
//! `u_x = (S/E)·x`, `u_y = −(ν S/E)·y`, reproduced nodally by Q1.
//!
//! Single source for the « élasticité » example of the mechanics book chapter;
//! runs under `cargo test`.

// ANCHOR: example
use pyrucast::aggregate::Aggregate;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::NodeField;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::assemble::{self, FluxDensity};
use pyrucast::ops::solver::lu::solve;
use pyrucast::ops::{build, mesher};
use pyrucast::store::insert;
use pyrucast::Result;

#[test]
fn elasticity_unit_square_uniaxial_tension() -> Result<()> {
    const E: f64 = 210.0; // Young's modulus
    const NU: f64 = 0.3; // Poisson's ratio
    const S: f64 = 2.0; // traction on the right edge
    const N: usize = 2; // N×N QUA4 grid
    let h = 1.0 / N as f64;

    // ── Maillage QUA4 sur [0,1]² ───────────────────────────────────────────
    let coords = insert(Coords::new(2)?);
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * h, j as f64 * h],
            )?);
        }
    }
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

    // ── Modèle : élasticité plane stress + appuis (rollers) ────────────────
    let roller = |nodes: &[Node], var: &str, dual: &str| -> Result<Model> {
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
        let multiplier = mesher::barycenter(&imposed)?;
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
    let left: Vec<Node> = (0..=N).map(|j| grid[idx(0, j)].clone()).collect();
    let bottom: Vec<Node> = (0..=N).map(|i| grid[idx(i, 0)].clone()).collect();
    let mut model = Model::elasticity(&fes, ElasticityModel::PlaneStress)?;
    model = model.union(&roller(&left, "u_x", "f_x")?)?;
    model = model.union(&roller(&bottom, "u_y", "f_y")?)?;

    let materials = build::material_field(&model, &[("E", E), ("nu", NU)])?;

    // ── Chargement : traction S sur le bord droit (charges nodales cohérentes
    //    via l'opérateur flux, sur la composante f_x) ────────────────────────
    let mut right_edge = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for j in 0..N {
        right_edge.add_cell(&[grid[idx(N, j)].id(), grid[idx(N, j + 1)].id()])?;
    }
    let right_fes = FiniteElementSpace::lagrange1(&right_edge)?;
    let traction = assemble::flux(&right_fes.get(0)?, FluxDensity::Uniform(S), "f_x")?;
    let rhs = NodeField::from_sub(traction);

    // ── Assemblage + résolution ────────────────────────────────────────────
    let stiffness = assemble::stiffness(&model, &materials)?;
    let solution = solve(&stiffness, &rhs)?;

    // ── Comparaison à l'analytique u_x = (S/E)·x, u_y = −(ν S/E)·y ─────────
    let tol = 1e-10;
    for j in 0..=N {
        for i in 0..=N {
            let (x, y) = (i as f64 * h, j as f64 * h);
            let ux = solution.value(grid[idx(i, j)].id(), "u_x")?;
            let uy = solution.value(grid[idx(i, j)].id(), "u_y")?;
            assert!((ux - S / E * x).abs() < tol, "u_x({x},{y}) = {ux}");
            assert!((uy + NU * S / E * y).abs() < tol, "u_y({x},{y}) = {uy}");
        }
    }
    Ok(())
}
// ANCHOR_END: example

/// Homogeneous Dirichlet `var = 0` on every node of `nodes`.
fn clamp(nodes: &[Node], var: &str, dual: &str) -> Result<Model> {
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
    let multiplier = mesher::barycenter(&imposed)?;
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

/// 3-D solid: uniaxial tension of a unit HEX8 cube. Symmetry rollers on the
/// three faces through the origin, traction `S` on the `x = 1` face ⇒
/// `u_x = (S/E)·x`, `u_y = −(ν S/E)·y`, `u_z = −(ν S/E)·z`.
#[test]
fn elasticity_unit_cube_uniaxial_tension() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;
    const S: f64 = 2.0;

    let coords = insert(Coords::new(3)?);
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let nodes: Vec<Node> = points
        .iter()
        .map(|c| Node::create_in(coords.clone(), c))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
    mesh.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    let pick = |ids: &[usize]| ids.iter().map(|&i| nodes[i].clone()).collect::<Vec<_>>();
    let mut model = Model::elasticity(&fes, ElasticityModel::Solid)?;
    model = model.union(&clamp(&pick(&[0, 3, 4, 7]), "u_x", "f_x")?)?; // x = 0 face
    model = model.union(&clamp(&pick(&[0, 1, 4, 5]), "u_y", "f_y")?)?; // y = 0 face
    model = model.union(&clamp(&pick(&[0, 1, 2, 3]), "u_z", "f_z")?)?; // z = 0 face

    let materials = build::material_field(&model, &[("E", E), ("nu", NU)])?;

    // Traction S on the x = 1 face (QUA4 [1, 2, 6, 5]).
    let mut face = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    face.add_cell(&[nodes[1].id(), nodes[2].id(), nodes[6].id(), nodes[5].id()])?;
    let face_fes = FiniteElementSpace::lagrange1(&face)?;
    let traction = assemble::flux(&face_fes.get(0)?, FluxDensity::Uniform(S), "f_x")?;
    let rhs = NodeField::from_sub(traction);

    let solution = solve(&assemble::stiffness(&model, &materials)?, &rhs)?;

    let tol = 1e-10;
    for (i, c) in points.iter().enumerate() {
        let id = nodes[i].id();
        assert!((solution.value(id, "u_x")? - S / E * c[0]).abs() < tol);
        assert!((solution.value(id, "u_y")? + NU * S / E * c[1]).abs() < tol);
        assert!((solution.value(id, "u_z")? + NU * S / E * c[2]).abs() < tol);
    }
    Ok(())
}
