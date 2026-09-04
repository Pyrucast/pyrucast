//! Node-to-surface contact exercised end-to-end through the public API —
//! the `Contact` sub-model + the active-set solver.
//!
//! **Patch test** (2-D, plane stress, `ν = 0`): two elastic blocks stacked in
//! `y` with an initial gap `g₀`, every `u_x` blocked (uniaxial column). A
//! pressure `S` on top closes the contact and must transmit a **uniform**
//! stress `σ_yy = −S` through the interface: displacements are piecewise
//! linear in `y` (the top block additionally translated by `−g₀`) and the
//! contact reactions are the consistent nodal forces of the pressure. Lifting
//! the top block instead opens every pair (`λ = 0`, bottom block untouched).

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::models::Physics;
use pyrucast::ops::mesh;
use pyrucast::ops::model;
use pyrucast::ops::solver::unilateral;
use pyrucast::Result;

const E: f64 = 100.0;
const S: f64 = 5.0; // applied pressure
const G0: f64 = 0.01; // initial gap between the blocks
const N: usize = 2; // N×N QUA4 grid per block
const TOL: f64 = 1e-9;

/// An N×N QUA4 block `[0,1] × [y0, y0+1]` in `coords`; returns its node grid.
fn block(coords: &Handle<Coords>, y0: f64) -> Result<(Vec<Node>, SubMesh)> {
    let h = 1.0 / N as f64;
    let mut grid: Vec<Node> = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            grid.push(Node::create_in(
                coords.clone(),
                &[i as f64 * h, y0 + j as f64 * h],
            )?);
        }
    }
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
    for j in 0..N {
        for i in 0..N {
            sm.add_cell(&[
                grid[idx(i, j)].id(),
                grid[idx(i + 1, j)].id(),
                grid[idx(i + 1, j + 1)].id(),
                grid[idx(i, j + 1)].id(),
            ])?;
        }
    }
    Ok((grid, sm))
}

/// Homogeneous Dirichlet `var = 0` on every node of `nodes`.
fn clamp(target: &Model, nodes: &[Node], var: &str) -> Result<Model> {
    let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?);
    let multiplier = mesh::barycenter(&imposed)?;
    model::dirichlet(target, var, &imposed, &multiplier, Default::default())
}

/// The full two-block setup. Returns the two node grids, the assembled model
/// (elasticity + rollers + contact), the contact model object (for
/// `contact_gaps` / multiplier reads) and the material field.
struct TwoBlocks {
    bottom: Vec<Node>,
    top: Vec<Node>,
    model: Model,
    contact: Model,
    slave_nodes: Vec<NodeId>,
}

fn two_blocks() -> Result<TwoBlocks> {
    let coords = Handle::new(Coords::new(2)?);
    let (bottom, bottom_sm) = block(&coords, 0.0)?;
    let (top, top_sm) = block(&coords, 1.0 + G0)?;
    let mut mesh = Mesh::from_submesh(bottom_sm);
    mesh.add_sub(Handle::new(top_sm))?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    let idx = |i: usize, j: usize| j * (N + 1) + i;

    // Master: the top edge of the bottom block, run in the −x direction so the
    // facet normal (t_y, −t_x) points +y, toward the slave (top) block.
    let mut master = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in (0..N).rev() {
        master.add_cell(&[bottom[idx(i + 1, N)].id(), bottom[idx(i, N)].id()])?;
    }
    // Slave: the bottom edge nodes of the top block.
    let slave_nodes_v: Vec<Node> = (0..=N).map(|i| top[idx(i, 0)].clone()).collect();
    let slave = Mesh::from_submesh(SubMesh::poi1_from_nodes(&slave_nodes_v)?);

    let elasticite = model::elasticity(&fes, Kinematics::PlaneStress)?;
    let contact = model::contact(
        &elasticite,
        &slave,
        &master,
        vec!["u_x".into(), "u_y".into()],
    )?;

    // Uniaxial column: u_x = 0 everywhere, u_y = 0 on the bottom edge.
    let all_nodes: Vec<Node> = bottom.iter().chain(top.iter()).cloned().collect();
    let bottom_edge: Vec<Node> = (0..=N).map(|i| bottom[idx(i, 0)].clone()).collect();

    let mut model = model::elasticity(&fes, Kinematics::PlaneStress)?;
    model = model.union(&clamp(&model, &all_nodes, "u_x")?)?;
    model = model.union(&clamp(&model, &bottom_edge, "u_y")?)?;
    model = model.union(&contact)?;

    let slave_nodes = slave_nodes_v.iter().map(|n| n.id()).collect();

    Ok(TwoBlocks {
        bottom,
        top,
        model,
        contact,
        slave_nodes,
    })
}

/// Pressing the top block down transmits the uniform pressure exactly through
/// the closed contact (patch test), reactions included.
#[test]
fn patch_test_uniform_pressure_through_contact() -> Result<()> {
    let tb = two_blocks()?;
    let coords_h = tb.bottom[0].coords();
    let idx = |i: usize, j: usize| j * (N + 1) + i;
    let h = 1.0 / N as f64;

    // Pressure S downward on the top edge of the top block.
    let mut top_edge = Mesh::from_submesh(SubMesh::new(coords_h.clone(), ElementType::SEG2));
    for i in 0..N {
        top_edge.add_cell(&[tb.top[idx(i, N)].id(), tb.top[idx(i + 1, N)].id()])?;
    }
    let top_fes = FiniteElementSpace::lagrange1(&top_edge)?;
    // La pression est un terme du modèle : elle le rejoint, sa densité rejoint
    // le matériau, et on lui demande sa contribution.
    let model = tb
        .model
        .union(&model::flux(&top_fes, "f_y".into(), Physics::Mechanical)?)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", 0.0), ("phi_f_y", -S)],
    )?;
    let traction = pyrucast::ops::node_field::external_forces(&model, &materials)?;
    let rhs = traction.union(&model.contact_gaps()?)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = unilateral::solve(&k, &model, &rhs)?;

    // Bottom block: u_y = −(S/E)·y; top block: −(S/E)·y shifted by the closed
    // gap (its own coordinates start at 1 + g₀ but its base lands at the
    // compressed interface).
    for j in 0..=N {
        for i in 0..=N {
            let y = j as f64 * h;
            let got = solution.value(tb.bottom[idx(i, j)].id(), "u_y")?;
            let expected = -S / E * y;
            assert!(
                (got - expected).abs() < TOL,
                "bottom u_y({y}) = {got}, expected {expected}"
            );
            let got = solution.value(tb.top[idx(i, j)].id(), "u_y")?;
            let expected = -S / E * (1.0 + y) - G0;
            assert!(
                (got - expected).abs() < TOL,
                "top u_y = {got}, expected {expected}"
            );
        }
    }

    // Contact reactions: −λᵢ are the consistent nodal forces of the pressure
    // (S·h/2 at the corners, S·h inside), so Σ(−λᵢ) = S.
    let mults: Vec<NodeId> = {
        let mut v = Vec::new();
        for h in &tb.contact {
            v.extend(h.read().multiplier_nodes());
        }
        v
    };
    let mut total = 0.0;
    for (r, m) in mults.iter().enumerate() {
        let lambda = solution.value(*m, "lambda_contact")?;
        assert!(
            lambda <= TOL,
            "active ≥ relation {r} must carry λ ≤ 0, got {lambda}"
        );
        total += -lambda;
    }
    assert!(
        (total - S).abs() < TOL,
        "Σ reactions = {total}, expected {S}"
    );
    Ok(())
}

/// Lifting the top block opens every pair: λ = 0 everywhere, the gap grows,
/// and the bottom block stays untouched.
#[test]
fn separation_releases_every_pair() -> Result<()> {
    let tb = two_blocks()?;
    let idx = |i: usize, j: usize| j * (N + 1) + i;

    // Impose u_y = +0.1 on the top edge of the top block (lift it).
    let lift = 0.1;
    let top_edge_nodes: Vec<Node> = (0..=N).map(|i| tb.top[idx(i, N)].clone()).collect();
    let lift_model = clamp(&tb.model, &top_edge_nodes, "u_y")?;
    let model = tb.model.union(&lift_model)?;
    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", 0.0)])?;

    let rhs = lift_model
        .constraint_rhs(
            &top_edge_nodes
                .iter()
                .map(|n| (n.id(), lift))
                .collect::<Vec<_>>(),
        )?
        .union(&model.contact_gaps()?)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = unilateral::solve(&k, &model, &rhs)?;

    // Top block translates rigidly; bottom block stays put.
    for j in 0..=N {
        for i in 0..=N {
            let got = solution.value(tb.top[idx(i, j)].id(), "u_y")?;
            assert!((got - lift).abs() < TOL, "top u_y = {got}, expected {lift}");
            let got = solution.value(tb.bottom[idx(i, j)].id(), "u_y")?;
            assert!(got.abs() < TOL, "bottom u_y = {got}, expected 0");
        }
    }
    // Every contact multiplier is exactly zero (released).
    let mults: Vec<NodeId> = {
        let mut v = Vec::new();
        for h in &tb.contact {
            v.extend(h.read().multiplier_nodes());
        }
        v
    };
    for m in &mults {
        let lambda = solution.value(*m, "lambda_contact")?;
        assert!(lambda.abs() < TOL, "released pair must carry λ = 0");
    }
    // And the slave nodes moved away from the master (gap opened).
    for s in &tb.slave_nodes {
        assert!(solution.value(*s, "u_y")? > 0.0);
    }
    Ok(())
}

/// 3-D: two stacked HEX8 cubes touching (`g₀ = 0`), pressure on top — uniform
/// `σ_zz = −S` transmitted, `u_z = −(S/E)·z` continuous across the interface.
#[test]
fn contact_3d_two_cubes() -> Result<()> {
    let coords = Handle::new(Coords::new(3)?);
    let cube = |z0: f64| -> Result<(Vec<Node>, SubMesh)> {
        let pts = [
            [0.0, 0.0, z0],
            [1.0, 0.0, z0],
            [1.0, 1.0, z0],
            [0.0, 1.0, z0],
            [0.0, 0.0, z0 + 1.0],
            [1.0, 0.0, z0 + 1.0],
            [1.0, 1.0, z0 + 1.0],
            [0.0, 1.0, z0 + 1.0],
        ];
        let nodes: Vec<Node> = pts
            .iter()
            .map(|c| Node::create_in(coords.clone(), c))
            .collect::<Result<_>>()?;
        let mut sm = SubMesh::new(coords.clone(), ElementType::HEX8);
        sm.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
        Ok((nodes, sm))
    };
    let (bottom, bottom_sm) = cube(0.0)?;
    let (top, top_sm) = cube(1.0)?;
    let mut mesh = Mesh::from_submesh(bottom_sm);
    mesh.add_sub(Handle::new(top_sm))?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;

    // Master: top face of the bottom cube ([4,5,6,7] is CCW seen from +z, so
    // the QUA4 normal points +z, toward the slave cube).
    let mut master = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    master.add_cell(&[
        bottom[4].id(),
        bottom[5].id(),
        bottom[6].id(),
        bottom[7].id(),
    ])?;
    // Slave: bottom face nodes of the top cube.
    let slave_nodes: Vec<Node> = top[0..4].to_vec();
    let slave = Mesh::from_submesh(SubMesh::poi1_from_nodes(&slave_nodes)?);

    let elasticite = model::elasticity(&fes, Kinematics::Full3D)?;
    let contact = model::contact(
        &elasticite,
        &slave,
        &master,
        vec!["u_x".into(), "u_y".into(), "u_z".into()],
    )?;

    // Uniaxial column: u_x = u_y = 0 everywhere, u_z = 0 at the base.
    let all_nodes: Vec<Node> = bottom.iter().chain(top.iter()).cloned().collect();
    let mut model = model::elasticity(&fes, Kinematics::Full3D)?;
    model = model.union(&clamp(&model, &all_nodes, "u_x")?)?;
    model = model.union(&clamp(&model, &all_nodes, "u_y")?)?;
    model = model.union(&clamp(&model, &bottom[0..4], "u_z")?)?;
    model = model.union(&contact)?;
    // Pressure S downward on the top face of the top cube.
    let mut face = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    face.add_cell(&[top[4].id(), top[5].id(), top[6].id(), top[7].id()])?;
    let face_fes = FiniteElementSpace::lagrange1(&face)?;
    let model = model.union(&model::flux(&face_fes, "f_z".into(), Physics::Mechanical)?)?;

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", 0.0), ("phi_f_z", -S)],
    )?;
    let traction = pyrucast::ops::node_field::external_forces(&model, &materials)?;
    let rhs = traction.union(&model.contact_gaps()?)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = unilateral::solve(&k, &model, &rhs)?;

    // u_z = −(S/E)·z on both cubes (touching: no gap offset).
    for (nodes, z0) in [(&bottom, 0.0), (&top, 1.0)] {
        for (i, n) in nodes.iter().enumerate() {
            let z = z0 + if i < 4 { 0.0 } else { 1.0 };
            let got = solution.value(n.id(), "u_z")?;
            let expected = -S / E * z;
            assert!(
                (got - expected).abs() < TOL,
                "u_z(z={z}) = {got}, expected {expected}"
            );
        }
    }
    // Total contact reaction = S · area = S.
    let mults: Vec<NodeId> = {
        let mut v = Vec::new();
        for h in &contact {
            v.extend(h.read().multiplier_nodes());
        }
        v
    };
    let total: f64 = mults
        .iter()
        .map(|m| -solution.value(*m, "lambda_contact").unwrap())
        .sum();
    assert!((total - S).abs() < TOL);
    Ok(())
}
