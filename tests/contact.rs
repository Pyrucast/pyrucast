//! Node-to-surface contact exercised end-to-end through the public API —
//! the `Contact` sub-model + the active-set loop of `solve`.
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
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::models::Constraint;
use pyrucast::ops::element_field::StrainMeasure;
use pyrucast::ops::model;
use pyrucast::ops::solver;
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
    model::dirichlet(&imposed, target, var, Default::default())
}

/// The full two-block setup. Returns the two node grids, the assembled model
/// (elasticity + rollers + contact), the contact model object (for its gap and
/// multiplier reads) and the material field.
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

    let elasticite = model::elasticity(
        &fes,
        Kinematics::PlaneStress,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )?;
    let contact = model::contact(
        &slave,
        &master,
        &elasticite,
        vec!["u_x".into(), "u_y".into()],
    )?;

    // Uniaxial column: u_x = 0 everywhere, u_y = 0 on the bottom edge.
    let all_nodes: Vec<Node> = bottom.iter().chain(top.iter()).cloned().collect();
    let bottom_edge: Vec<Node> = (0..=N).map(|i| bottom[idx(i, 0)].clone()).collect();

    let mut model = model::elasticity(
        &fes,
        Kinematics::PlaneStress,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )?;
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
    // The pressure is a term of the model: it joins it, its density joins
    // the material, and we ask it for its contribution.
    let model = tb.model.union(&model::flux(&top_fes, &tb.model, "f_y")?)?;
    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", 0.0), ("phi_f_y", -S)],
    )?;
    // The pressure and the contact's own gap: every load the model carries.
    let rhs = pyrucast::ops::node_field::external_forces(&model, &materials)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solver::solve(&k, &rhs)?;

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
        let lambda = solution.value(*m, "lambda")?;
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
        .union(&pyrucast::ops::node_field::external_forces(
            &model, &materials,
        )?)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solver::solve(&k, &rhs)?;

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
        let lambda = solution.value(*m, "lambda")?;
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

    let elasticite = model::elasticity(
        &fes,
        Kinematics::Full3D,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )?;
    let contact = model::contact(
        &slave,
        &master,
        &elasticite,
        vec!["u_x".into(), "u_y".into(), "u_z".into()],
    )?;

    // Uniaxial column: u_x = u_y = 0 everywhere, u_z = 0 at the base.
    let all_nodes: Vec<Node> = bottom.iter().chain(top.iter()).cloned().collect();
    let mut model = model::elasticity(
        &fes,
        Kinematics::Full3D,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )?;
    model = model.union(&clamp(&model, &all_nodes, "u_x")?)?;
    model = model.union(&clamp(&model, &all_nodes, "u_y")?)?;
    model = model.union(&clamp(&model, &bottom[0..4], "u_z")?)?;
    model = model.union(&contact)?;
    // Pressure S downward on the top face of the top cube.
    let mut face = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    face.add_cell(&[top[4].id(), top[5].id(), top[6].id(), top[7].id()])?;
    let face_fes = FiniteElementSpace::lagrange1(&face)?;
    let model = model.union(&model::flux(&face_fes, &model, "f_z")?)?;

    let materials = pyrucast::ops::element_field::material_field(
        &model,
        &[("E", E), ("nu", 0.0), ("phi_f_z", -S)],
    )?;
    // The pressure and the contact's own gap: every load the model carries.
    let rhs = pyrucast::ops::node_field::external_forces(&model, &materials)?;

    let k = pyrucast::ops::matrix::stiffness(&model, &materials)?;
    let solution = solver::solve(&k, &rhs)?;

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
        .map(|m| -solution.value(*m, "lambda").unwrap())
        .sum();
    assert!((total - S).abs() < TOL);
    Ok(())
}

/// The contact's right-hand side is a load it carries: on its own,
/// `external_forces` gives back `−g₀` at each multiplier node — the two blocks
/// stand `G0` apart.
#[test]
fn the_contact_carries_its_gap() -> Result<()> {
    let tb = two_blocks()?;
    let materials =
        pyrucast::ops::element_field::material_field(&tb.model, &[("E", E), ("nu", 0.0)])?;
    let gaps = pyrucast::ops::node_field::external_forces(&tb.contact, &materials)?;
    let mult = tb.contact.multiplier_mesh()?;
    for r in 0..=N {
        let m = mult.node(0, r, 0)?.id();
        assert!((gaps.value(m, "imposed")? + G0).abs() < TOL);
    }
    Ok(())
}

// ─── Updating the contact to a displaced configuration ──────────────────────

/// A master polyline whose facets turn — A(0,0), B(1,0), C(2,0.5), D(3,1.5),
/// run right to left so the normals point up — two slave nodes above it, and
/// the bar model the contact couples into. Returns the nodes (A, B, C, D, then
/// the slaves) and the contact model.
fn turning_floor() -> Result<(Vec<Node>, Model)> {
    let coords = Handle::new(Coords::new(2)?);
    let at = |x: f64, y: f64| Node::create_in(coords.clone(), &[x, y]);
    let nodes = vec![
        at(0.0, 0.0)?,
        at(1.0, 0.0)?,
        at(2.0, 0.5)?,
        at(3.0, 1.5)?,
        at(0.6, 0.8)?,
        at(1.4, 0.9)?,
    ];
    let mut floor = SubMesh::new(coords.clone(), ElementType::SEG2);
    for i in (0..3).rev() {
        floor.add_cell(&[nodes[i + 1].id(), nodes[i].id()])?;
    }
    let master = Mesh::from_submesh(floor);
    let slave = Mesh::from_submesh(SubMesh::poi1_from_nodes(&nodes[4..])?);
    let mut bars = SubMesh::new(coords, ElementType::SEG2);
    bars.add_cell(&[nodes[4].id(), nodes[5].id()])?;
    let target = model::truss(
        &FiniteElementSpace::lagrange1(&Mesh::from_submesh(bars))?,
        StrainMeasure::Linear,
    )?;
    let contact = model::contact(&slave, &master, &target, vec!["u_x".into(), "u_y".into()])?;
    Ok((nodes, contact))
}

/// The relations and gaps of the one contact sub-model of `m`.
fn relations_and_gaps(m: &Model) -> Result<(Vec<pyrucast::models::Relation>, Vec<f64>)> {
    let h = m.get(0)?;
    let sub = h.read();
    let pyrucast::containers::model::SubModel::Contact(c) = &*sub else {
        panic!("a contact sub-model");
    };
    Ok((c.relations()?, c.gaps()))
}

/// At `u = 0`, updating gives back the contact as it was built: same terms,
/// same coefficients, same gaps, the same multiplier nodes (the same handles).
#[test]
fn updating_at_rest_gives_the_contact_back() -> Result<()> {
    let (_, contact) = turning_floor()?;
    let rest = model::update_contact(&contact, &NodeField::empty());
    let (built, g0) = relations_and_gaps(&contact)?;
    let (again, g) = relations_and_gaps(&rest)?;
    assert_eq!(g, g0);
    for (a, b) in built.iter().zip(&again) {
        assert_eq!(a.multiplier_node, b.multiplier_node);
        let coefficients = |r: &pyrucast::models::Relation| -> Vec<_> {
            r.terms
                .iter()
                .map(|t| (t.node, t.variable.clone(), t.coefficient))
                .collect()
        };
        assert_eq!(coefficients(a), coefficients(b));
    }
    assert!(Handle::same_object(
        &contact.multiplier_mesh()?.get(0)?,
        &rest.multiplier_mesh()?.get(0)?
    ));
    Ok(())
}

/// The independent oracle: lay the positions `X + u` in a configuration of
/// their own and project there. For any `u` — not a rigid motion — the updated
/// relation must carry the normal and the weights of that projection, and its
/// gap must be the one measured there: `g̃ + C·u = g(X + u)`.
#[test]
fn the_updated_relation_is_the_gap_linearised_at_x_plus_u() -> Result<()> {
    let (nodes, contact) = turning_floor()?;
    let support = Mesh::from_submesh(SubMesh::poi1_from_nodes(&nodes)?);
    let u = NodeField::new(&support, vec!["u_x".into(), "u_y".into()])?;
    let shifts = [
        (0.02, -0.01),
        (-0.03, 0.04),
        (0.05, 0.10),
        (0.00, -0.02),
        (0.55, -0.45),
        (0.30, -0.20),
    ];
    for (node, (ux, uy)) in nodes.iter().zip(shifts) {
        u.get(0)?.write().set_value(node.id(), "u_x", ux)?;
        u.get(0)?.write().set_value(node.id(), "u_y", uy)?;
    }
    let updated = model::update_contact(&contact, &u);
    let (relations, gaps) = relations_and_gaps(&updated)?;

    // The oracle: the projection on a configuration laid at X + u.
    let coords = nodes[0].coords();
    let displaced = coords.write().add_config("displaced");
    pyrucast::ops::coords::displace_from(&u, 0, displaced, Some(vec!["u_x".into(), "u_y".into()]))?;
    let master = {
        let mut floor = SubMesh::new(coords.clone(), ElementType::SEG2);
        for i in (0..3).rev() {
            floor.add_cell(&[nodes[i + 1].id(), nodes[i].id()])?;
        }
        Mesh::from_submesh(floor)
    };
    let projections = pyrucast::ops::coords::in_config(&coords, displaced, || {
        let points: Vec<Vec<f64>> = nodes[4..]
            .iter()
            .map(|n| coords.read().position(n.id()).map(|p| p.to_vec()))
            .collect::<Result<_>>()?;
        pyrucast::ops::geom::project_points(&master, &points)
    })?;

    for ((relation, &g), p) in relations.iter().zip(&gaps).zip(&projections) {
        // C·u, from the relation's own terms.
        let mut cu = 0.0;
        for t in &relation.terms {
            cu += t.coefficient * u.value_opt(t.node, &t.variable).unwrap_or(0.0);
        }
        assert!(
            (g + cu - p.gap).abs() < 1e-12,
            "g̃ + C·u = {} vs {}",
            g + cu,
            p.gap
        );
        // The slave carries +n, each master node −Nᵢ·n.
        let slave = relation.terms[0].node;
        for (a, variable) in ["u_x", "u_y"].iter().enumerate() {
            let on = |node| -> f64 {
                relation
                    .terms
                    .iter()
                    .filter(|t| t.node == node && t.variable == *variable)
                    .map(|t| t.coefficient)
                    .sum()
            };
            assert!((on(slave) - p.normal[a]).abs() < 1e-12);
            for (&m, &w) in p.nodes.iter().zip(&p.weights) {
                assert!((on(m) + w * p.normal[a]).abs() < 1e-12);
            }
        }
    }
    // Not vacuous: the second slave changed facet, and its gap changed.
    let (_, g0) = relations_and_gaps(&contact)?;
    assert!((gaps[1] - g0[1]).abs() > 1e-3);
    Ok(())
}

/// A slave that slides past the end of its facet is paired with the next one —
/// nodes and weights computed by hand — and keeps its multiplier node.
#[test]
fn a_sliding_slave_moves_to_the_next_facet() -> Result<()> {
    let (nodes, contact) = turning_floor()?;
    let support = Mesh::from_submesh(SubMesh::poi1_from_nodes(&nodes[4..5])?);
    let u = NodeField::new(&support, vec!["u_x".into(), "u_y".into()])?;
    // The first slave, at (0.6, 0.8) above A–B, slides to (1.2, 0.3): its
    // closest point is now on B–C.
    u.get(0)?.write().set_value(nodes[4].id(), "u_x", 0.6)?;
    u.get(0)?.write().set_value(nodes[4].id(), "u_y", -0.5)?;
    let updated = model::update_contact(&contact, &u);
    let (before, _) = relations_and_gaps(&contact)?;
    let (after, _) = relations_and_gaps(&updated)?;
    let masters = |r: &pyrucast::models::Relation| -> Vec<NodeId> {
        let slave = r.terms[0].node;
        let mut m: Vec<NodeId> = r
            .terms
            .iter()
            .map(|t| t.node)
            .filter(|&n| n != slave)
            .collect();
        m.sort();
        m.dedup();
        m
    };
    let (a, b, c) = (nodes[0].id(), nodes[1].id(), nodes[2].id());
    assert_eq!(masters(&before[0]), {
        let mut v = vec![b, a];
        v.sort();
        v
    });
    assert_eq!(masters(&after[0]), {
        let mut v = vec![c, b];
        v.sort();
        v
    });
    assert_eq!(after[0].multiplier_node, before[0].multiplier_node);
    // By hand: B–C runs along t = (1, 0.5)/√1.25; the point (1.2, 0.3) projects
    // at s = (0.2·1 + 0.3·0.5)/1.25 = 0.28 of the way from B, so N_B = 0.72,
    // N_C = 0.28, and the normal is (−0.5, 1)/√1.25.
    let norm = 1.25f64.sqrt();
    let (n_x, n_y) = (-0.5 / norm, 1.0 / norm);
    let on = |node, variable: &str| -> f64 {
        after[0]
            .terms
            .iter()
            .filter(|t| t.node == node && t.variable == variable)
            .map(|t| t.coefficient)
            .sum()
    };
    assert!((on(b, "u_y") + 0.72 * n_y).abs() < 1e-9);
    assert!((on(c, "u_y") + 0.28 * n_y).abs() < 1e-9);
    assert!((on(c, "u_x") + 0.28 * n_x).abs() < 1e-9);
    Ok(())
}

/// A model without contact comes back as it was: the same sub-models.
#[test]
fn a_model_without_contact_is_unchanged() -> Result<()> {
    let (nodes, _) = turning_floor()?;
    let mut bars = SubMesh::new(nodes[0].coords(), ElementType::SEG2);
    bars.add_cell(&[nodes[4].id(), nodes[5].id()])?;
    let bar = model::truss(
        &FiniteElementSpace::lagrange1(&Mesh::from_submesh(bars))?,
        StrainMeasure::Linear,
    )?;
    let same = model::update_contact(&bar, &NodeField::empty());
    assert_eq!(same.len(), bar.len());
    assert!(Handle::same_object(&same.get(0)?, &bar.get(0)?));
    Ok(())
}
