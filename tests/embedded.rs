//! Embedded (immersed) constraint exercised end-to-end through the public API.
//!
//! A single HEX8 host carries linear heat conduction (`k = 1`); its eight
//! corners are pinned by Dirichlet to a linear field `T(x) = 1 + 2x + 3y + 4z`,
//! which the trilinear HEX8 interpolation reproduces exactly in the interior. An
//! immersed node sits inside the cube, tied to the host by an `Embedded`
//! constraint. The solved temperature at the immersed node must equal the host
//! interpolation there — i.e. the linear field evaluated at its coordinates.

use pyrucast::aggregate::Aggregate;
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::ops::assemble::stiffness;
use pyrucast::ops::mesher::barycenter;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::{insert, Handle};
use pyrucast::Result;

const TOL: f64 = 1e-9;

/// The linear field the host is pinned to.
fn field(c: &[f64]) -> f64 {
    1.0 + 2.0 * c[0] + 3.0 * c[1] + 4.0 * c[2]
}

#[test]
fn immersed_node_follows_host_interpolation() -> Result<()> {
    let coords = insert(Coords::new(3)?);
    let corner_coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let corners: Vec<Node> = corner_coords
        .iter()
        .map(|c| Node::create_in(coords.clone(), c))
        .collect::<Result<_>>()?;

    // HEX8 host with linear heat conduction (k = 1).
    let mut host = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
    host.add_cell(&corners.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&host)?;
    let sub: Handle<SubFiniteElementSpace> = fes.get(0)?;
    let mut mat = SubElementField::new(sub.clone(), vec!["k".into()])?;
    mat.set_uniform("k", 1.0)?;
    let mut materials = ElementField::empty();
    materials.add_sub(insert(mat))?;

    let mut model = Model::empty();
    model.add_sub(insert(SubModel::heat_conduction(sub)?))?;

    // Dirichlet pinning all eight corners to the linear field.
    let mut corner_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    for n in &corners {
        corner_sm.add_cell(&[n.id()])?;
    }
    let corner_mesh = Mesh::from_submesh(corner_sm);
    let corner_mult = barycenter(&corner_mesh)?;
    let dir = SubModel::dirichlet("T".into(), "q".into(), &corner_mesh, &corner_mult, None, None)?;
    let dir_mult_nodes = dir.multiplier_nodes()?; // paired corner-for-corner
    model.add_sub(insert(dir))?;

    // Immersed node inside the cube, tied to the host by an Embedded constraint.
    let p = Node::create_in(coords.clone(), &[0.3, 0.6, 0.2])?;
    let mut bar_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    bar_sm.add_cell(&[p.id()])?;
    let bar = Mesh::from_submesh(bar_sm);
    let emb = SubModel::embedded(
        &bar,
        &host,
        vec![("T".into(), "q".into())],
        None,
        None,
        None,
    )?;
    let emb_mult = emb.multiplier_nodes()?[0];
    model.add_sub(insert(emb))?;

    // RHS: imposed_T = field(corner) at each Dirichlet multiplier, 0 (tie) at the
    // embedded multiplier — all under the shared component name "imposed_T".
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    for &m in &dir_mult_nodes {
        rhs_sm.add_cell(&[m])?;
    }
    rhs_sm.add_cell(&[emb_mult])?;
    let rhs_sm = insert(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into()])?;
    for (m, c) in dir_mult_nodes.iter().zip(corner_coords.iter()) {
        rhs.set_value(*m, "imposed_T", field(c))?;
    }
    rhs.set_value(emb_mult, "imposed_T", 0.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let solution = solve(&k, &rhs)?;

    // Corners recover the linear field (sanity), and the immersed node too.
    for (n, c) in corners.iter().zip(corner_coords.iter()) {
        let got = solution.value(n.id(), "T")?;
        assert!((got - field(c)).abs() < TOL, "corner: {got} vs {}", field(c));
    }
    let got = solution.value(p.id(), "T")?;
    let expected = field(&[0.3, 0.6, 0.2]); // 1 + 0.6 + 1.8 + 0.8 = 4.2
    assert!(
        (got - expected).abs() < TOL,
        "immersed node: got {got}, expected {expected}"
    );
    Ok(())
}
