//! Embedded (immersed) constraint exercised end-to-end through the public API.
//!
//! A single HEX8 host carries linear heat conduction (`k = 1`); its eight
//! corners are pinned by Dirichlet to a linear field `T(x) = 1 + 2x + 3y + 4z`,
//! which the trilinear HEX8 interpolation reproduces exactly in the interior. An
//! immersed node sits inside the cube, tied to the host by an `Embedded`
//! constraint. The solved temperature at the immersed node must equal the host
//! interpolation there — i.e. the linear field evaluated at its coordinates.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::matrix::stiffness;
use pyrucast::ops::mesh::barycenter;
use pyrucast::ops::model;
use pyrucast::ops::node_field::FluxDensity;
use pyrucast::ops::solver::lu::solve;
use pyrucast::Result;

const TOL: f64 = 1e-9;

/// The linear field the host is pinned to.
fn field(c: &[f64]) -> f64 {
    1.0 + 2.0 * c[0] + 3.0 * c[1] + 4.0 * c[2]
}

#[test]
fn immersed_node_follows_host_interpolation() -> Result<()> {
    let coords = Handle::new(Coords::new(3)?);
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
    materials.add_sub(Handle::new(mat))?;

    let mut model = Model::empty();
    model.add_sub(Handle::new(SubModel::heat_conduction(sub)?))?;

    // Dirichlet pinning all eight corners to the linear field.
    let mut corner_sm = SubMesh::new(coords.clone(), ElementType::POI1);
    for n in &corners {
        corner_sm.add_cell(&[n.id()])?;
    }
    let corner_mesh = Mesh::from_submesh(corner_sm);
    let corner_mult = barycenter(&corner_mesh)?;
    let dir = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &corner_mesh,
        &corner_mult,
        None,
        None,
        Default::default(),
    )?;
    let dir_mult_nodes = dir.multiplier_nodes()?; // paired corner-for-corner
    model.add_sub(Handle::new(dir))?;

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
    model.add_sub(Handle::new(emb))?;

    // RHS: imposed_T = field(corner) at each Dirichlet multiplier, 0 (tie) at the
    // embedded multiplier — all under the shared component name "imposed_T".
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    for &m in &dir_mult_nodes {
        rhs_sm.add_cell(&[m])?;
    }
    rhs_sm.add_cell(&[emb_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
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
        assert!(
            (got - field(c)).abs() < TOL,
            "corner: {got} vs {}",
            field(c)
        );
    }
    let got = solution.value(p.id(), "T")?;
    let expected = field(&[0.3, 0.6, 0.2]); // 1 + 0.6 + 1.8 + 0.8 = 4.2
    assert!(
        (got - expected).abs() < TOL,
        "immersed node: got {got}, expected {expected}"
    );
    Ok(())
}

/// The motivating case: a bar node « baignée » in a 3-D **elasticity** volume,
/// tied in all three displacement components. Uniaxial tension of a unit HEX8
/// cube gives the linear field `u_x = (S/E)x`, `u_y = −(νS/E)y`,
/// `u_z = −(νS/E)z`; the immersed node must follow that field at its location —
/// a genuine vector (multi-component) embedded constraint.
#[test]
fn immersed_node_follows_host_displacement_field() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;
    const S: f64 = 2.0;

    let coords = Handle::new(Coords::new(3)?);
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
    let mut host = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
    host.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&host)?;

    // Symmetry rollers on the three faces through the origin.
    let clamp = |ids: &[usize], var: &str, dual: &str| -> Result<Model> {
        let picked: Vec<Node> = ids.iter().map(|&i| nodes[i].clone()).collect();
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(&picked)?);
        let mult = barycenter(&imposed)?;
        model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &mult,
            None,
            None,
            Default::default(),
        )
    };
    let mut model = model::elasticity(&fes, Kinematics::Full3D)?;
    model = model.union(&clamp(&[0, 3, 4, 7], "u_x", "f_x")?)?;
    model = model.union(&clamp(&[0, 1, 4, 5], "u_y", "f_y")?)?;
    model = model.union(&clamp(&[0, 1, 2, 3], "u_z", "f_z")?)?;

    // Immersed node tied in all three components (g = 0, rigid tie ⇒ no RHS slot).
    let pc = [0.4, 0.7, 0.2];
    let p = Node::create_in(coords.clone(), &pc)?;
    let bar = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&p))?);
    model = model.union(&model::embedded(
        &bar,
        &host,
        vec![
            ("u_x".into(), "f_x".into()),
            ("u_y".into(), "f_y".into()),
            ("u_z".into(), "f_z".into()),
        ],
        None,
        None,
        None,
    )?)?;

    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU)])?;

    // Traction S on the x = 1 face (QUA4 [1, 2, 6, 5]).
    let mut face = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    face.add_cell(&[nodes[1].id(), nodes[2].id(), nodes[6].id(), nodes[5].id()])?;
    let face_fes = FiniteElementSpace::lagrange1(&face)?;
    let rhs = pyrucast::ops::node_field::flux(&face_fes, FluxDensity::Uniform(S), "f_x")?;

    let solution = solve(&stiffness(&model, &materials)?, &rhs)?;

    // The immersed node follows the analytic (linear) displacement field.
    let tol = 1e-9;
    let ux = solution.value(p.id(), "u_x")?;
    let uy = solution.value(p.id(), "u_y")?;
    let uz = solution.value(p.id(), "u_z")?;
    assert!((ux - S / E * pc[0]).abs() < tol, "u_x: {ux}");
    assert!((uy + NU * S / E * pc[1]).abs() < tol, "u_y: {uy}");
    assert!((uz + NU * S / E * pc[2]).abs() < tol, "u_z: {uz}");
    Ok(())
}

/// Multi-component right-hand side: a **per-component** offset `g_c` at the
/// immersed node, `u_c(p) − Σᵢ Nᵢ·u_c(hostᵢ) = g_c`, routed through
/// `constraint_rhs_by_index` (relations are component-major: index
/// `c·n + r`). Each component must pick up its **own** slot, so the immersed
/// node reads `interpolation_c + g_c` — proof the per-relation imposed-value
/// slot works for a multi-dual constraint.
#[test]
fn embedded_per_component_offset() -> Result<()> {
    const E: f64 = 210.0;
    const NU: f64 = 0.3;
    const S: f64 = 2.0;
    let g = [0.01, -0.02, 0.03]; // offsets on u_x, u_y, u_z

    let coords = Handle::new(Coords::new(3)?);
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
    let mut host = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
    host.add_cell(&nodes.iter().map(|n| n.id()).collect::<Vec<_>>())?;
    let fes = FiniteElementSpace::lagrange1(&host)?;

    let clamp = |ids: &[usize], var: &str, dual: &str| -> Result<Model> {
        let picked: Vec<Node> = ids.iter().map(|&i| nodes[i].clone()).collect();
        let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(&picked)?);
        let mult = barycenter(&imposed)?;
        model::dirichlet(
            var.into(),
            dual.into(),
            &imposed,
            &mult,
            None,
            None,
            Default::default(),
        )
    };
    let mut model = model::elasticity(&fes, Kinematics::Full3D)?;
    model = model.union(&clamp(&[0, 3, 4, 7], "u_x", "f_x")?)?;
    model = model.union(&clamp(&[0, 1, 4, 5], "u_y", "f_y")?)?;
    model = model.union(&clamp(&[0, 1, 2, 3], "u_z", "f_z")?)?;

    let pc = [0.4, 0.7, 0.2];
    let p = Node::create_in(coords.clone(), &pc)?;
    let bar = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&p))?);
    let embedded = SubModel::embedded(
        &bar,
        &host,
        vec![
            ("u_x".into(), "f_x".into()),
            ("u_y".into(), "f_y".into()),
            ("u_z".into(), "f_z".into()),
        ],
        None,
        None,
        None,
    )?;
    // Per-component g via relation index (1 immersed node ⇒ index = component).
    let emb_rhs = embedded.constraint_rhs_by_index(&[(0, g[0]), (1, g[1]), (2, g[2])])?;
    let mut emb_model = Model::empty();
    emb_model.add_sub(Handle::new(embedded))?;
    model = model.union(&emb_model)?;

    let materials = pyrucast::ops::element_field::material_field(&model, &[("E", E), ("nu", NU)])?;

    let mut face = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    face.add_cell(&[nodes[1].id(), nodes[2].id(), nodes[6].id(), nodes[5].id()])?;
    let face_fes = FiniteElementSpace::lagrange1(&face)?;
    let mut rhs = pyrucast::ops::node_field::flux(&face_fes, FluxDensity::Uniform(S), "f_x")?;
    for sm in &emb_rhs {
        rhs.add_sub(sm.clone())?;
    }

    let solution = solve(&stiffness(&model, &materials)?, &rhs)?;

    // Immersed node = analytic interpolation + its own per-component offset.
    let tol = 1e-9;
    assert!((solution.value(p.id(), "u_x")? - (S / E * pc[0] + g[0])).abs() < tol);
    assert!((solution.value(p.id(), "u_y")? - (-NU * S / E * pc[1] + g[1])).abs() < tol);
    assert!((solution.value(p.id(), "u_z")? - (-NU * S / E * pc[2] + g[2])).abs() < tol);
    Ok(())
}
