//! Multi-point constraints (MPC) exercised end-to-end through the public API.
//!
//! A 1-D heat-conduction bar `-u'' = 0` on `[0, 1]` (SEG2 grid, `k = 1`) whose
//! analytical solution is linear. MPCs are imposed by Lagrange multipliers on the
//! same augmented system as Dirichlet, so a well-posed set of relations recovers
//! `u(x) = x` and the equivalence with Dirichlet holds for the single-term case.

use pyrucast::aggregate::Aggregate;
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::models::mpc::MpcTerm;
use pyrucast::ops::assemble::stiffness;
use pyrucast::ops::mesher::barycenter;
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::{insert, Handle};
use pyrucast::Result;

const N_ELEMS: usize = 4;
const H: f64 = 1.0 / N_ELEMS as f64;
const TOL: f64 = 1e-10;

/// Build the `[0, 1]` SEG2 bar with a `k = 1` heat-conduction sub-model already
/// added; returns the nodes, the base model and the material field.
fn heat_bar() -> Result<(Vec<Node>, Handle<Coords>, Model, ElementField)> {
    let coords = insert(Coords::new(1)?);
    let nodes: Vec<Node> = (0..=N_ELEMS)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * H]))
        .collect::<Result<_>>()?;

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
    for i in 0..N_ELEMS {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let sub: Handle<SubFiniteElementSpace> = fes.get(0)?;

    let mut mat = SubElementField::new(sub.clone(), vec!["k".into()])?;
    mat.set_uniform("k", 1.0)?;
    let mut materials = ElementField::empty();
    materials.add_sub(insert(mat))?;

    let mut model = Model::empty();
    model.add_sub(insert(SubModel::heat_conduction(sub)?))?;

    Ok((nodes, coords, model, materials))
}

/// A POI1 mesh carrying a single node.
fn poi1(node: &Node) -> Result<Mesh> {
    Ok(Mesh::from_submesh(SubMesh::poi1_from_nodes(
        std::slice::from_ref(node),
    )?))
}

/// A two-term difference relation `1·T(node4) − 1·T(node0) = 1`, combined with a
/// Dirichlet `T(node0) = 0`, must recover `u(x) = x` (so `T(node4) = 1`). This
/// exercises a genuine multi-term MPC with a non-zero right-hand side `g`.
#[test]
fn mpc_difference_relation_recovers_linear_solution() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    // Dirichlet T(node0) = 0.
    let imposed0 = poi1(&nodes[0])?;
    let mult0 = barycenter(&imposed0)?;
    let dir = SubModel::dirichlet("T".into(), "q".into(), &imposed0, &mult0, None, None)?;
    let dir_mult = dir.multiplier_nodes()?[0];
    model.add_sub(insert(dir))?;

    // MPC: T(node4) − T(node0) = 1. `dual_of` finds "q" for us.
    let dual = model.dual_of("T")?.expect("heat conduction declares T");
    assert_eq!(dual, "q");
    let mesh_last = poi1(&nodes[N_ELEMS])?;
    let mesh_first = poi1(&nodes[0])?;
    let mult_mpc_mesh = barycenter(&mesh_last)?;
    let terms = vec![
        MpcTerm::new(&mesh_last, "T".into(), dual.clone(), 1.0)?,
        MpcTerm::new(&mesh_first, "T".into(), dual, -1.0)?,
    ];
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None)?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(insert(mpc))?;

    // RHS: imposed_T = 0 at the Dirichlet multiplier, mpc_rhs = 1 (g) at the MPC one.
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = insert(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
    rhs.set_value(dir_mult, "imposed_T", 0.0)?;
    rhs.set_value(mpc_mult, "mpc_rhs", 1.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let solution = solve(&k, &rhs)?;

    for (i, node) in nodes.iter().enumerate() {
        let got = solution.value(node.id(), "T")?;
        let expected = i as f64 * H;
        assert!(
            (got - expected).abs() < TOL,
            "T at node {i}: got {got}, expected {expected}"
        );
    }
    // The relation itself holds exactly.
    let t_last = solution.value(nodes[N_ELEMS].id(), "T")?;
    let t_first = solution.value(nodes[0].id(), "T")?;
    assert!((t_last - t_first - 1.0).abs() < TOL);
    Ok(())
}

/// A single-term MPC `1·T = u_d` (coefficient 1, `g = u_d`) must produce exactly
/// the same solution as an equivalent Dirichlet — the MPC generalises Dirichlet.
#[test]
fn single_term_mpc_matches_dirichlet() -> Result<()> {
    // Reference: two Dirichlet conditions T(0) = 0, T(1) = 1.
    let solve_dirichlet = || -> Result<NodeField> {
        let (nodes, coords, mut model, materials) = heat_bar()?;
        let left = poi1(&nodes[0])?;
        let right = poi1(&nodes[N_ELEMS])?;
        let mult_l = barycenter(&left)?;
        let mult_r = barycenter(&right)?;
        let dl = SubModel::dirichlet("T".into(), "q".into(), &left, &mult_l, None, None)?;
        let dr = SubModel::dirichlet("T".into(), "q".into(), &right, &mult_r, None, None)?;
        let (nl, nr) = (dl.multiplier_nodes()?[0], dr.multiplier_nodes()?[0]);
        model.add_sub(insert(dl))?;
        model.add_sub(insert(dr))?;
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nr])?;
        let sm = insert(sm);
        let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into()])?;
        rhs.set_value(nl, "imposed_T", 0.0)?;
        rhs.set_value(nr, "imposed_T", 1.0)?;
        let k = stiffness(&model, &materials)?;
        solve(&k, &NodeField::from_sub(rhs))
    };

    // MPC variant: Dirichlet T(0) = 0 + single-term MPC 1·T(node4) = 1.
    let solve_mpc = || -> Result<NodeField> {
        let (nodes, coords, mut model, materials) = heat_bar()?;
        let left = poi1(&nodes[0])?;
        let mult_l = barycenter(&left)?;
        let dl = SubModel::dirichlet("T".into(), "q".into(), &left, &mult_l, None, None)?;
        let nl = dl.multiplier_nodes()?[0];
        model.add_sub(insert(dl))?;

        let right = poi1(&nodes[N_ELEMS])?;
        let mult_mpc = barycenter(&right)?;
        let terms = vec![MpcTerm::new(&right, "T".into(), "q".into(), 1.0)?];
        let mpc = SubModel::mpc(terms, &mult_mpc, None, None)?;
        let nm = mpc.multiplier_nodes()?[0];
        model.add_sub(insert(mpc))?;

        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nm])?;
        let sm = insert(sm);
        let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
        rhs.set_value(nl, "imposed_T", 0.0)?;
        rhs.set_value(nm, "mpc_rhs", 1.0)?;
        let k = stiffness(&model, &materials)?;
        solve(&k, &NodeField::from_sub(rhs))
    };

    let (nodes, ..) = heat_bar()?;
    let dir = solve_dirichlet()?;
    let mpc = solve_mpc()?;
    for node in &nodes {
        let a = dir.value(node.id(), "T")?;
        let b = mpc.value(node.id(), "T")?;
        assert!(
            (a - b).abs() < TOL,
            "node {:?}: dirichlet {a} vs mpc {b}",
            node.id()
        );
    }
    Ok(())
}
