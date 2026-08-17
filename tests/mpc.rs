//! Multi-point constraints (MPC) exercised end-to-end through the public API.
//!
//! A 1-D heat-conduction bar `-u'' = 0` on `[0, 1]` (SEG2 grid, `k = 1`) whose
//! analytical solution is linear. MPCs are imposed by Lagrange multipliers on the
//! same augmented system as Dirichlet, so a well-posed set of relations recovers
//! `u(x) = x` and the equivalence with Dirichlet holds for the single-term case.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use pyrucast::containers::matrix::{DofOrdering, SubMatrix};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::mpc::MpcTerm;
use pyrucast::ops::matrix::stiffness;
use pyrucast::ops::mesh::barycenter;
use pyrucast::ops::solver::eliminate::{self, Condensation};
use pyrucast::ops::solver::lu::solve;
use pyrucast::store::Handle;
use pyrucast::Result;

const N_ELEMS: usize = 4;
const H: f64 = 1.0 / N_ELEMS as f64;
const TOL: f64 = 1e-10;

/// Build the `[0, 1]` SEG2 bar with a `k = 1` heat-conduction sub-model already
/// added; returns the nodes, the base model and the material field.
fn heat_bar() -> Result<(Vec<Node>, Handle<Coords>, Model, ElementField)> {
    let coords = Handle::new(Coords::new(1)?);
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
    materials.add_sub(Handle::new(mat))?;

    let mut model = Model::empty();
    model.add_sub(Handle::new(SubModel::heat_conduction(sub)?))?;

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
    let dir = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &imposed0,
        &mult0,
        None,
        None,
        Default::default(),
    )?;
    let dir_mult = dir.multiplier_nodes()?[0];
    model.add_sub(Handle::new(dir))?;

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
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None, Default::default())?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    // RHS: imposed_T = 0 at the Dirichlet multiplier, mpc_rhs = 1 (g) at the MPC one.
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
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
        let dl = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &left,
            &mult_l,
            None,
            None,
            Default::default(),
        )?;
        let dr = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &right,
            &mult_r,
            None,
            None,
            Default::default(),
        )?;
        let (nl, nr) = (dl.multiplier_nodes()?[0], dr.multiplier_nodes()?[0]);
        model.add_sub(Handle::new(dl))?;
        model.add_sub(Handle::new(dr))?;
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nr])?;
        let sm = Handle::new(sm);
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
        let dl = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &left,
            &mult_l,
            None,
            None,
            Default::default(),
        )?;
        let nl = dl.multiplier_nodes()?[0];
        model.add_sub(Handle::new(dl))?;

        let right = poi1(&nodes[N_ELEMS])?;
        let mult_mpc = barycenter(&right)?;
        let terms = vec![MpcTerm::new(&right, "T".into(), "q".into(), 1.0)?];
        let mpc = SubModel::mpc(terms, &mult_mpc, None, None, Default::default())?;
        let nm = mpc.multiplier_nodes()?[0];
        model.add_sub(Handle::new(mpc))?;

        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nm])?;
        let sm = Handle::new(sm);
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

/// The **elimination** solver must reproduce the Lagrange solution on a genuine
/// two-term MPC combined with a Dirichlet — with disjoint slaves (non-chained,
/// the v1 scope). `2·T(node4) − 1·T(node2) = 1.5` (slave node4, master node2)
/// plus `T(node0) = 0` (slave node0). Both methods solve the same constrained
/// energy minimisation, so their fields coincide node-for-node and the relation
/// holds exactly. (The field is *not* `u = x`: a multi-term relation injects
/// reactions at both term nodes, so the minimiser bends.)
#[test]
fn mpc_elimination_matches_lagrange() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    // Dirichlet T(node0) = 0.
    let imposed0 = poi1(&nodes[0])?;
    let mult0 = barycenter(&imposed0)?;
    let dir = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &imposed0,
        &mult0,
        None,
        None,
        Default::default(),
    )?;
    let dir_mult = dir.multiplier_nodes()?[0];
    model.add_sub(Handle::new(dir))?;

    // MPC: 2·T(node4) − 1·T(node2) = 1.5 (slave = node4, master = node2).
    let dual = model.dual_of("T")?.expect("heat conduction declares T");
    let mesh4 = poi1(&nodes[N_ELEMS])?;
    let mesh2 = poi1(&nodes[N_ELEMS / 2])?;
    let mult_mpc_mesh = barycenter(&mesh4)?;
    let terms = vec![
        MpcTerm::new(&mesh4, "T".into(), dual.clone(), 2.0)?,
        MpcTerm::new(&mesh2, "T".into(), dual, -1.0)?,
    ];
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None, Default::default())?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    // RHS: imposed_T = 0, mpc_rhs = 1.5 (= 2·1 − 1·0.5).
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
    rhs.set_value(dir_mult, "imposed_T", 0.0)?;
    rhs.set_value(mpc_mult, "mpc_rhs", 1.5)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let lagrange = solve(&k, &rhs)?;
    let elim = eliminate::solve(&k, &model, &rhs)?;

    // The two methods produce the same field, node-for-node.
    for (i, node) in nodes.iter().enumerate() {
        let a = lagrange.value(node.id(), "T")?;
        let b = elim.value(node.id(), "T")?;
        assert!(
            (a - b).abs() < TOL,
            "node {i}: lagrange {a} vs elimination {b}"
        );
    }
    // The Dirichlet and the MPC relation hold exactly on the elimination field.
    let t0 = elim.value(nodes[0].id(), "T")?;
    let t2 = elim.value(nodes[N_ELEMS / 2].id(), "T")?;
    let t4 = elim.value(nodes[N_ELEMS].id(), "T")?;
    assert!(t0.abs() < TOL, "Dirichlet T(0) = {t0}");
    assert!(
        (2.0 * t4 - t2 - 1.5).abs() < TOL,
        "relation: 2·{t4} − {t2} ≠ 1.5"
    );
    Ok(())
}

/// A single-term MPC solved by **elimination** matches the equivalent Dirichlet:
/// `1·T(node4) = 1` (with Dirichlet `T(node0) = 0`) gives the same field as two
/// Dirichlet conditions. The single-term relation is `a_s = 1`, masters = ∅.
#[test]
fn single_term_mpc_elimination_equals_dirichlet() -> Result<()> {
    // Reference: two Dirichlet T(0) = 0, T(1) = 1 (Lagrange).
    let reference = || -> Result<NodeField> {
        let (nodes, coords, mut model, materials) = heat_bar()?;
        let left = poi1(&nodes[0])?;
        let right = poi1(&nodes[N_ELEMS])?;
        let mult_l = barycenter(&left)?;
        let mult_r = barycenter(&right)?;
        let dl = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &left,
            &mult_l,
            None,
            None,
            Default::default(),
        )?;
        let dr = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &right,
            &mult_r,
            None,
            None,
            Default::default(),
        )?;
        let (nl, nr) = (dl.multiplier_nodes()?[0], dr.multiplier_nodes()?[0]);
        model.add_sub(Handle::new(dl))?;
        model.add_sub(Handle::new(dr))?;
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nr])?;
        let sm = Handle::new(sm);
        let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into()])?;
        rhs.set_value(nl, "imposed_T", 0.0)?;
        rhs.set_value(nr, "imposed_T", 1.0)?;
        let k = stiffness(&model, &materials)?;
        solve(&k, &NodeField::from_sub(rhs))
    };

    // Dirichlet T(0) = 0 + single-term MPC 1·T(node4) = 1, solved by elimination.
    let by_elimination = || -> Result<NodeField> {
        let (nodes, coords, mut model, materials) = heat_bar()?;
        let left = poi1(&nodes[0])?;
        let mult_l = barycenter(&left)?;
        let dl = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &left,
            &mult_l,
            None,
            None,
            Default::default(),
        )?;
        let nl = dl.multiplier_nodes()?[0];
        model.add_sub(Handle::new(dl))?;

        let right = poi1(&nodes[N_ELEMS])?;
        let mult_mpc = barycenter(&right)?;
        let terms = vec![MpcTerm::new(&right, "T".into(), "q".into(), 1.0)?];
        let mpc = SubModel::mpc(terms, &mult_mpc, None, None, Default::default())?;
        let nm = mpc.multiplier_nodes()?[0];
        model.add_sub(Handle::new(mpc))?;

        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[nl])?;
        sm.add_cell(&[nm])?;
        let sm = Handle::new(sm);
        let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
        rhs.set_value(nl, "imposed_T", 0.0)?;
        rhs.set_value(nm, "mpc_rhs", 1.0)?;
        let k = stiffness(&model, &materials)?;
        eliminate::solve(&k, &model, &NodeField::from_sub(rhs))
    };

    let (nodes, ..) = heat_bar()?;
    let dir = reference()?;
    let mpc = by_elimination()?;
    for node in &nodes {
        let a = dir.value(node.id(), "T")?;
        let b = mpc.value(node.id(), "T")?;
        assert!(
            (a - b).abs() < TOL,
            "node {:?}: dirichlet {a} vs elimination {b}",
            node.id()
        );
    }
    Ok(())
}

/// Elimination recovers the multiplier-equivalent **reaction** in post-processing:
/// on `-u'' = 0` with `T(0) = 0`, `T(1) = 1`, the reaction at each constrained
/// node (its dual `"q"` row, with `a_s = 1`) equals the Lagrange multiplier
/// `+1` at x=0, `-1` at x=1.
#[test]
fn elimination_recovers_reaction_equals_multiplier() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;
    let left = poi1(&nodes[0])?;
    let right = poi1(&nodes[N_ELEMS])?;
    let mult_l = barycenter(&left)?;
    let mult_r = barycenter(&right)?;
    let dl = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &left,
        &mult_l,
        None,
        None,
        Default::default(),
    )?;
    let dr = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &right,
        &mult_r,
        None,
        None,
        Default::default(),
    )?;
    let (nl, nr) = (dl.multiplier_nodes()?[0], dr.multiplier_nodes()?[0]);
    model.add_sub(Handle::new(dl))?;
    model.add_sub(Handle::new(dr))?;

    let mut sm = SubMesh::new(coords, ElementType::POI1);
    sm.add_cell(&[nl])?;
    sm.add_cell(&[nr])?;
    let sm = Handle::new(sm);
    let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into()])?;
    rhs.set_value(nl, "imposed_T", 0.0)?;
    rhs.set_value(nr, "imposed_T", 1.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let lagrange = solve(&k, &rhs)?;
    let elim = eliminate::solve(&k, &model, &rhs)?;

    // Lagrange multipliers (reference boundary fluxes).
    let lambda_l = lagrange.value(nl, "lambda_T")?;
    let lambda_r = lagrange.value(nr, "lambda_T")?;
    assert!((lambda_l - 1.0).abs() < TOL);
    assert!((lambda_r + 1.0).abs() < TOL);

    // Elimination reactions land in the slaves' own dual row "q" at the
    // constrained physics nodes; with a_s = 1 they equal the multipliers.
    let react_l = elim.value(nodes[0].id(), "q")?;
    let react_r = elim.value(nodes[N_ELEMS].id(), "q")?;
    assert!(
        (react_l - lambda_l).abs() < TOL,
        "left reaction {react_l} vs λ {lambda_l}"
    );
    assert!(
        (react_r - lambda_r).abs() < TOL,
        "right reaction {react_r} vs λ {lambda_r}"
    );
    Ok(())
}

/// Periodicity by elimination: `T(node4) − T(node0) = 0` (a disjoint slave) forces
/// both ends equal; anchored by `T(node1) = 0.5`, the `-u'' = 0` solution is the
/// constant field `u ≡ 0.5`.
#[test]
fn elimination_periodicity_constant_field() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    // Dirichlet T(node1) = 0.5 (interior anchor).
    let anchor = poi1(&nodes[1])?;
    let mult_a = barycenter(&anchor)?;
    let dir = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &anchor,
        &mult_a,
        None,
        None,
        Default::default(),
    )?;
    let dir_mult = dir.multiplier_nodes()?[0];
    model.add_sub(Handle::new(dir))?;

    // Periodicity T(node4) − T(node0) = 0.
    let dual = model.dual_of("T")?.expect("heat conduction declares T");
    let mesh4 = poi1(&nodes[N_ELEMS])?;
    let mesh0 = poi1(&nodes[0])?;
    let mult_mpc_mesh = barycenter(&mesh4)?;
    let terms = vec![
        MpcTerm::new(&mesh4, "T".into(), dual.clone(), 1.0)?,
        MpcTerm::new(&mesh0, "T".into(), dual, -1.0)?,
    ];
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None, Default::default())?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    // RHS: imposed_T = 0.5 at the anchor, mpc_rhs = 0.
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
    rhs.set_value(dir_mult, "imposed_T", 0.5)?;
    rhs.set_value(mpc_mult, "mpc_rhs", 0.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let elim = eliminate::solve(&k, &model, &rhs)?;

    for node in &nodes {
        let got = elim.value(node.id(), "T")?;
        assert!(
            (got - 0.5).abs() < TOL,
            "T at {:?}: got {got}, expected 0.5",
            node.id()
        );
    }
    Ok(())
}

/// Chaining is out of v1 scope: two relations that would eliminate the **same**
/// DOF (a Dirichlet and a single-term MPC both on node0) must be rejected.
#[test]
fn elimination_rejects_chaining() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    let imposed0 = poi1(&nodes[0])?;
    let mult0 = barycenter(&imposed0)?;
    let dir = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &imposed0,
        &mult0,
        None,
        None,
        Default::default(),
    )?;
    let dir_mult = dir.multiplier_nodes()?[0];
    model.add_sub(Handle::new(dir))?;

    // A single-term MPC on the *same* node0 — its only DOF is already a slave.
    let mesh0 = poi1(&nodes[0])?;
    let mult_mpc_mesh = barycenter(&mesh0)?;
    let terms = vec![MpcTerm::new(&mesh0, "T".into(), "q".into(), 1.0)?];
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None, Default::default())?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let rhs = NodeField::from_sub(SubNodeField::from_poi1(
        &rhs_sm,
        vec!["imposed_T".into(), "mpc_rhs".into()],
    )?);

    let k = stiffness(&model, &materials)?;
    assert!(eliminate::solve(&k, &model, &rhs).is_err());
    Ok(())
}

/// A relation whose only term has a zero coefficient has no valid denominator
/// `a_s` and must be rejected.
#[test]
fn elimination_rejects_zero_coefficient() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    let mesh4 = poi1(&nodes[N_ELEMS])?;
    let mult_mpc_mesh = barycenter(&mesh4)?;
    let terms = vec![MpcTerm::new(&mesh4, "T".into(), "q".into(), 0.0)?];
    let mpc = SubModel::mpc(terms, &mult_mpc_mesh, None, None, Default::default())?;
    let mpc_mult = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[mpc_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let rhs = NodeField::from_sub(SubNodeField::from_poi1(&rhs_sm, vec!["mpc_rhs".into()])?);

    let k = stiffness(&model, &materials)?;
    assert!(eliminate::solve(&k, &model, &rhs).is_err());
    Ok(())
}

/// The condensation is cached on the matrix after the first elimination solve,
/// reused on the second (identical result), and invalidated when the matrix
/// changes.
#[test]
fn elimination_condensation_cached_then_invalidated() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    let left = poi1(&nodes[0])?;
    let mult_l = barycenter(&left)?;
    let dl = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &left,
        &mult_l,
        None,
        None,
        Default::default(),
    )?;
    let nl = dl.multiplier_nodes()?[0];
    model.add_sub(Handle::new(dl))?;

    let right = poi1(&nodes[N_ELEMS])?;
    let mult_mpc = barycenter(&right)?;
    let terms = vec![MpcTerm::new(&right, "T".into(), "q".into(), 1.0)?];
    let mpc = SubModel::mpc(terms, &mult_mpc, None, None, Default::default())?;
    let nm = mpc.multiplier_nodes()?[0];
    model.add_sub(Handle::new(mpc))?;

    let mut sm = SubMesh::new(coords, ElementType::POI1);
    sm.add_cell(&[nl])?;
    sm.add_cell(&[nm])?;
    let sm = Handle::new(sm);
    let mut rhs = SubNodeField::from_poi1(&sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
    rhs.set_value(nl, "imposed_T", 0.0)?;
    rhs.set_value(nm, "mpc_rhs", 1.0)?;
    let rhs = NodeField::from_sub(rhs);

    let mut k = stiffness(&model, &materials)?;
    assert!(k.cached_factorization::<Condensation>().is_none());

    let s1 = eliminate::solve(&k, &model, &rhs)?;
    assert!(k.cached_factorization::<Condensation>().is_some());
    let s2 = eliminate::solve(&k, &model, &rhs)?;
    for node in &nodes {
        assert_eq!(s1.value(node.id(), "T")?, s2.value(node.id(), "T")?);
    }

    // Mutating the matrix invalidates the cached condensation.
    let other = Handle::new(Coords::new(1)?);
    let b = Node::create_in(other.clone(), &[0.0])?;
    let bsm = {
        let mut bsm = SubMesh::new(other, ElementType::POI1);
        bsm.add_cell(&[b.id()])?;
        Handle::new(bsm)
    };
    let mut block = SubMatrix::new(
        bsm.clone(),
        bsm,
        vec!["q".into()],
        vec!["T".into()],
        DofOrdering::NodesThenVars,
        false,
    )?;
    block.add_entry(b.id(), "q", b.id(), "T", 4.0)?;
    k.add_sub(Handle::new(block))?;
    assert!(k.cached_factorization::<Condensation>().is_none());
    Ok(())
}
