//! Unilateral (inequality) constraints exercised end-to-end through the
//! public API — the active-set solver `ops::solver::unilateral`.
//!
//! A 1-D heat-conduction bar `-T'' = 0` on `[0, 1]` (SEG2 grid, `k = 1`),
//! `T(0) = 0` (equality Dirichlet) and a flux load `q` at the right end: the
//! unconstrained solution is `T(x) = q·x`. A unilateral bound `T(1) ⋈ a` then
//! either stays inactive (the unconstrained solution is feasible, `λ = 0`) or
//! becomes active (`T(1) = a`, the multiplier carries the blocked flux):
//!
//! - active: `T(x) = a·x`, and the balance at the right node gives
//!   `λ = q − a` (`≤ 0` for `≥`, `≥ 0` for `≤`);
//! - inactive: `T(x) = q·x` and `λ = 0` exactly.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::{ElementField, SubElementField};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::{Model, SubModel};
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::mpc::MpcTerm;
use pyrucast::models::RelationSense;
use pyrucast::ops::assemble::stiffness;
use pyrucast::ops::mesher::barycenter;
use pyrucast::ops::solver::{eliminate, lu, unilateral};
use pyrucast::store::{insert, Handle};
use pyrucast::Result;

const N_ELEMS: usize = 4;
const H: f64 = 1.0 / N_ELEMS as f64;
const TOL: f64 = 1e-10;

/// Build the `[0, 1]` SEG2 bar with a `k = 1` heat-conduction sub-model already
/// added; returns the nodes, the shared coords, the model and the material.
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

/// The complete unilateral setup: `T(0) = 0`, flux `q` at the right end,
/// `T(1) ⋈ bound` with the given sense. Returns the model, the rhs, the nodes
/// and the two multiplier nodes (equality Dirichlet, unilateral bound).
struct Setup {
    nodes: Vec<Node>,
    model: Model,
    materials: ElementField,
    rhs: NodeField,
    uni_mult: NodeId,
}

fn bounded_bar(q: f64, bound: f64, sense: RelationSense) -> Result<Setup> {
    let (nodes, coords, mut model, materials) = heat_bar()?;

    // Equality Dirichlet T(0) = 0.
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
    model.add_sub(insert(dir))?;

    // Unilateral bound T(1) ⋈ bound.
    let imposed1 = poi1(&nodes[N_ELEMS])?;
    let mult1 = barycenter(&imposed1)?;
    let uni = SubModel::dirichlet("T".into(), "q".into(), &imposed1, &mult1, None, None, sense)?;
    let uni_mult = uni.multiplier_nodes()?[0];
    model.add_sub(insert(uni))?;

    // RHS: flux q at the right physics node, u_d at both multiplier slots.
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[nodes[N_ELEMS].id()])?;
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[uni_mult])?;
    let rhs_sm = insert(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["q".into(), "imposed_T".into()])?;
    rhs.set_value(nodes[N_ELEMS].id(), "q", q)?;
    rhs.set_value(dir_mult, "imposed_T", 0.0)?;
    rhs.set_value(uni_mult, "imposed_T", bound)?;
    let rhs = NodeField::from_sub(rhs);

    Ok(Setup {
        nodes,
        model,
        materials,
        rhs,
        uni_mult,
    })
}

/// Assert `T(x) = slope·x` at every node and `λ = lambda` at the bound.
fn check(setup: &Setup, solution: &NodeField, slope: f64, lambda: f64) -> Result<()> {
    for (i, node) in setup.nodes.iter().enumerate() {
        let got = solution.value(node.id(), "T")?;
        let expected = slope * i as f64 * H;
        assert!(
            (got - expected).abs() < TOL,
            "T at node {i}: got {got}, expected {expected}"
        );
    }
    let got = solution.value(setup.uni_mult, "lambda_T")?;
    assert!(
        (got - lambda).abs() < TOL,
        "lambda: got {got}, expected {lambda}"
    );
    Ok(())
}

/// `T(1) ≥ 2` with a strong push `q = 5`: the unconstrained `T(1) = 5` is
/// feasible — the relation releases, `T(x) = 5x`, `λ = 0` exactly.
#[test]
fn greater_equal_releases_when_feasible() -> Result<()> {
    let setup = bounded_bar(5.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &solution, 5.0, 0.0)
}

/// `T(1) ≥ 2` with a weak push `q = 1`: the unconstrained `T(1) = 1` violates
/// the bound — the relation holds active, `T(x) = 2x`, `λ = q − a = −1 ≤ 0`.
#[test]
fn greater_equal_holds_when_violated() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &solution, 2.0, -1.0)
}

/// `T(1) ≤ 2` with a strong push `q = 5`: the bound blocks, `T(x) = 2x`,
/// `λ = q − a = 3 ≥ 0` (the blocked flux).
#[test]
fn less_equal_holds_when_violated() -> Result<()> {
    let setup = bounded_bar(5.0, 2.0, RelationSense::LessEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &solution, 2.0, 3.0)
}

/// `T(1) ≤ 2` with a weak push `q = 1`: feasible — released, `T(x) = x`, `λ = 0`.
#[test]
fn less_equal_releases_when_feasible() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::LessEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &solution, 1.0, 0.0)
}

/// Warm start: a second solve on the same matrix reuses the converged status
/// (same result), and a rhs change that flips the status re-iterates correctly
/// from the cached state.
#[test]
fn warm_start_survives_a_status_flip() -> Result<()> {
    // Start active (q = 1 against T(1) ≥ 2)…
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let first = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &first, 2.0, -1.0)?;

    // …re-solve identically (pure warm start, no refactorization)…
    let again = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    check(&setup, &again, 2.0, -1.0)?;

    // …then push hard (q = 5): the relation must release from the warm start.
    let strong = bounded_bar(5.0, 2.0, RelationSense::GreaterEqual)?;
    let flipped = unilateral::solve(&setup.model, &k, &strong.rhs)?;
    check(&strong, &flipped, 5.0, 0.0)
}

/// A two-term unilateral MPC `T(1) − T(0) ≥ 1` with no load: the unconstrained
/// solution `T ≡ 0` violates it — active, `T(x) = x` (same as the equality MPC);
/// with `g = −1` it is feasible — released, `T ≡ 0`.
#[test]
fn unilateral_mpc_difference_relation() -> Result<()> {
    for (g, slope) in [(1.0, 1.0), (-1.0, 0.0)] {
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
        model.add_sub(insert(dir))?;

        let dual = model.dual_of("T")?.expect("heat conduction declares T");
        let mesh_last = poi1(&nodes[N_ELEMS])?;
        let mesh_first = poi1(&nodes[0])?;
        let mult_mpc_mesh = barycenter(&mesh_last)?;
        let terms = vec![
            MpcTerm::new(&mesh_last, "T".into(), dual.clone(), 1.0)?,
            MpcTerm::new(&mesh_first, "T".into(), dual, -1.0)?,
        ];
        let mpc = SubModel::mpc(
            terms,
            &mult_mpc_mesh,
            None,
            None,
            RelationSense::GreaterEqual,
        )?;
        let mpc_mult = mpc.multiplier_nodes()?[0];
        model.add_sub(insert(mpc))?;

        let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
        rhs_sm.add_cell(&[dir_mult])?;
        rhs_sm.add_cell(&[mpc_mult])?;
        let rhs_sm = insert(rhs_sm);
        let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed_T".into(), "mpc_rhs".into()])?;
        rhs.set_value(dir_mult, "imposed_T", 0.0)?;
        rhs.set_value(mpc_mult, "mpc_rhs", g)?;
        let rhs = NodeField::from_sub(rhs);

        let k = stiffness(&model, &materials)?;
        let solution = unilateral::solve(&model, &k, &rhs)?;
        for (i, node) in nodes.iter().enumerate() {
            let got = solution.value(node.id(), "T")?;
            let expected = slope * i as f64 * H;
            assert!(
                (got - expected).abs() < TOL,
                "g={g}: T at node {i}: got {got}, expected {expected}"
            );
        }
        // Complementarity on the multiplier.
        let lambda = solution.value(mpc_mult, "lambda_mpc")?;
        if slope == 0.0 {
            assert!(lambda.abs() < TOL, "released relation must carry λ = 0");
        } else {
            assert!(lambda < 0.0, "active ≥ relation must carry λ ≤ 0");
        }
    }
    Ok(())
}

/// The two active-set back-ends — the Schur base-reuse method (default) and the
/// per-status refactorization — must reach the **same** solution on every
/// scenario (active, released, both senses). The Schur path here has a
/// non-singular inequality-free base (`T(0) = 0` pins the bar), so it never
/// falls back.
#[test]
fn schur_and_refactorize_agree() -> Result<()> {
    use pyrucast::ops::solver::unilateral::{ActiveSetMethod, UnilateralOptions};

    let cases = [
        (5.0, 2.0, RelationSense::GreaterEqual),
        (1.0, 2.0, RelationSense::GreaterEqual),
        (5.0, 2.0, RelationSense::LessEqual),
        (1.0, 2.0, RelationSense::LessEqual),
    ];
    for (q, bound, sense) in cases {
        let setup = bounded_bar(q, bound, sense)?;
        let k = stiffness(&setup.model, &setup.materials)?;
        let schur = unilateral::solve_with_options(
            &setup.model,
            &k,
            &setup.rhs,
            &UnilateralOptions {
                active_set: ActiveSetMethod::SchurComplement,
                ..Default::default()
            },
        )?;
        let refac = unilateral::solve_with_options(
            &setup.model,
            &k,
            &setup.rhs,
            &UnilateralOptions {
                active_set: ActiveSetMethod::Refactorize,
                ..Default::default()
            },
        )?;
        for node in &setup.nodes {
            let a = schur.value(node.id(), "T")?;
            let b = refac.value(node.id(), "T")?;
            assert!((a - b).abs() < TOL, "T mismatch q={q} sense={sense:?}");
        }
        let la = schur.value(setup.uni_mult, "lambda_T")?;
        let lb = refac.value(setup.uni_mult, "lambda_T")?;
        assert!((la - lb).abs() < TOL, "λ mismatch q={q} sense={sense:?}");
    }
    Ok(())
}

/// When the inequality-free base is **singular** (a structure that only holds
/// once the contact is active — here a bar with a single bound and no equality
/// Dirichlet, so the released state is pure Neumann), the Schur path must fall
/// back to refactorization and still return the correct solution.
#[test]
fn schur_falls_back_when_base_is_singular() -> Result<()> {
    use pyrucast::ops::solver::unilateral::{ActiveSetMethod, UnilateralOptions};

    let (nodes, coords, mut model, materials) = heat_bar()?;
    // A single `T(1) ≤ 2` bound, no other constraint.
    let imp = poi1(&nodes[N_ELEMS])?;
    let mult = barycenter(&imp)?;
    let uni = SubModel::dirichlet(
        "T".into(),
        "q".into(),
        &imp,
        &mult,
        None,
        None,
        RelationSense::LessEqual,
    )?;
    let uni_mult = uni.multiplier_nodes()?[0];
    model.add_sub(insert(uni))?;

    // Flux q = 5 at the right end drives T up into the bound ⇒ the relation is
    // active, `T ≡ 2` (a flat field satisfying the pinned end and zero flux).
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[nodes[N_ELEMS].id()])?;
    rhs_sm.add_cell(&[uni_mult])?;
    let rhs_sm = insert(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["q".into(), "imposed_T".into()])?;
    rhs.set_value(nodes[N_ELEMS].id(), "q", 5.0)?;
    rhs.set_value(uni_mult, "imposed_T", 2.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let schur = unilateral::solve_with_options(
        &model,
        &k,
        &rhs,
        &UnilateralOptions {
            active_set: ActiveSetMethod::SchurComplement,
            ..Default::default()
        },
    )?;
    let refac = unilateral::solve_with_options(
        &model,
        &k,
        &rhs,
        &UnilateralOptions {
            active_set: ActiveSetMethod::Refactorize,
            ..Default::default()
        },
    )?;
    for node in &nodes {
        let a = schur.value(node.id(), "T")?;
        let b = refac.value(node.id(), "T")?;
        assert!((a - b).abs() < TOL, "fallback must match refactorize");
        assert!((a - 2.0).abs() < TOL, "active ≤ bound ⇒ T ≡ 2");
    }
    Ok(())
}

/// A model whose constraints are all equalities routes through the plain LU
/// path: `solve_unilateral` and `solve` agree to machine precision.
#[test]
fn all_equality_model_falls_back_to_plain_solve() -> Result<()> {
    let setup = bounded_bar(3.0, 0.5, RelationSense::Equality)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let via_unilateral = unilateral::solve(&setup.model, &k, &setup.rhs)?;
    let via_lu = lu::solve(&k, &setup.rhs)?;
    for node in &setup.nodes {
        let a = via_unilateral.value(node.id(), "T")?;
        let b = via_lu.value(node.id(), "T")?;
        assert!((a - b).abs() < TOL);
    }
    Ok(())
}

/// The elimination (condensation) solver enforces every relation
/// unconditionally: it must reject a unilateral model with a clear error.
#[test]
fn eliminate_rejects_unilateral_relations() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let err = eliminate::solve(&setup.model, &k, &setup.rhs).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("solve_unilateral"),
        "error should point at solve_unilateral, got: {msg}"
    );
    Ok(())
}
