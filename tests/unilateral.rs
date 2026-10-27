//! Unilateral (inequality) constraints exercised end-to-end through the
//! public API — the active-set loop of `ops::solver::solve`, which the matrix
//! sends down that path when its relation rows carry an inequality.
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
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::RelationSense;
use pyrucast::ops::matrix::stiffness;
use pyrucast::ops::solver::{self, ActiveSetMethod, RelationMethod, SolveOptions};
use pyrucast::Result;

const N_ELEMS: usize = 4;
const H: f64 = 1.0 / N_ELEMS as f64;
const TOL: f64 = 1e-10;

/// Build the `[0, 1]` SEG2 bar with a `k = 1` heat-conduction sub-model already
/// added; returns the nodes, the shared coords, the model and the material.
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
    model.add_sub(Handle::new(SubModel::heat_conduction(
        sub,
        MaterialSymmetry::Isotropic,
    )?))?;

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
    let dir = SubModel::dirichlet(&imposed0, &model, "T", Default::default())?;
    let dir_mult = dir.multiplier_nodes()[0];
    model.add_sub(Handle::new(dir))?;

    // Unilateral bound T(1) ⋈ bound.
    let imposed1 = poi1(&nodes[N_ELEMS])?;
    let uni = SubModel::dirichlet(&imposed1, &model, "T", sense)?;
    let uni_mult = uni.multiplier_nodes()[0];
    model.add_sub(Handle::new(uni))?;

    // RHS: flux q at the right physics node, u_d at both multiplier slots.
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[nodes[N_ELEMS].id()])?;
    rhs_sm.add_cell(&[dir_mult])?;
    rhs_sm.add_cell(&[uni_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["q".into(), "imposed".into()])?;
    rhs.set_value(nodes[N_ELEMS].id(), "q", q)?;
    rhs.set_value(dir_mult, "imposed", 0.0)?;
    rhs.set_value(uni_mult, "imposed", bound)?;
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
    let got = solution.value(setup.uni_mult, "lambda")?;
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
    let solution = solver::solve(&k, &setup.rhs)?;
    check(&setup, &solution, 5.0, 0.0)
}

/// `T(1) ≥ 2` with a weak push `q = 1`: the unconstrained `T(1) = 1` violates
/// the bound — the relation holds active, `T(x) = 2x`, `λ = q − a = −1 ≤ 0`.
#[test]
fn greater_equal_holds_when_violated() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = solver::solve(&k, &setup.rhs)?;
    check(&setup, &solution, 2.0, -1.0)
}

/// `T(1) ≤ 2` with a strong push `q = 5`: the bound blocks, `T(x) = 2x`,
/// `λ = q − a = 3 ≥ 0` (the blocked flux).
#[test]
fn less_equal_holds_when_violated() -> Result<()> {
    let setup = bounded_bar(5.0, 2.0, RelationSense::LessEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = solver::solve(&k, &setup.rhs)?;
    check(&setup, &solution, 2.0, 3.0)
}

/// `T(1) ≤ 2` with a weak push `q = 1`: feasible — released, `T(x) = x`, `λ = 0`.
#[test]
fn less_equal_releases_when_feasible() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::LessEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = solver::solve(&k, &setup.rhs)?;
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
    let first = solver::solve(&k, &setup.rhs)?;
    check(&setup, &first, 2.0, -1.0)?;

    // …re-solve identically (pure warm start, no refactorization)…
    let again = solver::solve(&k, &setup.rhs)?;
    check(&setup, &again, 2.0, -1.0)?;

    // …then push hard (q = 5): the relation must release from the warm start.
    let strong = bounded_bar(5.0, 2.0, RelationSense::GreaterEqual)?;
    let flipped = solver::solve(&k, &strong.rhs)?;
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
        let dir = SubModel::dirichlet(&imposed0, &model, "T", Default::default())?;
        let dir_mult = dir.multiplier_nodes()[0];
        model.add_sub(Handle::new(dir))?;

        let mesh_last = poi1(&nodes[N_ELEMS])?;
        let mesh_first = poi1(&nodes[0])?;
        let mpc = SubModel::mpc(
            &[(&mesh_last, "T", 1.0), (&mesh_first, "T", -1.0)],
            &model,
            RelationSense::GreaterEqual,
        )?;
        let mpc_mult = mpc.multiplier_nodes()[0];
        model.add_sub(Handle::new(mpc))?;

        let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
        rhs_sm.add_cell(&[dir_mult])?;
        rhs_sm.add_cell(&[mpc_mult])?;
        let rhs_sm = Handle::new(rhs_sm);
        let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["imposed".into()])?;
        rhs.set_value(dir_mult, "imposed", 0.0)?;
        rhs.set_value(mpc_mult, "imposed", g)?;
        let rhs = NodeField::from_sub(rhs);

        let k = stiffness(&model, &materials)?;
        let solution = solver::solve(&k, &rhs)?;
        for (i, node) in nodes.iter().enumerate() {
            let got = solution.value(node.id(), "T")?;
            let expected = slope * i as f64 * H;
            assert!(
                (got - expected).abs() < TOL,
                "g={g}: T at node {i}: got {got}, expected {expected}"
            );
        }
        // Complementarity on the multiplier.
        let lambda = solution.value(mpc_mult, "lambda")?;
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
    let cases = [
        (5.0, 2.0, RelationSense::GreaterEqual),
        (1.0, 2.0, RelationSense::GreaterEqual),
        (5.0, 2.0, RelationSense::LessEqual),
        (1.0, 2.0, RelationSense::LessEqual),
    ];
    for (q, bound, sense) in cases {
        let setup = bounded_bar(q, bound, sense)?;
        let k = stiffness(&setup.model, &setup.materials)?;
        let schur = solver::solve_with_options(
            &k,
            &setup.rhs,
            &SolveOptions {
                active_set: ActiveSetMethod::SchurComplement,
                ..Default::default()
            },
        )?;
        let refac = solver::solve_with_options(
            &k,
            &setup.rhs,
            &SolveOptions {
                active_set: ActiveSetMethod::Refactorize,
                ..Default::default()
            },
        )?;
        for node in &setup.nodes {
            let a = schur.value(node.id(), "T")?;
            let b = refac.value(node.id(), "T")?;
            assert!((a - b).abs() < TOL, "T mismatch q={q} sense={sense:?}");
        }
        let la = schur.value(setup.uni_mult, "lambda")?;
        let lb = refac.value(setup.uni_mult, "lambda")?;
        assert!((la - lb).abs() < TOL, "λ mismatch q={q} sense={sense:?}");
    }
    Ok(())
}

/// The sense travels in the matrix: the same bar, the same load, the bound
/// declared `≥` or `=` — `solve` reads which from the blocks and answers
/// differently. Feasible push (`q = 5 > 2`): the inequality releases
/// (`T(1) = 5`), the equation holds (`T(1) = 2`).
#[test]
fn the_matrix_carries_the_sense() -> Result<()> {
    let bound = bounded_bar(5.0, 2.0, RelationSense::GreaterEqual)?;
    let equation = bounded_bar(5.0, 2.0, RelationSense::Equality)?;
    let free = solver::solve(&stiffness(&bound.model, &bound.materials)?, &bound.rhs)?;
    let held = solver::solve(
        &stiffness(&equation.model, &equation.materials)?,
        &equation.rhs,
    )?;
    let right = |s: &Setup| s.nodes[N_ELEMS].id();
    assert!((free.value(right(&bound), "T")? - 5.0).abs() < TOL);
    assert!((held.value(right(&equation), "T")? - 2.0).abs() < TOL);
    Ok(())
}

/// A sum of matrices keeps the blocks of both operands, so the sense of a
/// relation survives `K_physics + K_constraints` — what an operator built
/// piece by piece (stiffness, mass, geometric stiffness) relies on. Active
/// bound (`q = 1 < 2`): `T = 2x`, `λ = q − a = −1`.
#[test]
fn the_sense_survives_a_sum() -> Result<()> {
    use pyrucast::models::Physics;
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let physics = stiffness(&setup.model.exclude(Physics::Constraint), &setup.materials)?;
    let relations = stiffness(&setup.model.filter(Physics::Constraint), &setup.materials)?;
    let k = (&physics + &relations)?;
    let solution = solver::solve(&k, &setup.rhs)?;
    check(&setup, &solution, 2.0, -1.0)
}

/// When the inequality-free base is **singular** (a structure that only holds
/// once the contact is active — here a bar with a single bound and no equality
/// Dirichlet, so the released state is pure Neumann), the Schur path must fall
/// back to refactorization and still return the correct solution.
#[test]
fn schur_falls_back_when_base_is_singular() -> Result<()> {
    let (nodes, coords, mut model, materials) = heat_bar()?;
    // A single `T(1) ≤ 2` bound, no other constraint.
    let imp = poi1(&nodes[N_ELEMS])?;
    let uni = SubModel::dirichlet(&imp, &model, "T", RelationSense::LessEqual)?;
    let uni_mult = uni.multiplier_nodes()[0];
    model.add_sub(Handle::new(uni))?;

    // Flux q = 5 at the right end drives T up into the bound ⇒ the relation is
    // active, `T ≡ 2` (a flat field satisfying the pinned end and zero flux).
    let mut rhs_sm = SubMesh::new(coords, ElementType::POI1);
    rhs_sm.add_cell(&[nodes[N_ELEMS].id()])?;
    rhs_sm.add_cell(&[uni_mult])?;
    let rhs_sm = Handle::new(rhs_sm);
    let mut rhs = SubNodeField::from_poi1(&rhs_sm, vec!["q".into(), "imposed".into()])?;
    rhs.set_value(nodes[N_ELEMS].id(), "q", 5.0)?;
    rhs.set_value(uni_mult, "imposed", 2.0)?;
    let rhs = NodeField::from_sub(rhs);

    let k = stiffness(&model, &materials)?;
    let schur = solver::solve_with_options(
        &k,
        &rhs,
        &SolveOptions {
            active_set: ActiveSetMethod::SchurComplement,
            ..Default::default()
        },
    )?;
    let refac = solver::solve_with_options(
        &k,
        &rhs,
        &SolveOptions {
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

/// A matrix whose relations are all equations goes down the plain LU path, and
/// the bound then holds as an equation: `T(1) = a`, so `T(x) = a·x`.
#[test]
fn all_equality_model_falls_back_to_plain_solve() -> Result<()> {
    let setup = bounded_bar(3.0, 0.5, RelationSense::Equality)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let solution = solver::solve(&k, &setup.rhs)?;
    for (i, node) in setup.nodes.iter().enumerate() {
        let x = i as f64 / N_ELEMS as f64;
        assert!((solution.value(node.id(), "T")? - 0.5 * x).abs() < TOL);
    }
    Ok(())
}

/// The elimination (condensation) solver enforces every relation
/// unconditionally: it must reject a unilateral model with a clear error.
#[test]
fn eliminate_rejects_unilateral_relations() -> Result<()> {
    let setup = bounded_bar(1.0, 2.0, RelationSense::GreaterEqual)?;
    let k = stiffness(&setup.model, &setup.materials)?;
    let eliminate = SolveOptions {
        relations: RelationMethod::Eliminate,
        ..Default::default()
    };
    let err = solver::solve_with_options(&k, &setup.rhs, &eliminate).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("relations=\"lagrange\""),
        "error should point at the Lagrange path, got: {msg}"
    );
    Ok(())
}
