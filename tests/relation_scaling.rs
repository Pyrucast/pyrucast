//! A relation holds to round-off whatever the stiffness it is set against.
//!
//! A clamp is a row of 1's next to a stiffness of `E·h`: some 10⁹ for a steel
//! part. Factorized as assembled, the clamped nodes of a steel bar kept a
//! displacement of a few millionths of the tip's; the solver now brings the
//! relations to the structure's scale before factorizing, and they hold.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::containers::field::SubField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::node_field::NodeField;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::symmetry::MaterialSymmetry;
use pyrucast::models::tensor::Kinematics;
use pyrucast::models::RelationSense;
use pyrucast::ops::element_field::{material_field, StrainMeasure};
use pyrucast::ops::solver::{RelationMethod, SolveOptions};
use pyrucast::ops::{matrix, mesh, model, solver};

/// A bar 1 × 0.1 × 0.1 of 20 × 2 × 2 HEX8, clamped on x = 0 and pulled down at
/// x = 1, of Young's modulus `young`: the largest displacement on the clamp,
/// and the tip's.
fn clamped_bar(young: f64) -> (f64, f64) {
    let coords = Handle::new(Coords::new(3).unwrap());
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
    let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
    let edge = mesh::line(&a, &b, 20, ElementType::SEG2).unwrap();
    let face = mesh::extrude(&edge, &[0.0, 0.1, 0.0], 2).unwrap();
    let bar = mesh::extrude(&face, &[0.0, 0.0, 0.1], 2).unwrap();
    let fes = FiniteElementSpace::lagrange(&bar).unwrap();
    let mut solid = model::elasticity(
        &fes,
        Kinematics::Full3D,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )
    .unwrap();
    let clamp = mesh::consolidate(
        &mesh::points_on_plane(&bar, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], None).unwrap(),
    )
    .unwrap();
    for variable in ["u_x", "u_y", "u_z"] {
        let blocked = model::dirichlet(&clamp, &solid, variable, RelationSense::Equality).unwrap();
        solid = solid.union(&blocked).unwrap();
    }
    let materials = material_field(&solid, &[("E", young), ("nu", 0.3)]).unwrap();
    let k = matrix::stiffness(&solid, &materials).unwrap();

    let tip = mesh::consolidate(
        &mesh::points_on_plane(&bar, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], None).unwrap(),
    )
    .unwrap();
    let force = NodeField::new(&tip, vec!["f_z".into()]).unwrap();
    force
        .get(0)
        .unwrap()
        .write()
        .set_uniform("f_z", -1000.0 * young / 200e9)
        .unwrap();
    let u = solver::solve(&k, &force).unwrap();

    let clamp_nodes = clamp.get(0).unwrap().read().connectivity().to_vec();
    let mut on_clamp = 0.0_f64;
    for node in clamp_nodes {
        for c in ["u_x", "u_y", "u_z"] {
            on_clamp = on_clamp.max(u.value(node, c).unwrap().abs());
        }
    }
    let tip_node = tip.get(0).unwrap().read().connectivity()[0];
    (on_clamp, u.value(tip_node, "u_z").unwrap().abs())
}

#[test]
fn a_clamp_holds_on_a_steel_bar_as_on_a_soft_one() {
    let (soft_clamp, soft_tip) = clamped_bar(1.0);
    let (steel_clamp, steel_tip) = clamped_bar(200e9);
    // Same load scaled with E: the same displacements.
    assert!(
        (soft_tip - steel_tip).abs() < 1e-9 * soft_tip,
        "{soft_tip} ≠ {steel_tip}"
    );
    assert!(
        soft_clamp <= 1e-14 * soft_tip,
        "soft: {soft_clamp:e} on the clamp"
    );
    // Before the relations were scaled: 5e-9 on the clamp, 3.4e-6 of the tip.
    assert!(
        steel_clamp <= 1e-14 * steel_tip,
        "steel: {steel_clamp:e} on the clamp"
    );
}

/// A steel bar clamped on x = 0 and pulled down at x = 1, with a **released**
/// bound `u_z ≥ −1` on every node of its free end — far from touching, so each
/// bound's multiplier is 0 and the solution is the bar's alone. A released
/// row once solved `1·λ = 0` next to coefficients of 10¹²: λ came back at the
/// round-off of the solve, some newtons, and pushed the bar.
#[test]
fn released_bounds_push_nothing_and_both_methods_agree() {
    let young = 200e9;
    let coords = Handle::new(Coords::new(3).unwrap());
    let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
    let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
    let edge = mesh::line(&a, &b, 20, ElementType::SEG2).unwrap();
    let face = mesh::extrude(&edge, &[0.0, 0.1, 0.0], 2).unwrap();
    let bar = mesh::extrude(&face, &[0.0, 0.0, 0.1], 2).unwrap();
    let fes = FiniteElementSpace::lagrange(&bar).unwrap();
    let mut solid = model::elasticity(
        &fes,
        Kinematics::Full3D,
        MaterialSymmetry::Isotropic,
        StrainMeasure::Linear,
    )
    .unwrap();
    let clamp = mesh::consolidate(
        &mesh::points_on_plane(&bar, &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], None).unwrap(),
    )
    .unwrap();
    let tip = mesh::consolidate(
        &mesh::points_on_plane(&bar, &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], None).unwrap(),
    )
    .unwrap();
    for variable in ["u_x", "u_y", "u_z"] {
        let blocked = model::dirichlet(&clamp, &solid, variable, RelationSense::Equality).unwrap();
        solid = solid.union(&blocked).unwrap();
    }
    let bound = model::dirichlet(&tip, &solid, "u_z", RelationSense::GreaterEqual).unwrap();
    let bound_multipliers = bound.multiplier_mesh().unwrap();
    solid = solid.union(&bound).unwrap();
    let materials = material_field(&solid, &[("E", young), ("nu", 0.3)]).unwrap();
    let k = matrix::stiffness(&solid, &materials).unwrap();

    // The load at the tip, and the bounds' right-hand side −1 (metres).
    let force = NodeField::new(&tip, vec!["f_z".into()]).unwrap();
    force
        .get(0)
        .unwrap()
        .write()
        .set_uniform("f_z", -1000.0)
        .unwrap();
    let gaps = NodeField::new(&bound_multipliers, vec!["imposed".into()]).unwrap();
    gaps.get(0)
        .unwrap()
        .write()
        .set_uniform("imposed", -1.0)
        .unwrap();
    let rhs = force.union(&gaps).unwrap();

    let lagrange = solver::solve(&k, &rhs).unwrap();
    let eliminate = SolveOptions {
        relations: RelationMethod::Eliminate,
        ..Default::default()
    };
    let elimination = solver::solve_with_options(&k, &rhs, &eliminate).unwrap();

    let mut tip_scale = 0.0_f64;
    let mut worst = 0.0_f64;
    for &node in tip.get(0).unwrap().read().connectivity() {
        let (l, e) = (
            lagrange.value(node, "u_z").unwrap(),
            elimination.value(node, "u_z").unwrap(),
        );
        tip_scale = tip_scale.max(l.abs());
        worst = worst.max((l - e).abs());
    }
    assert!(
        worst <= 1e-12 * tip_scale,
        "u_z: {worst:e} apart, of {tip_scale:e}"
    );
    for &m in bound_multipliers.get(0).unwrap().read().connectivity() {
        assert_eq!(lagrange.value(m, "lambda").unwrap(), 0.0);
        assert_eq!(elimination.value(m, "lambda").unwrap(), 0.0);
    }
}
