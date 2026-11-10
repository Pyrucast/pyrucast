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
