//! Source of the Rust examples of `book/src/triangulation.md`.
//!
//! The page pulls these functions through `{{#include …:anchor}}` and
//! exécute. L'ancre couvre la **fonction entière**, signature comprise : en
//! Rust all code lives in a `fn`, and mdbook does not strip the indentation.
//!
//! Voir `book/src/developper/documentation-et-tests.md`.

// ANCHOR: aire_signee
use pyrucast::atoms::Point2;
use pyrucast::ops::mesh::triangulation::signed_area;

#[test]
fn l_aire_signee_donne_le_sens_de_parcours() {
    // Carré unitaire CCW — aire = +1.
    let pts = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];
    assert!((signed_area(&pts) - 1.0).abs() < 1e-12);

    // The same square CW — area = -1.
    let pts_cw: Vec<_> = pts.iter().cloned().rev().collect();
    assert!((signed_area(&pts_cw) + 1.0).abs() < 1e-12);
}
// ANCHOR_END: aire_signee

// ANCHOR: ear_clip
use pyrucast::ops::mesh::triangulation::ear_clip_2d;

#[test]
fn le_decoupage_par_oreilles_rend_n_moins_2_triangles() {
    // Pentagone CCW quelconque.
    let pts = vec![
        Point2::new(0.0, 0.0),
        Point2::new(2.0, 0.0),
        Point2::new(2.5, 1.5),
        Point2::new(1.0, 2.5),
        Point2::new(-0.5, 1.5),
    ];
    let triangles = ear_clip_2d(&pts).unwrap();
    // n - 2 = 3 triangles, indices into pts.
    assert_eq!(triangles.len(), 3);

    // Checking that a triangle is CCW (signed area > 0).
    for [i, j, k] in &triangles {
        let area = signed_area(&[pts[*i], pts[*j], pts[*k]]);
        assert!(area > 0.0, "triangle non-CCW détecté");
    }
}
// ANCHOR_END: ear_clip

// ANCHOR: repere_local
use pyrucast::atoms::Point3;
use pyrucast::ops::mesh::triangulation::{in_plane_basis, newell_normal};

#[test]
fn un_contour_3d_planaire_se_ramene_a_un_repere_local() {
    // A triangle in the y = 0 plane (the xz plane).
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 0.0, 1.0),
    ];

    let normal = newell_normal(&pts).unwrap();
    // Normale attendue : (0, -1, 0) ou (0, 1, 0) selon le sens.
    assert!(normal.y.abs() > 0.99);

    let (u, v) = in_plane_basis(normal);
    // u and v are orthogonal to each other and to the normal.
    assert!(u.dot(&v).abs() < 1e-12);
    assert!(u.dot(&normal).abs() < 1e-12);

    // Projecting a point into the local frame (u, v).
    let origin = Point3::new(0.0, 0.0, 0.0);
    let p = Point3::new(0.5, 0.0, 0.5);
    let pu = (p - origin).dot(&u);
    let pv = (p - origin).dot(&v);
    println!("({pu:.3}, {pv:.3})");
}
// ANCHOR_END: repere_local

// ANCHOR: delaunay
use pyrucast::ops::mesh::triangulation::delaunay_2d;

#[test]
fn delaunay_maille_un_nuage_de_points() {
    let pts = vec![
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 0.0),
        Point2::new(3.0, 3.0),
        Point2::new(0.0, 3.0),
        Point2::new(1.5, 1.5), // point intérieur
    ];
    let triangles = delaunay_2d(&pts).unwrap();
    // 4 points = 2 triangles Delaunay ; le 5e point intérieur en ajoute d'autres.
    println!("{} triangles", triangles.len());
    assert!(triangles.len() >= 2);
}
// ANCHOR_END: delaunay

// ANCHOR: polygone_troue
use pyrucast::ops::mesh::triangulation::triangulate_polygon_with_holes;

#[test]
fn un_polygone_troue_se_triangule_directement() {
    // Contour extérieur : carré 4×4.
    let outer = vec![
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(4.0, 4.0),
        Point2::new(0.0, 4.0),
    ];
    // Trou : carré 2×2 centré.
    let hole = vec![
        Point2::new(1.0, 1.0),
        Point2::new(3.0, 1.0),
        Point2::new(3.0, 3.0),
        Point2::new(1.0, 3.0),
    ];
    let triangles = triangulate_polygon_with_holes(&outer, &[hole]).unwrap();
    // Area = 16 - 4 = 12; without Steiner: 6 raw triangles.
    println!("{} triangles", triangles.len());
    assert!(!triangles.is_empty());
}
// ANCHOR_END: polygone_troue

// ANCHOR: raffinement
use pyrucast::ops::mesh::triangulation::{
    triangulate_polygon_with_holes_refined, RefinementOptions,
};

#[test]
fn le_raffinement_de_ruppert_insere_des_points_de_steiner() {
    let outer = vec![
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(4.0, 4.0),
        Point2::new(0.0, 4.0),
    ];
    let opts = RefinementOptions {
        max_edge_length: Some(1.0),
        min_angle_deg: Some(20.0),
    };
    // The refinement inserts Steiner points: the function therefore returns
    // **the points** (input + Steiner) *and* the triangles indexing them.
    let (points, triangles) = triangulate_polygon_with_holes_refined(&outer, &[], opts).unwrap();
    println!(
        "{} triangles après raffinement, {} points",
        triangles.len(),
        points.len()
    );
    assert!(points.len() > outer.len());
}
// ANCHOR_END: raffinement
