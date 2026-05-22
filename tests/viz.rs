//! Integration tests for the visualization layer.
//!
//! Compiled and run only when the `viz` feature is on:
//! `cargo test --features viz --test viz`.

#![cfg(feature = "viz")]

use pyrucast::color::RgbColor;
use pyrucast::configuration::Configuration;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::node::Node;
use pyrucast::store::insert;
use pyrucast::viz::View;

fn tmpdir() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "pyrucast-viz-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn build_two_triangles() -> SubMesh {
    let cfg = insert(Configuration::new(3).unwrap());
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
    let c = Node::create_in(cfg.clone(), &[1.0, 1.0, 0.0]).unwrap();
    let d = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.5]).unwrap();
    let mut sm = SubMesh::new(cfg, ElementType::TRI3);
    sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
    sm.add_cell(&[a.id(), c.id(), d.id()]).unwrap();
    sm
}

#[test]
fn submesh_exports_png() {
    let sm = build_two_triangles();
    let dir = tmpdir();
    let path = dir.join("tri.png");
    sm.plot(Some(View::iso()), Some(&path)).unwrap();
    let meta = std::fs::metadata(&path).unwrap();
    assert!(meta.len() > 0, "PNG should not be empty");
}

#[test]
fn submesh_exports_svg() {
    let sm = build_two_triangles();
    let dir = tmpdir();
    let path = dir.join("tri.svg");
    sm.plot(Some(View::front()), Some(&path)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(!bytes.is_empty(), "SVG should not be empty");
    let text = String::from_utf8(bytes).expect("SVG should be valid UTF-8");
    assert!(text.contains("<svg"), "should look like SVG");
}

#[test]
fn submesh_default_view_is_iso() {
    let sm = build_two_triangles();
    let dir = tmpdir();
    let path = dir.join("tri-default.png");
    sm.plot(None, Some(&path)).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
}

#[test]
fn submesh_renders_every_element_type() {
    use pyrucast::configuration::NodeId;
    let cfg = insert(Configuration::new(3).unwrap());
    let n: Vec<_> = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .iter()
    .map(|c| Node::create_in(cfg.clone(), c).unwrap())
    .collect();

    let cases: Vec<(ElementType, Vec<NodeId>)> = vec![
        (ElementType::POI1, vec![n[0].id()]),
        (ElementType::SEG2, vec![n[0].id(), n[1].id()]),
        (ElementType::TRI3, vec![n[0].id(), n[1].id(), n[2].id()]),
        (
            ElementType::QUA4,
            vec![n[0].id(), n[1].id(), n[2].id(), n[3].id()],
        ),
        (
            ElementType::TET4,
            vec![n[0].id(), n[1].id(), n[2].id(), n[4].id()],
        ),
        (ElementType::HEX8, n.iter().map(|nn| nn.id()).collect()),
    ];

    let dir = tmpdir();
    for (et, ids) in cases {
        let mut sm = SubMesh::new(cfg.clone(), et);
        sm.add_cell(&ids).unwrap();
        let path = dir.join(format!("{}.png", et.name().to_ascii_lowercase()));
        sm.plot(Some(View::iso()), Some(&path))
            .unwrap_or_else(|e| panic!("{}: {}", et, e));
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 0, "{}: PNG should not be empty", et);
    }
}

#[test]
fn unsupported_extension_errors() {
    let sm = build_two_triangles();
    let dir = tmpdir();
    let path = dir.join("tri.jpg");
    let err = sm.plot(None, Some(&path)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("png") || msg.contains("svg"),
        "error should mention supported extensions, got: {msg}"
    );
}

#[test]
fn mesh_plot_renders_each_submesh_with_its_color() {
    let cfg = insert(Configuration::new(3).unwrap());
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
    let c = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.0]).unwrap();
    let d = Node::create_in(cfg.clone(), &[2.0, 0.0, 0.0]).unwrap();
    let e = Node::create_in(cfg.clone(), &[2.0, 1.0, 0.0]).unwrap();

    let sm_red_handle = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.set_face_color(RgbColor::new(220, 60, 60));
        insert(sm)
    };
    let sm_blue_handle = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[b.id(), d.id(), e.id()]).unwrap();
        sm.set_face_color(RgbColor::new(60, 60, 220));
        insert(sm)
    };

    let mut mesh = Mesh::new(cfg);
    mesh.add_submesh(sm_red_handle).unwrap();
    mesh.add_submesh(sm_blue_handle).unwrap();

    let dir = tmpdir();
    let path = dir.join("mesh.svg");
    mesh.plot(None, Some(&path)).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    // The SVG embeds the fill colours as hex; both faces should appear.
    let lower = text.to_ascii_lowercase();
    assert!(
        lower.contains("dc3c3c") || lower.contains("rgb(220,60,60)"),
        "red face colour should appear in SVG"
    );
    assert!(
        lower.contains("3c3cdc") || lower.contains("rgb(60,60,220)"),
        "blue face colour should appear in SVG"
    );
}

#[test]
fn mesh_renders_mixed_element_types() {
    let cfg = insert(Configuration::new(2).unwrap());
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
    let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

    let sm_pts = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        insert(sm)
    };
    let sm_tri = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        insert(sm)
    };

    let mut mesh = Mesh::new(cfg);
    mesh.add_submesh(sm_pts).unwrap();
    mesh.add_submesh(sm_tri).unwrap();

    let dir = tmpdir();
    let path = dir.join("mixed.png");
    mesh.plot(None, Some(&path)).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
}

#[test]
fn face_color_roundtrip_on_submesh() {
    let cfg = insert(Configuration::new(2).unwrap());
    let sm_handle = insert(SubMesh::new(cfg, ElementType::TRI3));
    pyrucast::store::with_mut(&sm_handle, |s| {
        s.set_face_color(RgbColor::new(1, 2, 3));
    })
    .unwrap();
    let c = pyrucast::store::with(&sm_handle, |s| s.face_color()).unwrap();
    assert_eq!(c, RgbColor::new(1, 2, 3));
}
