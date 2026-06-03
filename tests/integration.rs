//! Integration test: verifies that pyrucast's public API is usable from
//! an external crate (Phase 0 + Phase 1 + Phase 2).

use pyrucast::aggregate::Aggregate;
use pyrucast::containers::mesh::configuration::{Configuration, NodeId};
use pyrucast::containers::mesh::element_type::ElementType;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::mesh::node::Node;
use pyrucast::persist::Persist;
use pyrucast::store::{compact, insert, live_count, swap_out, with, with_mut};
use pyrucast::{PyrucastError, Result};

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct Point {
    x: f64,
    y: f64,
}

#[test]
fn persist_roundtrip_via_public_api() -> Result<()> {
    let a = Point { x: 1.5, y: -2.0 };
    let bytes = a.to_bytes()?;
    let b = Point::from_bytes(&bytes)?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn human_display_for_error() {
    let e = PyrucastError::Message("test".into());
    assert_eq!(e.to_string(), "test");
    let e = PyrucastError::StaleHandle;
    assert!(e.to_string().contains("stale handle"));
}

#[test]
fn version_exposed() {
    assert_eq!(pyrucast::VERSION, env!("CARGO_PKG_VERSION"));
}

// ─── Store integration tests (Phase 1) ──────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct IntegrationSample {
    values: Vec<f64>,
    name: String,
}

#[test]
fn full_store_cycle_via_public_api() -> Result<()> {
    let h = insert(IntegrationSample {
        values: vec![1.0, 2.0, 3.0],
        name: "alpha".into(),
    });
    with(&h, |e| {
        assert_eq!(e.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.name, "alpha");
    })?;

    with_mut(&h, |e| e.values.push(4.0))?;
    with(&h, |e| assert_eq!(e.values.len(), 4))?;

    swap_out(&h)?;
    with(&h, |e| assert_eq!(e.name, "alpha"))?;

    drop(h);
    assert_eq!(live_count::<IntegrationSample>(), 0);
    compact::<IntegrationSample>();
    Ok(())
}

// ─── Configuration + Node integration tests (Phase 2) ───────────────────────

#[test]
fn configuration_cycle_via_store() -> Result<()> {
    let h = insert(Configuration::new(2)?);
    let a: NodeId = with_mut(&h, |c| c.add_node(&[0.0, 0.0]))??;
    let b: NodeId = with_mut(&h, |c| c.add_node(&[1.0, 0.0]))??;

    with(&h, |c| {
        assert_eq!(c.node_count(), 2);
        assert!(c.is_alive(a));
        assert!(c.is_alive(b));
    })?;

    // Initial refcount = 1 ⇒ gc collects nothing.
    with_mut(&h, |c| assert_eq!(c.gc(), 0))?;

    // decrefs + gc collects both.
    with_mut(&h, |c| {
        c.decref(a).unwrap();
        c.decref(b).unwrap();
        assert_eq!(c.gc(), 2);
    })?;

    with(&h, |c| {
        assert!(!c.is_alive(a));
        assert!(!c.is_alive(b));
    })?;
    Ok(())
}

#[test]
fn node_protects_from_gc() -> Result<()> {
    let h = insert(Configuration::new(2)?);
    let n = Node::create_in(h.clone(), &[3.0, 4.0])?;
    let id = n.id();
    assert_eq!(n.coord()?, vec![3.0, 4.0]);

    // Cloning shares the id and doubles the refcount.
    let m = n.clone();
    with(&h, |c| assert_eq!(c.refcount(id), 2))?;
    drop(n);
    with(&h, |c| assert_eq!(c.refcount(id), 1))?;
    with_mut(&h, |c| assert_eq!(c.gc(), 0))?;
    drop(m);
    with_mut(&h, |c| assert_eq!(c.gc(), 1))?;
    Ok(())
}

// ─── Mesh / SubMesh integration tests (Phase 2 step 2) ──────────────────────

#[test]
fn submesh_protects_nodes_via_refcount() -> Result<()> {
    let cfg = insert(Configuration::new(2)?);
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0])?;
    let cc = Node::create_in(cfg.clone(), &[0.5, 1.0])?;

    let sm_handle = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), cc.id()])?;
        insert(sm)
    };

    // Nodes AND SubMesh each hold one ref ⇒ refcount = 2.
    with(&cfg, |c| {
        assert_eq!(c.refcount(a.id()), 2);
        assert_eq!(c.refcount(b.id()), 2);
        assert_eq!(c.refcount(cc.id()), 2);
    })?;

    // Drop the user Nodes: only the SubMesh keeps referencing.
    let (ida, idb, idc) = (a.id(), b.id(), cc.id());
    drop(a);
    drop(b);
    drop(cc);
    with(&cfg, |c| {
        assert_eq!(c.refcount(ida), 1);
        assert_eq!(c.refcount(idb), 1);
        assert_eq!(c.refcount(idc), 1);
    })?;
    // gc must still collect nothing.
    with_mut(&cfg, |c| assert_eq!(c.gc(), 0))?;

    // Drop the SubMesh ⇒ all nodes drop to 0 ⇒ gc collects.
    drop(sm_handle);
    with_mut(&cfg, |c| assert_eq!(c.gc(), 3))?;
    Ok(())
}

#[test]
fn fill_surface_from_circle_contour() -> Result<()> {
    // Build a closed SEG2 circle (8 segments) and fill it with TRI3
    // through the public API. The result must have 6 triangles
    // (n - 2 with n = 8) and a total area close to π·r².
    let cfg = insert(Configuration::new(2)?);
    let center = Node::create_in(cfg.clone(), &[0.0, 0.0])?;
    let circle = pyrucast::ops::mesher::circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 8)?;
    let tri = pyrucast::ops::mesher::fill_surface(&circle, ElementType::TRI3, None)?;
    assert_eq!(tri.element_types()?, vec![ElementType::TRI3]);
    assert_eq!(tri.cell_count()?, 6);

    // Sum of signed triangle areas should approximate the inscribed
    // octagon's area (= 8 · 0.5 · r² · sin(2π/8) = 2√2 ≈ 2.8284).
    let mut total = 0.0;
    for ci in 0..6 {
        let p0 = tri.node(0, ci, 0)?.coord()?;
        let p1 = tri.node(0, ci, 1)?.coord()?;
        let p2 = tri.node(0, ci, 2)?.coord()?;
        let area = 0.5
            * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        assert!(area > 0.0, "triangle {} not CCW", ci);
        total += area;
    }
    let expected = 2.0 * std::f64::consts::SQRT_2;
    assert!(
        (total - expected).abs() < 1e-10,
        "total area {} ≠ inscribed octagon area {}",
        total,
        expected
    );
    Ok(())
}

#[test]
fn mesh_composed_of_multiple_submeshes() -> Result<()> {
    let cfg = insert(Configuration::new(2)?);
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0])?;
    let cc = Node::create_in(cfg.clone(), &[0.5, 1.0])?;

    let sm_pts = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        sm.add_cell(&[cc.id()])?;
        insert(sm)
    };
    let sm_tri = {
        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), cc.id()])?;
        insert(sm)
    };

    let mesh = {
        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_pts)?;
        mesh.add_sub(sm_tri)?;
        mesh
    };

    assert_eq!(mesh.len(), 2);
    let total = mesh.cell_count()?;
    assert_eq!(total, 4); // 3 points + 1 triangle
    Ok(())
}
