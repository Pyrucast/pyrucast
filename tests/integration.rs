//! Integration test: verifies that pyrucast's public API is usable from
//! an external crate (Phase 0 + Phase 1 + Phase 2).

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::element_type::ElementType;
use pyrucast::atoms::node::Node;
use pyrucast::atoms::NodeId;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::persist::Persist;
use pyrucast::store::Handle;
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
    let e = PyrucastError::MeshSealed;
    assert!(e.to_string().contains("sealed"));
}

#[test]
fn version_exposed() {
    assert_eq!(pyrucast::VERSION, env!("CARGO_PKG_VERSION"));
}

// ─── Handle integration tests ───────────────────────────────────────────────

#[derive(Debug)]
struct IntegrationSample {
    values: Vec<f64>,
    name: String,
}

#[test]
fn full_handle_cycle_via_public_api() {
    let h = Handle::new(IntegrationSample {
        values: vec![1.0, 2.0, 3.0],
        name: "alpha".into(),
    });
    {
        let e = h.read();
        assert_eq!(e.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.name, "alpha");
    }

    h.write().values.push(4.0);
    assert_eq!(h.read().values.len(), 4);

    // A clone names the same object; the object outlives the first handle.
    let g = h.clone();
    assert!(g.same_object(&h));
    drop(h);
    assert_eq!(g.read().name, "alpha");
}

// ─── Coords + Node integration tests (Phase 2) ───────────────────────

#[test]
fn coords_cycle_via_store() -> Result<()> {
    let h = Handle::new(Coords::new(2)?);
    let a: NodeId = h.write().add_node(&[0.0, 0.0])?;
    let b: NodeId = h.write().add_node(&[1.0, 0.0])?;

    {
        let c = h.read();
        assert_eq!(c.node_count(), 2);
        assert!(c.is_alive(a));
        assert!(c.is_alive(b));
    }

    // Initial refcount = 1 ⇒ gc collects nothing.
    assert_eq!(h.write().gc(), 0);

    // decrefs + gc collects both.
    {
        let mut c = h.write();
        c.decref(a).unwrap();
        c.decref(b).unwrap();
        assert_eq!(c.gc(), 2);
    }

    {
        let c = h.read();
        assert!(!c.is_alive(a));
        assert!(!c.is_alive(b));
    }
    Ok(())
}

#[test]
fn node_protects_from_gc() -> Result<()> {
    let h = Handle::new(Coords::new(2)?);
    let n = Node::create_in(h.clone(), &[3.0, 4.0])?;
    let id = n.id();
    assert_eq!(n.position()?, vec![3.0, 4.0]);

    // Cloning shares the id and doubles the refcount.
    let m = n.clone();
    assert_eq!(h.read().refcount(id), 2);
    drop(n);
    assert_eq!(h.read().refcount(id), 1);
    assert_eq!(h.write().gc(), 0);
    drop(m);
    assert_eq!(h.write().gc(), 1);
    Ok(())
}

// ─── Mesh / SubMesh integration tests (Phase 2 step 2) ──────────────────────

#[test]
fn submesh_protects_nodes_via_refcount() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let cc = Node::create_in(coords.clone(), &[0.5, 1.0])?;

    let sm_handle = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), cc.id()])?;
        Handle::new(sm)
    };

    // Nodes AND SubMesh each hold one ref ⇒ refcount = 2.
    {
        let c = coords.read();
        assert_eq!(c.refcount(a.id()), 2);
        assert_eq!(c.refcount(b.id()), 2);
        assert_eq!(c.refcount(cc.id()), 2);
    }

    // Drop the user Nodes: only the SubMesh keeps referencing.
    let (ida, idb, idc) = (a.id(), b.id(), cc.id());
    drop(a);
    drop(b);
    drop(cc);
    {
        let c = coords.read();
        assert_eq!(c.refcount(ida), 1);
        assert_eq!(c.refcount(idb), 1);
        assert_eq!(c.refcount(idc), 1);
    }
    // gc must still collect nothing.
    assert_eq!(coords.write().gc(), 0);

    // Drop the SubMesh ⇒ all nodes drop to 0 ⇒ gc collects.
    drop(sm_handle);
    assert_eq!(coords.write().gc(), 3);
    Ok(())
}

#[test]
fn triangulate_surface_from_circle_contour() -> Result<()> {
    // Build a closed SEG2 circle (8 segments, CCW) and mesh its interior
    // with TRI3 through the public API. The constrained-Delaunay mesher
    // creates interior nodes, so it yields more than the boundary-only 6
    // triangles, while conserving the inscribed octagon's area.
    let coords = Handle::new(Coords::new(2)?);
    let center = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let circle = pyrucast::ops::mesh::circle(&center, &[0.0, 0.0, 1.0], 1.0, 8, ElementType::SEG2)?;
    let tri = pyrucast::ops::mesh::triangulate_surface(&circle, ElementType::TRI3, Some(0.25))?;
    assert_eq!(tri.element_types()?, vec![ElementType::TRI3]);
    assert!(tri.cell_count()? > 6, "expected interior nodes to be added");

    // Sum of signed triangle areas equals the inscribed octagon's area
    // (= 8 · 0.5 · r² · sin(2π/8) = 2√2 ≈ 2.8284), whatever the refinement.
    let mut total = 0.0;
    for ci in 0..tri.cell_count()? {
        let p0 = tri.node(0, ci, 0)?.position()?;
        let p1 = tri.node(0, ci, 1)?.position()?;
        let p2 = tri.node(0, ci, 2)?.position()?;
        let area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
        assert!(area > 0.0, "triangle {} not CCW", ci);
        total += area;
    }
    let expected = 2.0 * std::f64::consts::SQRT_2;
    assert!(
        (total - expected).abs() < 1e-9,
        "total area {} ≠ inscribed octagon area {}",
        total,
        expected
    );
    Ok(())
}

#[test]
fn mesh_composed_of_multiple_submeshes() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let cc = Node::create_in(coords.clone(), &[0.5, 1.0])?;

    let sm_pts = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()])?;
        sm.add_cell(&[b.id()])?;
        sm.add_cell(&[cc.id()])?;
        Handle::new(sm)
    };
    let sm_tri = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), cc.id()])?;
        Handle::new(sm)
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
