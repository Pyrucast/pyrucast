use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};

/// Build an open arc of `n_elems` elements from `node_a` to `node_b`,
/// following the circle centred on `center` that passes through both, of
/// the given `element_type` (`SEG2` or `SEG3`).
///
/// `node_a` and `node_b` must be equidistant from `center` (within
/// relative tolerance) and not aligned with it, so that the plane of the
/// arc (`center`, `node_a`, `node_b`) is well defined. The shorter of the
/// two arcs joining them is built.
///
/// Both endpoint nodes are re-used (their refcount is incremented);
/// `n_elems - 1` intermediate corner nodes are created at evenly spaced
/// angles. For `SEG3`, each element also gets its mid-node at the half
/// angle — **on the arc**, not on the chord.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(3).unwrap());
/// # let p = |x: &[f64]| Node::create_in(coords.clone(), x).unwrap();
/// // A quarter circle: `node_a` and `node_b` are on the arc, `center` at the
/// // centre. The radius comes from the distance, not from an argument.
/// let a = mesh::arc(&p(&[1.0, 0.0, 0.0]), &p(&[0.0, 0.0, 0.0]),
///                   &p(&[0.0, 1.0, 0.0]), 4, ElementType::SEG2)?;
/// assert_eq!(a.cell_count(), 4);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn arc(
    node_a: &Node,
    center: &Node,
    node_b: &Node,
    n_elems: usize,
    element_type: ElementType,
) -> Result<Mesh> {
    if !matches!(element_type, ElementType::SEG2 | ElementType::SEG3) {
        return Err(PyrucastError::Message(format!(
            "arc: unsupported element type {element_type} (expected SEG2 or SEG3)"
        )));
    }
    if n_elems == 0 {
        return Err(PyrucastError::Message("arc: n_elems must be ≥ 1".into()));
    }

    let coords = center.coords();
    let coords_a = node_a.coords();
    let coords_b = node_b.coords();
    if !coords.same_object(&coords_a) || !coords.same_object(&coords_b) {
        return Err(PyrucastError::Message(
            "arc: nodeA, center and nodeB belong to different Coords".into(),
        ));
    }

    let a = node_a.position()?;
    let c = center.position()?;
    let b = node_b.position()?;
    let dim = c.len();
    if a.len() != dim || b.len() != dim {
        return Err(PyrucastError::Message(
            "arc: nodes have incompatible dimensions".into(),
        ));
    }
    if !(2..=3).contains(&dim) {
        return Err(PyrucastError::Message(
            "arc: node dimension must be 2 or 3".into(),
        ));
    }

    use crate::atoms::Vector3;
    let to_v3 = |p: &[f64]| {
        Vector3::new(
            p.first().copied().unwrap_or(0.0),
            p.get(1).copied().unwrap_or(0.0),
            p.get(2).copied().unwrap_or(0.0),
        )
    };
    let centre = to_v3(&c);
    let va = to_v3(&a) - centre;
    let vb = to_v3(&b) - centre;
    let radius_a = va.norm();
    let radius_b = vb.norm();
    if radius_a < 1e-12 || radius_b < 1e-12 {
        return Err(PyrucastError::Message(
            "arc: nodeA and nodeB must not coincide with center".into(),
        ));
    }
    if (radius_a - radius_b).abs() > 1e-9 * radius_a.max(radius_b) {
        return Err(PyrucastError::Message(format!(
            "arc: nodeA and nodeB are not equidistant from center (radii {radius_a} and {radius_b})"
        )));
    }

    let normal = va.cross(&vb);
    let normal_norm = normal.norm();
    if normal_norm < 1e-9 * radius_a * radius_b {
        return Err(PyrucastError::Message(
            "arc: nodeA, center and nodeB are colinear (arc plane is undefined)".into(),
        ));
    }
    let n_hat = normal / normal_norm;
    let u = va / radius_a;
    let v = n_hat.cross(&u);
    let theta = (va.dot(&vb) / (radius_a * radius_b))
        .clamp(-1.0, 1.0)
        .acos();

    // The two ends are the caller's own nodes; the intermediate corners, then
    // the SEG3 mid-nodes at the half angles, are laid out flat and created in
    // one locked pass.
    let quadratic = element_type == ElementType::SEG3;
    let n_corners = n_elems - 1;
    let n_mids = if quadratic { n_elems } else { 0 };
    let on_arc = |s: f64| centre + radius_a * ((s * theta).cos() * u + (s * theta).sin() * v);
    let mut flat: Vec<f64> = Vec::with_capacity((n_corners + n_mids) * dim);
    for i in 1..n_elems {
        flat.extend_from_slice(&on_arc(i as f64 / n_elems as f64).as_slice()[..dim]);
    }
    for i in 0..n_mids {
        flat.extend_from_slice(&on_arc((i as f64 + 0.5) / n_elems as f64).as_slice()[..dim]);
    }
    let first = coords.write().add_nodes(&flat)?.start;
    let at = |i: usize| -> NodeId {
        if i == 0 {
            node_a.id()
        } else if i == n_elems {
            node_b.id()
        } else {
            NodeId(first + (i - 1) as u32)
        }
    };
    let mid = |i: usize| NodeId(first + (n_corners + i) as u32);

    let conn: Vec<NodeId> = if quadratic {
        (0..n_elems)
            .flat_map(|i| [at(i), at(i + 1), mid(i)])
            .collect()
    } else {
        (0..n_elems).flat_map(|i| [at(i), at(i + 1)]).collect()
    };
    let sm = SubMesh::from_connectivity(coords.clone(), element_type, conn)?;
    // The arc owns the fresh nodes now; hand back the unit `add_nodes` gave.
    let owned: Vec<NodeId> = (first..first + (n_corners + n_mids) as u32)
        .map(NodeId)
        .collect();
    coords.write().decref_all(&owned)?;
    Ok(Mesh::from_submesh(sm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use std::f64::consts::PI;

    #[test]
    fn arc_quarter_circle_2d() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        let mesh = arc(&a, &center, &b, 3, ElementType::SEG2).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count(), 3);

        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(mesh.node(0, 2, 1).unwrap().id(), b.id());

        // All nodes lie on the unit circle.
        for ei in 0..3 {
            for corner in 0..2 {
                let p = mesh.node(0, ei, corner).unwrap().position().unwrap();
                let dist = (p[0].powi(2) + p[1].powi(2)).sqrt();
                assert!((dist - 1.0).abs() < 1e-12);
            }
        }

        // Midpoint node (t=1/3, 2/3) angles: 30° and 60°.
        let mid1 = mesh.node(0, 1, 0).unwrap().position().unwrap();
        assert!((mid1[0] - (PI / 6.0).cos()).abs() < 1e-12);
        assert!((mid1[1] - (PI / 6.0).sin()).abs() < 1e-12);
    }

    #[test]
    fn arc_reuses_endpoint_nodes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[-2.0 * 0.5, 2.0 * 0.8660254037844387]).unwrap();

        let mesh = arc(&a, &center, &b, 1, ElementType::SEG2).unwrap();
        assert_eq!(mesh.cell_count(), 1);
        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(mesh.node(0, 0, 1).unwrap().id(), b.id());
    }

    #[test]
    fn arc_seg3_promotes_to_quadratic() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        let mesh = arc(&a, &center, &b, 2, ElementType::SEG3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG3]);
        assert_eq!(mesh.cell_count(), 2);
    }

    #[test]
    fn arc_seg3_mid_nodes_lie_on_the_arc_at_the_half_angle() {
        // A quarter of the unit circle in 2 SEG3: the mid-nodes sit at 22.5°
        // and 67.5°, on the circle — a chord midpoint would be at radius
        // cos(22.5°) ≈ 0.924.
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        let mesh = arc(&a, &center, &b, 2, ElementType::SEG3).unwrap();
        for (cell, angle) in [(0, PI / 8.0), (1, 3.0 * PI / 8.0)] {
            let m = mesh.node(0, cell, 2).unwrap().position().unwrap();
            assert!((m[0] - angle.cos()).abs() < 1e-12, "{m:?}");
            assert!((m[1] - angle.sin()).abs() < 1e-12, "{m:?}");
        }
        // The corners are shared between consecutive cells.
        assert_eq!(
            mesh.node(0, 0, 1).unwrap().id(),
            mesh.node(0, 1, 0).unwrap().id()
        );
        assert_eq!(mesh.node(0, 1, 1).unwrap().id(), b.id());
    }

    #[test]
    fn arc_rejects_unequal_radii() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();
        assert!(arc(&a, &center, &b, 3, ElementType::SEG2).is_err());
    }

    #[test]
    fn arc_rejects_colinear_points() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[-1.0, 0.0]).unwrap();
        assert!(arc(&a, &center, &b, 3, ElementType::SEG2).is_err());
    }

    #[test]
    fn arc_rejects_coincident_with_center() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        assert!(arc(&a, &center, &b, 3, ElementType::SEG2).is_err());
    }

    #[test]
    fn arc_rejects_zero_elems() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        assert!(arc(&a, &center, &b, 0, ElementType::SEG2).is_err());
    }

    #[test]
    fn arc_3d_plane() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let a = Node::create_in(coords.clone(), &[2.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[0.0, 0.0, 2.0]).unwrap();

        let mesh = arc(&a, &center, &b, 4, ElementType::SEG2).unwrap();
        for ei in 0..4 {
            let p = mesh.node(0, ei, 0).unwrap().position().unwrap();
            assert!((p[1]).abs() < 1e-12);
            let dist = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
            assert!((dist - 2.0).abs() < 1e-10);
        }
    }
}
