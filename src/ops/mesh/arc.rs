use crate::atoms::ElementType;
use crate::atoms::Node;
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
/// angles. For `SEG3`, the result is then promoted to quadratic (one
/// mid-edge node per element) via [`super::to_quadratic`].
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

    let mut nodes: Vec<Node> = Vec::with_capacity(n_elems + 1);
    nodes.push(Node::acquire(coords.clone(), node_a.id())?);
    for i in 1..n_elems {
        let t = i as f64 / n_elems as f64;
        let phi = t * theta;
        let p3 = centre + radius_a * (phi.cos() * u + phi.sin() * v);
        nodes.push(Node::create_in(coords.clone(), &p3.as_slice()[..dim])?);
    }
    nodes.push(Node::acquire(coords.clone(), node_b.id())?);

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    for i in 0..n_elems {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }

    if element_type == ElementType::SEG3 {
        mesh = super::to_quadratic(&mesh)?;
    }
    Ok(mesh)
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
        assert_eq!(mesh.cell_count().unwrap(), 3);

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
        assert_eq!(mesh.cell_count().unwrap(), 1);
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
        assert_eq!(mesh.cell_count().unwrap(), 2);
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
