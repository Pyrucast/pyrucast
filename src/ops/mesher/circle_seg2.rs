use crate::error::{PyrucastError, Result};
use crate::containers::mesh::ElementType;
use crate::containers::mesh::Node;
use crate::containers::mesh::{Mesh, SubMesh};

/// Build a closed circle of `n_elems` SEG2 elements.
///
/// The circle lies in the plane perpendicular to `normal`, centred on
/// `center`, with the given `radius`. `normal` must be a 3-component
/// vector (regardless of node dimension); for 2-D meshes the normal
/// should point along z (`[0, 0, ±1]`).
///
/// `n_elems` must be ≥ 3 and `radius` must be > 0. `n_elems` nodes are
/// created at evenly spaced angles; the center node itself is **not**
/// included in the mesh. The first and last elements share node 0,
/// closing the loop.
///
/// The in-plane basis `(u, v)` is built by Gram-Schmidt against the
/// least-aligned coordinate axis so that `(u, v, n̂)` is right-handed.
pub fn circle_seg2(center: &Node, normal: &[f64], radius: f64, n_elems: usize) -> Result<Mesh> {
    use std::f64::consts::PI;

    if n_elems < 3 {
        return Err(PyrucastError::Message(
            "circle_seg2: n_elems must be ≥ 3".into(),
        ));
    }
    if radius <= 0.0 {
        return Err(PyrucastError::Message(
            "circle_seg2: radius must be > 0".into(),
        ));
    }
    if normal.len() != 3 {
        return Err(PyrucastError::Message(
            "circle_seg2: normal must have exactly 3 components".into(),
        ));
    }

    let cfg = center.configuration();
    let center_coords = center.coord()?;
    let dim = center_coords.len();
    if !(2..=3).contains(&dim) {
        return Err(PyrucastError::Message(
            "circle_seg2: node dimension must be 2 or 3".into(),
        ));
    }

    use crate::containers::mesh::Vector3;
    use crate::ops::mesher::triangulation::in_plane_basis;
    let n_vec = Vector3::new(normal[0], normal[1], normal[2]);
    if n_vec.norm() < 1e-15 {
        return Err(PyrucastError::Message(
            "circle_seg2: normal vector must not be zero".into(),
        ));
    }
    let n = n_vec.normalize();
    let (u, v) = in_plane_basis(n);

    let centre = Vector3::new(
        center_coords.first().copied().unwrap_or(0.0),
        center_coords.get(1).copied().unwrap_or(0.0),
        center_coords.get(2).copied().unwrap_or(0.0),
    );
    let mut nodes: Vec<Node> = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        let theta = 2.0 * PI * i as f64 / n_elems as f64;
        let p3 = centre + radius * (theta.cos() * u + theta.sin() * v);
        nodes.push(Node::create_in(cfg.clone(), &p3.as_slice()[..dim])?);
    }

    let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
    for i in 0..n_elems {
        mesh.add_cell(&[nodes[i].id(), nodes[(i + 1) % n_elems].id()])?;
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    #[test]
    fn circle_seg2_basic_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mesh = circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 4).unwrap();

        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 4);

        let n0 = mesh.node(0, 0, 0).unwrap();
        assert!((n0.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n0.coord().unwrap()[1]).abs() < 1e-12);
        let n1 = mesh.node(0, 1, 0).unwrap();
        assert!((n1.coord().unwrap()[0]).abs() < 1e-12);
        assert!((n1.coord().unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn circle_seg2_closed_loop() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mesh = circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 6).unwrap();

        let last_end = mesh.node(0, 5, 1).unwrap();
        let first_start = mesh.node(0, 0, 0).unwrap();
        assert_eq!(last_end.id(), first_start.id());
    }

    #[test]
    fn circle_seg2_radius_and_center_offset() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[1.0, 2.0]).unwrap();
        let mesh = circle_seg2(&center, &[0.0, 0.0, 1.0], 3.0, 8).unwrap();

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().coord().unwrap();
            let dist = ((c[0] - 1.0).powi(2) + (c[1] - 2.0).powi(2)).sqrt();
            assert!((dist - 3.0).abs() < 1e-10, "element {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_seg2_3d_xz_plane() {
        let cfg = insert(Configuration::new(3).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let mesh = circle_seg2(&center, &[0.0, 1.0, 0.0], 2.0, 8).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 8);

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().coord().unwrap();
            assert!((c[1]).abs() < 1e-12, "element {ei}: y={}", c[1]);
            let dist = (c[0].powi(2) + c[2].powi(2)).sqrt();
            assert!((dist - 2.0).abs() < 1e-10, "element {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_seg2_rejects_too_few_elements() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle_seg2(&center, &[0.0, 0.0, 1.0], 1.0, 2).is_err());
    }

    #[test]
    fn circle_seg2_rejects_nonpositive_radius() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle_seg2(&center, &[0.0, 0.0, 1.0], 0.0, 4).is_err());
        assert!(circle_seg2(&center, &[0.0, 0.0, 1.0], -1.0, 4).is_err());
    }

    #[test]
    fn circle_seg2_rejects_zero_normal() {
        let cfg = insert(Configuration::new(2).unwrap());
        let center = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle_seg2(&center, &[0.0, 0.0, 0.0], 1.0, 4).is_err());
    }
}
