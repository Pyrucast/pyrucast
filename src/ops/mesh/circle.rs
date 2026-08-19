use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};

/// Build a closed circle of `n_elems` elements, of the given
/// `element_type` (`SEG2` or `SEG3`).
///
/// The circle lies in the plane perpendicular to `normal`, centred on
/// `center`, with the given `radius`. `normal` must be a 3-component
/// vector (regardless of node dimension); for 2-D meshes the normal
/// should point along z (`[0, 0, ±1]`).
///
/// `n_elems` must be ≥ 3 and `radius` must be > 0. `n_elems` nodes are
/// created at evenly spaced angles; the center node itself is **not**
/// included in the mesh. The first and last elements share node 0,
/// closing the loop. For `SEG3`, the result is then promoted to
/// quadratic (one mid-edge node per element) via [`super::to_quadratic`].
///
/// The in-plane basis `(u, v)` is built by Gram-Schmidt against the
/// least-aligned coordinate axis so that `(u, v, n̂)` is right-handed.
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
/// // Un cercle **fermé** : autant de nœuds que de mailles, le dernier
/// // rejoignant le premier.
/// let c = mesh::circle(&p(&[0.0, 0.0, 0.0]), &[0.0, 0.0, 1.0], 1.0, 8,
///                      ElementType::SEG2)?;
/// assert_eq!(c.cell_count()?, 8);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn circle(
    center: &Node,
    normal: &[f64],
    radius: f64,
    n_elems: usize,
    element_type: ElementType,
) -> Result<Mesh> {
    use std::f64::consts::PI;

    if !matches!(element_type, ElementType::SEG2 | ElementType::SEG3) {
        return Err(PyrucastError::Message(format!(
            "circle: unsupported element type {element_type} (expected SEG2 or SEG3)"
        )));
    }
    if n_elems < 3 {
        return Err(PyrucastError::Message("circle: n_elems must be ≥ 3".into()));
    }
    if radius <= 0.0 {
        return Err(PyrucastError::Message("circle: radius must be > 0".into()));
    }
    if normal.len() != 3 {
        return Err(PyrucastError::Message(
            "circle: normal must have exactly 3 components".into(),
        ));
    }

    let coords = center.coords();
    let center_coords = center.position()?;
    let dim = center_coords.len();
    if !(2..=3).contains(&dim) {
        return Err(PyrucastError::Message(
            "circle: node dimension must be 2 or 3".into(),
        ));
    }

    use crate::atoms::Vector3;
    use crate::ops::mesh::triangulation::in_plane_basis;
    let n_vec = Vector3::new(normal[0], normal[1], normal[2]);
    if n_vec.norm() < 1e-15 {
        return Err(PyrucastError::Message(
            "circle: normal vector must not be zero".into(),
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
        nodes.push(Node::create_in(coords.clone(), &p3.as_slice()[..dim])?);
    }

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    for i in 0..n_elems {
        mesh.add_cell(&[nodes[i].id(), nodes[(i + 1) % n_elems].id()])?;
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

    #[test]
    fn circle_basic_2d() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mesh = circle(&center, &[0.0, 0.0, 1.0], 1.0, 4, ElementType::SEG2).unwrap();

        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 4);

        let n0 = mesh.node(0, 0, 0).unwrap();
        assert!((n0.position().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n0.position().unwrap()[1]).abs() < 1e-12);
        let n1 = mesh.node(0, 1, 0).unwrap();
        assert!((n1.position().unwrap()[0]).abs() < 1e-12);
        assert!((n1.position().unwrap()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn circle_closed_loop() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mesh = circle(&center, &[0.0, 0.0, 1.0], 1.0, 6, ElementType::SEG2).unwrap();

        let last_end = mesh.node(0, 5, 1).unwrap();
        let first_start = mesh.node(0, 0, 0).unwrap();
        assert_eq!(last_end.id(), first_start.id());
    }

    #[test]
    fn circle_radius_and_center_offset() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[1.0, 2.0]).unwrap();
        let mesh = circle(&center, &[0.0, 0.0, 1.0], 3.0, 8, ElementType::SEG2).unwrap();

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().position().unwrap();
            let dist = ((c[0] - 1.0).powi(2) + (c[1] - 2.0).powi(2)).sqrt();
            assert!((dist - 3.0).abs() < 1e-10, "element {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_3d_xz_plane() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let mesh = circle(&center, &[0.0, 1.0, 0.0], 2.0, 8, ElementType::SEG2).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 8);

        for ei in 0..8 {
            let c = mesh.node(0, ei, 0).unwrap().position().unwrap();
            assert!((c[1]).abs() < 1e-12, "element {ei}: y={}", c[1]);
            let dist = (c[0].powi(2) + c[2].powi(2)).sqrt();
            assert!((dist - 2.0).abs() < 1e-10, "element {ei}: distance={dist}");
        }
    }

    #[test]
    fn circle_seg3_promotes_to_quadratic() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mesh = circle(&center, &[0.0, 0.0, 1.0], 1.0, 6, ElementType::SEG3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG3]);
        assert_eq!(mesh.cell_count().unwrap(), 6);
    }

    #[test]
    fn circle_rejects_too_few_elements() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle(&center, &[0.0, 0.0, 1.0], 1.0, 2, ElementType::SEG2).is_err());
    }

    #[test]
    fn circle_rejects_nonpositive_radius() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle(&center, &[0.0, 0.0, 1.0], 0.0, 4, ElementType::SEG2).is_err());
        assert!(circle(&center, &[0.0, 0.0, 1.0], -1.0, 4, ElementType::SEG2).is_err());
    }

    #[test]
    fn circle_rejects_zero_normal() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle(&center, &[0.0, 0.0, 0.0], 1.0, 4, ElementType::SEG2).is_err());
    }

    #[test]
    fn circle_rejects_unsupported_element_type() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let center = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        assert!(circle(&center, &[0.0, 0.0, 1.0], 1.0, 4, ElementType::TRI3).is_err());
    }
}
