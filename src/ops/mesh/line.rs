use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};

/// Build a mesh of `n_elems` elements along the straight line from node
/// `a` to node `b`, of the given `element_type` (`SEG2` or `SEG3`).
///
/// Both nodes must belong to the same `Coords` and have the same
/// coordinate dimension. `n_elems` must be ≥ 1.
///
/// The two endpoint nodes are re-used (their refcount is incremented);
/// `n_elems - 1` intermediate corner nodes are created at evenly spaced
/// positions. For `SEG3`, the result is then promoted to quadratic (one
/// mid-edge node per element) via [`super::to_quadratic`].
pub fn line(a: &Node, b: &Node, n_elems: usize, element_type: ElementType) -> Result<Mesh> {
    if !matches!(element_type, ElementType::SEG2 | ElementType::SEG3) {
        return Err(PyrucastError::Message(format!(
            "line: unsupported element type {element_type} (expected SEG2 or SEG3)"
        )));
    }
    if n_elems == 0 {
        return Err(PyrucastError::Message("line: n_elems must be ≥ 1".into()));
    }
    let coords = a.coords();
    let coords_b = b.coords();
    if coords.index() != coords_b.index() || coords.generation() != coords_b.generation() {
        return Err(PyrucastError::Message(
            "line: nodes belong to different Coords".into(),
        ));
    }
    let coords_a = a.position()?;
    let coords_b = b.position()?;
    if coords_a.len() != coords_b.len() {
        return Err(PyrucastError::Message(
            "line: nodes have incompatible dimensions".into(),
        ));
    }

    // n_elems+1 nodes: a, n_elems-1 intermediate, b.
    let mut nodes: Vec<Node> = Vec::with_capacity(n_elems + 1);
    nodes.push(Node::acquire(coords.clone(), a.id())?);
    for i in 1..n_elems {
        let t = i as f64 / n_elems as f64;
        let coord: Vec<f64> = coords_a
            .iter()
            .zip(coords_b.iter())
            .map(|(&ca, &cb)| ca + t * (cb - ca))
            .collect();
        nodes.push(Node::create_in(coords.clone(), &coord)?);
    }
    nodes.push(Node::acquire(coords.clone(), b.id())?);

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
    use crate::store::insert;

    #[test]
    fn line_basic() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[6.0]).unwrap();

        let mesh = line(&a, &b, 3, ElementType::SEG2).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 3);

        let n00 = mesh.node(0, 0, 0).unwrap();
        let n10 = mesh.node(0, 1, 0).unwrap();
        let n20 = mesh.node(0, 2, 0).unwrap();
        assert_eq!(n00.position().unwrap(), vec![0.0]);
        assert!((n10.position().unwrap()[0] - 2.0).abs() < 1e-12);
        assert!((n20.position().unwrap()[0] - 4.0).abs() < 1e-12);

        // last node of the last cell = node b
        let n21 = mesh.node(0, 2, 1).unwrap();
        assert_eq!(n21.id(), b.id());
    }

    #[test]
    fn line_one_element() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();

        let mesh = line(&a, &b, 1, ElementType::SEG2).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(mesh.node(0, 0, 1).unwrap().id(), b.id());
    }

    #[test]
    fn line_zero_elems_is_error() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        assert!(line(&a, &b, 0, ElementType::SEG2).is_err());
    }

    #[test]
    fn line_seg3_basic() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[4.0]).unwrap();

        let mesh = line(&a, &b, 2, ElementType::SEG3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG3]);
        assert_eq!(mesh.cell_count().unwrap(), 2);

        // Corners at 0, 2, 4; mid-edge nodes at 1 and 3.
        let n00 = mesh.node(0, 0, 0).unwrap();
        let n01 = mesh.node(0, 0, 1).unwrap();
        let n02 = mesh.node(0, 0, 2).unwrap();
        assert_eq!(n00.position().unwrap(), vec![0.0]);
        assert!((n01.position().unwrap()[0] - 2.0).abs() < 1e-12);
        assert!((n02.position().unwrap()[0] - 1.0).abs() < 1e-12);

        let n11 = mesh.node(0, 1, 1).unwrap();
        assert_eq!(n11.id(), b.id());
    }

    #[test]
    fn line_unsupported_element_type_is_error() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        assert!(line(&a, &b, 1, ElementType::TRI3).is_err());
    }
}
