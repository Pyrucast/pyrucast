use crate::error::{PyrucastError, Result};
use crate::containers::mesh::element_type::ElementType;
use crate::containers::mesh::node::Node;
use crate::containers::mesh::Mesh;

/// Build a mesh of `n_elems` SEG2 elements along the straight line from
/// node `a` to node `b`.
///
/// Both nodes must belong to the same `Configuration` and have the same
/// coordinate dimension. `n_elems` must be ≥ 1.
///
/// The two endpoint nodes are re-used (their refcount is incremented);
/// `n_elems - 1` intermediate nodes are created at evenly spaced positions.
pub fn line_seg2(a: &Node, b: &Node, n_elems: usize) -> Result<Mesh> {
    if n_elems == 0 {
        return Err(PyrucastError::Message(
            "line_seg2: n_elems must be ≥ 1".into(),
        ));
    }
    let cfg = a.configuration();
    let cfg_b = b.configuration();
    if cfg.index() != cfg_b.index() || cfg.generation() != cfg_b.generation() {
        return Err(PyrucastError::Message(
            "line_seg2: nodes belong to different Configurations".into(),
        ));
    }
    let coords_a = a.coord()?;
    let coords_b = b.coord()?;
    if coords_a.len() != coords_b.len() {
        return Err(PyrucastError::Message(
            "line_seg2: nodes have incompatible dimensions".into(),
        ));
    }

    // n_elems+1 nodes: a, n_elems-1 intermediate, b.
    let mut nodes: Vec<Node> = Vec::with_capacity(n_elems + 1);
    nodes.push(Node::acquire(cfg.clone(), a.id())?);
    for i in 1..n_elems {
        let t = i as f64 / n_elems as f64;
        let coords: Vec<f64> = coords_a
            .iter()
            .zip(coords_b.iter())
            .map(|(&ca, &cb)| ca + t * (cb - ca))
            .collect();
        nodes.push(Node::create_in(cfg.clone(), &coords)?);
    }
    nodes.push(Node::acquire(cfg.clone(), b.id())?);

    let mut mesh = Mesh::with_element_type(cfg, ElementType::SEG2);
    for i in 0..n_elems {
        mesh.add_cell(&[nodes[i].id(), nodes[i + 1].id()])?;
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::configuration::Configuration;
    use crate::containers::mesh::element_type::ElementType;
    use crate::containers::mesh::node::Node;
    use crate::store::insert;

    #[test]
    fn line_seg2_basic() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[6.0]).unwrap();

        let mesh = line_seg2(&a, &b, 3).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(mesh.cell_count().unwrap(), 3);

        let n00 = mesh.node(0, 0, 0).unwrap();
        let n10 = mesh.node(0, 1, 0).unwrap();
        let n20 = mesh.node(0, 2, 0).unwrap();
        assert_eq!(n00.coord().unwrap(), vec![0.0]);
        assert!((n10.coord().unwrap()[0] - 2.0).abs() < 1e-12);
        assert!((n20.coord().unwrap()[0] - 4.0).abs() < 1e-12);

        // last node of the last cell = node b
        let n21 = mesh.node(0, 2, 1).unwrap();
        assert_eq!(n21.id(), b.id());
    }

    #[test]
    fn line_seg2_one_element() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();

        let mesh = line_seg2(&a, &b, 1).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        assert_eq!(mesh.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(mesh.node(0, 0, 1).unwrap().id(), b.id());
    }

    #[test]
    fn line_seg2_zero_elems_is_error() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        assert!(line_seg2(&a, &b, 0).is_err());
    }
}
