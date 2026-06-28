use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Sweep two SEG2 meshes into a QUA4 mesh by building `n_layers` layers.
///
/// Both meshes must be single-submesh SEG2 meshes with the same number of
/// elements, attached to the same `Coords`. `n_layers` must be ≥ 1.
///
/// Column `j` of `mesh_a` is linearly interpolated with column `j` of
/// `mesh_b` to produce the intermediate layers. Endpoint nodes from both
/// meshes are re-used (refcount incremented); intermediate nodes are
/// created at evenly spaced positions.
///
/// QUA4 node order per element (counterclockwise, `mesh_a` side first):
/// `[k][j]`, `[k][j+1]`, `[k+1][j+1]`, `[k+1][j]`.
pub fn sweep_qua4(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
    crate::ops::mesher::sweep::qua4_between(mesh_a, mesh_b, n_layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::ops::mesher::line_seg2::line_seg2;
    use crate::store::insert;

    #[test]
    fn sweep_qua4_basic() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let mesh_a = line_seg2(&a0, &a2, 2).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let mesh_b = line_seg2(&b0, &b2, 2).unwrap();

        let qua = sweep_qua4(&mesh_a, &mesh_b, 2).unwrap();
        assert_eq!(qua.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(qua.cell_count().unwrap(), 4);

        let n00 = qua.node(0, 0, 0).unwrap();
        assert_eq!(n00.coord().unwrap(), vec![0.0, 0.0]);
        let n01 = qua.node(0, 0, 1).unwrap();
        assert!((n01.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n01.coord().unwrap()[1]).abs() < 1e-12);
        let n02 = qua.node(0, 0, 2).unwrap();
        assert!((n02.coord().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((n02.coord().unwrap()[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sweep_qua4_one_layer_reuses_endpoints() {
        let coords = insert(Coords::new(2).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mesh_a = line_seg2(&a0, &a1, 1).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mesh_b = line_seg2(&b0, &b1, 1).unwrap();

        let qua = sweep_qua4(&mesh_a, &mesh_b, 1).unwrap();
        assert_eq!(qua.cell_count().unwrap(), 1);

        assert_eq!(qua.node(0, 0, 0).unwrap().id(), a0.id());
        assert_eq!(qua.node(0, 0, 1).unwrap().id(), a1.id());
        assert_eq!(qua.node(0, 0, 2).unwrap().id(), b1.id());
        assert_eq!(qua.node(0, 0, 3).unwrap().id(), b0.id());
    }

    #[test]
    fn sweep_qua4_rejects_zero_layers() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let m = line_seg2(&a, &b, 1).unwrap();
        assert!(sweep_qua4(&m, &m, 0).is_err());
    }

    #[test]
    fn sweep_qua4_rejects_mismatched_elem_counts() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let m1 = line_seg2(&a, &b, 1).unwrap();
        let m2 = line_seg2(&a, &c, 2).unwrap();
        assert!(sweep_qua4(&m1, &m2, 1).is_err());
    }

    #[test]
    fn sweep_qua4_rejects_non_seg2() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let seg = line_seg2(&a, &b, 1).unwrap();

        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut tri_mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        tri_mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        assert!(sweep_qua4(&tri_mesh, &seg, 1).is_err());
        assert!(sweep_qua4(&seg, &tri_mesh, 1).is_err());
    }
}
