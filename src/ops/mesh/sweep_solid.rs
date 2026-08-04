use crate::containers::mesh::Mesh;
use crate::error::Result;

/// Sweep two matching surface meshes into a solid mesh, building `n_layers`
/// layers of solid cells between `mesh_a` and `mesh_b`.
///
/// The 3-D companion of [`sweep`](fn@crate::ops::mesh::sweep):
/// where that one links two SEG2 contours into a QUA4 strip, this one links
/// two surface meshes into a solid — TRI3 faces sweep into PENTA6 prisms,
/// QUA4 faces into HEX8 hexahedra. Both meshes must be single-submesh
/// meshes of the same surface type, with the same number of cells and a
/// consistent node correspondence, attached to the same `Coords`.
pub fn sweep_solid(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Result<Mesh> {
    crate::ops::mesh::sweep_kernel::solid_between(mesh_a, mesh_b, n_layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::store::insert;

    /// Build a single-TRI3 mesh on `coords` from three coordinates, returning
    /// the mesh and its three corner nodes.
    fn tri(coords: &crate::store::Handle<Coords>, pts: [[f64; 3]; 3]) -> (Mesh, [Node; 3]) {
        let n0 = Node::create_in(coords.clone(), &pts[0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &pts[1]).unwrap();
        let n2 = Node::create_in(coords.clone(), &pts[2]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        (m, [n0, n1, n2])
    }

    #[test]
    fn sweep_tri3_to_penta6() {
        let coords = insert(Coords::new(3).unwrap());
        let (a, na) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let (b, nb) = tri(&coords, [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]);

        let solid = sweep_solid(&a, &b, 2).unwrap();
        assert_eq!(solid.element_types().unwrap(), vec![ElementType::PENTA6]);
        assert_eq!(solid.cell_count().unwrap(), 2);

        // Bottom of the first prism reuses mesh_a nodes.
        assert_eq!(solid.node(0, 0, 0).unwrap().id(), na[0].id());
        assert_eq!(solid.node(0, 0, 1).unwrap().id(), na[1].id());
        assert_eq!(solid.node(0, 0, 2).unwrap().id(), na[2].id());
        // Top of the last prism reuses mesh_b nodes.
        assert_eq!(solid.node(0, 1, 3).unwrap().id(), nb[0].id());
        assert_eq!(solid.node(0, 1, 4).unwrap().id(), nb[1].id());
        assert_eq!(solid.node(0, 1, 5).unwrap().id(), nb[2].id());
        // Interpolated interior layer sits at z = 1.
        let mid = solid.node(0, 0, 3).unwrap();
        assert!((mid.position().unwrap()[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sweep_shared_edge_stays_shared() {
        // Two triangles sharing an edge → two prisms sharing a quad face.
        let coords = insert(Coords::new(3).unwrap());
        let a0 = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let a1 = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let a2 = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let a3 = Node::create_in(coords.clone(), &[1.0, 1.0, 0.0]).unwrap();
        let mut a = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        a.add_cell(&[a0.id(), a1.id(), a2.id()]).unwrap();
        a.add_cell(&[a1.id(), a3.id(), a2.id()]).unwrap();

        let b0 = Node::create_in(coords.clone(), &[0.0, 0.0, 1.0]).unwrap();
        let b1 = Node::create_in(coords.clone(), &[1.0, 0.0, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[0.0, 1.0, 1.0]).unwrap();
        let b3 = Node::create_in(coords.clone(), &[1.0, 1.0, 1.0]).unwrap();
        let mut b = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        b.add_cell(&[b0.id(), b1.id(), b2.id()]).unwrap();
        b.add_cell(&[b1.id(), b3.id(), b2.id()]).unwrap();

        let solid = sweep_solid(&a, &b, 3).unwrap();
        assert_eq!(solid.cell_count().unwrap(), 6);
        // Node a1 is shared by both source triangles; both prisms of the
        // first layer must reference the same node for it.
        assert_eq!(
            solid.node(0, 0, 1).unwrap().id(), // cell 0, local 1 = a1
            solid.node(0, 1, 0).unwrap().id(), // cell 1, local 0 = a1
        );
    }

    #[test]
    fn sweep_rejects_zero_layers() {
        let coords = insert(Coords::new(3).unwrap());
        let (a, _) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert!(sweep_solid(&a, &a, 0).is_err());
    }

    #[test]
    fn sweep_rejects_mismatched_types() {
        let coords = insert(Coords::new(3).unwrap());
        let (a, _) = tri(&coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let q0 = Node::create_in(coords.clone(), &[0.0, 0.0, 1.0]).unwrap();
        let q1 = Node::create_in(coords.clone(), &[1.0, 0.0, 1.0]).unwrap();
        let q2 = Node::create_in(coords.clone(), &[1.0, 1.0, 1.0]).unwrap();
        let q3 = Node::create_in(coords.clone(), &[0.0, 1.0, 1.0]).unwrap();
        let mut q = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        q.add_cell(&[q0.id(), q1.id(), q2.id(), q3.id()]).unwrap();
        assert!(sweep_solid(&a, &q, 1).is_err());
    }
}
