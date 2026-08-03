//! Per-element barycentre (centroid) mesher.
//!
//! [`barycenter`] turns any mesh into a POI1 mesh holding one fresh node at
//! the centre of gravity of each element, **submesh by submesh** (same number
//! of submeshes as the input, one POI1 cell per input element). A POI1 input
//! is copied node-for-node at colocated coordinates (the centroid of a single
//! point is the point itself) — the natural way to mint colocated support
//! nodes (e.g. Lagrange multipliers) for a set of nodes.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;
use crate::store::{insert, read, write};

/// Build a POI1 mesh of per-element centroids, preserving submesh structure.
///
/// The result has the **same number of submeshes** as `mesh`; each output
/// submesh is a POI1 submesh with one cell per element of the matching input
/// submesh, holding a **new** node placed at the element's centroid (the
/// arithmetic mean of its nodes' coordinates). The nodes are minted in the
/// input mesh's [`Coords`](crate::coords::Coords);
/// each output POI1 submesh owns the sole initial refcount of the nodes it
/// mints (handed over via
/// [`SubMesh::add_cell_taking`](crate::containers::mesh::SubMesh::add_cell_taking)),
/// so the nodes live exactly as long as the returned mesh. An empty input
/// submesh yields an empty POI1 submesh (the count is preserved); `mesh`
/// itself is left untouched.
pub fn barycenter(mesh: &Mesh) -> Result<Mesh> {
    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (coords, element_type, conn) = {
            let sm = read(sm_handle)?;
            (sm.coords(), sm.element_type(), sm.connectivity().to_vec())
        };
        let npc = element_type.nodes_per_cell();
        let n_cells = conn.len() / npc;

        // Compute every centroid under a read lock, then mint the nodes under
        // a write lock — two separate critical sections (and `add_cell_taking`
        // takes its own read lock, so it must run after the write lock drops).
        let centroids: Vec<Vec<f64>> = {
            let c = read(&coords)?;
            (0..n_cells)
                .map(|cell| {
                    let ids = &conn[cell * npc..(cell + 1) * npc];
                    let dim = c.coord(ids[0])?.len();
                    let mut centroid = vec![0.0; dim];
                    for &nid in ids {
                        for (acc, &x) in centroid.iter_mut().zip(c.coord(nid)?) {
                            *acc += x;
                        }
                    }
                    for x in &mut centroid {
                        *x /= npc as f64;
                    }
                    Ok(centroid)
                })
                .collect::<Result<_>>()?
        };

        let new_ids: Vec<NodeId> = {
            let mut c = write(&coords)?;
            centroids
                .iter()
                .map(|coord| c.add_node(coord))
                .collect::<Result<_>>()?
        };

        let mut out_sm = SubMesh::new(coords, ElementType::POI1);
        for nid in &new_ids {
            out_sm.add_cell_taking(&[*nid])?;
        }
        result.add_sub(insert(out_sm))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::store::{insert, read};

    /// A POI1 input yields one colocated (same coordinates) fresh node per
    /// point, with distinct ids.
    #[test]
    fn poi1_input_colocates_fresh_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[3.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        let bary = barycenter(&mesh).unwrap();
        assert_eq!(bary.len(), 1);
        assert_eq!(bary.element_types().unwrap(), vec![ElementType::POI1]);
        assert_eq!(bary.cell_count().unwrap(), 2);

        let m0 = bary.node(0, 0, 0).unwrap();
        let m1 = bary.node(0, 1, 0).unwrap();
        // Fresh nodes, distinct from the inputs but at the same coordinates.
        assert_ne!(m0.id(), a.id());
        assert_ne!(m1.id(), b.id());
        let c = read(&coords).unwrap();
        assert_eq!(c.coord(m0.id()).unwrap(), &[0.0, 0.0]);
        assert_eq!(c.coord(m1.id()).unwrap(), &[3.0, 1.0]);
    }

    /// A TRI3 element yields a single POI1 node at the triangle's centroid.
    #[test]
    fn tri3_centroid_is_mean_of_vertices() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[3.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 3.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let bary = barycenter(&mesh).unwrap();
        assert_eq!(bary.cell_count().unwrap(), 1);
        let centroid = bary.node(0, 0, 0).unwrap();
        let cf = read(&coords).unwrap();
        assert_eq!(cf.coord(centroid.id()).unwrap(), &[1.0, 1.0]);
    }

    /// Submesh structure is preserved: a POI1 + TRI3 input gives two POI1
    /// output submeshes, and each minted node has refcount 1 (owned solely by
    /// the output mesh).
    #[test]
    fn preserves_submesh_count_and_owns_new_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 3.0]).unwrap();

        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();

        let bary = barycenter(&mesh).unwrap();
        assert_eq!(bary.len(), 2, "same number of submeshes");
        assert_eq!(
            bary.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::POI1]
        );
        assert_eq!(bary.cell_counts().unwrap(), vec![1, 1]);

        // Each minted node is owned solely by the output mesh.
        let m_point = bary.node(0, 0, 0).unwrap();
        let m_tri = bary.node(1, 0, 0).unwrap();
        let (pid, tid) = (m_point.id(), m_tri.id());
        {
            let cf = read(&coords).unwrap();
            // refcount 2: the output POI1 submesh + the `Node` handle above.
            assert_eq!(cf.refcount(pid), 2);
            assert_eq!(cf.refcount(tid), 2);
        }
        drop(m_point);
        drop(m_tri);
        drop(bary);
        // After dropping the output mesh (and the `Node` handles), the minted
        // nodes are released — refcount back to 0.
        let cf = read(&coords).unwrap();
        assert_eq!(cf.refcount(pid), 0);
        assert_eq!(cf.refcount(tid), 0);
    }

    #[test]
    fn empty_mesh_gives_empty_mesh() {
        let bary = barycenter(&Mesh::empty()).unwrap();
        assert_eq!(bary.len(), 0);
    }
}
