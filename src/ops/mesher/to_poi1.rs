use crate::aggregate::Aggregate;
use crate::error::Result;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{insert, with};

/// Convert a mesh to POI1, **submesh by submesh**.
///
/// The result has the **same number of submeshes** as `mesh`; each output
/// submesh is a POI1 submesh holding the de-duplicated list of nodes used
/// by the corresponding input submesh, in order of first appearance. A
/// POI1 input submesh is therefore copied node-for-node; an empty input
/// submesh yields an empty POI1 submesh (so the count is preserved).
///
/// Every node referenced by the result is increfed afresh by the new POI1
/// submeshes; `mesh` itself is left untouched.
pub fn to_poi1(mesh: &Mesh) -> Result<Mesh> {
    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (cfg, connectivity) =
            with(sm_handle, |s| (s.configuration(), s.connectivity().to_vec()))?;
        let mut poi = SubMesh::new(cfg, ElementType::POI1);
        let mut seen: Vec<NodeId> = Vec::new();
        for nid in connectivity {
            if !seen.contains(&nid) {
                seen.push(nid);
                poi.add_cell(&[nid])?;
            }
        }
        result.add_sub(insert(poi))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::containers::mesh::Mesh;
    use crate::store::{insert, with};

    #[test]
    fn tri3_submesh_becomes_unique_node_list() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
        let d = Node::create_in(cfg.clone(), &[1.5, 1.0]).unwrap();

        // Two triangles sharing the edge (b, c): 4 distinct nodes total.
        let mut tri = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let poi = to_poi1(&tri).unwrap();
        assert_eq!(poi.submesh_count(), 1);
        assert_eq!(poi.element_types().unwrap(), vec![ElementType::POI1]);
        // 6 connectivity entries but only 4 unique nodes.
        assert_eq!(poi.cell_count().unwrap(), 4);

        let ids: Vec<_> = (0..4).map(|i| poi.node(0, i, 0).unwrap().id()).collect();
        assert_eq!(ids, vec![a.id(), b.id(), c.id(), d.id()]);
    }

    #[test]
    fn preserves_submesh_count_and_increfs_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        // One POI1 submesh + one TRI3 submesh.
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();

        // refcounts before: a is in POI1 + TRI3 + Node = 3; b,c in TRI3 + Node = 2.
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 3);
            assert_eq!(cf.refcount(b.id()), 2);
        })
        .unwrap();

        let poi = to_poi1(&mesh).unwrap();
        assert_eq!(poi.submesh_count(), 2, "same number of submeshes");
        assert_eq!(
            poi.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::POI1]
        );
        assert_eq!(poi.cell_counts().unwrap(), vec![1, 3]);

        // The new POI1 submeshes increfed every node they reference.
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 5); // +POI1(sub0) +POI1(sub1)
            assert_eq!(cf.refcount(b.id()), 3); // +POI1(sub1)
        })
        .unwrap();

        drop(poi);
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 3);
            assert_eq!(cf.refcount(b.id()), 2);
        })
        .unwrap();
    }

    #[test]
    fn empty_mesh_gives_empty_mesh() {
        let poi = to_poi1(&Mesh::empty()).unwrap();
        assert_eq!(poi.submesh_count(), 0);
    }
}
