use crate::aggregate::Aggregate;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;

/// Convert a mesh to POI1, **zone by zone**: each zone gives its POI1
/// companion ([`SubMesh::to_poi1`]).
///
/// The result has the **same number of zones** as `mesh`. A POI1 zone listing
/// each node once comes back **as itself** (same handle, no copy); any other
/// zone gives the POI1 zone of its distinct nodes, in order of first
/// appearance, built once and cached on it; an empty zone gives an empty POI1
/// zone, so the count is preserved.
///
/// The result **shares** those zones: calling this twice on the same mesh
/// yields the same zones, and a field built on one pairs with a field built on
/// the other. A built companion holds its own references on its nodes for as
/// long as the zone's cache keeps it — a mutation of the input zone, or its
/// drop, releases them. `mesh`'s own connectivity is left untouched.
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
/// // A mesh's **distinct** nodes, one POI1 cell each: two cells that
/// // share a node count it once only.
/// let l = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[2.0, 0.0, 0.0]), 2, ElementType::SEG2)?;
/// let points = mesh::to_poi1(&l)?;
/// assert_eq!(points.cell_count(), 3);
/// // Those points are already their own companion: no copy.
/// assert!(mesh::to_poi1(&points)?.get(0)?.same_object(&points.get(0)?));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
///
/// [`SubMesh::to_poi1`]: crate::containers::mesh::SubMesh::to_poi1
pub fn to_poi1(mesh: &Mesh) -> Result<Mesh> {
    let mut result = Mesh::empty();
    for zone in mesh {
        result.add_sub(SubMesh::to_poi1(zone))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    #[test]
    fn tri3_submesh_becomes_unique_node_list() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.5, 1.0]).unwrap();

        // Two triangles sharing the edge (b, c): 4 distinct nodes total.
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let poi = to_poi1(&tri).unwrap();
        assert_eq!(poi.len(), 1);
        assert_eq!(poi.element_types().unwrap(), vec![ElementType::POI1]);
        // 6 connectivity entries but only 4 unique nodes.
        assert_eq!(poi.cell_count(), 4);

        let ids: Vec<_> = (0..4).map(|i| poi.node(0, i, 0).unwrap().id()).collect();
        assert_eq!(ids, vec![a.id(), b.id(), c.id(), d.id()]);
    }

    #[test]
    fn preserves_submesh_count_and_increfs_nodes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        // One POI1 submesh + one TRI3 submesh.
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        mesh.add_sub(sm_tri).unwrap();

        // refcounts before: a is in POI1 + TRI3 + Node = 3; b,c in TRI3 + Node = 2.
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 3);
            assert_eq!(cf.refcount(b.id()), 2);
        }

        let poi = to_poi1(&mesh).unwrap();
        assert_eq!(poi.len(), 2, "same number of submeshes");
        assert_eq!(
            poi.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::POI1]
        );
        assert_eq!(poi.cell_counts().unwrap(), vec![1, 3]);

        // The POI1 zone is its own companion: the very zone, not a copy.
        assert!(poi.get(0).unwrap().same_object(&mesh.get(0).unwrap()));

        // Only the TRI3's companion, a new zone, increfed its nodes.
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 4); // +POI1(sub1)
            assert_eq!(cf.refcount(b.id()), 3); // +POI1(sub1)
        }

        // Dropping the result gives nothing back: its zones are the source's
        // POI1 zone and the TRI3's memoized companion, which the source holds.
        drop(poi);
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 4);
            assert_eq!(cf.refcount(b.id()), 3);
        }

        // Dropping the source drops its caches, and the companions with them.
        drop(mesh);
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 1); // the `Node` handle alone
            assert_eq!(cf.refcount(b.id()), 1);
        }
    }

    #[test]
    fn empty_mesh_gives_empty_mesh() {
        let poi = to_poi1(&Mesh::empty()).unwrap();
        assert_eq!(poi.len(), 0);
    }
}
