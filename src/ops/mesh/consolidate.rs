use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::handle::Handle;
use std::collections::HashSet;

/// Fuse the zones of the same element type into one, dropping duplicate
/// cells (identical node sequences).
///
/// Types appear in their first-seen order. A type held by **a single zone
/// with no duplicate cell** has nothing to merge: that zone comes back as
/// itself, and the result **shares** it with `mesh` — mutating one
/// (`add_cell`, `merge_nodes` in place) mutates the other, and a sealed zone
/// stays sealed; [`SubMesh::duplicate`] makes an independent copy. Any other
/// type gives a new zone, with the face colour of the first zone of that type,
/// which increfs its nodes afresh. `mesh` itself is left untouched; an empty
/// `mesh` gives an empty mesh.
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
/// // Merges the zones of the same element type into one.
/// let a = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[1.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let b = mesh::line(&p(&[1.0, 0.0, 0.0]), &p(&[2.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let two = a.union(&b)?;
/// assert_eq!(two.len(), 2);
/// let one = mesh::consolidate(&two);
/// assert_eq!(one.len(), 1);
/// assert_eq!(one.cell_count(), 2);
/// // A zone with nothing to merge comes back as itself.
/// assert!(mesh::consolidate(&a).get(0)?.same_object(&a.get(0)?));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn consolidate(mesh: &Mesh) -> Mesh {
    let mut result = Mesh::empty();
    let Some(first) = mesh.iter().next() else {
        return result;
    };
    // One `Coords` for every zone, the mesh's own check: the zones built here
    // attach to it, and `push` needs no `add_sub` check to know it.
    let coords = first.read().coords();

    // Collect types in first-seen order.
    let mut ordered_types: Vec<ElementType> = Vec::new();
    for sm_handle in mesh {
        let et = sm_handle.read().element_type();
        if !ordered_types.contains(&et) {
            ordered_types.push(et);
        }
    }

    for et in ordered_types {
        let npc = et.nodes_per_cell();
        // Never empty: `et` was read off one of these zones.
        let zones: Vec<&Handle<SubMesh>> = mesh
            .iter()
            .filter(|h| h.read().element_type() == et)
            .collect();

        // The surviving cells are gathered whole, then posted in one locked
        // pass over the `Coords`.
        let mut kept: Vec<NodeId> = Vec::new();
        let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
        for zone in &zones {
            // Read under the guard: a staging copy here would be paid once
            // per (zone, element type) pair.
            let s = zone.read();
            for chunk in s.connectivity().chunks(npc) {
                if seen.insert(chunk.to_vec()) {
                    kept.extend_from_slice(chunk);
                }
            }
        }

        // One zone without a duplicate cell has nothing to merge: the zone
        // itself is the result, not a copy of it.
        if let [only] = zones.as_slice()
            && kept.len() == only.read().connectivity().len()
        {
            result.push((*only).clone());
            result.post_push();
            continue;
        }

        // Face colour from the first zone of this type.
        let first_color = zones[0].read().face_color();
        let mut new_sm = SubMesh::from_connectivity(coords.clone(), et, kept)
            .expect("whole cells of live nodes: the input zones hold them");
        new_sm.set_face_color(first_color);
        result.push(Handle::new(new_sm));
        result.post_push();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;

    #[test]
    fn merges_same_type_submeshes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        // Two separate TRI3 submeshes with one cell each.
        let sm1 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[b.id(), c.id(), a.id()]).unwrap(); // new cell
            Handle::new(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm1).unwrap();
        mesh.add_sub(sm2).unwrap();
        assert_eq!(mesh.len(), 2);

        let c2 = consolidate(&mesh);
        assert_eq!(c2.len(), 1, "must merge the two TRI3 submeshes");
        assert_eq!(c2.cell_count(), 2, "two distinct cells must be kept");
        assert_eq!(c2.element_types().unwrap(), vec![ElementType::TRI3]);
    }

    #[test]
    fn removes_duplicate_cells() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm1 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap(); // exact duplicate
            Handle::new(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm1).unwrap();
        mesh.add_sub(sm2).unwrap();

        let c2 = consolidate(&mesh);
        assert_eq!(c2.len(), 1);
        assert_eq!(c2.cell_count(), 1, "the duplicate must be removed");
    }

    #[test]
    fn preserves_distinct_types() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        let sm_poi = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            Handle::new(sm)
        };
        // Second TRI3 with a duplicate.
        let sm_tri2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_poi).unwrap();
        mesh.add_sub(sm_tri2).unwrap();

        let c2 = consolidate(&mesh);
        assert_eq!(c2.len(), 2, "TRI3 + POI1");
        assert_eq!(
            c2.element_types().unwrap(),
            vec![ElementType::TRI3, ElementType::POI1],
            "first-seen order"
        );
        assert_eq!(c2.cell_counts().unwrap(), vec![1, 1]);
    }

    #[test]
    fn keeps_a_zone_with_nothing_to_merge() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut tri = SubMesh::new(coords.clone(), ElementType::TRI3);
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let tri = Handle::new(tri);
        let mut points = SubMesh::new(coords.clone(), ElementType::POI1);
        points.add_cell(&[a.id()]).unwrap();
        points.add_cell(&[b.id()]).unwrap();
        let points = Handle::new(points);
        let mut mesh = Mesh::empty();
        mesh.add_sub(tri.clone()).unwrap();
        mesh.add_sub(points.clone()).unwrap();
        let refs_before = coords.read().refcount(a.id());

        let c2 = consolidate(&mesh);
        // Each type has one zone and no duplicate: both zones come back as
        // they are, and no node gains a reference.
        assert!(c2.get(0).unwrap().same_object(&tri));
        assert!(c2.get(1).unwrap().same_object(&points));
        assert_eq!(coords.read().refcount(a.id()), refs_before);
    }

    #[test]
    fn rebuilds_a_single_zone_holding_a_duplicate_cell() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut seg = SubMesh::new(coords.clone(), ElementType::SEG2);
        seg.add_cell(&[a.id(), b.id()]).unwrap();
        seg.add_cell(&[a.id(), b.id()]).unwrap(); // the same cell again
        let seg = Handle::new(seg);
        let mut mesh = Mesh::empty();
        mesh.add_sub(seg.clone()).unwrap();

        let c2 = consolidate(&mesh);
        // Something to drop: a new zone, without the duplicate.
        assert!(!c2.get(0).unwrap().same_object(&seg));
        assert_eq!(c2.cell_count(), 1);
        assert_eq!(seg.read().cell_count(), 2);
    }
}
