use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;
use crate::handle::Handle;
use std::collections::HashSet;

/// Fuse submeshes of the same element type into one, dropping duplicate
/// cells (identical node sequences).
///
/// Types appear in their first-seen order; the face colour of the first
/// submesh of each type is kept. Every node referenced by the result is
/// increfed afresh by the new submeshes; `mesh` itself is left untouched.
///
/// Errors if `mesh` has no submeshes (no Coords to attach to).
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
/// // Fusionne les zones de même type d'élément en une seule.
/// let a = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[1.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let b = mesh::line(&p(&[1.0, 0.0, 0.0]), &p(&[2.0, 0.0, 0.0]), 1, ElementType::SEG2)?;
/// let deux = a.union(&b)?;
/// assert_eq!(deux.len(), 2);
/// let une = mesh::consolidate(&deux)?;
/// assert_eq!(une.len(), 1);
/// assert_eq!(une.cell_count()?, 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn consolidate(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();

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

        // Face colour from the first submesh of this type.
        let first_color = mesh
            .iter()
            .find(|h| h.read().element_type() == et)
            .map(|h| -> Result<_> { Ok(h.read().face_color()) })
            .transpose()?
            .unwrap_or_default();

        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(first_color);

        let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
        for sm_handle in mesh {
            let sm_et = sm_handle.read().element_type();
            if sm_et != et {
                continue;
            }
            let conn = sm_handle.read().connectivity().to_vec();
            for chunk in conn.chunks(npc) {
                if seen.insert(chunk.to_vec()) {
                    new_sm.add_cell(chunk)?;
                }
            }
        }

        result.add_sub(Handle::new(new_sm))?;
    }

    Ok(result)
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

        let c2 = consolidate(&mesh).unwrap();
        assert_eq!(c2.len(), 1, "must merge the two TRI3 submeshes");
        assert_eq!(
            c2.cell_count().unwrap(),
            2,
            "two distinct cells must be kept"
        );
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

        let c2 = consolidate(&mesh).unwrap();
        assert_eq!(c2.len(), 1);
        assert_eq!(c2.cell_count().unwrap(), 1, "the duplicate must be removed");
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

        let c2 = consolidate(&mesh).unwrap();
        assert_eq!(c2.len(), 2, "TRI3 + POI1");
        assert_eq!(
            c2.element_types().unwrap(),
            vec![ElementType::TRI3, ElementType::POI1],
            "first-seen order"
        );
        assert_eq!(c2.cell_counts().unwrap(), vec![1, 1]);
    }
}
