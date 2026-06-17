use crate::aggregate::Aggregate;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;
use crate::store::{insert, read};
use std::collections::HashSet;

/// Fuse submeshes of the same element type into one, dropping duplicate
/// cells (identical node sequences).
///
/// Types appear in their first-seen order; the face colour of the first
/// submesh of each type is kept. Every node referenced by the result is
/// increfed afresh by the new submeshes; `mesh` itself is left untouched.
///
/// Errors if `mesh` has no submeshes (no Coords to attach to).
pub fn consolidate(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();

    // Collect types in first-seen order.
    let mut ordered_types: Vec<ElementType> = Vec::new();
    for sm_handle in mesh {
        let et = read(sm_handle)?.element_type();
        if !ordered_types.contains(&et) {
            ordered_types.push(et);
        }
    }

    for et in ordered_types {
        let npc = et.nodes_per_cell();

        // Face colour from the first submesh of this type.
        let first_color = mesh
            .iter()
            .find(|h| read(h).map(|s| s.element_type()).ok() == Some(et))
            .map(|h| -> Result<_> { Ok(read(h)?.face_color()) })
            .transpose()?
            .unwrap_or_default();

        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(first_color);

        let mut seen: HashSet<Vec<NodeId>> = HashSet::new();
        for sm_handle in mesh {
            let sm_et = read(sm_handle)?.element_type();
            if sm_et != et {
                continue;
            }
            let conn = read(sm_handle)?.connectivity().to_vec();
            for chunk in conn.chunks(npc) {
                if seen.insert(chunk.to_vec()) {
                    new_sm.add_cell(chunk)?;
                }
            }
        }

        result.add_sub(insert(new_sm))?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    #[test]
    fn merges_same_type_submeshes() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        // Two separate TRI3 submeshes with one cell each.
        let sm1 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[b.id(), c.id(), a.id()]).unwrap(); // new cell
            insert(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm1).unwrap();
        mesh.add_sub(sm2).unwrap();
        assert_eq!(mesh.len(), 2);

        let c2 = consolidate(&mesh).unwrap();
        assert_eq!(c2.len(), 1, "must merge the two TRI3 submeshes");
        assert_eq!(c2.cell_count().unwrap(), 2, "two distinct cells must be kept");
        assert_eq!(c2.element_types().unwrap(), vec![ElementType::TRI3]);
    }

    #[test]
    fn removes_duplicate_cells() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm1 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap(); // exact duplicate
            insert(sm)
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sm_poi = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            insert(sm)
        };
        // Second TRI3 with a duplicate.
        let sm_tri2 = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
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
