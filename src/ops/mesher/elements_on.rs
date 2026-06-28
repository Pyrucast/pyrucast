use crate::aggregate::Aggregate;
use crate::containers::mesh::{Mesh, NodeId};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};
use std::collections::HashSet;

/// Keep the elements of `mesh` that **rest on** the nodes of `points` —
/// the Cast3m `ELEM … APPUYE` operator.
///
/// `points` is any mesh (typically a POI1 points mesh): only the **set of
/// nodes it references** matters, not its element types. A cell of `mesh`
/// is kept depending on `strict`:
///
/// - `strict = true` — **every** node of the cell must be in the set
///   (Cast3m `APPUYE STRICTEMENT`);
/// - `strict = false` — **at least one** node of the cell must be in the
///   set (Cast3m `APPUYE`).
///
/// The result mirrors `mesh` **submesh by submesh** (same order, same
/// element types and face colours, [[feedback-viz-per-element]] — zones
/// stay separate): each
/// output submesh holds the kept cells of the matching input submesh,
/// possibly empty. Use [`consolidate`](crate::ops::mesher::consolidate) to
/// drop or fuse the empty/redundant zones afterwards. Kept cells reference
/// the original nodes (refcount bumped); `mesh` itself is left untouched.
///
/// Both meshes must live on the **same `Coords`** (node ids are only
/// meaningful within one) — otherwise an error is returned. An empty
/// `points` (no referenced node) keeps nothing.
pub fn elements_on(mesh: &Mesh, points: &Mesh, strict: bool) -> Result<Mesh> {
    // Collect the allowed node set from `points`, checking Coords coherence.
    let mut allowed: HashSet<NodeId> = HashSet::new();
    if !points.is_empty() && !mesh.is_empty() {
        let mc = mesh.coords()?;
        let pc = points.coords()?;
        if mc.index() != pc.index() || mc.generation() != pc.generation() {
            return Err(PyrucastError::Message(
                "elements_on: mesh and points are not attached to the same Coords".into(),
            ));
        }
    }
    for sm in points {
        allowed.extend(read(sm)?.connectivity().iter().copied());
    }

    let mut out = Mesh::empty();
    for sm in mesh {
        let src = read(sm)?;
        let et = src.element_type();
        let npc = et.nodes_per_cell();
        let conn = src.connectivity();
        let mut kept = crate::containers::mesh::SubMesh::new(src.coords(), et);
        kept.set_face_color(src.face_color());
        for cell in conn.chunks_exact(npc) {
            let on = if strict {
                cell.iter().all(|n| allowed.contains(n))
            } else {
                cell.iter().any(|n| allowed.contains(n))
            };
            if on {
                kept.add_cell(cell)?;
            }
        }
        out.add_sub(insert(kept))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, ElementType, Node, SubMesh};
    use crate::store::insert;

    /// Five nodes on a line and a TRI3 mesh of two triangles sharing edge
    /// (1, 2): cell0 = (0,1,2), cell1 = (1,3,4). Returns (coords, nodes, mesh).
    fn two_triangles() -> (
        crate::store::Handle<Coords>,
        Vec<Node>,
        Mesh,
    ) {
        let coords = insert(Coords::new(2).unwrap());
        let n: Vec<Node> = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        mesh.add_cell(&[n[1].id(), n[3].id(), n[4].id()]).unwrap();
        (coords, n, mesh)
    }

    /// Cell node ids of submesh `zone` of `mesh`, grouped per cell.
    fn cells(mesh: &Mesh, zone: usize) -> Vec<Vec<NodeId>> {
        let s = read(&mesh.get(zone).unwrap()).unwrap();
        let npc = s.element_type().nodes_per_cell();
        s.connectivity().chunks_exact(npc).map(|c| c.to_vec()).collect()
    }

    #[test]
    fn strict_keeps_only_fully_supported_cells() {
        let (coords, n, mesh) = two_triangles();
        // Points = {0, 1, 2}: only cell0 has all its nodes inside.
        let pts = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(coords.clone(), &[n[0].id(), n[1].id(), n[2].id()])
                .unwrap(),
        );
        let r = elements_on(&mesh, &pts, true).unwrap();
        assert_eq!(r.len(), 1, "one submesh per input submesh");
        assert_eq!(r.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(cells(&r, 0), vec![vec![n[0].id(), n[1].id(), n[2].id()]]);
    }

    #[test]
    fn loose_keeps_cells_with_any_node() {
        let (coords, n, mesh) = two_triangles();
        // Points = {1}: shared node → both triangles touch it.
        let pts = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(coords.clone(), &[n[1].id()]).unwrap(),
        );
        let r = elements_on(&mesh, &pts, false).unwrap();
        assert_eq!(cells(&r, 0).len(), 2, "both cells share node 1");

        // Points = {3}: only cell1 uses it.
        let pts3 = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(coords.clone(), &[n[3].id()]).unwrap(),
        );
        let r3 = elements_on(&mesh, &pts3, false).unwrap();
        assert_eq!(cells(&r3, 0), vec![vec![n[1].id(), n[3].id(), n[4].id()]]);
    }

    #[test]
    fn strict_can_yield_empty_but_keeps_the_zone() {
        let (coords, n, mesh) = two_triangles();
        // Points = {0}: no triangle has *all* nodes inside.
        let pts = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(coords.clone(), &[n[0].id()]).unwrap(),
        );
        let r = elements_on(&mesh, &pts, true).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(read(&r.get(0).unwrap()).unwrap().cell_count(), 0);
    }

    #[test]
    fn empty_points_keeps_nothing() {
        let (_coords, _n, mesh) = two_triangles();
        let r = elements_on(&mesh, &Mesh::empty(), false).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(read(&r.get(0).unwrap()).unwrap().cell_count(), 0);
    }

    #[test]
    fn preserves_submesh_structure_and_increfs() {
        let (coords, n, _m) = two_triangles();
        // Two zones: a TRI3 and a SEG2, on the same coords.
        let mut mesh = Mesh::empty();
        let tri_color = crate::containers::mesh::RgbColor::new(10, 20, 30);
        {
            let mut tri = SubMesh::new(coords.clone(), ElementType::TRI3);
            tri.set_face_color(tri_color);
            tri.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
            mesh.add_sub(insert(tri)).unwrap();
            let mut seg = SubMesh::new(coords.clone(), ElementType::SEG2);
            seg.add_cell(&[n[3].id(), n[4].id()]).unwrap();
            mesh.add_sub(insert(seg)).unwrap();
        }
        // All nodes allowed → everything kept, both zones present.
        let pts = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(
                coords.clone(),
                &[n[0].id(), n[1].id(), n[2].id(), n[3].id(), n[4].id()],
            )
            .unwrap(),
        );
        let before = read(&coords).unwrap().refcount(n[0].id());
        let r = elements_on(&mesh, &pts, true).unwrap();
        assert_eq!(
            r.element_types().unwrap(),
            vec![ElementType::TRI3, ElementType::SEG2]
        );
        assert_eq!(r.cell_counts().unwrap(), vec![1, 1]);
        // Face colour of the TRI3 zone is carried over.
        assert_eq!(read(&r.get(0).unwrap()).unwrap().face_color(), tri_color);
        // n[0] is referenced once more by the kept TRI3 cell.
        assert_eq!(read(&coords).unwrap().refcount(n[0].id()), before + 1);
        drop(r);
        assert_eq!(read(&coords).unwrap().refcount(n[0].id()), before);
    }

    #[test]
    fn mismatched_coords_errors() {
        let (_c1, _n1, mesh) = two_triangles();
        let c2 = insert(Coords::new(2).unwrap());
        let m2 = Node::create_in(c2.clone(), &[0.0, 0.0]).unwrap();
        let pts = Mesh::from_submesh(
            SubMesh::poi1_from_node_ids(c2.clone(), &[m2.id()]).unwrap(),
        );
        assert!(elements_on(&mesh, &pts, false).is_err());
    }
}
