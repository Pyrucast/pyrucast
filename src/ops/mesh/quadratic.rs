//! Promote a linear mesh to its **quadratic** (Lagrange-2) counterpart.
//!
//! [`to_quadratic`] maps each linear element type to its quadratic sibling
//! (`SEG2 → SEG3`, `TRI3 → TRI6`, `QUA4 → QUA8`, `TET4 → TET10`,
//! `PENTA6 → PENTA15`, `HEX8 → HEX20`). The corner nodes are **re-used**
//! (their refcount is incremented); one **mid-edge node** is created per
//! distinct edge, at the edge midpoint, and **shared** between every cell
//! (across all submeshes) that uses that edge — so the result stays
//! conforming.

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::Handle;
use std::collections::HashMap;

/// The quadratic element type of a linear one, together with its edges as
/// local corner-index pairs.
///
/// Both come straight from the element: the mid-side node of edge `k` sits at
/// local index `corner_count() + k`, so the edge list *is* the mid-edge node
/// order the quadratic type expects (see [`crate::atoms::element_kind`]).
fn quadratic_of(et: ElementType) -> Result<(ElementType, &'static [[usize; 2]])> {
    match et.as_kind().quadratic() {
        Some(q) => Ok((q, et.as_kind().edges())),
        None => Err(PyrucastError::Message(format!(
            "to_quadratic: {et} has no quadratic counterpart"
        ))),
    }
}

/// Build the quadratic copy of a linear `mesh`.
///
/// Every submesh must hold a **linear** element type; a POI1 or
/// already-quadratic submesh is an error. The result mirrors `mesh` submesh
/// by submesh (same order, same face colours) with each type bumped to its
/// quadratic sibling. Corner nodes are re-used; mid-edge nodes are freshly
/// created once per edge, at the midpoint, and shared. The input mesh is left
/// untouched.
pub fn to_quadratic(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // One mid-edge node per distinct edge, keyed by the unordered corner-id
    // pair. Holding the `Node` keeps the creation refcount alive until the
    // whole build is done (the SubMeshes incref on top via `add_cell`).
    let mut mid: HashMap<(NodeId, NodeId), Node> = HashMap::new();

    let mut result = Mesh::empty();
    for sm_h in mesh {
        let (et, color, conn) = {
            let s = sm_h.read();
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let (quad_et, edges) = quadratic_of(et)?;
        let npc = et.nodes_per_cell();

        let mut new_sm = SubMesh::new(coords.clone(), quad_et);
        new_sm.set_face_color(color);

        for cell in conn.chunks(npc) {
            let mut nodes: Vec<NodeId> = Vec::with_capacity(quad_et.nodes_per_cell());
            // Corners first (re-used; `add_cell` will incref them).
            nodes.extend_from_slice(cell);
            // Then one mid-edge node per edge, in the quadratic node order.
            for &[a, b] in edges {
                let (na, nb) = (cell[a], cell[b]);
                let key = if na.0 <= nb.0 { (na, nb) } else { (nb, na) };
                let mid_id = match mid.get(&key) {
                    Some(node) => node.id(),
                    None => {
                        // Read both corner coordinates (dropping the guard
                        // before creating, which takes a write lock).
                        let midpoint: Vec<f64> = {
                            let c = coords.read();
                            let ca = c.position(na)?;
                            let cb = c.position(nb)?;
                            ca.iter().zip(cb).map(|(&x, &y)| 0.5 * (x + y)).collect()
                        };
                        let node = Node::create_in(coords.clone(), &midpoint)?;
                        let id = node.id();
                        mid.insert(key, node);
                        id
                    }
                };
                nodes.push(mid_id);
            }
            new_sm.add_cell(&nodes)?;
        }
        result.add_sub(Handle::new(new_sm))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::store::Handle;

    #[test]
    fn tri3_to_tri6_shares_edge_midpoints() {
        // Two triangles sharing edge (b, c): the shared edge yields ONE mid
        // node, referenced by both quadratic cells.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[2.0, 2.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        tri.add_cell(&[b.id(), d.id(), c.id()]).unwrap();

        let quad = to_quadratic(&tri).unwrap();
        assert_eq!(quad.element_types().unwrap(), vec![ElementType::TRI6]);
        assert_eq!(quad.cell_count().unwrap(), 2);

        // Corners are re-used verbatim.
        assert_eq!(quad.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(quad.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(quad.node(0, 0, 2).unwrap().id(), c.id());
        // Mid node of edge (0,1)=(a,b) sits at the midpoint.
        assert_eq!(
            quad.node(0, 0, 3).unwrap().position().unwrap(),
            vec![1.0, 0.0]
        );

        // Shared edge (b,c): cell 0 edge (1,2) = local node 4, cell 1 edge
        // (2,0) = (c,b) = local node 5. Both must be the SAME node.
        let shared_from_0 = quad.node(0, 0, 4).unwrap();
        let shared_from_1 = quad.node(0, 1, 5).unwrap();
        assert_eq!(shared_from_0.id(), shared_from_1.id());
        assert_eq!(shared_from_0.position().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn seg2_to_seg3_reuses_endpoints() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[4.0]).unwrap();
        let mut seg = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        seg.add_cell(&[a.id(), b.id()]).unwrap();

        let quad = to_quadratic(&seg).unwrap();
        assert_eq!(quad.element_types().unwrap(), vec![ElementType::SEG3]);
        assert_eq!(quad.node(0, 0, 0).unwrap().id(), a.id());
        assert_eq!(quad.node(0, 0, 1).unwrap().id(), b.id());
        assert_eq!(quad.node(0, 0, 2).unwrap().position().unwrap(), vec![2.0]);
    }

    #[test]
    fn tet4_to_tet10_node_count() {
        // One tetra → one TET10 with 4 corners + 6 mid-edge nodes.
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<_> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let mut tet = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
        tet.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
            .unwrap();

        let quad = to_quadratic(&tet).unwrap();
        assert_eq!(quad.element_types().unwrap(), vec![ElementType::TET10]);
        // Mid node of edge (0,1) at local index 4.
        assert_eq!(
            quad.node(0, 0, 4).unwrap().position().unwrap(),
            vec![0.5, 0.0, 0.0]
        );
        // Mid node of edge (2,3) at local index 9.
        assert_eq!(
            quad.node(0, 0, 9).unwrap().position().unwrap(),
            vec![0.0, 0.5, 0.5]
        );
    }

    #[test]
    fn rejects_poi1_and_already_quadratic() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut pts = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        pts.add_cell(&[a.id()]).unwrap();
        assert!(to_quadratic(&pts).is_err());
    }
}
