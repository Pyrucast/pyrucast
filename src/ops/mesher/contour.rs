//! Boundary-extraction operator: the contour(s) of a surface mesh.
//!
//! [`contour`] is the inverse companion of
//! [`crate::ops::mesher::surface()`] / [`crate::ops::mesher::fill_surface()`]
//! (whose input is exactly one closed SEG2 loop): it takes a surface mesh
//! (TRI3 / QUA4 cells) and returns its boundary as one or more closed SEG2
//! loops, one per submesh.
//!
//! A *boundary* edge is an element edge used by exactly one cell; interior
//! edges are shared by two cells (with opposite orientations) and cancel.
//! The boundary edges are then chained into closed loops: a simply-connected
//! domain yields a single loop, a domain with holes or several disjoint
//! pieces yields several — hence several submeshes.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{ElementType, Mesh, NodeId, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read};
use std::collections::HashMap;

/// Undirected edge key: the two node ids sorted, so an edge and its reverse
/// share the same key (and interior edges collide).
fn edge_key(u: NodeId, v: NodeId) -> (NodeId, NodeId) {
    if u.0 <= v.0 {
        (u, v)
    } else {
        (v, u)
    }
}

/// Extract the boundary of a **surface** mesh (TRI3 / QUA4 cells) as one or
/// more closed SEG2 loops.
///
/// Every element edge is taken with the cell's CCW orientation; an edge used
/// by exactly one cell is a *boundary* edge (interior edges appear twice,
/// with opposite orientations, and cancel). Boundary edges from all surface
/// submeshes are pooled together — so the QUA4 + TRI3 output of
/// [`surface()`](crate::ops::mesher::surface) yields a single shared
/// boundary — then chained into closed loops.
///
/// The result is a [`Mesh`] with **one SEG2 submesh per loop**: one submesh
/// for a simply-connected domain, several when the domain has holes or
/// disjoint components. Each loop keeps the CCW boundary orientation, so the
/// outer loop runs counter-clockwise and hole loops clockwise — the
/// orientation [`surface()`](crate::ops::mesher::surface) expects of its
/// input contour. The original nodes are reused (and re-referenced).
///
/// POI1 submeshes are ignored (a point has no edge). Errors if the mesh has
/// no surface cells, if it carries cells that are neither POI1, TRI3 nor
/// QUA4 (1-D and 3-D contours are not handled yet), or if the boundary is
/// not a clean set of closed loops (an open or non-manifold edge).
pub fn contour(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // 1. Count every element edge across all surface submeshes. A boundary
    //    edge is one whose undirected key occurs exactly once.
    let mut counts: HashMap<(NodeId, NodeId), u32> = HashMap::new();
    let mut any_surface = false;
    for sm in mesh {
        let (et, conn) = {
            let s = read(sm)?;
            (s.element_type(), s.connectivity().to_vec())
        };
        match et {
            ElementType::POI1 => continue, // no edges — tolerated
            ElementType::TRI3 | ElementType::QUA4 => {}
            other => {
                return Err(PyrucastError::Message(format!(
                    "contour: only surface meshes (TRI3/QUA4) are supported, got {}",
                    other
                )));
            }
        }
        any_surface = true;
        let npc = et.nodes_per_cell();
        for cell in conn.chunks(npc) {
            for i in 0..npc {
                let key = edge_key(cell[i], cell[(i + 1) % npc]);
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    if !any_surface {
        return Err(PyrucastError::Message(
            "contour: mesh has no surface cells (TRI3/QUA4)".into(),
        ));
    }

    // 2. Re-walk the cells in order, collecting boundary edges with their
    //    cell orientation. `adj` maps a from-node to the to-nodes still to be
    //    consumed; `order` keeps the from-nodes in a deterministic order so
    //    the loops (and their starts) are reproducible.
    let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut order: Vec<NodeId> = Vec::new();
    for sm in mesh {
        let (et, conn) = {
            let s = read(sm)?;
            (s.element_type(), s.connectivity().to_vec())
        };
        if et == ElementType::POI1 {
            continue;
        }
        let npc = et.nodes_per_cell();
        for cell in conn.chunks(npc) {
            for i in 0..npc {
                let (u, v) = (cell[i], cell[(i + 1) % npc]);
                if counts.get(&edge_key(u, v)) == Some(&1) {
                    adj.entry(u).or_default().push(v);
                    order.push(u);
                }
            }
        }
    }

    // 3. Chain the boundary edges into closed loops.
    let mut result = Mesh::empty();
    for &start in &order {
        // Skip a from-node whose boundary edge was already consumed.
        if adj.get(&start).is_none_or(|outs| outs.is_empty()) {
            continue;
        }
        let mut chain = Vec::new();
        let mut current = start;
        loop {
            let outs = adj.get_mut(&current).filter(|o| !o.is_empty()).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "contour: boundary is not a closed loop (open or non-manifold at node {})",
                    current
                ))
            })?;
            let next = outs.remove(0);
            chain.push(current);
            if next == start {
                break;
            }
            current = next;
        }
        let mut sub = SubMesh::new(coords.clone(), ElementType::SEG2);
        let n = chain.len();
        for i in 0..n {
            sub.add_cell(&[chain[i], chain[(i + 1) % n]])?;
        }
        result.add_sub(insert(sub))?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, Node};
    use crate::store::insert;

    /// signed area of a node-id loop, read from `coords` (2-D).
    fn signed_area(coords: &crate::store::Handle<Coords>, loop_ids: &[NodeId]) -> f64 {
        let c = read(coords).unwrap();
        let n = loop_ids.len();
        let mut a = 0.0;
        for i in 0..n {
            let p = c.coord(loop_ids[i]).unwrap();
            let q = c.coord(loop_ids[(i + 1) % n]).unwrap();
            a += p[0] * q[1] - q[0] * p[1];
        }
        a / 2.0
    }

    /// Ordered node-id loop of submesh `s` (chains its SEG2 connectivity).
    fn loop_of(mesh: &Mesh, s: usize) -> Vec<NodeId> {
        let sm = mesh.get(s).unwrap();
        let conn = read(&sm).unwrap().connectivity().to_vec();
        // The submesh is stored as consecutive segments (a,b)(b,c)…; the
        // first node of each segment, in order, is the loop.
        conn.chunks(2).map(|seg| seg[0]).collect()
    }

    #[test]
    fn single_triangle_gives_one_triloop() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let ct = contour(&m).unwrap();
        assert_eq!(ct.len(), 1);
        assert_eq!(ct.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(ct.cell_counts().unwrap(), vec![3]);
        // CCW input → CCW boundary (positive signed area).
        assert!(signed_area(&coords, &loop_of(&ct, 0)) > 0.0);
    }

    #[test]
    fn two_triangles_sharing_diagonal_give_one_quadloop() {
        // Unit square split into two triangles along the (b,d) diagonal; the
        // shared diagonal is interior and must not appear in the contour.
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), d.id()]).unwrap();
        m.add_cell(&[b.id(), c.id(), d.id()]).unwrap();

        let ct = contour(&m).unwrap();
        assert_eq!(ct.len(), 1);
        assert_eq!(ct.cell_counts().unwrap(), vec![4]);
    }

    #[test]
    fn grid_with_hole_gives_two_loops() {
        // 3×3 grid of QUA4 with the centre cell removed: outer loop (12
        // segments) + inner hole loop (4 segments).
        let coords = insert(Coords::new(2).unwrap());
        let mut ids = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                ids.push(
                    Node::create_in(coords.clone(), &[i as f64, j as f64])
                        .unwrap()
                        .id(),
                );
            }
        }
        let at = |i: usize, j: usize| ids[j * 4 + i];
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        for j in 0..3 {
            for i in 0..3 {
                if i == 1 && j == 1 {
                    continue; // drill the hole
                }
                // CCW: (i,j)(i+1,j)(i+1,j+1)(i,j+1)
                m.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                    .unwrap();
            }
        }

        let ct = contour(&m).unwrap();
        assert_eq!(ct.len(), 2, "outer loop + hole loop");
        let mut sizes = ct.cell_counts().unwrap();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![4, 12]);

        // Outer loop CCW (area > 0), hole loop CW (area < 0).
        let areas: Vec<f64> = (0..2).map(|s| signed_area(&coords, &loop_of(&ct, s))).collect();
        assert!(areas.iter().any(|&a| a > 0.0), "an outer (CCW) loop");
        assert!(areas.iter().any(|&a| a < 0.0), "a hole (CW) loop");
    }

    #[test]
    fn reuses_nodes_and_increfs() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        // a: TRI3 + Node = 2 before contour.
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
        let ct = contour(&m).unwrap();
        // Boundary SEG2 references a in two segments (incoming + outgoing).
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 4);
        drop(ct);
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
    }

    #[test]
    fn no_surface_cells_is_error() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        assert!(contour(&m).is_err());
    }

    #[test]
    fn volume_cells_are_rejected() {
        let coords = insert(Coords::new(3).unwrap());
        let ns: Vec<NodeId> = (0..4)
            .map(|k| {
                Node::create_in(coords.clone(), &[k as f64, 0.0, 0.0])
                    .unwrap()
                    .id()
            })
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
        m.add_cell(&ns).unwrap();
        assert!(contour(&m).is_err());
    }
}
