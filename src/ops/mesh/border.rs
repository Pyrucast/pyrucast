//! Boundary-extraction operator: the border loop(s) of a surface mesh.
//!
//! [`border`] is the inverse companion of
//! [`crate::ops::mesh::triangulate_surface()`]
//! (whose input is one or more closed SEG2 loops): it takes a surface mesh
//! (TRI3 / QUA4 cells) and returns its boundary as one or more closed SEG2
//! loops, one per submesh.
//!
//! A *boundary* edge is an element edge used by exactly one cell; interior
//! edges are shared by two cells (with opposite orientations) and cancel.
//! The boundary edges are then chained into closed loops: a simply-connected
//! domain yields a single loop, a domain with holes or several disjoint
//! pieces yields several — hence several submeshes.
//!
//! With an `angle_deg` given, each closed loop is further split into open
//! *arêtes* (edges) at its **corner** nodes — where the boundary changes
//! direction by more than `angle_deg` degrees — the 1-D analogue of the
//! dihedral-angle face splitting done by [`crate::ops::mesh::skin()`].

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::store::Handle;
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
/// [`triangulate_surface()`](fn@crate::ops::mesh::triangulate_surface) yields a
/// single shared boundary — then chained into closed loops.
///
/// The result is a [`Mesh`] with **one SEG2 submesh per loop**: one submesh
/// for a simply-connected domain, several when the domain has holes or
/// disjoint components. Each loop keeps the CCW boundary orientation, so the
/// outer loop runs counter-clockwise and hole loops clockwise — the
/// orientation [`triangulate_surface()`](fn@crate::ops::mesh::triangulate_surface)
/// expects of its input contour. The original nodes are reused (and re-referenced).
///
/// With `angle_deg = Some(a)`, every closed loop is split into open **arêtes**
/// at its corner nodes: a node is a corner when the boundary turns by more
/// than `a` degrees there (the angle between the incoming and outgoing edge
/// directions). Each arête — a maximal run of near-collinear segments between
/// two corners — becomes its own `SEG2` submesh (a straight side of a square
/// yields one arête, so a square boundary yields four). A loop with no corner
/// (a smoothly curved boundary whose every turn stays under `a`) is kept as a
/// single closed loop. `angle_deg = None` keeps every boundary as one closed
/// loop (no splitting) — the default.
///
/// POI1 submeshes are ignored (a point has no edge). Errors if the mesh has
/// no surface cells, if it carries cells that are neither POI1, TRI3 nor
/// QUA4 (1-D and 3-D borders are not handled here — see
/// [`skin()`](fn@crate::ops::mesh::skin) for the boundary of a volume),
/// or if the boundary is not a clean set of closed loops (an open or
/// non-manifold edge).
pub fn border(mesh: &Mesh, angle_deg: Option<f64>) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // 1. Count every element edge across all surface submeshes. A boundary
    //    edge is one whose undirected key occurs exactly once.
    let mut counts: HashMap<(NodeId, NodeId), u32> = HashMap::new();
    let mut any_surface = false;
    for sm in mesh {
        let (et, conn) = {
            let s = sm.read();
            (s.element_type(), s.connectivity().to_vec())
        };
        match et {
            ElementType::POI1 => continue, // no edges — tolerated
            ElementType::TRI3 | ElementType::QUA4 => {}
            other => {
                return Err(PyrucastError::Message(format!(
                    "border: only surface meshes (TRI3/QUA4) are supported, got {}",
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
            "border: mesh has no surface cells (TRI3/QUA4)".into(),
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
            let s = sm.read();
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

    // 3. Chain the boundary edges into closed loops (node-id sequences).
    let mut loops: Vec<Vec<NodeId>> = Vec::new();
    for &start in &order {
        // Skip a from-node whose boundary edge was already consumed.
        if adj.get(&start).is_none_or(|outs| outs.is_empty()) {
            continue;
        }
        let mut chain = Vec::new();
        let mut current = start;
        loop {
            let outs = adj
                .get_mut(&current)
                .filter(|o| !o.is_empty())
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "border: boundary is not a closed loop (open or non-manifold at node {})",
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
        loops.push(chain);
    }

    // 4. Emit one SEG2 submesh per loop, or — when an angle is given — per
    //    open arête once each loop is split at its corner nodes.
    let mut result = Mesh::empty();
    match angle_deg {
        None => {
            for chain in &loops {
                emit_closed(&mut result, &coords, chain)?;
            }
        }
        Some(a) => {
            let cos_tol = a.to_radians().cos();
            // Decide the split under the coords read guard (positions only);
            // build the submeshes afterwards, since `add_cell` write-locks
            // `Coords` and must not run while the read guard is held.
            let mut pieces: Vec<(Vec<NodeId>, bool)> = Vec::new(); // (nodes, closed)
            {
                let c = coords.read();
                for chain in &loops {
                    let corners = corner_indices(&c, chain, cos_tol)?;
                    if corners.is_empty() {
                        pieces.push((chain.clone(), true)); // smooth loop: keep closed
                    } else {
                        for arc in split_at_corners(chain, &corners) {
                            pieces.push((arc, false));
                        }
                    }
                }
            }
            for (nodes, closed) in &pieces {
                if *closed {
                    emit_closed(&mut result, &coords, nodes)?;
                } else {
                    emit_open(&mut result, &coords, nodes)?;
                }
            }
        }
    }
    Ok(result)
}

/// Append a **closed** SEG2 loop (segments wrap from the last node back to the
/// first) as one submesh of `result`.
fn emit_closed(result: &mut Mesh, coords: &Handle<Coords>, chain: &[NodeId]) -> Result<()> {
    let mut sub = SubMesh::new(coords.clone(), ElementType::SEG2);
    let n = chain.len();
    for i in 0..n {
        sub.add_cell(&[chain[i], chain[(i + 1) % n]])?;
    }
    result.add_sub(Handle::new(sub))
}

/// Append an **open** SEG2 polyline (an arête: no wrap-around) as one submesh.
fn emit_open(result: &mut Mesh, coords: &Handle<Coords>, arc: &[NodeId]) -> Result<()> {
    let mut sub = SubMesh::new(coords.clone(), ElementType::SEG2);
    for pair in arc.windows(2) {
        sub.add_cell(pair)?;
    }
    result.add_sub(Handle::new(sub))
}

/// Indices, into the cyclic `chain`, of the **corner** nodes: those where the
/// boundary turns by more than the tolerance (dot of the unit incoming and
/// outgoing edge directions below `cos_tol`). Degenerate (zero-length) edges
/// never make a corner.
fn corner_indices(c: &Coords, chain: &[NodeId], cos_tol: f64) -> Result<Vec<usize>> {
    let k = chain.len();
    let mut corners = Vec::new();
    for i in 0..k {
        let prev = c.position(chain[(i + k - 1) % k])?;
        let cur = c.position(chain[i])?;
        let next = c.position(chain[(i + 1) % k])?;
        if let Some(cos_turn) = turn_cosine(prev, cur, next) {
            if cos_turn < cos_tol {
                corners.push(i);
            }
        }
    }
    Ok(corners)
}

/// Cosine of the turning angle at `cur` between the incoming edge
/// (`prev`→`cur`) and the outgoing edge (`cur`→`next`), or `None` if either
/// edge has zero length. Works in any dimension (2-D or planar 3-D).
fn turn_cosine(prev: &[f64], cur: &[f64], next: &[f64]) -> Option<f64> {
    let din: Vec<f64> = cur.iter().zip(prev).map(|(a, b)| a - b).collect();
    let dout: Vec<f64> = next.iter().zip(cur).map(|(a, b)| a - b).collect();
    let dot: f64 = din.iter().zip(&dout).map(|(a, b)| a * b).sum();
    let nin = din.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nout = dout.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nin == 0.0 || nout == 0.0 {
        return None;
    }
    Some((dot / (nin * nout)).clamp(-1.0, 1.0))
}

/// Split a cyclic `chain` at the given `corners` (indices, non-empty) into
/// open arêtes. Each arête runs from one corner to the next (inclusive of both
/// endpoints), following the loop's order; with a single corner the whole loop
/// becomes one open polyline starting and ending at that corner.
fn split_at_corners(chain: &[NodeId], corners: &[usize]) -> Vec<Vec<NodeId>> {
    let k = chain.len();
    let m = corners.len();
    let mut arcs = Vec::with_capacity(m);
    for t in 0..m {
        let a = corners[t];
        let b = corners[(t + 1) % m];
        let mut arc = vec![chain[a]];
        let mut idx = a;
        loop {
            idx = (idx + 1) % k;
            arc.push(chain[idx]);
            if idx == b {
                break;
            }
        }
        arcs.push(arc);
    }
    arcs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::store::Handle;

    /// signed area of a node-id loop, read from `coords` (2-D).
    fn signed_area(coords: &crate::store::Handle<Coords>, loop_ids: &[NodeId]) -> f64 {
        let c = coords.read();
        let n = loop_ids.len();
        let mut a = 0.0;
        for i in 0..n {
            let p = c.position(loop_ids[i]).unwrap();
            let q = c.position(loop_ids[(i + 1) % n]).unwrap();
            a += p[0] * q[1] - q[0] * p[1];
        }
        a / 2.0
    }

    /// Ordered node-id loop of submesh `s` (chains its SEG2 connectivity).
    fn loop_of(mesh: &Mesh, s: usize) -> Vec<NodeId> {
        let sm = mesh.get(s).unwrap();
        let conn = sm.read().connectivity().to_vec();
        // The submesh is stored as consecutive segments (a,b)(b,c)…; the
        // first node of each segment, in order, is the loop.
        conn.chunks(2).map(|seg| seg[0]).collect()
    }

    #[test]
    fn single_triangle_gives_one_triloop() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let ct = border(&m, None).unwrap();
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), d.id()]).unwrap();
        m.add_cell(&[b.id(), c.id(), d.id()]).unwrap();

        let ct = border(&m, None).unwrap();
        assert_eq!(ct.len(), 1);
        assert_eq!(ct.cell_counts().unwrap(), vec![4]);
    }

    #[test]
    fn square_split_by_angle_gives_four_open_aretes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        m.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();

        // Four right-angle corners → four open arêtes, one segment each
        // (a closed loop would be a single 4-segment submesh).
        let ar = border(&m, Some(45.0)).unwrap();
        assert_eq!(ar.len(), 4);
        assert_eq!(ar.cell_counts().unwrap(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn collinear_boundary_segments_stay_in_one_arete() {
        // Two unit squares side by side (2×1): the mid nodes on the top and
        // bottom edges are collinear, so those edges are not split at them.
        let coords = Handle::new(Coords::new(2).unwrap());
        let bl = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let bm = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let br = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let tr = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        let tm = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let tl = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        m.add_cell(&[bl.id(), bm.id(), tm.id(), tl.id()]).unwrap();
        m.add_cell(&[bm.id(), br.id(), tr.id(), tm.id()]).unwrap();

        // Four rectangle corners → four arêtes; the bottom and top runs each
        // carry two collinear segments, the sides one.
        let ar = border(&m, Some(45.0)).unwrap();
        assert_eq!(ar.len(), 4);
        let mut sizes = ar.cell_counts().unwrap();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 1, 2, 2]);
    }

    #[test]
    fn large_angle_keeps_a_triangle_boundary_closed() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        // Every vertex turns by ≤135°, below a 170° threshold → no corner,
        // so the boundary stays a single closed loop (3 segments).
        let closed = border(&m, Some(170.0)).unwrap();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed.cell_counts().unwrap(), vec![3]);

        // A small threshold does split it, into three open arêtes.
        let split = border(&m, Some(30.0)).unwrap();
        assert_eq!(split.len(), 3);
        assert_eq!(split.cell_counts().unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn grid_with_hole_gives_two_loops() {
        // 3×3 grid of QUA4 with the centre cell removed: outer loop (12
        // segments) + inner hole loop (4 segments).
        let coords = Handle::new(Coords::new(2).unwrap());
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

        let ct = border(&m, None).unwrap();
        assert_eq!(ct.len(), 2, "outer loop + hole loop");
        let mut sizes = ct.cell_counts().unwrap();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![4, 12]);

        // Outer loop CCW (area > 0), hole loop CW (area < 0).
        let areas: Vec<f64> = (0..2)
            .map(|s| signed_area(&coords, &loop_of(&ct, s)))
            .collect();
        assert!(areas.iter().any(|&a| a > 0.0), "an outer (CCW) loop");
        assert!(areas.iter().any(|&a| a < 0.0), "a hole (CW) loop");
    }

    #[test]
    fn reuses_nodes_and_increfs() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        // a: TRI3 + Node = 2 before contour.
        assert_eq!(coords.read().refcount(a.id()), 2);
        let ct = border(&m, None).unwrap();
        // Boundary SEG2 references a in two segments (incoming + outgoing).
        assert_eq!(coords.read().refcount(a.id()), 4);
        drop(ct);
        assert_eq!(coords.read().refcount(a.id()), 2);
    }

    #[test]
    fn no_surface_cells_is_error() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        assert!(border(&m, None).is_err());
    }

    #[test]
    fn volume_cells_are_rejected() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let ns: Vec<NodeId> = (0..4)
            .map(|k| {
                Node::create_in(coords.clone(), &[k as f64, 0.0, 0.0])
                    .unwrap()
                    .id()
            })
            .collect();
        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
        m.add_cell(&ns).unwrap();
        assert!(border(&m, None).is_err());
    }
}
