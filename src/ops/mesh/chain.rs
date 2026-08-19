//! Chain ordering: [`chain`] re-sorts a line mesh's cells head-to-tail.
//!
//! [`orient`](fn@super::orient) makes the cells of a curve *point* the same
//! way, but leaves them where they are in the connectivity; `chain` is the
//! complement that puts them in **traversal order**, so that reading the cells
//! one after the other walks the curve from one end to the other:
//!
//! ```text
//!   5 6                    1 2
//!   1 2      chain  →      2 3
//!   3 4                    3 4
//!   2 3                    4 5
//!   4 5                    5 6
//! ```
//!
//! Each submesh is chained **independently** (it keeps its cells, its element
//! type and its face colour) and must be a single continuous chain: every node
//! carries one or two segments, and the cells form **one** connected piece,
//! either open — two free ends — or closed into a loop. Anything else is an
//! error: a node with three segments (branching), or several disjoint pieces.
//! Split such a mesh into one submesh per branch first.
//!
//! Cells are flipped as needed along the way (a `SEG3`'s mid node stays in the
//! middle, [`ElementType::reversal_permutation`] does the work), so `chain`
//! also orients — no need to call `orient` first.
//!
//! # Where the chain starts
//!
//! An open chain is read from a free end. If exactly one of the two ends is
//! already the **tail** of its segment, that one starts the walk: a curve that
//! was already consistently oriented keeps its direction, and `chain` is then a
//! pure permutation of the cells. Otherwise (both ends are tails, or neither
//! is) the end with the lowest node id starts, which keeps the result
//! deterministic. A closed loop has no free end: it starts at its lowest node
//! id, leaving in the direction of the segment that already has it as tail.
//!
//! Like the other mesher operators, `chain` builds a **fresh mesh** mirroring
//! the input submesh by submesh (same element types, same face colours, same
//! shared `Coords` and node ids); the input is left untouched.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::collections::HashMap;

/// Re-order the cells of a line mesh into a continuous chain.
///
/// Each `SEG2`/`SEG3` submesh is sorted so that consecutive cells share a node
/// — the connectivity reads as a walk along the curve — and flipped where
/// needed so each cell leaves where the previous one arrived. The companion of
/// [`orient`](fn@super::orient), which fixes the cells' direction but not
/// their order.
///
/// The submesh must form **one** continuous chain, open or closed; a node
/// shared by three segments (branching) or several disjoint pieces are
/// rejected. Non-line submeshes (`POI1`, surfaces, volumes) are rejected too.
/// The result mirrors `mesh` (same submeshes, types, colours, shared
/// `Coords`); the input is untouched.
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
/// // Réordonne des segments épars en une ligne continue. Une ligne déjà
/// // chaînée en ressort inchangée.
/// let l = mesh::line(&p(&[0.0, 0.0, 0.0]), &p(&[3.0, 0.0, 0.0]), 3, ElementType::SEG2)?;
/// assert_eq!(mesh::chain(&l)?.cell_count()?, 3);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn chain(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = sm_handle.read();
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        if !matches!(et, ElementType::SEG2 | ElementType::SEG3) {
            return Err(PyrucastError::Message(format!(
                "chain: expects a line mesh (SEG2 or SEG3), got {et}"
            )));
        }
        let npc = et.nodes_per_cell();
        let cells: Vec<&[NodeId]> = conn.chunks(npc).collect();
        let order = chain_order(&cells)?;

        let perm = et.reversal_permutation();
        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(color);
        for (cell, reversed) in order {
            let nodes: Vec<NodeId> = if reversed {
                perm.iter().map(|&i| cells[cell][i]).collect()
            } else {
                cells[cell].to_vec()
            };
            new_sm.add_cell(&nodes)?;
        }
        result.add_sub(Handle::new(new_sm))?;
    }
    Ok(result)
}

/// The traversal order of one submesh's cells, as `(cell index, reversed)`
/// pairs — `reversed` telling whether the cell must be flipped to leave from
/// the node the walk has reached.
fn chain_order(cells: &[&[NodeId]]) -> Result<Vec<(usize, bool)>> {
    if cells.is_empty() {
        return Ok(Vec::new());
    }
    // A cell's two ends are its corners; a SEG3's mid node plays no part here.
    let ends = |c: usize| (cells[c][0], cells[c][1]);

    // ── 1. Node → the cells it ends, rejecting branching on the way.
    let mut incident: HashMap<NodeId, Vec<usize>> = HashMap::new();
    for (c, cell) in cells.iter().enumerate() {
        if cell[0] == cell[1] {
            return Err(PyrucastError::Message(format!(
                "chain: cell {c} starts and ends on node {}",
                cell[0].0
            )));
        }
        for &n in &cell[..2] {
            let at = incident.entry(n).or_default();
            at.push(c);
            if at.len() > 2 {
                return Err(PyrucastError::Message(format!(
                    "chain: node {} carries {} segments; a chain admits at most 2 \
                     (split the branches into separate submeshes)",
                    n.0,
                    at.len()
                )));
            }
        }
    }

    // ── 2. Where to start: a free end for an open chain (see module doc), the
    //       lowest node id for a closed loop.
    let mut free: Vec<NodeId> = incident
        .iter()
        .filter(|(_, at)| at.len() == 1)
        .map(|(&n, _)| n)
        .collect();
    free.sort_unstable_by_key(|n| n.0);
    let start = match free.len() {
        // Closed loop (or a single piece of it — connectivity is checked at
        // the end anyway).
        0 => *incident.keys().min_by_key(|n| n.0).unwrap(),
        2 => {
            let tail = |n: NodeId| ends(incident[&n][0]).0 == n;
            match (tail(free[0]), tail(free[1])) {
                (false, true) => free[1],
                _ => free[0],
            }
        }
        // With at most two segments per node the mesh is a disjoint union of
        // paths and loops, so free ends come in pairs: more than two of them
        // means more than one path.
        k => {
            return Err(PyrucastError::Message(format!(
                "chain: {k} free ends found; the submesh is not one continuous \
                 chain (disjoint pieces)"
            )))
        }
    };

    // ── 3. Walk from `start`, taking at each node the segment not yet used.
    //       At the start node, prefer the one that already has it as tail so an
    //       already-oriented loop keeps its direction.
    let mut used = vec![false; cells.len()];
    let mut order = Vec::with_capacity(cells.len());
    let mut current = start;
    loop {
        let at = &incident[&current];
        let next = at
            .iter()
            .copied()
            .filter(|&c| !used[c])
            .min_by_key(|&c| (ends(c).0 != current, c));
        let Some(c) = next else { break };
        let (a, b) = ends(c);
        let reversed = b == current;
        used[c] = true;
        order.push((c, reversed));
        current = if reversed { a } else { b };
    }

    if order.len() != cells.len() {
        return Err(PyrucastError::Message(format!(
            "chain: only {} of the {} cells are reachable from the chain's \
             start; the submesh is not one continuous chain (disjoint pieces)",
            order.len(),
            cells.len()
        )));
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A `Coords` holding `n` nodes on the x axis, node id `i` at `x = i`, so
    /// tests can name nodes by their index.
    fn line_nodes(n: usize) -> (Handle<Coords>, Vec<NodeId>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let ids = (0..n)
            .map(|i| {
                Node::create_in(coords.clone(), &[i as f64, 0.0])
                    .unwrap()
                    .id()
            })
            .collect();
        (coords, ids)
    }

    fn conn_of(mesh: &Mesh, sub: usize) -> Vec<Vec<u32>> {
        mesh.cells(sub)
            .unwrap()
            .map(|c| c.node_ids().unwrap().into_iter().map(|n| n.0).collect())
            .collect()
    }

    fn seg2_mesh(coords: &Handle<Coords>, cells: &[[NodeId; 2]]) -> Mesh {
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        for c in cells {
            sm.add_cell(c).unwrap();
        }
        Mesh::from_submesh(sm)
    }

    #[test]
    fn chain_sorts_a_shuffled_open_chain() {
        let (coords, n) = line_nodes(5);
        // Fed out of order: 3→4, 0→1, 2→3, 1→2.
        let mesh = seg2_mesh(
            &coords,
            &[[n[3], n[4]], [n[0], n[1]], [n[2], n[3]], [n[1], n[2]]],
        );

        let out = conn_of(&chain(&mesh).unwrap(), 0);
        assert_eq!(
            out,
            vec![
                vec![n[0].0, n[1].0],
                vec![n[1].0, n[2].0],
                vec![n[2].0, n[3].0],
                vec![n[3].0, n[4].0],
            ]
        );
        // Idempotent: chaining an already-chained mesh changes nothing.
        assert_eq!(conn_of(&chain(&chain(&mesh).unwrap()).unwrap(), 0), out);
    }

    #[test]
    fn chain_flips_the_cells_it_walks_backwards() {
        let (coords, n) = line_nodes(4);
        // Middle segment fed reversed, last one too.
        let mesh = seg2_mesh(&coords, &[[n[0], n[1]], [n[2], n[1]], [n[3], n[2]]]);

        let out = conn_of(&chain(&mesh).unwrap(), 0);
        assert_eq!(
            out,
            vec![
                vec![n[0].0, n[1].0],
                vec![n[1].0, n[2].0],
                vec![n[2].0, n[3].0],
            ]
        );
    }

    #[test]
    fn chain_starts_an_open_chain_at_the_tail_end() {
        let (coords, n) = line_nodes(3);
        // Consistently oriented 2→1→0: the walk must keep that direction
        // instead of restarting from the lower id.
        let mesh = seg2_mesh(&coords, &[[n[1], n[0]], [n[2], n[1]]]);

        let out = conn_of(&chain(&mesh).unwrap(), 0);
        assert_eq!(out, vec![vec![n[2].0, n[1].0], vec![n[1].0, n[0].0]]);
    }

    #[test]
    fn chain_walks_a_closed_loop_from_its_lowest_node() {
        let (coords, n) = line_nodes(4);
        let mesh = seg2_mesh(
            &coords,
            &[[n[2], n[3]], [n[1], n[2]], [n[3], n[0]], [n[0], n[1]]],
        );

        let out = conn_of(&chain(&mesh).unwrap(), 0);
        assert_eq!(
            out,
            vec![
                vec![n[0].0, n[1].0],
                vec![n[1].0, n[2].0],
                vec![n[2].0, n[3].0],
                vec![n[3].0, n[0].0],
            ]
        );
    }

    #[test]
    fn chain_keeps_seg3_mid_nodes_in_the_middle() {
        let (coords, n) = line_nodes(5); // 0,2,4 corners; 1,3 mid nodes.
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG3);
        sm.add_cell(&[n[4], n[2], n[3]]).unwrap(); // 2→4 fed reversed
        sm.add_cell(&[n[0], n[2], n[1]]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let out = conn_of(&chain(&mesh).unwrap(), 0);
        assert_eq!(
            out,
            vec![vec![n[0].0, n[2].0, n[1].0], vec![n[2].0, n[4].0, n[3].0],]
        );
    }

    #[test]
    fn chain_mirrors_the_submeshes_and_shares_the_nodes() {
        let (coords, n) = line_nodes(4);
        let mut a = SubMesh::new(coords.clone(), ElementType::SEG2);
        a.add_cell(&[n[1], n[2]]).unwrap();
        a.add_cell(&[n[0], n[1]]).unwrap();
        let mut b = SubMesh::new(coords.clone(), ElementType::SEG2);
        b.add_cell(&[n[2], n[3]]).unwrap();
        let mut mesh = Mesh::from_submesh(a);
        mesh.add_sub(Handle::new(b)).unwrap();

        // Each submesh is chained on its own: two submeshes in, two out.
        let out = chain(&mesh).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.coords().unwrap().same_object(&coords));
        assert_eq!(
            conn_of(&out, 0),
            vec![vec![n[0].0, n[1].0], vec![n[1].0, n[2].0]]
        );
        assert_eq!(conn_of(&out, 1), vec![vec![n[2].0, n[3].0]]);
    }

    #[test]
    fn chain_rejects_branching_disjoint_pieces_and_surfaces() {
        let (coords, n) = line_nodes(4);

        // Branching: node 1 carries three segments.
        let star = seg2_mesh(&coords, &[[n[0], n[1]], [n[1], n[2]], [n[1], n[3]]]);
        let err = chain(&star).unwrap_err().to_string();
        assert!(err.contains("3 segments"), "{err}");

        // Two disjoint pieces: 0→1 apart from 2→3.
        let split = seg2_mesh(&coords, &[[n[0], n[1]], [n[2], n[3]]]);
        let err = chain(&split).unwrap_err().to_string();
        assert!(err.contains("disjoint"), "{err}");

        // A closed loop plus a disjoint segment: no free end, caught by the
        // reachability check.
        let (coords5, m) = line_nodes(5);
        let loop_plus = seg2_mesh(
            &coords5,
            &[[m[0], m[1]], [m[1], m[2]], [m[2], m[0]], [m[3], m[4]]],
        );
        let err = chain(&loop_plus).unwrap_err().to_string();
        assert!(err.contains("disjoint"), "{err}");

        // Not a line mesh at all.
        let mut tri = SubMesh::new(coords.clone(), ElementType::TRI3);
        tri.add_cell(&[n[0], n[1], n[2]]).unwrap();
        let err = chain(&Mesh::from_submesh(tri)).unwrap_err().to_string();
        assert!(err.contains("SEG2 or SEG3"), "{err}");
    }
}
