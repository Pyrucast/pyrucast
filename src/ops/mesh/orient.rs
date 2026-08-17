//! Orientation clean-up: [`orient`] (harmonise) and [`invert`] (flip).
//!
//! Both rewrite a mesh's connectivity so that its cells carry a **consistent
//! orientation**, or the opposite one — working uniformly on 1-D (`SEG*`), 2-D
//! (`TRI*`/`QUA*`) and 3-D (`TET*`/`PENTA*`/`HEX*`) cells, linear or quadratic.
//!
//! # Uniform framework: oriented facets
//!
//! The oriented boundary of a cell is a signed sum of its **codimension-1
//! facets**:
//!
//! - `SEG*` (d = 1): facets are the two end nodes — tail (sign `-1`) and head
//!   (sign `+1`).
//! - `TRI*` / `QUA*` (d = 2): facets are the oriented edges.
//! - `TET*` / `PENTA*` / `HEX*` (d = 3): facets are the outward-oriented faces.
//!
//! Each occurrence is reduced to a pair `(key, sign)` where `key` is the sorted
//! list of the facet's corner node ids and `sign ∈ {-1, +1}` encodes the
//! facet's orientation relative to a canonical orientation of that key. Two
//! cells sharing a facet are **consistently oriented** iff that facet receives
//! **opposite** signs from them. Facet keys of different arities (1 node vs 2 vs
//! ≥ 3) never collide, so a mixed 1-D/2-D/3-D mesh separates into per-dimension
//! components automatically.
//!
//! [`orient`] propagates a consistent orientation across shared facets by a
//! breadth-first walk of the dual graph (cells linked by shared facets); each
//! connected component is seeded by its lowest-indexed cell, which keeps its
//! orientation (a deterministic, bit-for-bit reproducible choice). It never
//! picks an absolute "outward" sense — apply [`invert`] to flip a whole mesh
//! (e.g. to turn a hole's boundary inside-out). Non-manifold facets (shared by
//! more than two cells) impose no constraint and are skipped.
//!
//! [`invert`] simply applies [`ElementType::reversal_permutation`] to every
//! cell.
//!
//! Like the other mesher operators, both build a **fresh mesh** that mirrors the
//! input submesh by submesh (same element types, same face colours, same shared
//! `Coords` and node ids); the input is left untouched.

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::Result;
use crate::handle::Handle;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Reverse the orientation of **every** cell of `mesh` (cast3m `INVE`).
///
/// Each cell is rewritten through [`ElementType::reversal_permutation`]: a
/// surface cell's winding is flipped, a segment is traversed the other way, a
/// volume cell is mirrored. `POI1` cells (no orientation) pass through
/// unchanged. The result mirrors `mesh` — same submeshes, element types and
/// face colours, sharing its `Coords` and node ids — and `mesh` is untouched.
///
/// This is the unconditional flip; [`orient`] is the consistency pass.
pub fn invert(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;
    let mut result = Mesh::empty();
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = sm_handle.read();
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let perm = et.reversal_permutation();
        let npc = et.nodes_per_cell();
        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(color);
        for chunk in conn.chunks(npc) {
            let reordered: Vec<NodeId> = perm.iter().map(|&i| chunk[i]).collect();
            new_sm.add_cell(&reordered)?;
        }
        result.add_sub(Handle::new(new_sm))?;
    }
    Ok(result)
}

/// Harmonise the orientation of `mesh`'s cells (cast3m `ORIE`).
///
/// Cells that share a facet are made **consistently oriented** (all normals of
/// a surface point the same way, all segments of a curve run head-to-tail, all
/// volume cells share the same handedness). Each connected component is seeded
/// by its lowest-indexed cell, which keeps its orientation; the rest are
/// flipped as needed. The absolute sense is **not** chosen — for a closed
/// surface `orient` leaves it all-outward or all-inward depending on the seed;
/// use [`invert`] to pick the other one (e.g. to define the inside of a hole).
///
/// Works on any dimension (`SEG*`, `TRI*`/`QUA*`, `TET*`/`PENTA*`/`HEX*`,
/// linear or quadratic) and on mixed meshes, which split into per-dimension
/// components. Non-manifold facets (more than two incident cells) impose no
/// constraint. The result mirrors `mesh` (same submeshes, types, colours,
/// shared `Coords`); the input is untouched.
pub fn orient(mesh: &Mesh) -> Result<Mesh> {
    let coords = mesh.coords()?;

    // ── 1. Walk every cell in submesh iteration order, assigning it a global
    //       index and recording the oriented facets of each orientable cell.
    let mut n_cells = 0usize;
    // facet key (sorted corner ids) → list of (global cell index, facet sign).
    let mut facet_map: HashMap<Vec<NodeId>, Vec<(usize, i8)>> = HashMap::new();

    for sm_handle in mesh {
        let (et, conn) = {
            let s = sm_handle.read();
            (s.element_type(), s.connectivity().to_vec())
        };
        let npc = et.nodes_per_cell();
        let nc = et.as_kind().corner_count();
        for chunk in conn.chunks(npc) {
            let gid = n_cells;
            for (key, sign) in oriented_facets(et, &chunk[..nc]) {
                facet_map.entry(key).or_default().push((gid, sign));
            }
            n_cells += 1;
        }
    }

    // ── 2. Dual adjacency from manifold facets (exactly two incident cells).
    //       Edge relation: the two cells must give the facet opposite signs.
    let mut adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n_cells];
    for occ in facet_map.values() {
        if occ.len() != 2 {
            continue; // boundary (1) or non-manifold (>2): no constraint.
        }
        let (a, sa) = occ[0];
        let (b, sb) = occ[1];
        // `same == true`: the two cells currently agree on the facet's sign, so
        // they are inconsistent and exactly one of them must be flipped.
        let same = sa == sb;
        adj[a].push((b, same));
        adj[b].push((a, same));
    }
    // Deterministic neighbour order (independent of HashMap iteration).
    for nbrs in &mut adj {
        nbrs.sort_unstable();
    }

    // ── 3. BFS 2-colouring: `flip[c]` tells whether cell `c` is reversed.
    //       Seed each component with its lowest-indexed cell (flip = false).
    let mut flip = vec![false; n_cells];
    let mut visited = vec![false; n_cells];
    for seed in 0..n_cells {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let mut queue = VecDeque::new();
        queue.push_back(seed);
        while let Some(cur) = queue.pop_front() {
            for &(nb, same) in &adj[cur] {
                // Consistent (opposite signs) ⇒ same flip state; inconsistent
                // (`same`) ⇒ opposite flip state.
                let want = flip[cur] ^ same;
                if !visited[nb] {
                    visited[nb] = true;
                    flip[nb] = want;
                    queue.push_back(nb);
                }
                // A conflicting re-visit (want != flip[nb]) means the component
                // is not consistently orientable; we keep the first assignment.
            }
        }
    }

    // ── 4. Emit the fresh mesh, reversing the cells BFS flagged.
    let mut result = Mesh::empty();
    let mut gid = 0usize;
    for sm_handle in mesh {
        let (et, color, conn) = {
            let s = sm_handle.read();
            (s.element_type(), s.face_color(), s.connectivity().to_vec())
        };
        let perm = et.reversal_permutation();
        let npc = et.nodes_per_cell();
        let mut new_sm = SubMesh::new(coords.clone(), et);
        new_sm.set_face_color(color);
        for chunk in conn.chunks(npc) {
            let reordered: Vec<NodeId> = if flip[gid] {
                perm.iter().map(|&i| chunk[i]).collect()
            } else {
                chunk.to_vec()
            };
            new_sm.add_cell(&reordered)?;
            gid += 1;
        }
        result.add_sub(Handle::new(new_sm))?;
    }
    Ok(result)
}

/// Oriented facets of one cell as `(sorted-corner-ids key, sign)` pairs.
///
/// Reads the element's own facet table
/// ([`ElementKind::facets`](crate::atoms::ElementKind::facets)) and keeps only
/// each facet's **corners**: two adjacent cells agree on the corners of the
/// facet they share whatever their degree, so a quadratic mesh orients exactly
/// like its linear parent.
fn oriented_facets(et: ElementType, corners: &[NodeId]) -> Vec<(Vec<NodeId>, i8)> {
    // d = 1: the two ends. Tail is -1, head is +1; two segments sharing a node
    // are consistent iff it is one's head and the other's tail.
    if matches!(et, ElementType::SEG2 | ElementType::SEG3) {
        return vec![(vec![corners[0]], -1), (vec![corners[1]], 1)];
    }
    et.as_kind()
        .facets()
        .iter()
        .map(|f| {
            let nodes: Vec<NodeId> = f.corners().iter().map(|&i| corners[i]).collect();
            let sign = facet_sign(&nodes);
            let mut key = nodes;
            key.sort_unstable();
            (key, sign)
        })
        .collect()
}

/// Orientation of a facet's directed corner sequence relative to the canonical
/// orientation of its (sorted) key: `+1` or `-1`. Reversing the sequence flips
/// the sign; the value is invariant to the starting corner (cyclic rotation).
fn facet_sign(seq: &[NodeId]) -> i8 {
    let n = seq.len();
    debug_assert!(n >= 2);
    if n == 2 {
        // Edge: canonical direction is min → max.
        return if seq[0] < seq[1] { 1 } else { -1 };
    }
    // Polygon face: anchor at the minimum corner, compare its two neighbours.
    let m = (0..n).min_by_key(|&i| seq[i]).unwrap();
    let fwd = seq[(m + 1) % n];
    let bwd = seq[(m + n - 1) % n];
    if fwd < bwd {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// The orientation-reversing reflection defining the permutation: swap the
    /// first two axes (2-D/3-D), or negate the single axis (`SEG*`, 1-D).
    fn reflect(coord: &[f64]) -> Vec<f64> {
        match coord.len() {
            1 => vec![-coord[0]],
            _ => {
                let mut c = coord.to_vec();
                c.swap(0, 1);
                c
            }
        }
    }

    /// `reversal_permutation` must carry every node — corners AND mid-edge,
    /// face-center and body-center nodes — to the slot the defining reflection
    /// sends it to. This is the definitive proof of the per-type tables.
    #[test]
    fn reversal_permutation_matches_the_reflection() {
        // Every type with a reference frame; `POI1` has no orientation to
        // reverse. Derived from `ElementType::ALL` so a new type is covered
        // the day it is declared.
        for &et in ElementType::ALL
            .iter()
            .filter(|et| et.topological_dim() > 0)
        {
            let nodes = et.as_kind().ref_nodes();
            let p = et.reversal_permutation();
            for i in 0..nodes.len() {
                let want = reflect(nodes[i]);
                let got = nodes[p[i]];
                for (a, b) in want.iter().zip(got.iter()) {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "{et}: node {i} → slot {}: expected {want:?}, got {got:?}",
                        p[i]
                    );
                }
            }
        }
    }

    // ── Helpers for the functional tests ────────────────────────────────────

    fn conn_of(mesh: &Mesh, sub: usize) -> Vec<Vec<u32>> {
        mesh.cells(sub)
            .unwrap()
            .map(|c| c.node_ids().unwrap().into_iter().map(|n| n.0).collect())
            .collect()
    }

    fn signed_area(coords: &Handle<Coords>, tri: &[u32]) -> f64 {
        let c = coords.read();
        let p: Vec<Vec<f64>> = tri
            .iter()
            .map(|&i| c.position(NodeId(i)).unwrap().to_vec())
            .collect();
        0.5 * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
            - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
    }

    fn signed_volume(coords: &Handle<Coords>, tet: &[u32]) -> f64 {
        let c = coords.read();
        let p: Vec<Vec<f64>> = tet
            .iter()
            .map(|&i| c.position(NodeId(i)).unwrap().to_vec())
            .collect();
        let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let d = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
        (a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0])
            + a[2] * (b[0] * d[1] - b[1] * d[0]))
            / 6.0
    }

    #[test]
    fn orient_harmonises_a_mixed_winding_surface() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        // First triangle CCW (the seed). Second shares edge (0,2) but is fed CW.
        sm.add_cell(&[n[0], n[1], n[2]]).unwrap();
        sm.add_cell(&[n[0], n[3], n[2]]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let out = orient(&mesh).unwrap();
        let cells = conn_of(&out, 0);
        for tri in &cells {
            assert!(
                signed_area(&coords, tri) > 0.0,
                "triangle {tri:?} not CCW after orient"
            );
        }
        // Seed kept verbatim; the second triangle reversed to (0,2,3).
        assert_eq!(cells[0], vec![n[0].0, n[1].0, n[2].0]);
        assert_eq!(cells[1], vec![n[0].0, n[2].0, n[3].0]);

        // orient is idempotent.
        let out2 = orient(&out).unwrap();
        assert_eq!(conn_of(&out2, 0), cells);
    }

    #[test]
    fn invert_flips_every_cell() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n[0], n[1], n[2]]).unwrap(); // CCW
        let mesh = Mesh::from_submesh(sm);
        assert!(signed_area(&coords, &conn_of(&mesh, 0)[0]) > 0.0);

        let flipped = invert(&mesh).unwrap();
        assert!(signed_area(&coords, &conn_of(&flipped, 0)[0]) < 0.0);
        // Double invert restores the original connectivity.
        let back = invert(&flipped).unwrap();
        assert_eq!(conn_of(&back, 0), conn_of(&mesh, 0));
    }

    #[test]
    fn orient_chains_segments_head_to_tail() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[n[0], n[1]]).unwrap(); // seed 0→1
        sm.add_cell(&[n[2], n[1]]).unwrap(); // fed reversed: should become 1→2
        sm.add_cell(&[n[2], n[0]]).unwrap(); // 2→0
        let mesh = Mesh::from_submesh(sm);

        let out = conn_of(&orient(&mesh).unwrap(), 0);
        assert_eq!(out[0], vec![n[0].0, n[1].0]);
        assert_eq!(out[1], vec![n[1].0, n[2].0]);
        assert_eq!(out[2], vec![n[2].0, n[0].0]);
    }

    #[test]
    fn orient_harmonises_adjacent_tets() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<NodeId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.3, 1.0],  // apex above the base
            [0.3, 0.3, -1.0], // apex below the base
        ]
        .iter()
        .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
        .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TET4);
        sm.add_cell(&[n[0], n[1], n[2], n[3]]).unwrap(); // seed, positive volume
        sm.add_cell(&[n[0], n[1], n[2], n[4]]).unwrap(); // inconsistent (negative)
        let mesh = Mesh::from_submesh(sm);
        assert!(signed_volume(&coords, &conn_of(&mesh, 0)[1]) < 0.0);

        let out = conn_of(&orient(&mesh).unwrap(), 0);
        // Consistent adjacency of this bipyramid ⇒ both tets positive volume.
        assert!(signed_volume(&coords, &out[0]) > 0.0);
        assert!(signed_volume(&coords, &out[1]) > 0.0);
    }

    #[test]
    fn operators_share_coords_and_preserve_nodes() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n[0], n[1], n[2]]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Output reuses the same Coords and the same node ids (no fresh nodes).
        let out = orient(&mesh).unwrap();
        assert!(out.coords().unwrap().same_object(&coords));
        let mut ids = conn_of(&out, 0)[0].clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![n[0].0, n[1].0, n[2].0]);
    }
}
