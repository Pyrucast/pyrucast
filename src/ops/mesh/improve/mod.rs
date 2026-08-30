//! Improving a surface mesh that already exists: its geometry, its topology,
//! and the triangles left in it.
//!
//! The three operators here — [`regularize`](fn@crate::ops::mesh::regularize),
//! [`cleanup`](fn@crate::ops::mesh::cleanup) and
//! [`merge_triangles`](fn@crate::ops::mesh::merge_triangles) — do what the
//! pavers already do to their own work at the end of a run, but on any mesh and
//! one at a time. Composing them is the caller's business: the useful order is
//! usually triangles, then topology, then geometry, but there is no reason to
//! bake that in.
//!
//! ## Two margins, and only one of them is geometry
//!
//! A mesh can be poor in two unrelated ways, and it matters not to confuse
//! them. **Geometry** is where the nodes sit, and smoothing fixes it.
//! **Topology** is who is next to whom, and smoothing cannot touch it — an
//! interior node with three quadrangles around it has corners averaging 120°,
//! and no amount of moving will make them square, because the angles around a
//! node sum to 2π whatever the positions. That is why `cleanup` exists beside
//! `regularize` rather than inside it.
//!
//! The line is not a wall. `cleanup`'s one move that *removes* a node leaves
//! the ring round it stretched, and cannot be judged without relaxing that
//! ring first; it does so, and keeps the relaxation when it keeps the move.
//! What it does not do is smooth the rest of the mesh — that is still
//! `regularize`'s job, and running `cleanup` first is what unlocks it.
//!
//! ## Which way the cells turn
//!
//! Every quality measure here is **signed**: a quadrangle read the wrong way
//! round scores negative, which is how an inverted cell is told from a merely
//! poor one. That signal is worth keeping — it is what stops smoothing from
//! turning a cell inside out — so the winding is normalised instead of the
//! measure being made blind to it. Each cell is stored counter-clockwise
//! whatever the caller sent, and the result is wound back the way the mesh came
//! in.
//!
//! It matters because a clockwise mesh is ordinary, not exotic: a paver hands
//! back the winding of the contour it was given, so any domain meshed from an
//! inverted outer loop comes out clockwise. Read as it stands, such a mesh
//! scores negative everywhere and **every** pass here refuses to touch it —
//! silently, since nothing about it is invalid.
//!
//! ## The boundary is never touched
//!
//! Every node on a boundary edge — an edge carried by exactly one cell — is
//! pinned: it does not move, and nothing discards it. The mesh therefore keeps
//! exactly the boundary it came with, which is the same promise the pavers
//! make about the contour they were given.
//!
//! This is the guarantee that holds across all three operators, and the one to
//! reason with. "Nothing moves at all" is *not* one of them: `cleanup` relaxes
//! the ring of a node it gives up, because that move cannot be judged
//! otherwise. What is pinned stays pinned either way.
//!
//! Boundary edges are counted here rather than obtained from
//! [`border`](fn@crate::ops::mesh::border), which chains them into closed loops
//! and fails when it cannot. Pinning needs no chaining, and a mesh with a crack
//! in it — precisely the sort one reaches for these operators to improve —
//! should not be refused for a reason that has nothing to do with the job.

pub mod cleanup;
pub mod merge_triangles;
pub mod regularize;

pub use cleanup::cleanup;
pub use merge_triangles::merge_triangles;
pub use regularize::regularize;

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, Node, NodeId, Point2};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::collections::HashMap;

/// A surface mesh flattened into the arrays the improvement passes work on.
///
/// The same shape the pavers hand to [`paving::smooth`](super::paving::smooth)
/// and [`paving::cleanup`](super::paving::cleanup), so those two are reused as
/// they are rather than reimplemented against a different layout.
pub(crate) struct Surface {
    pub pts: Vec<Point2>,
    pub quads: Vec<[u32; 4]>,
    pub tris: Vec<[u32; 3]>,
    /// `false` for a node on the boundary: it must neither move nor be given
    /// up, or the mesh would come back with a different boundary from the one
    /// it went in with.
    pub movable: Vec<bool>,
    /// Index → the caller's node, in the order the nodes were first met.
    ids: Vec<NodeId>,
    coords: Handle<Coords>,
    /// The positions as they were read, so the passes that move nodes can be
    /// told from the ones that only say they might.
    pts0: Vec<Point2>,
    /// `true` when the mesh came in wound clockwise. Every cell is stored
    /// counter-clockwise whatever the caller sent, and this is what puts the
    /// winding back on the way out.
    clockwise: bool,
}

impl Surface {
    /// Read `mesh`'s `QUA4` and `TRI3` cells. `op` names the calling operator
    /// and only ever appears in error messages.
    pub(crate) fn read(mesh: &Mesh, op: &str) -> Result<Surface> {
        let coords = mesh.coords()?;
        let dim = coords.read().dim();
        if dim != 2 {
            return Err(PyrucastError::Message(format!(
                "{op}: only 2-D surface meshes are supported, got dim={dim}"
            )));
        }

        let mut index: HashMap<NodeId, u32> = HashMap::new();
        let mut surf = Surface {
            pts: Vec::new(),
            quads: Vec::new(),
            tris: Vec::new(),
            movable: Vec::new(),
            ids: Vec::new(),
            coords: coords.clone(),
            pts0: Vec::new(),
            clockwise: false,
        };
        // Which way the caller's cells turn. Counted rather than assumed: a
        // mesh may hold both, and the majority is what the result is wound
        // back to.
        let (mut cw, mut ccw) = (0usize, 0usize);
        let guard = coords.read();
        let mut cells = 0usize;
        for sm in mesh {
            let (et, conn) = {
                let s = sm.read();
                (s.element_type(), s.connectivity().to_vec())
            };
            match et {
                ElementType::POI1 | ElementType::SEG2 => continue,
                ElementType::TRI3 | ElementType::QUA4 => {}
                other => {
                    return Err(PyrucastError::Message(format!(
                        "{op}: only TRI3 and QUA4 cells can be improved, got {other}"
                    )));
                }
            }
            let npc = et.nodes_per_cell();
            for cell in conn.chunks(npc) {
                let mut local = [0u32; 4];
                for (k, &id) in cell.iter().enumerate() {
                    local[k] = match index.get(&id) {
                        Some(&i) => i,
                        None => {
                            let p = guard.position(id)?;
                            let i = surf.pts.len() as u32;
                            surf.pts.push(Point2::new(p[0], p[1]));
                            surf.movable.push(true);
                            surf.ids.push(id);
                            index.insert(id, i);
                            i
                        }
                    };
                }
                // Stored counter-clockwise, always: every quality measure
                // below is signed, and a cell read the wrong way round scores
                // negative — which reads as *inverted* and freezes the passes
                // that are meant to improve it.
                let ring: Vec<Point2> =
                    local[..npc].iter().map(|&i| surf.pts[i as usize]).collect();
                if crate::ops::mesh::triangulation::signed_area(&ring) < 0.0 {
                    local[..npc].reverse();
                    cw += 1;
                } else {
                    ccw += 1;
                }
                if npc == 3 {
                    surf.tris.push([local[0], local[1], local[2]]);
                } else {
                    surf.quads.push(local);
                }
                cells += 1;
            }
        }
        drop(guard);
        surf.pts0 = surf.pts.clone();
        surf.clockwise = cw > ccw;
        if cells == 0 {
            return Err(PyrucastError::Message(format!(
                "{op}: mesh has no surface cells (TRI3/QUA4)"
            )));
        }
        surf.pin_boundary();
        Ok(surf)
    }

    /// Pin every node carried by an edge that only one cell uses.
    fn pin_boundary(&mut self) {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        let bump = |a: u32, b: u32, c: &mut HashMap<(u32, u32), u32>| {
            *c.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        };
        for q in &self.quads {
            for i in 0..4 {
                bump(q[i], q[(i + 1) % 4], &mut counts);
            }
        }
        for t in &self.tris {
            for i in 0..3 {
                bump(t[i], t[(i + 1) % 3], &mut counts);
            }
        }
        for ((a, b), n) in counts {
            if n == 1 {
                self.movable[a as usize] = false;
                self.movable[b as usize] = false;
            }
        }
    }

    /// How many nodes sit on the boundary — the ones nothing may move.
    pub(crate) fn pinned(&self) -> usize {
        self.movable.iter().filter(|m| !**m).count()
    }

    /// Write the positions back onto the caller's own nodes and hand back the
    /// mesh unchanged otherwise.
    ///
    /// The one operator that mutates rather than copies. Pinned nodes are
    /// never written, so a node shared with another mesh only ever moves if it
    /// was interior to this one.
    pub(crate) fn write_positions(&self, mesh: &Mesh) -> Result<Mesh> {
        for (i, &id) in self.ids.iter().enumerate() {
            if !self.movable[i] {
                continue;
            }
            let node = Node::acquire(self.coords.clone(), id)?;
            node.set_position(&[self.pts[i].x, self.pts[i].y])?;
        }
        // The same mesh back: an aggregate over the very same submesh slots,
        // whose nodes have moved.
        mesh.subset(0..mesh.len())
    }

    /// Build a fresh mesh: only the nodes that actually moved are duplicated,
    /// every other one being shared with the caller's mesh.
    ///
    /// Compared against the positions as read rather than against `movable`,
    /// because a movable node is one a pass *may* move and most of them never
    /// do — `cleanup` relaxes a handful of rings and leaves the rest of the
    /// mesh exactly where it found it.
    pub(crate) fn to_mesh(&self, op: &str) -> Result<Mesh> {
        // The nodes a pass actually moved, created in one locked pass.
        let mut ids = self.ids.clone();
        let mut moved: Vec<usize> = Vec::new();
        let mut flat: Vec<f64> = Vec::new();
        for i in 0..ids.len() {
            if !self.movable[i] || self.pts[i] == self.pts0[i] {
                continue;
            }
            flat.extend_from_slice(&[self.pts[i].x, self.pts[i].y]);
            moved.push(i);
        }
        let first = self.coords.write().add_nodes(&flat)?.start;
        for (rank, &i) in moved.iter().enumerate() {
            ids[i] = NodeId(first + rank as u32);
        }
        let mesh = self.emit(&ids, op)?;
        // The relaxed mesh owns them now; hand back the unit `add_nodes` gave.
        let owned: Vec<NodeId> = (first..first + moved.len() as u32).map(NodeId).collect();
        self.coords.write().decref_all(&owned)?;
        Ok(mesh)
    }

    /// Build a fresh mesh over the caller's own nodes — for the passes that
    /// change connectivity and never geometry.
    pub(crate) fn to_mesh_same_nodes(&self, op: &str) -> Result<Mesh> {
        let ids = self.ids.clone();
        self.emit(&ids, op)
    }

    fn emit(&self, ids: &[NodeId], op: &str) -> Result<Mesh> {
        let mut mesh = Mesh::empty();
        if !self.quads.is_empty() {
            let conn: Vec<NodeId> = self
                .quads
                .iter()
                .flat_map(|q| {
                    let mut c = q.map(|v| ids[v as usize]);
                    if self.clockwise {
                        c.reverse();
                    }
                    c
                })
                .collect();
            mesh.add_sub(Handle::new(SubMesh::from_connectivity(
                self.coords.clone(),
                ElementType::QUA4,
                conn,
            )?))?;
        }
        if !self.tris.is_empty() {
            let conn: Vec<NodeId> = self
                .tris
                .iter()
                .flat_map(|t| {
                    let mut c = t.map(|v| ids[v as usize]);
                    if self.clockwise {
                        c.reverse();
                    }
                    c
                })
                .collect();
            mesh.add_sub(Handle::new(SubMesh::from_connectivity(
                self.coords.clone(),
                ElementType::TRI3,
                conn,
            )?))?;
        }
        if mesh.is_empty() {
            return Err(PyrucastError::Message(format!("{op}: produced no cell")));
        }
        Ok(mesh)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::cleanup::cleanup;
    use super::merge_triangles::merge_triangles;
    use super::regularize::regularize;
    use super::*;
    use crate::handle::Handle;
    use crate::ops::mesh::grid_surface;

    /// The circle `grid_surface` finds hardest: no axis-aligned edge, so the
    /// whole boundary falls to the frontal band, which is where every poor cell
    /// and every triangle ends up.
    fn paved_circle() -> Mesh {
        let coords = Handle::new(Coords::new(2).unwrap());
        let pts: Vec<(f64, f64)> = (0..60)
            .map(|i| {
                let t = i as f64 / 60.0 * std::f64::consts::TAU;
                (t.cos(), t.sin())
            })
            .collect();
        let n = pts.len();
        let mut fine = Vec::new();
        for i in 0..n {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            let k = ((len / 0.05).round() as usize).max(1);
            for j in 0..k {
                let t = j as f64 / k as f64;
                fine.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
            }
        }
        let ids: Vec<NodeId> = fine
            .iter()
            .map(|&(x, y)| Node::create_in(coords.clone(), &[x, y]).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        let m = ids.len();
        for i in 0..m {
            sm.add_cell(&[ids[i], ids[(i + 1) % m]]).unwrap();
        }
        let contour = Mesh::from_submesh(sm);
        grid_surface(
            &contour,
            ElementType::QUA4,
            Some(0.05),
            0,
            false,
            crate::ops::mesh::FrontRelax::Free,
        )
        .unwrap()
    }

    /// Cell count, triangle count, worst normalised Jacobian, the 1st
    /// percentile of it, and the boundary edges.
    fn look(mesh: &Mesh) -> (usize, usize, f64, f64, Vec<(NodeId, NodeId)>) {
        let (mut cells, mut tris) = (0usize, 0usize);
        let mut qs: Vec<f64> = Vec::new();
        let mut used: HashMap<(NodeId, NodeId), usize> = HashMap::new();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                let nodes = cell.nodes().unwrap();
                let p: Vec<Point2> = nodes
                    .iter()
                    .map(|n| {
                        let v = n.position().unwrap();
                        Point2::new(v[0], v[1])
                    })
                    .collect();
                let k = p.len();
                cells += 1;
                if k == 3 {
                    tris += 1;
                }
                for i in 0..k {
                    let (a, b) = (nodes[i].id(), nodes[(i + 1) % k].id());
                    let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                    *used.entry(key).or_insert(0) += 1;
                }
                let signed: f64 = 0.5
                    * (0..k)
                        .map(|i| p[i].x * p[(i + 1) % k].y - p[(i + 1) % k].x * p[i].y)
                        .sum::<f64>();
                let p: Vec<Point2> = if signed < 0.0 {
                    p.iter().rev().copied().collect()
                } else {
                    p
                };
                let mut w = 1.0f64;
                for i in 0..k {
                    let u = p[(i + 1) % k] - p[i];
                    let v = p[(i + k - 1) % k] - p[i];
                    w = w.min((u.x * v.y - u.y * v.x) / (u.norm() * v.norm()));
                }
                qs.push(w);
            }
        }
        qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut boundary: Vec<(NodeId, NodeId)> = used
            .iter()
            .filter(|(_, v)| **v == 1)
            .map(|(k, _)| *k)
            .collect();
        boundary.sort_unstable();
        (cells, tris, qs[0], qs[qs.len() / 100], boundary)
    }

    #[test]
    fn regularize_improves_the_mesh_and_keeps_its_boundary() {
        let mesh = paved_circle();
        let (cells, tris, worst, p1, boundary) = look(&mesh);
        let out = regularize(&mesh, 40, true, false).unwrap();
        let (cells2, tris2, worst2, p12, boundary2) = look(&out);

        assert_eq!(
            (cells, tris),
            (cells2, tris2),
            "nothing may be added or lost"
        );
        assert!(worst2 > worst, "worst cell: {worst} → {worst2}");
        assert!(p12 > p1, "1st percentile: {p1} → {p12}");
        // The boundary is untouched, edge for edge and node for node: the
        // pinned nodes are the same ones and they were never duplicated.
        assert_eq!(boundary, boundary2);
    }

    #[test]
    fn regularize_in_place_moves_the_caller_s_own_nodes() {
        let mesh = paved_circle();
        let (_, _, worst, _, boundary) = look(&mesh);
        let same = regularize(&mesh, 40, true, true).unwrap();
        let (_, _, worst2, _, boundary2) = look(&same);
        assert!(worst2 > worst, "{worst} → {worst2}");
        assert_eq!(boundary, boundary2);
        // The input mesh sees the move too — that is what in place means.
        let (_, _, worst3, _, _) = look(&mesh);
        assert_eq!(worst2, worst3);
    }

    #[test]
    fn the_angular_rule_and_the_laplacian_both_hold_the_line() {
        // Neither dominates — measured, the angular is better through the bulk
        // and the Laplacian in the tail — but both must improve the worst cell
        // and neither may ever invert one.
        let mesh = paved_circle();
        let (_, _, worst, _, _) = look(&mesh);
        for angular in [true, false] {
            let out = regularize(&mesh, 40, angular, false).unwrap();
            let (_, _, w, _, _) = look(&out);
            assert!(w > worst, "angular={angular}: {worst} → {w}");
            assert!(w > 0.0, "angular={angular} inverted a cell");
        }
    }

    #[test]
    fn cleanup_keeps_the_mesh_conforming_and_the_boundary_whole() {
        // On a freshly paved mesh there is nothing left to do — the paver runs
        // the same pass at the end of its own run — so this checks the lift is
        // faithful, not that it finds work.
        let mesh = paved_circle();
        let (cells, _, _, _, boundary) = look(&mesh);
        let out = cleanup(&mesh).unwrap();
        let (cells2, _, worst2, _, boundary2) = look(&out);
        assert!(cells2 <= cells, "cleanup may only remove cells");
        assert!(worst2 > 0.0, "cleanup inverted a cell");
        assert_eq!(boundary, boundary2);
    }

    #[test]
    fn merge_triangles_removes_them_two_at_a_time_and_never_inverts() {
        let mesh = paved_circle();
        let (_, tris, _, _, boundary) = look(&mesh);
        assert!(tris >= 2, "the fixture must have triangles to remove");
        let out = merge_triangles(&mesh).unwrap();
        let (_, tris2, worst2, _, boundary2) = look(&out);

        assert!(tris2 < tris, "triangles: {tris} → {tris2}");
        // Two at a time, always: `4Q + 3T = 2·E_interior + E_boundary` fixes
        // T's parity to the boundary's, which nothing here may change.
        assert_eq!(
            (tris - tris2) % 2,
            0,
            "{tris} → {tris2} is not a whole number of pairs"
        );
        assert!(worst2 > 0.0, "a merge was forced through an invalid cell");
        assert_eq!(boundary, boundary2);
    }

    /// The same mesh with every cell read backwards.
    fn wound_the_other_way(mesh: &Mesh) -> Mesh {
        let coords = mesh.coords().unwrap();
        let mut out = Mesh::empty();
        for si in 0..mesh.len() {
            let et = mesh.get(si).unwrap().read().element_type();
            let mut sub = SubMesh::new(coords.clone(), et);
            for cell in mesh.cells(si).unwrap() {
                let mut ids: Vec<NodeId> = cell.nodes().unwrap().iter().map(|n| n.id()).collect();
                ids.reverse();
                sub.add_cell(&ids).unwrap();
            }
            out.add_sub(Handle::new(sub)).unwrap();
        }
        out
    }

    #[test]
    fn cleanup_leaves_the_caller_s_own_nodes_alone_outside_the_rings_it_relaxes() {
        // A collapse removes a node, so the ring it leaves is stretched until
        // something relaxes it — which is why the move is judged *after* a
        // trial relaxation, and why that relaxation is kept when the move is.
        // `cleanup` therefore does move nodes, but only round the rings it
        // collapsed: everywhere else the caller's own nodes come back
        // untouched, not copies of them.
        let mesh = paved_circle();
        let mut before: Vec<(NodeId, [f64; 2])> = Vec::new();
        for si in 0..mesh.len() {
            for cell in mesh.cells(si).unwrap() {
                for n in cell.nodes().unwrap() {
                    let p = n.position().unwrap();
                    before.push((n.id(), [p[0], p[1]]));
                }
            }
        }
        before.sort_by_key(|(id, _)| *id);
        before.dedup_by_key(|(id, _)| *id);

        let out = cleanup(&mesh).unwrap();
        assert!(
            look(&out).0 < look(&mesh).0,
            "the fixture must give it work"
        );

        let (mut shared, mut fresh) = (0usize, 0usize);
        for si in 0..out.len() {
            for cell in out.cells(si).unwrap() {
                for n in cell.nodes().unwrap() {
                    match before.binary_search_by_key(&n.id(), |(id, _)| *id) {
                        // A node it kept: the caller's own, at the very place
                        // it was — bit for bit, not to within a tolerance.
                        Ok(k) => {
                            let p = n.position().unwrap();
                            assert_eq!([p[0], p[1]], before[k].1, "a shared node moved");
                            shared += 1;
                        }
                        Err(_) => fresh += 1,
                    }
                }
            }
        }
        assert!(fresh > 0, "the rings it collapsed must have been relaxed");
        assert!(
            fresh * 20 < shared,
            "only the collapsed rings may move: {fresh} fresh for {shared} shared"
        );

        // And the input mesh is untouched: relaxing is not done in place.
        for (id, was) in &before {
            let n = Node::acquire(mesh.coords().unwrap(), *id).unwrap();
            let p = n.position().unwrap();
            assert_eq!([p[0], p[1]], *was);
        }
    }

    #[test]
    fn a_clockwise_mesh_is_improved_just_as_much_as_a_counter_clockwise_one() {
        // A paver hands back the winding of the contour it was given, so a
        // domain meshed from an inverted outer loop comes out clockwise —
        // ordinary, and valid. Read as it stands, every signed quality measure
        // here scores it negative, which reads as *inverted*: the smoothing
        // guard `after > 0.0` rejects every move and the passes freeze on a
        // mesh with nothing wrong with it. Hence the winding is normalised on
        // the way in and put back on the way out.
        let mesh = paved_circle();
        let flipped = wound_the_other_way(&mesh);
        let (cells, tris, worst, p1, _) = look(&mesh);
        let (cells_f, tris_f, worst_f, p1_f, _) = look(&flipped);
        assert_eq!((cells, tris), (cells_f, tris_f), "the fixture must match");

        let round = |m: &Mesh| {
            let m = merge_triangles(m).unwrap();
            let m = cleanup(&m).unwrap();
            regularize(&m, 30, true, false).unwrap()
        };
        let (c1, t1, w1, q1, _) = look(&round(&mesh));
        let (c2, t2, w2, q2, _) = look(&round(&flipped));

        assert!(
            w1 > worst && w2 > worst_f,
            "both must improve the worst cell"
        );
        assert!(q1 > p1 && q2 > p1_f, "both must improve the 1st percentile");
        assert_eq!((c1, t1), (c2, t2), "and reach the very same mesh");
        assert!((w1 - w2).abs() < 1e-12, "{w1} vs {w2}");

        // The result keeps the winding it came in with: the caller gets back
        // the mesh they gave, not a silently re-oriented one.
        let out = cleanup(&flipped).unwrap();
        let (n_out, _, _, _, _) = look(&out);
        let mut cw = 0usize;
        for si in 0..out.len() {
            for cell in out.cells(si).unwrap() {
                let p: Vec<Point2> = cell
                    .nodes()
                    .unwrap()
                    .iter()
                    .map(|n| {
                        let v = n.position().unwrap();
                        Point2::new(v[0], v[1])
                    })
                    .collect();
                if crate::ops::mesh::triangulation::signed_area(&p) < 0.0 {
                    cw += 1;
                }
            }
        }
        assert_eq!(cw, n_out, "every cell must come back clockwise");
    }

    #[test]
    fn the_three_compose_and_the_composition_converges() {
        // The result that justifies the three operators: repeating the round
        // gets further than one pass, every intermediate mesh stays valid, and
        // it settles rather than drifting.
        let mut mesh = paved_circle();
        let (_, tris0, worst0, p10, boundary) = look(&mesh);
        let mut seen = Vec::new();
        for _ in 0..4 {
            mesh = merge_triangles(&mesh).unwrap();
            mesh = cleanup(&mesh).unwrap();
            mesh = regularize(&mesh, 30, true, false).unwrap();
            mesh = regularize(&mesh, 15, false, false).unwrap();
            let (_, t, w, _, b) = look(&mesh);
            assert!(w > 0.0, "a round left an invalid cell");
            assert_eq!(boundary, b, "a round changed the boundary");
            seen.push(t);
        }
        let (_, tris1, worst1, p11, _) = look(&mesh);
        assert!(tris1 < tris0 / 2, "triangles: {tris0} → {tris1}");
        assert!(worst1 > worst0, "worst cell: {worst0} → {worst1}");
        assert!(p11 > p10, "1st percentile: {p10} → {p11}");
        assert_eq!(seen[2], seen[3], "the composition must settle: {seen:?}");
    }
}
