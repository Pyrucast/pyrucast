//! Reading and validating the closed surface handed to the tetrahedral
//! mesher.
//!
//! Everything downstream — the Delaunay kernel, boundary recovery, the
//! inside/outside classification — assumes it is working on a **watertight,
//! consistently oriented, non-degenerate** triangulated surface. Rather than
//! let a violated assumption surface later as a mysterious failure deep in
//! the kernel, every assumption is checked once, here, against the raw
//! input, and reported with the `NodeId`s the caller can actually act on.
//!
//! The contract this module enforces:
//!
//! - every submesh is `TRI3`, in a 3-D `Coords`;
//! - no two distinct nodes sit at the same place, and no facet is flat;
//! - no facet is repeated;
//! - each **directed** edge `(a, b)` is used exactly once across all facets,
//!   and its reverse `(b, a)` exists. That single condition is closedness,
//!   manifoldness and orientation consistency at once: an unmatched
//!   direction means a hole, a repeated one means two facets disagree on
//!   which way is out;
//! - the enclosed signed volume is positive, i.e. the normals point **out
//!   of the material**. Internal cavities follow from that same convention
//!   with no extra input: a cavity is a closed surface whose normals point
//!   into the hole, and it subtracts from the total.

use std::collections::HashMap;

use crate::aggregate::Aggregate;
use crate::containers::mesh::{ElementType, Mesh, NodeId};
use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;
use crate::store::read;

use super::intersect::first_self_intersection;
use super::predicates::collinear3d;
use super::recovery::Stuck;

/// Two nodes closer than this fraction of the bounding-box diagonal are
/// treated as the same point and rejected as a duplicate.
///
/// This is a validation threshold, not a geometric decision: it only decides
/// which inputs to *refuse*. Every predicate the mesher actually builds
/// topology from is exact.
const COINCIDENT_NODE_TOL: f64 = 1e-12;

/// A validated closed surface, ready to be tetrahedralized.
///
/// Node identity is kept twice over: `points[i]` is the position of the node
/// whose store-level identity is `node_ids[i]`. The mesher works on the
/// compact local indices and only goes back to `NodeId`s when materializing
/// the result, which is what lets the envelope's nodes be reused verbatim.
#[derive(Debug, Clone)]
pub struct Envelope {
    points: Vec<[f64; 3]>,
    node_ids: Vec<NodeId>,
    facets: Vec<[u32; 3]>,
    volume: f64,
    /// What each subdivision point was put there to cut, in the order they
    /// were added. Kept so the operator can try to undo them afterwards.
    added: Vec<Origin>,
}

/// The facets a subdivision point's neighbourhood would become without it:
/// the halves that go away, and the whole pieces that replace them.
pub type Merge = (Vec<[u32; 3]>, Vec<[u32; 3]>);

/// The piece of the envelope a subdivision point was placed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The middle of this edge.
    Edge(u32, u32),
    /// The centre of this facet.
    Facet([u32; 3]),
}

impl Envelope {
    /// Read `mesh` as a closed oriented `TRI3` surface and check it.
    ///
    /// Errors carry the operator's name and, where a specific node or facet
    /// is at fault, its `NodeId`s.
    pub fn extract(mesh: &Mesh, cancel: &dyn Cancel) -> Result<Envelope> {
        if mesh.is_empty() {
            return Err(PyrucastError::Message(
                "mesh_volume: the envelope is empty".into(),
            ));
        }
        let coords_handle = mesh.coords()?;
        let dim = read(&coords_handle)?.dim();
        if dim != 3 {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: the envelope must be 3-D, got dim={dim}"
            )));
        }

        // One pass over the connectivity, then a single read guard on the
        // coordinates: the same shape as `skin` and `border`.
        let mut node_ids: Vec<NodeId> = Vec::new();
        let mut local: HashMap<NodeId, u32> = HashMap::new();
        let mut facets: Vec<[u32; 3]> = Vec::new();
        for sm_handle in mesh {
            let sm = read(sm_handle)?;
            let et = sm.element_type();
            if et != ElementType::TRI3 {
                return Err(PyrucastError::Message(format!(
                    "mesh_volume: the envelope must be made of TRI3 facets, got {et}. \
                     A quadrangle has no single plane to respect, so it is refused rather \
                     than silently split; mesh the surface in TRI3 first."
                )));
            }
            for cell in sm.connectivity().chunks(3) {
                let mut f = [0u32; 3];
                for (k, &nid) in cell.iter().enumerate() {
                    f[k] = *local.entry(nid).or_insert_with(|| {
                        node_ids.push(nid);
                        (node_ids.len() - 1) as u32
                    });
                }
                facets.push(f);
            }
        }

        let points: Vec<[f64; 3]> = {
            let c = read(&coords_handle)?;
            node_ids
                .iter()
                .map(|&nid| {
                    let p = c.coord(nid)?;
                    Ok([p[0], p[1], p[2]])
                })
                .collect::<Result<_>>()?
        };

        let envelope = Envelope {
            points,
            node_ids,
            facets,
            volume: 0.0,
            added: Vec::new(),
        };
        envelope.validate(cancel)
    }

    /// Run every structural check, returning the envelope with its signed
    /// volume filled in.
    ///
    /// The checks run cheapest first, and each one assumes the previous ones
    /// passed — which is also what makes the diagnostics specific.
    fn validate(mut self, cancel: &dyn Cancel) -> Result<Envelope> {
        if self.points.len() < 4 || self.facets.len() < 4 {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: a closed surface needs at least 4 nodes and 4 facets, got {} and {}",
                self.points.len(),
                self.facets.len()
            )));
        }
        cancel.check()?;
        self.check_no_coincident_nodes()?;
        cancel.check()?;
        self.check_facets_are_sound()?;
        cancel.check()?;
        self.check_edges_pair_up()?;
        cancel.check()?;
        self.check_no_self_intersection()?;
        cancel.check()?;
        self.volume = self.signed_volume();
        if self.volume <= 0.0 {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: the envelope encloses a signed volume of {:.6e}, so its normals \
                 point into the material instead of out of it — use invert() on the surface. \
                 An internal cavity, on the other hand, is meant to have inward normals: only \
                 the total must be positive.",
                self.volume
            )));
        }
        Ok(self)
    }

    /// Reject distinct nodes sitting at (nearly) the same place.
    ///
    /// Checked before the edge pairing, because a duplicated node tears the
    /// surface apart topologically and would otherwise be reported as a hole
    /// — a true but thoroughly unhelpful diagnosis.
    fn check_no_coincident_nodes(&self) -> Result<()> {
        let tol = COINCIDENT_NODE_TOL * self.bbox_diagonal();
        // A cell size of `tol` means coincident points land in the same cell
        // or in one of its 26 neighbours.
        let cell = if tol > 0.0 { tol } else { 1.0 };
        let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let tol2 = tol * tol;

        for (i, p) in self.points.iter().enumerate() {
            let base = [
                (p[0] / cell).floor() as i64,
                (p[1] / cell).floor() as i64,
                (p[2] / cell).floor() as i64,
            ];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = [base[0] + dx, base[1] + dy, base[2] + dz];
                        for &j in grid.get(&key).map_or(&[][..], |v| v.as_slice()) {
                            let q = &self.points[j as usize];
                            let d2: f64 = (0..3).map(|k| (p[k] - q[k]).powi(2)).sum();
                            if d2 <= tol2 {
                                return Err(PyrucastError::Message(format!(
                                    "mesh_volume: nodes {} and {} of the envelope are at the \
                                     same place {:?} — weld them with merge_nodes() first",
                                    self.node_ids[j as usize].0, self.node_ids[i].0, p
                                )));
                            }
                        }
                    }
                }
            }
            grid.entry(base).or_default().push(i as u32);
        }
        Ok(())
    }

    /// Reject facets that repeat a node, that are flat, or that are
    /// duplicated.
    fn check_facets_are_sound(&self) -> Result<()> {
        let mut seen: HashMap<[u32; 3], usize> = HashMap::new();
        for (fi, f) in self.facets.iter().enumerate() {
            if let Some(repeated) = (0..3).find(|&k| f[k] == f[(k + 1) % 3]) {
                return Err(PyrucastError::Message(format!(
                    "mesh_volume: facet {fi} of the envelope uses node {} twice",
                    self.node_ids[f[repeated] as usize].0
                )));
            }
            let (a, b, c) = (
                &self.points[f[0] as usize],
                &self.points[f[1] as usize],
                &self.points[f[2] as usize],
            );
            if collinear3d(a, b, c) {
                return Err(PyrucastError::Message(format!(
                    "mesh_volume: facet {fi} of the envelope is flat — its nodes {}, {} and {} \
                     are collinear, so it has no normal",
                    self.node_ids[f[0] as usize].0,
                    self.node_ids[f[1] as usize].0,
                    self.node_ids[f[2] as usize].0
                )));
            }
            let mut key = *f;
            key.sort_unstable();
            if let Some(prev) = seen.insert(key, fi) {
                return Err(PyrucastError::Message(format!(
                    "mesh_volume: facets {prev} and {fi} of the envelope share the same three \
                     nodes {}, {}, {}",
                    self.node_ids[key[0] as usize].0,
                    self.node_ids[key[1] as usize].0,
                    self.node_ids[key[2] as usize].0
                )));
            }
        }
        Ok(())
    }

    /// The single test that establishes closedness, manifoldness and a
    /// consistent orientation: every directed edge appears exactly once, and
    /// its reverse exists.
    fn check_edges_pair_up(&self) -> Result<()> {
        let mut owner: HashMap<(u32, u32), usize> = HashMap::with_capacity(3 * self.facets.len());
        for (fi, f) in self.facets.iter().enumerate() {
            for k in 0..3 {
                let edge = (f[k], f[(k + 1) % 3]);
                if let Some(prev) = owner.insert(edge, fi) {
                    return Err(PyrucastError::Message(format!(
                        "mesh_volume: facets {prev} and {fi} of the envelope both walk the edge \
                         ({}, {}) in the same direction, so they disagree on which side is out \
                         — orient the surface consistently (see orient()) or remove the overlap",
                        self.node_ids[edge.0 as usize].0, self.node_ids[edge.1 as usize].0
                    )));
                }
            }
        }
        for (&(u, v), &fi) in &owner {
            if !owner.contains_key(&(v, u)) {
                return Err(PyrucastError::Message(format!(
                    "mesh_volume: edge ({}, {}) of facet {fi} has no facet on its other side, \
                     so the envelope is not closed",
                    self.node_ids[u as usize].0, self.node_ids[v as usize].0
                )));
            }
        }
        Ok(())
    }

    /// Reject a surface that passes through itself.
    ///
    /// Run last: it is the only check whose cost is more than linear, and it
    /// is also the only one that assumes the facets are already known to be
    /// non-degenerate.
    fn check_no_self_intersection(&self) -> Result<()> {
        match first_self_intersection(&self.points, &self.facets) {
            None => Ok(()),
            Some((i, j)) => Err(PyrucastError::Message(format!(
                "mesh_volume: facets {i} (nodes {}, {}, {}) and {j} (nodes {}, {}, {}) of the \
                 envelope pass through each other, so the surface has no well-defined inside",
                self.node_ids[self.facets[i][0] as usize].0,
                self.node_ids[self.facets[i][1] as usize].0,
                self.node_ids[self.facets[i][2] as usize].0,
                self.node_ids[self.facets[j][0] as usize].0,
                self.node_ids[self.facets[j][1] as usize].0,
                self.node_ids[self.facets[j][2] as usize].0,
            ))),
        }
    }

    /// Volume enclosed by the oriented surface, by the divergence theorem:
    /// the sum of the signed volumes of the tetrahedra joining each facet to
    /// a common apex.
    ///
    /// The apex is the bounding-box centre rather than the origin, so a body
    /// sitting far from the origin does not lose precision to cancellation.
    fn signed_volume(&self) -> f64 {
        let o = self.bbox_centre();
        let sum: f64 = self
            .facets
            .iter()
            .map(|f| {
                let p = |i: u32| {
                    let q = &self.points[i as usize];
                    [q[0] - o[0], q[1] - o[1], q[2] - o[2]]
                };
                let (a, b, c) = (p(f[0]), p(f[1]), p(f[2]));
                a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0])
            })
            .sum();
        sum / 6.0
    }

    fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    }

    fn bbox_centre(&self) -> [f64; 3] {
        let (lo, hi) = self.bbox();
        [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ]
    }

    /// Diagonal of the bounding box — the natural length scale of the body,
    /// used to turn relative tolerances into absolute ones.
    pub fn bbox_diagonal(&self) -> f64 {
        let (lo, hi) = self.bbox();
        (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt()
    }

    /// Node positions, indexed by local index.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Store-level identity of each local index.
    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }

    /// Outward-oriented triangles, as local indices.
    pub fn facets(&self) -> &[[u32; 3]] {
        &self.facets
    }

    /// Volume of material enclosed by the surface — the reference the final
    /// mesh is checked against.
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// How many of the points came from the caller.
    ///
    /// Anything beyond that was put there by [`Self::subdivide`] and has no
    /// `NodeId` yet; the operator creates those when it writes the mesh out.
    pub fn given_node_count(&self) -> usize {
        self.node_ids.len()
    }

    /// The subdivision points, most recent first, with what each was cutting.
    ///
    /// Reverse order because a later point may sit on a facet an earlier one
    /// created: undoing them the other way round would leave the envelope
    /// referring to facets that no longer exist.
    pub fn added_points(&self) -> Vec<(u32, Origin)> {
        (0..self.added.len())
            .rev()
            .map(|i| ((self.node_ids.len() + i) as u32, self.added[i]))
            .collect()
    }

    /// The facets that would take the place of those around `m` if it went
    /// away, or `None` when its neighbourhood has moved on since it was put
    /// there.
    ///
    /// A point in the middle of an edge is surrounded by pairs of facets that
    /// were one facet before the cut; merging each pair back is exactly the
    /// inverse of the cut.
    pub fn merge_around(&self, m: u32, origin: Origin) -> Option<Merge> {
        let Origin::Edge(u, v) = origin else {
            return None; // a facet's centre has three neighbours, not two
        };
        let touching: Vec<[u32; 3]> = self
            .facets
            .iter()
            .filter(|f| f.contains(&m))
            .copied()
            .collect();
        if touching.len() != 4 {
            return None; // cut again since, or already partly undone
        }

        // Pair up the halves: one holds `u`, its mate holds `v`, and they
        // share the apex the original facet was pointing at.
        let mut merged: Vec<[u32; 3]> = Vec::with_capacity(2);
        for a in &touching {
            let &apex = a.iter().find(|&&x| x != m && x != u && x != v)?;
            if !a.contains(&u) {
                continue;
            }
            // Rebuild the original winding by putting `v` where `m` was.
            let whole: [u32; 3] = a.map(|x| if x == m { v } else { x });
            if !touching
                .iter()
                .any(|b| b.contains(&v) && b.contains(&apex) && b != a)
            {
                return None;
            }
            merged.push(whole);
        }
        (merged.len() == 2).then_some((touching, merged))
    }

    /// Undo a subdivision: drop `m` and put the merged facets in place.
    ///
    /// The point keeps its slot in `points` — indices are handed out to the
    /// mesh and must not shift — it simply stops being used.
    pub fn unsplit(&mut self, dropped: &[[u32; 3]], merged: &[[u32; 3]]) {
        self.facets
            .retain(|f| !dropped.iter().any(|d| sorted(d) == sorted(f)));
        self.facets.extend_from_slice(merged);
        self.volume = self.signed_volume();
    }

    /// Cut the envelope finer at the places named, keeping its shape.
    ///
    /// An **edge** is split at its middle, and both facets along it become
    /// two. A **facet** is split at its centre into three. In each case the
    /// new point lies on the piece it divides, so the surface is the same
    /// surface — only its triangulation is finer. That is the whole trade
    /// the caller is making by allowing it: the solid keeps its form, the
    /// skin of the result no longer matches the mesh that was handed in.
    ///
    /// Subdividing is what makes recovery possible at all when it is stuck:
    /// a segment surrounded closely enough by its own subdivisions is
    /// recovered by the Delaunay triangulation on its own.
    pub fn subdivide(&mut self, at: &[Stuck]) -> Result<()> {
        let mut edges: Vec<(u32, u32)> = Vec::new();
        let mut facets: Vec<[u32; 3]> = Vec::new();
        for s in at {
            match *s {
                Stuck::Edge(u, v) => edges.push(if u < v { (u, v) } else { (v, u) }),
                Stuck::Facet(f) => facets.push(f),
            }
        }
        edges.sort_unstable();
        edges.dedup();
        if edges.is_empty() && facets.is_empty() {
            return Ok(());
        }

        // Every edge gets its midpoint first, so a facet carrying more than
        // one split edge is cut for all of them in the same pass — cutting
        // only some of them would tear the surface open along the others.
        let mut middle: HashMap<(u32, u32), u32> = HashMap::with_capacity(edges.len());
        for &(u, v) in &edges {
            let (a, b) = (self.points[u as usize], self.points[v as usize]);
            self.points.push([
                (a[0] + b[0]) / 2.0,
                (a[1] + b[1]) / 2.0,
                (a[2] + b[2]) / 2.0,
            ]);
            middle.insert((u, v), (self.points.len() - 1) as u32);
            self.added.push(Origin::Edge(u, v));
        }
        let split_of = |u: u32, v: u32| -> Option<u32> {
            middle.get(&if u < v { (u, v) } else { (v, u) }).copied()
        };

        let mut next: Vec<[u32; 3]> = Vec::with_capacity(self.facets.len() + 3 * edges.len());
        for f in &self.facets {
            // Name the sides so the templates below read the same way round
            // as the facet: side `k` runs from `f[k]` to `f[k + 1]`.
            let cut: [Option<u32>; 3] = [0, 1, 2].map(|k: usize| split_of(f[k], f[(k + 1) % 3]));
            match (cut[0], cut[1], cut[2]) {
                (None, None, None) => next.push(*f),
                // One side cut: split the facet across to the opposite corner.
                (Some(m), None, None) => {
                    next.push([f[0], m, f[2]]);
                    next.push([m, f[1], f[2]]);
                }
                (None, Some(m), None) => {
                    next.push([f[1], m, f[0]]);
                    next.push([m, f[2], f[0]]);
                }
                (None, None, Some(m)) => {
                    next.push([f[2], m, f[1]]);
                    next.push([m, f[0], f[1]]);
                }
                // Two sides cut: a corner triangle plus the quadrilateral
                // left over, itself cut in two.
                (Some(m), Some(n), None) => {
                    next.push([m, f[1], n]);
                    next.push([f[0], m, n]);
                    next.push([f[0], n, f[2]]);
                }
                (None, Some(m), Some(n)) => {
                    next.push([m, f[2], n]);
                    next.push([f[1], m, n]);
                    next.push([f[1], n, f[0]]);
                }
                (Some(n), None, Some(m)) => {
                    next.push([m, f[0], n]);
                    next.push([f[2], m, n]);
                    next.push([f[2], n, f[1]]);
                }
                // All three: the plain four-way split.
                (Some(m), Some(n), Some(p)) => {
                    next.push([f[0], m, p]);
                    next.push([m, f[1], n]);
                    next.push([p, n, f[2]]);
                    next.push([m, n, p]);
                }
            }
        }
        self.facets = next;

        for f in &facets {
            let Some(pos) = self.facets.iter().position(|g| g == f) else {
                continue; // already cut by one of the edge splits
            };
            let (a, b, c) = (
                self.points[f[0] as usize],
                self.points[f[1] as usize],
                self.points[f[2] as usize],
            );
            self.points.push([
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ]);
            let m = (self.points.len() - 1) as u32;
            self.added.push(Origin::Facet(*f));
            self.facets.swap_remove(pos);
            self.facets.push([f[0], f[1], m]);
            self.facets.push([f[1], f[2], m]);
            self.facets.push([f[2], f[0], m]);
        }

        self.volume = self.signed_volume();
        Ok(())
    }
}

fn sorted(f: &[u32; 3]) -> [u32; 3] {
    let mut k = *f;
    k.sort_unstable();
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Coords, Node, SubMesh};
    use crate::interrupt::NoCancel;
    use crate::store::insert;
    use crate::store::Handle;

    /// The eight corners of an axis-aligned box, in the order
    /// `(x, y, z) = 000, 100, 110, 010, 001, 101, 111, 011`.
    fn box_nodes(coords: &Handle<Coords>, lo: [f64; 3], hi: [f64; 3]) -> Vec<NodeId> {
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect()
    }

    /// The twelve triangles of a box, normals pointing away from its inside.
    const BOX_FACETS: [[usize; 3]; 12] = [
        [0, 3, 2],
        [0, 2, 1], // z = lo
        [4, 5, 6],
        [4, 6, 7], // z = hi
        [0, 1, 5],
        [0, 5, 4], // y = lo
        [1, 2, 6],
        [1, 6, 5], // x = hi
        [2, 3, 7],
        [2, 7, 6], // y = hi
        [3, 0, 4],
        [3, 4, 7], // x = lo
    ];

    fn surface(coords: &Handle<Coords>, nodes: &[NodeId], facets: &[[usize; 3]]) -> Mesh {
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in facets {
            sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                .unwrap();
        }
        Mesh::from_submesh(sm)
    }

    fn unit_box(coords: &Handle<Coords>) -> Mesh {
        let nodes = box_nodes(coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        surface(coords, &nodes, &BOX_FACETS)
    }

    fn message(e: PyrucastError) -> String {
        e.to_string()
    }

    #[test]
    fn accepts_a_box_and_measures_its_volume() {
        let coords = insert(Coords::new(3).unwrap());
        let env = Envelope::extract(&unit_box(&coords), &NoCancel).unwrap();
        assert_eq!(env.points().len(), 8);
        assert_eq!(env.facets().len(), 12);
        assert!((env.volume() - 1.0).abs() < 1e-15, "{}", env.volume());
        assert!((env.bbox_diagonal() - 3f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn volume_is_measured_far_from_the_origin() {
        // The apex of the volume sum is the bounding-box centre, so a body
        // sitting a long way out keeps its precision.
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [1e6, 1e6, 1e6], [1e6 + 2.0, 1e6 + 3.0, 1e6 + 4.0]);
        let env = Envelope::extract(&surface(&coords, &nodes, &BOX_FACETS), &NoCancel).unwrap();
        assert!((env.volume() - 24.0).abs() < 1e-9, "{}", env.volume());
    }

    #[test]
    fn accepts_a_hollow_box_and_subtracts_the_cavity() {
        // Outer shell outward, inner shell inward: a 1×1×1 box with a
        // 0.5-side cavity holds 1 − 0.125 of material.
        let coords = insert(Coords::new(3).unwrap());
        let outer = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let inner = box_nodes(&coords, [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in &BOX_FACETS {
            sm.add_cell(&[outer[f[0]], outer[f[1]], outer[f[2]]])
                .unwrap();
            // Reversed: the cavity's normals point into the hole.
            sm.add_cell(&[inner[f[0]], inner[f[2]], inner[f[1]]])
                .unwrap();
        }
        let env = Envelope::extract(&Mesh::from_submesh(sm), &NoCancel).unwrap();
        assert_eq!(env.facets().len(), 24);
        assert!(
            (env.volume() - (1.0 - 0.125)).abs() < 1e-15,
            "{}",
            env.volume()
        );
    }

    #[test]
    fn rejects_a_non_tri3_envelope() {
        let coords = insert(Coords::new(3).unwrap());
        let n = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        sm.add_cell(&[n[0], n[1], n[2], n[3]]).unwrap();
        let err = message(Envelope::extract(&Mesh::from_submesh(sm), &NoCancel).unwrap_err());
        assert!(err.contains("TRI3"), "{err}");
        assert!(err.contains("QUA4"), "{err}");
    }

    #[test]
    fn rejects_a_two_dimensional_coords() {
        let coords = insert(Coords::new(2).unwrap());
        let ids: Vec<NodeId> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&ids).unwrap();
        let err = message(Envelope::extract(&Mesh::from_submesh(sm), &NoCancel).unwrap_err());
        assert!(err.contains("3-D"), "{err}");
    }

    #[test]
    fn rejects_an_open_surface() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Drop one facet: the box now has a triangular hole.
        let err = message(
            Envelope::extract(&surface(&coords, &nodes, &BOX_FACETS[1..]), &NoCancel).unwrap_err(),
        );
        assert!(err.contains("not closed"), "{err}");
    }

    #[test]
    fn rejects_an_inconsistently_oriented_surface() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut facets = BOX_FACETS.to_vec();
        facets[5].swap(1, 2); // one facet now faces the other way
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &facets), &NoCancel).unwrap_err());
        assert!(err.contains("same direction"), "{err}");
    }

    #[test]
    fn rejects_inward_normals() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let flipped: Vec<[usize; 3]> = BOX_FACETS.iter().map(|f| [f[0], f[2], f[1]]).collect();
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &flipped), &NoCancel).unwrap_err());
        assert!(err.contains("invert()"), "{err}");
    }

    #[test]
    fn rejects_a_duplicated_facet() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut facets = BOX_FACETS.to_vec();
        facets.push(BOX_FACETS[0]);
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &facets), &NoCancel).unwrap_err());
        assert!(err.contains("same three"), "{err}");
    }

    #[test]
    fn rejects_a_repeated_node_in_a_facet() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut facets = BOX_FACETS.to_vec();
        facets[0] = [0, 3, 3];
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &facets), &NoCancel).unwrap_err());
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn rejects_coincident_nodes() {
        let coords = insert(Coords::new(3).unwrap());
        let mut nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // A ninth node exactly on top of corner 0, used in its place by one
        // facet: the surface looks torn even though it draws the same shape.
        let twin = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0])
            .unwrap()
            .id();
        nodes.push(twin);
        let mut facets = BOX_FACETS.to_vec();
        facets[0] = [8, 3, 2];
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &facets), &NoCancel).unwrap_err());
        assert!(err.contains("merge_nodes()"), "{err}");
    }

    #[test]
    fn rejects_a_flat_facet() {
        let coords = insert(Coords::new(3).unwrap());
        let mut nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // A node exactly halfway along the edge 0-1, used to build a facet
        // whose three nodes are collinear.
        nodes.push(
            Node::create_in(coords.clone(), &[0.5, 0.0, 0.0])
                .unwrap()
                .id(),
        );
        let mut facets = BOX_FACETS.to_vec();
        facets[0] = [0, 8, 1];
        let err =
            message(Envelope::extract(&surface(&coords, &nodes, &facets), &NoCancel).unwrap_err());
        assert!(err.contains("flat"), "{err}");
    }

    #[test]
    fn rejects_an_envelope_too_small_to_close() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let err = message(
            Envelope::extract(&surface(&coords, &nodes, &BOX_FACETS[..2]), &NoCancel).unwrap_err(),
        );
        assert!(err.contains("at least 4"), "{err}");
    }

    #[test]
    fn rejects_a_self_intersecting_envelope() {
        // Two boxes that overlap in space but share no node: closed,
        // manifold, consistently oriented — and yet with no inside.
        let coords = insert(Coords::new(3).unwrap());
        let a = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_nodes(&coords, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for nodes in [&a, &b] {
            for f in &BOX_FACETS {
                sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                    .unwrap();
            }
        }
        let err = message(Envelope::extract(&Mesh::from_submesh(sm), &NoCancel).unwrap_err());
        assert!(err.contains("pass through each other"), "{err}");
    }

    #[test]
    fn accepts_two_disjoint_bodies() {
        // Nothing in the contract forbids meshing two separate solids at
        // once: each is closed, and the volumes simply add up.
        let coords = insert(Coords::new(3).unwrap());
        let a = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_nodes(&coords, [5.0, 0.0, 0.0], [7.0, 1.0, 1.0]);
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for nodes in [&a, &b] {
            for f in &BOX_FACETS {
                sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                    .unwrap();
            }
        }
        let env = Envelope::extract(&Mesh::from_submesh(sm), &NoCancel).unwrap();
        assert!((env.volume() - 3.0).abs() < 1e-15, "{}", env.volume());
    }

    #[test]
    fn stops_on_a_preset_cancellation_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(3).unwrap());
        let flag = AtomicBool::new(true);
        let err = Envelope::extract(&unit_box(&coords), &flag).unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    fn rejects_an_empty_mesh() {
        let err = message(Envelope::extract(&Mesh::empty(), &NoCancel).unwrap_err());
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn accepts_an_envelope_split_across_submeshes() {
        // `skin` hands back one submesh per flat face; the envelope is the
        // union of them and must be read as a single surface.
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut mesh = Mesh::empty();
        for pair in BOX_FACETS.chunks(2) {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            for f in pair {
                sm.add_cell(&[nodes[f[0]], nodes[f[1]], nodes[f[2]]])
                    .unwrap();
            }
            mesh.add_sub(insert(sm)).unwrap();
        }
        let env = Envelope::extract(&mesh, &NoCancel).unwrap();
        assert_eq!(env.facets().len(), 12);
        assert_eq!(env.points().len(), 8);
        assert!((env.volume() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn local_indices_map_back_to_the_input_nodes() {
        let coords = insert(Coords::new(3).unwrap());
        let nodes = box_nodes(&coords, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let env = Envelope::extract(&surface(&coords, &nodes, &BOX_FACETS), &NoCancel).unwrap();
        // Every envelope node is one of the input nodes, at its position.
        let c = read(&coords).unwrap();
        for (i, &nid) in env.node_ids().iter().enumerate() {
            assert!(nodes.contains(&nid));
            assert_eq!(c.coord(nid).unwrap(), env.points()[i]);
        }
    }
}
