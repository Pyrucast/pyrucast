//! Incremental Delaunay tetrahedralization of a point cloud.
//!
//! The kernel every later phase builds on: an array of tetrahedra with
//! face adjacency, grown one point at a time by the Bowyer-Watson rule —
//! delete every tetrahedron whose circumsphere contains the new point, then
//! fill the hole by joining the point to each face of its boundary.
//!
//! Two properties make the rest of the mesher possible:
//!
//! - **every decision is exact.** Membership in a cavity is
//!   [`insphere`] and nothing else, so the cavity is always the true one.
//!   That is what guarantees it stays *star-shaped* — visible in its
//!   entirety from the new point — which is in turn what makes the refill
//!   produce well-formed tetrahedra. An approximate predicate breaks that
//!   invariant on cospherical input and corrupts the adjacency, which is the
//!   classic way an incremental mesher hangs or emits overlapping cells.
//! - **the result does not depend on the machine.** Insertion order comes
//!   from a spatial sort keyed on the coordinates and an index-seeded
//!   round assignment, never from a random generator, so the same cloud
//!   always yields the same triangulation.
//!
//! Cospherical points are not perturbed away: `insphere == 0` simply means
//! "not in the cavity". The cavity stays minimal and star-shaped, and the
//! triangulation is a valid Delaunay triangulation — one of the several the
//! degenerate configuration admits, chosen consistently.
//!
//! The cloud is bootstrapped inside a large enclosing tetrahedron, whose
//! four corners and every tetrahedron touching them are removed at the end.
//! What survives triangulates the convex hull of the input.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;

use super::predicates::{insphere, orient3d};

/// Sentinel for "no neighbour through this face".
pub const NO_TET: u32 = u32::MAX;

/// The cells around an edge and the ring of vertices left when it goes.
#[derive(Debug, Clone)]
pub struct EdgeFan {
    /// Cells sharing the edge, in walking order.
    pub cells: Vec<u32>,
    /// Vertices facing the edge: a cycle when the fan is closed, a path
    /// from one outer face to the other when it is open.
    pub link: Vec<u32>,
    /// Whether the fan wraps around the edge.
    pub closed: bool,
}

/// What a region swap is allowed to do to the outer surface of the region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// The new cells must present exactly the old outer faces. Anything else
    /// is a crack or an overlap.
    Preserved,
    /// The outer faces may be re-cut, but only where the region met nothing
    /// at all — the outer surface of the whole mesh. This is what lets the
    /// 2-2 flip swap the diagonal of a flat quadrilateral on the hull; the
    /// swept volume is checked instead.
    MayRecutHull,
}

/// The face opposite each vertex, wound so that its normal points **out of**
/// the tetrahedron.
///
/// Identical to the table [`crate::ops::mesh::skin()`] uses to peel a
/// `TET4` mesh, so a face handed to either means the same thing.
pub const FACE_OF: [[usize; 3]; 4] = [
    [1, 2, 3], // opposite vertex 0
    [0, 3, 2], // opposite vertex 1
    [0, 1, 3], // opposite vertex 2
    [0, 2, 1], // opposite vertex 3
];

/// How far outside the cloud the bootstrap tetrahedron is pushed, as a
/// multiple of the bounding-box radius.
///
/// Large enough that its corners never take part in a circumsphere decision
/// among real points, so removing them at the end leaves exactly the
/// Delaunay triangulation of the cloud.
const ENCLOSURE_MARGIN: f64 = 1.0e3;

#[derive(Debug, Clone, Copy)]
struct Tet {
    v: [u32; 4],
    /// `nb[i]` is the tetrahedron across the face opposite vertex `i`.
    nb: [u32; 4],
    dead: bool,
}

/// A tetrahedral mesh with face adjacency.
///
/// Slots are stable: killing a tetrahedron leaves a hole that a later
/// allocation reuses, and indices handed out earlier stay meaningful for
/// live tetrahedra.
#[derive(Debug, Clone)]
pub struct TetMesh {
    points: Vec<[f64; 3]>,
    tets: Vec<Tet>,
    free: Vec<u32>,
    /// Where the next point location walk starts.
    hint: u32,
    /// Scratch for cavity marking, stamped per insertion so it never needs
    /// clearing.
    mark: Vec<u32>,
    stamp: u32,
    /// One incident tetrahedron per vertex, so a star can be found without
    /// scanning. Refreshed on every allocation, and repaired in place by the
    /// lookup itself when it turns out to be stale — hence the [`Cell`],
    /// which lets a `&self` query mend its own index.
    vertex_tet: Vec<Cell<u32>>,
}

impl TetMesh {
    /// Delaunay tetrahedralization of `points`.
    ///
    /// The point indices of the result are those of `points`; the four
    /// bootstrap corners are appended internally and gone by the time this
    /// returns.
    pub fn delaunay(points: &[[f64; 3]], cancel: &dyn Cancel) -> Result<TetMesh> {
        if points.len() < 4 {
            return Err(PyrucastError::Message(format!(
                "triangulate_volume: a tetrahedralization needs at least 4 points, got {}",
                points.len()
            )));
        }
        let mut mesh = TetMesh::bootstrap(points);
        for &i in &spatial_order(points) {
            cancel.check()?;
            mesh.insert(i)?;
        }
        mesh.drop_enclosure();
        Ok(mesh)
    }

    /// The cloud plus a big enclosing tetrahedron, as a single cell.
    fn bootstrap(points: &[[f64; 3]]) -> TetMesh {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let centre = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let radius = (0..3)
            .map(|k| (hi[k] - lo[k]).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);

        // A regular tetrahedron of circumradius `s` has inradius `s / 3`, so
        // this one comfortably contains the ball the cloud sits in.
        let s = 3.0 * ENCLOSURE_MARGIN * radius;
        let d = s / 3f64.sqrt();
        // Ordered so that the cell is positively oriented (checked below).
        let corners = [
            [centre[0] + d, centre[1] + d, centre[2] + d],
            [centre[0] - d, centre[1] + d, centre[2] - d],
            [centre[0] + d, centre[1] - d, centre[2] - d],
            [centre[0] - d, centre[1] - d, centre[2] + d],
        ];

        let n = points.len() as u32;
        let mut all = points.to_vec();
        all.extend_from_slice(&corners);
        debug_assert!(
            orient3d(&corners[0], &corners[1], &corners[2], &corners[3]) > 0.0,
            "the bootstrap tetrahedron must be positively oriented"
        );

        TetMesh {
            points: all,
            tets: vec![Tet {
                v: [n, n + 1, n + 2, n + 3],
                nb: [NO_TET; 4],
                dead: false,
            }],
            free: Vec::new(),
            hint: 0,
            mark: vec![0; 1],
            stamp: 0,
            vertex_tet: vec![Cell::new(0); (n + 4) as usize],
        }
    }

    /// Insert point `p_idx`, replacing every tetrahedron whose circumsphere
    /// contains it.
    fn insert(&mut self, p_idx: u32) -> Result<()> {
        let p = self.points[p_idx as usize];
        let seed = self.locate(&p)?;
        let cavity = self.collect_cavity(&p, seed);
        // Each boundary face is wound outwards from the cavity, so `p` lies
        // below it; reversing two of its vertices makes the new cell
        // positively oriented. A cavity that is not star-shaped shows up
        // here as a non-positive cell, which `replace_region` rejects —
        // exact predicates rule that out, so it would mean a bug.
        let new: Vec<[u32; 4]> = self
            .cavity_boundary(&cavity)
            .into_iter()
            .map(|f| [f[0], f[2], f[1], p_idx])
            .collect();
        let created = self.replace_region(&cavity, &new, "the Bowyer-Watson cavity")?;
        self.hint = *created.last().unwrap_or(&self.hint);
        Ok(())
    }

    /// Swap one set of tetrahedra for another covering the same region.
    ///
    /// This is the single primitive every structural change goes through —
    /// point insertion, the flips, boundary recovery. It re-links the
    /// adjacency from the faces alone, and in doing so **proves** the swap is
    /// sound: the new cells must be positively oriented, and their outer
    /// faces must match the old region's outer faces exactly. A replacement
    /// that would leave a crack or an overlap cannot go through unnoticed.
    ///
    /// Returns the slots of the new tetrahedra.
    pub(super) fn replace_region(
        &mut self,
        old: &[u32],
        new: &[[u32; 4]],
        what: &str,
    ) -> Result<Vec<u32>> {
        self.replace_region_with(old, new, what, Boundary::Preserved)
    }

    /// [`Self::replace_region`], with control over what may happen to the
    /// region's outer surface.
    pub(super) fn replace_region_with(
        &mut self,
        old: &[u32],
        new: &[[u32; 4]],
        what: &str,
        boundary: Boundary,
    ) -> Result<Vec<u32>> {
        for v in new {
            if self.orientation(v) <= 0.0 {
                return Err(PyrucastError::Message(format!(
                    "triangulate_volume: replacing {what} would create an inverted or flat \
                     tetrahedron (internal error)"
                )));
            }
        }

        // The region's outer faces, with whatever lies beyond each, captured
        // before anything is torn down.
        let in_old: HashSet<u32> = old.iter().copied().collect();
        let mut outside: HashMap<[u32; 3], u32> = HashMap::with_capacity(2 * old.len() + 4);
        for &t in old {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n != NO_TET && in_old.contains(&n) {
                    continue; // interior to the region
                }
                outside.insert(sorted3(self.face(t as usize, i)), n);
            }
        }

        // Whether the new cells tile the same region as the old ones is
        // decided by their faces alone, so it is decided *here* — before
        // anything is torn down. A caller then never has to keep a copy of
        // the mesh in case the swap is refused, and that matters: the copy is
        // O(mesh) while the swap is O(region), so paying it once per swap
        // turns any pass that makes many swaps quadratic.
        let mut counts: HashMap<[u32; 3], usize> = HashMap::with_capacity(4 * new.len());
        for v in new {
            for f in FACE_OF {
                *counts
                    .entry(sorted3([v[f[0]], v[f[1]], v[f[2]]]))
                    .or_insert(0) += 1;
            }
        }
        // A face used twice by the new cells is interior to them; used once,
        // it has to be one the region already presented.
        let unmatched = counts
            .iter()
            .filter(|(key, &n)| n == 1 && !outside.contains_key(*key))
            .count();
        if unmatched > 0 {
            // Faces the new cells do not present are a crack — unless the
            // caller asked for the hull to be re-cut, in which case both the
            // faces dropped and the faces gained must lie where the region
            // met nothing at all, and the swept volume must be unchanged.
            let dropped_faced_nothing = outside
                .iter()
                .all(|(key, &n)| counts.contains_key(key) || n == NO_TET);
            let volume_kept = {
                let before: f64 = old
                    .iter()
                    .map(|&t| self.orientation(&self.tets[t as usize].v))
                    .sum();
                let after: f64 = new.iter().map(|v| self.orientation(v)).sum();
                (after - before).abs() <= 1e-9 * before.abs().max(after.abs())
            };
            if boundary != Boundary::MayRecutHull || !dropped_faced_nothing || !volume_kept {
                return Err(PyrucastError::Message(format!(
                    "triangulate_volume: replacing {what} would leave {unmatched} unmatched face(s) — \
                     the new cells do not tile the same region (internal error)"
                )));
            }
        }

        // The vertices about to lose their cells: their hints have to be
        // pointed somewhere live again afterwards, or the next star walk
        // falls back to scanning the whole mesh — which is O(n) per query
        // and quietly turns the whole recovery quadratic.
        let mut touched: Vec<u32> = old.iter().flat_map(|&t| self.tets[t as usize].v).collect();
        touched.sort_unstable();
        touched.dedup();

        for &t in old {
            self.kill(t);
        }
        let created: Vec<u32> = new.iter().map(|&v| self.alloc(v)).collect();

        let mut pending: HashMap<[u32; 3], (u32, usize)> = HashMap::with_capacity(new.len());
        for &t in &created {
            for i in 0..4 {
                let key = sorted3(self.face(t as usize, i));
                if let Some(&n) = outside.get(&key) {
                    self.tets[t as usize].nb[i] = n;
                    if n != NO_TET {
                        self.relink(n, &key, t);
                    }
                } else if let Some((other, j)) = pending.remove(&key) {
                    self.tets[t as usize].nb[i] = other;
                    self.tets[other as usize].nb[j] = t;
                } else {
                    pending.insert(key, (t, i));
                }
            }
        }
        // `alloc` has re-pointed every vertex the new cells use; the rest —
        // vertices the region gave up — are found on the cells just outside
        // it, which the region touched by definition.
        for x in touched {
            let hint = self.vertex_tet[x as usize].get();
            let good = hint != NO_TET
                && !self.tets[hint as usize].dead
                && self.tets[hint as usize].v.contains(&x);
            if good {
                continue;
            }
            if let Some(&n) = outside.values().find(|&&n| {
                n != NO_TET && !self.tets[n as usize].dead && self.tets[n as usize].v.contains(&x)
            }) {
                self.vertex_tet[x as usize].set(n);
            }
        }

        debug_assert_eq!(
            pending.len(),
            unmatched,
            "the pre-flight face count must agree with what the relinking found"
        );
        Ok(created)
    }

    /// Walk to the tetrahedron containing `p`.
    ///
    /// From a cell, `p` is outside through any face it lies strictly above;
    /// stepping through that face gets monotonically closer, and the walk
    /// ends where no face is crossed.
    fn locate(&self, p: &[f64; 3]) -> Result<usize> {
        let mut t = self.hint as usize;
        if t >= self.tets.len() || self.tets[t].dead {
            t = self.first_live().ok_or_else(|| {
                PyrucastError::Message("triangulate_volume: the triangulation is empty".into())
            })?;
        }
        // Generous, but finite: a correct walk is far shorter, and a budget
        // is what turns a hypothetical cycle into an error instead of a hang.
        let budget = 8 * self.tets.len() + 64;
        for step in 0..budget {
            let tet = self.tets[t];
            let mut moved = false;
            // Rotating the first face examined with the step count keeps the
            // walk from oscillating between two cells on degenerate input,
            // deterministically.
            for k in 0..4 {
                let i = (k + step) % 4;
                let f = self.face(t, i);
                if orient3d(
                    &self.points[f[0] as usize],
                    &self.points[f[1] as usize],
                    &self.points[f[2] as usize],
                    p,
                ) > 0.0
                {
                    let n = tet.nb[i];
                    if n == NO_TET {
                        return Err(PyrucastError::Message(
                            "triangulate_volume: a point fell outside the bootstrap tetrahedron"
                                .into(),
                        ));
                    }
                    t = n as usize;
                    moved = true;
                    break;
                }
            }
            if !moved {
                return Ok(t);
            }
        }
        Err(PyrucastError::Message(
            "triangulate_volume: point location did not converge (internal error)".into(),
        ))
    }

    /// Every tetrahedron whose circumsphere strictly contains `p`, found by
    /// walking outwards from `seed` across faces.
    ///
    /// The set is connected — a property of the Delaunay triangulation, and
    /// the reason a face walk finds all of it.
    fn collect_cavity(&mut self, p: &[f64; 3], seed: usize) -> Vec<u32> {
        self.stamp += 1;
        let stamp = self.stamp;
        self.mark.resize(self.tets.len(), 0);

        let mut cavity = vec![seed as u32];
        self.mark[seed] = stamp;
        let mut stack = vec![seed as u32];
        while let Some(t) = stack.pop() {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n == NO_TET || self.mark[n as usize] == stamp {
                    continue;
                }
                let v = self.tets[n as usize].v;
                if insphere(
                    &self.points[v[0] as usize],
                    &self.points[v[1] as usize],
                    &self.points[v[2] as usize],
                    &self.points[v[3] as usize],
                    p,
                ) > 0.0
                {
                    self.mark[n as usize] = stamp;
                    cavity.push(n);
                    stack.push(n);
                }
            }
        }
        cavity
    }

    /// Faces of the cavity that face outwards.
    fn cavity_boundary(&self, cavity: &[u32]) -> Vec<[u32; 3]> {
        let stamp = self.stamp;
        let mut faces = Vec::with_capacity(2 * cavity.len() + 2);
        for &t in cavity {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n != NO_TET && self.mark[n as usize] == stamp {
                    continue; // interior to the cavity
                }
                faces.push(self.face(t as usize, i));
            }
        }
        faces
    }

    /// Point `outside`'s face matching `f` back at the new tetrahedron `t`.
    fn relink(&mut self, outside: u32, f: &[u32; 3], t: u32) {
        let key = sorted3(*f);
        for i in 0..4 {
            if sorted3(self.face(outside as usize, i)) == key {
                self.tets[outside as usize].nb[i] = t;
                return;
            }
        }
        debug_assert!(false, "the outside cell must own the boundary face");
    }

    /// Remove the bootstrap tetrahedron's corners and everything touching
    /// them.
    fn drop_enclosure(&mut self) {
        let first_corner = (self.points.len() - 4) as u32;
        let doomed: Vec<u32> = (0..self.tets.len() as u32)
            .filter(|&t| {
                !self.tets[t as usize].dead
                    && self.tets[t as usize].v.iter().any(|&v| v >= first_corner)
            })
            .collect();
        for t in doomed {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n != NO_TET {
                    // Sever the link from the surviving side too.
                    for j in 0..4 {
                        if self.tets[n as usize].nb[j] == t {
                            self.tets[n as usize].nb[j] = NO_TET;
                        }
                    }
                }
            }
            self.kill(t);
        }
        self.points.truncate(first_corner as usize);
        self.vertex_tet.truncate(first_corner as usize);
        // Wholesale demolition leaves most hints pointing at nothing. Since
        // the fallback is a scan of the entire mesh, and nothing repairs a
        // hint once it is stale, one bad hint costs a scan on *every* later
        // query for that vertex — which is what turns recovery quadratic.
        // One pass now fixes all of them.
        for slot in &self.vertex_tet {
            slot.set(NO_TET);
        }
        for t in 0..self.tets.len() {
            if self.tets[t].dead {
                continue;
            }
            for &x in &self.tets[t].v {
                self.vertex_tet[x as usize].set(t as u32);
            }
        }
        self.hint = self.first_live().unwrap_or(0) as u32;
    }

    /// Move a point. The caller is responsible for checking that the cells
    /// around it stay well formed.
    pub(super) fn set_point(&mut self, i: u32, p: [f64; 3]) {
        self.points[i as usize] = p;
    }

    /// Add a point to the cloud, returning its index.
    pub(super) fn add_point(&mut self, p: [f64; 3]) -> u32 {
        self.points.push(p);
        self.vertex_tet.push(Cell::new(NO_TET));
        (self.points.len() - 1) as u32
    }

    /// Insert `p` inside the region bounded by `walls`, returning the cells
    /// that replaced its cavity.
    ///
    /// This is Bowyer-Watson again, with two differences that matter once
    /// the mesh is no longer a plain Delaunay triangulation. The cavity
    /// **stops at a wall**, so a point can never eat through the surface it
    /// is meant to stay inside; and star-shapedness, which the unconstrained
    /// version gets for free, is now checked and enforced by shrinking the
    /// cavity — stopping at walls can leave a face the point cannot see, and
    /// joining to it would fold the mesh.
    ///
    /// `wall_ok` is consulted for every wall the cavity runs into; a `false`
    /// calls the whole insertion off. That is how the caller keeps a point
    /// from being placed too near the surface — the condition Delaunay
    /// refinement needs in order to terminate at all.
    ///
    /// `None` when `p` does not land in a cell of `region`, or when a wall
    /// refused it.
    pub(super) fn insert_within(
        &mut self,
        p: [f64; 3],
        region: &[bool],
        walls: &HashSet<[u32; 3]>,
        wall_ok: &dyn Fn(&[[f64; 3]; 3]) -> bool,
    ) -> Result<Option<Vec<u32>>> {
        let Some(seed) = self.locate_within(&p, region, walls) else {
            return Ok(None);
        };

        self.stamp += 1;
        let stamp = self.stamp;
        self.mark.resize(self.tets.len(), 0);
        let mut cavity = vec![seed as u32];
        self.mark[seed] = stamp;
        let mut stack = vec![seed as u32];
        while let Some(t) = stack.pop() {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n == NO_TET || self.mark[n as usize] == stamp {
                    continue;
                }
                let face = self.face(t as usize, i);
                if walls.contains(&sorted3(face)) {
                    let corners = [
                        self.points[face[0] as usize],
                        self.points[face[1] as usize],
                        self.points[face[2] as usize],
                    ];
                    if !wall_ok(&corners) {
                        return Ok(None); // too close to the surface
                    }
                    continue;
                }
                if !region.get(n as usize).copied().unwrap_or(false) {
                    continue; // beyond the region
                }
                let v = self.tets[n as usize].v;
                if insphere(
                    &self.points[v[0] as usize],
                    &self.points[v[1] as usize],
                    &self.points[v[2] as usize],
                    &self.points[v[3] as usize],
                    &p,
                ) > 0.0
                {
                    self.mark[n as usize] = stamp;
                    cavity.push(n);
                    stack.push(n);
                }
            }
        }

        // Shrink until every face of the cavity is visible from `p`.
        let idx = self.add_point(p);
        loop {
            let faces = self.cavity_boundary(&cavity);
            let blind = faces
                .iter()
                .find(|f| self.orientation(&[f[0], f[2], f[1], idx]) <= 0.0);
            let Some(&f) = blind else { break };
            // Drop the cell that owns the offending face; the seed never is,
            // since `p` lies inside it.
            let key = sorted3(f);
            let owner = cavity.iter().position(|&t| {
                t != seed as u32 && (0..4).any(|i| sorted3(self.face(t as usize, i)) == key)
            });
            match owner {
                Some(k) => {
                    self.mark[cavity[k] as usize] = 0;
                    cavity.swap_remove(k);
                }
                None => {
                    self.points.truncate(idx as usize);
                    self.vertex_tet.truncate(idx as usize);
                    return Ok(None);
                }
            }
        }

        let new: Vec<[u32; 4]> = self
            .cavity_boundary(&cavity)
            .into_iter()
            .map(|f| [f[0], f[2], f[1], idx])
            .collect();
        let created = self.replace_region(&cavity, &new, "a refinement insertion")?;
        self.hint = *created.last().unwrap_or(&self.hint);
        Ok(Some(created))
    }

    /// Walk to the cell of `region` holding `p`, never crossing a wall.
    fn locate_within(
        &self,
        p: &[f64; 3],
        region: &[bool],
        walls: &HashSet<[u32; 3]>,
    ) -> Option<usize> {
        let mut t = self.hint as usize;
        if t >= self.tets.len() || self.tets[t].dead || !region.get(t).copied().unwrap_or(false) {
            t = (0..self.tets.len())
                .find(|&i| !self.tets[i].dead && region.get(i).copied().unwrap_or(false))?;
        }
        let budget = 8 * self.tets.len() + 64;
        for step in 0..budget {
            let mut moved = false;
            for k in 0..4 {
                let i = (k + step) % 4;
                let f = self.face(t, i);
                if orient3d(
                    &self.points[f[0] as usize],
                    &self.points[f[1] as usize],
                    &self.points[f[2] as usize],
                    p,
                ) <= 0.0
                {
                    continue;
                }
                // `p` is beyond this face; a wall there means it is outside.
                if walls.contains(&sorted3(f)) {
                    return None;
                }
                let n = self.tets[t].nb[i];
                if n == NO_TET || !region.get(n as usize).copied().unwrap_or(false) {
                    return None;
                }
                t = n as usize;
                moved = true;
                break;
            }
            if !moved {
                return Some(t);
            }
        }
        None
    }

    // ─── Storage ────────────────────────────────────────────────────────

    fn alloc(&mut self, v: [u32; 4]) -> u32 {
        let tet = Tet {
            v,
            nb: [NO_TET; 4],
            dead: false,
        };
        let slot = match self.free.pop() {
            Some(i) => {
                self.tets[i as usize] = tet;
                i
            }
            None => {
                self.tets.push(tet);
                self.mark.push(0);
                (self.tets.len() - 1) as u32
            }
        };
        for &x in &v {
            if self.vertex_tet.len() <= x as usize {
                self.vertex_tet.resize(x as usize + 1, Cell::new(NO_TET));
            }
            self.vertex_tet[x as usize].set(slot);
        }
        slot
    }

    fn kill(&mut self, t: u32) {
        self.tets[t as usize].dead = true;
        self.free.push(t);
    }

    fn first_live(&self) -> Option<usize> {
        (0..self.tets.len()).find(|&t| !self.tets[t].dead)
    }

    // ─── Neighbourhood queries ──────────────────────────────────────────

    /// A live tetrahedron having `v` as a vertex.
    fn seed_tet_for(&self, v: u32) -> Option<u32> {
        if let Some(slot) = self.vertex_tet.get(v as usize) {
            let t = slot.get();
            if t != NO_TET && !self.tets[t as usize].dead && self.tets[t as usize].v.contains(&v) {
                return Some(t);
            }
        }
        let found = (0..self.tets.len() as u32)
            .find(|&t| !self.tets[t as usize].dead && self.tets[t as usize].v.contains(&v))?;
        // Remember it: the fallback is a scan of the whole mesh, so paying
        // it twice for the same vertex is pure waste.
        self.vertex_tet[v as usize].set(found);
        Some(found)
    }

    /// Every tetrahedron having `v` as a vertex.
    ///
    /// Found by walking outwards from one of them: crossing any face that
    /// contains `v` lands on another cell that does too, and the star is
    /// connected.
    pub(super) fn tets_around_vertex(&self, v: u32) -> Vec<u32> {
        let Some(seed) = self.seed_tet_for(v) else {
            return Vec::new();
        };
        let mut star = vec![seed];
        let mut seen: HashSet<u32> = HashSet::from([seed]);
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            let cell = self.tets[t as usize];
            let at = cell
                .v
                .iter()
                .position(|&x| x == v)
                .expect("star cell holds v");
            for i in 0..4 {
                if i == at {
                    continue; // the one face that misses `v`
                }
                let n = cell.nb[i];
                if n != NO_TET && seen.insert(n) {
                    star.push(n);
                    stack.push(n);
                }
            }
        }
        star
    }

    /// The closed ring of tetrahedra sharing edge `(u, v)`, in walking order.
    ///
    /// `None` when the edge is absent, or when its fan is open — which for
    /// an interior edge means it reaches the outer boundary, and makes the
    /// edge unflippable.
    pub(super) fn tets_around_edge(&self, u: u32, v: u32) -> Option<Vec<u32>> {
        let start = self
            .tets_around_vertex(u)
            .into_iter()
            .find(|&t| self.tets[t as usize].v.contains(&v))?;

        let mut ring = vec![start];
        let mut prev = NO_TET;
        let mut cur = start;
        loop {
            let cell = self.tets[cur as usize];
            // The two faces holding the whole edge are those opposite the
            // two vertices that are neither `u` nor `v`.
            let next = (0..4)
                .filter(|&i| cell.v[i] != u && cell.v[i] != v)
                .map(|i| cell.nb[i])
                .find(|&n| n != prev && n != NO_TET)?;
            if next == start {
                return Some(ring);
            }
            ring.push(next);
            prev = cur;
            cur = next;
            if ring.len() > self.tets.len() {
                return None; // not a ring; refuse rather than spin
            }
        }
    }

    /// The vertex of `t` that does not lie on face `f`.
    pub(super) fn apex_beyond(&self, t: usize, f: &[u32; 3]) -> Option<u32> {
        let cell = self.tets.get(t)?;
        if cell.dead {
            return None;
        }
        cell.v.iter().copied().find(|x| !f.contains(x))
    }

    /// Temporary probe accessor.
    pub fn has_edge_pub(&self, u: u32, v: u32) -> bool {
        self.has_edge(u, v)
    }

    /// Whether `(u, v)` is an edge of some tetrahedron.
    pub(super) fn has_edge(&self, u: u32, v: u32) -> bool {
        self.tets_around_vertex(u)
            .iter()
            .any(|&t| self.tets[t as usize].v.contains(&v))
    }

    /// The cells around an edge, in walking order, with the ring of
    /// vertices they leave behind once the edge is taken away.
    ///
    /// `cells[i]` and `cells[i + 1]` share the face holding the edge and
    /// `link[i + 1]`. A **closed** fan wraps around, and `link` is a cycle of
    /// `cells.len()` vertices; an **open** one stops at the outer surface,
    /// and `link` is a path of `cells.len() + 1`.
    pub(super) fn edge_fan(&self, u: u32, v: u32) -> Option<EdgeFan> {
        let start = self
            .tets_around_vertex(u)
            .into_iter()
            .find(|&t| self.tets[t as usize].v.contains(&v))?;

        // Walk one way; if the surface stops us, walk the other way from the
        // start and put the two halves together.
        let (mut cells, closed) = self.walk_fan(u, v, start, NO_TET)?;
        if !closed {
            let mut back = self
                .walk_fan(u, v, start, cells.get(1).copied().unwrap_or(NO_TET))?
                .0;
            back.reverse();
            back.pop(); // `start` is in both halves
            back.extend_from_slice(&cells);
            cells = back;
        }

        // The vertex two consecutive cells share, besides the edge itself.
        let others = |t: u32| -> Vec<u32> {
            self.tets[t as usize]
                .v
                .iter()
                .copied()
                .filter(|&x| x != u && x != v)
                .collect()
        };
        let mut link: Vec<u32> = Vec::with_capacity(cells.len() + 1);
        if !closed {
            let (a, b) = (others(cells[0]), others(cells[1 % cells.len()]));
            link.push(if cells.len() == 1 {
                a[0]
            } else {
                *a.iter().find(|x| !b.contains(x))?
            });
        }
        for w in cells.windows(2) {
            let (a, b) = (others(w[0]), others(w[1]));
            link.push(*a.iter().find(|x| b.contains(x))?);
        }
        if !closed {
            let last = others(*cells.last()?);
            let prev = others(cells[cells.len().saturating_sub(2)]);
            link.push(if cells.len() == 1 {
                last[1]
            } else {
                *last.iter().find(|x| !prev.contains(x))?
            });
        }
        Some(EdgeFan {
            cells,
            link,
            closed,
        })
    }

    /// Walk the cells around edge `(u, v)` from `start`, away from `avoid`.
    ///
    /// Returns the cells visited and whether the walk closed on itself.
    fn walk_fan(&self, u: u32, v: u32, start: u32, avoid: u32) -> Option<(Vec<u32>, bool)> {
        let mut cells = vec![start];
        let mut prev = avoid;
        let mut cur = start;
        loop {
            let cell = self.tets[cur as usize];
            let next = (0..4)
                .filter(|&i| cell.v[i] != u && cell.v[i] != v)
                .map(|i| cell.nb[i])
                .find(|&n| n != prev && n != NO_TET);
            match next {
                None => return Some((cells, false)),
                Some(n) if n == start => return Some((cells, true)),
                Some(n) => {
                    cells.push(n);
                    prev = cur;
                    cur = n;
                    if cells.len() > self.tets.len() {
                        return None;
                    }
                }
            }
        }
    }

    /// The tetrahedra having both `u` and `v` as vertices — the fan around
    /// the edge, whether or not it closes into a ring.
    pub(super) fn tets_with_edge(&self, u: u32, v: u32) -> Vec<u32> {
        let mut fan: Vec<u32> = self
            .tets_around_vertex(u)
            .into_iter()
            .filter(|&t| self.tets[t as usize].v.contains(&v))
            .collect();
        fan.sort_unstable();
        fan
    }

    /// The outward faces of a set of cells taken together — the surface of
    /// the region they fill.
    pub(super) fn region_boundary(&self, cells: &[u32]) -> Vec<[u32; 3]> {
        let inside: HashSet<u32> = cells.iter().copied().collect();
        let mut faces = Vec::with_capacity(2 * cells.len() + 2);
        for &t in cells {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n == NO_TET || !inside.contains(&n) {
                    faces.push(self.face(t as usize, i));
                }
            }
        }
        faces
    }

    /// Whether face `f` of cell `t` meets nothing — it is on the outer
    /// surface of the whole mesh.
    pub(super) fn face_is_free(&self, t: u32, f: &[u32; 3]) -> bool {
        let key = sorted3(*f);
        (0..4).any(|i| {
            sorted3(self.face(t as usize, i)) == key && self.tets[t as usize].nb[i] == NO_TET
        })
    }

    /// Whether `(a, b, c)` is a face of some tetrahedron.
    pub fn has_face(&self, f: &[u32; 3]) -> bool {
        self.face_owners(f).is_some()
    }

    /// The tetrahedra owning face `f`, at most two, with which of their
    /// faces it is.
    pub(super) fn face_owners(&self, f: &[u32; 3]) -> Option<Vec<(u32, usize)>> {
        let key = sorted3(*f);
        let owners: Vec<(u32, usize)> = self
            .tets_around_vertex(f[0])
            .into_iter()
            .filter_map(|t| {
                (0..4)
                    .find(|&i| sorted3(self.face(t as usize, i)) == key)
                    .map(|i| (t, i))
            })
            .collect();
        (!owners.is_empty()).then_some(owners)
    }

    // ─── Reading the mesh ───────────────────────────────────────────────

    /// The vertices of face `i` of tetrahedron `t`, wound outwards.
    pub fn face(&self, t: usize, i: usize) -> [u32; 3] {
        let v = self.tets[t].v;
        let f = FACE_OF[i];
        [v[f[0]], v[f[1]], v[f[2]]]
    }

    /// Node positions, indexed as the tetrahedra reference them.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Vertices of a live tetrahedron, or `None` if the slot is free.
    pub fn tet(&self, t: usize) -> Option<[u32; 4]> {
        let cell = self.tets.get(t)?;
        (!cell.dead).then_some(cell.v)
    }

    /// The tetrahedron across face `i` of `t`, if any.
    pub fn neighbour(&self, t: usize, i: usize) -> Option<usize> {
        match self.tets[t].nb[i] {
            NO_TET => None,
            n => Some(n as usize),
        }
    }

    /// Number of slots, live or free — the bound for an index loop.
    pub fn slot_count(&self) -> usize {
        self.tets.len()
    }

    /// Live tetrahedra, with their slot index.
    pub fn iter(&self) -> impl Iterator<Item = (usize, [u32; 4])> + '_ {
        (0..self.tets.len()).filter_map(|t| self.tet(t).map(|v| (t, v)))
    }

    /// Number of live tetrahedra.
    pub fn len(&self) -> usize {
        self.tets.iter().filter(|t| !t.dead).count()
    }

    /// Whether no tetrahedron is left.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Six times the signed volume of a tetrahedron — positive for every
    /// well-formed cell.
    pub fn orientation(&self, v: &[u32; 4]) -> f64 {
        orient3d(
            &self.points[v[0] as usize],
            &self.points[v[1] as usize],
            &self.points[v[2] as usize],
            &self.points[v[3] as usize],
        )
    }

    /// Total volume of the live tetrahedra.
    pub fn volume(&self) -> f64 {
        self.iter().map(|(_, v)| self.orientation(&v)).sum::<f64>() / 6.0
    }

    /// Check the structural invariants, for tests and for the operator's
    /// final validation.
    ///
    /// Returns a description of the first violation found.
    pub fn find_defect(&self) -> Option<String> {
        for (t, v) in self.iter() {
            if self.orientation(&v) <= 0.0 {
                return Some(format!("tetrahedron {t} is inverted or flat"));
            }
            let mut sorted = v;
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                return Some(format!("tetrahedron {t} repeats a vertex"));
            }
            for i in 0..4 {
                let Some(n) = self.neighbour(t, i) else {
                    continue;
                };
                if self.tet(n).is_none() {
                    return Some(format!("tetrahedron {t} points at dead slot {n}"));
                }
                // Adjacency is symmetric, and the shared face agrees.
                let back = (0..4).find(|&j| self.neighbour(n, j) == Some(t));
                match back {
                    None => return Some(format!("tetrahedron {n} does not point back at {t}")),
                    Some(j) => {
                        if sorted3(self.face(t, i)) != sorted3(self.face(n, j)) {
                            return Some(format!(
                                "tetrahedra {t} and {n} disagree on their shared face"
                            ));
                        }
                    }
                }
            }
        }
        None
    }
}

fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

// ─── Insertion order ────────────────────────────────────────────────────

/// The order points are inserted in: a BRIO — rounds of geometrically
/// growing size, each swept along a space-filling curve.
///
/// The spatial sweep keeps consecutive insertions close together, so the
/// location walk barely moves; the rounds keep the early triangulation
/// spread over the whole cloud, which is what bounds the work per insertion.
/// Both come from the data and the index, never from a random generator, so
/// the order is reproducible.
fn spatial_order(points: &[[f64; 3]]) -> Vec<u32> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut scale = [0.0f64; 3];
    for k in 0..3 {
        let span = hi[k] - lo[k];
        scale[k] = if span > 0.0 {
            ((1u32 << 21) - 1) as f64 / span
        } else {
            0.0
        };
    }

    let mut keys: Vec<(u8, u64, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q = [
                ((p[0] - lo[0]) * scale[0]) as u32,
                ((p[1] - lo[1]) * scale[1]) as u32,
                ((p[2] - lo[2]) * scale[2]) as u32,
            ];
            // Higher rounds hold fewer points and go in first.
            let level = (splitmix(i as u64) | (1 << 20)).trailing_zeros() as u8;
            (u8::MAX - level, morton(q), i as u32)
        })
        .collect();
    keys.sort_unstable();
    keys.into_iter().map(|(_, _, i)| i).collect()
}

/// Interleave the bits of three 21-bit coordinates — the Z-order curve.
fn morton(q: [u32; 3]) -> u64 {
    let spread = |mut x: u64| -> u64 {
        x &= 0x1f_ffff;
        x = (x | x << 32) & 0x001f_0000_0000_ffff;
        x = (x | x << 16) & 0x001f_0000_ff00_00ff;
        x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
        x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
        x = (x | x << 2) & 0x1249_2492_4924_9249;
        x
    };
    spread(q[0] as u64) | spread(q[1] as u64) << 1 | spread(q[2] as u64) << 2
}

/// Deterministic index mixer — the reproducible stand-in for the coin flip
/// BRIO calls for.
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interrupt::NoCancel;

    fn build(points: &[[f64; 3]]) -> TetMesh {
        let mesh = TetMesh::delaunay(points, &NoCancel).unwrap();
        assert_eq!(mesh.find_defect(), None);
        mesh
    }

    /// Points of a regular grid, which is massively cospherical.
    fn grid(n: usize) -> Vec<[f64; 3]> {
        let mut v = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    v.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        v
    }

    /// A deterministic scatter inside the unit cube.
    fn scatter(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let f = |s: u64| (splitmix(i as u64 ^ s) >> 11) as f64 / (1u64 << 53) as f64;
                [f(1), f(2), f(3)]
            })
            .collect()
    }

    #[test]
    fn a_single_tetrahedron_survives_as_itself() {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mesh = build(&p);
        assert_eq!(mesh.len(), 1);
        assert!((mesh.volume() - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn a_cube_is_filled_with_positive_cells() {
        let p = grid(2);
        let mesh = build(&p);
        // Any tetrahedralization of a cube has at least 5 cells.
        assert!(mesh.len() >= 5, "{}", mesh.len());
        assert!((mesh.volume() - 1.0).abs() < 1e-14, "{}", mesh.volume());
    }

    #[test]
    fn the_cells_fill_the_convex_hull_exactly() {
        // The triangulation of a point cloud covers its convex hull, so for
        // a box-shaped cloud the volumes must agree exactly.
        for n in [3, 4, 5] {
            let p = grid(n);
            let mesh = build(&p);
            let side = (n - 1) as f64;
            let expected = side.powi(3);
            assert!(
                (mesh.volume() - expected).abs() < 1e-12 * expected,
                "n={n}: {} vs {expected}",
                mesh.volume()
            );
        }
    }

    #[test]
    fn a_scattered_cloud_triangulates_its_hull() {
        let p = scatter(400);
        let mesh = build(&p);
        assert!(mesh.len() > 400, "{}", mesh.len());
        // Every cell is positive and they tile a subset of the unit cube.
        assert!(
            mesh.volume() > 0.5 && mesh.volume() <= 1.0,
            "{}",
            mesh.volume()
        );
    }

    #[test]
    fn every_input_point_is_used() {
        let p = scatter(200);
        let mesh = build(&p);
        let mut used = vec![false; p.len()];
        for (_, v) in mesh.iter() {
            for &i in &v {
                used[i as usize] = true;
            }
        }
        assert!(used.iter().all(|&u| u), "some points ended up unused");
    }

    #[test]
    fn no_bootstrap_corner_leaks_into_the_result() {
        let p = scatter(150);
        let mesh = build(&p);
        assert_eq!(mesh.points().len(), p.len());
        for (_, v) in mesh.iter() {
            assert!(v.iter().all(|&i| (i as usize) < p.len()));
        }
    }

    #[test]
    fn the_delaunay_property_holds() {
        // The defining property: no point sits inside any circumsphere.
        // Checked exhaustively on a cloud small enough to afford it.
        let p = scatter(120);
        let mesh = build(&p);
        for (t, v) in mesh.iter() {
            for (i, q) in p.iter().enumerate() {
                if v.contains(&(i as u32)) {
                    continue;
                }
                let s = insphere(
                    &p[v[0] as usize],
                    &p[v[1] as usize],
                    &p[v[2] as usize],
                    &p[v[3] as usize],
                    q,
                );
                assert!(
                    s <= 0.0,
                    "point {i} lies inside the circumsphere of tet {t}"
                );
            }
        }
    }

    #[test]
    fn cospherical_input_is_handled_without_perturbation() {
        // A regular grid is the worst case for in-sphere degeneracy: the
        // exact predicate must resolve it without any jitter.
        let p = grid(4);
        let mesh = build(&p);
        assert!((mesh.volume() - 27.0).abs() < 1e-12, "{}", mesh.volume());
    }

    #[test]
    fn coplanar_input_is_rejected_rather_than_mangled() {
        // Four points in a plane enclose no volume; the walk cannot place a
        // fifth, so this must fail cleanly instead of building nonsense.
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mesh = TetMesh::delaunay(&p, &NoCancel).unwrap();
        assert_eq!(mesh.find_defect(), None);
        assert_eq!(mesh.len(), 0, "a flat cloud encloses nothing");
    }

    #[test]
    fn too_few_points_is_an_error() {
        let err = TetMesh::delaunay(&[[0.0, 0.0, 0.0]], &NoCancel).unwrap_err();
        assert!(err.to_string().contains("at least 4"), "{err}");
    }

    #[test]
    fn the_result_does_not_depend_on_the_run() {
        let p = scatter(300);
        let reference: Vec<[u32; 4]> = build(&p).iter().map(|(_, v)| v).collect();
        for _ in 0..3 {
            let again: Vec<[u32; 4]> = build(&p).iter().map(|(_, v)| v).collect();
            assert_eq!(reference, again);
        }
    }

    #[test]
    fn insertion_order_is_a_permutation_of_the_input() {
        let p = scatter(500);
        let mut order = spatial_order(&p);
        assert_eq!(order.len(), p.len());
        order.sort_unstable();
        assert!(order.iter().enumerate().all(|(i, &j)| i as u32 == j));
    }

    #[test]
    fn stops_on_a_preset_cancellation_flag() {
        use std::sync::atomic::AtomicBool;
        let flag = AtomicBool::new(true);
        let err = TetMesh::delaunay(&scatter(50), &flag).unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    #[ignore = "performance check, run explicitly with --ignored"]
    fn perf_hundred_k_points_under_30s() {
        let p = scatter(100_000);
        let start = std::time::Instant::now();
        let mesh = TetMesh::delaunay(&p, &NoCancel).unwrap();
        let elapsed = start.elapsed();
        assert!(mesh.len() > 500_000, "only {} cells", mesh.len());
        assert!(elapsed.as_secs_f64() < 30.0, "took {elapsed:?}");
    }
}
