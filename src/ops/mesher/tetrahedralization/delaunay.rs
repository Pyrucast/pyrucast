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

use std::collections::HashMap;

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;

use super::predicates::{insphere, orient3d};

/// Sentinel for "no neighbour through this face".
pub const NO_TET: u32 = u32::MAX;

/// The face opposite each vertex, wound so that its normal points **out of**
/// the tetrahedron.
///
/// Identical to the table [`crate::ops::mesher::skin()`] uses to peel a
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
                "mesh_volume: a tetrahedralization needs at least 4 points, got {}",
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
        }
    }

    /// Insert point `p_idx`, replacing every tetrahedron whose circumsphere
    /// contains it.
    fn insert(&mut self, p_idx: u32) -> Result<()> {
        let p = self.points[p_idx as usize];
        let seed = self.locate(&p)?;
        let cavity = self.collect_cavity(&p, seed);
        let boundary = self.cavity_boundary(&cavity);
        self.refill(p_idx, &cavity, &boundary)
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
                PyrucastError::Message("mesh_volume: the triangulation is empty".into())
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
                            "mesh_volume: a point fell outside the bootstrap tetrahedron".into(),
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
            "mesh_volume: point location did not converge (internal error)".into(),
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

    /// Faces of the cavity that face outwards, each with the tetrahedron
    /// beyond it and which of that cell's faces it is.
    fn cavity_boundary(&self, cavity: &[u32]) -> Vec<([u32; 3], u32)> {
        let stamp = self.stamp;
        let mut faces = Vec::with_capacity(2 * cavity.len() + 2);
        for &t in cavity {
            for i in 0..4 {
                let n = self.tets[t as usize].nb[i];
                if n != NO_TET && self.mark[n as usize] == stamp {
                    continue; // interior to the cavity
                }
                faces.push((self.face(t as usize, i), n));
            }
        }
        faces
    }

    /// Replace the cavity with one tetrahedron per boundary face, joining it
    /// to the new point.
    fn refill(&mut self, p_idx: u32, cavity: &[u32], boundary: &[([u32; 3], u32)]) -> Result<()> {
        for &t in cavity {
            self.kill(t);
        }

        // `f` is wound outwards from the cavity, so `p` lies below it;
        // reversing two of its vertices makes the new cell positively
        // oriented.
        let mut created: Vec<u32> = Vec::with_capacity(boundary.len());
        for &(f, outside) in boundary {
            let v = [f[0], f[2], f[1], p_idx];
            if orient3d(
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                &self.points[v[3] as usize],
            ) <= 0.0
            {
                // Only reachable if the cavity was not star-shaped, which
                // exact predicates rule out — so this is a bug, not bad input.
                return Err(PyrucastError::Message(
                    "mesh_volume: the Bowyer-Watson cavity was not star-shaped \
                     (internal error)"
                        .into(),
                ));
            }
            let t = self.alloc(v);
            // The face opposite the new point is `f` itself: it keeps the
            // neighbour the cavity had there.
            self.tets[t as usize].nb[3] = outside;
            if outside != NO_TET {
                self.relink(outside, &f, t);
            }
            created.push(t);
        }

        self.link_siblings(&created);
        self.hint = *created.last().unwrap_or(&self.hint);
        Ok(())
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

    /// Join the freshly created tetrahedra to each other: they meet along
    /// the faces that contain the new point, two at a time.
    fn link_siblings(&mut self, created: &[u32]) {
        let mut seen: HashMap<[u32; 3], (u32, usize)> = HashMap::with_capacity(3 * created.len());
        for &t in created {
            // Face 3 is the one opposite the new point, already linked.
            for i in 0..3 {
                let key = sorted3(self.face(t as usize, i));
                match seen.remove(&key) {
                    Some((other, j)) => {
                        self.tets[t as usize].nb[i] = other;
                        self.tets[other as usize].nb[j] = t;
                    }
                    None => {
                        seen.insert(key, (t, i));
                    }
                }
            }
        }
        debug_assert!(
            seen.is_empty(),
            "every interior face of the refill is shared by exactly two cells"
        );
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
        self.hint = self.first_live().unwrap_or(0) as u32;
    }

    // ─── Storage ────────────────────────────────────────────────────────

    fn alloc(&mut self, v: [u32; 4]) -> u32 {
        let tet = Tet {
            v,
            nb: [NO_TET; 4],
            dead: false,
        };
        match self.free.pop() {
            Some(i) => {
                self.tets[i as usize] = tet;
                i
            }
            None => {
                self.tets.push(tet);
                self.mark.push(0);
                (self.tets.len() - 1) as u32
            }
        }
    }

    fn kill(&mut self, t: u32) {
        self.tets[t as usize].dead = true;
        self.free.push(t);
    }

    fn first_live(&self) -> Option<usize> {
        (0..self.tets.len()).find(|&t| !self.tets[t].dead)
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
