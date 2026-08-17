//! The advancing front of the volume mesher: the surface still facing the
//! unmeshed void, and everything that happens to it.
//!
//! Offsetting the whole envelope by one distance is the naive version, and it
//! fails the moment the solid stops being thick everywhere: one narrow neck
//! caps the thickness of every layer, over the whole part. A front does three
//! things instead, and they are what this module is:
//!
//! - **it advances by what the room allows.** Each node's step is capped by
//!   how far the front is from *itself* there — half the distance to the
//!   nearest facet it does not belong to. Where the solid is thick the layer
//!   is full depth; where it pinches, the layer thins rather than the whole
//!   mesh giving up;
//! - **it seams.** Two parts of the front that end up within touching distance
//!   are welded into one, which is how a thin region closes and stops being a
//!   void at all;
//! - **it gets smoothed.** The nodes it created are relaxed under a validity
//!   guard, so a step forced short by a neighbour is not left as a kink.

use super::shell::{Facet, Shell};
use crate::atoms::{Node, NodeId, Point3, Vector3};
use crate::coords::Coords;
use crate::error::Result;
use crate::handle::Handle;
use std::collections::HashMap;

/// Every node and cell the mesher has produced, on flat arrays.
pub struct Fabric {
    /// Store identity of each node, `None` until it is created.
    pub ids: Vec<NodeId>,
    pub pts: Vec<Point3>,
    /// `false` for a node of the caller's envelope, which never moves.
    pub movable: Vec<bool>,
    pub hexes: Vec<[u32; 8]>,
    pub prisms: Vec<[u32; 6]>,
    kept: Vec<Node>,
}

impl Fabric {
    fn add(&mut self, p: Point3, coords: &Handle<Coords>) -> Result<u32> {
        let node = Node::create_in(coords.clone(), &[p.x, p.y, p.z])?;
        self.ids.push(node.id());
        self.kept.push(node);
        self.pts.push(p);
        self.movable.push(true);
        Ok((self.pts.len() - 1) as u32)
    }
}

/// The surface still facing the void, plus the mesh grown behind it.
pub struct Front {
    pub fab: Fabric,
    /// Facets of the current front, indexing [`Fabric::pts`].
    pub facets: Vec<Facet>,
    coords: Handle<Coords>,
}

/// Fraction of the local room a layer may take. Half lets two fronts
/// approaching head-on meet in the middle rather than through each other.
const ROOM_SHARE: f64 = 0.45;

/// Factor a refused layer's step is multiplied by before being retried.
const RETREAT: f64 = 0.6;

/// How many times a layer is retried, shorter each time.
const RETREAT_STEPS: usize = 6;

/// Two front nodes closer than this many local sizes are candidates to weld.
const WELD_FACTOR: f64 = 0.35;

/// How squarely two front nodes must face each other before they may be
/// seamed: their normals' dot product has to be below `-FACING`.
const FACING: f64 = 0.5;

impl Front {
    /// Start a front from a closed envelope.
    pub fn new(shell: Shell, coords: Handle<Coords>) -> Front {
        let n = shell.points.len();
        Front {
            fab: Fabric {
                ids: shell.nodes,
                pts: shell.points,
                movable: vec![false; n],
                hexes: Vec::new(),
                prisms: Vec::new(),
                kept: Vec::new(),
            },
            facets: shell.facets,
            coords,
        }
    }

    /// The distinct nodes the front currently stands on, in ascending order.
    pub fn ring(&self) -> Vec<u32> {
        let mut r: Vec<u32> = self
            .facets
            .iter()
            .flat_map(|f| f.corners())
            .copied()
            .collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// Outward unit normal of a facet, and its area, by Newell's method.
    pub fn facet_normal(&self, f: &Facet) -> (Vector3, f64) {
        let c = f.corners();
        let mut n = Vector3::zeros();
        for i in 0..c.len() {
            let a = self.fab.pts[c[i] as usize];
            let b = self.fab.pts[c[(i + 1) % c.len()] as usize];
            n.x += (a.y - b.y) * (a.z + b.z);
            n.y += (a.z - b.z) * (a.x + b.x);
            n.z += (a.x - b.x) * (a.y + b.y);
        }
        let len = n.norm();
        if len == 0.0 {
            (Vector3::zeros(), 0.0)
        } else {
            (n / len, len * 0.5)
        }
    }

    /// Signed volume the front encloses, positive when its normals point away
    /// from the void.
    pub fn volume(&self) -> f64 {
        let mut v = 0.0;
        for f in &self.facets {
            let c = f.corners();
            for i in 1..c.len() - 1 {
                let (a, b, d) = (
                    self.fab.pts[c[0] as usize],
                    self.fab.pts[c[i] as usize],
                    self.fab.pts[c[i + 1] as usize],
                );
                v += a.coords.dot(&b.coords.cross(&d.coords));
            }
        }
        v / 6.0
    }

    /// Mean edge length of the current front.
    pub fn mean_edge(&self) -> f64 {
        let (mut total, mut n) = (0.0, 0usize);
        for f in &self.facets {
            let c = f.corners();
            for i in 0..c.len() {
                total += (self.fab.pts[c[(i + 1) % c.len()] as usize]
                    - self.fab.pts[c[i] as usize])
                    .norm();
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            total / n as f64
        }
    }

    /// Offset direction of each ring node for a unit step — the point where
    /// its incident facets meet once each is pushed in by 1.
    ///
    /// Solved in the least-squares sense, because the averaged normal is not
    /// merely inaccurate but can be *tangent* to an incident facet and offset
    /// it by nothing at all. See [`super::shell`] for the derivation; the
    /// solution is linear in the step, so one unit solve serves every node's
    /// own distance.
    pub fn unit_offsets(&self, ring: &[u32]) -> Vec<Vector3> {
        use nalgebra::Matrix3;
        let normals: Vec<(Vector3, f64)> =
            self.facets.iter().map(|f| self.facet_normal(f)).collect();
        let mut incident: HashMap<u32, Vec<usize>> = HashMap::new();
        for (fi, f) in self.facets.iter().enumerate() {
            for &v in f.corners() {
                incident.entry(v).or_default().push(fi);
            }
        }
        ring.iter()
            .map(|v| {
                let Some(faces) = incident.get(v) else {
                    return Vector3::zeros();
                };
                let mut ata = Matrix3::zeros();
                let mut atb = Vector3::zeros();
                let mut mean = Vector3::zeros();
                for &fi in faces {
                    let (n, area) = normals[fi];
                    ata += n * n.transpose() * area;
                    atb -= n * area;
                    mean += n * area;
                }
                if mean.norm() == 0.0 {
                    return Vector3::zeros();
                }
                let trace = ata.trace().max(f64::MIN_POSITIVE);
                ata += Matrix3::identity() * (trace * TANGENT_REGULARISATION);
                let d = ata
                    .try_inverse()
                    .map(|inv| inv * atb)
                    .unwrap_or(-mean.normalize());
                if d.norm() > MAX_OFFSET_RATIO {
                    d.normalize() * MAX_OFFSET_RATIO
                } else {
                    d
                }
            })
            .collect()
    }

    /// How far each ring node may advance before the front would run into
    /// itself, measured as a share of the distance to the nearest facet the
    /// node does not belong to.
    ///
    /// This is what makes a layer local. A global thickness is capped by the
    /// narrowest neck of the whole part; here the neck thins its own layer and
    /// leaves the rest at full depth.
    pub fn room(&self, ring: &[u32]) -> Vec<f64> {
        let grid = FacetGrid::build(self);
        let mut incident: HashMap<u32, Vec<usize>> = HashMap::new();
        for (fi, f) in self.facets.iter().enumerate() {
            for &v in f.corners() {
                incident.entry(v).or_default().push(fi);
            }
        }
        ring.iter()
            .map(|&v| {
                let p = self.fab.pts[v as usize];
                let mine = incident.get(&v);
                let mut best = f64::INFINITY;
                for fi in grid.near(p, grid.reach) {
                    if mine.is_some_and(|m| m.contains(&fi)) {
                        continue;
                    }
                    best = best.min(point_facet_distance(self, p, &self.facets[fi]));
                }
                if best.is_finite() {
                    best * ROOM_SHARE
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }

    /// Place one round of cells, each facet advancing only if it can.
    ///
    /// This is what makes the method a plastering rather than an offset: a
    /// facet that cannot advance — because the cell would come out inside out,
    /// or because it has run out of room — simply stays where it is while its
    /// neighbours go on without it. The step that leaves behind is closed by a
    /// **side wall**, a fresh front quadrangle spanning the old edge and the
    /// new one.
    ///
    /// The side wall's winding is forced, not chosen. If facet `A` advances
    /// and its neighbour `B` does not, the edge `(u, w)` they shared is used
    /// by `B` alone now, in the direction `(w, u)`; something must use it as
    /// `(u, w)` for the front to stay a closed oriented surface, and something
    /// must use the new edge as `(w', u')`. The quadrangle `[u, w, w', u']`
    /// does exactly both.
    ///
    /// Returns `false` when not one facet could move, which is the front
    /// saying it has nowhere left to go.
    pub fn advance(&mut self, step: f64) -> Result<bool> {
        let ring = self.ring();
        let at: HashMap<u32, usize> = ring.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let dir = self.unit_offsets(&ring);
        let room = self.room(&ring);
        let want: Vec<f64> = room.iter().map(|&r| step.min(r)).collect();

        // Retreat locally first: a facet whose cell is inside out pulls its own
        // nodes back, and only what is still impossible afterwards is held.
        let mut scale = vec![1.0f64; ring.len()];
        let mut moving: Vec<bool> = vec![true; self.facets.len()];
        for round in 0..=RETREAT_STEPS {
            let moved: Vec<Point3> = (0..ring.len())
                .map(|i| self.fab.pts[ring[i] as usize] + dir[i] * (want[i] * scale[i]))
                .collect();
            let mut stuck = Vec::new();
            for (fi, f) in self.facets.iter().enumerate() {
                if !moving[fi] {
                    continue;
                }
                let c = f.corners();
                let outer: Vec<Point3> = c.iter().map(|&v| self.fab.pts[v as usize]).collect();
                let inner: Vec<Point3> = c.iter().map(|&v| moved[at[&v]]).collect();
                if !layer_cell_is_valid(&outer, &inner) {
                    stuck.push(fi);
                }
            }
            if stuck.is_empty() {
                return self.commit(&at, &moved, &moving).map(|()| true);
            }
            if round == RETREAT_STEPS {
                // Out of patience: these facets stay put, the rest advance.
                for fi in stuck {
                    moving[fi] = false;
                }
                if !moving.iter().any(|&m| m) {
                    return Ok(false);
                }
                return self.commit(&at, &moved, &moving).map(|()| true);
            }
            for fi in stuck {
                for &v in self.facets[fi].corners() {
                    scale[at[&v]] *= RETREAT;
                }
            }
        }
        Ok(false)
    }

    /// Write the advancing facets' cells into the fabric, move those facets
    /// onto their offsets, and wall in the steps that leaves.
    fn commit(
        &mut self,
        at: &HashMap<u32, usize>,
        moved: &[Point3],
        moving: &[bool],
    ) -> Result<()> {
        // Only the nodes an advancing facet stands on get a new copy; a node
        // used solely by facets staying put does not move at all.
        let mut fresh: HashMap<u32, u32> = HashMap::new();
        for (fi, f) in self.facets.iter().enumerate() {
            if moving[fi] {
                for &v in f.corners() {
                    fresh.entry(v).or_insert(u32::MAX);
                }
            }
        }
        let mut keys: Vec<u32> = fresh.keys().copied().collect();
        keys.sort_unstable();
        for v in keys {
            let id = self.fab.add(moved[at[&v]], &self.coords)?;
            fresh.insert(v, id);
        }

        for (fi, f) in self.facets.iter().enumerate() {
            if !moving[fi] {
                continue;
            }
            match f {
                Facet::Quad(q) => {
                    let n: Vec<u32> = q.iter().map(|v| fresh[v]).collect();
                    self.fab
                        .hexes
                        .push([n[0], n[1], n[2], n[3], q[0], q[1], q[2], q[3]]);
                }
                Facet::Tri(t) => {
                    let n: Vec<u32> = t.iter().map(|v| fresh[v]).collect();
                    self.fab.prisms.push([n[0], n[1], n[2], t[0], t[1], t[2]]);
                }
            }
        }

        // Which directed edges the facets left behind still carry, so a wall
        // goes up exactly where an advancing facet abandoned one.
        let mut held: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        for (fi, f) in self.facets.iter().enumerate() {
            if moving[fi] {
                continue;
            }
            let c = f.corners();
            for i in 0..c.len() {
                held.insert((c[i], c[(i + 1) % c.len()]));
            }
        }
        let mut walls: Vec<Facet> = Vec::new();
        for (fi, f) in self.facets.iter().enumerate() {
            if !moving[fi] {
                continue;
            }
            let c = f.corners();
            for i in 0..c.len() {
                let (u, w) = (c[i], c[(i + 1) % c.len()]);
                // The neighbour across this edge walks it the other way. If it
                // stayed behind, the edge is now unbalanced and needs a wall.
                if held.contains(&(w, u)) {
                    walls.push(Facet::Quad([u, w, fresh[&w], fresh[&u]]));
                }
            }
        }

        for (fi, f) in self.facets.iter_mut().enumerate() {
            if !moving[fi] {
                continue;
            }
            match f {
                Facet::Quad(q) => *q = q.map(|v| fresh[&v]),
                Facet::Tri(t) => *t = t.map(|v| fresh[&v]),
            }
        }
        self.facets.extend(walls);
        Ok(())
    }

    /// Weld front nodes that have come within touching distance of each other.
    ///
    /// This is the seam. Two parts of the front approaching in a thin region
    /// are identified into one, which closes the void there instead of leaving
    /// a sliver no cell can fill. Facets that collapse in the process — their
    /// corners no longer distinct — leave the front: they have no void left in
    /// front of them.
    ///
    /// Returns how many nodes were welded away.
    pub fn weld(&mut self) -> usize {
        let ring = self.ring();
        if ring.len() < 2 {
            return 0;
        }
        let tol = self.mean_edge() * WELD_FACTOR;
        let grid = PointGrid::build(&self.fab.pts, &ring, tol);
        let mut alias: HashMap<u32, u32> = HashMap::new();
        let neighbours = self.front_neighbours();
        let normal = self.node_normals();

        for &v in &ring {
            if alias.contains_key(&v) {
                continue;
            }
            let p = self.fab.pts[v as usize];
            for w in grid.near(p, tol) {
                if w <= v || alias.contains_key(&w) {
                    continue;
                }
                // Two corners of the same facet are supposed to be close —
                // welding them would collapse that facet, which is a
                // degeneracy and not a seam. A seam joins two *different*
                // parts of the front.
                if neighbours.get(&v).is_some_and(|s| s.contains(&w)) {
                    continue;
                }
                if (self.fab.pts[w as usize] - p).norm() >= tol {
                    continue;
                }
                // A seam joins two parts of the front that **face each
                // other**. Two nodes on the same smooth patch are close and
                // point the same way; welding them would fold the surface
                // rather than close a void. Only opposing normals mean the
                // front has come round on itself.
                if normal[&v].dot(&normal[&w]) > -FACING {
                    continue;
                }
                // A node of the caller's envelope keeps its place, so it is
                // always the survivor.
                if self.fab.movable[w as usize] {
                    alias.insert(w, v);
                } else if self.fab.movable[v as usize] {
                    alias.insert(v, w);
                    break;
                }
            }
        }
        if alias.is_empty() {
            return 0;
        }
        let resolve = |mut v: u32| {
            let mut hops = 0;
            while let Some(&next) = alias.get(&v) {
                v = next;
                hops += 1;
                if hops > 8 {
                    break;
                }
            }
            v
        };
        for h in self.fab.hexes.iter_mut() {
            *h = h.map(resolve);
        }
        for p in self.fab.prisms.iter_mut() {
            *p = p.map(resolve);
        }
        for f in self.facets.iter_mut() {
            match f {
                Facet::Quad(q) => *q = q.map(resolve),
                Facet::Tri(t) => *t = t.map(resolve),
            }
        }
        // A facet whose corners are no longer distinct has collapsed onto an
        // edge or a point: there is nothing in front of it any more.
        self.facets.retain(|f| {
            let mut c: Vec<u32> = f.corners().to_vec();
            c.sort_unstable();
            let n = c.len();
            c.dedup();
            c.len() == n
        });
        // And where two facets have been welded onto the *same* nodes, they
        // face each other across nothing: the void closed there. Both go, or
        // the front would carry the same surface twice, once each way round,
        // and no longer say which side is out.
        let mut times: HashMap<Vec<u32>, usize> = HashMap::new();
        for f in &self.facets {
            let mut k = f.corners().to_vec();
            k.sort_unstable();
            *times.entry(k).or_insert(0) += 1;
        }
        self.facets.retain(|f| {
            let mut k = f.corners().to_vec();
            k.sort_unstable();
            times[&k] == 1
        });
        alias.len()
    }

    /// Area-weighted unit normal at each front node.
    fn node_normals(&self) -> HashMap<u32, Vector3> {
        let mut acc: HashMap<u32, Vector3> = HashMap::new();
        for f in &self.facets {
            let (n, area) = self.facet_normal(f);
            for &v in f.corners() {
                *acc.entry(v).or_insert_with(Vector3::zeros) += n * area;
            }
        }
        acc.into_iter()
            .map(|(v, n)| {
                let len = n.norm();
                (v, if len == 0.0 { n } else { n / len })
            })
            .collect()
    }

    /// Which front nodes share a facet — edge neighbours *and* the diagonal
    /// partners across a quadrangle, since welding either pair would collapse
    /// the facet rather than seam anything.
    fn front_neighbours(&self) -> HashMap<u32, Vec<u32>> {
        let mut out: HashMap<u32, Vec<u32>> = HashMap::new();
        for f in &self.facets {
            let c = f.corners();
            for i in 0..c.len() {
                for j in 0..c.len() {
                    if i != j {
                        out.entry(c[i]).or_default().push(c[j]);
                    }
                }
            }
        }
        out
    }
}

/// Diagonal nudge making the normal equations solvable on a flat patch.
const TANGENT_REGULARISATION: f64 = 1e-12;

/// Largest unit offset, at a sharp corner where the offset planes meet far off.
const MAX_OFFSET_RATIO: f64 = 6.0;

/// Is the cell between an outer facet and its offset copy right way out?
///
/// Checked as the sign of the volume of every corner tetrahedron, which is the
/// same test the finite-element Jacobian makes at the Gauss points and catches
/// a cell that is merely twisted as well as one that is inverted.
pub fn layer_cell_is_valid(outer: &[Point3], inner: &[Point3]) -> bool {
    let n = outer.len();
    for i in 0..n {
        let a = inner[i];
        let b = inner[(i + 1) % n];
        let c = inner[(i + n - 1) % n];
        let d = outer[i];
        let v = (b - a).cross(&(c - a)).dot(&(d - a));
        if v <= 0.0 {
            return false;
        }
    }
    true
}

/// Distance from a point to a facet, approximated by its triangles.
fn point_facet_distance(front: &Front, p: Point3, f: &Facet) -> f64 {
    let c = f.corners();
    let mut best = f64::INFINITY;
    for i in 1..c.len() - 1 {
        best = best.min(point_triangle_distance(
            p,
            front.fab.pts[c[0] as usize],
            front.fab.pts[c[i] as usize],
            front.fab.pts[c[i + 1] as usize],
        ));
    }
    best
}

/// Distance from `p` to the triangle `abc`.
fn point_triangle_distance(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let n = (b - a).cross(&(c - a));
    let len = n.norm();
    if len == 0.0 {
        return (p - a).norm();
    }
    let n = n / len;
    let plane = (p - a).dot(&n);
    let q = p - n * plane;
    // Inside the triangle? Then the plane distance is the answer.
    let inside = [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(u, w)| (w - u).cross(&(q - u)).dot(&n) >= 0.0);
    if inside {
        return plane.abs();
    }
    [(a, b), (b, c), (c, a)]
        .iter()
        .map(|&(u, w)| point_segment_distance(p, u, w))
        .fold(f64::INFINITY, f64::min)
}

fn point_segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    if len2 == 0.0 {
        return (p - a).norm();
    }
    let t = ((p - a).dot(&ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm()
}

/// Uniform grid over the front's facets, for "what is near this point".
struct FacetGrid {
    lo: Point3,
    cell: f64,
    dim: [usize; 3],
    buckets: Vec<Vec<usize>>,
    reach: f64,
}

impl FacetGrid {
    fn build(front: &Front) -> FacetGrid {
        let (mut lo, mut hi) = (
            Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for f in &front.facets {
            for &v in f.corners() {
                let p = front.fab.pts[v as usize];
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        let n = front.facets.len().max(1);
        let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let volume = span.iter().map(|s| s.max(1e-300)).product::<f64>();
        let cell = (volume / n as f64)
            .cbrt()
            .max(front.mean_edge())
            .max(1e-300);
        let dim = [
            ((span[0] / cell).ceil() as usize + 1).min(n + 2),
            ((span[1] / cell).ceil() as usize + 1).min(n + 2),
            ((span[2] / cell).ceil() as usize + 1).min(n + 2),
        ];
        let mut grid = FacetGrid {
            lo,
            cell,
            dim,
            buckets: vec![Vec::new(); dim[0] * dim[1] * dim[2]],
            // Two fronts more than a few cells apart cannot collide within a
            // layer, so the search stops there.
            reach: cell * 3.0,
        };
        for (fi, f) in front.facets.iter().enumerate() {
            let mut flo = [usize::MAX; 3];
            let mut fhi = [0usize; 3];
            for &v in f.corners() {
                let p = front.fab.pts[v as usize];
                for k in 0..3 {
                    let i = grid.index(p[k], k);
                    flo[k] = flo[k].min(i);
                    fhi[k] = fhi[k].max(i);
                }
            }
            for z in flo[2]..=fhi[2] {
                for y in flo[1]..=fhi[1] {
                    for x in flo[0]..=fhi[0] {
                        let at = grid.at(x, y, z);
                        grid.buckets[at].push(fi);
                    }
                }
            }
        }
        grid
    }

    fn index(&self, v: f64, k: usize) -> usize {
        (((v - self.lo[k]) / self.cell).floor().max(0.0) as usize).min(self.dim[k] - 1)
    }

    fn at(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dim[1] + y) * self.dim[0] + x
    }

    fn near(&self, p: Point3, radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let lo: Vec<usize> = (0..3).map(|k| self.index(p[k] - radius, k)).collect();
        let hi: Vec<usize> = (0..3).map(|k| self.index(p[k] + radius, k)).collect();
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    out.extend_from_slice(&self.buckets[self.at(x, y, z)]);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Uniform grid over a set of points, for the weld's proximity queries.
struct PointGrid {
    lo: Point3,
    cell: f64,
    dim: [usize; 3],
    buckets: Vec<Vec<u32>>,
}

impl PointGrid {
    fn build(pts: &[Point3], ring: &[u32], cell: f64) -> PointGrid {
        let (mut lo, mut hi) = (
            Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for &v in ring {
            let p = pts[v as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let n = ring.len().max(1);
        let cell = cell.max(1e-300);
        let dim = [
            (((hi[0] - lo[0]) / cell).ceil() as usize + 1).min(n + 2),
            (((hi[1] - lo[1]) / cell).ceil() as usize + 1).min(n + 2),
            (((hi[2] - lo[2]) / cell).ceil() as usize + 1).min(n + 2),
        ];
        let mut grid = PointGrid {
            lo,
            cell,
            dim,
            buckets: vec![Vec::new(); dim[0] * dim[1] * dim[2]],
        };
        for &v in ring {
            let p = pts[v as usize];
            let i = grid.at(p);
            grid.buckets[i].push(v);
        }
        grid
    }

    fn index(&self, v: f64, k: usize) -> usize {
        (((v - self.lo[k]) / self.cell).floor().max(0.0) as usize).min(self.dim[k] - 1)
    }

    fn at(&self, p: Point3) -> usize {
        let (x, y, z) = (
            self.index(p[0], 0),
            self.index(p[1], 1),
            self.index(p[2], 2),
        );
        (z * self.dim[1] + y) * self.dim[0] + x
    }

    fn near(&self, p: Point3, radius: f64) -> Vec<u32> {
        let mut out = Vec::new();
        let lo: Vec<usize> = (0..3).map(|k| self.index(p[k] - radius, k)).collect();
        let hi: Vec<usize> = (0..3).map(|k| self.index(p[k] + radius, k)).collect();
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let i = (z * self.dim[1] + y) * self.dim[0] + x;
                    out.extend_from_slice(&self.buckets[i]);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// A closed box of six QUA4 facets, normals outward.
    fn box_front(hi: [f64; 3]) -> Front {
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<NodeId> = (0..8)
            .map(|i| {
                let p = [
                    if i & 1 == 0 { 0.0 } else { hi[0] },
                    if i & 2 == 0 { 0.0 } else { hi[1] },
                    if i & 4 == 0 { 0.0 } else { hi[2] },
                ];
                Node::create_in(coords.clone(), &p).unwrap().id()
            })
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        for f in [
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 2, 3, 1],
            [4, 5, 7, 6],
        ] {
            sm.add_cell(&[n[f[0]], n[f[1]], n[f[2]], n[f[3]]]).unwrap();
        }
        let shell = Shell::extract(&Mesh::from_submesh(sm), "test").unwrap();
        Front::new(shell, coords)
    }

    #[test]
    fn the_unit_offset_at_a_box_corner_is_the_diagonal() {
        // The point of solving rather than averaging: a step along the
        // averaged normal would offset each face by only 1/√3.
        let f = box_front([1.0, 1.0, 1.0]);
        let ring = f.ring();
        let d = f.unit_offsets(&ring);
        for (i, &v) in ring.iter().enumerate() {
            let p = f.fab.pts[v as usize];
            for k in 0..3 {
                let want = if p[k] == 0.0 { 1.0 } else { -1.0 };
                assert!(
                    (d[i][k] - want).abs() < 1e-9,
                    "node {v} axis {k}: {} wanted {want}",
                    d[i][k]
                );
            }
        }
    }

    #[test]
    fn the_unit_offset_moves_every_incident_facet_by_one() {
        // A tetrahedron's corner is the case the averaged normal cannot do at
        // all: it lies *in* one of the incident faces, which a step along it
        // then fails to offset by anything.
        let coords = Handle::new(Coords::new(3).unwrap());
        let n: Vec<NodeId> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        for f in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]] {
            sm.add_cell(&[n[f[0]], n[f[1]], n[f[2]]]).unwrap();
        }
        let shell = Shell::extract(&Mesh::from_submesh(sm), "test").unwrap();
        let front = Front::new(shell, coords);
        let ring = front.ring();
        let at: HashMap<u32, usize> = ring.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let d = front.unit_offsets(&ring);
        for f in &front.facets {
            let (normal, _) = front.facet_normal(f);
            for &v in f.corners() {
                let moved = -d[at[&v]].dot(&normal);
                assert!(moved >= 1.0 - 1e-9, "facet moved {moved} at node {v}");
            }
        }
    }

    #[test]
    fn the_room_is_half_the_way_across_a_thin_slab() {
        // A slab 0.2 thick: a node on one face has the opposite face 0.2 away,
        // so it may take a share of that and no more. This is what keeps a
        // layer from being capped by the thinnest part of the whole solid.
        let f = box_front([4.0, 0.2, 4.0]);
        let ring = f.ring();
        let room = f.room(&ring);
        for (i, &v) in ring.iter().enumerate() {
            assert!(
                room[i] <= 0.2 * ROOM_SHARE + 1e-9,
                "node {v} thinks it has {} of room across a 0.2 slab",
                room[i]
            );
        }
    }

    /// A slab `n × n` quadrangles across and one thick — subdivided, so that
    /// a node on the top face and the one below it do **not** share a facet
    /// and can therefore seam.
    fn slab_front(n: usize, thickness: f64) -> Front {
        let coords = Handle::new(Coords::new(3).unwrap());
        let w = n + 1;
        let at = |i: usize, j: usize, top: bool| (if top { w * w } else { 0 }) + j * w + i;
        let mut ids = Vec::new();
        for top in [false, true] {
            for j in 0..w {
                for i in 0..w {
                    let p = [
                        i as f64 / n as f64,
                        if top { thickness } else { 0.0 },
                        j as f64 / n as f64,
                    ];
                    ids.push(Node::create_in(coords.clone(), &p).unwrap().id());
                }
            }
        }
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        let mut quad = |a: usize, b: usize, c: usize, d: usize| {
            sm.add_cell(&[ids[a], ids[b], ids[c], ids[d]]).unwrap();
        };
        for j in 0..n {
            for i in 0..n {
                // Bottom faces down (outward), top faces up.
                quad(
                    at(i, j, false),
                    at(i + 1, j, false),
                    at(i + 1, j + 1, false),
                    at(i, j + 1, false),
                );
                quad(
                    at(i, j, true),
                    at(i, j + 1, true),
                    at(i + 1, j + 1, true),
                    at(i + 1, j, true),
                );
            }
        }
        for i in 0..n {
            quad(
                at(i, 0, false),
                at(i, 0, true),
                at(i + 1, 0, true),
                at(i + 1, 0, false),
            );
            quad(
                at(i + 1, n, false),
                at(i + 1, n, true),
                at(i, n, true),
                at(i, n, false),
            );
            quad(
                at(0, i + 1, false),
                at(0, i + 1, true),
                at(0, i, true),
                at(0, i, false),
            );
            quad(
                at(n, i, false),
                at(n, i, true),
                at(n, i + 1, true),
                at(n, i + 1, false),
            );
        }
        let shell = Shell::extract(&Mesh::from_submesh(sm), "test").unwrap();
        Front::new(shell, coords)
    }

    #[test]
    fn a_thin_slab_seams_shut_instead_of_leaving_a_sliver() {
        // Ask for a layer twenty times thicker than the slab. Each node's step
        // is clamped by the room it has, the two faces meet in the middle, and
        // the seam welds them: the void closes instead of becoming a sliver no
        // cell could fill.
        let mut f = slab_front(4, 0.05);
        let facets = f.facets.len();
        assert!(f.advance(1.0).unwrap());
        let welded = f.weld();
        assert!(welded > 0, "the two faces should have welded together");
        assert!(
            f.facets.len() < facets,
            "seaming should retire the facets it closed: {facets} → {}",
            f.facets.len()
        );
        // What welded is gone from the front for good.
        let ring = f.ring();
        for &v in &ring {
            let p = f.fab.pts[v as usize];
            assert!(p.y.is_finite());
        }
    }

    #[test]
    fn holding_a_facet_back_raises_a_wall_and_keeps_the_front_closed() {
        // Drive `commit` directly with a mask that holds one facet: this is
        // the mechanism the front relies on when a cell cannot be laid, and it
        // is worth testing on its own because the room cap keeps it from
        // firing on well-proportioned parts.
        let mut f = box_front([1.0, 1.0, 1.0]);
        let ring = f.ring();
        let at: HashMap<u32, usize> = ring.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let dir = f.unit_offsets(&ring);
        let moved: Vec<Point3> = (0..ring.len())
            .map(|i| f.fab.pts[ring[i] as usize] + dir[i] * 0.2)
            .collect();

        let mut moving = vec![true; f.facets.len()];
        moving[0] = false; // one face stays behind
        f.commit(&at, &moved, &moving).unwrap();

        // Five cells, not six, and four walls closing the step round the face
        // that stayed.
        assert_eq!(f.fab.hexes.len(), 5);
        assert_eq!(f.facets.len(), 6 + 4, "four walls should have gone up");
        // And the front is still a closed oriented surface, which is the whole
        // point of the walls.
        let mut balance: HashMap<(u32, u32), i32> = HashMap::new();
        for facet in &f.facets {
            let c = facet.corners();
            for i in 0..c.len() {
                let (a, b) = (c[i], c[(i + 1) % c.len()]);
                let (k, d) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                *balance.entry(k).or_insert(0) += d;
            }
        }
        assert!(
            balance.values().all(|&v| v == 0),
            "the walls failed to balance the front's edges"
        );
    }

    #[test]
    fn the_front_stays_a_closed_oriented_surface_while_it_advances() {
        // The invariant the side walls exist to preserve: every edge is used
        // exactly twice, once in each direction. Break it and the front no
        // longer says which side is out.
        let mut f = slab_front(4, 0.3);
        for _ in 0..3 {
            f.advance(0.5).unwrap();
            let mut balance: HashMap<(u32, u32), i32> = HashMap::new();
            for facet in &f.facets {
                let c = facet.corners();
                for i in 0..c.len() {
                    let (a, b) = (c[i], c[(i + 1) % c.len()]);
                    let (k, d) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                    *balance.entry(k).or_insert(0) += d;
                }
            }
            assert!(
                balance.values().all(|&v| v == 0),
                "the front stopped being a closed oriented surface"
            );
        }
    }

    #[test]
    fn advancing_leaves_one_cell_per_facet_and_all_right_way_out() {
        let mut f = box_front([1.0, 1.0, 1.0]);
        assert!(f.advance(0.2).unwrap());
        assert_eq!(f.fab.hexes.len(), 6);
        let patch = super::super::smooth::Patch {
            hexes: &f.fab.hexes,
            prisms: &f.fab.prisms,
            movable: &f.fab.movable,
        };
        assert!(
            super::super::smooth::worst_quality(&f.fab.pts, &patch) > 0.0,
            "a cell came out inside out"
        );
    }
}
