//! Bowyer-Watson Delaunay triangulation in 2-D.
//!
//! The public entry point [`delaunay_2d`] takes a flat slice of
//! [`Point2`] and returns the Delaunay triangulation as triples of
//! indices into the input. Triangles are oriented CCW.
//!
//! The algorithm is the textbook Bowyer-Watson incremental insertion
//! with a single bounding "super-triangle":
//!
//! 1. Build a super-triangle large enough to contain all input points.
//! 2. Insert each input point one at a time:
//!    - find every triangle whose circumcircle contains the new point
//!      (the "bad" triangles),
//!    - remove them — the union of their cells forms a star-shaped
//!      polygon around the new point,
//!    - re-triangulate the cavity by connecting every boundary edge to
//!      the new point.
//! 3. Drop every triangle that still references a super-triangle vertex.
//!
//! For simplicity (cast3m philosophy: clear over clever), the
//! neighbour topology is rebuilt globally after each insertion. This
//! gives O(N²) total complexity which is fine for the contour sizes
//! we expect (a few hundred to a few thousand points). A future
//! optimisation can swap this for local neighbour patching without
//! changing the public API.
//!
//! Constraint enforcement (edges that must appear in the final mesh)
//! and hole removal will be layered on top in subsequent commits.

use crate::error::{PyrucastError, Result};
use crate::containers::mesh::{Point2, Vector2};
use super::{cross2, point_in_triangle};
use std::collections::{HashMap, HashSet};

/// Refinement criteria applied after the constrained Delaunay
/// triangulation is built.
///
/// Both options are independent and additive. When both are set, a
/// triangle is "bad" if it violates **either** criterion.
///
/// # Termination
/// Ruppert's algorithm is **only proven to terminate** for
/// `min_angle_deg ≤ 20.7`. Higher thresholds usually work in practice
/// but may diverge on pathological inputs; pyrucast guards against this
/// by capping the total number of inserted Steiner points and returning
/// a clear error if the cap is hit.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefinementOptions {
    /// Maximum allowed length for any triangle edge. Triangles having an
    /// edge longer than this are split by inserting their circumcenter
    /// (or the midpoint of an encroached constrained edge — see Ruppert).
    pub max_edge_length: Option<f64>,
    /// Minimum allowed triangle angle, in degrees. Equivalent to a
    /// circumradius-to-shortest-edge ratio bound:
    /// `r / L_min ≤ 1 / (2 sin α)`.
    pub min_angle_deg: Option<f64>,
}

impl RefinementOptions {
    /// True iff the options request any refinement work.
    pub fn is_active(&self) -> bool {
        self.max_edge_length.is_some() || self.min_angle_deg.is_some()
    }
}

/// Sentinel for "no neighbour" in the topology arrays.
const NO_NEIGHBOUR: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
struct Triangle {
    /// Vertex indices into [`Cdt::points`], in CCW order.
    v: [usize; 3],
    /// Neighbours: `n[k]` is the triangle sharing the edge **opposite**
    /// to vertex `v[k]` (i.e. the edge `v[(k+1)%3] → v[(k+2)%3]`).
    /// [`NO_NEIGHBOUR`] for a hull edge.
    n: [usize; 3],
    /// `false` once the triangle is logically removed; the slot is kept
    /// to preserve stable indices but ignored by iteration.
    alive: bool,
}

pub(super) struct Cdt {
    /// All vertices: the first `n_input` are user-supplied; the last 3
    /// are the super-triangle.
    points: Vec<Point2>,
    n_input: usize,
    triangles: Vec<Triangle>,
}

impl Cdt {
    /// Create a CDT initialised with a super-triangle wrapping `input_points`.
    pub(super) fn new(input_points: &[Point2]) -> Self {
        let n_input = input_points.len();
        let mut points: Vec<Point2> = Vec::with_capacity(n_input + 3);
        points.extend_from_slice(input_points);

        let [sa, sb, sc] = super_triangle(input_points);
        points.push(sa);
        points.push(sb);
        points.push(sc);
        let triangles = vec![Triangle {
            v: [n_input, n_input + 1, n_input + 2],
            n: [NO_NEIGHBOUR; 3],
            alive: true,
        }];

        Self {
            points,
            n_input,
            triangles,
        }
    }

    /// Insert the input point of index `p_idx` (in `0..n_input`) using
    /// Bowyer-Watson. Errors only if the bad-triangle search comes up
    /// empty, which would mean the super-triangle does not actually
    /// contain `p` — a bug, not a user error.
    pub(super) fn insert_point(&mut self, p_idx: usize) -> Result<()> {
        let p = self.points[p_idx];

        // 1. Collect every alive triangle whose circumcircle contains p.
        let mut bad: Vec<usize> = Vec::new();
        for (t_idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                continue;
            }
            let a = self.points[t.v[0]];
            let b = self.points[t.v[1]];
            let c = self.points[t.v[2]];
            if in_circle(a, b, c, p) > 0.0 {
                bad.push(t_idx);
            }
        }
        if bad.is_empty() {
            return Err(PyrucastError::Message(format!(
                "cdt::insert_point: no bad triangle for point #{} — super-triangle too small?",
                p_idx
            )));
        }

        // 2. Boundary of the cavity = edges of bad triangles whose opposite
        //    neighbour is NOT itself bad.
        let bad_set: std::collections::HashSet<usize> = bad.iter().copied().collect();
        let mut cavity_edges: Vec<(usize, usize)> = Vec::new();
        for &t_idx in &bad {
            let t = self.triangles[t_idx];
            for k in 0..3 {
                let nb = t.n[k];
                let is_outside = nb == NO_NEIGHBOUR || !bad_set.contains(&nb);
                if is_outside {
                    // Edge opposite to vertex k is (v[(k+1)%3], v[(k+2)%3]) in CCW order.
                    cavity_edges.push((t.v[(k + 1) % 3], t.v[(k + 2) % 3]));
                }
            }
        }

        // 3. Retire the bad triangles.
        for &t_idx in &bad {
            self.triangles[t_idx].alive = false;
        }

        // 4. For each cavity edge (a, b) — already CCW from the dead triangle —
        //    create a new triangle (a, b, p). It is automatically CCW because
        //    p lies on the same side of (a, b) as the dead triangle's third vertex.
        for (a, b) in &cavity_edges {
            self.triangles.push(Triangle {
                v: [*a, *b, p_idx],
                n: [NO_NEIGHBOUR; 3],
                alive: true,
            });
        }

        // 5. Rebuild neighbour topology globally. O(N) per insertion; this
        //    keeps the bookkeeping local and easy to audit. The total cost is
        //    O(N²), which is fine for the contour sizes pyrucast targets.
        self.rebuild_neighbours();
        Ok(())
    }

    /// Recompute every triangle's neighbour array from its connectivity.
    fn rebuild_neighbours(&mut self) {
        // edge (min, max) → list of (triangle_idx, opposite_vertex_local_idx)
        let mut edge_map: HashMap<(usize, usize), [(usize, usize); 2]> = HashMap::new();
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

        for (t_idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                continue;
            }
            for k in 0..3 {
                let a = t.v[(k + 1) % 3];
                let b = t.v[(k + 2) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                let count = edge_count.entry(key).or_insert(0);
                let entry = edge_map.entry(key).or_insert([(NO_NEIGHBOUR, 0); 2]);
                if *count < 2 {
                    entry[*count] = (t_idx, k);
                }
                *count += 1;
            }
        }

        for t_idx in 0..self.triangles.len() {
            if !self.triangles[t_idx].alive {
                continue;
            }
            for k in 0..3 {
                let a = self.triangles[t_idx].v[(k + 1) % 3];
                let b = self.triangles[t_idx].v[(k + 2) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = &edge_map[&key];
                let count = edge_count[&key];
                let neighbour = if count != 2 {
                    NO_NEIGHBOUR
                } else if entry[0].0 == t_idx {
                    entry[1].0
                } else {
                    entry[0].0
                };
                self.triangles[t_idx].n[k] = neighbour;
            }
        }
    }

    /// Force the edge `(a, b)` to appear in the triangulation.
    ///
    /// Both indices must refer to input points already inserted. If the
    /// edge already exists, this is a no-op. Otherwise the function:
    /// 1. walks the triangles that the segment `(a, b)` crosses,
    /// 2. retires them,
    /// 3. rebuilds two polygons (one each side of the new edge),
    /// 4. re-triangulates each polygon by ear clipping.
    ///
    /// The Delaunay empty-circle property is **not** preserved across
    /// the constrained edge — by definition of a CDT.
    pub(super) fn insert_constraint(&mut self, a: usize, b: usize) -> Result<()> {
        if a == b {
            return Err(PyrucastError::Message(format!(
                "cdt::insert_constraint: degenerate edge ({}, {})",
                a, b
            )));
        }
        if self.edge_exists(a, b) {
            return Ok(());
        }

        let crossed = self.triangles_crossing_segment(a, b)?;
        let crossed_set: std::collections::HashSet<usize> = crossed.iter().copied().collect();

        // External boundary of the cavity = edges of crossed triangles whose
        // opposite neighbour is **not** itself crossed.
        let mut external_edges: Vec<(usize, usize)> = Vec::new();
        for &t_idx in &crossed {
            let t = self.triangles[t_idx];
            for k in 0..3 {
                let nb = t.n[k];
                let is_outside = nb == NO_NEIGHBOUR || !crossed_set.contains(&nb);
                if is_outside {
                    let p = t.v[(k + 1) % 3];
                    let q = t.v[(k + 2) % 3];
                    external_edges.push((p, q));
                }
            }
        }

        for &t_idx in &crossed {
            self.triangles[t_idx].alive = false;
        }

        // Split external edges into "left" and "right" of the oriented line a→b.
        let pa = self.points[a];
        let pb = self.points[b];
        let mut left_edges: Vec<(usize, usize)> = Vec::new();
        let mut right_edges: Vec<(usize, usize)> = Vec::new();
        for &(u, v) in &external_edges {
            let pu = self.points[u];
            let pv = self.points[v];
            let mid = Point2::from((pu.coords + pv.coords) * 0.5);
            let side = orient2d(pa, pb, mid);
            if side > 0.0 {
                left_edges.push((u, v));
            } else if side < 0.0 {
                right_edges.push((u, v));
            } else {
                return Err(PyrucastError::Message(format!(
                    "cdt::insert_constraint: edge ({}, {}) is collinear with constraint a={} b={}",
                    u, v, a, b
                )));
            }
        }

        let left_chain = build_chain(&left_edges, a, b)?;
        let right_chain = build_chain(&right_edges, b, a)?;

        for chain in [left_chain, right_chain] {
            if chain.len() < 3 {
                return Err(PyrucastError::Message(format!(
                    "cdt::insert_constraint: side polygon has only {} vertices",
                    chain.len()
                )));
            }
            let pts: Vec<Point2> = chain.iter().map(|&v| self.points[v]).collect();
            let tris = crate::ops::mesher::triangulation::ear_clip_2d(&pts)?;
            for [i, j, k] in tris {
                self.triangles.push(Triangle {
                    v: [chain[i], chain[j], chain[k]],
                    n: [NO_NEIGHBOUR; 3],
                    alive: true,
                });
            }
        }

        self.rebuild_neighbours();
        Ok(())
    }

    /// True iff `(a, b)` is an edge of some alive triangle.
    fn edge_exists(&self, a: usize, b: usize) -> bool {
        for t in &self.triangles {
            if !t.alive {
                continue;
            }
            for k in 0..3 {
                let (i, j) = (t.v[k], t.v[(k + 1) % 3]);
                if (i == a && j == b) || (i == b && j == a) {
                    return true;
                }
            }
        }
        false
    }

    /// Walk the triangles strictly crossed by the open segment `(a, b)`,
    /// in order from `a` to `b`.
    fn triangles_crossing_segment(&self, a: usize, b: usize) -> Result<Vec<usize>> {
        let pa = self.points[a];
        let pb = self.points[b];

        // Find a starting triangle: one with `a` as a vertex whose
        // opposite edge is strictly crossed by (a, b).
        let mut start: Option<usize> = None;
        for (t_idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                continue;
            }
            let a_pos = match t.v.iter().position(|&v| v == a) {
                Some(p) => p,
                None => continue,
            };
            let p = t.v[(a_pos + 1) % 3];
            let q = t.v[(a_pos + 2) % 3];
            let pp = self.points[p];
            let pq = self.points[q];
            if segments_cross_strict(pa, pb, pp, pq) {
                start = Some(t_idx);
                break;
            }
        }
        let mut current = start.ok_or_else(|| {
            PyrucastError::Message(format!(
                "cdt::insert_constraint: cannot start walk from vertex {}",
                a
            ))
        })?;
        let mut crossed = vec![current];

        loop {
            if self.triangles[current].v.contains(&b) {
                break;
            }
            let t = self.triangles[current];
            let mut moved = false;
            for k in 0..3 {
                let p = t.v[(k + 1) % 3];
                let q = t.v[(k + 2) % 3];
                // Skip the edge opposite to vertex `a` (we came in through there).
                if p == a || q == a {
                    continue;
                }
                let pp = self.points[p];
                let pq = self.points[q];
                if segments_cross_strict(pa, pb, pp, pq) {
                    let nb = t.n[k];
                    if nb == NO_NEIGHBOUR {
                        return Err(PyrucastError::Message(format!(
                            "cdt::insert_constraint: walk fell off the hull (a={}, b={})",
                            a, b
                        )));
                    }
                    current = nb;
                    crossed.push(current);
                    moved = true;
                    break;
                }
            }
            if !moved {
                return Err(PyrucastError::Message(format!(
                    "cdt::insert_constraint: walk got stuck (a={}, b={})",
                    a, b
                )));
            }
            if crossed.len() > self.triangles.len() {
                return Err(PyrucastError::Message(format!(
                    "cdt::insert_constraint: walk did not terminate (a={}, b={})",
                    a, b
                )));
            }
        }
        Ok(crossed)
    }

    /// Insert a point with the **constrained** Bowyer-Watson variant:
    /// the cavity is grown by BFS from the triangle containing `p`,
    /// never crossing a constrained edge. This keeps every forced edge
    /// of the CDT intact even when the new point would otherwise pull
    /// triangles across it.
    ///
    /// `p_idx` must point at an entry of `self.points` that the caller
    /// already pushed; the function does not allocate. Returns an error
    /// if the seed triangle (the one that contains the point) cannot be
    /// found — typically because `p` lies on a constrained edge.
    fn insert_point_constrained(
        &mut self,
        p_idx: usize,
        constrained_edges: &HashSet<(usize, usize)>,
    ) -> Result<()> {
        let p = self.points[p_idx];

        // 1. Locate a triangle containing `p`.
        let mut seed: Option<usize> = None;
        for (idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                continue;
            }
            let a = self.points[t.v[0]];
            let b = self.points[t.v[1]];
            let c = self.points[t.v[2]];
            if point_in_triangle(p, a, b, c) {
                seed = Some(idx);
                break;
            }
        }
        let seed = seed.ok_or_else(|| {
            PyrucastError::Message(format!(
                "cdt::insert_point_constrained: no triangle contains point #{}",
                p_idx
            ))
        })?;

        // 2. BFS from `seed`, only crossing non-constrained edges and only
        //    when in_circle > 0.
        let mut bad: Vec<usize> = Vec::new();
        let mut visited = vec![false; self.triangles.len()];
        let mut queue = vec![seed];
        visited[seed] = true;
        while let Some(t_idx) = queue.pop() {
            let t = self.triangles[t_idx];
            if !t.alive {
                continue;
            }
            let a = self.points[t.v[0]];
            let b = self.points[t.v[1]];
            let c = self.points[t.v[2]];
            if in_circle(a, b, c, p) <= 0.0 {
                continue;
            }
            bad.push(t_idx);
            for k in 0..3 {
                let nb = t.n[k];
                if nb == NO_NEIGHBOUR || visited[nb] {
                    continue;
                }
                let va = t.v[(k + 1) % 3];
                let vb = t.v[(k + 2) % 3];
                let key = if va < vb { (va, vb) } else { (vb, va) };
                if constrained_edges.contains(&key) {
                    continue;
                }
                visited[nb] = true;
                queue.push(nb);
            }
        }

        if bad.is_empty() {
            return Err(PyrucastError::Message(format!(
                "cdt::insert_point_constrained: point #{} did not produce a cavity",
                p_idx
            )));
        }

        // 3. Cavity boundary: edges of bad triangles whose opposite
        //    neighbour is not itself bad.
        let bad_set: HashSet<usize> = bad.iter().copied().collect();
        let mut cavity_edges: Vec<(usize, usize)> = Vec::new();
        for &t_idx in &bad {
            let t = self.triangles[t_idx];
            for k in 0..3 {
                let nb = t.n[k];
                if nb == NO_NEIGHBOUR || !bad_set.contains(&nb) {
                    cavity_edges.push((t.v[(k + 1) % 3], t.v[(k + 2) % 3]));
                }
            }
        }

        for &t_idx in &bad {
            self.triangles[t_idx].alive = false;
        }
        for (a, b) in cavity_edges {
            self.triangles.push(Triangle {
                v: [a, b, p_idx],
                n: [NO_NEIGHBOUR; 3],
                alive: true,
            });
        }
        self.rebuild_neighbours();
        Ok(())
    }

    /// Circumcenter of the triangle at index `t_idx`, or `None` if it
    /// is degenerate (collinear vertices).
    fn triangle_circumcenter(&self, t_idx: usize) -> Option<Point2> {
        let t = self.triangles[t_idx];
        let a = self.points[t.v[0]];
        let b = self.points[t.v[1]];
        let c = self.points[t.v[2]];
        circumcenter(a, b, c)
    }

    /// Squared length of the longest edge of triangle `t_idx`.
    fn triangle_longest_edge_sq(&self, t_idx: usize) -> f64 {
        let t = self.triangles[t_idx];
        let a = self.points[t.v[0]];
        let b = self.points[t.v[1]];
        let c = self.points[t.v[2]];
        let ab = (b - a).norm_squared();
        let bc = (c - b).norm_squared();
        let ca = (a - c).norm_squared();
        ab.max(bc).max(ca)
    }

    /// Squared length of the shortest edge of triangle `t_idx`.
    fn triangle_shortest_edge_sq(&self, t_idx: usize) -> f64 {
        let t = self.triangles[t_idx];
        let a = self.points[t.v[0]];
        let b = self.points[t.v[1]];
        let c = self.points[t.v[2]];
        let ab = (b - a).norm_squared();
        let bc = (c - b).norm_squared();
        let ca = (a - c).norm_squared();
        ab.min(bc).min(ca)
    }

    /// Squared circumradius of triangle `t_idx`, or `f64::INFINITY` if
    /// the triangle is degenerate.
    fn triangle_circumradius_sq(&self, t_idx: usize) -> f64 {
        let Some(c) = self.triangle_circumcenter(t_idx) else {
            return f64::INFINITY;
        };
        let t = self.triangles[t_idx];
        (self.points[t.v[0]] - c).norm_squared()
    }

    /// Find one interior triangle that violates the given criteria.
    /// `outside[i] = true` for triangles to skip (super-triangle, holes,
    /// outer exterior, dead).
    ///
    /// - `max_edge_sq`: triangles with longest edge² > this are bad,
    /// - `radius_ratio_sq_threshold`: triangles with
    ///   `circumradius² / shortest_edge² > this` are skinny.
    fn find_bad_interior_triangle(
        &self,
        outside: &[bool],
        max_edge_sq: Option<f64>,
        radius_ratio_sq_threshold: Option<f64>,
    ) -> Option<usize> {
        for (idx, t) in self.triangles.iter().enumerate() {
            if !t.alive || outside[idx] {
                continue;
            }
            if let Some(max_sq) = max_edge_sq {
                if self.triangle_longest_edge_sq(idx) > max_sq {
                    return Some(idx);
                }
            }
            if let Some(threshold) = radius_ratio_sq_threshold {
                let r2 = self.triangle_circumradius_sq(idx);
                let l2 = self.triangle_shortest_edge_sq(idx);
                if l2 > 0.0 && r2 / l2 > threshold {
                    return Some(idx);
                }
            }
        }
        None
    }

    /// First constrained edge whose diametral disk strictly contains a
    /// point of `self.points` other than its two endpoints and the
    /// three super-triangle vertices.
    fn first_encroached_constraint(
        &self,
        constrained_edges: &HashSet<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        for &(a, b) in constrained_edges {
            if self.constraint_has_encroaching_point(a, b, None) {
                return Some((a, b));
            }
        }
        None
    }

    /// True iff some point of `self.points` strictly lies in the
    /// diametral disk of the constrained edge `(a, b)`. `extra_point`
    /// is also tested if supplied (used to check whether the
    /// circumcenter we are about to insert encroaches).
    fn constraint_has_encroaching_point(
        &self,
        a: usize,
        b: usize,
        extra_point: Option<Point2>,
    ) -> bool {
        let pa = self.points[a];
        let pb = self.points[b];
        let mid = Point2::from((pa.coords + pb.coords) * 0.5);
        let r2 = (pb - pa).norm_squared() * 0.25;
        let strict = 1e-12;
        for (i, &p) in self.points.iter().enumerate() {
            if i == a || i == b {
                continue;
            }
            if i >= self.n_input && i < self.n_input + 3 {
                continue; // super-triangle vertex
            }
            if (p - mid).norm_squared() < r2 - strict {
                return true;
            }
        }
        if let Some(p) = extra_point {
            if (p - mid).norm_squared() < r2 - strict {
                return true;
            }
        }
        false
    }

    /// If `p` lies strictly in the diametral disk of any constrained
    /// edge, return one such edge. Used by Ruppert's algorithm before
    /// inserting a circumcenter.
    fn encroached_constraint_by(
        &self,
        p: Point2,
        constrained_edges: &HashSet<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        let strict = 1e-12;
        for &(a, b) in constrained_edges {
            let pa = self.points[a];
            let pb = self.points[b];
            let mid = Point2::from((pa.coords + pb.coords) * 0.5);
            let r2 = (pb - pa).norm_squared() * 0.25;
            if (p - mid).norm_squared() < r2 - strict {
                return Some((a, b));
            }
        }
        None
    }

    /// Split the constrained edge `(a, b)` at its midpoint: the midpoint
    /// is added to `self.points`, inserted as a constrained Steiner
    /// point, and `constrained_edges` is updated to replace `(a, b)`
    /// with `(a, m)` and `(m, b)`. Returns the new point index `m`.
    fn split_constraint(
        &mut self,
        a: usize,
        b: usize,
        constrained_edges: &mut HashSet<(usize, usize)>,
    ) -> Result<usize> {
        let pa = self.points[a];
        let pb = self.points[b];
        let mid = Point2::from((pa.coords + pb.coords) * 0.5);
        let m_idx = self.points.len();
        self.points.push(mid);

        // Remove the original constraint *before* the insertion so that
        // the BFS can grow the cavity across it.
        let key_ab = if a < b { (a, b) } else { (b, a) };
        constrained_edges.remove(&key_ab);

        self.insert_point_constrained(m_idx, constrained_edges)?;

        // The two halves are the new constraints.
        let key_am = if a < m_idx { (a, m_idx) } else { (m_idx, a) };
        let key_mb = if m_idx < b { (m_idx, b) } else { (b, m_idx) };
        constrained_edges.insert(key_am);
        constrained_edges.insert(key_mb);
        Ok(m_idx)
    }

    /// Ruppert-style refinement loop.
    ///
    /// Repeats two steps until neither applies:
    /// 1. Split every constrained edge that has a vertex inside its
    ///    diametral disk (an *encroachment*).
    /// 2. Pick a bad interior triangle (size or angle criterion) and
    ///    try to insert its circumcenter — unless that circumcenter
    ///    encroaches a constrained edge, in which case split the
    ///    constraint instead.
    ///
    /// Total Steiner-point insertions are capped at
    /// `50 × initial_input_count + 1000` as a divergence guard.
    pub(super) fn refine(
        &mut self,
        opts: &RefinementOptions,
        constrained_edges: &mut HashSet<(usize, usize)>,
    ) -> Result<()> {
        if !opts.is_active() {
            return Ok(());
        }
        let max_edge_sq = opts.max_edge_length.map(|h| h * h);
        // Skinniness threshold expressed as (circumradius / shortest_edge)².
        let radius_ratio_sq_threshold = opts.min_angle_deg.map(|deg| {
            let s = deg.to_radians().sin();
            let r = 1.0 / (2.0 * s);
            r * r
        });

        let max_inserts = self.n_input * 50 + 1000;
        let initial_points = self.points.len();

        loop {
            if self.points.len() >= initial_points + max_inserts {
                return Err(PyrucastError::Message(format!(
                    "cdt::refine: did not converge after {} Steiner insertions \
                     (max_edge_length={:?}, min_angle_deg={:?}); criteria may be too tight",
                    max_inserts, opts.max_edge_length, opts.min_angle_deg
                )));
            }

            // 1. Encroached constraint?
            if let Some((a, b)) = self.first_encroached_constraint(constrained_edges) {
                self.split_constraint(a, b, constrained_edges)?;
                continue;
            }

            // 2. Find a bad interior triangle.
            let outside = self.flood_fill_outside(constrained_edges);
            let bad = self.find_bad_interior_triangle(
                &outside,
                max_edge_sq,
                radius_ratio_sq_threshold,
            );
            let Some(t_idx) = bad else {
                return Ok(());
            };

            let Some(cc) = self.triangle_circumcenter(t_idx) else {
                return Err(PyrucastError::Message(format!(
                    "cdt::refine: triangle {} has no circumcenter",
                    t_idx
                )));
            };

            // If the circumcenter would encroach a constraint, split
            // that constraint instead.
            if let Some((a, b)) = self.encroached_constraint_by(cc, constrained_edges) {
                self.split_constraint(a, b, constrained_edges)?;
                continue;
            }

            let new_idx = self.points.len();
            self.points.push(cc);
            self.insert_point_constrained(new_idx, constrained_edges)?;
        }
    }

    /// Return every alive triangle whose three vertices are all input
    /// points (i.e. drop triangles still touching the super-triangle).
    /// Each triangle is returned as `[i, j, k]` with `i, j, k < n_input`.
    pub(super) fn extract_input_triangles(&self) -> Vec<[usize; 3]> {
        let mut out = Vec::new();
        for t in &self.triangles {
            if !t.alive {
                continue;
            }
            if t.v.iter().all(|&v| v < self.n_input) {
                out.push(t.v);
            }
        }
        out
    }

    /// Colour every triangle "outside" or "inside" by parity of the
    /// number of constrained edges crossed on any walk from the
    /// super-triangle.
    ///
    /// Triangles touching the super-triangle are seeded as **outside**.
    /// Crossing a constrained edge flips the colour; crossing a non-
    /// constrained edge preserves it. This is the standard "two-
    /// colouring across constraints" for polygon-with-holes flood-fill:
    /// 0 constraints crossed ⇒ outside the outer loop, 1 ⇒ inside the
    /// outer loop and outside every hole, 2 ⇒ inside a hole, etc.
    ///
    /// Returns a `Vec<bool>` of length `self.triangles.len()` where
    /// `true` marks a triangle to drop (outside the outer loop or
    /// inside a hole). Dead triangles are marked `true` so the caller
    /// can ignore them uniformly.
    fn flood_fill_outside(
        &self,
        constrained_edges: &std::collections::HashSet<(usize, usize)>,
    ) -> Vec<bool> {
        let n = self.triangles.len();
        // None ⇒ not visited; Some(true) ⇒ outside; Some(false) ⇒ inside.
        let mut colour: Vec<Option<bool>> = vec![None; n];
        let mut queue: Vec<usize> = Vec::new();

        for (idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                colour[idx] = Some(true); // dead = drop
                continue;
            }
            // Super-triangle sentinels live at indices [n_input, n_input + 3);
            // anything past that is a Steiner point added by refinement.
            if t.v.iter().any(|&v| v >= self.n_input && v < self.n_input + 3) {
                colour[idx] = Some(true);
                queue.push(idx);
            }
        }

        while let Some(t_idx) = queue.pop() {
            let t = self.triangles[t_idx];
            let c = colour[t_idx].unwrap();
            for k in 0..3 {
                let nb = t.n[k];
                if nb == NO_NEIGHBOUR || colour[nb].is_some() {
                    continue;
                }
                let a = t.v[(k + 1) % 3];
                let b = t.v[(k + 2) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                let next_c = if constrained_edges.contains(&key) {
                    !c
                } else {
                    c
                };
                colour[nb] = Some(next_c);
                queue.push(nb);
            }
        }

        colour.into_iter().map(|c| c.unwrap_or(true)).collect()
    }

    /// Return every triangle judged to be **inside** the polygon
    /// defined by the constrained edges (i.e. neither outside the
    /// outer loop nor inside any hole).
    pub(super) fn extract_interior_with_constraints(
        &self,
        constrained_edges: &std::collections::HashSet<(usize, usize)>,
    ) -> Vec<[usize; 3]> {
        let outside = self.flood_fill_outside(constrained_edges);
        let mut out = Vec::new();
        for (idx, t) in self.triangles.iter().enumerate() {
            if outside[idx] {
                continue;
            }
            if t.v.iter().all(|&v| v < self.n_input) {
                out.push(t.v);
            }
        }
        out
    }
}

/// Walk the chain implied by an unordered edge set between `start` and `end`.
fn build_chain(edges: &[(usize, usize)], start: usize, end: usize) -> Result<Vec<usize>> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in edges {
        adj.entry(u).or_default().push(v);
        adj.entry(v).or_default().push(u);
    }
    let mut chain = vec![start];
    let mut prev: Option<usize> = None;
    let mut current = start;
    loop {
        let nexts = adj.get(&current).ok_or_else(|| {
            PyrucastError::Message(format!("cdt::build_chain: vertex {} has no edges", current))
        })?;
        let next = nexts
            .iter()
            .find(|&&x| Some(x) != prev)
            .copied()
            .ok_or_else(|| {
                PyrucastError::Message(format!("cdt::build_chain: dead end at vertex {}", current))
            })?;
        chain.push(next);
        if next == end {
            break;
        }
        prev = Some(current);
        current = next;
        if chain.len() > edges.len() + 1 {
            return Err(PyrucastError::Message(
                "cdt::build_chain: chain failed to reach the endpoint".into(),
            ));
        }
    }
    Ok(chain)
}

#[inline]
fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    cross2(a, b, c)
}

/// Circumcenter of the triangle `(a, b, c)`. Returns `None` if the
/// three vertices are (nearly) collinear.
fn circumcenter(a: Point2, b: Point2, c: Point2) -> Option<Point2> {
    let d = 2.0
        * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-15 {
        return None;
    }
    let am = a.x * a.x + a.y * a.y;
    let bm = b.x * b.x + b.y * b.y;
    let cm = c.x * c.x + c.y * c.y;
    let ux = (am * (b.y - c.y) + bm * (c.y - a.y) + cm * (a.y - b.y)) / d;
    let uy = (am * (c.x - b.x) + bm * (a.x - c.x) + cm * (b.x - a.x)) / d;
    Some(Point2::new(ux, uy))
}

/// True iff the two open segments `(a, b)` and `(c, d)` cross strictly
/// (i.e. their interiors intersect; shared endpoints do not count).
fn segments_cross_strict(a: Point2, b: Point2, c: Point2, d: Point2) -> bool {
    let o1 = orient2d(a, b, c);
    let o2 = orient2d(a, b, d);
    let o3 = orient2d(c, d, a);
    let o4 = orient2d(c, d, b);
    o1 * o2 < 0.0 && o3 * o4 < 0.0
}

/// Bounding super-triangle of an input point set. Returns three
/// vertices large enough to enclose every point with margin.
fn super_triangle(points: &[Point2]) -> [Point2; 3] {
    let mut min = Vector2::new(f64::INFINITY, f64::INFINITY);
    let mut max = Vector2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in points {
        min = min.zip_map(&p.coords, f64::min);
        max = max.zip_map(&p.coords, f64::max);
    }
    let extents = (max - min).map(|d| d.max(1.0));
    let dmax = extents.x.max(extents.y);
    let centre = (min + max) * 0.5;
    // A wide isoceles triangle sitting under the centre. 20× the AABB
    // diagonal is overkill for robustness but trivial in cost.
    let r = 20.0 * dmax;
    [
        Point2::new(centre.x - r, centre.y - r),
        Point2::new(centre.x + r, centre.y - r),
        Point2::new(centre.x, centre.y + 2.0 * r),
    ]
}

/// Sign of the in-circle predicate: `> 0` means `d` lies **inside** the
/// circumcircle of `(a, b, c)`, assuming `(a, b, c)` is CCW.
#[inline]
fn in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let va = a - d;
    let vb = b - d;
    let vc = c - d;
    let am = va.norm_squared();
    let bm = vb.norm_squared();
    let cm = vc.norm_squared();
    va.x * (vb.y * cm - bm * vc.y)
        - va.y * (vb.x * cm - bm * vc.x)
        + am * (vb.x * vc.y - vb.y * vc.x)
}

/// Delaunay triangulation of a 2-D point set by Bowyer-Watson.
///
/// Returns one triangle per `[i, j, k]` triple (CCW), where `i, j, k`
/// are indices into the input `points` slice. The number of returned
/// triangles equals `2 · n - h - 2`, where `n = points.len()` and `h`
/// is the size of the convex hull (textbook identity); the function
/// does not check it.
///
/// # Errors
/// - `points.len() < 3`,
/// - two input points are exactly equal (the algorithm cannot resolve
///   coincident sites without symbolic perturbation, which is out of
///   scope for this iteration).
///
/// # Example
/// ```
/// use pyrucast::containers::mesh::Point2;
/// use pyrucast::ops::mesher::triangulation::delaunay_2d;
///
/// let pts = vec![
///     Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
/// ];
/// let tris = delaunay_2d(&pts).unwrap();
/// assert_eq!(tris.len(), 2);
/// ```
pub fn delaunay_2d(points: &[Point2]) -> Result<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 3 {
        return Err(PyrucastError::Message(format!(
            "delaunay_2d: need ≥ 3 points, got {}",
            n
        )));
    }
    // Detect coincident points (would derail the in_circle predicate).
    for i in 0..n {
        for j in (i + 1)..n {
            if (points[i] - points[j]).norm_squared() < 1e-24 {
                return Err(PyrucastError::Message(format!(
                    "delaunay_2d: points {} and {} are (nearly) coincident",
                    i, j
                )));
            }
        }
    }

    let mut cdt = Cdt::new(points);
    for i in 0..n {
        cdt.insert_point(i)?;
    }
    Ok(cdt.extract_input_triangles())
}

/// Constrained Delaunay triangulation: every edge in `constraints` is
/// guaranteed to appear in the output.
///
/// `points` is a flat list of 2-D points; `constraints` lists pairs of
/// indices into `points` that must remain as triangulation edges. The
/// function inserts every point first (Bowyer-Watson), then forces each
/// constraint by retiring the triangles it crosses and re-triangulating
/// the two resulting polygons by ear clipping.
///
/// Returns one CCW triangle per `[i, j, k]` triple, indexing into
/// `points`. Triangles still touching the bounding super-triangle are
/// discarded, but no further "inside the polygon" filtering is applied
/// at this layer — that is the job of the caller (`ops::mesher::fill_surface`
/// in the hole-removal step).
///
/// # Errors
/// - same as [`delaunay_2d`] for the point set,
/// - a constraint references a point index outside `0..points.len()`,
/// - a constraint cannot be enforced (e.g. its segment lies on the hull
///   in a way the walk cannot follow, or it crosses another already-
///   forced constraint).
pub fn constrained_delaunay_2d(
    points: &[Point2],
    constraints: &[(usize, usize)],
) -> Result<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 3 {
        return Err(PyrucastError::Message(format!(
            "constrained_delaunay_2d: need ≥ 3 points, got {}",
            n
        )));
    }
    for (i, j) in constraints {
        if *i >= n || *j >= n {
            return Err(PyrucastError::Message(format!(
                "constrained_delaunay_2d: constraint ({}, {}) out of bounds (n={})",
                i, j, n
            )));
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (points[i] - points[j]).norm_squared() < 1e-24 {
                return Err(PyrucastError::Message(format!(
                    "constrained_delaunay_2d: points {} and {} are (nearly) coincident",
                    i, j
                )));
            }
        }
    }

    let mut cdt = Cdt::new(points);
    for i in 0..n {
        cdt.insert_point(i)?;
    }
    for &(a, b) in constraints {
        cdt.insert_constraint(a, b)?;
    }
    Ok(cdt.extract_input_triangles())
}

/// Triangulate the interior of a closed polygon with optional holes.
///
/// `outer` is the ordered (CCW or CW) vertex list of the outer boundary;
/// `holes` lists the ordered vertex lists of each hole. Each loop closes
/// implicitly from its last vertex back to its first.
///
/// All vertices are flattened into a single list internally: the outer
/// loop first, then each hole concatenated in order. The returned
/// triangles use indices into this flat list, with the convention:
/// - outer indices: `0..outer.len()`
/// - hole `h` indices: `outer.len() + sum(holes[0..h].len()) + i`
///   for `i in 0..holes[h].len()`.
///
/// Algorithm:
/// 1. Build a constrained Delaunay triangulation with every loop edge
///    enforced as a constraint.
/// 2. Flood-fill from the bounding super-triangle through every
///    non-constrained edge; the visited triangles are everything that
///    is either outside the outer loop **or** inside a hole.
/// 3. Return the complement — the triangles that survived the
///    flood-fill, all CCW.
///
/// # Errors
/// - any loop has `< 3` vertices,
/// - two vertices coincide,
/// - a constraint cannot be enforced (e.g. loops cross),
/// - the flood-fill ends up empty (likely a degenerate or self-
///   intersecting polygon).
pub fn triangulate_polygon_with_holes(
    outer: &[Point2],
    holes: &[Vec<Point2>],
) -> Result<Vec<[usize; 3]>> {
    if outer.len() < 3 {
        return Err(PyrucastError::Message(format!(
            "triangulate_polygon_with_holes: outer loop must have ≥ 3 vertices, got {}",
            outer.len()
        )));
    }
    for (h, hole) in holes.iter().enumerate() {
        if hole.len() < 3 {
            return Err(PyrucastError::Message(format!(
                "triangulate_polygon_with_holes: hole #{} must have ≥ 3 vertices, got {}",
                h,
                hole.len()
            )));
        }
    }

    // Flatten points: outer first, then each hole.
    let mut points: Vec<Point2> = outer.to_vec();
    let mut hole_starts: Vec<usize> = Vec::with_capacity(holes.len());
    for hole in holes {
        hole_starts.push(points.len());
        points.extend_from_slice(hole);
    }
    let n_total = points.len();

    // Build closed-loop edge constraints.
    let mut constraints: Vec<(usize, usize)> = Vec::new();
    let n_outer = outer.len();
    for i in 0..n_outer {
        constraints.push((i, (i + 1) % n_outer));
    }
    for (h, hole) in holes.iter().enumerate() {
        let start = hole_starts[h];
        let n_hole = hole.len();
        for i in 0..n_hole {
            constraints.push((start + i, start + (i + 1) % n_hole));
        }
    }

    // Coincident-point check (across the whole flat list).
    for i in 0..n_total {
        for j in (i + 1)..n_total {
            if (points[i] - points[j]).norm_squared() < 1e-24 {
                return Err(PyrucastError::Message(format!(
                    "triangulate_polygon_with_holes: points {} and {} are (nearly) coincident",
                    i, j
                )));
            }
        }
    }

    let mut cdt = Cdt::new(&points);
    for i in 0..n_total {
        cdt.insert_point(i)?;
    }
    let mut edge_set: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::with_capacity(constraints.len());
    for &(a, b) in &constraints {
        cdt.insert_constraint(a, b)?;
        edge_set.insert(if a < b { (a, b) } else { (b, a) });
    }

    let tris = cdt.extract_interior_with_constraints(&edge_set);
    if tris.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_polygon_with_holes: no interior triangle survived — degenerate polygon?"
                .into(),
        ));
    }
    Ok(tris)
}

/// Same as [`triangulate_polygon_with_holes`] plus Ruppert refinement.
///
/// In addition to the inputs accepted by the unrefined variant, this
/// function takes a [`RefinementOptions`] descriptor (a maximum edge
/// length, a minimum triangle angle, or both). After building the CDT
/// it inserts Steiner points to satisfy the criteria.
///
/// Because the refinement step adds **new** points, the function returns
/// a pair `(points, triangles)`: `points` contains every vertex of the
/// final triangulation (input vertices first, then Steiner points in
/// insertion order); `triangles` indexes into that combined list. When
/// `options` is inactive (both fields `None`), the result is identical
/// to [`triangulate_polygon_with_holes`] except that `points` echoes
/// the input flat list.
pub fn triangulate_polygon_with_holes_refined(
    outer: &[Point2],
    holes: &[Vec<Point2>],
    options: RefinementOptions,
) -> Result<(Vec<Point2>, Vec<[usize; 3]>)> {
    if outer.len() < 3 {
        return Err(PyrucastError::Message(format!(
            "triangulate_polygon_with_holes_refined: outer loop must have ≥ 3 vertices, got {}",
            outer.len()
        )));
    }
    for (h, hole) in holes.iter().enumerate() {
        if hole.len() < 3 {
            return Err(PyrucastError::Message(format!(
                "triangulate_polygon_with_holes_refined: hole #{} must have ≥ 3 vertices, got {}",
                h,
                hole.len()
            )));
        }
    }
    if let Some(h) = options.max_edge_length {
        if h.is_nan() || h <= 0.0 {
            return Err(PyrucastError::Message(format!(
                "triangulate_polygon_with_holes_refined: max_edge_length must be > 0, got {}",
                h
            )));
        }
    }
    if let Some(a) = options.min_angle_deg {
        if !(a > 0.0 && a < 60.0) {
            return Err(PyrucastError::Message(format!(
                "triangulate_polygon_with_holes_refined: min_angle_deg must be in (0, 60), got {}",
                a
            )));
        }
    }

    // Flatten and build constraints — same as the unrefined façade.
    let mut points: Vec<Point2> = outer.to_vec();
    let mut hole_starts: Vec<usize> = Vec::with_capacity(holes.len());
    for hole in holes {
        hole_starts.push(points.len());
        points.extend_from_slice(hole);
    }
    let n_total = points.len();

    let mut constraints: Vec<(usize, usize)> = Vec::new();
    let n_outer = outer.len();
    for i in 0..n_outer {
        constraints.push((i, (i + 1) % n_outer));
    }
    for (h, hole) in holes.iter().enumerate() {
        let start = hole_starts[h];
        let n_hole = hole.len();
        for i in 0..n_hole {
            constraints.push((start + i, start + (i + 1) % n_hole));
        }
    }

    for i in 0..n_total {
        for j in (i + 1)..n_total {
            if (points[i] - points[j]).norm_squared() < 1e-24 {
                return Err(PyrucastError::Message(format!(
                    "triangulate_polygon_with_holes_refined: points {} and {} are (nearly) coincident",
                    i, j
                )));
            }
        }
    }

    let mut cdt = Cdt::new(&points);
    for i in 0..n_total {
        cdt.insert_point(i)?;
    }
    let mut edge_set: HashSet<(usize, usize)> = HashSet::with_capacity(constraints.len());
    for &(a, b) in &constraints {
        cdt.insert_constraint(a, b)?;
        edge_set.insert(if a < b { (a, b) } else { (b, a) });
    }

    cdt.refine(&options, &mut edge_set)?;

    // Build the final list of "user-visible" points: input points first,
    // then Steiner points (everything past index `n_input + 3`).
    let n_input = cdt.n_input;
    let mut out_points: Vec<Point2> = Vec::with_capacity(cdt.points.len() - 3);
    out_points.extend_from_slice(&cdt.points[..n_input]);
    if cdt.points.len() > n_input + 3 {
        out_points.extend_from_slice(&cdt.points[n_input + 3..]);
    }

    // Re-map triangle vertex indices: Steiner indices (≥ n_input + 3)
    // shift down by 3 in the user-visible list.
    let remap = |v: usize| -> usize {
        if v < n_input {
            v
        } else {
            v - 3
        }
    };

    let outside = cdt.flood_fill_outside(&edge_set);
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for (idx, t) in cdt.triangles.iter().enumerate() {
        if outside[idx] {
            continue;
        }
        // Skip triangles still touching the super-triangle.
        if t.v.iter().any(|&v| v >= n_input && v < n_input + 3) {
            continue;
        }
        tris.push([remap(t.v[0]), remap(t.v[1]), remap(t.v[2])]);
    }
    if tris.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_polygon_with_holes_refined: no interior triangle survived".into(),
        ));
    }
    Ok((out_points, tris))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p2(x: f64, y: f64) -> Point2 {
        Point2::new(x, y)
    }

    fn signed_area(a: Point2, b: Point2, c: Point2) -> f64 {
        0.5 * cross2(a, b, c)
    }

    fn assert_all_ccw(tris: &[[usize; 3]], pts: &[Point2]) {
        for [i, j, k] in tris {
            let s = signed_area(pts[*i], pts[*j], pts[*k]);
            assert!(s > 0.0, "triangle [{i},{j},{k}] not CCW (area={s})");
        }
    }

    fn total_area(tris: &[[usize; 3]], pts: &[Point2]) -> f64 {
        tris.iter()
            .map(|[i, j, k]| signed_area(pts[*i], pts[*j], pts[*k]))
            .sum()
    }

    #[test]
    fn delaunay_single_triangle() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.0, 1.0)];
        let tris = delaunay_2d(&pts).unwrap();
        assert_eq!(tris.len(), 1);
        assert_all_ccw(&tris, &pts);
    }

    #[test]
    fn delaunay_square_two_triangles() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        let tris = delaunay_2d(&pts).unwrap();
        assert_eq!(tris.len(), 2);
        assert_all_ccw(&tris, &pts);
        assert!((total_area(&tris, &pts) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn delaunay_pentagon_three_triangles() {
        let n = 5;
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                p2(t.cos(), t.sin())
            })
            .collect();
        let tris = delaunay_2d(&pts).unwrap();
        assert_eq!(tris.len(), 3); // 2n - h - 2 with h = n = 5 ⇒ 3
        assert_all_ccw(&tris, &pts);
        let expected = 0.5 * (n as f64) * (2.0 * std::f64::consts::PI / n as f64).sin();
        assert!((total_area(&tris, &pts) - expected).abs() < 1e-10);
    }

    #[test]
    fn delaunay_with_interior_point() {
        // Square + a point in the middle: should produce 4 triangles
        // (a fan from the centre to each square corner is the Delaunay
        // triangulation when the point is centred).
        let pts = vec![
            p2(0.0, 0.0),
            p2(1.0, 0.0),
            p2(1.0, 1.0),
            p2(0.0, 1.0),
            p2(0.5, 0.5),
        ];
        let tris = delaunay_2d(&pts).unwrap();
        assert_eq!(tris.len(), 4); // 2·5 - 4 - 2
        assert_all_ccw(&tris, &pts);
        assert!((total_area(&tris, &pts) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn delaunay_grid_3x3() {
        // 9 grid points; convex hull = 4 corners ⇒ #tri = 2·9 - 4 - 2 = 12.
        let mut pts = Vec::with_capacity(9);
        for j in 0..3 {
            for i in 0..3 {
                pts.push(p2(i as f64, j as f64));
            }
        }
        let tris = delaunay_2d(&pts).unwrap();
        // The hull is the outer 8 boundary points ⇒ h = 8 ⇒ #tri = 8.
        assert_eq!(tris.len(), 8);
        assert_all_ccw(&tris, &pts);
        assert!((total_area(&tris, &pts) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn delaunay_rejects_fewer_than_three_points() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0)];
        assert!(delaunay_2d(&pts).is_err());
    }

    #[test]
    fn delaunay_rejects_coincident_points() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.5, 1.0), p2(1.0, 0.0)];
        assert!(delaunay_2d(&pts).is_err());
    }

    fn triangulation_has_edge(tris: &[[usize; 3]], a: usize, b: usize) -> bool {
        tris.iter().any(|[i, j, k]| {
            let e = [(*i, *j), (*j, *k), (*k, *i)];
            e.iter().any(|&(p, q)| (p == a && q == b) || (p == b && q == a))
        })
    }

    #[test]
    fn cdt_no_constraint_matches_delaunay() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        let unconstrained = delaunay_2d(&pts).unwrap();
        let constrained = constrained_delaunay_2d(&pts, &[]).unwrap();
        assert_eq!(unconstrained.len(), constrained.len());
    }

    #[test]
    fn cdt_redundant_constraint_is_noop() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.0, 1.0)];
        let tris = constrained_delaunay_2d(&pts, &[(0, 1)]).unwrap();
        assert_eq!(tris.len(), 1);
        assert_all_ccw(&tris, &pts);
    }

    #[test]
    fn cdt_forces_long_rectangle_diagonal() {
        let pts = vec![p2(0.0, 0.0), p2(2.0, 0.0), p2(2.0, 1.0), p2(0.0, 1.0)];
        let unconstrained = delaunay_2d(&pts).unwrap();
        assert!(triangulation_has_edge(&unconstrained, 0, 2));
        assert!(!triangulation_has_edge(&unconstrained, 1, 3));

        let constrained = constrained_delaunay_2d(&pts, &[(1, 3)]).unwrap();
        assert_eq!(constrained.len(), 2);
        assert!(
            triangulation_has_edge(&constrained, 1, 3),
            "forced edge (1, 3) missing: {:?}",
            constrained
        );
        assert_all_ccw(&constrained, &pts);
        assert!((total_area(&constrained, &pts) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cdt_forces_edge_across_interior_point() {
        let pts = vec![
            p2(0.0, 0.0),
            p2(2.0, 0.0),
            p2(2.0, 2.0),
            p2(0.0, 2.0),
            p2(1.0, 0.6),
        ];
        let unconstrained = delaunay_2d(&pts).unwrap();
        let constrained = constrained_delaunay_2d(&pts, &[(0, 2)]).unwrap();
        assert!(triangulation_has_edge(&constrained, 0, 2));
        assert_all_ccw(&constrained, &pts);
        assert!((total_area(&constrained, &pts) - 4.0).abs() < 1e-12);
        assert_eq!(constrained.len(), unconstrained.len());
    }

    #[test]
    fn cdt_rejects_degenerate_constraint() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.5, 1.0)];
        assert!(constrained_delaunay_2d(&pts, &[(1, 1)]).is_err());
    }

    #[test]
    fn cdt_rejects_out_of_bounds_constraint() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.5, 1.0)];
        assert!(constrained_delaunay_2d(&pts, &[(0, 5)]).is_err());
    }

    #[test]
    fn holes_square_no_holes_matches_plain_triangulation() {
        let outer = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        let tris = triangulate_polygon_with_holes(&outer, &[]).unwrap();
        assert_eq!(tris.len(), 2);
        assert_all_ccw(&tris, &outer);
        assert!((total_area(&tris, &outer) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn holes_square_with_one_square_hole() {
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 4.0), p2(0.0, 4.0)];
        let hole = vec![p2(1.0, 1.0), p2(3.0, 1.0), p2(3.0, 3.0), p2(1.0, 3.0)];
        let tris = triangulate_polygon_with_holes(&outer, std::slice::from_ref(&hole)).unwrap();

        let mut all: Vec<Point2> = outer.clone();
        all.extend_from_slice(&hole);
        assert_all_ccw(&tris, &all);

        let area = total_area(&tris, &all);
        assert!((area - 12.0).abs() < 1e-12, "area = {}", area);

        for k in 0..4 {
            assert!(triangulation_has_edge(&tris, 4 + k, 4 + (k + 1) % 4));
        }
    }

    #[test]
    fn holes_square_with_two_holes() {
        let outer = vec![p2(0.0, 0.0), p2(6.0, 0.0), p2(6.0, 4.0), p2(0.0, 4.0)];
        let h1 = vec![p2(1.0, 1.0), p2(2.0, 1.0), p2(2.0, 2.0), p2(1.0, 2.0)];
        let h2 = vec![p2(4.0, 2.0), p2(5.0, 2.0), p2(5.0, 3.0), p2(4.0, 3.0)];
        let tris = triangulate_polygon_with_holes(&outer, &[h1.clone(), h2.clone()]).unwrap();
        let mut all: Vec<Point2> = outer.clone();
        all.extend_from_slice(&h1);
        all.extend_from_slice(&h2);
        assert_all_ccw(&tris, &all);

        let area = total_area(&tris, &all);
        assert!((area - 22.0).abs() < 1e-12, "area = {}", area);
    }

    #[test]
    fn holes_rejects_outer_with_fewer_than_three_vertices() {
        let outer = vec![p2(0.0, 0.0), p2(1.0, 0.0)];
        assert!(triangulate_polygon_with_holes(&outer, &[]).is_err());
    }

    #[test]
    fn holes_rejects_undersized_hole() {
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 4.0), p2(0.0, 4.0)];
        let bad_hole = vec![p2(1.0, 1.0), p2(2.0, 2.0)];
        assert!(triangulate_polygon_with_holes(&outer, &[bad_hole]).is_err());
    }

    #[test]
    fn holes_orientation_independent() {
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 4.0), p2(0.0, 4.0)];
        let hole_ccw = vec![p2(1.0, 1.0), p2(3.0, 1.0), p2(3.0, 3.0), p2(1.0, 3.0)];
        let hole_cw = vec![p2(1.0, 1.0), p2(1.0, 3.0), p2(3.0, 3.0), p2(3.0, 1.0)];

        let tris_ccw = triangulate_polygon_with_holes(&outer, &[hole_ccw]).unwrap();
        let tris_cw = triangulate_polygon_with_holes(&outer, &[hole_cw]).unwrap();
        assert_eq!(tris_ccw.len(), tris_cw.len());
    }

    fn min_angle_deg(tris: &[[usize; 3]], pts: &[Point2]) -> f64 {
        let mut min_deg = f64::INFINITY;
        for [i, j, k] in tris {
            let a = pts[*i];
            let b = pts[*j];
            let c = pts[*k];
            for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)] {
                let e1 = v - u;
                let e2 = w - u;
                let cos = e1.dot(&e2) / (e1.norm() * e2.norm());
                let ang = cos.clamp(-1.0, 1.0).acos().to_degrees();
                if ang < min_deg {
                    min_deg = ang;
                }
            }
        }
        min_deg
    }

    fn max_edge_length(tris: &[[usize; 3]], pts: &[Point2]) -> f64 {
        let mut m = 0.0_f64;
        for [i, j, k] in tris {
            for (u, v) in [(*i, *j), (*j, *k), (*k, *i)] {
                let l = (pts[v] - pts[u]).norm();
                if l > m {
                    m = l;
                }
            }
        }
        m
    }

    #[test]
    fn refine_inactive_options_is_noop() {
        let outer = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        let (pts, tris) =
            triangulate_polygon_with_holes_refined(&outer, &[], RefinementOptions::default())
                .unwrap();
        assert_eq!(pts.len(), outer.len());
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn refine_size_only_inserts_steiner_points() {
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 4.0), p2(0.0, 4.0)];
        let opts = RefinementOptions {
            max_edge_length: Some(1.5),
            min_angle_deg: None,
        };
        let (pts, tris) = triangulate_polygon_with_holes_refined(&outer, &[], opts).unwrap();
        assert!(pts.len() > outer.len(), "no Steiner points were added");
        assert_all_ccw(&tris, &pts);
        assert!(max_edge_length(&tris, &pts) <= 1.5 + 1e-9);
        // Area conserved (4×4 = 16).
        assert!((total_area(&tris, &pts) - 16.0).abs() < 1e-9);
    }

    #[test]
    fn refine_angle_only_improves_min_angle() {
        // A moderately thin rectangle (4×1) has minimum angles around 14°
        // in its initial Delaunay triangulation. Refining with min_angle =
        // 20° should bring every triangle above the threshold.
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 1.0), p2(0.0, 1.0)];
        let opts = RefinementOptions {
            max_edge_length: None,
            min_angle_deg: Some(20.0),
        };
        let (pts, tris) = triangulate_polygon_with_holes_refined(&outer, &[], opts).unwrap();
        assert!(pts.len() > outer.len(), "no Steiner points were added");
        assert_all_ccw(&tris, &pts);
        // Some numerical slack — Ruppert can leave a couple of degrees.
        let m = min_angle_deg(&tris, &pts);
        assert!(m >= 19.0, "min angle still bad: {} deg", m);
        // Area conserved (4.0).
        assert!((total_area(&tris, &pts) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn refine_size_with_hole() {
        // 4×4 square with a 2×2 hole, plus size criterion.
        let outer = vec![p2(0.0, 0.0), p2(4.0, 0.0), p2(4.0, 4.0), p2(0.0, 4.0)];
        let hole = vec![p2(1.0, 1.0), p2(3.0, 1.0), p2(3.0, 3.0), p2(1.0, 3.0)];
        let opts = RefinementOptions {
            max_edge_length: Some(1.0),
            min_angle_deg: None,
        };
        let (pts, tris) =
            triangulate_polygon_with_holes_refined(&outer, &[hole], opts).unwrap();
        assert_all_ccw(&tris, &pts);
        assert!(max_edge_length(&tris, &pts) <= 1.0 + 1e-9);
        // Total area = 16 - 4 = 12.
        assert!((total_area(&tris, &pts) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn refine_rejects_invalid_options() {
        let outer = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0)];
        let bad_h = RefinementOptions {
            max_edge_length: Some(-0.5),
            min_angle_deg: None,
        };
        assert!(triangulate_polygon_with_holes_refined(&outer, &[], bad_h).is_err());
        let bad_a = RefinementOptions {
            max_edge_length: None,
            min_angle_deg: Some(90.0),
        };
        assert!(triangulate_polygon_with_holes_refined(&outer, &[], bad_a).is_err());
    }

    #[test]
    fn delaunay_satisfies_empty_circle_property() {
        let pts = vec![
            p2(0.0, 0.0),
            p2(4.0, 0.0),
            p2(4.0, 3.0),
            p2(0.0, 3.0),
            p2(1.0, 1.0),
            p2(3.0, 2.0),
            p2(2.0, 2.5),
        ];
        let tris = delaunay_2d(&pts).unwrap();
        for [i, j, k] in &tris {
            let a = pts[*i];
            let b = pts[*j];
            let c = pts[*k];
            for (m, &p) in pts.iter().enumerate() {
                if m == *i || m == *j || m == *k {
                    continue;
                }
                let v = in_circle(a, b, c, p);
                assert!(
                    v <= 1e-9,
                    "point #{m} lies inside circumcircle of [{i},{j},{k}] (val={v})"
                );
            }
        }
    }
}
