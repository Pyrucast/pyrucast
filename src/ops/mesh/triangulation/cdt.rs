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

use super::{cross2, point_in_triangle};
use crate::atoms::{Point2, Vector2};
use crate::error::{PyrucastError, Result};
use std::collections::{BTreeSet, HashMap, HashSet};

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
///
/// ```
/// # use pyrucast::atoms::Point2;
/// # use pyrucast::ops::mesh::triangulation;
/// # use pyrucast::ops::mesh::triangulation::RefinementOptions;
/// // Par défaut, aucun raffinement demandé.
/// assert!(!RefinementOptions::default().is_active());
/// // Ruppert n'est **prouvé terminer** que jusqu'à 20,7° ; au-delà, un
/// // plafond de points de Steiner rend une erreur claire plutôt qu'une
/// // divergence.
/// let o = RefinementOptions { max_edge_length: Some(0.5), min_angle_deg: Some(20.0) };
/// assert!(o.is_active());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::Point2;
    /// # use pyrucast::ops::mesh::triangulation;
    /// # use pyrucast::ops::mesh::triangulation::RefinementOptions;
    /// assert!(!RefinementOptions::default().is_active());
    /// assert!(RefinementOptions { max_edge_length: Some(0.5), ..Default::default() }
    ///     .is_active());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
            if in_circle_tolerant(a, b, c, p) {
                bad.push(t_idx);
            }
        }
        if bad.is_empty() {
            return Err(PyrucastError::Message(format!(
                "cdt::insert_point: no bad triangle for point #{} — super-triangle too small?",
                p_idx
            )));
        }

        // 1b. Enforce a star-shaped cavity. `in_circle_tolerant` treats
        //     cocircular points (which `arc`/`circle` contours produce by
        //     the dozen) as "inside" to grow the cavity in one step, but
        //     that tolerance can wrongly pull in a triangle the point does
        //     not actually see, making the cavity non-star-shaped. The fan
        //     fill below would then emit overlapping/inverted triangles,
        //     which the next insertion sees as even more "bad" ones — an
        //     exponential blow-up. Drop any cavity triangle exposing a
        //     boundary edge the point cannot see (`orient2d <= 0`) until
        //     the cavity is star-shaped about `p`. Retain the triangle that
        //     geometrically contains `p` as a seed anchor.
        let mut bad_set: std::collections::HashSet<usize> = bad.iter().copied().collect();
        let anchor = bad
            .iter()
            .copied()
            .find(|&t_idx| {
                let t = self.triangles[t_idx];
                point_in_triangle(
                    p,
                    self.points[t.v[0]],
                    self.points[t.v[1]],
                    self.points[t.v[2]],
                )
            })
            .or_else(|| bad.first().copied());
        loop {
            let mut to_remove: Option<usize> = None;
            'outer: for &t_idx in &bad {
                if Some(t_idx) == anchor {
                    continue;
                }
                let t = self.triangles[t_idx];
                for k in 0..3 {
                    let nb = t.n[k];
                    if nb == NO_NEIGHBOUR || !bad_set.contains(&nb) {
                        let va = t.v[(k + 1) % 3];
                        let vb = t.v[(k + 2) % 3];
                        if orient2d(self.points[va], self.points[vb], p) <= 0.0 {
                            to_remove = Some(t_idx);
                            break 'outer;
                        }
                    }
                }
            }
            match to_remove {
                Some(t_idx) => {
                    bad_set.remove(&t_idx);
                    bad.retain(|&x| x != t_idx);
                }
                None => break,
            }
        }

        // 2. Boundary of the cavity = edges of bad triangles whose opposite
        //    neighbour is NOT itself bad.
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

        // Collinear-overlap case. The Delaunay triangulation of collinear
        // points (a whole straight boundary side pre-discretized into many
        // segments) is degenerate: the wanted edge (a, b) may run *along*
        // longer existing edges rather than crossing any triangle, so the
        // crossing walk finds no start. Two sub-cases, both resolved by
        // splitting at the collinear vertex nearest `a`:
        //  - a vertex `c` lies strictly on the open segment (a, b): enforce
        //    (a, c) then (c, b);
        //  - an existing edge (a, c) is collinear with (a, b) and reaches
        //    *past* `b` (so (a, b) is a prefix of it): flip that edge so a
        //    shorter one toward `b` can form, then retry.
        if let Some(c) = self.vertex_on_segment(a, b) {
            self.insert_constraint(a, c)?;
            return self.insert_constraint(c, b);
        }
        if self.split_overlapping_collinear_edge(a, b)? {
            return self.insert_constraint(a, b);
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
            let tris = crate::ops::mesh::triangulation::ear_clip_2d(&pts)?;
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

    /// Handle a wanted edge (a, b) that is the *prefix* of a longer
    /// existing edge (a, c): `b` lies strictly on the interior of edge
    /// (a, c). Split each of the (at most two) triangles incident to edge
    /// (a, c) at `b`, so edges (a, b) and (b, c) replace (a, c). Returns
    /// `true` if such an edge was found and split.
    ///
    /// This unblocks constraint enforcement along a collinear boundary,
    /// where Delaunay of the collinear points may connect `a` directly to
    /// a vertex beyond `b`.
    fn split_overlapping_collinear_edge(&mut self, a: usize, b: usize) -> Result<bool> {
        let pa = self.points[a];
        let pb = self.points[b];
        let ab = pb - pa;
        let len2 = ab.norm_squared();
        if len2 == 0.0 {
            return Ok(false);
        }

        // Find an alive triangle with an edge (a, c) collinear with (a, b)
        // where `b` lies strictly between `a` and `c`.
        let mut target_c: Option<usize> = None;
        'search: for t in &self.triangles {
            if !t.alive {
                continue;
            }
            let Some(pos) = t.v.iter().position(|&v| v == a) else {
                continue;
            };
            for &c in [t.v[(pos + 1) % 3], t.v[(pos + 2) % 3]].iter() {
                if c == b {
                    continue;
                }
                let pc = self.points[c] - pa;
                // Collinear with a→b, same direction, and reaching past b.
                if orient2d(pa, pb, self.points[c]).abs() > 1e-9 * len2 {
                    continue;
                }
                let t_along = pc.dot(&ab) / len2;
                if t_along > 1.0 + 1e-12 {
                    target_c = Some(c);
                    break 'search;
                }
            }
        }
        let Some(c) = target_c else {
            return Ok(false);
        };

        // Retire both triangles on edge (a, c) and split each at `b`. `b`
        // lies on segment (a, c), so for a triangle (a, c, apex) the two
        // halves (a, b, apex) and (b, c, apex) preserve the winding.
        let incident: Vec<usize> = self
            .triangles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.alive && edge_in_triangle(t, a, c))
            .map(|(i, _)| i)
            .collect();
        if incident.is_empty() {
            return Ok(false);
        }
        for t_idx in incident {
            let t = self.triangles[t_idx];
            // Apex = the vertex that is neither a nor c.
            let apex = *t.v.iter().find(|&&v| v != a && v != c).unwrap();
            // Preserve orientation: find a/c order as they appear CCW.
            let pos_a = t.v.iter().position(|&v| v == a).unwrap();
            let next = t.v[(pos_a + 1) % 3];
            self.triangles[t_idx].alive = false;
            let (first, second) = if next == c {
                // ... a, c, apex ... → (a,b,apex),(b,c,apex)
                ([a, b, apex], [b, c, apex])
            } else {
                // ... a, apex, c ... → (a,apex,b),(b,apex,c)
                ([a, apex, b], [b, apex, c])
            };
            self.triangles.push(Triangle {
                v: first,
                n: [NO_NEIGHBOUR; 3],
                alive: true,
            });
            self.triangles.push(Triangle {
                v: second,
                n: [NO_NEIGHBOUR; 3],
                alive: true,
            });
        }
        self.rebuild_neighbours();
        Ok(true)
    }

    /// The input vertex nearest `a` that lies strictly on the open
    /// segment `(a, b)` (collinear and strictly between the endpoints),
    /// or `None`. Used to split a constraint that overlaps other vertices
    /// — the degenerate collinear-boundary case.
    fn vertex_on_segment(&self, a: usize, b: usize) -> Option<usize> {
        let pa = self.points[a];
        let pb = self.points[b];
        let ab = pb - pa;
        let len2 = ab.norm_squared();
        if len2 == 0.0 {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, q) in self.points.iter().enumerate() {
            if i == a || i == b {
                continue;
            }
            // Super-triangle sentinels are never on a real constraint.
            if i >= self.n_input && i < self.n_input + 3 {
                continue;
            }
            // `orient2d` is twice the (a, b, q) area = len(ab) · perp_dist,
            // so `|orient2d| < 1e-9 · len²` means `perp_dist < 1e-9 · len`:
            // collinear to a length-relative tolerance.
            if orient2d(pa, pb, *q).abs() > 1e-9 * len2 {
                continue;
            }
            // Parametric position along a→b, strictly inside (0, 1).
            let t = (q - pa).dot(&ab) / len2;
            if t > 1e-12 && t < 1.0 - 1e-12 && best.map(|(_, bt)| t < bt).unwrap_or(true) {
                best = Some((i, t));
            }
        }
        best.map(|(i, _)| i)
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
        constrained_edges: &BTreeSet<(usize, usize)>,
    ) -> Result<()> {
        self.insert_point_constrained_seeded(p_idx, None, constrained_edges)
    }

    /// Insert `p_idx` into the constrained triangulation.
    ///
    /// `seed_hint`, when given, is used as the cavity BFS seed instead of
    /// searching for the triangle geometrically containing the point.
    /// The refiner uses this when it knows exactly which triangle a new
    /// point is meant to retire: the geometric seed search returns the
    /// *first* triangle containing the point, and with an inclusive
    /// point-in-triangle test (a point on a shared edge counts as inside
    /// both) it can pick a neighbour that a constraint then walls off
    /// from the intended triangle — leaving that triangle alive and the
    /// refiner looping on it.
    fn insert_point_constrained_seeded(
        &mut self,
        p_idx: usize,
        seed_hint: Option<usize>,
        constrained_edges: &BTreeSet<(usize, usize)>,
    ) -> Result<()> {
        let p = self.points[p_idx];

        // 1. Locate a triangle containing `p` (or use the caller's hint).
        let seed = match seed_hint.filter(|&s| self.triangles[s].alive) {
            Some(s) => s,
            None => {
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
                seed.ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "cdt::insert_point_constrained: no triangle contains point #{}",
                        p_idx
                    ))
                })?
            }
        };

        // 2. BFS from `seed`, only crossing non-constrained edges and only
        //    when in_circle > 0. `seed` itself is exempt from that test: `p`
        //    was located strictly inside it (`point_in_triangle`), which
        //    mathematically guarantees `p` is inside its circumcircle too —
        //    but with near-cocircular input (e.g. several points sampled on
        //    the same construction circle, as `arc`/`circle` produce), plain
        //    floating-point `in_circle` can land it right on that boundary
        //    and misreport ≤ 0. Trusting the exact containment test instead
        //    of the numerically fragile one avoids a spurious "did not
        //    produce a cavity" failure.
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
            if t_idx != seed && !in_circle_tolerant(a, b, c, p) {
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

        // 2b. Enforce a star-shaped cavity. The fan fill below builds a
        //     triangle (a, b, p) for each cavity boundary edge (a, b),
        //     which is only valid (CCW, non-overlapping) when `p` sees
        //     every boundary edge — i.e. the cavity is star-shaped about
        //     `p`. `in_circle` growth stopped at constraints can leave it
        //     otherwise, spawning inverted triangles. Iteratively drop any
        //     cavity triangle that exposes a boundary edge `p` does not see
        //     (`orient2d(pa, pb, p) <= 0`); the `seed` is kept always
        //     (`p` lies inside it, so it is always visible). Repeat until
        //     every boundary edge is visible.
        let mut bad_set: HashSet<usize> = bad.iter().copied().collect();
        loop {
            let mut to_remove: Option<usize> = None;
            'outer: for &t_idx in &bad {
                if t_idx == seed {
                    continue;
                }
                let t = self.triangles[t_idx];
                for k in 0..3 {
                    let nb = t.n[k];
                    if nb == NO_NEIGHBOUR || !bad_set.contains(&nb) {
                        let va = t.v[(k + 1) % 3];
                        let vb = t.v[(k + 2) % 3];
                        if orient2d(self.points[va], self.points[vb], p) <= 0.0 {
                            to_remove = Some(t_idx);
                            break 'outer;
                        }
                    }
                }
            }
            match to_remove {
                Some(t_idx) => {
                    bad_set.remove(&t_idx);
                    bad.retain(|&x| x != t_idx);
                }
                None => break,
            }
        }

        // 3. Cavity boundary: edges of bad triangles whose opposite
        //    neighbour is not itself bad.
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
            if let Some(max_sq) = max_edge_sq
                && self.triangle_longest_edge_sq(idx) > max_sq
            {
                return Some(idx);
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
        constrained_edges: &BTreeSet<(usize, usize)>,
        min_split_len_sq: f64,
    ) -> Option<(usize, usize)> {
        for &(a, b) in constrained_edges {
            // Never split a subsegment already shorter than the floor: doing
            // so would recurse forever where two constraints meet at a small
            // angle (Ruppert's classic non-termination). At that size the
            // subsegment is already far finer than the target anyway.
            if (self.points[b] - self.points[a]).norm_squared() <= min_split_len_sq {
                continue;
            }
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
        if let Some(p) = extra_point
            && (p - mid).norm_squared() < r2 - strict
        {
            return true;
        }
        false
    }

    /// If `p` lies strictly in the diametral disk of any constrained
    /// edge, return one such edge. Used by Ruppert's algorithm before
    /// inserting a circumcenter.
    fn encroached_constraint_by(
        &self,
        p: Point2,
        constrained_edges: &BTreeSet<(usize, usize)>,
        min_split_len_sq: f64,
    ) -> Option<(usize, usize)> {
        let strict = 1e-12;
        for &(a, b) in constrained_edges {
            let pa = self.points[a];
            let pb = self.points[b];
            // Do not report an already-tiny subsegment as encroached — see
            // `first_encroached_constraint` for why splitting it loops.
            if (pb - pa).norm_squared() <= min_split_len_sq {
                continue;
            }
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
        constrained_edges: &mut BTreeSet<(usize, usize)>,
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
        constrained_edges: &mut BTreeSet<(usize, usize)>,
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

        // Floor on subsegment splitting. Two constraints meeting at a small
        // angle make Ruppert recurse forever, each split shrinking the
        // subsegment without ever clearing the encroachment. Refuse to split
        // any subsegment already much finer than the target size (or, absent a
        // size target, than the shortest initial constraint). This trades the
        // formal angle guarantee — unattainable for arbitrary input angles
        // anyway — for guaranteed termination; the few triangles left near a
        // sharp corner stay a touch coarse.
        let mut shortest_input_sq = f64::INFINITY;
        for &(a, b) in constrained_edges.iter() {
            let l2 = (self.points[b] - self.points[a]).norm_squared();
            if l2 > 0.0 && l2 < shortest_input_sq {
                shortest_input_sq = l2;
            }
        }
        if !shortest_input_sq.is_finite() {
            shortest_input_sq = 0.0;
        }
        let min_split_len_sq = match opts.max_edge_length {
            Some(h) => (h / 8.0).powi(2).min(shortest_input_sq * 0.25),
            None => shortest_input_sq * 0.0625,
        };

        loop {
            if self.points.len() >= initial_points + max_inserts {
                return Err(PyrucastError::Message(format!(
                    "cdt::refine: did not converge after {} Steiner insertions \
                     (max_edge_length={:?}, min_angle_deg={:?}); criteria may be too tight",
                    max_inserts, opts.max_edge_length, opts.min_angle_deg
                )));
            }

            // 1. Encroached constraint?
            if let Some((a, b)) =
                self.first_encroached_constraint(constrained_edges, min_split_len_sq)
            {
                self.split_constraint(a, b, constrained_edges)?;
                continue;
            }

            // 2. Find a bad interior triangle.
            let outside = self.flood_fill_outside(constrained_edges);
            let bad =
                self.find_bad_interior_triangle(&outside, max_edge_sq, radius_ratio_sq_threshold);
            let Some(t_idx) = bad else {
                return Ok(());
            };

            let Some(cc) = self.triangle_circumcenter(t_idx) else {
                // Near-zero-area (collinear) triangle: its circumcenter is
                // undefined, but cocircular input (`circle`/`arc`-built
                // contours routinely produce many points on one circle) can
                // transiently create these during refinement. Fall back to
                // the same in-domain split used when a circumcenter cannot
                // be inserted (constrained edge → split it, else centroid).
                self.split_triangle_longest_edge(t_idx, min_split_len_sq, constrained_edges)?;
                continue;
            };

            // If the circumcenter would encroach a constraint, split
            // that constraint instead.
            if let Some((a, b)) =
                self.encroached_constraint_by(cc, constrained_edges, min_split_len_sq)
            {
                self.split_constraint(a, b, constrained_edges)?;
                continue;
            }

            // The circumcenter of an obtuse boundary triangle can fall
            // *outside* the domain (past the outer loop or inside a hole)
            // without lying in any constrained edge's diametral disk.
            // Inserting it there would not improve the bad triangle. The
            // circumcenter can also lie inside the domain yet be separated
            // from the bad triangle by a hole boundary, so that the
            // constrained cavity BFS never reaches — and never retires —
            // the bad triangle. In both cases the refiner would pick the
            // same triangle again forever. Fall back to bisecting the bad
            // triangle's longest edge, which always shrinks it and
            // terminates for a size criterion.
            let bad_tri_vertices = self.triangles[t_idx].v;
            if !self.centroid_inside_constraints(cc, constrained_edges) {
                self.split_triangle_longest_edge(t_idx, min_split_len_sq, constrained_edges)?;
                continue;
            }

            let new_idx = self.points.len();
            self.points.push(cc);
            self.insert_point_constrained(new_idx, constrained_edges)?;

            // Did the insertion actually retire the bad triangle? The
            // geometric seed search starts from whatever triangle contains
            // `cc`; a constraint can wall that triangle off from the bad
            // one, so the cavity BFS never reaches — never retires — it,
            // and the refiner would loop on the same triangle forever.
            // Fall back to a guaranteed-interior centroid split, seeded
            // directly from the bad triangle.
            if self.triangle_alive_with_vertices(bad_tri_vertices) {
                self.points.pop();
                self.split_triangle_longest_edge(t_idx, min_split_len_sq, constrained_edges)?;
            }
        }
    }

    /// Shrink triangle `t_idx` by longest-edge bisection (Rivara) when its
    /// circumcenter cannot be used — because it lies outside the domain, or
    /// is walled off from `t_idx` by a constraint so the cavity BFS never
    /// retires the triangle.
    ///
    /// The longest edge drives the split:
    /// - a **constrained** longest edge → [`split_constraint`], which
    ///   subdivides the boundary (the Ruppert response to an encroached
    ///   subsegment);
    /// - a **free** longest edge → a point at its midpoint nudged a hair
    ///   toward the opposite vertex, so it is *strictly interior* to
    ///   `t_idx`. Inserted seeded from `t_idx`, it always retires the
    ///   triangle while keeping the Bowyer-Watson cavity star-shaped — a
    ///   point placed exactly *on* the edge (or the raw centroid seeded
    ///   across a constraint) can instead invert neighbouring triangles.
    ///
    /// Splitting toward the longest edge shrinks it and terminates for a
    /// size criterion, unlike a plain centroid insertion (which leaves a
    /// long-edged child and cascades).
    ///
    /// An edge already at or below `min_split_len_sq` is never chosen (that
    /// would recurse forever near a small input angle); if every edge is
    /// that fine, split toward the first edge anyway — the nudged point is
    /// still strictly interior and still retires the triangle.
    fn split_triangle_longest_edge(
        &mut self,
        t_idx: usize,
        min_split_len_sq: f64,
        constrained_edges: &mut BTreeSet<(usize, usize)>,
    ) -> Result<()> {
        // Longest edge of the triangle above the split floor.
        let t = self.triangles[t_idx];
        let mut best: Option<((usize, usize), f64)> = None;
        for k in 0..3 {
            let a = t.v[k];
            let b = t.v[(k + 1) % 3];
            let len2 = (self.points[b] - self.points[a]).norm_squared();
            if len2 > min_split_len_sq && best.map(|(_, l)| len2 > l).unwrap_or(true) {
                best = Some(((a, b), len2));
            }
        }

        if let Some(((a, b), _)) = best {
            let key = if a < b { (a, b) } else { (b, a) };
            if constrained_edges.contains(&key) {
                self.split_constraint(a, b, constrained_edges)?;
                return Ok(());
            }
        }
        let (a, b) = best.map(|(e, _)| e).unwrap_or((t.v[0], t.v[1]));
        // Opposite vertex (the one not on the chosen edge).
        let opp = *t.v.iter().find(|&&v| v != a && v != b).unwrap();
        let edge_mid = (self.points[a].coords + self.points[b].coords) * 0.5;
        let toward_opp = self.points[opp].coords - edge_mid;
        // 1e-3 of the way to the opposite vertex: safely interior, yet
        // still essentially bisecting the long edge.
        let p = Point2::from(edge_mid + toward_opp * 1e-3);
        let new_idx = self.points.len();
        self.points.push(p);
        self.insert_point_constrained_seeded(new_idx, Some(t_idx), constrained_edges)?;
        Ok(())
    }

    /// True iff some alive triangle has exactly the vertex set `v` (in any
    /// rotation/orientation).
    fn triangle_alive_with_vertices(&self, v: [usize; 3]) -> bool {
        let mut target = v;
        target.sort_unstable();
        self.triangles.iter().any(|t| {
            if !t.alive {
                return false;
            }
            let mut w = t.v;
            w.sort_unstable();
            w == target
        })
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

    /// Classify every triangle as "outside" or "inside" the region
    /// enclosed by the constrained edges.
    ///
    /// The constrained edges form the closed loops of the outer boundary
    /// and every hole. A triangle is **inside** iff its centroid lies
    /// inside the outer loop and outside every hole — decided by the
    /// even-odd (ray-crossing) rule against the constrained edges taken
    /// as a soup of line segments: a horizontal ray cast from the
    /// centroid crosses an odd number of constrained edges iff the
    /// centroid is inside.
    ///
    /// This is a purely geometric test, independent of the triangle
    /// adjacency graph. An earlier version flood-filled a two-colouring
    /// across the constraints, but that is fragile: refinement (boundary
    /// edge splitting, Steiner insertion) can leave a constrained loop
    /// that is not a clean wall in the neighbour graph, and the parity
    /// then leaks into interior pockets (or out through the boundary).
    ///
    /// Returns a `Vec<bool>` of length `self.triangles.len()` where
    /// `true` marks a triangle to drop (outside the outer loop, inside a
    /// hole, dead, or still touching the super-triangle).
    fn flood_fill_outside(
        &self,
        constrained_edges: &std::collections::BTreeSet<(usize, usize)>,
    ) -> Vec<bool> {
        let n = self.triangles.len();
        let mut outside = vec![true; n];

        for (idx, t) in self.triangles.iter().enumerate() {
            if !t.alive {
                continue; // dead = drop
            }
            // Super-triangle sentinels live at indices [n_input, n_input + 3);
            // anything past that is a Steiner point added by refinement.
            if t.v
                .iter()
                .any(|&v| v >= self.n_input && v < self.n_input + 3)
            {
                continue; // touches the super-triangle = drop
            }
            let a = self.points[t.v[0]];
            let b = self.points[t.v[1]];
            let c = self.points[t.v[2]];
            let centroid = Point2::from((a.coords + b.coords + c.coords) / 3.0);
            outside[idx] = !self.centroid_inside_constraints(centroid, constrained_edges);
        }

        outside
    }

    /// Even-odd test: `true` iff `p` lies inside the region bounded by
    /// the constrained edges. A ray is cast in the `+x` direction and the
    /// number of constrained edges it crosses is counted modulo two.
    fn centroid_inside_constraints(
        &self,
        p: Point2,
        constrained_edges: &std::collections::BTreeSet<(usize, usize)>,
    ) -> bool {
        let mut inside = false;
        for &(a, b) in constrained_edges {
            let pa = self.points[a];
            let pb = self.points[b];
            // Half-open in y ([min, max)) so a vertex shared by two edges is
            // counted exactly once; guards against the ray grazing a vertex.
            let (y0, y1) = (pa.y, pb.y);
            if (y0 <= p.y) != (y1 <= p.y) {
                // x-coordinate of the edge at height p.y.
                let t = (p.y - y0) / (y1 - y0);
                let x_cross = pa.x + t * (pb.x - pa.x);
                if x_cross > p.x {
                    inside = !inside;
                }
            }
        }
        inside
    }

    /// Return every triangle judged to be **inside** the polygon
    /// defined by the constrained edges (i.e. neither outside the
    /// outer loop nor inside any hole).
    pub(super) fn extract_interior_with_constraints(
        &self,
        constrained_edges: &std::collections::BTreeSet<(usize, usize)>,
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

/// True iff triangle `t` has both `a` and `b` as vertices (i.e. `(a, b)`
/// is one of its three edges).
#[inline]
fn edge_in_triangle(t: &Triangle, a: usize, b: usize) -> bool {
    t.v.contains(&a) && t.v.contains(&b)
}

/// Circumcenter of the triangle `(a, b, c)`. Returns `None` if the
/// three vertices are (nearly) collinear.
fn circumcenter(a: Point2, b: Point2, c: Point2) -> Option<Point2> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
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
    va.x * (vb.y * cm - bm * vc.y) - va.y * (vb.x * cm - bm * vc.x)
        + am * (vb.x * vc.y - vb.y * vc.x)
}

/// `in_circle(a, b, c, d) > 0`, but tolerant of points that are meant to be
/// exactly cocircular — `circle`/`arc`-built contours routinely hand the
/// triangulator dozens of points sampled on the very same circle. In exact
/// arithmetic every triangle formed from three such points has that same
/// circle as its own circumcircle, so a fourth one is mathematically
/// borderline (`in_circle == 0`); plain `f64` rounding then decides the sign
/// arbitrarily, triangle by triangle. Left as a strict `> 0.0` test, this
/// turns the *initial* Delaunay triangulation of a cocircular point set into
/// an essentially random pick among many degenerate options — often skinny —
/// and later drives `refine` into a cascade of near-useless single-triangle
/// insertions trying to fix them up one at a time. Treating "borderline" as
/// "inside" instead makes Bowyer-Watson grow the full cavity across a
/// cocircular cluster in one step, the well-conditioned choice.
#[inline]
fn in_circle_tolerant(a: Point2, b: Point2, c: Point2, d: Point2) -> bool {
    let scale = a
        .coords
        .norm_squared()
        .max(b.coords.norm_squared())
        .max(c.coords.norm_squared())
        .max(d.coords.norm_squared())
        .max(1.0);
    in_circle(a, b, c, d) > -1e-9 * scale * scale
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
/// use pyrucast::atoms::Point2;
/// use pyrucast::ops::mesh::triangulation::delaunay_2d;
///
/// let pts = vec![
///     Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0), Point2::new(0.0, 1.0),
/// ];
/// let tris = delaunay_2d(&pts).unwrap();
/// assert_eq!(tris.len(), 2);
/// ```
///
/// ```
/// # use pyrucast::atoms::Point2;
/// # use pyrucast::ops::mesh::triangulation;
/// // L'enveloppe convexe d'un carré : deux triangles, quel que soit le
/// // découpage choisi par la condition de Delaunay.
/// let carre = [Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
///              Point2::new(1.0, 1.0), Point2::new(0.0, 1.0)];
/// assert_eq!(triangulation::delaunay_2d(&carre)?.len(), 2);
/// // Moins de trois points : rien à trianguler.
/// assert!(triangulation::delaunay_2d(&carre[..2]).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
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
/// at this layer — that is the job of the caller (`ops::mesh::triangulate_surface`
/// in the hole-removal step).
///
/// # Errors
/// - same as [`delaunay_2d`] for the point set,
/// - a constraint references a point index outside `0..points.len()`,
/// - a constraint cannot be enforced (e.g. its segment lies on the hull
///   in a way the walk cannot follow, or it crosses another already-
///   forced constraint).
///
/// ```
/// # use pyrucast::atoms::Point2;
/// # use pyrucast::ops::mesh::triangulation;
/// // Une contrainte force une arête à survivre à la triangulation, même
/// // si Delaunay seul aurait choisi l'autre diagonale.
/// let carre = [Point2::new(0.0, 0.0), Point2::new(1.0, 0.0),
///              Point2::new(1.0, 1.0), Point2::new(0.0, 1.0)];
/// let cells = triangulation::constrained_delaunay_2d(&carre, &[(1, 3)])?;
/// assert_eq!(cells.len(), 2);
/// // La diagonale (1,3) est bien une arête d'un des deux triangles.
/// assert!(cells.iter().any(|t| t.contains(&1) && t.contains(&3)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
///
/// ```
/// # use pyrucast::atoms::Point2;
/// # use pyrucast::ops::mesh::triangulation;
/// // Un carré percé d'un carré : la couronne est maillée, le trou reste
/// // vide. Huit sommets, huit triangles.
/// let outer = vec![Point2::new(0.0, 0.0), Point2::new(3.0, 0.0),
///                  Point2::new(3.0, 3.0), Point2::new(0.0, 3.0)];
/// let hole = vec![Point2::new(1.0, 1.0), Point2::new(2.0, 1.0),
///                 Point2::new(2.0, 2.0), Point2::new(1.0, 2.0)];
/// let cells = triangulation::triangulate_polygon_with_holes(&outer, &[hole])?;
/// assert_eq!(cells.len(), 8);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    let mut edge_set: std::collections::BTreeSet<(usize, usize)> =
        std::collections::BTreeSet::new();
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
///
/// ```
/// # use pyrucast::atoms::Point2;
/// # use pyrucast::ops::mesh::triangulation;
/// # use pyrucast::ops::mesh::triangulation::RefinementOptions;
/// // Le raffinement **ajoute des points** : la fonction rend donc les
/// // sommets en plus des mailles, contrairement à sa version brute.
/// let outer = vec![Point2::new(0.0, 0.0), Point2::new(3.0, 0.0),
///                  Point2::new(3.0, 3.0), Point2::new(0.0, 3.0)];
/// let (pts, cells) = triangulation::triangulate_polygon_with_holes_refined(
///     &outer, &[], RefinementOptions { max_edge_length: Some(0.5),
///                                      min_angle_deg: Some(20.0) })?;
/// assert!(pts.len() > outer.len()); // des points de Steiner ont été posés
/// assert!(cells.len() > 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    if let Some(h) = options.max_edge_length
        && (h.is_nan() || h <= 0.0)
    {
        return Err(PyrucastError::Message(format!(
            "triangulate_polygon_with_holes_refined: max_edge_length must be > 0, got {}",
            h
        )));
    }
    if let Some(a) = options.min_angle_deg
        && !(a > 0.0 && a < 60.0)
    {
        return Err(PyrucastError::Message(format!(
            "triangulate_polygon_with_holes_refined: min_angle_deg must be in (0, 60), got {}",
            a
        )));
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
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();
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
            e.iter()
                .any(|&(p, q)| (p == a && q == b) || (p == b && q == a))
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
        // A rectangle's two diagonals are equal length, so its 4 corners are
        // exactly cocircular: either diagonal is an equally valid choice for
        // the unconstrained triangulation (which one comes out is
        // floating-point tie-breaking, not a quality difference). The test
        // forces whichever one didn't come out on its own.
        let has_02 = triangulation_has_edge(&unconstrained, 0, 2);
        let has_13 = triangulation_has_edge(&unconstrained, 1, 3);
        assert!(
            has_02 ^ has_13,
            "expected exactly one diagonal, got (0,2)={has_02} (1,3)={has_13}"
        );
        let (forced_a, forced_b) = if has_02 { (1, 3) } else { (0, 2) };

        let constrained = constrained_delaunay_2d(&pts, &[(forced_a, forced_b)]).unwrap();
        assert_eq!(constrained.len(), 2);
        assert!(
            triangulation_has_edge(&constrained, forced_a, forced_b),
            "forced edge ({forced_a}, {forced_b}) missing: {:?}",
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
        let (pts, tris) = triangulate_polygon_with_holes_refined(&outer, &[hole], opts).unwrap();
        assert_all_ccw(&tris, &pts);
        assert!(max_edge_length(&tris, &pts) <= 1.0 + 1e-9);
        // Total area = 16 - 4 = 12.
        assert!((total_area(&tris, &pts) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn refine_cocircular_boundary_converges() {
        // Every point of a regular polygon lies on the same circle by
        // construction — exactly what `circle`/`arc`-built contours hand the
        // mesher. Any three of them share that circle as their circumcircle,
        // so `in_circle` is mathematically borderline for a fourth: without
        // tolerance for that, the initial Delaunay triangulation used to
        // come out arbitrarily skinny and drive `refine` into thousands of
        // near-useless single-triangle insertions instead of converging.
        let n = 20;
        let r = 1.0;
        let outer: Vec<Point2> = (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                p2(r * t.cos(), r * t.sin())
            })
            .collect();
        let opts = RefinementOptions {
            max_edge_length: Some(0.3),
            min_angle_deg: None,
        };
        let (pts, tris) = triangulate_polygon_with_holes_refined(&outer, &[], opts).unwrap();
        assert_all_ccw(&tris, &pts);
        assert!(max_edge_length(&tris, &pts) <= 0.3 + 1e-9);
        let poly_area = 0.5 * n as f64 * r * r * (2.0 * std::f64::consts::PI / n as f64).sin();
        assert!(
            (total_area(&tris, &pts) - poly_area).abs() < 1e-6,
            "area drift: got {}, expected {}",
            total_area(&tris, &pts),
            poly_area
        );
    }

    #[test]
    fn refine_plate_with_hole_stays_inside_contour() {
        // Reproduces the `maillage_test.py` geometry: a plate whose right
        // edge is a half-circle bulge (radius 0.05 around (0.3,0.05)), with a
        // concentric circular hole (radius 0.035). Both boundaries are built
        // from cocircular arc/circle points. The refined mesh must stay
        // strictly inside the outer loop and outside the hole.
        let cx = 0.30_f64;
        let cy = 0.05_f64;
        let r_arc = 0.05_f64;
        let r_hole = 0.035_f64;
        let mut outer: Vec<Point2> = vec![p2(0.0, 0.0), p2(0.30, 0.0)];
        for (a0, a1) in [(-90.0_f64, 0.0_f64), (0.0, 90.0)] {
            for i in 1..=6 {
                let t = i as f64 / 6.0;
                let phi = (a0 + t * (a1 - a0)).to_radians();
                outer.push(p2(cx + r_arc * phi.cos(), cy + r_arc * phi.sin()));
            }
        }
        outer.push(p2(0.0, 0.10));
        let hole: Vec<Point2> = (0..10)
            .map(|i| {
                let phi = (i as f64 / 10.0) * std::f64::consts::TAU;
                p2(cx + r_hole * phi.cos(), cy + r_hole * phi.sin())
            })
            .collect();

        let opts = RefinementOptions {
            max_edge_length: Some(0.025),
            min_angle_deg: None,
        };
        let (pts, tris) =
            triangulate_polygon_with_holes_refined(&outer, std::slice::from_ref(&hole), opts)
                .unwrap();
        assert_all_ccw(&tris, &pts);

        // Total triangle area must equal outer polygon area minus hole area.
        let poly_area = |loop_pts: &[Point2]| -> f64 {
            let n = loop_pts.len();
            let mut a = 0.0;
            for i in 0..n {
                let p = loop_pts[i];
                let q = loop_pts[(i + 1) % n];
                a += p.x * q.y - q.x * p.y;
            }
            0.5 * a.abs()
        };
        let expected = poly_area(&outer) - poly_area(&hole);
        let got = total_area(&tris, &pts);
        assert!(
            (got - expected).abs() < 1e-9,
            "area mismatch: got {}, expected {} (spills outside contour / into hole)",
            got,
            expected
        );
    }

    /// Build the `maillage_test.py` plate + concentric hole with each
    /// straight side pre-discretized into `n_side` segments (as
    /// `line(.., n_side)` produces), refine to `max_edge_length = 0.025`,
    /// and check the result stays inside the contour (area = outer − hole).
    fn check_plate_with_hole_refines(n_side: usize) {
        let cx = 0.30_f64;
        let cy = 0.05_f64;
        let r_arc = 0.05_f64;
        let r_hole = 0.035_f64;

        // Each vertex appears exactly once (as `mesh::consolidate` merges the
        // shared endpoints of adjacent edges): every side contributes its
        // start point and its interior points but not its end point, which
        // is the next side's start.
        let mut outer: Vec<Point2> = Vec::new();
        // Bottom edge p1(0,0) → p2(0.3,0).
        for i in 0..n_side {
            let t = i as f64 / n_side as f64;
            outer.push(p2(0.3 * t, 0.0));
        }
        // Two arcs around (cx, cy): p2 → p3 → p4, 6 segments each. Start p2
        // is already in the list; the last pushed point is p4, dropped
        // below so the top edge re-adds it exactly once.
        for (a0, a1) in [(-90.0_f64, 0.0_f64), (0.0, 90.0)] {
            for i in 1..=6 {
                let t = i as f64 / 6.0;
                let phi = (a0 + t * (a1 - a0)).to_radians();
                outer.push(p2(cx + r_arc * phi.cos(), cy + r_arc * phi.sin()));
            }
        }
        outer.pop(); // drop p4; the top edge re-introduces it.
                     // Top edge p4(0.3,0.1) → p5(0,0.1).
        for i in 0..n_side {
            let t = i as f64 / n_side as f64;
            outer.push(p2(0.3 - 0.3 * t, 0.1));
        }
        // Left edge p5(0,0.1) → p1(0,0).
        for i in 0..n_side {
            let t = i as f64 / n_side as f64;
            outer.push(p2(0.0, 0.1 - 0.1 * t));
        }

        let hole: Vec<Point2> = (0..10)
            .map(|i| {
                let phi = (i as f64 / 10.0) * std::f64::consts::TAU;
                p2(cx + r_hole * phi.cos(), cy + r_hole * phi.sin())
            })
            .collect();

        let opts = RefinementOptions {
            max_edge_length: Some(0.025),
            min_angle_deg: None,
        };
        let (pts, tris) =
            triangulate_polygon_with_holes_refined(&outer, std::slice::from_ref(&hole), opts)
                .unwrap_or_else(|e| panic!("n_side={n_side}: {e}"));
        assert_all_ccw(&tris, &pts);

        let poly_area = |loop_pts: &[Point2]| -> f64 {
            let n = loop_pts.len();
            let mut a = 0.0;
            for i in 0..n {
                let p = loop_pts[i];
                let q = loop_pts[(i + 1) % n];
                a += p.x * q.y - q.x * p.y;
            }
            0.5 * a.abs()
        };
        let expected = poly_area(&outer) - poly_area(&hole);
        let got = total_area(&tris, &pts);
        assert!(
            (got - expected).abs() < 1e-9,
            "n_side={n_side}: area mismatch: got {got}, expected {expected}",
        );
    }

    #[test]
    fn refine_plate_with_hole_finely_discretized_contour_converges() {
        // Same plate + hole as `refine_plate_with_hole_stays_inside_contour`,
        // but with the straight sides pre-discretized (as `line(.., n)`
        // produces) — many boundary vertices far finer than the 0.025
        // target. The refiner must converge and stay inside the contour at
        // every discretization, not explode or fail the constraint walk.
        for n_side in [10usize, 20, 40] {
            check_plate_with_hole_refines(n_side);
        }
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
