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
use crate::triangulation::Point2;
use std::collections::HashMap;

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
}

/// Bounding super-triangle of an input point set. Returns three
/// vertices large enough to enclose every point with margin.
fn super_triangle(points: &[Point2]) -> [Point2; 3] {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in points {
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let dx = (max_x - min_x).max(1.0);
    let dy = (max_y - min_y).max(1.0);
    let dmax = dx.max(dy);
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    // A wide isoceles triangle sitting under the centre. 20× the AABB
    // diagonal is overkill for robustness but trivial in cost.
    let r = 20.0 * dmax;
    [(cx - r, cy - r), (cx + r, cy - r), (cx, cy + 2.0 * r)]
}

/// Sign of the in-circle predicate: `> 0` means `d` lies **inside** the
/// circumcircle of `(a, b, c)`, assuming `(a, b, c)` is CCW.
#[inline]
fn in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let ax = a.0 - d.0;
    let ay = a.1 - d.1;
    let bx = b.0 - d.0;
    let by = b.1 - d.1;
    let cx = c.0 - d.0;
    let cy = c.1 - d.1;
    let am = ax * ax + ay * ay;
    let bm = bx * bx + by * by;
    let cm = cx * cx + cy * cy;
    ax * (by * cm - bm * cy) - ay * (bx * cm - bm * cx) + am * (bx * cy - by * cx)
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
/// use pyrucast::triangulation::delaunay_2d;
///
/// let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
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
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            if dx * dx + dy * dy < 1e-24 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_area(a: Point2, b: Point2, c: Point2) -> f64 {
        0.5 * ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0))
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
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let tris = delaunay_2d(&pts).unwrap();
        assert_eq!(tris.len(), 1);
        assert_all_ccw(&tris, &pts);
    }

    #[test]
    fn delaunay_square_two_triangles() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
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
                (t.cos(), t.sin())
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
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
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
                pts.push((i as f64, j as f64));
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
        let pts = vec![(0.0, 0.0), (1.0, 0.0)];
        assert!(delaunay_2d(&pts).is_err());
    }

    #[test]
    fn delaunay_rejects_coincident_points() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (1.0, 0.0)];
        assert!(delaunay_2d(&pts).is_err());
    }

    #[test]
    fn delaunay_satisfies_empty_circle_property() {
        // Pick a small random-ish point set and verify that no point
        // sits inside any triangle's circumcircle. We use the same
        // in_circle predicate as the algorithm — this is a coherence
        // check, not a proof of correctness, but catches gross errors.
        let pts = vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 3.0),
            (0.0, 3.0),
            (1.0, 1.0),
            (3.0, 2.0),
            (2.0, 2.5),
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
