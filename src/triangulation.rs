//! 2-D triangulation primitives.
//!
//! Pure-geometry helpers shared by mesh builders. The functions here
//! operate on plain arrays of 2-D points and integer indices; they do
//! not know about [`crate::mesh::Mesh`] or the refcount machinery —
//! that wiring lives in [`crate::mesh`].
//!
//! For now only one algorithm is exposed: [`ear_clip_2d`], the classic
//! ear-clipping triangulation of a **simple closed polygon** without
//! holes and without Steiner points. It is the simplest building block;
//! later iterations of the meshing pipeline (holes, refinement) will
//! layer on top of it.

use crate::error::{PyrucastError, Result};

/// 2-D point as `(x, y)`.
pub type Point2 = (f64, f64);

/// Signed area of the polygon `points`, taken as a closed loop
/// (`points[n-1]` connects back to `points[0]`; the last vertex must not
/// repeat the first).
///
/// Returns a **positive** value for a counter-clockwise polygon, a
/// negative value for clockwise, and a value close to zero for a
/// degenerate (collinear) one. Uses the shoelace formula.
pub fn signed_area(points: &[Point2]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % n];
        s += x1 * y2 - x2 * y1;
    }
    0.5 * s
}

/// Cross product `(b - a) × (c - a)` in 2-D.
#[inline]
fn cross2(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// True if `p` lies in the closed triangle `(a, b, c)`. Robust to the
/// triangle's winding (CW or CCW): the test checks that `p` is on the
/// same side of every edge.
fn point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Triangulate a simple closed polygon by ear clipping.
///
/// `points[i]` is the i-th vertex of the polygon in order; edge `i`
/// joins `points[i]` to `points[i + 1]`, and the loop closes from
/// `points[n - 1]` to `points[0]`. The polygon must be **simple**
/// (no self-intersection); its orientation can be CW or CCW —
/// the function detects it and normalises internally.
///
/// Returns exactly `n - 2` triangles as triples of **indices into the
/// input `points`**. Triangles are always oriented **CCW** in the
/// plane, regardless of the input orientation.
///
/// # Errors
/// - `points.len() < 3`,
/// - polygon has zero (or near-zero) signed area — degenerate / collinear,
/// - no ear can be clipped (typically a non-simple polygon).
///
/// # Example
/// ```
/// use pyrucast::triangulation::ear_clip_2d;
///
/// // Unit square, CCW.
/// let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
/// let tris = ear_clip_2d(&pts).unwrap();
/// assert_eq!(tris.len(), 2);
/// ```
pub fn ear_clip_2d(points: &[Point2]) -> Result<Vec<[usize; 3]>> {
    let n = points.len();
    if n < 3 {
        return Err(PyrucastError::Message(format!(
            "ear_clip_2d: polygon must have ≥ 3 vertices, got {}",
            n
        )));
    }
    let area = signed_area(points);
    if area.abs() < 1e-15 {
        return Err(PyrucastError::Message(
            "ear_clip_2d: polygon has zero (or near-zero) signed area".into(),
        ));
    }

    // Work on a CCW view of the indices: if the input is CW, reverse
    // the index sequence so the algorithm always sees CCW. The output
    // triangles built from these indices are then CCW in the plane.
    let mut active: Vec<usize> = if area < 0.0 {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };

    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n - 2);

    while active.len() > 3 {
        let m = active.len();
        let mut ear: Option<usize> = None;
        for i in 0..m {
            let ip = (i + m - 1) % m;
            let in_ = (i + 1) % m;
            let a = points[active[ip]];
            let b = points[active[i]];
            let c = points[active[in_]];
            // Convex vertex in CCW ⇔ cross2(a, b, c) > 0.
            if cross2(a, b, c) <= 0.0 {
                continue;
            }
            // No other vertex of the polygon must lie inside the candidate ear.
            let mut contains = false;
            for (j, &idx) in active.iter().enumerate() {
                if j == ip || j == i || j == in_ {
                    continue;
                }
                if point_in_triangle(points[idx], a, b, c) {
                    contains = true;
                    break;
                }
            }
            if !contains {
                ear = Some(i);
                break;
            }
        }
        let Some(ear_i) = ear else {
            return Err(PyrucastError::Message(
                "ear_clip_2d: no ear found — polygon is non-simple or degenerate".into(),
            ));
        };
        let ip = (ear_i + m - 1) % m;
        let in_ = (ear_i + 1) % m;
        triangles.push([active[ip], active[ear_i], active[in_]]);
        active.remove(ear_i);
    }
    // Last remaining triangle.
    triangles.push([active[0], active[1], active[2]]);
    Ok(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {} ≈ {} (tol={})", a, b, tol);
    }

    #[test]
    fn signed_area_unit_square_ccw() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert_close(signed_area(&pts), 1.0, 1e-12);
    }

    #[test]
    fn signed_area_unit_square_cw() {
        let pts = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        assert_close(signed_area(&pts), -1.0, 1e-12);
    }

    #[test]
    fn signed_area_degenerate() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert!(signed_area(&pts).abs() < 1e-12);
    }

    #[test]
    fn ear_clip_triangle() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0], [0, 1, 2]);
    }

    #[test]
    fn ear_clip_square_ccw_gives_two_triangles() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 2);
        // Every triangle must be CCW (positive area).
        for [i, j, k] in &tris {
            let area = cross2(pts[*i], pts[*j], pts[*k]);
            assert!(area > 0.0, "triangle {:?} not CCW", [i, j, k]);
        }
    }

    #[test]
    fn ear_clip_square_cw_still_ccw_output() {
        // Same vertices but listed CW.
        let pts = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 2);
        for [i, j, k] in &tris {
            let area = cross2(pts[*i], pts[*j], pts[*k]);
            assert!(area > 0.0, "triangle {:?} not CCW", [i, j, k]);
        }
    }

    #[test]
    fn ear_clip_concave_l_shape() {
        // L-shaped polygon, 6 vertices, CCW.
        //   (0,3)─(1,3)
        //     │     │
        //     │   (1,1)─(3,1)
        //     │             │
        //   (0,0)──────── (3,0)
        let pts = vec![
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 4); // n - 2

        // Total area must equal the L-shape area (5).
        let mut total = 0.0;
        for [i, j, k] in &tris {
            total += 0.5 * cross2(pts[*i], pts[*j], pts[*k]);
        }
        assert_close(total, 5.0, 1e-12);

        // Every vertex of the polygon must appear at least once.
        let mut used = [false; 6];
        for [i, j, k] in &tris {
            used[*i] = true;
            used[*j] = true;
            used[*k] = true;
        }
        assert!(used.iter().all(|&u| u));
    }

    #[test]
    fn ear_clip_rejects_too_few_vertices() {
        assert!(ear_clip_2d(&[(0.0, 0.0), (1.0, 0.0)]).is_err());
    }

    #[test]
    fn ear_clip_rejects_degenerate_polygon() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        assert!(ear_clip_2d(&pts).is_err());
    }

    #[test]
    fn ear_clip_pentagon_uses_all_vertices_once() {
        // Regular pentagon, CCW.
        let n = 5;
        let mut pts: Vec<Point2> = Vec::with_capacity(n);
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            pts.push((t.cos(), t.sin()));
        }
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), n - 2);
        // Sum of triangle areas ≈ pentagon area (≈ 2.3776 for unit-circle pentagon).
        let mut total = 0.0;
        for [i, j, k] in &tris {
            total += 0.5 * cross2(pts[*i], pts[*j], pts[*k]);
        }
        let expected = 0.5 * (n as f64) * (2.0 * std::f64::consts::PI / n as f64).sin();
        assert_close(total, expected, 1e-10);
    }
}
