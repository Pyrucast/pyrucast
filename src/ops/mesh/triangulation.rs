//! 2-D / 3-D geometry primitives shared by the mesh builders.
//!
//! All vector / point arithmetic in this module is built on
//! [`nalgebra`]: 2-D points are [`nalgebra::Point2<f64>`], 3-D points
//! are [`nalgebra::Point3<f64>`], and free vectors are
//! [`nalgebra::Vector2`] / [`nalgebra::Vector3`]. The canonical aliases
//! ([`crate::atoms::Point2`], [`crate::atoms::Point3`],
//! [`crate::atoms::Vector2`], [`crate::atoms::Vector3`])
//! live under [`crate::atoms::point`] and are used here directly.
//!
//! The module exposes:
//! - [`signed_area`] / [`ear_clip_2d`] — ear clipping triangulation of
//!   a simple closed polygon,
//! - [`newell_normal`] / [`in_plane_basis`] — best-fit plane utilities
//!   for nearly-planar 3-D polygons,
//! - [`delaunay_2d`], [`constrained_delaunay_2d`],
//!   [`triangulate_polygon_with_holes`] — full constrained Delaunay
//!   pipeline (Bowyer-Watson + edge enforcement + parity flood-fill,
//!   in a private `cdt` sub-module).

use crate::error::{PyrucastError, Result};

mod cdt;
pub use cdt::{
    constrained_delaunay_2d, delaunay_2d, triangulate_polygon_with_holes,
    triangulate_polygon_with_holes_refined, RefinementOptions,
};

use crate::atoms::{Point2, Point3, Vector3};

/// Unit normal of a 3-D polygon by **Newell's method**.
///
/// The polygon is given as an ordered list of vertices `points`, with
/// the loop closing implicitly from `points[n - 1]` back to `points[0]`.
/// Returns `None` if the polygon is degenerate (collinear / zero area)
/// or has fewer than 3 vertices.
///
/// The method sums signed components of consecutive edges in a way that
/// is robust to small departures from planarity: the magnitude of the
/// raw sum equals **twice the area projected onto each axis plane**, so
/// the dominant terms come from the largest-area projection.
///
/// # Example
/// ```
/// use pyrucast::atoms::Point3;
/// use pyrucast::ops::mesh::triangulation::newell_normal;
///
/// // Unit square in the plane z = 0, CCW seen from +z.
/// let pts = vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 0.0, 0.0),
///     Point3::new(1.0, 1.0, 0.0),
///     Point3::new(0.0, 1.0, 0.0),
/// ];
/// let n = newell_normal(&pts).unwrap();
/// assert!((n.z - 1.0).abs() < 1e-12);
/// ```
pub fn newell_normal(points: &[Point3]) -> Option<Vector3> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    let mut nrm = Vector3::zeros();
    for i in 0..n {
        let p = &points[i];
        let q = &points[(i + 1) % n];
        nrm.x += (p.y - q.y) * (p.z + q.z);
        nrm.y += (p.z - q.z) * (p.x + q.x);
        nrm.z += (p.x - q.x) * (p.y + q.y);
    }
    let mag = nrm.norm();
    if mag < 1e-15 {
        return None;
    }
    Some(nrm / mag)
}

/// Orthonormal in-plane basis `(u, v)` such that `(u, v, normal)` is
/// right-handed, given a **unit** normal.
///
/// Built by Gram-Schmidt against the coordinate axis least aligned with
/// `normal`. The choice of axis is deterministic — same input ⇒ same
/// basis — but otherwise arbitrary; do not rely on `u` or `v` having a
/// specific direction.
pub fn in_plane_basis(normal: Vector3) -> (Vector3, Vector3) {
    let abs_n = normal.map(|x| x.abs());
    let e: Vector3 = if abs_n.x <= abs_n.y && abs_n.x <= abs_n.z {
        Vector3::x()
    } else if abs_n.y <= abs_n.z {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let u = (e - normal * e.dot(&normal)).normalize();
    let v = normal.cross(&u);
    (u, v)
}

/// Signed area of the polygon `points`, taken as a closed loop
/// (`points[n-1]` connects back to `points[0]`; the last vertex must
/// not repeat the first).
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
        let p = &points[i];
        let q = &points[(i + 1) % n];
        s += p.x * q.y - q.x * p.y;
    }
    0.5 * s
}

/// Cross product `(b - a) × (c - a)` in 2-D — the *perp dot* product.
#[inline]
pub(crate) fn cross2(a: Point2, b: Point2, c: Point2) -> f64 {
    (b - a).perp(&(c - a))
}

/// True if `p` lies in the closed triangle `(a, b, c)`. Robust to the
/// triangle's winding (CW or CCW).
pub(crate) fn point_in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
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
/// use pyrucast::atoms::Point2;
/// use pyrucast::ops::mesh::triangulation::ear_clip_2d;
///
/// // Unit square, CCW.
/// let pts = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(1.0, 0.0),
///     Point2::new(1.0, 1.0),
///     Point2::new(0.0, 1.0),
/// ];
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
    // the index sequence so the algorithm always sees CCW.
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
            if cross2(a, b, c) <= 0.0 {
                continue;
            }
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
    triangles.push([active[0], active[1], active[2]]);
    Ok(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {} ≈ {} (tol={})", a, b, tol);
    }

    fn p2(x: f64, y: f64) -> Point2 {
        Point2::new(x, y)
    }
    fn p3(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }

    #[test]
    fn signed_area_unit_square_ccw() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        assert_close(signed_area(&pts), 1.0, 1e-12);
    }

    #[test]
    fn signed_area_unit_square_cw() {
        let pts = vec![p2(0.0, 0.0), p2(0.0, 1.0), p2(1.0, 1.0), p2(1.0, 0.0)];
        assert_close(signed_area(&pts), -1.0, 1e-12);
    }

    #[test]
    fn signed_area_degenerate() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(2.0, 0.0)];
        assert!(signed_area(&pts).abs() < 1e-12);
    }

    #[test]
    fn ear_clip_triangle() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(0.0, 1.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0], [0, 1, 2]);
    }

    #[test]
    fn ear_clip_square_ccw_gives_two_triangles() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(1.0, 1.0), p2(0.0, 1.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 2);
        for [i, j, k] in &tris {
            let area = cross2(pts[*i], pts[*j], pts[*k]);
            assert!(area > 0.0, "triangle {:?} not CCW", [i, j, k]);
        }
    }

    #[test]
    fn ear_clip_square_cw_still_ccw_output() {
        let pts = vec![p2(0.0, 0.0), p2(0.0, 1.0), p2(1.0, 1.0), p2(1.0, 0.0)];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 2);
        for [i, j, k] in &tris {
            let area = cross2(pts[*i], pts[*j], pts[*k]);
            assert!(area > 0.0, "triangle {:?} not CCW", [i, j, k]);
        }
    }

    #[test]
    fn ear_clip_concave_l_shape() {
        let pts = vec![
            p2(0.0, 0.0),
            p2(3.0, 0.0),
            p2(3.0, 1.0),
            p2(1.0, 1.0),
            p2(1.0, 3.0),
            p2(0.0, 3.0),
        ];
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), 4);

        let mut total = 0.0;
        for [i, j, k] in &tris {
            total += 0.5 * cross2(pts[*i], pts[*j], pts[*k]);
        }
        assert_close(total, 5.0, 1e-12);

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
        assert!(ear_clip_2d(&[p2(0.0, 0.0), p2(1.0, 0.0)]).is_err());
    }

    #[test]
    fn ear_clip_rejects_degenerate_polygon() {
        let pts = vec![p2(0.0, 0.0), p2(1.0, 0.0), p2(2.0, 0.0)];
        assert!(ear_clip_2d(&pts).is_err());
    }

    #[test]
    fn newell_unit_square_in_xy_plane() {
        let pts = vec![
            p3(0.0, 0.0, 0.0),
            p3(1.0, 0.0, 0.0),
            p3(1.0, 1.0, 0.0),
            p3(0.0, 1.0, 0.0),
        ];
        let n = newell_normal(&pts).unwrap();
        assert_close(n.x, 0.0, 1e-12);
        assert_close(n.y, 0.0, 1e-12);
        assert_close(n.z, 1.0, 1e-12);
    }

    #[test]
    fn newell_square_in_xz_plane() {
        let pts = vec![
            p3(0.0, 0.0, 0.0),
            p3(1.0, 0.0, 0.0),
            p3(1.0, 0.0, 1.0),
            p3(0.0, 0.0, 1.0),
        ];
        let n = newell_normal(&pts).unwrap();
        assert_close(n.x, 0.0, 1e-12);
        assert_close(n.y, -1.0, 1e-12);
        assert_close(n.z, 0.0, 1e-12);
    }

    #[test]
    fn newell_translated_polygon_same_normal() {
        let p1 = vec![p3(0.0, 0.0, 0.0), p3(1.0, 0.0, 0.0), p3(0.0, 1.0, 0.0)];
        let p2: Vec<Point3> = p1
            .iter()
            .map(|p| p3(p.x + 10.0, p.y - 5.0, p.z + 3.0))
            .collect();
        let n1 = newell_normal(&p1).unwrap();
        let n2 = newell_normal(&p2).unwrap();
        assert!((n1 - n2).norm() < 1e-12);
    }

    #[test]
    fn newell_rejects_collinear_polygon() {
        let pts = vec![p3(0.0, 0.0, 0.0), p3(1.0, 0.0, 0.0), p3(2.0, 0.0, 0.0)];
        assert!(newell_normal(&pts).is_none());
    }

    #[test]
    fn newell_rejects_too_few_vertices() {
        assert!(newell_normal(&[p3(0.0, 0.0, 0.0), p3(1.0, 0.0, 0.0)]).is_none());
    }

    #[test]
    fn in_plane_basis_is_orthonormal_right_handed() {
        let normals = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 2.0, 3.0).normalize(),
        ];
        for n in normals {
            let (u, v) = in_plane_basis(n);
            assert_close(u.norm(), 1.0, 1e-12);
            assert_close(v.norm(), 1.0, 1e-12);
            assert_close(u.dot(&v), 0.0, 1e-12);
            assert_close(u.dot(&n), 0.0, 1e-12);
            assert_close(v.dot(&n), 0.0, 1e-12);
            // Right-handed: u × v == n.
            assert!((u.cross(&v) - n).norm() < 1e-12);
        }
    }

    #[test]
    fn ear_clip_pentagon_uses_all_vertices_once() {
        let n = 5;
        let mut pts: Vec<Point2> = Vec::with_capacity(n);
        for i in 0..n {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            pts.push(p2(t.cos(), t.sin()));
        }
        let tris = ear_clip_2d(&pts).unwrap();
        assert_eq!(tris.len(), n - 2);
        let mut total = 0.0;
        for [i, j, k] in &tris {
            total += 0.5 * cross2(pts[*i], pts[*j], pts[*k]);
        }
        let expected = 0.5 * (n as f64) * (2.0 * std::f64::consts::PI / n as f64).sin();
        assert_close(total, expected, 1e-10);
    }
}
