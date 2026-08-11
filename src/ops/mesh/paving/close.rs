//! Closing a loop: turning what is left of a front into elements.
//!
//! Used in two places — on the small loops the paver deliberately stops
//! advancing, and as the safety net if a loop ever refuses to make progress.
//! Because it accepts *any* simple polygon and always returns a conforming
//! filling, the paver can never fail outright: at worst it degrades to a
//! decomposition of the leftover polygon.
//!
//! ## Why nothing here may split an edge
//!
//! Every edge of a loop being closed is already an edge of a quadrangle laid
//! by the previous row. Inserting a node in the middle of one would leave that
//! quadrangle with a node hanging on its side — a T-junction, and a
//! non-conforming mesh. So the decomposition works strictly with the vertices
//! it is given.
//!
//! ## Parity
//!
//! A polygon with an **even** number of sides always decomposes into
//! quadrangles alone; an odd one always leaves exactly one triangle. The
//! recursion below simply chooses diagonals that keep both halves even, and
//! when handed an odd polygon it clips one triangular ear first — which is the
//! one triangle, spent where it costs the least.
//!
//! Parity is therefore decided by the contour, not by anything the paver does:
//! a row provably preserves it (see [`super::row`]), and a seam removes two
//! slots. That is why the all-quadrangle guarantee is enforced once, at the
//! entrance, by making every input loop even.

use super::geom::{orient, quad_is_valid, quad_quality, segments_cross, tri_quality};
use crate::atoms::{Point2, Vector2};

/// How many diagonals are weighed on a large polygon before one is picked.
const DIAGONAL_SAMPLES: usize = 256;

/// The elements a closure produced.
///
/// `added` holds points the closure had to create, addressed by the caller as
/// `pts.len() + i`. Adding a point *inside* the polygon is always safe — it is
/// splitting an *edge* that would leave a T-junction.
#[derive(Default)]
pub struct Closure {
    pub quads: Vec<[u32; 4]>,
    pub tris: Vec<[u32; 3]>,
    pub added: Vec<Point2>,
}

/// Fill the simple polygon `verts` (counter-clockwise, material inside).
///
/// Leaves exactly one triangle when the polygon has an odd number of sides,
/// and none when it is even — unless the polygon is so tangled that no
/// diagonal is usable, in which case it falls back to a fan of triangles
/// rather than giving up.
pub fn close(pts: &[Point2], verts: &[u32]) -> Closure {
    let mut out = Closure::default();
    let base = pts.len() as u32;
    fill(pts, verts, base, &mut out);
    out
}

/// Where a point created by the closure will land once the caller has stored
/// it. Keeps `fill` able to read back its own additions.
fn at(pts: &[Point2], out: &Closure, i: u32) -> Point2 {
    match (i as usize).checked_sub(pts.len()) {
        Some(k) => out.added[k],
        None => pts[i as usize],
    }
}

fn fill(pts: &[Point2], v: &[u32], base: u32, out: &mut Closure) {
    let n = v.len();
    match n {
        0..=2 => {}
        3 => out.tris.push([v[0], v[1], v[2]]),
        4 => {
            let c = [
                at(pts, out, v[0]),
                at(pts, out, v[1]),
                at(pts, out, v[2]),
                at(pts, out, v[3]),
            ];
            if quad_is_valid(c) {
                out.quads.push([v[0], v[1], v[2], v[3]]);
            } else if let Some(g) = centre_split(&c) {
                // A reflex corner is shared out between two quadrangles hinged
                // on an interior point. This keeps a triangle-free result
                // where the two-triangle fallback below would break it.
                let gi = base + out.added.len() as u32;
                out.added.push(g);
                let r = (0..4)
                    .find(|&i| orient(c[(i + 3) % 4], c[i], c[(i + 1) % 4]) <= 0.0)
                    .unwrap_or(0);
                out.quads.push([v[r], v[(r + 1) % 4], v[(r + 2) % 4], gi]);
                out.quads.push([v[(r + 2) % 4], v[(r + 3) % 4], v[r], gi]);
            } else {
                // A reflex quadrangle has a negative Jacobian at that corner,
                // so it is not an element any solver can use. Two triangles
                // are worse for the quadrangle count and better for everything
                // else — but only across the diagonal **through the reflex
                // corner**: the other one falls outside the quadrangle and
                // yields an inverted triangle, which is no improvement at all.
                //
                // Which corner is the reflex one is a question with an answer
                // only while the quadrangle is simple. Let two of its sides
                // cross and it has two, or none that helps, and the diagonal
                // named by the first is as likely to be the wrong one — that is
                // where the reversed slivers came from. So both diagonals are
                // weighed and the one that leaves two sound triangles wins;
                // the reflex corner only decides when neither does.
                let diagonal = |r: usize| {
                    [
                        [v[r], v[(r + 1) % 4], v[(r + 2) % 4]],
                        [v[r], v[(r + 2) % 4], v[(r + 3) % 4]],
                    ]
                };
                let sound = |r: usize| {
                    diagonal(r).iter().all(|t| {
                        orient(at(pts, out, t[0]), at(pts, out, t[1]), at(pts, out, t[2])) > 0.0
                    })
                };
                let r = (0..2).find(|&r| sound(r)).unwrap_or_else(|| {
                    (0..4)
                        .find(|&i| orient(c[(i + 3) % 4], c[i], c[(i + 1) % 4]) <= 0.0)
                        .unwrap_or(0)
                });
                out.tris.extend(diagonal(r));
            }
        }
        _ => {
            if n % 2 == 1 {
                // Clip the best ear, then the rest is even.
                if let Some(i) = best_ear(pts, v) {
                    let (a, b, c) = (v[(i + n - 1) % n], v[i], v[(i + 1) % n]);
                    out.tris.push([a, b, c]);
                    let rest: Vec<u32> = (0..n).filter(|&j| j != i).map(|j| v[j]).collect();
                    fill(pts, &rest, base, out);
                    return;
                }
            }
            match best_diagonal(pts, v) {
                Some((i, j)) => {
                    let left: Vec<u32> = (i..=j).map(|t| v[t]).collect();
                    let right: Vec<u32> = (j..n).chain(0..=i).map(|t| v[t]).collect();
                    fill(pts, &left, base, out);
                    fill(pts, &right, base, out);
                }
                None => fan(pts, v, out),
            }
        }
    }
}

/// An interior point that turns a reflex quadrangle into two valid ones.
///
/// The diagonal from the reflex corner to the opposite one already lies inside
/// the quadrangle — that is what makes the two-triangle split work — but a
/// point taken *on* it is collinear with both its ends, so each quadrangle
/// would have a degenerate corner. The candidates are therefore taken beside
/// the diagonal, offset perpendicular to it on both sides, nearest first.
///
/// Returns `None` when the shape is too pinched for any of them, leaving the
/// caller to fall back to triangles.
fn centre_split(c: &[Point2; 4]) -> Option<Point2> {
    let r = (0..4).find(|&i| orient(c[(i + 3) % 4], c[i], c[(i + 1) % 4]) <= 0.0)?;
    let (a, b) = (c[r], c[(r + 2) % 4]);
    let d = b - a;
    let len = d.norm();
    if len == 0.0 {
        return None;
    }
    let normal = Vector2::new(-d.y, d.x) / len;
    for &t in &[0.5, 0.4, 0.6, 0.3, 0.7] {
        let mid = Point2::from(a.coords * (1.0 - t) + b.coords * t);
        // Nearest offsets first: on a sliver only a hair's breadth off the
        // diagonal keeps both halves inside.
        for &off in &[
            0.02, -0.02, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.35, -0.35,
        ] {
            let g = mid + normal * (off * len);
            if quad_is_valid([c[r], c[(r + 1) % 4], c[(r + 2) % 4], g])
                && quad_is_valid([c[(r + 2) % 4], c[(r + 3) % 4], c[r], g])
            {
                return Some(g);
            }
        }
    }
    None
}

/// Last resort, once every quadrangle-preserving option is exhausted: cut the
/// polygon into triangles.
///
/// **Ears first.** Ear clipping walks the polygon's own corners and so follows
/// any shape that is simple, star-shaped or not; a fan only works from a vertex
/// that sees the whole polygon, and on anything else it reaches straight across
/// a concavity and lays a triangle outside the material — reversed, with a
/// negative Jacobian. The fan is kept for the case ear clipping refuses, which
/// is a polygon whose sides cross: it has no triangulation at all, and the fan
/// at least covers it.
fn fan(pts: &[Point2], v: &[u32], out: &mut Closure) {
    let poly: Vec<Point2> = v.iter().map(|&i| pts[i as usize]).collect();
    if let Ok(ears) = crate::ops::mesh::triangulation::ear_clip_2d(&poly) {
        let sound = ears
            .iter()
            .all(|t| orient(poly[t[0]], poly[t[1]], poly[t[2]]) > 0.0);
        if sound {
            out.tris
                .extend(ears.iter().map(|t| [v[t[0]], v[t[1]], v[t[2]]]));
            return;
        }
    }

    let n = v.len();
    let mut best = (f64::NEG_INFINITY, 0usize);
    for a in 0..n {
        let worst = (1..n - 1)
            .map(|t| {
                tri_quality(
                    pts[v[a] as usize],
                    pts[v[(a + t) % n] as usize],
                    pts[v[(a + t + 1) % n] as usize],
                )
            })
            .fold(f64::INFINITY, f64::min);
        if worst > best.0 {
            best = (worst, a);
        }
    }
    let a = best.1;
    for t in 1..n - 1 {
        out.tris.push([v[a], v[(a + t) % n], v[(a + t + 1) % n]]);
    }
}

/// The convex vertex whose ear is best shaped and does not cut the polygon.
fn best_ear(pts: &[Point2], v: &[u32]) -> Option<usize> {
    let n = v.len();
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        let (a, b, c) = (v[(i + n - 1) % n], v[i], v[(i + 1) % n]);
        let (pa, pb, pc) = (pts[a as usize], pts[b as usize], pts[c as usize]);
        if orient(pa, pb, pc) <= 0.0 {
            continue;
        }
        if !diagonal_is_free(pts, v, (i + n - 1) % n, (i + 1) % n) {
            continue;
        }
        let q = tri_quality(pa, pb, pc);
        if best.is_none_or(|(bq, _)| q > bq) {
            best = Some((q, i));
        }
    }
    best.map(|(_, i)| i)
}

/// The diagonal that splits the polygon into two even halves and leaves the
/// best-shaped pieces.
///
/// Every diagonal is weighed while the polygon is small. Past that, only a
/// bounded sample is: judging all of them costs `O(n³)` once the clearance
/// test is counted, and a leftover polygon of a few hundred sides — which
/// happens when a loop stalls — would then cost more than the entire mesh
/// around it.
fn best_diagonal(pts: &[Point2], v: &[u32]) -> Option<(usize, usize)> {
    let n = v.len();
    let step = (n * n / DIAGONAL_SAMPLES).max(1);
    let mut seen = 0usize;
    let mut best: Option<(f64, usize, usize)> = None;
    for i in 0..n {
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            seen += 1;
            if step > 1 && !seen.is_multiple_of(step) {
                continue;
            }
            // Both halves must stay even, which happens exactly when the
            // diagonal skips an odd number of vertices.
            if (j - i) % 2 == 0 {
                continue;
            }
            if !diagonal_is_free(pts, v, i, j) {
                continue;
            }
            let score = piece_score(pts, v, i, j);
            if best.is_none_or(|(bs, _, _)| score > bs) {
                best = Some((score, i, j));
            }
        }
    }
    best.map(|(_, i, j)| (i, j))
}

/// Quality of the worse of the two pieces a diagonal makes, judging a piece
/// by its shape when it is a quadrangle and by its aspect otherwise.
fn piece_score(pts: &[Point2], v: &[u32], i: usize, j: usize) -> f64 {
    let n = v.len();
    let left: Vec<u32> = (i..=j).map(|t| v[t]).collect();
    let right: Vec<u32> = (j..n).chain(0..=i).map(|t| v[t]).collect();
    let one = |p: &[u32]| -> f64 {
        if p.len() == 4 {
            quad_quality([
                pts[p[0] as usize],
                pts[p[1] as usize],
                pts[p[2] as usize],
                pts[p[3] as usize],
            ])
        } else {
            // Prefer balanced splits while the pieces are still large.
            0.25 / (p.len() as f64)
        }
    };
    one(&left).min(one(&right))
}

/// Does the chord `(i, j)` stay strictly inside the polygon, crossing no
/// side and no other chord endpoint?
fn diagonal_is_free(pts: &[Point2], v: &[u32], i: usize, j: usize) -> bool {
    let n = v.len();
    let (a, b) = (pts[v[i] as usize], pts[v[j] as usize]);
    for t in 0..n {
        let u = (t + 1) % n;
        if t == i || t == j || u == i || u == j {
            continue;
        }
        if segments_cross(a, b, pts[v[t] as usize], pts[v[u] as usize]) {
            return false;
        }
    }
    // Interior test: the chord must leave `i` inside the material wedge.
    let (prev, next) = (
        pts[v[(i + n - 1) % n] as usize],
        pts[v[(i + 1) % n] as usize],
    );
    let cur = a;
    let convex = orient(prev, cur, next) > 0.0;
    let left = orient(cur, next, b) > 0.0;
    let right = orient(prev, cur, b) > 0.0;
    if convex {
        left && right
    } else {
        left || right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> (Vec<Point2>, Vec<u32>) {
        let pts: Vec<Point2> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * std::f64::consts::TAU;
                Point2::new(t.cos(), t.sin())
            })
            .collect();
        (pts, (0..n as u32).collect())
    }

    /// Every polygon edge must be used once and every interior edge twice.
    fn is_conforming(c: &Closure, boundary: &[u32]) -> bool {
        use std::collections::HashMap;
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        let mut bump = |a: u32, b: u32| {
            let key = if a < b { (a, b) } else { (b, a) };
            *count.entry(key).or_insert(0) += 1;
        };
        for q in &c.quads {
            for i in 0..4 {
                bump(q[i], q[(i + 1) % 4]);
            }
        }
        for t in &c.tris {
            for i in 0..3 {
                bump(t[i], t[(i + 1) % 3]);
            }
        }
        let n = boundary.len();
        for i in 0..n {
            let (a, b) = (boundary[i], boundary[(i + 1) % n]);
            let key = if a < b { (a, b) } else { (b, a) };
            if count.remove(&key) != Some(1) {
                return false;
            }
        }
        count.values().all(|&v| v == 2)
    }

    fn area(c: &Closure, pts: &[Point2]) -> f64 {
        let tri = |a: Point2, b: Point2, d: Point2| orient(a, b, d) * 0.5;
        let mut s = 0.0;
        for q in c.quads.iter() {
            let p: Vec<Point2> = q.iter().map(|&i| pts[i as usize]).collect();
            s += tri(p[0], p[1], p[2]) + tri(p[0], p[2], p[3]);
        }
        for t in c.tris.iter() {
            s += tri(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
        }
        s
    }

    #[test]
    fn an_even_polygon_closes_with_no_triangle_at_all() {
        for n in [4, 6, 8, 10, 12, 16, 20] {
            let (pts, v) = ring(n);
            let c = close(&pts, &v);
            assert!(c.tris.is_empty(), "n={n} left {} triangles", c.tris.len());
            assert_eq!(c.quads.len(), n / 2 - 1, "n={n}");
            assert!(is_conforming(&c, &v), "n={n}");
        }
    }

    #[test]
    fn an_odd_polygon_leaves_exactly_one_triangle() {
        for n in [3, 5, 7, 9, 11, 15] {
            let (pts, v) = ring(n);
            let c = close(&pts, &v);
            assert_eq!(c.tris.len(), 1, "n={n}");
            assert!(is_conforming(&c, &v), "n={n}");
        }
    }

    #[test]
    fn the_closure_covers_the_polygon_exactly() {
        for n in [4, 5, 6, 9, 12] {
            let (pts, v) = ring(n);
            let c = close(&pts, &v);
            let poly = crate::ops::mesh::triangulation::signed_area(&pts);
            assert!((area(&c, &pts) - poly).abs() < 1e-9, "n={n}");
        }
    }

    #[test]
    fn a_concave_polygon_is_closed_without_leaving_the_material() {
        // An L-shaped hexagon: the naive diagonal (0, 3) would run outside.
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 1.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let v: Vec<u32> = (0..6).collect();
        let c = close(&pts, &v);
        assert!(c.tris.is_empty());
        assert!(is_conforming(&c, &v));
        let poly = crate::ops::mesh::triangulation::signed_area(&pts);
        assert!((area(&c, &pts) - poly).abs() < 1e-12);
    }
}
