//! Plane geometry for the paver, with every topological decision resting on
//! the exact predicate [`orient2d`].
//!
//! The paver never asks "is this number small?"; it asks "is this point left
//! of that line?" and gets an answer that is exactly right, including on
//! degenerate input. That is what lets the front invariant — a set of simple,
//! pairwise disjoint loops — be *maintained* rather than *hoped for*.

use crate::atoms::{Point2, Vector2};
use crate::ops::mesh::predicates::orient2d;

/// Exact orientation of the triangle `(a, b, c)`: `> 0` counter-clockwise,
/// `< 0` clockwise, `0` exactly collinear.
#[inline]
pub fn orient(a: Point2, b: Point2, c: Point2) -> f64 {
    orient2d(&[a.x, a.y], &[b.x, b.y], &[c.x, c.y])
}

/// Rotate `v` counter-clockwise by `ang` radians.
#[inline]
pub fn rot(v: Vector2, ang: f64) -> Vector2 {
    let (s, c) = ang.sin_cos();
    Vector2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Interior angle at `c` in a front walked `p → c → n`, in `[0, 2π)`.
///
/// The front keeps material on its left, so the interior angle is the one
/// swept counter-clockwise from `n - c` round to `p - c`. A convex corner of
/// a counter-clockwise square measures `π/2`; a reflex corner measures
/// `3π/2`.
pub fn interior_angle(p: Point2, c: Point2, n: Point2) -> f64 {
    let a = n - c;
    let b = p - c;
    let ang = (a.x * b.y - a.y * b.x).atan2(a.dot(&b));
    if ang < 0.0 {
        ang + std::f64::consts::TAU
    } else {
        ang
    }
}

/// The unit direction that bisects the interior angle at `c`, pointing into
/// the material.
///
/// Built by rotating `n - c` counter-clockwise by half the interior angle,
/// which is correct for reflex corners too — unlike the sum of the two unit
/// edge vectors, which flips to the outside beyond `π`.
pub fn interior_direction(p: Point2, c: Point2, n: Point2, frac: f64) -> Option<Vector2> {
    let a = n - c;
    let na = a.norm();
    if na == 0.0 {
        return None;
    }
    let theta = interior_angle(p, c, n);
    Some(rot(a / na, theta * frac))
}

/// Do the closed segments `a1a2` and `b1b2` cross?
///
/// Exact, and deliberately inclusive: touching counts as crossing, because a
/// front edge that merely grazes another already breaks the invariant. The
/// caller is responsible for not testing two edges that legitimately share an
/// endpoint.
pub fn segments_cross(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> bool {
    let d1 = orient(a1, a2, b1);
    let d2 = orient(a1, a2, b2);
    let d3 = orient(b1, b2, a1);
    let d4 = orient(b1, b2, a2);

    if ((d1 > 0.0) != (d2 > 0.0) || d1 == 0.0 || d2 == 0.0)
        && ((d3 > 0.0) != (d4 > 0.0) || d3 == 0.0 || d4 == 0.0)
    {
        // Proper crossing, or a touch. Collinear overlap needs the extra
        // range test below; everything else is settled here.
        if d1 != 0.0 || d2 != 0.0 || d3 != 0.0 || d4 != 0.0 {
            return true;
        }
        return overlap_1d(a1, a2, b1) || overlap_1d(a1, a2, b2) || overlap_1d(b1, b2, a1);
    }
    false
}

/// Is `p`, known to be collinear with `a` and `b`, inside the segment `ab`?
fn overlap_1d(a: Point2, b: Point2, p: Point2) -> bool {
    p.x >= a.x.min(b.x) && p.x <= a.x.max(b.x) && p.y >= a.y.min(b.y) && p.y <= a.y.max(b.y)
}

/// Is the closed polygon through `p` **simple** — do two of its sides meet
/// anywhere other than at the vertex they share?
///
/// A simple polygon always admits a filling whose every element turns the same
/// way as it does; one that crosses itself admits none, since the two lobes on
/// either side of the crossing turn opposite ways. So this is the predicate
/// that says whether a front loop can be closed at all, and the sign of its
/// area is not: a fold whose two lobes nearly cancel keeps whichever sign the
/// larger one happens to have.
///
/// Exact, and not quadratic: the answer is [`segments_cross`] on every pair of
/// sides sharing no endpoint, but the pairs worth asking about are found
/// through a grid. Asking all of them is what a front of ten thousand nodes
/// cannot afford.
pub fn polygon_is_simple(p: &[Point2]) -> bool {
    let n = p.len();
    if n < 4 {
        return true;
    }
    let grid = super::proximity::EdgeGrid::of_ring(p, 0.0);
    for i in 0..n {
        let (a, b) = (p[i], p[(i + 1) % n]);
        for j in grid.near_segment(a, b) {
            let j = j as usize;
            // Skip the side itself and the two it shares an endpoint with:
            // touching there is the polygon being closed, not a crossing.
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            if segments_cross(a, b, p[j], p[(j + 1) % n]) {
                return false;
            }
        }
    }
    true
}

/// Is the quadrangle `q` (in order) convex and counter-clockwise?
///
/// This is exactly the condition for a `QUA4` to have a strictly positive
/// bilinear Jacobian everywhere: a reflex corner makes the Jacobian negative
/// there, which no finite-element code can integrate. Convexity is therefore
/// not a matter of taste here but the validity criterion itself.
pub fn quad_is_valid(q: [Point2; 4]) -> bool {
    for i in 0..4 {
        if orient(q[(i + 3) % 4], q[i], q[(i + 1) % 4]) <= 0.0 {
            return false;
        }
    }
    true
}

/// Shape quality of a quadrangle in `[0, 1]`: the scaled Jacobian, i.e. the
/// worst of its four corner cross-products normalised by the incident edge
/// lengths. `1` is a square, `0` a degenerate corner, negative a tangle.
/// How square a quadrangle is: the worst of its four corners, each scored by
/// the **mean ratio** `2·(a × b) / (|a|² + |b|²)`.
///
/// Not `sin θ`, which is what a normalised Jacobian measures and what this
/// returned until it was measured. The sine sees only the angle, so a 10:1
/// rectangle scores a perfect 1 — and the smoothing guard, which refuses a move
/// that lowers the worst incident quality, was therefore free to squash a cell
/// flat so long as it kept the corners square. The mean ratio is the same
/// quantity times `2|a||b| / (|a|² + |b|²)`, which is 1 for equal edges and
/// falls away as they differ: it reaches 1 only when the corner is square
/// **and** its two edges are the same length.
pub fn quad_quality(q: [Point2; 4]) -> f64 {
    let mut worst = f64::INFINITY;
    for i in 0..4 {
        let prev = q[(i + 3) % 4];
        let cur = q[i];
        let next = q[(i + 1) % 4];
        let a = next - cur;
        let b = prev - cur;
        let (na, nb) = (a.norm(), b.norm());
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        worst = worst.min(2.0 * (a.x * b.y - a.y * b.x) / (na * na + nb * nb));
    }
    worst
}

/// Shape quality of a triangle in `[0, 1]`: `4/√3 · area / (longest edge)²`
/// normalised so that an equilateral triangle scores `1`.
pub fn tri_quality(a: Point2, b: Point2, c: Point2) -> f64 {
    let area = orient(a, b, c) * 0.5;
    let l = [
        (b - a).norm_squared(),
        (c - b).norm_squared(),
        (a - c).norm_squared(),
    ];
    let lmax = l[0].max(l[1]).max(l[2]);
    if lmax == 0.0 {
        return 0.0;
    }
    4.0 / 3.0_f64.sqrt() * area / lmax
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    fn p(x: f64, y: f64) -> Point2 {
        Point2::new(x, y)
    }

    #[test]
    fn interior_angle_reads_convex_and_reflex_corners() {
        // Corner of a counter-clockwise square: 90°.
        let a = interior_angle(p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0));
        assert!((a - FRAC_PI_2).abs() < 1e-12, "{a}");
        // Straight front: 180°.
        let a = interior_angle(p(0.0, 0.0), p(1.0, 0.0), p(2.0, 0.0));
        assert!((a - PI).abs() < 1e-12, "{a}");
        // Reflex corner: 270°.
        let a = interior_angle(p(1.0, 1.0), p(1.0, 0.0), p(0.0, 0.0));
        assert!((a - 3.0 * FRAC_PI_2).abs() < 1e-12, "{a}");
        // Always in [0, 2π).
        for &q in &[p(0.5, -1.0), p(-1.0, 0.3), p(2.0, 2.0)] {
            let a = interior_angle(p(0.0, 0.0), p(1.0, 0.0), q);
            assert!((0.0..TAU).contains(&a));
        }
    }

    #[test]
    fn interior_direction_points_into_the_material() {
        // Convex corner: the bisector heads up-left into the square.
        let d = interior_direction(p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), 0.5).unwrap();
        assert!(d.x < 0.0 && d.y > 0.0, "{d:?}");
        // Reflex corner: walking (1,1) → (1,0) → (0,0) keeps material at
        // `x > 1` then at `y < 0`, so the bisector of the 270° sector heads
        // down and to the right — the direction the naive "sum of unit edge
        // vectors" formula gets backwards.
        let d = interior_direction(p(1.0, 1.0), p(1.0, 0.0), p(0.0, 0.0), 0.5).unwrap();
        assert!(d.x > 0.0 && d.y < 0.0, "{d:?}");
    }

    #[test]
    fn segment_crossing_is_exact_at_touches() {
        assert!(segments_cross(
            p(0.0, 0.0),
            p(2.0, 2.0),
            p(0.0, 2.0),
            p(2.0, 0.0)
        ));
        assert!(!segments_cross(
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(0.0, 1.0),
            p(1.0, 1.0)
        ));
        // A T-touch counts as a crossing.
        assert!(segments_cross(
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.0, 0.0),
            p(1.0, 1.0)
        ));
        // Collinear overlap counts too.
        assert!(segments_cross(
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.0, 0.0),
            p(3.0, 0.0)
        ));
        // Collinear but disjoint does not.
        assert!(!segments_cross(
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(2.0, 0.0),
            p(3.0, 0.0)
        ));
    }

    #[test]
    fn a_fold_is_seen_even_when_its_area_stays_positive() {
        // The predicate the paver leans on to know a loop can be filled, and
        // the reason it is not the sign of the area: this pentagon is a front
        // that crossed itself, yet its two lobes leave a positive remainder.
        // It is the shape that used to be filled with a reversed sliver.
        let fold = [
            p(0.739_583_257_097_729_7, -0.311_133_281_363_632_94),
            p(0.801_277_069_162_058_2, -0.261_809_826_963_276_04),
            p(0.777_038_034_503_386_2, -0.259_915_317_245_687_84),
            p(0.773_227_538_342_794_9, -0.280_237_839_791_400_27),
            p(0.759_661_065_245_258_8, -0.295_096_673_379_862_95),
        ];
        assert!(crate::ops::mesh::triangulation::signed_area(&fold) > 0.0);
        assert!(!polygon_is_simple(&fold));

        // The plain cases either way.
        let square = [p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
        assert!(polygon_is_simple(&square));
        let l = [
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(2.0, 1.0),
            p(1.0, 1.0),
            p(1.0, 2.0),
            p(0.0, 2.0),
        ];
        assert!(polygon_is_simple(&l));
        // A bow tie: the crossing is proper, and its area cancels to zero.
        let bow = [p(0.0, 0.0), p(1.0, 0.0), p(0.0, 1.0), p(1.0, 1.0)];
        assert!(!polygon_is_simple(&bow));
        // A side running back over the previous one, touching no vertex.
        let spike = [p(0.0, 0.0), p(2.0, 0.0), p(1.0, 0.0), p(1.0, 1.0)];
        assert!(!polygon_is_simple(&spike));
    }

    #[test]
    fn quad_validity_is_convexity() {
        let sq = [p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0)];
        assert!(quad_is_valid(sq));
        assert!((quad_quality(sq) - 1.0).abs() < 1e-12);
        // Reflex corner: invalid, and the scaled Jacobian is negative.
        let dart = [p(0.0, 0.0), p(1.0, 0.0), p(0.4, 0.4), p(0.0, 1.0)];
        assert!(!quad_is_valid(dart));
        assert!(quad_quality(dart) < 0.0);
        // Clockwise square: invalid.
        let cw = [p(0.0, 1.0), p(1.0, 1.0), p(1.0, 0.0), p(0.0, 0.0)];
        assert!(!quad_is_valid(cw));
    }
}
