//! Exact triangle-triangle intersection, and the sweep that applies it to a
//! whole surface.
//!
//! A self-intersecting envelope has no well-defined inside, so no correct
//! tetrahedralization exists. Left undetected it does not announce itself:
//! the Delaunay kernel builds something, boundary recovery half-succeeds,
//! and the failure finally surfaces as a leak in the inside/outside flood
//! fill, far from the cause. Finding it up front turns that into a sentence
//! naming the two offending facets.
//!
//! Every decision here goes through the exact predicates, so a surface is
//! never rejected — nor accepted — on a rounding artefact.
//!
//! **Scope.** Only facet pairs sharing *no* node are tested. Facets that
//! share a node or an edge legitimately touch there, and separating a
//! genuine overlap from that contact needs a different argument; those cases
//! are left to the conformity checks further down the pipeline. Surfaces
//! that fold onto themselves — by far the common failure — always involve
//! pairs that share nothing, and are caught here.

use std::collections::HashMap;

use crate::parallel::*;

use super::predicates::{orient2d, orient3d};

/// Index of the first pair of facets found to intersect, if any.
///
/// `points` holds the node positions and `facets` the triangles as indices
/// into it. The returned pair is ordered and is the smallest such pair in
/// lexicographic order, so the answer does not depend on the thread count.
pub fn first_self_intersection(points: &[[f64; 3]], facets: &[[u32; 3]]) -> Option<(usize, usize)> {
    let grid = Grid::build(points, facets);

    // One candidate per facet, gathered in parallel; the first `Some` in
    // facet order is then the lexicographic minimum.
    let hits: Vec<Option<(usize, usize)>> = (0..facets.len())
        .into_par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|i| {
            let fi = facets[i];
            let mut best: Option<(usize, usize)> = None;
            for j in grid.candidates_after(points, facets, i) {
                let fj = facets[j];
                // Facets meeting at a node are allowed to touch there.
                if fi.iter().any(|a| fj.contains(a)) {
                    continue;
                }
                if triangles_intersect(&tri(points, &fi), &tri(points, &fj))
                    && best.is_none_or(|(_, b)| j < b)
                {
                    best = Some((i, j));
                }
            }
            best
        })
        .collect();

    hits.into_iter().flatten().next()
}

fn tri<'a>(points: &'a [[f64; 3]], f: &[u32; 3]) -> [&'a [f64; 3]; 3] {
    [
        &points[f[0] as usize],
        &points[f[1] as usize],
        &points[f[2] as usize],
    ]
}

// ─── Broad phase ────────────────────────────────────────────────────────

/// Uniform grid over the facet bounding boxes — the same shape as the one
/// [`crate::ops::geom::locate_points`] uses to prune its candidates.
struct Grid {
    lo: [f64; 3],
    inv: [f64; 3],
    res: [usize; 3],
    buckets: HashMap<[usize; 3], Vec<u32>>,
}

impl Grid {
    fn build(points: &[[f64; 3]], facets: &[[u32; 3]]) -> Grid {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        // Roughly one cell per facet, so a bucket holds O(1) of them.
        let n = (facets.len().max(1) as f64).cbrt().ceil() as usize;
        let res = [n.clamp(1, 128); 3];
        let mut inv = [0.0f64; 3];
        for k in 0..3 {
            let span = hi[k] - lo[k];
            inv[k] = if span > 0.0 {
                res[k] as f64 / span
            } else {
                0.0
            };
        }

        let mut grid = Grid {
            lo,
            inv,
            res,
            buckets: HashMap::new(),
        };
        for (i, f) in facets.iter().enumerate() {
            let (flo, fhi) = facet_bbox(points, f);
            grid.for_each_cell(flo, fhi, |cell, g| {
                g.buckets.entry(cell).or_default().push(i as u32)
            });
        }
        grid
    }

    fn cell_of(&self, x: [f64; 3]) -> [usize; 3] {
        let mut c = [0usize; 3];
        for k in 0..3 {
            let t = ((x[k] - self.lo[k]) * self.inv[k]).floor();
            c[k] = (t.max(0.0) as usize).min(self.res[k] - 1);
        }
        c
    }

    /// Visit every cell the box `[lo, hi]` overlaps.
    fn for_each_cell(
        &mut self,
        lo: [f64; 3],
        hi: [f64; 3],
        mut f: impl FnMut([usize; 3], &mut Self),
    ) {
        let (a, b) = (self.cell_of(lo), self.cell_of(hi));
        for x in a[0]..=b[0] {
            for y in a[1]..=b[1] {
                for z in a[2]..=b[2] {
                    f([x, y, z], self);
                }
            }
        }
    }

    /// Facets after `i` whose bounding box shares a cell with facet `i`'s.
    fn candidates_after(&self, points: &[[f64; 3]], facets: &[[u32; 3]], i: usize) -> Vec<usize> {
        let (flo, fhi) = facet_bbox(points, &facets[i]);
        let (a, b) = (self.cell_of(flo), self.cell_of(fhi));
        let mut out: Vec<usize> = Vec::new();
        for x in a[0]..=b[0] {
            for y in a[1]..=b[1] {
                for z in a[2]..=b[2] {
                    if let Some(bucket) = self.buckets.get(&[x, y, z]) {
                        out.extend(bucket.iter().map(|&j| j as usize).filter(|&j| j > i));
                    }
                }
            }
        }
        // A facet spanning several cells shows up once per shared cell.
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn facet_bbox(points: &[[f64; 3]], f: &[u32; 3]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &v in f {
        let p = &points[v as usize];
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

// ─── Narrow phase ───────────────────────────────────────────────────────

/// Whether two closed triangles have any point in common.
///
/// Assumes both are non-degenerate, which the envelope validation has
/// already established.
pub fn triangles_intersect(t: &[&[f64; 3]; 3], u: &[&[f64; 3]; 3]) -> bool {
    let du = [
        orient3d(u[0], u[1], u[2], t[0]),
        orient3d(u[0], u[1], u[2], t[1]),
        orient3d(u[0], u[1], u[2], t[2]),
    ];
    if all_strictly_positive(&du) || all_strictly_negative(&du) {
        return false; // t sits wholly on one side of u's plane
    }
    let dt = [
        orient3d(t[0], t[1], t[2], u[0]),
        orient3d(t[0], t[1], t[2], u[1]),
        orient3d(t[0], t[1], t[2], u[2]),
    ];
    if all_strictly_positive(&dt) || all_strictly_negative(&dt) {
        return false;
    }

    if du.iter().all(|&d| d == 0.0) {
        return coplanar_triangles_overlap(t, u);
    }
    // Non-coplanar: the two planes meet along a line, and any shared point
    // lies on it — so an edge of one triangle must reach the other.
    (0..3).any(|k| segment_hits_triangle(t[k], t[(k + 1) % 3], u))
        || (0..3).any(|k| segment_hits_triangle(u[k], u[(k + 1) % 3], t))
}

fn all_strictly_positive(d: &[f64; 3]) -> bool {
    d.iter().all(|&x| x > 0.0)
}

fn all_strictly_negative(d: &[f64; 3]) -> bool {
    d.iter().all(|&x| x < 0.0)
}

/// Whether the closed segment `pq` meets the closed triangle `t`.
fn segment_hits_triangle(p: &[f64; 3], q: &[f64; 3], t: &[&[f64; 3]; 3]) -> bool {
    let sp = orient3d(t[0], t[1], t[2], p);
    let sq = orient3d(t[0], t[1], t[2], q);
    if (sp > 0.0 && sq > 0.0) || (sp < 0.0 && sq < 0.0) {
        return false; // both endpoints strictly on the same side
    }
    if sp == 0.0 && sq == 0.0 {
        return coplanar_segment_hits_triangle(p, q, t);
    }
    // The segment meets the plane exactly once, within its own span. That
    // crossing point is inside the triangle when the line pq passes on the
    // same side of all three edges.
    let s = [
        orient3d(p, q, t[0], t[1]),
        orient3d(p, q, t[1], t[2]),
        orient3d(p, q, t[2], t[0]),
    ];
    same_sign_ignoring_zeros(&s)
}

/// True when the non-zero entries all share one sign — a zero means the
/// line grazes that edge, which still counts as touching.
fn same_sign_ignoring_zeros(s: &[f64; 3]) -> bool {
    !(s.iter().any(|&x| x > 0.0) && s.iter().any(|&x| x < 0.0))
}

// ─── Coplanar cases, resolved in the dominant projection ────────────────

/// Drop the axis the triangle's normal is most aligned with, so the
/// projection of a non-degenerate triangle is never degenerate.
fn dominant_projection(t: &[&[f64; 3]; 3]) -> (usize, usize) {
    let e1 = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
    let e2 = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let drop = (0..3)
        .max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
        .unwrap();
    match drop {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    }
}

fn project(p: &[f64; 3], axes: (usize, usize)) -> [f64; 2] {
    [p[axes.0], p[axes.1]]
}

fn coplanar_triangles_overlap(t: &[&[f64; 3]; 3], u: &[&[f64; 3]; 3]) -> bool {
    let axes = dominant_projection(t);
    let tp = [
        project(t[0], axes),
        project(t[1], axes),
        project(t[2], axes),
    ];
    let up = [
        project(u[0], axes),
        project(u[1], axes),
        project(u[2], axes),
    ];
    // Either a vertex of one lies in the other (covers containment), or two
    // edges cross (covers partial overlap).
    up.iter().any(|p| point_in_triangle_2d(p, &tp))
        || tp.iter().any(|p| point_in_triangle_2d(p, &up))
        || (0..3).any(|a| {
            (0..3).any(|b| segments_meet_2d(&tp[a], &tp[(a + 1) % 3], &up[b], &up[(b + 1) % 3]))
        })
}

fn coplanar_segment_hits_triangle(p: &[f64; 3], q: &[f64; 3], t: &[&[f64; 3]; 3]) -> bool {
    let axes = dominant_projection(t);
    let tp = [
        project(t[0], axes),
        project(t[1], axes),
        project(t[2], axes),
    ];
    let (p2, q2) = (project(p, axes), project(q, axes));
    point_in_triangle_2d(&p2, &tp)
        || point_in_triangle_2d(&q2, &tp)
        || (0..3).any(|k| segments_meet_2d(&p2, &q2, &tp[k], &tp[(k + 1) % 3]))
}

/// Closed point-in-triangle test: on an edge counts as inside.
fn point_in_triangle_2d(p: &[f64; 2], t: &[[f64; 2]; 3]) -> bool {
    let s = [
        orient2d(&t[0], &t[1], p),
        orient2d(&t[1], &t[2], p),
        orient2d(&t[2], &t[0], p),
    ];
    !(s.iter().any(|&x| x > 0.0) && s.iter().any(|&x| x < 0.0))
}

/// Whether two closed 2-D segments share a point, collinear overlaps
/// included.
fn segments_meet_2d(p1: &[f64; 2], p2: &[f64; 2], q1: &[f64; 2], q2: &[f64; 2]) -> bool {
    let d1 = orient2d(q1, q2, p1);
    let d2 = orient2d(q1, q2, p2);
    let d3 = orient2d(p1, p2, q1);
    let d4 = orient2d(p1, p2, q2);

    if ((d1 > 0.0) != (d2 > 0.0))
        && ((d1 < 0.0) != (d2 < 0.0))
        && ((d3 > 0.0) != (d4 > 0.0))
        && ((d3 < 0.0) != (d4 < 0.0))
    {
        return true; // each segment straddles the other's line
    }
    // Touching or collinear: an endpoint lies on the other segment.
    (d1 == 0.0 && on_segment_2d(q1, q2, p1))
        || (d2 == 0.0 && on_segment_2d(q1, q2, p2))
        || (d3 == 0.0 && on_segment_2d(p1, p2, q1))
        || (d4 == 0.0 && on_segment_2d(p1, p2, q2))
}

/// Whether `p`, already known to be collinear with `ab`, lies within it.
fn on_segment_2d(a: &[f64; 2], b: &[f64; 2], p: &[f64; 2]) -> bool {
    (0..2).all(|k| p[k] >= a[k].min(b[k]) && p[k] <= a[k].max(b[k]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> [[f64; 3]; 3] {
        [*a, *b, *c]
    }

    fn hit(x: &[[f64; 3]; 3], y: &[[f64; 3]; 3]) -> bool {
        let a = [&x[0], &x[1], &x[2]];
        let b = [&y[0], &y[1], &y[2]];
        let forward = triangles_intersect(&a, &b);
        // The relation is symmetric; a test that says otherwise is a bug.
        assert_eq!(forward, triangles_intersect(&b, &a));
        forward
    }

    #[test]
    fn separated_triangles_do_not_intersect() {
        let a = t(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        let b = t(&[0.0, 0.0, 1.0], &[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0]);
        assert!(!hit(&a, &b));
    }

    #[test]
    fn a_blade_through_a_triangle_intersects() {
        // A vertical triangle whose edge pierces a horizontal one.
        let flat = t(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0]);
        let blade = t(&[0.5, 0.5, -1.0], &[0.5, 0.5, 1.0], &[1.5, 1.5, 1.0]);
        assert!(hit(&flat, &blade));
    }

    #[test]
    fn a_blade_beside_a_triangle_does_not_intersect() {
        let flat = t(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0]);
        // Crosses the plane, but outside the triangle.
        let blade = t(&[3.0, 3.0, -1.0], &[3.0, 3.0, 1.0], &[4.0, 4.0, 1.0]);
        assert!(!hit(&flat, &blade));
    }

    #[test]
    fn a_vertex_touching_a_face_counts_as_intersecting() {
        let flat = t(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0]);
        let spike = t(&[0.5, 0.5, 0.0], &[0.5, 0.5, 1.0], &[1.5, 0.5, 1.0]);
        assert!(hit(&flat, &spike));
    }

    #[test]
    fn a_vertex_one_ulp_above_a_face_does_not() {
        let flat = t(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0]);
        let spike = t(
            &[0.5, 0.5, f64::MIN_POSITIVE],
            &[0.5, 0.5, 1.0],
            &[1.5, 0.5, 1.0],
        );
        assert!(!hit(&flat, &spike));
    }

    #[test]
    fn coplanar_overlapping_triangles_intersect() {
        let a = t(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0]);
        let b = t(&[0.5, 0.5, 0.0], &[2.5, 0.5, 0.0], &[0.5, 2.5, 0.0]);
        assert!(hit(&a, &b));
    }

    #[test]
    fn a_coplanar_triangle_inside_another_intersects() {
        let a = t(&[0.0, 0.0, 0.0], &[4.0, 0.0, 0.0], &[0.0, 4.0, 0.0]);
        let b = t(&[1.0, 1.0, 0.0], &[2.0, 1.0, 0.0], &[1.0, 2.0, 0.0]);
        assert!(hit(&a, &b));
    }

    #[test]
    fn coplanar_disjoint_triangles_do_not_intersect() {
        let a = t(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        let b = t(&[3.0, 3.0, 0.0], &[4.0, 3.0, 0.0], &[3.0, 4.0, 0.0]);
        assert!(!hit(&a, &b));
    }

    #[test]
    fn coplanar_triangles_touching_along_an_edge_intersect() {
        // Two halves of a square: they legitimately share the diagonal, and
        // the predicate reports the contact. The sweep never sees such a
        // pair, since they share two nodes.
        let a = t(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        let b = t(&[1.0, 0.0, 0.0], &[1.0, 1.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(hit(&a, &b));
    }

    // ─── The sweep ──────────────────────────────────────────────────────

    /// A tetrahedron, outward-oriented.
    fn tetra_points() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let f = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (p, f)
    }

    #[test]
    fn a_sound_surface_reports_nothing() {
        let (p, f) = tetra_points();
        assert_eq!(first_self_intersection(&p, &f), None);
    }

    #[test]
    fn a_folded_surface_is_caught() {
        // Two tetrahedra sharing no node, placed so they interpenetrate.
        let (mut p, mut f) = tetra_points();
        let shift = [0.2, 0.2, 0.2];
        let base = p.len() as u32;
        let (q, g) = tetra_points();
        p.extend(
            q.iter()
                .map(|x| [x[0] + shift[0], x[1] + shift[1], x[2] + shift[2]]),
        );
        f.extend(g.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        assert!(first_self_intersection(&p, &f).is_some());
    }

    #[test]
    fn the_reported_pair_is_the_lexicographic_minimum() {
        // Determinism matters: the same input must name the same pair on
        // every run and every thread count.
        let (mut p, mut f) = tetra_points();
        let base = p.len() as u32;
        let (q, g) = tetra_points();
        p.extend(q.iter().map(|x| [x[0] + 0.2, x[1] + 0.2, x[2] + 0.2]));
        f.extend(g.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));

        let first = first_self_intersection(&p, &f).unwrap();
        for _ in 0..8 {
            assert_eq!(first_self_intersection(&p, &f), Some(first));
        }
        // It really is the smallest intersecting pair.
        let mut expected = None;
        'outer: for i in 0..f.len() {
            for j in i + 1..f.len() {
                if f[i].iter().any(|a| f[j].contains(a)) {
                    continue;
                }
                let (a, b) = (tri(&p, &f[i]), tri(&p, &f[j]));
                if triangles_intersect(&a, &b) {
                    expected = Some((i, j));
                    break 'outer;
                }
            }
        }
        assert_eq!(Some(first), expected);
    }

    #[test]
    fn touching_facets_of_a_sound_surface_are_not_flagged() {
        // A box: many facet pairs touch along shared edges and corners, and
        // none of it is a self-intersection.
        let p: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let f: Vec<[u32; 3]> = vec![
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        assert_eq!(first_self_intersection(&p, &f), None);
    }
}
