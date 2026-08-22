//! Topological cleanup: fixing *how the quadrangles are connected*, which
//! smoothing cannot touch.
//!
//! Smoothing moves nodes; it cannot change who is next to whom. So it can make
//! a bad connectivity look tidier, and never make it good. Three defects
//! survive paving and all are connectivity, not geometry:
//!
//! - a **doublet** — an interior node with only two quadrangles around it,
//!   which therefore share *two* edges. Whatever smoothing does, the node sits
//!   in a wedge and both cells stay pinched. Merging the two into one removes
//!   the node and the wedge together.
//! - a node with the **wrong valence**. In a quadrangle mesh an interior node
//!   wants four cells around it: with three, the corners average 120°, with
//!   five, 72°. Neither can be smoothed to a right angle, because the angles
//!   around a node must sum to 2π whatever the positions.
//! - a **triangle wedged** between two quadrangles at an interior node. Three
//!   cells round a node are already one too few, and neither move above can
//!   reach this one: the doublet rule wants two cells, the diagonal switch
//!   wants two quadrangles. But the three cover a **pentagon**, whose only
//!   decomposition is one quadrangle and one triangle — so the node is given
//!   up and the pentagon re-cut. Two cells for three, and the triangle that
//!   comes out is the whole wedge instead of the sliver in it.
//!
//! ## What a node ought to have
//!
//! The same rule as the row classification, and for the same reason: a node
//! spanning an interior angle `θ` wants `round(θ / 90°)` cells around it. The
//! discrete `θ` is just the sum of the incident corner angles, which is `2π`
//! at an interior node — giving the familiar four — and less at the boundary,
//! giving three along a straight edge and two at a right-angled corner. One
//! formula covers both, so no node needs to be special-cased as "boundary".
//!
//! ## The move
//!
//! Two quadrangles sharing an edge form a hexagon, and a hexagon splits into
//! two quadrangles across any of its three diagonals. Switching diagonal moves
//! one unit of valence from the two nodes on the old diagonal to the two on
//! the new one. That is the whole repertoire: it changes no node count and no
//! boundary, so it cannot make the mesh non-conforming, and it is applied only
//! when it strictly lowers the total valence error and leaves both cells
//! convex.

use super::geom::{quad_is_valid, quad_quality};
use crate::atoms::Point2;
use std::collections::HashMap;

/// Passes over the mesh. Each one is a sweep of doublet removal followed by a
/// sweep of diagonal switches; a pass that changes nothing ends the loop.
const ROUNDS: usize = 6;

/// A switch must not drop the worse of the two cells below this share of what
/// it was, so valence is never bought at the price of a sliver.
const QUALITY_FLOOR: f64 = 0.7;

/// What the cleanup changed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub doublets: usize,
    pub switches: usize,
    /// Nodes given up by [`absorb_triangle_corners`].
    pub absorbed: usize,
}

/// Clean up `quads` in place. `tris` are read for incidence but never
/// modified — they are the few cells paving could not make square, and moving
/// them around would not improve anything.
///
/// Quadrangles that disappear are left as `None`; the caller compacts.
pub fn run(
    pts: &[Point2],
    movable: &[bool],
    quads: &mut Vec<[u32; 4]>,
    tris: &mut [[u32; 3]],
) -> Report {
    let mut report = Report::default();
    let mut alive = vec![true; quads.len()];
    for _ in 0..ROUNDS {
        let before = (report.doublets, report.switches, report.absorbed);
        report.doublets += remove_doublets(pts, movable, quads, tris, &mut alive);
        report.absorbed += absorb_triangle_corners(pts, movable, quads, tris, &mut alive);
        report.switches += switch_diagonals(pts, quads, tris, &mut alive);
        if (report.doublets, report.switches, report.absorbed) == before {
            break;
        }
    }
    let mut kept = Vec::with_capacity(quads.len());
    for (i, q) in quads.iter().enumerate() {
        if alive[i] {
            kept.push(*q);
        }
    }
    *quads = kept;
    report
}

/// Cells around each node: `2 * index` for a live quadrangle, `2 * index + 1`
/// for a triangle.
fn incidence(quads: &[[u32; 4]], tris: &[[u32; 3]], alive: &[bool], n_pts: usize) -> Vec<Vec<u32>> {
    let mut inc = vec![Vec::new(); n_pts];
    for (i, q) in quads.iter().enumerate() {
        if alive[i] {
            for &v in q {
                inc[v as usize].push((i as u32) * 2);
            }
        }
    }
    for (i, t) in tris.iter().enumerate() {
        for &v in t {
            inc[v as usize].push((i as u32) * 2 + 1);
        }
    }
    inc
}

/// Sum of the incident corner angles at each node — the discrete interior
/// angle, `2π` in the interior and less on the boundary.
fn interior_angles(
    pts: &[Point2],
    quads: &[[u32; 4]],
    tris: &[[u32; 3]],
    inc: &[Vec<u32>],
) -> Vec<f64> {
    let corner = |a: Point2, c: Point2, b: Point2| {
        let (u, w) = (a - c, b - c);
        let (nu, nw) = (u.norm(), w.norm());
        if nu == 0.0 || nw == 0.0 {
            0.0
        } else {
            (u.dot(&w) / (nu * nw)).clamp(-1.0, 1.0).acos()
        }
    };
    (0..pts.len())
        .map(|v| {
            inc[v]
                .iter()
                .map(|&e| {
                    if e.is_multiple_of(2) {
                        let q = quads[(e / 2) as usize];
                        let i = q.iter().position(|&x| x as usize == v).unwrap();
                        corner(
                            pts[q[(i + 3) % 4] as usize],
                            pts[v],
                            pts[q[(i + 1) % 4] as usize],
                        )
                    } else {
                        let t = tris[(e / 2) as usize];
                        let i = t.iter().position(|&x| x as usize == v).unwrap();
                        corner(
                            pts[t[(i + 2) % 3] as usize],
                            pts[v],
                            pts[t[(i + 1) % 3] as usize],
                        )
                    }
                })
                .sum()
        })
        .collect()
}

/// How many cells a node of interior angle `theta` wants around it.
fn wanted(theta: f64) -> usize {
    ((theta / std::f64::consts::FRAC_PI_2).round() as i64).clamp(1, 6) as usize
}

/// Absorb a node that carries a triangle and two quadrangles into the two
/// cells it can make instead.
///
/// A small triangle wedged between two quadrangles at an interior node is a
/// defect nothing else here can reach: the doublet rule needs two cells, the
/// diagonal switch works on quadrangle pairs, and both leave triangles alone.
/// But three cells round a node are three cells too many — an interior node
/// wants four — and the shape they cover is a **pentagon**, whose only
/// decomposition is one quadrangle and one triangle.
///
/// So the node is given up and the pentagon re-cut. Two cells replace three,
/// the triangle that comes out is the whole wedge rather than the sliver in
/// it, and the node's valence problem goes with it. The cut is taken across
/// whichever of the five diagonals leaves the worst of the two cells best, and
/// the move is refused outright if that is not better than what was there —
/// a pentagon can be too bent to improve on.
fn absorb_triangle_corners(
    pts: &[Point2],
    movable: &[bool],
    quads: &mut [[u32; 4]],
    tris: &mut [[u32; 3]],
    alive: &mut [bool],
) -> usize {
    let inc = incidence(quads, tris, alive, pts.len());
    let mut done = 0;
    let mut spent = vec![false; pts.len()];
    for v in 0..pts.len() {
        if !movable[v] || inc[v].len() != 3 || spent[v] {
            continue;
        }
        let tri_at: Vec<u32> = inc[v]
            .iter()
            .copied()
            .filter(|e| !e.is_multiple_of(2))
            .collect();
        let quad_at: Vec<u32> = inc[v]
            .iter()
            .copied()
            .filter(|e| e.is_multiple_of(2))
            .collect();
        if tri_at.len() != 1 || quad_at.len() != 2 {
            continue;
        }
        let ti = (tri_at[0] / 2) as usize;
        let (q1, q2) = ((quad_at[0] / 2) as usize, (quad_at[1] / 2) as usize);
        if q1 == q2 || !alive[q1] || !alive[q2] {
            continue;
        }

        // Each cell contributes the path round it that does **not** pass
        // through `v`; chained, the three paths are the pentagon.
        let t = tris[ti];
        let k = t.iter().position(|&x| x as usize == v).unwrap();
        let far_tri = [t[(k + 1) % 3], t[(k + 2) % 3]];
        let far_quad = |q: [u32; 4]| {
            let k = q.iter().position(|&x| x as usize == v).unwrap();
            [q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]]
        };
        let (fa, fb) = (far_quad(quads[q1]), far_quad(quads[q2]));
        // The triangle ends where one quadrangle starts, and that one ends
        // where the other starts. Anything else is not a fan round `v`.
        let (first, second) = if far_tri[1] == fa[0] && fa[2] == fb[0] && fb[2] == far_tri[0] {
            (fa, fb)
        } else if far_tri[1] == fb[0] && fb[2] == fa[0] && fa[2] == far_tri[0] {
            (fb, fa)
        } else {
            continue;
        };
        let ring = [far_tri[0], first[0], first[1], first[2], second[1]];
        if ring.iter().collect::<std::collections::HashSet<_>>().len() != 5 {
            continue;
        }
        let p: Vec<Point2> = ring.iter().map(|&i| pts[i as usize]).collect();
        if !super::geom::polygon_is_simple(&p)
            || crate::ops::mesh::triangulation::signed_area(&p) <= 0.0
        {
            continue;
        }

        // What is there now, and what the best cut would give.
        let was = quad_quality([
            pts[quads[q1][0] as usize],
            pts[quads[q1][1] as usize],
            pts[quads[q1][2] as usize],
            pts[quads[q1][3] as usize],
        ])
        .min(quad_quality([
            pts[quads[q2][0] as usize],
            pts[quads[q2][1] as usize],
            pts[quads[q2][2] as usize],
            pts[quads[q2][3] as usize],
        ]))
        .min(super::geom::tri_quality(
            pts[t[0] as usize],
            pts[t[1] as usize],
            pts[t[2] as usize],
        ));
        let mut best: Option<(f64, [u32; 4], [u32; 3])> = None;
        for c in 0..5 {
            // The cut from `ring[c]`: a quadrangle over the next three, and a
            // triangle over the last two.
            let quad = [
                ring[c],
                ring[(c + 1) % 5],
                ring[(c + 2) % 5],
                ring[(c + 3) % 5],
            ];
            let tri = [ring[c], ring[(c + 3) % 5], ring[(c + 4) % 5]];
            let qp = [
                pts[quad[0] as usize],
                pts[quad[1] as usize],
                pts[quad[2] as usize],
                pts[quad[3] as usize],
            ];
            let tp = [
                pts[tri[0] as usize],
                pts[tri[1] as usize],
                pts[tri[2] as usize],
            ];
            if !quad_is_valid(qp) {
                continue;
            }
            let score = quad_quality(qp).min(super::geom::tri_quality(tp[0], tp[1], tp[2]));
            if score > 0.0 && best.is_none_or(|(b, _, _)| score > b) {
                best = Some((score, quad, tri));
            }
        }
        let Some((score, quad, tri)) = best else {
            continue;
        };
        if score <= was {
            continue;
        }
        quads[q1] = quad;
        alive[q2] = false;
        tris[ti] = tri;
        for &r in &ring {
            spent[r as usize] = true;
        }
        spent[v] = true;
        done += 1;
    }
    done
}

/// Merge the two quadrangles around every interior node that has only two.
fn remove_doublets(
    pts: &[Point2],
    movable: &[bool],
    quads: &mut [[u32; 4]],
    tris: &[[u32; 3]],
    alive: &mut [bool],
) -> usize {
    let inc = incidence(quads, tris, alive, pts.len());
    let mut removed = 0;
    for v in 0..pts.len() {
        // A node the caller pinned is on the contour and must stay.
        if !movable[v] || inc[v].len() != 2 || inc[v].iter().any(|e| !e.is_multiple_of(2)) {
            continue;
        }
        let (i1, i2) = ((inc[v][0] / 2) as usize, (inc[v][1] / 2) as usize);
        if !alive[i1] || !alive[i2] || i1 == i2 {
            continue;
        }
        let rot = |q: [u32; 4]| {
            let k = q.iter().position(|&x| x as usize == v).unwrap();
            [q[k], q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]]
        };
        let (q1, q2) = (rot(quads[i1]), rot(quads[i2]));
        // Both cells run counter-clockwise and lie on opposite sides, so a
        // genuine doublet has them sharing exactly the two edges at `v`.
        if q1[1] != q2[3] || q1[3] != q2[1] {
            continue;
        }
        let merged = [q1[1], q1[2], q1[3], q2[2]];
        if !quad_is_valid([
            pts[merged[0] as usize],
            pts[merged[1] as usize],
            pts[merged[2] as usize],
            pts[merged[3] as usize],
        ]) {
            continue;
        }
        quads[i1] = merged;
        alive[i2] = false;
        removed += 1;
    }
    removed
}

/// Re-split pairs of neighbouring quadrangles across a better diagonal.
fn switch_diagonals(
    pts: &[Point2],
    quads: &mut [[u32; 4]],
    tris: &[[u32; 3]],
    alive: &mut [bool],
) -> usize {
    let inc = incidence(quads, tris, alive, pts.len());
    let theta = interior_angles(pts, quads, tris, &inc);
    let mut val: Vec<i64> = inc.iter().map(|e| e.len() as i64).collect();
    let want: Vec<i64> = theta.iter().map(|&t| wanted(t) as i64).collect();

    // Edges shared by exactly two live quadrangles.
    let mut shared: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, q) in quads.iter().enumerate() {
        if !alive[i] {
            continue;
        }
        for t in 0..4 {
            let (a, b) = (q[t], q[(t + 1) % 4]);
            shared
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_default()
                .push(i);
        }
    }
    // Triangle edges are off limits: re-splitting across one would need the
    // triangle rebuilt too.
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            shared
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_default()
                .push(usize::MAX);
        }
    }
    let mut keys: Vec<(u32, u32)> = shared.keys().copied().collect();
    keys.sort_unstable();

    fn err(val: &[i64], want: &[i64], v: usize, d: i64) -> i64 {
        ((val[v] + d) - want[v]).pow(2)
    }
    let mut switches = 0;
    for key in keys {
        let cells = &shared[&key];
        if cells.len() != 2 || cells.contains(&usize::MAX) {
            continue;
        }
        let (i1, i2) = (cells[0], cells[1]);
        if !alive[i1] || !alive[i2] {
            continue;
        }
        let (q1, q2) = (quads[i1], quads[i2]);
        // Orient both on the shared edge: `q1` reads (u, w, c, d) and `q2`
        // reads (w, u, e, f), so the union is the hexagon u e f w c d.
        let Some(a) = (0..4).find(|&t| {
            (q1[t], q1[(t + 1) % 4]) == key || (q1[t], q1[(t + 1) % 4]) == (key.1, key.0)
        }) else {
            continue;
        };
        let (u, w) = (q1[a], q1[(a + 1) % 4]);
        let (c, d) = (q1[(a + 2) % 4], q1[(a + 3) % 4]);
        let Some(b) = (0..4).find(|&t| q2[t] == w && q2[(t + 1) % 4] == u) else {
            continue;
        };
        let (e, f) = (q2[(b + 2) % 4], q2[(b + 3) % 4]);
        let hexagon = [u, e, f, w, c, d];

        let e0 = |v: u32, d: i64| err(&val, &want, v as usize, d);
        let before = e0(u, 0) + e0(w, 0) + e0(e, 0) + e0(c, 0) + e0(f, 0) + e0(d, 0);
        let at = |i: u32| pts[i as usize];
        let worst_now = quad_quality([at(u), at(w), at(c), at(d)]).min(quad_quality([
            at(w),
            at(u),
            at(e),
            at(f),
        ]));

        let mut best: Option<(i64, [[u32; 4]; 2])> = None;
        for shift in [1usize, 2] {
            // Diagonal between hexagon[shift] and hexagon[shift + 3].
            let h = |k: usize| hexagon[(k + shift) % 6];
            let (p, q) = (h(0), h(3));
            let pair = [[h(0), h(1), h(2), h(3)], [h(3), h(4), h(5), h(0)]];
            if !pair
                .iter()
                .all(|c| quad_is_valid([at(c[0]), at(c[1]), at(c[2]), at(c[3])]))
            {
                continue;
            }
            let worst = pair
                .iter()
                .map(|c| quad_quality([at(c[0]), at(c[1]), at(c[2]), at(c[3])]))
                .fold(f64::INFINITY, f64::min);
            if worst < worst_now * QUALITY_FLOOR {
                continue;
            }
            let after = before - e0(u, 0) - e0(w, 0) + e0(u, -1) + e0(w, -1) - e0(p, 0) - e0(q, 0)
                + e0(p, 1)
                + e0(q, 1);
            if after < before && best.is_none_or(|(bs, _)| after < bs) {
                best = Some((after, pair));
            }
        }
        if let Some((_, pair)) = best {
            val[u as usize] -= 1;
            val[w as usize] -= 1;
            val[pair[0][0] as usize] += 1;
            val[pair[0][3] as usize] += 1;
            quads[i1] = pair[0];
            quads[i2] = pair[1];
            switches += 1;
        }
    }
    switches
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A regular grid of `n × n` quadrangles.
    fn grid(n: usize) -> (Vec<Point2>, Vec<bool>, Vec<[u32; 4]>) {
        let w = n + 1;
        let mut pts = Vec::new();
        let mut movable = Vec::new();
        for j in 0..w {
            for i in 0..w {
                pts.push(Point2::new(i as f64, j as f64));
                movable.push(i != 0 && j != 0 && i != w - 1 && j != w - 1);
            }
        }
        let mut quads = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = (j * w + i) as u32;
                quads.push([a, a + 1, a + w as u32 + 1, a + w as u32]);
            }
        }
        (pts, movable, quads)
    }

    #[test]
    fn a_clean_grid_is_left_alone() {
        let (pts, movable, mut quads) = grid(4);
        let before = quads.clone();
        let report = run(&pts, &movable, &mut quads, &mut []);
        assert_eq!(report, Report::default(), "{report:?}");
        assert_eq!(quads, before);
    }

    #[test]
    fn the_wanted_valence_is_four_inside_and_follows_the_corner_outside() {
        // 2π interior, π along a straight boundary, π/2 at a square corner.
        assert_eq!(wanted(std::f64::consts::TAU), 4);
        assert_eq!(wanted(std::f64::consts::PI), 2);
        assert_eq!(wanted(std::f64::consts::FRAC_PI_2), 1);
        // And a grid reads exactly that off its own geometry.
        let (pts, _, quads) = grid(4);
        let inc = incidence(&quads, &[], &vec![true; quads.len()], pts.len());
        let theta = interior_angles(&pts, &quads, &[], &inc);
        let w = 5;
        assert_eq!(wanted(theta[w + 1]), 4, "interior");
        assert_eq!(wanted(theta[1]), 2, "boundary edge");
        assert_eq!(wanted(theta[0]), 1, "corner");
    }

    #[test]
    fn a_doublet_is_absorbed_into_its_neighbour() {
        // One interior quadrangle of a 3×3 grid split by an extra node into
        // two cells sharing two edges — the textbook doublet.
        let (mut pts, mut movable, mut quads) = grid(3);
        let centre = quads[4];
        let mid =
            Point2::from((pts[centre[0] as usize].coords + pts[centre[2] as usize].coords) * 0.5);
        let v = pts.len() as u32;
        pts.push(mid);
        movable.push(true);
        quads[4] = [centre[0], centre[1], centre[2], v];
        quads.push([centre[2], centre[3], centre[0], v]);

        let n_before = quads.len();
        let report = run(&pts, &movable, &mut quads, &mut []);
        assert_eq!(report.doublets, 1, "{report:?}");
        assert_eq!(quads.len(), n_before - 1);
        // The node is gone from the connectivity.
        assert!(quads.iter().all(|q| !q.contains(&v)));
        // And the original cell is back.
        assert!(quads.iter().any(|q| {
            let mut s = *q;
            s.sort_unstable();
            let mut t = centre;
            t.sort_unstable();
            s == t
        }));
    }

    #[test]
    fn a_triangle_wedged_between_two_quadrangles_gives_its_node_up() {
        // The defect nothing else here reaches: a small triangle and two
        // quadrangles round one interior node. The doublet rule wants two
        // cells, the diagonal switch wants two quadrangles, and both leave
        // triangles alone — so the node keeps its three cells and its wedge.
        //
        // The three cover a pentagon, whose only decomposition is one
        // quadrangle and one triangle. Two cells for three, and the triangle
        // that comes out is the whole wedge instead of the sliver in it.
        //
        // Five points on a circle, and a sixth pushed right up against one
        // side, which is what makes the triangle the sliver it is.
        let ring: Vec<Point2> = (0..5)
            .map(|i| {
                let t = i as f64 / 5.0 * std::f64::consts::TAU;
                Point2::new(t.cos(), t.sin())
            })
            .collect();
        let mut pts = ring.clone();
        pts.push(Point2::from((ring[0].coords + ring[1].coords) * 0.475));
        let v = 5u32;
        let movable = vec![false, false, false, false, false, true];
        // The fan round `v`, walked the same way as the pentagon.
        let mut quads = vec![[v, 1, 2, 3], [v, 3, 4, 0]];
        let mut tris = vec![[v, 0, 1]];

        let report = run(&pts, &movable, &mut quads, &mut tris);
        assert_eq!(report.absorbed, 1, "{report:?}");
        assert_eq!(quads.len(), 1);
        assert_eq!(tris.len(), 1);
        // The node is gone from every cell.
        assert!(quads.iter().all(|q| !q.contains(&v)));
        assert!(tris.iter().all(|t| !t.contains(&v)));
        // And the two cells still cover the pentagon exactly.
        let area = |p: &[Point2]| {
            0.5 * (0..p.len())
                .map(|i| p[i].x * p[(i + 1) % p.len()].y - p[(i + 1) % p.len()].x * p[i].y)
                .sum::<f64>()
        };
        let q = quads[0].map(|i| pts[i as usize]);
        let t = tris[0].map(|i| pts[i as usize]);
        assert!((area(&q) + area(&t) - area(&ring)).abs() < 1e-12);
    }

    #[test]
    fn a_switch_lowers_the_valence_error_and_keeps_the_mesh_conforming() {
        let (pts, movable, mut quads) = grid(4);
        // Cells 5 and 6 of a 4×4 grid are [6, 7, 12, 11] and [7, 8, 13, 12],
        // sharing the edge (7, 12). Re-split their union across a corner
        // diagonal instead: nodes 7 and 12 drop to valence 3, nodes 8 and 11
        // climb to 5, and no amount of smoothing can put that right.
        assert_eq!(quads[5], [6, 7, 12, 11]);
        assert_eq!(quads[6], [7, 8, 13, 12]);
        quads[5] = [8, 13, 12, 11];
        quads[6] = [11, 6, 7, 8];

        let report = run(&pts, &movable, &mut quads, &mut []);
        assert!(report.switches >= 1, "{report:?}");

        // The grid is back: every interior node has four cells again.
        let inc = incidence(&quads, &[], &vec![true; quads.len()], pts.len());
        for v in [7usize, 12, 8, 11] {
            assert_eq!(inc[v].len(), 4, "node {v} still has {}", inc[v].len());
        }

        // And every interior edge still carries exactly two cells.
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for q in &quads {
            for t in 0..4 {
                let (a, b) = (q[t], q[(t + 1) % 4]);
                *count
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        assert!(count.values().all(|&c| c <= 2), "an edge grew a third cell");
    }
}
