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
//! - an interior node with only **three cells** around it, whatever they are.
//!   Its corners average 120°, and no smoothing will square them because the
//!   angles round a node sum to 2π whatever the positions. Neither other move
//!   here reaches it: the doublet rule wants two cells, the diagonal switch
//!   wants two quadrangles. But the star of such a node always re-cuts with a
//!   cell fewer — see [`collapse_valence3`] for why, and for the four shapes
//!   it takes — so the node is given up along with the cell its star can do
//!   without.
//! - a node with the **wrong valence** otherwise. In a quadrangle mesh an
//!   interior node wants four cells around it: with five, the corners average
//!   72°, which no smoothing squares either.
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
//! ## The moves
//!
//! One of them moves nodes, the other does not, and the difference is not a
//! detail: a switch leaves every node where smoothing put it, so what it
//! measures is final, while a collapse *removes* a node and cannot help
//! leaving the ring round it stretched. Judged on the spot, every collapse
//! worth making looks like a mistake.
//!
//! For the valence, the move is the **diagonal switch**: two quadrangles
//! sharing an edge form a hexagon, and a hexagon splits into two quadrangles
//! across any of its three diagonals. Switching diagonal moves one unit of
//! valence from the two nodes on the old diagonal to the two on the new one.
//! It changes no node count and no boundary, so it cannot make the mesh
//! non-conforming, and it is applied only when it strictly lowers the total
//! valence error and leaves both cells convex.
//!
//! For a node that has three cells and no business having them, the move is
//! the **collapse**: the star is thrown away and its boundary re-cut with a
//! cell fewer. It is the only move here that removes a node, and therefore the
//! only one judged **after** a relaxation of the ring it just sewed — kept
//! along with the move, undone whole with it.
//!
//! ## Which way the cells turn
//!
//! Every quality measure below is signed, so a cell read clockwise scores
//! negative and is taken for inverted. The pavers keep their fabric
//! counter-clockwise throughout and check it on the way out, and the entry
//! point for an outside mesh normalises the winding before handing it over —
//! see [`Surface::read`](crate::ops::mesh::improve::Surface). Nothing here
//! needs to ask.

use super::geom::{quad_is_valid, quad_quality};
use crate::atoms::{Point2, Vector2};
use std::collections::HashMap;

/// Passes over the mesh. Each one is a sweep of doublet removal followed by a
/// sweep of diagonal switches; a pass that changes nothing ends the loop.
const ROUNDS: usize = 6;

/// A switch must not drop the worse of the two cells below this share of what
/// it was, so valence is never bought at the price of a sliver.
///
/// It applies to the switch and not to the collapse, because the two are not
/// the same kind of move: a switch leaves every node exactly where smoothing
/// put it, so what it measures is final, while a collapse *removes* a node and
/// necessarily leaves the ring stretched until something relaxes it. Measuring
/// a collapse straight away measures a state no caller ever sees; it is judged
/// after a trial relaxation instead — see [`collapse_valence3`].
const QUALITY_FLOOR: f64 = 0.7;

/// Sweeps of the trial relaxation a collapse is judged after. The ring has at
/// most six nodes, so this costs nothing and there is no reason to skimp.
const TRIAL_SWEEPS: usize = 8;

/// Under-relaxation of that trial, the pavers' own figure.
const TRIAL_RELAX: f64 = 0.6;

/// What the cleanup changed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub doublets: usize,
    pub switches: usize,
    /// Nodes given up by [`absorb_triangle_corners`].
    pub absorbed: usize,
}

/// Clean up `quads` and `tris` in place.
///
/// Triangles are read for incidence, reshaped where a collapsed node leaves a
/// better one behind, and dropped **two at a time** by the collapse of a node
/// whose star holds two of them — which is the only count that keeps the
/// parity tying `T` to the boundary's edge count. Turning a lone triangle into
/// a quadrangle is [`merge_triangles`][mt]'s job, not this one.
///
/// Both lists come back compacted.
///
/// [mt]: fn@crate::ops::mesh::improve::merge_triangles::merge_triangles
pub fn run(
    pts: &mut [Point2],
    movable: &[bool],
    quads: &mut Vec<[u32; 4]>,
    tris: &mut Vec<[u32; 3]>,
) -> Report {
    let mut report = Report::default();
    let mut alive = vec![true; quads.len()];
    let mut alive_tri = vec![true; tris.len()];
    for _ in 0..ROUNDS {
        let before = (report.doublets, report.switches, report.absorbed);
        report.doublets += remove_doublets(pts, movable, quads, tris, &mut alive, &alive_tri);
        report.absorbed += collapse_valence3(pts, movable, quads, tris, &mut alive, &mut alive_tri);
        report.switches += switch_diagonals(pts, quads, tris, &mut alive, &alive_tri);
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
    let mut kept = Vec::with_capacity(tris.len());
    for (i, t) in tris.iter().enumerate() {
        if alive_tri[i] {
            kept.push(*t);
        }
    }
    *tris = kept;
    report
}

/// Cells around each node: `2 * index` for a live quadrangle, `2 * index + 1`
/// for a live triangle.
fn incidence(
    quads: &[[u32; 4]],
    tris: &[[u32; 3]],
    alive: &[bool],
    alive_tri: &[bool],
    n_pts: usize,
) -> Vec<Vec<u32>> {
    let mut inc = vec![Vec::new(); n_pts];
    for (i, q) in quads.iter().enumerate() {
        if alive[i] {
            for &v in q {
                inc[v as usize].push((i as u32) * 2);
            }
        }
    }
    for (i, t) in tris.iter().enumerate() {
        if alive_tri[i] {
            for &v in t {
                inc[v as usize].push((i as u32) * 2 + 1);
            }
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

/// Give up an interior node that has only three cells around it, along with
/// the one cell its star can do without.
///
/// Round a node with `q` quadrangles and `t` triangles, each quadrangle lays
/// two edges that do not touch the node and each triangle lays one: the star
/// is bounded by a polygon of **`n = 2q + t`** sides. And a decomposition of
/// an `n`-gon into `q'` quadrangles and `t'` triangles with no interior node
/// satisfies `2q' + t' = n - 2`. Put the two together and the re-cut always
/// exists, always with fewer cells than the star it replaces:
///
/// | `q, t` | boundary | before | after | cells |
/// |---|---|---|---|---|
/// | 3, 0 | hexagon | 3 quadrangles | 2 quadrangles | 3 → 2 |
/// | 2, 1 | pentagon | 2 quadrangles, 1 triangle | 1 of each | 3 → 2 |
/// | 1, 2 | quadrangle | 1 quadrangle, 2 triangles | 1 quadrangle | 3 → 1 |
/// | 0, 3 | triangle | 3 triangles | 1 triangle | 3 → 1 |
///
/// Three cells round an interior node is one too few whatever they are — the
/// corners average 120° and no smoothing will square them, since the angles
/// round a node sum to 2π whatever the positions. Neither other move here can
/// reach the defect: the doublet rule wants two cells and the diagonal switch
/// wants two quadrangles. This one takes all four cases at once, and it is the
/// only pass that removes a node.
///
/// The bottom two rows shed a triangle each time — two of them, so the parity
/// that ties `T` to the boundary's edge count is untouched, and neither the
/// count nor the boundary can drift.
///
/// The cut is taken across whichever rotation leaves the mesh best, and the
/// move is applied when **either** the valence error strictly falls **or** the
/// worst of the new cells beats the worst of the old — a node may be worth
/// giving up for its shape alone, and an ill-placed one is worth giving up
/// even when the shape barely moves. It is refused outright when a cell would
/// come out inverted.
fn collapse_valence3(
    pts: &mut [Point2],
    movable: &[bool],
    quads: &mut [[u32; 4]],
    tris: &mut [[u32; 3]],
    alive: &mut [bool],
    alive_tri: &mut [bool],
) -> usize {
    // Kept up to date as the pass goes: a collapse rewrites the cells round
    // its ring, and the neighbourhood of every later candidate is read from
    // here. Refusing to touch a ring an earlier collapse reached would be the
    // cheap answer, and an expensive one — it costs more than half the moves.
    let mut inc = incidence(quads, tris, alive, alive_tri, pts.len());
    let theta = interior_angles(pts, quads, tris, &inc);
    let val: Vec<i64> = inc.iter().map(|e| e.len() as i64).collect();
    let want: Vec<i64> = theta.iter().map(|&t| wanted(t) as i64).collect();

    let mut done = 0;
    let mut spent = vec![false; pts.len()];
    for v in 0..pts.len() {
        if !movable[v] || inc[v].len() != 3 || spent[v] {
            continue;
        }
        let cells = inc[v].clone();
        if cells
            .iter()
            .any(|&e| !live(e, alive, alive_tri) || count_of(&cells, e) != 1)
        {
            continue;
        }
        let Some(ring) = star_ring(v, &cells, quads, tris) else {
            continue;
        };
        let p: Vec<Point2> = ring.iter().map(|&i| pts[i as usize]).collect();
        if !super::geom::polygon_is_simple(&p)
            || crate::ops::mesh::triangulation::signed_area(&p) <= 0.0
        {
            continue;
        }

        // What the star is worth now, and what it costs in valence.
        let was = cells
            .iter()
            .map(|&e| quality_of(e, pts, quads, tris))
            .fold(f64::INFINITY, f64::min);
        let mut had = vec![0i64; ring.len()];
        for &e in &cells {
            for &x in nodes_of(e, quads, tris) {
                if x as usize != v
                    && let Some(k) = ring.iter().position(|&r| r == x)
                {
                    had[k] += 1;
                }
            }
        }

        let mut best: Option<(i64, f64, Cut)> = None;
        for c in 0..cut_count(ring.len()) {
            let cut = cut_at(&ring, c);
            let worst = cut.worst_quality(pts);
            if worst <= 0.0 {
                continue;
            }
            let mut gets = vec![0i64; ring.len()];
            cut.for_each_node(|x| {
                if let Some(k) = ring.iter().position(|&r| r == x) {
                    gets[k] += 1;
                }
            });
            // `v` goes, so its own error goes with it; every ring node moves by
            // what it gains less what it had.
            let mut gain = (val[v] - want[v]).pow(2);
            for (k, &r) in ring.iter().enumerate() {
                let r = r as usize;
                let after = val[r] + gets[k] - had[k];
                gain += (val[r] - want[r]).pow(2) - (after - want[r]).pow(2);
            }
            if best.is_none_or(|(bg, bq, _)| (gain, worst) > (bg, bq)) {
                best = Some((gain, worst, cut));
            }
        }
        let Some((gain, worst, cut)) = best else {
            continue;
        };
        // Nothing to gain, nothing to try. The trial below judges the
        // *neighbourhood*, which is local by construction and cannot see that a
        // cell already middling somewhere else has become the worst in the
        // mesh; this asks first whether the move is worth anything at all, in
        // valence or in shape, and most of what it turns away is worth nothing.
        //
        // It costs a few good moves — dropping it takes the boîte from 185
        // irregular nodes to 162 — and buys the guarantee that the worst cell
        // never goes backwards: without it that same mesh falls from 0,461 to
        // 0,436, below the 0,456 it started at, and `grid_surface` on the
        // house from 0,420 to 0,346. A worst cell that recedes is what breaks
        // a computation; thirteen irregular nodes are not.
        if gain <= 0 && worst <= was {
            continue;
        }

        // The cells beyond the star that the relaxation can still reach, and
        // what the neighbourhood is worth as it stands.
        let mut around: Vec<u32> = Vec::new();
        for &r in &ring {
            for &e in &inc[r as usize] {
                if !cells.contains(&e) && !around.contains(&e) && live(e, alive, alive_tri) {
                    around.push(e);
                }
            }
        }
        let worst_before = around
            .iter()
            .chain(cells.iter())
            .map(|&e| quality_of(e, pts, quads, tris))
            .fold(f64::INFINITY, f64::min);

        // The star dies, and the cut moves into its slots. It always fits:
        // `2q' + t' = 2q + t - 2` with `t' ≤ 1` leaves `q' ≤ q` and `t' ≤ t`
        // in every one of the four cases, so nothing is ever added — the
        // slots the cut does not want simply stay dead.
        let (mut free_q, mut free_t) = (Vec::new(), Vec::new());
        let (mut was_q, mut was_t) = (Vec::new(), Vec::new());
        for &e in &cells {
            let i = (e / 2) as usize;
            if e.is_multiple_of(2) {
                was_q.push((i, quads[i]));
                alive[i] = false;
                free_q.push(i);
            } else {
                was_t.push((i, tris[i]));
                alive_tri[i] = false;
                free_t.push(i);
            }
        }
        let mut now = around.clone();
        for (q, &i) in cut.quads().iter().zip(free_q.iter()) {
            quads[i] = *q;
            alive[i] = true;
            now.push((i as u32) * 2);
        }
        for (t, &i) in cut.tris().iter().zip(free_t.iter()) {
            tris[i] = *t;
            alive_tri[i] = true;
            now.push((i as u32) * 2 + 1);
        }

        // Judged on the mesh it becomes, not the one it leaves behind for an
        // instant — and the relaxation that gets it there is *kept*, so what
        // was measured is what the caller receives. Measuring positions one
        // then throws away is measuring a mesh nobody gets: the move is taken
        // on the strength of a relaxation that never happens, and the cell it
        // was supposed to save stays poor.
        let held: Vec<Point2> = ring.iter().map(|&r| pts[r as usize]).collect();
        relax_ring(pts, movable, &ring, &now, quads, tris);
        let worst_after = now
            .iter()
            .map(|&e| quality_of(e, pts, quads, tris))
            .fold(f64::INFINITY, f64::min);
        if worst_after < worst_before {
            for (&r, &p) in ring.iter().zip(held.iter()) {
                pts[r as usize] = p;
            }
            for &(i, q) in &was_q {
                quads[i] = q;
                alive[i] = true;
            }
            for &(i, t) in &was_t {
                tris[i] = t;
                alive_tri[i] = true;
            }
            continue;
        }

        // The star's cells left the ring's incidence lists and the cut's
        // joined them; `v` is out of the mesh altogether.
        for &r in &ring {
            let l = &mut inc[r as usize];
            l.retain(|e| !cells.contains(e));
            for &e in &now {
                if !l.contains(&e) && nodes_of(e, quads, tris).contains(&r) {
                    l.push(e);
                }
            }
            spent[r as usize] = true;
        }
        inc[v].clear();
        spent[v] = true;
        done += 1;
    }
    done
}

/// Relax the ring's movable nodes over the cells that now cover it.
///
/// A collapse *removes* a node, so the cells round it come out stretched and
/// stay that way until something moves them — which something always does: the
/// pavers smooth after every row, and `regularize` is the caller's next step.
/// Measuring the move before that measures a state nobody ever receives, and
/// refuses the very moves that pay off most.
///
/// So the ring is relaxed and the move judged on the result — and when the
/// move is kept the relaxation is kept with it, since a verdict on positions
/// one then discards is a verdict on a mesh nobody receives.
///
/// The sweep carries the pavers' own guard — a step is taken only if every
/// cell round the node stays the right way round. Without it the trial is a
/// bare Laplacian, which near a concave corner walks to a place the real
/// smoother, monotone and guarded, will never reach: the trial then reports a
/// move as good on the strength of a relaxation that does not happen, and the
/// cell it was meant to save is left at whatever the collapse made of it.
fn relax_ring(
    pts: &mut [Point2],
    movable: &[bool],
    ring: &[u32],
    patch: &[u32],
    quads: &[[u32; 4]],
    tris: &[[u32; 3]],
) {
    let mut neighbours: Vec<(u32, Vec<u32>)> = Vec::new();
    for &r in ring {
        if !movable[r as usize] {
            continue;
        }
        let mut nb: Vec<u32> = Vec::new();
        for &e in patch {
            let c = nodes_of(e, quads, tris);
            let n = c.len();
            let Some(k) = c.iter().position(|&x| x == r) else {
                continue;
            };
            for x in [c[(k + 1) % n], c[(k + n - 1) % n]] {
                if !nb.contains(&x) {
                    nb.push(x);
                }
            }
        }
        if !nb.is_empty() {
            neighbours.push((r, nb));
        }
    }
    for _ in 0..TRIAL_SWEEPS {
        for (r, nb) in &neighbours {
            let mut c = Vector2::zeros();
            for &x in nb {
                c += pts[x as usize].coords;
            }
            c /= nb.len() as f64;
            let at = pts[*r as usize];
            let cand = Point2::from(at.coords * (1.0 - TRIAL_RELAX) + c * TRIAL_RELAX);
            pts[*r as usize] = cand;
            if !patch_is_valid(pts, patch, *r, quads, tris) {
                pts[*r as usize] = at;
            }
        }
    }
}

/// Whether every cell of `patch` that uses `r` is still the right way round.
fn patch_is_valid(
    pts: &[Point2],
    patch: &[u32],
    r: u32,
    quads: &[[u32; 4]],
    tris: &[[u32; 3]],
) -> bool {
    patch.iter().all(|&e| {
        let c = nodes_of(e, quads, tris);
        !c.contains(&r) || quality_of(e, pts, quads, tris) > 0.0
    })
}

/// A re-cut of a star's boundary ring: at most two quadrangles and one
/// triangle, which is all `2q' + t' = n - 2` allows for `n ≤ 6`.
#[derive(Clone, Copy)]
struct Cut {
    quads: [[u32; 4]; 2],
    n_quads: usize,
    tri: Option<[u32; 3]>,
}

impl Cut {
    fn quads(&self) -> &[[u32; 4]] {
        &self.quads[..self.n_quads]
    }

    fn tris(&self) -> &[[u32; 3]] {
        self.tri.as_slice()
    }

    fn worst_quality(&self, pts: &[Point2]) -> f64 {
        let at = |i: u32| pts[i as usize];
        let mut worst = f64::INFINITY;
        for q in self.quads() {
            let p = [at(q[0]), at(q[1]), at(q[2]), at(q[3])];
            if !quad_is_valid(p) {
                return 0.0;
            }
            worst = worst.min(quad_quality(p));
        }
        for t in self.tris() {
            worst = worst.min(super::geom::tri_quality(at(t[0]), at(t[1]), at(t[2])));
        }
        worst
    }

    fn for_each_node(&self, mut f: impl FnMut(u32)) {
        for q in self.quads() {
            q.iter().for_each(|&x| f(x));
        }
        for t in self.tris() {
            t.iter().for_each(|&x| f(x));
        }
    }
}

/// How many distinct minimal cuts an `n`-gon has: one per rotation that gives
/// a different answer, and `0` for a ring this pass does not handle.
fn cut_count(n: usize) -> usize {
    match n {
        6 => 3, // the three main diagonals
        5 => 5, // which corner the triangle is taken from
        3 | 4 => 1,
        _ => 0,
    }
}

/// The `c`-th minimal cut of `ring`, read from `ring[c]` onwards.
fn cut_at(ring: &[u32], c: usize) -> Cut {
    let n = ring.len();
    let at = |k: usize| ring[(k + c) % n];
    let none = [[0u32; 4]; 2];
    match n {
        6 => Cut {
            quads: [[at(0), at(1), at(2), at(3)], [at(3), at(4), at(5), at(0)]],
            n_quads: 2,
            tri: None,
        },
        5 => Cut {
            quads: [[at(0), at(1), at(2), at(3)], [0; 4]],
            n_quads: 1,
            tri: Some([at(0), at(3), at(4)]),
        },
        4 => Cut {
            quads: [[at(0), at(1), at(2), at(3)], [0; 4]],
            n_quads: 1,
            tri: None,
        },
        _ => Cut {
            quads: none,
            n_quads: 0,
            tri: Some([at(0), at(1), at(2)]),
        },
    }
}

/// The polygon bounding `v`'s star, walked once round — or `None` when the
/// cells do not chain into a fan, which is what a non-manifold node looks like
/// from here.
///
/// Each cell contributes the path round it that does **not** pass through `v`,
/// and the end of one is the start of the next.
fn star_ring(v: usize, cells: &[u32], quads: &[[u32; 4]], tris: &[[u32; 3]]) -> Option<Vec<u32>> {
    let path = |e: u32| -> Vec<u32> {
        let c = nodes_of(e, quads, tris);
        let n = c.len();
        let k = c.iter().position(|&x| x as usize == v).unwrap();
        (1..n).map(|j| c[(k + j) % n]).collect()
    };
    let mut left: Vec<Vec<u32>> = cells.iter().map(|&e| path(e)).collect();
    let mut chain = vec![left.remove(0)];
    while !left.is_empty() {
        let tail = *chain.last()?.last()?;
        let i = left.iter().position(|p| p[0] == tail)?;
        chain.push(left.remove(i));
    }
    if *chain.last()?.last()? != chain[0][0] {
        return None;
    }
    let mut ring = vec![chain[0][0]];
    for p in &chain {
        ring.extend_from_slice(&p[1..]);
    }
    ring.pop(); // the walk comes back to where it started
    (ring.iter().collect::<std::collections::HashSet<_>>().len() == ring.len()).then_some(ring)
}

/// The nodes of the cell `e` refers to.
fn nodes_of<'a>(e: u32, quads: &'a [[u32; 4]], tris: &'a [[u32; 3]]) -> &'a [u32] {
    if e.is_multiple_of(2) {
        &quads[(e / 2) as usize]
    } else {
        &tris[(e / 2) as usize]
    }
}

/// The shape quality of the cell `e` refers to.
fn quality_of(e: u32, pts: &[Point2], quads: &[[u32; 4]], tris: &[[u32; 3]]) -> f64 {
    let at = |i: u32| pts[i as usize];
    if e.is_multiple_of(2) {
        let q = quads[(e / 2) as usize];
        quad_quality([at(q[0]), at(q[1]), at(q[2]), at(q[3])])
    } else {
        let t = tris[(e / 2) as usize];
        super::geom::tri_quality(at(t[0]), at(t[1]), at(t[2]))
    }
}

/// Whether the cell `e` refers to is still in the mesh.
fn live(e: u32, alive: &[bool], alive_tri: &[bool]) -> bool {
    if e.is_multiple_of(2) {
        alive[(e / 2) as usize]
    } else {
        alive_tri[(e / 2) as usize]
    }
}

/// How many times `e` appears in `cells` — more than once means the same cell
/// touches the node twice, which is not a fan.
fn count_of(cells: &[u32], e: u32) -> usize {
    cells.iter().filter(|&&x| x == e).count()
}

/// Merge the two quadrangles around every interior node that has only two.
fn remove_doublets(
    pts: &[Point2],
    movable: &[bool],
    quads: &mut [[u32; 4]],
    tris: &[[u32; 3]],
    alive: &mut [bool],
    alive_tri: &[bool],
) -> usize {
    let inc = incidence(quads, tris, alive, alive_tri, pts.len());
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
    alive_tri: &[bool],
) -> usize {
    let inc = incidence(quads, tris, alive, alive_tri, pts.len());
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
    for (i, t) in tris.iter().enumerate() {
        if !alive_tri[i] {
            continue;
        }
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
        let (mut pts, movable, mut quads) = grid(4);
        let before = quads.clone();
        let report = run(&mut pts, &movable, &mut quads, &mut Vec::new());
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
        let inc = incidence(&quads, &[], &vec![true; quads.len()], &[], pts.len());
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
        let report = run(&mut pts, &movable, &mut quads, &mut Vec::new());
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

        let report = run(&mut pts, &movable, &mut quads, &mut tris);
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

    /// The six points of a regular hexagon, counter-clockwise.
    fn hexagon() -> Vec<Point2> {
        (0..6)
            .map(|i| {
                let t = i as f64 / 6.0 * std::f64::consts::TAU;
                Point2::new(t.cos(), t.sin())
            })
            .collect()
    }

    #[test]
    fn three_quadrangles_round_a_node_become_two() {
        // The case the pass was generalised for, and the one a paver leaves
        // most of: no triangle in sight, three quadrangles fanned round one
        // interior node. Their outline is a hexagon, a hexagon splits into two
        // quadrangles across any of its three main diagonals, and the node
        // goes with the third cell.
        //
        // The node is pushed hard against one side, which is what makes the
        // three cells worse than the two that replace them.
        let ring = hexagon();
        let mut pts = ring.clone();
        pts.push(Point2::from((ring[0].coords + ring[1].coords) * 0.5 * 0.92));
        let v = 6u32;
        let movable = vec![false, false, false, false, false, false, true];
        let mut quads = vec![[v, 0, 1, 2], [v, 2, 3, 4], [v, 4, 5, 0]];
        let mut tris: Vec<[u32; 3]> = Vec::new();

        let report = run(&mut pts, &movable, &mut quads, &mut tris);
        assert_eq!(report.absorbed, 1, "{report:?}");
        assert_eq!(quads.len(), 2, "three cells for two");
        assert!(tris.is_empty(), "no triangle may appear");
        assert!(quads.iter().all(|q| !q.contains(&v)), "the node is gone");

        // The two still cover the hexagon exactly, so the boundary is intact.
        let area = |p: &[Point2]| {
            0.5 * (0..p.len())
                .map(|i| p[i].x * p[(i + 1) % p.len()].y - p[(i + 1) % p.len()].x * p[i].y)
                .sum::<f64>()
        };
        let covered: f64 = quads
            .iter()
            .map(|q| area(&q.map(|i| pts[i as usize])))
            .sum();
        assert!((covered - area(&ring)).abs() < 1e-12, "{covered}");
    }

    #[test]
    fn a_quadrangle_and_two_triangles_round_a_node_become_one_quadrangle() {
        // The best of the four cases: the outline of the star is a
        // *quadrangle*, so one cell replaces three and the two triangles leave
        // together — which is the only count that keeps the parity tying `T`
        // to the boundary's edge count.
        let ring: Vec<Point2> = vec![
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(-1.0, 0.0),
            Point2::new(0.0, -1.0),
        ];
        let mut pts = ring.clone();
        pts.push(Point2::new(0.0, -0.4)); // the node, well off centre
        let v = 4u32;
        let movable = vec![false, false, false, false, true];
        let mut quads = vec![[v, 0, 1, 2]];
        let mut tris = vec![[v, 2, 3], [v, 3, 0]];

        let report = run(&mut pts, &movable, &mut quads, &mut tris);
        assert_eq!(report.absorbed, 1, "{report:?}");
        assert_eq!(quads.len(), 1, "one cell for three");
        assert_eq!(tris.len(), 0, "both triangles go, and they go together");
        assert!(!quads[0].contains(&v), "the node is gone");
        let mut got = quads[0];
        got.sort_unstable();
        assert_eq!(got, [0, 1, 2, 3], "the cell is the ring itself");
    }

    #[test]
    fn three_triangles_round_a_node_become_one() {
        // The fourth case, and the one the rule reaches without being told to:
        // no quadrangle in sight, so `n = 2q + t` is 3 and the star's outline
        // is a plain triangle. One cell replaces three, and two triangles
        // leave together.
        let ring: Vec<Point2> = (0..3)
            .map(|i| {
                let t = i as f64 / 3.0 * std::f64::consts::TAU;
                Point2::new(t.cos(), t.sin())
            })
            .collect();
        let mut pts = ring.clone();
        pts.push(Point2::from((ring[0].coords + ring[1].coords) * 0.5 * 0.96));
        let v = 3u32;
        let movable = vec![false, false, false, true];
        let mut quads: Vec<[u32; 4]> = Vec::new();
        let mut tris = vec![[v, 0, 1], [v, 1, 2], [v, 2, 0]];

        let report = run(&mut pts, &movable, &mut quads, &mut tris);
        assert_eq!(report.absorbed, 1, "{report:?}");
        assert!(quads.is_empty(), "no quadrangle may appear from nothing");
        assert_eq!(tris.len(), 1, "one cell for three");
        let mut got = tris[0];
        got.sort_unstable();
        assert_eq!(got, [0, 1, 2], "the cell is the ring itself");
    }

    #[test]
    fn the_ring_of_a_star_has_2q_plus_t_sides() {
        // The identity the whole pass rests on: a quadrangle lays two edges
        // that do not touch the node, a triangle lays one.
        // Cells are referred to as `2 * index` for a quadrangle and
        // `2 * index + 1` for a triangle, which is what the numbers below are.
        let v = 9usize;
        let q3 = [[9u32, 0, 1, 2], [9, 2, 3, 4], [9, 4, 5, 0]];
        let ring = star_ring(v, &[0, 2, 4], &q3, &[]).expect("three quadrangles");
        assert_eq!(ring.len(), 6, "3 quadrangles → hexagon");

        let q2 = [[9u32, 0, 1, 2], [9, 2, 3, 4]];
        let t1 = [[9u32, 4, 0]];
        let ring = star_ring(v, &[0, 2, 1], &q2, &t1).expect("two quadrangles, one triangle");
        assert_eq!(ring.len(), 5, "2 quadrangles + 1 triangle → pentagon");

        let q1 = [[9u32, 0, 1, 2]];
        let t2 = [[9u32, 2, 3], [9, 3, 0]];
        let ring = star_ring(v, &[0, 1, 3], &q1, &t2).expect("one quadrangle, two triangles");
        assert_eq!(ring.len(), 4, "1 quadrangle + 2 triangles → quadrangle");

        let t3 = [[9u32, 0, 1], [9, 1, 2], [9, 2, 0]];
        let ring = star_ring(v, &[1, 3, 5], &[], &t3).expect("three triangles");
        assert_eq!(ring.len(), 3, "3 triangles → triangle");

        // And a fan that does not close is refused rather than guessed at.
        assert!(star_ring(v, &[0, 2, 0], &q3, &[]).is_none());
    }

    #[test]
    fn a_switch_lowers_the_valence_error_and_keeps_the_mesh_conforming() {
        let (mut pts, movable, mut quads) = grid(4);
        // Cells 5 and 6 of a 4×4 grid are [6, 7, 12, 11] and [7, 8, 13, 12],
        // sharing the edge (7, 12). Re-split their union across a corner
        // diagonal instead: nodes 7 and 12 drop to valence 3, nodes 8 and 11
        // climb to 5, and no amount of smoothing can put that right.
        assert_eq!(quads[5], [6, 7, 12, 11]);
        assert_eq!(quads[6], [7, 8, 13, 12]);
        quads[5] = [8, 13, 12, 11];
        quads[6] = [11, 6, 7, 8];

        let report = run(&mut pts, &movable, &mut quads, &mut Vec::new());
        assert!(report.switches >= 1, "{report:?}");

        // The grid is back: every interior node has four cells again.
        let inc = incidence(&quads, &[], &vec![true; quads.len()], &[], pts.len());
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
