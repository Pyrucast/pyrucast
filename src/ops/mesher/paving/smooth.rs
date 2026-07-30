//! Constrained smoothing of a quadrangle-dominant mesh.
//!
//! Paving places nodes from the front's geometry alone, which is a good guess
//! and no more: the row it lays next, and the seams that fire around it, both
//! change what a good position would have been. Smoothing is what turns those
//! guesses into a mesh, and it runs between rows rather than only at the end,
//! so a distorted patch is corrected while it is still small.
//!
//! Two rules make it safe to run anywhere:
//!
//! - a node the caller pinned — every node of the user's contour — never
//!   moves, so the meshed domain keeps exactly the boundary it was given;
//! - a move is applied only if every incident element stays valid *and* the
//!   worst incident quality does not get worse. A plain Laplacian pass has
//!   neither guarantee and will happily turn a valid quadrangle inside out
//!   near a concave boundary.
//!
//! The sweep is Jacobi, not Gauss-Seidel: every candidate is computed from the
//! same previous positions, so the result does not depend on node numbering or
//! on how the work was split across threads. That independence has a price —
//! two moves that are each safe against the *old* positions can be unsafe
//! together — so every sweep ends with a repair pass that puts back any node
//! sitting on an element the sweep turned invalid. Reverting always restores a
//! configuration known to be good, so the repair terminates, and the sweep
//! cannot hand back a mesh worse formed than the one it was given.

use super::geom::{quad_quality, tri_quality};
use crate::containers::mesh::{Point2, Vector2};
use crate::parallel::*;

/// Relaxation applied to the Laplacian step. Under-relaxing keeps the pass
/// from overshooting on the freshly laid row, where neighbours are themselves
/// still moving.
const RELAX: f64 = 0.6;

/// Passes of the repair loop. Each one reverts at least one node, so the
/// configuration walks back toward the sweep's starting point and cannot cycle.
const REPAIR_ROUNDS: usize = 8;

/// The mesh being smoothed, as flat arrays.
pub struct Patch<'a> {
    pub quads: &'a [[u32; 4]],
    pub tris: &'a [[u32; 3]],
    /// `false` for a node that must not move.
    pub movable: &'a [bool],
}

/// Per-node incidence, built once and reused across sweeps.
pub struct Incidence {
    ring_start: Vec<u32>,
    ring: Vec<u32>,
    elem_start: Vec<u32>,
    /// Element references, encoded as `2 * index` for a quadrangle and
    /// `2 * index + 1` for a triangle.
    elem: Vec<u32>,
}

impl Incidence {
    pub fn build(patch: &Patch, n_pts: usize) -> Incidence {
        let mut ring_sets: Vec<Vec<u32>> = vec![Vec::new(); n_pts];
        let mut elems: Vec<Vec<u32>> = vec![Vec::new(); n_pts];
        for (i, q) in patch.quads.iter().enumerate() {
            for t in 0..4 {
                let v = q[t] as usize;
                ring_sets[v].push(q[(t + 1) % 4]);
                ring_sets[v].push(q[(t + 3) % 4]);
                elems[v].push((i as u32) * 2);
            }
        }
        for (i, tri) in patch.tris.iter().enumerate() {
            for t in 0..3 {
                let v = tri[t] as usize;
                ring_sets[v].push(tri[(t + 1) % 3]);
                ring_sets[v].push(tri[(t + 2) % 3]);
                elems[v].push((i as u32) * 2 + 1);
            }
        }
        let mut inc = Incidence {
            ring_start: Vec::with_capacity(n_pts + 1),
            ring: Vec::new(),
            elem_start: Vec::with_capacity(n_pts + 1),
            elem: Vec::new(),
        };
        for v in 0..n_pts {
            inc.ring_start.push(inc.ring.len() as u32);
            ring_sets[v].sort_unstable();
            ring_sets[v].dedup();
            inc.ring.extend_from_slice(&ring_sets[v]);
            inc.elem_start.push(inc.elem.len() as u32);
            elems[v].sort_unstable();
            elems[v].dedup();
            inc.elem.extend_from_slice(&elems[v]);
        }
        inc.ring_start.push(inc.ring.len() as u32);
        inc.elem_start.push(inc.elem.len() as u32);
        inc
    }

    #[inline]
    fn ring_of(&self, v: usize) -> &[u32] {
        &self.ring[self.ring_start[v] as usize..self.ring_start[v + 1] as usize]
    }

    #[inline]
    fn elems_of(&self, v: usize) -> &[u32] {
        &self.elem[self.elem_start[v] as usize..self.elem_start[v + 1] as usize]
    }

    /// Number of elements around `v` — the valence the cleanup pass reads.
    pub fn valence(&self, v: usize) -> usize {
        self.elems_of(v).len()
    }
}

/// Worst element quality around `v`, with the node placed at `at`.
fn worst_around(patch: &Patch, inc: &Incidence, pts: &[Point2], v: usize, at: Point2) -> f64 {
    let get = |i: u32| if i as usize == v { at } else { pts[i as usize] };
    let mut worst = f64::INFINITY;
    for &e in inc.elems_of(v) {
        let q = if e % 2 == 0 {
            let c = patch.quads[(e / 2) as usize];
            quad_quality([get(c[0]), get(c[1]), get(c[2]), get(c[3])])
        } else {
            let c = patch.tris[(e / 2) as usize];
            tri_quality(get(c[0]), get(c[1]), get(c[2]))
        };
        worst = worst.min(q);
    }
    worst
}

/// Run `iters` smoothing sweeps over `pts`.
///
/// Only nodes listed in `active` are considered, which is what lets the paver
/// smooth the strip it has just laid without touching the rest of the mesh.
/// Pass `None` to sweep everything.
pub fn smooth(
    pts: &mut [Point2],
    patch: &Patch,
    inc: &Incidence,
    active: Option<&[u32]>,
    iters: usize,
) {
    let owned: Vec<u32>;
    let nodes: &[u32] = match active {
        Some(a) => a,
        None => {
            owned = (0..pts.len() as u32).collect();
            &owned
        }
    };
    for _ in 0..iters {
        let old = pts.to_vec();
        let moves: Vec<(u32, Point2)> = nodes
            .par_iter()
            .with_min_len(MIN_PARALLEL_LEN)
            .filter_map(|&vi| {
                let v = vi as usize;
                if !patch.movable[v] {
                    return None;
                }
                let ring = inc.ring_of(v);
                if ring.is_empty() {
                    return None;
                }
                let mut c = Vector2::zeros();
                for &nb in ring {
                    c += old[nb as usize].coords;
                }
                c /= ring.len() as f64;
                let cand = Point2::from(old[v].coords * (1.0 - RELAX) + c * RELAX);
                let before = worst_around(patch, inc, &old, v, old[v]);
                let after = worst_around(patch, inc, &old, v, cand);
                (after > 0.0 && after >= before).then_some((vi, cand))
            })
            .collect();
        if moves.is_empty() {
            break;
        }
        let mut moved = vec![false; pts.len()];
        for (v, p) in moves {
            pts[v as usize] = p;
            moved[v as usize] = true;
        }
        repair(pts, &old, patch, &mut moved);
    }
}

/// Put back every node the sweep left on an invalid element.
fn repair(pts: &mut [Point2], old: &[Point2], patch: &Patch, moved: &mut [bool]) {
    for _ in 0..REPAIR_ROUNDS {
        let mut reverted = false;
        let mut undo = |c: &[u32], pts: &mut [Point2], moved: &mut [bool]| {
            for &v in c {
                if moved[v as usize] {
                    pts[v as usize] = old[v as usize];
                    moved[v as usize] = false;
                    reverted = true;
                }
            }
        };
        for q in patch.quads {
            let c = [
                pts[q[0] as usize],
                pts[q[1] as usize],
                pts[q[2] as usize],
                pts[q[3] as usize],
            ];
            if !super::geom::quad_is_valid(c) {
                undo(q, pts, moved);
            }
        }
        for t in patch.tris {
            if super::geom::orient(pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize])
                <= 0.0
            {
                undo(t, pts, moved);
            }
        }
        if !reverted {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3×3 grid of quadrangles whose interior nodes have been shoved off
    /// their proper places.
    fn wobbly_grid() -> (Vec<Point2>, Vec<[u32; 4]>, Vec<bool>) {
        let n = 4;
        let mut pts = Vec::new();
        let mut movable = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let boundary = i == 0 || j == 0 || i == n - 1 || j == n - 1;
                let (mut x, mut y) = (i as f64, j as f64);
                if !boundary {
                    x += 0.35;
                    y -= 0.3;
                }
                pts.push(Point2::new(x, y));
                movable.push(!boundary);
            }
        }
        let mut quads = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                let a = (j * n + i) as u32;
                quads.push([a, a + 1, a + n as u32 + 1, a + n as u32]);
            }
        }
        (pts, quads, movable)
    }

    fn worst(pts: &[Point2], quads: &[[u32; 4]]) -> f64 {
        quads
            .iter()
            .map(|q| {
                quad_quality([
                    pts[q[0] as usize],
                    pts[q[1] as usize],
                    pts[q[2] as usize],
                    pts[q[3] as usize],
                ])
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn smoothing_improves_the_worst_element_and_keeps_the_boundary() {
        let (mut pts, quads, movable) = wobbly_grid();
        let before = worst(&pts, &quads);
        let fixed: Vec<Point2> = pts.clone();
        let patch = Patch {
            quads: &quads,
            tris: &[],
            movable: &movable,
        };
        let inc = Incidence::build(&patch, pts.len());
        smooth(&mut pts, &patch, &inc, None, 30);
        let after = worst(&pts, &quads);
        assert!(after > before, "{before} → {after}");
        for (i, m) in movable.iter().enumerate() {
            if !m {
                assert_eq!(pts[i], fixed[i], "boundary node {i} moved");
            }
        }
    }

    #[test]
    fn smoothing_never_makes_an_element_invalid() {
        let (mut pts, quads, movable) = wobbly_grid();
        let patch = Patch {
            quads: &quads,
            tris: &[],
            movable: &movable,
        };
        let inc = Incidence::build(&patch, pts.len());
        for _ in 0..10 {
            smooth(&mut pts, &patch, &inc, None, 3);
            assert!(worst(&pts, &quads) > 0.0);
        }
    }

    #[test]
    fn a_grid_interior_node_has_valence_four() {
        let (pts, quads, movable) = wobbly_grid();
        let patch = Patch {
            quads: &quads,
            tris: &[],
            movable: &movable,
        };
        let inc = Incidence::build(&patch, pts.len());
        assert_eq!(inc.valence(5), 4);
        assert_eq!(inc.valence(0), 1);
    }
}
