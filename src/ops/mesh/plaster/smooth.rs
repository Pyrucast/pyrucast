//! Relaxing the layer's nodes, under a guard that no cell may turn inside out.
//!
//! An advancing layer leaves kinks: a node whose step was cut short by the
//! room it had, or by a neighbour's cell going bad, sits out of line with the
//! nodes either side of it. Smoothing is what pulls those back — and only
//! those, since a node of the caller's envelope never moves.
//!
//! The guard is the same quantity the finite-element assembly will look at:
//! the scaled Jacobian at each corner, which is the determinant of the three
//! edges leaving it, normalised by their lengths. It is positive exactly when
//! the cell is right way out at that corner, and near `1` when the corner is
//! square. Requiring that the worst corner around a node does not get worse is
//! therefore not a proxy for quality — it is the thing itself.

use crate::atoms::Point3;

/// The three edges leaving each corner of a `HEX8`, in an order whose
/// determinant is positive on the reference element.
const HEX_CORNERS: [[usize; 4]; 8] = [
    [0, 1, 3, 4],
    [1, 2, 0, 5],
    [2, 3, 1, 6],
    [3, 0, 2, 7],
    [4, 7, 5, 0],
    [5, 4, 6, 1],
    [6, 5, 7, 2],
    [7, 6, 4, 3],
];

/// Likewise for a `PENTA6`.
const PRISM_CORNERS: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [1, 2, 0, 4],
    [2, 0, 1, 5],
    [3, 5, 4, 0],
    [4, 3, 5, 1],
    [5, 4, 3, 2],
];

/// Relaxation applied to the Laplacian step.
const RELAX: f64 = 0.5;

/// The cells being smoothed.
pub struct Patch<'a> {
    pub hexes: &'a [[u32; 8]],
    pub prisms: &'a [[u32; 6]],
    /// `false` for a node that must not move.
    pub movable: &'a [bool],
}

/// Worst scaled Jacobian over a cell's corners: `> 0` right way out, `1` for a
/// perfect cube or prism, `≤ 0` inside out.
fn cell_quality(pts: &[Point3], cell: &[u32], corners: &[[usize; 4]]) -> f64 {
    let mut worst = f64::INFINITY;
    for c in corners {
        let o = pts[cell[c[0]] as usize];
        let e = [
            pts[cell[c[1]] as usize] - o,
            pts[cell[c[2]] as usize] - o,
            pts[cell[c[3]] as usize] - o,
        ];
        let lengths: f64 = e.iter().map(|v| v.norm()).product();
        if lengths == 0.0 {
            return 0.0;
        }
        worst = worst.min(e[0].cross(&e[1]).dot(&e[2]) / lengths);
    }
    worst
}

/// Worst scaled Jacobian in the whole patch — the number the mesher is judged
/// on, and the one the guard below preserves.
pub fn worst_quality(pts: &[Point3], patch: &Patch) -> f64 {
    let h = patch
        .hexes
        .iter()
        .map(|c| cell_quality(pts, c, &HEX_CORNERS));
    let p = patch
        .prisms
        .iter()
        .map(|c| cell_quality(pts, c, &PRISM_CORNERS));
    h.chain(p).fold(f64::INFINITY, f64::min)
}

/// Per-node incidence and neighbour ring, built once.
pub struct Incidence {
    ring: Vec<Vec<u32>>,
    /// `2 * index` for a hexahedron, `2 * index + 1` for a prism.
    cells: Vec<Vec<u32>>,
}

impl Incidence {
    pub fn build(patch: &Patch, n_pts: usize) -> Incidence {
        let mut ring: Vec<Vec<u32>> = vec![Vec::new(); n_pts];
        let mut cells: Vec<Vec<u32>> = vec![Vec::new(); n_pts];
        let edge = |a: u32, b: u32, ring: &mut Vec<Vec<u32>>| {
            ring[a as usize].push(b);
            ring[b as usize].push(a);
        };
        for (i, c) in patch.hexes.iter().enumerate() {
            for k in &HEX_CORNERS {
                for t in 1..4 {
                    edge(c[k[0]], c[k[t]], &mut ring);
                }
            }
            for &v in c {
                cells[v as usize].push((i as u32) * 2);
            }
        }
        for (i, c) in patch.prisms.iter().enumerate() {
            for k in &PRISM_CORNERS {
                for t in 1..4 {
                    edge(c[k[0]], c[k[t]], &mut ring);
                }
            }
            for &v in c {
                cells[v as usize].push((i as u32) * 2 + 1);
            }
        }
        for r in ring.iter_mut() {
            r.sort_unstable();
            r.dedup();
        }
        for c in cells.iter_mut() {
            c.sort_unstable();
            c.dedup();
        }
        Incidence { ring, cells }
    }
}

/// Worst quality among the cells around `v`, with the node placed at `at`.
fn worst_around(patch: &Patch, inc: &Incidence, pts: &[Point3], v: usize, at: Point3) -> f64 {
    let mut moved: Vec<Point3> = Vec::new();
    let mut worst = f64::INFINITY;
    for &e in &inc.cells[v] {
        let (cell, corners): (&[u32], &[[usize; 4]]) = if e.is_multiple_of(2) {
            (&patch.hexes[(e / 2) as usize], &HEX_CORNERS)
        } else {
            (&patch.prisms[(e / 2) as usize], &PRISM_CORNERS)
        };
        moved.clear();
        moved.extend(
            cell.iter()
                .map(|&i| if i as usize == v { at } else { pts[i as usize] }),
        );
        let local: Vec<u32> = (0..cell.len() as u32).collect();
        worst = worst.min(cell_quality(&moved, &local, corners));
    }
    worst
}

/// Run `iters` smoothing sweeps over the movable nodes.
///
/// Gauss–Seidel, in node order: each move is judged against the mesh as it
/// stands, so a move that is accepted can never be undone by one accepted
/// after it. That costs the parallelism a Jacobi sweep would allow and buys
/// the guarantee outright — the worst cell can only improve.
pub fn smooth(pts: &mut [Point3], patch: &Patch, inc: &Incidence, iters: usize) -> usize {
    let mut moves = 0;
    for _ in 0..iters {
        let mut changed = 0;
        for v in 0..pts.len() {
            if !patch.movable[v] || inc.ring[v].is_empty() {
                continue;
            }
            let mut centre = Point3::origin().coords;
            for &nb in &inc.ring[v] {
                centre += pts[nb as usize].coords;
            }
            centre /= inc.ring[v].len() as f64;
            let cand = Point3::from(pts[v].coords * (1.0 - RELAX) + centre * RELAX);
            let before = worst_around(patch, inc, pts, v, pts[v]);
            let after = worst_around(patch, inc, pts, v, cand);
            if after > before {
                pts[v] = cand;
                changed += 1;
            }
        }
        moves += changed;
        if changed == 0 {
            break;
        }
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 × 2 × 2 block of unit hexahedra with the centre node shoved off.
    fn wobbly_block() -> (Vec<Point3>, Vec<[u32; 8]>, Vec<bool>) {
        let w = 3;
        let mut pts = Vec::new();
        let mut movable = Vec::new();
        for k in 0..w {
            for j in 0..w {
                for i in 0..w {
                    let interior = i == 1 && j == 1 && k == 1;
                    let mut p = Point3::new(i as f64, j as f64, k as f64);
                    if interior {
                        p += crate::atoms::Vector3::new(0.45, -0.4, 0.35);
                    }
                    pts.push(p);
                    movable.push(interior);
                }
            }
        }
        let at = |i: usize, j: usize, k: usize| ((k * w + j) * w + i) as u32;
        let mut hexes = Vec::new();
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    hexes.push([
                        at(i, j, k),
                        at(i + 1, j, k),
                        at(i + 1, j + 1, k),
                        at(i, j + 1, k),
                        at(i, j, k + 1),
                        at(i + 1, j, k + 1),
                        at(i + 1, j + 1, k + 1),
                        at(i, j + 1, k + 1),
                    ]);
                }
            }
        }
        (pts, hexes, movable)
    }

    #[test]
    fn a_perfect_block_scores_one_everywhere() {
        let (mut pts, hexes, movable) = wobbly_block();
        pts[13] = Point3::new(1.0, 1.0, 1.0); // put the centre back
        let patch = Patch {
            hexes: &hexes,
            prisms: &[],
            movable: &movable,
        };
        assert!((worst_quality(&pts, &patch) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn smoothing_recovers_the_shoved_node_and_improves_the_worst_cell() {
        let (mut pts, hexes, movable) = wobbly_block();
        let patch = Patch {
            hexes: &hexes,
            prisms: &[],
            movable: &movable,
        };
        let inc = Incidence::build(&patch, pts.len());
        let before = worst_quality(&pts, &patch);
        smooth(&mut pts, &patch, &inc, 40);
        let after = worst_quality(&pts, &patch);
        assert!(after > before, "{before} → {after}");
        // The centre node's proper place is (1, 1, 1).
        assert!(
            (pts[13] - Point3::new(1.0, 1.0, 1.0)).norm() < 1e-6,
            "centre at {:?}",
            pts[13]
        );
    }

    #[test]
    fn smoothing_never_moves_a_pinned_node_and_never_makes_things_worse() {
        let (mut pts, hexes, movable) = wobbly_block();
        let fixed = pts.clone();
        let patch = Patch {
            hexes: &hexes,
            prisms: &[],
            movable: &movable,
        };
        let inc = Incidence::build(&patch, pts.len());
        let before = worst_quality(&pts, &patch);
        smooth(&mut pts, &patch, &inc, 10);
        assert!(worst_quality(&pts, &patch) >= before);
        for (i, m) in movable.iter().enumerate() {
            if !m {
                assert_eq!(pts[i], fixed[i], "pinned node {i} moved");
            }
        }
    }
}
