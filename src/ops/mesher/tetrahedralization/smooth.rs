//! Getting rid of slivers.
//!
//! Refinement leaves one kind of bad cell behind, and it is the worst kind: a
//! **sliver**, four corners close to one plane and spread evenly around a
//! circle. Every measure refinement watches says it is fine — its edges are
//! all of a decent length, its circumsphere is small — and yet it has almost
//! no volume:
//!
//! ```text
//!        b                  the four corners sit near one circle, so no
//!       / \                 edge is short and the circumsphere is tight;
//!    a ─────── c            the cell is nonetheless flat, and its element
//!       \ /                 matrix nearly singular
//!        d
//! ```
//!
//! No node can be inserted to remove it — that is a theorem, not a gap in
//! the implementation — so the mesh has to be improved rather than
//! subdivided, by the two moves that change a mesh without changing what it
//! fills:
//!
//! - **reconnection**: the same nodes, joined differently. A sliver is often
//!   one flip away from two decent cells.
//! - **smoothing**: the same connections, with a node moved. Only interior
//!   nodes move; the envelope's own nodes are the caller's and are never
//!   touched, which is what keeps the surface exactly as it came in.
//!
//! Both are judged on the **smallest dihedral angle** of the cells they
//! touch, and applied only when it improves. That makes the pass
//! monotone — it can only make the worst angle in a neighbourhood better —
//! so it can be run until it stops paying without risk of undoing itself.

use std::collections::HashSet;

use crate::error::Result;
use crate::interrupt::Cancel;

use super::delaunay::TetMesh;
use super::flips::flip23;

/// Below this angle (degrees) a cell is worth working on.
const POOR: f64 = 25.0;

/// Sweeps of reconnection and smoothing before the pass gives up.
const ROUNDS: usize = 6;

/// How far a node is moved toward its target, in decreasing steps.
///
/// The full step is right when the neighbourhood is roomy and wrong when it
/// is tight, so the shorter ones are there to be fallen back on.
const STEPS: [f64; 4] = [1.0, 0.6, 0.3, 0.1];

/// Improve the badly shaped cells of `mesh`, in place.
///
/// `movable` marks the nodes that may be moved — the interior ones. `walls`
/// are the envelope's facets, which no reconnection may disturb. `inside`
/// says which slots hold material.
///
/// Returns the smallest dihedral angle left, in degrees.
pub fn smooth(
    mesh: &mut TetMesh,
    inside: &[bool],
    movable: &[bool],
    walls: &HashSet<[u32; 3]>,
    cancel: &dyn Cancel,
) -> Result<f64> {
    for _ in 0..ROUNDS {
        cancel.check()?;
        let reconnected = reconnect(mesh, inside, walls)?;
        let moved = relax(mesh, inside, movable);
        if !reconnected && !moved {
            break;
        }
    }
    Ok(worst_angle(mesh, inside))
}

/// Flip the faces of poor cells, keeping what improves them.
///
/// The outcome of a 2-3 flip is known before it is made — three cells, on
/// vertices already in hand — so it is judged first and applied only if it
/// pays. Trying it and undoing it would mean copying the mesh once per
/// candidate face, which on a large mesh costs more than the whole pass.
fn reconnect(mesh: &mut TetMesh, inside: &[bool], walls: &HashSet<[u32; 3]>) -> Result<bool> {
    let mut changed = false;
    for t in 0..mesh.slot_count() {
        let Some(v) = mesh.tet(t) else { continue };
        if !inside.get(t).copied().unwrap_or(false) || min_dihedral(mesh, &v) >= POOR {
            continue;
        }
        for i in 0..4 {
            let f = mesh.face(t, i);
            let mut key = f;
            key.sort_unstable();
            if walls.contains(&key) {
                continue; // the envelope is not ours to re-cut
            }
            let Some(n) = mesh.neighbour(t, i) else {
                continue;
            };
            let Some(w) = mesh.tet(n) else { continue };
            let Some(e) = mesh.apex_beyond(n, &f) else {
                continue;
            };
            let d = v[i];

            let before = min_dihedral(mesh, &v).min(min_dihedral(mesh, &w));
            let candidates = [[f[0], f[1], d, e], [f[1], f[2], d, e], [f[2], f[0], d, e]];
            if candidates.iter().any(|c| mesh.orientation(c) <= 0.0) {
                continue; // the pair is not convex across this face
            }
            let after = candidates
                .iter()
                .map(|c| min_dihedral(mesh, c))
                .fold(f64::INFINITY, f64::min);
            if after <= before {
                continue;
            }
            if flip23(mesh, t, i)?.is_some() {
                changed = true;
                break;
            }
        }
    }
    Ok(changed)
}

/// Move each movable node toward wherever its own cells are least pinched.
fn relax(mesh: &mut TetMesh, inside: &[bool], movable: &[bool]) -> bool {
    let mut changed = false;
    for v in 0..mesh.points().len() as u32 {
        if !movable.get(v as usize).copied().unwrap_or(false) {
            continue;
        }
        // Every cell holding the node, not only the ones made of material:
        // moving it moves them all, and one turned inside out is a broken
        // mesh whether or not it is kept in the end.
        let star: Vec<[u32; 4]> = mesh
            .tets_around_vertex(v)
            .into_iter()
            .filter_map(|t| mesh.tet(t as usize))
            .collect();
        if star.is_empty() {
            continue;
        }
        let cells: Vec<[u32; 4]> = mesh
            .tets_around_vertex(v)
            .into_iter()
            .filter(|&t| inside.get(t as usize).copied().unwrap_or(false))
            .filter_map(|t| mesh.tet(t as usize))
            .collect();
        if cells.is_empty() {
            continue;
        }

        let before = cells
            .iter()
            .map(|c| min_dihedral(mesh, c))
            .fold(f64::INFINITY, f64::min);
        if before >= POOR {
            continue;
        }

        // The average of the neighbours pulls a pinched node toward the
        // roomiest spot its own cells leave it.
        let here = mesh.points()[v as usize];
        let mut sum = [0.0f64; 3];
        let mut count = 0.0;
        for c in &cells {
            for &x in c.iter().filter(|&&x| x != v) {
                let p = mesh.points()[x as usize];
                for k in 0..3 {
                    sum[k] += p[k];
                }
                count += 1.0;
            }
        }
        if count == 0.0 {
            continue;
        }
        let target = [sum[0] / count, sum[1] / count, sum[2] / count];

        let mut best = (before, here);
        for step in STEPS {
            let trial = [
                here[0] + step * (target[0] - here[0]),
                here[1] + step * (target[1] - here[1]),
                here[2] + step * (target[2] - here[2]),
            ];
            mesh.set_point(v, trial);
            // A move that turns any cell inside out is no move at all.
            let ok = star.iter().all(|c| mesh.orientation(c) > 0.0);
            let after = if ok {
                cells
                    .iter()
                    .map(|c| min_dihedral(mesh, c))
                    .fold(f64::INFINITY, f64::min)
            } else {
                f64::NEG_INFINITY
            };
            if after > best.0 {
                best = (after, trial);
            }
        }
        mesh.set_point(v, best.1);
        if best.1 != here {
            changed = true;
        }
    }
    changed
}

/// The smallest dihedral angle of the mesh's material, in degrees.
pub fn worst_angle(mesh: &TetMesh, inside: &[bool]) -> f64 {
    mesh.iter()
        .filter(|(t, _)| inside.get(*t).copied().unwrap_or(false))
        .map(|(_, v)| min_dihedral(mesh, &v))
        .fold(f64::INFINITY, f64::min)
}

/// The smallest angle between two faces of a cell, in degrees.
///
/// Along an edge, the dihedral is read by flattening the two opposite
/// corners into the plane across it: project each onto the plane
/// perpendicular to the edge, and take the angle between what is left.
fn min_dihedral(mesh: &TetMesh, v: &[u32; 4]) -> f64 {
    const EDGES: [(usize, usize, usize, usize); 6] = [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ];
    let p = mesh.points();
    let mut worst = f64::INFINITY;
    for (a, b, c, d) in EDGES {
        let (pa, pb) = (p[v[a] as usize], p[v[b] as usize]);
        let e = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let len = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        if len == 0.0 {
            return 0.0;
        }
        let unit = [e[0] / len, e[1] / len, e[2] / len];
        let flatten = |x: [f64; 3]| {
            let w = [x[0] - pa[0], x[1] - pa[1], x[2] - pa[2]];
            let along = w[0] * unit[0] + w[1] * unit[1] + w[2] * unit[2];
            [
                w[0] - along * unit[0],
                w[1] - along * unit[1],
                w[2] - along * unit[2],
            ]
        };
        let (u, w) = (flatten(p[v[c] as usize]), flatten(p[v[d] as usize]));
        let (nu, nw) = (
            (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt(),
            (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt(),
        );
        if nu == 0.0 || nw == 0.0 {
            return 0.0;
        }
        let cos = ((u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (nu * nw)).clamp(-1.0, 1.0);
        worst = worst.min(cos.acos().to_degrees());
    }
    worst
}
