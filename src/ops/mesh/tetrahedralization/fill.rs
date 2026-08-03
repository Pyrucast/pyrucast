//! Filling a small closed region with tetrahedra taken from its own
//! vertices.
//!
//! Boundary recovery keeps running into the same question: *this pocket of
//! the mesh is in the way — can it be rebuilt without the edge that blocks
//! me?* For a pocket of a handful of cells the question has a complete
//! answer, found by search rather than by guessing at a pattern.
//!
//! The search grows a filling one cell at a time from the region's own
//! surface. At every step there is a set of **open faces** — faces that have
//! material on one side and nothing yet on the other — and the next cell is
//! whatever hangs off one of them. A cell either closes an open face or
//! opens a new one, so the region is complete exactly when nothing is open;
//! that the cells then tile it, rather than overlap, is settled by their
//! volumes adding up to the region's own.
//!
//! Everything is decided by [`orient3d`], so a filling is never accepted on
//! a rounding artefact, and the flat cells that a pattern-based filling
//! would produce on coplanar input are simply never generated.
//!
//! This is complete where a pattern is not. Removing an edge, for instance,
//! is often done by re-cutting the ring of vertices around it and hanging
//! the two endpoints off each triangle — but that misses the fillings
//! containing a cell made of four ring vertices, which on a box is exactly
//! the one that works.

use std::collections::{HashMap, HashSet};

use crate::interrupt::NoCancel;

use super::delaunay::TetMesh;
use super::predicates::orient3d;

/// Boundary faces up to which the exhaustive filler is worth trying.
///
/// It is complete, so it is the only one that can prove a small pocket has
/// no filling — but it is exponential, so past this it costs more than it
/// can return.
pub const EXHAUSTIVE_LIMIT: usize = 12;

/// Cells examined before a search gives up.
///
/// The search is exponential in the pocket's size, so the budget is what
/// makes it a *bounded* tool rather than a gamble. A pocket small enough to
/// be worth rebuilding is filled in a few dozen steps; recovery tries many
/// pockets, so each one has to stay cheap.
pub const DEFAULT_BUDGET: usize = 3_000;

/// What the filling must and must not contain.
///
/// The two ways boundary recovery steers a rebuild: *make this edge go
/// away*, and *make this facet appear*.
#[derive(Debug, Default, Clone, Copy)]
pub struct Constraints<'a> {
    /// No cell may hold both ends of any of these.
    pub without_edges: &'a [(u32, u32)],
    /// Each of these must end up a face of the filling. No cell may be cut
    /// by one either, which is what makes the requirement reachable rather
    /// than a lottery: treating the triangle as a wall leaves the search no
    /// way to fill across it.
    pub with_faces: &'a [[u32; 3]],
    /// Each of these must end up an edge of the filling.
    pub with_edges: &'a [(u32, u32)],
}

/// Fill the region enclosed by `boundary` by triangulating its own vertices
/// and keeping what lands inside.
///
/// The exhaustive search above is complete but exponential, so it runs out
/// of road at half a dozen cells — and the pockets that matter are larger.
/// This one does not search at all. It **computes** the one canonical
/// candidate, the Delaunay triangulation of the region's vertices, and asks
/// a single question of it: does it contain every face of the region's
/// surface? If it does, the cells that fall inside tile the region exactly
/// and the job is done in the time of one triangulation. If it does not,
/// there is nothing to backtrack over — the caller widens the region and
/// asks again, which is a different question rather than the same one
/// retried.
///
/// Nothing here can be *asked* for, since a Delaunay triangulation is what
/// it is: `want` is checked after the fact, not steered towards. A caller
/// that needs a particular face or edge puts it in `boundary`, where it is
/// part of the question instead of a hope.
///
/// The error reports **which faces of the surface the triangulation does not
/// contain**. That is not a diagnostic but the next instruction: absorb the
/// cells beyond them and ask again, and those faces stop being on the surface
/// at all. Widening blindly instead almost never lands on a surface the
/// triangulation happens to respect. It is empty when the refusal was for
/// some other reason — one that widening will not mend.
pub fn delaunay_fill(
    points: &[[f64; 3]],
    boundary: &[[u32; 3]],
    want: Constraints<'_>,
) -> Result<Vec<[u32; 4]>, Vec<[u32; 3]>> {
    // The region's own vertices, renumbered from zero: the triangulation
    // knows nothing of the mesh this pocket came out of.
    let mut vs: Vec<u32> = boundary.iter().flatten().copied().collect();
    vs.sort_unstable();
    vs.dedup();
    if vs.len() < 4 {
        return Err(Vec::new());
    }
    let local: Vec<[f64; 3]> = vs.iter().map(|&i| points[i as usize]).collect();
    let down = |g: u32| vs.binary_search(&g).ok().map(|i| i as u32);
    let up = |l: u32| vs[l as usize];

    let mut walls: Vec<[u32; 3]> = Vec::with_capacity(boundary.len());
    for f in boundary {
        let Some(w) = (|| Some([down(f[0])?, down(f[1])?, down(f[2])?]))() else {
            return Err(Vec::new());
        };
        walls.push(w);
    }
    let Ok(mesh) = TetMesh::delaunay(&local, &NoCancel) else {
        return Err(Vec::new());
    };

    // The one question worth asking. A surface the triangulation does not
    // contain is a surface it cannot be cut along.
    let missing: Vec<[u32; 3]> = walls
        .iter()
        .filter(|f| !mesh.has_face(f))
        .map(|f| [up(f[0]), up(f[1]), up(f[2])])
        .collect();
    if !missing.is_empty() {
        return Err(missing);
    }
    let barrier: HashSet<[u32; 3]> = walls.iter().map(key).collect();

    // Inside is the side the surface faces away from, so each wall names the
    // cell beyond it that belongs to the region.
    let mut inside = vec![false; mesh.slot_count()];
    let mut stack: Vec<usize> = Vec::new();
    for f in &walls {
        let Some(owners) = mesh.face_owners(f) else {
            return Err(Vec::new());
        };
        for (t, i) in owners {
            let Some(apex) = mesh.tet(t as usize).map(|c| c[i]) else {
                return Err(Vec::new());
            };
            let p = mesh.points();
            if orient3d(
                &p[f[0] as usize],
                &p[f[1] as usize],
                &p[f[2] as usize],
                &p[apex as usize],
            ) < 0.0
                && !inside[t as usize]
            {
                inside[t as usize] = true;
                stack.push(t as usize);
            }
        }
    }
    if stack.is_empty() {
        return Err(Vec::new());
    }
    while let Some(t) = stack.pop() {
        for i in 0..4 {
            if barrier.contains(&key(&mesh.face(t, i))) {
                continue;
            }
            if let Some(n) = mesh.neighbour(t, i) {
                if !inside[n] {
                    inside[n] = true;
                    stack.push(n);
                }
            }
        }
    }

    let cells: Vec<[u32; 4]> = mesh
        .iter()
        .filter(|(t, _)| inside[*t])
        .map(|(_, v)| [up(v[0]), up(v[1]), up(v[2]), up(v[3])])
        .collect();
    if cells.is_empty() {
        return Err(Vec::new());
    }

    // Every wall being present does not make the flooded part *the region*.
    // A pocket carved out of an existing mesh can have a surface that folds
    // back on itself, and the flood then settles on a different solid with
    // the same skin — cells that tile something the caller never asked to
    // fill. Volume is what tells the two apart, and it is not expensive.
    let enclosed = enclosed_volume(points, boundary);
    let filled: f64 = cells
        .iter()
        .map(|c| {
            let p = |i: u32| points[i as usize];
            orient3d(&p(c[0]), &p(c[1]), &p(c[2]), &p(c[3])) / 6.0
        })
        .sum();
    if enclosed <= 0.0 || (filled - enclosed).abs() > 1e-9 * enclosed {
        return Err(Vec::new());
    }

    // Whatever the caller needed has to have come out of it anyway.
    let ok = want.with_faces.iter().all(|f| {
        cells
            .iter()
            .any(|c| faces_of(c).iter().any(|g| key(g) == key(f)))
    }) && want
        .with_edges
        .iter()
        .all(|&(u, v)| cells.iter().any(|c| c.contains(&u) && c.contains(&v)))
        && !want
            .without_edges
            .iter()
            .any(|&(u, v)| cells.iter().any(|c| c.contains(&u) && c.contains(&v)));
    // A constraint the triangulation did not meet is not something growing
    // will mend, so nothing is reported to grow across.
    if ok {
        Ok(cells)
    } else {
        Err(Vec::new())
    }
}

/// The volume a closed, outward-wound surface encloses.
///
/// By the divergence theorem the sum over faces of the signed volume of
/// `(face, o)` is the enclosed volume for **any** reference point `o`, inside
/// the surface or not — the parts outside cancel. `o` is taken as the origin.
fn enclosed_volume(points: &[[f64; 3]], boundary: &[[u32; 3]]) -> f64 {
    const O: [f64; 3] = [0.0, 0.0, 0.0];
    boundary
        .iter()
        .map(|f| {
            let p = |i: u32| points[i as usize];
            // An outward face and a point on the material side make a
            // negatively oriented cell, so the sign is turned round here.
            -orient3d(&p(f[0]), &p(f[1]), &p(f[2]), &O) / 6.0
        })
        .sum()
}

/// Fill the region enclosed by `boundary` with tetrahedra drawn from its own
/// vertices.
///
/// `boundary` faces are wound **outwards**, away from the region.
///
/// Returns `None` when no filling exists, or when `budget` runs out first.
pub fn fill(
    points: &[[f64; 3]],
    boundary: &[[u32; 3]],
    want: Constraints<'_>,
    budget: usize,
) -> Option<Vec<[u32; 4]>> {
    if boundary.len() < 4 {
        return None;
    }
    let mut vertices: Vec<u32> = boundary.iter().flatten().copied().collect();
    vertices.sort_unstable();
    vertices.dedup();
    if vertices.len() < 4 {
        return None;
    }

    // Six times the volume the surface encloses, by the divergence theorem.
    // The filling has to reach exactly this, which is what rules out cells
    // that stray outside the region.
    let origin = points[vertices[0] as usize];
    let target: f64 = boundary
        .iter()
        .map(|f| {
            let at = |i: u32| {
                let p = points[i as usize];
                [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]
            };
            let (a, b, c) = (at(f[0]), at(f[1]), at(f[2]));
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum();
    if target <= 0.0 {
        return None; // not a closed region wound outwards
    }
    let slack = 1e-9 * target;

    // Open faces, keyed by their vertices, holding the winding the next cell
    // must present as its own outward face.
    let mut open: HashMap<[u32; 3], [u32; 3]> = HashMap::with_capacity(boundary.len() * 2);
    for f in boundary {
        if open.insert(key(f), *f).is_some() {
            return None; // a face used twice: not a simple surface
        }
    }

    let mut placed: Vec<[u32; 4]> = Vec::new();
    let mut budget = budget;
    grow(
        points,
        &vertices,
        want,
        target,
        slack,
        &mut open,
        &mut placed,
        0.0,
        &mut budget,
    )
    .then_some(placed)
}

#[allow(clippy::too_many_arguments)]
fn grow(
    points: &[[f64; 3]],
    vertices: &[u32],
    want: Constraints<'_>,
    target: f64,
    slack: f64,
    open: &mut HashMap<[u32; 3], [u32; 3]>,
    placed: &mut Vec<[u32; 4]>,
    volume: f64,
    budget: &mut usize,
) -> bool {
    if open.is_empty() {
        if (volume - target).abs() > slack {
            return false;
        }
        // Every required facet has to have turned up somewhere.
        return want.with_faces.iter().all(|f| {
            placed
                .iter()
                .any(|c| faces_of(c).iter().any(|g| key(g) == key(f)))
        });
    }
    if *budget == 0 {
        return false;
    }

    // Always work on the same face for a given state, so the search — and
    // therefore the mesh — does not depend on hash iteration order.
    let face = *open
        .iter()
        .min_by_key(|(k, _)| **k)
        .map(|(_, f)| f)
        .expect("open is not empty");

    for &apex in vertices {
        // Checked before anything is touched, so unwinding leaves the open
        // set exactly as this frame found it.
        if *budget == 0 {
            return false;
        }
        if face.contains(&apex) {
            continue;
        }
        // The cell sits below the outward face, so two of its vertices swap.
        let cell = [face[0], face[2], face[1], apex];
        if want
            .without_edges
            .iter()
            .any(|&(u, v)| cell.contains(&u) && cell.contains(&v))
        {
            continue;
        }
        if want.with_faces.iter().any(|f| cut_by(points, &cell, f)) {
            continue;
        }
        // A cell straddling a required edge would bury it. Refusing those
        // keeps a corridor open along the segment, which is what turns the
        // requirement from a lottery into a constraint the search can use.
        if want
            .with_edges
            .iter()
            .any(|&(u, v)| blocks_edge(points, &cell, u, v))
        {
            continue;
        }
        let v6 = orient3d(
            &points[cell[0] as usize],
            &points[cell[1] as usize],
            &points[cell[2] as usize],
            &points[cell[3] as usize],
        );
        if v6 <= 0.0 || volume + v6 > target + slack {
            continue;
        }
        *budget -= 1;

        // A face of the new cell either meets one that was waiting — and the
        // two windings must be opposite — or becomes one itself.
        let faces = faces_of(&cell);
        let mut closed: Vec<([u32; 3], [u32; 3])> = Vec::with_capacity(4);
        let mut opened: Vec<[u32; 3]> = Vec::with_capacity(4);
        let mut ok = true;
        for f in &faces {
            let k = key(f);
            match open.remove(&k) {
                Some(expected) => {
                    if same_winding(&expected, f) {
                        closed.push((k, expected));
                    } else {
                        open.insert(k, expected);
                        ok = false;
                        break;
                    }
                }
                None => {
                    // The neighbour on the far side will present the reverse.
                    open.insert(k, [f[0], f[2], f[1]]);
                    opened.push(k);
                }
            }
        }

        if ok {
            placed.push(cell);
            if grow(
                points,
                vertices,
                want,
                target,
                slack,
                open,
                placed,
                volume + v6,
                budget,
            ) {
                return true;
            }
            placed.pop();
        }
        for k in opened {
            open.remove(&k);
        }
        for (k, expected) in closed {
            open.insert(k, expected);
        }
    }
    false
}

/// The four outward faces of a cell.
fn faces_of(c: &[u32; 4]) -> [[u32; 3]; 4] {
    [
        [c[1], c[2], c[3]],
        [c[0], c[3], c[2]],
        [c[0], c[1], c[3]],
        [c[0], c[2], c[1]],
    ]
}

/// Whether an edge of `cell` runs through the inside of triangle `f`.
///
/// A cell that cuts a required facet would bury it, so the search never
/// places one.
fn cut_by(points: &[[f64; 3]], cell: &[u32; 4], f: &[u32; 3]) -> bool {
    let side = |x: u32| {
        orient3d(
            &points[f[0] as usize],
            &points[f[1] as usize],
            &points[f[2] as usize],
            &points[x as usize],
        )
    };
    for a in 0..4 {
        for b in a + 1..4 {
            let (p, q) = (cell[a], cell[b]);
            if f.contains(&p) || f.contains(&q) {
                continue;
            }
            let (sp, sq) = (side(p), side(q));
            if !((sp > 0.0 && sq < 0.0) || (sp < 0.0 && sq > 0.0)) {
                continue;
            }
            // The crossing point is inside the triangle when the line passes
            // on one same side of all three edges.
            let s = [
                orient3d(
                    &points[p as usize],
                    &points[q as usize],
                    &points[f[0] as usize],
                    &points[f[1] as usize],
                ),
                orient3d(
                    &points[p as usize],
                    &points[q as usize],
                    &points[f[1] as usize],
                    &points[f[2] as usize],
                ),
                orient3d(
                    &points[p as usize],
                    &points[q as usize],
                    &points[f[2] as usize],
                    &points[f[0] as usize],
                ),
            ];
            if s.iter().all(|&x| x > 0.0) || s.iter().all(|&x| x < 0.0) {
                return true;
            }
        }
    }
    false
}

/// Whether `cell` stands in the way of the segment `(u, v)`.
///
/// It does when one of its faces is pierced by the segment, or when one of
/// its edges crosses the segment flat in a shared plane — the two ways a
/// cell can occupy the space the edge needs.
fn blocks_edge(points: &[[f64; 3]], cell: &[u32; 4], u: u32, v: u32) -> bool {
    if cell.contains(&u) && cell.contains(&v) {
        return false; // this is the cell that carries the edge
    }
    let at = |x: u32| &points[x as usize];
    for f in faces_of(cell) {
        if f.contains(&u) || f.contains(&v) {
            continue;
        }
        let side = |x: u32| orient3d(at(f[0]), at(f[1]), at(f[2]), at(x));
        let (su, sv) = (side(u), side(v));
        if (su > 0.0 && sv < 0.0) || (su < 0.0 && sv > 0.0) {
            let s = [
                orient3d(at(u), at(v), at(f[0]), at(f[1])),
                orient3d(at(u), at(v), at(f[1]), at(f[2])),
                orient3d(at(u), at(v), at(f[2]), at(f[0])),
            ];
            if s.iter().all(|&x| x > 0.0) || s.iter().all(|&x| x < 0.0) {
                return true;
            }
        }
    }
    for a in 0..4 {
        for b in a + 1..4 {
            let (p, q) = (cell[a], cell[b]);
            if p == u || q == u || p == v || q == v {
                continue;
            }
            if orient3d(at(u), at(v), at(p), at(q)) != 0.0 {
                continue; // skew: they cannot meet
            }
            // Coplanar: each segment must separate the other's ends, tested
            // against a point lifted off their common plane.
            let e1 = [
                at(v)[0] - at(u)[0],
                at(v)[1] - at(u)[1],
                at(v)[2] - at(u)[2],
            ];
            let e2 = [
                at(p)[0] - at(u)[0],
                at(p)[1] - at(u)[1],
                at(p)[2] - at(u)[2],
            ];
            let lift = [
                at(u)[0] + e1[1] * e2[2] - e1[2] * e2[1],
                at(u)[1] + e1[2] * e2[0] - e1[0] * e2[2],
                at(u)[2] + e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let straddles = |a: u32, b: u32, c: u32, d: u32| {
                let sc = orient3d(at(a), at(b), &lift, at(c));
                let sd = orient3d(at(a), at(b), &lift, at(d));
                (sc > 0.0 && sd < 0.0) || (sc < 0.0 && sd > 0.0)
            };
            if straddles(u, v, p, q) && straddles(p, q, u, v) {
                return true;
            }
        }
    }
    false
}

fn key(f: &[u32; 3]) -> [u32; 3] {
    let mut k = *f;
    k.sort_unstable();
    k
}

/// Whether two triples name the same triangle traversed the same way.
fn same_winding(a: &[u32; 3], b: &[u32; 3]) -> bool {
    (0..3).any(|r| a[0] == b[r] && a[1] == b[(r + 1) % 3] && a[2] == b[(r + 2) % 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight corners of the unit cube, `000, 100, 110, 010, 001, …`.
    fn cube() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    }

    /// A box surface split along the diagonals of `{0, 2, 5, 7}`.
    const ALTERNATING: [[u32; 3]; 12] = [
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 7],
        [5, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 5],
        [2, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 7],
        [0, 4, 7],
    ];

    /// The same corners with every face split the other way.
    const UNFILLABLE: [[u32; 3]; 12] = [
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

    fn volume_of(points: &[[f64; 3]], cells: &[[u32; 4]]) -> f64 {
        cells
            .iter()
            .map(|c| {
                orient3d(
                    &points[c[0] as usize],
                    &points[c[1] as usize],
                    &points[c[2] as usize],
                    &points[c[3] as usize],
                )
            })
            .sum::<f64>()
            / 6.0
    }

    /// Every face is used twice with opposite windings, except the ones the
    /// region was given.
    fn check_tiles(cells: &[[u32; 4]], boundary: &[[u32; 3]]) {
        let mut seen: HashMap<[u32; 3], i32> = HashMap::new();
        for c in cells {
            for f in [
                [c[1], c[2], c[3]],
                [c[0], c[3], c[2]],
                [c[0], c[1], c[3]],
                [c[0], c[2], c[1]],
            ] {
                *seen.entry(key(&f)).or_default() += 1;
            }
        }
        for f in boundary {
            assert_eq!(seen.remove(&key(f)), Some(1), "boundary face {f:?}");
        }
        for (f, n) in seen {
            assert_eq!(n, 2, "interior face {f:?} used {n} time(s)");
        }
    }

    #[test]
    fn fills_a_box_whose_faces_allow_it() {
        let p = cube();
        let cells = fill(&p, &ALTERNATING, Constraints::default(), DEFAULT_BUDGET)
            .expect("a filling exists");
        assert!((volume_of(&p, &cells) - 1.0).abs() < 1e-14);
        check_tiles(&cells, &ALTERNATING);
    }

    #[test]
    fn refuses_a_box_whose_faces_forbid_it() {
        // No tetrahedralization of the box exists on these eight corners
        // with these diagonals — the search must exhaust and say so.
        let p = cube();
        assert_eq!(
            fill(&p, &UNFILLABLE, Constraints::default(), DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn honours_a_forbidden_edge() {
        let p = cube();
        // 0–2 is a boundary diagonal, so no filling can avoid it.
        assert_eq!(
            fill(
                &p,
                &ALTERNATING,
                Constraints {
                    without_edges: &[(0, 2)],
                    ..Default::default()
                },
                DEFAULT_BUDGET
            ),
            None
        );
        // 0–6 is the body diagonal: fillings exist both with and without it.
        let without = fill(
            &p,
            &ALTERNATING,
            Constraints {
                without_edges: &[(0, 6)],
                ..Default::default()
            },
            DEFAULT_BUDGET,
        )
        .expect("a filling exists");
        assert!(without.iter().all(|c| !(c.contains(&0) && c.contains(&6))));
        assert!((volume_of(&p, &without) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn fills_a_tetrahedron_with_itself() {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let boundary = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let cells =
            fill(&p, &boundary, Constraints::default(), DEFAULT_BUDGET).expect("a filling exists");
        assert_eq!(cells.len(), 1);
        assert!((volume_of(&p, &cells) - 1.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn refuses_a_surface_wound_inwards() {
        let p = cube();
        let flipped: Vec<[u32; 3]> = ALTERNATING.iter().map(|f| [f[0], f[2], f[1]]).collect();
        assert_eq!(
            fill(&p, &flipped, Constraints::default(), DEFAULT_BUDGET),
            None
        );
    }

    #[test]
    fn the_filling_does_not_depend_on_the_run() {
        let p = cube();
        let reference = fill(&p, &ALTERNATING, Constraints::default(), DEFAULT_BUDGET).unwrap();
        for _ in 0..5 {
            assert_eq!(
                fill(&p, &ALTERNATING, Constraints::default(), DEFAULT_BUDGET).unwrap(),
                reference
            );
        }
    }

    // ─── The Delaunay filler ────────────────────────────────────────────

    /// Five points whose hull is a bipyramid, and that hull's six faces.
    ///
    /// A convex region's surface is the convex hull of its own vertices, and
    /// the hull is always part of the Delaunay triangulation — so this is
    /// the case the filler is bound to get right.
    fn bipyramid() -> (Vec<[f64; 3]>, [[u32; 3]; 6]) {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, -1.0],
            [0.2, 0.2, 1.0],
        ];
        let faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 0, 4],
            [1, 0, 3],
            [2, 1, 3],
            [0, 2, 3],
        ];
        (p, faces)
    }

    #[test]
    fn delaunay_fills_a_convex_region() {
        let (p, faces) = bipyramid();
        let cells = delaunay_fill(&p, &faces, Constraints::default())
            .expect("a convex region's own hull is always in its triangulation");
        check_tiles(&cells, &faces);
        // The two pyramids, however the middle is cut.
        assert!((volume_of(&p, &cells) - 1.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn delaunay_declines_the_diagonals_a_box_does_not_choose() {
        // The eight corners of a box are cospherical, and the triangulation
        // picks its own diagonals. Asked for the other ones, the filler says
        // no rather than returning a surface that is not the one requested —
        // which is the signal the caller needs in order to widen the region.
        let p = cube();
        let missing = delaunay_fill(&p, &ALTERNATING, Constraints::default())
            .expect_err("the triangulation does not hold these diagonals");
        assert!(!missing.is_empty(), "it must say which faces are missing");
    }

    #[test]
    fn delaunay_declines_a_surface_wound_inwards() {
        let p = cube();
        let flipped: Vec<[u32; 3]> = ALTERNATING.iter().map(|f| [f[0], f[2], f[1]]).collect();
        assert!(delaunay_fill(&p, &flipped, Constraints::default()).is_err());
    }

    #[test]
    fn delaunay_fills_a_non_convex_region() {
        // Two cells of a box, taken together: their union is not convex, so
        // the triangulation of their corners covers more than the region and
        // the surplus has to be dropped.
        let p = cube();
        let boundary = [
            [0, 2, 1],
            [0, 1, 5],
            [0, 5, 2],
            [1, 2, 5],
            [0, 3, 2],
            [0, 2, 7],
            [0, 7, 3],
            [2, 3, 7],
        ];
        // Whether it succeeds depends on the triangulation; what must never
        // happen is a filling that does not tile the region.
        if let Ok(cells) = delaunay_fill(&p, &boundary, Constraints::default()) {
            check_tiles(&cells, &boundary);
        }
    }

    #[test]
    fn delaunay_honours_what_the_caller_needs() {
        let (p, faces) = bipyramid();
        let cells = delaunay_fill(&p, &faces, Constraints::default()).unwrap();
        // An edge the filling does have can be asked for; one it does not
        // want makes it decline rather than pretend.
        let (u, v) = (cells[0][0], cells[0][1]);
        assert!(delaunay_fill(
            &p,
            &faces,
            Constraints {
                with_edges: &[(u, v)],
                ..Default::default()
            }
        )
        .is_ok());
        assert!(delaunay_fill(
            &p,
            &faces,
            Constraints {
                without_edges: &[(u, v)],
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn a_tight_budget_gives_up_rather_than_running_on() {
        let p = cube();
        assert_eq!(fill(&p, &UNFILLABLE, Constraints::default(), 5), None);
    }
}
