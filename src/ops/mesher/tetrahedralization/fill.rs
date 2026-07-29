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

use std::collections::HashMap;

use super::predicates::orient3d;

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

    #[test]
    fn a_tight_budget_gives_up_rather_than_running_on() {
        let p = cube();
        assert_eq!(fill(&p, &UNFILLABLE, Constraints::default(), 5), None);
    }
}
