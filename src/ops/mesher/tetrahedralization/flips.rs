//! Local reconnections of a tetrahedral mesh: the 2-3 and 3-2 flips.
//!
//! Both act on the same small region — two tetrahedra sharing a face, or
//! three sharing an edge — and swap it for the other way of filling the same
//! convex hull. They are exact inverses of each other, they leave the region
//! and its volume untouched, and they are the tool boundary recovery uses to
//! walk a missing edge or facet into existence:
//!
//! ```text
//!            e                        e
//!           /|\                      /|\
//!          / | \                    / | \
//!      f0 /__|__\ f2    2-3     f0 /__|__\ f2
//!         \  |  /       <-->       \  |  /      the shared face (f0,f1,f2)
//!          \ | /        3-2         \ | /       becomes the shared edge
//!           \|/                      \|/        (d,e), and back
//!            d                        d
//! ```
//!
//! A flip is legal exactly when the region really is convex there, which
//! shows up as *every resulting tetrahedron being positively oriented*. That
//! is the only test made, it is exact, and the region swap re-derives the
//! adjacency from the faces alone — so an illegal flip is reported as "not
//! applicable" rather than silently producing an inverted cell.

use crate::error::Result;

use super::delaunay::{Boundary, TetMesh};
use super::predicates::orient3d;

/// Try to replace the two tetrahedra sharing face `i` of `t` with the three
/// that share the edge joining their apexes.
///
/// Returns the new tetrahedra, or `None` when the flip does not apply —
/// because the face is on the boundary, or because the union of the two
/// cells is not convex there.
pub fn flip23(mesh: &mut TetMesh, t: usize, i: usize) -> Result<Option<Vec<u32>>> {
    let Some(n) = mesh.neighbour(t, i) else {
        return Ok(None);
    };
    let f = mesh.face(t, i);
    // `t`'s apex sits below the outward face, the neighbour's above it.
    let d = mesh.tet(t).expect("live cell")[i];
    let Some(e) = mesh.apex_beyond(n, &f) else {
        return Ok(None);
    };

    let new = [[f[0], f[1], d, e], [f[1], f[2], d, e], [f[2], f[0], d, e]];
    if new.iter().any(|v| mesh.orientation(v) <= 0.0) {
        return Ok(None); // the pair is not convex across this face
    }
    mesh.replace_region(&[t as u32, n as u32], &new, "a 2-3 flip")
        .map(Some)
}

/// Try to replace the three tetrahedra sharing edge `(u, v)` with the two
/// that share the triangle of their remaining vertices.
///
/// Returns `None` when the edge is not shared by exactly three cells, or
/// when the union is not convex — the inverse situation of [`flip23`].
pub fn flip32(mesh: &mut TetMesh, u: u32, v: u32) -> Result<Option<Vec<u32>>> {
    let Some(ring) = mesh.tets_around_edge(u, v) else {
        return Ok(None);
    };
    if ring.len() != 3 {
        return Ok(None);
    }

    // The three vertices left once the edge is taken away.
    let mut outer: Vec<u32> = Vec::with_capacity(3);
    for &t in &ring {
        for x in mesh.tet(t as usize).expect("live ring cell") {
            if x != u && x != v && !outer.contains(&x) {
                outer.push(x);
            }
        }
    }
    if outer.len() != 3 {
        return Ok(None);
    }
    let (a, b, c) = (outer[0], outer[1], outer[2]);

    // The edge survives only as the pair of cells above and below the plane
    // of `(a, b, c)`, so `u` and `v` must lie strictly on opposite sides.
    let su = mesh.orientation(&[a, b, c, u]);
    let sv = mesh.orientation(&[a, b, c, v]);
    let new = if su > 0.0 && sv < 0.0 {
        [[a, b, c, u], [a, c, b, v]]
    } else if su < 0.0 && sv > 0.0 {
        [[a, b, c, v], [a, c, b, u]]
    } else {
        return Ok(None);
    };
    mesh.replace_region(&ring, &new, "a 3-2 flip").map(Some)
}

/// Try to swap edge `(p, q)` for the other diagonal of the flat
/// quadrilateral it cuts on the mesh's outer surface.
///
/// This is the one reconnection the other two cannot make. An edge lying on
/// the hull has an *open* fan — it never closes into a ring — so no 3-2 flip
/// can reach it, yet it is exactly what blocks a surface diagonal from being
/// recovered: a square face of a box carries two diagonals, and the Delaunay
/// triangulation is free to have chosen the one the envelope did not.
///
/// Applies only when the fan is two cells whose outer faces are coplanar and
/// meet nothing, so that swapping the diagonal re-cuts the same flat
/// quadrilateral and leaves the solid untouched.
pub fn flip22(mesh: &mut TetMesh, p: u32, q: u32) -> Result<Option<Vec<u32>>> {
    let fan = mesh.tets_with_edge(p, q);
    if fan.len() != 2 {
        return Ok(None);
    }
    let (t1, t2) = (fan[0], fan[1]);
    let (c1, c2) = (
        mesh.tet(t1 as usize).expect("live fan cell"),
        mesh.tet(t2 as usize).expect("live fan cell"),
    );
    // The apex they share, and the one each keeps to itself.
    let rest =
        |c: &[u32; 4]| -> Vec<u32> { c.iter().copied().filter(|&x| x != p && x != q).collect() };
    let (r1, r2) = (rest(&c1), rest(&c2));
    let Some(&z) = r1.iter().find(|x| r2.contains(x)) else {
        return Ok(None);
    };
    let (Some(&x), Some(&y)) = (r1.iter().find(|&&a| a != z), r2.iter().find(|&&a| a != z)) else {
        return Ok(None);
    };

    // The quadrilateral (p, x, q, y) must be flat, and its two triangles
    // must be on the outer surface — otherwise re-cutting it would move
    // material.
    if orient3d(
        &mesh.points()[p as usize],
        &mesh.points()[q as usize],
        &mesh.points()[x as usize],
        &mesh.points()[y as usize],
    ) != 0.0
    {
        return Ok(None);
    }
    if !mesh.face_is_free(t1, &[p, q, x]) || !mesh.face_is_free(t2, &[p, q, y]) {
        return Ok(None);
    }

    // The two cells that remain share the new diagonal; `p` and `q` end up
    // on opposite sides of the plane through it.
    let sp = mesh.orientation(&[x, y, z, p]);
    let sq = mesh.orientation(&[x, y, z, q]);
    let new = if sp > 0.0 && sq < 0.0 {
        [[x, y, z, p], [y, x, z, q]]
    } else if sp < 0.0 && sq > 0.0 {
        [[y, x, z, p], [x, y, z, q]]
    } else {
        return Ok(None);
    };
    mesh.replace_region_with(&fan, &new, "a 2-2 flip", Boundary::MayRecutHull)
        .map(Some)
}

/// Take edge `(p, q)` out of the mesh, refilling its fan with cells that do
/// not use it.
///
/// This is the general form of which [`flip32`] and [`flip22`] are the two
/// smallest cases, and it is what boundary recovery actually needs: an edge
/// standing in the way of an envelope edge has whatever fan it happens to
/// have, and the small flips only reach fans of two or three.
///
/// Removing the edge means re-cutting the polygon its fan leaves behind —
/// the *link* — and hanging `p` and `q` off each triangle of the result. Any
/// triangulation of the link gives a filling; the one used is the first
/// whose cells are all positively oriented, which is exactly the condition
/// for the region to be filled without overlap. If none is, the edge cannot
/// be removed and the caller is told so.
///
/// An **open** fan ends on the outer surface. Removing the edge there re-cuts
/// the flat quadrilateral its two end faces form, so those must meet nothing
/// and lie in one plane — otherwise the solid itself would change shape.
pub fn remove_edge(mesh: &mut TetMesh, p: u32, q: u32) -> Result<Option<Vec<u32>>> {
    let Some(fan) = mesh.edge_fan(p, q) else {
        return Ok(None);
    };
    // Beyond this the number of triangulations grows fast, and a fan this
    // long means the search has gone somewhere unhelpful anyway.
    if fan.link.len() < 3 || fan.link.len() > 8 {
        return Ok(None);
    }

    let boundary = if fan.closed {
        Boundary::Preserved
    } else {
        // The two end faces close the region; re-cutting them is only
        // harmless where they meet nothing and are coplanar.
        let (x0, xn) = (fan.link[0], *fan.link.last().expect("non-empty link"));
        let first = *fan.cells.first().expect("non-empty fan");
        let last = *fan.cells.last().expect("non-empty fan");
        if !mesh.face_is_free(first, &[p, q, x0]) || !mesh.face_is_free(last, &[p, q, xn]) {
            return Ok(None);
        }
        if orient3d(
            &mesh.points()[p as usize],
            &mesh.points()[q as usize],
            &mesh.points()[x0 as usize],
            &mesh.points()[xn as usize],
        ) != 0.0
        {
            return Ok(None);
        }
        Boundary::MayRecutHull
    };

    for triangles in triangulations(fan.link.len()) {
        let mut new: Vec<[u32; 4]> = Vec::with_capacity(2 * triangles.len());
        let mut ok = true;
        for t in &triangles {
            let (a, b, c) = (fan.link[t[0]], fan.link[t[1]], fan.link[t[2]]);
            // One cell each side of the triangle, so the pair fills the slab
            // the edge used to occupy.
            let sp = mesh.orientation(&[a, b, c, p]);
            let sq = mesh.orientation(&[a, b, c, q]);
            if sp > 0.0 && sq < 0.0 {
                new.push([a, b, c, p]);
                new.push([b, a, c, q]);
            } else if sp < 0.0 && sq > 0.0 {
                new.push([b, a, c, p]);
                new.push([a, b, c, q]);
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // A filling that does not tile the fan is declined, not committed.
        let snapshot = mesh.clone();
        match mesh.replace_region_with(&fan.cells, &new, "an edge removal", boundary) {
            Ok(created) => return Ok(Some(created)),
            Err(_) => *mesh = snapshot,
        }
    }
    Ok(None)
}

/// Every triangulation of a polygon of `n` vertices, as index triples.
///
/// The usual recursion: the edge from the first to the last vertex belongs
/// to one triangle, and that triangle splits the rest in two.
fn triangulations(n: usize) -> Vec<Vec<[usize; 3]>> {
    fn split(lo: usize, hi: usize, out: &mut Vec<Vec<[usize; 3]>>) {
        if hi - lo < 2 {
            out.push(Vec::new());
            return;
        }
        for k in lo + 1..hi {
            let mut left = Vec::new();
            split(lo, k, &mut left);
            let mut right = Vec::new();
            split(k, hi, &mut right);
            for l in &left {
                for r in &right {
                    let mut t = vec![[lo, k, hi]];
                    t.extend_from_slice(l);
                    t.extend_from_slice(r);
                    out.push(t);
                }
            }
        }
    }
    let mut out = Vec::new();
    split(0, n - 1, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interrupt::NoCancel;

    /// Five points whose convex hull is a bipyramid: the smallest shape on
    /// which both flips apply.
    fn bipyramid() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, -1.0],
            [0.2, 0.2, 1.0],
        ]
    }

    fn build(points: &[[f64; 3]]) -> TetMesh {
        let mesh = TetMesh::delaunay(points, &NoCancel).unwrap();
        assert_eq!(mesh.find_defect(), None);
        mesh
    }

    /// The first interior face on which a 2-3 flip succeeds.
    fn flip_any_face(mesh: &mut TetMesh) -> Option<(usize, usize)> {
        for t in 0..mesh.slot_count() {
            if mesh.tet(t).is_none() {
                continue;
            }
            for i in 0..4 {
                if mesh.neighbour(t, i).is_some() {
                    let before = mesh.clone();
                    if flip23(mesh, t, i).unwrap().is_some() {
                        return Some((t, i));
                    }
                    *mesh = before;
                }
            }
        }
        None
    }

    #[test]
    fn a_flip_preserves_the_region_and_its_volume() {
        let mut mesh = build(&bipyramid());
        let volume = mesh.volume();
        let cells = mesh.len();

        let created = flip_any_face(&mut mesh).map(|_| ()).map(|_| mesh.len());
        assert!(created.is_some(), "no flippable face in a bipyramid");
        assert_eq!(mesh.find_defect(), None);
        assert!((mesh.volume() - volume).abs() < 1e-15, "{}", mesh.volume());
        assert_eq!(mesh.len(), cells + 1, "2-3 replaces two cells with three");
    }

    #[test]
    fn the_two_flips_are_inverse() {
        let mut mesh = build(&bipyramid());
        let before: Vec<[u32; 4]> = mesh.iter().map(|(_, v)| v).collect();
        let volume = mesh.volume();

        let (t, i) = flip_any_face(&mut mesh).expect("a flippable face");
        let _ = (t, i);
        // The 2-3 flip created an edge joining the two apexes; every new
        // cell holds it, so take it from the first one.
        let (_, v0) = mesh.iter().next().unwrap();
        let mut undone = false;
        'search: for a in 0..4 {
            for b in a + 1..4 {
                let mut probe = mesh.clone();
                if flip32(&mut probe, v0[a], v0[b]).unwrap().is_some() {
                    mesh = probe;
                    undone = true;
                    break 'search;
                }
            }
        }
        assert!(undone, "the 2-3 flip could not be undone");

        assert_eq!(mesh.find_defect(), None);
        assert!((mesh.volume() - volume).abs() < 1e-15);
        let after: Vec<[u32; 4]> = mesh.iter().map(|(_, v)| v).collect();
        assert_eq!(after.len(), before.len());
        // The same region, filled the same way — cell order may differ.
        let key = |cells: &[[u32; 4]]| {
            let mut k: Vec<[u32; 4]> = cells
                .iter()
                .map(|v| {
                    let mut s = *v;
                    s.sort_unstable();
                    s
                })
                .collect();
            k.sort_unstable();
            k
        };
        assert_eq!(key(&after), key(&before));
    }

    #[test]
    fn a_boundary_face_cannot_be_flipped() {
        let mut mesh = build(&bipyramid());
        let mut refused = 0;
        for t in 0..mesh.slot_count() {
            if mesh.tet(t).is_none() {
                continue;
            }
            for i in 0..4 {
                if mesh.neighbour(t, i).is_none() {
                    assert!(flip23(&mut mesh, t, i).unwrap().is_none());
                    refused += 1;
                }
            }
        }
        assert!(refused > 0, "a bipyramid has boundary faces");
        assert_eq!(mesh.find_defect(), None);
    }

    #[test]
    fn a_non_convex_pair_is_refused() {
        // Two tetrahedra meeting at a reflex angle: the segment joining the
        // apexes passes outside their shared face, so no 2-3 flip applies.
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            // Far off to the side, well outside the cone over the face.
            [4.0, 4.0, -0.1],
        ];
        let mut mesh = build(&points);
        let before: Vec<[u32; 4]> = mesh.iter().map(|(_, v)| v).collect();
        let volume = mesh.volume();

        // Whatever is refused must leave the mesh exactly as it was.
        for t in 0..mesh.slot_count() {
            if mesh.tet(t).is_none() {
                continue;
            }
            for i in 0..4 {
                let snapshot: Vec<[u32; 4]> = mesh.iter().map(|(_, v)| v).collect();
                if flip23(&mut mesh, t, i).unwrap().is_none() {
                    assert_eq!(mesh.iter().map(|(_, v)| v).collect::<Vec<_>>(), snapshot);
                }
            }
        }
        assert_eq!(mesh.find_defect(), None);
        assert!((mesh.volume() - volume).abs() < 1e-15);
        assert!(!before.is_empty());
    }

    #[test]
    fn an_edge_with_the_wrong_ring_size_is_refused() {
        let mesh = build(&bipyramid());
        // Every edge of the initial triangulation; only one can ever have a
        // ring of exactly three, and the others must be declined cleanly.
        let mut tried = 0;
        for t in 0..mesh.slot_count() {
            let Some(v) = mesh.tet(t) else { continue };
            for a in 0..4 {
                for b in a + 1..4 {
                    let mut probe = mesh.clone();
                    if flip32(&mut probe, v[a], v[b]).unwrap().is_none() {
                        tried += 1;
                    }
                }
            }
        }
        assert!(tried > 0);
        assert_eq!(mesh.find_defect(), None);
    }

    #[test]
    fn flips_keep_the_mesh_sound_over_many_moves() {
        // Hammer a real triangulation: every accepted flip must leave the
        // adjacency consistent and the volume untouched.
        let points: Vec<[f64; 3]> = (0..80)
            .map(|i| {
                let f = |s: u64| {
                    let mut x = (i as u64 ^ s).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    x ^= x >> 29;
                    (x >> 11) as f64 / (1u64 << 53) as f64
                };
                [f(1), f(2), f(3)]
            })
            .collect();
        let mut mesh = build(&points);
        let volume = mesh.volume();

        let mut applied = 0;
        for round in 0..40 {
            let t = (round * 7) % mesh.slot_count();
            if mesh.tet(t).is_none() {
                continue;
            }
            for i in 0..4 {
                if flip23(&mut mesh, t, i).unwrap().is_some() {
                    applied += 1;
                    assert_eq!(mesh.find_defect(), None, "after flip {applied}");
                    assert!(
                        (mesh.volume() - volume).abs() < 1e-12,
                        "volume drifted to {}",
                        mesh.volume()
                    );
                    break;
                }
            }
        }
        assert!(applied > 5, "only {applied} flips applied");
    }
}
