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

use super::delaunay::{Boundary, EdgeFan, TetMesh};
use super::fill::{fill, Constraints, DEFAULT_BUDGET};

/// Cells a pocket may hold before widening it is abandoned. Past this the
/// exhaustive rebuild costs more than it is worth.
const MAX_REGION: usize = 16;
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

/// Take edge `(p, q)` out of the mesh, rebuilding its fan without it.
///
/// This is the general form of which [`flip32`] and [`flip22`] are the two
/// smallest cases, and it is what boundary recovery actually needs: an edge
/// standing in the way of an envelope edge has whatever fan it happens to
/// have, and the small flips only reach fans of two or three cells.
///
/// The fan is handed to [`fill`] as a closed region with the edge forbidden,
/// so the answer is complete: if any way of rebuilding that pocket without
/// the edge exists, it is found; if none does, the edge genuinely cannot go
/// and the caller is told so. Enumerating a *pattern* instead — re-cutting
/// the ring of vertices and hanging `p` and `q` off each triangle — misses
/// the fillings that use a cell of four ring vertices, which on a box is
/// precisely the one that works.
///
/// An **open** fan ends on the outer surface. Removing the edge there re-cuts
/// the flat quadrilateral its two end faces form along the other diagonal,
/// so those faces must meet nothing and lie in one plane — otherwise the
/// solid itself would change shape.
pub fn remove_edge(
    mesh: &mut TetMesh,
    p: u32,
    q: u32,
    protect: &[[u32; 3]],
) -> Result<Option<Vec<u32>>> {
    let Some(fan) = mesh.edge_fan(p, q) else {
        return Ok(None);
    };
    // The fan alone is often too tight: which cells may exist around a
    // vertex is forced by its link, and those forced choices can leave the
    // rest of a small pocket with no way to close. Widening the pocket by a
    // layer of neighbours gives the rebuild the room it needs.
    let mut region = fan.cells.clone();
    loop {
        if let Some(created) = rebuild_without(mesh, p, q, &fan, &region, protect)? {
            return Ok(Some(created));
        }
        let wider = grow(mesh, &region);
        if wider.len() == region.len() || wider.len() > MAX_REGION {
            return Ok(None);
        }
        region = wider;
    }
}

/// Rebuild `region` without the edge `(p, q)`, or report that it cannot be.
fn rebuild_without(
    mesh: &mut TetMesh,
    p: u32,
    q: u32,
    fan: &EdgeFan,
    region: &[u32],
    protect: &[[u32; 3]],
) -> Result<Option<Vec<u32>>> {
    let mut boundary = mesh.region_boundary(region);

    let mode = if fan.closed {
        Boundary::Preserved
    } else {
        let (x0, xn) = (fan.link[0], *fan.link.last().expect("non-empty link"));
        let first = *fan.cells.first().expect("non-empty fan");
        let last = *fan.cells.last().expect("non-empty fan");
        if !mesh.face_is_free(first, &[p, q, x0]) || !mesh.face_is_free(last, &[p, q, xn]) {
            return Ok(None);
        }
        let pts = mesh.points();
        if orient3d(
            &pts[p as usize],
            &pts[q as usize],
            &pts[x0 as usize],
            &pts[xn as usize],
        ) != 0.0
        {
            return Ok(None); // the two end faces are not one flat quadrilateral
        }
        // Re-cutting swaps those two triangles for two others. An envelope
        // facet is not ours to swap: it is the answer, not scratch space.
        if protect
            .iter()
            .any(|g| sorted(g) == sorted(&[p, q, x0]) || sorted(g) == sorted(&[p, q, xn]))
        {
            return Ok(None);
        }
        let Some(recut) = recut_end_faces(mesh, fan, p, q) else {
            return Ok(None);
        };
        boundary.retain(|f| {
            let k = sorted(f);
            k != sorted(&[p, q, x0]) && k != sorted(&[p, q, xn])
        });
        boundary.extend_from_slice(&recut);
        Boundary::MayRecutHull
    };

    // Whatever of the envelope this pocket already holds has to survive the
    // rebuild, or recovery spends its passes trading one facet for another.
    let keep = relevant(mesh, region, protect);
    let Some(new) = fill(
        mesh.points(),
        &boundary,
        Constraints {
            without_edges: &[(p, q)],
            with_faces: &keep,
            ..Default::default()
        },
        DEFAULT_BUDGET,
    ) else {
        return Ok(None);
    };
    // A filling the mesh will not take is declined, not committed by halves.
    let snapshot = mesh.clone();
    match mesh.replace_region_with(region, &new, "an edge removal", mode) {
        Ok(created) => Ok(Some(created)),

        Err(_) => {
            *mesh = snapshot;
            Ok(None)
        }
    }
}

/// The envelope facets a rebuild of this region could disturb: those wholly
/// inside it that the mesh already holds.
///
/// Handing them to the search as walls is what stops recovery from undoing
/// its own work — without it, freeing one edge quietly buries a facet won
/// earlier, and the sweep chases its tail.
pub fn relevant(mesh: &TetMesh, region: &[u32], protect: &[[u32; 3]]) -> Vec<[u32; 3]> {
    if protect.is_empty() {
        return Vec::new();
    }
    // The region's own vertices, not just those on its surface: while the
    // envelope is being recovered a facet of a concave body is an *interior*
    // face of the triangulation, and those are exactly the ones a rebuild
    // would quietly bury.
    let mut here: Vec<u32> = region
        .iter()
        .filter_map(|&t| mesh.tet(t as usize))
        .flatten()
        .collect();
    here.sort_unstable();
    here.dedup();
    protect
        .iter()
        .filter(|f| f.iter().all(|x| here.binary_search(x).is_ok()) && mesh.has_face(f))
        .copied()
        .collect()
}

/// The region plus every cell touching it through a face.
fn grow(mesh: &TetMesh, region: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = region.to_vec();
    for &t in region {
        for i in 0..4 {
            if let Some(n) = mesh.neighbour(t as usize, i) {
                if !out.contains(&(n as u32)) {
                    out.push(n as u32);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// The two triangles replacing an open fan's end faces, cut along the other
/// diagonal of their flat quadrilateral and facing the same way.
fn recut_end_faces(mesh: &TetMesh, fan: &EdgeFan, p: u32, q: u32) -> Option<[[u32; 3]; 2]> {
    let (x0, xn) = (fan.link[0], *fan.link.last()?);
    // The apex of the first cell lies strictly inside the solid, below the
    // end face; a replacement facing the same way keeps it below too.
    let inside = mesh.apex_beyond(*fan.cells.first()? as usize, &[p, q, x0])?;
    let facing_out = |f: [u32; 3]| -> [u32; 3] {
        let pts = mesh.points();
        if orient3d(
            &pts[f[0] as usize],
            &pts[f[1] as usize],
            &pts[f[2] as usize],
            &pts[inside as usize],
        ) < 0.0
        {
            f
        } else {
            [f[0], f[2], f[1]]
        }
    };
    Some([facing_out([x0, xn, p]), facing_out([x0, xn, q])])
}

fn sorted(f: &[u32; 3]) -> [u32; 3] {
    let mut k = *f;
    k.sort_unstable();
    k
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
