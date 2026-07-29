//! Re-cutting a flat strip of the mesh's outer surface.
//!
//! One family of envelope edges cannot be recovered by rebuilding a pocket of
//! cells, however thoroughly: those that run **inside a flat face of the
//! solid**. There the obstruction is the other diagonal of that face, and
//! swapping it is not a question about volume at all — it is the plain 2-D
//! question of how to cut a polygon. Worse, the swap is often illegal on its
//! own: the quadrilateral formed by two adjacent surface triangles may be
//! non-convex, so the diagonal cannot simply be flipped, and a neighbour has
//! to move first.
//!
//! Both difficulties disappear once the problem is treated as what it is.
//! This module walks the segment across the flat strip of surface triangles
//! it crosses, cuts that strip's polygon afresh with the segment as one of
//! its diagonals, and hands the cells behind it to
//! [`fill`](super::fill) to be rebuilt against the new surface. No
//! sequence of flips is searched for, because none is needed.
//!
//! The walk is exact: which triangle the segment enters, and which edge it
//! leaves through, are decided by [`orient2d`] in the strip's own plane.
//! Re-cutting preserves the polygon, hence the solid's shape, and the region
//! swap re-checks that the volume is unchanged before committing.

use std::collections::HashMap;

use crate::containers::mesh::Point2;
use crate::error::Result;

use super::delaunay::{Boundary, TetMesh};
use super::fill::{fill, Constraints, DEFAULT_BUDGET};
use super::predicates::{orient2d, orient3d};
use crate::ops::mesher::triangulation::{ear_clip_2d, signed_area};

/// Triangles a single strip may hold before the walk is called off.
const STRIP_LIMIT: usize = 256;

/// Make `(u, v)` an edge of the mesh by re-cutting the flat surface strip it
/// runs across.
///
/// Returns `false` when the segment does not run along a flat piece of the
/// outer surface, or when the strip cannot be re-cut — in which case nothing
/// has been touched.
pub fn recut_flat_strip(mesh: &mut TetMesh, u: u32, v: u32, protect: &[[u32; 3]]) -> Result<bool> {
    let Some(strip) = walk_strip(mesh, u, v) else {
        return Ok(false);
    };
    let faces: Vec<[u32; 3]> = strip
        .iter()
        .map(|&(t, i)| mesh.face(t as usize, i))
        .collect();
    // The strip is about to be cut differently; an envelope facet caught in
    // it would be destroyed, so the whole re-cut is off.
    if faces
        .iter()
        .any(|f| protect.iter().any(|g| sorted(g) == sorted(f)))
    {
        return Ok(false);
    }
    let Some(loop_) = boundary_loop(&faces) else {
        return Ok(false);
    };
    let Some(new_faces) = recut(mesh, &loop_, u, v, &strip) else {
        return Ok(false);
    };

    // The cells behind the strip are rebuilt against the new surface.
    let mut region: Vec<u32> = strip.iter().map(|&(t, _)| t).collect();
    region.sort_unstable();
    region.dedup();
    let mut boundary = mesh.region_boundary(&region);
    let old: Vec<[u32; 3]> = faces.iter().map(sorted).collect();
    boundary.retain(|f| !old.contains(&sorted(f)));
    boundary.extend_from_slice(&new_faces);

    let keep = super::flips::relevant(mesh, &region, protect);
    let Some(cells) = fill(
        mesh.points(),
        &boundary,
        Constraints {
            with_faces: &keep,
            ..Default::default()
        },
        DEFAULT_BUDGET,
    ) else {
        return Ok(false);
    };
    let snapshot = mesh.clone();
    match mesh.replace_region_with(&region, &cells, "a surface re-cut", Boundary::MayRecutHull) {
        Ok(_) => Ok(true),
        Err(_) => {
            *mesh = snapshot;
            Ok(false)
        }
    }
}

// ─── Walking the strip ──────────────────────────────────────────────────

/// The outer-surface triangles the open segment `(u, v)` crosses, in order,
/// provided they all lie in one plane with it.
fn walk_strip(mesh: &TetMesh, u: u32, v: u32) -> Option<Vec<(u32, usize)>> {
    let start = hull_faces_at(mesh, u)
        .into_iter()
        .find(|&(t, i)| coplanar_with(mesh, &mesh.face(t as usize, i), v))?;
    let plane = Plane::of(mesh, &mesh.face(start.0 as usize, start.1))?;
    let at = |x: u32| plane.project(&mesh.points()[x as usize]);

    // Which triangle around `u` the segment sets off into: the one whose
    // wedge at `u` contains the direction of `v`.
    let mut current = hull_faces_at(mesh, u).into_iter().find(|&(t, i)| {
        let f = mesh.face(t as usize, i);
        coplanar_with(mesh, &f, v) && wedge_holds(&at, &f, u, v)
    })?;

    let mut strip = vec![current];
    // The edge the segment left through: opposite `u` in the first triangle.
    let f = mesh.face(current.0 as usize, current.1);
    if f.contains(&v) {
        return Some(strip); // `u` and `v` already share a triangle
    }
    let mut entry: [u32; 2] = {
        let rest: Vec<u32> = f.iter().copied().filter(|&x| x != u).collect();
        [rest[0], rest[1]]
    };

    for _ in 0..STRIP_LIMIT {
        // Cross the edge into the triangle on its other side.
        let next = hull_faces_at(mesh, entry[0]).into_iter().find(|&(t, i)| {
            (t, i) != current
                && !strip.contains(&(t, i))
                && mesh.face(t as usize, i).contains(&entry[1])
                && coplanar_with(mesh, &mesh.face(t as usize, i), v)
        })?;
        current = next;
        strip.push(current);

        let f = mesh.face(current.0 as usize, current.1);
        let z = *f.iter().find(|&&x| x != entry[0] && x != entry[1])?;
        if z == v {
            return Some(strip);
        }
        // The segment leaves through whichever of the two remaining edges
        // straddles it, which is the one whose far end is on the other side
        // of the line from `z`.
        let side = |x: u32| orient2d(&at(u), &at(v), &at(x));
        let sz = side(z);
        if sz == 0.0 {
            return None; // a node sits on the segment: not our case
        }
        let keep = if side(entry[0]) * sz < 0.0 {
            entry[0]
        } else if side(entry[1]) * sz < 0.0 {
            entry[1]
        } else {
            return None;
        };
        entry = [keep, z];
    }
    None
}

/// The outer-surface faces having `v` as a vertex.
fn hull_faces_at(mesh: &TetMesh, v: u32) -> Vec<(u32, usize)> {
    let mut out: Vec<(u32, usize)> = Vec::new();
    for t in mesh.tets_around_vertex(v) {
        for i in 0..4 {
            if mesh.neighbour(t as usize, i).is_none() && mesh.face(t as usize, i).contains(&v) {
                out.push((t, i));
            }
        }
    }
    out.sort_unstable();
    out
}

fn coplanar_with(mesh: &TetMesh, f: &[u32; 3], x: u32) -> bool {
    let p = mesh.points();
    orient3d(
        &p[f[0] as usize],
        &p[f[1] as usize],
        &p[f[2] as usize],
        &p[x as usize],
    ) == 0.0
}

/// Whether the direction from `u` to `v` points into the triangle `f`, which
/// has `u` as one of its vertices.
fn wedge_holds(at: &impl Fn(u32) -> [f64; 2], f: &[u32; 3], u: u32, v: u32) -> bool {
    let rest: Vec<u32> = f.iter().copied().filter(|&x| x != u).collect();
    if rest.len() != 2 {
        return false;
    }
    let (x, y) = (rest[0], rest[1]);
    let (pu, pv, px, py) = (at(u), at(v), at(x), at(y));
    // `v` must lie on `y`'s side of the ray to `x`, and on `x`'s side of the
    // ray to `y`.
    let a = orient2d(&pu, &px, &pv) * orient2d(&pu, &px, &py);
    let b = orient2d(&pu, &py, &pv) * orient2d(&pu, &py, &px);
    a > 0.0 && b > 0.0
}

// ─── Re-cutting ─────────────────────────────────────────────────────────

/// The boundary of a set of triangles glued along shared edges, as one loop
/// of vertices in order.
fn boundary_loop(faces: &[[u32; 3]]) -> Option<Vec<u32>> {
    let mut next: HashMap<u32, u32> = HashMap::new();
    let mut seen: HashMap<(u32, u32), u32> = HashMap::new();
    for f in faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            *seen.entry(key).or_default() += 1;
        }
    }
    for f in faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = if a < b { (a, b) } else { (b, a) };
            if seen[&key] == 1 && next.insert(a, b).is_some() {
                return None; // a vertex with two ways out: not a simple loop
            }
        }
    }
    if next.is_empty() {
        return None;
    }
    let start = *next.keys().min()?;
    let mut loop_ = vec![start];
    let mut cur = start;
    while let Some(&n) = next.get(&cur) {
        if n == start {
            return (loop_.len() == next.len()).then_some(loop_);
        }
        loop_.push(n);
        cur = n;
        if loop_.len() > next.len() {
            return None;
        }
    }
    None
}

/// Cut the strip's polygon again, this time with `(u, v)` as a diagonal.
fn recut(
    mesh: &TetMesh,
    loop_: &[u32],
    u: u32,
    v: u32,
    strip: &[(u32, usize)],
) -> Option<Vec<[u32; 3]>> {
    let iu = loop_.iter().position(|&x| x == u)?;
    let iv = loop_.iter().position(|&x| x == v)?;
    let plane = Plane::of(mesh, &mesh.face(strip[0].0 as usize, strip[0].1))?;

    // The diagonal splits the loop into two chains, each a polygon in its
    // own right.
    let chain = |from: usize, to: usize| -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = from;
        loop {
            out.push(loop_[i]);
            if i == to {
                return out;
            }
            i = (i + 1) % loop_.len();
        }
    };
    let halves = [chain(iu, iv), chain(iv, iu)];

    // A cell behind the strip gives the inside, which fixes which way the
    // new triangles must face.
    let anchor = mesh.tet(strip[0].0 as usize)?[strip[0].1];

    let mut out: Vec<[u32; 3]> = Vec::new();
    for half in &halves {
        if half.len() < 3 {
            return None;
        }
        let pts: Vec<Point2> = half
            .iter()
            .map(|&x| {
                let q = plane.project(&mesh.points()[x as usize]);
                Point2::new(q[0], q[1])
            })
            .collect();
        if signed_area(&pts) == 0.0 {
            return None;
        }
        for t in ear_clip_2d(&pts).ok()? {
            out.push(facing_out(
                mesh,
                [half[t[0]], half[t[1]], half[t[2]]],
                anchor,
            ));
        }
    }
    Some(out)
}

/// The triangle wound so that `inside` lies below it.
fn facing_out(mesh: &TetMesh, f: [u32; 3], inside: u32) -> [u32; 3] {
    let p = mesh.points();
    if orient3d(
        &p[f[0] as usize],
        &p[f[1] as usize],
        &p[f[2] as usize],
        &p[inside as usize],
    ) < 0.0
    {
        f
    } else {
        [f[0], f[2], f[1]]
    }
}

fn sorted(f: &[u32; 3]) -> [u32; 3] {
    let mut k = *f;
    k.sort_unstable();
    k
}

/// The strip's plane, reduced to the two coordinates that describe it
/// without degenerating.
struct Plane {
    axes: (usize, usize),
}

impl Plane {
    fn of(mesh: &TetMesh, f: &[u32; 3]) -> Option<Plane> {
        let p = mesh.points();
        let (a, b, c) = (p[f[0] as usize], p[f[1] as usize], p[f[2] as usize]);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        // Dropping the axis the normal leans on keeps the projection of a
        // non-degenerate triangle non-degenerate.
        let drop = (0..3).max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))?;
        Some(Plane {
            axes: match drop {
                0 => (1, 2),
                1 => (2, 0),
                _ => (0, 1),
            },
        })
    }

    fn project(&self, p: &[f64; 3]) -> [f64; 2] {
        [p[self.axes.0], p[self.axes.1]]
    }
}
