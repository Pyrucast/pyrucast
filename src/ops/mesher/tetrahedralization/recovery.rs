//! Putting the envelope back into the triangulation.
//!
//! The Delaunay triangulation of the envelope's nodes covers their convex
//! hull, and it owes nothing to the surface those nodes came from: an edge
//! of the envelope may be crossed by a tetrahedron, and a facet may be
//! pierced by an edge. Before the inside can be told from the outside, every
//! envelope edge and every envelope facet has to *appear* in the
//! triangulation.
//!
//! Both are recovered by local reconnection — a 2-3 flip, or the removal of
//! a whole edge — driven by an exact answer to "what is in the way?":
//!
//! - a **missing edge** `(u, v)` is blocked by faces its segment passes
//!   through, and by edges its segment crosses. Flipping the obstruction
//!   away moves the triangulation toward holding the edge.
//! - a **missing facet**, once its three edges are present, is blocked only
//!   by edges piercing its interior. Removing them makes it appear.
//!
//! No Steiner point is ever placed on the envelope, at any stage: that is
//! the contract, and it is also what makes failure possible. Some polyhedra
//! — Schönhardt's twisted prism is the textbook one — admit **no**
//! tetrahedralization at all without a new boundary vertex. On those, and on
//! anything the flip search cannot untangle, this module reports precisely
//! which edge or facet is stuck rather than returning a mesh that does not
//! match the surface it was given.

use std::collections::HashSet;

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;

use super::delaunay::TetMesh;
use super::envelope::Envelope;
use super::flips::{flip23, remove_edge};
use super::intersect::segment_hits_triangle;
use super::predicates::orient3d;

/// Flips allowed per edge or facet before the search is called off.
///
/// Recovery normally takes a handful; a run that reaches this many is not
/// converging, and stopping turns a hang into a diagnosis.
const FLIP_BUDGET: usize = 512;

/// Make every edge and every facet of `envelope` present in `mesh`.
pub fn recover(mesh: &mut TetMesh, envelope: &Envelope, cancel: &dyn Cancel) -> Result<()> {
    // Edges first: a facet cannot be recovered while its own sides are
    // missing.
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(3 * envelope.facets().len());
    for f in envelope.facets() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            edges.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    edges.sort_unstable();
    edges.dedup();

    for &(u, v) in &edges {
        cancel.check()?;
        if mesh.has_edge(u, v) {
            continue;
        }
        recover_edge(mesh, u, v)?;
        if !mesh.has_edge(u, v) {
            return Err(stuck(mesh, "edge", &[u, v]));
        }
    }

    for f in envelope.facets() {
        cancel.check()?;
        if mesh.has_face(f) {
            continue;
        }
        recover_facet(mesh, f)?;
        if !mesh.has_face(f) {
            return Err(stuck(mesh, "facet", f));
        }
    }
    Ok(())
}

/// The diagnosis handed back when recovery gives up.
fn stuck(mesh: &TetMesh, what: &str, vertices: &[u32]) -> PyrucastError {
    let where_ = vertices
        .iter()
        .map(|&i| {
            let p = mesh.points()[i as usize];
            format!("({:.6}, {:.6}, {:.6})", p[0], p[1], p[2])
        })
        .collect::<Vec<_>>()
        .join(" – ");
    PyrucastError::Message(format!(
        "mesh_volume: cannot fit the envelope's {what} {where_} into the mesh without adding a \
         node on the surface, which the contract forbids. Refine the surface mesh around it — \
         some shapes (a twisted prism, a very flat sliver of a solid) admit no tetrahedral \
         mesh at all on their own nodes."
    ))
}

// ─── Edges ──────────────────────────────────────────────────────────────

/// Flip obstructions away until `(u, v)` is an edge of the mesh, or until
/// nothing applies.
fn recover_edge(mesh: &mut TetMesh, u: u32, v: u32) -> Result<()> {
    for _ in 0..FLIP_BUDGET {
        if mesh.has_edge(u, v) {
            return Ok(());
        }
        if !clear_one_obstruction(mesh, u, v)? {
            return Ok(()); // caller reports the failure
        }
    }
    Ok(())
}

/// Remove one thing standing between `u` and `v`, if any move applies.
///
/// Everything the segment runs through is fair game, not just what touches
/// `u`: an envelope edge spanning a wide flat face is typically blocked far
/// from either end.
fn clear_one_obstruction(mesh: &mut TetMesh, u: u32, v: u32) -> Result<bool> {
    let corridor = corridor(mesh, u, v);

    // A face the segment goes through has to open up. The 2-3 flip that
    // opens it replaces it by an edge pointing further along.
    for &t in &corridor {
        for i in 0..4 {
            let f = mesh.face(t as usize, i);
            if f.contains(&u) || f.contains(&v) {
                continue;
            }
            if pierces_triangle_between(mesh, u, v, &f) && flip23(mesh, t as usize, i)?.is_some() {
                return Ok(true);
            }
        }
    }

    // Otherwise the segment runs exactly through an existing edge — the two
    // diagonals of a flat quadrilateral crossing at its centre, say. That
    // edge has to go.
    for &t in &corridor {
        let cell = mesh.tet(t as usize).expect("live corridor cell");
        for a in 0..4 {
            for b in a + 1..4 {
                let (p, q) = (cell[a], cell[b]);
                if p == u || q == u || p == v || q == v {
                    continue;
                }
                if segments_cross(mesh, u, v, p, q) && retire_edge(mesh, p, q)? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Every cell the segment from `u` to `v` passes through or touches.
///
/// Grown outwards from `u`'s star: a cell the segment meets is always a
/// neighbour of another one it meets, so the walk finds all of them.
fn corridor(mesh: &TetMesh, u: u32, v: u32) -> Vec<u32> {
    let touched = |t: u32| -> bool {
        let cell = mesh.tet(t as usize).expect("live cell");
        if cell.contains(&u) || cell.contains(&v) {
            return true;
        }
        let pts = mesh.points();
        (0..4).any(|i| {
            let f = mesh.face(t as usize, i);
            segment_hits_triangle(
                &pts[u as usize],
                &pts[v as usize],
                &[
                    &pts[f[0] as usize],
                    &pts[f[1] as usize],
                    &pts[f[2] as usize],
                ],
            )
        })
    };

    let mut seen: HashSet<u32> = HashSet::new();
    let mut out: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = mesh
        .tets_around_vertex(u)
        .into_iter()
        .filter(|&t| seen.insert(t))
        .collect();
    while let Some(t) = stack.pop() {
        if !touched(t) {
            continue;
        }
        out.push(t);
        for i in 0..4 {
            if let Some(n) = mesh.neighbour(t as usize, i) {
                if seen.insert(n as u32) {
                    stack.push(n as u32);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Work toward removing edge `(p, q)`, returning whether anything moved.
///
/// Removing it outright is the direct route. When no re-cut of its link
/// works — the link is too long, or every triangulation of it would fold —
/// the fan is thinned instead: a 2-3 flip on a face that holds the whole
/// edge trades the two cells carrying it there for one, so each such flip
/// shortens the fan by one and walks it toward a shape that can be undone.
fn retire_edge(mesh: &mut TetMesh, p: u32, q: u32) -> Result<bool> {
    if remove_edge(mesh, p, q)?.is_some() {
        return Ok(true);
    }
    for t in mesh.tets_with_edge(p, q) {
        let cell = mesh.tet(t as usize).expect("live fan cell");
        for i in 0..4 {
            // The face opposite `p` or `q` does not hold the edge at all.
            if cell[i] == p || cell[i] == q {
                continue;
            }
            if flip23(mesh, t as usize, i)?.is_some() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

// ─── Facets ─────────────────────────────────────────────────────────────

/// Flip away the edges piercing `f` until the facet appears.
fn recover_facet(mesh: &mut TetMesh, f: &[u32; 3]) -> Result<()> {
    for _ in 0..FLIP_BUDGET {
        if mesh.has_face(f) {
            return Ok(());
        }
        let Some((p, q)) = piercing_edge(mesh, f) else {
            return Ok(());
        };
        if !retire_edge(mesh, p, q)? {
            return Ok(()); // caller reports the failure
        }
    }
    Ok(())
}

/// An edge of the mesh crossing the interior of triangle `f`.
fn piercing_edge(mesh: &TetMesh, f: &[u32; 3]) -> Option<(u32, u32)> {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut candidates: Vec<u32> = Vec::new();
    for &v in f {
        for t in mesh.tets_around_vertex(v) {
            if seen.insert(t) {
                candidates.push(t);
            }
        }
    }
    // One layer further out: a piercing edge need not touch the facet's own
    // vertices.
    for i in 0..candidates.len() {
        let t = candidates[i];
        for k in 0..4 {
            if let Some(n) = mesh.neighbour(t as usize, k) {
                if seen.insert(n as u32) {
                    candidates.push(n as u32);
                }
            }
        }
    }

    for t in candidates {
        let cell = mesh.tet(t as usize)?;
        for a in 0..4 {
            for b in a + 1..4 {
                let (p, q) = (cell[a], cell[b]);
                if f.contains(&p) || f.contains(&q) {
                    continue;
                }
                if pierces_triangle_between(mesh, p, q, f) {
                    return Some((p, q));
                }
            }
        }
    }
    None
}

// ─── Exact geometric questions ──────────────────────────────────────────

/// Whether the open segment `(p, q)` crosses the interior of triangle `f`,
/// with its two ends strictly on opposite sides.
fn pierces_triangle_between(mesh: &TetMesh, p: u32, q: u32, f: &[u32; 3]) -> bool {
    let pts = mesh.points();
    let side = |x: u32| {
        orient3d(
            &pts[f[0] as usize],
            &pts[f[1] as usize],
            &pts[f[2] as usize],
            &pts[x as usize],
        )
    };
    let (sp, sq) = (side(p), side(q));
    if !((sp > 0.0 && sq < 0.0) || (sp < 0.0 && sq > 0.0)) {
        return false;
    }
    line_through_triangle(mesh, p, q, f)
}

/// Whether the line through `a` and `b` meets triangle `f` strictly inside.
fn line_through_triangle(mesh: &TetMesh, a: u32, b: u32, f: &[u32; 3]) -> bool {
    let p = mesh.points();
    let side = |i: usize, j: usize| {
        orient3d(
            &p[a as usize],
            &p[b as usize],
            &p[f[i] as usize],
            &p[f[j] as usize],
        )
    };
    let s = [side(0, 1), side(1, 2), side(2, 0)];
    (s.iter().all(|&x| x > 0.0)) || (s.iter().all(|&x| x < 0.0))
}

/// Whether the segments `(u, v)` and `(p, q)` genuinely cross — coplanar and
/// meeting at a point interior to both.
fn segments_cross(mesh: &TetMesh, u: u32, v: u32, p: u32, q: u32) -> bool {
    let pts = mesh.points();
    let at = |x: u32| &pts[x as usize];
    // Coplanar, or they miss each other in space.
    if orient3d(at(u), at(v), at(p), at(q)) != 0.0 {
        return false;
    }
    // In their common plane, each segment must separate the other's ends.
    // A fourth point off the plane turns each side test into an orient3d.
    let n = plane_probe(at(u), at(v), at(p));
    let straddles = |a: u32, b: u32, c: u32, d: u32| {
        let sc = orient3d(at(a), at(b), &n, at(c));
        let sd = orient3d(at(a), at(b), &n, at(d));
        (sc > 0.0 && sd < 0.0) || (sc < 0.0 && sd > 0.0)
    };
    straddles(u, v, p, q) && straddles(p, q, u, v)
}

/// A point off the plane of `a`, `b`, `c`, used as the apex that turns a
/// planar side test into a signed volume.
fn plane_probe(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> [f64; 3] {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    [a[0] + n[0], a[1] + n[1], a[2] + n[2]]
}
