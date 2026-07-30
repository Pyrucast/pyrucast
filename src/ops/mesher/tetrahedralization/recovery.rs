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

use std::collections::{HashMap, HashSet};

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;

use super::delaunay::TetMesh;
use super::envelope::Envelope;
use super::fill::{delaunay_fill, fill, Constraints, DEFAULT_BUDGET};
use super::flips::{flip23, relevant, remove_edge};
use super::intersect::{segment_hits_triangle, triangles_intersect};
use super::predicates::{orient2d, orient3d};
use super::surface::recut_flat_strip;

/// How hard recovery should try before declaring a piece stuck.
///
/// Fighting for an envelope edge is worth it only when there is no other way
/// out. A caller that has allowed the envelope to be subdivided *does* have
/// another way out, and a cheaper one: the exhaustive pocket rebuilds cost
/// far more than cutting the offending edge in two and starting again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Effort {
    /// Largest pocket a rebuild may grow to. Zero forbids widening, and with
    /// it the wholesale rebuilds, leaving only the local moves.
    pub max_region: usize,
    /// Moves spent on one edge or facet before it is declared stuck.
    pub flips: usize,
    /// Sweeps over the whole envelope before what is left is reported.
    pub passes: usize,
}

impl Effort {
    /// Everything, for a caller that must succeed here or not at all.
    pub const THOROUGH: Effort = Effort {
        max_region: 16,
        flips: 512,
        passes: 8,
    };
    /// Local moves only, and few of them, for a caller that will subdivide
    /// instead. Insisting is pointless when giving up costs one round of
    /// cutting the offending piece in two.
    pub const QUICK: Effort = Effort {
        max_region: 0,
        flips: 24,
        passes: 2,
    };
}

/// The envelope's facets, indexed for the questions recovery keeps asking of
/// them.
///
/// Recovery consults the envelope constantly — *is this face one of yours?*,
/// *does this edge belong to one of your facets?* — and answering by walking
/// the list turns the whole phase quadratic: a scan of tens of thousands of
/// facets, inside a loop, inside a loop. Indexing once costs one pass and
/// makes every answer local.
pub struct Protected<'a> {
    facets: &'a [[u32; 3]],
    by_key: HashSet<[u32; 3]>,
    by_edge: HashMap<(u32, u32), Vec<[u32; 3]>>,
    by_vertex: HashMap<u32, Vec<[u32; 3]>>,
}

impl<'a> Protected<'a> {
    pub fn new(facets: &'a [[u32; 3]]) -> Protected<'a> {
        let mut by_key = HashSet::with_capacity(facets.len());
        let mut by_edge: HashMap<(u32, u32), Vec<[u32; 3]>> = HashMap::new();
        let mut by_vertex: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
        for f in facets {
            by_key.insert(sorted(f));
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                by_edge
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_default()
                    .push(*f);
                by_vertex.entry(f[k]).or_default().push(*f);
            }
        }
        Protected {
            facets,
            by_key,
            by_edge,
            by_vertex,
        }
    }

    /// Every facet, in the order they came.
    pub fn all(&self) -> &[[u32; 3]] {
        self.facets
    }

    /// Whether `f` is one of the envelope's facets.
    pub fn holds(&self, f: &[u32; 3]) -> bool {
        self.by_key.contains(&sorted(f))
    }

    /// The facets having `(p, q)` as a side — at most two.
    pub fn on_edge(&self, p: u32, q: u32) -> &[[u32; 3]] {
        self.by_edge
            .get(&if p < q { (p, q) } else { (q, p) })
            .map_or(&[][..], |v| v.as_slice())
    }

    /// The facets having `v` as a corner.
    pub fn at(&self, v: u32) -> &[[u32; 3]] {
        self.by_vertex.get(&v).map_or(&[][..], |v| v.as_slice())
    }
}

fn sorted(f: &[u32; 3]) -> [u32; 3] {
    let mut k = *f;
    k.sort_unstable();
    k
}

/// A piece of the envelope recovery could not fit into the mesh.
///
/// A missing edge takes its facets down with it, so an edge is reported in
/// preference to the facets that depend on it: subdividing the edge is what
/// frees them all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stuck {
    /// Both ends, as local envelope indices.
    Edge(u32, u32),
    /// A facet whose three sides are present but which is still buried.
    Facet([u32; 3]),
}

/// Make every edge and every facet of `envelope` present in `mesh`.
///
/// Returns what could **not** be fitted; an empty list means the envelope is
/// wholly in the mesh. Reporting rather than failing lets the caller decide
/// what to do about it — refuse, or subdivide the offending pieces and come
/// back.
pub fn recover(
    mesh: &mut TetMesh,
    envelope: &Envelope,
    effort: Effort,
    cancel: &dyn Cancel,
) -> Result<Vec<Stuck>> {
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(3 * envelope.facets().len());
    for f in envelope.facets() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            edges.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let protect = Protected::new(envelope.facets());

    // Recovering one facet can undo another, so the envelope is swept again
    // until nothing is missing — or until a sweep gains nothing, which is
    // the signal that the rest is genuinely out of reach and that grinding
    // through more sweeps only costs time.
    let mut previously_missing = usize::MAX;
    for _ in 0..effort.passes {
        // Edges first: a facet cannot be recovered while its own sides are
        // missing.
        for &(u, v) in &edges {
            cancel.check()?;
            if !mesh.has_edge(u, v) {
                recover_edge(mesh, u, v, &protect, effort)?;
            }
        }
        for f in envelope.facets() {
            cancel.check()?;
            if !mesh.has_face(f) {
                recover_facet(mesh, f, &protect, effort)?;
            }
        }
        let missing = envelope
            .facets()
            .iter()
            .filter(|f| !mesh.has_face(f))
            .count();
        if missing == 0 {
            return Ok(Vec::new());
        }
        if missing >= previously_missing {
            break;
        }
        previously_missing = missing;
    }

    let mut stuck: Vec<Stuck> = edges
        .iter()
        .filter(|&&(u, v)| !mesh.has_edge(u, v))
        .map(|&(u, v)| Stuck::Edge(u, v))
        .collect();
    // Only facets whose sides are all in place: the others go away with the
    // edge they are waiting on.
    stuck.extend(
        envelope
            .facets()
            .iter()
            .filter(|f| !mesh.has_face(f) && (0..3).all(|k| mesh.has_edge(f[k], f[(k + 1) % 3])))
            .map(|f| Stuck::Facet(*f)),
    );
    Ok(stuck)
}

/// The diagnosis handed back when the caller will not subdivide.
pub fn describe(mesh: &TetMesh, stuck: &Stuck) -> PyrucastError {
    let (what, vertices) = match stuck {
        Stuck::Edge(u, v) => ("edge", vec![*u, *v]),
        Stuck::Facet(f) => ("facet", f.to_vec()),
    };
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
         node on the surface. Pass allow_surface_nodes to let the mesher subdivide the envelope \
         there — its shape is kept, only its facets are cut finer — or refine the surface mesh \
         yourself around that spot."
    ))
}

// ─── Edges ──────────────────────────────────────────────────────────────

/// Flip obstructions away until `(u, v)` is an edge of the mesh, or until
/// nothing applies.
fn recover_edge(
    mesh: &mut TetMesh,
    u: u32,
    v: u32,
    protect: &Protected<'_>,
    effort: Effort,
) -> Result<()> {
    for _ in 0..effort.flips {
        if mesh.has_edge(u, v) {
            return Ok(());
        }
        // Nothing left to unpick means the segment is not crossing the mesh
        // but running along a flat piece of its surface, or that every
        // obstruction is held in place by a facet already won. Re-cut the
        // surface, then fall back to rebuilding the corridor whole.
        if !clear_one_obstruction(mesh, u, v, protect, effort)?
            && !recut_flat_strip(mesh, u, v, protect, effort.max_region)?
            && !rebuild_along(mesh, u, v, protect, effort)?
        {
            return Ok(()); // caller reports the failure
        }
    }
    Ok(())
}

/// Rebuild the cells the segment runs through, this time with the edge
/// imposed.
///
/// The counterpart of [`rebuild_around`] for edges, and the last thing tried:
/// when the obstructions hold each other in place — each one only removable
/// at the cost of a facet already won — nothing local can move, and the way
/// out is to stop unpicking the pocket and rebuild it whole, with the edge
/// handed to the search as something the filling must contain.
fn rebuild_along(
    mesh: &mut TetMesh,
    u: u32,
    v: u32,
    protect: &Protected<'_>,
    effort: Effort,
) -> Result<bool> {
    if effort.max_region == 0 {
        return Ok(false);
    }
    let mut region = corridor(mesh, u, v);
    if region.is_empty() {
        return Ok(false);
    }
    loop {
        let boundary = mesh.region_boundary(&region);
        let keep = relevant(mesh, &region, protect);
        if let Some(cells) = fill(
            mesh.points(),
            &boundary,
            Constraints {
                with_faces: &keep,
                with_edges: &[(u, v)],
                ..Default::default()
            },
            DEFAULT_BUDGET,
        ) {
            if mesh
                .replace_region(&region, &cells, "an edge recovery")
                .is_ok()
            {
                return Ok(true);
            }
        }
        let wider = grow_region(mesh, &region);
        if wider.len() == region.len() || wider.len() > effort.max_region {
            return Ok(false);
        }
        region = wider;
    }
}

/// Remove one thing standing between `u` and `v`, if any move applies.
///
/// Everything the segment runs through is fair game, not just what touches
/// `u`: an envelope edge spanning a wide flat face is typically blocked far
/// from either end.
fn clear_one_obstruction(
    mesh: &mut TetMesh,
    u: u32,
    v: u32,
    protect: &Protected<'_>,
    effort: Effort,
) -> Result<bool> {
    let corridor = corridor(mesh, u, v);

    // A face the segment goes through has to open up. The 2-3 flip that
    // opens it replaces it by an edge pointing further along.
    for &t in &corridor {
        for i in 0..4 {
            let f = mesh.face(t as usize, i);
            if f.contains(&u) || f.contains(&v) {
                continue;
            }
            // A 2-3 flip destroys the face it opens; an envelope facet
            // that has already been won must not be traded away for progress
            // elsewhere.
            if protect.holds(&f) {
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
                if segments_cross(mesh, u, v, p, q) {
                    let ok = retire_edge(mesh, p, q, protect, effort)?;
                    if ok {
                        return Ok(true);
                    }
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
fn retire_edge(
    mesh: &mut TetMesh,
    p: u32,
    q: u32,
    protect: &Protected<'_>,
    effort: Effort,
) -> Result<bool> {
    // An edge that is a side of an envelope facet already in place cannot be
    // taken out without taking the facet with it. Refusing here is what keeps
    // recovery from trading one facet for its neighbour, over and over.
    if protect.on_edge(p, q).iter().any(|g| mesh.has_face(g)) {
        return Ok(false);
    }
    if remove_edge(mesh, p, q, protect, effort.max_region)?.is_some() {
        return Ok(true);
    }
    for t in mesh.tets_with_edge(p, q) {
        let cell = mesh.tet(t as usize).expect("live fan cell");
        for i in 0..4 {
            // The face opposite `p` or `q` does not hold the edge at all.
            if cell[i] == p || cell[i] == q {
                continue;
            }
            if protect.holds(&mesh.face(t as usize, i)) {
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

/// Make facet `f` appear, first by clearing what pierces it, then — if that
/// is not enough — by rebuilding the pocket it runs through around it.
fn recover_facet(
    mesh: &mut TetMesh,
    f: &[u32; 3],
    envelope_facets: &Protected<'_>,
    effort: Effort,
) -> Result<()> {
    for _ in 0..effort.flips {
        if mesh.has_face(f) {
            return Ok(());
        }
        let Some((p, q)) = piercing_edge(mesh, f) else {
            break;
        };
        if !retire_edge(mesh, p, q, envelope_facets, effort)? {
            break;
        }
    }
    if mesh.has_face(f) {
        return Ok(());
    }
    rebuild_around(mesh, f, envelope_facets, effort)
}

/// Rebuild the cells the facet runs through, this time with the facet
/// imposed.
///
/// Clearing obstructions one at a time only works while each one *can* be
/// cleared on its own. When they hold each other in place — the usual
/// situation for a facet lying in a flat plane inside the solid — the way
/// out is to stop negotiating with the existing cells and rebuild the pocket
/// wholesale, with the facet handed to the search as a wall it may not
/// cross. It then either finds a filling that has the facet, or proves there
/// is none.
///
/// This is the one place a Delaunay retriangulation of the pocket can stand
/// in for the exhaustive search, and it is worth a great deal: the search is
/// exponential and gives up at a handful of cells, while a triangulation
/// costs the same whatever the pocket's size. The facet goes in as a wall
/// with **both** its faces, so the pocket is cut in two along it and each
/// half is filled from its own side — a facet the triangulation contains is
/// a facet recovered. When the triangulation refuses, it names the face it
/// stumbled on, and taking in the cell beyond that face makes the refusal
/// impossible to repeat.
fn rebuild_around(
    mesh: &mut TetMesh,
    f: &[u32; 3],
    protect: &Protected<'_>,
    effort: Effort,
) -> Result<()> {
    if effort.max_region == 0 {
        return Ok(());
    }
    let mut region = cells_meeting(mesh, f);
    if region.is_empty() {
        return Ok(());
    }
    loop {
        let boundary = mesh.region_boundary(&region);
        let mut want = relevant(mesh, &region, protect);
        if !want.contains(f) {
            want.push(*f);
        }
        // The facet, seen from both sides: to the triangulation it is a wall
        // running through the pocket, and the flood fills up to it from
        // either side rather than across it.
        let mut split = boundary.clone();
        split.push(*f);
        split.push([f[0], f[2], f[1]]);

        let outcome = delaunay_fill(
            mesh.points(),
            &split,
            Constraints {
                with_faces: &want,
                ..Default::default()
            },
        )
        .or_else(|missing| {
            fill(
                mesh.points(),
                &boundary,
                Constraints {
                    with_faces: &want,
                    ..Default::default()
                },
                DEFAULT_BUDGET,
            )
            .ok_or(missing)
        });
        let missing = match outcome {
            Ok(cells) => {
                if mesh
                    .replace_region(&region, &cells, "a facet recovery")
                    .is_ok()
                {
                    return Ok(());
                }
                Vec::new()
            }
            Err(missing) => missing,
        };

        // Widening across what the triangulation stumbled on, when it said;
        // a whole extra layer when it did not.
        let mut wider = region.clone();
        if !grow_across(mesh, &mut wider, &missing) {
            wider = grow_region(mesh, &region);
        }
        if wider.len() == region.len() || wider.len() > effort.max_region {
            return Ok(()); // caller reports the failure
        }
        region = wider;
    }
}

/// Take in the cells lying beyond `faces`, and say whether anything moved.
///
/// A face with nothing beyond it is on the outer surface of the mesh, so
/// there is nothing to swallow and the pocket cannot be widened there.
fn grow_across(mesh: &TetMesh, region: &mut Vec<u32>, faces: &[[u32; 3]]) -> bool {
    let mut held: HashSet<u32> = region.iter().copied().collect();
    let mut grew = false;
    for f in faces {
        let Some(owners) = mesh.face_owners(f) else {
            continue;
        };
        for (t, _) in owners {
            if held.insert(t) {
                region.push(t);
                grew = true;
            }
        }
    }
    grew
}

/// Cells whose closed shape meets the triangle `f`.
fn cells_meeting(mesh: &TetMesh, f: &[u32; 3]) -> Vec<u32> {
    let pts = mesh.points();
    let tri = [
        &pts[f[0] as usize],
        &pts[f[1] as usize],
        &pts[f[2] as usize],
    ];
    let meets = |t: u32| -> bool {
        let cell = mesh.tet(t as usize).expect("live cell");
        if f.iter().any(|x| cell.contains(x)) {
            return true;
        }
        (0..4).any(|i| {
            let g = mesh.face(t as usize, i);
            triangles_intersect(
                &tri,
                &[
                    &pts[g[0] as usize],
                    &pts[g[1] as usize],
                    &pts[g[2] as usize],
                ],
            )
        })
    };

    let mut seen: HashSet<u32> = HashSet::new();
    let mut out: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = f
        .iter()
        .flat_map(|&v| mesh.tets_around_vertex(v))
        .filter(|&t| seen.insert(t))
        .collect();
    while let Some(t) = stack.pop() {
        if !meets(t) {
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

/// The region plus every cell touching it through a face.
fn grow_region(mesh: &TetMesh, region: &[u32]) -> Vec<u32> {
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

/// An edge of the mesh crossing the interior of triangle `f`.
fn piercing_edge(mesh: &TetMesh, f: &[u32; 3]) -> Option<(u32, u32)> {
    let mut seen: HashSet<u32> = HashSet::new();
    let mut candidates: Vec<u32> = Vec::new();
    #[allow(clippy::needless_range_loop)]
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

    for &t in &candidates {
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
    // Nothing goes through the facet from one side to the other; look again
    // for edges lying flat *in* its plane and crossing it there, which block
    // it just as surely.
    for &t in &candidates {
        let cell = mesh.tet(t as usize)?;
        for a in 0..4 {
            for b in a + 1..4 {
                let (p, q) = (cell[a], cell[b]);
                if f.contains(&p) && f.contains(&q) {
                    continue;
                }
                if crosses_in_plane(mesh, p, q, f) {
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

/// Whether the segment `(p, q)` lies in the plane of `f` and crosses it
/// there.
///
/// A facet can be blocked without anything going through it: an edge flat in
/// its own plane, cutting across it, keeps it out of the mesh just the same.
fn crosses_in_plane(mesh: &TetMesh, p: u32, q: u32, f: &[u32; 3]) -> bool {
    let pts = mesh.points();
    let side = |x: u32| {
        orient3d(
            &pts[f[0] as usize],
            &pts[f[1] as usize],
            &pts[f[2] as usize],
            &pts[x as usize],
        )
    };
    if side(p) != 0.0 || side(q) != 0.0 {
        return false;
    }

    // Drop the axis the facet's normal leans on, so its projection stays a
    // proper triangle.
    let at = |x: u32| pts[x as usize];
    let (a, b, c) = (at(f[0]), at(f[1]), at(f[2]));
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let drop = (0..3)
        .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .expect("three axes");
    let (i, j) = match drop {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let flat = |x: u32| [pts[x as usize][i], pts[x as usize][j]];
    let tri = [flat(f[0]), flat(f[1]), flat(f[2])];
    let (pp, qq) = (flat(p), flat(q));

    let strictly_inside = |x: &[f64; 2]| {
        let s = [
            orient2d(&tri[0], &tri[1], x),
            orient2d(&tri[1], &tri[2], x),
            orient2d(&tri[2], &tri[0], x),
        ];
        s.iter().all(|&v| v > 0.0) || s.iter().all(|&v| v < 0.0)
    };
    if strictly_inside(&pp) || strictly_inside(&qq) {
        return true;
    }
    // Or it properly cuts one of the facet's own sides.
    (0..3).any(|k| {
        let (e0, e1) = (&tri[k], &tri[(k + 1) % 3]);
        let (d0, d1) = (orient2d(e0, e1, &pp), orient2d(e0, e1, &qq));
        let (d2, d3) = (orient2d(&pp, &qq, e0), orient2d(&pp, &qq, e1));
        d0 * d1 < 0.0 && d2 * d3 < 0.0
    })
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
