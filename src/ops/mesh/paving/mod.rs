//! Frontal paving: the machinery behind [`pave_surface`](fn@super::pave_surface).
//!
//! The front starts as the domain's boundary and walks inward, laying a whole
//! row of quadrangles at a time ([`row`]) until each loop is small enough to
//! close ([`close`]). Two things keep that from going wrong:
//!
//! - **the invariant.** The front is always a set of simple, pairwise disjoint
//!   loops, each still turning the way it was born — material on its left.
//!   Nothing is committed that would break it: a row whose quadrangles are not
//!   all strictly convex, whose new edges would cross the front, or that would
//!   leave its loop reversed, is refused and retried closer in — and neither is
//!   the relaxation that follows it, which is put back whole if it leaves the
//!   loop crossing itself. Every such decision goes through the exact predicate
//!   in [`geom`], so it is a fact and not an estimate.
//! - **the way out.** When two parts of the front come within touching
//!   distance they are seamed together ([`Front::merge`](front::Front::merge)),
//!   which splits a loop in two where the domain is concave and joins two loops
//!   into one where a hole is being swallowed. Holes therefore need no special
//!   handling anywhere in this module.
//!
//! Paving can stall — a loop that neither advances nor seams. That is not
//! treated as a failure: after a few stalled turns the loop is handed to
//! [`close`], which fills any simple polygon. The paver can degrade, and what
//! it degrades to is quality, never validity — nothing leaves here holding a
//! cell that is turned inside out or flat, since such a cell has a negative
//! Jacobian and no solver can integrate it. The last word belongs to
//! [`reject_cells_turned_the_wrong_way`](crate::ops::mesh::triangulation::reject_cells_turned_the_wrong_way),
//! which refuses the fabric outright rather than hand one back. It is a last
//! resort and not a way out: every guard upstream exists so that it never
//! fires, and where a valid mesh can be laid it is laid.

pub mod cleanup;
pub mod close;
pub mod front;
pub mod geom;
pub mod grid;
pub mod grid2;
pub mod proximity;
pub mod row;
pub mod smooth;

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, Node, NodeId, Point2};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::interrupt::Cancel;
use crate::ops::mesh::contour::{Contour, Domain};
use front::Front;
use geom::segments_cross;
use proximity::EdgeGrid;
use row::{Corner, RowPlan};
use std::collections::HashMap;

/// Loops of this size or smaller are closed rather than advanced.
const CLOSE_AT: usize = 6;

/// Factor a refused row's advance is multiplied by before retrying.
const RETREAT: f64 = 0.55;

/// Softer retreat applied to the neighbours of a blamed slot.
const RETREAT_NEIGHBOUR: f64 = 0.8;

/// How many times a row is retried, shorter each time, before giving up.
const RETREAT_STEPS: usize = 8;

/// Smallest advance, in units of the target size, a row may still be laid at.
/// Below it the row would be a layer of slivers, and the loop is better served
/// by a seam or a closure. Measured: raising it further starts costing area
/// coverage on a concave boundary.
const RETREAT_FLOOR: f64 = 0.5;

/// Two front nodes closer than this many target sizes are seamed together.
const SEAM_FACTOR: f64 = 0.72;

/// How many times `unstick` doubles its search radius before giving up.
const CHORD_RADIUS_STEPS: usize = 4;

/// How many of the shortest chords `unstick` checks for clearance per radius.
const CHORD_CANDIDATES: usize = 8;

/// Stalled turns tolerated on one loop before it is closed outright.
const MAX_STALL: u32 = 2;

/// Widest fold, in units of the target size, written off as two fronts
/// grazing rather than reported as a region that could not be meshed.
const FOLD_TOLERANCE: f64 = 0.5;

/// Smoothing sweeps run over the finished mesh.
const FINAL_SWEEPS: usize = 12;

/// Relaxation passes run along a freshly advanced front.
const FRONT_SWEEPS: usize = 4;

/// Weight the front relaxation gives to a node's own position.
const FRONT_RELAX: f64 = 0.5;

/// The mesh a paved domain produced, in the domain's local 2-D frame.
pub struct Fabric {
    pub pts: Vec<Point2>,
    /// `false` for a node that must keep its position: the contour's, and the
    /// grid core's, which are square by construction and have nothing to gain
    /// from being smoothed.
    pub movable: Vec<bool>,
    /// `true` for a node nothing may **discard** — the caller's contour, and
    /// only it. Distinct from `movable` on purpose: a core node must not be
    /// moved, but it may be given up. Conflating the two left the weld unable
    /// to close a slither made of core nodes, and a slither that cannot be
    /// welded is a crack in the mesh.
    pinned: Vec<bool>,
    pub quads: Vec<[u32; 4]>,
    /// `true` for a quadrangle a collapse has retired. The slot stays, so
    /// `incident` keeps indexing `quads` as it always did; the corpses are
    /// swept once paving is over.
    dead: Vec<bool>,
    pub tris: Vec<[u32; 3]>,
    /// Store identity of the contour nodes, in the order they occupy the first
    /// entries of `pts`. Shorter than `pts`: everything past it is new.
    pub contour_ids: Vec<NodeId>,
    /// Quadrangles touching each vertex — needed by the seam, which rewrites
    /// connectivity rather than adding geometry.
    incident: Vec<Vec<u32>>,
}

impl Fabric {
    fn add(&mut self, p: Point2, movable: bool, pinned: bool) -> u32 {
        self.pts.push(p);
        self.movable.push(movable);
        self.pinned.push(pinned);
        self.incident.push(Vec::new());
        (self.pts.len() - 1) as u32
    }

    fn push_quad(&mut self, q: [u32; 4]) {
        let i = self.quads.len() as u32;
        self.quads.push(q);
        self.dead.push(false);
        for &c in &q {
            self.incident[c as usize].push(i);
        }
    }

    /// Retire a quadrangle: it leaves every incidence list at once, so no
    /// later step can reach it, and its slot is emptied by `bury_dead_quads`.
    fn kill_quad(&mut self, qi: u32) {
        self.dead[qi as usize] = true;
        for c in self.quads[qi as usize] {
            self.incident[c as usize].retain(|&x| x != qi);
        }
    }
}

/// Drop the quadrangles a collapse retired, once nothing indexes them any
/// more. Called at the end of paving: until then the slots must stay put,
/// because `incident` addresses `quads` by index.
fn bury_dead_quads(fab: &mut Fabric) {
    if !fab.dead.iter().any(|&d| d) {
        return;
    }
    let mut kept = Vec::with_capacity(fab.quads.len());
    for (i, q) in fab.quads.iter().enumerate() {
        if !fab.dead[i] {
            kept.push(*q);
        }
    }
    fab.quads = kept;
    fab.dead.clear();
    fab.incident.clear();
}

/// Pave one domain: its outer loop, minus its holes.
///
/// `target` is the wanted element size; `None` takes the mean boundary edge
/// length. With `all_quad`, a loop having an odd number of segments gets one
/// extra node on its longest segment — the only way to reach a triangle-free
/// mesh, since parity is a property of the boundary that paving cannot change.
pub fn pave(
    domain: &Domain,
    target: Option<f64>,
    all_quad: bool,
    cancel: &dyn Cancel,
    op: &str,
) -> Result<Fabric> {
    pave_inner(domain, target, all_quad, None, cancel, op)
}

/// Pave `domain` around a structured core: a tensor grid ([`grid`]) fills the
/// inside, and the front paves only what the grid could not reach.
///
/// The grid shares the contour's nodes wherever a grid node lands on one, so
/// the two boundaries have edges in common; those cancel, and what is left is
/// the band. On a rectilinear domain laid out for the grid, nothing is left.
///
/// A leftover loop that belongs entirely to the core is **frozen**: it stays
/// live, so the front sees it, keeps clear of it and seams onto it, but lays
/// no row of its own. That is the whole difference from treating the core as
/// an ordinary hole. With two live fronts the collision line floats wherever
/// the two happen to meet, and it is that line that carries the valence
/// defects; with one, the front *lands*, on an interface that was chosen.
///
/// `band` is extra clearance in cells — see [`grid_surface`](fn@super::grid_surface).
pub fn pave_grid(
    domain: &Domain,
    target: Option<f64>,
    all_quad: bool,
    band: usize,
    cancel: &dyn Cancel,
    op: &str,
) -> Result<Fabric> {
    pave_inner(domain, target, all_quad, Some((band, false)), cancel, op)
}

/// Like [`pave_grid`], but with the core built by [`grid2`] — lines one per
/// contour node, bands too thin collapsed, rows free to bend.
pub fn pave_grid2(
    domain: &Domain,
    target: Option<f64>,
    all_quad: bool,
    band: usize,
    cancel: &dyn Cancel,
    op: &str,
) -> Result<Fabric> {
    pave_inner(domain, target, all_quad, Some((band, true)), cancel, op)
}

fn pave_inner(
    domain: &Domain,
    target: Option<f64>,
    all_quad: bool,
    band: Option<(usize, bool)>,
    cancel: &dyn Cancel,
    op: &str,
) -> Result<Fabric> {
    let mut fab = Fabric {
        pts: Vec::new(),
        movable: Vec::new(),
        pinned: Vec::new(),
        quads: Vec::new(),
        dead: Vec::new(),
        tris: Vec::new(),
        contour_ids: Vec::new(),
        incident: Vec::new(),
    };

    // ── Seed the front with the contour ───────────────────────────────────
    let mut loops: Vec<Vec<u32>> = Vec::new();
    let mut perimeter = 0.0;
    let mut segments = 0usize;
    for l in std::iter::once(&domain.outer).chain(&domain.holes) {
        let mut verts = Vec::with_capacity(l.pts.len());
        for (k, p) in l.pts.iter().enumerate() {
            fab.contour_ids.push(l.node_ids[k]);
            verts.push(fab.add(*p, false, true));
        }
        let n = l.pts.len();
        for i in 0..n {
            perimeter += (l.pts[(i + 1) % n] - l.pts[i]).norm();
        }
        segments += n;
        loops.push(verts);
    }
    let target = match target {
        Some(t) => t,
        None => perimeter / segments.max(1) as f64,
    };

    // ── Parity, settled once and for all at the entrance ──────────────────
    // The contour is the caller's and is never touched, so an odd loop cannot
    // be fixed here: it simply has no triangle-free filling, and saying so is
    // more useful than quietly producing one triangle anyway.
    if all_quad {
        for (k, verts) in loops.iter().enumerate() {
            if !verts.len().is_multiple_of(2) {
                let which = if k == 0 { "outer boundary" } else { "hole" };
                return Err(PyrucastError::Message(format!(
                    "{op}: all_quad was asked for, but the {which} loop has {} \
                     segments — an odd number. A polygon with an odd number of sides has no \
                     filling by quadrangles alone, and paving cannot change that parity. \
                     Re-mesh that loop with an even number of segments.",
                    verts.len()
                )));
            }
        }
    }

    // ── The structured core, when one was asked for ───────────────────────
    // It writes its cells straight into the fabric and hands back the loops of
    // what is left to fill: the contour, the core's boundary, and neither
    // where the two meet. A loop that is entirely the core's is *frozen* — it
    // bounds the band on the inside and is landed on, never advanced from.
    // Anything else, including a loop that runs partly along each, advances
    // normally.
    let seeds: Vec<(Vec<u32>, bool)> = match band {
        None => loops.iter().map(|l| (l.clone(), false)).collect(),
        Some((band, false)) => grid::build(&mut fab, domain, &loops, target, band).band,
        Some((band, true)) => grid2::build(&mut fab, domain, &loops, target, band).band,
    };

    let mut front = Front::new();
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for (verts, frozen) in &seeds {
        if verts.len() < 3 {
            continue;
        }
        if *frozen {
            front.add_frozen_loop(verts);
        } else {
            stack.push((front.add_loop(verts), 0));
        }
    }

    // ── Advance ───────────────────────────────────────────────────────────
    // Generous, but finite: a loop that stops making progress gets closed, so
    // the cap only ever catches pathological input.
    let cap = 64 * segments + 4096;
    let mut steps = 0usize;
    while let Some((rep, stalls)) = stack.pop() {
        if !front.is_alive(rep) {
            continue;
        }
        cancel.check()?;
        steps += 1;

        let n = front.loop_len(rep);
        if n < 3 {
            front.kill_loop(rep);
            continue;
        }
        if n <= CLOSE_AT || steps > cap {
            close_loop(&mut fab, &mut front, rep, target, op)?;
            continue;
        }
        if stalls > MAX_STALL {
            // Cutting the loop in two usually gets it moving again; closing it
            // outright would fill a large polygon with a decomposition, which
            // is a far worse mesh than two more rows would have been.
            if !unstick(&fab, &mut front, rep, target, all_quad, &mut stack) {
                close_loop(&mut fab, &mut front, rep, target, op)?;
            }
            continue;
        }

        match try_row(&mut fab, &mut front, rep, target, all_quad) {
            // The row consumed the loop outright.
            Some(None) => {}
            Some(Some(new_rep)) => {
                let seamed = match find_seam(&front, &fab, new_rep, target, all_quad) {
                    Some((a, b)) => seam(&mut fab, &mut front, a, b, &mut stack),
                    None => false,
                };
                if !seamed {
                    stack.push((new_rep, 0));
                }
            }
            None => {
                let seamed = match find_seam(&front, &fab, rep, target, all_quad) {
                    Some((a, b)) => seam(&mut fab, &mut front, a, b, &mut stack),
                    None => false,
                };
                if !seamed {
                    stack.push((rep, stalls + 1));
                }
            }
        }
    }

    // ── Finish ────────────────────────────────────────────────────────────
    // Connectivity first, positions after: smoothing a node that has the
    // wrong number of cells around it only spreads the error over its
    // neighbours, so the cleanup has to come first for the sweep to have
    // something worth polishing.
    bury_dead_quads(&mut fab);
    cleanup::run(&fab.pts, &fab.movable, &mut fab.quads, &fab.tris);
    compact(&mut fab);

    let patch = smooth::Patch {
        quads: &fab.quads,
        tris: &fab.tris,
        movable: &fab.movable,
    };
    let inc = smooth::Incidence::build(&patch, fab.pts.len());
    let mut pts = std::mem::take(&mut fab.pts);
    smooth::smooth(&mut pts, &patch, &inc, None, FINAL_SWEEPS);
    fab.pts = pts;

    all_cells_are_the_right_way_round(&fab, op)?;
    Ok(fab)
}

/// Refuse a fabric holding a cell that is not strictly the right way round.
///
/// The last word on what leaves this module: every route in has its own guard,
/// exact and local, and this is the net under all of them. What it catches is a
/// guard that let something through — see
/// [`reject_cells_turned_the_wrong_way`](crate::ops::mesh::triangulation::reject_cells_turned_the_wrong_way)
/// for why nothing may leave without it.
fn all_cells_are_the_right_way_round(fab: &Fabric, op: &str) -> Result<()> {
    crate::ops::mesh::triangulation::reject_cells_turned_the_wrong_way(
        &fab.pts,
        || {
            fab.quads
                .iter()
                .map(|q| q.as_slice())
                .chain(fab.tris.iter().map(|t| t.as_slice()))
        },
        op,
    )
}

/// Turn the per-domain fabrics into a `Mesh` on the contour's own `Coords`.
///
/// `op` names the calling operator and only ever appears in the error.
pub fn materialize(parsed: &Contour, fabrics: Vec<Fabric>, op: &str) -> Result<Mesh> {
    let coords = &parsed.coords;
    let mut quad_sub: Option<SubMesh> = None;
    let mut tri_sub: Option<SubMesh> = None;
    let mut kept: Vec<Node> = Vec::new();

    for fab in fabrics {
        let mut flat: Vec<NodeId> = fab.contour_ids.clone();
        for p in &fab.pts[fab.contour_ids.len()..] {
            let node = Node::create_in(coords.clone(), &parsed.frame.to_world(*p, parsed.dim))?;
            flat.push(node.id());
            kept.push(node);
        }
        if !fab.quads.is_empty() {
            let sub =
                quad_sub.get_or_insert_with(|| SubMesh::new(coords.clone(), ElementType::QUA4));
            for q in &fab.quads {
                sub.add_cell(&[
                    flat[q[0] as usize],
                    flat[q[1] as usize],
                    flat[q[2] as usize],
                    flat[q[3] as usize],
                ])?;
            }
        }
        if !fab.tris.is_empty() {
            let sub =
                tri_sub.get_or_insert_with(|| SubMesh::new(coords.clone(), ElementType::TRI3));
            for t in &fab.tris {
                sub.add_cell(&[
                    flat[t[0] as usize],
                    flat[t[1] as usize],
                    flat[t[2] as usize],
                ])?;
            }
        }
    }

    let mut mesh = Mesh::empty();
    if let Some(q) = quad_sub
        && q.cell_count() > 0
    {
        mesh.add_sub(Handle::new(q))?;
    }
    if let Some(t) = tri_sub
        && t.cell_count() > 0
    {
        mesh.add_sub(Handle::new(t))?;
    }
    drop(kept);
    if mesh.is_empty() {
        return Err(PyrucastError::Message(format!("{op}: produced no cell")));
    }
    Ok(mesh)
}

/// Drop the nodes no cell refers to any more — the ones a doublet removal
/// took out — and renumber the rest. Contour nodes are kept whatever happens:
/// they are the caller's, and they occupy the first entries by construction.
fn compact(fab: &mut Fabric) {
    let n_contour = fab.contour_ids.len();
    let mut used = vec![false; fab.pts.len()];
    used[..n_contour].fill(true);
    for q in &fab.quads {
        for &v in q {
            used[v as usize] = true;
        }
    }
    for t in &fab.tris {
        for &v in t {
            used[v as usize] = true;
        }
    }
    if used.iter().all(|&u| u) {
        return;
    }
    let mut remap = vec![u32::MAX; fab.pts.len()];
    let (mut pts, mut movable, mut pinned) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..fab.pts.len() {
        if used[i] {
            remap[i] = pts.len() as u32;
            pts.push(fab.pts[i]);
            movable.push(fab.movable[i]);
            pinned.push(fab.pinned[i]);
        }
    }
    for q in fab.quads.iter_mut() {
        *q = q.map(|v| remap[v as usize]);
    }
    for t in fab.tris.iter_mut() {
        *t = t.map(|v| remap[v as usize]);
    }
    fab.pts = pts;
    fab.movable = movable;
    fab.pinned = pinned;
    fab.incident.clear();
}

/// Attempt one row on the loop at `rep`, retreating where the geometry refuses
/// it.
///
/// `Some(Some(r))` advanced the loop to `r`, `Some(None)` finished it off, and
/// `None` means the row could not be laid at all.
fn try_row(
    fab: &mut Fabric,
    front: &mut Front,
    rep: u32,
    target: f64,
    all_quad: bool,
) -> Option<Option<u32>> {
    let slots = front.loop_slots(rep);
    let n = slots.len();
    let p: Vec<Point2> = slots
        .iter()
        .map(|&s| fab.pts[front.vertex(s) as usize])
        .collect();
    // The advance is the target size, and **not** the front's own spacing.
    //
    // Following the spacing is the tempting rule and it is the wrong one: a
    // contracting front has spacing under the target, so a row that advances by
    // it lays cells narrower than they are tall, and the next row inherits the
    // narrower spacing. The elongation compounds inward. Blending the two
    // halves the effect without curing it — measured on a crenellated profile,
    // a half-and-half blend gives a median aspect of 1.68 against 1.36 for the
    // flat target, 475 cells against 407, and a worst Jacobian of 0.333
    // against 0.409. Every shape tried came out with fewer cells and a better
    // worst cell.
    //
    // What brings the spacing back to the target is not the advance but the
    // front's own repertoire: [`row::REFINE_RATIO`] splits an edge that has
    // grown too long, and the seam removes two slots where it has grown too
    // short.
    let mut base: Vec<f64> = vec![target; n];
    aim_at_frozen(front, fab, &p, &mut base, target);
    let grid = EdgeGrid::build(front, &fab.pts, target);
    let was = crate::ops::mesh::triangulation::signed_area(&p);

    let mut scale = vec![1.0f64; n];
    for _ in 0..RETREAT_STEPS {
        // A row laid at a fraction of the size asked for is not an advance, it
        // is a layer of slivers. The retreat divides the advance without any
        // limit of its own — eight steps at 55 % reach 1.5 % of what was asked,
        // and on a circle 88 % of the rows used to be laid below a fifth. Once
        // every node has been pushed under the floor the row is given up, and
        // the loop is seamed or closed instead, which costs nothing like as
        // much: on that circle the worst Jacobian went from 0.188 to 0.308, the
        // size ratio from 9.6 to 2.2, and the four holes it used to leave
        // disappeared with the slivers that caused them.
        if (0..n).all(|i| base[i] * scale[i] < RETREAT_FLOOR * target) {
            return None;
        }
        let sz = |i: usize| base[i] * scale[i];
        let want = |i: usize| base[i];
        match row::plan(front, &fab.pts, rep, &sz, &want, all_quad) {
            Ok(plan) => {
                if keeps_orientation(&plan, was) && chain_is_free(front, fab, &grid, &plan) {
                    let out = commit(fab, front, rep, plan);
                    if let Some(new_rep) = out {
                        relax_front(fab, front, new_rep);
                    }
                    return Some(out);
                }
                // A collision says the row went too far, but not where, so
                // everything steps back together.
                for s in scale.iter_mut() {
                    *s *= RETREAT;
                }
            }
            Err(blame) => {
                if blame.is_empty() {
                    return None;
                }
                for i in blame {
                    scale[i] *= RETREAT;
                    scale[(i + 1) % n] *= RETREAT_NEIGHBOUR;
                    scale[(i + n - 1) % n] *= RETREAT_NEIGHBOUR;
                }
            }
        }
    }
    None
}

/// How far ahead a frozen loop still counts as the thing being aimed at.
const AIM_RANGE: f64 = 4.5;

/// Shorten the advance so the front *lands* on a frozen loop instead of
/// creeping up to it.
///
/// Without this, a front facing a structured core two cells away asks for a
/// full-size row, which would cross the core; the row is refused, retried at
/// 55 % of the distance, refused again, and the band ends up filled with three
/// or four squashed rows where two square ones belonged. The retreat is a
/// reaction to a collision, and it has no idea a collision is *coming*.
///
/// So each front node that can see a frozen loop within a few cells divides
/// the distance to it into the whole number of rows that comes closest to the
/// target size, and asks for exactly that. The last of those rows arrives on
/// the core, where the seam takes over. A node that sees no frozen loop keeps
/// the advance it had.
fn aim_at_frozen(front: &Front, fab: &Fabric, p: &[Point2], base: &mut [f64], target: f64) {
    let frozen: Vec<Point2> = front
        .frozen_slots()
        .map(|s| fab.pts[front.vertex(s) as usize])
        .collect();
    if frozen.is_empty() {
        return;
    }
    let reach = AIM_RANGE * target;
    let cells = proximity::PointGrid::build(&frozen, target);
    for (i, b) in base.iter_mut().enumerate() {
        if let Some(d) = cells.nearest_within(&frozen, p[i], reach) {
            let rows = (d / target).round().max(1.0);
            *b = (d / rows).clamp(0.5 * target, 2.0 * target);
        }
    }
}

/// Does the advanced loop still turn the same way as the one it replaces?
///
/// A row *consumes* material, so it can only ever shrink the region its loop
/// bounds — never turn it inside out. The two are told apart by the sign of
/// the signed area, and not by its magnitude: an outer loop shrinks toward
/// zero from above, a hole's front grows away from zero from below, and both
/// keep their sign for as long as they live.
///
/// The row that breaks this is the one laid on a ring already thinner than the
/// cell it wants to lay — two fronts that met head-on and left a slither
/// between them. Its quadrangles pass every local test, the chain crosses
/// nothing, and the loop comes out reversed. From then on each further row
/// *inflates* it: it leaves the material, balloons, and drags every seam it
/// meets out with it, surfacing much later as a fold nowhere near where it
/// began. Refusing the row here leaves the slither for [`close_loop`], which
/// fills or welds it in place.
fn keeps_orientation(plan: &RowPlan, was: f64) -> bool {
    if plan.chain.len() < 3 {
        // The loop closed on itself; there is no successor to orient.
        return true;
    }
    let ring: Vec<Point2> = plan.chain.iter().map(|&i| plan.pts[i as usize]).collect();
    let now = crate::ops::mesh::triangulation::signed_area(&ring);
    now == 0.0 || was == 0.0 || now.is_sign_positive() == was.is_sign_positive()
}

/// Would the advanced front still be a set of simple, disjoint loops?
///
/// The new chain has to clear the whole live front — including the loop it is
/// replacing, since the two bound the strip being filled and must not touch.
fn chain_is_free(front: &Front, fab: &Fabric, grid: &EdgeGrid, plan: &RowPlan) -> bool {
    let m = plan.chain.len();
    if m < 3 {
        // The loop closed on itself; the quadrangles alone have to be sound,
        // and `row::plan` has already established that.
        return true;
    }
    let at = |i: u32| plan.pts[i as usize];
    for i in 0..m {
        let (a, b) = (at(plan.chain[i]), at(plan.chain[(i + 1) % m]));
        for s in grid.near_segment(a, b) {
            let c = fab.pts[front.vertex(s) as usize];
            let d = fab.pts[front.vertex(front.next(s)) as usize];
            if segments_cross(a, b, c, d) {
                return false;
            }
        }
    }
    // And against itself: an advanced front that crosses itself has left the
    // invariant, whatever it does to the rest of the front.
    let ring: Vec<Point2> = plan.chain.iter().map(|&i| at(i)).collect();
    geom::polygon_is_simple(&ring)
}

/// Write a planned row into the mesh and advance the front over it.
fn commit(fab: &mut Fabric, front: &mut Front, rep: u32, plan: RowPlan) -> Option<u32> {
    let base = fab.pts.len() as u32;
    for p in &plan.pts {
        fab.add(*p, true, false);
    }
    let map = |c: Corner| match c {
        Corner::Old(i) => i,
        Corner::New(i) => base + i,
    };
    for q in &plan.quads {
        fab.push_quad([map(q[0]), map(q[1]), map(q[2]), map(q[3])]);
    }
    let verts: Vec<u32> = plan.chain.iter().map(|&i| base + i).collect();
    if verts.len() < 3 {
        front.kill_loop(rep);
        return None;
    }
    Some(front.relink_loop(rep, &verts))
}

/// Straighten a freshly advanced front.
///
/// Without this the front keeps whatever kinks the row's bisector placement
/// left behind, and they compound: a node at 216° hands its neighbours a
/// sector no template can fill well, the row is refused, and the loop stalls.
/// Relaxing *along the front* rather than toward the mesh is what matters —
/// a node's only committed neighbours are behind it, so an ordinary Laplacian
/// would just pull the row back where it came from.
///
/// ## The front may not relax across itself
///
/// The per-node guard here weighs the quadrangles a node already carries, and
/// those are all *behind* it — nothing in it looks at the line ahead. So every
/// quadrangle can stay convex while two stretches of the front slide over each
/// other, and the invariant this module rests on is gone without a single local
/// test complaining. What comes of it comes later and elsewhere: the ring ends
/// up crossed, and a crossed ring has no filling in which every cell turns the
/// same way. On a circle of twenty sides at a fifth of its radius, that is the
/// fold the paver used to give up on, and the mesh was refused outright.
///
/// So the loop is asked, once, whether it is still simple, and if the
/// relaxation cost it that the whole thing is put back. Once for the call and
/// not once per sweep, on measurement: per sweep costs **+31 %** on a mesh the
/// front lays alone, per call **+14 %**, and the two rescue the same cases —
/// with the coarser one giving the better mesh on the circle above (0.211
/// against 0.105 on the worst cell). Where a grid does the paving it is free:
/// +0.7 % on a banded circle, +1.4 % on a plain grid, both inside the noise.
fn relax_front(fab: &mut Fabric, front: &Front, rep: u32) {
    let slots = front.loop_slots(rep);
    let n = slots.len();
    if n < 4 {
        return;
    }
    let verts: Vec<u32> = slots.iter().map(|&s| front.vertex(s)).collect();
    let before: Vec<Point2> = verts.iter().map(|&v| fab.pts[v as usize]).collect();
    let was_simple = geom::polygon_is_simple(&before);
    for _ in 0..FRONT_SWEEPS {
        let old: Vec<Point2> = verts.iter().map(|&v| fab.pts[v as usize]).collect();
        for i in 0..n {
            let v = verts[i];
            if !fab.movable[v as usize] {
                continue;
            }
            let cand = Point2::from(
                old[i].coords * FRONT_RELAX
                    + (old[(i + n - 1) % n].coords + old[(i + 1) % n].coords)
                        * ((1.0 - FRONT_RELAX) * 0.5),
            );
            let keep = fab.pts[v as usize];
            fab.pts[v as usize] = cand;
            // Every quadrangle touching the node, not merely those of the row
            // just laid: a seam rewrites older quadrangles onto a front
            // vertex, and moving it would silently invert them.
            let ok = fab.incident[v as usize].iter().all(|&qi| {
                let q = fab.quads[qi as usize];
                geom::quad_is_valid([
                    fab.pts[q[0] as usize],
                    fab.pts[q[1] as usize],
                    fab.pts[q[2] as usize],
                    fab.pts[q[3] as usize],
                ])
            });
            if !ok {
                fab.pts[v as usize] = keep;
            }
        }
    }
    // A relaxation that cost the loop its simplicity is put back whole — see
    // above for why the per-node guard cannot see it coming.
    let now: Vec<Point2> = verts.iter().map(|&v| fab.pts[v as usize]).collect();
    if was_simple && !geom::polygon_is_simple(&now) {
        for (i, &v) in verts.iter().enumerate() {
            fab.pts[v as usize] = before[i];
        }
    }
}

/// Fill what is left of a loop with elements and retire it.
fn close_loop(fab: &mut Fabric, front: &mut Front, rep: u32, target: f64, op: &str) -> Result<()> {
    let verts: Vec<u32> = front
        .loop_slots(rep)
        .iter()
        .map(|&s| front.vertex(s))
        .collect();
    front.kill_loop(rep);

    // A front loop bounds material on its left, so a closed one encloses
    // positive area and does not cross itself. Either failing means two lines
    // of front have folded over each other.
    //
    // Two lines of front meeting head-on normally leave a slither of overlap
    // between them, which is degenerate and covers nothing; discarding that is
    // right. What is not right is discarding a region that was supposed to hold
    // cells, so the two are told apart by **width**, not by area: a slither is
    // thin however long it runs, and one spanning twenty edges encloses twenty
    // times the area of one spanning a single edge without being any more of a
    // region. Mean width is `2·area/P`, so the test puts the area against half
    // the perimeter times the width tolerated.
    //
    // The sign of the area does not see every slither, which is why the loop is
    // asked whether it is *simple* as well. A fold whose two lobes are of
    // comparable size encloses almost nothing either way, and the sign left
    // over is the larger lobe's — an accident. Filling such a loop is what used
    // to leave reversed cells in the mesh: no decomposition of a self-crossing
    // polygon can have every piece turning the same way, so at least one comes
    // out with a negative Jacobian, and a circle produced two of them.
    let poly: Vec<Point2> = verts.iter().map(|&v| fab.pts[v as usize]).collect();
    let area = crate::ops::mesh::triangulation::signed_area(&poly);
    if area <= 0.0 || !geom::polygon_is_simple(&poly) {
        let perimeter: f64 = (0..poly.len())
            .map(|i| (poly[(i + 1) % poly.len()] - poly[i]).norm())
            .sum();
        if area.abs() <= FOLD_TOLERANCE * target * 0.5 * perimeter {
            // Welding the two lines shut costs nothing and leaves the mesh
            // whole; simply dropping the ring would leave a crack behind, and a
            // crack is a hole in the connectivity even when it has no area
            // worth speaking of.
            weld(fab, &verts);
            return Ok(());
        }
        if area <= 0.0 {
            // Wide *and* the wrong way round: the front turned itself inside
            // out, and the region between its two lines is one no filling can
            // reach. Dropping it silently would return a mesh with a hole in
            // it, so it is an error — and one worth locating, since it always
            // comes from a contour too coarse or too uneven for the size asked.
            let centre = poly
                .iter()
                .fold(Point2::origin(), |acc, p| acc + p.coords)
                .coords
                / poly.len() as f64;
            return Err(PyrucastError::Message(format!(
                "{op}: the advancing front folded onto itself near ({:.6}, {:.6}) in the \
                 meshing plane, leaving a region that cannot be filled. The contour there is \
                 too coarse or too uneven for the element size asked of it: discretise it \
                 closer to the target size, or ask for a larger one.",
                centre.x, centre.y
            )));
        }
        // Wide, tangled, and still holding the material it was born with: a
        // tangle over a real region is not a slither to weld nor a reversal to
        // refuse. The decomposition covers it bar the sliver or two the tangle
        // itself costs, and that is a better answer than no mesh at all.
    }
    let filled = close::close(&fab.pts, &verts);
    for p in &filled.added {
        fab.add(*p, true, false);
    }
    for q in filled.quads {
        fab.push_quad(q);
    }
    fab.tris.extend(filled.tris);
    Ok(())
}

/// Cut a stalled loop in two along the shortest admissible chord.
///
/// Returns `false` when no chord is usable, in which case the caller falls
/// back to closing the loop outright.
fn unstick(
    fab: &Fabric,
    front: &mut Front,
    rep: u32,
    target: f64,
    all_quad: bool,
    stack: &mut Vec<(u32, u32)>,
) -> bool {
    let slots = front.loop_slots(rep);
    let n = slots.len();
    if n < 8 {
        return false;
    }
    let rank: HashMap<u32, usize> = slots.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let grid = EdgeGrid::build(front, &fab.pts, target);

    // The chord that helps is a short one, across whatever neck the loop got
    // stuck on — not a diameter. Widening the search radius by steps finds it
    // through the grid, which keeps this linear in the front length; testing
    // every pair of slots is quadratic and, on a loop that stalls again and
    // again, ends up dominating the whole mesh.
    for step in 0..CHORD_RADIUS_STEPS {
        let radius = target * (1 << step) as f64;
        // Gather first, check clearance afterwards and only on the shortest
        // few. Clearance sweeps the grid along the whole chord, so on a long
        // one it touches most of the front; running it on every candidate is
        // what turns a stubborn loop into minutes of work.
        let mut cand: Vec<(f64, usize, usize)> = Vec::new();
        for (i, &sa) in slots.iter().enumerate() {
            let pa = fab.pts[front.vertex(sa) as usize];
            for sb in grid.near_point(pa, radius) {
                let Some(&j) = rank.get(&sb) else { continue };
                let gap = (j + n - i) % n;
                // Both sides must be worth paving, and — under the
                // all-quadrangle guarantee — both must stay even. A chord adds
                // one slot to each side, so an odd gap is what keeps parity.
                if gap < 3 || n - gap < 3 || (all_quad && gap.is_multiple_of(2)) {
                    continue;
                }
                let d = (fab.pts[front.vertex(sb) as usize] - pa).norm();
                if d < radius {
                    cand.push((d, i, j));
                }
            }
        }
        cand.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let best = cand
            .iter()
            .take(CHORD_CANDIDATES)
            .find(|&&(_, i, j)| seam_is_clear(front, fab, &grid, slots[i], slots[j]))
            .copied();
        if let Some((_, i, j)) = best {
            let (ra, rb) = front.split_by_chord(slots[i], slots[j]);
            stack.push((ra, 0));
            stack.push((rb, 0));
            return true;
        }
    }
    false
}

/// Sew a flattened ring shut by identifying the vertices that face each other
/// across it — the closest pair that are not already neighbours, taken again
/// and again until nothing is left to sew.
///
/// Pairing the ends against each other instead, as the ring order suggests,
/// is wrong: a flattened ring is a lens, and its two ends are the *thin* part.
///
/// ## Identifying is not always enough
///
/// A lens is sewn shut by identification alone only when its two sides carry
/// the *same number* of nodes. They often do not: a ring whose one side runs
/// `A B C` against a plain `A C` on the other has a node too many, and no pair
/// of vertices can be identified to get rid of it — `A` and `B` are two corners
/// of the same quadrangle, and merging them would fold that quadrangle onto
/// itself, which [`merge_into`] rightly refuses.
///
/// So the leftover node is **collapsed** instead: the quadrangle holding both
/// gives up the corner and becomes a triangle. That is the only move available
/// here, and it costs one square cell; leaving the ring open costs a crack —
/// a hole in the mesh, which the smoothing then opens into a visible triangular
/// gap. Correctness before shape, so the collapse wins.
fn weld(fab: &mut Fabric, ring: &[u32]) {
    let mut verts: Vec<u32> = ring.to_vec();
    while verts.len() >= 4 {
        let n = verts.len();
        let mut best: Option<(f64, usize, usize)> = None;
        for i in 0..n {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                let d = (fab.pts[verts[j] as usize] - fab.pts[verts[i] as usize]).norm();
                if best.is_none_or(|(bd, _, _)| d < bd) {
                    best = Some((d, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else { break };
        let (a, b) = (verts[i], verts[j]);
        if !merge_into(fab, a, b) && !merge_into(fab, b, a) {
            break;
        }
        verts.remove(j);
    }
    close_cracks(fab, verts);
}

/// Close whatever the identification pass left open in a retired ring.
///
/// Only edges that are genuinely open are touched — carried by one quadrangle
/// and nothing else — and never a stretch of the caller's contour, whose outer
/// side is *meant* to have nothing behind it. Everything else in the ring is
/// already sound and is left exactly as it is.
fn close_cracks(fab: &mut Fabric, mut verts: Vec<u32>) {
    // Each turn of the loop removes one vertex, so the ring bounds the work.
    for _ in 0..verts.len() {
        let n = verts.len();
        if n < 3 || !ring_has_a_crack(fab, &verts) {
            return;
        }
        // The shortest pair goes first, adjacent ones included: after the
        // identification pass it is precisely a ring *edge* that has to go.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (fab.pts[verts[j] as usize] - fab.pts[verts[i] as usize]).norm();
                pairs.push((d, i, j));
            }
        }
        pairs.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let done = pairs.iter().find_map(|&(_, i, j)| {
            let (a, b) = (verts[i], verts[j]);
            if collapse_into(fab, a, b) {
                Some(j)
            } else if collapse_into(fab, b, a) {
                Some(i)
            } else {
                None
            }
        });
        // Nothing admissible left: the ring keeps its crack rather than the
        // mesh keeping a tangle.
        let Some(gone) = done else { return };
        verts.remove(gone);
    }
}

/// Does the ring still have an edge with nothing on its far side?
fn ring_has_a_crack(fab: &Fabric, verts: &[u32]) -> bool {
    let n = verts.len();
    (0..n).any(|i| {
        let (a, b) = (verts[i], verts[(i + 1) % n]);
        // A contour edge carries one quadrangle by right: outside it is the
        // end of the domain, not a hole in it.
        if fab.pinned[a as usize] && fab.pinned[b as usize] {
            return false;
        }
        let quads = fab.incident[a as usize]
            .iter()
            .filter(|&&qi| fab.quads[qi as usize].contains(&b))
            .count();
        let tris = fab
            .tris
            .iter()
            .filter(|t| t.contains(&a) && t.contains(&b))
            .count();
        quads + tris == 1
    })
}

/// Identify `drop` into `keep`, letting a quadrangle that holds both give up a
/// corner and live on as a triangle.
///
/// The same rewrite as [`merge_into`], with one case answered differently: a
/// quadrangle holding both vertices is refused there and collapsed here. Only
/// **adjacent** corners may collapse — merging two opposite ones would pinch
/// the cell into a bowtie — and the triangle left over must turn the right way
/// like any other cell.
fn collapse_into(fab: &mut Fabric, keep: u32, drop: u32) -> bool {
    if keep == drop || fab.pinned[drop as usize] {
        return false;
    }
    let touching = fab.incident[drop as usize].clone();
    let rewrite = |q: [u32; 4]| q.map(|c| if c == drop { keep } else { c });
    let point = |v: u32| fab.pts[v as usize];
    let turns_right = |t: [u32; 3]| geom::orient(point(t[0]), point(t[1]), point(t[2])) > 0.0;

    // What each incident quadrangle becomes: `None` to collapse to a triangle.
    let mut fate: Vec<(u32, Option<[u32; 4]>)> = Vec::with_capacity(touching.len());
    for &qi in &touching {
        let q = fab.quads[qi as usize];
        let Some(d) = q.iter().position(|&c| c == drop) else {
            continue;
        };
        match q.iter().position(|&c| c == keep) {
            Some(k) => {
                if (k + 2) % 4 == d {
                    // Opposite corners: the collapse would be a bowtie.
                    return false;
                }
                if !turns_right([q[(d + 1) % 4], q[(d + 2) % 4], q[(d + 3) % 4]]) {
                    return false;
                }
                fate.push((qi, None));
            }
            None => {
                let r = rewrite(q);
                if !geom::quad_is_valid([point(r[0]), point(r[1]), point(r[2]), point(r[3])]) {
                    return false;
                }
                fate.push((qi, Some(r)));
            }
        }
    }

    // Triangles are not indexed by vertex, so they are swept for. One holding
    // both vertices would flatten, and there is no triangle left to keep.
    let mut tri_fate: Vec<(usize, [u32; 3])> = Vec::new();
    for (i, t) in fab.tris.iter().enumerate() {
        if !t.contains(&drop) {
            continue;
        }
        if t.contains(&keep) {
            return false;
        }
        let r = t.map(|c| if c == drop { keep } else { c });
        if !turns_right(r) {
            return false;
        }
        tri_fate.push((i, r));
    }

    // As in `merge_into`: no edge may come out with three cells on it. Only
    // edges reaching the surviving vertex can, so counting those is enough —
    // over every cell the operation leaves behind, triangles included, since a
    // collapse turns a diagonal into an edge.
    let mut around: HashMap<u32, usize> = HashMap::new();
    let mut count = |cell: &[u32]| {
        let k = cell.len();
        for t in 0..k {
            if cell[t] == keep {
                *around.entry(cell[(t + 1) % k]).or_insert(0) += 1;
                *around.entry(cell[(t + k - 1) % k]).or_insert(0) += 1;
            }
        }
    };
    for &(qi, r) in &fate {
        match r {
            Some(r) => count(&r),
            None => {
                let q = fab.quads[qi as usize];
                let d = q.iter().position(|&c| c == drop).unwrap();
                count(&[q[(d + 1) % 4], q[(d + 2) % 4], q[(d + 3) % 4]]);
            }
        }
    }
    for &qi in fab.incident[keep as usize].iter() {
        if !touching.contains(&qi) {
            count(&fab.quads[qi as usize]);
        }
    }
    for (i, t) in fab.tris.iter().enumerate() {
        match tri_fate.iter().find(|&&(j, _)| j == i) {
            Some(&(_, r)) => count(&r),
            None => count(t),
        }
    }
    if around.values().any(|&c| c > 2) {
        return false;
    }

    for (qi, r) in fate {
        match r {
            Some(r) => {
                fab.quads[qi as usize] = r;
                fab.incident[keep as usize].push(qi);
            }
            None => {
                let q = fab.quads[qi as usize];
                let d = q.iter().position(|&c| c == drop).unwrap();
                fab.tris
                    .push([q[(d + 1) % 4], q[(d + 2) % 4], q[(d + 3) % 4]]);
                fab.kill_quad(qi);
            }
        }
    }
    for (i, r) in tri_fate {
        fab.tris[i] = r;
    }
    fab.incident[drop as usize].clear();
    true
}

/// The closest pair of front nodes that ought to be merged, if any.
///
/// Candidates come from the whole live front, not just the loop being worked
/// on: a loop about to touch a *different* loop is exactly the case that
/// swallows a hole.
fn find_seam(
    front: &Front,
    fab: &Fabric,
    rep: u32,
    target: f64,
    all_quad: bool,
) -> Option<(u32, u32)> {
    let radius = SEAM_FACTOR * target;
    let slots = front.loop_slots(rep);
    let n = slots.len();
    let rank: HashMap<u32, usize> = slots.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let grid = EdgeGrid::build(front, &fab.pts, target);

    let mut best: Option<(f64, u32, u32)> = None;
    for (i, &a) in slots.iter().enumerate() {
        let pa = fab.pts[front.vertex(a) as usize];
        for b in grid.near_point(pa, radius) {
            if b == a || front.next(a) == b || front.next(b) == a {
                continue;
            }
            // A merge leaves two slots carrying the *same* vertex, one per
            // resulting ring — that is what a pinch point is. They sit at
            // distance zero and are not front neighbours, so they look like
            // the perfect seam candidate; merging them again would undo the
            // split and loop for ever.
            if front.vertex(a) == front.vertex(b) {
                continue;
            }
            let pb = fab.pts[front.vertex(b) as usize];
            let d = (pb - pa).norm();
            if d >= radius {
                continue;
            }
            if let Some(&j) = rank.get(&b) {
                let gap = (j + n - i) % n;
                if gap < 2 || gap > n - 2 {
                    continue;
                }
                // A split has to leave two even loops, or the all-quadrangle
                // guarantee is lost on both halves at once.
                if all_quad && gap % 2 == 1 {
                    continue;
                }
            }
            if !seam_is_clear(front, fab, &grid, a, b) {
                continue;
            }
            let key = (d, a.min(b), a.max(b));
            if best.is_none_or(|cur| key < cur) {
                best = Some(key);
            }
        }
    }
    best.map(|(_, a, b)| (a, b))
}

/// Does the segment joining two front nodes stay inside the material and
/// clear of the front?
///
/// Both halves matter. Crossing no front edge is not enough: a chord can leave
/// the material through a concave corner and come back, meeting nothing on the
/// way. Such a seam looks admissible and quietly folds the front over itself.
fn seam_is_clear(front: &Front, fab: &Fabric, grid: &EdgeGrid, a: u32, b: u32) -> bool {
    let pa = fab.pts[front.vertex(a) as usize];
    let pb = fab.pts[front.vertex(b) as usize];
    // The chord has to set off into the material at both ends.
    for (s, from, to) in [(a, pa, pb), (b, pb, pa)] {
        let prev = fab.pts[front.vertex(front.prev(s)) as usize];
        let next = fab.pts[front.vertex(front.next(s)) as usize];
        let convex = geom::orient(prev, from, next) > 0.0;
        let left = geom::orient(from, next, to) > 0.0;
        let right = geom::orient(prev, from, to) > 0.0;
        let inside = if convex { left && right } else { left || right };
        if !inside {
            return false;
        }
    }
    for s in grid.near_segment(pa, pb) {
        let (u, w) = (s, front.next(s));
        if u == a || u == b || w == a || w == b {
            continue;
        }
        if segments_cross(
            pa,
            pb,
            fab.pts[front.vertex(u) as usize],
            fab.pts[front.vertex(w) as usize],
        ) {
            return false;
        }
    }
    true
}

/// Merge two front nodes and queue whatever loops come out of it.
///
/// The two nodes are **identified**, not averaged into a third one. Averaging
/// looks natural and quietly loses area: the quadrangles already laid stay
/// attached to the two original vertices, so the strip between them and the
/// new front is never filled by anything. Rewriting every reference to one
/// vertex into the other closes the front with no gap at all, at the cost of
/// stretching the quadrangles that used it — by less than the seam radius, and
/// the smoothing pass takes it from there.
///
/// Returns `false` when the rewrite is not admissible, in which case the
/// caller treats the loop as stalled rather than corrupting the mesh.
fn seam(fab: &mut Fabric, front: &mut Front, a: u32, b: u32, stack: &mut Vec<(u32, u32)>) -> bool {
    // Either vertex may be the survivor. Trying both roughly doubles how often
    // a seam is admissible, which matters because a seam is the only way a
    // contracting front sheds nodes: refuse too many and the front keeps every
    // node it started with while its edges shrink, until no row fits at all.
    for (x, y) in [(a, b), (b, a)] {
        if merge_into(fab, front.vertex(x), front.vertex(y)) {
            let (m1, m2) = front.merge(a, b, front.vertex(x));
            stack.push((m1, 0));
            stack.push((m2, 0));
            return true;
        }
    }
    false
}

/// Rewrite every reference to `loser`'s vertex into `winner`'s, if the result
/// is a sound mesh. Positions are left exactly where they are: the vertex that
/// survives keeps its own, so nothing already laid moves.
fn merge_into(fab: &mut Fabric, keep: u32, drop: u32) -> bool {
    if keep == drop {
        return false;
    }
    // The discarded position disappears from the mesh, so it must not be one
    // of the caller's contour nodes — the survivor may well be. This is why
    // both directions are worth trying: when one side is pinned, the other
    // one goes.
    // Note `pinned` and not `movable`: a grid core's node is held still, but
    // it is ours and may be given up, which is what lets a slither between
    // core and contour be welded shut instead of left as a crack.
    if fab.pinned[drop as usize] {
        return false;
    }
    let touching = fab.incident[drop as usize].clone();
    let rewrite = |q: [u32; 4]| q.map(|c| if c == drop { keep } else { c });

    for &qi in &touching {
        let q = fab.quads[qi as usize];
        // A quadrangle holding both would collapse onto itself.
        if q.contains(&keep) {
            return false;
        }
        let q = rewrite(q);
        if !geom::quad_is_valid([
            fab.pts[q[0] as usize],
            fab.pts[q[1] as usize],
            fab.pts[q[2] as usize],
            fab.pts[q[3] as usize],
        ]) {
            return false;
        }
    }

    // Identifying two vertices can leave an edge with three cells on it, when
    // both of them already had a neighbour in common. Only edges reaching the
    // surviving vertex can be affected, so counting those is enough.
    let mut around: HashMap<u32, usize> = HashMap::new();
    for &qi in touching.iter().chain(&fab.incident[keep as usize]) {
        let q = rewrite(fab.quads[qi as usize]);
        for t in 0..4 {
            if q[t] == keep {
                *around.entry(q[(t + 1) % 4]).or_insert(0) += 1;
                *around.entry(q[(t + 3) % 4]).or_insert(0) += 1;
            }
        }
    }
    if around.values().any(|&c| c > 2) {
        return false;
    }

    for &qi in &touching {
        fab.quads[qi as usize] = rewrite(fab.quads[qi as usize]);
        fab.incident[keep as usize].push(qi);
    }
    fab.incident[drop as usize].clear();
    true
}
