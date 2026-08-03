//! Frontal paving: the machinery behind [`pave_surface`](fn@super::pave_surface).
//!
//! The front starts as the domain's boundary and walks inward, laying a whole
//! row of quadrangles at a time ([`row`]) until each loop is small enough to
//! close ([`close`]). Two things keep that from going wrong:
//!
//! - **the invariant.** The front is always a set of simple, pairwise disjoint
//!   loops. Nothing is committed that would break it: a row whose quadrangles
//!   are not all strictly convex, or whose new edges would cross the front, is
//!   refused and retried closer in. Every such decision goes through the exact
//!   predicate in [`geom`], so it is a fact and not an estimate.
//! - **the way out.** When two parts of the front come within touching
//!   distance they are seamed together ([`Front::merge`](front::Front::merge)),
//!   which splits a loop in two where the domain is concave and joins two loops
//!   into one where a hole is being swallowed. Holes therefore need no special
//!   handling anywhere in this module.
//!
//! Paving can stall — a loop that neither advances nor seams. That is not
//! treated as a failure: after a few stalled turns the loop is handed to
//! [`close`], which fills any simple polygon. The paver can degrade, but it
//! cannot fail to return a conforming mesh.

pub mod cleanup;
pub mod close;
pub mod front;
pub mod geom;
pub mod proximity;
pub mod row;
pub mod smooth;

use crate::atoms::{NodeId, Point2};
use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;
use crate::ops::mesher::contour::Domain;
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

/// Two front nodes closer than this many target sizes are seamed together.
const SEAM_FACTOR: f64 = 0.72;

/// How many times `unstick` doubles its search radius before giving up.
const CHORD_RADIUS_STEPS: usize = 4;

/// How many of the shortest chords `unstick` checks for clearance per radius.
const CHORD_CANDIDATES: usize = 8;

/// Stalled turns tolerated on one loop before it is closed outright.
const MAX_STALL: u32 = 2;

/// Largest fold, in units of the target cell's area, written off as two
/// fronts grazing rather than reported as a region that could not be meshed.
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
    /// `false` for a node that must keep its position — every node of the
    /// user's contour.
    pub movable: Vec<bool>,
    pub quads: Vec<[u32; 4]>,
    pub tris: Vec<[u32; 3]>,
    /// Store identity of the contour nodes, in the order they occupy the first
    /// entries of `pts`. Shorter than `pts`: everything past it is new.
    pub contour_ids: Vec<NodeId>,
    /// Quadrangles touching each vertex — needed by the seam, which rewrites
    /// connectivity rather than adding geometry.
    incident: Vec<Vec<u32>>,
}

impl Fabric {
    fn add(&mut self, p: Point2, movable: bool) -> u32 {
        self.pts.push(p);
        self.movable.push(movable);
        self.incident.push(Vec::new());
        (self.pts.len() - 1) as u32
    }

    fn push_quad(&mut self, q: [u32; 4]) {
        let i = self.quads.len() as u32;
        self.quads.push(q);
        for &c in &q {
            self.incident[c as usize].push(i);
        }
    }
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
) -> Result<Fabric> {
    let mut fab = Fabric {
        pts: Vec::new(),
        movable: Vec::new(),
        quads: Vec::new(),
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
            verts.push(fab.add(*p, false));
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
                    "pave_surface: all_quad was asked for, but the {which} loop has {} \
                     segments — an odd number. A polygon with an odd number of sides has no \
                     filling by quadrangles alone, and paving cannot change that parity. \
                     Re-mesh that loop with an even number of segments.",
                    verts.len()
                )));
            }
        }
    }

    let mut front = Front::new();
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for verts in &loops {
        if verts.len() >= 3 {
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
            close_loop(&mut fab, &mut front, rep, target)?;
            continue;
        }
        if stalls > MAX_STALL {
            // Cutting the loop in two usually gets it moving again; closing it
            // outright would fill a large polygon with a decomposition, which
            // is a far worse mesh than two more rows would have been.
            if !unstick(&fab, &mut front, rep, target, all_quad, &mut stack) {
                close_loop(&mut fab, &mut front, rep, target)?;
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
    Ok(fab)
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
    let (mut pts, mut movable) = (Vec::new(), Vec::new());
    for i in 0..fab.pts.len() {
        if used[i] {
            remap[i] = pts.len() as u32;
            pts.push(fab.pts[i]);
            movable.push(fab.movable[i]);
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
    // The advance follows the front's own spacing, pulled toward the target so
    // an unevenly discretised contour converges to the wanted size instead of
    // propagating its own irregularity inward for ever.
    let base: Vec<f64> = (0..n)
        .map(|i| {
            let span = 0.5 * ((p[(i + 1) % n] - p[i]).norm() + (p[i] - p[(i + n - 1) % n]).norm());
            (0.5 * span + 0.5 * target).clamp(0.5 * target, 2.0 * target)
        })
        .collect();
    let grid = EdgeGrid::build(front, &fab.pts, target);

    let mut scale = vec![1.0f64; n];
    for _ in 0..RETREAT_STEPS {
        let sz = |i: usize| base[i] * scale[i];
        let want = |i: usize| base[i];
        match row::plan(front, &fab.pts, rep, &sz, &want, all_quad) {
            Ok(plan) => {
                if chain_is_free(front, fab, &grid, &plan) {
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
    // And against itself. Asking every pair would be quadratic in the front
    // length, which on a fine mesh is the dominant cost of the whole paver.
    let ring: Vec<Point2> = plan.chain.iter().map(|&i| at(i)).collect();
    let own = EdgeGrid::of_ring(&ring, 0.0);
    for i in 0..m {
        let (a, b) = (ring[i], ring[(i + 1) % m]);
        for j in own.near_segment(a, b) {
            let j = j as usize;
            // Skip itself and the two edges it shares an endpoint with.
            if j == i || (j + 1) % m == i || (i + 1) % m == j {
                continue;
            }
            if segments_cross(a, b, ring[j], ring[(j + 1) % m]) {
                return false;
            }
        }
    }
    true
}

/// Write a planned row into the mesh and advance the front over it.
fn commit(fab: &mut Fabric, front: &mut Front, rep: u32, plan: RowPlan) -> Option<u32> {
    let base = fab.pts.len() as u32;
    for p in &plan.pts {
        fab.add(*p, true);
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
fn relax_front(fab: &mut Fabric, front: &Front, rep: u32) {
    let slots = front.loop_slots(rep);
    let n = slots.len();
    if n < 4 {
        return;
    }
    let verts: Vec<u32> = slots.iter().map(|&s| front.vertex(s)).collect();
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
}

/// Fill what is left of a loop with elements and retire it.
fn close_loop(fab: &mut Fabric, front: &mut Front, rep: u32, target: f64) -> Result<()> {
    let verts: Vec<u32> = front
        .loop_slots(rep)
        .iter()
        .map(|&s| front.vertex(s))
        .collect();
    // A front loop bounds material on its left, so a closed one encloses
    // positive area. A non-positive one means two parts of the front folded
    // over each other, and whatever is between them cannot be meshed. Dropping
    // it silently would return a mesh with a hole in it, so it is an error —
    // and one worth locating, since it always comes from a contour that is too
    // coarse or too uneven for the size asked of it.
    let poly: Vec<Point2> = verts.iter().map(|&v| fab.pts[v as usize]).collect();
    let area = crate::ops::mesher::triangulation::signed_area(&poly);
    if area <= 0.0 {
        // Two fronts meeting head-on normally leave a slither of overlap
        // between them, which is degenerate and covers nothing; discarding
        // that is right. What is not right is discarding a region that was
        // supposed to hold cells, so the two are told apart by size.
        if -area <= FOLD_TOLERANCE * target * target {
            // Two lines of front lying on top of each other. Welding them
            // shut costs nothing and leaves the mesh whole; simply dropping
            // the ring would leave a crack behind, and a crack is a hole in
            // the connectivity even when it has no area worth speaking of.
            weld(fab, front, rep);
            front.kill_loop(rep);
            return Ok(());
        }
        let centre = poly
            .iter()
            .fold(Point2::origin(), |acc, p| acc + p.coords)
            .coords
            / poly.len() as f64;
        return Err(PyrucastError::Message(format!(
            "pave_surface: the advancing front folded onto itself near ({:.6}, {:.6}) in the \
             meshing plane, leaving a region that cannot be filled. The contour there is too \
             coarse or too uneven for the element size asked of it: discretise it closer to \
             the target size, or ask for a larger one.",
            centre.x, centre.y
        )));
    }
    let filled = close::close(&fab.pts, &verts);
    for p in &filled.added {
        fab.add(*p, true);
    }
    for q in filled.quads {
        fab.push_quad(q);
    }
    fab.tris.extend(filled.tris);
    front.kill_loop(rep);
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
/// Whatever refuses to merge is left alone: the ring is retired either way,
/// and a refused weld costs a crack, not correctness.
fn weld(fab: &mut Fabric, front: &Front, rep: u32) {
    let mut verts: Vec<u32> = front
        .loop_slots(rep)
        .iter()
        .map(|&s| front.vertex(s))
        .collect();
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
    if !fab.movable[drop as usize] {
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
