//! Frontal surface mesher: fill the interior of a closed contour with a
//! size-controlled, node-creating advancing front.
//!
//! Unlike [`crate::ops::mesher::fill_surface()`] — which triangulates the
//! contour nodes themselves (ear clipping / constrained Delaunay, plus an
//! optional refinement pass) — [`surface()`] generates **interior nodes**
//! to honour a target element size, the way a classic surface operator
//! does. The method:
//!
//! 1. **Corner peeling** — repeatedly clip the sharpest convex corner of
//!    the front as a triangle (an "ear"), so acute corners are resolved
//!    first; a closing edge longer than `1.5 × size` is **bisected** with a
//!    fresh node instead, splitting the ear into two triangles.
//! 2. **Frontal layer** — when no corner is sharp enough to peel, the whole
//!    front is offset inward by ~one element size and a strip of elements is
//!    paved between the front and its offset; the offset becomes the new
//!    front and the process recurses. A strip cell is a quadrangle in QUA4
//!    mode and two triangles in TRI3 mode.
//! 3. **Fan closure** — once a convex front has shrunk to roughly a point,
//!    it is closed with a fan around its centroid: triangles in TRI3 mode, a
//!    fan of quadrangles (with at most one leftover triangle) in QUA4 mode.
//!
//! `element_type` may be **TRI3** or **QUA4**. As with the classic operator,
//! a QUA4 mesh is *quad-dominant* but may contain a few triangles (sharp
//! corners and fan leftovers); the result is then a [`Mesh`] with a QUA4
//! submesh and, where needed, a TRI3 submesh.
//!
//! A **single** closed contour is handled, in **2-D** or as a (nearly)
//! planar loop in **3-D** — projected onto its best-fit plane, paved there,
//! then lifted back. The target size is **uniform**. Holes (multiple loops)
//! and a per-node density field are layered on top in later steps.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{
    ElementType, Mesh, Node, NodeId, Point2, Point3, SubMesh, Vector2, Vector3,
};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::ops::mesher::triangulation::{cross2, point_in_triangle, signed_area};
use crate::store::read;
use std::collections::HashMap;

/// Geometric tolerance factor applied to the model size to decide
/// degeneracy (zero-length edges, collapsed directions).
const EPS_FACTOR: f64 = 1e-9;

/// Interior-angle ceiling (radians) for the *sharp* peeling pass: only
/// corners strictly sharper than this are clipped before the front is
/// advanced as a layer. ~91.7°, matching the classic operator's threshold.
const SHARP_PEEL_ANGLE: f64 = 1.6;

/// Inward offset of a frontal layer, as a fraction of the target size.
const LAYER_OFFSET: f64 = 0.85;

/// Fill the interior of a closed **SEG2** contour using the frontal method
/// described in the module docs.
///
/// `contour` must currently be a [`Mesh`] with **exactly one** SEG2 submesh
/// forming a single closed simple loop, attached to a **2-D** `Coords`.
/// `element_type` is [`ElementType::TRI3`] or [`ElementType::QUA4`];
/// `target_size` sets the desired element edge length, `None` uses the mean
/// length of the contour's segments.
///
/// The original contour nodes are reused (and re-referenced); interior
/// nodes are created in the same `Coords`. Output elements are oriented
/// **CCW**. In QUA4 mode the result may carry both a QUA4 and a TRI3 submesh
/// (quad-dominant with a few triangles).
///
/// This is the uninterruptible convenience form; for a long mesh that a
/// caller may want to stop early, use [`surface_cancellable`].
pub fn surface(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
) -> Result<Mesh> {
    surface_cancellable(contour, element_type, target_size, &NoCancel)
}

/// Like [`surface`], but polls `cancel` periodically so the paving can be
/// stopped early (returning [`PyrucastError::Interrupted`]). The frontend
/// chooses what `cancel` means — a timeout, an external flag, or, in the
/// Python binding, a `Ctrl+C` via `Python::check_signals`.
pub fn surface_cancellable(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    let nbnn = match element_type {
        ElementType::TRI3 => 3usize,
        ElementType::QUA4 => 4usize,
        other => {
            return Err(PyrucastError::Message(format!(
                "surface: only TRI3 and QUA4 are supported, got {}",
                other
            )))
        }
    };
    if let Some(h) = target_size {
        if h <= 0.0 || h.is_nan() {
            return Err(PyrucastError::Message(format!(
                "surface: target_size must be > 0, got {}",
                h
            )));
        }
    }
    if contour.len() != 1 {
        return Err(PyrucastError::Message(format!(
            "surface: exactly one contour (SEG2 submesh) is supported for now, got {}",
            contour.len()
        )));
    }

    let coords = contour.coords()?;
    let dim = read(&coords)?.dim();
    if dim != 2 && dim != 3 {
        return Err(PyrucastError::Message(format!(
            "surface: contour must be 2-D or 3-D, got dim={}",
            dim
        )));
    }

    // 1. Validate the submesh and trace its ordered closed chain of nodes.
    let chain = trace_single_loop(contour)?;
    let n0 = chain.len();

    // 2. Collect 2-D points to pave. In 2-D, the coordinates directly; in
    //    3-D, the contour must be (nearly) planar — project it onto the
    //    best-fit plane (Newell normal through the centroid), exactly as
    //    `fill_surface` does, and remember the mapping to lift interior
    //    points back to 3-D.
    let mut projection: Option<Projection3D> = None;
    let points: Vec<Point2> = if dim == 2 {
        let c = read(&coords)?;
        let mut pts = Vec::with_capacity(n0);
        for &id in &chain {
            let s = c.coord(id)?;
            pts.push(Point2::new(s[0], s[1]));
        }
        pts
    } else {
        let pts3: Vec<Point3> = {
            let c = read(&coords)?;
            let mut v = Vec::with_capacity(n0);
            for &id in &chain {
                let s = c.coord(id)?;
                v.push(Point3::new(s[0], s[1], s[2]));
            }
            v
        };
        let normal = crate::ops::mesher::triangulation::newell_normal(&pts3).ok_or_else(|| {
            PyrucastError::Message("surface: 3-D contour is collinear or zero-area".into())
        })?;
        let origin: Point3 = {
            let sum: Vector3 = pts3.iter().map(|p| p.coords).sum();
            Point3::from(sum / pts3.len() as f64)
        };
        let mut bb_min = Vector3::repeat(f64::INFINITY);
        let mut bb_max = Vector3::repeat(f64::NEG_INFINITY);
        let mut max_dev = 0.0_f64;
        for p in &pts3 {
            max_dev = max_dev.max((p - origin).dot(&normal).abs());
            bb_min = bb_min.zip_map(&p.coords, f64::min);
            bb_max = bb_max.zip_map(&p.coords, f64::max);
        }
        let diag = (bb_max - bb_min).norm();
        let tol = 1e-6 * diag;
        if max_dev > tol {
            return Err(PyrucastError::Message(format!(
                "surface: contour is not planar — max deviation {:.3e} exceeds tolerance {:.3e} (1e-6 × diag={:.3e})",
                max_dev, tol, diag
            )));
        }
        let (u, v) = crate::ops::mesher::triangulation::in_plane_basis(normal);
        let pts2: Vec<Point2> = pts3
            .iter()
            .map(|p| {
                let d = p - origin;
                Point2::new(d.dot(&u), d.dot(&v))
            })
            .collect();
        projection = Some(Projection3D { origin, u, v });
        pts2
    };

    // 3. Pave.
    let paved = pave_single(&points, target_size, nbnn, cancel)?;

    // 4. Create one node per interior (Steiner) point; map every point
    //    index to a NodeId. Lift back to 3-D through the projection when set.
    let mut flat_to_node: Vec<NodeId> = Vec::with_capacity(paved.points.len());
    flat_to_node.extend_from_slice(&chain);
    let mut _steiner: Vec<Node> = Vec::with_capacity(paved.points.len() - n0);
    for p in &paved.points[n0..] {
        let coord: Vec<f64> = match &projection {
            None => vec![p.x, p.y],
            Some(proj) => {
                let p3 = proj.origin + proj.u * p.x + proj.v * p.y;
                vec![p3.x, p3.y, p3.z]
            }
        };
        let node = Node::create_in(coords.clone(), &coord)?;
        flat_to_node.push(node.id());
        _steiner.push(node);
    }

    // 5. Build the mesh — a QUA4 submesh and/or a TRI3 submesh.
    let mut parts: Vec<Mesh> = Vec::with_capacity(2);
    if !paved.quads.is_empty() {
        let mut qm = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        for [i, j, k, l] in &paved.quads {
            qm.add_cell(&[
                flat_to_node[*i],
                flat_to_node[*j],
                flat_to_node[*k],
                flat_to_node[*l],
            ])?;
        }
        parts.push(qm);
    }
    if !paved.tris.is_empty() {
        let mut tm = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        for [i, j, k] in &paved.tris {
            tm.add_cell(&[flat_to_node[*i], flat_to_node[*j], flat_to_node[*k]])?;
        }
        parts.push(tm);
    }
    if parts.is_empty() {
        return Err(PyrucastError::Message(
            "surface: paving produced no element".into(),
        ));
    }
    let mut mesh = parts.remove(0);
    for p in &parts {
        mesh = mesh.union(p)?;
    }
    Ok(mesh)
}

/// Trace the single closed loop of a one-submesh SEG2 contour into the
/// ordered list of node ids it visits. Mirrors the validation done by
/// [`crate::ops::mesher::fill_surface()`] for one loop.
fn trace_single_loop(contour: &Mesh) -> Result<Vec<NodeId>> {
    let sm = contour.get(0)?;
    let (et, n_elems, conn) = {
        let s = read(&sm)?;
        (s.element_type(), s.cell_count(), s.connectivity().to_vec())
    };
    if et != ElementType::SEG2 {
        return Err(PyrucastError::Message(format!(
            "surface: contour submesh must be SEG2, got {}",
            et
        )));
    }
    if n_elems < 3 {
        return Err(PyrucastError::Message(format!(
            "surface: contour must have ≥ 3 segments, got {}",
            n_elems
        )));
    }
    let mut next_node: std::collections::HashMap<NodeId, NodeId> =
        std::collections::HashMap::with_capacity(n_elems);
    for i in 0..n_elems {
        let a = conn[2 * i];
        let b = conn[2 * i + 1];
        if next_node.insert(a, b).is_some() {
            return Err(PyrucastError::Message(format!(
                "surface: node {} starts more than one segment",
                a
            )));
        }
    }
    let start = conn[0];
    let mut chain = Vec::with_capacity(n_elems);
    chain.push(start);
    let mut current = *next_node.get(&start).ok_or_else(|| {
        PyrucastError::Message(format!("surface: node {} has no outgoing segment", start))
    })?;
    while current != start {
        if chain.len() > n_elems {
            return Err(PyrucastError::Message(
                "surface: contour is not a closed simple loop".into(),
            ));
        }
        chain.push(current);
        current = *next_node.get(&current).ok_or_else(|| {
            PyrucastError::Message(format!("surface: node {} has no outgoing segment", current))
        })?;
    }
    if chain.len() != n_elems {
        return Err(PyrucastError::Message(format!(
            "surface: contour has multiple disjoint loops ({} nodes traced out of {})",
            chain.len(),
            n_elems
        )));
    }
    Ok(chain)
}

/// Mapping from the paving plane back to 3-D, for a projected contour:
/// `p3 = origin + u·px + v·py`.
struct Projection3D {
    origin: Point3,
    u: Vector3,
    v: Vector3,
}

/// Output of the pure 2-D pavement: the full point list (originals first,
/// then interior points) and the elements as index tuples into it.
struct Paved {
    points: Vec<Point2>,
    tris: Vec<[usize; 3]>,
    quads: Vec<[usize; 4]>,
}

/// Core 2-D frontal mesher for a single closed ring.
///
/// `ring` lists the ring points in order (the loop closes implicitly from
/// the last back to the first; the last point must not repeat the first).
/// `nbnn` is 3 (TRI3) or 4 (QUA4). The returned [`Paved::points`] starts
/// with the input points **unchanged and in order**, followed by the
/// interior points created during paving; all elements are CCW.
///
/// Operates purely on plain vectors — no store access — so it stays a clean
/// target for future intra-operator parallelism.
fn pave_single(
    ring: &[Point2],
    target_size: Option<f64>,
    nbnn: usize,
    cancel: &dyn Cancel,
) -> Result<Paved> {
    let n0 = ring.len();
    if n0 < 3 {
        return Err(PyrucastError::Message(format!(
            "surface: ring must have ≥ 3 points, got {}",
            n0
        )));
    }
    let area = signed_area(ring);
    if area.abs() < 1e-15 {
        return Err(PyrucastError::Message(
            "surface: contour has zero (or near-zero) area — degenerate".into(),
        ));
    }

    let mut pts: Vec<Point2> = ring.to_vec();

    // Front as a ring of indices into `pts`, normalised to CCW.
    let mut front: Vec<usize> = if area < 0.0 {
        (0..n0).rev().collect()
    } else {
        (0..n0).collect()
    };

    // Uniform target size: given, else the mean contour edge length.
    let mut perim = 0.0;
    for i in 0..n0 {
        perim += (ring[(i + 1) % n0] - ring[i]).norm();
    }
    let xmoy = target_size.unwrap_or(perim / n0 as f64);
    if xmoy <= 0.0 || xmoy.is_nan() {
        return Err(PyrucastError::Message(
            "surface: could not determine a positive element size".into(),
        ));
    }
    let eps = xmoy * EPS_FACTOR;

    let mut tris: Vec<[usize; 3]> = Vec::new();
    let mut quads: Vec<[usize; 4]> = Vec::new();

    // Generous safety cap: paving must shrink the front each iteration.
    let mut guard = 0usize;
    let cap = 1000 + 200 * n0 * n0;

    loop {
        guard += 1;
        if guard > cap {
            return Err(PyrucastError::Message(
                "surface: frontal paving did not converge (possibly non-simple contour)".into(),
            ));
        }
        // Cooperative cancellation point. Each iteration is a coarse event
        // (a whole peeled ear or a paved layer), so one check per turn is
        // cheap — a no-op for `NoCancel`, one signal check per layer for the
        // Python token.
        cancel.check()?;

        let m = front.len();
        if m == 0 {
            break;
        }
        if m == 3 {
            tris.push([front[0], front[1], front[2]]);
            break;
        }
        if m < 3 {
            return Err(PyrucastError::Message(
                "surface: front collapsed to fewer than 3 nodes".into(),
            ));
        }
        // Close a convex 4-node front directly as one quad in QUA4 mode.
        if nbnn == 4 && m == 4 && is_convex(&pts, &front, eps) {
            quads.push([front[0], front[1], front[2], front[3]]);
            break;
        }

        // 1. Sharp peel: clip the sharpest convex ear (θ < SHARP_PEEL_ANGLE).
        if let Some(i) = sharpest_ear(&pts, &front, SHARP_PEEL_ANGLE) {
            peel_or_bisect(&mut pts, &mut front, &mut tris, i, xmoy);
            continue;
        }

        // 2. Frontal layer when the (convex) front is smooth.
        if is_convex(&pts, &front, eps) {
            advance_layer(&mut pts, &mut front, &mut tris, &mut quads, nbnn, xmoy, eps);
            continue;
        }

        // 3. Concave front, no sharp ear: fall back to plain ear clipping
        //    (any convex corner), which triangulates any simple polygon.
        if let Some(i) = sharpest_ear(&pts, &front, std::f64::consts::PI) {
            peel_or_bisect(&mut pts, &mut front, &mut tris, i, xmoy);
            continue;
        }

        return Err(PyrucastError::Message(
            "surface: no ear found — contour is likely non-simple".into(),
        ));
    }

    // Post-pass: local quality-improving edge flips. Peeling, layer
    // splitting and fan closure each choose a triangulation greedily and
    // locally; whichever one produced the eventual worst sliver, a flip
    // pass catches it uniformly by looking only at the final triangle
    // shapes, regardless of which stage created them.
    improve_triangulation_by_flips(&pts, &mut tris);

    Ok(Paved {
        points: pts,
        tris,
        quads,
    })
}

/// Smallest interior angle of triangle `(a, b, c)`, in radians.
fn min_angle(a: Point2, b: Point2, c: Point2) -> f64 {
    let angle_at = |p: Point2, q: Point2, r: Point2| -> f64 {
        let u: Vector2 = q - p;
        let v: Vector2 = r - p;
        u.angle(&v)
    };
    angle_at(a, b, c)
        .min(angle_at(b, c, a))
        .min(angle_at(c, a, b))
}

/// Repeatedly flip interior edges of a pure-triangle mesh when doing so
/// strictly improves the worst angle of the two triangles it borders — a
/// Lawson-style local quality pass. Only ever touches an edge shared by
/// exactly two triangles whose four vertices form a convex quad (so the
/// flip is always a valid re-triangulation); the outer boundary is never a
/// shared edge and so is never touched. Runs to a fixed point (bounded by a
/// small pass cap — each pass strictly increases the mesh's worst angle
/// somewhere, so it cannot cycle).
fn improve_triangulation_by_flips(pts: &[Point2], tris: &mut [[usize; 3]]) {
    const MAX_PASSES: usize = 8;
    for _ in 0..MAX_PASSES {
        // edge (min, max) -> (triangle index, position of the edge's start
        // vertex within that triangle, in CCW order).
        let mut edge_owner: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut any_flip = false;
        for (t_idx, t) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edge_owner.entry(key).or_insert((t_idx, k));
            }
        }
        for t_idx in 0..tris.len() {
            for k in 0..3 {
                // Re-read fresh each iteration: an earlier (t_idx, k) or an
                // earlier t_idx in this same pass may have already flipped
                // this very triangle as someone else's owner.
                let t = tris[t_idx];
                let u = t[k];
                let v = t[(k + 1) % 3];
                let c1 = t[(k + 2) % 3];
                let key = if u < v { (u, v) } else { (v, u) };
                let Some(&(owner_idx, owner_k)) = edge_owner.get(&key) else {
                    continue;
                };
                if owner_idx == t_idx {
                    continue; // this triangle is the (u, v)-direction owner
                }
                let ot = tris[owner_idx];
                // `ot` holds the edge as (v, u) in its own CCW order. If a
                // flip already touched `owner_idx` or `t_idx` this pass, the
                // map entry is stale — this check catches that and skips
                // rather than acting on it; the next pass rebuilds it fresh.
                if ot[owner_k] != v || ot[(owner_k + 1) % 3] != u {
                    continue; // boundary edge, or stale/inconsistent map entry
                }
                let c2 = ot[(owner_k + 2) % 3];

                // Candidate CCW quad around the shared edge: u, c2, v, c1.
                if cross2(pts[u], pts[c2], pts[v]) <= 0.0
                    || cross2(pts[c2], pts[v], pts[c1]) <= 0.0
                    || cross2(pts[v], pts[c1], pts[u]) <= 0.0
                    || cross2(pts[c1], pts[u], pts[c2]) <= 0.0
                {
                    continue; // not convex: flipping would invert a triangle
                }

                let current_worst =
                    min_angle(pts[u], pts[v], pts[c1]).min(min_angle(pts[v], pts[u], pts[c2]));
                let flipped_worst =
                    min_angle(pts[u], pts[c2], pts[c1]).min(min_angle(pts[c2], pts[v], pts[c1]));
                if flipped_worst > current_worst + 1e-9 {
                    tris[t_idx] = [u, c2, c1];
                    tris[owner_idx] = [c2, v, c1];
                    any_flip = true;
                }
            }
        }
        if !any_flip {
            break;
        }
    }
}

/// Signed area of the polygon described by `front` (indices into `pts`).
fn front_area(pts: &[Point2], front: &[usize]) -> f64 {
    let poly: Vec<Point2> = front.iter().map(|&i| pts[i]).collect();
    signed_area(&poly)
}

/// True if every corner of the (CCW) front turns left — i.e. the polygon is
/// convex within `eps`.
fn is_convex(pts: &[Point2], front: &[usize], eps: f64) -> bool {
    let m = front.len();
    for i in 0..m {
        let a = pts[front[(i + m - 1) % m]];
        let b = pts[front[i]];
        let c = pts[front[(i + 1) % m]];
        if cross2(a, b, c) < -eps {
            return false;
        }
    }
    true
}

/// Find the sharpest convex ear of the front whose interior angle is below
/// `max_angle` and whose triangle is empty of other front vertices.
/// Returns the position in `front`, or `None`.
fn sharpest_ear(pts: &[Point2], front: &[usize], max_angle: f64) -> Option<usize> {
    let m = front.len();
    let mut best: Option<(f64, usize)> = None;
    for i in 0..m {
        let ip = (i + m - 1) % m;
        let in_ = (i + 1) % m;
        let a = pts[front[ip]];
        let b = pts[front[i]];
        let c = pts[front[in_]];
        // Convex corner of a CCW ring.
        if cross2(a, b, c) <= 0.0 {
            continue;
        }
        let u: Vector2 = a - b;
        let v: Vector2 = c - b;
        let theta = u.angle(&v);
        if theta >= max_angle {
            continue;
        }
        // Empty-triangle test: no other front vertex inside (a, b, c).
        let mut empty = true;
        for (j, &idx) in front.iter().enumerate() {
            if j == ip || j == i || j == in_ {
                continue;
            }
            if point_in_triangle(pts[idx], a, b, c) {
                empty = false;
                break;
            }
        }
        if !empty {
            continue;
        }
        if best.map(|(t, _)| theta < t).unwrap_or(true) {
            best = Some((theta, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Clip the ear at position `i`: emit one triangle and drop the apex, or —
/// when the closing edge is longer than `1.5 × xmoy` and the front still has
/// more than three nodes — bisect it with a fresh node, emitting two
/// triangles and replacing the apex by the new node.
fn peel_or_bisect(
    pts: &mut Vec<Point2>,
    front: &mut Vec<usize>,
    tris: &mut Vec<[usize; 3]>,
    i: usize,
    xmoy: f64,
) {
    let m = front.len();
    let ip = (i + m - 1) % m;
    let in_ = (i + 1) % m;
    let a = front[ip];
    let b = front[i];
    let c = front[in_];
    let closing = (pts[a] - pts[c]).norm();
    if m > 3 && closing > 1.5 * xmoy {
        let mid = Point2::from((pts[a].coords + pts[c].coords) * 0.5);
        let midx = pts.len();
        pts.push(mid);
        tris.push([a, b, midx]);
        tris.push([b, c, midx]);
        front[i] = midx;
    } else {
        tris.push([a, b, c]);
        front.remove(i);
    }
}

/// Advance the whole (convex) front inward by one layer: offset every node
/// along its inward bisector by `LAYER_OFFSET × xmoy`, pave a strip of cells
/// between the front and its offset, and replace the front by the offset.
/// When the offset would collapse the front, close it with a fan instead
/// (and empty the front). Strip and fan cells are quadrangles in QUA4 mode
/// (`nbnn == 4`), triangles in TRI3 mode.
fn advance_layer(
    pts: &mut Vec<Point2>,
    front: &mut Vec<usize>,
    tris: &mut Vec<[usize; 3]>,
    quads: &mut Vec<[usize; 4]>,
    nbnn: usize,
    xmoy: f64,
    eps: f64,
) {
    let m = front.len();
    let d = LAYER_OFFSET * xmoy;

    let mut off: Vec<Point2> = Vec::with_capacity(m);
    for k in 0..m {
        let prev = pts[front[(k + m - 1) % m]];
        let cur = pts[front[k]];
        let nxt = pts[front[(k + 1) % m]];
        let d1: Vector2 = cur - prev;
        let d2: Vector2 = nxt - cur;
        let l1 = d1.norm();
        let l2 = d2.norm();
        if l1 < eps || l2 < eps {
            off.push(cur);
            continue;
        }
        // Inward bisector: each edge direction rotated +90° (CCW interior on
        // the left), weighted by the *opposite* edge length.
        let dir = Vector2::new(-l2 * d1.y - l1 * d2.y, l2 * d1.x + l1 * d2.x);
        let nrm = dir.norm();
        if nrm < eps {
            off.push(cur);
            continue;
        }
        // Never push a node in further than its own local front spacing
        // supports: the strip cell it forms is as wide as `d_local` and as
        // long as the front edges around it, so a fixed, spacing-agnostic
        // `d` turns an unevenly discretised front (e.g. a coarse `arc`
        // sitting next to a much finer `line`) into elongated sliver
        // triangles right where the front is locally dense.
        let local_scale = 0.5 * (l1 + l2);
        let d_local = d.min(LAYER_OFFSET * local_scale);
        off.push(cur + dir * (d_local / nrm));
    }

    let cur_area = front_area(pts, front);
    let inner_area = signed_area(&off);
    if inner_area <= 0.3 * cur_area {
        // Collapse: close with a fan around the centroid.
        let mut sum = Vector2::zeros();
        for &k in front.iter() {
            sum += pts[k].coords;
        }
        let centroid = Point2::from(sum / m as f64);
        let ci = pts.len();
        pts.push(centroid);
        close_fan(front, tris, quads, nbnn, ci);
        front.clear();
        return;
    }

    // Create the inner-ring nodes and pave the strip.
    let base = pts.len();
    pts.extend_from_slice(&off);
    let inner: Vec<usize> = (base..base + m).collect();
    for k in 0..m {
        let kn = (k + 1) % m;
        let o = front[k];
        let p = front[kn];
        let q = inner[kn];
        let r = inner[k];
        if nbnn == 4 {
            quads.push([o, p, q, r]);
        } else {
            // A strip cell along curved (or unevenly spaced) front is
            // rarely a nice parallelogram, so the two ways to cut it into
            // triangles — diagonal (o, q) or diagonal (p, r) — are not
            // equivalent: one routinely leaves a sliver the other avoids.
            // Pick whichever split has the better worst angle instead of
            // always cutting along (o, q).
            let split_oq = min_angle(pts[o], pts[p], pts[q]).min(min_angle(pts[o], pts[q], pts[r]));
            let split_pr = min_angle(pts[o], pts[p], pts[r]).min(min_angle(pts[p], pts[q], pts[r]));
            if split_oq >= split_pr {
                tris.push([o, p, q]);
                tris.push([o, q, r]);
            } else {
                tris.push([o, p, r]);
                tris.push([p, q, r]);
            }
        }
    }
    *front = inner;
}

/// Close the current ring `front` with a fan around the centroid node `ci`.
/// TRI3: one triangle per edge. QUA4: a quad per pair of edges, plus at most
/// one leftover triangle when the ring has an odd node count.
fn close_fan(
    front: &[usize],
    tris: &mut Vec<[usize; 3]>,
    quads: &mut Vec<[usize; 4]>,
    nbnn: usize,
    ci: usize,
) {
    let m = front.len();
    if nbnn == 4 {
        let mut k = 0;
        while k + 2 <= m {
            quads.push([front[k], front[(k + 1) % m], front[(k + 2) % m], ci]);
            k += 2;
        }
        if k < m {
            // Odd leftover edge (m odd): close it with a triangle.
            tris.push([front[k], front[(k + 1) % m], ci]);
        }
    } else {
        for k in 0..m {
            tris.push([front[k], front[(k + 1) % m], ci]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::store::{insert, read, Handle};

    fn build_contour_2d(coords: Handle<Coords>, pts: &[(f64, f64)]) -> Mesh {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y)| Node::create_in(coords.clone(), &[x, y]).unwrap())
            .collect();
        let mut contour = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        contour
    }

    /// Total CCW area over every submesh (TRI3 and/or QUA4), asserting each
    /// element is convex and CCW.
    fn total_ccw_area(mesh: &Mesh) -> f64 {
        let types = mesh.element_types().unwrap();
        let counts = mesh.cell_counts().unwrap();
        let mut total = 0.0;
        for (si, et) in types.iter().enumerate() {
            let npc = et.nodes_per_cell();
            for ci in 0..counts[si] {
                let p: Vec<Vec<f64>> = (0..npc)
                    .map(|ni| mesh.node(si, ci, ni).unwrap().coord().unwrap())
                    .collect();
                // Shoelace area of the (CCW) polygon.
                let mut a = 0.0;
                for k in 0..npc {
                    let u = &p[k];
                    let w = &p[(k + 1) % npc];
                    a += u[0] * w[1] - w[0] * u[1];
                }
                a *= 0.5;
                assert!(a > 0.0, "submesh {} cell {} not CCW (area {})", si, ci, a);
                total += a;
            }
        }
        total
    }

    fn regular_polygon(n: usize, r: f64) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (r * t.cos(), r * t.sin())
            })
            .collect()
    }

    /// Axis-aligned square `[0, s]²` boundary, one node every `step` along
    /// the perimeter (CCW), corners included.
    fn square_boundary(s: f64, step: f64) -> Vec<(f64, f64)> {
        let m = (s / step).round() as usize;
        let mut pts = Vec::new();
        for i in 0..m {
            pts.push((i as f64 * step, 0.0));
        }
        for i in 0..m {
            pts.push((s, i as f64 * step));
        }
        for i in 0..m {
            pts.push((s - i as f64 * step, s));
        }
        for i in 0..m {
            pts.push((0.0, s - i as f64 * step));
        }
        pts
    }

    /// Square `[0, s]²` boundary discretized at `fine` spacing on three
    /// sides and at `coarse` spacing on the fourth (bottom) — mimicking a
    /// `line`/`arc` density mismatch on the same contour.
    fn square_boundary_heterogeneous(s: f64, fine: f64, coarse: f64) -> Vec<(f64, f64)> {
        let m_coarse = (s / coarse).round() as usize;
        let m_fine = (s / fine).round() as usize;
        let mut pts = Vec::new();
        for i in 0..m_coarse {
            pts.push((i as f64 * coarse, 0.0));
        }
        for i in 0..m_fine {
            pts.push((s, i as f64 * fine));
        }
        for i in 0..m_fine {
            pts.push((s - i as f64 * fine, s));
        }
        for i in 0..m_fine {
            pts.push((0.0, s - i as f64 * fine));
        }
        pts
    }

    /// Smallest interior angle, in degrees, over every triangle of a TRI3
    /// mesh (assumes a single submesh).
    fn min_angle_deg(mesh: &Mesh) -> f64 {
        let n = mesh.cell_count().unwrap();
        let mut worst = 180.0_f64;
        for ci in 0..n {
            let p: Vec<Vec<f64>> = (0..3)
                .map(|ni| mesh.node(0, ci, ni).unwrap().coord().unwrap())
                .collect();
            for k in 0..3 {
                let a = &p[k];
                let b = &p[(k + 1) % 3];
                let c = &p[(k + 2) % 3];
                let u = [a[0] - b[0], a[1] - b[1]];
                let v = [c[0] - b[0], c[1] - b[1]];
                let dot = u[0] * v[0] + u[1] * v[1];
                let nu = (u[0] * u[0] + u[1] * u[1]).sqrt();
                let nv = (v[0] * v[0] + v[1] * v[1]).sqrt();
                let cos_t = (dot / (nu * nv)).clamp(-1.0, 1.0);
                let deg = cos_t.acos().to_degrees();
                if deg < worst {
                    worst = deg;
                }
            }
        }
        worst
    }

    #[test]
    fn surface_heterogeneous_boundary_spacing_avoids_slivers() {
        // Before `advance_layer` scaled its inward offset to the local front
        // spacing, a front with one side much coarser than the rest (as a
        // `line`/`arc` mix on the same contour easily produces) degraded
        // into elongated sliver triangles — legal (positive area) but with
        // interior angles down near a few degrees. This is a coarse
        // regression floor, not a tight quality bound.
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(
            coords.clone(),
            &square_boundary_heterogeneous(4.0, 0.2, 1.0),
        );
        let tri = surface(&contour, ElementType::TRI3, Some(0.2)).unwrap();
        assert!((total_ccw_area(&tri) - 16.0).abs() < 1e-6);
        let worst = min_angle_deg(&tri);
        assert!(worst > 5.0, "sliver triangle: min angle {worst} deg");
    }

    #[test]
    fn improve_triangulation_flips_bad_diagonal() {
        // A trapezoid (not a rectangle: its two diagonals aren't symmetric,
        // so one triangulation is genuinely better than the other). Split
        // along diagonal (0, 2), the worse of the two choices.
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(5.0, 0.0),
            Point2::new(5.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        let mut tris = vec![[0usize, 1, 2], [0, 2, 3]];
        let before = min_angle(pts[0], pts[1], pts[2]).min(min_angle(pts[0], pts[2], pts[3]));

        improve_triangulation_by_flips(&pts, &mut tris);

        let after = min_angle(pts[tris[0][0]], pts[tris[0][1]], pts[tris[0][2]]).min(min_angle(
            pts[tris[1][0]],
            pts[tris[1][1]],
            pts[tris[1][2]],
        ));
        assert!(
            after > before,
            "flip did not improve worst angle: before={before} after={after}"
        );
        for t in &tris {
            assert!(
                !t.contains(&0) || !t.contains(&2),
                "still split along (0, 2): {t:?}"
            );
        }
    }

    #[test]
    fn surface_square_peels_to_two_triangles() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(
            coords.clone(),
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert!((total_ccw_area(&tri) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_rejects_unsupported_element() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(coords, &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert!(surface(&contour, ElementType::TET4, None).is_err());
    }

    #[test]
    fn surface_rejects_multiple_contours() {
        let coords = insert(Coords::new(2).unwrap());
        let outer = build_contour_2d(
            coords.clone(),
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        );
        let hole = build_contour_2d(coords, &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]);
        let combined = outer.union(&hole).unwrap();
        assert!(surface(&combined, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn surface_square_refined_conserves_area_and_size() {
        let coords = insert(Coords::new(2).unwrap());
        // Boundary already discretized at the target size: interior gets
        // filled. The mesher does not subdivide boundary segments (the input
        // discretization fixes boundary spacing), so we discretize here.
        let contour = build_contour_2d(coords.clone(), &square_boundary(4.0, 1.0));
        let h = 1.0;
        let tri = surface(&contour, ElementType::TRI3, Some(h)).unwrap();
        let n = tri.cell_count().unwrap();
        assert!(n > 2, "expected interior nodes/refinement, got {} cells", n);
        assert!((total_ccw_area(&tri) - 16.0).abs() < 1e-9);
        let mut max_edge = 0.0_f64;
        for ci in 0..n {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            for (u, w) in [(&p0, &p1), (&p1, &p2), (&p2, &p0)] {
                let e = ((w[0] - u[0]).powi(2) + (w[1] - u[1]).powi(2)).sqrt();
                max_edge = max_edge.max(e);
            }
        }
        assert!(
            max_edge < 4.0 * h,
            "max edge {} too large for size {}",
            max_edge,
            h
        );
    }

    #[test]
    fn surface_circle_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let r = 5.0;
        let nseg = 40;
        let contour = build_contour_2d(coords.clone(), &regular_polygon(nseg, r));
        let tri = surface(&contour, ElementType::TRI3, Some(1.0)).unwrap();
        let n = tri.cell_count().unwrap();
        assert!(n > nseg, "circle should be filled with interior nodes");
        let poly_area =
            0.5 * (nseg as f64) * r * r * (2.0 * std::f64::consts::PI / nseg as f64).sin();
        assert!(
            (total_ccw_area(&tri) - poly_area).abs() < 1e-6,
            "area drift: got {}, expected {}",
            total_ccw_area(&tri),
            poly_area
        );
    }

    #[test]
    fn surface_concave_l_shape_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let l = [
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ];
        let contour = build_contour_2d(coords.clone(), &l);
        let tri = surface(&contour, ElementType::TRI3, Some(0.75)).unwrap();
        assert!((total_ccw_area(&tri) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn surface_works_with_cw_contour() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(
            coords.clone(),
            &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
        );
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert!((total_ccw_area(&tri) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_reuses_contour_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(
            coords.clone(),
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let before = read(&coords).unwrap().node_count();
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        let after = read(&coords).unwrap().node_count();
        assert_eq!(before, after, "no interior node expected for coarse square");
        assert_eq!(tri.cell_count().unwrap(), 2);
    }

    #[test]
    fn surface_qua4_square_is_one_quad() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(
            coords.clone(),
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let q = surface(&contour, ElementType::QUA4, Some(10.0)).unwrap();
        assert_eq!(q.element_types().unwrap(), vec![ElementType::QUA4]);
        assert_eq!(q.cell_count().unwrap(), 1);
        assert!((total_ccw_area(&q) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_qua4_circle_is_quad_dominant_and_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let r = 5.0;
        let nseg = 40;
        let contour = build_contour_2d(coords.clone(), &regular_polygon(nseg, r));
        let mesh = surface(&contour, ElementType::QUA4, Some(1.0)).unwrap();
        let types = mesh.element_types().unwrap();
        assert!(types.contains(&ElementType::QUA4), "no quads produced");
        // Quad-dominant: quads outnumber any triangles.
        let counts = mesh.cell_counts().unwrap();
        let nq: usize = types
            .iter()
            .zip(&counts)
            .filter(|(t, _)| **t == ElementType::QUA4)
            .map(|(_, c)| *c)
            .sum();
        let nt: usize = types
            .iter()
            .zip(&counts)
            .filter(|(t, _)| **t == ElementType::TRI3)
            .map(|(_, c)| *c)
            .sum();
        assert!(
            nq > nt,
            "expected quad-dominant mesh, got {} quads {} tris",
            nq,
            nt
        );
        let poly_area =
            0.5 * (nseg as f64) * r * r * (2.0 * std::f64::consts::PI / nseg as f64).sin();
        assert!(
            (total_ccw_area(&mesh) - poly_area).abs() < 1e-6,
            "area drift: got {}",
            total_ccw_area(&mesh)
        );
    }

    #[test]
    fn surface_qua4_refined_square_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let mesh = {
            let contour = build_contour_2d(coords.clone(), &square_boundary(4.0, 1.0));
            surface(&contour, ElementType::QUA4, Some(1.0)).unwrap()
        };
        assert!((total_ccw_area(&mesh) - 16.0).abs() < 1e-9);
    }

    fn build_contour_3d(coords: Handle<Coords>, pts: &[(f64, f64, f64)]) -> Mesh {
        let nodes: Vec<Node> = pts
            .iter()
            .map(|&(x, y, z)| Node::create_in(coords.clone(), &[x, y, z]).unwrap())
            .collect();
        let mut contour = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        let n = nodes.len();
        for i in 0..n {
            contour
                .add_cell(&[nodes[i].id(), nodes[(i + 1) % n].id()])
                .unwrap();
        }
        contour
    }

    /// Total area of a 3-D mesh's elements (any submesh), via the magnitude
    /// of the fan cross products around node 0 of each cell.
    fn total_area_3d(mesh: &Mesh) -> f64 {
        let types = mesh.element_types().unwrap();
        let counts = mesh.cell_counts().unwrap();
        let mut total = 0.0;
        for (si, et) in types.iter().enumerate() {
            let npc = et.nodes_per_cell();
            for ci in 0..counts[si] {
                let p: Vec<Vec<f64>> = (0..npc)
                    .map(|ni| mesh.node(si, ci, ni).unwrap().coord().unwrap())
                    .collect();
                for k in 1..npc - 1 {
                    let e1 = [p[k][0] - p[0][0], p[k][1] - p[0][1], p[k][2] - p[0][2]];
                    let e2 = [
                        p[k + 1][0] - p[0][0],
                        p[k + 1][1] - p[0][1],
                        p[k + 1][2] - p[0][2],
                    ];
                    let cr = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    total += 0.5 * (cr[0].powi(2) + cr[1].powi(2) + cr[2].powi(2)).sqrt();
                }
            }
        }
        total
    }

    #[test]
    fn surface_3d_square_in_z_plane_conserves_area() {
        let coords = insert(Coords::new(3).unwrap());
        let contour = build_contour_3d(
            coords.clone(),
            &[
                (0.0, 0.0, 5.0),
                (4.0, 0.0, 5.0),
                (4.0, 4.0, 5.0),
                (0.0, 4.0, 5.0),
            ],
        );
        let tri = surface(&contour, ElementType::TRI3, Some(1.0)).unwrap();
        let n = tri.cell_count().unwrap();
        // Every node sits on the plane z = 5.
        for ci in 0..n {
            for ni in 0..3 {
                let p = tri.node(0, ci, ni).unwrap().coord().unwrap();
                assert!((p[2] - 5.0).abs() < 1e-9, "node off plane: z={}", p[2]);
            }
        }
        assert!((total_area_3d(&tri) - 16.0).abs() < 1e-9);
    }

    #[test]
    fn surface_3d_tilted_square_conserves_area() {
        let s = 1.0_f64 / 2.0_f64.sqrt();
        let coords = insert(Coords::new(3).unwrap());
        let contour = build_contour_3d(
            coords.clone(),
            &[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, s, s), (0.0, s, s)],
        );
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert!((total_area_3d(&tri) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_cancellable_stops_on_preset_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(2).unwrap());
        // A contour big enough to take several paving iterations.
        let contour = build_contour_2d(coords.clone(), &regular_polygon(64, 5.0));
        // Already-cancelled token: the first poll trips.
        let flag = AtomicBool::new(true);
        let err = surface_cancellable(&contour, ElementType::TRI3, Some(0.2), &flag).unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    fn surface_cancellable_completes_when_not_cancelled() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(2).unwrap());
        let contour = build_contour_2d(coords.clone(), &regular_polygon(24, 3.0));
        let flag = AtomicBool::new(false);
        let tri = surface_cancellable(&contour, ElementType::TRI3, Some(1.0), &flag).unwrap();
        assert!(tri.cell_count().unwrap() > 0);
    }

    #[test]
    fn surface_3d_rejects_non_planar() {
        let coords = insert(Coords::new(3).unwrap());
        let contour = build_contour_3d(
            coords.clone(),
            &[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.5),
                (0.0, 1.0, 0.0),
            ],
        );
        let err = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap_err();
        assert!(format!("{}", err).contains("not planar"));
    }
}
