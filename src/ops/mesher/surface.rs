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
//!    front is offset inward by ~one element size and a strip of triangles
//!    is paved between the front and its offset; the offset becomes the new
//!    front and the process recurses.
//! 3. **Centroid fan** — once a convex front has shrunk to roughly a point,
//!    it is closed with a fan of triangles around its centroid.
//!
//! This first cut handles a **single planar (2-D) contour** with **TRI3**
//! elements and a **uniform** target size. Holes (multiple loops), the
//! quadrangle variant, the per-node density field and 3-D contour
//! projection are layered on top in later steps.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{ElementType, Mesh, Node, NodeId, Point2, SubMesh, Vector2};
use crate::error::{PyrucastError, Result};
use crate::ops::mesher::triangulation::{cross2, point_in_triangle, signed_area};
use crate::store::read;

/// Geometric tolerance factor applied to the model size to decide
/// degeneracy (zero-length edges, collapsed directions).
const EPS_FACTOR: f64 = 1e-9;

/// Interior-angle ceiling (radians) for the *sharp* peeling pass: only
/// corners strictly sharper than this are clipped before the front is
/// advanced as a layer. ~91.7°, matching the classic operator's TRI3
/// threshold.
const SHARP_PEEL_ANGLE: f64 = 1.6;

/// Inward offset of a frontal layer, as a fraction of the target size.
const LAYER_OFFSET: f64 = 0.85;

/// Fill the interior of a closed **SEG2** contour with **TRI3** elements
/// using the frontal method described in the module docs.
///
/// `contour` must currently be a [`Mesh`] with **exactly one** SEG2 submesh
/// forming a single closed simple loop, attached to a **2-D** `Coords`.
/// `target_size` sets the desired element edge length; `None` uses the mean
/// length of the contour's segments.
///
/// The original contour nodes are reused (and re-referenced); interior
/// nodes are created in the same `Coords`. Output triangles are oriented
/// **CCW**.
pub fn surface(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
) -> Result<Mesh> {
    if element_type != ElementType::TRI3 {
        return Err(PyrucastError::Message(format!(
            "surface: only TRI3 is supported for now, got {}",
            element_type
        )));
    }
    if let Some(h) = target_size {
        if !(h > 0.0) {
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
    if dim != 2 {
        return Err(PyrucastError::Message(format!(
            "surface: only 2-D contours are supported for now, got dim={}",
            dim
        )));
    }

    // 1. Validate the submesh and trace its ordered closed chain of nodes.
    let chain = trace_single_loop(contour)?;
    let n0 = chain.len();

    // 2. Collect the 2-D points in chain order.
    let points: Vec<Point2> = {
        let c = read(&coords)?;
        let mut pts = Vec::with_capacity(n0);
        for &id in &chain {
            let s = c.coord(id)?;
            pts.push(Point2::new(s[0], s[1]));
        }
        pts
    };

    // 3. Pave.
    let (all_pts, triangles) = pave_tri3_single(&points, target_size)?;

    // 4. Create one node per interior (Steiner) point; map every point
    //    index to a NodeId.
    let mut flat_to_node: Vec<NodeId> = Vec::with_capacity(all_pts.len());
    flat_to_node.extend_from_slice(&chain);
    let mut _steiner: Vec<Node> = Vec::with_capacity(all_pts.len() - n0);
    for p in &all_pts[n0..] {
        let node = Node::create_in(coords.clone(), &[p.x, p.y])?;
        flat_to_node.push(node.id());
        _steiner.push(node);
    }

    // 5. Build the TRI3 mesh.
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
    for [i, j, k] in triangles {
        mesh.add_cell(&[flat_to_node[i], flat_to_node[j], flat_to_node[k]])?;
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

/// Core 2-D frontal mesher for a single closed ring, TRI3.
///
/// `ring` lists the ring points in order (the loop closes implicitly from
/// the last back to the first; the last point must not repeat the first).
/// Returns the full point list — the first `ring.len()` entries are the
/// input points **unchanged and in order**, followed by the interior points
/// created during paving — together with the triangles as index triples
/// into that list. All triangles are oriented CCW.
fn pave_tri3_single(
    ring: &[Point2],
    target_size: Option<f64>,
) -> Result<(Vec<Point2>, Vec<[usize; 3]>)> {
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
    if !(xmoy > 0.0) {
        return Err(PyrucastError::Message(
            "surface: could not determine a positive element size".into(),
        ));
    }
    let eps = xmoy * EPS_FACTOR;

    let mut tris: Vec<[usize; 3]> = Vec::new();

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

        let m = front.len();
        if m == 0 {
            break;
        }
        if m == 3 {
            tris.push([front[0], front[1], front[2]]);
            break;
        }
        if m < 3 {
            // A degenerate sliver remains; nothing valid to emit.
            return Err(PyrucastError::Message(
                "surface: front collapsed to fewer than 3 nodes".into(),
            ));
        }

        // 1. Sharp peel: clip the sharpest convex ear (θ < SHARP_PEEL_ANGLE).
        if let Some(i) = sharpest_ear(&pts, &front, SHARP_PEEL_ANGLE) {
            peel_or_bisect(&mut pts, &mut front, &mut tris, i, xmoy);
            continue;
        }

        // 2. Frontal layer when the (convex) front is smooth.
        if is_convex(&pts, &front, eps) {
            advance_layer(&mut pts, &mut front, &mut tris, xmoy, eps);
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

    Ok((pts, tris))
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
/// along its inward bisector by `LAYER_OFFSET × xmoy`, pave a strip of
/// triangles between the front and its offset, and replace the front by the
/// offset. When the offset would collapse the front, close it with a
/// centroid fan instead (and empty the front).
fn advance_layer(
    pts: &mut Vec<Point2>,
    front: &mut Vec<usize>,
    tris: &mut Vec<[usize; 3]>,
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
        off.push(cur + dir * (d / nrm));
    }

    let cur_area = front_area(pts, front);
    let inner_area = signed_area(&off);
    if inner_area <= 0.3 * cur_area {
        // Collapse: close with a centroid fan.
        let mut sum = Vector2::zeros();
        for &k in front.iter() {
            sum += pts[k].coords;
        }
        let centroid = Point2::from(sum / m as f64);
        let ci = pts.len();
        pts.push(centroid);
        for k in 0..m {
            tris.push([front[k], front[(k + 1) % m], ci]);
        }
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
        tris.push([o, p, q]);
        tris.push([o, q, r]);
    }
    *front = inner;
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

    /// Total CCW area of the output triangles, and a check that each is CCW.
    fn total_ccw_area(tri: &Mesh) -> f64 {
        let n = tri.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n {
            let p0 = tri.node(0, ci, 0).unwrap().coord().unwrap();
            let p1 = tri.node(0, ci, 1).unwrap().coord().unwrap();
            let p2 = tri.node(0, ci, 2).unwrap().coord().unwrap();
            let a = 0.5
                * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]));
            assert!(a > 0.0, "triangle {} not CCW (signed area {})", ci, a);
            total += a;
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

    #[test]
    fn surface_square_peels_to_two_triangles() {
        let coords = insert(Coords::new(2).unwrap());
        // Unit square, target size ≥ diagonal ⇒ pure peeling, no interior nodes.
        let contour =
            build_contour_2d(coords.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert_eq!(tri.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert!((total_ccw_area(&tri) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_rejects_non_tri3() {
        let coords = insert(Coords::new(2).unwrap());
        let contour =
            build_contour_2d(coords, &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        assert!(surface(&contour, ElementType::QUA4, None).is_err());
    }

    #[test]
    fn surface_rejects_multiple_contours() {
        let coords = insert(Coords::new(2).unwrap());
        let outer =
            build_contour_2d(coords.clone(), &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
        let hole =
            build_contour_2d(coords, &[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]);
        let combined = outer.union(&hole).unwrap();
        assert!(surface(&combined, ElementType::TRI3, None).is_err());
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
        // No edge wildly larger than a few target sizes.
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
        assert!(max_edge < 4.0 * h, "max edge {} too large for size {}", max_edge, h);
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
        // Polygonal area of the inscribed regular n-gon.
        let poly_area = 0.5 * (nseg as f64) * r * r
            * (2.0 * std::f64::consts::PI / nseg as f64).sin();
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
        // Clockwise unit square.
        let contour =
            build_contour_2d(coords.clone(), &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert_eq!(tri.cell_count().unwrap(), 2);
        assert!((total_ccw_area(&tri) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn surface_reuses_contour_nodes() {
        let coords = insert(Coords::new(2).unwrap());
        let contour =
            build_contour_2d(coords.clone(), &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let before = read(&coords).unwrap().node_count();
        let tri = surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        // Pure peeling on the unit square adds no interior node.
        let after = read(&coords).unwrap().node_count();
        assert_eq!(before, after, "no interior node expected for coarse square");
        // The two triangles reference the four original nodes only.
        assert_eq!(tri.cell_count().unwrap(), 2);
    }
}
