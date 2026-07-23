//! Volumetric mesher: fill the interior of a closed triangular surface
//! envelope with size-controlled tetrahedra.
//!
//! This is the 3-D companion of [`crate::ops::mesher::triangulate_surface()`]
//! (which meshes the interior of a closed 2-D contour). Filling the empty interior of a
//! surface is what a volume operator does; here it is built on a robust
//! **Delaunay** core rather than an advancing front:
//!
//! 1. **Interior points** — a grid of candidate nodes at the target spacing is
//!    generated inside the envelope (none when the target size exceeds the
//!    geometry, so the fill then uses the boundary nodes alone).
//! 2. **Delaunay tetrahedralization** — the boundary nodes and the generated
//!    interior nodes are tetrahedralized with the incremental Bowyer–Watson
//!    algorithm. A tiny deterministic jitter is applied to the *connectivity*
//!    computation so degenerate, cospherical inputs (a cube's eight corners,
//!    say) are handled without ambiguity; the output keeps the exact
//!    coordinates.
//! 3. **Carving** — tetrahedra whose centroid lies outside the original
//!    envelope are discarded (solid-angle winding test). For a convex
//!    envelope every tetrahedron is kept and the fill tiles the domain
//!    exactly; for a mildly concave one the carving trims the overhang.
//!
//! The envelope must be a **closed, consistently oriented TRI3 surface** (one
//! or more submeshes, all TRI3) attached to a **3-D** `Coords`. The target
//! size is **uniform**. The output is a [`Mesh`] with a single TET4 submesh:
//! the original surface nodes are reused, interior nodes are created in the
//! same `Coords`. This first version targets **convex or mildly concave**
//! envelopes; strong concavities (where the Delaunay boundary departs from the
//! input surface), internal cavities, a variable density field and QUA4 input
//! are left to later steps.

use crate::containers::mesh::{ElementType, Mesh, Node, NodeId, Point3, SubMesh, Vector3};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::store::read;
use std::collections::HashMap;

/// Geometric tolerance factor applied to the model size to decide degeneracy
/// (zero-volume tetrahedra, collapsed faces).
const EPS_FACTOR: f64 = 1e-9;

/// Fill the interior of a closed **TRI3** surface envelope with tetrahedra
/// using the Delaunay method described in the module docs.
///
/// `envelope` must be a [`Mesh`] whose submeshes are **all TRI3**, forming a
/// single closed, consistently oriented surface attached to a **3-D**
/// `Coords`. `target_size` sets the desired element edge length; `None` uses
/// the mean edge length of the envelope's faces.
///
/// The original surface nodes are reused (and re-referenced); interior nodes
/// are created in the same `Coords`. Output tetrahedra follow the
/// [`ElementType::TET4`] convention (positive signed volume — face 0-1-2 CCW
/// seen from node 3).
///
/// This is the uninterruptible convenience form; for a long mesh that a
/// caller may want to stop early, use [`volume_cancellable`].
pub fn volume(envelope: &Mesh, target_size: Option<f64>) -> Result<Mesh> {
    volume_cancellable(envelope, target_size, &NoCancel)
}

/// Like [`volume`], but polls `cancel` periodically so the meshing can be
/// stopped early (returning [`PyrucastError::Interrupted`]). The frontend
/// chooses what `cancel` means — a timeout, an external flag, or, in the
/// Python binding, a `Ctrl+C` via `Python::check_signals`.
pub fn volume_cancellable(
    envelope: &Mesh,
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if let Some(h) = target_size {
        if h <= 0.0 || h.is_nan() {
            return Err(PyrucastError::Message(format!(
                "volume: target_size must be > 0, got {}",
                h
            )));
        }
    }

    let coords = envelope.coords()?;
    let dim = read(&coords)?.dim();
    if dim != 3 {
        return Err(PyrucastError::Message(format!(
            "volume: envelope must be 3-D, got dim={}",
            dim
        )));
    }

    // 1. Trace the surface into a compact point list (boundary nodes first)
    //    and its triangular faces as index triples into that list.
    let (mut points, node_ids, faces) = trace_surface(envelope)?;
    let n0 = points.len();

    // 2. Validate: a closed, consistently oriented triangular manifold.
    validate_closed_oriented(&faces)?;

    // 3. Mesh: generate interior points, Delaunay, then carve to the domain.
    let tets = pave_volume(&mut points, &faces, target_size, cancel)?;
    if tets.is_empty() {
        return Err(PyrucastError::Message(
            "volume: produced no tetrahedron (degenerate or non-closed envelope?)".into(),
        ));
    }

    // 4. Materialise: reuse boundary nodes, create one node per interior
    //    point, then build the TET4 mesh.
    let mut flat_to_node: Vec<NodeId> = Vec::with_capacity(points.len());
    flat_to_node.extend_from_slice(&node_ids);
    // Keep the interior `Node`s alive until the cells reference them.
    let mut _interior: Vec<Node> = Vec::with_capacity(points.len() - n0);
    for p in &points[n0..] {
        let node = Node::create_in(coords.clone(), &[p.x, p.y, p.z])?;
        flat_to_node.push(node.id());
        _interior.push(node);
    }

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TET4));
    for [i, j, k, l] in &tets {
        mesh.add_cell(&[
            flat_to_node[*i],
            flat_to_node[*j],
            flat_to_node[*k],
            flat_to_node[*l],
        ])?;
    }
    Ok(mesh)
}

/// Trace the envelope into `(points, node_ids, faces)`:
/// - `points[i]` is the coordinate of boundary node `i` (compact 0-based);
/// - `node_ids[i]` is that node's [`NodeId`] (for reuse on output);
/// - `faces` lists each TRI3 as a triple of indices into `points`.
///
/// Errors if a submesh is not TRI3 or if there are fewer than four faces.
type TracedSurface = (Vec<Point3>, Vec<NodeId>, Vec<[usize; 3]>);
fn trace_surface(envelope: &Mesh) -> Result<TracedSurface> {
    let coords = envelope.coords()?;
    let c = read(&coords)?;

    let mut index_of: HashMap<NodeId, usize> = HashMap::new();
    let mut node_ids: Vec<NodeId> = Vec::new();
    let mut points: Vec<Point3> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    let compact = |id: NodeId,
                   index_of: &mut HashMap<NodeId, usize>,
                   node_ids: &mut Vec<NodeId>,
                   points: &mut Vec<Point3>|
     -> Result<usize> {
        if let Some(&i) = index_of.get(&id) {
            return Ok(i);
        }
        let s = c.coord(id)?;
        let i = points.len();
        points.push(Point3::new(s[0], s[1], s[2]));
        node_ids.push(id);
        index_of.insert(id, i);
        Ok(i)
    };

    for sm in envelope {
        let (et, conn) = {
            let s = read(sm)?;
            (s.element_type(), s.connectivity().to_vec())
        };
        if et != ElementType::TRI3 {
            return Err(PyrucastError::Message(format!(
                "volume: envelope must be a TRI3 surface, got {}",
                et
            )));
        }
        for tri in conn.chunks(3) {
            let a = compact(tri[0], &mut index_of, &mut node_ids, &mut points)?;
            let b = compact(tri[1], &mut index_of, &mut node_ids, &mut points)?;
            let d = compact(tri[2], &mut index_of, &mut node_ids, &mut points)?;
            faces.push([a, b, d]);
        }
    }

    if faces.len() < 4 {
        return Err(PyrucastError::Message(format!(
            "volume: envelope must have ≥ 4 triangular faces, got {}",
            faces.len()
        )));
    }
    Ok((points, node_ids, faces))
}

/// Check that the faces form a **closed, consistently oriented** triangular
/// manifold: every directed edge occurs exactly once and its reverse occurs
/// exactly once (so every undirected edge is shared by exactly two faces with
/// opposite orientation). Errors otherwise — an open boundary, a non-manifold
/// edge, or an inconsistent winding.
fn validate_closed_oriented(faces: &[[usize; 3]]) -> Result<()> {
    let mut dir: HashMap<(usize, usize), u32> = HashMap::new();
    for f in faces {
        for &(u, v) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            *dir.entry((u, v)).or_insert(0) += 1;
        }
    }
    for (&(u, v), &count) in &dir {
        if count != 1 {
            return Err(PyrucastError::Message(format!(
                "volume: surface is non-manifold — directed edge ({}, {}) used {} times \
                 (each must appear once)",
                u, v, count
            )));
        }
        if dir.get(&(v, u)) != Some(&1) {
            return Err(PyrucastError::Message(format!(
                "volume: surface is open or inconsistently oriented — edge ({}, {}) has no \
                 matching opposite face",
                u, v
            )));
        }
    }
    Ok(())
}

/// Signed volume (×6) of tetrahedron `(a, b, c, d)`; positive iff `d` lies on
/// the side of face `(a, b, c)` its normal `cross(b-a, c-a)` points to.
#[inline]
fn tet_vol6(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    (b - a).cross(&(c - a)).dot(&(d - a))
}

/// Core mesher (no store access — a clean target for future intra-operator
/// parallelism). `points` starts with the boundary nodes (unchanged, in
/// order) and is extended in place with the generated interior points; the
/// returned tetrahedra index into it, each oriented to positive signed volume.
fn pave_volume(
    points: &mut Vec<Point3>,
    faces: &[[usize; 3]],
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Vec<[usize; 4]>> {
    let n0 = points.len();

    // Bounding box and characteristic scale of the boundary.
    let mut lo = Vector3::repeat(f64::INFINITY);
    let mut hi = Vector3::repeat(f64::NEG_INFINITY);
    for p in points.iter() {
        lo = lo.zip_map(&p.coords, f64::min);
        hi = hi.zip_map(&p.coords, f64::max);
    }
    let diag = (hi - lo).norm();
    if diag <= 0.0 {
        return Err(PyrucastError::Message(
            "volume: envelope is degenerate (zero extent)".into(),
        ));
    }

    // Uniform target size: given, else the mean boundary edge length.
    let mut perim = 0.0;
    let mut nedges = 0usize;
    for f in faces {
        let a = points[f[0]];
        let b = points[f[1]];
        let c = points[f[2]];
        perim += (b - a).norm() + (c - b).norm() + (a - c).norm();
        nedges += 3;
    }
    let h = target_size.unwrap_or(perim / nedges as f64);
    if h <= 0.0 || h.is_nan() {
        return Err(PyrucastError::Message(
            "volume: could not determine a positive element size".into(),
        ));
    }
    let eps = diag * EPS_FACTOR;

    // 1. Interior points: a grid at spacing `h`, kept when strictly inside the
    //    envelope and not too close to a boundary node (avoids slivers).
    let nx = ((hi.x - lo.x) / h).floor() as i64;
    let ny = ((hi.y - lo.y) / h).floor() as i64;
    let nz = ((hi.z - lo.z) / h).floor() as i64;
    let min_sep = 0.5 * h;
    for ix in 1..=nx.max(0) {
        cancel.check()?;
        for iy in 1..=ny.max(0) {
            for iz in 1..=nz.max(0) {
                let p = Point3::new(
                    lo.x + ix as f64 * h,
                    lo.y + iy as f64 * h,
                    lo.z + iz as f64 * h,
                );
                if !point_inside_envelope(p, faces, points, eps) {
                    continue;
                }
                // Reject if it nearly coincides with a boundary node.
                let mut too_close = false;
                for q in points[..n0].iter() {
                    if (p - q).norm() < min_sep {
                        too_close = true;
                        break;
                    }
                }
                if !too_close {
                    points.push(p);
                }
            }
        }
    }

    // 2. Delaunay tetrahedralization of all points (Bowyer–Watson). A tiny
    //    deterministic jitter on the connectivity coordinates removes
    //    cospherical degeneracies; the output indexes the un-jittered points.
    let jitter_amp = diag * 1e-7;
    let jpts: Vec<Point3> = points
        .iter()
        .enumerate()
        .map(|(i, p)| *p + jitter(i) * jitter_amp)
        .collect();
    let center = Point3::from((lo + hi) * 0.5);
    let all_tets = bowyer_watson(&jpts, center, diag, cancel)?;

    // 3. Carve to the domain: keep tetrahedra whose centroid is inside the
    //    original envelope, oriented to positive signed volume (using the
    //    exact coordinates).
    // Tetrahedra spanning four coplanar boundary points (a face quadruple on
    // a cospherical input, say) come back degenerate in the exact coordinates;
    // they carry no volume, so dropping them changes nothing but keeps the
    // output strictly positively oriented.
    let vol6_eps = diag * diag * diag * 1e-9;
    let mut tets: Vec<[usize; 4]> = Vec::with_capacity(all_tets.len());
    for mut t in all_tets {
        let a = points[t[0]];
        let b = points[t[1]];
        let c = points[t[2]];
        let d = points[t[3]];
        let v6 = tet_vol6(a, b, c, d);
        if v6.abs() < vol6_eps {
            continue; // degenerate sliver
        }
        let centroid = Point3::from((a.coords + b.coords + c.coords + d.coords) / 4.0);
        if !point_inside_envelope(centroid, faces, points, eps) {
            continue;
        }
        if v6 < 0.0 {
            t.swap(1, 2);
        }
        tets.push(t);
    }
    Ok(tets)
}

/// Small deterministic per-index unit-ish jitter vector (components in
/// `[-0.5, 0.5]`), used to break cospherical/coplanar degeneracies in the
/// Delaunay connectivity without disturbing the output coordinates.
fn jitter(i: usize) -> Vector3 {
    let h = |k: usize| -> f64 {
        let x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (k as u64).wrapping_mul(0x632BE5AB);
        ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    Vector3::new(h(1), h(2), h(3))
}

/// Incremental Bowyer–Watson Delaunay tetrahedralization of `pts`. A large
/// super-tetrahedron (built around `center` with extent set by `diag`)
/// bootstraps the construction and is removed at the end, leaving only
/// tetrahedra whose four vertices are input points (indices `< pts.len()`).
fn bowyer_watson(
    pts: &[Point3],
    center: Point3,
    diag: f64,
    cancel: &dyn Cancel,
) -> Result<Vec<[usize; 4]>> {
    let n = pts.len();
    // Super-tetrahedron vertices (a regular tetra of circumradius `r`), far
    // enough out that every input point sits well inside.
    let r = diag * 1000.0 + 1.0;
    let dirs = [
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(2.0 * 2.0_f64.sqrt() / 3.0, 0.0, -1.0 / 3.0),
        Vector3::new(-2.0_f64.sqrt() / 3.0, (2.0_f64 / 3.0).sqrt(), -1.0 / 3.0),
        Vector3::new(-2.0_f64.sqrt() / 3.0, -(2.0_f64 / 3.0).sqrt(), -1.0 / 3.0),
    ];
    let mut work: Vec<Point3> = pts.to_vec();
    for d in &dirs {
        work.push(center + d * r);
    }

    let mut tets: Vec<[usize; 4]> = vec![[n, n + 1, n + 2, n + 3]];

    for p in 0..n {
        cancel.check()?;
        let pp = work[p];

        // Tetrahedra whose circumsphere contains `pp` form the cavity.
        let mut bad: Vec<usize> = Vec::new();
        for (ti, t) in tets.iter().enumerate() {
            match circumsphere(work[t[0]], work[t[1]], work[t[2]], work[t[3]]) {
                Some((c, r2)) => {
                    if (pp - c).norm_squared() <= r2 * (1.0 + 1e-12) {
                        bad.push(ti);
                    }
                }
                None => bad.push(ti), // degenerate tet → drop it into the cavity
            }
        }

        // Cavity boundary: faces used by exactly one bad tetrahedron.
        let mut face_count: HashMap<[usize; 3], (u32, [usize; 3])> = HashMap::new();
        for &ti in &bad {
            for f in tet_faces(tets[ti]) {
                let mut key = f;
                key.sort_unstable();
                let e = face_count.entry(key).or_insert((0, f));
                e.0 += 1;
                e.1 = f;
            }
        }

        // Remove the cavity tetrahedra, then fan the boundary faces to `pp`.
        let badset: std::collections::HashSet<usize> = bad.into_iter().collect();
        let mut next: Vec<[usize; 4]> = tets
            .iter()
            .enumerate()
            .filter(|(i, _)| !badset.contains(i))
            .map(|(_, t)| *t)
            .collect();
        for (_, (count, f)) in face_count {
            if count != 1 {
                continue;
            }
            let mut nt = [f[0], f[1], f[2], p];
            if tet_vol6(work[nt[0]], work[nt[1]], work[nt[2]], work[nt[3]]) < 0.0 {
                nt.swap(1, 2);
            }
            next.push(nt);
        }
        tets = next;
    }

    // Drop every tetrahedron still touching a super-vertex.
    tets.retain(|t| t.iter().all(|&v| v < n));
    Ok(tets)
}

/// The four triangular faces of a tetrahedron (each omitting one vertex).
#[inline]
fn tet_faces(t: [usize; 4]) -> [[usize; 3]; 4] {
    [
        [t[1], t[2], t[3]],
        [t[0], t[2], t[3]],
        [t[0], t[1], t[3]],
        [t[0], t[1], t[2]],
    ]
}

/// Circumsphere of tetrahedron `(a, b, c, d)`: `(center, radius²)`, or `None`
/// when the four points are (near) coplanar. Solves the linear system for the
/// centre equidistant from all four vertices.
fn circumsphere(a: Point3, b: Point3, c: Point3, d: Point3) -> Option<(Point3, f64)> {
    use nalgebra::Matrix3;
    let ba = b - a;
    let ca = c - a;
    let da = d - a;
    let mat = Matrix3::from_rows(&[ba.transpose(), ca.transpose(), da.transpose()]);
    let rhs = Vector3::new(ba.norm_squared(), ca.norm_squared(), da.norm_squared()) * 0.5;
    let sol = mat.lu().solve(&rhs)?;
    Some((a + sol, sol.norm_squared()))
}

/// True if `p` lies inside the closed, consistently oriented triangular
/// surface `boundary` (indices into `points`), via the signed solid angle
/// (winding number). The total solid angle subtended by a closed surface is
/// `±4π` from an interior point and `0` from an exterior one — a robust test
/// with no ray/edge degeneracies. A point coincident with a surface vertex is
/// treated as inside.
fn point_inside_envelope(p: Point3, boundary: &[[usize; 3]], points: &[Point3], eps: f64) -> bool {
    let mut total = 0.0;
    for f in boundary {
        let a: Vector3 = points[f[0]] - p;
        let b: Vector3 = points[f[1]] - p;
        let c: Vector3 = points[f[2]] - p;
        let la = a.norm();
        let lb = b.norm();
        let lc = c.norm();
        if la < eps || lb < eps || lc < eps {
            return true; // on a boundary vertex
        }
        // Van Oosterom–Strackee: signed solid angle of triangle (a, b, c).
        let num = a.dot(&b.cross(&c));
        let den = la * lb * lc + a.dot(&b) * lc + a.dot(&c) * lb + b.dot(&c) * la;
        total += 2.0 * num.atan2(den);
    }
    total.abs() > 2.0 * std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::store::{insert, Handle};

    /// Build a closed TRI3 surface from explicit points and triangles,
    /// auto-orienting every triangle so its normal points **away** from
    /// `inside` (consistent outward winding for a convex body).
    fn build_surface(
        coords: Handle<Coords>,
        pts: &[(f64, f64, f64)],
        tris: &[[usize; 3]],
        inside: (f64, f64, f64),
    ) -> Mesh {
        let nodes: Vec<NodeId> = pts
            .iter()
            .map(|&(x, y, z)| Node::create_in(coords.clone(), &[x, y, z]).unwrap().id())
            .collect();
        let p: Vec<Point3> = pts.iter().map(|&(x, y, z)| Point3::new(x, y, z)).collect();
        let ctr = Point3::new(inside.0, inside.1, inside.2);
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        for t in tris {
            let a = p[t[0]];
            let b = p[t[1]];
            let c = p[t[2]];
            let nrm = (b - a).cross(&(c - a));
            // Normal should point away from the interior point.
            let t = if nrm.dot(&(ctr - a)) > 0.0 {
                [t[0], t[2], t[1]]
            } else {
                *t
            };
            mesh.add_cell(&[nodes[t[0]], nodes[t[1]], nodes[t[2]]])
                .unwrap();
        }
        mesh
    }

    /// Sum of the (positive) volumes of every TET4 cell, asserting each cell
    /// has strictly positive signed volume (correct TET4 orientation).
    fn total_tet_volume(mesh: &Mesh) -> f64 {
        let types = mesh.element_types().unwrap();
        assert_eq!(types, vec![ElementType::TET4]);
        let n = mesh.cell_count().unwrap();
        let mut total = 0.0;
        for ci in 0..n {
            let p: Vec<Vec<f64>> = (0..4)
                .map(|ni| mesh.node(0, ci, ni).unwrap().coord().unwrap())
                .collect();
            let a = Point3::new(p[0][0], p[0][1], p[0][2]);
            let b = Point3::new(p[1][0], p[1][1], p[1][2]);
            let c = Point3::new(p[2][0], p[2][1], p[2][2]);
            let d = Point3::new(p[3][0], p[3][1], p[3][2]);
            let v6 = tet_vol6(a, b, c, d);
            assert!(
                v6 > 0.0,
                "cell {} has non-positive signed volume {}",
                ci,
                v6
            );
            total += v6 / 6.0;
        }
        total
    }

    /// Volume enclosed by a closed TRI3 envelope, via the divergence theorem
    /// (sum of `p0·(p1×p2)/6` over its triangles). Sign depends on the
    /// envelope orientation, so the magnitude is the geometric volume — the
    /// reference a correct fill must reproduce.
    fn enclosed_volume(mesh: &Mesh) -> f64 {
        let counts = mesh.cell_counts().unwrap();
        let mut v6 = 0.0;
        for (si, &cnt) in counts.iter().enumerate() {
            for ci in 0..cnt {
                let p: Vec<Vec<f64>> = (0..3)
                    .map(|ni| mesh.node(si, ci, ni).unwrap().coord().unwrap())
                    .collect();
                let a = Point3::new(p[0][0], p[0][1], p[0][2]);
                let b = Point3::new(p[1][0], p[1][1], p[1][2]);
                let c = Point3::new(p[2][0], p[2][1], p[2][2]);
                v6 += a.coords.dot(&b.coords.cross(&c.coords));
            }
        }
        (v6 / 6.0).abs()
    }

    fn tetrahedron(coords: Handle<Coords>) -> Mesh {
        let pts = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ];
        let tris = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        build_surface(coords, &pts, &tris, (0.25, 0.25, 0.25))
    }

    /// Triangular prism: the triangle `(0,0)-(1,0)-(0,1)` extruded to
    /// `z = height` (volume `0.5·height`). A convex, non-cospherical solid.
    fn prism(coords: Handle<Coords>, height: f64) -> Mesh {
        let pts = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, height),
            (1.0, 0.0, height),
            (0.0, 1.0, height),
        ];
        let tris = [
            [0, 1, 2], // bottom cap
            [3, 4, 5], // top cap
            [0, 1, 4],
            [0, 4, 3], // side 0-1
            [1, 2, 5],
            [1, 5, 4], // side 1-2
            [2, 0, 3],
            [2, 3, 5], // side 2-0
        ];
        build_surface(coords, &pts, &tris, (0.25, 0.25, height / 2.0))
    }

    /// Unit cube `[0, s]³` surface, each square face split into two triangles.
    /// The eight corners are cospherical — a Delaunay-degenerate input that
    /// the jittered connectivity must still handle.
    fn cube(coords: Handle<Coords>, s: f64) -> Mesh {
        let pts = [
            (0.0, 0.0, 0.0),
            (s, 0.0, 0.0),
            (s, s, 0.0),
            (0.0, s, 0.0),
            (0.0, 0.0, s),
            (s, 0.0, s),
            (s, s, s),
            (0.0, s, s),
        ];
        let quads = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for q in &quads {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
        }
        build_surface(coords, &pts, &tris, (s / 2.0, s / 2.0, s / 2.0))
    }

    /// Regular octahedron with vertices at `±1` on each axis (volume 4/3).
    fn octahedron(coords: Handle<Coords>) -> Mesh {
        let pts = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ];
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for &xi in &[0usize, 1] {
            for &yi in &[2usize, 3] {
                for &zi in &[4usize, 5] {
                    tris.push([xi, yi, zi]);
                }
            }
        }
        build_surface(coords, &pts, &tris, (0.0, 0.0, 0.0))
    }

    #[test]
    fn volume_tetrahedron_is_one_tet() {
        let coords = insert(Coords::new(3).unwrap());
        let mesh = volume(&tetrahedron(coords.clone()), Some(10.0)).unwrap();
        assert_eq!(mesh.cell_count().unwrap(), 1);
        assert!((total_tet_volume(&mesh) - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn volume_coarse_prism_conserves_volume() {
        let coords = insert(Coords::new(3).unwrap());
        // Size larger than the prism: no interior nodes, fills from corners.
        let env = prism(coords.clone(), 1.0);
        let want = enclosed_volume(&env);
        let mesh = volume(&env, Some(10.0)).unwrap();
        assert!(
            (total_tet_volume(&mesh) - want).abs() < 1e-9,
            "prism volume drift: got {}, want {}",
            total_tet_volume(&mesh),
            want
        );
    }

    #[test]
    fn volume_refined_prism_creates_interior_and_conserves_volume() {
        let coords = insert(Coords::new(3).unwrap());
        let env = prism(coords.clone(), 1.0);
        let want = enclosed_volume(&env);
        let before = read(&coords).unwrap().node_count();
        let mesh = volume(&env, Some(0.3)).unwrap();
        let after = read(&coords).unwrap().node_count();
        assert!(after > before, "expected interior nodes to be created");
        assert!(mesh.cell_count().unwrap() > 3, "expected a refined fill");
        assert!(
            (total_tet_volume(&mesh) - want).abs() < 1e-9,
            "prism volume drift: got {}, want {}",
            total_tet_volume(&mesh),
            want
        );
    }

    #[test]
    fn volume_cube_conserves_volume() {
        // Despite cospherical corners, the jittered Delaunay fills the cube.
        let coords = insert(Coords::new(3).unwrap());
        let env = cube(coords.clone(), 1.0);
        let want = enclosed_volume(&env);
        let mesh = volume(&env, Some(10.0)).unwrap();
        assert!(
            (total_tet_volume(&mesh) - want).abs() < 1e-9,
            "cube volume drift: got {}, want {}",
            total_tet_volume(&mesh),
            want
        );
    }

    #[test]
    fn volume_refined_cube_conserves_volume() {
        let coords = insert(Coords::new(3).unwrap());
        let env = cube(coords.clone(), 1.0);
        let want = enclosed_volume(&env);
        let mesh = volume(&env, Some(0.34)).unwrap();
        assert!(mesh.cell_count().unwrap() > 6, "expected a refined fill");
        assert!(
            (total_tet_volume(&mesh) - want).abs() < 1e-9,
            "cube volume drift: got {}, want {}",
            total_tet_volume(&mesh),
            want
        );
    }

    #[test]
    fn volume_octahedron_conserves_volume() {
        let coords = insert(Coords::new(3).unwrap());
        let env = octahedron(coords.clone());
        let want = enclosed_volume(&env); // 4/3
        let mesh = volume(&env, Some(0.5)).unwrap();
        assert!(
            (total_tet_volume(&mesh) - want).abs() < 1e-9,
            "octahedron volume drift: got {}, want {}",
            total_tet_volume(&mesh),
            want
        );
    }

    #[test]
    fn volume_reuses_boundary_nodes() {
        let coords = insert(Coords::new(3).unwrap());
        let mesh = volume(&tetrahedron(coords.clone()), Some(10.0)).unwrap();
        // The single tet reuses the 4 boundary nodes — no node created.
        assert_eq!(read(&coords).unwrap().node_count(), 4);
        assert_eq!(mesh.cell_count().unwrap(), 1);
    }

    #[test]
    fn volume_rejects_open_surface() {
        // Seven of the octahedron's eight faces: a closed surface with a hole
        // (≥ 4 faces, so it reaches — and trips — the manifold/closure check).
        let coords = insert(Coords::new(3).unwrap());
        let pts = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ];
        let mut tris: Vec<[usize; 3]> = Vec::new();
        for &xi in &[0usize, 1] {
            for &yi in &[2usize, 3] {
                for &zi in &[4usize, 5] {
                    tris.push([xi, yi, zi]);
                }
            }
        }
        tris.pop(); // drop one face → open surface
        let open = build_surface(coords.clone(), &pts, &tris, (0.0, 0.0, 0.0));
        let err = volume(&open, Some(10.0)).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("open or inconsistently oriented") || msg.contains("non-manifold"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn volume_rejects_non_tri3() {
        let coords = insert(Coords::new(3).unwrap());
        let ns: Vec<NodeId> = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|&(x, y, z)| Node::create_in(coords.clone(), &[x, y, z]).unwrap().id())
        .collect();
        let mut quad = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
        quad.add_cell(&ns).unwrap();
        assert!(volume(&quad, None).is_err());
    }

    #[test]
    fn volume_rejects_2d_coords() {
        let coords = insert(Coords::new(2).unwrap());
        let ns: Vec<NodeId> = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
            .iter()
            .map(|&(x, y)| Node::create_in(coords.clone(), &[x, y]).unwrap().id())
            .collect();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&ns).unwrap();
        let err = volume(&tri, None).unwrap_err();
        assert!(format!("{}", err).contains("3-D"));
    }

    #[test]
    fn volume_cancellable_stops_on_preset_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(3).unwrap());
        let flag = AtomicBool::new(true);
        let err = volume_cancellable(&octahedron(coords.clone()), Some(0.3), &flag).unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    fn volume_cancellable_completes_when_not_cancelled() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(3).unwrap());
        let flag = AtomicBool::new(false);
        let mesh = volume_cancellable(&octahedron(coords.clone()), Some(0.5), &flag).unwrap();
        assert!(mesh.cell_count().unwrap() > 0);
    }
}
