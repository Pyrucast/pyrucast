//! Surface mesher: fill one or more closed 2-D contours (with holes) with
//! TRI3 or QUA4 cells.
//!
//! Pipeline per domain (an outer CCW loop plus its CW hole loops):
//! 1. Insert all boundary points into an unconstrained Delaunay
//!    triangulation (incremental Bowyer-Watson, super-triangle bootstrap).
//! 2. Recover every boundary/hole edge that isn't already present (corridor
//!    removal + ear-clipping of the two flanking polygons), then mark every
//!    loop edge constrained and drop the super-triangle.
//! 3. Legalize (Delaunay-improving flips that never cross a constraint).
//! 4. Excavate: flood fill from a triangle known to be inside the outer
//!    loop, never crossing a constrained edge, to separate domain interior
//!    from holes and from convex-hull pockets outside a concave boundary.
//! 5. Ruppert refinement: split skinny/oversized inside triangles by
//!    circumcenter insertion, or split an encroached constrained edge at its
//!    midpoint instead (the standard boundary-robustness rule).
//! 6. Light Laplacian smoothing, then (for QUA4) greedy triangle-pair
//!    recombination.
//!
//! All in-circle/orientation decisions run on a tiny deterministic jitter of
//! the coordinates (index-seeded, not random) so cocircular/collinear input
//! never produces an ambiguous case. A 3-D contour is fit to its best plane
//! (Newell's method) and meshed in that local 2-D frame.
//!
//! Multiple disjoint outer loops are independent domains; each is meshed on
//! its own point set with no shared mutable state, so the per-domain work is
//! embarrassingly parallel (kept sequential here since the domain loop
//! shares a `&dyn Cancel` token, which is not `Sync`). Smoothing and the
//! QUA4 recombination pass run over plain local data and use `rayon`.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{
    Coords, ElementType, Mesh, Node, NodeId, Point2, Point3, SubMesh, Vector2, Vector3,
};
use crate::error::{PyrucastError, Result};
use crate::interrupt::{Cancel, NoCancel};
use crate::store::{insert, read, Handle};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

// Ruppert refinement provably terminates for a minimum-angle threshold up
// to ~20.7°; staying just under it keeps the point count finite.
const MIN_ANGLE_DEG: f64 = 20.0;

/// Mesh the interior of `contour` (one or more closed SEG2 loops, CCW
/// outer / CW hole per [`crate::ops::mesher::border()`]'s convention) with
/// `element_type` (TRI3 or QUA4) cells. `target_size` sets the desired edge
/// length; `None` uses the mean boundary edge length of each domain.
///
/// This is the uninterruptible convenience form; for a long mesh a caller
/// may want to stop early, use [`triangulate_surface_cancellable`].
pub fn triangulate_surface(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
) -> Result<Mesh> {
    triangulate_surface_cancellable(contour, element_type, target_size, &NoCancel)
}

/// Like [`triangulate_surface`], but polls `cancel` periodically so meshing can be
/// stopped early (returning [`PyrucastError::Interrupted`]).
pub fn triangulate_surface_cancellable(
    contour: &Mesh,
    element_type: ElementType,
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> {
    if !matches!(element_type, ElementType::TRI3 | ElementType::QUA4) {
        return Err(PyrucastError::Message(format!(
            "triangulate_surface: element_type must be TRI3 or QUA4, got {}",
            element_type
        )));
    }
    if let Some(h) = target_size {
        if h <= 0.0 || h.is_nan() {
            return Err(PyrucastError::Message(format!(
                "triangulate_surface: target_size must be > 0, got {}",
                h
            )));
        }
    }
    let coords_handle = contour.coords()?;
    let dim = read(&coords_handle)?.dim();
    if dim != 2 && dim != 3 {
        return Err(PyrucastError::Message(format!(
            "triangulate_surface: contour must be 2-D or 3-D, got dim={}",
            dim
        )));
    }

    let loops = extract_loops(contour)?;
    if loops.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_surface: contour has no boundary loop".into(),
        ));
    }
    let frame = Frame::fit(dim, &loops)?;
    let loops2d: Vec<Loop2D> = loops
        .iter()
        .map(|l| Loop2D {
            node_ids: l.node_ids.clone(),
            pts: l.world_pts.iter().map(|p| frame.to_local(p)).collect(),
        })
        .collect();
    let domains = build_domains(loops2d)?;

    let mut results = Vec::with_capacity(domains.len());
    for d in &domains {
        results.push(mesh_domain(d, element_type, target_size, cancel)?);
    }

    materialize(coords_handle, &frame, dim, element_type, results)
}

// ─── Contour parsing ──────────────────────────────────────────────────────

struct LoopData {
    node_ids: Vec<NodeId>,
    world_pts: Vec<Vec<f64>>,
}

fn extract_loops(mesh: &Mesh) -> Result<Vec<LoopData>> {
    let coords = mesh.coords()?;
    let c = read(&coords)?;
    let mut loops = Vec::new();
    for sm in mesh {
        let s = read(sm)?;
        if s.element_type() != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "triangulate_surface: contour submeshes must be SEG2, got {}",
                s.element_type()
            )));
        }
        let conn = s.connectivity();
        let n = conn.len() / 2;
        if n < 3 {
            return Err(PyrucastError::Message(
                "triangulate_surface: a boundary loop needs at least 3 segments".into(),
            ));
        }
        let mut next: HashMap<NodeId, NodeId> = HashMap::new();
        for pair in conn.chunks(2) {
            if next.insert(pair[0], pair[1]).is_some() {
                return Err(PyrucastError::Message(
                    "triangulate_surface: a boundary submesh is not a simple loop (branching)"
                        .into(),
                ));
            }
        }
        let start = conn[0];
        let mut chain = Vec::with_capacity(n);
        let mut cur = start;
        let mut seen = HashSet::new();
        for _ in 0..n {
            if !seen.insert(cur) {
                return Err(PyrucastError::Message(
                    "triangulate_surface: a boundary submesh is not a simple loop (repeated node)"
                        .into(),
                ));
            }
            chain.push(cur);
            cur = *next.get(&cur).ok_or_else(|| {
                PyrucastError::Message(
                    "triangulate_surface: a boundary submesh is not closed".into(),
                )
            })?;
        }
        if cur != start {
            return Err(PyrucastError::Message(
                "triangulate_surface: a boundary submesh is not closed".into(),
            ));
        }
        let world_pts: Result<Vec<Vec<f64>>> = chain
            .iter()
            .map(|&nid| Ok(c.coord(nid)?.to_vec()))
            .collect();
        loops.push(LoopData {
            node_ids: chain,
            world_pts: world_pts?,
        });
    }
    Ok(loops)
}

// ─── Planar frame (2-D native, or best-fit plane for a 3-D contour) ───────

enum Frame {
    Planar2D,
    Planar3D {
        origin: Point3,
        u: Vector3,
        v: Vector3,
    },
}

impl Frame {
    fn fit(dim: u8, loops: &[LoopData]) -> Result<Frame> {
        if dim == 2 {
            return Ok(Frame::Planar2D);
        }
        let mut origin = Vector3::zeros();
        let mut count = 0usize;
        for l in loops {
            for p in &l.world_pts {
                origin += Vector3::new(p[0], p[1], p[2]);
                count += 1;
            }
        }
        if count == 0 {
            return Err(PyrucastError::Message(
                "triangulate_surface: empty contour".into(),
            ));
        }
        origin /= count as f64;
        let mut normal = Vector3::zeros();
        for l in loops {
            let pts = &l.world_pts;
            let n = pts.len();
            for i in 0..n {
                let a = Vector3::new(pts[i][0], pts[i][1], pts[i][2]);
                let b = Vector3::new(
                    pts[(i + 1) % n][0],
                    pts[(i + 1) % n][1],
                    pts[(i + 1) % n][2],
                );
                normal.x += (a.y - b.y) * (a.z + b.z);
                normal.y += (a.z - b.z) * (a.x + b.x);
                normal.z += (a.x - b.x) * (a.y + b.y);
            }
        }
        let nn = normal.norm();
        if nn < 1e-30 {
            return Err(PyrucastError::Message(
                "triangulate_surface: contour points are collinear or degenerate".into(),
            ));
        }
        normal /= nn;
        let helper = if normal.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        let u = (helper - normal * helper.dot(&normal)).normalize();
        let v = normal.cross(&u);
        Ok(Frame::Planar3D {
            origin: Point3::from(origin),
            u,
            v,
        })
    }

    fn to_local(&self, p: &[f64]) -> Point2 {
        match self {
            Frame::Planar2D => Point2::new(p[0], p[1]),
            Frame::Planar3D { origin, u, v } => {
                let d = Vector3::new(p[0], p[1], p[2]) - origin.coords;
                Point2::new(d.dot(u), d.dot(v))
            }
        }
    }

    fn to_world(&self, p: Point2, dim: u8) -> Vec<f64> {
        match self {
            Frame::Planar2D => vec![p.x, p.y],
            Frame::Planar3D { origin, u, v } => {
                debug_assert_eq!(dim, 3);
                let w = origin.coords + u * p.x + v * p.y;
                vec![w.x, w.y, w.z]
            }
        }
    }
}

// ─── Domains: outer CCW loop + its CW hole loops ──────────────────────────

struct Loop2D {
    node_ids: Vec<NodeId>,
    pts: Vec<Point2>,
}

struct Domain {
    outer: Loop2D,
    holes: Vec<Loop2D>,
}

fn signed_area(pts: &[Point2]) -> f64 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    a * 0.5
}

fn point_in_polygon(p: Point2, poly: &[Point2]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i].x, poly[i].y);
        let (xj, yj) = (poly[j].x, poly[j].y);
        if (yi > p.y) != (yj > p.y) && p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn build_domains(loops: Vec<Loop2D>) -> Result<Vec<Domain>> {
    let mut outers = Vec::new();
    let mut holes = Vec::new();
    for l in loops {
        let a = signed_area(&l.pts);
        if a.abs() < 1e-300 {
            return Err(PyrucastError::Message(
                "triangulate_surface: a boundary loop has zero area".into(),
            ));
        }
        if a > 0.0 {
            outers.push(l);
        } else {
            holes.push(l);
        }
    }
    if outers.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_surface: no counter-clockwise (outer) loop found".into(),
        ));
    }
    let mut domains: Vec<Domain> = outers
        .into_iter()
        .map(|o| Domain {
            outer: o,
            holes: Vec::new(),
        })
        .collect();
    'hole: for h in holes {
        let p = h.pts[0];
        for d in domains.iter_mut() {
            if point_in_polygon(p, &d.outer.pts) {
                d.holes.push(h);
                continue 'hole;
            }
        }
        return Err(PyrucastError::Message(
            "triangulate_surface: a hole (clockwise) loop is not contained in any outer loop"
                .into(),
        ));
    }
    Ok(domains)
}

// ─── Geometric primitives (2-D) ───────────────────────────────────────────

#[inline]
fn orient(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[inline]
fn in_circle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;
    let ad = ax * ax + ay * ay;
    let bd = bx * bx + by * by;
    let cd = cx * cx + cy * cy;
    ax * (by * cd - bd * cy) - ay * (bx * cd - bd * cx) + ad * (bx * cy - by * cx)
}

fn circumcenter(a: Point2, b: Point2, c: Point2) -> Point2 {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y))
        / d;
    let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x))
        / d;
    Point2::new(ux, uy)
}

fn bbox(pts: &[Point2]) -> (Point2, Point2) {
    let mut lo = Point2::new(f64::INFINITY, f64::INFINITY);
    let mut hi = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in pts {
        lo.x = lo.x.min(p.x);
        lo.y = lo.y.min(p.y);
        hi.x = hi.x.max(p.x);
        hi.y = hi.y.max(p.y);
    }
    (lo, hi)
}

/// Small deterministic per-index jitter (components in `[-0.5, 0.5]`), used
/// to break cocircular/collinear ambiguities in connectivity decisions.
fn jitter2(i: usize) -> Vector2 {
    let h = |k: u64| -> f64 {
        let x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ k.wrapping_mul(0x632BE5AB);
        ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    Vector2::new(h(1), h(2))
}

/// Minimum interior angle (radians) of triangle `(a, b, c)`; `0` if
/// degenerate. Used by the angle-improvement flip pass.
fn tri_min_angle(a: Point2, b: Point2, c: Point2) -> f64 {
    let ab = (b - a).norm();
    let bc = (c - b).norm();
    let ca = (a - c).norm();
    if ab == 0.0 || bc == 0.0 || ca == 0.0 {
        return 0.0;
    }
    let cos_a = ((ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca)).clamp(-1.0, 1.0);
    let cos_b = ((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc)).clamp(-1.0, 1.0);
    let cos_c = ((bc * bc + ca * ca - ab * ab) / (2.0 * bc * ca)).clamp(-1.0, 1.0);
    cos_a.max(cos_b).max(cos_c).acos()
}

fn point_in_tri_strict(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    let o1 = orient(a, b, p);
    let o2 = orient(b, c, p);
    let o3 = orient(c, a, p);
    (o1 > 0.0 && o2 > 0.0 && o3 > 0.0) || (o1 < 0.0 && o2 < 0.0 && o3 < 0.0)
}

/// Ear-clip a simple polygon (vertex indices into `pts`) into triangles.
fn ear_clip(poly: &[u32], pts: &[Point2]) -> Vec<[u32; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut area = 0.0;
    for i in 0..n {
        let p = pts[poly[i] as usize];
        let q = pts[poly[(i + 1) % n] as usize];
        area += p.x * q.y - q.x * p.y;
    }
    if area < 0.0 {
        idx.reverse();
    }
    let mut tris = Vec::new();
    let mut guard = 0usize;
    while idx.len() > 3 {
        guard += 1;
        if guard > n * n + 16 {
            break;
        }
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let prev = idx[(i + m - 1) % m];
            let cur = idx[i];
            let next = idx[(i + 1) % m];
            let (a, b, c) = (poly[prev], poly[cur], poly[next]);
            let (pa, pb, pc) = (pts[a as usize], pts[b as usize], pts[c as usize]);
            if orient(pa, pb, pc) <= 0.0 {
                continue;
            }
            let mut ok = true;
            for &k in &idx {
                if k == prev || k == cur || k == next {
                    continue;
                }
                if point_in_tri_strict(pts[poly[k] as usize], pa, pb, pc) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            tris.push([a, b, c]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push([poly[idx[0]], poly[idx[1]], poly[idx[2]]]);
    }
    tris
}

// ─── Constrained Delaunay triangulation engine ────────────────────────────

#[derive(Clone, Copy)]
struct Tri {
    v: [u32; 3],
    nbr: [i32; 3],
    cons: [bool; 3],
    dead: bool,
}

impl Default for Tri {
    fn default() -> Self {
        Tri {
            v: [0, 0, 0],
            nbr: [-1, -1, -1],
            cons: [false, false, false],
            dead: false,
        }
    }
}

struct Cdt {
    pts: Vec<Point2>,
    jpts: Vec<Point2>,
    is_boundary: Vec<bool>,
    inside: Vec<bool>,
    tris: Vec<Tri>,
    free: Vec<usize>,
    hint: usize,
    jitter_amp: f64,
    super_base: usize,
}

impl Cdt {
    /// Bootstrap with a super-triangle enclosing every point in `pts`, ready
    /// for unconstrained incremental insertion of those `pts.len()` points.
    fn new(mut pts: Vec<Point2>, mut is_boundary: Vec<bool>, jitter_amp: f64) -> Self {
        let base = pts.len();
        let (lo, hi) = bbox(&pts);
        let dx = (hi.x - lo.x).max(1e-12);
        let dy = (hi.y - lo.y).max(1e-12);
        let dmax = dx.max(dy);
        let cx = (lo.x + hi.x) * 0.5;
        let cy = (lo.y + hi.y) * 0.5;
        let r = dmax * 1000.0 + 1.0;
        pts.push(Point2::new(cx - 2.0 * r, cy - r));
        pts.push(Point2::new(cx + 2.0 * r, cy - r));
        pts.push(Point2::new(cx, cy + 2.0 * r));
        is_boundary.push(false);
        is_boundary.push(false);
        is_boundary.push(false);
        let jpts: Vec<Point2> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| p + jitter2(i) * jitter_amp)
            .collect();
        let t0 = Tri {
            v: [base as u32, (base + 1) as u32, (base + 2) as u32],
            ..Tri::default()
        };
        Self {
            pts,
            jpts,
            is_boundary,
            inside: vec![false],
            tris: vec![t0],
            free: Vec::new(),
            hint: 0,
            jitter_amp,
            super_base: base,
        }
    }

    fn add_point(&mut self, p: Point2, is_bnd: bool) -> usize {
        let idx = self.pts.len();
        self.pts.push(p);
        self.jpts.push(p + jitter2(idx) * self.jitter_amp);
        self.is_boundary.push(is_bnd);
        idx
    }

    fn alloc_tri(&mut self) -> usize {
        let idx = match self.free.pop() {
            Some(idx) => idx,
            None => {
                self.tris.push(Tri::default());
                self.tris.len() - 1
            }
        };
        if idx >= self.inside.len() {
            self.inside.resize(idx + 1, false);
        } else {
            self.inside[idx] = false;
        }
        idx
    }

    fn local_edge_index(&self, ti: usize, u: u32, v: u32) -> Option<usize> {
        let t = &self.tris[ti];
        (0..3).find(|&e| t.v[e] == u && t.v[(e + 1) % 3] == v)
    }

    fn find_directed_edge(&self, a: u32, b: u32) -> Option<(usize, usize)> {
        for (i, t) in self.tris.iter().enumerate() {
            if t.dead {
                continue;
            }
            if let Some(e) = (0..3).find(|&e| t.v[e] == a && t.v[(e + 1) % 3] == b) {
                return Some((i, e));
            }
        }
        None
    }

    /// `new_idx` now owns directed edge `(u, v)`; point the outside
    /// triangle's matching reverse edge `(v, u)` at it.
    fn restitch(&mut self, outside: i32, u: u32, v: u32, new_idx: usize) {
        if outside < 0 {
            return;
        }
        let o = outside as usize;
        if self.tris[o].dead {
            return;
        }
        if let Some(e) = self.local_edge_index(o, v, u) {
            self.tris[o].nbr[e] = new_idx as i32;
        }
    }

    /// Visibility walk from `start` towards `p` (jittered space). Falls back
    /// to a linear scan if the walk doesn't converge quickly (rare).
    fn locate(&self, mut t: usize, p: Point2) -> usize {
        if self.tris[t].dead {
            t = (0..self.tris.len())
                .find(|&i| !self.tris[i].dead)
                .unwrap_or(t);
        }
        let cap = self.tris.len() * 4 + 64;
        for _ in 0..cap {
            if self.tris[t].dead {
                t = (0..self.tris.len())
                    .find(|&i| !self.tris[i].dead)
                    .unwrap_or(t);
                continue;
            }
            let tri = self.tris[t];
            let (a, b, c) = (
                self.jpts[tri.v[0] as usize],
                self.jpts[tri.v[1] as usize],
                self.jpts[tri.v[2] as usize],
            );
            let o0 = orient(a, b, p);
            let o1 = orient(b, c, p);
            let o2 = orient(c, a, p);
            if o0 >= 0.0 && o1 >= 0.0 && o2 >= 0.0 {
                return t;
            }
            let (edge, _) = [o0, o1, o2]
                .iter()
                .enumerate()
                .min_by(|x, y| x.1.total_cmp(y.1))
                .unwrap();
            let n = tri.nbr[edge];
            if n < 0 {
                return t;
            }
            t = n as usize;
        }
        // Fallback: linear scan for a containing (or closest) triangle.
        for (i, tri) in self.tris.iter().enumerate() {
            if tri.dead {
                continue;
            }
            let (a, b, c) = (
                self.jpts[tri.v[0] as usize],
                self.jpts[tri.v[1] as usize],
                self.jpts[tri.v[2] as usize],
            );
            if orient(a, b, p) >= -1e-12 && orient(b, c, p) >= -1e-12 && orient(c, a, p) >= -1e-12 {
                return i;
            }
        }
        t
    }

    /// Split triangle `t` (which strictly contains `p_idx`) into three,
    /// reusing slot `t`. Returns the three slots (all incident to `p_idx`).
    fn split_tri3(&mut self, t: usize, p_idx: usize) -> [usize; 3] {
        let tri = self.tris[t];
        let (a, b, c) = (tri.v[0], tri.v[1], tri.v[2]);
        let p = p_idx as u32;
        let t0 = t;
        let t1 = self.alloc_tri();
        let t2 = self.alloc_tri();
        self.tris[t0] = Tri {
            v: [a, b, p],
            nbr: [tri.nbr[0], t1 as i32, t2 as i32],
            cons: [tri.cons[0], false, false],
            dead: false,
        };
        self.tris[t1] = Tri {
            v: [b, c, p],
            nbr: [tri.nbr[1], t2 as i32, t0 as i32],
            cons: [tri.cons[1], false, false],
            dead: false,
        };
        self.tris[t2] = Tri {
            v: [c, a, p],
            nbr: [tri.nbr[2], t0 as i32, t1 as i32],
            cons: [tri.cons[2], false, false],
            dead: false,
        };
        self.restitch(tri.nbr[0], a, b, t0);
        self.restitch(tri.nbr[1], b, c, t1);
        self.restitch(tri.nbr[2], c, a, t2);
        [t0, t1, t2]
    }

    /// Split the (non-constrained, two-sided) edge `e` of triangle `t` at
    /// `p_idx` lying on it, replacing the two adjacent triangles by four.
    /// Returns the four slots (all incident to `p_idx`).
    fn split_edge_2(&mut self, t: usize, e: usize, p_idx: usize) -> [usize; 4] {
        let ti = self.tris[t];
        let u = ti.v[e];
        let v = ti.v[(e + 1) % 3];
        let w = ti.v[(e + 2) % 3];
        let n_vw = ti.nbr[(e + 1) % 3];
        let c_vw = ti.cons[(e + 1) % 3];
        let n_wu = ti.nbr[(e + 2) % 3];
        let c_wu = ti.cons[(e + 2) % 3];
        let tj = ti.nbr[e] as usize;
        let ej = self
            .local_edge_index(tj, v, u)
            .expect("split_edge_2: inconsistent adjacency");
        let tjt = self.tris[tj];
        let x = tjt.v[(ej + 2) % 3];
        let n_ux = tjt.nbr[(ej + 1) % 3];
        let c_ux = tjt.cons[(ej + 1) % 3];
        let n_xv = tjt.nbr[(ej + 2) % 3];
        let c_xv = tjt.cons[(ej + 2) % 3];
        let p = p_idx as u32;

        self.tris[t].dead = true;
        self.free.push(t);
        self.tris[tj].dead = true;
        self.free.push(tj);
        let a = self.alloc_tri();
        let b = self.alloc_tri();
        let c = self.alloc_tri();
        let d = self.alloc_tri();
        // A=(u,p,w) B=(p,v,w) C=(v,p,x) D=(p,u,x)
        self.tris[a] = Tri {
            v: [u, p, w],
            nbr: [d as i32, b as i32, n_wu],
            cons: [false, false, c_wu],
            dead: false,
        };
        self.tris[b] = Tri {
            v: [p, v, w],
            nbr: [c as i32, n_vw, a as i32],
            cons: [false, c_vw, false],
            dead: false,
        };
        self.tris[c] = Tri {
            v: [v, p, x],
            nbr: [b as i32, d as i32, n_xv],
            cons: [false, false, c_xv],
            dead: false,
        };
        self.tris[d] = Tri {
            v: [p, u, x],
            nbr: [a as i32, n_ux, c as i32],
            cons: [false, c_ux, false],
            dead: false,
        };
        self.restitch(n_wu, w, u, a);
        self.restitch(n_vw, v, w, b);
        self.restitch(n_xv, x, v, c);
        self.restitch(n_ux, u, x, d);
        [a, b, c, d]
    }

    /// Insert point `p_idx` (already appended via [`Cdt::add_point`]) by the
    /// classic locate → split → Lawson-flip scheme. Every flip keeps `p_idx`
    /// a vertex of both resulting triangles, so the star of `p_idx` stays a
    /// simple fan (no pinched cavity). Never flips a constrained edge.
    /// Returns every triangle currently incident to `p_idx`.
    fn insert_point(
        &mut self,
        p_idx: usize,
        start_hint: usize,
        cancel: &dyn Cancel,
    ) -> Result<Vec<usize>> {
        cancel.check()?;
        let p = self.jpts[p_idx];
        let t = self.locate(start_hint, p);
        let tri = self.tris[t];
        let (ja, jb, jc) = (
            self.jpts[tri.v[0] as usize],
            self.jpts[tri.v[1] as usize],
            self.jpts[tri.v[2] as usize],
        );
        let o = [orient(ja, jb, p), orient(jb, jc, p), orient(jc, ja, p)];
        let (emin, &omin) = o
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.total_cmp(y.1))
            .unwrap();

        let seed: Vec<usize> = if omin > 0.0 {
            self.split_tri3(t, p_idx).to_vec()
        } else if self.tris[t].nbr[emin] >= 0 && !self.tris[t].cons[emin] {
            self.split_edge_2(t, emin, p_idx).to_vec()
        } else {
            // On a hull/constrained edge: fall back to a 3-split (a degenerate
            // sliver may appear; legalization and later smoothing absorb it).
            self.split_tri3(t, p_idx).to_vec()
        };

        let mut incident = seed.clone();
        let mut stack = seed;
        // Lawson flips terminate in a valid triangulation; the cap is a
        // floating-point safety net against a non-convergent cycle.
        let mut guard = 0usize;
        let cap = self.tris.len() * 8 + 256;
        while let Some(ti) = stack.pop() {
            guard += 1;
            if guard > cap {
                break;
            }
            if self.tris[ti].dead {
                continue;
            }
            let l = match self.tris[ti].v.iter().position(|&x| x == p_idx as u32) {
                Some(l) => l,
                None => continue,
            };
            let e = (l + 1) % 3; // edge opposite p
            if self.tris[ti].cons[e] {
                continue;
            }
            let nb = self.tris[ti].nbr[e];
            if nb < 0 {
                continue;
            }
            let nb = nb as usize;
            let u = self.tris[ti].v[e];
            let v = self.tris[ti].v[(e + 1) % 3];
            let ej = match self.local_edge_index(nb, v, u) {
                Some(x) => x,
                None => continue,
            };
            let w = self.tris[nb].v[(ej + 2) % 3];
            let (ju, jv, jw) = (
                self.jpts[u as usize],
                self.jpts[v as usize],
                self.jpts[w as usize],
            );
            if in_circle(ju, jv, p, jw) <= 0.0 {
                continue; // locally Delaunay
            }
            // Convexity guard: only flip when the quad (p, u, w, v) is convex,
            // so the flip can't create an inverted triangle and cycle forever.
            let (rp, ru, rv, rw) = (
                self.pts[p_idx],
                self.pts[u as usize],
                self.pts[v as usize],
                self.pts[w as usize],
            );
            if !(orient(rp, ru, rw) > 0.0 && orient(rp, rw, rv) > 0.0) {
                continue;
            }
            let (n1, n2) = self.flip_edge(ti, e);
            for nt in [n1, n2] {
                if !incident.contains(&nt) {
                    incident.push(nt);
                }
                stack.push(nt);
            }
        }
        if let Some(&last) = incident.last() {
            self.hint = last;
        }
        Ok(incident)
    }

    fn mark_constrained_if_present(&mut self, a: u32, b: u32) -> bool {
        let found_ab = self.find_directed_edge(a, b);
        let found_ba = self.find_directed_edge(b, a);
        if let Some((i, e)) = found_ab {
            self.tris[i].cons[e] = true;
        }
        if let Some((j, e2)) = found_ba {
            self.tris[j].cons[e2] = true;
        }
        found_ab.is_some() && found_ba.is_some()
    }

    /// Walk the corridor of triangles the open segment `(a, b)` passes
    /// through, returning `(crossed triangles, upper chain, lower chain)`.
    /// Both chains run `a -> ... -> b`; together with the diagonal `(a, b)`
    /// each bounds a simple polygon on one side of the recovered edge.
    fn corridor(&self, a: u32, b: u32) -> Result<(Vec<usize>, Vec<u32>, Vec<u32>)> {
        let pa = self.jpts[a as usize];
        let pb = self.jpts[b as usize];
        let mut start_tri: Option<(usize, u32, u32)> = None;
        'outer: for (i, t) in self.tris.iter().enumerate() {
            if t.dead {
                continue;
            }
            for k in 0..3 {
                if t.v[k] != a {
                    continue;
                }
                let m = t.v[(k + 1) % 3];
                let n = t.v[(k + 2) % 3];
                let om = orient(pa, self.jpts[m as usize], pb);
                let on = orient(pa, self.jpts[n as usize], pb);
                if om >= 0.0 && on <= 0.0 {
                    start_tri = Some((i, m, n));
                    break 'outer;
                }
            }
        }
        let (mut cur, mut p, mut q) = start_tri.ok_or_else(|| {
            PyrucastError::Message(
                "triangulate_surface: could not recover a boundary edge (degenerate contour?)"
                    .into(),
            )
        })?;
        let mut crossed = vec![cur];
        let mut upper = vec![a, p];
        let mut lower = vec![a, q];
        let mut le = self.local_edge_index(cur, p, q).ok_or_else(|| {
            PyrucastError::Message(
                "triangulate_surface: internal error recovering a boundary edge".into(),
            )
        })?;
        loop {
            let nb = self.tris[cur].nbr[le];
            if nb < 0 {
                return Err(PyrucastError::Message(
                    "triangulate_surface: boundary edge recovery ran off the mesh (degenerate contour?)"
                        .into(),
                ));
            }
            let nb = nb as usize;
            let nt = self.tris[nb];
            let r =
                nt.v.iter()
                    .copied()
                    .find(|&x| x != p && x != q)
                    .ok_or_else(|| {
                        PyrucastError::Message(
                            "triangulate_surface: internal error recovering a boundary edge".into(),
                        )
                    })?;
            crossed.push(nb);
            if r == b {
                break;
            }
            let side = orient(pa, pb, self.jpts[r as usize]);
            if side > 0.0 {
                upper.push(r);
                p = r;
            } else {
                lower.push(r);
                q = r;
            }
            le = self.local_edge_index(nb, p, q).ok_or_else(|| {
                PyrucastError::Message(
                    "triangulate_surface: internal error recovering a boundary edge".into(),
                )
            })?;
            cur = nb;
        }
        upper.push(b);
        lower.push(b);
        Ok((crossed, upper, lower))
    }

    /// Recover boundary edge `(a, b)` if it isn't already present: remove
    /// the crossed corridor and ear-clip the two flanking polygons. Doesn't
    /// bother fixing neighbor pointers or constraint flags — callers rebuild
    /// those globally afterward (cheap: recovery only ever runs over the
    /// small boundary point set).
    fn recover_edge(&mut self, a: u32, b: u32) -> Result<()> {
        if self.find_directed_edge(a, b).is_some() || self.find_directed_edge(b, a).is_some() {
            return Ok(());
        }
        let (crossed, upper, lower) = self.corridor(a, b)?;
        for &ti in &crossed {
            self.tris[ti].dead = true;
            self.free.push(ti);
        }
        let up_tris = ear_clip(&upper, &self.jpts);
        let lo_tris = ear_clip(&lower, &self.jpts);
        for tv in up_tris.into_iter().chain(lo_tris) {
            let idx = self.alloc_tri();
            self.tris[idx] = Tri {
                v: tv,
                ..Tri::default()
            };
        }
        Ok(())
    }

    fn rebuild_neighbors(&mut self) {
        let mut edge_map: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, t) in self.tris.iter().enumerate() {
            if t.dead {
                continue;
            }
            for e in 0..3 {
                edge_map.insert((t.v[e], t.v[(e + 1) % 3]), i);
            }
        }
        for i in 0..self.tris.len() {
            if self.tris[i].dead {
                continue;
            }
            for e in 0..3 {
                let u = self.tris[i].v[e];
                let v = self.tris[i].v[(e + 1) % 3];
                self.tris[i].nbr[e] = edge_map.get(&(v, u)).map(|&ti| ti as i32).unwrap_or(-1);
            }
        }
    }

    fn remove_super_triangle(&mut self) {
        let base = self.super_base as u32;
        for i in 0..self.tris.len() {
            if self.tris[i].dead {
                continue;
            }
            if self.tris[i].v.iter().any(|&v| v >= base) {
                self.tris[i].dead = true;
                self.free.push(i);
            }
        }
        self.pts.truncate(self.super_base);
        self.jpts.truncate(self.super_base);
        self.is_boundary.truncate(self.super_base);
    }

    /// Standard Delaunay diagonal flip of the edge `ti`'s local edge `e`
    /// (never called on a constrained edge). Reuses slots `ti`/`tj` in
    /// place; returns their (unchanged) indices.
    fn flip_edge(&mut self, ti: usize, e: usize) -> (usize, usize) {
        let tj = self.tris[ti].nbr[e] as usize;
        let t = self.tris[ti];
        let u = t.v[e];
        let v = t.v[(e + 1) % 3];
        let a = t.v[(e + 2) % 3];
        let ej = self
            .local_edge_index(tj, v, u)
            .expect("flip_edge: inconsistent adjacency");
        let tj_ = self.tris[tj];
        let b = tj_.v[(ej + 2) % 3];

        let out_au = t.nbr[(e + 2) % 3];
        let cons_au = t.cons[(e + 2) % 3];
        let out_va = t.nbr[(e + 1) % 3];
        let cons_va = t.cons[(e + 1) % 3];
        let out_ub = tj_.nbr[(ej + 1) % 3];
        let cons_ub = tj_.cons[(ej + 1) % 3];
        let out_bv = tj_.nbr[(ej + 2) % 3];
        let cons_bv = tj_.cons[(ej + 2) % 3];

        self.tris[ti] = Tri {
            v: [a, u, b],
            nbr: [out_au, out_ub, tj as i32],
            cons: [cons_au, cons_ub, false],
            dead: false,
        };
        self.tris[tj] = Tri {
            v: [a, b, v],
            nbr: [ti as i32, out_bv, out_va],
            cons: [false, cons_bv, cons_va],
            dead: false,
        };

        self.restitch(out_au, a, u, ti);
        self.restitch(out_ub, u, b, ti);
        self.restitch(out_bv, b, v, tj);
        self.restitch(out_va, v, a, tj);
        (ti, tj)
    }

    /// Delaunay-legalize every non-constrained edge (bounded Lawson-flip
    /// sweep); never touches a constrained edge, so domain boundaries are
    /// preserved exactly.
    fn legalize(&mut self) {
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        for i in 0..self.tris.len() {
            if self.tris[i].dead {
                continue;
            }
            for e in 0..3 {
                queue.push_back((i, e));
            }
        }
        let cap = self.tris.len() * 20 + 100;
        let mut guard = 0usize;
        while let Some((ti, e)) = queue.pop_front() {
            guard += 1;
            if guard > cap {
                break;
            }
            if self.tris[ti].dead || self.tris[ti].cons[e] {
                continue;
            }
            let nb = self.tris[ti].nbr[e];
            if nb < 0 {
                continue;
            }
            let nb = nb as usize;
            if self.tris[nb].dead {
                continue;
            }
            let t = self.tris[ti];
            let u = t.v[e];
            let v = t.v[(e + 1) % 3];
            let apex_t = t.v[(e + 2) % 3];
            let ej = match self.local_edge_index(nb, v, u) {
                Some(x) => x,
                None => continue,
            };
            let apex_n = self.tris[nb].v[(ej + 2) % 3];
            let (ju, jv, ja, jb) = (
                self.jpts[u as usize],
                self.jpts[v as usize],
                self.jpts[apex_t as usize],
                self.jpts[apex_n as usize],
            );
            if in_circle(ju, jv, ja, jb) <= 0.0 {
                continue;
            }
            let (pu, pv, pa, pb) = (
                self.pts[u as usize],
                self.pts[v as usize],
                self.pts[apex_t as usize],
                self.pts[apex_n as usize],
            );
            if !(orient(pa, pu, pb) > 0.0 && orient(pa, pb, pv) > 0.0) {
                continue; // flip would invert/degenerate a triangle
            }
            let (nt1, nt2) = self.flip_edge(ti, e);
            for tt in [nt1, nt2] {
                for ee in 0..3 {
                    queue.push_back((tt, ee));
                }
            }
        }
    }

    fn flood_fill(&mut self, seed: usize, cancel: &dyn Cancel) -> Result<()> {
        let mut stack = vec![seed];
        self.inside[seed] = true;
        while let Some(ti) = stack.pop() {
            cancel.check()?;
            let t = self.tris[ti];
            for e in 0..3 {
                if t.cons[e] {
                    continue;
                }
                let nb = t.nbr[e];
                if nb < 0 {
                    continue;
                }
                let nb = nb as usize;
                if !self.inside[nb] {
                    self.inside[nb] = true;
                    stack.push(nb);
                }
            }
        }
        Ok(())
    }

    fn quality(&self, ti: usize) -> (f64, f64, f64) {
        let t = &self.tris[ti];
        let p = [
            self.pts[t.v[0] as usize],
            self.pts[t.v[1] as usize],
            self.pts[t.v[2] as usize],
        ];
        let ab = (p[1] - p[0]).norm();
        let bc = (p[2] - p[1]).norm();
        let ca = (p[0] - p[2]).norm();
        let max_edge = ab.max(bc).max(ca);
        let min_edge = ab.min(bc).min(ca);
        let cos_a = ((ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca)).clamp(-1.0, 1.0);
        let cos_b = ((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc)).clamp(-1.0, 1.0);
        let cos_c = ((bc * bc + ca * ca - ab * ab) / (2.0 * bc * ca)).clamp(-1.0, 1.0);
        let min_angle = cos_a.max(cos_b).max(cos_c).acos();
        (min_angle, max_edge, min_edge)
    }

    /// Find a constrained subsegment that point `p` **encroaches** (lies
    /// strictly inside its diametral circle, i.e. subtends an obtuse angle at
    /// `p`), scanning the constrained edges of `seed_a`, `seed_b` and their
    /// immediate neighbors. Returns the most deeply encroached one. Local by
    /// design — a circumcenter only ever encroaches a subsegment bordering the
    /// triangle it falls in or an adjacent one — so the scan stays O(1).
    fn find_encroached(&self, p: Point2, seed_a: usize, seed_b: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, f64)> = None;
        let visit = |cti: usize, best: &mut Option<(usize, usize, f64)>| {
            let t = self.tris[cti];
            if t.dead {
                return;
            }
            for e in 0..3 {
                if !t.cons[e] {
                    continue;
                }
                let pu = self.pts[t.v[e] as usize];
                let pv = self.pts[t.v[(e + 1) % 3] as usize];
                let dot = (p - pu).dot(&(p - pv));
                if dot < 0.0 && best.is_none_or(|(_, _, d)| -dot > d) {
                    *best = Some((cti, e, -dot));
                }
            }
        };
        for &s in &[seed_a, seed_b] {
            if self.tris[s].dead {
                continue;
            }
            visit(s, &mut best);
            for nb in self.tris[s].nbr {
                if nb >= 0 {
                    visit(nb as usize, &mut best);
                }
            }
        }
        best.map(|(a, b, _)| (a, b))
    }

    /// Min-angle optimization: flip a non-constrained interior edge whenever
    /// doing so raises the smaller of the two incident triangles' minimum
    /// angles. Unlike Delaunay legalization (which optimizes the in-circle
    /// criterion), this directly targets the worst angle, dissolving slivers
    /// the Delaunay mesh leaves behind. Bounded pass count (flips can cycle
    /// at ties, so a strict-improvement margin plus a cap guarantee halting).
    fn improve_angles(&mut self, passes: usize) {
        for _ in 0..passes {
            let mut changed = false;
            for ti in 0..self.tris.len() {
                if self.tris[ti].dead || !self.inside[ti] {
                    continue;
                }
                for e in 0..3 {
                    if self.tris[ti].cons[e] {
                        continue;
                    }
                    let nb = self.tris[ti].nbr[e];
                    if nb < 0 {
                        continue;
                    }
                    let nb = nb as usize;
                    if self.tris[nb].dead || !self.inside[nb] {
                        continue;
                    }
                    let t = self.tris[ti];
                    let u = t.v[e];
                    let v = t.v[(e + 1) % 3];
                    let a = t.v[(e + 2) % 3];
                    let ej = match self.local_edge_index(nb, v, u) {
                        Some(x) => x,
                        None => continue,
                    };
                    let b = self.tris[nb].v[(ej + 2) % 3];
                    let (pu, pv, pa, pb) = (
                        self.pts[u as usize],
                        self.pts[v as usize],
                        self.pts[a as usize],
                        self.pts[b as usize],
                    );
                    // Flip only across a convex quad (else it would invert).
                    if !(orient(pa, pu, pb) > 0.0 && orient(pa, pb, pv) > 0.0) {
                        continue;
                    }
                    let cur = tri_min_angle(pu, pv, pa).min(tri_min_angle(pv, pu, pb));
                    let flipped = tri_min_angle(pa, pu, pb).min(tri_min_angle(pa, pb, pv));
                    if flipped > cur + 1e-6 {
                        self.flip_edge(ti, e);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Split constrained edge `ti`'s local edge `e` at new point `m`,
    /// fanning both adjacent triangles into two. Returns the four new
    /// triangle indices `(a,m,p)`, `(m,b,p)`, `(b,m,q)`, `(m,a,q)`.
    /// The `q`-side (across the constrained edge) may not exist at all — a
    /// domain edge that lies on the convex hull of the whole point set (e.g.
    /// every edge of a plain square) has no outside triangle — in which case
    /// only `(i1, i2)` are created and the third return value is `None`.
    fn split_constrained_edge(
        &mut self,
        ti: usize,
        e: usize,
        m: u32,
    ) -> (usize, usize, Option<(usize, usize)>) {
        let t = self.tris[ti];
        let a = t.v[e];
        let b = t.v[(e + 1) % 3];
        let p = t.v[(e + 2) % 3];
        let tj_raw = t.nbr[e];

        let out_bp = t.nbr[(e + 1) % 3];
        let cons_bp = t.cons[(e + 1) % 3];
        let out_pa = t.nbr[(e + 2) % 3];
        let cons_pa = t.cons[(e + 2) % 3];

        self.tris[ti].dead = true;
        self.free.push(ti);

        let i1 = self.alloc_tri();
        let i2 = self.alloc_tri();
        self.tris[i1] = Tri {
            v: [a, m, p],
            nbr: [-1, i2 as i32, out_pa],
            cons: [true, false, cons_pa],
            dead: false,
        };
        self.tris[i2] = Tri {
            v: [m, b, p],
            nbr: [-1, out_bp, i1 as i32],
            cons: [true, cons_bp, false],
            dead: false,
        };
        self.restitch(out_pa, p, a, i1);
        self.restitch(out_bp, b, p, i2);

        let tj_raw = if tj_raw < 0 {
            return (i1, i2, None);
        } else {
            tj_raw
        };
        let tj = tj_raw as usize;
        let ej = self
            .local_edge_index(tj, b, a)
            .expect("split_constrained_edge: inconsistent adjacency");
        let tj_ = self.tris[tj];
        let q = tj_.v[(ej + 2) % 3];
        let out_aq = tj_.nbr[(ej + 1) % 3];
        let cons_aq = tj_.cons[(ej + 1) % 3];
        let out_qb = tj_.nbr[(ej + 2) % 3];
        let cons_qb = tj_.cons[(ej + 2) % 3];

        self.tris[tj].dead = true;
        self.free.push(tj);

        let i3 = self.alloc_tri();
        let i4 = self.alloc_tri();
        self.tris[i3] = Tri {
            v: [b, m, q],
            nbr: [i2 as i32, i4 as i32, out_qb],
            cons: [true, false, cons_qb],
            dead: false,
        };
        self.tris[i4] = Tri {
            v: [m, a, q],
            nbr: [i1 as i32, out_aq, i3 as i32],
            cons: [true, cons_aq, false],
            dead: false,
        };
        self.tris[i1].nbr[0] = i4 as i32;
        self.tris[i2].nbr[0] = i3 as i32;

        self.restitch(out_qb, q, b, i3);
        self.restitch(out_aq, a, q, i4);

        (i1, i2, Some((i3, i4)))
    }

    /// Ruppert refinement of every `inside` triangle: split by circumcenter
    /// insertion, or split an encroached constrained edge at its midpoint
    /// instead (never inserting a point that would violate a boundary).
    /// Ruppert refinement. When `freeze_boundary` is set, no constrained
    /// (contour) edge is ever split: the boundary keeps exactly the input
    /// nodes. Any refinement step that would have bisected a boundary
    /// subsegment (a circumcenter encroaching it, or landing outside the
    /// domain) is skipped instead — trading a slightly worse triangle near
    /// that edge for an untouched contour.
    fn refine(&mut self, h: f64, freeze_boundary: bool, cancel: &dyn Cancel) -> Result<()> {
        let min_angle_rad = MIN_ANGLE_DEG.to_radians();
        let edge_floor = h * 1e-2;
        let mut worklist: VecDeque<usize> = VecDeque::new();
        let mut queued: Vec<bool> = vec![false; self.tris.len()];
        for i in 0..self.tris.len() {
            if !self.tris[i].dead && self.inside[i] {
                worklist.push_back(i);
                queued[i] = true;
            }
        }
        let max_new_points = (self.pts.len() * 100).max(20_000_000);
        let mut created = 0usize;
        while let Some(ti) = worklist.pop_front() {
            if ti < queued.len() {
                queued[ti] = false;
            }
            cancel.check()?;
            if self.tris[ti].dead || !self.inside[ti] {
                continue;
            }
            let (min_ang, max_edge, shortest_edge) = self.quality(ti);
            let bad_size = max_edge > h;
            let bad_angle = min_ang < min_angle_rad;
            if !bad_size && !bad_angle {
                continue;
            }
            if bad_angle && !bad_size && shortest_edge < edge_floor {
                continue; // can't safely shrink this any further
            }
            if created > max_new_points {
                return Err(PyrucastError::Message(
                    "triangulate_surface: refinement did not converge within the point budget"
                        .into(),
                ));
            }
            let t = self.tris[ti];
            let (pa, pb, pc) = (
                self.pts[t.v[0] as usize],
                self.pts[t.v[1] as usize],
                self.pts[t.v[2] as usize],
            );
            if orient(pa, pb, pc).abs() < 1e-24 {
                continue; // degenerate triangle: no usable circumcenter
            }
            let cc = circumcenter(pa, pb, pc);
            if !cc.x.is_finite() || !cc.y.is_finite() {
                continue;
            }

            // Where does the circumcenter land? `locate` is robust (edge walk
            // plus a linear-scan fallback) and always returns a triangle of the
            // covered region. `land_ok` = it sits in an interior triangle, i.e.
            // strictly inside the domain.
            let land = self.locate(ti, cc);
            let land_ok = !self.tris[land].dead && self.inside[land] && {
                let lt = self.tris[land];
                let (a, b, c) = (
                    self.pts[lt.v[0] as usize],
                    self.pts[lt.v[1] as usize],
                    self.pts[lt.v[2] as usize],
                );
                orient(a, b, cc) >= 0.0 && orient(b, c, cc) >= 0.0 && orient(c, a, cc) >= 0.0
            };

            // Ruppert's rule: a circumcenter that **encroaches** a boundary
            // subsegment (lies in its diametral circle) splits that subsegment
            // rather than being inserted — even when it lands inside the
            // domain. This is what refines the boundary itself (a long boundary
            // edge whose circumcenter sits in its diametral circle gets
            // bisected). Scanned locally around `ti` and `land`.
            if let Some((cti, ce)) = self.find_encroached(cc, ti, land) {
                if freeze_boundary {
                    // Contour is frozen: never bisect the boundary. Drop this
                    // circumcenter rather than splitting the subsegment it
                    // encroaches.
                    continue;
                }
                let ct = self.tris[cti];
                let (u, v) = (ct.v[ce], ct.v[(ce + 1) % 3]);
                let mid =
                    Point2::from((self.pts[u as usize].coords + self.pts[v as usize].coords) * 0.5);
                let was_inside = self.inside[cti];
                let m_idx = self.add_point(mid, true);
                let (i1, i2, other) = self.split_constrained_edge(cti, ce, m_idx as u32);
                self.inside[i1] = was_inside;
                self.inside[i2] = was_inside;
                let mut touched = vec![i1, i2];
                if let Some((i3, i4)) = other {
                    self.inside[i3] = !was_inside;
                    self.inside[i4] = !was_inside;
                    touched.push(i3);
                    touched.push(i4);
                }
                // The triangle we were refining may still be bad — requeue it.
                if !self.tris[ti].dead && self.inside[ti] {
                    touched.push(ti);
                }
                created += 1;
                for idx in touched {
                    if self.inside[idx] {
                        if idx >= queued.len() {
                            queued.resize(idx + 1, false);
                        }
                        if !queued[idx] {
                            worklist.push_back(idx);
                            queued[idx] = true;
                        }
                    }
                }
            } else if land_ok {
                // Circumcenter is inside the domain — insert it, unless it would
                // sit too close to an existing vertex (Ruppert "concentric
                // shell": near a small input angle this cascades into slivers).
                let r = (cc - pa).norm();
                let lt = self.tris[land];
                let mut dmin = f64::INFINITY;
                for ct in std::iter::once(land).chain(
                    lt.nbr
                        .iter()
                        .copied()
                        .filter(|&n| n >= 0)
                        .map(|n| n as usize),
                ) {
                    if self.tris[ct].dead {
                        continue;
                    }
                    for &vv in &self.tris[ct].v {
                        let d = (cc - self.pts[vv as usize]).norm();
                        if d < dmin {
                            dmin = d;
                        }
                    }
                }
                if dmin < 0.5 * r {
                    continue;
                }
                let p_idx = self.add_point(cc, false);
                let new_tris = self.insert_point(p_idx, land, cancel)?;
                created += 1;
                for idx in new_tris {
                    if self.tris[idx].dead {
                        continue;
                    }
                    self.inside[idx] = true;
                    if idx >= queued.len() {
                        queued.resize(idx + 1, false);
                    }
                    if !queued[idx] {
                        worklist.push_back(idx);
                        queued[idx] = true;
                    }
                }
            } else if freeze_boundary {
                // Contour is frozen: a circumcenter outside the domain would
                // normally split the boundary it hides behind. Skip it.
                continue;
            } else {
                // Circumcenter is outside the domain (a hole, the exterior, or
                // beyond a concave boundary): it encroaches a boundary
                // subsegment. Split the constrained edge the straight path
                // `ti -> cc` crosses; if the walk finds none, split `ti`'s
                // longest constrained edge; if `ti` has none, skip.
                let split = match self.walk_toward(ti, cc) {
                    WalkEnd::Blocked(cti, ce) => Some((cti, ce)),
                    WalkEnd::Inside => {
                        let t = self.tris[ti];
                        let elen = |e: usize| {
                            (self.pts[t.v[e] as usize] - self.pts[t.v[(e + 1) % 3] as usize]).norm()
                        };
                        (0..3)
                            .filter(|&e| t.cons[e])
                            .max_by(|&a, &b| elen(a).total_cmp(&elen(b)))
                            .map(|e| (ti, e))
                    }
                };
                let (cti, ce) = match split {
                    Some(x) => x,
                    None => continue,
                };
                let ct = self.tris[cti];
                let (u, v) = (ct.v[ce], ct.v[(ce + 1) % 3]);
                let mid =
                    Point2::from((self.pts[u as usize].coords + self.pts[v as usize].coords) * 0.5);
                let was_inside = self.inside[cti];
                let m_idx = self.add_point(mid, true);
                let (i1, i2, other) = self.split_constrained_edge(cti, ce, m_idx as u32);
                self.inside[i1] = was_inside;
                self.inside[i2] = was_inside;
                let mut touched = vec![i1, i2];
                if let Some((i3, i4)) = other {
                    self.inside[i3] = !was_inside;
                    self.inside[i4] = !was_inside;
                    touched.push(i3);
                    touched.push(i4);
                }
                created += 1;
                for idx in touched {
                    if self.inside[idx] {
                        if idx >= queued.len() {
                            queued.resize(idx + 1, false);
                        }
                        if !queued[idx] {
                            worklist.push_back(idx);
                            queued[idx] = true;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Walk from triangle `start` toward point `target` (real coordinates),
    /// crossing shared edges. Returns the triangle containing `target`, or
    /// the first constrained edge the straight path would cross (meaning
    /// `target` is occluded behind a domain boundary).
    fn walk_toward(&self, start: usize, target: Point2) -> WalkEnd {
        let mut cur = start;
        let ctr = |t: &Tri, pts: &[Point2]| {
            Point2::from(
                (pts[t.v[0] as usize].coords
                    + pts[t.v[1] as usize].coords
                    + pts[t.v[2] as usize].coords)
                    / 3.0,
            )
        };
        let cap = self.tris.len() * 4 + 64;
        for _ in 0..cap {
            let t = self.tris[cur];
            let (a, b, c) = (
                self.pts[t.v[0] as usize],
                self.pts[t.v[1] as usize],
                self.pts[t.v[2] as usize],
            );
            if orient(a, b, target) >= 0.0
                && orient(b, c, target) >= 0.0
                && orient(c, a, target) >= 0.0
            {
                return WalkEnd::Inside;
            }
            // Cross the edge whose far side the target lies on, stepping from
            // the triangle centroid toward the target.
            let src = ctr(&t, &self.pts);
            let mut stepped = false;
            for e in 0..3 {
                let u = self.pts[t.v[e] as usize];
                let v = self.pts[t.v[(e + 1) % 3] as usize];
                // Target is outside this edge, and the segment (src,target)
                // crosses the edge line on the outward side.
                if orient(u, v, target) < 0.0 && segments_straddle(src, target, u, v) {
                    if t.cons[e] {
                        return WalkEnd::Blocked(cur, e);
                    }
                    let nb = t.nbr[e];
                    if nb < 0 {
                        return WalkEnd::Inside;
                    }
                    cur = nb as usize;
                    stepped = true;
                    break;
                }
            }
            if !stepped {
                return WalkEnd::Inside;
            }
        }
        WalkEnd::Inside
    }
}

enum WalkEnd {
    /// The straight path reached `target` without crossing a constraint (the
    /// landing triangle is found separately via [`Cdt::locate`]).
    Inside,
    /// The path crosses the constrained edge `e` of triangle `ti`.
    Blocked(usize, usize),
}

/// Do open segments `(p1,p2)` and `(q1,q2)` properly straddle each other's
/// supporting lines? Used to pick the crossed edge in the visibility walk.
fn segments_straddle(p1: Point2, p2: Point2, q1: Point2, q2: Point2) -> bool {
    let d1 = orient(q1, q2, p1);
    let d2 = orient(q1, q2, p2);
    let d3 = orient(p1, p2, q1);
    let d4 = orient(p1, p2, q2);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

// ─── Per-domain driver ─────────────────────────────────────────────────────

struct DomainResult {
    boundary_node_ids: Vec<NodeId>,
    pts: Vec<Point2>,
    n_boundary: usize,
    tris: Vec<[u32; 3]>,
    quads: Vec<[u32; 4]>,
    leftover_tris: Vec<[u32; 3]>,
}

fn collect_inside_tris(cdt: &Cdt) -> Vec<[u32; 3]> {
    cdt.tris
        .iter()
        .enumerate()
        .filter(|(i, t)| !t.dead && cdt.inside[*i])
        .map(|(_, t)| t.v)
        .collect()
}

fn mesh_domain(
    domain: &Domain,
    element_type: ElementType,
    target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<DomainResult> {
    let mut boundary_node_ids = Vec::new();
    let mut pts0: Vec<Point2> = Vec::new();
    let mut loop_ranges: Vec<(usize, usize)> = Vec::new();
    for l in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let start = pts0.len();
        boundary_node_ids.extend_from_slice(&l.node_ids);
        pts0.extend_from_slice(&l.pts);
        loop_ranges.push((start, l.pts.len()));
    }
    let n_boundary = pts0.len();

    let mut perim = 0.0;
    let mut nedge = 0usize;
    for &(start, len) in &loop_ranges {
        for i in 0..len {
            let a = pts0[start + i];
            let b = pts0[start + (i + 1) % len];
            perim += (b - a).norm();
            nedge += 1;
        }
    }
    let h = target_size.unwrap_or(perim / nedge.max(1) as f64);
    if h <= 0.0 || h.is_nan() {
        return Err(PyrucastError::Message(
            "triangulate_surface: could not determine a positive element size".into(),
        ));
    }

    let (lo, hi) = bbox(&pts0);
    let diag = ((hi.x - lo.x).powi(2) + (hi.y - lo.y).powi(2)).sqrt();
    if diag <= 0.0 {
        return Err(PyrucastError::Message(
            "triangulate_surface: contour is degenerate (zero extent)".into(),
        ));
    }
    let jitter_amp = diag * 1e-7;

    let mut cdt = Cdt::new(pts0, vec![true; n_boundary], jitter_amp);
    for i in 0..n_boundary {
        let h0 = cdt.hint;
        cdt.insert_point(i, h0, cancel)?;
    }

    let mut constraint_edges: Vec<(u32, u32)> = Vec::new();
    for &(start, len) in &loop_ranges {
        for i in 0..len {
            constraint_edges.push(((start + i) as u32, (start + (i + 1) % len) as u32));
        }
    }
    for &(a, b) in &constraint_edges {
        cdt.recover_edge(a, b)?;
    }
    cdt.rebuild_neighbors();
    for &(a, b) in &constraint_edges {
        if !cdt.mark_constrained_if_present(a, b) {
            return Err(PyrucastError::Message(
                "triangulate_surface: internal error recovering a boundary edge".into(),
            ));
        }
    }
    cdt.remove_super_triangle();
    cdt.rebuild_neighbors();
    cdt.legalize();

    let seed_edge = constraint_edges[0];
    let (seed, _) = cdt
        .find_directed_edge(seed_edge.0, seed_edge.1)
        .ok_or_else(|| {
            PyrucastError::Message(
                "triangulate_surface: internal error locating the domain interior".into(),
            )
        })?;
    cdt.flood_fill(seed, cancel)?;

    // Freeze the contour: the returned mesh reuses exactly the input boundary
    // nodes (same NodeId, same position) and never subdivides a contour edge.
    cdt.refine(h, true, cancel)?;

    // Quality post-processing: alternate min-angle flips (dissolve slivers the
    // Delaunay mesh leaves near curved boundaries) with inversion-safe
    // Laplacian smoothing (equalize interior node spacing). A few rounds bring
    // the worst angles up without disturbing the boundary.
    cdt.improve_angles(6);
    for _ in 0..3 {
        cancel.check()?;
        let tris = collect_inside_tris(&cdt);
        smooth(&mut cdt.pts, &cdt.is_boundary, &tris, 2);
        cdt.improve_angles(4);
    }

    let tris = collect_inside_tris(&cdt);
    if tris.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_surface: produced no cell for a domain (degenerate contour?)".into(),
        ));
    }

    let (quads, leftover_tris) = if element_type == ElementType::QUA4 {
        recombine_to_quads(&cdt.pts, &tris)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(DomainResult {
        boundary_node_ids,
        pts: cdt.pts,
        n_boundary,
        tris: if element_type == ElementType::TRI3 {
            tris
        } else {
            Vec::new()
        },
        quads,
        leftover_tris,
    })
}

// ─── Smoothing and QUA4 recombination (pure, parallelizable) ─────────────

/// Laplacian smoothing of interior nodes. A proposed move is rejected if it
/// would invert (or degenerate) any incident triangle, so the total meshed
/// area is preserved exactly and no cell flips.
fn smooth(pts: &mut [Point2], is_boundary: &[bool], tris: &[[u32; 3]], iters: usize) {
    let n = pts.len();
    let mut ring: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ti, t) in tris.iter().enumerate() {
        for e in 0..3 {
            let a = t[e] as usize;
            ring[a].push(t[(e + 1) % 3]);
            incident[a].push(ti as u32);
        }
    }
    for _ in 0..iters {
        let old = pts.to_vec();
        pts.par_iter_mut().enumerate().for_each(|(i, p)| {
            if is_boundary[i] || ring[i].is_empty() {
                return;
            }
            let mut c = Vector2::zeros();
            for &nb in &ring[i] {
                c += old[nb as usize].coords;
            }
            c /= ring[i].len() as f64;
            let cand = Point2::from(old[i].coords * 0.5 + c * 0.5);
            let safe = incident[i].iter().all(|&ti| {
                let t = tris[ti as usize];
                let q = |k: u32| {
                    if k as usize == i {
                        cand
                    } else {
                        old[k as usize]
                    }
                };
                orient(q(t[0]), q(t[1]), q(t[2])) > 0.0
            });
            if safe {
                *p = cand;
            }
        });
    }
}

/// Quality score of the quad `(a, u, b, v)` (CCW), or `None` if non-convex.
/// Higher is better (closer to a square).
fn quad_quality(pts: &[Point2], a: u32, u: u32, b: u32, v: u32) -> Option<f64> {
    let quad = [
        pts[a as usize],
        pts[u as usize],
        pts[b as usize],
        pts[v as usize],
    ];
    let mut min_angle = f64::INFINITY;
    for i in 0..4 {
        let prev = quad[(i + 3) % 4];
        let cur = quad[i];
        let next = quad[(i + 1) % 4];
        if orient(prev, cur, next) <= 0.0 {
            return None;
        }
        let v1 = prev - cur;
        let v2 = next - cur;
        let cosang = (v1.dot(&v2) / (v1.norm() * v2.norm())).clamp(-1.0, 1.0);
        min_angle = min_angle.min(cosang.acos());
    }
    Some(min_angle)
}

/// Greedy triangle-pair recombination into quads, best-quality pairs first.
/// Triangles that can't be paired stay as leftovers.
fn recombine_to_quads(pts: &[Point2], tris: &[[u32; 3]]) -> (Vec<[u32; 4]>, Vec<[u32; 3]>) {
    let mut edge_owner: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        for e in 0..3 {
            edge_owner.insert((t[e], t[(e + 1) % 3]), i);
        }
    }
    let candidates: Vec<(f64, usize, usize, u32, u32)> = tris
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, t)| {
            let edge_owner = &edge_owner;
            (0..3).filter_map(move |e| {
                let (u, v) = (t[e], t[(e + 1) % 3]);
                let &j = edge_owner.get(&(v, u))?;
                if j <= i {
                    return None;
                }
                let a = t[(e + 2) % 3];
                let tj = tris[j];
                let b = tj.into_iter().find(|&x| x != u && x != v)?;
                let score = quad_quality(pts, a, u, b, v)?;
                Some((score, i, j, u, v))
            })
        })
        .collect();
    let mut candidates = candidates;
    candidates.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

    let mut used = vec![false; tris.len()];
    let mut quads = Vec::new();
    for (_, i, j, u, v) in candidates {
        if used[i] || used[j] {
            continue;
        }
        used[i] = true;
        used[j] = true;
        let t = tris[i];
        let e = (0..3).find(|&e| t[e] == u && t[(e + 1) % 3] == v).unwrap();
        let a = t[(e + 2) % 3];
        let tj = tris[j];
        let b = tj.into_iter().find(|&x| x != u && x != v).unwrap();
        quads.push([a, u, b, v]);
    }
    let leftover: Vec<[u32; 3]> = tris
        .iter()
        .enumerate()
        .filter(|(i, _)| !used[*i])
        .map(|(_, t)| *t)
        .collect();
    (quads, leftover)
}

// ─── Materialization ────────────────────────────────────────────────────

fn materialize(
    coords_handle: Handle<Coords>,
    frame: &Frame,
    dim: u8,
    element_type: ElementType,
    results: Vec<DomainResult>,
) -> Result<Mesh> {
    let mut tri_sub: Option<SubMesh> = None;
    let mut quad_sub: Option<SubMesh> = None;
    let mut kept_nodes: Vec<Node> = Vec::new();

    for r in results {
        let mut flat: Vec<NodeId> = r.boundary_node_ids.clone();
        for p in &r.pts[r.n_boundary..] {
            let coord = frame.to_world(*p, dim);
            let node = Node::create_in(coords_handle.clone(), &coord)?;
            flat.push(node.id());
            kept_nodes.push(node);
        }
        match element_type {
            ElementType::TRI3 => {
                let sub = tri_sub
                    .get_or_insert_with(|| SubMesh::new(coords_handle.clone(), ElementType::TRI3));
                for t in &r.tris {
                    sub.add_cell(&[
                        flat[t[0] as usize],
                        flat[t[1] as usize],
                        flat[t[2] as usize],
                    ])?;
                }
            }
            ElementType::QUA4 => {
                let qsub = quad_sub
                    .get_or_insert_with(|| SubMesh::new(coords_handle.clone(), ElementType::QUA4));
                for q in &r.quads {
                    qsub.add_cell(&[
                        flat[q[0] as usize],
                        flat[q[1] as usize],
                        flat[q[2] as usize],
                        flat[q[3] as usize],
                    ])?;
                }
                if !r.leftover_tris.is_empty() {
                    let tsub = tri_sub.get_or_insert_with(|| {
                        SubMesh::new(coords_handle.clone(), ElementType::TRI3)
                    });
                    for t in &r.leftover_tris {
                        tsub.add_cell(&[
                            flat[t[0] as usize],
                            flat[t[1] as usize],
                            flat[t[2] as usize],
                        ])?;
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    let mut mesh = Mesh::empty();
    if let Some(q) = quad_sub {
        if q.cell_count() > 0 {
            mesh.add_sub(insert(q))?;
        }
    }
    if let Some(t) = tri_sub {
        if t.cell_count() > 0 {
            mesh.add_sub(insert(t))?;
        }
    }
    drop(kept_nodes);
    if mesh.is_empty() {
        return Err(PyrucastError::Message(
            "triangulate_surface: produced no cell".into(),
        ));
    }
    Ok(mesh)
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::store::insert;

    fn loop_mesh(coords: Handle<Coords>, pts: &[(f64, f64)]) -> Mesh {
        let ids: Vec<NodeId> = pts
            .iter()
            .map(|&(x, y)| Node::create_in(coords.clone(), &[x, y]).unwrap().id())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        let n = ids.len();
        for i in 0..n {
            sm.add_cell(&[ids[i], ids[(i + 1) % n]]).unwrap();
        }
        Mesh::from_submesh(sm)
    }

    fn square(coords: Handle<Coords>, s: f64) -> Mesh {
        loop_mesh(coords, &[(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)])
    }

    /// A square with `n` segments per side. Since the contour is now frozen
    /// (never subdivided by refinement), a mesh denser than the input contour
    /// requires the boundary to be pre-discretized — exactly as real usage
    /// does (`mesher::line(p1, p2, 15)`).
    fn discretized_square(coords: Handle<Coords>, s: f64, n: usize) -> Mesh {
        let mut pts = Vec::new();
        let corners = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)];
        for k in 0..4 {
            let (x0, y0) = corners[k];
            let (x1, y1) = corners[(k + 1) % 4];
            for i in 0..n {
                let t = i as f64 / n as f64;
                pts.push((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t));
            }
        }
        loop_mesh(coords, &pts)
    }

    fn cell_area(pts: &[Vec<f64>]) -> f64 {
        let n = pts.len();
        let mut a = 0.0;
        for i in 0..n {
            let p = &pts[i];
            let q = &pts[(i + 1) % n];
            a += p[0] * q[1] - q[0] * p[1];
        }
        (a * 0.5).abs()
    }

    fn mesh_area(mesh: &Mesh) -> f64 {
        let counts = mesh.cell_counts().unwrap();
        let types = mesh.element_types().unwrap();
        let mut total = 0.0;
        for (si, &cnt) in counts.iter().enumerate() {
            let npc = types[si].nodes_per_cell();
            for ci in 0..cnt {
                let pts: Vec<Vec<f64>> = (0..npc)
                    .map(|ni| mesh.node(si, ci, ni).unwrap().coord().unwrap())
                    .collect();
                total += cell_area(&pts);
            }
        }
        total
    }

    #[test]
    fn square_tri3_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = discretized_square(coords, 1.0, 8);
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(0.15)).unwrap();
        assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TRI3]);
        assert!(mesh.cell_count().unwrap() > 20);
        assert!(
            (mesh_area(&mesh) - 1.0).abs() < 1e-9,
            "area drift: {}",
            mesh_area(&mesh)
        );
    }

    #[test]
    fn square_tri3_coarse_still_covers_boundary() {
        // Size larger than the square: still must mesh from the 4 corners alone.
        let coords = insert(Coords::new(2).unwrap());
        let contour = square(coords, 1.0);
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(10.0)).unwrap();
        assert!(mesh.cell_count().unwrap() >= 2);
        assert!((mesh_area(&mesh) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn square_qua4_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = discretized_square(coords, 1.0, 6);
        let mesh = triangulate_surface(&contour, ElementType::QUA4, Some(0.2)).unwrap();
        assert!(mesh.element_types().unwrap().contains(&ElementType::QUA4));
        assert!(
            (mesh_area(&mesh) - 1.0).abs() < 1e-9,
            "area drift: {}",
            mesh_area(&mesh)
        );
    }

    #[test]
    fn square_with_hole_conserves_area() {
        let coords = insert(Coords::new(2).unwrap());
        let mut contour = square(coords.clone(), 2.0);
        // CW hole: a small square traversed clockwise.
        let hole = loop_mesh(coords, &[(0.9, 0.9), (0.9, 1.1), (1.1, 1.1), (1.1, 0.9)]);
        contour.add_sub(hole.get(0).unwrap()).unwrap();
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(0.15)).unwrap();
        let want = 4.0 - 0.04;
        assert!(
            (mesh_area(&mesh) - want).abs() < 1e-6,
            "area drift: {}",
            mesh_area(&mesh)
        );
    }

    #[test]
    fn two_disjoint_squares_are_independent_domains() {
        let coords = insert(Coords::new(2).unwrap());
        let mut contour = loop_mesh(
            coords.clone(),
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        );
        let other = loop_mesh(coords, &[(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]);
        contour.add_sub(other.get(0).unwrap()).unwrap();
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(0.3)).unwrap();
        assert!((mesh_area(&mesh) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn contour_nodes_are_frozen() {
        // The meshed result must reuse exactly the input contour nodes (same
        // NodeId, same position) and add no node on a contour edge.
        let coords = insert(Coords::new(2).unwrap());
        let contour = discretized_square(coords.clone(), 1.0, 10);
        let mut boundary: HashSet<NodeId> = HashSet::new();
        for sm in &contour {
            for &nid in read(sm).unwrap().connectivity() {
                boundary.insert(nid);
            }
        }
        let before: HashMap<NodeId, Vec<f64>> = boundary
            .iter()
            .map(|&nid| (nid, read(&coords).unwrap().coord(nid).unwrap().to_vec()))
            .collect();

        // A fine target size that would trigger heavy Ruppert refinement.
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(0.05)).unwrap();

        let mut used: HashSet<NodeId> = HashSet::new();
        for sm in &mesh {
            for &nid in read(sm).unwrap().connectivity() {
                used.insert(nid);
            }
        }
        // Every input boundary node is still used, unmoved.
        for (&nid, p0) in &before {
            assert!(used.contains(&nid), "contour node {nid:?} dropped");
            let p1 = read(&coords).unwrap().coord(nid).unwrap().to_vec();
            assert_eq!(&p1, p0, "contour node {nid:?} moved");
        }
        // No new node sits on the (axis-aligned) contour edges.
        for &nid in &used {
            if boundary.contains(&nid) {
                continue;
            }
            let p = read(&coords).unwrap().coord(nid).unwrap().to_vec();
            let on_edge = (p[0] <= 1e-12 || (p[0] - 1.0).abs() <= 1e-12)
                || (p[1] <= 1e-12 || (p[1] - 1.0).abs() <= 1e-12);
            assert!(
                !on_edge,
                "new node {nid:?} at {p:?} was placed on a contour edge"
            );
        }
    }

    #[test]
    fn rejects_bad_element_type() {
        let coords = insert(Coords::new(2).unwrap());
        let contour = square(coords, 1.0);
        assert!(triangulate_surface(&contour, ElementType::TET4, None).is_err());
    }

    #[test]
    fn rejects_non_seg2_contour() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let contour = Mesh::from_submesh(sm);
        assert!(triangulate_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn rejects_open_loop() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        sm.add_cell(&[b.id(), c.id()]).unwrap();
        // missing closing segment c -> a
        let contour = Mesh::from_submesh(sm);
        assert!(triangulate_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn rejects_all_holes_no_outer() {
        let coords = insert(Coords::new(2).unwrap());
        // Clockwise loop only: no outer (CCW) boundary.
        let contour = loop_mesh(coords, &[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]);
        assert!(triangulate_surface(&contour, ElementType::TRI3, None).is_err());
    }

    #[test]
    fn planar_3d_contour_is_meshed_in_its_plane() {
        let coords = insert(Coords::new(3).unwrap());
        let ids: Vec<NodeId> = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|&(x, y, z)| Node::create_in(coords.clone(), &[x, y, z]).unwrap().id())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        let n = ids.len();
        for i in 0..n {
            sm.add_cell(&[ids[i], ids[(i + 1) % n]]).unwrap();
        }
        let contour = Mesh::from_submesh(sm);
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(0.3)).unwrap();
        assert!(mesh.cell_count().unwrap() > 0);
    }

    #[test]
    fn cancellable_stops_on_preset_flag() {
        use std::sync::atomic::AtomicBool;
        let coords = insert(Coords::new(2).unwrap());
        let contour = square(coords, 1.0);
        let flag = AtomicBool::new(true);
        let err = triangulate_surface_cancellable(&contour, ElementType::TRI3, Some(0.1), &flag)
            .unwrap_err();
        assert!(matches!(err, PyrucastError::Interrupted));
    }

    #[test]
    #[ignore]
    fn perf_hundred_k_triangles_under_30s() {
        // A plate-with-hole domain at the scale of formation/maillage_test.py:
        // 0.30 x 0.10 m rectangle with a r=0.035 m circular hole.
        let coords = insert(Coords::new(2).unwrap());
        let mut contour = loop_mesh(
            coords.clone(),
            &[(0.0, 0.0), (0.30, 0.0), (0.30, 0.10), (0.0, 0.10)],
        );
        let n = 64;
        let (cx, cy, r) = (0.75 * 0.30, 0.05, 0.035);
        let hole_pts: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let a = k as f64 / n as f64 * std::f64::consts::TAU;
                // clockwise (hole) orientation
                (cx + r * a.cos(), cy - r * a.sin())
            })
            .collect();
        let hole = loop_mesh(coords, &hole_pts);
        contour.add_sub(hole.get(0).unwrap()).unwrap();

        let start = std::time::Instant::now();
        let mesh = triangulate_surface(&contour, ElementType::TRI3, Some(7.8e-4)).unwrap();
        let elapsed = start.elapsed();
        let n_cells = mesh.cell_count().unwrap();
        println!("perf: {} TRI3 cells in {:?}", n_cells, elapsed);
        assert!(n_cells > 80_000, "only {} cells", n_cells);
        assert!(elapsed.as_secs_f64() < 30.0, "took {:?}", elapsed);
    }
}
