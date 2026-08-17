//! Shared front-end of the surface meshers: read a closed `SEG2` contour,
//! fit it to a plane, and sort its loops into domains.
//!
//! Both surface meshers — [`triangulate_surface`](fn@super::triangulate_surface)
//! (constrained Delaunay) and [`pave_surface`](fn@super::pave_surface) (frontal
//! paving) — take the *same* input and disagree only on how they fill it.
//! Everything up to "a list of domains, each an outer loop plus its holes,
//! expressed in a local 2-D frame" is therefore factored out here so the two
//! cannot drift apart.
//!
//! The pipeline is [`parse()`]:
//!
//! 1. [`extract_loops`] walks every `SEG2` submesh into one closed simple
//!    chain of `NodeId`s, rejecting branching, repetition and open chains.
//! 2. [`Frame::fit`] keeps a 2-D contour as it is, and fits a 3-D one to its
//!    best plane by Newell's method — the mesh is then built in that local
//!    frame and lifted back with [`Frame::to_world`].
//! 3. [`build_domains`] reads the loops' orientation: counter-clockwise is an
//!    outer boundary, clockwise is a hole, and each hole joins the domain that
//!    contains it. Several disjoint outer loops are several independent
//!    domains.
//!
//! Error messages are prefixed with the caller's operator name so the user
//! reads `pave_surface: …` or `triangulate_surface: …` as appropriate.

use crate::atoms::{ElementType, NodeId, Point2, Point3, Vector3};
use crate::containers::mesh::Mesh;
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::collections::{HashMap, HashSet};

/// One closed boundary loop, still in world coordinates.
pub struct LoopData {
    pub node_ids: Vec<NodeId>,
    pub world_pts: Vec<Vec<f64>>,
}

/// One closed boundary loop, projected into the local 2-D frame.
pub struct Loop2D {
    pub node_ids: Vec<NodeId>,
    pub pts: Vec<Point2>,
}

/// An outer counter-clockwise loop and the clockwise loops it contains.
pub struct Domain {
    pub outer: Loop2D,
    pub holes: Vec<Loop2D>,
}

/// Everything a surface mesher needs before it starts filling: the contour's
/// `Coords`, its dimension, the plane it was fitted to, and its domains.
pub struct Contour {
    pub coords: Handle<Coords>,
    pub dim: u8,
    pub frame: Frame,
    pub domains: Vec<Domain>,
}

/// Read `contour` and prepare it for meshing. `op` names the calling operator
/// and only ever appears in error messages.
pub fn parse(contour: &Mesh, op: &str) -> Result<Contour> {
    let coords = contour.coords()?;
    let dim = coords.read().dim();
    if dim != 2 && dim != 3 {
        return Err(PyrucastError::Message(format!(
            "{op}: contour must be 2-D or 3-D, got dim={dim}"
        )));
    }
    let loops = extract_loops(contour, op)?;
    if loops.is_empty() {
        return Err(PyrucastError::Message(format!(
            "{op}: contour has no boundary loop"
        )));
    }
    let frame = Frame::fit(dim, &loops, op)?;
    let loops2d: Vec<Loop2D> = loops
        .iter()
        .map(|l| Loop2D {
            node_ids: l.node_ids.clone(),
            pts: l.world_pts.iter().map(|p| frame.to_local(p)).collect(),
        })
        .collect();
    let domains = build_domains(loops2d, op)?;
    Ok(Contour {
        coords,
        dim,
        frame,
        domains,
    })
}

// ─── Contour parsing ──────────────────────────────────────────────────────

/// Walk each `SEG2` submesh into a single closed simple loop of nodes.
pub fn extract_loops(mesh: &Mesh, op: &str) -> Result<Vec<LoopData>> {
    let coords = mesh.coords()?;
    let c = coords.read();
    let mut loops = Vec::new();
    for sm in mesh {
        let s = sm.read();
        if s.element_type() != ElementType::SEG2 {
            return Err(PyrucastError::Message(format!(
                "{op}: contour submeshes must be SEG2, got {}",
                s.element_type()
            )));
        }
        let conn = s.connectivity();
        let n = conn.len() / 2;
        if n < 3 {
            return Err(PyrucastError::Message(format!(
                "{op}: a boundary loop needs at least 3 segments"
            )));
        }
        let mut next: HashMap<NodeId, NodeId> = HashMap::new();
        for pair in conn.chunks(2) {
            if next.insert(pair[0], pair[1]).is_some() {
                return Err(PyrucastError::Message(format!(
                    "{op}: a boundary submesh is not a simple loop (branching)"
                )));
            }
        }
        let start = conn[0];
        let mut chain = Vec::with_capacity(n);
        let mut cur = start;
        let mut seen = HashSet::new();
        for _ in 0..n {
            if !seen.insert(cur) {
                return Err(PyrucastError::Message(format!(
                    "{op}: a boundary submesh is not a simple loop (repeated node)"
                )));
            }
            chain.push(cur);
            cur = *next.get(&cur).ok_or_else(|| {
                PyrucastError::Message(format!("{op}: a boundary submesh is not closed"))
            })?;
        }
        if cur != start {
            return Err(PyrucastError::Message(format!(
                "{op}: a boundary submesh is not closed"
            )));
        }
        let world_pts: Result<Vec<Vec<f64>>> = chain
            .iter()
            .map(|&nid| Ok(c.position(nid)?.to_vec()))
            .collect();
        loops.push(LoopData {
            node_ids: chain,
            world_pts: world_pts?,
        });
    }
    Ok(loops)
}

// ─── Planar frame (2-D native, or best-fit plane for a 3-D contour) ───────

/// The plane the contour is meshed in.
pub enum Frame {
    /// The contour's `Coords` is already 2-D; local and world coincide.
    Planar2D,
    /// A 3-D contour, fitted to the plane `(origin; u, v)`.
    Planar3D {
        origin: Point3,
        u: Vector3,
        v: Vector3,
    },
}

impl Frame {
    /// Fit the loops to their best plane (Newell's method) when `dim == 3`.
    pub fn fit(dim: u8, loops: &[LoopData], op: &str) -> Result<Frame> {
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
            return Err(PyrucastError::Message(format!("{op}: empty contour")));
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
            return Err(PyrucastError::Message(format!(
                "{op}: contour points are collinear or degenerate"
            )));
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

    /// World point → local 2-D frame.
    pub fn to_local(&self, p: &[f64]) -> Point2 {
        match self {
            Frame::Planar2D => Point2::new(p[0], p[1]),
            Frame::Planar3D { origin, u, v } => {
                let d = Vector3::new(p[0], p[1], p[2]) - origin.coords;
                Point2::new(d.dot(u), d.dot(v))
            }
        }
    }

    /// Local 2-D frame → world point of dimension `dim`.
    pub fn to_world(&self, p: Point2, dim: u8) -> Vec<f64> {
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

/// Crossing-number test: is `p` inside the closed polygon `poly`?
pub fn point_in_polygon(p: Point2, poly: &[Point2]) -> bool {
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

/// Sort loops by orientation: counter-clockwise loops become domains,
/// clockwise loops become holes of the domain that contains them.
pub fn build_domains(loops: Vec<Loop2D>, op: &str) -> Result<Vec<Domain>> {
    let mut outers = Vec::new();
    let mut holes = Vec::new();
    for l in loops {
        let a = super::triangulation::signed_area(&l.pts);
        if a.abs() < 1e-300 {
            return Err(PyrucastError::Message(format!(
                "{op}: a boundary loop has zero area"
            )));
        }
        if a > 0.0 {
            outers.push(l);
        } else {
            holes.push(l);
        }
    }
    if outers.is_empty() {
        return Err(PyrucastError::Message(format!(
            "{op}: no counter-clockwise (outer) loop found"
        )));
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
        return Err(PyrucastError::Message(format!(
            "{op}: a hole (clockwise) loop is not contained in any outer loop"
        )));
    }
    Ok(domains)
}
