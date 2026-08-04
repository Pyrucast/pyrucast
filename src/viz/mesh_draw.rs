//! `Drawable` implementations for [`SubMesh`] and [`Mesh`].
//!
//! Every supported element type is converted into one of three rendering
//! primitives:
//!
//! - **Point** — POI1
//! - **Segment** — SEG2
//! - **Face** — TRI3 / QUA4 / TET4 (each volume face) / HEX8 (each volume face)
//!
//! The pipeline is the painter's algorithm: every primitive is projected
//! to 2D + depth, the list is sorted far → near, then primitives are drawn
//! in order — filled with [`SubMesh::face_color`] for faces, drawn with
//! the same colour for points/segments, and overlaid with black edges for
//! face boundaries.

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::atoms::RgbColor;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::Result;
use crate::store::{read, Handle};
use crate::viz::camera::{Bbox3, Projector};
use crate::viz::drawable::{pl_err, Drawable};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::atoms::Point3;
use std::collections::{HashMap, HashSet};

/// Pad world coordinates to a 3-D point, filling missing components with 0.0.
pub(crate) fn pad3(coords: &[f64]) -> Point3 {
    Point3::new(
        coords.first().copied().unwrap_or(0.0),
        coords.get(1).copied().unwrap_or(0.0),
        coords.get(2).copied().unwrap_or(0.0),
    )
}

/// Read every node referenced by `connectivity` from `coords`, padded to 3-D.
fn read_points(coords: &Handle<Coords>, connectivity: &[NodeId]) -> Result<Vec<Point3>> {
    let c = read(coords)?;
    connectivity
        .iter()
        .map(|&nid| c.position(nid).map(pad3))
        .collect()
}

/// Geometric primitive emitted by a submesh and consumed by the painter.
///
/// `Face::verts` is 3 (triangle) or 4 (quadrangle) vertices in CCW order
/// when seen from outside the solid; the renderer fills the polygon and
/// then overlays its boundary as a black wireframe.
#[derive(Debug, Clone)]
pub(crate) enum Primitive {
    Point {
        p: Point3,
        color: RgbColor,
    },
    Segment {
        a: Point3,
        b: Point3,
        color: RgbColor,
    },
    /// Filled polygon; `outline: false` skips the black wireframe (used
    /// by the interpolated renderer for its interior sub-faces).
    Face {
        verts: Vec<Point3>,
        color: RgbColor,
        outline: bool,
    },
    /// Stroke-only closed polyline, drawn slightly towards the viewer —
    /// the element boundary on top of its (outline-free) sub-faces.
    Wire {
        verts: Vec<Point3>,
    },
}

// ─── Element-type → primitives ──────────────────────────────────────────────

/// Faces of a TET4 — each oriented outwards (CCW seen from outside).
pub(crate) const TET4_FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];

/// Faces of a HEX8 — bot / top / 4 lateral, in the convention used by
/// [`crate::ops::mesh::extrude`]: HEX8 = [bot[0..4], top[0..4]], both CCW seen from
/// outside the lateral surface.
pub(crate) const HEX8_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1], // bottom (normal opposed to extrusion direction)
    [4, 5, 6, 7], // top
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

/// Faces of a PENTA6 (prism) — two triangular caps then three
/// quadrilateral sides, each oriented outwards, in the convention used by
/// [`crate::ops::mesh::extrude`]: PENTA6 = [bot[0..3], top[0..3]]. The
/// caps carry 3 indices, the sides 4, so the faces are stored as slices.
pub(crate) const PENTA6_FACES: [&[usize]; 5] = [
    &[0, 2, 1],    // bottom triangle (normal opposed to extrusion direction)
    &[3, 4, 5],    // top triangle
    &[0, 1, 4, 3], // side
    &[1, 2, 5, 4], // side
    &[2, 0, 3, 5], // side
];

/// Faces of a PYRA5 (pyramid) — the square base, wound so its normal points
/// away from the apex, then the four triangles round the sides.
pub(crate) const PYRA5_FACES: [&[usize]; 5] = [
    &[0, 3, 2, 1], // base
    &[0, 1, 4],
    &[1, 2, 4],
    &[2, 3, 4],
    &[3, 0, 4],
];

/// Edges of each element type, as local node-index pairs — used by the
/// wireframe rendering style. POI1 has no edge (it draws as a dot).
#[rustfmt::skip]
fn element_edges(et: ElementType) -> &'static [[usize; 2]] {
    match et {
        ElementType::POI1 => &[],
        ElementType::SEG2 => &[[0, 1]],
        ElementType::TRI3 => &[[0, 1], [1, 2], [2, 0]],
        ElementType::QUA4 => &[[0, 1], [1, 2], [2, 3], [3, 0]],
        ElementType::TET4 => &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        ElementType::PYRA5 => &[
            [0, 1], [1, 2], [2, 3], [3, 0],
            [0, 4], [1, 4], [2, 4], [3, 4],
        ],
        ElementType::PENTA6 => &[
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
            [0, 3], [1, 4], [2, 5],
        ],
        ElementType::HEX8 => &[
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        // Quadratic types: each geometric edge is split at its mid node so
        // the wireframe passes through the mid-side nodes.
        ElementType::SEG3 => &[[0, 2], [2, 1]],
        ElementType::TRI6 => &[
            [0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0],
        ],
        ElementType::QUA8 | ElementType::QUA9 => &[
            [0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 0],
        ],
        ElementType::TET10 => &[
            [0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 0],
            [0, 7], [7, 3], [1, 8], [8, 3], [2, 9], [9, 3],
        ],
        ElementType::PENTA15 => &[
            [0, 6], [6, 1], [1, 7], [7, 2], [2, 8], [8, 0],
            [3, 9], [9, 4], [4, 10], [10, 5], [5, 11], [11, 3],
            [0, 12], [12, 3], [1, 13], [13, 4], [2, 14], [14, 5],
        ],
        ElementType::HEX20 | ElementType::HEX27 => &[
            [0, 8], [8, 1], [1, 9], [9, 2], [2, 10], [10, 3], [3, 11], [11, 0],
            [4, 12], [12, 5], [5, 13], [13, 6], [6, 14], [14, 7], [7, 15], [15, 4],
            [0, 16], [16, 4], [1, 17], [17, 5], [2, 18], [18, 6], [3, 19], [19, 7],
        ],
    }
}

/// `(cell, local_face)` pairs whose face lies on the **boundary** of a
/// volume submesh — a face shared by two cells (tetrahedra / hexahedra)
/// sits inside the solid and is never visible once the skin is drawn
/// opaque, so the renderer drops it: only the outer surface is emitted.
/// This both fixes the see-through look of solid 3-D meshes and cuts the
/// face count roughly in half.
///
/// Returns `None` for point / line / surface element types, where every
/// primitive is kept (nothing is ever hidden behind another cell).
///
/// A face is keyed by the **set** of its global node ids (sorted), so the
/// two cells sharing it produce the same key regardless of orientation.
pub(crate) fn boundary_faces(et: ElementType, conn: &[NodeId]) -> Option<HashSet<(usize, usize)>> {
    // Quadratic volumes reuse their linear parent's corner-face tables: the
    // face key is built from corner node ids (indices < corner count), which
    // two adjacent cells share, so interior-face culling still works.
    let faces: Vec<&[usize]> = match et {
        ElementType::TET4 | ElementType::TET10 => TET4_FACES.iter().map(|f| f.as_slice()).collect(),
        ElementType::HEX8 | ElementType::HEX20 | ElementType::HEX27 => {
            HEX8_FACES.iter().map(|f| f.as_slice()).collect()
        }
        ElementType::PENTA6 | ElementType::PENTA15 => PENTA6_FACES.to_vec(),
        ElementType::PYRA5 => PYRA5_FACES.to_vec(),
        _ => return None,
    };
    let npc = et.nodes_per_cell();
    if npc == 0 {
        return Some(HashSet::new());
    }
    let n_cells = conn.len() / npc;
    let face_key = |cell: usize, f: &[usize]| -> Vec<u32> {
        let base = cell * npc;
        let mut k: Vec<u32> = f.iter().map(|&li| conn[base + li].0).collect();
        k.sort_unstable();
        k
    };
    let mut count: HashMap<Vec<u32>, usize> = HashMap::new();
    for cell in 0..n_cells {
        for f in &faces {
            *count.entry(face_key(cell, f)).or_insert(0) += 1;
        }
    }
    let mut keep = HashSet::new();
    for cell in 0..n_cells {
        for (fi, f) in faces.iter().enumerate() {
            if count.get(&face_key(cell, f)) == Some(&1) {
                keep.insert((cell, fi));
            }
        }
    }
    Some(keep)
}

/// Build the rendering primitives of a single `SubMesh`. The output is
/// empty for an empty submesh; every supported element type produces at
/// least one primitive per cell.
///
/// If `colors_per_cell` is `Some`, each cell's primitives use the
/// supplied colour at index `cell_idx`; otherwise the submesh's own
/// `face_color` is used uniformly. The length of `colors_per_cell`
/// must equal the number of cells when provided.
pub(crate) fn submesh_primitives(sm: &SubMesh) -> Result<Vec<Primitive>> {
    submesh_primitives_impl(sm, None)
}

/// Variant of [`submesh_primitives`] with one colour per cell (one cell
/// can contribute several primitives, e.g. TET4 → 4 faces; all faces of
/// a given cell share the cell's colour).
pub(crate) fn submesh_primitives_with_colors(
    sm: &SubMesh,
    colors_per_cell: &[RgbColor],
) -> Result<Vec<Primitive>> {
    submesh_primitives_impl(sm, Some(colors_per_cell))
}

fn submesh_primitives_impl(
    sm: &SubMesh,
    colors_per_cell: Option<&[RgbColor]>,
) -> Result<Vec<Primitive>> {
    let coords = sm.coords();
    let pts = read_points(&coords, sm.connectivity())?;
    let default_color = sm.face_color();
    let et = sm.element_type();
    let npc = et.nodes_per_cell();
    let n_cells = pts.len().checked_div(npc).unwrap_or(0);
    if let Some(colors) = colors_per_cell {
        if colors.len() != n_cells {
            return Err(crate::error::PyrucastError::Message(format!(
                "submesh_primitives_with_colors: got {} colors for {} cells",
                colors.len(),
                n_cells
            )));
        }
    }
    let cell_color = |i: usize| -> RgbColor {
        match colors_per_cell {
            Some(c) => c[i],
            None => default_color,
        }
    };
    let mut out: Vec<Primitive> = Vec::new();

    // Volume cells: keep only boundary faces (interior faces are hidden
    // inside the opaque solid). `None` for non-volume types → keep all.
    let keep = boundary_faces(et, sm.connectivity());

    match et {
        ElementType::POI1 => {
            for i in 0..n_cells {
                out.push(Primitive::Point {
                    p: pts[i],
                    color: cell_color(i),
                });
            }
        }
        ElementType::SEG2 => {
            for i in 0..n_cells {
                out.push(Primitive::Segment {
                    a: pts[2 * i],
                    b: pts[2 * i + 1],
                    color: cell_color(i),
                });
            }
        }
        ElementType::TRI3 => {
            for i in 0..n_cells {
                out.push(Primitive::Face {
                    verts: vec![pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]],
                    color: cell_color(i),
                    outline: true,
                });
            }
        }
        ElementType::QUA4 => {
            for i in 0..n_cells {
                out.push(Primitive::Face {
                    verts: vec![pts[4 * i], pts[4 * i + 1], pts[4 * i + 2], pts[4 * i + 3]],
                    color: cell_color(i),
                    outline: true,
                });
            }
        }
        ElementType::TET4 => {
            for i in 0..n_cells {
                let base = 4 * i;
                let c = cell_color(i);
                for (fi, face) in TET4_FACES.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(i, fi))) {
                        continue;
                    }
                    out.push(Primitive::Face {
                        verts: vec![
                            pts[base + face[0]],
                            pts[base + face[1]],
                            pts[base + face[2]],
                        ],
                        color: c,
                        outline: true,
                    });
                }
            }
        }
        ElementType::HEX8 => {
            for i in 0..n_cells {
                let base = 8 * i;
                let c = cell_color(i);
                for (fi, face) in HEX8_FACES.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(i, fi))) {
                        continue;
                    }
                    out.push(Primitive::Face {
                        verts: vec![
                            pts[base + face[0]],
                            pts[base + face[1]],
                            pts[base + face[2]],
                            pts[base + face[3]],
                        ],
                        color: c,
                        outline: true,
                    });
                }
            }
        }
        ElementType::PYRA5 => {
            for i in 0..n_cells {
                let base = 5 * i;
                let c = cell_color(i);
                for (fi, face) in PYRA5_FACES.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(i, fi))) {
                        continue;
                    }
                    out.push(Primitive::Face {
                        verts: face.iter().map(|&li| pts[base + li]).collect(),
                        color: c,
                        outline: true,
                    });
                }
            }
        }
        ElementType::PENTA6 => {
            for i in 0..n_cells {
                let base = 6 * i;
                let c = cell_color(i);
                for (fi, face) in PENTA6_FACES.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(i, fi))) {
                        continue;
                    }
                    out.push(Primitive::Face {
                        verts: face.iter().map(|&li| pts[base + li]).collect(),
                        color: c,
                        outline: true,
                    });
                }
            }
        }
        // Quadratic types: draw the linearized skin over the corner nodes
        // (the interpolated renderer in `subdivide` shows the curvature).
        ElementType::SEG3 => {
            for i in 0..n_cells {
                let base = 3 * i;
                let c = cell_color(i);
                out.push(Primitive::Segment {
                    a: pts[base],
                    b: pts[base + 2],
                    color: c,
                });
                out.push(Primitive::Segment {
                    a: pts[base + 2],
                    b: pts[base + 1],
                    color: c,
                });
            }
        }
        ElementType::TRI6 => {
            for i in 0..n_cells {
                let base = 6 * i;
                out.push(Primitive::Face {
                    verts: vec![pts[base], pts[base + 1], pts[base + 2]],
                    color: cell_color(i),
                    outline: true,
                });
            }
        }
        ElementType::QUA8 | ElementType::QUA9 => {
            let npc = et.nodes_per_cell();
            for i in 0..n_cells {
                let base = npc * i;
                out.push(Primitive::Face {
                    verts: vec![pts[base], pts[base + 1], pts[base + 2], pts[base + 3]],
                    color: cell_color(i),
                    outline: true,
                });
            }
        }
        ElementType::TET10 | ElementType::HEX20 | ElementType::HEX27 | ElementType::PENTA15 => {
            let faces: &[&[usize]] = match et {
                ElementType::TET10 => &[&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[1, 2, 3]],
                ElementType::HEX20 | ElementType::HEX27 => &[
                    &[0, 3, 2, 1],
                    &[4, 5, 6, 7],
                    &[0, 1, 5, 4],
                    &[1, 2, 6, 5],
                    &[2, 3, 7, 6],
                    &[3, 0, 4, 7],
                ],
                _ => &PENTA6_FACES, // PENTA15
            };
            let npc = et.nodes_per_cell();
            for i in 0..n_cells {
                let base = npc * i;
                let c = cell_color(i);
                for (fi, face) in faces.iter().enumerate() {
                    if keep.as_ref().is_some_and(|k| !k.contains(&(i, fi))) {
                        continue;
                    }
                    out.push(Primitive::Face {
                        verts: face.iter().map(|&li| pts[base + li]).collect(),
                        color: c,
                        outline: true,
                    });
                }
            }
        }
    }

    Ok(out)
}

/// Bbox covering every vertex used by `prims`.
pub(crate) fn primitives_bbox(prims: &[Primitive]) -> Bbox3 {
    let mut bb = Bbox3::empty();
    for prim in prims {
        match prim {
            Primitive::Point { p, .. } => bb.extend(*p),
            Primitive::Segment { a, b, .. } => {
                bb.extend(*a);
                bb.extend(*b);
            }
            Primitive::Face { verts, .. } | Primitive::Wire { verts } => {
                for v in verts {
                    bb.extend(*v);
                }
            }
        }
    }
    bb
}

// ─── Renderer ───────────────────────────────────────────────────────────────

/// Projected primitive: 2-D screen coordinates + a single depth value used
/// to order the painter's pass.
#[derive(Debug, Clone)]
enum ProjPrim {
    Point {
        p: (f64, f64),
        color: RgbColor,
        depth: f64,
    },
    Segment {
        a: (f64, f64),
        b: (f64, f64),
        color: RgbColor,
        depth: f64,
    },
    Face {
        verts: Vec<(f64, f64)>,
        color: RgbColor,
        depth: f64,
        outline: bool,
    },
    Wire {
        verts: Vec<(f64, f64)>,
        depth: f64,
    },
}

impl ProjPrim {
    fn depth(&self) -> f64 {
        match self {
            ProjPrim::Point { depth, .. }
            | ProjPrim::Segment { depth, .. }
            | ProjPrim::Face { depth, .. }
            | ProjPrim::Wire { depth, .. } => *depth,
        }
    }
}

// ─── Viewport clipping ──────────────────────────────────────────────────────
//
// `plotters`' cartesian coordinate maps a value *outside* the axis range onto
// the range boundary (it clamps, it does not drop the point). Left as-is, a
// node that leaves the visible window while zooming or panning gets pinned to
// the window edge instead of disappearing — it looks like the mesh is being
// dragged and deformed along the border. We therefore clip every primitive to
// the visible rectangle *before* handing it to plotters: fully-outside
// primitives are dropped, and segments / polygons are cut at the boundary.

/// The visible rectangle in projected (screen) coordinates.
#[derive(Debug, Clone, Copy)]
struct Viewport {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
}

impl Viewport {
    fn contains(&self, (x, y): (f64, f64)) -> bool {
        x >= self.xmin && x <= self.xmax && y >= self.ymin && y <= self.ymax
    }

    /// Liang–Barsky segment clip. Returns the visible sub-segment, or `None`
    /// when the segment lies entirely outside the viewport.
    fn clip_segment(&self, a: (f64, f64), b: (f64, f64)) -> Option<((f64, f64), (f64, f64))> {
        let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        // Each boundary as (p, q): the point is inside when p*t <= q.
        let checks = [
            (-dx, a.0 - self.xmin),
            (dx, self.xmax - a.0),
            (-dy, a.1 - self.ymin),
            (dy, self.ymax - a.1),
        ];
        for (p, q) in checks {
            if p == 0.0 {
                // Line parallel to this boundary: reject if outside it.
                if q < 0.0 {
                    return None;
                }
            } else {
                let r = q / p;
                if p < 0.0 {
                    if r > t1 {
                        return None;
                    }
                    if r > t0 {
                        t0 = r;
                    }
                } else {
                    if r < t0 {
                        return None;
                    }
                    if r < t1 {
                        t1 = r;
                    }
                }
            }
        }
        let ca = (a.0 + t0 * dx, a.1 + t0 * dy);
        let cb = (a.0 + t1 * dx, a.1 + t1 * dy);
        Some((ca, cb))
    }

    /// Is `p` on the kept side of boundary `b` (0=left, 1=right, 2=bottom,
    /// 3=top)?
    fn inside(&self, b: usize, p: (f64, f64)) -> bool {
        match b {
            0 => p.0 >= self.xmin,
            1 => p.0 <= self.xmax,
            2 => p.1 >= self.ymin,
            _ => p.1 <= self.ymax,
        }
    }

    /// Crossing point of edge `s`→`e` with boundary line `b`.
    fn intersect(&self, b: usize, s: (f64, f64), e: (f64, f64)) -> (f64, f64) {
        match b {
            0 => intersect_x(s, e, self.xmin),
            1 => intersect_x(s, e, self.xmax),
            2 => intersect_y(s, e, self.ymin),
            _ => intersect_y(s, e, self.ymax),
        }
    }

    /// Sutherland–Hodgman polygon clip against the viewport rectangle.
    /// Returns the (possibly empty) clipped polygon.
    fn clip_polygon(&self, verts: &[(f64, f64)]) -> Vec<(f64, f64)> {
        // Clip successively against each of the four half-planes (left, right,
        // bottom, top).
        let mut poly = verts.to_vec();
        for b in 0..4 {
            if poly.is_empty() {
                break;
            }
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(poly.len() + 1);
            for i in 0..poly.len() {
                let cur = poly[i];
                let prev = poly[(i + poly.len() - 1) % poly.len()];
                let cur_in = self.inside(b, cur);
                let prev_in = self.inside(b, prev);
                if cur_in {
                    if !prev_in {
                        out.push(self.intersect(b, prev, cur));
                    }
                    out.push(cur);
                } else if prev_in {
                    out.push(self.intersect(b, prev, cur));
                }
            }
            poly = out;
        }
        poly
    }
}

/// Intersection of segment `s`→`e` with the vertical line `x = xb`.
fn intersect_x(s: (f64, f64), e: (f64, f64), xb: f64) -> (f64, f64) {
    let dx = e.0 - s.0;
    if dx == 0.0 {
        return (xb, s.1);
    }
    let t = (xb - s.0) / dx;
    (xb, s.1 + t * (e.1 - s.1))
}

/// Intersection of segment `s`→`e` with the horizontal line `y = yb`.
fn intersect_y(s: (f64, f64), e: (f64, f64), yb: f64) -> (f64, f64) {
    let dy = e.1 - s.1;
    if dy == 0.0 {
        return (s.0, yb);
    }
    let t = (yb - s.1) / dy;
    (s.0 + t * (e.0 - s.0), yb)
}

/// Clip every edge of a closed vertex loop against the viewport, returning
/// the visible sub-segments. Each edge is clipped on its own so an edge that
/// leaves and re-enters the window doesn't get bridged by a false chord.
fn clip_loop_edges(verts: &[(f64, f64)], vp: &Viewport) -> Vec<((f64, f64), (f64, f64))> {
    let n = verts.len();
    if n < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        if let Some(seg) = vp.clip_segment(a, b) {
            out.push(seg);
        }
    }
    out
}

/// Common rendering core: project, sort far → near, draw points / segments
/// / filled faces with black face boundaries on top. Shared by `SubMesh`
/// and `Mesh`.
pub(crate) fn render_primitives<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    view: &View,
    prims: &[Primitive],
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    if prims.is_empty() {
        return Ok(());
    }

    // Axisymmetric plots may be swept into the body of revolution they
    // describe first: every backend and every colouring path funnels through
    // here, so the sweep is written once (see `crate::viz::revolve`).
    let swept;
    let prims = match view.revolve {
        Some(rev) => {
            swept = crate::viz::revolve::revolve_primitives(prims, rev);
            &swept[..]
        }
        None => prims,
    };

    let bbox = primitives_bbox(prims);
    let proj = Projector::new(view, bbox.center());

    let mut projected: Vec<ProjPrim> = prims
        .iter()
        .map(|prim| match prim {
            Primitive::Point { p, color } => {
                let v = proj.project(*p);
                ProjPrim::Point {
                    p: (v.x, v.y),
                    color: *color,
                    depth: v.z,
                }
            }
            Primitive::Segment { a, b, color } => {
                let va = proj.project(*a);
                let vb = proj.project(*b);
                ProjPrim::Segment {
                    a: (va.x, va.y),
                    b: (vb.x, vb.y),
                    color: *color,
                    depth: 0.5 * (va.z + vb.z),
                }
            }
            Primitive::Face {
                verts,
                color,
                outline,
            } => {
                let projected_verts: Vec<_> = verts.iter().map(|v| proj.project(*v)).collect();
                let n = projected_verts.len().max(1);
                let depth: f64 = projected_verts.iter().map(|v| v.z).sum::<f64>() / n as f64;
                let v2d: Vec<(f64, f64)> = projected_verts.iter().map(|v| (v.x, v.y)).collect();
                ProjPrim::Face {
                    verts: v2d,
                    color: *color,
                    depth,
                    outline: *outline,
                }
            }
            Primitive::Wire { verts } => {
                let projected_verts: Vec<_> = verts.iter().map(|v| proj.project(*v)).collect();
                let n = projected_verts.len().max(1);
                // Small bias towards the viewer so the wire is drawn on
                // top of the coplanar sub-faces it delimits.
                let depth: f64 = projected_verts.iter().map(|v| v.z).sum::<f64>() / n as f64
                    - 1e-3 * bbox.diagonal().max(1.0);
                let v2d: Vec<(f64, f64)> = projected_verts.iter().map(|v| (v.x, v.y)).collect();
                ProjPrim::Wire { verts: v2d, depth }
            }
        })
        .collect();

    // Painter's algorithm: far first.
    projected.sort_by(|x, y| {
        y.depth()
            .partial_cmp(&x.depth())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Size the viewport from the 3-D bbox diagonal — invariant under the
    // camera's yaw/pitch — so the figure doesn't "breathe" while the user
    // drags to rotate. The center of the projection of bbox.center() is
    // (0, 0) by construction (it's the Projector target). For a degenerate
    // bbox (a single point), fall back to a unit-sized viewport so the
    // point is still visible.
    let diag = bbox.diagonal();
    let radius = if diag.is_finite() && diag > 0.0 {
        0.5 * diag
    } else {
        1.0
    };

    let (w, h) = area.dim_in_pixel();
    let aspect = if h == 0 { 1.0 } else { w as f64 / h as f64 };
    let scale = view.scale.max(1e-6);
    let mut dx = (2.0 * radius / scale).max(1e-9);
    let mut dy = (2.0 * radius / scale).max(1e-9);
    if dx / dy > aspect {
        dy = dx / aspect;
    } else {
        dx = dy * aspect;
    }
    let margin = 0.05;
    dx *= 1.0 + 2.0 * margin;
    dy *= 1.0 + 2.0 * margin;
    let xmin = -dx / 2.0;
    let xmax = dx / 2.0;
    let ymin = -dy / 2.0;
    let ymax = dy / 2.0;
    let viewport = Viewport {
        xmin,
        xmax,
        ymin,
        ymax,
    };

    let mut chart = ChartBuilder::on(area)
        .margin(5)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(pl_err)?;
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_axes()
        .draw()
        .map_err(pl_err)?;

    let edge_style = ShapeStyle::from(&BLACK).stroke_width(1);

    for p in &projected {
        match p {
            ProjPrim::Point { p, color, .. } => {
                // A point has no extent to clip; just drop it when off-screen
                // so it never gets pinned to the window border.
                if !viewport.contains(*p) {
                    continue;
                }
                let style = ShapeStyle {
                    color: RGBAColor(color.r, color.g, color.b, 1.0),
                    filled: true,
                    stroke_width: 0,
                };
                chart
                    .draw_series(std::iter::once(Circle::new(*p, 3, style)))
                    .map_err(pl_err)?;
            }
            ProjPrim::Segment { a, b, color, .. } => {
                let Some((ca, cb)) = viewport.clip_segment(*a, *b) else {
                    continue;
                };
                let style = ShapeStyle {
                    color: RGBAColor(color.r, color.g, color.b, 1.0),
                    filled: false,
                    stroke_width: 2,
                };
                chart
                    .draw_series(LineSeries::new(vec![ca, cb], style))
                    .map_err(pl_err)?;
            }
            ProjPrim::Face {
                verts,
                color,
                outline,
                ..
            } => {
                // Clip the polygon to the viewport so a partially-visible face
                // stays inside the window instead of being clamped to the edge.
                let clipped = viewport.clip_polygon(verts);
                if clipped.len() < 3 {
                    continue;
                }
                // Faces are opaque so the painter's pass performs hidden-
                // surface removal: a near face fully overwrites the ones
                // behind it. This is what makes a solid 3-D mesh read as a
                // solid instead of a translucent shell you can see through.
                let face_rgba = RGBAColor(color.r, color.g, color.b, 1.0);
                let face_style = ShapeStyle {
                    color: face_rgba,
                    filled: true,
                    stroke_width: 0,
                };
                chart
                    .draw_series(std::iter::once(Polygon::new(clipped.clone(), face_style)))
                    .map_err(pl_err)?;
                if *outline {
                    // Draw the outline edge-by-edge, clipping each edge so the
                    // boundary follows the window instead of the axis border.
                    for (ca, cb) in clip_loop_edges(verts, &viewport) {
                        chart
                            .draw_series(LineSeries::new(vec![ca, cb], edge_style))
                            .map_err(pl_err)?;
                    }
                }
            }
            ProjPrim::Wire { verts, .. } => {
                for (ca, cb) in clip_loop_edges(verts, &viewport) {
                    chart
                        .draw_series(LineSeries::new(vec![ca, cb], edge_style))
                        .map_err(pl_err)?;
                }
            }
        }
    }

    Ok(())
}

// ─── Drawable for SubMesh ───────────────────────────────────────────────────

impl Drawable for SubMesh {
    fn bbox(&self) -> Result<Bbox3> {
        let coords = self.coords();
        let pts = read_points(&coords, self.connectivity())?;
        let mut b = Bbox3::empty();
        for p in pts {
            b.extend(p);
        }
        Ok(b)
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let prims = submesh_primitives(self)?;
        render_primitives(area, view, &prims)
    }

    fn is_axisymmetric(&self) -> bool {
        read(&self.coords())
            .map(|c| c.is_axisymmetric())
            .unwrap_or(false)
    }
}

// ─── Drawable for Mesh ──────────────────────────────────────────────────────

impl Drawable for Mesh {
    fn bbox(&self) -> Result<Bbox3> {
        let mut b = Bbox3::empty();
        for i in 0..self.len() {
            let sm = self.get(i)?;
            let smb = read(&sm)?.bbox()?;
            if !smb.is_empty() {
                b.extend(smb.min);
                b.extend(smb.max);
            }
        }
        Ok(b)
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let mut all = Vec::new();
        for i in 0..self.len() {
            let sm = self.get(i)?;
            let mut prims = submesh_primitives(&*read(&sm)?)?;
            all.append(&mut prims);
        }
        render_primitives(area, view, &all)
    }

    fn is_axisymmetric(&self) -> bool {
        self.coords()
            .and_then(|c| Ok(read(&c)?.is_axisymmetric()))
            .unwrap_or(false)
    }
}

// ─── Wireframe style ──────────────────────────────────────────────────────

/// Wireframe primitives of a submesh: **every** distinct element edge as a
/// segment — the interior edges of volume cells included — plus a dot per
/// POI1. Edges shared by several cells of the submesh are emitted once.
/// Unlike the surface style, nothing is hidden: this is the see-through
/// "fil de fer" rendering. Lines take the submesh's `face_color` (as SEG2
/// cells already do), so the components of a `Mesh` stay distinguishable.
pub(crate) fn submesh_wireframe_primitives(sm: &SubMesh) -> Result<Vec<Primitive>> {
    let coords = sm.coords();
    let conn = sm.connectivity();
    let pts = read_points(&coords, conn)?;
    let et = sm.element_type();
    let npc = et.nodes_per_cell();
    let color = sm.face_color();
    let mut out = Vec::new();
    if et == ElementType::POI1 {
        for &p in &pts {
            out.push(Primitive::Point { p, color });
        }
        return Ok(out);
    }
    let edges = element_edges(et);
    let n_cells = pts.len().checked_div(npc).unwrap_or(0);
    // Deduplicate edges by their (sorted) global node-id pair.
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for cell in 0..n_cells {
        let base = cell * npc;
        for e in edges {
            let (mut ga, mut gb) = (conn[base + e[0]].0, conn[base + e[1]].0);
            if ga > gb {
                std::mem::swap(&mut ga, &mut gb);
            }
            if seen.insert((ga, gb)) {
                out.push(Primitive::Segment {
                    a: pts[base + e[0]],
                    b: pts[base + e[1]],
                    color,
                });
            }
        }
    }
    Ok(out)
}

/// Wireframe `Drawable` wrapper over a [`SubMesh`] — same bbox as the
/// solid view, but draws all edges instead of filled faces.
pub(crate) struct SubMeshWire<'a>(pub &'a SubMesh);

impl Drawable for SubMeshWire<'_> {
    fn bbox(&self) -> Result<Bbox3> {
        self.0.bbox()
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let prims = submesh_wireframe_primitives(self.0)?;
        render_primitives(area, view, &prims)
    }

    fn is_axisymmetric(&self) -> bool {
        self.0.is_axisymmetric()
    }
}

/// Wireframe `Drawable` wrapper over a [`Mesh`] — every submesh drawn as
/// edges, each in its own `face_color`.
pub(crate) struct MeshWire<'a>(pub &'a Mesh);

impl Drawable for MeshWire<'_> {
    fn bbox(&self) -> Result<Bbox3> {
        self.0.bbox()
    }

    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let mut all = Vec::new();
        for i in 0..self.0.len() {
            let sm = self.0.get(i)?;
            all.extend(submesh_wireframe_primitives(&*read(&sm)?)?);
        }
        render_primitives(area, view, &all)
    }

    fn is_axisymmetric(&self) -> bool {
        self.0.is_axisymmetric()
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::store::insert;

    fn unit_viewport() -> Viewport {
        Viewport {
            xmin: 0.0,
            xmax: 1.0,
            ymin: 0.0,
            ymax: 1.0,
        }
    }

    #[test]
    fn clip_segment_fully_inside_is_unchanged() {
        let vp = unit_viewport();
        let (a, b) = vp.clip_segment((0.2, 0.2), (0.8, 0.8)).unwrap();
        assert_eq!(a, (0.2, 0.2));
        assert_eq!(b, (0.8, 0.8));
    }

    #[test]
    fn clip_segment_fully_outside_is_dropped() {
        let vp = unit_viewport();
        assert!(vp.clip_segment((2.0, 2.0), (3.0, 3.0)).is_none());
        // Parallel to and outside a boundary (both endpoints left of xmin).
        assert!(vp.clip_segment((-1.0, 0.2), (-1.0, 0.8)).is_none());
    }

    #[test]
    fn clip_segment_crossing_is_cut_at_boundary() {
        let vp = unit_viewport();
        // From centre out through the right edge.
        let (a, b) = vp.clip_segment((0.5, 0.5), (1.5, 0.5)).unwrap();
        assert_eq!(a, (0.5, 0.5));
        assert!((b.0 - 1.0).abs() < 1e-12);
        assert!((b.1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn clip_polygon_keeps_inner_part() {
        let vp = unit_viewport();
        // A triangle poking out the top-right corner.
        let tri = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)];
        let clipped = vp.clip_polygon(&tri);
        // Clipped polygon stays within the viewport.
        assert!(clipped.len() >= 3);
        for p in &clipped {
            assert!(p.0 >= vp.xmin - 1e-9 && p.0 <= vp.xmax + 1e-9);
            assert!(p.1 >= vp.ymin - 1e-9 && p.1 <= vp.ymax + 1e-9);
        }
    }

    #[test]
    fn clip_polygon_fully_outside_is_empty() {
        let vp = unit_viewport();
        let tri = [(2.0, 2.0), (3.0, 2.0), (2.0, 3.0)];
        assert!(vp.clip_polygon(&tri).len() < 3);
    }

    #[test]
    fn world_per_pixel_scales_with_zoom() {
        use crate::viz::camera::world_per_pixel;
        let mut bb = Bbox3::empty();
        bb.extend(Point3::new(0.0, 0.0, 0.0));
        bb.extend(Point3::new(1.0, 1.0, 1.0));
        let mut view = crate::viz::View::iso();
        let wpp1 = world_per_pixel(&view, &bb, 800, 600);
        view.scale = 2.0;
        let wpp2 = world_per_pixel(&view, &bb, 800, 600);
        // Zooming in (larger scale) shrinks the world span per pixel.
        assert!(wpp2 < wpp1);
        assert!((wpp1 / wpp2 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn bbox_covers_all_nodes_2d() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 3.0]).unwrap();

        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let bb = sm.bbox().unwrap();
        assert_eq!(bb.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(bb.max, Point3::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn primitives_poi1_emits_points() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 2.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.add_cell(&[b.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 2);
        assert!(matches!(prims[0], Primitive::Point { .. }));
        assert!(matches!(prims[1], Primitive::Point { .. }));
    }

    #[test]
    fn primitives_seg2_emits_segments() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 1);
        assert!(matches!(prims[0], Primitive::Segment { .. }));
    }

    #[test]
    fn primitives_qua4_emits_one_face_per_cell() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::QUA4);
        sm.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 1);
        match &prims[0] {
            Primitive::Face { verts, .. } => assert_eq!(verts.len(), 4),
            other => panic!("expected Face, got {:?}", other),
        }
    }

    #[test]
    fn primitives_tet4_emits_four_triangular_faces() {
        let coords = insert(Coords::new(3).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        sm.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 4);
        for p in &prims {
            match p {
                Primitive::Face { verts, .. } => assert_eq!(verts.len(), 3),
                other => panic!("expected triangular Face, got {:?}", other),
            }
        }
    }

    /// Six tets fanned around the main diagonal of a cube: the six faces
    /// containing that diagonal are each shared by two tets (interior),
    /// the other twelve are the cube's skin (boundary).
    fn cube_six_tets() -> SubMesh {
        let coords = insert(Coords::new(3).unwrap());
        #[rustfmt::skip]
        let corners = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let nodes: Vec<_> = corners
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        #[rustfmt::skip]
        let tets = [
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
        ];
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        for t in &tets {
            sm.add_cell(&[
                nodes[t[0]].id(),
                nodes[t[1]].id(),
                nodes[t[2]].id(),
                nodes[t[3]].id(),
            ])
            .unwrap();
        }
        sm
    }

    #[test]
    fn boundary_faces_keeps_only_the_skin() {
        let sm = cube_six_tets();
        let keep = boundary_faces(ElementType::TET4, sm.connectivity()).unwrap();
        // 6 tets × 4 faces = 24; 6 interior faces shared by two cells
        // remove 12 instances, leaving 12 boundary triangles.
        assert_eq!(keep.len(), 12);
    }

    #[test]
    fn boundary_faces_culls_shared_hex_face() {
        // Two unit hexes stacked along Z share their common quad: that
        // face is interior, the other ten are boundary.
        let coords = insert(Coords::new(3).unwrap());
        #[rustfmt::skip]
        let pts = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0],
        ];
        let n: Vec<_> = pts
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords, ElementType::HEX8);
        sm.add_cell(&(0..8).map(|i| n[i].id()).collect::<Vec<_>>())
            .unwrap();
        sm.add_cell(&(4..12).map(|i| n[i].id()).collect::<Vec<_>>())
            .unwrap();
        let keep = boundary_faces(ElementType::HEX8, sm.connectivity()).unwrap();
        // 2 × 6 = 12 faces, one shared pair removed → 10 boundary quads.
        assert_eq!(keep.len(), 10);
    }

    #[test]
    fn boundary_faces_is_none_for_surface_types() {
        // Surface / line / point types never hide faces behind a cell.
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        assert!(boundary_faces(ElementType::TRI3, sm.connectivity()).is_none());
    }

    #[test]
    fn tet_mesh_primitives_drop_interior_faces() {
        let sm = cube_six_tets();
        let prims = submesh_primitives(&sm).unwrap();
        // Only the 12 boundary triangles are emitted, not all 24 faces.
        assert_eq!(prims.len(), 12);
        for p in &prims {
            assert!(matches!(p, Primitive::Face { verts, .. } if verts.len() == 3));
        }
    }

    #[test]
    fn primitives_hex8_emits_six_quadrangular_faces() {
        let coords = insert(Coords::new(3).unwrap());
        let n: Vec<_> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|c| Node::create_in(coords.clone(), c).unwrap())
        .collect();
        let mut sm = SubMesh::new(coords, ElementType::HEX8);
        let ids: Vec<_> = n.iter().map(|nn| nn.id()).collect();
        sm.add_cell(&ids).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 6);
        for p in &prims {
            match p {
                Primitive::Face { verts, .. } => assert_eq!(verts.len(), 4),
                other => panic!("expected quadrangular Face, got {:?}", other),
            }
        }
    }

    #[test]
    fn submesh_primitives_carries_face_color() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.set_face_color(RgbColor::new(10, 20, 30));
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 1);
        match &prims[0] {
            Primitive::Face { color, .. } => {
                assert_eq!(*color, RgbColor::new(10, 20, 30));
            }
            other => panic!("expected Face, got {:?}", other),
        }
    }

    #[test]
    fn wireframe_emits_every_distinct_edge_including_interior() {
        // The 6-tet cube has 12 cube edges + 6 face diagonals + 1 space
        // diagonal (the shared 0–6 edge, purely interior) = 19 distinct
        // edges. The surface style would never show that interior edge.
        let sm = cube_six_tets();
        let prims = submesh_wireframe_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 19);
        for p in &prims {
            assert!(matches!(p, Primitive::Segment { .. }));
        }
    }

    #[test]
    fn wireframe_of_one_triangle_is_three_edges() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let prims = submesh_wireframe_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 3);
    }

    #[test]
    fn wireframe_of_poi1_is_points() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        let prims = submesh_wireframe_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 1);
        assert!(matches!(prims[0], Primitive::Point { .. }));
    }

    #[test]
    fn wireframe_shares_edges_between_adjacent_cells() {
        // Two triangles sharing edge (b, c): 3 + 3 − 1 shared = 5 edges.
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.add_cell(&[b.id(), d.id(), c.id()]).unwrap();
        let prims = submesh_wireframe_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 5);
    }

    #[test]
    fn pad3_truncates_and_pads() {
        assert_eq!(pad3(&[]), Point3::new(0.0, 0.0, 0.0));
        assert_eq!(pad3(&[1.0]), Point3::new(1.0, 0.0, 0.0));
        assert_eq!(pad3(&[1.0, 2.0]), Point3::new(1.0, 2.0, 0.0));
        assert_eq!(pad3(&[1.0, 2.0, 3.0]), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(pad3(&[1.0, 2.0, 3.0, 4.0]), Point3::new(1.0, 2.0, 3.0));
    }
}
