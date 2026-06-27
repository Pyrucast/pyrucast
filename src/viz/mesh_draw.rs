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
use crate::containers::mesh::RgbColor;
use crate::containers::mesh::{Coords, NodeId};
use crate::containers::mesh::ElementType;
use crate::error::Result;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{read, Handle};
use crate::viz::camera::{Bbox3, Projector};
use crate::viz::drawable::{pl_err, Drawable};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::containers::mesh::Point3;
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
fn read_points(
    coords: &Handle<Coords>,
    connectivity: &[NodeId],
) -> Result<Vec<Point3>> {
    let c = read(coords)?;
    connectivity
        .iter()
        .map(|&nid| c.coord(nid).map(pad3))
        .collect()
}

/// Geometric primitive emitted by a submesh and consumed by the painter.
///
/// `Face::verts` is 3 (triangle) or 4 (quadrangle) vertices in CCW order
/// when seen from outside the solid; the renderer fills the polygon and
/// then overlays its boundary as a black wireframe.
#[derive(Debug, Clone)]
pub(crate) enum Primitive {
    Point { p: Point3, color: RgbColor },
    Segment { a: Point3, b: Point3, color: RgbColor },
    /// Filled polygon; `outline: false` skips the black wireframe (used
    /// by the interpolated renderer for its interior sub-faces).
    Face { verts: Vec<Point3>, color: RgbColor, outline: bool },
    /// Stroke-only closed polyline, drawn slightly towards the viewer —
    /// the element boundary on top of its (outline-free) sub-faces.
    Wire { verts: Vec<Point3> },
}

// ─── Element-type → primitives ──────────────────────────────────────────────

/// Faces of a TET4 — each oriented outwards (CCW seen from outside).
pub(crate) const TET4_FACES: [[usize; 3]; 4] = [
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
];

/// Faces of a HEX8 — bot / top / 4 lateral, in the convention used by
/// [`crate::ops::mesher::extrude`]: HEX8 = [bot[0..4], top[0..4]], both CCW seen from
/// outside the lateral surface.
pub(crate) const HEX8_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1], // bottom (normal opposed to extrusion direction)
    [4, 5, 6, 7], // top
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

/// Edges of each element type, as local node-index pairs — used by the
/// wireframe rendering style. POI1 has no edge (it draws as a dot).
fn element_edges(et: ElementType) -> &'static [[usize; 2]] {
    match et {
        ElementType::POI1 => &[],
        ElementType::SEG2 => &[[0, 1]],
        ElementType::TRI3 => &[[0, 1], [1, 2], [2, 0]],
        ElementType::QUA4 => &[[0, 1], [1, 2], [2, 3], [3, 0]],
        ElementType::TET4 => &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        ElementType::HEX8 => &[
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
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
pub(crate) fn boundary_faces(
    et: ElementType,
    conn: &[NodeId],
) -> Option<HashSet<(usize, usize)>> {
    let faces: Vec<&[usize]> = match et {
        ElementType::TET4 => TET4_FACES.iter().map(|f| f.as_slice()).collect(),
        ElementType::HEX8 => HEX8_FACES.iter().map(|f| f.as_slice()).collect(),
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
                out.push(Primitive::Point { p: pts[i], color: cell_color(i) });
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
                    verts: vec![
                        pts[4 * i],
                        pts[4 * i + 1],
                        pts[4 * i + 2],
                        pts[4 * i + 3],
                    ],
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
    }

    Ok(out)
}

/// Bbox covering every vertex used by `prims`.
fn primitives_bbox(prims: &[Primitive]) -> Bbox3 {
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
    Point { p: (f64, f64), color: RgbColor, depth: f64 },
    Segment { a: (f64, f64), b: (f64, f64), color: RgbColor, depth: f64 },
    Face { verts: Vec<(f64, f64)>, color: RgbColor, depth: f64, outline: bool },
    Wire { verts: Vec<(f64, f64)>, depth: f64 },
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
            Primitive::Face { verts, color, outline } => {
                let projected_verts: Vec<_> =
                    verts.iter().map(|v| proj.project(*v)).collect();
                let n = projected_verts.len().max(1);
                let depth: f64 =
                    projected_verts.iter().map(|v| v.z).sum::<f64>() / n as f64;
                let v2d: Vec<(f64, f64)> =
                    projected_verts.iter().map(|v| (v.x, v.y)).collect();
                ProjPrim::Face {
                    verts: v2d,
                    color: *color,
                    depth,
                    outline: *outline,
                }
            }
            Primitive::Wire { verts } => {
                let projected_verts: Vec<_> =
                    verts.iter().map(|v| proj.project(*v)).collect();
                let n = projected_verts.len().max(1);
                // Small bias towards the viewer so the wire is drawn on
                // top of the coplanar sub-faces it delimits.
                let depth: f64 = projected_verts.iter().map(|v| v.z).sum::<f64>()
                    / n as f64
                    - 1e-3 * bbox.diagonal().max(1.0);
                let v2d: Vec<(f64, f64)> =
                    projected_verts.iter().map(|v| (v.x, v.y)).collect();
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
                let style = ShapeStyle {
                    color: RGBAColor(color.r, color.g, color.b, 1.0),
                    filled: false,
                    stroke_width: 2,
                };
                chart
                    .draw_series(LineSeries::new(vec![*a, *b], style))
                    .map_err(pl_err)?;
            }
            ProjPrim::Face { verts, color, outline, .. } => {
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
                    .draw_series(std::iter::once(Polygon::new(
                        verts.clone(),
                        face_style,
                    )))
                    .map_err(pl_err)?;
                if *outline {
                    // Close the loop for the wireframe overlay.
                    let mut closed = verts.clone();
                    if let Some(first) = verts.first() {
                        closed.push(*first);
                    }
                    chart
                        .draw_series(LineSeries::new(closed, edge_style))
                        .map_err(pl_err)?;
                }
            }
            ProjPrim::Wire { verts, .. } => {
                let mut closed = verts.clone();
                if let Some(first) = verts.first() {
                    closed.push(*first);
                }
                chart
                    .draw_series(LineSeries::new(closed, edge_style))
                    .map_err(pl_err)?;
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

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let prims = submesh_primitives(self)?;
        render_primitives(area, view, &prims)
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

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
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

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        let prims = submesh_wireframe_primitives(self.0)?;
        render_primitives(area, view, &prims)
    }
}

/// Wireframe `Drawable` wrapper over a [`Mesh`] — every submesh drawn as
/// edges, each in its own `face_color`.
pub(crate) struct MeshWire<'a>(pub &'a Mesh);

impl Drawable for MeshWire<'_> {
    fn bbox(&self) -> Result<Bbox3> {
        self.0.bbox()
    }

    fn draw_on<DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, Shift>,
        view: &View,
    ) -> Result<()>
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
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::Node;
    use crate::store::insert;

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
        let corners = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ];
        let nodes: Vec<_> = corners
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        let tets = [
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
        ];
        let mut sm = SubMesh::new(coords, ElementType::TET4);
        for t in &tets {
            sm.add_cell(&[
                nodes[t[0]].id(), nodes[t[1]].id(), nodes[t[2]].id(), nodes[t[3]].id(),
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
        sm.add_cell(&(0..8).map(|i| n[i].id()).collect::<Vec<_>>()).unwrap();
        sm.add_cell(&(4..12).map(|i| n[i].id()).collect::<Vec<_>>()).unwrap();
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
