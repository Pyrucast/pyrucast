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

use crate::containers::mesh::color::RgbColor;
use crate::containers::mesh::configuration::{Configuration, NodeId};
use crate::containers::mesh::element_type::ElementType;
use crate::error::Result;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{with, Handle};
use crate::viz::camera::{Bbox3, Projector};
use crate::viz::drawable::{pl_err, Drawable};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::containers::mesh::point::Point3;

/// Pad world coordinates to a 3-D point, filling missing components with 0.0.
fn pad3(coords: &[f64]) -> Point3 {
    Point3::new(
        coords.first().copied().unwrap_or(0.0),
        coords.get(1).copied().unwrap_or(0.0),
        coords.get(2).copied().unwrap_or(0.0),
    )
}

/// Read every node referenced by `connectivity` from `config`, padded to 3-D.
fn read_points(
    config: &Handle<Configuration>,
    connectivity: &[NodeId],
) -> Result<Vec<Point3>> {
    with(config, |c| -> Result<Vec<Point3>> {
        connectivity
            .iter()
            .map(|&nid| c.coord(nid).map(pad3))
            .collect()
    })?
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
    Face { verts: Vec<Point3>, color: RgbColor },
}

// ─── Element-type → primitives ──────────────────────────────────────────────

/// Faces of a TET4 — each oriented outwards (CCW seen from outside).
const TET4_FACES: [[usize; 3]; 4] = [
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
];

/// Faces of a HEX8 — bot / top / 4 lateral, in the convention used by
/// [`crate::ops::mesher::extrude`]: HEX8 = [bot[0..4], top[0..4]], both CCW seen from
/// outside the lateral surface.
const HEX8_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1], // bottom (normal opposed to extrusion direction)
    [4, 5, 6, 7], // top
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

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
    let cfg = sm.configuration();
    let pts = read_points(&cfg, sm.connectivity())?;
    let default_color = sm.face_color();
    let et = sm.element_type();
    let npc = et.nodes_per_cell();
    let n_cells = if npc == 0 { 0 } else { pts.len() / npc };
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
                });
            }
        }
        ElementType::TET4 => {
            for i in 0..n_cells {
                let base = 4 * i;
                let c = cell_color(i);
                for face in &TET4_FACES {
                    out.push(Primitive::Face {
                        verts: vec![
                            pts[base + face[0]],
                            pts[base + face[1]],
                            pts[base + face[2]],
                        ],
                        color: c,
                    });
                }
            }
        }
        ElementType::HEX8 => {
            for i in 0..n_cells {
                let base = 8 * i;
                let c = cell_color(i);
                for face in &HEX8_FACES {
                    out.push(Primitive::Face {
                        verts: vec![
                            pts[base + face[0]],
                            pts[base + face[1]],
                            pts[base + face[2]],
                            pts[base + face[3]],
                        ],
                        color: c,
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
            Primitive::Face { verts, .. } => {
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
    Face { verts: Vec<(f64, f64)>, color: RgbColor, depth: f64 },
}

impl ProjPrim {
    fn depth(&self) -> f64 {
        match self {
            ProjPrim::Point { depth, .. }
            | ProjPrim::Segment { depth, .. }
            | ProjPrim::Face { depth, .. } => *depth,
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
            Primitive::Face { verts, color } => {
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
                }
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
            ProjPrim::Face { verts, color, .. } => {
                let face_rgba = RGBAColor(color.r, color.g, color.b, 0.85);
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
    }

    Ok(())
}

// ─── Drawable for SubMesh ───────────────────────────────────────────────────

impl Drawable for SubMesh {
    fn bbox(&self) -> Result<Bbox3> {
        let cfg = self.configuration();
        let pts = read_points(&cfg, self.connectivity())?;
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
        for i in 0..self.submesh_count() {
            let sm = self.submesh(i)?;
            let smb = with(&sm, |s| s.bbox())??;
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
        for i in 0..self.submesh_count() {
            let sm = self.submesh(i)?;
            let mut prims = with(&sm, |s| submesh_primitives(s))??;
            all.append(&mut prims);
        }
        render_primitives(area, view, &all)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::configuration::Configuration;
    use crate::containers::mesh::node::Node;
    use crate::store::insert;

    #[test]
    fn bbox_covers_all_nodes_2d() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[1.0, 3.0]).unwrap();

        let mut sm = SubMesh::new(cfg, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let bb = sm.bbox().unwrap();
        assert_eq!(bb.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(bb.max, Point3::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn primitives_poi1_emits_points() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 2.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.add_cell(&[b.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 2);
        assert!(matches!(prims[0], Primitive::Point { .. }));
        assert!(matches!(prims[1], Primitive::Point { .. }));
    }

    #[test]
    fn primitives_seg2_emits_segments() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let prims = submesh_primitives(&sm).unwrap();
        assert_eq!(prims.len(), 1);
        assert!(matches!(prims[0], Primitive::Segment { .. }));
    }

    #[test]
    fn primitives_qua4_emits_one_face_per_cell() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::QUA4);
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
        let cfg = insert(Configuration::new(3).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.0]).unwrap();
        let d = Node::create_in(cfg.clone(), &[0.0, 0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::TET4);
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

    #[test]
    fn primitives_hex8_emits_six_quadrangular_faces() {
        let cfg = insert(Configuration::new(3).unwrap());
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
        .map(|c| Node::create_in(cfg.clone(), c).unwrap())
        .collect();
        let mut sm = SubMesh::new(cfg, ElementType::HEX8);
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
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::TRI3);
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
    fn pad3_truncates_and_pads() {
        assert_eq!(pad3(&[]), Point3::new(0.0, 0.0, 0.0));
        assert_eq!(pad3(&[1.0]), Point3::new(1.0, 0.0, 0.0));
        assert_eq!(pad3(&[1.0, 2.0]), Point3::new(1.0, 2.0, 0.0));
        assert_eq!(pad3(&[1.0, 2.0, 3.0]), Point3::new(1.0, 2.0, 3.0));
        assert_eq!(pad3(&[1.0, 2.0, 3.0, 4.0]), Point3::new(1.0, 2.0, 3.0));
    }
}
