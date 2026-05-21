//! `Drawable` implementations for [`SubMesh`] and [`Mesh`].
//!
//! Currently only `TRI3` cells are rendered; other element types are
//! ignored when aggregating a [`Mesh`], and a `SubMesh` of another type
//! returns a clear error when plotted on its own. The rendering pipeline
//! is the painter's algorithm:
//!
//! 1. project every vertex into 2D + depth;
//! 2. sort triangles by mean depth (far → near);
//! 3. fill faces with [`SubMesh::face_color`], then overlay black edges.

use crate::color::RgbColor;
use crate::configuration::{Configuration, NodeId};
use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::mesh::{Mesh, SubMesh};
use crate::store::{with, Handle};
use crate::viz::camera::{Bbox3, Projector};
use crate::viz::drawable::{pl_err, Drawable};
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

/// Pad world coordinates to `[x, y, z]`, filling missing components with 0.0.
fn pad3(coords: &[f64]) -> [f64; 3] {
    let mut p = [0.0; 3];
    for (k, &c) in coords.iter().take(3).enumerate() {
        p[k] = c;
    }
    p
}

/// Read every node referenced by `connectivity` from `config`, padded to 3D.
fn read_points(
    config: &Handle<Configuration>,
    connectivity: &[NodeId],
) -> Result<Vec<[f64; 3]>> {
    with(config, |c| -> Result<Vec<[f64; 3]>> {
        connectivity
            .iter()
            .map(|&nid| c.coord(nid).map(pad3))
            .collect()
    })?
}

/// Triangle with its display colour, expressed in world coordinates.
#[derive(Debug, Clone, Copy)]
struct ColoredTri {
    p: [[f64; 3]; 3],
    color: RgbColor,
}

/// Build the coloured triangles of a `SubMesh`. Returns an empty vector
/// for any element type that isn't `TRI3` (so [`Drawable for Mesh`] can
/// just concatenate without errors).
fn submesh_triangles(sm: &SubMesh) -> Result<Vec<ColoredTri>> {
    if sm.element_type() != ElementType::TRI3 {
        return Ok(Vec::new());
    }
    let cfg = sm.configuration();
    let pts = read_points(&cfg, sm.connectivity())?;
    let color = sm.face_color();
    let n = pts.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(ColoredTri {
            p: [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]],
            color,
        });
    }
    Ok(out)
}

/// Compute the union bbox of a slice of triangles.
fn triangles_bbox(tris: &[ColoredTri]) -> Bbox3 {
    let mut b = Bbox3::empty();
    for t in tris {
        for p in &t.p {
            b.extend(*p);
        }
    }
    b
}

/// Common rendering core: project, sort by depth, draw filled faces then
/// black wireframe overlay. Shared by `SubMesh` and `Mesh`.
fn render_triangles<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    view: &View,
    tris: &[ColoredTri],
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    if tris.is_empty() {
        return Ok(());
    }

    let bbox = triangles_bbox(tris);
    let proj = Projector::new(view, bbox.center());

    // Project every vertex once.
    struct Proj2 {
        a: [f64; 3],
        b: [f64; 3],
        c: [f64; 3],
        color: RgbColor,
        depth: f64,
    }
    let mut projected: Vec<Proj2> = tris
        .iter()
        .map(|t| {
            let a = proj.project(t.p[0]);
            let b = proj.project(t.p[1]);
            let c = proj.project(t.p[2]);
            let depth = (a[2] + b[2] + c[2]) / 3.0;
            Proj2 {
                a,
                b,
                c,
                color: t.color,
                depth,
            }
        })
        .collect();

    // Painter's algorithm: far triangles first.
    projected.sort_by(|x, y| {
        y.depth
            .partial_cmp(&x.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute screen-space bbox, then expand to match the area's aspect ratio
    // and apply the user-supplied scale and a small visual margin.
    let mut sx_min = f64::INFINITY;
    let mut sx_max = f64::NEG_INFINITY;
    let mut sy_min = f64::INFINITY;
    let mut sy_max = f64::NEG_INFINITY;
    for p in &projected {
        for v in [p.a, p.b, p.c] {
            sx_min = sx_min.min(v[0]);
            sx_max = sx_max.max(v[0]);
            sy_min = sy_min.min(v[1]);
            sy_max = sy_max.max(v[1]);
        }
    }
    if !sx_min.is_finite() || !sy_min.is_finite() {
        return Ok(());
    }
    let (w, h) = area.dim_in_pixel();
    let aspect = if h == 0 { 1.0 } else { w as f64 / h as f64 };
    let cx = 0.5 * (sx_min + sx_max);
    let cy = 0.5 * (sy_min + sy_max);
    let scale = view.scale.max(1e-6);
    let mut dx = ((sx_max - sx_min) / scale).max(1e-9);
    let mut dy = ((sy_max - sy_min) / scale).max(1e-9);
    if dx / dy > aspect {
        dy = dx / aspect;
    } else {
        dx = dy * aspect;
    }
    let margin = 0.05;
    dx *= 1.0 + 2.0 * margin;
    dy *= 1.0 + 2.0 * margin;
    let xmin = cx - dx / 2.0;
    let xmax = cx + dx / 2.0;
    let ymin = cy - dy / 2.0;
    let ymax = cy + dy / 2.0;

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
    for t in &projected {
        let face = RGBAColor(t.color.r, t.color.g, t.color.b, 0.85);
        let face_style = ShapeStyle {
            color: face,
            filled: true,
            stroke_width: 0,
        };
        let pa = (t.a[0], t.a[1]);
        let pb = (t.b[0], t.b[1]);
        let pc = (t.c[0], t.c[1]);
        chart
            .draw_series(std::iter::once(Polygon::new(
                vec![pa, pb, pc],
                face_style,
            )))
            .map_err(pl_err)?;
        chart
            .draw_series(LineSeries::new(vec![pa, pb, pc, pa], edge_style))
            .map_err(pl_err)?;
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
        if self.element_type() != ElementType::TRI3 {
            return Err(PyrucastError::Message(format!(
                "viz: SubMesh<{}> not supported yet (only TRI3 is implemented)",
                self.element_type()
            )));
        }
        let tris = submesh_triangles(self)?;
        render_triangles(area, view, &tris)
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
            let mut tris = with(&sm, |s| submesh_triangles(s))??;
            all.append(&mut tris);
        }
        render_triangles(area, view, &all)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Configuration;
    use crate::node::Node;
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
        assert_eq!(bb.min, [0.0, 0.0, 0.0]);
        assert_eq!(bb.max, [2.0, 3.0, 0.0]);
    }

    #[test]
    fn submesh_triangles_skips_non_tri3() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        let tris = submesh_triangles(&sm).unwrap();
        assert!(tris.is_empty());
    }

    #[test]
    fn submesh_triangles_carries_face_color() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(cfg, ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.set_face_color(RgbColor::new(10, 20, 30));
        let tris = submesh_triangles(&sm).unwrap();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0].color, RgbColor::new(10, 20, 30));
    }

    #[test]
    fn pad3_truncates_and_pads() {
        assert_eq!(pad3(&[]), [0.0, 0.0, 0.0]);
        assert_eq!(pad3(&[1.0]), [1.0, 0.0, 0.0]);
        assert_eq!(pad3(&[1.0, 2.0]), [1.0, 2.0, 0.0]);
        assert_eq!(pad3(&[1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert_eq!(pad3(&[1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0]);
    }
}
