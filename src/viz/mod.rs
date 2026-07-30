//! Visualization — PNG / SVG export and (optionally) interactive window.
//!
//! Gated behind the `viz` feature: pure-CPU rendering via `plotters` so the
//! whole stack works on Linux and Windows without GPU drivers. The optional
//! `viz-interactive` feature adds a `winit` + `softbuffer` window with
//! mouse-driven rotation and zoom.
//!
//! User-facing entry points live on the visualized objects themselves
//! (e.g. [`crate::containers::mesh::SubMesh::plot`]). Internals:
//!
//! - [`View`] is a small point-of-view descriptor (yaw, pitch, scale, target).
//! - `Bbox3` is the axis-aligned 3D bounding box, used to centre and scale.
//! - `Projector` (in `camera`) maps 3D world coordinates to a 2D screen.
//! - [`Drawable`] (in [`drawable`]) is the internal trait every visualizable
//!   object implements; backends iterate over it the same way for PNG, SVG
//!   and the live window.
//!
//! # Example
//!
//! ```no_run
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::insert;
//! use pyrucast::viz::View;
//!
//! let coords = insert(Coords::new(3).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
//! let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
//!
//! let mut sm = SubMesh::new(coords, ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! // Export to a PNG file in iso view (requires feature `viz`).
//! # #[cfg(feature = "viz")]
//! sm.plot(Some(View::iso()), Some(std::path::Path::new("triangle.png"))).unwrap();
//! ```

pub mod axes;
pub mod camera;
pub mod curve;
pub mod drawable;
pub mod field_color;
pub mod mesh_draw;
pub mod overlay;
pub mod subdivide;
#[cfg(feature = "viz-interactive")]
pub mod window;

use crate::containers::field::Field as _;
use crate::error::{PyrucastError, Result};
use crate::viz::drawable::Drawable;
use std::path::Path;

pub use field_color::Colormap;

// ─── Point of view ──────────────────────────────────────────────────────────

/// User-facing camera descriptor.
///
/// Orthographic camera placed on a sphere around `target`:
/// - `yaw` rotates around the world Z axis (azimuth, in degrees),
/// - `pitch` is the elevation above the world XY plane (degrees),
/// - `scale` shrinks (>1) or enlarges (<1) the visible window; `1.0` fits
///   the bounding box exactly.
/// - `target` is the point the camera looks at; `None` means "centre of
///   the visualized object's bounding box".
///
/// Conventions:
/// - `yaw = 0, pitch = 0` → camera on +X, looking at origin, +Z up.
/// - `yaw = 90, pitch = 0` → camera on +Y.
/// - `yaw = 0, pitch = 90` → camera above, top-down view.
#[derive(Debug, Clone, Copy)]
pub struct View {
    pub yaw: f64,
    pub pitch: f64,
    pub scale: f64,
    pub target: Option<crate::containers::mesh::Point3>,
    /// Show the orientation gizmo (small red/green/blue axes triad in the
    /// bottom-left corner) on top of the rendered object.
    pub show_axes: bool,
}

impl View {
    /// Front view: yaw = 0, pitch = 0.
    pub fn front() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            scale: 1.0,
            target: None,
            show_axes: true,
        }
    }

    /// Top view: looking down the −Z axis.
    pub fn top() -> Self {
        Self {
            yaw: 0.0,
            pitch: 90.0,
            scale: 1.0,
            target: None,
            show_axes: true,
        }
    }

    /// Side view: yaw = 90, pitch = 0.
    pub fn side() -> Self {
        Self {
            yaw: 90.0,
            pitch: 0.0,
            scale: 1.0,
            target: None,
            show_axes: true,
        }
    }

    /// Isometric view: yaw = 45, pitch ≈ 35.264.
    pub fn iso() -> Self {
        Self {
            yaw: 45.0,
            pitch: 35.264_389_682_754_654,
            scale: 1.0,
            target: None,
            show_axes: true,
        }
    }
}

impl Default for View {
    fn default() -> Self {
        Self::iso()
    }
}

// ─── Mesh rendering style ─────────────────────────────────────────────────────

/// How a **geometry-only** mesh plot is drawn (no field colouring).
///
/// - [`Surface`](MeshStyle::Surface) — the default — fills the outer skin
///   opaquely (boundary faces of volume cells), hiding what is behind it.
/// - [`Wireframe`](MeshStyle::Wireframe) draws **every** element edge as a
///   line, interior edges of volume cells included, with no fill — a
///   see-through "fil de fer".
///
/// The style only applies to plain mesh plots; it is meaningless when a
/// field colours the cells (the field always paints faces).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MeshStyle {
    /// Opaque outer surface (default).
    #[default]
    Surface,
    /// All edges as a wireframe.
    Wireframe,
}

// ─── Colour scale ─────────────────────────────────────────────────────────────

/// User override for the colour-scale bounds of a field plot.
///
/// A `None` bound means "use the data's own min (resp. max) for that
/// end", so the two bounds can be pinned independently — pass only
/// `vmax` to clamp the top of the scale and let the bottom follow the
/// data, matching the `vmin` / `vmax` convention of common plotting
/// libraries. `cmap` chooses the colour gradient (default
/// [`Colormap::Viridis`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorScale {
    pub cmap: Colormap,
    pub vmin: Option<f64>,
    pub vmax: Option<f64>,
}

impl ColorScale {
    /// Resolve the effective `(vmin, vmax)` against the data-derived range.
    pub fn resolve(&self, data_min: f64, data_max: f64) -> (f64, f64) {
        (self.vmin.unwrap_or(data_min), self.vmax.unwrap_or(data_max))
    }
}

// ─── Output format dispatch ─────────────────────────────────────────────────

/// Picture format inferred from the output path's extension.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SaveFormat {
    Png,
    Svg,
}

impl SaveFormat {
    pub(crate) fn from_path(path: &Path) -> Result<Self> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
        {
            Some(ref s) if s == "png" => Ok(SaveFormat::Png),
            Some(ref s) if s == "svg" => Ok(SaveFormat::Svg),
            Some(other) => Err(PyrucastError::Message(format!(
                "unsupported viz extension: \"{}\" (expected .png or .svg)",
                other
            ))),
            None => Err(PyrucastError::Message(
                "viz output path has no extension (expected .png or .svg)".into(),
            )),
        }
    }
}

// ─── Default image size for file export ─────────────────────────────────────

pub(crate) const DEFAULT_WIDTH: u32 = 800;
pub(crate) const DEFAULT_HEIGHT: u32 = 600;

// ─── Render dispatch ────────────────────────────────────────────────────────

/// Render a [`Drawable`] either to a file (PNG / SVG) or, if `save` is
/// `None`, to an interactive window. Headless export works without
/// `viz-interactive`; the interactive path requires it.
pub(crate) fn render<D: Drawable>(
    object: &D,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    match save {
        Some(path) => render_to_file(object, view, path, title),
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive(object, view, title)
            }
            #[cfg(not(feature = "viz-interactive"))]
            {
                let _ = title;
                Err(PyrucastError::Message(
                    "interactive viz disabled — recompile with --features viz-interactive \
                     or pass an output path to save a PNG/SVG"
                        .into(),
                ))
            }
        }
    }
}

/// Render a [`crate::containers::mesh::SubMesh`] in the chosen
/// [`MeshStyle`] (geometry only, no field). `Surface` draws the opaque
/// skin; `Wireframe` draws every edge.
pub(crate) fn render_submesh_styled(
    sm: &crate::containers::mesh::SubMesh,
    view: Option<View>,
    save: Option<&Path>,
    style: MeshStyle,
    title: Option<&str>,
) -> Result<()> {
    match style {
        MeshStyle::Surface => render(sm, view, save, title),
        MeshStyle::Wireframe => render(&mesh_draw::SubMeshWire(sm), view, save, title),
    }
}

/// Render a [`crate::containers::mesh::Mesh`] in the chosen [`MeshStyle`]
/// (geometry only, no field).
pub(crate) fn render_mesh_styled(
    mesh: &crate::containers::mesh::Mesh,
    view: Option<View>,
    save: Option<&Path>,
    style: MeshStyle,
    title: Option<&str>,
) -> Result<()> {
    match style {
        MeshStyle::Surface => render(mesh, view, save, title),
        MeshStyle::Wireframe => render(&mesh_draw::MeshWire(mesh), view, save, title),
    }
}

/// A field to colour a plot by — the **uniform** entry point of the viz
/// layer: node fields and element fields are accepted interchangeably.
pub enum FieldArg<'a> {
    Node(&'a crate::containers::node_field::NodeField),
    Element(&'a crate::containers::element_field::ElementField),
}

impl FieldArg<'_> {
    /// Take the zero-copy views (read guards on every zone).
    fn data(&self) -> Result<field_color::FieldData> {
        Ok(match self {
            FieldArg::Node(f) => field_color::FieldData::Node(f.view()?),
            FieldArg::Element(f) => field_color::FieldData::Element(f.view()?),
        })
    }
}

/// Render a [`crate::containers::mesh::Mesh`] coloured by a field
/// component (node or element field, see [`FieldArg`]).
/// File export draws the supplied `component`; the interactive window
/// adds a clickable button (top-centre) and a `Tab` keyboard shortcut
/// to cycle through every component (union of the zones').
#[allow(clippy::too_many_arguments)] // see `Mesh::plot_with_field`
pub(crate) fn render_mesh_with_field(
    mesh: &crate::containers::mesh::Mesh,
    field: FieldArg<'_>,
    component: Option<&str>,
    scale: ColorScale,
    smooth: usize,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    let data = field.data()?;
    let resolved = field_color::resolve_component(&data, component)?;
    match save {
        Some(path) => {
            let drawable = field_color::MeshFieldView {
                mesh,
                field: &data,
                component: resolved,
                scale,
                smooth,
            };
            render_to_file(&drawable, view, path, title)
        }
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive_mesh_field(
                    mesh, &data, resolved, scale, smooth, view, title,
                )
            }
            #[cfg(not(feature = "viz-interactive"))]
            {
                Err(PyrucastError::Message(
                    "interactive viz disabled — recompile with --features viz-interactive \
                     or pass an output path to save a PNG/SVG"
                        .into(),
                ))
            }
        }
    }
}

/// Render a [`crate::containers::mesh::SubMesh`] (by handle, so
/// element-field zones can be matched by identity) coloured by a field
/// component. Same semantics as `render_mesh_with_field`.
#[allow(clippy::too_many_arguments)] // see `Mesh::plot_with_field`
pub fn render_submesh_with_field(
    submesh: &crate::store::Handle<crate::containers::mesh::SubMesh>,
    field: FieldArg<'_>,
    component: Option<&str>,
    scale: ColorScale,
    smooth: usize,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    let data = field.data()?;
    let resolved = field_color::resolve_component(&data, component)?;
    match save {
        Some(path) => {
            let drawable = field_color::SubMeshFieldView {
                submesh,
                field: &data,
                component: resolved,
                scale,
                smooth,
            };
            render_to_file(&drawable, view, path, title)
        }
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive_submesh_field(
                    submesh, &data, resolved, scale, smooth, view, title,
                )
            }
            #[cfg(not(feature = "viz-interactive"))]
            {
                Err(PyrucastError::Message(
                    "interactive viz disabled — recompile with --features viz-interactive \
                     or pass an output path to save a PNG/SVG"
                        .into(),
                ))
            }
        }
    }
}

/// The `(point, value)` cloud of a [`NodeField`] for one resolved
/// `component`, ready for a [`field_color::NodeFieldPointsView`]. Nodes the
/// component does not cover are skipped.
pub(crate) fn node_field_points(
    field: &crate::containers::node_field::NodeField,
    component: &str,
) -> Result<Vec<(crate::containers::mesh::Point3, f64)>> {
    use crate::viz::mesh_draw::pad3;
    let view = field.view()?;
    let coords = field.coords()?;
    let c = crate::store::read(&coords)?;
    let mut points = Vec::new();
    for nid in field.node_ids()? {
        if let Some(val) = view.value_opt(nid, component) {
            points.push((pad3(c.coord(nid)?), val));
        }
    }
    Ok(points)
}

/// Bounding box of a [`NodeField`]'s support nodes (component-independent),
/// for centring a point-cloud view (interactive only).
#[cfg(feature = "viz-interactive")]
pub(crate) fn node_field_bbox(
    field: &crate::containers::node_field::NodeField,
) -> Result<crate::viz::camera::Bbox3> {
    use crate::viz::mesh_draw::pad3;
    let coords = field.coords()?;
    let c = crate::store::read(&coords)?;
    let mut bb = crate::viz::camera::Bbox3::empty();
    for nid in field.node_ids()? {
        bb.extend(pad3(c.coord(nid)?));
    }
    Ok(bb)
}

/// Render a [`crate::containers::node_field::NodeField`] alone, as a
/// coloured point cloud over its support nodes (a POI1 support carries
/// no connectivity — plot a mesh with `field=` for surfaces).
pub(crate) fn render_node_field_points(
    field: &crate::containers::node_field::NodeField,
    component: Option<&str>,
    scale: ColorScale,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    let data = field_color::FieldData::Node(field.view()?);
    let resolved = field_color::resolve_component(&data, component)?.to_string();
    drop(data);
    let points = node_field_points(field, &resolved)?;
    let drawable = field_color::NodeFieldPointsView {
        points,
        component: &resolved,
        scale,
    };
    render(&drawable, view, save, title)
}

// ─── Evolution: per-frame field source ──────────────────────────────────────

/// One tabulated frame of an evolution of fields — a whole node or element
/// field. The viz layer renders the selected frame and (interactively) lets a
/// slider pick among them.
pub enum FrameField {
    Node(crate::containers::node_field::NodeField),
    Element(crate::containers::element_field::ElementField),
}

/// Render an evolution of fields. On `save`, a **single** frame is written
/// (`frame`, default = last). Interactive (`save=None`) opens the window with
/// a frame slider (← / → also step) on top of the usual field controls.
///
/// `mesh` supplies the surface geometry; when `None`, node frames are drawn as
/// a coloured point cloud (element frames require a `mesh`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_evolution_field(
    mesh: Option<&crate::containers::mesh::Mesh>,
    frames: &[FrameField],
    abscissas: &[f64],
    abscissa_label: &str,
    component: Option<&str>,
    scale: ColorScale,
    smooth: usize,
    frame: Option<usize>,
    view: Option<View>,
    save: Option<&Path>,
) -> Result<()> {
    if frames.is_empty() {
        return Err(PyrucastError::Message(
            "evolution plot: no tabulated frame".into(),
        ));
    }
    match save {
        Some(path) => {
            let k = frame.unwrap_or(frames.len() - 1);
            if k >= frames.len() {
                return Err(PyrucastError::Message(format!(
                    "evolution plot: frame {} out of range (have {})",
                    k,
                    frames.len()
                )));
            }
            render_one_frame(
                mesh,
                &frames[k],
                component,
                scale,
                smooth,
                view,
                Some(path),
                None,
            )
        }
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive_evolution(
                    mesh,
                    frames,
                    abscissas,
                    abscissa_label,
                    component,
                    scale,
                    smooth,
                    view.unwrap_or_default(),
                    None,
                )
            }
            #[cfg(not(feature = "viz-interactive"))]
            {
                let _ = (abscissas, abscissa_label, smooth);
                Err(PyrucastError::Message(
                    "interactive viz disabled — recompile with --features viz-interactive \
                     or pass an output path to save a PNG/SVG"
                        .into(),
                ))
            }
        }
    }
}

/// Render a single evolution frame onto a mesh (or as a point cloud when
/// `mesh` is `None` and the frame is a node field).
#[allow(clippy::too_many_arguments)]
fn render_one_frame(
    mesh: Option<&crate::containers::mesh::Mesh>,
    frame: &FrameField,
    component: Option<&str>,
    scale: ColorScale,
    smooth: usize,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    match (mesh, frame) {
        (Some(m), FrameField::Node(f)) => render_mesh_with_field(
            m,
            FieldArg::Node(f),
            component,
            scale,
            smooth,
            view,
            save,
            title,
        ),
        (Some(m), FrameField::Element(f)) => render_mesh_with_field(
            m,
            FieldArg::Element(f),
            component,
            scale,
            smooth,
            view,
            save,
            title,
        ),
        (None, FrameField::Node(f)) => {
            render_node_field_points(f, component, scale, view, save, title)
        }
        (None, FrameField::Element(_)) => Err(PyrucastError::Message(
            "evolution plot: element-field frames require a mesh".into(),
        )),
    }
}

/// Render an X-Y curve: one or more labelled `(x, y)` line series on a
/// Cartesian chart. The 3-D gizmo is disabled (a curve has no camera);
/// `save=None` opens the interactive window (where the chart is static).
pub(crate) fn render_curve(
    series: Vec<(String, Vec<(f64, f64)>)>,
    x_label: &str,
    y_label: &str,
    title: &str,
    view: Option<View>,
    save: Option<&Path>,
) -> Result<()> {
    let plot = curve::CurvePlot {
        series,
        x_label: x_label.to_string(),
        y_label: y_label.to_string(),
        title: title.to_string(),
    };
    let mut view = view.unwrap_or_default();
    view.show_axes = false;
    // The curve's own caption already sits at the top of the chart, so we do
    // not repeat it as a bottom figure title.
    render(&plot, Some(view), save, None)
}

/// Draw `object` (and its gizmo) onto `area`, mapping plotters errors.
fn draw_object<DB, D>(
    area: &plotters::drawing::DrawingArea<DB, plotters::coord::Shift>,
    object: &D,
    view: &View,
) -> Result<()>
where
    DB: plotters::prelude::DrawingBackend,
    DB::ErrorType: 'static,
    D: Drawable,
{
    object.draw_on(area, view).map_err(|e| match e {
        PyrucastError::Message(m) => PyrucastError::Message(format!("plotters: {m}")),
        other => other,
    })?;
    if view.show_axes {
        axes::draw_gizmo(area, view)?;
    }
    Ok(())
}

/// Fill `area` white, then draw `object` — reserving a bottom band for the
/// figure `title` when one is given (drawn centred in the band).
fn draw_root<DB, D>(
    area: &plotters::drawing::DrawingArea<DB, plotters::coord::Shift>,
    object: &D,
    view: &View,
    title: Option<&str>,
) -> Result<()>
where
    DB: plotters::prelude::DrawingBackend,
    DB::ErrorType: 'static,
    D: Drawable,
{
    use plotters::prelude::*;
    let map_err = |e: Box<dyn std::error::Error + Send + Sync>| {
        PyrucastError::Message(format!("plotters: {e}"))
    };
    area.fill(&WHITE).map_err(|e| map_err(Box::new(e)))?;
    match title {
        Some(t) => {
            let (_, h) = area.dim_in_pixel();
            let split_at = h.saturating_sub(overlay::FIGURE_TITLE_BAND) as i32;
            let (main, footer) = area.split_vertically(split_at);
            draw_object(&main, object, view)?;
            overlay::draw_figure_title(&footer, t)?;
        }
        None => draw_object(area, object, view)?,
    }
    Ok(())
}

fn render_to_file<D: Drawable>(
    object: &D,
    view: View,
    path: &Path,
    title: Option<&str>,
) -> Result<()> {
    use plotters::prelude::*;

    let fmt = SaveFormat::from_path(path)?;
    let w = DEFAULT_WIDTH;
    let h = DEFAULT_HEIGHT;

    let map_err = |e: Box<dyn std::error::Error + Send + Sync>| {
        PyrucastError::Message(format!("plotters: {e}"))
    };

    // A blank title is treated as "no title" (full-height plot).
    let title = title.filter(|t| !t.trim().is_empty());

    match fmt {
        SaveFormat::Png => {
            let backend = BitMapBackend::new(path, (w, h));
            let area = backend.into_drawing_area();
            draw_root(&area, object, &view, title)?;
            area.present().map_err(|e| map_err(Box::new(e)))?;
        }
        SaveFormat::Svg => {
            let backend = SVGBackend::new(path, (w, h));
            let area = backend.into_drawing_area();
            draw_root(&area, object, &view, title)?;
            area.present().map_err(|e| map_err(Box::new(e)))?;
        }
    }
    Ok(())
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_documented_angles() {
        assert_eq!(View::front().yaw, 0.0);
        assert_eq!(View::front().pitch, 0.0);
        assert_eq!(View::top().pitch, 90.0);
        assert_eq!(View::side().yaw, 90.0);
        let iso = View::iso();
        assert_eq!(iso.yaw, 45.0);
        assert!((iso.pitch - 35.264_389_682_754_654).abs() < 1e-12);
    }

    #[test]
    fn default_is_iso() {
        let d = View::default();
        let iso = View::iso();
        assert_eq!(d.yaw, iso.yaw);
        assert_eq!(d.pitch, iso.pitch);
        assert_eq!(d.scale, iso.scale);
    }

    #[test]
    fn save_format_from_path() {
        use std::path::PathBuf;
        assert!(matches!(
            SaveFormat::from_path(&PathBuf::from("a.png")).unwrap(),
            SaveFormat::Png
        ));
        assert!(matches!(
            SaveFormat::from_path(&PathBuf::from("a.PNG")).unwrap(),
            SaveFormat::Png
        ));
        assert!(matches!(
            SaveFormat::from_path(&PathBuf::from("a.svg")).unwrap(),
            SaveFormat::Svg
        ));
        assert!(SaveFormat::from_path(&PathBuf::from("a.jpg")).is_err());
        assert!(SaveFormat::from_path(&PathBuf::from("noext")).is_err());
    }
}
