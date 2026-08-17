//! Visualization — PNG / SVG / SVGZ export and (optionally) interactive window.
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
//! - `shrink_svg` strips from the vector output what the SVG backend repeats
//!   on every tag; `.svgz` gzips that same markup, for figures kept by the
//!   hundred rather than published.
//!
//! # Example
//!
//! ```no_run
//! use pyrucast::coords::Coords;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::atoms::Node;
//! use pyrucast::handle::Handle;
//! use pyrucast::viz::View;
//!
//! let coords = Handle::new(Coords::new(3).unwrap());
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
pub mod revolve;
pub mod subdivide;
#[cfg(feature = "viz-interactive")]
pub mod window;

use crate::containers::field::Field as _;
use crate::error::{PyrucastError, Result};
use crate::viz::drawable::Drawable;
use std::path::Path;

pub use field_color::Colormap;
pub use revolve::Revolve;

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
    pub target: Option<crate::atoms::Point3>,
    /// Show the orientation gizmo (small red/green/blue axes triad in the
    /// bottom-left corner) on top of the rendered object.
    pub show_axes: bool,
    /// Sweep an [axisymmetric](crate::coords::Coords::axisymmetric)
    /// meridian plot into the body of revolution it describes (see
    /// [`Revolve`]). `None` — the default — keeps the flat `(r, z)` section.
    /// Only accepted on axisymmetric geometry; the interactive window toggles
    /// it with the top-left button or the `R` key.
    pub revolve: Option<Revolve>,
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
            revolve: None,
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
            revolve: None,
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
            revolve: None,
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
            revolve: None,
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
    /// The same SVG, gzipped. Same drawing, same markup, about a tenth of the
    /// bytes on disk — for piling up figures rather than for publishing them.
    Svgz,
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
            // Only `.svgz` — `figure.svg.gz` has `gz` for an extension and
            // falls through to the error, which is the honest answer.
            Some(ref s) if s == "svgz" => Ok(SaveFormat::Svgz),
            Some(other) => Err(PyrucastError::Message(format!(
                "unsupported viz extension: \"{}\" (expected .png, .svg or .svgz)",
                other
            ))),
            None => Err(PyrucastError::Message(
                "viz output path has no extension (expected .png, .svg or .svgz)".into(),
            )),
        }
    }
}

// ─── Default image size for file export ─────────────────────────────────────

pub(crate) const DEFAULT_WIDTH: u32 = 800;
pub(crate) const DEFAULT_HEIGHT: u32 = 600;

// ─── Render dispatch ────────────────────────────────────────────────────────

/// Reject a [`Revolve`] asked of a geometry that is not axisymmetric — the
/// sweep reads the abscissa as a radius, which is only a radius in the
/// meridian frame.
pub(crate) fn check_revolve<D: Drawable>(object: &D, view: &View) -> Result<()> {
    if view.revolve.is_some() && !object.is_axisymmetric() {
        return Err(PyrucastError::Message(
            "revolve: the plotted geometry is not axisymmetric — build it on \
             Coords::axisymmetric() (Python: Coords.axisymmetric()), whose \
             abscissa is the radius the sweep turns around"
                .into(),
        ));
    }
    Ok(())
}

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
    check_revolve(object, &view)?;
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
    check_revolve(mesh, &view)?;
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
    submesh: &crate::handle::Handle<crate::containers::mesh::SubMesh>,
    field: FieldArg<'_>,
    component: Option<&str>,
    scale: ColorScale,
    smooth: usize,
    view: Option<View>,
    save: Option<&Path>,
    title: Option<&str>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    check_revolve(&*submesh.read(), &view)?;
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
) -> Result<Vec<(crate::atoms::Point3, f64)>> {
    use crate::viz::mesh_draw::pad3;
    let view = field.view()?;
    let coords = field.coords()?;
    let c = coords.read();
    let mut points = Vec::new();
    for nid in field.node_ids()? {
        if let Some(val) = view.value_opt(nid, component) {
            points.push((pad3(c.position(nid)?), val));
        }
    }
    Ok(points)
}

/// Whether a [`NodeField`] lives on axisymmetric coordinates — the point cloud
/// it draws alone can then be swept like any other axisymmetric plot.
pub(crate) fn node_field_is_axisymmetric(field: &crate::containers::node_field::NodeField) -> bool {
    field
        .coords()
        .map(|c| c.read().is_axisymmetric())
        .unwrap_or(false)
}

/// Bounding box of a [`NodeField`]'s support nodes (component-independent),
/// for centring a point-cloud view (interactive only).
#[cfg(feature = "viz-interactive")]
pub(crate) fn node_field_bbox(
    field: &crate::containers::node_field::NodeField,
) -> Result<crate::viz::camera::Bbox3> {
    use crate::viz::mesh_draw::pad3;
    let coords = field.coords()?;
    let c = coords.read();
    let mut bb = crate::viz::camera::Bbox3::empty();
    for nid in field.node_ids()? {
        bb.extend(pad3(c.position(nid)?));
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
        axisymmetric: node_field_is_axisymmetric(field),
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
    // A curve is a flat chart: no camera, hence no body to sweep either.
    view.revolve = None;
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
        // One drawing for both: `.svgz` is the `.svg` this would have written,
        // run through gzip. Rendered to a string rather than straight to the
        // file so the markup can go through `shrink_svg` on the way out.
        SaveFormat::Svg | SaveFormat::Svgz => {
            let mut markup = String::new();
            {
                let backend = SVGBackend::with_string(&mut markup, (w, h));
                let area = backend.into_drawing_area();
                draw_root(&area, object, &view, title)?;
                area.present().map_err(|e| map_err(Box::new(e)))?;
            }
            let markup = shrink_svg(&markup);
            if matches!(fmt, SaveFormat::Svgz) {
                use std::io::Write as _;
                let file = std::fs::File::create(path)?;
                let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
                gz.write_all(markup.as_bytes())?;
                gz.finish()?;
            } else {
                std::fs::write(path, markup)?;
            }
        }
    }
    Ok(())
}

/// Strip the style attributes the SVG backend repeats on every single tag it
/// writes, which on a mesh figure outweigh the geometry three to one.
///
/// Every rewrite here leans on the format being inherited rather than
/// restated, so the picture comes out pixel for pixel the same:
///
/// - `opacity="1"` is the SVG default and says nothing;
/// - `fill="none"` only ever qualifies a polyline — every other shape the
///   backend emits carries its own `fill` — so it can sit on the root element
///   and reach them all;
/// - whichever `stroke-width` the figure uses most — 2 for a bare mesh, 1 once
///   a field paints the faces and the edges thin out — moves to the root too,
///   the other widths keeping their own value.
///
/// That last one needs every stroked tag to state its width, and the backend
/// leaves the attribute out wherever the default of 1 already applies. Those
/// tags are given it back first, which is a no-op on the picture and makes
/// the hoist safe.
fn shrink_svg(markup: &str) -> String {
    let out = markup.replace(" opacity=\"1\"", "");
    let out = pin_implicit_stroke_widths(&out.replace("<polyline fill=\"none\" ", "<polyline "));

    let prevailing = ["0", "1", "2"]
        .into_iter()
        .max_by_key(|w| out.matches(&format!(" stroke-width=\"{w}\"")).count())
        .expect("the list is not empty");
    let out = out.replace(&format!(" stroke-width=\"{prevailing}\""), "");
    out.replacen(
        "<svg ",
        &format!("<svg fill=\"none\" stroke-width=\"{prevailing}\" "),
        1,
    )
}

/// Spell out the stroke width of tags that stroke without one, which the SVG
/// default puts at 1. Restating a default changes nothing on screen; it only
/// keeps such a tag from picking up a hoisted width meant for its neighbours.
fn pin_implicit_stroke_widths(markup: &str) -> String {
    let mut out = String::with_capacity(markup.len());
    for piece in markup.split_inclusive('>') {
        match piece.rfind(if piece.ends_with("/>") { "/>" } else { ">" }) {
            Some(cut) if piece.contains("stroke=\"#") && !piece.contains("stroke-width=") => {
                out.push_str(&piece[..cut]);
                out.push_str(" stroke-width=\"1\"");
                out.push_str(&piece[cut..]);
            }
            _ => out.push_str(piece),
        }
    }
    out
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
        assert!(matches!(
            SaveFormat::from_path(&PathBuf::from("a.svgz")).unwrap(),
            SaveFormat::Svgz
        ));
        assert!(SaveFormat::from_path(&PathBuf::from("a.jpg")).is_err());
        assert!(SaveFormat::from_path(&PathBuf::from("noext")).is_err());
        // `a.svg.gz` names `gz` as its extension, which is nothing we write.
        assert!(SaveFormat::from_path(&PathBuf::from("a.svg.gz")).is_err());
    }

    #[test]
    fn shrink_svg_hoists_what_every_tag_repeats() {
        let markup = concat!(
            r#"<svg width="8" height="8" xmlns="http://www.w3.org/2000/svg">"#,
            r##"<polyline fill="none" opacity="1" stroke="#000000" stroke-width="2" points="0,0 1,1 "/>"##,
            r##"<polyline fill="none" opacity="1" stroke="#000000" stroke-width="2" points="1,1 2,0 "/>"##,
            r##"<polyline fill="none" opacity="1" stroke="#FF0000" stroke-width="1" points="0,2 2,2 "/>"##,
            r##"<polygon opacity="1" fill="#B4C8E6" points="0,0 1,0 1,1 "/>"##,
            "</svg>",
        );
        let out = super::shrink_svg(markup);
        assert!(!out.contains("opacity=\"1\""), "opacity is the SVG default");
        assert!(
            out.contains(r#"<svg fill="none" stroke-width="2""#),
            "fill and the prevailing width move onto the root"
        );
        assert!(
            out.contains(r##"<polyline stroke="#000000" points="0,0 1,1 "/>"##),
            "an edge at the prevailing width states neither"
        );
        // The odd one out keeps saying how thick it is.
        assert!(out.contains(r##"<polyline stroke="#FF0000" stroke-width="1""##));
        // A shape that paints itself keeps saying so, or it would inherit
        // `none` from the root and vanish.
        assert!(out.contains(r##"<polygon fill="#B4C8E6""##));
    }

    #[test]
    fn shrink_svg_pins_an_implicit_width_before_hoisting() {
        // This rectangle strokes without saying how thick: it rides the SVG
        // default of 1, and would thicken under the mesh's hoisted 2.
        let markup = concat!(
            r#"<svg width="8" height="8">"#,
            r##"<rect fill="none" stroke="#6E6E6E" x="0" y="0" width="4" height="4"/>"##,
            r##"<polyline stroke="#000000" stroke-width="2" points="0,0 1,1 "/>"##,
            r##"<polyline stroke="#000000" stroke-width="2" points="1,1 2,2 "/>"##,
            "</svg>",
        );
        let out = super::shrink_svg(markup);
        assert!(
            out.contains(r#"<svg fill="none" stroke-width="2""#),
            "the mesh width still gets hoisted"
        );
        assert!(
            out.contains(
                r##"stroke="#6E6E6E" x="0" y="0" width="4" height="4" stroke-width="1"/>"##
            ),
            "the rectangle is pinned to the default it was riding"
        );
    }
}
