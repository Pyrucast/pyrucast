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
//! - [`Bbox3`] is the axis-aligned 3D bounding box, used to centre and scale.
//! - [`Projector`] (in [`camera`]) maps 3D world coordinates to a 2D screen.
//! - [`Drawable`] (in [`drawable`]) is the internal trait every visualizable
//!   object implements; backends iterate over it the same way for PNG, SVG
//!   and the live window.
//!
//! # Example
//!
//! ```no_run
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::insert;
//! use pyrucast::viz::View;
//!
//! let cfg = insert(Configuration::new(3).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.0]).unwrap();
//!
//! let mut sm = SubMesh::new(cfg, ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! // Export to a PNG file in iso view (requires feature `viz`).
//! # #[cfg(feature = "viz")]
//! sm.plot(Some(View::iso()), Some(std::path::Path::new("triangle.png"))).unwrap();
//! ```

pub mod axes;
pub mod camera;
pub mod drawable;
pub mod field_color;
pub mod mesh_draw;
pub mod overlay;
#[cfg(feature = "viz-interactive")]
pub mod window;

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
        match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
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
) -> Result<()> {
    let view = view.unwrap_or_default();
    match save {
        Some(path) => render_to_file(object, view, path),
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive(object, view)
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

/// Render a [`crate::containers::mesh::Mesh`] coloured by a `SubNodeField` component.
/// File export draws the supplied `component`; the interactive window
/// adds a clickable button (top-centre) and a `Tab` keyboard shortcut
/// to cycle through every component.
pub(crate) fn render_mesh_with_field(
    mesh: &crate::containers::mesh::Mesh,
    field: &crate::containers::node_field::SubNodeField,
    component: Option<&str>,
    scale: ColorScale,
    view: Option<View>,
    save: Option<&Path>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    let resolved = field_color::resolve_component(field, component)?;
    match save {
        Some(path) => {
            let drawable = field_color::MeshFieldView {
                mesh,
                field,
                component: resolved,
                scale,
            };
            render_to_file(&drawable, view, path)
        }
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive_mesh_field(mesh, field, resolved, scale, view)
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

/// Render a [`crate::containers::mesh::SubMesh`] coloured by a `SubNodeField`
/// component. Same semantics as [`render_mesh_with_field`].
pub(crate) fn render_submesh_with_field(
    submesh: &crate::containers::mesh::SubMesh,
    field: &crate::containers::node_field::SubNodeField,
    component: Option<&str>,
    scale: ColorScale,
    view: Option<View>,
    save: Option<&Path>,
) -> Result<()> {
    let view = view.unwrap_or_default();
    let resolved = field_color::resolve_component(field, component)?;
    match save {
        Some(path) => {
            let drawable = field_color::SubMeshFieldView {
                submesh,
                field,
                component: resolved,
                scale,
            };
            render_to_file(&drawable, view, path)
        }
        None => {
            #[cfg(feature = "viz-interactive")]
            {
                window::run_interactive_submesh_field(submesh, field, resolved, scale, view)
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

fn render_to_file<D: Drawable>(object: &D, view: View, path: &Path) -> Result<()> {
    use plotters::prelude::*;

    let fmt = SaveFormat::from_path(path)?;
    let w = DEFAULT_WIDTH;
    let h = DEFAULT_HEIGHT;

    let map_err = |e: Box<dyn std::error::Error + Send + Sync>| {
        PyrucastError::Message(format!("plotters: {e}"))
    };

    match fmt {
        SaveFormat::Png => {
            let backend = BitMapBackend::new(path, (w, h));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).map_err(|e| map_err(Box::new(e)))?;
            object.draw_on(&area, &view).map_err(|e| match e {
                PyrucastError::Message(m) => PyrucastError::Message(format!("plotters: {m}")),
                other => other,
            })?;
            if view.show_axes {
                axes::draw_gizmo(&area, &view)?;
            }
            area.present().map_err(|e| map_err(Box::new(e)))?;
        }
        SaveFormat::Svg => {
            let backend = SVGBackend::new(path, (w, h));
            let area = backend.into_drawing_area();
            area.fill(&WHITE).map_err(|e| map_err(Box::new(e)))?;
            object.draw_on(&area, &view).map_err(|e| match e {
                PyrucastError::Message(m) => PyrucastError::Message(format!("plotters: {m}")),
                other => other,
            })?;
            if view.show_axes {
                axes::draw_gizmo(&area, &view)?;
            }
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
