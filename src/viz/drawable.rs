//! Internal trait shared by every visualizable object.
//!
//! Backends (PNG, SVG, interactive window) all go through this trait, so
//! the rendering code is written once. Implementations live next to the
//! data type (e.g. [`mesh_draw`](super::mesh_draw) for `SubMesh`).

use crate::error::{PyrucastError, Result};
use crate::viz::camera::Bbox3;
use crate::viz::View;
use plotters::coord::Shift;
use plotters::prelude::*;

/// Map any plotters error into our [`PyrucastError`].
pub(crate) fn pl_err<E: std::error::Error + Send + Sync + 'static>(
    e: DrawingAreaErrorKind<E>,
) -> PyrucastError {
    PyrucastError::Message(format!("plotters: {e}"))
}

/// A visualizable object.
///
/// Implementors expose the 3D bounding box (used to centre and scale the
/// view) and a `draw_on` method that emits 2D primitives onto a plotters
/// drawing area according to `view`.
pub trait Drawable {
    /// World-space bounding box. Used to default-target the camera and to
    /// dimension the viewport. Returning an empty bbox is legal and leads
    /// to a blank picture.
    fn bbox(&self) -> Result<Bbox3>;

    /// Draw the object onto `area` from the viewpoint described by `view`.
    fn draw_on<DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>, view: &View) -> Result<()>
    where
        DB::ErrorType: 'static;

    /// Whether the object's geometry lives in an
    /// [axisymmetric](crate::coords::Coords::axisymmetric) frame —
    /// the only case where [`View::revolve`](crate::viz::View::revolve) means
    /// something. Objects carrying no geometry (a curve) keep the default.
    fn is_axisymmetric(&self) -> bool {
        false
    }
}
