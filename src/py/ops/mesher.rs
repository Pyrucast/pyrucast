//! Python wrappers for the mesher operations in [`crate::ops::mesher`].
//!
//! Free functions that build or transform a [`PyMesh`]. Kept here —
//! mirroring `src/ops/mesher/` — rather than on the `Mesh` class, per the
//! `py/ops/` convention (operations live with operations).

use crate::containers::mesh::ElementType;
use crate::py::configuration::PyConfiguration;
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build a points (POI1) mesh holding every live node of `config`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn from_live_nodes(config: PyRef<PyConfiguration>) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::from_live_nodes(config.handle.clone())?;
    Ok(PyMesh { inner: mesh })
}

/// Convert a mesh to POI1, submesh by submesh.
///
/// Returns a new mesh with the same number of submeshes; each output
/// submesh is a POI1 submesh holding the de-duplicated nodes of the
/// corresponding input submesh, in order of first appearance.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_poi1(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::to_poi1(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Fuse submeshes of the same element type into one and drop duplicate
/// cells. Returns a new mesh with one submesh per element type, in
/// first-seen order.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn consolidate(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::consolidate(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Build a line of `n_elems` SEG2 elements from node `a` to node `b`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn line_seg2(a: PyRef<PyNode>, b: PyRef<PyNode>, n_elems: usize) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::line_seg2(a.as_node(), b.as_node(), n_elems)?;
    Ok(PyMesh { inner: mesh })
}

/// Build a closed circle of `n_elems` SEG2 elements, centred on `center`
/// in the plane defined by `normal`, with the given `radius`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn circle_seg2(
    center: PyRef<PyNode>,
    normal: Vec<f64>,
    radius: f64,
    n_elems: usize,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::circle_seg2(center.as_node(), &normal, radius, n_elems)?;
    Ok(PyMesh { inner: mesh })
}

/// Sweep two SEG2 line meshes into a QUA4 mesh, building `n_layers` layers
/// of quads between `mesh_a` and `mesh_b`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sweep_qua4(mesh_a: PyRef<PyMesh>, mesh_b: PyRef<PyMesh>, n_layers: usize) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::sweep_qua4(&mesh_a.inner, &mesh_b.inner, n_layers)?;
    Ok(PyMesh { inner: mesh })
}

/// Extrude `mesh` by `n_layers` layers along `direction` (the total
/// displacement vector). SEG2 → QUA4, QUA4 → HEX8.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn extrude(mesh: PyRef<PyMesh>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::extrude(&mesh.inner, &direction, n_layers)?;
    Ok(PyMesh { inner: result })
}

/// Fill the interior of a closed SEG2 `contour` with `element_type` cells
/// (triangulation). `max_edge_length` / `min_angle_deg` optionally refine
/// the result.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, max_edge_length=None, min_angle_deg=None))]
pub fn fill_surface(
    contour: PyRef<PyMesh>,
    element_type: &str,
    max_edge_length: Option<f64>,
    min_angle_deg: Option<f64>,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type).ok_or_else(|| {
        PyValueError::new_err(format!("unknown element type: {element_type}"))
    })?;
    let refinement = if max_edge_length.is_some() || min_angle_deg.is_some() {
        Some(crate::ops::mesher::triangulation::RefinementOptions {
            max_edge_length,
            min_angle_deg,
        })
    } else {
        None
    };
    let mesh = crate::ops::mesher::fill_surface(&contour.inner, et, refinement)?;
    Ok(PyMesh { inner: mesh })
}
