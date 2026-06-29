//! Python wrappers for [`crate::ops::export`] — write meshes and fields to
//! external file formats.

use crate::py::element_field::PyElementField;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::path::Path;

/// Write `mesh` to a legacy **VTK** file (`UNSTRUCTURED_GRID`, ASCII) that
/// ParaView reads natively.
///
/// With `field=None` only the geometry is written. Pass a `NodeField` to add
/// it as `POINT_DATA` (one scalar array per component, the nodal value at
/// each point) or an `ElementField` to add it as `CELL_DATA` (one array per
/// component, the per-cell mean of that cell's Gauss values). An element
/// field must come from a space built on **this** mesh, so its cells line up.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, path, field=None))]
pub fn export_vtk(
    mesh: PyRef<PyMesh>,
    path: &str,
    field: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let path = Path::new(path);
    match field {
        None => crate::ops::export::write_vtk_mesh(&mesh.inner, path)?,
        Some(obj) => {
            if let Ok(nf) = obj.extract::<PyRef<PyNodeField>>() {
                crate::ops::export::write_vtk_node_field(&mesh.inner, &nf.inner, path)?;
            } else if let Ok(ef) = obj.extract::<PyRef<PyElementField>>() {
                crate::ops::export::write_vtk_element_field(&mesh.inner, &ef.inner, path)?;
            } else {
                return Err(PyTypeError::new_err(
                    "field must be a NodeField or an ElementField",
                ));
            }
        }
    }
    Ok(())
}
