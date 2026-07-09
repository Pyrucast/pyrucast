//! Python wrappers for the operations in [`crate::ops`].
//!
//! Free functions (factories and transforms) live here, one module per
//! `src/ops/<family>/` subtree, mirroring the Rust layout so the wrapper of
//! an operation sits at the matching path. Type wrappers stay in
//! `src/py/<type>.rs` (mirroring `src/containers/`); this `ops` tree holds
//! the *verbs*, those the *nouns*.

pub mod assemble;
pub mod behavior;
pub mod build;
pub mod export;
pub mod field;
pub mod internal_forces;
pub mod mesher;
pub mod solver;

use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Consolidate a container — fuse redundant sub-objects « au plus juste ».
///
/// Dispatches on the argument type (Python has a single top-level name
/// for the two themed Rust ops):
///
/// - `Mesh` → `ops::mesher::consolidate`: fuse submeshes of the same
///   element type, drop duplicate cells;
/// - `NodeField` → `ops::field::consolidate_node`: fuse zones with the same
///   component set, dedupe interface nodes after a coherence check.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn consolidate(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(mesh) = obj.extract::<PyRef<PyMesh>>() {
        let result = crate::ops::mesher::consolidate(&mesh.inner)?;
        return Ok(Py::new(py, PyMesh { inner: result })?.into_any());
    }
    if let Ok(field) = obj.extract::<PyRef<PyNodeField>>() {
        let result = crate::ops::field::consolidate_node(&field.inner)?;
        return Ok(Py::new(py, PyNodeField { inner: result })?.into_any());
    }
    Err(PyTypeError::new_err("expected a Mesh or a NodeField"))
}
