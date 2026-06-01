//! Python wrappers for the field operations in [`crate::ops::field`].
//!
//! Free functions that build or transform a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/field/` — rather than on the `NodeField` class, per
//! the `py/ops/` convention (operations live with operations).

use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use crate::store::{insert, with};
use pyo3::prelude::*;

/// Build a `NodeField` carrying the coordinates of every node of `mesh`.
///
/// One component per requested axis (`"X"`, `"Y"`, `"Z"`). `components=None`
/// requests all the axes the mesh's `Configuration` has (`["X"]` in 1-D,
/// `["X", "Y"]` in 2-D, `["X", "Y", "Z"]` in 3-D). A non-POI1 mesh is
/// converted to POI1 internally (see `to_poi1`); the support is the unique
/// nodes of the mesh, in order of first appearance.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn coordinates(
    mesh: PyRef<PyMesh>,
    components: Option<Vec<String>>,
) -> PyResult<PyNodeField> {
    let field = crate::ops::field::coordinates(&mesh.inner, components)?;
    Ok(PyNodeField {
        handle: insert(field),
    })
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `NodeField` with the same components, supported on the
/// unique nodes of `mesh` (order of first appearance). Nodes of `mesh`
/// absent from `field` are assigned `0.0`. Errors if `mesh` and `field`
/// are attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    let result = with(&field.handle, |nf| crate::ops::field::restrict(nf, &mesh.inner))??;
    Ok(PyNodeField {
        handle: insert(result),
    })
}

/// Merge two node fields over the union of their supports.
///
/// Keeps each field's value where only one is defined, `0.0` where
/// neither is. Errors if the two fields hold different values at the same
/// `(node, component)` pair, or are attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge(a: PyRef<PyNodeField>, b: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    // The store mutex is per-type and non-reentrant: clone `b` out before
    // locking `a` rather than nesting two `with::<NodeField>` calls.
    let fb = with(&b.handle, |f| f.clone())?;
    let result = with(&a.handle, |fa| crate::ops::field::merge(fa, &fb))??;
    Ok(PyNodeField {
        handle: insert(result),
    })
}
