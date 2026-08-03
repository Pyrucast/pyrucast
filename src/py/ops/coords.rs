//! Python wrappers for [`crate::ops::coords`] — the two operators that
//! write back into the coordinate store.

use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Set node coordinates from `field` (absolute): for every node, the
/// active-set coordinate on axis `a` becomes `field.value(node,
/// components[a])`. `components` lists one component name per spatial axis,
/// in axis order; `None` → `["X", "Y", "Z"][:dim]`. Mutates the field's
/// `Coords` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "set_coordinates")]
#[pyo3(signature = (field, components=None))]
pub fn set(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    crate::ops::coords::set(&field.inner, components)?;
    Ok(())
}

/// Displace nodes by `field` (incremental): `coord[a] += field.value(node,
/// components[a])` on the active coordinate set. `components` lists one
/// displacement-component name per spatial axis, in axis order; `None` →
/// `["ux", "uy", "uz"][:dim]`. Mutates the field's `Coords` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn displace(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    crate::ops::coords::displace(&field.inner, components)?;
    Ok(())
}
