//! Python wrappers for [`crate::ops::coords`] — the three operators that
//! write back into the coordinates. `in_config` is the method
//! `Coords.in_config`, a context manager.

use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Set node coordinates from `field` (absolute): for every node, the
/// active-set coordinate on axis `a` becomes `field.value(node,
/// components[a])`. `components` lists one component name per spatial axis,
/// in axis order; `None` → `["X", "Y", "Z"][:dim]`. Mutates the field's
/// `Coords` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "set_positions")]
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

/// Lay configuration `target` of the field's `Coords` at configuration
/// `reference` displaced by `field`: `x = X + u`, in absolute — called again,
/// it lands at `X + u` again, nothing accumulates. A node the field does not
/// define stays at its reference position; the active configuration is left as
/// it is. `components` names one displacement component per axis; `None` →
/// `["ux", "uy", "uz"][:dim]`. Cast3M `FORM` with a displacement, without
/// changing the active configuration.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, reference, target, components=None))]
pub fn displace_from(
    field: PyRef<PyNodeField>,
    reference: usize,
    target: usize,
    components: Option<Vec<String>>,
) -> PyResult<()> {
    crate::ops::coords::displace_from(&field.inner, reference, target, components)?;
    Ok(())
}
