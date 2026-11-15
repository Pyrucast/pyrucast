//! Python wrappers for [`crate::ops::coords`] — the three operators that
//! write back into the coordinates. `in_config` is the method
//! `Coords.in_config`, a context manager.

use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Set the coordinates of the nodes of `field` to its values, in absolute.
///
/// For every node of the field, the coordinate on axis `a` of the active
/// configuration becomes the value of `components[a]`. The `Coords` of the
/// field is changed in place.
///
/// # Parameters
///
/// - `field`: the positions, at the nodes; every node must carry every
///   component.
/// - `components`: one component name per spatial axis, in axis order; `None`
///   → `["X", "Y", "Z"][:dim]`, the components of
///   `pyrucast.node_field.positions`.
///
/// # Raises
///
/// `RuntimeError` if a node of `field` lacks one of `components`, or if
/// `components` does not hold one name per axis.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "set_positions")]
#[pyo3(signature = (field, components=None))]
pub fn set(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    crate::ops::coords::set(&field.inner, components)?;
    Ok(())
}

/// Move the nodes of `field` by its values, an increment: `x += u`.
///
/// The increment is added to the active configuration of the field's
/// `Coords`, in place, each distinct node once. What the field carries is
/// applied, and nothing else: a node carrying none of the components (a
/// multiplier node of a solution) does not move, and a component a node lacks
/// moves it by 0 along that axis.
///
/// # Parameters
///
/// - `field`: the displacement, at the nodes.
/// - `components`: one displacement component per spatial axis, in axis
///   order; `None` → `["u_x", "u_y", "u_z"][:dim]`.
///
/// # Raises
///
/// `RuntimeError` if `components` does not hold one name per axis.
///
/// # See also
///
/// `pyrucast.coords.displace_from`, which lays a configuration at `X + u`
/// without accumulating.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn displace(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    crate::ops::coords::displace(&field.inner, components)?;
    Ok(())
}

/// Lay configuration `target` at `reference` displaced by `field`: `x = X + u`.
///
/// The position is absolute: called again, it lands at `X + u` again, nothing
/// accumulates. What the field carries is applied, and nothing else: a node it
/// does not define, or that carries none of the components (a multiplier node
/// of a solution), stays at its reference position, and a component a node
/// lacks counts as 0 — the output of `pyrucast.solver.solve` is given as it
/// is. The active configuration is left as it is; `Coords.in_config` computes
/// on `target`.
///
/// # Parameters
///
/// - `field`: the displacement `u`, at the nodes.
/// - `reference`: the index of the configuration holding `X`, left unchanged.
/// - `target`: the index of the configuration to write, another than
///   `reference`; what it held before is replaced.
/// - `components`: one displacement component per spatial axis, in axis
///   order; `None` → `["u_x", "u_y", "u_z"][:dim]`.
///
/// # Raises
///
/// `RuntimeError` if `reference` or `target` is not a configuration of the
/// `Coords`, if `target` is `reference`, if `components` does not hold one name
/// per axis, or if a node would get a negative radius on an axisymmetric
/// `Coords`.
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
