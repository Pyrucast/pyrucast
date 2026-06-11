//! Python wrappers for the field operations in [`crate::ops::field`].
//!
//! Free functions that build or transform a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/field/` — rather than on the `SubNodeField` class, per
//! the `py/ops/` convention (operations live with operations).

use crate::py::element_field::PyElementField;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use crate::store::{insert, with};
use pyo3::prelude::*;

/// Build a `SubNodeField` carrying the coordinates of every node of `mesh`.
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

/// Set node coordinates from `field` (absolute): for every node, the
/// active-set coordinate on axis `a` becomes `field.value(node,
/// components[a])`. `components` lists one component name per spatial axis,
/// in axis order; `None` → `["X", "Y", "Z"][:dim]`. Mutates the field's
/// `Configuration` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn set_coordinates(
    field: PyRef<PyNodeField>,
    components: Option<Vec<String>>,
) -> PyResult<()> {
    with(&field.handle, |nf| {
        crate::ops::field::set_coordinates(nf, components)
    })??;
    Ok(())
}

/// Displace nodes by `field` (incremental): `coord[a] += field.value(node,
/// components[a])` on the active coordinate set. `components` lists one
/// displacement-component name per spatial axis, in axis order; `None` →
/// `["ux", "uy", "uz"][:dim]`. Mutates the field's `Configuration` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn displace(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    with(&field.handle, |nf| {
        crate::ops::field::displace(nf, components)
    })??;
    Ok(())
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `SubNodeField` with the same components, supported on the
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
    // locking `a` rather than nesting two `with::<SubNodeField>` calls.
    let fb = with(&b.handle, |f| f.clone())?;
    let result = with(&a.handle, |fa| crate::ops::field::merge(fa, &fb))??;
    Ok(PyNodeField {
        handle: insert(result),
    })
}

/// Gradient `∇f` of a node `field` at the Gauss points of `fespace`.
///
/// Geometric and physics-agnostic: each component of `field` is
/// differentiated w.r.t. every spatial axis, giving an `ElementField` with
/// one component `grad_<name>_<axis>` per (input component, axis) pair
/// (`grad_T_x`, …). Feed the result to `integrate_behavior`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn gradient(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::field::gradient(&field.handle, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Linearized (small-strain) deformation `ε = ½(∇u + ∇uᵀ)` of a displacement
/// field `u` at the Gauss points of `fespace`.
///
/// `u` must carry exactly `space_dim` components, taken in order as the
/// displacement along x, y, z. Returns the symmetric strain tensor
/// (`eps_xx`, `eps_xy`, … in tensor convention). The only deformation
/// measure for now; non-linear ones will share this shape.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn deformation(
    u: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::field::deformation(&u.handle, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}
