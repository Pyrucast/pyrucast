//! Python wrappers for the field operations in [`crate::ops::field`].
//!
//! Free functions that build or transform a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/field/` — rather than on the field classes, per
//! the `py/ops/` convention (operations live with operations).
//!
//! Transitional note: the Rust ops still work on a single
//! `SubNodeField`; this wrapper bridges through `Aggregate::unit()`
//! (unitary aggregates only) until the ops are generalised to the
//! `NodeField` aggregate.

use crate::py::element_field::PyElementField;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Build a `NodeField` carrying the coordinates of every node of `mesh`
/// — one `SubNodeField` per submesh, on the distinct nodes of its zone.
///
/// One component per requested axis (`"X"`, `"Y"`, `"Z"`). `components=None`
/// requests all the axes the mesh's `Configuration` has (`["X"]` in 1-D,
/// `["X", "Y"]` in 2-D, `["X", "Y", "Z"]` in 3-D).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn coordinates(
    mesh: PyRef<PyMesh>,
    components: Option<Vec<String>>,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::coordinates(&mesh.inner, components)?,
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
    crate::ops::field::set_coordinates(&field.inner, components)?;
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
    crate::ops::field::displace(&field.inner, components)?;
    Ok(())
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `NodeField` with one zone per submesh of `mesh`,
/// carrying the union of `field`'s components. Nodes of `mesh` absent
/// from `field` are assigned `0.0`. Errors if `mesh` and `field` are
/// attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::restrict(&field.inner, &mesh.inner)?,
    })
}

/// Merge two node fields « au plus juste »: structural union of their
/// zones, consolidated — zones sharing a component set are fused, the
/// others stay separate (nothing is densified).
///
/// Errors if the two fields hold different values at the same
/// `(node, component)` pair, or are attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge(a: PyRef<PyNodeField>, b: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::merge(&a.inner, &b.inner)?,
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
    let ef = crate::ops::field::gradient(&field.inner, &fespace.inner)?;
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
    let ef = crate::ops::field::deformation(&u.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Timoshenko-beam section strains `(kappa, gamma)` of a `(w, theta)` node
/// field at the Gauss points of `fespace`. Feed the result to
/// `integrate_behavior` of a Timoshenko model to obtain the section forces
/// `M = E·I·κ` and `V = G·A_s·γ`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn beam_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::field::beam_deformation(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}
