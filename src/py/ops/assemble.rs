//! Python wrappers for the assembly operations in [`crate::ops::assemble`].
//!
//! Free functions that assemble a global [`PyMatrix`] from a model. Kept
//! here — mirroring `src/ops/assemble/` — per the `py/ops/` convention.

use crate::py::element_field::PyElementField;
use crate::py::matrix::PyMatrix;
use crate::py::model::PyModel;
use pyo3::prelude::*;

/// Assemble the stiffness matrix `K` of `model`.
///
/// `materials` carries the per-zone material data: every sub-model that
/// needs it picks the `SubElementField` whose FE subspace matches its own.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn stiffness(model: PyRef<PyModel>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
    let k = crate::ops::assemble::stiffness(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: k })
}

/// Assemble the mass matrix `M` of `model`.
///
/// v0 stub: no physics has a mass term yet, so this returns an empty
/// finalized `Matrix` with the model's DOF layout.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn mass(model: PyRef<PyModel>) -> PyResult<PyMatrix> {
    let m = crate::ops::assemble::mass(&model.inner)?;
    Ok(PyMatrix { inner: m })
}
