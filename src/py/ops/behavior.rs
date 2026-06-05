//! Python wrappers for the behaviour operators in [`crate::ops::behavior`].
//!
//! Free functions that integrate the constitutive law of a model (Cast3m
//! `COMP`). Kept here — mirroring `src/ops/behavior.rs` — per the `py/ops/`
//! convention (their identity is the *operation*, not the `ElementField`
//! they return).

use crate::py::element_field::PyElementField;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Compute the deformation field of `model` from a nodal `solution`.
///
/// Returns one `SubElementField` per behaviour-bearing sub-model (in model
/// order), each on its own FE subspace, carrying the deformation components
/// of its physics (`grad_T_x`, … for heat conduction). Constraint
/// sub-models (Dirichlet, …) are skipped.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn deformation(model: PyRef<PyModel>, solution: PyRef<PyNodeField>) -> PyResult<PyElementField> {
    let ef = crate::ops::behavior::deformation(&model.inner, &solution.handle)?;
    Ok(PyElementField { inner: ef })
}

/// Integrate the constitutive law of `model` (Cast3m `COMP`).
///
/// `deformation` is the behaviour-input field (typically from
/// `deformation(model, solution)`); `materials` supplies the per-zone
/// material data. Returns the material-state field (dual flux/stress +
/// updated internal variables) of every behaviour-bearing sub-model.
///
/// For a linear law the result is consistent with the assembled stiffness
/// (`∫ Bᵀ·flux = K·u`); a non-linear law is the exact response.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn integrate_behavior(
    model: PyRef<PyModel>,
    deformation: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef =
        crate::ops::behavior::integrate(&model.inner, &deformation.inner, &materials.inner)?;
    Ok(PyElementField { inner: ef })
}
