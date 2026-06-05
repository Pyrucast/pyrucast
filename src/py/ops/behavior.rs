//! Python wrappers for the behaviour operator in [`crate::ops::behavior`].
//!
//! Free function that integrates the constitutive law of a model (Cast3m
//! `COMP`). Kept here — mirroring `src/ops/behavior.rs` — per the `py/ops/`
//! convention (its identity is the *operation*, not the `ElementField` it
//! returns). The deformation input is built separately by
//! `gradient` / `deformation` (see [`crate::py::ops::field`]).

use crate::py::element_field::PyElementField;
use crate::py::model::PyModel;
use pyo3::prelude::*;

/// Integrate the constitutive law of `model` (Cast3m `COMP`).
///
/// `deformation` is the behaviour-input field (from `gradient(field,
/// fespace)` or `deformation(u, fespace)`); `materials` supplies the
/// per-zone material data. Returns the material-state field (dual
/// flux/stress + updated internal variables) of every behaviour-bearing
/// sub-model.
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
