//! Python wrappers for the behaviour operator in [`crate::ops::element_field::behavior`].
//!
//! Free function that integrates the constitutive law of a model (Cast3m
//! `COMP`). Kept here — mirroring `src/ops/behavior.rs` — per the `py/ops/`
//! convention (its identity is the *operation*, not the `ElementField` it
//! returns). The deformation input is built separately by
//! `gradient` / `deformation` (see [`crate::py::ops::field`]).

use crate::py::element_field::PyElementField;
use crate::py::model::PyModel;
use pyo3::prelude::*;

/// Integrate the constitutive law of `model` (Cast3m `COMP`), stepping A → B.
///
/// `deformation` is the **end-of-step** behaviour input ε(B) (from
/// `gradient(field, fespace)` or `deformation(u, fespace)`); `prev` is the
/// **converged output of the previous step** (the state at A — stress,
/// internal variables and start-of-step strain), or `None` on the first step;
/// `materials` supplies the per-zone material data; `dt` is the time increment
/// (`None` if the law is rate-independent). Returns the material-state field at
/// B (dual flux/stress + updated internal variables) of every behaviour-bearing
/// sub-model — feed it back as `prev` at the next step.
///
/// For a linear law the result is consistent with the assembled stiffness
/// (`∫ Bᵀ·flux = K·u`); a non-linear law is the exact response.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, deformation, materials, prev=None, dt=None))]
pub fn integrate_behavior(
    model: PyRef<PyModel>,
    deformation: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    prev: Option<PyRef<PyElementField>>,
    dt: Option<f64>,
) -> PyResult<PyElementField> {
    let prev_inner = prev.as_ref().map(|p| &p.inner);
    let ef = crate::ops::element_field::behavior::integrate(
        &model.inner,
        &deformation.inner,
        prev_inner,
        &materials.inner,
        dt,
    )?;
    Ok(PyElementField { inner: ef })
}
