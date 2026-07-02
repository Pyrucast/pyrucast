//! Python wrappers for the internal-force operator in
//! [`crate::ops::internal_forces`].
//!
//! Free functions computing the internal nodal forces `f = ∫ Bᵀ σ dΩ` (Cast3m
//! `BSIG`) from a stress field — the transpose of the deformation operator `B`.
//! Kept here — mirroring `src/ops/internal_forces.rs` — per the `py/ops/`
//! convention. The stress input is built separately by `integrate_behavior`
//! (`COMP`, see [`crate::py::ops::behavior`]).

use crate::py::element_field::PyElementField;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of `model` (Cast3m `BSIG`).
///
/// `stresses` is the material-state field produced by `integrate_behavior`
/// (`COMP`). Each behaviour-bearing sub-model applies its own `Bᵀ` (continuum
/// solid, bar or beam) and the forces are scattered to the nodes. Returns a
/// `NodeField` whose components are each sub-model's dual variables (`f_x`, …
/// for a solid/bar; `f_w`, `m_theta` for a beam).
///
/// For a linear law the result equals the assembled stiffness applied to the
/// solution (`K·u`); a non-linear law gives the exact internal forces, so
/// `r = f_ext − f_int` is the residual.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn internal_forces(
    model: PyRef<PyModel>,
    stresses: PyRef<PyElementField>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::internal_forces::internal_forces(&model.inner, &stresses.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Internal nodal forces of a **continuum-mechanics** stress field, without a
/// model (Cast3m `BSIG` for a plain solid).
///
/// Convenience for the volumetric case (elasticity, Mazars, plasticity), where
/// `B` is the universal symmetric gradient: it needs only the geometry
/// (`fespace`) and the Voigt stress (`sigma_xx`, `sigma_xy`, …). Returns a
/// `NodeField` with `space_dim` components `f_x, f_y, f_z` per node. **Bars and
/// beams are not covered** — use `internal_forces(model, stresses)` for those.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn internal_forces_continuum(
    stresses: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyNodeField> {
    let nf =
        crate::ops::internal_forces::internal_forces_continuum(&stresses.inner, &fespace.inner)?;
    Ok(PyNodeField { inner: nf })
}
