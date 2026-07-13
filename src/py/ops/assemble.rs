//! Python wrappers for the assembly operations in [`crate::ops::assemble`].
//!
//! Free functions that assemble a global [`PyMatrix`] from a model, plus the
//! internal nodal forces (`BSIG`, `∫ Bᵀ σ`). Kept here — mirroring
//! `src/ops/assemble/` — per the `py/ops/` convention.

use crate::containers::node_field::NodeField;
use crate::ops::assemble::FluxDensity;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::{PyFiniteElementSpace, PySubFiniteElementSpace};
use crate::py::matrix::PyMatrix;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use pyo3::exceptions::PyValueError;
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

/// Consistent nodal loads of a distributed flux over `fespace` — the analogue
/// of Cast3m `FLUX` / `PRES`: `∫ density · N_i dΓ`, returned as a `NodeField`
/// carrying the single component `component` (the model's dual variable, e.g.
/// `"q"` for heat conduction).
///
/// `density` is either a **float** (uniform density) or a single-component
/// `SubElementField` (per-Gauss density). The element measure comes from the
/// FE subspace, so a `SEG2` edge in a 2-D mesh integrates as a line, a surface
/// mesh as an area.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn flux(
    fespace: PyRef<PySubFiniteElementSpace>,
    density: &Bound<'_, PyAny>,
    component: &str,
) -> PyResult<PyNodeField> {
    let sub = if let Ok(value) = density.extract::<f64>() {
        crate::ops::assemble::flux(&fespace.handle, FluxDensity::Uniform(value), component)?
    } else if let Ok(field) = density.extract::<PyRef<PySubElementField>>() {
        crate::ops::assemble::flux(
            &fespace.handle,
            FluxDensity::Field(&field.handle),
            component,
        )?
    } else {
        return Err(PyValueError::new_err(
            "flux: `density` doit être un float ou un SubElementField",
        ));
    };
    Ok(PyNodeField {
        inner: NodeField::from_sub(sub),
    })
}

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
    let nf = crate::ops::assemble::internal_forces(&model.inner, &stresses.inner)?;
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
        crate::ops::assemble::internal_forces_continuum(&stresses.inner, &fespace.inner)?;
    Ok(PyNodeField { inner: nf })
}
