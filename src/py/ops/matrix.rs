//! Python wrappers for [`crate::ops::matrix`] — the assemblers, which
//! produce a `Matrix`.

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
    let k = crate::ops::matrix::stiffness(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: k })
}

/// Assemble the consistent mass matrix `M` of `model` (Cast3M `MASS`), or the
/// heat-capacity matrix `C` for a thermal model (Cast3M `CAPA`).
///
/// Mechanics assembles `M = ∫ ρ Nᵀ N` (material `rho`); heat conduction
/// assembles `C = ∫ ρ cp Nᵀ N` (material `rho`, `cp`). `materials` carries the
/// per-zone coefficients, exactly like [`stiffness`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn mass(model: PyRef<PyModel>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::mass(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: m })
}

/// Lump an assembled matrix into a diagonal one by row-sum concentration
/// (Cast3M `LUMP`). Applied to a consistent mass / capacity matrix it yields the
/// diagonal (lumped) mass, conserving the total mass.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn lump(matrix: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::lump(&matrix.inner)?;
    Ok(PyMatrix { inner: m })
}

/// Assemble the geometric (initial-stress) stiffness `K_g` of `model` (Cast3M
/// `KSIG`), from the current stress field `stress` (Voigt-named `sigma_*`).
/// `materials` resolves each mechanical zone, exactly like [`stiffness`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn geometric(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
    stress: PyRef<PyElementField>,
) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::geometric(&model.inner, &materials.inner, &stress.inner)?;
    Ok(PyMatrix { inner: m })
}

/// Assemble the consistent (algorithmic) tangent `K_t = ∫ Bᵀ D_alg B` of `model`
/// (Cast3M `KTAN`), from the behaviour `state` (which carries `D_alg` besides the
/// stress). `materials` resolves each zone, like [`stiffness`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn tangent(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
    state: PyRef<PyElementField>,
) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::tangent(&model.inner, &materials.inner, &state.inner)?;
    Ok(PyMatrix { inner: m })
}
