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
/// (Cast3M `KTAN`). `D_alg` is evaluated **at the Gauss point**, from the same
/// inputs `integrate_behavior` takes — no field of moduli is materialised, since
/// this assembler would be its only reader. `prev=None` means the rest state.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, materials, deformation, prev=None, dt=None))]
pub fn tangent(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
    deformation: PyRef<PyElementField>,
    prev: Option<PyRef<PyElementField>>,
    dt: Option<f64>,
) -> PyResult<PyMatrix> {
    let prev_inner = prev.as_ref().map(|p| &p.inner);
    let m = crate::ops::matrix::tangent(
        &model.inner,
        &materials.inner,
        &deformation.inner,
        prev_inner,
        dt,
    )?;
    Ok(PyMatrix { inner: m })
}

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs ci-dessus (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Aucune logique : chaque méthode rappelle la
// fonction libre, receveur compris.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// Voir `pyrucast.matrix.stiffness`.
    fn stiffness_matrix(
        slf: PyRef<'_, Self>,
        materials: PyRef<PyElementField>,
    ) -> PyResult<PyMatrix> {
        super::matrix::stiffness(slf, materials)
    }

    /// Voir `pyrucast.matrix.mass`.
    fn mass_matrix(slf: PyRef<'_, Self>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
        super::matrix::mass(slf, materials)
    }

    /// Voir `pyrucast.matrix.geometric`.
    fn geometric_matrix(
        slf: PyRef<'_, Self>,
        materials: PyRef<PyElementField>,
        stress: PyRef<PyElementField>,
    ) -> PyResult<PyMatrix> {
        super::matrix::geometric(slf, materials, stress)
    }

    /// Voir `pyrucast.matrix.tangent`.
    #[pyo3(signature = (materials, deformation, prev=None, dt=None))]
    fn tangent_matrix(
        slf: PyRef<'_, Self>,
        materials: PyRef<PyElementField>,
        deformation: PyRef<PyElementField>,
        prev: Option<PyRef<PyElementField>>,
        dt: Option<f64>,
    ) -> PyResult<PyMatrix> {
        super::matrix::tangent(slf, materials, deformation, prev, dt)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMatrix {
    /// Voir `pyrucast.matrix.lump`.
    fn lump(slf: PyRef<'_, Self>) -> PyResult<PyMatrix> {
        super::matrix::lump(slf)
    }
}
