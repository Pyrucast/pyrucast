//! Python wrappers for [`crate::ops::matrix`] — the assemblers, which
//! produce a `Matrix`.

use crate::py::element_field::PyElementField;
use crate::py::matrix::PyMatrix;
use crate::py::model::PyModel;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// Assemble the stiffness matrix `K` of `model`.
///
/// One matrix covers every unknown of the model, multipliers included: a physics
/// contributes its block, a constraint the two halves `C` and `Cᵀ` of its
/// relations. Boundary conditions have no special status — they are sub-models
/// like the others.
///
/// # Parameters
///
/// - `model`: the physics and the constraints to assemble.
/// - `materials`: the material data, zone by zone. Every sub-model that needs
///   some picks the zone whose FE subspace is its own; a sub-model without
///   material (a constraint) ignores it.
///
/// # Returns
///
/// The matrix, assembled on the configuration active at the call.
///
/// # Raises
///
/// `RuntimeError` if no zone of `materials` lies on the FE subspace of a
/// sub-model with the components its physics requires.
#[py_op(method_on = PyModel, name = "stiffness_matrix")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn stiffness(model: PyRef<PyModel>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
    let k = crate::ops::matrix::stiffness(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: k })
}

/// Assemble the mass matrix `M` of `model`, or its heat-capacity matrix `C`.
///
/// Mechanics assembles the consistent mass `M = ∫ ρ N_i N_j`, heat conduction the
/// capacity `C = ∫ ρ c_p N_i N_j`. A physics without a mass term (a boundary
/// exchange, a constraint) contributes nothing.
///
/// # Parameters
///
/// - `model`: the physics to assemble.
/// - `materials`: the material data, zone by zone, as for [`stiffness`]. It must
///   carry `rho` on a mechanical zone, `rho` and `cp` on a thermal one: the
///   stiffness does not read them, so the material field holds them only if
///   given.
///
/// # Returns
///
/// The matrix, assembled on the configuration active at the call.
///
/// # Raises
///
/// `RuntimeError` if `rho` or `cp` is missing on a zone that needs it.
#[py_op(method_on = PyModel, name = "mass_matrix")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn mass(model: PyRef<PyModel>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::mass(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: m })
}

/// Lump an assembled matrix into a diagonal one, each term the sum of its row.
///
/// Applied to a consistent mass or capacity matrix, it yields the lumped
/// (diagonal) mass, which conserves the total mass:
/// `Σ_i M_lump[i,i] = Σ_ij M[i,j]`. It is the cheap, uncoupled form explicit
/// time schemes use.
///
/// # Parameters
///
/// - `matrix`: an assembled, square matrix whose row `i` and column `i` carry
///   the same node.
///
/// # Returns
///
/// A new matrix with the rows and the columns of `matrix`, in the same order,
/// and one diagonal block per row support: it is solved, multiplied by a field
/// and saved like `matrix`, which is left untouched.
///
/// # Raises
///
/// `RuntimeError` if `matrix` has no assembled values (an output of `|` or
/// `filter`, before `assemble()`).
#[py_op(method_on = PyMatrix)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn lump(matrix: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::lump(&matrix.inner)?;
    Ok(PyMatrix { inner: m })
}

/// Assemble the geometric (initial-stress) stiffness `K_g` of `model`.
///
/// `K_g[(i,a),(j,b)] = δ_ab ∫ ∇N_i · σ · ∇N_j` is the stiffening a stress in
/// place brings: the term of buckling and of prestressed analyses. It does not
/// depend on the constitutive law. A physics without geometric stiffness (heat
/// conduction, a constraint) contributes nothing.
///
/// # Parameters
///
/// - `model`: the physics to assemble.
/// - `materials`: the material data, zone by zone, as for [`stiffness`].
/// - `stress`: the current Cauchy stress at the Gauss points, components
///   `sigma_*` — typically what `element_field.integrate_behavior` returns. Each
///   zone is picked by its FE subspace, like `materials`.
///
/// # Returns
///
/// The matrix, assembled on the configuration active at the call.
///
/// # Raises
///
/// `RuntimeError` if `stress` lacks a `sigma_*` component on a mechanical zone.
#[py_op(method_on = PyModel, name = "geometric_matrix")]
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

/// Assemble the consistent (algorithmic) tangent `K_t = ∫ Bᵀ D_alg B` of `model`.
///
/// `D_alg` is the derivative of the step `ε(B) ↦ σ(B)`, the state at A held
/// fixed: both ends of the step are in its definition, hence the same inputs as
/// `element_field.integrate_behavior`. This is the matrix that gives a Newton
/// iteration its quadratic convergence.
///
/// A linear physics (conduction, diffusion, exchanges, trusses, beams, shells)
/// and the constraints have no law to differentiate: their tangent is their
/// stiffness, assembled without reading `deformation`.
///
/// # Parameters
///
/// - `model`: the physics and the constraints to assemble.
/// - `materials`: the material data, zone by zone, as for [`stiffness`].
/// - `deformation`: the input of the laws at the end of the step, `ε(B)` — from
///   `element_field.deformation`.
/// - `prev`: the state at A, the converged output of the previous step; on a
///   first step, `element_field.initial_state(model, materials)`.
/// - `dt`: the time increment; `None` when no law depends on the rate.
///
/// # Returns
///
/// The matrix, assembled on the configuration active at the call.
///
/// # Raises
///
/// `RuntimeError` if a non-linear law of `model` has no tangent of its own
/// (`model.mazars`, `model.damage_tc`, `model.damage_sic_sic`): the call fails
/// rather than return an empty block.
///
/// # Notes
///
/// `D_alg` is evaluated at each Gauss point during the assembly: no field of
/// moduli is stored between the law and this function.
#[py_op(method_on = PyModel, name = "tangent_matrix")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, materials, deformation, prev, dt=None))]
pub fn tangent(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
    deformation: PyRef<PyElementField>,
    prev: PyRef<PyElementField>,
    dt: Option<f64>,
) -> PyResult<PyMatrix> {
    let m = crate::ops::matrix::tangent(
        &model.inner,
        &materials.inner,
        &deformation.inner,
        &prev.inner,
        dt,
    )?;
    Ok(PyMatrix { inner: m })
}

/// `matrix` projected on `modes`: `K*ᵢⱼ = φᵢᵀ K φⱼ`, on the modes' support.
///
/// Each row of `matrix` meets a shape through the primal its dual is paired with
/// (the mode's `"pairs"`): a row `f_x` reads `u_x`. The relations' rows and the
/// multipliers' columns pair nothing, so a stiffness projects the same with or
/// without them. A symmetric `matrix` gives a symmetric one, exactly.
///
/// # Parameters
///
/// - `matrix`: an assembled matrix on the modes' unknowns.
/// - `modes`: the modes, as `pyrucast.solver.eigen` returns them or written by
///   hand; they share one support whose nodes are exactly theirs.
///
/// # Returns
///
/// A `Matrix` on the support of the modes, rows `f_alpha` and columns `alpha`, one
/// of each per mode, at its node. It adds to any other projection on the same
/// modes, and `pyrucast.solver.solve` returns its coordinates there. An empty
/// list gives an empty matrix.
///
/// # Raises
///
/// `ValueError` if a mode lacks a key. `RuntimeError` if the modes do not share
/// one support whose nodes are exactly theirs — a slice or a union of lists needs
/// its own: `s = pyrucast.mesh.poi1_from_nodes([m["node"] for m in modes])`, then
/// `[{**m, "support": s} for m in modes]` — or if a mode is complex, or if
/// `matrix` has no assembled values or carries none of the modes' unknowns.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "project_on_basis_matrix")]
pub fn project_on_basis(
    matrix: PyRef<PyMatrix>,
    modes: Vec<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyMatrix> {
    let modes = crate::py::ops::solver::modes_from_dicts(&modes)?;
    Ok(PyMatrix {
        inner: crate::ops::matrix::project_on_basis(&matrix.inner, &modes)?,
    })
}
