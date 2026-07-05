//! Python wrapper for the solver operations in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `NodeField` result).

use crate::ops::solver::lu::{SolveMethod, SolveOptions};
use crate::ops::solver::unilateral::{ActiveSetMethod, UnilateralOptions};
use crate::py::matrix::PyMatrix;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use crate::py::signals::PySignals;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Resolve the optional `method` string into a [`SolveMethod`] (the direct
/// back-end for the — possibly reduced — system). Shared by `solve` and
/// `solve_eliminate`.
fn parse_method(method: Option<&str>) -> PyResult<SolveMethod> {
    match method {
        None | Some("lu") | Some("LU") => Ok(SolveMethod::Lu),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown solver method '{other}' (expected 'lu')"
        ))),
    }
}

/// Resolve the optional `active_set` string into an [`ActiveSetMethod`] for
/// `solve_unilateral` (`"schur"` default, or `"refactorize"`).
fn parse_active_set(active_set: Option<&str>) -> PyResult<ActiveSetMethod> {
    match active_set {
        None | Some("schur") | Some("Schur") => Ok(ActiveSetMethod::SchurComplement),
        Some("refactorize") | Some("Refactorize") => Ok(ActiveSetMethod::Refactorize),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown active-set method '{other}' (expected 'schur' or 'refactorize')"
        ))),
    }
}

/// Solve the linear system `A·x = b` for `x` (sparse LU, faer).
///
/// `matrix` is the finalized system `A`; `rhs` is the right-hand side `b`
/// as a `NodeField` (read through the aggregate, zones resolved per DOF).
/// Returns the solution `x` as a single-zone `NodeField` over the
/// column-DOF nodes.
///
/// `method` selects the direct solver (currently only `"lu"`, the default).
/// `cache` (default `True`) reuses a factorization stored transparently on the
/// matrix: the first solve factorizes, later solves on the same matrix reuse the
/// factors (much cheaper). The cache is cleared automatically when the matrix
/// changes.
///
/// A `Ctrl+C` is honoured at the solver's phase boundaries. The factorization
/// itself is a single library call and is not interrupted mid-way; when it is
/// already cached, only the (cheap) substitution runs.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (matrix, rhs, method=None, cache=true))]
pub fn solve(
    py: Python<'_>,
    matrix: PyRef<PyMatrix>,
    rhs: PyRef<PyNodeField>,
    method: Option<String>,
    cache: bool,
) -> PyResult<PyNodeField> {
    let options = SolveOptions {
        method: parse_method(method.as_deref())?,
        cache,
    };
    let solution = crate::ops::solver::lu::solve_cancellable_with_options(
        &matrix.inner,
        &rhs.inner,
        &options,
        &PySignals(py),
    )?;
    Ok(PyNodeField { inner: solution })
}

/// Solve `model`'s constrained system by **master/slave elimination**
/// (condensation) — the alternative to the Lagrange-multiplier path of
/// [`solve`].
///
/// `model` is the constrained model; `matrix` is its assembled (saddle-point)
/// stiffness; `rhs` is the load field (its right-hand sides `g` live at the
/// multiplier nodes' imposed-value slots). Each linear relation eliminates one
/// slave DOF, so the system solved is smaller and definite (no multiplier DOFs).
/// The solution carries the primal field at every physics node plus each slave's
/// reaction (the multiplier equivalent) in its dual row.
///
/// A model with no constraint falls back to a plain [`solve`]. v1 scope:
/// non-chained, disjoint slaves (a slave DOF may not appear in another relation).
///
/// `method` selects the direct back-end for the reduced system (currently only
/// `"lu"`). `cache` (default `True`) reuses the condensation stored transparently
/// on the matrix, cleared when the matrix changes. `Ctrl+C` is honoured at phase
/// boundaries.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, matrix, rhs, method=None, cache=true))]
pub fn solve_eliminate(
    py: Python<'_>,
    model: PyRef<PyModel>,
    matrix: PyRef<PyMatrix>,
    rhs: PyRef<PyNodeField>,
    method: Option<String>,
    cache: bool,
) -> PyResult<PyNodeField> {
    let options = SolveOptions {
        method: parse_method(method.as_deref())?,
        cache,
    };
    let solution = crate::ops::solver::eliminate::solve_cancellable_with_options(
        &model.inner,
        &matrix.inner,
        &rhs.inner,
        &options,
        &PySignals(py),
    )?;
    Ok(PyNodeField { inner: solution })
}

/// Solve `model`'s system with **unilateral** (inequality) constraints by the
/// active-set (status) method — the operator for constraints built with
/// `sense=">="` / `"<="` (Dirichlet, MPC).
///
/// `model` is the constrained model; `matrix` is its assembled (saddle-point)
/// stiffness; `rhs` is the load field (the right-hand sides `g` live at the
/// multiplier nodes' imposed-value slots). The status loop starts with every
/// inequality active (or from the previous converged status when `cache` is
/// on — a warm start), solves, releases the relations whose reaction pulls,
/// activates the relations whose gap penetrates, and repeats until the status
/// is stable. Inactive relations report `λ = 0` in the solution.
///
/// A model with no inequality relation falls back to a plain `solve`.
///
/// `method` selects the direct back-end of each iteration (currently only
/// `"lu"`). `active_set` selects how each status's system is factorized:
/// `"schur"` (default) factorizes the inequality-free base once and updates it
/// per status (falling back to refactorization when that base is singular),
/// `"refactorize"` refactorizes the full system at each status change. `cache`
/// (default `True`) stores the active-set state transparently on the matrix
/// (cleared when the matrix changes). `max_iter` (default `100`) bounds the
/// status loop; `tol` (default `1e-10`) is the sign tolerance on the multiplier
/// and the gap. `Ctrl+C` is honoured at each iteration boundary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, matrix, rhs, method=None, active_set=None, cache=true, max_iter=100, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
pub fn solve_unilateral(
    py: Python<'_>,
    model: PyRef<PyModel>,
    matrix: PyRef<PyMatrix>,
    rhs: PyRef<PyNodeField>,
    method: Option<String>,
    active_set: Option<String>,
    cache: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyNodeField> {
    let options = UnilateralOptions {
        method: parse_method(method.as_deref())?,
        active_set: parse_active_set(active_set.as_deref())?,
        cache,
        max_iter,
        tol,
    };
    let solution = crate::ops::solver::unilateral::solve_cancellable_with_options(
        &model.inner,
        &matrix.inner,
        &rhs.inner,
        &options,
        &PySignals(py),
    )?;
    Ok(PyNodeField { inner: solution })
}
