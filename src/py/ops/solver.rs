//! Python wrapper for the solver operations in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `NodeField` result).

use crate::ops::solver::lu::{SolveMethod, SolveOptions};
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
