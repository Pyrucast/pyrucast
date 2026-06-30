//! Python wrapper for the solver operations in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `NodeField` result).

use crate::ops::solver::lu::{SolveMethod, SolveOptions};
use crate::py::matrix::PyMatrix;
use crate::py::node_field::PyNodeField;
use crate::py::signals::PySignals;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    let method = match method.as_deref() {
        None | Some("lu") | Some("LU") => SolveMethod::Lu,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown solver method '{other}' (expected 'lu')"
            )))
        }
    };
    let options = SolveOptions { method, cache };
    let solution = crate::ops::solver::lu::solve_cancellable_with_options(
        &matrix.inner,
        &rhs.inner,
        &options,
        &PySignals(py),
    )?;
    Ok(PyNodeField { inner: solution })
}
