//! Python wrapper for the solver operations in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `NodeField` result).

use crate::py::matrix::PyMatrix;
use crate::py::node_field::PyNodeField;
use crate::py::signals::PySignals;
use pyo3::prelude::*;

/// Solve the linear system `A·x = b` for `x` (dense LU).
///
/// `matrix` is the finalized system `A`; `rhs` is the right-hand side `b`
/// as a `NodeField` (read through the aggregate, zones resolved per DOF).
/// Returns the solution `x` as a single-zone `NodeField` over the
/// column-DOF nodes.
///
/// A `Ctrl+C` is honoured at the solver's phase boundaries (assembly,
/// dense conversion, factorization). The dense factorization itself is a
/// single library call and is not interrupted mid-way.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn solve(
    py: Python<'_>,
    matrix: PyRef<PyMatrix>,
    rhs: PyRef<PyNodeField>,
) -> PyResult<PyNodeField> {
    let solution =
        crate::ops::solver::lu::solve_cancellable(&matrix.inner, &rhs.inner, &PySignals(py))?;
    Ok(PyNodeField { inner: solution })
}
