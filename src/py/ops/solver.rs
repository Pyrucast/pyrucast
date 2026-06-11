//! Python wrapper for the solver operations in [`crate::ops::solver`].
//!
//! Free function solving a linear system into a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/solver/` — per the `py/ops/` convention (its identity
//! is the *operation*, not its `SubNodeField` result).

use crate::py::matrix::PyMatrix;
use crate::py::node_field::PyNodeField;
use crate::store::{insert, with};
use pyo3::prelude::*;

/// Solve the linear system `A·x = b` for `x` (dense LU).
///
/// `matrix` is the finalized system `A`; `rhs` is the right-hand side `b`
/// as a `SubNodeField`. Returns the solution `x` as a `SubNodeField` over the
/// same support.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn solve(matrix: PyRef<PyMatrix>, rhs: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    let solution = with(&rhs.handle, |r| {
        crate::ops::solver::lu::solve(&matrix.inner, r)
    })??;
    Ok(PyNodeField {
        handle: insert(solution),
    })
}
