//! Python wrapper for [`crate::ops::solver::lu::solve`].

use crate::py::matrix::PyMatrix;
use crate::py::node_field::PyNodeField;
use crate::store::{insert, with};
use pyo3::prelude::*;

/// `pyrucast.solve(matrix, rhs) -> NodeField`
///
/// Dense LU solver. See [`crate::ops::solver::lu::solve`] for the semantics
/// of the rhs and of the returned NodeField.
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
