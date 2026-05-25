//! Python wrappers for [`crate::containers::matrix::SubMatrix`] and the
//! aggregate [`crate::containers::matrix::Matrix`].

use crate::aggregate::Aggregate;
use crate::containers::matrix::{Matrix, SubMatrix};
use crate::containers::mesh::configuration::NodeId;
use crate::store::{insert, with, Handle};
use pyo3::prelude::*;

// ─── PySubMatrix ───────────────────────────────────────────────────────────

/// Python wrapper for [`SubMatrix`] — one COO block of the global matrix.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubMatrix")]
pub struct PySubMatrix {
    pub(crate) handle: Handle<SubMatrix>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubMatrix {
    /// `SubMatrix(symmetric=False)`.
    #[new]
    #[pyo3(signature = (symmetric=false))]
    fn py_new(symmetric: bool) -> PyResult<Self> {
        Ok(Self {
            handle: insert(SubMatrix::new(symmetric)),
        })
    }

    fn add_entry(
        &self,
        row_node: u32,
        row_field: &str,
        col_node: u32,
        col_field: &str,
        value: f64,
    ) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |m| {
            m.add_entry(NodeId(row_node), row_field, NodeId(col_node), col_field, value)
        })?;
        Ok(())
    }

    fn get(
        &self,
        row_node: u32,
        row_field: &str,
        col_node: u32,
        col_field: &str,
    ) -> PyResult<f64> {
        Ok(with(&self.handle, |m| {
            m.get(NodeId(row_node), row_field, NodeId(col_node), col_field)
        })?)
    }

    fn n_rows(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.n_rows())?)
    }

    fn n_cols(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.n_cols())?)
    }

    fn entry_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.entry_count())?)
    }

    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(with(&self.handle, |m| m.symmetric())?)
    }

    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |m| m.field_names().to_vec())?)
    }

    /// `(node_id, field_name)` tuples for each row, in order.
    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| {
            m.row_dofs()
                .iter()
                .map(|d| (d.node_id.0, m.field_name(d.field_idx).to_string()))
                .collect()
        })?)
    }

    /// `(node_id, field_name)` tuples for each column, in order.
    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| {
            m.col_dofs()
                .iter()
                .map(|d| (d.node_id.0, m.field_name(d.field_idx).to_string()))
                .collect()
        })?)
    }

    /// Dense row-major buffer, length `n_rows × n_cols`.
    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |m| m.dense())?)
    }

    /// `y = A · x` (dense).
    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |m| m.mul_dense(&x))??)
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples, in insertion order.
    fn entries(&self) -> PyResult<Vec<(u32, String, u32, String, f64)>> {
        Ok(with(&self.handle, |m| {
            m.iter_entries()
                .map(|(r, c, v)| {
                    (
                        r.node_id.0,
                        m.field_name(r.field_idx).to_string(),
                        c.node_id.0,
                        m.field_name(c.field_idx).to_string(),
                        v,
                    )
                })
                .collect()
        })?)
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.entry_count())?)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |m| format!("{:?}", m))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |m| format!("{}", m))?)
    }
}

// ─── PyMatrix (aggregate) ──────────────────────────────────────────────────

/// Python wrapper for the aggregate [`Matrix`]. Read-only: every accessor
/// unions the underlying [`PySubMatrix`] blocks on the fly.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Matrix")]
pub struct PyMatrix {
    pub(crate) handle: Handle<Matrix>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMatrix {
    /// `Matrix()` — empty aggregate. Populate via `add_sub_matrix`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            handle: insert(Matrix::empty()),
        })
    }

    fn add_sub_matrix(&self, sub: PyRef<'_, PySubMatrix>) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |m| m.add_sub(sub.handle.clone()))??;
        Ok(())
    }

    fn sub_matrix_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.len())?)
    }

    fn sub_matrix(&self, i: usize) -> PyResult<PySubMatrix> {
        let h = with(&self.handle, |m| Aggregate::get(m, i))??;
        Ok(PySubMatrix { handle: h })
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.len())?)
    }

    fn n_rows(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.n_rows())??)
    }

    fn n_cols(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.n_cols())??)
    }

    fn entry_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |m| m.entry_count())??)
    }

    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(with(&self.handle, |m| m.symmetric())??)
    }

    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |m| m.field_names())??)
    }

    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| m.row_dofs())??
            .into_iter()
            .map(|(n, name)| (n.0, name))
            .collect())
    }

    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| m.col_dofs())??
            .into_iter()
            .map(|(n, name)| (n.0, name))
            .collect())
    }

    fn get(
        &self,
        row_node: u32,
        row_field: &str,
        col_node: u32,
        col_field: &str,
    ) -> PyResult<f64> {
        Ok(with(&self.handle, |m| {
            m.get(NodeId(row_node), row_field, NodeId(col_node), col_field)
        })??)
    }

    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |m| m.dense())??)
    }

    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |m| m.mul_dense(&x))??)
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples — every entry across every block, in block-insertion order.
    fn entries(&self) -> PyResult<Vec<(u32, String, u32, String, f64)>> {
        Ok(with(&self.handle, |m| m.iter_entries())??
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |m| format!("{:?}", m))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |m| format!("{}", m))?)
    }
}
