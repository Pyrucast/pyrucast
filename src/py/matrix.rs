//! Python wrappers for [`crate::containers::matrix::SubMatrix`] and the
//! aggregate [`crate::containers::matrix::Matrix`].

use crate::aggregate::Aggregate;
use crate::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use crate::containers::mesh::configuration::NodeId;
use crate::py::mesh::PySubMesh;
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
    /// `SubMatrix(row_mesh, col_mesh, dual_vars, primal_vars, ordering="nodes_then_vars", symmetric=False)`.
    ///
    /// `ordering` is either `"nodes_then_vars"` (default) or `"vars_then_nodes"`.
    #[new]
    #[pyo3(signature = (row_mesh, col_mesh, dual_vars, primal_vars, ordering="nodes_then_vars", symmetric=false))]
    fn py_new(
        row_mesh: PyRef<'_, PySubMesh>,
        col_mesh: PyRef<'_, PySubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: &str,
        symmetric: bool,
    ) -> PyResult<Self> {
        let ord = match ordering {
            "nodes_then_vars" => DofOrdering::NodesThenVars,
            "vars_then_nodes" => DofOrdering::VarsThenNodes,
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown ordering '{other}'; expected 'nodes_then_vars' or 'vars_then_nodes'"
            ))),
        };
        let sub = SubMatrix::new(
            row_mesh.handle.clone(),
            col_mesh.handle.clone(),
            dual_vars,
            primal_vars,
            ord,
            symmetric,
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { handle: insert(sub) })
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
        })??;
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
        Ok(with(&self.handle, |m| m.field_names())?)
    }

    /// `(node_id, field_name)` tuples for each row, in order.
    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| {
            m.row_dofs().into_iter().map(|(nid, name)| (nid.0, name)).collect()
        })?)
    }

    /// `(node_id, field_name)` tuples for each column, in order.
    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(with(&self.handle, |m| {
            m.col_dofs().into_iter().map(|(nid, name)| (nid.0, name)).collect()
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
            m.iter_entries().into_iter().map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v)).collect()
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

/// Python wrapper for the aggregate [`Matrix`].
///
/// Owns the `Matrix` struct directly — no longer stored in the global
/// store. Identity is the Python object identity itself.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Matrix")]
pub struct PyMatrix {
    pub(crate) inner: Matrix,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMatrix {
    /// `Matrix()` — empty aggregate. Populate via `add_sub_matrix`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self { inner: Matrix::empty() })
    }

    fn add_sub_matrix(&mut self, sub: PyRef<'_, PySubMatrix>) -> PyResult<()> {
        self.inner.add_sub(sub.handle.clone())?;
        Ok(())
    }

    /// Build the global DOF table and CSR. Must be called before any
    /// solver-facing method (`dense`, `mul_dense`, …).
    fn finalize(&mut self) -> PyResult<()> {
        self.inner.finalize()?;
        Ok(())
    }

    fn sub_matrix_count(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn sub_matrix(&self, i: usize) -> PyResult<PySubMatrix> {
        let h = Aggregate::get(&self.inner, i)?;
        Ok(PySubMatrix { handle: h })
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn n_rows(&self) -> PyResult<usize> {
        Ok(self.inner.n_rows()?)
    }

    fn n_cols(&self) -> PyResult<usize> {
        Ok(self.inner.n_cols()?)
    }

    fn entry_count(&self) -> PyResult<usize> {
        Ok(self.inner.entry_count()?)
    }

    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(self.inner.symmetric()?)
    }

    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.field_names()?)
    }

    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .inner
            .row_dofs()?
            .into_iter()
            .map(|(n, name)| (n.0, name))
            .collect())
    }

    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .inner
            .col_dofs()?
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
        Ok(self
            .inner
            .get(NodeId(row_node), row_field, NodeId(col_node), col_field)?)
    }

    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.dense()?)
    }

    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.inner.mul_dense(&x)?)
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples — every entry across every block, in block-insertion order.
    fn entries(&self) -> PyResult<Vec<(u32, String, u32, String, f64)>> {
        Ok(self
            .inner
            .iter_entries()?
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}
