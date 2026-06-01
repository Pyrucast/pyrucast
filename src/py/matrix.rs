//! Python wrappers for [`crate::containers::matrix::SubMatrix`] and the
//! aggregate [`crate::containers::matrix::Matrix`].

use crate::aggregate::Aggregate;
use crate::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use crate::py::mesh::submesh_handle;
use crate::py::node::PyNode;
use crate::store::{insert, with, Handle};
use pyo3::prelude::*;

// ─── PySubMatrix ───────────────────────────────────────────────────────────

/// Python wrapper for [`SubMatrix`] — one COO block of the global matrix.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubMatrix")]
pub struct PySubMatrix {
    pub(crate) handle: Handle<SubMatrix>,
}

/// `SubMatrix` is a **view** into a `Matrix` block, obtained by indexing
/// (`matrix[i]`) — it is never constructed directly from Python. Build a
/// block at the parent level with `Matrix.block(...)` (a unit `Matrix`),
/// composed with `+` (see `CONVENTIONS.md`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubMatrix {
    fn add_entry(
        &self,
        row_node: PyRef<'_, PyNode>,
        row_field: &str,
        col_node: PyRef<'_, PyNode>,
        col_field: &str,
        value: f64,
    ) -> PyResult<()> {
        let (rn, cn) = (row_node.as_node().id(), col_node.as_node().id());
        crate::store::with_mut(&self.handle, |m| {
            m.add_entry(rn, row_field, cn, col_field, value)
        })??;
        Ok(())
    }

    fn get(
        &self,
        row_node: PyRef<'_, PyNode>,
        row_field: &str,
        col_node: PyRef<'_, PyNode>,
        col_field: &str,
    ) -> PyResult<f64> {
        let (rn, cn) = (row_node.as_node().id(), col_node.as_node().id());
        Ok(with(&self.handle, |m| {
            m.get(rn, row_field, cn, col_field)
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
    /// `Matrix()` — empty aggregate. Populate via `add_sub_matrix`, or
    /// build blocks with `Matrix.block(...)` and compose them with `+`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self { inner: Matrix::empty() })
    }

    /// `Matrix.block(row_support, col_support, dual_vars, primal_vars, ordering="nodes_then_vars", symmetric=False)`
    /// — a single-block `Matrix` (unit aggregate). `row_support` /
    /// `col_support` may each be a `SubMesh` view or a **unitary** `Mesh`.
    /// `ordering` is `"nodes_then_vars"` (default) or `"vars_then_nodes"`.
    /// Fill entries via the block view (`block[0].add_entry(...)`) and
    /// compose several blocks with `+`, then `finalize()`.
    #[classmethod]
    #[pyo3(signature = (row_support, col_support, dual_vars, primal_vars, ordering="nodes_then_vars", symmetric=false))]
    fn block(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        row_support: &Bound<'_, PyAny>,
        col_support: &Bound<'_, PyAny>,
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
        let row = submesh_handle(row_support)?;
        let col = submesh_handle(col_support)?;
        let sub = SubMatrix::new(row, col, dual_vars, primal_vars, ord, symmetric)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut m = Matrix::empty();
        m.add_sub(insert(sub))?;
        Ok(Self { inner: m })
    }

    /// `matrix_a + matrix_b` — merge two matrices into a fresh aggregate
    /// (union of blocks, first-seen order). Block handles are **shared**
    /// (refcount bump). Call `finalize()` on the result before solving.
    fn __add__(&self, other: PyRef<PyMatrix>) -> PyResult<PyMatrix> {
        let inner = self.inner.merge(&other.inner)?;
        Ok(PyMatrix { inner })
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

    /// The sole block **view** of a unitary matrix (exactly one block),
    /// else a clear error — e.g. `matrix.unit().add_entry(...)`.
    fn unit(&self) -> PyResult<PySubMatrix> {
        let h = self.inner.unit()?;
        Ok(PySubMatrix { handle: h })
    }

    /// `matrix[i]` → `SubMatrix` view on block `i` (negative indices
    /// supported; `IndexError` out of range), so `for block in matrix:`
    /// works and blocks are reachable as views.
    fn __getitem__(&self, idx: isize) -> PyResult<PySubMatrix> {
        let n = self.inner.len();
        let i = crate::aggregate::normalize_index(idx, n).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "Matrix index {idx} out of range (len={n})"
            ))
        })?;
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
        row_node: PyRef<'_, PyNode>,
        row_field: &str,
        col_node: PyRef<'_, PyNode>,
        col_field: &str,
    ) -> PyResult<f64> {
        let (rn, cn) = (row_node.as_node().id(), col_node.as_node().id());
        Ok(self.inner.get(rn, row_field, cn, col_field)?)
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
