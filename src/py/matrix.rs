//! Python wrappers for [`crate::containers::matrix::SubMatrix`] and the
//! aggregate [`crate::containers::matrix::Matrix`].

use crate::aggregate::Aggregate;
use crate::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use crate::py::mesh::submesh_handle;
use crate::py::node::PyNode;
use crate::py::node_field::PyNodeField;
use crate::store::{insert, read, write, Handle};
use pyo3::prelude::*;

/// Flattened COO entries as exposed to Python: one
/// `(row_node, row_field, col_node, col_field, value)` tuple per entry.
type PyMatrixEntries = Vec<(u32, String, u32, String, f64)>;

// ─── PySubMatrix ───────────────────────────────────────────────────────────

/// One block (a COO sub-matrix) of a global `Matrix`, viewed by indexing
/// (`matrix[i]`) — never constructed directly. Build a block at the parent
/// level with `Matrix.block(...)` (a unit `Matrix`), composed with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubMatrix")]
pub struct PySubMatrix {
    pub(crate) handle: Handle<SubMatrix>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubMatrix {
    /// Accumulate `value` at row `(row_node, row_field)`, column
    /// `(col_node, col_field)` (COO entry).
    fn add_entry(
        &self,
        row_node: PyRef<'_, PyNode>,
        row_field: &str,
        col_node: PyRef<'_, PyNode>,
        col_field: &str,
        value: f64,
    ) -> PyResult<()> {
        let (rn, cn) = (row_node.as_node().id(), col_node.as_node().id());
        write(&self.handle)?.add_entry(rn, row_field, cn, col_field, value)?;
        Ok(())
    }

    /// Value at row `(row_node, row_field)`, column `(col_node, col_field)`.
    fn get(
        &self,
        row_node: PyRef<'_, PyNode>,
        row_field: &str,
        col_node: PyRef<'_, PyNode>,
        col_field: &str,
    ) -> PyResult<f64> {
        let (rn, cn) = (row_node.as_node().id(), col_node.as_node().id());
        Ok(read(&self.handle)?.get(rn, row_field, cn, col_field))
    }

    /// Number of rows of this block.
    fn n_rows(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.n_rows())
    }

    /// Number of columns of this block.
    fn n_cols(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.n_cols())
    }

    /// Number of stored COO entries.
    fn entry_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.entry_count())
    }

    /// Whether this block is declared symmetric.
    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(read(&self.handle)?.symmetric())
    }

    /// Variable (field) names this block addresses.
    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.field_names())
    }

    /// `(node_id, field_name)` tuples for each row, in order.
    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(read(&self.handle)?
            .row_dofs()
            .into_iter()
            .map(|(nid, name)| (nid.0, name))
            .collect())
    }

    /// `(node_id, field_name)` tuples for each column, in order.
    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(read(&self.handle)?
            .col_dofs()
            .into_iter()
            .map(|(nid, name)| (nid.0, name))
            .collect())
    }

    /// Dense row-major buffer, length `n_rows × n_cols`.
    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(read(&self.handle)?.dense())
    }

    /// `y = A · x` (dense).
    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(read(&self.handle)?.mul_dense(&x)?)
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples, in insertion order.
    fn entries(&self) -> PyResult<PyMatrixEntries> {
        Ok(read(&self.handle)?
            .iter_entries()
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.entry_count())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
    }
}

// ─── PyMatrix (aggregate) ──────────────────────────────────────────────────

/// A global finite-element matrix, assembled from rectangular blocks
/// (`SubMatrix`).
///
/// Build blocks with `Matrix.block(...)`, compose them with `|`, then call
/// `finalize()` before solving. Index it (`matrix[i]`) to reach a block.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Matrix")]
pub struct PyMatrix {
    pub(crate) inner: Matrix,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMatrix {
    /// `Matrix()` — empty aggregate. Populate via `add_sub_matrix`, or
    /// build blocks with `Matrix.block(...)` and compose them with `|`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            inner: Matrix::empty(),
        })
    }

    /// `Matrix.block(row_support, col_support, dual_vars, primal_vars, ordering="nodes_then_vars", symmetric=False)`
    /// — a single-block `Matrix` (unit aggregate). `row_support` /
    /// `col_support` may each be a `SubMesh` view or a **unitary** `Mesh`.
    /// `ordering` is `"nodes_then_vars"` (default) or `"vars_then_nodes"`.
    /// Fill entries via the block view (`block[0].add_entry(...)`) and
    /// compose several blocks with `|`, then `finalize()`.
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
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown ordering '{other}'; expected 'nodes_then_vars' or 'vars_then_nodes'"
                )))
            }
        };
        let row = submesh_handle(row_support)?;
        let col = submesh_handle(col_support)?;
        let sub = SubMatrix::new(row, col, dual_vars, primal_vars, ord, symmetric)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut m = Matrix::empty();
        m.add_sub(insert(sub))?;
        Ok(Self { inner: m })
    }

    /// Build the global DOF table and CSR. Must be called before any
    /// solver-facing method (`dense`, `mul_dense`, …).
    fn finalize(&mut self) -> PyResult<()> {
        self.inner.finalize()?;
        Ok(())
    }

    /// Total number of rows of the (finalized) global matrix.
    fn n_rows(&self) -> PyResult<usize> {
        Ok(self.inner.n_rows()?)
    }

    /// Total number of columns of the (finalized) global matrix.
    fn n_cols(&self) -> PyResult<usize> {
        Ok(self.inner.n_cols()?)
    }

    /// Total number of stored entries across all blocks.
    fn entry_count(&self) -> PyResult<usize> {
        Ok(self.inner.entry_count()?)
    }

    /// Whether the matrix is declared symmetric.
    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(self.inner.symmetric()?)
    }

    /// Variable (field) names across the whole matrix.
    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.field_names()?)
    }

    /// `(node_id, field_name)` of each global row, in order.
    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .inner
            .row_dofs()?
            .into_iter()
            .map(|(n, name)| (n.0, name))
            .collect())
    }

    /// `(node_id, field_name)` of each global column, in order.
    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .inner
            .col_dofs()?
            .into_iter()
            .map(|(n, name)| (n.0, name))
            .collect())
    }

    /// Value at row `(row_node, row_field)`, column `(col_node, col_field)`.
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

    /// Dense row-major buffer of the finalized matrix (`n_rows × n_cols`).
    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.dense()?)
    }

    /// `y = A · x` against a dense vector `x`.
    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.inner.mul_dense(&x)?)
    }

    /// `y = A · x` against a `NodeField` (`matrix * field`). `x` is read at the
    /// matrix's column DOFs; the result is a fresh `NodeField` over its row DOFs.
    fn __mul__(&self, rhs: PyRef<'_, PyNodeField>) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: self.inner.mul_field(&rhs.inner)?,
        })
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples — every entry across every block, in block-insertion order.
    fn entries(&self) -> PyResult<PyMatrixEntries> {
        Ok(self
            .inner
            .iter_entries()?
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }
}

crate::impl_aggregate_pymethods!(PyMatrix, PySubMatrix, "Matrix", sub_matrix, Matrix);
crate::impl_dump_pymethod!(handle PySubMatrix, handle);
