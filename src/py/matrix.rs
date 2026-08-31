//! Python wrappers for [`crate::containers::matrix::SubMatrix`] and the
//! aggregate [`crate::containers::matrix::Matrix`].

use crate::aggregate::Aggregate;
use crate::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use crate::handle::Handle;
use crate::models::Physics;
use crate::py::mesh::submesh_handle;
use crate::py::node::PyNode;
use crate::py::node_field::PyNodeField;
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
        self.handle
            .write()
            .add_entry(rn, row_field, cn, col_field, value)?;
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
        Ok(self.handle.read().get(rn, row_field, cn, col_field))
    }

    /// Number of rows of this block.
    fn n_rows(&self) -> PyResult<usize> {
        Ok(self.handle.read().n_rows())
    }

    /// Number of columns of this block.
    fn n_cols(&self) -> PyResult<usize> {
        Ok(self.handle.read().n_cols())
    }

    /// Number of stored COO entries.
    fn entry_count(&self) -> PyResult<usize> {
        Ok(self.handle.read().entry_count())
    }

    /// Whether this block is declared symmetric.
    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(self.handle.read().symmetric())
    }

    /// The scalar factor applied to every value this block emits (`1.0` unless
    /// the parent `Matrix` was built via `matrix * scalar` / `matrix / scalar`).
    #[getter]
    fn factor(&self) -> PyResult<f64> {
        Ok(self.handle.read().factor())
    }

    /// The physics nature(s) of the sub-model that produced this block, as a list
    /// of tags (`"mechanical"`, `"thermal"`, `"constraint"`, `"other"`). **Empty**
    /// for a block built outside assembly (the "rien" case), or several tags for a
    /// coupled physics. Set by the assembler; used by `Matrix.filter`.
    fn physics(&self) -> PyResult<Vec<String>> {
        Ok(self
            .handle
            .read()
            .physics()
            .iter()
            .map(|p| p.name().to_string())
            .collect())
    }

    /// Variable (field) names this block addresses.
    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(self.handle.read().field_names())
    }

    /// `(node_id, field_name)` tuples for each row, in order.
    fn row_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .handle
            .read()
            .row_dofs()
            .into_iter()
            .map(|(nid, name)| (nid.0, name))
            .collect())
    }

    /// `(node_id, field_name)` tuples for each column, in order.
    fn col_dofs(&self) -> PyResult<Vec<(u32, String)>> {
        Ok(self
            .handle
            .read()
            .col_dofs()
            .into_iter()
            .map(|(nid, name)| (nid.0, name))
            .collect())
    }

    /// Dense row-major buffer, length `n_rows × n_cols`.
    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().dense())
    }

    /// `y = A · x` (dense).
    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().mul_dense(&x)?)
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples, in insertion order.
    fn entries(&self) -> PyResult<PyMatrixEntries> {
        Ok(self
            .handle
            .read()
            .iter_entries()
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.handle.read().entry_count())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.handle.read()))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", *self.handle.read()))
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
        m.add_sub(Handle::new(sub))?;
        Ok(Self { inner: m })
    }

    /// Re-assemble this matrix **from its blocks alone** — no `Model` —
    /// mutating it in place. The composition path: after combining blocks of
    /// any provenance (via `matrix * scalar` / `matrix / scalar`, `|` union,
    /// `add_sub`, `filter`, …), including *computed* ones (which `finalize()`
    /// refuses — the element kernel lives outside `containers`), call this to
    /// fold everything into one CSR. Needed, for instance, to solve
    /// `(M/dt + K) u = …` : `sys = (m / dt) | k; sys.assemble();
    /// pyrucast.solver.solve(sys, rhs)`.
    fn assemble(&mut self) -> PyResult<()> {
        self.inner.assemble()?;
        Ok(())
    }

    /// Build the global DOF table and CSR. Must be called before any
    /// solver-facing method (`dense`, `mul_dense`, …).
    fn finalize(&mut self) -> PyResult<()> {
        self.inner.finalize()?;
        Ok(())
    }

    /// `Matrix.filter(physics)` — a new `Matrix` holding only the blocks **whose
    /// nature set contains** the given physics (`"mechanical"`, `"thermal"`,
    /// `"constraint"`, `"other"`, `"diffusion"`, `"radiation"`). The result is
    /// **not** finalized — call `assemble` (or `finalize` for literal-only
    /// blocks) before solving.
    fn filter(&self, physics: Physics) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.filter(physics)?,
        })
    }

    /// `Matrix.physics()` — the list of physics natures present across the
    /// matrix's blocks (first-seen, deduplicated). Empty if no block is tagged;
    /// several tags when the matrix aggregates several physics (e.g. a heat model
    /// with a Dirichlet → `["thermal", "constraint"]`).
    fn physics(&self) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .physics()
            .iter()
            .map(|p| p.name().to_string())
            .collect())
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
        Ok(self.inner.entry_count())
    }

    /// Whether the matrix is declared symmetric.
    #[getter]
    fn symmetric(&self) -> PyResult<bool> {
        Ok(self.inner.symmetric())
    }

    /// Variable (field) names across the whole matrix.
    fn field_names(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.field_names())
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
        Ok(self.inner.get(rn, row_field, cn, col_field))
    }

    /// Dense row-major buffer of the finalized matrix (`n_rows × n_cols`).
    fn dense(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.dense()?)
    }

    /// `y = A · x` against a dense vector `x`.
    fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.inner.mul_dense(&x)?)
    }

    /// `matrix / scalar` — a fresh `Matrix` whose blocks carry the divided
    /// `factor` (lazy). Not finalized; see `__mul__`.
    fn __truediv__(&self, rhs: f64) -> PyResult<PyMatrix> {
        Ok(PyMatrix {
            inner: (&self.inner / rhs)?,
        })
    }

    /// List of `(row_node, row_field, col_node, col_field, value)`
    /// tuples — every entry across every block, in block-insertion order.
    fn entries(&self) -> PyResult<PyMatrixEntries> {
        Ok(self
            .inner
            .iter_entries()
            .into_iter()
            .map(|(rn, rf, cn, cf, v)| (rn.0, rf, cn.0, cf, v))
            .collect())
    }
}

// Polymorphic product — **closed block**, undecorated on purpose (see
// `impl_aggregate_pymethods!`): its `.pyi` entries are the hand-written
// overloads submitted just below.
#[pymethods]
impl PyMatrix {
    /// `matrix * x` — either a matrix-vector product against a `NodeField` (`x`
    /// read at the matrix's column DOFs, result a fresh `NodeField` over its row
    /// DOFs) or, for a `float`, a scalar scale: a fresh `Matrix` whose blocks
    /// carry the scaled `factor` (lazy — no value is rewritten). The scaled
    /// result is **not** finalized; call `finalize()` (or `assemble` for computed
    /// blocks) before solving or querying it.
    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = rhs.py();
        if let Ok(field) = rhs.extract::<PyRef<'_, PyNodeField>>() {
            let inner = self.inner.mul_field(&field.inner)?;
            return Ok(Py::new(py, PyNodeField { inner })?.into_any());
        }
        let s: f64 = rhs.extract()?;
        let inner = (&self.inner * s)?;
        Ok(Py::new(py, PyMatrix { inner })?.into_any())
    }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PyMatrix:
    @overload
    def __mul__(self, rhs: pyo3_stub_gen.RustType["PyNodeField"]) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`matrix * x` → the matrix-vector product: `x` is read at the column
        DOFs, the result is a fresh `NodeField` over the row DOFs."""
    @overload
    def __mul__(self, rhs: float) -> pyo3_stub_gen.RustType["PyMatrix"]:
        """`matrix * scalar` → a fresh `Matrix` whose blocks carry the scaled
        `factor` (lazy — no value is rewritten). **Not** finalized: call
        `finalize()` (or `assemble` for computed blocks) before solving."""
    "# }
}

crate::impl_aggregate_pymethods!(
    PyMatrix,
    PySubMatrix,
    "Matrix",
    sub_matrix,
    Matrix,
    r#"
class PyMatrix:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubMatrix"]:
        """`matrix[i]` → the `SubMatrix` view of block i (one rectangular COO
        block, with its row and column supports)."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyMatrix"]:
        """`matrix[i:j:k]` → a fresh `Matrix` holding the sliced blocks, shared
        with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyMatrix"] | pyo3_stub_gen.RustType["PySubMatrix"]) -> pyo3_stub_gen.RustType["PyMatrix"]:
        """`matrix | other` → a fresh `Matrix` holding the blocks of both, in
        first-seen order and deduplicated by object identity. Assemble the global
        operator this way, then call `finalize()` before solving."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubMatrix"]) -> pyo3_stub_gen.RustType["PyMatrix"]:
        """`sub_matrix | matrix` — the mirror of `matrix | sub_matrix`,
        differing only in that the lone block comes first."""
    "#,
    r#"
class PySubMatrix:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubMatrix"]) -> pyo3_stub_gen.RustType["PyMatrix"]:
        """`sub_matrix | sub_matrix` → a fresh `Matrix` holding both blocks."""
    "#
);
crate::impl_dump_pymethod!(handle PySubMatrix, handle);
