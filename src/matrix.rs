//! Sparse matrix indexed by **named DOFs** `(NodeId, field_name)`.
//!
//! [`Matrix`] is the output container of [`crate::model::Model`] assembly
//! (stiffness, mass, …). Rows and columns are identified by a [`DofId`] —
//! a pair `(NodeId, field index)` where the field index points into a
//! small per-matrix table of names. This keeps storage compact (no string
//! per entry) while preserving the semantics: every row/column tells the
//! user which variable at which node it represents.
//!
//! Storage is **COO** (coordinate triplet list): each insertion appends a
//! `(row_idx, col_idx, value)` entry. Multiple insertions at the same
//! `(row_idx, col_idx)` are kept as-is and **sum** when the matrix is
//! read or densified. This makes assembly trivially incremental and
//! commutes with the order in which sub-models contribute.
//!
//! A `symmetric: bool` flag records whether the assembler intends the
//! matrix to be numerically symmetric (`A[i, j] = A[j, i]` for all paired
//! row/column indices). The flag is **informative only**: the storage
//! does not de-duplicate the lower triangle. Solvers that exploit
//! symmetry (e.g. Cholesky) read the flag to decide on factorization;
//! solvers that do not just see the full COO list.
//!
//! Row and column DOF sets can have **different sizes** (rectangular
//! matrices — e.g. the Lagrange-multiplier block of a Dirichlet
//! constraint). They can also have **different field names** (rows
//! tagged with dual variables such as `q` while columns are tagged with
//! primal variables such as `T`).
//!
//! # Example
//!
//! ```
//! use pyrucast::configuration::NodeId;
//! use pyrucast::matrix::Matrix;
//!
//! let mut k = Matrix::new(true);
//! // Heat conduction on two nodes:
//! //  row `q` at node i × col `T` at node j.
//! k.add_entry(NodeId(0), "q", NodeId(0), "T", 2.0);
//! k.add_entry(NodeId(0), "q", NodeId(1), "T", -1.0);
//! k.add_entry(NodeId(1), "q", NodeId(0), "T", -1.0);
//! k.add_entry(NodeId(1), "q", NodeId(1), "T", 2.0);
//!
//! assert_eq!(k.n_rows(), 2);
//! assert_eq!(k.n_cols(), 2);
//! assert!(k.symmetric());
//! assert_eq!(k.get(NodeId(0), "q", NodeId(0), "T"), 2.0);
//! assert_eq!(k.get(NodeId(1), "q", NodeId(0), "T"), -1.0);
//!
//! // Repeated insertions at the same (row, col) accumulate:
//! k.add_entry(NodeId(0), "q", NodeId(0), "T", 1.5);
//! assert_eq!(k.get(NodeId(0), "q", NodeId(0), "T"), 3.5);
//! ```

use crate::configuration::NodeId;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── DofId ─────────────────────────────────────────────────────────────────

/// Identifier of one row or one column of a [`Matrix`].
///
/// A DOF is the pair `(node_id, field_idx)` where `field_idx` indexes into
/// the owning matrix's field-name table (see [`Matrix::field_name`]). Two
/// DOFs compare equal iff both components match.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct DofId {
    /// Node this DOF lives on.
    pub node_id: NodeId,
    /// Index into the owning matrix's field-name table.
    pub field_idx: u32,
}

// ─── Matrix ────────────────────────────────────────────────────────────────

/// Sparse matrix in COO format with DOFs labelled by `(NodeId, field name)`.
#[derive(Serialize, Deserialize)]
pub struct Matrix {
    /// Field names referenced by `row_dofs` and `col_dofs`.
    ///
    /// A name appears at most once in this table; `field_idx` of a DOF
    /// indexes into it.
    field_names: Vec<String>,
    /// Row DOFs, in insertion order.
    row_dofs: Vec<DofId>,
    /// Column DOFs, in insertion order.
    col_dofs: Vec<DofId>,
    /// COO entries `(row_idx, col_idx, value)`. May contain duplicates;
    /// reads sum them.
    entries: Vec<(u32, u32, f64)>,
    symmetric: bool,
}

impl Matrix {
    /// Build an empty matrix. The `symmetric` flag is **informative**: it
    /// is read by solvers that can exploit symmetry, but the storage is
    /// not de-duplicated.
    pub fn new(symmetric: bool) -> Self {
        Self {
            field_names: Vec::new(),
            row_dofs: Vec::new(),
            col_dofs: Vec::new(),
            entries: Vec::new(),
            symmetric,
        }
    }

    /// Whether the assembler declared the matrix numerically symmetric.
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }

    /// Number of distinct row DOFs (number of rows of the dense view).
    pub fn n_rows(&self) -> usize {
        self.row_dofs.len()
    }

    /// Number of distinct column DOFs (number of columns of the dense view).
    pub fn n_cols(&self) -> usize {
        self.col_dofs.len()
    }

    /// Number of COO entries stored (counting duplicates at the same
    /// `(row, col)`).
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Field-name table referenced by row and column DOFs.
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Name of the field with index `idx`. Panics on out-of-range index.
    pub fn field_name(&self, idx: u32) -> &str {
        &self.field_names[idx as usize]
    }

    /// Index of `name` in the field-name table, or `None` if absent.
    pub fn field_index(&self, name: &str) -> Option<u32> {
        self.field_names
            .iter()
            .position(|n| n == name)
            .map(|i| i as u32)
    }

    /// All row DOFs, in insertion order.
    pub fn row_dofs(&self) -> &[DofId] {
        &self.row_dofs
    }

    /// All column DOFs, in insertion order.
    pub fn col_dofs(&self) -> &[DofId] {
        &self.col_dofs
    }

    /// Append an entry at `(row_node, row_field, col_node, col_field)`.
    ///
    /// The row and column DOFs are created on first use; the field names
    /// are interned in the matrix's table. Repeated calls at the same
    /// `(row, col)` accumulate: the final value at that position is the
    /// sum of all `value`s added.
    pub fn add_entry(
        &mut self,
        row_node: NodeId,
        row_field: &str,
        col_node: NodeId,
        col_field: &str,
        value: f64,
    ) {
        let row_field_idx = self.intern_field(row_field);
        let col_field_idx = self.intern_field(col_field);
        let row_idx = self.find_or_insert_row(row_node, row_field_idx);
        let col_idx = self.find_or_insert_col(col_node, col_field_idx);
        self.entries.push((row_idx, col_idx, value));
    }

    /// Sum of all entries at `(row, col)`. Returns `0.0` if no entry has
    /// ever been added there (or if the DOFs were never seen by the
    /// matrix).
    pub fn get(
        &self,
        row_node: NodeId,
        row_field: &str,
        col_node: NodeId,
        col_field: &str,
    ) -> f64 {
        let row = match self
            .field_index(row_field)
            .and_then(|fi| self.row_index(row_node, fi))
        {
            Some(i) => i,
            None => return 0.0,
        };
        let col = match self
            .field_index(col_field)
            .and_then(|fi| self.col_index(col_node, fi))
        {
            Some(i) => i,
            None => return 0.0,
        };
        self.entries
            .iter()
            .filter(|&&(r, c, _)| r == row && c == col)
            .map(|&(_, _, v)| v)
            .sum()
    }

    /// Iterate over the raw COO triplets, in insertion order. Each
    /// triplet is `(row_dof, col_dof, value)`.
    pub fn iter_entries(&self) -> impl Iterator<Item = (DofId, DofId, f64)> + '_ {
        self.entries.iter().map(move |&(r, c, v)| {
            (
                self.row_dofs[r as usize],
                self.col_dofs[c as usize],
                v,
            )
        })
    }

    /// Materialise the matrix as a flat row-major dense buffer of length
    /// `n_rows × n_cols`, with `out[i * n_cols + j]` = sum of all COO
    /// entries at `(row i, col j)`.
    ///
    /// Convenient for testing and for hand-off to a dense linear solver.
    pub fn dense(&self) -> Vec<f64> {
        let nr = self.row_dofs.len();
        let nc = self.col_dofs.len();
        let mut out = vec![0.0; nr * nc];
        for &(r, c, v) in &self.entries {
            out[r as usize * nc + c as usize] += v;
        }
        out
    }

    /// Apply this matrix to a dense vector `x` of length `n_cols`,
    /// returning a dense vector `y = A · x` of length `n_rows`.
    ///
    /// Returns an error if `x.len() != n_cols`.
    pub fn mul_dense(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_cols() {
            return Err(PyrucastError::Message(format!(
                "mul_dense: x has length {} but matrix has {} columns",
                x.len(),
                self.n_cols()
            )));
        }
        let mut y = vec![0.0; self.n_rows()];
        for &(r, c, v) in &self.entries {
            y[r as usize] += v * x[c as usize];
        }
        Ok(y)
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn intern_field(&mut self, name: &str) -> u32 {
        if let Some(idx) = self.field_names.iter().position(|n| n == name) {
            return idx as u32;
        }
        self.field_names.push(name.to_string());
        (self.field_names.len() - 1) as u32
    }

    fn find_or_insert_row(&mut self, node_id: NodeId, field_idx: u32) -> u32 {
        let dof = DofId { node_id, field_idx };
        if let Some(idx) = self.row_dofs.iter().position(|d| *d == dof) {
            return idx as u32;
        }
        self.row_dofs.push(dof);
        (self.row_dofs.len() - 1) as u32
    }

    fn find_or_insert_col(&mut self, node_id: NodeId, field_idx: u32) -> u32 {
        let dof = DofId { node_id, field_idx };
        if let Some(idx) = self.col_dofs.iter().position(|d| *d == dof) {
            return idx as u32;
        }
        self.col_dofs.push(dof);
        (self.col_dofs.len() - 1) as u32
    }

    fn row_index(&self, node_id: NodeId, field_idx: u32) -> Option<u32> {
        let dof = DofId { node_id, field_idx };
        self.row_dofs
            .iter()
            .position(|d| *d == dof)
            .map(|i| i as u32)
    }

    fn col_index(&self, node_id: NodeId, field_idx: u32) -> Option<u32> {
        let dof = DofId { node_id, field_idx };
        self.col_dofs
            .iter()
            .position(|d| *d == dof)
            .map(|i| i as u32)
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("n_rows", &self.row_dofs.len())
            .field("n_cols", &self.col_dofs.len())
            .field("entries", &self.entries.len())
            .field("symmetric", &self.symmetric)
            .field("fields", &self.field_names)
            .finish()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix: {} row(s) × {} col(s), {} entries{}",
            self.row_dofs.len(),
            self.col_dofs.len(),
            self.entries.len(),
            if self.symmetric { ", symmetric" } else { "" }
        )
    }
}

// ─── Python binding ────────────────────────────────────────────────────────

#[cfg(feature = "python-api")]
mod python {
    use super::*;
    use crate::store::{insert, with, Handle};
    use pyo3::exceptions::PyIndexError;
    use pyo3::prelude::*;

    /// Python wrapper for [`Matrix`].
    #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
    #[pyclass(name = "Matrix")]
    pub struct PyMatrix {
        pub(crate) handle: Handle<Matrix>,
    }

    #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
    #[pymethods]
    impl PyMatrix {
        /// `Matrix(symmetric=False)`.
        #[new]
        #[pyo3(signature = (symmetric=false))]
        fn py_new(symmetric: bool) -> PyResult<Self> {
            Ok(Self {
                handle: insert(Matrix::new(symmetric)),
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

        /// Matrix-vector product `y = A · x` (dense).
        fn mul_dense(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
            Ok(with(&self.handle, |m| m.mul_dense(&x))??)
        }

        /// Iterator-like materialisation: list of `(row_node, row_field,
        /// col_node, col_field, value)` tuples, in insertion order.
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

    // Silence unused-import warning from the conditional bring-in.
    #[allow(dead_code)]
    fn _silence(_: PyIndexError) {}
}

#[cfg(feature = "python-api")]
pub use python::PyMatrix;

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(i: u32) -> NodeId {
        NodeId(i)
    }

    #[test]
    fn empty_matrix() {
        let m = Matrix::new(false);
        assert_eq!(m.n_rows(), 0);
        assert_eq!(m.n_cols(), 0);
        assert_eq!(m.entry_count(), 0);
        assert!(!m.symmetric());
    }

    #[test]
    fn symmetric_flag_round_trip() {
        let m = Matrix::new(true);
        assert!(m.symmetric());
    }

    #[test]
    fn add_entry_interns_fields_and_dofs() {
        let mut m = Matrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(0), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        assert_eq!(m.field_names().len(), 2); // "q" + "T"
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.entry_count(), 4);
    }

    #[test]
    fn get_unknown_returns_zero() {
        let m = Matrix::new(false);
        assert_eq!(m.get(nid(0), "x", nid(0), "y"), 0.0);
    }

    #[test]
    fn get_sums_duplicates() {
        let mut m = Matrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(0), "T", 1.5);
        m.add_entry(nid(0), "q", nid(0), "T", -0.5);
        assert_eq!(m.get(nid(0), "q", nid(0), "T"), 3.0);
    }

    #[test]
    fn dense_matches_get() {
        let mut m = Matrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(0), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        let d = m.dense();
        assert_eq!(d, vec![2.0, -1.0, -1.0, 2.0]);
        // Order of rows / cols is insertion order.
        for (i, rd) in m.row_dofs().iter().enumerate() {
            for (j, cd) in m.col_dofs().iter().enumerate() {
                let v = m.get(rd.node_id, m.field_name(rd.field_idx), cd.node_id, m.field_name(cd.field_idx));
                assert_eq!(d[i * m.n_cols() + j], v);
            }
        }
    }

    #[test]
    fn mul_dense_against_known_matrix() {
        // K = [[2, -1], [-1, 2]],  x = [1, 1],  y = K x = [1, 1]
        let mut m = Matrix::new(true);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(0), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        let y = m.mul_dense(&[1.0, 1.0]).unwrap();
        assert_eq!(y, vec![1.0, 1.0]);
        let y = m.mul_dense(&[1.0, 2.0]).unwrap();
        assert_eq!(y, vec![0.0, 3.0]);
    }

    #[test]
    fn mul_dense_rejects_wrong_size() {
        let mut m = Matrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 1.0);
        assert!(m.mul_dense(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn rectangular_matrix_distinct_row_and_col_dofs() {
        // Lagrange-multiplier block: rows = multiplier DOFs, cols = primal DOFs.
        let mut c = Matrix::new(false);
        c.add_entry(nid(100), "T", nid(3), "T", 1.0); // mult node 100 constrains real node 3
        c.add_entry(nid(101), "T", nid(7), "T", 1.0);
        assert_eq!(c.n_rows(), 2);
        assert_eq!(c.n_cols(), 2);
        assert_eq!(c.field_names().len(), 1); // both "T" map to one field
        // Verify the row DOFs are at multiplier nodes, col DOFs at real nodes:
        assert_eq!(c.row_dofs()[0].node_id, nid(100));
        assert_eq!(c.col_dofs()[0].node_id, nid(3));
    }

    #[test]
    fn iter_entries_preserves_insertion_order() {
        let mut m = Matrix::new(false);
        m.add_entry(nid(0), "a", nid(0), "b", 1.0);
        m.add_entry(nid(1), "a", nid(1), "b", 2.0);
        m.add_entry(nid(0), "a", nid(0), "b", 3.0); // duplicate of first
        let entries: Vec<_> = m.iter_entries().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].2, 1.0);
        assert_eq!(entries[1].2, 2.0);
        assert_eq!(entries[2].2, 3.0);
    }

    #[test]
    fn distinct_field_names_distinct_dofs() {
        let mut m = Matrix::new(false);
        // Same node, different fields → different DOFs.
        m.add_entry(nid(0), "ux", nid(0), "ux", 1.0);
        m.add_entry(nid(0), "uy", nid(0), "uy", 2.0);
        m.add_entry(nid(0), "ux", nid(0), "uy", 3.0);
        m.add_entry(nid(0), "uy", nid(0), "ux", 4.0);
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.field_names(), &["ux".to_string(), "uy".to_string()]);
    }

    #[test]
    fn round_trip_serde() {
        let mut m = Matrix::new(true);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        use crate::persist::Persist;
        let bytes = m.to_bytes().unwrap();
        let m2 = Matrix::from_bytes(&bytes).unwrap();
        assert_eq!(m2.n_rows(), 2);
        assert_eq!(m2.n_cols(), 2);
        assert!(m2.symmetric());
        assert_eq!(m2.get(nid(0), "q", nid(0), "T"), 2.0);
        assert_eq!(m2.get(nid(0), "q", nid(1), "T"), -1.0);
        assert_eq!(m2.get(nid(1), "q", nid(1), "T"), 2.0);
    }

    #[test]
    fn debug_and_display() {
        let mut m = Matrix::new(true);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        let d = format!("{:?}", m);
        assert!(d.contains("Matrix"));
        assert!(d.contains("n_rows"));
        assert!(d.contains("symmetric"));
        let s = format!("{}", m);
        assert!(s.contains("Matrix"));
        assert!(s.contains("1 row"));
        assert!(s.contains("symmetric"));
    }
}
