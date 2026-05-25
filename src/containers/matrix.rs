//! Sparse matrix indexed by **named DOFs** `(NodeId, field_name)`.
//!
//! Hierarchy:
//!
//! - [`SubMatrix`] — one COO block. Carries its own field-name table and
//!   its own row/col DOF lists. Mutating: `add_entry` appends a
//!   `(row_idx, col_idx, value)` triplet; duplicates at the same
//!   `(row, col)` accumulate when the block is read or densified.
//! - [`Matrix`] — aggregate of [`SubMatrix`] blocks (one
//!   `Vec<Handle<SubMatrix>>`), produced by
//!   [`crate::containers::model::Model`] assembly (one block per
//!   sub-model). Read-only: every accessor unions the blocks on the fly.
//!
//! A `symmetric: bool` flag lives on each [`SubMatrix`]. The aggregate
//! [`Matrix`] is reported symmetric iff every one of its blocks is. The
//! flag is **informative only**: storage is not de-duplicated.
//!
//! # Example — single block
//!
//! ```
//! use pyrucast::containers::mesh::configuration::NodeId;
//! use pyrucast::containers::matrix::SubMatrix;
//!
//! let mut k = SubMatrix::new(true);
//! k.add_entry(NodeId(0), "q", NodeId(0), "T", 2.0);
//! k.add_entry(NodeId(0), "q", NodeId(1), "T", -1.0);
//! k.add_entry(NodeId(1), "q", NodeId(0), "T", -1.0);
//! k.add_entry(NodeId(1), "q", NodeId(1), "T", 2.0);
//!
//! assert_eq!(k.n_rows(), 2);
//! assert_eq!(k.n_cols(), 2);
//! assert!(k.symmetric());
//! assert_eq!(k.get(NodeId(0), "q", NodeId(0), "T"), 2.0);
//! ```
//!
//! # Example — aggregate
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::containers::mesh::configuration::NodeId;
//! use pyrucast::containers::matrix::{Matrix, SubMatrix};
//! use pyrucast::store::insert;
//!
//! let mut a = SubMatrix::new(true);
//! a.add_entry(NodeId(0), "q", NodeId(0), "T", 2.0);
//! a.add_entry(NodeId(0), "q", NodeId(1), "T", -1.0);
//!
//! let mut b = SubMatrix::new(true);
//! b.add_entry(NodeId(1), "q", NodeId(0), "T", -1.0);
//! b.add_entry(NodeId(1), "q", NodeId(1), "T", 2.0);
//!
//! let mut k = Matrix::empty();
//! k.add_sub(insert(a)).unwrap();
//! k.add_sub(insert(b)).unwrap();
//!
//! assert_eq!(k.n_rows().unwrap(), 2);
//! assert_eq!(k.n_cols().unwrap(), 2);
//! assert!(k.symmetric().unwrap());
//! assert_eq!(k.get(NodeId(0), "q", NodeId(0), "T").unwrap(), 2.0);
//! assert_eq!(k.get(NodeId(1), "q", NodeId(1), "T").unwrap(), 2.0);
//! ```

use crate::containers::mesh::configuration::NodeId;
use crate::error::{PyrucastError, Result};
use crate::store::{with, Handle};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── DofId ─────────────────────────────────────────────────────────────────

/// Identifier of one row or one column of a [`SubMatrix`].
///
/// A DOF is the pair `(node_id, field_idx)` where `field_idx` indexes into
/// the owning sub-matrix's field-name table (see [`SubMatrix::field_name`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct DofId {
    /// Node this DOF lives on.
    pub node_id: NodeId,
    /// Index into the owning sub-matrix's field-name table.
    pub field_idx: u32,
}

// ─── SubMatrix ─────────────────────────────────────────────────────────────

/// One sparse COO block with DOFs labelled by `(NodeId, field name)`.
#[derive(Serialize, Deserialize)]
pub struct SubMatrix {
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

impl SubMatrix {
    /// Build an empty sub-matrix. The `symmetric` flag is **informative**:
    /// it is read by solvers that can exploit symmetry, but the storage
    /// is not de-duplicated.
    pub fn new(symmetric: bool) -> Self {
        Self {
            field_names: Vec::new(),
            row_dofs: Vec::new(),
            col_dofs: Vec::new(),
            entries: Vec::new(),
            symmetric,
        }
    }

    /// Whether the assembler declared this block numerically symmetric.
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
    /// are interned in the sub-matrix's table. Repeated calls at the
    /// same `(row, col)` accumulate.
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
    /// ever been added there.
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

    /// Iterate over the raw COO triplets, in insertion order.
    pub fn iter_entries(&self) -> impl Iterator<Item = (DofId, DofId, f64)> + '_ {
        self.entries.iter().map(move |&(r, c, v)| {
            (
                self.row_dofs[r as usize],
                self.col_dofs[c as usize],
                v,
            )
        })
    }

    /// Materialise as a row-major dense buffer of length `n_rows × n_cols`.
    pub fn dense(&self) -> Vec<f64> {
        let m = self.to_dmatrix();
        let mut out = Vec::with_capacity(m.nrows() * m.ncols());
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out.push(m[(i, j)]);
            }
        }
        out
    }

    /// Materialise as a [`nalgebra::DMatrix<f64>`] of size
    /// `n_rows × n_cols`. Entries at the same `(row, col)` are summed.
    pub fn to_dmatrix(&self) -> DMatrix<f64> {
        let nr = self.row_dofs.len();
        let nc = self.col_dofs.len();
        let mut out = DMatrix::<f64>::zeros(nr, nc);
        for &(r, c, v) in &self.entries {
            out[(r as usize, c as usize)] += v;
        }
        out
    }

    /// Convert this block to a [`nalgebra_sparse::CooMatrix`].
    pub fn to_coo(&self) -> CooMatrix<f64> {
        let nr = self.row_dofs.len();
        let nc = self.col_dofs.len();
        let mut coo = CooMatrix::<f64>::new(nr, nc);
        for &(r, c, v) in &self.entries {
            coo.push(r as usize, c as usize, v);
        }
        coo
    }

    /// Convert this block to a [`nalgebra_sparse::CsrMatrix`].
    pub fn to_csr(&self) -> CsrMatrix<f64> {
        CsrMatrix::from(&self.to_coo())
    }

    /// Convert this block to a [`nalgebra_sparse::CscMatrix`].
    pub fn to_csc(&self) -> CscMatrix<f64> {
        CscMatrix::from(&self.to_coo())
    }

    /// `y = A · x` (dense). Returns an error if `x.len() != n_cols`.
    pub fn mul_dense(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_cols() {
            return Err(PyrucastError::Message(format!(
                "mul_dense: x has length {} but sub-matrix has {} columns",
                x.len(),
                self.n_cols()
            )));
        }
        let csr = self.to_csr();
        let x_vec = DVector::<f64>::from_column_slice(x);
        let y_vec: DVector<f64> = &csr * &x_vec;
        Ok(y_vec.iter().copied().collect())
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

impl fmt::Debug for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubMatrix")
            .field("n_rows", &self.row_dofs.len())
            .field("n_cols", &self.col_dofs.len())
            .field("entries", &self.entries.len())
            .field("symmetric", &self.symmetric)
            .field("fields", &self.field_names)
            .finish()
    }
}

impl fmt::Display for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubMatrix: {} row(s) × {} col(s), {} entries{}",
            self.row_dofs.len(),
            self.col_dofs.len(),
            self.entries.len(),
            if self.symmetric { ", symmetric" } else { "" }
        )
    }
}

// ─── Matrix (aggregate) ────────────────────────────────────────────────────

/// Aggregate of [`SubMatrix`] blocks. Read-only: every accessor unions
/// the contributions of all blocks on the fly.
///
/// Internally a `Vec<Handle<SubMatrix>>` — see [`crate::aggregate::Aggregate`].
/// Two blocks may share row/col DOFs; the aggregate sums their entries
/// at coincident `(row, col)`.
#[derive(Serialize, Deserialize, Default)]
pub struct Matrix {
    subs: Vec<Handle<SubMatrix>>,
}

crate::impl_aggregate!(Matrix, SubMatrix, sub_matrix, "sub-matrix(es)", {
    fn display_extra(&self) -> Option<String> {
        let n_rows = self.n_rows().unwrap_or(0);
        let n_cols = self.n_cols().unwrap_or(0);
        let sym = self.symmetric().unwrap_or(false);
        Some(format!(
            ", {} row(s) × {} col(s){}",
            n_rows,
            n_cols,
            if sym { ", symmetric" } else { "" }
        ))
    }
});

/// One row or column DOF of an aggregate [`Matrix`], in materialized form.
pub type NamedDof = (NodeId, String);

impl Matrix {
    /// Aggregate is symmetric iff every block is. Vacuously true for an
    /// empty aggregate.
    pub fn symmetric(&self) -> Result<bool> {
        for h in self {
            if !with(h, |s| s.symmetric())? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Union of all row DOFs across blocks, in first-seen order.
    pub fn row_dofs(&self) -> Result<Vec<NamedDof>> {
        let mut out: Vec<NamedDof> = Vec::new();
        for h in self {
            with(h, |s| {
                for dof in s.row_dofs() {
                    let name = s.field_name(dof.field_idx).to_string();
                    let pair = (dof.node_id, name);
                    if !out.contains(&pair) {
                        out.push(pair);
                    }
                }
            })?;
        }
        Ok(out)
    }

    /// Union of all column DOFs across blocks, in first-seen order.
    pub fn col_dofs(&self) -> Result<Vec<NamedDof>> {
        let mut out: Vec<NamedDof> = Vec::new();
        for h in self {
            with(h, |s| {
                for dof in s.col_dofs() {
                    let name = s.field_name(dof.field_idx).to_string();
                    let pair = (dof.node_id, name);
                    if !out.contains(&pair) {
                        out.push(pair);
                    }
                }
            })?;
        }
        Ok(out)
    }

    /// Union of all field names across blocks, first-seen order.
    pub fn field_names(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for h in self {
            with(h, |s| {
                for name in s.field_names() {
                    if !out.contains(name) {
                        out.push(name.clone());
                    }
                }
            })?;
        }
        Ok(out)
    }

    /// Number of distinct row DOFs (union across blocks).
    pub fn n_rows(&self) -> Result<usize> {
        Ok(self.row_dofs()?.len())
    }

    /// Number of distinct column DOFs (union across blocks).
    pub fn n_cols(&self) -> Result<usize> {
        Ok(self.col_dofs()?.len())
    }

    /// Total COO entries stored across all blocks (counting duplicates).
    pub fn entry_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for h in self {
            total += with(h, |s| s.entry_count())?;
        }
        Ok(total)
    }

    /// Sum of `(row, col)` contributions across every block. `0.0` if no
    /// block has an entry at this coordinate.
    pub fn get(
        &self,
        row_node: NodeId,
        row_field: &str,
        col_node: NodeId,
        col_field: &str,
    ) -> Result<f64> {
        let mut total = 0.0;
        for h in self {
            total += with(h, |s| s.get(row_node, row_field, col_node, col_field))?;
        }
        Ok(total)
    }

    /// All COO entries, in `(block_idx, sub-block_insertion)` order, with
    /// DOFs materialised as `(NodeId, field_name)` pairs.
    pub fn iter_entries(&self) -> Result<Vec<(NodeId, String, NodeId, String, f64)>> {
        let mut out = Vec::new();
        for h in self {
            with(h, |s| {
                for (r, c, v) in s.iter_entries() {
                    out.push((
                        r.node_id,
                        s.field_name(r.field_idx).to_string(),
                        c.node_id,
                        s.field_name(c.field_idx).to_string(),
                        v,
                    ));
                }
            })?;
        }
        Ok(out)
    }

    /// Materialise as a [`nalgebra::DMatrix<f64>`] of size
    /// `n_rows × n_cols` (union DOFs in first-seen order). Block
    /// contributions are summed at coincident `(row, col)`.
    pub fn to_dmatrix(&self) -> Result<DMatrix<f64>> {
        let row_dofs = self.row_dofs()?;
        let col_dofs = self.col_dofs()?;
        let mut out = DMatrix::<f64>::zeros(row_dofs.len(), col_dofs.len());
        for h in self {
            with(h, |s| {
                for (rdof, cdof, v) in s.iter_entries() {
                    let r_name = s.field_name(rdof.field_idx);
                    let c_name = s.field_name(cdof.field_idx);
                    let i = row_dofs
                        .iter()
                        .position(|(n, nm)| *n == rdof.node_id && nm == r_name)
                        .expect("row dof must be in union");
                    let j = col_dofs
                        .iter()
                        .position(|(n, nm)| *n == cdof.node_id && nm == c_name)
                        .expect("col dof must be in union");
                    out[(i, j)] += v;
                }
            })?;
        }
        Ok(out)
    }

    /// Row-major dense buffer of length `n_rows × n_cols`.
    pub fn dense(&self) -> Result<Vec<f64>> {
        let m = self.to_dmatrix()?;
        let mut out = Vec::with_capacity(m.nrows() * m.ncols());
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out.push(m[(i, j)]);
            }
        }
        Ok(out)
    }

    /// Aggregate COO with rows / cols in the union order. Duplicates at
    /// the same `(row, col)` are preserved (one triplet per block contribution).
    pub fn to_coo(&self) -> Result<CooMatrix<f64>> {
        let row_dofs = self.row_dofs()?;
        let col_dofs = self.col_dofs()?;
        let mut coo = CooMatrix::<f64>::new(row_dofs.len(), col_dofs.len());
        for h in self {
            with(h, |s| {
                for (rdof, cdof, v) in s.iter_entries() {
                    let r_name = s.field_name(rdof.field_idx);
                    let c_name = s.field_name(cdof.field_idx);
                    let i = row_dofs
                        .iter()
                        .position(|(n, nm)| *n == rdof.node_id && nm == r_name)
                        .expect("row dof must be in union");
                    let j = col_dofs
                        .iter()
                        .position(|(n, nm)| *n == cdof.node_id && nm == c_name)
                        .expect("col dof must be in union");
                    coo.push(i, j, v);
                }
            })?;
        }
        Ok(coo)
    }

    /// Aggregate CSR view (sums duplicates as part of the conversion).
    pub fn to_csr(&self) -> Result<CsrMatrix<f64>> {
        Ok(CsrMatrix::from(&self.to_coo()?))
    }

    /// Aggregate CSC view (sums duplicates as part of the conversion).
    pub fn to_csc(&self) -> Result<CscMatrix<f64>> {
        Ok(CscMatrix::from(&self.to_coo()?))
    }

    /// `y = A · x` (dense). `x` is read in the union column order.
    pub fn mul_dense(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n_cols = self.n_cols()?;
        if x.len() != n_cols {
            return Err(PyrucastError::Message(format!(
                "mul_dense: x has length {} but matrix has {} columns",
                x.len(),
                n_cols
            )));
        }
        let csr = self.to_csr()?;
        let x_vec = DVector::<f64>::from_column_slice(x);
        let y_vec: DVector<f64> = &csr * &x_vec;
        Ok(y_vec.iter().copied().collect())
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::store::insert;

    fn nid(i: u32) -> NodeId {
        NodeId(i)
    }

    // ── SubMatrix tests ─────────────────────────────────────────────────────

    #[test]
    fn empty_sub_matrix() {
        let m = SubMatrix::new(false);
        assert_eq!(m.n_rows(), 0);
        assert_eq!(m.n_cols(), 0);
        assert_eq!(m.entry_count(), 0);
        assert!(!m.symmetric());
    }

    #[test]
    fn symmetric_flag_round_trip() {
        let m = SubMatrix::new(true);
        assert!(m.symmetric());
    }

    #[test]
    fn add_entry_interns_fields_and_dofs() {
        let mut m = SubMatrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(0), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        assert_eq!(m.field_names().len(), 2);
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.entry_count(), 4);
    }

    #[test]
    fn get_unknown_returns_zero() {
        let m = SubMatrix::new(false);
        assert_eq!(m.get(nid(0), "x", nid(0), "y"), 0.0);
    }

    #[test]
    fn get_sums_duplicates() {
        let mut m = SubMatrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(0), "T", 1.5);
        m.add_entry(nid(0), "q", nid(0), "T", -0.5);
        assert_eq!(m.get(nid(0), "q", nid(0), "T"), 3.0);
    }

    #[test]
    fn dense_matches_get() {
        let mut m = SubMatrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(0), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        let d = m.dense();
        assert_eq!(d, vec![2.0, -1.0, -1.0, 2.0]);
    }

    #[test]
    fn mul_dense_against_known_matrix() {
        let mut m = SubMatrix::new(true);
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
        let mut m = SubMatrix::new(false);
        m.add_entry(nid(0), "q", nid(0), "T", 1.0);
        assert!(m.mul_dense(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn rectangular_sub_matrix_distinct_row_and_col_dofs() {
        let mut c = SubMatrix::new(false);
        c.add_entry(nid(100), "T", nid(3), "T", 1.0);
        c.add_entry(nid(101), "T", nid(7), "T", 1.0);
        assert_eq!(c.n_rows(), 2);
        assert_eq!(c.n_cols(), 2);
        assert_eq!(c.field_names().len(), 1);
    }

    #[test]
    fn sub_iter_entries_preserves_insertion_order() {
        let mut m = SubMatrix::new(false);
        m.add_entry(nid(0), "a", nid(0), "b", 1.0);
        m.add_entry(nid(1), "a", nid(1), "b", 2.0);
        m.add_entry(nid(0), "a", nid(0), "b", 3.0);
        let entries: Vec<_> = m.iter_entries().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].2, 1.0);
        assert_eq!(entries[1].2, 2.0);
        assert_eq!(entries[2].2, 3.0);
    }

    #[test]
    fn sub_round_trip_serde() {
        let mut m = SubMatrix::new(true);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        m.add_entry(nid(0), "q", nid(1), "T", -1.0);
        m.add_entry(nid(1), "q", nid(1), "T", 2.0);
        use crate::persist::Persist;
        let bytes = m.to_bytes().unwrap();
        let m2 = SubMatrix::from_bytes(&bytes).unwrap();
        assert_eq!(m2.n_rows(), 2);
        assert_eq!(m2.n_cols(), 2);
        assert!(m2.symmetric());
        assert_eq!(m2.get(nid(0), "q", nid(0), "T"), 2.0);
    }

    #[test]
    fn sub_debug_and_display() {
        let mut m = SubMatrix::new(true);
        m.add_entry(nid(0), "q", nid(0), "T", 2.0);
        let d = format!("{:?}", m);
        assert!(d.contains("SubMatrix"));
        assert!(d.contains("n_rows"));
        assert!(d.contains("symmetric"));
        let s = format!("{}", m);
        assert!(s.contains("SubMatrix"));
        assert!(s.contains("1 row"));
        assert!(s.contains("symmetric"));
    }

    // ── Matrix (aggregate) tests ────────────────────────────────────────────

    #[test]
    fn empty_aggregate_is_vacuous_symmetric() {
        let m = Matrix::empty();
        assert_eq!(m.n_rows().unwrap(), 0);
        assert_eq!(m.n_cols().unwrap(), 0);
        assert!(m.symmetric().unwrap());
        assert_eq!(m.entry_count().unwrap(), 0);
    }

    #[test]
    fn aggregate_unions_dofs_and_sums_at_coincidence() {
        let mut a = SubMatrix::new(true);
        a.add_entry(nid(0), "q", nid(0), "T", 2.0);
        a.add_entry(nid(0), "q", nid(1), "T", -1.0);

        let mut b = SubMatrix::new(true);
        b.add_entry(nid(0), "q", nid(0), "T", 0.5); // coincident with a
        b.add_entry(nid(1), "q", nid(0), "T", -1.0);
        b.add_entry(nid(1), "q", nid(1), "T", 2.0);

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        assert_eq!(k.n_rows().unwrap(), 2);
        assert_eq!(k.n_cols().unwrap(), 2);
        assert!(k.symmetric().unwrap());
        assert_eq!(k.get(nid(0), "q", nid(0), "T").unwrap(), 2.5);
        assert_eq!(k.get(nid(0), "q", nid(1), "T").unwrap(), -1.0);
        assert_eq!(k.get(nid(1), "q", nid(0), "T").unwrap(), -1.0);
        assert_eq!(k.get(nid(1), "q", nid(1), "T").unwrap(), 2.0);
    }

    #[test]
    fn aggregate_symmetric_is_and_of_subs() {
        let a = SubMatrix::new(true);
        let b = SubMatrix::new(false);
        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();
        assert!(!k.symmetric().unwrap());
    }

    #[test]
    fn aggregate_to_dmatrix_layout_is_union_first_seen() {
        let mut a = SubMatrix::new(false);
        a.add_entry(nid(0), "q", nid(0), "T", 2.0);
        let mut b = SubMatrix::new(false);
        b.add_entry(nid(1), "q", nid(1), "T", 3.0);

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        let d = k.to_dmatrix().unwrap();
        assert_eq!(d.nrows(), 2);
        assert_eq!(d.ncols(), 2);
        assert_eq!(d[(0, 0)], 2.0);
        assert_eq!(d[(0, 1)], 0.0);
        assert_eq!(d[(1, 0)], 0.0);
        assert_eq!(d[(1, 1)], 3.0);
    }

    #[test]
    fn aggregate_mul_dense_matches_dense() {
        let mut a = SubMatrix::new(true);
        a.add_entry(nid(0), "q", nid(0), "T", 2.0);
        a.add_entry(nid(0), "q", nid(1), "T", -1.0);
        let mut b = SubMatrix::new(true);
        b.add_entry(nid(1), "q", nid(0), "T", -1.0);
        b.add_entry(nid(1), "q", nid(1), "T", 2.0);

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        assert_eq!(k.mul_dense(&[1.0, 1.0]).unwrap(), vec![1.0, 1.0]);
        assert_eq!(k.mul_dense(&[1.0, 2.0]).unwrap(), vec![0.0, 3.0]);
    }

    #[test]
    fn aggregate_entries_concatenates_blocks() {
        let mut a = SubMatrix::new(false);
        a.add_entry(nid(0), "q", nid(0), "T", 1.0);
        let mut b = SubMatrix::new(false);
        b.add_entry(nid(1), "q", nid(1), "T", 2.0);

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        let entries = k.iter_entries().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].4, 1.0);
        assert_eq!(entries[1].4, 2.0);
    }

    #[test]
    fn aggregate_debug_and_display() {
        let mut a = SubMatrix::new(true);
        a.add_entry(nid(0), "q", nid(0), "T", 2.0);
        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        let d = format!("{:?}", k);
        assert!(d.contains("Matrix"));
        let s = format!("{}", k);
        assert!(s.contains("Matrix"));
        assert!(s.contains("1 row"));
        assert!(s.contains("symmetric"));
    }
}
