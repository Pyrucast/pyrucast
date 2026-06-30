//! Sparse matrix indexed by **named DOFs** `(NodeId, field_name)`.
//!
//! Hierarchy:
//!
//! - [`SubMatrix`] — one COO block.  Structure is fully declared at
//!   construction time via two POI1 [`SubMesh`] handles (row/col node
//!   sets), two variable-name lists (dual / primal), and a
//!   [`DofOrdering`] that maps `(node_local_idx, var_idx)` to a flat
//!   matrix-row or -column index.  The actual non-zeros are stored in a
//!   [`nalgebra_sparse::CooMatrix`]; [`SubMatrix::add_entry`] appends
//!   triplets that accumulate on `get` / densification.
//! - [`Matrix`] — aggregate of [`SubMatrix`] blocks (one
//!   `Vec<Handle<SubMatrix>>`), produced by
//!   [`crate::containers::model::Model`] assembly (one or several blocks
//!   per sub-model).  Read-only: every accessor unions the blocks on the
//!   fly.
//!
//! A `symmetric: bool` flag lives on each [`SubMatrix`]. The aggregate
//! [`Matrix`] is reported symmetric iff every one of its blocks is. The
//! flag is **informative only**: storage is never de-duplicated.
//!
//! # DOF layout
//!
//! Given a [`SubMatrix`] with `n_rn` row-support nodes, `n_dv` dual
//! variables, and [`DofOrdering::NodesThenVars`]:
//!
//! ```text
//! row index i = node_local * n_dv + var_idx
//! ```
//!
//! With [`DofOrdering::VarsThenNodes`]:
//!
//! ```text
//! row index i = var_idx * n_rn + node_local
//! ```
//!
//! The same formula applies symmetrically to columns with `n_cn` (col
//! nodes) and `n_pv` (primal variables).
//!
//! # Example — single block
//!
//! ```
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::containers::matrix::{SubMatrix, DofOrdering};
//! use pyrucast::store::insert;
//!
//! let coords = insert(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
//! sm.add_cell(&[a.id()]).unwrap();
//! sm.add_cell(&[b.id()]).unwrap();
//! let support = insert(sm);
//!
//! let mut k = SubMatrix::new(
//!     support.clone(), support.clone(),
//!     vec!["q".into()], vec!["T".into()],
//!     DofOrdering::NodesThenVars, true,
//! ).unwrap();
//! k.add_entry(a.id(), "q", a.id(), "T",  2.0).unwrap();
//! k.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
//! k.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
//! k.add_entry(b.id(), "q", b.id(), "T",  2.0).unwrap();
//!
//! assert_eq!(k.n_rows(), 2);
//! assert_eq!(k.n_cols(), 2);
//! assert!(k.symmetric());
//! assert_eq!(k.get(a.id(), "q", a.id(), "T"), 2.0);
//! ```

use crate::aggregate::Aggregate;
use crate::containers::mesh::Coords;
use crate::containers::mesh::NodeId;
use crate::containers::mesh::SubMesh;
use crate::error::{PyrucastError, Result};
use crate::store::{read, Handle};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A single COO entry with DOFs materialised as `(NodeId, var_name)` pairs:
/// `(row_node, row_var, col_node, col_var, value)`.
pub type MatrixEntry = (NodeId, String, NodeId, String, f64);

// ─── DofOrdering ───────────────────────────────────────────────────────────

/// How `(node_local_idx, var_idx)` maps to a flat matrix-row or -column
/// index inside a [`SubMatrix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DofOrdering {
    /// `index = var_idx * n_nodes + node_local`
    ///
    /// All nodes for variable 0, then all nodes for variable 1, …
    VarsThenNodes,
    /// `index = node_local * n_vars + var_idx`
    ///
    /// All variables for node 0, then all variables for node 1, …
    NodesThenVars,
}

impl DofOrdering {
    /// `(node_local, var_idx)` → flat matrix index.
    pub fn to_index(
        self,
        node_local: usize,
        var_idx: usize,
        n_nodes: usize,
        n_vars: usize,
    ) -> usize {
        match self {
            DofOrdering::VarsThenNodes => var_idx * n_nodes + node_local,
            DofOrdering::NodesThenVars => node_local * n_vars + var_idx,
        }
    }

    /// flat matrix index → `(node_local, var_idx)`.
    pub fn from_index(self, idx: usize, n_nodes: usize, n_vars: usize) -> (usize, usize) {
        match self {
            DofOrdering::VarsThenNodes => (idx % n_nodes, idx / n_nodes),
            DofOrdering::NodesThenVars => (idx / n_vars, idx % n_vars),
        }
    }
}

impl crate::dump::Dump for DofOrdering {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        format!("{self:?}")
    }
}

// ─── SubMatrix ─────────────────────────────────────────────────────────────

/// One sparse COO block whose DOF layout is fully described by two POI1
/// sub-meshes and variable-name lists.
///
/// The **row DOF** at matrix index `i` is
/// `(row_nodes[node_local], dual_vars[var_idx])` where
/// `(node_local, var_idx) = ordering.from_index(i, n_row_nodes, n_dual_vars)`.
/// Columns are symmetric with `col_nodes` and `primal_vars`.
#[derive(Serialize, Deserialize)]
pub struct SubMatrix {
    /// POI1 mesh: cell `k` holds the k-th row-support node.
    row_support: Handle<SubMesh>,
    /// POI1 mesh: cell `k` holds the k-th col-support node.
    col_support: Handle<SubMesh>,
    /// Snapshot of `row_support` connectivity (one NodeId per cell).
    row_nodes: Vec<NodeId>,
    /// Snapshot of `col_support` connectivity (one NodeId per cell).
    col_nodes: Vec<NodeId>,
    /// Row variable names (dual variables).
    dual_vars: Vec<String>,
    /// Column variable names (primal variables).
    primal_vars: Vec<String>,
    /// `(node_local, var_idx)` ↔ matrix index mapping.
    ordering: DofOrdering,
    /// COO data, sized `(n_row_nodes × n_dual_vars) × (n_col_nodes × n_primal_vars)`.
    #[serde(with = "coo_serde")]
    coo: CooMatrix<f64>,
    symmetric: bool,
    /// `NodeId → local position` for O(1) `add_entry`, derived from
    /// `row_nodes` / `col_nodes`. Not serialized; built lazily on first use
    /// (the support is fixed at construction).
    #[serde(skip)]
    row_index: HashMap<NodeId, u32>,
    #[serde(skip)]
    col_index: HashMap<NodeId, u32>,
}

impl SubMatrix {
    /// Build a new block.
    ///
    /// `row_support` / `col_support` must be POI1 sub-meshes whose cells
    /// define the row/col node sequence. `dual_vars` / `primal_vars` are
    /// the row/column variable names; they must be non-empty.
    pub fn new(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetric: bool,
    ) -> Result<Self> {
        let row_nodes: Vec<NodeId> = read(&row_support)?.connectivity().to_vec();
        let col_nodes: Vec<NodeId> = read(&col_support)?.connectivity().to_vec();
        let nrows = row_nodes.len() * dual_vars.len();
        let ncols = col_nodes.len() * primal_vars.len();
        Ok(Self {
            row_support,
            col_support,
            row_nodes,
            col_nodes,
            dual_vars,
            primal_vars,
            ordering,
            coo: CooMatrix::new(nrows, ncols),
            symmetric,
            row_index: HashMap::new(),
            col_index: HashMap::new(),
        })
    }

    /// Whether the assembler declared this block numerically symmetric.
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }

    /// Number of row DOFs = `n_row_nodes × n_dual_vars`.
    pub fn n_rows(&self) -> usize {
        self.coo.nrows()
    }

    /// Number of column DOFs = `n_col_nodes × n_primal_vars`.
    pub fn n_cols(&self) -> usize {
        self.coo.ncols()
    }

    /// Number of COO triplets stored (counting duplicates at the same
    /// `(row, col)`).
    pub fn entry_count(&self) -> usize {
        self.coo.nnz()
    }

    /// Row variable names (dual variables).
    pub fn dual_vars(&self) -> &[String] {
        &self.dual_vars
    }

    /// Column variable names (primal variables).
    pub fn primal_vars(&self) -> &[String] {
        &self.primal_vars
    }

    /// DOF ordering strategy.
    pub fn ordering(&self) -> DofOrdering {
        self.ordering
    }

    /// Handle to the row-support POI1 sub-mesh.
    pub fn row_support(&self) -> &Handle<SubMesh> {
        &self.row_support
    }

    /// Handle to the col-support POI1 sub-mesh.
    pub fn col_support(&self) -> &Handle<SubMesh> {
        &self.col_support
    }

    /// Union of `dual_vars` and `primal_vars`, in that order, without
    /// duplicates.
    pub fn field_names(&self) -> Vec<String> {
        let mut out = self.dual_vars.clone();
        for pv in &self.primal_vars {
            if !out.contains(pv) {
                out.push(pv.clone());
            }
        }
        out
    }

    /// All row DOFs in matrix-row order: `(NodeId, var_name)` for each
    /// row index `0..n_rows`.
    pub fn row_dofs(&self) -> Vec<(NodeId, String)> {
        let n_nodes = self.row_nodes.len();
        let n_vars = self.dual_vars.len();
        (0..self.coo.nrows())
            .map(|i| {
                let (nl, vi) = self.ordering.from_index(i, n_nodes, n_vars);
                (self.row_nodes[nl], self.dual_vars[vi].clone())
            })
            .collect()
    }

    /// All column DOFs in matrix-column order: `(NodeId, var_name)` for
    /// each column index `0..n_cols`.
    pub fn col_dofs(&self) -> Vec<(NodeId, String)> {
        let n_nodes = self.col_nodes.len();
        let n_vars = self.primal_vars.len();
        (0..self.coo.ncols())
            .map(|i| {
                let (nl, vi) = self.ordering.from_index(i, n_nodes, n_vars);
                (self.col_nodes[nl], self.primal_vars[vi].clone())
            })
            .collect()
    }

    /// Append an entry at `(row_node, row_var) × (col_node, col_var)`.
    ///
    /// Returns an error if either `row_node` / `col_node` is not in its
    /// respective support, or if the variable name is not in the
    /// `dual_vars` / `primal_vars` list.  Repeated calls at the same
    /// `(row, col)` accumulate.
    pub fn add_entry(
        &mut self,
        row_node: NodeId,
        row_var: &str,
        col_node: NodeId,
        col_var: &str,
        value: f64,
    ) -> Result<()> {
        let n_rn = self.row_nodes.len();
        let n_dv = self.dual_vars.len();
        let n_cn = self.col_nodes.len();
        let n_pv = self.primal_vars.len();

        // O(1) node → local position (maps built lazily; support is fixed).
        self.ensure_node_indices();
        let rnl = *self.row_index.get(&row_node).ok_or_else(|| {
            PyrucastError::Message(format!("add_entry: row node {row_node:?} not in row_support"))
        })? as usize;
        let rvi = self
            .dual_vars
            .iter()
            .position(|v| v == row_var)
            .ok_or_else(|| {
                PyrucastError::Message(format!("add_entry: row var '{row_var}' not in dual_vars"))
            })?;
        let cnl = *self.col_index.get(&col_node).ok_or_else(|| {
            PyrucastError::Message(format!("add_entry: col node {col_node:?} not in col_support"))
        })? as usize;
        let cvi = self
            .primal_vars
            .iter()
            .position(|v| v == col_var)
            .ok_or_else(|| {
                PyrucastError::Message(format!("add_entry: col var '{col_var}' not in primal_vars"))
            })?;

        let ri = self.ordering.to_index(rnl, rvi, n_rn, n_dv);
        let ci = self.ordering.to_index(cnl, cvi, n_cn, n_pv);
        self.coo.push(ri, ci, value);
        Ok(())
    }

    /// Build the `NodeId → local position` maps from the support node lists, on
    /// first use (idempotent). First occurrence wins, matching the previous
    /// `position` lookup.
    fn ensure_node_indices(&mut self) {
        if self.row_index.is_empty() && !self.row_nodes.is_empty() {
            self.row_index.reserve(self.row_nodes.len());
            for (i, &n) in self.row_nodes.iter().enumerate() {
                self.row_index.entry(n).or_insert(i as u32);
            }
        }
        if self.col_index.is_empty() && !self.col_nodes.is_empty() {
            self.col_index.reserve(self.col_nodes.len());
            for (i, &n) in self.col_nodes.iter().enumerate() {
                self.col_index.entry(n).or_insert(i as u32);
            }
        }
    }

    /// COO entries in **local** index form `(row, col, value)` — the block's own
    /// numbering. Used by the aggregate to scatter into the global matrix via a
    /// per-block translation table.
    pub fn local_triplets(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.coo.triplet_iter().map(|(r, c, &v)| (r, c, v))
    }

    /// Handle to the `Coords` backing this block's row support (the col support
    /// shares it in any assembled system).
    pub fn coords(&self) -> Result<Handle<Coords>> {
        Ok(read(&self.row_support)?.coords())
    }

    /// Sum of all entries at `(row_node, row_var) × (col_node, col_var)`.
    /// Returns `0.0` if the DOF pair is unknown or has no entry.
    pub fn get(&self, row_node: NodeId, row_var: &str, col_node: NodeId, col_var: &str) -> f64 {
        let n_rn = self.row_nodes.len();
        let n_dv = self.dual_vars.len();
        let n_cn = self.col_nodes.len();
        let n_pv = self.primal_vars.len();

        let rnl = match self.row_nodes.iter().position(|&n| n == row_node) {
            Some(i) => i,
            None => return 0.0,
        };
        let rvi = match self.dual_vars.iter().position(|v| v == row_var) {
            Some(i) => i,
            None => return 0.0,
        };
        let cnl = match self.col_nodes.iter().position(|&n| n == col_node) {
            Some(i) => i,
            None => return 0.0,
        };
        let cvi = match self.primal_vars.iter().position(|v| v == col_var) {
            Some(i) => i,
            None => return 0.0,
        };

        let ri = self.ordering.to_index(rnl, rvi, n_rn, n_dv);
        let ci = self.ordering.to_index(cnl, cvi, n_cn, n_pv);

        self.coo
            .row_indices()
            .iter()
            .zip(self.coo.col_indices())
            .zip(self.coo.values())
            .filter(|&((&r, &c), _)| r == ri && c == ci)
            .map(|(_, &v)| v)
            .sum()
    }

    /// All COO triplets, in insertion order, with DOFs materialised as
    /// `(NodeId, var_name)` pairs.
    pub fn iter_entries(&self) -> Vec<MatrixEntry> {
        let n_rn = self.row_nodes.len();
        let n_dv = self.dual_vars.len();
        let n_cn = self.col_nodes.len();
        let n_pv = self.primal_vars.len();

        self.coo
            .row_indices()
            .iter()
            .zip(self.coo.col_indices())
            .zip(self.coo.values())
            .map(|((&ri, &ci), &v)| {
                let (rnl, rvi) = self.ordering.from_index(ri, n_rn, n_dv);
                let (cnl, cvi) = self.ordering.from_index(ci, n_cn, n_pv);
                (
                    self.row_nodes[rnl],
                    self.dual_vars[rvi].clone(),
                    self.col_nodes[cnl],
                    self.primal_vars[cvi].clone(),
                    v,
                )
            })
            .collect()
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

    /// Materialise as a [`nalgebra::DMatrix<f64>`]. Entries at the same
    /// `(row, col)` are summed.
    pub fn to_dmatrix(&self) -> DMatrix<f64> {
        let nr = self.coo.nrows();
        let nc = self.coo.ncols();
        let mut out = DMatrix::<f64>::zeros(nr, nc);
        for ((&r, &c), &v) in self
            .coo
            .row_indices()
            .iter()
            .zip(self.coo.col_indices())
            .zip(self.coo.values())
        {
            out[(r, c)] += v;
        }
        out
    }

    /// The internal COO matrix (cloned).
    pub fn to_coo(&self) -> CooMatrix<f64> {
        self.coo.clone()
    }

    /// Convert this block to a [`nalgebra_sparse::CsrMatrix`].
    pub fn to_csr(&self) -> CsrMatrix<f64> {
        CsrMatrix::from(&self.coo)
    }

    /// Convert this block to a [`nalgebra_sparse::CscMatrix`].
    pub fn to_csc(&self) -> CscMatrix<f64> {
        CscMatrix::from(&self.coo)
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
}

impl fmt::Debug for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubMatrix")
            .field("n_rows", &self.coo.nrows())
            .field("n_cols", &self.coo.ncols())
            .field("entries", &self.coo.nnz())
            .field("symmetric", &self.symmetric)
            .field("dual_vars", &self.dual_vars)
            .field("primal_vars", &self.primal_vars)
            .field("ordering", &self.ordering)
            .finish()
    }
}

impl fmt::Display for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nr = self.coo.nrows();
        write!(
            f,
            "SubMatrix: {} row(s) × {} col(s), {} entries{}",
            nr,
            self.coo.ncols(),
            self.coo.nnz(),
            if self.symmetric { ", symmetric" } else { "" }
        )
    }
}

/// Format a DOF `(node, var)` pair as the grid label `(node,var)`.
fn dof_label((n, v): &NamedDof) -> String {
    format!("({n},{v})")
}

impl crate::dump::Dump for SubMatrix {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        let row_labels: Vec<String> = self.row_dofs().iter().map(dof_label).collect();
        let col_labels: Vec<String> = self.col_dofs().iter().map(dof_label).collect();
        let data = self.dense();
        format!(
            "{self}\n{}",
            crate::dump::labeled_grid(&row_labels, &col_labels, &data, opts)
        )
    }
}

// ─── CooMatrix serde ───────────────────────────────────────────────────────

mod coo_serde {
    use nalgebra_sparse::CooMatrix;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct CooData {
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    }

    pub fn serialize<S: Serializer>(
        coo: &CooMatrix<f64>,
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        CooData {
            nrows: coo.nrows(),
            ncols: coo.ncols(),
            row_indices: coo.row_indices().to_vec(),
            col_indices: coo.col_indices().to_vec(),
            values: coo.values().to_vec(),
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<CooMatrix<f64>, D::Error> {
        let data = CooData::deserialize(d)?;
        let mut coo = CooMatrix::new(data.nrows, data.ncols);
        for ((r, c), v) in data
            .row_indices
            .into_iter()
            .zip(data.col_indices)
            .zip(data.values)
        {
            coo.push(r, c, v);
        }
        Ok(coo)
    }
}

// ─── Matrix (aggregate) ────────────────────────────────────────────────────

/// Snapshot produced by [`Matrix::finalize`]: DOF tables + assembled CSR.
struct AssembledData {
    row_dofs: Vec<NamedDof>,
    col_dofs: Vec<NamedDof>,
    csr: CsrMatrix<f64>,
}

/// Aggregate of [`SubMatrix`] blocks.
///
/// Call [`Matrix::finalize`] before passing to a solver. Solver-facing methods
/// (`to_csr`, `to_dmatrix`, `mul_dense`, `dense`, `to_coo`, `to_csc`) return
/// an error if the matrix has not been finalized. `add_sub` invalidates the
/// assembled state.
#[derive(Serialize, Deserialize, Default)]
pub struct Matrix {
    subs: Vec<Handle<SubMatrix>>,
    #[serde(skip)]
    assembled: Option<AssembledData>,
    /// Transparently cached factorization (e.g. the solver's sparse LU), reused
    /// across solves on the same matrix. Derived state: never serialized,
    /// type-erased so `containers` stays decoupled from the solver, and cleared
    /// whenever the matrix changes (`add_sub` → `post_push`). Interior mutability
    /// so `solve(&Matrix)` can fill it under a shared store read lock.
    #[serde(skip)]
    factorization: parking_lot::Mutex<Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>>,
}

crate::impl_aggregate!(Matrix, SubMatrix, sub_matrix, "sub-matrix(es)", {
    fn post_push(&mut self) {
        self.assembled = None;
        // The matrix content changed ⇒ any cached factorization is stale.
        *self.factorization.get_mut() = None;
    }
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

/// One row or column DOF of an aggregate [`Matrix`], in materialised form.
pub type NamedDof = (NodeId, String);

impl Matrix {
    /// Build the global DOF union + CSR. Must be called before any
    /// solver-facing method (`to_csr`, `to_dmatrix`, `mul_dense`, `dense`,
    /// `to_coo`, `to_csc`). Idempotent: a second call is a no-op if no
    /// `add_sub` has occurred since the last `finalize`.
    pub fn finalize(&mut self) -> Result<()> {
        if self.assembled.is_some() {
            return Ok(());
        }
        let row_dofs = self.collect_row_dofs()?;
        let col_dofs = self.collect_col_dofs()?;
        let coo = self.build_coo(&row_dofs, &col_dofs)?;
        let csr = CsrMatrix::from(&coo);
        self.assembled = Some(AssembledData {
            row_dofs,
            col_dofs,
            csr,
        });
        Ok(())
    }

    fn assembled_or_err(&self) -> Result<&AssembledData> {
        self.assembled.as_ref().ok_or_else(|| {
            PyrucastError::Message(
                "Matrix has not been finalized; call finalize() before solving".into(),
            )
        })
    }

    /// The cached factorization downcast to `T`, if one is present and of that
    /// type. Lets the solver reuse a previous factorization transparently.
    pub fn cached_factorization<T: std::any::Any + Send + Sync>(
        &self,
    ) -> Option<std::sync::Arc<T>> {
        let arc = self.factorization.lock().as_ref().cloned()?;
        arc.downcast::<T>().ok()
    }

    /// Store a freshly computed factorization for transparent reuse. Cleared
    /// automatically whenever the matrix changes (`add_sub`).
    pub fn store_factorization(&self, factorization: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        *self.factorization.lock() = Some(factorization);
    }

    // ── Block-traversal helpers (no finalize required) ──────────────────

    fn collect_row_dofs(&self) -> Result<Vec<NamedDof>> {
        self.collect_dofs(true)
    }

    fn collect_col_dofs(&self) -> Result<Vec<NamedDof>> {
        self.collect_dofs(false)
    }

    /// Deduplicated concatenation of the blocks' row (or col) DOFs — the global
    /// DOF list. O(total block DOFs) via a hash set (no quadratic `contains`).
    ///
    /// Order: **solver order** when the backing `Coords` carries a
    /// [`permutation`](crate::containers::mesh::Coords::permutation) (stable
    /// sort by the node's permutation index, so the per-node variable order is
    /// preserved); otherwise **first-seen** (identical to the historical
    /// behaviour, hence bit-for-bit stable when no permutation is set).
    fn collect_dofs(&self, row: bool) -> Result<Vec<NamedDof>> {
        let mut seen: std::collections::HashSet<NamedDof> = std::collections::HashSet::new();
        let mut out: Vec<NamedDof> = Vec::new();
        for h in self {
            let sub = read(h)?;
            let dofs = if row { sub.row_dofs() } else { sub.col_dofs() };
            for d in dofs {
                if seen.insert(d.clone()) {
                    out.push(d);
                }
            }
        }
        if let Some(first) = self.iter().next() {
            let coords_h = read(first)?.coords()?;
            if let Some(perm) = read(&coords_h)?.permutation() {
                out.sort_by_key(|(n, _)| perm[n.0 as usize]);
            }
        }
        Ok(out)
    }

    /// Assemble the global COO from the blocks, mapping each block's **local**
    /// DOF indices to global ones via a per-block translation table (built once
    /// from the global DOF maps). O(total block DOFs + nnz) — no per-entry
    /// search.
    fn build_coo(&self, row_dofs: &[NamedDof], col_dofs: &[NamedDof]) -> Result<CooMatrix<f64>> {
        let row_map: HashMap<NamedDof, usize> =
            row_dofs.iter().cloned().enumerate().map(|(i, d)| (d, i)).collect();
        let col_map: HashMap<NamedDof, usize> =
            col_dofs.iter().cloned().enumerate().map(|(i, d)| (d, i)).collect();
        let mut coo = CooMatrix::<f64>::new(row_dofs.len(), col_dofs.len());
        for h in self {
            let sub = read(h)?;
            // local DOF index → global index (the "simple remap").
            let trow: Vec<usize> = sub.row_dofs().iter().map(|d| row_map[d]).collect();
            let tcol: Vec<usize> = sub.col_dofs().iter().map(|d| col_map[d]).collect();
            for (ri, ci, v) in sub.local_triplets() {
                coo.push(trow[ri], tcol[ci], v);
            }
        }
        Ok(coo)
    }

    // ── Inspection (always available) ───────────────────────────────────

    /// Aggregate is symmetric iff every block is. Vacuously true for an
    /// empty aggregate.
    pub fn symmetric(&self) -> Result<bool> {
        for h in self {
            if !read(h)?.symmetric() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Union of all row DOFs across blocks, in first-seen order.
    /// If finalized, returns the cached order (consistent with the CSR).
    pub fn row_dofs(&self) -> Result<Vec<NamedDof>> {
        if let Some(a) = &self.assembled {
            return Ok(a.row_dofs.clone());
        }
        self.collect_row_dofs()
    }

    /// Union of all column DOFs across blocks, in first-seen order.
    /// If finalized, returns the cached order (consistent with the CSR).
    pub fn col_dofs(&self) -> Result<Vec<NamedDof>> {
        if let Some(a) = &self.assembled {
            return Ok(a.col_dofs.clone());
        }
        self.collect_col_dofs()
    }

    /// Union of all field names (dual + primal) across blocks.
    pub fn field_names(&self) -> Result<Vec<String>> {
        let mut out: Vec<String> = Vec::new();
        for h in self {
            for name in read(h)?.field_names() {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        Ok(out)
    }

    /// Number of distinct row DOFs.
    pub fn n_rows(&self) -> Result<usize> {
        Ok(self.row_dofs()?.len())
    }

    /// Number of distinct column DOFs.
    pub fn n_cols(&self) -> Result<usize> {
        Ok(self.col_dofs()?.len())
    }

    /// Total COO entries stored across all blocks (counting duplicates).
    pub fn entry_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for h in self {
            total += read(h)?.entry_count();
        }
        Ok(total)
    }

    /// Sum of contributions at `(row, col)` across every block.
    pub fn get(
        &self,
        row_node: NodeId,
        row_field: &str,
        col_node: NodeId,
        col_field: &str,
    ) -> Result<f64> {
        let mut total = 0.0;
        for h in self {
            total += read(h)?.get(row_node, row_field, col_node, col_field);
        }
        Ok(total)
    }

    /// All COO entries across every block, in block-insertion order.
    pub fn iter_entries(&self) -> Result<Vec<MatrixEntry>> {
        let mut out = Vec::new();
        for h in self {
            out.extend(read(h)?.iter_entries());
        }
        Ok(out)
    }

    // ── Solver-facing (require finalize) ────────────────────────────────

    /// Assembled CSR. Requires [`finalize`](Self::finalize).
    pub fn to_csr(&self) -> Result<&CsrMatrix<f64>> {
        Ok(&self.assembled_or_err()?.csr)
    }

    /// Assembled dense matrix. Requires [`finalize`](Self::finalize).
    pub fn to_dmatrix(&self) -> Result<DMatrix<f64>> {
        let a = self.assembled_or_err()?;
        let nr = a.row_dofs.len();
        let nc = a.col_dofs.len();
        let mut out = DMatrix::<f64>::zeros(nr, nc);
        for (r, c, &v) in a.csr.triplet_iter() {
            out[(r, c)] += v;
        }
        Ok(out)
    }

    /// Row-major dense buffer. Requires [`finalize`](Self::finalize).
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

    /// Assembled COO (rebuilt from the CSR). Requires [`finalize`](Self::finalize).
    pub fn to_coo(&self) -> Result<CooMatrix<f64>> {
        let a = self.assembled_or_err()?;
        let mut coo = CooMatrix::<f64>::new(a.row_dofs.len(), a.col_dofs.len());
        for (r, c, &v) in a.csr.triplet_iter() {
            coo.push(r, c, v);
        }
        Ok(coo)
    }

    /// Assembled CSC. Requires [`finalize`](Self::finalize).
    pub fn to_csc(&self) -> Result<CscMatrix<f64>> {
        Ok(CscMatrix::from(&self.assembled_or_err()?.csr))
    }

    /// `y = A · x` (dense). Requires [`finalize`](Self::finalize).
    pub fn mul_dense(&self, x: &[f64]) -> Result<Vec<f64>> {
        let a = self.assembled_or_err()?;
        let nc = a.col_dofs.len();
        if x.len() != nc {
            return Err(PyrucastError::Message(format!(
                "mul_dense: x has length {} but matrix has {} columns",
                x.len(),
                nc
            )));
        }
        let x_vec = DVector::<f64>::from_column_slice(x);
        let y_vec: DVector<f64> = &a.csr * &x_vec;
        Ok(y_vec.iter().copied().collect())
    }
}

impl crate::dump::Dump for Matrix {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        // Build the global labelled grid on the fly — `collect_*_dofs` and
        // `build_coo` take `&self`, so no `finalize()` (which needs `&mut`)
        // is required: a matrix dumps the same content whether assembled or
        // not.
        let grid = (|| -> Result<String> {
            let row_dofs = self.collect_row_dofs()?;
            let col_dofs = self.collect_col_dofs()?;
            let coo = self.build_coo(&row_dofs, &col_dofs)?;
            let nc = col_dofs.len();
            let mut data = vec![0.0f64; row_dofs.len() * nc];
            for ((&r, &c), &v) in coo
                .row_indices()
                .iter()
                .zip(coo.col_indices())
                .zip(coo.values())
            {
                data[r * nc + c] += v;
            }
            let row_labels: Vec<String> = row_dofs.iter().map(dof_label).collect();
            let col_labels: Vec<String> = col_dofs.iter().map(dof_label).collect();
            Ok(crate::dump::labeled_grid(
                &row_labels,
                &col_labels,
                &data,
                opts,
            ))
        })();
        match grid {
            Ok(g) => format!("{self}\n{g}"),
            Err(e) => format!("{self}\n<{e}>"),
        }
    }
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Coords;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    /// Build a POI1 SubMesh with `n` fresh nodes in a new 1-D Coords.
    /// Returns `(coords, nodes, support_handle)`.
    fn make_poi1(n: usize) -> (Handle<Coords>, Vec<Node>, Handle<SubMesh>) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for node in &nodes {
            sm.add_cell(&[node.id()]).unwrap();
        }
        (coords, nodes, insert(sm))
    }

    // ── SubMatrix tests ─────────────────────────────────────────────────────

    #[test]
    fn empty_sub_matrix() {
        let (_cfg, _nodes, sup) = make_poi1(2);
        let m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.entry_count(), 0);
        assert!(!m.symmetric());
    }

    #[test]
    fn symmetric_flag_round_trip() {
        let (_cfg, _nodes, sup) = make_poi1(1);
        let m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        assert!(m.symmetric());
    }

    #[test]
    fn add_entry_and_field_names() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", b, "T", -1.0).unwrap();
        m.add_entry(b, "q", a, "T", -1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();
        // dual_vars + primal_vars = 2 distinct names
        assert_eq!(m.field_names().len(), 2);
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.entry_count(), 4);
    }

    #[test]
    fn get_unknown_returns_zero() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        // unknown field → 0.0
        assert_eq!(m.get(nodes[0].id(), "x", nodes[0].id(), "y"), 0.0);
    }

    #[test]
    fn get_sums_duplicates() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", a, "T", 1.5).unwrap();
        m.add_entry(a, "q", a, "T", -0.5).unwrap();
        assert_eq!(m.get(a, "q", a, "T"), 3.0);
    }

    #[test]
    fn dense_matches_get() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", b, "T", -1.0).unwrap();
        m.add_entry(b, "q", a, "T", -1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();
        assert_eq!(m.dense(), vec![2.0, -1.0, -1.0, 2.0]);
    }

    #[test]
    fn dump_labels_grid_with_dofs() {
        use crate::dump::{Dump, DumpOptions};
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", b, "T", -1.0).unwrap();
        m.add_entry(b, "q", a, "T", -1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();

        let s = m.render(&DumpOptions::default());
        let mut lines = s.lines();
        assert_eq!(
            lines.next().unwrap(),
            "SubMatrix: 2 row(s) × 2 col(s), 4 entries"
        );
        // In-line DOF labels on both axes + values at default precision.
        assert!(s.contains(&format!("({a},q)")), "row label:\n{s}");
        assert!(s.contains(&format!("({a},T)")), "col label:\n{s}");
        assert!(s.contains("2.000") && s.contains("-1.000"), "values:\n{s}");
        assert_eq!(s.lines().count(), 4, "summary + header + 2 rows:\n{s}");
    }

    #[test]
    fn mul_dense_against_known_matrix() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", b, "T", -1.0).unwrap();
        m.add_entry(b, "q", a, "T", -1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();
        assert_eq!(m.mul_dense(&[1.0, 1.0]).unwrap(), vec![1.0, 1.0]);
        assert_eq!(m.mul_dense(&[1.0, 2.0]).unwrap(), vec![0.0, 3.0]);
    }

    #[test]
    fn mul_dense_rejects_wrong_size() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 1.0).unwrap();
        // 1 col, but x has 2 elements
        assert!(m.mul_dense(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn rectangular_sub_matrix_distinct_row_and_col_supports() {
        let (_cfg_r, row_nodes, row_sup) = make_poi1(2);
        let (_cfg_c, col_nodes, col_sup) = make_poi1(2);
        let mut c = SubMatrix::new(
            row_sup,
            col_sup,
            vec!["T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        c.add_entry(row_nodes[0].id(), "T", col_nodes[0].id(), "T", 1.0)
            .unwrap();
        c.add_entry(row_nodes[1].id(), "T", col_nodes[1].id(), "T", 1.0)
            .unwrap();
        assert_eq!(c.n_rows(), 2);
        assert_eq!(c.n_cols(), 2);
        // "T" appears in both dual and primal — field_names deduplicates
        assert_eq!(c.field_names().len(), 1);
    }

    #[test]
    fn iter_entries_preserves_insertion_order() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();
        m.add_entry(a, "q", a, "T", 3.0).unwrap();
        let entries = m.iter_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].4, 1.0);
        assert_eq!(entries[1].4, 2.0);
        assert_eq!(entries[2].4, 3.0);
    }

    #[test]
    fn sub_round_trip_serde() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        m.add_entry(a, "q", b, "T", -1.0).unwrap();
        m.add_entry(b, "q", b, "T", 2.0).unwrap();
        use crate::persist::Persist;
        let bytes = m.to_bytes().unwrap();
        let m2 = SubMatrix::from_bytes(&bytes).unwrap();
        assert_eq!(m2.n_rows(), 2);
        assert_eq!(m2.n_cols(), 2);
        assert!(m2.symmetric());
        assert_eq!(m2.get(a, "q", a, "T"), 2.0);
    }

    #[test]
    fn sub_debug_and_display() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        let d = format!("{:?}", m);
        assert!(d.contains("SubMatrix"));
        assert!(d.contains("n_rows"));
        assert!(d.contains("symmetric"));
        let s = format!("{}", m);
        assert!(s.contains("SubMatrix"));
        assert!(s.contains("1 row"));
        assert!(s.contains("symmetric"));
    }

    #[test]
    fn dof_ordering_vars_then_nodes() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        // 2 vars × 2 nodes = 4×4 block
        let (_cfg2, nodes2, sup2) = make_poi1(2);
        let (c, d) = (nodes2[0].id(), nodes2[1].id());
        let mut m = SubMatrix::new(
            sup,
            sup2,
            vec!["p".into(), "q".into()],
            vec!["u".into(), "v".into()],
            DofOrdering::VarsThenNodes,
            false,
        )
        .unwrap();
        // With VarsThenNodes: row 0 = (p, node_a), row 1 = (p, node_b),
        //                     row 2 = (q, node_a), row 3 = (q, node_b)
        m.add_entry(a, "p", c, "u", 1.0).unwrap();
        m.add_entry(b, "p", d, "v", 2.0).unwrap();
        m.add_entry(a, "q", c, "v", 3.0).unwrap();
        assert_eq!(m.get(a, "p", c, "u"), 1.0);
        assert_eq!(m.get(b, "p", d, "v"), 2.0);
        assert_eq!(m.get(a, "q", c, "v"), 3.0);
        assert_eq!(m.get(b, "q", d, "u"), 0.0);

        // row_dofs in VarsThenNodes order: (p,a),(p,b),(q,a),(q,b)
        let rdofs = m.row_dofs();
        assert_eq!(rdofs[0], (a, "p".to_string()));
        assert_eq!(rdofs[1], (b, "p".to_string()));
        assert_eq!(rdofs[2], (a, "q".to_string()));
        assert_eq!(rdofs[3], (b, "q".to_string()));
    }

    #[test]
    fn dof_ordering_nodes_then_vars() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let (_cfg2, nodes2, sup2) = make_poi1(2);
        let (c, _d) = (nodes2[0].id(), nodes2[1].id());
        let mut m = SubMatrix::new(
            sup,
            sup2,
            vec!["p".into(), "q".into()],
            vec!["u".into(), "v".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        // With NodesThenVars: row 0 = (node_a, p), row 1 = (node_a, q),
        //                     row 2 = (node_b, p), row 3 = (node_b, q)
        m.add_entry(a, "p", c, "u", 7.0).unwrap();
        assert_eq!(m.get(a, "p", c, "u"), 7.0);

        let rdofs = m.row_dofs();
        assert_eq!(rdofs[0], (a, "p".to_string()));
        assert_eq!(rdofs[1], (a, "q".to_string()));
        assert_eq!(rdofs[2], (b, "p".to_string()));
        assert_eq!(rdofs[3], (b, "q".to_string()));
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
        // Single configuration: 5 distinct nodes to avoid NodeId collisions.
        let (coords, nodes, _) = make_poi1(5);
        let (na, ca0, ca1, m0, m1) = (
            nodes[0].id(),
            nodes[1].id(),
            nodes[2].id(),
            nodes[3].id(),
            nodes[4].id(),
        );

        // Block a: 1 row (na) × 2 cols (ca0, ca1)
        let mut row_a = SubMesh::new(coords.clone(), ElementType::POI1);
        row_a.add_cell(&[na]).unwrap();
        let mut col_a = SubMesh::new(coords.clone(), ElementType::POI1);
        col_a.add_cell(&[ca0]).unwrap();
        col_a.add_cell(&[ca1]).unwrap();
        let mut a = SubMatrix::new(
            insert(row_a),
            insert(col_a),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        a.add_entry(na, "q", ca0, "T", 2.0).unwrap();
        a.add_entry(na, "q", ca1, "T", -1.0).unwrap();

        // Block b: 2 rows (m0, m1) × 2 cols (m0, m1)
        let mut row_b = SubMesh::new(coords.clone(), ElementType::POI1);
        row_b.add_cell(&[m0]).unwrap();
        row_b.add_cell(&[m1]).unwrap();
        let mut col_b = SubMesh::new(coords.clone(), ElementType::POI1);
        col_b.add_cell(&[m0]).unwrap();
        col_b.add_cell(&[m1]).unwrap();
        let mut b = SubMatrix::new(
            insert(row_b),
            insert(col_b),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        b.add_entry(m0, "q", m0, "T", 0.5).unwrap();
        b.add_entry(m1, "q", m0, "T", -1.0).unwrap();
        b.add_entry(m1, "q", m1, "T", 2.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        // Union of row DOFs: (na,"q"), (m0,"q"), (m1,"q") — 3
        // Union of col DOFs: (ca0,"T"), (ca1,"T"), (m0,"T"), (m1,"T") — 4
        assert_eq!(k.n_rows().unwrap(), 3);
        assert_eq!(k.n_cols().unwrap(), 4);
    }

    #[test]
    fn aggregate_symmetric_is_and_of_subs() {
        let (_cfg, _nodes, sup) = make_poi1(1);
        let a = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        let b = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();
        assert!(!k.symmetric().unwrap());
    }

    #[test]
    fn aggregate_to_dmatrix_layout_is_union_first_seen() {
        // Two distinct nodes in the SAME configuration to avoid NodeId collisions.
        let (coords, nodes, _) = make_poi1(2);
        let na = nodes[0].id();
        let nb = nodes[1].id();

        let mut sm_a = SubMesh::new(coords.clone(), ElementType::POI1);
        sm_a.add_cell(&[na]).unwrap();
        let sup_a = insert(sm_a);

        let mut sm_b = SubMesh::new(coords.clone(), ElementType::POI1);
        sm_b.add_cell(&[nb]).unwrap();
        let sup_b = insert(sm_b);

        let mut a = SubMatrix::new(
            sup_a.clone(),
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        a.add_entry(na, "q", na, "T", 2.0).unwrap();

        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        b.add_entry(nb, "q", nb, "T", 3.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();
        k.finalize().unwrap();

        let d = k.to_dmatrix().unwrap();
        assert_eq!(d.nrows(), 2);
        assert_eq!(d.ncols(), 2);
        assert_eq!(d[(0, 0)], 2.0);
        assert_eq!(d[(0, 1)], 0.0);
        assert_eq!(d[(1, 0)], 0.0);
        assert_eq!(d[(1, 1)], 3.0);
    }

    #[test]
    fn aggregate_dof_order_follows_permutation() {
        // Same data as the first-seen test: na→2.0, nb→3.0, but a permutation
        // orders nb's DOF before na's (solver order).
        let (coords, nodes, _) = make_poi1(2);
        let na = nodes[0].id();
        let nb = nodes[1].id();

        let mk = |nid| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nid]).unwrap();
            insert(sm)
        };
        let sup_a = mk(na);
        let sup_b = mk(nb);
        let mut a = SubMatrix::new(
            sup_a.clone(),
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        a.add_entry(na, "q", na, "T", 2.0).unwrap();
        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        b.add_entry(nb, "q", nb, "T", 3.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();

        // Transposition of the identity ⇒ nb sorts before na.
        let cap = read(&coords).unwrap().capacity();
        let mut perm: Vec<u32> = (0..cap as u32).collect();
        perm.swap(na.0 as usize, nb.0 as usize);
        crate::store::write(&coords).unwrap().set_permutation(perm).unwrap();

        k.finalize().unwrap();

        let rows = k.row_dofs().unwrap();
        assert_eq!(rows[0], (nb, "q".to_string()));
        assert_eq!(rows[1], (na, "q".to_string()));
        let d = k.to_dmatrix().unwrap();
        assert_eq!(d[(0, 0)], 3.0); // nb first
        assert_eq!(d[(1, 1)], 2.0); // na second
    }

    #[test]
    fn not_finalized_yields_error() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        m.add_entry(a, "q", a, "T", 1.0).unwrap();
        let mut k = Matrix::empty();
        k.add_sub(insert(m)).unwrap();
        // must not succeed before finalize
        assert!(k.to_csr().is_err());
        assert!(k.to_dmatrix().is_err());
        assert!(k.mul_dense(&[1.0]).is_err());
        k.finalize().unwrap();
        assert!(k.to_csr().is_ok());
    }

    #[test]
    fn aggregate_mul_dense_matches_dense() {
        let (_cfg_a, nodes_a, sup_a) = make_poi1(2);
        let (na0, na1) = (nodes_a[0].id(), nodes_a[1].id());
        // row block a: only node na0 as row
        let mut row_a = SubMesh::new(read(&sup_a).unwrap().coords(), ElementType::POI1);
        row_a.add_cell(&[na0]).unwrap();
        let row_a_h = insert(row_a);

        let mut a = SubMatrix::new(
            row_a_h,
            sup_a.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        a.add_entry(na0, "q", na0, "T", 2.0).unwrap();
        a.add_entry(na0, "q", na1, "T", -1.0).unwrap();

        let mut row_b = SubMesh::new(read(&sup_a).unwrap().coords(), ElementType::POI1);
        row_b.add_cell(&[na1]).unwrap();
        let row_b_h = insert(row_b);

        let mut b = SubMatrix::new(
            row_b_h,
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        b.add_entry(na1, "q", na0, "T", -1.0).unwrap();
        b.add_entry(na1, "q", na1, "T", 2.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(insert(a)).unwrap();
        k.add_sub(insert(b)).unwrap();
        k.finalize().unwrap();

        assert_eq!(k.mul_dense(&[1.0, 1.0]).unwrap(), vec![1.0, 1.0]);
        assert_eq!(k.mul_dense(&[1.0, 2.0]).unwrap(), vec![0.0, 3.0]);
    }

    #[test]
    fn aggregate_entries_concatenates_blocks() {
        let (_cfg_a, nodes_a, sup_a) = make_poi1(1);
        let (_cfg_b, nodes_b, sup_b) = make_poi1(1);
        let na = nodes_a[0].id();
        let nb = nodes_b[0].id();

        let mut a = SubMatrix::new(
            sup_a.clone(),
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        a.add_entry(na, "q", na, "T", 1.0).unwrap();

        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        b.add_entry(nb, "q", nb, "T", 2.0).unwrap();

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
        let (_cfg, nodes, sup) = make_poi1(1);
        let a_id = nodes[0].id();
        let mut a = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        a.add_entry(a_id, "q", a_id, "T", 2.0).unwrap();
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
