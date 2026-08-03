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
//! use pyrucast::coords::Coords;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::atoms::Node;
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
use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::Mesh;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::models::{MatrixKind, Physics};
use crate::parallel::*;
use crate::store::{insert, read, Handle};
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
/// The recipe a **computed** [`SubMatrix`] carries *instead of* stored values:
/// how to evaluate its contribution on the fly. The global assembler drives the
/// sub-model's [`element_matrix`](crate::models::SubModelKind::element_matrix) kernel
/// over `fespace`'s cells and scatters the result straight into the global
/// matrix — a computed block never materialises a COO (its own `coo` stays an
/// empty, correctly-sized placeholder, so structural queries still work).
#[derive(Clone, Serialize, Deserialize)]
pub struct ComputedRecipe {
    /// Sub-model whose element kernel produces the contribution.
    pub submodel: Handle<SubModel>,
    /// FE subspaces the kernel integrates over. Usually one; several (sharing one
    /// submesh, differing by quadrature) for a multi-quadrature element. The
    /// primary (index 0) drives the cell loop and the scatter numbering.
    pub fespaces: Vec<Handle<SubFiniteElementSpace>>,
    /// Material field for the kernel; `Some` iff the physics declares one.
    pub material: Option<Handle<SubElementField>>,
    /// Which element matrix this recipe produces — the discriminant the scatter
    /// dispatches on to pick the sub-model's kernel
    /// ([`SubModelKind::matrix_element`](crate::models::SubModelKind::matrix_element)).
    /// Defaults to [`MatrixKind::Stiffness`] for backward-compatible deserialization.
    #[serde(default)]
    pub kind: MatrixKind,
    /// Current stress / algorithmic-tangent field the kernel reads — `Some` only
    /// for the state-dependent kinds (geometric stiffness, consistent tangent).
    #[serde(default)]
    pub state: Option<Handle<SubElementField>>,
}

/// Default value of [`SubMatrix::factor`] for pre-existing serialized data
/// that predates the field.
fn default_factor() -> f64 {
    1.0
}

#[derive(Clone, Serialize, Deserialize)]
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
    /// `Some` ⇒ this is a **computed** block: `coo` is an empty placeholder and
    /// the contribution is produced on the fly by the global assembler from this
    /// recipe. `None` ⇒ **literal** block, `coo` holds the values (the historical
    /// behaviour, unchanged).
    #[serde(default)]
    recipe: Option<ComputedRecipe>,
    /// The set of [`Physics`] natures of the sub-model that produced this block,
    /// set by the assembler ([`crate::ops::assemble`]) for **both** the computed
    /// and the literal path (so a Dirichlet C/Cᵀ pair is tagged too). **Empty**
    /// for a block built directly, outside assembly (the « rien » case), or
    /// carrying several natures for a coupled physics. Consumed by
    /// [`Matrix::filter`](Matrix::filter).
    #[serde(default)]
    physics: Vec<Physics>,
    /// Lazy scalar scale applied to every value this block emits — at direct
    /// accessors (`get`, `dense`, …) and at global assembly (`build_global_triplets`,
    /// [`crate::ops::assemble::scatter`]) alike. Defaults to `1.0`; set via
    /// `Mul<f64>`/`Div<f64>` ([`std::ops::Mul`], [`std::ops::Div`]) rather than
    /// eagerly rewriting `coo`, so it works for a **computed** block too (its
    /// values don't exist until assembly evaluates the recipe). Never touches
    /// `local_coo_arrays`/`local_triplets`, which stay raw — every consumer of
    /// those applies the factor itself.
    #[serde(default = "default_factor")]
    factor: f64,
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
        // The block's row/col numbering snapshots these supports; freeze them.
        crate::containers::mesh::seal(&row_support)?;
        crate::containers::mesh::seal(&col_support)?;
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
            recipe: None,
            physics: Vec::new(),
            factor: 1.0,
            row_index: HashMap::new(),
            col_index: HashMap::new(),
        })
    }

    /// Build a **computed** block: a sized-but-empty placeholder that carries a
    /// [`ComputedRecipe`] instead of values. Its structure (supports, vars,
    /// ordering, dimensions) is fully defined; the values are produced by the
    /// global assembler, which drives `submodel`'s element kernel over
    /// `recipe.fespace` and scatters straight into the global matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn computed(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetric: bool,
        recipe: ComputedRecipe,
    ) -> Result<Self> {
        let row_nodes: Vec<NodeId> = read(&row_support)?.connectivity().to_vec();
        let col_nodes: Vec<NodeId> = read(&col_support)?.connectivity().to_vec();
        // The block's row/col numbering snapshots these supports; freeze them.
        crate::containers::mesh::seal(&row_support)?;
        crate::containers::mesh::seal(&col_support)?;
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
            recipe: Some(recipe),
            physics: Vec::new(),
            factor: 1.0,
            row_index: HashMap::new(),
            col_index: HashMap::new(),
        })
    }

    /// Build a block from an already-assembled COO whose indices are this
    /// block's **local** numbering (`(node_local, var_idx)` via `ordering`,
    /// nodes positioned as in `row_support` / `col_support`). Lets an assembler
    /// produce all entries in parallel and hand them over in one shot, bypassing
    /// the per-entry [`add_entry`](Self::add_entry).
    #[allow(clippy::too_many_arguments)]
    pub fn from_coo(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetric: bool,
        coo: CooMatrix<f64>,
    ) -> Result<Self> {
        let row_nodes: Vec<NodeId> = read(&row_support)?.connectivity().to_vec();
        let col_nodes: Vec<NodeId> = read(&col_support)?.connectivity().to_vec();
        // The block's row/col numbering snapshots these supports; freeze them.
        crate::containers::mesh::seal(&row_support)?;
        crate::containers::mesh::seal(&col_support)?;
        let nrows = row_nodes.len() * dual_vars.len();
        let ncols = col_nodes.len() * primal_vars.len();
        if coo.nrows() != nrows || coo.ncols() != ncols {
            return Err(PyrucastError::Message(format!(
                "from_coo: COO is {}×{} but the support/vars imply {}×{}",
                coo.nrows(),
                coo.ncols(),
                nrows,
                ncols
            )));
        }
        Ok(Self {
            row_support,
            col_support,
            row_nodes,
            col_nodes,
            dual_vars,
            primal_vars,
            ordering,
            coo,
            symmetric,
            recipe: None,
            physics: Vec::new(),
            factor: 1.0,
            row_index: HashMap::new(),
            col_index: HashMap::new(),
        })
    }

    /// Whether this is a **computed** block (carries a [`ComputedRecipe`], no
    /// stored values) rather than a literal one.
    pub fn is_computed(&self) -> bool {
        self.recipe.is_some()
    }

    /// The block's [`ComputedRecipe`], or `None` for a literal block.
    pub fn recipe(&self) -> Option<&ComputedRecipe> {
        self.recipe.as_ref()
    }

    /// The set of [`Physics`] natures of the sub-model that produced this block —
    /// **empty** for a block built outside assembly (the « rien » case), one entry
    /// for a plain physics, several for a coupled one. Set by the assembler on
    /// every block it emits (see [`crate::ops::assemble`]).
    pub fn physics(&self) -> &[Physics] {
        &self.physics
    }

    /// Tag this block with the [`Physics`] nature set of its producing sub-model —
    /// the assembler calls this on each emitted block so [`Matrix::filter`] can
    /// select by nature (matched by containment).
    pub fn set_physics(&mut self, physics: Vec<Physics>) {
        self.physics = physics;
    }

    /// The scalar factor applied to every value this block emits (`1.0` unless
    /// scaled via `Mul<f64>`/`Div<f64>`) — see the struct-level field doc for
    /// exactly where it is and isn't applied.
    pub fn factor(&self) -> f64 {
        self.factor
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
        if self.is_computed() {
            return Err(PyrucastError::Message(
                "add_entry: this is a computed block (its values come from its \
                 recipe at assembly time); literal entries cannot be added"
                    .into(),
            ));
        }
        let n_rn = self.row_nodes.len();
        let n_dv = self.dual_vars.len();
        let n_cn = self.col_nodes.len();
        let n_pv = self.primal_vars.len();

        // O(1) node → local position (maps built lazily; support is fixed).
        self.ensure_node_indices();
        let rnl = *self.row_index.get(&row_node).ok_or_else(|| {
            PyrucastError::Message(format!(
                "add_entry: row node {row_node:?} not in row_support"
            ))
        })? as usize;
        let rvi = self
            .dual_vars
            .iter()
            .position(|v| v == row_var)
            .ok_or_else(|| {
                PyrucastError::Message(format!("add_entry: row var '{row_var}' not in dual_vars"))
            })?;
        let cnl = *self.col_index.get(&col_node).ok_or_else(|| {
            PyrucastError::Message(format!(
                "add_entry: col node {col_node:?} not in col_support"
            ))
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
    /// per-block translation table. **Not** scaled by [`SubMatrix::factor`]: the
    /// caller applies it (every consumer inside `containers`/`ops::assemble` does).
    pub fn local_triplets(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.coo.triplet_iter().map(|(r, c, &v)| (r, c, v))
    }

    /// The block's COO as raw parallel slices `(rows, cols, values)`, in
    /// **local** index form. Same data as [`local_triplets`](Self::local_triplets)
    /// but indexable, so the aggregate can remap the entries in parallel. **Not**
    /// scaled by [`SubMatrix::factor`] — see [`local_triplets`](Self::local_triplets).
    pub fn local_coo_arrays(&self) -> (&[usize], &[usize], &[f64]) {
        (
            self.coo.row_indices(),
            self.coo.col_indices(),
            self.coo.values(),
        )
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

        let raw: f64 = self
            .coo
            .row_indices()
            .iter()
            .zip(self.coo.col_indices())
            .zip(self.coo.values())
            .filter(|&((&r, &c), _)| r == ri && c == ci)
            .map(|(_, &v)| v)
            .sum();
        raw * self.factor
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
                    v * self.factor,
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
            out[(r, c)] += v * self.factor;
        }
        out
    }

    /// The internal COO matrix, with [`SubMatrix::factor`] baked in (cloned as-is
    /// when the factor is `1.0`, rebuilt with scaled values otherwise).
    pub fn to_coo(&self) -> CooMatrix<f64> {
        if self.factor == 1.0 {
            return self.coo.clone();
        }
        let mut out = CooMatrix::new(self.coo.nrows(), self.coo.ncols());
        for (r, c, &v) in self.coo.triplet_iter() {
            out.push(r, c, v * self.factor);
        }
        out
    }

    /// Convert this block to a [`nalgebra_sparse::CsrMatrix`] (factor applied).
    pub fn to_csr(&self) -> CsrMatrix<f64> {
        CsrMatrix::from(&self.to_coo())
    }

    /// Convert this block to a [`nalgebra_sparse::CscMatrix`] (factor applied).
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
}

// ─── SubMatrix scalar operators ─────────────────────────────────────────────
//
// `blk * s` / `blk / s` only touch `factor` — never the stored `coo` — so they
// are zero-copy and work identically for a literal block (values live in `coo`)
// and a computed one (values don't exist until assembly evaluates the recipe).
// No `Add`/`Sub<f64>`: shifting a matrix by a constant has no physical meaning.

impl std::ops::Mul<f64> for SubMatrix {
    type Output = SubMatrix;
    fn mul(mut self, rhs: f64) -> SubMatrix {
        self.factor *= rhs;
        self
    }
}

impl std::ops::Mul<f64> for &SubMatrix {
    type Output = SubMatrix;
    fn mul(self, rhs: f64) -> SubMatrix {
        self.clone() * rhs
    }
}

impl std::ops::Div<f64> for SubMatrix {
    type Output = SubMatrix;
    fn div(mut self, rhs: f64) -> SubMatrix {
        self.factor /= rhs;
        self
    }
}

impl std::ops::Div<f64> for &SubMatrix {
    type Output = SubMatrix;
    fn div(self, rhs: f64) -> SubMatrix {
        self.clone() / rhs
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
            .field("factor", &self.factor)
            .finish()
    }
}

impl fmt::Display for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Literal blocks render exactly as before; computed blocks have no
        // stored entries, so report their recipe instead of an entry count.
        let entries: std::borrow::Cow<str> = if self.is_computed() {
            "computed (values from recipe)".into()
        } else {
            format!("{} entries", self.coo.nnz()).into()
        };
        let physics: std::borrow::Cow<str> = if self.physics.is_empty() {
            "".into()
        } else {
            let tags: Vec<&str> = self.physics.iter().map(|p| p.to_tag()).collect();
            format!(", {}", tags.join("+")).into()
        };
        let factor: std::borrow::Cow<str> = if self.factor == 1.0 {
            "".into()
        } else {
            format!(", ×{}", self.factor).into()
        };
        write!(
            f,
            "SubMatrix: {} row(s) × {} col(s), {}{}{}{}",
            self.coo.nrows(),
            self.coo.ncols(),
            entries,
            if self.symmetric { ", symmetric" } else { "" },
            physics,
            factor,
        )
    }
}

/// Format a DOF `(node, var)` pair as the grid label `(node,var)`.
fn dof_label((n, v): &NamedDof) -> String {
    format!("({n},{v})")
}

impl crate::dump::Dump for SubMatrix {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        // A computed block holds no values — show its structure, not a grid of
        // zeros. The recipe's handles identify its sub-model / FE subspace.
        if self.is_computed() {
            let recipe = self.recipe.as_ref().expect("is_computed ⇒ recipe");
            return format!(
                "{self}\n  recipe: submodel {:?}, fespaces {:?}{}\n  dual_vars: [{}]\n  primal_vars: [{}]",
                recipe.submodel,
                recipe.fespaces,
                recipe
                    .material
                    .as_ref()
                    .map(|m| format!(", material {m:?}"))
                    .unwrap_or_default(),
                self.dual_vars.join(", "),
                self.primal_vars.join(", "),
            );
        }
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

/// Global CSR sparsity pattern plus the DOF numbering it indexes.
///
/// A pure function of a model's **block structure** — not of the material
/// values — so it can be built once and reused across assemblies of the same
/// model (materials change, sparsity does not). The assembler
/// ([`crate::ops::assemble`]) builds it from a [`Matrix`]'s blocks and caches it
/// on the model (see `Model::matrix_pattern`); the numeric scatter then only
/// fills the values at each entry's fixed slot.
#[derive(Clone)]
pub struct AssemblyPattern {
    /// Global row DOFs, in CSR row order.
    pub row_dofs: Vec<NamedDof>,
    /// Global column DOFs, in CSR column order.
    pub col_dofs: Vec<NamedDof>,
    /// CSR row offsets, length `row_dofs.len() + 1`.
    pub row_offsets: Vec<usize>,
    /// CSR column indices, sorted within each row, length `row_offsets[nrows]`.
    pub col_indices: Vec<usize>,
    /// Precomputed value-array slot of every entry each block contributes, one
    /// entry per block in the matrix's `subs` order. Since the pattern is
    /// material-independent and cached, the `binary_search` that maps a
    /// block-local `(r, c)` to its CSR slot is paid **once here**, not on every
    /// assembly's numeric scatter. The scatter reads these back directly (see
    /// [`crate::ops::assemble::scatter`]).
    pub block_slots: Vec<BlockSlots>,
}

/// Precomputed CSR value-array slots for one block, aligned with the order that
/// block emits its entries at scatter time. A computed block's cells are
/// evaluated cell-by-cell, so its slots are grouped per cell, matching
/// `element_block_triplets_per_cell`'s `per_cell` (and, entry-for-entry, the
/// `(li, di, lj, pj)` emission order of `element_block_pattern`). A literal
/// block's slots follow its `local_coo_arrays` order.
#[derive(Clone)]
pub enum BlockSlots {
    /// One slot list per cell (computed block).
    Computed(Vec<Vec<usize>>),
    /// One slot per COO entry (literal block).
    Literal(Vec<usize>),
}

impl AssemblyPattern {
    /// Value-array slot of global entry `(r, c)`. `c` must be present in row
    /// `r`'s column set — it is, for any entry a block contributes (the pattern
    /// was built from exactly those entries).
    ///
    /// Only the pattern build calls this; the numeric scatter reads the
    /// precomputed [`AssemblyPattern::block_slots`] instead.
    #[inline]
    pub(crate) fn slot(&self, r: usize, c: usize) -> usize {
        let base = self.row_offsets[r];
        let seg = &self.col_indices[base..self.row_offsets[r + 1]];
        base + seg
            .binary_search(&c)
            .expect("scatter: entry (r, c) absent from the CSR pattern")
    }

    /// Number of stored entries (CSR `nnz`).
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }
}

/// Sort each row segment `pairs[bounds[i]..bounds[i+1]]` by column, in place and
/// in parallel. `bounds` are absolute offsets into the original buffer (so
/// `bounds[0]` is this slice's base); recursion splits the **row range** and the
/// buffer together via `split_at_mut`, giving each task a disjoint slice. The
/// sort is stable, preserving the stream order of equal columns.
fn sort_rows_in_place(pairs: &mut [(usize, f64)], bounds: &[usize]) {
    let nrows = bounds.len() - 1;
    let base = bounds[0];
    // Below this many entries, the task-spawn overhead outweighs the work: sort
    // the remaining rows serially.
    const SERIAL_BELOW: usize = 4096;
    if nrows <= 1 || pairs.len() < SERIAL_BELOW {
        for i in 0..nrows {
            pairs[bounds[i] - base..bounds[i + 1] - base].sort_by_key(|&(c, _)| c);
        }
        return;
    }
    let mid = nrows / 2;
    let (left, right) = pairs.split_at_mut(bounds[mid] - base);
    rayon::join(
        || sort_rows_in_place(left, &bounds[..=mid]),
        || sort_rows_in_place(right, &bounds[mid..]),
    );
}

/// Build a CSR matrix from unsorted global `(row, col, value)` triplets,
/// **summing duplicates**, in parallel. Equivalent to
/// `CsrMatrix::from(&CooMatrix::try_from_triplets(…))`.
///
/// Uses a counting sort by row (cache-friendly bucket scatter) so the only
/// comparison sort is *within* each row — tiny segments sorted across rows in
/// parallel ([`sort_rows_in_place`]). The histogram, scatter and final
/// dedup-and-sum scan are O(nnz) serial passes. The per-row sort is stable, so
/// equal `(row, col)` entries are summed in stream order — bit-for-bit identical
/// to the serial path.
fn csr_from_triplets_parallel(
    nrows: usize,
    ncols: usize,
    triplets: Vec<(usize, usize, f64)>,
) -> Result<CsrMatrix<f64>> {
    let nnz = triplets.len();
    // 1. Entries per row → exclusive prefix sum → per-row bucket bounds.
    let mut bounds = vec![0usize; nrows + 1];
    for &(r, _, _) in &triplets {
        bounds[r + 1] += 1;
    }
    for r in 0..nrows {
        bounds[r + 1] += bounds[r];
    }
    // 2. Scatter (col, val) into each row's bucket, preserving stream order.
    let mut cursor: Vec<usize> = bounds[..nrows].to_vec();
    let mut pairs = vec![(0usize, 0.0f64); nnz];
    for (r, c, v) in triplets {
        pairs[cursor[r]] = (c, v);
        cursor[r] += 1;
    }
    // 3. Sort each row's columns, in place and across rows in parallel.
    sort_rows_in_place(&mut pairs, &bounds);
    // 4. Serial scan: emit one CSR entry per distinct (row, col), summing dups.
    let mut row_offsets = vec![0usize; nrows + 1];
    let mut col_indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut values: Vec<f64> = Vec::with_capacity(nnz);
    for r in 0..nrows {
        let mut last_col: Option<usize> = None;
        for &(c, v) in &pairs[bounds[r]..bounds[r + 1]] {
            if last_col == Some(c) {
                *values.last_mut().unwrap() += v;
            } else {
                col_indices.push(c);
                values.push(v);
                row_offsets[r + 1] += 1;
                last_col = Some(c);
            }
        }
    }
    for r in 0..nrows {
        row_offsets[r + 1] += row_offsets[r];
    }
    CsrMatrix::try_from_csr_data(nrows, ncols, row_offsets, col_indices, values)
        .map_err(|e| PyrucastError::Message(format!("csr_from_triplets_parallel: {e}")))
}

impl Matrix {
    /// Build the global DOF union + CSR. Must be called before any
    /// solver-facing method (`to_csr`, `to_dmatrix`, `mul_dense`, `dense`,
    /// `to_coo`, `to_csc`). Idempotent: a second call is a no-op if no
    /// `add_sub` has occurred since the last `finalize`.
    pub fn finalize(&mut self) -> Result<()> {
        if self.assembled.is_some() {
            return Ok(());
        }
        // A *computed* block carries no values: producing them means driving a
        // model kernel that lives in `crate::models` (outside `containers`).
        // Assembling it here would force a matrix↔kernel cycle, so we don't —
        // the global assembler in `ops::assemble::stiffness` handles computed
        // blocks and injects the finished CSR via `set_assembled`.
        for h in &*self {
            if read(h)?.is_computed() {
                return Err(PyrucastError::Message(
                    "Matrix::finalize: this matrix carries a computed block; \
                     assemble it with ops::assemble::assemble(&mut m) (or \
                     ops::assemble::stiffness), which scatters the kernel into \
                     the global CSR — finalize() cannot (it must not reach into \
                     the model/kernel)"
                        .into(),
                ));
            }
        }
        let row_dofs = self.collect_row_dofs()?;
        let col_dofs = self.collect_col_dofs()?;
        let triplets = self.build_global_triplets(&row_dofs, &col_dofs)?;
        let csr = csr_from_triplets_parallel(row_dofs.len(), col_dofs.len(), triplets)?;
        self.assembled = Some(AssembledData {
            row_dofs,
            col_dofs,
            csr,
        });
        Ok(())
    }

    /// Inject a globally-assembled CSR built by an external assembler
    /// ([`crate::ops::assemble`]), bypassing [`finalize`](Self::finalize).
    /// `row_dofs` / `col_dofs` must be this matrix' global DOF union (as
    /// returned by [`row_dofs`](Self::row_dofs) / [`col_dofs`](Self::col_dofs))
    /// and index `csr`. This is the path for matrices carrying *computed*
    /// blocks, which `finalize` cannot assemble on its own (it would have to
    /// reach into the model/kernel — the cycle Option B avoids).
    pub(crate) fn set_assembled(
        &mut self,
        row_dofs: Vec<NamedDof>,
        col_dofs: Vec<NamedDof>,
        csr: CsrMatrix<f64>,
    ) {
        self.assembled = Some(AssembledData {
            row_dofs,
            col_dofs,
            csr,
        });
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
    pub fn store_factorization(
        &self,
        factorization: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) {
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
    /// [`permutation`](crate::coords::Coords::permutation) (stable
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

    /// Map every block's **local** triplets to global `(row, col, value)`
    /// arrays via a per-block translation table (built once from the global DOF
    /// maps). The remap is index-preserving, so the concatenated stream — blocks
    /// in order, entries in COO order — matches the old serial scatter. The
    /// per-block remap runs in parallel. O(total block DOFs + nnz), no per-entry
    /// search.
    fn build_global_triplets(
        &self,
        row_dofs: &[NamedDof],
        col_dofs: &[NamedDof],
    ) -> Result<Vec<(usize, usize, f64)>> {
        let row_map: HashMap<NamedDof, usize> = row_dofs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, d)| (d, i))
            .collect();
        let col_map: HashMap<NamedDof, usize> = col_dofs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, d)| (d, i))
            .collect();
        let mut out: Vec<(usize, usize, f64)> = Vec::new();
        for h in self {
            let sub = read(h)?;
            // local DOF index → global index (the "simple remap").
            let trow: Vec<usize> = sub.row_dofs().iter().map(|d| row_map[d]).collect();
            let tcol: Vec<usize> = sub.col_dofs().iter().map(|d| col_map[d]).collect();
            let (lr, lc, lv) = sub.local_coo_arrays();
            let factor = sub.factor();
            let block: Vec<(usize, usize, f64)> = (0..lv.len())
                .into_par_iter()
                .with_min_len(MIN_PARALLEL_LEN)
                .map(|k| (trow[lr[k]], tcol[lc[k]], lv[k] * factor))
                .collect();
            if out.is_empty() {
                out = block;
            } else {
                out.extend(block);
            }
        }
        Ok(out)
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
        // When assembled, the CSR is the source of truth — and the only place a
        // *computed* block's values live (its COO is empty). Fall back to the
        // per-block COO sum only for an unassembled (literal) matrix.
        if let Some(a) = &self.assembled {
            let r = a
                .row_dofs
                .iter()
                .position(|(n, v)| *n == row_node && v == row_field);
            let c = a
                .col_dofs
                .iter()
                .position(|(n, v)| *n == col_node && v == col_field);
            return Ok(match (r, c) {
                (Some(r), Some(c)) => a.csr.get_entry(r, c).map(|e| e.into_value()).unwrap_or(0.0),
                _ => 0.0,
            });
        }
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

    /// [`Mesh`] over the blocks' distinct **row** supports (the dual side:
    /// where a right-hand side is read and where `A · x` lands) — one POI1
    /// submesh per distinct support, **sharing the blocks' own handles**
    /// (nothing copied, first-seen order).
    ///
    /// The projection target for building a field that combines with this
    /// matrix's row-side fields: `restrict(&f_ext, &k.row_mesh()?)` lands the
    /// external forces on the very supports [`mul_field`](Self::mul_field)
    /// (internal forces `K·u`) lives on, so `&f_ext_r - &f_int` aligns zone by
    /// zone instead of passing through. Available before
    /// [`finalize`](Self::finalize) (the supports are structural).
    pub fn row_mesh(&self) -> Result<Mesh> {
        self.support_mesh(true)
    }

    /// [`Mesh`] over the blocks' distinct **column** supports (the primal
    /// side: where a `solve` solution lives). Column twin of
    /// [`row_mesh`](Self::row_mesh) — e.g. to project an initial or imposed
    /// field onto the exact supports of the solution before combining.
    pub fn col_mesh(&self) -> Result<Mesh> {
        self.support_mesh(false)
    }

    /// A fresh [`Matrix`] holding only the blocks **whose nature set contains** the
    /// given [`Physics`] (`k.filter(Physics::Mechanical)` → every block that is at
    /// least mechanical). The matrix-side twin of
    /// [`Model::filter`](crate::containers::model::Model::filter).
    ///
    /// Block order is preserved and handles are **shared** (refcount bump) via
    /// [`Aggregate::subset`]. Blocks with an **empty** nature set (built outside
    /// assembly — the « rien » case) are never selected by a concrete nature; tag
    /// them [`Physics::Other`] to reach them with `filter(Physics::Other)`. The
    /// result is **not assembled** — like any matrix with freshly added blocks,
    /// call [`crate::ops::assemble::assemble`] before handing it to a solver.
    pub fn filter(&self, physics: Physics) -> Result<Matrix> {
        let mut indices: Vec<usize> = Vec::new();
        for (i, h) in self.iter().enumerate() {
            if read(h)?.physics().contains(&physics) {
                indices.push(i);
            }
        }
        self.subset(indices)
    }

    /// The set of [`Physics`] natures present across this matrix's blocks —
    /// first-seen order, deduplicated. Empty if no block is tagged (« rien »).
    /// A matrix aggregating several physics reports **several** tags here (e.g.
    /// a heat model with a Dirichlet → `[Thermal, Constraint]`).
    pub fn physics(&self) -> Result<Vec<Physics>> {
        let mut out: Vec<Physics> = Vec::new();
        for h in self {
            for &p in read(h)?.physics() {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        Ok(out)
    }

    /// Shared body of `row_mesh` / `col_mesh`.
    fn support_mesh(&self, rows: bool) -> Result<Mesh> {
        let mut out = Mesh::empty();
        for h in self {
            let s = read(h)?;
            let sup = if rows {
                s.row_support.clone()
            } else {
                s.col_support.clone()
            };
            if !out.iter().any(|m| m.same_slot(&sup)) {
                out.add_sub(sup)?;
            }
        }
        Ok(out)
    }

    /// Wrap a flat **column-ordered** vector (`assembled.col_dofs` order — the
    /// layout a solver's solution comes in) into a [`NodeField`] whose zones
    /// **share the blocks' `col_support` handles**: one zone per distinct
    /// column support, carrying the union of the primal variables the blocks
    /// declare on it. No support `SubMesh` is materialised — the zones sit on
    /// the very POI1 supports the sub-models built once at construction, so
    /// the output satisfies [`same_support`](crate::containers::field::SubField::same_support)
    /// with any other field on those supports, across solves and re-assemblies.
    ///
    /// A `(node, variable)` pair a support carries but the DOF table does not
    /// reads as `0.0` (the [`NodeField::gather`] convention). Interface nodes
    /// shared by several supports are stored once per zone, with equal values
    /// by construction (they come from the same vector).
    ///
    /// Requires [`finalize`](Self::finalize).
    pub fn field_from_col_values(&self, x: &[f64]) -> Result<NodeField> {
        self.field_from_flat_values(x, false)
    }

    /// Row-side twin of [`field_from_col_values`](Self::field_from_col_values):
    /// wrap a flat **row-ordered** vector (`assembled.row_dofs` order — the
    /// layout `A · x` comes in) into a [`NodeField`] whose zones share the
    /// blocks' `row_support` handles and carry their dual variables.
    pub fn field_from_row_values(&self, y: &[f64]) -> Result<NodeField> {
        self.field_from_flat_values(y, true)
    }

    /// Shared body of `field_from_{col,row}_values` (`rows` picks the side).
    fn field_from_flat_values(&self, values: &[f64], rows: bool) -> Result<NodeField> {
        let a = self.assembled_or_err()?;
        let dofs = if rows { &a.row_dofs } else { &a.col_dofs };
        if values.len() != dofs.len() {
            return Err(PyrucastError::Message(format!(
                "field_from_{}_values: {} value(s) for {} DOF(s)",
                if rows { "row" } else { "col" },
                values.len(),
                dofs.len()
            )));
        }
        // Global (node, variable) → flat index, one hash pass.
        let index: HashMap<(NodeId, &str), usize> = dofs
            .iter()
            .enumerate()
            .map(|(i, (nid, name))| ((*nid, name.as_str()), i))
            .collect();

        // Group the blocks by support slot; union their variables per group.
        // Same slot ⇒ same sealed POI1 ⇒ same node list, snapshot it once.
        struct Group {
            support: Handle<SubMesh>,
            nodes: Vec<NodeId>,
            vars: Vec<String>,
        }
        let mut groups: Vec<Group> = Vec::new();
        for h in self {
            let s = read(h)?;
            let (support, nodes, vars) = if rows {
                (s.row_support.clone(), &s.row_nodes, s.dual_vars())
            } else {
                (s.col_support.clone(), &s.col_nodes, s.primal_vars())
            };
            match groups.iter_mut().find(|g| g.support.same_slot(&support)) {
                Some(g) => {
                    for v in vars {
                        if !g.vars.contains(v) {
                            g.vars.push(v.clone());
                        }
                    }
                }
                None => groups.push(Group {
                    support,
                    nodes: nodes.clone(),
                    vars: vars.to_vec(),
                }),
            }
        }

        // One zone per group, on the block's own support handle. The field's
        // row order is the support's cell order — exactly `group.nodes` (both
        // snapshot the same sealed connectivity) — so values are written
        // positionally, no per-node lookup.
        let mut out = NodeField::default();
        for g in &groups {
            use crate::containers::field::SubField;
            let mut sub = SubNodeField::from_poi1(&g.support, g.vars.clone())?;
            let ncomp = g.vars.len();
            let vals = sub.values_mut();
            for (ni, nid) in g.nodes.iter().enumerate() {
                for (ci, var) in g.vars.iter().enumerate() {
                    if let Some(&gi) = index.get(&(*nid, var.as_str())) {
                        vals[ni * ncomp + ci] = values[gi];
                    }
                }
            }
            out.add_sub(insert(sub))?;
        }
        Ok(out)
    }

    /// `y = A · x` against a [`NodeField`]. The column vector `x` is read from
    /// `x_field` at the matrix's **column** DOFs (aggregate resolution, first
    /// zone wins; a DOF no zone defines reads as `0.0`); the result `y` is a
    /// `NodeField` whose zones **share the blocks' row supports**
    /// ([`field_from_row_values`](Self::field_from_row_values)) and carry
    /// their dual variables.
    ///
    /// Columns carry the **primal** variables and rows the **dual** ones
    /// (`K · u = f`), so this maps a *primal* field (e.g. `"T"`, `"u"`) to a
    /// *dual* one (e.g. `"q"`). That is the exact mirror of
    /// [`crate::ops::solver::lu::solve`], which reads a *dual* right-hand side at the
    /// rows and produces a *primal* solution at the columns. Both use
    /// [`NodeField::gather`] / [`NodeField::from_dof_values`] to bridge the
    /// abstract field and the flat DOF vector. Requires
    /// [`finalize`](Self::finalize); the `*` operator (`&matrix * &field`) is
    /// sugar for this method.
    pub fn mul_field(&self, x_field: &NodeField) -> Result<NodeField> {
        let x = x_field.gather(&self.col_dofs()?)?;
        let y = self.mul_dense(&x)?;
        self.field_from_row_values(&y)
    }

    /// A fresh [`Matrix`] with every block replaced by `f(block.clone())`, each
    /// re-inserted under a **new store slot**. Backs the scalar operators
    /// (`Mul<f64>`/`Div<f64>`, which only touch each clone's `factor`). Never
    /// mutates `self` or any of its blocks in place: `add_sub`/`union`/`filter`
    /// share `Handle<SubMatrix>`s (same store slot, refcount bump — see
    /// [`Aggregate::subset`]), so scaling in place would silently rescale every
    /// other `Matrix` aliasing the same block. Like [`filter`](Self::filter), the
    /// result is **not assembled**.
    fn map_blocks(&self, f: impl Fn(SubMatrix) -> SubMatrix) -> Result<Matrix> {
        let mut out = Matrix::empty();
        for h in self {
            let scaled = f((*read(h)?).clone());
            out.add_sub(insert(scaled))?;
        }
        Ok(out)
    }
}

impl std::ops::Mul<&NodeField> for &Matrix {
    type Output = Result<NodeField>;
    /// `&matrix * &field` — sugar for [`Matrix::mul_field`]. Fallible (the
    /// matrix must be finalized), so the result is a `Result`: `(&k * &x)?`.
    fn mul(self, rhs: &NodeField) -> Self::Output {
        self.mul_field(rhs)
    }
}

impl std::ops::Mul<&NodeField> for Matrix {
    type Output = Result<NodeField>;
    fn mul(self, rhs: &NodeField) -> Self::Output {
        self.mul_field(rhs)
    }
}

// ─── Matrix scalar operators ────────────────────────────────────────────────
//
// `&matrix * s` / `&matrix / s` — a fresh `Matrix` whose blocks are scaled
// clones of `self`'s (see `map_blocks`). Fallible (store reads), like the
// crate's other `Matrix` operators. No `Matrix + Matrix`: the assembler already
// sums contributions landing on the same global `(row, col)`
// ([`crate::ops::assemble::assemble`]), so `M/dt + K` is `(&(&m / dt)? | &k)?`
// followed by `ops::assemble::assemble(&mut sys)` — see `book/src/matrix.md`.

impl std::ops::Mul<f64> for &Matrix {
    type Output = Result<Matrix>;
    fn mul(self, rhs: f64) -> Self::Output {
        self.map_blocks(|b| b * rhs)
    }
}

impl std::ops::Mul<f64> for Matrix {
    type Output = Result<Matrix>;
    fn mul(self, rhs: f64) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl std::ops::Div<f64> for &Matrix {
    type Output = Result<Matrix>;
    fn div(self, rhs: f64) -> Self::Output {
        self.map_blocks(|b| b / rhs)
    }
}

impl std::ops::Div<f64> for Matrix {
    type Output = Result<Matrix>;
    fn div(self, rhs: f64) -> Self::Output {
        (&self).div(rhs)
    }
}

impl crate::dump::Dump for Matrix {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        // Build the global labelled grid on the fly — `collect_*_dofs` and
        // `build_global_triplets` take `&self`, so no `finalize()` (which needs
        // `&mut`) is required: a matrix dumps the same content whether assembled
        // or not.
        let grid = (|| -> Result<String> {
            // When assembled, dump the cached CSR (the single source of truth —
            // and the only correct view for a matrix with *computed* blocks,
            // whose values the literal triplet path does not carry). Otherwise
            // build the labelled grid on the fly from the literal blocks.
            let (row_dofs, col_dofs, data) = if let Some(a) = &self.assembled {
                let nc = a.col_dofs.len();
                let mut data = vec![0.0f64; a.row_dofs.len() * nc];
                for (r, c, v) in a.csr.triplet_iter() {
                    data[r * nc + c] = *v;
                }
                (a.row_dofs.clone(), a.col_dofs.clone(), data)
            } else {
                let row_dofs = self.collect_row_dofs()?;
                let col_dofs = self.collect_col_dofs()?;
                let triplets = self.build_global_triplets(&row_dofs, &col_dofs)?;
                let nc = col_dofs.len();
                let mut data = vec![0.0f64; row_dofs.len() * nc];
                for (r, c, v) in triplets {
                    data[r * nc + c] += v;
                }
                (row_dofs, col_dofs, data)
            };
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
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::coords::Coords;
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
        m.set_physics(vec![Physics::Thermal]);
        use crate::persist::Persist;
        let bytes = m.to_bytes().unwrap();
        let m2 = SubMatrix::from_bytes(&bytes).unwrap();
        assert_eq!(m2.n_rows(), 2);
        assert_eq!(m2.n_cols(), 2);
        assert!(m2.symmetric());
        assert_eq!(m2.get(a, "q", a, "T"), 2.0);
        // The physics tag set survives the round trip.
        assert_eq!(m2.physics(), &[Physics::Thermal]);
    }

    #[test]
    fn sub_round_trip_serde_with_non_default_factor() {
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
        let m = m * 2.5;
        use crate::persist::Persist;
        let bytes = m.to_bytes().unwrap();
        let m2 = SubMatrix::from_bytes(&bytes).unwrap();
        assert_eq!(m2.factor(), 2.5);
        assert_eq!(m2.get(a, "q", a, "T"), 5.0);
    }

    #[test]
    fn sub_matrix_mul_and_div_scale_only_the_factor() {
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
        assert_eq!(m.factor(), 1.0);

        // Reference version clones, leaving `m` untouched.
        let scaled = &m * 3.0;
        assert_eq!(scaled.factor(), 3.0);
        assert_eq!(scaled.get(a, "q", a, "T"), 6.0);
        assert_eq!(scaled.dense(), vec![6.0]);
        assert_eq!(scaled.iter_entries()[0].4, 6.0);
        assert_eq!(
            m.get(a, "q", a, "T"),
            2.0,
            "reference Mul must not mutate m"
        );

        // Consuming version chains: ×3 then ÷2 ⇒ factor 1.5.
        let halved = scaled / 2.0;
        assert_eq!(halved.factor(), 1.5);
        assert_eq!(halved.get(a, "q", a, "T"), 3.0);
        assert_eq!(halved.to_coo().values(), &[3.0]);
        assert_eq!(halved.to_csr().values(), &[3.0]);
        assert_eq!(halved.to_csc().values(), &[3.0]);
        assert_eq!(halved.to_dmatrix()[(0, 0)], 3.0);
        assert_eq!(halved.mul_dense(&[1.0]).unwrap(), vec![3.0]);
        // The raw local form is untouched by the factor.
        assert_eq!(halved.local_coo_arrays().2, &[2.0]);
    }

    #[test]
    fn physics_tag_set_empty_multiple_and_other() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, _b) = (nodes[0].id(), nodes[1].id());
        let make = || {
            SubMatrix::new(
                sup.clone(),
                sup.clone(),
                vec!["q".into()],
                vec!["T".into()],
                DofOrdering::NodesThenVars,
                false,
            )
            .unwrap()
        };

        // A fresh block is untagged — the "rien" case.
        let bare = make();
        assert!(bare.physics().is_empty());

        // A coupled block carries several natures; filter matches by containment.
        let mut coupled = make();
        coupled.set_physics(vec![Physics::Mechanical, Physics::Thermal]);
        assert_eq!(coupled.physics(), &[Physics::Mechanical, Physics::Thermal]);

        // An explicit "other" nature is filterable, unlike the empty set.
        let mut other = make();
        other.set_physics(vec![Physics::Other]);

        let mut k = Matrix::empty();
        k.add_sub(insert(bare)).unwrap();
        k.add_sub(insert(coupled)).unwrap();
        k.add_sub(insert(other)).unwrap();
        let _ = a; // silence unused in some build configs

        // Containment: the coupled block appears under both its natures.
        assert_eq!(k.filter(Physics::Mechanical).unwrap().len(), 1);
        assert_eq!(k.filter(Physics::Thermal).unwrap().len(), 1);
        // Only the explicitly-tagged block is reached by Other; the bare one never.
        assert_eq!(k.filter(Physics::Other).unwrap().len(), 1);
        // The aggregate reports every distinct nature present (bare contributes none).
        let present = k.physics().unwrap();
        assert!(present.contains(&Physics::Mechanical));
        assert!(present.contains(&Physics::Thermal));
        assert!(present.contains(&Physics::Other));
        assert_eq!(present.len(), 3);
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
    fn aggregate_scale_is_isolated_from_the_source_matrix() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut blk = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        blk.add_entry(a, "q", a, "T", 2.0).unwrap();
        blk.add_entry(b, "q", b, "T", 3.0).unwrap();
        let mut orig = Matrix::empty();
        orig.add_sub(insert(blk)).unwrap();

        let scaled = (&orig * 10.0).unwrap();

        // A new store slot per block — no aliasing with the source's blocks.
        let orig_h = orig.iter().next().unwrap();
        let scaled_h = scaled.iter().next().unwrap();
        assert!(!orig_h.same_slot(scaled_h));

        // Values diverge accordingly: the source is untouched.
        assert_eq!(orig.get(a, "q", a, "T").unwrap(), 2.0);
        assert_eq!(orig.get(b, "q", b, "T").unwrap(), 3.0);
        assert_eq!(scaled.get(a, "q", a, "T").unwrap(), 20.0);
        assert_eq!(scaled.get(b, "q", b, "T").unwrap(), 30.0);

        // `/` divides the factor, chaining from the already-scaled matrix.
        let halved = (&scaled / 2.0).unwrap();
        assert_eq!(halved.get(a, "q", a, "T").unwrap(), 10.0);
        assert_eq!(
            scaled.get(a, "q", a, "T").unwrap(),
            20.0,
            "/ must not mutate its source either"
        );
    }

    /// `M/dt + K` ≡ `(M/dt) | K` followed by `ops::assemble::assemble` — no
    /// dedicated `Matrix + Matrix` operator is needed, because the assembler
    /// already sums contributions landing on the same global `(row, col)`.
    /// `K` carries a DOF (`c`) that `M` doesn't (mirroring a Dirichlet
    /// multiplier row/column, which only ever enters the stiffness matrix) —
    /// the union must still assemble correctly, leaving that entry untouched by
    /// `M`'s contribution.
    #[test]
    fn union_and_reassemble_combines_scaled_mass_with_stiffness() {
        let (coords, nodes, _) = make_poi1(3);
        let (a, b, c) = (nodes[0].id(), nodes[1].id(), nodes[2].id());

        let mut sup_k = SubMesh::new(coords.clone(), ElementType::POI1);
        sup_k.add_cell(&[a]).unwrap();
        sup_k.add_cell(&[b]).unwrap();
        sup_k.add_cell(&[c]).unwrap();
        let sup_k = insert(sup_k);
        let mut k_blk = SubMatrix::new(
            sup_k.clone(),
            sup_k,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        k_blk.add_entry(a, "q", a, "T", 2.0).unwrap();
        k_blk.add_entry(a, "q", b, "T", -1.0).unwrap();
        k_blk.add_entry(b, "q", a, "T", -1.0).unwrap();
        k_blk.add_entry(b, "q", b, "T", 2.0).unwrap();
        k_blk.add_entry(c, "q", c, "T", 5.0).unwrap(); // Lagrange-only DOF
        let mut k = Matrix::empty();
        k.add_sub(insert(k_blk)).unwrap();

        let mut sup_m = SubMesh::new(coords.clone(), ElementType::POI1);
        sup_m.add_cell(&[a]).unwrap();
        sup_m.add_cell(&[b]).unwrap();
        let sup_m = insert(sup_m);
        let mut m_blk = SubMatrix::new(
            sup_m.clone(),
            sup_m,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        m_blk.add_entry(a, "q", a, "T", 4.0).unwrap();
        m_blk.add_entry(b, "q", b, "T", 4.0).unwrap();
        let mut m = Matrix::empty();
        m.add_sub(insert(m_blk)).unwrap();

        let dt = 0.5;
        let m_dt = (&m / dt).unwrap(); // factor = 1/0.5 = 2 ⇒ diag(8, 8)

        let mut sys = m_dt.union(&k).unwrap();
        crate::ops::assemble::assemble(&mut sys).unwrap();

        assert_eq!(sys.n_rows().unwrap(), 3);
        assert_eq!(sys.n_cols().unwrap(), 3);
        assert_eq!(sys.get(a, "q", a, "T").unwrap(), 2.0 + 8.0);
        assert_eq!(sys.get(a, "q", b, "T").unwrap(), -1.0);
        assert_eq!(sys.get(b, "q", a, "T").unwrap(), -1.0);
        assert_eq!(sys.get(b, "q", b, "T").unwrap(), 2.0 + 8.0);
        // Untouched by M (which doesn't carry the c DOF at all).
        assert_eq!(sys.get(c, "q", c, "T").unwrap(), 5.0);
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
        crate::store::write(&coords)
            .unwrap()
            .set_permutation(perm)
            .unwrap();

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
    fn aggregate_mul_field_matches_mul_dense_and_operator() {
        use crate::containers::node_field::{NodeField, SubNodeField};

        // K = [[2,-1],[-1,2]] with dual rows "q" and primal columns "T".
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut sm = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        sm.add_entry(a, "q", a, "T", 2.0).unwrap();
        sm.add_entry(a, "q", b, "T", -1.0).unwrap();
        sm.add_entry(b, "q", a, "T", -1.0).unwrap();
        sm.add_entry(b, "q", b, "T", 2.0).unwrap();
        let mut k = Matrix::empty();
        k.add_sub(insert(sm)).unwrap();
        k.finalize().unwrap();

        // x = T:[1, 2] over both column nodes ⇒ y = K·x = [0, 3] at the "q" rows.
        let mut x_sub = SubNodeField::from_poi1(&sup, vec!["T".into()]).unwrap();
        x_sub.set_value(a, "T", 1.0).unwrap();
        x_sub.set_value(b, "T", 2.0).unwrap();
        let x = NodeField::from_sub(x_sub);

        let y = k.mul_field(&x).unwrap();
        assert_eq!(y.value(a, "q").unwrap(), 0.0);
        assert_eq!(y.value(b, "q").unwrap(), 3.0);
        // The result lives on the row DOFs — component is "q", not "T".
        assert!(y.value_opt(a, "T").unwrap().is_none());

        // The `*` operator is sugar for `mul_field`.
        let y_op = (&k * &x).unwrap();
        assert_eq!(y_op.value(a, "q").unwrap(), 0.0);
        assert_eq!(y_op.value(b, "q").unwrap(), 3.0);

        // A column DOF the field does not define contributes 0: with only
        // T(a)=3 set, y = K·[3, 0] = [6, -3].
        let mut x2_sub = SubNodeField::from_poi1(&sup, vec!["T".into()]).unwrap();
        x2_sub.set_value(a, "T", 3.0).unwrap();
        let x2 = NodeField::from_sub(x2_sub);
        let y2 = k.mul_field(&x2).unwrap();
        assert_eq!(y2.value(a, "q").unwrap(), 6.0);
        assert_eq!(y2.value(b, "q").unwrap(), -3.0);
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

    // ── field_from_{col,row}_values ─────────────────────────────────────────

    /// Saddle-point-shaped aggregate (K, C, Cᵀ): the output has one zone per
    /// distinct column support, **sharing the blocks' own handles**, and every
    /// column DOF reads back its slot in the flat vector.
    #[test]
    fn field_from_col_values_shares_block_supports_and_orders_values() {
        // Two supports on one Coords: phys (2 nodes) and mult (1 node).
        let coords = insert(Coords::new(1).unwrap());
        let phys_nodes: Vec<Node> = (0..2)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mult_node = Node::create_in(coords.clone(), &[10.0]).unwrap();
        let phys = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in &phys_nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        let mult = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[mult_node.id()]).unwrap();
            insert(sm)
        };

        // K (phys × phys), C (mult × phys), Cᵀ (phys × mult).
        let mut k = SubMatrix::new(
            phys.clone(),
            phys.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        k.add_entry(phys_nodes[0].id(), "q", phys_nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut c = SubMatrix::new(
            mult.clone(),
            phys.clone(),
            vec!["imposed_T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        c.add_entry(mult_node.id(), "imposed_T", phys_nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut ct = SubMatrix::new(
            phys.clone(),
            mult.clone(),
            vec!["q".into()],
            vec!["lambda_T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        ct.add_entry(phys_nodes[0].id(), "q", mult_node.id(), "lambda_T", 1.0)
            .unwrap();

        let mut m = Matrix::empty();
        m.add_sub(insert(k)).unwrap();
        m.add_sub(insert(c)).unwrap();
        m.add_sub(insert(ct)).unwrap();

        // Not finalized yet ⇒ error.
        assert!(m.field_from_col_values(&[0.0]).is_err());
        m.finalize().unwrap();

        // x holds its own flat index at every column DOF.
        let col_dofs = m.col_dofs().unwrap();
        let x: Vec<f64> = (0..col_dofs.len()).map(|i| i as f64).collect();
        let f = m.field_from_col_values(&x).unwrap();

        // One zone per distinct column support (phys ← K+C, mult ← Cᵀ),
        // each sharing the block's own handle — nothing rebuilt.
        assert_eq!(f.len(), 2);
        assert!(read(&f.get(0).unwrap()).unwrap().support().same_slot(&phys));
        assert!(read(&f.get(1).unwrap()).unwrap().support().same_slot(&mult));

        // Every column DOF reads back its slot; the aggregate is coherent.
        for (i, (nid, var)) in col_dofs.iter().enumerate() {
            assert_eq!(f.value(*nid, var).unwrap(), i as f64);
        }
        f.check().unwrap();

        // Wrong vector length is rejected.
        assert!(m.field_from_col_values(&x[..1]).is_err());
    }

    /// Two blocks on the **same** column support with different primal vars:
    /// one output zone carrying the union of the variables.
    #[test]
    fn field_from_col_values_unions_vars_on_a_shared_support() {
        use crate::containers::field::SubField;
        let (_cfg, nodes, sup) = make_poi1(2);
        let mut a = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        a.add_entry(nodes[0].id(), "q", nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut b = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["r".into()],
            vec!["P".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        b.add_entry(nodes[1].id(), "r", nodes[1].id(), "P", 1.0)
            .unwrap();

        let mut m = Matrix::empty();
        m.add_sub(insert(a)).unwrap();
        m.add_sub(insert(b)).unwrap();
        m.finalize().unwrap();

        let col_dofs = m.col_dofs().unwrap();
        let x: Vec<f64> = (0..col_dofs.len()).map(|i| 10.0 + i as f64).collect();
        let f = m.field_from_col_values(&x).unwrap();

        assert_eq!(f.len(), 1, "same support ⇒ one zone");
        {
            let z = read(&f.get(0).unwrap()).unwrap();
            assert!(z.support().same_slot(&sup));
            assert_eq!(SubField::components(&*z), &["T", "P"]);
        }
        for (i, (nid, var)) in col_dofs.iter().enumerate() {
            assert_eq!(f.value(*nid, var).unwrap(), 10.0 + i as f64);
        }
    }

    /// `row_mesh` / `col_mesh` expose the blocks' supports (shared handles,
    /// deduplicated); a field `restrict`ed onto them lands on those very
    /// supports, so it combines zone by zone with `mul_field`'s output —
    /// the external-forces-minus-internal-forces pattern.
    #[test]
    fn row_mesh_enables_zone_aligned_residual() {
        use crate::containers::node_field::SubNodeField;
        // Saddle-point shape: phys (2 nodes) and mult (1 node) supports.
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let nm = Node::create_in(coords.clone(), &[10.0]).unwrap();
        let phys = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[n0.id()]).unwrap();
            sm.add_cell(&[n1.id()]).unwrap();
            insert(sm)
        };
        let mult = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nm.id()]).unwrap();
            insert(sm)
        };
        // K (phys × phys) and C (mult × phys): row supports {phys, mult},
        // col supports {phys} — K and C share the phys column support.
        let mut k = SubMatrix::new(
            phys.clone(),
            phys.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        k.add_entry(n0.id(), "q", n0.id(), "T", 2.0).unwrap();
        k.add_entry(n1.id(), "q", n1.id(), "T", 2.0).unwrap();
        let mut c = SubMatrix::new(
            mult.clone(),
            phys.clone(),
            vec!["imposed_T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        c.add_entry(nm.id(), "imposed_T", n0.id(), "T", 1.0)
            .unwrap();
        let mut m = Matrix::empty();
        m.add_sub(insert(k)).unwrap();
        m.add_sub(insert(c)).unwrap();

        // Meshes: available pre-finalize, deduplicated, sharing the handles.
        let rm = m.row_mesh().unwrap();
        assert_eq!(rm.len(), 2);
        assert!(rm.get(0).unwrap().same_slot(&phys));
        assert!(rm.get(1).unwrap().same_slot(&mult));
        let cm = m.col_mesh().unwrap();
        assert_eq!(cm.len(), 1, "K and C share the phys column support");
        assert!(cm.get(0).unwrap().same_slot(&phys));

        m.finalize().unwrap();

        // f_int = A · x with x: T = [1, 1] on phys.
        let x = NodeField::from_sub(
            SubNodeField::from_poi1(&phys, vec!["T".into()])
                .map(|mut s| {
                    s.set_value(n0.id(), "T", 1.0).unwrap();
                    s.set_value(n1.id(), "T", 1.0).unwrap();
                    s
                })
                .unwrap(),
        );
        let f_int = m.mul_field(&x).unwrap();

        // External forces on their own support (as `flux` would build them),
        // projected onto the matrix's row mesh: lands on the blocks' handles.
        let f_ext = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[n0.id()]).unwrap();
            sm.add_cell(&[n1.id()]).unwrap();
            let mut s = SubNodeField::from_poi1(&insert(sm), vec!["q".into()]).unwrap();
            s.set_value(n0.id(), "q", 5.0).unwrap();
            s.set_value(n1.id(), "q", 5.0).unwrap();
            NodeField::from_sub(s)
        };
        let f_ext_r = crate::ops::field::restrict(&f_ext, &rm).unwrap();
        for (za, zb) in f_ext_r.iter().zip(f_int.iter()) {
            let sa = read(za).unwrap().support();
            let sb = read(zb).unwrap().support();
            assert!(
                sa.same_slot(&sb),
                "restrict must land on the block supports"
            );
        }

        // Zone-aligned residual: q combines (not passthrough) on phys.
        // K·x: q = 2 at n0 and n1; C·x contributes imposed_T = 1 at nm.
        let r = (&f_ext_r - &f_int).unwrap();
        assert_eq!(r.value(n0.id(), "q").unwrap(), 3.0); // 5 − 2 ⇒ aligned
        assert_eq!(r.value(n1.id(), "q").unwrap(), 3.0);
        // `imposed_T` exists on the f_int side only (restrict carries the
        // source field's components) ⇒ union semantics pass it through RAW
        // (+1, not −1) — the documented `merge_components` behaviour.
        assert_eq!(r.value(nm.id(), "imposed_T").unwrap(), 1.0);

        // Strict residual (every component subtracted, missing read as 0):
        // reproject onto f_int's exact supports AND components.
        let f_ext_like = crate::ops::field::restrict_like(&f_ext, &f_int).unwrap();
        let r2 = (&f_ext_like - &f_int).unwrap();
        assert_eq!(r2.value(n0.id(), "q").unwrap(), 3.0);
        assert_eq!(r2.value(nm.id(), "imposed_T").unwrap(), -1.0); // 0 − 1
    }

    /// Row-side twin: zones on the blocks' row supports, dual variables.
    #[test]
    fn field_from_row_values_uses_row_supports_and_dual_vars() {
        let coords = insert(Coords::new(1).unwrap());
        let r0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let r1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c0 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let sup_r = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[r0.id()]).unwrap();
            sm.add_cell(&[r1.id()]).unwrap();
            insert(sm)
        };
        let sup_c = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[c0.id()]).unwrap();
            insert(sm)
        };
        let mut blk = SubMatrix::new(
            sup_r.clone(),
            sup_c,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            false,
        )
        .unwrap();
        blk.add_entry(r0.id(), "q", c0.id(), "T", 1.0).unwrap();

        let mut m = Matrix::empty();
        m.add_sub(insert(blk)).unwrap();
        m.finalize().unwrap();

        let f = m.field_from_row_values(&[3.0, 7.0]).unwrap();
        assert_eq!(f.len(), 1);
        assert!(read(&f.get(0).unwrap())
            .unwrap()
            .support()
            .same_slot(&sup_r));
        assert_eq!(f.value(r0.id(), "q").unwrap(), 3.0);
        assert_eq!(f.value(r1.id(), "q").unwrap(), 7.0);
    }
}
