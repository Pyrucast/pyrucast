//! Sparse matrix indexed by **named DOFs** `(NodeId, field_name)`.
//!
//! Hierarchy:
//!
//! - [`SubMatrix`] — one COO block.  Structure is fully declared at
//!   construction time via two POI1 [`SubMesh`] handles (row/col node
//!   sets), two variable-name lists (dual / primal), and a
//!   [`DofOrdering`] that maps `(node_local_idx, var_idx)` to a flat
//!   matrix-row or -column index.  The node sets are **read from the
//!   supports**, never copied into the block: the supports are sealed, so
//!   the numbering cannot drift (see `book/src/developper/parallelisme.md`
//!   § Zéro-copie).  The actual non-zeros are stored in a
//!   [`nalgebra_sparse::CooMatrix`]; [`SubMatrix::add_entry`] appends
//!   triplets that accumulate on `get` / densification.
//! - [`Matrix`] — aggregate of [`SubMatrix`] blocks (one
//!   `Vec<Handle<SubMatrix>>`), produced by
//!   [`crate::containers::model::Model`] assembly (one or several blocks
//!   per sub-model).  Read-only: every accessor unions the blocks on the
//!   fly.
//!
//! Each [`SubMatrix`] declares what share of the matrix's symmetry it carries
//! ([`Symmetry`]), and the aggregate [`Matrix`] adds those declarations up
//! ([`Matrix::symmetric`]). Storage is never de-duplicated — a symmetric matrix
//! still holds both triangles — but the declaration *is* consulted: it decides
//! whether the assembled CSR may be handed to the factorization as its own CSC.
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
//! use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
//! use pyrucast::handle::Handle;
//!
//! let coords = Handle::new(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
//! sm.add_cell(&[a.id()]).unwrap();
//! sm.add_cell(&[b.id()]).unwrap();
//! let support = Handle::new(sm);
//!
//! let mut k = SubMatrix::new(
//!     support.clone(), support.clone(),
//!     vec!["q".into()], vec!["T".into()],
//!     DofOrdering::NodesThenVars, Symmetry::Full,
//! );
//! k.add_entry(a.id(), "q", a.id(), "T",  2.0).unwrap();
//! k.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
//! k.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
//! k.add_entry(b.id(), "q", b.id(), "T",  2.0).unwrap();
//!
//! assert_eq!(k.n_rows(), 2);
//! assert_eq!(k.n_cols(), 2);
//! assert!(k.is_symmetric());
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
use crate::handle::{Handle, ReadGuard};
use crate::models::{MatrixKind, Physics};
use crate::parallel::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A single COO entry with DOFs materialised as `(NodeId, var_name)` pairs:
/// `(row_node, row_var, col_node, col_var, value)`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let bloc = || {
/// #     let mut z = SubMatrix::new(
/// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #         DofOrdering::NodesThenVars, Symmetry::Full);
/// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// #     z
/// # };
/// # use pyrucast::containers::matrix::MatrixEntry;
/// // An entry whose DOFs are **materialized**: no more indices, but
/// // `(node, variable)` already resolved.
/// let entrees: Vec<MatrixEntry> = bloc().iter_entries();
/// assert_eq!(entrees.len(), 4);
/// assert_eq!((entrees[0].1.as_str(), entrees[0].3.as_str()), ("q", "T"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type MatrixEntry = (NodeId, String, NodeId, String, f64);

// ─── DofOrdering ───────────────────────────────────────────────────────────

/// How `(node_local_idx, var_idx)` maps to a flat matrix-row or -column
/// index inside a [`SubMatrix`].
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let bloc = || {
/// #     let mut z = SubMatrix::new(
/// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #         DofOrdering::NodesThenVars, Symmetry::Full);
/// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// #     z
/// # };
/// // Two ways to flatten `(node, variable)` into a row index. On two nodes
/// // and two variables, they coincide only at the ends.
/// assert_eq!(DofOrdering::NodesThenVars.to_index(1, 0, 2, 2), 2);
/// assert_eq!(DofOrdering::VarsThenNodes.to_index(1, 0, 2, 2), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// // `NodesThenVars`: every variable of node 0, then those of node 1.
    /// assert_eq!(DofOrdering::NodesThenVars.to_index(1, 1, 2, 2), 3);
    /// // `VarsThenNodes`: every node of variable 0, then those of variable 1.
    /// assert_eq!(DofOrdering::VarsThenNodes.to_index(1, 1, 2, 2), 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// // The exact inverse of `to_index`, for both layouts.
    /// for o in [DofOrdering::NodesThenVars, DofOrdering::VarsThenNodes] {
    ///     for i in 0..4 {
    ///         let (nl, vi) = o.from_index(i, 2, 2);
    ///         assert_eq!(o.to_index(nl, vi, 2, 2), i);
    ///     }
    /// }
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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

/// What a computed block's kernel reads **besides** its material, named per
/// [`MatrixKind`] rather than encoded in an `Option`.
///
/// A stiffness and a mass need nothing; a geometric stiffness needs the current
/// stress; a consistent tangent needs what the constitutive law needs — because
/// it is evaluated at the Gauss point rather than read back from a field that
/// nobody else would consume.
///
/// ```
/// # use pyrucast::containers::matrix::KernelInputs;
/// // A stiffness reads only its material — and that is the default, because
/// // this is the case for three matrix kinds out of four.
/// assert!(matches!(KernelInputs::default(), KernelInputs::MaterialOnly));
/// ```
#[derive(Clone, Default, Serialize, Deserialize)]
pub enum KernelInputs {
    /// Stiffness, mass: the material is enough.
    #[default]
    MaterialOnly,
    /// Geometric stiffness: the current stress.
    State(Handle<SubElementField>),
    /// Consistent tangent: exactly what `behavior::integrate` is given. `prev`
    /// is **not** an `Option` — the rest state is materialised at the operator
    /// boundary, so below that line there is always a real field.
    Behavior {
        deformation: Handle<SubElementField>,
        prev: Handle<SubElementField>,
        dt: f64,
    },
}

/// One sparse COO block whose DOF layout is fully described by two POI1
/// sub-meshes and variable-name lists.
///
/// The **row DOF** at matrix index `i` is
/// `(row_support.connectivity()[node_local], dual_vars[var_idx])` where
/// `(node_local, var_idx) = ordering.from_index(i, n_row_nodes, n_dual_vars)`.
/// Columns are symmetric with `col_support` and `primal_vars`. The block keeps
/// **no node list of its own**: it reads its supports in place, which is sound
/// because both are sealed at construction. Their nodes must be **distinct** —
/// the shape `to_poi1` produces — since `node_local` is looked up through
/// [`SubMesh::node_index`], a rank that only matches the position without
/// repeats.
/// The recipe a **computed** [`SubMatrix`] carries *instead of* stored values:
/// how to evaluate its contribution on the fly. The global assembler drives the
/// sub-model's [`element_matrix`](crate::models::Domain::element_matrix) kernel
/// over `fespace`'s cells and scatters the result straight into the global
/// matrix — a computed block never materialises a COO (its own `coo` stays an
/// empty, correctly-sized placeholder, so structural queries still work).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::matrix::{ComputedRecipe, DofOrdering, KernelInputs, SubMatrix, Symmetry};
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::element_field;
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let support = zone.read().submesh().read().to_poi1().unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let mat = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
/// // A **computed** block materializes no value: it carries the recipe
/// // that the global assembler unwinds cell by cell, straight into the CSR.
/// let recette = ComputedRecipe {
///     submodel: modele.get(0)?,
///     fespaces: vec![zone.clone()],
///     material: Some(mat.get(0)?),
///     kind: MatrixKind::Stiffness,
///     inputs: KernelInputs::MaterialOnly,
///     col_fespaces: Vec::new(),
/// };
/// let z = SubMatrix::computed(
///     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
///     DofOrdering::NodesThenVars, Symmetry::Full, recette);
/// assert!(z.is_computed());
/// // Its structure is complete — hence the structural queries that
/// // work — but it counts **no** stored entry.
/// assert_eq!((z.n_rows(), z.n_cols()), (3, 3));
/// assert_eq!(z.entry_count(), 0);
/// // And nothing can be added to it by hand.
/// # let mut z = z;
/// assert!(z.add_entry(n[0].id(), "q", n[0].id(), "T", 1.0).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// What this kind's kernel reads **besides** the material — see
    /// [`KernelInputs`].
    #[serde(default)]
    pub inputs: KernelInputs,
    /// FE subspaces carrying the **columns** when this block couples two meshes
    /// (an interface exchange law). **Empty** — the overwhelming case — means
    /// rows and columns live on the same mesh and `fespaces` drives both; the
    /// scatter routes on exactly that emptiness.
    #[serde(default)]
    pub col_fespaces: Vec<Handle<SubFiniteElementSpace>>,
}

/// Default value of [`SubMatrix::factor`] for pre-existing serialized data
/// that predates the field.
fn default_factor() -> f64 {
    1.0
}

/// Identity of a symmetric **pair** of blocks — see [`Symmetry::Half`].
///
/// Its high 63 bits are a hash of what defines the pair; bit 0 tells the two
/// members apart. Only the producer that builds both halves mints one, so a
/// `Half` never arrives alone by accident.
///
/// ```
/// # use pyrucast::containers::matrix::{PairId, Symmetry};
/// // Two halves of one pair name the same identity and differ in its low bit.
/// let Symmetry::Half(id) = Symmetry::Half(42 as PairId) else { unreachable!() };
/// assert_eq!(id & !1, 42);
/// ```
pub type PairId = u64;

/// FNV-1a over `bytes`, folded into the running hash `h`.
///
/// Hand-rolled because the pair identity must be **deterministic across runs** —
/// it is serialized, and two saves of the same objects must give the same bytes
/// (`tests/archive.rs`). `DefaultHasher` guarantees no such stability, and a
/// random id would need a dependency this crate does not have.
fn fnv(h: u64, bytes: &[u8]) -> u64 {
    let mut h = h;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The seed every pair hash starts from (the FNV-1a offset basis).
pub(crate) const FNV_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// Fold a node list into a running pair hash.
pub(crate) fn hash_nodes(h: u64, nodes: &[NodeId]) -> u64 {
    nodes.iter().fold(h, |h, n| fnv(h, &n.0.to_le_bytes()))
}

/// Fold a name into a running pair hash.
pub(crate) fn hash_name(h: u64, name: &str) -> u64 {
    fnv(h, name.as_bytes())
}

/// Fold an `f64` into a running pair hash, by its bits.
pub(crate) fn hash_f64(h: u64, v: f64) -> u64 {
    fnv(h, &v.to_bits().to_le_bytes())
}

/// The two identities of one symmetric pair, from a hash of its content.
///
/// Handed back **together** because they only mean anything together: no
/// producer can obtain one half without the other, which is what keeps a lone
/// `Half` from ever being minted by hand.
///
/// `content` should fold in everything that distinguishes this pair from any
/// other — both supports' nodes, the variable names, any coefficient — through
/// [`hash_nodes`], [`hash_name`] and [`hash_f64`] starting from [`FNV_SEED`].
/// Two genuinely distinct pairs that hashed alike would be *rejected* by
/// [`Matrix::symmetric`], never wrongly accepted: the error is to forget a
/// symmetry, never to invent one.
pub(crate) fn mint_pair(content: u64) -> (PairId, PairId) {
    // Bit 0 is not a property of either block — it is a label saying "these two
    // are not the same one". Which member gets which is irrelevant; that they
    // differ is what the count below relies on.
    let base = content & !1;
    (base, base | 1)
}

/// What share of the matrix's symmetry a [`SubMatrix`] carries.
///
/// The property at stake is the assembled array's: `A[i][j] == A[j][i]` on the
/// CSR. A block **declares** its share and is believed — a model knows what it
/// writes, and nothing here verifies it.
///
/// Declaring correctly is therefore the producer's whole job, and the aggregate
/// adds up declarations without correcting them. In particular an **empty block
/// is symmetric** (it contributes nothing that could break `A = Aᵀ`), so it
/// declares [`Full`](Self::Full) — the rule has no exception for it.
///
/// ```
/// # use pyrucast::containers::matrix::Symmetry;
/// # use pyrucast::named::Named;
/// // Only the two nameable shares can be parsed: a half means nothing alone.
/// assert_eq!(Symmetry::parse("full")?, Symmetry::Full);
/// assert!(Symmetry::parse("half").is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Symmetry {
    /// This block is symmetric on its own — the usual case for a block whose
    /// row and column supports coincide (a stiffness, a mass, a Gram matrix).
    Full,
    /// This block carries **half** of a symmetry: its transpose is the other
    /// block sharing this [`PairId`]. Neither is symmetric alone — a Dirichlet's
    /// `C` and `Cᵀ` are rectangular — and only the aggregate can see whether
    /// both are present.
    Half(PairId),
    /// This block carries none of it.
    None,
}

impl crate::named::Named for Symmetry {
    const LABEL: &'static str = "symmetry";
    // `Half` is deliberately absent: a half cannot be named into existence, it
    // only means anything paired. Refusing "half" with `expected full|none` is
    // the right message for someone declaring a block by hand.
    const VALUES: &'static [Self] = &[Symmetry::Full, Symmetry::None];

    fn name(self) -> &'static str {
        match self {
            Symmetry::Full => "full",
            Symmetry::Half(_) => "half",
            Symmetry::None => "none",
        }
    }
}

///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let mut bloc = SubMatrix::new(
/// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #     DofOrdering::NodesThenVars, Symmetry::Full);
/// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// // A **block**: its row and column supports, its variables
/// // dual and primal, and its entries. Nothing lives in the aggregate.
/// assert_eq!((bloc.n_rows(), bloc.n_cols()), (2, 2));
/// assert_eq!(bloc.dual_vars(), &["q".to_string()]);
/// assert_eq!(bloc.get(a.id(), "q", a.id(), "T"), 2.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct SubMatrix {
    /// POI1 mesh: cell `k` holds the k-th row-support node.
    row_support: Handle<SubMesh>,
    /// POI1 mesh: cell `k` holds the k-th col-support node.
    col_support: Handle<SubMesh>,
    /// Row variable names (dual variables).
    dual_vars: Vec<String>,
    /// Column variable names (primal variables).
    primal_vars: Vec<String>,
    /// `(node_local, var_idx)` ↔ matrix index mapping.
    ordering: DofOrdering,
    /// COO data, sized `(n_row_nodes × n_dual_vars) × (n_col_nodes × n_primal_vars)`.
    #[serde(with = "coo_serde")]
    coo: CooMatrix<f64>,
    /// What share of the matrix's symmetry this block carries — **declared**
    /// by its producer, believed without verification. See [`Symmetry`].
    symmetry: Symmetry,
    /// `Some` ⇒ this is a **computed** block: `coo` is an empty placeholder and
    /// the contribution is produced on the fly by the global assembler from this
    /// recipe. `None` ⇒ **literal** block, `coo` holds the values (the historical
    /// behaviour, unchanged).
    #[serde(default)]
    recipe: Option<ComputedRecipe>,
    /// The set of [`Physics`] natures of the sub-model that produced this block,
    /// set by the assembler ([`crate::ops::matrix`]) for **both** the computed
    /// and the literal path (so a Dirichlet C/Cᵀ pair is tagged too). **Empty**
    /// for a block built directly, outside assembly (the « rien » case), or
    /// carrying several natures for a coupled physics. Consumed by
    /// [`Matrix::filter`](Matrix::filter).
    #[serde(default)]
    physics: Vec<Physics>,
    /// Lazy scalar scale applied to every value this block emits — at direct
    /// accessors (`get`, `dense`, …) and at global assembly (`build_global_triplets`,
    /// [`crate::ops::scatter`]) alike. Defaults to `1.0`; set via
    /// `Mul<f64>`/`Div<f64>` ([`std::ops::Mul`], [`std::ops::Div`]) rather than
    /// eagerly rewriting `coo`, so it works for a **computed** block too (its
    /// values don't exist until assembly evaluates the recipe). Never touches
    /// `local_coo_arrays`/`local_triplets`, which stay raw — every consumer of
    /// those applies the factor itself.
    #[serde(default = "default_factor")]
    factor: f64,
}

impl SubMatrix {
    /// Build a new block.
    ///
    /// `row_support` / `col_support` must be POI1 sub-meshes whose cells
    /// define the row/col node sequence, and whose nodes are **distinct** —
    /// what `to_poi1` produces, and what every physics hands over. A repeated
    /// node is not rejected but addresses the wrong row: the block looks
    /// `node_local` up through [`SubMesh::node_index`], a deduplicated rank
    /// that parts from the flat position as soon as a node appears twice.
    /// `dual_vars` / `primal_vars` are the row/column variable names; they must
    /// be non-empty.
    ///
    /// Both supports are sealed here: the block reads their connectivity in
    /// place ever after, it keeps no copy.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // The block knows its two POI1 supports and its two sets of variables:
    /// // its dimensions follow, nodes × variables on each side.
    /// assert_eq!((bloc.n_rows(), bloc.n_cols()), (2, 2));
    /// ```
    pub fn new(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetry: Symmetry,
    ) -> Self {
        // The block's row/col numbering *is* these supports' connectivity, read
        // in place on every access rather than copied; freeze them so it holds.
        crate::containers::mesh::seal(&row_support);
        crate::containers::mesh::seal(&col_support);
        let nrows = row_support.read().connectivity().len() * dual_vars.len();
        let ncols = col_support.read().connectivity().len() * primal_vars.len();
        Self {
            row_support,
            col_support,
            dual_vars,
            primal_vars,
            ordering,
            coo: CooMatrix::new(nrows, ncols),
            symmetry,
            recipe: None,
            physics: Vec::new(),
            factor: 1.0,
        }
    }

    /// Build a **computed** block: a sized-but-empty placeholder that carries a
    /// [`ComputedRecipe`] instead of values. Its structure (supports, vars,
    /// ordering, dimensions) is fully defined; the values are produced by the
    /// global assembler, which drives `submodel`'s element kernel over
    /// `recipe.fespace` and scatters straight into the global matrix.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::ElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::{ComputedRecipe, DofOrdering, KernelInputs, SubMatrix, Symmetry};
    /// # use pyrucast::models::MatrixKind;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::element_field;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let modele = model::heat_conduction(&fes).unwrap();
    /// # let mat = element_field::material_field(&modele, &[("k", 1.0)]).unwrap();
    /// // The same seen from the constructor: a sized, empty template,
    /// // whose values will come from the sub-model's kernel.
    /// let z = SubMatrix::computed(
    ///     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    ///     DofOrdering::NodesThenVars, Symmetry::Full,
    ///     ComputedRecipe {
    ///         submodel: modele.get(0)?,
    ///         fespaces: vec![zone.clone()],
    ///         material: Some(mat.get(0)?),
    ///         kind: MatrixKind::Stiffness,
    ///         inputs: KernelInputs::MaterialOnly,
    ///         col_fespaces: Vec::new(),
    ///     });
    /// assert!(z.recipe().is_some());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn computed(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetry: Symmetry,
        recipe: ComputedRecipe,
    ) -> Self {
        // The block's row/col numbering *is* these supports' connectivity, read
        // in place on every access rather than copied; freeze them so it holds.
        crate::containers::mesh::seal(&row_support);
        crate::containers::mesh::seal(&col_support);
        let nrows = row_support.read().connectivity().len() * dual_vars.len();
        let ncols = col_support.read().connectivity().len() * primal_vars.len();
        Self {
            row_support,
            col_support,
            dual_vars,
            primal_vars,
            ordering,
            coo: CooMatrix::new(nrows, ncols),
            symmetry,
            recipe: Some(recipe),
            physics: Vec::new(),
            factor: 1.0,
        }
    }

    /// Build a block from an already-assembled COO whose indices are this
    /// block's **local** numbering (`(node_local, var_idx)` via `ordering`,
    /// nodes positioned as in `row_support` / `col_support`). Lets an assembler
    /// produce all entries in parallel and hand them over in one shot, bypassing
    /// the per-entry [`add_entry`](Self::add_entry).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use nalgebra_sparse::CooMatrix;
    /// // The assembler's path: produce every entry in parallel,
    /// // then hand them back in one go, without going through `add_entry`.
    /// let mut coo = CooMatrix::new(2, 2);
    /// coo.push(0, 0, 2.0);
    /// coo.push(1, 1, 2.0);
    /// let z = SubMatrix::from_coo(
    ///     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    ///     DofOrdering::NodesThenVars, Symmetry::Full, coo)?;
    /// assert_eq!(z.get(a.id(), "q", a.id(), "T"), 2.0);
    /// // Indices are **local** to the block, and its size must match.
    /// assert!(SubMatrix::from_coo(
    ///     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    ///     DofOrdering::NodesThenVars, Symmetry::Full, CooMatrix::new(3, 3)).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn from_coo(
        row_support: Handle<SubMesh>,
        col_support: Handle<SubMesh>,
        dual_vars: Vec<String>,
        primal_vars: Vec<String>,
        ordering: DofOrdering,
        symmetry: Symmetry,
        coo: CooMatrix<f64>,
    ) -> Result<Self> {
        // The block's row/col numbering *is* these supports' connectivity, read
        // in place on every access rather than copied; freeze them so it holds.
        crate::containers::mesh::seal(&row_support);
        crate::containers::mesh::seal(&col_support);
        let nrows = row_support.read().connectivity().len() * dual_vars.len();
        let ncols = col_support.read().connectivity().len() * primal_vars.len();
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
            dual_vars,
            primal_vars,
            ordering,
            coo,
            symmetry,
            recipe: None,
            physics: Vec::new(),
            factor: 1.0,
        })
    }

    /// Whether this is a **computed** block (carries a [`ComputedRecipe`], no
    /// stored values) rather than a literal one.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // A hand-built block carries its values; a *computed* block carries
    /// // a recipe the assembler unwinds.
    /// assert!(!bloc.is_computed());
    /// ```
    pub fn is_computed(&self) -> bool {
        self.recipe.is_some()
    }

    /// The block's [`ComputedRecipe`], or `None` for a literal block.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert!(bloc.recipe().is_none()); // not a recipe: values
    /// ```
    pub fn recipe(&self) -> Option<&ComputedRecipe> {
        self.recipe.as_ref()
    }

    /// The set of [`Physics`] natures of the sub-model that produced this block —
    /// **empty** for a block built outside assembly (the « rien » case), one entry
    /// for a plain physics, several for a coupled one. Set by the assembler on
    /// every block it emits (see [`crate::ops::matrix`]).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// # use pyrucast::models::Physics;
    /// // A hand-built block has no kind: `Matrix::filter` will
    /// // select it by none. Tagging it `Other` makes it reachable.
    /// assert!(bloc.physics().is_empty());
    /// bloc.set_physics(vec![Physics::Other]);
    /// assert_eq!(bloc.physics(), &[Physics::Other]);
    /// ```
    pub fn physics(&self) -> &[Physics] {
        &self.physics
    }

    /// Tag this block with the [`Physics`] nature set of its producing sub-model —
    /// the assembler calls this on each emitted block so [`Matrix::filter`] can
    /// select by nature (matched by containment).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::models::Physics;
    /// // The kind travels with the block up to the assembled matrix:
    /// // this is what `Matrix::filter` reads.
    /// let mut z = bloc();
    /// z.set_physics(vec![Physics::Thermal]);
    /// assert_eq!(z.physics(), &[Physics::Thermal]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_physics(&mut self, physics: Vec<Physics>) {
        self.physics = physics;
    }

    /// The scalar factor applied to every value this block emits (`1.0` unless
    /// scaled via `Mul<f64>`/`Div<f64>`) — see the struct-level field doc for
    /// exactly where it is and isn't applied.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // The block's scale factor, applied at assembly — what
    /// // `&matrix / dt` sets without rewriting a single value.
    /// assert_eq!(bloc.factor(), 1.0);
    /// ```
    pub fn factor(&self) -> f64 {
        self.factor
    }

    /// Whether the assembler declared this block numerically symmetric.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert!(bloc.is_symmetric()); // declared at construction
    /// ```
    pub fn is_symmetric(&self) -> bool {
        matches!(self.symmetry, Symmetry::Full)
    }

    /// What share of the matrix's symmetry this block declares.
    ///
    /// [`is_symmetric`](Self::is_symmetric) answers the yes/no question about
    /// this block alone; this one also tells a half from nothing, which only
    /// [`Matrix::symmetric`] can resolve.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone()]).unwrap().get(0).unwrap();
    /// # let bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// assert_eq!(bloc.symmetry(), Symmetry::Full);
    /// ```
    pub fn symmetry(&self) -> Symmetry {
        self.symmetry
    }

    /// Number of row DOFs = `n_row_nodes × n_dual_vars`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // nodes of the row support × dual variables.
    /// assert_eq!(bloc.n_rows(), 2);
    /// ```
    pub fn n_rows(&self) -> usize {
        self.coo.nrows()
    }

    /// Number of column DOFs = `n_col_nodes × n_primal_vars`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.n_cols(), 2);
    /// ```
    pub fn n_cols(&self) -> usize {
        self.coo.ncols()
    }

    /// Number of COO triplets stored (counting duplicates at the same
    /// `(row, col)`).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.entry_count(), 4); // the four `add_entry` of the setup
    /// ```
    pub fn entry_count(&self) -> usize {
        self.coo.nnz()
    }

    /// Row variable names (dual variables).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.dual_vars(), &["q".to_string()]); // côté lignes
    /// ```
    pub fn dual_vars(&self) -> &[String] {
        &self.dual_vars
    }

    /// Column variable names (primal variables).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.primal_vars(), &["T".to_string()]); // côté colonnes
    /// ```
    pub fn primal_vars(&self) -> &[String] {
        &self.primal_vars
    }

    /// DOF ordering strategy.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // The DOF ordering inside the block: nodes first, or variables first.
    /// assert_eq!(bloc.ordering(), DofOrdering::NodesThenVars);
    /// ```
    pub fn ordering(&self) -> DofOrdering {
        self.ordering
    }

    /// Handle to the row-support POI1 sub-mesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::handle::Handle as H;
    /// // The **row** support — the dual variables live there.
    /// assert!(H::same_object(bloc().row_support(), &support));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn row_support(&self) -> &Handle<SubMesh> {
        &self.row_support
    }

    /// Handle to the col-support POI1 sub-mesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::handle::Handle as H;
    /// // The **column** one. It differs from the former as soon as the block
    /// // couples two meshes — a Lagrange block, an interface law.
    /// assert!(H::same_object(bloc().col_support(), &support));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn col_support(&self) -> &Handle<SubMesh> {
        &self.col_support
    }

    /// Union of `dual_vars` and `primal_vars`, in that order, without
    /// duplicates.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // The union of both sets, deduplicated.
    /// assert_eq!(bloc.field_names(), vec!["q".to_string(), "T".to_string()]);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// let lignes = bloc.row_dofs();
    /// assert_eq!(lignes.len(), bloc.n_rows());
    /// assert_eq!(lignes[0], (a.id(), "q".to_string()));
    /// ```
    pub fn row_dofs(&self) -> Vec<(NodeId, String)> {
        self.row_dofs_with(&self.row_support.read())
    }

    /// [`row_dofs`](Self::row_dofs) reading a support guard the caller already
    /// holds — the zero-copy form (see `book/src/developper/parallelisme.md`).
    pub(crate) fn row_dofs_with(&self, row_support: &SubMesh) -> Vec<(NodeId, String)> {
        let nodes = row_support.connectivity();
        let n_nodes = nodes.len();
        let n_vars = self.dual_vars.len();
        (0..self.coo.nrows())
            .map(|i| {
                let (nl, vi) = self.ordering.from_index(i, n_nodes, n_vars);
                (nodes[nl], self.dual_vars[vi].clone())
            })
            .collect()
    }

    /// All column DOFs in matrix-column order: `(NodeId, var_name)` for
    /// each column index `0..n_cols`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.col_dofs()[0], (a.id(), "T".to_string()));
    /// ```
    pub fn col_dofs(&self) -> Vec<(NodeId, String)> {
        self.col_dofs_with(&self.col_support.read())
    }

    /// [`col_dofs`](Self::col_dofs) reading a support guard the caller already
    /// holds.
    pub(crate) fn col_dofs_with(&self, col_support: &SubMesh) -> Vec<(NodeId, String)> {
        let nodes = col_support.connectivity();
        let n_nodes = nodes.len();
        let n_vars = self.primal_vars.len();
        (0..self.coo.ncols())
            .map(|i| {
                let (nl, vi) = self.ordering.from_index(i, n_nodes, n_vars);
                (nodes[nl], self.primal_vars[vi].clone())
            })
            .collect()
    }

    /// Row DOFs as packed [`DofKey`]s, in the same order as [`row_dofs`] —
    /// the allocation-free twin the assembler numbers with.
    ///
    /// `slot_of` names each variable by its index in the aggregate's table
    /// ([`Matrix::dof_vars`]); it is resolved **once per block**, so no DOF ever
    /// touches a string.
    pub(crate) fn row_dof_keys(&self, slot_of: &HashMap<String, u32>) -> Vec<DofKey> {
        self.row_dof_keys_with(&self.row_support.read(), slot_of)
    }

    /// [`row_dof_keys`](Self::row_dof_keys) reading a support guard the caller
    /// already holds — the zero-copy form (see `book/src/developper/parallelisme.md`).
    pub(crate) fn row_dof_keys_with(
        &self,
        row_support: &SubMesh,
        slot_of: &HashMap<String, u32>,
    ) -> Vec<DofKey> {
        Self::keys(
            row_support.connectivity(),
            &self.var_slots(&self.dual_vars, slot_of),
            self.ordering,
            self.coo.nrows(),
        )
    }

    /// Column DOFs as packed [`DofKey`]s — the column twin of
    /// [`row_dof_keys`](Self::row_dof_keys).
    pub(crate) fn col_dof_keys(&self, slot_of: &HashMap<String, u32>) -> Vec<DofKey> {
        self.col_dof_keys_with(&self.col_support.read(), slot_of)
    }

    /// [`col_dof_keys`](Self::col_dof_keys) reading a support guard the caller
    /// already holds.
    pub(crate) fn col_dof_keys_with(
        &self,
        col_support: &SubMesh,
        slot_of: &HashMap<String, u32>,
    ) -> Vec<DofKey> {
        Self::keys(
            col_support.connectivity(),
            &self.var_slots(&self.primal_vars, slot_of),
            self.ordering,
            self.coo.ncols(),
        )
    }

    /// This block's variable names as slots in the aggregate's table.
    ///
    /// Infallible by construction: `slot_of` is always built from the
    /// aggregate's own [`field_names`](Matrix::field_names), which is the union
    /// of every block's dual and primal variables — so a block looks its own
    /// names up in a table that was made to contain them. A miss would mean the
    /// caller built the table from something else, which is a bug here and not
    /// a condition a user can produce.
    fn var_slots(&self, vars: &[String], slot_of: &HashMap<String, u32>) -> Vec<u32> {
        vars.iter()
            .map(|v| {
                *slot_of.get(v).unwrap_or_else(|| {
                    panic!("variable '{v}' is absent from the matrix name table")
                })
            })
            .collect()
    }

    /// `n` DOF keys in matrix order, from a node list and its variables' slots.
    fn keys(nodes: &[NodeId], slots: &[u32], ordering: DofOrdering, n: usize) -> Vec<DofKey> {
        let (n_nodes, n_vars) = (nodes.len(), slots.len());
        (0..n)
            .map(|i| {
                let (nl, vi) = ordering.from_index(i, n_nodes, n_vars);
                dof_key(nodes[nl], slots[vi])
            })
            .collect()
    }

    /// Append an entry at `(row_node, row_var) × (col_node, col_var)`.
    ///
    /// Returns an error if either `row_node` / `col_node` is not in its
    /// respective support, or if the variable name is not in the
    /// `dual_vars` / `primal_vars` list.  Repeated calls at the same
    /// `(row, col)` accumulate.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// let mut z = SubMatrix::new(
    ///     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    ///     DofOrdering::NodesThenVars, Symmetry::Full);
    /// // Two calls at the same place **accumulate** — that is what lets the
    /// // assembly pour cell by cell without reading anything back.
    /// z.add_entry(a.id(), "q", a.id(), "T", 1.0)?;
    /// z.add_entry(a.id(), "q", a.id(), "T", 1.0)?;
    /// assert_eq!(z.get(a.id(), "q", a.id(), "T"), 2.0);
    /// // A node outside the support, or an undeclared variable, is an error.
    /// assert!(z.add_entry(a.id(), "f_x", a.id(), "T", 1.0).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn add_entry(
        &mut self,
        row_node: NodeId,
        row_var: &str,
        col_node: NodeId,
        col_var: &str,
        value: f64,
    ) -> Result<()> {
        // The supports' own `NodeId → position` tables, borrowed in place. Rows
        // and columns usually share one handle, so ask that lock only once.
        // The computed-block refusal lives in `add_entry_with`, tested once here
        // rather than twice.
        let (row_g, col_g) = self.support_guards();
        let row = &row_g;
        let col = col_g.as_deref().unwrap_or(&row_g);
        self.add_entry_with(row, col, row_node, row_var, col_node, col_var, value)
    }

    /// [`add_entry`](Self::add_entry) reading support guards the caller already
    /// holds — the zero-copy form (see `book/src/developper/parallelisme.md`).
    /// Pass the same reference twice for a square block.
    ///
    /// A caller filling a block in a loop should hold the guards once and use
    /// this, rather than let every entry re-lock the supports.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_entry_with(
        &mut self,
        row_support: &SubMesh,
        col_support: &SubMesh,
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
        let n_rn = row_support.connectivity().len();
        let n_dv = self.dual_vars.len();
        let n_cn = col_support.connectivity().len();
        let n_pv = self.primal_vars.len();

        let rnl = *row_support.node_index().get(&row_node).ok_or_else(|| {
            PyrucastError::Message(format!(
                "add_entry: row node {row_node:?} not in row_support"
            ))
        })?;
        let rvi = self
            .dual_vars
            .iter()
            .position(|v| v == row_var)
            .ok_or_else(|| {
                PyrucastError::Message(format!("add_entry: row var '{row_var}' not in dual_vars"))
            })?;
        let cnl = *col_support.node_index().get(&col_node).ok_or_else(|| {
            PyrucastError::Message(format!(
                "add_entry: col node {col_node:?} not in col_support"
            ))
        })?;
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

    /// COO entries in **local** index form `(row, col, value)` — the block's own
    /// numbering. Used by the aggregate to scatter into the global matrix via a
    /// per-block translation table. **Not** scaled by [`SubMatrix::factor`]: the
    /// caller applies it (every consumer inside `containers`/`ops::matrix` does).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// // The entries in **local** numbering, as the aggregate
    /// // pours into the global matrix through its translation table.
    /// let z = bloc();
    /// let t: Vec<_> = z.local_triplets().collect();
    /// assert_eq!(t.len(), 4);
    /// assert_eq!(t[0], (0, 0, 2.0));
    /// // They do **not** carry the factor: the caller applies it.
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn local_triplets(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.coo.triplet_iter().map(|(r, c, &v)| (r, c, v))
    }

    /// The block's COO as raw parallel slices `(rows, cols, values)`, in
    /// **local** index form. Same data as [`local_triplets`](Self::local_triplets)
    /// but indexable, so the aggregate can remap the entries in parallel. **Not**
    /// scaled by [`SubMatrix::factor`] — see [`local_triplets`](Self::local_triplets).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // The three COO arrays **local** to the block, as the assembler
    /// // scatters into the global pattern.
    /// let (lignes, colonnes, valeurs) = bloc.local_coo_arrays();
    /// assert_eq!((lignes.len(), colonnes.len(), valeurs.len()), (4, 4, 4));
    /// ```
    pub fn local_coo_arrays(&self) -> (&[usize], &[usize], &[f64]) {
        (
            self.coo.row_indices(),
            self.coo.col_indices(),
            self.coo.values(),
        )
    }

    /// Handle to the `Coords` backing this block's row support (the col support
    /// shares it in any assembled system).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// # use pyrucast::handle::Handle as H;
    /// assert!(H::same_object(&bloc.coords(), &coords));
    /// ```
    pub fn coords(&self) -> Handle<Coords> {
        self.row_support.read().coords()
    }

    /// Sum of all entries at `(row_node, row_var) × (col_node, col_var)`.
    /// Returns `0.0` if the DOF pair is unknown or has no entry.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.get(a.id(), "q", b.id(), "T"), -1.0);
    /// // An absent coordinate returns **zero**, not an error: reading
    /// // a sparse block does not tell "zero" from "outside the pattern".
    /// assert_eq!(bloc.get(a.id(), "q", a.id(), "absente"), 0.0);
    /// ```
    pub fn get(&self, row_node: NodeId, row_var: &str, col_node: NodeId, col_var: &str) -> f64 {
        let (row_g, col_g) = self.support_guards();
        self.get_with(
            &row_g,
            col_g.as_deref().unwrap_or(&row_g),
            row_node,
            row_var,
            col_node,
            col_var,
        )
    }

    /// Read guards on both supports, taking a **single** one when rows and
    /// columns share the handle — the square case every physics produces.
    /// Asking the same `RwLock` twice in one thread is what this avoids.
    fn support_guards(&self) -> (ReadGuard<SubMesh>, Option<ReadGuard<SubMesh>>) {
        let row = self.row_support.read();
        let col = if self.col_support.same_object(&self.row_support) {
            None
        } else {
            Some(self.col_support.read())
        };
        (row, col)
    }

    /// [`get`](Self::get) reading support guards the caller already holds — the
    /// zero-copy form (see `book/src/developper/parallelisme.md`). Pass the same
    /// reference twice when the block is square on one support.
    pub(crate) fn get_with(
        &self,
        row_support: &SubMesh,
        col_support: &SubMesh,
        row_node: NodeId,
        row_var: &str,
        col_node: NodeId,
        col_var: &str,
    ) -> f64 {
        let n_rn = row_support.connectivity().len();
        let n_dv = self.dual_vars.len();
        let n_cn = col_support.connectivity().len();
        let n_pv = self.primal_vars.len();

        // `node_index` is the support's own `NodeId → position` table, built
        // once and shared by every consumer — first occurrence wins, as the
        // linear `position` scan this replaces did.
        let rnl = match row_support.node_index().get(&row_node) {
            Some(&i) => i,
            None => return 0.0,
        };
        let rvi = match self.dual_vars.iter().position(|v| v == row_var) {
            Some(i) => i,
            None => return 0.0,
        };
        let cnl = match col_support.node_index().get(&col_node) {
            Some(&i) => i,
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // Insertion order preserved; variable names already resolved.
    /// assert_eq!(bloc.iter_entries().len(), 4);
    /// assert_eq!(bloc.iter_entries()[0].4, 2.0);
    /// ```
    pub fn iter_entries(&self) -> Vec<MatrixEntry> {
        let (row_g, col_g) = self.support_guards();
        self.iter_entries_with(&row_g, col_g.as_deref().unwrap_or(&row_g))
    }

    /// [`iter_entries`](Self::iter_entries) reading support guards the caller
    /// already holds. Pass the same reference twice for a square block.
    pub(crate) fn iter_entries_with(
        &self,
        row_support: &SubMesh,
        col_support: &SubMesh,
    ) -> Vec<MatrixEntry> {
        let row_nodes = row_support.connectivity();
        let col_nodes = col_support.connectivity();
        let n_rn = row_nodes.len();
        let n_dv = self.dual_vars.len();
        let n_cn = col_nodes.len();
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
                    row_nodes[rnl],
                    self.dual_vars[rvi].clone(),
                    col_nodes[cnl],
                    self.primal_vars[cvi].clone(),
                    v * self.factor,
                )
            })
            .collect()
    }

    /// Materialise as a row-major dense buffer of length `n_rows × n_cols`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.dense(), vec![2.0, -1.0, -1.0, 2.0]); // ligne-major
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.to_dmatrix()[(0, 1)], -1.0);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.to_coo().nnz(), 4);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.to_csr().nnz(), 4);
    /// ```
    pub fn to_csr(&self) -> CsrMatrix<f64> {
        CsrMatrix::from(&self.to_coo())
    }

    /// Convert this block to a [`nalgebra_sparse::CscMatrix`] (factor applied).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// assert_eq!(bloc.to_csc().nrows(), 2);
    /// ```
    pub fn to_csc(&self) -> CscMatrix<f64> {
        CscMatrix::from(&self.to_coo())
    }

    /// `y = A · x` (dense). Returns an error if `x.len() != n_cols`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, SubMatrix, Symmetry};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = mesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap().get(0).unwrap();
    /// # let mut bloc = SubMatrix::new(
    /// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #     DofOrdering::NodesThenVars, Symmetry::Full);
    /// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// # bloc.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// // [[2, -1], [-1, 2]] · [1, 1] = [1, 1].
    /// assert_eq!(bloc.mul_dense(&[1.0, 1.0]).unwrap(), vec![1.0, 1.0]);
    /// ```
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

/// Reject a divisor that no matrix can survive.
///
/// A scalar operator yields the value type — as it does for every number in
/// `std` — so `Div<f64>` has no fallible form to return. A divisor that would
/// make every value non-finite therefore stops here, rather than seeding an
/// `inf` that surfaces as a `NaN` inside the solver several calls later. The
/// Python surface, where a zero can actually arrive from user data, raises
/// `ZeroDivisionError` instead (`crate::py::matrix`).
#[track_caller]
fn check_divisor(rhs: f64) -> f64 {
    assert!(
        rhs != 0.0 && rhs.is_finite(),
        "matrix division by {rhs}: every value of the result would be non-finite"
    );
    rhs
}

// ─── SubMatrix scalar operators ─────────────────────────────────────────────
//
// `blk * s` / `blk / s` / `-blk` only touch `factor` — never the stored `coo` —
// so they are zero-copy and work identically for a literal block (values live
// in `coo`) and a computed one (values don't exist until assembly evaluates the
// recipe). No `Add`/`Sub<f64>`: shifting a matrix by a constant has no physical
// meaning. Block **plus** block, on the other hand, is a sum of contributions
// and yields a two-block `Matrix` — see the aggregate operators.

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

// Scalar on the left — `2.0 * blk` reads as the mathematics does. `f64` is a
// foreign type, so these live here rather than as a blanket impl.
impl std::ops::Mul<SubMatrix> for f64 {
    type Output = SubMatrix;
    fn mul(self, rhs: SubMatrix) -> SubMatrix {
        rhs * self
    }
}

impl std::ops::Mul<&SubMatrix> for f64 {
    type Output = SubMatrix;
    fn mul(self, rhs: &SubMatrix) -> SubMatrix {
        rhs * self
    }
}

impl std::ops::Div<f64> for SubMatrix {
    type Output = SubMatrix;
    #[track_caller]
    fn div(mut self, rhs: f64) -> SubMatrix {
        self.factor /= check_divisor(rhs);
        self
    }
}

impl std::ops::Div<f64> for &SubMatrix {
    type Output = SubMatrix;
    #[track_caller]
    fn div(self, rhs: f64) -> SubMatrix {
        self.clone() / rhs
    }
}

impl std::ops::Neg for SubMatrix {
    type Output = SubMatrix;
    fn neg(self) -> SubMatrix {
        self * -1.0
    }
}

impl std::ops::Neg for &SubMatrix {
    type Output = SubMatrix;
    fn neg(self) -> SubMatrix {
        self * -1.0
    }
}

impl fmt::Debug for SubMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubMatrix")
            .field("n_rows", &self.coo.nrows())
            .field("n_cols", &self.coo.ncols())
            .field("entries", &self.coo.nnz())
            .field("symmetry", &self.symmetry)
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
            let tags: Vec<&str> = self.physics.iter().map(|p| p.name()).collect();
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
            if self.is_symmetric() {
                ", symmetric"
            } else {
                ""
            },
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
        // What `Display` does not say — the DOF ordering decides how to read the
        // grid that follows, and the factor decides its values. Without this
        // line, the "content" level would teach less than the structure does.
        format!(
            "{self}\n  symmetry: {}, ordering: {:?}, factor: {:?}\n  dual_vars: [{}], primal_vars: [{}]\n{}",
            crate::named::Named::name(self.symmetry),
            self.ordering,
            self.factor,
            self.dual_vars.join(", "),
            self.primal_vars.join(", "),
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

/// The assembled CSR, its **index arrays shared** with whatever produced them.
///
/// A `CsrMatrix` owns its three arrays, so storing one here meant copying the
/// sparsity out of the assembler's (already shared, already memoised) pattern at
/// every assembly — twenty gigabytes of `memcpy` on a solid mesh, to reproduce
/// something that had not changed. Only the values are new from one assembly to
/// the next; the offsets and the column indices are the pattern's, held by
/// `Arc`. A `CsrMatrix` is built on demand, where one is actually wanted.
pub(crate) struct AssembledCsr {
    pub(crate) row_offsets: std::sync::Arc<Vec<usize>>,
    pub(crate) col_indices: std::sync::Arc<Vec<usize>>,
    pub(crate) values: Vec<f64>,
    pub(crate) ncols: usize,
}

impl AssembledCsr {
    fn nrows(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    /// Slot of `(r, c)` in `values`, or `None` when the pattern has no such
    /// entry. The columns of a row are sorted, so this is a binary search.
    fn slot(&self, r: usize, c: usize) -> Option<usize> {
        let (lo, hi) = (*self.row_offsets.get(r)?, *self.row_offsets.get(r + 1)?);
        self.col_indices[lo..hi]
            .binary_search(&c)
            .ok()
            .map(|k| lo + k)
    }

    /// Every stored entry as `(row, col, value)`, in CSR order.
    fn triplets(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.nrows()).flat_map(move |r| {
            let (lo, hi) = (self.row_offsets[r], self.row_offsets[r + 1]);
            (lo..hi).map(move |k| (r, self.col_indices[k], self.values[k]))
        })
    }

    /// Materialise a `CsrMatrix` — the one place the index arrays are copied,
    /// and only for a caller that asked for that type.
    fn to_csr(&self) -> Result<CsrMatrix<f64>> {
        CsrMatrix::try_from_csr_data(
            self.nrows(),
            self.ncols,
            (*self.row_offsets).clone(),
            (*self.col_indices).clone(),
            self.values.clone(),
        )
        .map_err(|e| PyrucastError::Message(format!("to_csr: invalid CSR: {e}")))
    }
}

/// Snapshot produced by [`Matrix::finalize`]: DOF numbering + assembled CSR.
///
/// The numbering is held as packed [`DofKey`]s over a shared name table, not as
/// a `Vec<(NodeId, String)>`: on a solid mesh the materialised form is thirty
/// million `String`s, rebuilt at every assembly, for a handful of distinct
/// names.
struct AssembledData {
    vars: std::sync::Arc<Vec<String>>,
    /// Shared with the pattern that produced them, like the sparsity beside
    /// them: the numbering is the pattern's, and an assembled matrix borrows it
    /// rather than copying eight bytes per degree of freedom out of it at every
    /// assembly — twice, rows and columns.
    row_keys: std::sync::Arc<Vec<DofKey>>,
    col_keys: std::sync::Arc<Vec<DofKey>>,
    csr: AssembledCsr,
}

/// Aggregate of [`SubMatrix`] blocks.
///
/// Call [`Matrix::finalize`] before passing to a solver. Solver-facing methods
/// (`to_csr`, `to_dmatrix`, `mul_dense`, `dense`, `to_coo`, `to_csc`) return
/// an error if the matrix has not been finalized. `add_sub` invalidates the
/// assembled state.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let mut bloc = SubMatrix::new(
/// #     support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #     DofOrdering::NodesThenVars, Symmetry::Full);
/// # bloc.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// # bloc.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// // The aggregate: blocks, a global numbering, an assembled state.
/// let mut k = Matrix::empty();
/// k.add_sub(Handle::new(bloc))?;
/// k.finalize()?;
/// assert_eq!((k.n_rows()?, k.n_cols()?), (2, 2));
/// assert_eq!(k.to_csr()?.nnz(), 2);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
        let sym = self.symmetric();
        Some(format!(
            ", {} row(s) × {} col(s){}",
            n_rows,
            n_cols,
            if sym { ", symmetric" } else { "" }
        ))
    }
});

/// One row or column DOF of an aggregate [`Matrix`], in materialised form.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let bloc = || {
/// #     let mut z = SubMatrix::new(
/// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #         DofOrdering::NodesThenVars, Symmetry::Full);
/// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// #     z
/// # };
/// # use pyrucast::containers::matrix::NamedDof;
/// let mut k = Matrix::empty();
/// k.add_sub(Handle::new(bloc()))?;
/// k.finalize()?;
/// // A row or column DOF of the aggregate, in materialized form.
/// let dofs: Vec<NamedDof> = k.row_dofs()?;
/// assert_eq!(dofs[0], (a.id(), "q".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub type NamedDof = (NodeId, String);

/// A DOF in **packed** form: the node id in the high 32 bits, the index of its
/// variable in the matrix's name table in the low 32.
///
/// The materialised [`NamedDof`] carries a `String` per DOF. That is one heap
/// allocation, one hash of a string and one clone for every degree of freedom
/// of the problem — thirty million of each on a solid mesh, paid again at every
/// assembly, to re-express what a handful of variable names already say. The
/// key says the same thing in eight bytes, and the names live once in the
/// table the key indexes ([`Matrix::dof_vars`]).
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::containers::matrix::{dof_key, dof_node, dof_var, DofKey};
/// // Node on top, variable below: two integers in one `u64`.
/// let k: DofKey = dof_key(NodeId(7), 2);
/// assert_eq!(dof_node(k), NodeId(7));
/// assert_eq!(dof_var(k), 2);
/// // Key order is nodes first, then variables — a sort by
/// // key therefore lays the DOFs out node by node.
/// assert!(dof_key(NodeId(7), 2) < dof_key(NodeId(8), 0));
/// ```
pub type DofKey = u64;

/// Pack `(node, var_slot)` into a [`DofKey`].
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::containers::matrix::{dof_key, dof_node, dof_var};
/// // Node on top, variable below: both fit in a single `u64`.
/// let k = dof_key(NodeId(7), 2);
/// assert_eq!((dof_node(k), dof_var(k)), (NodeId(7), 2));
/// ```
#[inline]
pub fn dof_key(node: NodeId, var_slot: u32) -> DofKey {
    ((node.0 as u64) << 32) | var_slot as u64
}

/// The node of a [`DofKey`].
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::containers::matrix::{dof_key, dof_node};
/// assert_eq!(dof_node(dof_key(NodeId(7), 2)), NodeId(7));
/// ```
#[inline]
pub fn dof_node(key: DofKey) -> NodeId {
    NodeId((key >> 32) as u32)
}

/// The variable slot of a [`DofKey`] — an index into the matrix's name table
/// ([`Matrix::dof_vars`]).
///
/// ```
/// # use pyrucast::atoms::NodeId;
/// # use pyrucast::containers::matrix::{dof_key, dof_var};
/// assert_eq!(dof_var(dof_key(NodeId(7), 2)), 2);
/// ```
#[inline]
pub fn dof_var(key: DofKey) -> u32 {
    (key & 0xffff_ffff) as u32
}

/// A **seen-set** over DOF keys, direct-addressed when the node ids are dense
/// enough to make that cheaper than hashing.
///
/// Node ids index `Coords` directly, so `node × n_vars + var` is a perfect hash
/// whenever the id space is not riddled with holes. When it is — a long-lived
/// `Coords` that has seen many deletions — the flat table would dwarf the DOF
/// set it describes, and a `HashSet` of the (already compact) keys is the
/// honest fallback.
enum DofSeen {
    /// One flag per `(node, var)` slot: `node × n_vars + var`.
    Dense {
        flags: Vec<bool>,
        n_vars: usize,
    },
    Sparse(std::collections::HashSet<DofKey>),
}

impl DofSeen {
    /// Sized for `n_vars` variables over node ids up to `max_node`, holding
    /// `n_dofs` DOFs. Goes dense unless the flat table would cost more than
    /// eight flags per DOF actually stored.
    fn new(max_node: u32, n_vars: usize, n_dofs: usize) -> Self {
        let slots = (max_node as usize + 1).saturating_mul(n_vars.max(1));
        if slots <= n_dofs.saturating_mul(8).max(1024) {
            DofSeen::Dense {
                flags: vec![false; slots],
                n_vars: n_vars.max(1),
            }
        } else {
            DofSeen::Sparse(std::collections::HashSet::with_capacity(n_dofs))
        }
    }

    /// Record `key`; `true` the first time it is seen.
    #[inline]
    fn insert(&mut self, key: DofKey) -> bool {
        match self {
            DofSeen::Dense { flags, n_vars } => {
                let i = dof_node(key).0 as usize * *n_vars + dof_var(key) as usize;
                let fresh = !flags[i];
                flags[i] = true;
                fresh
            }
            DofSeen::Sparse(set) => set.insert(key),
        }
    }
}

/// Spell a packed key for an error message — `"node 9 · imposed_T"`.
fn dof_text(key: DofKey, vars: &[String]) -> String {
    format!("node {} · {}", dof_node(key).0, vars[dof_var(key) as usize])
}

/// Record one conjugate pair in the two global orders.
///
/// The two sides advance **together** or not at all. A DOF that is new on one
/// side and already placed on the other means the same dual DOF has been
/// declared conjugate to two different primal ones: a contradiction between
/// blocks, reported rather than papered over — the reader counts, it does not
/// correct.
fn push_conjugate(
    seen_row: &mut DofSeen,
    seen_col: &mut DofSeen,
    out_row: &mut Vec<DofKey>,
    out_col: &mut Vec<DofKey>,
    vars: &[String],
    r: DofKey,
    c: DofKey,
) -> Result<()> {
    let fresh_row = seen_row.insert(r);
    let fresh_col = seen_col.insert(c);
    if fresh_row != fresh_col {
        return Err(PyrucastError::Message(format!(
            "Matrix: {} and {} are declared conjugate, but one of them is already \
             paired with another DOF — two blocks disagree on what faces what",
            dof_text(r, vars),
            dof_text(c, vars)
        )));
    }
    if fresh_row {
        out_row.push(r);
        out_col.push(c);
    }
    Ok(())
}

/// Both sides of a conjugate run must hold the same number of DOFs.
fn conjugate_lengths(rows: usize, cols: usize, what: &str) -> Result<()> {
    if rows != cols {
        return Err(PyrucastError::Message(format!(
            "Matrix: {what} faces {rows} row DOF(s) with {cols} column DOF(s); a block \
             declaring symmetry must be square, and the two members of a pair must be \
             each other's transpose"
        )));
    }
    Ok(())
}

/// Global CSR sparsity pattern plus the DOF numbering it indexes.
///
/// A pure function of a model's **block structure** — not of the material
/// values — so it can be built once and reused across assemblies of the same
/// model (materials change, sparsity does not). The assembler
/// ([`crate::ops::matrix`]) builds it from a [`Matrix`]'s blocks and caches it
/// on the model (see `Model::matrix_pattern`); the numeric scatter then only
/// fills the values at each entry's fixed slot.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let bloc = || {
/// #     let mut z = SubMatrix::new(
/// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #         DofOrdering::NodesThenVars, Symmetry::Full);
/// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// #     z
/// # };
/// # use pyrucast::ops::scatter;
/// let mut k = Matrix::empty();
/// k.add_sub(Handle::new(bloc()))?;
/// k.finalize()?;
/// // A pure function of the model's **structure**, not of its materials:
/// // built once, reused from one assembly to the next.
/// let motif = scatter::build_pattern(&k)?;
/// assert_eq!(motif.nnz(), 4);
/// assert_eq!(motif.row_offsets.len(), motif.row_keys.len() + 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone)]
pub struct AssemblyPattern {
    /// The variable name table the DOF keys index — interned once for the whole
    /// pattern, rows and columns together.
    pub vars: std::sync::Arc<Vec<String>>,
    /// Global row DOFs as packed [`DofKey`]s, in CSR row order. Held by `Arc`
    /// for the same reason as the sparsity below: every matrix assembled from
    /// this pattern **shares** the numbering instead of copying it.
    pub row_keys: std::sync::Arc<Vec<DofKey>>,
    /// Global column DOFs as packed [`DofKey`]s, in CSR column order. Shared
    /// like [`row_keys`](Self::row_keys).
    pub col_keys: std::sync::Arc<Vec<DofKey>>,
    /// CSR row offsets, length `row_keys.len() + 1`. Held by `Arc` because the
    /// assembled matrices built from this pattern **share** it rather than each
    /// copying the sparsity out.
    pub row_offsets: std::sync::Arc<Vec<usize>>,
    /// CSR column indices, sorted within each row, length `row_offsets[nrows]`.
    /// Shared like [`row_offsets`](Self::row_offsets).
    pub col_indices: std::sync::Arc<Vec<usize>>,
    /// Precomputed value-array slot of every entry each block contributes, one
    /// entry per block in the matrix's `subs` order. Since the pattern is
    /// material-independent and cached, the `binary_search` that maps a
    /// block-local `(r, c)` to its CSR slot is paid **once here**, not on every
    /// assembly's numeric scatter. The scatter reads these back directly (see
    /// [`crate::ops::scatter`]).
    pub block_slots: Vec<BlockSlots>,
}

/// Precomputed CSR value-array slots for one block, aligned with the order that
/// block emits its entries at scatter time. A computed block's cells are
/// evaluated cell-by-cell, so its slots are grouped per cell, matching
/// `element_block_triplets_per_cell`'s `per_cell` (and, entry-for-entry, the
/// `(li, di, lj, pj)` emission order of `element_block_pattern`). A literal
/// block's slots follow its `local_coo_arrays` order.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
/// # use pyrucast::containers::mesh::SubMesh;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(1).unwrap());
/// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
/// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
/// # let support = {
/// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
/// #     sm.add_cell(&[a.id()]).unwrap();
/// #     sm.add_cell(&[b.id()]).unwrap();
/// #     Handle::new(sm)
/// # };
/// # let bloc = || {
/// #     let mut z = SubMatrix::new(
/// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
/// #         DofOrdering::NodesThenVars, Symmetry::Full);
/// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
/// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
/// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
/// #     z
/// # };
/// # use pyrucast::containers::matrix::BlockSlots;
/// # use pyrucast::ops::scatter;
/// let mut k = Matrix::empty();
/// k.add_sub(Handle::new(bloc()))?;
/// k.finalize()?;
/// // A **literal** block lays its slots out in its COO order;
/// // a computed block groups them per cell, in the order the kernel
/// // produit.
/// let motif = scatter::build_pattern(&k)?;
/// match &motif.block_slots[0] {
///     BlockSlots::Literal(slots) => assert_eq!(slots.len(), 4),
///     _ => unreachable!("this block carries its values"),
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone)]
pub enum BlockSlots {
    /// A computed block, one **flat** run of `stride` slots per cell, in the
    /// kernel's `(li, di, lj, pj)` emission order.
    ///
    /// A `Vec` per cell meant ten million allocations on a solid mesh, each
    /// holding a few hundred indices. One buffer holds them all instead, and
    /// `u32` is twice as narrow as the `usize` it replaces.
    Computed { slots: Vec<u32>, stride: usize },
    /// A computed block whose trailing primal variable is **factored out**:
    /// `stride` base slots per cell in `(li, di, lj)` order, the entry for
    /// primal variable `pj` sitting at `base + pj`.
    ///
    /// The columns of one node are consecutive integers under
    /// [`DofOrdering::NodesThenVars`], and a CSR row keeps its columns sorted,
    /// so they land in consecutive slots. Storing one base instead of
    /// `n_primal` slots divides this block's index memory by as much — three,
    /// for a 3-D elasticity. The assembler only takes this form after
    /// **checking** the consecutiveness on the block's own numbering.
    ComputedBlocked {
        bases: Vec<u32>,
        stride: usize,
        n_primal: usize,
    },
    /// One slot per COO entry (literal block).
    Literal(Vec<usize>),
}

impl BlockSlots {
    /// Hand `add(slot, value)` every entry of `cell`, pairing that cell's `ke`
    /// with the CSR slot each value lands in.
    ///
    /// The two computed forms differ only in how the slot is found, and both
    /// walk `ke` in the kernel's `(li, di, lj, pj)` emission order — so the
    /// scatter is written once, and the compression stays an implementation
    /// detail of the pattern. A literal block has no cells and contributes
    /// nothing here.
    pub(crate) fn each_entry(&self, cell: usize, ke: &[f64], mut add: impl FnMut(usize, f64)) {
        match self {
            BlockSlots::Computed { slots, stride } => {
                let base = cell * stride;
                for (k, &v) in ke.iter().enumerate() {
                    add(slots[base + k] as usize, v);
                }
            }
            BlockSlots::ComputedBlocked {
                bases,
                stride,
                n_primal,
            } => {
                // `k = (…, lj) · n_primal + pj`, and the pattern checked that a
                // node's primal columns sit in consecutive slots — so one base
                // per `(li, di, lj)` names all `n_primal` of them.
                let row = &bases[cell * stride..(cell + 1) * stride];
                let mut k = 0;
                for &base in row {
                    for pj in 0..*n_primal {
                        add(base as usize + pj, ke[k]);
                        k += 1;
                    }
                }
            }
            BlockSlots::Literal(_) => {}
        }
    }
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::ops::scatter;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// let motif = scatter::build_pattern(&k)?;
    /// // The number of **stored** entries — the CSR's, not the blocks'.
    /// assert_eq!(motif.nnz(), motif.col_indices.len());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// The row DOFs in materialised `(node, variable name)` form.
    ///
    /// The pattern numbers with packed keys; this is the translation for the
    /// callers that speak names, and it allocates a `String` per DOF — so it
    /// belongs at an API edge, never inside an assembly.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix, scatter};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// # let motif = scatter::build_pattern(&k)?;
    /// // The pattern numbers in keys; names, it gives back on demand.
    /// assert_eq!(motif.row_dofs()[0], (a.id(), "q".to_string()));
    /// assert_eq!(motif.row_dofs().len(), motif.row_keys.len());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn row_dofs(&self) -> Vec<NamedDof> {
        self.named(&self.row_keys)
    }

    /// The column DOFs in materialised form — see
    /// [`row_dofs`](Self::row_dofs).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix, scatter};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// # let motif = scatter::build_pattern(&k)?;
    /// // Côté colonne, la variable primale.
    /// assert_eq!(motif.col_dofs()[0], (a.id(), "T".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn col_dofs(&self) -> Vec<NamedDof> {
        self.named(&self.col_keys)
    }

    /// Wrap `values` — the numeric phase's output, in this pattern's slot order
    /// — into an assembled CSR that **shares** this pattern's sparsity.
    pub(crate) fn assembled_csr(&self, values: Vec<f64>) -> AssembledCsr {
        AssembledCsr {
            row_offsets: self.row_offsets.clone(),
            col_indices: self.col_indices.clone(),
            values,
            ncols: self.col_keys.len(),
        }
    }

    fn named(&self, keys: &[DofKey]) -> Vec<NamedDof> {
        keys.iter()
            .map(|&k| (dof_node(k), self.vars[dof_var(k) as usize].clone()))
            .collect()
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
#[allow(clippy::type_complexity)]
fn csr_from_triplets_parallel(
    nrows: usize,
    _ncols: usize,
    triplets: Vec<(usize, usize, f64)>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
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
    (row_offsets, col_indices, values)
}

impl Matrix {
    /// Build the global DOF union + CSR. Must be called before any
    /// solver-facing method (`to_csr`, `to_dmatrix`, `mul_dense`, `dense`,
    /// `to_coo`, `to_csc`). Idempotent: a second call is a no-op if no
    /// `add_sub` has occurred since the last `finalize`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// // The size reads off the blocks, without assembly…
    /// assert_eq!((k.n_rows()?, k.n_cols()?), (2, 2));
    /// // …but the sparse views exist only once the matrix is finalized.
    /// assert!(k.to_csr().is_err());
    /// k.finalize()?;
    /// assert_eq!(k.to_csr()?.nnz(), 4);
    /// // Idempotent: a second call reassembles nothing.
    /// k.finalize()?;
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn finalize(&mut self) -> Result<()> {
        if self.assembled.is_some() {
            return Ok(());
        }
        // A *computed* block carries no values: producing them means driving a
        // model kernel that lives in `crate::models` (outside `containers`).
        // Assembling it here would force a matrix↔kernel cycle, so we don't —
        // the global assembler in `ops::matrix::stiffness` handles computed
        // blocks and injects the finished CSR via `set_assembled`.
        for h in &*self {
            if h.read().is_computed() {
                return Err(PyrucastError::Message(
                    "Matrix::finalize: this matrix carries a computed block; \
                     assemble it with m.assemble() (or \
                     ops::matrix::stiffness), which scatters the kernel into \
                     the global CSR — finalize() cannot (it must not reach into \
                     the model/kernel)"
                        .into(),
                ));
            }
        }
        let vars = std::sync::Arc::new(self.field_names());
        let (row_keys, col_keys) = self.dof_orders(&vars)?;
        let triplets = self.build_global_triplets(&vars, &row_keys, &col_keys)?;
        let (row_offsets, col_indices, values) =
            csr_from_triplets_parallel(row_keys.len(), col_keys.len(), triplets);
        let csr = AssembledCsr {
            row_offsets: std::sync::Arc::new(row_offsets),
            col_indices: std::sync::Arc::new(col_indices),
            values,
            ncols: col_keys.len(),
        };
        self.assembled = Some(AssembledData {
            vars,
            row_keys: std::sync::Arc::new(row_keys),
            col_keys: std::sync::Arc::new(col_keys),
            csr,
        });
        Ok(())
    }

    /// Inject a globally-assembled CSR built by an external assembler
    /// ([`crate::ops::matrix`]), bypassing [`finalize`](Self::finalize).
    /// `row_dofs` / `col_dofs` must be this matrix' global DOF union (as
    /// returned by [`row_dofs`](Self::row_dofs) / [`col_dofs`](Self::col_dofs))
    /// and index `csr`. This is the path for matrices carrying *computed*
    /// blocks, which `finalize` cannot assemble on its own (it would have to
    /// reach into the model/kernel — the cycle Option B avoids).
    pub(crate) fn set_assembled(
        &mut self,
        vars: std::sync::Arc<Vec<String>>,
        row_keys: std::sync::Arc<Vec<DofKey>>,
        col_keys: std::sync::Arc<Vec<DofKey>>,
        csr: AssembledCsr,
    ) {
        self.assembled = Some(AssembledData {
            vars,
            row_keys,
            col_keys,
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use std::sync::Arc;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// assert!(k.cached_factorization::<usize>().is_none()); // rien encore
    /// k.store_factorization(Arc::new(42usize));
    /// // The downcast is checked: another type yields nothing.
    /// assert!(k.cached_factorization::<String>().is_none());
    /// assert_eq!(k.cached_factorization::<usize>().as_deref(), Some(&42));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn cached_factorization<T: std::any::Any + Send + Sync>(
        &self,
    ) -> Option<std::sync::Arc<T>> {
        let arc = self.factorization.lock().as_ref().cloned()?;
        arc.downcast::<T>().ok()
    }

    /// Store a freshly computed factorization for transparent reuse. Cleared
    /// automatically whenever the matrix changes (`add_sub`).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use std::sync::Arc;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// // The solver drops its factorization here to reuse it as is at the next
    /// // step; any change to the matrix wipes it.
    /// k.store_factorization(Arc::new(42usize));
    /// assert_eq!(k.cached_factorization::<usize>().as_deref(), Some(&42));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn store_factorization(
        &self,
        factorization: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) {
        *self.factorization.lock() = Some(factorization);
    }

    // ── Block-traversal helpers (no finalize required) ──────────────────

    /// Deduplicated concatenation of the blocks' row (or col) DOFs, as packed
    /// [`DofKey`]s — the global DOF numbering.
    ///
    /// O(total block DOFs), and **allocation-free per DOF**: the variable names
    /// are resolved to slots once per block, the dedup runs on a direct-addressed
    /// seen-set over the packed keys ([`DofSeen`]), and nothing here builds a
    /// `String`.
    ///
    /// Order: **solver order** when the backing `Coords` carries a
    /// [`permutation`](crate::coords::Coords::permutation) (stable
    /// sort by the node's permutation index, so the per-node variable order is
    /// preserved); otherwise **first-seen** (identical to the historical
    /// behaviour, hence bit-for-bit stable when no permutation is set).
    fn collect_dof_keys(&self, row: bool, vars: &[String]) -> Result<Vec<DofKey>> {
        let slot_of: HashMap<String, u32> = vars
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();

        // Size the seen-set before filling it: the DOF total bounds the output,
        // the largest node id bounds the direct-addressed table.
        let (mut total, mut max_node) = (0usize, 0u32);
        for h in self {
            let sub = h.read();
            let support = if row {
                sub.row_support().read()
            } else {
                sub.col_support().read()
            };
            total += if row { sub.n_rows() } else { sub.n_cols() };
            for n in support.connectivity() {
                max_node = max_node.max(n.0);
            }
        }

        let mut seen = DofSeen::new(max_node, vars.len(), total);
        let mut out: Vec<DofKey> = Vec::with_capacity(total);
        for h in self {
            let sub = h.read();
            let keys = if row {
                sub.row_dof_keys_with(&sub.row_support().read(), &slot_of)
            } else {
                sub.col_dof_keys_with(&sub.col_support().read(), &slot_of)
            };
            for k in keys {
                if seen.insert(k) {
                    out.push(k);
                }
            }
        }

        if let Some(first) = self.iter().next() {
            let coords_h = first.read().coords();
            if let Some(perm) = coords_h.read().permutation() {
                // Stable, so a node's variables keep their relative order.
                out.par_sort_by_key(|&k| perm[dof_node(k).0 as usize]);
            }
        }
        Ok(out)
    }

    /// The two global DOF orders, row and column.
    ///
    /// On a matrix that [declares itself symmetric](Self::symmetric) they are
    /// built **together**, so that rank `i` holds conjugate DOFs on both sides —
    /// which is what makes the assembled array symmetric and not merely the
    /// operator it represents. Otherwise the two orders are collected
    /// independently, as they always were: a rectangular matrix has no conjugate
    /// to pair with, and demanding the alignment would refuse it.
    fn dof_orders(&self, vars: &[String]) -> Result<(Vec<DofKey>, Vec<DofKey>)> {
        if self.symmetric() {
            self.collect_conjugate_dof_keys(vars)
        } else {
            Ok((
                self.collect_dof_keys(true, vars)?,
                self.collect_dof_keys(false, vars)?,
            ))
        }
    }

    /// Both DOF orders in **one** walk, conjugate rank for rank.
    ///
    /// Nothing here learns that `q` is the dual of `T`. It reads only which
    /// *positions* correspond, which the blocks already declare:
    ///
    /// - a `Full` block is square over one support (a physics block hands the
    ///   **same handle** to both sides) with its dual and primal variables
    ///   matched by position, so its row `k` faces its own column `k`;
    /// - a `Half` pair crosses: [`pairs_are_mutual_transposes`] already holds
    ///   that one block's row support **is** the other's column support, by
    ///   handle identity. So `a`'s rows face `b`'s columns, and `a`'s columns
    ///   face `b`'s rows.
    ///
    /// [`pairs_are_mutual_transposes`]: Self::pairs_are_mutual_transposes
    fn collect_conjugate_dof_keys(&self, vars: &[String]) -> Result<(Vec<DofKey>, Vec<DofKey>)> {
        let slot_of: HashMap<String, u32> = vars
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();

        // Both sides bound the seen-sets here, the walk filling them in step.
        let (mut total, mut max_node) = (0usize, 0u32);
        for h in self {
            let sub = h.read();
            total += sub.n_rows();
            for support in [sub.row_support(), sub.col_support()] {
                for n in support.read().connectivity() {
                    max_node = max_node.max(n.0);
                }
            }
        }

        let mut seen_row = DofSeen::new(max_node, vars.len(), total);
        let mut seen_col = DofSeen::new(max_node, vars.len(), total);
        let mut out_row: Vec<DofKey> = Vec::with_capacity(total);
        let mut out_col: Vec<DofKey> = Vec::with_capacity(total);
        let mut consumed = vec![false; self.len()];

        for (i, h) in self.iter().enumerate() {
            if consumed[i] {
                continue;
            }
            let a = h.read();
            // Each support guard is taken and dropped on its own: the two sides
            // of a pair share one `SubMesh`, and holding both at once would mean
            // two read locks on the same lock.
            let a_rows = {
                let g = a.row_support().read();
                a.row_dof_keys_with(&g, &slot_of)
            };
            let a_cols = {
                let g = a.col_support().read();
                a.col_dof_keys_with(&g, &slot_of)
            };
            match a.symmetry() {
                Symmetry::Full => {
                    conjugate_lengths(a_rows.len(), a_cols.len(), "a Full block")?;
                    for (&r, &c) in a_rows.iter().zip(a_cols.iter()) {
                        push_conjugate(
                            &mut seen_row,
                            &mut seen_col,
                            &mut out_row,
                            &mut out_col,
                            vars,
                            r,
                            c,
                        )?;
                    }
                }
                Symmetry::Half(id) => {
                    // The other member: same group, **other** low bit. Two blocks
                    // sharing the bit are the same member and do not cross.
                    let partner = self.iter().enumerate().position(|(j, g)| {
                        j != i
                            && !consumed[j]
                            && matches!(g.read().symmetry(),
                                Symmetry::Half(o) if o & !1 == id & !1 && o & 1 != id & 1)
                    });
                    let Some(j) = partner else {
                        return Err(PyrucastError::Message(
                            "Matrix: a Half block has no partner in this matrix, so the two \
                             DOF orders cannot be paired — the symmetry declaration and the \
                             blocks disagree"
                                .into(),
                        ));
                    };
                    let p = self.iter().nth(j).expect("index from position").read();
                    let p_rows = {
                        let g = p.row_support().read();
                        p.row_dof_keys_with(&g, &slot_of)
                    };
                    let p_cols = {
                        let g = p.col_support().read();
                        p.col_dof_keys_with(&g, &slot_of)
                    };
                    conjugate_lengths(a_rows.len(), p_cols.len(), "a Half pair")?;
                    conjugate_lengths(p_rows.len(), a_cols.len(), "a Half pair")?;
                    for (&r, &c) in a_rows.iter().zip(p_cols.iter()) {
                        push_conjugate(
                            &mut seen_row,
                            &mut seen_col,
                            &mut out_row,
                            &mut out_col,
                            vars,
                            r,
                            c,
                        )?;
                    }
                    for (&r, &c) in p_rows.iter().zip(a_cols.iter()) {
                        push_conjugate(
                            &mut seen_row,
                            &mut seen_col,
                            &mut out_row,
                            &mut out_col,
                            vars,
                            r,
                            c,
                        )?;
                    }
                    consumed[j] = true;
                }
                // `symmetric()` returned true, so no block declares `None`.
                Symmetry::None => unreachable!("a None block cannot reach the conjugate walk"),
            }
        }

        if let Some(first) = self.iter().next() {
            let coords_h = first.read().coords();
            if let Some(perm) = coords_h.read().permutation() {
                // Sorted as **pairs**, on the row's node — the conjugate shares
                // it — so the two sides cannot drift apart. Stable, so a node's
                // variables keep their relative order.
                let mut pairs: Vec<(DofKey, DofKey)> = out_row.into_iter().zip(out_col).collect();
                pairs.par_sort_by_key(|&(r, _)| perm[dof_node(r).0 as usize]);
                (out_row, out_col) = pairs.into_iter().unzip();
            }
        }
        Ok((out_row, out_col))
    }

    /// Materialise packed keys into `(node, variable name)` pairs.
    fn name_dofs(keys: &[DofKey], vars: &[String]) -> Vec<NamedDof> {
        keys.iter()
            .map(|&k| (dof_node(k), vars[dof_var(k) as usize].clone()))
            .collect()
    }

    /// Map every block's **local** triplets to global `(row, col, value)`
    /// arrays via a per-block translation table (built once from the global DOF
    /// maps). The remap is index-preserving, so the concatenated stream — blocks
    /// in order, entries in COO order — matches the old serial scatter. The
    /// per-block remap runs in parallel. O(total block DOFs + nnz), no per-entry
    /// search.
    fn build_global_triplets(
        &self,
        vars: &[String],
        row_keys: &[DofKey],
        col_keys: &[DofKey],
    ) -> Result<Vec<(usize, usize, f64)>> {
        let slot_of: HashMap<String, u32> = vars
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();
        let row_map: HashMap<DofKey, usize> =
            row_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let col_map: HashMap<DofKey, usize> =
            col_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let mut out: Vec<(usize, usize, f64)> = Vec::new();
        for h in self {
            let sub = h.read();
            // local DOF index → global index (the "simple remap").
            let trow: Vec<usize> = sub
                .row_dof_keys(&slot_of)
                .iter()
                .map(|k| row_map[k])
                .collect();
            let tcol: Vec<usize> = sub
                .col_dof_keys(&slot_of)
                .iter()
                .map(|k| col_map[k])
                .collect();
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

    /// Whether the assembled matrix satisfies `A[i][j] == A[j][i]`, by adding up
    /// what its blocks [declare](Symmetry). Vacuously true for an empty aggregate.
    ///
    /// A block declaring [`Full`](Symmetry::Full) carries its own symmetry. A
    /// [`Half`](Symmetry::Half) carries only one side of one, so the aggregate
    /// must find the other: among the blocks sharing a [`PairId`], the two
    /// members — told apart by its low bit — must be present **in equal,
    /// non-zero numbers**.
    ///
    /// Counting the members rather than the blocks is what lets a constraint
    /// declared twice (four blocks, two of each) stand, while a pair broken by
    /// [`Aggregate::subset`] (two blocks, both the same member) falls. A bare
    /// "exactly two" would lose the first; a parity check would accept the second.
    ///
    /// Blocks are a handful, so the tally lives in one small `Vec` — and this runs
    /// once when a matrix is about to be solved, never inside a loop.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // A Galerkin stiffness is symmetric, and its Dirichlet's two rectangular
    /// // blocks are a pair — so the aggregate concludes it is.
    /// assert!(k.symmetric());
    /// ```
    pub fn symmetric(&self) -> bool {
        // One entry per pair met: its identity with the low bit cleared, and how
        // many blocks of each member it has been handed.
        let mut pairs: Vec<(PairId, [usize; 2])> = Vec::new();
        for h in self {
            match h.read().symmetry() {
                Symmetry::Full => {}
                Symmetry::None => return false,
                Symmetry::Half(id) => {
                    let (base, member) = (id & !1, (id & 1) as usize);
                    match pairs.iter_mut().find(|(b, _)| *b == base) {
                        Some((_, seen)) => seen[member] += 1,
                        None => {
                            let mut seen = [0usize; 2];
                            seen[member] = 1;
                            pairs.push((base, seen));
                        }
                    }
                }
            }
        }
        if pairs.iter().any(|(_, seen)| seen[0] != seen[1]) {
            return false;
        }
        debug_assert!(
            self.pairs_are_mutual_transposes(),
            "a Half group holds blocks that are not each other's transpose — \
             two distinct pairs hashed alike, or a pair was minted wrongly"
        );
        true
    }

    /// Every `Half` group holds blocks whose supports and variables cross — the
    /// structural half of what `Half` claims.
    ///
    /// Only a `debug_assert!` calls this. It cannot check the low bit, which
    /// encodes a producer's choice rather than a fact, and it is not meant to
    /// stand between a hash collision and a wrong answer in release: it catches a
    /// pair minted wrongly while that mistake is still cheap to fix.
    fn pairs_are_mutual_transposes(&self) -> bool {
        let halves: Vec<_> = self
            .iter()
            .filter_map(|h| match h.read().symmetry() {
                Symmetry::Half(id) => Some((id & !1, h)),
                _ => None,
            })
            .collect();
        halves.iter().all(|(base, h)| {
            let a = h.read();
            halves.iter().any(|(other_base, g)| {
                if other_base != base || Handle::same_object(h, g) {
                    return false;
                }
                let b = g.read();
                // Only the **supports** cross. The variable names do not: the
                // transpose swaps the dual and primal *roles*, and a constraint
                // gives those roles different names on each side (`imposed_T`
                // against `lambda_T` at the multiplier). There is nothing to
                // compare there.
                Handle::same_object(a.row_support(), b.col_support())
                    && Handle::same_object(a.col_support(), b.row_support())
            })
        })
    }

    /// Union of all row DOFs across blocks, in first-seen order.
    /// If finalized, returns the cached order (consistent with the CSR).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Un DDL de ligne : (nœud, variable **duale**).
    /// let lignes = k.row_dofs().unwrap();
    /// assert_eq!(lignes.len(), k.n_rows().unwrap());
    /// assert_eq!(lignes[0].1, "q");
    /// ```
    pub fn row_dofs(&self) -> Result<Vec<NamedDof>> {
        let vars = self.dof_vars();
        Ok(Self::name_dofs(&self.row_dof_keys()?, &vars))
    }

    /// Union of all column DOFs across blocks, in first-seen order.
    /// If finalized, returns the cached order (consistent with the CSR).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Un DDL de colonne : (nœud, variable **primale**).
    /// assert_eq!(k.col_dofs().unwrap()[0].1, "T");
    /// ```
    pub fn col_dofs(&self) -> Result<Vec<NamedDof>> {
        let vars = self.dof_vars();
        Ok(Self::name_dofs(&self.col_dof_keys()?, &vars))
    }

    /// The matrix's **variable name table** — the names a [`DofKey`]'s slot
    /// indexes. Interned once, rows and columns together.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::{dof_node, dof_var, Matrix};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // The compact form and the named form say the same thing: names are
    /// // live in the table, the key carries only their index.
    /// let noms = k.dof_vars();
    /// let cles = k.row_dof_keys()?;
    /// let nommes = k.row_dofs()?;
    /// assert_eq!(dof_node(cles[0]), nommes[0].0);
    /// assert_eq!(noms[dof_var(cles[0]) as usize], nommes[0].1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn dof_vars(&self) -> std::sync::Arc<Vec<String>> {
        if let Some(a) = &self.assembled {
            return a.vars.clone();
        }
        std::sync::Arc::new(self.field_names())
    }

    /// Row DOFs as packed [`DofKey`]s, in CSR row order — the numbering the
    /// assembler and the solver index with.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::{dof_node, dof_var, Matrix};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // The compact form of a row DOF: the node, and the index of the
    /// // dual name in the table — not the name itself.
    /// let cles = k.row_dof_keys()?;
    /// assert_eq!(dof_node(cles[0]), a.id());
    /// assert_eq!(k.dof_vars()[dof_var(cles[0]) as usize], "q");
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn row_dof_keys(&self) -> Result<std::sync::Arc<Vec<DofKey>>> {
        if let Some(a) = &self.assembled {
            return Ok(a.row_keys.clone());
        }
        Ok(self.dof_key_orders()?.0)
    }

    /// Both DOF orders at once — what a caller needing the two should ask for.
    ///
    /// On a symmetric matrix the two are built in a single conjugate walk, so
    /// asking for them separately would run that walk twice for one half of its
    /// result each time.
    pub(crate) fn dof_key_orders(
        &self,
    ) -> Result<(std::sync::Arc<Vec<DofKey>>, std::sync::Arc<Vec<DofKey>>)> {
        if let Some(a) = &self.assembled {
            return Ok((a.row_keys.clone(), a.col_keys.clone()));
        }
        let vars = self.field_names();
        let (row, col) = self.dof_orders(&vars)?;
        Ok((std::sync::Arc::new(row), std::sync::Arc::new(col)))
    }

    /// Column DOFs as packed [`DofKey`]s, in CSR column order — the column twin
    /// of [`row_dof_keys`](Self::row_dof_keys).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::{dof_node, dof_var, Matrix};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Same shape on the column side, over the **primal** variables.
    /// let cles = k.col_dof_keys()?;
    /// assert_eq!(dof_node(cles[0]), a.id());
    /// assert_eq!(k.dof_vars()[dof_var(cles[0]) as usize], "T");
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn col_dof_keys(&self) -> Result<std::sync::Arc<Vec<DofKey>>> {
        if let Some(a) = &self.assembled {
            return Ok(a.col_keys.clone());
        }
        Ok(self.dof_key_orders()?.1)
    }

    /// Union of all field names (dual + primal) across blocks.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // interned once only, rows and columns together.
    /// let noms = k.field_names();
    /// assert!(noms.contains(&"T".to_string()) && noms.contains(&"q".to_string()));
    /// ```
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // interned once only, rows and columns together.
    /// let noms = k.field_names();
    /// assert!(noms.contains(&"T".to_string()) && noms.contains(&"q".to_string()));
    /// ```
    pub fn field_names(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for h in self {
            for name in h.read().field_names() {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        out
    }

    /// Number of distinct row DOFs.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// assert_eq!(k.n_rows().unwrap(), 2); // two nodes, one dual variable
    /// ```
    pub fn n_rows(&self) -> Result<usize> {
        Ok(self.row_dof_keys()?.len())
    }

    /// Number of distinct column DOFs.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// assert_eq!(k.n_cols().unwrap(), 2);
    /// ```
    pub fn n_cols(&self) -> Result<usize> {
        Ok(self.col_dof_keys()?.len())
    }

    /// Total COO entries stored across all blocks (counting duplicates).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Counts the entries **stored** in the blocks. A *computed* block — what
    /// // `stiffness` produces — stores none: its values are born at assembly
    /// // time and live in the global CSR.
    /// assert_eq!(k.entry_count(), 0);
    /// assert_eq!(k.to_csr().unwrap().nnz(), 4); // the CSR does carry them
    /// ```
    pub fn entry_count(&self) -> usize {
        let mut total = 0usize;
        for h in self {
            total += h.read().entry_count();
        }
        total
    }

    /// Sum of contributions at `(row, col)` across every block.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Un SEG2 de longueur 1, k = 1 : K = [[1, -1], [-1, 1]].
    /// assert_eq!(k.get(a.id(), "q", a.id(), "T"), 1.0);
    /// assert_eq!(k.get(a.id(), "q", b.id(), "T"), -1.0);
    /// // A coordinate absent from the pattern is zero: reading does not raise.
    /// assert_eq!(k.get(a.id(), "q", a.id(), "absente"), 0.0);
    /// ```
    pub fn get(&self, row_node: NodeId, row_field: &str, col_node: NodeId, col_field: &str) -> f64 {
        // When assembled, the CSR is the source of truth — and the only place a
        // *computed* block's values live (its COO is empty). Fall back to the
        // per-block COO sum only for an unassembled (literal) matrix.
        if let Some(a) = &self.assembled {
            let slot = |name: &str| a.vars.iter().position(|v| v == name).map(|i| i as u32);
            let find = |keys: &[DofKey], node: NodeId, name: &str| {
                slot(name).and_then(|s| keys.iter().position(|&k| k == dof_key(node, s)))
            };
            let r = find(&a.row_keys, row_node, row_field);
            let c = find(&a.col_keys, col_node, col_field);
            return match (r, c) {
                (Some(r), Some(c)) => a.csr.slot(r, c).map_or(0.0, |k| a.csr.values[k]),
                _ => 0.0,
            };
        }
        let mut total = 0.0;
        for h in self {
            total += h.read().get(row_node, row_field, col_node, col_field);
        }
        total
    }

    /// All COO entries across every block, in block-insertion order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // One 5-tuple per entry: row node, dual var, column node,
    /// // primal var, value — names are already resolved there. Like
    /// // `entry_count` walks the entries **stored**: a
    /// // computed one has none, its values living in the global CSR.
    /// assert!(k.iter_entries().is_empty());
    /// ```
    pub fn iter_entries(&self) -> Vec<MatrixEntry> {
        let mut out = Vec::new();
        for h in self {
            out.extend(h.read().iter_entries());
        }
        out
    }

    // ── Solver-facing (require finalize) ────────────────────────────────

    /// Assembled CSR, **materialised**. Requires [`finalize`](Self::finalize).
    ///
    /// The assembled state holds the sparsity by `Arc`, shared with the
    /// assembler's memoised pattern, and only the values as its own. Building a
    /// `CsrMatrix` — which owns all three arrays — therefore copies the
    /// sparsity, so ask for one only where that type is genuinely needed; to
    /// read the assembled matrix, [`csr_arrays`](Self::csr_arrays) borrows it.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // The CSR **is** the assembled state: borrowed, not rebuilt.
    /// assert_eq!(k.to_csr().unwrap().nnz(), 4);
    /// ```
    pub fn to_csr(&self) -> Result<CsrMatrix<f64>> {
        self.assembled_or_err()?.csr.to_csr()
    }

    /// The assembled CSR **borrowed**: `(row_offsets, col_indices, values)`.
    /// Requires [`finalize`](Self::finalize).
    ///
    /// The reading form of [`to_csr`](Self::to_csr), copying nothing.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // The CSR's three arrays, borrowed — nothing is copied.
    /// let (offsets, cols, values) = k.csr_arrays()?;
    /// assert_eq!(offsets.len(), k.n_rows()? + 1);
    /// assert_eq!(cols.len(), values.len());
    /// assert_eq!(values.len(), k.to_csr()?.nnz());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn csr_arrays(&self) -> Result<(&[usize], &[usize], &[f64])> {
        let a = self.assembled_or_err()?;
        Ok((&a.csr.row_offsets, &a.csr.col_indices, &a.csr.values))
    }

    /// Assembled dense matrix. Requires [`finalize`](Self::finalize).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // A nalgebra view (column-major), ready for LU or Cholesky.
    /// let m = k.to_dmatrix().unwrap();
    /// assert_eq!((m.nrows(), m.ncols()), (2, 2));
    /// ```
    pub fn to_dmatrix(&self) -> Result<DMatrix<f64>> {
        let a = self.assembled_or_err()?;
        let nr = a.row_keys.len();
        let nc = a.col_keys.len();
        let mut out = DMatrix::<f64>::zeros(nr, nc);
        for (r, c, v) in a.csr.triplets() {
            out[(r, c)] += v;
        }
        Ok(out)
    }

    /// Row-major dense buffer. Requires [`finalize`](Self::finalize).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Vue dense ligne-major, longueur n_rows × n_cols.
    /// let d = k.dense().unwrap();
    /// assert_eq!(d.len(), k.n_rows().unwrap() * k.n_cols().unwrap());
    /// assert_eq!(d, vec![1.0, -1.0, -1.0, 1.0]);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// assert_eq!(k.to_coo().unwrap().nnz(), 4);
    /// ```
    pub fn to_coo(&self) -> Result<CooMatrix<f64>> {
        let a = self.assembled_or_err()?;
        let mut coo = CooMatrix::<f64>::new(a.row_keys.len(), a.col_keys.len());
        for (r, c, v) in a.csr.triplets() {
            coo.push(r, c, v);
        }
        Ok(coo)
    }

    /// Assembled CSC. Requires [`finalize`](Self::finalize).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// assert_eq!(k.to_csc().unwrap().ncols(), 2);
    /// ```
    pub fn to_csc(&self) -> Result<CscMatrix<f64>> {
        Ok(CscMatrix::from(&self.assembled_or_err()?.csr.to_csr()?))
    }

    /// `y = A · x` (dense). Requires [`finalize`](Self::finalize).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // K · [1, 1] = 0: the rigid mode of a conduction.
    /// assert_eq!(k.mul_dense(&[1.0, 1.0]).unwrap(), vec![0.0, 0.0]);
    /// ```
    pub fn mul_dense(&self, x: &[f64]) -> Result<Vec<f64>> {
        let a = self.assembled_or_err()?;
        let nc = a.col_keys.len();
        if x.len() != nc {
            return Err(PyrucastError::Message(format!(
                "mul_dense: x has length {} but matrix has {} columns",
                x.len(),
                nc
            )));
        }
        // Row-wise SpMV straight on the CSR arrays: each row's dot product is
        // written once, so the result is independent of the thread count.
        let (offsets, cols, vals) = (&a.csr.row_offsets, &a.csr.col_indices, &a.csr.values);
        let mut y = vec![0.0f64; a.csr.nrows()];
        y.par_iter_mut()
            .with_min_len(MIN_PARALLEL_LEN)
            .enumerate()
            .for_each(|(r, out)| {
                let (lo, hi) = (offsets[r], offsets[r + 1]);
                *out = (lo..hi).map(|k| vals[k] * x[cols[k]]).sum();
            });
        Ok(y)
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // Le support **ligne** : la cible de projection d'un second membre.
    /// assert_eq!(k.row_mesh().unwrap().cell_count(), 2);
    /// ```
    pub fn row_mesh(&self) -> Result<Mesh> {
        self.support_mesh(true)
    }

    /// [`Mesh`] over the blocks' distinct **column** supports (the primal
    /// side: where a `solve` solution lives). Column twin of
    /// [`row_mesh`](Self::row_mesh) — e.g. to project an initial or imposed
    /// field onto the exact supports of the solution before combining.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// // The **column** support: that of a field being multiplied.
    /// assert_eq!(k.col_mesh().unwrap().cell_count(), 2);
    /// ```
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
    /// call [`Matrix::assemble`] before handing it to a solver.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// # use pyrucast::models::Physics;
    /// # use pyrucast::ops::model;
    /// // The result is **not** assembled: call `assemble` before solving.
    /// let thermique = k.filter(Physics::Thermal);
    /// assert_eq!(thermique.len(), 1);
    /// assert!(k.filter(Physics::Mechanical).is_empty());
    /// ```
    pub fn filter(&self, physics: Physics) -> Matrix {
        // Built block by block rather than through `subset`: that one is
        // fallible on an out-of-range index, which selecting by predicate
        // cannot produce — and `Matrix` is the one aggregate with no
        // `check_push`, so appending never fails either.
        let mut out = Matrix::empty();
        for h in self {
            if h.read().physics().contains(&physics) {
                out.push(h.clone());
            }
        }
        out.post_push();
        out
    }

    /// The set of [`Physics`] natures present across this matrix's blocks —
    /// first-seen order, deduplicated. Empty if no block is tagged (« rien »).
    /// A matrix aggregating several physics reports **several** tags here (e.g.
    /// a heat model with a Dirichlet → `[Thermal, Constraint]`).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::Matrix;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::{element_field, matrix};
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    /// # mesh.add_cell(&[a.id(), b.id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    /// # let model = model::heat_conduction(&fes).unwrap();
    /// # let materials = element_field::material_field(&model, &[("k", 1.0)]).unwrap();
    /// # let k = matrix::stiffness(&model, &materials).unwrap();
    /// # use pyrucast::models::Physics;
    /// # use pyrucast::ops::model;
    /// // The kinds **present**, deduplicated.
    /// assert!(k.physics().contains(&Physics::Thermal));
    /// ```
    pub fn physics(&self) -> Vec<Physics> {
        let mut out: Vec<Physics> = Vec::new();
        for h in self {
            for &p in h.read().physics() {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Shared body of `row_mesh` / `col_mesh`.
    fn support_mesh(&self, rows: bool) -> Result<Mesh> {
        let mut out = Mesh::empty();
        for h in self {
            let s = h.read();
            let sup = if rows {
                s.row_support.clone()
            } else {
                s.col_support.clone()
            };
            if !out.iter().any(|m| m.same_object(&sup)) {
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::containers::field::SubField;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// // A flat vector, in column DOF order, becomes a field again —
    /// // laid on the **existing** POI1 supports, materializing no other.
    /// let f = k.field_from_col_values(&[10.0, 20.0])?;
    /// assert_eq!(f.get(0)?.read().value(b.id(), "T")?, 20.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_from_col_values(&self, x: &[f64]) -> Result<NodeField> {
        self.field_from_flat_values(x, false)
    }

    /// Row-side twin of [`field_from_col_values`](Self::field_from_col_values):
    /// wrap a flat **row-ordered** vector (`assembled.row_dofs` order — the
    /// layout `A · x` comes in) into a [`NodeField`] whose zones share the
    /// blocks' `row_support` handles and carry their dual variables.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::containers::field::SubField;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// // Le jumeau côté lignes : la disposition dans laquelle arrive `A · x`,
    /// // and whose components are the **dual** variables.
    /// let f = k.field_from_row_values(&[1.0, 2.0])?;
    /// assert_eq!(f.get(0)?.read().value(b.id(), "q")?, 2.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn field_from_row_values(&self, y: &[f64]) -> Result<NodeField> {
        self.field_from_flat_values(y, true)
    }

    /// Shared body of `field_from_{col,row}_values` (`rows` picks the side).
    fn field_from_flat_values(&self, values: &[f64], rows: bool) -> Result<NodeField> {
        let a = self.assembled_or_err()?;
        let keys = if rows { &a.row_keys } else { &a.col_keys };
        if values.len() != keys.len() {
            return Err(PyrucastError::Message(format!(
                "field_from_{}_values: {} value(s) for {} DOF(s)",
                if rows { "row" } else { "col" },
                values.len(),
                keys.len()
            )));
        }
        // Global DOF key → flat index, one hash pass over eight-byte keys.
        let index: HashMap<DofKey, usize> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let slot_of: HashMap<&str, u32> = a
            .vars
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i as u32))
            .collect();

        // Group the blocks by support slot; union their variables per group.
        // Same slot ⇒ same sealed POI1 ⇒ same node list, read from the support.
        struct Group {
            support: Handle<SubMesh>,
            vars: Vec<String>,
        }
        let mut groups: Vec<Group> = Vec::new();
        for h in self {
            let s = h.read();
            let (support, vars) = if rows {
                (s.row_support().clone(), s.dual_vars())
            } else {
                (s.col_support().clone(), s.primal_vars())
            };
            match groups.iter_mut().find(|g| g.support.same_object(&support)) {
                Some(g) => {
                    for v in vars {
                        if !g.vars.contains(v) {
                            g.vars.push(v.clone());
                        }
                    }
                }
                None => groups.push(Group {
                    support,
                    vars: vars.to_vec(),
                }),
            }
        }

        // One zone per group, on the block's own support handle. The field's
        // row order is the support's cell order, which is the very connectivity
        // read below, so values are written positionally, no per-node lookup.
        let mut out = NodeField::default();
        for g in &groups {
            use crate::containers::field::SubField;
            // Before the guard: `from_poi1` seals, which reads this same handle.
            let mut sub = SubNodeField::from_poi1(&g.support, g.vars.clone())?;
            let ncomp = g.vars.len();
            let vals = sub.values_mut();
            // The variables are resolved to slots once per zone, never per node.
            let slots: Vec<Option<u32>> = g
                .vars
                .iter()
                .map(|v| slot_of.get(v.as_str()).copied())
                .collect();
            let support = g.support.read();
            for (ni, nid) in support.connectivity().iter().enumerate() {
                for (ci, slot) in slots.iter().enumerate() {
                    let Some(slot) = slot else { continue };
                    if let Some(&gi) = index.get(&dof_key(*nid, *slot)) {
                        vals[ni * ncomp + ci] = values[gi];
                    }
                }
            }
            drop(support);
            out.add_sub(Handle::new(sub))?;
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix, Symmetry};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(1).unwrap());
    /// # let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
    /// # let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
    /// # let support = {
    /// #     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    /// #     sm.add_cell(&[a.id()]).unwrap();
    /// #     sm.add_cell(&[b.id()]).unwrap();
    /// #     Handle::new(sm)
    /// # };
    /// # let bloc = || {
    /// #     let mut z = SubMatrix::new(
    /// #         support.clone(), support.clone(), vec!["q".into()], vec!["T".into()],
    /// #         DofOrdering::NodesThenVars, Symmetry::Full);
    /// #     z.add_entry(a.id(), "q", a.id(), "T", 2.0).unwrap();
    /// #     z.add_entry(a.id(), "q", b.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", a.id(), "T", -1.0).unwrap();
    /// #     z.add_entry(b.id(), "q", b.id(), "T", 2.0).unwrap();
    /// #     z
    /// # };
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::node_field::NodeField;
    /// let mut k = Matrix::empty();
    /// k.add_sub(Handle::new(bloc()))?;
    /// k.finalize()?;
    /// // `x` is read at the **column** DOFs (primal), the result lives on the
    /// // **Row** DOFs (dual): K · u = f. The `*` operator is its sugar.
    /// let x = NodeField::from_submesh(&k.col_mesh()?.get(0)?, vec!["T".into()])?;
    /// x.get(0)?.write().add_to_component("T", 1.0)?;
    /// let y = k.mul_field(&x)?;
    /// // This block's rows sum to 1, hence K · 1 = 1.
    /// assert_eq!(y.get(0)?.read().value(a.id(), "q")?, 1.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn mul_field(&self, x_field: &NodeField) -> Result<NodeField> {
        let x = x_field.gather(&self.col_dofs()?)?;
        let y = self.mul_dense(&x)?;
        self.field_from_row_values(&y)
    }

    /// A fresh [`Matrix`] with every block replaced by `f(block.clone())`, each
    /// re-inserted under a **new store slot**. Never mutates `self` or any of its
    /// blocks in place: `add_sub`/`union`/`filter` share `Handle<SubMatrix>`s
    /// (same store slot, refcount bump — see [`Aggregate::subset`]), so mutating
    /// one would silently reach every other `Matrix` aliasing it. Like
    /// [`filter`](Self::filter), the result is **not assembled**.
    fn map_blocks(&self, f: impl Fn(SubMatrix) -> SubMatrix) -> Matrix {
        let mut out = Matrix::empty();
        for h in self {
            let mapped = f((*h.read()).clone());
            out.push(Handle::new(mapped));
        }
        out.post_push();
        out
    }

    /// A fresh [`Matrix`] whose values are `self`'s put through `f` — the engine
    /// behind `Mul<f64>`, `Div<f64>` and `Neg`, which pass `|v| v * s`,
    /// `|v| v / s` and `|v| -v`.
    ///
    /// `f` lands on each block's [`factor`](SubMatrix::factor), never on the
    /// stored `coo`, so a computed block scales exactly like a literal one.
    ///
    /// **The assembled CSR comes along, scaled.** Scaling every block by the
    /// same amount scales every entry by it, leaving the sparsity untouched, so
    /// the assembly stays valid: only `values` is walked. Dropping it instead —
    /// as this did until the operators were reworked — meant the mandatory
    /// `assemble()` re-ran every element kernel over the whole mesh to apply one
    /// scalar. The index arrays and the name table are `Arc`s and are shared
    /// rather than copied ([`AssembledCsr`]). Note that `(Σ v) · s` is not, bit
    /// for bit, the `Σ (v · s)` a re-assembly would compute: both are the same
    /// quantity to within rounding, each reproducible.
    ///
    /// The cached factorization is *not* carried over: the fresh `Matrix` starts
    /// without one.
    fn scaled(&self, f: impl Fn(f64) -> f64) -> Matrix {
        let mut out = self.map_blocks(|mut b| {
            b.factor = f(b.factor);
            b
        });
        // After `map_blocks`, whose `post_push` has just cleared this.
        out.assembled = self.assembled.as_ref().map(|a| AssembledData {
            vars: a.vars.clone(),
            row_keys: a.row_keys.clone(),
            col_keys: a.col_keys.clone(),
            csr: AssembledCsr {
                row_offsets: a.csr.row_offsets.clone(),
                col_indices: a.csr.col_indices.clone(),
                values: a.csr.values.iter().map(|&v| f(v)).collect(),
                ncols: a.csr.ncols,
            },
        });
        out
    }

    /// [`scaled`](Self::scaled) for an owner: same result, but a block **no one
    /// else holds** is rescaled where it lies instead of being copied.
    ///
    /// A block is rebuilt only to protect the other matrices sharing it. When
    /// [`Handle::is_sole_owner`] says there are none — the common case for a
    /// matrix just back from [`crate::ops::matrix::stiffness`] — there is nobody
    /// to protect, and a literal block's COO need not be copied at all.
    fn scale_in_place(&mut self, f: impl Fn(f64) -> f64) {
        for h in self.items_mut() {
            if h.is_sole_owner() {
                let mut blk = h.write();
                blk.factor = f(blk.factor);
            } else {
                let mut blk = (*h.read()).clone();
                blk.factor = f(blk.factor);
                *h = Handle::new(blk);
            }
        }
        if let Some(a) = &mut self.assembled {
            for v in &mut a.csr.values {
                *v = f(*v);
            }
        }
        // Scaled values ⇒ any factorization of the old ones is stale. Done by
        // hand: nothing was pushed, so `post_push` never ran.
        *self.factorization.get_mut() = None;
    }

    /// A fresh [`Matrix`] holding `self`'s blocks then `other`'s, **shared** and
    /// deliberately **not** deduplicated — the engine behind `Add`/`Sub`.
    ///
    /// This is what parts `+` from `|`: [`Aggregate::union`] drops a block whose
    /// slot it already holds, so `k | k` is `k`, whereas a sum must count a
    /// contribution as many times as it is handed over — `k + k` is `2k`. The
    /// assembler sums whatever lands on the same global `(row, col)` and walks
    /// blocks by position, so a block present twice simply contributes twice.
    ///
    /// Like [`filter`](Self::filter), the result is **not assembled**.
    fn concat(&self, other: &Matrix) -> Matrix {
        let mut out = Matrix::empty();
        for h in self.iter().chain(other.iter()) {
            out.push(h.clone());
        }
        out.post_push();
        out
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
// `&matrix * s`, `/ s` and `-matrix` — a fresh `Matrix` whose blocks are scaled
// clones of `self`'s, carrying the assembled CSR along, scaled (see `scaled`).
// The owning forms rescale in place whatever block no one else holds
// (`scale_in_place`). **Infallible**: cloning a block and appending it cannot
// fail, `Matrix` being the one aggregate that declares no `check_push` — except
// a division, which refuses a divisor no matrix can survive (`check_divisor`).

impl std::ops::Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Self::Output {
        self.scaled(|v| v * rhs)
    }
}

impl std::ops::Mul<f64> for Matrix {
    type Output = Matrix;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self.scale_in_place(|v| v * rhs);
        self
    }
}

// Scalar on the left — `2.0 * k` reads as the mathematics does.
impl std::ops::Mul<Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix {
        rhs * self
    }
}

impl std::ops::Mul<&Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix {
        rhs * self
    }
}

impl std::ops::Div<f64> for &Matrix {
    type Output = Matrix;
    #[track_caller]
    fn div(self, rhs: f64) -> Self::Output {
        let d = check_divisor(rhs);
        self.scaled(|v| v / d)
    }
}

impl std::ops::Div<f64> for Matrix {
    type Output = Matrix;
    #[track_caller]
    fn div(mut self, rhs: f64) -> Self::Output {
        let d = check_divisor(rhs);
        self.scale_in_place(|v| v / d);
        self
    }
}

impl std::ops::Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        self.scaled(|v| -v)
    }
}

impl std::ops::Neg for Matrix {
    type Output = Matrix;
    fn neg(mut self) -> Matrix {
        self.scale_in_place(|v| -v);
        self
    }
}

// ─── Matrix algebra: `a + b`, `a - b` ───────────────────────────────────────
//
// A sum is **structural**: the result holds the blocks of both operands, shared
// and not deduplicated, and the assembler does the adding — it already sums
// whatever lands on the same global `(row, col)`. So `M/dt + K` costs a handful
// of refcount bumps, no value is touched, and a computed block stays computed.
// Like `filter`, the result is not assembled: `assemble()` before solving.
//
// This is where `+` parts from `|`. `union` drops a block it already holds, so
// `k | k` is `k` — right for composing an operator out of distinct pieces,
// wrong for a sum, which must count a contribution once per handing over.
// `k + k` is `2k`. Reach for `|` to assemble one operator from its parts, for
// `+` to add two operators.
//
// `a - b` negates `b`'s blocks, which copies them (the factor lives in the
// block); `a + b` copies nothing.

impl SubMatrix {
    /// This block alone in a fresh one-block [`Matrix`], copied into a new store
    /// slot — the operand the mixed block/matrix sums are built from.
    fn as_matrix(&self) -> Matrix {
        let mut out = Matrix::empty();
        out.push(Handle::new(self.clone()));
        out.post_push();
        out
    }

    /// This block negated, alone in a fresh [`Matrix`] — the right-hand side of
    /// a subtraction.
    fn as_negated_matrix(&self) -> Matrix {
        (-self).as_matrix()
    }
}

/// Write the three owned/borrowed variants of a binary operator whose reference
/// form (`&lhs op &rhs`) is already spelled out just above.
macro_rules! forward_ref_binop {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty) => {
        impl std::ops::$trait<$rhs> for &$lhs {
            type Output = Matrix;
            fn $method(self, rhs: $rhs) -> Matrix {
                std::ops::$trait::$method(self, &rhs)
            }
        }
        impl std::ops::$trait<&$rhs> for $lhs {
            type Output = Matrix;
            fn $method(self, rhs: &$rhs) -> Matrix {
                std::ops::$trait::$method(&self, rhs)
            }
        }
        impl std::ops::$trait<$rhs> for $lhs {
            type Output = Matrix;
            fn $method(self, rhs: $rhs) -> Matrix {
                std::ops::$trait::$method(&self, &rhs)
            }
        }
    };
}

impl std::ops::Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.concat(rhs)
    }
}

impl std::ops::Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.concat(&rhs.map_blocks(|b| -b))
    }
}

impl std::ops::Add<&SubMatrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &SubMatrix) -> Matrix {
        self.concat(&rhs.as_matrix())
    }
}

impl std::ops::Sub<&SubMatrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &SubMatrix) -> Matrix {
        self.concat(&rhs.as_negated_matrix())
    }
}

impl std::ops::Add<&Matrix> for &SubMatrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        self.as_matrix().concat(rhs)
    }
}

impl std::ops::Sub<&Matrix> for &SubMatrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix {
        self.as_matrix().concat(&rhs.map_blocks(|b| -b))
    }
}

impl std::ops::Add<&SubMatrix> for &SubMatrix {
    type Output = Matrix;
    fn add(self, rhs: &SubMatrix) -> Matrix {
        self.as_matrix().concat(&rhs.as_matrix())
    }
}

impl std::ops::Sub<&SubMatrix> for &SubMatrix {
    type Output = Matrix;
    fn sub(self, rhs: &SubMatrix) -> Matrix {
        self.as_matrix().concat(&rhs.as_negated_matrix())
    }
}

forward_ref_binop!(Add, add, Matrix, Matrix);
forward_ref_binop!(Sub, sub, Matrix, Matrix);
forward_ref_binop!(Add, add, Matrix, SubMatrix);
forward_ref_binop!(Sub, sub, Matrix, SubMatrix);
forward_ref_binop!(Add, add, SubMatrix, Matrix);
forward_ref_binop!(Sub, sub, SubMatrix, Matrix);
forward_ref_binop!(Add, add, SubMatrix, SubMatrix);
forward_ref_binop!(Sub, sub, SubMatrix, SubMatrix);

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
                let nc = a.col_keys.len();
                let mut data = vec![0.0f64; a.row_keys.len() * nc];
                for (r, c, v) in a.csr.triplets() {
                    data[r * nc + c] = v;
                }
                (
                    Self::name_dofs(&a.row_keys, &a.vars),
                    Self::name_dofs(&a.col_keys, &a.vars),
                    data,
                )
            } else {
                let vars = self.field_names();
                let (row_keys, col_keys) = self.dof_orders(&vars)?;
                let triplets = self.build_global_triplets(&vars, &row_keys, &col_keys)?;
                let row_dofs = Self::name_dofs(&row_keys, &vars);
                let col_dofs = Self::name_dofs(&col_keys, &vars);
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

// ─── Archive ────────────────────────────────────────────────────────────────

impl crate::archive::Archivable for SubMatrix {
    const TAG: &'static str = "SubMatrix";
    // No `on_load`: the block keeps no node list of its own — it reads its
    // supports, which the archive restores with it.
}

impl crate::archive::Archivable for Matrix {
    const TAG: &'static str = "Matrix";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Build a POI1 SubMesh with `n` fresh nodes in a new 1-D Coords.
    /// Returns `(coords, nodes, support_handle)`.
    fn make_poi1(n: usize) -> (Handle<Coords>, Vec<Node>, Handle<SubMesh>) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for node in &nodes {
            sm.add_cell(&[node.id()]).unwrap();
        }
        (coords, nodes, Handle::new(sm))
    }

    /// Build a POI1 SubMesh whose connectivity **repeats** its first node:
    /// cells are `[a, b, a, c, …]`. Returns `(coords, nodes, support_handle)`,
    /// where `nodes` holds the distinct nodes in first-appearance order.
    fn make_poi1_repeated(n: usize) -> (Handle<Coords>, Vec<Node>, Handle<SubMesh>) {
        let coords = Handle::new(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[nodes[0].id()]).unwrap();
        for node in &nodes[1..] {
            sm.add_cell(&[node.id()]).unwrap();
            sm.add_cell(&[nodes[0].id()]).unwrap();
        }
        (coords, nodes, Handle::new(sm))
    }

    /// Why a block's support must have **distinct** nodes.
    ///
    /// The block reads its numbering straight from the support: row `i` is
    /// `(connectivity()[node_local], dual_vars[var])`, and `add_entry` / `get`
    /// find `node_local` through [`SubMesh::node_index`]. The two only agree
    /// when no node repeats — `node_index` stores a **dedup-compacted rank**,
    /// the connectivity a **flat position**.
    ///
    /// Every support in practice comes from `to_poi1`, which deduplicates, so
    /// the two coincide. This test pins both halves so the requirement cannot
    /// be forgotten: it is documented on the constructors, never validated.
    #[test]
    fn node_index_is_the_flat_position_only_on_a_deduplicated_support() {
        // Distinct nodes: rank and position coincide, for every node.
        let (_c, nodes, sup) = make_poi1(4);
        let guard = sup.read();
        for (position, nid) in guard.connectivity().iter().enumerate() {
            assert_eq!(
                guard.node_index()[nid],
                position,
                "deduplicated support: rank and position must agree on {nid:?}"
            );
        }
        assert_eq!(guard.node_index().len(), nodes.len());
        drop(guard);

        // Repeated node: they part company, and the gap is not benign — the
        // compacted rank names a row the flat position does not own.
        let (_c2, nodes2, sup2) = make_poi1_repeated(3);
        let guard2 = sup2.read();
        // Connectivity is [a, b, a, c, a]: `b` agrees at 1, `c` does not —
        // position 3 in the connectivity, rank 2 once deduplicated.
        assert_eq!(guard2.connectivity().len(), 5);
        assert_eq!(guard2.node_index()[&nodes2[1].id()], 1);
        assert_eq!(guard2.connectivity()[3], nodes2[2].id());
        assert_eq!(guard2.node_index()[&nodes2[2].id()], 2);
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
            Symmetry::None,
        );
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.entry_count(), 0);
        assert!(!m.is_symmetric());
    }

    /// The declared share survives a save and a reload — deserialization being
    /// the fourth way a block enters the world, alongside the three
    /// constructors. A `Half` that lost its identity on the way back would
    /// silently stop pairing, and nothing would say so.
    ///
    /// The value alone is round-tripped, not a whole block: a `SubMatrix` holds
    /// `Handle`s, which refuse to serialize outside an archive.
    #[test]
    fn symmetry_round_trips_through_serde() {
        let (left, right) = mint_pair(0x1234_5678_9abc_def0);
        for declared in [Symmetry::Full, Symmetry::None, Symmetry::Half(left)] {
            let bytes =
                bincode::serde::encode_to_vec(declared, bincode::config::standard()).unwrap();
            let (back, _): (Symmetry, usize) =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(back, declared, "{declared:?} did not survive");
        }
        assert_ne!(left, right);
    }

    /// The two identities of a pair differ, and only in the bit that says so.
    /// Two contents that differ give two pairs that differ.
    #[test]
    fn mint_pair_splits_one_content_in_two() {
        let (a, b) = mint_pair(0xdead_beef_0000_1111);
        assert_ne!(a, b);
        assert_eq!(a & !1, b & !1, "both halves name the same pair");
        assert_eq!(a & 1, 0);
        assert_eq!(b & 1, 1);
        // Minting is a function of the content alone — the same content reminted
        // gives the same pair, which is what lets an archive be reproducible.
        assert_eq!(mint_pair(0xdead_beef_0000_1111), (a, b));

        let mut h = FNV_SEED;
        h = hash_name(h, "T");
        let mut other = FNV_SEED;
        other = hash_name(other, "P");
        assert_ne!(mint_pair(h).0 & !1, mint_pair(other).0 & !1);
    }

    /// A pair counts as one symmetry when both halves are there, and as none
    /// when one has been sliced away. Neither half is symmetric on its own.
    #[test]
    fn a_half_counts_only_with_its_other_half() {
        // One configuration, so the two node ids cannot collide.
        let (coords, nodes, _) = make_poi1(2);
        let (a, m) = (nodes[0].id(), nodes[1].id());
        let one = |n: NodeId| {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[n]).unwrap();
            Handle::new(sm)
        };
        let (sup_a, sup_m) = (one(a), one(m));
        let (id_c, id_ct) = mint_pair(0x5555_5555_5555_5555);
        // C: rows on the multiplier, columns on the constrained node — and Cᵀ
        // the other way round, with the variables crossed.
        let c = SubMatrix::new(
            sup_m.clone(),
            sup_a.clone(),
            vec!["imposed_T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Half(id_c),
        );
        let ct = SubMatrix::new(
            sup_a,
            sup_m,
            vec!["T".into()],
            vec!["imposed_T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Half(id_ct),
        );
        assert!(!c.is_symmetric(), "a half is not symmetric on its own");

        let mut both = Matrix::empty();
        both.add_sub(Handle::new(c)).unwrap();
        both.add_sub(Handle::new(ct)).unwrap();
        assert!(both.symmetric(), "the pair is complete");

        let halved = both.subset([0]).unwrap();
        assert!(!halved.symmetric(), "one half alone carries no symmetry");
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
        assert_eq!(
            s.lines().count(),
            6,
            "summary + 2 metadata lines + header + 2 rows:\n{s}"
        );
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
            Symmetry::Full,
        );
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
    fn sub_matrix_mul_and_div_scale_only_the_factor() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let mut m = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
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
                Symmetry::None,
            )
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
        k.add_sub(Handle::new(bare)).unwrap();
        k.add_sub(Handle::new(coupled)).unwrap();
        k.add_sub(Handle::new(other)).unwrap();
        let _ = a; // silence unused in some build configs

        // Containment: the coupled block appears under both its natures.
        assert_eq!(k.filter(Physics::Mechanical).len(), 1);
        assert_eq!(k.filter(Physics::Thermal).len(), 1);
        // Only the explicitly-tagged block is reached by Other; the bare one never.
        assert_eq!(k.filter(Physics::Other).len(), 1);
        // The aggregate reports every distinct nature present (bare contributes none).
        let present = k.physics();
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
            Symmetry::Full,
        );
        m.add_entry(a, "q", a, "T", 2.0).unwrap();
        let d = format!("{:?}", m);
        assert!(d.contains("SubMatrix"));
        assert!(d.contains("n_rows"));
        assert!(d.contains("symmetry"));
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
            Symmetry::None,
        );
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
            Symmetry::None,
        );
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
        assert!(m.symmetric());
        assert_eq!(m.entry_count(), 0);
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

        // Block a: 1 row (na) × 2 cols (ca0, ca1) — rectangular, hence `None`:
        // a block that is not square cannot carry any symmetry.
        let mut row_a = SubMesh::new(coords.clone(), ElementType::POI1);
        row_a.add_cell(&[na]).unwrap();
        let mut col_a = SubMesh::new(coords.clone(), ElementType::POI1);
        col_a.add_cell(&[ca0]).unwrap();
        col_a.add_cell(&[ca1]).unwrap();
        let mut a = SubMatrix::new(
            Handle::new(row_a),
            Handle::new(col_a),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
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
            Handle::new(row_b),
            Handle::new(col_b),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        b.add_entry(m0, "q", m0, "T", 0.5).unwrap();
        b.add_entry(m1, "q", m0, "T", -1.0).unwrap();
        b.add_entry(m1, "q", m1, "T", 2.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();

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
            Symmetry::Full,
        );
        let b = SubMatrix::new(
            sup.clone(),
            sup,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();
        assert!(!k.symmetric());
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
            Symmetry::Full,
        );
        blk.add_entry(a, "q", a, "T", 2.0).unwrap();
        blk.add_entry(b, "q", b, "T", 3.0).unwrap();
        let mut orig = Matrix::empty();
        orig.add_sub(Handle::new(blk)).unwrap();

        let scaled = &orig * 10.0;

        // A new store slot per block — no aliasing with the source's blocks.
        let orig_h = orig.iter().next().unwrap();
        let scaled_h = scaled.iter().next().unwrap();
        assert!(!orig_h.same_object(scaled_h));

        // Values diverge accordingly: the source is untouched.
        assert_eq!(orig.get(a, "q", a, "T"), 2.0);
        assert_eq!(orig.get(b, "q", b, "T"), 3.0);
        assert_eq!(scaled.get(a, "q", a, "T"), 20.0);
        assert_eq!(scaled.get(b, "q", b, "T"), 30.0);

        // `/` divides the factor, chaining from the already-scaled matrix.
        let halved = &scaled / 2.0;
        assert_eq!(halved.get(a, "q", a, "T"), 10.0);
        assert_eq!(
            scaled.get(a, "q", a, "T"),
            20.0,
            "/ must not mutate its source either"
        );
    }

    /// A one-block literal matrix on `sup`, carrying `value` on the diagonal of
    /// each node of `nodes`.
    fn diag(sup: &Handle<SubMesh>, nodes: &[NodeId], value: f64) -> Matrix {
        let mut blk = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        for &n in nodes {
            blk.add_entry(n, "q", n, "T", value).unwrap();
        }
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(blk)).unwrap();
        m
    }

    /// Scaling an **assembled** matrix carries the CSR along, scaled, instead of
    /// dropping it. What that buys: the result is usable without a second
    /// assembly — which, on a matrix with computed blocks, would re-run every
    /// element kernel to apply one scalar.
    #[test]
    fn scaling_carries_the_assembled_csr() {
        let (_cfg, nodes, sup) = make_poi1(2);
        let (a, b) = (nodes[0].id(), nodes[1].id());
        let mut k = diag(&sup, &[a, b], 2.0);
        k.finalize().unwrap();

        let scaled = &k * 3.0;

        // Assembled on arrival: no `finalize()` / `assemble()` in between.
        assert_eq!(scaled.to_csr().unwrap().values(), &[6.0, 6.0]);
        assert_eq!(scaled.get(a, "q", a, "T"), 6.0);
        // The sparsity is untouched, so the index arrays are *shared*, not copied.
        let (ka, sa) = (
            k.assembled.as_ref().unwrap(),
            scaled.assembled.as_ref().unwrap(),
        );
        assert!(std::sync::Arc::ptr_eq(
            &ka.csr.row_offsets,
            &sa.csr.row_offsets
        ));
        assert!(std::sync::Arc::ptr_eq(
            &ka.csr.col_indices,
            &sa.csr.col_indices
        ));
        // And the source keeps its own values.
        assert_eq!(k.to_csr().unwrap().values(), &[2.0, 2.0]);

        // An unassembled source still yields an unassembled result.
        let raw = diag(&sup, &[a], 1.0);
        assert!((&raw * 2.0).to_csr().is_err());
    }

    /// The owning `*` rescales a block **no one else holds** where it lies; a
    /// block another matrix shares is rebuilt, so that other matrix is spared.
    #[test]
    fn owned_scaling_reuses_a_block_no_one_else_holds() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();

        let sole = diag(&sup, &[a], 2.0);
        let slot = sole.iter().next().unwrap().id();
        let scaled = sole * 3.0;
        assert_eq!(
            scaled.iter().next().unwrap().id(),
            slot,
            "sole owner: in place"
        );
        assert_eq!(scaled.get(a, "q", a, "T"), 6.0);

        // Now with a second holder of the same block.
        let shared = diag(&sup, &[a], 2.0);
        let alias = shared.subset([0]).unwrap();
        let slot = shared.iter().next().unwrap().id();
        let scaled = shared * 3.0;
        assert_ne!(scaled.iter().next().unwrap().id(), slot, "shared: rebuilt");
        assert_eq!(scaled.get(a, "q", a, "T"), 6.0);
        assert_eq!(
            alias.get(a, "q", a, "T"),
            2.0,
            "the aliasing matrix is spared"
        );
    }

    /// What parts `+` from `|`: a sum counts a contribution once per handing
    /// over, a union drops a block whose slot it already holds.
    #[test]
    fn sum_counts_a_shared_block_twice_where_union_drops_it() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let k = diag(&sup, &[a], 2.0);

        let mut sum = &k + &k;
        sum.finalize().unwrap();
        assert_eq!(sum.len(), 2);
        assert_eq!(sum.get(a, "q", a, "T"), 4.0, "k + k is 2k");

        let mut union = k.union(&k).unwrap();
        union.finalize().unwrap();
        assert_eq!(union.len(), 1);
        assert_eq!(union.get(a, "q", a, "T"), 2.0, "k | k is k");

        // The sum shares its operands' blocks — nothing is copied.
        assert!(sum
            .iter()
            .next()
            .unwrap()
            .same_object(k.iter().next().unwrap()));
        // …and leaves the source alone.
        assert_eq!(k.get(a, "q", a, "T"), 2.0);
    }

    /// `a - b` negates the right-hand side's blocks, which copies them; `a + b`
    /// copies nothing. Both leave their operands alone.
    #[test]
    fn difference_negates_only_the_right_hand_side() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let x = diag(&sup, &[a], 5.0);
        let y = diag(&sup, &[a], 2.0);

        let mut d = &x - &y;
        d.finalize().unwrap();
        assert_eq!(d.get(a, "q", a, "T"), 3.0);
        assert_eq!(y.get(a, "q", a, "T"), 2.0, "the subtrahend is untouched");
        assert!(!d
            .iter()
            .nth(1)
            .unwrap()
            .same_object(y.iter().next().unwrap()));

        // A matrix minus itself is zero — the structural sum, then the assembler.
        let mut z = &x - &x;
        z.finalize().unwrap();
        assert_eq!(z.get(a, "q", a, "T"), 0.0);
    }

    /// The scalar reads on either side, and the unary minus agrees with `× -1`.
    #[test]
    fn scalar_on_the_left_and_unary_minus() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let k = diag(&sup, &[a], 2.0);

        assert_eq!((2.0 * &k).get(a, "q", a, "T"), 4.0);
        assert_eq!((-&k).get(a, "q", a, "T"), -2.0);
        assert_eq!((-&k).get(a, "q", a, "T"), (&k * -1.0).get(a, "q", a, "T"));

        // On a block too.
        let blk = (*k.iter().next().unwrap().read()).clone();
        assert_eq!((3.0 * &blk).factor(), 3.0);
        assert_eq!((-&blk).factor(), -1.0);
    }

    /// A block and an aggregate mix in a sum, either way round.
    #[test]
    fn a_block_and_a_matrix_mix_in_a_sum() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let a = nodes[0].id();
        let k = diag(&sup, &[a], 2.0);
        let blk = (*k.iter().next().unwrap().read()).clone();

        for mut s in [&k + &blk, &blk + &k] {
            s.finalize().unwrap();
            assert_eq!(s.len(), 2);
            assert_eq!(s.get(a, "q", a, "T"), 4.0);
        }
        let mut two_blocks = &blk + &blk;
        two_blocks.finalize().unwrap();
        assert_eq!(two_blocks.get(a, "q", a, "T"), 4.0);

        let mut d = &k - &blk;
        d.finalize().unwrap();
        assert_eq!(d.get(a, "q", a, "T"), 0.0);
    }

    /// A divisor no matrix survives is refused at the operator, not left to
    /// surface as a `NaN` inside the solver.
    #[test]
    #[should_panic(expected = "matrix division by 0")]
    fn division_by_zero_is_refused() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let k = diag(&sup, &[nodes[0].id()], 2.0);
        let _ = &k / 0.0;
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn division_by_infinity_is_refused() {
        let (_cfg, nodes, sup) = make_poi1(1);
        let k = diag(&sup, &[nodes[0].id()], 2.0);
        let _ = &k / f64::INFINITY;
    }

    /// `M/dt + K` through the **union**, which stays the way to compose one
    /// operator out of distinct parts (`+` is the algebraic sum — see
    /// `sum_counts_a_shared_block_twice_where_union_drops_it`). `K` carries a
    /// DOF (`c`) that `M` doesn't (mirroring a Dirichlet multiplier row/column,
    /// which only ever enters the stiffness matrix) — the union must still
    /// assemble correctly, leaving that entry untouched by `M`'s contribution.
    #[test]
    fn union_and_reassemble_combines_scaled_mass_with_stiffness() {
        let (coords, nodes, _) = make_poi1(3);
        let (a, b, c) = (nodes[0].id(), nodes[1].id(), nodes[2].id());

        let mut sup_k = SubMesh::new(coords.clone(), ElementType::POI1);
        sup_k.add_cell(&[a]).unwrap();
        sup_k.add_cell(&[b]).unwrap();
        sup_k.add_cell(&[c]).unwrap();
        let sup_k = Handle::new(sup_k);
        let mut k_blk = SubMatrix::new(
            sup_k.clone(),
            sup_k,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        k_blk.add_entry(a, "q", a, "T", 2.0).unwrap();
        k_blk.add_entry(a, "q", b, "T", -1.0).unwrap();
        k_blk.add_entry(b, "q", a, "T", -1.0).unwrap();
        k_blk.add_entry(b, "q", b, "T", 2.0).unwrap();
        k_blk.add_entry(c, "q", c, "T", 5.0).unwrap(); // Lagrange-only DOF
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(k_blk)).unwrap();

        let mut sup_m = SubMesh::new(coords.clone(), ElementType::POI1);
        sup_m.add_cell(&[a]).unwrap();
        sup_m.add_cell(&[b]).unwrap();
        let sup_m = Handle::new(sup_m);
        let mut m_blk = SubMatrix::new(
            sup_m.clone(),
            sup_m,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        m_blk.add_entry(a, "q", a, "T", 4.0).unwrap();
        m_blk.add_entry(b, "q", b, "T", 4.0).unwrap();
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(m_blk)).unwrap();

        let dt = 0.5;
        let m_dt = &m / dt; // factor = 1/0.5 = 2 ⇒ diag(8, 8)

        let mut sys = m_dt.union(&k).unwrap();
        sys.assemble().unwrap();

        assert_eq!(sys.n_rows().unwrap(), 3);
        assert_eq!(sys.n_cols().unwrap(), 3);
        assert_eq!(sys.get(a, "q", a, "T"), 2.0 + 8.0);
        assert_eq!(sys.get(a, "q", b, "T"), -1.0);
        assert_eq!(sys.get(b, "q", a, "T"), -1.0);
        assert_eq!(sys.get(b, "q", b, "T"), 2.0 + 8.0);
        // Untouched by M (which doesn't carry the c DOF at all).
        assert_eq!(sys.get(c, "q", c, "T"), 5.0);
    }

    #[test]
    fn aggregate_to_dmatrix_layout_is_union_first_seen() {
        // Two distinct nodes in the SAME configuration to avoid NodeId collisions.
        let (coords, nodes, _) = make_poi1(2);
        let na = nodes[0].id();
        let nb = nodes[1].id();

        let mut sm_a = SubMesh::new(coords.clone(), ElementType::POI1);
        sm_a.add_cell(&[na]).unwrap();
        let sup_a = Handle::new(sm_a);

        let mut sm_b = SubMesh::new(coords.clone(), ElementType::POI1);
        sm_b.add_cell(&[nb]).unwrap();
        let sup_b = Handle::new(sm_b);

        let mut a = SubMatrix::new(
            sup_a.clone(),
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        a.add_entry(na, "q", na, "T", 2.0).unwrap();

        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        b.add_entry(nb, "q", nb, "T", 3.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();
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
            Handle::new(sm)
        };
        let sup_a = mk(na);
        let sup_b = mk(nb);
        let mut a = SubMatrix::new(
            sup_a.clone(),
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        a.add_entry(na, "q", na, "T", 2.0).unwrap();
        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        b.add_entry(nb, "q", nb, "T", 3.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();

        // Transposition of the identity ⇒ nb sorts before na.
        let cap = coords.read().capacity();
        let mut perm: Vec<u32> = (0..cap as u32).collect();
        perm.swap(na.0 as usize, nb.0 as usize);
        coords.write().set_permutation(perm).unwrap();

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
            Symmetry::None,
        );
        m.add_entry(a, "q", a, "T", 1.0).unwrap();
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(m)).unwrap();
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
        let mut row_a = SubMesh::new(sup_a.read().coords(), ElementType::POI1);
        row_a.add_cell(&[na0]).unwrap();
        let row_a_h = Handle::new(row_a);

        // One row against a two-node column support: rectangular, so `None`.
        let mut a = SubMatrix::new(
            row_a_h,
            sup_a.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        a.add_entry(na0, "q", na0, "T", 2.0).unwrap();
        a.add_entry(na0, "q", na1, "T", -1.0).unwrap();

        let mut row_b = SubMesh::new(sup_a.read().coords(), ElementType::POI1);
        row_b.add_cell(&[na1]).unwrap();
        let row_b_h = Handle::new(row_b);

        let mut b = SubMatrix::new(
            row_b_h,
            sup_a,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        b.add_entry(na1, "q", na0, "T", -1.0).unwrap();
        b.add_entry(na1, "q", na1, "T", 2.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();
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
            Symmetry::Full,
        );
        sm.add_entry(a, "q", a, "T", 2.0).unwrap();
        sm.add_entry(a, "q", b, "T", -1.0).unwrap();
        sm.add_entry(b, "q", a, "T", -1.0).unwrap();
        sm.add_entry(b, "q", b, "T", 2.0).unwrap();
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(sm)).unwrap();
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
        assert!(y.value_opt(a, "T").is_none());

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
            Symmetry::None,
        );
        a.add_entry(na, "q", na, "T", 1.0).unwrap();

        let mut b = SubMatrix::new(
            sup_b.clone(),
            sup_b,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        b.add_entry(nb, "q", nb, "T", 2.0).unwrap();

        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
        k.add_sub(Handle::new(b)).unwrap();

        let entries = k.iter_entries();
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
            Symmetry::Full,
        );
        a.add_entry(a_id, "q", a_id, "T", 2.0).unwrap();
        let mut k = Matrix::empty();
        k.add_sub(Handle::new(a)).unwrap();
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
        let coords = Handle::new(Coords::new(1).unwrap());
        let phys_nodes: Vec<Node> = (0..2)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mult_node = Node::create_in(coords.clone(), &[10.0]).unwrap();
        let phys = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in &phys_nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            Handle::new(sm)
        };
        let mult = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[mult_node.id()]).unwrap();
            Handle::new(sm)
        };

        // K (phys × phys), C (mult × phys), Cᵀ (phys × mult).
        let mut k = SubMatrix::new(
            phys.clone(),
            phys.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        k.add_entry(phys_nodes[0].id(), "q", phys_nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut c = SubMatrix::new(
            mult.clone(),
            phys.clone(),
            vec!["imposed_T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        c.add_entry(mult_node.id(), "imposed_T", phys_nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut ct = SubMatrix::new(
            phys.clone(),
            mult.clone(),
            vec!["q".into()],
            vec!["lambda_T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        ct.add_entry(phys_nodes[0].id(), "q", mult_node.id(), "lambda_T", 1.0)
            .unwrap();

        let mut m = Matrix::empty();
        m.add_sub(Handle::new(k)).unwrap();
        m.add_sub(Handle::new(c)).unwrap();
        m.add_sub(Handle::new(ct)).unwrap();

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
        assert!(f.get(0).unwrap().read().support().same_object(&phys));
        assert!(f.get(1).unwrap().read().support().same_object(&mult));

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
            Symmetry::None,
        );
        a.add_entry(nodes[0].id(), "q", nodes[0].id(), "T", 1.0)
            .unwrap();
        let mut b = SubMatrix::new(
            sup.clone(),
            sup.clone(),
            vec!["r".into()],
            vec!["P".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        b.add_entry(nodes[1].id(), "r", nodes[1].id(), "P", 1.0)
            .unwrap();

        let mut m = Matrix::empty();
        m.add_sub(Handle::new(a)).unwrap();
        m.add_sub(Handle::new(b)).unwrap();
        m.finalize().unwrap();

        let col_dofs = m.col_dofs().unwrap();
        let x: Vec<f64> = (0..col_dofs.len()).map(|i| 10.0 + i as f64).collect();
        let f = m.field_from_col_values(&x).unwrap();

        assert_eq!(f.len(), 1, "same support ⇒ one zone");
        {
            let z = f.get(0).unwrap().read();
            assert!(z.support().same_object(&sup));
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
        let coords = Handle::new(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let nm = Node::create_in(coords.clone(), &[10.0]).unwrap();
        let phys = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[n0.id()]).unwrap();
            sm.add_cell(&[n1.id()]).unwrap();
            Handle::new(sm)
        };
        let mult = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nm.id()]).unwrap();
            Handle::new(sm)
        };
        // K (phys × phys) and C (mult × phys): row supports {phys, mult},
        // col supports {phys} — K and C share the phys column support.
        let mut k = SubMatrix::new(
            phys.clone(),
            phys.clone(),
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::Full,
        );
        k.add_entry(n0.id(), "q", n0.id(), "T", 2.0).unwrap();
        k.add_entry(n1.id(), "q", n1.id(), "T", 2.0).unwrap();
        let mut c = SubMatrix::new(
            mult.clone(),
            phys.clone(),
            vec!["imposed_T".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        c.add_entry(nm.id(), "imposed_T", n0.id(), "T", 1.0)
            .unwrap();
        let mut m = Matrix::empty();
        m.add_sub(Handle::new(k)).unwrap();
        m.add_sub(Handle::new(c)).unwrap();

        // Meshes: available pre-finalize, deduplicated, sharing the handles.
        let rm = m.row_mesh().unwrap();
        assert_eq!(rm.len(), 2);
        assert!(rm.get(0).unwrap().same_object(&phys));
        assert!(rm.get(1).unwrap().same_object(&mult));
        let cm = m.col_mesh().unwrap();
        assert_eq!(cm.len(), 1, "K and C share the phys column support");
        assert!(cm.get(0).unwrap().same_object(&phys));

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
            let mut s = SubNodeField::from_poi1(&Handle::new(sm), vec!["q".into()]).unwrap();
            s.set_value(n0.id(), "q", 5.0).unwrap();
            s.set_value(n1.id(), "q", 5.0).unwrap();
            NodeField::from_sub(s)
        };
        let f_ext_r = crate::ops::node_field::restrict(&f_ext, &rm).unwrap();
        for (za, zb) in f_ext_r.iter().zip(f_int.iter()) {
            let sa = za.read().support();
            let sb = zb.read().support();
            assert!(
                sa.same_object(&sb),
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
        let f_ext_like = crate::ops::node_field::restrict_like(&f_ext, &f_int).unwrap();
        let r2 = (&f_ext_like - &f_int).unwrap();
        assert_eq!(r2.value(n0.id(), "q").unwrap(), 3.0);
        assert_eq!(r2.value(nm.id(), "imposed_T").unwrap(), -1.0); // 0 − 1
    }

    /// Row-side twin: zones on the blocks' row supports, dual variables.
    #[test]
    fn field_from_row_values_uses_row_supports_and_dual_vars() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let r0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let r1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c0 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let sup_r = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[r0.id()]).unwrap();
            sm.add_cell(&[r1.id()]).unwrap();
            Handle::new(sm)
        };
        let sup_c = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[c0.id()]).unwrap();
            Handle::new(sm)
        };
        let mut blk = SubMatrix::new(
            sup_r.clone(),
            sup_c,
            vec!["q".into()],
            vec!["T".into()],
            DofOrdering::NodesThenVars,
            Symmetry::None,
        );
        blk.add_entry(r0.id(), "q", c0.id(), "T", 1.0).unwrap();

        let mut m = Matrix::empty();
        m.add_sub(Handle::new(blk)).unwrap();
        m.finalize().unwrap();

        let f = m.field_from_row_values(&[3.0, 7.0]).unwrap();
        assert_eq!(f.len(), 1);
        assert!(f.get(0).unwrap().read().support().same_object(&sup_r));
        assert_eq!(f.value(r0.id(), "q").unwrap(), 3.0);
        assert_eq!(f.value(r1.id(), "q").unwrap(), 7.0);
    }
}
