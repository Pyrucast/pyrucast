//! Per-physics implementations of [`crate::containers::model::SubModel`]
//! variants.
//!
//! Each file here owns the **specifics** of one physics: a struct holding
//! its supports (FE spaces, materials, node sets) plus an [`impl SubModelKind`]
//! carrying *all* of its behaviour — variable names, material contract,
//! local assembly, and rendering. The
//! [`crate::containers::model::SubModel`] enum exists **only** for storage
//! and serialization; it dispatches every call through
//! [`SubModel::as_kind`](crate::containers::model::SubModel::as_kind)
//! so no generic code (the assembler, `Dump`, …) ever needs a per-variant
//! `match`.
//!
//! # Adding a new physics
//!
//! 1. add `models/<name>.rs` with a struct + `impl SubModelKind` (and a
//!    `new(...)` constructor doing any build-time work);
//! 2. add one variant to [`crate::containers::model::SubModel`];
//! 3. add one arm to
//!    [`SubModel::as_kind`](crate::containers::model::SubModel::as_kind);
//! 4. declare the parent-level operator with
//!    [`physics_operator!`](crate::physics_operator) **in the same file**: it
//!    emits the Rust operator *and* its `#[pyfunction]`, so nothing is written
//!    by hand under `py/`;
//! 5. raccord: `pub use` in [`crate::ops::model`], and the registration in the
//!    (flat) `#[pymodule]` of `lib.rs`.
//!
//! Everything else is generic. See the book chapter *« Ajouter une
//! physique »* for the full walkthrough.

use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::SubNodeField;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use serde::{Deserialize, Serialize};

pub mod beam;
pub mod bernoulli;
pub mod boundary_transfer;
pub mod contact;
pub mod continuum;
pub mod damage;
pub mod dirichlet;
pub mod elasticity;
pub mod embedded;
pub mod fick;
pub mod frame;
pub mod frame3d;
pub mod heat_conduction;
pub mod interface_transfer;
pub mod kernel;
pub mod mpc;
pub mod plasticity;
pub mod radiation;
pub mod shell;
pub mod symmetry;
pub mod tensor;
pub mod timoshenko;
pub mod transfer;
pub mod truss;

pub use kernel::CellGeom;

/// The kind of element matrix a physics contributes — the discriminant that
/// makes the whole assembly pipeline (recipe → scatter → per-kind pattern cache)
/// matrix-agnostic. One `assemble_*` entry point per variant
/// ([`crate::ops::matrix`]) drives the **same** machinery with a different
/// per-element kernel; a physics that has no term for a given kind contributes
/// nothing (its [`matrix_layout`](SubModelKind::matrix_layout) returns `None`).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Quatre natures de matrice, et **une seule** machinerie : un noyau par
/// // élément diffère, le reste est partagé. Une physique sans terme pour
/// // une nature n'y contribue rien.
/// assert_eq!(MatrixKind::COUNT, 4);
/// assert_eq!(MatrixKind::Stiffness.index(), 0);
/// assert!(volume.as_kind().matrix_layout(MatrixKind::Stiffness).is_some());
/// // Une contrainte n'a pas de disposition matricielle : elle passe par
/// // `contributions`, non par un noyau d'élément.
/// assert!(appui.as_kind().matrix_layout(MatrixKind::Stiffness).is_none());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixKind {
    /// Stiffness / conductivity `K` — `∫ Bᵀ D B` (Cast3M `RIGI` / `COND`).
    #[default]
    Stiffness,
    /// Mass / capacity `M` — `∫ ρ Nᵀ N` (Cast3M `MASS` / `CAPA`).
    Mass,
    /// Geometric (initial-stress) stiffness `K_g` — `∫ Gᵀ σ̂ G` (Cast3M `KSIG`).
    Geometric,
    /// Consistent tangent `K_t` — `∫ Bᵀ D_alg B` (Cast3M `KTAN`).
    Tangent,
}

impl MatrixKind {
    /// Number of variants — the width of the per-kind pattern cache.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
    /// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
    /// #     None, None, RelationSense::Equality).unwrap();
    /// // Quatre natures de matrice, et **une seule** machinerie : un noyau par
    /// // élément diffère, le reste est partagé. Une physique sans terme pour
    /// // une nature n'y contribue rien.
    /// assert_eq!(MatrixKind::COUNT, 4);
    /// assert_eq!(MatrixKind::Stiffness.index(), 0);
    /// assert!(volume.as_kind().matrix_layout(MatrixKind::Stiffness).is_some());
    /// // Une contrainte n'a pas de disposition matricielle : elle passe par
    /// // `contributions`, non par un noyau d'élément.
    /// assert!(appui.as_kind().matrix_layout(MatrixKind::Stiffness).is_none());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub const COUNT: usize = 4;

    /// Dense index in `0..COUNT`, for indexing per-kind caches.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
    /// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
    /// #     None, None, RelationSense::Equality).unwrap();
    /// // L'index sert de rang dans le cache de motifs, un par nature.
    /// assert!(MatrixKind::Stiffness.index() < MatrixKind::COUNT);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Structural declaration a volumetric physics gives so the **global**
/// assembler ([`crate::ops::matrix::stiffness`]) can build its stiffness
/// contribution as a *computed* [`SubMatrix`] — a recipe, no eagerly
/// materialised values — and scatter it straight into the global CSR.
///
/// Every field mirrors, one-for-one, what the physics'
/// [`build_stiffness_blocks`](SubModelKind::build_stiffness_blocks) would pass to
/// [`kernel::assemble_block`]. The
/// literal `build_stiffness_blocks` is **kept** alongside it as the bit-for-bit
/// equivalence reference. Volumetric blocks are square on a single support, so
/// one [`SubMesh`] gives both the row and column node sequence.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // La déclaration **structurelle** d'une physique de volume : de quoi
/// // bâtir un bloc *calculé* et le verser droit dans le CSR global. Chaque
/// // champ reflète un argument d'`assemble_block`.
/// let l = volume.as_kind().matrix_layout(MatrixKind::Stiffness).unwrap();
/// assert_eq!(l.dual_vars, vec!["q".to_string()]);
/// assert_eq!(l.primal_vars, vec!["T".to_string()]);
/// assert!(l.symmetric);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MatrixLayout {
    /// FE subspaces the element kernel integrates over. **Give a `Vec`**: a
    /// single subspace for a plain volumetric physics, or several — sharing one
    /// submesh, differing only by quadrature — for a multi-quadrature element
    /// (a shear-deformable beam, a shell). The primary (index 0) drives the cell
    /// loop and the DOF numbering; [`element_matrix`](Domain::element_matrix)
    /// receives one [`CellGeom`] per subspace, in this order.
    pub fespaces: Vec<Handle<SubFiniteElementSpace>>,
    /// POI1 sub-mesh giving the block's row **and** column node sequence.
    pub support: Handle<SubMesh>,
    /// Row variable names (dual).
    pub dual_vars: Vec<String>,
    /// Column variable names (primal).
    pub primal_vars: Vec<String>,
    /// `(node_local, var)` ↔ matrix-index ordering.
    pub ordering: DofOrdering,
    /// Whether the block is numerically symmetric.
    pub symmetric: bool,
}

/// One stiffness contribution of a sub-model, as handed to the global
/// assembler ([`crate::ops::matrix::stiffness`]).
///
/// A sub-model declares *how* each of its blocks is produced without the
/// assembler needing to know its concrete type: it just iterates
/// [`SubModelKind::contributions`] and folds each variant in. This is the seam that
/// keeps `Dirichlet`, a volumetric physics, and (later) a coupling/contact
/// sub-model on one uniform path — the discriminant is the variant, not a
/// per-type `match` in the assembler.
///
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// # use pyrucast::models::Contribution;
/// // Le discriminant qui met sur une seule voie une contrainte, une
/// // physique de volume et un couplage : c'est la variante qui décide, non
/// // un `match` par type dans l'assembleur.
/// let c = appui.as_kind().contributions(MatrixKind::Stiffness, None)?;
/// assert!(!c.is_empty());
/// // Un appui apporte des blocs **littéraux** ; une physique de volume,
/// // une recette calculée.
/// assert!(matches!(c[0], Contribution::Literal(_)));
/// assert!(matches!(
///     volume.as_kind().contributions(MatrixKind::Stiffness, None)?[0],
///     Contribution::Computed(_)));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub enum Contribution {
    /// A block integrated on the fly and scattered straight into the global CSR
    /// (no values materialised): the fast path of every volumetric physics. The
    /// assembler resolves the material and wraps this
    /// [`MatrixLayout`] into a computed [`SubMatrix`].
    Computed(MatrixLayout),
    /// One-or-more blocks whose values the sub-model has already filled in
    /// (`Dirichlet`'s C / Cᵀ, MPC, …). Scattered by the literal path.
    Literal(Vec<SubMatrix>),
    /// An **off-diagonal** block coupling two distinct meshes — integrated on the
    /// fly like [`Computed`](Self::Computed), but with its rows on one mesh and
    /// its columns on another (an interface exchange law, `h(c₁ − c₂)`).
    ///
    /// Everything below this seam was already row/col-asymmetric —
    /// [`SubMatrix::computed`](crate::containers::matrix::SubMatrix::computed)
    /// takes both supports, and so do the scatter and the kernel drivers. Only
    /// [`MatrixLayout`] collapsed them into one field, which is why this is a
    /// separate layout rather than an extra field there: adding one would have
    /// touched every existing physics for a need none of them has.
    Coupling(CouplingLayout),
}

/// Structural declaration of an **inter-mesh** block: rows integrated on one
/// mesh, columns on another, paired cell by cell.
///
/// The two sides must be **conforming** — same element type, same cell count,
/// cell `i` of one facing cell `i` of the other. That is checked when the block
/// is built, and reported rather than approximated: a non-matching interface is
/// a meshing problem, not something an assembler should paper over.
// ANCHOR: coupling_layout
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// # use pyrucast::containers::matrix::DofOrdering;
/// # use pyrucast::models::CouplingLayout;
/// // Ce qu'une loi d'**interface** décrit : des sous-espaces de ligne *et*
/// // de colonne, sur deux maillages en vis-à-vis, là où un
/// // [`MatrixLayout`] tient sur un seul support. La conformité est
/// // vérifiée à la construction du bloc, et **signalée** plutôt
/// // qu'approximée : une interface qui ne correspond pas est un problème
/// // de maillage.
/// let l = CouplingLayout {
///     fespaces: vec![zone.clone()],
///     col_fespaces: vec![zone.clone()],
///     row_support: zone.read().submesh().read().to_poi1()?,
///     col_support: zone.read().submesh().read().to_poi1()?,
///     dual_vars: vec!["q".into()],
///     primal_vars: vec!["T".into()],
///     ordering: DofOrdering::NodesThenVars,
/// };
/// assert_eq!(l.fespaces.len(), l.col_fespaces.len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct CouplingLayout {
    /// FE subspaces carrying the **rows** (the primary drives the cell loop).
    pub fespaces: Vec<Handle<SubFiniteElementSpace>>,
    /// FE subspaces carrying the **columns**, on the facing mesh.
    pub col_fespaces: Vec<Handle<SubFiniteElementSpace>>,
    /// POI1 sub-mesh giving the block's row node sequence.
    pub row_support: Handle<SubMesh>,
    /// POI1 sub-mesh giving the block's column node sequence.
    pub col_support: Handle<SubMesh>,
    /// Row variable names (dual).
    pub dual_vars: Vec<String>,
    /// Column variable names (primal).
    pub primal_vars: Vec<String>,
    /// `(node_local, var)` ↔ matrix-index ordering.
    pub ordering: DofOrdering,
}
// ANCHOR_END: coupling_layout

/// The **nature** of a physics — its coarse classification, orthogonal to the
/// `Domain` / `Constraint` capability axis. It answers « quel champ de physique »
/// where the capability seams answer « domaine ou contrainte ».
///
/// A physics declares a **set** of natures (usually one): a plain physics is
/// single-natured, a coupled physics (e.g. a future thermo-mechanical element)
/// spans several, and a block that belongs to none is left **untagged** (an empty
/// set — the « rien » case for hand-built / other matrices). [`Other`](Self::Other)
/// is the explicit odd-one-out nature, for a block one *wants* classified as
/// « autre » rather than merely untagged.
///
/// A single nature is fully determined by the
/// [`SubModel`](crate::containers::model::SubModel) variant, so
/// [`SubModelKind::physics`] returns a per-physics **constant** slice (like
/// [`label`](SubModelKind::label)) rather than a stored field. The set travels
/// with each assembled block onto the [`SubMatrix`], and feeds the
/// [`Model::filter`](crate::containers::model::Model::filter) /
/// [`Matrix::filter`](crate::containers::matrix::Matrix::filter) selectors
/// (which match by **containment**).
///
/// ```
/// # use pyrucast::models::Physics;
/// // La nature d'un sous-modèle voyage avec chaque bloc assemblé : c'est
/// // elle que `Model::filter` et `Matrix::filter` lisent.
/// assert_eq!(Physics::ALL.len(), 6);
/// // Diffusion et thermique sont **distinctes** malgré leur laplacien
/// // commun : un problème couplé doit pouvoir choisir l'une sans l'autre.
/// assert_ne!(Physics::Diffusion, Physics::Thermal);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Physics {
    /// Solid mechanics — elasticity, plasticity, damage, bars, beams, frames.
    Mechanical,
    /// Heat conduction (and future thermal physics).
    Thermal,
    /// A Lagrange constraint — Dirichlet, MPC, embedded, contact.
    Constraint,
    /// « Autre / rien » — a nature for a block that fits none of the above but is
    /// still explicitly classified (as opposed to simply untagged).
    Other,
    /// Mass transport — Fickian diffusion and its interface laws. Distinct from
    /// [`Thermal`](Self::Thermal) despite sharing the Laplacian: the variables
    /// are a concentration and a mass flux, not a temperature and a heat flux,
    /// and a coupled problem must be able to select one without the other.
    Diffusion,
    /// Radiative exchange. Carried **in addition to**
    /// [`Thermal`](Self::Thermal) by a radiation sub-model — so it stays part of
    /// a thermal assembly, while `filter("radiation")` isolates the non-linear
    /// boundary term on its own.
    Radiation,
}

impl Physics {
    /// The lowercase name of this nature (the inverse of
    /// [`from_name`](crate::named::Named::from_name)).
    ///
    /// ```
    /// # use pyrucast::models::Physics;
    /// # use pyrucast::named::Named;
    /// // Réciproque exacte de `from_name`, pour les six natures.
    /// assert!(Physics::ALL.iter().all(|p| Physics::from_name(p.name()) == Some(*p)));
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Mechanical => "mechanical",
            Self::Thermal => "thermal",
            Self::Constraint => "constraint",
            Self::Other => "other",
            Self::Diffusion => "diffusion",
            Self::Radiation => "radiation",
        }
    }

    /// Every nature, in declaration order — the single source for the tag list
    /// quoted in the `filter` error messages, so a new nature cannot be added
    /// without the messages following.
    ///
    /// ```
    /// # use pyrucast::models::Physics;
    /// // L'unique source de la liste citée par les messages de `filter` : une
    /// // nature ne peut pas être ajoutée sans que les messages suivent.
    /// assert!(Physics::ALL.contains(&Physics::Radiation));
    /// ```
    pub const ALL: [Physics; 6] = [
        Self::Mechanical,
        Self::Thermal,
        Self::Constraint,
        Self::Other,
        Self::Diffusion,
        Self::Radiation,
    ];
}

impl crate::named::Named for Physics {
    const LABEL: &'static str = "physics";
    const VALUES: &'static [Self] = &Self::ALL;

    fn name(self) -> &'static str {
        Physics::name(self)
    }
}

impl std::fmt::Display for Physics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The behaviour contract of one physics, co-located with its data struct.
///
/// Generic code calls these through
/// [`SubModel::as_kind`](crate::containers::model::SubModel::as_kind);
/// the [`SubModel`](crate::containers::model::SubModel) enum itself carries
/// no logic. Most methods have sensible defaults so a physics overrides
/// only what is specific to it (a plain domain physics typically implements
/// `primal_vars`, `dual_vars`, `as_domain` + the [`Domain`] capability,
/// `element_matrix`, `stiffness_layout`, `label` and `render`).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Le contrat de base, avec des défauts partout où c'est possible : une
/// // physique n'écrit que ce qui lui est propre. C'est par lui que passe
/// // **toute** la couche modèle, `as_kind` étant l'unique `match`.
/// let k = volume.as_kind();
/// assert_eq!(k.primal_vars(), vec!["T".to_string()]);
/// assert!(k.as_domain().is_some());   // une physique de volume
/// assert!(k.as_constraint().is_none());
/// // Et réciproquement pour un appui : c'est un fait de **compilation**,
/// // non une erreur d'exécution.
/// assert!(appui.as_kind().as_domain().is_none());
/// assert!(appui.as_kind().as_constraint().is_some());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait SubModelKind: Sync {
    /// Primal variable names introduced by this physics (column labels).
    fn primal_vars(&self) -> Vec<String>;

    /// Dual variable names introduced by this physics (row labels).
    fn dual_vars(&self) -> Vec<String>;

    /// This physics' set of [`Physics`] natures — a per-variant **constant**
    /// slice, the classification counterpart of [`label`](Self::label). A plain
    /// physics returns a single-element slice; a coupled physics returns several.
    /// Required so every physics declares its nature(s) explicitly at its
    /// definition site. Matched by containment in the `filter` selectors.
    fn physics(&self) -> &'static [Physics];

    /// Borrow this sub-model as a [`Domain`] capability, or `None` (default) if
    /// it is not a domain physics (a constraint such as `Dirichlet`). A domain
    /// overrides this to return `Some(self)`. This is the seam the assembler,
    /// the material builders and [`crate::ops::element_field::behavior`] use — they never
    /// assume every sub-model reads material or integrates a behaviour.
    fn as_domain(&self) -> Option<&dyn Domain> {
        None
    }

    /// Borrow this sub-model as a [`Constraint`] capability, or `None`
    /// (default) if it imposes no Lagrange constraint (every plain volumetric
    /// physics). A constraint (`Dirichlet`, later MPC / strong contact)
    /// overrides this to return `Some(self)`. This is the seam the multiplier
    /// forwarders on [`SubModel`](crate::containers::model::SubModel) use — they
    /// never assume every sub-model carries multipliers.
    fn as_constraint(&self) -> Option<&dyn Constraint> {
        None
    }

    /// Dispatch the per-cell element kernel for `kind` — the single seam the
    /// global assembler drives (via a [`ComputedRecipe`](crate::containers::matrix::ComputedRecipe)),
    /// routing to [`element_matrix`](Domain::element_matrix) /
    /// [`element_mass`](Domain::element_mass) /
    /// [`element_geometric`](Domain::element_geometric) /
    /// [`element_tangent`](Domain::element_tangent). `state` is `Some(_)` only for
    /// the kinds that consume the current stress/tangent field (geometric,
    /// tangent); the others ignore it.
    #[allow(clippy::too_many_arguments)]
    fn matrix_element(
        &self,
        kind: MatrixKind,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        // The **one** bridge between the base contract and the element kernels,
        // which live on `Domain`: only a physics that produces a matrix has
        // them, and only such a physics declares a material FE subspace.
        let domain = self.as_domain().ok_or_else(|| {
            PyrucastError::Message(format!(
                "{}: no element kernel — this sub-model declares no Domain",
                self.label()
            ))
        })?;
        // The state exists for the two kinds that read it and for no other:
        // **that** is what the `Option` says, and it dies here rather than in
        // each kernel below.
        let with_state = |what: &str| -> Result<&SubElementField> {
            state.ok_or_else(|| {
                PyrucastError::Message(format!(
                    "{}: the {what} matrix reads a state field, and none was supplied",
                    self.label()
                ))
            })
        };
        match kind {
            MatrixKind::Stiffness => domain.element_matrix(geoms, material, lay, ke),
            MatrixKind::Mass => domain.element_mass(geoms, material, lay, ke),
            MatrixKind::Geometric => {
                domain.element_geometric(geoms, material, lay, with_state("geometric")?, ke)
            }
            MatrixKind::Tangent => {
                domain.element_tangent(geoms, material, lay, with_state("tangent")?, ke)
            }
        }
    }

    /// This sub-model's contributions of a given [`MatrixKind`], as the global
    /// assembler consumes them. **Default**: derived from
    /// [`matrix_layout`](Self::matrix_layout) — `Some(layout)` yields a single
    /// [`Contribution::Computed`] (a volumetric physics, integrated straight
    /// into the CSR); `None` falls back — **only for `Stiffness`** — to a
    /// [`Contribution::Literal`] built from
    /// [`build_stiffness_blocks`](Self::build_stiffness_blocks), and yields
    /// nothing for the other kinds (a physics with no term for that kind).
    ///
    /// A volumetric physics writes only its per-kind `element_*` + `*_layout`
    /// and takes the default. A sub-model whose blocks are *not* a single
    /// layout-driven integral (a constraint such as `Dirichlet`, which only
    /// contributes to `Stiffness`) overrides **this** method — that override is
    /// the extension seam, not a special case buried in the assembler.
    ///
    /// `material` is `Some(_)` iff [`Domain::material_fespace`] is declared
    /// (the assembler guarantees it); it is only consulted on the literal path —
    /// the computed path resolves material itself from the layout.
    fn contributions(
        &self,
        kind: MatrixKind,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        Ok(match self.matrix_layout(kind) {
            Some(layout) => vec![Contribution::Computed(layout)],
            // Here the `Option` earns its keep: a **constraint** sub-model has
            // no material at all, and that is what distinguishes it from a
            // domain. Below this line the question is settled.
            None if kind == MatrixKind::Stiffness => {
                let material = material.ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{}: a stiffness assembled cell by cell needs material data",
                        self.label()
                    ))
                })?;
                vec![Contribution::Literal(
                    self.build_stiffness_blocks(material)?,
                )]
            }
            None => Vec::new(),
        })
    }

    /// Build and fill the stiffness [`SubMatrix`] block(s) of this physics.
    /// `material` is `Some(_)` iff [`Domain::material_fespace`] is declared
    /// (the assembler guarantees it).
    ///
    /// **Default**: derived from [`stiffness_layout`](Self::stiffness_layout) —
    /// a single block on that layout, filled by
    /// [`element_matrix`](Domain::element_matrix) via [`kernel::assemble_block`].
    /// A plain volumetric physics therefore writes only `element_matrix` +
    /// `stiffness_layout` and gets this for free; this literal path serves as
    /// the bit-for-bit reference of the computed (scatter) path. A sub-model
    /// with **no** layout does not touch this method — it overrides
    /// [`contributions`](Self::contributions) instead (see `Dirichlet`).
    fn build_stiffness_blocks(&self, material: &Handle<SubElementField>) -> Result<Vec<SubMatrix>> {
        let Some(layout) = self.stiffness_layout() else {
            return Err(PyrucastError::Message(format!(
                "{}: build_stiffness_blocks has no default without a \
                 stiffness_layout — override it (e.g. a constraint such as \
                 Dirichlet, or a multi-block physics)",
                self.label()
            )));
        };
        let domain = self.as_domain().ok_or_else(|| {
            PyrucastError::Message(format!(
                "{}: no element kernel — this sub-model declares no Domain",
                self.label()
            ))
        })?;
        // Resolved once for the zone, before the parallel region: the closure
        // below captures the table, so no cell ever matches a component name.
        let lay = domain.element_layout(MatrixKind::Stiffness, &material.read(), None)?;
        let block = kernel::assemble_block(
            &layout.fespaces,
            &layout.support,
            &layout.support,
            layout.dual_vars,
            layout.primal_vars,
            layout.ordering,
            layout.symmetric,
            material,
            None,
            |geoms, m, _state, ke| domain.element_matrix(geoms, m, &lay, ke),
        )?;
        Ok(vec![block])
    }

    /// Structural layout of this physics' stiffness block, or `None` (default)
    /// for a physics assembled the literal way (constraints such as `Dirichlet`,
    /// or any multi-block physics). When `Some`, it drives **both** paths from a
    /// single description: the global assembler
    /// ([`crate::ops::matrix::stiffness`]) builds a *computed*
    /// [`SubMatrix`] and scatters [`element_matrix`](Domain::element_matrix)
    /// straight into the CSR (never materialising values), and the default
    /// [`build_stiffness_blocks`](Self::build_stiffness_blocks) produces the
    /// *literal* equivalent from the same layout + kernel.
    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        None
    }

    /// Structural layout of this physics' **mass / capacity** block, or `None`
    /// (default: no mass term). A continuum physics returns the same layout as
    /// its stiffness (same fespaces / support / vars) — only the kernel differs.
    fn mass_layout(&self) -> Option<MatrixLayout> {
        None
    }

    /// Structural layout of this physics' **geometric-stiffness** block, or
    /// `None` (default: none).
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        None
    }

    /// Structural layout of this physics' **consistent-tangent** block, or
    /// `None` (default: none).
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        None
    }

    /// Dispatch to the per-kind layout ([`stiffness_layout`](Self::stiffness_layout)
    /// / [`mass_layout`](Self::mass_layout) /
    /// [`geometric_layout`](Self::geometric_layout) /
    /// [`tangent_layout`](Self::tangent_layout)). This is the seam
    /// [`contributions`](Self::contributions) and the global assembler drive; a
    /// physics implements the per-kind primitives, not this dispatcher.
    fn matrix_layout(&self, kind: MatrixKind) -> Option<MatrixLayout> {
        match kind {
            MatrixKind::Stiffness => self.stiffness_layout(),
            MatrixKind::Mass => self.mass_layout(),
            MatrixKind::Geometric => self.geometric_layout(),
            MatrixKind::Tangent => self.tangent_layout(),
        }
    }

    /// Local internal-force vector of one cell — the pure, sequential kernel
    /// that applies `Bᵀ` to the stress (Cast3m `BSIG`). It is the **transpose**
    /// of this physics' deformation operator `B` (the same `B` behind its
    /// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation) /
    /// [`crate::ops::element_field::beam_deformation`](fn@crate::ops::element_field::beam_deformation)),
    /// so it mirrors [`Domain::integrate_point`]'s producer.
    ///
    /// Fills `fe` — the cell's local force vector, node-major / variable-minor
    /// (`fe[li * n_dual + di]`, `di` indexing [`dual_vars`](Self::dual_vars)) —
    /// from the cell geometry and the `stress` (the [`Domain::integrate_point`]
    /// output) borrowed in place. `geoms` holds one [`CellGeom`] per FE subspace
    /// of [`stiffness_layout`](Self::stiffness_layout), in that order.
    ///
    /// **Default**: the continuum-mechanics `f_{i,a} = Σ_g Σ_b (∂N_i/∂x_b) σ_ab`
    /// — one [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)
    /// per row of the symmetric stress tensor `σ`, read in Voigt naming
    /// (`sigma_xx`, `sigma_xy`, …). A displacement physics (elasticity, Mazars,
    /// plasticity) gets it for free; a physics whose dual is not a displacement
    /// vector (heat, bar, beam) overrides it.
    /// Components this internal-force kernel reads from the state field, in the
    /// slot order its indices assume — the counterpart of
    /// [`Domain::deformation_reads`] on the `Bᵀσ` side.
    ///
    /// Default: the continuum stress tensor, plus the hoop `σ_θθ` on a body of
    /// revolution, matching the default kernel below.
    fn internal_force_reads(&self) -> Vec<String> {
        let Some(layout) = self.stiffness_layout() else {
            return Vec::new();
        };
        let space_dim = layout.fespaces[0].read().space_dim();
        let mut names = continuum::internal_force::stress_matrix_reads(space_dim);
        if layout.fespaces[0].read().is_axisymmetric() {
            names.push("sigma_zz".to_string());
        }
        names
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        continuum::internal_force::continuum_internal_force_element(geoms, stress, lay, fe)
    }

    /// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of this physics (Cast3m `BSIG`),
    /// scattered to a [`SubNodeField`] on the block's node support. `stress` is
    /// this physics' [`Domain::integrate_behavior`] output.
    ///
    /// **Provided**: drives [`internal_force_element`](Self::internal_force_element)
    /// in parallel over the FE subspaces of
    /// [`stiffness_layout`](Self::stiffness_layout) (same geometry as the
    /// stiffness) and scatters to that layout's node support. A physics with no
    /// stiffness layout (a constraint such as `Dirichlet`) has no internal-force
    /// contribution and errors here. For a **linear** law the result equals
    /// `K·u` (the stiffness applied to the solution).
    fn build_internal_forces(&self, stress: &Handle<SubElementField>) -> Result<SubNodeField> {
        let Some(layout) = self.stiffness_layout() else {
            return Err(PyrucastError::Message(format!(
                "{}: build_internal_forces has no default without a stiffness_layout \
                 (e.g. a constraint such as Dirichlet)",
                self.label()
            )));
        };
        let stress_guard = stress.read();
        // Resolved once for the zone, before the parallel region.
        let names = self.internal_force_reads();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let lay = stress_guard.resolve_components(&refs, "stress")?;
        kernel::scatter_to_nodes(
            &layout.fespaces,
            &layout.support,
            layout.dual_vars,
            |geoms, fe| self.internal_force_element(geoms, &stress_guard, &lay, fe),
        )
    }

    /// Short type label, e.g. `"HeatConduction"` (used by `Debug` and the
    /// default `display`).
    fn label(&self) -> &'static str;

    /// One-line summary for `Display`. Default: `SubModel<{label}>`.
    fn display(&self) -> String {
        format!("SubModel<{}>", self.label())
    }

    /// Full multi-line rendering for [`crate::dump::Dump`].
    fn render(&self, opts: &DumpOptions) -> String;
}

/// A sub-model that imposes a **Lagrange constraint** — an optional capability,
/// not part of the base [`SubModelKind`] contract. A constraint implements it
/// *and* returns `Some(self)` from [`SubModelKind::as_constraint`]; a plain
/// volumetric physics implements neither, so it simply has no multipliers.
///
/// This is the seam of the constraint family — `Dirichlet` today, MPC and strong
/// contact later — and the natural home for the multiplier-driven logic they
/// share (e.g. the future condensation of the multiplier DOFs).
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // La capacité **contrainte** : des relations, chacune avec son nœud
/// // multiplicateur et la composante où son second membre s'écrit.
/// let c = appui.as_kind().as_constraint().unwrap();
/// let r = c.relations()?;
/// assert_eq!(r.len(), 1);
/// assert_eq!(r[0].terms.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait Constraint {
    /// POI1 mesh carrying this constraint's multiplier nodes. Borrowed from the
    /// sub-model (the user supplied it); generic code clones it when an owned
    /// [`Mesh`] is needed.
    fn multiplier_mesh(&self) -> &Mesh;

    /// The linear relations this constraint imposes, in a **method-neutral**
    /// form: one [`Relation`] per multiplier node, each carrying its terms
    /// `(node, variable, target_dual, coefficient)`. It is the single source of
    /// truth the imposition methods consume — the Lagrange path
    /// ([`SubModelKind::contributions`]) builds its `C` / `Cᵀ` blocks from the
    /// same relations a future master/slave *elimination* will read, so neither
    /// re-parses the user's mesh-per-term input.
    fn relations(&self) -> Result<Vec<Relation>>;
}

/// One term of a linear constraint relation, carried **method-neutrally**:
/// `coefficient · u(node, variable)`. Its reaction `coefficient · λ` lands in
/// `target_dual`, the dual (residual) row of `variable` in its physics.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Un terme de relation : un nœud, sa variable, et le dual de la
/// // physique visée — c'est ce dernier qui accroche la contrainte à la
/// // physique qu'elle contraint.
/// let r = appui.as_kind().as_constraint().unwrap().relations()?;
/// assert_eq!(r[0].terms[0].node, n[0].id());
/// assert_eq!(r[0].terms[0].variable, "T");
/// assert_eq!(r[0].terms[0].target_dual, "q");
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ConstraintTerm {
    /// Constrained node (a column of the target physics' stiffness).
    pub node: NodeId,
    /// Constrained primal variable (e.g. `"u_x"`).
    pub variable: String,
    /// Dual row of `variable` where the reaction is injected (e.g. `"f_x"`).
    pub target_dual: String,
    /// Scalar coefficient `aₖ`.
    pub coefficient: f64,
}

/// The sense of a constraint relation: an **equality** (the default, always
/// enforced) or a **unilateral inequality** solved by the active-set operator
/// [`solver::unilateral`](crate::ops::solver::unilateral).
///
/// The KKT conditions follow the sign convention of the existing Lagrange
/// blocks (`Cᵀ·λ` is *added* to the target's dual row):
///
/// - [`GreaterEqual`](Self::GreaterEqual): `C·u ≥ g`, `λ ≥ 0`, `λ·(C·u − g) = 0`;
/// - [`LessEqual`](Self::LessEqual): `C·u ≤ g`, `λ ≤ 0`, `λ·(C·u − g) = 0`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Égalité, ou inégalité — ce qui fait passer la contrainte du solveur
/// // direct à l'ensemble actif.
/// assert_eq!(RelationSense::parse(None)?, RelationSense::Equality);
/// assert_eq!(RelationSense::parse(Some(">="))?, RelationSense::GreaterEqual);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationSense {
    /// `Σₖ aₖ·uₖ = g` — always enforced (both solve paths).
    #[default]
    Equality,
    /// `Σₖ aₖ·uₖ ≥ g` — enforced only while active (reaction `λ ≥ 0`).
    GreaterEqual,
    /// `Σₖ aₖ·uₖ ≤ g` — enforced only while active (reaction `λ ≤ 0`).
    LessEqual,
}

impl RelationSense {
    /// Parse the user-facing spelling: `"="` (equality), `">="` (`≥`), `"<="`
    /// (`≤`). `None` defaults to equality. The single place the string form is
    /// interpreted (Rust convenience constructors and the Python bindings).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
    /// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
    /// #     None, None, RelationSense::Equality).unwrap();
    /// // L'**unique** endroit où la forme en chaîne est interprétée : les
    /// // constructeurs Rust de confort et les liaisons Python y passent tous.
    /// assert_eq!(RelationSense::parse(None)?, RelationSense::Equality);
    /// assert_eq!(RelationSense::parse(Some("<="))?, RelationSense::LessEqual);
    /// assert!(RelationSense::parse(Some("≠")).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn parse(sense: Option<&str>) -> Result<Self> {
        match sense {
            None | Some("=") | Some("==") => Ok(Self::Equality),
            Some(">=") | Some("≥") => Ok(Self::GreaterEqual),
            Some("<=") | Some("≤") => Ok(Self::LessEqual),
            Some(other) => Err(PyrucastError::Message(format!(
                "unknown relation sense '{other}' (expected '=', '>=' or '<=')"
            ))),
        }
    }
}

impl std::fmt::Display for RelationSense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Equality => "=",
            Self::GreaterEqual => ">=",
            Self::LessEqual => "<=",
        })
    }
}

/// One linear constraint relation `Σₖ coeffₖ · u(nodeₖ, varₖ) ⋈ g` (`⋈` is the
/// [`sense`](Self::sense) — `=` by default), enforced by a fresh
/// `multiplier_node` whose solved value is the reaction. The right-hand
/// side `g` is supplied by the user in the load field at the multiplier node's
/// imposed-value component, so it is **not** carried here.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Une relation : son multiplicateur, la composante où son `g` s'écrit,
/// // et ses termes. C'est la couture partagée par toutes les contraintes.
/// let r = appui.as_kind().as_constraint().unwrap().relations()?;
/// assert_eq!(r[0].imposed_value, "imposed_T");
/// assert_eq!(r[0].multiplier_node, mult.node(0, 0, 0)?.id());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Relation {
    /// The Lagrange-multiplier node that enforces this relation.
    pub multiplier_node: NodeId,
    /// The constraint's own dual (imposed-value) component this relation writes
    /// its `g` into, at `multiplier_node` (e.g. `imposed_T`, `mpc_rhs`,
    /// `imposed_u_y`). A single-dual constraint (`Dirichlet`, `Mpc`) repeats its
    /// one name; a multi-component constraint (`Embedded`) names the relation's
    /// component, so the RHS helper and the elimination path read the right slot.
    pub imposed_value: String,
    /// The terms summed on the left-hand side.
    pub terms: Vec<ConstraintTerm>,
    /// Equality (default) or unilateral inequality — read by the solvers only,
    /// the Lagrange blocks are assembled identically either way.
    pub sense: RelationSense,
}

/// Build the Lagrange `C` / `Cᵀ` block pair for **one** constraint term over one
/// paired submesh (multiplier ↔ constrained, one cell per relation, paired
/// element-for-element), every entry equal to `coefficient`:
///
/// - **`C`**  — rows `(multiplier, imposed_value)`, cols `(constrained, variable)`:
///   the constraint-equation row;
/// - **`Cᵀ`** — rows `(constrained, target_dual)`, cols `(multiplier, multiplier)`:
///   the reaction in the target's dual row.
///
/// Shared by [`dirichlet::Dirichlet`] (one term, `coefficient = 1`) and
/// [`mpc::Mpc`] (one call per term): the single place the saddle-point bordering
/// blocks are filled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn constraint_block_pair(
    multiplier_sm: &Handle<SubMesh>,
    constrained_sm: &Handle<SubMesh>,
    variable: &str,
    target_dual: &str,
    multiplier: &str,
    imposed_value: &str,
    coefficient: f64,
) -> Result<(SubMatrix, SubMatrix)> {
    let mult_nodes: Vec<NodeId> = multiplier_sm.read().connectivity().to_vec();
    let cons_nodes: Vec<NodeId> = constrained_sm.read().connectivity().to_vec();
    // C block: rows = multiplier × imposed_value, cols = constrained × variable.
    let mut c = SubMatrix::new(
        multiplier_sm.clone(),
        constrained_sm.clone(),
        vec![imposed_value.to_string()],
        vec![variable.to_string()],
        DofOrdering::NodesThenVars,
        false,
    )?;
    // Cᵀ block: rows = constrained × target_dual, cols = multiplier × multiplier.
    let mut ct = SubMatrix::new(
        constrained_sm.clone(),
        multiplier_sm.clone(),
        vec![target_dual.to_string()],
        vec![multiplier.to_string()],
        DofOrdering::NodesThenVars,
        false,
    )?;
    for (cons, mult) in cons_nodes.iter().zip(mult_nodes.iter()) {
        c.add_entry(*mult, imposed_value, *cons, variable, coefficient)?;
        ct.add_entry(*cons, target_dual, *mult, multiplier, coefficient)?;
    }
    Ok((c, ct))
}

/// A static component list in the owned form [`Domain::material_components`]
/// returns — the one line every physics whose contract is fixed needs.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// # use pyrucast::models::owned_components;
/// // Le raccourci que toute physique emploie pour rendre son contrat
/// // matériau sans le retaper.
/// assert_eq!(owned_components(&["E", "nu"]),
///            vec!["E".to_string(), "nu".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn owned_components(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// Where a physics' components sit in the rows its kernel is handed.
///
/// A constitutive kernel runs once per Gauss point — some tens of millions of
/// times in a non-linear solve — so it reads by **index**, never by name: a name
/// lookup is a string comparison that re-proves at every point what is a
/// property of the zone. This table is that proof, done once, before the
/// parallel region: it translates the canonical order a physics declares
/// ([`Domain::deformation_reads`], [`Domain::state_reads`],
/// [`Domain::material_components`]) into positions in the actual fields.
///
/// Resolving rather than assuming also makes the convention **checked**: a field
/// built by hand, in another order or missing a component, is refused by
/// [`Domain::zone_layout`] with a message naming the field and the gap — instead
/// of silently feeding a permuted tensor to the law.
/// ```
/// # use pyrucast::models::ZoneLayout;
/// // Une table dit où lire, pas quoi lire : la physique déclare son ordre
/// // canonique, la zone le traduit en positions.
/// let lay = ZoneLayout {
///     deformation: vec![2, 0, 1],
///     state: Vec::new(),
///     material: vec![0, 1],
///     optional_material: Vec::new(),
/// };
/// let ligne = [10.0, 20.0, 30.0];
/// // La première composante de la convention est en troisième position.
/// assert_eq!(ligne[lay.deformation[0] as usize], 30.0);
/// ```
pub struct ZoneLayout {
    /// Position of each [`Domain::deformation_reads`] component.
    pub deformation: Vec<u32>,
    /// Position of each [`Domain::state_reads`] component in the `prev` row.
    pub state: Vec<u32>,
    /// Position of each **required** material component, in
    /// [`Domain::material_components`] order.
    pub material: Vec<u32>,
    /// Position of each **optional** material component, in
    /// [`Domain::optional_material_components`] order;
    /// [`ABSENT_COMPONENT`](crate::containers::field::ABSENT_COMPONENT) where
    /// the caller supplied none.
    pub optional_material: Vec<u32>,
}

/// What a **matrix** kernel reads, resolved once per zone — the counterpart of
/// [`ZoneLayout`] on the element side.
///
/// The two paths of a physics now hold the same shape: the point kernel gets a
/// [`ZoneLayout`] from [`Domain::zone_layout`], the element kernel gets an
/// `ElementLayout` from [`Domain::element_layout`], and neither looks a
/// component up by name. A name search is a string comparison; done inside a
/// per-cell loop it re-proves at every cell what is a property of the *zone*.
///
/// ```
/// # use pyrucast::models::ElementLayout;
/// // Une table dit où lire. La physique déclare son ordre canonique
/// // (`material_components`, `element_state_reads`), la zone le traduit.
/// let lay = ElementLayout {
///     material: vec![1, 0],
///     optional_material: Vec::new(),
///     state: Vec::new(),
/// };
/// let ligne = [0.3, 210_000.0];
/// // `E` est déclaré en premier, mais rangé en second dans ce champ-là.
/// assert_eq!(ligne[lay.material[0] as usize], 210_000.0);
/// ```
pub struct ElementLayout {
    /// Position of each **required** material component, in
    /// [`Domain::material_components`] order.
    pub material: Vec<u32>,
    /// Position of each **optional** material component, in
    /// [`Domain::optional_material_components`] order;
    /// [`ABSENT_COMPONENT`](crate::containers::field::ABSENT_COMPONENT) where
    /// the caller supplied none.
    pub optional_material: Vec<u32>,
    /// Position of each [`Domain::element_state_reads`] component in the state
    /// row — empty for the kinds that read no state (stiffness, mass).
    pub state: Vec<u32>,
}

/// A **domain** sub-model — an optional capability, not part of the base
/// [`SubModelKind`] contract. A domain is a physics defined *over a region*: it
/// reads material data **and** integrates a constitutive law over its cells. A
/// domain implements this trait *and* returns `Some(self)` from
/// [`SubModelKind::as_domain`]; a constraint such as `Dirichlet` implements
/// neither, so its absence of material and behaviour is a compile-time fact, not
/// a runtime error.
///
/// Material and behaviour are **one** capability here, not two: the material
/// *parametrises* the constitutive law (`σ = D(E,ν):ε`, `M = E·I·κ`, …), so
/// every domain has both. That includes linear elements whose law is trivial (a
/// bar's `N = E·A·ε`, a beam's section forces) — the triviality is in the *kernel*
/// [`integrate_point`](Self::integrate_point), not in whether the capability
/// exists. This mirrors the *Domaine* row of the sub-model natures.
///
/// An implementer writes the material declaration
/// ([`material_fespace`](Self::material_fespace)), the behaviour kernel
/// ([`integrate_point`](Self::integrate_point) +
/// [`behavior_output_components`](Self::behavior_output_components)) and the FE
/// subspace [`behavior_fespace`](Self::behavior_fespace); the stiffness kernel
/// and the matrix kernels ([`element_matrix`](Self::element_matrix) & consorts) are
/// the parallel driver [`integrate_behavior`](Self::integrate_behavior) is
/// provided.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::{Model, SubModel};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Domain, MatrixKind, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # let volume = SubModel::heat_conduction(zone.clone()).unwrap();
/// # let appui = SubModel::dirichlet("T".into(), "q".into(), &impose, &mult,
/// #     None, None, RelationSense::Equality).unwrap();
/// // Matière et comportement sont **une seule** capacité : le matériau
/// // paramètre la loi de comportement, il n'a pas de sens sans elle.
/// let d = volume.as_kind().as_domain().unwrap();
/// assert_eq!(d.material_components(), vec!["k".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub trait Domain: Sync {
    /// FE subspace on which this domain expects its material data.
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace>;

    /// Material component names this domain requires, or `None` if it declares a
    /// material FE subspace but constrains no particular component. Default:
    /// `None`.
    ///
    /// **Owned**, not `&'static`: a physics whose variables are given by the
    /// caller derives its coefficient names from them — a transfer law wants one
    /// `h_<variable>` per transferred quantity, which no static table can hold.
    /// The DOF names ([`SubModelKind::primal_vars`]) had always been owned for
    /// the same reason; this closes the asymmetry. It is read once per sub-model
    /// when the material field is built, so the allocation is not on any hot
    /// path. Most implementers hand a static list to
    /// [`owned_components`](fn@owned_components) and are done.
    fn material_components(&self) -> Vec<String> {
        Vec::new()
    }

    /// Material component names this domain **accepts but does not require**:
    /// passed through the material channel if supplied (kept by
    /// [`material_field`](fn@crate::ops::element_field::material_field)), never demanded
    /// at assembly (only the *required* components discriminate the material
    /// zone). Read by an ancillary operator — e.g. `alpha` (thermal expansion)
    /// consumed by
    /// [`crate::ops::element_field::thermal_strain`](fn@crate::ops::element_field::thermal_strain).
    /// Default: `&[]`.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &[]
    }

    /// FE subspace this domain integrates its constitutive behaviour on. Its
    /// deformation input is produced geometrically by
    /// [`crate::ops::element_field::gradient`](fn@crate::ops::element_field::gradient) /
    /// [`crate::ops::element_field::deformation`](fn@crate::ops::element_field::deformation),
    /// and [`crate::ops::element_field::behavior`] uses this handle to pair the per-zone
    /// deformation field with its sub-model. Usually the same FE subspace as
    /// [`material_fespace`](Self::material_fespace).
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace>;

    /// Output component names of the material-state field produced by
    /// [`integrate_point`](Self::integrate_point) — the dual flux/stress
    /// followed by the updated internal state (`VAR1`), in order.
    fn behavior_output_components(&self) -> Vec<String>;

    /// Components this kernel reads from the **deformation** row, in the slot
    /// order its indices assume — the physics' own convention (`eps_xx`,
    /// `eps_yy`, `eps_xy` for a plane continuum; `grad_T_x`, … for conduction).
    ///
    /// Read once per zone by [`zone_layout`](Self::zone_layout), never at a
    /// Gauss point. Declaring it is physics — naming what the law consumes —
    /// and it is what lets the kernel index instead of search.
    fn deformation_reads(&self) -> Vec<String>;

    /// Components this kernel reads from the **previous state** row, in slot
    /// order. Default: none — most laws have no history.
    ///
    /// Every declared component must exist in the state: what a law reads back
    /// is what it wrote, so a mismatch is a real inconsistency, caught once per
    /// zone rather than turned into a silent zero at each point.
    fn state_reads(&self) -> Vec<String> {
        Vec::new()
    }

    /// Resolve this physics' conventions against the fields it will actually be
    /// handed — **the upstream half of index-based reading**, and the only place
    /// component names are matched at all.
    ///
    /// **Provided**, and rarely worth redefining. Required components must be
    /// present (a missing one names itself in the error); optional material
    /// components may be absent, marked
    /// [`ABSENT_COMPONENT`](crate::containers::field::ABSENT_COMPONENT).
    fn zone_layout(
        &self,
        deformation: &SubElementField,
        prev: &SubElementField,
        material: &SubElementField,
    ) -> Result<ZoneLayout> {
        let def_names = self.deformation_reads();
        let state_names = self.state_reads();
        let mat_names = self.material_components();
        let def_refs: Vec<&str> = def_names.iter().map(String::as_str).collect();
        let state_refs: Vec<&str> = state_names.iter().map(String::as_str).collect();
        let mat_refs: Vec<&str> = mat_names.iter().map(String::as_str).collect();
        Ok(ZoneLayout {
            deformation: deformation.resolve_components(&def_refs, "deformation")?,
            state: prev.resolve_components(&state_refs, "previous state")?,
            material: material.resolve_components(&mat_refs, "material")?,
            optional_material: material
                .resolve_optional_components(self.optional_material_components()),
        })
    }

    /// Components the **matrix** kernel of `kind` reads from the state field, in
    /// the slot order its indices assume — the counterpart of
    /// [`state_reads`](Self::state_reads) on the element side.
    ///
    /// Default: none. A stiffness or a mass reads only the material, so only the
    /// two state-consuming kinds — `Geometric` (the current stress) and
    /// `Tangent` (the algorithmic moduli) — declare anything here. The material
    /// side needs no declaration at all: [`material_components`](Self::material_components)
    /// already is it.
    fn element_state_reads(&self, _kind: MatrixKind) -> Vec<String> {
        Vec::new()
    }

    /// Resolve this physics' matrix conventions against the fields it will be
    /// handed — **the upstream half of index-based reading**, and the only place
    /// a matrix kernel's component names are matched at all.
    ///
    /// **Provided**, and rarely worth redefining; the mirror of
    /// [`zone_layout`](Self::zone_layout). The driver calls it once per zone,
    /// before the parallel region, and hands the result to every cell.
    fn element_layout(
        &self,
        kind: MatrixKind,
        material: &SubElementField,
        state: Option<&SubElementField>,
    ) -> Result<ElementLayout> {
        let mat_names = self.material_components();
        let mat_refs: Vec<&str> = mat_names.iter().map(String::as_str).collect();
        let state_names = self.element_state_reads(kind);
        let state_refs: Vec<&str> = state_names.iter().map(String::as_str).collect();
        let state_lay = match (state, state_refs.is_empty()) {
            (_, true) => Vec::new(),
            (Some(s), false) => s.resolve_components(&state_refs, "state")?,
            // The assembler supplies a state for exactly the kinds that read
            // one; a kernel declaring reads without one is a wiring mistake,
            // caught here rather than at a Gauss point.
            (None, false) => {
                return Err(PyrucastError::Message(format!(
                    "{kind:?}: this kernel reads {state_refs:?} from the state field, \
                     and none was supplied"
                )))
            }
        };
        Ok(ElementLayout {
            material: material.resolve_components(&mat_refs, "material")?,
            optional_material: material
                .resolve_optional_components(self.optional_material_components()),
            state: state_lay,
        })
    }

    /// Local element stiffness matrix of one cell — the pure, sequential kernel
    /// a physics author writes (the stiffness counterpart of
    /// [`integrate_point`](Self::integrate_point)). Fills `ke` (row-major,
    /// node-major / variable-minor: `ke[(li*n_dual+di) * n_cols_loc + (lj*n_primal+pj)]`)
    /// from the cell geometry and material.
    ///
    /// `geoms` holds one [`CellGeom`] per FE subspace declared in
    /// [`SubModelKind::stiffness_layout`], in that order: a plain volumetric
    /// physics reads `geoms[0]`, a multi-quadrature element (a shear-deformable
    /// beam, a shell) reads each — e.g. `geoms[0]` full Gauss for bending,
    /// `geoms[1]` reduced for shear.
    ///
    /// `lay` is [`element_layout`](Self::element_layout)'s answer, resolved once
    /// for the zone: read the material by `lay.material[k]`, **never by name** —
    /// a name search inside a per-cell loop re-proves at every cell what is a
    /// property of the zone.
    ///
    /// It **never sees rayon, the store, or a lock**: the assembler drives it in
    /// parallel over all cells. Required — every physics that produces a matrix
    /// has a stiffness.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()>;

    /// Local element **mass** matrix of one cell (`∫ ρ Nᵀ N` for mechanics,
    /// `∫ ρ c Nᵀ N` for the thermal capacity) — the mass counterpart of
    /// [`element_matrix`](Self::element_matrix). Default errors: a physics may
    /// legitimately have no mass term.
    fn element_mass(
        &self,
        _geoms: &[CellGeom],
        _material: &SubElementField,
        _lay: &ElementLayout,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(
            "no mass kernel — element_mass is undefined for this physics".into(),
        ))
    }

    /// Local element **geometric (initial-stress) stiffness** of one cell
    /// (`∫ Gᵀ σ̂ G`). `state` carries this physics' current stress field (the
    /// [`integrate_behavior`](Self::integrate_behavior) output), read by
    /// `lay.state[k]`. Default errors: not every physics buckles.
    fn element_geometric(
        &self,
        _geoms: &[CellGeom],
        _material: &SubElementField,
        _lay: &ElementLayout,
        _state: &SubElementField,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(
            "no geometric-stiffness kernel — element_geometric is undefined for this physics"
                .into(),
        ))
    }

    /// Local element **consistent tangent** of one cell (`∫ Bᵀ D_alg B`).
    /// `state` carries the algorithmic tangent moduli produced by
    /// [`integrate_point`](Self::integrate_point), read by `lay.state[k]`.
    /// Default errors: a linear law has no tangent of its own.
    fn element_tangent(
        &self,
        _geoms: &[CellGeom],
        _material: &SubElementField,
        _lay: &ElementLayout,
        _state: &SubElementField,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(
            "no tangent kernel — element_tangent is undefined for this physics".into(),
        ))
    }

    /// Local element **coupling** matrix of one facing cell pair — the kernel
    /// behind [`Contribution::Coupling`]. `row_geoms` describes the cell on the
    /// row mesh, `col_geoms` the facing cell on the column mesh; `ke` is
    /// `(row nodes × dual) × (col nodes × primal)`, same node-major layout as
    /// [`element_matrix`](Self::element_matrix).
    ///
    /// It is the physics' job to carry the **sign**: an exchange law contributes
    /// `+h∫NᵢNⱼ` on its two diagonal blocks and `−h∫NᵢNⱼ` off-diagonal, and since
    /// the two go through different kernels there is no factor to thread through
    /// the assembler. Default errors: only an interface couples two meshes.
    fn coupling_element(
        &self,
        _kind: MatrixKind,
        _row_geoms: &[CellGeom],
        _col_geoms: &[CellGeom],
        _material: &SubElementField,
        _lay: &ElementLayout,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(
            "no coupling kernel — coupling_element is undefined for this physics".into(),
        ))
    }

    /// The material state **at rest**, before any step — the `prev` of the first
    /// step. **Provided**: zero on every output component, which is the rest
    /// state of nearly every law (σ = 0, ε = 0, ε_p = 0, p = 0).
    ///
    /// It exists so that `prev` is **always** a real field. The alternative —
    /// an `Option` threaded down to the Gauss point — made every physics ask
    /// « is there a state yet? » some tens of millions of times per solve, to
    /// answer the same thing every time; and it is a question about the *step*,
    /// not about the point. The field has to be allocated for the step's output
    /// anyway, so materializing it costs one buffer that was already coming.
    ///
    /// Redefine it where the rest state is **not** the zero state: a law whose
    /// internal variable starts from a material constant (Gurson's initial
    /// porosity `f_0`) seeds it here, once, instead of testing for emptiness at
    /// every Gauss point.
    fn initial_state(&self, _material: &SubElementField) -> Result<SubElementField> {
        SubElementField::new(self.behavior_fespace(), self.behavior_output_components())
    }

    /// Whether this domain's law needs the **time increment** — a creep or
    /// viscoplastic law, whose answer depends on how long the step lasted.
    /// Default: `false`.
    ///
    /// Read once, by [`crate::ops::element_field::behavior::integrate`], which
    /// refuses the whole integration when such a law is handed no `dt`. The
    /// kernel then receives a plain `f64` and never asks: the question belongs
    /// to the step, and answering it per Gauss point answered it some tens of
    /// millions of times to say the same thing.
    fn requires_dt(&self) -> bool {
        false
    }

    /// Constitutive law at **one Gauss point** — the pure, sequential kernel a
    /// physics author writes. Integrates the step **A → B** for cell `geom.cell`
    /// at Gauss point `g`:
    ///
    /// Every input is the **row of this Gauss point** — a borrowed slice of the
    /// field's own buffer, never a copy — read through `lay`:
    ///
    /// - `deformation` is the **end-of-step** kinematics ε(B) (the strain, the
    ///   temperature gradient `∇T`, …) produced by a *geometric* operator, in
    ///   [`deformation_reads`](Self::deformation_reads) order;
    /// - `prev` is the **converged state at the start of the step A** — the
    ///   dual flux/stress σ(A), the internal variables `VAR(A)`, and (for laws
    ///   that form an increment) the start-of-step kinematics ε(A). It is
    ///   **always** a real row: on the first step it is
    ///   [`initial_state`](Self::initial_state), the rest state;
    /// - `material` is this cell's material data, empty for a physics that
    ///   declares none;
    /// - `dt` is the time increment, `0.0` for a rate-independent law — a law
    ///   that needs one is guaranteed to get it, because
    ///   [`crate::ops::element_field::behavior::integrate`] refuses the whole
    ///   integration otherwise.
    ///
    /// Write the [`behavior_output_components`](Self::behavior_output_components)
    /// values — the material state at B (σ(B), `VAR(B)`, and any echoed ε(B)) —
    /// into `out`. It **never sees rayon, the store, or a lock**:
    /// [`integrate_behavior`](Self::integrate_behavior) drives it in parallel
    /// over all cells.
    ///
    /// Two invariants hold here, and they are what makes this kernel fast: **no
    /// test that upstream already settled** (a component's presence, a field's
    /// shape, an absent increment) and **no dynamic allocation** — no `Vec`, no
    /// `String`, no `format!`, no intermediate struct. What is left is the
    /// physics, including its own branches.
    // A constitutive kernel legitimately needs its geometry, the Gauss index,
    // its layout, the A→B kinematics, material, time step and output slot.
    #[allow(clippy::too_many_arguments)]
    fn integrate_point(
        &self,
        geom: &CellGeom,
        g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        prev: &[f64],
        material: &[f64],
        dt: f64,
        out: &mut [f64],
    ) -> Result<()>;

    /// Integrate the constitutive law (Cast3m `COMP`). **Provided**: drives the
    /// point kernel [`integrate_point`](Self::integrate_point) in parallel over
    /// the behaviour FE subspace via [`kernel::element_pointwise`].
    ///
    /// This is the **incremental montage** A → B: `deformation` carries the
    /// end-of-step kinematics ε(B) alone, and `prev` — the *converged output of
    /// the previous step*, or [`initial_state`](Self::initial_state) on the
    /// first one — carries the whole state at A (σ(A), `VAR(A)`, ε(A)).
    /// Returns the **material-state** field
    /// at B: the dual flux/stress followed by the updated internal-state
    /// variables (`VAR1`), which becomes the next step's `prev`. Where
    /// [`SubModelKind::build_stiffness_blocks`] is the *linearization* of the
    /// law, this is its *exact* response: for a linear law the two agree
    /// (`∫ Bᵀ·flux = K·u`); a non-linear law departs from that tangent.
    ///
    /// The **layout is resolved here**, once, and `dt` captured by the point
    /// closure: below this line the kernel indexes and computes, and does
    /// nothing else.
    fn integrate_behavior(
        &self,
        deformation: &Handle<SubElementField>,
        prev: &Handle<SubElementField>,
        material: &Handle<SubElementField>,
        dt: f64,
    ) -> Result<SubElementField> {
        let fespace = self.behavior_fespace();
        let out_components = self.behavior_output_components();
        // One resolution per zone, against the fields actually handed over —
        // which is also where a hand-built field is refused.
        let lay = self.zone_layout(&deformation.read(), &prev.read(), &material.read())?;
        kernel::element_pointwise(
            &fespace,
            deformation,
            prev,
            material,
            out_components,
            |geom, g, def, prev, mat, out| {
                self.integrate_point(geom, g, &lay, def, prev, mat, dt, out)
            },
        )
    }
}

/// Declare a physics' **parent-level operator** and its Python twin, from a
/// single place in the physics' own file.
///
/// The author of a physics writes physics and documentation; the plumbing —
/// the sweep, the `#[pyfunction]`, the stub attribute, the unwrapping of the
/// Python wrappers — is emitted here. They never open `src/py/`.
///
/// Two doc blocks, because the two audiences differ: the Rust one carries the
/// `///` comment and its doctest, the Python one a string literal, which is
/// what lands in the `.pyi`. Sharing a single block would put a Rust doctest
/// into a Python docstring. Same shape as `py_field_unary!` in
/// `src/py/ops/field.rs`, the house precedent.
///
/// ```
/// # use pyrucast::ops::model;
/// # use pyrucast::named::Named;
/// // L'opérateur produit est une fonction libre ordinaire.
/// assert_eq!(pyrucast::atoms::ElementType::from_name("SEG2").is_some(), true);
/// ```
#[macro_export]
macro_rules! physics_operator {

    // ── Alias : une façade nommée par-dessus un opérateur générique ─────────
    //
    // Les lois d'écoulement et d'endommagement sont des **attributs** d'une
    // physique unique (cf. `models::plasticity::law`), pas des physiques de plus. Elles
    // se lisent pourtant mieux nommées au site d'appel — `drucker_prager(fes,
    // m)` plutôt que `plasticity_with_law(fes, m, PlasticLaw::DruckerPrager)`.
    // La façade ne duplique rien : elle transmet à l'opérateur générique.
    (
        $(#[$rust_doc:meta])*
        pub fn $name:ident(fes $(, $arg:ident : $ty:ty)* $(,)?) = $target:path, $fixed:expr;
        python: $py_doc:literal
    ) => {
        $(#[$rust_doc])*
        pub fn $name(
            fes: &$crate::containers::finite_element_space::FiniteElementSpace,
            $($arg: $ty,)*
        ) -> $crate::error::Result<$crate::containers::model::Model> {
            $target(fes, $($arg,)* $fixed)
        }

        #[cfg(feature = "python-api")]
        ::paste::paste! {
        /// The Python face of this physics — generated, never hand-written.
        pub mod [<$name _py>] {
            #[allow(unused_imports)]
            use super::*;

            #[doc = $py_doc]
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pyfunction)]
            #[::pyo3::pyfunction]
            pub fn $name(
                fespace: ::pyo3::PyRef<$crate::py::finite_element_space::PyFiniteElementSpace>,
                $($arg: $ty,)*
            ) -> ::pyo3::PyResult<$crate::py::model::PyModel> {
                Ok($crate::py::model::PyModel {
                    inner: super::$name(&fespace.inner $(, $arg)*)?,
                })
            }
        } }
    };
    (
        $(#[$rust_doc:meta])*
        pub fn $name:ident(fes $(, $arg:ident : $ty:ty)* $(,)?) via $sub:path;
        python: $py_doc:literal
    ) => {
        $(#[$rust_doc])*
        pub fn $name(
            fes: &$crate::containers::finite_element_space::FiniteElementSpace,
            $($arg: $ty,)*
        ) -> $crate::error::Result<$crate::containers::model::Model> {
            // Un argument non-`Copy` (`Vec<(String, String)>`) serait déplacé
            // à la première zone : la fermeture est `FnMut`, elle en voit
            // plusieurs. On clone donc, ce qui ne coûte rien sur les énumérés.
            #[allow(clippy::clone_on_copy)]
            $crate::ops::model::spanning(fes, |zone| $sub(zone $(, $arg.clone())*))
        }

        #[cfg(feature = "python-api")]
        ::paste::paste! {
        /// The Python face of this physics — generated, never hand-written.
        pub mod [<$name _py>] {
            // Les types des arguments sont écrits dans la portée du fichier de
            // la physique ; une physique sans argument n'en importe aucun.
            #[allow(unused_imports)]
            use super::*;

            #[doc = $py_doc]
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pyfunction)]
            #[::pyo3::pyfunction]
            pub fn $name(
                fespace: ::pyo3::PyRef<$crate::py::finite_element_space::PyFiniteElementSpace>,
                $($arg: $ty,)*
            ) -> ::pyo3::PyResult<$crate::py::model::PyModel> {
                Ok($crate::py::model::PyModel {
                    inner: super::$name(&fespace.inner $(, $arg)*)?,
                })
            }
        } }
    };
}
