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
//! 4. expose it via `Model::<name>` (Rust) and a `#[classmethod]` (Python).
//!
//! Everything else is generic. See the book chapter *« Ajouter une
//! physique »* for the full walkthrough.

use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::SubNodeField;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::store::{read, Handle};
use serde::{Deserialize, Serialize};

pub mod bernoulli;
pub mod contact;
pub mod convection;
pub mod damage;
pub mod dirichlet;
pub mod elasticity;
pub mod embedded;
pub mod fick;
pub mod follower_pressure;
pub mod frame;
pub mod frame3d;
pub mod heat_conduction;
pub mod interface_transfer;
pub mod kernel;
pub mod mpc;
pub mod plastic;
pub mod plasticity;
pub mod radiation;
pub mod symmetry;
pub mod timoshenko;
pub mod truss;

pub use kernel::CellGeom;

/// Axis suffixes used by the continuum-mechanics internal-force kernel to read
/// Voigt-named stress components (`sigma_xx`, `sigma_xy`, …).
const VOIGT_AXES: [&str; 3] = ["x", "y", "z"];

/// The kind of element matrix a physics contributes — the discriminant that
/// makes the whole assembly pipeline (recipe → scatter → per-kind pattern cache)
/// matrix-agnostic. One `assemble_*` entry point per variant
/// ([`crate::ops::matrix`]) drives the **same** machinery with a different
/// per-element kernel; a physics that has no term for a given kind contributes
/// nothing (its [`matrix_layout`](SubModelKind::matrix_layout) returns `None`).
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
    pub const COUNT: usize = 4;

    /// Dense index in `0..COUNT`, for indexing per-kind caches.
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
pub struct MatrixLayout {
    /// FE subspaces the element kernel integrates over. **Give a `Vec`**: a
    /// single subspace for a plain volumetric physics, or several — sharing one
    /// submesh, differing only by quadrature — for a multi-quadrature element
    /// (a shear-deformable beam, a shell). The primary (index 0) drives the cell
    /// loop and the DOF numbering; [`element_matrix`](SubModelKind::element_matrix)
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
    /// Parse from a lowercase tag (`"mechanical"`, `"thermal"`, `"constraint"`,
    /// `"other"`, `"diffusion"`, `"radiation"`) — the Python-facing spelling,
    /// mirroring
    /// [`ElasticityModel::from_tag`](crate::models::elasticity::ElasticityModel::from_tag).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "mechanical" => Some(Self::Mechanical),
            "thermal" => Some(Self::Thermal),
            "constraint" => Some(Self::Constraint),
            "other" => Some(Self::Other),
            "diffusion" => Some(Self::Diffusion),
            "radiation" => Some(Self::Radiation),
            _ => None,
        }
    }

    /// The lowercase tag for this nature (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
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
    pub const ALL: [Physics; 6] = [
        Self::Mechanical,
        Self::Thermal,
        Self::Constraint,
        Self::Other,
        Self::Diffusion,
        Self::Radiation,
    ];

    /// The accepted tags, `|`-joined — for error messages.
    pub fn tag_list() -> String {
        Self::ALL
            .iter()
            .map(|p| p.to_tag())
            .collect::<Vec<_>>()
            .join("|")
    }
}

impl std::fmt::Display for Physics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
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

    /// Local element stiffness matrix of one cell — the pure, sequential kernel
    /// a physics author writes (the stiffness counterpart of
    /// [`Domain::integrate_point`]). Fills `ke` (row-major,
    /// node-major / variable-minor: `ke[(li*n_dual+di) * n_cols_loc + (lj*n_primal+pj)]`)
    /// from the cell geometry and material. `material` is `Some(_)` iff the
    /// physics declares a [`Domain::material_fespace`].
    ///
    /// `geoms` holds one [`CellGeom`] per FE subspace declared in
    /// [`stiffness_layout`](Self::stiffness_layout), in that order: a plain
    /// volumetric physics reads `geoms[0]`, a multi-quadrature element (a
    /// shear-deformable beam, a shell) reads each — e.g. `geoms[0]` full Gauss
    /// for bending, `geoms[1]` reduced for shear.
    ///
    /// It **never sees rayon, the store, or a lock**: the assembler drives it in
    /// parallel over all cells. Default errors (a physics with no element kernel,
    /// e.g. a constraint such as `Dirichlet`).
    fn element_matrix(
        &self,
        _geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no element kernel — element_matrix is undefined",
            self.label()
        )))
    }

    /// Local element **mass** matrix of one cell (`∫ ρ Nᵀ N` for mechanics,
    /// `∫ ρ c Nᵀ N` for the thermal capacity) — the mass counterpart of
    /// [`element_matrix`](Self::element_matrix). Default errors (a physics with
    /// no mass term). See [`matrix_element`](Self::matrix_element).
    fn element_mass(
        &self,
        _geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no mass kernel — element_mass is undefined",
            self.label()
        )))
    }

    /// Local element **geometric (initial-stress) stiffness** of one cell
    /// (`∫ Gᵀ σ̂ G`). `state` carries this physics' current stress field (the
    /// [`Domain::integrate_behavior`] output, Voigt-named). Default errors.
    fn element_geometric(
        &self,
        _geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _state: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no geometric-stiffness kernel — element_geometric is undefined",
            self.label()
        )))
    }

    /// Local element **consistent tangent** of one cell (`∫ Bᵀ D_alg B`).
    /// `state` carries the algorithmic tangent moduli produced by
    /// [`Domain::integrate_point`]. Default errors.
    fn element_tangent(
        &self,
        _geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _state: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no tangent kernel — element_tangent is undefined",
            self.label()
        )))
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
    /// the assembler. Default errors.
    fn coupling_element(
        &self,
        _kind: MatrixKind,
        _row_geoms: &[CellGeom],
        _col_geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no coupling kernel — coupling_element is undefined",
            self.label()
        )))
    }

    /// Dispatch the per-cell element kernel for `kind` — the single seam the
    /// global assembler drives (via a [`ComputedRecipe`](crate::containers::matrix::ComputedRecipe)),
    /// routing to [`element_matrix`](Self::element_matrix) /
    /// [`element_mass`](Self::element_mass) /
    /// [`element_geometric`](Self::element_geometric) /
    /// [`element_tangent`](Self::element_tangent). `state` is `Some(_)` only for
    /// the kinds that consume the current stress/tangent field (geometric,
    /// tangent); the others ignore it.
    fn matrix_element(
        &self,
        kind: MatrixKind,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        match kind {
            MatrixKind::Stiffness => self.element_matrix(geoms, material, ke),
            MatrixKind::Mass => self.element_mass(geoms, material, ke),
            MatrixKind::Geometric => self.element_geometric(geoms, material, state, ke),
            MatrixKind::Tangent => self.element_tangent(geoms, material, state, ke),
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
            None if kind == MatrixKind::Stiffness => {
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
    /// [`element_matrix`](Self::element_matrix) via [`kernel::assemble_block`].
    /// A plain volumetric physics therefore writes only `element_matrix` +
    /// `stiffness_layout` and gets this for free; this literal path serves as
    /// the bit-for-bit reference of the computed (scatter) path. A sub-model
    /// with **no** layout does not touch this method — it overrides
    /// [`contributions`](Self::contributions) instead (see `Dirichlet`).
    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let Some(layout) = self.stiffness_layout() else {
            return Err(PyrucastError::Message(format!(
                "{}: build_stiffness_blocks has no default without a \
                 stiffness_layout — override it (e.g. a constraint such as \
                 Dirichlet, or a multi-block physics)",
                self.label()
            )));
        };
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
            |geoms, m, _state, ke| self.element_matrix(geoms, m, ke),
        )?;
        Ok(vec![block])
    }

    /// Structural layout of this physics' stiffness block, or `None` (default)
    /// for a physics assembled the literal way (constraints such as `Dirichlet`,
    /// or any multi-block physics). When `Some`, it drives **both** paths from a
    /// single description: the global assembler
    /// ([`crate::ops::matrix::stiffness`]) builds a *computed*
    /// [`SubMatrix`] and scatters [`element_matrix`](Self::element_matrix)
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
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        continuum_internal_force_element(geoms, stress, fe)
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
        let stress_guard = read(stress)?;
        kernel::scatter_to_nodes(
            &layout.fespaces,
            &layout.support,
            layout.dual_vars,
            |geoms, fe| self.internal_force_element(geoms, &stress_guard, fe),
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
    let mult_nodes: Vec<NodeId> = read(multiplier_sm)?.connectivity().to_vec();
    let cons_nodes: Vec<NodeId> = read(constrained_sm)?.connectivity().to_vec();
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
/// [`element_matrix`](SubModelKind::element_matrix) stays on the base trait, and
/// the parallel driver [`integrate_behavior`](Self::integrate_behavior) is
/// provided.
pub trait Domain: Sync {
    /// FE subspace on which this domain expects its material data.
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace>;

    /// Material component names this domain requires, or `None` if it declares a
    /// material FE subspace but constrains no particular component. Default:
    /// `None`.
    fn material_components(&self) -> Option<&'static [&'static str]> {
        None
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
    fn behavior_output_components(&self) -> Result<Vec<String>>;

    /// Constitutive law at **one Gauss point** — the pure, sequential kernel a
    /// physics author writes. Integrates the step **A → B** for cell `geom.cell`
    /// at Gauss point `g`:
    ///
    /// - `deformation` is the **end-of-step** kinematics ε(B) (the strain, the
    ///   temperature gradient `∇T`, …) produced by a *geometric* operator;
    /// - `prev` is the **converged state at the start of the step A** — the
    ///   dual flux/stress σ(A), the internal variables `VAR(A)`, and (for laws
    ///   that form an increment) the start-of-step kinematics ε(A) — read by
    ///   component name. It is `None` on the first step, where A is the reference
    ///   configuration (σ(A) = 0, ε(A) = 0);
    /// - `material` is the per-zone material data, `Some(_)` iff the domain
    ///   declares a [`material_fespace`](Self::material_fespace);
    /// - `dt` is the time increment, `None` for a rate-independent law (a
    ///   rate/viscous law errors when it is `None`).
    ///
    /// Write the [`behavior_output_components`](Self::behavior_output_components)
    /// values — the material state at B (σ(B), `VAR(B)`, and any echoed ε(B)) —
    /// into `out`. It **never sees rayon, the store, or a lock**:
    /// [`integrate_behavior`](Self::integrate_behavior) drives it in parallel
    /// over all cells.
    // A constitutive kernel legitimately needs its geometry, the A→B kinematics
    // (`deformation`, `prev`), material, Gauss index, time step and output slot.
    #[allow(clippy::too_many_arguments)]
    fn integrate_point(
        &self,
        geom: &CellGeom,
        deformation: &SubElementField,
        prev: Option<&SubElementField>,
        material: Option<&SubElementField>,
        g: usize,
        dt: Option<f64>,
        out: &mut [f64],
    ) -> Result<()>;

    /// Integrate the constitutive law (Cast3m `COMP`). **Provided**: drives the
    /// point kernel [`integrate_point`](Self::integrate_point) in parallel over
    /// the behaviour FE subspace via [`kernel::element_pointwise`].
    ///
    /// This is the **incremental montage** A → B: `deformation` carries the
    /// end-of-step kinematics ε(B) alone, and `prev` — the *converged output of
    /// the previous step* — carries the whole state at A (σ(A), `VAR(A)`, ε(A)).
    /// `prev` is `None` on the first step. Returns the **material-state** field
    /// at B: the dual flux/stress followed by the updated internal-state
    /// variables (`VAR1`), which becomes the next step's `prev`. Where
    /// [`SubModelKind::build_stiffness_blocks`] is the *linearization* of the
    /// law, this is its *exact* response: for a linear law the two agree
    /// (`∫ Bᵀ·flux = K·u`); a non-linear law departs from that tangent.
    ///
    /// `prev`/`dt` are captured by the point closure (the `prev` guard is held
    /// for the whole parallel region), so [`kernel::element_pointwise`] stays a
    /// generic single-input driver.
    fn integrate_behavior(
        &self,
        deformation: &Handle<SubElementField>,
        prev: Option<&Handle<SubElementField>>,
        material: Option<&Handle<SubElementField>>,
        dt: Option<f64>,
    ) -> Result<SubElementField> {
        let fespace = self.behavior_fespace();
        let out_components = self.behavior_output_components()?;
        let prev_guard = prev.map(read).transpose()?;
        let prev_ref = prev_guard.as_deref();
        kernel::element_pointwise(
            &fespace,
            deformation,
            material,
            out_components,
            |geom, def, mat, g, out| self.integrate_point(geom, def, prev_ref, mat, g, dt, out),
        )
    }
}

/// Continuum-mechanics internal-force element kernel `f_{i,a} = Σ_g Σ_b
/// (∂N_i/∂x_b) σ_ab |J| w` — one [`crate::ops::node_field::divergence`](fn@crate::ops::node_field::divergence)
/// per row of the symmetric stress tensor `σ` (read in Voigt naming). Backs both
/// the [`SubModelKind::internal_force_element`] default (elasticity, Mazars,
/// plasticity) and the model-free
/// [`crate::ops::node_field::internal_forces_continuum`] operator. Fills
/// `fe` node-major / axis-minor (`fe[i * space_dim + a]`).
///
/// On an **axisymmetric** geometry the radial row gains the hoop term
/// `f_{i,r} += (N_i / r) σ_θθ` — the transpose of the `N_i / r` row the
/// strain-displacement matrix `B` carries there, so `∫ Bᵀσ` keeps matching `K·u`
/// for a linear law.
pub(crate) fn continuum_internal_force_element(
    geoms: &[CellGeom],
    stress: &SubElementField,
    fe: &mut [f64],
) -> Result<()> {
    let geom = &geoms[0];
    let d = geom.space_dim;
    let n_nodes = geom.n_nodes;
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?; // [i * d + b]
        let w = geom.det_j_w(g)?;
        let sig = voigt_stress_matrix(stress, geom.cell, g, d)?; // [a * d + b]
                                                                 // `sigma_zz` is the hoop stress and only exists on a body of revolution.
        let hoop = if geom.axisymmetric {
            Some((
                geom.n_at_g(g)?,
                stress.value(geom.cell, g, "sigma_zz")? / geom.radius(g)?,
            ))
        } else {
            None
        };
        for i in 0..n_nodes {
            for a in 0..d {
                let mut s = 0.0;
                for b in 0..d {
                    s += dn[i * d + b] * sig[a * d + b];
                }
                fe[i * d + a] += s * w;
            }
            if let Some((n, s_hoop)) = hoop {
                fe[i * d] += n[i] * s_hoop * w;
            }
        }
    }
    Ok(())
}

/// Read the symmetric `d×d` stress tensor at `(cell, g)` from a Voigt-named
/// stress field (`sigma_xx`, `sigma_yy`, `sigma_xy`, …), as a flat row-major
/// matrix `[a * d + b]`. Backs the continuum-mechanics
/// [`SubModelKind::internal_force_element`] default; reads by component name, so a
/// state field carrying extra `VAR1` components (Mazars) is handled transparently.
pub(crate) fn voigt_stress_matrix(
    stress: &SubElementField,
    cell: usize,
    g: usize,
    d: usize,
) -> Result<Vec<f64>> {
    let mut sig = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in i..d {
            let name = format!("sigma_{}{}", VOIGT_AXES[i], VOIGT_AXES[j]);
            let v = stress.value(cell, g, &name)?;
            sig[i * d + j] = v;
            sig[j * d + i] = v; // symmetric
        }
    }
    Ok(sig)
}
