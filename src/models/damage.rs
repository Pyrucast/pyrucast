//! Scalar and orthotropic **damage** — the physics, for any damage law.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`], and the **same
//! elastic stiffness** as iteration operator. The constitutive update is a
//! *secant* scalar-damage law: the stress is the elastic (effective) stress
//! scaled by `(1 − D)`, with `D ∈ [0, 1)` a scalar damage built from the
//! equivalent strain.
//!
//! Equivalent strain `ε̃ = √(Σ ⟨ε_I⟩₊²)` (positive parts of the principal
//! strains). Damage grows with the history variable `κ = maxₜ ε̃`, initialised
//! at the threshold `eps_d0`. Two damage branches `D_t` (tension) and `D_c`
//! (compression) are blended by weights `α_t`, `α_c` derived from the
//! tension/compression split of the effective stress:
//!
//! ```text
//! D_t = 1 − eps_d0(1−A_t)/κ − A_t / exp(B_t (κ − eps_d0))
//! D_c = 1 − eps_d0(1−A_c)/κ − A_c / exp(B_c (κ − eps_d0))
//! D   = α_t D_t + α_c D_c            (shear coefficient β fixed to 1)
//! σ   = (1 − D) · D_el : ε
//! ```
//!
//! Material components `E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`. The
//! single internal variable `kappa` comes in as the previous-step state `prev`
//! (`κ(A)`, floored at `eps_d0` in the update, so `None` on the first step is
//! fine) and out as the updated `VAR1`, alongside the scalar `damage`. The
//! effective stress is a function of the current total strain `ε(B)` — damage
//! mechanics has no strain increment; only `κ` is history.
//!
//! The equivalent strain is built from the **principal strains of the full 3-D
//! tensor**, so the 2-D models differ only in how that tensor is reconstructed:
//! plane strain forces `ε_zz = 0`, plane stress derives it, and **axisymmetric**
//! reads the measured hoop `ε_θθ = u_r/r`.
//!
//! As for plasticity, the Newton loop driving the load increments lives in
//! Python, not in Rust; this module provides the point-wise update only.

pub mod damage_tc;
pub mod mazars;
pub mod sic_sic;

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::owned_components;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Which damage law a [`Damage`] sub-model obeys.
///
/// The same attribute pattern as [`PlasticLaw`](crate::models::plasticity::law::PlasticLaw):
/// the DOFs, the elastic operator and the incremental montage are shared, and
/// only the law that turns a strain into a degraded stress differs.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // Même motif que `PlasticLaw` : les DDL, l'opérateur élastique et le
/// // montage incrémental sont partagés ; seule diffère la loi qui dégrade
/// // la contrainte.
/// assert_eq!(DamageLaw::ALL.len(), 3);
/// assert_eq!(DamageLaw::Mazars.internal_names(), vec!["kappa".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageLaw {
    /// Mazars — one scalar, two branches blended. Concrete.
    #[default]
    Mazars,
    /// Two damages, tension and compression apart — recovers the compressive
    /// stiffness when a crack closes.
    DamageTc,
    /// Orthotropic damage of a woven ceramic-matrix composite, one damage per
    /// weave direction.
    SicSic,
}

/// What a damage law has to say about itself.
///
/// The counterpart of `PlasticLawKind` on the
/// damage side, and the same division of labour as
/// [`SubModelKind`] one level up: the enum
/// [`DamageLaw`] carries the **identity** — what an archive stores — and the
/// trait carries the **behaviour**, so a single `match`
/// (`DamageLaw::as_law`) relates the two.
///
/// Adding a law: a unit struct and its `impl` in the law's own file, plus one
/// arm in `as_law`.
pub(crate) trait DamageLawKind: Sync {
    /// The material components the law reads. `space_dim` matters for a law
    /// whose orthotropy has a different count in plane and in space.
    fn material_components(&self, space_dim: usize) -> &'static [&'static str];

    /// Advance the law by one strain increment, for one Gauss point.
    fn update(
        &self,
        eps: &[f64; 6],
        prev: &[f64],
        mat: &MatRead,
        space_dim: usize,
    ) -> Result<DamageUpdate>;

    /// The law's internal variables, beyond the reported `damage`.
    fn internal_names(&self) -> Vec<String>;
}

impl DamageLaw {
    /// The behaviour behind this identity — **the only `match` per law**.
    pub(crate) fn as_law(self) -> &'static dyn DamageLawKind {
        match self {
            Self::Mazars => &mazars::Mazars,
            Self::DamageTc => &damage_tc::DamageTc,
            Self::SicSic => &sic_sic::SicSic,
        }
    }

    /// The lowercase name (the inverse of
    /// [`from_name`](crate::named::Named::from_name)).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// # use pyrucast::named::Named;
    /// // Réciproque exacte de `from_name`, pour les trois lois.
    /// assert!(DamageLaw::ALL.iter()
    ///     .all(|l| DamageLaw::from_name(l.name()) == Some(*l)));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Self::Mazars => "mazars",
            Self::DamageTc => "damage_tc",
            Self::SicSic => "sic_sic",
        }
    }

    /// Every law, in declaration order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// assert_eq!(DamageLaw::ALL, [DamageLaw::Mazars, DamageLaw::DamageTc, DamageLaw::SicSic]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub const ALL: [DamageLaw; 3] = [Self::Mazars, Self::DamageTc, Self::SicSic];

    /// The material components this law requires.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// // Mazars : un seuil et deux branches. SiC/SiC porte en plus les axes
    /// // du tissage, donc **plus de composantes en 3-D qu'en 2-D**.
    /// assert!(DamageLaw::Mazars.material_components(2).contains(&"eps_d0"));
    /// assert!(DamageLaw::SicSic.material_components(3).len()
    ///         > DamageLaw::SicSic.material_components(2).len());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn material_components(self, space_dim: usize) -> &'static [&'static str] {
        self.as_law().material_components(space_dim)
    }

    /// The law's internal variables, beyond the reported `damage`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// // L'état de la loi, au-delà du `damage` rapporté pour la visualisation.
    /// assert_eq!(DamageLaw::Mazars.internal_names(), vec!["kappa".to_string()]);
    /// // Damage-TC en porte quatre : deux seuils et deux endommagements.
    /// assert_eq!(DamageLaw::DamageTc.internal_names().len(), 4);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn internal_names(self) -> Vec<String> {
        self.as_law().internal_names()
    }

    /// One step of the law, at a Gauss point.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// // Sous le seuil `eps_d0`, rien ne s'endommage.
    /// let petit = [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let u = DamageLaw::Mazars.update(&petit, &[0.0], &mat, 2)?;
    /// assert_eq!(u.damage, 0.0);
    ///
    /// // Au-delà, l'endommagement croît et la contrainte est **dégradée** :
    /// // elle tombe sous la contrainte élastique correspondante.
    /// let grand = [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let u = DamageLaw::Mazars.update(&grand, &[0.0], &mat, 2)?;
    /// assert!(u.damage > 0.0 && u.damage < 1.0);
    /// let (lambda, mu) = damage::lame(30_000.0, 0.2);
    /// assert!(u.sigma[0] < damage::elastic_stress(&grand, lambda, mu)[0]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn update(
        self,
        eps: &[f64; 6],
        prev: &[f64],
        mat: &MatRead,
        space_dim: usize,
    ) -> Result<DamageUpdate> {
        self.as_law().update(eps, prev, mat, space_dim)
    }
}

impl crate::named::Named for DamageLaw {
    const LABEL: &'static str = "damage law";
    const VALUES: &'static [Self] = &Self::ALL;

    fn name(self) -> &'static str {
        DamageLaw::name(self)
    }
}

impl std::fmt::Display for DamageLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What a damage law returns for one Gauss point.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // `damage` est un **résumé** pour la visualisation ; l'état est `vars`.
/// // Une loi à plusieurs endommagements y rapporte le pire.
/// let u = DamageLaw::Mazars.update(&[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0], &mat, 2)?;
/// assert_eq!(u.vars.len(), DamageLaw::Mazars.internal_names().len());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct DamageUpdate {
    /// The degraded stress, full 3-D Voigt.
    pub sigma: [f64; 6],
    /// A scalar summary of the damage, for visualisation. The **state** is
    /// `vars`; a law with several damages reports the worst here.
    pub damage: f64,
    /// The law's internal variables, in [`DamageLaw::internal_names`] order.
    pub vars: Vec<f64>,
}

/// A cell's material, read by name — the same shape every law wants.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // La forme que veut chaque loi : le matériau d'une maille, lu par nom.
/// assert_eq!(mat.get("eps_d0")?, 1e-4);
/// assert_eq!(mat.cell, 0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MatRead<'a> {
    /// The material field, exposed so a law can reach the shared frame reader.
    pub field: &'a SubElementField,
    /// The cell this reads.
    pub cell: usize,
}

impl MatRead<'_> {
    /// A material component of this cell, by name.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let mat = MatRead { field: &materiau, cell: 0 };
    /// assert_eq!(mat.get("A_t")?, 0.8);
    /// // Une constante absente est une erreur, pas un zéro silencieux.
    /// assert!(mat.get("sigma_y").is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn get(&self, name: &str) -> Result<f64> {
        self.field.value(self.cell, 0, name)
    }
}

/// Lamé coefficients `(λ, μ)` from `E`, `nu` — shared by every law.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// let (lambda, mu) = damage::lame(30_000.0, 0.2);
/// assert!((mu - 30_000.0 / 2.4).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic (effective) stress, full 3-D Voigt, from a **tensor** strain.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // La contrainte **effective**, celle que la loi dégrade ensuite.
/// let (lambda, mu) = damage::lame(30_000.0, 0.2);
/// let s = damage::elastic_stress(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], lambda, mu);
/// assert!((s[5] - 2.0 * mu).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elastic_stress(eps: &[f64; 6], lambda: f64, mu: f64) -> [f64; 6] {
    let tr = eps[0] + eps[1] + eps[2];
    [
        lambda * tr + 2.0 * mu * eps[0],
        lambda * tr + 2.0 * mu * eps[1],
        lambda * tr + 2.0 * mu * eps[2],
        2.0 * mu * eps[3],
        2.0 * mu * eps[4],
        2.0 * mu * eps[5],
    ]
}

/// Positive part `⟨x⟩₊ = max(x, 0)`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // La partie positive, dont se servent les lois pour séparer traction
/// // et compression.
/// assert_eq!((damage::pos(3.0), damage::pos(-3.0)), (3.0, 0.0));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn pos(x: f64) -> f64 {
    x.max(0.0)
}

/// Where each **axisymmetric** Voigt slot `[rr, zz, θθ, rz]` sits in the full
/// 3-D order `[xx, yy, zz, yz, xz, xy]` — the damage law itself stays 3-D
/// (principal strains of the full tensor), only the projection changes.
const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order (matching [`crate::models::elasticity`]).
fn stress_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    if space_dim == 2 && model.is_axisymmetric() {
        // [rr, zz, θθ, rz] — the hoop is `zz`, Cast3M naming.
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_xy".into(),
        ]
    } else if space_dim == 2 {
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()]
    } else {
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ]
    }
}

/// Damage on an FE subspace. Same supports as
/// [`crate::models::elasticity::Elasticity`]; material is supplied at
/// assembly / integration time.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Interpolation, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # use pyrucast::models::damage::Damage;
/// # use pyrucast::models::elasticity::ElasticityModel;
/// // Mazars par défaut : un seuil et deux branches, traction et compression.
/// let d = Damage::new(zone.clone(), ElasticityModel::PlaneStress)?;
/// assert!(d.material_components().unwrap().contains(&"eps_d0".to_string()));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Damage {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
    pub(crate) law: DamageLaw,
}

impl Damage {
    /// **Mazars** damage on an FE subspace — the default law.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::damage::Damage;
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// // Mazars par défaut : un seuil et deux branches, traction et compression.
    /// let d = Damage::new(zone.clone(), ElasticityModel::PlaneStress)?;
    /// assert!(d.material_components().unwrap().contains(&"eps_d0".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        Self::with_law(fespace, model, DamageLaw::Mazars)
    }

    /// Damage with an explicit law, on an FE subspace with the given 2-D/3-D
    /// model. Errors if
    /// `model` is inconsistent with the space dimension (same rule as
    /// [`crate::models::elasticity::Elasticity::new`]).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Interpolation, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Domain, SubModelKind};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # use pyrucast::models::damage::{Damage, DamageLaw};
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// // La loi explicite. Damage-TC suit deux endommagements, donc réclame
    /// // deux résistances là où Mazars n'en demande qu'une.
    /// let tc = Damage::with_law(
    ///     zone.clone(), ElasticityModel::PlaneStress, DamageLaw::DamageTc)?;
    /// assert!(tc.material_components().unwrap().contains(&"f_t".to_string()));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn with_law(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
        law: DamageLaw,
    ) -> Result<Self> {
        let (submesh, space_dim, ref_dim, axisymmetric) = {
            let s = fespace.read();
            (
                s.submesh(),
                s.space_dim(),
                s.ref_dim()?,
                s.is_axisymmetric(),
            )
        };
        crate::models::elasticity::check_continuum_dimensions("Damage", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (2, ElasticityModel::Axisymmetric) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Damage: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ solid)"
            )));
        }
        // Same two-way agreement as `Elasticity::new`.
        if axisymmetric != model.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Damage: model {model:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` model"
                )
            } else {
                "Damage: the `axisymmetric` model requires an axisymmetric geometry \
                 (build the Coords with Coords::axisymmetric)"
                    .into()
            }));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
            law,
        })
    }
}

impl SubModelKind for Damage {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// The consistent mass matrix shares the stiffness layout (mass is
    /// law-independent).
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    /// The geometric stiffness shares the stiffness layout (initial-stress term
    /// is law-independent given the current stress).
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // Iteration operator = elastic (undamaged) stiffness. Reuse the
        // elasticity element kernel; it reads only `E` and `nu`.
        let mat = material.expect("Damage requires a material field");
        elasticity::element_stiffness(
            geom,
            mat,
            self.model,
            crate::models::symmetry::MaterialSymmetry::Isotropic,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Damage requires a material field");
        elasticity::element_mass(geom, mat, ke)
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("geometric stiffness requires the current stress field");
        elasticity::element_geometric(geom, stress, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Damage"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Damage({:?}, {})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model, self.law
        )
    }
}

impl Domain for Damage {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(
            self.law.material_components(self.space_dim),
        ))
    }

    /// `alpha` (thermal expansion) and `rho` (density) — the same pair
    /// [`elasticity`] accepts, and for the same
    /// reasons.
    ///
    /// `alpha` is read by an **ancillary** operator,
    /// [`thermal_strain`](fn@crate::ops::element_field::thermal_strain), which
    /// subtracts the expansion before the mechanical law sees anything: the
    /// return mapping never touches it. Leaving it out therefore excluded
    /// thermal expansion from plasticity and damage for no reason at all —
    /// `material_field` drops a component the physics does not declare, so the
    /// operator then found no zone carrying it.
    ///
    /// `rho` is required only by the mass matrix, never by the
    /// stiffness/behaviour assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha", "rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim, self.model);
        comps.push("damage".into());
        // The law's own history and per-direction damages.
        comps.extend(self.law.internal_names());
        Ok(comps)
    }

    /// One damage step at a Gauss point. Output layout = stress (Voigt, `v`) +
    /// the reported `damage` + the law's own internal variables.
    fn integrate_point(
        &self,
        geom: &CellGeom,
        deformation: &SubElementField,
        prev: Option<&SubElementField>,
        material: Option<&SubElementField>,
        g: usize,
        _dt: Option<f64>,
        out: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Damage declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let read = MatRead { field: mat, cell };
        // End-of-step strain ε(B); the law's history from `prev` (absent on the
        // first step, where every variable starts at zero).
        let eps = read_strain(deformation, cell, g, d, read.get("nu")?, self.model)?;
        // Left **empty** on the first step, where `prev` is `None`, so a law can
        // tell « no state yet » from « state that is zero ».
        let prev_vars: Vec<f64> = match prev {
            None => Vec::new(),
            Some(_) => self
                .law
                .internal_names()
                .iter()
                .map(|n| prev_opt(prev, cell, g, n))
                .collect(),
        };

        let update = self.law.update(&eps, &prev_vars, &read, d)?;
        let v = stress_names(d, self.model).len();
        for r in 0..v {
            out[r] = voigt_stress(&update.sigma, d, self.model, r);
        }
        out[v] = update.damage;
        for (i, value) in update.vars.iter().enumerate() {
            out[v + 1 + i] = *value;
        }
        Ok(())
    }
}

// The constitutive cores live in [`crate::models::damage`]'s submodules, one
// per law — shared helpers (Lamé, the elastic stress, the positive part) in the
// module root below. What remains here is the physics: the DOFs, the layouts,
// and the plumbing between the field components and the full-3-D strain the
// laws work in.

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// Read a component, returning `0.0` when absent (first step has no state).
fn read_opt(f: &SubElementField, cell: usize, g: usize, name: &str) -> f64 {
    if f.component_index(name).is_some() {
        f.value(cell, g, name).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Read a component from the optional previous-state field `prev`, defaulting to
/// `0.0` when there is no previous step (`None`) or the component is absent.
fn prev_opt(prev: Option<&SubElementField>, cell: usize, g: usize, name: &str) -> f64 {
    prev.map_or(0.0, |f| read_opt(f, cell, g, name))
}

/// Reconstruct the full 3-D tensor strain. Plane strain forces `ε_zz = 0`;
/// plane stress sets `ε_zz = -ν/(1-ν)(ε_xx+ε_yy)` (the elastic-damaged
/// out-of-plane strain, since the `(1-D)` factor cancels in `σ_zz = 0`).
fn read_strain(
    f: &SubElementField,
    cell: usize,
    g: usize,
    space_dim: usize,
    nu: f64,
    model: ElasticityModel,
) -> Result<[f64; 6]> {
    let mut eps = [0.0; 6];
    if space_dim == 2 {
        eps[0] = f.value(cell, g, "eps_xx")?;
        eps[1] = f.value(cell, g, "eps_yy")?;
        eps[5] = f.value(cell, g, "eps_xy")?;
        if model == ElasticityModel::PlaneStress {
            eps[2] = -nu / (1.0 - nu) * (eps[0] + eps[1]);
        } else if model.is_axisymmetric() {
            // The hoop ε_θθ = u_r/r is measured by `deformation`, not assumed.
            eps[2] = f.value(cell, g, "eps_zz")?;
        }
    } else {
        for (k, suf) in ["xx", "yy", "zz", "yz", "xz", "xy"].iter().enumerate() {
            eps[k] = f.value(cell, g, &format!("eps_{suf}"))?;
        }
    }
    Ok(eps)
}

/// Project the full 3-D stress to the model's Voigt slot `r`.
fn voigt_stress(sigma: &[f64; 6], space_dim: usize, model: ElasticityModel, r: usize) -> f64 {
    if space_dim == 2 && model.is_axisymmetric() {
        sigma[AXI_TO_3D[r]]
    } else if space_dim == 2 {
        match r {
            0 => sigma[0],
            1 => sigma[1],
            _ => sigma[5],
        }
    } else {
        sigma[r]
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn unit_quad(model: ElasticityModel) -> Damage {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Damage::new(fes.get(0).unwrap(), model).unwrap()
    }

    fn material(mz: &Damage) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            mz.fespace.clone(),
            mazars::MATERIAL.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        mat.set_uniform("E", 30_000.0).unwrap(); // ~ concrete (MPa)
        mat.set_uniform("nu", 0.2).unwrap();
        mat.set_uniform("eps_d0", 1e-4).unwrap();
        mat.set_uniform("A_t", 0.8).unwrap();
        mat.set_uniform("B_t", 20_000.0).unwrap();
        mat.set_uniform("A_c", 1.4).unwrap();
        mat.set_uniform("B_c", 1_900.0).unwrap();
        Handle::new(mat)
    }

    fn strain_field(mz: &Damage, eps_xx: f64) -> Handle<SubElementField> {
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        s.set_uniform("eps_xx", eps_xx).unwrap();
        Handle::new(s)
    }

    #[test]
    fn vars_and_material() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        assert_eq!(mz.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(mz.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(
            mz.material_components(),
            Some(owned_components(mazars::MATERIAL))
        );
    }

    /// Below the damage threshold the response is elastic: D = 0 and σ_xx is
    /// the linear plane-stress stress.
    #[test]
    fn undamaged_below_threshold() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        let eps0 = 1e-5; // < eps_d0 = 1e-4
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap().abs() < 1e-14);
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-6);
        }
    }

    /// Above the threshold in tension, damage develops (0 < D < 1) and the
    /// stress is reduced below the elastic prediction.
    #[test]
    fn damages_in_tension() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        let eps0 = 5e-4; // > eps_d0
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            let d = out.value(0, g, "damage").unwrap();
            assert!(d > 0.0 && d < 1.0, "D = {d}");
            // Damaged stress strictly below the elastic prediction.
            assert!(out.value(0, g, "sigma_xx").unwrap() < c * eps0);
            assert!(out.value(0, g, "kappa").unwrap() >= eps0 - 1e-12);
        }
    }

    /// History variable κ is monotone: unloading to a smaller strain does not
    /// reduce κ, and does not heal damage.
    #[test]
    fn kappa_is_monotone() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        // Load to 5e-4.
        let s1 = strain_field(&mz, 5e-4);
        let st1 = mz.integrate_behavior(&s1, None, Some(&mat), None).unwrap();
        let k1 = st1.value(0, 0, "kappa").unwrap();
        let d1 = st1.value(0, 0, "damage").unwrap();

        // Unload to 2e-4, feeding the step-1 state (κ) via `prev`.
        let prev = Handle::new(st1);
        let s2 = strain_field(&mz, 2e-4);
        let st2 = mz
            .integrate_behavior(&s2, Some(&prev), Some(&mat), None)
            .unwrap();
        assert!((st2.value(0, 0, "kappa").unwrap() - k1).abs() < 1e-12);
        // Damage unchanged on unloading (same κ).
        assert!((st2.value(0, 0, "damage").unwrap() - d1).abs() < 1e-9);
    }

    /// Solid 3-D uniaxial tension also triggers tensile damage.
    #[test]
    fn solid_3d_damages() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]).unwrap();
        let n = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
            p(1.0, 0.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 1.0),
        ];
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
        mesh.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
            .unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mz = Damage::new(fes.get(0).unwrap(), ElasticityModel::Solid).unwrap();
        let mat = material(&mz);
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            ["xx", "yy", "zz", "yz", "xz", "xy"]
                .iter()
                .map(|x| format!("eps_{x}"))
                .collect(),
        )
        .unwrap();
        s.set_uniform("eps_xx", 5e-4).unwrap();
        let s = Handle::new(s);
        let out = mz.integrate_behavior(&s, None, Some(&mat), None).unwrap();
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap() > 0.0);
        }
    }
}
