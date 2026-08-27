//! Damage laws — identity, contract, and what they share.
//!
//! `<physique>/law.rs` : the enum an archive stores, the trait the physics
//! calls, and the data both need. The physics itself is one level up in
//! `damage.rs`; each law sits in its own file next door. Same shape as
//! `plasticity/law.rs`.

use crate::error::Result;
use serde::{Deserialize, Serialize};
/// Which damage law a [`Damage`](super::Damage) sub-model obeys.
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
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
/// # use pyrucast::models::elasticity;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
            Self::Mazars => &super::mazars::Mazars,
            Self::DamageTc => &super::damage_tc::DamageTc,
            Self::SicSic => &super::sic_sic::SicSic,
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
    /// # use pyrucast::models::damage::{self};
    /// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
    /// # use pyrucast::models::elasticity;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
    /// # use pyrucast::models::damage::{self};
    /// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
    /// # use pyrucast::models::elasticity;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
    /// # use pyrucast::models::damage::{self};
    /// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
    /// # use pyrucast::models::elasticity;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
    /// # use pyrucast::models::damage::{self};
    /// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
    /// # use pyrucast::models::elasticity;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
    /// # use pyrucast::models::damage::{self};
    /// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
    /// # use pyrucast::models::elasticity;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
    /// let (lambda, mu) = elasticity::lame(30_000.0, 0.2);
    /// assert!(u.sigma[0] < elasticity::elastic_stress(&grand, lambda, mu)[0]);
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
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
/// # use pyrucast::models::elasticity;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
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
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
/// # use pyrucast::models::elasticity;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
/// // La forme que veut chaque loi : la ligne matériau d'une maille, lue par
/// // position — la troisième composante du contrat est `eps_d0`.
/// assert_eq!(mat.get(2), 1e-4);
/// assert_eq!(mat.row.len(), 7);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MatRead<'a> {
    /// The cell's material row.
    pub row: &'a [f64],
    /// Where each component of the law's contract sits in it, resolved once for
    /// the zone.
    pub idx: &'a [u32],
}

impl MatRead<'_> {
    /// The `k`-th component of this law's material contract, for this cell.
    ///
    /// No name, no search, no `Result`: the component's presence and its
    /// position were settled when the zone layout was resolved.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::damage::law::MatRead;
    /// # use pyrucast::models::damage::mazars;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0)?, mazars::MATERIAL.iter().map(|s| s.to_string()).collect(),
    /// #     &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0])?;
    /// let idx = materiau.resolve_components(mazars::MATERIAL, "material")?;
    /// let mat = MatRead { row: materiau.point_values(0, 0)?, idx: &idx };
    /// assert_eq!(mat.get(2), 1e-4); // eps_d0, troisième du contrat
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn get(&self, k: usize) -> f64 {
        self.row[self.idx[k] as usize]
    }
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
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{self, DamageLaw, MatRead};
/// # use pyrucast::models::elasticity;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
/// // La partie positive, dont se servent les lois pour séparer traction
/// // et compression.
/// assert_eq!((law::pos(3.0), law::pos(-3.0)), (3.0, 0.0));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn pos(x: f64) -> f64 {
    x.max(0.0)
}
