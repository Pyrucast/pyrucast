//! The material row of one cell, read **by position** — the shape every
//! constitutive law of the continuum receives.
//!
//! One reader for the three families: a stateless law, a return-map law and a
//! direct-update law all get the same thing, because the question « where does
//! this law's third material constant sit in this field? » was settled once for
//! the zone, and has nothing to do with how the law integrates.
//!
//! Neither accessor returns a `Result` or an `Option`: the presence of a
//! required component was proved when the layout was resolved, and an absent
//! *optional* one yields the caller's default rather than a value to unwrap.

use crate::containers::field::ABSENT_COMPONENT;

/// The material of one cell, plus where each component of a law's contract sits
/// in it.
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
/// # use pyrucast::models::continuum::material::MatRead;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// // Un matériau rangé « à l'envers » du contrat de la loi.
/// let materiau = SubElementField::from_uniform_per_component(
///     fes.get(0)?, vec!["nu".into(), "E".into()], &[0.3, 210_000.0])?;
/// let idx = materiau.resolve_components(&["E", "nu"], "material")?;
/// let mat = MatRead::new(materiau.point_values(0, 0)?, &idx, &[]);
/// // La table absorbe l'écart : `E` reste le premier du contrat.
/// assert_eq!(mat.get(0), 210_000.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MatRead<'a> {
    /// The cell's material row.
    pub row: &'a [f64],
    /// Where each component of the law's contract sits in it, resolved once for
    /// the zone.
    pub idx: &'a [u32],
    /// Where each **optional** component sits, in
    /// [`Domain::optional_material_components`](crate::models::Domain::optional_material_components)
    /// order, [`ABSENT_COMPONENT`] where the caller supplied none. Empty for a
    /// law that declares no optional component — an empty slice, never an
    /// `Option`: there is nothing to unwrap about « this law has none ».
    pub opt_idx: &'a [u32],
}

impl<'a> MatRead<'a> {
    /// The material of one cell: its row, the positions of the law's own
    /// contract, and those of its optional components.
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
    /// # use pyrucast::models::continuum::material::MatRead;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let materiau = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?;
    /// let idx = materiau.resolve_components(&["E", "nu"], "material")?;
    /// // Une loi sans composante facultative passe une tranche **vide**,
    /// // jamais un `Option` : il n'y a rien à déballer.
    /// let mat = MatRead::new(materiau.point_values(0, 0)?, &idx, &[]);
    /// assert_eq!(mat.get(1), 0.3);
    /// assert!(mat.opt_idx.is_empty());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(row: &'a [f64], idx: &'a [u32], opt_idx: &'a [u32]) -> Self {
        Self { row, idx, opt_idx }
    }

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
    /// # use pyrucast::models::continuum::material::MatRead;
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
    /// let mat = MatRead::new(materiau.point_values(0, 0)?, &idx, &[]);
    /// assert_eq!(mat.get(2), 1e-4); // eps_d0, troisième du contrat
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn get(&self, k: usize) -> f64 {
        self.row[self.idx[k] as usize]
    }

    /// The `k`-th **optional** component, or `default` where the caller supplied
    /// none.
    ///
    /// Absence is a fact of the **zone**, settled when the layout was resolved:
    /// the branch here is on a resolved index, never on a name.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::{SubField, ABSENT_COMPONENT};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::material::MatRead;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let materiau = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["E".into(), "nu".into(), "rho".into()],
    ///     &[210_000.0, 0.3, 7.8e-9])?;
    /// let idx = materiau.resolve_components(&["E", "nu"], "material")?;
    /// // Le contrat optionnel est `["alpha", "rho"]` : `alpha` manque, `rho` non.
    /// let opt = materiau.resolve_optional_components(&["alpha", "rho"]);
    /// let mat = MatRead::new(materiau.point_values(0, 0)?, &idx, &opt);
    /// assert_eq!(opt[0], ABSENT_COMPONENT);
    /// assert_eq!(mat.optional(0, 1.2e-5), 1.2e-5); // alpha : le défaut
    /// assert_eq!(mat.optional(1, 0.0), 7.8e-9);    // rho : la valeur fournie
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn optional(&self, k: usize, default: f64) -> f64 {
        match self.opt_idx[k] {
            ABSENT_COMPONENT => default,
            i => self.row[i as usize],
        }
    }
}
