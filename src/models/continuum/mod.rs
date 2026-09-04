//! The **continuum in small strains** — the modelling three mechanical law
//! families stand on.
//!
//! A physics of the continuum is two independent things: *how the medium is
//! discretised* — the strain-displacement operator `B`, the Voigt nomenclature,
//! the kinematic hypothesis, the element kernels that integrate `Bᵀ · B` — and
//! *which constitutive law* runs at the Gauss point. This module is the first.
//! [`elasticity`](crate::models::elasticity),
//! [`plasticity`](crate::models::plasticity) and
//! [`damage`](crate::models::damage) each hold a [`Continuum`] and differ only
//! by their law.
//!
//! Before it existed, the modelling lived in `elasticity.rs` and the other two
//! reached into it — nine calls apiece. That worked, and said the wrong thing:
//! plasticity does not borrow from elasticity, they share a modelling.
//!
//! ## Axes
//!
//! Two hypotheses combine freely here and must not be confused:
//!
//! - the **kinematic** one ([`Kinematics`]) — plane stress, plane strain,
//!   axisymmetric, solid — which is a dimensional reduction, and
//! - the **material symmetry**
//!   ([`MaterialSymmetry`]) — which
//!   belongs to the law, and is passed to the kernels rather than stored here.

pub mod elastic;
pub mod internal_force;
pub mod material;
pub mod voigt;

use crate::containers::element_field::SubElementField;
use crate::containers::field::ABSENT_COMPONENT;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::SubMesh;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::symmetry::{self, MaterialSymmetry};
use crate::models::tensor::{stress_names, Kinematics};
use crate::models::{CellGeom, ElementLayout, MatrixKind};

use elastic::RHO_SLOT;
use voigt::{b_matrix_into, voigt_size, VoigtRows};

/// Reject an FE subspace whose elements are a **manifold** in their space
/// (`ref_dim < space_dim`) for a continuum-mechanics physics named `label`.
///
/// The continuum kernels build `B` from `∂N_i/∂x_a`, which on a manifold is the
/// *tangent* gradient: the resulting `Bᵀ D B` would be rank-deficient in the
/// normal direction and silently meaningless. A boundary sub-mesh (`SEG2` in
/// 2-D, `TRI3` in 3-D) is a support for loads
/// ([`flux`](crate::models::flux)) or convection, not a solid — and a
/// structural element (bar, beam) is a different physics with its own kernel.
fn check_continuum_dimensions(label: &str, space_dim: usize, ref_dim: usize) -> Result<()> {
    if ref_dim != space_dim {
        return Err(PyrucastError::Message(format!(
            "{label}: a {ref_dim}-D element in a {space_dim}-D space is a manifold, not a \
             solid — a boundary mesh carries loads (flux, convection), and a bar or beam \
             is a structural physics of its own (truss, frame, timoshenko)"
        )));
    }
    Ok(())
}

/// The continuum modelling of one FE subspace: its geometry, its support, and
/// the kinematic hypothesis under which strains are measured.
///
/// Held by every continuum physics — what they add is a law, not a geometry.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::continuum::Continuum;
/// # use pyrucast::models::tensor::Kinematics;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// // La modélisation, indépendamment de toute loi : c'est elle qui nomme
/// // les déformations que le noyau lira.
/// let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
/// assert_eq!(c.strain_reads(), ["eps_xx", "eps_yy", "eps_xy"]);
/// // Et c'est elle qui refuse une cinématique impossible dans cet espace.
/// assert!(Continuum::new(fes.get(0)?, Kinematics::Full3D, "Elasticity").is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Continuum {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) kinematics: Kinematics,
}

impl Continuum {
    /// The continuum modelling of `fespace` under `kinematics`, validated.
    ///
    /// `label` names the physics in the error messages — the three checks it
    /// runs used to be copied verbatim into every continuum constructor, and
    /// only the label differed:
    ///
    /// 1. the elements are solids, not a manifold in their space;
    /// 2. the kinematics is possible in that space dimension;
    /// 3. the kinematics and the geometry agree **both ways** about axisymmetry
    ///    — the `2πr` measure comes from the `Coords` while the hoop row comes
    ///    from the kinematics, so a mismatch would silently pair a plane
    ///    constitutive law with a revolved measure.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// // Le `label` n'est là que pour se nommer quand une des trois
    /// // cohérences est violée — ici la cinématique et l'espace.
    /// match Continuum::new(fes.get(0)?, Kinematics::Full3D, "Elasticity") {
    ///     Err(e) => assert!(format!("{e}").starts_with("Elasticity:")),
    ///     Ok(_) => panic!("une cinématique 3-D dans un espace 2-D est refusée"),
    /// }
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        fespace: Handle<SubFiniteElementSpace>,
        kinematics: Kinematics,
        label: &str,
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
        check_continuum_dimensions(label, space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, kinematics) {
            (2, Kinematics::PlaneStress | Kinematics::PlaneStrain) => true,
            (2, Kinematics::Axisymmetric) => true,
            (3, Kinematics::Full3D) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "{label}: kinematics {kinematics:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ full_3d)"
            )));
        }
        if axisymmetric != kinematics.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "{label}: kinematics {kinematics:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` kinematics (its integrals already \
                     carry the 2πr factor)"
                )
            } else {
                format!(
                    "{label}: the `axisymmetric` kinematics requires an axisymmetric geometry \
                     (build the Coords with Coords::axisymmetric)"
                )
            }));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            kinematics,
        })
    }

    /// The FE subspace this modelling stands on.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// // Le sous-espace rendu est celui qu'on lui a donné : une poignée,
    /// // clonée, non un nouveau sous-espace.
    /// assert_eq!(c.fespace().read().cell_count(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// POI1 support over the subspace's unique nodes.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// // Le support POI1 porte les nœuds uniques du sous-espace.
    /// assert_eq!(c.support().read().cell_count(), 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn support(&self) -> Handle<SubMesh> {
        self.support.clone()
    }

    /// Dimension of the ambient space.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// assert_eq!(c.space_dim(), 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn space_dim(&self) -> usize {
        self.space_dim
    }

    /// The kinematic hypothesis strains are measured under.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// assert_eq!(c.kinematics(), Kinematics::PlaneStress);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn kinematics(&self) -> Kinematics {
        self.kinematics
    }

    /// The strain components a law of this modelling reads, in Voigt order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// // Trois composantes en plan ; l'axisymétrie en ajoute le cerceau.
    /// assert_eq!(c.strain_reads(), ["eps_xx", "eps_yy", "eps_xy"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn strain_reads(&self) -> Vec<String> {
        voigt::strain_reads(self.space_dim, self.kinematics)
    }

    /// The stress components a law of this modelling writes, in Voigt order.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// assert_eq!(c.stress_names(), ["sigma_xx", "sigma_yy", "sigma_xy"]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn stress_names(&self) -> Vec<String> {
        stress_names(self.space_dim, self.kinematics)
    }

    /// What a continuum-mechanics **matrix** kernel reads from the state field,
    /// by [`MatrixKind`] — the shared declaration of elasticity, plasticity and
    /// damage, which run the very same kernels.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::MatrixKind;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let c = Continuum::new(fes.get(0)?, Kinematics::PlaneStress, "Elasticity")?;
    /// // Une raideur ne lit aucun état ; la raideur géométrique lit la contrainte.
    /// assert!(c.element_state_reads(MatrixKind::Stiffness).is_empty());
    /// assert_eq!(c.element_state_reads(MatrixKind::Geometric),
    ///            ["sigma_xx", "sigma_xy", "sigma_yy"]);
    /// // La tangente non plus : elle est évaluée au point, pas relue.
    /// assert!(c.element_state_reads(MatrixKind::Tangent).is_empty());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> {
        match kind {
            // The current Cauchy stress, Voigt-named, closed by the hoop `σ_θθ`
            // on a body of revolution — the same list `Bᵀσ` reads.
            MatrixKind::Geometric => {
                let mut names = internal_force::stress_matrix_reads(self.space_dim);
                if self.kinematics.is_axisymmetric() {
                    names.push("sigma_zz".to_string());
                }
                names
            }
            // A stiffness, a mass and — depuis que la tangente est évaluée au
            // point plutôt que relue d'un champ — une tangente ne lisent que le
            // matériau.
            MatrixKind::Stiffness | MatrixKind::Mass | MatrixKind::Tangent => Vec::new(),
        }
    }

    /// Element kernel: local stiffness `K_e = Σ_g (Bᵀ D B) |J| w` of one cell,
    /// written into `ke` (flat row-major, side `space_dim·n_nodes`, **node-major
    /// / component-minor** dof order `dof = node·space_dim + component`). Pure
    /// and sequential — driven in parallel by
    /// [`crate::models::kernel::assemble_block`]. Law-independent: the three
    /// families call it, their iteration operator being the elastic stiffness.
    ///
    /// ```
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::kernel::assemble_block;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
    /// #                vec!["u_x".to_string(), "u_y".to_string()]);
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?);
    /// # use pyrucast::models::ElementLayout;
    /// let c = Continuum::new(zone.clone(), Kinematics::PlaneStress, "Elasticity")?;
    /// // Le noyau d'élément, tel que le pilote l'appelle. La matrice de
    /// // raideur d'un solide libre est **singulière** : ses lignes somment à
    /// // zéro, un mouvement de corps rigide n'engendrant aucune force.
    /// //
    /// // `E` puis `nu` : le champ est rangé dans l'ordre du contrat, donc la
    /// // table est l'identité. Un vrai assemblage la ferait résoudre par
    /// // `Domain::element_layout`, qui accepte n'importe quel ordre.
    /// let lay = ElementLayout { material: vec![0, 1], optional_material: vec![], state: vec![] };
    /// let (duals, primals) = vars();
    /// let bloc = assemble_block(
    ///     std::slice::from_ref(&zone), &support, &support, duals, primals,
    ///     DofOrdering::NodesThenVars, true, &mat, None,
    ///     |geoms, m, _s, ke| c.element_stiffness(
    ///         &geoms[0], m, &lay, MaterialSymmetry::Isotropic, ke),
    /// )?;
    /// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
    /// assert!(total.abs() < 1e-6);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_stiffness(
        &self,
        geom: &CellGeom,
        material: &SubElementField,
        lay: &ElementLayout,
        symmetry: MaterialSymmetry,
        ke: &mut [f64],
    ) -> Result<()> {
        let n_nodes = geom.n_nodes;
        let space_dim = geom.space_dim;
        let dofs = space_dim * n_nodes;
        // Constants read at Gauss 0 — constant material per cell — **by index**:
        // the zone resolved the names once, so this matches none and allocates
        // nothing.
        let mut d = [[0.0_f64; 6]; 6];
        let v = symmetry::elastic_constitutive_into(
            material.row(geom.cell, 0),
            &lay.material,
            symmetry,
            self.kinematics,
            space_dim,
            &mut d,
        )?;
        // Les trois tampons vivent **hors** de la boucle : un point de Gauss ne
        // doit rien allouer, et `B` comme `D·B` sont de taille fixe une fois la
        // maille connue.
        let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
        let mut b: VoigtRows = [[0.0; MAX_CELL_DOFS]; 6];
        let mut db: VoigtRows = [[0.0; MAX_CELL_DOFS]; 6];
        for g in 0..geom.n_gauss {
            // On a body of revolution the hoop row needs `N` and `r` at this point.
            let hoop = if self.kinematics.is_axisymmetric() {
                Some((geom.n_at_g(g), geom.radius(g)))
            } else {
                None
            };
            geom.dn_dx(g, &mut dn_buf[..n_nodes * space_dim])?;
            b_matrix_into(
                &dn_buf[..n_nodes * space_dim],
                n_nodes,
                space_dim,
                hoop,
                &mut b,
            );
            // DB = D·B  (voigt × dofs).
            for r in 0..v {
                for c in 0..dofs {
                    let mut acc = 0.0;
                    for w in 0..v {
                        acc += d[r][w] * b[w][c];
                    }
                    db[r][c] = acc;
                }
            }
            let w = geom.det_j_w(g);
            for r in 0..dofs {
                for c in 0..dofs {
                    let mut acc = 0.0;
                    for vv in 0..v {
                        acc += b[vv][r] * db[vv][c];
                    }
                    ke[r * dofs + c] += acc * w;
                }
            }
        }
        Ok(())
    }

    /// Element kernel: local **consistent mass** `M_e = Σ_g ρ (Nᵀ N) |J| w` of
    /// one cell, written into `ke` (same flat row-major, **node-major /
    /// component-minor** dof order as [`element_stiffness`](Self::element_stiffness)).
    /// The vector shape-function matrix is block-diagonal, so
    /// `M[(i,a),(j,b)] = δ_ab ρ ∫ N_i N_j`. Density `ρ` is read from the optional
    /// material component `rho` (constant per cell). Pure, sequential and
    /// law-independent.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::kernel::assemble_block;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
    /// #                vec!["u_x".to_string(), "u_y".to_string()]);
    /// # let mat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["rho".into()], &[2.0])?);
    /// # use pyrucast::models::ElementLayout;
    /// # use pyrucast::containers::field::ABSENT_COMPONENT;
    /// # let c = Continuum::new(zone.clone(), Kinematics::PlaneStress, "Elasticity")?;
    /// // La masse ne lit que `rho`, deuxième composante **facultative** du
    /// // contrat du continuum (`["alpha", "rho"]`) : `alpha` est absente ici.
    /// let lay = ElementLayout {
    ///     material: vec![],
    ///     optional_material: vec![ABSENT_COMPONENT, 0],
    ///     state: vec![],
    /// };
    /// // La masse totale se retrouve dans la somme des entrées, une fois par
    /// // direction : ρ × aire × space_dim.
    /// let (duals, primals) = vars();
    /// let bloc = assemble_block(
    ///     std::slice::from_ref(&zone), &support, &support, duals, primals,
    ///     DofOrdering::NodesThenVars, true, &mat, None,
    ///     |geoms, m, _s, ke| c.element_mass(&geoms[0], m, &lay, ke),
    /// )?;
    /// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
    /// assert!((total - 2.0 * 0.5 * 2.0).abs() < 1e-9);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_mass(
        &self,
        geom: &CellGeom,
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let n_nodes = geom.n_nodes;
        let space_dim = geom.space_dim;
        let dofs = space_dim * n_nodes;
        // `rho` is the second optional component of the continuum contract
        // (`["alpha", "rho"]`): absent, there is no mass to integrate.
        let i_rho = lay.optional_material[RHO_SLOT];
        if i_rho == ABSENT_COMPONENT {
            return Err(PyrucastError::Message(
                "Continuum mass matrix: material component `rho` (density) is required".into(),
            ));
        }
        let rho = material.row(geom.cell, 0)[i_rho as usize];
        for g in 0..geom.n_gauss {
            let n = geom.n_at_g(g);
            let w = geom.det_j_w(g) * rho;
            for i in 0..n_nodes {
                for j in 0..n_nodes {
                    let m = n[i] * n[j] * w;
                    for a in 0..space_dim {
                        let r = i * space_dim + a;
                        let c = j * space_dim + a;
                        ke[r * dofs + c] += m;
                    }
                }
            }
        }
        Ok(())
    }

    /// Element kernel: local **geometric (initial-stress) stiffness**
    ///   `Kg[(i,a),(j,b)] = δ_ab Σ_g Σ_cd (∂N_i/∂x_c) σ_cd (∂N_j/∂x_e) |J| w`
    /// of one cell, written into `ke` (same flat, node-major / component-minor
    /// dof order as [`element_stiffness`](Self::element_stiffness)). The scalar
    /// `∇N_i·σ·∇N_j` is applied to each displacement component's diagonal block
    /// (`δ_ab`). The current Cauchy stress `σ` (Voigt-named) is read from
    /// `state` per Gauss point. Pure, sequential and law-independent.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::matrix::DofOrdering;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::Continuum;
    /// # use pyrucast::models::kernel::assemble_block;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let support = zone.read().submesh().read().to_poi1().unwrap();
    /// # let vars = || (vec!["f_x".to_string(), "f_y".to_string()],
    /// #                vec!["u_x".to_string(), "u_y".to_string()]);
    /// # let etat = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(),
    /// #     vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
    /// #     &[100.0, 0.0, 0.0])?);
    /// # // La raideur géométrique ne lit aucun matériau ; l'assembleur en veut
    /// # // un, on lui en donne un qui ne sert à rien plutôt qu'une Option.
    /// # let bidon = Handle::new(SubElementField::from_uniform_per_component(
    /// #     zone.clone(), vec!["E".into(), "nu".into()], &[210_000.0, 0.3])?);
    /// # use pyrucast::models::ElementLayout;
    /// # let c = Continuum::new(zone.clone(), Kinematics::PlaneStress, "Elasticity")?;
    /// // La convention lit `[σ_xx, σ_xy, σ_yy]` ; le champ ci-dessus est rangé
    /// // `[σ_xx, σ_yy, σ_xy]`. C'est tout ce que la table absorbe.
    /// let lay = ElementLayout {
    ///     material: vec![],
    ///     optional_material: vec![],
    ///     state: vec![0, 2, 1],
    /// };
    /// // La raideur **géométrique**, celle du flambement : elle vient de l'état
    /// // de contrainte, non du matériau. Sous traction elle est définie
    /// // positive ; c'est son signe qui décide de la charge critique.
    /// let (duals, primals) = vars();
    /// let bloc = assemble_block(
    ///     std::slice::from_ref(&zone), &support, &support, duals, primals,
    ///     DofOrdering::NodesThenVars, true, &bidon, Some(&etat),
    ///     |geoms, _m, s, ke| c.element_geometric(&geoms[0], s.unwrap(), &lay, ke),
    /// )?;
    /// // Elle est singulière elle aussi : les modes rigides n'y coûtent rien.
    /// let total: f64 = bloc.iter_entries().into_iter().map(|(_, _, _, _, v)| v).sum();
    /// assert!(total.abs() < 1e-9);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_geometric(
        &self,
        geom: &CellGeom,
        stress: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let n_nodes = geom.n_nodes;
        let d = geom.space_dim;
        let dofs = d * n_nodes;
        // `lay.state` is the Voigt stress in `stress_matrix_reads` order, closed
        // by the hoop `σ_θθ` on a body of revolution — resolved once for the zone.
        let lay = &lay.state;
        let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
        for g in 0..geom.n_gauss {
            let dn = &mut dn_buf[..n_nodes * d]; // [i * d + c]
            geom.dn_dx(g, dn)?;
            let w = geom.det_j_w(g);
            let row = stress.row(geom.cell, g);
            let mut sig = [0.0_f64; 9]; // [c * d + e]
            internal_force::voigt_stress_matrix(row, lay, d, &mut sig);
            // On a body of revolution the hoop strain's own non-linear part,
            // ½(u_r/r)², contributes `σ_θθ N_i N_j / r²` on the radial diagonal —
            // the initial-stress counterpart of the `N_i / r` row of `B`.
            let hoop = if geom.axisymmetric {
                let r = geom.radius(g);
                // The hoop closes the read list, exactly as it does for `Bᵀσ`.
                Some((geom.n_at_g(g), row[lay[lay.len() - 1] as usize] / (r * r)))
            } else {
                None
            };
            for i in 0..n_nodes {
                for j in 0..n_nodes {
                    // Scalar gᵢⱼ = Σ_{c,e} (∂N_i/∂x_c) σ_ce (∂N_j/∂x_e).
                    let mut gij = 0.0;
                    for c in 0..d {
                        for e in 0..d {
                            gij += dn[i * d + c] * sig[c * d + e] * dn[j * d + e];
                        }
                    }
                    gij *= w;
                    // Same scalar on every component's diagonal block (δ_ab).
                    for a in 0..d {
                        ke[(i * d + a) * dofs + (j * d + a)] += gij;
                    }
                    if let Some((n, s_hoop)) = hoop {
                        ke[(i * d) * dofs + (j * d)] += s_hoop * n[i] * n[j] * w;
                    }
                }
            }
        }
        Ok(())
    }

    /// La délégation complète : `∫ Bᵀ D B` d'une maille, où `D` vient du
    /// `tangent_point` de `domain`. C'est tout ce qu'une physique du continuum a
    /// à écrire pour son `element_tangent` — une ligne.
    ///
    /// Générique sur `D`, donc monomorphisé : le `tangent_point` appelé à chaque
    /// point de Gauss est un appel **statique**.
    #[allow(clippy::too_many_arguments)]
    /// La délégation complète : `∫ Bᵀ D B` d'une maille, où `D` vient du
    /// `tangent_point` de `domain`. C'est tout ce qu'une physique du continuum a
    /// à écrire pour son `element_tangent` — une ligne.
    ///
    /// Générique sur `D`, donc monomorphisé : le `tangent_point` appelé à chaque
    /// point de Gauss est un appel **statique**.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::ElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::ops::{element_field, matrix, model};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id(), n[3].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&mesh)?;
    /// # let modele = model::elasticity(&fes, Kinematics::PlaneStress)?;
    /// # let materiaux = element_field::material_field(
    /// #     &modele, &[("E", 210_000.0), ("nu", 0.3)])?;
    /// # let eps = ElementField::new(
    /// #     &fes, vec!["eps_xx".into(), "eps_yy".into(), "eps_xy".into()])?;
    /// // Bout en bout : la tangente d'une loi linéaire **est** sa raideur, et
    /// // elle l'atteint par la voie commune — sans qu'aucun champ de modules
    /// // n'ait été matérialisé pour l'y porter.
    /// let kt = matrix::tangent(&modele, &materiaux, &eps, None, None)?;
    /// let k = matrix::stiffness(&modele, &materiaux)?;
    /// assert_eq!(kt.dense()?, k.dense()?);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_tangent_of<B: crate::models::Behavior + ?Sized>(
        &self,
        behavior: &B,
        geoms: &[CellGeom],
        lay: &crate::models::ZoneLayout,
        deformation: &SubElementField,
        prev: &SubElementField,
        material: &SubElementField,
        dt: f64,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        self.element_tangent_with(
            geom,
            |g, d| {
                behavior.tangent_point(
                    geom,
                    g,
                    lay,
                    deformation.row(geom.cell, g),
                    prev.row(geom.cell, g),
                    material.row(geom.cell, g),
                    dt,
                    d,
                )
            },
            ke,
        )
    }

    pub(crate) fn element_tangent_with<F>(
        &self,
        geom: &CellGeom,
        mut d_at: F,
        ke: &mut [f64],
    ) -> Result<()>
    where
        F: FnMut(usize, &mut [[f64; 6]; 6]) -> Result<()>,
    {
        let n_nodes = geom.n_nodes;
        let space_dim = geom.space_dim;
        let dofs = space_dim * n_nodes;
        let v = voigt_size(space_dim, self.kinematics);
        // Mêmes tampons de pile que `element_stiffness`, hors de la boucle.
        let mut dn_buf = [0.0_f64; MAX_CELL_DOFS];
        let mut b: VoigtRows = [[0.0; MAX_CELL_DOFS]; 6];
        let mut db: VoigtRows = [[0.0; MAX_CELL_DOFS]; 6];
        let mut d = [[0.0_f64; 6]; 6];
        for g in 0..geom.n_gauss {
            // Same hoop row as `element_stiffness` on a body of revolution.
            let hoop = if self.kinematics.is_axisymmetric() {
                Some((geom.n_at_g(g), geom.radius(g)))
            } else {
                None
            };
            geom.dn_dx(g, &mut dn_buf[..n_nodes * space_dim])?;
            b_matrix_into(
                &dn_buf[..n_nodes * space_dim],
                n_nodes,
                space_dim,
                hoop,
                &mut b,
            );
            // Le fournisseur écrit `D` dans le tampon de l'appelant. Il est
            // **générique**, jamais un `&dyn Fn` : un appel virtuel par point de
            // Gauss se voit, et ce noyau en fait un par point.
            d_at(g, &mut d)?;
            // DB = D·B (voigt × dofs), then Kᵉ += Bᵀ (DB) · |J| w.
            for r in 0..v {
                for c in 0..dofs {
                    let mut acc = 0.0;
                    for w in 0..v {
                        acc += d[r][w] * b[w][c];
                    }
                    db[r][c] = acc;
                }
            }
            let w = geom.det_j_w(g);
            for r in 0..dofs {
                for c in 0..dofs {
                    let mut acc = 0.0;
                    for vv in 0..v {
                        acc += b[vv][r] * db[vv][c];
                    }
                    ke[r * dofs + c] += acc * w;
                }
            }
        }
        Ok(())
    }
}
