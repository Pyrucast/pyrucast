//! Surface exchange with an imposed ambient — the Robin (film) boundary.
//!
//! On a boundary `Γ`, the outward flux obeys `q·n = h·(a − a_ext)`: Newton's law
//! of cooling when `a` is a temperature, a surface mass-transfer law when it is a
//! concentration, a Winkler elastic foundation when it is a displacement. The
//! weak form,
//!
//! ```text
//! ∮_Γ (q·n) δa dΓ = ∮_Γ h·(a − a_ext) δa dΓ,
//! ```
//!
//! splits into a **film matrix** and an **ambient load**:
//!
//! ```text
//! K_ij = h ∫_Γ N_i N_j dΓ    (this sub-model, into the stiffness),
//! f_i  = h·a_ext ∫_Γ N_i dΓ  (a right-hand side, built with
//!                             crate::models::flux — not stored here).
//! ```
//!
//! Only the first is here, and that is the whole difference from
//! [`interface_transfer`](crate::models::interface_transfer): there the far side
//! is an **unknown**, so what would be this right-hand side becomes a coupling
//! block in the matrix. The two share their kernel, in
//! [`transfer`](crate::models::transfer).
//!
//! ## What is exchanged is the caller's to say
//!
//! The sub-model is given `(primal, dual)` pairs and derives everything from
//! them — the DOF names it couples into, the coefficient `h_<primal>` it reads,
//! the flux `flux_<primal>` it reports. Passing `("T", "q")` reproduces the
//! classical thermal film; passing the three displacement pairs gives an elastic
//! foundation, with a stiffness per direction. The names being those of the bulk
//! physics is what makes the boundary term **couple straight into** it, exactly
//! as it always did for conduction.
//!
//! **No normal is needed.** The normal is already consumed in passing from `q·n`
//! to `h·(a − a_ext)`; what remains under the integral is a scalar times the
//! surface measure `dΓ = |J|`, which
//! [`CellGeom::det_j_w`](crate::models::kernel::CellGeom::det_j_w) returns as
//! `√det(JᵀJ)` — a magnitude, invariant under the boundary mesh's orientation
//! (winding). Contrast a pressure or a signed flux, where the direction matters.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::SubModel;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::transfer::{
    ambient_name, coefficient_name, exchange_matrix, flux_name, internal_force, material_contract,
    physics_slice,
};
use crate::models::ElementLayout;
use crate::models::ZoneLayout;
use crate::models::{Behavior, CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Surface exchange with an imposed ambient, on a boundary FE subspace.
///
/// Material data (the coefficients `h_<primal>`) is **not** stored here; it is
/// supplied at assembly time via [`crate::ops::matrix::stiffness`], read from
/// the boundary cells of the material field.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let zone = fes.get(0).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # let mult = mesh::barycenter(&impose).unwrap();
/// # use pyrucast::models::boundary_transfer::BoundaryTransfer;
/// # use pyrucast::models::Domain;
/// // L'échange de surface : nommer les DDL de la physique de volume est ce
/// // qui fait que le terme s'y couple.
/// let b = BoundaryTransfer::new(
///     zone.clone(), vec![("T".into(), "q".into())], Physics::Thermal)?;
/// assert_eq!(b.primal_vars(), vec!["T".to_string()]);
/// assert!(b.material_components().contains(&"h_T".to_string()));
/// // Une liste vide n'a ni matrice ni coefficient : refusée.
/// assert!(BoundaryTransfer::new(zone.clone(), vec![], Physics::Thermal).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct BoundaryTransfer {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh, built
    /// once at construction. Reused as the row/col support of every assembled
    /// film block — no per-assembly rebuild.
    pub(crate) support: Handle<SubMesh>,
    /// The transferred quantities, as `(primal, dual)` pairs.
    pub(crate) components: Vec<(String, String)>,
    /// The physics nature this exchange belongs to — what `model.filter(…)`
    /// selects it by. It cannot be deduced from the variable names, which are
    /// free, so the caller declares it.
    pub(crate) physics: Physics,
}

impl BoundaryTransfer {
    /// Surface exchange on a boundary FE subspace (an edge mesh in 2-D, a
    /// surface mesh in 3-D). Builds the stable POI1 [`SubMesh`] covering the
    /// subspace's unique nodes (reused as the row/col support of every assembled
    /// block). Errors on an empty `components`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, Physics, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # use pyrucast::models::boundary_transfer::BoundaryTransfer;
    /// # use pyrucast::models::Domain;
    /// // L'échange de surface : nommer les DDL de la physique de volume est ce
    /// // qui fait que le terme s'y couple.
    /// let b = BoundaryTransfer::new(
    ///     zone.clone(), vec![("T".into(), "q".into())], Physics::Thermal)?;
    /// assert_eq!(b.primal_vars(), vec!["T".to_string()]);
    /// assert!(b.material_components().contains(&"h_T".to_string()));
    /// // Une liste vide n'a ni matrice ni coefficient : refusée.
    /// assert!(BoundaryTransfer::new(zone.clone(), vec![], Physics::Thermal).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        fespace: Handle<SubFiniteElementSpace>,
        components: Vec<(String, String)>,
        physics: Physics,
    ) -> Result<Self> {
        material_contract("BoundaryTransfer", &components)?;
        let submesh = fespace.read().submesh();
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            components,
            physics,
        })
    }
}

impl SubModelKind for BoundaryTransfer {
    fn primal_vars(&self) -> Vec<String> {
        self.components.iter().map(|(p, _)| p.clone()).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        self.components.iter().map(|(_, d)| d.clone()).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn as_behavior(&self) -> Option<&dyn Behavior> {
        Some(self)
    }

    /// The ambient, integrated on the same boundary as the film it belongs to.
    /// This is the term that used to be written by hand through the free `flux`
    /// operator, where forgetting it read as an ambient of zero.
    fn external_force_contribution(&self) -> Vec<crate::models::ResidualContribution> {
        self.stiffness_layout()
            .map(crate::models::ResidualContribution::Computed)
            .into_iter()
            .collect()
    }

    fn external_force_element(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        fe: &mut [f64],
    ) -> Result<()> {
        self.ambient_element(geoms, material, lay, fe)
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

    /// Internal nodal fluxes `q_i = ∫ N_i · flux dΓ` — the **`N`-weighted**
    /// boundary counterpart of the `Bᵀ` continuum default. For this linear law it
    /// equals `(K·a)_i`, so it fits the « internal forces == K·u » invariant.
    fn internal_force_reads(&self) -> Vec<String> {
        self.components
            .iter()
            .map(|(p, _)| crate::models::transfer::flux_name(p))
            .collect()
    }

    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        lay: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        internal_force(&geoms[0], stress, lay, fe)
    }

    fn physics(&self) -> &'static [Physics] {
        physics_slice(self.physics)
    }

    fn label(&self) -> &'static str {
        "BoundaryTransfer"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<BoundaryTransfer>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

impl Domain for BoundaryTransfer {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// One coefficient per transferred quantity, named after it — `h_T`,
    /// `h_c_H2`, `h_u_x`. Derived, which is why the contract had to become
    /// owned.
    /// The coefficients first, then the ambients — `h_T, …, a_ext_T, …`. The
    /// order matters and is the contract: the exchange kernel indexes the head
    /// of the list, the ambient kernel its tail, so adding the second family
    /// left every existing index where it was.
    fn material_components(&self) -> Vec<String> {
        let h = self.components.iter().map(|(p, _)| coefficient_name(p));
        let a = self.components.iter().map(|(p, _)| ambient_name(p));
        h.chain(a).collect()
    }

    /// The film matrix — the exchange kernel with both sides on the same cell,
    /// which is exactly what an interface's diagonal block is.
    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material;
        exchange_matrix(
            geom,
            geom,
            mat,
            &lay.material[..self.components.len()],
            1.0,
            ke,
        )
    }
}

impl Behavior for BoundaryTransfer {
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Vec<String> {
        self.components.iter().map(|(p, _)| flux_name(p)).collect()
    }

    /// The linear film law, one quantity at a time: the weak-form flux density
    /// `flux_<primal> = h_<primal> · <primal>` at one Gauss point, from the
    /// interpolated field. This is what the assembled film matrix integrates
    /// (`∫ N_i·flux = (K·a)_i`); the ambient part `h·a_ext` lives in the load,
    /// not here. No internal state.
    fn deformation_reads(&self) -> Vec<String> {
        self.components.iter().map(|(p, _)| p.clone()).collect()
    }

    fn integrate_point(
        &self,
        _geom: &CellGeom,
        _g: usize,
        lay: &ZoneLayout,
        deformation: &[f64],
        _prev: &[f64],
        material: &[f64],
        _dt: f64,
        out: &mut [f64],
    ) -> Result<()> {
        for v in 0..self.components.len() {
            let h = material[lay.material[v] as usize];
            out[v] = h * deformation[lay.deformation[v] as usize];
        }
        Ok(())
    }
}

impl BoundaryTransfer {
    /// The ambient term `∫ h·a_ext·N dΓ` of one cell — the half of the film law
    /// that does not depend on `u`, and so belongs on the right of the equals
    /// sign rather than in the matrix.
    ///
    /// It reads the tail of the material layout, the ambients, and the head of
    /// it, the coefficients: the product `h·a_ext` is what the weak form
    /// integrates, not either factor alone.
    fn ambient_element(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let n = self.components.len();
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g);
            let w = geom.det_j_w(g);
            let row = material.row(geom.cell, g);
            for v in 0..n {
                let hw = row[lay.material[v] as usize] * row[lay.material[n + v] as usize] * w;
                if hw == 0.0 {
                    continue;
                }
                for i in 0..geom.n_nodes {
                    fe[i * n + v] += hw * shape[i];
                }
            }
        }
        Ok(())
    }
}

crate::physics_operator! {
    /// Surface exchange `Model` spanning **every** subspace of a *boundary*
    /// `fes` — one [`SubModel::BoundaryTransfer`] per
    /// [`SubFiniteElementSpace`].
    /// Parent-level operator; the
    /// coefficients `h_<primal>` are supplied at assembly time. Couples into the
    /// bulk physics whose DOFs it names:
    ///
    /// ```text
    /// model::heat_conduction(&bulk)?.union(
    ///     &model::boundary_transfer(&skin, vec![("T".into(), "q".into())], Physics::Thermal)?)?
    /// ```
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::{Model, SubModel};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::models::{Physics, RelationSense};
    /// # use pyrucast::ops::mesh;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # let mult = mesh::barycenter(&impose).unwrap();
    /// # let mut bord = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # bord.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let fes_bord = FiniteElementSpace::lagrange1(&Mesh::from_submesh(bord))?;
    /// let m = model::boundary_transfer(
    ///     &fes_bord, vec![("T".into(), "q".into())], Physics::Thermal)?;
    /// assert_eq!(m.primal_vars(), vec!["T".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn boundary_transfer(fes, components: Vec<(String, String)>, physics: Physics) via SubModel::boundary_transfer;
    python: "`model.boundary_transfer(fespace, components, physics)` — surface\nexchange with an **imposed ambient** (Robin / film) spanning every\nsubspace of a *boundary* `fespace` (edge mesh in 2-D, surface mesh in\n3-D).\n\n`components` is a list of `(primal, dual)` pairs — naming the bulk\nphysics' own DOFs is what makes the boundary term couple into it:\n\n| you write | you get |\n|---|---|\n| `[(\"T\", \"q\")], \"thermal\"` | Newton's law of cooling |\n| `[(\"c_H2\", \"j_H2\")], \"diffusion\"` | a surface mass-transfer law |\n| `[(\"u_x\", \"f_x\"), (\"u_y\", \"f_y\")], \"mechanical\"` | a Winkler elastic foundation |\n\nThe coefficients `h_<primal>` (one per pair) are supplied at assembly\ntime; the ambient value enters as a load `h·a_ext·∫N_i dΓ`, built with\n`flux(...)`. Compose with `|`:\n`model.heat_conduction(bulk) | model.boundary_transfer(skin, [(\"T\", \"q\")], \"thermal\")`."
}
