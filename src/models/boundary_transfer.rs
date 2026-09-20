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
//! K_ij = h ∫_Γ N_i N_j dΓ    (the film, into the stiffness),
//! f_i  = h·a_ext ∫_Γ N_i dΓ  (the ambient, into the external forces).
//! ```
//!
//! Both are this sub-model's, `a_ext` being a material component beside `h`.
//! That the far side is a **datum** is the whole difference from
//! [`interface_transfer`](crate::models::interface_transfer): there it is an
//! **unknown**, so what is here a right-hand side becomes a coupling block in
//! the matrix. The two share their kernel, in
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
//! ## The nature comes from the target
//!
//! The pairs name rows of a physics that already exists, so the exchange is
//! built **against** it: the model it couples into is passed as `target`, and
//! every pair must be one that model assembles. That is the proof the term
//! lands somewhere — a mistyped name used to build an exchange coupled to
//! nothing — and it gives the nature for free: `("T", "q")` does not say it is
//! thermal, the conduction that assembles it does.
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
use crate::containers::model::{Model, SubModel};
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::handle::Handle;
use crate::models::transfer::{
    ambient_name, coefficient_name, exchange_matrix, material_contract, physics_slice,
    target_physics,
};
use crate::models::ElementLayout;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
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
/// # use pyrucast::models::boundary_transfer::BoundaryTransfer;
/// # use pyrucast::models::{Domain, SubModelKind};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// // The surface exchange, against the conduction it cools: naming its DOFs
/// // is what makes the term couple to it.
/// let conduction = model::heat_conduction(&fes)?;
/// let b = BoundaryTransfer::new(zone, &conduction, vec![("T".into(), "q".into())])?;
/// assert_eq!(b.primal_vars(), vec!["T".to_string()]);
/// assert_eq!(b.material_components(), vec!["h_T".to_string(), "a_ext_T".to_string()]);
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
    /// selects it by. The variable names are free and cannot imply it, so it
    /// is read from the target at construction and kept.
    pub(crate) physics: Physics,
}

impl BoundaryTransfer {
    /// Surface exchange on a boundary FE subspace (an edge mesh in 2-D, a
    /// surface mesh in 3-D), coupling into `target`.
    ///
    /// `target` is the model whose unknowns the exchange acts on. Every
    /// `(primal, dual)` pair must be one it assembles — `primal` among its
    /// unknowns, paired with the row `dual` — and all must belong to one
    /// nature, which the exchange then carries.
    ///
    /// Builds the stable POI1 [`SubMesh`] covering the subspace's unique nodes
    /// (reused as the row/col support of every assembled block). Errors on an
    /// empty `components`, on a pair `target` does not assemble, and on pairs
    /// of two natures.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::boundary_transfer::BoundaryTransfer;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::{Physics, SubModelKind};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// let conduction = model::heat_conduction(&fes)?;
    /// // The kind is not an argument: it comes from the conduction, which
    /// // assemble `T` et `q`.
    /// let film = BoundaryTransfer::new(zone.clone(), &conduction, vec![("T".into(), "q".into())])?;
    /// assert_eq!(film.physics(), &[Physics::Thermal]);
    /// // A pair the target does not assemble would couple to nothing: refused.
    /// assert!(BoundaryTransfer::new(zone.clone(), &conduction, vec![("u_x".into(), "f_x".into())])
    ///     .is_err());
    /// // An exchange carries one kind: thermal and mechanical mixed, refused.
    /// let deux = conduction.union(&model::elasticity(&fes, Kinematics::PlaneStress)?)?;
    /// assert!(BoundaryTransfer::new(
    ///     zone.clone(), &deux, vec![("T".into(), "q".into()), ("u_x".into(), "f_x".into())])
    ///     .is_err());
    /// // Une liste vide n'a ni matrice ni coefficient : refusée.
    /// assert!(BoundaryTransfer::new(zone, &conduction, vec![]).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        fespace: Handle<SubFiniteElementSpace>,
        target: &Model,
        components: Vec<(String, String)>,
    ) -> Result<Self> {
        material_contract("BoundaryTransfer", &components)?;
        let physics = target_physics("BoundaryTransfer", target, &components)?;
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

    /// The **primal itself**, read at the Gauss points — not a stress, and not
    /// the output of a law.
    ///
    /// A boundary transfer has no constitutive law: its `h·a` is the
    /// coefficient of its own operator applied at a point, and `h` is the very
    /// coefficient that built `∫ h NᵀN`. It used to declare a behaviour whose
    /// kernel was `out[v] = h * deformation[v]` — recomputing, point by point,
    /// what the matrix already knew — because `Domain` demanded one. It no
    /// longer does.
    fn internal_force_reads(&self) -> Vec<String> {
        self.components.iter().map(|(p, _)| p.clone()).collect()
    }

    /// `q_i = ∫ h·a·N_i dΓ` — the internal half of the film law, the exact
    /// mirror of its ambient half, and equal to `(K·a)_i` as it must be.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        primal: &SubElementField,
        lay: &[u32],
        material: &SubElementField,
        mat: &[u32],
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let n = lay.len();
        for g in 0..geom.n_gauss {
            let shape = geom.n_at_g(g);
            let w = geom.det_j_w(g);
            let a = primal.row(geom.cell, g);
            let h = material.row(geom.cell, g);
            for v in 0..n {
                let hw = h[mat[v] as usize] * a[lay[v] as usize] * w;
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
    /// Parent-level operator; the coefficients `h_<primal>` and the ambients
    /// `a_ext_<primal>` are supplied at assembly time. Couples into `target`,
    /// the bulk physics whose DOFs it names and whose nature it takes:
    ///
    /// ```text
    /// let conduction = model::heat_conduction(&bulk)?;
    /// conduction.union(&model::boundary_transfer(&skin, &conduction, vec![("T".into(), "q".into())])?)?
    /// ```
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::Physics;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let mut bord = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # bord.add_cell(&[n[0].id(), n[1].id()])?;
    /// # let fes_bord = FiniteElementSpace::lagrange1(&Mesh::from_submesh(bord))?;
    /// let conduction = model::heat_conduction(&fes)?;
    /// let film = model::boundary_transfer(&fes_bord, &conduction, vec![("T".into(), "q".into())])?;
    /// assert_eq!(film.primal_vars(), vec!["T".to_string()]);
    /// // United with the conduction, it files under the same kind as it.
    /// let m = conduction.union(&film)?;
    /// assert_eq!(m.filter(Physics::Thermal).len(), 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn boundary_transfer(fes, target, components: Vec<(String, String)>) via SubModel::boundary_transfer;
    python: "`model.boundary_transfer(fespace, target, components)` — surface\nexchange with an **imposed ambient** (Robin / film) spanning every\nsubspace of a *boundary* `fespace` (edge mesh in 2-D, surface mesh in\n3-D), coupling into the model `target`.\n\n`components` is a list of `(primal, dual)` pairs, each one that `target`\nassembles — naming the bulk physics' own DOFs is what makes the boundary\nterm couple into it:\n\n| `target` | `components` | you get |\n|---|---|---|\n| `heat_conduction` | `[(\"T\", \"q\")]` | Newton's law of cooling |\n| `fick(..., \"H2\")` | `[(\"c_H2\", \"j_H2\")]` | a surface mass-transfer law |\n| `elasticity` | `[(\"u_x\", \"f_x\"), (\"u_y\", \"f_y\")]` | a Winkler elastic foundation |\n\nThe nature (`\"thermal\"`, `\"diffusion\"`, `\"mechanical\"`) is not an\nargument: it is the one of the sub-model of `target` assembling the pairs.\nA pair `target` does not assemble, or pairs of two natures, raise.\n\nMaterial, per pair: the coefficient `h_<primal>` and the ambient\n`a_ext_<primal>`. The film `h∫NᵢNⱼ` goes into the stiffness, the ambient\nterm `h·a_ext·∫Nᵢ dΓ` comes out of `node_field.external_forces(...)`.\nCompose with `|`:\n`conduction = model.heat_conduction(bulk)`\n`model = conduction | model.boundary_transfer(skin, conduction, [(\"T\", \"q\")])`."
}
