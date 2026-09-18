//! Distributed load — the consistent nodal forces of a given density,
//! `f_i = ∫_Γ φ N_i dΓ` (Cast3M `FLUX` / `SOUR` / `PRES` on a fixed geometry).
//!
//! The first sub-model whose whole term sits on the **right** of
//! `Σ f_int = Σ f_ext`: its derivative with respect to `u` is zero, so it
//! contributes to no matrix at all. It is a [`Domain`] — it integrates on an FE
//! subspace and reads its density there — without being a
//! [`Behavior`](crate::models::Behavior): there is no law to evaluate, only a
//! given value to weight by the shape functions.
//!
//! That shape is exactly what the old free operator could not express. As
//! `ops::node_field::flux` it produced the same numbers, but from outside the
//! model: nothing recorded that the term existed, nothing archived it with the
//! problem, and forgetting to union it into the load was silent. As a
//! sub-model it enters `r = Σ rᵢ` like every other term.
//!
//! The density lives in the material, named after the row it feeds — `phi_q`
//! for a heat source, `phi_f_x` for a traction. Uniform or varying per Gauss
//! point is the material field's business, not this physics'.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::containers::model::{Model, SubModel};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::kernel::MAX_CELL_DOFS;
use crate::models::transfer::{owner_physics, physics_slice};
use crate::models::{
    CellGeom, Contribution, Domain, ElementLayout, MatrixKind, MatrixLayout, Physics,
    ResidualContribution, SubModelKind,
};
use serde::{Deserialize, Serialize};

/// The material component carrying the density of a distributed load, named
/// after the dual row it feeds.
///
/// ```
/// # use pyrucast::models::flux;
/// assert_eq!(flux::density_name("q"), "phi_q");
/// assert_eq!(flux::density_name("f_x"), "phi_f_x");
/// ```
pub fn density_name(dual: &str) -> String {
    format!("phi_{dual}")
}

/// A distributed load on one FE subspace: `∫ φ N dΓ`, into one dual row.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::flux::Flux;
/// # use pyrucast::models::{Domain, MatrixKind, Physics, SubModelKind};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
/// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let zone = fes.get(0).unwrap();
/// // The target: the model being loaded. It is the one that owns the "q" row.
/// let cible = model::heat_conduction(&fes)?;
/// let charge = Flux::new(zone, &cible, "q".into())?;
/// // A load has no primal: it writes into the dual row of another
/// // autre physique, et n'introduit aucune inconnue.
/// assert!(charge.primal_vars().is_empty());
/// assert_eq!(charge.dual_vars(), vec!["q".to_string()]);
/// // Its derivative is zero: no matrix, of any kind.
/// assert!(charge.matrix_layout(MatrixKind::Stiffness).is_none());
/// assert_eq!(charge.material_components(), vec!["phi_q".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Flux {
    fespace: Handle<SubFiniteElementSpace>,
    support: Handle<SubMesh>,
    dual: String,
    physics: Physics,
}

impl Flux {
    /// Build a distributed load on `fespace`, feeding the `dual` row.
    ///
    /// `target` is the model this load feeds. The nature of the load cannot be
    /// deduced from the row name — the caller chooses it freely — but the model
    /// being loaded knows it: the sub-model that declares `dual` gives both the
    /// physics and the proof that the row is assembled by someone.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::flux::Flux;
    /// # use pyrucast::models::{Physics, SubModelKind};
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let zone = fes.get(0).unwrap();
    /// # let cible = model::heat_conduction(&fes)?;
    /// assert!(Flux::new(zone.clone(), &cible, "q".into()).is_ok());
    /// // An empty dual row designates nothing: refused at construction.
    /// assert!(Flux::new(zone.clone(), &cible, String::new()).is_err());
    /// // A row the target does not assemble either: that is the whole point
    /// // de recevoir le modèle chargé.
    /// assert!(Flux::new(zone, &cible, "f_x".into()).is_err());
    /// // A formulation owning its own interpolation (a beam) cannot receive a
    /// // consistent load from here: it alone knows its shape functions. Settled
    /// // at construction, not at the Gauss point.
    /// # use pyrucast::atoms::Interpolation;
    /// let poutre = FiniteElementSpace::new(&maillage, Interpolation::ModelEmbedded).unwrap();
    /// assert!(Flux::new(poutre.get(0).unwrap(), &cible, "q".into()).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        fespace: Handle<SubFiniteElementSpace>,
        target: &Model,
        dual: String,
    ) -> Result<Self> {
        if dual.is_empty() {
            return Err(PyrucastError::Message(
                "Flux: name the dual row this load feeds, e.g. \"q\" for a heat source or \
                 \"f_x\" for a traction"
                    .into(),
            ));
        }
        // The load's kind is that of the term it loads. The row's name does not give
        // it — the user picks it freely — but the loaded model does: we look for the
        // sub-model that owns that dual.
        // A row nobody assembles used to build a mute load; it is now a construction
        // error.
        let physics = owner_physics(target, |sub| sub.as_kind().dual_vars().contains(&dual))
            .ok_or_else(|| {
                PyrucastError::Message(format!(
                    "Flux: `{dual}` is not a dual row of the model it loads — it declares {:?}",
                    target.dual_vars()
                ))
            })?;
        // A distributed load weights by the **field's** shape functions: it needs
        // one, and that is a fact of the zone, settled here once and for all rather
        // than at every Gauss point. A formulation owning its own interpolation (a
        // `ModelEmbedded` beam) must supply its consistent load itself.

        crate::models::kernel::require_field_basis(&fespace, "shape values")?;
        let submesh = fespace.read().submesh();
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            dual,
            physics,
        })
    }

    /// Layout of the term: the subspace it integrates on, the node support it
    /// scatters to, and the single row it feeds.
    fn layout(&self) -> MatrixLayout {
        MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: vec![self.dual.clone()],
            primal_vars: Vec::new(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: false,
        }
    }
}

impl SubModelKind for Flux {
    /// None: a load introduces no unknown. It writes into a row it does not
    /// own — the dual of the physics it loads — which is what makes it a term
    /// with a target rather than one of its own.
    fn primal_vars(&self) -> Vec<String> {
        Vec::new()
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.dual.clone()]
    }

    fn physics(&self) -> &'static [Physics] {
        physics_slice(self.physics)
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    /// Nothing, for any kind. `∂r/∂u = 0` is the whole point: a given density
    /// does not move when the solution does. The default would fall back to
    /// literal stiffness blocks, which is the constraint path, not this one.
    fn contributions(
        &self,
        _kind: MatrixKind,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        Ok(Vec::new())
    }

    fn external_force_contribution(&self) -> Vec<ResidualContribution> {
        vec![ResidualContribution::Computed(self.layout())]
    }

    /// `f_i = Σ_g φ(g) N_i(g) |J|_g w_g` — the consistent nodal load, weighted
    /// by the shape functions rather than split evenly.
    fn external_force_element(
        &self,
        geoms: &[CellGeom],
        material: &SubElementField,
        lay: &ElementLayout,
        fe: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let phi = lay.material[0] as usize;
        // On a C¹ basis (a Bernoulli beam) the field's basis is not the geometric
        // one: it is the basis that carries the nodal moments of a distributed load.
        // The buffer is on the stack, outside the loop.
        let mut n_buf = [0.0_f64; MAX_CELL_DOFS];
        for g in 0..geom.n_gauss {
            let shape = geom.field_n_at_g(g, &mut n_buf);
            let w = geom.det_j_w(g) * material.row(geom.cell, g)[phi];
            if w == 0.0 {
                continue;
            }
            for i in 0..geom.n_nodes {
                fe[i] += shape[i] * w;
            }
        }
        Ok(())
    }

    fn label(&self) -> &'static str {
        "Flux"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        format!(
            "Flux<{}> sur {} maille(s) → {}",
            self.physics.name(),
            self.fespace.read().cell_count(),
            self.dual
        )
    }
}

impl Domain for Flux {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// The density, named after the row it feeds — `phi_q`, `phi_f_x`. Required:
    /// a load whose density nobody supplied is not a load of zero, it is a
    /// mistake, and it says so at assembly.
    fn material_components(&self) -> Vec<String> {
        vec![density_name(&self.dual)]
    }
}

crate::physics_operator! {
    /// Distributed-load `Model` spanning **every** subspace of `fes`, feeding
    /// the `dual` row of `target`; the density is supplied at assembly time, in
    /// the material, as `phi_<dual>`.
    ///
    /// The subspace comes first — it is what the operator sweeps — and the
    /// loaded model follows: a load owns no unknown of its own, so it is the
    /// target that says which physics the row belongs to, and that the row is
    /// assembled by someone at all.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::model::Model;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// # sm.add_cell(&[n[0].id(), n[1].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// // A load distributed over the thermal model it feeds: that model owns the
    /// // "q" row, and gives its kind.
    /// let conduction = model::heat_conduction(&fes)?;
    /// let charge = model::flux(&fes, &conduction, "q".into())?;
    /// assert_eq!(charge.dual_vars(), vec!["q".to_string()]);
    /// assert!(charge.primal_vars().is_empty());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn flux(fes, target, dual: String) via SubModel::flux;
    python: "`model.flux(fespace, target, dual)` — une **charge répartie** :\nles forces nodales cohérentes `∫ φ N dΓ` d'une densité donnée, versées dans\nla ligne duale `dual` (`\"q\"` pour une source de chaleur, `\"f_x\"` pour une\ntraction) du modèle `target`, qui lui donne sa nature et atteste que la\nligne est bien assemblée par quelqu'un.\n\nMatériau : `phi_<dual>` — la densité, uniforme ou variable par point de\nGauss selon le champ fourni.\n\nSa dérivée par rapport à la solution est **nulle** : elle ne contribue à\naucune matrice, seulement au second membre, qu'on récupère par\n`node_field.external_forces(model, materials)`. C'est ce qui la distingue\nd'une physique : elle n'introduit aucune inconnue et écrit dans la ligne\nduale d'une autre."
}
