//! Dirichlet constraint via Lagrange multipliers.
//!
//! A Dirichlet sub-model imposes `u = u_d` on a set of constrained nodes. It
//! is a *constraint*, not a volumetric physics: it carries no material and no
//! constitutive law. It is built from **two meshes supplied by the user** — it
//! creates no node and never mutates the
//! [`Coords`](crate::coords::Coords):
//!
//! - `imposed_mesh`  — POI1 (for now): the constrained nodes (shared with the
//!   target physics), one node per cell;
//! - `multiplier_mesh` — POI1: the support of the Lagrange multipliers, paired
//!   element-for-element with `imposed_mesh` (same submesh structure, same
//!   per-submesh cell count). Typically built from `imposed_mesh` with the
//!   generic [`barycenter`](crate::ops::mesh::barycenter()) mesher (colocated fresh nodes),
//!   but the user is free to colocate, offset, or even reuse the constrained
//!   nodes themselves.
//!
//! # Variables
//!
//! A Dirichlet sub-model imposes the **primal of another model** yet owns its
//! own pair of variables; the names keep the two apart:
//!
//! - `imposed_variable` (e.g. `"T"`) — the constrained primal of the **target**
//!   physics (a *column* of the target's stiffness);
//! - `target_dual` (e.g. `"q"`) — the **target's** dual variable; the row into
//!   which the constraint reaction `Cᵀ` is added (it cannot be derived from
//!   `imposed_variable` — the `T → q` map is target-specific);
//! - `multiplier` (default `lambda_<imposed_variable>`) — this sub-model's own
//!   **primal**: the Lagrange multiplier, an unknown of the augmented system,
//!   whose solved value is the **reaction**;
//! - `imposed_value` (default `imposed_<imposed_variable>`) — this sub-model's
//!   own **dual**: the constraint-equation row, and the **slot** at which the
//!   user writes the imposed value `u_d` in the load `SubNodeField` (on the
//!   multiplier node).
//!
//! `multiplier` and `imposed_value` are derived from `imposed_variable` but can
//! be overridden by hand.

use crate::aggregate::Aggregate;
use crate::atoms::{ElementType, NodeId};
use crate::containers::element_field::SubElementField;
use crate::containers::mesh::Mesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::{
    constraint_block_pair, Constraint, ConstraintTerm, Contribution, MatrixKind, Physics, Relation,
    RelationSense, SubModelKind,
};
use serde::{Deserialize, Serialize};

/// Default multiplier (primal) name for a constrained variable: `lambda_<v>`.
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
/// # use pyrucast::models::dirichlet;
/// // Dérivés plutôt que tabulés : `multiplier` et `imposed_value` laissés
/// // à `None` prennent ces valeurs.
/// assert_eq!(
///     (dirichlet::default_multiplier("T").as_str(),
///      dirichlet::default_imposed_value("T").as_str()),
///     ("lambda_T", "imposed_T"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn default_multiplier(imposed_variable: &str) -> String {
    format!("lambda_{imposed_variable}")
}

/// Default imposed-value (dual) name for a constrained variable: `imposed_<v>`.
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
/// # use pyrucast::models::dirichlet;
/// // Dérivés plutôt que tabulés : `multiplier` et `imposed_value` laissés
/// // à `None` prennent ces valeurs.
/// assert_eq!(
///     (dirichlet::default_multiplier("T").as_str(),
///      dirichlet::default_imposed_value("T").as_str()),
///     ("lambda_T", "imposed_T"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn default_imposed_value(imposed_variable: &str) -> String {
    format!("imposed_{imposed_variable}")
}

/// Dirichlet constraint imposed via Lagrange multipliers.
///
/// See the module documentation for the meaning of the four variable names and
/// the two meshes. The imposed value `u_d` is **not** stored here: the user
/// supplies it through the load `SubNodeField` at the multiplier node's
/// `imposed_value` component.
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
/// # use pyrucast::models::dirichlet::{self, Dirichlet};
/// # let cible = pyrucast::ops::model::heat_conduction(&fes).unwrap();
/// // Un appui : `u = u_d` aux nœuds imposés, par multiplicateur de Lagrange.
/// let d = Dirichlet::new(&cible, "T", &impose, &mult,
///                        RelationSense::Equality)?;
/// // Sans noms donnés, les défauts se dérivent de la variable imposée.
/// assert_eq!(d.relations()?.len(), 1);
/// assert_eq!(d.relations()?[0].imposed_value, dirichlet::default_imposed_value("T"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize)]
pub struct Dirichlet {
    /// Constrained primal of the target physics (e.g. `"T"`).
    pub(crate) imposed_variable: String,
    /// Target physics's dual; row where the reaction `Cᵀ` lands (e.g. `"q"`).
    pub(crate) target_dual: String,
    /// This sub-model's primal — the Lagrange multiplier (e.g. `"lambda_T"`).
    pub(crate) multiplier: String,
    /// This sub-model's dual — constraint row + imposed-value slot (e.g. `"imposed_T"`).
    pub(crate) imposed_value: String,
    /// POI1 mesh of the constrained nodes (one node per cell), as given by the
    /// user. Shares the submeshes (refcounts keep the nodes alive).
    pub(crate) imposed_mesh: Mesh,
    /// POI1 mesh of the multiplier nodes, paired element-for-element with
    /// `imposed_mesh` (same submesh structure, same per-submesh cell count).
    pub(crate) multiplier_mesh: Mesh,
    /// Equality (default) or unilateral inequality (`u ≥ u_d` / `u ≤ u_d`),
    /// solved by the active-set operator.
    #[serde(default)]
    pub(crate) sense: RelationSense,
}

impl Dirichlet {
    /// Build a Dirichlet constraint imposing `imposed_variable = u_d` on the
    /// nodes of `imposed_mesh`, with multipliers living on `multiplier_mesh`.
    ///
    /// `target_dual` is the dual variable of the target physics (e.g. `"q"`).
    /// `multiplier` / `imposed_value` default to `lambda_<imposed_variable>` /
    /// `imposed_<imposed_variable>` when `None`. Both meshes must be POI1 (for
    /// now), share one [`Coords`](crate::coords::Coords),
    /// and pair element-for-element (same number of submeshes, same cell count
    /// per submesh).
    ///
    /// `sense` turns the constraint unilateral: `GreaterEqual` imposes
    /// `u ≥ u_d` (`LessEqual`: `u ≤ u_d`), enforced only while in contact — such
    /// a model is solved by the active-set operator
    /// [`unilateral`](crate::ops::solver::unilateral).
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
    /// # use pyrucast::models::dirichlet::{self, Dirichlet};
    /// # let cible = pyrucast::ops::model::heat_conduction(&fes).unwrap();
    /// // Un appui : `u = u_d` aux nœuds imposés, par multiplicateur de Lagrange.
    /// let d = Dirichlet::new(&cible, "T", &impose, &mult,
    ///                        RelationSense::Equality)?;
    /// // Sans noms donnés, les défauts se dérivent de la variable imposée.
    /// assert_eq!(d.relations()?.len(), 1);
    /// assert_eq!(d.relations()?[0].imposed_value, dirichlet::default_imposed_value("T"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        target: &crate::containers::model::Model,
        variable: &str,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        sense: RelationSense,
    ) -> Result<Self> {
        // The target's own names, read from the target rather than retyped: the
        // `T → q` map is its business, and a mistyped one used to build a
        // constraint on a variable nobody assembles — a singular matrix, with
        // nothing to point at.
        let imposed_variable = variable.to_string();
        let target_dual = target.dual_of(variable).ok_or_else(|| {
            PyrucastError::Message(format!(
                "Dirichlet: `{variable}` is not a primal of the model it constrains — it \
                 declares {:?}",
                target.primal_vars()
            ))
        })?;
        if imposed_mesh.cell_count() == 0 {
            return Err(PyrucastError::Message(
                "Dirichlet: imposed_mesh must constrain at least one node".into(),
            ));
        }
        let n_sub = imposed_mesh.len();
        if multiplier_mesh.len() != n_sub {
            return Err(PyrucastError::Message(format!(
                "Dirichlet: imposed_mesh has {} submesh(es) but multiplier_mesh has {}",
                n_sub,
                multiplier_mesh.len()
            )));
        }
        // NodeIds are Coords-relative: both meshes must share it.
        let coords_i = imposed_mesh.coords()?;
        let coords_m = multiplier_mesh.coords()?;
        if !coords_i.same_object(&coords_m) {
            return Err(PyrucastError::Message(
                "Dirichlet: imposed_mesh and multiplier_mesh must share a Coords".into(),
            ));
        }
        // POI1 (for now) + element-for-element pairing (equal cell counts).
        for i in 0..n_sub {
            let imp = imposed_mesh.get(i)?;
            let mult = multiplier_mesh.get(i)?;
            let (iet, icount) = {
                let s = imp.read();
                (s.element_type(), s.cell_count())
            };
            let (met, mcount) = {
                let s = mult.read();
                (s.element_type(), s.cell_count())
            };
            if iet != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: imposed_mesh submesh {i} must be POI1, got {iet}"
                )));
            }
            if met != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: multiplier_mesh submesh {i} must be POI1, got {met}"
                )));
            }
            if icount != mcount {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: submesh {i} has {icount} constrained node(s) but \
                     {mcount} multiplier(s)"
                )));
            }
        }

        let multiplier = default_multiplier(&imposed_variable);
        let imposed_value = default_imposed_value(&imposed_variable);

        Ok(Self {
            imposed_variable,
            target_dual,
            multiplier,
            imposed_value,
            imposed_mesh: share(imposed_mesh)?,
            multiplier_mesh: share(multiplier_mesh)?,
            sense,
        })
    }
}

impl SubModelKind for Dirichlet {
    fn primal_vars(&self) -> Vec<String> {
        vec![self.multiplier.clone()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.imposed_value.clone()]
    }

    /// Its relations, applied to the solution: the reaction `Cᵀ λ` on the
    /// constrained nodes, and `C·u` on its own row.
    fn internal_force_contribution(&self) -> Vec<crate::models::ResidualContribution> {
        vec![crate::models::ResidualContribution::Relations]
    }

    fn as_constraint(&self) -> Option<&dyn Constraint> {
        Some(self)
    }

    /// Dirichlet contributes its **literal** C / Cᵀ blocks directly — it has no
    /// [`stiffness_layout`](SubModelKind::stiffness_layout) (nothing is
    /// integrated on a cell), so it bypasses the layout-driven default and
    /// returns a single [`Contribution::Literal`] carrying the filled blocks.
    /// This is *the* seam that keeps the assembler free of any Dirichlet-specific
    /// special case.
    fn contributions(
        &self,
        kind: MatrixKind,
        _material: Option<&crate::handle::Handle<SubElementField>>,
    ) -> Result<Vec<Contribution>> {
        // A constraint only enters the global (stiffness) matrix — no
        // mass/geometric/tangent term.
        if kind != MatrixKind::Stiffness {
            return Ok(Vec::new());
        }
        // One C / Cᵀ pair per submesh pair — the user's submeshes are used
        // directly as row/col supports (no flattening), so the submesh structure
        // carried through `barycenter` is preserved. Dirichlet is the single-term
        // relation with coefficient 1, so it defers to the shared block builder.
        let mut blocks = Vec::with_capacity(self.imposed_mesh.len() * 2);
        for i in 0..self.imposed_mesh.len() {
            let imp_sm = self.imposed_mesh.get(i)?;
            let mult_sm = self.multiplier_mesh.get(i)?;
            let (c_block, ct_block) = constraint_block_pair(
                &mult_sm,
                &imp_sm,
                &self.imposed_variable,
                &self.target_dual,
                &self.multiplier,
                &self.imposed_value,
                1.0,
            )?;
            blocks.push(c_block);
            blocks.push(ct_block);
        }
        Ok(vec![Contribution::Literal(blocks)])
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Constraint]
    }

    fn label(&self) -> &'static str {
        "Dirichlet"
    }

    fn display(&self) -> String {
        let n = self.imposed_mesh.cell_count();
        format!(
            "SubModel<Dirichlet({})>: {} constrained node(s)",
            self.imposed_variable, n
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let nc = self.imposed_mesh.cell_count();
        let nm = self.multiplier_mesh.cell_count();
        format!(
            "SubModel<Dirichlet({imposed_variable})>\n  primal var(s): {multiplier} \
             (multiplier)\n  dual var(s):   {imposed_value} (imposed value)\n  \
             targets primary dual: {target_dual}\n  sense: {sense}\n  \
             constrained: {nc} node(s)\n  multipliers: {nm} node(s)",
            imposed_variable = self.imposed_variable,
            multiplier = self.multiplier,
            imposed_value = self.imposed_value,
            target_dual = self.target_dual,
            sense = self.sense,
        )
    }
}

impl Constraint for Dirichlet {
    fn multiplier_mesh(&self) -> &Mesh {
        &self.multiplier_mesh
    }

    /// Dirichlet is a single-term relation `1·u ⋈ u_d` per constrained node
    /// (`⋈` is the sub-model's sense, `=` by default).
    fn relations(&self) -> Result<Vec<Relation>> {
        let mut relations = Vec::with_capacity(self.imposed_mesh.cell_count());
        for i in 0..self.imposed_mesh.len() {
            let imposed_nodes: Vec<NodeId> =
                self.imposed_mesh.get(i)?.read().connectivity().to_vec();
            let multiplier_nodes: Vec<NodeId> =
                self.multiplier_mesh.get(i)?.read().connectivity().to_vec();
            for (imp, mult) in imposed_nodes.iter().zip(multiplier_nodes.iter()) {
                relations.push(Relation {
                    multiplier_node: *mult,
                    imposed_value: self.imposed_value.clone(),
                    terms: vec![ConstraintTerm {
                        node: *imp,
                        variable: self.imposed_variable.clone(),
                        target_dual: self.target_dual.clone(),
                        coefficient: 1.0,
                    }],
                    sense: self.sense,
                });
            }
        }
        Ok(relations)
    }
}

/// Clone a mesh by sharing its submeshes — increfs each submesh handle so the
/// nodes stay alive for the lifetime of the sub-model.
fn share(mesh: &Mesh) -> Result<Mesh> {
    let mut out = Mesh::empty();
    for sm in mesh {
        out.add_sub(sm.clone())?;
    }
    Ok(out)
}
