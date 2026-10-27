//! Dirichlet constraint via Lagrange multipliers.
//!
//! A Dirichlet sub-model imposes `u = u_d` on a set of constrained nodes. It
//! is a *constraint*, not a volumetric physics: it carries no material and no
//! constitutive law. It is built from **one mesh supplied by the user**,
//! `imposed_mesh` — POI1 (for now): the constrained nodes (shared with the
//! target physics), one node per cell, one relation per cell.
//!
//! Each relation gets a multiplier node of its own, minted at construction by
//! the [`barycenter`](crate::ops::mesh::barycenter()) mesher — colocated with
//! its constrained node, in the same submesh structure. The user reaches them
//! through [`Constraint::multiplier_mesh`], or never touches them at all:
//! [`SubModel::constraint_rhs`](crate::containers::model::SubModel::constraint_rhs)
//! writes `u_d` keyed by the constrained node.
//!
//! # Variables
//!
//! A Dirichlet sub-model imposes the **primal of another model** yet owns its
//! own pair of variables:
//!
//! - `imposed_variable` (e.g. `"T"`) — the constrained primal of the **target**
//!   physics (a *column* of the target's stiffness);
//! - `target_dual` (e.g. `"q"`) — the **target's** dual variable; the row into
//!   which the constraint reaction `Cᵀ` is added (it cannot be derived from
//!   `imposed_variable` — the `T → q` map is target-specific);
//! - [`MULTIPLIER`] — this sub-model's own **primal**: the Lagrange
//!   multiplier, an unknown of the augmented system, whose solved value is the
//!   **reaction**;
//! - [`IMPOSED`] — this sub-model's own **dual**: the constraint-equation row,
//!   and the **slot** at which the imposed value `u_d` is written in the load
//!   `SubNodeField` (on the multiplier node).
//!
//! The last two are the names of every constraint relation.

use crate::aggregate::Aggregate;
use crate::atoms::ElementType;
use crate::containers::element_field::SubElementField;
use crate::containers::mesh::Mesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::transfer::{constrained_physics, constraint_physics};
use crate::models::{
    constraint_block_pair, Constraint, ConstraintTerm, Contribution, MatrixKind, Physics, Relation,
    RelationSense, SubModelKind, IMPOSED, MULTIPLIER,
};
use crate::ops::mesh::barycenter;
use serde::{Deserialize, Serialize};

/// Dirichlet constraint imposed via Lagrange multipliers.
///
/// See the module documentation for the meaning of the variable names and the
/// two meshes. The imposed value `u_d` is **not** stored here: the user
/// supplies it through the load `SubNodeField` at the multiplier node's
/// [`IMPOSED`] component.
///
/// ```
/// # use pyrucast::models::symmetry::MaterialSymmetry;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::{Constraint, RelationSense, SubModelKind};
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
/// # use pyrucast::models::dirichlet::Dirichlet;
/// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
/// // A support: `u = u_d` at the imposed nodes, through a Lagrange multiplier.
/// let d = Dirichlet::new(&cible, "T", &impose, RelationSense::Equality)?;
/// assert_eq!(d.relations()?.len(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize)]
pub struct Dirichlet {
    /// Constrained primal of the target physics (e.g. `"T"`).
    pub(crate) imposed_variable: String,
    /// Target physics's dual; row where the reaction `Cᵀ` lands (e.g. `"q"`).
    pub(crate) target_dual: String,
    /// POI1 mesh of the constrained nodes (one node per cell), as given by the
    /// user. Shares the submeshes (refcounts keep the nodes alive).
    pub(crate) imposed_mesh: Mesh,
    /// POI1 mesh of the multiplier nodes, minted at construction and paired
    /// element-for-element with `imposed_mesh` (same submesh structure, same
    /// per-submesh cell count).
    pub(crate) multiplier_mesh: Mesh,
    /// Equality (default) or unilateral inequality (`u ≥ u_d` / `u ≤ u_d`),
    /// solved by the active-set operator.
    #[serde(default)]
    pub(crate) sense: RelationSense,
    /// Nature of the physics it constrains (`Mechanical` for a `u_x`), read
    /// from the target at construction.
    pub(crate) target_physics: Option<Physics>,
}

impl Dirichlet {
    /// Build a Dirichlet constraint imposing `variable = u_d` on the nodes of
    /// `imposed_mesh`, one relation per node, each with a multiplier node of
    /// its own minted beside it.
    ///
    /// `variable` is a primal of `target`, whose dual row (e.g. `"q"`) is read
    /// from it. `imposed_mesh` must be POI1 (for now).
    ///
    /// `sense` turns the constraint unilateral: `GreaterEqual` imposes
    /// `u ≥ u_d` (`LessEqual`: `u ≤ u_d`), enforced only while in contact — such
    /// a model is solved by the active-set operator
    /// [`unilateral`](crate::ops::solver::unilateral).
    ///
    /// Errors when `variable` is not a primal of `target`, when `imposed_mesh`
    /// holds no node, or when one of its submeshes is not POI1.
    ///
    /// ```
    /// # use pyrucast::models::symmetry::MaterialSymmetry;
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::{Constraint, RelationSense, SubModelKind};
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
    /// # let impose = mesh::poi1_from_nodes(&n[..1]).unwrap();
    /// # use pyrucast::models::dirichlet::Dirichlet;
    /// # let cible = pyrucast::ops::model::heat_conduction(&fes, MaterialSymmetry::Isotropic).unwrap();
    /// let d = Dirichlet::new(&cible, "T", &impose, RelationSense::Equality)?;
    /// // The multiplier node is the constraint's own, minted beside the
    /// // imposed node.
    /// let m = d.multiplier_mesh();
    /// assert_eq!(m.cell_count(), 1);
    /// assert_ne!(m.node(0, 0, 0)?.id(), n[0].id());
    /// assert_eq!(m.node(0, 0, 0)?.position()?, vec![0.0, 0.0]);
    /// // A variable the target does not assemble is refused.
    /// assert!(Dirichlet::new(&cible, "u_x", &impose, RelationSense::Equality).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(
        target: &crate::containers::model::Model,
        variable: &str,
        imposed_mesh: &Mesh,
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
        for (i, imp) in imposed_mesh.into_iter().enumerate() {
            let iet = imp.read().element_type();
            if iet != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: imposed_mesh submesh {i} must be POI1, got {iet}"
                )));
            }
        }
        // One fresh node per relation, beside its constrained node.
        let multiplier_mesh = barycenter(imposed_mesh)?;
        let target_physics = constrained_physics(target, [imposed_variable.as_str()]);

        Ok(Self {
            imposed_variable,
            target_dual,
            imposed_mesh: share(imposed_mesh)?,
            multiplier_mesh,
            sense,
            target_physics,
        })
    }
}

impl SubModelKind for Dirichlet {
    fn primal_vars(&self) -> Vec<String> {
        vec![MULTIPLIER.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![IMPOSED.to_string()]
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
        // A constraint only enters the global (stiffness) matrix — no mass or
        // geometric term. A tangent request reaches it as a stiffness one
        // (`tangent_is_stiffness`), so it never arrives here.
        if kind != MatrixKind::Stiffness {
            return Ok(Vec::new());
        }
        // One C / Cᵀ pair per submesh pair — the submeshes are used directly as
        // row/col supports (no flattening), so the submesh structure carried
        // through `barycenter` is preserved. Dirichlet is the single-term
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
                1.0,
            )?;
            blocks.push(c_block);
            blocks.push(ct_block);
        }
        Ok(vec![Contribution::Literal(blocks)])
    }

    /// A linear relation: its tangent block **is** its stiffness block.
    fn tangent_is_stiffness(&self) -> bool {
        true
    }

    fn physics(&self) -> &'static [Physics] {
        constraint_physics(self.target_physics)
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
            "SubModel<Dirichlet({imposed_variable})>\n  primal var(s): {MULTIPLIER} \
             (multiplier)\n  dual var(s):   {IMPOSED} (imposed value)\n  \
             targets primary dual: {target_dual}\n  sense: {sense}\n  \
             constrained: {nc} node(s)\n  multipliers: {nm} node(s)",
            imposed_variable = self.imposed_variable,
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
            // Read in place: the loop below only copies ids out by value and
            // clones strings, it never re-locks either submesh.
            let imposed = self.imposed_mesh.get(i)?.read();
            let multiplier = self.multiplier_mesh.get(i)?.read();
            for (imp, mult) in imposed
                .connectivity()
                .iter()
                .zip(multiplier.connectivity().iter())
            {
                relations.push(Relation {
                    multiplier_node: *mult,
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
