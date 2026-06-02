//! Physical model — orchestrator of sub-models that assemble into a
//! [`Matrix`].
//!
//! The model layer is the **physics-aware** counterpart of the
//! geometry layer (`Mesh`, `SubMesh`) and the interpolation layer
//! (`FiniteElementSpace`, `SubFiniteElementSpace`). A [`Model`] is an aggregate of
//! [`SubModel`]s, each binding **one or more FE spaces** to a
//! [`Physics`] (the actual law). The Model is a pure orchestrator: it
//! enumerates the DOFs of its sub-models, dimensions a [`Matrix`], and
//! loops over the sub-models to accumulate the contributions.
//!
//! # Architecture
//!
//! ```text
//! Model
//! ├── sub_models: Vec<Handle<SubModel>>
//! ├── primal_vars(): Vec<String>      # union over sub-models — columns
//! └── dual_vars():   Vec<String>      # union over sub-models — rows
//!
//! ops::assemble (operators, not Model methods)
//! ├── stiffness(model, materials) -> Matrix   # rows: dual × cols: primal
//! └── mass(model)                 -> Matrix   # same DOF layout, may be empty
//!
//! SubModel
//! └── physics: Physics
//!
//! Physics  (enum à variantes spécialisées)
//! ├── HeatConduction { fespace, material }
//! ├── Dirichlet     { ... }            # constraint = Lagrange multiplier
//! └── ...
//! ```
//!
//! The model layer is purely matrix-producing. Loads (right-hand side
//! vectors) are entirely the user's responsibility: read
//! `model.dual_vars()`, build a [`crate::containers::node_field::NodeField`] with
//! the matching component names, and feed `Matrix + NodeField` to the
//! solver.
//!
//! # Lagrange multipliers and DOF identification
//!
//! `Physics::Dirichlet` introduces new DOFs of two kinds, both living
//! on **multiplier nodes** that the sub-model creates on the fly in the
//! [`Configuration`]:
//!
//! - the **primal** of the constraint sub-model is `lambda_<var>` at the
//!   multiplier nodes — the Lagrange multiplier itself, an unknown of
//!   the augmented system;
//! - the **dual** of the constraint sub-model is `<var>` (the same
//!   string as the primal being constrained) at the multiplier nodes —
//!   the constraint equation row, in units of the primary variable.
//!
//! Different `(NodeId, field_name)` pairs distinguish the multiplier
//! DOFs from the primary DOFs even when the field names happen to
//! collide. The Matrix's symmetric flag is purely informative; this
//! v0 stores both the `C` (constraint) and `Cᵀ` (its transpose) blocks
//! explicitly so the dense LU solver gets a well-posed system without
//! relying on the symmetry contract.
//!
//! # Example: 1-D heat conduction with a Dirichlet condition
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::containers::mesh::{Configuration, NodeId};
//! use pyrucast::containers::element_field::SubElementField;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::model::{Model, Physics, SubModel};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::ops::assemble;
//! use pyrucast::store::{insert, with};
//!
//! // 1-D Configuration with two nodes spanning [0, 1].
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.subspace(0).unwrap();
//!
//! // Conductivity k = 1, uniform — passed at assembly time, not stored in the model.
//! let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
//! mat.set_uniform("k", 1.0).unwrap();
//! use pyrucast::containers::element_field::ElementField;
//! let mut materials = ElementField::empty();
//! materials.add_sub(insert(mat)).unwrap();
//!
//! let mut model = Model::empty();
//! model
//!     .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
//!     .unwrap();
//! model
//!     .add_sub(insert(
//!         SubModel::dirichlet("T".into(), "q".into(), std::slice::from_ref(&a)).unwrap(),
//!     ))
//!     .unwrap();
//!
//! let k = assemble::stiffness(&model, &materials).unwrap();
//! // 2 real DOFs ("T") + 1 multiplier DOF + 2 real rows ("q") + 1 multiplier row.
//! assert_eq!(k.n_rows().unwrap(), 3);
//! assert_eq!(k.n_cols().unwrap(), 3);
//! ```

use crate::containers::mesh::{Node, NodeId};
use crate::containers::element_field::SubElementField;
use crate::error::Result;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::aggregate::Aggregate;
use crate::models::{dirichlet, heat_conduction};
use crate::store::{insert, with, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Physics ───────────────────────────────────────────────────────────────

/// One physical law instance, bound to its supports (FE spaces, materials,
/// node sets).
///
/// New physics are added by extending this enum. Each variant must be
/// supported by the assembly dispatch in [`Model::stiffness`] /
/// [`Model::mass`].
#[derive(Clone, Serialize, Deserialize)]
pub enum Physics {
    /// Linear heat conduction.
    ///
    /// - primal variable: `"T"` (temperature, columns).
    /// - dual variable:   `"q"` (heat flux row labels).
    /// - Material data (conductivity `"k"`, …) is **not** stored here;
    ///   it is supplied at assembly time via [`crate::ops::assemble::stiffness`].
    HeatConduction {
        fespace: Handle<SubFiniteElementSpace>,
        /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh,
        /// built once at construction. Reused as the row/col support of
        /// every assembled stiffness block — no per-assembly rebuild.
        support: Handle<SubMesh>,
    },

    /// Dirichlet constraint imposed via Lagrange multipliers.
    ///
    /// Conceptually: for each constrained primary node `n`, the system
    /// is augmented with one multiplier `λ_n` (a new node introduced
    /// on the fly at the same coordinates as `n`) and a pair of unit
    /// entries enforcing `u_n = u_d_n`. The imposed value `u_d_n` is
    /// **not** part of this enum: the user supplies it through the
    /// load `NodeField` at the multiplier node's `<var>` component.
    ///
    /// - primal variable: `"lambda_<primal_var>"` (multiplier DOFs,
    ///   added to the global column set on `multiplier_nodes`).
    /// - dual variable:   `<primal_var>` itself (constraint equation
    ///   row labels, added to the global row set on
    ///   `multiplier_nodes`).
    /// - `primal_dual` is the dual variable name of the **primary
    ///   physics** that this constraint targets (`"q"` for heat
    ///   conduction, `"f_x"` for elasticity in `x`, …): it tells the
    ///   constraint where in the row index to write the `Cᵀ` block.
    Dirichlet {
        primal_var: String,
        primal_dual: String,
        /// POI1 SubMesh holding the per-node refcounts on the constrained
        /// nodes. Connectivity is the constrained-node sequence (immutable
        /// once inserted into the store).
        constrained_support: Handle<SubMesh>,
        /// POI1 SubMesh owning the multiplier nodes (one cell per
        /// multiplier, in the same order as the constrained nodes). The
        /// SubMesh holds the only refcount on each multiplier; its
        /// `Drop` collects them.
        multiplier_support: Handle<SubMesh>,
    },
}

impl Physics {
    /// Primal variable names introduced by this physics (column labels of
    /// the Matrix block contributed by this physics).
    pub fn primal_vars(&self) -> Vec<String> {
        match self {
            Physics::HeatConduction { .. } => vec![heat_conduction::PRIMAL_VAR.to_string()],
            Physics::Dirichlet { primal_var, .. } => vec![dirichlet::multiplier_name(primal_var)],
        }
    }

    /// Dual variable names introduced by this physics (row labels of the
    /// Matrix block contributed by this physics).
    pub fn dual_vars(&self) -> Vec<String> {
        match self {
            Physics::HeatConduction { .. } => vec![heat_conduction::DUAL_VAR.to_string()],
            Physics::Dirichlet { primal_var, .. } => vec![primal_var.clone()],
        }
    }

    /// Names of the material components this physics expects in its
    /// material [`SubElementField`], or `None` if it doesn't need any
    /// material data (e.g. `Dirichlet`).
    ///
    /// The list is the **contract** between the physics and any material
    /// provider: [`SubModel::build_material_field`] uses it to know what
    /// to create, and [`crate::ops::assemble::stiffness`] uses it to
    /// validate the supplied material early, before the per-cell loop.
    pub fn material_components(&self) -> Option<&'static [&'static str]> {
        const HC_COMPS: &[&str] = &[heat_conduction::MATERIAL_COMPONENT];
        match self {
            Physics::HeatConduction { .. } => Some(HC_COMPS),
            Physics::Dirichlet { .. } => None,
        }
    }
}

// ─── SubModel ──────────────────────────────────────────────────────────────

/// One physics + its support binding. A [`Model`] is a `Vec<Handle<SubModel>>`.
#[derive(Serialize, Deserialize)]
pub struct SubModel {
    physics: Physics,
}

impl SubModel {
    /// Wrap an existing `Physics` instance into a `SubModel`.
    pub fn new(physics: Physics) -> Self {
        Self { physics }
    }

    /// Heat-conduction sub-model on an FE subspace.
    ///
    /// Material data (conductivity `"k"`, …) is supplied separately at
    /// assembly time via [`crate::ops::assemble::stiffness`], keeping
    /// the model immutable and material-independent.
    ///
    /// A stable POI1 [`SubMesh`] covering the unique nodes of the FE
    /// subspace is built once and stored — reused as the row/col support
    /// of every assembled stiffness block.
    pub fn heat_conduction(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let submesh = with(&fespace, |s| s.submesh())?;
        let (cfg, flat_conn) = with(&submesh, |s| {
            (s.configuration(), s.connectivity().to_vec())
        })?;
        let mut unique_nodes: Vec<NodeId> = Vec::new();
        for &nid in &flat_conn {
            if !unique_nodes.contains(&nid) {
                unique_nodes.push(nid);
            }
        }
        let mut poi1 = SubMesh::new(cfg, ElementType::POI1);
        for &nid in &unique_nodes {
            poi1.add_cell(&[nid])?;
        }
        let support = insert(poi1);
        Ok(Self {
            physics: Physics::HeatConduction { fespace, support },
        })
    }

    /// Dirichlet sub-model: enforce `<primal_var> = u_d` on each
    /// `constrained_node`, with `u_d` supplied later by the user
    /// through the load `NodeField`.
    ///
    /// `primal_dual` is the dual variable name of the primary physics
    /// whose primal is being constrained (e.g. `"q"` for heat
    /// conduction, `"f_x"` for elasticity in `x`).
    ///
    /// One new node per constraint is added to the `Configuration` at
    /// the same coordinates as the constrained node — these are the
    /// multiplier nodes that carry the `lambda_<primal_var>` and
    /// `<primal_var>` DOFs of the constraint sub-model.
    pub fn dirichlet(
        primal_var: String,
        primal_dual: String,
        constrained_nodes: &[Node],
    ) -> Result<Self> {
        let built = dirichlet::build(constrained_nodes)?;
        Ok(Self {
            physics: Physics::Dirichlet {
                primal_var,
                primal_dual,
                constrained_support: built.constrained_support,
                multiplier_support: built.multiplier_support,
            },
        })
    }

    /// The physics carried by this sub-model.
    pub fn physics(&self) -> &Physics {
        &self.physics
    }

    /// Multiplier node ids introduced by this sub-model. Non-empty only
    /// for Lagrange variants (`Dirichlet`, future `MultipointConstraint`,
    /// …); empty for the other physics.
    ///
    /// Useful for the user who needs to write the imposed value `u_d` at
    /// the multiplier node's `<primal_var>` component of the load
    /// `NodeField`.
    pub fn multiplier_nodes(&self) -> Result<Vec<NodeId>> {
        match &self.physics {
            Physics::Dirichlet { multiplier_support, .. } => {
                with(multiplier_support, |s| s.connectivity().to_vec())
            }
            _ => Ok(Vec::new()),
        }
    }

    /// POI1 [`Mesh`] of the multiplier nodes (shares the multiplier
    /// support submesh — zero-copy). Empty for non-Lagrange physics.
    ///
    /// This is the user-facing handle to the multiplier nodes: build a
    /// load [`crate::containers::node_field::NodeField`] on its single
    /// submesh to impose the constrained values.
    pub fn multiplier_mesh(&self) -> Result<Mesh> {
        let mut mesh = Mesh::empty();
        if let Physics::Dirichlet { multiplier_support, .. } = &self.physics {
            mesh.add_sub(multiplier_support.clone())?;
        }
        Ok(mesh)
    }

    /// FE subspace on which this sub-model expects its material data, or
    /// `None` if this physics doesn't need material data (e.g. `Dirichlet`).
    pub fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        match &self.physics {
            Physics::HeatConduction { fespace, .. } => Some(fespace.clone()),
            Physics::Dirichlet { .. } => None,
        }
    }

    /// Material component names this sub-model expects, or `None` if it
    /// doesn't need material data. Thin pass-through of
    /// [`Physics::material_components`].
    pub fn material_components(&self) -> Option<&'static [&'static str]> {
        self.physics.material_components()
    }

    /// Primal variable names introduced by this sub-model.
    pub fn primal_vars(&self) -> Vec<String> {
        self.physics.primal_vars()
    }

    /// Dual variable names introduced by this sub-model.
    pub fn dual_vars(&self) -> Vec<String> {
        self.physics.dual_vars()
    }

    /// Build and fill the stiffness [`SubMatrix`] block(s) for this sub-model.
    ///
    /// - `HeatConduction` → 1 block (square, symmetric). `material` must be
    ///   `Some(_)` — the caller ([`crate::ops::assemble::stiffness`]) always
    ///   provides it.
    /// - `Dirichlet`      → 2 blocks: the C block and the Cᵀ block.
    ///   `material` is ignored.
    pub(crate) fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        match &self.physics {
            Physics::HeatConduction { fespace, support } => {
                let mat = material.expect("HeatConduction requires a material field");
                let mut block = SubMatrix::new(
                    support.clone(), support.clone(),
                    vec![heat_conduction::DUAL_VAR.to_string()],
                    vec![heat_conduction::PRIMAL_VAR.to_string()],
                    DofOrdering::NodesThenVars, true,
                )?;
                heat_conduction::assemble_stiffness(fespace, mat, &mut block)?;
                Ok(vec![block])
            }
            Physics::Dirichlet {
                primal_var, primal_dual,
                constrained_support, multiplier_support,
            } => {
                let _ = material;
                let constrained_nodes: Vec<NodeId> =
                    with(constrained_support, |s| s.connectivity().to_vec())?;
                let multiplier_nodes: Vec<NodeId> =
                    with(multiplier_support, |s| s.connectivity().to_vec())?;
                let lambda_name = dirichlet::multiplier_name(primal_var);
                // C block: rows = multiplier × primal_var, cols = constrained × primal_var
                let mut c_block = SubMatrix::new(
                    multiplier_support.clone(), constrained_support.clone(),
                    vec![primal_var.clone()],
                    vec![primal_var.clone()],
                    DofOrdering::NodesThenVars, true,
                )?;
                // Cᵀ block: rows = constrained × primal_dual, cols = multiplier × lambda
                let mut ct_block = SubMatrix::new(
                    constrained_support.clone(), multiplier_support.clone(),
                    vec![primal_dual.clone()],
                    vec![lambda_name],
                    DofOrdering::NodesThenVars, true,
                )?;
                dirichlet::assemble_blocks(
                    &constrained_nodes, &multiplier_nodes,
                    primal_var, primal_dual,
                    &mut c_block, &mut ct_block,
                )?;
                Ok(vec![c_block, ct_block])
            }
        }
    }
}

impl fmt::Debug for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubModel")
            .field(
                "physics",
                &match &self.physics {
                    Physics::HeatConduction { .. } => "HeatConduction",
                    Physics::Dirichlet { .. }    => "Dirichlet",
                },
            )
            .finish()
    }
}

impl fmt::Display for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.physics {
            Physics::HeatConduction { .. } => write!(f, "SubModel<HeatConduction>"),
            Physics::Dirichlet {
                primal_var,
                constrained_support,
                ..
            } => {
                let n = with(constrained_support, |s| s.cell_count())
                    .unwrap_or(0);
                write!(
                    f,
                    "SubModel<Dirichlet({})>: {} constrained node(s)",
                    primal_var, n
                )
            }
        }
    }
}

impl crate::dump::Dump for SubModel {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        let primal = self.physics.primal_vars().join(", ");
        let dual = self.physics.dual_vars().join(", ");
        match &self.physics {
            Physics::HeatConduction { support, .. } => {
                let n = with(support, |s| s.cell_count()).unwrap_or(0);
                format!(
                    "SubModel<HeatConduction>\n  primal var(s): {primal}\n  \
                     dual var(s):   {dual}\n  support: {n} node(s)"
                )
            }
            Physics::Dirichlet {
                primal_var,
                primal_dual,
                constrained_support,
                multiplier_support,
            } => {
                let nc = with(constrained_support, |s| s.cell_count()).unwrap_or(0);
                let nm = with(multiplier_support, |s| s.cell_count()).unwrap_or(0);
                format!(
                    "SubModel<Dirichlet({primal_var})>\n  primal var(s): {primal} (multipliers)\n  \
                     dual var(s):   {dual}\n  targets primary dual: {primal_dual}\n  \
                     constrained: {nc} node(s)\n  multipliers: {nm} node(s)"
                )
            }
        }
    }
}

// ─── Model ─────────────────────────────────────────────────────────────────

/// Aggregate of sub-models. Produces matrices on explicit demand.
///
/// Internally a `Vec<Handle<SubModel>>` — see [`Aggregate`]. The Handle
/// refcount keeps each sub-model alive as long as any `Model` (or
/// `PySubModel`) references it; dropping the last reference triggers the
/// sub-model's `Drop` (which releases the Lagrange-multiplier nodes for
/// Dirichlet sub-models).
#[derive(Serialize, Deserialize, Default)]
pub struct Model {
    subs: Vec<Handle<SubModel>>,
}

crate::impl_aggregate!(Model, SubModel, sub_model, "sub-model(s)");
crate::impl_aggregate_dump!(Model);

impl Model {
    /// Heat-conduction `Model` spanning **every** subspace of `fes` — one
    /// [`Physics::HeatConduction`] sub-model per [`SubFiniteElementSpace`].
    ///
    /// This is the parent-level named constructor (see `CONVENTIONS.md`):
    /// it consumes the FE-space *parent* and returns a `Model`, so the
    /// caller never builds a `SubModel` by hand. A single-subspace `fes`
    /// yields the unit case; several subspaces yield one zone each.
    /// Compose heterogeneous physics with `+` (merge), e.g.
    /// `Model::heat_conduction(&fes)? + Model::dirichlet(...)?`.
    pub fn heat_conduction(fes: &FiniteElementSpace) -> Result<Self> {
        let mut model = Self::empty();
        for sub in fes {
            model.add_sub(insert(SubModel::heat_conduction(sub.clone())?))?;
        }
        Ok(model)
    }

    /// Dirichlet `Model` (a single sub-model) constraining `<primal_var>`
    /// on `constrained_nodes` via Lagrange multipliers. Parent-level
    /// named constructor — see [`SubModel::dirichlet`] for the semantics
    /// of `primal_var` / `primal_dual`.
    pub fn dirichlet(
        primal_var: String,
        primal_dual: String,
        constrained_nodes: &[Node],
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::dirichlet(
            primal_var,
            primal_dual,
            constrained_nodes,
        )?))?;
        Ok(model)
    }

    /// Primal variable names — union over all sub-models, first-seen order.
    /// These are the **column labels** of the assembled matrices and the
    /// component names of the solution `NodeField`.
    pub fn primal_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in self {
            all.extend(with(h, |s| s.primal_vars())?);
        }
        Ok(union_names(all))
    }

    /// Dual variable names — union over all sub-models, first-seen order.
    /// These are the **row labels** of the assembled matrices and the
    /// component names of the load `NodeField`.
    pub fn dual_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in self {
            all.extend(with(h, |s| s.dual_vars())?);
        }
        Ok(union_names(all))
    }

}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn union_names<I: IntoIterator<Item = String>>(iter: I) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in iter {
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::containers::mesh::Node;
    use crate::ops::assemble;
    use crate::store::{insert, with_mut};

    /// Returns `(cfg, a_id, b_id, model, materials)`.
    fn build_seg2_heat_model(
        length: f64,
        k: f64,
        dirichlet_at_left: bool,
    ) -> (Handle<Configuration>, NodeId, NodeId, Model, ElementField) {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();

        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", k).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        if dirichlet_at_left {
            model
                .add_sub(insert(
                    SubModel::dirichlet("T".into(), "q".into(), std::slice::from_ref(&a))
                        .unwrap(),
                ))
                .unwrap();
        }
        (cfg, a.id(), b.id(), model, materials)
    }

    #[test]
    fn primal_dual_vars_for_heat_conduction_alone() {
        let (_cfg, _, _, model, _mat) = build_seg2_heat_model(1.0, 1.0, false);
        assert_eq!(model.primal_vars().unwrap(), vec!["T".to_string()]);
        assert_eq!(model.dual_vars().unwrap(), vec!["q".to_string()]);
    }

    #[test]
    fn primal_dual_vars_include_lagrange_after_dirichlet() {
        let (_cfg, _, _, model, _mat) = build_seg2_heat_model(1.0, 1.0, true);
        assert_eq!(
            model.primal_vars().unwrap(),
            vec!["T".to_string(), "lambda_T".to_string()]
        );
        // Dual side: "q" from heat conduction + "T" (dual of Dirichlet,
        // same string but on different (NodeId, name) pairs).
        assert_eq!(
            model.dual_vars().unwrap(),
            vec!["q".to_string(), "T".to_string()]
        );
    }

    /// Heat conduction on `[0, L]` with one SEG2 of length L and k = 1:
    /// `K_local = (k/L) [[1, -1], [-1, 1]]` (analytical, see Hughes 1.4).
    #[test]
    fn heat_conduction_assembles_analytical_seg2_stiffness() {
        let length = 2.0;
        let k_val = 1.5;
        let (_cfg, a_id, b_id, model, materials) = build_seg2_heat_model(length, k_val, false);
        let k = assemble::stiffness(&model, &materials).unwrap();

        assert_eq!(k.n_rows().unwrap(), 2);
        assert_eq!(k.n_cols().unwrap(), 2);
        let expected = k_val / length;
        let tol = 1e-12;
        assert!((k.get(a_id, "q", a_id, "T").unwrap() - expected).abs() < tol);
        assert!((k.get(a_id, "q", b_id, "T").unwrap() + expected).abs() < tol);
        assert!((k.get(b_id, "q", a_id, "T").unwrap() + expected).abs() < tol);
        assert!((k.get(b_id, "q", b_id, "T").unwrap() - expected).abs() < tol);
    }

    /// Two SEG2 elements sharing the middle node form the classic
    /// tridiagonal `(1, -2, 1)` pattern after assembly.
    #[test]
    fn two_seg2_assembly_is_tridiagonal() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let k = assemble::stiffness(&model, &materials).unwrap();
        assert_eq!(k.n_rows().unwrap(), 3);
        assert_eq!(k.n_cols().unwrap(), 3);

        let v = |i: NodeId, j: NodeId| k.get(i, "q", j, "T").unwrap();
        let tol = 1e-12;
        // h = 1 ⇒ K_global = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]].
        assert!((v(n0.id(), n0.id()) - 1.0).abs() < tol);
        assert!((v(n0.id(), n1.id()) + 1.0).abs() < tol);
        assert!((v(n0.id(), n2.id())).abs() < tol);
        assert!((v(n1.id(), n0.id()) + 1.0).abs() < tol);
        assert!((v(n1.id(), n1.id()) - 2.0).abs() < tol);
        assert!((v(n1.id(), n2.id()) + 1.0).abs() < tol);
        assert!((v(n2.id(), n0.id())).abs() < tol);
        assert!((v(n2.id(), n1.id()) + 1.0).abs() < tol);
        assert!((v(n2.id(), n2.id()) - 1.0).abs() < tol);
    }

    /// Dirichlet on the left node creates a multiplier node and writes
    /// both `C` and `Cᵀ` entries (each value 1.0).
    #[test]
    fn dirichlet_adds_one_multiplier_node_and_two_block_entries() {
        let (cfg, a_id, _b_id, model, materials) = build_seg2_heat_model(1.0, 1.0, true);

        // The Configuration grew by one node (the multiplier).
        let n_nodes = with(&cfg, |c| c.node_count()).unwrap();
        assert_eq!(n_nodes, 3);

        let k = assemble::stiffness(&model, &materials).unwrap();
        // 2 real "q" rows + 1 multiplier "T" row = 3 rows.
        // 2 real "T" cols + 1 multiplier "lambda_T" col = 3 cols.
        assert_eq!(k.n_rows().unwrap(), 3);
        assert_eq!(k.n_cols().unwrap(), 3);

        // Find the multiplier node id: the only NodeId that appears in
        // a row labelled "T" of K.
        let row_dofs = k.row_dofs().unwrap();
        let mult = row_dofs
            .iter()
            .find(|(_, name)| name == "T")
            .expect("multiplier row missing")
            .0;

        // C entry: (mult, "T") × (a_id, "T") = 1
        assert_eq!(k.get(mult, "T", a_id, "T").unwrap(), 1.0);
        // Cᵀ entry: (a_id, "q") × (mult, "lambda_T") = 1
        assert_eq!(k.get(a_id, "q", mult, "lambda_T").unwrap(), 1.0);
        // Ensure lambda_T appears as a column.
        let col_dofs = k.col_dofs().unwrap();
        let lambda_col_present = col_dofs
            .iter()
            .any(|(n, name)| name == "lambda_T" && *n == mult);
        assert!(lambda_col_present);
    }

    /// SubModel Drop on Dirichlet decrements the refcounts it took (one
    /// on each constrained node, one on each multiplier node).
    #[test]
    fn dropping_dirichlet_releases_node_refcounts() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let a_id = a.id();

        // Before adding the sub-model: the Node holds 1 ref.
        with(&cfg, |c| assert_eq!(c.refcount(a_id), 1)).unwrap();

        let sub = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            std::slice::from_ref(&a),
        )
        .unwrap();

        // After: the Node + the SubModel each hold 1 ref ⇒ 2.
        with(&cfg, |c| assert_eq!(c.refcount(a_id), 2)).unwrap();
        // The multiplier node has refcount 1 (owned by the sub-model).
        let mult_id = sub.multiplier_nodes().unwrap()[0];
        with(&cfg, |c| assert_eq!(c.refcount(mult_id), 1)).unwrap();

        drop(sub);
        // Now back to 1 (only the Node remains).
        with(&cfg, |c| assert_eq!(c.refcount(a_id), 1)).unwrap();
        // And the multiplier is collectable.
        with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
    }

    #[test]
    fn dirichlet_empty_constraint_list_rejected() {
        assert!(SubModel::dirichlet("T".into(), "q".into(), &[]).is_err());
    }

    #[test]
    fn heat_conduction_errors_on_missing_k_component() {
        // Material has only "rho_cp", not "k": stiffness assembly must fail.
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let mat = SubElementField::new(sub.clone(), vec!["rho_cp".into()]).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        assert!(assemble::stiffness(&model, &materials).is_err());
    }

    #[test]
    fn empty_model_has_no_vars_and_empty_mass() {
        let model = Model::empty();
        assert_eq!(model.primal_vars().unwrap(), Vec::<String>::new());
        assert_eq!(model.dual_vars().unwrap(), Vec::<String>::new());
        let m = assemble::mass(&model).unwrap();
        assert_eq!(m.n_rows().unwrap(), 0);
        assert_eq!(m.n_cols().unwrap(), 0);
    }

    /// Parent-level `Model::heat_conduction(&fes)` builds one sub-model per
    /// subspace and matches the hand-rolled `SubModel + add_sub` path, and
    /// `+` (merge) composes it with a Dirichlet `Model`.
    #[test]
    fn parent_constructors_span_subspaces_and_compose() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        // Two SEG2 zones, one SubMesh each → fes with two subspaces.
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // One HeatConduction sub-model per subspace.
        let hc = Model::heat_conduction(&fes).unwrap();
        assert_eq!(hc.sub_model_count(), 2);
        assert_eq!(hc.primal_vars().unwrap(), vec!["T".to_string()]);
        assert_eq!(hc.dual_vars().unwrap(), vec!["q".to_string()]);

        // Compose with a Dirichlet model via `+` (merge).
        let dir = Model::dirichlet("T".into(), "q".into(), std::slice::from_ref(&n0)).unwrap();
        assert_eq!(dir.sub_model_count(), 1);
        let full = (&hc + &dir).unwrap();
        assert_eq!(full.sub_model_count(), 3);
        assert_eq!(
            full.primal_vars().unwrap(),
            vec!["T".to_string(), "lambda_T".to_string()]
        );
    }

    /// Single-subspace `fes` → unit `Model` (the common case).
    #[test]
    fn parent_heat_conduction_unit_case() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let model = Model::heat_conduction(&fes).unwrap();
        assert_eq!(model.sub_model_count(), 1);
    }

    #[test]
    fn debug_and_display() {
        let (_cfg, _, _, model, _mat) = build_seg2_heat_model(1.0, 1.0, true);
        let d = format!("{:?}", model);
        assert!(d.contains("Model"));
        let s = format!("{}", model);
        assert!(s.contains("Model"));
        assert!(s.contains("2 sub-model"));
    }

    /// Two SEG2 zones with **different** conductivities, each carried by
    /// its own sub-model and its own SubElementField in a shared
    /// ElementField. The assembler must pick the right material per
    /// SubFiniteElementSpace.
    #[test]
    fn assemble_picks_per_zone_material() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        // Zone A: SEG2 on [0, 1]. Zone B: SEG2 on [1, 2]. Each as its own
        // SubMesh inside one Mesh.
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();

        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub_a = fes.subspace(0).unwrap();
        let sub_b = fes.subspace(1).unwrap();

        // Different conductivities on each zone.
        let k_a = 1.0;
        let k_b = 4.0;
        let mut mat_a = SubElementField::new(sub_a.clone(), vec!["k".into()]).unwrap();
        mat_a.set_uniform("k", k_a).unwrap();
        let mut mat_b = SubElementField::new(sub_b.clone(), vec!["k".into()]).unwrap();
        mat_b.set_uniform("k", k_b).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat_a)).unwrap();
        materials.add_sub(insert(mat_b)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub_a).unwrap()))
            .unwrap();
        model
            .add_sub(insert(SubModel::heat_conduction(sub_b).unwrap()))
            .unwrap();

        let k = assemble::stiffness(&model, &materials).unwrap();

        // For a SEG2 of length h = 1 and conductivity k:
        // K_local = (k / h) [[1, -1], [-1, 1]] = k [[1, -1], [-1, 1]].
        let tol = 1e-12;
        let v = |i: NodeId, j: NodeId| k.get(i, "q", j, "T").unwrap();
        // Diagonal at n0 = k_a only.
        assert!((v(n0.id(), n0.id()) - k_a).abs() < tol);
        // Diagonal at n1 = k_a + k_b (shared node).
        assert!((v(n1.id(), n1.id()) - (k_a + k_b)).abs() < tol);
        // Diagonal at n2 = k_b only.
        assert!((v(n2.id(), n2.id()) - k_b).abs() < tol);
        // Off-diagonals.
        assert!((v(n0.id(), n1.id()) + k_a).abs() < tol);
        assert!((v(n1.id(), n2.id()) + k_b).abs() < tol);
    }

    #[test]
    fn physics_declares_material_components() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let hc = SubModel::heat_conduction(sub).unwrap();
        assert_eq!(hc.material_components(), Some(&["k"][..]));

        let dir = SubModel::dirichlet("T".into(), "q".into(), std::slice::from_ref(&a)).unwrap();
        assert!(dir.material_components().is_none());
    }

    /// `assemble::stiffness` must fail with a clear error when no
    /// SubElementField matches a HeatConduction's FE subspace.
    #[test]
    fn assemble_errors_when_no_material_matches_fespace() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();

        // Empty ElementField — no SubElementField matches anything.
        let materials = ElementField::empty();
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let err = assemble::stiffness(&model, &materials).unwrap_err();
        assert!(format!("{}", err).contains("no SubElementField"));
    }
}
