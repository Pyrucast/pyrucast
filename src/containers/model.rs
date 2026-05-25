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
//! ├── dual_vars():   Vec<String>      # union over sub-models — rows
//! ├── stiffness() -> Matrix            # rows: dual × cols: primal
//! └── mass()      -> Matrix            # same DOF layout, may be empty
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
//! use pyrucast::mesh::configuration::{Configuration, NodeId};
//! use pyrucast::containers::element_field::SubElementField;
//! use pyrucast::mesh::element_type::ElementType;
//! use pyrucast::finite_element_space::FiniteElementSpace;
//! use pyrucast::mesh::Mesh;
//! use pyrucast::containers::model::{Model, Physics, SubModel};
//! use pyrucast::mesh::node::Node;
//! use pyrucast::store::{insert, with};
//!
//! // 1-D Configuration with two nodes spanning [0, 1].
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.subspace(0).unwrap();
//!
//! // Conductivity k = 1, uniform.
//! let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
//! mat.set_uniform("k", 1.0).unwrap();
//! let mat_h = insert(mat);
//!
//! let mut model = Model::new();
//! model
//!     .add_sub_model(insert(SubModel::heat_conduction(sub, mat_h)))
//!     .unwrap();
//! model
//!     .add_sub_model(insert(
//!         SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![a.id()]).unwrap(),
//!     ))
//!     .unwrap();
//!
//! let k = model.stiffness().unwrap();
//! // 2 real DOFs ("T") + 1 multiplier DOF + 2 real rows ("q") + 1 multiplier row.
//! assert_eq!(k.n_rows(), 3);
//! assert_eq!(k.n_cols(), 3);
//! ```

use crate::aggregate::Aggregate;
use crate::mesh::configuration::{Configuration, NodeId};
use crate::containers::element_field::SubElementField;
use crate::error::Result;
use crate::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::Matrix;
use crate::mesh::SubMesh;
use crate::models::{dirichlet, heat_conduction};
use crate::store::{with, Handle};
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
    /// - The `material` [`SubElementField`] **must** carry a component
    ///   named `"k"` (isotropic conductivity at each Gauss point). The
    ///   optional `"rho_cp"` component is reserved for the mass
    ///   matrix; not used in this v0.
    HeatConduction {
        fespace: Handle<SubFiniteElementSpace>,
        material: Handle<SubElementField>,
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
        /// POI1 SubMesh holding the per-node refcounts on the
        /// constrained nodes. Cells are in the same order as
        /// `constrained_nodes`.
        constrained_support: Handle<SubMesh>,
        /// POI1 SubMesh owning the multiplier nodes (one cell per
        /// multiplier, in the same order as `multiplier_nodes`). The
        /// SubMesh holds the only refcount on each multiplier; its
        /// `Drop` collects them.
        multiplier_support: Handle<SubMesh>,
        /// Cache of `constrained_support`'s connectivity for the
        /// assembly hot path.
        constrained_nodes: Vec<NodeId>,
        /// Cache of `multiplier_support`'s connectivity. One multiplier
        /// node per constrained node, in the same order.
        multiplier_nodes: Vec<NodeId>,
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

    /// Heat-conduction sub-model on an FE subspace, with material
    /// properties supplied by a [`SubElementField`].
    ///
    /// The `material` field **must** define a component named `"k"`
    /// (isotropic conductivity). The check is performed at assembly,
    /// not at construction.
    pub fn heat_conduction(
        fespace: Handle<SubFiniteElementSpace>,
        material: Handle<SubElementField>,
    ) -> Self {
        Self {
            physics: Physics::HeatConduction { fespace, material },
        }
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
        config: Handle<Configuration>,
        primal_var: String,
        primal_dual: String,
        constrained_nodes: Vec<NodeId>,
    ) -> Result<Self> {
        let built = dirichlet::build(config, &constrained_nodes)?;
        Ok(Self {
            physics: Physics::Dirichlet {
                primal_var,
                primal_dual,
                constrained_support: built.constrained_support,
                multiplier_support: built.multiplier_support,
                constrained_nodes,
                multiplier_nodes: built.multiplier_nodes,
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
    pub fn multiplier_nodes(&self) -> &[NodeId] {
        match &self.physics {
            Physics::Dirichlet { multiplier_nodes, .. } => multiplier_nodes,
            _ => &[],
        }
    }

    /// Primal variable names introduced by this sub-model.
    pub fn primal_vars(&self) -> Vec<String> {
        self.physics.primal_vars()
    }

    /// Dual variable names introduced by this sub-model.
    pub fn dual_vars(&self) -> Vec<String> {
        self.physics.dual_vars()
    }

    /// Assemble the local stiffness contribution of this sub-model into
    /// the provided global `Matrix`.
    pub fn assemble_stiffness(&self, k: &mut Matrix) -> Result<()> {
        match &self.physics {
            Physics::HeatConduction { fespace, material } => {
                heat_conduction::assemble_stiffness(fespace, material, k)
            }
            Physics::Dirichlet {
                primal_var,
                primal_dual,
                constrained_nodes,
                multiplier_nodes,
                ..
            } => {
                dirichlet::assemble_block(
                    constrained_nodes,
                    multiplier_nodes,
                    primal_var,
                    primal_dual,
                    k,
                );
                Ok(())
            }
        }
    }

    /// Assemble the local mass contribution of this sub-model into the
    /// provided global `Matrix`. Returns `Ok(())` and produces no entries
    /// when the physics has no inertial term (Dirichlet, …) — this is
    /// the **v0**, mass assembly for `HeatConduction` is intentionally
    /// left as a stub.
    pub fn assemble_mass(&self, _m: &mut Matrix) -> Result<()> {
        // v0: mass is not yet wired for any physics. Adding `rho_cp`
        // to HeatConduction would be additive once the integrand
        // ∫ N_i N_j dx is implemented.
        Ok(())
    }
}

impl fmt::Debug for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubModel")
            .field(
                "physics",
                &match &self.physics {
                    Physics::HeatConduction { .. } => "HeatConduction",
                    Physics::Dirichlet { .. } => "Dirichlet",
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
                constrained_nodes,
                ..
            } => write!(
                f,
                "SubModel<Dirichlet({})>: {} constrained node(s)",
                primal_var,
                constrained_nodes.len()
            ),
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
    sub_models: Vec<Handle<SubModel>>,
}

impl Aggregate for Model {
    type Sub = SubModel;
    fn items(&self) -> &[Handle<SubModel>] {
        &self.sub_models
    }
    fn items_mut(&mut self) -> &mut Vec<Handle<SubModel>> {
        &mut self.sub_models
    }
}

impl Model {
    /// Build an empty Model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sub-model. Validation (consistency of supports, materials,
    /// etc.) is deferred to assembly.
    pub fn add_sub_model(&mut self, sub: Handle<SubModel>) -> Result<()> {
        self.sub_models.push(sub);
        Ok(())
    }

    /// Number of sub-models. Alias of [`Aggregate::len`].
    pub fn sub_model_count(&self) -> usize {
        self.len()
    }

    /// Handle to the `i`-th sub-model (cloned). Alias of [`Aggregate::get`].
    pub fn sub_model(&self, i: usize) -> Result<Handle<SubModel>> {
        self.get(i)
    }

    /// Primal variable names — union over all sub-models, first-seen order.
    /// These are the **column labels** of the assembled matrices and the
    /// component names of the solution `NodeField`.
    pub fn primal_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in &self.sub_models {
            all.extend(with(h, |s| s.primal_vars())?);
        }
        Ok(union_names(all))
    }

    /// Dual variable names — union over all sub-models, first-seen order.
    /// These are the **row labels** of the assembled matrices and the
    /// component names of the load `NodeField`.
    pub fn dual_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in &self.sub_models {
            all.extend(with(h, |s| s.dual_vars())?);
        }
        Ok(union_names(all))
    }

    /// Assemble the stiffness matrix `K` of the full model.
    pub fn stiffness(&self) -> Result<Matrix> {
        let mut symmetric = true;
        for h in &self.sub_models {
            let is_sym = with(h, |s| {
                matches!(
                    s.physics(),
                    Physics::HeatConduction { .. } | Physics::Dirichlet { .. }
                )
            })?;
            if !is_sym {
                symmetric = false;
                break;
            }
        }
        let mut k = Matrix::new(symmetric);
        for h in &self.sub_models {
            with(h, |sub| sub.assemble_stiffness(&mut k))??;
        }
        Ok(k)
    }

    /// Assemble the mass matrix `M` of the full model. In this v0 each
    /// physics's `assemble_mass` is a stub, so the returned matrix is
    /// empty unless a future physics fills it.
    pub fn mass(&self) -> Result<Matrix> {
        let mut m = Matrix::new(true);
        for h in &self.sub_models {
            with(h, |sub| sub.assemble_mass(&mut m))??;
        }
        Ok(m)
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("sub_model_count", &self.sub_models.len())
            .finish()
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model: {} sub-model(s)", self.sub_models.len())
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
    use crate::mesh::configuration::Configuration;
    use crate::mesh::element_type::ElementType;
    use crate::finite_element_space::FiniteElementSpace;
    use crate::mesh::Mesh;
    use crate::mesh::node::Node;
    use crate::store::{insert, with_mut};

    /// Build a 1-D heat-conduction model on a single SEG2 element of
    /// length `length`, uniform conductivity `k`, with optional Dirichlet
    /// at the left node.
    fn build_seg2_heat_model(
        length: f64,
        k: f64,
        dirichlet_at_left: bool,
    ) -> (Handle<Configuration>, NodeId, NodeId, Model) {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[length]).unwrap();
        let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();

        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", k).unwrap();
        let mat_h = insert(mat);

        let mut model = Model::new();
        model
            .add_sub_model(insert(SubModel::heat_conduction(sub, mat_h)))
            .unwrap();
        if dirichlet_at_left {
            model
                .add_sub_model(insert(
                    SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![a.id()])
                        .unwrap(),
                ))
                .unwrap();
        }
        (cfg, a.id(), b.id(), model)
    }

    #[test]
    fn primal_dual_vars_for_heat_conduction_alone() {
        let (_cfg, _, _, model) = build_seg2_heat_model(1.0, 1.0, false);
        assert_eq!(model.primal_vars().unwrap(), vec!["T".to_string()]);
        assert_eq!(model.dual_vars().unwrap(), vec!["q".to_string()]);
    }

    #[test]
    fn primal_dual_vars_include_lagrange_after_dirichlet() {
        let (_cfg, _, _, model) = build_seg2_heat_model(1.0, 1.0, true);
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
        let (_cfg, a_id, b_id, model) = build_seg2_heat_model(length, k_val, false);
        let k = model.stiffness().unwrap();

        assert_eq!(k.n_rows(), 2);
        assert_eq!(k.n_cols(), 2);
        let expected = k_val / length;
        let tol = 1e-12;
        assert!((k.get(a_id, "q", a_id, "T") - expected).abs() < tol);
        assert!((k.get(a_id, "q", b_id, "T") + expected).abs() < tol);
        assert!((k.get(b_id, "q", a_id, "T") + expected).abs() < tol);
        assert!((k.get(b_id, "q", b_id, "T") - expected).abs() < tol);
    }

    /// Two SEG2 elements sharing the middle node form the classic
    /// tridiagonal `(1, -2, 1)` pattern after assembly.
    #[test]
    fn two_seg2_assembly_is_tridiagonal() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mat_h = insert(mat);

        let mut model = Model::new();
        model
            .add_sub_model(insert(SubModel::heat_conduction(sub, mat_h)))
            .unwrap();
        let k = model.stiffness().unwrap();
        assert_eq!(k.n_rows(), 3);
        assert_eq!(k.n_cols(), 3);

        let v = |i: NodeId, j: NodeId| k.get(i, "q", j, "T");
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
        let (cfg, a_id, _b_id, model) = build_seg2_heat_model(1.0, 1.0, true);

        // The Configuration grew by one node (the multiplier).
        let n_nodes = with(&cfg, |c| c.node_count()).unwrap();
        assert_eq!(n_nodes, 3);

        let k = model.stiffness().unwrap();
        // 2 real "q" rows + 1 multiplier "T" row = 3 rows.
        // 2 real "T" cols + 1 multiplier "lambda_T" col = 3 cols.
        assert_eq!(k.n_rows(), 3);
        assert_eq!(k.n_cols(), 3);

        // Find the multiplier node id: the only NodeId that appears in
        // a row labelled "T" of K.
        let t_idx = k.field_index("T").unwrap();
        let lambda_idx = k.field_index("lambda_T").unwrap();
        let mult = k
            .row_dofs()
            .iter()
            .find(|d| d.field_idx == t_idx)
            .expect("multiplier row missing")
            .node_id;

        // C entry: (mult, "T") × (a_id, "T") = 1
        assert_eq!(k.get(mult, "T", a_id, "T"), 1.0);
        // Cᵀ entry: (a_id, "q") × (mult, "lambda_T") = 1
        assert_eq!(k.get(a_id, "q", mult, "lambda_T"), 1.0);
        // Ensure lambda_T appears as a column.
        let lambda_col_present = k
            .col_dofs()
            .iter()
            .any(|d| d.field_idx == lambda_idx && d.node_id == mult);
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
            cfg.clone(),
            "T".into(),
            "q".into(),
            vec![a_id],
        )
        .unwrap();

        // After: the Node + the SubModel each hold 1 ref ⇒ 2.
        with(&cfg, |c| assert_eq!(c.refcount(a_id), 2)).unwrap();
        // The multiplier node has refcount 1 (owned by the sub-model).
        let mult_id = match sub.physics() {
            Physics::Dirichlet { multiplier_nodes, .. } => multiplier_nodes[0],
            _ => unreachable!(),
        };
        with(&cfg, |c| assert_eq!(c.refcount(mult_id), 1)).unwrap();

        drop(sub);
        // Now back to 1 (only the Node remains).
        with(&cfg, |c| assert_eq!(c.refcount(a_id), 1)).unwrap();
        // And the multiplier is collectable.
        with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
    }

    #[test]
    fn dirichlet_empty_constraint_list_rejected() {
        let cfg = insert(Configuration::new(1).unwrap());
        assert!(SubModel::dirichlet(cfg, "T".into(), "q".into(), vec![]).is_err());
    }

    #[test]
    fn heat_conduction_errors_on_missing_k_component() {
        // Material has only "rho_cp", not "k": stiffness assembly must fail.
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::with_element_type(cfg, ElementType::SEG2);
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.subspace(0).unwrap();
        let mat = SubElementField::new(sub.clone(), vec!["rho_cp".into()]).unwrap();
        let mat_h = insert(mat);
        let mut model = Model::new();
        model
            .add_sub_model(insert(SubModel::heat_conduction(sub, mat_h)))
            .unwrap();
        assert!(model.stiffness().is_err());
    }

    #[test]
    fn empty_model_produces_empty_matrices() {
        let model = Model::new();
        let k = model.stiffness().unwrap();
        let m = model.mass().unwrap();
        assert_eq!(k.n_rows(), 0);
        assert_eq!(k.n_cols(), 0);
        assert_eq!(m.n_rows(), 0);
        assert_eq!(m.n_cols(), 0);
    }

    #[test]
    fn debug_and_display() {
        let (_cfg, _, _, model) = build_seg2_heat_model(1.0, 1.0, true);
        let d = format!("{:?}", model);
        assert!(d.contains("Model"));
        let s = format!("{}", model);
        assert!(s.contains("Model"));
        assert!(s.contains("2 sub-model"));
    }
}
