//! Physical model — orchestrator of sub-models that assemble into a
//! [`crate::containers::matrix::Matrix`].
//!
//! The model layer is the **physics-aware** counterpart of the
//! geometry layer (`Mesh`, `SubMesh`) and the interpolation layer
//! (`FiniteElementSpace`, `SubFiniteElementSpace`). A [`Model`] is an aggregate of
//! [`SubModel`]s; each [`SubModel`] is one physics instance (a variant of
//! the enum) that owns its supports and dispatches its behaviour through
//! the [`SubModelKind`] trait. The Model is a pure orchestrator: it enumerates
//! the DOFs of its sub-models, dimensions a
//! [`crate::containers::matrix::Matrix`], and
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
//! ops::matrix (operators, not Model methods)
//! ├── stiffness(model, materials) -> Matrix   # rows: dual × cols: primal
//! └── mass(model)                 -> Matrix   # same DOF layout, may be empty
//!
//! SubModel  (enum : stockage + sérialisation ; dispatch via as_kind())
//! ├── HeatConduction(HeatConduction)
//! ├── Dirichlet(Dirichlet)             # constraint = Lagrange multiplier
//! └── ...
//!
//! SubModelKind  (trait : tout le comportement, co-localisé par physique)
//! └── primal_vars / dual_vars / material_* / build_*_blocks / render / ...
//! ```
//!
//! The model layer is purely matrix-producing. Loads (right-hand side
//! vectors) are entirely the user's responsibility: read
//! `model.dual_vars()`, build a [`crate::containers::node_field::SubNodeField`] with
//! the matching component names, and feed `Matrix + SubNodeField` to the
//! solver.
//!
//! # Lagrange multipliers and DOF identification
//!
//! `SubModel::Dirichlet` introduces new DOFs of two kinds, both living on the
//! **multiplier nodes** supplied by the user (`multiplier_mesh`; the sub-model
//! creates no node):
//!
//! - the **primal** of the constraint sub-model is `multiplier` (default
//!   `lambda_<imposed_variable>`) at the multiplier nodes — the Lagrange
//!   multiplier itself, an unknown of the augmented system whose solved value
//!   is the reaction;
//! - the **dual** of the constraint sub-model is `imposed_value` (default
//!   `imposed_<imposed_variable>`) at the multiplier nodes — the constraint
//!   equation row, and the slot at which the user writes the imposed value.
//!
//! Distinct `(NodeId, field_name)` pairs keep these multiplier DOFs apart from
//! the primary DOFs. The blocks `C` (constraint) and `Cᵀ` (its transpose — the
//! reaction in the target's `target_dual` row) are stored explicitly and each
//! marked **non-symmetric**: only their union `C ∪ Cᵀ` is symmetric, a global
//! property of the saddle-point system (the dense LU solver ignores the flag
//! anyway).
//!
//! # Example: 1-D heat conduction with a Dirichlet condition
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::atoms::NodeId;
//! use pyrucast::coords::Coords;
//! use pyrucast::containers::element_field::SubElementField;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::field::SubField;
//! use pyrucast::containers::model::{Model, SubModel};
//! use pyrucast::atoms::Node;
//! use pyrucast::ops::matrix;
//! use pyrucast::ops::mesh;
//! use pyrucast::store::insert;
//!
//! // 1-D Coords with two nodes spanning [0, 1].
//! let coords = insert(Coords::new(1).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.get(0).unwrap();
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
//! // Dirichlet on node `a`: the imposed POI1 mesh + a colocated multiplier
//! // support minted by the `barycenter` mesher.
//! let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
//! let multiplier = mesh::barycenter(&imposed).unwrap();
//! model
//!     .add_sub(insert(
//!         SubModel::dirichlet("T".into(), "q".into(), &imposed, &multiplier, None, None, Default::default())
//!             .unwrap(),
//!     ))
//!     .unwrap();
//!
//! let k = pyrucast::ops::matrix::stiffness(&model, &materials).unwrap();
//! // 2 real DOFs ("T") + 1 multiplier DOF + 2 real rows ("q") + 1 multiplier row.
//! assert_eq!(k.n_rows().unwrap(), 3);
//! assert_eq!(k.n_cols().unwrap(), 3);
//! ```

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::matrix::AssemblyPattern;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::ElasticityModel;
use crate::models::symmetry::MaterialSymmetry;
use crate::models::{
    contact, convection, dirichlet, elasticity, embedded, fick, frame, frame3d, heat_conduction,
    interface_transfer, mazars, mpc, plasticity, timoshenko, truss, Constraint, MatrixKind,
    Physics, RelationSense, SubModelKind,
};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::{Arc, OnceLock};

/// Record `g` at a relation's multiplier node in the `constraint_rhs*` helpers,
/// erroring when two entries hit the **same** relation with different values.
/// `caller` names the method for the message.
fn insert_relation_value(
    values: &mut std::collections::HashMap<(NodeId, String), f64>,
    multiplier_node: NodeId,
    slot: &str,
    g: f64,
    caller: &str,
) -> Result<()> {
    if let Some(prev) = values.insert((multiplier_node, slot.to_string()), g) {
        if prev != g {
            return Err(PyrucastError::Message(format!(
                "{caller}: conflicting values ({prev} and {g}) for the same relation \
                 (multiplier node {multiplier_node}, slot {slot})"
            )));
        }
    }
    Ok(())
}

// ─── SubModel ──────────────────────────────────────────────────────────────

/// One physics instance, bound to its supports (FE spaces, materials, node
/// sets). A [`Model`] is a `Vec<Handle<SubModel>>`.
///
/// This enum is a **pure storage + dispatch** shell: it derives
/// `Serialize`/`Deserialize` (so models persist through the `bincode`
/// backbone), and forwards every behavioural call to the variant's
/// [`SubModelKind`] implementation through [`SubModel::as_kind`]. All physics
/// logic lives in the per-variant structs under [`crate::models`].
///
/// Adding a physics means adding **one variant here** and **one arm to
/// [`SubModel::as_kind`]** — no other site in this file changes.
#[derive(Serialize, Deserialize)]
pub enum SubModel {
    /// Linear heat conduction — see [`heat_conduction::HeatConduction`].
    HeatConduction(heat_conduction::HeatConduction),
    /// Surface convection (Robin / film) boundary — see
    /// [`convection::Convection`].
    Convection(convection::Convection),
    /// Dirichlet constraint via Lagrange multipliers — see
    /// [`dirichlet::Dirichlet`].
    Dirichlet(dirichlet::Dirichlet),
    /// Multi-point constraint (linear relations) via Lagrange multipliers —
    /// see [`mpc::Mpc`].
    Mpc(mpc::Mpc),
    /// Embedded (immersed) constraint tying immersed nodes to a host
    /// interpolation — see [`embedded::Embedded`].
    Embedded(embedded::Embedded),
    /// Node-to-surface contact (unilateral, frictionless) — see
    /// [`contact::Contact`].
    Contact(contact::Contact),
    /// Truss / bar (axial-force) element — see [`truss::Truss`].
    Truss(truss::Truss),
    /// Linear elasticity (2-D plane / 3-D solid) — see [`elasticity::Elasticity`].
    Elasticity(elasticity::Elasticity),
    /// Perfect von Mises elastoplasticity — see [`plasticity::Plasticity`].
    Plasticity(plasticity::Plasticity),
    /// Mazars isotropic damage — see [`mazars::Mazars`].
    Mazars(mazars::Mazars),
    /// Timoshenko beam (shear-deformable bending) — see [`timoshenko::Timoshenko`].
    Timoshenko(timoshenko::Timoshenko),
    /// Planar frame / portique (axial + bending + shear) — see [`frame::Frame`].
    Frame(frame::Frame),
    /// 3-D Timoshenko frame (space frame, 6 DOF/node) — see [`frame3d::Frame3d`].
    Frame3d(frame3d::Frame3d),
    // New variants go **at the end**: `bincode` serialises the variant index, so
    // inserting one in the middle would silently misread every saved model.
    /// Fickian diffusion (concentration / mass flux) — see [`fick::Fick`].
    Fick(fick::Fick),
    /// Exchange law across an interface between two meshes — see
    /// [`interface_transfer::InterfaceTransfer`].
    InterfaceTransfer(interface_transfer::InterfaceTransfer),
}

impl SubModel {
    /// Borrow the variant as its [`SubModelKind`] behaviour. This is the
    /// **only** per-variant `match` in the model layer; every generic
    /// method (variable names, material contract, assembly, rendering)
    /// dispatches through it.
    pub fn as_kind(&self) -> &dyn SubModelKind {
        match self {
            SubModel::HeatConduction(p) => p,
            SubModel::Convection(p) => p,
            SubModel::Dirichlet(p) => p,
            SubModel::Mpc(p) => p,
            SubModel::Embedded(p) => p,
            SubModel::Contact(p) => p,
            SubModel::Truss(p) => p,
            SubModel::Elasticity(p) => p,
            SubModel::Plasticity(p) => p,
            SubModel::Mazars(p) => p,
            SubModel::Timoshenko(p) => p,
            SubModel::Frame(p) => p,
            SubModel::Frame3d(p) => p,
            SubModel::Fick(p) => p,
            SubModel::InterfaceTransfer(p) => p,
        }
    }

    /// Heat-conduction sub-model on an FE subspace.
    ///
    /// Material data (conductivity `"k"`, …) is supplied separately at
    /// assembly time via [`crate::ops::matrix::stiffness`], keeping
    /// the model immutable and material-independent. See
    /// [`heat_conduction::HeatConduction::new`] for the support it builds.
    pub fn heat_conduction(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Self::heat_conduction_with_symmetry(fespace, MaterialSymmetry::Isotropic)
    }

    /// Heat conduction with an explicit material symmetry — an orthotropic or
    /// anisotropic conductivity carries its constants **and its axes** through
    /// the material field. See [`crate::models::symmetry`].
    pub fn heat_conduction_with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        Ok(SubModel::HeatConduction(
            heat_conduction::HeatConduction::with_symmetry(fespace, symmetry)?,
        ))
    }

    /// Surface-convection (Robin / film) sub-model on a boundary FE subspace.
    ///
    /// Same DOFs (`"T"`/`"q"`) as [`Self::heat_conduction`], so it couples into
    /// the conduction stiffness. The film coefficient `"h"` is supplied at
    /// assembly time via [`crate::ops::matrix::stiffness`]; the external
    /// temperature enters as a load (`h·T_ext ∫N_i dΓ`) built with
    /// [`flux`](fn@crate::ops::node_field::flux). See
    /// [`convection::Convection::new`].
    pub fn convection(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Convection(convection::Convection::new(fespace)?))
    }

    /// Truss / bar sub-model on a `SEG2` FE subspace. Material data (`E`, `A`)
    /// is supplied at assembly time. See [`truss::Truss::new`].
    pub fn truss(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Truss(truss::Truss::new(fespace)?))
    }

    /// **Isotropic** linear-elasticity sub-model on an FE subspace, with the
    /// given 2-D/3-D model. Material data (`E`, `nu`) is supplied at assembly
    /// time. See [`elasticity::Elasticity::new`].
    pub fn elasticity(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
    ) -> Result<Self> {
        Self::elasticity_with_symmetry(fespace, model, MaterialSymmetry::Isotropic)
    }

    /// Linear elasticity with an explicit material symmetry — orthotropic and
    /// anisotropic materials carry their constants **and their axes** through the
    /// material field. See [`crate::models::symmetry`] for the contracts.
    pub fn elasticity_with_symmetry(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        Ok(SubModel::Elasticity(elasticity::Elasticity::with_symmetry(
            fespace, model, symmetry,
        )?))
    }

    /// Fickian-diffusion sub-model on an FE subspace (primal `c`, dual `j`).
    /// Material data (the diffusivity, isotropic or oriented) is supplied at
    /// assembly time. See [`fick::Fick::with_symmetry`].
    pub fn fick(
        fespace: Handle<SubFiniteElementSpace>,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        Ok(SubModel::Fick(fick::Fick::with_symmetry(
            fespace, symmetry,
        )?))
    }

    /// Interface-exchange sub-model between two **conforming** boundary FE
    /// subspaces — `j·n = h(c₁ − c₂)`, or its thermal twin. The transfer
    /// coefficient `h` is supplied at assembly time. See
    /// [`interface_transfer::InterfaceTransfer::new`].
    pub fn interface_transfer(
        side_a: Handle<SubFiniteElementSpace>,
        side_b: Handle<SubFiniteElementSpace>,
        kind: interface_transfer::TransferKind,
        tol: f64,
    ) -> Result<Self> {
        Ok(SubModel::InterfaceTransfer(
            interface_transfer::InterfaceTransfer::new(side_a, side_b, kind, tol)?,
        ))
    }

    /// Perfect-plasticity sub-model on an FE subspace, with the given 2-D/3-D
    /// model. Material (`E`, `nu`, `sigma_y`) is supplied at assembly /
    /// integration time. See [`plasticity::Plasticity::new`].
    pub fn plasticity(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
    ) -> Result<Self> {
        Ok(SubModel::Plasticity(plasticity::Plasticity::new(
            fespace, model,
        )?))
    }

    /// Mazars-damage sub-model on an FE subspace, with the given 2-D/3-D model.
    /// Material (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at
    /// assembly / integration time. See [`mazars::Mazars::new`].
    pub fn mazars(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        Ok(SubModel::Mazars(mazars::Mazars::new(fespace, model)?))
    }

    /// Timoshenko-beam sub-model on a 1-D `SEG2` FE subspace (full Gauss).
    /// Material data (`E`, `I`, `G`, `A_s`) is supplied at assembly time. See
    /// [`timoshenko::Timoshenko::new`].
    pub fn timoshenko(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Timoshenko(timoshenko::Timoshenko::new(fespace)?))
    }

    /// Planar-frame sub-model on a 2-D `SEG2` FE subspace (axial + bending +
    /// shear, any orientation). Material (`E`, `A`, `I`, `G`, `A_s`) is supplied
    /// at assembly time. See [`frame::Frame::new`].
    pub fn frame(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Frame(frame::Frame::new(fespace)?))
    }

    /// 3-D Timoshenko-frame sub-model on a 3-D `SEG2` FE subspace (6 DOF/node).
    /// Material (`E, A, I_y, I_z, J, G, A_sy, A_sz`) is supplied at assembly
    /// time. See [`frame3d::Frame3d::new`].
    pub fn frame3d(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Frame3d(frame3d::Frame3d::new(fespace)?))
    }

    /// Dirichlet sub-model: enforce `imposed_variable = u_d` on the nodes of
    /// `imposed_mesh`, with multipliers living on `multiplier_mesh` and `u_d`
    /// supplied later by the user through the load `SubNodeField`.
    ///
    /// `target_dual` is the dual variable name of the target physics whose
    /// primal is being constrained (e.g. `"q"` for heat conduction, `"f_x"`
    /// for elasticity in `x`). `multiplier` / `imposed_value` default to
    /// `lambda_<imposed_variable>` / `imposed_<imposed_variable>` when `None`.
    /// `sense` (default equality) turns the constraint unilateral (`u ≥ u_d` /
    /// `u ≤ u_d`), solved by the active-set operator
    /// [`unilateral`](crate::ops::solver::unilateral).
    /// See [`dirichlet::Dirichlet::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn dirichlet(
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: RelationSense,
    ) -> Result<Self> {
        Ok(SubModel::Dirichlet(dirichlet::Dirichlet::new(
            imposed_variable,
            target_dual,
            imposed_mesh,
            multiplier_mesh,
            multiplier,
            imposed_value,
            sense,
        )?))
    }

    /// Multi-point constraint sub-model: impose a linear relation
    /// `Σₖ aₖ·u(nodeₖ, varₖ) = g` per relation, via Lagrange multipliers.
    ///
    /// `terms` are the `(mesh, variable, target_dual, coefficient)` terms
    /// (build them with [`mpc::MpcTerm::new`]); `multiplier_mesh` carries one
    /// `λ` node per relation. `multiplier` / `imposed_value` default to
    /// `lambda_mpc` / `mpc_rhs`. The right-hand side `g` is written by the user
    /// in the load field at the multiplier node's `imposed_value` component
    /// (default `0`). `sense` (default equality) turns the relations unilateral
    /// (`Σ aₖ·uₖ ≥ g` / `≤ g`), solved by the active-set operator
    /// [`unilateral`](crate::ops::solver::unilateral). See [`mpc::Mpc::new`].
    pub fn mpc(
        terms: Vec<mpc::MpcTerm>,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: RelationSense,
    ) -> Result<Self> {
        Ok(SubModel::Mpc(mpc::Mpc::new(
            terms,
            multiplier_mesh,
            multiplier,
            imposed_value,
            sense,
        )?))
    }

    /// Embedded (immersed) constraint sub-model: tie each node of `immersed` to
    /// the interpolation of `host` at that node, for every `(variable,
    /// target_dual)` in `components` (e.g. `[("u_x","f_x"), ("u_y","f_y"),
    /// ("u_z","f_z")]`) — a bar « baignée » in a volume.
    ///
    /// The coupling weights are the host shape functions at each immersed node,
    /// computed once at build by locating the node in the host. `multipliers` /
    /// `imposed_values` default to `lambda_<variable>` / `imposed_<variable>`;
    /// `tol` is the location tolerance (default `1e-6`). The right-hand side `g`
    /// defaults to `0` (a rigid tie). See [`embedded::Embedded::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn embedded(
        immersed: &Mesh,
        host: &Mesh,
        components: Vec<(String, String)>,
        multipliers: Option<Vec<String>>,
        imposed_values: Option<Vec<String>>,
        tol: Option<f64>,
    ) -> Result<Self> {
        Ok(SubModel::Embedded(embedded::Embedded::new(
            immersed,
            host,
            components,
            multipliers,
            imposed_values,
            tol,
        )?))
    }

    /// Node-to-surface contact sub-model: prevent the nodes of `slave` from
    /// penetrating the oriented `master` surface mesh — one **unilateral**
    /// relation (`≥`) per slave node, paired at build with its closest master
    /// facet ([`crate::ops::geom::project_points`]).
    ///
    /// `components` couples the displacement components through the facet
    /// normal: one `(variable, target_dual)` pair per space dimension, in
    /// ambient order (e.g. `[("u_x","f_x"), ("u_y","f_y")]`). `multiplier` /
    /// `imposed_value` default to `lambda_contact` / `contact_gap`. Solve with
    /// [`unilateral`](crate::ops::solver::unilateral); build the `−g₀`
    /// right-hand side with [`Model::contact_gaps`]. See
    /// [`contact::Contact::new`].
    pub fn contact(
        slave: &Mesh,
        master: &Mesh,
        components: Vec<(String, String)>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        Ok(SubModel::Contact(contact::Contact::new(
            slave,
            master,
            components,
            multiplier,
            imposed_value,
        )?))
    }

    /// The dual (residual) variable conjugate to a primal `variable`, by the
    /// positional pairing `primal_vars[i] ↔ dual_vars[i]` this sub-model
    /// declares, or `None` if it declares no such primal. A helper for the MPC
    /// mise-en-donnée: `dual_of("u_x") == Some("f_x")`.
    pub fn dual_of(&self, variable: &str) -> Option<String> {
        let kind = self.as_kind();
        let position = kind.primal_vars().iter().position(|p| p == variable)?;
        kind.dual_vars().get(position).cloned()
    }

    /// Multiplier node ids introduced by this sub-model. Non-empty only
    /// for Lagrange variants (`Dirichlet`, future `MultipointConstraint`,
    /// …); empty for the other physics.
    ///
    /// Useful for the user who needs to write the imposed value `u_d` at
    /// the multiplier node's `imposed_value` component of the load
    /// `SubNodeField`.
    pub fn multiplier_nodes(&self) -> Result<Vec<NodeId>> {
        let mut out = Vec::new();
        if let Some(constraint) = self.as_kind().as_constraint() {
            for sm in constraint.multiplier_mesh() {
                out.extend(read(sm)?.connectivity().iter().copied());
            }
        }
        Ok(out)
    }

    /// POI1 [`Mesh`] of the multiplier nodes (shares the multiplier submeshes
    /// — zero-copy). Empty for non-Lagrange physics.
    ///
    /// This is the user-facing handle to the multiplier nodes: build a load
    /// [`crate::containers::node_field::SubNodeField`] on it to impose the
    /// constrained values.
    pub fn multiplier_mesh(&self) -> Result<Mesh> {
        let mut mesh = Mesh::empty();
        if let Some(constraint) = self.as_kind().as_constraint() {
            for sm in constraint.multiplier_mesh() {
                mesh.add_sub(sm.clone())?;
            }
        }
        Ok(mesh)
    }

    /// Build the constraint load (right-hand side) for this sub-model from a set
    /// of `(constrained_node, g)` pairs — the mise-en-donnée helper that spares
    /// the user from rebuilding the multiplier mesh and remembering the
    /// imposed-value component name by hand.
    ///
    /// Returns a fresh [`NodeField`] over **this constraint's multiplier nodes**,
    /// carrying the single component the constraint uses as its imposed-value
    /// slot — its dual variable, e.g. `imposed_T` for a `Dirichlet` on `T` or
    /// `mpc_rhs` for an `Mpc`. Every multiplier node is present; each cited
    /// relation gets its `g`, the others default to `0`. Union it into the global
    /// load with `|` (`load | constraint_rhs`).
    ///
    /// A node **keys the relation it belongs to**: for `Dirichlet` the single
    /// constrained node, for `Mpc` any of the relation's term nodes. The method
    /// looks up that relation's multiplier node (via
    /// [`Constraint::relations`]) and writes
    /// `g` there. When a node participates in several relations (so node keying
    /// is ambiguous), key by index instead with
    /// [`constraint_rhs_by_index`](Self::constraint_rhs_by_index).
    ///
    /// Errors when `self` is not a constraint, when a cited node is constrained
    /// by none of its relations, when a node keys **several** relations (node
    /// keying is ambiguous there), or when two cited nodes key the *same*
    /// relation with conflicting values.
    pub fn constraint_rhs(&self, imposed: &[(NodeId, f64)]) -> Result<NodeField> {
        let (constraint, components) = self.constraint_components()?;
        let relations = constraint.relations()?;

        // Map each constrained term node → the index of its relation, marking
        // `None` as soon as a node keys two *distinct* relations (a
        // multi-component `Embedded` node keys one relation per component, so it
        // is ambiguous here — the message points to the by-index variant).
        let mut node_to_rel: std::collections::HashMap<NodeId, Option<usize>> =
            std::collections::HashMap::new();
        for (i, rel) in relations.iter().enumerate() {
            for term in &rel.terms {
                node_to_rel
                    .entry(term.node)
                    .and_modify(|slot| {
                        if *slot != Some(i) {
                            *slot = None;
                        }
                    })
                    .or_insert(Some(i));
            }
        }

        // Resolve the requested (node, g) → (multiplier node, slot, g).
        let mut values: std::collections::HashMap<(NodeId, String), f64> =
            std::collections::HashMap::new();
        for &(node, g) in imposed {
            let idx = match node_to_rel.get(&node) {
                None => {
                    return Err(PyrucastError::Message(format!(
                        "constraint_rhs: node {node} is not constrained by this {}",
                        self.as_kind().label()
                    )))
                }
                Some(None) => {
                    return Err(PyrucastError::Message(format!(
                        "constraint_rhs: node {node} keys several relations of this {} — \
                         node keying is ambiguous, use constraint_rhs_by_index",
                        self.as_kind().label()
                    )))
                }
                Some(Some(i)) => *i,
            };
            let rel = &relations[idx];
            insert_relation_value(
                &mut values,
                rel.multiplier_node,
                &rel.imposed_value,
                g,
                "constraint_rhs",
            )?;
        }

        self.multiplier_load(&components, &values)
    }

    /// Like [`constraint_rhs`](Self::constraint_rhs) but each relation is keyed
    /// by its **index** in [`Constraint::relations`]
    /// order (`0`-based) rather than by a node — the way to reach a relation
    /// whose nodes are shared with others, where node keying would be ambiguous.
    ///
    /// `imposed` is a set of `(relation_index, g)` pairs; the returned field and
    /// the union semantics are identical to `constraint_rhs`. Errors when `self`
    /// is not a constraint, when an index is out of range, or when two pairs
    /// target the same relation with conflicting values.
    pub fn constraint_rhs_by_index(&self, imposed: &[(usize, f64)]) -> Result<NodeField> {
        let (constraint, components) = self.constraint_components()?;
        let relations = constraint.relations()?;

        let mut values: std::collections::HashMap<(NodeId, String), f64> =
            std::collections::HashMap::new();
        for &(idx, g) in imposed {
            let rel = relations.get(idx).ok_or_else(|| {
                PyrucastError::Message(format!(
                    "constraint_rhs_by_index: relation index {idx} out of range (this \
                     constraint has {} relation(s))",
                    relations.len()
                ))
            })?;
            insert_relation_value(
                &mut values,
                rel.multiplier_node,
                &rel.imposed_value,
                g,
                "constraint_rhs_by_index",
            )?;
        }

        self.multiplier_load(&components, &values)
    }

    /// Borrow this sub-model as a [`Constraint`](crate::models::Constraint) and
    /// return its imposed-value components (its duals — one for a single-dual
    /// constraint, several for a multi-component `Embedded`). Shared entry check
    /// of the `constraint_rhs*` helpers.
    fn constraint_components(&self) -> Result<(&dyn Constraint, Vec<String>)> {
        let kind = self.as_kind();
        let constraint = kind.as_constraint().ok_or_else(|| {
            PyrucastError::Message(format!(
                "constraint_rhs: sub-model {} is not a constraint (it has no multipliers)",
                kind.label()
            ))
        })?;
        let components = kind.dual_vars();
        if components.is_empty() {
            return Err(PyrucastError::Message(
                "constraint_rhs: the constraint owns no dual (imposed-value) variable".into(),
            ));
        }
        Ok((constraint, components))
    }

    /// Build the constraint load over **every** multiplier node: one
    /// zero-initialised zone per multiplier submesh (carrying **all** the
    /// constraint's imposed-value `components`, sharing the supports and
    /// preserving the structure carried through `barycenter`), then drop each
    /// resolved `(multiplier node, slot) → g` in at its component. Shared tail of
    /// the `constraint_rhs*` helpers.
    fn multiplier_load(
        &self,
        components: &[String],
        values: &std::collections::HashMap<(NodeId, String), f64>,
    ) -> Result<NodeField> {
        let mut field = NodeField::default();
        for sm in &self.multiplier_mesh()? {
            let mut sub = SubNodeField::from_poi1(sm, components.to_vec())?;
            let nids: Vec<NodeId> = read(sm)?.connectivity().to_vec();
            for nid in nids {
                for comp in components {
                    if let Some(&g) = values.get(&(nid, comp.clone())) {
                        sub.set_value(nid, comp, g)?;
                    }
                }
            }
            field.add_sub(insert(sub))?;
        }
        Ok(field)
    }

    /// FE subspace on which this sub-model expects its material data, or
    /// `None` if this physics doesn't need material data (e.g. `Dirichlet`).
    pub fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        self.as_kind().as_domain().map(|d| d.material_fespace())
    }

    /// Material component names this sub-model expects, or `None` if it
    /// doesn't need material data. Thin pass-through of
    /// [`Domain::material_components`](crate::models::Domain::material_components).
    pub fn material_components(&self) -> Option<&'static [&'static str]> {
        self.as_kind()
            .as_domain()
            .and_then(|d| d.material_components())
    }

    /// Optional material component names this sub-model accepts (never required).
    /// Thin pass-through of
    /// [`Domain::optional_material_components`](crate::models::Domain::optional_material_components).
    pub fn optional_material_components(&self) -> &'static [&'static str] {
        self.as_kind()
            .as_domain()
            .map(|d| d.optional_material_components())
            .unwrap_or(&[])
    }

    /// Primal variable names introduced by this sub-model.
    pub fn primal_vars(&self) -> Vec<String> {
        self.as_kind().primal_vars()
    }

    /// Dual variable names introduced by this sub-model.
    pub fn dual_vars(&self) -> Vec<String> {
        self.as_kind().dual_vars()
    }

    /// This sub-model's set of [`Physics`] natures — a per-variant constant
    /// slice, determined entirely by the variant (a single nature today; several
    /// for a future coupled physics). Feeds [`Model::filter`] and travels with
    /// each assembled block onto the
    /// [`SubMatrix`](crate::containers::matrix::SubMatrix).
    pub fn physics(&self) -> &'static [Physics] {
        self.as_kind().physics()
    }

    /// Whether this sub-model carries a constitutive behaviour that can be
    /// integrated via `integrate_behavior` from a
    /// deformation field. `true` for volumetric physics, `false` for
    /// constraints (`Dirichlet`).
    pub fn has_behavior(&self) -> bool {
        self.as_kind().as_domain().is_some()
    }

    /// FE subspace this sub-model integrates its behaviour on, or `None`
    /// for a constraint sub-model. The operators in
    /// [`crate::ops::element_field::behavior`] use it to pair the per-zone deformation
    /// field with its sub-model.
    pub fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        self.as_kind().as_domain().map(|d| d.behavior_fespace())
    }

    /// Integrate this sub-model's constitutive law (Cast3m `COMP`), stepping
    /// A → B. The caller ([`crate::ops::element_field::behavior::integrate`]) supplies the
    /// matching per-zone end-of-step `deformation` ε(B) (from
    /// [`crate::ops::element_field::gradient`] / [`crate::ops::element_field::deformation`]),
    /// the previous converged state `prev` (state at A, `None` on the first
    /// step), the `material`, and the time increment `dt` (`None` if
    /// rate-independent).
    pub(crate) fn integrate_behavior(
        &self,
        deformation: &Handle<SubElementField>,
        prev: Option<&Handle<SubElementField>>,
        material: Option<&Handle<SubElementField>>,
        dt: Option<f64>,
    ) -> Result<SubElementField> {
        self.as_kind()
            .as_domain()
            .ok_or_else(|| {
                crate::error::PyrucastError::Message(format!(
                    "{}: no behaviour — integrate_behavior is undefined",
                    self.as_kind().label()
                ))
            })?
            .integrate_behavior(deformation, prev, material, dt)
    }

    /// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of this sub-model (Cast3m `BSIG`).
    /// The caller ([`crate::ops::node_field::internal_forces`]) supplies the
    /// matching per-zone `stress` (this sub-model's
    /// [`integrate_behavior`](Self::integrate_behavior) output).
    pub(crate) fn build_internal_forces(
        &self,
        stress: &Handle<SubElementField>,
    ) -> Result<crate::containers::node_field::SubNodeField> {
        self.as_kind().build_internal_forces(stress)
    }
}

impl fmt::Debug for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubModel")
            .field("kind", &self.as_kind().label())
            .field("physics", &self.as_kind().physics())
            .finish()
    }
}

impl fmt::Display for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_kind().display())
    }
}

impl crate::dump::Dump for SubModel {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        self.as_kind().render(opts)
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
    /// Memoised global CSR sparsity, **one slot per [`MatrixKind`]** (see
    /// [`Model::matrix_pattern`]). Each kind has its own block topology (mass and
    /// stiffness may span different sub-models), hence its own sparsity. Derived
    /// state: never serialized, and cleared whenever the model changes
    /// (`add_sub` → `post_push`), since a new sub-model changes the DOF layout.
    #[serde(skip)]
    matrix_patterns: [OnceLock<Arc<AssemblyPattern>>; MatrixKind::COUNT],
}

crate::impl_aggregate!(Model, SubModel, sub_model, "sub-model(s)", {
    fn post_push(&mut self) {
        // The block set changed ⇒ every cached sparsity is stale.
        self.matrix_patterns = Default::default();
    }
});
crate::impl_aggregate_dump!(Model);

impl Model {
    /// The model's [`AssemblyPattern`] for `kind`, built once and reused. On the
    /// first call `build` runs and its result is cached; later calls (same
    /// model + kind, e.g. a Newton loop re-assembling with new materials) return
    /// the cached pattern without touching the store — this is what makes
    /// repeated assembly scale, the sparsity being material-independent. The
    /// cache is cleared by `add_sub` (via `post_push`).
    pub(crate) fn matrix_pattern(
        &self,
        kind: MatrixKind,
        build: impl FnOnce() -> Result<AssemblyPattern>,
    ) -> Result<Arc<AssemblyPattern>> {
        let cell = &self.matrix_patterns[kind.index()];
        if let Some(p) = cell.get() {
            return Ok(p.clone());
        }
        let p = Arc::new(build()?);
        // A concurrent first caller may have won the race; either Arc is a valid
        // pattern for this (unchanged) model, so keep whichever landed.
        let _ = cell.set(p.clone());
        Ok(p)
    }
}

impl Model {
    /// Heat-conduction `Model` spanning **every** subspace of `fes` — one
    /// [`SubModel::HeatConduction`] sub-model per [`SubFiniteElementSpace`].
    ///
    /// This is the parent-level named constructor (see `CONVENTIONS.md`):
    /// it consumes the FE-space *parent* and returns a `Model`, so the
    /// caller never builds a `SubModel` by hand. A single-subspace `fes`
    /// yields the unit case; several subspaces yield one zone each.
    /// Compose heterogeneous physics with `union` (Python `|`), e.g.
    /// `Model::heat_conduction(&fes)?.union(&Model::dirichlet(...)?)?`.
    pub fn heat_conduction(fes: &FiniteElementSpace) -> Result<Self> {
        Self::heat_conduction_with_symmetry(fes, MaterialSymmetry::Isotropic)
    }

    /// Heat-conduction `Model` spanning **every** subspace of `fes`, with an
    /// explicit material symmetry. Parent-level named constructor; the
    /// conductivity (`k`, or `k_1…` / `k_11…` plus the material axes) is supplied
    /// at assembly time.
    pub fn heat_conduction_with_symmetry(
        fes: &FiniteElementSpace,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        let mut model = Self::empty();
        for sub in fes {
            model.add_sub(insert(SubModel::heat_conduction_with_symmetry(
                sub.clone(),
                symmetry,
            )?))?;
        }
        Ok(model)
    }

    /// Fickian-diffusion `Model` spanning **every** subspace of `fes` (same
    /// symmetry for all). Parent-level named constructor; the diffusivity is
    /// supplied at assembly time.
    pub fn fick(fes: &FiniteElementSpace, symmetry: MaterialSymmetry) -> Result<Self> {
        let mut model = Self::empty();
        for sub in fes {
            model.add_sub(insert(SubModel::fick(sub.clone(), symmetry)?))?;
        }
        Ok(model)
    }

    /// Interface-exchange `Model` between two conforming boundary FE spaces —
    /// one [`SubModel::InterfaceTransfer`] per subspace pair, taken in order.
    /// Parent-level named constructor; `h` is supplied at assembly time.
    ///
    /// The two spaces must hold the **same number** of subspaces: an interface
    /// pairs zone with zone, and a mismatch is a modelling error, not something
    /// to resolve by convention.
    pub fn interface_transfer(
        side_a: &FiniteElementSpace,
        side_b: &FiniteElementSpace,
        kind: interface_transfer::TransferKind,
        tol: f64,
    ) -> Result<Self> {
        if side_a.len() != side_b.len() {
            return Err(PyrucastError::Message(format!(
                "interface_transfer: the two sides must hold the same number of FE subspaces \
                 — {} facing {}",
                side_a.len(),
                side_b.len()
            )));
        }
        let mut model = Self::empty();
        for (a, b) in side_a.into_iter().zip(side_b) {
            model.add_sub(insert(SubModel::interface_transfer(
                a.clone(),
                b.clone(),
                kind,
                tol,
            )?))?;
        }
        Ok(model)
    }

    /// Surface-convection (Robin / film) `Model` spanning **every** subspace of
    /// a *boundary* `fes` — one [`SubModel::Convection`] per
    /// [`SubFiniteElementSpace`]. Parent-level named constructor; the film
    /// coefficient `"h"` is supplied at assembly time. Couples into a heat
    /// conduction model on the shared `"T"`/`"q"` DOFs:
    /// `Model::heat_conduction(&bulk)?.union(&Model::convection(&boundary)?)?`.
    pub fn convection(fes: &FiniteElementSpace) -> Result<Self> {
        let mut model = Self::empty();
        for sub in fes {
            model.add_sub(insert(SubModel::convection(sub.clone())?))?;
        }
        Ok(model)
    }

    /// Truss / bar `Model` spanning **every** subspace of `fes` — one
    /// [`SubModel::Truss`] per [`SubFiniteElementSpace`]. Parent-level named
    /// constructor; material (`E`, `A`) is supplied at assembly time.
    pub fn truss(fes: &FiniteElementSpace) -> Result<Self> {
        let mut model = Self::empty();
        for sub in fes {
            model.add_sub(insert(SubModel::truss(sub.clone())?))?;
        }
        Ok(model)
    }

    /// Linear-elasticity `Model` spanning **every** subspace of `fes` (same
    /// 2-D/3-D `model` for all). Parent-level named constructor; material
    /// (`E`, `nu`) is supplied at assembly time.
    pub fn elasticity(fes: &FiniteElementSpace, model: ElasticityModel) -> Result<Self> {
        Self::elasticity_with_symmetry(fes, model, MaterialSymmetry::Isotropic)
    }

    /// Linear-elasticity `Model` spanning **every** subspace of `fes`, with an
    /// explicit material symmetry. Parent-level named constructor; the elastic
    /// constants and, for an oriented material, its axes are supplied at assembly
    /// time.
    pub fn elasticity_with_symmetry(
        fes: &FiniteElementSpace,
        model: ElasticityModel,
        symmetry: MaterialSymmetry,
    ) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::elasticity_with_symmetry(
                sub.clone(),
                model,
                symmetry,
            )?))?;
        }
        Ok(out)
    }

    /// Perfect-plasticity `Model` spanning **every** subspace of `fes` (same
    /// 2-D/3-D `model` for all). Parent-level named constructor; material
    /// (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time.
    pub fn plasticity(fes: &FiniteElementSpace, model: ElasticityModel) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::plasticity(sub.clone(), model)?))?;
        }
        Ok(out)
    }

    /// Mazars-damage `Model` spanning **every** subspace of `fes` (same 2-D/3-D
    /// `model` for all). Parent-level named constructor; material
    /// (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at assembly
    /// / integration time.
    pub fn mazars(fes: &FiniteElementSpace, model: ElasticityModel) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::mazars(sub.clone(), model)?))?;
        }
        Ok(out)
    }

    /// Timoshenko-beam `Model` spanning **every** subspace of `fes`.
    /// Parent-level named constructor; material (`E`, `I`, `G`, `A_s`) is
    /// supplied at assembly time.
    pub fn timoshenko(fes: &FiniteElementSpace) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::timoshenko(sub.clone())?))?;
        }
        Ok(out)
    }

    /// Planar-frame `Model` spanning **every** subspace of `fes`. Parent-level
    /// named constructor; material (`E`, `A`, `I`, `G`, `A_s`) is supplied at
    /// assembly time.
    pub fn frame(fes: &FiniteElementSpace) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::frame(sub.clone())?))?;
        }
        Ok(out)
    }

    /// 3-D Timoshenko-frame `Model` spanning **every** subspace of `fes`.
    /// Parent-level named constructor; material
    /// (`E, A, I_y, I_z, J, G, A_sy, A_sz`) is supplied at assembly time.
    pub fn frame3d(fes: &FiniteElementSpace) -> Result<Self> {
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::frame3d(sub.clone())?))?;
        }
        Ok(out)
    }

    /// Dirichlet `Model` (a single sub-model) constraining `imposed_variable`
    /// on the nodes of `imposed_mesh` via Lagrange multipliers carried by
    /// `multiplier_mesh`. Parent-level named constructor — see
    /// [`SubModel::dirichlet`] for the semantics of the four variable names
    /// and the two meshes.
    #[allow(clippy::too_many_arguments)]
    pub fn dirichlet(
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: RelationSense,
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::dirichlet(
            imposed_variable,
            target_dual,
            imposed_mesh,
            multiplier_mesh,
            multiplier,
            imposed_value,
            sense,
        )?))?;
        Ok(model)
    }

    /// Multi-point constraint `Model` (a single sub-model) imposing a linear
    /// relation per relation via Lagrange multipliers. Parent-level named
    /// constructor — see [`SubModel::mpc`] and [`mpc::Mpc::new`] for the
    /// mesh-per-term layout and the two variable names.
    pub fn mpc(
        terms: Vec<mpc::MpcTerm>,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: RelationSense,
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::mpc(
            terms,
            multiplier_mesh,
            multiplier,
            imposed_value,
            sense,
        )?))?;
        Ok(model)
    }

    /// Embedded (immersed) constraint `Model` (a single sub-model) tying each
    /// node of `immersed` to the interpolation of `host` at that node, for every
    /// `(variable, target_dual)` in `components`. Parent-level named constructor
    /// — see [`SubModel::embedded`] and [`embedded::Embedded::new`] for the
    /// coupling weights, the per-component variable names and the errors.
    #[allow(clippy::too_many_arguments)]
    pub fn embedded(
        immersed: &Mesh,
        host: &Mesh,
        components: Vec<(String, String)>,
        multipliers: Option<Vec<String>>,
        imposed_values: Option<Vec<String>>,
        tol: Option<f64>,
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::embedded(
            immersed,
            host,
            components,
            multipliers,
            imposed_values,
            tol,
        )?))?;
        Ok(model)
    }

    /// Node-to-surface contact `Model` (a single sub-model) preventing the
    /// nodes of `slave` from penetrating the oriented `master` surface.
    /// Parent-level named constructor — see [`SubModel::contact`] and
    /// [`contact::Contact::new`] for the pairing, the normal coupling and the
    /// errors. Solve with [`unilateral`](crate::ops::solver::unilateral).
    pub fn contact(
        slave: &Mesh,
        master: &Mesh,
        components: Vec<(String, String)>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::contact(
            slave,
            master,
            components,
            multiplier,
            imposed_value,
        )?))?;
        Ok(model)
    }

    /// The contact right-hand side `−g₀`: a [`NodeField`] carrying, at each
    /// contact multiplier node's `imposed_value` slot, minus the initial signed
    /// gap of its relation — the load a contact model needs so that
    /// non-penetration reads `g₀ + C·u ≥ 0`. Merge it into the global load with
    /// `|`. The model must hold **exactly one** [`SubModel::Contact`]; pairs
    /// initially touching contribute `0` (omitting this helper entirely treats
    /// *every* pair as touching).
    pub fn contact_gaps(&self) -> Result<NodeField> {
        let mut found: Option<Handle<SubModel>> = None;
        for h in self {
            if matches!(&*read(h)?, SubModel::Contact(_)) {
                if found.is_some() {
                    return Err(PyrucastError::Message(
                        "contact_gaps: model holds several contact sub-models; call it on \
                         a single-contact model (the `contact` object)"
                            .into(),
                    ));
                }
                found = Some(h.clone());
            }
        }
        let handle = found.ok_or_else(|| {
            PyrucastError::Message("contact_gaps: model holds no contact sub-model".into())
        })?;
        let sub = read(&handle)?;
        let SubModel::Contact(c) = &*sub else {
            unreachable!("filtered on Contact above");
        };
        let pairs: Vec<(usize, f64)> = c
            .gaps()
            .iter()
            .enumerate()
            .map(|(i, &g0)| (i, -g0))
            .collect();
        sub.constraint_rhs_by_index(&pairs)
    }

    /// POI1 [`Mesh`] of every constraint multiplier node across the model
    /// (shares the multiplier submeshes — zero-copy). Empty for a model with no
    /// constraint. The handle to the multipliers of a constraint that minted
    /// them itself (e.g. [`embedded`](Self::embedded), whose multiplier mesh is
    /// internal): read the nodes or build the load field on it.
    pub fn multiplier_mesh(&self) -> Result<Mesh> {
        let mut mesh = Mesh::empty();
        for h in self {
            if let Some(constraint) = read(h)?.as_kind().as_constraint() {
                for sm in constraint.multiplier_mesh() {
                    mesh.add_sub(sm.clone())?;
                }
            }
        }
        Ok(mesh)
    }

    /// The dual (residual) variable conjugate to a primal `variable`, searched
    /// across all sub-models (first match), or `None` if no sub-model declares
    /// it. A helper for the MPC mise-en-donnée: `model.dual_of("ux")` returns the
    /// dual to pass in an [`mpc::MpcTerm`].
    pub fn dual_of(&self, variable: &str) -> Result<Option<String>> {
        for h in self {
            if let Some(dual) = read(h)?.dual_of(variable) {
                return Ok(Some(dual));
            }
        }
        Ok(None)
    }

    /// Build the constraint load (right-hand side) from `(constrained_node, g)`
    /// pairs — the model-level entry point of the mise-en-donnée helper. The
    /// model must hold **exactly one** constraint sub-model (as returned by the
    /// `dirichlet` / `mpc` constructors); it delegates to
    /// [`SubModel::constraint_rhs`], whose doc describes the node keying and the
    /// returned field.
    pub fn constraint_rhs(&self, imposed: &[(NodeId, f64)]) -> Result<NodeField> {
        read(&self.sole_constraint()?)?.constraint_rhs(imposed)
    }

    /// Build the constraint load from `(relation_index, g)` pairs — the
    /// model-level entry point that delegates to
    /// [`SubModel::constraint_rhs_by_index`] (relation keyed by its index rather
    /// than a node). The model must hold **exactly one** constraint sub-model.
    pub fn constraint_rhs_by_index(&self, imposed: &[(usize, f64)]) -> Result<NodeField> {
        read(&self.sole_constraint()?)?.constraint_rhs_by_index(imposed)
    }

    /// The model's single constraint sub-model, or an error when it holds none
    /// or several. Shared lookup of the `constraint_rhs*` model-level helpers.
    fn sole_constraint(&self) -> Result<Handle<SubModel>> {
        let mut found: Option<Handle<SubModel>> = None;
        for h in self {
            if read(h)?.as_kind().as_constraint().is_some() {
                if found.is_some() {
                    return Err(PyrucastError::Message(
                        "constraint_rhs: model holds several constraint sub-models; call it \
                         on a single-constraint model (e.g. the `dirichlet` / `mpc` object)"
                            .into(),
                    ));
                }
                found = Some(h.clone());
            }
        }
        found.ok_or_else(|| {
            PyrucastError::Message("constraint_rhs: model holds no constraint sub-model".into())
        })
    }

    /// Primal variable names — union over all sub-models, first-seen order.
    /// These are the **column labels** of the assembled matrices and the
    /// component names of the solution `SubNodeField`.
    pub fn primal_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in self {
            all.extend(read(h)?.primal_vars());
        }
        Ok(union_names(all))
    }

    /// Dual variable names — union over all sub-models, first-seen order.
    /// These are the **row labels** of the assembled matrices and the
    /// component names of the load `SubNodeField`.
    pub fn dual_vars(&self) -> Result<Vec<String>> {
        let mut all: Vec<String> = Vec::new();
        for h in self {
            all.extend(read(h)?.dual_vars());
        }
        Ok(union_names(all))
    }

    /// Rebuild the [`FiniteElementSpace`] this model integrates on: the
    /// behaviour subspace of each domain sub-model (constraints, which have
    /// none, are skipped), in order — **one subspace per domain sub-model**, a
    /// 1-to-1 correspondence (no deduplication). Subspace handles are **shared**,
    /// not copied. This is the exact inverse of the per-subspace constructors
    /// (`Model::elasticity(fes)` and friends build one sub-model per subspace).
    /// Errors if the model has no domain sub-model (nothing to integrate on).
    /// Combined with [`FiniteElementSpace::mesh`], lets a caller recover the FE
    /// space (and mesh) from the model alone; a single sub-model's own subspace
    /// is [`SubModel::behavior_fespace`].
    pub fn fespace(&self) -> Result<FiniteElementSpace> {
        let mut fes = FiniteElementSpace::empty();
        for h in self {
            if let Some(sub) = read(h)?.behavior_fespace() {
                fes.add_sub(sub)?;
            }
        }
        if fes.is_empty() {
            return Err(PyrucastError::Message(
                "Model::fespace: model has no domain sub-model to integrate on".into(),
            ));
        }
        Ok(fes)
    }

    /// A fresh [`Model`] holding only the sub-models **whose nature set contains**
    /// the given [`Physics`] (`model.filter(Physics::Mechanical)` → every
    /// sub-model that is at least mechanical, including a coupled one).
    ///
    /// Sub-model order is preserved and the handles are **shared** (refcount
    /// bump, no deep copy) via [`Aggregate::subset`];
    /// the result may be empty. The matrix-side counterpart is
    /// [`Matrix::filter`](crate::containers::matrix::Matrix::filter).
    pub fn filter(&self, physics: Physics) -> Result<Model> {
        let mut indices: Vec<usize> = Vec::new();
        for (i, h) in self.iter().enumerate() {
            if read(h)?.physics().contains(&physics) {
                indices.push(i);
            }
        }
        self.subset(indices)
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
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::element_field::ElementField;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::store::{insert, write};

    /// Returns `(coords, a_id, b_id, model, materials)`.
    fn build_seg2_heat_model(
        length: f64,
        k: f64,
        dirichlet_at_left: bool,
    ) -> (Handle<Coords>, NodeId, NodeId, Model, ElementField) {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[length]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();

        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", k).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        if dirichlet_at_left {
            let imposed =
                Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
            let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
            model
                .add_sub(insert(
                    SubModel::dirichlet(
                        "T".into(),
                        "q".into(),
                        &imposed,
                        &multiplier,
                        None,
                        None,
                        Default::default(),
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        (coords, a.id(), b.id(), model, materials)
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
        // Dual side: "q" from heat conduction + "imposed_T" (the Dirichlet
        // dual — a distinct name, no longer colliding with the primal "T").
        assert_eq!(
            model.dual_vars().unwrap(),
            vec!["q".to_string(), "imposed_T".to_string()]
        );
    }

    #[test]
    fn physics_nature_per_submodel() {
        // Heat conduction (Thermal) + a Dirichlet constraint (Constraint).
        let (_cfg, _, _, model, _mat) = build_seg2_heat_model(1.0, 1.0, true);
        let natures: Vec<Vec<Physics>> = model
            .iter()
            .map(|h| read(h).unwrap().physics().to_vec())
            .collect();
        assert_eq!(
            natures,
            vec![vec![Physics::Thermal], vec![Physics::Constraint]]
        );
    }

    #[test]
    fn filter_selects_submodels_by_physics() {
        let (_cfg, _, _, model, _mat) = build_seg2_heat_model(1.0, 1.0, true);
        assert_eq!(model.len(), 2);

        let thermal = model.filter(Physics::Thermal).unwrap();
        assert_eq!(thermal.len(), 1);
        assert_eq!(
            read(&thermal.get(0).unwrap()).unwrap().physics(),
            &[Physics::Thermal]
        );

        let constraint = model.filter(Physics::Constraint).unwrap();
        assert_eq!(constraint.len(), 1);
        assert_eq!(
            read(&constraint.get(0).unwrap()).unwrap().physics(),
            &[Physics::Constraint]
        );

        // A nature no sub-model has yields an empty model.
        let mechanical = model.filter(Physics::Mechanical).unwrap();
        assert_eq!(mechanical.len(), 0);
    }

    #[test]
    fn assembled_blocks_carry_physics_and_matrix_filters() {
        use crate::store::read as store_read;
        // Heat conduction (one computed block) + Dirichlet (a literal C/Cᵀ pair).
        let (_cfg, _, _, model, materials) = build_seg2_heat_model(1.0, 1.0, true);
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();

        // Every assembled block is tagged (non-empty), computed and literal alike.
        for h in &k {
            assert!(
                !store_read(h).unwrap().physics().is_empty(),
                "assembled block is tagged"
            );
        }
        // The matrix as a whole reports both natures present.
        let present = k.physics().unwrap();
        assert!(present.contains(&Physics::Thermal));
        assert!(present.contains(&Physics::Constraint));

        // The constraint filter keeps only the Dirichlet C/Cᵀ pair.
        let constraint = k.filter(Physics::Constraint).unwrap();
        assert_eq!(constraint.len(), 2);
        for h in &constraint {
            assert_eq!(store_read(h).unwrap().physics(), &[Physics::Constraint]);
        }

        // The thermal filter keeps only the heat-conduction block.
        let thermal = k.filter(Physics::Thermal).unwrap();
        assert_eq!(thermal.len(), 1);
    }

    /// Heat conduction on `[0, L]` with one SEG2 of length L and k = 1:
    /// `K_local = (k/L) [[1, -1], [-1, 1]]` (analytical, see Hughes 1.4).
    #[test]
    fn heat_conduction_assembles_analytical_seg2_stiffness() {
        let length = 2.0;
        let k_val = 1.5;
        let (_cfg, a_id, b_id, model, materials) = build_seg2_heat_model(length, k_val, false);
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();

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
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();
        let mut mat = SubElementField::new(sub.clone(), vec!["k".into()]).unwrap();
        mat.set_uniform("k", 1.0).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
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
        let (coords, a_id, _b_id, model, materials) = build_seg2_heat_model(1.0, 1.0, true);

        // The Coords grew by one node (the multiplier).
        let n_nodes = read(&coords).unwrap().node_count();
        assert_eq!(n_nodes, 3);

        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();
        // 2 real "q" rows + 1 multiplier "T" row = 3 rows.
        // 2 real "T" cols + 1 multiplier "lambda_T" col = 3 cols.
        assert_eq!(k.n_rows().unwrap(), 3);
        assert_eq!(k.n_cols().unwrap(), 3);

        // Find the multiplier node id: the only NodeId that appears in
        // a row labelled "imposed_T" of K.
        let row_dofs = k.row_dofs().unwrap();
        let mult = row_dofs
            .iter()
            .find(|(_, name)| name == "imposed_T")
            .expect("multiplier row missing")
            .0;

        // C entry: (mult, "imposed_T") × (a_id, "T") = 1
        assert_eq!(k.get(mult, "imposed_T", a_id, "T").unwrap(), 1.0);
        // Cᵀ entry: (a_id, "q") × (mult, "lambda_T") = 1
        assert_eq!(k.get(a_id, "q", mult, "lambda_T").unwrap(), 1.0);
        // Ensure lambda_T appears as a column.
        let col_dofs = k.col_dofs().unwrap();
        let lambda_col_present = col_dofs
            .iter()
            .any(|(n, name)| name == "lambda_T" && *n == mult);
        assert!(lambda_col_present);
    }

    /// The multiplier nodes (supplied by the user via `multiplier_mesh`) stay
    /// alive as long as **either** the user's mesh **or** the sub-model holds
    /// their submesh; once both are gone the node is released and collectable.
    /// The sub-model creates no node and never mutates the Coords.
    #[test]
    fn multiplier_nodes_live_with_their_mesh() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();

        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        // The multiplier node is owned by the multiplier submesh (refcount 1;
        // the transient Node below is dropped at the end of the statement).
        let mult_id = multiplier.node(0, 0, 0).unwrap().id();
        assert_eq!(read(&coords).unwrap().refcount(mult_id), 1);

        let sub = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        // Sharing the submesh handles does not touch node refcounts.
        assert_eq!(read(&coords).unwrap().refcount(mult_id), 1);
        assert_eq!(sub.multiplier_nodes().unwrap(), vec![mult_id]);

        // Drop the user's multiplier mesh: the sub-model still holds the
        // submesh, so the node lives on.
        drop(multiplier);
        assert_eq!(read(&coords).unwrap().refcount(mult_id), 1);

        // Drop the sub-model too: the last holder of the multiplier submesh is
        // gone ⇒ the node is released and collectable.
        drop(sub);
        assert_eq!(read(&coords).unwrap().refcount(mult_id), 0);
        assert_eq!(write(&coords).unwrap().gc(), 1);
    }

    #[test]
    fn dirichlet_empty_imposed_mesh_rejected() {
        let empty = Mesh::empty();
        assert!(SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &empty,
            &empty,
            None,
            None,
            Default::default()
        )
        .is_err());
    }

    #[test]
    fn heat_conduction_errors_on_missing_k_component() {
        // Material has only "rho_cp", not "k": stiffness assembly must fail.
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();
        let mat = SubElementField::new(sub.clone(), vec!["rho_cp".into()]).unwrap();
        let mut materials = ElementField::empty();
        materials.add_sub(insert(mat)).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        assert!(crate::ops::matrix::stiffness(&model, &materials).is_err());
    }

    #[test]
    fn empty_model_has_no_vars_and_empty_mass() {
        let model = Model::empty();
        assert_eq!(model.primal_vars().unwrap(), Vec::<String>::new());
        assert_eq!(model.dual_vars().unwrap(), Vec::<String>::new());
        let m = crate::ops::matrix::mass(&model, &ElementField::empty()).unwrap();
        assert_eq!(m.n_rows().unwrap(), 0);
        assert_eq!(m.n_cols().unwrap(), 0);
    }

    /// Parent-level `Model::heat_conduction(&fes)` builds one sub-model per
    /// subspace and matches the hand-rolled `SubModel` + `add_sub` path, and
    /// `union` composes it with a Dirichlet `Model`.
    #[test]
    fn parent_constructors_span_subspaces_and_compose() {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();

        // Two SEG2 zones, one SubMesh each → fes with two subspaces.
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        // One HeatConduction sub-model per subspace.
        let hc = Model::heat_conduction(&fes).unwrap();
        assert_eq!(hc.len(), 2);
        assert_eq!(hc.primal_vars().unwrap(), vec!["T".to_string()]);
        assert_eq!(hc.dual_vars().unwrap(), vec!["q".to_string()]);

        // Compose with a Dirichlet model via `union` (Python `|`).
        let imp = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&n0)).unwrap());
        let mlt = crate::ops::mesh::barycenter(&imp).unwrap();
        let dir = Model::dirichlet(
            "T".into(),
            "q".into(),
            &imp,
            &mlt,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        assert_eq!(dir.len(), 1);
        let full = hc.union(&dir).unwrap();
        assert_eq!(full.len(), 3);
        assert_eq!(
            full.primal_vars().unwrap(),
            vec!["T".to_string(), "lambda_T".to_string()]
        );
    }

    /// Single-subspace `fes` → unit `Model` (the common case).
    #[test]
    fn parent_heat_conduction_unit_case() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let model = Model::heat_conduction(&fes).unwrap();
        assert_eq!(model.len(), 1);
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
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();

        // Zone A: SEG2 on [0, 1]. Zone B: SEG2 on [1, 2]. Each as its own
        // SubMesh inside one Mesh.
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();

        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub_a = fes.get(0).unwrap();
        let sub_b = fes.get(1).unwrap();

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

        let k = crate::ops::matrix::stiffness(&model, &materials).unwrap();

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
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();
        let hc = SubModel::heat_conduction(sub).unwrap();
        assert_eq!(hc.material_components(), Some(&["k"][..]));

        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        let dir = SubModel::dirichlet(
            "T".into(),
            "q".into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
        .unwrap();
        assert!(dir.material_components().is_none());
    }

    /// `crate::ops::matrix::stiffness` must fail with a clear error when no
    /// SubElementField matches a HeatConduction's FE subspace.
    #[test]
    fn assemble_errors_when_no_material_matches_fespace() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let sub = fes.get(0).unwrap();

        // Empty ElementField — no SubElementField matches anything.
        let materials = ElementField::empty();
        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::heat_conduction(sub).unwrap()))
            .unwrap();
        let err = crate::ops::matrix::stiffness(&model, &materials).unwrap_err();
        assert!(format!("{}", err).contains("no SubElementField"));
    }

    // ── constraint_rhs (mise-en-donnée helper) ──────────────────────────────

    /// Dirichlet: keying by the constrained node writes `g` at the multiplier
    /// node, on the constraint's dual component (`imposed_T`), and the field
    /// lives on the multiplier — not the constrained — node.
    #[test]
    fn constraint_rhs_dirichlet_writes_g_at_multiplier() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords, &[0.0]).unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        let dirichlet = Model::dirichlet(
            "T".into(),
            "q".into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
        .unwrap();

        let rhs = dirichlet.constraint_rhs(&[(a.id(), 3.0)]).unwrap();
        let mult_id = multiplier.node(0, 0, 0).unwrap().id();
        assert_eq!(rhs.value(mult_id, "imposed_T").unwrap(), 3.0);
        // Written on the multiplier node, not the constrained one.
        assert!(rhs.value(a.id(), "imposed_T").is_err());
    }

    /// MPC: a relation is keyed by *any* of its term nodes, both resolving to
    /// the same multiplier node and value. Non-cited relations default to 0.
    #[test]
    fn constraint_rhs_mpc_keyed_by_any_term_node() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords, &[1.0]).unwrap();
        let mesh_a =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let mesh_b =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&b)).unwrap());
        let mult = crate::ops::mesh::barycenter(&mesh_b).unwrap();
        let terms = vec![
            crate::models::mpc::MpcTerm::new(&mesh_b, "T".into(), "q".into(), 1.0).unwrap(),
            crate::models::mpc::MpcTerm::new(&mesh_a, "T".into(), "q".into(), -1.0).unwrap(),
        ];
        let model = Model::mpc(terms, &mult, None, None, Default::default()).unwrap();
        let mult_id = mult.node(0, 0, 0).unwrap().id();

        let rhs = model.constraint_rhs(&[(b.id(), 2.5)]).unwrap();
        assert_eq!(rhs.value(mult_id, "mpc_rhs").unwrap(), 2.5);
        // The other term node keys the same relation → same field.
        let rhs2 = model.constraint_rhs(&[(a.id(), 2.5)]).unwrap();
        assert_eq!(rhs2.value(mult_id, "mpc_rhs").unwrap(), 2.5);
    }

    /// A model without any constraint sub-model rejects the call.
    #[test]
    fn constraint_rhs_errors_without_constraint() {
        let (_c, _a, _b, model, _m) = build_seg2_heat_model(1.0, 1.0, false);
        assert!(model.constraint_rhs(&[]).is_err());
    }

    /// A node keying two *distinct* relations (here node `a`, a term of both) is
    /// ambiguous and rejected; a node keying a single relation still resolves.
    #[test]
    fn constraint_rhs_rejects_node_keying_several_relations() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let m0 = Node::create_in(coords.clone(), &[0.5]).unwrap();
        let m1 = Node::create_in(coords.clone(), &[1.5]).unwrap();

        // Term 1 uses `a` in *both* relations; term 2 uses `b` then `c`.
        let mut t1 = SubMesh::new(coords.clone(), ElementType::POI1);
        t1.add_cell(&[a.id()]).unwrap();
        t1.add_cell(&[a.id()]).unwrap();
        let t1_mesh = Mesh::from_submesh(t1);
        let mut t2 = SubMesh::new(coords.clone(), ElementType::POI1);
        t2.add_cell(&[b.id()]).unwrap();
        t2.add_cell(&[c.id()]).unwrap();
        let t2_mesh = Mesh::from_submesh(t2);
        let mut mm = SubMesh::new(coords, ElementType::POI1);
        mm.add_cell(&[m0.id()]).unwrap();
        mm.add_cell(&[m1.id()]).unwrap();
        let mult = Mesh::from_submesh(mm);

        let terms = vec![
            crate::models::mpc::MpcTerm::new(&t1_mesh, "T".into(), "q".into(), 1.0).unwrap(),
            crate::models::mpc::MpcTerm::new(&t2_mesh, "T".into(), "q".into(), -1.0).unwrap(),
        ];
        let model = Model::mpc(terms, &mult, None, None, Default::default()).unwrap();

        // `a` keys both relations → ambiguous.
        assert!(model.constraint_rhs(&[(a.id(), 1.0)]).is_err());
        // `b` keys only the first relation → resolves to m0.
        let rhs = model.constraint_rhs(&[(b.id(), 1.0)]).unwrap();
        assert_eq!(rhs.value(m0.id(), "mpc_rhs").unwrap(), 1.0);
        assert_eq!(rhs.value(m1.id(), "mpc_rhs").unwrap(), 0.0);

        // Keying by *index* sidesteps the ambiguity: relation 1 → m1.
        let by_idx = model.constraint_rhs_by_index(&[(1, 4.0)]).unwrap();
        assert_eq!(by_idx.value(m1.id(), "mpc_rhs").unwrap(), 4.0);
        assert_eq!(by_idx.value(m0.id(), "mpc_rhs").unwrap(), 0.0);
    }

    /// `constraint_rhs_by_index` writes `g` at the relation's multiplier node and
    /// rejects an out-of-range index.
    #[test]
    fn constraint_rhs_by_index_writes_and_bounds_checks() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords, &[0.0]).unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        let dirichlet = Model::dirichlet(
            "T".into(),
            "q".into(),
            &imposed,
            &multiplier,
            None,
            None,
            Default::default(),
        )
        .unwrap();

        let mult_id = multiplier.node(0, 0, 0).unwrap().id();
        let rhs = dirichlet.constraint_rhs_by_index(&[(0, 2.0)]).unwrap();
        assert_eq!(rhs.value(mult_id, "imposed_T").unwrap(), 2.0);
        // Only one relation (index 0) → index 1 is out of range.
        assert!(dirichlet.constraint_rhs_by_index(&[(1, 2.0)]).is_err());
    }
}
