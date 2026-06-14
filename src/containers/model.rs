//! Physical model — orchestrator of sub-models that assemble into a
//! [`crate::containers::matrix::Matrix`].
//!
//! The model layer is the **physics-aware** counterpart of the
//! geometry layer (`Mesh`, `SubMesh`) and the interpolation layer
//! (`FiniteElementSpace`, `SubFiniteElementSpace`). A [`Model`] is an aggregate of
//! [`SubModel`]s; each [`SubModel`] is one physics instance (a variant of
//! the enum) that owns its supports and dispatches its behaviour through
//! the [`Physics`] trait. The Model is a pure orchestrator: it enumerates
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
//! ops::assemble (operators, not Model methods)
//! ├── stiffness(model, materials) -> Matrix   # rows: dual × cols: primal
//! └── mass(model)                 -> Matrix   # same DOF layout, may be empty
//!
//! SubModel  (enum : stockage + sérialisation ; dispatch via as_physics())
//! ├── HeatConduction(HeatConduction)
//! ├── Dirichlet(Dirichlet)             # constraint = Lagrange multiplier
//! └── ...
//!
//! Physics  (trait : tout le comportement, co-localisé par physique)
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
//! use pyrucast::containers::mesh::{Configuration, NodeId};
//! use pyrucast::containers::element_field::SubElementField;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::field::SubField;
//! use pyrucast::containers::model::{Model, SubModel};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::ops::assemble;
//! use pyrucast::ops::mesher;
//! use pyrucast::store::insert;
//!
//! // 1-D Configuration with two nodes spanning [0, 1].
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
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
//! let multiplier = mesher::barycenter(&imposed).unwrap();
//! model
//!     .add_sub(insert(
//!         SubModel::dirichlet("T".into(), "q".into(), &imposed, &multiplier, None, None)
//!             .unwrap(),
//!     ))
//!     .unwrap();
//!
//! let k = assemble::stiffness(&model, &materials).unwrap();
//! // 2 real DOFs ("T") + 1 multiplier DOF + 2 real rows ("q") + 1 multiplier row.
//! assert_eq!(k.n_rows().unwrap(), 3);
//! assert_eq!(k.n_cols().unwrap(), 3);
//! ```

use crate::containers::mesh::NodeId;
use crate::containers::element_field::SubElementField;
use crate::error::Result;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::matrix::SubMatrix;
use crate::containers::mesh::Mesh;
use crate::aggregate::Aggregate;
use crate::models::elasticity::ElasticityModel;
use crate::models::{dirichlet, elasticity, frame, heat_conduction, timoshenko, truss, Physics};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SubModel ──────────────────────────────────────────────────────────────

/// One physics instance, bound to its supports (FE spaces, materials, node
/// sets). A [`Model`] is a `Vec<Handle<SubModel>>`.
///
/// This enum is a **pure storage + dispatch** shell: it derives
/// `Serialize`/`Deserialize` (so models persist through the `bincode`
/// backbone), and forwards every behavioural call to the variant's
/// [`Physics`] implementation through [`SubModel::as_physics`]. All physics
/// logic lives in the per-variant structs under [`crate::models`].
///
/// Adding a physics means adding **one variant here** and **one arm to
/// [`SubModel::as_physics`]** — no other site in this file changes.
#[derive(Serialize, Deserialize)]
pub enum SubModel {
    /// Linear heat conduction — see [`heat_conduction::HeatConduction`].
    HeatConduction(heat_conduction::HeatConduction),
    /// Dirichlet constraint via Lagrange multipliers — see
    /// [`dirichlet::Dirichlet`].
    Dirichlet(dirichlet::Dirichlet),
    /// Truss / bar (axial-force) element — see [`truss::Truss`].
    Truss(truss::Truss),
    /// Linear elasticity (2-D plane / 3-D solid) — see [`elasticity::Elasticity`].
    Elasticity(elasticity::Elasticity),
    /// Timoshenko beam (shear-deformable bending) — see [`timoshenko::Timoshenko`].
    Timoshenko(timoshenko::Timoshenko),
    /// Planar frame / portique (axial + bending + shear) — see [`frame::Frame`].
    Frame(frame::Frame),
}

impl SubModel {
    /// Borrow the variant as its [`Physics`] behaviour. This is the
    /// **only** per-variant `match` in the model layer; every generic
    /// method (variable names, material contract, assembly, rendering)
    /// dispatches through it.
    pub fn as_physics(&self) -> &dyn Physics {
        match self {
            SubModel::HeatConduction(p) => p,
            SubModel::Dirichlet(p) => p,
            SubModel::Truss(p) => p,
            SubModel::Elasticity(p) => p,
            SubModel::Timoshenko(p) => p,
            SubModel::Frame(p) => p,
        }
    }

    /// Heat-conduction sub-model on an FE subspace.
    ///
    /// Material data (conductivity `"k"`, …) is supplied separately at
    /// assembly time via [`crate::ops::assemble::stiffness`], keeping
    /// the model immutable and material-independent. See
    /// [`heat_conduction::HeatConduction::new`] for the support it builds.
    pub fn heat_conduction(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::HeatConduction(
            heat_conduction::HeatConduction::new(fespace)?,
        ))
    }

    /// Truss / bar sub-model on a `SEG2` FE subspace. Material data (`E`, `A`)
    /// is supplied at assembly time. See [`truss::Truss::new`].
    pub fn truss(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        Ok(SubModel::Truss(truss::Truss::new(fespace)?))
    }

    /// Linear-elasticity sub-model on an FE subspace, with the given 2-D/3-D
    /// model. Material data (`E`, `nu`) is supplied at assembly time. See
    /// [`elasticity::Elasticity::new`].
    pub fn elasticity(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
    ) -> Result<Self> {
        Ok(SubModel::Elasticity(elasticity::Elasticity::new(fespace, model)?))
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

    /// Dirichlet sub-model: enforce `imposed_variable = u_d` on the nodes of
    /// `imposed_mesh`, with multipliers living on `multiplier_mesh` and `u_d`
    /// supplied later by the user through the load `SubNodeField`.
    ///
    /// `target_dual` is the dual variable name of the target physics whose
    /// primal is being constrained (e.g. `"q"` for heat conduction, `"f_x"`
    /// for elasticity in `x`). `multiplier` / `imposed_value` default to
    /// `lambda_<imposed_variable>` / `imposed_<imposed_variable>` when `None`.
    /// See [`dirichlet::Dirichlet::new`].
    pub fn dirichlet(
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        Ok(SubModel::Dirichlet(dirichlet::Dirichlet::new(
            imposed_variable,
            target_dual,
            imposed_mesh,
            multiplier_mesh,
            multiplier,
            imposed_value,
        )?))
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
        if let Some(mesh) = self.as_physics().multiplier_mesh() {
            for sm in mesh {
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
        if let Some(src) = self.as_physics().multiplier_mesh() {
            for sm in src {
                mesh.add_sub(sm.clone())?;
            }
        }
        Ok(mesh)
    }

    /// FE subspace on which this sub-model expects its material data, or
    /// `None` if this physics doesn't need material data (e.g. `Dirichlet`).
    pub fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        self.as_physics().material_fespace()
    }

    /// Material component names this sub-model expects, or `None` if it
    /// doesn't need material data. Thin pass-through of
    /// [`Physics::material_components`].
    pub fn material_components(&self) -> Option<&'static [&'static str]> {
        self.as_physics().material_components()
    }

    /// Primal variable names introduced by this sub-model.
    pub fn primal_vars(&self) -> Vec<String> {
        self.as_physics().primal_vars()
    }

    /// Dual variable names introduced by this sub-model.
    pub fn dual_vars(&self) -> Vec<String> {
        self.as_physics().dual_vars()
    }

    /// Build and fill the stiffness [`SubMatrix`] block(s) for this
    /// sub-model. Pure dispatch to the physics's
    /// [`Physics::build_stiffness_blocks`]; the caller
    /// ([`crate::ops::assemble::stiffness`]) supplies `material` iff the
    /// physics declares a [`material_fespace`](Self::material_fespace).
    pub(crate) fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        self.as_physics().build_stiffness_blocks(material)
    }

    /// Whether this sub-model carries a constitutive behaviour that can be
    /// integrated via [`integrate_behavior`](Self::integrate_behavior) from a
    /// deformation field. `true` for volumetric physics, `false` for
    /// constraints (`Dirichlet`).
    pub fn has_behavior(&self) -> bool {
        self.as_physics().behavior_fespace().is_some()
    }

    /// FE subspace this sub-model integrates its behaviour on, or `None`
    /// for a constraint sub-model. The operators in
    /// [`crate::ops::behavior`] use it to pair the per-zone deformation
    /// field with its sub-model.
    pub fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        self.as_physics().behavior_fespace()
    }

    /// Integrate this sub-model's constitutive law (Cast3m `COMP`). The
    /// caller ([`crate::ops::behavior::integrate`]) supplies the matching
    /// per-zone deformation `input` (from [`crate::ops::field::gradient`] /
    /// [`crate::ops::field::deformation`]) and `material`.
    pub(crate) fn integrate_behavior(
        &self,
        input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<SubElementField> {
        self.as_physics().integrate_behavior(input, material)
    }
}

impl fmt::Debug for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubModel")
            .field("physics", &self.as_physics().label())
            .finish()
    }
}

impl fmt::Display for SubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_physics().display())
    }
}

impl crate::dump::Dump for SubModel {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        self.as_physics().render(opts)
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
    /// [`SubModel::HeatConduction`] sub-model per [`SubFiniteElementSpace`].
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
        let mut out = Self::empty();
        for sub in fes {
            out.add_sub(insert(SubModel::elasticity(sub.clone(), model)?))?;
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

    /// Dirichlet `Model` (a single sub-model) constraining `imposed_variable`
    /// on the nodes of `imposed_mesh` via Lagrange multipliers carried by
    /// `multiplier_mesh`. Parent-level named constructor — see
    /// [`SubModel::dirichlet`] for the semantics of the four variable names
    /// and the two meshes.
    pub fn dirichlet(
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        let mut model = Self::empty();
        model.add_sub(insert(SubModel::dirichlet(
            imposed_variable,
            target_dual,
            imposed_mesh,
            multiplier_mesh,
            multiplier,
            imposed_value,
        )?))?;
        Ok(model)
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
    use crate::containers::field::SubField;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::SubMesh;
    use crate::containers::element_field::ElementField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::containers::mesh::Node;
    use crate::ops::assemble;
    use crate::store::{insert, write};

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
            let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
            model
                .add_sub(insert(
                    SubModel::dirichlet(
                        "T".into(),
                        "q".into(),
                        &imposed,
                        &multiplier,
                        None,
                        None,
                    )
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
        // Dual side: "q" from heat conduction + "imposed_T" (the Dirichlet
        // dual — a distinct name, no longer colliding with the primal "T").
        assert_eq!(
            model.dual_vars().unwrap(),
            vec!["q".to_string(), "imposed_T".to_string()]
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
        let sub = fes.get(0).unwrap();
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
        let n_nodes = read(&cfg).unwrap().node_count();
        assert_eq!(n_nodes, 3);

        let k = assemble::stiffness(&model, &materials).unwrap();
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
    /// The sub-model creates no node and never mutates the Configuration.
    #[test]
    fn multiplier_nodes_live_with_their_mesh() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();

        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
        // The multiplier node is owned by the multiplier submesh (refcount 1;
        // the transient Node below is dropped at the end of the statement).
        let mult_id = multiplier.node(0, 0, 0).unwrap().id();
        assert_eq!(read(&cfg).unwrap().refcount(mult_id), 1);

        let sub =
            SubModel::dirichlet("T".into(), "q".into(), &imposed, &multiplier, None, None)
                .unwrap();
        // Sharing the submesh handles does not touch node refcounts.
        assert_eq!(read(&cfg).unwrap().refcount(mult_id), 1);
        assert_eq!(sub.multiplier_nodes().unwrap(), vec![mult_id]);

        // Drop the user's multiplier mesh: the sub-model still holds the
        // submesh, so the node lives on.
        drop(multiplier);
        assert_eq!(read(&cfg).unwrap().refcount(mult_id), 1);

        // Drop the sub-model too: the last holder of the multiplier submesh is
        // gone ⇒ the node is released and collectable.
        drop(sub);
        assert_eq!(read(&cfg).unwrap().refcount(mult_id), 0);
        assert_eq!(write(&cfg).unwrap().gc(), 1);
    }

    #[test]
    fn dirichlet_empty_imposed_mesh_rejected() {
        let empty = Mesh::empty();
        assert!(
            SubModel::dirichlet("T".into(), "q".into(), &empty, &empty, None, None).is_err()
        );
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
        let sub = fes.get(0).unwrap();
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
        assert_eq!(hc.len(), 2);
        assert_eq!(hc.primal_vars().unwrap(), vec!["T".to_string()]);
        assert_eq!(hc.dual_vars().unwrap(), vec!["q".to_string()]);

        // Compose with a Dirichlet model via `+` (merge).
        let imp = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&n0)).unwrap());
        let mlt = crate::ops::mesher::barycenter(&imp).unwrap();
        let dir = Model::dirichlet("T".into(), "q".into(), &imp, &mlt, None, None).unwrap();
        assert_eq!(dir.len(), 1);
        let full = (&hc + &dir).unwrap();
        assert_eq!(full.len(), 3);
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
        let sub = fes.get(0).unwrap();
        let hc = SubModel::heat_conduction(sub).unwrap();
        assert_eq!(hc.material_components(), Some(&["k"][..]));

        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
        let dir =
            SubModel::dirichlet("T".into(), "q".into(), &imposed, &multiplier, None, None)
                .unwrap();
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
        let sub = fes.get(0).unwrap();

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
