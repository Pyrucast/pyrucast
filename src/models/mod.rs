//! Per-physics implementations of [`crate::containers::model::SubModel`]
//! variants.
//!
//! Each file here owns the **specifics** of one physics: a struct holding
//! its supports (FE spaces, materials, node sets) plus an [`impl Physics`]
//! carrying *all* of its behaviour — variable names, material contract,
//! local assembly, and rendering. The
//! [`crate::containers::model::SubModel`] enum exists **only** for storage
//! and serialization; it dispatches every call through
//! [`SubModel::as_physics`](crate::containers::model::SubModel::as_physics)
//! so no generic code (the assembler, `Dump`, …) ever needs a per-variant
//! `match`.
//!
//! # Adding a new physics
//!
//! 1. add `models/<name>.rs` with a struct + `impl Physics` (and a
//!    `new(...)` constructor doing any build-time work);
//! 2. add one variant to [`crate::containers::model::SubModel`];
//! 3. add one arm to
//!    [`SubModel::as_physics`](crate::containers::model::SubModel::as_physics);
//! 4. expose it via `Model::<name>` (Rust) and a `#[classmethod]` (Python).
//!
//! Everything else is generic. See the book chapter *« Ajouter une
//! physique »* for the full walkthrough.

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::SubMatrix;
use crate::containers::mesh::Mesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::store::Handle;

pub mod dirichlet;
pub mod heat_conduction;

/// The behaviour contract of one physics, co-located with its data struct.
///
/// Generic code calls these through
/// [`SubModel::as_physics`](crate::containers::model::SubModel::as_physics);
/// the [`SubModel`](crate::containers::model::SubModel) enum itself carries
/// no logic. Most methods have sensible defaults so a physics overrides
/// only what is specific to it (a plain volumetric physics typically
/// implements just `primal_vars`, `dual_vars`, `material_*`,
/// `build_stiffness_blocks`, `label` and `render`).
pub trait Physics {
    /// Primal variable names introduced by this physics (column labels).
    fn primal_vars(&self) -> Vec<String>;

    /// Dual variable names introduced by this physics (row labels).
    fn dual_vars(&self) -> Vec<String>;

    /// Material component names this physics requires, or `None` if it
    /// needs no material data. Default: `None`.
    fn material_components(&self) -> Option<&'static [&'static str]> {
        None
    }

    /// FE subspace on which this physics expects its material data, or
    /// `None` if it needs none. Default: `None`.
    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        None
    }

    /// POI1 mesh carrying this physics's multiplier nodes, for Lagrange
    /// variants (`Dirichlet`, …). `None` (default) for every physics that
    /// introduces no multipliers. Borrowed from the physics (the user supplied
    /// it); generic code clones it when an owned `Mesh` is needed.
    fn multiplier_mesh(&self) -> Option<&Mesh> {
        None
    }

    /// Build and fill the stiffness [`SubMatrix`] block(s) of this physics.
    /// `material` is `Some(_)` iff [`material_fespace`](Self::material_fespace)
    /// is `Some(_)` (the assembler guarantees it).
    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>>;

    /// Build and fill the mass [`SubMatrix`] block(s) of this physics.
    /// Default: no mass term (empty).
    fn build_mass_blocks(
        &self,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        Ok(Vec::new())
    }

    /// FE subspace this physics integrates its **constitutive behaviour**
    /// on, or `None` (default) for a physics that carries no behaviour
    /// (constraints such as `Dirichlet`).
    ///
    /// When `Some(_)`, the physics must implement
    /// [`integrate_behavior`](Self::integrate_behavior); its deformation
    /// input is produced geometrically by [`crate::ops::field::gradient`] /
    /// [`crate::ops::field::deformation`], and [`crate::ops::behavior`] uses
    /// this handle to pair the per-zone deformation field with its
    /// sub-model. For a plain volumetric physics it is the same FE subspace
    /// as [`material_fespace`](Self::material_fespace).
    fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        None
    }

    /// Integrate the constitutive law point-by-point (Cast3m `COMP` —
    /// « intégrer le comportement »).
    ///
    /// `input` carries, at every `(cell, Gauss)` point, the deformation
    /// measure (the temperature gradient `∇T` for heat conduction, the
    /// strain `ε` for elasticity, …) produced by a *geometric* operator —
    /// [`crate::ops::field::gradient`] / [`crate::ops::field::deformation`],
    /// independent of any model — followed by the input internal-state
    /// variables (`VAR0`). `material` is `Some(_)` iff this physics declares
    /// a [`material_fespace`](Self::material_fespace) (the operator
    /// guarantees it).
    ///
    /// Returns the **material-state** field: the dual flux/stress followed
    /// by the updated internal-state variables (`VAR1`). Where
    /// [`build_stiffness_blocks`](Self::build_stiffness_blocks) is the
    /// *linearization* of the law, this is its *exact* response: for a
    /// linear law the two agree (`∫ Bᵀ·flux = K·u`); a non-linear law
    /// departs from that tangent.
    ///
    /// Default: errors — a physics with no behaviour.
    fn integrate_behavior(
        &self,
        _input: &Handle<SubElementField>,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<SubElementField> {
        Err(PyrucastError::Message(format!(
            "{}: no behaviour — integrate_behavior is undefined",
            self.label()
        )))
    }

    /// Short type label, e.g. `"HeatConduction"` (used by `Debug` and the
    /// default `display`).
    fn label(&self) -> &'static str;

    /// One-line summary for `Display`. Default: `SubModel<{label}>`.
    fn display(&self) -> String {
        format!("SubModel<{}>", self.label())
    }

    /// Full multi-line rendering for [`crate::dump::Dump`].
    fn render(&self, opts: &DumpOptions) -> String;
}
