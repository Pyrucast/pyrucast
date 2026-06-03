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
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
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

    /// POI1 support owning this physics's multiplier nodes, for Lagrange
    /// variants (`Dirichlet`, …). `None` (default) for every physics that
    /// introduces no multipliers.
    fn multiplier_support(&self) -> Option<&Handle<SubMesh>> {
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
