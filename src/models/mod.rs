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
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::node_field::SubNodeField;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::store::Handle;

/// Spatial-axis suffixes for the Voigt stress-component names read by the
/// continuum-mechanics [`Physics::internal_force_element`] default.
const VOIGT_AXES: [&str; 3] = ["x", "y", "z"];

pub mod dirichlet;
pub mod elasticity;
pub mod frame;
pub mod frame3d;
pub mod heat_conduction;
pub mod kernel;
pub mod mazars;
pub mod plasticity;
pub mod timoshenko;
pub mod truss;

pub use kernel::CellGeom;

/// Structural declaration a volumetric physics gives so the **global**
/// assembler ([`crate::ops::assemble::stiffness`]) can build its stiffness
/// contribution as a *computed* [`SubMatrix`] — a recipe, no eagerly
/// materialised values — and scatter it straight into the global CSR.
///
/// Every field mirrors, one-for-one, what the physics'
/// [`build_stiffness_blocks`](Physics::build_stiffness_blocks) would pass to
/// [`kernel::assemble_block`]. The
/// literal `build_stiffness_blocks` is **kept** alongside it as the bit-for-bit
/// equivalence reference. Volumetric blocks are square on a single support, so
/// one [`SubMesh`] gives both the row and column node sequence.
pub struct StiffnessLayout {
    /// FE subspaces the element kernel integrates over. **Give a `Vec`**: a
    /// single subspace for a plain volumetric physics, or several — sharing one
    /// submesh, differing only by quadrature — for a multi-quadrature element
    /// (a shear-deformable beam, a shell). The primary (index 0) drives the cell
    /// loop and the DOF numbering; [`element_matrix`](Physics::element_matrix)
    /// receives one [`CellGeom`] per subspace, in this order.
    pub fespaces: Vec<Handle<SubFiniteElementSpace>>,
    /// POI1 sub-mesh giving the block's row **and** column node sequence.
    pub support: Handle<SubMesh>,
    /// Row variable names (dual).
    pub dual_vars: Vec<String>,
    /// Column variable names (primal).
    pub primal_vars: Vec<String>,
    /// `(node_local, var)` ↔ matrix-index ordering.
    pub ordering: DofOrdering,
    /// Whether the block is numerically symmetric.
    pub symmetric: bool,
}

/// The behaviour contract of one physics, co-located with its data struct.
///
/// Generic code calls these through
/// [`SubModel::as_physics`](crate::containers::model::SubModel::as_physics);
/// the [`SubModel`](crate::containers::model::SubModel) enum itself carries
/// no logic. Most methods have sensible defaults so a physics overrides
/// only what is specific to it (a plain volumetric physics typically
/// implements just `primal_vars`, `dual_vars`, `material_*`,
/// `build_stiffness_blocks`, `label` and `render`).
pub trait Physics: Sync {
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

    /// Local element stiffness matrix of one cell — the pure, sequential kernel
    /// a physics author writes (the stiffness counterpart of
    /// [`integrate_point`](Self::integrate_point)). Fills `ke` (row-major,
    /// node-major / variable-minor: `ke[(li*n_dual+di) * n_cols_loc + (lj*n_primal+pj)]`)
    /// from the cell geometry and material. `material` is `Some(_)` iff the
    /// physics declares a [`material_fespace`](Self::material_fespace).
    ///
    /// `geoms` holds one [`CellGeom`] per FE subspace declared in
    /// [`stiffness_layout`](Self::stiffness_layout), in that order: a plain
    /// volumetric physics reads `geoms[0]`, a multi-quadrature element (a
    /// shear-deformable beam, a shell) reads each — e.g. `geoms[0]` full Gauss
    /// for bending, `geoms[1]` reduced for shear.
    ///
    /// It **never sees rayon, the store, or a lock**: the assembler drives it in
    /// parallel over all cells. Default errors (a physics with no element kernel,
    /// e.g. a constraint such as `Dirichlet`).
    fn element_matrix(
        &self,
        _geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        _ke: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no element kernel — element_matrix is undefined",
            self.label()
        )))
    }

    /// Build and fill the stiffness [`SubMatrix`] block(s) of this physics.
    /// `material` is `Some(_)` iff [`material_fespace`](Self::material_fespace)
    /// is `Some(_)` (the assembler guarantees it).
    ///
    /// **Default**: derived from [`stiffness_layout`](Self::stiffness_layout) —
    /// a single block on that layout, filled by
    /// [`element_matrix`](Self::element_matrix) via [`kernel::assemble_block`].
    /// A plain volumetric physics therefore writes only `element_matrix` +
    /// `stiffness_layout` and gets this for free. A physics with **no** layout
    /// (a constraint such as `Dirichlet`, or any multi-block physics) must
    /// override it. This literal path also serves as the bit-for-bit reference
    /// of the computed (scatter) path.
    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let Some(layout) = self.stiffness_layout() else {
            return Err(PyrucastError::Message(format!(
                "{}: build_stiffness_blocks has no default without a \
                 stiffness_layout — override it (e.g. a constraint such as \
                 Dirichlet, or a multi-block physics)",
                self.label()
            )));
        };
        let block = kernel::assemble_block(
            &layout.fespaces,
            &layout.support,
            &layout.support,
            layout.dual_vars,
            layout.primal_vars,
            layout.ordering,
            layout.symmetric,
            material,
            |geoms, m, ke| self.element_matrix(geoms, m, ke),
        )?;
        Ok(vec![block])
    }

    /// Structural layout of this physics' stiffness block, or `None` (default)
    /// for a physics assembled the literal way (constraints such as `Dirichlet`,
    /// or any multi-block physics). When `Some`, it drives **both** paths from a
    /// single description: the global assembler
    /// ([`crate::ops::assemble::stiffness`]) builds a *computed*
    /// [`SubMatrix`] and scatters [`element_matrix`](Self::element_matrix)
    /// straight into the CSR (never materialising values), and the default
    /// [`build_stiffness_blocks`](Self::build_stiffness_blocks) produces the
    /// *literal* equivalent from the same layout + kernel.
    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        None
    }

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
    /// input is produced geometrically by [`crate::ops::field::gradient`](fn@crate::ops::field::gradient) /
    /// [`crate::ops::field::deformation`](fn@crate::ops::field::deformation), and [`crate::ops::behavior`] uses
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
    /// [`crate::ops::field::gradient`](fn@crate::ops::field::gradient) / [`crate::ops::field::deformation`](fn@crate::ops::field::deformation),
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
    /// Output component names of the material-state field produced by
    /// [`integrate_point`](Self::integrate_point) — the dual flux/stress
    /// followed by the updated internal state (`VAR1`), in order. Implemented by
    /// every behaviour-bearing physics; default errors.
    fn behavior_output_components(&self) -> Result<Vec<String>> {
        Err(PyrucastError::Message(format!(
            "{}: no behaviour — behavior_output_components is undefined",
            self.label()
        )))
    }

    /// Constitutive law at **one Gauss point** — the pure, sequential kernel a
    /// physics author writes. For cell `geom.cell` at Gauss point `g`, read the
    /// deformation (+ `VAR0`) from `input` and the material from `material`
    /// (both borrowed in place), and write the
    /// [`behavior_output_components`](Self::behavior_output_components) values
    /// into `out`. `material` is `Some(_)` iff the physics declares a
    /// [`material_fespace`](Self::material_fespace).
    ///
    /// It **never sees rayon, the store, or a lock**:
    /// [`integrate_behavior`](Self::integrate_behavior) drives it in parallel
    /// over all cells. Default errors (a physics with no behaviour).
    fn integrate_point(
        &self,
        _geom: &CellGeom,
        _input: &SubElementField,
        _material: Option<&SubElementField>,
        _g: usize,
        _out: &mut [f64],
    ) -> Result<()> {
        Err(PyrucastError::Message(format!(
            "{}: no behaviour — integrate_point is undefined",
            self.label()
        )))
    }

    /// Integrate the constitutive law (Cast3m `COMP`). **Provided**: drives the
    /// point kernel [`integrate_point`](Self::integrate_point) in parallel over
    /// the behaviour FE subspace via [`kernel::integrate_pointwise`]. A physics
    /// implements the point kernel +
    /// [`behavior_output_components`](Self::behavior_output_components), **not**
    /// this. A physics with no
    /// behaviour FE subspace falls through to a clear error here.
    fn integrate_behavior(
        &self,
        input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<SubElementField> {
        let fespace = self.behavior_fespace().ok_or_else(|| {
            PyrucastError::Message(format!(
                "{}: no behaviour — integrate_behavior is undefined",
                self.label()
            ))
        })?;
        let out_components = self.behavior_output_components()?;
        kernel::integrate_pointwise(
            &fespace,
            input,
            material,
            out_components,
            |geom, inp, mat, g, out| self.integrate_point(geom, inp, mat, g, out),
        )
    }

    /// Local internal-force vector of one cell — the pure, sequential kernel
    /// that applies `Bᵀ` to the stress (Cast3m `BSIG`). It is the **transpose**
    /// of this physics' deformation operator `B` (the same `B` behind its
    /// [`crate::ops::field::deformation`](fn@crate::ops::field::deformation) /
    /// [`crate::ops::field::beam_deformation`](fn@crate::ops::field::beam_deformation)),
    /// so it mirrors [`integrate_point`](Self::integrate_point)'s producer.
    ///
    /// Fills `fe` — the cell's local force vector, node-major / variable-minor
    /// (`fe[li * n_dual + di]`, `di` indexing [`dual_vars`](Self::dual_vars)) —
    /// from the cell geometry and the `stress` (the [`integrate_point`](Self::integrate_point)
    /// output) borrowed in place. `geoms` holds one [`CellGeom`] per FE subspace
    /// of [`stiffness_layout`](Self::stiffness_layout), in that order.
    ///
    /// **Default**: the continuum-mechanics `f_{i,a} = Σ_g Σ_b (∂N_i/∂x_b) σ_ab`
    /// — one [`crate::ops::field::divergence`](fn@crate::ops::field::divergence)
    /// per row of the symmetric stress tensor `σ`, read in Voigt naming
    /// (`sigma_xx`, `sigma_xy`, …). A displacement physics (elasticity, Mazars,
    /// plasticity) gets it for free; a physics whose dual is not a displacement
    /// vector (heat, bar, beam) overrides it.
    fn internal_force_element(
        &self,
        geoms: &[CellGeom],
        stress: &SubElementField,
        fe: &mut [f64],
    ) -> Result<()> {
        continuum_internal_force_element(geoms, stress, fe)
    }

    /// Internal nodal forces `f = ∫ Bᵀ σ dΩ` of this physics (Cast3m `BSIG`),
    /// scattered to a [`SubNodeField`] on the block's node support. `stress` is
    /// this physics' [`integrate_behavior`](Self::integrate_behavior) output.
    ///
    /// **Provided**: drives [`internal_force_element`](Self::internal_force_element)
    /// in parallel over the FE subspaces of
    /// [`stiffness_layout`](Self::stiffness_layout) (same geometry as the
    /// stiffness) and scatters to that layout's node support. A physics with no
    /// stiffness layout (a constraint such as `Dirichlet`) has no internal-force
    /// contribution and errors here. For a **linear** law the result equals
    /// `K·u` (the stiffness applied to the solution).
    fn build_internal_forces(&self, stress: &Handle<SubElementField>) -> Result<SubNodeField> {
        let Some(layout) = self.stiffness_layout() else {
            return Err(PyrucastError::Message(format!(
                "{}: build_internal_forces has no default without a stiffness_layout \
                 (e.g. a constraint such as Dirichlet)",
                self.label()
            )));
        };
        kernel::internal_forces(
            &layout.fespaces,
            &layout.support,
            layout.dual_vars,
            stress,
            |geoms, s, fe| self.internal_force_element(geoms, s, fe),
        )
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

/// Continuum-mechanics internal-force element kernel `f_{i,a} = Σ_g Σ_b
/// (∂N_i/∂x_b) σ_ab |J| w` — one [`crate::ops::field::divergence`](fn@crate::ops::field::divergence)
/// per row of the symmetric stress tensor `σ` (read in Voigt naming). Backs both
/// the [`Physics::internal_force_element`] default (elasticity, Mazars,
/// plasticity) and the model-free
/// [`crate::ops::internal_forces::internal_forces_continuum`] operator. Fills
/// `fe` node-major / axis-minor (`fe[i * space_dim + a]`).
pub(crate) fn continuum_internal_force_element(
    geoms: &[CellGeom],
    stress: &SubElementField,
    fe: &mut [f64],
) -> Result<()> {
    let geom = &geoms[0];
    let d = geom.space_dim;
    let n_nodes = geom.n_nodes;
    for g in 0..geom.n_gauss {
        let dn = geom.dn_dx(g)?; // [i * d + b]
        let w = geom.det_j_w(g)?;
        let sig = voigt_stress_matrix(stress, geom.cell, g, d)?; // [a * d + b]
        for i in 0..n_nodes {
            for a in 0..d {
                let mut s = 0.0;
                for b in 0..d {
                    s += dn[i * d + b] * sig[a * d + b];
                }
                fe[i * d + a] += s * w;
            }
        }
    }
    Ok(())
}

/// Read the symmetric `d×d` stress tensor at `(cell, g)` from a Voigt-named
/// stress field (`sigma_xx`, `sigma_yy`, `sigma_xy`, …), as a flat row-major
/// matrix `[a * d + b]`. Backs the continuum-mechanics
/// [`Physics::internal_force_element`] default; reads by component name, so a
/// state field carrying extra `VAR1` components (Mazars) is handled transparently.
fn voigt_stress_matrix(
    stress: &SubElementField,
    cell: usize,
    g: usize,
    d: usize,
) -> Result<Vec<f64>> {
    let mut sig = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in i..d {
            let name = format!("sigma_{}{}", VOIGT_AXES[i], VOIGT_AXES[j]);
            let v = stress.value(cell, g, &name)?;
            sig[i * d + j] = v;
            sig[j * d + i] = v; // symmetric
        }
    }
    Ok(sig)
}
