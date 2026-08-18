//! Scalar and orthotropic **damage** — the physics, for any damage law.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`], and the **same
//! elastic stiffness** as iteration operator. The constitutive update is a
//! *secant* scalar-damage law: the stress is the elastic (effective) stress
//! scaled by `(1 − D)`, with `D ∈ [0, 1)` a scalar damage built from the
//! equivalent strain.
//!
//! Equivalent strain `ε̃ = √(Σ ⟨ε_I⟩₊²)` (positive parts of the principal
//! strains). Damage grows with the history variable `κ = maxₜ ε̃`, initialised
//! at the threshold `eps_d0`. Two damage branches `D_t` (tension) and `D_c`
//! (compression) are blended by weights `α_t`, `α_c` derived from the
//! tension/compression split of the effective stress:
//!
//! ```text
//! D_t = 1 − eps_d0(1−A_t)/κ − A_t / exp(B_t (κ − eps_d0))
//! D_c = 1 − eps_d0(1−A_c)/κ − A_c / exp(B_c (κ − eps_d0))
//! D   = α_t D_t + α_c D_c            (shear coefficient β fixed to 1)
//! σ   = (1 − D) · D_el : ε
//! ```
//!
//! Material components `E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`. The
//! single internal variable `kappa` comes in as the previous-step state `prev`
//! (`κ(A)`, floored at `eps_d0` in the update, so `None` on the first step is
//! fine) and out as the updated `VAR1`, alongside the scalar `damage`. The
//! effective stress is a function of the current total strain `ε(B)` — damage
//! mechanics has no strain increment; only `κ` is history.
//!
//! The equivalent strain is built from the **principal strains of the full 3-D
//! tensor**, so the 2-D models differ only in how that tensor is reconstructed:
//! plane strain forces `ε_zz = 0`, plane stress derives it, and **axisymmetric**
//! reads the measured hoop `ε_θθ = u_r/r`.
//!
//! As for plasticity, the Newton loop driving the load increments lives in
//! Python, not in Rust; this module provides the point-wise update only.

pub mod damage_tc;
pub mod mazars;
pub mod sic_sic;

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::owned_components;
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Which damage law a [`Damage`] sub-model obeys.
///
/// The same attribute pattern as [`PlasticLaw`](crate::models::plastic::PlasticLaw):
/// the DOFs, the elastic operator and the incremental montage are shared, and
/// only the law that turns a strain into a degraded stress differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DamageLaw {
    /// Mazars — one scalar, two branches blended. Concrete.
    #[default]
    Mazars,
    /// Two damages, tension and compression apart — recovers the compressive
    /// stiffness when a crack closes.
    DamageTc,
    /// Orthotropic damage of a woven ceramic-matrix composite, one damage per
    /// weave direction.
    SicSic,
}

impl DamageLaw {
    /// Parse from a lowercase tag (`"mazars"`, `"damage_tc"`, `"sic_sic"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "mazars" => Some(Self::Mazars),
            "damage_tc" => Some(Self::DamageTc),
            "sic_sic" => Some(Self::SicSic),
            _ => None,
        }
    }

    /// The lowercase tag (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Mazars => "mazars",
            Self::DamageTc => "damage_tc",
            Self::SicSic => "sic_sic",
        }
    }

    /// Every law, in declaration order.
    pub const ALL: [DamageLaw; 3] = [Self::Mazars, Self::DamageTc, Self::SicSic];

    /// The accepted tags, `|`-joined — for error messages.
    pub fn tag_list() -> String {
        Self::ALL
            .iter()
            .map(|l| l.to_tag())
            .collect::<Vec<_>>()
            .join("|")
    }

    /// The material components this law requires.
    pub fn material_components(self, space_dim: usize) -> &'static [&'static str] {
        match self {
            Self::Mazars => mazars::MATERIAL,
            Self::DamageTc => damage_tc::MATERIAL,
            Self::SicSic if space_dim == 2 => sic_sic::MATERIAL_2D,
            Self::SicSic => sic_sic::MATERIAL_3D,
        }
    }

    /// The law's internal variables, beyond the reported `damage`.
    pub fn internal_names(self) -> Vec<String> {
        match self {
            Self::Mazars => vec!["kappa".into()],
            Self::DamageTc => vec![
                "r_plus".into(),
                "r_minus".into(),
                "d_plus".into(),
                "d_minus".into(),
            ],
            Self::SicSic => vec![
                "kappa_1".into(),
                "kappa_2".into(),
                "kappa_3".into(),
                "d_1".into(),
                "d_2".into(),
                "d_3".into(),
            ],
        }
    }

    /// One step of the law, at a Gauss point.
    pub fn update(
        self,
        eps: &[f64; 6],
        prev: &[f64],
        mat: &MatRead,
        space_dim: usize,
    ) -> Result<DamageUpdate> {
        match self {
            Self::Mazars => mazars::update(eps, prev, mat),
            Self::DamageTc => damage_tc::update(eps, prev, mat),
            Self::SicSic => sic_sic::update(eps, prev, mat, space_dim),
        }
    }
}

impl std::fmt::Display for DamageLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

/// What a damage law returns for one Gauss point.
pub struct DamageUpdate {
    /// The degraded stress, full 3-D Voigt.
    pub sigma: [f64; 6],
    /// A scalar summary of the damage, for visualisation. The **state** is
    /// `vars`; a law with several damages reports the worst here.
    pub damage: f64,
    /// The law's internal variables, in [`DamageLaw::internal_names`] order.
    pub vars: Vec<f64>,
}

/// A cell's material, read by name — the same shape every law wants.
pub struct MatRead<'a> {
    /// The material field, exposed so a law can reach the shared frame reader.
    pub field: &'a SubElementField,
    /// The cell this reads.
    pub cell: usize,
}

impl MatRead<'_> {
    /// A material component of this cell, by name.
    pub fn get(&self, name: &str) -> Result<f64> {
        self.field.value(self.cell, 0, name)
    }
}

/// Lamé coefficients `(λ, μ)` from `E`, `nu` — shared by every law.
pub fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic (effective) stress, full 3-D Voigt, from a **tensor** strain.
pub fn elastic_stress(eps: &[f64; 6], lambda: f64, mu: f64) -> [f64; 6] {
    let tr = eps[0] + eps[1] + eps[2];
    [
        lambda * tr + 2.0 * mu * eps[0],
        lambda * tr + 2.0 * mu * eps[1],
        lambda * tr + 2.0 * mu * eps[2],
        2.0 * mu * eps[3],
        2.0 * mu * eps[4],
        2.0 * mu * eps[5],
    ]
}

/// Positive part `⟨x⟩₊ = max(x, 0)`.
pub fn pos(x: f64) -> f64 {
    x.max(0.0)
}

/// Where each **axisymmetric** Voigt slot `[rr, zz, θθ, rz]` sits in the full
/// 3-D order `[xx, yy, zz, yz, xz, xy]` — the damage law itself stays 3-D
/// (principal strains of the full tensor), only the projection changes.
const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order (matching [`crate::models::elasticity`]).
fn stress_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    if space_dim == 2 && model.is_axisymmetric() {
        // [rr, zz, θθ, rz] — the hoop is `zz`, Cast3M naming.
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_xy".into(),
        ]
    } else if space_dim == 2 {
        vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()]
    } else {
        vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ]
    }
}

/// Damage on an FE subspace. Same supports as
/// [`crate::models::elasticity::Elasticity`]; material is supplied at
/// assembly / integration time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Damage {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
    pub(crate) law: DamageLaw,
}

impl Damage {
    /// **Mazars** damage on an FE subspace — the default law.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        Self::with_law(fespace, model, DamageLaw::Mazars)
    }

    /// Damage with an explicit law, on an FE subspace with the given 2-D/3-D
    /// model. Errors if
    /// `model` is inconsistent with the space dimension (same rule as
    /// [`crate::models::elasticity::Elasticity::new`]).
    pub fn with_law(
        fespace: Handle<SubFiniteElementSpace>,
        model: ElasticityModel,
        law: DamageLaw,
    ) -> Result<Self> {
        let (submesh, space_dim, ref_dim, axisymmetric) = {
            let s = fespace.read();
            (
                s.submesh(),
                s.space_dim(),
                s.ref_dim()?,
                s.is_axisymmetric(),
            )
        };
        crate::models::elasticity::check_continuum_dimensions("Damage", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (2, ElasticityModel::Axisymmetric) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Damage: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ solid)"
            )));
        }
        // Same two-way agreement as `Elasticity::new`.
        if axisymmetric != model.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Damage: model {model:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` model"
                )
            } else {
                "Damage: the `axisymmetric` model requires an axisymmetric geometry \
                 (build the Coords with Coords::axisymmetric)"
                    .into()
            }));
        }
        let support = submesh.read().to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
            law,
        })
    }
}

impl SubModelKind for Damage {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<MatrixLayout> {
        Some(MatrixLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// The consistent mass matrix shares the stiffness layout (mass is
    /// law-independent).
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    /// The geometric stiffness shares the stiffness layout (initial-stress term
    /// is law-independent given the current stress).
    fn geometric_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // Iteration operator = elastic (undamaged) stiffness. Reuse the
        // elasticity element kernel; it reads only `E` and `nu`.
        let mat = material.expect("Damage requires a material field");
        elasticity::element_stiffness(
            geom,
            mat,
            self.model,
            crate::models::symmetry::MaterialSymmetry::Isotropic,
            ke,
        )
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Damage requires a material field");
        elasticity::element_mass(geom, mat, ke)
    }

    fn element_geometric(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let stress = state.expect("geometric stiffness requires the current stress field");
        elasticity::element_geometric(geom, stress, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Damage"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = self.support.read().cell_count();
        format!(
            "SubModel<Damage({:?}, {})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model, self.law
        )
    }
}

impl Domain for Damage {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<Vec<String>> {
        Some(owned_components(
            self.law.material_components(self.space_dim),
        ))
    }

    /// `alpha` (thermal expansion) and `rho` (density) — the same pair
    /// [`elasticity`] accepts, and for the same
    /// reasons.
    ///
    /// `alpha` is read by an **ancillary** operator,
    /// [`thermal_strain`](fn@crate::ops::element_field::thermal_strain), which
    /// subtracts the expansion before the mechanical law sees anything: the
    /// return mapping never touches it. Leaving it out therefore excluded
    /// thermal expansion from plasticity and damage for no reason at all —
    /// `material_field` drops a component the physics does not declare, so the
    /// operator then found no zone carrying it.
    ///
    /// `rho` is required only by the mass matrix, never by the
    /// stiffness/behaviour assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha", "rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim, self.model);
        comps.push("damage".into());
        // The law's own history and per-direction damages.
        comps.extend(self.law.internal_names());
        Ok(comps)
    }

    /// One damage step at a Gauss point. Output layout = stress (Voigt, `v`) +
    /// the reported `damage` + the law's own internal variables.
    fn integrate_point(
        &self,
        geom: &CellGeom,
        deformation: &SubElementField,
        prev: Option<&SubElementField>,
        material: Option<&SubElementField>,
        g: usize,
        _dt: Option<f64>,
        out: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Damage declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let read = MatRead { field: mat, cell };
        // End-of-step strain ε(B); the law's history from `prev` (absent on the
        // first step, where every variable starts at zero).
        let eps = read_strain(deformation, cell, g, d, read.get("nu")?, self.model)?;
        // Left **empty** on the first step, where `prev` is `None`, so a law can
        // tell « no state yet » from « state that is zero ».
        let prev_vars: Vec<f64> = match prev {
            None => Vec::new(),
            Some(_) => self
                .law
                .internal_names()
                .iter()
                .map(|n| prev_opt(prev, cell, g, n))
                .collect(),
        };

        let update = self.law.update(&eps, &prev_vars, &read, d)?;
        let v = stress_names(d, self.model).len();
        for r in 0..v {
            out[r] = voigt_stress(&update.sigma, d, self.model, r);
        }
        out[v] = update.damage;
        for (i, value) in update.vars.iter().enumerate() {
            out[v + 1 + i] = *value;
        }
        Ok(())
    }
}

// The constitutive cores live in [`crate::models::damage`]'s submodules, one
// per law — shared helpers (Lamé, the elastic stress, the positive part) in the
// module root below. What remains here is the physics: the DOFs, the layouts,
// and the plumbing between the field components and the full-3-D strain the
// laws work in.

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// Read a component, returning `0.0` when absent (first step has no state).
fn read_opt(f: &SubElementField, cell: usize, g: usize, name: &str) -> f64 {
    if f.component_index(name).is_some() {
        f.value(cell, g, name).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Read a component from the optional previous-state field `prev`, defaulting to
/// `0.0` when there is no previous step (`None`) or the component is absent.
fn prev_opt(prev: Option<&SubElementField>, cell: usize, g: usize, name: &str) -> f64 {
    prev.map_or(0.0, |f| read_opt(f, cell, g, name))
}

/// Reconstruct the full 3-D tensor strain. Plane strain forces `ε_zz = 0`;
/// plane stress sets `ε_zz = -ν/(1-ν)(ε_xx+ε_yy)` (the elastic-damaged
/// out-of-plane strain, since the `(1-D)` factor cancels in `σ_zz = 0`).
fn read_strain(
    f: &SubElementField,
    cell: usize,
    g: usize,
    space_dim: usize,
    nu: f64,
    model: ElasticityModel,
) -> Result<[f64; 6]> {
    let mut eps = [0.0; 6];
    if space_dim == 2 {
        eps[0] = f.value(cell, g, "eps_xx")?;
        eps[1] = f.value(cell, g, "eps_yy")?;
        eps[5] = f.value(cell, g, "eps_xy")?;
        if model == ElasticityModel::PlaneStress {
            eps[2] = -nu / (1.0 - nu) * (eps[0] + eps[1]);
        } else if model.is_axisymmetric() {
            // The hoop ε_θθ = u_r/r is measured by `deformation`, not assumed.
            eps[2] = f.value(cell, g, "eps_zz")?;
        }
    } else {
        for (k, suf) in ["xx", "yy", "zz", "yz", "xz", "xy"].iter().enumerate() {
            eps[k] = f.value(cell, g, &format!("eps_{suf}"))?;
        }
    }
    Ok(eps)
}

/// Project the full 3-D stress to the model's Voigt slot `r`.
fn voigt_stress(sigma: &[f64; 6], space_dim: usize, model: ElasticityModel, r: usize) -> f64 {
    if space_dim == 2 && model.is_axisymmetric() {
        sigma[AXI_TO_3D[r]]
    } else if space_dim == 2 {
        match r {
            0 => sigma[0],
            1 => sigma[1],
            _ => sigma[5],
        }
    } else {
        sigma[r]
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::Mesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn unit_quad(model: ElasticityModel) -> Damage {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Damage::new(fes.get(0).unwrap(), model).unwrap()
    }

    fn material(mz: &Damage) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            mz.fespace.clone(),
            mazars::MATERIAL.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        mat.set_uniform("E", 30_000.0).unwrap(); // ~ concrete (MPa)
        mat.set_uniform("nu", 0.2).unwrap();
        mat.set_uniform("eps_d0", 1e-4).unwrap();
        mat.set_uniform("A_t", 0.8).unwrap();
        mat.set_uniform("B_t", 20_000.0).unwrap();
        mat.set_uniform("A_c", 1.4).unwrap();
        mat.set_uniform("B_c", 1_900.0).unwrap();
        Handle::new(mat)
    }

    fn strain_field(mz: &Damage, eps_xx: f64) -> Handle<SubElementField> {
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        s.set_uniform("eps_xx", eps_xx).unwrap();
        Handle::new(s)
    }

    #[test]
    fn vars_and_material() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        assert_eq!(mz.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(mz.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(
            mz.material_components(),
            Some(owned_components(mazars::MATERIAL))
        );
    }

    /// Below the damage threshold the response is elastic: D = 0 and σ_xx is
    /// the linear plane-stress stress.
    #[test]
    fn undamaged_below_threshold() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        let eps0 = 1e-5; // < eps_d0 = 1e-4
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap().abs() < 1e-14);
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-6);
        }
    }

    /// Above the threshold in tension, damage develops (0 < D < 1) and the
    /// stress is reduced below the elastic prediction.
    #[test]
    fn damages_in_tension() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        let eps0 = 5e-4; // > eps_d0
        let strain = strain_field(&mz, eps0);
        let out = mz
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        let (e, nu) = (30_000.0, 0.2);
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            let d = out.value(0, g, "damage").unwrap();
            assert!(d > 0.0 && d < 1.0, "D = {d}");
            // Damaged stress strictly below the elastic prediction.
            assert!(out.value(0, g, "sigma_xx").unwrap() < c * eps0);
            assert!(out.value(0, g, "kappa").unwrap() >= eps0 - 1e-12);
        }
    }

    /// History variable κ is monotone: unloading to a smaller strain does not
    /// reduce κ, and does not heal damage.
    #[test]
    fn kappa_is_monotone() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        let mat = material(&mz);
        // Load to 5e-4.
        let s1 = strain_field(&mz, 5e-4);
        let st1 = mz.integrate_behavior(&s1, None, Some(&mat), None).unwrap();
        let k1 = st1.value(0, 0, "kappa").unwrap();
        let d1 = st1.value(0, 0, "damage").unwrap();

        // Unload to 2e-4, feeding the step-1 state (κ) via `prev`.
        let prev = Handle::new(st1);
        let s2 = strain_field(&mz, 2e-4);
        let st2 = mz
            .integrate_behavior(&s2, Some(&prev), Some(&mat), None)
            .unwrap();
        assert!((st2.value(0, 0, "kappa").unwrap() - k1).abs() < 1e-12);
        // Damage unchanged on unloading (same κ).
        assert!((st2.value(0, 0, "damage").unwrap() - d1).abs() < 1e-9);
    }

    /// Solid 3-D uniaxial tension also triggers tensile damage.
    #[test]
    fn solid_3d_damages() {
        let coords = Handle::new(Coords::new(3).unwrap());
        let p = |x: f64, y: f64, z: f64| Node::create_in(coords.clone(), &[x, y, z]).unwrap();
        let n = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 0.0),
            p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0),
            p(1.0, 0.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 1.0),
        ];
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
        mesh.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
            .unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mz = Damage::new(fes.get(0).unwrap(), ElasticityModel::Solid).unwrap();
        let mat = material(&mz);
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            ["xx", "yy", "zz", "yz", "xz", "xy"]
                .iter()
                .map(|x| format!("eps_{x}"))
                .collect(),
        )
        .unwrap();
        s.set_uniform("eps_xx", 5e-4).unwrap();
        let s = Handle::new(s);
        let out = mz.integrate_behavior(&s, None, Some(&mat), None).unwrap();
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap() > 0.0);
        }
    }
}
