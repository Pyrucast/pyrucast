//! Mazars isotropic damage — classic two-variable formulation (Mazars 1986).
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
//! As for plasticity, the Newton loop driving the load increments lives in
//! Python (see `ROADMAP.md`); this module provides the point-wise update only.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::DofOrdering;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::{CellGeom, Domain, Physics, StiffnessLayout, SubModelKind};
use crate::store::{read, Handle};
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by the Mazars model.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu", "eps_d0", "A_t", "B_t", "A_c", "B_c"];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order (matching [`crate::models::elasticity`]).
fn stress_names(space_dim: usize) -> Vec<String> {
    if space_dim == 2 {
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

/// Mazars damage on an FE subspace. Same supports as
/// [`crate::models::elasticity::Elasticity`]; material is supplied at
/// assembly / integration time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Mazars {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
}

impl Mazars {
    /// Mazars damage on an FE subspace, with the given 2-D/3-D model. Errors if
    /// `model` is inconsistent with the space dimension (same rule as
    /// [`crate::models::elasticity::Elasticity::new`]).
    pub fn new(fespace: Handle<SubFiniteElementSpace>, model: ElasticityModel) -> Result<Self> {
        let (submesh, space_dim) = {
            let s = read(&fespace)?;
            (s.submesh(), s.space_dim())
        };
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Mazars: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain, 3-D ⇒ solid)"
            )));
        }
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
        })
    }
}

impl SubModelKind for Mazars {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn as_domain(&self) -> Option<&dyn Domain> {
        Some(self)
    }

    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        Some(StiffnessLayout {
            fespaces: vec![self.fespace.clone()],
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: DofOrdering::NodesThenVars,
            symmetric: true,
        })
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
        let mat = material.expect("Mazars requires a material field");
        elasticity::element_stiffness(geom, mat, self.model, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
    }

    fn label(&self) -> &'static str {
        "Mazars"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Mazars({:?})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

impl Domain for Mazars {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim);
        comps.push("damage".into());
        comps.push("kappa".into());
        Ok(comps)
    }

    /// Mazars damage update at one Gauss point. Output layout = stress (Voigt,
    /// `v`) + `damage` + `kappa`.
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
        let mat = material.expect("Mazars declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let p = MazarsParams {
            e: mat.value(cell, 0, "E")?,
            nu: mat.value(cell, 0, "nu")?,
            eps_d0: mat.value(cell, 0, "eps_d0")?,
            a_t: mat.value(cell, 0, "A_t")?,
            b_t: mat.value(cell, 0, "B_t")?,
            a_c: mat.value(cell, 0, "A_c")?,
            b_c: mat.value(cell, 0, "B_c")?,
        };
        // End-of-step strain ε(B); history variable κ(A) from `prev` (floored at
        // `eps_d0` in the update, so `None`/absent — the first step — is fine).
        let eps = read_strain(deformation, cell, g, d, p.nu, self.model)?;
        let kappa_old = prev_opt(prev, cell, g, "kappa");
        let (sigma, damage, kappa) = mazars_update(&eps, kappa_old, &p);
        let v = stress_names(d).len();
        for r in 0..v {
            out[r] = voigt_stress(&sigma, d, r);
        }
        out[v] = damage;
        out[v + 1] = kappa;
        Ok(())
    }
}

// ─── Constitutive core (pure, store-free) ────────────────────────────────────

/// Material parameters of the Mazars model at one Gauss point.
struct MazarsParams {
    e: f64,
    nu: f64,
    eps_d0: f64,
    a_t: f64,
    b_t: f64,
    a_c: f64,
    b_c: f64,
}

/// Lamé coefficients `(λ, μ)` from `E`, `nu`.
fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic (effective) stress, full 3-D Voigt `[xx, yy, zz, yz, xz, xy]`
/// from a **tensor** strain: `σ̃ = λ tr(ε) I + 2μ ε`.
fn elastic_stress(eps: &[f64; 6], lambda: f64, mu: f64) -> [f64; 6] {
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

/// One damage branch `D = 1 − eps_d0(1−A)/κ − A / exp(B (κ − eps_d0))`,
/// clamped to `[0, 1)`.
fn damage_branch(kappa: f64, eps_d0: f64, a: f64, b: f64) -> f64 {
    let d = 1.0 - eps_d0 * (1.0 - a) / kappa - a / (b * (kappa - eps_d0)).exp();
    d.clamp(0.0, 1.0 - 1e-12)
}

/// Mazars point update. Returns `(stress, damage, kappa)` (stress full 3-D Voigt).
fn mazars_update(eps: &[f64; 6], kappa_old: f64, p: &MazarsParams) -> ([f64; 6], f64, f64) {
    let (lambda, mu) = lame(p.e, p.nu);
    let sigma_eff = elastic_stress(eps, lambda, mu);

    // Principal strains (coaxial with the effective stress, isotropic elasticity).
    let tensor = Matrix3::new(
        eps[0], eps[5], eps[4], // [εxx, εxy, εxz]
        eps[5], eps[1], eps[3], // [εxy, εyy, εyz]
        eps[4], eps[3], eps[2], // [εxz, εyz, εzz]
    );
    let e_pr = tensor.symmetric_eigenvalues();

    // Equivalent strain ε̃ = √(Σ ⟨ε_I⟩₊²).
    let eps_eq = (e_pr.iter().map(|&x| pos(x).powi(2)).sum::<f64>()).sqrt();

    // History variable: never below the threshold, never decreasing.
    let kappa = kappa_old.max(p.eps_d0).max(eps_eq);
    if kappa <= p.eps_d0 {
        return (sigma_eff, 0.0, kappa); // undamaged
    }

    // Tension/compression split of the effective principal stresses
    // σ̃_I = λ·tr + 2μ·ε_I, then strains induced by each part via the
    // isotropic compliance (all coaxial ⇒ work in principal space).
    let tr = e_pr[0] + e_pr[1] + e_pr[2];
    let st: [f64; 3] = std::array::from_fn(|i| lambda * tr + 2.0 * mu * e_pr[i]);
    let stp: [f64; 3] = std::array::from_fn(|i| pos(st[i]));
    let stn: [f64; 3] = std::array::from_fn(|i| st[i].min(0.0));
    let sum_p: f64 = stp.iter().sum();
    let sum_n: f64 = stn.iter().sum();
    // ε^t_I = [(1+ν)σ̃⁺_I − ν Σσ̃⁺] / E ; ε^c_I likewise from σ̃⁻.
    let eps_t: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stp[i] - p.nu * sum_p) / p.e);
    let eps_c: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stn[i] - p.nu * sum_n) / p.e);

    let denom = eps_eq * eps_eq;
    let mut alpha_t = 0.0;
    let mut alpha_c = 0.0;
    if denom > 0.0 {
        for i in 0..3 {
            let w = pos(e_pr[i]);
            alpha_t += pos(eps_t[i]) * w;
            alpha_c += pos(eps_c[i]) * w;
        }
        alpha_t /= denom;
        alpha_c /= denom;
    }
    let alpha_t = alpha_t.clamp(0.0, 1.0);
    let alpha_c = alpha_c.clamp(0.0, 1.0);

    let d_t = damage_branch(kappa, p.eps_d0, p.a_t, p.b_t);
    let d_c = damage_branch(kappa, p.eps_d0, p.a_c, p.b_c);
    // β fixed to 1 (no shear correction).
    let damage = (alpha_t * d_t + alpha_c * d_c).clamp(0.0, 1.0 - 1e-12);

    let sigma: [f64; 6] = std::array::from_fn(|i| (1.0 - damage) * sigma_eff[i]);
    (sigma, damage, kappa)
}

/// Positive part `⟨x⟩₊ = max(x, 0)`.
fn pos(x: f64) -> f64 {
    x.max(0.0)
}

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
        }
    } else {
        for (k, suf) in ["xx", "yy", "zz", "yz", "xz", "xy"].iter().enumerate() {
            eps[k] = f.value(cell, g, &format!("eps_{suf}"))?;
        }
    }
    Ok(eps)
}

/// Project the full 3-D stress to the model's Voigt slot `r`.
fn voigt_stress(sigma: &[f64; 6], space_dim: usize, r: usize) -> f64 {
    if space_dim == 2 {
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
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node};
    use crate::store::insert;

    fn unit_quad(model: ElasticityModel) -> Mazars {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Mazars::new(fes.get(0).unwrap(), model).unwrap()
    }

    fn material(mz: &Mazars) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            mz.fespace.clone(),
            MATERIAL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
        )
        .unwrap();
        mat.set_uniform("E", 30_000.0).unwrap(); // ~ concrete (MPa)
        mat.set_uniform("nu", 0.2).unwrap();
        mat.set_uniform("eps_d0", 1e-4).unwrap();
        mat.set_uniform("A_t", 0.8).unwrap();
        mat.set_uniform("B_t", 20_000.0).unwrap();
        mat.set_uniform("A_c", 1.4).unwrap();
        mat.set_uniform("B_c", 1_900.0).unwrap();
        insert(mat)
    }

    fn strain_field(mz: &Mazars, eps_xx: f64) -> Handle<SubElementField> {
        let mut s = SubElementField::new(
            mz.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        s.set_uniform("eps_xx", eps_xx).unwrap();
        insert(s)
    }

    #[test]
    fn vars_and_material() {
        let mz = unit_quad(ElasticityModel::PlaneStress);
        assert_eq!(mz.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(mz.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(mz.material_components(), Some(MATERIAL_COMPONENTS));
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
        let prev = insert(st1);
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
        let coords = insert(Coords::new(3).unwrap());
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
        let mz = Mazars::new(fes.get(0).unwrap(), ElasticityModel::Solid).unwrap();
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
        let s = insert(s);
        let out = mz.integrate_behavior(&s, None, Some(&mat), None).unwrap();
        for g in 0..out.gauss_count() {
            assert!(out.value(0, g, "damage").unwrap() > 0.0);
        }
    }
}
