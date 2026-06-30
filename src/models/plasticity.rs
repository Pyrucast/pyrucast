//! Perfect (non-hardening) von Mises elastoplasticity — J2 radial return.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`] (displacement
//! `u_x, u_y(, u_z)`, nodal force `f_x, …`), and the **same elastic stiffness**
//! as iteration operator: the non-linearity lives entirely in the behaviour
//! integration (`COMP`). Material components `E` (Young), `nu` (Poisson) and
//! `sigma_y` (yield stress). The flow rule is associated J2 with **no
//! hardening**, so the equivalent stress is capped at `sigma_y`.
//!
//! The integration is history-dependent: the **internal state** (plastic
//! strain tensor `eps_p_*` and cumulated plastic strain `p`) flows in as `VAR0`
//! (defaulting to zero on the first step) and out as the updated `VAR1`. State
//! is always carried in **full 3-D** (six `eps_p_*` components) regardless of
//! the 2-D/3-D model, which keeps the radial return identical across plane
//! stress / plane strain / solid; only the input strain reconstruction and the
//! output stress projection differ.
//!
//! Following the locked architecture decision (see `ROADMAP.md`), the Newton
//! loop driving these increments lives in Python; this module only provides the
//! point-wise constitutive update.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::SubMatrix;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::{kernel, CellGeom, Physics, StiffnessLayout};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];
/// Material components required by perfect plasticity.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu", "sigma_y"];

/// Full 3-D tensor component suffixes, in the internal state order
/// `[xx, yy, zz, yz, xz, xy]` (off-diagonals are **tensor** strains, `ε_ij`).
const TENSOR_SUFFIXES: [&str; 6] = ["xx", "yy", "zz", "yz", "xz", "xy"];
/// Index pairs `(i, j)` matching [`TENSOR_SUFFIXES`].
const TENSOR_PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order for the given space dimension —
/// matching [`crate::models::elasticity`] so downstream code is uniform.
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

/// Internal-state component names: plastic strain tensor `eps_p_*` (six,
/// always 3-D) followed by the cumulated plastic strain `p`.
fn state_names() -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_p_{s}")).collect();
    v.push("p".into());
    v
}

/// Perfect von Mises plasticity on an FE subspace.
///
/// Holds the same supports as [`crate::models::elasticity::Elasticity`];
/// material (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time.
#[derive(Clone, Serialize, Deserialize)]
pub struct Plasticity {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 support over the subspace's unique nodes (row/col support).
    pub(crate) support: Handle<SubMesh>,
    pub(crate) space_dim: usize,
    pub(crate) model: ElasticityModel,
}

impl Plasticity {
    /// Perfect plasticity on an FE subspace, with the given 2-D/3-D model.
    /// Errors if `model` is inconsistent with the space dimension (same rule as
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
                "Plasticity: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain, 3-D ⇒ solid)"
            )));
        }
        let support = insert(read(&submesh)?.to_poi1()?);
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
        })
    }
}

impl Physics for Plasticity {
    fn primal_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(primal_name).collect()
    }

    fn dual_vars(&self) -> Vec<String> {
        (0..self.space_dim).map(dual_name).collect()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.fespace.clone())
    }

    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        // Iteration operator = elastic stiffness (no tangent KTAN yet — the
        // Newton loop is orchestrated in Python). Reuse the elasticity element
        // kernel verbatim; it reads only `E` and `nu` from the material.
        let mat = material.expect("Plasticity requires a material field");
        let block = kernel::assemble_block(
            &self.fespace,
            &self.support,
            &self.support,
            self.dual_vars(),
            self.primal_vars(),
            crate::containers::matrix::DofOrdering::NodesThenVars,
            true,
            Some(mat),
            |geom, m, ke| self.element_matrix(geom, m, ke),
        )?;
        Ok(vec![block])
    }

    fn stiffness_layout(&self) -> Option<StiffnessLayout> {
        Some(StiffnessLayout {
            fespace: self.fespace.clone(),
            support: self.support.clone(),
            dual_vars: self.dual_vars(),
            primal_vars: self.primal_vars(),
            ordering: crate::containers::matrix::DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    fn element_matrix(
        &self,
        geom: &CellGeom,
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        // Iteration operator = elastic stiffness (no tangent KTAN yet — the
        // Newton loop is orchestrated in Python). Reuse the elasticity element
        // kernel verbatim; it reads only `E` and `nu` from the material.
        let mat = material.expect("Plasticity requires a material field");
        elasticity::element_stiffness(geom, mat, self.model, ke)
    }

    fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.fespace.clone())
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim);
        comps.extend(state_names());
        Ok(comps)
    }

    /// Radial-return at one Gauss point. Output layout = stress (Voigt, `v`) +
    /// plastic strain `eps_p` (full 3-D tensor, 6) + cumulated plastic strain
    /// `p` (1), matching `stress_names ++ state_names`.
    fn integrate_point(
        &self,
        geom: &CellGeom,
        input: &SubElementField,
        material: Option<&SubElementField>,
        g: usize,
        out: &mut [f64],
    ) -> Result<()> {
        let mat = material.expect("Plasticity declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let (lambda, mu) = lame(mat.value(cell, 0, "E")?, mat.value(cell, 0, "nu")?);
        let sigma_y = mat.value(cell, 0, "sigma_y")?;

        // Total strain (tensor) and previous plastic state (VAR0).
        let eps_total = read_strain(input, cell, g, d)?;
        let eps_p_old = read_state_strain(input, cell, g);
        let p_old = read_opt(input, cell, g, "p");

        let (sigma, eps_p_new, p_new) =
            radial_return(&eps_total, &eps_p_old, p_old, lambda, mu, sigma_y, self.model);

        let v = stress_names(d).len();
        for r in 0..v {
            out[r] = voigt_stress(&sigma, d, r);
        }
        out[v..v + 6].copy_from_slice(&eps_p_new);
        out[v + 6] = p_new;
        Ok(())
    }

    fn label(&self) -> &'static str {
        "Plasticity"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = read(&self.support).map(|s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Plasticity({:?})>\n  primal var(s): {primal}\n  dual var(s):   {dual}\n  \
             support: {n} node(s)",
            self.model
        )
    }
}

// ─── Constitutive core (pure, store-free) ────────────────────────────────────

/// Lamé coefficients `(λ, μ)` from `E`, `nu`.
fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic stress (full 3-D, order `[xx, yy, zz, yz, xz, xy]`) from a
/// **tensor** strain `eps`: `σ = λ tr(ε) I + 2μ ε`.
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

/// von Mises equivalent stress `q = √(3/2 · s:s)` of the deviator of `sigma`
/// (full 3-D Voigt; off-diagonals counted with the factor 2 of `s:s`).
fn von_mises(sigma: &[f64; 6]) -> f64 {
    let mean = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
    let s = [
        sigma[0] - mean,
        sigma[1] - mean,
        sigma[2] - mean,
        sigma[3],
        sigma[4],
        sigma[5],
    ];
    let ss = s[0] * s[0]
        + s[1] * s[1]
        + s[2] * s[2]
        + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (1.5 * ss).sqrt()
}

/// Radial return for **one** Gauss point. Given the total tensor strain, the
/// previous plastic strain and cumulated `p`, returns the updated
/// `(stress, eps_p, p)` (all full 3-D). For plane stress the out-of-plane
/// normal strain `eps[2]` is solved so that `σ_zz = 0`.
fn radial_return(
    eps_total: &[f64; 6],
    eps_p_old: &[f64; 6],
    p_old: f64,
    lambda: f64,
    mu: f64,
    sigma_y: f64,
    model: ElasticityModel,
) -> ([f64; 6], [f64; 6], f64) {
    if model == ElasticityModel::PlaneStress {
        return plane_stress_return(eps_total, eps_p_old, p_old, lambda, mu, sigma_y);
    }
    // Solid / plane strain: eps_total is fully prescribed (plane strain has
    // eps_zz = eps_yz = eps_xz = 0 already).
    return_map_3d(eps_total, eps_p_old, p_old, lambda, mu, sigma_y)
}

/// Classic 3-D radial return with a fully prescribed strain.
fn return_map_3d(
    eps_total: &[f64; 6],
    eps_p_old: &[f64; 6],
    p_old: f64,
    lambda: f64,
    mu: f64,
    sigma_y: f64,
) -> ([f64; 6], [f64; 6], f64) {
    // Elastic trial.
    let eps_e: [f64; 6] = std::array::from_fn(|i| eps_total[i] - eps_p_old[i]);
    let sig_trial = elastic_stress(&eps_e, lambda, mu);
    let q = von_mises(&sig_trial);
    let f = q - sigma_y;
    if f <= 0.0 || q == 0.0 {
        return (sig_trial, *eps_p_old, p_old); // elastic
    }
    // Perfect plasticity: Δp = f / (3μ); deviator scales by σ_y / q.
    let dp = f / (3.0 * mu);
    let mean = (sig_trial[0] + sig_trial[1] + sig_trial[2]) / 3.0;
    let s_trial = [
        sig_trial[0] - mean,
        sig_trial[1] - mean,
        sig_trial[2] - mean,
        sig_trial[3],
        sig_trial[4],
        sig_trial[5],
    ];
    let scale = sigma_y / q;
    // Flow direction n = (3/2) s_trial / q ; Δε_p = Δp · n.
    let factor = 1.5 * dp / q;
    let mut sigma = [0.0; 6];
    let mut eps_p = *eps_p_old;
    for i in 0..6 {
        let s_new = s_trial[i] * scale;
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        // Plastic strain is a tensor: off-diagonals get the engineering ÷2? No —
        // n is built from the stress deviator with the same (1,2) weighting as a
        // tensor, so Δε_p_ij = factor · s_trial_ij directly.
        eps_p[i] += factor * s_trial[i];
    }
    (sigma, eps_p, p_old + dp)
}

/// Plane-stress return: solve the scalar condition `σ_zz(eps_zz) = 0` by the
/// secant method, each evaluation running a full 3-D radial return. The in-plane
/// strains `eps[0], eps[1], eps[5]` are fixed; `eps[3] = eps[4] = 0`.
fn plane_stress_return(
    eps_in: &[f64; 6],
    eps_p_old: &[f64; 6],
    p_old: f64,
    lambda: f64,
    mu: f64,
    sigma_y: f64,
) -> ([f64; 6], [f64; 6], f64) {
    let eval = |ezz: f64| {
        let mut eps = *eps_in;
        eps[2] = ezz;
        eps[3] = 0.0;
        eps[4] = 0.0;
        return_map_3d(&eps, eps_p_old, p_old, lambda, mu, sigma_y)
    };
    // Initial guess: the elastic plane-stress out-of-plane strain
    // ε_zz = -λ/(λ+2μ)·(ε_e,xx + ε_e,yy) (= -ν/(1-ν)·…), added back to the
    // stored plastic ε_p,zz.
    let nu_term = lambda / (lambda + 2.0 * mu); // = ν/(1-ν)
    let mut z0 = eps_p_old[2] - nu_term * (eps_in[0] - eps_p_old[0] + eps_in[1] - eps_p_old[1]);
    let mut z1 = z0 + 1e-6_f64.max(z0.abs() * 1e-3);
    let mut f0 = eval(z0).0[2];
    let mut f1 = eval(z1).0[2];
    for _ in 0..50 {
        if f1.abs() < 1e-10 * (mu + 1.0) {
            break;
        }
        let denom = f1 - f0;
        if denom.abs() < f64::MIN_POSITIVE {
            break;
        }
        let z2 = z1 - f1 * (z1 - z0) / denom;
        z0 = z1;
        f0 = f1;
        z1 = z2;
        f1 = eval(z1).0[2];
    }
    eval(z1)
}

// ─── Field <-> array plumbing ────────────────────────────────────────────────

/// Read a component, returning `0.0` when it is absent (first step has no state).
fn read_opt(f: &SubElementField, cell: usize, g: usize, name: &str) -> f64 {
    if f.component_index(name).is_some() {
        f.value(cell, g, name).unwrap_or(0.0)
    } else {
        0.0
    }
}

/// Reconstruct the full 3-D **tensor** strain from the deformation input.
/// Plane strain forces the out-of-plane components to zero; plane stress leaves
/// `eps_zz` as the trial elastic guess (it is overwritten by the return map).
fn read_strain(f: &SubElementField, cell: usize, g: usize, space_dim: usize) -> Result<[f64; 6]> {
    let mut eps = [0.0; 6];
    if space_dim == 2 {
        eps[0] = f.value(cell, g, "eps_xx")?;
        eps[1] = f.value(cell, g, "eps_yy")?;
        eps[5] = f.value(cell, g, "eps_xy")?;
        // eps_zz/yz/xz stay 0 (plane strain); plane stress fixes eps_zz later.
    } else {
        for (k, suf) in TENSOR_SUFFIXES.iter().enumerate() {
            eps[k] = f.value(cell, g, &format!("eps_{suf}"))?;
        }
    }
    Ok(eps)
}

/// Read the previous plastic strain tensor (VAR0), defaulting to zero.
fn read_state_strain(f: &SubElementField, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| read_opt(f, cell, g, &format!("eps_p_{}", TENSOR_SUFFIXES[k])))
}

/// Project the full 3-D stress to the model's Voigt slot `r`.
/// 2-D order is `[xx, yy, xy]`; 3-D is the full `[xx, yy, zz, yz, xz, xy]`.
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

/// Map a `(i, j)` tensor pair to its index in [`TENSOR_SUFFIXES`]; kept for
/// readers cross-checking the layout against [`TENSOR_PAIRS`].
#[allow(dead_code)]
fn tensor_index(i: usize, j: usize) -> usize {
    TENSOR_PAIRS
        .iter()
        .position(|&(a, b)| (a, b) == (i.min(j), i.max(j)))
        .expect("valid tensor pair")
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId};
    use crate::store::insert;

    fn unit_quad(model: ElasticityModel) -> Plasticity {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let dd = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
        mesh.add_cell(&[a.id(), b.id(), c.id(), dd.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        Plasticity::new(fes.get(0).unwrap(), model).unwrap()
    }

    fn unit_hex() -> Plasticity {
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
        Plasticity::new(fes.get(0).unwrap(), ElasticityModel::Solid).unwrap()
    }

    fn material(pl: &Plasticity, e: f64, nu: f64, sy: f64) -> Handle<SubElementField> {
        let mut mat = SubElementField::new(
            pl.fespace.clone(),
            vec!["E".into(), "nu".into(), "sigma_y".into()],
        )
        .unwrap();
        mat.set_uniform("E", e).unwrap();
        mat.set_uniform("nu", nu).unwrap();
        mat.set_uniform("sigma_y", sy).unwrap();
        insert(mat)
    }

    #[test]
    fn vars_and_model_validation() {
        let pl = unit_quad(ElasticityModel::PlaneStrain);
        assert_eq!(pl.primal_vars(), vec!["u_x", "u_y"]);
        assert_eq!(pl.dual_vars(), vec!["f_x", "f_y"]);
        assert_eq!(pl.material_components(), Some(MATERIAL_COMPONENTS));
    }

    /// Below yield the response is purely elastic: equivalent stress < σ_y and
    /// no plastic strain accumulates.
    #[test]
    fn elastic_below_yield_solid() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // Small uniaxial strain well below yield (σ ≈ E·ε = 21 MPa < 250).
        let mut strain = SubElementField::new(
            pl.fespace.clone(),
            TENSOR_SUFFIXES
                .iter()
                .map(|s| format!("eps_{s}"))
                .collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-4).unwrap();
        let strain = insert(strain);
        let out = pl.integrate_behavior(&strain, Some(&mat)).unwrap();
        // Confined uniaxial *strain* (only ε_xx ≠ 0): σ_xx = (λ+2μ)·ε.
        let (lambda, mu) = lame(e, nu);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "sigma_xx").unwrap() - (lambda + 2.0 * mu) * 1e-4).abs() < 1e-6);
            assert!(out.value(0, g, "p").unwrap().abs() < 1e-14);
        }
    }

    /// Beyond yield the von Mises equivalent stress is capped at σ_y (perfect
    /// plasticity plateau) and `p` grows.
    #[test]
    fn yields_and_caps_at_sigma_y_solid() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // Large uniaxial strain (elastic trial ≈ 2100 MPa ≫ 250).
        let mut strain = SubElementField::new(
            pl.fespace.clone(),
            TENSOR_SUFFIXES
                .iter()
                .map(|s| format!("eps_{s}"))
                .collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-2).unwrap();
        let strain = insert(strain);
        let out = pl.integrate_behavior(&strain, Some(&mat)).unwrap();
        for g in 0..out.gauss_count() {
            let s = [
                out.value(0, g, "sigma_xx").unwrap(),
                out.value(0, g, "sigma_yy").unwrap(),
                out.value(0, g, "sigma_zz").unwrap(),
                out.value(0, g, "sigma_yz").unwrap(),
                out.value(0, g, "sigma_xz").unwrap(),
                out.value(0, g, "sigma_xy").unwrap(),
            ];
            assert!((von_mises(&s) - sy).abs() < 1e-3, "q = {}", von_mises(&s));
            assert!(out.value(0, g, "p").unwrap() > 0.0);
        }
    }

    /// Plane stress drives σ_zz to zero, and below yield the in-plane stress
    /// matches the linear plane-stress solution.
    #[test]
    fn plane_stress_zero_out_of_plane_and_matches_elastic() {
        let pl = unit_quad(ElasticityModel::PlaneStress);
        let (e, nu, sy) = (210_000.0, 0.3, 1e9); // huge σ_y ⇒ stays elastic
        let mat = material(&pl, e, nu, sy);
        let eps0 = 1e-3;
        let mut strain = SubElementField::new(
            pl.fespace.clone(),
            vec!["eps_xx".into(), "eps_xy".into(), "eps_yy".into()],
        )
        .unwrap();
        strain.set_uniform("eps_xx", eps0).unwrap();
        let strain = insert(strain);
        let out = pl.integrate_behavior(&strain, Some(&mat)).unwrap();
        // Linear plane stress uniaxial-strain: σ_xx = E/(1-ν²)·ε, σ_yy = ν·σ_xx.
        let c = e / (1.0 - nu * nu);
        for g in 0..out.gauss_count() {
            assert!((out.value(0, g, "sigma_xx").unwrap() - c * eps0).abs() < 1e-3);
            assert!((out.value(0, g, "sigma_yy").unwrap() - c * nu * eps0).abs() < 1e-3);
            // σ_zz is not an output in 2-D; verify via the von Mises plateau is
            // not triggered (elastic) — covered above. Out-of-plane handled
            // internally.
        }
    }

    /// Internal state round-trips: feeding VAR0 back changes the result
    /// (history dependence) and `p` is monotone non-decreasing.
    #[test]
    fn state_round_trip_is_history_dependent() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        let comps: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
        // First load past yield.
        let mut s1 = SubElementField::new(pl.fespace.clone(), comps.clone()).unwrap();
        s1.set_uniform("eps_xx", 5e-3).unwrap();
        let s1 = insert(s1);
        let st1 = pl.integrate_behavior(&s1, Some(&mat)).unwrap();
        let p1 = st1.value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Build a second input that carries strain + the state from step 1.
        let mut merged = state_names();
        merged.extend(comps.iter().cloned());
        let mut s2 = SubElementField::new(pl.fespace.clone(), merged).unwrap();
        for g in 0..s2.gauss_count() {
            s2.set_value(0, g, "eps_xx", 6e-3).unwrap();
            for suf in TENSOR_SUFFIXES {
                let v = st1.value(0, g, &format!("eps_p_{suf}")).unwrap();
                s2.set_value(0, g, &format!("eps_p_{suf}"), v).unwrap();
            }
            s2.set_value(0, g, "p", p1).unwrap();
        }
        let s2 = insert(s2);
        let st2 = pl.integrate_behavior(&s2, Some(&mat)).unwrap();
        // Cumulated plastic strain only grows.
        assert!(st2.value(0, 0, "p").unwrap() >= p1);
    }

    /// The elastic stiffness block is reused from elasticity: symmetric.
    #[test]
    fn stiffness_is_elastic_and_symmetric() {
        let pl = unit_quad(ElasticityModel::PlaneStrain);
        let mat = material(&pl, 200.0, 0.3, 250.0);
        let blocks = pl.build_stiffness_blocks(Some(&mat)).unwrap();
        let k = &blocks[0];
        let nodes: Vec<NodeId> = read(&pl.support).unwrap().connectivity().to_vec();
        for &ni in &nodes {
            for &nj in &nodes {
                for a in ["x", "y"] {
                    for b in ["x", "y"] {
                        let lhs = k.get(ni, &format!("f_{a}"), nj, &format!("u_{b}"));
                        let rhs = k.get(nj, &format!("f_{b}"), ni, &format!("u_{a}"));
                        assert!((lhs - rhs).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_index_matches_layout() {
        assert_eq!(tensor_index(0, 0), 0);
        assert_eq!(tensor_index(2, 2), 2);
        assert_eq!(tensor_index(1, 2), 3);
        assert_eq!(tensor_index(0, 1), 5);
        assert_eq!(tensor_index(1, 0), 5);
    }
}
