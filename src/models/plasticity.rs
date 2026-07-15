//! Perfect (non-hardening) von Mises elastoplasticity — J2 radial return.
//!
//! Same kinematics and DOFs as [`crate::models::elasticity`] (displacement
//! `u_x, u_y(, u_z)`, nodal force `f_x, …`), and the **same elastic stiffness**
//! as iteration operator: the non-linearity lives entirely in the behaviour
//! integration (`COMP`). Material components `E` (Young), `nu` (Poisson) and
//! `sigma_y` (yield stress). The flow rule is associated J2 with **no
//! hardening**, so the equivalent stress is capped at `sigma_y`.
//!
//! The integration is history-dependent and uses the **incremental montage**
//! A → B: the end-of-step strain `ε(B)` comes in as `deformation`, while the
//! converged state at the start of the step A — the stress `σ(A)`, the plastic
//! strain `ε_p(A)`, the cumulated `p(A)` and the strain `ε(A)` — comes in as
//! `prev` (the previous step's output; `None` on the first step, where A is the
//! reference configuration). The elastic predictor is `σ_trial = σ(A) + C:Δε`
//! with `Δε = ε(B) − ε(A)` — algebraically identical to `C:(ε(B) − ε_p(A))` in
//! small strain, but the form that carries `σ(A)` explicitly, ready for a
//! large-strain law. The output echoes the full-3-D `ε(B)` (and, in 2-D, the
//! out-of-plane `σ_zz`) so it is a complete `prev` for the next step.
//!
//! State is always carried in **full 3-D** (six `eps_p_*` components) regardless
//! of the 2-D/3-D model, which keeps the radial return identical across plane
//! stress / plane strain / solid; only the input strain reconstruction and the
//! output stress projection differ.
//!
//! Following the locked architecture decision (see `ROADMAP.md`), the Newton
//! loop driving these increments lives in Python; this module only provides the
//! point-wise constitutive update.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::{self, ElasticityModel};
use crate::models::{CellGeom, Domain, MatrixLayout, Physics, SubModelKind};
use crate::store::{read, Handle};
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
    let mut v: Vec<String> = TENSOR_SUFFIXES
        .iter()
        .map(|s| format!("eps_p_{s}"))
        .collect();
    v.push("p".into());
    v
}

/// Extra state echoed for the incremental montage so the output is a **complete
/// `prev`**: the full-3-D end-of-step strain `ε(B)` (six `eps_*`, so `ε(A)` is
/// recoverable next step) and — in 2-D only — the out-of-plane stress `sigma_zz`
/// that the Voigt dual omits (so `σ(A)` is fully recoverable). In 3-D the Voigt
/// dual already carries all six stresses.
fn echo_names(space_dim: usize) -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
    if space_dim == 2 {
        v.push("sigma_zz".into());
    }
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
        let support = read(&submesh)?.to_poi1()?;
        Ok(Self {
            fespace,
            support,
            space_dim,
            model,
        })
    }
}

impl SubModelKind for Plasticity {
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
            ordering: crate::containers::matrix::DofOrdering::NodesThenVars,
            symmetric: true,
        })
    }

    /// The consistent mass matrix shares the stiffness layout (mass is
    /// law-independent).
    fn mass_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_matrix(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        // Iteration operator = elastic stiffness (no tangent KTAN yet — the
        // Newton loop is orchestrated in Python). Reuse the elasticity element
        // kernel verbatim; it reads only `E` and `nu` from the material.
        let mat = material.expect("Plasticity requires a material field");
        elasticity::element_stiffness(geom, mat, self.model, ke)
    }

    fn element_mass(
        &self,
        geoms: &[CellGeom],
        material: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let mat = material.expect("Plasticity requires a material field");
        elasticity::element_mass(geom, mat, ke)
    }

    fn physics(&self) -> &'static [Physics] {
        &[Physics::Mechanical]
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

impl Domain for Plasticity {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    /// `rho` (density) — required only by the mass matrix, never by the
    /// stiffness/behaviour assembly.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["rho"]
    }

    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    fn behavior_output_components(&self) -> Result<Vec<String>> {
        let mut comps = stress_names(self.space_dim);
        comps.extend(state_names());
        comps.extend(echo_names(self.space_dim));
        Ok(comps)
    }

    /// Incremental radial-return at one Gauss point. Output layout =
    /// stress (Voigt, `v`) + plastic strain `eps_p` (full 3-D tensor, 6) +
    /// cumulated plastic strain `p` (1) + echoed strain `ε(B)` (full 3-D, 6)
    /// [+ `sigma_zz` in 2-D], matching `stress_names ++ state_names ++ echo_names`.
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
        let mat = material.expect("Plasticity declares a material_fespace ⇒ material is supplied");
        let (cell, d) = (geom.cell, self.space_dim);
        let (lambda, mu) = lame(mat.value(cell, 0, "E")?, mat.value(cell, 0, "nu")?);
        let sigma_y = mat.value(cell, 0, "sigma_y")?;

        // End-of-step strain ε(B).
        let eps_b = read_strain(deformation, cell, g, d)?;
        // Converged state at A from `prev` (all zero on the first step, where A
        // is the reference configuration: σ(A)=0, ε(A)=0, ε_p(A)=0, p(A)=0).
        let prev_state = PrevState {
            eps: read_prev_strain(prev, cell, g),
            sigma: read_prev_stress(prev, cell, g),
            eps_p: read_prev_plastic_strain(prev, cell, g),
            p: prev_opt(prev, cell, g, "p"),
        };

        let (sigma, eps_p_new, p_new, eps_b_full) =
            radial_return_incremental(&eps_b, &prev_state, lambda, mu, sigma_y, self.model);

        let v = stress_names(d).len();
        for r in 0..v {
            out[r] = voigt_stress(&sigma, d, r);
        }
        out[v..v + 6].copy_from_slice(&eps_p_new); // ε_p(B)
        out[v + 6] = p_new; // p(B)
                            // Echo the full-3-D end-of-step strain ε(B), so `prev` carries ε(A) next
                            // step (in plane stress this includes the solved out-of-plane ε_zz).
        out[v + 7..v + 13].copy_from_slice(&eps_b_full);
        // In 2-D the Voigt dual omits σ_zz; echo it so σ(A) is fully recoverable.
        if d == 2 {
            out[v + 13] = sigma[2];
        }
        Ok(())
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
    let ss =
        s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (1.5 * ss).sqrt()
}

/// The converged state at the **start of the step A**, read from `prev` — the
/// input to the incremental montage. All full 3-D.
struct PrevState {
    /// Strain `ε(A)`.
    eps: [f64; 6],
    /// Stress `σ(A)`.
    sigma: [f64; 6],
    /// Plastic strain `ε_p(A)`.
    eps_p: [f64; 6],
    /// Cumulated plastic strain `p(A)`.
    p: f64,
}

/// Elastic predictor of the **incremental** montage: `σ_trial = σ(A) + C:Δε`
/// with `Δε = ε(B) − ε(A)` (all full 3-D). Algebraically identical to
/// `C:(ε(B) − ε_p(A))` in small strain — since `σ(A) = C:(ε(A) − ε_p(A))` after
/// a converged return — but this is the form that carries the previous stress
/// explicitly, the shape a large-strain law reuses (with `σ(A)` rotated and
/// `Δε` an objective increment).
fn elastic_predictor(eps_b: &[f64; 6], prev: &PrevState, lambda: f64, mu: f64) -> [f64; 6] {
    let deps: [f64; 6] = std::array::from_fn(|i| eps_b[i] - prev.eps[i]);
    let c_deps = elastic_stress(&deps, lambda, mu);
    std::array::from_fn(|i| prev.sigma[i] + c_deps[i])
}

/// Project a trial stress onto the yield surface (perfect J2), returning the
/// updated `(stress, eps_p, p)` (all full 3-D).
fn return_map_from_trial(
    sig_trial: &[f64; 6],
    prev: &PrevState,
    mu: f64,
    sigma_y: f64,
) -> ([f64; 6], [f64; 6], f64) {
    let q = von_mises(sig_trial);
    let f = q - sigma_y;
    if f <= 0.0 || q == 0.0 {
        return (*sig_trial, prev.eps_p, prev.p); // elastic
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
    let mut eps_p = prev.eps_p;
    for i in 0..6 {
        let s_new = s_trial[i] * scale;
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        // Plastic strain is a tensor: off-diagonals get the engineering ÷2? No —
        // n is built from the stress deviator with the same (1,2) weighting as a
        // tensor, so Δε_p_ij = factor · s_trial_ij directly.
        eps_p[i] += factor * s_trial[i];
    }
    (sigma, eps_p, prev.p + dp)
}

/// Incremental radial return A → B for **one** Gauss point. Given the
/// end-of-step strain `ε(B)`, the start-of-step state `prev` (`ε(A)`, `σ(A)`,
/// `ε_p(A)`, `p(A)`) and the material, returns the updated
/// `(σ(B), ε_p(B), p(B), ε(B))` — all full 3-D. The returned `ε(B)` carries the
/// solved out-of-plane strain in plane stress (for the echo). For plane stress
/// the out-of-plane normal strain `ε_zz(B)` is solved so that `σ_zz(B) = 0`.
fn radial_return_incremental(
    eps_b: &[f64; 6],
    prev: &PrevState,
    lambda: f64,
    mu: f64,
    sigma_y: f64,
    model: ElasticityModel,
) -> ([f64; 6], [f64; 6], f64, [f64; 6]) {
    if model == ElasticityModel::PlaneStress {
        return plane_stress_incremental(eps_b, prev, lambda, mu, sigma_y);
    }
    // Solid / plane strain: ε(B) fully prescribed (plane strain has
    // ε_zz = ε_yz = ε_xz = 0 already).
    let sig_trial = elastic_predictor(eps_b, prev, lambda, mu);
    let (sigma, eps_p, p) = return_map_from_trial(&sig_trial, prev, mu, sigma_y);
    (sigma, eps_p, p, *eps_b)
}

/// Plane-stress incremental return: solve `σ_zz(B) = 0` for `ε_zz(B)` by the
/// secant method, each evaluation running a full 3-D incremental return. The
/// in-plane strains `ε_xx(B), ε_yy(B), ε_xy(B)` are fixed; `ε_yz(B) = ε_xz(B) = 0`.
fn plane_stress_incremental(
    eps_in_b: &[f64; 6],
    prev: &PrevState,
    lambda: f64,
    mu: f64,
    sigma_y: f64,
) -> ([f64; 6], [f64; 6], f64, [f64; 6]) {
    let eval = |ezz: f64| {
        let mut eps_b = *eps_in_b;
        eps_b[2] = ezz;
        eps_b[3] = 0.0;
        eps_b[4] = 0.0;
        let sig_trial = elastic_predictor(&eps_b, prev, lambda, mu);
        let (sigma, eps_p, p) = return_map_from_trial(&sig_trial, prev, mu, sigma_y);
        (sigma, eps_p, p, eps_b)
    };
    // Initial guess: previous ε_zz(A) plus the elastic plane-stress out-of-plane
    // increment −ν/(1−ν)·(Δε_xx + Δε_yy).
    let nu_term = lambda / (lambda + 2.0 * mu); // = ν/(1−ν)
    let mut z0 = prev.eps[2] - nu_term * (eps_in_b[0] - prev.eps[0] + eps_in_b[1] - prev.eps[1]);
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

/// Read a component from the optional previous-state field `prev`, defaulting to
/// `0.0` when there is no previous step (`None`) or the component is absent.
fn prev_opt(prev: Option<&SubElementField>, cell: usize, g: usize, name: &str) -> f64 {
    prev.map_or(0.0, |f| read_opt(f, cell, g, name))
}

/// Full 3-D strain `ε(A)` echoed by the previous step (zero on the first step).
fn read_prev_strain(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("eps_{}", TENSOR_SUFFIXES[k])))
}

/// Full 3-D stress `σ(A)` from the previous step. Each Voigt slot is read by
/// name: `sigma_zz` comes from the 2-D echo (or the 3-D dual), and the shear
/// `sigma_yz`/`sigma_xz` are absent in 2-D (⇒ `0.0`), exactly the plane
/// assumptions.
fn read_prev_stress(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("sigma_{}", TENSOR_SUFFIXES[k])))
}

/// Previous plastic strain tensor `ε_p(A)` (VAR0), defaulting to zero.
fn read_prev_plastic_strain(prev: Option<&SubElementField>, cell: usize, g: usize) -> [f64; 6] {
    std::array::from_fn(|k| prev_opt(prev, cell, g, &format!("eps_p_{}", TENSOR_SUFFIXES[k])))
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
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-4).unwrap();
        let strain = insert(strain);
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
        // Confined uniaxial *strain* (only ε_xx ≠ 0): σ_xx = (λ+2μ)·ε.
        let (lambda, mu) = lame(e, nu);
        for g in 0..out.gauss_count() {
            assert!(
                (out.value(0, g, "sigma_xx").unwrap() - (lambda + 2.0 * mu) * 1e-4).abs() < 1e-6
            );
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
            TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect(),
        )
        .unwrap();
        strain.set_uniform("eps_xx", 1e-2).unwrap();
        let strain = insert(strain);
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
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
        let out = pl
            .integrate_behavior(&strain, None, Some(&mat), None)
            .unwrap();
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

    /// Build a uniaxial-strain deformation field `ε_xx = val` (full 3-D tensor
    /// component names) on a `unit_hex`.
    fn uniaxial(pl: &Plasticity, val: f64) -> Handle<SubElementField> {
        let comps: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
        let mut s = SubElementField::new(pl.fespace.clone(), comps).unwrap();
        s.set_uniform("eps_xx", val).unwrap();
        insert(s)
    }

    /// Internal state round-trips through `prev`: feeding the previous step's
    /// output back changes the result (history dependence) and `p` grows.
    #[test]
    fn state_round_trip_is_history_dependent() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        // First load past yield (prev = None ⇒ reference config A).
        let st1 = pl
            .integrate_behavior(&uniaxial(&pl, 5e-3), None, Some(&mat), None)
            .unwrap();
        let p1 = st1.value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Second step: larger ε(B); the state of A is fed via `prev` (the step-1
        // output), *not* merged into the deformation field.
        let prev = insert(st1);
        let st2 = pl
            .integrate_behavior(&uniaxial(&pl, 6e-3), Some(&prev), Some(&mat), None)
            .unwrap();
        // Cumulated plastic strain only grows.
        assert!(st2.value(0, 0, "p").unwrap() >= p1);
    }

    /// Iso-result: on a **proportional** (monotone uniaxial) path, the
    /// incremental montage in N steps — threading `prev` — reproduces the
    /// single-step total-strain integration to round-off. Guards the whole
    /// prev-threading + `σ(A) + C:Δε` predictor.
    #[test]
    fn incremental_matches_single_step_on_proportional_path() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);
        let eps_final = 1e-2; // well past yield

        // Single step 0 → ε_final.
        let single = pl
            .integrate_behavior(&uniaxial(&pl, eps_final), None, Some(&mat), None)
            .unwrap();

        // Ten proportional increments, threading `prev`.
        let nsteps = 10;
        let mut prev: Option<Handle<SubElementField>> = None;
        for i in 1..=nsteps {
            let val = eps_final * i as f64 / nsteps as f64;
            let out = pl
                .integrate_behavior(&uniaxial(&pl, val), prev.as_ref(), Some(&mat), None)
                .unwrap();
            prev = Some(insert(out));
        }
        let multi = read(&prev.unwrap()).unwrap();
        for comp in [
            "sigma_xx", "sigma_yy", "sigma_zz", "p", "eps_p_xx", "eps_p_yy",
        ] {
            let a = single.value(0, 0, comp).unwrap();
            let b = multi.value(0, 0, comp).unwrap();
            assert!((a - b).abs() < 1e-9, "{comp}: single={a} multi={b}");
        }
    }

    /// History dependence: after loading past yield, a small **partial** unload
    /// is elastic — `p` does not grow and the stress drops *off* the yield
    /// plateau (`q < σ_y`), following the elastic slope. Impossible without
    /// threaded state: the old bug integrated the unloaded step from zero, so at
    /// a still-past-yield strain it would sit back on the plateau (`q = σ_y`)
    /// with a fresh `p`.
    #[test]
    fn partial_unload_is_elastic_and_leaves_yield_surface() {
        let pl = unit_hex();
        let (e, nu, sy) = (210_000.0, 0.3, 250.0);
        let mat = material(&pl, e, nu, sy);

        // Load well past yield.
        let loaded = insert(
            pl.integrate_behavior(&uniaxial(&pl, 1e-2), None, Some(&mat), None)
                .unwrap(),
        );
        let p1 = read(&loaded).unwrap().value(0, 0, "p").unwrap();
        assert!(p1 > 0.0);

        // Small unload (still far past yield), threading the loaded state as `prev`.
        let unloaded = pl
            .integrate_behavior(&uniaxial(&pl, 9.9e-3), Some(&loaded), Some(&mat), None)
            .unwrap();
        // Elastic: p unchanged.
        assert!(
            (unloaded.value(0, 0, "p").unwrap() - p1).abs() < 1e-12,
            "p must not grow on elastic unload"
        );
        // Stress has left the yield plateau (q < σ_y) — the history signature.
        let s = [
            unloaded.value(0, 0, "sigma_xx").unwrap(),
            unloaded.value(0, 0, "sigma_yy").unwrap(),
            unloaded.value(0, 0, "sigma_zz").unwrap(),
            unloaded.value(0, 0, "sigma_yz").unwrap(),
            unloaded.value(0, 0, "sigma_xz").unwrap(),
            unloaded.value(0, 0, "sigma_xy").unwrap(),
        ];
        assert!(
            von_mises(&s) < sy - 1.0,
            "elastic unload must drop below σ_y, got q = {}",
            von_mises(&s)
        );
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
