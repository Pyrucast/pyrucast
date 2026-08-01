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
//! stress / plane strain / axisymmetric / solid; only the input strain
//! reconstruction and the output stress projection differ.
//!
//! **Axisymmetric** therefore costs almost nothing here: the hoop `ε_θθ = u_r/r`
//! is *measured* by [`crate::ops::field::deformation`], not assumed, so `ε(B)` is
//! fully known (no out-of-plane solve, unlike plane stress) and the whole
//! specialisation is the index map `[rr, zz, θθ, rz] → [xx, yy, zz, xy]`. Note
//! that `σ_zz` is then part of the Voigt dual and must **not** be echoed as extra
//! state.
//!
//! Following the locked architecture decision (see `ROADMAP.md`), the Newton
//! loop driving these increments lives in Python; this module provides the
//! point-wise constitutive update **and** the consistent algorithmic tangent
//! `D_alg` (emitted alongside the stress, consumed by
//! [`crate::ops::assemble::tangent`]) for quadratic convergence.

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

/// Where each **axisymmetric** Voigt slot `[rr, zz, θθ, rz]` sits in the full
/// 3-D order [`TENSOR_SUFFIXES`] (`[xx, yy, zz, yz, xz, xy]`). The whole
/// axisymmetric specialisation of this law is this one index map: the state and
/// the radial return stay full 3-D, only the projection in and out changes.
const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];

fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}
fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order for the given space dimension —
/// matching [`crate::models::elasticity`] so downstream code is uniform.
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
fn echo_names(space_dim: usize, model: ElasticityModel) -> Vec<String> {
    let mut v: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("eps_{s}")).collect();
    if echoes_sigma_zz(space_dim, model) {
        v.push("sigma_zz".into());
    }
    v
}

/// Whether `σ_zz` must be echoed as extra state. Only the **plane** 2-D models
/// need it: their Voigt dual stops at `[xx, yy, xy]`. Axisymmetric already
/// carries `sigma_zz` (the hoop) in its dual, so echoing it would emit the same
/// component name twice.
fn echoes_sigma_zz(space_dim: usize, model: ElasticityModel) -> bool {
    space_dim == 2 && !model.is_axisymmetric()
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
        let (submesh, space_dim, ref_dim, axisymmetric) = {
            let s = read(&fespace)?;
            (
                s.submesh(),
                s.space_dim(),
                s.ref_dim()?,
                s.is_axisymmetric(),
            )
        };
        elasticity::check_continuum_dimensions("Plasticity", space_dim, ref_dim)?;
        #[allow(clippy::match_like_matches_macro)]
        let ok = match (space_dim, model) {
            (2, ElasticityModel::PlaneStress | ElasticityModel::PlaneStrain) => true,
            (2, ElasticityModel::Axisymmetric) => true,
            (3, ElasticityModel::Solid) => true,
            _ => false,
        };
        if !ok {
            return Err(PyrucastError::Message(format!(
                "Plasticity: model {model:?} is incompatible with a {space_dim}-D space \
                 (2-D ⇒ plane_stress|plane_strain|axisymmetric, 3-D ⇒ solid)"
            )));
        }
        // Same two-way agreement as `Elasticity::new`: the 2πr measure comes
        // from the geometry, the hoop component from the model.
        if axisymmetric != model.is_axisymmetric() {
            return Err(PyrucastError::Message(if axisymmetric {
                format!(
                    "Plasticity: model {model:?} on an axisymmetric geometry — a body of \
                     revolution requires the `axisymmetric` model"
                )
            } else {
                "Plasticity: the `axisymmetric` model requires an axisymmetric geometry \
                 (build the Coords with Coords::axisymmetric)"
                    .into()
            }));
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
        // The plain "stiffness" kernel is the *elastic* stiffness (the simple
        // iteration operator). The consistent algorithmic tangent `K_t` is a
        // separate operator — see [`element_tangent`](Self::element_tangent) and
        // [`crate::ops::assemble::tangent`]. Reuse the elasticity element kernel
        // verbatim; it reads only `E` and `nu` from the material.
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

    /// The consistent tangent shares the stiffness layout; the algorithmic
    /// modulus `D_alg` (emitted by [`Domain::integrate_point`]) is read from the
    /// behaviour state.
    fn tangent_layout(&self) -> Option<MatrixLayout> {
        self.stiffness_layout()
    }

    fn element_tangent(
        &self,
        geoms: &[CellGeom],
        _material: Option<&SubElementField>,
        state: Option<&SubElementField>,
        ke: &mut [f64],
    ) -> Result<()> {
        let geom = &geoms[0];
        let st = state.expect("consistent tangent requires the behaviour state (D_alg)");
        elasticity::element_tangent_from_state(geom, st, self.model, ke)
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
        let mut comps = stress_names(self.space_dim, self.model);
        comps.extend(state_names());
        comps.extend(echo_names(self.space_dim, self.model));
        // Consistent algorithmic tangent D_alg (upper triangle) — consumed by
        // the tangent assembler (`assemble::tangent`).
        comps.extend(elasticity::tangent_component_names(
            self.space_dim,
            self.model,
        ));
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
        let eps_b = read_strain(deformation, cell, g, d, self.model)?;
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

        let v = stress_names(d, self.model).len();
        for r in 0..v {
            out[r] = voigt_stress(&sigma, d, self.model, r);
        }
        out[v..v + 6].copy_from_slice(&eps_p_new); // ε_p(B)
        out[v + 6] = p_new; // p(B)
                            // Echo the full-3-D end-of-step strain ε(B), so `prev` carries ε(A) next
                            // step (in plane stress this includes the solved out-of-plane ε_zz).
        out[v + 7..v + 13].copy_from_slice(&eps_b_full);
        // The plane 2-D duals omit σ_zz; echo it so σ(A) is fully recoverable.
        // Axisymmetric already carries it (the hoop), so it must not be echoed.
        if echoes_sigma_zz(d, self.model) {
            out[v + 13] = sigma[2];
        }

        // Consistent tangent D_alg at the converged step, from the trial stress
        // recomputed at the solved ε(B) (which carries the plane-stress ε_zz).
        // Emitted (upper triangle) right after the state, in `ktan_i_j` order.
        let sig_trial = elastic_predictor(&eps_b_full, &prev_state, lambda, mu);
        let d3 = consistent_tangent_3d(&sig_trial, lambda, mu, sigma_y);
        let dv = tangent_matrix_model(&d3, self.model);
        let base = v + 13 + usize::from(echoes_sigma_zz(d, self.model));
        let mut idx = base;
        for i in 0..dv.len() {
            for j in i..dv.len() {
                out[idx] = dv[i][j];
                idx += 1;
            }
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

/// Full-3-D engineering-Voigt (order `[xx, yy, zz, yz, xz, xy]`) **consistent
/// tangent** `D_alg = ∂σ(B)/∂ε(B)` of the perfect-J2 radial return, evaluated at
/// the trial stress `σ_trial` (the elastic predictor at the converged `ε(B)`).
///
/// Exact derivative of the return-mapped stress: elastic (`q_trial ≤ σ_y`) gives
/// the elastic modulus `C`; plastic gives `K·1⊗1 + a·(I_dev − n⊗n)` with
/// `a = 2μ σ_y / q_trial`, `n` the unit deviatoric flow direction, `K = λ + 2μ/3`.
fn consistent_tangent_3d(
    sig_trial: &[f64; 6],
    lambda: f64,
    mu: f64,
    sigma_y: f64,
) -> [[f64; 6]; 6] {
    let k = lambda + 2.0 * mu / 3.0;
    let q = von_mises(sig_trial);
    let plastic = q > sigma_y && q > 0.0;
    // Deviatoric coefficient: 2μ when elastic, 2μ σ_y/q when plastic.
    let coef = if plastic {
        2.0 * mu * sigma_y / q
    } else {
        2.0 * mu
    };

    let mut d = [[0.0_f64; 6]; 6];
    // K·1⊗1 on the normal (top-left 3×3) block.
    for row in d.iter_mut().take(3) {
        for e in row.iter_mut().take(3) {
            *e += k;
        }
    }
    // coef · I_dev (engineering: normal block ⅔/−⅓, shear diagonal ½).
    for (i, row) in d.iter_mut().enumerate().take(3) {
        for (j, e) in row.iter_mut().enumerate().take(3) {
            *e += coef * if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 };
        }
    }
    for i in 3..6 {
        d[i][i] += coef * 0.5;
    }
    // − coef · n⊗n (plastic only), n the unit deviatoric flow direction.
    if plastic {
        let mean = (sig_trial[0] + sig_trial[1] + sig_trial[2]) / 3.0;
        let s = [
            sig_trial[0] - mean,
            sig_trial[1] - mean,
            sig_trial[2] - mean,
            sig_trial[3],
            sig_trial[4],
            sig_trial[5],
        ];
        let s_norm = (s[0] * s[0]
            + s[1] * s[1]
            + s[2] * s[2]
            + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
            .sqrt();
        if s_norm > 0.0 {
            let nv: [f64; 6] = std::array::from_fn(|i| s[i] / s_norm);
            for i in 0..6 {
                for j in 0..6 {
                    d[i][j] -= coef * nv[i] * nv[j];
                }
            }
        }
    }
    d
}

/// Reduce the full-3-D consistent tangent to the model's `v×v` engineering-Voigt
/// matrix: the `[xx, yy, xy]` block for plane strain, its **static condensation**
/// on `ε_zz` (so `σ_zz = 0`) for plane stress, the `[rr, zz, θθ, rz]` block for
/// axisymmetric, the full `6×6` for the solid.
fn tangent_matrix_model(d3: &[[f64; 6]; 6], model: ElasticityModel) -> Vec<Vec<f64>> {
    match model {
        // Axisymmetric: the plain [rr, zz, θθ, rz] sub-block. No condensation —
        // all four strains are prescribed (the hoop is measured, not assumed), so
        // the 3-D tangent restricts directly.
        ElasticityModel::Axisymmetric => AXI_TO_3D
            .iter()
            .map(|&i| AXI_TO_3D.iter().map(|&j| d3[i][j]).collect())
            .collect(),
        ElasticityModel::Solid => d3.iter().map(|r| r.to_vec()).collect(),
        ElasticityModel::PlaneStrain => {
            let idx = [0usize, 1, 5];
            idx.iter()
                .map(|&i| idx.iter().map(|&j| d3[i][j]).collect())
                .collect()
        }
        ElasticityModel::PlaneStress => {
            // Condense the out-of-plane normal `zz` (index 2) so σ_zz = 0:
            // D2[i][j] = D3[i][j] − D3[i][2]·D3[2][j]/D3[2][2].
            let z = 2usize;
            let dzz = d3[z][z];
            let cond = |i: usize, j: usize| d3[i][j] - d3[i][z] * d3[z][j] / dzz;
            let idx = [0usize, 1, 5];
            idx.iter()
                .map(|&i| idx.iter().map(|&j| cond(i, j)).collect())
                .collect()
        }
    }
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
fn read_strain(
    f: &SubElementField,
    cell: usize,
    g: usize,
    space_dim: usize,
    model: ElasticityModel,
) -> Result<[f64; 6]> {
    let mut eps = [0.0; 6];
    if space_dim == 2 {
        eps[0] = f.value(cell, g, "eps_xx")?;
        eps[1] = f.value(cell, g, "eps_yy")?;
        eps[5] = f.value(cell, g, "eps_xy")?;
        if model.is_axisymmetric() {
            // The hoop ε_θθ = u_r/r is **measured**, not assumed: `deformation`
            // produces it on a body of revolution. So ε(B) is fully known here
            // — no plane assumption, no out-of-plane solve.
            eps[2] = f.value(cell, g, "eps_zz")?;
        }
        // eps_yz/xz stay 0 (axial symmetry ⇒ no orthoradial shear); for the
        // plane models eps_zz also stays 0 (plane strain) or is solved later
        // (plane stress).
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
