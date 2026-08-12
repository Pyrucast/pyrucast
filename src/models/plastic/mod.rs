//! The shared machinery of rate-independent plasticity.
//!
//! Every elastoplastic law in pyrucast is the **same physics** — the same DOFs,
//! the same elastic stiffness as iteration operator, the same incremental
//! montage A → B, the same internal state — differing only in its **yield
//! surface** and flow rule. So the yield law is an *attribute* of the plasticity
//! physics ([`PlasticLaw`]), not a physics of its own; that mirrors Cast3M,
//! where `PLASTIQUE PARFAIT`, `PLASTIQUE ISOTROPE`, `PLASTIQUE DRUCKER_PRAGER`
//! and `PLASTIQUE OTTOSEN` are variants of one formulation.
//!
//! What lives here is everything the laws share:
//!
//! - the state at the start of the step ([`PrevState`]) and the elastic
//!   predictor `σ_trial = σ(A) + C:Δε`;
//! - the **plane-stress secant loop**, which solves `σ_zz(B) = 0` around any
//!   law by re-running it — so no law implements plane stress itself;
//! - the **cutting-plane** return mapping, for surfaces with no closed form;
//! - the **consistent tangent**, analytic where a closed form exists and by
//!   finite differences otherwise.
//!
//! State is always carried in **full 3-D** (six `eps_p_*` and a cumulated `p`)
//! whatever the 2-D model, which keeps every return map identical across plane
//! stress / plane strain / axisymmetric / solid: only the projections in and out
//! differ.
//!
//! ## Closed form where it exists, iteration where it does not
//!
//! von Mises (with or without hardening) and Drucker-Prager have closed-form
//! returns, and they use them — exact, one step, no tolerance. Ottosen's
//! four-parameter surface does not: its Lode-angle dependence makes the normal
//! `∂f/∂σ` painful to derive and easy to get subtly wrong. It goes through the
//! **cutting-plane** algorithm with a *numerically differentiated* normal, which
//! needs only the scalar `f(σ)`. The criterion is then exact and the gradient
//! accurate to a central difference — a far better trade than a hand-derived
//! gradient nobody can check.
//!
//! The same reasoning drives the **tangent**, one axis further: only von Mises
//! keeps an analytic `D_alg`, because only its closed form has been checked
//! against a finite difference. Drucker-Prager's derivation looked right and was
//! 24 % off; the numerical tangent that replaced it cannot be mis-derived, costs
//! twelve evaluations of a closed-form update, and keeps Newton quadratic. Both
//! routes are consumed identically by [`crate::ops::matrix::tangent`].
//!
//! ## Two honest limitations
//!
//! **The stored tangent is symmetric.** `D_alg` travels through the state field
//! as its upper triangle (`ktan_i_j`, i ≤ j) and is read back mirrored, so the
//! format cannot carry the genuinely non-symmetric tangent of a *non-associated*
//! law. Drucker-Prager's is therefore symmetrised — the usual engineering
//! compromise, costing Newton its quadratic rate on that law but nothing else,
//! and keeping every downstream consumer (state layout, solver, pattern cache)
//! symmetric.
//!
//! **A doubly numerical tangent is only so accurate.** Ottosen differentiates
//! `f` to get its normal, and the tangent then differentiates that whole
//! iterative map; the two error scales compound to roughly 10 % against the
//! exact derivative. Newton still converges — it needs a tangent good enough to
//! converge, not one good to machine precision — and `tests/plastic_laws.rs`
//! states the figure rather than hiding it behind a loose tolerance for
//! everything.

use crate::containers::element_field::SubElementField;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::ElasticityModel;
use serde::{Deserialize, Serialize};

pub mod drucker_prager;
pub mod ottosen;
pub mod viscous;
pub mod von_mises;

/// Full 3-D tensor component suffixes, in the internal state order
/// `[xx, yy, zz, yz, xz, xy]` (off-diagonals are **tensor** strains, `ε_ij`).
pub const TENSOR_SUFFIXES: [&str; 6] = ["xx", "yy", "zz", "yz", "xz", "xy"];

/// Which yield surface and flow rule an elastoplastic model obeys.
///
/// An attribute of the plasticity physics, not a physics of its own: the DOFs,
/// the elastic operator, the internal state and the incremental montage are the
/// same for all of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlasticLaw {
    /// von Mises with **no** hardening — the yield stress is a constant.
    #[default]
    Perfect,
    /// von Mises with linear **isotropic** hardening, `σ_y(p) = σ_y + H·p`.
    Isotropic,
    /// Drucker-Prager, pressure-sensitive, with **non-associated** flow (a
    /// dilatancy distinct from the friction) — soils, rocks, powders.
    DruckerPrager,
    /// Ottosen's four-parameter criterion — concrete, with a Lode-angle
    /// dependence that distinguishes tension from compression.
    Ottosen,
    // ── Rate-**dependent** laws. New variants go at the end: `bincode`
    // serialises the index.
    /// Norton-Odqvist secondary creep, `ṗ = (q/K)^n` — no yield threshold.
    CreepNorton,
    /// Blackburn creep: a saturating primary stage plus a steady secondary one.
    CreepBlackburn,
    /// Lemaitre primary creep, by strain hardening.
    CreepLemaitre,
    /// Chaboche viscoplasticity — kinematic (Armstrong-Frederick) and isotropic
    /// hardening, usable under cyclic loading.
    ViscoplasticChaboche,
    /// The above coupled to Lemaitre's ductile damage — tertiary creep and
    /// rupture.
    ViscoplasticLemaitreChaboche,
}

impl PlasticLaw {
    /// Parse from a lowercase tag (`"perfect"`, `"isotropic"`,
    /// `"drucker_prager"`, `"ottosen"`).
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "perfect" => Some(Self::Perfect),
            "isotropic" => Some(Self::Isotropic),
            "drucker_prager" => Some(Self::DruckerPrager),
            "ottosen" => Some(Self::Ottosen),
            "creep_norton" => Some(Self::CreepNorton),
            "creep_blackburn" => Some(Self::CreepBlackburn),
            "creep_lemaitre" => Some(Self::CreepLemaitre),
            "viscoplastic_chaboche" => Some(Self::ViscoplasticChaboche),
            "viscoplastic_lemaitre_chaboche" => Some(Self::ViscoplasticLemaitreChaboche),
            _ => None,
        }
    }

    /// The lowercase tag (the inverse of [`from_tag`](Self::from_tag)).
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Perfect => "perfect",
            Self::Isotropic => "isotropic",
            Self::DruckerPrager => "drucker_prager",
            Self::Ottosen => "ottosen",
            Self::CreepNorton => "creep_norton",
            Self::CreepBlackburn => "creep_blackburn",
            Self::CreepLemaitre => "creep_lemaitre",
            Self::ViscoplasticChaboche => "viscoplastic_chaboche",
            Self::ViscoplasticLemaitreChaboche => "viscoplastic_lemaitre_chaboche",
        }
    }

    /// Every law, in declaration order — the source of the `|`-joined tag list
    /// quoted in error messages, so a new law cannot be added without them
    /// following.
    pub const ALL: [PlasticLaw; 9] = [
        Self::Perfect,
        Self::Isotropic,
        Self::DruckerPrager,
        Self::Ottosen,
        Self::CreepNorton,
        Self::CreepBlackburn,
        Self::CreepLemaitre,
        Self::ViscoplasticChaboche,
        Self::ViscoplasticLemaitreChaboche,
    ];

    /// The accepted tags, `|`-joined — for error messages.
    pub fn tag_list() -> String {
        Self::ALL
            .iter()
            .map(|l| l.to_tag())
            .collect::<Vec<_>>()
            .join("|")
    }

    /// The material components this law requires. Elasticity (`E`, `nu`) is
    /// common to all; what follows describes the surface.
    pub fn material_components(self) -> &'static [&'static str] {
        match self {
            Self::Perfect => &["E", "nu", "sigma_y"],
            Self::Isotropic => &["E", "nu", "sigma_y", "H"],
            Self::DruckerPrager => &["E", "nu", "alpha", "k", "psi"],
            Self::Ottosen => &["E", "nu", "a", "b", "k_1", "k_2", "sigma_c"],
            Self::CreepNorton => &["E", "nu", "K", "n"],
            Self::CreepBlackburn => &["E", "nu", "A_1", "alpha_1", "r_1", "B_s", "beta_s"],
            Self::CreepLemaitre => &["E", "nu", "K", "N", "M"],
            Self::ViscoplasticChaboche => &["E", "nu", "k", "K", "n", "C_1", "gamma_1", "b", "Q"],
            Self::ViscoplasticLemaitreChaboche => &[
                "E", "nu", "k", "K", "n", "C_1", "gamma_1", "b", "Q", "S", "s", "D_c",
            ],
        }
    }

    /// Whether this law has an **analytic consistent tangent** that has been
    /// validated against a finite difference. The others take the numerical
    /// route.
    ///
    /// Only von Mises qualifies, and deliberately: Drucker-Prager's hand
    /// derivation was 24 % off — plausible, and wrong — until
    /// `tests/plastic_laws.rs` caught it. A tangent nobody can check is worth
    /// less than one that costs twelve extra evaluations.
    fn has_analytic_tangent(self) -> bool {
        matches!(self, Self::Perfect | Self::Isotropic)
    }

    /// The law's **own** internal variables, beyond `ε_p` and `p`. Empty for a
    /// law that needs nothing more; a back stress or a damage otherwise.
    ///
    /// These become extra components of the behaviour output, so a law can grow
    /// its state without any other file changing.
    pub fn internal_names(self) -> Vec<String> {
        match self {
            Self::Perfect
            | Self::Isotropic
            | Self::DruckerPrager
            | Self::Ottosen
            | Self::CreepNorton
            | Self::CreepLemaitre => Vec::new(),
            // The primary creep strain, tracked apart from the total so the law
            // integrates correctly under a varying load.
            Self::CreepBlackburn => vec!["p_prim".to_string()],
            // The back stress (a full tensor) and the isotropic drag.
            Self::ViscoplasticChaboche => back_stress_names(false),
            // …plus the damage.
            Self::ViscoplasticLemaitreChaboche => back_stress_names(true),
        }
    }

    /// Whether this law is **rate-dependent** — it needs the time increment, and
    /// erroring without one is better than silently integrating a viscous law as
    /// if it were instantaneous.
    pub fn is_viscous(self) -> bool {
        matches!(
            self,
            Self::CreepNorton
                | Self::CreepBlackburn
                | Self::CreepLemaitre
                | Self::ViscoplasticChaboche
                | Self::ViscoplasticLemaitreChaboche
        )
    }

    /// Project a trial stress onto this law's yield surface.
    ///
    /// `dt` is the time increment: `None` for a rate-independent law, and
    /// **required** by a viscous one.
    pub fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: Option<f64>,
    ) -> Result<PlasticStep> {
        if self.is_viscous() && dt.is_none() {
            return Err(PyrucastError::Message(format!(
                "plasticity ({self}): this law is rate-dependent and needs a time increment —                  pass `dt` to integrate_behavior"
            )));
        }
        match self {
            Self::Perfect => von_mises::return_map(trial, prev, mat, 0.0),
            Self::Isotropic => von_mises::return_map(trial, prev, mat, mat.get("H")?),
            Self::DruckerPrager => drucker_prager::return_map(trial, prev, mat),
            Self::Ottosen => ottosen::return_map(trial, prev, mat),
            // Viscous from here on: `dt` is present, the guard above saw to it.
            Self::CreepNorton => viscous::norton(trial, prev, mat, dt.unwrap()),
            Self::CreepBlackburn => viscous::blackburn(trial, prev, mat, dt.unwrap()),
            Self::CreepLemaitre => viscous::lemaitre(trial, prev, mat, dt.unwrap()),
            Self::ViscoplasticChaboche => viscous::chaboche(trial, prev, mat, dt.unwrap(), false),
            Self::ViscoplasticLemaitreChaboche => {
                viscous::chaboche(trial, prev, mat, dt.unwrap(), true)
            }
        }
    }
}

impl std::fmt::Display for PlasticLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

/// The internal-variable names of a Chaboche-family law: the back stress
/// (a full 3-D tensor), the isotropic drag, and optionally the damage.
fn back_stress_names(damage: bool) -> Vec<String> {
    let mut names: Vec<String> = TENSOR_SUFFIXES.iter().map(|s| format!("X_{s}")).collect();
    names.push("R".to_string());
    if damage {
        names.push("damage".to_string());
    }
    names
}

// ─── Material parameters at one Gauss point ─────────────────────────────────

/// The material a law reads, resolved for one cell.
///
/// The elastic constants are pre-computed (every law needs them); the rest is
/// looked up by name, so adding a law adds no plumbing here.
pub struct MatParams<'a> {
    /// Lamé's first coefficient.
    pub lambda: f64,
    /// Shear modulus.
    pub mu: f64,
    material: &'a SubElementField,
    cell: usize,
}

impl<'a> MatParams<'a> {
    /// Read `E` and `nu` for this cell and pre-compute the Lamé coefficients.
    pub fn new(material: &'a SubElementField, cell: usize) -> Result<Self> {
        let (lambda, mu) = lame(
            material.value(cell, 0, "E")?,
            material.value(cell, 0, "nu")?,
        );
        Ok(Self {
            lambda,
            mu,
            material,
            cell,
        })
    }

    /// A material component of this cell, by name.
    pub fn get(&self, name: &str) -> Result<f64> {
        self.material.value(self.cell, 0, name)
    }

    /// Bulk modulus `K = λ + 2μ/3`.
    pub fn bulk(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }
}

// ─── Elastic kinematics, shared by every law ────────────────────────────────

/// Lamé coefficients `(λ, μ)` from `E`, `nu`.
pub fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic stress (full 3-D, order `[xx, yy, zz, yz, xz, xy]`) from a
/// **tensor** strain: `σ = λ tr(ε) I + 2μ ε`.
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

/// First invariant `I₁ = tr(σ)`.
pub fn i1(sigma: &[f64; 6]) -> f64 {
    sigma[0] + sigma[1] + sigma[2]
}

/// The stress deviator `s = σ − (I₁/3)·I` (same Voigt order).
pub fn deviator(sigma: &[f64; 6]) -> [f64; 6] {
    let mean = i1(sigma) / 3.0;
    [
        sigma[0] - mean,
        sigma[1] - mean,
        sigma[2] - mean,
        sigma[3],
        sigma[4],
        sigma[5],
    ]
}

/// Second deviatoric invariant `J₂ = ½ s:s` (off-diagonals counted twice).
pub fn j2(sigma: &[f64; 6]) -> f64 {
    let s = deviator(sigma);
    0.5 * (s[0] * s[0]
        + s[1] * s[1]
        + s[2] * s[2]
        + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
}

/// Third deviatoric invariant `J₃ = det(s)`.
pub fn j3(sigma: &[f64; 6]) -> f64 {
    let s = deviator(sigma);
    // det of the symmetric tensor [[s0, s5, s4], [s5, s1, s3], [s4, s3, s2]].
    s[0] * (s[1] * s[2] - s[3] * s[3]) - s[5] * (s[5] * s[2] - s[3] * s[4])
        + s[4] * (s[5] * s[3] - s[1] * s[4])
}

/// von Mises equivalent stress `q = √(3 J₂)`.
pub fn von_mises_stress(sigma: &[f64; 6]) -> f64 {
    (3.0 * j2(sigma)).sqrt()
}

/// The converged state at the **start of the step A** — the input to the
/// incremental montage. All full 3-D.
#[derive(Clone, Default)]
pub struct PrevState {
    /// Strain `ε(A)`.
    pub eps: [f64; 6],
    /// Stress `σ(A)`.
    pub sigma: [f64; 6],
    /// Plastic strain `ε_p(A)`.
    pub eps_p: [f64; 6],
    /// Cumulated plastic strain `p(A)`.
    pub p: f64,
    /// The law's **own** internal variables at A, in
    /// [`PlasticLaw::internal_names`] order — a back stress, a damage, whatever
    /// the law carries beyond `ε_p` and `p`. Empty for the laws that carry
    /// nothing more, which is most of them.
    pub vars: Vec<f64>,
}

impl PrevState {
    /// Internal variable `i`, or `0` when the state does not carry it (the first
    /// step, where A is the reference configuration).
    pub fn var(&self, i: usize) -> f64 {
        self.vars.get(i).copied().unwrap_or(0.0)
    }
}

/// The updated state at the **end of the step B**, as a law returns it.
pub struct PlasticStep {
    /// Stress `σ(B)`, full 3-D.
    pub sigma: [f64; 6],
    /// Plastic strain `ε_p(B)`, full 3-D.
    pub eps_p: [f64; 6],
    /// Cumulated plastic strain `p(B)`.
    pub p: f64,
    /// The law's own internal variables at B, in [`PlasticLaw::internal_names`]
    /// order.
    pub vars: Vec<f64>,
}

impl PlasticStep {
    /// An **elastic** step: the trial stress stands, nothing evolves.
    pub fn elastic(trial: &[f64; 6], prev: &PrevState) -> Self {
        Self {
            sigma: *trial,
            eps_p: prev.eps_p,
            p: prev.p,
            vars: prev.vars.clone(),
        }
    }
}

/// Elastic predictor of the **incremental** montage: `σ_trial = σ(A) + C:Δε`
/// with `Δε = ε(B) − ε(A)`.
///
/// Algebraically identical to `C:(ε(B) − ε_p(A))` in small strain, but this is
/// the form that carries `σ(A)` explicitly — the shape a large-strain law reuses
/// (with `σ(A)` rotated and `Δε` an objective increment).
pub fn elastic_predictor(eps_b: &[f64; 6], prev: &PrevState, lambda: f64, mu: f64) -> [f64; 6] {
    let deps: [f64; 6] = std::array::from_fn(|i| eps_b[i] - prev.eps[i]);
    let c_deps = elastic_stress(&deps, lambda, mu);
    std::array::from_fn(|i| prev.sigma[i] + c_deps[i])
}

// ─── The incremental step, around any law ───────────────────────────────────

/// One incremental step A → B at a Gauss point, for **any** law.
///
/// Returns `(σ(B), ε_p(B), p(B), ε(B))`. The returned strain is what should be
/// echoed as `ε(A)` of the next step: in plane stress it carries the
/// out-of-plane `ε_zz` solved here, which the caller could not know.
pub fn incremental_step(
    law: PlasticLaw,
    eps_b: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    model: ElasticityModel,
    dt: Option<f64>,
) -> Result<(PlasticStep, [f64; 6])> {
    if model == ElasticityModel::PlaneStress {
        return plane_stress_step(law, eps_b, prev, mat, dt);
    }
    // Solid / plane strain / axisymmetric: ε(B) is fully prescribed.
    let trial = elastic_predictor(eps_b, prev, mat.lambda, mat.mu);
    Ok((law.return_map(&trial, prev, mat, dt)?, *eps_b))
}

/// Plane stress, around any law: solve `σ_zz(B) = 0` for `ε_zz(B)` by the secant
/// method, each evaluation running a full 3-D return.
///
/// Written once here rather than in each law: the out-of-plane condition is a
/// property of the **kinematics**, not of the yield surface, so no law should
/// have to know about it.
fn plane_stress_step(
    law: PlasticLaw,
    eps_in_b: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: Option<f64>,
) -> Result<(PlasticStep, [f64; 6])> {
    let eval = |ezz: f64| -> Result<(PlasticStep, [f64; 6])> {
        let mut eps_b = *eps_in_b;
        eps_b[2] = ezz;
        eps_b[3] = 0.0;
        eps_b[4] = 0.0;
        let trial = elastic_predictor(&eps_b, prev, mat.lambda, mat.mu);
        Ok((law.return_map(&trial, prev, mat, dt)?, eps_b))
    };
    // Initial guess: ε_zz(A) plus the elastic plane-stress out-of-plane
    // increment −ν/(1−ν)·(Δε_xx + Δε_yy).
    let nu_term = mat.lambda / (mat.lambda + 2.0 * mat.mu); // = ν/(1−ν)
    let mut z0 = prev.eps[2] - nu_term * (eps_in_b[0] - prev.eps[0] + eps_in_b[1] - prev.eps[1]);
    let mut z1 = z0 + 1e-6_f64.max(z0.abs() * 1e-3);
    let mut f0 = eval(z0)?.0.sigma[2];
    let mut f1 = eval(z1)?.0.sigma[2];
    for _ in 0..50 {
        if f1.abs() < 1e-10 * (mat.mu + 1.0) {
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
        f1 = eval(z1)?.0.sigma[2];
    }
    eval(z1)
}

// ─── The consistent tangent ─────────────────────────────────────────────────

/// The full-3-D engineering-Voigt consistent tangent `D_alg = ∂σ(B)/∂ε(B)`.
///
/// Analytic for von Mises, whose closed form is validated against a finite
/// difference; **numerical** for the others. Both are exact enough for Newton to
/// converge quadratically, and the numerical route costs twelve evaluations of a
/// cheap update — far less than a hand-derived tangent nobody can check, which
/// in the Drucker-Prager case turned out to be 24 % wrong.
pub fn consistent_tangent(
    law: PlasticLaw,
    eps_b: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: Option<f64>,
) -> Result<[[f64; 6]; 6]> {
    let d = raw_consistent_tangent(law, eps_b, prev, mat, dt)?;
    // `D_alg` travels through the state field as its **upper triangle**
    // (`ktan_i_j`, i ≤ j) and is read back mirrored, so the format can only
    // carry a symmetric tangent. Non-associated flow produces a genuinely
    // non-symmetric one, which is therefore **symmetrised** here — the usual
    // engineering compromise, and stated rather than hidden.
    //
    // The cost is Newton's *quadratic* rate on a non-associated law; it still
    // converges, one order slower. The gain is that every consumer of a tangent
    // — the state layout, the solver, the pattern cache — stays symmetric. For
    // an associated law (von Mises, Ottosen) this is a no-op, bit for bit.
    Ok(symmetrise(d))
}

/// `½(D + Dᵀ)` — exact, and the identity on an already-symmetric matrix.
fn symmetrise(d: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = 0.5 * (d[i][j] + d[j][i]);
        }
    }
    out
}

/// The tangent before symmetrisation — analytic where validated, numerical
/// otherwise.
fn raw_consistent_tangent(
    law: PlasticLaw,
    eps_b: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: Option<f64>,
) -> Result<[[f64; 6]; 6]> {
    if law.has_analytic_tangent() {
        let trial = elastic_predictor(eps_b, prev, mat.lambda, mat.mu);
        let hardening = match law {
            PlasticLaw::Perfect => 0.0,
            PlasticLaw::Isotropic => mat.get("H")?,
            _ => unreachable!("only von Mises has an analytic tangent"),
        };
        return Ok(von_mises::tangent(&trial, mat, hardening, prev.p));
    }
    finite_difference_tangent(law, eps_b, prev, mat, dt)
}

/// `∂σ/∂ε` by central differences on the return map, in engineering Voigt.
///
/// The perturbation is applied to the **tensor** strain; the shear columns are
/// halved on the way out, which is exactly what turns `∂σ/∂ε_ij` into
/// `∂σ/∂γ_ij`.
fn finite_difference_tangent(
    law: PlasticLaw,
    eps_b: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    dt: Option<f64>,
) -> Result<[[f64; 6]; 6]> {
    // A strain-sized step: relative to the strain itself when it is meaningful,
    // to the elastic strain scale otherwise.
    // The step is relative to the strain itself. It must stay well above the
    // noise of the return map (an iterative one for some laws converges to a
    // tolerance, not exactly) and well below the curvature scale of the surface;
    // `1e-6·‖ε‖` sits comfortably between the two.
    let scale = eps_b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-8);
    let h = 1e-6 * scale;
    let mut d = [[0.0; 6]; 6];
    for j in 0..6 {
        let run = |sign: f64| -> Result<[f64; 6]> {
            let mut e = *eps_b;
            e[j] += sign * h;
            let trial = elastic_predictor(&e, prev, mat.lambda, mat.mu);
            Ok(law.return_map(&trial, prev, mat, dt)?.sigma)
        };
        let (sp, sm) = (run(1.0)?, run(-1.0)?);
        // Engineering shear: γ = 2ε, so a column against a tensor shear is twice
        // the column against the engineering one.
        let factor = if j < 3 { 1.0 } else { 0.5 };
        for i in 0..6 {
            d[i][j] = factor * (sp[i] - sm[i]) / (2.0 * h);
        }
    }
    Ok(d)
}

/// The elastic modulus in full-3-D engineering Voigt — the tangent wherever the
/// step stayed elastic, and the starting point of every analytic one.
pub fn elastic_tangent(lambda: f64, mu: f64) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = if i == j { lambda + 2.0 * mu } else { lambda };
        }
    }
    for i in 3..6 {
        c[i][i] = mu;
    }
    c
}

/// Guard a law's material against a value that would make its return map
/// meaningless, with a message naming the law and the constant.
pub fn require_positive(law: PlasticLaw, name: &str, value: f64) -> Result<f64> {
    if value <= 0.0 {
        return Err(PyrucastError::Message(format!(
            "plasticity ({law}): {name} = {value} must be positive"
        )));
    }
    Ok(value)
}
