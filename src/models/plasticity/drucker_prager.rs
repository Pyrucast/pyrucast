//! Drucker-Prager plasticity — pressure-sensitive, with non-associated flow,
//! hardening, and a limiting surface.
//!
//! Soils, rocks, concrete and powders are **stronger in compression than in
//! tension**: their yielding depends on the hydrostatic pressure, which von
//! Mises ignores entirely. Drucker-Prager is the simplest surface that captures
//! it — a cone about the hydrostatic axis:
//!
//! ```text
//! f(σ) = α·I₁ + β·q − k          q = √(3 J₂),  I₁ = tr σ
//! ```
//!
//! `α` is the friction coefficient — the cone's slope, `α = 0` giving back von
//! Mises — `β` weights the deviatoric part, and `k` the cohesion. `α` travels
//! through the material field as **`friction`**: `alpha` is the thermal
//! expansion everywhere else in the crate, and a single component name cannot
//! carry two meanings.
//!
//! ## Why the flow is non-associated
//!
//! Associated flow on this cone (`g = f`) would make the material dilate under
//! shear by exactly the amount its friction implies — which for real granular
//! media is far too much. So the plastic potential carries its **own** slope,
//! the dilatancy `ψ`, and its own deviatoric weight `δ`:
//!
//! ```text
//! g(σ) = ψ·I₁ + δ·q          ψ ≤ α
//! ```
//!
//! `ψ = α, δ = β` recovers associated flow; `ψ = 0` gives isochoric plastic flow
//! with frictional strength. The price is a **non-symmetric** consistent tangent
//! — which the assembler already supports, `MatrixLayout` carrying a `symmetric`
//! flag.
//!
//! ## Hardening, and the surface it stops at
//!
//! The cohesion grows with the accumulated plastic strain, `dk = H·dp`, `H`
//! algebraic so that a negative one softens. Left alone that would grow without
//! bound, so a second **ultimate** surface bounds it:
//!
//! ```text
//! initial   α·I₁ + β·q = k
//! ultimate  α_u·I₁ + β_u·q = k_u
//! ```
//!
//! The two are read as the ends of one interpolation, driven by the hardening
//! itself:
//!
//! ```text
//! λ = clamp( H·p / (k_u − k), 0, 1 )
//! α(p) = (1−λ)α + λα_u        β(p) = (1−λ)β + λβ_u        k(p) = (1−λ)k + λk_u
//! ```
//!
//! which is worth stating plainly because it is an **interpretation**: Cast3M's
//! `PLASTIQUE DRUCKER_PRAGER` gives the two surfaces and `dK = H·dp` without
//! saying how they meet. Reading them as one interpolating surface has two
//! merits — `k(p) = k + H·p` exactly while the limit is not reached, so the
//! stated hardening law is reproduced verbatim; and the yield surface stays
//! **single**, so the return map keeps one cone and one apex instead of growing
//! a corner between two.
//!
//! ## What the parameters reduce to
//!
//! Six of the nine are optional, and their defaults are what makes the simple
//! cone the zero-configuration case:
//!
//! | Cast3M | here | défaut | ce qu'il fait |
//! |---|---|---|---|
//! | `ALFA` | `friction` | requis | pente du cône |
//! | `K` | `k` | requis | cohésion |
//! | `GAMM` | `psi` | requis | dilatance du potentiel |
//! | `BETA` | `beta` | 1 | poids déviatorique du critère |
//! | `DELT` | `delta` | 1 | poids déviatorique du potentiel |
//! | `H` | `H` | 0 | module d'écrouissage |
//! | `ETA` | `friction_ult` | `friction` | pente de la surface ultime |
//! | `MU` | `beta_ult` | `beta` | poids déviatorique ultime |
//! | `KL` | `k_ult` | `k` | cohésion ultime |
//!
//! With every default taken, `k_u = k` leaves no room to harden, the
//! interpolation is frozen at the initial surface, and what remains is the
//! perfectly plastic cone. Cast3M's **`DRUCKER_PARFAIT`** is then one more step:
//! `psi = friction` (and `delta = beta`), which makes the flow associated.
//!
//! ## The apex
//!
//! A cone has a tip, at `I₁ = k/α`, and a trial stress beyond it returns to that
//! point rather than to the cone's flank — the smooth return would otherwise
//! overshoot into a stress state with a negative equivalent stress, which has no
//! meaning. Detecting it is the one branch this law needs, and it is exactly the
//! case a naive implementation gets wrong under strong tension.

use super::law::PlasticLawKind;
use crate::error::Result;
use crate::models::elasticity::ElasticityModel;
use crate::models::plasticity::law::{
    deviator, i1, require_positive, von_mises_stress, MatParams, PlasticLaw, PlasticStep, PrevState,
};

/// The nine parameters of the general surface.
#[derive(Clone, Copy)]
struct Params {
    /// Initial surface `α·I₁ + β·q = k`.
    friction: f64,
    beta: f64,
    k: f64,
    /// Ultimate surface, the interpolation's far end.
    friction_ult: f64,
    beta_ult: f64,
    k_ult: f64,
    /// Hardening modulus, algebraic.
    hardening: f64,
    /// Flow potential `ψ·I₁ + δ·q`.
    psi: f64,
    delta: f64,
}

/// An optional constitutive parameter, or its default.
///
/// Absent from the material field means « take the default », not an error: the
/// three-parameter cone is the common case and must stay writable in three
/// numbers.
fn optional(mat: &MatParams, name: &str, default: f64) -> f64 {
    mat.get(name).unwrap_or(default)
}

/// Read the nine and reject a non-physical set.
fn params(mat: &MatParams) -> Result<Params> {
    let friction = mat.get("friction")?;
    let k = require_positive(PlasticLaw::DruckerPrager, "k", mat.get("k")?)?;
    let psi = mat.get("psi")?;
    let beta = require_positive(
        PlasticLaw::DruckerPrager,
        "beta",
        optional(mat, "beta", 1.0),
    )?;
    let beta_ult = require_positive(
        PlasticLaw::DruckerPrager,
        "beta_ult",
        optional(mat, "beta_ult", beta),
    )?;
    Ok(Params {
        friction,
        beta,
        k,
        friction_ult: optional(mat, "friction_ult", friction),
        beta_ult,
        k_ult: require_positive(
            PlasticLaw::DruckerPrager,
            "k_ult",
            optional(mat, "k_ult", k),
        )?,
        hardening: optional(mat, "H", 0.0),
        psi,
        delta: require_positive(
            PlasticLaw::DruckerPrager,
            "delta",
            optional(mat, "delta", 1.0),
        )?,
    })
}

impl Params {
    /// Whether the surface can move at all — the fast path when it cannot.
    fn hardens(&self) -> bool {
        self.hardening != 0.0 && (self.k_ult - self.k).abs() > f64::EPSILON
    }

    /// The interpolation's progress at an accumulated plastic strain `p`.
    fn lambda(&self, p: f64) -> f64 {
        if !self.hardens() {
            return 0.0;
        }
        (self.hardening * p / (self.k_ult - self.k)).clamp(0.0, 1.0)
    }

    /// The surface `(α, β, k)` currently active.
    fn surface(&self, p: f64) -> (f64, f64, f64) {
        let l = self.lambda(p);
        (
            (1.0 - l) * self.friction + l * self.friction_ult,
            (1.0 - l) * self.beta + l * self.beta_ult,
            (1.0 - l) * self.k + l * self.k_ult,
        )
    }

    /// The yield function at a trial state reduced by a multiplier `dl`.
    ///
    /// `q` and `I₁` follow the flow, `p` accumulates the multiplier, and the
    /// surface is re-read at the new `p` — which is what makes the consistency
    /// condition non-linear as soon as the material hardens.
    fn residual(&self, dl: f64, q_tr: f64, i1_tr: f64, p_prev: f64, mu: f64, bulk: f64) -> f64 {
        let (a, b, c) = self.surface(p_prev + dl);
        a * (i1_tr - 9.0 * bulk * self.psi * dl) + b * (q_tr - 3.0 * mu * self.delta * dl) - c
    }
}

/// Return onto the Drucker-Prager cone, with the apex case handled separately.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::plasticity;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "friction".into(), "k".into(), "psi".into()], &[210000.0, 0.3, 0.5, 100.0, 0.2]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: Vec::new() };
/// // Sensible à la pression : une traction hydrostatique plastifie, là où
/// // von Mises la laisserait passer indéfiniment. Le retour se fait sur
/// // l'**apex** du cône, à I₁ = k/α, quelle que soit l'intensité.
/// for s in [200.0, 1000.0] {
///     let pas = plasticity::drucker_prager::return_map(
///         &[s, s, s, 0.0, 0.0, 0.0], &repos, &mat)?;
///     assert!((law::i1(&pas.sigma) - 200.0).abs() < 1e-9);
///     // L'apex ne produit aucun écoulement **déviatorique** : `p`, qui
///     // cumule celui-ci, reste nul, tandis que ε_p gonfle en volume.
///     assert_eq!(pas.p, 0.0);
///     assert!(law::i1(&pas.eps_p) > 0.0);
/// }
///
/// // Hors de l'apex, `p` croît, et l'écoulement est **non associé** : la
/// // dilatance `psi` diffère du frottement, donc ε_p n'est pas isochore.
/// let pas = plasticity::drucker_prager::return_map(
///     &[400.0, 0.0, 0.0, 0.0, 0.0, 0.0], &repos, &mat)?;
/// assert!(pas.p > 0.0);
/// assert!(law::i1(&pas.eps_p) > 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn return_map(trial: &[f64; 6], prev: &PrevState, mat: &MatParams) -> Result<PlasticStep> {
    let p = params(mat)?;
    let (mu, bulk) = (mat.mu, mat.bulk());
    let q_tr = von_mises_stress(trial);
    let i1_tr = i1(trial);

    let (a0, b0, c0) = p.surface(prev.p);
    if a0 * i1_tr + b0 * q_tr - c0 <= 0.0 {
        return Ok(PlasticStep::elastic(trial, prev)); // elastic
    }

    let dlambda = solve_multiplier(&p, q_tr, i1_tr, prev.p, mu, bulk);
    let q_new = q_tr - 3.0 * mu * p.delta * dlambda;

    if q_new >= 0.0 && q_tr > 0.0 {
        let s_tr = deviator(trial);
        let mean_new = (i1_tr - 9.0 * bulk * p.psi * dlambda) / 3.0;
        let scale = q_new / q_tr;
        let mut sigma = [0.0; 6];
        let mut eps_p = prev.eps_p;
        for i in 0..6 {
            let s_new = s_tr[i] * scale;
            sigma[i] = if i < 3 { s_new + mean_new } else { s_new };
            // Δε_p = Δλ·(∂g/∂σ): deviatoric part δ·(3Δλ/2q)·s, plus the dilatant
            // volumetric part ψΔλ on the diagonal.
            eps_p[i] += 1.5 * p.delta * dlambda / q_tr * s_tr[i];
            if i < 3 {
                eps_p[i] += p.psi * dlambda;
            }
        }
        return Ok(PlasticStep {
            sigma,
            eps_p,
            p: prev.p + dlambda,
            vars: Vec::new(),
        });
    }

    // Apex return: the flank solution would push the equivalent stress negative,
    // which is meaningless. The whole deviator is shed and the stress collapses
    // onto the tip.
    apex_return(trial, prev, mat, &p)
}

/// The plastic multiplier, from the consistency condition.
///
/// Without hardening the surface is fixed and the condition is **linear** —
/// `∂g/∂σ` maps through the elastic operator to `3μδ` on the deviator and
/// `9Kψ` on the trace — so the multiplier is a quotient and no iteration
/// happens. That is the path every non-hardening model takes, unchanged.
///
/// With hardening the surface moves as the multiplier grows, and the condition
/// becomes non-linear in it. A safeguarded Newton solves it: the linear solution
/// is both the first iterate and an upper bracket, since a surface that hardens
/// (`H > 0`) can only need *less* plastic flow than a frozen one, and a
/// softening surface is caught by the bisection fallback.
fn solve_multiplier(p: &Params, q_tr: f64, i1_tr: f64, p_prev: f64, mu: f64, bulk: f64) -> f64 {
    let (a, b, c) = p.surface(p_prev);
    let slope = 3.0 * mu * b * p.delta + 9.0 * bulk * a * p.psi;
    let linear = if slope.abs() > f64::EPSILON {
        (a * i1_tr + b * q_tr - c) / slope
    } else {
        0.0
    };
    if !p.hardens() {
        return linear;
    }

    // Bracket: the residual is positive at 0 (we only got here after yielding)
    // and decreasing in `dl` for any physical set; grow the upper bound until it
    // turns, so bisection always has a sign change to work with.
    let mut hi = linear.max(1e-12);
    for _ in 0..60 {
        if p.residual(hi, q_tr, i1_tr, p_prev, mu, bulk) <= 0.0 {
            break;
        }
        hi *= 2.0;
    }
    let (mut lo, mut dl) = (0.0, hi.min(linear.max(0.0)));
    for _ in 0..50 {
        let r = p.residual(dl, q_tr, i1_tr, p_prev, mu, bulk);
        if r.abs() < 1e-12 * c.abs().max(1.0) {
            return dl;
        }
        if r > 0.0 {
            lo = dl;
        } else {
            hi = dl;
        }
        // Newton on a numerical slope, bisected whenever it leaves the bracket.
        let h = (hi - lo).max(1e-14) * 1e-6;
        let d = (p.residual(dl + h, q_tr, i1_tr, p_prev, mu, bulk) - r) / h;
        let next = if d.abs() > f64::EPSILON {
            dl - r / d
        } else {
            dl
        };
        dl = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }
    dl
}

/// Collapse onto the cone's tip: `s = 0`, `α·I₁ = k`.
///
/// With `α = 0` the cone is a cylinder and has no apex, so this cannot be
/// reached — the flank return always succeeds there (it *is* von Mises).
///
/// The surface is read at the plastic strain the shed deviator implies, so a
/// hardening material's tip sits where its hardening has taken it.
fn apex_return(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    p: &Params,
) -> Result<PlasticStep> {
    // The equivalent plastic strain shed in this step, in the same measure the
    // flank return uses (the multiplier of the deviatoric flow).
    let dp = von_mises_stress(trial) / (3.0 * mat.mu * p.delta);
    let (a, _, c) = p.surface(prev.p + dp);
    let mean_new = if a.abs() > f64::EPSILON {
        c / a / 3.0
    } else {
        i1(trial) / 3.0
    };
    let s_tr = deviator(trial);
    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    // Everything the elastic predictor built beyond the apex becomes plastic.
    let vol_drop = (i1(trial) - 3.0 * mean_new) / (9.0 * mat.bulk());
    for i in 0..6 {
        sigma[i] = if i < 3 { mean_new } else { 0.0 };
        eps_p[i] += s_tr[i] / (2.0 * mat.mu);
        if i < 3 {
            eps_p[i] += vol_drop;
        }
    }
    Ok(PlasticStep {
        sigma,
        eps_p,
        p: prev.p + dp,
        vars: Vec::new(),
    })
}

// ─── On the tangent ─────────────────────────────────────────────────────────
//
// Drucker-Prager takes its consistent tangent by finite differences, from
// [`crate::models::plasticity::law::consistent_tangent`].
//
// That is a deliberate reversal of an earlier hand derivation. Non-associated
// flow makes `∂σ/∂ε` pick up an `m⊗n` term with `m ≠ n`, and getting its
// engineering-Voigt weighting right is fiddly: the derivation that *looked*
// correct was 24 % off, and only the finite-difference oracle in
// `tests/plastic_laws.rs` said so. A numerical tangent cannot be mis-derived,
// costs twelve evaluations of the update, and leaves Newton's quadratic
// convergence intact. Hardening only strengthens the argument — the return is no
// longer even closed-form.
//
// The **apex** is the one case the numerical route still needs told about: the
// return pins the stress there, so `∂σ/∂ε` genuinely vanishes and the finite
// difference finds zero of its own accord. A body entirely at its apex therefore
// assembles a singular tangent — not an artefact, but the honest report that
// such a material carries no further load.

/// Drucker-Prager: pressure-sensitive, non-associated flow.
pub(crate) struct DruckerPrager;

impl PlasticLawKind for DruckerPrager {
    fn material_components(&self) -> &'static [&'static str] {
        &["E", "nu", "friction", "k", "psi"]
    }

    fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        _dt: Option<f64>,
    ) -> Result<PlasticStep> {
        return_map(trial, prev, mat)
    }
}

crate::physics_operator! {
    /// [`model::drucker_prager`](crate::ops::model::drucker_prager()) — pressure-sensitive plasticity
    /// with **non-associated** flow: `f = q + α·I₁ − k`, plastic potential
    /// `g = q + ψ·I₁`. Material `E`, `nu`, `alpha` (friction), `k` (cohesion),
    /// `psi` (dilatancy).
    ///
    /// `ψ = α` recovers associated flow; `ψ < α` is the usual choice for soils
    /// and rocks, whose measured dilatancy is far below what friction alone
    /// would imply. A non-associated law has a **non-symmetric** tangent.
    /// Returns beyond the cone's apex (`I₁ = k/α`) collapse onto the tip.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::elasticity::ElasticityModel;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let m = model::drucker_prager(&fes, ElasticityModel::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn drucker_prager(fes, model: ElasticityModel) = crate::ops::model::plasticity_with_law, PlasticLaw::DruckerPrager;
    python: "`model.drucker_prager(fespace, model)` — pressure-sensitive plasticity\nwith **non-associated** flow: `f = q + α·I₁ − k`, plastic potential\n`g = q + ψ·I₁`. Material `E`, `nu`, `alpha` (friction), `k` (cohesion),\n`psi` (dilatancy).\n\n`ψ = α` recovers associated flow; `ψ < α` is the usual choice for soils\nand rocks, whose measured dilatancy is far below what friction alone\nwould imply. A non-associated law has a **non-symmetric** tangent.\nReturns beyond the cone's apex (`I₁ = k/α`) collapse onto the tip."
}
