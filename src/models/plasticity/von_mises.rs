//! von Mises (J2) plasticity — perfect, or with linear isotropic hardening.
//!
//! The archetypal metal-plasticity law: yielding depends on the **deviator**
//! alone, so it is insensitive to hydrostatic pressure and plastic flow is
//! isochoric. The yield surface is a cylinder about the hydrostatic axis,
//!
//! ```text
//! f(σ, p) = q − σ_y(p)          q = √(3 J₂)
//! ```
//!
//! with `σ_y(p) = σ_y + H·p`. `H = 0` is the perfect law — one code path serves
//! both, which is why they share this file rather than duplicating a return map
//! that differs by a single term.
//!
//! ## The closed-form return
//!
//! Associated flow on a cylinder means the return is **radial**: the deviator
//! is scaled, its direction untouched, and the hydrostatic part is left alone.
//! Consistency gives the multiplier in one step, with no iteration:
//!
//! ```text
//! Δp = (q_trial − σ_y(p_A)) / (3μ + H)
//! ```
//!
//! That closed form is why von Mises does not go through the cutting plane: an
//! exact answer beats a converged one.

use super::law::PlasticLawKind;
use crate::error::Result;
use crate::models::elasticity::elastic_tangent;
use crate::models::plasticity::law::PlasticLaw;
use crate::models::plasticity::law::{MatParams, PlasticStep, PrevState};
use crate::models::tensor::Kinematics;
use crate::models::tensor::{deviator, von_mises_stress};

/// Radial return onto `q = σ_y + H·p`.
///
/// `hardening` is `H`; pass `0.0` for the perfect law. Returns the updated
/// `(σ, ε_p, p)`, all full 3-D.
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
/// # use pyrucast::models::elasticity;
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into()], &[210000.0, 0.3, 250.0]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: Vec::new() };
/// // Sans écrouissage (`hardening = 0`), la projection ramène **exactement**
/// // q sur σ_y, et la déformation plastique est purement déviatorique.
/// let trial = [400.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let pas = plasticity::von_mises::return_map(&trial, &repos, &mat, 0.0)?;
/// assert!((tensor::von_mises_stress(&pas.sigma) - 250.0).abs() < 1e-6);
/// assert!(tensor::i1(&pas.eps_p).abs() < 1e-12); // écoulement isochore
///
/// // Avec écrouissage isotrope, le seuil monte de H·p : la contrainte
/// // retenue est **plus grande**.
/// let dur = plasticity::von_mises::return_map(&trial, &repos, &mat, 20_000.0)?;
/// assert!(tensor::von_mises_stress(&dur.sigma) > 250.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn return_map(
    trial: &[f64; 6],
    prev: &PrevState,
    mat: &MatParams,
    hardening: f64,
) -> Result<PlasticStep> {
    let sigma_y0 = mat.get("sigma_y")?;
    let q = von_mises_stress(trial);
    let yield_now = sigma_y0 + hardening * prev.p;
    let f = q - yield_now;
    if f <= 0.0 || q == 0.0 {
        return Ok(PlasticStep::elastic(trial, prev)); // elastic
    }
    // Consistency: q − 3μΔp = σ_y + H(p + Δp).
    let dp = f / (3.0 * mat.mu + hardening);
    let s_trial = deviator(trial);
    let mean = (trial[0] + trial[1] + trial[2]) / 3.0;
    let scale = (yield_now + hardening * dp) / q;
    // Flow direction n = (3/2)·s/q, so Δε_p = Δp·n = (3Δp/2q)·s — a tensor, so
    // the off-diagonals take the same factor with no engineering doubling.
    let factor = 1.5 * dp / q;

    let mut sigma = [0.0; 6];
    let mut eps_p = prev.eps_p;
    for i in 0..6 {
        let s_new = s_trial[i] * scale;
        sigma[i] = if i < 3 { s_new + mean } else { s_new };
        eps_p[i] += factor * s_trial[i];
    }
    Ok(PlasticStep {
        sigma,
        eps_p,
        p: prev.p + dp,
        vars: Vec::new(),
    })
}

/// The consistent tangent `D_alg = ∂σ(B)/∂ε(B)` of the radial return, evaluated
/// at the trial stress — the classical algorithmic modulus,
///
/// ```text
/// D = K·1⊗1 + 2μθ·I_dev − 2μθ̄·n̂⊗n̂
/// θ = σ_y(p+Δp)/q_trial          θ̄ = 3μ/(3μ + H) − (1 − θ)
/// ```
/// # use pyrucast::models::elasticity;
///
/// with `n̂ = s_trial/‖s_trial‖` the **unit** deviatoric direction. For `H = 0`
/// the two coefficients collapse (`θ̄ = θ`) and this is exactly the perfect-J2
/// tangent — which is why hardening costs one extra term rather than a second
/// derivation.
///
/// Note the difference from the **continuum** elastoplastic modulus: `θ` here
/// accounts for the *finite* step, and dropping it would cost Newton its
/// quadratic convergence.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::elasticity;
/// # use pyrucast::models::plasticity;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into()], &[210000.0, 0.3, 250.0]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: Vec::new() };
/// // Sous le seuil, la tangente cohérente **est** la tangente élastique.
/// let sous = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let d = plasticity::von_mises::tangent(&sous, &mat, 0.0, 0.0);
/// assert_eq!(d, elasticity::elastic_tangent(mat.lambda, mat.mu));
///
/// // Au-delà, elle s'assouplit : le module apparent chute.
/// let au_dela = [400.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let dp = plasticity::von_mises::tangent(&au_dela, &mat, 0.0, 0.0);
/// assert!(dp[0][0] < d[0][0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn tangent(trial: &[f64; 6], mat: &MatParams, hardening: f64, p_prev: f64) -> [[f64; 6]; 6] {
    let (lambda, mu) = (mat.lambda, mat.mu);
    let sigma_y0 = mat.get("sigma_y").unwrap_or(0.0);
    let q = von_mises_stress(trial);
    let yield_now = sigma_y0 + hardening * p_prev;
    if q <= yield_now || q == 0.0 {
        return elastic_tangent(lambda, mu);
    }
    let dp = (q - yield_now) / (3.0 * mu + hardening);
    let theta = (yield_now + hardening * dp) / q;
    let theta_bar = 3.0 * mu / (3.0 * mu + hardening) - (1.0 - theta);

    let k = lambda + 2.0 * mu / 3.0;
    let coef = 2.0 * mu * theta;
    let mut d = [[0.0_f64; 6]; 6];
    // K·1⊗1 on the normal (top-left 3×3) block.
    for row in d.iter_mut().take(3) {
        for e in row.iter_mut().take(3) {
            *e += k;
        }
    }
    // 2μθ · I_dev (engineering: normal block ⅔/−⅓, shear diagonal ½).
    for (i, row) in d.iter_mut().enumerate().take(3) {
        for (j, e) in row.iter_mut().enumerate().take(3) {
            *e += coef * if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 };
        }
    }
    for i in 3..6 {
        d[i][i] += coef * 0.5;
    }
    // − 2μθ̄ · n̂⊗n̂, with `n̂` unit in the Frobenius sense (off-diagonals of the
    // deviator counted twice, as `s:s` does).
    let s = deviator(trial);
    let s_norm =
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
            .sqrt();
    if s_norm > 0.0 {
        let nv: [f64; 6] = std::array::from_fn(|i| s[i] / s_norm);
        let c = 2.0 * mu * theta_bar;
        for i in 0..6 {
            for j in 0..6 {
                d[i][j] -= c * nv[i] * nv[j];
            }
        }
    }
    d
}

/// The perfect von Mises law — no hardening.
pub(crate) struct Perfect;

impl PlasticLawKind for Perfect {
    fn name(&self) -> &'static str {
        "perfect"
    }

    fn material_components(&self) -> &'static [&'static str] {
        &["E", "nu", "sigma_y"]
    }

    fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        _dt: Option<f64>,
    ) -> Result<PlasticStep> {
        return_map(trial, prev, mat, 0.0)
    }

    fn analytic_tangent(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
    ) -> Option<Result<[[f64; 6]; 6]>> {
        Some(Ok(tangent(trial, mat, 0.0, prev.p)))
    }
}

/// von Mises with **linear isotropic hardening**, `σ_y(p) = σ_y + H·p`.
pub(crate) struct Isotropic;

impl PlasticLawKind for Isotropic {
    fn name(&self) -> &'static str {
        "isotropic"
    }

    fn material_components(&self) -> &'static [&'static str] {
        &["E", "nu", "sigma_y", "H"]
    }

    fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        _dt: Option<f64>,
    ) -> Result<PlasticStep> {
        return_map(trial, prev, mat, mat.get("H")?)
    }

    fn analytic_tangent(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
    ) -> Option<Result<[[f64; 6]; 6]>> {
        Some(mat.get("H").map(|h| tangent(trial, mat, h, prev.p)))
    }
}

crate::physics_operator! {
    /// [`model::plasticity_perfect`](crate::ops::model::plasticity_perfect()) — **perfect** (non-hardening)
    /// von Mises elastoplasticity spanning every subspace of `fespace`. `kinematics`
    /// is `"plane_stress"` / `"plane_strain"` / `"axisymmetric"` (2-D) or
    /// `"full_3d"` (3-D). Same DOFs as elasticity (`u_x, u_y(, u_z)`); material
    /// (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time. The
    /// behaviour integration (`COMP`) carries the plastic-strain +
    /// cumulated-`p` internal state (`VAR0`→`VAR1`) and emits the consistent
    /// tangent `D_alg`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let m = model::plasticity_perfect(&fes, Kinematics::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn plasticity_perfect(fes, kinematics: Kinematics) = crate::ops::model::plasticity_with_law, PlasticLaw::Perfect;
    python: "`kinematics.plasticity_perfect(fespace, kinematics)` — **perfect** (non-hardening)\nvon Mises elastoplasticity spanning every subspace of `fespace`. `kinematics`\nis `\"plane_stress\"` / `\"plane_strain\"` / `\"axisymmetric\"` (2-D) or\n`\"solid\"` (3-D). Same DOFs as elasticity (`u_x, u_y(, u_z)`); material\n(`E`, `nu`, `sigma_y`) is supplied at assembly / integration time. The\nbehaviour integration (`COMP`) carries the plastic-strain +\ncumulated-`p` internal state (`VAR0`→`VAR1`) and emits the consistent\ntangent `D_alg`."
}

crate::physics_operator! {
    /// [`model::plasticity_isotropic`](crate::ops::model::plasticity_isotropic()) — von Mises with **linear
    /// isotropic hardening**, `σ_y(p) = σ_y + H·p`. Material `E`, `nu`,
    /// `sigma_y`, `H`; everything else as `plasticity_perfect` (`H = 0` would
    /// give it back exactly).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::ops::model;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// let m = model::plasticity_isotropic(&fes, Kinematics::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn plasticity_isotropic(fes, kinematics: Kinematics) = crate::ops::model::plasticity_with_law, PlasticLaw::Isotropic;
    python: "`kinematics.plasticity_isotropic(fespace, kinematics)` — von Mises with **linear\nisotropic hardening**, `σ_y(p) = σ_y + H·p`. Material `E`, `nu`,\n`sigma_y`, `H`; everything else as `plasticity_perfect` (`H = 0` would\ngive it back exactly)."
}
