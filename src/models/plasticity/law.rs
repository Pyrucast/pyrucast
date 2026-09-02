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
//! whatever the 2-D kinematics, which keeps every return map identical across plane
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

use crate::error::{PyrucastError, Result};
use crate::models::continuum::elastic::{elastic_stress, lame};
use crate::models::continuum::material::MatRead;
use crate::models::tensor::symmetrise;
use crate::models::tensor::Kinematics;
use crate::models::TangentSource;
use serde::{Deserialize, Serialize};

/// Full 3-D tensor component suffixes, in the internal state order
/// `[xx, yy, zz, yz, xz, xy]` (off-diagonals are **tensor** strains, `ε_ij`).
///
/// ```
/// # use pyrucast::models::plasticity::law;
/// // L'ordre interne de l'état : les hors-diagonaux sont des déformations
/// // **tensorielles** ε_ij, non les doubles de l'ingénieur.
/// assert_eq!(law::TENSOR_SUFFIXES, ["xx", "yy", "zz", "yz", "xz", "xy"]);
/// ```
pub const TENSOR_SUFFIXES: [&str; 6] = ["xx", "yy", "zz", "yz", "xz", "xy"];

/// The most internal variables any law carries — the six back-stress components
/// of Chaboche, its isotropic drag `R` and Lemaitre's damage.
///
/// It bounds a **stack** buffer in the constitutive kernel, which is why it is a
/// constant rather than a `Vec`'s length: the state at a Gauss point is read
/// into an array, not allocated.
/// ```
/// # use pyrucast::models::plasticity::law::{PlasticLaw, MAX_INTERNAL_VARS};
/// // Elle borne l'état interne de toutes les lois — la plus fournie est la
/// // viscoplasticité de Chaboche endommageable.
/// assert!(PlasticLaw::ALL
///     .iter()
///     .all(|l| l.internal_names().len() <= MAX_INTERNAL_VARS));
/// ```
pub const MAX_INTERNAL_VARS: usize = 8;

/// Which yield surface and flow rule an elastoplastic kinematics obeys.
///
/// An attribute of the plasticity physics, not a physics of its own: the DOFs,
/// the elastic operator, the internal state and the incremental montage are the
/// same for all of them.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::continuum::elastic;
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
/// #     &[210_000.0, 0.3, 250.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[] };
/// // Une loi déclare **elle-même** le matériau qu'elle exige et l'état
/// // qu'elle porte : c'est ce qui permet d'en ajouter une sans toucher
/// // au reste.
/// assert!(PlasticLaw::Perfect.material_components().contains(&"sigma_y"));
/// assert!(PlasticLaw::Perfect.internal_names().is_empty());
/// assert_eq!(PlasticLaw::Gurson.internal_names(), vec!["porosity".to_string()]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// Gurson-Tvergaard-Needleman — plasticity of a porous metal, where the
    /// porosity itself shrinks the yield surface. Ductile rupture.
    Gurson,
}

/// The **return-map** family of constitutive laws: the sub-model applies the
/// elastic predictor, the law projects the trial stress back onto its surface.
///
/// The trait is named after that **integration structure**, not after a physical
/// family, because the structure is what fixes its signature — a trial stress in,
/// a [`PlasticStep`] out. That is also why
/// [`ViscoplasticLemaitreChaboche`](PlasticLaw::ViscoplasticLemaitreChaboche),
/// physically « viscoplasticity + ductile damage », lives here rather than with
/// the damage laws: it is return-map-shaped. Its siblings are
/// [`StatelessLawKind`](crate::models::elasticity::law::StatelessLawKind)
/// (`σ = f(ε)`, no state) and
/// [`DirectUpdateLawKind`](crate::models::damage::law::DirectUpdateLawKind)
/// (state, but no predictor).
///
/// Every law of this family shares the same physics — same DOFs, same elastic
/// operator, same incremental A → B mounting, same state. A law differs only by
/// its **yield surface**, its flow rule, and what those two need. This trait is
/// that difference, and nothing else.
///
/// Same division of labour as [`SubModelKind`](crate::models::SubModelKind) one
/// level up: the enum [`PlasticLaw`] carries the **physical identity** — it is
/// what an archive stores, and a closure has no name — and the trait carries the
/// **integration structure**, so that a single `match` (`PlasticLaw::as_law`)
/// relates the two and no other code dispatches per law.
///
/// Adding a law: a unit struct and its `impl` in the law's own file, plus one
/// arm in `as_law`. Nothing else in this module changes.
pub(crate) trait ReturnMapLawKind: Sync {
    /// The material components the law reads, in the order they are documented.
    fn material_components(&self) -> &'static [&'static str];

    /// Project a trial stress onto the yield surface.
    ///
    /// `dt` is always supplied. A law that ignores it is rate-independent and
    /// says so through [`is_rate_dependent`](Self::is_rate_dependent); a law
    /// that reads it is guaranteed a caller-supplied increment, because the
    /// behaviour operator refuses the whole integration otherwise — once, before
    /// any point is touched, rather than at each of them.
    fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: f64,
    ) -> Result<PlasticStep>;

    /// The law's **own** internal variables, beyond `ε_p` and `p`. Most laws
    /// need none.
    fn internal_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Material component seeding each internal variable at rest, in
    /// [`internal_names`](Self::internal_names) order. Empty — the usual case —
    /// means every internal variable starts at zero.
    ///
    /// This is how a law states that its rest state is **not** the zero state:
    /// Gurson's porosity starts at `f_0`, and a material that begins as a
    /// perfect solid never damages. The initial state is built once, before the
    /// first step, so the law never has to tell « no state yet » from « state
    /// that is zero » at a Gauss point.
    fn initial_internal_sources(&self) -> &'static [&'static str] {
        &[]
    }

    /// Whether the law needs the time increment. Erroring without one beats
    /// integrating a viscous law as if it were instantaneous.
    fn is_rate_dependent(&self) -> bool {
        false
    }

    /// The consistent tangent, **analytically**, when the law has one that has
    /// been confronted with a finite difference. `None` takes the numerical
    /// route, which cannot be mis-derived.
    ///
    /// One method where there used to be a flag, a `match` on the hardening and
    /// an `unreachable!()` — a proof obligation the compiler did not check.
    /// Only von Mises overrides it.
    /// Optional material components the law accepts beyond the required ones.
    /// `alpha` (thermal expansion) and `rho` (density) for every law; a law with
    /// a richer surface adds its own.
    fn optional_material_components(&self) -> &'static [&'static str] {
        &["alpha", "rho"]
    }

    /// Where this law's tangent comes from. Default: `Numerical` — a law says
    /// nothing, so it is differentiated. The two von Mises variants say
    /// otherwise.
    fn tangent_source(&self) -> TangentSource {
        TangentSource::Numerical
    }

    fn analytic_tangent(
        &self,
        _trial: &[f64; 6],
        _prev: &PrevState,
        _mat: &MatParams,
    ) -> Option<Result<[[f64; 6]; 6]>> {
        None
    }

    /// One incremental step A → B at a Gauss point, for **any** law.
    ///
    /// Returns `(σ(B), ε_p(B), p(B), ε(B))`. The returned strain is what should be
    /// echoed as `ε(A)` of the next step: in plane stress it carries the
    /// out-of-plane `ε_zz` solved here, which the caller could not know.
    ///
    fn incremental_step(
        &self,
        eps_b: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        kinematics: Kinematics,
        dt: f64,
    ) -> Result<(PlasticStep, [f64; 6])> {
        if kinematics == Kinematics::PlaneStress {
            return self.plane_stress_step(eps_b, prev, mat, dt);
        }
        // Solid / plane strain / axisymmetric: ε(B) is fully prescribed.
        let trial = elastic_predictor(eps_b, prev, mat.lambda, mat.mu);
        Ok((self.return_map(&trial, prev, mat, dt)?, *eps_b))
    }

    /// Plane stress, around any law: solve `σ_zz(B) = 0` for `ε_zz(B)` by the secant
    /// method, each evaluation running a full 3-D return.
    ///
    /// Written once here rather than in each law: the out-of-plane condition is a
    /// property of the **kinematics**, not of the yield surface, so no law should
    /// have to know about it.
    fn plane_stress_step(
        &self,
        eps_in_b: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: f64,
    ) -> Result<(PlasticStep, [f64; 6])> {
        let eval = |ezz: f64| -> Result<(PlasticStep, [f64; 6])> {
            let mut eps_b = *eps_in_b;
            eps_b[2] = ezz;
            eps_b[3] = 0.0;
            eps_b[4] = 0.0;
            let trial = elastic_predictor(&eps_b, prev, mat.lambda, mat.mu);
            Ok((self.return_map(&trial, prev, mat, dt)?, eps_b))
        };
        // Initial guess: ε_zz(A) plus the elastic plane-stress out-of-plane
        // increment −ν/(1−ν)·(Δε_xx + Δε_yy).
        let nu_term = mat.lambda / (mat.lambda + 2.0 * mat.mu); // = ν/(1−ν)
        let mut z0 =
            prev.eps[2] - nu_term * (eps_in_b[0] - prev.eps[0] + eps_in_b[1] - prev.eps[1]);
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

    /// The full-3-D engineering-Voigt consistent tangent `D_alg = ∂σ(B)/∂ε(B)`.
    ///
    /// Analytic for von Mises, whose closed form is validated against a finite
    /// difference; **numerical** for the others. Both are exact enough for Newton to
    /// converge quadratically, and the numerical route costs twelve evaluations of a
    /// cheap update — far less than a hand-derived tangent nobody can check, which
    /// in the Drucker-Prager case turned out to be 24 % wrong.
    ///
    fn consistent_tangent(
        &self,
        eps_b: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: f64,
    ) -> Result<[[f64; 6]; 6]> {
        let d = self.raw_consistent_tangent(eps_b, prev, mat, dt)?;
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

    /// The tangent before symmetrisation — analytic where validated, numerical
    /// otherwise.
    fn raw_consistent_tangent(
        &self,
        eps_b: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: f64,
    ) -> Result<[[f64; 6]; 6]> {
        let trial = elastic_predictor(eps_b, prev, mat.lambda, mat.mu);
        if let Some(analytic) = self.analytic_tangent(&trial, prev, mat) {
            return analytic;
        }
        self.finite_difference_tangent(eps_b, prev, mat, dt)
    }

    /// `∂σ/∂ε` by central differences on the return map, in engineering Voigt.
    ///
    /// The perturbation is applied to the **tensor** strain; the shear columns are
    /// halved on the way out, which is exactly what turns `∂σ/∂ε_ij` into
    /// `∂σ/∂γ_ij`.
    fn finite_difference_tangent(
        &self,
        eps_b: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: f64,
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
                Ok(self.return_map(&trial, prev, mat, dt)?.sigma)
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
}

impl PlasticLaw {
    /// Where this law's tangent comes from — what it costs, in one word.
    ///
    /// ```
    /// # use pyrucast::models::TangentSource;
    /// # use pyrucast::models::plasticity::law::PlasticLaw;
    /// // Von Mises a sa forme fermée ; Drucker-Prager est dérivé numériquement.
    /// assert_eq!(PlasticLaw::Isotropic.tangent_source(), TangentSource::Analytic);
    /// assert_eq!(PlasticLaw::DruckerPrager.tangent_source(), TangentSource::Numerical);
    /// ```
    pub fn tangent_source(self) -> TangentSource {
        self.as_law().tangent_source()
    }

    /// The behaviour behind this identity — **the only `match` per law**, on
    /// the kinematics of [`SubModel::as_kind`](crate::containers::model::SubModel::as_kind).
    /// The enum is what an archive stores; the trait is what the physics calls.
    pub(crate) fn as_law(self) -> &'static dyn ReturnMapLawKind {
        match self {
            Self::Perfect => &super::von_mises::Perfect,
            Self::Isotropic => &super::von_mises::Isotropic,
            Self::DruckerPrager => &super::drucker_prager::DruckerPrager,
            Self::Ottosen => &super::ottosen::Ottosen,
            Self::CreepNorton => &super::viscous::CreepNorton,
            Self::CreepBlackburn => &super::viscous::CreepBlackburn,
            Self::CreepLemaitre => &super::viscous::CreepLemaitre,
            Self::ViscoplasticChaboche => &super::viscous::ViscoplasticChaboche,
            Self::ViscoplasticLemaitreChaboche => &super::viscous::ViscoplasticLemaitreChaboche,
            Self::Gurson => &super::gurson::Gurson,
        }
    }

    /// The lowercase name (the inverse of
    /// [`from_name`](crate::named::Named::from_name)).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// # use pyrucast::named::Named;
    /// // Réciproque exacte de `from_name`, pour les dix lois.
    /// assert!(PlasticLaw::ALL.iter()
    ///     .all(|l| PlasticLaw::from_name(l.name()) == Some(*l)));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn name(self) -> &'static str {
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
            Self::Gurson => "gurson",
        }
    }

    /// Every law, in declaration order — the source of the `|`-joined tag list
    /// quoted in error messages, so a new law cannot be added without them
    /// following.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // La liste exhaustive, dont se servent la surface Python et les messages.
    /// assert_eq!(PlasticLaw::ALL.len(), 10);
    /// assert!(PlasticLaw::ALL.contains(&PlasticLaw::Perfect));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub const ALL: [PlasticLaw; 10] = [
        Self::Perfect,
        Self::Isotropic,
        Self::DruckerPrager,
        Self::Ottosen,
        Self::CreepNorton,
        Self::CreepBlackburn,
        Self::CreepLemaitre,
        Self::ViscoplasticChaboche,
        Self::ViscoplasticLemaitreChaboche,
        Self::Gurson,
    ];

    /// The material components this law requires. Elasticity (`E`, `nu`) is
    /// common to all; what follows describes the surface.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Perfect : un seuil constant. Isotropic : plus le module d'écrouissage.
    /// assert!(PlasticLaw::Perfect.material_components().contains(&"sigma_y"));
    /// assert!(PlasticLaw::Isotropic.material_components().contains(&"H"));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn material_components(self) -> &'static [&'static str] {
        self.as_law().material_components()
    }

    /// The law's **own** internal variables, beyond `ε_p` and `p`. Empty for a
    /// law that needs nothing more; a back stress or a damage otherwise.
    ///
    /// These become extra components of the behaviour output, so a law can grow
    /// its state without any other file changing.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Ce que la loi porte **au-delà** de ε_p et p — rien pour la plupart.
    /// assert!(PlasticLaw::Perfect.internal_names().is_empty());
    /// // La porosité **est** l'état d'une loi de métal poreux.
    /// assert_eq!(PlasticLaw::Gurson.internal_names(), vec!["porosity".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn internal_names(self) -> Vec<String> {
        self.as_law().internal_names()
    }

    /// Whether this law is **rate-dependent** — it needs the time increment, and
    /// erroring without one is better than silently integrating a viscous law as
    /// if it were instantaneous.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Une loi visqueuse exige l'incrément de temps ; intégrer sans lui,
    /// // comme si la loi était instantanée, serait faux en silence.
    /// assert!(!PlasticLaw::Perfect.is_viscous());
    /// assert!(PlasticLaw::CreepNorton.is_viscous());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn is_viscous(self) -> bool {
        self.as_law().is_rate_dependent()
    }

    /// Project a trial stress onto this law's yield surface.
    ///
    /// `dt` is the time increment: `None` for a rate-independent law, and
    /// **required** by a viscous one.
    ///
    /// ```
    /// # use pyrucast::models::tensor;
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Sous le seuil, la contrainte d'essai passe telle quelle.
    /// let sous = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let pas = PlasticLaw::Perfect.return_map(&sous, &repos, &mat, None)?;
    /// assert_eq!(pas.sigma, sous);
    /// assert_eq!(pas.p, 0.0);
    ///
    /// // Au-dessus, elle est **projetée** sur la surface de charge : von Mises
    /// // sans écrouissage ramène exactement q à σ_y, et p devient non nul.
    /// let au_dessus = [400.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let pas = PlasticLaw::Perfect.return_map(&au_dessus, &repos, &mat, None)?;
    /// assert!((tensor::von_mises_stress(&pas.sigma) - 250.0).abs() < 1e-6);
    /// assert!(pas.p > 0.0);
    ///
    /// // Une loi visqueuse sans `dt` est refusée plutôt qu'approximée.
    /// assert!(PlasticLaw::CreepNorton.return_map(&sous, &repos, &mat, None).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        dt: Option<f64>,
    ) -> Result<PlasticStep> {
        // A public entry point, and therefore the place that still asks: a
        // caller reaching a law directly may have no increment to give. Below
        // this line `dt` is a plain `f64` — the kernels never ask again.
        if self.is_viscous() && dt.is_none() {
            return Err(PyrucastError::Message(format!(
                "plasticity ({self}): this law is rate-dependent and needs a time increment — \
                 pass `dt` to integrate_behavior"
            )));
        }
        self.as_law()
            .return_map(trial, prev, mat, dt.unwrap_or(0.0))
    }
}

impl crate::named::Named for PlasticLaw {
    const LABEL: &'static str = "plastic law";
    const VALUES: &'static [Self] = &Self::ALL;

    fn name(self) -> &'static str {
        PlasticLaw::name(self)
    }
}

impl std::fmt::Display for PlasticLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Material parameters at one Gauss point ─────────────────────────────────

/// The material a law reads, resolved for one cell.
///
/// The elastic constants are pre-computed (every law needs them); the rest is
/// looked up by name, so adding a law adds no plumbing here.
///
/// ```
/// # use pyrucast::models::continuum::elastic;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::plasticity::law::{self, MatParams};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0)?, vec!["E".into(), "nu".into(), "sigma_y".into()],
/// #     &[210_000.0, 0.3, 250.0])?;
/// // Les constantes élastiques sont **pré-calculées** — toute loi en a
/// // besoin — le reste se cherche par nom, de sorte qu'ajouter une loi
/// // n'ajoute aucune plomberie ici.
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// let m = MatParams::new(materiau.point_values(0, 0)?, &idx_mat, &opt_mat);
/// assert_eq!((m.lambda, m.mu), elastic::lame(210_000.0, 0.3));
/// assert_eq!(m.get(2), 250.0); // σ_y, troisième du contrat
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MatParams<'a> {
    /// Lamé's first coefficient.
    pub lambda: f64,
    /// Shear modulus.
    pub mu: f64,
    /// The material row and its position tables — the reader the three law
    /// families share. `MatParams` is that reader plus the two constants every
    /// plastic contract opens with.
    mat: MatRead<'a>,
}

impl<'a> MatParams<'a> {
    /// The material of one cell: its row, plus where each component of the law's
    /// own contract sits in it (resolved once for the zone).
    ///
    /// Every plastic contract opens with `E`, `nu` — the two constants no law
    /// can do without — so the Lamé coefficients are pre-computed here and the
    /// rest is read by [`get`](Self::get), positionally.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::continuum::elastic;
    /// # use pyrucast::models::plasticity::law::{MatParams, PlasticLaw};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0)?, vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0])?;
    /// // La résolution a lieu **une fois par zone** ; le noyau, lui, indexe.
    /// let idx = materiau.resolve_components(
    ///     PlasticLaw::Perfect.material_components(), "material")?;
    /// let m = MatParams::new(materiau.point_values(0, 0)?, &idx, &[]);
    /// assert_eq!((m.lambda, m.mu), elastic::lame(210_000.0, 0.3));
    /// assert_eq!(m.get(2), 250.0); // σ_y, troisième du contrat
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn new(row: &'a [f64], idx: &'a [u32], opt_idx: &'a [u32]) -> Self {
        let (lambda, mu) = lame(row[idx[0] as usize], row[idx[1] as usize]);
        Self {
            lambda,
            mu,
            mat: MatRead::new(row, idx, opt_idx),
        }
    }

    /// The `k`-th component of this law's own material contract, for this cell. No name, no search, no `Result`: the component's presence was
    /// settled when the layout was resolved.
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{MatParams, PlasticLaw};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// let materiau = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["E".into(), "nu".into(), "sigma_y".into()],
    ///     &[210_000.0, 0.3, 250.0])?;
    /// let idx = materiau.resolve_components(
    ///     PlasticLaw::Perfect.material_components(), "material")?;
    /// let m = MatParams::new(materiau.point_values(0, 0)?, &idx, &[]);
    /// // Le contrat de la loi est `["E", "nu", "sigma_y"]`.
    /// assert_eq!(m.get(2), 250.0);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn get(&self, k: usize) -> f64 {
        self.mat.get(k)
    }

    /// The `k`-th component of this law's
    /// [`optional_material_components`](crate::models::Domain::optional_material_components),
    /// or `default` where the caller supplied none.
    ///
    /// Absence is a fact of the **zone**, settled when the layout was resolved:
    /// the branch here is on a resolved index, never on a name.
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{MatParams, PlasticLaw};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
    /// # use pyrucast::containers::field::ABSENT_COMPONENT;
    /// let materiau = SubElementField::from_uniform_per_component(
    ///     fes.get(0)?, vec!["E".into(), "nu".into(), "sigma_y".into(), "H".into()],
    ///     &[210_000.0, 0.3, 250.0, 1_000.0])?;
    /// let idx = materiau.resolve_components(
    ///     PlasticLaw::Perfect.material_components(), "material")?;
    /// // Deux facultatives : `H`, fournie, et `rho`, absente.
    /// let opt = materiau.resolve_optional_components(&["H", "rho"]);
    /// let m = MatParams::new(materiau.point_values(0, 0)?, &idx, &opt);
    /// assert_eq!(m.optional(0, 0.0), 1_000.0);
    /// assert_eq!(m.optional(1, 7_800.0), 7_800.0); // le défaut
    /// assert_eq!(opt[1], ABSENT_COMPONENT);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn optional(&self, k: usize, default: f64) -> f64 {
        self.mat.optional(k, default)
    }

    /// Bulk modulus `K = λ + 2μ/3`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // K = λ + 2μ/3 — ce dont a besoin toute loi sensible à la pression.
    /// assert!((mat.bulk() - (mat.lambda + 2.0 * mat.mu / 3.0)).abs() < 1e-9);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn bulk(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }
}

// ─── Elastic kinematics, shared by every law ────────────────────────────────

/// The converged state at the **start of the step A** — the input to the
/// incremental montage. All full 3-D.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
/// #     &[210_000.0, 0.3, 250.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[] };
/// // L'état au début du pas A : c'est de là que part chaque intégration.
/// assert_eq!(repos.p, 0.0);
/// assert_eq!(repos.var(0), 0.0); // aucune variable interne portée
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Clone, Default)]
pub struct PrevState<'a> {
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
    ///
    /// **Borrowed**: the kernel reads them into a stack buffer bounded by
    /// [`MAX_INTERNAL_VARS`] and lends it here, so a Gauss point allocates
    /// nothing.
    pub vars: &'a [f64],
}

impl PrevState<'_> {
    /// Internal variable `i`, or `0` when the state does not carry it (the first
    /// step, where A is the reference configuration).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Rend `0` plutôt que d'échouer quand l'état ne porte pas la variable —
    /// // ce qui est le cas au premier pas, où A est la configuration de départ.
    /// assert_eq!(repos.var(0), 0.0);
    /// let avec = PrevState { vars: &[0.02], ..repos };
    /// assert_eq!(avec.var(0), 0.02);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn var(&self, i: usize) -> f64 {
        self.vars.get(i).copied().unwrap_or(0.0)
    }
}

/// The updated state at the **end of the step B**, as a law returns it.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
/// #     &[210_000.0, 0.3, 250.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[] };
/// // Ce qu'une loi rend en fin de pas B : contrainte, déformation
/// // plastique, plasticité cumulée, et ses variables internes.
/// let trial = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let pas = PlasticStep::elastic(&trial, &repos);
/// assert_eq!(pas.sigma, trial);
/// assert_eq!(pas.p, 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct PlasticStep {
    /// Stress `σ(B)`, full 3-D.
    pub sigma: [f64; 6],
    /// Plastic strain `ε_p(B)`, full 3-D.
    pub eps_p: [f64; 6],
    /// Cumulated plastic strain `p(B)`.
    pub p: f64,
    /// The law's own internal variables at B, in [`PlasticLaw::internal_names`]
    /// order.
    pub vars: [f64; MAX_INTERNAL_VARS],
    /// How many of `vars` this law actually carries — the rest is padding.
    pub n_vars: usize,
}

impl PlasticStep {
    /// An **elastic** step: the trial stress stands, nothing evolves.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::element_field::SubElementField;
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
    /// # let materiau = SubElementField::from_uniform_per_component(
    /// #     fes.get(0).unwrap(),
    /// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
    /// #     &[210_000.0, 0.3, 250.0]).unwrap();
    /// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
    /// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
    /// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
    /// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
    /// #                         vars: &[] };
    /// // Un pas **élastique** : la contrainte d'essai tient, rien n'évolue.
    /// let trial = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    /// let pas = PlasticStep::elastic(&trial, &repos);
    /// assert_eq!((pas.sigma, pas.eps_p, pas.p), (trial, repos.eps_p, repos.p));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn elastic(trial: &[f64; 6], prev: &PrevState) -> Self {
        Self::new(*trial, prev.eps_p, prev.p, prev.vars)
    }

    /// A step carrying `vars` as its internal state.
    ///
    /// The array is **fixed-size and lives on the stack**: a return mapping runs
    /// at every Gauss point of every cell, so a `Vec` here was one allocation
    /// per point for a handful of numbers. [`PrevState`] had always read them
    /// through a slice; this closes the asymmetry.
    ///
    /// ```
    /// # use pyrucast::models::plasticity::law::PlasticStep;
    /// // Deux variables internes, et rien d'alloué pour les porter.
    /// let pas = PlasticStep::new([0.0; 6], [0.0; 6], 0.0, &[0.3, 1.5]);
    /// assert_eq!(pas.internal(), &[0.3, 1.5]);
    /// ```
    pub fn new(sigma: [f64; 6], eps_p: [f64; 6], p: f64, vars: &[f64]) -> Self {
        let mut v = [0.0_f64; MAX_INTERNAL_VARS];
        v[..vars.len()].copy_from_slice(vars);
        Self {
            sigma,
            eps_p,
            p,
            vars: v,
            n_vars: vars.len(),
        }
    }

    /// The internal variables this step carries — the live prefix of `vars`.
    ///
    /// ```
    /// # use pyrucast::models::plasticity::law::PlasticStep;
    /// // Une loi sans variable interne en rend une tranche vide, non un `Vec`.
    /// let pas = PlasticStep::new([0.0; 6], [0.0; 6], 0.0, &[]);
    /// assert!(pas.internal().is_empty());
    /// ```
    pub fn internal(&self) -> &[f64] {
        &self.vars[..self.n_vars]
    }
}

/// Elastic predictor of the **incremental** montage: `σ_trial = σ(A) + C:Δε`
/// with `Δε = ε(B) − ε(A)`.
///
/// Algebraically identical to `C:(ε(B) − ε_p(A))` in small strain, but this is
/// the form that carries `σ(A)` explicitly — the shape a large-strain law reuses
/// (with `σ(A)` rotated and `Δε` an objective increment).
///
/// ```
/// # use pyrucast::models::continuum::elastic;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::plasticity::law::{self, MatParams, PlasticLaw, PlasticStep, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(),
/// #     vec!["E".into(), "nu".into(), "sigma_y".into()],
/// #     &[210_000.0, 0.3, 250.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[] };
/// // σ_trial = σ(A) + C:Δε — la forme qui porte σ(A) explicitement, celle
/// // qu'une loi en grandes déformations reprend telle quelle.
/// let eps_b = [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let trial = law::elastic_predictor(&eps_b, &repos, mat.lambda, mat.mu);
/// assert!(trial[0] > 0.0);
/// // Partant du repos, elle coïncide avec C:ε.
/// let direct = elastic::elastic_stress(&eps_b, mat.lambda, mat.mu);
/// assert!((trial[0] - direct[0]).abs() < 1e-9);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn elastic_predictor(eps_b: &[f64; 6], prev: &PrevState, lambda: f64, mu: f64) -> [f64; 6] {
    let deps: [f64; 6] = std::array::from_fn(|i| eps_b[i] - prev.eps[i]);
    let c_deps = elastic_stress(&deps, lambda, mu);
    std::array::from_fn(|i| prev.sigma[i] + c_deps[i])
}

// ─── The incremental step, around any law ───────────────────────────────────

// ─── The consistent tangent ─────────────────────────────────────────────────

/// Guard a law's material against a value that would make its return map
/// meaningless, with a message naming the law and the constant.
///
/// ```
/// # use pyrucast::models::plasticity::law;
/// # use pyrucast::models::plasticity::law::PlasticLaw;
/// // Un garde-fou qui nomme la loi **et** la constante fautive ; il rend la
/// // valeur, pour s'enchaîner à la lecture du matériau.
/// let l = PlasticLaw::Perfect;
/// assert_eq!(law::require_positive(l, "sigma_y", 250.0)?, 250.0);
/// assert!(law::require_positive(l, "sigma_y", 0.0).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn require_positive(law: PlasticLaw, name: &str, value: f64) -> Result<f64> {
    if value <= 0.0 {
        return Err(PyrucastError::Message(format!(
            "plasticity ({law}): {name} = {value} must be positive"
        )));
    }
    Ok(value)
}
