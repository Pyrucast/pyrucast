//! Gurson-Tvergaard-Needleman — plasticity of a **porous** metal.
//!
//! A ductile metal does not fail by reaching a stress: it fails because voids
//! nucleate, grow and coalesce until the ligaments between them cannot carry the
//! load. Gurson's surface makes the porosity `f` an explicit internal variable
//! and lets it shrink the yield surface:
//!
//! ```text
//! Φ = (q/σ_y)² + 2q₁ f* cosh(3q₂ σ_m / 2σ_y) − (1 + q₃ f*²)
//! ```
//!
//! At `f = 0` this collapses to `q = σ_y` — von Mises exactly. As `f` grows the
//! surface contracts, and the `cosh` makes that contraction depend on the
//! **hydrostatic** stress: voids grow under triaxial tension and close under
//! compression. That pressure sensitivity is the whole point, and it is why a
//! J2 law can never predict ductile rupture.
//!
//! ## Coalescence: `f*` rather than `f`
//!
//! Voids do not weaken a material gradually all the way to failure — beyond a
//! critical porosity they **coalesce**, and the collapse accelerates sharply.
//! Tvergaard and Needleman kinematics that by feeding the surface an *effective*
//! porosity:
//!
//! ```text
//! f* = f                                        if f ≤ f_c
//! f* = f_c + (1/q₁ − f_c)·(f − f_c)/(f_f − f_c)  otherwise
//! ```
//!
//! which reaches `1/q₁` — where the surface has shrunk to nothing — at the
//! failure porosity `f_f`. Without this the kinematics predicts a far too ductile
//! material.
//!
//! ## Growth
//!
//! ```text
//! ḟ = (1 − f)·tr(ε̇_p)
//! ```
//!
//! Mass conservation, no more: the voids grow with the **volumetric** plastic
//! strain. Plastic flow on this surface is *not* isochoric — that is precisely
//! what distinguishes it from von Mises — so `tr(ε̇_p)` is non-zero, and the
//! `cosh` makes it grow with triaxiality. Nucleation (a strain- or
//! stress-driven source term) is **not** modelled: only growth from an initial
//! porosity `f_0`.
//!
//! ## Integration
//!
//! The surface is scalar and closed-form, but its return has no closed form, so
//! it goes through the same **cutting plane with a numerically differentiated
//! normal** as [`ottosen`](crate::models::plasticity::ottosen). The normal here has
//! both a deviatoric and a volumetric part, which is exactly what makes the
//! plastic flow dilatant.

use super::law::PlasticLawKind;
use crate::error::{PyrucastError, Result};
use crate::models::elasticity::elastic_stress;
use crate::models::plasticity::law::{
    require_positive, MatParams, PlasticLaw, PlasticStep, PrevState,
};
use crate::models::tensor::Kinematics;
use crate::models::tensor::{i1, von_mises_stress};

/// Positions in this law's material contract,
/// `["E", "nu", "sigma_y", "q_1", "q_2", "q_3", "f_0", "f_c", "f_f"]`.
const SIGMA_Y: usize = 2;
const Q_1: usize = 3;
const Q_2: usize = 4;
const Q_3: usize = 5;
const F_C: usize = 7;
const F_F: usize = 8;

/// The **effective** porosity of Tvergaard and Needleman — `f` below the
/// coalescence threshold, accelerating to `1/q₁` at failure above it.
fn effective_porosity(f: f64, q1: f64, f_c: f64, f_f: f64) -> f64 {
    if f <= f_c {
        return f;
    }
    let f_u = 1.0 / q1;
    let span = (f_f - f_c).max(1e-12);
    (f_c + (f_u - f_c) * (f - f_c) / span).min(f_u)
}

/// Gurson's yield function. Negative inside the elastic domain, and exactly
/// `(q/σ_y)² − 1` when the porosity vanishes.
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
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into(), "q_1".into(), "q_2".into(), "q_3".into(), "f_0".into(), "f_c".into(), "f_f".into()], &[210000.0, 0.3, 250.0, 1.5, 1.0, 2.25, 0.001, 0.15, 0.25]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[0.001] };
/// // La porosité **rétrécit** la surface de charge : à contrainte égale, un
/// // métal plus poreux est plus près de céder.
/// let s = [200.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let sain = plasticity::gurson::yield_function(&s, 0.001, &mat)?;
/// let poreux = plasticity::gurson::yield_function(&s, 0.05, &mat)?;
/// assert!(poreux > sain);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn yield_function(sigma: &[f64; 6], f: f64, mat: &MatParams) -> Result<f64> {
    let sigma_y = require_positive(PlasticLaw::Gurson, "sigma_y", mat.get(SIGMA_Y))?;
    let (q1, q2, q3) = (mat.get(Q_1), mat.get(Q_2), mat.get(Q_3));
    let (f_c, f_f) = (mat.get(F_C), mat.get(F_F));
    let f_star = effective_porosity(f, q1, f_c, f_f);

    let q = von_mises_stress(sigma);
    let sigma_m = i1(sigma) / 3.0;
    // The argument is clamped: `cosh` overflows well before the stress state it
    // would represent is physical, and an infinity here would poison the whole
    // return rather than merely bounding it.
    let arg = (1.5 * q2 * sigma_m / sigma_y).clamp(-500.0, 500.0);
    Ok((q / sigma_y).powi(2) + 2.0 * q1 * f_star * arg.cosh() - (1.0 + q3 * f_star * f_star))
}

/// `∂Φ/∂σ` by central differences, in the **tensor** sense.
fn numerical_normal(sigma: &[f64; 6], f: f64, mat: &MatParams, scale: f64) -> Result<[f64; 6]> {
    let h = 1e-6 * scale;
    let mut n = [0.0; 6];
    for (i, ni) in n.iter_mut().enumerate() {
        let mut plus = *sigma;
        let mut minus = *sigma;
        plus[i] += h;
        minus[i] -= h;
        *ni = (yield_function(&plus, f, mat)? - yield_function(&minus, f, mat)?) / (2.0 * h);
    }
    Ok(n)
}

/// Cutting-plane return onto Gurson's surface, with the porosity updated from
/// the volumetric plastic flow it produces.
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
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into(), "q_1".into(), "q_2".into(), "q_3".into(), "f_0".into(), "f_c".into(), "f_f".into()], &[210000.0, 0.3, 250.0, 1.5, 1.0, 2.25, 0.001, 0.15, 0.25]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatParams::new(materiau.point_values(0, 0).unwrap(), &idx_mat, &opt_mat);
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: &[0.001] };
/// // La porosité **est** l'état : elle croît avec l'écoulement, et c'est
/// // elle qui mène à la rupture ductile.
/// let trial = [800.0, 400.0, 400.0, 0.0, 0.0, 0.0];
/// let pas = plasticity::gurson::return_map(&trial, &repos, &mat)?;
/// assert!(pas.p > 0.0);
/// assert_eq!(pas.vars.len(), 1);
/// assert!(pas.vars[0] >= repos.vars[0]); // la porosité ne décroît pas
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn return_map(trial: &[f64; 6], prev: &PrevState, mat: &MatParams) -> Result<PlasticStep> {
    let sigma_y = mat.get(SIGMA_Y);
    // The porosity at A — the law's own variable, always carried: the state at
    // rest is seeded with `f_0` by `initial_internal_sources`, so there is no
    // « first step » to recognise here. A material starting as a perfect solid
    // would have no void to grow and would never damage.
    let f_a = prev.var(0);

    if yield_function(trial, f_a, mat)? <= 0.0 {
        let mut step = PlasticStep::elastic(trial, prev);
        step.vars = vec![f_a];
        return Ok(step);
    }

    let mut sigma = *trial;
    let mut eps_p = prev.eps_p;
    let mut p = prev.p;
    let mut f = f_a;

    for _ in 0..100 {
        let phi = yield_function(&sigma, f, mat)?;
        if phi.abs() <= 1e-12 {
            break;
        }
        let n = numerical_normal(&sigma, f, mat, sigma_y)?;
        let cn = elastic_stress(&n, mat.lambda, mat.mu);
        let ncn: f64 = (0..6)
            .map(|i| n[i] * cn[i] * if i < 3 { 1.0 } else { 2.0 })
            .sum();
        if ncn.abs() < f64::MIN_POSITIVE {
            return Err(PyrucastError::Message(
                "plasticity (gurson): the yield normal is degenerate — the porosity may have \
                 reached f_f, where the surface has collapsed"
                    .into(),
            ));
        }
        let dlambda = phi / ncn;
        for i in 0..6 {
            sigma[i] -= dlambda * cn[i];
            eps_p[i] += dlambda * n[i];
        }
        // Void growth: mass conservation on the volumetric plastic increment.
        // The normal's trace is non-zero — the `cosh` sees to that — which is
        // why a porous law dilates where von Mises cannot.
        let dev_vol = dlambda * (n[0] + n[1] + n[2]);
        f = (f + (1.0 - f) * dev_vol).clamp(0.0, mat.get(F_F));
        p += dlambda.abs();
    }

    if yield_function(&sigma, f, mat)?.abs() > 1e-6 {
        return Err(PyrucastError::Message(
            "plasticity (gurson): the return mapping did not converge in 100 iterations — the \
             strain increment may be too large, or the porosity too close to f_f"
                .into(),
        ));
    }
    Ok(PlasticStep {
        sigma,
        eps_p,
        p,
        vars: vec![f],
    })
}

/// Gurson-Tvergaard-Needleman porous plasticity.
pub(crate) struct Gurson;

impl PlasticLawKind for Gurson {
    fn material_components(&self) -> &'static [&'static str] {
        &[
            "E", "nu", "sigma_y", "q_1", "q_2", "q_3", "f_0", "f_c", "f_f",
        ]
    }

    /// The porosity, which **is** the state of a porous law.
    fn internal_names(&self) -> Vec<String> {
        vec!["porosity".to_string()]
    }

    /// The porosity starts at `f_0`, never at zero: a material that begins as a
    /// perfect solid has no void to grow and never damages.
    fn initial_internal_sources(&self) -> &'static [&'static str] {
        &["f_0"]
    }

    fn return_map(
        &self,
        trial: &[f64; 6],
        prev: &PrevState,
        mat: &MatParams,
        _dt: f64,
    ) -> Result<PlasticStep> {
        return_map(trial, prev, mat)
    }
}

crate::physics_operator! {
    /// [`model::gurson`](crate::ops::model::gurson()) — Gurson-Tvergaard-Needleman plasticity
    /// of a **porous** metal, where the porosity shrinks the yield surface.
    /// Material `E`, `nu`, `sigma_y`, `q_1`, `q_2`, `q_3`, `f_0`, `f_c`, `f_f`.
    ///
    /// A ductile metal fails because voids grow and coalesce, not because a
    /// stress is reached. The `cosh` term makes the surface **pressure
    /// sensitive**, so voids grow under triaxial tension and close under
    /// compression — which a J2 law cannot express, and which is why it can
    /// never predict ductile rupture. Beyond `f_c` the effective porosity
    /// accelerates towards `1/q_1`, modelling coalescence.
    ///
    /// The porosity is exposed as the internal variable `porosity`, starting
    /// from `f_0`. Void **nucleation** is not modelled — only growth.
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
    /// let m = model::gurson(&fes, Kinematics::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn gurson(fes, kinematics: Kinematics) = crate::ops::model::plasticity_with_law, PlasticLaw::Gurson;
    python: "`model.gurson(fespace, kinematics)` — Gurson-Tvergaard-Needleman plasticity\nof a **porous** metal, where the porosity shrinks the yield surface.\nMaterial `E`, `nu`, `sigma_y`, `q_1`, `q_2`, `q_3`, `f_0`, `f_c`, `f_f`.\n\nA ductile metal fails because voids grow and coalesce, not because a\nstress is reached. The `cosh` term makes the surface **pressure\nsensitive**, so voids grow under triaxial tension and close under\ncompression — which a J2 law cannot express, and which is why it can\nnever predict ductile rupture. Beyond `f_c` the effective porosity\naccelerates towards `1/q_1`, modelling coalescence.\n\nThe porosity is exposed as the internal variable `porosity`, starting\nfrom `f_0`. Void **nucleation** is not modelled — only growth."
}
