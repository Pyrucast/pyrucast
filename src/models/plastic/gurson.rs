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
//! Tvergaard and Needleman model that by feeding the surface an *effective*
//! porosity:
//!
//! ```text
//! f* = f                                        if f ≤ f_c
//! f* = f_c + (1/q₁ − f_c)·(f − f_c)/(f_f − f_c)  otherwise
//! ```
//!
//! which reaches `1/q₁` — where the surface has shrunk to nothing — at the
//! failure porosity `f_f`. Without this the model predicts a far too ductile
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
//! normal** as [`ottosen`](crate::models::plastic::ottosen). The normal here has
//! both a deviatoric and a volumetric part, which is exactly what makes the
//! plastic flow dilatant.

use crate::error::{PyrucastError, Result};
use crate::models::plastic::{
    elastic_stress, i1, require_positive, von_mises_stress, MatParams, PlasticLaw, PlasticStep,
    PrevState,
};

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
/// # use pyrucast::models::plastic::{self, MatParams, PlasticLaw, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into(), "q_1".into(), "q_2".into(), "q_3".into(), "f_0".into(), "f_c".into(), "f_f".into()], &[210000.0, 0.3, 250.0, 1.5, 1.0, 2.25, 0.001, 0.15, 0.25]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: vec![0.001] };
/// // La porosité **rétrécit** la surface de charge : à contrainte égale, un
/// // métal plus poreux est plus près de céder.
/// let s = [200.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let sain = plastic::gurson::yield_function(&s, 0.001, &mat)?;
/// let poreux = plastic::gurson::yield_function(&s, 0.05, &mat)?;
/// assert!(poreux > sain);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn yield_function(sigma: &[f64; 6], f: f64, mat: &MatParams) -> Result<f64> {
    let sigma_y = require_positive(PlasticLaw::Gurson, "sigma_y", mat.get("sigma_y")?)?;
    let (q1, q2, q3) = (mat.get("q_1")?, mat.get("q_2")?, mat.get("q_3")?);
    let (f_c, f_f) = (mat.get("f_c")?, mat.get("f_f")?);
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
/// # use pyrucast::models::plastic::{self, MatParams, PlasticLaw, PrevState};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "sigma_y".into(), "q_1".into(), "q_2".into(), "q_3".into(), "f_0".into(), "f_c".into(), "f_f".into()], &[210000.0, 0.3, 250.0, 1.5, 1.0, 2.25, 0.001, 0.15, 0.25]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: vec![0.001] };
/// // La porosité **est** l'état : elle croît avec l'écoulement, et c'est
/// // elle qui mène à la rupture ductile.
/// let trial = [800.0, 400.0, 400.0, 0.0, 0.0, 0.0];
/// let pas = plastic::gurson::return_map(&trial, &repos, &mat)?;
/// assert!(pas.p > 0.0);
/// assert_eq!(pas.vars.len(), 1);
/// assert!(pas.vars[0] >= repos.vars[0]); // la porosité ne décroît pas
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn return_map(trial: &[f64; 6], prev: &PrevState, mat: &MatParams) -> Result<PlasticStep> {
    let sigma_y = mat.get("sigma_y")?;
    // The porosity at A: the law's own variable, or the **initial** porosity on
    // the very first step, where the state carries nothing yet. A default of
    // zero would start the material as a perfect solid and never let it damage.
    let f_a = if prev.vars.is_empty() {
        mat.get("f_0")?
    } else {
        prev.var(0)
    };

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
        f = (f + (1.0 - f) * dev_vol).clamp(0.0, mat.get("f_f")?);
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
