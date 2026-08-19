//! Ottosen's four-parameter criterion — concrete.
//!
//! Concrete fails very differently in tension and in compression, and its
//! strength under a given pressure depends on **which** deviatoric direction the
//! stress points in. Neither von Mises (blind to pressure) nor Drucker-Prager
//! (blind to that direction) captures it. Ottosen's surface does, through a
//! dependence on the **Lode angle**:
//!
//! ```text
//! f(σ) = a·J₂/σ_c² + λ(θ)·√J₂/σ_c + b·I₁/σ_c − 1
//! ```
//!
//! ```text
//! λ(θ) = k₁·cos[⅓ arccos(k₂ cos3θ)]              if cos3θ ≥ 0
//! λ(θ) = k₁·cos[π/3 − ⅓ arccos(−k₂ cos3θ)]       if cos3θ < 0
//! ```
//!
//! with `cos3θ = (3√3/2)·J₃/J₂^{3/2}`. The four parameters `a`, `b`, `k₁`, `k₂`
//! are fitted to the uniaxial tensile and compressive strengths, the biaxial
//! compressive strength and one triaxial point; `σ_c` is the compressive
//! strength that scales the whole surface.
//!
//! The meridians are **curved** (the `J₂` term) and the deviatoric section is a
//! smooth rounded triangle that opens out towards compression — which is the
//! whole point.
//!
//! ## Integrated by cutting plane, with a numerical normal
//!
//! There is no usable closed-form return onto this surface. Worse, the normal
//! `∂f/∂σ` requires differentiating `λ(θ)` through `arccos` and `J₃` — an
//! expression long enough that a sign error in it would be invisible in review
//! and would show up only as a slightly wrong flow direction.
//!
//! So the return goes through the **cutting-plane** algorithm, which needs only
//! the scalar `f(σ)`, with the normal obtained by **central differences**. The
//! criterion is then exact, and the gradient accurate to `O(h²)`. Trading an
//! unverifiable analytic gradient for a numerical one that cannot be
//! mis-derived is the right trade here; the consistent tangent follows the same
//! reasoning ([`crate::models::plastic::consistent_tangent`]).
//!
//! Flow is **associated** (`g = f`), the usual choice for this criterion.

use crate::error::{PyrucastError, Result};
use crate::models::plastic::{
    elastic_stress, i1, j2, j3, require_positive, MatParams, PlasticLaw, PlasticStep, PrevState,
};

/// Ottosen's yield function, exactly as written above. Negative inside the
/// elastic domain.
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
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "a".into(), "b".into(), "k_1".into(), "k_2".into(), "sigma_c".into()], &[30000.0, 0.2, 1.2759, 3.1962, 11.7365, 0.9801, 30.0]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: Vec::new() };
/// // Le critère distingue traction et compression par l'angle de Lode :
/// // à contrainte équivalente égale, la traction est bien plus pénalisante.
/// let q = 20.0;
/// let traction = plastic::ottosen::yield_function(&[q, 0.0, 0.0, 0.0, 0.0, 0.0], &mat)?;
/// let compression =
///     plastic::ottosen::yield_function(&[-q, 0.0, 0.0, 0.0, 0.0, 0.0], &mat)?;
/// assert!(traction > compression);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn yield_function(sigma: &[f64; 6], mat: &MatParams) -> Result<f64> {
    let a = mat.get("a")?;
    let b = mat.get("b")?;
    let k1 = mat.get("k_1")?;
    let k2 = mat.get("k_2")?;
    let sc = require_positive(PlasticLaw::Ottosen, "sigma_c", mat.get("sigma_c")?)?;

    let j2v = j2(sigma);
    let sqrt_j2 = j2v.max(0.0).sqrt();
    // A purely hydrostatic stress has no deviatoric direction; the Lode term
    // vanishes with √J₂ anyway, so `λ` may be evaluated at any angle there.
    let cos3t = if j2v > 1e-30 {
        (3.0 * 3.0_f64.sqrt() / 2.0 * j3(sigma) / j2v.powf(1.5)).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    let lambda = if cos3t >= 0.0 {
        k1 * ((k2 * cos3t).clamp(-1.0, 1.0).acos() / 3.0).cos()
    } else {
        k1 * (std::f64::consts::FRAC_PI_3 - (-k2 * cos3t).clamp(-1.0, 1.0).acos() / 3.0).cos()
    };
    Ok(a * j2v / (sc * sc) + lambda * sqrt_j2 / sc + b * i1(sigma) / sc - 1.0)
}

/// `∂f/∂σ` by central differences, in the **tensor** sense (the caller contracts
/// it with the elastic modulus, which is also tensorial).
fn numerical_normal(sigma: &[f64; 6], mat: &MatParams, scale: f64) -> Result<[f64; 6]> {
    let h = 1e-6 * scale;
    let mut n = [0.0; 6];
    for (i, ni) in n.iter_mut().enumerate() {
        let mut plus = *sigma;
        let mut minus = *sigma;
        plus[i] += h;
        minus[i] -= h;
        *ni = (yield_function(&plus, mat)? - yield_function(&minus, mat)?) / (2.0 * h);
    }
    Ok(n)
}

/// Cutting-plane return onto Ottosen's surface.
///
/// The algorithm is deliberately the *semi-implicit* one: the normal is
/// re-evaluated at the current iterate rather than solved for implicitly. It
/// converges robustly on a strongly curved surface without needing second
/// derivatives, which is what makes it the right choice here.
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
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "a".into(), "b".into(), "k_1".into(), "k_2".into(), "sigma_c".into()], &[30000.0, 0.2, 1.2759, 3.1962, 11.7365, 0.9801, 30.0]).unwrap();
/// # let mat = MatParams::new(&materiau, 0).unwrap();
/// # let repos = PrevState { eps: [0.0; 6], sigma: [0.0; 6], eps_p: [0.0; 6], p: 0.0,
/// #                         vars: Vec::new() };
/// // Un essai hors surface est ramené dessus : la fonction de charge
/// // s'annule à la solution.
/// let trial = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let pas = plastic::ottosen::return_map(&trial, &repos, &mat)?;
/// assert!(plastic::ottosen::yield_function(&pas.sigma, &mat)?.abs() < 1e-4);
/// assert!(pas.p > 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn return_map(trial: &[f64; 6], prev: &PrevState, mat: &MatParams) -> Result<PlasticStep> {
    let sc = mat.get("sigma_c")?;
    if yield_function(trial, mat)? <= 0.0 {
        return Ok(PlasticStep::elastic(trial, prev)); // elastic
    }

    let mut sigma = *trial;
    let mut eps_p = prev.eps_p;
    let mut p = prev.p;
    // `f` is dimensionless (normalised by σ_c), so the tolerance is too.
    let tol = 1e-12;

    for _ in 0..100 {
        let f = yield_function(&sigma, mat)?;
        if f.abs() <= tol {
            break;
        }
        let n = numerical_normal(&sigma, mat, sc)?;
        // The flow increment for a unit multiplier, and its elastic image.
        // `n` is a tensor gradient, so the shear components are already the
        // tensorial ones — `elastic_stress` consumes exactly that.
        let cn = elastic_stress(&n, mat.lambda, mat.mu);
        // n : C : n, with the off-diagonals counted twice (a tensor double dot).
        let ncn: f64 = (0..6)
            .map(|i| n[i] * cn[i] * if i < 3 { 1.0 } else { 2.0 })
            .sum();
        if ncn.abs() < f64::MIN_POSITIVE {
            return Err(PyrucastError::Message(
                "plasticity (ottosen): the yield normal is degenerate — check that a, b, k_1, \
                 k_2 describe a convex surface"
                    .into(),
            ));
        }
        let dlambda = f / ncn;
        for i in 0..6 {
            sigma[i] -= dlambda * cn[i];
            eps_p[i] += dlambda * n[i];
        }
        p += dlambda.abs();
    }

    if yield_function(&sigma, mat)?.abs() > 1e-6 {
        return Err(PyrucastError::Message(
            "plasticity (ottosen): the return mapping did not converge in 100 iterations — the \
             strain increment may be too large, or the parameter set non-convex"
                .into(),
        ));
    }
    Ok(PlasticStep {
        sigma,
        eps_p,
        p,
        vars: Vec::new(),
    })
}
