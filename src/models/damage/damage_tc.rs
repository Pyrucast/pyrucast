//! Damage TC — **two** damage variables, one for tension and one for compression.
//!
//! Mazars blends its two branches into a single scalar `D`. That is economical,
//! but it means a material damaged in compression is equally damaged in tension:
//! the model cannot represent a crack that **closes** and carries load again.
//!
//! Damage TC (Faria-Oliver-Cervera) keeps the two apart:
//!
//! ```text
//! σ = (1 − d⁺)·σ̃⁺ + (1 − d⁻)·σ̃⁻
//! ```
//!
//! where `σ̃⁺` and `σ̃⁻` are the positive and negative parts of the **effective**
//! stress, split on its principal values. Each part is degraded by its own
//! variable, so unloading from tension into compression recovers the compressive
//! stiffness — the *unilateral* effect, which is what makes the law usable under
//! cyclic loading and what a single scalar cannot express.
//!
//! ## The two drivers
//!
//! Each damage has its own equivalent stress and its own history variable:
//!
//! ```text
//! τ⁺ = √(σ̃⁺ : ε)                          (tensile energy)
//! τ⁻ = √(√3·(K·σ̃⁻_oct + τ̃⁻_oct))           (compressive, octahedral)
//! r⁺ = max(r₀⁺, max_t τ⁺)                 r⁻ = max(r₀⁻, max_t τ⁻)
//! ```
//!
//! and each evolves by its own softening law — exponential in tension (a brittle
//! crack), and with a hardening-then-softening shape in compression.
//!
//! `r₀⁺ = f_t/√E` and `r₀⁻ = f_c/√E` are the thresholds; `A_t` and `A_c` set how
//! fast each branch softens.

use crate::error::Result;
use crate::models::damage::{elastic_stress, lame, pos, DamageUpdate, MatRead};
use nalgebra::Matrix3;

/// The law's material contract.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "f_t".into(), "f_c".into(), "A_t".into(), "A_c".into()], &[30000.0, 0.2, 3.0, 30.0, 0.5, 0.5]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // Deux résistances et deux fragilités : traction et compression sont
/// // suivies séparément.
/// assert!(damage::damage_tc::MATERIAL.contains(&"f_t"));
/// assert!(damage::damage_tc::MATERIAL.contains(&"f_c"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL: &[&str] = &["E", "nu", "f_t", "f_c", "A_t", "A_c"];

/// One Damage-TC step.
///
/// `prev` carries `[r⁺, r⁻]`; the update returns them alongside the two damages
/// and the degraded stress.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self, DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "f_t".into(), "f_c".into(), "A_t".into(), "A_c".into()], &[30000.0, 0.2, 3.0, 30.0, 0.5, 0.5]).unwrap();
/// # let mat = MatRead { field: &materiau, cell: 0 };
/// // Deux endommagements distincts : la traction en active un, la
/// // compression l'autre — c'est ce qui **restitue la raideur** quand une
/// // fissure se referme.
/// let traction = damage::damage_tc::update(&[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 4], &mat)?;
/// let compression =
///     damage::damage_tc::update(&[-1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0; 4], &mat)?;
/// assert_eq!(traction.vars.len(), 4); // r⁺, r⁻, d⁺, d⁻
/// assert!(traction.vars[2] > compression.vars[2]); // d⁺ : la traction seule
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn update(eps: &[f64; 6], prev: &[f64], mat: &MatRead) -> Result<DamageUpdate> {
    let e = mat.get("E")?;
    let nu = mat.get("nu")?;
    let (f_t, f_c) = (mat.get("f_t")?, mat.get("f_c")?);
    let (a_t, a_c) = (mat.get("A_t")?, mat.get("A_c")?);
    let (lambda, mu) = lame(e, nu);

    let sigma_eff = elastic_stress(eps, lambda, mu);

    // Principal effective stresses, and the tension/compression split. Isotropic
    // elasticity keeps stress and strain coaxial, so the split can be done in
    // the principal frame and rotated back.
    let tensor = Matrix3::new(
        sigma_eff[0],
        sigma_eff[5],
        sigma_eff[4],
        sigma_eff[5],
        sigma_eff[1],
        sigma_eff[3],
        sigma_eff[4],
        sigma_eff[3],
        sigma_eff[2],
    );
    let eig = tensor.symmetric_eigen();
    let principal = eig.eigenvalues;
    let vectors = eig.eigenvectors;

    let mut plus = Matrix3::zeros();
    let mut minus = Matrix3::zeros();
    for i in 0..3 {
        let v = vectors.column(i);
        let outer = v * v.transpose();
        plus += outer * pos(principal[i]);
        minus += outer * principal[i].min(0.0);
    }

    // ── The two equivalent stresses ────────────────────────────────────────
    // Tensile: the elastic energy stored by the positive part.
    let eps_tensor = Matrix3::new(
        eps[0], eps[5], eps[4], eps[5], eps[1], eps[3], eps[4], eps[3], eps[2],
    );
    let tau_plus = pos((plus.component_mul(&eps_tensor)).sum()).sqrt();

    // Compressive: the octahedral measure, which is what makes the compressive
    // branch respond to confinement rather than to extension alone.
    let oct_normal = (minus[(0, 0)] + minus[(1, 1)] + minus[(2, 2)]) / 3.0;
    let dev = minus - Matrix3::identity() * oct_normal;
    let oct_shear = ((dev.component_mul(&dev)).sum() / 3.0).max(0.0).sqrt();
    let k_param = 0.171; // the usual value, from the biaxial/uniaxial strength ratio
    let tau_minus = (3.0_f64.sqrt() * (k_param * oct_normal + oct_shear).abs()).sqrt();

    // ── History variables: damage never heals ──────────────────────────────
    let r0_plus = (f_t / e.sqrt()).max(1e-30);
    let r0_minus = (f_c / e.sqrt()).max(1e-30);
    let r_plus = prev
        .first()
        .copied()
        .unwrap_or(0.0)
        .max(r0_plus)
        .max(tau_plus);
    let r_minus = prev
        .get(1)
        .copied()
        .unwrap_or(0.0)
        .max(r0_minus)
        .max(tau_minus);

    // ── The two softening laws ─────────────────────────────────────────────
    // Tension: exponential softening, the brittle response of a crack.
    let d_plus = if r_plus > r0_plus {
        (1.0 - (r0_plus / r_plus) * (a_t * (1.0 - r_plus / r0_plus)).exp()).clamp(0.0, 1.0 - 1e-12)
    } else {
        0.0
    };
    // Compression: hardening then softening — concrete crushes, it does not snap.
    let d_minus = if r_minus > r0_minus {
        (1.0 - (r0_minus / r_minus) * (1.0 - a_c) - a_c * (2.0 * (1.0 - r_minus / r0_minus)).exp())
            .clamp(0.0, 1.0 - 1e-12)
    } else {
        0.0
    };

    // ── The degraded stress: each part by its own damage ───────────────────
    let degraded = plus * (1.0 - d_plus) + minus * (1.0 - d_minus);
    let sigma = [
        degraded[(0, 0)],
        degraded[(1, 1)],
        degraded[(2, 2)],
        degraded[(1, 2)],
        degraded[(0, 2)],
        degraded[(0, 1)],
    ];

    Ok(DamageUpdate {
        sigma,
        // The reported scalar is the worse of the two — a summary for
        // visualisation, not the state, which is the pair below.
        damage: d_plus.max(d_minus),
        vars: vec![r_plus, r_minus, d_plus, d_minus],
    })
}
