//! Mazars isotropic damage — the classical concrete law.
//!
//! Damage is driven by an **equivalent strain** built from the positive
//! principal strains, `ε̃ = √(Σ⟨ε_I⟩₊²)`, so it responds to extension and is
//! blind to hydrostatic compression. The stress is the effective one, degraded:
//! `σ = (1 − D)·C:ε`.
//!
//! Two branches, tension and compression, are blended by the weights `α_t`,
//! `α_c` derived from the split of the effective stress — which is what lets one
//! law describe a material an order of magnitude stronger in compression.
//!
//! The single history variable is `κ = max_t ε̃`: damage never heals.

use super::law::DamageLawKind;
use crate::error::Result;
use crate::models::damage::law::DamageLaw;
use crate::models::damage::law::{pos, DamageUpdate, MatRead};
use crate::models::elasticity::{elastic_stress, lame};
use crate::models::tensor::Kinematics;
use nalgebra::Matrix3;

/// Positions in this law's material contract, [`MATERIAL`].
const E: usize = 0;
const NU: usize = 1;
const EPS_D0: usize = 2;
const A_T: usize = 3;
const B_T: usize = 4;
const A_C: usize = 5;
const B_C: usize = 6;

/// Material parameters of the Mazars kinematics at one Gauss point.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm))?;
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0)?, vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(),
/// #                       "B_t".into(), "A_c".into(), "B_c".into()],
/// #     &[30_000.0, 0.2, 1e-4, 0.8, 20_000.0, 1.4, 1_850.0])?;
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
/// // Les paramètres de Mazars, lus une fois par maille : le seuil `eps_d0`
/// // et les deux branches, traction et compression. Ils ne sont pas
/// // exposés champ par champ — c'est `update` qui les emploie.
/// let u = damage::mazars::update(&[1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0], &mat)?;
/// assert!(u.damage > 0.0);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct MazarsParams {
    e: f64,
    nu: f64,
    eps_d0: f64,
    a_t: f64,
    b_t: f64,
    a_c: f64,
    b_c: f64,
}

/// One damage branch `D = 1 − eps_d0(1−A)/κ − A / exp(B (κ − eps_d0))`,
/// clamped to `[0, 1)`.
fn damage_branch(kappa: f64, eps_d0: f64, a: f64, b: f64) -> f64 {
    let d = 1.0 - eps_d0 * (1.0 - a) / kappa - a / (b * (kappa - eps_d0)).exp();
    d.clamp(0.0, 1.0 - 1e-12)
}

/// Mazars point update. Returns `(stress, damage, kappa)` (stress full 3-D Voigt).
fn mazars_update(eps: &[f64; 6], kappa_old: f64, p: &MazarsParams) -> ([f64; 6], f64, f64) {
    let (lambda, mu) = lame(p.e, p.nu);
    let sigma_eff = elastic_stress(eps, lambda, mu);

    // Principal strains (coaxial with the effective stress, isotropic elasticity).
    let tensor = Matrix3::new(
        eps[0], eps[5], eps[4], // [εxx, εxy, εxz]
        eps[5], eps[1], eps[3], // [εxy, εyy, εyz]
        eps[4], eps[3], eps[2], // [εxz, εyz, εzz]
    );
    let e_pr = tensor.symmetric_eigenvalues();

    // Equivalent strain ε̃ = √(Σ ⟨ε_I⟩₊²).
    let eps_eq = (e_pr.iter().map(|&x| pos(x).powi(2)).sum::<f64>()).sqrt();

    // History variable: never below the threshold, never decreasing.
    let kappa = kappa_old.max(p.eps_d0).max(eps_eq);
    if kappa <= p.eps_d0 {
        return (sigma_eff, 0.0, kappa); // undamaged
    }

    // Tension/compression split of the effective principal stresses
    // σ̃_I = λ·tr + 2μ·ε_I, then strains induced by each part via the
    // isotropic compliance (all coaxial ⇒ work in principal space).
    let tr = e_pr[0] + e_pr[1] + e_pr[2];
    let st: [f64; 3] = std::array::from_fn(|i| lambda * tr + 2.0 * mu * e_pr[i]);
    let stp: [f64; 3] = std::array::from_fn(|i| pos(st[i]));
    let stn: [f64; 3] = std::array::from_fn(|i| st[i].min(0.0));
    let sum_p: f64 = stp.iter().sum();
    let sum_n: f64 = stn.iter().sum();
    // ε^t_I = [(1+ν)σ̃⁺_I − ν Σσ̃⁺] / E ; ε^c_I likewise from σ̃⁻.
    let eps_t: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stp[i] - p.nu * sum_p) / p.e);
    let eps_c: [f64; 3] = std::array::from_fn(|i| ((1.0 + p.nu) * stn[i] - p.nu * sum_n) / p.e);

    let denom = eps_eq * eps_eq;
    let mut alpha_t = 0.0;
    let mut alpha_c = 0.0;
    if denom > 0.0 {
        for i in 0..3 {
            let w = pos(e_pr[i]);
            alpha_t += pos(eps_t[i]) * w;
            alpha_c += pos(eps_c[i]) * w;
        }
        alpha_t /= denom;
        alpha_c /= denom;
    }
    let alpha_t = alpha_t.clamp(0.0, 1.0);
    let alpha_c = alpha_c.clamp(0.0, 1.0);

    let d_t = damage_branch(kappa, p.eps_d0, p.a_t, p.b_t);
    let d_c = damage_branch(kappa, p.eps_d0, p.a_c, p.b_c);
    // β fixed to 1 (no shear correction).
    let damage = (alpha_t * d_t + alpha_c * d_c).clamp(0.0, 1.0 - 1e-12);

    let sigma: [f64; 6] = std::array::from_fn(|i| (1.0 - damage) * sigma_eff[i]);
    (sigma, damage, kappa)
}

/// The law's material contract and its history variable.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
/// // Un seuil `eps_d0`, puis deux branches : traction (A_t, B_t) et
/// // compression (A_c, B_c), mélangées par la part de traction.
/// assert!(damage::mazars::MATERIAL.contains(&"eps_d0"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub const MATERIAL: &[&str] = &["E", "nu", "eps_d0", "A_t", "B_t", "A_c", "B_c"];

/// One Mazars step: `(σ, D, κ)` from the strain and the previous history.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::SubElementField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::damage::{self};
/// # use pyrucast::models::damage::law::{DamageLaw, MatRead};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
/// # let materiau = SubElementField::from_uniform_per_component(
/// #     fes.get(0).unwrap(), vec!["E".into(), "nu".into(), "eps_d0".into(), "A_t".into(), "B_t".into(), "A_c".into(), "B_c".into()], &[30000.0, 0.2, 0.0001, 0.8, 20000.0, 1.4, 1850.0]).unwrap();
/// # let idx_mat: Vec<u32> = (0..materiau.point_values(0, 0).unwrap().len() as u32).collect();
/// # let opt_mat = [pyrucast::containers::field::ABSENT_COMPONENT; 8];
/// # let mat = MatRead { row: materiau.point_values(0, 0).unwrap(), idx: &idx_mat };
/// // `kappa` est la mémoire de la loi : il **ne décroît pas**. Décharger
/// // après avoir endommagé ne répare rien.
/// let grand = [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let charge = damage::mazars::update(&grand, &[0.0], &mat)?;
/// let petit = [1e-5, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let decharge = damage::mazars::update(&petit, &charge.vars, &mat)?;
/// assert_eq!(decharge.vars[0], charge.vars[0]);
/// // `damage` est recalculé depuis κ, d'où l'égalité à l'arrondi près.
/// assert!((decharge.damage - charge.damage).abs() < 1e-12);
/// // La contrainte, elle, retombe : la raideur est celle du matériau
/// // endommagé, pas celle du matériau sain.
/// assert!(decharge.sigma[0] < charge.sigma[0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn update(eps: &[f64; 6], prev: &[f64], mat: &MatRead) -> Result<DamageUpdate> {
    let p = MazarsParams {
        e: mat.get(E),
        nu: mat.get(NU),
        eps_d0: mat.get(EPS_D0),
        a_t: mat.get(A_T),
        b_t: mat.get(B_T),
        a_c: mat.get(A_C),
        b_c: mat.get(B_C),
    };
    let kappa_old = prev.first().copied().unwrap_or(0.0);
    let (sigma, damage, kappa) = mazars_update(eps, kappa_old, &p);
    Ok(DamageUpdate {
        sigma,
        damage,
        vars: vec![kappa],
    })
}

/// Mazars isotropic damage — the classical concrete law.
pub(crate) struct Mazars;

impl DamageLawKind for Mazars {
    fn material_components(&self, _space_dim: usize) -> &'static [&'static str] {
        MATERIAL
    }

    fn internal_names(&self) -> Vec<String> {
        vec!["kappa".into()]
    }

    fn update(
        &self,
        eps: &[f64; 6],
        prev: &[f64],
        mat: &MatRead,
        _space_dim: usize,
    ) -> Result<DamageUpdate> {
        update(eps, prev, mat)
    }
}

crate::physics_operator! {
    /// [`model::mazars`](crate::ops::model::mazars()) — Mazars isotropic damage spanning every
    /// subspace of `fespace`. `kinematics` is `"plane_stress"` / `"plane_strain"` /
    /// `"axisymmetric"` (2-D) or `"full_3d"` (3-D). Same DOFs as elasticity; material
    /// (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at
    /// assembly / integration time. The behaviour integration (`COMP`) carries
    /// the scalar history variable `kappa` (`VAR0`→`VAR1`) and outputs `damage`.
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
    /// let m = model::mazars(&fes, Kinematics::PlaneStrain)?;
    /// assert_eq!(m.primal_vars()?, vec!["u_x".to_string(), "u_y".to_string()]);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn mazars(fes, kinematics: Kinematics) = crate::ops::model::damage_with_law, DamageLaw::Mazars;
    python: "`kinematics.mazars(fespace, kinematics)` — Mazars isotropic damage spanning every\nsubspace of `fespace`. `kinematics` is `\"plane_stress\"` / `\"plane_strain\"` /\n`\"axisymmetric\"` (2-D) or `\"solid\"` (3-D). Same DOFs as elasticity; material\n(`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at\nassembly / integration time. The behaviour integration (`COMP`) carries\nthe scalar history variable `kappa` (`VAR0`→`VAR1`) and outputs `damage`."
}
