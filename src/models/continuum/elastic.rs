//! The **elastic operator** of the continuum — shared by every mechanical law
//! family, not the private property of one of them.
//!
//! Linear elasticity uses it as its whole constitutive law; a return-map law
//! uses it as its elastic **predictor** (`plasticity::law`), and a damage law as
//! the undamaged modulus it degrades. That is why it lives here and not in
//! [`crate::models::elasticity`]: three families call it, and none of them is
//! borrowing from another.
//!
//! Material contracts are **disjoint by symmetry**, which is what lets an
//! isotropic and an orthotropic zone live on one mesh without consolidation:
//! the assembler resolves a material zone by its required component set.

use crate::models::symmetry::MaterialSymmetry;
use crate::models::tensor::Kinematics;

/// Material components required by **isotropic** linear elasticity.
const MATERIAL_COMPONENTS: &[&str] = &["E", "nu"];

/// Material a continuum law **accepts without requiring**: the thermal-expansion
/// coefficient, consumed by
/// [`thermal_strain`](fn@crate::ops::element_field::thermal_strain), and the
/// density, which only the mass matrix wants. The order is the contract —
/// `ElementLayout::optional_material` resolves it slot for slot.
pub(crate) const OPTIONAL_COMPONENTS: &[&str] = &["alpha", "rho"];

/// Position of `rho` in [`OPTIONAL_COMPONENTS`].
pub(crate) const RHO_SLOT: usize = 1;

/// Orthotropic constants plus the in-plane material axis (2-D).
const ORTHOTROPIC_2D: &[&str] = &[
    "E_1", "E_2", "E_3", "nu_12", "nu_13", "nu_23", "G_12", "G_13", "G_23", "V1X", "V1Y",
];
/// Orthotropic constants plus the two material axes (3-D).
const ORTHOTROPIC_3D: &[&str] = &[
    "E_1", "E_2", "E_3", "nu_12", "nu_13", "nu_23", "G_12", "G_13", "G_23", "V1X", "V1Y", "V1Z",
    "V2X", "V2Y", "V2Z",
];
/// The 21 anisotropic constants plus the in-plane material axis (2-D).
const ANISOTROPIC_2D: &[&str] = &[
    "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_22", "C_23", "C_24", "C_25", "C_26", "C_33",
    "C_34", "C_35", "C_36", "C_44", "C_45", "C_46", "C_55", "C_56", "C_66", "V1X", "V1Y",
];
/// The 21 anisotropic constants plus the two material axes (3-D).
const ANISOTROPIC_3D: &[&str] = &[
    "C_11", "C_12", "C_13", "C_14", "C_15", "C_16", "C_22", "C_23", "C_24", "C_25", "C_26", "C_33",
    "C_34", "C_35", "C_36", "C_44", "C_45", "C_46", "C_55", "C_56", "C_66", "V1X", "V1Y", "V1Z",
    "V2X", "V2Y", "V2Z",
];

/// The material contract of a symmetry in a space of dimension `space_dim`:
/// the constants of the law, followed by the frame components it needs. Because
/// the assembler resolves a material zone by its **required component set**
/// ([`crate::ops::matrix::assemble_kind`]), these disjoint contracts let an
/// isotropic and an orthotropic zone live on one mesh without any consolidation.
pub(crate) fn material_contract(
    symmetry: MaterialSymmetry,
    space_dim: usize,
) -> &'static [&'static str] {
    match (symmetry, space_dim) {
        (MaterialSymmetry::Isotropic, _) => MATERIAL_COMPONENTS,
        (MaterialSymmetry::Orthotropic, 2) => ORTHOTROPIC_2D,
        (MaterialSymmetry::Orthotropic, _) => ORTHOTROPIC_3D,
        (MaterialSymmetry::Anisotropic, 2) => ANISOTROPIC_2D,
        (MaterialSymmetry::Anisotropic, _) => ANISOTROPIC_3D,
    }
}

/// Isotropic constitutive (Voigt) matrix `D` from `E`, `nu` and the kinematics.
///
/// ```
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::models::continuum::elastic;
/// // Contraintes planes : σ_zz = 0, la souplesse hors plan est condensée.
/// // Déformations planes : ε_zz = 0, le matériau est plus **raide**.
/// let cp = elastic::constitutive(210e3, 0.3, Kinematics::PlaneStress, 2);
/// let dp = elastic::constitutive(210e3, 0.3, Kinematics::PlaneStrain, 2);
/// assert!(dp[0][0] > cp[0][0]);
/// // En contraintes planes, D₀₀ = E/(1−ν²).
/// assert!((cp[0][0] - 210e3 / (1.0 - 0.09)).abs() < 1e-6);
/// // Le bloc de cisaillement vaut μ dans les deux cas (Voigt de l'ingénieur).
/// assert!((cp[2][2] - dp[2][2]).abs() < 1e-6);
/// ```
pub fn constitutive(e: f64, nu: f64, kinematics: Kinematics, space_dim: usize) -> Vec<Vec<f64>> {
    let mut d = [[0.0_f64; 6]; 6];
    let v = constitutive_into(e, nu, kinematics, space_dim, &mut d);
    d[..v].iter().map(|r| r[..v].to_vec()).collect()
}

/// [`constitutive`] writing into a caller-owned buffer, returning the Voigt size
/// `v` it filled (`d[..v][..v]`).
///
/// The form a constitutive kernel calls: at most thirty-six numbers, on the
/// stack. Building two levels of `Vec` for them is nothing once per assembly and
/// a great deal once per Gauss point of every iteration.
/// ```
/// # use pyrucast::models::continuum::elastic;
/// # use pyrucast::models::tensor::Kinematics;
/// // La même matrice que `constitutive`, écrite sur la pile : `v` dit
/// // combien de lignes et de colonnes ont été remplies.
/// let mut d = [[0.0_f64; 6]; 6];
/// let v = elastic::constitutive_into(210_000.0, 0.3, Kinematics::PlaneStress, 2, &mut d);
/// assert_eq!(v, 3);
/// let attendu = elastic::constitutive(210_000.0, 0.3, Kinematics::PlaneStress, 2);
/// assert_eq!(d[0][..v], attendu[0][..]);
/// ```
pub fn constitutive_into(
    e: f64,
    nu: f64,
    kinematics: Kinematics,
    space_dim: usize,
    d: &mut [[f64; 6]; 6],
) -> usize {
    match (space_dim, kinematics) {
        (2, Kinematics::PlaneStress) => {
            let c = e / (1.0 - nu * nu);
            d[0][0] = c;
            d[0][1] = c * nu;
            d[1][0] = c * nu;
            d[1][1] = c;
            d[2][2] = c * (1.0 - nu) / 2.0;
            3
        }
        (2, Kinematics::PlaneStrain) => {
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            d[0][0] = c * (1.0 - nu);
            d[0][1] = c * nu;
            d[1][0] = c * nu;
            d[1][1] = c * (1.0 - nu);
            d[2][2] = c * (1.0 - 2.0 * nu) / 2.0;
            3
        }
        (2, Kinematics::Axisymmetric) => {
            // Voigt order [rr, zz, θθ, rz]: the three normal directions are
            // mutually orthogonal, so the 3×3 normal block is the isotropic one
            // (as in plane strain, with θθ restored) and `rz` is the lone shear.
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let (d_n, d_off) = (c * (1.0 - nu), c * nu);
            for i in 0..3 {
                for j in 0..3 {
                    d[i][j] = if i == j { d_n } else { d_off };
                }
            }
            d[3][3] = c * (1.0 - 2.0 * nu) / 2.0;
            4
        }
        _ => {
            // 3-D solid (Voigt order [xx, yy, zz, yz, xz, xy]).
            let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let g = c * (1.0 - 2.0 * nu) / 2.0;
            for i in 0..3 {
                for j in 0..3 {
                    d[i][j] = if i == j { c * (1.0 - nu) } else { c * nu };
                }
            }
            d[3][3] = g;
            d[4][4] = g;
            d[5][5] = g;
            6
        }
    }
}

/// Lamé coefficients `(λ, μ)` from `E`, `nu`.
///
/// ```
/// # use pyrucast::models::continuum::elastic;
/// let (lambda, mu) = elastic::lame(210_000.0, 0.3);
/// // μ = E / 2(1+ν), λ = Eν / (1+ν)(1−2ν).
/// assert!((mu - 210_000.0 / 2.6).abs() < 1e-9);
/// assert!((lambda - 121_153.846_153_85).abs() < 1e-6);
/// ```
pub fn lame(e: f64, nu: f64) -> (f64, f64) {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    (lambda, mu)
}

/// Isotropic elastic stress (full 3-D, order `[xx, yy, zz, yz, xz, xy]`) from a
/// **tensor** strain: `σ = λ tr(ε) I + 2μ ε`.
///
/// ```
/// # use pyrucast::models::continuum::elastic;
/// let (lambda, mu) = elastic::lame(210_000.0, 0.3);
/// // Un cisaillement pur **tensoriel** ε_xy = 1 donne σ_xy = 2μ.
/// let s = elastic::elastic_stress(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], lambda, mu);
/// assert!((s[5] - 2.0 * mu).abs() < 1e-9);
/// assert!(s[0].abs() < 1e-9); // trace nulle ⇒ pas de part sphérique
/// ```
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

/// The elastic modulus in full-3-D engineering Voigt — the tangent wherever the
/// step stayed elastic, and the starting point of every analytic one.
///
/// ```
/// # use pyrucast::models::continuum::elastic;
/// let (lambda, mu) = elastic::lame(210_000.0, 0.3);
/// let d = elastic::elastic_tangent(lambda, mu);
/// // Voigt **de l'ingénieur** : le bloc de cisaillement vaut μ, non 2μ.
/// assert!((d[3][3] - mu).abs() < 1e-9);
/// assert!((d[0][0] - (lambda + 2.0 * mu)).abs() < 1e-9);
/// ```
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
