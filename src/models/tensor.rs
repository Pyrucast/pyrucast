//! The **Voigt** representation of a symmetric tensor — the kinematic
//! hypothesis that fixes its layout, the names that go with it, and the algebra
//! on top.
//!
//! Nothing here belongs to one physics. [`Kinematics`] is used by elasticity,
//! plasticity and damage alike; the invariants by anything reading a stress.
//! The type used to be called `Kinematics` and to live in
//! `models/elasticity.rs` — a name and a place that said elasticity where it
//! meant *how a 3-D problem is reduced*.
//!
//! Voigt order is `[xx, yy, zz, yz, xz, xy]`, with **engineering** shear.

use serde::{Deserialize, Serialize};

/// Which 2-D assumption (or 3-D solid) to use for the constitutive matrix.
///
/// ```
/// # use pyrucast::models::tensor::Kinematics;
/// # use pyrucast::models::elasticity::{self};
/// // La cinématique choisie décide du nombre de composantes de Voigt.
/// assert_eq!(elasticity::constitutive(210e3, 0.3, Kinematics::PlaneStress, 2).len(), 3);
/// assert_eq!(elasticity::constitutive(210e3, 0.3, Kinematics::Axisymmetric, 2).len(), 4);
/// assert_eq!(elasticity::constitutive(210e3, 0.3, Kinematics::Full3D, 3).len(), 6);
/// ```
/// # use pyrucast::models::tensor::Kinematics;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kinematics {
    /// 2-D plane stress (thin plate loaded in its plane).
    PlaneStress,
    /// 2-D plane strain (long prismatic body, `εzz = 0`).
    PlaneStrain,
    /// 2-D meridian plane of a body of revolution: four Voigt components, the
    /// hoop strain `ε_θθ = u_r / r` among them. Requires an axisymmetric
    /// geometry.
    Axisymmetric,
    /// The **unreduced** problem: full 3-D, no plane assumption.
    ///
    /// Named for what it is rather than for the object it describes —
    /// the other three name a hypothesis, this one names its absence.
    /// The name it answers to stays `"full_3d"`, the term the field uses
    /// (Cast3M, Abaqus) and the one the Cast3M correspondence documents.
    Full3D,
}

impl Kinematics {
    /// Whether this kinematics carries the hoop (θθ) component — i.e. is
    /// [`Axisymmetric`](Self::Axisymmetric).
    ///
    /// ```
    /// # use pyrucast::models::tensor::Kinematics;
    /// # use pyrucast::models::elasticity::{self};
    /// // Seul le plan méridien d'un corps de révolution porte la déformation
    /// // orthoradiale ε_θθ = u_r / r.
    /// assert!(Kinematics::Axisymmetric.is_axisymmetric());
    /// assert!(!Kinematics::PlaneStrain.is_axisymmetric());
    /// ```
    /// # use pyrucast::models::tensor::Kinematics;
    pub fn is_axisymmetric(self) -> bool {
        self == Self::Axisymmetric
    }
}

impl crate::named::Named for Kinematics {
    const LABEL: &'static str = "kinematics";
    const VALUES: &'static [Self] = &[
        Self::PlaneStress,
        Self::PlaneStrain,
        Self::Axisymmetric,
        Self::Full3D,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::PlaneStress => "plane_stress",
            Self::PlaneStrain => "plane_strain",
            Self::Axisymmetric => "axisymmetric",
            Self::Full3D => "full_3d",
        }
    }

    /// `solid` — the field's own word (Cast3M, Abaqus), and the one the
    /// Cast3M correspondence documents. Accepted on input, never printed back:
    /// the canonical name states the *hypothesis*, the alias names the object.
    fn aliases() -> &'static [(&'static str, Self)] {
        &[("solid", Self::Full3D)]
    }
}

/// Axis suffixes for the vector components, indexed by spatial direction.
const AXES: [&str; 3] = ["x", "y", "z"];

/// Where each **axisymmetric** Voigt slot `[rr, zz, θθ, rz]` sits in the full
/// 3-D order [`TENSOR_SUFFIXES`] (`[xx, yy, zz, yz, xz, xy]`). The whole
/// axisymmetric specialisation of this law is this one index map: the state and
/// the radial return stay full 3-D, only the projection in and out changes.
const AXI_TO_3D: [usize; 4] = [0, 1, 2, 5];

pub(crate) fn primal_name(a: usize) -> String {
    format!("u_{}", AXES[a])
}

pub(crate) fn dual_name(a: usize) -> String {
    format!("f_{}", AXES[a])
}

/// Stress component names in Voigt order. Axisymmetric names the hoop `θθ`
/// component `sigma_zz`, after Cast3M (`x = r`, `y = z`).
/// Project the full 3-D stress to the kinematics's Voigt slot `r`.
/// 2-D order is `[xx, yy, xy]`; 3-D is the full `[xx, yy, zz, yz, xz, xy]`.
pub(crate) fn voigt_stress(
    sigma: &[f64; 6],
    space_dim: usize,
    kinematics: Kinematics,
    r: usize,
) -> f64 {
    if space_dim == 2 && kinematics.is_axisymmetric() {
        sigma[AXI_TO_3D[r]]
    } else if space_dim == 2 {
        match r {
            0 => sigma[0],
            1 => sigma[1],
            _ => sigma[5],
        }
    } else {
        sigma[r]
    }
}

pub(crate) fn stress_names(space_dim: usize, kinematics: Kinematics) -> Vec<String> {
    match (space_dim, kinematics) {
        (2, Kinematics::Axisymmetric) => vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_xy".into(),
        ],
        (2, _) => vec!["sigma_xx".into(), "sigma_yy".into(), "sigma_xy".into()],
        _ => vec![
            "sigma_xx".into(),
            "sigma_yy".into(),
            "sigma_zz".into(),
            "sigma_yz".into(),
            "sigma_xz".into(),
            "sigma_xy".into(),
        ],
    }
}

/// First invariant `I₁ = tr(σ)`.
///
/// ```
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law;
/// // I₁ = tr(σ) : la part sphérique, trois fois la pression moyenne.
/// assert_eq!(tensor::i1(&[1.0, 2.0, 3.0, 9.0, 9.0, 9.0]), 6.0);
/// ```
/// # use pyrucast::models::tensor;
pub fn i1(sigma: &[f64; 6]) -> f64 {
    sigma[0] + sigma[1] + sigma[2]
}

/// The stress deviator `s = σ − (I₁/3)·I` (same Voigt order).
///
/// ```
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law;
/// // Le déviateur est de trace nulle, et laisse les cisaillements intacts.
/// let s = tensor::deviator(&[3.0, 0.0, 0.0, 0.0, 0.0, 5.0]);
/// assert!(tensor::i1(&s).abs() < 1e-12);
/// assert_eq!(s[5], 5.0);
/// ```
/// # use pyrucast::models::tensor;
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
///
/// ```
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law;
/// // J₂ = ½ s:s, les hors-diagonaux comptés **deux fois**.
/// assert_eq!(tensor::j2(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 1.0);
/// // Insensible à la pression : ajouter une part sphérique ne change rien.
/// let a = tensor::j2(&[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
/// let b = tensor::j2(&[101.0, 99.0, 100.0, 0.0, 0.0, 0.0]);
/// assert!((a - b).abs() < 1e-9);
/// ```
/// # use pyrucast::models::tensor;
pub fn j2(sigma: &[f64; 6]) -> f64 {
    let s = deviator(sigma);
    0.5 * (s[0] * s[0]
        + s[1] * s[1]
        + s[2] * s[2]
        + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
}

/// Third deviatoric invariant `J₃ = det(s)`.
///
/// ```
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law;
/// // J₃ = det(s) — ce qui distingue traction et compression, et fait
/// // l'angle de Lode des critères à quatre paramètres.
/// assert!(tensor::j3(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]).abs() < 1e-12);
/// assert!(tensor::j3(&[2.0, -1.0, -1.0, 0.0, 0.0, 0.0]) > 0.0);
/// ```
/// # use pyrucast::models::tensor;
pub fn j3(sigma: &[f64; 6]) -> f64 {
    let s = deviator(sigma);
    // det of the symmetric tensor [[s0, s5, s4], [s5, s1, s3], [s4, s3, s2]].
    s[0] * (s[1] * s[2] - s[3] * s[3]) - s[5] * (s[5] * s[2] - s[3] * s[4])
        + s[4] * (s[5] * s[3] - s[1] * s[4])
}

/// von Mises equivalent stress `q = √(3 J₂)`.
///
/// ```
/// # use pyrucast::models::tensor;
/// # use pyrucast::models::plasticity::law;
/// // q = √(3 J₂). En traction uniaxiale, q vaut la contrainte appliquée.
/// let q = tensor::von_mises_stress(&[300.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// assert!((q - 300.0).abs() < 1e-9);
/// ```
/// # use pyrucast::models::tensor;
pub fn von_mises_stress(sigma: &[f64; 6]) -> f64 {
    (3.0 * j2(sigma)).sqrt()
}

/// `½(D + Dᵀ)` — exact, and the identity on an already-symmetric matrix.
pub(crate) fn symmetrise(d: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = 0.5 * (d[i][j] + d[j][i]);
        }
    }
    out
}
