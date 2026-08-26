//! Tensor algebra on **Voigt** vectors — invariants and deviator.
//!
//! Nothing here knows about plasticity: these are the quantities any physics
//! reading a stress or a strain may need. Damage already recomputes some of
//! them by hand (`damage/damage_tc.rs`), which is the reason they live here
//! rather than inside one physics.
//!
//! Voigt order is `[xx, yy, zz, xy, yz, zx]`, with **engineering** shear.

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
