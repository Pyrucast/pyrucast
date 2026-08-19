//! What the two beam theories share: the **configuration**, and the exact
//! Timoshenko bending block.
//!
//! ## The configuration is read, not chosen
//!
//! A beam is in one of three configurations — pure bending, plane frame, space
//! frame — and which one is settled by the dimension of the mesh it lives on.
//! It was once an argument, and every constructor rejected any value but the
//! matching one: an argument that can hold exactly one value carries no
//! information, it only offers a way to contradict oneself.
//!
//! Everything that distinguishes the three follows from the dimension, and none
//! of it is a choice:
//!
//! | `Coords` | DOFs per node | gains |
//! |---|---|---|
//! | 1-D | `w`, `theta` | nothing — pure bending |
//! | 2-D | `u_x, u_y, r_z` | the axial term, and a rotation to the global axes |
//! | 3-D | six | the axial, the torsion, and bending about **two** axes |
//!
//! The material and the section forces differ between the *theories*, not the
//! configurations, so each theory declares its own — Bernoulli asks for neither
//! `G` nor `A_s`, and reports no shear force.
//!
//! ## The bending block
//!
//! [`bending_4x4`] is the **exact** Timoshenko element: the closed form of the
//! solution of the two coupled second-order equations on a span free of
//! distributed load. Its shape functions are cubic in the deflection and
//! quadratic in the rotation, and they carry the material through
//!
//! ```text
//! Φ = 12·E·I / (G·A_s·L²)
//! ```
//!
//! — the ratio of bending to shear compliance. `Φ → 0` (a slender member, or an
//! infinite shear stiffness) recovers the Euler-Bernoulli matrix exactly, which
//! is the sense in which Bernoulli is the shear-free limit of this element.
//!
//! That `Φ` is why the basis cannot be tabulated by a finite-element space: it
//! depends on the **material**, not only on the reference element. The space
//! therefore declares
//! [`ModelEmbedded`](crate::atoms::Interpolation::ModelEmbedded) — the
//! formulation owns its interpolation — and this file is where it is owned.

use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};

/// Which configuration a beam is in — the kinematics, not the theory.
///
/// ```
/// # use pyrucast::models::beam::{self, BeamModel};
/// // La configuration se **lit sur la géométrie** : la dimension de
/// // l'espace décide du nombre de DDL par nœud.
/// assert_eq!(BeamModel::Planar1d.dofs_per_node(), 2); // flèche, rotation
/// assert_eq!(BeamModel::Frame2d.dofs_per_node(), 3);
/// assert_eq!(BeamModel::Frame3d.dofs_per_node(), 6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamModel {
    /// Pure bending in a 1-D configuration: deflection and section rotation.
    Planar1d,
    /// Plane frame: axial + bending, rotated to the global axes.
    Frame2d,
    /// Space frame: axial, torsion and bending about two principal axes.
    Frame3d,
}

impl BeamModel {
    /// The configuration, **read from the geometry**.
    ///
    /// ```
    /// # use pyrucast::models::beam::{self, BeamModel};
    /// assert_eq!(BeamModel::from_space_dim(2)?, BeamModel::Frame2d);
    /// assert!(BeamModel::from_space_dim(4).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_space_dim(space_dim: usize) -> Result<Self> {
        match space_dim {
            1 => Ok(Self::Planar1d),
            2 => Ok(Self::Frame2d),
            3 => Ok(Self::Frame3d),
            d => Err(PyrucastError::Message(format!(
                "a beam lives in a 1-, 2- or 3-D configuration, got {d}-D"
            ))),
        }
    }

    /// The lowercase name of the configuration — for messages and rendering.
    ///
    /// ```
    /// # use pyrucast::models::beam::{self, BeamModel};
    /// assert_eq!(BeamModel::Frame3d.to_tag(), "frame_3d");
    /// ```
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Planar1d => "planar_1d",
            Self::Frame2d => "frame_2d",
            Self::Frame3d => "frame_3d",
        }
    }

    /// Primal DOF names, per node.
    ///
    /// ```
    /// # use pyrucast::models::beam::{self, BeamModel};
    /// // Une poutre plane porte flèche et rotation ; un portique spatial y
    /// // ajoute l'axial, la torsion et la seconde flexion.
    /// assert_eq!(BeamModel::Planar1d.primal().len(), 2);
    /// assert_eq!(BeamModel::Frame3d.primal().len(), 6);
    /// ```
    pub fn primal(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["w", "theta"],
            Self::Frame2d => &["u_x", "u_y", "r_z"],
            Self::Frame3d => &["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"],
        }
    }

    /// Dual DOF names, per node.
    ///
    /// ```
    /// # use pyrucast::models::beam::{self, BeamModel};
    /// // Autant de duales que de primales, appariées **par position**.
    /// assert_eq!(BeamModel::Frame2d.dual().len(), BeamModel::Frame2d.primal().len());
    /// ```
    pub fn dual(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["f_w", "m_theta"],
            Self::Frame2d => &["f_x", "f_y", "m_z"],
            Self::Frame3d => &["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"],
        }
    }

    /// Degrees of freedom per node — the side of the element matrix is twice it.
    ///
    /// ```
    /// # use pyrucast::models::beam::{self, BeamModel};
    /// assert_eq!(BeamModel::Frame3d.dofs_per_node(), BeamModel::Frame3d.primal().len());
    /// ```
    pub fn dofs_per_node(self) -> usize {
        self.primal().len()
    }
}

impl std::fmt::Display for BeamModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_tag())
    }
}

/// The exact Timoshenko bending stiffness over `[w_A, θ_A, w_B, θ_B]`.
///
/// `gas` is `G·A_s`; passing `None` drops the shear compliance entirely
/// (`Φ = 0`) and returns the Euler-Bernoulli matrix — used by the beam whose
/// theory has no shear, so that the two share one derivation instead of two.
///
/// ```
/// # use pyrucast::models::beam::{self, BeamModel};
/// // `gas = None` retire la souplesse de cisaillement et redonne la
/// // matrice d'Euler-Bernoulli — une seule dérivation pour les deux
/// // théories. Ses termes classiques : 12EI/L³ et 4EI/L.
/// let k = beam::bending_4x4(2.0, None, 1.0);
/// assert!((k[0][0] - 24.0).abs() < 1e-9);
/// assert!((k[1][1] - 8.0).abs() < 1e-9);
/// // Elle est singulière : une **translation d'ensemble** — les DDL étant
/// // [w_A, θ_A, w_B, θ_B], le mode (1, 0, 1, 0) — n'engendre aucune force.
/// assert!((k[0][0] + k[0][2]).abs() < 1e-9);
/// // Avec cisaillement, elle s'assouplit.
/// assert!(beam::bending_4x4(2.0, Some(1.0), 1.0)[0][0] < k[0][0]);
/// ```
pub fn bending_4x4(ei: f64, gas: Option<f64>, l: f64) -> Vec<Vec<f64>> {
    // `Φ` is the ratio of bending to shear compliance.
    let phi = phi(ei, gas, l);
    let c = ei / (l * l * l * (1.0 + phi));
    let (l2, k1, k2) = (l * l, 12.0 * c, 6.0 * l * c);
    let k3 = (4.0 + phi) * l2 * c;
    let k4 = (2.0 - phi) * l2 * c;
    vec![
        vec![k1, k2, -k1, k2],
        vec![k2, k3, -k2, k4],
        vec![-k1, -k2, k1, -k2],
        vec![k2, k4, -k2, k3],
    ]
}

// ─── The consistent mass ────────────────────────────────────────────────────

/// The **shape functions** of the exact element, at `ξ = x/L ∈ [0, 1]`.
///
/// Returned as `(N_w, N_θ)`: the deflection interpolation and the *independent*
/// rotation interpolation, both over `[w_A, θ_A, w_B, θ_B]`. They are cubic and
/// quadratic respectively, and both carry `Φ`.
///
/// At `Φ = 0` they collapse to the Hermite cubics and to their own derivative —
/// `θ = w'`, which is Euler-Bernoulli. A test asserts exactly that, so the two
/// theories stay one derivation apart here as well.
fn shape_functions(phi: f64, l: f64, xi: f64) -> ([f64; 4], [f64; 4]) {
    let (x2, x3) = (xi * xi, xi * xi * xi);
    let d = 1.0 / (1.0 + phi);
    let n_w = [
        d * (2.0 * x3 - 3.0 * x2 - phi * xi + 1.0 + phi),
        d * l * (x3 - (2.0 + phi / 2.0) * x2 + (1.0 + phi / 2.0) * xi),
        d * (-2.0 * x3 + 3.0 * x2 + phi * xi),
        d * l * (x3 - (1.0 - phi / 2.0) * x2 - (phi / 2.0) * xi),
    ];
    let n_t = [
        d * (6.0 / l) * (x2 - xi),
        d * (3.0 * x2 - (4.0 + phi) * xi + 1.0 + phi),
        d * (6.0 / l) * (xi - x2),
        d * (3.0 * x2 - (2.0 - phi) * xi),
    ];
    (n_w, n_t)
}

/// Four-point Gauss-Legendre on `[0, 1]` — exact to degree 7, where the mass
/// integrand reaches 6.
const GAUSS_01: [(f64, f64); 4] = [
    (0.069_431_844_202_973_71, 0.173_927_422_568_726_9),
    (0.330_009_478_207_571_9, 0.326_072_577_431_273_1),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_1),
    (0.930_568_155_797_026_3, 0.173_927_422_568_726_9),
];

/// The **consistent mass** of the exact beam element, over
/// `[w_A, θ_A, w_B, θ_B]`:
///
/// ```text
/// M = ∫ ρA · N_wᵀ N_w dx  +  ∫ ρI · N_θᵀ N_θ dx
/// ```
///
/// the second term being the **rotary inertia** of the section.
///
/// It is *integrated* from the element's own shape functions rather than
/// transcribed from the published table of `Φ`-polynomials. That table is
/// correct, but a coefficient mistyped out of twenty would produce a plausible,
/// symmetric, positive-definite matrix describing a different beam — the
/// failure mode that cost a wrong tangent earlier in this project. Integrating
/// cannot be mistyped: the shape functions are the same ones the stiffness
/// uses, and four Gauss points make the quadrature exact.
///
/// `gas = None` drops the shear compliance (`Φ = 0`) and yields the classical
/// Euler-Bernoulli consistent mass.
///
/// ```
/// # use pyrucast::models::beam::{self, BeamModel};
/// // Intégrée avec les **mêmes** fonctions de forme que la raideur, à
/// // quatre points de Gauss : la quadrature est exacte et rien n'est
/// // retranscrit à la main.
/// let m = beam::mass_4x4(3.0, 0.0, 1.0, None, 2.0);
/// // La somme des termes de translation redonne la masse totale ρA·L.
/// let masse: f64 = [0usize, 2].iter()
///     .flat_map(|&i| [0usize, 2].iter().map(move |&j| (i, j)))
///     .map(|(i, j)| m[i][j]).sum();
/// assert!((masse - 6.0).abs() < 1e-9);
/// ```
pub fn mass_4x4(rho_a: f64, rho_i: f64, ei: f64, gas: Option<f64>, l: f64) -> Vec<Vec<f64>> {
    let phi = phi(ei, gas, l);
    let mut m = vec![vec![0.0_f64; 4]; 4];
    for (xi, w) in GAUSS_01 {
        let (n_w, n_t) = shape_functions(phi, l, xi);
        let dx = w * l; // dx = L dξ
        for a in 0..4 {
            for b in 0..4 {
                m[a][b] += (rho_a * n_w[a] * n_w[b] + rho_i * n_t[a] * n_t[b]) * dx;
            }
        }
    }
    m
}

// ─── Section strains, from the element's own interpolation ──────────────────

/// The generalised strains of the exact element at `ξ = x/L ∈ [0, 1]`, over
/// `[w_A, θ_A, w_B, θ_B]`: the **curvature** `κ = θ'` and the **shear strain**
/// `γ = w' − θ`.
///
/// Both come from the same shape functions the stiffness and mass use, so the
/// three finally describe one element. And they say something the linear
/// element could not:
///
/// - `κ` is **linear** in `ξ` — the moment varies along an unloaded span, as
///   `M' = V` requires;
/// - `γ` is **constant**, and equals `−Φ/(L(1+Φ))` per unit of the first degree
///   of freedom: an unloaded span carries a constant shear force, which is
///   `V' = 0`.
///
/// Both depend on the material through `Φ`. A recovery that did not take a
/// material could therefore only ever report a mean — which is what the
/// previous one did.
///
/// ```
/// # use pyrucast::models::beam::{self, BeamModel};
/// // Une rotation d'ensemble ne déforme rien : ni courbure, ni distorsion.
/// let (k, g) = beam::section_strains(0.0, 2.0, &[0.0, 0.5, 1.0, 0.5], 0.5);
/// assert!(k.abs() < 1e-12 && g.abs() < 1e-12);
/// // La distorsion γ est **constante** le long de la travée — un effort
/// // tranchant constant sur une travée non chargée.
/// let d = [0.0, 0.0, 1.0, 0.0];
/// let a = beam::section_strains(0.5, 2.0, &d, 0.1).1;
/// let b = beam::section_strains(0.5, 2.0, &d, 0.9).1;
/// assert!((a - b).abs() < 1e-12);
/// // La courbure, elle, varie : c'est ce que `M' = V` exige.
/// assert!((beam::section_strains(0.5, 2.0, &d, 0.1).0
///          - beam::section_strains(0.5, 2.0, &d, 0.9).0).abs() > 1e-9);
/// ```
pub fn section_strains(phi: f64, l: f64, d: &[f64; 4], xi: f64) -> (f64, f64) {
    let dd = 1.0 / (1.0 + phi);
    // ∂N_θ/∂ξ — the curvature is its physical derivative, `κ = (1/L)·∂N_θ/∂ξ·d`.
    let dn_t = [
        dd * (6.0 / l) * (2.0 * xi - 1.0),
        dd * (6.0 * xi - (4.0 + phi)),
        dd * (6.0 / l) * (1.0 - 2.0 * xi),
        dd * (6.0 * xi - (2.0 - phi)),
    ];
    // ∂N_w/∂ξ, for `γ = (1/L)·∂N_w/∂ξ·d − N_θ·d`.
    let (x2, _) = (xi * xi, ());
    let dn_w = [
        dd * (6.0 * x2 - 6.0 * xi - phi),
        dd * l * (3.0 * x2 - (4.0 + phi) * xi + 1.0 + phi / 2.0),
        dd * (-6.0 * x2 + 6.0 * xi + phi),
        dd * l * (3.0 * x2 - (2.0 - phi) * xi - phi / 2.0),
    ];
    let (_, n_t) = shape_functions(phi, l, xi);
    let kappa: f64 = (0..4).map(|i| dn_t[i] * d[i]).sum::<f64>() / l;
    let gamma: f64 = (0..4).map(|i| (dn_w[i] / l - n_t[i]) * d[i]).sum();
    (kappa, gamma)
}

/// `Φ = 12EI/(G·A_s·L²)`, the ratio the whole element hangs on. `gas = None`
/// means no shear compliance.
///
/// ```
/// # use pyrucast::models::beam::{self, BeamModel};
/// // Φ = 12EI/(G·A_s·L²), le rapport dont dépend tout l'élément.
/// assert_eq!(beam::phi(1.0, None, 2.0), 0.0); // pas de souplesse de cisaillement
/// assert!((beam::phi(1.0, Some(3.0), 2.0) - 1.0).abs() < 1e-12);
/// // Il s'efface quand la poutre s'allonge : la théorie mince est la limite.
/// assert!(beam::phi(1.0, Some(3.0), 100.0) < 1e-3);
/// ```
pub fn phi(ei: f64, gas: Option<f64>, l: f64) -> f64 {
    match gas {
        Some(g) if g.abs() > f64::MIN_POSITIVE => 12.0 * ei / (g * l * l),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Φ = 0` is Euler-Bernoulli, term for term. The two theories then share
    /// one derivation, and the shear-free limit is not a separate claim to
    /// maintain.
    #[test]
    fn no_shear_compliance_recovers_euler_bernoulli() {
        let (ei, l) = (2.5, 1.7);
        let k = bending_4x4(ei, None, l);
        let c = ei / (l * l * l);
        let want = [
            [12.0 * c, 6.0 * l * c, -12.0 * c, 6.0 * l * c],
            [6.0 * l * c, 4.0 * l * l * c, -6.0 * l * c, 2.0 * l * l * c],
            [-12.0 * c, -6.0 * l * c, 12.0 * c, -6.0 * l * c],
            [6.0 * l * c, 2.0 * l * l * c, -6.0 * l * c, 4.0 * l * l * c],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (k[i][j] - want[i][j]).abs() < 1e-12,
                    "[{i}][{j}]: {} vs {}",
                    k[i][j],
                    want[i][j]
                );
            }
        }
    }

    /// A huge shear stiffness converges to the same limit — the continuity that
    /// makes "Bernoulli is the shear-free Timoshenko" true in arithmetic and
    /// not only in words.
    #[test]
    fn a_stiff_shear_converges_to_the_shear_free_matrix() {
        let (ei, l) = (2.5, 1.7);
        let free = bending_4x4(ei, None, l);
        let stiff = bending_4x4(ei, Some(1e12), l);
        for i in 0..4 {
            for j in 0..4 {
                assert!((stiff[i][j] - free[i][j]).abs() < 1e-6 * free[0][0].abs());
            }
        }
    }

    /// Shear compliance **softens** the element: every diagonal bending term
    /// drops. A sign slip in `Φ` would stiffen it instead, and no closed-form
    /// comparison at a single `Φ` would notice.
    #[test]
    fn shear_compliance_softens_the_element() {
        let (ei, l) = (2.5, 1.7);
        let free = bending_4x4(ei, None, l);
        let soft = bending_4x4(ei, Some(0.5), l);
        assert!(soft[0][0] < free[0][0], "the deflection term should soften");
        assert!(soft[1][1] < free[1][1], "the rotation term should soften");
    }

    /// The two rigid-body modes cost no energy: a translation `[1,0,1,0]` and a
    /// rotation `[−L/2, 1, L/2, 1]`.
    #[test]
    fn rigid_body_modes_are_free() {
        let (ei, l, gas) = (2.5, 1.7, 3.0);
        let k = bending_4x4(ei, Some(gas), l);
        for mode in [[1.0, 0.0, 1.0, 0.0], [-l / 2.0, 1.0, l / 2.0, 1.0]] {
            for row in k.iter() {
                let f: f64 = (0..4).map(|j| row[j] * mode[j]).sum();
                assert!(f.abs() < 1e-9 * k[0][0].abs(), "rigid mode carries {f}");
            }
        }
    }
    /// At `Φ = 0` the shape functions are the Hermite cubics, and the rotation
    /// interpolation is their own derivative — `θ = w'`, which *is*
    /// Euler-Bernoulli. The two theories stay one derivation apart.
    #[test]
    fn without_shear_the_rotation_is_the_slope() {
        let l = 1.7;
        for k in 0..=10 {
            let xi = k as f64 / 10.0;
            let (n_w, n_t) = shape_functions(0.0, l, xi);
            // d(N_w)/dx by a central difference, against N_θ.
            let h = 1e-6;
            let (plus, _) = shape_functions(0.0, l, xi + h);
            let (minus, _) = shape_functions(0.0, l, xi - h);
            for i in 0..4 {
                let dn_dx = (plus[i] - minus[i]) / (2.0 * h * l);
                assert!(
                    (dn_dx - n_t[i]).abs() < 1e-6,
                    "N{}'({xi}) = {dn_dx} vs Nθ = {}",
                    i + 1,
                    n_t[i]
                );
            }
            let _ = n_w;
        }
    }

    /// The shear-free consistent mass is the classical Euler-Bernoulli table,
    /// `ρAL/420 · [156, 22L, 54, −13L; …]` — twelve numbers nobody disputes,
    /// and an oracle this integration owes nothing to.
    #[test]
    fn without_shear_the_mass_is_the_classical_table() {
        let (rho_a, l) = (2.5, 1.7);
        let m = mass_4x4(rho_a, 0.0, 1.0, None, l);
        let c = rho_a * l / 420.0;
        let (l2, ll) = (l * l, l);
        let want = [
            [156.0 * c, 22.0 * ll * c, 54.0 * c, -13.0 * ll * c],
            [22.0 * ll * c, 4.0 * l2 * c, 13.0 * ll * c, -3.0 * l2 * c],
            [54.0 * c, 13.0 * ll * c, 156.0 * c, -22.0 * ll * c],
            [-13.0 * ll * c, -3.0 * l2 * c, -22.0 * ll * c, 4.0 * l2 * c],
        ];
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (m[i][j] - want[i][j]).abs() < 1e-10 * want[0][0],
                    "[{i}][{j}]: {} vs {}",
                    m[i][j],
                    want[i][j]
                );
            }
        }
    }

    /// Whatever `Φ`, a **rigid translation** carries exactly the mass of the
    /// member, `ρA·L`. This is the one property no interpolation may break, and
    /// it is what a mistyped coefficient would show up in.
    #[test]
    fn a_rigid_translation_carries_the_whole_mass() {
        let (rho_a, rho_i, ei, l) = (2.5, 0.4, 3.0, 1.7);
        for gas in [None, Some(1e-2), Some(1.0), Some(1e6)] {
            let m = mass_4x4(rho_a, rho_i, ei, gas, l);
            let unit = [1.0, 0.0, 1.0, 0.0]; // w = 1 everywhere
            let total: f64 = (0..4)
                .map(|a| (0..4).map(|b| unit[a] * m[a][b] * unit[b]).sum::<f64>())
                .sum();
            assert!(
                (total - rho_a * l).abs() < 1e-10 * rho_a * l,
                "gas={gas:?}: {total} vs {}",
                rho_a * l
            );
        }
    }

    /// The mass is symmetric and positive definite — checked through its
    /// diagonal and its quadratic form on a spread of vectors, since a mass that
    /// were not would let an eigenvalue solver return an imaginary frequency.
    #[test]
    fn the_mass_is_symmetric_and_positive() {
        let m = mass_4x4(2.5, 0.4, 3.0, Some(1.0), 1.7);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (m[i][j] - m[j][i]).abs() < 1e-12,
                    "asymmetric at [{i}][{j}]"
                );
            }
            assert!(m[i][i] > 0.0, "non-positive diagonal at {i}");
        }
        for v in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, -1.0],
            [0.3, 0.7, -0.2, 0.9],
        ] {
            let q: f64 = (0..4)
                .map(|a| (0..4).map(|b| v[a] * m[a][b] * v[b]).sum::<f64>())
                .sum();
            assert!(q > 0.0, "non-positive quadratic form: {q}");
        }
    }

    /// Shear compliance **redistributes** mass without creating any: the total
    /// is invariant (above), while the rotation terms change. A mass that
    /// ignored `Φ` would be identical for every shear stiffness.
    #[test]
    fn shear_compliance_changes_the_rotational_terms() {
        let (rho_a, rho_i, ei, l) = (2.5, 0.4, 3.0, 1.7);
        let stiff = mass_4x4(rho_a, rho_i, ei, Some(1e6), l);
        let soft = mass_4x4(rho_a, rho_i, ei, Some(0.5), l);
        assert!(
            (stiff[1][1] - soft[1][1]).abs() > 1e-3 * stiff[1][1],
            "Φ leaves the rotational mass untouched: {} vs {}",
            stiff[1][1],
            soft[1][1]
        );
    }

    /// The shear strain is **constant** along the element and the curvature
    /// **linear** — `V' = 0` and `M' = V` on an unloaded span. The linear
    /// element could report neither: its curvature was constant and its shear
    /// oscillated, which is why the old recovery had to average.
    #[test]
    fn the_shear_is_constant_and_the_curvature_linear() {
        let (l, ph) = (1.7, 0.8);
        let d = [0.0, 0.0, 0.5, 0.2]; // a clamped-free state
        let sample: Vec<(f64, f64)> = (0..=10)
            .map(|k| section_strains(ph, l, &d, k as f64 / 10.0))
            .collect();
        // γ constant.
        for (_, g) in &sample {
            assert!(
                (g - sample[0].1).abs() < 1e-12,
                "shear varies: {g} vs {}",
                sample[0].1
            );
        }
        // κ linear: its second difference vanishes.
        for w in sample.windows(3) {
            let d2 = w[0].0 - 2.0 * w[1].0 + w[2].0;
            assert!(d2.abs() < 1e-12, "curvature is not linear: {d2}");
        }
        // …and it genuinely varies, so "linear" is not "constant".
        assert!((sample[0].0 - sample[10].0).abs() > 1e-6);
    }

    /// Without shear compliance the curvature is the second derivative of the
    /// deflection, and the shear strain vanishes — Euler-Bernoulli, once more
    /// as the `Φ = 0` limit rather than as a separate derivation.
    #[test]
    fn without_shear_the_strain_is_bernoullis() {
        let l = 1.7;
        let d = [0.1, -0.2, 0.5, 0.3];
        for k in 0..=10 {
            let xi = k as f64 / 10.0;
            let (kappa, gamma) = section_strains(0.0, l, &d, xi);
            assert!(gamma.abs() < 1e-12, "shear-free γ = {gamma}");
            // κ against a central difference of the deflection interpolation.
            let h = 1e-4;
            let w_of = |x: f64| -> f64 {
                let (n_w, _) = shape_functions(0.0, l, x);
                (0..4).map(|i| n_w[i] * d[i]).sum()
            };
            let fd = (w_of(xi + h) - 2.0 * w_of(xi) + w_of(xi - h)) / (h * h * l * l);
            assert!((kappa - fd).abs() < 1e-4, "κ({xi}) = {kappa} vs w'' = {fd}");
        }
    }
}
