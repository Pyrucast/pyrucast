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
    pub fn to_tag(self) -> &'static str {
        match self {
            Self::Planar1d => "planar_1d",
            Self::Frame2d => "frame_2d",
            Self::Frame3d => "frame_3d",
        }
    }

    /// Primal DOF names, per node.
    pub fn primal(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["w", "theta"],
            Self::Frame2d => &["u_x", "u_y", "r_z"],
            Self::Frame3d => &["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"],
        }
    }

    /// Dual DOF names, per node.
    pub fn dual(self) -> &'static [&'static str] {
        match self {
            Self::Planar1d => &["f_w", "m_theta"],
            Self::Frame2d => &["f_x", "f_y", "m_z"],
            Self::Frame3d => &["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"],
        }
    }

    /// Degrees of freedom per node — the side of the element matrix is twice it.
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
pub fn bending_4x4(ei: f64, gas: Option<f64>, l: f64) -> Vec<Vec<f64>> {
    let phi = match gas {
        // `Φ` is the ratio of bending to shear compliance. A vanishing `G·A_s`
        // would mean a member with no shear stiffness at all, which is not a
        // beam; guarding keeps the division meaningful rather than infinite.
        Some(g) if g.abs() > f64::MIN_POSITIVE => 12.0 * ei / (g * l * l),
        _ => 0.0,
    };
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
}
