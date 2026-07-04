//! Finite-element interpolations on the reference element.
//!
//! An [`Interpolation`] is the **mathematical recipe** that builds the
//! shape functions `N_i(ξ)` and their reference derivatives
//! `∂N_i/∂ξ_j` for a given [`ElementType`]. It is independent of any
//! particular cell or coordinate set: every evaluation lives in the
//! reference frame of the element type (see [`crate::containers::mesh::element_type`]).
//!
//! Adding a new interpolation means:
//! - adding a variant to [`Interpolation`];
//! - extending [`Interpolation::is_compatible_with`], [`Interpolation::shape`]
//!   and [`Interpolation::dshape_dxi`] for every supported `ElementType`.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::Interpolation;
//!
//! // Lagrange-1 shape functions of a TRI3 at the centroid (1/3, 1/3).
//! let n = Interpolation::Lagrange1
//!     .shape(ElementType::TRI3, &[1.0 / 3.0, 1.0 / 3.0])
//!     .unwrap();
//! assert_eq!(n.len(), 3);
//! let s: f64 = n.iter().sum();
//! assert!((s - 1.0).abs() < 1e-12);  // partition of unity
//! ```

use crate::containers::mesh::ElementType;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported finite-element interpolations.
///
/// Names mix cast3m and standard FE terminology. See the module-level
/// documentation for the conventions on reference frames and node order.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    /// Linear Lagrange (P1 / Q1). One shape function per geometric node,
    /// equal to 1 at "its" node and 0 at the others. Defined for the linear
    /// element types (`SEG2`, `TRI3`, `QUA4`, `TET4`, `PENTA6`, `HEX8`).
    Lagrange1,
    /// Quadratic Lagrange (P2 / Q2 serendipity). One shape function per
    /// geometric node — corners **and** mid-edge nodes. Defined for the
    /// quadratic element types (`SEG3`, `TRI6`, `QUA8`, `TET10`, `PENTA15`,
    /// `HEX20`). `QUA8`/`HEX20`/`PENTA15` are serendipity (edge nodes only).
    Lagrange2,
}

impl Interpolation {
    /// Whether this interpolation is defined for `element_type`.
    ///
    /// The interpolation **degree** must match the element type: `Lagrange1`
    /// for the linear types, `Lagrange2` for the quadratic (mid-edge) types.
    /// `POI1` has no reference frame and is always rejected.
    pub fn is_compatible_with(self, element_type: ElementType) -> bool {
        use ElementType::*;
        match self {
            Self::Lagrange1 => matches!(element_type, SEG2 | TRI3 | QUA4 | TET4 | PENTA6 | HEX8),
            Self::Lagrange2 => {
                matches!(
                    element_type,
                    SEG3 | TRI6 | QUA8 | QUA9 | TET10 | PENTA15 | HEX20
                )
            }
        }
    }

    /// Short name (cast3m-style).
    pub fn name(self) -> &'static str {
        match self {
            Self::Lagrange1 => "LAGRANGE1",
            Self::Lagrange2 => "LAGRANGE2",
        }
    }

    /// Parse from a short name (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "LAGRANGE1" | "LAG1" => Some(Self::Lagrange1),
            "LAGRANGE2" | "LAG2" => Some(Self::Lagrange2),
            _ => None,
        }
    }

    /// Evaluate the shape functions `N_i(ξ)` at the reference point `xi`.
    ///
    /// Returns a flat `Vec<f64>` of length `element_type.nodes_per_cell()`
    /// ordered like the cell's nodes. `xi` must have length
    /// `element_type.topological_dim()`.
    ///
    /// # Errors
    ///
    /// - `xi` has the wrong length;
    /// - the `(self, element_type)` pair is not supported (`POI1`, …).
    pub fn shape(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check_compat(element_type)?;
        check_xi_len(element_type, xi)?;
        match (self, element_type) {
            (Self::Lagrange1, ElementType::SEG2) => {
                let x = xi[0];
                Ok(vec![0.5 * (1.0 - x), 0.5 * (1.0 + x)])
            }
            (Self::Lagrange1, ElementType::TRI3) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![1.0 - a - b, a, b])
            }
            (Self::Lagrange1, ElementType::QUA4) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![
                    0.25 * (1.0 - a) * (1.0 - b),
                    0.25 * (1.0 + a) * (1.0 - b),
                    0.25 * (1.0 + a) * (1.0 + b),
                    0.25 * (1.0 - a) * (1.0 + b),
                ])
            }
            (Self::Lagrange1, ElementType::TET4) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                Ok(vec![1.0 - a - b - c, a, b, c])
            }
            (Self::Lagrange1, ElementType::HEX8) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let mut n = Vec::with_capacity(8);
                for &(xi_i, eta_i, zeta_i) in HEX8_REF_NODES.iter() {
                    n.push(0.125 * (1.0 + xi_i * a) * (1.0 + eta_i * b) * (1.0 + zeta_i * c));
                }
                Ok(n)
            }
            (Self::Lagrange1, ElementType::PENTA6) => {
                // Prism = TRI3 (barycentric ξ, η) ⊗ SEG2 (linear ζ ∈ [0, 1]).
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let (l1, l2, l3) = (1.0 - a - b, a, b);
                Ok(vec![
                    l1 * (1.0 - c), // node 0
                    l2 * (1.0 - c), // node 1
                    l3 * (1.0 - c), // node 2
                    l1 * c,         // node 3
                    l2 * c,         // node 4
                    l3 * c,         // node 5
                ])
            }
            (Self::Lagrange2, ElementType::SEG3) => Ok(seg3_shape(xi)),
            (Self::Lagrange2, ElementType::TRI6) => Ok(tri6_shape(xi)),
            (Self::Lagrange2, ElementType::QUA8) => Ok(qua8_shape(xi)),
            (Self::Lagrange2, ElementType::QUA9) => Ok(qua9_shape(xi)),
            (Self::Lagrange2, ElementType::TET10) => Ok(tet10_shape(xi)),
            (Self::Lagrange2, ElementType::PENTA15) => Ok(penta15_shape(xi)),
            (Self::Lagrange2, ElementType::HEX20) => Ok(hex20_shape(xi)),
            // Every other pair is incompatible and ruled out by check_compat.
            _ => unreachable!("incompatible (interpolation, element_type) reached shape()"),
        }
    }

    /// Evaluate the reference derivatives `∂N_i/∂ξ_j` at the reference
    /// point `xi`.
    ///
    /// Returns a flat row-major buffer of length
    /// `nodes_per_cell × topological_dim`, where entry
    /// `[i * topological_dim + j]` is `∂N_i/∂ξ_j`.
    ///
    /// # Errors
    ///
    /// - `xi` has the wrong length;
    /// - the `(self, element_type)` pair is not supported.
    pub fn dshape_dxi(self, element_type: ElementType, xi: &[f64]) -> Result<Vec<f64>> {
        self.check_compat(element_type)?;
        check_xi_len(element_type, xi)?;
        match (self, element_type) {
            (Self::Lagrange1, ElementType::SEG2) => Ok(vec![-0.5, 0.5]),
            (Self::Lagrange1, ElementType::TRI3) => Ok(vec![
                -1.0, -1.0, // dN1
                1.0, 0.0, // dN2
                0.0, 1.0, // dN3
            ]),
            (Self::Lagrange1, ElementType::QUA4) => {
                let (a, b) = (xi[0], xi[1]);
                Ok(vec![
                    -0.25 * (1.0 - b),
                    -0.25 * (1.0 - a), // dN1
                    0.25 * (1.0 - b),
                    -0.25 * (1.0 + a), // dN2
                    0.25 * (1.0 + b),
                    0.25 * (1.0 + a), // dN3
                    -0.25 * (1.0 + b),
                    0.25 * (1.0 - a), // dN4
                ])
            }
            (Self::Lagrange1, ElementType::TET4) => Ok(vec![
                -1.0, -1.0, -1.0, // dN1
                1.0, 0.0, 0.0, // dN2
                0.0, 1.0, 0.0, // dN3
                0.0, 0.0, 1.0, // dN4
            ]),
            (Self::Lagrange1, ElementType::HEX8) => {
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let mut out = Vec::with_capacity(8 * 3);
                for &(xi_i, eta_i, zeta_i) in HEX8_REF_NODES.iter() {
                    let one_a = 1.0 + xi_i * a;
                    let one_b = 1.0 + eta_i * b;
                    let one_c = 1.0 + zeta_i * c;
                    out.push(0.125 * xi_i * one_b * one_c);
                    out.push(0.125 * eta_i * one_a * one_c);
                    out.push(0.125 * zeta_i * one_a * one_b);
                }
                Ok(out)
            }
            (Self::Lagrange1, ElementType::PENTA6) => {
                // ∂N_i/∂(ξ, η, ζ) with L1 = 1-ξ-η, L2 = ξ, L3 = η and the
                // ζ ∈ [0, 1] linear factor.
                let (a, b, c) = (xi[0], xi[1], xi[2]);
                let (l1, l2, l3) = (1.0 - a - b, a, b);
                Ok(vec![
                    -(1.0 - c),
                    -(1.0 - c),
                    -l1, // dN0
                    1.0 - c,
                    0.0,
                    -l2, // dN1
                    0.0,
                    1.0 - c,
                    -l3, // dN2
                    -c,
                    -c,
                    l1, // dN3
                    c,
                    0.0,
                    l2, // dN4
                    0.0,
                    c,
                    l3, // dN5
                ])
            }
            (Self::Lagrange2, ElementType::SEG3) => Ok(seg3_dshape(xi)),
            (Self::Lagrange2, ElementType::TRI6) => Ok(tri6_dshape(xi)),
            (Self::Lagrange2, ElementType::QUA8) => Ok(qua8_dshape(xi)),
            (Self::Lagrange2, ElementType::QUA9) => Ok(qua9_dshape(xi)),
            (Self::Lagrange2, ElementType::TET10) => Ok(tet10_dshape(xi)),
            (Self::Lagrange2, ElementType::PENTA15) => Ok(penta15_dshape(xi)),
            (Self::Lagrange2, ElementType::HEX20) => Ok(hex20_dshape(xi)),
            _ => unreachable!("incompatible (interpolation, element_type) reached dshape_dxi()"),
        }
    }

    fn check_compat(self, element_type: ElementType) -> Result<()> {
        if !self.is_compatible_with(element_type) {
            return Err(PyrucastError::Message(format!(
                "interpolation {} is not defined for {}",
                self, element_type
            )));
        }
        Ok(())
    }
}

fn check_xi_len(element_type: ElementType, xi: &[f64]) -> Result<()> {
    let expected = element_type.topological_dim();
    if xi.len() != expected {
        return Err(PyrucastError::Message(format!(
            "xi has length {}, expected {} for {}",
            xi.len(),
            expected,
            element_type
        )));
    }
    Ok(())
}

/// Reference coordinates of the 8 HEX8 nodes, in local order
/// (bottom face CCW then top face CCW).
const HEX8_REF_NODES: [(f64, f64, f64); 8] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
];

// ─── Lagrange-2 (quadratic) shape functions and reference derivatives ────────
//
// Each `*_shape` returns the `n_nodes` values `N_i(ξ)`; each `*_dshape`
// returns the row-major buffer `dN[i * d_r + k] = ∂N_i/∂ξ_k`. Node order
// follows [`crate::containers::mesh::element_type`]: corners first (same as
// the linear parent), then mid-edge nodes in the documented edge order.

/// SEG3 on `ξ ∈ [-1, 1]`: corners at `∓1`, mid node at `0`.
fn seg3_shape(xi: &[f64]) -> Vec<f64> {
    let x = xi[0];
    vec![0.5 * x * (x - 1.0), 0.5 * x * (x + 1.0), 1.0 - x * x]
}

fn seg3_dshape(xi: &[f64]) -> Vec<f64> {
    let x = xi[0];
    vec![x - 0.5, x + 0.5, -2.0 * x]
}

/// TRI6, complete quadratic on the unit triangle. `L1 = 1-ξ-η`, `L2 = ξ`,
/// `L3 = η`; corners `L_i(2L_i-1)`, mid-edges `4 L_a L_b`.
fn tri6_shape(xi: &[f64]) -> Vec<f64> {
    let (a, b) = (xi[0], xi[1]);
    let (l1, l2, l3) = (1.0 - a - b, a, b);
    vec![
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
    ]
}

fn tri6_dshape(xi: &[f64]) -> Vec<f64> {
    let (a, b) = (xi[0], xi[1]);
    let (l1, l2, l3) = (1.0 - a - b, a, b);
    // dL1 = (-1,-1), dL2 = (1,0), dL3 = (0,1).
    vec![
        -(4.0 * l1 - 1.0),
        -(4.0 * l1 - 1.0), // dN0
        4.0 * l2 - 1.0,
        0.0, // dN1
        0.0,
        4.0 * l3 - 1.0, // dN2
        4.0 * (l1 - l2),
        -4.0 * l2, // dN3 = 4(L2 dL1 + L1 dL2)
        4.0 * l3,
        4.0 * l2, // dN4 = 4(L3 dL2 + L2 dL3)
        -4.0 * l3,
        4.0 * (l1 - l3), // dN5 = 4(L1 dL3 + L3 dL1)
    ]
}

/// QUA8 serendipity on `[-1, 1]²`.
fn qua8_shape(xi: &[f64]) -> Vec<f64> {
    let (a, b) = (xi[0], xi[1]);
    let corner = |xi_i: f64, eta_i: f64| {
        0.25 * (1.0 + xi_i * a) * (1.0 + eta_i * b) * (xi_i * a + eta_i * b - 1.0)
    };
    vec![
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
        0.5 * (1.0 - a * a) * (1.0 - b), // 4: (0,-1)
        0.5 * (1.0 + a) * (1.0 - b * b), // 5: (1,0)
        0.5 * (1.0 - a * a) * (1.0 + b), // 6: (0,1)
        0.5 * (1.0 - a) * (1.0 - b * b), // 7: (-1,0)
    ]
}

fn qua8_dshape(xi: &[f64]) -> Vec<f64> {
    let (a, b) = (xi[0], xi[1]);
    let mut out = Vec::with_capacity(16);
    for &(xi_i, eta_i) in &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let v = 1.0 + eta_i * b;
        let u = 1.0 + xi_i * a;
        out.push(0.25 * xi_i * v * (2.0 * xi_i * a + eta_i * b));
        out.push(0.25 * eta_i * u * (2.0 * eta_i * b + xi_i * a));
    }
    // Mid-edge nodes 4..7.
    out.extend_from_slice(&[
        -a * (1.0 - b),
        -0.5 * (1.0 - a * a), // 4
        0.5 * (1.0 - b * b),
        -b * (1.0 + a), // 5
        -a * (1.0 + b),
        0.5 * (1.0 - a * a), // 6
        -0.5 * (1.0 - b * b),
        -b * (1.0 - a), // 7
    ]);
    out
}

/// (ξ-position, η-position) of each QUA9 node, encoded as an index into the
/// 1-D quadratic Lagrange basis `[node@-1, node@0, node@+1]`. Corners 0..3,
/// mid-edges 4..7, center 8.
const QUA9_NODES: [(usize, usize); 9] = [
    (0, 0),
    (2, 0),
    (2, 2),
    (0, 2), // corners
    (1, 0),
    (2, 1),
    (1, 2),
    (0, 1), // mid-edges
    (1, 1), // center
];

/// 1-D quadratic Lagrange basis on `[-1, 1]` at `t`, nodes `-1, 0, +1`.
fn lag1d(t: f64) -> [f64; 3] {
    [0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)]
}

/// Its derivative.
fn dlag1d(t: f64) -> [f64; 3] {
    [t - 0.5, -2.0 * t, t + 0.5]
}

/// QUA9 biquadratic on `[-1, 1]²`: the full `Q2` tensor product (corners,
/// mid-edges, and a center node).
fn qua9_shape(xi: &[f64]) -> Vec<f64> {
    let (lx, ly) = (lag1d(xi[0]), lag1d(xi[1]));
    QUA9_NODES.iter().map(|&(i, j)| lx[i] * ly[j]).collect()
}

fn qua9_dshape(xi: &[f64]) -> Vec<f64> {
    let (lx, ly) = (lag1d(xi[0]), lag1d(xi[1]));
    let (dlx, dly) = (dlag1d(xi[0]), dlag1d(xi[1]));
    let mut out = Vec::with_capacity(18);
    for &(i, j) in QUA9_NODES.iter() {
        out.push(dlx[i] * ly[j]);
        out.push(lx[i] * dly[j]);
    }
    out
}

/// TET10, complete quadratic on the unit tetrahedron. `L0 = 1-ξ-η-ζ`,
/// `L1 = ξ`, `L2 = η`, `L3 = ζ`.
fn tet10_shape(xi: &[f64]) -> Vec<f64> {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    let (l0, l1, l2, l3) = (1.0 - a - b - c, a, b, c);
    vec![
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l0 * l1, // 4: (0,1)
        4.0 * l1 * l2, // 5: (1,2)
        4.0 * l2 * l0, // 6: (2,0)
        4.0 * l0 * l3, // 7: (0,3)
        4.0 * l1 * l3, // 8: (1,3)
        4.0 * l2 * l3, // 9: (2,3)
    ]
}

fn tet10_dshape(xi: &[f64]) -> Vec<f64> {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    let (l0, l1, l2, l3) = (1.0 - a - b - c, a, b, c);
    // Gradients of the barycentric coordinates (each a length-3 row).
    let d = [
        [-1.0, -1.0, -1.0], // dL0
        [1.0, 0.0, 0.0],    // dL1
        [0.0, 1.0, 0.0],    // dL2
        [0.0, 0.0, 1.0],    // dL3
    ];
    let corner = |li: f64, dl: [f64; 3]| dl.map(|g| (4.0 * li - 1.0) * g);
    // 4 * (L_b dL_a + L_a dL_b)
    let edge = |la: f64, lb: f64, da: [f64; 3], db: [f64; 3]| {
        [0, 1, 2].map(|k| 4.0 * (lb * da[k] + la * db[k]))
    };
    let rows = [
        corner(l0, d[0]),
        corner(l1, d[1]),
        corner(l2, d[2]),
        corner(l3, d[3]),
        edge(l0, l1, d[0], d[1]),
        edge(l1, l2, d[1], d[2]),
        edge(l2, l0, d[2], d[0]),
        edge(l0, l3, d[0], d[3]),
        edge(l1, l3, d[1], d[3]),
        edge(l2, l3, d[2], d[3]),
    ];
    rows.iter().flatten().copied().collect()
}

/// Reference coordinates of the 20 HEX20 nodes: 8 corners (as HEX8) then
/// 12 mid-edge nodes (exactly one zero coordinate each).
const HEX20_REF_NODES: [(f64, f64, f64); 20] = [
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (0.0, -1.0, -1.0), // 8: (0,1)
    (1.0, 0.0, -1.0),  // 9: (1,2)
    (0.0, 1.0, -1.0),  // 10: (2,3)
    (-1.0, 0.0, -1.0), // 11: (3,0)
    (0.0, -1.0, 1.0),  // 12: (4,5)
    (1.0, 0.0, 1.0),   // 13: (5,6)
    (0.0, 1.0, 1.0),   // 14: (6,7)
    (-1.0, 0.0, 1.0),  // 15: (7,4)
    (-1.0, -1.0, 0.0), // 16: (0,4)
    (1.0, -1.0, 0.0),  // 17: (1,5)
    (1.0, 1.0, 0.0),   // 18: (2,6)
    (-1.0, 1.0, 0.0),  // 19: (3,7)
];

/// HEX20 serendipity on `[-1, 1]³`.
fn hex20_shape(xi: &[f64]) -> Vec<f64> {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    HEX20_REF_NODES
        .iter()
        .map(|&(p, q, r)| {
            let zeros = [p, q, r].iter().filter(|&&x| x == 0.0).count();
            if zeros == 0 {
                // Corner.
                0.125
                    * (1.0 + p * a)
                    * (1.0 + q * b)
                    * (1.0 + r * c)
                    * (p * a + q * b + r * c - 2.0)
            } else if p == 0.0 {
                0.25 * (1.0 - a * a) * (1.0 + q * b) * (1.0 + r * c)
            } else if q == 0.0 {
                0.25 * (1.0 + p * a) * (1.0 - b * b) * (1.0 + r * c)
            } else {
                0.25 * (1.0 + p * a) * (1.0 + q * b) * (1.0 - c * c)
            }
        })
        .collect()
}

fn hex20_dshape(xi: &[f64]) -> Vec<f64> {
    let (a, b, c) = (xi[0], xi[1], xi[2]);
    let mut out = Vec::with_capacity(60);
    for &(p, q, r) in HEX20_REF_NODES.iter() {
        let zeros = [p, q, r].iter().filter(|&&x| x == 0.0).count();
        if zeros == 0 {
            let (u, v, w) = (1.0 + p * a, 1.0 + q * b, 1.0 + r * c);
            out.push(0.125 * p * v * w * (2.0 * p * a + q * b + r * c - 1.0));
            out.push(0.125 * q * u * w * (p * a + 2.0 * q * b + r * c - 1.0));
            out.push(0.125 * r * u * v * (p * a + q * b + 2.0 * r * c - 1.0));
        } else if p == 0.0 {
            let (vy, wz) = (1.0 + q * b, 1.0 + r * c);
            out.push(-0.5 * a * vy * wz);
            out.push(0.25 * (1.0 - a * a) * q * wz);
            out.push(0.25 * (1.0 - a * a) * vy * r);
        } else if q == 0.0 {
            let (ux, wz) = (1.0 + p * a, 1.0 + r * c);
            out.push(0.25 * p * (1.0 - b * b) * wz);
            out.push(-0.5 * b * ux * wz);
            out.push(0.25 * ux * (1.0 - b * b) * r);
        } else {
            let (ux, vy) = (1.0 + p * a, 1.0 + q * b);
            out.push(0.25 * p * vy * (1.0 - c * c));
            out.push(0.25 * ux * q * (1.0 - c * c));
            out.push(-0.5 * c * ux * vy);
        }
    }
    out
}

/// PENTA15 serendipity prism, triangle `(L1,L2,L3)` × `ζ ∈ [0, 1]`.
fn penta15_shape(xi: &[f64]) -> Vec<f64> {
    let (a, b, t) = (xi[0], xi[1], xi[2]);
    let (l1, l2, l3) = (1.0 - a - b, a, b);
    let bot = |li: f64| li * (2.0 * li - 1.0) * (1.0 - t) - 2.0 * li * t * (1.0 - t);
    let top = |li: f64| li * (2.0 * li - 1.0) * t - 2.0 * li * t * (1.0 - t);
    vec![
        bot(l1),
        bot(l2),
        bot(l3),
        top(l1),
        top(l2),
        top(l3),
        4.0 * l1 * l2 * (1.0 - t), // 6: bottom (0,1)
        4.0 * l2 * l3 * (1.0 - t), // 7: bottom (1,2)
        4.0 * l3 * l1 * (1.0 - t), // 8: bottom (2,0)
        4.0 * l1 * l2 * t,         // 9: top (3,4)
        4.0 * l2 * l3 * t,         // 10: top (4,5)
        4.0 * l3 * l1 * t,         // 11: top (5,3)
        4.0 * l1 * t * (1.0 - t),  // 12: vertical (0,3)
        4.0 * l2 * t * (1.0 - t),  // 13: vertical (1,4)
        4.0 * l3 * t * (1.0 - t),  // 14: vertical (2,5)
    ]
}

fn penta15_dshape(xi: &[f64]) -> Vec<f64> {
    let (a, b, t) = (xi[0], xi[1], xi[2]);
    let (l1, l2, l3) = (1.0 - a - b, a, b);
    // (a, b) gradients of the triangle coordinates.
    let dl = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]; // dL1, dL2, dL3
    let ls = [l1, l2, l3];

    // Corner node on triangle vertex `i`, `top` selects the ζ = 1 face.
    let corner = |i: usize, top: bool| {
        let li = ls[i];
        let f = li * (2.0 * li - 1.0); // f(Li)
        let coef = if top {
            (4.0 * li - 1.0) * t - 2.0 * (t - t * t)
        } else {
            (4.0 * li - 1.0) * (1.0 - t) - 2.0 * (t - t * t)
        };
        let dc = if top {
            f - 2.0 * li * (1.0 - 2.0 * t)
        } else {
            -f - 2.0 * li * (1.0 - 2.0 * t)
        };
        [coef * dl[i][0], coef * dl[i][1], dc]
    };
    // Mid-edge triangle node between vertices `i`,`j`, `top` selects ζ = 1.
    let tri_edge = |i: usize, j: usize, top: bool| {
        let (la, lb) = (ls[i], ls[j]);
        let zfac = if top { t } else { 1.0 - t };
        let da = 4.0 * zfac * (la * dl[j][0] + lb * dl[i][0]);
        let db = 4.0 * zfac * (la * dl[j][1] + lb * dl[i][1]);
        let dc = if top { 4.0 * la * lb } else { -4.0 * la * lb };
        [da, db, dc]
    };
    // Vertical mid-edge node on triangle vertex `i` (ζ = 1/2).
    let vertical = |i: usize| {
        let li = ls[i];
        [
            4.0 * (t - t * t) * dl[i][0],
            4.0 * (t - t * t) * dl[i][1],
            4.0 * li * (1.0 - 2.0 * t),
        ]
    };
    let rows = [
        corner(0, false),
        corner(1, false),
        corner(2, false),
        corner(0, true),
        corner(1, true),
        corner(2, true),
        tri_edge(0, 1, false),
        tri_edge(1, 2, false),
        tri_edge(2, 0, false),
        tri_edge(0, 1, true),
        tri_edge(1, 2, true),
        tri_edge(2, 0, true),
        vertical(0),
        vertical(1),
        vertical(2),
    ];
    rows.iter().flatten().copied().collect()
}

impl fmt::Display for Interpolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl crate::dump::Dump for Interpolation {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum of Lagrange shape functions equals 1 everywhere (partition of
    /// unity).
    fn check_partition_of_unity(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let n = interp.shape(et, xi).unwrap();
        let s: f64 = n.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "{} on {} at xi={:?}: sum N_i = {} ≠ 1",
            interp,
            et,
            xi,
            s
        );
    }

    /// Sum of reference derivatives along each direction is 0 (partition
    /// of unity, differentiated).
    fn check_derivatives_sum_to_zero(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let dn = interp.dshape_dxi(et, xi).unwrap();
        let n_nodes = et.nodes_per_cell();
        let ref_dim = et.topological_dim();
        for j in 0..ref_dim {
            let mut s = 0.0;
            for i in 0..n_nodes {
                s += dn[i * ref_dim + j];
            }
            assert!(
                s.abs() < 1e-12,
                "{} on {}: Σ_i dN_i/dξ_{} = {} ≠ 0",
                interp,
                et,
                j,
                s
            );
        }
    }

    /// At node `i`, `N_i = 1` and `N_j = 0` for `j ≠ i` (Kronecker
    /// property of Lagrange interpolations).
    fn check_kronecker(interp: Interpolation, et: ElementType, ref_nodes: &[Vec<f64>]) {
        for (i, xi) in ref_nodes.iter().enumerate() {
            let n = interp.shape(et, xi).unwrap();
            for (j, &v) in n.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-12,
                    "{} on {} at node {}: N_{} = {} ≠ {}",
                    interp,
                    et,
                    i,
                    j,
                    v,
                    expected
                );
            }
        }
    }

    /// Central-difference estimate of the reference derivatives, used to
    /// validate the hand-written analytic `dshape_dxi` of the quadratic
    /// elements.
    fn central_diff_dshape(interp: Interpolation, et: ElementType, xi: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        let n_nodes = et.nodes_per_cell();
        let d = et.topological_dim();
        let mut out = vec![0.0; n_nodes * d];
        for k in 0..d {
            let mut xp = xi.to_vec();
            let mut xm = xi.to_vec();
            xp[k] += h;
            xm[k] -= h;
            let np = interp.shape(et, &xp).unwrap();
            let nm = interp.shape(et, &xm).unwrap();
            for i in 0..n_nodes {
                out[i * d + k] = (np[i] - nm[i]) / (2.0 * h);
            }
        }
        out
    }

    fn check_dshape_matches_fd(interp: Interpolation, et: ElementType, xi: &[f64]) {
        let ana = interp.dshape_dxi(et, xi).unwrap();
        let fd = central_diff_dshape(interp, et, xi);
        for (k, (a, f)) in ana.iter().zip(&fd).enumerate() {
            assert!(
                (a - f).abs() < 1e-5,
                "{} on {} at {:?}: dN[{}] analytic {} vs FD {}",
                interp,
                et,
                xi,
                k,
                a,
                f
            );
        }
    }

    /// Reference node coordinates of a quadratic element type (corners then
    /// mid-edge nodes), mirroring the `element_type` documentation.
    fn lag2_ref_nodes(et: ElementType) -> Vec<Vec<f64>> {
        match et {
            ElementType::SEG3 => vec![vec![-1.0], vec![1.0], vec![0.0]],
            ElementType::TRI6 => vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.0],
                vec![0.5, 0.5],
                vec![0.0, 0.5],
            ],
            ElementType::QUA8 => vec![
                vec![-1.0, -1.0],
                vec![1.0, -1.0],
                vec![1.0, 1.0],
                vec![-1.0, 1.0],
                vec![0.0, -1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![-1.0, 0.0],
            ],
            ElementType::QUA9 => vec![
                vec![-1.0, -1.0],
                vec![1.0, -1.0],
                vec![1.0, 1.0],
                vec![-1.0, 1.0],
                vec![0.0, -1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![-1.0, 0.0],
                vec![0.0, 0.0], // center
            ],
            ElementType::TET10 => vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.5, 0.0, 0.0],
                vec![0.5, 0.5, 0.0],
                vec![0.0, 0.5, 0.0],
                vec![0.0, 0.0, 0.5],
                vec![0.5, 0.0, 0.5],
                vec![0.0, 0.5, 0.5],
            ],
            ElementType::PENTA15 => vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0],
                vec![0.0, 1.0, 1.0],
                vec![0.5, 0.0, 0.0],
                vec![0.5, 0.5, 0.0],
                vec![0.0, 0.5, 0.0],
                vec![0.5, 0.0, 1.0],
                vec![0.5, 0.5, 1.0],
                vec![0.0, 0.5, 1.0],
                vec![0.0, 0.0, 0.5],
                vec![1.0, 0.0, 0.5],
                vec![0.0, 1.0, 0.5],
            ],
            ElementType::HEX20 => HEX20_REF_NODES
                .iter()
                .map(|&(a, b, c)| vec![a, b, c])
                .collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn lagrange2_all_types() {
        let samples: &[(ElementType, &[&[f64]])] = &[
            (ElementType::SEG3, &[&[-0.4], &[0.2], &[0.9]]),
            (ElementType::TRI6, &[&[0.2, 0.3], &[0.1, 0.6]]),
            (ElementType::QUA8, &[&[-0.3, 0.5], &[0.7, -0.2]]),
            (ElementType::QUA9, &[&[-0.3, 0.5], &[0.7, -0.2]]),
            (ElementType::TET10, &[&[0.2, 0.3, 0.1], &[0.1, 0.1, 0.5]]),
            (ElementType::PENTA15, &[&[0.2, 0.3, 0.4], &[0.1, 0.5, 0.8]]),
            (ElementType::HEX20, &[&[-0.3, 0.5, 0.2], &[0.6, -0.4, 0.9]]),
        ];
        for &(et, pts) in samples {
            // Kronecker delta at the nodes.
            check_kronecker(Interpolation::Lagrange2, et, &lag2_ref_nodes(et));
            // Partition of unity, derivative sum, and analytic-vs-FD gradient.
            for xi in pts {
                check_partition_of_unity(Interpolation::Lagrange2, et, xi);
                check_derivatives_sum_to_zero(Interpolation::Lagrange2, et, xi);
                check_dshape_matches_fd(Interpolation::Lagrange2, et, xi);
            }
        }
    }

    #[test]
    fn lagrange_degree_matches_element_type() {
        // Degree mismatch is rejected both ways.
        assert!(Interpolation::Lagrange2.is_compatible_with(ElementType::TRI6));
        assert!(!Interpolation::Lagrange2.is_compatible_with(ElementType::TRI3));
        assert!(!Interpolation::Lagrange1.is_compatible_with(ElementType::TRI6));
        assert!(Interpolation::Lagrange1
            .shape(ElementType::HEX20, &[0.0; 3])
            .is_err());
        assert!(Interpolation::Lagrange2
            .shape(ElementType::HEX8, &[0.0; 3])
            .is_err());
    }

    #[test]
    fn lagrange1_seg2() {
        // Reference nodes: ξ = ±1
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::SEG2,
            &[vec![-1.0], vec![1.0]],
        );
        for xi in [-1.0, -0.3, 0.0, 0.7, 1.0] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::SEG2, &[xi]);
        }
    }

    #[test]
    fn lagrange1_tri3() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TRI3,
            &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        for (a, b) in [(0.25, 0.25), (1.0 / 3.0, 1.0 / 3.0), (0.5, 0.0), (0.0, 0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TRI3, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_qua4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::QUA4,
            &[
                vec![-1.0, -1.0],
                vec![1.0, -1.0],
                vec![1.0, 1.0],
                vec![-1.0, 1.0],
            ],
        );
        for (a, b) in [(0.0, 0.0), (0.3, -0.7), (-0.5, 0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::QUA4, &[a, b]);
        }
    }

    #[test]
    fn lagrange1_tet4() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::TET4,
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        for (a, b, c) in [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::TET4, &[a, b, c]);
        }
    }

    #[test]
    fn lagrange1_hex8() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::HEX8,
            &HEX8_REF_NODES
                .iter()
                .map(|&(a, b, c)| vec![a, b, c])
                .collect::<Vec<_>>(),
        );
        for (a, b, c) in [(0.0, 0.0, 0.0), (0.3, -0.7, 0.5), (-0.5, 0.5, -0.5)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
            check_derivatives_sum_to_zero(Interpolation::Lagrange1, ElementType::HEX8, &[a, b, c]);
        }
    }

    #[test]
    fn lagrange1_penta6() {
        check_kronecker(
            Interpolation::Lagrange1,
            ElementType::PENTA6,
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 1.0],
                vec![0.0, 1.0, 1.0],
            ],
        );
        for (a, b, c) in [(1.0 / 3.0, 1.0 / 3.0, 0.5), (0.1, 0.2, 0.7)] {
            check_partition_of_unity(Interpolation::Lagrange1, ElementType::PENTA6, &[a, b, c]);
            check_derivatives_sum_to_zero(
                Interpolation::Lagrange1,
                ElementType::PENTA6,
                &[a, b, c],
            );
        }
    }

    #[test]
    fn rejects_poi1() {
        assert!(!Interpolation::Lagrange1.is_compatible_with(ElementType::POI1));
        assert!(Interpolation::Lagrange1
            .shape(ElementType::POI1, &[])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::POI1, &[])
            .is_err());
    }

    #[test]
    fn rejects_bad_xi_length() {
        assert!(Interpolation::Lagrange1
            .shape(ElementType::SEG2, &[0.0, 0.0])
            .is_err());
        assert!(Interpolation::Lagrange1
            .dshape_dxi(ElementType::TRI3, &[0.0])
            .is_err());
    }

    #[test]
    fn display_and_parsing() {
        assert_eq!(format!("{}", Interpolation::Lagrange1), "LAGRANGE1");
        assert_eq!(
            Interpolation::from_name("lagrange1"),
            Some(Interpolation::Lagrange1)
        );
        assert_eq!(
            Interpolation::from_name("LAG1"),
            Some(Interpolation::Lagrange1)
        );
        assert_eq!(Interpolation::from_name("unknown"), None);
    }
}
