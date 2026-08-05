//! `PYRA5` — the 5-node pyramid.

use super::{ElementKind, Facet, Interpolation};
use crate::atoms::ElementType;

/// 5-node pyramid: a square base and an apex. Reference: `ζ ∈ [0, 1]` with
/// `ξ, η ∈ [-(1-ζ), +(1-ζ)]`, so the square shrinks to a point at the apex.
/// Local order: base CCW seen from the apex (nodes 0..3 at `ζ = 0`), then the
/// apex.
///
/// This is the element that makes a hexahedron and a tetrahedron meet: its
/// square face matches a `HEX8` face and its four triangles match `TET4`
/// faces, so a hexahedral layer closes onto a tetrahedral core without a
/// hanging node.
pub struct Pyra5;

/// Signs `(ξ_i, η_i)` of the four base nodes, counter-clockwise seen from the
/// apex.
const BASE: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// Below this distance from the apex the rational shape functions are taken at
/// their limit instead of evaluated, which would divide by zero.
const APEX_EPS: f64 = 1e-12;

const EDGES: &[[usize; 2]] = &[
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
];

/// Corner faces, outward-oriented: the square base first, wound so its normal
/// points away from the apex, then the four side triangles.
pub(super) const CORNER_FACES: [&[usize]; 5] = [
    &[0, 3, 2, 1],
    &[0, 1, 4],
    &[1, 2, 4],
    &[2, 3, 4],
    &[3, 0, 4],
];

const FACETS: &[Facet] = &[
    Facet {
        element_type: ElementType::QUA4,
        nodes: CORNER_FACES[0],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[1],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[2],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[3],
    },
    Facet {
        element_type: ElementType::TRI3,
        nodes: CORNER_FACES[4],
    },
];

impl ElementKind for Pyra5 {
    fn element_type(&self) -> ElementType {
        ElementType::PYRA5
    }

    fn ref_nodes(&self) -> &'static [&'static [f64]] {
        &[
            &[-1.0, -1.0, 0.0],
            &[1.0, -1.0, 0.0],
            &[1.0, 1.0, 0.0],
            &[-1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ]
    }

    fn reversal_permutation(&self) -> &'static [usize] {
        &[0, 3, 2, 1, 4]
    }

    fn corner_count(&self) -> usize {
        5
    }

    fn facets(&self) -> &'static [Facet] {
        FACETS
    }

    fn edges(&self) -> &'static [[usize; 2]] {
        EDGES
    }

    /// A quarter of the way up — the centroid of a pyramid, not the mean of
    /// its nodes.
    fn ref_centroid(&self) -> &'static [f64] {
        &[0.0, 0.0, 0.25]
    }

    /// Square base of side 2 tapering to a point: `∫₀¹ (2(1-ζ))² dζ = 4/3`.
    fn ref_measure(&self) -> f64 {
        4.0 / 3.0
    }

    /// The square cross-section shrinks as `ζ` climbs, so the bound on `ξ` and
    /// `η` is `1 - ζ` rather than a constant.
    fn contains_ref(&self, xi: &[f64], tol: f64) -> bool {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let half = 1.0 - c + tol;
        c >= -tol && c <= 1.0 + tol && a.abs() <= half && b.abs() <= half
    }

    fn clamp_ref(&self, _xi: &mut [f64]) {}
    fn degree(&self) -> Option<Interpolation> {
        Some(Interpolation::Lagrange1)
    }

    /// The pyramid is the one common element whose shape functions are **not**
    /// polynomial. Its square base has to collapse to a single point at the
    /// apex, and no polynomial does that while staying bilinear on the base.
    /// Writing `m = 1 - ζ` for the half-width of the cross-section, the base
    /// functions are the bilinear ones in the *scaled* coordinates `ξ/m`,
    /// `η/m`, weighted by `m`:
    ///
    /// ```text
    /// N_i = (m / 4) (1 + ξ_i ξ/m) (1 + η_i η/m)   for i = 0..3,    N_4 = ζ
    /// ```
    ///
    /// Expanded, the cross term is `ξ_i η_i ξη / (4m)` — the rational part,
    /// and the reason the pyramid needs a quadrature rule of its own. It is
    /// bounded (`|ξ|, |η| ≤ m` on the reference element, so the term is at
    /// most `m/4`) but genuinely singular *at* the apex, where the limit
    /// `N_4 = 1` is taken directly.
    fn shape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let m = 1.0 - c;
        if m <= APEX_EPS {
            out.copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0]);
            return;
        }
        let (u, v) = (a / m, b / m);
        for (i, &(si, ti)) in BASE.iter().enumerate() {
            out[i] = 0.25 * m * (1.0 + si * u) * (1.0 + ti * v);
        }
        out[4] = c;
    }

    /// Differentiating `N_i = (1/4)[m + ξ_i ξ + η_i η + ξ_i η_i ξη/m]` gives
    /// `∂/∂ξ = (ξ_i/4)(1 + η_i v)`, `∂/∂η = (η_i/4)(1 + ξ_i u)` and
    /// `∂/∂ζ = (1/4)(-1 + ξ_i η_i u v)`, with `u = ξ/m`, `v = η/m`. The three
    /// sums over the five nodes vanish, as they must.
    fn dshape_into(&self, xi: &[f64], out: &mut [f64]) {
        let (a, b, c) = (xi[0], xi[1], xi[2]);
        let m = 1.0 - c;
        if m <= APEX_EPS {
            // At the apex only the ζ derivative survives in the limit.
            out.copy_from_slice(&[
                0.0, 0.0, -0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25, 0.0, 0.0, -0.25, 0.0, 0.0, 1.0,
            ]);
            return;
        }
        let (u, v) = (a / m, b / m);
        for (i, &(si, ti)) in BASE.iter().enumerate() {
            out[3 * i] = 0.25 * si * (1.0 + ti * v);
            out[3 * i + 1] = 0.25 * ti * (1.0 + si * u);
            out[3 * i + 2] = 0.25 * (-1.0 + si * ti * u * v);
        }
        out[12] = 0.0;
        out[13] = 0.0;
        out[14] = 1.0;
    }
    /// A **conical** product rule: 2 × 2 Gauss–Legendre across the square
    /// cross-section, times a 2-point Gauss–Jacobi rule in `ζ`.
    ///
    /// A pyramid is not a product of simplices, so it needs a rule of its own.
    /// The Jacobi weight is what makes it work: writing a point as
    /// `ξ = a(1-ζ)`, `η = b(1-ζ)` with `a, b ∈ [-1, 1]`, the change of
    /// variables brings out `dξ dη = (1-ζ)² da db` — exactly the cross-section
    /// shrinking toward the apex. Integrating the `ζ` direction against that
    /// `(1-ζ)²` is a Gauss–Jacobi rule with `α = 2`, whose two nodes are the
    /// roots of `z² - (2/3) z + 1/15`, namely `1/3 ∓ √10/15`.
    ///
    /// The weights then sum to `2 × 2 × 1/3 = 4/3`, the reference volume.
    fn gauss(&self) -> (Vec<f64>, Vec<f64>) {
        let a = 1.0 / 3.0_f64.sqrt();
        let spread = 10.0_f64.sqrt() / 15.0;
        let (z0, z1) = (1.0 / 3.0 - spread, 1.0 / 3.0 + spread);
        // From the first two moments of `(1-z)²` on `[0, 1]`:
        // `w0 + w1 = 1/3` and `w0 z0 + w1 z1 = 1/12`.
        let w1 = (1.0 / 12.0 - z0 / 3.0) / (z1 - z0);
        let w0 = 1.0 / 3.0 - w1;
        let mut xi = Vec::with_capacity(8 * 3);
        let mut w = Vec::with_capacity(8);
        for &sa in &[-a, a] {
            for &sb in &[-a, a] {
                for &(z, wz) in &[(z0, w0), (z1, w1)] {
                    xi.push(sa * (1.0 - z));
                    xi.push(sb * (1.0 - z));
                    xi.push(z);
                    w.push(wz);
                }
            }
        }
        (xi, w)
    }

    fn vtk_code(&self) -> u8 {
        14 // VTK_PYRAMID
    }

    fn gmsh_code(&self) -> u32 {
        7
    }
}
