//! Match an external quadrature to pyrucast's, through the reference
//! elements.
//!
//! A format that stores values **at Gauss points** (MED `ON_GAUSS_PT`)
//! declares its rule in its **own** reference element: node coordinates,
//! point coordinates and weights. pyrucast's reference element is a different
//! but affinely equivalent one — same shape, other placement, other node
//! numbering. Fitting the affine map on the corners (numbered through the
//! format's [`NodeOrder`] permutation) brings pyrucast's points into the
//! format's frame, where they can be compared one by one:
//!
//! - [`match_gauss`] — import: which external point is pyrucast's point `g`?
//!   Errors when the two rules are not the same set of points.
//! - [`gauss_to_external`] — export: pyrucast's rule expressed in the
//!   format's frame, to be declared as it is.
//!
//! Both run **once per element type**, never per cell.

use crate::atoms::ElementType;
use crate::error::{PyrucastError, Result};
use crate::ops::mesh::arrays::NodeOrder;
use nalgebra::DMatrix;

/// Relative tolerance on a coordinate or a weight, scaled by the size of the
/// external reference element.
const TOL: f64 = 1e-8;

fn err(msg: impl Into<String>) -> PyrucastError {
    PyrucastError::Message(msg.into())
}

/// The affine map `x_ext = A ξ + b` from pyrucast's reference element to the
/// external one, fitted on the corners and checked on **every** node — a
/// mid-side node in the wrong place means the two elements are not the same.
struct RefMap {
    d: usize,
    /// Row-major `d × d`.
    a: Vec<f64>,
    b: Vec<f64>,
    det: f64,
    /// Size of the external element, the scale of the tolerances.
    scale: f64,
}

impl RefMap {
    fn fit(et: ElementType, order: NodeOrder, ext_ref_nodes: &[f64]) -> Result<Self> {
        let kind = et.as_kind();
        let d = kind.topological_dim();
        let n = kind.nodes_per_cell();
        if d == 0 {
            return Err(err(format!("{et}: no reference element to map")));
        }
        if ext_ref_nodes.len() != n * d {
            return Err(err(format!(
                "{et}: the reference element holds {n} nodes of {d} coordinates, got {} values",
                ext_ref_nodes.len()
            )));
        }
        let perm = order.permutation(et);
        let pyr = kind.ref_nodes();
        let ext = |i: usize| &ext_ref_nodes[perm[i] * d..perm[i] * d + d];

        // Least squares on the corners: rows [ξ, 1] → x_ext.
        let nc = kind.corner_count();
        let x = DMatrix::from_fn(nc, d + 1, |r, c| if c < d { pyr[r][c] } else { 1.0 });
        let y = DMatrix::from_fn(nc, d, |r, c| ext(r)[c]);
        let xt = x.transpose();
        let sol = (&xt * &x)
            .try_inverse()
            .ok_or_else(|| err(format!("{et}: degenerate external reference element")))?
            * (&xt * &y);
        let mut a = vec![0.0; d * d];
        let mut b = vec![0.0; d];
        for r in 0..d {
            for c in 0..d {
                a[r * d + c] = sol[(c, r)];
            }
            b[r] = sol[(d, r)];
        }
        let det = DMatrix::from_row_slice(d, d, &a).determinant();
        let scale = ext_ref_nodes
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(1.0);
        let map = Self {
            d,
            a,
            b,
            det,
            scale,
        };

        let mut img = vec![0.0; d];
        for (i, p) in pyr.iter().enumerate() {
            map.apply(p, &mut img);
            if img
                .iter()
                .zip(ext(i))
                .any(|(u, v)| (u - v).abs() > TOL * scale)
            {
                return Err(err(format!(
                    "{et}: the external reference element is not an affine image of pyrucast's \
                     (node {i} does not land where the {order:?} numbering puts it)"
                )));
            }
        }
        Ok(map)
    }

    /// pyrucast's Gauss rule for `et`, carried into the external frame.
    fn rule(&self, et: ElementType) -> (Vec<f64>, Vec<f64>) {
        let (xi, w) = et.as_kind().gauss();
        let mut out = vec![0.0; xi.len()];
        for (src, dst) in xi.chunks_exact(self.d).zip(out.chunks_exact_mut(self.d)) {
            self.apply(src, dst);
        }
        let jac = self.det.abs();
        (out, w.iter().map(|w| w * jac).collect())
    }

    fn apply(&self, xi: &[f64], out: &mut [f64]) {
        let d = self.d;
        for r in 0..d {
            out[r] = self.b[r] + (0..d).map(|c| self.a[r * d + c] * xi[c]).sum::<f64>();
        }
    }
}

/// pyrucast's Gauss rule for `et`, mapped into the external reference element
/// described by `ext_ref_nodes` (numbered in `order`, `dim` coordinates each):
/// `(xi, weights)`, points in pyrucast's order. The weights are scaled by the
/// ratio of the two reference measures.
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # use pyrucast::ops::mesh::gauss_map::gauss_to_external;
/// // A reference triangle twice as large as pyrucast's: same points, scaled,
/// // and weights four times as heavy.
/// let (xi, w) = gauss_to_external(
///     ElementType::TRI3, NodeOrder::Pyrucast, &[0.0, 0.0, 2.0, 0.0, 0.0, 2.0])?;
/// let (xi0, w0) = ElementType::TRI3.as_kind().gauss();
/// assert!(xi.iter().zip(&xi0).all(|(a, b)| (a - 2.0 * b).abs() < 1e-12));
/// assert!(w.iter().zip(&w0).all(|(a, b)| (a - 4.0 * b).abs() < 1e-12));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn gauss_to_external(
    et: ElementType,
    order: NodeOrder,
    ext_ref_nodes: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let map = RefMap::fit(et, order, ext_ref_nodes)?;
    Ok(map.rule(et))
}

/// For each of pyrucast's Gauss points of `et`, the index of the **same**
/// point in the external rule `(ext_xi, ext_weights)`, declared in the
/// external reference element `ext_ref_nodes` (numbered in `order`).
///
/// Errors when the two rules differ — other point count, a point pyrucast
/// does not have, or a different weight: values at those points cannot be
/// carried over as they are.
///
/// ```
/// # use pyrucast::atoms::ElementType;
/// # use pyrucast::ops::mesh::arrays::NodeOrder;
/// # use pyrucast::ops::mesh::gauss_map::{gauss_to_external, match_gauss};
/// let refs = [0.0, 0.0, 2.0, 0.0, 0.0, 2.0];
/// let (mut xi, mut w) = gauss_to_external(ElementType::TRI3, NodeOrder::Pyrucast, &refs)?;
/// // The external file lists the same points backwards.
/// xi = xi.chunks(2).rev().flatten().copied().collect();
/// w.reverse();
/// let found = match_gauss(ElementType::TRI3, NodeOrder::Pyrucast, &refs, &xi, &w)?;
/// assert_eq!(found, [2, 1, 0]);
/// // A one-point rule is not pyrucast's three-point one.
/// assert!(match_gauss(ElementType::TRI3, NodeOrder::Pyrucast, &refs, &[0.66, 0.66], &[2.0]).is_err());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn match_gauss(
    et: ElementType,
    order: NodeOrder,
    ext_ref_nodes: &[f64],
    ext_xi: &[f64],
    ext_weights: &[f64],
) -> Result<Vec<usize>> {
    let map = RefMap::fit(et, order, ext_ref_nodes)?;
    let (xi, w) = map.rule(et);
    let d = map.d;
    let n = w.len();
    if ext_weights.len() != n || ext_xi.len() != n * d {
        return Err(err(format!(
            "{et}: the external rule has {} point(s), pyrucast's has {n}",
            ext_weights.len()
        )));
    }
    let tol = TOL * map.scale;
    let wtol = TOL
        * w.iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
    let mut found = Vec::with_capacity(n);
    for (g, p) in xi.chunks_exact(d).enumerate() {
        let k = ext_xi
            .chunks_exact(d)
            .position(|q| p.iter().zip(q).all(|(u, v)| (u - v).abs() <= tol))
            .filter(|&k| (ext_weights[k] - w[g]).abs() <= wtol)
            .ok_or_else(|| {
                err(format!(
                    "{et}: Gauss point {g} of pyrucast's rule has no counterpart in the external \
                     rule (not the same quadrature)"
                ))
            })?;
        found.push(k);
    }
    Ok(found)
}
