//! Finite-element space — interpolation + quadrature layer on top of a mesh.
//!
//! Hierarchy mirroring [`crate::mesh`]:
//!
//! - [`SubFiniteElementSpace`] — one [`crate::containers::finite_element_space::Interpolation`] and one
//!   [`crate::containers::finite_element_space::QuadratureRule`] applied to a single
//!   [`crate::containers::mesh::SubMesh`]. It stores the **reference-space tables**
//!   that do not depend on the physical coordinates of the nodes
//!   (Gauss points and weights, shape functions and reference
//!   derivatives at those Gauss points) and computes the physical
//!   quantities — Jacobian, `|J|`, `dN/dx` — **on the fly** from the
//!   current node coordinates in the
//!   [`crate::containers::mesh::Configuration`].
//! - [`FiniteElementSpace`] — collection of `SubFiniteElementSpace` matching the
//!   submeshes of a [`crate::containers::mesh::Mesh`] one-for-one. The mesh handle
//!   is captured at construction.
//!
//! The mesh **topology** (connectivity, element types) is frozen at the
//! `FiniteElementSpace` construction. The mesh **geometry** (node
//! coordinates) may evolve later (e.g. mesh displacement); the
//! on-the-fly Jacobian computation always reflects the current
//! coordinates.
//!
//! POI1 submeshes are rejected: a point element has no reference frame.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, with};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.0, 2.0]).unwrap();
//!
//! let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.subspace(0).unwrap();
//! with(&sub, |s| {
//!     assert_eq!(s.gauss_count(), 3);
//!     // |J| of a triangle with vertices (0,0), (2,0), (0,2): the mapping
//!     // is linear, |J| = 4 (twice the area, since ref triangle has area 1/2).
//!     for g in 0..s.gauss_count() {
//!         let dj = s.det_jacobian(0, g).unwrap();
//!         assert!((dj - 4.0).abs() < 1e-12);
//!     }
//! })
//! .unwrap();
//! ```

pub mod element;
pub mod interpolation;
pub mod quadrature;

pub use element::{Element, ElementIter};
pub use interpolation::Interpolation;
pub use quadrature::QuadratureRule;

use crate::containers::mesh::{Configuration, NodeId};
use crate::error::{PyrucastError, Result};
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{insert, with, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SubFiniteElementSpace ────────────────────────────────────────────────────────────

/// Finite-element space attached to a single [`SubMesh`].
///
/// Stores only the **reference-space** tables (independent of the node
/// coordinates); physical quantities are computed on the fly.
#[derive(Serialize, Deserialize)]
pub struct SubFiniteElementSpace {
    submesh: Handle<SubMesh>,
    interpolation: Interpolation,
    quadrature: QuadratureRule,
    /// Geometric dimension of the owning `Configuration` at construction
    /// time (used only to size the on-the-fly Jacobians). The
    /// `Configuration` may not change dimension, so this is stable for
    /// the lifetime of the subspace.
    space_dim: usize,

    // Reference-space tables (invariant under mesh deformation):
    /// Flat `n_g × ref_dim` reference coordinates of the Gauss points.
    gauss_xi: Vec<f64>,
    /// `n_g` Gauss weights.
    gauss_w: Vec<f64>,
    /// Flat `n_g × n_nodes` values of `N_i(ξ_g)`.
    n_at_g: Vec<f64>,
    /// Flat `n_g × n_nodes × ref_dim` values of `∂N_i/∂ξ_j(ξ_g)`.
    dn_at_g: Vec<f64>,
}

impl SubFiniteElementSpace {
    /// Build a subspace over `submesh` with the given interpolation and
    /// quadrature.
    ///
    /// Validates structural compatibility only (POI1 rejected,
    /// `(ElementType, Interpolation)` pair supported,
    /// `space_dim ≥ ref_dim`). The Jacobian is **not** evaluated at
    /// construction.
    pub fn new(
        submesh: Handle<SubMesh>,
        interpolation: Interpolation,
        quadrature: QuadratureRule,
    ) -> Result<Self> {
        let (et, cfg) = with(&submesh, |s| (s.element_type(), s.configuration()))?;
        if et == ElementType::POI1 {
            return Err(PyrucastError::Message(
                "SubFiniteElementSpace: POI1 submesh is not supported (no reference frame)".into(),
            ));
        }
        if !interpolation.is_compatible_with(et) {
            return Err(PyrucastError::Message(format!(
                "SubFiniteElementSpace: interpolation {} not compatible with element type {}",
                interpolation, et
            )));
        }
        if !quadrature.is_compatible_with(et) {
            return Err(PyrucastError::Message(format!(
                "SubFiniteElementSpace: quadrature {} not compatible with element type {}",
                quadrature, et
            )));
        }
        let space_dim = with(&cfg, |c| c.dim())? as usize;
        let ref_dim = et.topological_dim();
        if space_dim < ref_dim {
            return Err(PyrucastError::Message(format!(
                "SubFiniteElementSpace: space dim {} < reference dim {} of {} (cannot define a Jacobian)",
                space_dim, ref_dim, et
            )));
        }

        let n_nodes = et.nodes_per_cell();
        let (gauss_xi, gauss_w) = quadrature.points(et)?;
        let n_g = gauss_w.len();

        let mut n_at_g = Vec::with_capacity(n_g * n_nodes);
        let mut dn_at_g = Vec::with_capacity(n_g * n_nodes * ref_dim);
        for g in 0..n_g {
            let xi = &gauss_xi[g * ref_dim..(g + 1) * ref_dim];
            n_at_g.extend_from_slice(&interpolation.shape(et, xi)?);
            dn_at_g.extend_from_slice(&interpolation.dshape_dxi(et, xi)?);
        }

        Ok(Self {
            submesh,
            interpolation,
            quadrature,
            space_dim,
            gauss_xi,
            gauss_w,
            n_at_g,
            dn_at_g,
        })
    }

    // ── Accessors (structural) ──────────────────────────────────────────────

    /// Handle to the underlying submesh (internal clone).
    pub fn submesh(&self) -> Handle<SubMesh> {
        self.submesh.clone()
    }

    /// Handle to the owning `Configuration` (internal clone).
    pub fn configuration(&self) -> Result<Handle<Configuration>> {
        with(&self.submesh, |s| s.configuration())
    }

    /// Interpolation in use.
    pub fn interpolation(&self) -> Interpolation {
        self.interpolation
    }

    /// Quadrature rule in use.
    pub fn quadrature(&self) -> QuadratureRule {
        self.quadrature
    }

    /// Element type of the submesh.
    pub fn element_type(&self) -> Result<ElementType> {
        with(&self.submesh, |s| s.element_type())
    }

    /// Reference dimension (= topological dim of the element type).
    pub fn ref_dim(&self) -> Result<usize> {
        Ok(self.element_type()?.topological_dim())
    }

    /// Geometric (physical) dimension of the underlying `Configuration`.
    pub fn space_dim(&self) -> usize {
        self.space_dim
    }

    /// Number of nodes per cell (= `element_type().nodes_per_cell()`).
    pub fn nodes_per_cell(&self) -> Result<usize> {
        Ok(self.element_type()?.nodes_per_cell())
    }

    /// Number of cells in the underlying submesh.
    pub fn cell_count(&self) -> Result<usize> {
        with(&self.submesh, |s| s.cell_count())
    }

    /// Number of Gauss points per cell.
    pub fn gauss_count(&self) -> usize {
        self.gauss_w.len()
    }

    // ── Reference-space accessors ───────────────────────────────────────────

    /// Reference coordinates of the `g`-th Gauss point (length `ref_dim`).
    pub fn gauss_xi(&self, g: usize) -> Result<&[f64]> {
        self.check_g(g)?;
        let ref_dim = self.ref_dim()?;
        Ok(&self.gauss_xi[g * ref_dim..(g + 1) * ref_dim])
    }

    /// Weight of the `g`-th Gauss point.
    pub fn gauss_weight(&self, g: usize) -> Result<f64> {
        self.check_g(g)?;
        Ok(self.gauss_w[g])
    }

    /// `N_i(ξ_g)` for all nodes `i` at the `g`-th Gauss point
    /// (length `nodes_per_cell`).
    pub fn n_at_g(&self, g: usize) -> Result<&[f64]> {
        self.check_g(g)?;
        let n_nodes = self.nodes_per_cell()?;
        Ok(&self.n_at_g[g * n_nodes..(g + 1) * n_nodes])
    }

    /// `∂N_i/∂ξ_j(ξ_g)` for all nodes at the `g`-th Gauss point.
    ///
    /// Flat row-major buffer of length `nodes_per_cell × ref_dim`, with
    /// `[i * ref_dim + j]` = `∂N_i/∂ξ_j`.
    pub fn dn_at_g(&self, g: usize) -> Result<&[f64]> {
        self.check_g(g)?;
        let n_nodes = self.nodes_per_cell()?;
        let ref_dim = self.ref_dim()?;
        let stride = n_nodes * ref_dim;
        Ok(&self.dn_at_g[g * stride..(g + 1) * stride])
    }

    // ── Physical quantities (on-the-fly) ────────────────────────────────────

    /// Jacobian `J = ∂x/∂ξ` of cell `cell_idx` at the `g`-th Gauss point.
    ///
    /// Flat row-major buffer of length `space_dim × ref_dim`, with
    /// `[a * ref_dim + k]` = `∂x_a/∂ξ_k`. Each entry is built from the
    /// **current** node coordinates in the `Configuration`.
    pub fn jacobian(&self, cell_idx: usize, g: usize) -> Result<Vec<f64>> {
        self.check_g(g)?;
        let coords = self.cell_node_coords(cell_idx)?;
        let dn = self.dn_at_g(g)?;
        Ok(build_jacobian(
            &coords,
            dn,
            self.space_dim,
            self.ref_dim()?,
            self.nodes_per_cell()?,
        ))
    }

    /// Determinant `|J|` of the Jacobian — `det(J)` if `space_dim ==
    /// ref_dim`, `sqrt(det(JᵀJ))` for manifold elements
    /// (`space_dim > ref_dim`). The returned value is always
    /// non-negative; it is the **measure scaling factor** to use in
    /// numerical integration.
    pub fn det_jacobian(&self, cell_idx: usize, g: usize) -> Result<f64> {
        let jac = self.jacobian(cell_idx, g)?;
        Ok(jacobian_measure(&jac, self.space_dim, self.ref_dim()?))
    }

    /// Physical derivatives `∂N_i/∂x_a` at cell `cell_idx`, Gauss point
    /// `g`.
    ///
    /// Flat row-major buffer of length `nodes_per_cell × space_dim`,
    /// with `[i * space_dim + a]` = `∂N_i/∂x_a`. For manifold elements
    /// (`space_dim > ref_dim`), the returned gradient is the **tangent**
    /// gradient on the embedded surface / curve.
    pub fn dn_dx(&self, cell_idx: usize, g: usize) -> Result<Vec<f64>> {
        let jac = self.jacobian(cell_idx, g)?;
        let dn_dxi = self.dn_at_g(g)?;
        let n_nodes = self.nodes_per_cell()?;
        let ref_dim = self.ref_dim()?;
        let space_dim = self.space_dim;
        build_dn_dx(&jac, dn_dxi, space_dim, ref_dim, n_nodes)
    }

    // ── Internals ───────────────────────────────────────────────────────────

    /// Read all node coordinates of a cell into a flat buffer
    /// `[n_nodes × space_dim]`, row-major (`[i * space_dim + a]`).
    fn cell_node_coords(&self, cell_idx: usize) -> Result<Vec<f64>> {
        let n_nodes = self.nodes_per_cell()?;
        let (cfg, ids): (Handle<Configuration>, Vec<NodeId>) =
            with(&self.submesh, |s| -> Result<_> {
                let total = s.cell_count();
                if cell_idx >= total {
                    return Err(PyrucastError::Message(format!(
                        "SubFiniteElementSpace: cell index {} ≥ cell_count {}",
                        cell_idx, total
                    )));
                }
                let conn = s.connectivity();
                let ids = conn[cell_idx * n_nodes..(cell_idx + 1) * n_nodes].to_vec();
                Ok((s.configuration(), ids))
            })??;
        let mut out = Vec::with_capacity(n_nodes * self.space_dim);
        with(&cfg, |c| -> Result<()> {
            for id in ids {
                out.extend_from_slice(c.coord(id)?);
            }
            Ok(())
        })??;
        Ok(out)
    }

    fn check_g(&self, g: usize) -> Result<()> {
        if g >= self.gauss_count() {
            return Err(PyrucastError::Message(format!(
                "SubFiniteElementSpace: gauss index {} ≥ n_g {}",
                g,
                self.gauss_count()
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for SubFiniteElementSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubFiniteElementSpace")
            .field("submesh", &self.submesh)
            .field("interpolation", &self.interpolation)
            .field("quadrature", &self.quadrature)
            .field("space_dim", &self.space_dim)
            .field("n_g", &self.gauss_count())
            .finish()
    }
}

impl fmt::Display for SubFiniteElementSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let et = self
            .element_type()
            .map(|e| e.name())
            .unwrap_or("?");
        write!(
            f,
            "SubFiniteElementSpace<{}, {}, {}>: {} Gauss point(s)",
            et,
            self.interpolation,
            self.quadrature,
            self.gauss_count()
        )
    }
}

impl crate::dump::Dump for SubFiniteElementSpace {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let ng = self.gauss_count();
        let ref_dim = if ng > 0 { self.gauss_xi.len() / ng } else { 0 };
        let mut headers = vec!["g".to_string()];
        headers.extend((0..ref_dim).map(|i| format!("ξ{i}")));
        headers.push("weight".to_string());
        let rows: Vec<Vec<String>> = (0..ng)
            .map(|g| {
                let mut row = vec![g.to_string()];
                for j in 0..ref_dim {
                    row.push(fmt_float(self.gauss_xi[g * ref_dim + j], opts.precision));
                }
                row.push(fmt_float(self.gauss_w[g], opts.precision));
                row
            })
            .collect();
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── FiniteElementSpace ────────────────────────────────────────────────────

/// Finite-element space attached to a [`Mesh`] — one [`SubFiniteElementSpace`] per
/// submesh, in the same order.
///
/// Topology (connectivity, element types) is frozen at construction:
/// each `SubFiniteElementSpace` captures its `SubMesh` handle. The node coordinates
/// in the underlying `Configuration` may evolve later; the on-the-fly
/// Jacobian computation always reflects the current coordinates.
#[derive(Serialize, Deserialize, Default)]
pub struct FiniteElementSpace {
    subs: Vec<Handle<SubFiniteElementSpace>>,
}

crate::impl_aggregate!(FiniteElementSpace, SubFiniteElementSpace, subspace, "subspace(s)");
crate::impl_aggregate_dump!(FiniteElementSpace);

impl FiniteElementSpace {
    /// Build a `FiniteElementSpace` by attaching the supplied
    /// `(interpolation, quadrature)` pair to each submesh of `mesh`, in
    /// order.
    ///
    /// `choices.len()` must equal `mesh.submesh_count()`. The mesh must
    /// have at least one submesh and none of them may be POI1.
    pub fn with(
        mesh: &Mesh,
        choices: &[(Interpolation, QuadratureRule)],
    ) -> Result<Self> {
        let n_sub = mesh.submesh_count();
        if n_sub == 0 {
            return Err(PyrucastError::Message(
                "FiniteElementSpace: mesh has no submesh".into(),
            ));
        }
        if choices.len() != n_sub {
            return Err(PyrucastError::Message(format!(
                "FiniteElementSpace: {} (interpolation, quadrature) pair(s) supplied for {} submesh(es)",
                choices.len(),
                n_sub
            )));
        }
        let mut subs = Vec::with_capacity(n_sub);
        for (i, &(interp, quad)) in choices.iter().enumerate() {
            let sm = mesh.submesh(i)?;
            let sub = SubFiniteElementSpace::new(sm, interp, quad)?;
            subs.push(insert(sub));
        }
        Ok(Self { subs })
    }

    /// Build a `FiniteElementSpace` using the same `interpolation` for
    /// every submesh, with the default Gauss quadrature.
    pub fn new(mesh: &Mesh, interpolation: Interpolation) -> Result<Self> {
        let n_sub = mesh.submesh_count();
        let choices: Vec<_> = (0..n_sub)
            .map(|_| (interpolation, QuadratureRule::Gauss))
            .collect();
        Self::with(mesh, &choices)
    }

    /// Build the default Lagrange-1 FE space over `mesh`. Equivalent to
    /// `FiniteElementSpace::new(mesh, Interpolation::Lagrange1)`.
    pub fn lagrange1(mesh: &Mesh) -> Result<Self> {
        Self::new(mesh, Interpolation::Lagrange1)
    }

    /// Element view on cell `cell_idx` of subspace `subspace_idx`.
    pub fn element(&self, subspace_idx: usize, cell_idx: usize) -> Result<Element> {
        let sub = self.subspace(subspace_idx)?;
        Element::new(sub, cell_idx)
    }

    /// Iterator over every element of subspace `subspace_idx`.
    pub fn elements(&self, subspace_idx: usize) -> Result<ElementIter> {
        let sub = self.subspace(subspace_idx)?;
        let n = with(&sub, |s| s.cell_count())??;
        Ok(ElementIter::new(sub, n))
    }
}

// ─── Numerical helpers ─────────────────────────────────────────────────────

/// Build the Jacobian `J[a*ref_dim + k] = Σ_i x_i[a] · dN_i/dξ_k`.
///
/// `coords` has layout `[i * space_dim + a]`, `dn_dxi` has layout
/// `[i * ref_dim + k]`.
fn build_jacobian(
    coords: &[f64],
    dn_dxi: &[f64],
    space_dim: usize,
    ref_dim: usize,
    n_nodes: usize,
) -> Vec<f64> {
    let mut jac = vec![0.0; space_dim * ref_dim];
    for a in 0..space_dim {
        for k in 0..ref_dim {
            let mut sum = 0.0;
            for i in 0..n_nodes {
                sum += coords[i * space_dim + a] * dn_dxi[i * ref_dim + k];
            }
            jac[a * ref_dim + k] = sum;
        }
    }
    jac
}

/// Compute `sqrt(det(JᵀJ))` — the measure scaling factor used in
/// numerical integration. For square `J` this equals `|det(J)|`.
fn jacobian_measure(jac: &[f64], space_dim: usize, ref_dim: usize) -> f64 {
    let g = gram_matrix(jac, space_dim, ref_dim);
    det_small(&g, ref_dim).max(0.0).sqrt()
}

/// Build `G = JᵀJ` of size `ref_dim × ref_dim`, row-major.
fn gram_matrix(jac: &[f64], space_dim: usize, ref_dim: usize) -> Vec<f64> {
    let mut g = vec![0.0; ref_dim * ref_dim];
    for i in 0..ref_dim {
        for j in 0..ref_dim {
            let mut s = 0.0;
            for a in 0..space_dim {
                s += jac[a * ref_dim + i] * jac[a * ref_dim + j];
            }
            g[i * ref_dim + j] = s;
        }
    }
    g
}

/// Determinant of a small (1×1, 2×2, 3×3) row-major square matrix.
fn det_small(m: &[f64], n: usize) -> f64 {
    match n {
        1 => m[0],
        2 => m[0] * m[3] - m[1] * m[2],
        3 => {
            m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
        }
        _ => unreachable!("det_small: only n ∈ {{1,2,3}} supported"),
    }
}

/// Invert a small (1×1, 2×2, 3×3) row-major square matrix.
///
/// Returns an error if the matrix is (numerically) singular.
fn inverse_small(m: &[f64], n: usize) -> Result<Vec<f64>> {
    let det = det_small(m, n);
    if det.abs() < f64::EPSILON {
        return Err(PyrucastError::Message(
            "inverse_small: singular matrix".into(),
        ));
    }
    let inv = match n {
        1 => vec![1.0 / m[0]],
        2 => {
            let d = det;
            vec![m[3] / d, -m[1] / d, -m[2] / d, m[0] / d]
        }
        3 => {
            let d = det;
            vec![
                (m[4] * m[8] - m[5] * m[7]) / d,
                (m[2] * m[7] - m[1] * m[8]) / d,
                (m[1] * m[5] - m[2] * m[4]) / d,
                (m[5] * m[6] - m[3] * m[8]) / d,
                (m[0] * m[8] - m[2] * m[6]) / d,
                (m[2] * m[3] - m[0] * m[5]) / d,
                (m[3] * m[7] - m[4] * m[6]) / d,
                (m[1] * m[6] - m[0] * m[7]) / d,
                (m[0] * m[4] - m[1] * m[3]) / d,
            ]
        }
        _ => unreachable!(),
    };
    Ok(inv)
}

/// Compute `dN_i/dx_a` from the Jacobian and reference derivatives.
///
/// Uses the unified formula `M = J · (JᵀJ)⁻¹`, which collapses to
/// `J⁻ᵀ` when `J` is square. Output layout: `[i * space_dim + a]`.
fn build_dn_dx(
    jac: &[f64],
    dn_dxi: &[f64],
    space_dim: usize,
    ref_dim: usize,
    n_nodes: usize,
) -> Result<Vec<f64>> {
    let g = gram_matrix(jac, space_dim, ref_dim);
    let g_inv = inverse_small(&g, ref_dim)?;

    // M[a*ref_dim + l] = Σ_k J[a*ref_dim + k] · G_inv[k*ref_dim + l]
    let mut m = vec![0.0; space_dim * ref_dim];
    for a in 0..space_dim {
        for l in 0..ref_dim {
            let mut s = 0.0;
            for k in 0..ref_dim {
                s += jac[a * ref_dim + k] * g_inv[k * ref_dim + l];
            }
            m[a * ref_dim + l] = s;
        }
    }

    // dN/dx[i*space_dim + a] = Σ_l M[a*ref_dim + l] · dN/dξ[i*ref_dim + l]
    let mut out = vec![0.0; n_nodes * space_dim];
    for i in 0..n_nodes {
        for a in 0..space_dim {
            let mut s = 0.0;
            for l in 0..ref_dim {
                s += m[a * ref_dim + l] * dn_dxi[i * ref_dim + l];
            }
            out[i * space_dim + a] = s;
        }
    }
    Ok(out)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::containers::mesh::Node;
    use crate::store::{insert, with};

    fn cfg2d() -> Handle<Configuration> {
        insert(Configuration::new(2).unwrap())
    }

    fn cfg3d() -> Handle<Configuration> {
        insert(Configuration::new(3).unwrap())
    }

    // ── SubFiniteElementSpace structural checks ────────────────────────────────────────

    #[test]
    fn rejects_poi1_submesh() {
        let cfg = cfg2d();
        let n = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[n.id()]).unwrap();
            insert(sm)
        };
        let err = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn rejects_mesh_with_lower_space_dim_than_ref_dim() {
        // 1-D Configuration but TRI3 (ref_dim = 2) → must be rejected.
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        assert!(SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).is_err());
    }

    // ── Jacobian: closed-form checks ────────────────────────────────────────

    /// SEG2 of length L in 1-D: J is constant, |J| = L/2.
    #[test]
    fn seg2_jacobian_1d() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[5.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        for g in 0..sub.gauss_count() {
            let jac = sub.jacobian(0, g).unwrap();
            assert_eq!(jac.len(), 1);
            assert!((jac[0] - 2.5).abs() < 1e-12);
            assert!((sub.det_jacobian(0, g).unwrap() - 2.5).abs() < 1e-12);
        }
    }

    /// SEG2 of length 3 in 2-D (line in the x-direction): the Jacobian is
    /// a 2×1 column [3/2, 0]; |J|_curve = 3/2.
    #[test]
    fn seg2_jacobian_in_plane() {
        let cfg = cfg2d();
        let a = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[3.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        for g in 0..sub.gauss_count() {
            let jac = sub.jacobian(0, g).unwrap();
            assert_eq!(jac.len(), 2);
            assert!((jac[0] - 1.5).abs() < 1e-12); // dx/dξ
            assert!(jac[1].abs() < 1e-12); // dy/dξ
            assert!((sub.det_jacobian(0, g).unwrap() - 1.5).abs() < 1e-12);
        }
    }

    /// TRI3 of vertices (0,0), (a,0), (0,b): |J| = a·b (twice the area,
    /// since ref triangle has area 1/2).
    #[test]
    fn tri3_jacobian_planar() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[3.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 4.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        for g in 0..sub.gauss_count() {
            let dj = sub.det_jacobian(0, g).unwrap();
            assert!((dj - 12.0).abs() < 1e-12);
        }
    }

    /// TRI3 living in 3-D (xy-plane): same |J| as the planar case
    /// (manifold area element).
    #[test]
    fn tri3_jacobian_manifold_in_3d() {
        let cfg = cfg3d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0, 7.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[3.0, 0.0, 7.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 4.0, 7.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        for g in 0..sub.gauss_count() {
            let dj = sub.det_jacobian(0, g).unwrap();
            assert!((dj - 12.0).abs() < 1e-12);
        }
    }

    /// QUA4 unit square aligned with axes: |J| = 1/4 at every Gauss point
    /// (∂x/∂ξ = ∂y/∂η = 1/2, cross-terms 0). Twice the centroid area? No:
    /// for [0,1]² with reference [-1,1]², |J| = 1/4 because each ref-unit
    /// maps to half a physical unit.
    #[test]
    fn qua4_jacobian_unit_square() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        // Integral of |J| over reference square should equal physical area = 1.
        let mut area = 0.0;
        for g in 0..sub.gauss_count() {
            area += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((area - 1.0).abs() < 1e-12);
    }

    /// HEX8 unit cube: similar to QUA4 test in 3-D.
    #[test]
    fn hex8_jacobian_unit_cube() {
        let cfg = cfg3d();
        let n: Vec<_> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(cfg.clone(), p).unwrap())
        .collect();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::HEX8);
            sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>()).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        let mut vol = 0.0;
        for g in 0..sub.gauss_count() {
            vol += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((vol - 1.0).abs() < 1e-12);
    }

    // ── dN/dx ───────────────────────────────────────────────────────────────

    /// On a TRI3 with vertices (0,0), (3,0), (0,4), the gradient of N_1 is
    /// `(-1/3, -1/4)` (constant across the cell, since Lagrange-1).
    #[test]
    fn tri3_dn_dx_constant() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[3.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 4.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();
        for g in 0..sub.gauss_count() {
            let dn = sub.dn_dx(0, g).unwrap();
            // ∂N_1/∂x = -1/3, ∂N_1/∂y = -1/4
            assert!((dn[0] - (-1.0 / 3.0)).abs() < 1e-12);
            assert!((dn[1] - (-1.0 / 4.0)).abs() < 1e-12);
            // ∂N_2/∂x = 1/3, ∂N_2/∂y = 0
            assert!((dn[2] - (1.0 / 3.0)).abs() < 1e-12);
            assert!(dn[3].abs() < 1e-12);
            // ∂N_3/∂x = 0, ∂N_3/∂y = 1/4
            assert!(dn[4].abs() < 1e-12);
            assert!((dn[5] - (1.0 / 4.0)).abs() < 1e-12);
        }
    }

    // ── On-the-fly under deformation ────────────────────────────────────────

    /// After moving a node, the on-the-fly Jacobian must reflect the new
    /// coordinates — the FE space caches nothing physical.
    #[test]
    fn jacobian_reflects_mesh_displacement() {
        let cfg = cfg2d();
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss).unwrap();

        let dj_before = sub.det_jacobian(0, 0).unwrap();
        assert!((dj_before - 0.5).abs() < 1e-12);

        // Stretch the SEG2 to length 4 (move node b from x=1 to x=4).
        crate::store::with_mut(&cfg, |c| c.set_coord(b.id(), &[4.0, 0.0])).unwrap().unwrap();

        let dj_after = sub.det_jacobian(0, 0).unwrap();
        assert!((dj_after - 2.0).abs() < 1e-12);
    }

    // ── FiniteElementSpace ──────────────────────────────────────────────────

    #[test]
    fn lagrange1_constructor_matches_submeshes_one_to_one() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(cfg.clone(), &[1.0, 1.0]).unwrap();

        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_qua).unwrap();

        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert_eq!(fes.subspace_count(), 2);
        with(&fes.subspace(0).unwrap(), |s| {
            assert_eq!(s.element_type().unwrap(), ElementType::TRI3);
            assert_eq!(s.gauss_count(), 3);
        })
        .unwrap();
        with(&fes.subspace(1).unwrap(), |s| {
            assert_eq!(s.element_type().unwrap(), ElementType::QUA4);
            assert_eq!(s.gauss_count(), 4);
        })
        .unwrap();
    }

    #[test]
    fn rejects_mesh_with_poi1_submesh() {
        let cfg = cfg2d();
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let mesh = crate::ops::mesher::from_live_nodes(cfg).unwrap();
        // from_live_nodes builds a POI1 mesh.
        assert!(mesh.submesh_count() >= 1);
        let _ = a; // keep alive
        assert!(FiniteElementSpace::lagrange1(&mesh).is_err());
    }

    #[test]
    fn rejects_empty_mesh() {
        let mesh = Mesh::empty();
        assert!(FiniteElementSpace::lagrange1(&mesh).is_err());
    }

    #[test]
    fn with_constructor_validates_length() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
        mesh.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        let too_few: Vec<(Interpolation, QuadratureRule)> = vec![];
        assert!(FiniteElementSpace::with(&mesh, &too_few).is_err());
        let too_many = vec![
            (Interpolation::Lagrange1, QuadratureRule::Gauss),
            (Interpolation::Lagrange1, QuadratureRule::Gauss),
        ];
        assert!(FiniteElementSpace::with(&mesh, &too_many).is_err());
    }

    // ── Display ─────────────────────────────────────────────────────────────

    #[test]
    fn display_and_debug() {
        let cfg = cfg2d();
        let n0 = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
        mesh.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let s = format!("{}", fes);
        assert!(s.contains("FiniteElementSpace"));
        assert!(s.contains("1 subspace"));
        let d = format!("{:?}", fes);
        assert!(d.contains("FiniteElementSpace"));

        with(&fes.subspace(0).unwrap(), |sub| {
            let s = format!("{}", sub);
            assert!(s.contains("SubFiniteElementSpace"));
            assert!(s.contains("TRI3"));
            assert!(s.contains("LAGRANGE1"));
            assert!(s.contains("GAUSS"));
        })
        .unwrap();
    }
}
