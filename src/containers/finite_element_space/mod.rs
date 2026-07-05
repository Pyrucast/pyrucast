//! Finite-element space — interpolation + quadrature layer on top of a mesh.
//!
//! Hierarchy mirroring [`crate::containers::mesh`]:
//!
//! - [`SubFiniteElementSpace`] — one [`crate::containers::finite_element_space::Interpolation`] and one
//!   [`crate::containers::finite_element_space::QuadratureRule`] applied to a single
//!   [`crate::containers::mesh::SubMesh`]. It stores the **reference-space tables**
//!   that do not depend on the physical coordinates of the nodes
//!   (Gauss points and weights, shape functions and reference
//!   derivatives at those Gauss points) and computes the physical
//!   quantities — Jacobian, `|J|`, `dN/dx` — **on the fly** from the
//!   current node coordinates in the
//!   [`crate::containers::mesh::Coords`].
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
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, read};
//!
//! let coords = insert(Coords::new(2).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
//! let c = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();
//!
//! let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//! let sub = fes.get(0).unwrap();
//! let s = read(&sub).unwrap();
//! assert_eq!(s.gauss_count(), 3);
//! // |J| of a triangle with vertices (0,0), (2,0), (0,2): the mapping
//! // is linear, |J| = 4 (twice the area, since ref triangle has area 1/2).
//! for g in 0..s.gauss_count() {
//!     let dj = s.det_jacobian(0, g).unwrap();
//!     assert!((dj - 4.0).abs() < 1e-12);
//! }
//! ```

pub mod element;
pub mod interpolation;
pub mod quadrature;

pub use element::{Element, ElementIter};
pub use interpolation::Interpolation;
pub use quadrature::QuadratureRule;

use crate::aggregate::Aggregate;
use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Coords, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

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
    /// Geometric dimension of the owning `Coords` at construction
    /// time (used only to size the on-the-fly Jacobians). The
    /// `Coords` may not change dimension, so this is stable for
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

    /// Cell colouring for conflict-free parallel assembly, computed once and
    /// memoised. Topological (depends only on the frozen connectivity), so it is
    /// invariant for the subspace's lifetime. Not serialised — recomputed after
    /// a load, like the lazy index maps on `SubMatrix`.
    #[serde(skip)]
    coloring: OnceLock<Vec<Vec<usize>>>,
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
        let (et, coords) = {
            let s = read(&submesh)?;
            (s.element_type(), s.coords())
        };
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
        let space_dim = read(&coords)?.dim() as usize;
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

        // Capturing the submesh in a finite-element space freezes its
        // connectivity: the reference-space tables and any assembled matrix
        // are cell-indexed and must not be invalidated by later `add_cell`s.
        let submesh = crate::containers::mesh::seal(&submesh)?;

        Ok(Self {
            submesh,
            interpolation,
            quadrature,
            space_dim,
            gauss_xi,
            gauss_w,
            n_at_g,
            dn_at_g,
            coloring: OnceLock::new(),
        })
    }

    /// Cell colouring for conflict-free parallel assembly, computed **once** and
    /// cached. Two cells of the same colour share no key, so their element
    /// matrices scatter into the global matrix in parallel without conflict.
    ///
    /// The `compute` closure is supplied by the caller (the assembly `ops`), so
    /// this container layer stays free of any assembly dependency and free to
    /// choose the conflict keys (cell nodes today; global/master DOFs once MPC
    /// condensation lands). It runs at most once per subspace.
    pub fn coloring(&self, compute: impl FnOnce() -> Vec<Vec<usize>>) -> &[Vec<usize>] {
        self.coloring.get_or_init(compute)
    }

    // ── Accessors (structural) ──────────────────────────────────────────────

    /// Handle to the underlying submesh (internal clone).
    pub fn submesh(&self) -> Handle<SubMesh> {
        self.submesh.clone()
    }

    /// Handle to the owning `Coords` (internal clone).
    pub fn coords(&self) -> Result<Handle<Coords>> {
        Ok(read(&self.submesh)?.coords())
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
        Ok(read(&self.submesh)?.element_type())
    }

    /// Reference dimension (= topological dim of the element type).
    pub fn ref_dim(&self) -> Result<usize> {
        Ok(self.element_type()?.topological_dim())
    }

    /// Geometric (physical) dimension of the underlying `Coords`.
    pub fn space_dim(&self) -> usize {
        self.space_dim
    }

    /// Number of nodes per cell (= `element_type().nodes_per_cell()`).
    pub fn nodes_per_cell(&self) -> Result<usize> {
        Ok(self.element_type()?.nodes_per_cell())
    }

    /// Number of cells in the underlying submesh.
    pub fn cell_count(&self) -> Result<usize> {
        Ok(read(&self.submesh)?.cell_count())
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
    /// **current** node coordinates in the `Coords`.
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
        let (coords, ids): (Handle<Coords>, Vec<NodeId>) = {
            let s = read(&self.submesh)?;
            let total = s.cell_count();
            if cell_idx >= total {
                return Err(PyrucastError::Message(format!(
                    "SubFiniteElementSpace: cell index {} ≥ cell_count {}",
                    cell_idx, total
                )));
            }
            let conn = s.connectivity();
            let ids = conn[cell_idx * n_nodes..(cell_idx + 1) * n_nodes].to_vec();
            (s.coords(), ids)
        };
        let mut out = Vec::with_capacity(n_nodes * self.space_dim);
        let c = read(&coords)?;
        for id in ids {
            out.extend_from_slice(c.coord(id)?);
        }
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
        let et = self.element_type().map(|e| e.name()).unwrap_or("?");
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
        let ref_dim = self.gauss_xi.len().checked_div(ng).unwrap_or(0);
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
/// in the underlying `Coords` may evolve later; the on-the-fly
/// Jacobian computation always reflects the current coordinates.
#[derive(Serialize, Deserialize, Default)]
pub struct FiniteElementSpace {
    subs: Vec<Handle<SubFiniteElementSpace>>,
}

crate::impl_aggregate!(
    FiniteElementSpace,
    SubFiniteElementSpace,
    subspace,
    "subspace(s)"
);
crate::impl_aggregate_dump!(FiniteElementSpace);

impl FiniteElementSpace {
    /// Build a `FiniteElementSpace` by attaching the supplied
    /// `(interpolation, quadrature)` pair to each submesh of `mesh`, in
    /// order.
    ///
    /// `choices.len()` must equal `mesh.len()`. The mesh must
    /// have at least one submesh and none of them may be POI1.
    pub fn with(mesh: &Mesh, choices: &[(Interpolation, QuadratureRule)]) -> Result<Self> {
        let n_sub = mesh.len();
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
            let sm = mesh.get(i)?;
            let sub = SubFiniteElementSpace::new(sm, interp, quad)?;
            subs.push(insert(sub));
        }
        Ok(Self { subs })
    }

    /// Build a `FiniteElementSpace` using the same `interpolation` for
    /// every submesh, with the default Gauss quadrature.
    pub fn new(mesh: &Mesh, interpolation: Interpolation) -> Result<Self> {
        let n_sub = mesh.len();
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
        let sub = self.get(subspace_idx)?;
        Element::new(sub, cell_idx)
    }

    /// Iterator over every element of subspace `subspace_idx`.
    pub fn elements(&self, subspace_idx: usize) -> Result<ElementIter> {
        let sub = self.get(subspace_idx)?;
        let n = read(&sub)?.cell_count()?;
        Ok(ElementIter::new(sub, n))
    }
}

// ─── Numerical helpers ─────────────────────────────────────────────────────

/// Build the Jacobian `J[a*ref_dim + k] = Σ_i x_i[a] · dN_i/dξ_k`.
///
/// `coords` has layout `[i * space_dim + a]`, `dn_dxi` has layout
/// `[i * ref_dim + k]`.
pub(crate) fn build_jacobian(
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
pub(crate) fn jacobian_measure(jac: &[f64], space_dim: usize, ref_dim: usize) -> f64 {
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
            m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
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
pub(crate) fn build_dn_dx(
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
    use crate::store::{insert, read, write};

    fn cfg2d() -> Handle<Coords> {
        insert(Coords::new(2).unwrap())
    }

    fn cfg3d() -> Handle<Coords> {
        insert(Coords::new(3).unwrap())
    }

    #[test]
    fn coloring_is_memoised() {
        let coords = cfg2d();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
        let first = sub.coloring(|| vec![vec![0usize]]).to_vec();
        // The second closure is never run: the memoised value is returned.
        let second = sub
            .coloring(|| panic!("compute must not run twice"))
            .to_vec();
        assert_eq!(first, vec![vec![0usize]]);
        assert_eq!(first, second);
    }

    #[test]
    fn new_seals_the_submesh() {
        let coords = cfg2d();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        assert!(!read(&sm).unwrap().is_sealed());
        let _sub =
            SubFiniteElementSpace::new(sm.clone(), Interpolation::Lagrange1, QuadratureRule::Gauss)
                .unwrap();
        // Building the space froze the submesh: no more cells can be added.
        assert!(read(&sm).unwrap().is_sealed());
        assert!(matches!(
            write(&sm)
                .unwrap()
                .add_cell(&[a.id(), b.id(), c.id()])
                .unwrap_err(),
            PyrucastError::MeshSealed
        ));
    }

    // ── SubFiniteElementSpace structural checks ────────────────────────────────────────

    #[test]
    fn rejects_poi1_submesh() {
        let coords = cfg2d();
        let n = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[n.id()]).unwrap();
            insert(sm)
        };
        let err = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn rejects_mesh_with_lower_space_dim_than_ref_dim() {
        // 1-D Coords but TRI3 (ref_dim = 2) → must be rejected.
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        assert!(
            SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
                .is_err()
        );
    }

    // ── Jacobian: closed-form checks ────────────────────────────────────────

    /// SEG2 of length L in 1-D: J is constant, |J| = L/2.
    #[test]
    fn seg2_jacobian_1d() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[5.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
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
        let coords = cfg2d();
        let a = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[3.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
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
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[3.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 4.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
        for g in 0..sub.gauss_count() {
            let dj = sub.det_jacobian(0, g).unwrap();
            assert!((dj - 12.0).abs() < 1e-12);
        }
    }

    /// TRI3 living in 3-D (xy-plane): same |J| as the planar case
    /// (manifold area element).
    #[test]
    fn tri3_jacobian_manifold_in_3d() {
        let coords = cfg3d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0, 7.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[3.0, 0.0, 7.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 4.0, 7.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
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
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n2.id(), n3.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
        // Integral of |J| over reference square should equal physical area = 1.
        let mut area = 0.0;
        for g in 0..sub.gauss_count() {
            area += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((area - 1.0).abs() < 1e-12);
    }

    /// QUA9 biquadratic on a straight unit square (mid-edge and center nodes
    /// at their natural positions): area = 1. Exercises the full Lagrange-2
    /// path for the 9-node quad.
    #[test]
    fn qua9_jacobian_unit_square() {
        let coords = cfg2d();
        // Corners, mid-edges, then center — all at their geometric positions.
        let pts: [[f64; 2]; 9] = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ];
        let ids: Vec<_> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::QUA9);
            sm.add_cell(&ids).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange2, QuadratureRule::Gauss)
            .unwrap();
        let mut area = 0.0;
        for g in 0..sub.gauss_count() {
            area += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((area - 4.0).abs() < 1e-12, "QUA9 area = {area}");
    }

    /// PENTA6 prism over the unit right triangle, height 2: its volume is
    /// `area(1/2) × height(2) = 1`. Exercises the full physical Jacobian /
    /// quadrature path for the prism, not just the reference rule.
    #[test]
    fn penta6_jacobian_unit_volume() {
        let coords = cfg3d();
        let n: Vec<_> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::PENTA6);
            sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
                .unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
        let mut vol = 0.0;
        for g in 0..sub.gauss_count() {
            vol += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((vol - 1.0).abs() < 1e-12, "prism volume = {vol}");
    }

    /// Build a straight-edged quadratic element (mid-edge nodes placed at the
    /// exact edge midpoints, so the geometry map is affine) from its corner
    /// coordinates and integrate the constant 1 over it — this exercises the
    /// full Lagrange-2 shape/derivative/Jacobian/quadrature path.
    fn quad_element_volume(et: ElementType, corners: &[[f64; 3]], edges: &[(usize, usize)]) -> f64 {
        let coords = cfg3d();
        let mut pts: Vec<[f64; 3]> = corners.to_vec();
        for &(a, b) in edges {
            pts.push([
                0.5 * (corners[a][0] + corners[b][0]),
                0.5 * (corners[a][1] + corners[b][1]),
                0.5 * (corners[a][2] + corners[b][2]),
            ]);
        }
        let ids: Vec<_> = pts
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, et);
            sm.add_cell(&ids).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange2, QuadratureRule::Gauss)
            .unwrap();
        (0..sub.gauss_count())
            .map(|g| sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap())
            .sum()
    }

    #[test]
    fn tet10_jacobian_volume() {
        // Straight tetra (0,0,0),(2,0,0),(0,3,0),(0,0,4): volume 24/6 = 4.
        let vol = quad_element_volume(
            ElementType::TET10,
            &[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            &[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
        );
        assert!((vol - 4.0).abs() < 1e-12, "TET10 volume = {vol}");
    }

    #[test]
    fn hex20_jacobian_unit_cube() {
        let vol = quad_element_volume(
            ElementType::HEX20,
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
        );
        assert!((vol - 1.0).abs() < 1e-12, "HEX20 volume = {vol}");
    }

    #[test]
    fn hex27_jacobian_cube() {
        // Straight [0,2]³ cube: all 27 nodes at their natural tensor
        // positions (corners, edges, faces, center), volume 8.
        let coords = cfg3d();
        // Reference layout in {-1,0,1}³ (pyrucast HEX27 order), mapped to
        // physical [0,2]³ via x = ξ + 1.
        let refc: [(f64, f64, f64); 27] = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (0.0, -1.0, -1.0),
            (1.0, 0.0, -1.0),
            (0.0, 1.0, -1.0),
            (-1.0, 0.0, -1.0),
            (0.0, -1.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        ];
        let ids: Vec<_> = refc
            .iter()
            .map(|&(x, y, z)| {
                Node::create_in(coords.clone(), &[x + 1.0, y + 1.0, z + 1.0])
                    .unwrap()
                    .id()
            })
            .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::HEX27);
            sm.add_cell(&ids).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange2, QuadratureRule::Gauss)
            .unwrap();
        let mut vol = 0.0;
        for g in 0..sub.gauss_count() {
            vol += sub.gauss_weight(g).unwrap() * sub.det_jacobian(0, g).unwrap();
        }
        assert!((vol - 8.0).abs() < 1e-12, "HEX27 volume = {vol}");
    }

    #[test]
    fn penta15_jacobian_volume() {
        // Prism over the unit right triangle, height 2: volume 1/2 × 2 = 1.
        let vol = quad_element_volume(
            ElementType::PENTA15,
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            &[
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3),
                (0, 3),
                (1, 4),
                (2, 5),
            ],
        );
        assert!((vol - 1.0).abs() < 1e-12, "PENTA15 volume = {vol}");
    }

    /// HEX8 unit cube: similar to QUA4 test in 3-D.
    #[test]
    fn hex8_jacobian_unit_cube() {
        let coords = cfg3d();
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
        .map(|p| Node::create_in(coords.clone(), p).unwrap())
        .collect();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::HEX8);
            sm.add_cell(&n.iter().map(|x| x.id()).collect::<Vec<_>>())
                .unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
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
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[3.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 4.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords, ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();
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
        let coords = cfg2d();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            insert(sm)
        };
        let sub = SubFiniteElementSpace::new(sm, Interpolation::Lagrange1, QuadratureRule::Gauss)
            .unwrap();

        let dj_before = sub.det_jacobian(0, 0).unwrap();
        assert!((dj_before - 0.5).abs() < 1e-12);

        // Stretch the SEG2 to length 4 (move node b from x=1 to x=4).
        write(&coords)
            .unwrap()
            .set_coord(b.id(), &[4.0, 0.0])
            .unwrap();

        let dj_after = sub.det_jacobian(0, 0).unwrap();
        assert!((dj_after - 2.0).abs() < 1e-12);
    }

    // ── FiniteElementSpace ──────────────────────────────────────────────────

    #[test]
    fn lagrange1_constructor_matches_submeshes_one_to_one() {
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();

        let mut mesh = Mesh::empty();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        let sm_qua = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
            sm.add_cell(&[n0.id(), n1.id(), n3.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_tri).unwrap();
        mesh.add_sub(sm_qua).unwrap();

        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert_eq!(fes.len(), 2);
        {
            let s = read(&fes.get(0).unwrap()).unwrap();
            assert_eq!(s.element_type().unwrap(), ElementType::TRI3);
            assert_eq!(s.gauss_count(), 3);
        }
        {
            let s = read(&fes.get(1).unwrap()).unwrap();
            assert_eq!(s.element_type().unwrap(), ElementType::QUA4);
            assert_eq!(s.gauss_count(), 4);
        }
    }

    #[test]
    fn rejects_mesh_with_poi1_submesh() {
        let coords = cfg2d();
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mesh = crate::ops::mesher::from_live_nodes(coords).unwrap();
        // from_live_nodes builds a POI1 mesh.
        assert!(mesh.len() >= 1);
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
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
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
        let coords = cfg2d();
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
        mesh.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let s = format!("{}", fes);
        assert!(s.contains("FiniteElementSpace"));
        assert!(s.contains("1 subspace"));
        let d = format!("{:?}", fes);
        assert!(d.contains("FiniteElementSpace"));

        {
            let sub = read(&fes.get(0).unwrap()).unwrap();
            let s = format!("{}", &*sub);
            assert!(s.contains("SubFiniteElementSpace"));
            assert!(s.contains("TRI3"));
            assert!(s.contains("LAGRANGE1"));
            assert!(s.contains("GAUSS"));
        }
    }
}
