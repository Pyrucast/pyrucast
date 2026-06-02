//! `Element` — lightweight view on a single element of a
//! [`SubFiniteElementSpace`].
//!
//! It is to [`SubFiniteElementSpace`] what [`crate::containers::mesh::Cell`]
//! is to [`crate::containers::mesh::SubMesh`] : a `(handle, cell_idx)`
//! pair that exposes the FE quantities (shape functions, Jacobian,
//! physical derivatives, …) for a single cell. Cloning an `Element` is
//! an `Arc` clone, so it is cheap to pass around and to create on the
//! fly inside an iterator.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::finite_element_space::FiniteElementSpace;
//! use pyrucast::containers::mesh::{Mesh, SubMesh};
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::insert;
//!
//! let cfg = insert(Configuration::new(1).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[2.0]).unwrap();
//! let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
//! mesh.add_cell(&[a.id(), b.id()]).unwrap();
//! let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
//!
//! for el in fes.elements(0).unwrap() {
//!     // |J| of a SEG2 of length 2 in 1-D is constant = 1.
//!     for g in 0..el.gauss_count() {
//!         assert!((el.det_jacobian(g).unwrap() - 1.0).abs() < 1e-12);
//!     }
//! }
//! ```

use std::fmt;

use crate::containers::mesh::NodeId;
use crate::containers::mesh::Cell;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::error::{PyrucastError, Result};
use crate::store::{with, Handle};

/// Lightweight view on a single element of a [`SubFiniteElementSpace`].
#[derive(Clone)]
pub struct Element {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    idx: usize,
}

impl Element {
    /// Build an element view. Errors if `idx ≥ cell_count()`.
    pub fn new(fespace: Handle<SubFiniteElementSpace>, idx: usize) -> Result<Self> {
        let n = with(&fespace, |s| s.cell_count())??;
        if idx >= n {
            return Err(PyrucastError::Message(format!(
                "element index {idx} out of range (cell_count={n})"
            )));
        }
        Ok(Self { fespace, idx })
    }

    /// Index of this element inside its parent [`SubFiniteElementSpace`].
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Handle to the parent [`SubFiniteElementSpace`] (internal clone).
    pub fn fespace(&self) -> Handle<SubFiniteElementSpace> {
        self.fespace.clone()
    }

    /// Underlying [`Cell`] view in the parent submesh.
    pub fn cell(&self) -> Result<Cell> {
        let sm = with(&self.fespace, |s| s.submesh())?;
        Cell::new(sm, self.idx)
    }

    // ── Structural accessors ────────────────────────────────────────────

    /// Number of nodes per element (= element type's nodes_per_cell).
    pub fn nodes_per_cell(&self) -> Result<usize> {
        with(&self.fespace, |s| s.nodes_per_cell())?
    }

    /// Geometric (physical) dimension of the underlying `Configuration`.
    pub fn space_dim(&self) -> Result<usize> {
        with(&self.fespace, |s| s.space_dim())
    }

    /// Reference dimension (= topological dim of the element type).
    pub fn ref_dim(&self) -> Result<usize> {
        with(&self.fespace, |s| s.ref_dim())?
    }

    /// Number of Gauss points per element.
    pub fn gauss_count(&self) -> usize {
        // Static across the fespace; cheap to look up via a `with`.
        with(&self.fespace, |s| s.gauss_count()).unwrap_or(0)
    }

    /// Connectivity (node ids) of this element.
    pub fn node_ids(&self) -> Result<Vec<NodeId>> {
        self.cell()?.node_ids()
    }

    // ── Reference-space accessors (shared with the FE space) ────────────

    /// Reference coordinates of the `g`-th Gauss point.
    pub fn gauss_xi(&self, g: usize) -> Result<Vec<f64>> {
        with(&self.fespace, |s| Ok(s.gauss_xi(g)?.to_vec()))?
    }

    /// Weight of the `g`-th Gauss point.
    pub fn gauss_weight(&self, g: usize) -> Result<f64> {
        with(&self.fespace, |s| s.gauss_weight(g))?
    }

    /// `N_i(ξ_g)` for all nodes `i` at the `g`-th Gauss point.
    pub fn n_at_g(&self, g: usize) -> Result<Vec<f64>> {
        with(&self.fespace, |s| Ok(s.n_at_g(g)?.to_vec()))?
    }

    /// `∂N_i/∂ξ_j(ξ_g)` for all nodes at the `g`-th Gauss point
    /// (flat row-major `[i * ref_dim + j]`).
    pub fn dn_at_g(&self, g: usize) -> Result<Vec<f64>> {
        with(&self.fespace, |s| Ok(s.dn_at_g(g)?.to_vec()))?
    }

    // ── Physical quantities (cell-specific, on-the-fly) ─────────────────

    /// Jacobian `J = ∂x/∂ξ` at the `g`-th Gauss point of this element.
    pub fn jacobian(&self, g: usize) -> Result<Vec<f64>> {
        with(&self.fespace, |s| s.jacobian(self.idx, g))?
    }

    /// `|J|` (measure scaling factor) at the `g`-th Gauss point.
    pub fn det_jacobian(&self, g: usize) -> Result<f64> {
        with(&self.fespace, |s| s.det_jacobian(self.idx, g))?
    }

    /// Physical derivatives `∂N_i/∂x_a` at the `g`-th Gauss point
    /// (flat row-major `[i * space_dim + a]`).
    pub fn dn_dx(&self, g: usize) -> Result<Vec<f64>> {
        with(&self.fespace, |s| s.dn_dx(self.idx, g))?
    }
}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Element").field("idx", &self.idx).finish()
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cell() {
            Ok(cell) => {
                let et = cell.element_type().map(|e| e.name()).unwrap_or("?");
                write!(f, "Element<{}> #{}", et, self.idx)
            }
            Err(_) => write!(f, "Element #{}", self.idx),
        }
    }
}

impl crate::dump::Dump for Element {
    fn dump_with(&self, opts: &crate::dump::DumpOptions) -> String {
        match self.cell() {
            Ok(cell) => format!(
                "Element #{} ({} Gauss point(s))\n{}",
                self.idx,
                self.gauss_count(),
                crate::dump::Dump::dump_with(&cell, opts)
            ),
            Err(e) => format!("Element #{} <{e}>", self.idx),
        }
    }
}

/// Iterator over the elements of a single [`SubFiniteElementSpace`].
#[derive(Clone)]
pub struct ElementIter {
    fespace: Handle<SubFiniteElementSpace>,
    next: usize,
    end: usize,
}

impl ElementIter {
    pub(crate) fn new(fespace: Handle<SubFiniteElementSpace>, end: usize) -> Self {
        Self { fespace, next: 0, end }
    }
}

impl Iterator for ElementIter {
    type Item = Element;
    fn next(&mut self) -> Option<Element> {
        if self.next < self.end {
            let el = Element {
                fespace: self.fespace.clone(),
                idx: self.next,
            };
            self.next += 1;
            Some(el)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ElementIter {}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::mesh::Configuration;
    use crate::containers::mesh::ElementType;
    use crate::containers::mesh::Node;
    use crate::store::insert;

    fn seg2_fes() -> (Handle<Configuration>, Vec<Node>, FiniteElementSpace) {
        let cfg = insert(Configuration::new(1).unwrap());
        let n0 = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::SEG2));
        mesh.add_cell(&[n0.id(), n1.id()]).unwrap();
        mesh.add_cell(&[n1.id(), n2.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        (cfg, vec![n0, n1, n2], fes)
    }

    #[test]
    fn element_exposes_cell_and_physical_quantities() {
        let (_cfg, nodes, fes) = seg2_fes();
        let el = fes.element(0, 0).unwrap();

        assert_eq!(el.index(), 0);
        assert_eq!(el.nodes_per_cell().unwrap(), 2);
        assert_eq!(el.node_ids().unwrap(), vec![nodes[0].id(), nodes[1].id()]);

        let cell = el.cell().unwrap();
        assert_eq!(cell.element_type().unwrap(), ElementType::SEG2);

        for g in 0..el.gauss_count() {
            // SEG2 of length 1 in 1-D → |J| = 0.5
            assert!((el.det_jacobian(g).unwrap() - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn elements_iterator_yields_all_elements_in_order() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let elements: Vec<Element> = fes.elements(0).unwrap().collect();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].index(), 0);
        assert_eq!(elements[1].index(), 1);
    }

    #[test]
    fn element_new_out_of_range_errors() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let sub = fes.subspace(0).unwrap();
        assert!(Element::new(sub, 5).is_err());
    }

    #[test]
    fn fespace_element_by_indices() {
        let (_cfg, _nodes, fes) = seg2_fes();
        let el = fes.element(0, 1).unwrap();
        assert_eq!(el.index(), 1);
        // submesh out of bounds
        assert!(fes.element(1, 0).is_err());
        // cell out of bounds
        assert!(fes.element(0, 7).is_err());
    }
}
