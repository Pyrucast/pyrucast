//! A `Cell` is a lightweight view on a single cell of a [`SubMesh`].
//!
//! It carries a cloned `Handle<SubMesh>` plus the cell's index — cloning
//! a `Cell` is just an `Arc` clone, so it is cheap to pass around and to
//! create on the fly inside an iterator. The actual node coordinates live
//! in the `Configuration` and are fetched on demand through
//! [`Cell::nodes`] / [`Cell::node_ids`].
//!
//! # Example
//!
//! ```
//! use pyrucast::configuration::Configuration;
//! use pyrucast::element_type::ElementType;
//! use pyrucast::mesh::{Mesh, SubMesh};
//! use pyrucast::node::Node;
//! use pyrucast::store::insert;
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut mesh = Mesh::with_element_type(cfg, ElementType::TRI3);
//! mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//!
//! let cell = mesh.cell(0, 0).unwrap();
//! assert_eq!(cell.nodes().unwrap().len(), 3);
//! for node in cell.nodes().unwrap() {
//!     let _ = node.coord().unwrap();
//! }
//! ```

use std::fmt;

use crate::configuration::NodeId;
use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::mesh::SubMesh;
use crate::node::Node;
use crate::store::{with, Handle};

/// Lightweight view on a single cell of a `SubMesh`.
#[derive(Clone)]
pub struct Cell {
    pub(crate) sm: Handle<SubMesh>,
    idx: usize,
}

impl Cell {
    /// Build a cell view. Errors if `idx` is past the submesh's
    /// `cell_count`.
    pub fn new(sm: Handle<SubMesh>, idx: usize) -> Result<Self> {
        let n = with(&sm, |s| s.cell_count())?;
        if idx >= n {
            return Err(PyrucastError::Message(format!(
                "cell index {idx} out of range (cell_count={n})"
            )));
        }
        Ok(Self { sm, idx })
    }

    /// Index of this cell inside its parent submesh.
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Element type of this cell (same as the parent submesh).
    pub fn element_type(&self) -> Result<ElementType> {
        with(&self.sm, |s| s.element_type())
    }

    /// Number of nodes that make up this cell (= `element_type().nodes_per_cell()`).
    pub fn nodes_per_cell(&self) -> Result<usize> {
        with(&self.sm, |s| s.element_type().nodes_per_cell())
    }

    /// Raw connectivity (node ids) of this cell, in submesh order.
    pub fn node_ids(&self) -> Result<Vec<NodeId>> {
        with(&self.sm, |s| {
            let npc = s.element_type().nodes_per_cell();
            s.connectivity()[self.idx * npc..(self.idx + 1) * npc].to_vec()
        })
    }

    /// Materialise the cell's nodes as a `Vec<Node>`. Each `Node`
    /// increments the node's refcount in the owning `Configuration`,
    /// matching the behaviour of `Configuration::add_node`.
    pub fn nodes(&self) -> Result<Vec<Node>> {
        let cfg = with(&self.sm, |s| s.configuration())?;
        let ids = self.node_ids()?;
        ids.into_iter()
            .map(|id| Node::acquire(cfg.clone(), id))
            .collect()
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cell").field("idx", &self.idx).finish()
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.element_type(), self.node_ids()) {
            (Ok(et), Ok(ids)) => {
                let raw: Vec<u32> = ids.into_iter().map(|n| n.0).collect();
                write!(f, "Cell<{}> #{}: {:?}", et, self.idx, raw)
            }
            _ => write!(f, "Cell #{}", self.idx),
        }
    }
}

/// Iterator over the cells of a single submesh.
#[derive(Clone)]
pub struct CellIter {
    sm: Handle<SubMesh>,
    next: usize,
    end: usize,
}

impl CellIter {
    pub(crate) fn new(sm: Handle<SubMesh>, end: usize) -> Self {
        Self { sm, next: 0, end }
    }
}

impl Iterator for CellIter {
    type Item = Cell;
    fn next(&mut self) -> Option<Cell> {
        if self.next < self.end {
            let c = Cell {
                sm: self.sm.clone(),
                idx: self.next,
            };
            self.next += 1;
            Some(c)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CellIter {}

// ─── Python binding ─────────────────────────────────────────────────────────

#[cfg(feature = "extension-module")]
mod python {
    use super::*;
    use crate::node::PyNode;
    use pyo3::exceptions::PyIndexError;
    use pyo3::prelude::*;

    /// Python wrapper for [`Cell`].
    #[pyclass(name = "Cell")]
    pub struct PyCell {
        pub(crate) inner: Cell,
    }

    impl PyCell {
        pub(crate) fn from_cell(c: Cell) -> Self {
            Self { inner: c }
        }
    }

    #[pymethods]
    impl PyCell {
        #[getter]
        fn index(&self) -> usize {
            self.inner.index()
        }

        #[getter]
        fn element_type(&self) -> PyResult<String> {
            Ok(self.inner.element_type()?.name().to_string())
        }

        /// Raw connectivity (list of node ids).
        #[getter]
        fn node_ids(&self) -> PyResult<Vec<u32>> {
            Ok(self
                .inner
                .node_ids()?
                .into_iter()
                .map(|n| n.0)
                .collect())
        }

        /// Materialised nodes (each one refcounted on the
        /// Configuration).
        fn nodes(&self) -> PyResult<Vec<PyNode>> {
            let nodes = self.inner.nodes()?;
            Ok(nodes.into_iter().map(PyNode::from_node).collect())
        }

        fn __len__(&self) -> PyResult<usize> {
            Ok(self.inner.nodes_per_cell()?)
        }

        /// `cell[j]` — j-th node of the cell. Supports negative indices
        /// and raises `IndexError` out of range so `for node in cell:` works.
        fn __getitem__(&self, idx: isize) -> PyResult<PyNode> {
            let npc = self.inner.nodes_per_cell()? as isize;
            let normalized = if idx < 0 { npc + idx } else { idx };
            if normalized < 0 || normalized >= npc {
                return Err(PyIndexError::new_err(format!(
                    "cell index {idx} out of range (len={npc})"
                )));
            }
            let ids = self.inner.node_ids()?;
            let id = ids[normalized as usize];
            let cfg = with(&self.inner.sm, |s| s.configuration())?;
            let node = Node::acquire(cfg, id)?;
            Ok(PyNode::from_node(node))
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(format!("{:?}", self.inner))
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(format!("{}", self.inner))
        }
    }
}

#[cfg(feature = "extension-module")]
pub use python::PyCell;

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Configuration;
    use crate::mesh::{Mesh, SubMesh};
    use crate::node::Node;
    use crate::store::insert;

    #[test]
    fn cell_exposes_ids_and_nodes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let h = insert(sm);

        let cell = Cell::new(h, 0).unwrap();
        assert_eq!(cell.element_type().unwrap(), ElementType::TRI3);
        assert_eq!(cell.nodes_per_cell().unwrap(), 3);
        assert_eq!(cell.node_ids().unwrap(), vec![a.id(), b.id(), c.id()]);
        let nodes = cell.nodes().unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].id(), a.id());
    }

    #[test]
    fn cell_new_rejects_out_of_range() {
        let cfg = insert(Configuration::new(2).unwrap());
        let sm = insert(SubMesh::new(cfg, ElementType::TRI3));
        assert!(Cell::new(sm, 0).is_err());
    }

    #[test]
    fn cells_iterator_yields_all_cells() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        let mut mesh = Mesh::with_element_type(cfg, ElementType::SEG2);
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        mesh.add_cell(&[b.id(), c.id()]).unwrap();

        let cells: Vec<_> = mesh.cells(0).unwrap().collect();
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].index(), 0);
        assert_eq!(cells[1].index(), 1);
        assert_eq!(cells[0].node_ids().unwrap(), vec![a.id(), b.id()]);
        assert_eq!(cells[1].node_ids().unwrap(), vec![b.id(), c.id()]);
    }
}
