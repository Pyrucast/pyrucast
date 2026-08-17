//! Python wrapper for [`crate::atoms::Cell`].

use crate::atoms::Cell;
use crate::atoms::Node;
use crate::py::node::PyNode;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// One cell of a mesh — the ordered nodes of a single element.
///
/// A read-only view obtained by indexing a submesh (`submesh[i]`) or via
/// `mesh.cell(submesh, cell)`. `len(cell)` is its node count; iterating a
/// cell yields its `Node`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Cell")]
pub struct PyCell {
    pub(crate) inner: Cell,
}

impl PyCell {
    pub(crate) fn from_cell(c: Cell) -> Self {
        Self { inner: c }
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyCell {
    /// Index of this cell within its submesh.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    /// Element type name of this cell (e.g. `"TRI3"`).
    #[getter]
    fn element_type(&self) -> PyResult<String> {
        Ok(self.inner.element_type()?.name().to_string())
    }

    /// Materialised nodes (each one refcounted on the
    /// Coords).
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
        let coords = self.inner.sm.read().coords();
        let node = Node::acquire(coords, id)?;
        Ok(PyNode::from_node(node))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

crate::impl_dump_pymethod!(value PyCell, inner);
