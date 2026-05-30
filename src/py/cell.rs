//! Python wrapper for [`crate::containers::mesh::Cell`].

use crate::containers::mesh::Cell;
use crate::containers::mesh::Node;
use crate::py::node::PyNode;
use crate::store::with;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// Python wrapper for [`Cell`].
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
