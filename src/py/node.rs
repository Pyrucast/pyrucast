//! Python wrapper for [`crate::mesh::node::Node`].

use crate::mesh::configuration::{Configuration, NodeId};
use crate::mesh::node::Node;
use crate::store::Handle;
use pyo3::prelude::*;

/// Python wrapper for [`Node`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Node")]
pub struct PyNode {
    node: Node,
}

impl PyNode {
    /// Build a `PyNode` from a handle and an id that have **already
    /// been incremented** on the Configuration side. For internal use
    /// by `PyConfiguration::add_node` / `acquire`.
    pub(crate) fn from_raw(handle: Handle<Configuration>, id: NodeId) -> Self {
        Self {
            node: Node::from_parts(handle, id),
        }
    }

    pub(crate) fn from_node(node: Node) -> Self {
        Self { node }
    }

    pub(crate) fn as_node(&self) -> &Node {
        &self.node
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNode {
    #[getter]
    fn id(&self) -> u32 {
        self.node.id().0
    }

    fn coord(&self) -> PyResult<Vec<f64>> {
        Ok(self.node.coord()?)
    }

    fn set_coord(&self, coords: Vec<f64>) -> PyResult<()> {
        self.node.set_coord(&coords)?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.node))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.node))
    }
}
