//! Python wrapper for [`crate::containers::mesh::Node`].

use crate::containers::mesh::{Configuration, NodeId};
use crate::containers::mesh::Node;
use crate::py::mesh::PyMesh;
use crate::store::Handle;
use pyo3::prelude::*;

/// A node of a `Configuration`: a stable identifier that carries the
/// configuration it belongs to (and therefore its coordinates).
///
/// Created via `Configuration.add_node([x, y, ...])`; passed wherever an
/// API needs a node.
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
    /// Stable integer id of this node within its `Configuration`.
    #[getter]
    fn id(&self) -> u32 {
        self.node.id().0
    }

    /// This node's coordinates in the active coordinate set.
    fn coord(&self) -> PyResult<Vec<f64>> {
        Ok(self.node.coord()?)
    }

    /// Overwrite this node's coordinates.
    fn set_coord(&self, coords: Vec<f64>) -> PyResult<()> {
        self.node.set_coord(&coords)?;
        Ok(())
    }

    /// `node + node` → a unitary POI1 `Mesh` over both nodes. Returns
    /// `NotImplemented` for any other right-hand type.
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.extract::<PyRef<'_, PyNode>>() {
            let mesh = (self.as_node() + o.as_node())?;
            return Ok(Py::new(py, PyMesh { inner: mesh })?.into_any());
        }
        Ok(py.NotImplemented())
    }

    /// Right-hand `mesh + node`: append this node to a **unitary POI1**
    /// `Mesh`, yielding a new one. Reached when the left operand's `__add__`
    /// returns `NotImplemented` (a `Mesh` doesn't know how to add a `Node`).
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(m) = other.extract::<PyRef<'_, PyMesh>>() {
            let mesh = (&m.inner + self.as_node())?;
            return Ok(Py::new(py, PyMesh { inner: mesh })?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.node))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.node))
    }
}

crate::impl_dump_pymethod!(value PyNode, node);
