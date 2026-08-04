//! Python wrapper for [`crate::atoms::Node`].

use crate::atoms::Node;
use crate::atoms::NodeId;
use crate::coords::Coords;
use crate::py::coords::PyCoords;
use crate::py::mesh::PyMesh;
use crate::store::Handle;
use pyo3::prelude::*;

/// A node of a `Coords`: a stable identifier that carries the
/// `Coords` it belongs to (and therefore its coordinates).
///
/// Created via `Coords.add_node([x, y, ...])`; passed wherever an
/// API needs a node.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Node")]
pub struct PyNode {
    node: Node,
}

impl PyNode {
    /// Build a `PyNode` from a handle and an id that have **already
    /// been incremented** on the Coords side. For internal use
    /// by `PyCoords::add_node` / `acquire`.
    pub(crate) fn from_raw(handle: Handle<Coords>, id: NodeId) -> Self {
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
    /// Stable integer id of this node within its `Coords`.
    #[getter]
    fn id(&self) -> u32 {
        self.node.id().0
    }

    /// This node's coordinates in the active coordinate set.
    fn position(&self) -> PyResult<Vec<f64>> {
        Ok(self.node.position()?)
    }

    /// Overwrite this node's coordinates.
    fn set_position(&self, coords: Vec<f64>) -> PyResult<()> {
        self.node.set_position(&coords)?;
        Ok(())
    }

    /// The `Coords` this node belongs to — the same safety net as
    /// `Mesh.coords()`, to get the handle back when it has been dropped
    /// on the Python side.
    fn coords(&self) -> PyCoords {
        PyCoords {
            handle: self.node.coords(),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.node))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.node))
    }
}

// Polymorphic union — **closed block**, undecorated on purpose (see
// `impl_aggregate_pymethods!`): its `.pyi` entries are the hand-written
// signatures submitted just below.
#[pymethods]
impl PyNode {
    /// `node | node` → a unitary POI1 `Mesh` over both nodes (the same
    /// union `|` as the aggregates). Returns `NotImplemented` for any other
    /// right-hand type.
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.extract::<PyRef<'_, PyNode>>() {
            let mesh = self.as_node().union(o.as_node())?;
            return Ok(Py::new(py, PyMesh { inner: mesh })?.into_any());
        }
        Ok(py.NotImplemented())
    }

    /// Right-hand `mesh | node`: append this node to a **unitary POI1**
    /// `Mesh`, yielding a new one. Reached when the left operand's `__or__`
    /// returns `NotImplemented` (a `Mesh` doesn't know how to union a `Node`).
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(m) = other.extract::<PyRef<'_, PyMesh>>() {
            let mesh = m.inner.union_node(self.as_node())?;
            return Ok(Py::new(py, PyMesh { inner: mesh })?.into_any());
        }
        Ok(py.NotImplemented())
    }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PyNode:
    def __or__(self, other: pyo3_stub_gen.RustType["PyNode"]) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`node | node` → a unitary POI1 `Mesh` over both nodes — the usual way
        to build the support of a point load or a boundary condition."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PyMesh"]) -> pyo3_stub_gen.RustType["PyMesh"]:
        """`mesh | node` → a fresh POI1 `Mesh` with this node appended. The
        left-hand `Mesh` must be unitary POI1."""
    "# }
}

crate::impl_dump_pymethod!(value PyNode, node);
