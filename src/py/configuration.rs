//! Python wrapper for [`crate::containers::mesh::Configuration`].

use crate::containers::mesh::{Configuration, NodeId};
use crate::py::node::PyNode;
use crate::store::{insert, with, with_mut, Handle};
use pyo3::prelude::*;

/// The registry of live nodes and their coordinates, in a fixed spatial
/// dimension.
///
/// Create one with `Configuration(dim)`, then add nodes with
/// `add_node([x, y, ...])`. Every mesh, node and field is attached to a
/// `Configuration`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Configuration")]
pub struct PyConfiguration {
    pub(crate) handle: Handle<Configuration>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyConfiguration {
    #[new]
    fn py_new(dim: u8) -> PyResult<Self> {
        let cfg = Configuration::new(dim)?;
        Ok(Self {
            handle: insert(cfg),
        })
    }

    #[getter]
    fn dim(&self) -> PyResult<u8> {
        Ok(with(&self.handle, |c| c.dim())?)
    }

    fn node_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |c| c.node_count())?)
    }

    fn capacity(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |c| c.capacity())?)
    }

    fn is_alive(&self, id: u32) -> PyResult<bool> {
        Ok(with(&self.handle, |c| c.is_alive(NodeId(id)))?)
    }

    /// Add a node at `coords` and return it as a `Node` (refcount = 1).
    fn add_node(&self, coords: Vec<f64>) -> PyResult<PyNode> {
        let id = with_mut(&self.handle, |c| c.add_node(&coords))??;
        Ok(PyNode::from_raw(self.handle.clone(), id))
    }

    /// Return an additional `Node` for an existing id (refcount += 1).
    fn acquire(&self, id: u32) -> PyResult<PyNode> {
        with_mut(&self.handle, |c| c.incref(NodeId(id)))??;
        Ok(PyNode::from_raw(self.handle.clone(), NodeId(id)))
    }

    fn refcount(&self, id: u32) -> PyResult<u32> {
        Ok(with(&self.handle, |c| c.refcount(NodeId(id)))?)
    }

    /// Run the garbage collector; return the number of collected nodes.
    fn gc(&self) -> PyResult<usize> {
        Ok(with_mut(&self.handle, |c| c.gc())?)
    }

    fn coord(&self, id: u32) -> PyResult<Vec<f64>> {
        let v = with(&self.handle, |c| c.coord(NodeId(id)).map(|s| s.to_vec()))??;
        Ok(v)
    }

    fn set_coord(&self, id: u32, coords: Vec<f64>) -> PyResult<()> {
        with_mut(&self.handle, |c| c.set_coord(NodeId(id), &coords))??;
        Ok(())
    }

    fn add_coord_set(&self, name: String) -> PyResult<usize> {
        Ok(with_mut(&self.handle, |c| c.add_coord_set(name))?)
    }

    fn switch_to(&self, set: usize) -> PyResult<()> {
        with_mut(&self.handle, |c| c.switch_to(set))??;
        Ok(())
    }

    #[getter]
    fn active_set(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |c| c.active_set())?)
    }

    fn set_names(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |c| c.set_names().to_vec())?)
    }

    fn permutation(&self) -> PyResult<Option<Vec<u32>>> {
        Ok(with(&self.handle, |c| c.permutation().map(|s| s.to_vec()))?)
    }

    fn set_permutation(&self, perm: Vec<u32>) -> PyResult<()> {
        with_mut(&self.handle, |c| c.set_permutation(perm))??;
        Ok(())
    }

    fn clear_permutation(&self) -> PyResult<()> {
        with_mut(&self.handle, |c| c.clear_permutation())?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |c| format!("{:?}", c))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |c| format!("{}", c))?)
    }
}
