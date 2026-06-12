//! Python wrapper for [`crate::containers::mesh::Configuration`].

use crate::containers::mesh::{Configuration, NodeId};
use crate::py::node::PyNode;
use crate::store::{insert, read, write, Handle};
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
    /// `Configuration(dim)` — an empty configuration in `dim` dimensions.
    #[new]
    fn py_new(dim: u8) -> PyResult<Self> {
        let cfg = Configuration::new(dim)?;
        Ok(Self {
            handle: insert(cfg),
        })
    }

    /// Spatial dimension of the coordinates (1, 2 or 3).
    #[getter]
    fn dim(&self) -> PyResult<u8> {
        Ok(read(&self.handle)?.dim())
    }

    /// Number of live nodes.
    fn node_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.node_count())
    }

    /// Number of allocated node slots (live plus not-yet-collected).
    fn capacity(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.capacity())
    }

    /// Whether node `id` is still live (not garbage-collected).
    ///
    /// Takes a **raw id**, not a `Node`, on purpose: a `Node` holds a
    /// refcount, so it could never be observed dead. This is the API for
    /// inspecting nodes you may no longer hold (post-GC checks).
    fn is_alive(&self, id: u32) -> PyResult<bool> {
        Ok(read(&self.handle)?.is_alive(NodeId(id)))
    }

    /// Add a node at `coords` and return it as a `Node` (refcount = 1).
    fn add_node(&self, coords: Vec<f64>) -> PyResult<PyNode> {
        let id = write(&self.handle)?.add_node(&coords)?;
        Ok(PyNode::from_raw(self.handle.clone(), id))
    }

    /// Return an additional `Node` for an existing id (refcount += 1).
    fn acquire(&self, id: u32) -> PyResult<PyNode> {
        write(&self.handle)?.incref(NodeId(id))?;
        Ok(PyNode::from_raw(self.handle.clone(), NodeId(id)))
    }

    /// Reference count of node `id` (how many holders keep it alive).
    ///
    /// Takes a **raw id** (see [`Self::is_alive`]): observing a refcount of
    /// 0 is impossible while holding the `Node` that would carry it.
    fn refcount(&self, id: u32) -> PyResult<u32> {
        Ok(read(&self.handle)?.refcount(NodeId(id)))
    }

    /// Run the garbage collector; return the number of collected nodes.
    fn gc(&self) -> PyResult<usize> {
        Ok(write(&self.handle)?.gc())
    }

    // Per-node coordinate access lives on `Node` (`node.coord()` /
    // `node.set_coord(...)`): a Node carries its Configuration, so the
    // (config, id) pair is never needed here.

    /// Add a named alternative coordinate set (same nodes, new coordinates);
    /// returns its index.
    fn add_coord_set(&self, name: String) -> PyResult<usize> {
        Ok(write(&self.handle)?.add_coord_set(name))
    }

    /// Make coordinate set `set` the active one.
    fn switch_to(&self, set: usize) -> PyResult<()> {
        write(&self.handle)?.switch_to(set)?;
        Ok(())
    }

    /// Index of the active coordinate set.
    #[getter]
    fn active_set(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.active_set())
    }

    /// Names of the coordinate sets, by index.
    fn set_names(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.set_names().to_vec())
    }

    /// Current node permutation (a renumbering), or `None` if unset.
    fn permutation(&self) -> PyResult<Option<Vec<u32>>> {
        Ok(read(&self.handle)?.permutation().map(|s| s.to_vec()))
    }

    /// Set a node permutation (a renumbering of the nodes).
    fn set_permutation(&self, perm: Vec<u32>) -> PyResult<()> {
        write(&self.handle)?.set_permutation(perm)?;
        Ok(())
    }

    /// Drop any node permutation.
    fn clear_permutation(&self) -> PyResult<()> {
        write(&self.handle)?.clear_permutation();
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
    }
}

crate::impl_dump_pymethod!(handle PyConfiguration, handle);
