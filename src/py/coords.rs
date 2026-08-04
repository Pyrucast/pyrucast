//! Python wrapper for [`crate::coords::Coords`].

use crate::atoms::NodeId;
use crate::coords::Coords;
use crate::py::node::PyNode;
use crate::store::{insert, read, write, Handle};
use pyo3::prelude::*;

/// The registry of live nodes and their coordinates, in a fixed spatial
/// dimension.
///
/// Create one with `Coords(dim)`, then add nodes with
/// `add_node([x, y, ...])`. Every mesh, node and field is attached to a
/// `Coords`.
///
/// `Coords.axisymmetric()` builds the 2-D meridian plane of a body of
/// revolution instead (`x = r ≥ 0`, `y = z`): every integral over it then runs
/// over the full ring, `dΩ = 2πr |J| dξ`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Coords")]
pub struct PyCoords {
    pub(crate) handle: Handle<Coords>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyCoords {
    /// `Coords(dim)` — an empty `Coords` in `dim` dimensions.
    #[new]
    fn py_new(dim: u8) -> PyResult<Self> {
        let coords = Coords::new(dim)?;
        Ok(Self {
            handle: insert(coords),
        })
    }

    /// `Coords.axisymmetric()` — the 2-D meridian plane of a body of
    /// revolution: `x = r` (radius, `≥ 0`) and `y = z` (axis). The dimension is
    /// necessarily 2, so it takes no argument.
    ///
    /// Every FE space built on it integrates over the full ring; mechanics adds
    /// the hoop strain through `Model.elasticity(fes, "axisymmetric")`.
    #[classmethod]
    fn axisymmetric(_cls: &pyo3::Bound<'_, pyo3::types::PyType>) -> PyResult<Self> {
        Ok(Self {
            handle: insert(Coords::axisymmetric()?),
        })
    }

    /// Spatial dimension of the coordinates (1, 2 or 3).
    #[getter]
    fn dim(&self) -> PyResult<u8> {
        Ok(read(&self.handle)?.dim())
    }

    /// Whether these coordinates describe a body of revolution.
    #[getter]
    fn is_axisymmetric(&self) -> PyResult<bool> {
        Ok(read(&self.handle)?.is_axisymmetric())
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

    // Per-node coordinate access lives on `Node` (`node.position()` /
    // `node.set_position(...)`): a Node carries its Coords, so the
    // (config, id) pair is never needed here.

    /// Add a named alternative configuration (same nodes, new coordinates);
    /// returns its index.
    fn add_config(&self, name: String) -> PyResult<usize> {
        Ok(write(&self.handle)?.add_config(name))
    }

    /// Make configuration `config` the active one.
    fn select(&self, config: usize) -> PyResult<()> {
        write(&self.handle)?.select(config)?;
        Ok(())
    }

    /// Index of the active configuration.
    #[getter]
    fn active(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.active())
    }

    /// Names of the configurations, by index.
    fn names(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.names().to_vec())
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

crate::impl_dump_pymethod!(handle PyCoords, handle);
