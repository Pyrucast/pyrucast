//! Python wrapper for [`crate::containers::node_field::SubNodeField`].

use crate::containers::node_field::SubNodeField;
use crate::py::mesh::{submesh_handle, PyMesh, PySubMesh};
use crate::py::node::PyNode;
use crate::store::{insert, with, with_mut, Handle};
use pyo3::prelude::*;

/// A field of values carried by mesh nodes — one scalar per
/// `(node, component)`.
///
/// Index it as `field[node, "X"]`. Build one with `coordinates(mesh)`, or
/// derive it with `restrict` / `merge`; add fields or scalars with `+`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "NodeField")]
pub struct PyNodeField {
    pub(crate) handle: Handle<SubNodeField>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// `SubNodeField(support, components)` — zero-initialized field over the
    /// POI1 nodes of `support`. `support` may be a `SubMesh` or a
    /// **unitary** `Mesh` (the parent→sub coercion: a one-submesh mesh is
    /// accepted directly, so callers rarely need `mesh[0]`).
    #[new]
    fn py_new(support: &Bound<'_, PyAny>, components: Vec<String>) -> PyResult<Self> {
        let sm_handle = submesh_handle(support)?;
        let nf = SubNodeField::from_poi1(&sm_handle, components)?;
        Ok(Self {
            handle: insert(nf),
        })
    }

    /// Number of nodes in the support.
    fn node_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.node_count())?)
    }

    /// Number of components stored per node.
    fn component_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.component_count())?)
    }

    /// Component names, in order.
    fn components(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |f| f.components().to_vec())?)
    }

    /// Value at node index `node_idx`, component index `comp_idx`.
    fn get(&self, node_idx: usize, comp_idx: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.get(node_idx, comp_idx))??)
    }

    /// Set the value at node index `node_idx`, component index `comp_idx`.
    fn set(&self, node_idx: usize, comp_idx: usize, value: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.set(node_idx, comp_idx, value))??;
        Ok(())
    }

    /// Value at `node`, component index `comp_idx`.
    fn get_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(with(&self.handle, |f| f.get_by_node(nid, comp_idx))??)
    }

    /// Set the value at `node`, component index `comp_idx`.
    fn set_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        with_mut(&self.handle, |f| {
            f.set_by_node(nid, comp_idx, value)
        })??;
        Ok(())
    }

    /// Index of component `name`, or `None` if unknown.
    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(with(&self.handle, |f| f.component_index(name))?)
    }

    /// All component values at node index `node_idx`, in order.
    fn node_values(&self, node_idx: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |f| f.node_values(node_idx).map(|s| s.to_vec()))??)
    }

    /// The POI1 `SubMesh` this field is supported on.
    fn support_submesh(&self) -> PyResult<PySubMesh> {
        let sm = with(&self.handle, |f| f.support_submesh())??;
        Ok(PySubMesh { handle: insert(sm) })
    }

    /// The POI1 `Mesh` this field is supported on.
    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mesh = with(&self.handle, |f| f.support_mesh())??;
        Ok(PyMesh { inner: mesh })
    }

    /// Value at `node` for the named `component`.
    fn value(&self, node: PyRef<'_, PyNode>, component: &str) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(with(&self.handle, |f| f.value(nid, component))??)
    }

    /// Set the value at `node` for the named `component`.
    fn set_value(&self, node: PyRef<'_, PyNode>, component: &str, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        with_mut(&self.handle, |f| f.set_value(nid, component, value))??;
        Ok(())
    }

    /// Smallest value of the named `component`.
    fn min(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(with(&self.handle, |f| SubField::min(f, component))??)
    }

    /// Largest value of the named `component`.
    fn max(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(with(&self.handle, |f| SubField::max(f, component))??)
    }

    /// Add `scalar` to every value of `component` (in place).
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.add_to_component(component, scalar))??;
        Ok(())
    }

    /// Subtract `scalar` from every value of `component` (in place).
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.sub_to_component(component, scalar))??;
        Ok(())
    }

    /// Multiply every value of `component` by `scalar` (in place).
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.mul_to_component(component, scalar))??;
        Ok(())
    }

    /// Divide every value of `component` by `scalar` (in place).
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.div_to_component(component, scalar))??;
        Ok(())
    }

    /// `field + x` — `x` may be a scalar (added to every value) or another
    /// `SubNodeField` (component-wise addition over the union of supports).
    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        let result = if let Ok(scalar) = rhs.extract::<f64>() {
            with(&self.handle, |f| f + scalar)?
        } else if let Ok(other) = rhs.extract::<PyRef<PyNodeField>>() {
            // The store mutex is per-type and non-reentrant, so we must not
            // nest two `with::<SubNodeField>` calls. Clone the rhs out first,
            // then operate while holding only the lhs lock.
            let fb = with(&other.handle, |f| f.clone())?;
            with(&self.handle, |a| a + &fb)??
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "SubNodeField + expects a float or a SubNodeField",
            ));
        };
        Ok(PyNodeField {
            handle: insert(result),
        })
    }

    fn __sub__(&self, rhs: f64) -> PyResult<PyNodeField> {
        let result = with(&self.handle, |f| f - rhs)?;
        Ok(PyNodeField {
            handle: insert(result),
        })
    }

    fn __mul__(&self, rhs: f64) -> PyResult<PyNodeField> {
        let result = with(&self.handle, |f| f * rhs)?;
        Ok(PyNodeField {
            handle: insert(result),
        })
    }

    fn __truediv__(&self, rhs: f64) -> PyResult<PyNodeField> {
        let result = with(&self.handle, |f| f / rhs)?;
        Ok(PyNodeField {
            handle: insert(result),
        })
    }

    /// `field[node, "UX"]` — raises IndexError if absent.
    fn __getitem__(&self, key: (PyRef<'_, PyNode>, String)) -> PyResult<f64> {
        let (node, comp) = key;
        let nid = node.as_node().id();
        Ok(with(&self.handle, |f| f.value(nid, &comp))??)
    }

    /// `field[node, "UX"] = v` — raises IndexError if absent.
    fn __setitem__(&self, key: (PyRef<'_, PyNode>, String), value: f64) -> PyResult<()> {
        let (node, comp) = key;
        let nid = node.as_node().id();
        with_mut(&self.handle, |f| f.set_value(nid, &comp, value))??;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{:?}", f))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{}", f))?)
    }
}

crate::impl_dump_pymethod!(handle PyNodeField, handle);
