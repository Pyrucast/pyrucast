//! Python wrapper for [`crate::containers::node_field::NodeField`].

use crate::containers::node_field::NodeField;
use crate::py::mesh::{submesh_handle, PyMesh, PySubMesh};
use crate::py::node::PyNode;
use crate::store::{insert, with, with_mut, Handle};
use pyo3::prelude::*;

/// Python wrapper for [`NodeField`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "NodeField")]
pub struct PyNodeField {
    pub(crate) handle: Handle<NodeField>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// `NodeField(support, components)` — zero-initialized field over the
    /// POI1 nodes of `support`. `support` may be a `SubMesh` or a
    /// **unitary** `Mesh` (the parent→sub coercion: a one-submesh mesh is
    /// accepted directly, so callers rarely need `mesh[0]`).
    #[new]
    fn py_new(support: &Bound<'_, PyAny>, components: Vec<String>) -> PyResult<Self> {
        let sm_handle = submesh_handle(support)?;
        let nf = NodeField::from_poi1(&sm_handle, components)?;
        Ok(Self {
            handle: insert(nf),
        })
    }

    fn node_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.node_count())?)
    }

    fn component_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.component_count())?)
    }

    fn components(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |f| f.components().to_vec())?)
    }

    fn get(&self, node_idx: usize, comp_idx: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.get(node_idx, comp_idx))??)
    }

    fn set(&self, node_idx: usize, comp_idx: usize, value: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.set(node_idx, comp_idx, value))??;
        Ok(())
    }

    fn get_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(with(&self.handle, |f| f.get_by_node(nid, comp_idx))??)
    }

    fn set_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        with_mut(&self.handle, |f| {
            f.set_by_node(nid, comp_idx, value)
        })??;
        Ok(())
    }

    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(with(&self.handle, |f| f.component_index(name))?)
    }

    fn node_values(&self, node_idx: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |f| f.node_values(node_idx).map(|s| s.to_vec()))??)
    }

    fn support_submesh(&self) -> PyResult<PySubMesh> {
        let sm = with(&self.handle, |f| f.support_submesh())??;
        Ok(PySubMesh { handle: insert(sm) })
    }

    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mesh = with(&self.handle, |f| f.support_mesh())??;
        Ok(PyMesh { inner: mesh })
    }

    fn value(&self, node: PyRef<'_, PyNode>, component: &str) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(with(&self.handle, |f| f.value(nid, component))??)
    }

    fn set_value(&self, node: PyRef<'_, PyNode>, component: &str, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        with_mut(&self.handle, |f| f.set_value(nid, component, value))??;
        Ok(())
    }

    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.add_to_component(component, scalar))??;
        Ok(())
    }

    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.sub_to_component(component, scalar))??;
        Ok(())
    }

    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.mul_to_component(component, scalar))??;
        Ok(())
    }

    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.div_to_component(component, scalar))??;
        Ok(())
    }

    /// `field + x` — `x` may be a scalar (added to every value) or another
    /// `NodeField` (component-wise addition over the union of supports).
    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        let result = if let Ok(scalar) = rhs.extract::<f64>() {
            with(&self.handle, |f| f + scalar)?
        } else if let Ok(other) = rhs.extract::<PyRef<PyNodeField>>() {
            // The store mutex is per-type and non-reentrant, so we must not
            // nest two `with::<NodeField>` calls. Clone the rhs out first,
            // then operate while holding only the lhs lock.
            let fb = with(&other.handle, |f| f.clone())?;
            with(&self.handle, |a| a + &fb)??
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "NodeField + expects a float or a NodeField",
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
