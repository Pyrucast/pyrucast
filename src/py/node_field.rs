//! Python wrapper for [`crate::containers::node_field::NodeField`].

use crate::containers::mesh::NodeId;
use crate::containers::node_field::NodeField;
use crate::py::mesh::{PyMesh, PySubMesh};
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
    #[new]
    fn py_new(submesh: PyRef<PySubMesh>, components: Vec<String>) -> PyResult<Self> {
        let sm_handle = submesh.handle.clone();
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

    fn get_by_node(&self, node_id: u32, comp_idx: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.get_by_node(NodeId(node_id), comp_idx))??)
    }

    fn set_by_node(&self, node_id: u32, comp_idx: usize, value: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| {
            f.set_by_node(NodeId(node_id), comp_idx, value)
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

    fn value(&self, node_id: u32, component: &str) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.value(NodeId(node_id), component))??)
    }

    fn set_value(&self, node_id: u32, component: &str, value: f64) -> PyResult<()> {
        with_mut(&self.handle, |f| f.set_value(NodeId(node_id), component, value))??;
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

    /// `field[node_id, "UX"]` — raises IndexError if absent.
    fn __getitem__(&self, key: (u32, String)) -> PyResult<f64> {
        let (node_id, comp) = key;
        Ok(with(&self.handle, |f| f.value(NodeId(node_id), &comp))??)
    }

    /// `field[node_id, "UX"] = v` — raises IndexError if absent.
    fn __setitem__(&self, key: (u32, String), value: f64) -> PyResult<()> {
        let (node_id, comp) = key;
        with_mut(&self.handle, |f| f.set_value(NodeId(node_id), &comp, value))??;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{:?}", f))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{}", f))?)
    }
}

/// Build a `NodeField` carrying the coordinates of every node of `mesh`.
///
/// One component per requested axis (`"X"`, `"Y"`, `"Z"`). `components=None`
/// requests all the axes the mesh's `Configuration` has (`["X"]` in 1-D,
/// `["X", "Y"]` in 2-D, `["X", "Y", "Z"]` in 3-D). A non-POI1 mesh is
/// converted to POI1 internally (see `to_poi1`); the support is the unique
/// nodes of the mesh, in order of first appearance.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn coordinates(
    mesh: PyRef<PyMesh>,
    components: Option<Vec<String>>,
) -> PyResult<PyNodeField> {
    let field = crate::ops::field::coordinates(&mesh.inner, components)?;
    Ok(PyNodeField {
        handle: insert(field),
    })
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `NodeField` with the same components, supported on the
/// unique nodes of `mesh` (order of first appearance). Nodes of `mesh`
/// absent from `field` are assigned `0.0`. Errors if `mesh` and `field`
/// are attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    let result = with(&field.handle, |nf| crate::ops::field::restrict(nf, &mesh.inner))??;
    Ok(PyNodeField {
        handle: insert(result),
    })
}

/// Merge two node fields over the union of their supports.
///
/// Keeps each field's value where only one is defined, `0.0` where
/// neither is. Errors if the two fields hold different values at the same
/// `(node, component)` pair, or are attached to different `Configuration`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge(a: PyRef<PyNodeField>, b: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    // The store mutex is per-type and non-reentrant: clone `b` out before
    // locking `a` rather than nesting two `with::<NodeField>` calls.
    let fb = with(&b.handle, |f| f.clone())?;
    let result = with(&a.handle, |fa| crate::ops::field::merge(fa, &fb))??;
    Ok(PyNodeField {
        handle: insert(result),
    })
}
