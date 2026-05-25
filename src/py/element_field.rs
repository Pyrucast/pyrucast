//! Python wrappers for [`crate::containers::element_field::SubElementField`] and
//! [`crate::containers::element_field::ElementField`].

use crate::containers::element_field::{ElementField, SubElementField};
use crate::py::finite_element_space::{PyFiniteElementSpace, PySubFiniteElementSpace};
use crate::store::{insert, with, Handle};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for [`SubElementField`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubElementField")]
pub struct PySubElementField {
    pub(crate) handle: Handle<SubElementField>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubElementField {
    /// `SubElementField(subfespace, components)` — zero-initialized
    /// sub-field on a single [`SubFiniteElementSpace`].
    #[new]
    fn py_new(fespace: PyRef<PySubFiniteElementSpace>, components: Vec<String>) -> PyResult<Self> {
        let field = SubElementField::new(fespace.handle.clone(), components)?;
        Ok(Self {
            handle: insert(field),
        })
    }

    /// Alternate constructor: uniform value per component.
    #[classmethod]
    fn from_uniform_per_component(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PySubFiniteElementSpace>,
        components: Vec<String>,
        values_per_component: Vec<f64>,
    ) -> PyResult<Self> {
        let field = SubElementField::from_uniform_per_component(
            fespace.handle.clone(),
            components,
            &values_per_component,
        )?;
        Ok(Self {
            handle: insert(field),
        })
    }

    fn cell_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.cell_count())?)
    }

    fn gauss_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.gauss_count())?)
    }

    fn component_count(&self) -> PyResult<usize> {
        Ok(with(&self.handle, |f| f.component_count())?)
    }

    fn components(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |f| f.components().to_vec())?)
    }

    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(with(&self.handle, |f| f.component_index(name))?)
    }

    fn get(&self, cell: usize, gauss: usize, comp: usize) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.get(cell, gauss, comp))??)
    }

    fn set(&self, cell: usize, gauss: usize, comp: usize, value: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.set(cell, gauss, comp, value))??;
        Ok(())
    }

    fn value(&self, cell: usize, gauss: usize, component: &str) -> PyResult<f64> {
        Ok(with(&self.handle, |f| f.value(cell, gauss, component))??)
    }

    fn set_value(
        &self,
        cell: usize,
        gauss: usize,
        component: &str,
        value: f64,
    ) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| {
            f.set_value(cell, gauss, component, value)
        })??;
        Ok(())
    }

    fn point_values(&self, cell: usize, gauss: usize) -> PyResult<Vec<f64>> {
        Ok(with(&self.handle, |f| {
            f.point_values(cell, gauss).map(|s| s.to_vec())
        })??)
    }

    fn set_uniform(&self, component: &str, value: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.set_uniform(component, value))??;
        Ok(())
    }

    fn set_cell_uniform(&self, cell: usize, component: &str, value: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| {
            f.set_cell_uniform(cell, component, value)
        })??;
        Ok(())
    }

    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.add_to_component(component, scalar))??;
        Ok(())
    }

    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.sub_to_component(component, scalar))??;
        Ok(())
    }

    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.mul_to_component(component, scalar))??;
        Ok(())
    }

    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        crate::store::with_mut(&self.handle, |f| f.div_to_component(component, scalar))??;
        Ok(())
    }

    // ── Scalar operators (return a new sub-field) ───────────────────────

    fn __add__(&self, rhs: f64) -> PyResult<PySubElementField> {
        let res = with(&self.handle, |f| f + rhs)?;
        Ok(PySubElementField {
            handle: insert(res),
        })
    }

    fn __sub__(&self, rhs: f64) -> PyResult<PySubElementField> {
        let res = with(&self.handle, |f| f - rhs)?;
        Ok(PySubElementField {
            handle: insert(res),
        })
    }

    fn __mul__(&self, rhs: f64) -> PyResult<PySubElementField> {
        let res = with(&self.handle, |f| f * rhs)?;
        Ok(PySubElementField {
            handle: insert(res),
        })
    }

    fn __truediv__(&self, rhs: f64) -> PyResult<PySubElementField> {
        let res = with(&self.handle, |f| f / rhs)?;
        Ok(PySubElementField {
            handle: insert(res),
        })
    }

    /// `field[cell, gauss, "name"]` — raises ValueError if the component
    /// is unknown.
    fn __getitem__(&self, key: (usize, usize, String)) -> PyResult<f64> {
        let (cell, gauss, comp) = key;
        with(&self.handle, |f| f.value(cell, gauss, &comp))?
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `field[cell, gauss, "name"] = value`.
    fn __setitem__(&self, key: (usize, usize, String), value: f64) -> PyResult<()> {
        let (cell, gauss, comp) = key;
        crate::store::with_mut(&self.handle, |f| f.set_value(cell, gauss, &comp, value))??;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{:?}", f))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |f| format!("{}", f))?)
    }
}

/// Python wrapper for [`ElementField`].
///
/// Owns the `ElementField` struct directly — no longer stored in the
/// global store. Identity is the Python object identity.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "ElementField")]
pub struct PyElementField {
    pub(crate) inner: ElementField,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// `ElementField(fespace, components)` — one sub-field per subspace
    /// of `fespace`, sharing the same `components` list. Zero-initialized.
    #[new]
    fn py_new(
        fespace: PyRef<PyFiniteElementSpace>,
        components: Vec<String>,
    ) -> PyResult<Self> {
        let ef = ElementField::new(&fespace.inner, components)?;
        Ok(Self { inner: ef })
    }

    /// Explicit `components` list per subspace.
    #[classmethod]
    fn with_components_per_subspace(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        components_per_subspace: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        let ef = ElementField::with(&fespace.inner, &components_per_subspace)?;
        Ok(Self { inner: ef })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
}

crate::impl_aggregate_pymethods!(PyElementField, PySubElementField, "ElementField", subfield_count, subfield);
