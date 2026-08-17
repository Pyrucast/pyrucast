//! Python wrappers for [`crate::ops::field`] — the operators polymorphic
//! over the field flavour, which give back a field of the caller's own kind.

use crate::handle::Handle;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::mesh::PyMesh;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Extract a component-name argument that is either a single `str` or a list of
/// `str` (e.g. the result of `model.primal_vars()`) into a `Vec<String>`.
pub(crate) fn extract_names(arg: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(name) = arg.extract::<String>() {
        return Ok(vec![name]);
    }
    if let Ok(names) = arg.extract::<Vec<String>>() {
        return Ok(names);
    }
    Err(PyTypeError::new_err(
        "components: expected a str or a list of str",
    ))
}

/// Node-by-node (or point-by-point) scalar product of two fields — Cast3M's
/// `PSCA`. Returns a **new field** of the same flavour as the inputs, carrying
/// a single `"psca"` component whose value at each node/point is `∑_c xᵣ,c·yᵣ,c`
/// (reduction over components only, the support is kept).
///
/// `x` and `y` must be the same flavour (`NodeField` / `SubNodeField` /
/// `ElementField` / `SubElementField`), sit on the same support/decomposition,
/// and carry the same components (aligned by name).
///
/// For the **global** scalar product (a single float over the whole field),
/// see `pyrucast.measure.xty`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn psca(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be NodeFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyNodeField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be ElementFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyElementField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubNodeFields"))?;
        let out = a.handle.read().pscal(&*b.handle.read())?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(out),
            },
        )?
        .into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubElementFields"))?;
        let out = a.handle.read().pscal(&*b.handle.read())?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: Handle::new(out),
            },
        )?
        .into_any());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

// ── Element-wise unary maths (numpy-style) ──────────────────────────────────
//
// `pyrucast.cos(field)`, `pyrucast.exp(field)`, … apply a scalar function to
// every value of a field, returning a **new** field of the same type. They
// accept any of the four field flavours (`NodeField` / `SubNodeField` /
// `ElementField` / `SubElementField`) and mirror `crate::ops::field::*`.
// Results are unguarded (numpy-like): `log` of ≤ 0 → `-inf`/`nan`, etc.

/// Generate a `#[pyfunction] $name(field)` that dispatches over the four field
/// wrapper types and applies the matching `ops::field::$name`.
macro_rules! py_field_unary {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            use crate::ops::field::$name as op;
            if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
                return Ok(Py::new(
                    py,
                    PyNodeField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
                return Ok(Py::new(
                    py,
                    PyElementField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
                let out = op(&*f.handle.read())?;
                return Ok(Py::new(
                    py,
                    PySubNodeField {
                        handle: Handle::new(out),
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
                let out = op(&*f.handle.read())?;
                return Ok(Py::new(
                    py,
                    PySubElementField {
                        handle: Handle::new(out),
                    },
                )?
                .into_any());
            }
            Err(PyTypeError::new_err(
                "expected a NodeField, SubNodeField, ElementField or SubElementField",
            ))
        }
    };
}

py_field_unary!(abs, "Element-wise absolute value of a field.");
py_field_unary!(
    sqrt,
    "Element-wise square root of a field (`nan` for negatives)."
);
py_field_unary!(exp, "Element-wise exponential `eˣ` of a field.");
py_field_unary!(
    log,
    "Element-wise natural logarithm of a field (`-inf`/`nan` for ≤ 0)."
);
py_field_unary!(log10, "Element-wise base-10 logarithm of a field.");
py_field_unary!(cos, "Element-wise cosine of a field (radians).");
py_field_unary!(sin, "Element-wise sine of a field (radians).");
py_field_unary!(tan, "Element-wise tangent of a field (radians).");
py_field_unary!(sinh, "Element-wise hyperbolic sine of a field.");
py_field_unary!(cosh, "Element-wise hyperbolic cosine of a field.");
py_field_unary!(tanh, "Element-wise hyperbolic tangent of a field.");

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs polymorphes (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Le type du receveur étant connu, la méthode
// court-circuite le dispatch à quatre branches de la fonction libre et rend un
// type **précis** au lieu de `Any`. `psca` n'y figure pas : symétrique.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// Voir `pyrucast.field.abs`.
    fn abs(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::abs(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sqrt`.
    fn sqrt(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::sqrt(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.exp`.
    fn exp(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::exp(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.log`.
    fn log(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::log(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.log10`.
    fn log10(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::log10(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.cos`.
    fn cos(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::cos(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sin`.
    fn sin(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::sin(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.tan`.
    fn tan(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::tan(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sinh`.
    fn sinh(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::sinh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.cosh`.
    fn cosh(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::cosh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.tanh`.
    fn tanh(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: crate::ops::field::tanh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyNodeField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyNodeField {
            inner: crate::ops::node_field::mask(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_nodes(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.field.filter_components`.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        use crate::containers::field::Field;
        let wanted = extract_names(components)?;
        Ok(PyNodeField {
            inner: self.inner.filter_components(wanted.as_slice())?,
        })
    }

    /// Voir `pyrucast.field.rename_component`.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<PyNodeField> {
        use crate::containers::field::Field;
        Ok(PyNodeField {
            inner: self.inner.rename_component(old, new)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// Voir `pyrucast.field.abs`.
    fn abs(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::abs(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sqrt`.
    fn sqrt(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::sqrt(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.exp`.
    fn exp(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::exp(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.log`.
    fn log(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::log(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.log10`.
    fn log10(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::log10(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.cos`.
    fn cos(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::cos(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sin`.
    fn sin(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::sin(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.tan`.
    fn tan(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::tan(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.sinh`.
    fn sinh(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::sinh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.cosh`.
    fn cosh(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::cosh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.tanh`.
    fn tanh(&self) -> PyResult<PyElementField> {
        Ok(PyElementField {
            inner: crate::ops::field::tanh(&self.inner)?,
        })
    }

    /// Voir `pyrucast.field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyElementField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyElementField {
            inner: crate::ops::element_field::mask(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_cells(&self.inner, &band, components)?,
        })
    }

    /// Voir `pyrucast.field.filter_components`.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<PyElementField> {
        use crate::containers::field::Field;
        let wanted = extract_names(components)?;
        Ok(PyElementField {
            inner: self.inner.filter_components(wanted.as_slice())?,
        })
    }

    /// Voir `pyrucast.field.rename_component`.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<PyElementField> {
        use crate::containers::field::Field;
        Ok(PyElementField {
            inner: self.inner.rename_component(old, new)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubNodeField {
    /// Voir `pyrucast.field.abs`.
    fn abs(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::abs(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sqrt`.
    fn sqrt(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::sqrt(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.exp`.
    fn exp(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::exp(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.log`.
    fn log(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::log(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.log10`.
    fn log10(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::log10(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.cos`.
    fn cos(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::cos(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sin`.
    fn sin(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::sin(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.tan`.
    fn tan(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::tan(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sinh`.
    fn sinh(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::sinh(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.cosh`.
    fn cosh(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::cosh(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.tanh`.
    fn tanh(&self) -> PyResult<PySubNodeField> {
        let out = crate::ops::field::tanh(&*self.handle.read())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PySubNodeField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        let out = crate::ops::node_field::mask_sub(&*self.handle.read(), &band, components);
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_sub_nodes(&*self.handle.read(), &band, components)?,
        })
    }

    /// Voir `pyrucast.field.filter_components`.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<PySubNodeField> {
        use crate::containers::field::SubField;
        let wanted = extract_names(components)?;
        let out = self.handle.read().select_components(wanted.as_slice())?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.rename_component`.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<PySubNodeField> {
        use crate::containers::field::SubField;
        let out = self.handle.read().rename_component(old, new)?;
        Ok(PySubNodeField {
            handle: Handle::new(out),
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubElementField {
    /// Voir `pyrucast.field.abs`.
    fn abs(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::abs(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sqrt`.
    fn sqrt(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::sqrt(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.exp`.
    fn exp(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::exp(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.log`.
    fn log(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::log(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.log10`.
    fn log10(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::log10(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.cos`.
    fn cos(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::cos(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sin`.
    fn sin(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::sin(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.tan`.
    fn tan(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::tan(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.sinh`.
    fn sinh(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::sinh(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.cosh`.
    fn cosh(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::cosh(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.tanh`.
    fn tanh(&self) -> PyResult<PySubElementField> {
        let out = crate::ops::field::tanh(&*self.handle.read())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.mask`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn mask(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PySubElementField> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        let out = crate::ops::element_field::mask_sub(&*self.handle.read(), &band, components);
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.mesh.select`.
    #[pyo3(signature = (ge=None, gt=None, le=None, lt=None, components=None))]
    fn select(
        &self,
        ge: Option<f64>,
        gt: Option<f64>,
        le: Option<f64>,
        lt: Option<f64>,
        components: Option<Vec<String>>,
    ) -> PyResult<PyMesh> {
        let band = crate::atoms::Band::new(ge, gt, le, lt)?;
        Ok(PyMesh {
            inner: crate::ops::mesh::select_sub_cells(&*self.handle.read(), &band, components)?,
        })
    }

    /// Voir `pyrucast.field.filter_components`.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<PySubElementField> {
        use crate::containers::field::SubField;
        let wanted = extract_names(components)?;
        let out = self.handle.read().select_components(wanted.as_slice())?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }

    /// Voir `pyrucast.field.rename_component`.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<PySubElementField> {
        use crate::containers::field::SubField;
        let out = self.handle.read().rename_component(old, new)?;
        Ok(PySubElementField {
            handle: Handle::new(out),
        })
    }
}
