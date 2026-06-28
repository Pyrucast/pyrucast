//! Python wrappers for [`crate::containers::element_field::SubElementField`] and
//! [`crate::containers::element_field::ElementField`].

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::store::{insert, read, write, Handle};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A **view** into one zone of an `ElementField`, obtained by indexing
/// (`element_field[i]`) — never constructed directly. Build at the parent
/// level instead: `ElementField(fes, components)` or
/// `material_field(model, ...)`, composed with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubElementField")]
pub struct PySubElementField {
    pub(crate) handle: Handle<SubElementField>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubElementField {
    /// Number of cells (elements) this field covers.
    fn cell_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.cell_count())
    }

    /// Number of Gauss points per cell.
    fn gauss_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.gauss_count())
    }

    /// Number of components stored per point.
    fn component_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.component_count())
    }

    /// Component names, in order.
    fn components(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.components().to_vec())
    }

    /// Index of component `name`, or `None` if unknown.
    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(read(&self.handle)?.component_index(name))
    }

    /// Value at `(cell, gauss)` for component index `comp`.
    fn get(&self, cell: usize, gauss: usize, comp: usize) -> PyResult<f64> {
        Ok(read(&self.handle)?.get(cell, gauss, comp)?)
    }

    /// Set the value at `(cell, gauss)` for component index `comp`.
    fn set(&self, cell: usize, gauss: usize, comp: usize, value: f64) -> PyResult<()> {
        write(&self.handle)?.set(cell, gauss, comp, value)?;
        Ok(())
    }

    /// Value at `(cell, gauss)` for the named `component`.
    fn value(&self, cell: usize, gauss: usize, component: &str) -> PyResult<f64> {
        Ok(read(&self.handle)?.value(cell, gauss, component)?)
    }

    /// Set the value at `(cell, gauss)` for the named `component`.
    fn set_value(&self, cell: usize, gauss: usize, component: &str, value: f64) -> PyResult<()> {
        write(&self.handle)?.set_value(cell, gauss, component, value)?;
        Ok(())
    }

    /// All component values at `(cell, gauss)`, in component order.
    fn point_values(&self, cell: usize, gauss: usize) -> PyResult<Vec<f64>> {
        Ok(read(&self.handle)?.point_values(cell, gauss)?.to_vec())
    }

    /// Set `component` to `value` at every point.
    fn set_uniform(&self, component: &str, value: f64) -> PyResult<()> {
        write(&self.handle)?.set_uniform(component, value)?;
        Ok(())
    }

    /// Set `component` to `value` at every point of `cell`.
    fn set_cell_uniform(&self, cell: usize, component: &str, value: f64) -> PyResult<()> {
        write(&self.handle)?.set_cell_uniform(cell, component, value)?;
        Ok(())
    }

    /// Smallest value of the named `component`.
    fn min(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::min(&*read(&self.handle)?, component)?)
    }

    /// Largest value of the named `component`.
    fn max(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::max(&*read(&self.handle)?, component)?)
    }

    /// Add `scalar` to every value of `component` (in place).
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from every value of `component` (in place).
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply every value of `component` by `scalar` (in place).
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide every value of `component` by `scalar` (in place).
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new sub-field) ───────────────────
    //
    // `rhs` may be a float (scalar broadcast over every point × component) or
    // another `SubElementField` (element-by-element, strict: same support and
    // same components). Division does not guard against zero (inf/nan).

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubElementField> {
        self.scalar_or_combine(rhs, |a, b| a + b)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubElementField> {
        self.scalar_or_combine(rhs, |a, b| a - b)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubElementField> {
        self.scalar_or_combine(rhs, |a, b| a * b)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubElementField> {
        self.scalar_or_combine(rhs, |a, b| a / b)
    }

    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float exponent → broadcast; `SubElementField` → strict
    /// element-by-element). The ternary `pow(x, y, z)` modulo form is
    /// rejected (meaningless on floats).
    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PySubElementField> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err(
                "field ** exponent does not support a modulo argument",
            ));
        }
        self.scalar_or_combine(exponent, |a, b| a.powf(b))
    }

    /// `field[cell, gauss, "name"]` — raises ValueError if the component
    /// is unknown.
    fn __getitem__(&self, key: (usize, usize, String)) -> PyResult<f64> {
        let (cell, gauss, comp) = key;
        read(&self.handle)?
            .value(cell, gauss, &comp)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `field[cell, gauss, "name"] = value`.
    fn __setitem__(&self, key: (usize, usize, String), value: f64) -> PyResult<()> {
        let (cell, gauss, comp) = key;
        write(&self.handle)?.set_value(cell, gauss, &comp, value)?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
    }
}

impl PySubElementField {
    /// Dispatch an arithmetic operator: float → scalar broadcast,
    /// `SubElementField` → strict element-by-element `combine`.
    fn scalar_or_combine(
        &self,
        rhs: &Bound<'_, PyAny>,
        op: fn(f64, f64) -> f64,
    ) -> PyResult<PySubElementField> {
        if let Ok(s) = rhs.extract::<f64>() {
            let out = read(&self.handle)?.map_all(|v| op(v, s));
            Ok(PySubElementField {
                handle: insert(out),
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PySubElementField>>() {
            let a = (*read(&self.handle)?).clone();
            let b = (*read(&other.handle)?).clone();
            Ok(PySubElementField {
                handle: insert(a.combine(&b, op)?),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float or a SubElementField",
            ))
        }
    }
}

/// A field of per-element values (e.g. material properties), one block per
/// finite-element zone.
///
/// Build with `ElementField(fes, components)` or `material_field(model, ...)`;
/// index it (`field[i]`) to reach a `SubElementField`, compose zones
/// with `|`.
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
    fn py_new(fespace: PyRef<PyFiniteElementSpace>, components: Vec<String>) -> PyResult<Self> {
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

    /// Union of the sub-fields' component names, first-seen order.
    fn components(&self) -> PyResult<Vec<String>> {
        use crate::containers::field::Field;
        Ok(Field::components(&self.inner)?)
    }

    /// Visualize this field on its own support: each zone knows its
    /// submesh through its FE subspace, so the mesh is reconstructed
    /// (shared, not copied) and coloured by `component` — per-element
    /// nodal fit of the Gauss values, the discontinuities between
    /// elements stay visible.
    ///
    /// Same `view` / `save` / `show_axes` / `component` / `vmin` /
    /// `vmax` / `cmap` / `smooth` semantics as `Mesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, component=None, vmin=None, vmax=None, cmap=None, smooth=4))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
        smooth: usize,
    ) -> PyResult<()> {
        let mut view = view
            .map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
                show_axes,
            })
            .unwrap_or_else(crate::viz::View::default);
        view.show_axes = show_axes;
        let scale = crate::viz::ColorScale {
            cmap: crate::py::mesh::parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        self.inner.plot(
            Some(view),
            save.as_deref(),
            component.as_deref(),
            scale,
            smooth,
        )?;
        Ok(())
    }

    /// Smallest value of `component` across the sub-fields defining it.
    fn min(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::min(&self.inner, component)?)
    }

    /// Largest value of `component` across the sub-fields defining it.
    fn max(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::max(&self.inner, component)?)
    }

    // ── Per-component scalar ops (in place, on every zone defining it) ──

    /// Add `scalar` to `component` on every zone that defines it.
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from `component` on every zone that defines it.
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply `component` by `scalar` on every zone that defines it.
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide `component` by `scalar` on every zone that defines it.
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new field) ───────────────────────
    //
    // `rhs`: a float (scalar over every zone), an `ElementField` (same
    // decomposition, strict), or a `SubElementField` (targeted zone update).

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyElementField> {
        self.binary(rhs, |a, b| a + b)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyElementField> {
        self.binary(rhs, |a, b| a - b)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyElementField> {
        self.binary(rhs, |a, b| a * b)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyElementField> {
        self.binary(rhs, |a, b| a / b)
    }

    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float → scalar, `ElementField` → strict same-decomposition,
    /// `SubElementField` → targeted zone). The ternary `pow(x, y, z)` modulo
    /// form is rejected.
    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PyElementField> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err(
                "field ** exponent does not support a modulo argument",
            ));
        }
        self.binary(exponent, |a, b| a.powf(b))
    }
}

impl PyElementField {
    /// Dispatch an arithmetic operator: float → scalar, `ElementField` →
    /// `combine_field` (same decomposition), `SubElementField` →
    /// `combine_subfield` (targeted zone update).
    fn binary(&self, rhs: &Bound<'_, PyAny>, op: fn(f64, f64) -> f64) -> PyResult<PyElementField> {
        use crate::containers::field::Field;
        if let Ok(s) = rhs.extract::<f64>() {
            Ok(PyElementField {
                inner: self.inner.combine_scalar(op, s)?,
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PyElementField>>() {
            Ok(PyElementField {
                inner: self.inner.combine_field(&other.inner, op)?,
            })
        } else if let Ok(sub) = rhs.extract::<PyRef<PySubElementField>>() {
            let s = (*read(&sub.handle)?).clone();
            Ok(PyElementField {
                inner: self.inner.combine_subfield(&s, op)?,
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float, an ElementField, or a SubElementField",
            ))
        }
    }
}

crate::impl_aggregate_pymethods!(
    PyElementField,
    PySubElementField,
    "ElementField",
    subfield,
    ElementField
);
crate::impl_dump_pymethod!(handle PySubElementField, handle);
