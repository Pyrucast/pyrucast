//! Python wrappers for [`crate::containers::element_field::SubElementField`] and
//! [`crate::containers::element_field::ElementField`].

use crate::atoms::Band;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::handle::Handle;
use crate::py::finite_element_space::PyFiniteElementSpace;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

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
        Ok(self.handle.read().cell_count())
    }

    /// Number of Gauss points per cell.
    fn gauss_count(&self) -> PyResult<usize> {
        Ok(self.handle.read().gauss_count())
    }

    /// Number of components stored per point.
    fn component_count(&self) -> PyResult<usize> {
        Ok(self.handle.read().component_count())
    }

    /// Component names, in order.
    fn components(&self) -> PyResult<Vec<String>> {
        Ok(self.handle.read().components().to_vec())
    }

    /// Index of component `name`, or `None` if unknown.
    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(self.handle.read().component_index(name))
    }

    /// Value at `(cell, gauss)` for component index `comp`.
    fn get(&self, cell: usize, gauss: usize, comp: usize) -> PyResult<f64> {
        Ok(self.handle.read().get(cell, gauss, comp)?)
    }

    /// Set the value at `(cell, gauss)` for component index `comp`.
    fn set(&self, cell: usize, gauss: usize, comp: usize, value: f64) -> PyResult<()> {
        self.handle.write().set(cell, gauss, comp, value)?;
        Ok(())
    }

    /// Value at `(cell, gauss)` for the named `component`.
    fn value(&self, cell: usize, gauss: usize, component: &str) -> PyResult<f64> {
        Ok(self.handle.read().value(cell, gauss, component)?)
    }

    /// Set the value at `(cell, gauss)` for the named `component`.
    fn set_value(&self, cell: usize, gauss: usize, component: &str, value: f64) -> PyResult<()> {
        self.handle
            .write()
            .set_value(cell, gauss, component, value)?;
        Ok(())
    }

    /// All component values at `(cell, gauss)`, in component order.
    fn point_values(&self, cell: usize, gauss: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().point_values(cell, gauss)?.to_vec())
    }

    /// Set `component` to `value` at every point.
    fn set_uniform(&self, component: &str, value: f64) -> PyResult<()> {
        self.handle.write().set_uniform(component, value)?;
        Ok(())
    }

    /// Set `component` to `value` at every point of `cell`.
    fn set_cell_uniform(&self, cell: usize, component: &str, value: f64) -> PyResult<()> {
        self.handle
            .write()
            .set_cell_uniform(cell, component, value)?;
        Ok(())
    }

    /// Smallest value of the named `component` — or, called without one, the
    /// smallest value of the **whole** field, every component pooled. Pooling
    /// reads the field as the flat list of its values: on components carrying
    /// different units it answers "the smallest number in there", not a
    /// physical quantity.
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::min(&*self.handle.read(), component)?)
    }

    /// Largest value of the named `component` — or, called without one, the
    /// largest value of the **whole** field, every component pooled (see
    /// `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::max(&*self.handle.read(), component)?)
    }

    /// Sum of the named `component` over the support (Σ over the Gauss points).
    /// Empty sums to `0.0`.
    fn sum(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::sum(&*self.handle.read(), component)?)
    }

    /// Add `scalar` to every value of `component` (in place).
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from every value of `component` (in place).
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply every value of `component` by `scalar` (in place).
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide every value of `component` by `scalar` (in place).
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new sub-field) ───────────────────
    //
    // `rhs` may be a float (scalar broadcast over every point × component) or
    // another `SubElementField` (per-component union with passthrough on a
    // shared support). Division does not guard against zero (inf/nan).

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

    /// `field[cell, gauss, "name"] = value`.
    fn __setitem__(&self, key: (usize, usize, String), value: f64) -> PyResult<()> {
        let (cell, gauss, comp) = key;
        self.handle.write().set_value(cell, gauss, &comp, value)?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.handle.read()))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", *self.handle.read()))
    }
}

impl PySubElementField {
    /// Dispatch an arithmetic operator: float → scalar broadcast,
    /// `SubElementField` → `merge_components` (union of components, passthrough).
    fn scalar_or_combine(
        &self,
        rhs: &Bound<'_, PyAny>,
        op: fn(f64, f64) -> f64,
    ) -> PyResult<PySubElementField> {
        if let Ok(s) = rhs.extract::<f64>() {
            let out = self.handle.read().map_all(|v| op(v, s));
            Ok(PySubElementField {
                handle: Handle::new(out),
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PySubElementField>>() {
            let a = (*self.handle.read()).clone();
            let b = (*other.handle.read()).clone();
            Ok(PySubElementField {
                handle: Handle::new(a.merge_components(&b, op)?),
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
        Ok(Field::components(&self.inner))
    }

    /// Visualize this field on its own support: each zone knows its
    /// submesh through its FE subspace, so the mesh is reconstructed
    /// (shared, not copied) and coloured by `component` — per-element
    /// nodal fit of the Gauss values, the discontinuities between
    /// elements stay visible.
    ///
    /// Same `view` / `save` / `show_axes` / `component` / `vmin` /
    /// `vmax` / `cmap` / `smooth` / `title` semantics as `Mesh.plot`.
    /// `revolve` / `revolve_angle` sweep an axisymmetric plot into its body
    /// of revolution — see `SubMesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, component=None, vmin=None, vmax=None, cmap=None, smooth=4, revolve=false, revolve_angle=360.0, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<crate::viz::Colormap>,
        smooth: usize,
        revolve: bool,
        revolve_angle: f64,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: cmap.unwrap_or_default(),
            vmin,
            vmax,
        };
        self.inner.plot(
            view,
            save.as_deref(),
            component.as_deref(),
            scale,
            smooth,
            title.as_deref(),
        )?;
        Ok(())
    }

    /// Smallest value of `component` across the zones defining it — or, called
    /// without a component, the smallest value of the **whole** field, every
    /// component of every zone pooled (see `SubElementField.min`).
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::min(&self.inner, component)?)
    }

    /// Largest value of `component` across the zones defining it — or, called
    /// without a component, the largest value of the **whole** field (see
    /// `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::max(&self.inner, component)?)
    }

    /// Sum of `component` across the sub-fields defining it (Σ over the whole
    /// field). Errors if no zone defines it.
    fn sum(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::sum(&self.inner, component)?)
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
    /// `merge_field` (per `(support, component)`, union/passthrough),
    /// `SubElementField` → `merge_subfield` (targeted zone update).
    fn binary(&self, rhs: &Bound<'_, PyAny>, op: fn(f64, f64) -> f64) -> PyResult<PyElementField> {
        use crate::containers::field::Field;
        if let Ok(s) = rhs.extract::<f64>() {
            Ok(PyElementField {
                inner: self.inner.combine_scalar(op, s)?,
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PyElementField>>() {
            Ok(PyElementField {
                inner: self.inner.merge_field(&other.inner, op)?,
            })
        } else if let Ok(sub) = rhs.extract::<PyRef<PySubElementField>>() {
            let s = (*sub.handle.read()).clone();
            Ok(PyElementField {
                inner: self.inner.merge_subfield(&s, op)?,
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float, an ElementField, or a SubElementField",
            ))
        }
    }
}

// Polymorphic subscript — **closed block**, undecorated on purpose (see
// `impl_aggregate_pymethods!`): its `.pyi` entries are the hand-written
// overloads submitted just below.
#[pymethods]
impl PySubElementField {
    /// Indexing dispatches on the key:
    /// - `field[cell, gauss, "name"]` → the **value** (ValueError if unknown);
    /// - `field["name"]` or `field[["a", "b"]]` → a **new sub-field** with only
    ///   those components (`filter_components`), so `u1[u2.components()]`
    ///   reprojects `u1` onto `u2`'s component set.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // (cell, gauss, component) → scalar value.
        if let Ok((cell, gauss, comp)) = key.extract::<(usize, usize, String)>() {
            let v = self
                .handle
                .read()
                .value(cell, gauss, &comp)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(v.into_pyobject(py)?.into_any().unbind());
        }
        // "name" or ["a", …] → component selection (a new sub-field).
        let names = crate::py::ops::field::extract_names(key)?;
        let out =
            crate::containers::field::SubField::select_components(&*self.handle.read(), names)?;
        Ok(Py::new(
            py,
            PySubElementField {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    }

    /// Comparison sugar → a per-component 0/1 mask (see `mask`), one value per
    /// Gauss point. `subfield >= x` / `> x` / `<= x` / `< x` test every
    /// component against `x`; `==` / `!=` and non-scalar right-hands fall
    /// back to `NotImplemented`.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(x) = other.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        let band = match op {
            CompareOp::Ge => Band::new(Some(x), None, None, None),
            CompareOp::Gt => Band::new(None, Some(x), None, None),
            CompareOp::Le => Band::new(None, None, Some(x), None),
            CompareOp::Lt => Band::new(None, None, None, Some(x)),
            CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
        }?;
        let out = crate::ops::element_field::mask_sub(&self.handle.read(), &band, None);
        Ok(Py::new(
            py,
            PySubElementField {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    }
}

// `__richcmp__` is a pyo3-only spelling: CPython exposes the comparison slots
// as `__ge__`/`__gt__`/`__le__`/`__lt__`, which is what the stub must declare.
#[pymethods]
impl PyElementField {
    /// Comparison sugar → a per-component 0/1 mask (see `mask`), one value per
    /// Gauss point. `field >= x` / `> x` / `<= x` / `< x` test every component
    /// against `x`; `==` / `!=` and non-scalar right-hands fall back to
    /// `NotImplemented`.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(x) = other.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        let band = match op {
            CompareOp::Ge => Band::new(Some(x), None, None, None),
            CompareOp::Gt => Band::new(None, Some(x), None, None),
            CompareOp::Le => Band::new(None, None, Some(x), None),
            CompareOp::Lt => Band::new(None, None, None, Some(x)),
            CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
        }?;
        let out = crate::ops::element_field::mask(&self.inner, &band, None)?;
        Ok(Py::new(py, PyElementField { inner: out })?.into_any())
    }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PyElementField:
    def __ge__(self, other: float) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field >= x` → a fresh `ElementField` of per-component 0/1 flags, one
        value per Gauss point (see `mask`), not a boolean."""
    def __gt__(self, other: float) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field > x` → a 0/1 mask field (see `__ge__`)."""
    def __le__(self, other: float) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field <= x` → a 0/1 mask field (see `__ge__`)."""
    def __lt__(self, other: float) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field < x` → a 0/1 mask field (see `__ge__`)."""
    "# }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PySubElementField:
    @overload
    def __getitem__(self, key: tuple[int, int, str]) -> float:
        """`field[cell, gauss, "name"]` → the value at that Gauss point of that
        cell, for that component. `ValueError` if any of the three is unknown."""
    @overload
    def __getitem__(self, key: str) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`field["sig_xx"]` → a fresh `SubElementField` on the same support,
        keeping only that component."""
    @overload
    def __getitem__(self, key: list[str]) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`field[["sig_xx", "sig_yy"]]` → a fresh `SubElementField` keeping only
        those components, so `u1[u2.components()]` reprojects `u1` onto `u2`'s
        component set."""
    def __ge__(self, other: float) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`subfield >= x` → a fresh `SubElementField` of per-component 0/1
        flags, one value per Gauss point (see `mask`), not a boolean."""
    def __gt__(self, other: float) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`subfield > x` → a 0/1 mask field (see `__ge__`)."""
    def __le__(self, other: float) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`subfield <= x` → a 0/1 mask field (see `__ge__`)."""
    def __lt__(self, other: float) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`subfield < x` → a 0/1 mask field (see `__ge__`)."""
    "# }
}

crate::impl_aggregate_pymethods!(
    PyElementField,
    PySubElementField,
    "ElementField",
    subfield,
    ElementField,
    r#"
class PyElementField:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubElementField"]:
        """`field[i]` → the `SubElementField` of zone i: the values carried by
        the Gauss points of one support."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field[i:j:k]` → a fresh `ElementField` over the sliced zones, shared
        with this one (no deep copy)."""
    @overload
    def __getitem__(self, key: str) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field["sig_xx"]` → a fresh `ElementField` keeping only that
        component, on every zone that carries it."""
    @overload
    def __getitem__(self, key: list[str]) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field[["sig_xx", "sig_yy"]]` → a fresh `ElementField` keeping only
        those components, on every zone that carries them."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyElementField"] | pyo3_stub_gen.RustType["PySubElementField"]) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`field | other` → a fresh `ElementField` holding the zones of both.
        Component-disjoint zones on one support stay **side by side** (unlike
        `NodeField`, which fuses them); a component carried by two zones on the
        same support is rejected — fuse those explicitly with `pyrucast.element_field.consolidate`."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubElementField"]) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`subfield | field` — the mirror of `field | subfield`, differing only
        in that the lone zone comes first."""
    "#,
    r#"
class PySubElementField:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubElementField"]) -> pyo3_stub_gen.RustType["PyElementField"]:
        """`subfield | subfield` → a fresh `ElementField` holding both zones,
        left side by side (never fused)."""
    "#,
    field_components
);
crate::impl_dump_pymethod!(handle PySubElementField, handle);
