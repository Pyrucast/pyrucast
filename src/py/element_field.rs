//! Python wrappers for [`crate::containers::element_field::SubElementField`] and
//! [`crate::containers::element_field::ElementField`].

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::handle::Handle;
use crate::py::finite_element_space::PyFiniteElementSpace;
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
    fn cell_count(&self) -> usize {
        self.handle.read().cell_count()
    }

    /// Number of Gauss points per cell.
    fn gauss_count(&self) -> usize {
        self.handle.read().gauss_count()
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

    // ── Arithmetic operators (return a new sub-field) ───────────────────
    //
    // `rhs` may be a float (scalar broadcast over every point × component) or
    // another `SubElementField` (per-component union with passthrough on a
    // shared support). Division does not guard against zero (inf/nan).

    /// `field[cell, gauss, "name"] = value`.
    fn __setitem__(&self, key: (usize, usize, String), value: f64) -> PyResult<()> {
        let (cell, gauss, comp) = key;
        self.handle.write().set_value(cell, gauss, &comp, value)?;
        Ok(())
    }

    /// Smallest value of the named `component` — or, called without one, the
    /// smallest value of the **whole** field, every component pooled. Pooling
    /// reads the field as the flat list of its values: on components carrying
    /// different units it answers "the smallest number in there", not a
    /// physical quantity.
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        Ok(SubField::min(&*self.handle.read(), component)?)
    }

    /// Largest value of the named `component` — or, called without one, the
    /// largest value of the **whole** field, every component pooled (see `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        Ok(SubField::max(&*self.handle.read(), component)?)
    }

    /// Sum of the named `component` over the support — Σ over the nodes, or over
    /// the Gauss points for a field by elements. The resultant of a nodal force
    /// field, one component at a time. An empty support sums to `0.0`.
    fn sum(&self, component: &str) -> PyResult<f64> {
        Ok(SubField::sum(&*self.handle.read(), component)?)
    }

    /// Component names, in order.
    fn components(&self) -> Vec<String> {
        self.handle.read().components().to_vec()
    }

    /// Number of components stored per node, or per Gauss point for a field by
    /// elements.
    fn component_count(&self) -> usize {
        self.handle.read().component_count()
    }

    /// Index of component `name`, or `None` if unknown — no default index would
    /// say "absent" without being mistaken for a real one.
    fn component_index(&self, name: &str) -> Option<usize> {
        self.handle.read().component_index(name)
    }

    /// Keep only the named components, in the order given.
    ///
    /// `components` is a single name or a list of names (e.g. the result of
    /// `model.primal_vars()`). Returns a **new** field of the caller's own kind,
    /// sharing its support; the original is untouched. Errors if a requested name
    /// is absent — filtering never invents a component.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<Self> {
        let wanted = crate::py::ops::field::extract_names(components)?;
        let out = self.handle.read().select_components(wanted.as_slice())?;
        Ok(Self {
            handle: Handle::new(out),
        })
    }

    /// Rename one component, `old` to `new`, leaving every other untouched.
    ///
    /// Returns a **new** field of the caller's own kind, on the same support. The
    /// component order is kept — renaming is not reordering. Errors if `old` is
    /// absent, or if `new` is already taken: a name is how a component is
    /// addressed, so two of them cannot share one.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<Self> {
        let out = self.handle.read().rename_component(old, new)?;
        Ok(Self {
            handle: Handle::new(out),
        })
    }
}

impl PySubElementField {
    /// Dispatch an arithmetic operator: float → scalar broadcast,
    /// `SubElementField` → `merge_components` (union of components, passthrough).
    pub(crate) fn scalar_or_combine(
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

    // ── Per-component scalar ops (in place, on every zone defining it) ──

    // ── Arithmetic operators (return a new field) ───────────────────────
    //
    // `rhs`: a float (scalar over every zone), an `ElementField` (same
    // decomposition, strict), or a `SubElementField` (targeted zone update).

    /// Smallest value of `component` across the zones defining it — or, called
    /// without a component, the smallest value of the **whole** field, every
    /// component of every zone pooled (see the sub-field's `min` for what
    /// pooling means).
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::min(&self.inner, component)?)
    }

    /// Largest value of `component` across the zones defining it — or, called
    /// without a component, the largest value of the **whole** field (see `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::max(&self.inner, component)?)
    }

    /// Sum of `component` across the zones defining it (Σ over the whole field)
    /// — the resultant of a nodal force field, one component at a time. A node
    /// carried by several zones counts once per zone that stores it. Errors if
    /// no zone defines the component.
    fn sum(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::sum(&self.inner, component)?)
    }

    /// Union of the zones' component names, first-seen order.
    fn components(&self) -> Vec<String> {
        use crate::containers::field::Field;
        Field::components(&self.inner)
    }

    /// Keep only the named components, in the order given.
    ///
    /// `components` is a single name or a list of names (e.g. the result of
    /// `model.primal_vars()`). Returns a **new** field of the caller's own kind,
    /// sharing its support; the original is untouched. Errors if a requested name
    /// is absent — filtering never invents a component.
    fn filter_components(&self, components: &Bound<'_, PyAny>) -> PyResult<Self> {
        use crate::containers::field::Field;
        let wanted = crate::py::ops::field::extract_names(components)?;
        Ok(Self {
            inner: self.inner.filter_components(wanted.as_slice())?,
        })
    }

    /// Rename one component, `old` to `new`, leaving every other untouched.
    ///
    /// Returns a **new** field of the caller's own kind, on the same support. The
    /// component order is kept — renaming is not reordering. Errors if `old` is
    /// absent, or if `new` is already taken: a name is how a component is
    /// addressed, so two of them cannot share one.
    fn rename_component(&self, old: &str, new: &str) -> PyResult<Self> {
        use crate::containers::field::Field;
        Ok(Self {
            inner: Field::rename_component(&self.inner, old, new)?,
        })
    }
}

impl PyElementField {
    /// Dispatch an arithmetic operator: float → scalar, `ElementField` →
    /// `merge_field` (per `(support, component)`, union/passthrough),
    /// `SubElementField` → `merge_subfield` (targeted zone update).
    pub(crate) fn binary(
        &self,
        rhs: &Bound<'_, PyAny>,
        op: fn(f64, f64) -> f64,
    ) -> PyResult<PyElementField> {
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

// ─── Opérateurs arithmétiques ───────────────────────────────────────────────
//
// Posés ici, et non dans `field_macros.rs` où vivent leurs macros : un slot
// doit être expansé dans le module qui déclare son `#[pyclass]`, sans quoi pyo3
// engendre un appel de trampoline `unsafe` que l'édition 2024 ne couvre plus.

crate::py_field_binary_op! {
    /// `field + other` — element-wise sum.
    [PyElementField], __add__, |a, b| a + b
}
crate::py_subfield_binary_op! {
    /// `field + other` — element-wise sum.
    [PySubElementField], __add__, |a, b| a + b
}
crate::py_field_binary_op! {
    /// `field - other` — element-wise difference.
    [PyElementField], __sub__, |a, b| a - b
}
crate::py_subfield_binary_op! {
    /// `field - other` — element-wise difference.
    [PySubElementField], __sub__, |a, b| a - b
}
crate::py_field_binary_op! {
    /// `field * other` — element-wise product.
    [PyElementField], __mul__, |a, b| a * b
}
crate::py_subfield_binary_op! {
    /// `field * other` — element-wise product.
    [PySubElementField], __mul__, |a, b| a * b
}
crate::py_field_binary_op! {
    /// `field / other` — element-wise quotient.
    [PyElementField], __truediv__, |a, b| a / b
}
crate::py_subfield_binary_op! {
    /// `field / other` — element-wise quotient.
    [PySubElementField], __truediv__, |a, b| a / b
}
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float → scalar, `ElementField` → strict same-decomposition,
    /// `SubElementField` → targeted zone). The ternary `pow(x, y, z)` modulo
    /// form is rejected.
    fn __pow__(&self, exponent: &Bound<'_, PyAny>, modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::py::field_slots::reject_modulo(modulo)?;
        self.binary(exponent, |a, b| a.powf(b))
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubElementField {
    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float exponent → broadcast; `SubElementField` → strict
    /// element-by-element). The ternary `pow(x, y, z)` modulo form is
    /// rejected (meaningless on floats).
    fn __pow__(&self, exponent: &Bound<'_, PyAny>, modulo: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::py::field_slots::reject_modulo(modulo)?;
        self.scalar_or_combine(exponent, |a, b| a.powf(b))
    }
}

// ─── Comparaisons ───────────────────────────────────────────────────────────
//
// Slot lui aussi, donc posé ici, et écrit à la main : il n'a qu'une forme, et
// ne peut pas en avoir d'autre. Ce qu'il partage avec ses trois jumeaux est sa
// sémantique, pas sa forme — `field_slots::band_of` dit quelle bande de valeurs
// porte une comparaison.
//
// Son bloc n'est **pas** décoré de `gen_stub_pymethods` : CPython expose ces
// comparaisons sous les noms `__ge__`/`__gt__`/`__le__`/`__lt__`, déjà déclarés
// à la main dans le stub — les faire engendrer ici les compterait deux fois.

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
        op: pyo3::pyclass::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Some(band) = crate::py::field_slots::band_of(op, other)? else {
            return Ok(py.NotImplemented());
        };
        let out = crate::ops::element_field::mask(&self.inner, &band, None)?;
        Ok(Py::new(py, Self { inner: out })?.into_any())
    }
}

#[pymethods]
impl PySubElementField {
    /// Comparison sugar → a per-component 0/1 mask (see `mask`), one value per
    /// Gauss point. `subfield >= x` / `> x` / `<= x` / `< x` test every
    /// component against `x`; `==` / `!=` and non-scalar right-hands fall back
    /// to `NotImplemented`.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: pyo3::pyclass::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Some(band) = crate::py::field_slots::band_of(op, other)? else {
            return Ok(py.NotImplemented());
        };
        // `mask_sub` ne rend pas de `Result` — la zone est seule, il n'y a pas
        // d'agrégat à reconstruire, donc rien à refuser.
        let out = crate::ops::element_field::mask_sub(&self.handle.read(), &band, None);
        Ok(Py::new(
            py,
            Self {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    }
}
