//! Python wrappers for [`crate::ops::field`] — the operators polymorphic
//! over the field flavour, which give back a field of the caller's own kind.

use crate::handle::Handle;
use crate::py::element_field::{PyElementField, PySubElementField};
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
/// wrapper types and applies the matching `ops::field::$name`, **and** the four
/// methods that are its « sujet » face. The documentation is written once, at
/// the call site, and reaches the free function and the four methods alike.
macro_rules! py_field_unary {
    ($(#[doc = $doc:literal])* $name:ident) => {
        $(#[doc = $doc])*
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

        $crate::py_field_transform! {
            $(#[doc = $doc])*
            Field unary: [PyNodeField, PyElementField], $name
        }
        $crate::py_field_transform! {
            $(#[doc = $doc])*
            SubField unary: [PySubNodeField, PySubElementField], $name
        }
    };
}

py_field_unary! {
    /// Element-wise absolute value of a field.
    abs
}
py_field_unary! {
    /// Element-wise square root of a field (`nan` for negatives).
    sqrt
}
py_field_unary! {
    /// Element-wise exponential `eˣ` of a field.
    exp
}
py_field_unary! {
    /// Element-wise natural logarithm of a field (`-inf`/`nan` for ≤ 0).
    log
}
py_field_unary! {
    /// Element-wise base-10 logarithm of a field.
    log10
}
py_field_unary! {
    /// Element-wise cosine of a field (radians).
    cos
}
py_field_unary! {
    /// Element-wise sine of a field (radians).
    sin
}
py_field_unary! {
    /// Element-wise tangent of a field (radians).
    tan
}
py_field_unary! {
    /// Element-wise hyperbolic sine of a field.
    sinh
}
py_field_unary! {
    /// Element-wise hyperbolic cosine of a field.
    cosh
}
py_field_unary! {
    /// Element-wise hyperbolic tangent of a field.
    tanh
}

// ── Composantes : filtrer, renommer ─────────────────────────────────────────
//
// Ces deux verbes n'ont **pas** de fonction libre : ils sont des méthodes et
// rien d'autre, la forme canonique elle-même (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Leur documentation n'a donc personne à qui
// renvoyer — elle est écrite ici, une fois par verbe, et la macro la pose sur
// les quatre saveurs. Le littéral traverse pyo3 et pyo3-stub-gen : l'aide
// complète part dans `__doc__` comme dans le stub.

/// Distribue les deux textes aux quatre saveurs. Le chapeau existe pour qu'ils
/// ne soient écrits **qu'une fois**, chacun devant le nom de la méthode qu'il
/// décrit : capturés ligne à ligne en littéraux, ils sont substitués avant que
/// pyo3 et pyo3-stub-gen ne lisent l'item, donc les deux y voient un vrai
/// texte.
macro_rules! py_field_components {
    (
        $(#[doc = $doc_filter:literal])* filter_components
        $(#[doc = $doc_rename:literal])* rename_component
    ) => {
        $crate::py_field_transform! {
            Field components: [PyNodeField, PyElementField]
            $(#[doc = $doc_filter])* filter_components
            $(#[doc = $doc_rename])* rename_component
        }
        $crate::py_field_transform! {
            SubField components: [PySubNodeField, PySubElementField]
            $(#[doc = $doc_filter])* filter_components
            $(#[doc = $doc_rename])* rename_component
        }
    };
}

py_field_components! {
    /// Keep only the named components, in the order given.
    ///
    /// `components` is a single name or a list of names (e.g. the result of
    /// `model.primal_vars()`). Returns a **new** field of the caller's own kind,
    /// sharing its support; the original is untouched. Errors if a requested name
    /// is absent — filtering never invents a component.
    filter_components
    /// Rename one component, `old` to `new`, leaving every other untouched.
    ///
    /// Returns a **new** field of the caller's own kind, on the same support. The
    /// component order is kept — renaming is not reordering. Errors if `old` is
    /// absent, or if `new` is already taken: a name is how a component is
    /// addressed, so two of them cannot share one.
    rename_component
}
