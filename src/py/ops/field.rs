//! Python wrappers for [`crate::ops::field`] — the operators polymorphic
//! over the field flavour, which give back a field of the caller's own kind.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node_field::{PyNodeField, PySubNodeField};
use crate::store::{insert, read};
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

/// Per-component 0/1 **mask** of a field against a value band — same flavour
/// and same structure as the input (Cast3M's `MASQUE`).
///
/// Unlike `pyrucast.mesh.select`, which extracts the passing support into a
/// `Mesh`, `mask` keeps the field's exact shape (zones, support, components)
/// and only rewrites the values: `1.0` where the band holds, `0.0` where it
/// does not — so the result is multipliable term by term with the input
/// (`field * mask(field, ge=0)` zeroes the negatives, component by component).
/// A `NodeField` masks per node, an `ElementField` per Gauss point.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`) — same rules as `pyrucast.mesh.select`. There is
/// **no** AND across components here: each value stands on its own.
///
/// `components=None` tests every component. A `components` list tests only
/// those; the others stay at `1.0` (identity for the product), and a zone
/// missing a listed component is left all-`1.0`. Errors if no bound is given,
/// or the lower one exceeds the upper.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
#[allow(clippy::too_many_arguments)]
pub fn mask(
    py: Python<'_>,
    field: &Bound<'_, PyAny>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    use crate::ops::field as ops;
    let band = ops::Band::new(ge, gt, le, lt)?;
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let out = ops::mask_nodes(&f.inner, &band, components)?;
        Ok(Py::new(py, PyNodeField { inner: out })?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        let out = ops::mask_cells(&f.inner, &band, components)?;
        Ok(Py::new(py, PyElementField { inner: out })?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        let out = ops::mask_sub_nodes(&*read(&f.handle)?, &band, components);
        Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        let out = ops::mask_sub_cells(&*read(&f.handle)?, &band, components);
        Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
            },
        )?
        .into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ))
    }
}

/// Keep only the named `components` of a field, zone by zone — Cast3M's `EXCO`
/// (component extraction).
///
/// `components` is a single name or a list of names; a list `model.primal_vars()`
/// is the intended input (strip a `solve` result of its dual/Lagrange unknowns).
/// In each zone the requested components are kept in the zone's **own** order;
/// a zone carrying **none** of them is dropped, and a zone carrying **only**
/// requested components has its sub-field **shared** (handle copied, not
/// duplicated). Names the field does not carry are ignored; errors if no zone
/// carries any of `components`.
///
/// Accepts a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`
/// and returns the same flavour.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn filter_components(
    py: Python<'_>,
    field: &Bound<'_, PyAny>,
    components: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use crate::containers::field::{Field, SubField};
    let wanted = extract_names(components)?;
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let inner = f.inner.filter_components(wanted.as_slice())?;
        return Ok(Py::new(py, PyNodeField { inner })?.into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        let inner = f.inner.filter_components(wanted.as_slice())?;
        return Ok(Py::new(py, PyElementField { inner })?.into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        let out = read(&f.handle)?.select_components(wanted.as_slice())?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        let out = read(&f.handle)?.select_components(wanted.as_slice())?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

/// Rename component `old` to `new` in a field, zone by zone — the renaming half
/// of Cast3M's `EXCO`. Values are preserved (metadata-only change). A zone
/// without `old` is carried unchanged; errors if no zone carries `old`, or if a
/// zone already has a component named `new`.
///
/// Accepts a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`
/// and returns the same flavour.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn rename_component(
    py: Python<'_>,
    field: &Bound<'_, PyAny>,
    old: &str,
    new: &str,
) -> PyResult<Py<PyAny>> {
    use crate::containers::field::{Field, SubField};
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let inner = f.inner.rename_component(old, new)?;
        return Ok(Py::new(py, PyNodeField { inner })?.into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        let inner = f.inner.rename_component(old, new)?;
        return Ok(Py::new(py, PyElementField { inner })?.into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        let out = read(&f.handle)?.rename_component(old, new)?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        let out = read(&f.handle)?.rename_component(old, new)?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
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
        let out = read(&a.handle)?.pscal(&*read(&b.handle)?)?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubElementFields"))?;
        let out = read(&a.handle)?.pscal(&*read(&b.handle)?)?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
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
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(
                    py,
                    PySubNodeField {
                        handle: insert(out),
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(
                    py,
                    PySubElementField {
                        handle: insert(out),
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
