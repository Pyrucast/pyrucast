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

/// One entry of a field constructor's `components`: a `(name, value)` pair,
/// the component holding `value` everywhere, or a bare name, holding zero.
#[derive(FromPyObject)]
pub(crate) enum InitialComponent {
    Valued(String, f64),
    Named(String),
}

#[cfg(feature = "stub-gen")]
impl pyo3_stub_gen::PyStubType for InitialComponent {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        String::type_output() | <(String, f64)>::type_output()
    }
}

impl InitialComponent {
    /// The names of `components`, in order.
    pub(crate) fn names(components: &[Self]) -> Vec<String> {
        components
            .iter()
            .map(|c| match c {
                Self::Valued(name, _) | Self::Named(name) => name.clone(),
            })
            .collect()
    }

    /// The names of `components` when they name the columns of a `values`
    /// array: bare names only. A `(name, value)` pair is refused, since the
    /// array gives the values and the pair would leave its column unknown.
    pub(crate) fn column_names(components: &[Self]) -> PyResult<Vec<String>> {
        components
            .iter()
            .map(|c| match c {
                Self::Named(name) => Ok(name.clone()),
                Self::Valued(name, _) => Err(PyTypeError::new_err(format!(
                    "components: ('{name}', value) given with values=; the array gives the \
                     values, and components only names its columns"
                ))),
            })
            .collect()
    }

    /// Give each `(name, value)` entry of `components` its value on every zone
    /// of `field`, just built on them — a bare name keeps the zero it was
    /// built with.
    pub(crate) fn fill<F: crate::containers::field::Field>(field: &F, components: &[Self])
    where
        F::Sub: crate::containers::field::SubField,
    {
        use crate::containers::field::SubField;
        for c in components {
            if let Self::Valued(name, value) = c {
                for zone in field.iter() {
                    zone.write()
                        .set_uniform(name, *value)
                        .expect("every zone was built with every name");
                }
            }
        }
    }
}

/// Scalar product of two fields node by node, `pᵢ = Σ_c xᵢ,c·yᵢ,c`.
///
/// The reduction runs over the components alone: the support is kept, one
/// value per node, or per Gauss point for a field at the Gauss points. The
/// components are aligned by name (their order may differ) and only those both
/// fields carry count; a support only one field carries has no counterpart and
/// gives no zone. Each value is written once, so the result does not depend on
/// the number of threads.
///
/// # Parameters
///
/// - `x`: a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`.
/// - `y`: a field of the same type as `x`; two sub-fields must sit on the same
///   support.
///
/// # Returns
///
/// A new field of the type of `x`, one zone per support both fields carry,
/// with a single component `"psca"`.
///
/// # Raises
///
/// - `TypeError` if `y` is not of the type of `x`, or `x` none of the four.
/// - `RuntimeError` if two sub-fields sit on different supports.
///
/// # See also
///
/// `pyrucast.measure.xty`, the global scalar product: a single number over
/// the whole field.
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
macro_rules! define_polymorphic_pyfunction {
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

        $crate::impl_field_transform_pymethod! {
            $(#[doc = $doc])*
            [PyNodeField, PyElementField], $name
        }
        $crate::impl_subfield_transform_pymethod! {
            $(#[doc = $doc])*
            [PySubNodeField, PySubElementField], $name
        }
    };
}

define_polymorphic_pyfunction! {
    /// Absolute value of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    abs
}
define_polymorphic_pyfunction! {
    /// Square root of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded. A negative value gives `nan`.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    sqrt
}
define_polymorphic_pyfunction! {
    /// Exponential `eˣ` of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    exp
}
define_polymorphic_pyfunction! {
    /// Natural logarithm of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded. Zero gives `-inf`, a negative value `nan`.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    log
}
define_polymorphic_pyfunction! {
    /// Base-10 logarithm of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded. Zero gives `-inf`, a negative value `nan`.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    log10
}
define_polymorphic_pyfunction! {
    /// Cosine of `field`, value by value, in radians.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    cos
}
define_polymorphic_pyfunction! {
    /// Sine of `field`, value by value, in radians.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    sin
}
define_polymorphic_pyfunction! {
    /// Tangent of `field`, value by value, in radians.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    tan
}
define_polymorphic_pyfunction! {
    /// Hyperbolic sine of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    sinh
}
define_polymorphic_pyfunction! {
    /// Hyperbolic cosine of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    cosh
}
define_polymorphic_pyfunction! {
    /// Hyperbolic tangent of `field`, value by value.
    ///
    /// Every value of every component is mapped by the function, as numpy does: the
    /// result is not guarded.
    ///
    /// # Parameters
    ///
    /// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or
    ///   `SubElementField`.
    ///
    /// # Returns
    ///
    /// A new field of the type of `field`, with the same structure.
    ///
    /// # Raises
    ///
    /// `TypeError` if `field` is none of the four field types.
    tanh
}
