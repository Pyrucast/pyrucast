//! Python wrappers for [`crate::ops::measure`] — the operators that reduce
//! containers to a number.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Integrate one component of a field over its support, `∫_Ω f dΩ`.
///
/// The integral uses the finite-element quadrature: it is the total of the
/// component — the resultant of a distributed force **density**, or the measure
/// of the domain for a field equal to 1. For a field of nodal forces, already
/// integrated, the resultant is a plain sum instead: `field.sum(component)`.
///
/// # Parameters
///
/// - `field`: a `NodeField` or an `ElementField`. A `NodeField` is interpolated
///   with the shape functions, `∫ Σ_i f_i N_i dΩ`; an `ElementField` is
///   integrated from its Gauss-point values, `Σ_cell Σ_g f·|J|·w`.
/// - `component`: the component to integrate.
/// - `fespace`: the FE space that interpolates a `NodeField` and gives the
///   quadrature — required for a `NodeField`, ignored for an `ElementField`.
///
/// # Returns
///
/// The integral, a `float`.
///
/// # Raises
///
/// - `TypeError` if `field` is a `NodeField` without `fespace`, or neither a
///   `NodeField` nor an `ElementField`.
/// - `RuntimeError` if a `NodeField` does not define `component` at every node
///   of `fespace`, or if no zone of an `ElementField` carries it.
///
/// # Notes
///
/// The cells are summed in parallel: the last bit of the result may change with
/// the number of threads.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, component, fespace=None))]
pub fn integral(
    field: &Bound<'_, PyAny>,
    component: &str,
    fespace: Option<PyRef<PyFiniteElementSpace>>,
) -> PyResult<f64> {
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let fes = fespace.ok_or_else(|| {
            PyTypeError::new_err("integral: a NodeField needs a FiniteElementSpace (fespace=...)")
        })?;
        return Ok(crate::ops::measure::integral(
            &f.inner, &fes.inner, component,
        )?);
    }
    if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        return Ok(crate::ops::measure::integral_element(&f.inner, component)?);
    }
    Err(PyTypeError::new_err(
        "integral: expected a NodeField (with fespace=...) or an ElementField",
    ))
}

/// Squared Euclidean norm `xᵀx = Σ v²` of a field.
///
/// The sum of the squares of every value of the field; a node that several
/// zones carry counts once. Restricted to some components, it measures the
/// norm of a residual on a subset of the degrees of freedom.
///
/// # Parameters
///
/// - `x`: a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`.
/// - `components`: the components to sum, the others ignored; `None` takes
///   every component. A name the field does not carry is ignored.
///
/// # Returns
///
/// The squared norm, a `float`; `0.0` for a field carrying none of
/// `components`.
///
/// # Raises
///
/// `TypeError` if `x` is none of the four field types.
///
/// # Notes
///
/// The sum is gathered in parallel: the last bit of the result may change with
/// the number of threads.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (x, components = None))]
pub fn xtx(x: &Bound<'_, PyAny>, components: Option<Vec<String>>) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    let names: Option<Vec<&str>> = components
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let names = names.as_deref();
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        return Ok(Field::xtx(&a.inner, names));
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        return Ok(Field::xtx(&a.inner, names));
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        return Ok(SubField::xtx(&*a.handle.read(), names));
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        return Ok(SubField::xtx(&*a.handle.read(), names));
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

/// Global scalar product `Σ xᵢ·yᵢ` of two whole fields, a single number.
///
/// The inner product of fields used for energies (`F·u`) and residual norms.
/// The sum runs over the supports both fields carry and, on each, over the
/// products `pairs` names; a node that several zones carry counts once. A
/// support on one side only has no counterpart and adds nothing.
///
/// # Parameters
///
/// - `x`: a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`.
/// - `y`: a field of the same type as `x`; two sub-fields must sit on the same
///   support.
/// - `pairs`: the products to sum, `(a, b)` multiplying component `a` of `x` by
///   component `b` of `y` — a displacement by its force,
///   `[("u_x", "f_x"), ("u_y", "f_y")]`. A component may appear in several
///   pairs; a pair one side does not carry adds nothing. `None` multiplies
///   each component by its namesake, whatever the order of the components.
///
/// # Returns
///
/// The scalar product, a `float`.
///
/// # Raises
///
/// - `TypeError` if `y` is not of the type of `x`, or `x` none of the four.
/// - `RuntimeError` if two sub-fields sit on different supports.
///
/// # Notes
///
/// The sum is gathered in parallel: the last bit of the result may change with
/// the number of threads.
///
/// # See also
///
/// `pyrucast.field.psca`, the scalar product node by node: a field, one value
/// per node.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (x, y, pairs = None))]
pub fn xty(
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    pairs: Option<Vec<(String, String)>>,
) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    let pairs: Option<Vec<(&str, &str)>> = pairs
        .as_ref()
        .map(|v| v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect());
    let pairs = pairs.as_deref();
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be NodeFields"))?;
        return Ok(a.inner.dot_field(&b.inner, pairs));
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be ElementFields"))?;
        return Ok(a.inner.dot_field(&b.inner, pairs));
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubNodeFields"))?;
        return Ok(a.handle.read().dot(&*b.handle.read(), pairs)?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubElementFields"))?;
        return Ok(a.handle.read().dot(&*b.handle.read(), pairs)?);
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}
