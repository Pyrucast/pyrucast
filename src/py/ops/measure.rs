//! Python wrappers for [`crate::ops::measure`] — the operators that reduce
//! containers to a number.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Integral `∫_Ω f dΩ` of a field over its support, using the finite-element
/// quadrature — the total of one `component` (e.g. the resultant of a
/// distributed force **density**).
///
/// - `NodeField`: interpolates with the shape functions, `∫ Σ_i f_i N_i dΩ` —
///   `fespace` is **required**.
/// - `ElementField`: integrates the Gauss-point values directly,
///   `Σ_cell Σ_g f·|J|·w` — `fespace` is ignored.
///
/// For a field of already-integrated **nodal** forces, the resultant is a plain
/// sum instead: `field.sum(component)`.
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

/// Squared Euclidean norm `xᵀx = Σ v²` of a field (Cast3M `XTX`) — the sum of
/// squares over every value of every zone. Accepts a `NodeField`,
/// `SubNodeField`, `ElementField` or `SubElementField`.
///
/// With `components` given, the sum is restricted to those components only
/// (the rest ignored); by default every component is taken.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (x, components = None))]
pub fn xtx(x: &Bound<'_, PyAny>, components: Option<Vec<String>>) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    let refs: Option<Vec<&str>> = components
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    macro_rules! aggregate {
        ($inner:expr) => {
            match &refs {
                Some(c) => $inner.xtx_components(c)?,
                None => $inner.xtx(),
            }
        };
    }
    macro_rules! sub {
        ($guard:expr) => {
            match &refs {
                Some(c) => $guard.xtx_components(c)?,
                None => $guard.xtx(),
            }
        };
    }
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        return Ok(aggregate!(a.inner));
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        return Ok(aggregate!(a.inner));
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        return Ok(sub!(a.handle.read()));
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        return Ok(sub!(a.handle.read()));
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

/// Global scalar product `∑ xᵢ · yᵢ` of two **whole** fields — Cast3M's `XTY`.
///
/// `x` and `y` must be the same flavour (`NodeField` / `SubNodeField` /
/// `ElementField` / `SubElementField`), sit on the same support/decomposition,
/// and carry the same components (aligned by name). Returns a single float —
/// the field inner product used for energies (`F·u`), residual norms, etc.
///
/// For the **node-by-node** scalar product (a field, one value per node),
/// see `pyrucast.field.psca`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn xty(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be NodeFields"))?;
        return Ok(a.inner.dot_field(&b.inner)?);
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be ElementFields"))?;
        return Ok(a.inner.dot_field(&b.inner)?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubNodeFields"))?;
        return Ok(a.handle.read().dot(&*b.handle.read())?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubElementFields"))?;
        return Ok(a.handle.read().dot(&*b.handle.read())?);
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}
