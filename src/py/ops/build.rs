//! Python wrappers for the material-field builders in [`crate::ops::build`].
//!
//! Free functions that build a material [`PyElementField`] (or a single
//! [`PySubElementField`]) from a model. Kept here — mirroring
//! `src/ops/build/` — per the `py/ops/` convention.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::model::{PyModel, PySubModel};
use crate::store::{insert, read};
use pyo3::prelude::*;

/// Build the material `SubElementField` of one sub-model.
///
/// `sub_material_field(sub_model, [("k", 1.0), ...])` — fresh
/// SubElementField on the sub-model's FE subspace, pre-filled with the
/// given uniform value per declared component. Errors for physics that
/// need no material (e.g. Dirichlet).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sub_material_field(
    sub_model: PyRef<PySubModel>,
    components_and_values: Vec<(String, f64)>,
) -> PyResult<PySubElementField> {
    let pairs: Vec<(&str, f64)> = components_and_values
        .iter()
        .map(|(c, v)| (c.as_str(), *v))
        .collect();
    let sub = crate::ops::build::sub_material_field(&*read(&sub_model.handle)?, &pairs)?;
    Ok(PySubElementField { handle: insert(sub) })
}

/// Build a material `ElementField` applying the same uniform
/// `(component, value)` pairs to every material-hungry sub-model of
/// `model`. Sub-models that need no material (Dirichlet, …) are skipped.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn material_field(
    model: PyRef<PyModel>,
    components_and_values: Vec<(String, f64)>,
) -> PyResult<PyElementField> {
    let pairs: Vec<(&str, f64)> = components_and_values
        .iter()
        .map(|(c, v)| (c.as_str(), *v))
        .collect();
    let ef = crate::ops::build::material_field(&model.inner, &pairs)?;
    Ok(PyElementField { inner: ef })
}

/// Build a material `ElementField` where each sub-model gets its own
/// `(component, value)` list. The outer list length must equal
/// `model.len()`. An empty inner list **skips** the matching
/// sub-model (typical for Dirichlet).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn material_field_per_sub_model(
    model: PyRef<PyModel>,
    components_and_values_per_sub_model: Vec<Vec<(String, f64)>>,
) -> PyResult<PyElementField> {
    // Materialise each inner Vec<(String, f64)> into a Vec<(&str, f64)>,
    // then collect slices into a Vec<&[(&str, f64)]>.
    let owned: Vec<Vec<(&str, f64)>> = components_and_values_per_sub_model
        .iter()
        .map(|v| v.iter().map(|(c, x)| (c.as_str(), *x)).collect())
        .collect();
    let slices: Vec<&[(&str, f64)]> = owned.iter().map(|v| v.as_slice()).collect();
    let ef = crate::ops::build::material_field_per_sub_model(&model.inner, &slices)?;
    Ok(PyElementField { inner: ef })
}
