//! Python wrappers for [`crate::archive`] — save a graph of objects, read it
//! back sharing what it shared.
//!
//! A dictionary going out, a dictionary coming back. The keys are yours: an
//! archive key may carry a space, an accent, a unit — anything a Python string
//! can hold.

use crate::archive::{self, ArchiveRoot, Root, Value};
use crate::py::coords::PyCoords;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::evolution::{PyEvolution, PySubEvolution};
use crate::py::finite_element_space::{PyFiniteElementSpace, PySubFiniteElementSpace};
use crate::py::matrix::{PyMatrix, PySubMatrix};
use crate::py::mesh::{PyMesh, PySubMesh};
use crate::py::model::{PyModel, PySubModel};
use crate::py::node_field::{PyNodeField, PySubNodeField};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

/// One value of the dictionary, held for the duration of the save.
///
/// An aggregate is kept as the caller's own borrow rather than a copy — an
/// aggregate is cheap to clone but `Matrix` deliberately is not clonable at
/// all, and copying to write would be the wrong habit anyway. Sub-objects are
/// handles, so sharing them *is* the cheap thing.
enum Held<'py> {
    Coords(PyRef<'py, PyCoords>),
    Mesh(PyRef<'py, PyMesh>),
    SubMesh(PyRef<'py, PySubMesh>),
    Fespace(PyRef<'py, PyFiniteElementSpace>),
    SubFespace(PyRef<'py, PySubFiniteElementSpace>),
    NodeField(PyRef<'py, PyNodeField>),
    SubNodeField(PyRef<'py, PySubNodeField>),
    ElementField(PyRef<'py, PyElementField>),
    SubElementField(PyRef<'py, PySubElementField>),
    Evolution(PyRef<'py, PyEvolution>),
    SubEvolution(PyRef<'py, PySubEvolution>),
    Model(PyRef<'py, PyModel>),
    SubModel(PyRef<'py, PySubModel>),
    Matrix(PyRef<'py, PyMatrix>),
    SubMatrix(PyRef<'py, PySubMatrix>),
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Bools(Vec<bool>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Texts(Vec<String>),
}

impl Held<'_> {
    /// The borrow `save` wants. One arm per kind, exhaustively.
    fn as_root(&self) -> &dyn ArchiveRoot {
        match self {
            Held::Coords(v) => &v.handle,
            Held::Mesh(v) => &v.inner,
            Held::SubMesh(v) => &v.handle,
            Held::Fespace(v) => &v.inner,
            Held::SubFespace(v) => &v.handle,
            Held::NodeField(v) => &v.inner,
            Held::SubNodeField(v) => &v.handle,
            Held::ElementField(v) => &v.inner,
            Held::SubElementField(v) => &v.handle,
            Held::Evolution(v) => &v.inner,
            Held::SubEvolution(v) => &v.handle,
            Held::Model(v) => &v.inner,
            Held::SubModel(v) => &v.handle,
            Held::Matrix(v) => &v.inner,
            Held::SubMatrix(v) => &v.handle,
            Held::Bool(v) => v,
            Held::Int(v) => v,
            Held::Float(v) => v,
            Held::Text(v) => v,
            Held::Bools(v) => v,
            Held::Ints(v) => v,
            Held::Floats(v) => v,
            Held::Texts(v) => v,
        }
    }
}

/// A homogeneous list of `bool` / `int` / `float` / `str`, or a clear refusal.
///
/// A list of lists, a dict, a mix of types: all say what is wrong and why,
/// rather than half-saving something.
fn extract_list<'py>(key: &str, list: &Bound<'py, PyList>) -> PyResult<Held<'py>> {
    if list.is_empty() {
        // Nothing to infer from; an empty list of strings and an empty list of
        // floats are the same object. Pick the one that costs nothing.
        return Ok(Held::Texts(Vec::new()));
    }
    let first = list.get_item(0)?;
    let kinds = |o: &Bound<'_, PyAny>| -> Option<u8> {
        // `bool` before `int`: in Python a bool *is* an int.
        if o.is_instance_of::<PyBool>() {
            Some(0)
        } else if o.is_instance_of::<PyInt>() {
            Some(1)
        } else if o.is_instance_of::<PyFloat>() {
            Some(2)
        } else if o.is_instance_of::<PyString>() {
            Some(3)
        } else {
            None
        }
    };
    let kind = kinds(&first).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "\"{key}\": a list may hold bool, int, float or str — not {}. \
             Nested lists and dicts are not archived.",
            first
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;
    for (i, item) in list.iter().enumerate() {
        if kinds(&item) != Some(kind) {
            return Err(PyTypeError::new_err(format!(
                "\"{key}\": a list must be homogeneous; item 0 is {} and item {i} is {}",
                first
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
                item.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            )));
        }
    }
    Ok(match kind {
        0 => Held::Bools(list.extract()?),
        1 => Held::Ints(extract_ints(key, list)?),
        2 => Held::Floats(list.extract()?),
        _ => Held::Texts(list.extract()?),
    })
}

/// Python integers are unbounded; the file stores 64 bits. Say so rather than
/// wrapping around.
fn extract_int(key: &str, obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    obj.extract::<i64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "\"{key}\": integer too large for the archive, which stores 64 bits \
             (|value| < 2^63). Save it as a str if you need it exact."
        ))
    })
}

fn extract_ints(key: &str, list: &Bound<'_, PyList>) -> PyResult<Vec<i64>> {
    list.iter().map(|o| extract_int(key, &o)).collect()
}

/// One entry of the dictionary handed to [`save`].
fn extract<'py>(key: &str, obj: &Bound<'py, PyAny>) -> PyResult<Held<'py>> {
    // Objects first: a `Py*` wrapper is not any of the scalar types.
    if let Ok(v) = obj.extract::<PyRef<'py, PyCoords>>() {
        return Ok(Held::Coords(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyMesh>>() {
        return Ok(Held::Mesh(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubMesh>>() {
        return Ok(Held::SubMesh(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyFiniteElementSpace>>() {
        return Ok(Held::Fespace(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubFiniteElementSpace>>() {
        return Ok(Held::SubFespace(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyNodeField>>() {
        return Ok(Held::NodeField(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubNodeField>>() {
        return Ok(Held::SubNodeField(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyElementField>>() {
        return Ok(Held::ElementField(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubElementField>>() {
        return Ok(Held::SubElementField(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyEvolution>>() {
        return Ok(Held::Evolution(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubEvolution>>() {
        return Ok(Held::SubEvolution(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyModel>>() {
        return Ok(Held::Model(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubModel>>() {
        return Ok(Held::SubModel(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PyMatrix>>() {
        return Ok(Held::Matrix(v));
    }
    if let Ok(v) = obj.extract::<PyRef<'py, PySubMatrix>>() {
        return Ok(Held::SubMatrix(v));
    }

    // Then the simple values. `bool` before `int`, since in Python it is one.
    if obj.is_instance_of::<PyBool>() {
        return Ok(Held::Bool(obj.extract()?));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Held::Int(extract_int(key, obj)?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Held::Float(obj.extract()?));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Held::Text(obj.extract()?));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return extract_list(key, list);
    }

    Err(PyTypeError::new_err(format!(
        "\"{key}\": {} cannot be archived. Save a pyrucast object, a bool, an \
         int, a float, a str, or a homogeneous list of those.",
        obj.get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default()
    )))
}

/// Turn one reloaded entry into the Python object that wraps it.
///
/// Exhaustive by construction: a type added to [`Root`] stops this compiling
/// until it is handled here too.
fn to_python(py: Python<'_>, root: Root) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match root {
        Root::Coords(handle) => Py::new(py, PyCoords { handle })?.into_any(),
        Root::Mesh(inner) => Py::new(py, PyMesh { inner })?.into_any(),
        Root::SubMesh(handle) => Py::new(py, PySubMesh { handle })?.into_any(),
        Root::FiniteElementSpace(inner) => Py::new(py, PyFiniteElementSpace { inner })?.into_any(),
        Root::SubFiniteElementSpace(handle) => {
            Py::new(py, PySubFiniteElementSpace { handle })?.into_any()
        }
        Root::NodeField(inner) => Py::new(py, PyNodeField { inner })?.into_any(),
        Root::SubNodeField(handle) => Py::new(py, PySubNodeField { handle })?.into_any(),
        Root::ElementField(inner) => Py::new(py, PyElementField { inner })?.into_any(),
        Root::SubElementField(handle) => Py::new(py, PySubElementField { handle })?.into_any(),
        Root::Evolution(inner) => Py::new(py, PyEvolution { inner })?.into_any(),
        Root::SubEvolution(handle) => Py::new(py, PySubEvolution { handle })?.into_any(),
        Root::Model(inner) => Py::new(py, PyModel { inner })?.into_any(),
        Root::SubModel(handle) => Py::new(py, PySubModel { handle })?.into_any(),
        Root::Matrix(inner) => Py::new(py, PyMatrix { inner })?.into_any(),
        Root::SubMatrix(handle) => Py::new(py, PySubMatrix { handle })?.into_any(),
        Root::Value(v) => match v {
            Value::Bool(x) => x.into_py_any(py)?,
            Value::Int(x) => x.into_py_any(py)?,
            Value::Float(x) => x.into_py_any(py)?,
            Value::Text(x) => x.into_py_any(py)?,
            Value::Bools(x) => x.into_py_any(py)?,
            Value::Ints(x) => x.into_py_any(py)?,
            Value::Floats(x) => x.into_py_any(py)?,
            Value::Texts(x) => x.into_py_any(py)?,
        },
    })
}

/// Write `objects` and everything they need to `path`.
///
/// The keys are yours — a space, an accent, a unit are all fine. Values may be
/// any pyrucast object, or a `bool` / `int` / `float` / `str` / homogeneous
/// list of those.
///
/// Objects shared by several entries are written **once**: two fields on one
/// support come back on one support, not on two copies of it. Saving the same
/// objects twice produces the same bytes.
///
/// Caches are never written — the assembled matrix, the factorization, the
/// colourings — because they can be rebuilt. Reference counts are not written
/// either: they are recounted at load, from the objects the file holds. A
/// `Node` you are holding is *not* archived.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn save(path: &str, objects: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut held: Vec<(String, Held)> = Vec::with_capacity(objects.len());
    for (key, value) in objects.iter() {
        let key: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("archive keys must be strings"))?;
        let value = extract(&key, &value)?;
        held.push((key, value));
    }
    let roots: Vec<(&str, &dyn ArchiveRoot)> = held
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_root()))
        .collect();
    archive::save(path, &roots)?;
    Ok(())
}

/// Read back what [`save`] wrote, as a dictionary.
///
/// The objects are **new**: nothing already alive in your session is touched,
/// they are simply added alongside.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn load(py: Python<'_>, path: &str) -> PyResult<Py<PyDict>> {
    let objects = archive::load(path)?;
    let out = PyDict::new(py);
    for (key, root) in objects.into_inner() {
        out.set_item(key, to_python(py, root)?)?;
    }
    Ok(out.unbind())
}
