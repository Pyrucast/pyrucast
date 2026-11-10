//! Python wrappers for [`crate::ops::evolution`] — the operators that
//! produce an `Evolution` from other containers.

use crate::py::evolution::PyEvolution;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// The values of `field` along `line`, one curve per component: the value
/// against the distance travelled along the line.
///
/// `line` is a POI1 mesh, read in the order of its cells, zone after zone:
/// that order is the path. The abscissa of a point is the length of the
/// path up to it — the sum of the straight distances from one point to the
/// next —, `0` at the first point; its ordinate, the value of the
/// component at that node (a node the field does not define reads zero).
/// `components=None` takes every component of the field, in its order;
/// each gives one `SubEvolution` of the returned `Evolution`, its
/// `ordinate_type` the component's name. A field of another mesh, or a
/// field at the Gauss points, is brought onto the line's points first with
/// `pyrucast.node_field.project`.
///
/// Raises when `line` is not POI1, when two consecutive points coincide,
/// for a component the field does not carry, and when the field and
/// `line` hang from different `Coords`.
#[py_op(method_on = PyNodeField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, line, components=None))]
pub fn along(
    field: PyRef<PyNodeField>,
    line: PyRef<PyMesh>,
    components: Option<Vec<String>>,
) -> PyResult<PyEvolution> {
    let wanted =
        components.unwrap_or_else(|| crate::containers::field::Field::components(&field.inner));
    let refs: Vec<&str> = wanted.iter().map(String::as_str).collect();
    Ok(PyEvolution {
        inner: crate::ops::evolution::along(&field.inner, &line.inner, &refs)?,
    })
}
