//! Python wrappers for [`crate::ops::evolution`] — the operators that
//! produce an `Evolution` from other containers.

use crate::py::evolution::PyEvolution;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// The values of `field` along `line`, one curve per component.
///
/// Each curve gives the value against the distance travelled along the line.
/// The order of the cells of `line`, zone after zone, is the path. The
/// abscissa of a point is the length of the path up to it — the sum of the
/// straight distances from one point to the next —, `0` at the first point;
/// its ordinate is the value of the component at that node (a node the field
/// does not define reads zero).
///
/// # Parameters
///
/// - `field`: the field to read, at the nodes. A field of another mesh, or a
///   field at the Gauss points, is first brought onto the points of the line
///   with `pyrucast.node_field.project`.
/// - `line`: the path, a POI1 mesh on the same `Coords` as `field`.
/// - `components`: the components to read, one curve each; `None` takes every
///   component of the field, in its order.
///
/// # Returns
///
/// An `Evolution` with one `SubEvolution` per component, in the order of
/// `components`, whose `ordinate_type` is the component's name.
///
/// # Raises
///
/// `RuntimeError` if `line` is not POI1, if two consecutive points coincide
/// (the abscissa would repeat), if `field` does not carry one of
/// `components`, or if `field` and `line` hang from different `Coords`.
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
