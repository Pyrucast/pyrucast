//! Python wrappers for [`crate::ops::export`] — write meshes and fields to
//! external file formats.

use crate::containers::mesh::Mesh;
use crate::py::element_field::PyElementField;
use crate::py::evolution::PyEvolution;
use crate::py::mesh::PyMesh;
use crate::py::node_field::PyNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

/// Write `mesh`, and a field if given, to a legacy VTK file for ParaView.
///
/// The file is an `UNSTRUCTURED_GRID` that ParaView reads natively. Every
/// sub-mesh is written; a 2-D `Coords` is completed with `z = 0`.
///
/// # Parameters
///
/// - `mesh`: the mesh to write.
/// - `path`: the file to write. For an `Evolution`, the index of the series, a
///   `.vtk.series` file.
/// - `field`: what to write on the mesh:
///   - `None`: the geometry alone;
///   - a `NodeField`: `POINT_DATA`, one scalar array per component, the nodal
///     value at each point (`0` where the field defines nothing);
///   - an `ElementField`: `CELL_DATA`, one array per component, the mean of the
///     Gauss values of each cell. The field must come from a space built on
///     **this** mesh, so that its cells line up;
///   - an `Evolution` of node or element fields: a **time series**. Each
///     tabulated value goes to its own file next to `path`, `stem_0000.vtk`,
///     `stem_0001.vtk`, … — `stem` being the name of `path` up to its first
///     dot —, and ParaView opens the index as one dataset with a time slider.
/// - `binary`: `False` writes text; `True` writes raw big-endian numbers, much
///   smaller and faster to read for a large mesh.
///
/// # Raises
///
/// - `TypeError` if `field` is none of the types above.
/// - `RuntimeError` if a file cannot be written.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, path, field=None, binary=false))]
pub fn export_vtk(
    mesh: PyRef<PyMesh>,
    path: &str,
    field: Option<&Bound<'_, PyAny>>,
    binary: bool,
) -> PyResult<()> {
    use crate::ops::export::VtkEncoding;
    let path = Path::new(path);
    let encoding = if binary {
        VtkEncoding::Binary
    } else {
        VtkEncoding::Ascii
    };
    match field {
        None => crate::ops::export::write_vtk_mesh(&mesh.inner, path, encoding)?,
        Some(obj) => {
            if let Ok(nf) = obj.extract::<PyRef<PyNodeField>>() {
                crate::ops::export::write_vtk_node_field(&mesh.inner, &nf.inner, path, encoding)?;
            } else if let Ok(ef) = obj.extract::<PyRef<PyElementField>>() {
                crate::ops::export::write_vtk_element_field(
                    &mesh.inner,
                    &ef.inner,
                    path,
                    encoding,
                )?;
            } else if let Ok(ev) = obj.extract::<PyRef<PyEvolution>>() {
                crate::ops::export::write_vtk_series(&mesh.inner, &ev.inner, path, encoding)?;
            } else {
                return Err(PyTypeError::new_err(
                    "field must be a NodeField, an ElementField or an Evolution",
                ));
            }
        }
    }
    Ok(())
}

/// Lay meshes and fields out as flat arrays, the shape `mesh.from_arrays` reads.
///
/// It is the one exit every exchange format goes through
/// (`pyrucast.export.to_gmsh`, `pyrucast.export.to_medcoupling`, the VTK
/// writer), and the mirror of `pyrucast.mesh.from_arrays`. The nodes are
/// numbered in the order they first appear; a cell present in several groups
/// is written once.
///
/// # Parameters
///
/// - `groups`: a `dict` from group name to `Mesh`, all on one `Coords`.
/// - `node_fields`: `NodeField`s, one row per exported node, `0` where a field
///   defines nothing.
/// - `element_fields`: `ElementField`s, on the cells their zones cover.
/// - `gauss`: `False` gives the Gauss **mean** per cell; `True` the raw values
///   per Gauss point, along with each type's rule.
/// - `order`: the node numbering inside a cell: `"pyrucast"` (the same as
///   VTK), `"gmsh"` or `"med"`.
/// - `first_tag`: the first node and cell tag (gmsh counts from 1,
///   medcoupling from 0).
///
/// # Returns
///
/// A `dict`:
///
/// - `"node_tags"`, `"node_coords"` (`dim` per node), `"dim"`;
/// - `"blocks"` — `(element_type, node_tags, cell_tags, groups)` per block of
///   cells sharing a type and a combination of groups; a cell of several
///   groups is exported once;
/// - `"node_fields"` — `(components, node_tags, values)`;
/// - `"cell_fields"` — `(components, cell_tags, values, layout)`, `layout`
///   being `"cell"` or the list of `(element_type, ref_nodes, xi, weights)`
///   rules, in pyrucast's reference element numbered in `order`.
///
/// Every array is an `Array`: read-only, and read **without a copy** by
/// `numpy.asarray` or `memoryview`.
///
/// # Raises
///
/// - `TypeError` if a value of `groups` is not a `Mesh`.
/// - `ValueError` if `order` is unknown.
/// - `RuntimeError` if the groups do not all lie on one `Coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (groups, *, node_fields = Vec::new(), element_fields = Vec::new(), gauss = false, order = crate::ops::mesh::NodeOrder::Pyrucast, first_tag = 1))]
pub fn to_arrays<'py>(
    py: Python<'py>,
    groups: &Bound<'py, PyDict>,
    node_fields: Vec<Py<PyNodeField>>,
    element_fields: Vec<Py<PyElementField>>,
    gauss: bool,
    order: crate::ops::mesh::NodeOrder,
    first_tag: i64,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::ops::export::arrays::{to_arrays, ElementLayout};
    use crate::py::arrays::PyArray;
    let named: Vec<(String, PyRef<'py, PyMesh>)> = groups
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<PyRef<'py, PyMesh>>()?)))
        .collect::<PyResult<_>>()?;
    let meshes: Vec<(String, &Mesh)> = named.iter().map(|(k, m)| (k.clone(), &m.inner)).collect();
    let nodes: Vec<PyRef<'py, PyNodeField>> =
        node_fields.iter().map(|f| f.bind(py).borrow()).collect();
    let elems: Vec<PyRef<'py, PyElementField>> =
        element_fields.iter().map(|f| f.bind(py).borrow()).collect();
    let layout = if gauss {
        ElementLayout::Gauss
    } else {
        ElementLayout::Cell
    };
    let node_refs: Vec<_> = nodes.iter().map(|f| &f.inner).collect();
    let elem_refs: Vec<_> = elems.iter().map(|f| (&f.inner, layout)).collect();
    let out = to_arrays(&meshes, &node_refs, &elem_refs, order, first_tag)?;

    let dict = PyDict::new(py);
    let node_tags = Py::new(py, PyArray::i64(out.node_tags))?;
    dict.set_item("node_tags", node_tags.clone_ref(py))?;
    dict.set_item("node_coords", PyArray::f64(out.node_coords))?;
    dict.set_item("dim", out.dim)?;
    let blocks = out
        .blocks
        .into_iter()
        .map(|b| {
            Ok((
                b.element_type.name(),
                Py::new(py, PyArray::i64(b.node_tags))?,
                Py::new(py, PyArray::i64(b.cell_tags))?,
                b.groups,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("blocks", blocks)?;
    let node_out = out
        .node_fields
        .into_iter()
        .map(|f| {
            Ok((
                f.components,
                node_tags.clone_ref(py),
                Py::new(py, PyArray::f64(f.values))?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("node_fields", node_out)?;
    let cell_out = out
        .cell_fields
        .into_iter()
        .map(|f| {
            let layout: Bound<'py, PyAny> = if gauss {
                f.rules
                    .into_iter()
                    .map(|r| {
                        Ok((
                            r.element_type.name(),
                            Py::new(py, PyArray::f64(r.ref_nodes))?,
                            Py::new(py, PyArray::f64(r.xi))?,
                            Py::new(py, PyArray::f64(r.weights))?,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()?
                    .into_pyobject(py)?
                    .into_any()
            } else {
                "cell".into_pyobject(py)?.into_any()
            };
            Ok((
                f.components,
                Py::new(py, PyArray::i64(f.cell_tags))?,
                Py::new(py, PyArray::f64(f.values))?,
                layout,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("cell_fields", cell_out)?;
    Ok(dict)
}
