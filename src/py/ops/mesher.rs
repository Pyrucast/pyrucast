//! Python wrappers for the mesher operations in [`crate::ops::mesher`].
//!
//! Free functions that build or transform a [`PyMesh`]. Kept here —
//! mirroring `src/ops/mesher/` — rather than on the `Mesh` class, per the
//! `py/ops/` convention (operations live with operations).

use crate::containers::mesh::ElementType;
use crate::containers::mesh::{Mesh, Node, SubMesh};
use crate::py::coords::PyCoords;
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use crate::py::signals::PySignals;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build a points (POI1) mesh holding every live node of `coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn from_live_nodes(coords: PyRef<PyCoords>) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::from_live_nodes(coords.handle.clone())?;
    Ok(PyMesh { inner: mesh })
}

/// Build a points (POI1) mesh with one point per node in `nodes`.
///
/// The Coords is taken from the nodes themselves (every `Node`
/// carries its own), so no Coords argument is needed. Returns a
/// Mesh with a single POI1 submesh; raises if `nodes` is empty.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn poi1_from_nodes(nodes: Vec<PyRef<PyNode>>) -> PyResult<PyMesh> {
    let ns: Vec<Node> = nodes.iter().map(|n| n.as_node().clone()).collect();
    let sm = SubMesh::poi1_from_nodes(&ns)?;
    Ok(PyMesh {
        inner: Mesh::from_submesh(sm),
    })
}

/// Convert a mesh to POI1, submesh by submesh.
///
/// Returns a new mesh with the same number of submeshes; each output
/// submesh is a POI1 submesh holding the de-duplicated nodes of the
/// corresponding input submesh, in order of first appearance.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_poi1(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::to_poi1(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Build a POI1 mesh of per-element centroids (centres of gravity), submesh
/// by submesh.
///
/// Returns a new mesh with the same number of submeshes; each output submesh
/// is a POI1 submesh with one **fresh** node per element of the corresponding
/// input submesh, placed at the element's centroid. A POI1 input is therefore
/// copied to colocated fresh nodes — handy to mint Lagrange-multiplier support
/// nodes from a set of constrained points.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn barycenter(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::barycenter(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Keep the elements of `mesh` resting on the nodes of `points`
/// (Cast3m `ELEM … APPUYE`).
///
/// Only the node set referenced by `points` matters (typically a POI1
/// points mesh). With `strict=True` a cell is kept when **all** its nodes
/// are in the set (`APPUYE STRICTEMENT`); with `strict=False` when **at
/// least one** is (`APPUYE`). The result mirrors `mesh` submesh by submesh
/// (same types, possibly-empty zones). Both meshes must share the same
/// `Coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, points, strict=true))]
pub fn elements_on(mesh: PyRef<PyMesh>, points: PyRef<PyMesh>, strict: bool) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::elements_on(&mesh.inner, &points.inner, strict)?;
    Ok(PyMesh { inner: result })
}

// `consolidate(mesh)` is exposed by the type-dispatching top-level
// wrapper in `crate::py::ops::consolidate` (shared with NodeField).

/// Weld together nodes closer than `tol`, redirecting the connectivity to one
/// representative per cluster.
///
/// Returns a new mesh mirroring `mesh` (same submeshes, types and colours)
/// with welded-away nodes redirected to their cluster representative — the
/// smallest-id node of the cluster, which keeps its own coordinates (no
/// averaging). Cells that collapse onto a repeated node (a degenerate segment,
/// triangle, …) are dropped. `tol` must be ≥ 0; `tol = 0` welds only exactly
/// coincident nodes. `mesh` itself is left untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge_nodes(mesh: PyRef<PyMesh>, tol: f64) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::merge_nodes(&mesh.inner, tol)?;
    Ok(PyMesh { inner: result })
}

/// Extract the boundary of a surface mesh (TRI3/QUA4) as closed SEG2 loops.
///
/// An element edge used by exactly one cell is a boundary edge; the boundary
/// edges (pooled across all surface submeshes) are chained into closed loops.
/// Returns a Mesh with one SEG2 submesh per loop — a single loop for a
/// simply-connected domain, several when the domain has holes or disjoint
/// pieces. Loops keep the CCW boundary orientation (outer loop CCW, holes
/// CW), so the result can feed straight back into `surface` / `fill_surface`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn contour(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::contour(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Build a line of `n_elems` SEG2 elements from node `a` to node `b`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn line_seg2(a: PyRef<PyNode>, b: PyRef<PyNode>, n_elems: usize) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::line_seg2(a.as_node(), b.as_node(), n_elems)?;
    Ok(PyMesh { inner: mesh })
}

/// Build a closed circle of `n_elems` SEG2 elements, centred on `center`
/// in the plane defined by `normal`, with the given `radius`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn circle_seg2(
    center: PyRef<PyNode>,
    normal: Vec<f64>,
    radius: f64,
    n_elems: usize,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::circle_seg2(center.as_node(), &normal, radius, n_elems)?;
    Ok(PyMesh { inner: mesh })
}

/// Sweep two SEG2 line meshes into a QUA4 mesh, building `n_layers` layers
/// of quads between `mesh_a` and `mesh_b`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sweep_qua4(
    mesh_a: PyRef<PyMesh>,
    mesh_b: PyRef<PyMesh>,
    n_layers: usize,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::sweep_qua4(&mesh_a.inner, &mesh_b.inner, n_layers)?;
    Ok(PyMesh { inner: mesh })
}

/// Extrude `mesh` by `n_layers` layers along `direction` (the total
/// displacement vector). SEG2 → QUA4, QUA4 → HEX8.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn extrude(mesh: PyRef<PyMesh>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::extrude(&mesh.inner, &direction, n_layers)?;
    Ok(PyMesh { inner: result })
}

/// Fill the interior of a closed SEG2 `contour` with `element_type` cells
/// (triangulation). `max_edge_length` / `min_angle_deg` optionally refine
/// the result.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, max_edge_length=None, min_angle_deg=None))]
pub fn fill_surface(
    contour: PyRef<PyMesh>,
    element_type: &str,
    max_edge_length: Option<f64>,
    min_angle_deg: Option<f64>,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let refinement = if max_edge_length.is_some() || min_angle_deg.is_some() {
        Some(crate::ops::mesher::triangulation::RefinementOptions {
            max_edge_length,
            min_angle_deg,
        })
    } else {
        None
    };
    let mesh = crate::ops::mesher::fill_surface(&contour.inner, et, refinement)?;
    Ok(PyMesh { inner: mesh })
}

/// Mesh the interior of a closed SEG2 `contour` with `element_type` cells
/// using a size-controlled advancing front that **creates interior nodes**
/// (unlike `fill_surface`, which only triangulates the contour nodes).
///
/// `size` sets the target element edge length; `None` uses the mean length
/// of the contour's segments. `element_type` is "TRI3" or "QUA4" (QUA4 is
/// quad-dominant: the result may also carry a few triangles). The contour
/// may be 2-D or a nearly planar loop in 3-D (projected, paved, lifted
/// back). A single contour is supported for now.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None))]
pub fn surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: &str,
    size: Option<f64>,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    // Poll Python signals while paving so a long mesh stays Ctrl+C-able.
    let mesh = crate::ops::mesher::surface_cancellable(&contour.inner, et, size, &PySignals(py))?;
    Ok(PyMesh { inner: mesh })
}

/// Fill the interior of a closed **TRI3** surface `envelope` with TET4 cells
/// using a size-controlled **Delaunay** fill that **creates interior nodes** —
/// the 3-D companion of `surface`.
///
/// `size` sets the target element edge length; `None` uses the mean edge
/// length of the envelope's faces. The envelope must be a closed,
/// consistently oriented TRI3 surface on a 3-D Coords. The result is a Mesh
/// with a single TET4 submesh; boundary nodes are reused. This first version
/// targets convex or mildly concave envelopes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (envelope, size=None))]
pub fn volume(py: Python<'_>, envelope: PyRef<PyMesh>, size: Option<f64>) -> PyResult<PyMesh> {
    // Poll Python signals while paving so a long mesh stays Ctrl+C-able.
    let mesh = crate::ops::mesher::volume_cancellable(&envelope.inner, size, &PySignals(py))?;
    Ok(PyMesh { inner: mesh })
}

/// Turn an ordered list of `(group name, Mesh)` pairs into a `dict` that
/// keeps the file order (Python dicts are insertion-ordered).
fn groups_to_dict<'py>(
    py: Python<'py>,
    groups: Vec<(String, Mesh)>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, mesh) in groups {
        dict.set_item(name, Py::new(py, PyMesh { inner: mesh })?)?;
    }
    Ok(dict)
}

/// Read a gmsh `.msh` file (ASCII MSH 2.2 or 4.1) into a `dict` mapping each
/// physical group name to a `Mesh`.
///
/// All returned meshes share a single `Coords`, so a node on the boundary
/// between two named groups is the same node on both sides — convenient to
/// pose boundary conditions on a named region. Inside a group's mesh there
/// is one submesh per element type. Elements with no physical group land
/// under the key `"<ungrouped>"`. The dict preserves the file order.
///
/// `dim` forces the coordinate dimension (extra coordinates dropped);
/// `None` infers it (2 if the mesh is planar on `z = 0`, else 3). Supported
/// element types: POI1, SEG2, TRI3, QUA4, TET4, HEX8; any other gmsh type
/// raises.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (path, dim=None))]
pub fn read_gmsh<'py>(
    py: Python<'py>,
    path: &str,
    dim: Option<u8>,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = crate::ops::mesher::read_gmsh(std::path::Path::new(path), dim)?;
    groups_to_dict(py, groups)
}

/// Like `read_gmsh`, but parsing the `.msh` text already held in a string
/// instead of reading from a path. Same `dict[str, Mesh]` result.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (text, dim=None))]
pub fn read_gmsh_str<'py>(
    py: Python<'py>,
    text: &str,
    dim: Option<u8>,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = crate::ops::mesher::read_gmsh_str(text, dim)?;
    groups_to_dict(py, groups)
}
