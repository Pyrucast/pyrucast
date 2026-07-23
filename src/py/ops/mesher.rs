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
/// CW), so the result can feed straight back into `triangulate_surface`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn contour(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::contour(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Harmonise the orientation of a mesh's cells (cast3m `ORIE`).
///
/// Cells sharing a facet are made consistently oriented — all normals of a
/// surface point the same way, all segments of a curve run head-to-tail, all
/// volume cells share one handedness — in any dimension (SEG/TRI/QUA/TET/
/// PENTA/HEX, linear or quadratic). Each connected component is seeded by its
/// lowest-indexed cell, which keeps its orientation; the absolute sense is not
/// chosen (use `invert` to flip a whole mesh, e.g. a hole's boundary). Returns
/// a fresh mesh sharing the input's nodes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn orient(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::orient(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Reverse the orientation of every cell of a mesh (cast3m `INVE`).
///
/// Flips each cell's winding/traversal/handedness (POI1 cells are unchanged),
/// in any dimension. Combined with `orient`, this selects the inside/outside
/// sense of a closed contour or surface. Returns a fresh mesh sharing the
/// input's nodes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn invert(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::invert(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Build a line of `n_elems` elements from node `a` to node `b`.
///
/// `element_type` is `"SEG2"` (default) or `"SEG3"`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (a, b, n_elems, element_type="SEG2"))]
pub fn line(
    a: PyRef<PyNode>,
    b: PyRef<PyNode>,
    n_elems: usize,
    element_type: &str,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesher::line(a.as_node(), b.as_node(), n_elems, et)?;
    Ok(PyMesh { inner: mesh })
}

/// Build a closed circle mesh of `n_elems` elements, centred on `center`,
/// lying in the plane defined by the 3-component `normal`, with the given
/// `radius`.
///
/// `element_type` is `"SEG2"` (default) or `"SEG3"`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (center, normal, radius, n_elems, element_type="SEG2"))]
pub fn circle(
    center: PyRef<PyNode>,
    normal: Vec<f64>,
    radius: f64,
    n_elems: usize,
    element_type: &str,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesher::circle(center.as_node(), &normal, radius, n_elems, et)?;
    Ok(PyMesh { inner: mesh })
}

/// Build an open arc of `n_elems` elements from `a` to `b`, following the
/// circle centred on `center` that passes through both (the shorter arc is
/// built).
///
/// `element_type` is `"SEG2"` (default) or `"SEG3"`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (a, center, b, n_elems, element_type="SEG2"))]
pub fn arc(
    a: PyRef<PyNode>,
    center: PyRef<PyNode>,
    b: PyRef<PyNode>,
    n_elems: usize,
    element_type: &str,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesher::arc(a.as_node(), center.as_node(), b.as_node(), n_elems, et)?;
    Ok(PyMesh { inner: mesh })
}

/// Sweep two SEG2 line meshes into a mesh of `element_type`, building
/// `n_layers` layers between `mesh_a` and `mesh_b`.
///
/// `element_type` is `"QUA4"` (default), `"TRI3"`, `"QUA8"`, `"QUA9"` or
/// `"TRI6"` — a `QUA4` mesh is always built first, then converted (diagonal
/// split for the triangles, promoted to quadratic for `QUA8`/`QUA9`/`TRI6`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh_a, mesh_b, n_layers, element_type="QUA4"))]
pub fn sweep(
    mesh_a: PyRef<PyMesh>,
    mesh_b: PyRef<PyMesh>,
    n_layers: usize,
    element_type: &str,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesher::sweep(&mesh_a.inner, &mesh_b.inner, n_layers, et)?;
    Ok(PyMesh { inner: mesh })
}

/// Build a structured surface bounded by four `SEG2` sides, by transfinite
/// interpolation (the Coons-patch generalization of `sweep` from two lines
/// to four). `side1`/`side3` and `side2`/`side4` are the two pairs of
/// **opposite** sides and must each have the same element count; the four
/// sides must form a closed contour, `side1 → side2 → side3 → side4 →
/// side1`, each sharing its end node with the next side's start node.
///
/// `element_type` is `"QUA4"` (default), `"TRI3"`, `"QUA8"`, `"QUA9"` or
/// `"TRI6"` — same conversion path as `sweep`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (side1, side2, side3, side4, element_type="QUA4"))]
pub fn transfinite(
    side1: PyRef<PyMesh>,
    side2: PyRef<PyMesh>,
    side3: PyRef<PyMesh>,
    side4: PyRef<PyMesh>,
    element_type: &str,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesher::transfinite(
        &side1.inner,
        &side2.inner,
        &side3.inner,
        &side4.inner,
        et,
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Extrude `mesh` by `n_layers` layers along `direction` (the total
/// displacement vector). SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn extrude(mesh: PyRef<PyMesh>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::extrude(&mesh.inner, &direction, n_layers)?;
    Ok(PyMesh { inner: result })
}

/// Sweep two matching surface meshes into a solid mesh, building `n_layers`
/// layers between `mesh_a` and `mesh_b`. The 3-D companion of `sweep`:
/// TRI3 faces → PENTA6 prisms, QUA4 faces → HEX8 hexahedra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sweep_solid(
    mesh_a: PyRef<PyMesh>,
    mesh_b: PyRef<PyMesh>,
    n_layers: usize,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesher::sweep_solid(&mesh_a.inner, &mesh_b.inner, n_layers)?;
    Ok(PyMesh { inner: mesh })
}

/// Build the **quadratic** (Lagrange-2) copy of a linear mesh: each element
/// type is bumped to its quadratic sibling (TRI3→TRI6, HEX8→HEX20, …). Corner
/// nodes are re-used; one mid-edge node is created per edge (at the midpoint)
/// and shared between the cells that use it. The original mesh is untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_quadratic(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::to_quadratic(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Translate `mesh` by `vector`, returning a fresh copy with its own nodes
/// (the original is left untouched). `vector` matches the mesh dimension.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn translate(mesh: PyRef<PyMesh>, vector: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::translate(&mesh.inner, &vector)?;
    Ok(PyMesh { inner: result })
}

/// Rotate `mesh` by `angle` (radians) about `center`, returning a fresh copy
/// with its own nodes (the original is left untouched).
///
/// In 2-D, `center` is a point and `axis` is ignored. In 3-D, the rotation is
/// about the line through `center` directed by `axis` (right-handed); `axis`
/// is required.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle, center, axis=None))]
pub fn rotate(
    mesh: PyRef<PyMesh>,
    angle: f64,
    center: Vec<f64>,
    axis: Option<Vec<f64>>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesher::rotate(&mesh.inner, angle, &center, axis.as_deref())?;
    Ok(PyMesh { inner: result })
}

/// Mesh the interior of a closed SEG2 `contour` with `element_type` cells
/// using a constrained-Delaunay + Ruppert-refinement mesher.
///
/// `contour` holds **one or more** closed SEG2 loops, each oriented by the
/// caller: a **counter-clockwise** loop is a domain's outer boundary, a
/// **clockwise** loop is a hole (contained in an outer loop). Several
/// disjoint CCW loops mesh several independent domains at once. `size` sets
/// the target element edge length; `None` uses the mean boundary edge length
/// per domain. `element_type` is "TRI3" or "QUA4" (QUA4 is quad-dominant:
/// the result may also carry a few boundary triangles). The contour may be
/// 2-D or a planar loop in 3-D (meshed in its best-fit plane, lifted back).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None))]
pub fn triangulate_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: &str,
    size: Option<f64>,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    // Poll Python signals while meshing so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesher::triangulate_surface_cancellable(
        &contour.inner,
        et,
        size,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Fill the interior of a closed **TRI3** surface `envelope` with TET4 cells
/// using a size-controlled **Delaunay** fill that **creates interior nodes** —
/// the 3-D companion of `triangulate_surface`.
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

/// Read a gmsh `.msh` file (ASCII or binary, MSH 2.2 or 4.1) into a `dict`
/// mapping each physical group name to a `Mesh`, adding the nodes to `coords`.
///
/// The nodes land in the `coords` you pass, so you keep the handle needed to
/// pose boundary conditions on a named region. Its dimension decides how
/// many of gmsh's three coordinates are kept (a 2-D `Coords` flattens onto
/// `xy`). All returned meshes share that single `Coords`, so a node on the
/// boundary between two named groups is the same node on both sides. Inside
/// a group's mesh there is one submesh per element type; elements with no
/// physical group land under the key `"<ungrouped>"`. The dict preserves the
/// file order.
///
/// Supported element types: POI1, SEG2, TRI3, QUA4, TET4, HEX8; any other
/// gmsh type raises.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn read_gmsh<'py>(
    py: Python<'py>,
    coords: PyRef<PyCoords>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = crate::ops::mesher::read_gmsh(coords.handle.clone(), std::path::Path::new(path))?;
    groups_to_dict(py, groups)
}

/// Like `read_gmsh`, but parsing the `.msh` text already held in a string
/// instead of reading from a path. Same `dict[str, Mesh]` result.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn read_gmsh_str<'py>(
    py: Python<'py>,
    coords: PyRef<PyCoords>,
    text: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = crate::ops::mesher::read_gmsh_str(coords.handle.clone(), text)?;
    groups_to_dict(py, groups)
}
