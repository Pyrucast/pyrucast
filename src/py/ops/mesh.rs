//! Python wrappers for the mesher operations in [`crate::ops::mesh`].
//!
//! Free functions that build or transform a [`PyMesh`]. Kept here —
//! mirroring `src/ops/mesher/` — rather than on the `Mesh` class, per the
//! `py/ops/` convention (operations live with operations).

use crate::atoms::ElementType;
use crate::containers::mesh::Mesh;
use crate::py::coords::PyCoords;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use crate::py::signals::PySignals;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build a points (POI1) mesh holding every live node of `coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn from_live_nodes(coords: PyRef<PyCoords>) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesh::from_live_nodes(coords.handle.clone())?;
    Ok(PyMesh { inner: mesh })
}

/// Convert a mesh to POI1, submesh by submesh.
///
/// Returns a new mesh with the same number of submeshes; each output
/// submesh is a POI1 submesh holding the de-duplicated nodes of the
/// corresponding input submesh, in order of first appearance.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_poi1(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::to_poi1(&mesh.inner)?;
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
    let result = crate::ops::mesh::barycenter(&mesh.inner)?;
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
    let result = crate::ops::mesh::elements_on(&mesh.inner, &points.inner, strict)?;
    Ok(PyMesh { inner: result })
}

// --- Node selection by geometric region (Cast3m `POIN … SPHE / CYLI / …`) ---
//
// One operator per (shape, side). They all return a POI1 mesh mirroring
// `mesh` submesh by submesh — one zone per input zone, possibly empty — and
// all take the same `tol`, the geometric precision of the test as a distance
// to the region's surface; `tol=None` means `1e-6 ×` the bounding-box
// diagonal. The odd one out is the *nearest* node, which returns a single
// node and lives on the class: `mesh.nearest_node(point)`.

/// Nodes **inside** the sphere of centre `center` and radius `radius`.
///
/// Keeps the nodes at a distance `≤ radius + tol` from `center`. In 2-D the
/// sphere is a disc — `center` just has to match the mesh dimension.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, center, radius, tol=None))]
pub fn points_in_sphere(
    mesh: PyRef<PyMesh>,
    center: Vec<f64>,
    radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_in_sphere(&mesh.inner, &center, radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the sphere of centre `center` and radius `radius`
/// (Cast3m `POIN … SPHE`).
///
/// Keeps the nodes whose distance to `center` is within `tol` of `radius`, on
/// either side. In 2-D this selects a circle.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, center, radius, tol=None))]
pub fn points_on_sphere(
    mesh: PyRef<PyMesh>,
    center: Vec<f64>,
    radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_on_sphere(&mesh.inner, &center, radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the plane through `origin` with normal `normal`
/// (Cast3m `POIN … PLAN`).
///
/// Keeps the nodes within `tol` of the plane — the usual way to grab a
/// boundary face of a box mesh. `normal` need not be normalized; in 2-D the
/// plane is a line.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, origin, normal, tol=None))]
pub fn points_on_plane(
    mesh: PyRef<PyMesh>,
    origin: Vec<f64>,
    normal: Vec<f64>,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_on_plane(&mesh.inner, &origin, &normal, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **below** the plane through `origin` with normal `normal` — the
/// half-space the normal points away from, plane included.
///
/// There is no `points_above_plane`: flip the normal and this is the other
/// half-space.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, origin, normal, tol=None))]
pub fn points_below_plane(
    mesh: PyRef<PyMesh>,
    origin: Vec<f64>,
    normal: Vec<f64>,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_below_plane(&mesh.inner, &origin, &normal, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the infinite line through `a` and `b`
/// (Cast3m `POIN … DROIT`).
///
/// Keeps the nodes at a distance `≤ tol` from the line, with no bound along
/// it — use `points_in_cylinder` with a small radius for a selection clipped
/// to the `a → b` segment.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, a, b, tol=None))]
pub fn points_on_line(
    mesh: PyRef<PyMesh>,
    a: Vec<f64>,
    b: Vec<f64>,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_on_line(&mesh.inner, &a, &b, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **inside** the finite cylinder of axis `base → top` and radius
/// `radius`.
///
/// The cylinder is capped: a node is kept when it is within `radius + tol` of
/// the axis **and** its axial coordinate falls between the two end sections.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, base, top, radius, tol=None))]
pub fn points_in_cylinder(
    mesh: PyRef<PyMesh>,
    base: Vec<f64>,
    top: Vec<f64>,
    radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_in_cylinder(&mesh.inner, &base, &top, radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the lateral surface of the finite cylinder of axis
/// `base → top` and radius `radius` (Cast3m `POIN … CYLI`).
///
/// The end discs are **not** part of the selection — they are flat faces, and
/// `points_on_plane` cuts those. This is how you grab the bore of a tube.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, base, top, radius, tol=None))]
pub fn points_on_cylinder(
    mesh: PyRef<PyMesh>,
    base: Vec<f64>,
    top: Vec<f64>,
    radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_on_cylinder(&mesh.inner, &base, &top, radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **inside** the cone of axis `base → top`, radius `base_radius` at
/// `base` and `top_radius` at `top`.
///
/// The shape is a truncated cone: `top_radius=0` (the default) gives a true
/// cone whose apex is `top`, `top_radius=base_radius` a cylinder. Capped like
/// `points_in_cylinder`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, base, top, base_radius, top_radius=0.0, tol=None))]
pub fn points_in_cone(
    mesh: PyRef<PyMesh>,
    base: Vec<f64>,
    top: Vec<f64>,
    base_radius: f64,
    top_radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result =
        crate::ops::mesh::points_in_cone(&mesh.inner, &base, &top, base_radius, top_radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the lateral surface of the cone of axis `base → top`, radius
/// `base_radius` at `base` and `top_radius` at `top` (Cast3m `POIN … CONE`).
///
/// The distance to the slanted surface is the perpendicular one, so the band
/// stays `tol` wide however steep the cone is. As for `points_on_cylinder`,
/// the end discs are not part of the selection.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, base, top, base_radius, top_radius=0.0, tol=None))]
pub fn points_on_cone(
    mesh: PyRef<PyMesh>,
    base: Vec<f64>,
    top: Vec<f64>,
    base_radius: f64,
    top_radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result =
        crate::ops::mesh::points_on_cone(&mesh.inner, &base, &top, base_radius, top_radius, tol)?;
    Ok(PyMesh { inner: result })
}

/// Nodes **inside** the torus of centre `center`, axis `axis`, major radius
/// `major_radius` and minor radius `minor_radius`.
///
/// The section is circular: the torus is the set of points at a distance
/// `≤ minor_radius` from the circle of radius `major_radius` drawn around
/// `center` in the plane normal to `axis`. **3-D only.**
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, center, axis, major_radius, minor_radius, tol=None))]
pub fn points_in_torus(
    mesh: PyRef<PyMesh>,
    center: Vec<f64>,
    axis: Vec<f64>,
    major_radius: f64,
    minor_radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_in_torus(
        &mesh.inner,
        &center,
        &axis,
        major_radius,
        minor_radius,
        tol,
    )?;
    Ok(PyMesh { inner: result })
}

/// Nodes **on** the torus of centre `center`, axis `axis`, major radius
/// `major_radius` and minor radius `minor_radius`.
///
/// Keeps the nodes within `tol` of the tube's surface. The torus being
/// closed, there is no cap to worry about here. **3-D only.**
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, center, axis, major_radius, minor_radius, tol=None))]
pub fn points_on_torus(
    mesh: PyRef<PyMesh>,
    center: Vec<f64>,
    axis: Vec<f64>,
    major_radius: f64,
    minor_radius: f64,
    tol: Option<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::points_on_torus(
        &mesh.inner,
        &center,
        &axis,
        major_radius,
        minor_radius,
        tol,
    )?;
    Ok(PyMesh { inner: result })
}

/// Fuse submeshes of the same element type into one, dropping duplicate cells
/// (identical node sequences).
///
/// Types appear in their first-seen order; the face colour of the first
/// submesh of each type is kept. `mesh` itself is left untouched.
///
/// Errors if `mesh` has no submesh (no `Coords` to attach to).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_mesh")]
pub fn consolidate(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    Ok(PyMesh {
        inner: crate::ops::mesh::consolidate(&mesh.inner)?,
    })
}

/// Weld together nodes closer than `tol`, redirecting the connectivity to one
/// representative per cluster.
///
/// Returns a new mesh mirroring `mesh` (same submeshes, types and colours)
/// with welded-away nodes redirected to their cluster representative — the
/// smallest-id node of the cluster, which keeps its own coordinates (no
/// averaging). Cells that collapse onto a repeated node (a degenerate segment,
/// triangle, …) are dropped. `tol` must be ≥ 0; `tol = 0` welds only exactly
/// coincident nodes. `mesh` itself is left untouched.
///
/// With `in_place=True` the connectivity of `mesh`'s **own** submeshes is
/// rewritten instead — the assumed side effect — and the very same mesh object
/// is returned. Since the union `mesh_a | mesh_b` shares its submeshes rather
/// than copying them, welding that union in place welds `mesh_a` and `mesh_b`
/// themselves: afterwards they really do share their interface nodes. The mesh
/// structure is preserved (same submeshes, same cells in the same order), so a
/// cell that *would* collapse is an error here instead of being dropped, as is
/// a submesh already sealed by a finite-element space, field or matrix. Both
/// are checked before anything is written: a rejected call changes nothing.
///
/// Every call prints a one-line tally on stdout once the weld is done — nodes
/// welded, cells dropped, tolerance used.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, tol, in_place=false))]
pub fn merge_nodes(
    py: Python<'_>,
    mesh: Py<PyMesh>,
    tol: f64,
    in_place: bool,
) -> PyResult<Py<PyMesh>> {
    let result = crate::ops::mesh::merge_nodes(&mesh.borrow(py).inner, tol, in_place)?;
    if in_place {
        // The operator welded `mesh`'s own submeshes and handed the same mesh
        // back; return the caller's object itself, so `out is mesh` holds.
        return Ok(mesh);
    }
    Py::new(py, PyMesh { inner: result })
}

/// Extract the boundary of a surface mesh (TRI3/QUA4) as SEG2 loops.
///
/// An element edge used by exactly one cell is a boundary edge; the boundary
/// edges (pooled across all surface submeshes) are chained into closed loops.
/// Returns a Mesh with one SEG2 submesh per loop — a single loop for a
/// simply-connected domain, several when the domain has holes or disjoint
/// pieces. Loops keep the CCW boundary orientation (outer loop CCW, holes
/// CW), so the result can feed straight back into `triangulate_surface`.
///
/// With `angle_deg` given, each loop is further split into open **arêtes** at
/// its corner nodes — where the boundary turns by more than `angle_deg`
/// degrees — one SEG2 submesh per arête (as `skin` splits a volume's skin into
/// flat faces). A loop with no such corner stays a single closed loop.
/// `angle_deg=None` (the default) keeps every boundary as one closed loop.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle_deg=None))]
pub fn border(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::border(&mesh.inner, angle_deg)?;
    Ok(PyMesh { inner: result })
}

/// Extract the boundary surface (skin) of a volume mesh, split by flat face.
///
/// Works on every volume type — TET4, PYRA5, PENTA6, HEX8 and their quadratic
/// counterparts TET10, PENTA15, HEX20, HEX27.
///
/// A volume-element facet (TET4 → 4 triangles, HEX8 → 6 quads, PENTA6 → 2
/// triangles + 3 quads, PYRA5 → 1 quad + 4 triangles) used by exactly one cell
/// lies on the boundary; sharing is decided on the facet's corners, so cells of
/// different degrees still cancel. The boundary facets (pooled across all
/// volume submeshes) are grouped into flat faces by flooding across shared
/// edges as long as neighbouring facets stay coplanar (their normals differ by
/// at most `angle_deg`, default 1°).
///
/// **A facet is emitted in its own type**: a HEX8 yields QUA4, a TET10 yields
/// TRI6, a HEX27 yields QUA9 — so the skin of a quadratic mesh is quadratic
/// and keeps its mid-side nodes. Returns a Mesh with one submesh per flat face
/// and per facet type — e.g. 6 submeshes for a cube, 5 for a prism (2 caps +
/// 3 sides), 5 for a pyramid (base + 4 triangles). Facets keep their outward
/// orientation; the original nodes are reused.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle_deg=None))]
pub fn skin(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::skin(&mesh.inner, angle_deg)?;
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
    let result = crate::ops::mesh::orient(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Re-order the cells of a line mesh into a continuous chain.
///
/// The complement of `orient`: where `orient` fixes the *direction* of the
/// cells, `chain` fixes their *order* — each SEG2/SEG3 submesh is sorted so
/// that consecutive cells share a node (and flipped where needed), so reading
/// the connectivity walks the curve from one end to the other. Each submesh is
/// chained on its own and must be one continuous chain, open or closed: a node
/// carrying three segments (branching) or disjoint pieces raise an error.
/// Returns a fresh mesh sharing the input's nodes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn chain(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::chain(&mesh.inner)?;
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
    let result = crate::ops::mesh::invert(&mesh.inner)?;
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
    let mesh = crate::ops::mesh::line(a.as_node(), b.as_node(), n_elems, et)?;
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
    let mesh = crate::ops::mesh::circle(center.as_node(), &normal, radius, n_elems, et)?;
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
    let mesh = crate::ops::mesh::arc(a.as_node(), center.as_node(), b.as_node(), n_elems, et)?;
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
    let mesh = crate::ops::mesh::sweep(&mesh_a.inner, &mesh_b.inner, n_layers, et)?;
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
    let mesh =
        crate::ops::mesh::transfinite(&side1.inner, &side2.inner, &side3.inner, &side4.inner, et)?;
    Ok(PyMesh { inner: mesh })
}

/// Extrude `mesh` by `n_layers` layers along `direction` (the total
/// displacement vector). SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn extrude(mesh: PyRef<PyMesh>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::extrude(&mesh.inner, &direction, n_layers)?;
    Ok(PyMesh { inner: result })
}

/// Revolve `mesh` by `n_layers` layers over a total `angle` (radians) — the
/// rotational companion of `extrude`. SEG2 → QUA4, TRI3 → PENTA6,
/// QUA4 → HEX8.
///
/// In 2-D the revolution is about the point `center` (counterclockwise for a
/// positive `angle`) and `axis` is ignored; in 3-D it is about the line
/// through `center` directed by `axis` (right-handed), which is then
/// required. `|angle|` may not exceed a full turn, and a full turn closes the
/// ring: the last node layer is the first one again, so there is no seam. No
/// node may lie on the axis — it would collapse the cells touching it.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle, n_layers, center, axis=None))]
pub fn revolve(
    mesh: PyRef<PyMesh>,
    angle: f64,
    n_layers: usize,
    center: Vec<f64>,
    axis: Option<Vec<f64>>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::revolve(&mesh.inner, angle, n_layers, &center, axis.as_deref())?;
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
    let mesh = crate::ops::mesh::sweep_solid(&mesh_a.inner, &mesh_b.inner, n_layers)?;
    Ok(PyMesh { inner: mesh })
}

/// Build the **quadratic** (Lagrange-2) copy of a linear mesh: each element
/// type is bumped to its quadratic sibling (TRI3→TRI6, HEX8→HEX20, …). Corner
/// nodes are re-used; one mid-edge node is created per edge (at the midpoint)
/// and shared between the cells that use it. The original mesh is untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_quadratic(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::to_quadratic(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Convert every submesh of `mesh` to `element_type`, splitting each cell into
/// cells of the target type **without moving or adding any node** on the
/// existing corners. Supported: identity (already the target type — copied
/// verbatim), `"QUA4"` → `"TRI3"` (two triangles per quad, `(0,2)` diagonal),
/// and `"HEX8"` → `"TET4"` (six tetrahedra per hex, a conforming space-filling
/// split). Corner nodes are re-used; face colours are preserved. To promote to
/// a quadratic type (`TRI3`→`TRI6`, …), which creates mid-edge nodes, use
/// `to_quadratic`. The original mesh is untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn convert(mesh: PyRef<PyMesh>, element_type: &str) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let result = crate::ops::mesh::convert(&mesh.inner, et)?;
    Ok(PyMesh { inner: result })
}

/// Translate `mesh` by `vector`, returning a fresh copy with its own nodes
/// (the original is left untouched). `vector` matches the mesh dimension.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn translate(mesh: PyRef<PyMesh>, vector: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::translate(&mesh.inner, &vector)?;
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
    let result = crate::ops::mesh::rotate(&mesh.inner, angle, &center, axis.as_deref())?;
    Ok(PyMesh { inner: result })
}

/// Mirror `mesh` through the point `center` (Cast3m `SYME … POINT`),
/// returning a fresh copy with its own nodes (the original is left
/// untouched). Every node goes to `2·center − x`.
///
/// In 3-D the map reverses orientation, so the cells are re-ordered (as
/// `invert` does) to keep the copy's Jacobians positive; in 2-D it is a plain
/// half-turn and nothing is re-ordered.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn symmetry_point(mesh: PyRef<PyMesh>, center: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::symmetry_point(&mesh.inner, &center)?;
    Ok(PyMesh { inner: result })
}

/// Mirror `mesh` through the infinite line running through `a` and `b`
/// (Cast3m `SYME … DROIT`), returning a fresh copy with its own nodes (the
/// original is left untouched).
///
/// In 2-D this is the mirror image about the line; in 3-D it is the half-turn
/// about it (for the mirror image through a plane, use `symmetry_plane`).
/// Orientation-reversing in 2-D only, where the cells are re-ordered (as
/// `invert` does) to keep the copy's Jacobians positive.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn symmetry_line(mesh: PyRef<PyMesh>, a: Vec<f64>, b: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::symmetry_line(&mesh.inner, &a, &b)?;
    Ok(PyMesh { inner: result })
}

/// Mirror `mesh` through the plane running through the three points `a`, `b`
/// and `c` (Cast3m `SYME … PLAN`), returning a fresh copy with its own nodes
/// (the original is left untouched). Only the plane the three points span
/// matters, not their order; they must not be aligned.
///
/// 3-D only — in 2-D the mirror about a line is `symmetry_line`. Always
/// orientation-reversing, so the cells are re-ordered (as `invert` does) to
/// keep the copy's Jacobians positive.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn symmetry_plane(
    mesh: PyRef<PyMesh>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::symmetry_plane(&mesh.inner, &a, &b, &c)?;
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
    let mesh = crate::ops::mesh::triangulate_surface_cancellable(
        &contour.inner,
        et,
        size,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Pave the inside of a closed contour with quadrangles, in rows walking
/// inward from the boundary.
///
/// The quadrangle-oriented companion of `triangulate_surface`, and the one to
/// reach for when the mesh is going to be computed on: paving lays QUA4 cells
/// down directly, in rows that follow the contour, instead of triangulating
/// and pairing triangles up afterwards.
///
/// `contour` holds **one or more** closed SEG2 loops, oriented by the caller
/// exactly as for `triangulate_surface`: a **counter-clockwise** loop is a
/// domain's outer boundary, a **clockwise** loop is a hole. Several disjoint
/// CCW loops pave several independent domains at once. `size` sets the target
/// element edge length; `None` uses the mean boundary edge length per domain.
/// `element_type` is "QUA4", "QUA8" or "QUA9". The contour may be 2-D or a
/// planar loop in 3-D (paved in its best-fit plane, then lifted back).
///
/// The contour is untouchable: its nodes come back at their own positions and
/// no node is ever added on a boundary edge. A contour the paver cannot work
/// with is therefore reported rather than worked around — `all_quad=True` on a
/// loop with an odd number of segments raises, since a polygon with an odd
/// number of sides has no filling by quadrangles alone and evening the count
/// out would mean adding a boundary node; so does a contour so coarse for the
/// requested size that the front folds onto itself.
///
/// With `all_quad=False` (the default) an odd loop simply costs one triangle,
/// returned in a separate TRI3 submesh, along with the few cells a distorted
/// leftover polygon could not make square.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, all_quad=false))]
pub fn pave_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: &str,
    size: Option<f64>,
    all_quad: bool,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    // Poll Python signals while paving so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesh::pave_surface_cancellable(
        &contour.inner,
        et,
        size,
        all_quad,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Move the interior nodes of a surface mesh to improve its cells, leaving the
/// connectivity and the boundary exactly as they are.
///
/// `sweeps` is how many passes to run. `angular=True` (the default) uses
/// angle-based smoothing, which aims at the right angles a quadrangle wants;
/// `False` uses the plain Laplacian, which aims at the one-ring's barycentre
/// and knows nothing about angles. `in_place=True` writes the new positions
/// onto your own nodes and hands the same mesh back; otherwise the moved nodes
/// are duplicated and a fresh mesh comes out, the boundary's nodes being
/// shared since they never moved.
///
/// Two guarantees hold whatever the rule: **no node on the boundary ever
/// moves**, and a position is taken only when every incident cell stays valid
/// and the worst incident quality does not get worse.
///
/// Smoothing cannot change who is next to whom, so it cannot fix a node with
/// the wrong number of cells around it — the angles around a node sum to 2π
/// whatever the positions. That is `cleanup`'s job, and running it first is
/// usually what unlocks the smoothing.
///
/// TRI3 and QUA4 only, in 2-D. POI1 and SEG2 submeshes are ignored.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, sweeps=20, angular=true, in_place=false))]
pub fn regularize(
    mesh: PyRef<PyMesh>,
    sweeps: usize,
    angular: bool,
    in_place: bool,
) -> PyResult<PyMesh> {
    let out = crate::ops::mesh::regularize(&mesh.inner, sweeps, angular, in_place)?;
    Ok(PyMesh { inner: out })
}

/// Fix the connectivity of a surface mesh: remove its doublets and switch the
/// diagonals that lower the valence error. No node moves.
///
/// A **doublet** is an interior node with only two quadrangles around it,
/// which therefore share two edges; the node sits in a wedge no smoothing can
/// open. A node of the **wrong valence** wants four cells and has three or
/// five, giving corners of 120° or 72° on average — and the angles around a
/// node sum to 2π whatever the positions, so smoothing will never square them.
///
/// The only move is the diagonal switch: two quadrangles sharing an edge form
/// a hexagon, which splits across any of its three diagonals. It changes no
/// node and no boundary, and is applied only when it strictly lowers the
/// valence error and leaves both cells convex.
///
/// Triangles are read for incidence and never touched — that is
/// `merge_triangles`'s job.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn cleanup(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let out = crate::ops::mesh::cleanup(&mesh.inner)?;
    Ok(PyMesh { inner: out })
}

/// Remove the triangles from a quadrangle-dominant mesh, in pairs.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge_triangles(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let out = crate::ops::mesh::merge_triangles(&mesh.inner)?;
    Ok(PyMesh { inner: out })
}

/// Mesh the inside of a closed contour with a structured grid core and a
/// frontal band — the regular-mesh companion of `pave_surface`.
///
/// Same input, same output, same promise that the contour is untouchable: only
/// the interior is obtained differently. `pave_surface` walks a front inward
/// until two of its rows meet, and that meeting line carries the valence
/// defects, the leftover triangles and the flattest cells — even on a plain
/// rectangle, which comes out as an onion with four diagonal seams.
/// `grid_surface` lays a tensor grid instead, and leaves the front only what
/// the grid could not reach.
///
/// The grid's lines are taken from the contour: every axis-aligned edge long
/// enough to be a feature pins a line, and the gaps are subdivided at about
/// `size`. A grid node landing on a contour node **is** that node, so the core
/// reaches the boundary rather than stopping short of it. On a rectilinear
/// domain laid out for the grid there is no band at all and the mesh is the
/// grid: every cell a rectangle, every Jacobian 1, no triangle.
///
/// That last part is the one thing asked of the caller. A grid can only meet a
/// contour whose nodes fall on grid lines, so **break every side at the
/// shape's own corners and let each piece take a whole number of cells**. A
/// contour that does not is not an error — it just gets more band and less
/// grid, down to the quality of `pave_surface` in the worst case.
///
/// `band` is extra clearance in cells between the core and the contour. Zero
/// is the useful value; raise it only for a contour the grid cannot meet, such
/// as a curve, where giving the front a couple of cells to work in beats
/// letting it fight for a sliver.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, band=0, all_quad=false))]
pub fn grid_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: &str,
    size: Option<f64>,
    band: usize,
    all_quad: bool,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesh::grid_surface_cancellable(
        &contour.inner,
        et,
        size,
        band,
        all_quad,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Mesh the inside of a closed contour with a structured grid core and a
/// frontal band, taking the grid's lines **one per contour node**.
///
/// The second of the two grid meshers, and the sibling of `grid_surface`, which
/// it does not replace: same input, same output, same untouchable contour, and
/// neither wins everywhere.
///
/// `grid_surface` pins a line on the coordinate each aligned side lies on, and
/// subdivides between two lines by whichever side spans them end to end. Every
/// line is straight. `grid_surface2` gives every node of the contour the line
/// that crosses it, collapses the bands too thin to be cells — welding each of
/// their edges onto the contour node at one end, or onto its midpoint — and
/// lets a grid node within a quarter cell of a contour node move onto it.
///
/// A row is therefore a polyline rather than a line, and that is the point: one
/// row can meet two facing walls at two different heights, so a wall cut into
/// ten can face a wall cut into eleven. `grid_surface` has to pick one of them
/// and sends the other to the band.
///
/// **Reach for `grid_surface2` on a rectilinear shape**, all the more so when
/// its sides were not cut at the corners facing them; **reach for
/// `grid_surface` on anything curved**, where the contour's nodes only tell you
/// where its vertices happened to fall. Measured worst cell, `grid_surface`
/// then `grid_surface2`: plate with a step off the grid 0.405 / 0.963, L with
/// arbitrary dimensions 0.437 / 0.979, crenellated profile with its base in one
/// run 0.287 / 0.651, house with a pitched roof 0.304 / 0.475. On a **circle**
/// the order reverses hard — 0.288 against 0.005 — and `grid_surface2` should
/// not be used: nothing dictates a grid line over most of a curve. The book's
/// *Mailler une géométrie* page puts all four surface meshers side by side.
///
/// `size` and `band` mean what they mean for `grid_surface`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, band=0, all_quad=false))]
pub fn grid_surface2(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: &str,
    size: Option<f64>,
    band: usize,
    all_quad: bool,
) -> PyResult<PyMesh> {
    let et = ElementType::from_name(element_type)
        .ok_or_else(|| PyValueError::new_err(format!("unknown element type: {element_type}")))?;
    let mesh = crate::ops::mesh::grid_surface2_cancellable(
        &contour.inner,
        et,
        size,
        band,
        all_quad,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Mesh the inside of a closed envelope with a hexahedral boundary layer over
/// a tetrahedral core — the 3-D companion of `pave_surface`.
///
/// Puts hexahedra where they matter, in the layer against the boundary where
/// gradients are steepest and an element's shape decides the accuracy, and
/// leaves the smooth interior to tetrahedra.
///
/// `layers` boundary layers are grown inward, each `thickness` deep;
/// `thickness=None` takes the envelope's mean edge length, which gives roughly
/// cube-shaped cells. `size` is the target element size for the tetrahedral
/// core. The envelope is a closed surface of QUA4 and/or TRI3 facets whose
/// normals point **out of the material**, exactly as for `triangulate_volume`;
/// its nodes are reused as they are.
///
/// The result carries a QUA4-born HEX8 submesh, a TRI3-born PENTA6 one, a
/// PYRA5 one and a TET4 one, each present only if non-empty. The pyramids are
/// the junction: the layer's inner faces are squares and a tetrahedron has
/// none, so without them the mesh could not be conforming.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (envelope, layers=1, thickness=None, size=None))]
pub fn pave_volume(
    py: Python<'_>,
    envelope: PyRef<PyMesh>,
    layers: usize,
    thickness: Option<f64>,
    size: Option<f64>,
) -> PyResult<PyMesh> {
    // Poll Python signals while meshing so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesh::pave_volume_cancellable(
        &envelope.inner,
        layers,
        thickness,
        size,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Fill the inside of a closed `TRI3` `envelope` with `TET4` cells — the 3-D
/// companion of `triangulate_surface`.
///
/// The envelope's normals must point **out of the material**; a concave
/// shape is fine, and an internal cavity is simply another closed surface
/// whose normals point into the hole. Its nodes are reused as they are, and
/// nodes are added inside the solid so the cells come out well shaped.
///
/// `size` is the target edge length; `None` takes the mean edge length of
/// the envelope. `allow_surface_nodes` lets the mesher cut the envelope
/// finer where it cannot otherwise fit it or make it usable: the shape is
/// kept — every added node lies on the edge or facet it divides — but the
/// skin of the result no longer matches the surface handed in, and a warning
/// on stderr says how many were added. Without it, such a surface is
/// refused rather than meshed badly.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (envelope, size=None, allow_surface_nodes=false))]
pub fn triangulate_volume(
    py: Python<'_>,
    envelope: PyRef<PyMesh>,
    size: Option<f64>,
    allow_surface_nodes: bool,
) -> PyResult<PyMesh> {
    // Poll Python signals while meshing so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesh::triangulate_volume_cancellable(
        &envelope.inner,
        size,
        allow_surface_nodes,
        &PySignals(py),
    )?;
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
/// Every element type pyrucast knows is read: POI1, SEG2, TRI3, QUA4, TET4,
/// PYRA5, PENTA6, HEX8 and the quadratic SEG3, TRI6, QUA8, QUA9, TET10,
/// PENTA15, HEX20, HEX27. Any other gmsh type raises, naming the codes it
/// does accept.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn read_gmsh<'py>(
    py: Python<'py>,
    coords: PyRef<PyCoords>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let groups = crate::ops::mesh::read_gmsh(coords.handle.clone(), std::path::Path::new(path))?;
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
    let groups = crate::ops::mesh::read_gmsh_str(coords.handle.clone(), text)?;
    groups_to_dict(py, groups)
}

/// Select the part of a field's support passing a value band, zone by zone —
/// a value-range filter returning a `Mesh`.
///
/// `field` may be a `NodeField` / `SubNodeField` (→ POI1 submeshes of the
/// passing **nodes**) or an `ElementField` / `SubElementField` (→ submeshes
/// of the passing **cells**, each of its zone's element type; a cell passes
/// only when *all* its Gauss points do). The result has one submesh per
/// processed zone.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a point/cell is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component of each zone. A `components`
/// list tests **only** those components, and only on the zones carrying
/// **all** of them — a zone missing any listed component is skipped (no
/// submesh). Errors if no bound is given, or the lower one exceeds the upper.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
#[allow(clippy::too_many_arguments)]
pub fn select(
    field: &Bound<'_, PyAny>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    let inner = if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        crate::ops::mesh::select_nodes(&f.inner, &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        crate::ops::mesh::select_cells(&f.inner, &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        crate::ops::mesh::select_sub_nodes(&f.handle.read(), &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        crate::ops::mesh::select_sub_cells(&f.handle.read(), &band, components)?
    } else {
        return Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ));
    };
    Ok(PyMesh { inner })
}

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs ci-dessus (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Aucune logique : chaque méthode rappelle la
// fonction libre, receveur compris.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMesh {
    /// Voir `pyrucast.mesh.to_poi1`.
    fn to_poi1(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::to_poi1(slf)
    }

    /// Voir `pyrucast.mesh.barycenter`.
    fn barycenter(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::barycenter(slf)
    }

    /// Voir `pyrucast.mesh.elements_on`.
    #[pyo3(signature = (points, strict=true))]
    fn elements_on(slf: PyRef<'_, Self>, points: PyRef<PyMesh>, strict: bool) -> PyResult<PyMesh> {
        super::mesh::elements_on(slf, points, strict)
    }

    /// Voir `pyrucast.mesh.points_in_sphere`.
    #[pyo3(signature = (center, radius, tol=None))]
    fn points_in_sphere(
        slf: PyRef<'_, Self>,
        center: Vec<f64>,
        radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_in_sphere(slf, center, radius, tol)
    }

    /// Voir `pyrucast.mesh.points_on_sphere`.
    #[pyo3(signature = (center, radius, tol=None))]
    fn points_on_sphere(
        slf: PyRef<'_, Self>,
        center: Vec<f64>,
        radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_sphere(slf, center, radius, tol)
    }

    /// Voir `pyrucast.mesh.points_on_plane`.
    #[pyo3(signature = (origin, normal, tol=None))]
    fn points_on_plane(
        slf: PyRef<'_, Self>,
        origin: Vec<f64>,
        normal: Vec<f64>,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_plane(slf, origin, normal, tol)
    }

    /// Voir `pyrucast.mesh.points_below_plane`.
    #[pyo3(signature = (origin, normal, tol=None))]
    fn points_below_plane(
        slf: PyRef<'_, Self>,
        origin: Vec<f64>,
        normal: Vec<f64>,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_below_plane(slf, origin, normal, tol)
    }

    /// Voir `pyrucast.mesh.points_on_line`.
    #[pyo3(signature = (a, b, tol=None))]
    fn points_on_line(
        slf: PyRef<'_, Self>,
        a: Vec<f64>,
        b: Vec<f64>,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_line(slf, a, b, tol)
    }

    /// Voir `pyrucast.mesh.points_in_cylinder`.
    #[pyo3(signature = (base, top, radius, tol=None))]
    fn points_in_cylinder(
        slf: PyRef<'_, Self>,
        base: Vec<f64>,
        top: Vec<f64>,
        radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_in_cylinder(slf, base, top, radius, tol)
    }

    /// Voir `pyrucast.mesh.points_on_cylinder`.
    #[pyo3(signature = (base, top, radius, tol=None))]
    fn points_on_cylinder(
        slf: PyRef<'_, Self>,
        base: Vec<f64>,
        top: Vec<f64>,
        radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_cylinder(slf, base, top, radius, tol)
    }

    /// Voir `pyrucast.mesh.points_in_cone`.
    #[pyo3(signature = (base, top, base_radius, top_radius=0.0, tol=None))]
    fn points_in_cone(
        slf: PyRef<'_, Self>,
        base: Vec<f64>,
        top: Vec<f64>,
        base_radius: f64,
        top_radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_in_cone(slf, base, top, base_radius, top_radius, tol)
    }

    /// Voir `pyrucast.mesh.points_on_cone`.
    #[pyo3(signature = (base, top, base_radius, top_radius=0.0, tol=None))]
    fn points_on_cone(
        slf: PyRef<'_, Self>,
        base: Vec<f64>,
        top: Vec<f64>,
        base_radius: f64,
        top_radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_cone(slf, base, top, base_radius, top_radius, tol)
    }

    /// Voir `pyrucast.mesh.points_in_torus`.
    #[pyo3(signature = (center, axis, major_radius, minor_radius, tol=None))]
    fn points_in_torus(
        slf: PyRef<'_, Self>,
        center: Vec<f64>,
        axis: Vec<f64>,
        major_radius: f64,
        minor_radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_in_torus(slf, center, axis, major_radius, minor_radius, tol)
    }

    /// Voir `pyrucast.mesh.points_on_torus`.
    #[pyo3(signature = (center, axis, major_radius, minor_radius, tol=None))]
    fn points_on_torus(
        slf: PyRef<'_, Self>,
        center: Vec<f64>,
        axis: Vec<f64>,
        major_radius: f64,
        minor_radius: f64,
        tol: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::points_on_torus(slf, center, axis, major_radius, minor_radius, tol)
    }

    /// Voir `pyrucast.mesh.consolidate`.
    fn consolidate(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::consolidate(slf)
    }

    /// Voir `pyrucast.mesh.merge_nodes`.
    #[pyo3(signature = (tol, in_place=false))]
    fn merge_nodes(
        slf: Py<Self>,
        py: Python<'_>,
        tol: f64,
        in_place: bool,
    ) -> PyResult<Py<PyMesh>> {
        super::mesh::merge_nodes(py, slf, tol, in_place)
    }

    /// Voir `pyrucast.mesh.border`.
    #[pyo3(signature = (angle_deg=None))]
    fn border(slf: PyRef<'_, Self>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
        super::mesh::border(slf, angle_deg)
    }

    /// Voir `pyrucast.mesh.skin`.
    #[pyo3(signature = (angle_deg=None))]
    fn skin(slf: PyRef<'_, Self>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
        super::mesh::skin(slf, angle_deg)
    }

    /// Voir `pyrucast.mesh.orient`.
    fn orient(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::orient(slf)
    }

    /// Voir `pyrucast.mesh.invert`.
    fn invert(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::invert(slf)
    }

    /// Voir `pyrucast.mesh.chain`.
    fn chain(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::chain(slf)
    }

    /// Voir `pyrucast.mesh.sweep`.
    #[pyo3(signature = (mesh_b, n_layers, element_type="QUA4"))]
    fn sweep(
        slf: PyRef<'_, Self>,
        mesh_b: PyRef<PyMesh>,
        n_layers: usize,
        element_type: &str,
    ) -> PyResult<PyMesh> {
        super::mesh::sweep(slf, mesh_b, n_layers, element_type)
    }

    /// Voir `pyrucast.mesh.transfinite`.
    #[pyo3(signature = (side2, side3, side4, element_type="QUA4"))]
    fn transfinite(
        slf: PyRef<'_, Self>,
        side2: PyRef<PyMesh>,
        side3: PyRef<PyMesh>,
        side4: PyRef<PyMesh>,
        element_type: &str,
    ) -> PyResult<PyMesh> {
        super::mesh::transfinite(slf, side2, side3, side4, element_type)
    }

    /// Voir `pyrucast.mesh.extrude`.
    fn extrude(slf: PyRef<'_, Self>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
        super::mesh::extrude(slf, direction, n_layers)
    }

    /// Voir `pyrucast.mesh.revolve`.
    #[pyo3(signature = (angle, n_layers, center, axis=None))]
    fn revolve(
        slf: PyRef<'_, Self>,
        angle: f64,
        n_layers: usize,
        center: Vec<f64>,
        axis: Option<Vec<f64>>,
    ) -> PyResult<PyMesh> {
        super::mesh::revolve(slf, angle, n_layers, center, axis)
    }

    /// Voir `pyrucast.mesh.sweep_solid`.
    fn sweep_solid(
        slf: PyRef<'_, Self>,
        mesh_b: PyRef<PyMesh>,
        n_layers: usize,
    ) -> PyResult<PyMesh> {
        super::mesh::sweep_solid(slf, mesh_b, n_layers)
    }

    /// Voir `pyrucast.mesh.to_quadratic`.
    fn to_quadratic(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::to_quadratic(slf)
    }

    /// Voir `pyrucast.mesh.convert`.
    fn convert(slf: PyRef<'_, Self>, element_type: &str) -> PyResult<PyMesh> {
        super::mesh::convert(slf, element_type)
    }

    /// Voir `pyrucast.mesh.translate`.
    fn translate(slf: PyRef<'_, Self>, vector: Vec<f64>) -> PyResult<PyMesh> {
        super::mesh::translate(slf, vector)
    }

    /// Voir `pyrucast.mesh.rotate`.
    #[pyo3(signature = (angle, center, axis=None))]
    fn rotate(
        slf: PyRef<'_, Self>,
        angle: f64,
        center: Vec<f64>,
        axis: Option<Vec<f64>>,
    ) -> PyResult<PyMesh> {
        super::mesh::rotate(slf, angle, center, axis)
    }

    /// Voir `pyrucast.mesh.symmetry_point`.
    fn symmetry_point(slf: PyRef<'_, Self>, center: Vec<f64>) -> PyResult<PyMesh> {
        super::mesh::symmetry_point(slf, center)
    }

    /// Voir `pyrucast.mesh.symmetry_line`.
    fn symmetry_line(slf: PyRef<'_, Self>, a: Vec<f64>, b: Vec<f64>) -> PyResult<PyMesh> {
        super::mesh::symmetry_line(slf, a, b)
    }

    /// Voir `pyrucast.mesh.symmetry_plane`.
    fn symmetry_plane(
        slf: PyRef<'_, Self>,
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::symmetry_plane(slf, a, b, c)
    }

    /// Voir `pyrucast.mesh.triangulate_surface`.
    #[pyo3(signature = (element_type, size=None))]
    fn triangulate_surface(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        element_type: &str,
        size: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::triangulate_surface(py, slf, element_type, size)
    }

    /// Voir `pyrucast.mesh.pave_surface`.
    #[pyo3(signature = (element_type, size=None, all_quad=false))]
    fn pave_surface(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        element_type: &str,
        size: Option<f64>,
        all_quad: bool,
    ) -> PyResult<PyMesh> {
        super::mesh::pave_surface(py, slf, element_type, size, all_quad)
    }

    /// Voir `pyrucast.mesh.regularize`.
    #[pyo3(signature = (sweeps=20, angular=true, in_place=false))]
    fn regularize(
        slf: PyRef<'_, Self>,
        sweeps: usize,
        angular: bool,
        in_place: bool,
    ) -> PyResult<PyMesh> {
        super::mesh::regularize(slf, sweeps, angular, in_place)
    }

    /// Voir `pyrucast.mesh.cleanup`.
    fn cleanup(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::cleanup(slf)
    }

    /// Voir `pyrucast.mesh.merge_triangles`.
    fn merge_triangles(slf: PyRef<'_, Self>) -> PyResult<PyMesh> {
        super::mesh::merge_triangles(slf)
    }

    /// Voir `pyrucast.mesh.grid_surface`.
    #[pyo3(signature = (element_type, size=None, band=0, all_quad=false))]
    fn grid_surface(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        element_type: &str,
        size: Option<f64>,
        band: usize,
        all_quad: bool,
    ) -> PyResult<PyMesh> {
        super::mesh::grid_surface(py, slf, element_type, size, band, all_quad)
    }

    /// Voir `pyrucast.mesh.grid_surface2`.
    #[pyo3(signature = (element_type, size=None, band=0, all_quad=false))]
    fn grid_surface2(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        element_type: &str,
        size: Option<f64>,
        band: usize,
        all_quad: bool,
    ) -> PyResult<PyMesh> {
        super::mesh::grid_surface2(py, slf, element_type, size, band, all_quad)
    }

    /// Voir `pyrucast.mesh.pave_volume`.
    #[pyo3(signature = (layers=1, thickness=None, size=None))]
    fn pave_volume(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        layers: usize,
        thickness: Option<f64>,
        size: Option<f64>,
    ) -> PyResult<PyMesh> {
        super::mesh::pave_volume(py, slf, layers, thickness, size)
    }

    /// Voir `pyrucast.mesh.triangulate_volume`.
    #[pyo3(signature = (size=None, allow_surface_nodes=false))]
    fn triangulate_volume(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        size: Option<f64>,
        allow_surface_nodes: bool,
    ) -> PyResult<PyMesh> {
        super::mesh::triangulate_volume(py, slf, size, allow_surface_nodes)
    }
}
