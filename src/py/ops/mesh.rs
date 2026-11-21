//! Python wrappers for the mesher operations in [`crate::ops::mesh`].
//!
//! Free functions that build or transform a [`PyMesh`]. Kept here —
//! mirroring `src/ops/mesher/` — rather than on the `Mesh` class, per the
//! `py/ops/` convention (operations live with operations).

use crate::atoms::ElementType;
use crate::containers::mesh::Mesh;
use crate::ops::mesh::{
    CellBlock, CellLayout, CellValues, FrontRelax, GaussRule, NodeOrder, NodeValues,
};
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
use pyrucast_macros::py_op;

/// A POI1 mesh holding every node of `coords`, referenced or not.
///
/// # Parameters
///
/// - `coords`: the coordinates whose nodes to take.
///
/// # Returns
///
/// A POI1 `Mesh` of one zone.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn from_live_nodes(coords: PyRef<PyCoords>) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesh::from_live_nodes(coords.handle.clone())?;
    Ok(PyMesh { inner: mesh })
}

/// A POI1 mesh with one point per node of `nodes`, in their order.
///
/// The `Coords` comes from the nodes themselves — every `Node` carries its own —,
/// so no `Coords` argument is needed.
///
/// # Parameters
///
/// - `nodes`: the nodes, all of one `Coords`.
///
/// # Returns
///
/// A POI1 `Mesh` of one zone.
///
/// # Raises
///
/// `RuntimeError` if `nodes` is empty.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn poi1_from_nodes(nodes: Vec<PyRef<PyNode>>) -> PyResult<PyMesh> {
    let ns: Vec<crate::atoms::Node> = nodes.iter().map(|n| n.as_node().clone()).collect();
    let mesh = crate::ops::mesh::poi1_from_nodes(&ns)?;
    Ok(PyMesh { inner: mesh })
}

/// The distinct nodes of `mesh`, as a POI1 mesh, submesh by submesh.
///
/// Each submesh gives its POI1 companion, `SubMesh.to_poi1()`: a POI1 submesh
/// that lists each node once comes back as itself, shared with `mesh`; any
/// other gives the POI1 submesh of its distinct nodes, built once and the same
/// at every call. Fields built on the result pair with fields built on `mesh`.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to take.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per submesh of `mesh`, holding its distinct nodes
/// in the order they first appear.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_poi1(mesh: PyRef<PyMesh>) -> PyMesh {
    PyMesh {
        inner: crate::ops::mesh::to_poi1(&mesh.inner),
    }
}

/// A POI1 mesh of one fresh node at the centroid of each cell of `mesh`.
///
/// A POI1 input is therefore copied to fresh nodes at the same places — handy to
/// create the support nodes of Lagrange multipliers from a set of constrained
/// points.
///
/// # Parameters
///
/// - `mesh`: the mesh whose cells to take.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per submesh of `mesh`, one fresh node per cell.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn barycenter(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::barycenter(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Keep the cells of `mesh` resting on the nodes of `points`.
///
/// Only the set of nodes `points` references matters, typically a POI1 mesh.
///
/// # Parameters
///
/// - `mesh`: the cells to filter.
/// - `points`: the nodes, on the same `Coords` as `mesh`.
/// - `strict`: `True` keeps a cell when **all** its nodes are in the set;
///   `False` when **at least one** is.
///
/// # Returns
///
/// A `Mesh` mirroring `mesh` submesh by submesh — same types, possibly empty
/// zones.
///
/// # Raises
///
/// `RuntimeError` if `mesh` and `points` are not attached to the same `Coords`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, points, strict=true))]
pub fn elements_on(mesh: PyRef<PyMesh>, points: PyRef<PyMesh>, strict: bool) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::elements_on(&mesh.inner, &points.inner, strict)?;
    Ok(PyMesh { inner: result })
}

/// The cells of `a` that `b` holds too.
///
/// Two cells are the same with the same type and the same nodes in the same
/// order: a cell numbered from another node, or turned over, is another cell.
/// Meshes on two different `Coords` share no cell.
///
/// # Parameters
///
/// - `a`: the cells to sort.
/// - `b`: the cells to look for in `a`.
///
/// # Returns
///
/// A `Mesh` mirroring `a` zone by zone — same types, same order of cells,
/// same face colours, possibly empty zones. A zone kept whole comes back as it
/// is, shared, so a field laid on it still combines with one laid on the
/// result.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn intersection(a: PyRef<PyMesh>, b: PyRef<PyMesh>) -> PyMesh {
    PyMesh {
        inner: crate::ops::mesh::intersection(&a.inner, &b.inner),
    }
}

/// The cells of `a` that `b` does not hold.
///
/// Two cells are the same with the same type and the same nodes in the same
/// order, as for `intersection`. With `intersection(a, b)`, it splits each zone of `a` in two; the symmetric
/// difference is `difference(a, b) | difference(b, a)`. Meshes on two
/// different `Coords` share no cell, so all of `a` is kept.
///
/// # Parameters
///
/// - `a`: the cells to sort.
/// - `b`: the cells to take out of `a`.
///
/// # Returns
///
/// A `Mesh` mirroring `a` zone by zone — same types, same order of cells,
/// same face colours, possibly empty zones. A zone kept whole comes back as it
/// is, shared.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn difference(a: PyRef<PyMesh>, b: PyRef<PyMesh>) -> PyMesh {
    PyMesh {
        inner: crate::ops::mesh::difference(&a.inner, &b.inner),
    }
}

// --- Node selection by geometric region (Cast3m `POIN … SPHE / CYLI / …`) ---
//
// One operator per (shape, side). They all return a POI1 mesh mirroring
// `mesh` submesh by submesh — one zone per input zone, possibly empty — and
// all take the same `tol`, the geometric precision of the test as a distance
// to the region's surface; `tol=None` means `1e-6 ×` the bounding-box
// diagonal. The odd one out is the *nearest* node, which returns a single
// node and lives on the class: `mesh.nearest_node(point)`.

/// Nodes inside the sphere of centre `center` and radius `radius`.
///
/// Keeps the nodes at a distance `≤ radius + tol` from `center`. In 2-D the
/// sphere is a disc.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `center`: the centre, with the dimension of the mesh.
/// - `radius`: the radius, `≥ 0`.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `radius` is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the sphere of centre `center` and radius `radius`.
///
/// Keeps the nodes whose distance to `center` is within `tol` of `radius`, on
/// either side. In 2-D this selects a circle.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `center`: the centre, with the dimension of the mesh.
/// - `radius`: the radius, `≥ 0`.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `radius` is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the plane through `origin` with normal `normal`.
///
/// Keeps the nodes within `tol` of the plane — the usual way to grab a boundary
/// face of a box mesh. In 2-D the plane is a line.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `origin`: a point of the plane.
/// - `normal`: the normal of the plane, not necessarily of unit length.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `normal` is the zero vector, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes below the plane through `origin`: the side `normal` points away from.
///
/// The half-space includes the plane. There is no `points_above_plane`: flip the
/// normal and this is the other half-space.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `origin`: a point of the plane.
/// - `normal`: the normal of the plane, pointing to the side left out.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `normal` is the zero vector, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the infinite line through `a` and `b`.
///
/// Keeps the nodes at a distance `≤ tol` from the line, with no bound along it —
/// `pyrucast.mesh.points_in_cylinder` with a small radius gives a selection
/// clipped to the segment `a → b`.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `a`, `b`: two distinct points of the line.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `a` and `b` coincide, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes inside the finite cylinder of axis `base → top` and radius `radius`.
///
/// The cylinder is capped: a node is kept when it is within `radius + tol` of the
/// axis **and** its axial coordinate falls between the two end sections.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `base`, `top`: the centres of the two end sections, distinct.
/// - `radius`: the radius, `≥ 0`.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `base` and `top` coincide, if `radius` is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the lateral surface of the cylinder of axis `base → top`.
///
/// The end discs are **not** part of the selection — they are flat faces, which
/// `pyrucast.mesh.points_on_plane` cuts. This is how you grab the bore of a tube.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `base`, `top`: the centres of the two end sections, distinct.
/// - `radius`: the radius, `≥ 0`.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `base` and `top` coincide, if `radius` is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes inside the truncated cone of axis `base → top`.
///
/// Its radius is `base_radius` at `base` and `top_radius` at `top`: the default
/// `top_radius=0` gives a true cone whose apex is `top`, `top_radius=base_radius`
/// a cylinder. Capped like `pyrucast.mesh.points_in_cylinder`.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `base`, `top`: the centres of the two end sections, distinct.
/// - `base_radius`: the radius at `base`, `≥ 0`.
/// - `top_radius`: the radius at `top`, `≥ 0`.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `base` and `top` coincide, if a radius is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the lateral surface of the truncated cone of axis `base → top`.
///
/// The distance to the slanted surface is the perpendicular one, so the band
/// stays `tol` wide however steep the cone is. As for
/// `pyrucast.mesh.points_on_cylinder`, the end discs are not part of the
/// selection.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `base`, `top`: the centres of the two end sections, distinct.
/// - `base_radius`: the radius at `base`, `≥ 0`.
/// - `top_radius`: the radius at `top`, `≥ 0`; `0` gives a true cone.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, if `base` and `top` coincide, if a radius is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes inside the torus of centre `center` and axis `axis`, in 3-D.
///
/// The section is circular: the torus is the set of points at a distance
/// `≤ minor_radius` from the circle of radius `major_radius` drawn around
/// `center` in the plane normal to `axis`.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `center`: the centre of the torus.
/// - `axis`: the axis of revolution.
/// - `major_radius`: the radius of the circle the tube follows.
/// - `minor_radius`: the radius of the tube.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if the mesh is not 3-D, if a point does not have the dimension of the mesh, if `axis` is the zero vector, if a radius is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Nodes on the torus of centre `center` and axis `axis`, in 3-D.
///
/// Keeps the nodes within `tol` of the tube's surface. The torus being closed,
/// there is no cap to worry about.
///
/// # Parameters
///
/// - `mesh`: the mesh whose nodes to test.
/// - `center`: the centre of the torus.
/// - `axis`: the axis of revolution.
/// - `major_radius`: the radius of the circle the tube follows.
/// - `minor_radius`: the radius of the tube.
/// - `tol`: the precision of the test, a distance to the surface of the region;
///   `None` takes `1e-6 ×` the diagonal of the mesh's bounding box.
///
/// # Returns
///
/// A POI1 `Mesh` with one zone per zone of `mesh`, possibly empty, holding the
/// nodes kept.
///
/// # Raises
///
/// `RuntimeError` if the mesh is not 3-D, if a point does not have the dimension of the mesh, if `axis` is the zero vector, if a radius is negative, or if `tol` is negative.
#[py_op(method_on = PyMesh)]
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

/// Fuse the submeshes of `mesh` of one element type, dropping duplicate cells.
///
/// A duplicate cell has the same node sequence as another. Types appear in the
/// order they are first seen. A type held by a single submesh with no duplicate
/// cell has nothing to merge: that submesh comes back as itself, shared with
/// `mesh` — mutating one mutates the other, and `duplicate()` makes an
/// independent copy. Any other type gives a new submesh, with the face colour
/// of the first submesh of that type.
///
/// # Parameters
///
/// - `mesh`: the mesh to consolidate, left untouched.
///
/// # Returns
///
/// A `Mesh`, one submesh per element type; empty for an empty `mesh`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_mesh")]
pub fn consolidate(mesh: PyRef<PyMesh>) -> PyMesh {
    PyMesh {
        inner: crate::ops::mesh::consolidate(&mesh.inner),
    }
}

/// Weld together the nodes of `mesh` closer than `tol`.
///
/// The connectivity is redirected to one representative per cluster — the
/// smallest-id node of the cluster, which keeps its own coordinates (no
/// averaging). A cluster is a connected component of the relation "closer than
/// `tol`": the weld propagates from node to node, so on a chain a—b—c where a and
/// c are further apart than `tol`, all three still end up as one — which is why a
/// `tol` of the order of the element size collapses a whole region rather than a
/// seam. Every call prints a one-line tally on stdout once the weld is done —
/// nodes welded, cells dropped, tolerance used.
///
/// # Parameters
///
/// - `mesh`: the mesh to weld.
/// - `tol`: the distance, `≥ 0`; `0` welds only coincident nodes.
/// - `in_place`: `False` builds a new mesh, where cells that collapse onto a
///   repeated node (a degenerate segment, triangle, …) are dropped. `True`
///   rewrites the connectivity of `mesh`'s **own** submeshes instead and returns
///   `mesh` itself. Since the union `mesh_a | mesh_b` shares its submeshes, welding
///   that union in place welds `mesh_a` and `mesh_b` themselves: afterwards they
///   share their interface nodes. The structure is kept (same submeshes, same
///   cells in the same order), so a cell that would collapse is an error, as is a
///   submesh already sealed by a finite-element space, a field or a matrix; both
///   are checked before anything is written.
///
/// # Returns
///
/// A new `Mesh` mirroring `mesh` (same submeshes, types and colours), or `mesh`
/// itself with `in_place=True`.
///
/// # Raises
///
/// `RuntimeError` if `tol` is negative, or, with `in_place=True`, for a cell that
/// would collapse or a sealed submesh.
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

/// The boundary of a surface mesh, as SEG2 loops.
///
/// An edge used by exactly one cell is a boundary edge; the boundary edges, pooled
/// across all surface submeshes, are chained into closed loops. The loops keep the
/// counterclockwise boundary orientation (outer loop counterclockwise, holes
/// clockwise), so the result feeds straight back into
/// `pyrucast.mesh.triangulate_surface`.
///
/// # Parameters
///
/// - `mesh`: a surface mesh, TRI3, QUA4, TRI6, QUA8 or QUA9.
/// - `angle_deg`: when given, each loop is split into open **sides** at its
///   corners — where the boundary turns by more than `angle_deg` degrees —, as
///   `pyrucast.mesh.skin` splits a volume's skin into flat faces; a loop with no
///   such corner stays whole. `None` keeps every loop whole.
///
/// # Returns
///
/// A `Mesh` with one SEG2 submesh per loop — or per side —, on the nodes of
/// `mesh`: a single loop for a simply connected domain, several with holes or
/// disjoint pieces.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not a surface mesh.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle_deg=None))]
pub fn border(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::border(&mesh.inner, angle_deg)?;
    Ok(PyMesh { inner: result })
}

/// The boundary surface of a volume mesh, one submesh per flat face.
///
/// A facet of a volume cell used by exactly one cell lies on the boundary;
/// sharing is decided on the facet's corners, so cells of different degrees still
/// cancel. The boundary facets, pooled across all volume submeshes, are grouped
/// into flat faces by flooding across shared edges as long as neighbouring facets
/// stay coplanar. **A facet keeps its own type**: a HEX8 yields QUA4, a TET10
/// TRI6, a HEX27 QUA9 — so the skin of a quadratic mesh is quadratic.
///
/// # Parameters
///
/// - `mesh`: a volume mesh — TET4, PYRA5, PENTA6, HEX8 or their quadratic
///   counterparts TET10, PENTA15, HEX20, HEX27.
/// - `angle_deg`: the largest angle, in degrees, between the normals of two
///   neighbouring facets of one flat face; `None` takes 1°.
///
/// # Returns
///
/// A `Mesh` with one submesh per flat face and per facet type — 6 for a cube, 5
/// for a prism, 5 for a pyramid. The facets keep their outward orientation and
/// the nodes of `mesh`.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not a 3-D volume mesh.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle_deg=None))]
pub fn skin(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
    // Python's optional kwarg is flattened **here**, at the boundary: past
    // this point the operator receives an angle, not an absence to test for.
    let angle_deg = angle_deg.unwrap_or(crate::ops::mesh::skin::DEFAULT_ANGLE_DEG);
    let result = crate::ops::mesh::skin(&mesh.inner, angle_deg)?;
    Ok(PyMesh { inner: result })
}

/// Make the orientation of the cells of `mesh` consistent.
///
/// Cells sharing a facet are oriented alike — all normals of a surface point the
/// same way, all segments of a curve run head to tail, all volume cells share one
/// handedness —, in any dimension, linear or quadratic. Each connected component
/// is seeded by its lowest-indexed cell, which keeps its orientation; the
/// absolute sense is not chosen (`pyrucast.mesh.invert` flips a whole mesh, a
/// hole's boundary for instance).
///
/// # Parameters
///
/// - `mesh`: the mesh to orient.
///
/// # Returns
///
/// A fresh `Mesh` sharing the nodes of `mesh`, which is left untouched.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn orient(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::orient(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Reorder the cells of a line mesh into a continuous chain.
///
/// The complement of `pyrucast.mesh.orient`: where `orient` fixes the
/// *direction* of the cells, `chain` fixes their *order* — each submesh is sorted
/// so that consecutive cells share a node, flipped where needed, so reading the
/// connectivity walks the curve from one end to the other.
///
/// # Parameters
///
/// - `mesh`: a line mesh, SEG2 or SEG3; each submesh is chained on its own and
///   must be one continuous chain, open or closed.
///
/// # Returns
///
/// A fresh `Mesh` sharing the nodes of `mesh`.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not a line mesh, for a node carrying three
/// segments (a branch), or for disjoint pieces in one submesh.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn chain(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::chain(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Reverse the orientation of every cell of `mesh`.
///
/// Flips each cell's winding, direction or handedness, in any dimension (POI1
/// cells are unchanged). With `pyrucast.mesh.orient`, it chooses the inside or
/// outside sense of a closed contour or surface.
///
/// # Parameters
///
/// - `mesh`: the mesh to flip.
///
/// # Returns
///
/// A fresh `Mesh` sharing the nodes of `mesh`, which is left untouched.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn invert(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::invert(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// A straight line of `n_elems` cells from node `a` to node `b`.
///
/// # Parameters
///
/// - `a`, `b`: the end nodes, of one `Coords`.
/// - `n_elems`: the number of cells, `≥ 1`.
/// - `element_type`: `"SEG2"` (the default) or `"SEG3"`.
///
/// # Returns
///
/// A `Mesh` of one zone, with fresh nodes between `a` and `b`.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type`.
/// - `RuntimeError` if `n_elems` is zero, or if `a` and `b` belong to different
///   `Coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (a, b, n_elems, element_type=None))]
pub fn line(
    a: PyRef<PyNode>,
    b: PyRef<PyNode>,
    n_elems: usize,
    element_type: Option<ElementType>,
) -> PyResult<PyMesh> {
    let element_type = element_type.unwrap_or(ElementType::SEG2);
    let mesh = crate::ops::mesh::line(a.as_node(), b.as_node(), n_elems, element_type)?;
    Ok(PyMesh { inner: mesh })
}

/// A closed circle of `n_elems` cells, centred on `center`.
///
/// # Parameters
///
/// - `center`: the centre node.
/// - `normal`: the normal of the circle's plane, three components even in 2-D.
/// - `radius`: the radius.
/// - `n_elems`: the number of cells.
/// - `element_type`: `"SEG2"` (the default) or `"SEG3"`.
///
/// # Returns
///
/// A `Mesh` of one zone, on fresh nodes.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type`.
/// - `RuntimeError` if `normal` does not have three components.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (center, normal, radius, n_elems, element_type=None))]
pub fn circle(
    center: PyRef<PyNode>,
    normal: Vec<f64>,
    radius: f64,
    n_elems: usize,
    element_type: Option<ElementType>,
) -> PyResult<PyMesh> {
    let element_type = element_type.unwrap_or(ElementType::SEG2);
    let mesh = crate::ops::mesh::circle(center.as_node(), &normal, radius, n_elems, element_type)?;
    Ok(PyMesh { inner: mesh })
}

/// An open arc of `n_elems` cells from `a` to `b`, centred on `center`.
///
/// The circle passes through both ends; the shorter arc is built.
///
/// # Parameters
///
/// - `a`, `b`: the end nodes.
/// - `center`: the centre node of the circle.
/// - `n_elems`: the number of cells, `≥ 1`.
/// - `element_type`: `"SEG2"` (the default) or `"SEG3"`.
///
/// # Returns
///
/// A `Mesh` of one zone, with fresh nodes between `a` and `b`.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type`.
/// - `RuntimeError` if `a`, `center` and `b` are aligned (the plane of the arc is
///   undefined).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (a, center, b, n_elems, element_type=None))]
pub fn arc(
    a: PyRef<PyNode>,
    center: PyRef<PyNode>,
    b: PyRef<PyNode>,
    n_elems: usize,
    element_type: Option<ElementType>,
) -> PyResult<PyMesh> {
    let element_type = element_type.unwrap_or(ElementType::SEG2);
    let mesh = crate::ops::mesh::arc(
        a.as_node(),
        center.as_node(),
        b.as_node(),
        n_elems,
        element_type,
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Fill the space between two line meshes with `n_layers` layers of cells.
///
/// Two **closed loops** (each ending on its own first node) give a closed ring,
/// the seam shared between its first and last cells.
///
/// # Parameters
///
/// - `mesh_a`, `mesh_b`: the two lines, with the same number of cells.
/// - `n_layers`: the number of layers, `≥ 1`.
/// - `element_type`: between SEG2 lines, `"QUA4"` (the default), `"TRI3"`,
///   `"QUA8"`, `"QUA9"` or `"TRI6"` — a QUA4 mesh is built first, then converted
///   (diagonal split for the triangles, promoted to quadratic for the others).
///   Between SEG3 lines, `"QUA8"` (the default) or `"QUA9"`, built directly: the
///   lines' own mid-nodes are reused, so a strip between two arcs follows them
///   exactly.
///
/// # Returns
///
/// A new `Mesh` sharing the nodes of `mesh_a` and `mesh_b`.
///
/// # Raises
///
/// `RuntimeError` if the two lines differ in cell count, for an unsupported
/// `element_type`, or for a closed loop swept onto an open line.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh_a, mesh_b, n_layers, element_type=None))]
pub fn sweep(
    mesh_a: PyRef<PyMesh>,
    mesh_b: PyRef<PyMesh>,
    n_layers: usize,
    element_type: Option<ElementType>,
) -> PyResult<PyMesh> {
    // `None`: the surface the lines' degree calls for.
    let element_type = match element_type {
        Some(et) => et,
        None if mesh_a.inner.element_types()? == [ElementType::SEG3] => ElementType::QUA8,
        None => ElementType::QUA4,
    };
    let mesh = crate::ops::mesh::sweep(&mesh_a.inner, &mesh_b.inner, n_layers, element_type)?;
    Ok(PyMesh { inner: mesh })
}

/// A structured surface bounded by four SEG2 sides, by transfinite interpolation.
///
/// The generalization of `pyrucast.mesh.sweep` from two lines to four (a Coons
/// patch).
///
/// # Parameters
///
/// - `side1`, `side2`, `side3`, `side4`: the four sides, forming a closed
///   contour `side1 → side2 → side3 → side4 → side1`, each ending on the start
///   node of the next. `side1`/`side3` and `side2`/`side4` are the two pairs of
///   **opposite** sides, each with the same cell count.
/// - `element_type`: `"QUA4"` (the default), `"TRI3"`, `"QUA8"`, `"QUA9"` or
///   `"TRI6"` — the same conversion path as `pyrucast.mesh.sweep`.
///
/// # Returns
///
/// A new `Mesh` on the nodes of the sides plus fresh interior nodes.
///
/// # Raises
///
/// `RuntimeError` if the sides do not close, if opposite sides differ in cell
/// count, or for an unsupported `element_type`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (side1, side2, side3, side4, element_type=None))]
pub fn transfinite(
    side1: PyRef<PyMesh>,
    side2: PyRef<PyMesh>,
    side3: PyRef<PyMesh>,
    side4: PyRef<PyMesh>,
    element_type: Option<ElementType>,
) -> PyResult<PyMesh> {
    let element_type = element_type.unwrap_or(ElementType::QUA4);
    let mesh = crate::ops::mesh::transfinite(
        &side1.inner,
        &side2.inner,
        &side3.inner,
        &side4.inner,
        element_type,
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Extrude `mesh` along `direction` in `n_layers` layers.
///
/// SEG2 → QUA4, TRI3 → PENTA6, QUA4 → HEX8, and their quadratic siblings
/// SEG3 → QUA8, TRI6 → PENTA15, QUA8 → HEX20.
///
/// The cells that fill the space — a line extruded in 2-D, a surface in 3-D —
/// come out right-handed (positive jacobian) whatever the side `direction`
/// points to: a source cell whose sweep would be left-handed is swept from
/// its reversed copy. Their faces, from `skin`, thus point out of the
/// material. A line extruded in 3-D gives a surface, whose cells keep the
/// source's order.
///
/// # Parameters
///
/// - `mesh`: the mesh to extrude.
/// - `direction`: the total displacement vector, with the dimension of the mesh.
/// - `n_layers`: the number of layers, `≥ 1`.
///
/// # Returns
///
/// A new `Mesh`, sharing the nodes of `mesh` on its first layer.
///
/// # Raises
///
/// `RuntimeError` if `n_layers` is zero, or if the produced cells need 3-D
/// coordinates the mesh does not have (a surface extruded in 2-D).
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn extrude(mesh: PyRef<PyMesh>, direction: Vec<f64>, n_layers: usize) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::extrude(&mesh.inner, &direction, n_layers)?;
    Ok(PyMesh { inner: result })
}

/// Revolve `mesh` about an axis in `n_layers` layers, over a total `angle`.
///
/// The rotational companion of `pyrucast.mesh.extrude`: SEG2 → QUA4,
/// TRI3 → PENTA6, QUA4 → HEX8, and in quadratic SEG3 → QUA8, TRI6 → PENTA15,
/// QUA8 → HEX20 (the mid-node of an edge along the sweep sits on the arc). A full
/// turn closes the ring: the last layer of nodes is the first one again.
///
/// A **line** may touch the axis by one corner: that node stays put, and the cells
/// touching it close into triangles — SEG2 → TRI3, SEG3 → TRI6 —, a submesh of
/// their own after the quadrangles. A line from the axis to the rim thus revolves
/// into a disc sector: a fan of triangles at the centre, a ring of quadrangles
/// around it, one node at the centre.
///
/// # Parameters
///
/// - `mesh`: the mesh to revolve. A surface cell may not touch the axis (its prism
///   would collapse), nor may a line lie along it.
/// - `angle`: the total angle, in radians, `|angle| ≤ 2π`; counterclockwise for a
///   positive angle in 2-D.
/// - `n_layers`: the number of layers, `≥ 1`.
/// - `center`: in 2-D, the centre of the revolution; in 3-D, a point of its axis.
/// - `axis`: in 3-D, the direction of the axis (right-handed), required; ignored
///   in 2-D.
///
/// # Returns
///
/// A new `Mesh`, sharing the nodes of `mesh` on its first layer.
///
/// # Raises
///
/// `RuntimeError` if `|angle|` exceeds a full turn, if the produced cells need 3-D
/// coordinates the mesh does not have, or for a cell the axis would collapse.
#[py_op(method_on = PyMesh)]
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

/// Fill the space between two matching surface meshes with `n_layers` layers.
///
/// The 3-D companion of `pyrucast.mesh.sweep`: TRI3 faces → PENTA6 prisms,
/// QUA4 faces → HEX8 hexahedra.
///
/// # Parameters
///
/// - `mesh_a`, `mesh_b`: the two surface meshes, TRI3 or QUA4, cell `i` of one
///   facing cell `i` of the other.
/// - `n_layers`: the number of layers, `≥ 1`.
///
/// # Returns
///
/// A new `Mesh` sharing the nodes of `mesh_a` and `mesh_b`.
///
/// # Raises
///
/// `RuntimeError` for a surface type other than TRI3 or QUA4, or if the two
/// meshes do not match.
#[py_op(method_on = PyMesh)]
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

/// The quadratic (Lagrange-2) copy of the linear mesh `mesh`.
///
/// Each element type is bumped to its quadratic sibling (TRI3 → TRI6,
/// HEX8 → HEX20, …). Corner nodes are reused; one node is created at the middle of
/// each edge and shared between the cells that use it.
///
/// # Parameters
///
/// - `mesh`: the linear mesh, left untouched.
///
/// # Returns
///
/// A new `Mesh`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn to_quadratic(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::to_quadratic(&mesh.inner)?;
    Ok(PyMesh { inner: result })
}

/// Split every cell of `mesh` into cells of `element_type`, adding no node.
///
/// Supported: the identity (already the target type, copied as it is),
/// `"QUA4"` → `"TRI3"` (two triangles per quadrangle, diagonal `(0, 2)`), and
/// `"HEX8"` → `"TET4"` (six tetrahedra per hexahedron, a conforming split).
/// Corner nodes are reused; face colours are kept. To promote to a quadratic type,
/// which creates mid-edge nodes, use `pyrucast.mesh.to_quadratic`.
///
/// # Parameters
///
/// - `mesh`: the mesh to convert, left untouched.
/// - `element_type`: the target element type.
///
/// # Returns
///
/// A new `Mesh` on the nodes of `mesh`.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type`.
/// - `RuntimeError` for a conversion that is not supported.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn convert(mesh: PyRef<PyMesh>, element_type: ElementType) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::convert(&mesh.inner, element_type)?;
    Ok(PyMesh { inner: result })
}

/// Copy `mesh`, onto fresh nodes or onto the same nodes.
///
/// Both modes keep the submesh order, element types, cell order and face colours,
/// and hand back a mesh that is **never sealed** — editable even when the source
/// has been frozen by a finite-element space, a field or a matrix.
///
/// # Parameters
///
/// - `mesh`: the mesh to copy, left untouched.
/// - `new_nodes`: `True` gives each distinct node of `mesh` one fresh node at the
///   same position (nodes shared between cells stay shared), so the two meshes
///   no longer move together. `False` copies the connectivity only: the two
///   meshes share their nodes, and moving one moves it in both.
///
/// # Returns
///
/// A new `Mesh`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, new_nodes=true))]
pub fn copy(mesh: PyRef<PyMesh>, new_nodes: bool) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::copy(&mesh.inner, new_nodes)?;
    Ok(PyMesh { inner: result })
}

/// Push every node of `mesh` radially onto the sphere of centre `center`.
///
/// Corners and mid-nodes alike move along the ray from `center` through them, to
/// the distance `radius`. In a plane through `center` the sphere is a circle: a
/// contour of that plane lands on it node for node, each at its own angle — the
/// inner loop of a ring, ready for `pyrucast.mesh.sweep`.
///
/// # Parameters
///
/// - `mesh`: the mesh to project, left untouched.
/// - `center`: the centre of the sphere.
/// - `radius`: the radius of the sphere.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes, one per distinct node (a closed loop stays
/// closed), in the same zones, types and colours.
///
/// # Raises
///
/// `RuntimeError` for a node on `center`, which has no direction.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn project_on_sphere(mesh: PyRef<PyMesh>, center: Vec<f64>, radius: f64) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::project_on_sphere(&mesh.inner, &center, radius)?;
    Ok(PyMesh { inner: result })
}

/// Translate `mesh` by `vector`.
///
/// # Parameters
///
/// - `mesh`: the mesh to move.
/// - `vector`: the translation, with the dimension of the mesh.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes; `mesh` is left untouched.
///
/// # Raises
///
/// `RuntimeError` if `vector` does not have the dimension of the mesh.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn translate(mesh: PyRef<PyMesh>, vector: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::translate(&mesh.inner, &vector)?;
    Ok(PyMesh { inner: result })
}

/// Rotate `mesh` by `angle` about `center`.
///
/// # Parameters
///
/// - `mesh`: the mesh to turn.
/// - `angle`: the angle, in radians.
/// - `center`: in 2-D, the centre of the rotation; in 3-D, a point of its axis.
/// - `axis`: in 3-D, the direction of the axis (right-handed), required; ignored
///   in 2-D.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes; `mesh` is left untouched.
///
/// # Raises
///
/// `RuntimeError` if a point does not have the dimension of the mesh, or in 3-D
/// without `axis`.
#[py_op(method_on = PyMesh)]
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

/// Mirror `mesh` through the point `center`: every node goes to `2·center − x`.
///
/// In 3-D the map reverses orientation, so the cells are reordered (as
/// `pyrucast.mesh.invert` does) to keep the copy's Jacobians positive; in 2-D it
/// is a plain half-turn and nothing is reordered.
///
/// # Parameters
///
/// - `mesh`: the mesh to mirror.
/// - `center`: the centre of symmetry, with the dimension of the mesh.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes; `mesh` is left untouched.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn symmetry_point(mesh: PyRef<PyMesh>, center: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::symmetry_point(&mesh.inner, &center)?;
    Ok(PyMesh { inner: result })
}

/// Mirror `mesh` through the infinite line through `a` and `b`.
///
/// In 2-D this is the mirror image about the line; in 3-D it is the half-turn
/// about it (the mirror image through a plane is
/// `pyrucast.mesh.symmetry_plane`). It reverses orientation in 2-D only, where
/// the cells are reordered (as `pyrucast.mesh.invert` does) to keep the copy's
/// Jacobians positive.
///
/// # Parameters
///
/// - `mesh`: the mesh to mirror.
/// - `a`, `b`: two distinct points of the line.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes; `mesh` is left untouched.
///
/// # Raises
///
/// `RuntimeError` if `a` and `b` coincide.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn symmetry_line(mesh: PyRef<PyMesh>, a: Vec<f64>, b: Vec<f64>) -> PyResult<PyMesh> {
    let result = crate::ops::mesh::symmetry_line(&mesh.inner, &a, &b)?;
    Ok(PyMesh { inner: result })
}

/// Mirror `mesh` through the plane through the points `a`, `b` and `c`, in 3-D.
///
/// Only the plane the three points span matters, not their order. The map always
/// reverses orientation, so the cells are reordered (as `pyrucast.mesh.invert`
/// does) to keep the copy's Jacobians positive. In 2-D the mirror about a line is
/// `pyrucast.mesh.symmetry_line`.
///
/// # Parameters
///
/// - `mesh`: the mesh to mirror, in 3-D.
/// - `a`, `b`, `c`: three points of the plane, not aligned.
///
/// # Returns
///
/// A fresh `Mesh` with its own nodes; `mesh` is left untouched.
///
/// # Raises
///
/// `RuntimeError` if the mesh is not 3-D, or if the three points are aligned.
#[py_op(method_on = PyMesh)]
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

/// Mesh the inside of the closed SEG2 loops of `contour`, Delaunay and Ruppert.
///
/// A constrained Delaunay triangulation refined by Ruppert's algorithm. The
/// contour may be 2-D, or a planar loop in 3-D, meshed in its best-fit plane and
/// lifted back.
///
/// # Parameters
///
/// - `contour`: **one or more** closed SEG2 loops, each oriented by the caller: a
///   **counterclockwise** loop is a domain's outer boundary, a **clockwise** loop
///   a hole inside an outer loop. Several disjoint counterclockwise loops mesh
///   several independent domains at once.
/// - `element_type`: `"TRI3"`, or `"QUA4"`, quadrangle-dominant: the result may
///   also carry a few triangles at the boundary.
/// - `size`: the target edge length; `None` takes the mean boundary edge length,
///   domain by domain.
///
/// # Returns
///
/// A new `Mesh` on the contour's nodes plus fresh interior nodes.
///
/// # Raises
///
/// `RuntimeError` for a loop of fewer than three segments or not closed, or for
/// an `element_type` other than TRI3 or QUA4.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None))]
pub fn triangulate_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: ElementType,
    size: Option<f64>,
) -> PyResult<PyMesh> {
    // Poll Python signals while meshing so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesh::triangulate_surface_cancellable(
        &contour.inner,
        element_type,
        size,
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Pave the inside of closed loops with quadrangles, in rows walking inward.
///
/// The quadrangle companion of `pyrucast.mesh.triangulate_surface`, and the one
/// to reach for when the mesh is going to be computed on: it lays quadrangles
/// down directly, in rows that follow the contour, instead of triangulating and
/// pairing triangles up afterwards. The contour is untouchable: its nodes come
/// back at their own positions and no node is ever added on a boundary edge. A
/// contour the paver cannot work with is therefore reported rather than worked
/// around.
///
/// # Parameters
///
/// - `contour`: **one or more** closed SEG2 loops, oriented as for
///   `pyrucast.mesh.triangulate_surface`: a **counterclockwise** loop is a
///   domain's outer boundary, a **clockwise** loop a hole. Several disjoint
///   counterclockwise loops mesh several independent domains at once. It may be
///   2-D, or a planar loop in 3-D, meshed in its best-fit plane and lifted back.
/// - `element_type`: `"QUA4"`, `"QUA8"` or `"QUA9"`.
/// - `size`: the target edge length; `None` takes the mean boundary edge length,
///   domain by domain.
/// - `prefer_quads`: `False` lets an odd loop cost one triangle, returned in a
///   separate TRI3 submesh along with the few cells a distorted leftover polygon
///   could not make square. `True` removes the first kind, not the second, and
///   raises on a loop with an odd number of segments: a polygon with an odd
///   number of sides has no filling by quadrangles alone, and evening the count
///   would mean adding a boundary node.
/// - `relax`: what the front may do to itself between two rows. After each row
///   the fresh chain is smoothed, which keeps the front from kinking — and, being
///   a Laplacian, rounds its corners off. A front sheds nodes only at its
///   **corners**, so once they are rounded away it keeps every node it has while
///   its perimeter shrinks, and the middle comes out finer than asked: a plain
///   20 × 20 square at size 1 gives 600 cells instead of 400.
///   - `"free"` (`None`, the default): a node moves wherever the smoothing
///     points. The only mode that never lets the front kink.
///   - `"along"`: the same step, kept only **along** the front: the spacing is
///     evened out, the shape is not. The square then comes out as the 400 exact
///     squares anyone would draw.
///   - `"none"`: the front stays exactly where the row put it.
///
///   `"along"` and `"none"` win on anything with corners to keep and can kink on
///   a curve, where there is nothing to preserve and everything to straighten.
///
/// # Returns
///
/// A new `Mesh` on the contour's nodes plus fresh interior nodes.
///
/// # Raises
///
/// - `ValueError` for an unknown `relax`.
/// - `RuntimeError` for an `element_type` other than QUA4, QUA8 or QUA9, with
///   `prefer_quads=True` on an odd loop, or for a contour too coarse for the size
///   asked, where the front folds onto itself.
/// - `RuntimeError` also for a run that fails to converge.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, prefer_quads=false, relax=None))]
pub fn pave_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: ElementType,
    size: Option<f64>,
    prefer_quads: bool,
    relax: Option<FrontRelax>,
) -> PyResult<PyMesh> {
    // Poll Python signals while paving so a long run stays Ctrl+C-able.
    let mesh = crate::ops::mesh::pave_surface_cancellable(
        &contour.inner,
        element_type,
        size,
        prefer_quads,
        relax.unwrap_or_default(),
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Move the interior nodes of a 2-D surface mesh to improve its cells.
///
/// The connectivity and the boundary stay exactly as they are. Two guarantees
/// hold: **no node on the boundary ever moves**, and a position is taken only
/// when every incident cell stays valid and the worst incident quality does not
/// get worse. Smoothing cannot fix a node with the wrong number of cells around
/// it — that is `pyrucast.mesh.cleanup`'s job, and running it first is usually
/// what unlocks the smoothing.
///
/// # Parameters
///
/// - `mesh`: a 2-D mesh; its TRI3 and QUA4 cells are smoothed, its POI1 and SEG2
///   submeshes ignored.
/// - `sweeps`: the number of passes.
/// - `angular`: `True` uses angle-based smoothing, which aims at the right angles
///   a quadrangle wants; `False` the plain Laplacian, which aims at the
///   barycentre of the neighbours and knows nothing about angles.
/// - `in_place`: `True` writes the new positions onto your own nodes and returns
///   `mesh` itself; `False` duplicates the moved nodes into a fresh mesh, the
///   boundary's nodes being shared since they never move.
///
/// # Returns
///
/// A new `Mesh`, or `mesh` itself with `in_place=True`.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not 2-D, or has no surface cells (TRI3/QUA4).
#[py_op(method_on = PyMesh)]
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

/// Fix the connectivity of a 2-D surface mesh: doublets and wrong valences.
///
/// It removes the doublets, gives up the nodes that have only three cells around
/// them, and switches the diagonals that lower the valence error. **No node of
/// the contour ever moves**, and none is ever given up: the mesh keeps exactly
/// the boundary it came with. Smoothing cannot fix these: the angles around a
/// node sum to 2π whatever the positions. Turning a lone triangle into a
/// quadrangle is `pyrucast.mesh.merge_triangles`'s job.
///
/// # Parameters
///
/// - `mesh`: a 2-D surface mesh, TRI3 and QUA4, never modified.
///
/// # Returns
///
/// A new `Mesh`, wound the way `mesh` was, clockwise or not. Your own nodes come
/// back untouched, apart from those on a ring a collapse relaxed, which are
/// duplicated.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not 2-D, or has no surface cells (TRI3/QUA4).
///
/// # Notes
///
/// A **doublet** is an interior node with only two quadrangles around it, which
/// therefore share two edges. A node of the **wrong valence** wants four cells
/// and has three or five. Three moves answer that:
///
/// - the **diagonal switch**, for a node of valence five or more: two quadrangles
///   sharing an edge form a hexagon, which splits across any of its three
///   diagonals. It is applied only when it strictly lowers the valence error and
///   leaves both cells convex.
/// - the **collapse**, for an interior node with only three cells: round a node
///   carrying `q` quadrangles and `t` triangles the star is bounded by a polygon
///   of `n = 2q + t` sides, which decomposes with no interior node into `q'`
///   quadrangles and `t'` triangles such that `2q' + t' = n − 2` — always a cell
///   fewer. So the node is given up along with that cell. Removing a node leaves
///   the ring round it stretched, so the move is judged after relaxing that ring;
///   a move that does not improve the neighbourhood's worst cell is undone whole.
/// - the **pair collapse**, for two interior nodes sharing an edge with at least
///   one short of a cell: together they carry `val(a) + val(b) − 2` cells and
///   both re-cut with two cells fewer.
///
/// Triangles only ever leave two at a time: their number has the parity of the
/// boundary's edge count, which nothing here may change.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn cleanup(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let out = crate::ops::mesh::cleanup(&mesh.inner)?;
    Ok(PyMesh { inner: out })
}

/// Remove the triangles from a quadrangle-dominant mesh `mesh`, in pairs.
///
/// # Parameters
///
/// - `mesh`: a 2-D surface mesh, TRI3 and QUA4.
///
/// # Returns
///
/// A new `Mesh`.
///
/// # Raises
///
/// `RuntimeError` if `mesh` is not 2-D, or has no surface cells (TRI3/QUA4).
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge_triangles(mesh: PyRef<PyMesh>) -> PyResult<PyMesh> {
    let out = crate::ops::mesh::merge_triangles(&mesh.inner)?;
    Ok(PyMesh { inner: out })
}

/// Mesh the inside of closed loops with a structured grid core and a frontal band.
///
/// The regular companion of `pyrucast.mesh.pave_surface`, with the same input,
/// output and untouchable contour. `pave_surface` walks a front inward until two
/// of its rows meet, and that meeting line carries the valence defects, the
/// leftover triangles and the flattest cells — even a plain rectangle comes out
/// as an onion with four diagonal seams. `grid_surface` lays a tensor grid
/// instead, and leaves the front only what the grid could not reach.
///
/// The grid's lines are taken from the contour: every axis-aligned edge long
/// enough to be a feature pins a line, and the gaps are subdivided at about
/// `size`. A grid node landing on a contour node **is** that node. On a
/// rectilinear domain laid out for the grid there is no band at all: every cell
/// a rectangle, no triangle. That is the one thing asked of the caller: **break
/// every side at the shape's own corners and let each piece take a whole number
/// of cells**. A contour that does not just gets more band and less grid, down to
/// the quality of `pave_surface` in the worst case.
///
/// # Parameters
///
/// - `contour`: **one or more** closed SEG2 loops, oriented as for
///   `pyrucast.mesh.triangulate_surface`: a **counterclockwise** loop is a
///   domain's outer boundary, a **clockwise** loop a hole. Several disjoint
///   counterclockwise loops mesh several independent domains at once. It may be
///   2-D, or a planar loop in 3-D, meshed in its best-fit plane and lifted back.
/// - `element_type`: `"QUA4"`, `"QUA8"` or `"QUA9"`.
/// - `size`: the target edge length; `None` takes the mean boundary edge length,
///   domain by domain.
/// - `band`: extra clearance, in cells, between the core and the contour. Zero is
///   the useful value; raise it only for a contour the grid cannot meet, such as
///   a curve, where giving the front a couple of cells to work in beats letting
///   it fight for a sliver.
/// - `prefer_quads`: `False` lets an odd loop cost one triangle, returned in a
///   separate TRI3 submesh along with the few cells a distorted leftover polygon
///   could not make square. `True` removes the first kind, not the second, and
///   raises on a loop with an odd number of segments: a polygon with an odd
///   number of sides has no filling by quadrangles alone, and evening the count
///   would mean adding a boundary node.
/// - `relax`: what the front may do to itself between two rows — `"free"`
///   (`None`, the default), `"along"` or `"none"`; see
///   `pyrucast.mesh.pave_surface`.
///
/// # Returns
///
/// A new `Mesh` on the contour's nodes plus fresh interior nodes.
///
/// # Raises
///
/// - `ValueError` for an unknown `relax`.
/// - `RuntimeError` for an `element_type` other than QUA4, QUA8 or QUA9, with
///   `prefer_quads=True` on an odd loop, or for a contour too coarse for the size
///   asked, where the front folds onto itself.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, band=0, prefer_quads=false, relax=None))]
pub fn grid_surface(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: ElementType,
    size: Option<f64>,
    band: usize,
    prefer_quads: bool,
    relax: Option<FrontRelax>,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesh::grid_surface_cancellable(
        &contour.inner,
        element_type,
        size,
        band,
        prefer_quads,
        relax.unwrap_or_default(),
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Mesh the inside of closed loops with a grid core, a line per contour node.
///
/// The sibling of `pyrucast.mesh.grid_surface`, which it does not replace: same
/// input, output and untouchable contour, and neither wins everywhere.
/// `grid_surface` pins a line on the coordinate each aligned side lies on; every
/// line is straight. `grid_surface2` gives every node of the contour the line
/// that crosses it, collapses the bands too thin to be cells, and lets a grid
/// node within a quarter cell of a contour node move onto it. A row is therefore
/// a polyline, so one row can meet two facing walls at two different heights: a
/// wall cut into ten can face a wall cut into eleven.
///
/// **Reach for it on a rectilinear shape**, all the more so when its sides were
/// not cut at the corners facing them; **reach for `grid_surface` on anything
/// curved**. Measured worst cell, `grid_surface` then `grid_surface2`: plate with
/// a step off the grid 0.405 / 0.963, L with arbitrary dimensions 0.437 / 0.979,
/// crenellated profile 0.323 / 0.651, house with a pitched roof 0.420 / 0.548. On
/// a **circle** the order reverses hard — 0.288 against 0.005 —: nothing dictates
/// a grid line over most of a curve.
///
/// # Parameters
///
/// - `contour`: **one or more** closed SEG2 loops, oriented as for
///   `pyrucast.mesh.triangulate_surface`: a **counterclockwise** loop is a
///   domain's outer boundary, a **clockwise** loop a hole. Several disjoint
///   counterclockwise loops mesh several independent domains at once. It may be
///   2-D, or a planar loop in 3-D, meshed in its best-fit plane and lifted back.
/// - `element_type`: `"QUA4"`, `"QUA8"` or `"QUA9"`.
/// - `size`: the target edge length; `None` takes the mean boundary edge length,
///   domain by domain.
/// - `band`: as for `pyrucast.mesh.grid_surface`.
/// - `prefer_quads`: `False` lets an odd loop cost one triangle, returned in a
///   separate TRI3 submesh along with the few cells a distorted leftover polygon
///   could not make square. `True` removes the first kind, not the second, and
///   raises on a loop with an odd number of segments: a polygon with an odd
///   number of sides has no filling by quadrangles alone, and evening the count
///   would mean adding a boundary node.
/// - `relax`: what the front may do to itself between two rows — `"free"`
///   (`None`, the default), `"along"` or `"none"`; see
///   `pyrucast.mesh.pave_surface`.
///
/// # Returns
///
/// A new `Mesh` on the contour's nodes plus fresh interior nodes.
///
/// # Raises
///
/// - `ValueError` for an unknown `relax`.
/// - `RuntimeError` for an `element_type` other than QUA4, QUA8 or QUA9, with
///   `prefer_quads=True` on an odd loop, or for a contour too coarse for the size
///   asked, where the front folds onto itself.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (contour, element_type, size=None, band=0, prefer_quads=false, relax=None))]
pub fn grid_surface2(
    py: Python<'_>,
    contour: PyRef<PyMesh>,
    element_type: ElementType,
    size: Option<f64>,
    band: usize,
    prefer_quads: bool,
    relax: Option<FrontRelax>,
) -> PyResult<PyMesh> {
    let mesh = crate::ops::mesh::grid_surface2_cancellable(
        &contour.inner,
        element_type,
        size,
        band,
        prefer_quads,
        relax.unwrap_or_default(),
        &PySignals(py),
    )?;
    Ok(PyMesh { inner: mesh })
}

/// Mesh a closed envelope: a hexahedral boundary layer over a tetrahedral core.
///
/// The 3-D companion of `pyrucast.mesh.pave_surface`. Hexahedra go where they
/// matter, in the layer against the boundary where gradients are steepest; the
/// smooth interior is left to tetrahedra. Pyramids join the two: the layer's
/// inner faces are squares and a tetrahedron has none, so without them the mesh
/// could not be conforming.
///
/// # Parameters
///
/// - `envelope`: a closed surface of QUA4 and/or TRI3 facets, its normals pointing
///   **out of the material**; its nodes are reused as they are.
/// - `layers`: the number of boundary layers, grown inward.
/// - `thickness`: the depth of each layer; `None` takes the envelope's mean edge
///   length, which gives roughly cube-shaped cells.
/// - `size`: the target cell size of the tetrahedral core.
///
/// # Returns
///
/// A new `Mesh` with a HEX8 submesh (born of the QUA4 facets), a PENTA6 one (born
/// of the TRI3 facets), a PYRA5 one and a TET4 one, each present only if not
/// empty.
///
/// # Raises
///
/// `RuntimeError` if `envelope` is not closed.
#[py_op(method_on = PyMesh)]
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

/// Fill the inside of a closed TRI3 `envelope` with TET4 cells.
///
/// The 3-D companion of `pyrucast.mesh.triangulate_surface`. A concave shape is
/// fine, and an internal cavity is simply another closed surface whose normals
/// point into the hole. The envelope's nodes are reused as they are, and nodes
/// are added inside the solid so the cells come out well shaped.
///
/// # Parameters
///
/// - `envelope`: the closed TRI3 surface, its normals pointing **out of the
///   material**.
/// - `size`: the target edge length; `None` takes the mean edge length of the
///   envelope.
/// - `allow_surface_nodes`: lets the mesher cut the envelope finer where it
///   cannot otherwise fit it or make it usable. The shape is kept — every added
///   node lies on the edge or facet it divides —, but the skin of the result no
///   longer matches the surface handed in, and a warning on stderr says how many
///   nodes were added. Without it, such a surface is refused rather than meshed
///   badly.
///
/// # Returns
///
/// A new `Mesh` of TET4 cells.
///
/// # Raises
///
/// `RuntimeError` if `envelope` is not a closed surface, or, without
/// `allow_surface_nodes`, for a surface the mesher cannot fit.
#[py_op(method_on = PyMesh)]
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

/// The pyrucast element type of a gmsh element-type code.
///
/// The translation `pyrucast.mesh.from_gmsh` applies to every block before
/// handing it to `pyrucast.mesh.from_arrays`.
///
/// # Parameters
///
/// - `code`: the gmsh code, `2` for a 3-node triangle.
///
/// # Returns
///
/// The name of the pyrucast element type, `"TRI3"` for `2`.
///
/// # Raises
///
/// `RuntimeError` for a gmsh type pyrucast has no counterpart for (third order and
/// above, …).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn element_type_from_gmsh(code: u32) -> PyResult<&'static str> {
    Ok(crate::ops::mesh::gmsh::element_type_from_gmsh(code)?.name())
}

/// Pyrucast's Gauss rule for `element_type`, carried into an external element.
///
/// The points are mapped into the external reference element and the weights
/// scaled by the ratio of the reference measures.
///
/// # Parameters
///
/// - `element_type`: the element type.
/// - `ref_nodes`: the nodes of the external reference element, `dim` coordinates
///   each, numbered in `order`.
/// - `order`: the numbering of `ref_nodes`: `"pyrucast"`, `"gmsh"` or `"med"`.
///
/// # Returns
///
/// `(xi, weights)`: the points, in pyrucast's order, and their weights.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type` or `order`.
/// - `RuntimeError` if the external element is not an affine image of pyrucast's
///   under that numbering.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn gauss_to_external(
    element_type: ElementType,
    ref_nodes: Vec<f64>,
    order: NodeOrder,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    Ok(crate::ops::mesh::gauss_map::gauss_to_external(
        element_type,
        order,
        &ref_nodes,
    )?)
}

/// Where each Gauss point of `element_type` sits in an external rule.
///
/// # Parameters
///
/// - `element_type`: the element type.
/// - `ref_nodes`: the nodes of the external reference element, numbered in
///   `order`.
/// - `xi`, `weights`: the external rule, declared in that element.
/// - `order`: the numbering of `ref_nodes`: `"pyrucast"`, `"gmsh"` or `"med"`.
///
/// # Returns
///
/// For each Gauss point of pyrucast, the index of the same point in `xi`.
///
/// # Raises
///
/// - `ValueError` for an unknown `element_type` or `order`.
/// - `RuntimeError` if the two rules are not the same quadrature.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn match_gauss(
    element_type: ElementType,
    ref_nodes: Vec<f64>,
    xi: Vec<f64>,
    weights: Vec<f64>,
    order: NodeOrder,
) -> PyResult<Vec<usize>> {
    Ok(crate::ops::mesh::gauss_map::match_gauss(
        element_type,
        order,
        &ref_nodes,
        &xi,
        &weights,
    )?)
}

/// Build meshes and fields from flat arrays, the shape `export.to_arrays` returns.
///
/// The one import every exchange format goes through
/// (`pyrucast.mesh.from_gmsh` and `pyrucast.mesh.from_medcoupling` are built on
/// it). Only the nodes a cell references are created. numpy arrays are read
/// **without being copied**, through the buffer protocol; plain Python lists work
/// too, at the cost of a conversion.
///
/// # Parameters
///
/// - `coords`: where the nodes are added; its dimension decides how many
///   coordinates are kept.
/// - `node_tags`: one integer tag per node (`uint64` or `int64`).
/// - `node_coords`: 1 to 3 coordinates per tag, in the same order, padded with
///   zeros.
/// - `blocks`: a sequence of `(element_type, node_tags, cell_tags, groups)`, one
///   per block of cells sharing a type and a combination of groups: the pyrucast
///   type name (`"TET4"`), the flat connectivity in node tags, one tag per cell
///   (or an empty sequence when no cell field names the block), and the names of
///   the groups the cells belong to. A 3-tuple without `cell_tags` is accepted
///   too.
/// - `node_fields`: a sequence of `(components, node_tags, values)`,
///   `len(node_tags) * len(components)` values, node-major. A node field keeps
///   the nodes the mesh created.
/// - `cell_fields`: a sequence of `(components, cell_tags, values, layout)`.
///   `layout` is `"cell"` (one value per cell and component) or a sequence of
///   Gauss rules `(element_type, ref_nodes, xi, weights)` in the format's own
///   reference element — one value per point, component and cell, the points in
///   the order of the rule, which must be pyrucast's. A cell field gets a zone on
///   every group all of whose cells it defines.
/// - `order`: the node numbering inside a cell: `"pyrucast"` (the same as VTK),
///   `"gmsh"` or `"med"`.
///
/// # Returns
///
/// `(meshes, node_fields, element_fields)`: a `dict` from group name to `Mesh`
/// (all on `coords`, in order of first appearance), then the fields in the order
/// they were given.
///
/// # Raises
///
/// - `ValueError` for an unknown `order` or element type.
/// - `RuntimeError` for a cell referencing a node tag that is not given.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (coords, node_tags, node_coords, blocks, *, node_fields = Vec::new(), cell_fields = Vec::new(), order = NodeOrder::Pyrucast))]
#[allow(clippy::too_many_arguments)]
pub fn from_arrays<'py>(
    py: Python<'py>,
    coords: PyRef<PyCoords>,
    node_tags: &Bound<'py, PyAny>,
    node_coords: &Bound<'py, PyAny>,
    blocks: &Bound<'py, PyAny>,
    node_fields: Vec<Py<PyAny>>,
    cell_fields: Vec<Py<PyAny>>,
    order: NodeOrder,
) -> PyResult<(Bound<'py, PyDict>, Vec<PyNodeField>, Vec<PyElementField>)> {
    // The tag type follows the node table: medcoupling hands `int64`, gmsh
    // `uint64`. Every other tag array is borrowed as the same type (copied
    // only if its dtype differs).
    let signed = pyo3::buffer::PyBuffer::<i64>::get(node_tags).is_ok();
    let node_fields: Vec<Bound<'py, PyAny>> =
        node_fields.into_iter().map(|f| f.into_bound(py)).collect();
    let cell_fields: Vec<Bound<'py, PyAny>> =
        cell_fields.into_iter().map(|f| f.into_bound(py)).collect();
    let inputs = ExchangeInputs::unpack(blocks, &node_fields, &cell_fields)?;
    let out = if signed {
        inputs.import::<i64>(coords.handle.clone(), node_tags, node_coords, order)?
    } else {
        inputs.import::<u64>(coords.handle.clone(), node_tags, node_coords, order)?
    };
    Ok((
        groups_to_dict(py, out.groups)?,
        out.node_fields
            .into_iter()
            .map(|inner| PyNodeField { inner })
            .collect(),
        out.element_fields
            .into_iter()
            .map(|inner| PyElementField { inner })
            .collect(),
    ))
}

/// The Python objects behind `from_arrays`' arguments, unpacked **first** so
/// they outlive the slices borrowed from them.
struct ExchangeInputs<'py> {
    blocks: Vec<BlockObjects<'py>>,
    node_fields: Vec<(Vec<String>, Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    cell_fields: Vec<CellFieldObjects<'py>>,
}

/// One block as given: `(element_type, node_tags, cell_tags, groups)`.
type BlockObjects<'py> = (
    ElementType,
    Bound<'py, PyAny>,
    Option<Bound<'py, PyAny>>,
    Vec<String>,
);

/// One cell field as given: `(components, cell_tags, values, rules)`, the
/// rules absent for the `"cell"` layout.
type CellFieldObjects<'py> = (
    Vec<String>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Option<Vec<RuleObjects<'py>>>,
);

/// One Gauss rule as given: `(element_type, ref_nodes, xi, weights)`.
type RuleObjects<'py> = (
    ElementType,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

fn nth<'py>(item: &Bound<'py, PyAny>, k: usize, what: &str) -> PyResult<Bound<'py, PyAny>> {
    item.get_item(k)
        .map_err(|_| PyValueError::new_err(format!("{what}: missing entry {k}")))
}

impl<'py> ExchangeInputs<'py> {
    fn unpack(
        blocks: &Bound<'py, PyAny>,
        node_fields: &[Bound<'py, PyAny>],
        cell_fields: &[Bound<'py, PyAny>],
    ) -> PyResult<Self> {
        let mut bs = Vec::new();
        for (i, item) in blocks.try_iter()?.enumerate() {
            let item = item?;
            let what = format!("block {i}: expected (element_type, node_tags, cell_tags, groups)");
            let (cells, groups) = match item.len()? {
                3 => (None, nth(&item, 2, &what)?),
                4 => (Some(nth(&item, 2, &what)?), nth(&item, 3, &what)?),
                _ => return Err(PyValueError::new_err(what)),
            };
            bs.push((
                nth(&item, 0, &what)?.extract()?,
                nth(&item, 1, &what)?,
                cells,
                groups.extract()?,
            ));
        }
        let mut nfs = Vec::new();
        for (i, item) in node_fields.iter().enumerate() {
            let what = format!("node field {i}: expected (components, node_tags, values)");
            nfs.push((
                nth(item, 0, &what)?.extract()?,
                nth(item, 1, &what)?,
                nth(item, 2, &what)?,
            ));
        }
        let mut cfs = Vec::new();
        for (i, item) in cell_fields.iter().enumerate() {
            let what = format!("cell field {i}: expected (components, cell_tags, values, layout)");
            let layout = nth(item, 3, &what)?;
            let rules = if let Ok(s) = layout.extract::<String>() {
                if !s.trim().eq_ignore_ascii_case("cell") {
                    return Err(PyValueError::new_err(format!(
                        "cell field {i}: layout '{s}' is neither 'cell' nor a list of Gauss rules"
                    )));
                }
                None
            } else {
                let mut rules = Vec::new();
                for r in layout.try_iter()? {
                    let r = r?;
                    let what = format!("cell field {i}: expected Gauss rules (element_type, ref_nodes, xi, weights)");
                    rules.push((
                        nth(&r, 0, &what)?.extract()?,
                        nth(&r, 1, &what)?,
                        nth(&r, 2, &what)?,
                        nth(&r, 3, &what)?,
                    ));
                }
                Some(rules)
            };
            cfs.push((
                nth(item, 0, &what)?.extract()?,
                nth(item, 1, &what)?,
                nth(item, 2, &what)?,
                rules,
            ));
        }
        Ok(Self {
            blocks: bs,
            node_fields: nfs,
            cell_fields: cfs,
        })
    }

    fn import<T>(
        &self,
        coords: crate::handle::Handle<crate::coords::Coords>,
        node_tags: &Bound<'py, PyAny>,
        node_coords: &Bound<'py, PyAny>,
        order: NodeOrder,
    ) -> PyResult<crate::ops::mesh::Imported>
    where
        T: crate::ops::mesh::Tag + pyo3::buffer::Element + for<'b, 'p> FromPyObject<'b, 'p>,
    {
        use crate::py::arrays::borrow;
        let tags = borrow::<T>(node_tags)?;
        let xyz = borrow::<f64>(node_coords)?;
        let conns = self
            .blocks
            .iter()
            .map(|b| borrow::<T>(&b.1))
            .collect::<PyResult<Vec<_>>>()?;
        let cells = self
            .blocks
            .iter()
            .map(|b| b.2.as_ref().map(borrow::<T>).transpose())
            .collect::<PyResult<Vec<_>>>()?;
        let blocks: Vec<CellBlock<'_, T>> = self
            .blocks
            .iter()
            .zip(&conns)
            .zip(&cells)
            .map(|((b, conn), cells)| CellBlock {
                element_type: b.0,
                node_tags: conn.as_slice(),
                cell_tags: cells.as_ref().map_or(&[], |c| c.as_slice()),
                groups: &b.3,
            })
            .collect();

        let nf_arrays = self
            .node_fields
            .iter()
            .map(|f| Ok((borrow::<T>(&f.1)?, borrow::<f64>(&f.2)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let node_values: Vec<NodeValues<'_, T>> = self
            .node_fields
            .iter()
            .zip(&nf_arrays)
            .map(|(f, (t, v))| NodeValues {
                components: &f.0,
                node_tags: t.as_slice(),
                values: v.as_slice(),
            })
            .collect();

        let cf_arrays = self
            .cell_fields
            .iter()
            .map(|f| Ok((borrow::<T>(&f.1)?, borrow::<f64>(&f.2)?)))
            .collect::<PyResult<Vec<_>>>()?;
        let rule_arrays = self
            .cell_fields
            .iter()
            .map(|f| {
                f.3.iter()
                    .flatten()
                    .map(|r| {
                        Ok((
                            borrow::<f64>(&r.1)?,
                            borrow::<f64>(&r.2)?,
                            borrow::<f64>(&r.3)?,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let rules: Vec<Vec<GaussRule<'_>>> = self
            .cell_fields
            .iter()
            .zip(&rule_arrays)
            .map(|(f, arrays)| {
                f.3.iter()
                    .flatten()
                    .zip(arrays)
                    .map(|(r, (refs, xi, w))| GaussRule {
                        element_type: r.0,
                        ref_nodes: refs.as_slice(),
                        xi: xi.as_slice(),
                        weights: w.as_slice(),
                    })
                    .collect()
            })
            .collect();
        let cell_values: Vec<CellValues<'_, T>> = self
            .cell_fields
            .iter()
            .zip(&cf_arrays)
            .zip(&rules)
            .map(|((f, (t, v)), rules)| CellValues {
                components: &f.0,
                cell_tags: t.as_slice(),
                values: v.as_slice(),
                layout: if f.3.is_some() {
                    CellLayout::Gauss(rules)
                } else {
                    CellLayout::Cell
                },
            })
            .collect();

        Ok(crate::ops::mesh::from_arrays(
            coords,
            tags.as_slice(),
            xyz.as_slice(),
            &blocks,
            &node_values,
            &cell_values,
            order,
        )?)
    }
}

/// Read a gmsh `.msh` file, one `Mesh` per physical group.
///
/// ASCII or binary, MSH 2.2 or 4.1. All returned meshes share one `Coords`, so a
/// node on the boundary between two named groups is the same node on both sides.
/// Inside a group's mesh there is one submesh per element type; elements with no
/// physical group land under the key `"<ungrouped>"`. Every element type pyrucast
/// knows is read: POI1, SEG2, TRI3, QUA4, TET4, PYRA5, PENTA6, HEX8 and the
/// quadratic SEG3, TRI6, QUA8, QUA9, TET10, PENTA15, HEX20, HEX27.
///
/// # Parameters
///
/// - `coords`: where the nodes are added; its dimension decides how many of
///   gmsh's three coordinates are kept (a 2-D `Coords` keeps `x` and `y`).
/// - `path`: the file to read.
///
/// # Returns
///
/// A `dict` from physical group name to `Mesh`, in the order of the file.
///
/// # Raises
///
/// `RuntimeError` if the file cannot be read or parsed, or for a gmsh element
/// type pyrucast does not know.
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

/// Like `pyrucast.mesh.read_gmsh`, from the `.msh` text held in a string.
///
/// # Parameters
///
/// - `coords`: where the nodes are added.
/// - `text`: the content of a `.msh` file.
///
/// # Returns
///
/// A `dict` from physical group name to `Mesh`, in the order of the text.
///
/// # Raises
///
/// `RuntimeError` if the text cannot be parsed, or for a gmsh element type
/// pyrucast does not know.
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

/// The part of the support of `field` whose values lie in a band, as a `Mesh`.
///
/// Zone by zone: a field at the nodes gives POI1 submeshes of the passing
/// **nodes**; a field at the Gauss points gives submeshes of the passing
/// **cells**, each of its zone's element type, a cell passing only when *all* its
/// Gauss points do. When several components are tested they combine with **AND**:
/// a node or a cell is kept only when *every* tested component is in the band. Its
/// sibling `pyrucast.node_field.mask` rewrites the values instead, keeping the
/// structure.
///
/// # Parameters
///
/// - `field`: a `NodeField`, `SubNodeField`, `ElementField` or `SubElementField`.
/// - `ge`, `gt`: the lower bound, `v ≥ ge` or `v > gt` — at most one of the two.
/// - `le`, `lt`: the upper bound, `v ≤ le` or `v < lt` — at most one of the two.
///   A bound left out leaves that side open; at least one bound is given.
/// - `components`: the components to test, only on the zones carrying **all** of
///   them — a zone missing one is skipped. `None` tests every component of each
///   zone.
///
/// # Returns
///
/// A `Mesh` with one submesh per zone processed.
///
/// # Raises
///
/// - `TypeError` if `field` is none of the four field types.
/// - `RuntimeError` if no bound is given, if both bounds of one side are, or if
///   the lower bound exceeds the upper one.
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
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        select_nodes(f, ge, gt, le, lt, components)
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        select_cells(f, ge, gt, le, lt, components)
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        select_sub_nodes(f, ge, gt, le, lt, components)
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        select_sub_cells(f, ge, gt, le, lt, components)
    } else {
        Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ))
    }
}

// The four flavours of `select`, each documented for its sole receiver. Not
// registered in the module — the free function above remains the polymorphic
// entry point — they give `#[py_op]` the method of each flavour.
// `#[pyfunction]` serves only to make valid the pyo3 signature that `#[py_op]`
// copies.

/// Select the **nodes** of this field passing a value band, zone by zone —
/// a value-range filter returning a `Mesh` with one POI1 submesh per
/// processed zone.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a node is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component of each zone. A `components`
/// list tests **only** those components, and only on the zones carrying
/// **all** of them — a zone missing any listed component is skipped (no
/// submesh). Errors if no bound is given, or the lower one exceeds the upper.
#[py_op(method_on = PyNodeField, name = "select")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn select_nodes(
    field: PyRef<PyNodeField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyMesh {
        inner: crate::ops::mesh::select_nodes(&field.inner, &band, components)?,
    })
}

/// Select the **cells** of this field passing a value band, zone by zone —
/// a value-range filter returning a `Mesh` with one submesh per processed
/// zone, each of its zone's element type. A cell passes only when *all* its
/// Gauss points do.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a cell is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component of each zone. A `components`
/// list tests **only** those components, and only on the zones carrying
/// **all** of them — a zone missing any listed component is skipped (no
/// submesh). Errors if no bound is given, or the lower one exceeds the upper.
#[py_op(method_on = PyElementField, name = "select")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn select_cells(
    field: PyRef<PyElementField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyMesh {
        inner: crate::ops::mesh::select_cells(&field.inner, &band, components)?,
    })
}

/// Select the **nodes** of this sub-field passing a value band — a
/// value-range filter returning a `Mesh` with a single POI1 submesh.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a node is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component. A `components` list tests
/// **only** those components; if the sub-field lacks any of them, nothing is
/// tested and the result is an **empty** `Mesh`. Errors if no bound is given,
/// or the lower one exceeds the upper.
#[py_op(method_on = PySubNodeField, name = "select")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn select_sub_nodes(
    field: PyRef<PySubNodeField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyMesh {
        inner: crate::ops::mesh::select_sub_nodes(&field.handle.read(), &band, components)?,
    })
}

/// Select the **cells** of this sub-field passing a value band — a
/// value-range filter returning a `Mesh` with a single submesh of the
/// sub-field's element type. A cell passes only when *all* its Gauss points
/// do.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a cell is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component. A `components` list tests
/// **only** those components; if the sub-field lacks any of them, nothing is
/// tested and the result is an **empty** `Mesh`. Errors if no bound is given,
/// or the lower one exceeds the upper.
#[py_op(method_on = PySubElementField, name = "select")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn select_sub_cells(
    field: PyRef<PySubElementField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyMesh {
        inner: crate::ops::mesh::select_sub_cells(&field.handle.read(), &band, components)?,
    })
}

// ─── Delegation methods ────────────────────────────────────────────────────
//
// The "subject" face of the operators above (`CONVENTIONS.md`, § "Le verbe
// exposé aussi en méthode"). They now come from `#[py_op]`, placed on the free
// function: the method is derived from its signature, and its documentation
// **copied** rather than pointed at — which is what makes it appear in full
// in `help()` as in the stub the IDEs read.
//
// This block therefore keeps only what breaks the mould. `merge_nodes` does:
// its receiver is a `Py<Self>` and its return a `Py<PyMesh>` — the object
// itself, not a borrowed view. `#[py_op]` emits only a `PyRef` receiver, and
// bending it for a single case would cost more than this very method.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMesh {
    /// Weld together the nodes of `mesh` closer than `tol`.
    ///
    /// The connectivity is redirected to one representative per cluster — the
    /// smallest-id node of the cluster, which keeps its own coordinates (no
    /// averaging). A cluster is a connected component of the relation "closer than
    /// `tol`": the weld propagates from node to node, so on a chain a—b—c where a and
    /// c are further apart than `tol`, all three still end up as one — which is why a
    /// `tol` of the order of the element size collapses a whole region rather than a
    /// seam. Every call prints a one-line tally on stdout once the weld is done —
    /// nodes welded, cells dropped, tolerance used.
    ///
    /// # Parameters
    ///
    /// - `mesh`: the mesh to weld.
    /// - `tol`: the distance, `≥ 0`; `0` welds only coincident nodes.
    /// - `in_place`: `False` builds a new mesh, where cells that collapse onto a
    ///   repeated node (a degenerate segment, triangle, …) are dropped. `True`
    ///   rewrites the connectivity of `mesh`'s **own** submeshes instead and returns
    ///   `mesh` itself. Since the union `mesh_a | mesh_b` shares its submeshes, welding
    ///   that union in place welds `mesh_a` and `mesh_b` themselves: afterwards they
    ///   share their interface nodes. The structure is kept (same submeshes, same
    ///   cells in the same order), so a cell that would collapse is an error, as is a
    ///   submesh already sealed by a finite-element space, a field or a matrix; both
    ///   are checked before anything is written.
    ///
    /// # Returns
    ///
    /// A new `Mesh` mirroring `mesh` (same submeshes, types and colours), or `mesh`
    /// itself with `in_place=True`.
    ///
    /// # Raises
    ///
    /// `RuntimeError` if `tol` is negative, or, with `in_place=True`, for a cell that
    /// would collapse or a sealed submesh.
    #[pyo3(signature = (tol, in_place=false))]
    fn merge_nodes(
        slf: Py<Self>,
        py: Python<'_>,
        tol: f64,
        in_place: bool,
    ) -> PyResult<Py<PyMesh>> {
        super::mesh::merge_nodes(py, slf, tol, in_place)
    }
}
