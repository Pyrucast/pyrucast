//! Python wrappers for [`crate::ops::node_field`] — the operators that
//! produce a `NodeField`.

use crate::aggregate::Aggregate;
use crate::handle::Handle;
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::mesh::PyMesh;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use crate::py::node_field::PySubNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// The positions of the nodes of `mesh`, as a field at the nodes.
///
/// The bridge from geometry to fields: the result can be plotted, derived, or
/// written back after a computation with `pyrucast.coords.set`. Positions are
/// read in the active configuration of the `Coords`.
///
/// # Parameters
///
/// - `mesh`: the nodes to read.
/// - `components`: the axes to read, among `"X"`, `"Y"`, `"Z"`, one component
///   each; `None` reads every axis of the `Coords` (`["X"]` in 1-D, `["X", "Y"]`
///   in 2-D, `["X", "Y", "Z"]` in 3-D).
///
/// # Returns
///
/// A `NodeField` with one zone per submesh of `mesh`, on the distinct nodes of
/// that submesh.
///
/// # Raises
///
/// `RuntimeError` for a component other than `"X"`, `"Y"`, `"Z"`, or an axis
/// beyond the dimension of the `Coords`.
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn positions(mesh: PyRef<PyMesh>, components: Option<Vec<String>>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::positions(&mesh.inner, components)?,
    })
}

/// Restrict `field` to the nodes of `mesh`.
///
/// Each zone of the result lies on the POI1 cloud of a submesh of `mesh` — its
/// distinct nodes, built once and kept with the submesh. Two restrictions onto
/// the **same** `mesh` share that support, so they combine directly:
/// `restrict(a, mesh) - restrict(b, mesh)` is the node-by-node difference.
/// That support is also the one a stiffness block over `mesh` uses, so
/// `K * restrict(f, mesh)` and `solve(K, f) - restrict(g, mesh)` line up too.
///
/// Element operations on the region (`gradient`, `integral`, `deformation`,
/// `interp_to_gauss`) take `mesh` as a separate argument and read the field by
/// node. To land on the support of an existing field rather than a mesh, use
/// `pyrucast.node_field.restrict_like`.
///
/// # Parameters
///
/// - `field`: the field to restrict.
/// - `mesh`: the region, on the same `Coords` as `field`.
///
/// # Returns
///
/// A new `NodeField` with one zone per submesh of `mesh`, each carrying the
/// union of the components of `field`. A node of `mesh` the field does not
/// define gets `0.0`; the nodes outside `mesh` are dropped.
///
/// # Raises
///
/// `RuntimeError` if `mesh` and `field` are attached to different `Coords`.
#[py_op(method_on = PyNodeField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::restrict(&field.inner, &mesh.inner)?,
    })
}

/// Read `field` at the nodes of `target`, through the cells of `mesh`.
///
/// A node of `target` inside a cell of `mesh` takes the value the cell gives
/// there:
///
/// - **a field at the nodes** (`NodeField`, `SubNodeField`): `Σ Nᵢ(ξ) f(nodeᵢ)`,
///   the shape functions of that cell at the node (a quadratic cell
///   interpolates quadratically). The field is read first zone first, a node
///   of `mesh` it does not define reading zero.
/// - **a field at the Gauss points** (`ElementField`, `SubElementField`): the
///   value that cell's **own** Gauss points give there — their least-squares
///   fit on the cell, the one a plot colours the cell with. Nothing is
///   averaged between cells: a line of points crossing from one cell to its
///   neighbour shows the jump between them. A point on a face shared by two
///   cells is read in one of them; to follow a field along a line, put the
///   points between the nodes.
///
/// A node of `target` outside `mesh` gets zero, as a field at the nodes reads a
/// node it does not define: a patch projects onto a larger mesh. The nodes of
/// `target` are located once.
///
/// # Parameters
///
/// - `field`: the field to read. Every component is carried.
/// - `mesh`: the cells that interpolate. For a field at the nodes, on the same
///   `Coords` as `field`; for a field at the Gauss points, the field's own
///   cells, each submesh carrying a zone of the field.
/// - `target`: the nodes to read at. It may live in another `Coords` than
///   `mesh` (a remeshed part), each read in its active configuration.
/// - `tol`: how far outside a cell's reference element, without unit, a node
///   still counts as inside.
///
/// # Returns
///
/// A field at the nodes with one zone per submesh of `target`, like
/// `pyrucast.node_field.positions`. An empty `field` or `target` gives an empty
/// field; an empty `mesh` gives zero at every node of `target`.
///
/// # Raises
///
/// - `TypeError` if `field` is none of the four field types.
/// - `RuntimeError` if `mesh` and `target` differ in dimension, if a field at
///   the nodes and `mesh` hang from different `Coords`, or if a submesh of
///   `mesh` carries no zone of a field at the Gauss points.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, mesh, target, tol=1e-6))]
pub fn project(
    field: &Bound<'_, PyAny>,
    mesh: PyRef<PyMesh>,
    target: PyRef<PyMesh>,
    tol: f64,
) -> PyResult<PyNodeField> {
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        project_node(f, mesh, target, tol)
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        project_element(f, mesh, target, tol)
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        project_sub_node(f, mesh, target, tol)
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        project_sub_element(f, mesh, target, tol)
    } else {
        Err(PyTypeError::new_err(
            "project: expected a NodeField, SubNodeField, ElementField or SubElementField",
        ))
    }
}

// The four flavours of `project`, each documented for its sole receiver. Not
// registered in the module — the free function above remains the polymorphic
// entry point — they give `#[py_op]` the method of each flavour.
// `#[pyfunction]` serves only to make valid the pyo3 signature that `#[py_op]`
// copies.

/// This nodal field interpolated by the cells of `mesh` at the nodes of
/// `target`: a node of `target` inside a cell of `mesh` takes `Σ Nᵢ(ξ)
/// f(nodeᵢ)`, the shape functions of that cell at the node (quadratic cells
/// interpolate quadratically); every component is carried. A node of `target`
/// outside `mesh` — farther than `tol` outside every cell's reference element
/// — gets zero. Read first zone first, a node of `mesh` the field does not
/// define reading zero. `mesh` and `target` may live in different `Coords`.
/// One zone per submesh of `target`. Errors: the field and `mesh` on
/// different `Coords`; `mesh` and `target` in different dimensions.
#[py_op(method_on = PyNodeField, name = "project")]
#[pyfunction]
#[pyo3(signature = (field, mesh, target, tol=1e-6))]
pub fn project_node(
    field: PyRef<PyNodeField>,
    mesh: PyRef<PyMesh>,
    target: PyRef<PyMesh>,
    tol: f64,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::project_node(&field.inner, &mesh.inner, &target.inner, tol)?,
    })
}

/// This zone interpolated by the cells of `mesh` at the nodes of `target` —
/// as `NodeField.project`, for a field of this one zone.
#[py_op(method_on = PySubNodeField, name = "project")]
#[pyfunction]
#[pyo3(signature = (field, mesh, target, tol=1e-6))]
pub fn project_sub_node(
    field: PyRef<PySubNodeField>,
    mesh: PyRef<PyMesh>,
    target: PyRef<PyMesh>,
    tol: f64,
) -> PyResult<PyNodeField> {
    let mut whole = crate::containers::node_field::NodeField::default();
    whole.add_sub(field.handle.clone())?;
    Ok(PyNodeField {
        inner: crate::ops::node_field::project_node(&whole, &mesh.inner, &target.inner, tol)?,
    })
}

/// This field at the Gauss points read at the nodes of `target`: a node of
/// `target` inside a cell of `mesh` takes the value that cell's own Gauss
/// points give there — their least-squares fit on the cell, the one a plot
/// colours the cell with. Nothing is averaged between cells, so a line of
/// points crossing from one cell to its neighbour shows the jump between
/// them; a point on a shared face is read in one of the two. `mesh` is the
/// field's own cells (each submesh must carry a zone of it). A node outside
/// `mesh` gets zero. One zone per submesh of `target`. Errors: a submesh of
/// `mesh` without a zone of the field; `mesh` and `target` in different
/// dimensions.
#[py_op(method_on = PyElementField, name = "project")]
#[pyfunction]
#[pyo3(signature = (field, mesh, target, tol=1e-6))]
pub fn project_element(
    field: PyRef<PyElementField>,
    mesh: PyRef<PyMesh>,
    target: PyRef<PyMesh>,
    tol: f64,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::project_element(
            &field.inner,
            &mesh.inner,
            &target.inner,
            tol,
        )?,
    })
}

/// This zone at the Gauss points read at the nodes of `target` — as
/// `ElementField.project`, for a field of this one zone.
#[py_op(method_on = PySubElementField, name = "project")]
#[pyfunction]
#[pyo3(signature = (field, mesh, target, tol=1e-6))]
pub fn project_sub_element(
    field: PyRef<PySubElementField>,
    mesh: PyRef<PyMesh>,
    target: PyRef<PyMesh>,
    tol: f64,
) -> PyResult<PyNodeField> {
    let mut whole = crate::containers::element_field::ElementField::empty();
    whole.add_sub(field.handle.clone())?;
    Ok(PyNodeField {
        inner: crate::ops::node_field::project_element(&whole, &mesh.inner, &target.inner, tol)?,
    })
}

/// Read `field` on the support and the components of `target`, zone by zone.
///
/// Unlike `pyrucast.node_field.restrict`, which lands on a support built from
/// a mesh and carries every component of `field`, this reuses each zone of
/// `target` as it is — same support, same components. The result combines with
/// `target` directly by the arithmetic operators: typically, to fold an
/// increment from `pyrucast.solver.solve`, which also carries the
/// multipliers, into a current solution (`target + restrict_like(du, target)`).
///
/// # Parameters
///
/// - `field`: the field to read.
/// - `target`: the field whose zones, supports and components the result
///   takes, on the same `Coords` as `field`.
///
/// # Returns
///
/// A new `NodeField` zone for zone like `target`. A `(node, component)` pair
/// takes the value of `field` when it defines it, `0.0` otherwise; the nodes
/// and components of `field` absent from `target` are dropped.
///
/// # Raises
///
/// `RuntimeError` if `target` and `field` are attached to different `Coords`.
#[py_op(method_on = PyNodeField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict_like(
    field: PyRef<PyNodeField>,
    target: PyRef<PyNodeField>,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::restrict_like(&field.inner, &target.inner)?,
    })
}

/// The union `a | b` of two fields at the nodes, consolidated.
///
/// The zones of both fields: those on the same support fused into one carrying
/// the union of their components, those on distinct supports side by side
/// (nothing is densified).
///
/// # Parameters
///
/// - `a`: the first field.
/// - `b`: the second field, on the same `Coords` as `a`.
///
/// # Returns
///
/// A new `NodeField`; `a` and `b` are left untouched.
///
/// # Raises
///
/// `RuntimeError` if `a` and `b` hold different values at the same
/// `(node, component)` pair, or are attached to different `Coords`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge(a: PyRef<PyNodeField>, b: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::merge(&a.inner, &b.inner)?,
    })
}

/// Fuse the zones of `field` that lie on the same support into one.
///
/// The zones on one support become a single zone carrying the union of their
/// components; zones on distinct supports stay side by side. The union `a | b`
/// and `pyrucast.node_field.merge` end with this step.
///
/// # Parameters
///
/// - `field`: the field to consolidate.
///
/// # Returns
///
/// A new `NodeField`; `field` itself is left untouched.
///
/// # Raises
///
/// `RuntimeError` if two zones disagree on a value at a shared
/// `(node, component)` pair.
#[py_op(method_on = PyNodeField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_node")]
pub fn consolidate(field: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::consolidate(&field.inner)?,
    })
}

/// Weak divergence `d_i = ∫ ∇N_i · A dΩ` of a quantity at the Gauss points.
///
/// The adjoint of `pyrucast.element_field.gradient`, accumulated per node:
/// integrated quantities, not the pointwise values of `∇·A`. The tensor case
/// is the internal forces of a continuum when the quantity is a Cauchy stress,
/// but the operator knows nothing of mechanics: it needs only the geometry and
/// the names. A node shared by several subspaces receives the sum of their
/// contributions.
///
/// # Parameters
///
/// - `field`: the quantity, at the Gauss points.
/// - `prefix`: the name of the quantity, `A`. Its rank is read off the names
///   the field carries: `A_x, A_y` is a vector; `A_xx, A_xy, A_yy` a symmetric
///   tensor, read whole (non-symmetric) when it also carries `A_yx` — as the
///   nominal stress of large strains. On an axisymmetric space the tensor also
///   carries its hoop term `A_zz`.
///
/// # Returns
///
/// A `NodeField`: a single `div_A` component for a vector, one component per
/// axis for a tensor (`div_A_x`, `div_A_y`, …).
///
/// # Raises
///
/// `RuntimeError` if `field` carries no `prefix` of either rank, or both.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn divergence(field: PyRef<PyElementField>, prefix: &str) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::divergence(&field.inner, prefix)?;
    Ok(PyNodeField { inner: nf })
}

/// Gather a field of one point per cell at the nodes, by summation.
///
/// Each node receives, per component, the sum of the values of the cells it
/// belongs to — across zones too. It is not a smoothing: no average, no weight.
/// Applied to the cell measures of `pyrucast.element_field.cell_integral`, it
/// gives the volume each node stands for.
///
/// # Parameters
///
/// - `field`: an `ElementField` with one point per cell, built on a space with
///   the `REDUCED` quadrature.
///
/// # Returns
///
/// A `NodeField` with the components of `field`.
///
/// # Raises
///
/// `RuntimeError` if a zone of `field` has more than one point per cell.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sum_to_nodes(field: PyRef<PyElementField>) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::sum_to_nodes(&field.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Internal nodal forces of `model`, the left side of `Σ f_int = Σ f_ext`.
///
/// The nodal mirror of `pyrucast.matrix.stiffness`: every sub-model is asked
/// for its term of the residual on the internal side. A sub-model carrying a
/// behaviour applies its own `Bᵀ` (continuum solid, bar or beam), which is why
/// the operator takes a model and not a mere FE space; a sub-model with no term
/// on this side declares none and does not appear in the result.
///
/// For a linear law the result equals the assembled stiffness applied to the
/// solution (`K·u`); a non-linear law gives the exact internal forces. A law in
/// large strains is served on the reference configuration by the nominal
/// stress `P = F·S`, read whole (`P_xx, P_xy, P_yx, P_yy`):
/// `f = ∫ P·∇₀N dV₀`. Its counterpart is `pyrucast.node_field.external_forces`,
/// and the gap between the two sums is the residual.
///
/// # Parameters
///
/// - `model`: the physics and the constraints.
/// - `state`: the material state at the Gauss points, as returned by
///   `pyrucast.element_field.integrate_behavior`. Each zone is picked by its FE
///   subspace.
/// - `solution`: the current solution, multipliers included. A term linear in
///   `u` reads it directly — a boundary transfer's `∫ h·a·N` has no law to go
///   through —, and a constraint draws its force `Cᵀλ` from it, spread over the
///   constrained nodes by the relation's coefficients. That force is the
///   opposite of the **reaction**: the reaction of a model's constraints is
///   `−internal_forces(model.filter("constraint"), …)`, whichever way
///   `pyrucast.solver.solve` treated the relations.
/// - `materials`: the material data, zone by zone, as for
///   `pyrucast.matrix.stiffness`.
///
/// # Returns
///
/// A `NodeField` whose components are the dual variables of each sub-model
/// (`f_x`, … for a solid or a bar; `f_w`, `m_theta` for a beam).
///
/// # Raises
///
/// `RuntimeError` if `state` or `materials` has no zone on the FE subspace of a
/// sub-model that reads it, or lacks a component it reads.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn internal_forces(
    model: PyRef<PyModel>,
    state: PyRef<PyElementField>,
    solution: PyRef<PyNodeField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::internal_forces(
        &model.inner,
        &state.inner,
        &solution.inner,
        &materials.inner,
    )?;
    Ok(PyNodeField { inner: nf })
}

/// External nodal forces of `model`, the right side of `Σ f_int = Σ f_ext`.
///
/// The counterpart of `pyrucast.node_field.internal_forces`. Every sub-model is
/// asked for its terms of the residual on the external side: the given data of
/// its weak form, on the right of the equals sign — the ambient of a boundary
/// transfer, a distributed flux. A physics whose term is entirely a response to
/// `u` (elasticity, conduction, a bar) has none. The terms loading one node are
/// added.
///
/// # Parameters
///
/// - `model`: the physics whose given data to integrate.
/// - `materials`: the material data, zone by zone, as for
///   `pyrucast.matrix.stiffness`; it carries the densities the terms read.
///
/// # Returns
///
/// A `NodeField` with a single zone over every loaded node. A model made only
/// of physics without such a term yields an empty field — the right answer,
/// not a failure.
///
/// # Raises
///
/// `RuntimeError` if `materials` has no zone on the FE subspace of a sub-model
/// that reads it, or lacks a component it reads.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn external_forces(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::external_forces(&model.inner, &materials.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// A 0/1 mask of `field` against a band of values, component by component.
///
/// The result has the structure of the input — same zones, same support, same
/// components —, only the values are rewritten: `1.0` inside the band, `0.0`
/// outside. It therefore multiplies term by term with the input. There is
/// **no** AND across components: each value stands on its own. The
/// comparisons `>=`, `>`, `<=`, `<` of a field against a number build the same
/// mask. Its sibling `pyrucast.mesh.select` extracts the passing nodes
/// instead, as a `Mesh`.
///
/// # Parameters
///
/// - `field`: a `NodeField` or a `SubNodeField`.
/// - `ge`, `gt`: the lower bound, `v ≥ ge` or `v > gt` — at most one of the two.
/// - `le`, `lt`: the upper bound, `v ≤ le` or `v < lt` — at most one of the two.
///   A bound left out leaves that side open; at least one bound is given.
/// - `components`: the components to test, the others left at `1.0` (neutral
///   for the product); a zone missing a listed component is left all `1.0`.
///   `None` tests every component.
///
/// # Returns
///
/// A field of the type of `field`, with the same structure.
///
/// # Raises
///
/// - `TypeError` if `field` is neither a `NodeField` nor a `SubNodeField`.
/// - `RuntimeError` if no bound is given, if both bounds of one side are, or
///   if the lower bound exceeds the upper one.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "mask_node")]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
#[allow(clippy::too_many_arguments)]
pub fn mask(
    py: Python<'_>,
    field: &Bound<'_, PyAny>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        Ok(Py::new(py, mask_node_field(f, ge, gt, le, lt, components)?)?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        Ok(Py::new(py, mask_sub_node_field(f, ge, gt, le, lt, components)?)?.into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a PyNodeField or PySubNodeField",
        ))
    }
}

// The two flavours of `mask`, each at its precise type. They are not registered
// in the module — the free function remains the only polymorphic entry point —
// but `#[py_op]` derives the receiver's method from them, carrying **their**
// documentation: that of one flavour, which need not mention the other.
// `#[pyfunction]` is there only to make `#[pyo3(signature = …)]` valid, which
// `#[py_op]` copies onto the method.

/// Per-component 0/1 **mask** of this field against a value band — same
/// structure as the field: same zones, same support, same components, only the
/// values are rewritten (`1.0` inside the band, `0.0` outside). The result is
/// therefore multipliable term by term with the field.
///
/// Its sibling `select` extracts the passing *nodes* instead, and produces a
/// `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product), and a zone missing a listed component is left all-`1.0`.
#[py_op(method_on = PyNodeField, name = "mask")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn mask_node_field(
    field: PyRef<PyNodeField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyNodeField> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyNodeField {
        inner: crate::ops::node_field::mask(&field.inner, &band, components)?,
    })
}

/// Per-component 0/1 **mask** of this sub-field against a value band — same
/// structure as the sub-field: same support, same components, only the values
/// are rewritten (`1.0` inside the band, `0.0` outside). The result is
/// therefore multipliable term by term with the sub-field.
///
/// Its sibling `select` extracts the passing *nodes* instead, and produces a
/// `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product). A listed component the sub-field does not carry leaves it
/// all-`1.0`.
#[py_op(method_on = PySubNodeField, name = "mask")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn mask_sub_node_field(
    field: PyRef<PySubNodeField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PySubNodeField> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    let out = crate::ops::node_field::mask_sub(&field.handle.read(), &band, components);
    Ok(PySubNodeField {
        handle: Handle::new(out),
    })
}
