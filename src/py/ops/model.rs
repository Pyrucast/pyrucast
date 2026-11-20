//! Python wrappers for [`crate::ops::model`] — the physics declarations,
//! which produce a `Model`.
//!
//! Each one spans the **whole** support it is given (one zone per subspace);
//! heterogeneous physics compose with `|`.

use crate::models::RelationSense;
use crate::ops::model;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// Exchange `j·n = h(c₁ − c₂)` across an interface whose two sides share no node.
///
/// This is what lets the field **jump** across the interface: with a shared node
/// it could not. The jump is `q/h` for a flux density `q`. The coefficient
/// `h_<variable>` is supplied at assembly time.
///
/// # Parameters
///
/// - `side_a`, `side_b`: the two facing **boundary** FE spaces. They must be
///   conforming — same element type, same cell count, and local node `k` of a
///   cell facing local node `k` of its counterpart, within `tol`. A non-matching
///   interface raises rather than being projected.
/// - `target`: the model it couples into; the dual row of each variable and the
///   nature are read from it.
/// - `variables`: the exchanged unknowns of `target`, each coupling into the
///   dual row `target` pairs it with: `["c_H2"]` on a `fick` model for a
///   coating, `["T"]` on a `heat_conduction` for a contact resistance, the
///   displacements on an `elasticity` for a bonded joint of finite stiffness.
/// - `tol`: the matching tolerance, a **distance** in the length unit of the
///   mesh.
///
/// # Returns
///
/// A `Model` coupling the two sides.
///
/// # Raises
///
/// `RuntimeError` if the two sides do not match, for a variable `target` does
/// not declare, or for variables of two natures.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (side_a, side_b, target, variables, tol = crate::models::interface_transfer::DEFAULT_TOL))]
pub fn interface_transfer(
    side_a: PyRef<PyFiniteElementSpace>,
    side_b: PyRef<PyFiniteElementSpace>,
    target: PyRef<PyModel>,
    variables: Vec<String>,
    tol: f64,
) -> PyResult<PyModel> {
    let inner =
        model::interface_transfer(&side_a.inner, &side_b.inner, &target.inner, variables, tol)?;
    Ok(PyModel { inner })
}

/// Hold `variable` at imposed values on the nodes of `imposed_mesh`.
///
/// A Dirichlet constraint, a single sub-model, through Lagrange multipliers. Each
/// relation, one per node, gets a multiplier node of its own, minted beside its
/// constrained node (`Model.multiplier_mesh()` gives them back), with the names
/// of every constraint relation, `lambda` and `imposed`. The imposed value `u_d`
/// is written in the load field at the multiplier node's `imposed` component —
/// `Model.constraint_rhs` does it from the constrained node. One variable, and
/// one only — compose for a clamp:
/// `dirichlet(nodes, m, "u_x") | dirichlet(nodes, m, "u_y")`.
///
/// # Parameters
///
/// - `imposed_mesh`: the mesh of the constrained nodes, of any element type —
///   an edge, a face, a selection of points. Its distinct nodes, all zones
///   together, are held, each once: a node that two zones share gets one
///   relation. A single zone is held through its POI1 companion,
///   `SubMesh.to_poi1()`, so the solution on that support pairs node by node
///   with the fields on that zone; several zones are fused into a new POI1 zone.
/// - `target`: the model being constrained; the dual row of `variable` is read
///   from it.
/// - `variable`: the unknown of `target` to hold, one of its primals.
/// - `sense`: `"="` (the default), `">="` or `"<="`. The last two make the
///   constraint unilateral (`u ≥ u_d`, `u ≤ u_d`): the matrix carries the sense,
///   and `pyrucast.solver.solve` runs its active-set loop.
///
/// # Returns
///
/// A `Model` of one constraint sub-model.
///
/// # Raises
///
/// - `ValueError` for an unknown `sense`.
/// - `RuntimeError` if `imposed_mesh` holds no node, or if `variable` is not a
///   primal of `target`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (imposed_mesh, target, variable, sense = RelationSense::Equality))]
pub fn dirichlet(
    imposed_mesh: PyRef<'_, PyMesh>,
    target: PyRef<'_, PyModel>,
    variable: &str,
    sense: RelationSense,
) -> PyResult<PyModel> {
    let inner = model::dirichlet(&imposed_mesh.inner, &target.inner, variable, sense)?;
    Ok(PyModel { inner })
}

/// Impose a linear relation `Σₖ aₖ·u(nodeₖ, varₖ) = g` between nodes, per relation.
///
/// A multi-point constraint, a single sub-model, through Lagrange multipliers.
/// Each relation gets a multiplier node of its own, minted beside the node of the
/// first term; `g` is written in the load field at its `imposed` component
/// (default `0`).
///
/// # Parameters
///
/// - `terms`: a list of `(mesh, variable, coefficient)` tuples. Each `mesh` is a
///   POI1 mesh, one node per relation, paired cell for cell with the others
///   (relation `r` = cell `r` of every mesh); each `variable` is a primal of
///   `target`.
/// - `target`: the model being constrained — the union of two models to tie two
///   bodies —; the dual rows are read from it.
/// - `sense`: `"="` (the default), `">="` or `"<="`. The last two make the
///   relations unilateral (`Σ aₖ·uₖ ≥ g`, `≤ g`): the matrix carries the sense,
///   and `pyrucast.solver.solve` runs its active-set loop.
///
/// # Returns
///
/// A `Model` of one constraint sub-model.
///
/// # Raises
///
/// - `ValueError` for an unknown `sense`.
/// - `RuntimeError` if the meshes of `terms` do not hold the same number of
///   nodes, or if a variable is not a primal of `target`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (terms, target, sense = RelationSense::Equality))]
pub fn mpc(
    py: Python<'_>,
    terms: Vec<(Py<PyMesh>, String, f64)>,
    target: PyRef<'_, PyModel>,
    sense: RelationSense,
) -> PyResult<PyModel> {
    // The borrows first, then the views on them: the tuples the operator reads
    // hold references into the borrowed meshes.
    let meshes: Vec<PyRef<'_, PyMesh>> = terms.iter().map(|(m, _, _)| m.borrow(py)).collect();
    let rust_terms: Vec<_> = meshes
        .iter()
        .zip(&terms)
        .map(|(mesh, (_, variable, coefficient))| (&mesh.inner, variable.as_str(), *coefficient))
        .collect();
    let inner = model::mpc(&rust_terms, &target.inner, sense)?;
    Ok(PyModel { inner })
}

/// Tie each node of `immersed` to the interpolation of `host` at that node.
///
/// An embedded constraint, a single sub-model, through Lagrange multipliers — a
/// bar embedded in a volume, for instance. The coupling weights are the host
/// shape functions at each immersed node, computed once at build by locating the
/// node in the host. Each relation — one per immersed node and variable — gets a
/// multiplier node of its own; the right-hand side defaults to `0`, a rigid tie.
///
/// # Parameters
///
/// - `immersed`: the immersed nodes' mesh, on the same `Coords` as `host`.
/// - `host`: the mesh they are immersed in.
/// - `target`: the model being constrained; the dual rows are read from it.
/// - `variables`: the primals of `target` to tie, `["u_x", "u_y", "u_z"]` for
///   instance.
/// - `tol`: the location tolerance, a gap in the host's **reference** element,
///   without dimension.
///
/// # Returns
///
/// A `Model` of one constraint sub-model.
///
/// # Raises
///
/// `RuntimeError` if an immersed node lies outside the host, or if a variable is
/// not a primal of `target`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (immersed, host, target, variables, tol = crate::models::embedded::DEFAULT_TOL))]
pub fn embedded(
    immersed: PyRef<'_, PyMesh>,
    host: PyRef<'_, PyMesh>,
    target: PyRef<'_, PyModel>,
    variables: Vec<String>,
    tol: f64,
) -> PyResult<PyModel> {
    let inner = model::embedded(&immersed.inner, &host.inner, &target.inner, variables, tol)?;
    Ok(PyModel { inner })
}

/// Node-to-surface contact: keep the nodes of `slave` out of the `master` surface.
///
/// One **unilateral** relation (`≥`) per slave node, a single sub-model. Each
/// slave node is paired at build with its closest master facet (projection
/// weights, facet normal, initial signed gap); the pairing is then fixed
/// (linearized, frictionless contact) until `pyrucast.model.update_contact`
/// pairs it again. Each relation gets a multiplier node of its own, minted
/// beside its slave node. The initial-gap right-hand side `-g0` is a load the
/// contact carries: `pyrucast.node_field.external_forces` returns it with the
/// other loads of the model. Solve with `pyrucast.solver.solve`, which runs its
/// active-set loop.
///
/// # Parameters
///
/// - `slave`: the nodes that must not penetrate.
/// - `master`: the surface mesh, consistently oriented, its normal pointing
///   toward the slave body.
/// - `target`: the model being constrained; the dual rows are read from it.
/// - `variables`: the displacement primals of `target`, one per spatial axis in
///   ambient order (`["u_x", "u_y"]`).
///
/// # Returns
///
/// A `Model` of one constraint sub-model.
///
/// # Raises
///
/// `RuntimeError` if a variable is not a primal of `target`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (slave, master, target, variables))]
pub fn contact(
    slave: PyRef<'_, PyMesh>,
    master: PyRef<'_, PyMesh>,
    target: PyRef<'_, PyModel>,
    variables: Vec<String>,
) -> PyResult<PyModel> {
    let inner = model::contact(&slave.inner, &master.inner, &target.inner, variables)?;
    Ok(PyModel { inner })
}

/// The model with its contacts paired again on the positions `X + u`.
///
/// Each slave node is paired again with its closest master facet on `X + u`, and
/// its relation becomes the gap linearized there, written in total displacement
/// from `X`: the right-hand side the contact carries is `-g~`, the gap of the new
/// pairing measured on `X`. The multiplier nodes are kept, so the unknowns
/// `lambda` and the rows `imposed` stay the same; at `u = 0`, the contact as it
/// was built. What large displacements call for: the step-by-step loop updates
/// its contacts each time it rebuilds its operator.
///
/// # Parameters
///
/// - `model`: the model whose contacts to update. Every other sub-model is
///   shared as it is; a model without contact comes back unchanged.
/// - `displacement`: `u`, read on each contact's variables (`u_x`, `u_y`,
///   `u_z`); a node or a component it lacks reads zero. `X` is the active
///   configuration.
///
/// # Returns
///
/// A new `Model`.
///
/// # Notes
///
/// A master facet collapsed by the displacement is not detected.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn update_contact(model: PyRef<'_, PyModel>, displacement: PyRef<'_, PyNodeField>) -> PyModel {
    PyModel {
        inner: model::update_contact(&model.inner, &displacement.inner),
    }
}
