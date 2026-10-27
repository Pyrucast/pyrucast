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
use pyo3::prelude::*;

/// `model.interface_transfer(side_a, side_b, target, components, tol=1e-9)` —
/// the exchange law `j·n = h(c₁ − c₂)` across an interface between two bodies
/// that do **not** share their nodes, coupling into the model `target`.
///
/// `components` is a list of `(primal, dual)` pairs, each one that `target`
/// assembles: `[("c_H2", "j_H2")]` on a `fick` model for a coating,
/// `[("T", "q")]` on a `heat_conduction` for a contact resistance, the
/// displacement pairs on an `elasticity` for a bonded joint of finite
/// stiffness. The nature is not an argument: it is the one of the sub-model
/// of `target` assembling the pairs, and a pair `target` does not assemble —
/// or pairs of two natures — raise. `h_<primal>` is supplied at assembly time.
///
/// `side_a` and `side_b` are the two facing **boundary** FE spaces, which
/// must be conforming — same element type, same cell count, and local node
/// `k` of a cell facing local node `k` of its counterpart, within `tol`. A
/// non-matching interface raises rather than being projected.
///
/// This is what lets the field **jump** across the interface: with a shared
/// node it could not. The jump is `q/h` for a flux density `q`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (side_a, side_b, target, components, tol = crate::models::interface_transfer::DEFAULT_TOL))]
pub fn interface_transfer(
    side_a: PyRef<PyFiniteElementSpace>,
    side_b: PyRef<PyFiniteElementSpace>,
    target: PyRef<PyModel>,
    components: Vec<(String, String)>,
    tol: f64,
) -> PyResult<PyModel> {
    let inner =
        model::interface_transfer(&side_a.inner, &side_b.inner, &target.inner, components, tol)?;
    Ok(PyModel { inner })
}

/// `model.dirichlet(target, variable, imposed_mesh, sense="=")` — Dirichlet
/// constraint (a single sub-model) holding `variable` on the nodes of
/// `imposed_mesh`, via Lagrange multipliers.
///
/// `target` is the model being constrained: `variable` must be one of its
/// primals, and its dual row is read from it — the `T → q` map is the target's
/// business, not something to retype. `imposed_mesh` is the POI1 mesh of
/// constrained nodes, one relation per node. Each relation gets a multiplier
/// node of its own, minted beside its constrained node (`Model.multiplier_mesh()`
/// gives them back); its names are those of every constraint relation, `lambda`
/// and `imposed`. The imposed value `u_d` is written in the load field at the
/// multiplier node's `imposed` component — `Model.constraint_rhs` does it from
/// the constrained node.
///
/// **One variable, and one only** — compose for a clamp:
/// `dirichlet(m, "u_x", …) | dirichlet(m, "u_y", …)`.
///
/// `sense` (`"="`, the default, `">="` or `"<="`) turns the constraint
/// unilateral (`u ≥ u_d` / `u ≤ u_d`) — such a model is solved with
/// `solve_unilateral`. See the model chapter of the book for the full semantics.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (target, variable, imposed_mesh, sense = RelationSense::Equality))]
pub fn dirichlet(
    target: PyRef<'_, PyModel>,
    variable: &str,
    imposed_mesh: PyRef<'_, PyMesh>,
    sense: RelationSense,
) -> PyResult<PyModel> {
    let inner = model::dirichlet(&target.inner, variable, &imposed_mesh.inner, sense)?;
    Ok(PyModel { inner })
}

/// `model.mpc(target, terms, sense="=")` — multi-point constraint (a single
/// sub-model) imposing, per relation, `Σₖ aₖ·u(nodeₖ, varₖ) = g` via Lagrange
/// multipliers.
///
/// `target` is the model being constrained — the union of two models to tie
/// two bodies. `terms` is a list of `(mesh, variable, coefficient)` tuples: each
/// `mesh` is a POI1 mesh (one node per relation), paired element-for-element
/// with the others (relation `r` = cell `r` of every mesh); each `variable` is
/// a primal of `target`, whose dual row is read from it. Each relation gets a
/// multiplier node of its own, minted beside the node of the first term; `g`
/// is written in the load field at its `imposed` component (default `0`).
/// `sense` (`"="`, the default, `">="` or `"<="`) turns the relations
/// unilateral (`Σ aₖ·uₖ ≥ g` / `≤ g`) — such a model is solved with
/// `solve_unilateral`. See the constraints chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (target, terms, sense = RelationSense::Equality))]
pub fn mpc(
    py: Python<'_>,
    target: PyRef<'_, PyModel>,
    terms: Vec<(Py<PyMesh>, String, f64)>,
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
    let inner = model::mpc(&target.inner, &rust_terms, sense)?;
    Ok(PyModel { inner })
}

/// `model.embedded(target, immersed, host, variables, tol=1e-6)` — embedded
/// (immersed) constraint (a single sub-model) tying each node of `immersed` to
/// the interpolation of `host` at that node, via Lagrange multipliers.
///
/// `immersed` and `host` are meshes sharing one Coords (e.g. a bar embedded in
/// a volume). `target` is the model being constrained; `variables` lists the
/// primals of `target` to tie (e.g. `["u_x", "u_y", "u_z"]`), each dual row
/// read from it. The coupling weights are the host shape functions at each
/// immersed node, computed once at build by locating the node in the host (an
/// immersed node outside the host is an error); `tol` is the location
/// tolerance. Each relation — one per immersed node and variable — gets a
/// multiplier node of its own; the right-hand side defaults to `0` (a rigid
/// tie). See the constraints chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (target, immersed, host, variables, tol = crate::models::embedded::DEFAULT_TOL))]
pub fn embedded(
    target: PyRef<'_, PyModel>,
    immersed: PyRef<'_, PyMesh>,
    host: PyRef<'_, PyMesh>,
    variables: Vec<String>,
    tol: f64,
) -> PyResult<PyModel> {
    let inner = model::embedded(&target.inner, &immersed.inner, &host.inner, variables, tol)?;
    Ok(PyModel { inner })
}

/// `model.contact(target, slave, master, variables)` — node-to-surface contact
/// (a single sub-model): prevent the nodes of `slave` from penetrating the
/// oriented `master` surface mesh, one **unilateral** relation (`≥`) per slave
/// node.
///
/// Each slave node is paired at build with its closest master facet
/// (projection weights, facet normal, initial signed gap); the pairing is
/// then fixed (linearised, frictionless contact). The master surface must
/// be consistently oriented with its normal pointing toward the slave
/// body. `target` is the model being constrained; `variables` lists its
/// displacement primals, one per space dimension in ambient order (e.g.
/// `["u_x", "u_y"]`), each dual row read from it. Each relation gets a
/// multiplier node of its own, minted beside its slave node.
///
/// Solve with `solve_unilateral`; build the initial-gap right-hand side
/// with `Model.contact_gaps()`. See the contact chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (target, slave, master, variables))]
pub fn contact(
    target: PyRef<'_, PyModel>,
    slave: PyRef<'_, PyMesh>,
    master: PyRef<'_, PyMesh>,
    variables: Vec<String>,
) -> PyResult<PyModel> {
    let inner = model::contact(&target.inner, &slave.inner, &master.inner, variables)?;
    Ok(PyModel { inner })
}
