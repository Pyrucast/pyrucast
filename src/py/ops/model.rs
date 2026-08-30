//! Python wrappers for [`crate::ops::model`] — the physics declarations,
//! which produce a `Model`.
//!
//! Each one spans the **whole** support it is given (one zone per subspace);
//! heterogeneous physics compose with `|`.

use crate::models::mpc::MpcTerm;
use crate::models::symmetry::MaterialSymmetry;
use crate::models::tensor::Kinematics;
use crate::models::{Physics, RelationSense};
use crate::ops::model;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::model::PyModel;
use pyo3::prelude::*;

/// `model.heat_conduction(fespace, symmetry=None)` — heat-conduction model
/// spanning **every** subspace of `fespace` (one zone per subspace). A
/// single-subspace space gives the unit case; several give one zone
/// each. Compose heterogeneous physics with `|`:
/// `model.heat_conduction(fes) | model.dirichlet(...)`.
///
/// `symmetry` is `"isotropic"` (the default), `"orthotropic"` or
/// `"anisotropic"`, and selects which conductivity the material field must
/// carry: the scalar `k`, the principal `k_1, k_2, k_3`, or the symmetric
/// tensor `k_11 … k_33`. The two oriented ones also require the material
/// axes — `V1X, V1Y` in 2-D, `V1X…V1Z, V2X…V2Z` in 3-D.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (fespace, symmetry=None))]
pub fn heat_conduction(
    fespace: PyRef<PyFiniteElementSpace>,
    symmetry: Option<MaterialSymmetry>,
) -> PyResult<PyModel> {
    let s = symmetry.unwrap_or(MaterialSymmetry::Isotropic);
    let inner = model::heat_conduction_with_symmetry(&fespace.inner, s)?;
    Ok(PyModel { inner })
}

/// `model.fick(fespace, species, symmetry=None)` — Fickian diffusion of one
/// named **species**, spanning every subspace of `fespace`.
///
/// Every name carries the species: DOFs are `c_<species>` (primal) and
/// `j_<species>` (dual), the diffusivity is `D_<species>`, the reported flux
/// `j_<species>_x…`. Two species therefore share a mesh without colliding —
/// and no bare `c` can be mistaken for anything else. Its physics nature is
/// `"diffusion"`, so `model.filter("diffusion")` isolates it from a thermal
/// or mechanical model it is composed with.
///
/// `symmetry` is `"isotropic"` (the default), `"orthotropic"` or
/// `"anisotropic"`, selecting the diffusivity the material field must carry:
/// `D_<species>`, `D_1_<species> …`, or the symmetric `D_11_<species> …` —
/// the oriented ones plus the material axes `V1X, V1Y, …`. Those axes and
/// the optional storage `poro` keep **bare** names: they belong to the
/// medium, not to what diffuses through it.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (fespace, species, symmetry=None))]
pub fn fick(
    fespace: PyRef<PyFiniteElementSpace>,
    species: &str,
    symmetry: Option<MaterialSymmetry>,
) -> PyResult<PyModel> {
    let s = symmetry.unwrap_or(MaterialSymmetry::Isotropic);
    let inner = model::fick_with_symmetry(&fespace.inner, s, species)?;
    Ok(PyModel { inner })
}

/// `model.interface_transfer(side_a, side_b, kind=None, tol=None)` — the
/// exchange law `j·n = h(c₁ − c₂)` across an interface between two bodies
/// that do **not** share their nodes. `kind` is `"mass"` (the default:
/// concentration `c`, flux `j`, nature `"diffusion"`) or `"thermal"` (a
/// contact resistance: `T`, `q`, nature `"thermal"`); `h` is supplied at
/// assembly time.
///
/// `side_a` and `side_b` are the two facing **boundary** FE spaces, which
/// must be conforming — same element type, same cell count, and local node
/// `k` of a cell facing local node `k` of its counterpart, within `tol`
/// (default `1e-9`). A non-matching interface raises rather than being
/// projected.
///
/// This is what lets the field **jump** across the interface: with a shared
/// node it could not. The jump is `q/h` for a flux density `q`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (side_a, side_b, components, physics, tol=None))]
pub fn interface_transfer(
    side_a: PyRef<PyFiniteElementSpace>,
    side_b: PyRef<PyFiniteElementSpace>,
    components: Vec<(String, String)>,
    physics: Physics,
    tol: Option<f64>,
) -> PyResult<PyModel> {
    let inner = model::interface_transfer(
        &side_a.inner,
        &side_b.inner,
        components,
        physics,
        tol.unwrap_or(1e-9),
    )?;
    Ok(PyModel { inner })
}

/// `model.elasticity(fespace, kinematics, symmetry=None)` — linear-elasticity
/// model spanning every subspace of `fespace`. `kinematics` is `"plane_stress"`,
/// `"plane_strain"` or `"axisymmetric"` (2-D), or `"full_3d"` (3-D). DOFs are
/// the vector displacement `u_x, u_y(, u_z)`; material is supplied at
/// assembly time.
///
/// `"axisymmetric"` requires a geometry built with `Coords.axisymmetric()`
/// (`x = r`, `y = z`): the hoop strain `ε_θθ = u_r / r` comes from the kinematics,
/// the `2πr` integration measure from the geometry, and the two must agree.
///
/// `symmetry` is the **material** axis, independent of the kinematic one:
/// `"isotropic"` (the default, material `E`, `nu`), `"orthotropic"`
/// (`E_1, E_2, E_3, nu_12, nu_13, nu_23, G_12, G_13, G_23`) or
/// `"anisotropic"` (the 21 constants `C_11 … C_66`). The two oriented ones
/// also require the material axes — `V1X, V1Y` in 2-D, `V1X…V1Z, V2X…V2Z`
/// in 3-D — which are orthonormalised internally.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (fespace, kinematics, symmetry=None))]
pub fn elasticity(
    fespace: PyRef<PyFiniteElementSpace>,
    kinematics: Kinematics,
    symmetry: Option<MaterialSymmetry>,
) -> PyResult<PyModel> {
    let s = symmetry.unwrap_or(MaterialSymmetry::Isotropic);
    let inner = model::elasticity_with_symmetry(&fespace.inner, kinematics, s)?;
    Ok(PyModel { inner })
}

/// `model.dirichlet(imposed_variable, target_dual, imposed_mesh,
/// multiplier_mesh, multiplier=None, imposed_value=None, sense="=")` —
/// Dirichlet constraint model (a single sub-model) imposed via Lagrange
/// multipliers.
///
/// `imposed_variable` is the constrained primary variable (e.g. `"T"`);
/// `target_dual` is the dual variable of the target physics it couples
/// into (e.g. `"q"` for heat conduction). `imposed_mesh` is the POI1 mesh
/// of constrained nodes; `multiplier_mesh` is the POI1 support of the
/// multipliers (same per-submesh cell count) — usually built from
/// `imposed_mesh` with the `barycenter` mesher. `multiplier` /
/// `imposed_value` override the derived names `lambda_<imposed_variable>` /
/// `imposed_<imposed_variable>`. The imposed value `u_d` is written by the
/// user in the load field at the multiplier node's `imposed_value`
/// component. `sense` (`"="`, `">="` or `"<="`, default `"="`) turns the
/// constraint unilateral (`u ≥ u_d` / `u ≤ u_d`) — such a model is solved
/// with `solve_unilateral`. See the model chapter of the book for the full
/// semantics.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (imposed_variable, target_dual, imposed_mesh, multiplier_mesh, multiplier=None, imposed_value=None, sense=None))]
#[allow(clippy::too_many_arguments)]
pub fn dirichlet(
    imposed_variable: String,
    target_dual: String,
    imposed_mesh: PyRef<'_, PyMesh>,
    multiplier_mesh: PyRef<'_, PyMesh>,
    multiplier: Option<String>,
    imposed_value: Option<String>,
    sense: Option<String>,
) -> PyResult<PyModel> {
    let inner = model::dirichlet(
        imposed_variable,
        target_dual,
        &imposed_mesh.inner,
        &multiplier_mesh.inner,
        multiplier,
        imposed_value,
        RelationSense::parse(sense.as_deref())?,
    )?;
    Ok(PyModel { inner })
}

/// `model.mpc(terms, multiplier_mesh, multiplier=None, imposed_value=None,
/// sense="=")` — multi-point constraint (a single sub-model) imposing, per
/// relation, `Σₖ aₖ·u(nodeₖ, varₖ) = g` via Lagrange multipliers.
///
/// `terms` is a list of `(mesh, variable, target_dual, coefficient)` tuples:
/// each `mesh` is a POI1 mesh (one node per relation), paired
/// element-for-element with the others and with `multiplier_mesh` (relation
/// `r` = cell `r` of every mesh). Find `target_dual` with
/// `model.dual_of(variable)`. `multiplier` / `imposed_value` override the
/// derived names `lambda_mpc` / `mpc_rhs`. The right-hand side `g` is written
/// by the user in the load field at the multiplier node's `imposed_value`
/// component (default `0`). `sense` (`"="`, `">="` or `"<="`, default `"="`)
/// turns the relations unilateral (`Σ aₖ·uₖ ≥ g` / `≤ g`) — such a model is
/// solved with `solve_unilateral`. See the constraints chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (terms, multiplier_mesh, multiplier=None, imposed_value=None, sense=None))]
pub fn mpc(
    py: Python<'_>,
    terms: Vec<(Py<PyMesh>, String, String, f64)>,
    multiplier_mesh: PyRef<'_, PyMesh>,
    multiplier: Option<String>,
    imposed_value: Option<String>,
    sense: Option<String>,
) -> PyResult<PyModel> {
    let mut rust_terms = Vec::with_capacity(terms.len());
    for (mesh, variable, target_dual, coefficient) in &terms {
        let mesh = mesh.borrow(py);
        rust_terms.push(MpcTerm::new(
            &mesh.inner,
            variable.clone(),
            target_dual.clone(),
            *coefficient,
        )?);
    }
    let inner = model::mpc(
        rust_terms,
        &multiplier_mesh.inner,
        multiplier,
        imposed_value,
        RelationSense::parse(sense.as_deref())?,
    )?;
    Ok(PyModel { inner })
}

/// `model.embedded(immersed, host, components, multipliers=None,
/// imposed_values=None, tol=None)` — embedded (immersed) constraint (a
/// single sub-model) tying each node of `immersed` to the interpolation of
/// `host` at that node, via Lagrange multipliers.
///
/// `immersed` and `host` are meshes sharing one Coords (e.g. a bar
/// « baignée » in a volume). `components` is a list of `(variable,
/// target_dual)` pairs — the field components to tie (e.g.
/// `[("u_x","f_x"), ("u_y","f_y"), ("u_z","f_z")]`); find each `target_dual`
/// with `model.dual_of(variable)`. The coupling weights are the host shape
/// functions at each immersed node, computed once at build by locating the
/// node in the host (an immersed node outside the host is an error).
/// `multipliers` / `imposed_values` override the per-component derived names
/// `lambda_<variable>` / `imposed_<variable>`; `tol` is the location
/// tolerance (default `1e-6`). The right-hand side `g` defaults to `0` (a
/// rigid tie). See the constraints chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (immersed, host, components, multipliers=None, imposed_values=None, tol=None))]
pub fn embedded(
    immersed: PyRef<'_, PyMesh>,
    host: PyRef<'_, PyMesh>,
    components: Vec<(String, String)>,
    multipliers: Option<Vec<String>>,
    imposed_values: Option<Vec<String>>,
    tol: Option<f64>,
) -> PyResult<PyModel> {
    let inner = model::embedded(
        &immersed.inner,
        &host.inner,
        components,
        multipliers,
        imposed_values,
        tol,
    )?;
    Ok(PyModel { inner })
}

/// `model.contact(slave, master, components, multiplier=None,
/// imposed_value=None)` — node-to-surface contact (a single sub-model):
/// prevent the nodes of `slave` from penetrating the oriented `master`
/// surface mesh, one **unilateral** relation (`≥`) per slave node.
///
/// Each slave node is paired at build with its closest master facet
/// (projection weights, facet normal, initial signed gap); the pairing is
/// then fixed (linearised, frictionless contact). The master surface must
/// be consistently oriented with its normal pointing toward the slave
/// body. `components` is a list of `(variable, target_dual)` pairs — one
/// per space dimension, in ambient order (e.g.
/// `[("u_x","f_x"), ("u_y","f_y")]`); find each `target_dual` with
/// `model.dual_of(variable)`. `multiplier` / `imposed_value` override the
/// derived names `lambda_contact` / `contact_gap`.
///
/// Solve with `solve_unilateral`; build the initial-gap right-hand side
/// with `contact_gaps()`. See the contact chapter of the book.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (slave, master, components, multiplier=None, imposed_value=None))]
pub fn contact(
    slave: PyRef<'_, PyMesh>,
    master: PyRef<'_, PyMesh>,
    components: Vec<(String, String)>,
    multiplier: Option<String>,
    imposed_value: Option<String>,
) -> PyResult<PyModel> {
    let inner = model::contact(
        &slave.inner,
        &master.inner,
        components,
        multiplier,
        imposed_value,
    )?;
    Ok(PyModel { inner })
}
