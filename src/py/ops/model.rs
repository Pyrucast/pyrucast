//! Python wrappers for [`crate::ops::model`] — the physics declarations,
//! which produce a `Model`.
//!
//! Each one spans the **whole** support it is given (one zone per subspace);
//! heterogeneous physics compose with `|`.

use crate::models::damage::DamageLaw;
use crate::models::elasticity::ElasticityModel;
use crate::models::mpc::MpcTerm;
use crate::models::plastic::PlasticLaw;
use crate::models::symmetry::MaterialSymmetry;
use crate::models::{Physics, RelationSense};
use crate::ops::model;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::model::PyModel;
use pyo3::prelude::*;

/// Build a damage `Model` for a given law, parsing the kinematic tag.
fn damage_with(
    fespace: &PyFiniteElementSpace,
    kinematics: ElasticityModel,
    law: DamageLaw,
) -> PyResult<PyModel> {
    Ok(PyModel {
        inner: model::damage_with_law(&fespace.inner, kinematics, law)?,
    })
}

/// Build an elastoplastic `Model` for a given yield law, parsing the kinematic
/// tag. Shared by the per-law operators, which differ only in the law they
/// name — the material contract follows from it.
fn plasticity_with(
    fespace: &PyFiniteElementSpace,
    kinematics: ElasticityModel,
    law: PlasticLaw,
) -> PyResult<PyModel> {
    Ok(PyModel {
        inner: model::plasticity_with_law(&fespace.inner, kinematics, law)?,
    })
}

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

/// `model.elasticity(fespace, model, symmetry=None)` — linear-elasticity
/// model spanning every subspace of `fespace`. `model` is `"plane_stress"`,
/// `"plane_strain"` or `"axisymmetric"` (2-D), or `"solid"` (3-D). DOFs are
/// the vector displacement `u_x, u_y(, u_z)`; material is supplied at
/// assembly time.
///
/// `"axisymmetric"` requires a geometry built with `Coords.axisymmetric()`
/// (`x = r`, `y = z`): the hoop strain `ε_θθ = u_r / r` comes from the model,
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
#[pyo3(signature = (fespace, model, symmetry=None))]
pub fn elasticity(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
    symmetry: Option<MaterialSymmetry>,
) -> PyResult<PyModel> {
    let s = symmetry.unwrap_or(MaterialSymmetry::Isotropic);
    let inner = model::elasticity_with_symmetry(&fespace.inner, model, s)?;
    Ok(PyModel { inner })
}

/// `model.plasticity_perfect(fespace, model)` — **perfect** (non-hardening)
/// von Mises elastoplasticity spanning every subspace of `fespace`. `model`
/// is `"plane_stress"` / `"plane_strain"` / `"axisymmetric"` (2-D) or
/// `"solid"` (3-D). Same DOFs as elasticity (`u_x, u_y(, u_z)`); material
/// (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time. The
/// behaviour integration (`COMP`) carries the plastic-strain +
/// cumulated-`p` internal state (`VAR0`→`VAR1`) and emits the consistent
/// tangent `D_alg`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn plasticity_perfect(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::Perfect)
}

/// `model.plasticity_isotropic(fespace, model)` — von Mises with **linear
/// isotropic hardening**, `σ_y(p) = σ_y + H·p`. Material `E`, `nu`,
/// `sigma_y`, `H`; everything else as `plasticity_perfect` (`H = 0` would
/// give it back exactly).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn plasticity_isotropic(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::Isotropic)
}

/// `model.drucker_prager(fespace, model)` — pressure-sensitive plasticity
/// with **non-associated** flow: `f = q + α·I₁ − k`, plastic potential
/// `g = q + ψ·I₁`. Material `E`, `nu`, `alpha` (friction), `k` (cohesion),
/// `psi` (dilatancy).
///
/// `ψ = α` recovers associated flow; `ψ < α` is the usual choice for soils
/// and rocks, whose measured dilatancy is far below what friction alone
/// would imply. A non-associated law has a **non-symmetric** tangent.
/// Returns beyond the cone's apex (`I₁ = k/α`) collapse onto the tip.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn drucker_prager(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::DruckerPrager)
}

/// `model.ottosen(fespace, model)` — Ottosen's four-parameter criterion for
/// concrete, whose strength depends on the pressure **and** on the Lode
/// angle (so tension and compression differ). Material `E`, `nu`, `a`, `b`,
/// `k_1`, `k_2`, `sigma_c`.
///
/// Integrated by a cutting-plane return with a numerically differentiated
/// normal: the criterion is exact, and the gradient — long enough that a
/// hand-derived one could not be checked — is a central difference.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn ottosen(fespace: PyRef<PyFiniteElementSpace>, model: ElasticityModel) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::Ottosen)
}

/// `model.creep_norton(fespace, model)` — Norton-Odqvist secondary creep,
/// `ṗ = (q/K)^n`. Material `E`, `nu`, `K`, `n`.
///
/// There is **no yield threshold**: any stress creeps, however slowly. Like
/// every rate-dependent law it needs the time increment —
/// `integrate_behavior(..., dt=...)` — and raises without one, because
/// integrating a creep law as if it were instantaneous would give a
/// plausible wrong answer.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn creep_norton(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::CreepNorton)
}

/// `model.creep_blackburn(fespace, model)` — a **saturating primary** creep
/// stage plus a steady secondary one, with Blackburn's `sinh` stress
/// dependence (which spans decades of stress where a power law cannot).
/// Material `E`, `nu`, `A_1`, `alpha_1`, `r_1`, `B_s`, `beta_s`.
///
/// The primary strain is tracked as its own internal variable (`p_prim`), so
/// the law integrates correctly under a varying load.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn creep_blackburn(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::CreepBlackburn)
}

/// `model.creep_lemaitre(fespace, model)` — Lemaitre primary creep by
/// **strain** hardening, `ṗ = (q/K)^N · p^(−M)`. Material `E`, `nu`, `K`,
/// `N`, `M`.
///
/// The accumulated strain itself slows the flow, producing a decelerating
/// primary stage with no explicit time dependence — which is what makes it
/// usable under a varying load, where a time-hardening form would be wrong.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn creep_lemaitre(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::CreepLemaitre)
}

/// `model.viscoplasticity_chaboche(fespace, model)` — a Norton flow on the
/// shifted overstress `J(σ − X) − R − k`, with Armstrong-Frederick kinematic
/// hardening and saturating isotropic hardening. Material `E`, `nu`, `k`,
/// `K`, `n`, `C_1`, `gamma_1`, `b`, `Q`.
///
/// The back stress `X` is what makes the law usable under **cyclic**
/// loading: it translates the yield surface, so reverse yielding happens
/// early — the Bauschinger effect, which no isotropic law can produce. It
/// costs seven internal variables (`X_xx…X_xy`, `R`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn viscoplasticity_chaboche(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::ViscoplasticChaboche)
}

/// `model.viscoplasticity_lemaitre_chaboche(fespace, model)` — Chaboche
/// viscoplasticity coupled to Lemaitre's ductile **damage**: the flow is
/// driven by the effective stress `σ/(1−D)`, and `Ḋ = (Y/S)^s·ṗ`. Material
/// as above, plus `S`, `s`, `D_c`.
///
/// A damaged material flows faster, which damages it more — the coupling
/// that produces tertiary creep and, at `D_c`, rupture. Adds `damage` to the
/// internal state.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn viscoplasticity_lemaitre_chaboche(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::ViscoplasticLemaitreChaboche)
}

/// `model.damage_tc(fespace, model)` — **two** damage variables, tension
/// and compression apart: `σ = (1−d⁺)σ̃⁺ + (1−d⁻)σ̃⁻`. Material `E`, `nu`,
/// `f_t`, `f_c`, `A_t`, `A_c`.
///
/// Mazars blends its two branches into one scalar, so a material damaged in
/// compression is equally damaged in tension and a crack that **closes**
/// cannot carry load again. Keeping the two apart recovers the compressive
/// stiffness on closure — the unilateral effect — which is what makes the
/// law usable under cyclic loading. State: `r_plus`, `r_minus`, `d_plus`,
/// `d_minus`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn damage_tc(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    damage_with(&fespace, model, DamageLaw::DamageTc)
}

/// `model.damage_sic_sic(fespace, model)` — **orthotropic** damage of a
/// woven SiC/SiC ceramic-matrix composite: one damage per weave direction.
/// Material `E`, `nu`, then `eps_0_i`, `eps_c_i`, `d_max_i` for `i = 1..3`,
/// plus the material axes (`V1X, V1Y[, V1Z, V2X…]`).
///
/// The matrix cracks in planes normal to the tows while the fibres keep
/// carrying load, so the stiffness falls **by direction** and by very
/// different amounts — which no scalar damage can express. The directions
/// are the same material axes an orthotropic elasticity uses, so a curved
/// part gets them right cell by cell.
///
/// Each damage **saturates** at `d_max_i` rather than reaching one: matrix
/// cracking does not take the whole stiffness, and a law that let it would
/// predict a collapse that does not happen. State: `kappa_1..3`, `d_1..3`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn damage_sic_sic(
    fespace: PyRef<PyFiniteElementSpace>,
    model: ElasticityModel,
) -> PyResult<PyModel> {
    damage_with(&fespace, model, DamageLaw::SicSic)
}

/// `model.gurson(fespace, model)` — Gurson-Tvergaard-Needleman plasticity
/// of a **porous** metal, where the porosity shrinks the yield surface.
/// Material `E`, `nu`, `sigma_y`, `q_1`, `q_2`, `q_3`, `f_0`, `f_c`, `f_f`.
///
/// A ductile metal fails because voids grow and coalesce, not because a
/// stress is reached. The `cosh` term makes the surface **pressure
/// sensitive**, so voids grow under triaxial tension and close under
/// compression — which a J2 law cannot express, and which is why it can
/// never predict ductile rupture. Beyond `f_c` the effective porosity
/// accelerates towards `1/q_1`, modelling coalescence.
///
/// The porosity is exposed as the internal variable `porosity`, starting
/// from `f_0`. Void **nucleation** is not modelled — only growth.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn gurson(fespace: PyRef<PyFiniteElementSpace>, model: ElasticityModel) -> PyResult<PyModel> {
    plasticity_with(&fespace, model, PlasticLaw::Gurson)
}

/// `model.mazars(fespace, model)` — Mazars isotropic damage spanning every
/// subspace of `fespace`. `model` is `"plane_stress"` / `"plane_strain"` /
/// `"axisymmetric"` (2-D) or `"solid"` (3-D). Same DOFs as elasticity; material
/// (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at
/// assembly / integration time. The behaviour integration (`COMP`) carries
/// the scalar history variable `kappa` (`VAR0`→`VAR1`) and outputs `damage`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn mazars(fespace: PyRef<PyFiniteElementSpace>, model: ElasticityModel) -> PyResult<PyModel> {
    let inner = model::mazars(&fespace.inner, model)?;
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
