//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::model::{Model, SubModel};
use crate::models::elasticity::ElasticityModel;
use crate::models::interface_transfer::TransferKind;
use crate::models::symmetry::MaterialSymmetry;
use crate::models::{mpc, Physics, RelationSense};
use crate::py::finite_element_space::{PyFiniteElementSpace, PySubFiniteElementSpace};
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use crate::py::node_field::PyNodeField;
use crate::store::{read, Handle};
use pyo3::prelude::*;

/// Parse the optional `symmetry` tag shared by every physics that reads an
/// oriented material (`elasticity`, `heat_conduction`, `fick`). `None` means the
/// isotropic default, so adding the axis broke no existing call.
fn parse_symmetry(what: &str, tag: Option<&str>) -> PyResult<MaterialSymmetry> {
    match tag {
        None => Ok(MaterialSymmetry::Isotropic),
        Some(t) => MaterialSymmetry::from_tag(t).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{what}: unknown symmetry '{t}' (expected isotropic|orthotropic|anisotropic)"
            ))
        }),
    }
}

/// A **view** into one sub-model of a `Model`, obtained by indexing
/// (`model[i]`) — never constructed directly. Build physics at the parent
/// level instead: `Model.heat_conduction(fes)` or `Model.dirichlet(...)`,
/// composed with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubModel")]
pub struct PySubModel {
    pub(crate) handle: Handle<SubModel>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubModel {
    /// Names of the primal (primary) variables of this sub-model.
    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.primal_vars())
    }

    /// Names of the dual variables of this sub-model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.dual_vars())
    }

    /// `sub.fespace()` — the `SubFiniteElementSpace` this sub-model integrates
    /// its behaviour on, or `None` for a constraint sub-model (Dirichlet, MPC…,
    /// which integrate nothing). The per-sub-model counterpart of
    /// `Model.fespace()`.
    fn fespace(&self) -> PyResult<Option<PySubFiniteElementSpace>> {
        Ok(read(&self.handle)?
            .behavior_fespace()
            .map(|handle| PySubFiniteElementSpace { handle }))
    }

    /// The physics nature(s) of this sub-model as a list of tags (`"mechanical"`,
    /// `"thermal"`, `"constraint"`, `"other"`). Determined entirely by the physics
    /// kind — one tag for a plain physics, several for a coupled one.
    fn physics(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?
            .physics()
            .iter()
            .map(|p| p.to_tag().to_string())
            .collect())
    }

    /// POI1 `Mesh` of the multiplier nodes (Lagrange physics only — empty
    /// otherwise). Build a load `SubNodeField` on `mesh[0]` to impose the
    /// constrained values, or read the nodes via `mesh.node(0, i, 0)`.
    fn multiplier_mesh(&self) -> PyResult<PyMesh> {
        let mesh = read(&self.handle)?.multiplier_mesh()?;
        Ok(PyMesh { inner: mesh })
    }

    /// Names of the material components this sub-model expects, or
    /// `None` for physics that don't need material data (Dirichlet, …).
    fn material_components(&self) -> PyResult<Option<Vec<String>>> {
        Ok(read(&self.handle)?
            .material_components()
            .map(|c| c.iter().map(|s| s.to_string()).collect()))
    }

    /// Whether this sub-model carries a constitutive behaviour that can be
    /// integrated with `deformation` / `integrate_behavior` (`True` for
    /// volumetric physics, `False` for constraints like Dirichlet).
    fn has_behavior(&self) -> PyResult<bool> {
        Ok(read(&self.handle)?.has_behavior())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
    }
}

/// A physics problem: a collection of sub-models (heat conduction,
/// Dirichlet BCs, ...) over finite-element spaces.
///
/// Build sub-models with `Model.heat_conduction(fes)` / `Model.dirichlet(...)`
/// and compose them with `|`; assemble with `stiffness` / `mass`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Model,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// `Model()` — an empty model; add physics with `|`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            inner: Model::empty(),
        })
    }

    /// `Model.heat_conduction(fespace, symmetry=None)` — heat-conduction model
    /// spanning **every** subspace of `fespace` (one zone per subspace). A
    /// single-subspace space gives the unit case; several give one zone
    /// each. Compose heterogeneous physics with `|`:
    /// `Model.heat_conduction(fes) | Model.dirichlet(...)`.
    ///
    /// `symmetry` is `"isotropic"` (the default), `"orthotropic"` or
    /// `"anisotropic"`, and selects which conductivity the material field must
    /// carry: the scalar `k`, the principal `k_1, k_2, k_3`, or the symmetric
    /// tensor `k_11 … k_33`. The two oriented ones also require the material
    /// axes — `V1X, V1Y` in 2-D, `V1X…V1Z, V2X…V2Z` in 3-D.
    #[classmethod]
    #[pyo3(signature = (fespace, symmetry=None))]
    fn heat_conduction(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        symmetry: Option<&str>,
    ) -> PyResult<Self> {
        let s = parse_symmetry("heat_conduction", symmetry)?;
        let inner = Model::heat_conduction_with_symmetry(&fespace.inner, s)?;
        Ok(Self { inner })
    }

    /// `Model.fick(fespace, symmetry=None)` — Fickian-diffusion model spanning
    /// **every** subspace of `fespace`. DOFs are the concentration `c` (primal)
    /// and the mass flux `j` (dual); its physics nature is `"diffusion"`, so
    /// `model.filter("diffusion")` isolates it from a thermal or mechanical
    /// model it is composed with.
    ///
    /// `symmetry` is `"isotropic"` (the default), `"orthotropic"` or
    /// `"anisotropic"`, selecting the diffusivity the material field must carry:
    /// `D`, `D_1, D_2, D_3`, or the symmetric `D_11 … D_33` — the oriented ones
    /// plus the material axes. The transient (storage) term is the mass matrix,
    /// which reads the optional `poro`.
    #[classmethod]
    #[pyo3(signature = (fespace, symmetry=None))]
    fn fick(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        symmetry: Option<&str>,
    ) -> PyResult<Self> {
        let s = parse_symmetry("fick", symmetry)?;
        let inner = Model::fick(&fespace.inner, s)?;
        Ok(Self { inner })
    }

    /// `Model.radiation(fespace)` — radiation to infinity on a *boundary*
    /// `fespace`: `q·n = σε(T⁴ − T_∞⁴)`. Same DOFs (`"T"`/`"q"`) as
    /// `heat_conduction`, so it composes with `|`:
    /// `Model.heat_conduction(bulk) | Model.radiation(boundary)`.
    ///
    /// Material: `emis` (emissivity) and `T_inf` (far-field temperature), plus an
    /// optional `sigma` overriding the SI Stefan-Boltzmann constant. With the
    /// default `sigma`, `T` is an **absolute** temperature — a fourth power has
    /// no invariance to shift an origin through.
    ///
    /// Unlike convection this law is non-linear, so it contributes three terms:
    /// the linearised film `4σεT_∞³∫NᵢNⱼ` as stiffness, the exact residual
    /// `∫Nᵢσε(T⁴ − T_∞⁴)` through `internal_forces`, and the consistent tangent
    /// `4σεT³∫NᵢNⱼ` through `matrix.tangent(...)`. Its natures are `"thermal"`
    /// **and** `"radiation"`.
    #[classmethod]
    fn radiation(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::radiation(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.interface_transfer(side_a, side_b, kind=None, tol=None)` — the
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
    #[classmethod]
    #[pyo3(signature = (side_a, side_b, kind=None, tol=None))]
    fn interface_transfer(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        side_a: PyRef<PyFiniteElementSpace>,
        side_b: PyRef<PyFiniteElementSpace>,
        kind: Option<&str>,
        tol: Option<f64>,
    ) -> PyResult<Self> {
        let k = match kind {
            None => TransferKind::Mass,
            Some(t) => TransferKind::from_tag(t).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "interface_transfer: unknown kind '{t}' (expected mass|thermal)"
                ))
            })?,
        };
        let inner =
            Model::interface_transfer(&side_a.inner, &side_b.inner, k, tol.unwrap_or(1e-9))?;
        Ok(Self { inner })
    }

    /// `Model.convection(fespace)` — surface-convection (Robin / film) model
    /// spanning **every** subspace of a *boundary* `fespace` (edge mesh in 2-D,
    /// surface mesh in 3-D). Same DOFs (`"T"`/`"q"`) as `heat_conduction`, so
    /// it couples in with `|`:
    /// `Model.heat_conduction(bulk) | Model.convection(boundary)`.
    /// The film coefficient `"h"` is supplied at assembly time; the external
    /// temperature enters as a load `h·T_ext·∫N_i dΓ`, built with `flux(...)`.
    #[classmethod]
    fn convection(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::convection(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.truss(fespace)` — truss / bar (axial-force) model spanning
    /// **every** subspace of `fespace` (SEG2 elements). DOFs are the vector
    /// displacement `u_x, u_y(, u_z)`; the orientation is taken from the node
    /// coordinates. Material (`E`, `A`) is supplied at assembly time.
    #[classmethod]
    fn truss(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::truss(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.elasticity(fespace, model, symmetry=None)` — linear-elasticity
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
    #[classmethod]
    #[pyo3(signature = (fespace, model, symmetry=None))]
    fn elasticity(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
        symmetry: Option<&str>,
    ) -> PyResult<Self> {
        let m = ElasticityModel::from_tag(model).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "elasticity: unknown model '{model}' \
                 (expected plane_stress|plane_strain|axisymmetric|solid)"
            ))
        })?;
        let s = parse_symmetry("elasticity", symmetry)?;
        let inner = Model::elasticity_with_symmetry(&fespace.inner, m, s)?;
        Ok(Self { inner })
    }

    /// `Model.plasticity(fespace, model)` — perfect von Mises elastoplasticity
    /// spanning every subspace of `fespace`. `model` is `"plane_stress"` /
    /// `"plane_strain"` / `"axisymmetric"` (2-D) or `"solid"` (3-D). Same DOFs as elasticity
    /// (`u_x, u_y(, u_z)`); material (`E`, `nu`, `sigma_y`) is supplied at
    /// assembly / integration time. The behaviour integration (`COMP`) carries
    /// the plastic-strain + cumulated-`p` internal state (`VAR0`→`VAR1`).
    #[classmethod]
    fn plasticity(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        let m = ElasticityModel::from_tag(model).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "plasticity: unknown model '{model}' \
                 (expected plane_stress|plane_strain|axisymmetric|solid)"
            ))
        })?;
        let inner = Model::plasticity(&fespace.inner, m)?;
        Ok(Self { inner })
    }

    /// `Model.mazars(fespace, model)` — Mazars isotropic damage spanning every
    /// subspace of `fespace`. `model` is `"plane_stress"` / `"plane_strain"` /
    /// `"axisymmetric"` (2-D) or `"solid"` (3-D). Same DOFs as elasticity; material
    /// (`E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c`) is supplied at
    /// assembly / integration time. The behaviour integration (`COMP`) carries
    /// the scalar history variable `kappa` (`VAR0`→`VAR1`) and outputs `damage`.
    #[classmethod]
    fn mazars(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        let m = ElasticityModel::from_tag(model).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "mazars: unknown model '{model}' \
                 (expected plane_stress|plane_strain|axisymmetric|solid)"
            ))
        })?;
        let inner = Model::mazars(&fespace.inner, m)?;
        Ok(Self { inner })
    }

    /// `Model.timoshenko(fespace)` — Timoshenko-beam model spanning every
    /// subspace of `fespace` (1-D `SEG2`). DOFs `w` (deflection) and `theta`
    /// (rotation); reduced shear integration avoids locking. Material
    /// (`E`, `I`, `G`, `A_s`) is supplied at assembly time.
    #[classmethod]
    fn timoshenko(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::timoshenko(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.frame(fespace)` — planar frame / portique model spanning every
    /// subspace of `fespace` (2-D `SEG2`): an oriented Timoshenko beam carrying
    /// axial + bending + shear. DOFs `u_x, u_y, rz`; material
    /// (`E`, `A`, `I`, `G`, `A_s`) is supplied at assembly time.
    #[classmethod]
    fn frame(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::frame(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.frame3d(fespace)` — 3-D Timoshenko frame (space frame) spanning
    /// every subspace of `fespace` (3-D `SEG2`). 6 DOFs/node: `u_x, u_y, u_z,
    /// r_x, r_y, r_z`. Section axes are auto-oriented (global-Z reference).
    /// Material (`E, A, I_y, I_z, J, G, A_sy, A_sz`) is supplied at assembly time.
    #[classmethod]
    fn frame3d(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::frame3d(&fespace.inner)?;
        Ok(Self { inner })
    }

    /// `Model.dirichlet(imposed_variable, target_dual, imposed_mesh,
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
    #[classmethod]
    #[pyo3(signature = (imposed_variable, target_dual, imposed_mesh, multiplier_mesh, multiplier=None, imposed_value=None, sense=None))]
    #[allow(clippy::too_many_arguments)]
    fn dirichlet(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: PyRef<'_, PyMesh>,
        multiplier_mesh: PyRef<'_, PyMesh>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: Option<String>,
    ) -> PyResult<Self> {
        let inner = Model::dirichlet(
            imposed_variable,
            target_dual,
            &imposed_mesh.inner,
            &multiplier_mesh.inner,
            multiplier,
            imposed_value,
            RelationSense::parse(sense.as_deref())?,
        )?;
        Ok(Self { inner })
    }

    /// `Model.mpc(terms, multiplier_mesh, multiplier=None, imposed_value=None,
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
    #[classmethod]
    #[pyo3(signature = (terms, multiplier_mesh, multiplier=None, imposed_value=None, sense=None))]
    fn mpc(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        terms: Vec<(Py<PyMesh>, String, String, f64)>,
        multiplier_mesh: PyRef<'_, PyMesh>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
        sense: Option<String>,
    ) -> PyResult<Self> {
        let mut rust_terms = Vec::with_capacity(terms.len());
        for (mesh, variable, target_dual, coefficient) in &terms {
            let mesh = mesh.borrow(py);
            rust_terms.push(mpc::MpcTerm::new(
                &mesh.inner,
                variable.clone(),
                target_dual.clone(),
                *coefficient,
            )?);
        }
        let inner = Model::mpc(
            rust_terms,
            &multiplier_mesh.inner,
            multiplier,
            imposed_value,
            RelationSense::parse(sense.as_deref())?,
        )?;
        Ok(Self { inner })
    }

    /// `Model.embedded(immersed, host, components, multipliers=None,
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
    #[classmethod]
    #[pyo3(signature = (immersed, host, components, multipliers=None, imposed_values=None, tol=None))]
    fn embedded(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        immersed: PyRef<'_, PyMesh>,
        host: PyRef<'_, PyMesh>,
        components: Vec<(String, String)>,
        multipliers: Option<Vec<String>>,
        imposed_values: Option<Vec<String>>,
        tol: Option<f64>,
    ) -> PyResult<Self> {
        let inner = Model::embedded(
            &immersed.inner,
            &host.inner,
            components,
            multipliers,
            imposed_values,
            tol,
        )?;
        Ok(Self { inner })
    }

    /// `Model.contact(slave, master, components, multiplier=None,
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
    #[classmethod]
    #[pyo3(signature = (slave, master, components, multiplier=None, imposed_value=None))]
    fn contact(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        slave: PyRef<'_, PyMesh>,
        master: PyRef<'_, PyMesh>,
        components: Vec<(String, String)>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> PyResult<Self> {
        let inner = Model::contact(
            &slave.inner,
            &master.inner,
            components,
            multiplier,
            imposed_value,
        )?;
        Ok(Self { inner })
    }

    /// `Model.contact_gaps()` — the contact right-hand side `−g₀`: a
    /// `NodeField` carrying, at each contact multiplier node's `imposed_value`
    /// slot, minus the initial signed gap of its relation, so that
    /// non-penetration reads `g₀ + C·u ≥ 0`. Merge it into the global load
    /// with `|`. The model must hold exactly one contact sub-model; omitting
    /// this helper treats every pair as initially touching (`g₀ = 0`).
    fn contact_gaps(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: self.inner.contact_gaps()?,
        })
    }

    /// Names of the primal (primary) variables across the whole model.
    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.primal_vars()?)
    }

    /// Names of the dual variables across the whole model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.dual_vars()?)
    }

    /// `Model.dual_of(variable)` — the dual (residual) variable conjugate to a
    /// primal `variable` (e.g. `"u_x" -> "f_x"`, `"T" -> "q"`), searched across
    /// all sub-models, or `None`. A helper to fill an MPC term's `target_dual`.
    fn dual_of(&self, variable: &str) -> PyResult<Option<String>> {
        Ok(self.inner.dual_of(variable)?)
    }

    /// `Model.fespace()` — the `FiniteElementSpace` this model integrates on,
    /// rebuilt from the behaviour subspaces of its domain sub-models (constraints
    /// skipped), deduplicated in first-seen order (shared handles). Raises if the
    /// model has no domain sub-model. Combined with `FiniteElementSpace.mesh()`,
    /// lets a caller recover the FE space and mesh from the model alone.
    fn fespace(&self) -> PyResult<PyFiniteElementSpace> {
        Ok(PyFiniteElementSpace {
            inner: self.inner.fespace()?,
        })
    }

    /// `Model.filter(physics)` — a new `Model` holding only the sub-models **whose
    /// nature set contains** the given physics. `physics` is a tag: `"mechanical"`,
    /// `"thermal"`, `"constraint"`, `"other"`, `"diffusion"` or `"radiation"`.
    /// Sub-model order is preserved; the result may be empty.
    ///
    /// A coupled physics declares several natures and is therefore returned by
    /// each of its filters — a radiation boundary is both `"thermal"` and
    /// `"radiation"`.
    fn filter(&self, physics: &str) -> PyResult<Self> {
        let p = Physics::from_tag(physics).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "filter: unknown physics '{physics}' (expected {})",
                Physics::tag_list()
            ))
        })?;
        Ok(Self {
            inner: self.inner.filter(p)?,
        })
    }

    /// `Model.multiplier_mesh()` — POI1 `Mesh` of the constraint multiplier
    /// nodes across the model (empty for a model with no constraint). The handle
    /// to the multiplier nodes of an `embedded` / `dirichlet` / `mpc` model whose
    /// multipliers it minted itself: read them via `mesh.node(0, i, 0)` or build
    /// a load `SubNodeField` on `mesh[0]`.
    fn multiplier_mesh(&self) -> PyResult<PyMesh> {
        Ok(PyMesh {
            inner: self.inner.multiplier_mesh()?,
        })
    }

    /// `Model.constraint_rhs(imposed)` — build the constraint load (right-hand
    /// side) of this model's constraint from `(node, g)` pairs.
    ///
    /// `imposed` is a list of `(Node, value)`: each node **keys the relation it
    /// belongs to** (for `dirichlet` the constrained node itself, for `mpc` any
    /// of the relation's term nodes), and `value` is the right-hand side `g`.
    /// Returns a fresh `NodeField` over the multiplier nodes, carrying the
    /// constraint's imposed-value component (its dual, e.g. `imposed_T` /
    /// `mpc_rhs`), with `g` at each cited relation and `0` elsewhere. Union it
    /// into the global load with `|`, e.g. `load | model.constraint_rhs([...])`.
    ///
    /// The model must hold exactly one constraint sub-model (the `dirichlet` /
    /// `mpc` object). Raises if a node constrains none of its relations, or keys
    /// several of them (ambiguous — drop it).
    fn constraint_rhs(
        &self,
        py: Python<'_>,
        imposed: Vec<(Py<PyNode>, f64)>,
    ) -> PyResult<PyNodeField> {
        let pairs: Vec<(NodeId, f64)> = imposed
            .iter()
            .map(|(node, g)| (node.borrow(py).as_node().id(), *g))
            .collect();
        let inner = self.inner.constraint_rhs(&pairs)?;
        Ok(PyNodeField { inner })
    }

    /// `Model.constraint_rhs_by_index(imposed)` — like `constraint_rhs` but each
    /// relation is keyed by its **index** (0-based, in `relations()` order)
    /// instead of a node.
    ///
    /// `imposed` is a list of `(relation_index, g)`. Use this when a node
    /// participates in several relations, where node keying would be ambiguous.
    /// Returns the same kind of `NodeField` over the multiplier nodes; union it
    /// with `|`. Raises if an index is out of range.
    fn constraint_rhs_by_index(&self, imposed: Vec<(usize, f64)>) -> PyResult<PyNodeField> {
        let inner = self.inner.constraint_rhs_by_index(&imposed)?;
        Ok(PyNodeField { inner })
    }
}

crate::impl_aggregate_pymethods!(
    PyModel,
    PySubModel,
    "Model",
    sub_model,
    Model,
    r#"
class PyModel:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubModel"]:
        """`model[i]` → the `SubModel` view of term i (a physics term, a
        Dirichlet condition, an MPC, ...)."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyModel"]:
        """`model[i:j:k]` → a fresh `Model` holding the sliced terms, shared
        with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyModel"] | pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`model | other` → a fresh `Model` holding the terms of both, in
        first-seen order and deduplicated by store slot. This is how a problem
        is composed: physics | boundary conditions | constraints."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`sub_model | model` — the mirror of `model | sub_model`, differing
        only in that the lone term comes first."""
    "#,
    r#"
class PySubModel:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`sub_model | sub_model` → a fresh `Model` holding both terms."""
    "#
);
crate::impl_dump_pymethod!(handle PySubModel, handle);
