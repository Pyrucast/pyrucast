//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::model::{Model, SubModel};
use crate::models::bernoulli::BeamModel;
use crate::models::damage::DamageLaw;
use crate::models::elasticity::ElasticityModel;
use crate::models::interface_transfer::TransferKind;
use crate::models::plastic::PlasticLaw;
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

/// Build an elastoplastic `Model` for a given yield law, parsing the kinematic
/// tag. Shared by the four per-law classmethods, which differ only in the law
/// they name — the material contract follows from it.
/// Build a damage `Model` for a given law, parsing the kinematic tag.
fn damage_with(fespace: &PyFiniteElementSpace, model: &str, law: DamageLaw) -> PyResult<PyModel> {
    let m = ElasticityModel::from_tag(model).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "damage ({law}): unknown model '{model}' \
             (expected plane_stress|plane_strain|axisymmetric|solid)"
        ))
    })?;
    Ok(PyModel {
        inner: Model::damage_with_law(&fespace.inner, m, law)?,
    })
}

fn plasticity_with(
    fespace: &PyFiniteElementSpace,
    model: &str,
    law: PlasticLaw,
) -> PyResult<PyModel> {
    let m = ElasticityModel::from_tag(model).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "plasticity ({law}): unknown model '{model}' \
             (expected plane_stress|plane_strain|axisymmetric|solid)"
        ))
    })?;
    Ok(PyModel {
        inner: Model::plasticity_with_law(&fespace.inner, m, law)?,
    })
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

    /// `Model.follower_pressure(fespace)` — a pressure that **turns with the
    /// surface** it acts on, on a *boundary* `fespace` (an edge mesh in 2-D, a
    /// surface mesh in 3-D). Material: `p`, the pressure.
    ///
    /// Unlike a dead load built once with `flux(...)`, its direction depends on
    /// the current displacement, so it is recomputed at each residual
    /// evaluation:
    ///
    /// ```text
    /// u → element_field.gradient → integrate_behavior → node_field.internal_forces
    /// ```
    ///
    /// It contributes **no matrix** — only internal forces. A positive `p`
    /// pushes *against* the boundary mesh's own normal, which follows its
    /// winding: orienting the boundary outwards gives the usual compressive
    /// sign.
    #[classmethod]
    fn follower_pressure(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::follower_pressure(&fespace.inner)?;
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

    /// `Model.plasticity_perfect(fespace, model)` — **perfect** (non-hardening)
    /// von Mises elastoplasticity spanning every subspace of `fespace`. `model`
    /// is `"plane_stress"` / `"plane_strain"` / `"axisymmetric"` (2-D) or
    /// `"solid"` (3-D). Same DOFs as elasticity (`u_x, u_y(, u_z)`); material
    /// (`E`, `nu`, `sigma_y`) is supplied at assembly / integration time. The
    /// behaviour integration (`COMP`) carries the plastic-strain +
    /// cumulated-`p` internal state (`VAR0`→`VAR1`) and emits the consistent
    /// tangent `D_alg`.
    #[classmethod]
    fn plasticity_perfect(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::Perfect)
    }

    /// `Model.plasticity_isotropic(fespace, model)` — von Mises with **linear
    /// isotropic hardening**, `σ_y(p) = σ_y + H·p`. Material `E`, `nu`,
    /// `sigma_y`, `H`; everything else as `plasticity_perfect` (`H = 0` would
    /// give it back exactly).
    #[classmethod]
    fn plasticity_isotropic(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::Isotropic)
    }

    /// `Model.drucker_prager(fespace, model)` — pressure-sensitive plasticity
    /// with **non-associated** flow: `f = q + α·I₁ − k`, plastic potential
    /// `g = q + ψ·I₁`. Material `E`, `nu`, `alpha` (friction), `k` (cohesion),
    /// `psi` (dilatancy).
    ///
    /// `ψ = α` recovers associated flow; `ψ < α` is the usual choice for soils
    /// and rocks, whose measured dilatancy is far below what friction alone
    /// would imply. A non-associated law has a **non-symmetric** tangent.
    /// Returns beyond the cone's apex (`I₁ = k/α`) collapse onto the tip.
    #[classmethod]
    fn drucker_prager(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::DruckerPrager)
    }

    /// `Model.ottosen(fespace, model)` — Ottosen's four-parameter criterion for
    /// concrete, whose strength depends on the pressure **and** on the Lode
    /// angle (so tension and compression differ). Material `E`, `nu`, `a`, `b`,
    /// `k_1`, `k_2`, `sigma_c`.
    ///
    /// Integrated by a cutting-plane return with a numerically differentiated
    /// normal: the criterion is exact, and the gradient — long enough that a
    /// hand-derived one could not be checked — is a central difference.
    #[classmethod]
    fn ottosen(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::Ottosen)
    }

    /// `Model.creep_norton(fespace, model)` — Norton-Odqvist secondary creep,
    /// `ṗ = (q/K)^n`. Material `E`, `nu`, `K`, `n`.
    ///
    /// There is **no yield threshold**: any stress creeps, however slowly. Like
    /// every rate-dependent law it needs the time increment —
    /// `integrate_behavior(..., dt=...)` — and raises without one, because
    /// integrating a creep law as if it were instantaneous would give a
    /// plausible wrong answer.
    #[classmethod]
    fn creep_norton(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::CreepNorton)
    }

    /// `Model.creep_blackburn(fespace, model)` — a **saturating primary** creep
    /// stage plus a steady secondary one, with Blackburn's `sinh` stress
    /// dependence (which spans decades of stress where a power law cannot).
    /// Material `E`, `nu`, `A_1`, `alpha_1`, `r_1`, `B_s`, `beta_s`.
    ///
    /// The primary strain is tracked as its own internal variable (`p_prim`), so
    /// the law integrates correctly under a varying load.
    #[classmethod]
    fn creep_blackburn(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::CreepBlackburn)
    }

    /// `Model.creep_lemaitre(fespace, model)` — Lemaitre primary creep by
    /// **strain** hardening, `ṗ = (q/K)^N · p^(−M)`. Material `E`, `nu`, `K`,
    /// `N`, `M`.
    ///
    /// The accumulated strain itself slows the flow, producing a decelerating
    /// primary stage with no explicit time dependence — which is what makes it
    /// usable under a varying load, where a time-hardening form would be wrong.
    #[classmethod]
    fn creep_lemaitre(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::CreepLemaitre)
    }

    /// `Model.viscoplasticity_chaboche(fespace, model)` — a Norton flow on the
    /// shifted overstress `J(σ − X) − R − k`, with Armstrong-Frederick kinematic
    /// hardening and saturating isotropic hardening. Material `E`, `nu`, `k`,
    /// `K`, `n`, `C_1`, `gamma_1`, `b`, `Q`.
    ///
    /// The back stress `X` is what makes the law usable under **cyclic**
    /// loading: it translates the yield surface, so reverse yielding happens
    /// early — the Bauschinger effect, which no isotropic law can produce. It
    /// costs seven internal variables (`X_xx…X_xy`, `R`).
    #[classmethod]
    fn viscoplasticity_chaboche(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::ViscoplasticChaboche)
    }

    /// `Model.viscoplasticity_lemaitre_chaboche(fespace, model)` — Chaboche
    /// viscoplasticity coupled to Lemaitre's ductile **damage**: the flow is
    /// driven by the effective stress `σ/(1−D)`, and `Ḋ = (Y/S)^s·ṗ`. Material
    /// as above, plus `S`, `s`, `D_c`.
    ///
    /// A damaged material flows faster, which damages it more — the coupling
    /// that produces tertiary creep and, at `D_c`, rupture. Adds `damage` to the
    /// internal state.
    #[classmethod]
    fn viscoplasticity_lemaitre_chaboche(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::ViscoplasticLemaitreChaboche)
    }

    /// `Model.damage_tc(fespace, model)` — **two** damage variables, tension
    /// and compression apart: `σ = (1−d⁺)σ̃⁺ + (1−d⁻)σ̃⁻`. Material `E`, `nu`,
    /// `f_t`, `f_c`, `A_t`, `A_c`.
    ///
    /// Mazars blends its two branches into one scalar, so a material damaged in
    /// compression is equally damaged in tension and a crack that **closes**
    /// cannot carry load again. Keeping the two apart recovers the compressive
    /// stiffness on closure — the unilateral effect — which is what makes the
    /// law usable under cyclic loading. State: `r_plus`, `r_minus`, `d_plus`,
    /// `d_minus`.
    #[classmethod]
    fn damage_tc(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        damage_with(&fespace, model, DamageLaw::DamageTc)
    }

    /// `Model.damage_sic_sic(fespace, model)` — **orthotropic** damage of a
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
    #[classmethod]
    fn damage_sic_sic(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        damage_with(&fespace, model, DamageLaw::SicSic)
    }

    /// `Model.gurson(fespace, model)` — Gurson-Tvergaard-Needleman plasticity
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
    #[classmethod]
    fn gurson(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        plasticity_with(&fespace, model, PlasticLaw::Gurson)
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

    /// `Model.bernoulli(fespace, model)` — the classical **Euler-Bernoulli**
    /// beam, where plane sections stay normal to the deflected axis and there is
    /// no transverse shear at all. `model` is `"planar_1d"` (DOFs `w`, `theta`;
    /// material `E`, `I`), `"frame_2d"` (`u_x, u_y, r_z`; `+ A`) or
    /// `"frame_3d"` (six DOFs; `+ I_y, I_z, J, G`).
    ///
    /// Hermite cubic interpolation makes it **nodally exact** wherever the
    /// interior carries no distributed load — one element per member suffices
    /// for a frame.
    ///
    /// Prefer `timoshenko` / `frame` / `frame3d` for a stocky member, where the
    /// shear compliance matters. Reaching Bernoulli by making the shear area
    /// huge would work in exact arithmetic and lock in floating point, which is
    /// why this is a physics of its own rather than a limiting case.
    #[classmethod]
    fn bernoulli(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
        model: &str,
    ) -> PyResult<Self> {
        let m = BeamModel::from_tag(model).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "bernoulli: unknown model '{model}' (expected {})",
                BeamModel::tag_list()
            ))
        })?;
        Ok(Self {
            inner: Model::bernoulli(&fespace.inner, m)?,
        })
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
