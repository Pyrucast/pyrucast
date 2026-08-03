//! Python wrappers for [`crate::ops::element_field`] — the operators that
//! produce an `ElementField`.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::model::{PyModel, PySubModel};
use crate::py::node_field::PyNodeField;
use crate::store::{insert, read};
use pyo3::prelude::*;

/// Fuse the zones of an element `field` sharing the same `FiniteElementSpace`
/// support into a single zone carrying the union of their components.
///
/// The counterpart of `|`, which leaves component-disjoint zones side by side:
/// this is how per-physics material zones built on one shared fespace become a
/// single material field readable by every physics. Components carried by two
/// zones must agree value by value, else it errors. `field` itself is left
/// untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_element")]
pub fn consolidate(field: PyRef<PyElementField>) -> PyResult<PyElementField> {
    Ok(PyElementField {
        inner: crate::ops::element_field::consolidate(&field.inner)?,
    })
}

/// Gradient `∇f` of a node `field` at the Gauss points of `fespace`.
///
/// Geometric and physics-agnostic: each component of `field` is
/// differentiated w.r.t. every spatial axis, giving an `ElementField` with
/// one component `grad_<name>_<axis>` per (input component, axis) pair
/// (`grad_T_x`, …). Feed the result to `integrate_behavior`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn gradient(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::gradient(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Linearized (small-strain) deformation `ε = ½(∇u + ∇uᵀ)` of a displacement
/// field `u` at the Gauss points of `fespace`.
///
/// `u` must carry exactly `space_dim` components, taken in order as the
/// displacement along x, y, z. Returns the symmetric strain tensor
/// (`eps_xx`, `eps_xy`, … in tensor convention). The only deformation
/// measure for now; non-linear ones will share this shape.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn deformation(
    u: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::deformation(&u.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Interpolate a nodal `field` to the Gauss points of `fespace`
/// (`f(ξ_g) = Σ_i f_i N_i(ξ_g)`), turning a per-node `NodeField` into a
/// per-element `ElementField` with the same component names. Cast3M `CHAN`
/// (nodes → Gauss).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn interp_to_gauss(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::interp_to_gauss(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Thermal (free-dilation) strain `ε_th = α·(T − t_ref)` at the Gauss points of
/// `fespace` — Cast3M `EPTH`. `temperature` is a per-element field carrying
/// `"T"` (e.g. from `interp_to_gauss`); `materials` carries `"alpha"` (supplied
/// via `material_field`). Returns the strain tensor in the same layout as
/// `deformation`, so `deformation(u, fespace) - thermal_strain(...)` is the
/// mechanical strain. Backbone of uncoupled thermomechanics: assemble the
/// thermal load with `internal_forces(integrate_behavior(model, ε_th,
/// materials))` and recover `σ = D:(ε − ε_th)`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn thermal_strain(
    temperature: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
    t_ref: f64,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::thermal_strain(
        &temperature.inner,
        &materials.inner,
        &fespace.inner,
        t_ref,
    )?;
    Ok(PyElementField { inner: ef })
}

/// Generalised section strains of an **oriented** `SEG2` frame element at the
/// Gauss points — the co-rotational counterpart of `beam_deformation` for the
/// `frame` (2-D) and `frame3d` (3-D) physics.
///
/// Where `beam_deformation` expects a 1-D `(w, theta)` beam already aligned
/// with its axis, a frame element sits arbitrarily in space and carries the
/// full displacement + rotation at each node: this operator builds the
/// element's local axes, rotates the nodal triples into them, then evaluates
/// the section strains from the local DOFs. Components are `eps, kappa, gamma`
/// in 2-D and `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z` in 3-D.
///
/// Feed the result to `integrate_behavior` to obtain the section forces
/// (`N = E·A·eps`, `M = E·I·kappa`, `V = G·A_s·gamma`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn frame_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::frame_deformation(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Timoshenko-beam section strains `(kappa, gamma)` of a `(w, theta)` node
/// field at the Gauss points of `fespace`. Feed the result to
/// `integrate_behavior` of a Timoshenko model to obtain the section forces
/// `M = E·I·κ` and `V = G·A_s·γ`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn beam_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::beam_deformation(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Build the material `SubElementField` of one sub-model.
///
/// `sub_material_field(sub_model, [("k", 1.0), ...])` — fresh
/// SubElementField on the sub-model's FE subspace, pre-filled with the
/// given uniform value per declared component. Errors for physics that
/// need no material (e.g. Dirichlet).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn sub_material_field(
    sub_model: PyRef<PySubModel>,
    components_and_values: Vec<(String, f64)>,
) -> PyResult<PySubElementField> {
    let pairs: Vec<(&str, f64)> = components_and_values
        .iter()
        .map(|(c, v)| (c.as_str(), *v))
        .collect();
    let sub = crate::ops::element_field::sub_material_field(&*read(&sub_model.handle)?, &pairs)?;
    Ok(PySubElementField {
        handle: insert(sub),
    })
}

/// Build a material `ElementField` applying the same uniform
/// `(component, value)` pairs to every material-hungry sub-model of
/// `model`. Sub-models that need no material (Dirichlet, …) are skipped.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn material_field(
    model: PyRef<PyModel>,
    components_and_values: Vec<(String, f64)>,
) -> PyResult<PyElementField> {
    let pairs: Vec<(&str, f64)> = components_and_values
        .iter()
        .map(|(c, v)| (c.as_str(), *v))
        .collect();
    let ef = crate::ops::element_field::material_field(&model.inner, &pairs)?;
    Ok(PyElementField { inner: ef })
}

/// Build a material `ElementField` where each sub-model gets its own
/// `(component, value)` list. The outer list length must equal
/// `model.len()`. An empty inner list **skips** the matching
/// sub-model (typical for Dirichlet).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn material_field_per_sub_model(
    model: PyRef<PyModel>,
    components_and_values_per_sub_model: Vec<Vec<(String, f64)>>,
) -> PyResult<PyElementField> {
    // Materialise each inner Vec<(String, f64)> into a Vec<(&str, f64)>,
    // then collect slices into a Vec<&[(&str, f64)]>.
    let owned: Vec<Vec<(&str, f64)>> = components_and_values_per_sub_model
        .iter()
        .map(|v| v.iter().map(|(c, x)| (c.as_str(), *x)).collect())
        .collect();
    let slices: Vec<&[(&str, f64)]> = owned.iter().map(|v| v.as_slice()).collect();
    let ef = crate::ops::element_field::material_field_per_sub_model(&model.inner, &slices)?;
    Ok(PyElementField { inner: ef })
}

/// Integrate the constitutive law of `model` (Cast3m `COMP`), stepping A → B.
///
/// `deformation` is the **end-of-step** behaviour input ε(B) (from
/// `gradient(field, fespace)` or `deformation(u, fespace)`); `prev` is the
/// **converged output of the previous step** (the state at A — stress,
/// internal variables and start-of-step strain), or `None` on the first step;
/// `materials` supplies the per-zone material data; `dt` is the time increment
/// (`None` if the law is rate-independent). Returns the material-state field at
/// B (dual flux/stress + updated internal variables) of every behaviour-bearing
/// sub-model — feed it back as `prev` at the next step.
///
/// For a linear law the result is consistent with the assembled stiffness
/// (`∫ Bᵀ·flux = K·u`); a non-linear law is the exact response.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, deformation, materials, prev=None, dt=None))]
pub fn integrate_behavior(
    model: PyRef<PyModel>,
    deformation: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    prev: Option<PyRef<PyElementField>>,
    dt: Option<f64>,
) -> PyResult<PyElementField> {
    let prev_inner = prev.as_ref().map(|p| &p.inner);
    let ef = crate::ops::element_field::behavior::integrate(
        &model.inner,
        &deformation.inner,
        prev_inner,
        &materials.inner,
        dt,
    )?;
    Ok(PyElementField { inner: ef })
}

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs ci-dessus (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Aucune logique : chaque méthode rappelle la
// fonction libre, receveur compris.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// Voir `pyrucast.element_field.consolidate`.
    fn consolidate(slf: PyRef<'_, Self>) -> PyResult<PyElementField> {
        super::element_field::consolidate(slf)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// Voir `pyrucast.element_field.gradient`.
    fn gradient(
        slf: PyRef<'_, Self>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<PyElementField> {
        super::element_field::gradient(slf, fespace)
    }

    /// Voir `pyrucast.element_field.interp_to_gauss`.
    fn interp_to_gauss(
        slf: PyRef<'_, Self>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<PyElementField> {
        super::element_field::interp_to_gauss(slf, fespace)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// Voir `pyrucast.element_field.material_field`.
    fn material_field(
        slf: PyRef<'_, Self>,
        components_and_values: Vec<(String, f64)>,
    ) -> PyResult<PyElementField> {
        super::element_field::material_field(slf, components_and_values)
    }

    /// Voir `pyrucast.element_field.material_field_per_sub_model`.
    fn material_field_per_sub_model(
        slf: PyRef<'_, Self>,
        components_and_values_per_sub_model: Vec<Vec<(String, f64)>>,
    ) -> PyResult<PyElementField> {
        super::element_field::material_field_per_sub_model(slf, components_and_values_per_sub_model)
    }

    /// Voir `pyrucast.element_field.integrate_behavior`.
    #[pyo3(signature = (deformation, materials, prev=None, dt=None))]
    fn integrate_behavior(
        slf: PyRef<'_, Self>,
        deformation: PyRef<PyElementField>,
        materials: PyRef<PyElementField>,
        prev: Option<PyRef<PyElementField>>,
        dt: Option<f64>,
    ) -> PyResult<PyElementField> {
        super::element_field::integrate_behavior(slf, deformation, materials, prev, dt)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubModel {
    /// Voir `pyrucast.element_field.sub_material_field`.
    fn material_field(
        slf: PyRef<'_, Self>,
        components_and_values: Vec<(String, f64)>,
    ) -> PyResult<PySubElementField> {
        super::element_field::sub_material_field(slf, components_and_values)
    }
}
