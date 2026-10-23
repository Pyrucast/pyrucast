//! Python wrappers for [`crate::ops::element_field`] — the operators that
//! produce an `ElementField`.

use crate::handle::Handle;
use crate::ops::element_field::{StrainMeasure, StressMeasure};
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::model::{PyModel, PySubModel};
use crate::py::node_field::PyNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// Fuse the zones of an element `field` sharing the same `FiniteElementSpace`
/// support into a single zone carrying the union of their components.
///
/// The counterpart of `|`, which leaves component-disjoint zones side by side:
/// this is how per-physics material zones built on one shared fespace become a
/// single material field readable by every physics. Components carried by two
/// zones must agree value by value, else it errors. `field` itself is left
/// untouched.
#[py_op(method_on = PyElementField)]
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
#[py_op(method_on = PyNodeField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn gradient(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::gradient(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Deformation of a displacement field `u` at the Gauss points of `fespace`.
///
/// `u` must carry exactly `space_dim` components, taken in order as the
/// displacement along x, y, z. `measure` chooses the deformation, with
/// `H = ∇u` and `F = I + H`:
///
/// | measure | formula | components |
/// |---|---|---|
/// | `"linear"` (default) | `ε = ½(H + Hᵀ)` | `eps_xx`, `eps_xy`, … (tensor convention) |
/// | `"gradient"` (`"F"`) | `F = I + H` | every `F_ij`, row-major |
/// | `"green_lagrange"` | `E = ½(FᵀF − I)` | `E_xx`, `E_xy`, … |
/// | `"right_cauchy_green"` (`"C"`) | `C = FᵀF` | `C_xx`, … |
/// | `"left_cauchy_green"` (`"b"`) | `b = FFᵀ` | `b_xx`, … |
/// | `"hencky"` (`"log"`) | `½ ln C` | `H_xx`, … |
/// | `"euler_almansi"` (`"almansi"`) | `e = ½(I − b⁻¹)` | `e_xx`, … |
///
/// On an axisymmetric space, the hoop entry `<prefix>_zz` is appended
/// (`H_θθ = u_r / r`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (u, fespace, measure=StrainMeasure::Linear))]
pub fn deformation(
    u: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
    measure: StrainMeasure,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::deformation(&u.inner, &fespace.inner, measure)?;
    Ok(PyElementField { inner: ef })
}

/// Interpolate a nodal `field` to the Gauss points of `fespace`
/// (`f(ξ_g) = Σ_i f_i N_i(ξ_g)`), turning a per-node `NodeField` into a
/// per-element `ElementField` with the same component names. Cast3M `CHAN`
/// (nodes → Gauss).
#[py_op(method_on = PyNodeField)]
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
///
/// `measure` is the strain the law reads (`"linear"`, `"green_lagrange"`,
/// `"hencky"`): the dilation of stretch `1 + α·ΔT` is given exactly in it —
/// `α·ΔT`, `½((1 + α·ΔT)² − 1)`, `ln(1 + α·ΔT)` — under its names (`eps_*`,
/// `E_*`, `H_*`).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (temperature, materials, fespace, t_ref, measure=StrainMeasure::Linear))]
pub fn thermal_strain(
    temperature: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
    t_ref: f64,
    measure: StrainMeasure,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::thermal_strain(
        &temperature.inner,
        &materials.inner,
        &fespace.inner,
        t_ref,
        measure,
    )?;
    Ok(PyElementField { inner: ef })
}

/// `stress` converted to the measure `target`, point by point, with the
/// deformation gradient `gradient` — the field `deformation(u, fespace,
/// "gradient")` produces (`F_xx, F_xy, …`).
///
/// `target` is `"cauchy"` (`sigma_*`, the true stress), `"kirchhoff"`
/// (`tau_* = J sigma_*`), `"pk1"` (`P_*`, the nominal stress `F S`, not
/// symmetric) or `"pk2"` (`S_*`, dual of Green–Lagrange). The measure of
/// `stress` is recognised by its names, one per zone. This is how the
/// Lagrangian stress a law writes in large transformations becomes the first
/// Piola–Kirchhoff `internal_forces` reads, or the Cauchy stress of the
/// results — the stress half of Cast3M `CONF`.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn stress_measure(
    stress: PyRef<PyElementField>,
    gradient: PyRef<PyElementField>,
    target: StressMeasure,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::stress_measure(&stress.inner, &gradient.inner, target)?;
    Ok(PyElementField { inner: ef })
}

/// Elastic stress `σ = D:ε` of a strain field at the Gauss points of every
/// continuum sub-model of `model` — Cast3M `ELAS` applied to a strain.
///
/// `D` is the operator each law predicts with: the Hooke matrix of an
/// elasticity, in its symmetry; the isotropic `D` behind a plasticity's elastic
/// predictor; the **undamaged** `D` a damage law degrades. `strain` is in
/// tensor convention, the layout of `deformation` and `thermal_strain`;
/// `materials` needs only each law's elastic constants (`E`, `nu` when
/// isotropic). One zone per continuum sub-model, its components the stresses
/// the law writes; a sub-model with no elastic operator (conduction, bar, beam,
/// constraint) contributes none.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn elastic_stress(
    model: PyRef<PyModel>,
    strain: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef =
        crate::ops::element_field::elastic_stress(&model.inner, &strain.inner, &materials.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Elastic strain `ε = D⁻¹:σ` of a stress field at the Gauss points of every
/// continuum sub-model of `model` — the inverse of `elastic_stress` (Cast3M
/// `ELAS` applied to a stress).
///
/// `stress` carries the stresses each law writes (`sigma_xx`, …; the state
/// returned by `integrate_behavior` will do); `materials` the elastic
/// constants. The result is in tensor convention, the layout of `deformation`,
/// so it subtracts from a strain component for component; in plane stress the
/// out-of-plane `eps_zz` is not written. Raises when `D` is singular at a point
/// (a zero modulus).
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn elastic_strain(
    model: PyRef<PyModel>,
    stress: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef =
        crate::ops::element_field::elastic_strain(&model.inner, &stress.inner, &materials.inner)?;
    Ok(PyElementField { inner: ef })
}

/// The input of every law of `model`, produced from the nodal `solution` — the
/// model's `B`, of which `internal_forces` is the transpose (Cast3M `EPSI`,
/// extended to every physics).
///
/// One zone per sub-model with a law, on its behaviour space, carrying what
/// that law reads: the small-strain tensor `eps_*` of a continuum or a bar, the
/// gradient `grad_T_*` of conduction, `T` itself for radiation, the section
/// strains of a beam, the generalized strains of a shell. Each law reads its
/// own unknowns of `solution` by name, so one solution — displacement,
/// temperature, multipliers — serves the whole model; sub-models without a law
/// produce nothing. `materials` is read by the beams alone (their `Φ`). The
/// result feeds `integrate_behavior` as is.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn strain(
    model: PyRef<PyModel>,
    solution: PyRef<PyNodeField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::strain(&model.inner, &solution.inner, &materials.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Integral of every component of `field` over **each cell** (Cast3M
/// `INTG 'ELEM'`), with the field's own quadrature: `Σ_g f(g)·|J|·w_g`, the
/// `2πr` included on a body of revolution. The result lives on `target`, a
/// space of the same mesh with **one point per cell** —
/// `FiniteElementSpace(mesh, quadrature="REDUCED")`, a single point at each
/// centroid. Integrating `1` gives every cell's measure. Raises when `target`
/// has several points per cell or does not cover the field's mesh.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn cell_integral(
    field: PyRef<PyElementField>,
    target: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::cell_integral(&field.inner, &target.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Generalised **section strains** of a beam at the Gauss points, in whichever
/// configuration the mesh puts it — the geometric producer of the behaviour
/// input for `timoshenko` and `bernoulli`.
///
/// | `Coords` | DOFs read | components produced |
/// |---|---|---|
/// | 1-D | `w, theta` | `kappa, gamma` |
/// | 2-D | `u_x, u_y, r_z` | `eps, kappa, gamma` |
/// | 3-D | six | `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z` |
///
/// For an oriented element it builds the local axes, rotates the nodal triples
/// into them, then evaluates the strains from the local DOFs. This replaces
/// `frame_deformation`, which was the same operator for the 2-D and 3-D cases.
///
/// The **material** is required, and that is the honest signature: the
/// element's interpolation depends on `Φ = 12EI/(G·A_s·L²)`, so its curvature
/// distribution cannot be recovered without knowing the shear stiffness. An
/// operator that took no material could only ever report a mean.
///
/// Feed the result to `integrate_behavior` to obtain the section forces
/// (`N = E·A·eps`, `M = E·I·kappa`, `V = G·A_s·gamma`). The moment then varies
/// along the element, as `M' = V` requires, while the shear stays constant.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn beam_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
    material: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef =
        crate::ops::element_field::beam_deformation(&field.inner, &fespace.inner, &material.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Generalised **strains of a shell** at the Gauss points — the geometric
/// producer of the behaviour input for `shell`.
///
/// `shell_deformation(field, fespace, model)` with `model` the formulation
/// (`"thick"` or `"kirchhoff"`), which decides the bending rows and whether a
/// transverse shear is reported at all:
///
/// | formulation | components produced |
/// |---|---|
/// | `thick` | `eps_xx…eps_xy`, `kappa_xx…kappa_xy`, `drill`, `gamma_xz, gamma_yz` |
/// | `kirchhoff` | the same, without the two shear strains |
///
/// All in the element's **local** frame. The transverse shear is
/// element-constant, sampled at the reduced point the stiffness integrates it
/// at — anywhere else would report a strain the element does not carry.
///
/// Feed the result to `integrate_behavior` to obtain the generalised forces
/// (`N = D_m·eps`, `M = D_b·kappa`, `Q = D_s·gamma`, and the drilling moment).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn shell_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
    model: &str,
) -> PyResult<PyElementField> {
    let model = <crate::models::shell::ShellModel as crate::named::Named>::parse(model)?;
    let ef = crate::ops::element_field::shell_deformation(&field.inner, &fespace.inner, model)?;
    Ok(PyElementField { inner: ef })
}

/// Build the material `SubElementField` of one sub-model.
///
/// `sub_material_field(sub_model, [("k", 1.0), ...])` — fresh
/// SubElementField on the sub-model's FE subspace, pre-filled with the
/// given uniform value per declared component. Errors for physics that
/// need no material (e.g. Dirichlet).
#[py_op(method_on = PySubModel, name = "material_field")]
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
    let sub = crate::ops::element_field::sub_material_field(&sub_model.handle.read(), &pairs)?;
    Ok(PySubElementField {
        handle: Handle::new(sub),
    })
}

/// Build a material `ElementField` applying the same uniform
/// `(component, value)` pairs to every material-hungry sub-model of
/// `model`. Sub-models that need no material (Dirichlet, …) are skipped.
#[py_op(method_on = PyModel)]
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
#[py_op(method_on = PyModel)]
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
#[py_op(method_on = PyModel)]
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

/// Per-component 0/1 **mask** of a field against a value band — same structure
/// as the input (Cast3M `MASQUE`): same zones, same support, same components,
/// only the values are rewritten (`1.0` inside the band, `0.0` outside). The
/// result is therefore multipliable term by term with the input.
///
/// `field` is an `ElementField` or a `SubElementField`, and the result is of
/// the same kind. Its sibling `pyrucast.mesh.select` extracts the passing
/// *support* instead, and produces a `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product), and a zone missing a listed component is left all-`1.0`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "mask_element")]
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
    if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        Ok(Py::new(py, mask_element_field(f, ge, gt, le, lt, components)?)?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        Ok(Py::new(py, mask_sub_element_field(f, ge, gt, le, lt, components)?)?.into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a PyElementField or PySubElementField",
        ))
    }
}

// The two flavours of `mask`: same setup as in `node_field` — unregistered, they
// serve only to give `#[py_op]` a method at the precise type, with its own
// documentation.

/// Per-component 0/1 **mask** of this field against a value band — same
/// structure as the field (Cast3M `MASQUE`): same zones, same support, same
/// components, same Gauss points, only the values are rewritten (`1.0` inside
/// the band, `0.0` outside). The result is therefore multipliable term by term
/// with the field.
///
/// Its sibling `select` extracts the passing *cells* instead, and produces a
/// `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product), and a zone missing a listed component is left all-`1.0`.
#[py_op(method_on = PyElementField, name = "mask")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn mask_element_field(
    field: PyRef<PyElementField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyElementField> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    Ok(PyElementField {
        inner: crate::ops::element_field::mask(&field.inner, &band, components)?,
    })
}

/// Per-component 0/1 **mask** of this sub-field against a value band — same
/// structure as the sub-field (Cast3M `MASQUE`): same support, same
/// components, same Gauss points, only the values are rewritten (`1.0` inside
/// the band, `0.0` outside). The result is therefore multipliable term by term
/// with the sub-field.
///
/// Its sibling `select` extracts the passing *cells* instead, and produces a
/// `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product). A listed component the sub-field does not carry leaves it
/// all-`1.0`.
#[py_op(method_on = PySubElementField, name = "mask")]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
pub fn mask_sub_element_field(
    field: PyRef<PySubElementField>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PySubElementField> {
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    let out = crate::ops::element_field::mask_sub(&field.handle.read(), &band, components);
    Ok(PySubElementField {
        handle: Handle::new(out),
    })
}
