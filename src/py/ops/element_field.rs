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

/// Fuse the zones of `field` that lie on the same FE subspace into one.
///
/// The counterpart of `|`, which leaves zones with distinct components side by
/// side and refuses two zones carrying the same component on one subspace. This
/// is how material zones built physics by physics on one shared FE space (`k`
/// for heat, `E`, `nu`, `alpha` for mechanics) become a single material field
/// that every physics reads.
///
/// # Parameters
///
/// - `field`: the field to consolidate.
///
/// # Returns
///
/// A new `ElementField`, one zone per FE subspace carrying the union of the
/// components; `field` itself is left untouched.
///
/// # Raises
///
/// `RuntimeError` if two zones on one subspace carry the same component.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_element")]
pub fn consolidate(field: PyRef<PyElementField>) -> PyResult<PyElementField> {
    Ok(PyElementField {
        inner: crate::ops::element_field::consolidate(&field.inner)?,
    })
}

/// Restrict `field` to the cells of `fespace`, with every component of `field`.
///
/// Same cells and same finite-element definition: a cell of `fespace` takes the values of `field` at its Gauss points when a
/// zone of `field` holds that cell (same nodes in the same order) under the
/// same definition: same element type, interpolation and quadrature. A zone of
/// another definition is not read. To move a field to another definition,
/// interpolate it from the nodes (`interp_to_gauss`) or integrate it cell by
/// cell (`cell_integral`). The part of a field on a mesh is
/// `restrict(f, f.support().restrict(part))`.
///
/// # Parameters
///
/// - `field`: the field to read.
/// - `fespace`: the subspaces the result lies on.
///
/// # Returns
///
/// A new `ElementField` with one zone per subspace of `fespace`, each subspace
/// once, each carrying the union of the components of `field`. A cell no zone
/// of `field` gives gets `0.0`; for each component, the first zone holding the
/// cell and carrying it wins. An empty field gives an empty field.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "restrict_element")]
pub fn restrict(
    field: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyElementField {
    PyElementField {
        inner: crate::ops::element_field::restrict(&field.inner, &fespace.inner),
    }
}

/// Read `field` on the subspaces **and the components** of `target`.
///
/// Same cells and same finite-element definition. Zone by zone, it is `restrict(field, target.support())` cut down to the
/// components of that zone of `target`: those `target` lacks are dropped,
/// those `field` lacks are created at `0.0`. The result combines with `target`
/// directly by the arithmetic operators (`target + restrict_like(f, target)`),
/// even when `field` lies on other subspaces of the same cells.
///
/// # Parameters
///
/// - `field`: the field to read.
/// - `target`: the field whose zones, subspaces and components the result
///   takes.
///
/// # Returns
///
/// A new `ElementField` zone for zone like `target`. A cell takes the values
/// of `field` only from a zone of the same definition holding it, `0.0`
/// otherwise. An empty field gives an empty field.
#[py_op(method_on = PyElementField)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "restrict_like_element")]
pub fn restrict_like(
    field: PyRef<PyElementField>,
    target: PyRef<PyElementField>,
) -> PyElementField {
    PyElementField {
        inner: crate::ops::element_field::restrict_like(&field.inner, &target.inner),
    }
}

/// Gradient `∇f` of a field at the nodes, at the Gauss points of `fespace`.
///
/// `∇f = Σ_i f_i ∇N_i`, evaluated at each Gauss point, cell by cell. The
/// operator is geometric and knows no physics; its result feeds
/// `pyrucast.element_field.integrate_behavior` (heat conduction reads
/// `grad_T_*`). A node the field does not define reads zero.
///
/// # Parameters
///
/// - `field`: the field to differentiate, at the nodes.
/// - `fespace`: the FE space whose cells and Gauss points to use.
///
/// # Returns
///
/// An `ElementField` on `fespace`, with one component `grad_<name>_<axis>` per
/// pair of input component and spatial axis (`grad_T_x`, `grad_T_y`, …).
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
/// With `H = ∇u` and `F = I + H`, the measure is one of:
///
/// | measure | formula | components |
/// |---|---|---|
/// | `"linear"` (default) | `ε = ½(H + Hᵀ)` | `eps_xx`, `eps_xy`, … (tensor convention) |
/// | `"gradient"` (`"F"`) | `F = I + H` | every `F_ij`, row by row |
/// | `"green_lagrange"` | `E = ½(FᵀF − I)` | `E_xx`, `E_xy`, … |
/// | `"right_cauchy_green"` (`"C"`) | `C = FᵀF` | `C_xx`, … |
/// | `"left_cauchy_green"` (`"b"`) | `b = FFᵀ` | `b_xx`, … |
/// | `"hencky"` (`"log"`) | `½ ln C` | `H_xx`, … |
/// | `"euler_almansi"` (`"almansi"`) | `e = ½(I − b⁻¹)` | `e_xx`, … |
/// | `"jacobian"` (`"J"`) | `J = det F` | `J` alone |
///
/// A symmetric measure gives one component `<prefix>_<ai><aj>` per independent
/// entry `i ≤ j`; the gradient `F`, not symmetric, gives them all. The linear
/// measure is in tensor convention, `eps_xy = ½(∂u_x/∂y + ∂u_y/∂x)`, not the
/// engineering shear `γ`. In 2-D the out-of-plane direction is not deformed and
/// its entries are not produced.
///
/// # Parameters
///
/// - `u`: the displacement, at the nodes; exactly one component per spatial
///   axis, taken in order as the displacement along x, y, z.
/// - `fespace`: the FE space whose cells and Gauss points to use.
/// - `measure`: the measure to compute, a name or its alias from the table.
///
/// # Returns
///
/// An `ElementField` on `fespace` with the components of the measure. On an
/// axisymmetric space, the hoop entry `<prefix>_zz`, built on `H_θθ = u_r / r`,
/// is appended.
///
/// # Raises
///
/// - `RuntimeError` if `u` does not carry one component per spatial axis.
/// - `ValueError` for an unknown `measure`.
///
/// # Notes
///
/// Nothing is checked at the Gauss point: a degenerate transformation
/// (`det F = 0`) gives non-finite values for Hencky and Almansi. Only
/// `"jacobian"` sees an inverted cell, where `J ≤ 0` at one of its points:
/// `C = FᵀF` stays positive definite when `F` changes sign.
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

/// Interpolate a field at the nodes to the Gauss points of `fespace`.
///
/// `f(ξ_g) = Σ_i f_i N_i(ξ_g)`: values, not derivatives — typically to carry a
/// nodal temperature to the Gauss points before
/// `pyrucast.element_field.thermal_strain`. A node the field does not define
/// reads zero.
///
/// # Parameters
///
/// - `field`: the field to interpolate, at the nodes.
/// - `fespace`: the FE space whose cells and Gauss points to use.
///
/// # Returns
///
/// An `ElementField` on `fespace` with the components of `field`, one value per
/// cell and Gauss point.
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

/// Thermal strain `ε_th = α·(T − t_ref)` at the Gauss points of `fespace`.
///
/// The strain of free dilation, for uncoupled thermomechanics. Its layout is the
/// one of `pyrucast.element_field.deformation` — normal components `α·ΔT`, zero
/// shears, the hoop `eps_zz` included on a body of revolution —, so
/// `deformation(u, fespace) - thermal_strain(...)` is the mechanical strain
/// `ε(u) − ε_th`. No coupling is done here: the thermal load and the stress
/// `σ = D:(ε − ε_th)` are built from the other operators.
///
/// # Parameters
///
/// - `temperature`: a field at the Gauss points carrying `"T"`, for instance
///   from `pyrucast.element_field.interp_to_gauss`.
/// - `materials`: the material field; `alpha` is read on `fespace`, an optional
///   component of the elasticity next to `E` and `nu`.
/// - `fespace`: the FE space whose Gauss points to use.
/// - `t_ref`: the reference temperature, where the strain is zero.
/// - `measure`: the strain the law reads, `"linear"`, `"green_lagrange"` or
///   `"hencky"`. The dilation of stretch `1 + α·ΔT` is given exactly in it —
///   `α·ΔT`, `½((1 + α·ΔT)² − 1)`, `ln(1 + α·ΔT)` — under its names (`eps_*`,
///   `E_*`, `H_*`).
///
/// # Returns
///
/// An `ElementField` on `fespace`, the strain tensor under the names of
/// `measure`.
///
/// # Raises
///
/// - `RuntimeError` if `materials` has no zone on `fespace` carrying `alpha`, if
///   `temperature` lacks `"T"`, or for a `measure` that is not a strain a law
///   reads (`"gradient"`, `"jacobian"`, …).
/// - `ValueError` for an unknown `measure`.
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

/// Convert `stress` to the measure `target`, point by point.
///
/// This is how the Lagrangian stress a law writes in large strains becomes the
/// first Piola–Kirchhoff stress `pyrucast.node_field.internal_forces` reads, or
/// the Cauchy stress of the results. A bar is recognised by its axial second
/// Piola–Kirchhoff force `S_n`: `"pk1"` turns it into the nominal force vector
/// `P_x, P_y(, P_z) = S_n F t₀`, `"cauchy"` into the true force `n = λ S_n`.
///
/// # Parameters
///
/// - `stress`: the stress to convert. Its measure is recognised by its names,
///   one per zone: `sigma_*`, `tau_*`, `P_*`, `S_*` or `T_*`.
/// - `gradient`: the deformation gradient `F_xx, F_xy, …` at the same points, as
///   `pyrucast.element_field.deformation(u, fespace, "gradient")` produces it.
/// - `target`: the measure wanted:
///   - `"cauchy"`: `sigma_*`, the true stress;
///   - `"kirchhoff"`: `tau_* = J sigma_*`;
///   - `"pk1"`: `P_*`, the nominal stress `F S`, not symmetric;
///   - `"pk2"`: `S_*`, dual of Green–Lagrange;
///   - `"hencky"`: `T_*`, dual of the Hencky strain, what a law reading Hencky
///     writes.
///
/// # Returns
///
/// An `ElementField` with the components of `target`.
///
/// # Raises
///
/// - `RuntimeError` if `gradient` lacks a component `F_*`.
/// - `ValueError` for an unknown `target`.
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

/// Turn the material axes of `field` with the part, by the rotation of `F`.
///
/// The axes `V1*` (and `V2*` in 3-D) are turned by the rotation `R` of the
/// polar decomposition `F = R U` at each point, `v = R v₀`, hence `k = R k₀ Rᵀ`.
/// Every other component is copied as it is: the material is the same, only its
/// axes follow the rotation. This is what a conduction or a diffusion computed on
/// the deformed part reads.
///
/// # Parameters
///
/// - `field`: the material field at the Gauss points.
/// - `gradient`: the deformation gradient `F_*`, as
///   `pyrucast.element_field.deformation(u, fespace, "gradient")` produces it on
///   the **reference** configuration.
///
/// # Returns
///
/// A new `ElementField`. A zone without axes (an isotropic material) comes back
/// as it is, shared.
///
/// # Raises
///
/// `RuntimeError` if a zone carries part of the axes only, or if `gradient` has
/// no zone on the FE space of a zone of `field`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn rotate_frame(
    field: PyRef<PyElementField>,
    gradient: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::rotate_frame(&field.inner, &gradient.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Elastic stress `σ = D:ε` of `strain`, for each continuum sub-model of `model`.
///
/// `D` is the operator each law predicts with, not the response of the law
/// itself: the Hooke matrix of an elasticity, in its symmetry; the isotropic `D`
/// behind a plasticity's elastic predictor; the **undamaged** `D` a damage law
/// degrades. It turns an imposed strain increment (thermal, given) into the
/// stress it would cause if nothing yielded.
///
/// # Parameters
///
/// - `model`: the sub-models whose elastic operator to apply.
/// - `strain`: the strain at the Gauss points, in tensor convention — the layout
///   of `pyrucast.element_field.deformation` and
///   `pyrucast.element_field.thermal_strain`.
/// - `materials`: the material field; only each law's elastic constants are read
///   (`E`, `nu` when isotropic).
///
/// # Returns
///
/// An `ElementField` with one zone per continuum sub-model, its components the
/// stresses the law writes. A sub-model with no elastic operator (conduction,
/// bar, beam, constraint) contributes none.
///
/// # Raises
///
/// `RuntimeError` if `materials` lacks an elastic constant of a law.
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

/// Elastic strain `ε = D⁻¹:σ` of `stress`, for each continuum sub-model of `model`.
///
/// The inverse of `pyrucast.element_field.elastic_stress`: it brings a stress
/// back to the elastic strain that carries it, for instance under other moduli
/// when the material varies with the temperature.
///
/// # Parameters
///
/// - `model`: the sub-models whose elastic operator to invert.
/// - `stress`: the stresses each law writes (`sigma_xx`, …); the state returned
///   by `pyrucast.element_field.integrate_behavior` will do.
/// - `materials`: the material field; only each law's elastic constants are read.
///
/// # Returns
///
/// An `ElementField` in tensor convention, the layout of
/// `pyrucast.element_field.deformation`, so it subtracts from a strain component
/// for component. In plane stress the out-of-plane `eps_zz` is not written.
///
/// # Raises
///
/// `RuntimeError` if `D` is singular at a point (a zero modulus), or if
/// `materials` lacks an elastic constant of a law.
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

/// Equivalent stress of `stress`: von Mises, or Hill's form with `coefficients`.
///
/// `σ_eq² = F(σyy − σzz)² + G(σzz − σxx)² + H(σxx − σyy)² + 2Lσyz² + 2Mσxz² + 2Nσxy²`
///
/// The model says which components to read, as they depend on the kinematics:
///
/// | kinematics | components read | taken as zero |
/// |---|---|---|
/// | `full_3d` | the six `sigma_*` | — |
/// | `plane_stress` | `sigma_xx`, `sigma_yy`, `sigma_xy` | `zz`, `yz`, `xz` |
/// | `plane_strain` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy` | `yz`, `xz` |
/// | `axisymmetric` | `sigma_xx`, `sigma_yy`, `sigma_zz` (hoop), `sigma_xy` | `yz`, `xz` |
///
/// # Parameters
///
/// - `model`: the continuum sub-models whose kinematics say what to read.
/// - `stress`: the stresses the laws write (`sigma_xx`, …); the output of
///   `pyrucast.element_field.elastic_stress` or the state of
///   `pyrucast.element_field.integrate_behavior` will do, whatever else they
///   carry.
/// - `coefficients`: `[F, G, H, L, M, N]`; the default
///   `[0.5, 0.5, 0.5, 1.5, 1.5, 1.5]` is von Mises, `√(3 J₂)`.
///
/// # Returns
///
/// An `ElementField` with one zone per continuum sub-model, on its Gauss points,
/// and the single component `"von_mises"`.
///
/// # Raises
///
/// `RuntimeError` for coefficients that let the form go negative (each of `L`,
/// `M`, `N`, `F + G`, `G + H`, `H + F`, `FG + GH + HF` must be `≥ 0`), and for a
/// stress missing a component its kinematics reads — in plane strain, a stress
/// without `sigma_zz`, which plastic laws carry and an elastic stress does not.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, stress, coefficients=crate::ops::element_field::VON_MISES))]
pub fn von_mises(
    model: PyRef<PyModel>,
    stress: PyRef<PyElementField>,
    coefficients: [f64; 6],
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::von_mises(&model.inner, &stress.inner, &coefficients)?;
    Ok(PyElementField { inner: ef })
}

/// The input of every law of `model`, computed from the nodal `solution`.
///
/// The model's `B`, of which `pyrucast.node_field.internal_forces` is the
/// transpose. Each law declares which producer feeds it, applied to its own
/// unknowns on its own space:
///
/// | sub-model | producer | zone returned |
/// |---|---|---|
/// | elasticity, plasticity, damage, bar | `deformation` | `eps_*`, or the measure of the law (`E_*`, `H_*`) |
/// | conduction, Fick | `gradient` | `grad_T_*`, `grad_c_<species>_*` |
/// | radiation | `interp_to_gauss` | `T` at the points |
/// | Bernoulli and Timoshenko beams | `beam_deformation` | `eps`, `kappa*`, `gamma*`, `torsion` |
/// | shell | `shell_deformation` | `eps_*`, `kappa_*`, `drill`, `gamma_*` |
///
/// # Parameters
///
/// - `model`: the laws whose input to compute.
/// - `solution`: the current solution. Each law reads its own unknowns by name,
///   so one solution — displacement, temperature, multipliers — serves the whole
///   model; an unknown it lacks reads zero.
/// - `materials`: the material field, read by the beams alone (their
///   `Φ = 12EI/(G·A_s·L²)`).
///
/// # Returns
///
/// An `ElementField` with one zone per sub-model carrying a law, on its behaviour
/// space; sub-models without a law (constraints, loads, exchanges) produce
/// nothing. It feeds `pyrucast.element_field.integrate_behavior` as it is.
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

/// Integrate every component of `field` over each cell.
///
/// `v_e = Σ_g f(g)·|J|·w_g`, with the field's own quadrature, the `2πr`
/// included on a body of revolution. Integrating `1` gives the measure of every
/// cell; `pyrucast.node_field.sum_to_nodes` then gathers it at the nodes.
///
/// # Parameters
///
/// - `field`: the field to integrate, at the Gauss points.
/// - `target`: a space of the same mesh with **one point per cell** —
///   `FiniteElementSpace(mesh, quadrature="REDUCED")`, a single point at each
///   centroid.
///
/// # Returns
///
/// An `ElementField` on `target`, with the components of `field`.
///
/// # Raises
///
/// `RuntimeError` if `target` has several points per cell or does not cover the
/// mesh of `field`.
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

/// Generalized section strains of a beam at the Gauss points of `fespace`.
///
/// The product `B·u` with the element's own `B`, for Bernoulli and Timoshenko
/// beams alike. The configuration is read on the dimension of the `Coords`:
///
/// | `Coords` | unknowns read | components produced |
/// |---|---|---|
/// | 1-D | `w, theta` | `kappa, gamma` |
/// | 2-D | `u_x, u_y, r_z` | `eps, kappa, gamma` |
/// | 3-D | six | `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z` |
///
/// For an oriented element, the local axes are built and the nodal triples
/// turned into them before the strains are evaluated. The strains are evaluated
/// at each Gauss point: the curvature varies along an unloaded span (`M' = V`),
/// the shear stays constant. Fed to `pyrucast.element_field.integrate_behavior`,
/// they give the section forces (`N = E·A·eps`, `M = E·I·kappa`,
/// `V = G·A_s·gamma`).
///
/// # Parameters
///
/// - `field`: the displacements and rotations, at the nodes.
/// - `fespace`: the FE space of the beam, on SEG2 cells.
/// - `material`: the material field. It is required: the element's
///   interpolation depends on `Φ = 12EI/(G·A_s·L²)`. A material without shear
///   constants — a Bernoulli beam's — means `Φ = 0`, so one operator serves both
///   theories.
///
/// # Returns
///
/// An `ElementField` on `fespace` with the components of the table.
///
/// # Raises
///
/// `RuntimeError` if the cells of `fespace` are not SEG2.
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

/// Generalized strains of a shell at the Gauss points of `fespace`.
///
/// Read from the six unknowns `u_x…u_z, r_x…r_z`, all in the **local** frame of
/// the facet. `drill` is the drilling residual `θ_z − ω_z`. The transverse shear
/// is constant per element, sampled at the reduced point where the stiffness
/// integrates it: anywhere else would report a strain the element does not
/// carry. Fed to `pyrucast.element_field.integrate_behavior`, they give the
/// generalized forces (`N = D_m·eps`, `M = D_b·kappa`, `Q = D_s·gamma`, and the
/// drilling moment).
///
/// # Parameters
///
/// - `field`: the displacements and rotations, at the nodes.
/// - `fespace`: the FE space of the shell.
/// - `formulation`: `"thick"` or `"kirchhoff"`, which decides the bending rows
///   and whether a transverse shear is reported:
///
///   | formulation | components produced |
///   |---|---|
///   | `"thick"` | `eps_xx…eps_xy`, `kappa_xx…kappa_xy`, `drill`, `gamma_xz, gamma_yz` |
///   | `"kirchhoff"` | the same, without the two shear strains |
///
/// # Returns
///
/// An `ElementField` on `fespace` with the components of the table.
///
/// # Raises
///
/// `RuntimeError` for an unknown `formulation`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn shell_deformation(
    field: PyRef<PyNodeField>,
    fespace: PyRef<PyFiniteElementSpace>,
    formulation: &str,
) -> PyResult<PyElementField> {
    let formulation =
        <crate::models::shell::ShellFormulation as crate::named::Named>::parse(formulation)?;
    let ef =
        crate::ops::element_field::shell_deformation(&field.inner, &fespace.inner, formulation)?;
    Ok(PyElementField { inner: ef })
}

/// Build a uniform material field for one sub-model.
///
/// A fresh zone on the FE subspace of the sub-model, filled with the given
/// values.
///
/// # Parameters
///
/// - `sub_model`: the sub-model whose material to build.
/// - `components_and_values`: the `(component, value)` pairs, one uniform value
///   per component, for instance `[("E", 210e9), ("nu", 0.3)]`.
///
/// # Returns
///
/// A `SubElementField` on the FE subspace of `sub_model`.
///
/// # Raises
///
/// `RuntimeError` if a component the physics requires is missing, or for a
/// physics that needs no material (a constraint).
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

/// Build a uniform material field for every sub-model of `model` that needs one.
///
/// The same pairs are applied to every sub-model that consumes material; the
/// others (constraints) are skipped. The field is then given as it is to
/// `pyrucast.matrix.stiffness` and `pyrucast.element_field.integrate_behavior`,
/// which pick for each sub-model the zone on its FE subspace. Once built, it is
/// an ordinary `ElementField`, which can vary in space or be scaled.
///
/// # Parameters
///
/// - `model`: the sub-models whose material to build.
/// - `components_and_values`: the `(component, value)` pairs, one uniform value
///   per component, for instance `[("E", 210e9), ("nu", 0.3)]`.
///
/// # Returns
///
/// An `ElementField` with one zone per sub-model that consumes material.
///
/// # Raises
///
/// `RuntimeError` if a component a physics requires is missing.
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

/// Build a material field with its own uniform values for each sub-model.
///
/// Different materials zone by zone: the n-th list of pairs goes to the n-th
/// sub-model of `model`.
///
/// # Parameters
///
/// - `model`: the sub-models whose material to build.
/// - `components_and_values_per_sub_model`: one list of `(component, value)`
///   pairs per sub-model of `model`, in order; an empty list skips that
///   sub-model (a constraint).
///
/// # Returns
///
/// An `ElementField` with one zone per non-empty list.
///
/// # Raises
///
/// `RuntimeError` if the number of lists is not the number of sub-models, or if
/// a component a physics requires is missing.
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

/// Integrate the constitutive laws of `model` over a step from A to B.
///
/// Each law takes the converged state at the start of the step, A, and the
/// strain at its end, B, and returns the state at B, point by point. For a
/// linear law the result is consistent with the assembled stiffness
/// (`∫ Bᵀ·flux = K·u`); a non-linear law gives its exact response. Constraint
/// sub-models, which have no law, are skipped; the zones of `deformation`,
/// `materials` and `prev` are paired by FE subspace.
///
/// # Parameters
///
/// - `model`: the laws to integrate.
/// - `deformation`: the input of the laws at the end of the step, `ε(B)` —
///   from `pyrucast.element_field.strain`, or a producer such as
///   `pyrucast.element_field.deformation` or `pyrucast.element_field.gradient`.
/// - `materials`: the material field, zone by zone.
/// - `prev`: the state at A, the output of the previous step (stress, internal
///   variables, strain at A); on a first step,
///   `pyrucast.element_field.initial_state(model, materials)`.
/// - `dt`: the time increment; `None` when no law depends on the rate.
///
/// # Returns
///
/// The material state at B of every sub-model carrying a law: the dual flux or
/// stress and the updated internal variables — to give back as `prev` at the
/// next step.
///
/// # Raises
///
/// `RuntimeError` if a rate-dependent law (creep, viscoplasticity) gets no `dt`.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (model, deformation, materials, prev, dt=None))]
pub fn integrate_behavior(
    model: PyRef<PyModel>,
    deformation: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    prev: PyRef<PyElementField>,
    dt: Option<f64>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::behavior::integrate(
        &model.inner,
        &deformation.inner,
        &prev.inner,
        &materials.inner,
        dt,
    )?;
    Ok(PyElementField { inner: ef })
}

/// The rest state of `model`, the `prev` of a first step.
///
/// What each law carries before anything happened, to give as `prev` to
/// `pyrucast.element_field.integrate_behavior` and `pyrucast.matrix.tangent`.
/// Nearly every law rests at zero; a law whose internal variable starts from a
/// material constant seeds it from `materials` (Gurson's porosity from `f_0`).
///
/// # Parameters
///
/// - `model`: the laws whose rest state to build.
/// - `materials`: the material field, read by the laws that seed a variable.
///
/// # Returns
///
/// An `ElementField` with one zone per sub-model carrying a law, on its
/// behaviour subspace, with every component that law writes. Constraint
/// sub-models, which have no law, are skipped.
#[py_op(method_on = PyModel)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn initial_state(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyElementField> {
    let ef = crate::ops::element_field::behavior::initial_state(&model.inner, &materials.inner)?;
    Ok(PyElementField { inner: ef })
}

/// A 0/1 mask of `field` against a band of values, component by component.
///
/// The result has the structure of the input — same zones, same support, same
/// components, same Gauss points —, only the values are rewritten: `1.0` inside
/// the band, `0.0` outside. It therefore multiplies term by term with the input.
/// There is **no** AND across components: each value stands on its own. The
/// comparisons `>=`, `>`, `<=`, `<` of a field against a number build the same
/// mask. Its sibling `pyrucast.mesh.select` extracts the passing cells instead,
/// as a `Mesh`.
///
/// # Parameters
///
/// - `field`: an `ElementField` or a `SubElementField`.
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
/// - `TypeError` if `field` is neither an `ElementField` nor a `SubElementField`.
/// - `RuntimeError` if no bound is given, if both bounds of one side are, or if
///   the lower bound exceeds the upper one.
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
/// structure as the field: same zones, same support, same components, same
/// Gauss points, only the values are rewritten (`1.0` inside the band, `0.0`
/// outside). The result is therefore multipliable term by term with the field.
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
/// structure as the sub-field: same support, same components, same Gauss
/// points, only the values are rewritten (`1.0` inside the band, `0.0`
/// outside). The result is therefore multipliable term by term with the
/// sub-field.
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
