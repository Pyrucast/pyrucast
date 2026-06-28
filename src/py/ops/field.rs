//! Python wrappers for the field operations in [`crate::ops::field`].
//!
//! Free functions that build or transform a [`PyNodeField`]. Kept here —
//! mirroring `src/ops/field/` — rather than on the field classes, per
//! the `py/ops/` convention (operations live with operations).
//!
//! Transitional note: the Rust ops still work on a single
//! `SubNodeField`; this wrapper bridges through `Aggregate::unit()`
//! (unitary aggregates only) until the ops are generalised to the
//! `NodeField` aggregate.

use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::node_field::{PyNodeField, PySubNodeField};
use crate::store::{insert, read};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Build a `NodeField` carrying the coordinates of every node of `mesh`
/// — one `SubNodeField` per submesh, on the distinct nodes of its zone.
///
/// One component per requested axis (`"X"`, `"Y"`, `"Z"`). `components=None`
/// requests all the axes the mesh's `Coords` has (`["X"]` in 1-D,
/// `["X", "Y"]` in 2-D, `["X", "Y", "Z"]` in 3-D).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn coordinates(
    mesh: PyRef<PyMesh>,
    components: Option<Vec<String>>,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::coordinates(&mesh.inner, components)?,
    })
}

/// Set node coordinates from `field` (absolute): for every node, the
/// active-set coordinate on axis `a` becomes `field.value(node,
/// components[a])`. `components` lists one component name per spatial axis,
/// in axis order; `None` → `["X", "Y", "Z"][:dim]`. Mutates the field's
/// `Coords` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn set_coordinates(
    field: PyRef<PyNodeField>,
    components: Option<Vec<String>>,
) -> PyResult<()> {
    crate::ops::field::set_coordinates(&field.inner, components)?;
    Ok(())
}

/// Displace nodes by `field` (incremental): `coord[a] += field.value(node,
/// components[a])` on the active coordinate set. `components` lists one
/// displacement-component name per spatial axis, in axis order; `None` →
/// `["ux", "uy", "uz"][:dim]`. Mutates the field's `Coords` in place.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, components=None))]
pub fn displace(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
    crate::ops::field::displace(&field.inner, components)?;
    Ok(())
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `NodeField` with one zone per submesh of `mesh`,
/// carrying the union of `field`'s components. Nodes of `mesh` absent
/// from `field` are assigned `0.0`. Errors if `mesh` and `field` are
/// attached to different `Coords`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::restrict(&field.inner, &mesh.inner)?,
    })
}

/// Merge two node fields « au plus juste »: structural union of their
/// zones, consolidated — zones sharing a component set are fused, the
/// others stay separate (nothing is densified).
///
/// Errors if the two fields hold different values at the same
/// `(node, component)` pair, or are attached to different `Coords`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn merge(a: PyRef<PyNodeField>, b: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::merge(&a.inner, &b.inner)?,
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
    let ef = crate::ops::field::gradient(&field.inner, &fespace.inner)?;
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
    let ef = crate::ops::field::deformation(&u.inner, &fespace.inner)?;
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
    let ef = crate::ops::field::beam_deformation(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Weak divergence `div F` of a per-element **vector** field — the adjoint of
/// `gradient`: `d_i = ∫ ∇N_i · F dΩ`, accumulated per node. The field must
/// carry exactly `space_dim` components (the vector components in order).
/// Returns a `NodeField` with a single `"div"` component (one zone per subspace).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn divergence(field: PyRef<PyElementField>) -> PyResult<PyNodeField> {
    let nf = crate::ops::field::divergence(&field.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Select the part of a field's support whose values fall in `[min, max]`,
/// zone by zone — a value-range filter returning a `Mesh`.
///
/// `field` may be a `NodeField` / `SubNodeField` (→ POI1 submeshes of the
/// passing **nodes**) or an `ElementField` / `SubElementField` (→ submeshes
/// of the passing **cells**, each of its zone's element type; a cell passes
/// only when *all* its Gauss points do). The result has one submesh per
/// processed zone.
///
/// At least one of `min` / `max` must be given (inclusive bounds). With
/// several components in play the bounds are combined with **AND**: a
/// point/cell is kept only when *every* tested component is in band.
///
/// `components=None` tests every component of each zone. A `components`
/// list tests **only** those components, and only on the zones carrying
/// **all** of them — a zone missing any listed component is skipped (no
/// submesh). Errors if both bounds are `None`, or `min > max`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, min=None, max=None, components=None))]
pub fn select(
    field: &Bound<'_, PyAny>,
    min: Option<f64>,
    max: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    use crate::ops::field as ops;
    let inner = if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        ops::select_nodes(&f.inner, min, max, components)?
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        ops::select_cells(&f.inner, min, max, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        ops::select_sub_nodes(&*read(&f.handle)?, min, max, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        ops::select_sub_cells(&*read(&f.handle)?, min, max, components)?
    } else {
        return Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ));
    };
    Ok(PyMesh { inner })
}

// ── Element-wise unary maths (numpy-style) ──────────────────────────────────
//
// `pyrucast.cos(field)`, `pyrucast.exp(field)`, … apply a scalar function to
// every value of a field, returning a **new** field of the same type. They
// accept any of the four field flavours (`NodeField` / `SubNodeField` /
// `ElementField` / `SubElementField`) and mirror `crate::ops::field::*`.
// Results are unguarded (numpy-like): `log` of ≤ 0 → `-inf`/`nan`, etc.

/// Generate a `#[pyfunction] $name(field)` that dispatches over the four field
/// wrapper types and applies the matching `ops::field::$name`.
macro_rules! py_field_unary {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
        #[pyfunction]
        pub fn $name(py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            use crate::ops::field::$name as op;
            if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
                return Ok(Py::new(py, PyNodeField { inner: op(&f.inner)? })?.into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
                return Ok(Py::new(py, PyElementField { inner: op(&f.inner)? })?.into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(py, PySubNodeField { handle: insert(out) })?.into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(py, PySubElementField { handle: insert(out) })?.into_any());
            }
            Err(PyTypeError::new_err(
                "expected a NodeField, SubNodeField, ElementField or SubElementField",
            ))
        }
    };
}

py_field_unary!(abs, "Element-wise absolute value of a field.");
py_field_unary!(sqrt, "Element-wise square root of a field (`nan` for negatives).");
py_field_unary!(exp, "Element-wise exponential `eˣ` of a field.");
py_field_unary!(log, "Element-wise natural logarithm of a field (`-inf`/`nan` for ≤ 0).");
py_field_unary!(log10, "Element-wise base-10 logarithm of a field.");
py_field_unary!(cos, "Element-wise cosine of a field (radians).");
py_field_unary!(sin, "Element-wise sine of a field (radians).");
py_field_unary!(tan, "Element-wise tangent of a field (radians).");
py_field_unary!(sinh, "Element-wise hyperbolic sine of a field.");
py_field_unary!(cosh, "Element-wise hyperbolic cosine of a field.");
py_field_unary!(tanh, "Element-wise hyperbolic tangent of a field.");
