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
pub fn coordinates(mesh: PyRef<PyMesh>, components: Option<Vec<String>>) -> PyResult<PyNodeField> {
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
pub fn set_coordinates(field: PyRef<PyNodeField>, components: Option<Vec<String>>) -> PyResult<()> {
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

/// Reproject `field` onto the exact support and components of `target`,
/// zone by zone.
///
/// Unlike `restrict` (which lands on a fresh support materialised from a
/// mesh, carrying the union of `field`'s components), this reuses each zone
/// of `target` as-is — same support, same component list — so the result is
/// on the very same support as `target` and combines with it directly by the
/// arithmetic operators (`target + restrict_like(field, target)`). A
/// `(node, component)` pair is filled from `field` when it covers it, `0.0`
/// otherwise; nodes and components of `field` absent from `target` are dropped.
/// Errors if `target` and `field` are attached to different `Coords`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict_like(
    field: PyRef<PyNodeField>,
    target: PyRef<PyNodeField>,
) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::field::restrict_like(&field.inner, &target.inner)?,
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
    let ef = crate::ops::field::interp_to_gauss(&field.inner, &fespace.inner)?;
    Ok(PyElementField { inner: ef })
}

/// Thermal (free-dilation) strain `ε_th = α·(T − t_ref)` at the Gauss points of
/// `fespace` — Cast3M `EPTH`. `temperature` is a per-element field carrying
/// `"T"` (e.g. from `interp_to_gauss`); `materials` carries `"alpha"` (supplied
/// via `material_field`). Returns the strain tensor in the same layout as
/// `deformation`, so `deformation(u, fespace) - thermal_strain(...)` is the
/// mechanical strain. Backbone of uncoupled thermomechanics: assemble the
/// thermal load with `internal_forces(model, integrate_behavior(model, ε_th,
/// materials))` and recover `σ = D:(ε − ε_th)`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn thermal_strain(
    temperature: PyRef<PyElementField>,
    materials: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
    t_ref: f64,
) -> PyResult<PyElementField> {
    let ef = crate::ops::field::thermal_strain(
        &temperature.inner,
        &materials.inner,
        &fespace.inner,
        t_ref,
    )?;
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

/// Integral `∫_Ω f dΩ` of a field over its support, using the finite-element
/// quadrature — the total of one `component` (e.g. the resultant of a
/// distributed force **density**).
///
/// - `NodeField`: interpolates with the shape functions, `∫ Σ_i f_i N_i dΩ` —
///   `fespace` is **required**.
/// - `ElementField`: integrates the Gauss-point values directly,
///   `Σ_cell Σ_g f·|J|·w` — `fespace` is ignored.
///
/// For a field of already-integrated **nodal** forces, the resultant is a plain
/// sum instead: `field.sum(component)`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, component, fespace=None))]
pub fn integral(
    field: &Bound<'_, PyAny>,
    component: &str,
    fespace: Option<PyRef<PyFiniteElementSpace>>,
) -> PyResult<f64> {
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let fes = fespace.ok_or_else(|| {
            PyTypeError::new_err("integral: a NodeField needs a FiniteElementSpace (fespace=...)")
        })?;
        return Ok(crate::ops::field::integral(
            &f.inner, &fes.inner, component,
        )?);
    }
    if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        return Ok(crate::ops::field::integral_element(&f.inner, component)?);
    }
    Err(PyTypeError::new_err(
        "integral: expected a NodeField (with fespace=...) or an ElementField",
    ))
}

/// Squared Euclidean norm `xᵀx = Σ v²` of a field (Cast3M `XTX`) — the sum of
/// squares over every value of every zone. Accepts a `NodeField`,
/// `SubNodeField`, `ElementField` or `SubElementField`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn xtx(x: &Bound<'_, PyAny>) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        return Ok(a.inner.xtx()?);
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        return Ok(a.inner.xtx()?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        return Ok(read(&a.handle)?.xtx());
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        return Ok(read(&a.handle)?.xtx());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

/// Select the part of a field's support passing a value band, zone by zone —
/// a value-range filter returning a `Mesh`.
///
/// `field` may be a `NodeField` / `SubNodeField` (→ POI1 submeshes of the
/// passing **nodes**) or an `ElementField` / `SubElementField` (→ submeshes
/// of the passing **cells**, each of its zone's element type; a cell passes
/// only when *all* its Gauss points do). The result has one submesh per
/// processed zone.
///
/// The band is set by the four comparison bounds — `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`); give at most one lower (`ge`/`gt`) and one upper
/// (`le`/`lt`), at least one overall. With several components in play they
/// are combined with **AND**: a point/cell is kept only when *every* tested
/// component is in band.
///
/// `components=None` tests every component of each zone. A `components`
/// list tests **only** those components, and only on the zones carrying
/// **all** of them — a zone missing any listed component is skipped (no
/// submesh). Errors if no bound is given, or the lower one exceeds the upper.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (field, ge=None, gt=None, le=None, lt=None, components=None))]
#[allow(clippy::too_many_arguments)]
pub fn select(
    field: &Bound<'_, PyAny>,
    ge: Option<f64>,
    gt: Option<f64>,
    le: Option<f64>,
    lt: Option<f64>,
    components: Option<Vec<String>>,
) -> PyResult<PyMesh> {
    use crate::ops::field as ops;
    let band = ops::Band::new(ge, gt, le, lt)?;
    let inner = if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        ops::select_nodes(&f.inner, &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        ops::select_cells(&f.inner, &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        ops::select_sub_nodes(&*read(&f.handle)?, &band, components)?
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        ops::select_sub_cells(&*read(&f.handle)?, &band, components)?
    } else {
        return Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ));
    };
    Ok(PyMesh { inner })
}

/// Per-component 0/1 **mask** of a field against a value band — same flavour
/// and same structure as the input (Cast3M's `MASQUE`).
///
/// Unlike [`select`](fn@select), which extracts the passing support into a
/// `Mesh`, `mask` keeps the field's exact shape (zones, support, components)
/// and only rewrites the values: `1.0` where the band holds, `0.0` where it
/// does not — so the result is multipliable term by term with the input
/// (`field * mask(field, ge=0)` zeroes the negatives, component by component).
/// A `NodeField` masks per node, an `ElementField` per Gauss point.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`) — same rules as [`select`](fn@select). There is
/// **no** AND across components here: each value stands on its own.
///
/// `components=None` tests every component. A `components` list tests only
/// those; the others stay at `1.0` (identity for the product), and a zone
/// missing a listed component is left all-`1.0`. Errors if no bound is given,
/// or the lower one exceeds the upper.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
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
    use crate::ops::field as ops;
    let band = ops::Band::new(ge, gt, le, lt)?;
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let out = ops::mask_nodes(&f.inner, &band, components)?;
        Ok(Py::new(py, PyNodeField { inner: out })?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
        let out = ops::mask_cells(&f.inner, &band, components)?;
        Ok(Py::new(py, PyElementField { inner: out })?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        let out = ops::mask_sub_nodes(&*read(&f.handle)?, &band, components);
        Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
        let out = ops::mask_sub_cells(&*read(&f.handle)?, &band, components);
        Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
            },
        )?
        .into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a NodeField, SubNodeField, ElementField or SubElementField",
        ))
    }
}

/// Global scalar product `∑ xᵢ · yᵢ` of two **whole** fields — Cast3M's `XTY`.
///
/// `x` and `y` must be the same flavour (`NodeField` / `SubNodeField` /
/// `ElementField` / `SubElementField`), sit on the same support/decomposition,
/// and carry the same components (aligned by name). Returns a single float —
/// the field inner product used for energies (`F·u`), residual norms, etc.
///
/// For the **node-by-node** scalar product (a field, one value per node),
/// see [`psca`](fn@psca).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn xty(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<f64> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be NodeFields"))?;
        return Ok(a.inner.dot_field(&b.inner)?);
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be ElementFields"))?;
        return Ok(a.inner.dot_field(&b.inner)?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubNodeFields"))?;
        return Ok(read(&a.handle)?.dot(&*read(&b.handle)?)?);
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("xty: both operands must be SubElementFields"))?;
        return Ok(read(&a.handle)?.dot(&*read(&b.handle)?)?);
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
}

/// Node-by-node (or point-by-point) scalar product of two fields — Cast3M's
/// `PSCA`. Returns a **new field** of the same flavour as the inputs, carrying
/// a single `"psca"` component whose value at each node/point is `∑_c xᵣ,c·yᵣ,c`
/// (reduction over components only, the support is kept).
///
/// `x` and `y` must be the same flavour (`NodeField` / `SubNodeField` /
/// `ElementField` / `SubElementField`), sit on the same support/decomposition,
/// and carry the same components (aligned by name).
///
/// For the **global** scalar product (a single float over the whole field),
/// see [`xty`](fn@xty).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn psca(py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use crate::containers::field::{Field, SubField};
    if let Ok(a) = x.extract::<PyRef<PyNodeField>>() {
        let b = y
            .extract::<PyRef<PyNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be NodeFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyNodeField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PyElementField>>() {
        let b = y
            .extract::<PyRef<PyElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be ElementFields"))?;
        let inner = a.inner.pscal_field(&b.inner)?;
        return Ok(Py::new(py, PyElementField { inner })?.into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubNodeField>>() {
        let b = y
            .extract::<PyRef<PySubNodeField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubNodeFields"))?;
        let out = read(&a.handle)?.pscal(&*read(&b.handle)?)?;
        return Ok(Py::new(
            py,
            PySubNodeField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    if let Ok(a) = x.extract::<PyRef<PySubElementField>>() {
        let b = y
            .extract::<PyRef<PySubElementField>>()
            .map_err(|_| PyTypeError::new_err("psca: both operands must be SubElementFields"))?;
        let out = read(&a.handle)?.pscal(&*read(&b.handle)?)?;
        return Ok(Py::new(
            py,
            PySubElementField {
                handle: insert(out),
            },
        )?
        .into_any());
    }
    Err(PyTypeError::new_err(
        "expected a NodeField, SubNodeField, ElementField or SubElementField",
    ))
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
                return Ok(Py::new(
                    py,
                    PyNodeField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PyElementField>>() {
                return Ok(Py::new(
                    py,
                    PyElementField {
                        inner: op(&f.inner)?,
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(
                    py,
                    PySubNodeField {
                        handle: insert(out),
                    },
                )?
                .into_any());
            }
            if let Ok(f) = field.extract::<PyRef<PySubElementField>>() {
                let out = op(&*read(&f.handle)?)?;
                return Ok(Py::new(
                    py,
                    PySubElementField {
                        handle: insert(out),
                    },
                )?
                .into_any());
            }
            Err(PyTypeError::new_err(
                "expected a NodeField, SubNodeField, ElementField or SubElementField",
            ))
        }
    };
}

py_field_unary!(abs, "Element-wise absolute value of a field.");
py_field_unary!(
    sqrt,
    "Element-wise square root of a field (`nan` for negatives)."
);
py_field_unary!(exp, "Element-wise exponential `eˣ` of a field.");
py_field_unary!(
    log,
    "Element-wise natural logarithm of a field (`-inf`/`nan` for ≤ 0)."
);
py_field_unary!(log10, "Element-wise base-10 logarithm of a field.");
py_field_unary!(cos, "Element-wise cosine of a field (radians).");
py_field_unary!(sin, "Element-wise sine of a field (radians).");
py_field_unary!(tan, "Element-wise tangent of a field (radians).");
py_field_unary!(sinh, "Element-wise hyperbolic sine of a field.");
py_field_unary!(cosh, "Element-wise hyperbolic cosine of a field.");
py_field_unary!(tanh, "Element-wise hyperbolic tangent of a field.");
