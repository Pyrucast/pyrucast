//! Python wrappers for [`crate::ops::node_field`] — the operators that
//! produce a `NodeField`.

use crate::handle::Handle;
use crate::py::element_field::PyElementField;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::py::model::PyModel;
use crate::py::node_field::PyNodeField;
use crate::py::node_field::PySubNodeField;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build a `NodeField` carrying the position of every node of `mesh`
/// — one `SubNodeField` per submesh, on the distinct nodes of its zone.
///
/// One component per requested axis (`"X"`, `"Y"`, `"Z"`). `components=None`
/// requests all the axes the mesh's `Coords` has (`["X"]` in 1-D,
/// `["X", "Y"]` in 2-D, `["X", "Y", "Z"]` in 3-D).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, components=None))]
pub fn positions(mesh: PyRef<PyMesh>, components: Option<Vec<String>>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::positions(&mesh.inner, components)?,
    })
}

/// Restrict `field` to the nodes used by `mesh`.
///
/// Returns a new `NodeField` with one zone per submesh of `mesh`, each
/// supported on the submesh's canonical POI1 node cloud (its distinct nodes,
/// materialised once and cached). Two restrictions onto the **same** `mesh`
/// share that support, so they combine directly: `restrict(a, mesh) -
/// restrict(b, mesh)` is the node-by-node difference. That support is also the
/// one a stiffness block over `mesh` uses, so `K * restrict(f, mesh)` and
/// `solve(K, f) - restrict(g, mesh)` line up too.
///
/// Each zone carries the union of `field`'s components; nodes of `mesh` absent
/// from `field` are assigned `0.0`. Element operations on the region
/// (`gradient`, `integral`, `deformation`, `interp_to_gauss`) take `mesh` as a
/// separate argument and read the field by node id. Use `restrict_like` to
/// land on the exact support of an existing field instead of a mesh.
///
/// Errors if `mesh` and `field` are attached to different `Coords`s.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn restrict(field: PyRef<PyNodeField>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::restrict(&field.inner, &mesh.inner)?,
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
        inner: crate::ops::node_field::restrict_like(&field.inner, &target.inner)?,
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
        inner: crate::ops::node_field::merge(&a.inner, &b.inner)?,
    })
}

/// Fuse the zones of a node `field` sharing the same component set into one,
/// deduping the nodes on their interface after a coherence check.
///
/// Errors if two zones disagree on a value at a shared `(node, component)`
/// pair. `field` itself is left untouched.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "consolidate_node")]
pub fn consolidate(field: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
    Ok(PyNodeField {
        inner: crate::ops::node_field::consolidate(&field.inner)?,
    })
}

/// Weak divergence `div F` of a per-element **vector** field — the adjoint of
/// `gradient`: `d_i = ∫ ∇N_i · F dΩ`, accumulated per node. The field must
/// carry exactly `space_dim` components (the vector components in order).
/// Returns a `NodeField` with a single `"div"` component (one zone per subspace).
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn divergence(field: PyRef<PyElementField>) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::divergence(&field.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Internal nodal forces of `model` — the left side of `Σ f_int = Σ f_ext`
/// (Cast3m `BSIG` for a continuum).
///
/// The nodal mirror of `matrix.stiffness(model, materials)`: every sub-model is
/// asked for its term of the residual on the internal side. `state` is the
/// material-state field produced by `integrate_behavior` (`COMP`); a
/// behaviour-bearing sub-model applies its own `Bᵀ` (continuum solid, bar or
/// beam), a sub-model with no term here declares none. Returns a `NodeField`
/// whose components are each sub-model's dual variables (`f_x`, … for a
/// solid/bar; `f_w`, `m_theta` for a beam).
///
/// For a linear law the result equals the assembled stiffness applied to the
/// solution (`K·u`); a non-linear law gives the exact internal forces. Its
/// counterpart is `external_forces(model, materials)`, and the gap between the
/// two sums is the residual.
///
/// `solution` is the current one, multipliers included: a term that is linear in
/// `u` reads it directly — a boundary transfer's `∫ h·a·N` has no law to go
/// through — and a constraint draws its reaction `Cᵀ λ` from it, spread over the
/// constrained nodes by the relation's coefficients.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn internal_forces(
    model: PyRef<PyModel>,
    state: PyRef<PyElementField>,
    solution: PyRef<PyNodeField>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::internal_forces(
        &model.inner,
        &state.inner,
        &solution.inner,
        &materials.inner,
    )?;
    Ok(PyNodeField { inner: nf })
}

/// External nodal forces of `model` — the right side of `Σ f_int = Σ f_ext`.
///
/// The counterpart of `internal_forces(model, state)`. Every sub-model is asked
/// for its terms of the residual on the external side: the given data of its
/// weak form, on the right of the equals sign. A physics whose term is entirely
/// a response to `u` (elasticity, conduction, a bar) has none, so a model made
/// only of those yields an empty field — which is the honest answer, not a
/// failure.
///
/// Splitting the two sides is what keeps signs out of the physics: an author
/// writes both halves positively, as the weak form reads, and the single
/// subtraction lives in the caller.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn external_forces(
    model: PyRef<PyModel>,
    materials: PyRef<PyElementField>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::external_forces(&model.inner, &materials.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Internal nodal forces of a **continuum-mechanics** stress field, without a
/// model (Cast3m `BSIG` for a plain solid).
///
/// Convenience for the volumetric case (elasticity, Mazars, plasticity), where
/// `B` is the universal symmetric gradient: it needs only the geometry
/// (`fespace`) and the Voigt stress (`sigma_xx`, `sigma_xy`, …). Returns a
/// `NodeField` with `space_dim` components `f_x, f_y, f_z` per node. **Bars and
/// beams are not covered** — use `internal_forces(model, state)` for those.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn internal_forces_continuum(
    stresses: PyRef<PyElementField>,
    fespace: PyRef<PyFiniteElementSpace>,
) -> PyResult<PyNodeField> {
    let nf = crate::ops::node_field::internal_forces_continuum(&stresses.inner, &fespace.inner)?;
    Ok(PyNodeField { inner: nf })
}

/// Per-component 0/1 **mask** of a field against a value band — same structure
/// as the input (Cast3M `MASQUE`): same zones, same support, same components,
/// only the values are rewritten (`1.0` inside the band, `0.0` outside). The
/// result is therefore multipliable term by term with the input.
///
/// Its sibling `pyrucast.mesh.select` extracts the passing *support* instead,
/// and produces a `Mesh`.
///
/// The band is set by the four comparison bounds `ge` (`≥`), `gt` (`>`),
/// `le` (`≤`), `lt` (`<`). There is **no** AND across components: each value
/// stands on its own. `components=None` tests every component; a `components`
/// list tests only those, leaving the others at `1.0` (identity for the
/// product), and a zone missing a listed component is left all-`1.0`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "mask_node")]
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
    let band = crate::atoms::Band::new(ge, gt, le, lt)?;
    if let Ok(f) = field.extract::<PyRef<PyNodeField>>() {
        let out = crate::ops::node_field::mask(&f.inner, &band, components)?;
        Ok(Py::new(py, PyNodeField { inner: out })?.into_any())
    } else if let Ok(f) = field.extract::<PyRef<PySubNodeField>>() {
        let out = crate::ops::node_field::mask_sub(&f.handle.read(), &band, components);
        Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    } else {
        Err(PyTypeError::new_err(
            "expected a PyNodeField or PySubNodeField",
        ))
    }
}

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// La face « sujet » des opérateurs ci-dessus (`CONVENTIONS.md` § « Le verbe
// exposé aussi en méthode »). Aucune logique : chaque méthode rappelle la
// fonction libre, receveur compris.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// Voir `pyrucast.node_field.consolidate`.
    fn consolidate(slf: PyRef<'_, Self>) -> PyResult<PyNodeField> {
        super::node_field::consolidate(slf)
    }

    /// Voir `pyrucast.node_field.restrict`.
    fn restrict(slf: PyRef<'_, Self>, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
        super::node_field::restrict(slf, mesh)
    }

    /// Voir `pyrucast.node_field.restrict_like`.
    fn restrict_like(slf: PyRef<'_, Self>, target: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
        super::node_field::restrict_like(slf, target)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyMesh {
    /// Voir `pyrucast.node_field.positions`.
    #[pyo3(signature = (components=None))]
    fn positions(slf: PyRef<'_, Self>, components: Option<Vec<String>>) -> PyResult<PyNodeField> {
        super::node_field::positions(slf, components)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// Voir `pyrucast.node_field.internal_forces`.
    fn internal_forces(
        slf: PyRef<'_, Self>,
        state: PyRef<PyElementField>,
        solution: PyRef<PyNodeField>,
        materials: PyRef<PyElementField>,
    ) -> PyResult<PyNodeField> {
        super::node_field::internal_forces(slf, state, solution, materials)
    }

    /// Voir `pyrucast.node_field.external_forces`.
    fn external_forces(
        slf: PyRef<'_, Self>,
        materials: PyRef<PyElementField>,
    ) -> PyResult<PyNodeField> {
        super::node_field::external_forces(slf, materials)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyElementField {
    /// Voir `pyrucast.node_field.divergence`.
    fn divergence(slf: PyRef<'_, Self>) -> PyResult<PyNodeField> {
        super::node_field::divergence(slf)
    }
}
