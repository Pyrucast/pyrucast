//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::containers::model::{Model, SubModel};
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::store::{read, Handle};
use pyo3::prelude::*;

/// A **view** into one sub-model of a `Model`, obtained by indexing
/// (`model[i]`) — never constructed directly. Build physics at the parent
/// level instead: `Model.heat_conduction(fes)` or `Model.dirichlet(...)`,
/// composed with `+`.
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
/// and compose them with `+`; assemble with `stiffness` / `mass`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Model,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// `Model()` — an empty model; add physics with `+`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            inner: Model::empty(),
        })
    }

    /// `Model.heat_conduction(fespace)` — heat-conduction model spanning
    /// **every** subspace of `fespace` (one zone per subspace). A
    /// single-subspace space gives the unit case; several give one zone
    /// each. Compose heterogeneous physics with `+`:
    /// `Model.heat_conduction(fes) + Model.dirichlet(...)`.
    #[classmethod]
    fn heat_conduction(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PyFiniteElementSpace>,
    ) -> PyResult<Self> {
        let inner = Model::heat_conduction(&fespace.inner)?;
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

    /// `Model.dirichlet(imposed_variable, target_dual, imposed_mesh,
    /// multiplier_mesh, multiplier=None, imposed_value=None)` — Dirichlet
    /// constraint model (a single sub-model) imposed via Lagrange multipliers.
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
    /// component. See the model chapter of the book for the full semantics.
    #[classmethod]
    #[pyo3(signature = (imposed_variable, target_dual, imposed_mesh, multiplier_mesh, multiplier=None, imposed_value=None))]
    fn dirichlet(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: PyRef<'_, PyMesh>,
        multiplier_mesh: PyRef<'_, PyMesh>,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> PyResult<Self> {
        let inner = Model::dirichlet(
            imposed_variable,
            target_dual,
            &imposed_mesh.inner,
            &multiplier_mesh.inner,
            multiplier,
            imposed_value,
        )?;
        Ok(Self { inner })
    }

    /// Names of the primal (primary) variables across the whole model.
    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.primal_vars()?)
    }

    /// Names of the dual variables across the whole model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.dual_vars()?)
    }

}

crate::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model);
crate::impl_aggregate_sub_add!(PySubModel, PyModel);
crate::impl_dump_pymethod!(handle PySubModel, handle);

