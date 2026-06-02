//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::Node;
use crate::containers::model::{Model, SubModel};
use crate::py::node::PyNode;
use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use crate::store::{with, Handle};
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
        Ok(with(&self.handle, |s| s.primal_vars())?)
    }

    /// Names of the dual variables of this sub-model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |s| s.dual_vars())?)
    }

    /// POI1 `Mesh` of the multiplier nodes (Lagrange physics only — empty
    /// otherwise). Build a load `NodeField` on `mesh[0]` to impose the
    /// constrained values, or read the nodes via `mesh.node(0, i, 0)`.
    fn multiplier_mesh(&self) -> PyResult<PyMesh> {
        let mesh = with(&self.handle, |s| s.multiplier_mesh())??;
        Ok(PyMesh { inner: mesh })
    }

    /// Names of the material components this sub-model expects, or
    /// `None` for physics that don't need material data (Dirichlet, …).
    fn material_components(&self) -> PyResult<Option<Vec<String>>> {
        Ok(with(&self.handle, |s| {
            s.material_components()
                .map(|c| c.iter().map(|s| s.to_string()).collect())
        })?)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{:?}", s))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{}", s))?)
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

    /// `Model.dirichlet(primal_var, primal_dual, constrained_nodes)` —
    /// Dirichlet-constraint model (a single sub-model) imposed via
    /// Lagrange multipliers. `constrained_nodes` is a list of `Node`
    /// objects. `primal_var` is the constrained primary variable (e.g.
    /// `"T"`); `primal_dual` is the dual variable of the primary physics
    /// it targets (e.g. `"q"` for heat conduction) — see the model
    /// chapter of the book for the full semantics.
    #[classmethod]
    fn dirichlet(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        primal_var: String,
        primal_dual: String,
        constrained_nodes: Vec<PyRef<'_, PyNode>>,
    ) -> PyResult<Self> {
        let nodes: Vec<Node> = constrained_nodes.iter().map(|n| n.as_node().clone()).collect();
        let inner = Model::dirichlet(primal_var, primal_dual, &nodes)?;
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
crate::impl_dump_pymethod!(handle PySubModel, handle);

