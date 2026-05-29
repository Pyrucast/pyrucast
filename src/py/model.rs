//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::configuration::NodeId;
use crate::containers::model::{Model, SubModel};
use crate::py::configuration::PyConfiguration;
use crate::py::element_field::PySubElementField;
use crate::py::finite_element_space::PySubFiniteElementSpace;
use crate::py::matrix::PyMatrix;
use crate::store::{insert, with, Handle};
use pyo3::prelude::*;

/// Python wrapper for [`SubModel`].
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubModel")]
pub struct PySubModel {
    pub(crate) handle: Handle<SubModel>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubModel {
    /// `SubModel.heat_conduction(fespace)` — heat-conduction sub-model on
    /// a finite-element subspace. Material data is supplied at assembly
    /// time via `assemble.stiffness(model, material)`.
    #[classmethod]
    fn heat_conduction(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        fespace: PyRef<PySubFiniteElementSpace>,
    ) -> PyResult<Self> {
        let sub = SubModel::heat_conduction(fespace.handle.clone())?;
        Ok(Self { handle: insert(sub) })
    }

    /// `SubModel.dirichlet(config, primal_var, primal_dual, constrained_node_ids)`
    /// — Dirichlet constraint via Lagrange multipliers. The multiplier
    /// nodes are created on the fly in `config` at the same coordinates
    /// as the constrained nodes.
    #[classmethod]
    fn dirichlet(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        config: PyRef<PyConfiguration>,
        primal_var: String,
        primal_dual: String,
        constrained_node_ids: Vec<u32>,
    ) -> PyResult<Self> {
        let nodes: Vec<NodeId> = constrained_node_ids.into_iter().map(NodeId).collect();
        let sub = SubModel::dirichlet(config.handle.clone(), primal_var, primal_dual, nodes)?;
        Ok(Self { handle: insert(sub) })
    }

    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |s| s.primal_vars())?)
    }

    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |s| s.dual_vars())?)
    }

    /// Multiplier node ids (Lagrange physics only — empty otherwise).
    fn multiplier_nodes(&self) -> PyResult<Vec<u32>> {
        let ids = with(&self.handle, |s| s.multiplier_nodes())??;
        Ok(ids.into_iter().map(|n| n.0).collect())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{:?}", s))?)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(with(&self.handle, |s| format!("{}", s))?)
    }
}

/// Python wrapper for [`Model`].
///
/// Owns the `Model` struct directly — no longer stored in the global
/// store. Identity is the Python object identity itself.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Model,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            inner: Model::empty(),
        })
    }

    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.primal_vars()?)
    }

    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.dual_vars()?)
    }

    fn stiffness(&self, material: PyRef<PySubElementField>) -> PyResult<PyMatrix> {
        let k = crate::ops::assemble::stiffness(&self.inner, &material.handle)?;
        Ok(PyMatrix { handle: insert(k) })
    }

    fn mass(&self) -> PyResult<PyMatrix> {
        let m_mat = self.inner.mass()?;
        Ok(PyMatrix { handle: insert(m_mat) })
    }

}

crate::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model);
