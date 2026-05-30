//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::configuration::NodeId;
use crate::containers::model::{Model, SubModel};
use crate::py::configuration::PyConfiguration;
use crate::py::element_field::{PyElementField, PySubElementField};
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

    /// Names of the material components this sub-model expects, or
    /// `None` for physics that don't need material data (Dirichlet, …).
    fn material_components(&self) -> PyResult<Option<Vec<String>>> {
        Ok(with(&self.handle, |s| {
            s.material_components()
                .map(|c| c.iter().map(|s| s.to_string()).collect())
        })?)
    }

    /// `sub_model.build_material_field([("k", 1.0), ...])` — fresh
    /// SubElementField on this sub-model's FE subspace, pre-filled with
    /// the given uniform value per component. Errors for physics that
    /// don't need a material (e.g. Dirichlet).
    fn build_material_field(
        &self,
        components_and_values: Vec<(String, f64)>,
    ) -> PyResult<PySubElementField> {
        let pairs: Vec<(&str, f64)> = components_and_values
            .iter()
            .map(|(c, v)| (c.as_str(), *v))
            .collect();
        let sub = with(&self.handle, |s| s.build_material_field(&pairs))??;
        Ok(PySubElementField { handle: insert(sub) })
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

    /// Assemble the stiffness matrix.
    ///
    /// `materials` carries the per-zone material data: every sub-model
    /// that needs it picks the [`crate::containers::element_field::SubElementField`]
    /// whose FE subspace matches its own.
    fn stiffness(&self, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
        let k = crate::ops::assemble::stiffness(&self.inner, &materials.inner)?;
        Ok(PyMatrix { inner: k })
    }

    /// `model.build_material_field([("k", 1.0), ...])` — material
    /// ElementField with the same uniform `(component, value)` pairs
    /// applied to every material-hungry sub-model. Sub-models that don't
    /// need a material (Dirichlet, …) are skipped.
    fn build_material_field(
        &self,
        components_and_values: Vec<(String, f64)>,
    ) -> PyResult<PyElementField> {
        let pairs: Vec<(&str, f64)> = components_and_values
            .iter()
            .map(|(c, v)| (c.as_str(), *v))
            .collect();
        let ef = self.inner.build_material_field(&pairs)?;
        Ok(PyElementField { inner: ef })
    }

    /// `model.build_material_field_per_sub_model([[("k", 1.0)], [], [("k", 4.0)]])`
    /// — material ElementField where each sub-model gets its own
    /// `(component, value)` list. The outer list length must equal
    /// `sub_model_count()`. An empty inner list **skips** the matching
    /// sub-model (typical for Dirichlet).
    fn build_material_field_per_sub_model(
        &self,
        components_and_values_per_sub_model: Vec<Vec<(String, f64)>>,
    ) -> PyResult<PyElementField> {
        // Materialise each inner Vec<(String, f64)> into a Vec<(&str, f64)>,
        // then collect slices into a Vec<&[(&str, f64)]>.
        let owned: Vec<Vec<(&str, f64)>> = components_and_values_per_sub_model
            .iter()
            .map(|v| v.iter().map(|(c, x)| (c.as_str(), *x)).collect())
            .collect();
        let slices: Vec<&[(&str, f64)]> = owned.iter().map(|v| v.as_slice()).collect();
        let ef = self.inner.build_material_field_per_sub_model(&slices)?;
        Ok(PyElementField { inner: ef })
    }
}

crate::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model);

/// Assemble the mass matrix `M` of `model`.
///
/// v0 stub: no physics has a mass term yet, so this returns an empty
/// finalized `Matrix` with the model's DOF layout.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn mass(model: PyRef<PyModel>) -> PyResult<PyMatrix> {
    let m = crate::ops::assemble::mass(&model.inner)?;
    Ok(PyMatrix { inner: m })
}
