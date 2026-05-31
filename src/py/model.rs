//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::containers::mesh::Node;
use crate::containers::model::{Model, SubModel};
use crate::py::element_field::{PyElementField, PySubElementField};
use crate::py::node::PyNode;
use crate::py::finite_element_space::PySubFiniteElementSpace;
use crate::py::matrix::PyMatrix;
use crate::py::mesh::PyMesh;
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

    /// `SubModel.dirichlet(primal_var, primal_dual, constrained_nodes)`
    /// — Dirichlet constraint via Lagrange multipliers. `constrained_nodes`
    /// is a list of `Node` objects (each carries its own `Configuration`);
    /// the multiplier nodes are created on the fly at the same coordinates
    /// as the constrained nodes.
    #[classmethod]
    fn dirichlet(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        primal_var: String,
        primal_dual: String,
        constrained_nodes: Vec<PyRef<'_, PyNode>>,
    ) -> PyResult<Self> {
        let nodes: Vec<Node> = constrained_nodes.iter().map(|n| n.as_node().clone()).collect();
        let sub = SubModel::dirichlet(primal_var, primal_dual, &nodes)?;
        Ok(Self { handle: insert(sub) })
    }

    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(with(&self.handle, |s| s.primal_vars())?)
    }

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

}

crate::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model);

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
    let sub = with(&sub_model.handle, |s| {
        crate::ops::build::sub_material_field(s, &pairs)
    })??;
    Ok(PySubElementField { handle: insert(sub) })
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
    let ef = crate::ops::build::material_field(&model.inner, &pairs)?;
    Ok(PyElementField { inner: ef })
}

/// Build a material `ElementField` where each sub-model gets its own
/// `(component, value)` list. The outer list length must equal
/// `model.sub_model_count()`. An empty inner list **skips** the matching
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
    let ef = crate::ops::build::material_field_per_sub_model(&model.inner, &slices)?;
    Ok(PyElementField { inner: ef })
}

/// Assemble the stiffness matrix `K` of `model`.
///
/// `materials` carries the per-zone material data: every sub-model that
/// needs it picks the `SubElementField` whose FE subspace matches its own.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
pub fn stiffness(model: PyRef<PyModel>, materials: PyRef<PyElementField>) -> PyResult<PyMatrix> {
    let k = crate::ops::assemble::stiffness(&model.inner, &materials.inner)?;
    Ok(PyMatrix { inner: k })
}

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
