//! Python wrappers for [`crate::containers::model::SubModel`] and [`crate::containers::model::Model`].

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::model::{Model, SubModel};
use crate::handle::Handle;
use crate::models::Physics;
use crate::py::finite_element_space::{PyFiniteElementSpace, PySubFiniteElementSpace};
use crate::py::mesh::PyMesh;
use crate::py::node::PyNode;
use crate::py::node_field::PyNodeField;
use pyo3::prelude::*;

/// A **view** into one sub-model of a `Model`, obtained by indexing
/// (`model[i]`) — never constructed directly. Build physics at the parent
/// level instead: `model.heat_conduction(fes)` or `model.dirichlet(...)`,
/// composed with `|`.
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
        Ok(self.handle.read().primal_vars())
    }

    /// Names of the dual variables of this sub-model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.handle.read().dual_vars())
    }

    /// `sub.fespace()` — the `SubFiniteElementSpace` this sub-model integrates
    /// its behaviour on, or `None` for a constraint sub-model (Dirichlet, MPC…,
    /// which integrate nothing). The per-sub-model counterpart of
    /// `Model.fespace()`.
    fn fespace(&self) -> PyResult<Option<PySubFiniteElementSpace>> {
        Ok(self
            .handle
            .read()
            .behavior_fespace()
            .map(|handle| PySubFiniteElementSpace { handle }))
    }

    /// The physics nature(s) of this sub-model as a list of tags (`"mechanical"`,
    /// `"thermal"`, `"constraint"`, `"other"`). Determined entirely by the physics
    /// kind — one tag for a plain physics, several for a coupled one.
    fn physics(&self) -> PyResult<Vec<String>> {
        Ok(self
            .handle
            .read()
            .physics()
            .iter()
            .map(|p| p.name().to_string())
            .collect())
    }

    /// POI1 `Mesh` of the multiplier nodes (Lagrange physics only — empty
    /// otherwise). Build a load `SubNodeField` on `mesh[0]` to impose the
    /// constrained values, or read the nodes via `mesh.node(0, i, 0)`.
    fn multiplier_mesh(&self) -> PyResult<PyMesh> {
        let mesh = self.handle.read().multiplier_mesh()?;
        Ok(PyMesh { inner: mesh })
    }

    /// Names of the material components this sub-model expects, or
    /// `None` for physics that don't need material data (Dirichlet, …).
    fn material_components(&self) -> PyResult<Option<Vec<String>>> {
        Ok(self
            .handle
            .read()
            .material_components()
            .map(|c| c.iter().map(|s| s.to_string()).collect()))
    }

    /// Whether this sub-model carries a constitutive behaviour that can be
    /// integrated with `deformation` / `integrate_behavior` (`True` for
    /// volumetric physics, `False` for constraints like Dirichlet).
    fn has_behavior(&self) -> PyResult<bool> {
        Ok(self.handle.read().has_behavior())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.handle.read()))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", *self.handle.read()))
    }
}

/// A physics problem: a collection of sub-models (heat conduction,
/// Dirichlet BCs, ...) over finite-element spaces.
///
/// Build sub-models with the operators of `pyrucast.model`
/// (`model.heat_conduction(fes)`, `model.dirichlet(...)`) and compose them
/// with `|`; assemble with `stiffness` / `mass`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "Model")]
pub struct PyModel {
    pub(crate) inner: Model,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyModel {
    /// `Model()` — an empty model; add physics with `|`.
    #[new]
    fn py_new() -> PyResult<Self> {
        Ok(Self {
            inner: Model::empty(),
        })
    }

    /// `Model.contact_gaps()` — the contact right-hand side `−g₀`: a
    /// `NodeField` carrying, at each contact multiplier node's `imposed_value`
    /// slot, minus the initial signed gap of its relation, so that
    /// non-penetration reads `g₀ + C·u ≥ 0`. Merge it into the global load
    /// with `|`. The model must hold exactly one contact sub-model; omitting
    /// this helper treats every pair as initially touching (`g₀ = 0`).
    fn contact_gaps(&self) -> PyResult<PyNodeField> {
        Ok(PyNodeField {
            inner: self.inner.contact_gaps()?,
        })
    }

    /// Names of the primal (primary) variables across the whole model.
    fn primal_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.primal_vars())
    }

    /// Names of the dual variables across the whole model.
    fn dual_vars(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.dual_vars())
    }

    /// `Model.dual_of(variable)` — the dual (residual) variable conjugate to a
    /// primal `variable` (e.g. `"u_x" -> "f_x"`, `"T" -> "q"`), searched across
    /// all sub-models, or `None`. A helper to fill an MPC term's `target_dual`.
    fn dual_of(&self, variable: &str) -> PyResult<Option<String>> {
        Ok(self.inner.dual_of(variable))
    }

    /// `Model.fespace()` — the `FiniteElementSpace` this model integrates on,
    /// rebuilt from the behaviour subspaces of its domain sub-models (constraints
    /// skipped), deduplicated in first-seen order (shared handles). Raises if the
    /// model has no domain sub-model. Combined with `FiniteElementSpace.mesh()`,
    /// lets a caller recover the FE space and mesh from the model alone.
    fn fespace(&self) -> PyResult<PyFiniteElementSpace> {
        Ok(PyFiniteElementSpace {
            inner: self.inner.fespace()?,
        })
    }

    /// `Model.filter(physics)` — a new `Model` holding only the sub-models **whose
    /// nature set contains** the given physics. `physics` is a tag: `"mechanical"`,
    /// `"thermal"`, `"constraint"`, `"other"`, `"diffusion"` or `"radiation"`.
    /// Sub-model order is preserved; the result may be empty.
    ///
    /// A coupled physics declares several natures and is therefore returned by
    /// each of its filters — a radiation boundary is both `"thermal"` and
    /// `"radiation"`.
    fn filter(&self, physics: Physics) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.filter(physics)?,
        })
    }

    /// `Model.multiplier_mesh()` — POI1 `Mesh` of the constraint multiplier
    /// nodes across the model (empty for a model with no constraint). The handle
    /// to the multiplier nodes of an `embedded` / `dirichlet` / `mpc` model whose
    /// multipliers it minted itself: read them via `mesh.node(0, i, 0)` or build
    /// a load `SubNodeField` on `mesh[0]`.
    fn multiplier_mesh(&self) -> PyResult<PyMesh> {
        Ok(PyMesh {
            inner: self.inner.multiplier_mesh()?,
        })
    }

    /// `Model.constraint_rhs(imposed)` — build the constraint load (right-hand
    /// side) of this model's constraint from `(node, g)` pairs.
    ///
    /// `imposed` is a list of `(Node, value)`: each node **keys the relation it
    /// belongs to** (for `dirichlet` the constrained node itself, for `mpc` any
    /// of the relation's term nodes), and `value` is the right-hand side `g`.
    /// Returns a fresh `NodeField` over the multiplier nodes, carrying the
    /// constraint's imposed-value component (its dual, e.g. `imposed_T` /
    /// `mpc_rhs`), with `g` at each cited relation and `0` elsewhere. Union it
    /// into the global load with `|`, e.g. `load | model.constraint_rhs([...])`.
    ///
    /// The model must hold exactly one constraint sub-model (the `dirichlet` /
    /// `mpc` object). Raises if a node constrains none of its relations, or keys
    /// several of them (ambiguous — drop it).
    fn constraint_rhs(
        &self,
        py: Python<'_>,
        imposed: Vec<(Py<PyNode>, f64)>,
    ) -> PyResult<PyNodeField> {
        let pairs: Vec<(NodeId, f64)> = imposed
            .iter()
            .map(|(node, g)| (node.borrow(py).as_node().id(), *g))
            .collect();
        let inner = self.inner.constraint_rhs(&pairs)?;
        Ok(PyNodeField { inner })
    }

    /// `Model.constraint_rhs_by_index(imposed)` — like `constraint_rhs` but each
    /// relation is keyed by its **index** (0-based, in `relations()` order)
    /// instead of a node.
    ///
    /// `imposed` is a list of `(relation_index, g)`. Use this when a node
    /// participates in several relations, where node keying would be ambiguous.
    /// Returns the same kind of `NodeField` over the multiplier nodes; union it
    /// with `|`. Raises if an index is out of range.
    fn constraint_rhs_by_index(&self, imposed: Vec<(usize, f64)>) -> PyResult<PyNodeField> {
        let inner = self.inner.constraint_rhs_by_index(&imposed)?;
        Ok(PyNodeField { inner })
    }
}

crate::impl_aggregate_pymethods!(
    PyModel,
    PySubModel,
    "Model",
    sub_model,
    Model,
    r#"
class PyModel:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubModel"]:
        """`model[i]` → the `SubModel` view of term i (a physics term, a
        Dirichlet condition, an MPC, ...)."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyModel"]:
        """`model[i:j:k]` → a fresh `Model` holding the sliced terms, shared
        with this one (no deep copy)."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyModel"] | pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`model | other` → a fresh `Model` holding the terms of both, in
        first-seen order and deduplicated by object identity. This is how a problem
        is composed: physics | boundary conditions | constraints."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`sub_model | model` — the mirror of `model | sub_model`, differing
        only in that the lone term comes first."""
    "#,
    r#"
class PySubModel:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubModel"]) -> pyo3_stub_gen.RustType["PyModel"]:
        """`sub_model | sub_model` → a fresh `Model` holding both terms."""
    "#
);
crate::impl_dump_pymethod!(handle PySubModel, handle);
